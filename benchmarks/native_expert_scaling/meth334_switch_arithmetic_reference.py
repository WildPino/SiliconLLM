"""Independent vectorized Torch arithmetic contract on official architecture.

This reference changes the numerical estimand explicitly. It never replaces
the original unmodified donor as a primary heldout quality comparator.
"""
from collections import Counter
from contextlib import contextmanager
from types import MethodType

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersLayerNorm


def specified_norm(module, hidden):
    assert hidden.dtype == torch.float32 and hidden.device.type == 'cpu'
    squares = hidden * hidden
    mean = squares.to(torch.float64).mean(-1, keepdim=True).to(torch.float32)
    root = torch.sqrt((mean + module.variance_epsilon).to(torch.float64)).to(torch.float32)
    scale = (1. / root.to(torch.float64)).to(torch.float32)
    return (hidden * scale) * module.weight


class SpecifiedArithmetic(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            a, b = args
            assert a.dtype == b.dtype == torch.float32
            assert a.device.type == b.device.type == 'cpu'
            self.calls['mm' if func is torch.ops.aten.mm.default else 'bmm'] += 1
            return func(a.to(torch.float64), b.to(torch.float64), **kwargs).to(torch.float32)
        if func is torch.ops.aten.addmm.default:
            bias, a, b = args
            assert bias.dtype == a.dtype == b.dtype == torch.float32
            # Original bound source has no projection/router biases. Handle an
            # explicit F32 bias only after the rounded matrix output if present.
            assert kwargs.get('alpha', 1) == 1 and kwargs.get('beta', 1) == 1
            self.calls['addmm'] += 1
            projected = torch.mm(a.to(torch.float64), b.to(torch.float64)).to(torch.float32)
            return bias + projected
        if func is torch.ops.aten._softmax.default:
            x, dimension, half_to_float = args
            assert x.dtype == torch.float32 and not half_to_float
            self.calls['softmax'] += 1
            shifted = x - x.max(dimension, keepdim=True).values
            exponentials = torch.exp(shifted)
            values = exponentials.to(torch.float64)
            return (values / values.sum(dimension, keepdim=True)).to(torch.float32)
        return func(*args, **kwargs)


@contextmanager
def specified_arithmetic(model):
    saved = []
    for module in model.modules():
        if type(module) is SwitchTransformersLayerNorm:
            saved.append((module, module.forward))
            module.forward = MethodType(specified_norm, module)
    assert saved
    mode = SpecifiedArithmetic()
    try:
        with mode:
            yield mode
    finally:
        for module, forward in saved:
            module.forward = forward


def primitive_controls():
    """NumPy vectorized F64 oracles, not copies of C reductions or architecture."""
    rng = np.random.default_rng(333)
    a = rng.normal(size=(5, 768)).astype(np.float32)
    b = rng.normal(size=(768, 7)).astype(np.float32)
    ba = rng.normal(size=(3, 5, 64)).astype(np.float32)
    bb = rng.normal(size=(3, 64, 9)).astype(np.float32)
    logits = rng.normal(size=(5, 256)).astype(np.float32)
    # Include large dynamic range and cancellation; operands remain original F32.
    a[0, :3] = [16777216., 1., -16777216.]
    b[:3, 0] = 1.
    with SpecifiedArithmetic() as mode:
        observed = torch.mm(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        observed_batched = torch.bmm(torch.from_numpy(ba), torch.from_numpy(bb)).numpy()
        observed_prob = torch.softmax(torch.from_numpy(logits), -1).numpy()
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    expected_batched = (ba.astype(np.float64) @ bb.astype(np.float64)).astype(np.float32)
    shifted = logits - logits.max(-1, keepdims=True)
    exponentials = np.exp(shifted).astype(np.float32).astype(np.float64)
    expected_prob = (exponentials / exponentials.sum(-1, keepdims=True)).astype(np.float32)
    weights = rng.uniform(.5, 1.5, size=(768,)).astype(np.float32)
    module = SwitchTransformersLayerNorm(768, eps=1e-6)
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(weights))
    actual_norm = specified_norm(module, torch.from_numpy(a)).detach().numpy()
    mean = (a * a).astype(np.float64).mean(-1, keepdims=True).astype(np.float32)
    scale = (np.float32(1.) / np.sqrt(mean + np.float32(1e-6))).astype(np.float32)
    expected_norm = (a * scale) * weights
    argument=torch.from_numpy(mean)+np.float32(1e-6)
    standard_root=torch.sqrt(argument)
    rounded_root=torch.sqrt(argument.to(torch.float64)).to(torch.float32)
    standard_reciprocal=1./rounded_root
    rounded_reciprocal=(1./rounded_root.to(torch.float64)).to(torch.float32)
    diagnostic={'standard_vs_rounded_sqrt_maxabs':float((standard_root-rounded_root).abs().max()),
                'standard_vs_rounded_reciprocal_maxabs':float((standard_reciprocal-rounded_reciprocal).abs().max())}
    errors = {'matrix_maxabs': float(np.max(np.abs(observed - expected))),
              'attention_matrix_maxabs': float(np.max(np.abs(observed_batched - expected_batched))),
              'softmax_maxabs': float(np.max(np.abs(observed_prob - expected_prob))),
              'norm_maxabs': float(np.max(np.abs(actual_norm - expected_norm)))}
    return {'seed': 333, 'norm_primitive_diagnostic':diagnostic, 'errors': errors, 'calls': dict(mode.calls),
            'passed': errors['matrix_maxabs'] == errors['attention_matrix_maxabs'] == 0.
            and errors['softmax_maxabs'] <= 2e-7 and errors['norm_maxabs'] <= 2e-7,
            'oracle': 'independent NumPy vectorized F64 products/reductions with declared F32 roundings'}
