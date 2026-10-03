"""Independent vectorized I64/scaling reference on serialized target weights."""
from contextlib import contextmanager
from types import MethodType
import numpy as np
import torch
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersLayerNorm
import meth334_switch_arithmetic_reference as R


def activation_quantize(values):
    assert values.dtype == np.float32 and np.isfinite(values).all()
    maximum = np.abs(values).max(-1, keepdims=True)
    scales = np.divide(maximum, np.float32(127.)).astype(np.float32)
    scales[maximum == 0.] = np.float32(1.)
    assert np.isfinite(scales).all() and (scales > 0.).all()
    codes = np.clip(np.rint(np.divide(values, scales)), -127., 127.).astype(np.int8)
    return codes, scales


def integer_projection(values, codes, scales):
    quantized, activation_scales = activation_quantize(values)
    dots = quantized.astype(np.int64) @ codes.astype(np.int64).T
    assert np.max(np.abs(dots)) <= 66064384
    result = ((dots.astype(np.float64) * scales.astype(np.float64)[None, :])
              * activation_scales.astype(np.float64)).astype(np.float32)
    return result, dots, quantized, activation_scales


class RoundedExpArithmetic(R.SpecifiedArithmetic):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._softmax.default:
            x, dimension, half_to_float = args
            assert x.dtype == torch.float32 and not half_to_float
            self.calls['softmax'] += 1
            shifted = x - x.max(dimension, keepdim=True).values
            exponentials = torch.exp(shifted.to(torch.float64)).to(torch.float32)
            values = exponentials.to(torch.float64)
            return (values / values.sum(dimension, keepdim=True)).to(torch.float32)
        return super().__torch_dispatch__(func, types, args, kwargs)


def target_control_model(model, entries, payload):
    """Official META architecture with actual target F32 controls only.

    Quantized Linear parameters remain shape descriptors on META; every such
    forward is replaced by serialized integer projection in compact_reference.
    """
    namespace = model.state_dict()
    assert set(namespace) == set(entries) and len(namespace) == 6392
    for name, descriptor in namespace.items():
        assert list(descriptor.shape) == entries[name]['shape']
        assert descriptor.dtype == torch.float32 and descriptor.device.type == 'meta'
    mapped = np.memmap(payload, dtype=np.uint8, mode='r')
    controls = {}; cache = {}
    for name, entry in entries.items():
        if entry['encoding'] == 0:
            key = (entry['offset'], entry['elements'])
            if key not in cache:
                view = np.ndarray(tuple(entry['shape']), dtype='<f4', buffer=mapped, offset=entry['offset'])
                cache[key] = torch.from_numpy(view.copy())
            controls[name] = cache[key]
    incompatible = model.load_state_dict(controls, strict=False, assign=True)
    quantized = {name for name, entry in entries.items() if entry['encoding'] == 1}
    assert not incompatible.unexpected_keys and set(incompatible.missing_keys) == quantized
    model.tie_weights(); model.requires_grad_(False); model.eval()
    meta = {name for name, value in model.state_dict().items() if value.device.type == 'meta'}
    assert meta == quantized - {'lm_head.weight'}
    assert all(value.dtype == torch.float32 for value in model.parameters())
    assert all(value.device.type == 'cpu' for name, value in model.state_dict().items() if name not in meta)
    linear_names = {name + '.weight' for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    assert quantized <= linear_names
    return {'namespace_shapes_exact': len(namespace), 'actual_f32_control_names': len(controls),
            'quantized_linear_names': len(quantized), 'remaining_meta_descriptors': len(meta),
            'f32_unique_loaded_bytes': sum(value.numel()*4 for value in cache.values()),
            'official_architecture_target_only_reference': True}


@contextmanager
def compact_reference(model, entries, payload):
    # One mapped payload, inexpensive per-matrix array views. No requantization
    # of donor weights and no full dequantized target model allocated.
    mapped = np.memmap(payload, dtype=np.uint8, mode='r')
    saved = []; mode = RoundedExpArithmetic()
    def projection(codes, scales):
        def forward(module, hidden):
            assert module.bias is None and hidden.dtype == torch.float32
            shape = tuple(hidden.shape[:-1])
            values = hidden.detach().numpy().reshape(-1, codes.shape[1])
            result, _, _, _ = integer_projection(values, codes, scales)
            mode.calls['integer_projection'] += 1
            return torch.from_numpy(result.reshape(*shape, codes.shape[0]))
        return forward
    for name, module in model.named_modules():
        if type(module) is SwitchTransformersLayerNorm:
            saved.append((module, module.forward)); module.forward = MethodType(R.specified_norm, module)
        elif isinstance(module, torch.nn.Linear):
            entry = entries[name + '.weight']
            if entry['encoding']:
                assert list(module.weight.shape) == entry['shape']
                codes = np.ndarray(tuple(entry['shape']), dtype=np.int8, buffer=mapped, offset=entry['offset'])
                scales = np.ndarray((entry['shape'][0],), dtype='<f4', buffer=mapped, offset=entry['scale_offset'])
                saved.append((module, module.forward)); module.forward = MethodType(projection(codes, scales), module)
    assert saved
    try:
        with mode:
            yield mode
    finally:
        for module, forward in saved:
            module.forward = forward
