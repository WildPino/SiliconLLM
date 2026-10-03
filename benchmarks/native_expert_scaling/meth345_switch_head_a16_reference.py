"""Independent I64 head A16 projection; existing qualified A8 upstream reference."""
from contextlib import contextmanager
from types import MethodType
import numpy as np
import torch
import meth336_switch_integer_reference as R


def projection(values, weights, row_scales):
    assert values.dtype == np.float32 and np.isfinite(values).all()
    assert values.shape[-1] <= 4096 and weights.dtype == np.int8
    assert np.all(weights != -128)
    maximum = np.abs(values).max(-1, keepdims=True)
    scales = (maximum / np.float32(32767)).astype(np.float32)
    scales[maximum == 0] = np.float32(1)
    assert np.isfinite(scales).all() and np.all(scales > 0)
    codes = np.clip(np.rint(values / scales), -32767, 32767).astype(np.int16)
    dots = codes.astype(np.int64) @ weights.astype(np.int64).T
    assert np.abs(dots).max() <= values.shape[-1] * 127 * 32767
    result = ((dots.astype(np.float64) * row_scales.astype(np.float64)[None, :])
              * scales.astype(np.float64)).astype(np.float32)
    return result, dots, codes, scales


def head_from_states(states, entries, payload):
    entry = entries['lm_head.weight']
    mapped = np.memmap(payload, dtype=np.uint8, mode='r')
    weights = np.ndarray(tuple(entry['shape']), dtype=np.int8,
                         buffer=mapped, offset=entry['offset'])
    scales = np.ndarray((entry['shape'][0],), dtype='<f4',
                        buffer=mapped, offset=entry['scale_offset'])
    d = weights.shape[1]
    factor = np.float32(1. / np.sqrt(np.float64(d)))
    return projection((states * factor).astype(np.float32), weights, scales)[0]


@contextmanager
def compact_reference(model, entries, payload):
    mapped = np.memmap(payload, dtype=np.uint8, mode='r')
    entry = entries['lm_head.weight']
    weights = np.ndarray(tuple(entry['shape']), dtype=np.int8,
                         buffer=mapped, offset=entry['offset'])
    scales = np.ndarray((entry['shape'][0],), dtype='<f4',
                        buffer=mapped, offset=entry['scale_offset'])
    with R.compact_reference(model, entries, payload) as mode:
        old_forward = model.lm_head.forward
        def forward(module, hidden):
            assert hidden.dtype == torch.float32 and module.bias is None
            shape = tuple(hidden.shape[:-1])
            values = hidden.detach().numpy().reshape(-1, weights.shape[1])
            result = projection(values, weights, scales)[0]
            mode.calls['head_a16_integer_projection'] += 1
            return torch.from_numpy(result.reshape(*shape, weights.shape[0]))
        model.lm_head.forward = MethodType(forward, model.lm_head)
        try:
            yield mode
        finally:
            model.lm_head.forward = old_forward
