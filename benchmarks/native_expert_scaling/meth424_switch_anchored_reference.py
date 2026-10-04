"""Detached local continuation of native primals, not a rounding derivative.

Offsets are constants for one anchor. Refreshing them during a finite-difference
evaluation would test a different, nondifferentiable function and is forbidden.
NumPy supplies independent finite differences; ordinary F64 torch supplies the
reference derivative. Native saved-primal rules are supplied by immutable422.
"""
import numpy as np
import torch


FIELDS = ('input', 'Bx', 'Ax', 'corrected_input', 'up_raw', 'up', 'down',
          'Ddown', 'Cdown', 'corrected_output', 'probability', 'weighted_down',
          'post', 'final', 'head_input', 'logits')


def rms(x, weight):
    return x/np.sqrt(np.mean(x*x)+1e-6)*weight


def native_rms(x, weight):
    mean = np.float32(np.sum((x*x).astype(np.float64))/len(x))
    scale = np.float32(1)/np.sqrt(np.float32(mean+np.float32(1e-6)))
    return (x*scale)*weight


def selected_probability(scores, chosen):
    p = np.exp(scores-np.max(scores)); return p[chosen]/np.sum(p)


def native_anchor(pre, scores, chosen, ff, fn, wi, wo, head, parameters):
    """Literal native values, including correction intermediates absent in418."""
    assert pre.dtype == scores.dtype == np.float32
    assert int(np.argmax(scores)) == chosen
    assert all(z.dtype == np.float32 and np.isfinite(z).all() for z in parameters.values())
    out = {}; out['input'] = native_rms(pre, ff)
    dot = lambda w, x: (w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    out['Bx'] = dot(parameters['B'], out['input']); out['Ax'] = dot(parameters['A'], out['Bx'])
    out['corrected_input'] = out['input']+out['Ax']
    out['up_raw'] = wi.native(out['corrected_input'])
    out['up'] = np.where(out['up_raw'] < 0, np.float32(0), out['up_raw'])
    out['down'] = wo.native(out['up'])
    out['Ddown'] = dot(parameters['D'], out['down']); out['Cdown'] = dot(parameters['C'], out['Ddown'])
    out['corrected_output'] = out['down']+out['Cdown']
    exp = np.exp((scores-scores[chosen]).astype(np.float64)).astype(np.float32).astype(np.float64)
    out['probability'] = np.float32(1/np.sum(exp))
    out['weighted_down'] = out['probability']*out['corrected_output']
    out['post'] = pre+out['weighted_down']; out['final'] = native_rms(out['post'], fn)
    out['head_input'] = out['final']*np.float32(1/np.sqrt(len(pre)))
    out['logits'] = head.native(out['head_input'])
    assert tuple(out) == FIELDS and all(np.isfinite(z).all() for z in out.values())
    return out


def detached_offsets(native, pre, scores, chosen, ff, fn, wi, wo, head, parameters):
    """Build each offset from the native values of its parents, independently."""
    q = {k: np.asarray(v, np.float64) for k, v in native.items()}
    parameters = {k: np.asarray(v, np.float64) for k, v in parameters.items()}
    pre = np.asarray(pre, np.float64); scores = np.asarray(scores, np.float64)
    d = len(pre); scale = float(np.float32(1/np.sqrt(d)))
    smooth_at_native = {
        'input': rms(pre, ff),
        'Bx': parameters['B'] @ q['input'], 'Ax': parameters['A'] @ q['Bx'],
        'corrected_input': q['input']+q['Ax'], 'up_raw': wi @ q['corrected_input'],
        'up': np.maximum(q['up_raw'], 0), 'down': wo @ q['up'],
        'Ddown': parameters['D'] @ q['down'], 'Cdown': parameters['C'] @ q['Ddown'],
        'corrected_output': q['down']+q['Cdown'],
        'probability': selected_probability(scores, chosen),
        'weighted_down': q['probability']*q['corrected_output'],
        'post': pre+q['weighted_down'], 'final': rms(q['post'], fn),
        'head_input': q['final']*scale, 'logits': head @ q['head_input']}
    return {k: q[k]-smooth_at_native[k] for k in FIELDS}


def numpy_path(pre, scores, chosen, ff, fn, wi, wo, head, parameters, offsets):
    """Independent literal NumPy continuation. Offsets stay fixed on perturbation."""
    assert set(offsets) == set(FIELDS)
    out = {}
    def put(name, value):
        offset = np.asarray(offsets[name], np.float64); value = np.asarray(value, np.float64)
        assert value.shape == offset.shape and np.isfinite(offset).all()
        out[name] = value+offset; return out[name]
    x = put('input', rms(pre, ff))
    bx = put('Bx', parameters['B'].dot(x)); ax = put('Ax', parameters['A'].dot(bx))
    corrected = put('corrected_input', x+ax)
    raw = put('up_raw', wi.dot(corrected)); up = put('up', np.maximum(raw, 0))
    down = put('down', wo.dot(up))
    dy = put('Ddown', parameters['D'].dot(down)); cy = put('Cdown', parameters['C'].dot(dy))
    corrected = put('corrected_output', down+cy)
    p = put('probability', selected_probability(scores, chosen))
    weighted = put('weighted_down', p*corrected); post = put('post', pre+weighted)
    final = put('final', rms(post, fn))
    head_input = put('head_input', final*float(np.float32(1/np.sqrt(len(pre)))))
    put('logits', head.dot(head_input))
    assert all(np.isfinite(z).all() for z in out.values())
    return out


def torch_path(pre, scores, chosen, ff, fn, wi, wo, head, parameters, offsets):
    """Ordinary F64 autograd; no custom backward or differentiated offset."""
    out = {}
    def put(name, value):
        off = torch.tensor(np.asarray(offsets[name], np.float64))
        assert not off.requires_grad and value.dtype == torch.float64 and value.shape == off.shape
        out[name] = value+off; return out[name]
    x = put('input', pre*torch.rsqrt(torch.mean(pre*pre)+1e-6)*ff)
    bx = put('Bx', parameters['B'] @ x); ax = put('Ax', parameters['A'] @ bx)
    corrected = put('corrected_input', x+ax)
    raw = put('up_raw', wi @ corrected); up = put('up', torch.relu(raw))
    down = put('down', wo @ up)
    dy = put('Ddown', parameters['D'] @ down); cy = put('Cdown', parameters['C'] @ dy)
    corrected = put('corrected_output', down+cy)
    p = put('probability', torch.softmax(scores, dim=0)[chosen])
    weighted = put('weighted_down', p*corrected); post = put('post', pre+weighted)
    final = put('final', post*torch.rsqrt(torch.mean(post*post)+1e-6)*fn)
    head_input = put('head_input', final*float(np.float32(1/np.sqrt(pre.numel()))))
    put('logits', head @ head_input)
    return out
