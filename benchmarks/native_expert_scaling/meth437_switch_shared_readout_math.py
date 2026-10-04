"""Shared two-factor output readout, native-valued with local approximate gradients."""
import hashlib
import numpy as np
import torch
import meth422_switch_function_autograd as G
import meth422_switch_function_gradient_contract as R
import meth424_switch_anchored_reference as L


def parameters(d, rank, seed, zero=True):
    rng = np.random.default_rng(seed); values = {'C': np.zeros((d, rank), np.float32), 'D': (rng.normal(size=(rank, d))/np.sqrt(d)).astype(np.float32)}
    if not zero: values['C'] = (rng.normal(size=(d, rank))*.02).astype(np.float32)
    return {k: torch.tensor(v, requires_grad=True) for k, v in values.items()}


def forward(feature, base, scores, chosen, buffers, fn, head, params):
    mean, std = (torch.from_numpy(v) for v in buffers); normalized = (feature-mean)/std
    low = G.NativeFloat.apply(normalized, params['D']); correction = G.NativeFloat.apply(low, params['C']); probability = G.NativeProbability.apply(scores, chosen)
    weighted = probability*correction; post = base+weighted; final = G.NativeRMS.apply(post, fn); hi = final*np.float32(1/np.sqrt(base.numel()))
    out = dict(normalized=normalized, low=low, correction=correction, probability=probability, weighted=weighted, post=post, final=final, head_input=hi)
    if head is not None: out['logits'] = G.NativeI8.apply(hi, head)
    return out


def qualify(label, feature, base, scores, chosen, buffers, fn, head, params, target, tiny, path, expected=None):
    pv = {k: v.detach().numpy().copy() for k, v in params.items()}; tf = torch.tensor(feature, requires_grad=True); tb = torch.tensor(base, requires_grad=True); ts = torch.tensor(scores, requires_grad=True)
    actual = forward(tf, tb, ts, chosen, buffers, fn, head, params); objective = G.prediction_loss(actual['logits'], torch.tensor(target)); objective.backward()
    ng = {k: v.grad.numpy().copy() for k, v in params.items()}; ng.update(feature=tf.grad.numpy().copy(), base=tb.grad.numpy().copy(), scores=ts.grad.numpy().copy())
    if tiny: assert all(np.linalg.norm(v) > 0 for v in ng.values())
    else:
        assert np.linalg.norm(ng['C']) > 1e-10 and np.linalg.norm(ng['base']) > 1e-10
        assert all(np.count_nonzero(ng[k]) == 0 for k in ('D', 'feature', 'scores'))
        R.C.exact(actual['post'].detach().numpy(), base); R.C.exact(actual['logits'].detach().numpy(), expected)
    dot = lambda w, x: (w.astype(np.float64)@x.astype(np.float64)).astype(np.float32)
    native = {}; native['normalized'] = (feature-buffers[0])/buffers[1]; native['low'] = dot(pv['D'], native['normalized']); native['correction'] = dot(pv['C'], native['low'])
    native['probability'] = np.asarray(np.float32(1/np.sum(np.exp((scores-scores[chosen]).astype(np.float64)).astype(np.float32).astype(np.float64))))
    native['weighted'] = native['probability']*native['correction']; native['post'] = base+native['weighted']; native['final'] = L.native_rms(native['post'], fn)
    scale = float(np.float32(1/np.sqrt(len(base)))); native['head_input'] = native['final']*np.float32(scale); native['logits'] = head.native(native['head_input'])
    for k, v in native.items(): R.C.exact(actual[k].detach().numpy(), v)
    q = {k: np.asarray(v, np.float64) for k, v in native.items()}; pp = {k: v.astype(np.float64) for k, v in pv.items()}; f, b, s = (v.astype(np.float64) for v in (feature, base, scores)); mu, std = [v.astype(np.float64) for v in buffers]; nf = fn.astype(np.float64); hw = head.smooth_weight().numpy()
    parent = {'normalized': (f-mu)/std, 'low': pp['D']@q['normalized'], 'correction': pp['C']@q['low'], 'probability': L.selected_probability(s, chosen),
        'weighted': q['probability']*q['correction'], 'post': b+q['weighted'], 'final': L.rms(q['post'], nf), 'head_input': q['final']*scale, 'logits': hw@q['head_input']}
    offsets = {k: q[k]-parent[k] for k in q}; fixed = {k: v.tobytes() for k, v in offsets.items()}
    def numpy_path(ff, bb, ss, p, off=offsets):
        z = {}; z['normalized'] = (ff-mu)/std+off['normalized']; z['low'] = p['D']@z['normalized']+off['low']; z['correction'] = p['C']@z['low']+off['correction']
        z['probability'] = L.selected_probability(ss, chosen)+off['probability']; z['weighted'] = z['probability']*z['correction']+off['weighted']; z['post'] = bb+z['weighted']+off['post']
        z['final'] = L.rms(z['post'], nf)+off['final']; z['head_input'] = z['final']*scale+off['head_input']; z['logits'] = hw@z['head_input']+off['logits']; return z
    ff = torch.tensor(f, requires_grad=True); bb = torch.tensor(b, requires_grad=True); ss = torch.tensor(s, requires_grad=True); tp = {k: torch.tensor(v, requires_grad=True) for k, v in pp.items()}; off = {k: torch.tensor(v) for k, v in offsets.items()}
    z = {}; z['normalized'] = (ff-torch.tensor(mu))/torch.tensor(std)+off['normalized']; z['low'] = tp['D']@z['normalized']+off['low']; z['correction'] = tp['C']@z['low']+off['correction']
    z['probability'] = torch.softmax(ss, dim=0)[chosen]+off['probability']; z['weighted'] = z['probability']*z['correction']+off['weighted']; z['post'] = bb+z['weighted']+off['post']
    z['final'] = z['post']*torch.rsqrt(torch.mean(z['post']*z['post'])+1e-6)*torch.tensor(nf)+off['final']; z['final'].retain_grad()
    z['head_input'] = z['final']*scale+off['head_input']; z['logits'] = head.smooth_weight()@z['head_input']+off['logits']; npout = numpy_path(f, b, s, pp); endpoints = {}
    for k in q:
        relative = max(R.relative(npout[k], q[k]), R.relative(z[k].detach().numpy(), q[k])); absolute = max(float(np.max(np.abs(npout[k]-q[k]))), float(np.max(np.abs(z[k].detach().numpy()-q[k]))))
        assert relative <= 1e-12 and absolute <= 1e-9, ('endpoint', label, k, relative, absolute); endpoints[k] = dict(relative=relative, absolute=absolute)
    G.prediction_loss(z['logits'], torch.tensor(target)).backward(); rg = {k: v.grad.numpy().copy() for k, v in tp.items()}; rg.update(feature=ff.grad.numpy().copy(), base=bb.grad.numpy().copy(), scores=ss.grad.numpy().copy())
    ge = {k: R.relative(ng[k], rg[k]) for k in rg}; assert max(ge.values()) <= 1e-3, ('gradient', label, ge)
    arrays = {}; checks = []; rng = np.random.default_rng(437+len(scores)); anchor_logits = npout['logits']; probability = R.probability(anchor_logits); maximum = int(np.argmax(anchor_logits))
    for k in ('C', 'D', 'feature', 'base', 'scores'):
        value = {'feature': f, 'base': b, 'scores': s}.get(k, pp.get(k))
        def objective_at(v):
            delta = numpy_path(v if k == 'feature' else f, v if k == 'base' else b, v if k == 'scores' else s, pp | {k: v} if k in pp else pp)['logits']-anchor_logits
            delta = delta-delta[maximum]; return np.log1p(np.sum(probability*np.expm1(delta)))-np.dot(target, delta)
        if tiny:
            fd = R.finite_difference(objective_at, value, 1e-4); er = R.relative(rg[k], fd); en = R.relative(ng[k], fd)
            assert er <= 1e-5 and en <= 1e-3, ('tiny_FD', label, k, er, en); arrays['FD_'+k] = fd; checks.append(dict(field=k, coordinates=value.size, reference_FD_error=er, native_FD_error=en))
        else:
            direction = rng.choice([-1., 1.], size=value.shape)/np.sqrt(value.size); fd = float((objective_at(value+1e-4*direction)-objective_at(value-1e-4*direction))/(2e-4)); pr = float(np.sum(rg[k]*direction)); pn = float(np.sum(ng[k]*direction)); er = abs(fd-pr)/max(abs(pr), 1e-8); en = abs(fd-pn)/max(abs(pn), 1e-8)
            assert er <= 1e-5 and en <= 1e-3, ('real_FD', label, k, er, en); arrays['direction_'+k] = direction; checks.append(dict(field=k, FD=fd, reference_projection=pr, native_projection=pn, reference_FD_error=er, native_FD_error=en))
    assert all(v.tobytes() == fixed[k] for k, v in offsets.items())
    dropped = offsets | {'logits': np.zeros_like(offsets['logits'])}; wrong = numpy_path(f, b, s, pp, dropped)['logits']; assert R.relative(wrong, q['logits']) > 1e-12 or np.max(np.abs(wrong-q['logits'])) > 1e-9
    negatives = {'dropped_head_offset_detected': True}
    if tiny:
        wrong = z['final'].grad.numpy()*nf/np.sqrt(np.mean(q['post']**2)+1e-6); rms_error = R.relative(wrong, rg['base']); assert rms_error > 1e-3
        assert R.relative(np.zeros_like(rg['scores']), rg['scores']) > 1e-3; negatives.update(detached_probability_detected=True, omitted_RMS_coupling_error=rms_error)
    for k in q: arrays['native_'+k] = native[k]; arrays['continuation_'+k] = npout[k]; arrays['offset_'+k] = offsets[k]
    for k in rg: arrays['native_gradient_'+k] = ng[k]; arrays['reference_gradient_'+k] = rg[k]
    np.savez(path, **arrays, **{'parameter_'+k: v for k, v in pv.items()}, feature=feature, base=base, scores=scores, target=target, mean=buffers[0], std=buffers[1], finalnorm=fn)
    return dict(label=label, endpoints=endpoints, gradient_errors=ge, FD=checks, negatives=negatives, offsets_fixed=True, zero_post_full_head_exact=not tiny, archive_path=str(path), approximate_derivative_not_rounding=True)
