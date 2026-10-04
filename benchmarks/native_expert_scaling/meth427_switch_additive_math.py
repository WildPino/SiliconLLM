"""Original-post-preserving native-valued functions and independent local FD."""
import numpy as np
import torch
import meth422_switch_function_autograd as G
import meth422_switch_function_gradient_contract as R
import meth424_switch_anchored_reference as L
from meth426_switch_pilot_math import affine, qualify_affine


def forward(arm, pre, base, scores, chosen, ff, fn, wi, wo, head, parameters):
    x = G.NativeRMS.apply(pre, ff); bx = G.NativeFloat.apply(x, parameters['B'])
    ax = G.NativeFloat.apply(bx, parameters['A']); out = dict(input=x, Bx=bx, Ax=ax)
    if arm == 'real':
        corrected = x+ax; raw = G.NativeI8.apply(corrected, wi); up = G.NativeReLU.apply(raw); feature = G.NativeI8.apply(up, wo)
        dy = G.NativeFloat.apply(feature, parameters['D']); cy = G.NativeFloat.apply(dy, parameters['C']); correction = cy
        out.update(corrected_input=corrected, up_raw=raw, up=up, feature=feature, Dy=dy, Cy=cy)
    else:
        assert arm == 'adapter'; dy = G.NativeFloat.apply(x, parameters['D']); cy = G.NativeFloat.apply(dy, parameters['C']); correction = ax+cy
        out.update(Dy=dy, Cy=cy)
    p = G.NativeProbability.apply(scores, chosen); weighted = p*correction; post = base+weighted
    final = G.NativeRMS.apply(post, fn); hi = final*np.float32(1/np.sqrt(pre.numel())); logits = G.NativeI8.apply(hi, head)
    out.update(correction=correction, probability=p, weighted=weighted, post=post, final=final, head_input=hi, logits=logits)
    return out


def qualify(arm, label, pre, base, scores, chosen, ff, fn, wi, wo, head, parameters, target, tiny, path, expected_logits=None):
    values = {k: z.detach().numpy().copy() for k, z in parameters.items()}
    tx = torch.tensor(pre.copy(), requires_grad=True); tb = torch.tensor(base.copy(), requires_grad=True); ts = torch.tensor(scores.copy(), requires_grad=True)
    actual = forward(arm, tx, tb, ts, chosen, ff, fn, wi, wo, head, parameters)
    loss = G.prediction_loss(actual['logits'], torch.tensor(target)); loss.backward()
    native_grad = {k: z.grad.numpy().copy() for k, z in parameters.items()}
    native_grad.update(pre=tx.grad.numpy().copy(), base=tb.grad.numpy().copy(), scores=ts.grad.numpy().copy())
    if tiny: assert all(np.linalg.norm(native_grad[k]) > 0 for k in ('A', 'B', 'C', 'D'))
    else:
        live = ('C',) if arm == 'real' else ('A', 'C')
        zero = ('A', 'B', 'D') if arm == 'real' else ('B', 'D')
        assert all(np.linalg.norm(native_grad[k]) > 1e-10 for k in live)
        assert all(np.count_nonzero(native_grad[k]) == 0 for k in zero)
        R.C.exact(actual['post'].detach().numpy(), base)
        if expected_logits is not None: R.C.exact(actual['logits'].detach().numpy(), expected_logits)
    # Independent literal NumPy native evaluation, including every correction node.
    anchor = {}; anchor['input'] = L.native_rms(pre, ff)
    dot = lambda w, x: (w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    anchor['Bx'] = dot(values['B'], anchor['input']); anchor['Ax'] = dot(values['A'], anchor['Bx'])
    if arm == 'real':
        anchor['corrected_input'] = anchor['input']+anchor['Ax']; anchor['up_raw'] = wi.native(anchor['corrected_input'])
        anchor['up'] = np.where(anchor['up_raw'] < 0, np.float32(0), anchor['up_raw']); anchor['feature'] = wo.native(anchor['up'])
        anchor['Dy'] = dot(values['D'], anchor['feature']); anchor['Cy'] = dot(values['C'], anchor['Dy']); anchor['correction'] = anchor['Cy'].copy()
    else:
        anchor['Dy'] = dot(values['D'], anchor['input']); anchor['Cy'] = dot(values['C'], anchor['Dy']); anchor['correction'] = anchor['Ax']+anchor['Cy']
    exp = np.exp((scores-scores[chosen]).astype(np.float64)).astype(np.float32).astype(np.float64)
    anchor['probability'] = np.float32(1/np.sum(exp)); anchor['weighted'] = anchor['probability']*anchor['correction']
    anchor['post'] = base+anchor['weighted']; anchor['final'] = L.native_rms(anchor['post'], fn)
    scale = float(np.float32(1/np.sqrt(len(pre))))
    anchor['head_input'] = anchor['final']*np.float32(scale); anchor['logits'] = head.native(anchor['head_input'])
    for k, z in actual.items(): R.C.exact(z.detach().numpy(), anchor[k])
    q = {k: np.asarray(z, np.float64) for k, z in anchor.items()}; pv = {k: z.astype(np.float64) for k, z in values.items()}
    x = pre.astype(np.float64); b = base.astype(np.float64); s = scores.astype(np.float64); f = ff.astype(np.float64); n = fn.astype(np.float64)
    wh = head.smooth_weight().numpy(); ww = wi.smooth_weight().numpy() if arm == 'real' else None; ow = wo.smooth_weight().numpy() if arm == 'real' else None
    at_native = {'input': L.rms(x, f), 'Bx': pv['B']@q['input'], 'Ax': pv['A']@q['Bx']}
    if arm == 'real':
        at_native.update(corrected_input=q['input']+q['Ax'], up_raw=ww@q['corrected_input'], up=np.maximum(q['up_raw'], 0), feature=ow@q['up'],
                         Dy=pv['D']@q['feature'], Cy=pv['C']@q['Dy'], correction=q['Cy'])
    else: at_native.update(Dy=pv['D']@q['input'], Cy=pv['C']@q['Dy'], correction=q['Ax']+q['Cy'])
    at_native.update(probability=L.selected_probability(s, chosen), weighted=q['probability']*q['correction'], post=b+q['weighted'],
                     final=L.rms(q['post'], n), head_input=q['final']*scale, logits=wh@q['head_input'])
    offsets = {k: q[k]-at_native[k] for k in q}; fingerprints = {k: v.tobytes() for k, v in offsets.items()}
    def numpy_path(xx, bb, ss, pp, off=offsets):
        z = {}; z['input'] = L.rms(xx, f)+off['input']; z['Bx'] = pp['B'].dot(z['input'])+off['Bx']; z['Ax'] = pp['A'].dot(z['Bx'])+off['Ax']
        if arm == 'real':
            z['corrected_input'] = z['input']+z['Ax']+off['corrected_input']; z['up_raw'] = ww.dot(z['corrected_input'])+off['up_raw']
            z['up'] = np.maximum(z['up_raw'], 0)+off['up']; z['feature'] = ow.dot(z['up'])+off['feature']
            z['Dy'] = pp['D'].dot(z['feature'])+off['Dy']; z['Cy'] = pp['C'].dot(z['Dy'])+off['Cy']; z['correction'] = z['Cy']+off['correction']
        else:
            z['Dy'] = pp['D'].dot(z['input'])+off['Dy']; z['Cy'] = pp['C'].dot(z['Dy'])+off['Cy']; z['correction'] = z['Ax']+z['Cy']+off['correction']
        z['probability'] = L.selected_probability(ss, chosen)+off['probability']; z['weighted'] = z['probability']*z['correction']+off['weighted']
        z['post'] = bb+z['weighted']+off['post']; z['final'] = L.rms(z['post'], n)+off['final']; z['head_input'] = z['final']*scale+off['head_input']
        z['logits'] = wh.dot(z['head_input'])+off['logits']; return z
    fx = torch.tensor(x, requires_grad=True); fb = torch.tensor(b, requires_grad=True); fs = torch.tensor(s, requires_grad=True)
    fp = {k: torch.tensor(v, requires_grad=True) for k, v in pv.items()}; off = {k: torch.tensor(v) for k, v in offsets.items()}
    z = {}; z['input'] = fx*torch.rsqrt(torch.mean(fx*fx)+1e-6)*torch.tensor(f)+off['input']
    z['Bx'] = fp['B']@z['input']+off['Bx']; z['Ax'] = fp['A']@z['Bx']+off['Ax']
    if arm == 'real':
        z['corrected_input'] = z['input']+z['Ax']+off['corrected_input']; z['up_raw'] = wi.smooth_weight()@z['corrected_input']+off['up_raw']
        z['up'] = torch.relu(z['up_raw'])+off['up']; z['feature'] = wo.smooth_weight()@z['up']+off['feature']
        z['Dy'] = fp['D']@z['feature']+off['Dy']; z['Cy'] = fp['C']@z['Dy']+off['Cy']; z['correction'] = z['Cy']+off['correction']
    else:
        z['Dy'] = fp['D']@z['input']+off['Dy']; z['Cy'] = fp['C']@z['Dy']+off['Cy']; z['correction'] = z['Ax']+z['Cy']+off['correction']
    z['probability'] = torch.softmax(fs, dim=0)[chosen]+off['probability']; z['weighted'] = z['probability']*z['correction']+off['weighted']
    z['post'] = fb+z['weighted']+off['post']; z['final'] = z['post']*torch.rsqrt(torch.mean(z['post']*z['post'])+1e-6)*torch.tensor(n)+off['final']
    z['head_input'] = z['final']*scale+off['head_input']; z['logits'] = head.smooth_weight()@z['head_input']+off['logits']
    npout = numpy_path(x, b, s, pv); endpoints = {}
    for k in q:
        errors = (R.relative(npout[k], q[k]), R.relative(z[k].detach().numpy(), q[k]))
        absolute = max(float(np.max(np.abs(npout[k]-q[k]))), float(np.max(np.abs(z[k].detach().numpy()-q[k]))))
        assert max(errors) <= 1e-12 and absolute <= 1e-9, ('additive_endpoint', label, k, errors, absolute)
        endpoints[k] = {'relative_max': max(errors), 'absolute_max': absolute}
    G.prediction_loss(z['logits'], torch.tensor(target)).backward()
    ref = {k: v.grad.numpy().copy() for k, v in fp.items()}; ref.update(pre=fx.grad.numpy().copy(), base=fb.grad.numpy().copy(), scores=fs.grad.numpy().copy())
    comparisons = {k: R.relative(native_grad[k], ref[k]) for k in ref}; assert max(comparisons.values()) <= 1e-3, ('additive_gradient', label, comparisons)
    base_logits = npout['logits'].copy(); prob0 = R.probability(base_logits); kmax = int(np.argmax(base_logits))
    arrays = {}; checks = []; rng = np.random.default_rng(419+len(scores)); crossings = 0
    for k in ('A', 'B', 'C', 'D', 'pre', 'scores', 'base'):
        value = x if k == 'pre' else b if k == 'base' else s if k == 'scores' else pv[k]
        def objective(v):
            nonlocal crossings
            output = numpy_path(v if k == 'pre' else x, v if k == 'base' else b, v if k == 'scores' else s, pv | {k: v} if k in pv else pv)
            if arm == 'real': crossings += int(np.count_nonzero((output['up_raw'] > 0) != (anchor['up_raw'] > 0)))
            delta = output['logits']-base_logits; delta -= delta[kmax]
            return np.log1p(np.sum(prob0*np.expm1(delta)))-np.dot(target, delta)
        if tiny:
            fd = R.finite_difference(objective, value, 1e-4); error = R.relative(ref[k], fd); native_error = R.relative(native_grad[k], fd)
            assert error <= 1e-5 and native_error <= 1e-3, ('additive_tiny_FD', label, k, error, native_error)
            arrays['FD_'+k] = fd; checks.append({'field': k, 'coordinates': value.size, 'reference_FD_error': error, 'native_FD_error': native_error})
        else:
            direction = rng.choice([-1., 1.], size=value.shape)/np.sqrt(value.size)
            fd = float((objective(value+1e-4*direction)-objective(value-1e-4*direction))/(2e-4)); p = float(np.sum(ref[k]*direction)); pn = float(np.sum(native_grad[k]*direction))
            error = abs(fd-p)/max(abs(p), 1e-8); en = abs(fd-pn)/max(abs(pn), 1e-8)
            assert error <= 1e-5 and en <= 1e-3, ('additive_real_FD', label, k, error, en)
            arrays['direction_'+k] = direction; checks.append({'field': k, 'FD': fd, 'projection': p, 'native_projection': pn, 'reference_FD_error': error, 'native_FD_error': en})
    assert crossings == 0 and all(v.tobytes() == fingerprints[k] for k, v in offsets.items())
    removed_offset = offsets | {'logits': np.zeros_like(offsets['logits'])}
    wrong = numpy_path(x, b, s, pv, removed_offset)['logits']
    assert R.relative(wrong, anchor['logits']) > 1e-12 or np.max(np.abs(wrong-anchor['logits'])) > 1e-9
    for k in q: arrays['native_'+k] = anchor[k]; arrays['continuation_'+k] = npout[k]; arrays['offset_'+k] = offsets[k]
    for k in ref: arrays['native_gradient_'+k] = native_grad[k]; arrays['reference_gradient_'+k] = ref[k]
    np.savez(path, **arrays, **{'factor_'+k: v for k, v in values.items()}, pre=pre, base=base, scores=scores, target=target)
    return {'label': label, 'arm': arm, 'endpoints': endpoints, 'gradients': comparisons, 'FD': checks, 'zero_init_original_post_logits_exact': not tiny,
            'ReLU_crossings': crossings, 'offsets_fixed': True, 'dropped_head_offset_detected': True, 'archive_path': str(path)}
