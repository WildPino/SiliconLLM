"""Native-valued rank8 adapter control and affine-selector prerequisites."""
import numpy as np
import torch
import meth422_switch_function_autograd as G
import meth422_switch_function_gradient_contract as R
import meth424_switch_anchored_reference as L


def affine(x, weight, bias):
    return G.NativeFloat.apply(x, weight)+bias


def adapter_forward(pre, scores, chosen, ff, fn, head, parameters):
    x = G.NativeRMS.apply(pre, ff)
    bx = G.NativeFloat.apply(x, parameters['B']); ax = G.NativeFloat.apply(bx, parameters['A'])
    dx = G.NativeFloat.apply(x, parameters['D']); cx = G.NativeFloat.apply(dx, parameters['C'])
    down = ax+cx; p = G.NativeProbability.apply(scores, chosen)
    post = pre+p*down; final = G.NativeRMS.apply(post, fn)
    hi = final*np.float32(1/np.sqrt(pre.numel())); logits = G.NativeI8.apply(hi, head)
    return dict(input=x, Bx=bx, Ax=ax, Dx=dx, Cx=cx, down=down, probability=p,
                post=post, final=final, head_input=hi, logits=logits)


def qualify_adapter(label, pre, scores, chosen, ff, fn, head, parameters, target, path, tiny):
    values = {k: z.detach().numpy().copy() for k, z in parameters.items()}
    px = torch.tensor(pre.copy(), requires_grad=True); sx = torch.tensor(scores.copy(), requires_grad=True)
    actual = adapter_forward(px, sx, chosen, ff, fn, head, parameters)
    loss = G.prediction_loss(actual['logits'], torch.tensor(target)); loss.backward()
    native_grads = {k: z.grad.numpy().copy() for k, z in parameters.items()}
    native_grads.update(pre=px.grad.numpy().copy(), scores=sx.grad.numpy().copy())
    # Literal NumPy native primals, independent of the custom autograd path.
    anchor = {}; anchor['input'] = L.native_rms(pre, ff)
    dot = lambda w, x: (w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    for small, outer, inner in (('Bx', 'Ax', ('B', 'A')), ('Dx', 'Cx', ('D', 'C'))):
        anchor[small] = dot(values[inner[0]], anchor['input']); anchor[outer] = dot(values[inner[1]], anchor[small])
    anchor['down'] = anchor['Ax']+anchor['Cx']
    exp = np.exp((scores-scores[chosen]).astype(np.float64)).astype(np.float32).astype(np.float64)
    anchor['probability'] = np.float32(1/np.sum(exp))
    anchor['post'] = pre+anchor['probability']*anchor['down']; anchor['final'] = L.native_rms(anchor['post'], fn)
    scale = float(np.float32(1/np.sqrt(len(pre))))
    anchor['head_input'] = anchor['final']*np.float32(scale); anchor['logits'] = head.native(anchor['head_input'])
    for k, z in actual.items(): R.C.exact(z.detach().numpy(), anchor[k])
    q = {k: np.asarray(v, np.float64) for k, v in anchor.items()}; pv = {k: v.astype(np.float64) for k, v in values.items()}
    x = pre.astype(np.float64); s = scores.astype(np.float64); ff64 = ff.astype(np.float64); fn64 = fn.astype(np.float64); hw = head.smooth_weight().numpy()
    offsets = {
        'input': q['input']-L.rms(x, ff64), 'Bx': q['Bx']-pv['B']@q['input'], 'Ax': q['Ax']-pv['A']@q['Bx'],
        'Dx': q['Dx']-pv['D']@q['input'], 'Cx': q['Cx']-pv['C']@q['Dx'], 'down': q['down']-q['Ax']-q['Cx'],
        'probability': q['probability']-L.selected_probability(s, chosen), 'post': q['post']-x-q['probability']*q['down'],
        'final': q['final']-L.rms(q['post'], fn64), 'head_input': q['head_input']-q['final']*scale,
        'logits': q['logits']-hw@q['head_input']}
    def numpy_path(x, s, p):
        z = {}; z['input'] = L.rms(x, ff64)+offsets['input']
        for small, outer, inner in (('Bx', 'Ax', ('B', 'A')), ('Dx', 'Cx', ('D', 'C'))):
            z[small] = p[inner[0]].dot(z['input'])+offsets[small]; z[outer] = p[inner[1]].dot(z[small])+offsets[outer]
        z['down'] = z['Ax']+z['Cx']+offsets['down']; z['probability'] = L.selected_probability(s, chosen)+offsets['probability']
        z['post'] = x+z['probability']*z['down']+offsets['post']; z['final'] = L.rms(z['post'], fn64)+offsets['final']
        z['head_input'] = z['final']*scale+offsets['head_input']; z['logits'] = hw.dot(z['head_input'])+offsets['logits']; return z
    tx = torch.tensor(x, requires_grad=True); ts = torch.tensor(s, requires_grad=True)
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in pv.items()}
    z = {}; off = {k: torch.tensor(v) for k, v in offsets.items()}
    z['input'] = tx*torch.rsqrt(torch.mean(tx*tx)+1e-6)*torch.tensor(ff64)+off['input']
    for small, outer, inner in (('Bx', 'Ax', ('B', 'A')), ('Dx', 'Cx', ('D', 'C'))):
        z[small] = tp[inner[0]] @ z['input']+off[small]; z[outer] = tp[inner[1]] @ z[small]+off[outer]
    z['down'] = z['Ax']+z['Cx']+off['down']; z['probability'] = torch.softmax(ts, dim=0)[chosen]+off['probability']
    z['post'] = tx+z['probability']*z['down']+off['post']
    z['final'] = z['post']*torch.rsqrt(torch.mean(z['post']*z['post'])+1e-6)*torch.tensor(fn64)+off['final']
    z['head_input'] = z['final']*scale+off['head_input']; z['logits'] = head.smooth_weight() @ z['head_input']+off['logits']
    npout = numpy_path(x, s, pv); endpoints = {}
    for k in q:
        errors = [R.relative(npout[k], q[k]), R.relative(z[k].detach().numpy(), q[k])]
        absmax = max(float(np.max(np.abs(npout[k]-q[k]))), float(np.max(np.abs(z[k].detach().numpy()-q[k]))))
        assert max(errors) <= 1e-12 and absmax <= 1e-9, ('adapter_endpoint', label, k, errors, absmax)
        endpoints[k] = {'maximum_relative': max(errors), 'absolute_max': absmax}
    G.prediction_loss(z['logits'], torch.tensor(target)).backward()
    refgrads = {k: v.grad.numpy().copy() for k, v in tp.items()}; refgrads.update(pre=tx.grad.numpy().copy(), scores=ts.grad.numpy().copy())
    errors = {k: R.relative(native_grads[k], refgrads[k]) for k in native_grads}; assert all(v <= 1e-3 for v in errors.values()), ('adapter_gradient', label, errors)
    checks = []; arrays = {}; rng = np.random.default_rng(425)
    base_logits = npout['logits'].copy(); base_probability = R.probability(base_logits)
    base_maximum = int(np.argmax(base_logits))
    for k in ('A', 'B', 'C', 'D', 'pre', 'scores'):
        v = x if k == 'pre' else s if k == 'scores' else pv[k]
        def changed_logits(a):
            xx = a if k == 'pre' else x; ss = a if k == 'scores' else s; pp = pv | {k: a} if k in pv else pv
            return numpy_path(xx, ss, pp)['logits']
        def old_objective(a): return R.numpy_loss(changed_logits(a), target)
        def objective(a):
            delta = changed_logits(a)-base_logits
            delta = delta-delta[base_maximum]
            return np.log1p(np.sum(base_probability*np.expm1(delta)))-np.dot(target, delta)
        if tiny:
            old_fd = R.finite_difference(old_objective, v, 1e-4)
            old_error = R.relative(refgrads[k], old_fd)
            if k == 'scores':
                assert abs(old_error-3.925822518311817e-5) <= 1e-12, ('original425_score_FD_not_reproduced', old_error)
            arrays['old_FD_'+k] = old_fd
            fd = R.finite_difference(objective, v, 1e-4); e = R.relative(refgrads[k], fd)
            assert e <= 1e-5 and R.relative(native_grads[k], fd) <= 1e-3, ('adapter_FD', label, k, e)
            arrays['FD_'+k] = fd; checks.append({'field': k, 'coordinates': v.size, 'FD_error': e, 'old_scalar_CE_subtraction_FD_error': old_error})
        else:
            direction = rng.choice([-1., 1.], size=v.shape)/np.sqrt(v.size); epsilon = 1e-4
            fd = float((objective(v+epsilon*direction)-objective(v-epsilon*direction))/(2*epsilon))
            projected = float(np.sum(refgrads[k]*direction)); e = abs(fd-projected)/max(abs(projected), 1e-8)
            assert e <= 1e-5, ('adapter_real_FD', label, k, e)
            arrays['direction_'+k] = direction; checks.append({'field': k, 'FD': fd, 'projection': projected, 'FD_error': e})
    for k in native_grads: arrays['native_gradient_'+k] = native_grads[k]; arrays['reference_gradient_'+k] = refgrads[k]
    for k in q: arrays['native_'+k] = anchor[k]; arrays['continuation_'+k] = npout[k]; arrays['offset_'+k] = offsets[k]
    np.savez(path, **arrays, **{'factor_'+k: v for k, v in values.items()}, pre=pre, scores=scores, target=target)
    return {'label': label, 'independent_native_and_continuation_endpoints': endpoints, 'gradients': errors, 'FD': checks, 'archive_path': str(path)}


def qualify_affine():
    rng = np.random.default_rng(425); x = rng.normal(size=7).astype(np.float32); checks = []
    for binary in (False, True):
        n = 1 if binary else 5; w = rng.normal(0, .02, (n, 7)).astype(np.float32); b = np.linspace(-.1, .1, n, dtype=np.float32)
        tw = torch.tensor(w, requires_grad=True); tb = torch.tensor(b, requires_grad=True)
        z = affine(torch.tensor(x), tw, tb); target = np.zeros(n); target[-1] = 1
        loss = torch.nn.functional.softplus(-z[0].to(torch.float64)) if binary else G.prediction_loss(z, torch.tensor(target))
        loss.backward(); native = z.detach().numpy().astype(np.float64)
        offset = native-(w.astype(np.float64)@x.astype(np.float64)+b.astype(np.float64))
        def objective(ww, bb):
            logits = ww.dot(x.astype(np.float64))+bb+offset
            return float(np.logaddexp(0., -logits[0])) if binary else R.numpy_loss(logits, target)
        for name, value, gradient in (('weight', w, tw.grad.numpy()), ('bias', b, tb.grad.numpy())):
            fd = R.finite_difference(lambda a: objective(a, b.astype(np.float64)) if name == 'weight' else objective(w.astype(np.float64), a), value.astype(np.float64))
            error = R.relative(gradient, fd); assert error <= 1e-5, ('affine_FD', binary, name, error)
            checks.append({'binary': binary, 'field': name, 'coordinates': value.size, 'FD_relative': error})
    ties = affine(torch.tensor(x), torch.zeros((5, 7)), torch.zeros(5)).numpy(); assert int(np.argmax(ties)) == 0
    assert float(affine(torch.tensor(x), torch.zeros((1, 7)), torch.tensor([-.001]))[0]) < 0
    return {'independent_all_coordinate_affine_CE_BCE_FD': checks, 'lower_ID_tie_and_initial_gate_off': True}
