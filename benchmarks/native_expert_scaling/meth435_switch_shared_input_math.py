"""Native-valued shared input mapper and independently anchored local derivative."""
import numpy as np
import torch
import meth422_switch_function_autograd as G
import meth422_switch_function_gradient_contract as R


def parameters(d, hidden, seed, zero=True):
    rng = np.random.default_rng(seed)
    values = {'W1': (rng.normal(size=(hidden, d))/np.sqrt(d)).astype(np.float32), 'b1': np.zeros(hidden, np.float32),
              'W2': np.zeros((d, hidden), np.float32), 'b2': np.zeros(d, np.float32)}
    if not zero:
        values['W2'] = (rng.normal(size=(d, hidden))*.03).astype(np.float32)
        values['b1'] = np.linspace(-.2, .2, hidden, dtype=np.float32); values['b2'] = (rng.normal(size=d)*.02).astype(np.float32)
    return {k: torch.tensor(v, requires_grad=True) for k, v in values.items()}


def forward(x, buffers, params):
    mx, sx, my, sy = (torch.from_numpy(v) for v in buffers)
    standardized = (x-mx)/sx; raw = G.NativeFloat.apply(standardized, params['W1'])+params['b1']
    hidden = G.NativeReLU.apply(raw); unit = G.NativeFloat.apply(hidden, params['W2'])+params['b2']; prediction = my+sy*unit
    return dict(standardized=standardized, raw=raw, hidden=hidden, unit=unit, prediction=prediction)


def loss(prediction, target, output_scale):
    return torch.mean(((prediction.to(torch.float64)-target.to(torch.float64))/torch.from_numpy(output_scale).to(torch.float64))**2)


def smooth(x, buffers, params):
    mx, sx, my, sy = [v.astype(np.float64) for v in buffers]; pp = {k: v.detach().numpy().astype(np.float64) for k, v in params.items()}
    z = (x.astype(np.float64)-mx)/sx; hidden = np.maximum(pp['W1']@z+pp['b1'], 0)
    return my+sy*(pp['W2']@hidden+pp['b2'])


def qualify(label, x, target, buffers, params, tiny, path):
    values = {k: v.detach().numpy().copy() for k, v in params.items()}; tx = torch.tensor(x, requires_grad=True)
    native = forward(tx, buffers, params); objective = loss(native['prediction'], torch.tensor(target), buffers[3]); objective.backward()
    ng = {k: v.grad.numpy().copy() for k, v in params.items()}; ng['x'] = tx.grad.numpy().copy()
    if tiny: assert all(np.linalg.norm(ng[k]) > 0 for k in ng)
    else:
        assert all(np.linalg.norm(ng[k]) > 1e-10 for k in ('W2', 'b2')) and all(np.count_nonzero(ng[k]) == 0 for k in ('W1', 'b1', 'x'))
        R.C.exact(native['prediction'].detach().numpy(), buffers[2])
    mx, sx, my, sy = [v.astype(np.float64) for v in buffers]; xx = x.astype(np.float64); tt = target.astype(np.float64); pp = {k: v.astype(np.float64) for k, v in values.items()}
    fmx, fsx, fmy, fsy = buffers
    anchor = {}; anchor['standardized'] = (x-fmx)/fsx
    anchor['raw'] = (pp['W1']@anchor['standardized'].astype(np.float64)).astype(np.float32)+values['b1']
    anchor['hidden'] = np.where(anchor['raw'] < 0, np.float32(0), anchor['raw'])
    anchor['unit'] = (pp['W2']@anchor['hidden'].astype(np.float64)).astype(np.float32)+values['b2']; anchor['prediction'] = fmy+fsy*anchor['unit']
    for k, v in anchor.items(): R.C.exact(native[k].detach().numpy(), v)
    q = {k: v.astype(np.float64) for k, v in anchor.items()}
    parent = {'standardized': (xx-mx)/sx, 'raw': pp['W1']@q['standardized']+pp['b1'], 'hidden': np.maximum(q['raw'], 0),
              'unit': pp['W2']@q['hidden']+pp['b2'], 'prediction': my+sy*q['unit']}
    offsets = {k: q[k]-parent[k] for k in q}; fixed = {k: v.tobytes() for k, v in offsets.items()}
    def numpy_path(input_x, p, off=offsets, relu=True):
        z = {}; z['standardized'] = (input_x-mx)/sx+off['standardized']; z['raw'] = p['W1']@z['standardized']+p['b1']+off['raw']
        z['hidden'] = (np.maximum(z['raw'], 0) if relu else z['raw'])+off['hidden']; z['unit'] = p['W2']@z['hidden']+p['b2']+off['unit']
        z['prediction'] = my+sy*z['unit']+off['prediction']; return z
    fx = torch.tensor(xx, requires_grad=True); fp = {k: torch.tensor(v, requires_grad=True) for k, v in pp.items()}; off = {k: torch.tensor(v) for k, v in offsets.items()}
    ref = {}; ref['standardized'] = (fx-torch.tensor(mx))/torch.tensor(sx)+off['standardized']
    ref['raw'] = fp['W1']@ref['standardized']+fp['b1']+off['raw']; ref['hidden'] = torch.relu(ref['raw'])+off['hidden']
    ref['unit'] = fp['W2']@ref['hidden']+fp['b2']+off['unit']; ref['prediction'] = torch.tensor(my)+torch.tensor(sy)*ref['unit']+off['prediction']
    npout = numpy_path(xx, pp); endpoints = {}
    for k in q:
        relative = max(R.relative(npout[k], q[k]), R.relative(ref[k].detach().numpy(), q[k])); absolute = max(float(np.max(np.abs(npout[k]-q[k]))), float(np.max(np.abs(ref[k].detach().numpy()-q[k]))))
        assert relative <= 1e-12 and absolute <= 1e-9, ('endpoint', label, k, relative, absolute)
        endpoints[k] = dict(relative=relative, absolute=absolute)
    torch.mean(((ref['prediction']-torch.tensor(tt))/torch.tensor(sy))**2).backward()
    rg = {k: v.grad.numpy().copy() for k, v in fp.items()}; rg['x'] = fx.grad.numpy().copy()
    gradient_errors = {k: R.relative(ng[k], rg[k]) for k in rg}; assert max(gradient_errors.values()) <= 1e-3, ('gradient', label, gradient_errors)
    arrays = {}; checks = []; rng = np.random.default_rng(435+len(x)); crossings = 0; baseline = npout['prediction']; residual = baseline-tt
    for k in ('W1', 'b1', 'W2', 'b2', 'x'):
        value = xx if k == 'x' else pp[k]
        def objective_at(v):
            nonlocal crossings
            z = numpy_path(v if k == 'x' else xx, pp | {k: v} if k in pp else pp)
            crossings += int(np.count_nonzero((z['raw'] > 0) != (anchor['raw'] > 0)))
            delta = z['prediction']-baseline
            return np.mean((2*residual*delta+delta*delta)/(sy*sy))
        if tiny:
            fd = R.finite_difference(objective_at, value, 1e-4); er = R.relative(rg[k], fd); en = R.relative(ng[k], fd)
            assert er <= 1e-5 and en <= 1e-3, ('tiny_FD', label, k, er, en)
            arrays['FD_'+k] = fd; checks.append(dict(field=k, coordinates=value.size, reference_FD_error=er, native_FD_error=en))
        else:
            direction = rng.choice([-1., 1.], size=value.shape)/np.sqrt(value.size); fd = float((objective_at(value+1e-4*direction)-objective_at(value-1e-4*direction))/(2e-4))
            pr = float(np.sum(rg[k]*direction)); pn = float(np.sum(ng[k]*direction)); er = abs(fd-pr)/max(abs(pr), 1e-8); en = abs(fd-pn)/max(abs(pn), 1e-8)
            assert er <= 1e-5 and en <= 1e-3, ('real_FD', label, k, er, en)
            arrays['direction_'+k] = direction; checks.append(dict(field=k, FD=fd, reference_projection=pr, native_projection=pn, reference_FD_error=er, native_FD_error=en))
    assert crossings == 0 and all(v.tobytes() == fixed[k] for k, v in offsets.items())
    if tiny:
        assert np.any(anchor['raw'] < 0) and np.any(anchor['raw'] > 0)
        assert R.relative(numpy_path(xx, pp, relu=False)['prediction'], baseline) > 1e-5
        wrong = offsets | {'prediction': np.zeros_like(offsets['prediction'])}
        assert R.relative(numpy_path(xx, pp, wrong)['prediction'], baseline) > 1e-12 or np.max(np.abs(numpy_path(xx, pp, wrong)['prediction']-baseline)) > 1e-9
    for k in q: arrays['native_'+k] = anchor[k]; arrays['continuation_'+k] = npout[k]; arrays['offset_'+k] = offsets[k]
    for k in rg: arrays['native_gradient_'+k] = ng[k]; arrays['reference_gradient_'+k] = rg[k]
    np.savez(path, **arrays, **{'parameter_'+k: v for k, v in values.items()}, x=x, target=target, mx=buffers[0], sx=buffers[1], my=buffers[2], sy=buffers[3])
    return dict(label=label, endpoints=endpoints, gradient_errors=gradient_errors, FD=checks, ReLU_crossings=crossings, offsets_fixed=True,
                tiny_negative_controls=tiny, zero_output_mean_exact=not tiny, archive_path=str(path))
