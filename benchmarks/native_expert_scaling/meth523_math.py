"""One donor folding/hinge rule and prospectively bounded energy decomposition."""
import itertools
import math
import numpy as np
from meth523_contract import ARMS


def quant(x):
    x = np.asarray(x, dtype='<f4'); mx = np.max(np.abs(x), axis=1)
    scale = np.divide(mx, np.float32(32767), dtype=np.float32); scale[mx == 0] = 1
    assert np.all(scale > 0) and np.isfinite(scale).all()
    code = np.clip(np.rint(np.divide(x, scale[:, None], dtype=np.float32)), -32767, 32767).astype('<i2')
    return code, scale


def anchor(q, uids):
    unit = q.astype('<f8'); norms = np.sqrt(np.sum(unit * unit, axis=1))
    unit = np.divide(unit, norms[:, None], out=np.zeros_like(unit), where=norms[:, None] != 0)
    return int(uids[int(np.argmax(unit @ unit.mean(axis=0)))])


def metrics(planes, f, y, p):
    rows = []
    for weighted in (False, True):
        v = planes.copy(); ref = y.astype('<f8') if weighted else f.astype('<f8')
        if weighted:
            v[:, :4] *= p.astype('<f8')[:, None, None]
            v[:, 4] = np.multiply(planes[:, 4].astype('<f4'), p[:, None], dtype=np.float32).astype('<f8')
        errors = v - ref[:, None, :]
        components = [v[:, 4] - v[:, 3], v[:, 3] - v[:, 2], v[:, 2] - v[:, 0], v[:, 0] - ref]
        diagonals = [np.sum(c * c, axis=1) for c in components]
        crosses = [np.sum(components[i] * components[j], axis=1) for i, j in itertools.combinations(range(4), 2)]
        bound = 3e-12 * np.maximum(1, np.abs(ref) + np.sum(np.abs(v), axis=1))
        ratio = np.max(np.abs(sum(components) - errors[:, 4]) / bound, axis=1)
        total = np.sum(errors[:, 4] ** 2, axis=1)
        energy_bound = 2e-11 * np.maximum(1, total + sum(diagonals) + 2 * sum(np.abs(c) for c in crosses))
        closure = np.abs(total - sum(diagonals) - 2 * sum(crosses)) / energy_bound
        assert np.isfinite(ratio).all() and np.max(ratio) <= 1 and np.max(closure) <= 1
        # 18 columns per weighting: source energy, five arm errors, four terms,
        # six cross terms, vector telescoping and energy closure ratios.
        rows.append(np.column_stack((np.sum(ref * ref, axis=1), *[np.sum(e * e, axis=1) for e in errors.transpose(1, 0, 2)],
            *diagonals, *crosses, ratio, closure)))
    return np.column_stack(rows)


def summarize(label, ids, data):
    total = np.sum(data[ids], axis=0) if len(ids) else np.zeros(36)
    out = {**label, 'count': len(ids), 'sums': total[:16].tolist() + total[18:34].tolist()}
    for prefix, off in (('unweighted', 0), ('weighted', 18)):
        den = float(total[off]); out[prefix + '_source_energy'] = den
        for i, name in enumerate(ARMS, 1):
            out[prefix + '_' + name + '_RMS'] = math.sqrt(float(total[off + i]) / den) if den else None
        out[prefix + '_identity_ratio'] = float(np.max(data[ids, off + 16])) if len(ids) else 0.
        out[prefix + '_closure_ratio'] = float(np.max(data[ids, off + 17])) if len(ids) else 0.
    return out


def eligibility(views):
    six = [v for v in views if v['kind'] == 'role']; assert len(six) == 6
    ans = {prefix + '_' + arm + '_ALL_six_RMS_1pct': all(v['count'] and v[prefix + '_' + arm + '_RMS'] is not None and v[prefix + '_' + arm + '_RMS'] <= .01 for v in six)
        for prefix, arm in (('unweighted', 'continuous_hybrid'), ('unweighted', 'physical_reference'), ('weighted', 'physical_reference'))}
    rare = [v for v in views if v['kind'] == 'rare' and v['development_class'] in ('1..4', '5..15') and v['split'] == 'consumed_validation']
    ans['ALL_nonempty_rare_consumed_physical_RMS_1pct'] = all(not v['count'] or all(v[p + '_physical_reference_RMS'] is not None and v[p + '_physical_reference_RMS'] <= .01 for p in ('weighted', 'unweighted')) for v in rare)
    ans['logical_active_coefficients_le_75pct_original'] = 1966080 / 4718592 <= .75
    return ans


def controls():
    I = np.array([[1, 0], [0, 1], [1, 1]], '<f8'); O = np.array([[1, 2, 3], [-1, 0, 1]], '<f8')
    s = np.array([1, 0, 1]); u = np.array([1]); l = O[:, [0, 2]] @ I[[0, 2]]
    records = []
    for x in (np.array([1, -.5]), np.array([-1, 2]), np.zeros(2)):
        z = I @ x; f = O @ np.maximum(z, 0); g = l @ x + O[:, u] @ np.maximum(z[u], 0)
        residual = O[:, [0, 2]] @ (np.maximum(z[[0, 2]], 0) - s[[0, 2]] * z[[0, 2]])
        assert np.array_equal(g + residual, f); records.append(dict(x=x.tolist(), source=f.tolist(), hybrid=g.tolist(), residual=residual.tolist()))
    assert records[0]['residual'] == [0., 0.] and records[1]['residual'] == [1., -1.]
    q, a = quant(np.array([[32767, -32767, 2.5, 3.5, -2.5, -3.5, 0, 1]], '<f4'))
    assert q.tolist() == [[32767, -32767, 2, 4, -2, -4, 0, 1]] and a.tolist() == [1.]
    q0, a0 = quant(np.zeros((1, 3), '<f4')); assert not q0.any() and a0.tolist() == [1.]
    wide = np.full(768, 32767, '<i2'); dot = float(wide.astype('<f8') @ wide.astype('<f8'))
    assert dot == 768 * 32767 ** 2 > 2 ** 31 and dot < 2 ** 53
    return dict(piecewise=records, I16_tie_codes=q.tolist(), zero_alpha=float(a0[0]), I16_wide_exact_dot=int(dot))
