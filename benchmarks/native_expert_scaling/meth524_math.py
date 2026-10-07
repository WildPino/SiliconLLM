"""Development-only radial residual information; bounded F64 reductions."""
import numpy as np

U = 2. ** -53


def gamma(n):
    assert n * U < .01
    return n * U / (1 - n * U)


def explained(s, d, ds, dd):
    a = np.sum(s * s, axis=-1)
    e = np.divide(a, d, out=np.zeros_like(a), where=d > 0)
    da = np.sum(2 * np.abs(s) * ds + ds * ds, axis=-1) + gamma(s.shape[-1] + 2) * a
    denom = d - dd
    b = np.divide(da + e * dd, denom, out=np.zeros_like(a), where=denom > 0) + U * np.abs(e)
    assert np.all((d == 0) | (denom > 0))
    return e, b


def panel(sign, residual, rho, p, source):
    n, dim = residual.shape; k = sign.shape[1]
    assert sign.shape[0] == n and np.isfinite(residual).all() and np.isfinite(rho).all()
    counts = np.sum(sign, axis=0, dtype=np.int64)
    valid = (counts >= 2) & (counts <= n - 2)
    answer = np.zeros((k, 4), '<f8')
    for arm, w in enumerate((np.ones(n), p * p)):
        v = (w * rho)[:, None] * residual
        t = (w * rho) * rho
        dv = gamma(n + 3) * np.sum(np.abs(v), axis=0)
        positive = sign.T.astype('<f8'); negative = (~sign).T.astype('<f8')
        sp, sm = positive @ v, negative @ v
        dp, dm = positive @ t, negative @ t
        st, total = np.sum(v, axis=0), np.sum(t)
        ep, bp = explained(sp, dp, dv, gamma(n + 3) * dp)
        em, bm = explained(sm, dm, dv, gamma(n + 3) * dm)
        et, bt = explained(st, np.asarray(total), dv, np.asarray(gamma(n + 3) * total))
        g = (ep + em) - et
        bg = bp + bm + bt + gamma(3) * (np.abs(ep) + np.abs(em) + abs(float(et)))
        energy = float(np.sum(w[:, None] * source * source))
        if energy == 0:
            assert not np.any(residual); continue
        de = gamma(n * dim + 4) * energy
        score = g / energy
        bound = bg / (energy - de) + np.abs(g) * de / (energy * (energy - de)) + U * np.abs(score)
        answer[:, arm] = score
        answer[:, 3] += bound
    answer[:, 2] = answer[:, 0] + answer[:, 1]
    # Positive-bound evaluation uses <10000 floating operations per scalar;
    # gamma(10000)<2e-12. Factor2 also protects final bound storage rounding.
    answer[:, 3] = np.nextafter(2 * (answer[:, 3] + U * (np.abs(answer[:, 0]) + np.abs(answer[:, 1]))), np.inf)
    assert np.isfinite(answer).all() and np.all(answer[:, 3] >= 0)
    assert np.all(answer[valid, 2] >= -answer[valid, 3])
    return answer, counts.astype('<u2'), valid


def representatives(sign, valid, excluded):
    reps = []; keys = set()
    for j in np.flatnonzero(valid):
        if int(j) in excluded: continue
        a = np.packbits(sign[:, j], bitorder='little').tobytes()
        b = np.packbits(~sign[:, j], bitorder='little').tobytes()
        key = min(a, b)
        if key not in keys: keys.add(key); reps.append(int(j))
    return reps


def choose(panel_values, reps):
    if not reps: return None, 'no_balanced_partition', None
    order = sorted(reps, key=lambda j: (-panel_values[j, 2], j)); j = order[0]
    lower = float(panel_values[j, 2] - panel_values[j, 3])
    if lower <= 1e-12:
        status = 'no_resolved_positive_gain' if panel_values[j, 2] + panel_values[j, 3] <= 1e-12 else 'unresolved_positive_gain'
        return None, status, j
    if len(order) > 1:
        upper = max(float(panel_values[t, 2] + panel_values[t, 3]) for t in order[1:])
        if lower <= upper: return None, 'unresolved_gain_order', j
    return j, 'split', j


def controls():
    sign = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]], bool)
    r = np.array([[1, 1], [-1, 1], [2, 0], [-2, 0]], '<f8')
    rho = np.array([1, 2, 3, 4], '<f8'); p = np.array([.5, .25, .5, .25], '<f8')
    source = r + np.array([.5, 1.])
    a, c, valid = panel(sign, r, rho, p, source)
    reps = representatives(sign, valid, [])
    assert reps == [0, 1] and c.tolist() == [2, 2, 2]
    assert np.array_equal(sign[:, 0], ~sign[:, 2])
    return dict(sign=sign.tolist(), residual=r.tolist(), rho=rho.tolist(), p=p.tolist(), source=source.tolist(),
                panel=a.tolist(), positive_counts=c.tolist(), representatives=reps,
                zero_sign_branch='nonpositive', winner=choose(a, reps)[0],
                hard_child_mass_fraction_control={'parents': ['1/2', '1/3', '1/6'], 'children': [0, 2, 1], 'sum': '1'})
