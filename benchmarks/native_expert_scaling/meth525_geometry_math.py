"""Certified row projector and nearest opposite-half-interval sphere query."""
import math
import numpy as np

U = 2. ** -53


def gamma(n): return n * U / (1 - n * U)


def rank_mod(rows, prime):
    inv = {}; a = np.empty(rows.shape, '<i8')
    for i, row in enumerate(rows):
        for j, f in enumerate(row):
            num, den = float(f).as_integer_ratio()
            if den not in inv: inv[den] = pow(den, prime - 2, prime)
            a[i, j] = (num * inv[den]) % prime
    pivots = []; cursor = 0
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[cursor:, col])
        if not len(candidates): continue
        pivot = cursor + int(candidates[0]); a[[cursor, pivot]] = a[[pivot, cursor]]
        a[cursor, col:] = (a[cursor, col:] * pow(int(a[cursor, col]), prime - 2, prime)) % prime
        for k in range(cursor + 1, len(a)):
            a[k, col:] = (a[k, col:] - a[k, col] * a[cursor, col:]) % prime
        pivots.append(col); cursor += 1
        if cursor == len(a): break
    return pivots


def independent_rows(router):
    keep = []; removed = []; known = {}
    for i, row in enumerate(router):
        ids = np.flatnonzero(row)
        if not len(ids): removed.append(dict(row=i, reason='exact_zero')); continue
        sign = 1 if row[ids[0]] > 0 else -1
        canonical = row.astype('<f8') * sign; canonical[canonical == 0] = 0
        key = canonical.tobytes()
        if key in known: removed.append(dict(row=i, reason='exact_signed_duplicate', representative=known[key][0], multiplier=sign * known[key][1]))
        else: known[key] = (i, sign); keep.append(i)
    return keep, removed


def projector(router, norm):
    keep, removed = independent_rows(router); c = router[keep].astype('<f8') * norm.astype('<f8')
    pivots = rank_mod(router[keep], 2147483647)
    q, sv, vt = np.linalg.svd(c.T, full_matrices=False)
    assert len(keep) > 0 and np.isfinite(q).all() and np.isfinite(sv).all()
    oq = float(np.linalg.norm(q.T @ q - np.eye(len(keep)), 'fro') + gamma(769) * np.linalg.norm(np.abs(q).T @ np.abs(q), 'fro'))
    ov = float(np.linalg.norm(vt @ vt.T - np.eye(len(keep)), 'fro') + gamma(len(keep) + 1) * np.linalg.norm(np.abs(vt) @ np.abs(vt).T, 'fro'))
    reconstructed = (q * sv) @ vt
    rec = float(np.linalg.norm(c.T - reconstructed, 'fro') + gamma(len(keep) + 3) * np.linalg.norm(np.abs(c.T) + (np.abs(q) * sv) @ np.abs(vt), 'fro'))
    lower = float(sv[-1] * math.sqrt(max(0, 1 - oq)) * math.sqrt(max(0, 1 - ov)) - 2 * rec)
    residue = c - (c @ q) @ q.T
    rb = float(np.linalg.norm(residue, 'fro') + gamma(768 + 2 * len(keep) + 5) * np.linalg.norm(np.abs(c) + (np.abs(c) @ np.abs(q)) @ np.abs(q).T, 'fro'))
    cnorm = float(np.linalg.norm(c, 'fro') * (1 + gamma(c.size + 1)))
    bound = 2 * (rb + cnorm * oq / (1 - oq)) / lower + 2 * oq / (1 - oq) if lower > 0 and oq < 1 else 1.
    report = dict(kept_rows=keep, removed_rows=removed, rank_prime=2147483647, rank=len(pivots), pivot_columns=pivots,
        basis_columns=len(keep), null_dimensions=768 - len(keep), singular_values=sv.tolist(),
        orthogonality_bound=oq, reconstruction_bound=2 * rec, minimum_singular_lower=lower,
        rowspace_residual_bound=2 * rb, projector_error_bound=float(2 * bound))
    eligible = len(pivots) == len(keep) and 768 - len(keep) >= 2 and lower > 0 and sv[-1] / sv[0] >= 1e-6 and bound < 1e-6
    return q, sv, vt, report, bool(eligible)


def panels(wi, norm, y, q, pb, signs, hinges):
    h = wi.astype('<f8') * norm.astype('<f8')
    a = (y @ q) @ q.T; b = y - a; radius = float(np.linalg.norm(b)); yn = float(np.linalg.norm(y))
    ph = (h @ q) @ q.T; k = h - ph; kn = np.linalg.norm(k, axis=1); hn = np.linalg.norm(h, axis=1)
    reduction = gamma(768 + 2 * q.shape[1] + 8)
    da = pb * yn + reduction * float(np.linalg.norm((np.abs(y) @ np.abs(q)) @ np.abs(q).T))
    db = da + gamma(2) * (yn + float(np.linalg.norm(a)))
    dr = db + gamma(770) * radius
    dk = pb * hn + reduction * np.linalg.norm((np.abs(h) @ np.abs(q)) @ np.abs(q).T, axis=1) + gamma(2) * (hn + np.linalg.norm(ph, axis=1))
    dkn = dk + gamma(770) * kn
    c = h @ a; dc = hn * da + gamma(769) * (np.abs(h) @ np.abs(a))
    spread = radius * kn; de = dc + radius * dkn + kn * dr + dr * dkn + gamma(3) * (np.abs(c) + spread)
    lower, upper = c - spread, c + spread
    target = np.where(signs == 0, upper / 2, lower / 2)
    values = np.zeros((3072, 7), '<f8'); values[:, 0] = c; values[:, 1] = spread; values[:, 2] = np.nextafter(2 * de, np.inf)
    values[:, 3] = target; values[:, 6] = kn
    status = np.zeros(3072, 'u1'); status[hinges] = 1
    candidates = (status == 0) & (lower + values[:, 2] < 0) & (upper - values[:, 2] > 0) & (kn > 2 * dkn)
    status[(status == 0) & ~candidates] = 2
    ids = np.flatnonzero(candidates)
    if not len(ids): return values, status, dict(radius=radius, y_norm=yn), None
    n = k[ids] / kn[ids, None]
    dn = (dk[ids] + dkn[ids]) / (kn[ids] - dkn[ids]) + gamma(2) * np.linalg.norm(n, axis=1)
    u = (target[ids] - c[ids]) / kn[ids]
    du = (.5 * values[ids, 2] + dc[ids]) / (kn[ids] - dkn[ids]) + np.abs(u) * dkn[ids] / (kn[ids] - dkn[ids]) + gamma(3) * np.abs(u)
    alignment = n @ b; dal = dn * np.linalg.norm(b) + db * np.linalg.norm(n, axis=1) + dn * db + gamma(769) * (np.abs(n) @ np.abs(b))
    tangent = b - alignment[:, None] * n; tn = np.linalg.norm(tangent, axis=1)
    dt = db + np.abs(alignment) * dn + np.linalg.norm(n, axis=1) * dal + dn * dal + gamma(3) * (np.linalg.norm(b) + np.abs(alignment) * np.linalg.norm(n, axis=1))
    dtn = dt + gamma(770) * tn
    arg = radius * radius - u * u
    safe = (radius > dr) & ((radius - dr) ** 2 > (np.abs(u) + du) ** 2) & (tn > 2 * dtn)
    status[ids[~safe]] = 3
    good = np.flatnonzero(safe)
    sqrt = np.sqrt(np.maximum(arg, 0))
    darg = 2 * radius * dr + dr * dr + 2 * np.abs(u) * du + du * du + gamma(3) * (radius ** 2 + u * u)
    dsqrt = np.zeros(len(ids))
    dsqrt[good] = darg[good] / (sqrt[good] + np.sqrt((radius - dr) ** 2 - (np.abs(u[good]) + du[good]) ** 2)) + U * sqrt[good]
    cross = alignment * u + tn * sqrt
    distance = 2 * radius ** 2 - 2 * cross
    error = 4 * radius * dr + 2 * dr * dr + 2 * (np.abs(alignment) * du + np.abs(u) * dal + dal * du + tn * dsqrt + sqrt * dtn + dtn * dsqrt + gamma(3) * (np.abs(alignment * u) + tn * sqrt)) + gamma(3) * (2 * radius ** 2 + 2 * np.abs(cross))
    values[ids[good], 4] = np.maximum(distance[good], 0)
    values[ids[good], 5] = np.nextafter(4 * error[good], np.inf)
    # Repeated signed/proportional code planes represent one physical query.
    keys = set(); reps = []
    for j in ids[good]:
        primitive = wi[j].astype('<i2'); g = int(np.gcd.reduce(np.abs(primitive))); assert g > 0
        primitive //= g; first = int(primitive[np.flatnonzero(primitive)[0]]); s = int(signs[j])
        if first < 0: primitive = -primitive; s = 1 - s
        key = (primitive.tobytes(), s)
        if key in keys: status[j] = 4
        else: keys.add(key); reps.append(int(j))
    if not reps: return values, status, dict(radius=radius, y_norm=yn), None
    order = sorted(reps, key=lambda j: (values[j, 4], j)); j = order[0]
    resolved = all(values[j, 4] + values[j, 5] < values[t, 4] - values[t, 5] for t in order[1:])
    local = int(np.searchsorted(ids, j))
    yp = a + u[local] * n[local] + sqrt[local] * tangent[local] / tn[local]
    unit_error = (dt[local] + dtn[local]) / (tn[local] - dtn[local]) + gamma(2)
    point_error = da + abs(u[local]) * dn[local] + du[local] * np.linalg.norm(n[local]) + dn[local] * du[local]
    point_error += sqrt[local] * unit_error + dsqrt[local] * (1 + unit_error)
    point_error += gamma(8) * (np.linalg.norm(a) + abs(u[local]) * np.linalg.norm(n[local]) + sqrt[local])
    assert np.isfinite(values).all() and np.isfinite(yp).all()
    return values, status, dict(radius=radius, y_norm=yn, reachable_representatives=len(reps), selected_neuron=j,
        order_resolved=bool(resolved), distance2=float(values[j, 4]), distance2_bound=float(values[j, 5]),
        target=float(target[j]), point_error_bound=float(8 * point_error)), yp


def quant(x):
    x = np.asarray(x, dtype='<f4'); a = np.float32(np.max(np.abs(x)) / np.float32(32767))
    if a == 0: a = np.float32(1)
    return np.clip(np.rint(np.divide(x, a, dtype=np.float32)), -32767, 32767).astype('<i2'), a


def router_physical(router, x):
    a = router.astype('<f8').reshape(128, 96, 8); b = x.astype('<f8').reshape(96, 8)
    lo = np.zeros((128, 4)); hi = np.zeros((128, 4))
    for i in range(96): lo += a[:, i, :4] * b[i, :4]; hi += a[:, i, 4:] * b[i, 4:]
    lanes = lo + hi; sums = np.zeros(128)
    for i in range(4): sums += lanes[:, i]
    scores = sums.astype('<f4'); e = int(np.argmax(scores)); delta = np.subtract(scores, scores[e], dtype=np.float32)
    exp = np.exp(delta.astype('<f8')).astype('<f4'); z = 0.
    for v in exp: z += float(v)
    return scores, e, np.float32(1. / z), z, exp
