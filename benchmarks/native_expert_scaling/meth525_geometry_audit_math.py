"""Independent QR-basis panels; explicit einsum reductions, no main imports."""
import numpy as np
U = 2. ** -53
def gamma(n): return n * U / (1 - n * U)

def panels(wi, norm, y, q, pb, signs, hinges):
    h = wi.astype('<f8') * norm.astype('<f8')
    a = np.einsum('k,ik->i', np.einsum('j,jk->k', y, q, optimize=False), q, optimize=False); b = y - a; radius = float(np.linalg.norm(b)); yn = float(np.linalg.norm(y))
    ph = np.einsum('ik,jk->ij', np.einsum('ij,jk->ik', h, q, optimize=False), q, optimize=False); k = h - ph; kn = np.linalg.norm(k, axis=1); hn = np.linalg.norm(h, axis=1)
    reduction = gamma(768 + 2 * q.shape[1] + 8)
    da = pb * yn + reduction * float(np.linalg.norm((np.abs(y) @ np.abs(q)) @ np.abs(q).T))
    db = da + gamma(2) * (yn + float(np.linalg.norm(a)))
    dr = db + gamma(770) * radius
    dk = pb * hn + reduction * np.linalg.norm((np.abs(h) @ np.abs(q)) @ np.abs(q).T, axis=1) + gamma(2) * (hn + np.linalg.norm(ph, axis=1))
    dkn = dk + gamma(770) * kn
    c = np.einsum('ij,j->i', h, a, optimize=False); dc = hn * da + gamma(769) * (np.abs(h) @ np.abs(a))
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
    alignment = np.einsum('ij,j->i', n, b, optimize=False); dal = dn * np.linalg.norm(b) + db * np.linalg.norm(n, axis=1) + dn * db + gamma(769) * (np.abs(n) @ np.abs(b))
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
