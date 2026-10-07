"""Development-only complement geometry, positive dyadic minors and receipts."""
import hashlib
import math
import numpy as np

P = 2147483647


def alpha_mod(bits):
    b = int(bits); exponent = (b >> 23) & 255; fraction = b & 0x7fffff
    assert b >> 31 == 0 and exponent != 255 and (exponent or fraction)
    mantissa = fraction if exponent == 0 else fraction + (1 << 23)
    power = -149 if exponent == 0 else exponent - 150
    return mantissa * (pow(2, power, P) if power >= 0 else pow((P + 1) // 2, -power, P)) % P


def field(q, bits):
    a = np.array([alpha_mod(b) for b in bits], '<i8')
    h = np.ones((len(q), q.shape[1] + 1), '<i8')
    h[:, :-1] = (q.astype('<i8') * a[:, None]) % P
    return h


def minor(a, guard):
    assert a.shape[0] == a.shape[1] and np.all((a >= 0) & (a < P))
    digest = hashlib.sha256(a.tobytes()).hexdigest()
    a = a.copy(); determinant = 1; pivots = []; swaps = []
    for col in range(len(a)):
        available = np.flatnonzero(a[col:, col])
        if not len(available):
            return {'field_SHA256': digest, 'dimension': len(a), 'determinant_mod_prime': 0,
                    'positive_real_full_rank_certificate': False, 'pivots': pivots, 'swaps': swaps}
        row = col + int(available[0])
        if row != col:
            a[[col, row]] = a[[row, col]]; determinant = -determinant % P; swaps.append([col, row])
        pivot = int(a[col, col]); pivots.append(pivot); determinant = determinant * pivot % P
        a[col, col:] = a[col, col:] * pow(pivot, -1, P) % P
        if col + 1 < len(a):
            a[col + 1:, col:] = (a[col + 1:, col:] - a[col + 1:, col, None] * a[col, None, col:]) % P
        if col % 8 == 0: guard()
    return {'field_SHA256': digest, 'dimension': len(a), 'determinant_mod_prime': determinant,
            'positive_real_full_rank_certificate': determinant != 0, 'pivots': pivots, 'swaps': swaps}


def envelope(a, x, rhs):
    error = a @ x - rhs
    bound = 5e-12 * np.maximum(1, np.abs(a) @ np.abs(x) + np.abs(rhs))
    ratio = float(np.max(np.abs(error) / bound))
    assert np.isfinite(ratio) and ratio <= 1
    return ratio


def greedy(z, uids, base, forbidden, guard):
    """Max raw whitened innovation; ascending UID resolves exact F64 ties."""
    rows, width = z.shape; original = len(base)
    assert np.all(uids[:-1] < uids[1:]) and original <= width
    Q = np.empty((width, width), '<f8')
    residual = z.copy()
    if original:
        q, r = np.linalg.qr(base.T, mode='reduced'); Q[:, :original] = q
        assert np.max(np.abs(q.T @ q - np.eye(original))) <= 5e-11
        for _ in range(2): residual -= (residual @ q) @ q.T
    available = ~np.isin(uids, forbidden)
    selected = []; innovation = []; relative = []; maxima = []
    norm = np.sum(z * z, axis=1)
    for rank in range(original, width):
        scores = np.sum(residual * residual, axis=1)
        scores[~available] = -1
        row = int(np.argmax(scores)); maximum = float(scores[row])
        assert maximum > 0 and np.isfinite(maximum)
        vector = residual[row].copy()
        for _ in range(2): vector -= Q[:, :rank] @ (Q[:, :rank].T @ vector)
        value = float(vector @ vector); assert value > 0 and abs(value - maximum) <= 1e-9 * max(1, maximum)
        unit = vector / math.sqrt(value); Q[:, rank] = unit
        selected.append(row); innovation.append(value); relative.append(math.sqrt(value / float(norm[row]))); maxima.append(maximum)
        for _ in range(2): residual -= (residual @ unit)[:, None] * unit[None, :]
        available[row] = False
        if rank % 8 == 0: guard()
    assert np.max(np.abs(Q.T @ Q - np.eye(width))) <= 5e-11
    return np.array(selected, '<u4'), np.array(innovation, '<f8'), np.array(relative, '<f8'), np.array(maxima, '<f8')


def stability(z):
    q, r = np.linalg.qr(z.T)
    ratio = envelope(q, r, z.T)
    orth = float(np.max(np.abs(q.T @ q - np.eye(z.shape[1]))))
    assert orth <= 5e-11
    singular = np.linalg.svd(z, compute_uv=False)
    assert np.isfinite(singular).all() and np.all(singular > 0)
    condition = float(singular[0] / singular[-1])
    return {'QR_envelope_ratio': ratio, 'QR_orthogonality': orth,
            'condition2': condition, 'sigma_max': float(singular[0]), 'sigma_min': float(singular[-1])}, singular


def controls():
    assert all(P % d for d in range(2, math.isqrt(P) + 1)) and (P - 1) ** 2 < 2 ** 62
    q = np.array([[1, 0], [0, 1], [-1, 1]], '<i2')
    bits = np.array([0x3e800000, 0x3fc00000, 0x3f000000], '<u4')
    cert = minor(field(q, bits), lambda: None)
    assert cert['determinant_mod_prime'] == 1
    duplicate = minor(np.repeat(field(q, bits)[:1], 3, axis=0), lambda: None)
    assert not duplicate['positive_real_full_rank_certificate']
    z = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [0., 0., 3.]], '<f8')
    at, inv, rel, score = greedy(z, np.arange(4), np.empty((0, 3)), np.array([], '<u4'), lambda: None)
    assert at.tolist() == [2, 1, 0] and inv.tolist() == [9., 4., 1.] and rel.tolist() == [1., 1., 1.]
    own = z[[2]]; at2, inv2, _, _ = greedy(z, np.arange(4), own, np.array([2]), lambda: None)
    assert at2.tolist() == [1, 0] and inv2.tolist() == [4., 1.]
    return {'prime': P, 'new_rational_minor': cert, 'duplicate_minor': duplicate,
            'new_alpha_bits': bits.tolist(), 'new_alpha_mod': [alpha_mod(v) for v in bits],
            'global_tie_order': at.tolist(), 'extension_order': at2.tolist()}
