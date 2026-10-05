"""Lossless signed nibble planes and exact integer negative-ReLU certificate."""
import numpy as np

D, F, E, N = 768, 3072, 128, 1344
MAX_T = 15 * 32767 * D
MAX_C = 8 * MAX_T
MAX_R2, MAX_Q2 = 64 * D, 32767**2 * D
assert MAX_C**2 < 2**63 and MAX_R2 * MAX_Q2 < 2**63

def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def pack(values):
    assert values.dtype == np.dtype('i1') and values.ndim == 2 and values.shape[1] % 2 == 0
    assert np.all((values >= -8) & (values <= 7))
    u = values.astype(np.int16) & 15
    return (u[:, ::2] | (u[:, 1::2] << 4)).astype('u1')

def unpack(packed):
    assert packed.dtype == np.dtype('u1') and packed.ndim == 2
    u = np.empty((len(packed), packed.shape[1] * 2), np.int16)
    u[:, ::2] = packed & 15; u[:, 1::2] = packed >> 4
    return np.where(u >= 8, u - 16, u).astype('i1')

def planes(w):
    assert w.dtype == np.dtype('i1') and w.ndim == 2
    v = w.astype(np.int16)
    h = (v // 16).astype('i1')
    b = (v - 16 * h.astype(np.int16) - 8).astype('i1')
    assert np.all(16 * h.astype(np.int16) + 8 + b.astype(np.int16) == v)
    hp, bp = pack(h), pack(b)
    exact(unpack(hp), h); exact(unpack(bp), b)
    r2 = np.sum(b.astype(np.int16)**2, axis=1, dtype=np.uint32).astype('<u4')
    assert np.all(r2 <= MAX_R2)
    return hp, bp, r2

def integer_dot(q, w):
    assert q.dtype == np.dtype('<i2') and w.dtype == np.dtype('i1') and q.shape[1] == w.shape[1]
    assert np.all(q != -32768) and q.shape[1] * 128 * 32767 < 2**53
    # Products, every partial absolute sum, and final sums are exactly F64 integers.
    value = q.astype(np.float64) @ w.astype(np.float64).T
    assert np.isfinite(value).all() and np.array_equal(value, np.rint(value))
    return value.astype('<i8')

def quant(x):
    assert x.ndim == 2 and x.dtype == np.dtype('<f4') and np.isfinite(x).all()
    maximum = np.max(np.abs(x), axis=1)
    alpha = (maximum / np.float32(32767)).astype('<f4')
    alpha[maximum == 0] = 1
    assert np.all(alpha > 0) and np.isfinite(alpha).all()
    q = np.clip(np.rint(np.divide(x, alpha[:, None], dtype=np.float32)), -32767, 32767).astype('<i2')
    return q, alpha

def scaled(dots, alpha, scales):
    assert dots.dtype == np.dtype('<i8') and alpha.dtype == scales.dtype == np.dtype('<f4')
    assert np.all(alpha > 0) and np.all(scales > 0)
    value = ((dots.astype(np.float64) * scales.astype(np.float64)) * alpha.astype(np.float64)[:, None]).astype('<f4')
    assert np.isfinite(value).all()
    return value

def certificate(t, r2, q2):
    assert t.dtype == np.dtype('<i4') and r2.dtype == np.dtype('<u4') and q2.dtype == np.dtype('<u8')
    c = 8 * t.astype(np.int64)
    assert np.all(np.abs(c) <= MAX_C) and np.all(r2 <= MAX_R2) and np.all(q2 <= MAX_Q2)
    bound = q2.astype(np.int64)[:, None] * r2.astype(np.int64)[None, :]
    margin = c * c - bound
    return (c < 0) & (margin > 0)

def candidate(q, alpha, hp, bp, r2, scales, guard):
    assert q.shape[1] == D and len(hp) == F
    h = unpack(hp)
    q64 = q.astype(np.int64)
    q2 = np.sum(q64 * q64, axis=1, dtype=np.int64).astype('<u8')
    t64 = 2 * integer_dot(q, h) + np.sum(q64, axis=1)[:, None]
    assert np.all(np.abs(t64) <= MAX_T)
    t = t64.astype('<i4')
    cert = certificate(t, r2, q2)
    raw = np.zeros((len(q), F), '<f4')
    for i in range(len(q)):
        selected = np.flatnonzero(~cert[i])
        # Only the selected packed fine rows are read by the candidate path.
        b = unpack(bp[selected])
        z = 8 * t[i, selected].astype(np.int64) + integer_dot(q[i:i + 1], b)[0]
        raw[i, selected] = scaled(z[None, :], alpha[i:i + 1], scales[selected])[0]
        guard()
    up = np.where(raw < 0, np.float32(0), raw).astype('<f4')
    hidden, hidden_alpha = quant(up)
    return t, q2, cert, raw, hidden, hidden_alpha

def sparse_down(codes, alpha, wo, scales, guard):
    assert codes.shape[1] == F and wo.shape == (D, F)
    outputs = np.empty((len(codes), D), '<f4')
    counts = []
    for i in range(len(codes)):
        ids = np.flatnonzero(codes[i])
        w = np.ascontiguousarray(wo[:, ids])
        dots = integer_dot(np.ascontiguousarray(codes[i:i + 1, ids]), w)
        outputs[i] = scaled(dots, alpha[i:i + 1], scales)[0]
        counts.append(len(ids)); guard()
    return outputs, counts

def controls(guard):
    values = np.arange(-128, 128, dtype=np.int16).astype('i1')[None, :]
    hp, bp, _ = planes(values)
    h, b = unpack(hp), unpack(bp)
    assert len({(int(x), int(y)) for x, y in zip(h[0], b[0])}) == 256
    assert (16 * h.astype(np.int16) + 8 + b.astype(np.int16)).astype('i1').tobytes() == values.tobytes()
    w = np.stack([np.full(D, -128, 'i1'), np.full(D, 127, 'i1'), np.zeros(D, 'i1'),
                  np.where(np.arange(D) % 2, -128, 127).astype('i1')])
    q = np.stack([np.full(D, 32767, '<i2'), np.full(D, -32767, '<i2'),
                  np.zeros(D, '<i2'), np.where(np.arange(D) % 2, -32767, 32767).astype('<i2')])
    hp, bp, r2 = planes(w); h, b = unpack(hp), unpack(bp)
    z = integer_dot(q, w)
    t = (2 * integer_dot(q, h) + np.sum(q.astype(np.int64), axis=1)[:, None]).astype('<i4')
    q2 = np.sum(q.astype(np.int64)**2, axis=1, dtype=np.int64).astype('<u8')
    cert = certificate(t, r2, q2)
    for i in range(len(q)):
        for r in range(len(w)):
            direct = sum(int(a) * int(x) for a, x in zip(w[r], q[i]))
            coarse = sum((16 * int(a) + 8) * int(x) for a, x in zip(h[r], q[i]))
            residual = sum(int(a) * int(x) for a, x in zip(b[r], q[i]))
            norm = sum(int(a)**2 for a in b[r]); qnorm = sum(int(x)**2 for x in q[i])
            assert direct == z[i, r] == coarse + residual and coarse == 8 * int(t[i, r])
            assert norm == int(r2[r]) and qnorm == int(q2[i]) and residual**2 <= norm * qnorm
            assert bool(cert[i, r]) == (coarse < 0 and coarse**2 > norm * qnorm)
            assert not cert[i, r] or direct < 0
    tie = certificate(np.array([[-1, -2, 1, 0]], '<i4'), np.array([1, 1, 1, 0], '<u4'), np.array([64], '<u8'))
    assert tie.tolist() == [[False, True, False, False]]
    assert certificate(np.array([[-1]], '<i4'), np.array([0], '<u4'), np.array([0], '<u8'))[0, 0]
    cast = np.asarray([1 + 2.**-24, 1 + 3 * 2.**-24, 2.**-149, 2.**-150, -2.**-150], '<f8').astype('<f4')
    assert cast.view('<u4').tolist() == [0x3f800000, 0x3f800002, 1, 0, 0x80000000]
    zeros, za = quant(np.array([[0., -0.]], '<f4'))
    assert np.all(zeros == 0) and za.tolist() == [1.]
    scales = np.array([.5, 2., .25, 1.], '<f4')
    for a in (np.full(len(q), .25, '<f4'), np.ones(len(q), '<f4')):
        ordinary = scaled(z, a, scales)
        reconstructed = scaled(8 * t.astype(np.int64) + integer_dot(q, b), a, scales)
        exact(ordinary, reconstructed)
    guard()
    return {'all256_coefficient_pairs_signed_nibble_pack_inverse': True,
            'signed_extrema_full_scalar_I64_dot_Cauchy_norm_certificate': True,
            'strict_equality_positive_zero_norm_certificate_edges': True,
            'dyadic_scale_F32_cast_ties_subnormal_signedzero_hidden_zero': True,
            'integer_comparison_overflow_bounds': True}
