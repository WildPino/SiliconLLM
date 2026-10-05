"""Decoded WI response and one full operator spectrum; no candidate export."""
import math
import numpy as np

D, F = 768, 3072
THRESHOLD_SQUARED = 0.05**2
DECISION_MARGIN = 1e-6
BASE_BYTES, RANK_BYTES, ORIGINAL_BYTES = 352202816, 15360, 605945856
TOTAL_RANK_CAP = (7 * ORIGINAL_BYTES - 10 * BASE_BYTES) // (10 * RANK_BYTES)


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def quant(x):
    assert x.dtype == np.dtype('<f4') and x.ndim == 2 and np.isfinite(x).all()
    maximum = np.max(np.abs(x), axis=1)
    with np.errstate(under='ignore'):
        alpha = (maximum / np.float32(32767)).astype('<f4')
    alpha[maximum == 0] = 1
    assert np.all(alpha > 0) and np.isfinite(alpha).all()
    return np.clip(np.rint(np.divide(x, alpha[:, None], dtype=np.float32)),
                   -32767, 32767).astype('<i2'), alpha


def response(q, alpha, wi, scales):
    assert q.dtype == np.dtype('<i2') and alpha.dtype == scales.dtype == np.dtype('<f4')
    assert wi.dtype == np.dtype('<i1') and q.shape[1] == wi.shape[1] <= 4096
    assert np.all(q != -32768) and np.all(alpha > 0) and np.all(scales >= 0)
    width = q.shape[1]
    assert width * 128 * 32767 < 2**53
    w = wi.astype(np.float64)
    codes = q.astype(np.float64)
    integer_dots = codes @ w.T
    assert np.array_equal(integer_dots, np.rint(integer_dots))
    native64 = (integer_dots * scales.astype(np.float64)) * alpha.astype(np.float64)[:, None]
    decoded = (codes * alpha.astype(np.float64)[:, None]) @ (w * scales.astype(np.float64)[:, None]).T
    magnitude = ((np.abs(codes) @ np.abs(w).T) * scales.astype(np.float64)) * alpha.astype(np.float64)[:, None]
    u = np.finfo(np.float64).eps / 2
    gamma = (width + 2) * u / (1 - (width + 2) * u)
    bound = 4 * gamma * magnitude
    difference = np.abs(decoded - native64)
    assert np.isfinite(decoded).all() and np.isfinite(native64).all()
    assert np.all(difference <= bound), 'decoded/native64 forward rounding envelope'
    native32 = native64.astype('<f4')
    assert np.isfinite(native32).all()
    active = magnitude > 0
    return decoded, native32, {
        'max_absolute_native64_difference': float(np.max(difference)),
        'max_difference_over_absolute_product_sum': float(np.max(difference[active] / magnitude[active])) if np.any(active) else 0.,
        'forward_envelope_coefficient': float(4 * gamma),
        'decoded_vs_nativeF32_error_energy': float(np.sum((decoded - native32.astype(np.float64))**2)),
        'nativeF32_reference_energy': float(np.sum(native32.astype(np.float64)**2))
    }


def representatives(queries, expert):
    indices = np.flatnonzero((queries['role'] == 0) & (queries['expert'] == expert))
    first = {}
    for i in indices:
        first.setdefault((int(queries['book'][i]), queries['codes'][i].tobytes(),
                          queries['alpha'][i].tobytes()), int(i))
    ids = np.asarray(sorted(first.values()), dtype='<u4')
    books = queries['book'][ids]
    distinct, counts = np.unique(books, return_counts=True)
    sizes = dict(zip(map(int, distinct), map(int, counts)))
    weights = np.asarray([1 / (len(distinct) * sizes[int(book)]) for book in books], dtype='<f8')
    assert len(ids) and abs(float(weights.sum()) - 1) <= 1e-12
    for book in distinct:
        assert abs(float(weights[books == book].sum()) - 1 / len(distinct)) <= 1e-12
    return ids, weights


def spectrum(y, guard):
    assert y.dtype == np.float64 and y.ndim == 2 and np.isfinite(y).all()
    energy = float(np.sum(y * y))
    u, s, vt = np.linalg.svd(y, full_matrices=False)
    guard()
    assert np.isfinite(s).all() and np.all(s >= 0) and np.all(s[:-1] >= s[1:])
    for axis in range(len(s)):
        if vt[axis, int(np.argmax(np.abs(vt[axis])))] < 0:
            vt[axis] *= -1
            u[:, axis] *= -1
    scale = math.sqrt(energy) if energy else 1.
    reconstruction = float(np.linalg.norm((u * s) @ vt - y) / scale)
    orthogonal = float(np.linalg.norm(vt @ vt.T - np.eye(len(s))))
    gram = y.T @ y
    eigen = float(np.linalg.norm(gram @ vt.T - vt.T * (s * s)) / (energy if energy else 1.))
    balance = abs(float(np.sum(s * s)) - energy) / (energy if energy else 1.)
    assert reconstruction <= 1e-10 and orthogonal <= 1e-10 * math.sqrt(len(s))
    assert eigen <= 1e-10 and balance <= 1e-10
    tail = np.concatenate((np.cumsum((s * s)[::-1])[::-1], np.zeros(1)))
    relative = tail / energy if energy else np.zeros_like(tail)
    margin = DECISION_MARGIN if energy else 0.
    lower, upper = np.maximum(relative - margin, 0), relative + margin
    nominal = int(np.flatnonzero(relative <= THRESHOLD_SQUARED)[0])
    required_low = int(np.flatnonzero(lower <= THRESHOLD_SQUARED)[0])
    required_high = int(np.flatnonzero(upper <= THRESHOLD_SQUARED)[0])
    def status(rank):
        j = min(rank, len(s))
        return 'PASS' if upper[j] <= THRESHOLD_SQUARED else ('FAIL' if lower[j] > THRESHOLD_SQUARED else 'INDETERMINATE')
    return s, vt, relative, {
        'energy': energy, 'reconstruction_relative': reconstruction,
        'orthogonal_residual': orthogonal, 'eigen_relative': eigen,
        'energy_relative_discrepancy': balance, 'zero_response': energy == 0,
        'required_rank_nominal': nominal, 'required_rank_lower': required_low,
        'required_rank_upper': required_high, 'rank32_status': status(32),
        'rank43_status': status(43), 'rank32_RMS': math.sqrt(float(relative[min(32, len(s))])),
        'rank43_RMS': math.sqrt(float(relative[min(43, len(s))])),
        'decision_guard_band_squared_relative': margin
    }


def controls(guard):
    wi = np.array([[-128, 127, 0], [1, -3, 2]], dtype='<i1')
    q = np.array([[32767, -32767, 1], [0, 0, 0]], dtype='<i2')
    alpha = np.array([.5, 1], dtype='<f4')
    scales = np.array([.25, 2], dtype='<f4')
    _, native, _ = response(q, alpha, wi, scales)
    expected = (((q.astype(np.int64) @ wi.astype(np.int64).T).astype(np.float64) * scales) * alpha[:, None]).astype('<f4')
    exact(native, expected)
    y = np.diag([8., 1., 0.])
    s, _, relative, stats = spectrum(y, guard)
    assert np.allclose(s, [8, 1, 0], rtol=0, atol=1e-12)
    assert abs(relative[1] - 1 / 65) < 1e-12 and stats['required_rank_nominal'] == 2
    assert spectrum(np.zeros((4, 3)), guard)[3]['required_rank_upper'] == 0
    assert TOTAL_RANK_CAP == 4684
    assert 10 * (BASE_BYTES + 107 * 43 * RANK_BYTES) <= 7 * ORIGINAL_BYTES
    assert 10 * (BASE_BYTES + 107 * 44 * RANK_BYTES) > 7 * ORIGINAL_BYTES
    return {'exact_integer_and_native_order': True, 'known_spectrum_and_zero_response': True,
            'uniform_and_variable_storage_arithmetic': True}
