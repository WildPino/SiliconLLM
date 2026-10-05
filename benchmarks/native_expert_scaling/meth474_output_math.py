"""One private output span from source functions; qualified projection envelopes."""
import math
import numpy as np

RANK = 32
NUMERICAL_DISTANCE_GUARD = 1e-8
NOMINAL_BANK_BYTES, ORIGINAL_BANK_BYTES = 405781568, 605945856


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def representatives(q, indices):
    first = {}
    for i in indices:
        first.setdefault((int(q['book'][i]), q['codes'][i].tobytes(), q['alpha'][i].tobytes()), int(i))
    ids = np.asarray(sorted(first.values()), dtype='<u4')
    books = q['book'][ids]
    distinct, counts = np.unique(books, return_counts=True)
    sizes = dict(zip(map(int, distinct), map(int, counts)))
    weights = np.asarray([1 / (len(distinct) * sizes[int(book)]) for book in books], dtype='<f8')
    assert len(ids) and abs(float(weights.sum()) - 1) <= 1e-12
    assert all(abs(float(weights[books == b].sum()) - 1 / len(distinct)) <= 1e-12 for b in distinct)
    return ids, weights


def basis(y, guard, rank=RANK):
    assert y.dtype == np.float64 and np.isfinite(y).all() and min(y.shape) >= rank
    energy = float(np.sum(y * y))
    u, s, vt = np.linalg.svd(y, full_matrices=False)
    guard()
    assert np.isfinite(s).all() and np.all(s >= 0) and np.all(s[:-1] >= s[1:])
    # Fixed policy includes returned numerical-null axes, with no validation choice.
    for j in range(len(s)):
        if u[int(np.argmax(np.abs(u[:, j]))), j] < 0:
            u[:, j] *= -1
            vt[j] *= -1
    denom = energy if energy else 1.
    reconstruction = float(np.linalg.norm((u * s) @ vt - y) / math.sqrt(denom))
    right_orth = float(np.linalg.norm(vt @ vt.T - np.eye(len(s))))
    left_orth = float(np.linalg.norm(u.T @ u - np.eye(len(s))))
    eigen = float(np.linalg.norm(y.T @ (y @ vt.T) - vt.T * (s * s)) / denom)
    balance = abs(float(np.sum(s * s)) - energy) / denom
    assert reconstruction <= 1e-10 and eigen <= 1e-10 and balance <= 1e-10
    assert max(right_orth, left_orth) <= 1e-10 * math.sqrt(len(s))
    top = u[:, :rank].copy()
    stored = top.astype('<f4')
    gram = stored.astype(np.float64).T @ stored.astype(np.float64)
    gram_orth = float(np.linalg.norm(gram - np.eye(rank)))
    assert gram_orth <= 1e-6 and float(np.sum(stored.astype(np.float64)**2)) < rank + 1
    orth, triangular = np.linalg.qr(stored.astype(np.float64), mode='reduced')
    qr_residual = float(np.linalg.norm(orth @ triangular - stored.astype(np.float64)))
    qr_orth = float(np.linalg.norm(orth.T @ orth - np.eye(rank)))
    assert qr_residual <= 1e-12 and qr_orth <= 1e-12
    tolerance = float(s[0] * max(y.shape) * np.finfo(np.float64).eps) if len(s) else 0.
    return top, stored, orth, triangular, s, vt, {
        'energy': energy, 'zero_response': energy == 0, 'reconstruction_relative': reconstruction,
        'left_orthogonal_residual': left_orth, 'right_orthogonal_residual': right_orth,
        'eigen_relative': eigen, 'energy_relative_discrepancy': balance,
        'stored_F32_Gram_orthogonal_residual': gram_orth, 'stored_F32_squared_Frobenius': float(np.sum(stored.astype(np.float64)**2)),
        'QR_reconstruction_absolute': qr_residual, 'QR_orthogonal_residual': qr_orth,
        'numerical_rank_tolerance': tolerance, 'numerical_rank': int(np.count_nonzero(s > tolerance)),
        'included_numerical_null_axes': max(0, rank - int(np.count_nonzero(s > tolerance)))
    }


def rounding_constants(dimension=768, rank=RANK):
    u64, u32 = 2.**-53, 2.**-24
    gamma = (2 * rank * u64) / (1 - 2 * rank * u64)
    eta = gamma * math.sqrt(rank + 1) / .99999
    relative = eta + u32 * (1 + eta)
    # Allows either IEEE gradual underflow or flushing at the FINAL F32 cast.
    absolute = math.sqrt(dimension) * 2.**-126
    return relative, absolute


def metrics(functions, orth, rank=RANK):
    f = functions.astype(np.float64)
    assert f.ndim == 2 and np.isfinite(f).all()
    reference = np.sum(f * f, axis=1)
    norm = np.sqrt(reference)
    assert np.all(norm < np.finfo(np.float32).max / 8)
    projection = (f @ orth) @ orth.T
    distance = np.linalg.norm(f - projection, axis=1)
    numerical = NUMERICAL_DISTANCE_GUARD * norm
    relative, absolute = rounding_constants(f.shape[1], rank)
    lower = np.maximum((1 - relative) * np.maximum(distance - numerical, 0) - relative * norm - absolute, 0)
    upper = distance + numerical + relative * norm + absolute
    lower[reference == 0] = 0
    upper[reference == 0] = 0  # exact coordinate0 constructs exact output0
    answer = np.column_stack((reference, distance * distance, lower * lower, upper * upper)).astype('<f8')
    assert np.isfinite(answer).all() and np.all(answer >= 0)
    return answer


def summary(values, threshold, weights=None):
    assert values.ndim == 2 and values.shape[1] == 4 and len(values)
    energies = np.sum(values if weights is None else values * weights[:, None], axis=0)
    reference = float(energies[0])
    rms = [math.sqrt(float(v) / reference) if reference else (0. if v == 0 else None) for v in energies[1:]]
    status = 'FAIL' if rms[1] is None or rms[1] > threshold else ('PASS' if rms[2] is not None and rms[2] <= threshold else 'INDETERMINATE')
    return {'count': len(values), 'threshold': threshold, 'reference_energy': reference,
            'ideal_projection_error_energy': float(energies[1]), 'conservative_lower_error_energy': float(energies[2]),
            'constructive_upper_error_energy': float(energies[3]), 'ideal_RMS': rms[0],
            'lower_RMS': rms[1], 'upper_RMS': rms[2], 'status': status}


def controls(guard):
    cast = np.asarray([1 + 2.**-24, 1 + 3 * 2.**-24, 2.**-149, 2.**-150,
                       -2.**-150, 2.**-126 - 2.**-150], dtype='<f8').astype('<f4')
    assert cast.view('<u4').tolist() == [0x3f800000, 0x3f800002, 1, 0, 0x80000000, 0x00800000]
    y = np.array([[3., 0, 0, -3], [0, 1., -1., 0], [0, 0, 0, 0]]) / 2
    _, _, orth, _, s, _, stats = basis(y, guard, rank=2)
    assert stats['numerical_rank'] == 2 and np.allclose(s * s, [4.5, .5, 0], atol=1e-12, rtol=0)
    m = metrics(np.array([[1, 2, 4], [0, 0, 0]], dtype='<f4'), orth, rank=2)
    assert abs(m[0, 1] - 16) <= 1e-12 and m[0, 2] <= 16 <= m[0, 3] and np.all(m[1] == 0)
    assert summary(m[:1], .05)['status'] == 'FAIL'
    assert summary(m[1:], .05)['status'] == 'PASS'
    assert basis(np.zeros((4, 3)), guard, rank=2)[-1]['included_numerical_null_axes'] == 2
    assert 10 * NOMINAL_BANK_BYTES <= 7 * ORIGINAL_BANK_BYTES
    return {'six_F32_ties_even_normal_subnormal_signedzero_casts': True,
            'known_output_spectrum_distance_envelopes_zero_and_null_axes': True,
            'complete_nominal_bank_storage_arithmetic': True}
