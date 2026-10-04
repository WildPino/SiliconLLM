"""Private matrix factors with coupled ReLU neuron permutations, no gradients."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


def assignment(reference_wi, target_wi):
    """Minimum unit-WI row distance, not minimum function/output distance."""
    def unit(a):
        norm = np.linalg.norm(a, axis=1)
        return a/np.where(norm > 0, norm, 1)[:, None]
    ref, target = unit(reference_wi), unit(target_wi)
    cost = np.sum(ref**2, axis=1)[:, None]+np.sum(target**2, axis=1)[None, :]-2*(ref @ target.T)
    assert np.isfinite(cost).all() and np.min(cost) >= -1e-10
    row, permutation = linear_sum_assignment(cost)
    assert np.array_equal(row, np.arange(len(row))) and np.array_equal(np.sort(permutation), row)
    identity = float(np.trace(cost))
    matched = float(np.sum(cost[row, permutation]))
    assert matched <= identity+1e-10*max(identity, 1)
    return permutation, {'identity_cost': identity, 'matched_cost': matched,
                         'identity_matches': int(np.sum(permutation == row))}


def spectrum(matrix):
    """Exact smaller-Gram eigensystem and balanced F32 private factors."""
    tall = matrix.shape[0] >= matrix.shape[1]
    gram = matrix.T @ matrix if tall else matrix @ matrix.T
    values, axes = np.linalg.eigh(gram)
    values, axes = values[::-1], axes[:, ::-1].copy()
    for j in range(len(values)):
        if axes[int(np.argmax(np.abs(axes[:, j]))), j] < 0:
            axes[:, j] *= -1
    energy = float(np.sum(matrix**2))
    assert energy > 0 and np.isfinite(values).all()
    assert np.min(values) >= -1e-10*max(energy, 1)
    assert np.linalg.norm(axes.T @ axes-np.eye(len(values))) <= 1e-10*np.sqrt(len(values))
    assert np.linalg.norm(gram @ axes-axes*values)/max(np.linalg.norm(gram), 1e-12) <= 1e-10
    assert abs(np.sum(values)-energy) <= 1e-10*max(energy, 1)
    return values, axes, energy, tall


def factors(matrix, spec, rank):
    values, axes, energy, tall = spec
    assert rank > 0 and np.min(values[:rank]) > 0
    root_singular = values[:rank]**.25
    direction = axes[:, :rank]
    if tall:
        u64 = (matrix @ direction)/root_singular[None, :]
        v64 = root_singular[:, None]*direction.T
    else:
        u64 = direction*root_singular[None, :]
        v64 = (direction.T @ matrix)/root_singular[:, None]
    residual = matrix-u64 @ v64
    tail = float(np.sum(residual**2))
    expected = energy-float(np.sum(values[:rank]))
    assert abs(tail-expected) <= 1e-10*energy
    u, v = u64.astype(np.float32), v64.astype(np.float32)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    stored_tail = float(np.sum((matrix-u.astype(np.float64) @ v.astype(np.float64))**2))
    return u, v, {'F64_optimal_Frobenius_energy_retained': 1-tail/energy,
                  'stored_F32_Frobenius_energy_retained': 1-stored_tail/energy,
                  'matrix_energy': energy, 'factor_array_bytes': u.nbytes+v.nbytes}


def project(x, pair):
    u, v = pair
    return (x @ v.astype(np.float64).T) @ u.astype(np.float64).T


def forward(x, wi, wo):
    return np.maximum(x @ wi.T, 0) @ wo.T


def factor_forward(x, wi_pair, wo_pair, base_wi=None, base_wo=None):
    up = project(x, wi_pair)
    if base_wi is not None:
        up += x @ base_wi.T
    activation = np.maximum(up, 0)
    down = project(activation, wo_pair)
    if base_wo is not None:
        down += activation @ base_wo.T
    return down


def relative_rows(value, reference):
    assert value.shape == reference.shape and np.isfinite(value).all() and np.isfinite(reference).all()
    return np.linalg.norm(value-reference, axis=1)/np.maximum(np.linalg.norm(reference, axis=1), 1e-12)


def tiny_qualification():
    cost = np.array([[3., 1., 4.], [4., 3., 1.], [1., 4., 3.]])
    row, col = linear_sum_assignment(cost)
    actual = float(np.sum(cost[row, col]))
    expected = min(sum(cost[i, p[i]] for i in range(3)) for p in itertools.permutations(range(3)))
    assert actual == expected == 3 and np.array_equal(col, [1, 2, 0])
    matrix = np.arange(77, dtype=np.float64).reshape(11, 7)/31+np.eye(11, 7)
    for a in (matrix, matrix.T):
        s = spectrum(a)
        u, v, stats = factors(a, s, 7)
        assert stats['F64_optimal_Frobenius_energy_retained'] >= 1-1e-10
        assert np.linalg.norm(u.astype(np.float64) @ v.astype(np.float64)-a)/np.linalg.norm(a) <= 1e-6
    return {'assignment_exhaustive3_factor_orientation_both_fullrank': True}
