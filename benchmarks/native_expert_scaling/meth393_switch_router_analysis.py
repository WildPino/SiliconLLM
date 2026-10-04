"""Oracle retained mass of full native router scores; diagnostic only."""
import struct
from pathlib import Path
import numpy as np
import meth392_switch_route_audit as R


def analyze(trace, output, n, s, t):
    data = Path(trace).read_bytes()
    assert data[:8] == b'SWRTA001' and struct.unpack_from('<2I', data, 8) == (n, 768)
    dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (n,))])
    count = 6 * (s + t)
    assert len(data) == 16 + count * dtype.itemsize
    rows = np.frombuffer(data, dtype=dtype, offset=16)
    assert np.array_equal(rows['index'], np.arange(count))
    assert np.all(rows['phase'][:6*s] == 0) and np.all(rows['phase'][6*s:] == 1)
    assert np.all(np.isfinite(rows['input'])) and np.all(np.isfinite(rows['scores']))
    encoder, decoder = R.route_bytes(Path(output).read_bytes(), n, s, t)
    routes = np.concatenate([encoder.reshape(-1), decoder.T.reshape(-1)])
    selected = np.argmax(rows['scores'], axis=1)
    assert np.array_equal(selected, routes['expert'])
    # Match original F32 subtraction, F64 exp rounded to F32, F64 sum.
    differences = rows['scores'] - rows['scores'][np.arange(count), selected, None]
    exponential = np.exp(differences.astype(np.float64)).astype(np.float32).astype(np.float64)
    total = exponential.sum(axis=1)
    probability = (1. / total).astype(np.float32)
    maximum_relative = float(np.max(np.abs(probability.astype(np.float64) / routes['probability'] - 1.)))
    assert maximum_relative <= 1e-6, maximum_relative
    mass = np.cumsum(np.sort(exponential, axis=1)[:, ::-1], axis=1) / total[:, None]
    minimum = np.argmax(mass >= 1. / 1.01, axis=1) + 1
    grid = sorted(set([1, 2, 4, 8, 16, 32, 64, 128, n]))
    multipliers = 1. / mass[:, np.asarray(grid) - 1]
    assert np.all(multipliers >= 1. - 2e-14) and np.all(multipliers[:, -1] <= 1. + 2e-14)
    assert np.all(np.diff(multipliers, axis=1) <= 2e-14)
    return {'n': n, 'routes': count, 'maximum_selected_probability_relative_error': maximum_relative, 'grid': grid}, minimum, multipliers


def summarize(parts):
    assert parts
    minimum = np.concatenate([v[1] for v in parts]); multipliers = np.concatenate([v[2] for v in parts])
    grid = parts[0][0]['grid']; assert all(v[0]['grid'] == grid for v in parts)
    quantiles = [0., .05, .5, .95, 1.]
    return {'route_queries': len(minimum), 'quantiles': quantiles,
            'minimum_oracle_candidates_for_1percent_scale_quantiles': np.quantile(minimum, quantiles).tolist(),
            'maximum_selected_probability_relative_error': max(v[0]['maximum_selected_probability_relative_error'] for v in parts),
            'oracle_shortlists': [{'candidates': k, 'scale_multiplier_quantiles': np.quantile(multipliers[:, j], quantiles).tolist(),
                                  'relative_scale_change_gt1percent_count': int(np.sum(multipliers[:, j] > 1.01)),
                                  'relative_scale_change_gt5percent_count': int(np.sum(multipliers[:, j] > 1.05))} for j, k in enumerate(grid)]}
