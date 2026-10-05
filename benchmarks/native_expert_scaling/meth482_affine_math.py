"""Frozen root LP and outward interval witnesses; no source replay."""
from decimal import Decimal, localcontext
import warnings
import numpy as np
from scipy.optimize import linprog, OptimizeWarning

D = 769
NEG, POS = -np.inf, np.inf
OPTIONS = {'presolve': True, 'time_limit': 25., 'dual_feasibility_tolerance': 1e-9,
           'primal_feasibility_tolerance': 1e-9, 'simplex_dual_edge_weight_strategy': 'devex', 'threads': 1}
CERT = np.dtype([('meta', '<u4', (6,)), ('value', '<f8', (8,))])
assert CERT.itemsize == 88

def up(x):
    return np.nextafter(x, POS)

def down(x):
    return np.nextafter(x, NEG)

def sum_upper_nonnegative(values):
    total = 0.
    for v in values:
        assert np.isfinite(v) and v >= 0
        total = float(up(total + float(v)))
    return total

def dot_intervals(x, theta):
    assert x.shape[1] == len(theta) - 1 and np.isfinite(x).all() and np.isfinite(theta).all()
    lower = np.full(len(x), float(theta[-1]), '<f8')
    upper = lower.copy()
    absolute = np.full(len(x), abs(float(theta[-1])), '<f8')
    for j in range(x.shape[1]):
        product = x[:, j].astype('<f8') * float(theta[j])
        assert np.isfinite(product).all()
        lower = down(lower + down(product))
        upper = up(upper + up(product))
        absolute = up(absolute + up(np.abs(product)))
    assert np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower <= upper)
    return lower, upper, absolute

def gamma_upper(n):
    with localcontext() as c:
        c.prec = 100
        u = Decimal(2) ** -53
        exact = n * u / (1 - n * u)
        return float(up(float(exact)))

def physical_intervals(x, theta32):
    assert theta32.dtype == np.dtype('<f4')
    lo, hi, absolute = dot_intervals(x, theta32.astype('<f8'))
    # Products of two finite F32 numbers are exact in F64. Any F64 summation
    # tree with D-1 additions, then one correctly rounded F32 cast; FTZ/DAZ off.
    error64 = up(gamma_upper(len(theta32) - 1) * absolute)
    magnitude = up(np.maximum(np.abs(lo), np.abs(hi)) + error64)
    error32 = up(up((2. ** -24) * magnitude) + 2. ** -150)
    error = up(error64 + error32)
    return lo, hi, down(lo - error), up(hi + error), absolute, error

def uniform_envelope(x):
    B = max(1., float(np.max(np.abs(x))))
    with localcontext() as c:
        c.prec = 100
        u = Decimal(2) ** -24
        u64 = Decimal(2) ** -53
        delta = Decimal(2) ** -150
        d = x.shape[1] + 1
        gamma = (d - 1) * u64 / (1 - (d - 1) * u64)
        coefficient_error = u + d * delta
        exported_norm = 1 + coefficient_error
        exact = Decimal(B) * (coefficient_error + (gamma + u * (1 + gamma)) * exported_norm) + delta
        return float(up(float(exact))), B

def dual_interval(x, y, weights):
    assert len(x) == len(y) == len(weights) and np.all(weights >= 0) and np.isfinite(weights).all()
    lo = np.zeros(x.shape[1] + 1, '<f8')
    hi = lo.copy()
    sum_lo = sum_hi = 0.
    for row, label, weight in zip(x, y, weights):
        h = np.append(row.astype('<f8'), 1.) * int(label)
        product = float(weight) * h
        lo = down(lo + down(product))
        hi = up(hi + up(product))
        sum_lo = float(down(sum_lo + float(weight)))
        sum_hi = float(up(sum_hi + float(weight)))
    if sum_lo <= 0:
        return lo, hi, sum_lo, sum_hi, None
    numerator = float(np.max(np.maximum(np.abs(lo), np.abs(hi))))
    bound = float(up(numerator / sum_lo))
    assert np.isfinite(bound) and bound >= 0
    return lo, hi, sum_lo, sum_hi, bound

def solve(x, y):
    assert np.isfinite(x).all() and np.all(np.isin(y, [-1, 1]))
    n, width = x.shape
    d = width + 1
    h = np.column_stack((x.astype('<f8'), np.ones(n))) * y[:, None]
    matrix = np.zeros((n + 1, 2 * d + 1), '<f8')
    matrix[:n, :d], matrix[:n, d:2*d], matrix[:n, -1] = -h, h, 1
    matrix[n, :2*d] = 1
    rhs = np.zeros(n + 1)
    rhs[-1] = 1
    objective = np.zeros(2 * d + 1)
    objective[-1] = -1
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always', OptimizeWarning)
        result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=(0, None), method='highs-ds', options=OPTIONS)
    warning_text = [str(w.message) for w in messages]
    assert all(isinstance(w.message, OptimizeWarning) and 'Unrecognized options' in str(w.message) and 'threads' in str(w.message) for w in messages)
    assert result.status in (0, 1), (result.status, result.message)
    available = result.x is not None
    raw_x = np.asarray(result.x, '<f8') if available else np.empty(0, '<f8')
    marginal = getattr(result.ineqlin, 'marginals', None)
    raw_dual = np.asarray(marginal, '<f8') if marginal is not None else np.empty(0, '<f8')
    assert np.isfinite(raw_x).all() and np.isfinite(raw_dual).all()
    theta = np.zeros(d, '<f8')
    if available:
        assert len(raw_x) == 2 * d + 1
        theta = raw_x[:d] - raw_x[d:2*d]
        # Conservative rescaling, not a second optimization. Strict sign and
        # physical relative stability are unchanged by an exact positive scale.
        divisor = max(1., 2 * sum_upper_nonnegative(np.abs(theta)))
        theta = theta / divisor
        assert sum_upper_nonnegative(np.abs(theta)) <= 1
    weights = np.zeros(n, '<f8')
    if len(raw_dual):
        assert len(raw_dual) == n + 1
        weights = np.maximum(0., -raw_dual[:n])
        maximum = float(weights.max())
        if maximum:
            weights /= maximum
    return {'status': int(result.status), 'success': bool(result.success), 'message': str(result.message),
            'nit': int(result.nit), 'fun': float(result.fun) if result.fun is not None else None,
            'available': available, 'warnings': warning_text}, raw_x, raw_dual, theta, theta.astype('<f4'), weights

def reports(unique, owner, links, certificate):
    available = certificate['meta'][:, 5] != 0
    y = 2 * certificate['meta'][:, 3].astype('<i4') - 1
    v = certificate['value']
    signed_lo = np.where(y > 0, v[:, 4], -v[:, 5])
    signed_hi = np.where(y > 0, v[:, 5], -v[:, 4])
    def summary(ids):
        n = len(ids)
        eligible = ids[available[ids]]
        result = {'count': n, 'available_candidate_count': len(eligible),
                  'source_ties': int(np.count_nonzero(certificate['meta'][ids, 4])),
                  'strict_physical_sign_sufficient': int(np.count_nonzero(available[ids] & (signed_lo[ids] > 0))),
                  'strict_physical_wrong_sign': int(np.count_nonzero(available[ids] & (signed_hi[ids] < 0))),
                  'physical_sign_not_proved': int(np.count_nonzero(available[ids] & (signed_lo[ids] <= 0)))}
        result.update(min_signed_physical_lower=float(signed_lo[eligible].min()) if len(eligible) else None,
                      max_arithmetic_error_upper=float(v[eligible, 7].max()) if len(eligible) else None)
        return result
    result = {key: [] for key in ('unique_views', 'occurrence_views', 'book_views', 'source_ID_views')}
    for bank in [-1, *range(12)]:
        eligible = np.ones(len(unique), bool) if bank < 0 else unique['meta'][:, 2] == bank
        for role, mask in (('ALL', np.ones(len(unique), bool)), ('development', (owner[:, 2] & 5) != 0), ('consumed_validation', (owner[:, 2] & 2) != 0)):
            result['unique_views'].append({'bank': bank, 'role': role, **summary(np.flatnonzero(mask & eligible))})
    for bank in range(12):
        for role in range(3):
            for mode in range(2):
                rows = links[(links[:, 6] == bank) & (links[:, 5] == role) & (links[:, 4] == mode)]
                result['occurrence_views'].append({'bank': bank, 'role': role, 'mode': mode,
                    'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)), **summary(rows[:, 12])})
                for e in range(128):
                    result['source_ID_views'].append({'bank': bank, 'role': role, 'mode': mode, 'source_ID': e, **summary(rows[rows[:, 9] == e, 12])})
    for book in range(192):
        for bank in range(12):
            for mode in range(2):
                rows = links[(links[:, 2] == book) & (links[:, 6] == bank) & (links[:, 4] == mode)]
                result['book_views'].append({'book': book, 'bank': bank, 'mode': mode, 'role': 0 if book < 64 else 1 if book < 128 else 2,
                    'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)), **summary(rows[:, 12])})
    assert [len(result[k]) for k in result] == [39, 72, 4608, 9216]
    assert all(sum(v['count'] for v in result[k]) == len(links) for k in ('occurrence_views', 'book_views', 'source_ID_views'))
    return result
