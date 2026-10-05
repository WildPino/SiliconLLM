"""Bounded active-row root inquiry with retained mathematical witnesses."""
import time
import struct
import numpy as np
from scipy.sparse import csc_matrix
import scipy.optimize._highspy._core as H
import meth482_affine_math as A

CERT = A.CERT
dot_intervals = A.dot_intervals
physical_intervals = A.physical_intervals
uniform_envelope = A.uniform_envelope
dual_interval = A.dual_interval
sum_upper_nonnegative = A.sum_upper_nonnegative
reports = A.reports

OPTIONS = {'solver': 'simplex', 'presolve': 'on', 'parallel': 'off', 'threads': 1,
           'random_seed': 0,
           'simplex_strategy': 1, 'simplex_dual_edge_weight_strategy': 1,
           'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9,
           'output_flag': False, 'log_to_console': False}
MAX_ROUNDS, ADD_COUNT, SEARCH_SECONDS = 8, 256, 25.


def require_ok(value):
    assert value == H.HighsStatus.kOk, str(value)


def require_ieee():
    tiny = np.array([np.nextafter(np.float64(0.), np.float64(1.))], '<f8')
    assert tiny.tobytes() == struct.pack('<Q', 1)
    assert (tiny*np.float64(1.)).tobytes() == struct.pack('<Q', 1)
    assert (tiny+tiny).tobytes() == struct.pack('<Q', 2)


def constraint_rows(x, y):
    n, width = x.shape
    d = width + 1
    h = np.column_stack((x.astype('<f8'), np.ones(n))) * y[:, None]
    rows = np.empty((n, 2*d+1), '<f8')
    rows[:, :d], rows[:, d:2*d], rows[:, -1] = -h, h, 1.
    return rows


def normalize(raw, d):
    theta = raw[:d] - raw[d:2*d]
    assert np.isfinite(theta).all()
    norm = sum_upper_nonnegative(np.abs(theta))
    divisor = max(1., float(A.up(float(A.up(norm)) * (1. + 2.**-40))))
    assert np.isfinite(divisor) and divisor > 0
    theta = theta / divisor
    assert sum_upper_nonnegative(np.abs(theta)) <= 1
    return theta, divisor


def solve(x, y, source_IDs=None, save_round=None):
    started = time.monotonic()
    assert x.dtype == np.dtype('<f4') and y.dtype == np.dtype('<i1')
    assert len(x) == len(y) > 0 and np.isfinite(x).all() and np.all(np.isin(y, [-1, 1]))
    n, width = x.shape
    d, columns = width+1, 2*(width+1)+1
    if source_IDs is None:
        source_IDs = np.arange(n)
    assert np.asarray(source_IDs).shape == (n,)
    active = np.sort(np.unique(source_IDs, return_index=True)[1]).astype('<u4')
    initial = active.copy()
    selected = np.zeros(n, bool)
    selected[active] = True
    envelope, _ = uniform_envelope(x)

    highs = H._Highs()
    options_verified = {}
    for key, value in OPTIONS.items():
        require_ok(highs.setOptionValue(key, value))
        option_status, actual = highs.getOptionValue(key)
        require_ok(option_status)
        assert type(actual) is type(value) and actual == value, (key, actual, value)
        options_verified[key] = actual
    matrix = np.zeros((len(active)+1, columns), '<f8')
    matrix[0, :2*d] = 1.
    matrix[1:] = constraint_rows(x[active], y[active])
    sparse = csc_matrix(matrix)
    lp = H.HighsLp()
    lp.num_col_, lp.num_row_ = columns, len(active)+1
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = columns, len(active)+1
    lp.a_matrix_.format_ = H.MatrixFormat.kColwise
    lp.a_matrix_.start_ = sparse.indptr.astype('<i4')
    lp.a_matrix_.index_ = sparse.indices.astype('<i4')
    lp.a_matrix_.value_ = sparse.data
    objective = np.zeros(columns)
    objective[-1] = -1.
    lp.col_cost_ = objective
    lp.col_lower_, lp.col_upper_ = np.zeros(columns), np.full(columns, np.inf)
    upper = np.zeros(len(active)+1)
    upper[0] = 1.
    lp.row_lower_, lp.row_upper_ = np.full(len(active)+1, -np.inf), upper
    require_ok(highs.passModel(lp))
    del matrix, sparse, lp
    best_primal = best_dual = None
    rounds = []
    stop = 'round_limit'
    for ordinal in range(MAX_ROUNDS):
        require_ieee()
        remaining = SEARCH_SECONDS - (time.monotonic()-started)
        if remaining <= 0:
            stop = 'search_wall_budget'
            break
        # A remaining-wall allowance is conservative even when the internal
        # run clock is cumulative. Never give a fresh25s to a later round.
        run_before = float(highs.getRunTime())
        require_ok(highs.setOptionValue('time_limit', remaining))
        option_status, actual_limit = highs.getOptionValue('time_limit')
        require_ok(option_status)
        assert type(actual_limit) is float and actual_limit == remaining
        run_status = highs.run()
        assert run_status != H.HighsStatus.kError, str(run_status)
        status = highs.getModelStatus()
        assert status in (H.HighsModelStatus.kOptimal, H.HighsModelStatus.kTimeLimit,
                          H.HighsModelStatus.kIterationLimit), str(status)
        info = highs.getInfo()
        solution = highs.getSolution()
        require_ieee()
        primal_valid, dual_valid = bool(solution.value_valid), bool(solution.dual_valid)
        raw = np.asarray(solution.col_value, '<f8') if primal_valid else np.empty(0, '<f8')
        dual = np.asarray(solution.row_dual, '<f8') if dual_valid else np.empty(0, '<f8')
        assert (not primal_valid or raw.shape == (columns,)) and np.isfinite(raw).all()
        assert (not dual_valid or dual.shape == (len(active)+1,)) and np.isfinite(dual).all()
        if status == H.HighsModelStatus.kOptimal:
            assert primal_valid and dual_valid
        theta = np.zeros(d, '<f8')
        divisor = 1.
        signed_lo, signed_hi = np.zeros(n), np.zeros(n)
        lower = None
        if primal_valid:
            theta, divisor = normalize(raw, d)
            lo, hi, _ = dot_intervals(x, theta)
            signed_lo, signed_hi = np.where(y > 0, lo, -hi), np.where(y > 0, hi, -lo)
            lower = float(signed_lo.min())
            if best_primal is None or lower > best_primal['lower']:
                best_primal = {'round': ordinal, 'raw': raw.copy(), 'theta': theta.copy(),
                               'lower': lower, 'divisor': divisor}
        weights = np.zeros(len(active), '<f8')
        dual_bound = None
        if dual_valid:
            weights = np.maximum(0., -dual[1:])
            maximum = float(weights.max())
            if maximum > 0:
                weights /= maximum
            _, _, _, _, dual_bound = dual_interval(x[active], y[active], weights)
            if dual_bound is not None and (best_dual is None or dual_bound < best_dual['bound']):
                best_dual = {'round': ordinal, 'raw': dual.copy(), 'active': active.copy(),
                             'weights': weights.copy(), 'bound': dual_bound}
        if lower is not None and dual_bound is not None:
            assert max(0., lower) <= dual_bound
        added = np.empty(0, '<u4')
        if best_primal is not None and best_primal['lower'] > envelope:
            reason = 'complete_development_lower_exceeds_uniform_envelope'
        elif best_dual is not None and best_dual['bound'] <= envelope:
            reason = 'verified_dual_excludes_uniform_margin_requirement'
        elif time.monotonic()-started >= SEARCH_SECONDS:
            reason = 'search_wall_budget'
        elif ordinal+1 == MAX_ROUNDS:
            reason = 'round_limit'
        elif not primal_valid:
            reason = 'no_valid_primal_for_separation'
        else:
            inactive = np.flatnonzero(~selected)
            eligible = inactive[signed_lo[inactive] <= envelope]
            # All development arrays are in ascending source UID order.
            order = np.lexsort((eligible, signed_lo[eligible]))
            added = eligible[order[:ADD_COUNT]].astype('<u4')
            reason = 'add_complete_scan_violations' if len(added) else 'no_inactive_uniform_violations'
        record = {'round': ordinal, 'active_count': len(active), 'added_count': len(added),
                  'model_status': int(status), 'model_status_name': str(status),
                  'message': highs.modelStatusToString(status), 'run_status': str(run_status),
                  'value_valid': primal_valid, 'dual_valid': dual_valid,
                  'primal_solution_status': int(info.primal_solution_status),
                  'dual_solution_status': int(info.dual_solution_status),
                  'simplex_nit': int(info.simplex_iteration_count),
                  'solver_run_seconds_before': run_before, 'solver_run_seconds_after': float(highs.getRunTime()),
                  'time_limit_wall_remainder': remaining, 'wall_seconds': time.monotonic()-started,
                  'candidate_unit_L1_norm_upper': sum_upper_nonnegative(np.abs(theta)),
                  'candidate_divisor': divisor, 'complete_development_margin_lower': lower,
                  'subset_dual_full_margin_upper': dual_bound, 'stop_or_add': reason}
        arrays = {'active_development_indices': active.copy(), 'added_development_indices': added,
                  'raw_primal': raw.copy(), 'raw_dual': dual.copy(), 'theta': theta,
                  'signed_development_lower': signed_lo, 'signed_development_upper': signed_hi,
                  'dual_weights_active': weights}
        if save_round is not None:
            save_round(ordinal, record, arrays)
        rounds.append(record)
        if reason != 'add_complete_scan_violations':
            stop = reason
            break
        if time.monotonic()-started >= SEARCH_SECONDS:
            stop = 'search_wall_budget_before_add'
            break
        added_rows = constraint_rows(x[added], y[added])
        csr_start = np.arange(0, len(added)*columns, columns, dtype='<i4')
        csr_indices = np.tile(np.arange(columns, dtype='<i4'), len(added))
        if time.monotonic()-started >= SEARCH_SECONDS:
            stop = 'search_wall_budget_after_row_construction'
            break
        require_ok(highs.addRows(len(added), np.full(len(added), -np.inf), np.zeros(len(added)),
                                 added_rows.size, csr_start, csr_indices, added_rows.ravel()))
        active = np.concatenate((active, added))
        assert len(np.unique(active)) == len(active)
        selected[added] = True
        del added_rows
    theta = best_primal['theta'] if best_primal is not None else np.zeros(d, '<f8')
    raw = best_primal['raw'] if best_primal is not None else np.empty(0, '<f8')
    dual = best_dual['raw'] if best_dual is not None else np.empty(0, '<f8')
    weights = np.zeros(n, '<f8')
    dual_indices = np.empty(0, '<u4')
    if best_dual is not None:
        dual_indices = best_dual['active']
        weights[dual_indices] = best_dual['weights']
    primal_round = best_primal['round'] if best_primal is not None else -1
    dual_round = best_dual['round'] if best_dual is not None else -1
    chosen = rounds[primal_round] if primal_round >= 0 else rounds[-1] if rounds else None
    solver = {'status': 0 if chosen and chosen['model_status_name'] == str(H.HighsModelStatus.kOptimal) else 1,
              'success': bool(chosen and chosen['model_status_name'] == str(H.HighsModelStatus.kOptimal)),
              'message': chosen['message'] if chosen else 'No solve before search wall budget',
              'nit': chosen['simplex_nit'] if chosen else 0,
              'fun': -float(raw[-1]) if len(raw) else None,
              'available': best_primal is not None, 'warnings': [],
              'primal_round': primal_round, 'dual_round': dual_round,
              'stop': stop, 'rounds': rounds, 'initial_active_count': len(initial),
              'search_wall_seconds': time.monotonic()-started,
              'options_verified': options_verified,
              'scope': 'Chosen active-LP status only; complete primal/dual interval checks determine full-domain statements.'}
    return solver, raw, dual, theta, theta.astype('<f4'), weights, dual_indices
