"""METH483-R1 full matrix readback and exact saved-round resumption."""
import time
import struct
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
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
           'output_flag': True, 'log_to_console': False, 'small_matrix_value': 1e-9}
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


def model_readback(highs, expected, status, operation, events):
    assert status in (H.HighsStatus.kOk, H.HighsStatus.kWarning), str(status)
    actual = highs.getLp()
    rows, columns = expected.shape
    assert (actual.num_row_, actual.num_col_) == (rows, columns)
    assert actual.sense_ == H.ObjSense.kMinimize and actual.offset_ == 0.
    cost = np.zeros(columns, '<f8'); cost[-1] = -1.
    upper = np.zeros(rows, '<f8'); upper[0] = 1.
    for observed, required in ((actual.col_cost_, cost), (actual.col_lower_, np.zeros(columns)),
                               (actual.col_upper_, np.full(columns, np.inf)),
                               (actual.row_lower_, np.full(rows, -np.inf)), (actual.row_upper_, upper)):
        assert np.asarray(observed, '<f8').tobytes() == np.asarray(required, '<f8').tobytes()
    matrix = actual.a_matrix_
    assert (matrix.num_row_, matrix.num_col_) == (rows, columns)
    assert matrix.format_ in (H.MatrixFormat.kColwise, H.MatrixFormat.kRowwise)
    pointer = np.asarray(matrix.start_, '<i4')
    indices = np.asarray(matrix.index_, '<i4')
    values = np.asarray(matrix.value_, '<f8')
    major = columns if matrix.format_ == H.MatrixFormat.kColwise else rows
    minor = rows if matrix.format_ == H.MatrixFormat.kColwise else columns
    assert pointer.shape == (major+1,) and pointer[0] == 0 and pointer[-1] == len(values)
    assert indices.shape == values.shape and np.all(np.diff(pointer) >= 0)
    assert np.isfinite(values).all() and np.all((indices >= 0) & (indices < minor))
    for k in range(major):
        assert len(np.unique(indices[pointer[k]:pointer[k+1]])) == pointer[k+1]-pointer[k]
    constructor = csc_matrix if matrix.format_ == H.MatrixFormat.kColwise else csr_matrix
    observed = constructor((values, indices, pointer), shape=(rows, columns)).toarray()
    dropped = (expected != 0.) & (np.abs(expected) <= OPTIONS['small_matrix_value'])
    wanted = expected.copy(); wanted[dropped] = 0.
    # Sparse storage omits mathematical zero irrespective of its sign bit.
    observed[observed == 0.] = 0.; wanted[wanted == 0.] = 0.
    assert observed.astype('<f8').tobytes() == wanted.astype('<f8').tobytes()
    locations = np.argwhere(dropped)
    events.append({'operation': operation, 'status': str(status), 'rows': rows, 'columns': columns,
                   'readback_format': str(matrix.format_), 'readback_nonzeros': len(values),
                   'original_dense_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                   'verified_dense_sha256': hashlib.sha256(wanted.tobytes()).hexdigest(),
                   'deleted_coefficients': [{'row': int(i), 'column': int(j),
                                             'F64_bits': int(expected[i,j].view('<u8')),
                                             'value': float(expected[i,j])} for i,j in locations]})


def saved_rounds(x, y, source_IDs, directory, prefix, count):
    """Reconstruct frozen completed vectors/selection without any solver call."""
    n, width = x.shape; d = width+1; columns = 2*d+1
    active = np.sort(np.unique(source_IDs, return_index=True)[1]).astype('<u4')
    initial = active.copy(); selected = np.zeros(n, bool); selected[active] = True
    envelope, _ = uniform_envelope(x)
    best_primal = best_dual = None; rounds = []
    for ordinal in range(count):
        stem = Path(directory) / (prefix+'_round'+str(ordinal))
        record = json.loads(stem.with_suffix('.json').read_bytes())
        with np.load(stem.with_suffix('.npz'), allow_pickle=False) as archive:
            assert set(archive.files) == {'active_development_indices','added_development_indices','raw_primal',
                'raw_dual','theta','signed_development_lower','signed_development_upper','dual_weights_active'}
            arrays = {key: archive[key] for key in archive.files}
        assert record['round'] == ordinal and record['active_count'] == len(active)
        assert arrays['active_development_indices'].dtype == np.dtype('<u4')
        assert arrays['active_development_indices'].tobytes() == active.tobytes()
        raw, dual = arrays['raw_primal'], arrays['raw_dual']
        assert raw.dtype == dual.dtype == np.dtype('<f8') and np.isfinite(raw).all() and np.isfinite(dual).all()
        assert raw.shape == ((columns,) if record['value_valid'] else (0,))
        assert dual.shape == ((len(active)+1,) if record['dual_valid'] else (0,))
        theta = np.zeros(d, '<f8'); divisor = 1.; lower = None
        signed_lo = np.zeros(n, '<f8'); signed_hi = signed_lo.copy()
        if record['value_valid']:
            theta, divisor = normalize(raw, d)
            lo, hi, _ = dot_intervals(x, theta)
            signed_lo, signed_hi = np.where(y > 0, lo, -hi), np.where(y > 0, hi, -lo)
            lower = float(signed_lo.min())
            if best_primal is None or lower > best_primal['lower']:
                best_primal = {'round': ordinal, 'raw': raw.copy(), 'theta': theta.copy(), 'lower': lower, 'divisor': divisor}
        weights = np.zeros(len(active), '<f8'); bound = None
        if record['dual_valid']:
            weights = np.maximum(0., -dual[1:]); maximum = float(weights.max())
            if maximum > 0.: weights /= maximum
            _, _, _, _, bound = dual_interval(x[active], y[active], weights)
            if bound is not None and (best_dual is None or bound < best_dual['bound']):
                best_dual = {'round': ordinal, 'raw': dual.copy(), 'active': active.copy(), 'weights': weights.copy(), 'bound': bound}
        for key, wanted in (('theta',theta),('signed_development_lower',signed_lo),
                            ('signed_development_upper',signed_hi),('dual_weights_active',weights)):
            assert arrays[key].dtype == wanted.dtype and arrays[key].tobytes() == wanted.tobytes(), key
        for key, wanted in (('candidate_divisor',divisor),('candidate_unit_L1_norm_upper',sum_upper_nonnegative(np.abs(theta))),
                            ('complete_development_margin_lower',lower),('subset_dual_full_margin_upper',bound)):
            assert record[key] == wanted, (key, record[key], wanted)
        added = arrays['added_development_indices']
        assert added.dtype == np.dtype('<u4') and added.shape == (record['added_count'],)
        if len(added):
            assert record['stop_or_add'] == 'add_complete_scan_violations'
            inactive = np.flatnonzero(~selected); eligible = inactive[signed_lo[inactive] <= envelope]
            wanted = eligible[np.lexsort((eligible,signed_lo[eligible]))[:ADD_COUNT]].astype('<u4')
            assert added.tobytes() == wanted.tobytes()
        rounds.append(record)
        if ordinal+1 < count:
            assert len(added) > 0
            active = np.concatenate((active,added)); selected[added] = True
    return {'rounds': rounds, 'best_primal': best_primal, 'best_dual': best_dual, 'initial': initial,
            'last_active': active, 'last_added': added.copy()}


def finish(n, d, best_primal, best_dual, rounds, initial, stop, seconds, options_verified, events):
    theta = best_primal['theta'] if best_primal is not None else np.zeros(d, '<f8')
    raw = best_primal['raw'] if best_primal is not None else np.empty(0, '<f8')
    dual = best_dual['raw'] if best_dual is not None else np.empty(0, '<f8')
    weights = np.zeros(n, '<f8'); dual_indices = np.empty(0, '<u4')
    if best_dual is not None:
        dual_indices = best_dual['active']; weights[dual_indices] = best_dual['weights']
    primal_round = best_primal['round'] if best_primal is not None else -1
    dual_round = best_dual['round'] if best_dual is not None else -1
    chosen = rounds[primal_round] if primal_round >= 0 else rounds[-1] if rounds else None
    optimal = bool(chosen and chosen['model_status_name'].endswith('kOptimal'))
    solver = {'status': 0 if optimal else 1, 'success': optimal,
              'message': chosen['message'] if chosen else 'No solve before search wall budget',
              'nit': chosen['simplex_nit'] if chosen else 0, 'fun': -float(raw[-1]) if len(raw) else None,
              'available': best_primal is not None, 'warnings': [], 'primal_round': primal_round, 'dual_round': dual_round,
              'stop': stop, 'rounds': rounds, 'initial_active_count': len(initial), 'search_wall_seconds': seconds,
              'options_verified': options_verified, 'interface_events': events,
              'scope': 'Chosen active-LP status only; full-domain original-input interval bounds determine science.'}
    return solver, raw, dual, theta, theta.astype('<f4'), weights, dual_indices


def solve(x, y, source_IDs=None, save_round=None, *, log_path, seed=None, paid_seconds=0., control_append=False):
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
    if seed is not None:
        assert len(seed['rounds']) == 1 and seed['rounds'][0]['round'] == 0
        assert seed['rounds'][0]['stop_or_add'] == 'add_complete_scan_violations'
        assert initial.tobytes() == seed['initial'].tobytes()
        active = np.concatenate((seed['last_active'], seed['last_added']))
    if control_append:
        assert seed is None and n == 2 and width == 2
        active = np.array([0], '<u4'); initial = active.copy()
    selected = np.zeros(n, bool)
    selected[active] = True
    envelope, _ = uniform_envelope(x)

    highs = H._Highs()
    options_verified = {}
    events = []
    assert not Path(log_path).exists()
    for key, value in {**OPTIONS, 'log_file': str(Path(log_path).resolve())}.items():
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
    model_readback(highs, matrix, highs.passModel(lp), 'passModel', events)
    if control_append:
        extra = constraint_rows(x[1:], y[1:])
        status = highs.addRows(1, np.array([-np.inf]), np.array([0.]), extra.size,
                               np.array([0], '<i4'), np.arange(columns, dtype='<i4'), extra.ravel())
        matrix = np.vstack((matrix,extra))
        model_readback(highs,matrix,status,'analytic_tiny_addRows',events)
        assert status == H.HighsStatus.kWarning and len(events[-1]['deleted_coefficients']) == 2
        active = np.array([0,1], '<u4'); selected[:] = True
    del sparse, lp
    best_primal = seed['best_primal'] if seed else None
    best_dual = seed['best_dual'] if seed else None
    rounds = list(seed['rounds']) if seed else []
    stop = 'round_limit'
    for ordinal in range(len(rounds), MAX_ROUNDS):
        require_ieee()
        remaining = SEARCH_SECONDS - paid_seconds - (time.monotonic()-started)
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
        elif paid_seconds + time.monotonic()-started >= SEARCH_SECONDS:
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
        if paid_seconds + time.monotonic()-started >= SEARCH_SECONDS:
            stop = 'search_wall_budget_before_add'
            break
        added_rows = constraint_rows(x[added], y[added])
        csr_start = np.arange(0, len(added)*columns, columns, dtype='<i4')
        csr_indices = np.tile(np.arange(columns, dtype='<i4'), len(added))
        if paid_seconds + time.monotonic()-started >= SEARCH_SECONDS:
            stop = 'search_wall_budget_after_row_construction'
            break
        status = highs.addRows(len(added), np.full(len(added), -np.inf), np.zeros(len(added)),
                               added_rows.size, csr_start, csr_indices, added_rows.ravel())
        matrix = np.vstack((matrix,added_rows))
        model_readback(highs,matrix,status,'addRows_after_round'+str(ordinal),events)
        active = np.concatenate((active, added))
        assert len(np.unique(active)) == len(active)
        selected[added] = True
        del added_rows
    return finish(n, d, best_primal, best_dual, rounds, initial, stop,
                  paid_seconds+time.monotonic()-started, options_verified, events)
