"""Source-only margin/moment diagnostics. No fitted candidate or native execution."""
import numpy as np

N, U = 128, 238872
UNIQUE = np.dtype([('meta', '<u4', (12,)), ('input_sha', 'u1', (32,)), ('score_sha', 'u1', (32,)),
                   ('value', '<f8', (3,)), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
NODE = np.dtype([('maximum', '<f4', (2,)), ('winner', '<u2', (2,)), ('mass', '<f8', (2,))])
TARGET = np.dtype([('unique_id', '<u4'), ('nodes', NODE, (127,))])
PRED = np.dtype([('meta', '<u4', (12,)), ('value', '<f4', (3,)), ('root', '<f8', (4,)),
                 ('vector_ids', '<u2', (42,)), ('score', '<f4', (42,)), ('direction', '<u4', (7,))])
PATH = np.dtype([('node', '<u2'), ('left_winner', 'u1'), ('right_winner', 'u1'), ('margin', '<f8')])
GEOMETRY = np.dtype([('meta', '<u4', (4,)), ('moment', '<f8', (7,)), ('path', PATH, (7,))])
assert (UNIQUE.itemsize, TARGET.itemsize, PRED.itemsize, PATH.itemsize, GEOMETRY.itemsize) == (3720, 3560, 372, 12, 156)
LOG_LOW, LOG_HIGH = np.log(.99), np.log(1.01)
ETA = min(-LOG_LOW, LOG_HIGH)

def moments(score, native):
    s = score.astype('<f8')
    mean = s.mean(axis=1, dtype='<f8')
    residual = s - mean[:, None]
    drift = residual.mean(axis=1, dtype='<f8')
    variance = (residual * residual).mean(axis=1, dtype='<f8')
    radius = np.abs(residual).max(axis=1)
    third = (np.abs(residual) ** 3).mean(axis=1, dtype='<f8')
    m2 = 1 + drift + variance / 2
    assert np.all(m2 > 0)
    A2 = mean + np.log(N) + np.log(m2)
    top = s.max(axis=1)
    real_A = top + np.log(np.exp(s - top[:, None]).sum(axis=1, dtype='<f8'))
    native_A = native[:, 0] + np.log(native[:, 2])
    result = np.stack((mean, drift, variance, radius, third, A2, native_A - real_A), axis=1)
    assert np.isfinite(result).all()
    positive = third > 0
    log_bound = np.full(len(s), -np.inf)
    log_bound[positive] = radius[positive] + np.log(third[positive]) - np.log(6.) - np.minimum(drift[positive], np.log(m2[positive]))
    # Observed F64 consistency of the mathematical remainder, not directed rounding.
    gap = np.abs(A2 - real_A)
    assert np.all(gap[~positive] <= 1e-11)
    needed = gap[positive] > 1e-11
    assert np.all(np.log(gap[positive][needed] - 1e-11) <= log_bound[positive][needed] + 1e-10)
    return result, log_bound, real_A

def paths(row, target, pred, tree):
    size = len(row)
    internal = [x for x in tree if x['count'] > 1]
    index = {v['node']: k for k, v in enumerate(internal)}
    nodes = np.zeros(size, '<u4')
    fields = np.zeros((size, 7), PATH)
    first = np.full(size, 2**32 - 1, '<u4')
    metrics = {'margin': np.zeros((size, 7)), 'differential': np.zeros((size, 7)),
               'common_error': np.zeros((size, 7)), 'wrong_branch': np.zeros((size, 7), bool),
               'source_tie': np.zeros((size, 7), bool), 'candidate_tie': np.zeros((size, 7), bool),
               'strict_sufficient': np.zeros((size, 7), bool)}
    slot = 0
    for depth in range(7):
        next_node = np.empty_like(nodes)
        forms = 1 if depth == 6 else 2
        for current in np.unique(nodes):
            local = np.flatnonzero(nodes == current)
            node = tree[int(current)]
            t = target['nodes'][local, index[int(current)]]
            left, right = t['maximum'][:, 0].astype('<f8'), t['maximum'][:, 1].astype('<f8')
            lw, rw = t['winner'][:, 0], t['winner'][:, 1]
            D = right - left
            expected = (right > left) | ((right == left) & (rw < lw))
            approximate = pred['score'][local, slot:slot + forms].max(axis=1).astype('<f8')
            approximate_r = pred['score'][local, slot + forms:slot + 2 * forms].max(axis=1).astype('<f8')
            differential = (approximate_r - approximate) - D
            children = [tree[node['left']], tree[node['right']]]
            lk = np.full(len(local), children[0]['first'])
            rk = np.full(len(local), children[1]['first'])
            if children[0]['count'] <= 2:
                lk = np.array(children[0]['ids'])[np.argmax(pred['score'][local, slot:slot + forms], axis=1)]
                rk = np.array(children[1]['ids'])[np.argmax(pred['score'][local, slot + forms:slot + 2 * forms], axis=1)]
            choice = (approximate_r > approximate) | ((approximate_r == approximate) & (rk < lk))
            assert np.array_equal(choice.astype('<u4'), pred['direction'][local, depth])
            sufficient = (D != 0) & (np.abs(differential) < np.abs(D))
            wrong = choice != expected
            assert not np.any(sufficient & wrong)
            fields['node'][local, depth] = current
            fields['left_winner'][local, depth], fields['right_winner'][local, depth] = lw, rw
            fields['margin'][local, depth] = D
            for name, value in (('margin', D), ('differential', differential), ('common_error', ((approximate - left) + (approximate_r - right)) / 2), ('wrong_branch', wrong), ('source_tie', D == 0), ('candidate_tie', approximate == approximate_r), ('strict_sufficient', sufficient)):
                metrics[name][local, depth] = value
            alive = first[local] == 2**32 - 1
            assert np.all(np.isin(row['meta'][local[alive], 7], node['ids']))
            diverged = alive & wrong
            first[local[diverged]] = depth
            next_node[local] = np.where(choice, node['right'], node['left'])
        nodes = next_node
        slot += 2 * forms
    assert slot == 26
    assert np.array_equal(first == 2**32 - 1, pred['meta'][:, 2] == row['meta'][:, 7])
    return fields, first, metrics

def full_node_views(row, owner, targets, tree, bank):
    views = []
    for k, node in enumerate(x for x in tree if x['count'] > 1):
        t = targets['nodes'][:, k]
        D = t['maximum'][:, 1].astype('<f8') - t['maximum'][:, 0].astype('<f8')
        right = (D > 0) | ((D == 0) & (t['winner'][:, 1] < t['winner'][:, 0]))
        winners = np.where(right, t['winner'][:, 1], t['winner'][:, 0])
        path = np.isin(row['meta'][:, 7], node['ids'])
        for role, mask in (('ALL', np.ones(len(row), bool)), ('development', (owner[:, 2] & 5) != 0), ('consumed_validation', (owner[:, 2] & 2) != 0)):
            a = np.abs(D[mask])
            count = len(a)
            assert count
            ordered = np.sort(a)
            buckets = [int(np.count_nonzero(a == 0))]
            buckets += [int(np.count_nonzero((a > lo) & (a <= hi))) for lo, hi in ((0, .001), (.001, .01), (.01, .1), (.1, 1))]
            buckets += [int(np.count_nonzero(a > 1))]
            views.append({'bank': bank, 'node': node['node'], 'role': role, 'count': count,
                          'source_right_count': int(np.count_nonzero(right[mask])), 'source_ties': buckets[0],
                          'absolute_margin_mean': float(a.mean(dtype='<f8')), 'absolute_margin_min': float(a.min()),
                          'absolute_margin_max': float(a.max()), 'absolute_margin_p95': float(ordered[(95 * count + 99) // 100 - 1]),
                          'absolute_margin_bins_0_1e3_1e2_1e1_1_inf': buckets,
                          'local_winner_ID_counts': np.bincount(winners[mask].astype('<i8'), minlength=128).tolist(),
                          'global_winner_path_count': int(np.count_nonzero(path & mask))})
    return views

def reports(unique, owner, links, geometry, path_metrics):
    m = geometry['moment']
    mean, drift, variance, radius, third, A2, native_rounding = m.T
    native_A = unique['value'][:, 0] + np.log(unique['value'][:, 2])
    log_ratio = unique['value'][:, 0] - A2 - np.log(unique['value'][:, 1])
    positive = third > 0
    log_bound = np.full(len(unique), -np.inf)
    log_bound[positive] = radius[positive] + np.log(third[positive]) - np.log(6.) - np.minimum(drift[positive], np.log(1 + drift[positive] + variance[positive] / 2))
    abs_round = np.abs(native_rounding)
    # Nominal screen only: rounded F64 evaluation is not an interval certificate.
    allowed = ETA - abs_round
    screen = np.zeros(len(unique), bool)
    good = allowed > 0
    screen[good] = log_bound[good] <= np.log(allowed[good])
    first = geometry['meta'][:, 3]
    def summarize(ids):
        n = len(ids)
        if not n:
            return {'count': 0, 'moment_log_ratio_outside_1percent': 0, 'moment_over_probability_witnesses': 0,
                    'moment_probability_above_1': 0, 'nominal_Taylor_screen_count': 0, 'first_divergence_counts': [0] * 7,
                    'visited_wrong_branches': 0, 'visited_strict_sufficient': 0, 'visited_source_ties': 0,
                    'visited_candidate_ties': 0, 'mean_absolute_native_A_error': None, 'max_absolute_native_A_error': None,
                    'max_absolute_native_rounding': None, 'variance_mean': None, 'radius_max': None,
                    'max_log_Taylor_bound': None, 'max_absolute_differential_error': None, 'mean_absolute_common_support_error': None}
        log_values = log_bound[ids]
        finite = log_values[np.isfinite(log_values)]
        return {'count': n, 'moment_log_ratio_outside_1percent': int(np.count_nonzero((log_ratio[ids] < LOG_LOW) | (log_ratio[ids] > LOG_HIGH))),
                'moment_over_probability_witnesses': int(np.count_nonzero(log_ratio[ids] > LOG_HIGH)),
                'moment_probability_above_1': int(np.count_nonzero(unique['value'][ids, 0] > A2[ids])),
                'nominal_Taylor_screen_count': int(np.count_nonzero(screen[ids])),
                'first_divergence_counts': np.bincount(first[ids][first[ids] < 7].astype('<i8'), minlength=7).tolist(),
                'visited_wrong_branches': int(np.count_nonzero(path_metrics['wrong_branch'][ids])),
                'visited_strict_sufficient': int(np.count_nonzero(path_metrics['strict_sufficient'][ids])),
                'visited_source_ties': int(np.count_nonzero(path_metrics['source_tie'][ids])),
                'visited_candidate_ties': int(np.count_nonzero(path_metrics['candidate_tie'][ids])),
                'mean_absolute_native_A_error': float(np.abs(A2[ids] - native_A[ids]).mean(dtype='<f8')),
                'max_absolute_native_A_error': float(np.abs(A2[ids] - native_A[ids]).max()),
                'max_absolute_native_rounding': float(abs_round[ids].max()), 'variance_mean': float(variance[ids].mean(dtype='<f8')),
                'radius_max': float(radius[ids].max()), 'max_log_Taylor_bound': float(finite.max()) if len(finite) else None,
                'max_absolute_differential_error': float(np.abs(path_metrics['differential'][ids]).max()),
                'mean_absolute_common_support_error': float(np.abs(path_metrics['common_error'][ids]).mean(dtype='<f8'))}
    result = {'unique_views': [], 'occurrence_views': [], 'book_views': [], 'source_ID_views': []}
    for bank in [-1, *range(12)]:
        eligible = np.ones(len(unique), bool) if bank < 0 else unique['meta'][:, 2] == bank
        for role, mask in (('ALL', np.ones(len(unique), bool)), ('development', (owner[:, 2] & 5) != 0), ('consumed_validation', (owner[:, 2] & 2) != 0)):
            result['unique_views'].append({'bank': bank, 'role': role, **summarize(np.flatnonzero(eligible & mask))})
    for bank in range(12):
        for role in range(3):
            for mode in range(2):
                rows = links[(links[:, 6] == bank) & (links[:, 5] == role) & (links[:, 4] == mode)]
                result['occurrence_views'].append({'bank': bank, 'role': role, 'mode': mode, 'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)), **summarize(rows[:, 12])})
                for e in range(128):
                    result['source_ID_views'].append({'bank': bank, 'role': role, 'mode': mode, 'source_ID': e, **summarize(rows[rows[:, 9] == e, 12])})
    for book in range(192):
        for bank in range(12):
            for mode in range(2):
                rows = links[(links[:, 2] == book) & (links[:, 6] == bank) & (links[:, 4] == mode)]
                result['book_views'].append({'book': book, 'bank': bank, 'mode': mode, 'role': 0 if book < 64 else 1 if book < 128 else 2, 'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)), **summarize(rows[:, 12])})
    assert [len(result[k]) for k in result] == [39, 72, 4608, 9216]
    assert all(sum(x['count'] for x in result[k]) == len(links) for k in ('occurrence_views', 'book_views', 'source_ID_views'))
    return result
