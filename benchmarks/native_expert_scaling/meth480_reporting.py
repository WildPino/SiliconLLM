"""Complete fixed report slots; no fit or selection from validation."""
import numpy as np

def reports(unique, owner, links, pred):
    U = len(unique)
    original = unique['meta'][:, 7]
    wrong = pred['meta'][:, 2] != original
    p = pred['value'][:, 1].astype('<f8')
    source_p = unique['value'][:, 1]
    relative = np.abs(p - source_p) / source_p
    invalid = (p < 0) | (p > 1) | ~np.isfinite(p)
    log_error = np.abs(pred['root'][:, 2] - (unique['value'][:, 0] + np.log(unique['value'][:, 2])))
    coefficient = pred['meta'][:, 7].astype('<f8') / 98304
    byte = pred['meta'][:, 8:11].astype('<f8').sum(axis=1) / 393216
    def summarize(uid):
        uid = np.asarray(uid)
        count = len(uid)
        if count:
            values = relative[uid]
            p95 = float(np.sort(values)[(95 * count + 99) // 100 - 1])
            same = uid[~wrong[uid]]
            return {'count': count, 'ID_differences': int(np.count_nonzero(wrong[uid])),
                    'invalid_probability_count': int(np.count_nonzero(invalid[uid])),
                    'p_relative_over_1percent': int(np.count_nonzero(values > .01)),
                    'p_relative_mean': float(values.mean(dtype='<f8')),
                    'p_relative_max': float(values.max()), 'p_relative_p95': p95,
                    'same_ID_count': len(same), 'same_ID_p_relative_max': float(relative[same].max()) if len(same) else None,
                    'log_partition_absolute_mean': float(log_error[uid].mean(dtype='<f8')),
                    'log_partition_absolute_max': float(log_error[uid].max()),
                    'coefficient_ratio_mean': float(coefficient[uid].mean(dtype='<f8')),
                    'coefficient_ratio_max': float(coefficient[uid].max()),
                    'coefficient_ratio_p95': float(np.sort(coefficient[uid])[(95 * count + 99) // 100 - 1]),
                    'charged_byte_ratio_mean': float(byte[uid].mean(dtype='<f8')),
                    'charged_byte_ratio_max': float(byte[uid].max()),
                    'charged_byte_ratio_p95': float(np.sort(byte[uid])[(95 * count + 99) // 100 - 1])}
        return {'count': 0, 'ID_differences': 0, 'invalid_probability_count': 0,
                'p_relative_over_1percent': 0, 'p_relative_mean': None, 'p_relative_max': None, 'p_relative_p95': None,
                'same_ID_count': 0, 'same_ID_p_relative_max': None, 'log_partition_absolute_mean': None,
                'log_partition_absolute_max': None, 'coefficient_ratio_mean': None, 'coefficient_ratio_max': None,
                'coefficient_ratio_p95': None, 'charged_byte_ratio_mean': None, 'charged_byte_ratio_max': None,
                'charged_byte_ratio_p95': None}
    result = {'unique_views': [], 'occurrence_views': [], 'book_views': [], 'source_ID_views': []}
    dev = (owner[:, 2] & 5) != 0
    val = (owner[:, 2] & 2) != 0
    for bank in [-1, *range(12)]:
        bank_mask = np.ones(U, bool) if bank < 0 else unique['meta'][:, 2] == bank
        for role, mask in (('ALL', np.ones(U, bool)), ('development', dev), ('consumed_validation', val)):
            result['unique_views'].append({'bank': bank, 'role': role, **summarize(np.flatnonzero(bank_mask & mask))})
    for bank in range(12):
        for role in range(3):
            for mode in range(2):
                rows = links[(links[:, 6] == bank) & (links[:, 5] == role) & (links[:, 4] == mode)]
                result['occurrence_views'].append({'bank': bank, 'role': role, 'mode': mode,
                    'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)),
                    **summarize(rows[:, 12])})
                for expert in range(128):
                    ids = rows[rows[:, 9] == expert, 12]
                    result['source_ID_views'].append({'bank': bank, 'role': role, 'mode': mode,
                                                     'source_ID': expert, **summarize(ids)})
    for book in range(192):
        role = 0 if book < 64 else 1 if book < 128 else 2
        for bank in range(12):
            for mode in range(2):
                rows = links[(links[:, 2] == book) & (links[:, 6] == bank) & (links[:, 4] == mode)]
                result['book_views'].append({'book': book, 'bank': bank, 'role': role, 'mode': mode,
                    'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)),
                    **summarize(rows[:, 12])})
    assert tuple(len(result[k]) for k in ('unique_views', 'occurrence_views', 'book_views', 'source_ID_views')) == (39, 72, 4608, 9216)
    assert all(sum(v['count'] for v in result[k]) == 387036 for k in ('occurrence_views', 'book_views', 'source_ID_views'))
    gates = {'ALL_original_source_ID_tie_conserved': not np.any(wrong),
             'EVERY_native_selected_probability_relative_within_1percent': bool(np.all(relative <= .01)),
             'EVERY_native_probability_in_0_1': not np.any(invalid),
             'EVERY_coefficient_work_within_80percent_flat': bool(np.all(coefficient <= .8)),
             'EVERY_charged_weight_metadata_bytes_within_80percent_flat': bool(np.all(byte <= .8))}
    return result, gates

def physical_diagnostics(unique, pred, models, memberships):
    differences, divergences = [], []
    for bank, model in enumerate(models):
        uid = np.flatnonzero(unique['meta'][:, 2] == bank)
        score = unique['score'][uid].astype('<f8')
        visited = pred['vector_ids'][uid]
        native_score = pred['score'][uid]
        with np.load(model['history_path'], allow_pickle=False) as history:
            parameter = history['parameter'][-1]
        for vector in range(128, 268):
            if vector < 252:
                group = model['groups'][(vector - 128) // 2]
                ids = group['ids']
                a = parameter[group['offset']:group['offset'] + group['size']].reshape(2, len(ids))[(vector - 128) % 2]
            else:
                ids = list(range(128))
                a = parameter[1280:3328].reshape(16, 128)[vector - 252]
            mask = visited == vector
            assert np.all(np.count_nonzero(mask, axis=1) <= 1)
            local = np.flatnonzero(np.any(mask, axis=1))
            if len(local):
                position = np.argmax(mask[local], axis=1)
                oracle = score[local][:, ids] @ a
                error = np.abs(oracle - native_score[local, position].astype('<f8'))
                maximum, mean = float(error.max()), float(error.mean(dtype='<f8'))
            else:
                maximum = mean = None
            differences.append({'bank': bank, 'vector': vector, 'visited_queries': len(local),
                                'cached_teacher_combo_minus_physical_dot_absolute_max': maximum,
                                'cached_teacher_combo_minus_physical_dot_absolute_mean': mean})
        original = unique['meta'][uid, 7]
        direction = pred['direction'][uid]
        node = np.zeros(len(uid), '<u4')
        alive = np.ones(len(uid), bool)
        first = np.zeros(7, '<u4')
        tree = memberships[bank]['nodes']
        for depth in range(7):
            next_node = np.zeros_like(node)
            for current in np.unique(node):
                local = np.flatnonzero(node == current)
                group = tree[int(current)]
                assert group['count'] > 1
                expected_right = np.isin(original[local], tree[group['right']]['ids'])
                in_node = np.isin(original[local], group['ids'])
                assert np.array_equal(alive[local], in_node)
                wrong = alive[local] & (direction[local, depth] != expected_right)
                first[depth] += int(np.count_nonzero(wrong))
                alive[local[wrong]] = False
                next_node[local] = np.where(direction[local, depth], group['right'], group['left'])
            node = next_node
        assert int(first.sum()) == int(np.count_nonzero(pred['meta'][uid, 2] != original))
        assert np.array_equal(np.array([tree[int(v)]['first'] for v in node]), pred['meta'][uid, 2])
        divergences.append({'bank': bank, 'unique_queries': len(uid), 'first_original_winner_divergence_by_depth': first.tolist()})
    assert len(differences) == 1680 and len(divergences) == 12
    return {'surrogate_to_physical_visited_dot_views': differences, 'first_source_path_divergence': divergences}
