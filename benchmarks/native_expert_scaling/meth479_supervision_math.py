"""Fixed wire/target definitions; no fitting or hidden subset of source queries."""
from pathlib import Path
import struct
import numpy as np

D, N = 768, 128
SOURCE = np.dtype([('meta', '<u4', (12,)), ('value', '<f8', (3,))])
UNIQUE = np.dtype([('meta', '<u4', (12,)), ('input_sha', 'u1', (32,)), ('score_sha', 'u1', (32,)),
                   ('value', '<f8', (3,)), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
NODE_TARGET = np.dtype([('maximum', '<f4', (2,)), ('winner', '<u2', (2,)), ('mass', '<f8', (2,))])
TARGET = np.dtype([('unique_id', '<u4'), ('nodes', NODE_TARGET, (127,))])
OWNER = np.dtype([('meta', '<u4', (8,))])
assert [v.itemsize for v in (SOURCE, UNIQUE, NODE_TARGET, TARGET, OWNER)] == [72, 3720, 28, 3560, 32]

def membership(path, bank, original_weights):
    nodes = []
    with Path(path).open('rb') as stream:
        assert stream.read(32) == struct.pack('<8s6I', b'M478TRE1', 128, bank, 768, 255, 80, 8)
        assert stream.read(128 * 768 * 4) == original_weights
        for j in range(255):
            fields = struct.unpack('<8I6d', stream.read(80))
            left, right, parent, count, first, res0, mode, res1 = fields[:8]
            assert res0 == res1 == 0
            if count > 1:
                assert len(stream.read(768 * 16)) == 768 * 16
            ids = list(struct.unpack('<' + str(count) + 'I', stream.read(count * 4)))
            assert ids == sorted(set(ids)) and ids[0] == first and ids[-1] < 128
            nodes.append({'node': j, 'left': left, 'right': right, 'parent': parent, 'count': count, 'first': first, 'ids': ids})
        assert not stream.read(1)
    assert nodes[0]['ids'] == list(range(128)) and nodes[0]['parent'] == 2**32 - 1
    for j, node in enumerate(nodes):
        if j:
            assert node['parent'] < j and j in (nodes[node['parent']]['left'], nodes[node['parent']]['right'])
        if node['count'] > 1:
            l, r = nodes[node['left']], nodes[node['right']]
            assert l['parent'] == r['parent'] == j and l['count'] == r['count'] == node['count'] // 2
            assert sorted(l['ids'] + r['ids']) == node['ids'] and not set(l['ids']) & set(r['ids'])
        else:
            assert node['left'] == node['right'] == 2**32 - 1
    assert sorted(v['first'] for v in nodes if v['count'] == 1) == list(range(128))
    return nodes

def targets(scores, source_fields, nodes, uid):
    assert scores.dtype == np.dtype('<f4') and scores.shape == (len(uid), 128)
    source_winner = np.argmax(scores, axis=1)
    q = np.arange(len(uid))
    assert np.array_equal(scores[q, source_winner].astype('<f8'), source_fields[:, 0])
    # Original F32 subtraction, binary64 exp, F32 term, ID-order F64 sum.
    terms = np.exp((scores - scores[q, source_winner, None]).astype('<f4').astype('<f8')).astype('<f4').astype('<f8')
    root = np.cumsum(terms, axis=1, dtype='<f8')[:, -1]
    assert np.array_equal(root.view('<u8'), source_fields[:, 2].copy().view('<u8'))
    assert np.array_equal((1 / root).astype('<f4').view('<u4'), source_fields[:, 1].astype('<f4').view('<u4'))
    result = np.zeros(len(uid), TARGET)
    result['unique_id'] = uid
    internal = [v for v in nodes if v['count'] > 1]
    assert len(internal) == 127
    winner_node_count = np.zeros((len(uid),), '<u4')
    path_probability = np.ones(len(uid), '<f8')
    for k, node in enumerate(internal):
        group = result['nodes'][:, k]
        for child in (0, 1):
            ids = np.array(nodes[node['left' if not child else 'right']]['ids'])
            chosen = ids[np.argmax(scores[:, ids], axis=1)]
            group['winner'][:, child] = chosen
            group['maximum'][:, child] = scores[q, chosen]
            group['mass'][:, child] = np.cumsum(terms[:, ids], axis=1, dtype='<f8')[:, -1]
        right = (group['maximum'][:, 1] > group['maximum'][:, 0]) | ((group['maximum'][:, 0] == group['maximum'][:, 1]) & (group['winner'][:, 1] < group['winner'][:, 0]))
        in_path = np.isin(source_winner, node['ids'])
        winner_node_count += in_path.astype('<u4')
        chosen = np.where(right, group['winner'][:, 1], group['winner'][:, 0])
        assert np.array_equal(chosen[in_path], source_winner[in_path])
        total = group['mass'].sum(axis=1)
        assert np.all(total[in_path] > 0)
        path_probability[in_path] *= group['mass'][q[in_path], right[in_path].astype(int)] / total[in_path]
    assert np.all(winner_node_count == 7)
    error = np.abs(path_probability - 1 / root)
    assert np.all(error <= 1e-11 + 1e-10 / root)
    return result, {'maximum_F64_path_probability_absolute_error': float(error.max()),
                    'F32_path_probability_differs_from_direct_native': int(np.count_nonzero(path_probability.astype('<f4').view('<u4') != source_fields[:, 1].astype('<f4').view('<u4'))),
                    'zero_child_mass_fields': int(np.count_nonzero(result['nodes']['mass'] == 0))}

def owner_summary(owner):
    meta = owner['meta']
    assert len(owner) and np.all(meta[1:, 0] > meta[:-1, 0])
    role = meta[:, 2]
    development = (role & 5) != 0
    validation = (role & 2) != 0
    return {'unique_inputs': len(owner), 'development_eligible': int(development.sum()),
            'validation_present': int(validation.sum()), 'validation_only': int(np.count_nonzero(validation & ~development)),
            'validation_overlaps_development': int(np.count_nonzero(validation & development)),
            'teacher_present': int(np.count_nonzero(meta[:, 3] & 1)), 'natural_present': int(np.count_nonzero(meta[:, 3] & 2)),
            'accepted_present': int(np.count_nonzero(meta[:, 4] & 2)), 'rejected_present': int(np.count_nonzero(meta[:, 4] & 1)),
            'occurrences': int(meta[:, 5].astype(np.uint64).sum())}

def occurrence_tables(links):
    assert links.shape == (387036, 13)
    result = []
    for bank in range(12):
        for role in range(3):
            for mode in range(2):
                rows = links[(links[:, 6] == bank) & (links[:, 5] == role) & (links[:, 4] == mode)]
                result.append({'bank': bank, 'role': role, 'mode': mode, 'queries': len(rows),
                               'accepted': int(rows[:, 10].sum(dtype=np.uint64)),
                               'rejected': int(np.count_nonzero(rows[:, 10] == 0)),
                               'selected_ID_counts': np.bincount(rows[:, 9], minlength=128).tolist()})
    assert sum(v['queries'] for v in result) == 387036
    return result

def exposure(bank, unique, owner, saved_targets, nodes, development_book_bits):
    ids = np.flatnonzero(unique['meta'][:, 2] == bank)
    dev = (owner['meta'][ids, 2] & 5) != 0
    source_winner = unique['meta'][ids, 7]
    records = saved_targets[ids]['nodes']
    bitmap = development_book_bits[ids]
    internal = [v for v in nodes if v['count'] > 1]
    output = []
    for k, node in enumerate(internal):
        group = records[:, k]
        label = (group['maximum'][:, 1] > group['maximum'][:, 0]) | ((group['maximum'][:, 0] == group['maximum'][:, 1]) & (group['winner'][:, 1] < group['winner'][:, 0]))
        path = np.isin(source_winner, node['ids'])
        def counts(mask):
            union = np.bitwise_or.reduce(bitmap[mask], axis=0, initial=0)
            return {'unique_inputs': int(mask.sum()), 'development_books': sum(int(v).bit_count() for v in union)}
        output.append({'node': node['node'], 'group_size': node['count'],
                       'ALL_development': counts(dev), 'source_winner_path_development': counts(dev & path),
                       'left_ALL_development': counts(dev & ~label), 'right_ALL_development': counts(dev & label),
                       'left_path_development': counts(dev & path & ~label), 'right_path_development': counts(dev & path & label),
                       'development_maximum_ties': int(np.count_nonzero(dev & (group['maximum'][:, 0] == group['maximum'][:, 1]))),
                       'zero_mass_pairs_ALL_unique': int(np.count_nonzero(group['mass'].sum(axis=1) == 0))})
    return output
