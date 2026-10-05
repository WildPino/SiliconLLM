"""One fixed weight-only balanced tree; no query/label fitted geometry."""
import math
from pathlib import Path
import struct
import numpy as np

D = 768
META = struct.Struct('<8I6d')
OBS = np.dtype([('meta', '<u4', (20,)), ('value', '<f8', (8,)), ('frontier', 'u1', (64,))])
assert META.size == 80 and OBS.itemsize == 208

def gamma(q):
    u = q * 2.**-53
    return u / (1 - u)

def norm_bound(x, q):
    value = float(np.linalg.norm(x))
    return float(np.nextafter(value * (1 + gamma(q)), np.inf))

def build_tree(weights, n, bank, destination, guard):
    assert weights.dtype == np.dtype('<f4') and weights.shape == (n, D)
    assert n in (128, 256) and 0 <= bank < 12 and np.isfinite(weights).all()
    w = weights.astype('<f8')
    nodes = []
    def build(ids, parent):
        index = len(nodes)
        node = {'ids': sorted(ids), 'parent': parent, 'left': 2**32-1, 'right': 2**32-1, 'mode': 0}
        nodes.append(node)
        rows = w[node['ids']]
        m = len(ids)
        padding = 8 * D + m + 16
        maxnorm = max(norm_bound(row, padding) for row in rows)
        if m == 1:
            node.update(radius=0., maxnorm=maxnorm, centernorm=0., meanerror=0., s0=0., s1=0.)
            return index
        center = rows.mean(axis=0)
        residual = rows - center
        centernorm = norm_bound(center, padding)
        radius0 = float(np.max(np.linalg.norm(residual, axis=1)))
        radius = float(np.nextafter(radius0 * (1 + gamma(padding)) + gamma(padding) * (maxnorm + centernorm), np.inf))
        absolute_mean_norm = norm_bound(np.abs(rows).mean(axis=0), padding)
        meanerror = float(np.nextafter(gamma(2*m+2) * absolute_mean_norm, np.inf))
        # Fixed runtime thin SVD. Leading direction is source-recorded; audit
        # checks its eigen residual/partition/radii, without fitting it again.
        u, singular, vt = np.linalg.svd(residual, full_matrices=False)
        assert np.isfinite(singular).all() and np.all(singular[:-1] >= singular[1:])
        s0, s1 = float(singular[0]), float(singular[1])
        if s0 == 0:
            mode = 1
            direction = np.zeros(D, '<f8'); direction[0] = 1
        elif s0 - s1 <= gamma(32*m+16) * s0:
            mode = 2
            direction = np.zeros(D, '<f8'); direction[int(np.argmax(np.sum(residual * residual, axis=0)))] = 1
        else:
            mode = 0
            direction = vt[0].copy()
            first = int(np.argmax(np.abs(direction)))
            if direction[first] < 0: direction *= -1
            eigen_residual = residual.T @ (residual @ direction) - s0*s0*direction
            assert float(np.linalg.norm(eigen_residual)) <= 1e-10 * max(s0*s0, float(np.sum(residual*residual)))
        assert abs(float(direction @ direction) - 1) <= 1e-11
        projections = residual @ direction
        ordered = sorted(zip(projections.tolist(), node['ids']))
        split = m // 2
        node.update(center=center, direction=direction, radius=radius, maxnorm=maxnorm,
                    centernorm=centernorm, meanerror=meanerror, s0=s0, s1=s1, mode=mode)
        node['left'] = build([item[1] for item in ordered[:split]], index)
        node['right'] = build([item[1] for item in ordered[split:]], index)
        guard()
        return index
    assert build(list(range(n)), 2**32-1) == 0 and len(nodes) == 2*n-1
    with Path(destination).open('xb') as stream:
        stream.write(struct.pack('<8s6I', b'M478TRE1', n, bank, D, len(nodes), 80, 8))
        stream.write(weights.tobytes())
        for node in nodes:
            ids = node['ids']; m = len(ids)
            stream.write(META.pack(node['left'], node['right'], node['parent'], m, ids[0], 0, node['mode'], 0,
                                   node['radius'], node['maxnorm'], node['centernorm'], node['meanerror'], node['s0'], node['s1']))
            if m > 1:
                stream.write(node['center'].astype('<f8').tobytes())
                stream.write(node['direction'].astype('<f8').tobytes())
            stream.write(struct.pack('<' + str(m) + 'I', *ids))
    return {'n': n, 'bank': bank, 'nodes': len(nodes), 'internal': n-1,
            'principal_nodes': sum(v['mode'] == 0 for v in nodes if len(v['ids']) > 1),
            'zero_nodes': sum(v['mode'] == 1 for v in nodes), 'degenerate_axis_nodes': sum(v['mode'] == 2 for v in nodes)}

def summarize(records):
    assert len(records)
    meta, values = records['meta'], records['value']
    ratios = values[:, 4]
    relative = np.abs(values[:, 1] / values[:, 0] - 1)
    centers, source, visits = meta[:, 10].astype(np.uint64), meta[:, 9].astype(np.uint64), meta[:, 11].astype(np.uint64)
    n = meta[:, 1].astype(np.uint64)
    assert np.array_equal(visits, centers+source) and np.array_equal(meta[:, 19], meta[:, 11])
    return {'queries': len(records), 'original_rows_mean': float(source.mean()), 'centroid_rows_mean': float(centers.mean()),
            'full_fallback_count': int(np.count_nonzero(meta[:, 13])), 'winner_mismatch_count': int(np.count_nonzero(meta[:, 7] != meta[:, 8])),
            'coefficient_ratio_mean': float(ratios.mean()), 'coefficient_ratio_p95': float(np.quantile(ratios, .95, method='linear')),
            'coefficient_ratio_max': float(ratios.max()), 'maximum_probability_relative_error': float(relative.max()),
            'mean_probability_relative_error': float(relative.mean()), 'heap_comparisons_mean': float(meta[:, 15].mean()),
            'aggregate_updates_mean': float(meta[:, 16].mean()), 'native_exp_calls_mean': float(meta[:, 17].mean()),
            'bound_exp_calls_mean': float(meta[:, 18].mean()), 'visited_state_records_mean': float(visits.mean()),
            'logical_weight_bytes_mean': float((4*D*source + 8*D*centers).mean()),
            'logical_norm_input_bytes_per_query': 4*D,
            'logical_node_descriptor_bytes_mean': float((80*visits).mean()),
            'flat_router_F32_weight_bytes_mean': float((4*D*n).mean())}

def reporting(observations):
    assert observations.shape == (96186,)
    meta = observations['meta']
    result = {'all': summarize(observations), 'targets': []}
    for n in (128, 256):
        target = {'n': n, 'all': summarize(observations[meta[:, 1] == n]), 'modes': {}, 'bank_modes': []}
        for mode, label in ((0, 'teacher'), (1, 'natural')):
            target['modes'][label] = summarize(observations[(meta[:, 1] == n) & (meta[:, 4] == mode)])
        for bank in range(12):
            for mode, label in ((0, 'teacher'), (1, 'natural')):
                rows = observations[(meta[:, 1] == n) & (meta[:, 5] == bank) & (meta[:, 4] == mode)]
                target['bank_modes'].append({'bank': bank, 'mode': label, **summarize(rows)})
        result['targets'].append(target)
    return result
