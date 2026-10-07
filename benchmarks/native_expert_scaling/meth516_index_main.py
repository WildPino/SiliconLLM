"""One fixed64 original-coordinate index and exact retained WI certificate screen."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth516_operations import Context, wire

N, D, H, E, K, BATCH = 17540, 768, 3072, 128, 64, 128
HEADER = struct.Struct('<8s4I')
RECORD_BYTES = K * 2 + H * K + H * 8
BASELINE = H * (D + 4)
INTRODUCED = H * K + H * 8 + K * 2 + D * 2 + K * 2 + H // 8 + 16 + HEADER.size
WORK = [('width', '<u2'), ('skipped', '<u2'), ('addressed_bytes', '<u4'), ('selected_query_norm2', '<u8')]


def summaries(np, m, occ, work, norms, index_bytes):
    dev, val = (m[:, 4] & 5) != 0, (m[:, 4] & 2) != 0

    def view(weights):
        at = np.flatnonzero(weights)
        total = int(weights[at].sum())
        if not total:
            return {'UIDs': 0, 'occurrences': 0, 'mean_byte_ratio': None, 'p95_byte_ratio': None}
        w = weights[at]
        ratios = work['addressed_bytes'][at].astype('<f8') / BASELINE
        width = work['width'][at].astype('<i8')
        energy = np.divide(work['selected_query_norm2'][at].astype('<f8'), norms[at],
                           out=np.zeros(len(at)), where=norms[at] != 0)
        return {'UIDs': len(at), 'occurrences': total,
                'mean_byte_ratio': float(np.sum(w * ratios) / total),
                'p95_byte_ratio': float(np.quantile(np.repeat(ratios, w), .95)),
                'mean_remaining_rows': float(np.sum(w * width) / total),
                'maximum_remaining_rows': int(width.max()),
                'unchanged_full_width_occurrences': int(w[width == H].sum()),
                'certified_row_occurrences': int(np.sum(w * (H - width))),
                'mean_selected_query_energy_fraction': float(np.sum(w * energy) / total),
                'prefix_integer_MACs': total * H * K,
                'fallback_integer_MACs': int(np.sum(w * width)) * D,
                'query_energy_integer_products': total * (D + K),
                'row_predicates_U64_products_and_squares_each': total * H,
                'coordinate_lookup_gathers': total * K, 'bitmap_words': total * (H // 64),
                'introduced_logical_bytes': total * INTRODUCED,
                'fallback_logical_bytes': int(np.sum(w * width)) * (D + 4)}

    views = {'development': view(dev.astype('<i8')), 'consumed': view(val.astype('<i8'))}
    for label, mode in [('natural_consumed', 1), ('teacher_consumed', 0)]:
        weights = np.bincount(occ[(occ[:, 7] == 1) & (occ[:, 6] == mode), 1], minlength=N)
        views[label] = view(weights)
    books = []
    for book in np.unique(occ[occ[:, 7] == 1, 4]):
        w = np.bincount(occ[(occ[:, 7] == 1) & (occ[:, 4] == book), 1], minlength=N)
        books.append({'book': int(book), **view(w)})
    assert len(books) == 64
    dev_counts = np.bincount(m[dev, 3], minlength=E)[m[:, 3]]
    rare = {label: view((val & flag).astype('<i8')) for label, flag in [
        ('dev_count1_4', (dev_counts >= 1) & (dev_counts <= 4)),
        ('dev_count5_15', (dev_counts >= 5) & (dev_counts <= 15))]}
    eligibility = {'consumed_mean_byte_ratio_le.75': views['consumed']['mean_byte_ratio'] <= .75,
                   'natural_mean_byte_ratio_le.75': views['natural_consumed']['mean_byte_ratio'] <= .75,
                   'natural_p95_byte_ratio_le1': views['natural_consumed']['p95_byte_ratio'] <= 1,
                   'all_consumed_book_mean_byte_ratio_le1': all(b['mean_byte_ratio'] <= 1 for b in books),
                   'complete_fallback_bank_plus_index_ratio_le1.20': (605945856 + index_bytes) * 5 <= 605945856 * 6}
    return views, books, rare, eligibility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-sha', required=True)
    args = ap.parse_args()
    ctx = None
    try:
        ctx = Context('main', args.binding_sha)
        np = ctx.numpy()
        m = wire(np, ctx.data('uid'), b'M493U001', 180, 0,
                 [('m', '<u4', (13,)), ('hash', 'u1', (128,))], N)['m']
        occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
        core = [('e', '<u4'), ('accept', '<u4'), ('alpha', '<f4'), ('x', '<f4', (D,)), ('q', '<i2', (D,))]
        inputs = wire(np, ctx.data('inputs'), b'M499INP1', 4624, D, [('core', core), ('p', '<f4')], N)
        q = inputs['core']['q']
        assert np.array_equal(m[:, 0], np.arange(N)) and np.all(m[:, 2] == 11) and np.all(m[:, 3] < E)
        assert np.array_equal(inputs['core']['e'], m[:, 3]) and np.all(inputs['core']['accept'] == 1)
        assert inputs['p'].tobytes() == m[:, 10].tobytes() and np.isfinite(inputs['p']).all() and np.all(inputs['p'] > 0)
        assert np.isfinite(inputs['core']['x']).all() and np.isfinite(inputs['core']['alpha']).all() and np.all(inputs['core']['alpha'] > 0)
        assert np.min(q) >= -32767 and np.max(q) <= 32767
        assert np.array_equal(occ[:, 0], np.arange(19962)) and np.all(occ[:, 12] == 1) and np.all(occ[:, 8] == 11)
        assert np.array_equal(np.bincount(occ[:, 1], minlength=N), m[:, 7])
        for oi, mi in [(14, 1), (11, 3), (15, 12), (16, 10)]:
            assert np.array_equal(occ[:, oi], m[occ[:, 1], mi])
        dev, val = (m[:, 4] & 5) != 0, (m[:, 4] & 2) != 0
        assert (int(dev.sum()), int(val.sum())) == (11721, 5819) and np.all(dev ^ val)
        norms = np.load(ctx.data('query_norm2.npy'), mmap_mode='r', allow_pickle=False)
        hidden, codes, scales = [np.load(ctx.data(k), mmap_mode='r', allow_pickle=False) for k in ['hidden', 'codes', 'scales']]
        assert norms.shape == (N,) and norms.dtype == np.dtype('<i8') and np.all(norms >= 0)
        assert hidden.shape == codes.shape == (N, H) and hidden.dtype == np.dtype('<f4') and codes.dtype == np.dtype('<i2')
        assert scales.shape == (N,) and scales.dtype == np.dtype('<f4')
        assert D * 128**2 * D * 32767**2 < 2**64 and K * 128 * 32767 < 2**28
        ctx.r['gates']['ALL17540_original_QUERY_hidden_roles_parent_p_and_occurrence_contracts'] = True
        masks = np.lib.format.open_memmap(ctx.out / 'omitted_masks.npy', mode='w+', dtype='u1', shape=(N, H // 8))
        work = np.lib.format.open_memmap(ctx.out / 'work.npy', mode='w+', dtype=WORK, shape=(N,))
        indexpath = ctx.out / 'prefix_index.bin'
        parents = []
        with indexpath.open('xb') as physical:
            physical.write(HEADER.pack(b'M516P001', E, D, H, K))
            for e in range(E):
                at = np.flatnonzero(m[:, 3] == e)
                train = np.flatnonzero((m[:, 3] == e) & dev)
                if len(train):
                    trainq = q[train].astype('<i8')
                    energy = np.sum(trainq * trainq, axis=0, dtype='<i8')
                    ids = np.lexsort((np.arange(D), -energy))[:K].astype('<u2')
                else:
                    ids = np.arange(K, dtype='<u2')
                original = ctx.weights(np, e)
                prefix = np.ascontiguousarray(original[:, ids], dtype='i1')
                prefix64 = prefix.astype('<i8')
                if e:
                    fullnorm = np.load(ctx.data(f'row_norm2_e{e:03}.npy'), allow_pickle=False)
                    assert fullnorm.shape == (H,) and fullnorm.dtype == np.dtype('<u4')
                    fullnorm = fullnorm.astype('<i8')
                else:
                    original32 = original.astype('<i4')
                    fullnorm = np.sum(original32 * original32, axis=1, dtype='<i8')
                residual = fullnorm - np.sum(prefix64 * prefix64, axis=1, dtype='<i8')
                assert np.all(residual >= 0) and np.all(residual <= D * 128**2)
                residual = residual.astype('<u8')
                row = ctx.b['parents'][e]['wi']
                row_scales = np.memmap(ctx.b['payload']['path'], mode='r', offset=row['scale_offset'], dtype='<f4', shape=(H,))
                assert np.isfinite(row_scales).all() and np.all(row_scales > 0)
                wo = ctx.b['parents'][e]['wo']
                wo_scales = np.memmap(ctx.b['payload']['path'], mode='r', offset=wo['scale_offset'], dtype='<f4', shape=(D,))
                assert np.isfinite(wo_scales).all() and np.all(wo_scales > 0)
                physical.write(ids.tobytes())
                physical.write(prefix.tobytes())
                physical.write(residual.tobytes())
                negative = certified = zero_hidden = 0
                prefix_float = prefix.astype('<f8').T
                for start in range(0, len(at), BATCH):
                    ix = at[start:start + BATCH]
                    query = q[ix][:, ids].astype('<i8')
                    selectednorm = np.sum(query * query, axis=1, dtype='<i8')
                    rest = norms[ix] - selectednorm
                    assert np.all(rest >= 0) and np.all(rest <= D * 32767**2)
                    exact_float = query.astype('<f8') @ prefix_float
                    assert np.array_equal(exact_float, np.rint(exact_float)) and np.max(np.abs(exact_float)) <= K * 128 * 32767
                    partial = exact_float.astype('<i8')
                    squared = (partial * partial).astype('<u8')
                    bound = rest.astype('<u8')[:, None] * residual[None, :]
                    reject = (partial <= 0) & (squared >= bound)
                    masks[ix] = np.packbits(reject, axis=1, bitorder='little')
                    skipped = np.sum(reject, axis=1, dtype='<i8')
                    width = H - skipped
                    work[ix] = np.rec.fromarrays([width, skipped, INTRODUCED + width * (D + 4), selectednorm], dtype=WORK)
                    h = hidden[ix]
                    modified = np.where(reject, np.float32(0), h)
                    assert np.isfinite(h).all() and np.all(h >= 0) and np.isfinite(scales[ix]).all() and np.all(scales[ix] > 0)
                    assert modified.tobytes() == h.tobytes()
                    maximum = np.max(np.abs(modified), axis=1)
                    assert maximum.tobytes() == np.max(np.abs(h), axis=1).tobytes()
                    alpha = (maximum / np.float32(32767)).astype('<f4')
                    alpha[maximum == 0] = 1
                    quantized = np.clip(np.rint(np.divide(modified, alpha[:, None], dtype=np.float32)), -32767, 32767).astype('<i2')
                    assert alpha.tobytes() == scales[ix].tobytes() and quantized.tobytes() == codes[ix].tobytes()
                    negative += int(np.sum(partial <= 0))
                    certified += int(skipped.sum())
                    zero_hidden += int(np.sum(h == 0))
                    ctx.guard()
                parents.append({'parent': e, 'UIDs': len(at), 'development_UIDs': len(train),
                                'no_development_uses_first64': not len(train), 'nonpositive_partial_cells': negative,
                                'certified_cells': certified, 'original_hidden_zero_cells': zero_hidden})
                if e % 16 == 15:
                    print(json.dumps({'main_parent_terminal': e, 'seconds': ctx.resources()['seconds']}), flush=True)
        masks.flush()
        work.flush()
        indexbytes = indexpath.stat().st_size
        assert indexbytes == HEADER.size + E * RECORD_BYTES == 28327960
        assert sum(v['UIDs'] for v in parents) == N
        ctx.r['gates'].update(ALL128_development_only_fixed64_original_physical_indices=True,
                              ALL17540_exact_U64_predicates_original_hidden_max_scale_codes_BYTE=True,
                              complete_original_fallback_storage_and_ALL_domain_costs=True)
        views, books, rare, eligible = summaries(np, m, occ, work, norms, indexbytes)
        ctx.finish({'index_bytes': indexbytes, 'original_fallback_bank_bytes': 605945856,
                    'baseline_WI_logical_bytes_per_UID': BASELINE, 'introduced_logical_bytes_per_UID': INTRODUCED,
                    'parents': parents, 'views': views, 'consumed_books': books, 'rare_consumed': rare,
                    'eligibility': eligible, 'new_prefix_integer_MACs': N * H * K,
                    'old_full_WI_WO_model_evaluations': 0, 'physical_DRAM_verified': False,
                    'decision': 'NEXT_NATIVE_EXACT_PARITY_AND_COMPLETE_COST' if all(eligible.values()) else 'CLOSE_THIS_FIXED64_QUERY_CERTIFICATE',
                    'logical_work_scope': '223400 introduced bytes +772 per unresolved row; all norm/prefix/predicate/gather/bitmap/fallback work charged. Original common WO/output excluded from both WI-only denominators, complete stored fallback included. Not measured DRAM, C latency or fresh quality.'})
    except BaseException:
        if ctx:
            ctx.fail()
        raise


if __name__ == '__main__':
    main()
