"""Independent physical inverse, I64 predicates, source-byte and occurrence audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth516_operations import Context, ROOT, DOC, wire

N, D, H = 17540, 768, 3072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-sha', required=True)
    ap.add_argument('--main-sha', required=True)
    args = ap.parse_args()
    ctx = None
    try:
        ctx = Context('audit', args.binding_sha)
        mainpath = DOC / 'meth516_main_result.json'
        assert ctx.digest(mainpath) == args.main_sha
        raw = json.loads(mainpath.read_bytes())
        assert raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        for v in raw['output_inventory']:
            assert ctx.digest(v['path']) == v['sha256']
        np = ctx.numpy()
        m = wire(np, ctx.data('uid'), b'M493U001', 180, 0,
                 [('m', '<u4', (13,)), ('hash', 'u1', (128,))], N)['m']
        occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
        core = [('e', '<u4'), ('accept', '<u4'), ('alpha', '<f4'), ('x', '<f4', (D,)), ('q', '<i2', (D,))]
        source_inputs = wire(np, ctx.data('inputs'), b'M499INP1', 4624, D, [('core', core), ('p', '<f4')], N)
        q = source_inputs['core']['q']
        assert np.array_equal(m[:, 0], np.arange(N)) and np.all(m[:, 2] == 11) and np.all(m[:, 3] < 128)
        assert np.array_equal(source_inputs['core']['e'], m[:, 3]) and np.all(source_inputs['core']['accept'] == 1)
        assert source_inputs['p'].tobytes() == m[:, 10].tobytes()
        assert np.isfinite(source_inputs['p']).all() and np.all(source_inputs['p'] > 0)
        assert np.isfinite(source_inputs['core']['x']).all() and np.isfinite(source_inputs['core']['alpha']).all() and np.all(source_inputs['core']['alpha'] > 0)
        assert int(q.min()) >= -32767 and int(q.max()) <= 32767
        assert np.array_equal(occ[:, 0], np.arange(19962)) and np.all(occ[:, 8] == 11) and np.all(occ[:, 12] == 1)
        assert np.array_equal(np.bincount(occ[:, 1], minlength=N), m[:, 7])
        for a, b in [(14, 1), (11, 3), (15, 12), (16, 10)]:
            assert np.array_equal(occ[:, a], m[occ[:, 1], b])
        dev, val = (m[:, 4] & 5) != 0, (m[:, 4] & 2) != 0
        assert np.all(dev ^ val) and (int(dev.sum()), int(val.sum())) == (11721, 5819)
        folder = ROOT / 'results/native_expert_scaling/meth516_index'
        data = (folder / 'prefix_index.bin').read_bytes()
        assert struct.unpack_from('<8sIIII', data) == (b'M516P001', 128, D, H, 64)
        assert len(data) == 24 + 128 * (128 + H * 64 + H * 8) == raw['index_bytes']
        masks = np.load(folder / 'omitted_masks.npy', mmap_mode='r', allow_pickle=False)
        work = np.load(folder / 'work.npy', mmap_mode='r', allow_pickle=False)
        normcache = np.load(ctx.data('query_norm2.npy'), mmap_mode='r', allow_pickle=False)
        hidden, codes, scales = [np.load(ctx.data(k), mmap_mode='r', allow_pickle=False) for k in ['hidden', 'codes', 'scales']]
        assert masks.shape == (N, H // 8) and masks.dtype == np.dtype('u1')
        expected_dtype = np.dtype([('width', '<u2'), ('skipped', '<u2'), ('addressed_bytes', '<u4'), ('selected_query_norm2', '<u8')])
        assert work.shape == (N,) and work.dtype == expected_dtype
        assert normcache.shape == (N,) and normcache.dtype == np.dtype('<i8')
        assert hidden.shape == codes.shape == (N, H) and hidden.dtype == np.dtype('<f4') and codes.dtype == np.dtype('<i2')
        assert scales.shape == (N,) and scales.dtype == np.dtype('<f4')
        controls = 0
        for wx in range(-2, 3):
            for wy in range(-2, 3):
                for qx in range(-6, 7):
                    for qy in range(-6, 7):
                        a = wx * qx
                        reject = a <= 0 and a * a >= wy * wy * qy * qy
                        if reject:
                            assert wx * qx + wy * qy <= 0
                        controls += 1
        assert (-6)**2 == 2**2 * 3**2 and -6 + 2 * 3 == 0
        worst = (768 * 128**2) * (768 * 32767**2)
        assert 2**63 < worst < 2**64
        assert int(np.uint64(768 * 128**2) * np.uint64(768 * 32767**2)) == worst
        assert int(np.uint64((64 * 128 * 32767)**2)) == (64 * 128 * 32767)**2 < 2**56
        ctx.r['gates']['independent_unbounded_integer_zero_sign_equality_and_U64_extreme_controls'] = True
        rebuilt_width = np.empty(N, dtype='<i8')
        rebuilt_selectednorm = np.empty(N, dtype='<u8')
        rebuilt_norm = np.empty(N, dtype='<i8')
        parents = []
        cursor = 24
        for e in range(128):
            ids = np.frombuffer(data, dtype='<u2', count=64, offset=cursor)
            cursor += 128
            prefix = np.frombuffer(data, dtype='i1', count=H * 64, offset=cursor).reshape(H, 64)
            cursor += H * 64
            complement = np.frombuffer(data, dtype='<u8', count=H, offset=cursor)
            cursor += H * 8
            train = np.flatnonzero(dev & (m[:, 3] == e))
            t = q[train].astype('<i8')
            energies = [int(np.dot(t[:, j], t[:, j])) for j in range(D)]
            expected_ids = sorted(range(D), key=lambda j: (-energies[j], j))[:64] if len(train) else list(range(64))
            assert ids.tolist() == expected_ids and len(set(ids.tolist())) == 64
            w = ctx.weights(np, e).astype('<i8')
            assert prefix.tobytes() == w[:, expected_ids].astype('i1').tobytes()
            rownorm = np.einsum('ij,ij->i', w, w, dtype='<i8')
            comp = rownorm - np.einsum('ij,ij->i', w[:, expected_ids], w[:, expected_ids], dtype='<i8')
            assert np.all(comp >= 0) and comp.astype('<u8').tobytes() == complement.tobytes()
            if e:
                assert rownorm.astype('<u4').tobytes() == np.load(ctx.data(f'row_norm2_e{e:03}.npy'), allow_pickle=False).tobytes()
            for op, count in [('wi', H), ('wo', D)]:
                row = ctx.b['parents'][e][op]
                s = np.memmap(ctx.b['payload']['path'], mode='r', offset=row['scale_offset'], dtype='<f4', shape=(count,))
                assert np.isfinite(s).all() and np.all(s > 0)
            at = np.flatnonzero(m[:, 3] == e)
            negative = certified = zero_hidden = 0
            for start in range(0, len(at), 128):
                ix = at[start:start + 128]
                x = q[ix].astype('<i8')
                totalq = np.einsum('ij,ij->i', x, x, dtype='<i8')
                sq = x[:, expected_ids]
                selectedq = np.einsum('ij,ij->i', sq, sq, dtype='<i8')
                assert np.all(totalq >= selectedq) and totalq.tobytes() == normcache[ix].tobytes()
                partial = sq @ w[:, expected_ids].T
                assert np.all(np.abs(partial) <= 64 * 128 * 32767)
                rhs = np.multiply(complement[None, :], (totalq - selectedq).astype('<u8')[:, None], dtype='<u8')
                lhs = np.square(partial, dtype='<i8').astype('<u8')
                reject = np.logical_and(np.logical_not(partial > 0), np.logical_not(lhs < rhs))
                assert np.packbits(reject, axis=1, bitorder='little').tobytes() == masks[ix].tobytes()
                skips = np.count_nonzero(reject, axis=1)
                widths = H - skips
                # Separate byte-price reconstruction from stored work and main's constants.
                introduced = 24 + 2 * 64 + H * 64 + 8 * H + 2 * 768 + 2 * 64 + H // 8 + expected_dtype.itemsize
                expected = np.empty(len(ix), dtype=expected_dtype)
                expected['width'], expected['skipped'] = widths, skips
                expected['addressed_bytes'] = introduced + widths * 768 + widths * 4
                expected['selected_query_norm2'] = selectedq
                assert expected.tobytes() == work[ix].tobytes()
                rebuilt_width[ix], rebuilt_selectednorm[ix], rebuilt_norm[ix] = widths, selectedq, totalq
                h = hidden[ix].copy()
                assert np.isfinite(h).all() and np.all(h >= 0) and np.isfinite(scales[ix]).all() and np.all(scales[ix] > 0)
                h[reject] = np.float32(0)
                assert h.tobytes() == hidden[ix].tobytes()
                maximum = np.max(np.abs(h), axis=1)
                assert maximum.tobytes() == np.max(np.abs(hidden[ix]), axis=1).tobytes()
                scale = maximum / np.float32(32767)
                scale[maximum == 0] = 1
                assert scale.astype('<f4').tobytes() == scales[ix].tobytes()
                quantized = np.rint(h / scale[:, None]).clip(-32767, 32767).astype('<i2')
                assert quantized.tobytes() == codes[ix].tobytes()
                negative += int(np.count_nonzero(partial <= 0))
                certified += int(skips.sum())
                zero_hidden += int(np.count_nonzero(h == 0))
                ctx.guard()
            parents.append({'parent': e, 'UIDs': len(at), 'development_UIDs': len(train),
                            'no_development_uses_first64': not len(train), 'nonpositive_partial_cells': negative,
                            'certified_cells': certified, 'original_hidden_zero_cells': zero_hidden})
            if e % 16 == 15:
                print(json.dumps({'audit_parent_terminal': e, 'seconds': ctx.resources()['seconds']}), flush=True)
        assert cursor == len(data) and parents == raw['parents']
        ctx.r['gates'].update(ALL128_physical_inverse_development_selection_full_and_complement_norms=True,
                              ALL53882880_I64_prefix_cells_predicates_masks_and_work_independent=True,
                              ALL17540_original_hidden_max_scale_codes_parent_p_WO_BYTE_identity=True)
        introduced = 223400
        baseline = 2371584

        def view(weights):
            ids = np.flatnonzero(weights)
            count = int(weights[ids].sum())
            if count == 0:
                return {'UIDs': 0, 'occurrences': 0, 'mean_byte_ratio': None, 'p95_byte_ratio': None}
            weight = weights[ids]
            widths = rebuilt_width[ids]
            ratios = (introduced + widths * 772).astype('<f8') / baseline
            fraction = np.divide(rebuilt_selectednorm[ids].astype('<f8'), rebuilt_norm[ids],
                                 out=np.zeros(len(ids)), where=rebuilt_norm[ids] != 0)
            return {'UIDs': len(ids), 'occurrences': count,
                    'mean_byte_ratio': float(np.sum(weight * ratios) / count),
                    'p95_byte_ratio': float(np.quantile(np.repeat(ratios, weight), .95)),
                    'mean_remaining_rows': float(np.sum(weight * widths) / count),
                    'maximum_remaining_rows': int(widths.max()),
                    'unchanged_full_width_occurrences': int(weight[widths == H].sum()),
                    'certified_row_occurrences': int(np.sum(weight * (3072 - widths))),
                    'mean_selected_query_energy_fraction': float(np.sum(weight * fraction) / count),
                    'prefix_integer_MACs': count * 3072 * 64,
                    'fallback_integer_MACs': int(np.sum(weight * widths)) * 768,
                    'query_energy_integer_products': count * 832,
                    'row_predicates_U64_products_and_squares_each': count * 3072,
                    'coordinate_lookup_gathers': count * 64, 'bitmap_words': count * 48,
                    'introduced_logical_bytes': count * 223400,
                    'fallback_logical_bytes': int(np.sum(weight * widths)) * 772}

        views = {'development': view(dev.astype('<i8')), 'consumed': view(val.astype('<i8'))}
        for name, mode in [('natural_consumed', 1), ('teacher_consumed', 0)]:
            weights = np.zeros(N, dtype='<i8')
            for row in occ:
                if row[7] == 1 and row[6] == mode:
                    weights[row[1]] += 1
            views[name] = view(weights)
        books = []
        for book in sorted({int(row[4]) for row in occ if row[7] == 1}):
            weights = np.zeros(N, dtype='<i8')
            for row in occ:
                if row[7] == 1 and row[4] == book:
                    weights[row[1]] += 1
            books.append({'book': book, **view(weights)})
        assert len(books) == 64
        devcount = [int(np.count_nonzero(dev & (m[:, 3] == e))) for e in range(128)]
        rare = {}
        for label, low, high in [('dev_count1_4', 1, 4), ('dev_count5_15', 5, 15)]:
            weights = np.array([int(bool(val[i]) and low <= devcount[int(m[i, 3])] <= high) for i in range(N)], dtype='<i8')
            rare[label] = view(weights)
        eligibility = {'consumed_mean_byte_ratio_le.75': views['consumed']['mean_byte_ratio'] <= .75,
                       'natural_mean_byte_ratio_le.75': views['natural_consumed']['mean_byte_ratio'] <= .75,
                       'natural_p95_byte_ratio_le1': views['natural_consumed']['p95_byte_ratio'] <= 1,
                       'all_consumed_book_mean_byte_ratio_le1': all(b['mean_byte_ratio'] <= 1 for b in books),
                       'complete_fallback_bank_plus_index_ratio_le1.20': 5 * (605945856 + len(data)) <= 6 * 605945856}
        assert (views, books, rare, eligibility) == (raw['views'], raw['consumed_books'], raw['rare_consumed'], raw['eligibility'])
        decision = 'NEXT_NATIVE_EXACT_PARITY_AND_COMPLETE_COST' if all(eligibility.values()) else 'CLOSE_THIS_FIXED64_QUERY_CERTIFICATE'
        assert decision == raw['decision'] and raw['new_prefix_integer_MACs'] == N * H * 64 == 3448504320
        assert raw['original_fallback_bank_bytes'] == 605945856 and raw['baseline_WI_logical_bytes_per_UID'] == baseline
        assert raw['introduced_logical_bytes_per_UID'] == introduced
        ctx.r['gates']['ALL_views_64_books_rare_and_frozen_economic_decisions_independent'] = True
        ctx.finish({'main_sha256': args.main_sha, 'views': views, 'consumed_books': books, 'rare_consumed': rare,
                    'eligibility': eligibility, 'decision': decision, 'index_bytes': len(data),
                    'unique_inputs_audited': N, 'physical_parents_audited': 128,
                    'partial_cells_audited': N * H, 'integer_prefix_MACs': N * H * 64,
                    'tiny_controls_checked': controls, 'worst_U64_product': worst,
                    'old_full_WI_WO_model_evaluations': 0, 'physical_DRAM_verified': False})
    except BaseException:
        if ctx:
            ctx.fail()
        raise


if __name__ == '__main__':
    main()
