"""513 independent Fraction ordering/full suffixes/physical bytes/query cost audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
from bisect import bisect_right
from fractions import Fraction
import json
from pathlib import Path
import struct
import traceback

from meth513_operations import Context, DOC, wire, write

N, D, H = 17540, 768, 3072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-sha', required=True)
    ap.add_argument('--main-sha', required=True)
    a = ap.parse_args()
    ctx = None
    try:
        ctx = Context('audit', a.binding_sha)
        rp = DOC / 'meth513_main_result.json'
        assert ctx.digest(rp) == a.main_sha
        raw = json.loads(rp.read_bytes())
        assert all(raw['gates'].values())
        for v in raw['output_inventory']:
            assert ctx.digest(v['path']) == v['sha256']
        np = ctx.numpy()
        folder = ctx.out.parent / 'meth513_index'
        m = wire(np, ctx.data('uid'), b'M493U001', 180, 0, [('m', '<u4', (13,)), ('hash', 'u1', (128,))], N)['m']
        occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
        norm = np.load(ctx.data('query_norm2.npy'), mmap_mode='r', allow_pickle=False)
        hidden, codes, scales = [np.load(ctx.data(k), mmap_mode='r', allow_pickle=False) for k in ['hidden', 'codes', 'scales']]
        mask = np.load(folder / 'omitted_masks.npy', mmap_mode='r', allow_pickle=False)
        work = np.load(folder / 'work.npy', mmap_mode='r', allow_pickle=False)
        assert mask.shape == (N, 384) and work.shape == (N,)
        assert np.array_equal(m[:, 0], np.arange(N)) and np.all(m[:, 2] == 11) and np.all(m[:, 3] < 128)
        assert np.array_equal(np.bincount(occ[:, 1], minlength=N), m[:, 7])
        dev, val = (m[:, 4] & 5) != 0, (m[:, 4] & 2) != 0
        assert (int(dev.sum()), int(val.sum())) == (11721, 5819) and np.all(dev ^ val)
        # Independent square-root-free geometric check across exhaustive integer controls.
        tested = 0
        for ax, ay in [(3, 0), (-2, 1)]:
            A = ax * ax + ay * ay
            for wx in range(-2, 3):
                for wy in range(-2, 3):
                    R, d = wx * wx + wy * wy, ax * wx + ay * wy
                    for qx in range(-6, 7):
                        for qy in range(-6, 7):
                            Q, t = qx * qx + qy * qy, ax * qx + ay * qy
                            hit = Q == 0 or R == 0 or (d * t < 0 and Fraction(d * d, R) > Fraction(A * Q - t * t, Q))
                            if hit:
                                assert wx * qx + wy * qy <= 0
                            tested += 1
        assert Fraction(144, 25) == Fraction(9 * 25 - 81, 25)
        ctx.r['gates']['independent_Fraction_tiny_sign_boundary_controls_and_original_roles'] = True
        prior = json.loads(Path(ctx.b['retained504_raw']['path']).read_bytes())
        reconstructed = np.empty(work.shape, dtype=work.dtype)
        reconstructed_masks = np.empty(mask.shape, dtype='u1')
        index_bytes = centres = checkpoints_checked = 0
        for e in range(128):
            at = np.flatnonzero(m[:, 3] == e)
            payload = (folder / f'e{e:03}.index.bin').read_bytes()
            index_bytes += len(payload)
            magic, parent, width, hidden_width, count, stride = struct.unpack_from('<8sHHHHI', payload)
            assert (magic, parent, width, hidden_width, stride) == (b'M513IDX1', e, D, H, 64)
            if not len(at):
                assert e == 0 and count == 0 and payload[20:] == bytes(384) and len(payload) == 404
                continue
            seeds = np.load(ctx.data(f'seed_codes_e{e:03}.npy'), allow_pickle=False)
            d = np.load(ctx.data(f'centre_dots_e{e:03}.npy'), allow_pickle=False)
            R = np.load(ctx.data(f'row_norm2_e{e:03}.npy'), allow_pickle=False)
            query = np.load(ctx.data(f'query_dots_e{e:03}.npy'), allow_pickle=False)
            assert count == len(seeds) == len(prior['experts'][e]['regions'])
            zero = sum(2 ** int(j) for j in np.flatnonzero(R == 0))
            assert payload[20:404] == zero.to_bytes(384, 'little')
            cursor = 404
            indices = []
            for c in range(count):
                uid, A, neg, pos = struct.unpack_from('<IQII', payload, cursor)
                cursor += 20
                assert uid == prior['experts'][e]['regions'][c]['seed_UID'] and dev[uid] and m[uid, 3] == e
                assert A == int(np.dot(seeds[c].astype('<i8'), seeds[c].astype('<i8'))) == int(norm[uid]) and A > 0
                assert payload[cursor:cursor + 1536] == seeds[c].astype('<i2').tobytes()
                cursor += 1536
                lists = []
                for sign, length in [(-1, neg), (1, pos)]:
                    expected = [(Fraction(int(d[c, j]) ** 2, int(R[j])), j) for j in range(H) if R[j] and int(d[c, j]) * sign > 0]
                    expected.sort()
                    ids = [j for _, j in expected]
                    assert len(ids) == length
                    for j in ids:
                        record = struct.unpack_from('<HQI', payload, cursor)
                        assert record == (j, int(d[c, j]) ** 2, int(R[j]))
                        cursor += 14
                    # ALL-position suffixes are independent of the sparse main checkpoint algorithm.
                    suffix = [0] * (length + 1)
                    for i in range(length - 1, -1, -1):
                        suffix[i] = suffix[i + 1] | (2 ** ids[i])
                    for slot in range((length + 63) // 64 + 1):
                        start = min(64 * slot, length)
                        assert payload[cursor:cursor + 384] == suffix[start].to_bytes(384, 'little')
                        cursor += 384
                        checkpoints_checked += 1
                    lists.append(([ratio for ratio, _ in expected], ids, suffix))
                indices.append((A, lists))
            assert cursor == len(payload)
            cuts = np.load(folder / f'e{e:03}.cuts.npy', mmap_mode='r', allow_pickle=False)
            assert cuts.shape == query.shape
            for local, k in enumerate(at):
                Q = int(norm[k])
                bits, cmps, edge, read, words, used = zero, 0, 0, 1940, 48, 0
                if Q == 0:
                    bits = (2 ** H) - 1
                    assert np.all(cuts[local] == H + 1)
                else:
                    for c, (A, lists) in enumerate(indices):
                        t = int(query[local, c])
                        delta = A * Q - t * t
                        assert delta >= 0
                        used += 1
                        read += 1556
                        if t == 0:
                            assert cuts[local, c] == H + 1
                            continue
                        ratios, ids, suffix = lists[0 if t > 0 else 1]
                        threshold = Fraction(delta, Q)
                        cut = bisect_right(ratios, threshold)
                        assert cut == cuts[local, c]
                        bits |= suffix[cut]
                        # Reconstruct exact search path separately for its charged work.
                        lower, upper = 0, len(ratios)
                        while lower < upper:
                            middle = (lower + upper) // 2
                            cmps += 1
                            read += 14
                            if ratios[middle] <= threshold:
                                lower = middle + 1
                            else:
                                upper = middle
                        assert lower == cut
                        if cut != len(ids):
                            edge_count = min(((cut + 63) // 64) * 64, len(ids)) - cut
                            edge += edge_count
                            read += 384 + 14 * edge_count
                            words += 48
                remaining = H - bits.bit_count()
                reconstructed[k] = (remaining, cmps, edge, used, read + remaining * 772, words)
                reconstructed_masks[k] = np.frombuffer(bits.to_bytes(384, 'little'), dtype='u1')
            assert reconstructed[at].tobytes() == work[at].tobytes()
            assert reconstructed_masks[at].tobytes() == mask[at].tobytes()
            flags = np.unpackbits(reconstructed_masks[at], axis=1, bitorder='little').astype(bool)
            assert np.all(hidden[at][flags] == 0) and np.all(codes[at][flags] == 0)
            # Every removed value is zero, hence the original maximum/scales/remaining codes survive.
            masked = np.where(flags, np.float32(0), hidden[at])
            assert masked.tobytes() == hidden[at].tobytes()
            maximum = np.max(masked, axis=1)
            alpha = (maximum / np.float32(32767)).astype('<f4')
            alpha[maximum == 0] = 1
            assert alpha.tobytes() == scales[at].tobytes()
            assert np.rint(masked / alpha[:, None]).astype('<i2').tobytes() == codes[at].tobytes()
            centres += count
            ctx.guard()
            if e % 16 == 15:
                print(json.dumps({'audit_expert_terminal': e, 'seconds': ctx.resources()['seconds']}), flush=True)
        assert centres == raw['centres'] == 637 and index_bytes == raw['index_bytes']
        ctx.r['gates'].update(ALL637_physical_headers_entries_suffix_checkpoints_Fraction_sorted=True,
                              ALL17540_independent_masks_cut_boundaries_and_exact_work=True,
                              ALL17540_original_hidden_scale_quantized_codes_preserved_BYTE=True)
        baseline = 2371584
        def summary(w):
            at = np.flatnonzero(w)
            total = int(w[at].sum())
            if not total:
                return {'UIDs': 0, 'occurrences': 0, 'mean_byte_ratio': None, 'p95_byte_ratio': None}
            ratio = reconstructed['addressed_bytes'][at].astype('<f8') / baseline
            return {'UIDs': len(at), 'occurrences': total,
                    'mean_byte_ratio': float(np.sum(w[at] * ratio) / total),
                    'p95_byte_ratio': float(np.quantile(np.repeat(ratio, w[at]), .95)),
                    'mean_remaining_rows': float(np.sum(w[at] * reconstructed['width'][at].astype('<i8')) / total),
                    'maximum_remaining_rows': int(reconstructed['width'][at].max()),
                    'unchanged_full_width_occurrences': int(w[at][reconstructed['width'][at] == H].sum()),
                    'binary_search_comparisons': int(np.sum(w[at] * reconstructed['comparisons'][at].astype('<i8'))),
                    'explicit_boundary_records': int(np.sum(w[at] * reconstructed['boundary'][at].astype('<i8'))),
                    'bitmap_union_words': int(np.sum(w[at] * reconstructed['bitmap_words'][at].astype('<i8'))),
                    'centre_dot_I16_products': int(np.sum(w[at] * reconstructed['centres'][at].astype('<i8'))) * 768,
                    'query_norm_I16_products': total * 768}
        v = {'development': summary(dev.astype('<i8')), 'consumed': summary(val.astype('<i8'))}
        for label, mode in [('natural_consumed', 1), ('teacher_consumed', 0)]:
            v[label] = summary(np.bincount(occ[(occ[:, 7] == 1) & (occ[:, 6] == mode), 1], minlength=N))
        books = []
        for book in np.unique(occ[occ[:, 7] == 1, 4]):
            weights = np.bincount(occ[(occ[:, 7] == 1) & (occ[:, 4] == book), 1], minlength=N)
            books.append({'book': int(book), **summary(weights)})
        dev_counts = np.bincount(m[dev, 3], minlength=128)[m[:, 3]]
        rare = {label: summary((val & flag).astype('<i8')) for label, flag in [
            ('dev_count1_4', (dev_counts >= 1) & (dev_counts <= 4)), ('dev_count5_15', (dev_counts >= 5) & (dev_counts <= 15))]}
        eligibility = {'consumed_mean_byte_ratio_le.75': v['consumed']['mean_byte_ratio'] <= .75,
                       'natural_mean_byte_ratio_le.75': v['natural_consumed']['mean_byte_ratio'] <= .75,
                       'natural_p95_byte_ratio_le1': v['natural_consumed']['p95_byte_ratio'] <= 1,
                       'all_consumed_book_mean_byte_ratio_le1': all(b['mean_byte_ratio'] <= 1 for b in books),
                       'complete_fallback_bank_plus_index_ratio_le1.20': Fraction(605945856 + index_bytes, 605945856) <= Fraction(6, 5)}
        assert (v, books, rare, eligibility) == (raw['views'], raw['consumed_books'], raw['rare_consumed'], raw['eligibility'])
        assert raw['decision'] == ('NEXT_NATIVE_EXACT_PARITY_AND_COMPLETE_COST' if all(eligibility.values()) else 'CLOSE_THIS_ADAPTIVE_UNION_INDEX')
        ctx.r['gates']['ALL_domain_book_rare_and_frozen_eligibility_independent'] = True
        ctx.finish({'main_sha256': a.main_sha, 'views': v, 'consumed_books': books, 'rare_consumed': rare,
                    'eligibility': eligibility, 'decision': raw['decision'], 'index_bytes': index_bytes,
                    'unique_inputs_audited': N, 'physical_checkpoints_audited': checkpoints_checked,
                    'controls_checked': tested, 'physical_DRAM_verified': False})
    except BaseException:
        if ctx:
            ctx.fail()
        else:
            p = DOC / 'meth513_audit_result.failure.json'
            if not p.exists():
                write(p, {'phase': 'pre-numerical context admission', 'traceback': traceback.format_exc(), 'new_numerical_model_C_calls': 0})
        raise


if __name__ == '__main__':
    main()
