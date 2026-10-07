"""513 physical fixed64 sign-list suffix index and exact per-row certificate unions."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
from functools import cmp_to_key
import json
from pathlib import Path
import struct
import traceback

from meth513_operations import Context, DOC, wire, write

N, D, H, STRIDE = 17540, 768, 3072, 64
HEADER = struct.Struct('<8sHHHHI')
CENTRE = struct.Struct('<IQII')
ENTRY = struct.Struct('<HQI')
MASK_BYTES = H // 8
FULL = (1 << H) - 1


def controls():
    checks = 0
    for ax, ay in [(3, 0), (-2, 1)]:
        A = ax * ax + ay * ay
        for wx in range(-2, 3):
            for wy in range(-2, 3):
                R, d = wx * wx + wy * wy, wx * ax + wy * ay
                for qx in range(-6, 7):
                    for qy in range(-6, 7):
                        Q, t = qx * qx + qy * qy, ax * qx + ay * qy
                        delta = A * Q - t * t
                        assert delta >= 0
                        certified = Q == 0 or R == 0 or d * t < 0 and d * d * Q > R * delta
                        if certified:
                            assert wx * qx + wy * qy <= 0
                        checks += 1
    assert not (144 * 25 > 25 * (9 * 25 - 9 * 9))  # strict boundary
    big = (768 * 128 * 32767) ** 2 * (768 * 32767 ** 2)
    assert 63 < big.bit_length() < 104
    return {'exhaustive_2D_sign_tests': checks, 'strict_boundary_and_large_products': True}


def views(np, m, occ, work, index_bytes):
    baseline = H * (D + 4)
    def stats(weights):
        at = np.flatnonzero(weights)
        total = int(weights[at].sum())
        if not total:
            return {'UIDs': 0, 'occurrences': 0, 'mean_byte_ratio': None, 'p95_byte_ratio': None}
        values = work['addressed_bytes'][at].astype('<f8') / baseline
        expanded = np.repeat(values, weights[at])
        return {'UIDs': len(at), 'occurrences': total,
                'mean_byte_ratio': float(np.sum(values * weights[at]) / total),
                'p95_byte_ratio': float(np.quantile(expanded, .95)),
                'mean_remaining_rows': float(np.sum(work['width'][at].astype('<i8') * weights[at]) / total),
                'maximum_remaining_rows': int(work['width'][at].max()),
                'unchanged_full_width_occurrences': int(weights[at][work['width'][at] == H].sum()),
                'binary_search_comparisons': int(np.sum(work['comparisons'][at].astype('<i8') * weights[at])),
                'explicit_boundary_records': int(np.sum(work['boundary'][at].astype('<i8') * weights[at])),
                'bitmap_union_words': int(np.sum(work['bitmap_words'][at].astype('<i8') * weights[at])),
                'centre_dot_I16_products': int(np.sum(work['centres'][at].astype('<i8') * weights[at])) * D,
                'query_norm_I16_products': total * D}
    dev, val = (m[:, 4] & 5) != 0, (m[:, 4] & 2) != 0
    v = {name: stats(flag.astype('<i8')) for name, flag in [('development', dev), ('consumed', val)]}
    for name, mode in [('natural_consumed', 1), ('teacher_consumed', 0)]:
        weights = np.bincount(occ[(occ[:, 7] == 1) & (occ[:, 6] == mode), 1], minlength=N)
        v[name] = stats(weights)
    books = []
    for book in sorted(set(int(x) for x in occ[occ[:, 7] == 1, 4])):
        weights = np.bincount(occ[(occ[:, 7] == 1) & (occ[:, 4] == book), 1], minlength=N)
        books.append({'book': book, **stats(weights)})
    counts = np.bincount(m[dev, 3], minlength=128)[m[:, 3]]
    rare = {name: stats((flag & val).astype('<i8')) for name, flag in [
        ('dev_count1_4', (counts >= 1) & (counts <= 4)), ('dev_count5_15', (counts >= 5) & (counts <= 15))]}
    original = 128 * (2 * H * D + 4 * (H + D))
    eligibility = {'consumed_mean_byte_ratio_le.75': v['consumed']['mean_byte_ratio'] <= .75,
                   'natural_mean_byte_ratio_le.75': v['natural_consumed']['mean_byte_ratio'] <= .75,
                   'natural_p95_byte_ratio_le1': v['natural_consumed']['p95_byte_ratio'] <= 1,
                   'all_consumed_book_mean_byte_ratio_le1': all(b['mean_byte_ratio'] <= 1 for b in books),
                   'complete_fallback_bank_plus_index_ratio_le1.20': (original + index_bytes) / original <= 1.20}
    return v, books, rare, eligibility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-sha', required=True)
    a = ap.parse_args()
    ctx = None
    try:
        ctx = Context('main', a.binding_sha)
        np = ctx.numpy()
        write(ctx.out / 'controls.json', controls())
        m = wire(np, ctx.data('uid'), b'M493U001', 180, 0, [('m', '<u4', (13,)), ('hash', 'u1', (128,))], N)['m']
        occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
        assert np.array_equal(m[:, 0], np.arange(N)) and np.all(m[:, 2] == 11) and np.all(m[:, 3] < 128)
        assert np.array_equal(np.bincount(occ[:, 1], minlength=N), m[:, 7])
        dev, val = (m[:, 4] & 5) != 0, (m[:, 4] & 2) != 0
        assert (int(dev.sum()), int(val.sum())) == (11721, 5819) and np.all(dev ^ val)
        hidden, codes, scales = [np.load(ctx.data(k), mmap_mode='r', allow_pickle=False) for k in ['hidden', 'codes', 'scales']]
        norm = np.load(ctx.data('query_norm2.npy'), mmap_mode='r', allow_pickle=False)
        prior = json.loads(Path(ctx.b['retained504_raw']['path']).read_bytes())
        omitted = np.lib.format.open_memmap(ctx.out / 'omitted_masks.npy', mode='w+', dtype='u1', shape=(N, MASK_BYTES))
        dtype = [('width', '<u2'), ('comparisons', '<u2'), ('boundary', '<u2'), ('centres', '<u2'), ('addressed_bytes', '<u4'), ('bitmap_words', '<u4')]
        work = np.lib.format.open_memmap(ctx.out / 'work.npy', mode='w+', dtype=dtype, shape=(N,))
        all_bytes = centres = 0
        ctx.r['gates']['original_UID_roles_occurrence_join_and_tiny_strict_integer_controls'] = True
        for e in range(128):
            at = np.flatnonzero(m[:, 3] == e)
            if not len(at):
                assert e == 0
                raw = HEADER.pack(b'M513IDX1', e, D, H, 0, STRIDE) + bytes(MASK_BYTES)
                (ctx.out / f'e{e:03}.index.bin').write_bytes(raw)
                all_bytes += len(raw)
                continue
            seed = np.load(ctx.data(f'seed_codes_e{e:03}.npy'), allow_pickle=False)
            dots = np.load(ctx.data(f'centre_dots_e{e:03}.npy'), allow_pickle=False)
            R = np.load(ctx.data(f'row_norm2_e{e:03}.npy'), allow_pickle=False)
            query = np.load(ctx.data(f'query_dots_e{e:03}.npy'), allow_pickle=False)
            assert seed.shape == (len(prior['experts'][e]['regions']), D) and dots.shape == (len(seed), H) and query.shape == (len(at), len(seed))
            zbits = sum(1 << int(j) for j in np.flatnonzero(R == 0))
            buffer = bytearray(HEADER.pack(b'M513IDX1', e, D, H, len(seed), STRIDE) + zbits.to_bytes(MASK_BYTES, 'little'))
            compiled = []
            for c, region in enumerate(prior['experts'][e]['regions']):
                uid, A = region['seed_UID'], int(region['A'])
                assert dev[uid] and m[uid, 3] == e and A == int(norm[uid]) == int(np.sum(seed[c].astype('<i8') ** 2)) and A > 0
                d = dots[c]
                def cmp(j, k):
                    x, y = int(d[j]) ** 2 * int(R[k]), int(d[k]) ** 2 * int(R[j])
                    return (x > y) - (x < y) or (j > k) - (j < k)
                orders = [sorted([j for j in range(H) if R[j] and (d[j] < 0 if sign == -1 else d[j] > 0)], key=cmp_to_key(cmp)) for sign in [-1, 1]]
                buffer += CENTRE.pack(uid, A, len(orders[0]), len(orders[1])) + seed[c].astype('<i2').tobytes()
                lists = []
                for order in orders:
                    triples = [(j, int(d[j]) ** 2, int(R[j])) for j in order]
                    for triple in triples:
                        buffer += ENTRY.pack(*triple)
                    checkpoints = [0] * ((len(order) + STRIDE - 1) // STRIDE + 1)
                    bits = 0
                    for i in range(len(order) - 1, -1, -1):
                        bits |= 1 << order[i]
                        if i % STRIDE == 0:
                            checkpoints[i // STRIDE] = bits
                    for bits in checkpoints:
                        buffer += bits.to_bytes(MASK_BYTES, 'little')
                    lists.append((triples, checkpoints))
                compiled.append((A, lists))
            path = ctx.out / f'e{e:03}.index.bin'
            path.write_bytes(buffer)
            loaded = path.read_bytes()
            assert loaded == buffer
            cursor = HEADER.size + MASK_BYTES
            compiled = []
            zbits = int.from_bytes(loaded[HEADER.size:cursor], 'little')
            for c in range(len(seed)):
                uid, A, neg, pos = CENTRE.unpack_from(loaded, cursor)
                cursor += CENTRE.size
                assert loaded[cursor:cursor + 2 * D] == seed[c].astype('<i2').tobytes()
                cursor += 2 * D
                lists = []
                for count in [neg, pos]:
                    triples = [ENTRY.unpack_from(loaded, cursor + ENTRY.size * i) for i in range(count)]
                    cursor += count * ENTRY.size
                    cp_count = (count + STRIDE - 1) // STRIDE + 1
                    checkpoints = [int.from_bytes(loaded[cursor + i * MASK_BYTES:cursor + (i + 1) * MASK_BYTES], 'little') for i in range(cp_count)]
                    cursor += cp_count * MASK_BYTES
                    lists.append((triples, checkpoints))
                compiled.append((A, lists))
            assert cursor == len(loaded)
            all_bytes += len(buffer)
            centres += len(seed)
            cuts = np.lib.format.open_memmap(ctx.out / f'e{e:03}.cuts.npy', mode='w+', dtype='<u2', shape=query.shape)
            for local, k in enumerate(at):
                Q = int(norm[k])
                bits, comparisons, boundary, accessed, words, visited = zbits, 0, 0, HEADER.size + MASK_BYTES + 2 * D, 48, 0
                if Q == 0:
                    bits = FULL
                    cuts[local] = H + 1
                else:
                    for c, (A, lists) in enumerate(compiled):
                        t = int(query[local, c])
                        delta = A * Q - t * t
                        assert delta >= 0
                        visited += 1
                        accessed += CENTRE.size + 2 * D
                        if t == 0:
                            cuts[local, c] = H + 1
                            continue
                        triples, checkpoints = lists[0 if t > 0 else 1]
                        lo, hi = 0, len(triples)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            _, n, r = triples[mid]
                            comparisons += 1
                            accessed += ENTRY.size
                            if n * Q > r * delta:
                                hi = mid
                            else:
                                lo = mid + 1
                        cuts[local, c] = lo
                        if lo == len(triples):
                            continue
                        stop = min((lo + STRIDE - 1) // STRIDE * STRIDE, len(triples))
                        bits |= checkpoints[(lo + STRIDE - 1) // STRIDE]
                        accessed += MASK_BYTES
                        words += 48
                        for j, _, _ in triples[lo:stop]:
                            bits |= 1 << j
                            boundary += 1
                            accessed += ENTRY.size
                width = H - bits.bit_count()
                omitted[k] = np.frombuffer(bits.to_bytes(MASK_BYTES, 'little'), dtype='u1')
                work[k] = (width, comparisons, boundary, visited, accessed + width * (D + 4), words)
            cuts.flush()
            flags = np.unpackbits(omitted[at], axis=1, bitorder='little').astype(bool)
            h = hidden[at]
            assert np.all(h[flags] == 0) and np.all(codes[at][flags] == 0)
            assert np.array_equal(np.max(np.where(flags, np.float32(0), h), axis=1), np.max(h, axis=1))
            assert np.isfinite(scales[at]).all() and np.all(scales[at] > 0)
            ctx.guard()
            if e % 16 == 15:
                print(json.dumps({'expert_terminal': e, 'seconds': ctx.resources()['seconds']}), flush=True)
        omitted.flush()
        work.flush()
        assert centres == 637 and all_bytes <= 40913988
        v, books, rare, eligible = views(np, m, occ, work, all_bytes)
        ctx.r['gates'].update(all637_development_only_physical_fixed64_indices=True,
                              all17540_union_masks_original_hidden_and_codes_zero_max_preserved=True,
                              complete_fallback_all128_and_ALL_views_logical_cost=True)
        ctx.finish({'centres': centres, 'index_bytes': all_bytes, 'original_fallback_bank_bytes': 605945856,
                    'views': v, 'consumed_books': books, 'rare_consumed': rare, 'eligibility': eligible,
                    'decision': 'NEXT_NATIVE_EXACT_PARITY_AND_COMPLETE_COST' if all(eligible.values()) else 'CLOSE_THIS_ADAPTIVE_UNION_INDEX',
                    'logical_work_scope': 'Prospective indexed query reads from encoded lists: bank header/zero mask, one I16 query norm read, all centre headers+I16 codes,14B/search/boundary record,384B/nonempty suffix checkpoint, remaining original WI code+scale. Bitmap union words/dot/norm/products charged separately; not Python or physical DRAM bytes.'})
    except BaseException:
        if ctx:
            ctx.fail()
        else:
            p = DOC / 'meth513_main_result.failure.json'
            if not p.exists():
                write(p, {'phase': 'pre-numerical context admission', 'traceback': traceback.format_exc(), 'new_numerical_model_C_calls': 0})
        raise


if __name__ == '__main__':
    main()
