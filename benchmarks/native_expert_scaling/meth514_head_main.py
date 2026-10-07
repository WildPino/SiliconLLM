"""Sole actual256 fixed8 trial: full scores, untouched tail, aligned donor states."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import meth514_operations as O


def rational(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'float': float(value)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-sha', required=True)
    args = ap.parse_args()
    ctx = O.Context('main')
    try:
        b = ctx.admit(args.binding_sha)
        np = ctx.numpy()
        head = b['source_head']
        with Path(head['path']).open('rb') as stream:
            stream.seek(head['offset'])
            W = np.frombuffer(stream.read(head['bytes']), dtype='<f4').reshape(32128, 768).astype('f8')
        assert np.isfinite(W).all()
        scale = np.float32(1 / np.sqrt(np.float64(768)))
        ids = np.arange(32128)
        total = {k: 0 for k in ['native_positions', 'common_positions', 'original_disagreements', 'full_source_recovered',
                                'full_source_introduced', 'full_source_remaining', 'hybrid_recovered', 'hybrid_introduced',
                                'hybrid_remaining', 'source_winner_not_in_top8', 'hybrid_differs_full_source', 'unrefined_tail_wins']}
        histogram = {}
        cases = []
        max_bridge = 0.
        minimum_full_margin = None
        for ordinal, row in enumerate(b['selected']):
            assert (row['book'], row['index']) == divmod(ordinal, 4)
            data = Path(row['native']['path']).read_bytes()
            assert data[:8] == b'SWR32O01'
            s, t, d, e, f, v, nr = struct.unpack_from('<7I', data, 8)
            enc, dec = 14 * 29 * 768, t * 14 * 768
            assert (s, d, e, f, v, nr) == (29, 768, 12, 12, 32128, 174 + 6 * t)
            assert len(data) == 36 + 4 * (enc + dec + t * 32128) + 12 * nr
            states = np.frombuffer(data, dtype='<f4', offset=36 + 4 * enc, count=dec).reshape(t, 14, 768)[:, -1]
            old = np.frombuffer(data, dtype='<f4', offset=36 + 4 * (enc + dec), count=t * 32128).reshape(t, 32128)
            assert np.isfinite(states).all() and np.isfinite(old).all() and old.argmax(1).tolist() == row['candidate_ids']
            xc = (states * scale).astype('f4')
            Hc = xc.astype('f8') @ W.T
            n = row['common_positions']
            assert t == len(row['candidate_ids']) <= 64 and 1 <= n <= 13
            with np.load(row['donor']['path'], allow_pickle=False) as reference:
                ds = reference['decoder'][:n, -1]
                dl = reference['logits'][:n]
                assert ds.shape == (n, 768) and dl.shape == (n, 32128) and np.isfinite(ds).all() and np.isfinite(dl).all()
                assert dl.argmax(1).tolist() == row['donor_ids'][:n]
                assert [0] + row['candidate_ids'][:n - 1] == [0] + row['donor_ids'][:n - 1]
                Hd = (ds * scale).astype('f4').astype('f8') @ W.T
            assert np.isfinite(Hc).all() and np.isfinite(Hd).all()
            errors = np.sqrt(np.sum((Hd - dl.astype('f8')) ** 2, axis=1) / np.sum(dl.astype('f8') ** 2, axis=1))
            assert np.isfinite(errors).all() and np.all(errors <= 1e-6) and Hd.argmax(1).tolist() == row['donor_ids'][:n]
            max_bridge = max(max_bridge, float(errors.max()))
            positions = []
            for step in range(t):
                oldwinner, fullwinner = int(old[step].argmax()), int(Hc[step].argmax())
                chosen = np.lexsort((ids, -old[step].astype('f8')))[:8]
                selected64 = W[chosen] @ xc[step].astype('f8')
                refined = selected64.astype('f4')
                hybrid = old[step].copy()
                hybrid[chosen] = refined
                winner = int(hybrid.argmax())
                rank = 1 + int(np.count_nonzero(old[step] > old[step, fullwinner])) + int(np.count_nonzero((old[step] == old[step, fullwinner]) & (ids < fullwinner)))
                top2 = np.partition(Hc[step], -2)[-2:]
                margin = float(top2.max() - top2.min())
                minimum_full_margin = margin if minimum_full_margin is None else min(minimum_full_margin, margin)
                p = {'step': step, 'candidate_id': oldwinner, 'full_source_id': fullwinner, 'hybrid_id': winner,
                     'full_source_I8_rank': rank, 'selected_IDs': chosen.tolist(), 'selected_old_F32': old[step, chosen].tolist(),
                     'selected_source_F64': selected64.tolist(), 'selected_refined_F32': refined.tolist(),
                     'full_source_maximum_F64': float(Hc[step, fullwinner]), 'full_source_top2_margin_F64': margin,
                     'full_source_maximum_ties': int(np.count_nonzero(Hc[step] == Hc[step, fullwinner])),
                     'hybrid_maximum_ties': int(np.count_nonzero(hybrid == hybrid[winner])), 'unrefined_tail_wins': bool(winner not in chosen),
                     'donor_history_aligned': step < n, 'donor_id': None, 'decomposition': None}
                total['native_positions'] += 1
                total['source_winner_not_in_top8'] += int(fullwinner not in chosen)
                total['hybrid_differs_full_source'] += int(winner != fullwinner)
                total['unrefined_tail_wins'] += int(p['unrefined_tail_wins'])
                histogram[str(rank)] = histogram.get(str(rank), 0) + 1
                if step < n:
                    donor = row['donor_ids'][step]
                    oldbad, fullbad, hybridbad = oldwinner != donor, fullwinner != donor, winner != donor
                    total['common_positions'] += 1
                    total['original_disagreements'] += int(oldbad)
                    for name, bad in [('full_source', fullbad), ('hybrid', hybridbad)]:
                        total[name + '_recovered'] += int(oldbad and not bad)
                        total[name + '_introduced'] += int(not oldbad and bad)
                        total[name + '_remaining'] += int(bad)
                    p.update(donor_id=donor, donor_bridge_relative_l2=float(errors[step]), donor_source_winner=int(Hd[step].argmax()),
                             recovered=oldbad and not hybridbad, introduced=not oldbad and hybridbad)
                    if oldbad:
                        assert row['divergent'] and step == row['common_prefix_length'] == n - 1
                        pairs = [[float(a[step, donor]), float(a[step, oldwinner])] for a in [old, dl, Hc, Hd]]
                        pair_values = [Fraction.from_float(pair[1]) - Fraction.from_float(pair[0]) for pair in pairs]
                        pl, pd, pc, ph = pair_values
                        readout, upstream, arithmetic = pl - pc, pc - ph, ph - pd
                        assert readout + upstream + arithmetic == pl - pd
                        p['decomposition'] = {'candidate_id': oldwinner, 'donor_id': donor, 'pairs_old_donor_sourceC_sourceD': pairs,
                                              'donor_margin': rational(-pd), 'total_pair_perturbation': rational(pl - pd),
                                              'readout': rational(readout), 'upstream': rational(upstream), 'arithmetic': rational(arithmetic)}
                positions.append(p)
            cases.append({'book': row['book'], 'index': row['index'], 'source_id': row['source_id'],
                          'native_sha256': row['native']['sha256'], 'donor_sha256': row['donor']['sha256'], 'positions': positions})
            del Hc, Hd, data
            ctx.guard()
            if ordinal % 16 == 15:
                print(json.dumps({'cases': ordinal + 1, 'seconds': ctx.resources()['seconds']}), flush=True)
        assert (len(cases), total['native_positions'], total['common_positions'], total['original_disagreements']) == (96, 1065, 949, 24)
        for name in ['full_source', 'hybrid']:
            assert total[name + '_remaining'] == 24 - total[name + '_recovered'] + total[name + '_introduced']
        eligibility = {'ALL1065_full_source_winner_in_fixed8': total['source_winner_not_in_top8'] == 0,
                       'ALL1065_hybrid_winner_equals_full_source': total['hybrid_differs_full_source'] == 0,
                       'ALL949_aligned_original_donor_head_bridge': True,
                       'common_history_hybrid_has_fewer_errors': total['hybrid_remaining'] < 24,
                       'common_history_hybrid_introduces_zero_errors': total['hybrid_introduced'] == 0}
        summary = {'counts': total, 'rank_histogram': histogram, 'maximum_donor_bridge_relative_l2': max_bridge,
                   'minimum_full_source_top2_margin_F64': minimum_full_margin, 'eligibility': eligibility,
                   'recipe_eligible_for_next_priced_stage': all(eligibility.values()),
                   'scope': 'Fixed original256 source-head operators on retained states. No new trajectories, quality, speed, DRAM or useful-n claim.'}
        ctx.r['gates'].update(ALL96_native_wires_and_original_references_joined=True, ALL949_common_histories_and_original24_decompositions=True,
                              ALL2014_source_full_vocabulary_vectors_finite=True, ALL1065_fixed8_refinement_AND_unmodified_tail_checked=True)
        ctx.finish({'K': 8, 'F32_tied_scale': float(scale), 'cases': cases, 'summary': summary, 'goal_complete': False})
    except BaseException:
        ctx.fail()
        raise


if __name__ == '__main__':
    main()
