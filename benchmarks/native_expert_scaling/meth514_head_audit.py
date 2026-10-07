"""Independent full-vocabulary/accurate-row audit; no main mathematical imports."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import meth514_operations as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-sha', required=True)
    ap.add_argument('--main-sha', required=True)
    args = ap.parse_args()
    ctx = O.Context('audit')
    try:
        b = ctx.admit(args.binding_sha)
        main_path = O.DOC / 'meth514_main_result.json'
        assert ctx.digest(main_path) == args.main_sha
        raw = json.loads(main_path.read_bytes())
        assert raw['kind'] == 'main' and raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        np = ctx.numpy()
        head = b['source_head']
        with Path(head['path']).open('rb') as stream:
            stream.seek(head['offset'])
            weights = np.frombuffer(stream.read(head['bytes']), dtype='<f4').reshape(32128, 768).astype('f8')
        assert np.isfinite(weights).all()
        scale = np.float32(1 / math.sqrt(768))
        assert float(scale) == raw['F32_tied_scale'] and b['K'] == raw['K'] == 8
        counts = {k: 0 for k in ['native_positions', 'common_positions', 'original_disagreements', 'full_source_recovered',
                                 'full_source_introduced', 'full_source_remaining', 'hybrid_recovered', 'hybrid_introduced',
                                 'hybrid_remaining', 'source_winner_not_in_top8', 'hybrid_differs_full_source', 'unrefined_tail_wins']}
        histogram = {}
        maximum_dot_error = maximum_margin_error = maximum_bridge = 0.
        selected_rows = decompositions = 0
        minmargin = None
        audits = []
        for ordinal, (row, observed) in enumerate(zip(b['selected'], raw['cases'])):
            assert (row['book'], row['index']) == (observed['book'], observed['index']) == divmod(ordinal, 4)
            assert row['source_id'] == observed['source_id'] and row['native']['sha256'] == observed['native_sha256'] and row['donor']['sha256'] == observed['donor_sha256']
            buf = Path(row['native']['path']).read_bytes()
            values = struct.unpack_from('<8s7I', buf)
            magic, source_n, steps, width, encoder_layers, decoder_layers, vocab, route_n = values
            assert (magic, source_n, width, encoder_layers, decoder_layers, vocab, route_n) == (b'SWR32O01', 29, 768, 12, 12, 32128, 174 + 6 * steps)
            assert len(buf) == 36 + (14 * source_n * width + 14 * steps * width + steps * vocab) * 4 + route_n * 12
            # Independent ndarray strides/layout and W@x.T full vocabulary path.
            decoder_start = 36 + 14 * source_n * width * 4
            decoder = np.ndarray((steps, 14, width), dtype='<f4', buffer=buf, offset=decoder_start)
            old = np.ndarray((steps, vocab), dtype='<f4', buffer=buf, offset=decoder_start + steps * 14 * width * 4)
            states = (decoder[:, 13, :] * scale).astype('f4')
            source_scores = (weights @ states.astype('f8').T).T
            assert np.isfinite(decoder).all() and np.isfinite(old).all() and np.isfinite(source_scores).all()
            assert old.argmax(1).tolist() == row['candidate_ids'] and len(observed['positions']) == steps <= 64
            prefix = 0
            while prefix < min(len(row['candidate_ids']), len(row['donor_ids'])) and row['candidate_ids'][prefix] == row['donor_ids'][prefix]:
                prefix += 1
            common = min(steps, len(row['donor_ids']), prefix + 1)
            assert common == row['common_positions'] and prefix == row['common_prefix_length']
            assert [0] + row['candidate_ids'][:common - 1] == [0] + row['donor_ids'][:common - 1]
            with np.load(row['donor']['path'], allow_pickle=False) as reference:
                ds = reference['decoder'][:common, 13, :]
                dl = reference['logits'][:common, :]
                assert ds.shape == (common, 768) and dl.shape == (common, 32128) and np.isfinite(ds).all() and np.isfinite(dl).all()
                assert dl.argmax(1).tolist() == row['donor_ids'][:common]
                donor_scores = (weights @ (ds * scale).astype('f4').astype('f8').T).T
            assert np.isfinite(donor_scores).all()
            bridge = np.linalg.norm(donor_scores - dl, axis=1) / np.linalg.norm(dl.astype('f8'), axis=1)
            assert np.isfinite(bridge).all() and np.all(bridge <= 1e-6) and donor_scores.argmax(1).tolist() == row['donor_ids'][:common]
            maximum_bridge = max(maximum_bridge, float(bridge.max()))
            for j, p in enumerate(observed['positions']):
                full = source_scores[j]
                ow = int(np.argmax(old[j]))
                fw = int(np.argmax(full))
                # Stable argsort starts in ID order, independently checking tie recipe.
                chosen = np.argsort(-old[j].astype('f8'), kind='stable')[:8].tolist()
                accurate = np.array([math.fsum((weights[k] * states[j].astype('f8')).tolist()) for k in chosen], dtype='f8')
                error = float(np.max(np.abs(accurate - np.array(p['selected_source_F64']))))
                maximum_dot_error = max(maximum_dot_error, error)
                assert error <= 2e-10
                refined = accurate.astype('f4')
                hybrid = old[j].copy()
                hybrid[chosen] = refined
                hw = int(hybrid.argmax())
                rank = sum(float(value) > float(old[j, fw]) or (float(value) == float(old[j, fw]) and index < fw) for index, value in enumerate(old[j])) + 1
                runner = float(np.partition(full, -2)[-2])
                margin = float(full[fw] - runner)
                maximum_margin_error = max(maximum_margin_error, abs(margin - p['full_source_top2_margin_F64']))
                assert abs(margin - p['full_source_top2_margin_F64']) <= 2e-10
                assert abs(float(full[fw]) - p['full_source_maximum_F64']) <= 2e-10
                minmargin = margin if minmargin is None else min(minmargin, margin)
                assert (j, ow, fw, hw, rank, chosen) == (p['step'], p['candidate_id'], p['full_source_id'], p['hybrid_id'], p['full_source_I8_rank'], p['selected_IDs'])
                assert old[j, chosen].tolist() == p['selected_old_F32'] and refined.tolist() == p['selected_refined_F32']
                assert int(np.sum(full == full[fw])) == p['full_source_maximum_ties'] and int(np.sum(hybrid == hybrid[hw])) == p['hybrid_maximum_ties']
                assert (hw not in chosen) == p['unrefined_tail_wins'] and (j < common) == p['donor_history_aligned']
                selected_rows += 8
                counts['native_positions'] += 1
                counts['source_winner_not_in_top8'] += int(fw not in chosen)
                counts['hybrid_differs_full_source'] += int(hw != fw)
                counts['unrefined_tail_wins'] += int(hw not in chosen)
                histogram[str(rank)] = histogram.get(str(rank), 0) + 1
                if j < common:
                    donor = row['donor_ids'][j]
                    assert p['donor_id'] == p['donor_source_winner'] == donor
                    assert abs(float(bridge[j]) - p['donor_bridge_relative_l2']) <= 2e-10
                    original_bad = ow != donor
                    counts['common_positions'] += 1
                    counts['original_disagreements'] += int(original_bad)
                    for name, winner in [('full_source', fw), ('hybrid', hw)]:
                        counts[name + '_recovered'] += int(original_bad and winner == donor)
                        counts[name + '_introduced'] += int(not original_bad and winner != donor)
                        counts[name + '_remaining'] += int(winner != donor)
                    assert p['recovered'] == (original_bad and hw == donor) and p['introduced'] == (not original_bad and hw != donor)
                    if original_bad:
                        assert j == prefix == common - 1 and row['divergent']
                        q = p['decomposition']
                        assert q['candidate_id'] == ow and q['donor_id'] == donor
                        pairs = q['pairs_old_donor_sourceC_sourceD']
                        assert pairs[:2] == [[float(a[j, donor]), float(a[j, ow])] for a in [old, dl]]
                        for saved, a in zip(pairs[2:], [source_scores, donor_scores]):
                            assert max(abs(saved[0] - float(a[j, donor])), abs(saved[1] - float(a[j, ow]))) <= 2e-10
                        pl, pd, pc, ph = [Fraction.from_float(values[1]) - Fraction.from_float(values[0]) for values in pairs]
                        expected = {'donor_margin': -pd, 'total_pair_perturbation': pl - pd, 'readout': pl - pc, 'upstream': pc - ph, 'arithmetic': ph - pd}
                        for key, val in expected.items():
                            assert q[key] == {'numerator': val.numerator, 'denominator': val.denominator, 'float': float(val)}
                        assert expected['readout'] + expected['upstream'] + expected['arithmetic'] == expected['total_pair_perturbation']
                        decompositions += 1
                    else:
                        assert p['decomposition'] is None
                else:
                    assert p['donor_id'] is p['decomposition'] is None
            audits.append({'book': row['book'], 'index': row['index'], 'native_positions': steps, 'common_positions': common, 'passed': True})
            del source_scores, donor_scores, buf
            ctx.guard()
            if ordinal % 16 == 15:
                print(json.dumps({'audited_cases': ordinal + 1, 'seconds': ctx.resources()['seconds']}), flush=True)
        assert len(audits) == len(raw['cases']) == len(b['selected']) == 96
        assert counts == raw['summary']['counts'] and histogram == raw['summary']['rank_histogram']
        assert selected_rows == 8520 and decompositions == 24 and counts['native_positions'] == 1065 and counts['common_positions'] == 949
        eligibility = {'ALL1065_full_source_winner_in_fixed8': counts['source_winner_not_in_top8'] == 0,
                       'ALL1065_hybrid_winner_equals_full_source': counts['hybrid_differs_full_source'] == 0,
                       'ALL949_aligned_original_donor_head_bridge': True,
                       'common_history_hybrid_has_fewer_errors': counts['hybrid_remaining'] < 24,
                       'common_history_hybrid_introduces_zero_errors': counts['hybrid_introduced'] == 0}
        assert eligibility == raw['summary']['eligibility'] and all(eligibility.values()) == raw['summary']['recipe_eligible_for_next_priced_stage']
        assert abs(minmargin - raw['summary']['minimum_full_source_top2_margin_F64']) <= 2e-10
        ctx.r['gates'].update(ALL96_independent_input_layout_and_common_history_joins=True, ALL2014_full_vocabulary_and949_donor_bridges=True,
                              ALL1065_full_argmax_ties_margins_ranks_tail_AND8520_fsum_rows=True, ALL24_exact_rational_decompositions=True,
                              ALL_recovery_introduction_counts_AND_frozen_decision_exact=True)
        summary = {'counts': counts, 'eligibility': eligibility, 'recipe_eligible_for_next_priced_stage': all(eligibility.values()),
                   'audited_cases': 96, 'full_vocabulary_cells': 2014 * 32128, 'selected_fsum_rows': selected_rows,
                   'decompositions': decompositions, 'maximum_selected_dot_absolute_error': maximum_dot_error,
                   'maximum_margin_absolute_error': maximum_margin_error, 'maximum_donor_bridge_relative_l2': maximum_bridge,
                   'scope': 'Independent finite retained-state audit; no native/full model/corpus query or new quality/rate.'}
        ctx.finish({'main_sha256': args.main_sha, 'audited_cases': audits, 'summary': summary, 'goal_complete': False})
    except BaseException:
        ctx.fail()
        raise


if __name__ == '__main__':
    main()
