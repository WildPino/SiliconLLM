"""Independent integer margins, reverse fold, rounding and vector/energy audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth523_operations import Context, DOC, ROOT, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    args = ap.parse_args(); ctx = None
    try:
        ctx = Context('audit', args.binding_sha); assert ctx.digest(DOC / 'meth523_main_result.json') == args.main_sha
        raw = json.loads((DOC / 'meth523_main_result.json').read_bytes()); assert raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        folder = ROOT / 'results/native_expert_scaling/meth523_fold'
        for v in raw['output_inventory']: assert Path(v['path']).stat().st_size == v['bytes'] and ctx.digest(v['path']) == v['sha256']
        terminal = json.loads((folder / 'terminal_resource.json').read_bytes()); assert terminal['result_sha256'] == args.main_sha
        ctx.r['gates']['complete_main_ALL_output_and_terminal_SHA_before_independent_math'] = True
        np = ctx.numpy(); import meth523_contract as S
        # S is wire/I/O specification only; do not import main transformation math.
        def nearest(x):
            x = np.asarray(x, dtype='<f4'); mx = np.max(np.abs(x), axis=1); a = (mx / np.float32(32767)).astype('<f4'); a[mx == 0] = 1
            z = np.divide(x, a[:, None], dtype=np.float32).astype('<f8'); lower = np.floor(z); remainder = z - lower
            rounded = lower + ((remainder > .5) | ((remainder == .5) & (np.remainder(lower, 2) != 0)))
            return np.clip(rounded, -32767, 32767).astype('<i2'), a
        ctl = json.loads((folder / 'controls.json').read_bytes())
        I = [[1, 0], [0, 1], [1, 1]]; O = [[1, 2, 3], [-1, 0, 1]]
        for row in ctl['piecewise']:
            x = [Fraction(v) for v in row['x']]; z = [sum(a * b for a, b in zip(w, x)) for w in I]
            f = [sum(a * max(v, 0) for a, v in zip(w, z)) for w in O]
            g = [w[0] * z[0] + w[2] * z[2] + w[1] * max(z[1], 0) for w in O]
            r = [w[0] * (max(z[0], 0) - z[0]) + w[2] * (max(z[2], 0) - z[2]) for w in O]
            assert f == [Fraction(v) for v in row['source']] and g == [Fraction(v) for v in row['hybrid']]
            assert r == [Fraction(v) for v in row['residual']] and [a + b for a, b in zip(g, r)] == f
        tie, scale = nearest([[32767, -32767, 2.5, 3.5, -2.5, -3.5, 0, 1]])
        assert tie.tolist() == ctl['I16_tie_codes'] == [[32767, -32767, 2, 4, -2, -4, 0, 1]] and scale.tolist() == [1.]
        assert ctl['I16_wide_exact_dot'] == sum(32767 * 32767 for _ in range(768)) > 2 ** 31
        zero, za = nearest([[0, 0, 0]]); assert not zero.any() and za.tolist() == [ctl['zero_alpha']] == [1.]
        ctx.r['gates']['independent_exact_Fraction_region_crossing_and_I16_rounding_controls'] = True
        m, occ, inputs, ref, target, hidden, original_codes, original_alpha, dev, counts = S.load(ctx)
        ctx.r['gates']['ALL_source_UID_input_parent_mass_role_and_occurrence_contracts'] = True
        freeze = json.loads((folder / 'development_freeze.json').read_bytes()); assert len(freeze['cases']) == 128
        stats = np.empty((17540, 36), '<f8'); tolerance = np.empty((17540, 36), '<f8'); seen = np.zeros(17540, bool); checked_cases = []
        assert int(S.get(folder, 'anchors', 0)) == 0xffffffff
        for e in range(1, 128):
            ix = np.flatnonzero(m[:, 3] == e); di = ix[dev[ix]]; isdev = dev[ix]
            wi, si, wo, so = S.weights(ctx, e)
            q = inputs['q'][ix].astype('<i8'); ax = inputs['alpha'][ix].astype('<f8')
            checked, a = nearest(inputs['x'][ix]); assert checked.tobytes() == inputs['q'][ix].tobytes() and a.tobytes() == inputs['alpha'][ix].tobytes()
            # Exact I64 GEMM, independent of main's exactly integral F64 GEMM.
            dots = q @ wi.astype('<i8').T; assert np.array_equal(dots, S.get(folder, 'signed_dots', ix))
            z = (dots.astype('<f8') * si.astype('<f8')) * ax[:, None]; h = np.where(z > 0, z, 0)
            assert h.astype('<f4').tobytes() == hidden[ix].tobytes()
            dunit = inputs['q'][di].astype('<f8'); norms = np.sqrt(np.einsum('ij,ij->i', dunit, dunit))
            dunit = np.divide(dunit, norms[:, None], out=np.zeros_like(dunit), where=norms[:, None] != 0)
            alignment = dunit @ (np.sum(dunit, axis=0) / len(di)); center = int(di[int(np.argmax(alignment))])
            assert center == int(S.get(folder, 'anchors', e)) == freeze['cases'][e]['anchor_UID']
            signs = dots[int(np.searchsorted(ix, center))] > 0; assert np.array_equal(signs, S.get(folder, 'signs', e))
            I = wi.astype('<f8') * si.astype('<f8')[:, None]; O = wo.astype('<f8') * so.astype('<f8')[:, None]
            r = np.where(signs, np.maximum(-z[isdev], 0), np.maximum(z[isdev], 0))
            scores = np.einsum('ij,ij->j', r, r) / len(di) * np.einsum('ij,ij->j', O, O)
            recorded_scores = S.get(folder, 'scores', e)
            assert np.all(np.abs(scores - recorded_scores) <= 5e-12 * np.maximum(1, np.abs(scores)))
            u = np.sort(np.lexsort((np.arange(3072), -recorded_scores))[:512]); assert np.array_equal(u, S.get(folder, 'hinges', e))
            # All scores independently qualified; canonical binaryF64 recipe orders
            # the retained score bits. No alternate ordering/grid is introduced.
            for name, v in (('wi_codes', wi[u]), ('wo_codes', wo[:, u]), ('wi_scales', si[u]), ('wo_scales', so)):
                assert S.get(folder, name, e).tobytes() == v.tobytes()
            explicit = np.zeros(3072, bool); explicit[u] = True; stable = np.flatnonzero(signs & ~explicit)[::-1]
            record = freeze['cases'][e]
            assert record['expert'] == e and record['development'] == len(di) and record['consumed'] == int((~isdev).sum())
            assert record['anchor_positive_rows'] == int(signs.sum()) and record['folded_positive_rows'] == len(stable) and record['explicit_hinges'] == 512
            assert all(raw['cases'][e][k] == v for k, v in record.items())
            L = S.get(folder, 'fold64', e); folded = O[:, stable] @ I[stable]
            fold_bound = 5e-12 * np.maximum(1, np.abs(O[:, stable]) @ np.abs(I[stable]))
            fold_ratio = float(np.max(np.abs(L - folded) / fold_bound)); assert fold_ratio <= 1
            code, scale = nearest(L.astype('<f4'))
            assert code.tobytes() == S.get(folder, 'fold_codes', e).tobytes() and scale.tobytes() == S.get(folder, 'fold_scales', e).tobytes()
            decoded = code.astype('<f8') * scale.astype('<f8')[:, None]; xhat = q.astype('<f8') * ax[:, None]
            positive = np.flatnonzero(signs)[::-1]; Lall = O[:, positive] @ I[positive]
            expected = np.empty((len(ix), 5, 768), '<f8')
            expected[:, 0] = np.einsum('ij,kj->ik', h, O, optimize=False)
            expected[:, 1] = xhat @ Lall.T
            hinge = np.einsum('ij,kj->ik', h[:, u], O[:, u], optimize=False)
            expected[:, 2] = xhat @ L.T + hinge; expected[:, 3] = xhat @ decoded.T + hinge
            qh, ah = nearest(h[:, u].astype('<f4'))
            assert raw['cases'][e]['source_nonzero_max'] == int(np.max(np.count_nonzero(original_codes[ix], axis=1)))
            assert raw['cases'][e]['global_hidden_max_retained_UIDs'] == int(np.sum(ah.view('<u4') == original_alpha[ix].view('<u4')))
            linear_dot = q @ code.astype('<i8').T; hinge_dot = qh.astype('<i8') @ S.get(folder, 'wo_codes', e).astype('<i8').T
            a = ((linear_dot.astype('<f8') * scale.astype('<f8')) * ax[:, None]).astype('<f4')
            b = ((hinge_dot.astype('<f8') * so.astype('<f8')) * ah.astype('<f8')[:, None]).astype('<f4')
            expected[:, 4] = np.add(a, b, dtype=np.float32).astype('<f8')
            planes = S.get(folder, 'predictions', ix)
            assert planes[:, 4].astype('<f4').tobytes() == expected[:, 4].astype('<f4').tobytes()
            absolute_source = np.abs(h) @ np.abs(O).T
            absolute_linear = np.abs(xhat) @ np.abs(Lall).T
            absolute_hybrid = np.abs(xhat) @ np.abs(L).T + np.abs(h[:, u]) @ np.abs(O[:, u]).T
            vb = 2e-11 * np.maximum(1, absolute_source + absolute_linear + absolute_hybrid)
            vector_ratio = float(np.max(np.abs(planes[:, :4] - expected[:, :4]) / vb[:, None, :])); assert vector_ratio <= 1
            assert np.multiply(ref[ix], inputs['p'][ix, None], dtype=np.float32).tobytes() == target[ix].tobytes()
            # Independent dot/cross constructions from qualified saved vectors.
            # Absolute term sums, not blanket relative tolerances, bound each dot.
            for weighted, offset in ((False, 0), (True, 18)):
                v = planes.copy(); truth = (target[ix] if weighted else ref[ix]).astype('<f8')
                if weighted:
                    v[:, :4] *= inputs['p'][ix].astype('<f8')[:, None, None]
                    v[:, 4] = np.multiply(planes[:, 4].astype('<f4'), inputs['p'][ix, None], dtype=np.float32).astype('<f8')
                err = v - truth[:, None, :]
                components = [v[:, 4] - v[:, 3], v[:, 3] - v[:, 2], v[:, 2] - v[:, 0], v[:, 0] - truth]
                pairs = [(truth, truth)] + [(err[:, i], err[:, i]) for i in range(5)]
                pairs += [(c, c) for c in components] + [(components[i], components[j]) for i, j in itertools.combinations(range(4), 2)]
                for col, (a, b) in enumerate(pairs):
                    stats[ix, offset + col] = np.einsum('ij,ij->i', a, b)
                    tolerance[ix, offset + col] = 3e-11 * np.maximum(1, np.sum(np.abs(a * b), axis=1))
                ib = 3e-12 * np.maximum(1, np.abs(truth) + np.sum(np.abs(v), axis=1))
                stats[ix, offset + 16] = np.max(np.abs(sum(components) - err[:, 4]) / ib, axis=1)
                total = stats[ix, offset + 5]; diag = stats[ix, offset + 6:offset + 10]; cross = stats[ix, offset + 10:offset + 16]
                eb = 2e-11 * np.maximum(1, total + diag.sum(axis=1) + 2 * np.abs(cross).sum(axis=1))
                stats[ix, offset + 17] = np.abs(total - diag.sum(axis=1) - 2 * cross.sum(axis=1)) / eb
                tolerance[ix, offset + 16:offset + 18] = 1
                assert np.max(stats[ix, offset + 16:offset + 18]) <= 1
            seen[ix] = True; checked_cases.append(dict(expert=e, fold_ratio=fold_ratio, vector_ratio=vector_ratio, I64_margin_rows=len(ix), physical_BYTE_rows=len(ix)))
            ctx.guard()
            if e % 16 == 0: print('independent_parent_complete=' + str(e), flush=True)
        assert seen.all()
        np.save(ctx.out / 'metrics.npy', stats, allow_pickle=False)
        write(ctx.out / 'verified_vectors.json', dict(cases=checked_cases, source_signed_rows=17540, continuous_shadows=17540, physical_BYTE_rows=17540))
        ctx.r['gates']['ALL127_integer_margins_dev_recipe_reverse_folds_codec_and_ALL17540_physical_BYTE_vectors'] = True
        original_stats = S.get(folder, 'metrics', slice(None)); columns = list(range(16)) + list(range(18, 34))
        energy_ratio = float(np.max(np.abs(stats[:, columns] - original_stats[:, columns]) / tolerance[:, columns])); assert energy_ratio <= 1
        assert np.max(original_stats[:, [16, 17, 34, 35]]) <= 1
        ctx.r['gates']['ALL_independent_square_cross_energy_bounds_and_four_term_identities'] = True
        # Independently recover every selector; do not call main's domain helper.
        domains = []
        for split, mask in (('development', dev), ('consumed_validation', ~dev)):
            domains.append((dict(kind='uid', split=split), np.where(mask)[0]))
        for label, lo, hi in (('0', 0, 0), ('1..4', 1, 4), ('5..15', 5, 15), ('>=16', 16, 17540)):
            for split, mask in (('development', dev), ('consumed_validation', ~dev)):
                domains.append((dict(kind='rare', development_class=label, split=split), np.where(mask & (counts[m[:, 3]] >= lo) & (counts[m[:, 3]] <= hi))[0]))
        for e in range(128):
            for split, mask in (('development', dev), ('consumed_validation', ~dev)):
                domains.append((dict(kind='parent', expert=e, split=split), np.where(mask & (m[:, 3] == e))[0]))
        for book in range(192):
            for mode in range(2): domains.append((dict(kind='book', book=book, role=book // 64, mode=mode), occ[np.where((occ[:, 4] == book) & (occ[:, 6] == mode))[0], 1]))
        for role in range(3):
            for mode in range(2): domains.append((dict(kind='role', role=role, mode=mode), occ[np.where((occ[:, 7] == role) & (occ[:, 6] == mode))[0], 1]))
        assert len(domains) == len(raw['views']) == 656
        audited_views = []
        for (label, ids), v in zip(domains, raw['views']):
            assert all(v[k] == value for k, value in label.items()) and v['count'] == len(ids)
            sums = [math.fsum(float(t) for t in stats[ids, j]) for j in columns]
            bounds = [math.fsum(float(t) for t in tolerance[ids, j]) + 3e-11 * max(1, math.fsum(abs(float(t)) for t in stats[ids, j])) for j in columns]
            assert all(abs(a - b) <= bound for a, b, bound in zip(sums, v['sums'], bounds))
            audited = {**label, 'count': len(ids)}
            for prefix, offset in (('unweighted', 0), ('weighted', 16)):
                den = sums[offset]; assert math.isclose(den, v[prefix + '_source_energy'], rel_tol=3e-11, abs_tol=3e-11)
                for j, arm in enumerate(S.ARMS, 1):
                    rms = math.sqrt(sums[offset + j] / den) if den else None; previous = v[prefix + '_' + arm + '_RMS']
                    assert (rms is None and previous is None) or (rms is not None and previous is not None and math.isclose(rms, previous, rel_tol=3e-10, abs_tol=3e-10))
                    audited[prefix + '_' + arm + '_RMS'] = rms
            audited_views.append(audited)
        six = [v for v in audited_views if v['kind'] == 'role']; rare = [v for v in audited_views if v['kind'] == 'rare' and v['development_class'] in ('1..4', '5..15') and v['split'] == 'consumed_validation']
        eligible = {prefix + '_' + arm + '_ALL_six_RMS_1pct': all(v['count'] and v[prefix + '_' + arm + '_RMS'] is not None and v[prefix + '_' + arm + '_RMS'] <= .01 for v in six)
            for prefix, arm in (('unweighted', 'continuous_hybrid'), ('unweighted', 'physical_reference'), ('weighted', 'physical_reference'))}
        eligible['ALL_nonempty_rare_consumed_physical_RMS_1pct'] = all(not v['count'] or all(v[p + '_physical_reference_RMS'] is not None and v[p + '_physical_reference_RMS'] <= .01 for p in ('weighted', 'unweighted')) for v in rare)
        eligible['logical_active_coefficients_le_75pct_original'] = (768 * 768 * 2 + 2 * 768 * 512) * 4 <= 3 * (2 * 768 * 3072)
        assert eligible == raw['eligibility']
        decision = 'ELIGIBLE_MATCHED_COUNT_TEST_NOT_WHOLE_PASS' if all(eligible.values()) else 'SINGLE_REFERENCE_SOURCE_FOLD_HINGES_RECIPE_CLOSED'
        assert decision == raw['decision']; ctx.r['gates']['ALL_independent_656_report_selectors_denominators_metrics_and_prospective_decisions'] = True
        ctx.finish(dict(main_sha256=args.main_sha, views=raw['views'], eligibility=eligible, decision=decision,
            independent_cases=checked_cases, maximum_energy_bound_ratio=energy_ratio,
            new_signed_WI_projection_rows=17540, new_continuous_source_shadows=17540, new_candidate_vectors=4 * 17540,
            old_native_source_WO_replays=0, model_calls=0, native_calls=0, readout_fits=0, physical_DRAM_verified=False,
            scope='Independent one-bank integer/sign/source-fold/encoded-reference arithmetic and statistics. No original native WO replay, C/timing/fresh/useful-n/DRAM/family/whole promotion.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
