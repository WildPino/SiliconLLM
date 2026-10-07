"""One full-source fold and512 original private hinges per exposed parent."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth523_operations import Context, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context('main', args.binding_sha); np = ctx.numpy()
        import meth523_contract as S
        import meth523_math as M
        write(ctx.out / 'controls.json', M.controls())
        ctx.r['gates']['NEW_piecewise_residual_I16_tie_zero_and_wide_integer_controls'] = True
        m, occ, inputs, ref, target, hidden, original_codes, original_alpha, dev, counts = S.load(ctx)
        ctx.r['gates']['ALL_original_UID_input_role_hidden_source_mass_and_occurrence_contracts'] = True
        S.create(ctx.out); cases = [dict(expert=0, development=0, consumed=0, status='EMPTY_UNCOMPILED_UNPROMOTED')]
        S.put(ctx.out, 'anchors', 0, np.uint32(0xffffffff))
        # Compile every parent from development alone, persist the complete
        # representation, then acquire any new consumed signed-margin data.
        for e in range(1, 128):
            di = np.flatnonzero(dev & (m[:, 3] == e)); wi, si, wo, so = S.weights(ctx, e)
            q = inputs['q'][di].astype('<f8'); dots = q @ wi.astype('<f8').T
            assert np.array_equal(dots, dots.astype('<i8')) and np.max(np.abs(dots)) <= 768 * 128 * 32767
            z = (dots * si.astype('<f8')) * inputs['alpha'][di].astype('<f8')[:, None]
            assert np.maximum(z, 0).astype('<f4').tobytes() == hidden[di].tobytes()
            center = M.anchor(inputs['q'][di], di); signs = dots[int(np.searchsorted(di, center))] > 0
            I = wi.astype('<f8') * si.astype('<f8')[:, None]; O = wo.astype('<f8') * so.astype('<f8')[:, None]
            residual = np.maximum(z, 0) - z * signs
            scores = np.mean(residual ** 2, axis=0) * np.sum(O ** 2, axis=0)
            assert np.isfinite(scores).all() and np.all(scores >= 0)
            ranked = np.lexsort((np.arange(3072), -scores)); u = np.sort(ranked[:512]); kept = np.zeros(3072, bool); kept[u] = True
            folded = np.flatnonzero(signs & ~kept); L = O[:, folded] @ I[folded]
            code, scale = M.quant(L.astype('<f4'))
            S.put(ctx.out, 'signed_dots', di, dots.astype('<i8'))
            for name, value in (('fold64', L), ('fold_codes', code), ('fold_scales', scale), ('wi_codes', wi[u]),
                ('wo_codes', wo[:, u]), ('wi_scales', si[u]), ('wo_scales', so), ('hinges', u.astype('<u2')),
                ('signs', signs.astype('u1')), ('scores', scores), ('anchors', np.uint32(center))):
                S.put(ctx.out, name, e, value)
            cases.append(dict(expert=e, development=len(di), consumed=int(np.sum((~dev) & (m[:, 3] == e))), anchor_UID=center,
                anchor_positive_rows=int(signs.sum()), folded_positive_rows=len(folded), explicit_hinges=512,
                cutoff_score=float(scores[ranked[511]]), next_score=float(scores[ranked[512]]),
                development_score_retained=float(scores[u].sum()), development_score_total=float(scores.sum()),
                status='ONE_SOURCE_FOLD_PLUS512_PRIVATE_HINGES'))
            ctx.guard()
        write(ctx.out / 'development_freeze.json', dict(cases=cases, rule='one dev mean-direction medoid; score sign-residual energy times WO column energy;512 descending score then ID'))
        ctx.r['gates']['ALL127_dev_ONLY_anchor_hinges_and_complete_fold_codec_frozen_BEFORE_consumed_margin_acquisition'] = True
        for e in range(1, 128):
            ix = np.flatnonzero(m[:, 3] == e); di = ix[dev[ix]]; local_dev = dev[ix]
            wi, si, wo, so = S.weights(ctx, e)
            q = inputs['q'][ix].astype('<f8'); alpha = inputs['alpha'][ix].astype('<f8')
            checked, checked_alpha = M.quant(inputs['x'][ix])
            assert checked.tobytes() == inputs['q'][ix].tobytes() and checked_alpha.tobytes() == inputs['alpha'][ix].tobytes()
            assert np.all(np.abs(q) <= 32767) and np.isfinite(alpha).all() and np.all(alpha > 0)
            dots = np.empty((len(ix), 3072), '<f8')
            dots[local_dev] = S.get(ctx.out, 'signed_dots', di)
            dots[~local_dev] = q[~local_dev] @ wi.astype('<f8').T
            assert np.array_equal(dots, dots.astype('<i8')) and np.max(np.abs(dots)) <= 768 * 128 * 32767
            # New signed dots retain negative information discarded by500.
            z = (dots * si.astype('<f8')) * alpha[:, None]
            h = np.maximum(z, 0); h32 = h.astype('<f4')
            assert h32.tobytes() == hidden[ix].tobytes()
            assert np.all(original_codes[ix] >= 0) and np.all(original_alpha[ix] > 0)
            center = int(S.get(ctx.out, 'anchors', e)); signs = S.get(ctx.out, 'signs', e).astype(bool)
            I = wi.astype('<f8') * si.astype('<f8')[:, None]; O = wo.astype('<f8') * so.astype('<f8')[:, None]
            u = S.get(ctx.out, 'hinges', e); L = S.get(ctx.out, 'fold64', e)
            Lall = O[:, signs] @ I[signs]
            code, scale = S.get(ctx.out, 'fold_codes', e), S.get(ctx.out, 'fold_scales', e)
            decoded = code.astype('<f8') * scale.astype('<f8')[:, None]
            xhat = q * alpha[:, None]; planes = np.empty((len(ix), 5, 768), '<f8')
            planes[:, 0] = h @ O.T
            planes[:, 1] = xhat @ Lall.T
            nonlinear = h[:, u] @ O[:, u].T
            planes[:, 2] = xhat @ L.T + nonlinear
            planes[:, 3] = xhat @ decoded.T + nonlinear
            qhu, ahu = M.quant(h32[:, u])
            dl = q @ code.astype('<f8').T; dw = qhu.astype('<f8') @ wo[:, u].astype('<f8').T
            assert np.array_equal(dl, dl.astype('<i8')) and np.array_equal(dw, dw.astype('<i8'))
            assert np.max(np.abs(dl)) <= 768 * 32767 ** 2 and np.max(np.abs(dw)) <= 512 * 128 * 32767
            linear_physical = ((dl * scale.astype('<f8')) * alpha[:, None]).astype('<f4')
            nonlinear_physical = ((dw * so.astype('<f8')) * ahu.astype('<f8')[:, None]).astype('<f4')
            planes[:, 4] = np.add(linear_physical, nonlinear_physical, dtype=np.float32).astype('<f8')
            assert np.isfinite(planes).all()
            assert np.multiply(ref[ix], inputs['p'][ix, None], dtype=np.float32).tobytes() == target[ix].tobytes()
            stats = M.metrics(planes, ref[ix], target[ix], inputs['p'][ix])
            for name, value in (('signed_dots', dots.astype('<i8')), ('predictions', planes), ('metrics', stats)):
                S.put(ctx.out, name, ix, value)
            cases[e].update(dict(
                source_nonzero_max=int(np.max(np.count_nonzero(original_codes[ix], axis=1))),
                global_hidden_max_retained_UIDs=int(np.sum(ahu.view('<u4') == original_alpha[ix].view('<u4')))))
            ctx.guard()
            if e % 16 == 0: print('parent_complete=' + str(e), flush=True)
        ctx.r['gates']['ALL127_signed_projection_positive_BYTE_anchor_hinges_fold_codec_and_reference_vectors'] = True
        data = S.get(ctx.out, 'metrics', slice(None))
        views = [M.summarize(label, ids, data) for label, ids in S.domain_ids(m, occ, counts)]
        assert len(views) == 656 and sum(v['count'] for v in views if v['kind'] == 'book') == 19962
        ctx.r['gates']['ALL_656_original_domains_and_four_term_weighted_unweighted_energy_identities'] = True
        eligible = M.eligibility(views)
        decision = 'ELIGIBLE_MATCHED_COUNT_TEST_NOT_WHOLE_PASS' if all(eligible.values()) else 'SINGLE_REFERENCE_SOURCE_FOLD_HINGES_RECIPE_CLOSED'
        ctx.r['gates']['frozen_512_single_reference_I16_recipe_ALL_six_rare_and_coefficient_decisions'] = True
        ctx.finish(dict(cases=cases, views=views, eligibility=eligible, decision=decision,
            new_signed_WI_projection_rows=17540, new_continuous_source_shadows=17540, new_candidate_vectors=4 * 17540,
            old_native_source_WO_replays=0, model_calls=0, native_calls=0, readout_fits=0,
            active_coefficient_bytes=1966080, active_MACs=1376256, physical_DRAM_verified=False,
            scope='One source bank, original parent ID and normalized p controls. Serialized-code physical arithmetic reference, no C kernel/timing, fresh quality, useful n, DRAM, family or whole-goal promotion.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
