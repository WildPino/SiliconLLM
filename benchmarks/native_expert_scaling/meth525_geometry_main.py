"""One dev anchor/source-boundary crossing per parent, with physical route reference."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth525_operations import Context, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context('main', args.binding_sha); np = ctx.numpy()
        import meth525_contract as S
        import meth525_geometry_math as M
        # New toy control: router [1,0,0], G=[2,1,1], y=[1,3,4], h=[0,-1,0].
        ctl = dict(original_y=[1, 3, 4], target_y=[1, -2.5, float(np.sqrt(18.75))],
                   old_score=2, new_score=2, old_norm2=26, target_preactivation=2.5,
                   exact_target_last_square='75/4', chosen_opposite_integer_sign=True)
        assert abs(sum(z * z for z in ctl['target_y']) - 26) < 1e-12
        assert all(2147483647 % d for d in range(3, __import__('math').isqrt(2147483647) + 1, 2))
        toy_w = np.zeros((3072, 3), 'i1'); toy_w[0] = [0, -1, 0]
        tv, ts, ti, ty = M.panels(toy_w, np.array([2, 1, 1], '<f4'), np.array(ctl['original_y'], '<f8'),
            np.array([[1], [0], [0]], '<f8'), 0., np.zeros(3072, 'u1'), np.arange(1, 3072))
        assert ti['selected_neuron'] == 0 and ti['order_resolved']
        assert np.linalg.norm(ty - ctl['target_y']) <= ti['point_error_bound']
        ctl['module_point_error_bound'] = ti['point_error_bound']
        write(ctx.out / 'controls.json', ctl)
        ctx.r['gates']['NEW_norm_slice_target_router_and_sign_controls'] = True
        m, occ, inputs, dev, counts = S.load(ctx)
        anchors, hinges, signs, cached = S.anchors(ctx, m, inputs, dev)
        ctx.r['gates']['ALL_UID_anchor_dev_source_cached_score_input_and_parent_mass_BYTES'] = True
        router = S.read_organ(ctx, 'router'); norm = S.read_organ(ctx, 'norm')
        assert np.isfinite(router).all() and np.isfinite(norm).all()
        norm_ok = bool(np.all(norm != 0))
        cases = []; constructed = 0; new_router_calls = 0
        xp_all = np.zeros((128, 768), '<f4'); yp_all = np.zeros((128, 768), '<f8')
        codes_all = np.zeros((128, 768), '<i2'); alpha_all = np.ones(128, '<f4')
        score_all = np.zeros((128, 128), '<f4'); exp_all = np.zeros((128, 128), '<f4')
        score_bounds = np.zeros((128, 128), '<f8')
        probabilities = np.zeros(128, '<f4')
        if norm_ok:
            q, sv, vt, projection, basis_ok = M.projector(router, norm)
            for name, array in (('basis', q), ('singular', sv), ('right_vectors', vt)):
                with (ctx.out / (name + '.npy')).open('xb') as file: np.save(file, array, allow_pickle=False)
        else: projection = dict(status='ZERO_NORM_COORDINATE_UNSUPPORTED'); basis_ok = False
        ctx.r['gates']['original_router_norm_encoding_and_exact_rank_or_explicit_applicability_stop'] = True
        if basis_ok:
            for e in range(1, 128):
                uid = int(anchors[e]); x0 = inputs['x'][uid].astype('<f8'); y0 = x0 / norm.astype('<f8')
                wi, si = S.parent(ctx, e)
                values, status, info, yp = M.panels(wi, norm, y0, q, projection['projector_error_bound'], signs[e], hinges[e])
                for name, array in (('panel', values), ('status', status)):
                    with (ctx.out / f'e{e:03d}_{name}.npy').open('xb') as file: np.save(file, array, allow_pickle=False)
                case = dict(expert=e, anchor_UID=uid, development=int(counts[e]), omitted_rows=2560,
                            certified_plane_representatives=int(np.sum(status == 0)), **info)
                if yp is None: case['status'] = 'NO_CERTIFIED_NORM_SLICE_CROSSING'
                elif not info['order_resolved']: case['status'] = 'NEAREST_TARGET_ORDER_UNRESOLVED'
                else:
                    j = info['selected_neuron']; x64 = yp * norm.astype('<f8'); xp = x64.astype('<f4')
                    assert np.isfinite(xp).all()
                    cq, ax = M.quant(xp)
                    dot = int(wi[j].astype('<i8') @ cq.astype('<i8'))
                    logits, winner, p, z, exps = M.router_physical(router, xp)
                    new_router_calls += 1; constructed += 1
                    yn = float(np.linalg.norm(y0)); metric32 = xp.astype('<f8') / norm.astype('<f8')
                    rounding_metric = float(np.linalg.norm(metric32 - yp))
                    norm_error = abs(float(np.linalg.norm(metric32)) - yn)
                    norm_bound = info['point_error_bound'] + rounding_metric + M.gamma(771) * (yn + np.linalg.norm(metric32))
                    cfull = router.astype('<f8') * norm.astype('<f8')
                    bridge = np.abs(yp * norm.astype('<f8') - xp.astype('<f8'))
                    initial_bridge = np.abs(y0 * norm.astype('<f8') - x0)
                    round_bound = (M.gamma(769) + 2. ** -24) * (np.abs(router.astype('<f8')) @ (np.abs(xp.astype('<f8')) + np.abs(x0)))
                    score_bound = np.linalg.norm(cfull, axis=1) * info['point_error_bound'] + np.abs(router.astype('<f8')) @ (bridge + initial_bridge) + round_bound
                    score_bound = np.nextafter(4 * score_bound, np.inf)
                    score_difference = np.abs(logits.astype('<f8') - cached[e].astype('<f8'))
                    assert np.all(score_difference <= score_bound) and norm_error <= 4 * norm_bound
                    ideal_change = cfull @ (yp - y0)
                    case.update(status='QUERY_CONSTRUCTED', nearest_target_interval=values[j].tolist(),
                        source_reference_sign=bool(signs[e, j]), source_integer_dot=dot,
                        source_new_sign=bool(dot > 0), source_row_scale=float(si[j]),
                        physical_parent=winner, physical_probability=float(p), physical_denominator=z,
                        original_probability=float(inputs['p'][uid]),
                        relative_probability_change=abs(float(p) / float(inputs['p'][uid]) - 1),
                        physical_score_max_difference=float(np.max(score_difference)),
                        physical_score_bound_max_ratio=float(np.max(score_difference / score_bound)),
                        ideal_score_change_max=float(np.max(np.abs(ideal_change))),
                        physical_metric_norm_error=norm_error, physical_metric_norm_bound=float(4 * norm_bound),
                        physical_metric_relative_norm_error=norm_error / yn,
                        same_parent=bool(winner == e), opposite_WI_code_sign=bool((dot > 0) != bool(signs[e, j])))
                    yp_all[e], xp_all[e], codes_all[e], alpha_all[e] = yp, xp, cq, ax
                    score_all[e], exp_all[e], probabilities[e] = logits, exps, p
                    score_bounds[e] = score_bound
                cases.append(case); ctx.guard()
                if e % 32 == 0: print('geometry_parent_complete=' + str(e), flush=True)
        ctx.r['gates']['ALL_original_omitted_plane_panels_bounds_and_nearest_orders_or_explicit_stop'] = True
        for name, array in (('metric_points', yp_all), ('physical_inputs', xp_all), ('physical_codes', codes_all),
                            ('physical_alpha', alpha_all), ('new_router_scores', score_all),
                            ('new_router_score_bounds', score_bounds),
                            ('new_router_exponents', exp_all), ('new_parent_probabilities', probabilities)):
            with (ctx.out / (name + '.npy')).open('xb') as file: np.save(file, array, allow_pickle=False)
        write(ctx.out / 'query_geometry_freeze.json', dict(projection=projection, cases=cases,
            source_response_labels_acquired=0, chosen_from_development_only=True, old_native_router_queries_replayed=0))
        ctx.r['gates']['new_query_F32_router_reference_I16_sign_and_norm_bound_checks_retained_without_labels'] = True
        eligible = dict(invertible_actual_norm_and_certified_router_rank_projection=norm_ok and basis_ok,
            ALL127_have_resolved_certified_new_geometry=constructed == 127,
            ALL_constructed_keep_physical_original_parent=all(v.get('same_parent', False) for v in cases) and constructed == 127,
            ALL_constructed_cross_omitted_source_WI_code_sign=all(v.get('opposite_WI_code_sign', False) for v in cases) and constructed == 127,
            ALL_constructed_physical_norm_relative_error_le_1e_minus6=all(v.get('physical_metric_relative_norm_error', 1.) <= 1e-6 for v in cases) and constructed == 127)
        decision = 'ELIGIBLE_ONE_NEW_SOURCE_QUERY_PER_PARENT_NOT_TRANSFER_PASS' if all(eligible.values()) else 'ROUTER_NEUTRAL_HALF_ENDPOINT_GEOMETRY_CRITERION_CLOSED'
        ctx.r['gates']['frozen_ALL127_geometry_parent_sign_norm_and_label_acquisition_decision'] = True
        ctx.finish(dict(projection=projection, cases=cases, constructed_queries=constructed, new_router_score_vectors=new_router_calls,
            selected_source_integer_dot_checks=constructed, eligibility=eligible, decision=decision, views=[],
            source_response_function_calls=0, native_calls=0, model_calls=0, readout_fits=0, full_source_WI_projection_rows=0,
            old_native_router_queries_replayed=0, consumed_rows_for_selection=0, physical_DRAM_verified=False,
            scope='ONE dev-only geometry query per exposed source parent; new physical-arithmetic router reference and selected WI sign only. No source labels, actual C/kernel, fresh quality/rate, useful n or whole transfer promotion.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
