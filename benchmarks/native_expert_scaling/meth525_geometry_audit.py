"""Independent modular rank, QR projector, candidate reductions and physical bytes."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth525_operations import Context, DOC, ROOT, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    args = ap.parse_args(); ctx = None
    try:
        ctx = Context('audit', args.binding_sha); rawpath = DOC / 'meth525_main_result.json'
        assert ctx.digest(rawpath) == args.main_sha
        raw = json.loads(rawpath.read_bytes()); assert raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        folder = ROOT / 'results/native_expert_scaling/meth525_geometry'
        for v in raw['output_inventory']: assert Path(v['path']).stat().st_size == v['bytes'] and ctx.digest(v['path']) == v['sha256']
        assert json.loads((folder / 'terminal_resource.json').read_bytes())['result_sha256'] == args.main_sha
        ctx.r['gates']['main_ALL_output_and_terminal_SHA_before_independent_algebra'] = True
        np = ctx.numpy(); import meth525_contract as S
        import meth525_geometry_audit_math as A
        ctl = json.loads((folder / 'controls.json').read_bytes())
        assert Fraction(ctl['exact_target_last_square']) + Fraction(25, 4) + 1 == ctl['old_norm2'] == 26
        assert ctl['target_y'][0] * 2 == ctl['old_score'] == ctl['new_score'] == 2
        assert -ctl['target_y'][1] == ctl['target_preactivation'] == 2.5 and ctl['chosen_opposite_integer_sign']
        assert abs(ctl['target_y'][2] ** 2 - 18.75) <= 1e-13
        prime = 1000000007
        assert all(prime % d for d in range(3, math.isqrt(prime) + 1, 2))
        ctx.r['gates']['independent_exact_norm_slice_sign_and_prime_controls'] = True
        m, occ, inputs, dev, counts = S.load(ctx); anchors, hinges, signs, cached = S.anchors(ctx, m, inputs, dev)
        router, norm = S.read_organ(ctx, 'router'), S.read_organ(ctx, 'norm')
        ctx.r['gates']['ALL_original_UID_dev_anchors_cached_input_scores_p_and_source_organ_contracts'] = True
        eps = 2. ** -53
        def gam(n): return n * eps / (1 - n * eps)
        report = raw['projection']; norm_ok = bool(np.all(norm != 0)); qr_projector_distance = 0.; rank = 0
        if norm_ok:
            keep = []; removed = []; keys = {}
            for i, row in enumerate(router):
                active = np.flatnonzero(row)
                if not len(active): removed.append(dict(row=i, reason='exact_zero')); continue
                sign = 1 if float(row[active[0]]) > 0 else -1
                key = tuple(float(v) * sign for v in row)
                if key in keys: removed.append(dict(row=i, reason='exact_signed_duplicate', representative=keys[key][0], multiplier=sign * keys[key][1]))
                else: keys[key] = (i, sign); keep.append(i)
            assert keep == report['kept_rows'] and removed == report['removed_rows']
            # Decode IEEE754 F32 bits into a different finite field, no F64 ratios.
            words = router[keep].view('<u4'); field = np.zeros(words.shape, '<i8'); powers = {}
            for i in range(len(keep)):
                for j, word in enumerate(words[i]):
                    word = int(word); exp = (word >> 23) & 255; mant = word & 0x7fffff
                    if exp: mant |= 0x800000; shift = exp - 150
                    else: shift = -149
                    if shift not in powers: powers[shift] = pow(2, shift, prime) if shift >= 0 else pow(pow(2, -shift, prime), prime - 2, prime)
                    field[i, j] = ((-mant if word >> 31 else mant) * powers[shift]) % prime
            for col in range(768):
                candidates = [i for i in range(rank, len(keep)) if field[i, col] != 0]
                if not candidates: continue
                k = candidates[0]; field[[rank, k]] = field[[k, rank]]
                inv = pow(int(field[rank, col]), prime - 2, prime)
                field[rank, col:] = field[rank, col:] * inv % prime
                for i in range(rank + 1, len(keep)):
                    field[i, col:] = (field[i, col:] - field[i, col] * field[rank, col:]) % prime
                rank += 1
                if rank == len(keep): break
            assert rank == report['rank']
            q = np.load(folder / 'basis.npy', allow_pickle=False)
            sv = np.load(folder / 'singular.npy', allow_pickle=False); vt = np.load(folder / 'right_vectors.npy', allow_pickle=False)
            c = router[keep].astype('<f8') * norm.astype('<f8')
            assert q.shape == (768, len(keep)) and sv.tolist() == report['singular_values'] and vt.shape == (len(keep), len(keep))
            qgram = np.einsum('ji,jk->ik', q, q, optimize=False)
            vgram = np.einsum('ij,kj->ik', vt, vt, optimize=False)
            oq = float(np.linalg.norm(qgram - np.eye(len(keep)), 'fro'))
            assert oq <= report['orthogonality_bound'] and np.linalg.norm(vgram - np.eye(len(keep)), 'fro') < 1e-10
            recon = np.einsum('ij,jk->ik', q * sv, vt, optimize=False)
            assert np.linalg.norm(c.T - recon, 'fro') <= report['reconstruction_bound']
            residue = c - np.einsum('ik,jk->ij', np.einsum('ij,jk->ik', c, q, optimize=False), q, optimize=False)
            assert np.linalg.norm(residue, 'fro') <= report['rowspace_residual_bound']
            # Recalculate the SVD certificate constants with explicit contractions.
            qa = np.einsum('ji,jk->ik', np.abs(q), np.abs(q), optimize=False)
            va = np.einsum('ij,kj->ik', np.abs(vt), np.abs(vt), optimize=False)
            oq_bound = oq + gam(769) * np.linalg.norm(qa, 'fro')
            ov_bound = np.linalg.norm(vgram - np.eye(len(keep)), 'fro') + gam(len(keep) + 1) * np.linalg.norm(va, 'fro')
            rec_abs = np.einsum('ij,jk->ik', np.abs(q) * sv, np.abs(vt), optimize=False)
            rec_bound = np.linalg.norm(c.T - recon, 'fro') + gam(len(keep) + 3) * np.linalg.norm(np.abs(c.T) + rec_abs, 'fro')
            caq = np.einsum('ij,jk->ik', np.abs(c), np.abs(q), optimize=False)
            caqq = np.einsum('ik,jk->ij', caq, np.abs(q), optimize=False)
            rb_bound = np.linalg.norm(residue, 'fro') + gam(768 + 2 * len(keep) + 5) * np.linalg.norm(np.abs(c) + caqq, 'fro')
            cnorm = np.linalg.norm(c, 'fro') * (1 + gam(c.size + 1))
            independent_lower = sv[-1] * math.sqrt(max(0, 1 - oq_bound)) * math.sqrt(max(0, 1 - ov_bound)) - 2 * rec_bound
            certificate_slack = gam(10000) * float(cnorm)
            assert abs(independent_lower - report['minimum_singular_lower']) <= certificate_slack
            assert abs(2 * rec_bound - report['reconstruction_bound']) <= certificate_slack
            assert abs(2 * rb_bound - report['rowspace_residual_bound']) <= certificate_slack
            independent_base = 2 * (rb_bound + cnorm * oq_bound / (1 - oq_bound)) / independent_lower + 2 * oq_bound / (1 - oq_bound) if independent_lower > 0 and oq_bound < 1 else 1.
            projector_slack = 16 * certificate_slack / independent_lower if independent_lower > 0 else gam(10000)
            assert abs(2 * independent_base - report['projector_error_bound']) <= projector_slack
            # Independently generated QR span; full column rank was certified modulo p.
            qr, triangular = np.linalg.qr(c.T, mode='reduced')
            qr_orth = np.linalg.norm(qr.T @ qr - np.eye(len(keep)), 'fro') + gam(769) * np.linalg.norm(np.abs(qr).T @ np.abs(qr), 'fro')
            inverse = np.linalg.solve(triangular, np.eye(len(keep))) if rank == len(keep) and sv[-1] / sv[0] >= 1e-6 and report['minimum_singular_lower'] > 0 else None
            if inverse is None:
                assert rank < len(keep) or sv[-1] / sv[0] < 1e-6 or report['minimum_singular_lower'] <= 0
                inverse = np.zeros_like(triangular)
            inverse_residue = np.linalg.norm(triangular @ inverse - np.eye(len(keep)), 'fro') + gam(len(keep) + 1) * np.linalg.norm(np.abs(triangular) @ np.abs(inverse), 'fro')
            qr_residue = np.linalg.norm(c.T - qr @ triangular, 'fro') + gam(len(keep) + 2) * np.linalg.norm(np.abs(c.T) + np.abs(qr) @ np.abs(triangular), 'fro')
            qr_lower = (1 - inverse_residue) * math.sqrt(1 - qr_orth) / (np.linalg.norm(inverse, 'fro') * (1 + gam(inverse.size + 1))) - 2 * qr_residue
            if qr_lower > 0:
                qr_span_residue = np.linalg.norm(c - (c @ qr) @ qr.T, 'fro') + gam(768 + 2 * len(keep) + 5) * np.linalg.norm(np.abs(c) + (np.abs(c) @ np.abs(qr)) @ np.abs(qr).T, 'fro')
                qr_bound = 4 * (qr_span_residue + np.linalg.norm(c, 'fro') * qr_orth / (1 - qr_orth)) / qr_lower + 4 * qr_orth / (1 - qr_orth)
                qr_projector_distance = float(np.linalg.norm(q @ q.T - qr @ qr.T, 2))
                assert qr_projector_distance <= report['projector_error_bound'] + qr_bound
            else: assert rank < len(keep) or sv[-1] / sv[0] < 1e-6 or report['minimum_singular_lower'] <= 0
            basis_ok = rank == len(keep) and 768 - len(keep) >= 2 and report['minimum_singular_lower'] > 0 and sv[-1] / sv[0] >= 1e-6 and report['projector_error_bound'] / 2 < 1e-6
        else: basis_ok = False; assert report['status'] == 'ZERO_NORM_COORDINATE_UNSUPPORTED'
        ctx.r['gates']['independent_F32_bit_modular_rank_QR_span_and_SVD_representation_certificate'] = True
        point = np.load(folder / 'metric_points.npy', allow_pickle=False)
        xp = np.load(folder / 'physical_inputs.npy', allow_pickle=False); codes = np.load(folder / 'physical_codes.npy', allow_pickle=False)
        alpha = np.load(folder / 'physical_alpha.npy', allow_pickle=False); scores = np.load(folder / 'new_router_scores.npy', allow_pickle=False)
        exponents = np.load(folder / 'new_router_exponents.npy', allow_pickle=False); probability = np.load(folder / 'new_parent_probabilities.npy', allow_pickle=False)
        score_bounds = np.load(folder / 'new_router_score_bounds.npy', allow_pickle=False)
        assert point.shape == xp.shape == codes.shape == (128, 768) and scores.shape == exponents.shape == (128, 128)
        assert alpha.shape == probability.shape == (128,) and score_bounds.shape == (128, 128) and score_bounds.dtype == np.dtype('<f8')
        assert point.dtype == np.dtype('<f8') and xp.dtype == np.dtype('<f4') and codes.dtype == np.dtype('<i2')
        assert alpha.dtype == scores.dtype == exponents.dtype == probability.dtype == np.dtype('<f4')
        def nearest(x):
            a = np.float32(np.float32(max(abs(float(v)) for v in x)) / np.float32(32767))
            if a == 0: a = np.float32(1)
            out = []
            for v in x:
                z = float(np.float32(v / a)); floor = math.floor(z); rem = z - floor
                out.append(max(-32767, min(32767, floor + int(rem > .5 or (rem == .5 and floor % 2 != 0)))))
            return np.array(out, '<i2'), a
        def physical(x):
            out = []
            for row in router:
                low = [0.] * 4; high = [0.] * 4
                for i in range(0, 768, 8):
                    for k in range(4):
                        low[k] += float(row[i + k]) * float(x[i + k])
                        high[k] += float(row[i + k + 4]) * float(x[i + k + 4])
                total = 0.
                for k in range(4): total += low[k] + high[k]
                out.append(np.float32(total))
            logits = np.array(out, '<f4'); e = max(range(128), key=lambda j: (float(logits[j]), -j))
            ev = np.array([np.float32(math.exp(float(np.float32(v - logits[e])))) for v in logits], '<f4')
            z = 0.
            for v in ev: z += float(v)
            return logits, ev, e, np.float32(1. / z), z
        max_panel_ratio = 0.; constructed = 0; verified_planes = 0; independent_norm_passes = []
        assert len(raw['cases']) == (127 if basis_ok else 0)
        for case in raw['cases']:
            e = case['expert']; uid = int(anchors[e]); assert case['anchor_UID'] == uid and case['development'] == int(counts[e])
            wi, si = S.parent(ctx, e); y = inputs['x'][uid].astype('<f8') / norm.astype('<f8')
            expected, expected_status, info, yp = A.panels(wi, norm, y, q, report['projector_error_bound'], signs[e], hinges[e])
            panel = np.load(folder / f'e{e:03d}_panel.npy', allow_pickle=False); status = np.load(folder / f'e{e:03d}_status.npy', allow_pickle=False)
            assert panel.shape == (3072, 7) and panel.dtype == np.dtype('<f8') and status.shape == (3072,) and status.dtype == np.dtype('u1')
            assert np.array_equal(status, expected_status)
            eb = panel[:, 2] + expected[:, 2]
            assert np.all(np.abs(panel[:, 0] - expected[:, 0]) <= eb) and np.all(np.abs(panel[:, 1] - expected[:, 1]) <= eb)
            assert np.all(np.abs(panel[:, 3] - expected[:, 3]) <= .5 * eb)
            good = (status == 0) | (status == 4)
            db = panel[good, 5] + expected[good, 5]
            discrepancy = np.abs(panel[good, 4] - expected[good, 4]); assert np.all(discrepancy <= db)
            if len(db): max_panel_ratio = max(max_panel_ratio, float(np.max(discrepancy / db)))
            verified_planes += 3072
            reps = np.flatnonzero(status == 0)
            if not len(reps): assert case['status'] == 'NO_CERTIFIED_NORM_SLICE_CROSSING'; continue
            order = sorted(reps, key=lambda j: (panel[j, 4], j)); j = int(order[0])
            resolved = all(panel[j, 4] + panel[j, 5] < panel[t, 4] - panel[t, 5] for t in order[1:])
            assert case['selected_neuron'] == j and case['order_resolved'] == resolved
            if not resolved: assert case['status'] == 'NEAREST_TARGET_ORDER_UNRESOLVED'; continue
            assert case['status'] == 'QUERY_CONSTRUCTED' and case['nearest_target_interval'] == panel[j].tolist()
            # Retained qualified representation bits define the physical query.
            assert np.linalg.norm(point[e] - yp) <= case['point_error_bound'] + info['point_error_bound']
            assert xp[e].tobytes() == (point[e] * norm.astype('<f8')).astype('<f4').tobytes()
            cq, ax = nearest(xp[e]); assert cq.tobytes() == codes[e].tobytes() and ax.tobytes() == alpha[e].tobytes()
            integer = sum(int(a) * int(b) for a, b in zip(wi[j], cq))
            assert integer == case['source_integer_dot'] and (integer > 0) == case['source_new_sign']
            logits, ev, winner, p, z = physical(xp[e])
            assert logits.tobytes() == scores[e].tobytes() and ev.tobytes() == exponents[e].tobytes() and p.tobytes() == probability[e].tobytes()
            assert winner == case['physical_parent'] and float(p) == case['physical_probability'] and z == case['physical_denominator']
            assert case['same_parent'] == (winner == e) and case['opposite_WI_code_sign'] == ((integer > 0) != bool(signs[e, j]))
            yn = math.sqrt(math.fsum(float(v) ** 2 for v in y)); newy = xp[e].astype('<f8') / norm.astype('<f8')
            norm_error = abs(math.sqrt(math.fsum(float(v) ** 2 for v in newy)) - yn)
            assert norm_error <= case['physical_metric_norm_bound']
            norm_eval_bound = gam(10000) * (yn + math.sqrt(math.fsum(float(v) ** 2 for v in newy)))
            assert abs(norm_error - case['physical_metric_norm_error']) <= norm_eval_bound
            assert abs(norm_error / yn - case['physical_metric_relative_norm_error']) <= norm_eval_bound / yn
            independent_norm_passes.append(norm_error / yn <= 1e-6)
            rounding_metric = math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(newy, point[e])))
            independent_norm_bound = 4 * (case['point_error_bound'] + rounding_metric + gam(771) * (yn + math.sqrt(math.fsum(float(v) ** 2 for v in newy))))
            assert abs(independent_norm_bound - case['physical_metric_norm_bound']) <= gam(10000) * independent_norm_bound
            # All old scores are cached; only the NEW physical vector was evaluated above.
            cfull = router.astype('<f8') * norm.astype('<f8')
            x0 = inputs['x'][uid].astype('<f8')
            bridge = np.abs(point[e] * norm.astype('<f8') - xp[e].astype('<f8'))
            initial_bridge = np.abs(y * norm.astype('<f8') - x0)
            expected_bounds = []; ideal_change = []
            for row, cr in zip(router, cfull):
                cn = math.sqrt(math.fsum(float(v) ** 2 for v in cr))
                bridge_sum = math.fsum(abs(float(r)) * float(b) for r, b in zip(row, bridge + initial_bridge))
                amplitude = math.fsum(abs(float(r)) * (abs(float(a)) + abs(float(b))) for r, a, b in zip(row, xp[e], x0))
                expected_bounds.append(math.nextafter(4 * (cn * case['point_error_bound'] + bridge_sum + (gam(769) + 2. ** -24) * amplitude), math.inf))
                ideal_change.append(math.fsum(float(r) * (float(a) - float(b)) for r, a, b in zip(cr, point[e], y)))
            expected_bounds = np.array(expected_bounds, '<f8')
            assert np.all(np.abs(score_bounds[e] - expected_bounds) <= gam(10000) * expected_bounds)
            difference = np.abs(logits.astype('<f8') - cached[e].astype('<f8'))
            assert np.all(difference <= score_bounds[e])
            assert float(np.max(difference)) == case['physical_score_max_difference']
            assert float(np.max(difference / score_bounds[e])) == case['physical_score_bound_max_ratio']
            ideal_eval = gam(10000) * float(np.max(np.abs(cfull) @ np.abs(point[e] - y)))
            assert abs(max(abs(v) for v in ideal_change) - case['ideal_score_change_max']) <= ideal_eval
            assert case['source_reference_sign'] == bool(signs[e, j]) and case['source_row_scale'] == float(si[j])
            assert case['original_probability'] == float(inputs['p'][uid])
            assert case['relative_probability_change'] == abs(float(p) / float(inputs['p'][uid]) - 1)
            h = wi[j].astype('<f8') * norm.astype('<f8')
            target = math.fsum(float(a) * float(b) for a, b in zip(h, point[e]))
            assert abs(target - panel[j, 3]) <= np.linalg.norm(h) * case['point_error_bound'] + gam(769) * float(np.abs(h) @ np.abs(point[e]))
            reconstructed_distance = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(y, point[e]))
            assert abs(reconstructed_distance - panel[j, 4]) <= panel[j, 5] + gam(771) * reconstructed_distance
            constructed += 1; ctx.guard()
            if e % 32 == 0: print('audited_geometry_parent=' + str(e), flush=True)
        ctx.r['gates']['ALL_candidate_interval_target_distance_panels_aliases_and_nearest_orders_independent'] = True
        ctx.r['gates']['ALL_new_query_F32_scores_exponents_mass_and_I16_source_sign_BYTES_independent'] = True
        assert constructed == raw['constructed_queries'] == raw['new_router_score_vectors'] == raw['selected_source_integer_dot_checks']
        eligibility = dict(invertible_actual_norm_and_certified_router_rank_projection=norm_ok and basis_ok,
            ALL127_have_resolved_certified_new_geometry=constructed == 127,
            ALL_constructed_keep_physical_original_parent=all(v.get('same_parent', False) for v in raw['cases']) and constructed == 127,
            ALL_constructed_cross_omitted_source_WI_code_sign=all(v.get('opposite_WI_code_sign', False) for v in raw['cases']) and constructed == 127,
            ALL_constructed_physical_norm_relative_error_le_1e_minus6=all(independent_norm_passes) and constructed == 127)
        decision = 'ELIGIBLE_ONE_NEW_SOURCE_QUERY_PER_PARENT_NOT_TRANSFER_PASS' if all(eligibility.values()) else 'ROUTER_NEUTRAL_HALF_ENDPOINT_GEOMETRY_CRITERION_CLOSED'
        assert eligibility == raw['eligibility'] and decision == raw['decision'] and len(raw['gates']) == 7
        ctx.r['gates']['unchanged_ALL127_fixed_geometry_eligibility_and_zero_source_label_decision'] = True
        write(ctx.out / 'verified_geometry.json', dict(main_sha256=args.main_sha, modular_rank_prime=prime, modular_rank=rank,
            verified_source_planes=verified_planes, maximum_panel_bound_ratio=max_panel_ratio, QR_projector_distance=qr_projector_distance,
            physical_query_vectors_BYTE_verified=constructed))
        ctx.finish(dict(main_sha256=args.main_sha, projection=raw['projection'], cases=raw['cases'], views=[],
            constructed_queries=constructed, new_router_score_vectors=constructed, selected_source_integer_dot_checks=constructed,
            eligibility=eligibility, decision=decision, verified_source_planes=verified_planes, maximum_panel_bound_ratio=max_panel_ratio,
            QR_projector_distance=qr_projector_distance, source_response_function_calls=0, native_calls=0, model_calls=0,
            readout_fits=0, full_source_WI_projection_rows=0, old_native_router_queries_replayed=0,
            consumed_rows_for_selection=0, physical_DRAM_verified=False,
            scope='Independent original-source geometry and physical-arithmetic reference query verification only; no new teacher function labels or useful large-n/whole promotion.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
