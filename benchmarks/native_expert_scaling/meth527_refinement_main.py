"""NEW exact coefficient energies and scalar bounds, zero old geometric/query replay."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth527_operations import Context, ROOT, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); args = ap.parse_args(); ctx = None
    try:
        ctx = Context('main', args.binding_sha); np = ctx.numpy()
        import meth527_contract as S
        import meth527_refinement_math as M
        toy_v = [2, 1, 1] + [1] * 765; toy_codes = np.zeros((1, 768), 'i1'); toy_codes[0, :2] = [-3, 4]
        assert M.energies(toy_codes, toy_v) == [52] and M.norm_up(52, 0) == 8.
        toy_d = 65 - 8 * math.sqrt(75 / 4)
        toy = np.array([[0, 5, 0, 2.5, toy_d, 0, 1]], '<f8')
        tb, ts, ta, tr = M.refine(5., math.sqrt(26), toy, np.ones(1), 0., 0.)
        assert ts[0] and Fraction.from_float(float(tr[0])) ** 2 <= Fraction.from_float(float(ta[0])) <= Fraction(75, 4)
        dlo, dhi = Fraction.from_float(toy_d - tb[0]), Fraction.from_float(toy_d + tb[0])
        assert ((65 - dhi) / 8) ** 2 <= Fraction(75, 4) <= ((65 - dlo) / 8) ** 2
        write(ctx.out / 'controls.json', dict(exact_row_energy=52, energy_norm_upper=8.,
            toy_distance=toy_d, toy_error_bound=float(tb[0]), toy_radicand_lower=float(ta[0]), toy_root_lower=float(tr[0]),
            exact_target_radicand='75/4', exact_distance='65-8*sqrt(75/4)'))
        ctx.r['gates']['NEW_exact_energy_limb_and_scalar_distance_radical_interval_controls'] = True
        norm = np.frombuffer(S.norm_bytes(ctx), '<f4'); assert np.isfinite(norm).all()
        v, power, bits = M.units(norm)
        encoding = dict(norm_denominator_power=power, norm_maximum_numerator_bits=bits,
                        energy_unsigned_bits=128, limb_bits=20, fixed_limbs=6, no_pickle=True)
        assert len(ctx.b['legacy_cases']) == 127
        ctx.r['gates']['original_G_dyadic_units_caps_admitted_exact_Q_bound_and_ALL127_scalar_cases'] = True
        h_all = np.zeros((128, 3072), '<f8'); bound_all = np.zeros_like(h_all)
        selected_all = np.zeros_like(h_all); rad_all = np.zeros_like(h_all); root_all = np.zeros_like(h_all)
        status_all = np.zeros((128, 3072), 'u1'); cases = []
        pb = ctx.b['exact_certificate']['projector_error_bound']; oq = ctx.b['exact_certificate']['orthogonality_upper']
        original = ROOT / 'results/native_expert_scaling/meth525_geometry'
        with (ctx.out / 'energy_numerators.bin').open('xb') as ef:
            ef.write(S.energy_header(power)); ef.write(b'\0' * (3072 * 16))
            for old in ctx.b['legacy_cases']:
                e = old['expert']; assert e == len(cases) + 1
                wi = np.frombuffer(S.wi_bytes(ctx, e), 'i1').reshape(3072, 768)
                energies = M.energies(wi, v); ef.write(b''.join(n.to_bytes(16, 'little') for n in energies))
                h = np.array([M.norm_up(n, power) for n in energies], '<f8'); h_all[e] = h
                panel = np.load(original / f'e{e:03d}_panel.npy', allow_pickle=False)
                status = np.load(original / f'e{e:03d}_status.npy', allow_pickle=False)
                assert panel.shape == (3072, 7) and panel.dtype == np.dtype('<f8') and status.dtype == np.dtype('u1')
                assert np.sum(status == 0) == 2560 and np.sum(status == 1) == 512
                fresh, safe, rad, root = M.refine(old['radius'], old['y_norm'], panel, h, pb, oq)
                good = status == 0
                usable = good & safe; bound_all[e, usable] = fresh[usable]
                rad_all[e, usable], root_all[e, usable] = rad[usable], root[usable]
                status_all[e, good & ~safe] = 2; status_all[e, usable] = 1
                retained = panel[:, 5].copy(); retained[usable] = np.minimum(retained[usable], fresh[usable]); selected_all[e] = retained
                order = sorted(np.flatnonzero(good), key=lambda j: (panel[j, 4], j)); j = int(order[0])
                assert j == old['selected_neuron']
                upper = math.nextafter(float(panel[j, 4] + retained[j]), math.inf)
                other_lower = min(math.nextafter(float(panel[t, 4] - retained[t]), -math.inf) for t in order[1:])
                resolved = bool(upper < other_lower)
                cases.append(dict(expert=e, anchor_UID=old['anchor_UID'], selected_neuron=j,
                    original_order_resolved=old['order_resolved'], order_resolved=resolved,
                    new_certified_rows=int(np.sum(usable)), old_bound_retained_rows=int(np.sum(good & ~safe)),
                    original_distance2=float(panel[j, 4]), original_selected_bound=float(panel[j, 5]),
                    selected_refined_bound=float(retained[j]), winner_interval_upper=upper,
                    nearest_competitor_interval_lower=other_lower,
                    geometric_relative_displacement_interval=M.relative_interval(float(panel[j, 4]), float(retained[j]), old['y_norm'])))
                ctx.guard()
                if e % 32 == 0: print('energy_bound_parent_complete=' + str(e), flush=True)
        ctx.r['gates']['ALL390144_NEW_exact_row_energies_safe_I64_U128_and_upward_norms_retained'] = True
        for name, array in (('row_norm_upper', h_all), ('new_distance_bounds', bound_all), ('intersection_distance_bounds', selected_all),
                            ('radicand_lower', rad_all), ('root_lower', root_all), ('bound_status', status_all)):
            with (ctx.out / (name + '.npy')).open('xb') as f: np.save(f, array, allow_pickle=False)
        ctx.r['gates']['ALL_original_central_scalars_forward_bounds_positive_radicals_and_interval_intersections'] = True
        preserved = all(not c['original_order_resolved'] or c['order_resolved'] for c in cases)
        eligibility = dict(exact_original_G_ALL_source_row_energy_and_outward_bound_contract=True,
            ALL_original_nominal_neuron_and_distance_identities_unchanged=True,
            ALL109_original_resolved_nearest_orders_preserved=preserved,
            ALL127_retained_nearest_target_orders_resolved=all(c['order_resolved'] for c in cases))
        decision = 'ELIGIBLE_ALL127_RETAINED_ORDERS_FOR_NEW18_PHYSICAL_QUERY_CHECKS' if all(eligibility.values()) else 'RETAINED_NEAREST_TARGET_BOUND_REFINEMENT_CRITERION_CLOSED'
        ctx.r['gates']['ALL127_order_decisions_ALL109_preservation_and_fixed_eligibility'] = True
        write(ctx.out / 'retained_order_freeze.json', dict(cases=cases, original_geometry_changed=False,
            original_query_vectors_recomputed=0, source_function_labels_acquired=0))
        ctx.r['gates']['new_bound_order_decisions_and_zero_query_source_label_scope_frozen'] = True
        ctx.finish(dict(encoding=encoding, cases=cases, resolved_orders=sum(c['order_resolved'] for c in cases),
            newly_resolved_orders=sum(c['order_resolved'] and not c['original_order_resolved'] for c in cases),
            coefficient_energy_rows=390144, coefficient_energy_limb_product_terms=1797783552,
            eligibility=eligibility, decision=decision, views=[], constructed_queries=0,
            new_router_score_vectors=0, selected_source_integer_dot_checks=0,
            source_response_function_calls=0, native_calls=0, model_calls=0, readout_fits=0,
            full_source_WI_projection_rows=0, old_native_router_queries_replayed=0, consumed_rows_for_selection=0,
            old525_geometry_or_queries_recomputed=0, physical_DRAM_verified=False,
            scope='NEW coefficient-energy and scalar error bounds on unchanged525 distance records only; no old projection/query replay, new geometry/label, function quality/useful n or whole promotion.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
