"""Original-F32/I8 exact energy audit and separate scalar-bound/order verification."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from array import array
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth527_operations import Context, DOC, ROOT, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    args = ap.parse_args(); ctx = None
    try:
        ctx = Context('audit', args.binding_sha); rp = DOC / 'meth527_main_result.json'
        assert ctx.digest(rp) == args.main_sha
        raw = json.loads(rp.read_bytes()); assert raw['binding_sha256'] == args.binding_sha and all(raw['gates'].values())
        folder = ROOT / 'results/native_expert_scaling/meth527_bound_refinement'
        for v in raw['output_inventory']: assert Path(v['path']).stat().st_size == v['bytes'] and ctx.digest(v['path']) == v['sha256']
        assert json.loads((folder / 'terminal_resource.json').read_bytes())['result_sha256'] == args.main_sha
        ctx.r['gates']['ALL_original_energy_bound_arrays_and_terminal_SHA_before_independent_verification'] = True
        np = ctx.numpy(); import meth527_contract as S
        import meth527_scalar_audit as A
        ctl = json.loads((folder / 'controls.json').read_bytes())
        assert (-3) ** 2 * 2 ** 2 + 4 ** 2 == ctl['exact_row_energy'] == 52
        assert ctl['energy_norm_upper'] == 8 and 8 ** 2 >= 52 and 7 ** 2 < 52
        lo = Fraction.from_float(ctl['toy_distance'] - ctl['toy_error_bound'])
        hi = Fraction.from_float(ctl['toy_distance'] + ctl['toy_error_bound'])
        assert ((65 - hi) / 8) ** 2 <= Fraction(75, 4) <= ((65 - lo) / 8) ** 2
        assert Fraction.from_float(ctl['toy_root_lower']) ** 2 <= Fraction.from_float(ctl['toy_radicand_lower']) <= Fraction(75, 4)
        expected_control = A.certificate(5., math.sqrt(26), 0., 2.5, 1., 1., 0., 0.)
        assert expected_control == (ctl['toy_error_bound'], ctl['toy_radicand_lower'], ctl['toy_root_lower'])
        ctx.r['gates']['independent_integer_energy_and_exact_distance_radical_interval_controls'] = True
        words = struct.unpack('<768I', S.norm_bytes(ctx)); pairs = []
        for word in words:
            exp = (word >> 23) & 255; assert exp != 255
            mant = word & 0x7fffff
            if exp: mant |= 0x800000
            assert mant != 0
            if word >> 31: mant = -mant
            shift = exp - 150 if exp else -149
            tz = (abs(mant) & -abs(mant)).bit_length() - 1; mant //= 1 << tz; shift += tz
            pairs.append((mant << shift, 0) if shift >= 0 else (mant, -shift))
        power = max(d for n, d in pairs); v = [n << (power - d) for n, d in pairs]
        bits = max(abs(n).bit_length() for n in v); assert power <= 64 and bits <= 52
        encoding = dict(norm_denominator_power=power, norm_maximum_numerator_bits=bits,
                        energy_unsigned_bits=128, limb_bits=20, fixed_limbs=6, no_pickle=True)
        assert raw['encoding'] == encoding; weights = [n * n for n in v]
        ctx.r['gates']['independent_original_F32_bit_decode_exact_G_units_and_U128_limits'] = True
        arrays = {}
        for key in ('row_norm_upper', 'new_distance_bounds', 'intersection_distance_bounds', 'radicand_lower', 'root_lower', 'bound_status'):
            a = np.load(folder / (key + '.npy'), allow_pickle=False)
            assert a.shape == (128, 3072) and a.dtype == np.dtype('u1' if key == 'bound_status' else '<f8')
            assert not np.any(a[0]); arrays[key] = a
        energy_path = folder / 'energy_numerators.bin'
        assert energy_path.stat().st_size == 32 + 128 * 3072 * 16
        original = ROOT / 'results/native_expert_scaling/meth525_geometry'
        cases = []; product_terms = 0; verified_bounds = 0
        pb = ctx.b['exact_certificate']['projector_error_bound']; oq = ctx.b['exact_certificate']['orthogonality_upper']
        with energy_path.open('rb') as ef:
            assert ef.read(32) == S.energy_header(power) and ef.read(3072 * 16) == b'\0' * (3072 * 16)
            for old in ctx.b['legacy_cases']:
                e = old['expert']; coded = array('b'); coded.frombytes(S.wi_bytes(ctx, e)); assert min(coded) >= -127
                wire = ef.read(3072 * 16); assert len(wire) == 3072 * 16
                panel = np.load(original / f'e{e:03d}_panel.npy', allow_pickle=False)
                status = np.load(original / f'e{e:03d}_status.npy', allow_pickle=False)
                assert np.sum(status == 0) == 2560 and np.sum(status == 1) == 512
                for j in range(3072):
                    start = j * 768
                    exact = sum(coded[start + k] ** 2 * weights[k] for k in range(768))
                    assert 0 <= exact < (1 << 128) and exact == int.from_bytes(wire[j * 16:(j + 1) * 16], 'little')
                    root = math.isqrt(exact); root += int(root * root < exact)
                    required = Fraction(root, 1 << power); hn = float(arrays['row_norm_upper'][e, j])
                    assert Fraction.from_float(hn) >= required and Fraction.from_float(math.nextafter(hn, -math.inf)) < required
                    product_terms += 768
                    if status[j] == 1:
                        assert all(arrays[key][e, j] == 0 for key in ('new_distance_bounds', 'intersection_distance_bounds', 'radicand_lower', 'root_lower', 'bound_status'))
                        continue
                    result = A.certificate(old['radius'], old['y_norm'], float(panel[j, 0]), float(panel[j, 3]), float(panel[j, 6]), hn, pb, oq)
                    if result is None:
                        assert arrays['bound_status'][e, j] == 2 and arrays['new_distance_bounds'][e, j] == 0
                        assert arrays['radicand_lower'][e, j] == arrays['root_lower'][e, j] == 0
                        selected = float(panel[j, 5])
                    else:
                        db, arg, rlo = result
                        assert arrays['bound_status'][e, j] == 1
                        assert db == arrays['new_distance_bounds'][e, j] and arg == arrays['radicand_lower'][e, j] and rlo == arrays['root_lower'][e, j]
                        assert Fraction.from_float(rlo) ** 2 <= Fraction.from_float(arg)
                        selected = min(float(panel[j, 5]), db)
                    assert selected == arrays['intersection_distance_bounds'][e, j]; verified_bounds += 1
                selected = arrays['intersection_distance_bounds'][e]
                reps = [j for j in range(3072) if status[j] == 0]
                order = sorted(reps, key=lambda j: (float(panel[j, 4]), j)); j = order[0]
                assert j == old['selected_neuron']
                upper = math.nextafter(float(panel[j, 4]) + float(selected[j]), math.inf)
                lower = min(math.nextafter(float(panel[k, 4]) - float(selected[k]), -math.inf) for k in order[1:])
                cases.append(dict(expert=e, anchor_UID=old['anchor_UID'], selected_neuron=j,
                    original_order_resolved=old['order_resolved'], order_resolved=bool(upper < lower),
                    new_certified_rows=int(np.sum(arrays['bound_status'][e] == 1)), old_bound_retained_rows=int(np.sum(arrays['bound_status'][e] == 2)),
                    original_distance2=float(panel[j, 4]), original_selected_bound=float(panel[j, 5]), selected_refined_bound=float(selected[j]),
                    winner_interval_upper=upper, nearest_competitor_interval_lower=lower,
                    geometric_relative_displacement_interval=A.relative_interval(float(panel[j, 4]), float(selected[j]), old['y_norm'])))
                ctx.guard()
                if e % 32 == 0: print('audited_energy_bound_parent=' + str(e), flush=True)
            assert not ef.read(1)
        assert product_terms == 299630592 and verified_bounds == 325120
        ctx.r['gates']['ALL390144_U128_row_energies_and_least_upward_norms_directly_exactly_verified'] = True
        ctx.r['gates']['ALL325120_original_scalar_bounds_exact_lower_root_squares_and_intersections_independent'] = True
        assert cases == raw['cases']
        preserved = all(not c['original_order_resolved'] or c['order_resolved'] for c in cases)
        eligibility = dict(exact_original_G_ALL_source_row_energy_and_outward_bound_contract=True,
            ALL_original_nominal_neuron_and_distance_identities_unchanged=True,
            ALL109_original_resolved_nearest_orders_preserved=preserved,
            ALL127_retained_nearest_target_orders_resolved=all(c['order_resolved'] for c in cases))
        decision = 'ELIGIBLE_ALL127_RETAINED_ORDERS_FOR_NEW18_PHYSICAL_QUERY_CHECKS' if all(eligibility.values()) else 'RETAINED_NEAREST_TARGET_BOUND_REFINEMENT_CRITERION_CLOSED'
        assert eligibility == raw['eligibility'] and decision == raw['decision'] and len(raw['gates']) == 7
        ctx.r['gates']['ALL_original_ID_distance_AND109_order_preservation_and_nonselecting_displacement_intervals'] = True
        ctx.r['gates']['unchanged_fixed_ALL127_order_eligibility_and_zero_new_query_source_label_decision'] = True
        write(ctx.out / 'verified_retained_bounds.json', dict(main_sha256=args.main_sha, verified_energy_rows=390144,
            verified_energy_direct_product_terms=product_terms, verified_scalar_rows=verified_bounds, original_cases_preserved=True))
        ctx.finish(dict(main_sha256=args.main_sha, encoding=encoding, cases=cases,
            resolved_orders=sum(c['order_resolved'] for c in cases), newly_resolved_orders=sum(c['order_resolved'] and not c['original_order_resolved'] for c in cases),
            coefficient_energy_rows=390144, coefficient_energy_limb_product_terms=1797783552,
            verified_energy_direct_product_terms=product_terms, verified_scalar_rows=verified_bounds,
            eligibility=eligibility, decision=decision, views=[], constructed_queries=0,
            new_router_score_vectors=0, selected_source_integer_dot_checks=0,
            source_response_function_calls=0, native_calls=0, model_calls=0, readout_fits=0,
            full_source_WI_projection_rows=0, old_native_router_queries_replayed=0, consumed_rows_for_selection=0,
            old525_geometry_or_queries_recomputed=0, physical_DRAM_verified=False,
            scope='Independent exact coefficient-energy and ORIGINAL-coordinate scalar-bound audit only; no new geometry/labels, old projection/query replay, function quality, useful n or whole claim.'))
    except BaseException:
        if ctx: ctx.fail()
        raise


if __name__ == '__main__': main()
