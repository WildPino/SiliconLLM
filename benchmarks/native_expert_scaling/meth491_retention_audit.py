"""Independent interval/exact-integer validation; no SVD, fitting or native call."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from decimal import Context as DecimalContext, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
from fractions import Fraction
import json
from pathlib import Path
import struct
import traceback
from meth491_operations import Context, DOC, ROOT

OUT = ROOT / 'results/native_expert_scaling/meth491_mass_interval'
AUD = ROOT / 'results/native_expert_scaling/meth491_mass_interval_audit'
RAW = DOC / 'meth491_mass_interval_result.json'
RET = DOC / 'RETENTION_491_20261006.json'

def audit():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--binding-sha', required=True); args = parser.parse_args()
    assert args.out.resolve() == RET.resolve()
    ctx = Context(AUD, RET, 300, 256 << 20)
    try:
        binding = ctx.admit(args.binding_sha); ctx.binding = binding
        completed = RAW.exists(); path = RAW if completed else RAW.with_suffix('.failure.json')
        ctx.head(path)
        raw_sha = ctx.digest(path); raw = json.loads(path.read_bytes())
        regpath = DOC / 'meth491_main_sole_first_registration.json'; ctx.head(regpath)
        registration = json.loads(regpath.read_bytes())
        assert registration['attempt'] == 1 and registration['process_instance'] == raw['process_instance']
        assert registration['record_sha256'] == raw_sha
        assert registration['actual_exit_code'] == 0 if completed else registration['actual_exit_code'] != 0
        eventpath = ROOT / 'results/native_expert_scaling/meth491_windows_terminal.json'
        windows = json.loads(eventpath.read_bytes())
        import datetime
        assert windows['query_available'] and windows['query_error'] is None and not windows['matching_scientific_events']
        assert windows['instances'][0]['pid'] == raw['process_instance']['pid']
        assert windows['instances'][0]['create_time_unix'] == raw['process_instance']['create_time_unix']
        assert all(datetime.datetime.fromisoformat(windows['instances'][0][key]) == datetime.datetime.fromisoformat(raw[key]) for key in ('start_utc', 'end_utc'))
        inventory = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': ctx.digest(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
        ctx.r['gates']['sole_actual_main_terminal_Windows_and_fresh_output_inventory'] = True
        ctx.r['raw'] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': raw_sha}
        ctx.r['main_windows_sha256'] = ctx.digest(eventpath)
        if not completed:
            result = ctx.finish({'main_completed': False, 'numerical_imports': False,
                'retained_partial_inventory': inventory, 'decision': 'FIRST_MAIN_FAULT_RETAINED_NO_WITNESS_PROMOTION'})
            print(json.dumps({'retention': str(RET), 'gates': result['gates'], 'decision': result['decision']})); return
        assert inventory == raw['output_inventory'] and not raw['commands']
        assert raw['optimizer_updates'] == raw['native_calls'] == 0 and all(raw['gates'].values())
        import numpy as np
        import threadpoolctl
        ctx.r['numerical_imports'] = True
        threadpoolctl.threadpool_limits(limits=1)
        assert all(pool['num_threads'] == 1 for pool in threadpoolctl.threadpool_info())
        tiny = np.array([1], '<u8').view('<f8')
        assert (tiny+tiny).view('<u8')[0] == 2
        assert float(np.float32(1+2**-24)) == float(np.float32(1-2**-25)) == 1
        near = DecimalContext(prec=200, rounding=ROUND_HALF_EVEN)
        down = DecimalContext(prec=200, rounding=ROUND_FLOOR)
        up = DecimalContext(prec=200, rounding=ROUND_CEILING)
        eta = Decimal.from_float(float(raw['eta']))
        inherited = json.loads(Path(binding['mass_source']['raw']['path']).read_bytes())
        assert raw['eta'] == inherited['log_error_budget'] and raw['decimal_precision'] == 100
        def bracket(value, operation):
            rounded = getattr(near, operation)(value)
            return near.next_minus(rounded), near.next_plus(rounded)
        negative_exp = bracket(eta.copy_negate(), 'exp'); positive_exp = bracket(eta, 'exp')
        def endpoint(q, is_right):
            q = Decimal.from_float(float(q)); logq = bracket(q, 'ln')
            bounds = []
            for exponent, offset in ((negative_exp, eta.copy_negate()), (positive_exp, eta)):
                prob_low = down.multiply(q, exponent[0]); prob_high = up.multiply(q, exponent[1])
                assert 0 < prob_low <= prob_high < 1
                complement_low = down.subtract(Decimal(1), prob_high)
                complement_high = up.subtract(Decimal(1), prob_low)
                comp_ln_low = bracket(complement_low, 'ln')[0]
                comp_ln_high = bracket(complement_high, 'ln')[1]
                low = down.subtract(down.add(logq[0], offset), comp_ln_high)
                high = up.subtract(up.add(logq[1], offset), comp_ln_low)
                bounds.append((low, high))
            if is_right:
                return bounds
            return [(bounds[1][1].copy_negate(), bounds[1][0].copy_negate()),
                    (bounds[0][1].copy_negate(), bounds[0][0].copy_negate())]
        def combine(weights, bounds):
            sums = [Decimal(0), Decimal(0)]
            for weight, limits in zip(weights, bounds):
                weight = int(weight); used = limits[0] if weight >= 0 else limits[1]
                low, high = used if weight >= 0 else (used[1], used[0])
                sums[0] = down.add(sums[0], down.multiply(Decimal(weight), low))
                sums[1] = up.add(sums[1], up.multiply(Decimal(weight), high))
            return sums
        controls = json.loads((OUT / 'math_controls.json').read_bytes())
        assert controls['F32_exact_decoder_controls'] == 9 and controls['q1_upper_unbounded']
        assert controls['eta0_qhalf_interval_contains_zero'] and controls['known_collinear_residual_zero']
        toy = combine([-1, 2, -1], [endpoint(q, True) for q in (.5, .95, .5)])
        assert Decimal(controls['known_collinear_contradiction_lower']) <= toy[0] <= toy[1] <= Decimal(controls['known_collinear_contradiction_upper'])
        assert toy[0] > 5
        ctx.r['gates']['independent200digit_known_exact_collinear_contradiction_control'] = True
        U, Q = 238872, 387036
        with Path(binding['data']['ownership.bin']['path']).open('rb') as stream:
            assert stream.read(24) == struct.pack('<8sIIQ', b'M479OWN1', 32, 0, U)
            owner = np.frombuffer(stream.read(), '<u4').reshape(U, 8)
        dtype = np.dtype([('uid', '<u4'), ('qleft', '<f8'), ('qright', '<f8'),
                          ('right', '<u4'), ('error', '<f8'), ('relative', '<f8'), ('ideal', '<f8')])
        with Path(binding['mass_source']['diagnostics']['path']).open('rb') as stream:
            assert stream.read(24) == struct.pack('<8sIIQ', b'M490DIA1', 48, 0, U)
            diag = np.frombuffer(stream.read(), dtype)
        ids = np.arange(U, dtype='<u4')
        assert np.array_equal(owner[:, 0], ids) and np.array_equal(diag['uid'], ids)
        q = np.where(diag['right'], diag['qright'], diag['qleft'])
        dev, val = (owner[:, 2] & 5) != 0, (owner[:, 2] & 2) != 0
        assert (int(dev.sum()), int(val.sum())) == (159414, 79458) and not np.any(dev & val)
        meta = np.empty((U, 12), '<u4')
        UNIQUE = np.dtype({'names': ['meta', 'x'], 'formats': [('<u4', (12,)), ('<f4', (768,))],
                           'offsets': [0, 136], 'itemsize': 3720})
        stored = {}; lookup = {}; loaded = {}
        for row in raw['witnesses']:
            bank = row['bank']; assert bank not in stored
            stored[bank] = row
            if row['selected_inputs'] == 770:
                with np.load(row['witness_path'], allow_pickle=False) as archive:
                    loaded[bank] = {key: archive[key].copy() for key in archive.files}
                h = loaded[bank]
                assert h['input'].shape == (770, 768) and h['input'].dtype == np.dtype('<f4')
                for index, uid in enumerate(h['uid']):
                    assert int(uid) not in lookup; lookup[int(uid)] = (bank, index)
        assert set(stored) == set(range(12))
        joined = set(); ctx.phase = 'ALL_canonical_metadata_and_exact_witness_input_join'
        with Path(binding['data']['unique_inputs.bin']['path']).open('rb') as stream:
            assert stream.read(24) == struct.pack('<8sIIQ', b'M479UNI1', 3720, 768, U)
            for start in range(0, U, 1024):
                data = stream.read(min(1024, U-start)*3720); rows = np.frombuffer(data, UNIQUE)
                assert len(rows) == min(1024, U-start)
                meta[start:start+len(rows)] = rows['meta']
                for item in rows:
                    uid = int(item['meta'][0])
                    if uid in lookup:
                        bank, index = lookup[uid]
                        assert uid not in joined and int(item['meta'][2]) == bank
                        assert item['x'].tobytes() == loaded[bank]['input'][index].tobytes(); joined.add(uid)
                ctx.guard()
            assert not stream.read(1)
        assert joined == set(lookup) and np.array_equal(meta[:, 0], ids) and np.array_equal(meta[:, 2], owner[:, 1])
        with Path(binding['data']['query_links.bin']['path']).open('rb') as stream:
            assert stream.read(24) == struct.pack('<8sIIQ', b'M479LNK1', 52, 0, Q)
            links = np.frombuffer(stream.read(), '<u4').reshape(Q, 13)
        assert np.array_equal(links[:, 6], meta[links[:, 12], 2]) and np.array_equal(links[:, 9], meta[links[:, 12], 7])
        assert raw['unique_inputs'] == U and raw['occurrences'] == Q
        for coverage in raw['coverage']:
            mask = owner[:, 1] == coverage['bank']
            if coverage['role'] != 'ALL':
                mask &= dev if coverage['role'] == 'development' else val
            assert coverage['count'] == int(mask.sum())
            assert coverage['source_ID_counts'] == np.bincount(meta[mask, 7], minlength=128).tolist()
        assert len(raw['coverage']) == 36
        ctx.r['gates']['ALL_source_roles_links_ID_counts_and_saved_F32_inputs_BYTE'] = True
        results = []; exact_coordinates = 0
        for bank in range(12):
            ctx.quiet('audit_bank'+str(bank)); ctx.phase = 'independent_exact_dyadic_and200digit_mass_witness'
            row = stored[bank]; development_ids = np.flatnonzero(dev & (owner[:, 1] == bank))
            eligible = []; first = {}
            for index, uid in enumerate(development_ids):
                value = Decimal.from_float(float(q[uid]))
                assert not negative_exp[0] <= value <= negative_exp[1], 'ambiguous interval eligibility'
                if value < negative_exp[0]:
                    eligible.append(int(uid)); first.setdefault(int(meta[uid, 7]), int(uid))
                if index % 1024 == 0:
                    ctx.guard()
            chosen = [first[key] for key in sorted(first)]
            seen = set(chosen)
            for uid in eligible:
                if len(chosen) >= 770:
                    break
                if uid not in seen:
                    chosen.append(uid); seen.add(uid)
            assert row['selected_inputs'] == len(chosen) and row['finite_two_sided_development'] == len(eligible)
            assert row['development_inputs'] == len(development_ids)
            assert row['not_two_sided_development'] == len(development_ids)-len(eligible)
            eligible_counts = np.bincount(meta[eligible, 7], minlength=128).tolist()
            assert row['finite_two_sided_source_ID_counts'] == eligible_counts
            exposed_counts = np.bincount(meta[development_ids, 7], minlength=128).tolist()
            assert row['missing_two_sided_source_IDs'] == [i for i in range(128) if exposed_counts[i] and not eligible_counts[i]]
            assert row['selected_source_ID_counts'] == np.bincount(meta[chosen, 7], minlength=128).tolist()
            if len(chosen) != 770:
                assert row['decision'] == 'ELIGIBILITY_INCONCLUSIVE_FEWER_THAN770_TWO_SIDED_STATES'
                results.append({'bank': bank, 'decision': row['decision']}); continue
            h = loaded[bank]; assert h['uid'].tolist() == chosen
            quantized = [round(float(value)*(1 << 40)) for value in h['basis']]
            assert h['quantized_weights'].tolist() == quantized
            with np.load(OUT / ('bank%02d_constructor.npz' % bank), allow_pickle=False) as constructor:
                assert all(constructor[key].tobytes() == h[key].tobytes() for key in constructor.files)
            plus = list(map(Decimal, row['plus_C_interval'])); minus = list(map(Decimal, row['minus_C_interval']))
            sign = 1 if plus[0] >= minus[0] else -1
            assert sign == row['sign']
            weights = [sign*value for value in quantized]
            assert h['signed_weights'].tolist() == weights and any(weights)
            endpoints = [endpoint(q[uid], bool(diag['right'][uid])) for uid in chosen]
            for integers, saved in ((quantized, plus), ([-v for v in quantized], minus)):
                independently = combine(integers, endpoints)
                assert saved[0] <= independently[0] <= independently[1] <= saved[1]
            selected = plus if sign == 1 else minus
            assert list(map(str, selected)) == row['chosen_C_interval']
            # Different decoder: Python float exact ratio, then a fixed F32 denominator.
            residual = [0] * 769
            for index, (xrow, weight) in enumerate(zip(h['input'], weights)):
                for column, value in enumerate(xrow):
                    numerator, denominator = float(value).as_integer_ratio()
                    assert (1 << 149) % denominator == 0
                    residual[column] += weight*numerator*((1 << 149)//denominator)
                residual[768] += weight*(1 << 149)
                if index % 64 == 0:
                    ctx.guard()
            assert list(map(str, residual)) == row['residual_numerators']
            maximum = max(map(abs, residual)); exact_coordinates += 769
            assert str(maximum) == row['residual_max_abs_numerator'] and row['residual_denominator_power2'] == 149
            if selected[0] <= 0:
                assert row['coefficient_L1_lower_bound'] is None and row['decision'] == 'THIS_FIXED_WITNESS_INCONCLUSIVE_NONPOSITIVE_GAP'
            elif maximum == 0:
                assert row['coefficient_L1_lower_bound'] == 'UNRESTRICTED_INFEASIBILITY'
                assert row['decision'] == 'UNRESTRICTED_IDEAL_AFFINE_INFEASIBILITY_EXACT_WITNESS'
            else:
                necessary = Fraction(selected[0])*(1 << 149)/maximum
                declared = Fraction(Decimal(row['coefficient_L1_lower_bound']))
                assert 0 < declared <= necessary and necessary-declared <= necessary/(10**98)
                assert row['decision'] == 'NECESSARY_IDEAL_RAW_COEFFICIENT_L1_LOWER_BOUND'
            singular = h['singular_values']; assert singular.shape == (769,) and np.all(singular >= 0) and np.all(np.diff(singular) <= 0)
            assert float(singular[0]) == row['singular_maximum'] and float(singular[-1]) == row['singular_minimum']
            results.append({'bank': bank, 'decision': row['decision'], 'coefficient_L1_lower_bound': row['coefficient_L1_lower_bound']})
            ctx.log(bank_audited=bank, decision=row['decision']); ctx.guard()
        ctx.r['gates']['ALL12_development_subsets_and_fixed_integer_constructor_without_SVD_replay'] = True
        ctx.r['gates']['ALL_saved_signed_exact_dyadic_residual_coordinates'] = True
        ctx.r['gates']['independent200digit_outward_mass_endpoints_C_and_parametric_norm_bounds'] = True
        assert raw['resource']['wall_seconds'] <= 180 and raw['resource']['parent_peak_bytes']+raw['resource']['native_peak_bytes'] <= 512 << 20
        ctx.r['gates']['main180s512MiB_audit300s256MiB_zero_model_native_fit_SVD_replay'] = True
        result = ctx.finish({'main_completed': True, 'witnesses_audited': len(results),
            'exact_residual_coordinates_audited': exact_coordinates, 'unique_inputs_joined': U,
            'native_calls': 0, 'optimizer_updates': 0, 'SVD_calls': 0,
            'witnesses': results, 'decision': 'ALL_FIXED_PARAMETRIC_MASS_WITNESSES_INDEPENDENTLY_ADMITTED',
            'scope': 'Each valid lower bound applies to complete ideal development constraints; no primal, C rounding, fresh quality, rate or generic affine-class promotion.'})
        print(json.dumps({'retention': str(RET), 'gates': result['gates'], 'witnesses': results, 'resource': result['resource']}))
    except BaseException as exc:
        ctx.fail(exc); ctx.done.set(); ctx.watchdog.cancel(); traceback.print_exc(); raise

if __name__ == '__main__':
    audit()
