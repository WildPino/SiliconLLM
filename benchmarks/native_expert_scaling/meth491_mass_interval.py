"""One development-only small dual witness per root; no model, fit or native call."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
import traceback
from meth491_operations import Context, DOC, ROOT, write

OUT = ROOT / 'results/native_expert_scaling/meth491_mass_interval'
RAW = DOC / 'meth491_mass_interval_result.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--binding-sha', required=True)
    args = parser.parse_args()
    assert args.out.resolve() == RAW.resolve()
    ctx = Context(OUT, RAW, 180, 512 << 20)
    try:
        binding = ctx.admit(args.binding_sha)
        ctx.binding = binding
        import numpy as np
        import threadpoolctl
        import meth491_mass_interval_math as math
        ctx.r['numerical_imports'] = True
        threadpoolctl.threadpool_limits(limits=1)
        assert all(p['num_threads'] == 1 for p in threadpoolctl.threadpool_info())
        tiny = np.array([1], '<u8').view('<f8')
        assert (tiny + tiny).view('<u8')[0] == 2
        assert float(np.float32(1 + 2**-24)) == float(np.float32(1 - 2**-25)) == 1
        assert np.rint(np.array([-.5, .5, 1.5, 2.5])).tolist() == [-0., 0., 2., 2.]
        ctx.r['threadpools'] = threadpoolctl.threadpool_info()
        source = json.loads(Path(binding['mass_source']['raw']['path']).read_bytes())
        ret = json.loads(Path(binding['mass_source']['retention']['path']).read_bytes())
        admission = json.loads(Path(binding['mass_source']['admission']['path']).read_bytes())
        assert all(source['gates'].values()) and all(ret['gates'].values()) and all(admission['gates'].values())
        assert admission['main_completed'] and ret['unique_inputs_audited'] == 238872
        eta = source['log_error_budget']
        ctx.phase = 'new_exact_dyadic_and_directed_interval_controls'
        write(OUT / 'math_controls.json', math.controls(eta))
        ctx.r['gates']['F32_dyadic_and_directed_mass_interval_controls'] = True
        interval = math.Intervals(eta)
        U, Q = 238872, 387036
        with Path(binding['data']['ownership.bin']['path']).open('rb') as stream:
            assert stream.read(24) == struct.pack('<8sIIQ', b'M479OWN1', 32, 0, U)
            owner = np.frombuffer(stream.read(), '<u4').reshape(U, 8)
        dtype = np.dtype([('uid', '<u4'), ('source', '<f8', (2,)), ('right', '<u4'),
                          ('error', '<f8'), ('relative', '<f8'), ('ideal', '<f8')])
        with Path(binding['mass_source']['diagnostics']['path']).open('rb') as stream:
            assert stream.read(24) == struct.pack('<8sIIQ', b'M490DIA1', 48, 0, U)
            diag = np.frombuffer(stream.read(), dtype)
        assert np.array_equal(owner[:, 0], np.arange(U)) and np.array_equal(diag['uid'], owner[:, 0])
        assert np.all(diag['right'] <= 1)
        dev, val = (owner[:, 2] & 5) != 0, (owner[:, 2] & 2) != 0
        assert (int(dev.sum()), int(val.sum())) == (159414, 79458) and not np.any(dev & val)
        selected_q = np.where(diag['right'], diag['source'][:, 1], diag['source'][:, 0])
        assert np.isfinite(selected_q).all() and np.all((selected_q > 0) & (selected_q <= 1))
        UNIQUE = np.dtype([('meta', '<u4', (12,)), ('hashes', 'u1', (64,)),
                           ('values', '<f8', (3,)), ('x', '<f4', (768,)), ('scores', '<f4', (128,))])
        meta = np.empty((U, 12), '<u4')
        ctx.r['source_scan_bytes'] = 0
        def scan():
            with Path(binding['data']['unique_inputs.bin']['path']).open('rb') as stream:
                assert stream.read(24) == struct.pack('<8sIIQ', b'M479UNI1', 3720, 768, U)
                ctx.r['source_scan_bytes'] += 24
                for start in range(0, U, 2048):
                    payload = stream.read(min(2048, U-start) * 3720)
                    rows = np.frombuffer(payload, UNIQUE)
                    assert len(rows) == min(2048, U-start)
                    ctx.r['source_scan_bytes'] += len(payload)
                    ctx.guard()
                    yield start, rows
                assert not stream.read(1)
        ctx.phase = 'ALL_source_roles_and_metadata_scan'
        for start, rows in scan():
            meta[start:start+len(rows)] = rows['meta']
        assert np.array_equal(meta[:, 0], owner[:, 0]) and np.array_equal(meta[:, 2], owner[:, 1])
        assert np.all(meta[:, 7] < 128) and np.all(owner[:, 1] < 12)
        with Path(binding['data']['query_links.bin']['path']).open('rb') as stream:
            assert stream.read(24) == struct.pack('<8sIIQ', b'M479LNK1', 52, 0, Q)
            links = np.frombuffer(stream.read(), '<u4').reshape(Q, 13)
        assert np.array_equal(links[:, 6], meta[links[:, 12], 2]) and np.array_equal(links[:, 9], meta[links[:, 12], 7])
        coverage = []
        subsets = []
        finite = np.zeros(U, bool)
        ctx.phase = 'ALL_development_finite_interval_eligibility_and_fixed_subsets'
        for bank in range(12):
            ctx.quiet('eligibility_bank'+str(bank))
            ids = np.flatnonzero((owner[:, 1] == bank) & dev)
            for offset, uid in enumerate(ids):
                finite[uid] = interval.finite(selected_q[uid])
                if offset % 1024 == 0:
                    ctx.guard()
            at = ids[finite[ids]]
            chosen = []
            for expert in range(128):
                eligible = at[meta[at, 7] == expert]
                if len(eligible):
                    chosen.append(int(eligible[0]))
            chosen_set = set(chosen)
            for uid in at:
                if len(chosen) == 770:
                    break
                if int(uid) not in chosen_set:
                    chosen.append(int(uid)); chosen_set.add(int(uid))
            ids_selected = np.array(chosen, '<u4')
            subsets.append(ids_selected)
            for role, mask in (('ALL', owner[:, 1] == bank), ('development', (owner[:, 1] == bank) & dev),
                               ('consumed_validation', (owner[:, 1] == bank) & val)):
                coverage.append({'bank': bank, 'role': role, 'count': int(mask.sum()),
                    'source_ID_counts': np.bincount(meta[mask, 7], minlength=128).tolist()})
            assert len(ids_selected) in (770, len(at))
            ctx.log(eligibility_bank=bank, finite_development=len(at), selected=len(ids_selected))
        ctx.r['gates']['ALL_UID_roles_links_ID_coverage_and_deterministic_development_subsets'] = True
        lookup = np.full(U, -1, '<i4')
        inputs = []
        for bank, subset in enumerate(subsets):
            lookup[subset] = bank * 770 + np.arange(len(subset))
            inputs.append(np.empty((len(subset), 768), '<f4'))
        filled = np.zeros(U, bool)
        ctx.phase = 'original_F32_witness_input_second_scan'
        for _, rows in scan():
            indices = rows['meta'][:, 0]
            assert np.isfinite(rows['x']).all()
            selected = rows[lookup[indices] >= 0]
            for row in selected:
                uid = int(row['meta'][0]); code = int(lookup[uid])
                assert not filled[uid] and code//770 == int(row['meta'][2])
                inputs[code//770][code % 770] = row['x']; filled[uid] = True
        assert np.all(filled[lookup >= 0])
        ctx.r['gates']['ALL_selected_inputs_BYTE_joined_to_original_canonical_F32'] = True
        witnesses = []
        ctx.r['witnesses'] = witnesses
        ctx.phase = 'ONE_SVD_integer_witness_and_exact_residual_per_root'
        for bank, (ids, x) in enumerate(zip(subsets, inputs)):
            ctx.quiet('witness_bank'+str(bank)); ctx.guard()
            row = {'bank': bank, 'development_inputs': int(np.count_nonzero(dev & (owner[:, 1] == bank))),
                   'finite_two_sided_development': int(np.count_nonzero(finite & (owner[:, 1] == bank))),
                   'selected_inputs': len(ids), 'selected_source_ID_counts': np.bincount(meta[ids, 7], minlength=128).tolist()}
            finite_ids = np.flatnonzero(finite & (owner[:, 1] == bank))
            row['not_two_sided_development'] = row['development_inputs']-len(finite_ids)
            row['finite_two_sided_source_ID_counts'] = np.bincount(meta[finite_ids, 7], minlength=128).tolist()
            exposure = np.bincount(meta[dev & (owner[:, 1] == bank), 7], minlength=128)
            row['missing_two_sided_source_IDs'] = [i for i in range(128) if exposure[i] and not row['finite_two_sided_source_ID_counts'][i]]
            if len(ids) != 770:
                row['decision'] = 'ELIGIBILITY_INCONCLUSIVE_FEWER_THAN770_TWO_SIDED_STATES'
                witnesses.append(row); continue
            phi = np.ones((770, 769), '<f8'); phi[:, :768] = x
            u, singular, vh = np.linalg.svd(phi, full_matrices=True)
            basis = u[:, -1].copy(); del u, vh, phi
            assert np.isfinite(basis).all() and np.isfinite(singular).all()
            assert abs(float(basis @ basis)-1) < 1e-10
            weights = np.rint(basis * (1 << 40)).astype('<i8')
            assert np.any(weights) and np.max(np.abs(weights)) <= 1 << 40
            np.savez(OUT / ('bank%02d_constructor.npz' % bank), uid=ids, basis=basis,
                     singular_values=singular, quantized_weights=weights)
            endpoints = [interval.endpoints(selected_q[uid], bool(diag['right'][uid])) for uid in ids]
            assert all(v is not None for v in endpoints)
            plus = interval.combination(weights, endpoints)
            minus = interval.combination(-weights, endpoints)
            sign = 1 if plus[0] >= minus[0] else -1
            signed = weights * sign
            chosen_bounds = plus if sign == 1 else minus
            path = OUT / ('bank%02d_witness.npz' % bank)
            np.savez(path, uid=ids, input=x, basis=basis, singular_values=singular,
                     quantized_weights=weights, signed_weights=signed)
            residual = math.exact_residual(x.view('<u4'), signed, ctx.guard)
            maximum = max(map(abs, residual))
            bound = interval.norm_bound(chosen_bounds[0], maximum)
            if chosen_bounds[0] <= 0:
                decision = 'THIS_FIXED_WITNESS_INCONCLUSIVE_NONPOSITIVE_GAP'
            elif maximum == 0:
                decision = 'UNRESTRICTED_IDEAL_AFFINE_INFEASIBILITY_EXACT_WITNESS'
            else:
                decision = 'NECESSARY_IDEAL_RAW_COEFFICIENT_L1_LOWER_BOUND'
            row.update(witness_path=str(path), sign=sign, plus_C_interval=list(map(str, plus)),
                minus_C_interval=list(map(str, minus)), chosen_C_interval=list(map(str, chosen_bounds)),
                residual_numerators=list(map(str, residual)), residual_denominator_power2=149,
                residual_max_abs_numerator=str(maximum), coefficient_L1_lower_bound=bound,
                singular_minimum=float(singular[-1]), singular_maximum=float(singular[0]),
                decision=decision)
            witnesses.append(row)
            ctx.r['completed_roots'] = bank+1
            ctx.log(witness_bank=bank, decision=decision, coefficient_L1_lower_bound=bound)
            ctx.guard()
        ctx.r['gates']['ALL12_fixed_constructor_outcomes_and_exact_dyadic_residuals'] = True
        result = ctx.finish({'eta': eta, 'decimal_precision': 100, 'quantization_power2': 40,
            'residual_denominator_power2': 149, 'unique_inputs': U, 'occurrences': Q,
            'development_inputs': 159414, 'consumed_validation_inputs': 79458,
            'coverage': coverage, 'witnesses': witnesses, 'native_calls': 0, 'optimizer_updates': 0,
            'decision': 'PARAMETRIC_MASS_INTERVAL_WITNESSES_REQUIRE_INDEPENDENT_ADMISSION',
            'scope': 'Ideal affine mass on raw F32 development states; subset dual direction only, no primal/physical/fresh quality/rate claim.'})
        print(json.dumps({'raw': str(RAW), 'gates': result['gates'],
            'witnesses': [{k: v[k] for k in ('bank', 'decision', 'coefficient_L1_lower_bound') if k in v} for v in witnesses],
            'resource': result['resource']}), flush=True)
    except BaseException as exc:
        ctx.fail(exc); ctx.done.set(); ctx.watchdog.cancel(); traceback.print_exc(); raise

if __name__ == '__main__':
    main()
