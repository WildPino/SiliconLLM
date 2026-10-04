"""Source128 private-matrix delta feasibility; F64 shadow, not native export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
import psutil
import scipy
import torch
from threadpoolctl import threadpool_info, threadpool_limits

import meth443_switch_native_function_causality as N
import meth446_switch_private_delta_math as H

M, R, G, X = N.M, N.R, N.G, N.X
PROTOCOL = M.DOC/'METH_446_SWITCH_PRIVATE_DELTA_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth446_switch_private_delta'
EXPERTS = (0, 1, 64, 127)
TARGETS = EXPERTS[1:]
RANKS = (16, 32, 64, 96, 128)
PARENT = {'meth443_switch_native_causality_result.json': 'b8e8614fb28ad50b91865dba7486c6e4cc6bb8c00cee3bbfe899ab495af5bf80',
          'meth445_switch_relative_contrast_result.json': 'ddd1cd0d8c49e5e6a6b5305eca54e46e12f528ebe8e508dcb175661020b8685f',
          'meth418_switch_function_capture_result.json': N.RECORDS['meth418_switch_function_capture_result.json'],
          'meth420_switch_function_gradient_result.failure.json': N.RECORDS['meth420_switch_function_gradient_result.failure.json']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic()
    numeric_start = None
    peak = hashed = 0
    stage = 'bindings'
    result = {'experiment': 'METH-446-source128-private-matrix-delta-permutation-feasibility', 'candidates': {}, 'alignments': {}}

    def guard():
        nonlocal peak
        memory = psutil.Process().memory_info()
        peak = max(peak, memory.rss, getattr(memory, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        elapsed = time.monotonic()-start
        assert peak <= 3 << 30 and size <= 384 << 20, '3GiB_384MiB'
        assert elapsed <= 900 and ((numeric_start is None and elapsed <= 300) or
               (numeric_start is not None and time.monotonic()-numeric_start <= 600)), 'admission300_numeric600_total900'

    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20):
                h.update(block)
                hashed += len(block)
                guard()
        return h.hexdigest()

    try:
        result['helper_sha256'] = {}
        for path in (Path(__file__), Path(H.__file__), PROTOCOL, Path(N.__file__), Path(X.__file__), Path(M.__file__),
                     Path(R.__file__), Path(R.C.__file__), Path(G.__file__), Path(R.B.__file__), Path(R.C.U.__file__)):
            M.committed(path)
            result['helper_sha256'][str(path)] = sha(path)
        records = {}
        seen = dict(result['helper_sha256'])
        files = {}
        for name, expected in PARENT.items():
            path = M.DOC/name
            M.committed(path)
            assert sha(path) == expected
            records[name] = record = json.loads(path.read_text(encoding='utf-8'))
            for path, expected in record['helper_sha256'].items():
                if path in seen:
                    assert seen[path] == expected
                    continue
                M.committed(Path(path))
                assert sha(path) == expected
                seen[path] = expected
            for item in record.get('output_inventory', []):
                if item['path'] in files:
                    assert files[item['path']] == item['sha256']
                    continue
                assert sha(item['path']) == item['sha256']
                files[item['path']] = item['sha256']
        parent = records['meth443_switch_native_causality_result.json']
        last = records['meth445_switch_relative_contrast_result.json']
        prior = records['meth418_switch_function_capture_result.json']
        baseline = records['meth420_switch_function_gradient_result.failure.json']
        assert all(parent['apparatus_gates'].values()) and parent['diagnostic_gates']['primary_native_identity_mean_KL_ge0_01']
        assert all(last['apparatus_gates'].values()) and last['decision']['passing_balanced_fixed_ranks'] == []
        assert all(prior['gates'].values()) and len(baseline['baselines']) == 384
        for item in baseline['baselines']:
            assert sha(item['archive_path']) == item['archive_sha256']
        for path, expected in parent['preserved_original_binary_sha256'].items():
            assert sha(path) == expected
        result['retained_record_sha256'] = PARENT
        result['preserved_original_binary_sha256'] = parent['preserved_original_binary_sha256']
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['name', 'cmdline']):
            if process.pid in own:
                continue
            name = (process.info['name'] or '').lower()
            argv = process.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons', []).append(process.pid)
                continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_job', process.pid, name)
        psutil.Process().cpu_affinity([0])
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6' and scipy.__version__ == '1.18.1'
        assert psutil.disk_usage(str(M.ROOT)).free >= 2 << 30
        name, expected = R.C.U.EXPORT[128]
        path = M.DOC/name
        M.committed(path)
        assert sha(path) == expected
        export = json.loads(path.read_text(encoding='utf-8'))
        artifact = export['artifact']
        assert artifact == parent['artifacts']['128']
        assert sha(artifact['payload']) == artifact['sha256'] and sha(artifact['manifest']) == artifact['manifest_sha256']
        R.B.read_manifest(artifact['manifest'], export['original_config'], export['tensors'], Path(artifact['payload']))
        initial = (Path(artifact['payload']).stat().st_size, Path(artifact['payload']).stat().st_mtime_ns)
        mapped = np.memmap(artifact['payload'], dtype='u1', mode='r')
        entries = export['tensors']
        result['artifacts'] = {'128': artifact}
        cache = X.ExpertCache(mapped, entries)
        ff = R.C.tensor(mapped, entries, 'decoder.block.11.layer.2.layer_norm.weight')
        caps = {a['label']: a for a in prior['captures']}
        old = {a['label']: a for a in baseline['baselines']}
        captured, keys, books = [], [], []
        for book in range(24):
            for case in range(4):
                item = caps[f'teacher.n128.book{book}.case{case}']
                previous = old[item['label']]
                assert item['prospective_split'] == ('development' if book < 18 else 'validation')
                assert previous['source_capture_sha256'] == item['capture_sha256'] and not previous['qualification_only_not_paired_training']
                data = Path(item['capture_path']).read_bytes()
                assert data[:8] == b'SWFUN001' and struct.unpack_from('<6I', data, 8) == (768, 3072, 32128, 128, 11, 1)
                assert len(data) == 32+14*R.C.DTYPE.itemsize
                rows = np.frombuffer(data, dtype=R.C.DTYPE, offset=32).copy()
                assert np.array_equal(rows['position'], np.arange(14)) and np.array_equal(rows['id'], item['decoder_ids'])
                with np.load(previous['archive_path'], allow_pickle=False) as archive:
                    assert np.array_equal(archive['source_ids'], item['source_ids']) and np.array_equal(archive['decoder_ids'], item['decoder_ids'])
                    for field in ('input', 'up_raw', 'up', 'down', 'probability', 'post', 'final', 'head_input'):
                        R.C.exact(archive[field], rows[field])
                captured.append(rows)
                books.extend([book]*14)
                keys.extend([f'book{book}.case{case}.position{j}:{item["pairing_sha256"]}' for j in range(14)])
        captured = np.concatenate(captured)
        assert captured.shape == (1344,) and keys[1008:] == parent['data']['keys']
        x = captured['input'].astype(np.float64)
        result['data'] = {'keys': keys, 'books': books, 'development_positions': 1008, 'validation_positions': 336,
            'forced_experts': list(EXPERTS), 'captured_selected_counts': {str(e): {'dev': int(np.sum(captured['expert'][:1008] == e)),
                'val': int(np.sum(captured['expert'][1008:] == e))} for e in EXPERTS},
            'scope': 'all fixed source input positions for each forced expert; natural routed exposure is reported separately'}
        numeric_start = time.monotonic()
        assert numeric_start-start <= 300, 'admission300_seconds'
        OUT.mkdir(parents=True)
        result['admission'] = {'seconds': numeric_start-start, 'bytes_hashed': hashed}
        print(json.dumps({'admission_complete': result['admission']}), flush=True)
        with threadpool_limits(limits=1), torch.no_grad():
            result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
                                 'CPU_affinity': [0], 'BLAS': threadpool_info(), 'GPU': False}
            runtime_paths = {a['filepath'] for a in result['runtime']['BLAS']}
            runtime_paths.add(sys.modules[H.linear_sum_assignment.__module__].__file__)
            result['runtime']['binary_sha256'] = {p: sha(p) for p in sorted(runtime_paths)}
            result['tiny_qualification'] = H.tiny_qualification()
            stage = 'all1344_original_selected_FFN_replay'
            for row in captured:
                R.C.exact(G.NativeRMS.apply(torch.from_numpy(row['pre'].copy()), ff).numpy(), row['input'])
                wi, wo = cache.get(int(row['expert']))
                raw = wi.native(row['input'])
                up = np.where(raw < 0, np.float32(0), raw)
                down = wo.native(up)
                R.C.exact(raw, row['up_raw'])
                R.C.exact(up, row['up'])
                R.C.exact(down, row['down'])
                guard()
            stage = 'all_four_forced_native_functions'
            native = np.empty((4, 1344, 768), np.float32)
            native_raw = np.empty((4, 1344, 3072), np.float32)
            weights, operators = {}, {}
            for slot, expert in enumerate(EXPERTS):
                wi, wo = cache.get(expert)
                operators[expert] = (wi, wo)
                weights[expert] = tuple(op.weights.astype(np.float64)*op.scales.astype(np.float64)[:, None] for op in (wi, wo))
                assert weights[expert][0].shape == (3072, 768) and weights[expert][1].shape == (768, 3072)
                for i, row in enumerate(captured):
                    raw = wi.native(row['input'])
                    native_raw[slot, i] = raw
                    native[slot, i] = wo.native(np.where(raw < 0, np.float32(0), raw))
                    if int(row['expert']) == expert:
                        R.C.exact(native[slot, i], row['down'])
                    guard()
            np.savez(OUT/'reference.npz', inputs=captured['input'], pre=captured['pre'], probability=captured['probability'],
                     selected_ids=captured['expert'], keys=np.asarray(keys), native_down=native,
                     base_wi_codes=operators[0][0].weights, base_wi_scales=operators[0][0].scales,
                     base_wo_codes=operators[0][1].weights, base_wo_scales=operators[0][1].scales)
            base_wi, base_wo = weights[0]
            shadows = {}
            result['full_dequant_shadow'] = {}
            for slot, expert in enumerate(EXPERTS):
                shadow = H.forward(x, *weights[expert])
                error = H.relative_rows(shadow, native[slot].astype(np.float64))
                assert float(np.max(error)) <= 1e-3, ('full_dequant_shadow_not_qualified', expert, float(np.max(error)))
                shadows[expert] = shadow
                result['full_dequant_shadow'][str(expert)] = {'median_relative': float(np.median(error)), 'p95_relative': float(np.quantile(error, .95)),
                    'max_relative': float(np.max(error)), 'per_position_relative': error.tolist()}
                guard()
            permutations = {}
            stage = 'unit_WI_assignment_and_native_permutation_replay'
            for slot, expert in enumerate(TARGETS, 1):
                permutation, stats = H.assignment(base_wi, weights[expert][0])
                permutations[expert] = permutation
                result['alignments'][str(expert)] = stats
                wi, wo = operators[expert]
                pwi = G.I8Operator(wi.weights[permutation], wi.scales[permutation])
                pwo = G.I8Operator(wo.weights[:, permutation], wo.scales)
                for i, row in enumerate(captured):
                    raw = pwi.native(row['input'])
                    R.C.exact(raw, native_raw[slot, i][permutation])
                    R.C.exact(pwo.native(np.where(raw < 0, np.float32(0), raw)), native[slot, i])
                    guard()
                R.C.exact(np.sort(permutation), np.arange(3072))
                aligned = (weights[expert][0][permutation], weights[expert][1][:, permutation])
                assert np.max(H.relative_rows(H.forward(x, *aligned), shadows[expert])) <= 1e-10
                guard()
            del native_raw
            parameters, validation, stats, factors_by_candidate = {}, {}, {}, {}

            def measure(value, expert):
                slot = EXPERTS.index(expert)
                error = H.relative_rows(value, native[slot].astype(np.float64))
                return {'dev_median_relative': float(np.median(error[:1008])), 'val_median_relative': float(np.median(error[1008:])),
                    'val_p95_relative': float(np.quantile(error[1008:], .95)), 'per_position_relative': error.tolist(),
                    'per_validation_book_median_relative': {str(b): float(np.median(error[(np.asarray(books) == b)])) for b in range(18, 24)}}

            result['base_only'] = {str(e): measure(shadows[0], e) for e in TARGETS}
            validation['base_only'] = shadows[0][1008:]
            for mode in ('full_matrix', 'delta_identity', 'delta_permuted'):
                stage = mode+'_spectra_and_candidates'
                for expert in TARGETS:
                    target_wi, target_wo = weights[expert]
                    if mode == 'delta_permuted':
                        p = permutations[expert]
                        target_wi, target_wo = target_wi[p], target_wo[:, p]
                    matrices = (target_wi, target_wo) if mode == 'full_matrix' else (target_wi-base_wi, target_wo-base_wo)
                    if mode != 'full_matrix':
                        assert R.relative(base_wi+matrices[0], target_wi) <= 1e-14 and R.relative(base_wo+matrices[1], target_wo) <= 1e-14
                    spectra = tuple(H.spectrum(a) for a in matrices)
                    stats[f'{mode}_e{expert}'] = {layer: {'eigenvalues': spec[0].tolist(), 'matrix_energy': spec[2]} for layer, spec in zip(('wi', 'wo'), spectra)}
                    for rank in RANKS:
                        pairs, spectral_stats = [], {}
                        for layer, a, spec in zip(('wi', 'wo'), matrices, spectra):
                            u, v, info = H.factors(a, spec, rank)
                            prefix = f'{mode}_e{expert}_r{rank}_{layer}'
                            parameters[prefix+'_U'], parameters[prefix+'_V'] = u, v
                            pairs.append((u, v))
                            spectral_stats[layer] = info
                        key = f'{mode}_e{expert}_r{rank}'
                        factors_by_candidate[key] = pairs
                        value = H.factor_forward(x, *pairs, **({} if mode == 'full_matrix' else {'base_wi': base_wi, 'base_wo': base_wo}))
                        validation[key] = value[1008:]
                        metrics = measure(value, expert)
                        coefficient_bytes = sum(v['factor_array_bytes'] for v in spectral_stats.values())
                        source_expert_bytes = 2*768*3072+4*(768+3072)
                        base_bytes = source_expert_bytes if mode != 'full_matrix' else 0
                        provenance_index_bytes = 3072*4 if mode == 'delta_permuted' else 0
                        bank_bytes = base_bytes+(128-(1 if mode != 'full_matrix' else 0))*(coefficient_bytes+provenance_index_bytes)
                        gates = {'val_median_function_relative_le0_05': metrics['val_median_relative'] <= .05,
                            'val_p95_function_relative_le0_10': metrics['val_p95_relative'] <= .10,
                            'each_matrix_stored_F32_energy_ge0_95': all(v['stored_F32_Frobenius_energy_retained'] >= .95 for v in spectral_stats.values()),
                            'nominal128_bank_storage_less_than_I8_source': bank_bytes < 128*source_expert_bytes}
                        result['candidates'][key] = {'metrics': metrics, 'spectra': spectral_stats, 'gates': gates, 'all_feasibility_gates': all(gates.values()),
                            'cost': {'private_factor_bytes': coefficient_bytes, 'shared_base_I8_bytes': base_bytes,
                                'alignment_conversion_index_bytes_charged_per_expert': provenance_index_bytes, 'nominal128_bank_bytes': bank_bytes,
                                'source128_bank_bytes': 128*source_expert_bytes, 'private_factor_coefficients': 2*rank*(768+3072),
                                'top1_arithmetic_coefficients': 2*rank*(768+3072)+(2*768*3072 if mode != 'full_matrix' else 0),
                                'scope': 'nominal extrapolation only; only3 targets measured, no headers/alignment for remaining IDs/native timing/DRAM'}}
                        print(json.dumps({'candidate': key, 'val_median': metrics['val_median_relative'], 'val_p95': metrics['val_p95_relative'], 'gates': gates}), flush=True)
                        guard()
            result['permutation_control'] = {}
            for mode in ('delta_identity', 'delta_permuted'):
                for rank in RANKS:
                    for index, expert in enumerate(TARGETS):
                        foreign = TARGETS[(index+1) % len(TARGETS)]
                        key, wrong = f'{mode}_e{expert}_r{rank}', f'{mode}_e{foreign}_r{rank}'
                        value = H.factor_forward(x, *factors_by_candidate[wrong], base_wi=base_wi, base_wo=base_wo)
                        validation[key+'_private_permutation'] = value[1008:]
                        metrics = measure(value, expert)
                        harm = metrics['val_median_relative']-result['candidates'][key]['metrics']['val_median_relative']
                        result['permutation_control'][key] = {'foreign_private_expert': foreign, 'metrics': metrics,
                            'val_median_relative_harm': harm, 'descriptive_harm_ge0_01': harm >= .01}
                        guard()
            np.savez(OUT/'factors.npz', **parameters, **{f'permutation_e{e}': p for e, p in permutations.items()})
            np.savez(OUT/'validation_functions.npz', **validation)
            result['matrix_spectra'] = stats
            result['apparatus_gates'] = {'all_fresh_sources_parents_archives': True, 'all1344_original_native_FFN_states_byte_exact': True,
                'all_four_forced_experts_on_all1344_source_inputs': True, 'all_three_matched_permutations_native_byte_exact': True,
                'full_dequant_shadow_all_positions_relative_le1e_minus3': True, 'tiny_assignment_and_both_factor_orientations_qualified': True,
                'all_matrix_spectra_tail_F32_factors_finite_qualified': True, 'same_ranks_factor_precision_inputs_no_gradient_fit': True,
                'base_removal_and_private_ID_cycle_controls_complete': True}
        assert initial == (Path(artifact['payload']).stat().st_size, Path(artifact['payload']).stat().st_mtime_ns)
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource'] = {'seconds': time.monotonic()-start, 'admission_seconds': numeric_start-start, 'numeric_and_reporting_seconds': time.monotonic()-numeric_start,
            'peak_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': sum(a['bytes'] for a in result['output_inventory']), 'optimizer_updates': 0}
        eligible = []
        for mode in ('full_matrix', 'delta_identity', 'delta_permuted'):
            for rank in RANKS:
                if all(result['candidates'][f'{mode}_e{e}_r{rank}']['all_feasibility_gates'] for e in TARGETS):
                    eligible.append({'mode': mode, 'rank': rank})
        result['decision'] = {'all_three_target_eligible_recipes': eligible,
            'next': 'qualify_native_factor_head_quality_and_actual_cost_before_training_or_fullbank_export' if eligible else
                    'fixed_reference0_Frobenius_private_delta_and_fullmatrix_factor_recipes_inadequate_before_native_export'}
        result['scope'] = 'One source128 bank11, reference0/targets1,64,127;1008dev/336consumed val source inputs forced for each function. Matching optimizes unit-WI cost only. Matrix Frobenius spectral factors are F32; candidate inference is dequantized F64 shadow, NOT native I8/A16 factor inference/export. No full-head KL/truth/task/rollout/n-usefulness/rate/LUT/actualDRAM/family/~100B proof; nominal128-bank cost extrapolation not transformed128-bank artifact. Closed older global-output recipes unchanged.'
        guard()
        M.write(args.out, result)
        print(json.dumps({'apparatus': result['apparatus_gates'], 'decision': result['decision'], 'resource': result['resource'], 'sha256': M.digest(args.out)}), flush=True)
    except BaseException as error:
        result.update(failure_stage=stage, error=repr(error), seconds=time.monotonic()-start, peak_bytes=peak, bytes_hashed=hashed)
        if OUT.exists():
            result['partial_output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'), result)
        raise


if __name__ == '__main__':
    main()
