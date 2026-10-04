"""Shared output-readout oracle feasibility with matched own-input control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth436_switch_oracle_function_input as Q
import meth437_switch_shared_readout_math as H
X, M, R, G = Q.X, Q.M, Q.R, Q.G
PROTOCOL = M.DOC/'METH_438_SWITCH_SHARED_READOUT_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth438_switch_shared_readout'
RECORDS = Q.RECORDS | {'meth436_switch_oracle_input_result.json': 'ece16dfce21260fa5a2fd98cb8595fe489c8e7447812cb05dfe85da76bedd6b4', 'meth437_switch_shared_readout_result.failure.json': 'b2ae21e98f04f14dd5e2e764818bff61bc1e44bd59b4c1b30448b15b8d69b0f4'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; updates = []; result = {'experiment': 'METH-438-shared-output-readout-oracle-with-own-input-control', 'updates': updates}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        assert peak <= 4 << 30 and time.monotonic()-start <= 720 and size <= 320 << 20, 'pilot_720sec_4GiB_320MiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        result['helper_sha256'] = {}
        for p in (Path(__file__), PROTOCOL, Path(H.__file__), Path(Q.__file__), Path(Q.Q.__file__), Path(X.__file__), Path(M.__file__), Path(R.__file__), Path(R.C.__file__), Path(G.__file__), Path(H.L.__file__)):
            M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        records = {}; seen = {}; files = {}
        for name, expected in RECORDS.items():
            p = M.DOC/name; M.committed(p); assert sha(p) == expected; records[name] = json.loads(p.read_text(encoding='utf-8'))
            for path, expected in records[name]['helper_sha256'].items():
                if path in seen: assert seen[path] == expected; continue
                M.committed(Path(path)); assert sha(path) == expected; seen[path] = expected
            for a in records[name].get('output_inventory', []):
                if a['path'] in files: assert files[a['path']] == a['sha256']; continue
                assert sha(a['path']) == a['sha256'] and Path(a['path']).stat().st_size == a['bytes']; files[a['path']] = a['sha256']
        prior = records['meth418_switch_function_capture_result.json']; baseline = records['meth420_switch_function_gradient_result.failure.json']; parent = records['meth431_switch_additive_pilot_result.json']; oracle = records['meth436_switch_oracle_input_result.json']
        assert all(oracle['apparatus_gates'].values()) and not oracle['positive_diagnostic_routes']
        first = records['meth437_switch_shared_readout_result.failure.json']; assert first['updates'] == [] and first['failure_stage'] == 'bindings' and 'concurrent_job' in first['error']
        result['prior437_stop_retained_before_numeric'] = True
        for a in baseline['baselines']: assert sha(a['archive_path']) == a['archive_sha256']
        for p, expected in parent['preserved_original_binary_sha256'].items(): assert sha(p) == expected
        result['retained_record_sha256'] = RECORDS; result['preserved_original_binary_sha256'] = parent['preserved_original_binary_sha256']
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons', []).append(p.pid); continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'; assert psutil.disk_usage(str(M.ROOT)).free >= 2 << 30
        mapped = {}; entries = {}; initial = {}
        for n, (name, expected) in R.C.U.EXPORT.items():
            p = M.DOC/name; M.committed(p); assert sha(p) == expected; e = json.loads(p.read_text(encoding='utf-8')); a = e['artifact']; assert a == parent['artifacts'][str(n)]
            assert sha(a['payload']) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']; R.B.read_manifest(a['manifest'], e['original_config'], e['tensors'], Path(a['payload']))
            mapped[n] = np.memmap(a['payload'], dtype='u1', mode='r'); entries[n] = e['tensors']; initial[n] = (Path(a['payload']).stat().st_size, Path(a['payload']).stat().st_mtime_ns)
        result['artifacts'] = parent['artifacts']; OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_affinity': [0], 'BLAS': threadpool_info(), 'GPU': False,
                'torch_CPU_DLL_sha256': sha(Path(torch.__file__).parent/'lib/torch_cpu.dll'), 'torch_C_extension_sha256': sha(Path(torch._C.__file__))}
            assert result['runtime']['torch_CPU_DLL_sha256'] == parent['runtime']['torch_CPU_DLL_sha256'] and result['runtime']['torch_C_extension_sha256'] == parent['runtime']['torch_C_extension_sha256']
            stage = 'paired_data'; data = X.load_pairs(prior, baseline, guard); assert data['keys'] == parent['data']['keys']; result['data'] = parent['data']
            for n in (256, 128): assert hashlib.sha256(data[f'p{n}'].tobytes()).hexdigest() == parent['data'][f'paired_probability{n}_sha256']
            captures = {a['label']: a for a in prior['captures']}; rows = {}
            for n in (128, 256): rows[n] = np.concatenate([np.frombuffer(Path(captures[f'teacher.n{n}.book{bi}.case{ci}']['capture_path']).read_bytes(), dtype=R.C.DTYPE, offset=32).copy() for bi in range(24) for ci in range(4)])
            assert np.array_equal(rows[128]['expert'], data['labels']); assert hashlib.sha256(rows[128]['input'].tobytes()).hexdigest() == oracle['oracle_input128_sha256']
            dev, val = data['dev'], data['val']; features = {'real': rows[128]['down'].copy(), 'adapter': rows[128]['input'].copy()}
            buffers = {arm: (np.mean(v[dev].astype(np.float64), axis=0).astype(np.float32), np.maximum(np.std(v[dev].astype(np.float64), axis=0), 1e-6).astype(np.float32)) for arm, v in features.items()}
            result['oracle_features'] = {'route': 'captured_source128_teacher_ID', 'feature_sha256': {a: hashlib.sha256(v.tobytes()).hexdigest() for a, v in features.items()}, 'development_only_population_stats_floor': 1e-6, 'shared_supervision_all1008': True}
            fn = R.C.tensor(mapped[256], entries[256], 'decoder.final_layer_norm.weight'); head = G.I8Operator(R.C.tensor(mapped[256], entries[256], 'lm_head.weight'), R.C.tensor(mapped[256], entries[256], 'lm_head.weight', 'scales'))
            stage = 'tiny_composed_contract'; rng = np.random.default_rng(437); tiny_head = G.I8Operator(rng.integers(-9, 10, (13, 7), dtype=np.int8), np.linspace(.04, .07, 13, dtype=np.float32))
            tiny_feature = rng.normal(size=7).astype(np.float32); tiny_buffers = (rng.normal(size=7).astype(np.float32)*.1, np.linspace(.7, 1.3, 7, dtype=np.float32)); target = R.probability(rng.normal(size=13))
            result['qualification'] = [H.qualify('tiny_all_five_fields', tiny_feature, np.linspace(.2, .8, 7, dtype=np.float32), np.linspace(-.7, .9, 5, dtype=np.float32), 4, tiny_buffers, np.linspace(.8, 1.2, 7, dtype=np.float32), tiny_head, H.parameters(7, 2, 437, False), target, True, OUT/'tiny_qualification.npz')]; guard()
            state = {arm: H.parameters(768, 32, 437) for arm in features}; assert all(state['real'][k].detach().numpy().tobytes() == state['adapter'][k].detach().numpy().tobytes() for k in ('C', 'D'))
            probe = np.zeros(32128, np.float64); probe[(int(np.argmax(data['logits'][0]))+16064)%32128] = 1
            result['real_numeric_probe_ONLY_NOT_training'] = {'target_onehot': int(np.argmax(probe)), 'same_rule430': True}
            for arm in features:
                stage = arm+'_real_initial_composed_contract'; result['qualification'].append(H.qualify(arm+'_initial_zero', features[arm][0], data['base'][0], data['scores'][0], int(data['old_ids'][0]), buffers[arm], fn, head, state[arm], probe, False, OUT/(arm+'_qualification.npz'), data['logits'][0])); guard()
            stage = 'all_initial_post_norm_head_inputs'; streamed = {}
            with torch.no_grad():
                for arm in features:
                    h = hashlib.sha256()
                    for i in range(1344):
                        z = H.forward(torch.from_numpy(features[arm][i]), torch.from_numpy(data['base'][i]), torch.from_numpy(data['scores'][i]), int(data['old_ids'][i]), buffers[arm], fn, None, state[arm])
                        for field in ('post', 'final', 'head_input'): R.C.exact(z[field].numpy(), rows[256][field][i])
                        h.update(z['head_input'].numpy().tobytes()); guard()
                    streamed[arm] = h.hexdigest()
            assert streamed['real'] == streamed['adapter']; result['zero_initialization'] = {'positions_per_arm': 1344, 'post_final_head_inputs_all_byte_exact418': True, 'head_input_stream_sha256': streamed, 'full_head_first_actual_two_exact420': True, 'all_other_full_head_identity_INHERITED420_same_operator_and_exact_inputs': True}
            result['geometry'] = {'shared_rank': 32, 'learned_coefficients_per_arm': 49152, 'fixed_coefficients_per_arm': 1536, 'stored_nominal_coefficient_bytes_per_arm': 202752,
                'learned_values_gradients_Adam_moments_plus_buffers_bytes_per_arm': 792576, 'no_private_ID_parameters': True, 'physical_DRAM_NOT_measured': True}
            result['training'] = {'passes': 3, 'batch': 32, 'last_batch': 16, 'updates_per_arm': 96, 'total_updates': 192, 'sample_evaluations': 6048, 'seed_both_arms': 437,
                'lr': .001, 'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': 0, 'gradient_clip': 1., 'loss': 'equal teacher128/256 mixture CE at original256 prefixes, no preservation penalty', 'final_state_only_no_validation_selection': True}
            for arm in features:
                stage = 'fixed_fit_'+arm
                for p in state[arm].values(): p.grad = None
                opt = X.optimizer(list(state[arm].values()), .001)
                for epoch in range(3):
                    order = np.random.default_rng(437+epoch).permutation(dev); epoch_loss = 0.
                    for begin in range(0, len(order), 32):
                        selected = order[begin:begin+32]; opt.zero_grad(set_to_none=True); total = 0.
                        for i in selected:
                            logits = H.forward(torch.from_numpy(features[arm][i]), torch.from_numpy(data['base'][i]), torch.from_numpy(data['scores'][i]), int(data['old_ids'][i]), buffers[arm], fn, head, state[arm])['logits']
                            loss = G.prediction_loss(logits, torch.from_numpy(.5*(data['p256'][i]+data['p128'][i])))/len(selected); assert bool(torch.isfinite(loss)); total += float(loss.detach()); loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(list(state[arm].values()), 1., error_if_nonfinite=True); opt.step()
                        assert all(bool(torch.isfinite(v).all()) for v in state[arm].values()) and all(bool(torch.isfinite(opt.state[p][k]).all()) for p in state[arm].values() for k in ('exp_avg', 'exp_avg_sq'))
                        updates.append({'arm': arm, 'epoch': epoch, 'positions': selected.tolist(), 'loss': total, 'gradient_norm_before_clip': float(norm)}); epoch_loss += total*len(selected); guard()
                    print(json.dumps({'arm': arm, 'completed_pass': epoch+1, 'updates': len(updates), 'mean_online_epoch_loss': epoch_loss/len(dev), 'seconds': time.monotonic()-start, 'peak_bytes': peak}), flush=True)
            assert len(updates) == 192 and sum(len(a['positions']) for a in updates) == 6048
            np.savez(OUT/'shared_readout_checkpoint.npz', **{arm+'_'+k: v.detach().numpy() for arm in state for k, v in state[arm].items()}, **{arm+'_'+k: buffers[arm][j] for arm in buffers for j, k in enumerate(('mean', 'std'))})
            old = X.losses(data['logits'][val], val, data); ob = X.summarize(old); assert ob == parent['no_added_validation']; result['no_added_validation'] = ob; losses = {}; masks = {}; books = data['books'][val]
            with torch.no_grad():
                for arm in features:
                    stage = 'final_validation_'+arm; output = np.empty((336, 32128), np.float32)
                    for j, i in enumerate(val):
                        output[j] = H.forward(torch.from_numpy(features[arm][i]), torch.from_numpy(data['base'][i]), torch.from_numpy(data['scores'][i]), int(data['old_ids'][i]), buffers[arm], fn, head, state[arm])['logits'].numpy(); guard()
                    np.save(OUT/(arm+'_validation_forced.npy'), output); loss = X.losses(output, val, data); del output
                    mask = (loss[:, 2]+.01 <= old[:, 2]) & (loss[:, 0] <= old[:, 0]+.02); hard = np.where(mask[:, None], loss, old); masks[arm] = mask; losses[arm] = hard
                    result[arm+'_forced'] = X.summarize(loss) | {'per_position_losses': loss.tolist()}; result[arm+'_oracle_validation'] = Q.Q.potential(hard, old, data['labels'][val], mask, books)
                stage = 'same_mask_real_primitive_permutation'; cache = X.ExpertCache(mapped[128], entries[128]); output = np.empty((336, 32128), np.float32)
                for j, i in enumerate(val):
                    wi, wo = cache.get((int(data['labels'][i])+1)%128); feature = wo.native(np.where((raw := wi.native(rows[128]['input'][i])) < 0, np.float32(0), raw))
                    output[j] = H.forward(torch.from_numpy(feature), torch.from_numpy(data['base'][i]), torch.from_numpy(data['scores'][i]), int(data['old_ids'][i]), buffers['real'], fn, head, state['real'])['logits'].numpy(); guard()
                np.save(OUT/'real_permuted_validation_forced.npy', output); loss = X.losses(output, val, data); del output; pv = X.summarize(np.where(masks['real'][:, None], loss, old)); result['same_mask_real_permutation'] = pv | {'per_position_forced_losses': loss.tolist()}
            rv, av = (result[a+'_oracle_validation'] for a in ('real', 'adapter')); result['removed_same_mask'] = ob
            result['potential_gates_NOT_deployable_capacity'] = {'mixture_gain_at_least1percent': (ob['equal_mixture_CE']-rv['equal_mixture_CE'])/ob['equal_mixture_CE'] >= .01,
                'teacher128_gain_at_least1percent': (ob['teacher128_CE']-rv['teacher128_CE'])/ob['teacher128_CE'] >= .01,
                'teacher256_mean_delta_at_most0_02': rv['teacher256_CE']-ob['teacher256_CE'] <= .02, 'every_book_teacher256_delta_at_most0_05': max(rv['book_teacher256_delta'].values()) <= .05,
                'actual_added_positions_at_least34': rv['added_positions'] >= 34, 'useful_ids_at_least8': rv['useful_ids'] >= 8,
                'beats_matched_own_input_control_by0_01': av['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01,
                'same_mask_permutation_harm_at_least0_01': pv['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01, 'same_mask_removal_harm_at_least0_01': ob['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01}
            result['apparatus_gates'] = {'all_parent_archives_and_sources_fresh_exact': True, 'paired_probabilities_keys_original_baseline_exact431': True,
                'new_shared_chain_tiny_and_two_actual_qualifications_before_fit': True, 'all1344_each_arm_zero_post_norm_head_inputs_exact': True,
                'same_seed_budget_stats_dev_only_96_updates_each': True, 'oracle_rule_and_same_mask_primitive_permutation_fixed': True, 'source_functions_and_original_core_unchanged': True}
        for n, a in ((n, parent['artifacts'][str(n)]) for n in (128, 256)): assert initial[n] == (Path(a['payload']).stat().st_size, Path(a['payload']).stat().st_mtime_ns)
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['decision'] = 'shared_readout_oracle_potential_ONLY_NEW_task_aware_available_input_and_routing_controls' if all(result['potential_gates_NOT_deployable_capacity'].values()) else 'fixed_shared_rank32_readout_oracle_recipe_ineligible_close_before_rank_lr_pass_seed_sweeps'
        result['resource'] = {'seconds': time.monotonic()-start, 'peak_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': sum(a['bytes'] for a in result['output_inventory']), 'optimizer_updates': len(updates)}
        result['scope'] = 'Correct source128 native feature versus own source128 input at SAME teacher IDs, shared rank32 readouts/all1008dev/336consumed-val, unavailable source128 core and target-aware oracle masks. Source functions frozen, only shared C/D trained, no private-ID/input map/selector/gate fit. Matched control has same core information/parameter/update budget. Constant shared-readout nominal bytes versus n do not establish realDRAM/native rate. One final bank original PLUS added when ON; no combined C model/whole quality/SAMEartifact50/LUT/usefulRAM-n/another-family/~100B. Local approximate gradient only.426/431/435 failed recipes remain closed.'
        guard(); M.write(args.out, result); print(json.dumps({'apparatus': result['apparatus_gates'], 'potential': result['potential_gates_NOT_deployable_capacity'], 'real_oracle_CE': rv['equal_mixture_CE'], 'adapter_oracle_CE': av['equal_mixture_CE'], 'decision': result['decision'], 'resource': result['resource'], 'sha256': M.digest(args.out)}), flush=True)
    except BaseException as error:
        result.update(failure_stage=stage, error=repr(error), seconds=time.monotonic()-start, peak_bytes=peak, bytes_hashed=hashed)
        if OUT.exists(): result['partial_output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
