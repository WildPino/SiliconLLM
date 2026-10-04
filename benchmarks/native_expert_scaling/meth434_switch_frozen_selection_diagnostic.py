"""Frozen additive checkpoint: functions versus selection, never oracle deployment."""
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
import meth431_switch_additive_pilot as X
M, R, G = X.M, X.R, X.G
PROTOCOL = M.DOC/'METH_434_SWITCH_FROZEN_SELECTION_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth434_switch_frozen_selection'
RECORDS = {
    'meth418_switch_function_capture_result.json': '4ebb37ed40d788eb85168d028b3e0b538cd7d65e009293cd32d5fb0eb117d829',
    'meth420_switch_function_gradient_result.failure.json': '17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427',
    'meth431_switch_additive_pilot_result.json': '648f37d9b0708488d1491b6af91776efb0bf062084f6cffb317e4af580ccb9e4',
    'meth433_switch_output_bound_result.json': '5296d450562c0cca1422ccc9d7739545b34cea111dc3def7f8c124a2f6015601'}


def potential(values, baseline, identities, mask, books):
    ids = identities[mask]; per_id = []
    for identity in np.unique(ids):
        selected = mask & (identities == identity); gain = float(np.mean(baseline[selected, 2]-values[selected, 2]))
        per_id.append({'id': int(identity), 'positions': int(np.sum(selected)), 'mean_mixture_gain': gain,
                       'useful_ge2_and_gain_ge0_01': bool(np.sum(selected) >= 2 and gain >= .01)})
    return X.summarize(values) | {'added_positions': int(np.sum(mask)), 'distinct_ids': len(per_id), 'useful_ids': sum(p['useful_ge2_and_gain_ge0_01'] for p in per_id),
        'per_id': per_id, 'book_teacher256_delta': {str(b): float(np.mean(values[books == b, 0]-baseline[books == b, 0])) for b in range(18, 24)},
        'oracle_mask': mask.tolist(), 'per_position_losses': values.tolist()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; result = {'experiment': 'METH-434-frozen-checkpoint-function-versus-selection-diagnostic', 'routes': {}}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        assert peak <= 4 << 30 and time.monotonic()-start <= 180 and size <= 320 << 20, 'diagnostic_180sec_4GiB_320MiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        result['helper_sha256'] = {}
        for p in (Path(__file__), PROTOCOL, Path(X.__file__), Path(M.__file__), Path(R.__file__), Path(R.C.__file__), Path(G.__file__), Path(X.H.__file__), Path(X.H.P.__file__)):
            M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        records = {}; seen = {}
        for name, expected in RECORDS.items():
            p = M.DOC/name; M.committed(p); assert sha(p) == expected; records[name] = json.loads(p.read_text(encoding='utf-8'))
            for path, expected in records[name]['helper_sha256'].items():
                if path in seen: assert seen[path] == expected; continue
                M.committed(Path(path)); assert sha(path) == expected; seen[path] = expected
            for a in records[name].get('output_inventory', []): assert sha(a['path']) == a['sha256'] and Path(a['path']).stat().st_size == a['bytes']
        prior = records['meth418_switch_function_capture_result.json']; baseline = records['meth420_switch_function_gradient_result.failure.json']; parent = records['meth431_switch_additive_pilot_result.json']; bound = records['meth433_switch_output_bound_result.json']
        assert all(prior['gates'].values()) and len(baseline['baselines']) == 384 and len(parent['updates']) == 6240 and sum(parent['capacity_pilot_gates'].values()) == 2
        assert bound['diagnostic_decisions']['both_prediction_gains_attainable_under_output_preservation_bounds'] and all(bound['apparatus_gates'].values())
        for a in baseline['baselines']: assert sha(a['archive_path']) == a['archive_sha256']
        for p, expected in parent['preserved_original_binary_sha256'].items(): assert sha(p) == expected
        result['retained_record_sha256'] = RECORDS; result['preserved_original_binary_sha256'] = parent['preserved_original_binary_sha256']
        checkpoint = next(a for a in parent['output_inventory'] if Path(a['path']).name == 'pilot_checkpoint.npz')
        assert checkpoint['sha256'] == '876a1da6da1bd077960907347649a896f7d4d355336468266f3c3e4afc84da4a'; result['frozen_checkpoint'] = checkpoint
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons', []).append(p.pid); continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        mapped = {}; entries = {}; initial = {}
        for n, (name, expected) in R.C.U.EXPORT.items():
            stage = f'fresh_source{n}'; p = M.DOC/name; M.committed(p); assert sha(p) == expected
            e = json.loads(p.read_text(encoding='utf-8')); a = e['artifact']; assert a == parent['artifacts'][str(n)]
            assert sha(a['payload']) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']
            R.B.read_manifest(a['manifest'], e['original_config'], e['tensors'], Path(a['payload']))
            mapped[n] = np.memmap(a['payload'], dtype='u1', mode='r'); entries[n] = e['tensors']; initial[n] = (Path(a['payload']).stat().st_size, Path(a['payload']).stat().st_mtime_ns)
        result['artifacts'] = parent['artifacts']; assert psutil.disk_usage(str(M.ROOT)).free >= 2 << 30; OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_affinity': [0], 'BLAS': threadpool_info(), 'GPU': False}
            stage = 'paired_data'; data = X.load_pairs(prior, baseline, guard)
            assert data['keys'] == parent['data']['keys']
            for n in (256, 128): assert hashlib.sha256(data[f'p{n}'].tobytes()).hexdigest() == parent['data'][f'paired_probability{n}_sha256']
            result['data'] = parent['data']; state = {}; gates = {}
            with np.load(checkpoint['path'], allow_pickle=False) as z:
                assert all(z[k].dtype == np.float32 and np.isfinite(z[k]).all() for k in z.files)
                classifier = tuple(torch.from_numpy(z[k].copy()) for k in ('classifier_weight', 'classifier_bias'))
                for arm in ('real', 'adapter'):
                    state[arm] = [{k: torch.from_numpy(z[arm+'_'+k][i].copy()) for k in ('A', 'B', 'C', 'D')} for i in range(128)]
                    gate_parameters = tuple(torch.from_numpy(z[arm+'_gate_'+k].copy()) for k in ('weight', 'bias'))
                    gates[arm], scores = X.predict_affine(*gate_parameters, data['input'], True)
                    assert np.array_equal(gates[arm], parent[arm+'_validation']['gate_mask'])
                    assert scores[:, 0].tobytes() == np.asarray(parent[arm+'_validation']['gate_scores'], np.float32).tobytes()
            identities, _ = X.predict_affine(*classifier, data['input'], False); assert np.array_equal(identities, parent['classifier']['actual_ids'])
            assert all(not p.requires_grad for arm in state.values() for params in arm for p in params.values())
            norm = tuple(R.C.tensor(mapped[256], entries[256], k) for k in ('decoder.block.11.layer.2.layer_norm.weight', 'decoder.final_layer_norm.weight'))
            head = G.I8Operator(R.C.tensor(mapped[256], entries[256], 'lm_head.weight'), R.C.tensor(mapped[256], entries[256], 'lm_head.weight', 'scales'))
            cache = X.ExpertCache(mapped[128], entries[128]); val = data['val']; old = X.losses(data['logits'][val], val, data); ob = X.summarize(old)
            assert ob == parent['no_added_validation']; val_books = data['books'][val]
            for route, selected in (('classifier', identities), ('teacher', data['labels'])):
                record = {'actual_ids': selected.tolist(), 'teacher_route_qualification_oracle_only': route == 'teacher'}; masks = {}; values = {}
                for arm in ('real', 'adapter'):
                    stage = f'{route}_{arm}_forced'; forced = X.forced(arm, val, selected, data, norm, head, state[arm], cache, guard)
                    np.save(OUT/f'{route}.{arm}.forced.npy', forced); loss = X.losses(forced, val, data)
                    if route == 'classifier':
                        hard = np.where(gates[arm][val, None], forced, data['logits'][val]); parent_array = next(a for a in parent['output_inventory'] if Path(a['path']).name == f'validation.{arm}.hard.npy')
                        R.C.exact(hard, np.load(parent_array['path'], mmap_mode='r')); assert np.array_equal(X.losses(hard, val, data), parent[arm+'_validation']['per_position_losses'])
                        del hard
                    mask = (loss[:, 2]+.01 <= old[:, 2]) & (loss[:, 0] <= old[:, 0]+.02)
                    hard_losses = np.where(mask[:, None], loss, old); masks[arm] = mask; values[arm] = hard_losses
                    record[arm+'_forced'] = X.summarize(loss) | {'per_position_losses': loss.tolist()}
                    record[arm+'_oracle_gate'] = potential(hard_losses, old, selected[val], mask, val_books)
                    del forced; guard()
                stage = route+'_same_mask_identity_permutation'; permuted = X.forced('real', val, selected, data, norm, head, state['real'], cache, guard, permutation=True)
                np.save(OUT/f'{route}.real.permuted.npy', permuted); perm_loss = X.losses(permuted, val, data); del permuted
                mask = masks['real']; perm_hard = np.where(mask[:, None], perm_loss, old); rv = record['real_oracle_gate']; av = record['adapter_oracle_gate']; pv = X.summarize(perm_hard)
                record['permutation_same_oracle_mask'] = pv | {'per_position_losses': perm_hard.tolist()}
                record['removed_same_oracle_mask'] = ob
                record['potential_gates_NOT_deployable_capacity'] = {
                    'mixture_gain_at_least1percent': (ob['equal_mixture_CE']-rv['equal_mixture_CE'])/ob['equal_mixture_CE'] >= .01,
                    'teacher128_gain_at_least1percent': (ob['teacher128_CE']-rv['teacher128_CE'])/ob['teacher128_CE'] >= .01,
                    'teacher256_mean_delta_at_most0_02': rv['teacher256_CE']-ob['teacher256_CE'] <= .02,
                    'every_book_teacher256_delta_at_most0_05': max(rv['book_teacher256_delta'].values()) <= .05,
                    'actual_added_positions_at_least34': rv['added_positions'] >= 34,
                    'useful_ids_at_least8': rv['useful_ids'] >= 8,
                    'beats_matched_adapter_oracle_by0_01': av['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01,
                    'same_mask_permutation_harm_at_least0_01': pv['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01,
                    'same_mask_removal_harm_at_least0_01': ob['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01}
                result['routes'][route] = record; guard()
            result['no_added'] = ob; result['apparatus_gates'] = {'full_sources_and_archives_fresh_exact': True, 'frozen_checkpoint_no_updates_or_parameters_changed': True,
                'actual_classifier_gate_scores_masks_exact431': True, 'both_original_actual_hard_matrices_and_losses_byte_exact431': True,
                'same_primitive_forward_and_probability_amplitude': True, 'fixed_oracle_rules_not_fit_or_tuned': True, 'permutation_same_mask_ids_factors': True}
        assert sha(checkpoint['path']) == checkpoint['sha256']
        for n, (name, _) in R.C.U.EXPORT.items():
            a = parent['artifacts'][str(n)]; assert initial[n] == (Path(a['payload']).stat().st_size, Path(a['payload']).stat().st_mtime_ns)
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*.npy'))]
        positive = [route for route, record in result['routes'].items() if all(record['potential_gates_NOT_deployable_capacity'].values())]
        result['decision'] = 'frozen_function_potential_with_oracle_selection_only_requires_NEW_learnable_selection_protocol' if positive else 'frozen_functions_do_not_meet_all_potential_gates_even_with_prespecified_oracle_selection_close_checkpoint_before_selector_fit'
        result['all_potential_gates_positive_routes'] = positive
        result['resource'] = {'seconds': time.monotonic()-start, 'peak_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': sum(a['bytes'] for a in result['output_inventory']), 'optimizer_updates': 0}
        result['scope'] = 'Consumed paired336 validation at fixed original256 prefixes only. Checkpoint431 immutable, two routes (actual classifier and source128 teacher IDs) and fixed gain/protection oracle masks are diagnostic upper opportunities, not deployable selectors/generalization. All128 functions available, no per-ID sweep. Equal-parameter matched adapter at SAME routes and independently fixed-rule oracle masks; primitive permutation retains real IDs/factors/mask. Removal restores exact original post/logits. Original PLUS selected extra function if deployed, teacher-oracle routing cost unavailable/unqualified. No new fit/checkpoint/gate threshold/corpus/GPU/T4/C artifact/whole donor quality/rate/LUT/DRAM/another-family/~100B claim.431/426 final capacity failures remain closed.'
        guard(); M.write(args.out, result); print(json.dumps({'apparatus_gates': result['apparatus_gates'], 'route_potential_gates': {k: v['potential_gates_NOT_deployable_capacity'] for k, v in result['routes'].items()}, 'decision': result['decision'], 'resource': result['resource'], 'sha256': M.digest(args.out)}), flush=True)
    except BaseException as error:
        result.update(failure_stage=stage, error=repr(error), seconds=time.monotonic()-start, peak_bytes=peak, bytes_hashed=hashed)
        if OUT.exists(): result['partial_output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': M.digest(p)} for p in sorted(OUT.glob('*.npy'))]
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
