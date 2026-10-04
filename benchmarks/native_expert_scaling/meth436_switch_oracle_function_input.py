"""Captured source128 WI input substituted into frozen additive output readouts."""
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
import meth434_switch_frozen_selection_diagnostic as Q
X, M, R, G = Q.X, Q.M, Q.R, Q.G
PROTOCOL = M.DOC/'METH_436_SWITCH_ORACLE_INPUT_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth436_switch_oracle_input'
RECORDS = Q.RECORDS | {
    'meth434_switch_frozen_selection_result.json': 'b237db3bc2bddd15a007415e883bb6d8e2e354802ab6c69495367c0fea7b3cd3',
    'meth435_switch_shared_input_result.json': '967676c9c6697bc75e5bd49391f1a6af2f745461afed75b2ce21ec3580bf0f51'}


def oracle(x128, base, scores, chosen, factors, wi, wo, head, finalnorm):
    tx = torch.from_numpy(x128); raw = G.NativeI8.apply(tx, wi); up = G.NativeReLU.apply(raw); feature = G.NativeI8.apply(up, wo)
    dy = G.NativeFloat.apply(feature, factors['D']); cy = G.NativeFloat.apply(dy, factors['C'])
    p = G.NativeProbability.apply(torch.from_numpy(scores), chosen); post = torch.from_numpy(base)+p*cy
    final = G.NativeRMS.apply(post, finalnorm); hi = final*np.float32(1/np.sqrt(768)); logits = G.NativeI8.apply(hi, head)
    return dict(raw=raw.numpy(), up=up.numpy(), feature=feature.numpy(), Dy=dy.numpy(), Cy=cy.numpy(), probability=p.numpy(), post=post.numpy(), final=final.numpy(), head_input=hi.numpy(), logits=logits.numpy())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; result = {'experiment': 'METH-436-frozen-readout-perfect-added-function-input-oracle', 'routes': {}}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        assert peak <= 4 << 30 and time.monotonic()-start <= 180 and size <= 192 << 20, 'oracle_180sec_4GiB_192MiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        result['helper_sha256'] = {}
        for p in (Path(__file__), PROTOCOL, Path(Q.__file__), Path(X.__file__), Path(M.__file__), Path(R.__file__), Path(R.C.__file__), Path(G.__file__), Path(X.H.__file__), Path(X.H.P.__file__)):
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
        prior = records['meth418_switch_function_capture_result.json']; baseline = records['meth420_switch_function_gradient_result.failure.json']; parent = records['meth431_switch_additive_pilot_result.json']; diagnostic = records['meth434_switch_frozen_selection_result.json']; fit = records['meth435_switch_shared_input_result.json']
        assert all(diagnostic['apparatus_gates'].values()) and not diagnostic['all_potential_gates_positive_routes'] and len(fit['updates']) == 512 and sum(fit['input_eligibility_gates_NOT_capacity'].values()) == 1
        for a in baseline['baselines']: assert sha(a['archive_path']) == a['archive_sha256']
        for p, expected in parent['preserved_original_binary_sha256'].items(): assert sha(p) == expected
        result['retained_record_sha256'] = RECORDS; result['preserved_original_binary_sha256'] = parent['preserved_original_binary_sha256']
        checkpoint = next(a for a in parent['output_inventory'] if Path(a['path']).name == 'pilot_checkpoint.npz'); result['frozen_checkpoint'] = checkpoint
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
        with threadpool_limits(limits=1), torch.no_grad():
            result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_affinity': [0], 'BLAS': threadpool_info(), 'GPU': False}
            stage = 'paired_replay'; data = X.load_pairs(prior, baseline, guard); assert data['keys'] == parent['data']['keys']; result['data'] = parent['data']
            for n in (256, 128): assert hashlib.sha256(data[f'p{n}'].tobytes()).hexdigest() == parent['data'][f'paired_probability{n}_sha256']
            with np.load(checkpoint['path'], allow_pickle=False) as z:
                factors = [{k: torch.from_numpy(z['real_'+k][i].copy()) for k in ('C', 'D')} for i in range(128)]
                identities, _ = X.predict_affine(*(torch.from_numpy(z[k].copy()) for k in ('classifier_weight', 'classifier_bias')), data['input'], False)
            assert np.array_equal(identities, parent['classifier']['actual_ids']) and all(not v.requires_grad for f in factors for v in f.values())
            captures = {a['label']: a for a in prior['captures']}; rows128 = np.concatenate([np.frombuffer(Path(captures[f'teacher.n128.book{bi}.case{ci}']['capture_path']).read_bytes(), dtype=R.C.DTYPE, offset=32).copy() for bi in range(24) for ci in range(4)])
            assert np.array_equal(rows128['expert'], data['labels']); result['oracle_input128_sha256'] = hashlib.sha256(rows128['input'].tobytes()).hexdigest(); assert result['oracle_input128_sha256'] == fit['data']['input128_sha256']
            fn = R.C.tensor(mapped[256], entries[256], 'decoder.final_layer_norm.weight'); head = G.I8Operator(R.C.tensor(mapped[256], entries[256], 'lm_head.weight'), R.C.tensor(mapped[256], entries[256], 'lm_head.weight', 'scales'))
            cache = X.ExpertCache(mapped[128], entries[128]); val = data['val']; books = data['books'][val]; old = X.losses(data['logits'][val], val, data); assert X.summarize(old) == parent['no_added_validation']; exact_fields = 0; source_features = 0
            for route, selected in (('classifier', identities), ('teacher', data['labels'])):
                losses = []; mask = None; record = {'actual_ids': selected.tolist(), 'source_input_and_teacher_route_oracle_NOT_deployable': True}
                for permutation in (False, True):
                    stage = f'{route}_oracle_input_permuted{permutation}'; output = np.empty((336, 32128), np.float32)
                    for j, i in enumerate(val):
                        identity = int(selected[i]); wi, wo = cache.get((identity+1)%128 if permutation else identity)
                        actual = oracle(rows128['input'][i], data['base'][i], data['scores'][i], int(data['old_ids'][i]), factors[identity], wi, wo, head, fn)
                        if not permutation:
                            if route == 'teacher':
                                for field, captured in (('raw', 'up_raw'), ('up', 'up'), ('feature', 'down')): R.C.exact(actual[field], rows128[captured][i]); source_features += 1
                            literal = {}; literal['raw'] = wi.native(rows128['input'][i]); literal['up'] = np.where(literal['raw'] < 0, np.float32(0), literal['raw']); literal['feature'] = wo.native(literal['up'])
                            for field, parameter, incoming in (('Dy', 'D', 'feature'), ('Cy', 'C', 'Dy')): literal[field] = (factors[identity][parameter].numpy().astype(np.float64)@literal[incoming].astype(np.float64)).astype(np.float32)
                            s = data['scores'][i]; literal['probability'] = np.asarray(np.float32(1/np.sum(np.exp((s-s[int(data['old_ids'][i])]).astype(np.float64)).astype(np.float32).astype(np.float64))))
                            literal['post'] = data['base'][i]+literal['probability']*literal['Cy']; literal['final'] = R.C.norm(literal['post'], fn); literal['head_input'] = literal['final']*np.float32(1/np.sqrt(768)); literal['logits'] = head.native(literal['head_input'])
                            for field in actual: R.C.exact(actual[field], literal[field]); exact_fields += 1
                        output[j] = actual['logits']; guard()
                    np.save(OUT/f'{route}.oracle_input.permuted{permutation}.npy', output)
                    loss = X.losses(output, val, data); del output
                    if not permutation:
                        mask = (loss[:, 2]+.01 <= old[:, 2]) & (loss[:, 0] <= old[:, 0]+.02); hard = np.where(mask[:, None], loss, old)
                        record['forced'] = X.summarize(loss) | {'per_position_losses': loss.tolist()}; record['oracle_gate'] = Q.potential(hard, old, selected[val], mask, books)
                    else: record['same_mask_permutation'] = X.summarize(np.where(mask[:, None], loss, old)) | {'per_position_forced_losses': loss.tolist()}
                rv = record['oracle_gate']; ob = parent['no_added_validation']; pv = record['same_mask_permutation']
                record['diagnostic_gates_NO_adapter_capacity_claim'] = {'mixture_gain_at_least1percent': (ob['equal_mixture_CE']-rv['equal_mixture_CE'])/ob['equal_mixture_CE'] >= .01,
                    'teacher128_gain_at_least1percent': (ob['teacher128_CE']-rv['teacher128_CE'])/ob['teacher128_CE'] >= .01,
                    'teacher256_mean_delta_at_most0_02': rv['teacher256_CE']-ob['teacher256_CE'] <= .02,
                    'every_book_teacher256_delta_at_most0_05': max(rv['book_teacher256_delta'].values()) <= .05,
                    'actual_added_positions_at_least34': rv['added_positions'] >= 34, 'useful_ids_at_least8': rv['useful_ids'] >= 8,
                    'same_mask_permutation_harm_at_least0_01': pv['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01,
                    'same_mask_removal_harm_at_least0_01': ob['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01}
                record['removed_same_mask'] = ob; record['prior434_unmodified_input'] = diagnostic['routes'][route]['real_oracle_gate']; result['routes'][route] = record
            assert source_features == 1008 and exact_fields == 6720
            result['apparatus_gates'] = {'all_parent_archives_and_sources_fresh_exact': True, 'paired_classifier_probabilities_original_baseline_exact431': True,
                'correct_teacher_function_full_up_down_byte_exact418_all336': True, 'all10_substitution_nodes_numpy_native_byte_exact_all672': True,
                'frozen_C_D_no_optimizer_or_other_parameters_changed': True, 'oracle_rule_and_same_mask_permutation_fixed': True}
            result['replay_counts'] = {'teacher_positions': 336, 'captured_up_down_fields': source_features, 'literal_native_fields': exact_fields}
        assert sha(checkpoint['path']) == checkpoint['sha256']
        for n, a in ((n, parent['artifacts'][str(n)]) for n in (128, 256)): assert initial[n] == (Path(a['payload']).stat().st_size, Path(a['payload']).stat().st_mtime_ns)
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*.npy'))]
        result['positive_diagnostic_routes'] = [k for k, v in result['routes'].items() if all(v['diagnostic_gates_NO_adapter_capacity_claim'].values())]
        result['decision'] = 'perfect_function_input_shows_potential_ONLY_NEW_joint_readout_controls_not_selector_rescue' if result['positive_diagnostic_routes'] else 'perfect_function_input_alone_insufficient_with_frozen_C_D_do_not_refine_input_mapper_as_only_change'
        result['resource'] = {'seconds': time.monotonic()-start, 'peak_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': sum(a['bytes'] for a in result['output_inventory']), 'optimizer_updates': 0}
        result['scope'] = 'Consumed336 validation. Oracle replaces added WI entry AFTER private input correction with source128 captured input, bypassing A/B; frozen431 C/D readout and original256 base/probability/finalnorm/head retained. Actual classifier and teacher IDs, fixed target-aware masks diagnostic only, require unavailable source128 core. No matched-parameter adapter claim because A/B bypassed; no complete9-gate capacity claim. Does not bound any refitted/readout/general nonlinear learner. No new fit/C engine/whole quality/rate/LUT/physicalDRAM/useful-n/another family/~100B claim.435 map and431/426 checkpoint recipes stay closed.'
        guard(); M.write(args.out, result); print(json.dumps({'apparatus': result['apparatus_gates'], 'routes': {k: {'forced': v['forced']['equal_mixture_CE'], 'oracle': v['oracle_gate']['equal_mixture_CE'], 'gates': v['diagnostic_gates_NO_adapter_capacity_claim']} for k, v in result['routes'].items()}, 'decision': result['decision'], 'resource': result['resource'], 'sha256': M.digest(args.out)}), flush=True)
    except BaseException as error:
        result.update(failure_stage=stage, error=repr(error), seconds=time.monotonic()-start, peak_bytes=peak, bytes_hashed=hashed)
        if OUT.exists(): result['partial_output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
