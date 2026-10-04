"""One prespecified shared nonlinear input feasibility fit, never capacity proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth435_switch_shared_input_math as H
R = H.R
PROTOCOL = M.DOC/'METH_435_SWITCH_SHARED_INPUT_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth435_switch_shared_input'
RECORDS = {
    'meth418_switch_function_capture_result.json': '4ebb37ed40d788eb85168d028b3e0b538cd7d65e009293cd32d5fb0eb117d829',
    'meth420_switch_function_gradient_result.failure.json': '17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427',
    'meth434_switch_frozen_selection_result.json': 'b237db3bc2bddd15a007415e883bb6d8e2e354802ab6c69495367c0fea7b3cd3'}


def load_inputs(prior, baseline, guard):
    cap = {a['label']: a for a in prior['captures']}; old = {a['label']: a for a in baseline['baselines']}
    inputs = {256: [], 128: []}; keys = []; books = []; routes = {256: [], 128: []}
    for bi in range(24):
        for ci in range(4):
            pair = [cap[f'teacher.n{n}.book{bi}.case{ci}'] for n in (256, 128)]
            assert pair[0]['pairing_sha256'] == pair[1]['pairing_sha256'] and pair[0]['decoder_ids'] == pair[1]['decoder_ids'] and pair[0]['source_ids'] == pair[1]['source_ids']
            for n, item in zip((256, 128), pair):
                assert item['prospective_split'] == ('development' if bi < 18 else 'validation')
                a = old[item['label']]; assert not a['qualification_only_not_paired_training'] and a['source_capture_sha256'] == item['capture_sha256']
                data = Path(item['capture_path']).read_bytes()
                assert data[:8] == b'SWFUN001' and struct.unpack_from('<6I', data, 8) == (768, 3072, 32128, n, 11, 1)
                assert len(data) == 32+14*R.C.DTYPE.itemsize
                rows = np.frombuffer(data, dtype=R.C.DTYPE, offset=32)
                assert np.array_equal(rows['position'], np.arange(14))
                with np.load(a['archive_path'], allow_pickle=False) as z:
                    R.C.exact(z['input'], rows['input']); assert np.array_equal(z['source_ids'], item['source_ids']) and np.array_equal(z['decoder_ids'], item['decoder_ids'])
                inputs[n].append(rows['input'].copy()); routes[n].extend(rows['expert'].tolist())
            keys.extend([f'book{bi}.case{ci}.position{j}:{pair[0]["pairing_sha256"]}' for j in range(14)]); books.extend([bi]*14); guard()
    x, y = (np.concatenate(inputs[n]) for n in (256, 128)); books = np.asarray(books)
    assert x.shape == y.shape == (1344, 768) and np.isfinite(x).all() and np.isfinite(y).all()
    return x, y, books, keys, routes


def metrics(prediction, target, baseline, books):
    p, y, m = [a.astype(np.float64) for a in (prediction, target, baseline)]
    e = np.sum((p-y)**2, axis=1); old = np.sum((m-y)**2, axis=1); yn = np.linalg.norm(y, axis=1); pn = np.linalg.norm(p, axis=1)
    rel = np.sqrt(e)/np.maximum(yn, 1e-12); cos = np.sum(p*y, axis=1)/np.maximum(pn*yn, 1e-12)
    return {'total_squared_error': float(np.sum(e)), 'mean_baseline_squared_error': float(np.sum(old)),
        'error_fraction_of_mean_baseline': float(np.sum(e)/np.sum(old)), 'median_relative_L2': float(np.median(rel)), 'p95_relative_L2': float(np.quantile(rel, .95)),
        'median_cosine': float(np.median(cos)), 'median_prediction_norm': float(np.median(pn)), 'median_target_norm': float(np.median(yn)),
        'row_squared_error': e.tolist(), 'row_baseline_squared_error': old.tolist(), 'row_relative_L2': rel.tolist(), 'row_cosine': cos.tolist(),
        'per_book': {str(b): {'squared_error': float(np.sum(e[books == b])), 'mean_baseline_squared_error': float(np.sum(old[books == b])),
            'median_relative_L2': float(np.median(rel[books == b])), 'p95_relative_L2': float(np.quantile(rel[books == b], .95))} for b in np.unique(books)}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; updates = []
    result = {'experiment': 'METH-435-shared-nonlinear-final-bank-input-transport', 'updates': updates}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        assert peak <= 2 << 30 and time.monotonic()-start <= 720 and size <= 16 << 20, 'fit_720sec_2GiB_16MiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        result['helper_sha256'] = {}
        for p in (Path(__file__), PROTOCOL, Path(H.__file__), Path(M.__file__), Path(R.__file__), Path(R.C.__file__), Path(H.G.__file__)):
            M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        records = {}; seen = {}
        for name, expected in RECORDS.items():
            p = M.DOC/name; M.committed(p); assert sha(p) == expected; records[name] = json.loads(p.read_text(encoding='utf-8'))
            for path, expected in records[name]['helper_sha256'].items():
                if path in seen: assert seen[path] == expected; continue
                M.committed(Path(path)); assert sha(path) == expected; seen[path] = expected
        prior = records['meth418_switch_function_capture_result.json']; baseline = records['meth420_switch_function_gradient_result.failure.json']; parent = records['meth434_switch_frozen_selection_result.json']
        assert all(prior['gates'].values()) and len(baseline['baselines']) == 384 and all(parent['apparatus_gates'].values()) and not parent['all_potential_gates_positive_routes']
        for a in prior['output_inventory']: assert sha(a['path']) == a['sha256'] and Path(a['path']).stat().st_size == a['bytes']
        for a in baseline['baselines']: assert sha(a['archive_path']) == a['archive_sha256']
        for p, expected in parent['preserved_original_binary_sha256'].items(): assert sha(p) == expected
        result['retained_record_sha256'] = RECORDS; result['preserved_original_binary_sha256'] = parent['preserved_original_binary_sha256']
        result['source_payload_identity_inherited_NOT_reread'] = parent['artifacts']
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons', []).append(p.pid); continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        assert psutil.disk_usage(str(M.ROOT)).free >= 2 << 30; OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_affinity': [0], 'BLAS': threadpool_info(), 'GPU': False}
            stage = 'paired_inputs'; x, y, books, keys, routes = load_inputs(prior, baseline, guard)
            assert keys == parent['data']['keys']
            dev = np.flatnonzero(books < 18); val = np.flatnonzero(books >= 18); assert len(dev) == 1008 and len(val) == 336
            buffers = tuple(a for v in (x[dev], y[dev]) for a in (np.mean(v.astype(np.float64), axis=0).astype(np.float32), np.maximum(np.std(v.astype(np.float64), axis=0), 1e-6).astype(np.float32)))
            assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in buffers) and all(np.all(buffers[i] > 0) for i in (1, 3))
            result['data'] = {'keys': keys, 'books': books.tolist(), 'routes': routes, 'development_positions': dev.tolist(), 'validation_positions': val.tolist(),
                'input256_sha256': hashlib.sha256(x.tobytes()).hexdigest(), 'input128_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                'development_statistics_only': True, 'population_std_floor': 1e-6, 'natural_used_for_fit_or_validation': False}
            result['qualification'] = []; stage = 'tiny_composed_derivative'; rng = np.random.default_rng(435)
            tx, ty = (rng.normal(size=7).astype(np.float32) for _ in range(2)); tb = (rng.normal(size=7).astype(np.float32)*.1, np.linspace(.7, 1.3, 7, dtype=np.float32), rng.normal(size=7).astype(np.float32)*.1, np.linspace(.8, 1.4, 7, dtype=np.float32))
            result['qualification'].append(H.qualify('tiny_all_fields', tx, ty, tb, H.parameters(7, 5, 435, False), True, OUT/'tiny_qualification.npz')); guard()
            stage = 'real_mean_initialization_derivative'; params = H.parameters(768, 128, 435)
            result['qualification'].append(H.qualify('real_initial_mean', x[dev[0]], y[dev[0]], buffers, params, False, OUT/'real_qualification.npz')); guard()
            result['geometry'] = {'input_dimension': 768, 'hidden': 128, 'output_dimension': 768, 'learned_coefficients': 197504, 'fixed_coefficients': 3072,
                'stored_bytes': 802304, 'learned_values_gradients_two_Adam_moments_bytes': 3160064, 'including_fixed_buffers_bytes': 3172352,
                'nominal_full_coefficient_access_bytes': 802304, 'actual_DRAM_NOT_measured': True, 'same_mapper_for_all_experts': True}
            assert sum(v.numel() for v in params.values()) == 197504
            for p in params.values(): p.grad = None
            optimizer = torch.optim.Adam(list(params.values()), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
            result['training'] = {'seed': 435, 'passes': 16, 'batch': 32, 'last_batch': 16, 'updates_per_pass': 32, 'total_updates': 512, 'sample_evaluations': 16128,
                'loss': 'mean coordinate squared error standardized by fixed development output std; F64 loss/native-valued F32 mapper', 'lr': .001, 'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': 0, 'gradient_clip': 1., 'checkpoint_selection': 'final only, no validation choice'}
            stage = 'fixed_512_updates'
            for epoch in range(16):
                order = np.random.default_rng(435+epoch).permutation(dev)
                for begin in range(0, len(order), 32):
                    selected = order[begin:begin+32]; optimizer.zero_grad(set_to_none=True); total = 0.
                    for i in selected:
                        prediction = H.forward(torch.from_numpy(x[i]), buffers, params)['prediction']; objective = H.loss(prediction, torch.from_numpy(y[i]), buffers[3])/len(selected)
                        assert bool(torch.isfinite(objective)); total += float(objective.detach()); objective.backward()
                    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True); optimizer.step()
                    assert all(bool(torch.isfinite(p).all()) for p in params.values())
                    updates.append({'update': len(updates)+1, 'pass': epoch, 'positions': selected.tolist(), 'loss': total, 'unclipped_gradient_norm': float(norm)}); guard()
                print(json.dumps({'completed_pass': epoch+1, 'updates': len(updates), 'seconds': time.monotonic()-start, 'peak_bytes': peak}), flush=True)
            assert len(updates) == 512 and sum(len(a['positions']) for a in updates) == 16128
            np.savez(OUT/'mapper_checkpoint.npz', **{k: v.detach().numpy() for k, v in params.items()}, mx=buffers[0], sx=buffers[1], my=buffers[2], sy=buffers[3])
            stage = 'final_only_predictions'; prediction = np.empty(y.shape, np.float32)
            with torch.no_grad():
                for i in range(len(x)):
                    prediction[i] = H.forward(torch.from_numpy(x[i]), buffers, params)['prediction'].numpy(); guard()
            continuous = np.stack([H.smooth(x[i], buffers, params) for i in val]); guard()
            assert np.isfinite(prediction).all() and np.isfinite(continuous).all()
            np.save(OUT/'development_predictions.npy', prediction[dev]); np.savez(OUT/'validation_predictions.npz', native=prediction[val], smooth_F64=continuous)
            mean = np.broadcast_to(buffers[2], y.shape)
            result['development'] = metrics(prediction[dev], y[dev], mean[dev], books[dev]); result['validation'] = metrics(prediction[val], y[val], mean[val], books[val])
            numeric = R.relative(prediction[val], continuous); result['native_global_smooth_prediction_relative_L2'] = numeric; v = result['validation']
            result['input_eligibility_gates_NOT_capacity'] = {'squared_error_le_half_development_mean_baseline': v['error_fraction_of_mean_baseline'] <= .5,
                'median_relative_L2_le0_25': v['median_relative_L2'] <= .25, 'p95_relative_L2_le0_50': v['p95_relative_L2'] <= .50,
                'native_global_F64_relative_le1e_minus5_and_finite': numeric <= 1e-5}
            result['apparatus_gates'] = {'capture_and_forward_archives_fresh_exact': True, 'paired_inputs_and_dev_only_normalization': True,
                'tiny_all_fields_and_real_initial_mean_qualified_before_fit': True, 'fixed_512_updates_final_only': True, 'no_other_parameters_or_source_functions_changed': True}
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['decision'] = 'input_transport_eligible_ONLY_for_NEW_coupled_function_prediction_controls' if all(result['input_eligibility_gates_NOT_capacity'].values()) else 'fixed_shared_nonlinear_input_mapper_ineligible_close_before_width_depth_optimizer_sweeps'
        result['resource'] = {'seconds': time.monotonic()-start, 'peak_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': sum(a['bytes'] for a in result['output_inventory']), 'optimizer_updates': len(updates)}
        result['scope'] = 'One final decoder bank11 shared native-valued F32 mapper, consumed405 paired1008dev/336val, not fresh generalization. Fixed normalization only from development, local approximate derivative, no derivative through rounding. Input geometry only, no token loss/selector/per-ID readout fit/oracle intervention. Parent434 source hashes inherited, no full source payload reread. No native C map/coupled functions/whole quality/rate/LUT/real DRAM/another family/~100B/capacity or useful-n claim. Fixed mapper cost does not establish end-to-end efficiency.431/426 checkpoint failures remain closed; all9 capacity and whole quality/SAMEartifact>=50 demands still required.'
        guard(); M.write(args.out, result); print(json.dumps({'input_eligibility_gates': result['input_eligibility_gates_NOT_capacity'], 'validation': {k: v for k, v in result['validation'].items() if not k.startswith('row_') and k != 'per_book'}, 'decision': result['decision'], 'resource': result['resource'], 'sha256': M.digest(args.out)}), flush=True)
    except BaseException as error:
        result.update(failure_stage=stage, error=repr(error), seconds=time.monotonic()-start, peak_bytes=peak, bytes_hashed=hashed)
        if OUT.exists(): result['partial_output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
