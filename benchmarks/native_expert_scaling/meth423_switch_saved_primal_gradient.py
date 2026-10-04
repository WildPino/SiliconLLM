"""Independent local-gradient diagnostic at the retained source128 failure."""
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
import meth422_switch_function_gradient_contract as R
import meth422_switch_function_autograd as G

PROTOCOL = M.DOC/'METH_423_SWITCH_SAVED_PRIMAL_GRADIENT_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth423_switch_saved_primal_gradient'
FAILURE = M.DOC/'meth422_switch_function_gradient_result.failure.json'
FAILURE_SHA = 'a35e8bf253d740fdf6335336ffbdc1619ef72659b8ff0524e9c9959c8ba465a0'


def loss_gradient(z, target):
    k = int(np.argmax(z)); shifted = z-z[k]; exp = np.exp(shifted); exp[k] = 0
    gradient = exp/(1+np.sum(exp))-target
    gradient[k] = 0; gradient[k] = -np.sum(gradient)
    return gradient


def local_gradient(row, parameters, ff_final, wi, wo, head, logits, target, mask):
    # Independent NumPy reconstruction of prescribed local saved-primal rules.
    d = 768; hgrad = (head.T.dot(loss_gradient(logits, target))).astype(np.float32)
    final_grad = hgrad*np.float32(1/np.sqrt(d))
    x = row['post'].astype(np.float64); g = final_grad.astype(np.float64)*ff_final
    r = 1/np.sqrt(np.sum(x*x)/d+1e-6)
    post_grad = (g*r-x*r**3*(np.dot(g, x)/d)).astype(np.float32)
    dy = post_grad*row['probability']
    Bx = parameters['B'].astype(np.float64).dot(row['input'].astype(np.float64)).astype(np.float32)
    Dy = parameters['D'].astype(np.float64).dot(row['down'].astype(np.float64)).astype(np.float32)
    grad_C = np.outer(dy.astype(np.float64), Dy.astype(np.float64)).astype(np.float32)
    up_grad = wo.T.dot(dy.astype(np.float64)).astype(np.float32)
    relu_grad = up_grad*mask.astype(np.float32)
    input_grad = wi.T.dot(relu_grad.astype(np.float64)).astype(np.float32)
    grad_A = np.outer(input_grad.astype(np.float64), Bx.astype(np.float64)).astype(np.float32)
    return {'A': grad_A, 'C': grad_C}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; result = {'experiment': 'METH-423-source128-local-saved-primal-gradient-diagnostic'}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        assert peak <= 2 << 30 and time.monotonic()-start <= 120, 'local_gradient_2min_2GiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        helpers = [Path(__file__), PROTOCOL, FAILURE, Path(R.__file__), Path(G.__file__), Path(R.C.__file__), Path(M.__file__), Path(R.B.__file__)]
        result['helper_sha256'] = {}
        for p in helpers: M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        assert sha(FAILURE) == FAILURE_SHA; failure = json.loads(FAILURE.read_text(encoding='utf-8'))
        assert 'real_STE_gradient' in failure['error'] and len(failure['real_gradient_controls']) == 1 and failure['real_gradient_controls'][0]['n'] == 256
        assert sha(failure['real_gradient_controls'][0]['archive_path']) == failure['real_gradient_controls'][0]['archive_sha256']
        assert sha(R.PRIOR) == R.PRIOR_SHA; prior = json.loads(R.PRIOR.read_text(encoding='utf-8')); assert all(prior['gates'].values())
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_model_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_affinity': [0], 'GPU': False}
        exports = {}; initial = {}; captured = {}
        for n, (name, expected) in R.C.U.EXPORT.items():
            stage = f'bindings{n}'; p = M.DOC/name; M.committed(p); assert sha(p) == expected
            e = json.loads(p.read_text(encoding='utf-8')); assert all(e['gates'].values()); a = e['artifact']; exports[n] = e
            payload = Path(a['payload']); initial[n] = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert a == prior['artifacts'][str(n)] and sha(payload) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']
            R.B.read_manifest(a['manifest'], e['original_config'], e['tensors'], payload)
            item = next(v for v in prior['captures'] if v['label'] == f'teacher.n{n}.book0.case0')
            for kind in ('capture', 'trace', 'output'): assert sha(item[kind+'_path']) == item[kind+'_sha256']
            data = Path(item['capture_path']).read_bytes(); assert data[:8] == b'SWFUN001' and len(data) == 32+14*R.C.DTYPE.itemsize
            row = np.frombuffer(data, dtype=R.C.DTYPE, offset=32)[0].copy()
            out = Path(item['output_path']).read_bytes(); R.C.R.R.route_bytes(out, n, 29, 14)
            logits = np.frombuffer(out, dtype='<f4', count=14*32128, offset=36+(14*29*768+14*14*768)*4).reshape(14, 32128)[0].copy()
            trace = Path(item['trace_path']).read_bytes(); dt = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (n,))])
            assert trace[:8] == b'SWRTA001' and struct.unpack_from('<2I', trace, 8) == (n, 768)
            scores = np.frombuffer(trace, dtype=dt, offset=16)[179]['scores'].copy(); assert int(np.argmax(scores)) == int(row['expert'])
            captured[n] = {'row': row, 'scores': scores, 'logits': logits, 'pair': item['pairing_sha256']}
        assert captured[128]['pair'] == captured[256]['pair']
        OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime']['BLAS'] = threadpool_info(); assert all(v['num_threads'] == 1 for v in threadpool_info())
            stage = 'local_gradient'; n = 128; item = captured[n]; row = item['row']; expert = int(row['expert'])
            mm = np.memmap(exports[n]['artifact']['payload'], dtype='u1', mode='r'); entries = exports[n]['tensors']
            ff = R.C.tensor(mm, entries, 'decoder.block.11.layer.2.layer_norm.weight'); fn = R.C.tensor(mm, entries, 'decoder.final_layer_norm.weight')
            ep = f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.'
            ops = [G.I8Operator(R.C.tensor(mm, entries, name), R.C.tensor(mm, entries, name, 'scales')) for name in (ep+'wi.weight', ep+'wo.weight', 'lm_head.weight')]
            wi, wo, head = ops; parameters = G.corrections(768, 8, 547, zero=True)
            pre = torch.tensor(row['pre'].copy()); scores = torch.tensor(item['scores']); target = torch.tensor(R.probability(captured[256]['logits']))
            output = G.forward(pre, scores, expert, ff, fn, wi, wo, head, parameters)
            for field, value in output.items(): R.C.exact(value.detach().numpy(), item['logits'] if field == 'logits' else row[field])
            native_loss = G.prediction_loss(output['logits'], target); native_loss.backward(); native = {k: z.grad.numpy().copy() for k, z in parameters.items()}
            assert np.count_nonzero(native['B']) == np.count_nonzero(native['D']) == 0
            sp = {k: z.detach().to(torch.float64).requires_grad_() for k, z in parameters.items()}
            xp = torch.tensor(row['pre'].astype(np.float64)); ss = torch.tensor(item['scores'].astype(np.float64))
            smooth = G.smooth_forward(xp, ss, expert, torch.tensor(ff.astype(np.float64)), torch.tensor(fn.astype(np.float64)), wi.smooth_weight(), wo.smooth_weight(), head.smooth_weight(), sp)
            G.prediction_loss(smooth, target).backward(); smooth_grad = {k: z.grad.numpy().copy() for k, z in sp.items()}
            x_smooth = row['pre'].astype(np.float64)/np.sqrt(np.mean(row['pre'].astype(np.float64)**2)+1e-6)*ff.astype(np.float64)
            up_smooth = wi.smooth_weight().numpy().dot(x_smooth); mask_native = row['up_raw'] > 0; mask_smooth = up_smooth > 0
            changed = np.flatnonzero(mask_native != mask_smooth); values = {k: z.detach().numpy() for k, z in parameters.items()}
            arrays = {'up_native': row['up_raw'], 'up_smooth': up_smooth, 'logits_native': item['logits'], 'logits_smooth': smooth.detach().numpy()}
            checks = []
            for use_smooth_mask in (False, True):
                for use_smooth_loss in (False, True):
                    mask = mask_smooth if use_smooth_mask else mask_native
                    logits = smooth.detach().numpy() if use_smooth_loss else item['logits'].astype(np.float64)
                    manual = local_gradient(row, values, fn.astype(np.float64), wi.smooth_weight().numpy(), wo.smooth_weight().numpy(), head.smooth_weight().numpy(), logits, target.numpy(), mask)
                    errors = {field: {'vs_native_autograd': R.relative(manual[field], native[field]), 'vs_global_smooth': R.relative(manual[field], smooth_grad[field])} for field in ('A', 'C')}
                    if not use_smooth_mask and not use_smooth_loss:
                        assert all(e['vs_native_autograd'] <= 1e-5 for e in errors.values()), ('local_reference', errors)
                    checks.append({'smooth_ReLU_mask': use_smooth_mask, 'smooth_loss_primals': use_smooth_loss, 'errors': errors})
                    for k in ('A', 'C'): arrays[f'manual.mask{int(use_smooth_mask)}.loss{int(use_smooth_loss)}.{k}'] = manual[k]
                    guard()
            for k in ('A', 'C'): arrays['native_gradient_'+k] = native[k]; arrays['smooth_gradient_'+k] = smooth_grad[k]
            result['diagnostic'] = {'n': n, 'source_expert': expert, 'rank': 8, 'native_zero_correction_forward_exact': True,
                'mixed_vs_global_gradient_relative': {k: R.relative(native[k], smooth_grad[k]) for k in ('A', 'C')},
                'smooth_vs_native_full_logits_relative': R.relative(smooth.detach().numpy(), item['logits']), 'ReLU_mask_changed_rows': changed.tolist(),
                'changed_raw_values': [{'row': int(j), 'native': float(row['up_raw'][j]), 'smooth': float(up_smooth[j])} for j in changed],
                'fixed_counterfactuals': checks}
            np.savez(OUT/'local_primal_and_gradient.npz', **arrays)
        for n, e in exports.items():
            p = Path(e['artifact']['payload']); assert initial[n] == (p.stat().st_size, p.stat().st_mtime_ns)
        size = sum(p.stat().st_size for p in OUT.glob('*')); assert size <= 8 << 20
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*'))]
        result['gates'] = {'retained_failure_and_whole_sources_exact': True, 'same_fixed_native_forward_exact': True,
            'independent_local_A_C_backward_reference_within1e_minus5': True, 'fixed_all_four_counterfactuals_and_finite_errors': True, 'CPU_budget_source_unchanged': True}
        result['resource'] = {'seconds': time.monotonic()-start, 'maximum_RSS_or_peak_working_set_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': size}
        result['decision'] = 'local_rules_conform_but_mixed_global_gradient_contract_remains_failed_consider_coherent_smooth_training_surrogate'
        result['scope'] = 'Diagnostic only on SAME source128 first point/seed547/rank8/target. Independent local saved-primal backward and fixed mask/loss counterfactuals, no fit license or threshold repair. No native rounding derivative claim. A coherent floating training surrogate must separately preserve bounded native forward fidelity, independent gradients/FD and new artifact whole quality/rate. No GPU/T4/network/new corpus/model.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': M.digest(args.out), 'diagnostic': result['diagnostic'], 'resource': result['resource'], 'decision': result['decision']}), flush=True)
    except BaseException as error:
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'maximum_checked_memory_bytes': peak, 'bytes_hashed': hashed})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
