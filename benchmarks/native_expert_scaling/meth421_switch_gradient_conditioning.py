"""Diagnose the retained real FD failure; no fit or gradient qualification."""
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
import meth420_switch_function_gradient_contract as R
import meth419_switch_function_autograd as G

PROTOCOL = M.DOC/'METH_421_SWITCH_GRADIENT_CONDITIONING_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth421_switch_gradient_conditioning'
FAILURE = M.DOC/'meth420_switch_function_gradient_result.failure.json'
FAILURE_SHA = '17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427'


def dot(matrix, x):
    # Avoid materializing the full vocabulary matrix in complex128.
    if np.iscomplexobj(x) and not np.iscomplexobj(matrix):
        return matrix.dot(x.real)+1j*matrix.dot(x.imag)
    return matrix.dot(x)


def complex_reference(pre, scores, chosen, ff, fn, wi, wo, head, parameters):
    x = pre/np.sqrt(np.sum(pre*pre)/pre.size+1e-6)*ff
    x = x+dot(parameters['A'], dot(parameters['B'], x))
    raw = dot(wi, x); up = np.where(raw.real > 0, raw, 0)
    down = dot(wo, up); down = down+dot(parameters['C'], dot(parameters['D'], down))
    exp = np.exp(scores-float(np.max(scores.real))); p = exp[chosen]/np.sum(exp)
    post = pre+p*down; final = post/np.sqrt(np.sum(post*post)/post.size+1e-6)*fn
    return dot(head, final/np.sqrt(pre.size)), raw


def centered_loss(z, target, selected):
    shifted = z-z[selected]; tail = np.delete(np.exp(shifted), selected)
    return np.log1p(np.sum(tail))-target.dot(shifted)


def torch_centered_loss(z, target, selected):
    shifted = z-z[selected]; exp = torch.exp(shifted)
    tail = torch.cat([exp[:selected], exp[selected+1:]])
    return torch.log1p(torch.sum(tail))-torch.dot(target, shifted)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; result = {'experiment': 'METH-421-real-gradient-check-conditioning-diagnostic', 'points': []}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        assert peak <= 3 << 30 and time.monotonic()-start <= 120, 'diagnostic_2min_3GiB'
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
        assert sha(FAILURE) == FAILURE_SHA; fail = json.loads(FAILURE.read_text(encoding='utf-8'))
        assert len(fail['baselines']) == 384 and sum(v['positions'] for v in fail['baselines']) == 4895
        assert 'real_directional_FD' in fail['error'] and fail['real_gradient_controls'] == []
        for item in fail['baselines']: assert sha(item['archive_path']) == item['archive_sha256']
        prior_path = R.PRIOR; M.committed(prior_path); assert sha(prior_path) == R.PRIOR_SHA
        prior = json.loads(prior_path.read_text(encoding='utf-8')); assert all(prior['gates'].values())
        result['fresh384_previous_baseline_archives_exact'] = True
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_model_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_affinity': psutil.Process().cpu_affinity(), 'GPU': False}
        mapped = {}; entries = {}; first = {}; initial = {}
        for n, (name, expected) in R.C.U.EXPORT.items():
            stage = f'bindings{n}'; p = M.DOC/name; M.committed(p); assert sha(p) == expected
            export = json.loads(p.read_text(encoding='utf-8')); assert all(export['gates'].values()); a = export['artifact']
            payload = Path(a['payload']); initial[n] = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert a == prior['artifacts'][str(n)] and sha(payload) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']
            R.B.read_manifest(a['manifest'], export['original_config'], export['tensors'], payload)
            mapped[n] = np.memmap(payload, dtype='u1', mode='r'); entries[n] = export['tensors']
            item = next(v for v in prior['captures'] if v['label'] == f'teacher.n{n}.book0.case0')
            for kind in ('capture', 'output', 'trace'): assert sha(item[kind+'_path']) == item[kind+'_sha256']
            data = Path(item['capture_path']).read_bytes(); assert data[:8] == b'SWFUN001' and len(data) == 32+14*R.C.DTYPE.itemsize
            row = np.frombuffer(data, dtype=R.C.DTYPE, offset=32)[0].copy()
            old = Path(item['output_path']).read_bytes(); logits = np.frombuffer(old, dtype='<f4', count=14*32128, offset=36+(14*29*768+14*14*768)*4).reshape(14, 32128)[0].copy()
            trace = Path(item['trace_path']).read_bytes(); dt = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (n,))])
            scores = np.frombuffer(trace, dtype=dt, offset=16)[179]['scores'].copy()
            assert int(np.argmax(scores)) == int(row['expert']); first[n] = {'row': row, 'scores': scores, 'logits': logits, 'item': item}
            print(json.dumps({'stage': stage, 'seconds': time.monotonic()-start}), flush=True)
        assert first[256]['item']['pairing_sha256'] == first[128]['item']['pairing_sha256']
        OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime']['BLAS'] = threadpool_info(); assert all(v['num_threads'] == 1 for v in threadpool_info())
            # Known analytic quadratic validates the imaginary directional arithmetic.
            z = np.array([.3, -.7, 1.2]); direction = np.array([.2, .4, -.1]); h = 1e-20
            slope = np.imag(np.sum((z+1j*h*direction)**2))/h
            assert abs(slope-2*np.dot(z, direction)) <= 1e-14
            for n in (256, 128):
                stage = f'diagnostic{n}'; item = first[n]; row = item['row']; expert = int(row['expert'])
                ff = R.C.tensor(mapped[n], entries[n], 'decoder.block.11.layer.2.layer_norm.weight').astype(np.float64)
                fn = R.C.tensor(mapped[n], entries[n], 'decoder.final_layer_norm.weight').astype(np.float64)
                ep = f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.'
                ops = [G.I8Operator(R.C.tensor(mapped[n], entries[n], name), R.C.tensor(mapped[n], entries[n], name, 'scales')) for name in (ep+'wi.weight', ep+'wo.weight', 'lm_head.weight')]
                wi, wo, head = [op.smooth_weight() for op in ops]
                parameters = G.corrections(768, 8, 419+n, zero=True)
                params = {k: z.detach().to(torch.float64).requires_grad_() for k, z in parameters.items()}
                pre = torch.tensor(row['pre'].astype(np.float64)); scores = torch.tensor(item['scores'].astype(np.float64))
                target = torch.tensor(R.probability(first[384-n]['logits']))
                logits = G.smooth_forward(pre, scores, expert, torch.tensor(ff), torch.tensor(fn), wi, wo, head, params)
                selected = int(torch.argmax(logits)); old_loss = G.prediction_loss(logits, target); stable_loss = torch_centered_loss(logits, target, selected)
                old_grads = torch.autograd.grad(old_loss, list(params.values()), retain_graph=True)
                stable_grads = torch.autograd.grad(stable_loss, list(params.values()))
                original = {k: z.detach().numpy().copy() for k, z in params.items()}
                kw = dict(pre=pre.numpy(), scores=scores.numpy(), chosen=expert, ff=ff, fn=fn, wi=wi.numpy(), wo=wo.numpy(), head=head.numpy(), parameters=original)
                reference_logits, raw = complex_reference(**kw)
                assert R.relative(logits.detach().numpy(), reference_logits) <= 1e-12
                assert abs(float(old_loss)-float(stable_loss)) <= 1e-12
                checks = []; rng = np.random.default_rng(419+n)
                arrays = {'old_logits': logits.detach().numpy(), 'reference_logits': reference_logits}
                for field, old_grad, stable_grad in zip(params, old_grads, stable_grads):
                    direction = rng.choice([-1., 1.], size=original[field].shape)/np.sqrt(original[field].size)
                    epsilon = 1e-4; plus = original | {field: original[field]+epsilon*direction}; minus = original | {field: original[field]-epsilon*direction}
                    zp = R.numpy_smooth(**(kw | {'parameters': plus})); zm = R.numpy_smooth(**(kw | {'parameters': minus}))
                    plain = (R.numpy_loss(zp, target.numpy())-R.numpy_loss(zm, target.numpy()))/(2*epsilon)
                    centered = (centered_loss(zp, target.numpy(), selected)-centered_loss(zm, target.numpy(), selected))/(2*epsilon)
                    perturbed = original | {field: original[field].astype(np.complex128)+1j*1e-20*direction}
                    zc, _ = complex_reference(**(kw | {'parameters': perturbed}))
                    imaginary = float(np.imag(centered_loss(zc, target.numpy(), selected))/1e-20)
                    expected = float(np.sum(old_grad.numpy()*direction)); stable_expected = float(np.sum(stable_grad.numpy()*direction))
                    def err(value, expected): return float(abs(value-expected)/max(abs(expected), 1e-8))
                    _, rp = complex_reference(**(kw | {'parameters': plus})); _, rm = complex_reference(**(kw | {'parameters': minus}))
                    crossings = int(np.count_nonzero((rp.real > 0) != (rm.real > 0)))
                    record = {'field': field, 'autograd_old_projection': expected, 'autograd_centered_projection': stable_expected,
                        'plain_central_FD': float(plain), 'centered_central_FD': float(centered), 'complex_step_centered': imaginary,
                        'plain_FD_old_relative': err(plain, expected), 'centered_FD_old_relative': err(centered, expected),
                        'complex_old_relative': err(imaginary, expected), 'complex_centered_relative': err(imaginary, stable_expected),
                        'old_centered_gradient_relative': R.relative(old_grad.numpy(), stable_grad.numpy()), 'ReLU_crossings_plus_minus': crossings}
                    assert np.isfinite(list(v for v in record.values() if isinstance(v, (int, float)))).all()
                    checks.append(record); arrays['direction_'+field] = direction; arrays['old_gradient_'+field] = old_grad.numpy(); arrays['centered_gradient_'+field] = stable_grad.numpy()
                    guard()
                path = OUT/f'diagnostic.n{n}.npz'; np.savez(path, **arrays)
                result['points'].append({'n': n, 'source_expert': expert, 'old_loss': float(old_loss.detach()), 'centered_loss': float(stable_loss.detach()),
                    'minimum_abs_smooth_WI_preReLU': float(np.min(np.abs(raw))), 'zero_smooth_WI_preReLU_rows': int(np.count_nonzero(raw == 0)),
                    'checks': checks, 'archive_path': str(path), 'archive_sha256': sha(path)})
                print(json.dumps({'n': n, 'checks': checks}), flush=True)
        for n, mm in mapped.items():
            p = Path(mm.filename); assert initial[n] == (p.stat().st_size, p.stat().st_mtime_ns)
        size = sum(p.stat().st_size for p in OUT.glob('*')); assert size <= 16 << 20
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*'))]
        complex_match = all(c['complex_old_relative'] <= 1e-5 for p in result['points'] for c in p['checks'])
        centered_match = all(c['centered_FD_old_relative'] <= 1e-5 for p in result['points'] for c in p['checks'])
        crossing = sum(c['ReLU_crossings_plus_minus'] for p in result['points'] for c in p['checks'])
        result['diagnostic'] = {'old_autograd_matches_independent_complex_at_same1e_minus5': complex_match,
            'centered_FD_matches_at_same1e_minus5': centered_match, 'total_ReLU_crossings': crossing,
            'original_plain_A_256_FD_error': result['points'][0]['checks'][0]['plain_FD_old_relative']}
        result['gates'] = {'prior_failure_and384_baseline_archives_fresh_exact': True, 'whole_source_and_manifest_identity': True,
            'two_fixed_no_fit_points_and_finite_records': True, 'analytic_complex_arithmetic_control': True, 'within_CPU_diagnostic_budget': True}
        result['resource'] = {'seconds': time.monotonic()-start, 'maximum_RSS_or_peak_working_set_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': size}
        result['decision'] = 'new_stable_loss_gradient_check_protocol_required_before_fit' if complex_match and centered_match and crossing == 0 else 'retain_unresolved_gradient_conditioning_before_fit'
        result['scope'] = 'Diagnostic ONLY, original420 FAIL stays failed. Same two real point/seed/rank/directions, original epsilon1e-4 plus centered algebra and fixed imaginary epsilon1e-20. No source-native/STE-gradient qualification or fit license; actual numerical slopes distinguish FD conditioning/crossings/mismatch. No new docs/GPU/T4/network/native model/quality/rate/capacity.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': M.digest(args.out), 'diagnostic': result['diagnostic'], 'resource': result['resource'], 'decision': result['decision']}), flush=True)
    except BaseException as error:
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'maximum_checked_memory_bytes': peak, 'bytes_hashed': hashed})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
