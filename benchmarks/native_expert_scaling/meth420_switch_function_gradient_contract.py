"""Frozen CPU native-forward/smooth-backward prerequisite; no fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth368_switch_bank_manifest as B
import meth418_switch_function_capture as C
import meth419_switch_function_autograd as G

PROTOCOL = M.DOC/'METH_420_SWITCH_FUNCTION_GRADIENT_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth420_switch_function_gradient'
PRIOR = M.DOC/'meth418_switch_function_capture_result.json'
PRIOR_SHA = '4ebb37ed40d788eb85168d028b3e0b538cd7d65e009293cd32d5fb0eb117d829'


def relative(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))


def probability(x):
    p = np.exp(np.asarray(x, np.float64)-np.max(x)); return p/np.sum(p)


def numpy_smooth(pre, scores, chosen, ff, fn, wi, wo, head, parameters):
    # Independent literal F64 reference, without torch/custom autograd helpers.
    x = (pre/np.sqrt(np.mean(pre**2)+1e-6))*ff
    x = x+parameters['A'].dot(parameters['B'].dot(x))
    up = np.maximum(wi.dot(x), 0.)
    down = wo.dot(up)
    down = down+parameters['C'].dot(parameters['D'].dot(down))
    post = pre+probability(scores)[chosen]*down
    final = (post/np.sqrt(np.mean(post**2)+1e-6))*fn
    return head.dot(final/np.sqrt(len(pre)))


def numpy_loss(logits, target):
    maximum = float(np.max(logits))
    return maximum+np.log(np.sum(np.exp(logits-maximum)))-np.dot(target, logits)


def finite_difference(function, value, epsilon=1e-4):
    result = np.zeros_like(value, dtype=np.float64)
    for index in np.ndindex(value.shape):
        plus = value.copy(); minus = value.copy(); plus[index] += epsilon; minus[index] -= epsilon
        result[index] = (function(plus)-function(minus))/(2*epsilon)
    return result


def tiny_contract():
    rng = np.random.default_rng(419)
    d, f, v, n, rank = 7, 11, 13, 5, 2
    pre = np.linspace(.2, .8, d, dtype=np.float32)
    ff = np.linspace(.8, 1.2, d, dtype=np.float32); fn = ff[::-1].copy()
    wi = G.I8Operator(rng.integers(1, 12, (f, d), dtype=np.int8), np.linspace(.02, .03, f, dtype=np.float32))
    wo = G.I8Operator(rng.integers(-10, 11, (d, f), dtype=np.int8), np.linspace(.015, .025, d, dtype=np.float32))
    head = G.I8Operator(rng.integers(-9, 10, (v, d), dtype=np.int8), np.linspace(.04, .07, v, dtype=np.float32))
    scores = np.linspace(-.7, .9, n, dtype=np.float32); chosen = n-1
    target = probability(rng.normal(size=v)); parameters = G.corrections(d, rank, 419, zero=False)
    xp = torch.tensor(pre, requires_grad=True); sp = torch.tensor(scores, requires_grad=True)
    output = G.forward(xp, sp, chosen, ff, fn, wi, wo, head, parameters)
    G.prediction_loss(output['logits'], torch.tensor(target)).backward()
    smooth_params = {k: z.detach().to(torch.float64).requires_grad_() for k, z in parameters.items()}
    x64 = torch.tensor(pre.astype(np.float64), requires_grad=True)
    s64 = torch.tensor(scores.astype(np.float64), requires_grad=True)
    smooth = G.smooth_forward(x64, s64, chosen, torch.tensor(ff.astype(np.float64)), torch.tensor(fn.astype(np.float64)), wi.smooth_weight(), wo.smooth_weight(), head.smooth_weight(), smooth_params)
    G.prediction_loss(smooth, torch.tensor(target)).backward()
    original = {k: z.detach().numpy().copy() for k, z in smooth_params.items()}
    kw = dict(pre=pre.astype(np.float64), scores=scores.astype(np.float64), chosen=chosen, ff=ff.astype(np.float64), fn=fn.astype(np.float64),
              wi=wi.smooth_weight().numpy(), wo=wo.smooth_weight().numpy(), head=head.smooth_weight().numpy(), parameters=original)
    def objective(**updates): return numpy_loss(numpy_smooth(**(kw | updates)), target)
    checks = []
    for field in ('pre', 'scores', 'A', 'B', 'C', 'D'):
        if field in ('pre', 'scores'):
            value = kw[field]; fd = finite_difference(lambda z: objective(**{field: z}), value)
            smooth_grad = (x64 if field == 'pre' else s64).grad.numpy()
            native_grad = (xp if field == 'pre' else sp).grad.numpy()
        else:
            value = original[field]
            fd = finite_difference(lambda z: objective(parameters=original | {field: z}), value)
            smooth_grad = smooth_params[field].grad.numpy(); native_grad = parameters[field].grad.numpy()
        smooth_error = relative(smooth_grad, fd); native_error = relative(native_grad, fd)
        assert smooth_error <= 1e-5 and native_error <= 1e-3, (field, smooth_error, native_error)
        checks.append({'field': field, 'coordinates': value.size, 'smooth_F64_vs_numpy_FD_relative': smooth_error, 'native_STE_vs_smooth_FD_relative': native_error})
    assert relative(smooth.detach().numpy(), numpy_smooth(**kw)) <= 1e-12
    # Individual RMS derivative: omitting its coupling term is a negative.
    dy = rng.normal(size=d); x = pre.astype(np.float64); w = ff.astype(np.float64)
    scalar = lambda z: np.dot((z/np.sqrt(np.mean(z*z)+1e-6))*w, dy)
    fd = finite_difference(scalar, x); tx = torch.tensor(pre, requires_grad=True)
    G.NativeRMS.apply(tx, ff).backward(torch.tensor(dy, dtype=torch.float32))
    rms_error = relative(tx.grad.numpy(), fd); assert rms_error <= 1e-5
    detached = dy*w/np.sqrt(np.mean(x*x)+1e-6)
    assert relative(detached, fd) > 1e-3
    # Individual full-softmax selected probability: all inactive derivatives matter.
    fd = finite_difference(lambda z: probability(z)[chosen], scores.astype(np.float64))
    ts = torch.tensor(scores, requires_grad=True); G.NativeProbability.apply(ts, chosen).backward()
    prob_error = relative(ts.grad.numpy(), fd); assert prob_error <= 1e-5
    wrong = np.zeros(n); wrong[chosen] = fd[chosen]; assert relative(wrong, fd) > 1e-3
    tied = torch.zeros(n); C.exact(G.NativeProbability.apply(tied, 0).numpy(), np.float32(1/n))
    try: G.NativeProbability.apply(tied, 1)
    except AssertionError: pass
    else: raise AssertionError('tie_lower_ID_not_detected')
    # ReLU negative/positive/zero derivative, and signed-zero forward policy.
    relu = np.array([-2., -0., 0., .4, 3.], np.float32); tr = torch.tensor(relu, requires_grad=True)
    rr = G.NativeReLU.apply(tr); rr.sum().backward()
    C.exact(rr.detach().numpy(), np.where(relu < 0, np.float32(0), relu))
    C.exact(tr.grad.numpy(), (relu > 0).astype(np.float32))
    assert np.isfinite([z['smooth_F64_vs_numpy_FD_relative'] for z in checks]).all()
    return {'checks': checks, 'RMS_smooth_FD_relative': rms_error, 'probability_smooth_FD_relative': prob_error,
            'negative_RMS_detached_relative': relative(detached, finite_difference(scalar, x)),
            'negative_softmax_missing_tail_relative': relative(wrong, fd), 'ReLU_zero_policy_and_tie_controls': True}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; output_bytes = 0; stage = 'bindings'
    result = {'experiment': 'METH-420-native-final-bank-forward-and-explicit-smooth-gradient-contract-path-repair', 'baselines': [], 'real_gradient_controls': []}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        assert peak <= 4 << 30 and time.monotonic()-start <= 1200 and output_bytes <= 2 << 30, 'gradient_contract_20min_4GiB_2GiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        helpers = [Path(__file__), Path(G.__file__), PROTOCOL, PRIOR, Path(C.__file__), Path(M.__file__), Path(B.__file__)]
        result['helper_sha256'] = {}
        for p in helpers: M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        first_failure = M.DOC/'meth419_switch_function_gradient_result.failure.json'; M.committed(first_failure)
        assert sha(first_failure) == '8e93936e536c509117986d67c0a5338d12969156564da6b5a590d95f79df3864'
        result['retained419_failure_sha256'] = sha(first_failure)
        assert sha(PRIOR) == PRIOR_SHA; prior = json.loads(PRIOR.read_text(encoding='utf-8'))
        assert all(prior['gates'].values()) and len(prior['captures']) == 384
        for p, expected in prior['helper_sha256'].items(): M.committed(Path(p)); assert sha(p) == expected
        C.source_identity()
        for item in prior['output_inventory']: assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
        result['fresh_prior_inventory_files'] = len(prior['output_inventory']); result['prior_sha256'] = PRIOR_SHA
        ancestors = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.pid in ancestors: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_background_daemons', []).append({'pid': p.pid, 'argv': argv}); continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_model_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); assert psutil.Process().cpu_affinity() == [0]
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        torch_dir = Path(torch.__file__).parent
        result['runtime'] = {'torch': torch.__version__, 'torch_git': torch.version.git_version, 'numpy': np.__version__, 'CPU_affinity': [0], 'torch_threads': 1,
            'torch_CPU_DLL_sha256': sha(torch_dir/'lib/torch_cpu.dll'), 'torch_C_extension_sha256': sha(next(torch_dir.glob('_C*.pyd'))), 'GPU_used': False}
        mapped = {}; entries = {}; heads = {}; norms = {}; initial = {}; exports = {}
        for n, (name, expected) in C.U.EXPORT.items():
            stage = f'fresh_payload{n}'; p = M.DOC/name; M.committed(p); assert sha(p) == expected
            e = json.loads(p.read_text(encoding='utf-8')); assert all(e['gates'].values()); exports[n] = e; a = e['artifact']
            assert a == prior['artifacts'][str(n)]; payload = Path(a['payload']); initial[n] = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert initial[n][0] == a['bytes'] and sha(payload) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']
            B.read_manifest(a['manifest'], e['original_config'], e['tensors'], payload)
            mapped[n] = np.memmap(payload, dtype='u1', mode='r'); entries[n] = e['tensors']
            norms[n] = (C.tensor(mapped[n], entries[n], 'decoder.block.11.layer.2.layer_norm.weight'), C.tensor(mapped[n], entries[n], 'decoder.final_layer_norm.weight'))
            heads[n] = G.I8Operator(C.tensor(mapped[n], entries[n], 'lm_head.weight'), C.tensor(mapped[n], entries[n], 'lm_head.weight', 'scales'))
            print(json.dumps({'stage': stage, 'seconds': time.monotonic()-start}), flush=True)
        result['artifacts'] = {str(n): e['artifact'] for n, e in exports.items()}
        assert psutil.disk_usage(str(M.ROOT)).free >= 2 << 30, 'gradient_output_disk_reserve'
        OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime']['BLAS'] = threadpool_info(); assert all(v['num_threads'] == 1 for v in threadpool_info())
            stage = 'tiny_gradient'; result['tiny'] = tiny_contract(); guard()
            cached = {}; first = {}; total_head_rows = 0; native_digest = hashlib.sha256()
            for index, item in enumerate(prior['captures']):
                n = item['n']; stage = 'baseline_'+item['label']; s, t = len(item['source_ids']), len(item['decoder_ids'])
                assert sha(item['capture_path']) == item['capture_sha256'] and sha(item['output_path']) == item['output_sha256'] and sha(item['trace_path']) == item['trace_sha256']
                data = Path(item['capture_path']).read_bytes(); assert data[:8] == b'SWFUN001' and struct.unpack_from('<6I', data, 8) == (768, 3072, 32128, n, 11, 1)
                assert len(data) == 32+t*C.DTYPE.itemsize
                rows = np.frombuffer(data, dtype=C.DTYPE, offset=32)
                assert np.array_equal(rows['position'], np.arange(t)) and np.array_equal(rows['id'], item['decoder_ids'])
                assert np.all(rows['source_tokens'] == s) and np.array_equal(rows['route_index'], 6*s+6*np.arange(t)+5)
                old = Path(item['output_path']).read_bytes(); C.R.R.route_bytes(old, n, s, t)
                logits = np.frombuffer(old, dtype='<f4', count=t*32128, offset=36+(14*s*768+t*14*768)*4).reshape(t, 32128)
                trace = Path(item['trace_path']).read_bytes(); dt = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (n,))])
                assert trace[:8] == b'SWRTA001' and struct.unpack_from('<2I', trace, 8) == (n, 768) and len(trace) == 16+6*(s+t)*dt.itemsize
                router = np.frombuffer(trace, dtype=dt, offset=16)[6*s+5::6]; assert len(router) == t
                values = {field: [] for field in ('input', 'up_raw', 'up', 'down', 'probability', 'post', 'final', 'head_input', 'logits')}
                for i, row in enumerate(rows):
                    expert = int(row['expert']); assert row['accepted'] == 1
                    if n not in cached or cached[n][0] != expert:
                        ep = f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.'
                        wi = G.I8Operator(C.tensor(mapped[n], entries[n], ep+'wi.weight'), C.tensor(mapped[n], entries[n], ep+'wi.weight', 'scales'))
                        wo = G.I8Operator(C.tensor(mapped[n], entries[n], ep+'wo.weight'), C.tensor(mapped[n], entries[n], ep+'wo.weight', 'scales'))
                        cached[n] = (expert, wi, wo)
                    _, wi, wo = cached[n]
                    with torch.no_grad(): output = G.forward(torch.tensor(row['pre'].copy()), torch.tensor(router[i]['scores'].copy()), expert, *norms[n], wi, wo, heads[n])
                    for field, value in output.items():
                        value = value.numpy(); C.exact(value, logits[i] if field == 'logits' else row[field]); values[field].append(value.copy())
                    for field, stem in (('input', 'wi'), ('up', 'wo'), ('head_input', 'head')):
                        scale, codes = C.quant(output[field].numpy()); C.exact(scale, row[stem+'_scale']); C.exact(codes, row[stem+'_codes'])
                    native_digest.update(output['logits'].numpy().tobytes()); total_head_rows += 32128
                    if item['label'] == f'teacher.n{n}.book0.case0' and i == 0:
                        first[n] = {'row': row.copy(), 'scores': router[i]['scores'].copy(), 'logits': logits[i].copy()}
                    guard()
                path = OUT/(item['label']+'.npz'); np.savez(path, **{field: np.stack(v) for field, v in values.items()},
                    source_ids=np.asarray(item['source_ids'], dtype='<u4'), decoder_ids=np.asarray(item['decoder_ids'], dtype='<u4'))
                output_bytes += path.stat().st_size; guard()
                result['baselines'].append({'n': n, 'label': item['label'], 'positions': t, 'archive_path': str(path), 'archive_sha256': sha(path),
                    'complete_native_states_and_ALL32128_logits_byte_exact': True, 'source_capture_sha256': item['capture_sha256'],
                    'qualification_only_not_paired_training': item['label'].startswith('natural')})
                if (index+1) % 32 == 0: print(json.dumps({'baseline_cases': index+1, 'seconds': time.monotonic()-start, 'output_bytes': output_bytes}), flush=True)
            assert len(first) == 2 and len(result['baselines']) == 384
            cached.clear(); stage = 'real_zero_rank8_gradients'
            for n in (256, 128):
                row = first[n]['row']; expert = int(row['expert']); ep = f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.'
                wi = G.I8Operator(C.tensor(mapped[n], entries[n], ep+'wi.weight'), C.tensor(mapped[n], entries[n], ep+'wi.weight', 'scales'))
                wo = G.I8Operator(C.tensor(mapped[n], entries[n], ep+'wo.weight'), C.tensor(mapped[n], entries[n], ep+'wo.weight', 'scales'))
                parameters = G.corrections(768, 8, 419+n, zero=True)
                pre = torch.tensor(row['pre'].copy(), requires_grad=True); scores = torch.tensor(first[n]['scores'], requires_grad=True)
                target = torch.tensor(probability(first[384-n]['logits']), dtype=torch.float64)
                output = G.forward(pre, scores, expert, *norms[n], wi, wo, heads[n], parameters)
                for field, value in output.items(): C.exact(value.detach().numpy(), first[n]['logits'] if field == 'logits' else row[field])
                loss = G.prediction_loss(output['logits'], target); loss.backward(); guard()
                native_grads = {k: z.grad.detach().numpy().copy() for k, z in parameters.items()}
                assert np.linalg.norm(native_grads['A']) > 1e-10 and np.linalg.norm(native_grads['C']) > 1e-10
                assert np.count_nonzero(native_grads['B']) == np.count_nonzero(native_grads['D']) == 0
                sp = {k: z.detach().to(torch.float64).requires_grad_() for k, z in parameters.items()}
                xp = torch.tensor(row['pre'].astype(np.float64), requires_grad=True); ss = torch.tensor(first[n]['scores'].astype(np.float64), requires_grad=True)
                smooth = G.smooth_forward(xp, ss, expert, *[torch.tensor(w.astype(np.float64)) for w in norms[n]], wi.smooth_weight(), wo.smooth_weight(), heads[n].smooth_weight(), sp)
                smooth_loss = G.prediction_loss(smooth, target); smooth_loss.backward(); guard()
                comparisons = {}; directions = {}; rng = np.random.default_rng(419+n)
                original = {k: z.detach().numpy().copy() for k, z in sp.items()}
                kw = dict(pre=xp.detach().numpy(), scores=ss.detach().numpy(), chosen=expert,
                    ff=norms[n][0].astype(np.float64), fn=norms[n][1].astype(np.float64), wi=wi.smooth_weight().numpy(), wo=wo.smooth_weight().numpy(), head=heads[n].smooth_weight().numpy(), parameters=original)
                for field in ('A', 'B', 'C', 'D'):
                    error = relative(native_grads[field], sp[field].grad.numpy()); assert error <= 1e-3, ('real_STE_gradient', n, field, error)
                    comparisons[field] = error
                    direction = rng.choice([-1., 1.], size=original[field].shape)/np.sqrt(original[field].size)
                    epsilon = 1e-4
                    plus = original | {field: original[field]+epsilon*direction}; minus = original | {field: original[field]-epsilon*direction}
                    fd = (numpy_loss(numpy_smooth(**(kw | {'parameters': plus})), target.numpy())-numpy_loss(numpy_smooth(**(kw | {'parameters': minus})), target.numpy()))/(2*epsilon)
                    expected = float(np.sum(sp[field].grad.numpy()*direction)); error_fd = abs(fd-expected)/max(abs(expected), 1e-8)
                    assert error_fd <= 1e-5, ('real_directional_FD', n, field, error_fd)
                    directions[field] = {'finite_difference': float(fd), 'autograd_projection': expected, 'relative_error': error_fd}
                for field, a, b in (('pre', pre.grad.numpy(), xp.grad.numpy()), ('scores', scores.grad.numpy(), ss.grad.numpy())):
                    error = relative(a, b); assert error <= 1e-3, ('real_STE_gradient', n, field, error); comparisons[field] = error
                forward_error = relative(output['logits'].detach().numpy(), smooth.detach().numpy()); assert forward_error <= 1e-3
                path = OUT/f'rank8.n{n}.npz'; np.savez(path, **{k: z.detach().numpy() for k, z in parameters.items()},
                    **{'native_grad_'+k: z for k, z in native_grads.items()}, native_logits=output['logits'].detach().numpy(), smooth_logits=smooth.detach().numpy())
                output_bytes += path.stat().st_size; guard()
                result['real_gradient_controls'].append({'n': n, 'source_expert': expert, 'correction_rank': 8, 'zero_correction_ALL_states_logits_exact': True,
                    'native_loss': float(loss.detach()), 'smooth_loss': float(smooth_loss.detach()), 'smooth_forward_relative': forward_error,
                    'gradient_native_STE_vs_smooth_relative': comparisons, 'directions': directions, 'A_C_gradient_nonzero_B_D_zero': True,
                    'archive_path': str(path), 'archive_sha256': sha(path)})
            # Wrong native head coefficient and detached probability are full-forward negatives.
            stage = 'real_negative_controls'; row = first[256]['row']; head = heads[256]
            col = int(np.argmax(np.abs(row['head_codes']))); wrong = head.integers[0].copy(); wrong[col] ^= 1
            scale, codes = C.quant(row['head_input']); value = np.float32((np.float64(wrong @ codes.astype(np.int64))*np.float64(head.scales[0]))*np.float64(scale))
            try: C.exact(value, first[256]['logits'][0])
            except AssertionError: pass
            else: raise AssertionError('undetected_head_code_bit')
            changed = row['pre']+row['down']; final = C.norm(changed, norms[256][1]); changed_logits = head.native(final*np.float32(1/np.sqrt(768)))
            assert changed_logits.tobytes() != first[256]['logits'].tobytes()
            result['negative_controls'] = {'wrong_head_coefficient_detected': True, 'dropped_selected_probability_detected': True,
                'RMS_missing_coupling_detected': True, 'softmax_missing_tail_detected': True, 'wrong_tie_ID_detected': True}
        for n, e in exports.items():
            p = Path(e['artifact']['payload']); assert initial[n] == (p.stat().st_size, p.stat().st_mtime_ns)
        result['full_vocabulary_rows_byte_exact'] = total_head_rows; result['all_replayed_logit_bytes_sha256'] = native_digest.hexdigest()
        result['resource_ledger'] = {'one_bank_added128_I8_coefficients': 128*2*768*3072,
            'one_bank_added128_I8_and_F32_scales_bytes': 128*(2*768*3072+4*(3072+768)),
            'rank8_four_factor_parameters_per_added_function': 4*768*8, 'ALL128_rank8_trainable_parameters': 128*4*768*8,
            'ALL128_rank8_F32_parameters_bytes': 128*4*768*8*4, 'ALL128_rank8_F32_gradients_bytes': 128*4*768*8*4,
            'ALL128_rank8_two_F32_Adam_moment_bytes': 128*4*768*8*8, 'one_selected_rank8_F32_coefficients_addressed_bytes': 4*768*8*4,
            'one_head_I8_plus_scales_bytes': 32128*768+32128*4, 'one_head_I64_forward_workspace_bytes': 32128*768*8,
            'one_head_F64_dequant_backward_bytes': 32128*768*8, 'selector_optimizer_and_actual_fit_budget_not_yet_fixed': True,
            'frozen_pretrained_source_WI_WO_optimized': False}
        files = sorted(OUT.glob('*')); output_bytes = sum(p.stat().st_size for p in files); guard()
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
        result['gates'] = {'fresh_sources_manifests_prior_inventory_exact': True, 'ALL384_complete_zero_change_states_and_ALL_head_logits_exact': True,
            'tiny_smooth_FD_AND_native_STE_error_within_fixed_bounds': True, 'both_actual_rank8_zero_forward_and_live_A_C_gradients': True,
            'both_actual_STE_smooth_gradient_AND_FD_controls': True, 'native_and_gradient_negative_controls_detected': True, 'CPU_budget_and_source_unchanged': True}
        result['resource'] = {'seconds': time.monotonic()-start, 'maximum_process_RSS_or_peak_working_set_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': output_bytes,
            'baseline_positions': sum(v['positions'] for v in result['baselines'])}
        result['decision'] = 'eligible_only_for_new_bounded_rank8_function_selector_fit_protocol'
        result['scope'] = 'CPU numeric prerequisite only. Forward preserves native I8/A16/I64/F64-scale/F32 values; selected original router scores inherited exact capture, not recomputed source SIMD router. Backward is a declared dequantized/smooth STE, not native rounding/top1 derivatives. Finite differences validate smooth reference, compare approximate saved-primal backward separately. ALL384/4895 original paired and natural observations replayed, natural excluded from fitting. Two fixed real first-position prediction-loss controls, no updates or capacity-benefit evidence. No new native artifact/model/quality/rate/DRAM/LUT/GPU/T4/network/additional family.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': M.digest(args.out), 'gates': result['gates'], 'resource': result['resource'], 'decision': result['decision']}), flush=True)
    except BaseException as error:
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'maximum_checked_memory_bytes': peak, 'bytes_hashed': hashed})
        if OUT.exists(): result['partial_output_bytes'] = sum(p.stat().st_size for p in OUT.glob('*'))
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
