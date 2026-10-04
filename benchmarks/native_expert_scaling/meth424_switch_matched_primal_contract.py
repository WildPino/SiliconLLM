"""Frozen CPU qualification of local matched-primal continuation; no fitting."""
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
import meth424_switch_anchored_reference as L

PROTOCOL = M.DOC/'METH_424_SWITCH_MATCHED_PRIMAL_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth424_switch_matched_primal'
RECORDS = {
    'meth420_switch_function_gradient_result.failure.json': '17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427',
    'meth422_switch_function_gradient_result.failure.json': 'a35e8bf253d740fdf6335336ffbdc1619ef72659b8ff0524e9c9959c8ba465a0',
    'meth423_switch_saved_primal_gradient_result.json': 'a8ef145bd2c0fa496d5bf7dbfbdfb3b349b5061502ba3546efa4c6c56c5f2424'}


def qualify(label, pre, scores, chosen, ff, fn, wi, wo, head, parameters, target, tiny, expected=None):
    values = {k: z.detach().numpy().copy() for k, z in parameters.items()}
    xp = torch.tensor(pre.copy(), requires_grad=True); ss = torch.tensor(scores.copy(), requires_grad=True)
    native = G.forward(xp, ss, chosen, ff, fn, wi, wo, head, parameters)
    anchor = L.native_anchor(pre, scores, chosen, ff, fn, wi, wo, head, values)
    for k, z in native.items():
        R.C.exact(z.detach().numpy(), anchor[k])
        if expected is not None: R.C.exact(z.detach().numpy(), expected[k])
    native_loss = G.prediction_loss(native['logits'], torch.tensor(target)); native_loss.backward()
    native_grad = {k: z.grad.numpy().copy() for k, z in parameters.items()}
    native_grad.update(pre=xp.grad.numpy().copy(), scores=ss.grad.numpy().copy())
    if tiny: assert all(np.linalg.norm(native_grad[k]) > 0 for k in ('A', 'B', 'C', 'D'))
    else:
        assert np.linalg.norm(native_grad['A']) > 1e-10 and np.linalg.norm(native_grad['C']) > 1e-10
        assert np.count_nonzero(native_grad['B']) == np.count_nonzero(native_grad['D']) == 0
    kw = dict(pre=pre.astype(np.float64), scores=scores.astype(np.float64), chosen=chosen,
              ff=ff.astype(np.float64), fn=fn.astype(np.float64), wi=wi.smooth_weight().numpy(),
              wo=wo.smooth_weight().numpy(), head=head.smooth_weight().numpy(),
              parameters={k: v.astype(np.float64) for k, v in values.items()})
    offsets = L.detached_offsets(anchor, **kw)
    offset_digest = lambda: {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in offsets.items()}
    frozen_offsets = offset_digest()
    numpy_out = L.numpy_path(**kw, offsets=offsets)
    tpre = torch.tensor(kw['pre'], requires_grad=True); tscores = torch.tensor(kw['scores'], requires_grad=True)
    tparams = {k: torch.tensor(v, requires_grad=True) for k, v in kw['parameters'].items()}
    torch_out = L.torch_path(tpre, tscores, chosen, *[torch.tensor(kw[k]) for k in ('ff', 'fn', 'wi', 'wo', 'head')], tparams, offsets)
    endpoints = {}
    for k in L.FIELDS:
        a = torch_out[k].detach().numpy(); b = numpy_out[k]; c = np.asarray(anchor[k], np.float64)
        errors = {'torch_vs_native_relative': R.relative(a, c), 'numpy_vs_native_relative': R.relative(b, c),
                  'torch_vs_native_absmax': float(np.max(np.abs(a-c))), 'numpy_vs_native_absmax': float(np.max(np.abs(b-c)))}
        assert errors['torch_vs_native_relative'] <= 1e-12 and errors['numpy_vs_native_relative'] <= 1e-12, ('endpoint_relative', label, k, errors)
        assert errors['torch_vs_native_absmax'] <= 1e-9 and errors['numpy_vs_native_absmax'] <= 1e-9, ('endpoint_absolute', label, k, errors)
        endpoints[k] = errors
    reference_loss = G.prediction_loss(torch_out['logits'], torch.tensor(target)); reference_loss.backward()
    reference_grad = {k: z.grad.numpy().copy() for k, z in tparams.items()}
    reference_grad.update(pre=tpre.grad.numpy().copy(), scores=tscores.grad.numpy().copy())
    comparisons = {k: R.relative(native_grad[k], reference_grad[k]) for k in native_grad}
    assert all(v <= 1e-3 for v in comparisons.values()), ('matched_gradient', label, comparisons)
    crossings = 0; base_mask = anchor['up_raw'] > 0
    def objective(field, value):
        nonlocal crossings
        update = {field: value} if field in ('pre', 'scores') else {'parameters': kw['parameters'] | {field: value}}
        output = L.numpy_path(**(kw | update), offsets=offsets)
        crossings += int(np.count_nonzero((output['up_raw'] > 0) != base_mask))
        return R.numpy_loss(output['logits'], target)
    checks = []; fd_vectors = {}; arrays = {}
    order = ('pre', 'scores', 'A', 'B', 'C', 'D') if tiny else ('A', 'B', 'C', 'D', 'pre', 'scores')
    rng = np.random.default_rng(419+len(scores)); epsilon = 1e-4
    for field in order:
        value = kw[field] if field in ('pre', 'scores') else kw['parameters'][field]
        if tiny:
            fd = R.finite_difference(lambda z: objective(field, z), value, epsilon)
            referr = R.relative(reference_grad[field], fd); nativeerr = R.relative(native_grad[field], fd)
            assert referr <= 1e-5 and nativeerr <= 1e-3, ('tiny_FD', field, referr, nativeerr)
            fd_vectors[field] = fd; arrays['finite_difference_'+field] = fd
            checks.append({'field': field, 'coordinates': int(value.size), 'reference_vs_numpy_FD_relative': referr, 'native_vs_numpy_FD_relative': nativeerr})
        else:
            direction = rng.choice([-1., 1.], size=value.shape)/np.sqrt(value.size)
            fd = (objective(field, value+epsilon*direction)-objective(field, value-epsilon*direction))/(2*epsilon)
            rp = float(np.sum(reference_grad[field]*direction)); npj = float(np.sum(native_grad[field]*direction))
            referr = abs(fd-rp)/max(abs(rp), 1e-8); nativeerr = abs(fd-npj)/max(abs(npj), 1e-8)
            assert referr <= 1e-5 and nativeerr <= 1e-3, ('real_FD', label, field, referr, nativeerr)
            arrays['direction_'+field] = direction
            checks.append({'field': field, 'epsilon': epsilon, 'finite_difference': float(fd), 'reference_projection': rp,
                           'native_projection': npj, 'reference_vs_FD_error': referr, 'native_vs_FD_error': nativeerr})
    assert crossings == 0, ('ReLU_crossings', label, crossings)
    assert offset_digest() == frozen_offsets, 'offsets_changed_during_FD'
    # Endpoint controls are meaningful against native values, not just twin equality.
    dropped = offsets | {'logits': np.zeros_like(offsets['logits'])}
    wrong_head = L.numpy_path(**kw, offsets=dropped)['logits']; wrong_head_error = R.relative(wrong_head, anchor['logits'])
    assert wrong_head_error > 1e-12 or np.max(np.abs(wrong_head-anchor['logits'])) > 1e-9, 'undetected_dropped_head_offset'
    malformed = offsets | {'logits': np.zeros((len(target)+1,), np.float64)}
    try: L.numpy_path(**kw, offsets=malformed)
    except AssertionError: pass
    else: raise AssertionError('undetected_offset_shape')
    negatives = {'dropped_head_offset_detected': True, 'dropped_head_offset_relative': wrong_head_error, 'malformed_offset_shape_detected': True}
    if tiny:
        # Detached selected probability has a zero score derivative, despite all
        # nonselected scores contributing to the complete coupled FD derivative.
        wrong_scores = np.zeros_like(reference_grad['scores'])
        error = R.relative(wrong_scores, fd_vectors['scores']); assert error > 1e-3
        negatives['detached_probability_coupled_score_gradient_error'] = error
    for k in L.FIELDS:
        arrays['native_'+k] = anchor[k]; arrays['numpy_continuation_'+k] = numpy_out[k]
        arrays['torch_continuation_'+k] = torch_out[k].detach().numpy(); arrays['offset_'+k] = offsets[k]
    for k in native_grad:
        arrays['native_gradient_'+k] = native_grad[k]; arrays['reference_gradient_'+k] = reference_grad[k]
    arrays.update({'factor_'+k: z for k, z in values.items()}); arrays.update(pre=pre, scores=scores, target=target)
    path = OUT/(label+'.npz'); np.savez(path, **arrays)
    return {'label': label, 'n': len(scores), 'selected': chosen, 'all_native_endpoints_exact_independent_anchor': True,
            'zero_forward_full_head_exact418': expected is not None, 'endpoints': endpoints, 'native_vs_reference_gradient_relative': comparisons,
            'finite_differences': checks, 'offset_hashes_unchanged': frozen_offsets, 'ReLU_crossings': crossings, 'negatives': negatives,
            'native_loss': float(native_loss.detach()), 'reference_loss': float(reference_loss.detach()), 'archive_path': str(path)}


def tiny_inputs():
    rng = np.random.default_rng(419); d, f, v, n = 7, 11, 13, 5
    pre = np.linspace(.2, .8, d, dtype=np.float32); ff = np.linspace(.8, 1.2, d, dtype=np.float32)
    wi = G.I8Operator(rng.integers(1, 12, (f, d), dtype=np.int8), np.linspace(.02, .03, f, dtype=np.float32))
    wo = G.I8Operator(rng.integers(-10, 11, (d, f), dtype=np.int8), np.linspace(.015, .025, d, dtype=np.float32))
    head = G.I8Operator(rng.integers(-9, 10, (v, d), dtype=np.int8), np.linspace(.04, .07, v, dtype=np.float32))
    return pre, np.linspace(-.7, .9, n, dtype=np.float32), n-1, ff, ff[::-1].copy(), wi, wo, head, G.corrections(d, 2, 419, zero=False), R.probability(rng.normal(size=v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'
    result = {'experiment': 'METH-424-matched-primal-local-continuation-contract', 'points': []}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.glob('*')) if OUT.exists() else 0
        assert peak <= 3 << 30 and time.monotonic()-start <= 300 and size <= 16 << 20, 'contract_5min_3GiB_16MiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        helpers = [Path(__file__), Path(L.__file__), PROTOCOL, Path(G.__file__), Path(R.__file__), Path(R.C.__file__), Path(R.B.__file__), Path(M.__file__)]
        result['helper_sha256'] = {}
        for p in helpers: M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        records = {}
        for name, expected in RECORDS.items():
            p = M.DOC/name; M.committed(p); assert sha(p) == expected; result['helper_sha256'][str(p)] = expected
            records[name] = json.loads(p.read_text(encoding='utf-8'))
        failure = records['meth422_switch_function_gradient_result.failure.json']; assert 'real_STE_gradient' in failure['error']
        for a in failure['real_gradient_controls']: assert sha(a['archive_path']) == a['archive_sha256']
        diagnostic = records['meth423_switch_saved_primal_gradient_result.json']; assert all(diagnostic['gates'].values())
        for a in diagnostic['output_inventory']: assert sha(a['path']) == a['sha256']
        prefix = b'def prediction_loss(logits, target):'; legacy = Path(G.__file__).with_name('meth419_switch_function_autograd.py')
        M.committed(legacy); assert Path(G.__file__).read_bytes().split(prefix)[0] == legacy.read_bytes().split(prefix)[0]
        assert sha(R.PRIOR) == R.PRIOR_SHA; M.committed(R.PRIOR); prior = json.loads(R.PRIOR.read_text(encoding='utf-8'))
        assert all(prior['gates'].values()) and len(prior['captures']) == 384
        for p, expected in prior['helper_sha256'].items(): M.committed(Path(p)); assert sha(p) == expected
        R.C.source_identity()
        for a in prior['output_inventory']: assert Path(a['path']).stat().st_size == a['bytes'] and sha(a['path']) == a['sha256']
        baselines = records['meth420_switch_function_gradient_result.failure.json']['baselines']; assert len(baselines) == 384
        for a in baselines: assert a['complete_native_states_and_ALL32128_logits_byte_exact'] and sha(a['archive_path']) == a['archive_sha256']
        result['inherited_forward'] = {'source_prefix_exact419': True, 'fresh418_inventory': len(prior['output_inventory']),
                                       'fresh420_archives': len(baselines), 'positions': sum(a['positions'] for a in baselines),
                                       'ALL_vocabulary_rows': sum(a['positions'] for a in baselines)*32128}
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_background_daemons', []).append(p.pid); continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_model_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        td = Path(torch.__file__).parent
        result['runtime'] = {'torch': torch.__version__, 'torch_git': torch.version.git_version, 'numpy': np.__version__, 'CPU_affinity': [0], 'torch_threads': 1,
            'torch_CPU_DLL_sha256': sha(td/'lib/torch_cpu.dll'), 'torch_C_extension_sha256': sha(next(td.glob('_C*.pyd'))), 'GPU': False}
        mapped = {}; entries = {}; norms = {}; heads = {}; exports = {}; first = {}; initial = {}
        for n, (name, expected) in R.C.U.EXPORT.items():
            stage = f'fresh_source{n}'; p = M.DOC/name; M.committed(p); assert sha(p) == expected
            e = json.loads(p.read_text(encoding='utf-8')); assert all(e['gates'].values()); a = e['artifact']; exports[n] = a
            assert a == prior['artifacts'][str(n)]; payload = Path(a['payload']); initial[n] = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert initial[n][0] == a['bytes'] and sha(payload) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']
            R.B.read_manifest(a['manifest'], e['original_config'], e['tensors'], payload)
            mapped[n] = np.memmap(payload, dtype='u1', mode='r'); entries[n] = e['tensors']
            norms[n] = tuple(R.C.tensor(mapped[n], entries[n], k) for k in ('decoder.block.11.layer.2.layer_norm.weight', 'decoder.final_layer_norm.weight'))
            heads[n] = G.I8Operator(R.C.tensor(mapped[n], entries[n], 'lm_head.weight'), R.C.tensor(mapped[n], entries[n], 'lm_head.weight', 'scales'))
            item = next(v for v in prior['captures'] if v['label'] == f'teacher.n{n}.book0.case0')
            for kind in ('capture', 'trace', 'output'): assert sha(item[kind+'_path']) == item[kind+'_sha256']
            data = Path(item['capture_path']).read_bytes(); assert data[:8] == b'SWFUN001' and len(data) == 32+14*R.C.DTYPE.itemsize
            row = np.frombuffer(data, dtype=R.C.DTYPE, offset=32)[0].copy()
            out = Path(item['output_path']).read_bytes(); R.C.R.R.route_bytes(out, n, 29, 14)
            logits = np.frombuffer(out, dtype='<f4', count=14*32128, offset=36+(14*29*768+14*14*768)*4).reshape(14, 32128)[0].copy()
            trace = Path(item['trace_path']).read_bytes(); dt = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (n,))])
            assert trace[:8] == b'SWRTA001' and struct.unpack_from('<2I', trace, 8) == (n, 768)
            scores = np.frombuffer(trace, dtype=dt, offset=16)[179]['scores'].copy(); assert int(np.argmax(scores)) == int(row['expert'])
            first[n] = {'row': row, 'scores': scores, 'logits': logits, 'pair': item['pairing_sha256']}
            print(json.dumps({'stage': stage, 'seconds': time.monotonic()-start}), flush=True)
        assert first[128]['pair'] == first[256]['pair']; result['artifacts'] = {str(k): v for k, v in exports.items()}
        assert psutil.disk_usage(str(M.ROOT)).free >= 2 << 30; OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime']['BLAS'] = threadpool_info(); assert all(x['num_threads'] == 1 for x in threadpool_info())
            stage = 'original_tiny_primitive_controls'; result['original422_tiny'] = R.tiny_contract(); guard()
            stage = 'tiny_matched_primal'; result['points'].append(qualify('tiny', *tiny_inputs(), tiny=True)); guard()
            for n in (256, 128):
                stage = f'real_matched_primal{n}'; row = first[n]['row']; expert = int(row['expert']); ep = f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.'
                wi = G.I8Operator(R.C.tensor(mapped[n], entries[n], ep+'wi.weight'), R.C.tensor(mapped[n], entries[n], ep+'wi.weight', 'scales'))
                wo = G.I8Operator(R.C.tensor(mapped[n], entries[n], ep+'wo.weight'), R.C.tensor(mapped[n], entries[n], ep+'wo.weight', 'scales'))
                expected = {k: first[n]['logits'] if k == 'logits' else row[k] for k in ('input', 'up_raw', 'up', 'down', 'probability', 'post', 'final', 'head_input', 'logits')}
                result['points'].append(qualify(f'real.n{n}', row['pre'].copy(), first[n]['scores'], expert, *norms[n], wi, wo, heads[n],
                    G.corrections(768, 8, 419+n, zero=True), R.probability(first[384-n]['logits']), tiny=False, expected=expected)); guard()
                print(json.dumps({'stage': stage, 'seconds': time.monotonic()-start}), flush=True)
            stage = 'native_negative_controls'; row = first[256]['row']; head = heads[256]
            col = int(np.argmax(np.abs(row['head_codes']))); wrong = head.integers[0].copy(); wrong[col] ^= 1
            scale, codes = R.C.quant(row['head_input']); value = np.float32(float(wrong @ codes.astype(np.int64))*float(head.scales[0])*float(scale))
            try: R.C.exact(value, first[256]['logits'][0])
            except AssertionError: pass
            else: raise AssertionError('undetected_head_code_bit')
            changed = R.C.norm(row['pre']+row['down'], norms[256][1]); changed_logits = head.native(changed*np.float32(1/np.sqrt(768)))
            assert changed_logits.tobytes() != first[256]['logits'].tobytes()
            result['negative_controls'] = {'wrong_head_coefficient': True, 'dropped_selected_probability': True, 'original_RMS_softmax_tie_ReLU_controls': True,
                                          'dropped_head_offset_ALL_points': True, 'malformed_offset_shape_ALL_points': True, 'coupled_detached_probability_gradient': True}
        for n, a in exports.items():
            p = Path(a['payload']); assert initial[n] == (p.stat().st_size, p.stat().st_mtime_ns)
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*'))]
        result['gates'] = {'prior_failures_retained_and_whole_sources_fresh': True, 'immutable_native_forward_ALL384_inherited': True,
            'independent_native_anchor_and_ALL_continuation_endpoints': True, 'tiny_nonzero_ALL_coordinate_independent_FD': True,
            'both_same_real_rank8_ALL_gradient_and_directional_FD': True, 'fixed_offsets_no_ReLU_crossings': True,
            'ALL_native_offset_and_gradient_negatives_detected': True, 'CPU_resource_and_source_unchanged': True}
        result['resource'] = {'seconds': time.monotonic()-start, 'maximum_RSS_or_peak_working_set_bytes': peak, 'bytes_hashed': hashed,
                              'output_bytes': sum(a['bytes'] for a in result['output_inventory'])}
        result['decision'] = 'eligible_only_for_NEW_bounded_rank8_function_selector_fit_protocol'
        result['scope'] = 'Local matched-primal approximate derivative qualification only. Constant offsets at one anchor, not true native rounding/top1 derivatives or a globally smooth native model. ALL384/4895/157266560-row forward inherited exact419/422 prefix and fresh420 archives; two zero-rank8 real forwards newly run. Full NumPy FD independent of ordinary F64 torch derivative. Original422 mixed/global comparison remains FAIL. No updates, selector, combined artifact, useful capacity, quality/rate/DRAM/LUT/GPU/T4/network/new corpus/additional family.'
        guard(); M.write(args.out, result)
        print(json.dumps({'sha256': M.digest(args.out), 'gates': result['gates'], 'resource': result['resource'], 'decision': result['decision']}), flush=True)
    except BaseException as error:
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'maximum_checked_memory_bytes': peak, 'bytes_hashed': hashed})
        if OUT.exists(): result['partial_output_bytes'] = sum(p.stat().st_size for p in OUT.glob('*'))
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
