"""Single-donor functional-contrast subspace oracle, never a runtime export."""
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
from threadpoolctl import threadpool_info, threadpool_limits

import meth443_switch_native_function_causality as N

M, R, G, X = N.M, N.R, N.G, N.X
PROTOCOL = M.DOC/'METH_444_SWITCH_FUNCTION_CONTRAST_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth444_switch_function_contrast'
PARENT = M.DOC/'meth443_switch_native_causality_result.json'
PARENT_SHA = 'b8e8614fb28ad50b91865dba7486c6e4cc6bb8c00cee3bbfe899ab495af5bf80'
RANKS = (0, 32, 64, 128, 256)


def information(logits):
    """Uniform do(E) Jensen-Shannon; independently check entropy difference."""
    q = np.stack([R.probability(z) for z in logits])
    lp = np.stack([N.log_probability(z) for z in logits])
    mix = np.mean(q, axis=0)
    lm = np.zeros_like(mix)
    np.log(mix, out=lm, where=mix > 0)
    assert np.all((q == 0) | (mix[None, :] > 0))
    js = np.mean(np.sum(q*(lp-lm), axis=1))
    separate = -np.dot(mix, lm)+np.mean(np.sum(q*lp, axis=1))
    assert np.isfinite(q).all() and np.isfinite(lp).all()
    assert abs(js-separate) <= 1e-10 and -1e-10 <= js <= np.log(128)+1e-10
    return q, lp, float(js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic()
    numeric_start = None
    peak = 0
    hashed = 0
    stage = 'bindings'
    result = {'experiment': 'METH-444-source128-functional-common-private-subspace-oracle', 'ranks': {}}

    def guard():
        nonlocal peak
        memory = psutil.Process().memory_info()
        peak = max(peak, memory.rss, getattr(memory, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        elapsed = time.monotonic()-start
        assert peak <= 4 << 30 and size <= 768 << 20, '4GiB_768MiB'
        assert elapsed <= 600 and ((numeric_start is None and elapsed <= 300) or
               (numeric_start is not None and time.monotonic()-numeric_start <= 300)), 'admission300_numeric300_total600'

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
        for path in (Path(__file__), PROTOCOL, Path(N.__file__), Path(X.__file__), Path(M.__file__),
                     Path(R.__file__), Path(R.C.__file__), Path(G.__file__), Path(R.B.__file__), Path(R.C.U.__file__)):
            M.committed(path)
            result['helper_sha256'][str(path)] = sha(path)
        M.committed(PARENT)
        assert sha(PARENT) == PARENT_SHA
        parent = json.loads(PARENT.read_text(encoding='utf-8'))
        assert all(parent['apparatus_gates'].values()) and parent['diagnostic_gates']['primary_native_identity_mean_KL_ge0_01']
        seen = dict(result['helper_sha256'])
        files = {}
        records = {}
        for name, expected in {'meth443_switch_native_causality_result.json': PARENT_SHA,
                'meth418_switch_function_capture_result.json': N.RECORDS['meth418_switch_function_capture_result.json'],
                'meth420_switch_function_gradient_result.failure.json': N.RECORDS['meth420_switch_function_gradient_result.failure.json']}.items():
            path = M.DOC/name
            M.committed(path)
            assert sha(path) == expected
            record = json.loads(path.read_text(encoding='utf-8'))
            records[name] = record
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
        prior = records['meth418_switch_function_capture_result.json']
        baseline = records['meth420_switch_function_gradient_result.failure.json']
        assert all(prior['gates'].values()) and len(baseline['baselines']) == 384
        for item in baseline['baselines']:
            assert sha(item['archive_path']) == item['archive_sha256']
        for path, expected in parent['preserved_original_binary_sha256'].items():
            assert sha(path) == expected
        result['retained_parent_sha256'] = PARENT_SHA
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
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
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
        ff = R.C.tensor(mapped, entries, 'decoder.block.11.layer.2.layer_norm.weight')
        fn = R.C.tensor(mapped, entries, 'decoder.final_layer_norm.weight')
        head = G.I8Operator(R.C.tensor(mapped, entries, 'lm_head.weight'), R.C.tensor(mapped, entries, 'lm_head.weight', 'scales'))
        cache = X.ExpertCache(mapped, entries)
        caps = {a['label']: a for a in prior['captures']}
        old = {a['label']: a for a in baseline['baselines']}
        anchors, scores, original, keys = [], [], [], []
        for book in range(24):
            case, position = book % 4, (5*book) % 14
            item = caps[f'teacher.n128.book{book}.case{case}']
            previous = old[item['label']]
            assert item['prospective_split'] == ('development' if book < 18 else 'validation')
            assert previous['source_capture_sha256'] == item['capture_sha256'] and not previous['qualification_only_not_paired_training']
            data = Path(item['capture_path']).read_bytes()
            assert data[:8] == b'SWFUN001' and struct.unpack_from('<6I', data, 8) == (768, 3072, 32128, 128, 11, 1)
            assert len(data) == 32+14*R.C.DTYPE.itemsize
            rows = np.frombuffer(data, dtype=R.C.DTYPE, offset=32)
            assert np.array_equal(rows['position'], np.arange(14)) and np.array_equal(rows['id'], item['decoder_ids'])
            with np.load(previous['archive_path'], allow_pickle=False) as archive:
                assert np.array_equal(archive['source_ids'], item['source_ids']) and np.array_equal(archive['decoder_ids'], item['decoder_ids'])
                for field in ('input', 'up_raw', 'up', 'down', 'probability', 'post', 'final', 'head_input'):
                    R.C.exact(archive[field], rows[field])
                original.append(archive['logits'][position].copy())
            trace = Path(item['trace_path']).read_bytes()
            dt = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (128,))])
            assert trace[:8] == b'SWRTA001' and struct.unpack_from('<2I', trace, 8) == (128, 768)
            assert len(trace) == 16+6*(29+14)*dt.itemsize
            trace_row = np.frombuffer(trace, dtype=dt, offset=16)[179+6*position]
            R.C.exact(trace_row['input'], rows[position]['input'])
            assert int(np.argmax(trace_row['scores'])) == int(rows[position]['expert'])
            scores.append(trace_row['scores'].copy())
            anchors.append(rows[position].copy())
            key = f'book{book}.case{case}.position{position}:{item["pairing_sha256"]}'
            if book >= 18:
                assert key == parent['data']['keys'][(book-18)*56+case*14+position]
            keys.append(key)
        anchors, scores, original = np.asarray(anchors), np.asarray(scores), np.asarray(original)
        assert anchors.shape == (24,) and original.shape == (24, 32128)
        numeric_start = time.monotonic()
        assert numeric_start-start <= 300, 'admission300_seconds'
        result['admission'] = {'seconds': numeric_start-start, 'bytes_hashed': hashed}
        guard()
        OUT.mkdir(parents=True)
        print(json.dumps({'admission_complete': result['admission']}), flush=True)
        with threadpool_limits(limits=1), torch.no_grad():
            result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_affinity': [0], 'BLAS': threadpool_info(), 'GPU': False}

            def logits_at(i, down):
                post = anchors[i]['pre']+anchors[i]['probability']*down
                final = G.NativeRMS.apply(torch.from_numpy(post), fn).numpy()
                return head.native(final*np.float32(1/np.sqrt(768)))

            stage = 'all24_original_native_replay'
            for i, row in enumerate(anchors):
                inp = G.NativeRMS.apply(torch.from_numpy(row['pre'].copy()), ff).numpy()
                R.C.exact(inp, row['input'])
                wi, wo = cache.get(int(row['expert']))
                raw = wi.native(inp)
                up = np.where(raw < 0, np.float32(0), raw)
                down = wo.native(up)
                p = G.NativeProbability.apply(torch.from_numpy(scores[i]), int(row['expert'])).numpy()
                post = row['pre']+p*down
                final = G.NativeRMS.apply(torch.from_numpy(post), fn).numpy()
                hi = final*np.float32(1/np.sqrt(768))
                for field, value in (('up_raw', raw), ('up', up), ('down', down), ('probability', p), ('post', post), ('final', final), ('head_input', hi)):
                    R.C.exact(value, row[field])
                for prefix, value in (('wi', inp), ('wo', up), ('head', hi)):
                    scale, codes = R.C.quant(value)
                    R.C.exact(scale, row[prefix+'_scale'])
                    R.C.exact(codes, row[prefix+'_codes'])
                R.C.exact(head.native(hi), original[i])
                guard()
            stage = 'all128_functions_all24_anchors'
            features = np.empty((24, 128, 768), np.float32)
            for expert in range(128):
                wi, wo = cache.get(expert)
                prefix = f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.'
                for i, row in enumerate(anchors):
                    raw = wi.native(row['input'])
                    up = np.where(raw < 0, np.float32(0), raw)
                    down = wo.native(up)
                    features[i, expert] = down
                    if i == 0:
                        scale, codes = R.C.quant(row['input'])
                        R.C.exact(R.C.matrix(mapped, entries, prefix+'wi.weight', codes, scale), raw)
                        scale, codes = R.C.quant(up)
                        R.C.exact(R.C.matrix(mapped, entries, prefix+'wo.weight', codes, scale), down)
                    if expert == int(row['expert']):
                        R.C.exact(raw, row['up_raw'])
                        R.C.exact(up, row['up'])
                        R.C.exact(down, row['down'])
                    guard()
            assert np.isfinite(features).all()
            means = np.mean(features.astype(np.float64), axis=1)
            contrasts = features.astype(np.float64)-means[:, None, :]
            total_energy = np.sum(features.astype(np.float64)**2, axis=(1, 2))
            private_energy = np.sum(contrasts**2, axis=(1, 2))
            common_energy = 128*np.sum(means**2, axis=1)
            assert np.all(private_energy > 0) and np.max(np.abs(total_energy-common_energy-private_energy)/total_energy) <= 1e-10
            stage = 'development_only_output_covariance'
            development = contrasts[:18].reshape(18*128, 768)
            covariance = development.T @ development/len(development)
            values, basis = np.linalg.eigh(covariance)
            values, basis = values[::-1], basis[:, ::-1].copy()
            for j in range(768):
                pivot = int(np.argmax(np.abs(basis[:, j])))
                if basis[pivot, j] < 0:
                    basis[:, j] *= -1
            assert np.min(values) >= -1e-10*max(float(values[0]), 1)
            assert R.relative(basis.T @ basis, np.eye(768)) <= 1e-10
            assert R.relative(covariance @ basis, basis*values) <= 1e-10
            assert abs(np.trace(covariance)-np.sum(values)) <= 1e-10*max(float(np.trace(covariance)), 1)
            complete = means[:, None, :]+(contrasts @ basis) @ basis.T
            full_error = R.relative(complete, features)
            assert full_error <= 1e-10
            np.savez(OUT/'geometry.npz', anchors=anchors, scores=scores, original_logits=original, keys=np.asarray(keys),
                     features=features, means=means, contrasts=contrasts, covariance=covariance, eigenvalues=values, basis=basis,
                     total_energy=total_energy, common_energy=common_energy, private_energy=private_energy)
            result['data'] = {'keys': keys, 'books': list(range(24)), 'cases': [b % 4 for b in range(24)],
                'positions': [(5*b) % 14 for b in range(24)], 'development_anchors': 18, 'validation_anchors': 6,
                'functions_per_anchor': 128, 'selected_ids': anchors['expert'].tolist(), 'selected_probabilities': anchors['probability'].tolist()}
            result['geometry'] = {'per_anchor_private_fraction': (private_energy/total_energy).tolist(),
                'per_anchor_common_fraction': (common_energy/total_energy).tolist(), 'eigenvalues': values.tolist(),
                'full_F64_recomposition_relative_error': full_error, 'best_development_linear_tail_bound_only': True}
            guard()
            stage = 'native_validation_all_functions_full_heads'
            native = np.empty((6, 128, 32128), np.float32)
            for a in range(6):
                i = a+18
                for expert in range(128):
                    native[a, expert] = logits_at(i, features[i, expert])
                    guard()
                R.C.exact(native[a, int(anchors[i]['expert'])], original[i])
            np.save(OUT/'validation_native_logits.npy', native)
            native_js = [information(z)[2] for z in native]
            result['native_reference'] = {'per_anchor_uniform_interventional_JS_nats': native_js, 'mean_JS_nats': float(np.mean(native_js)),
                'full_rank_native_control': 'original_features_passthrough_byte_exact_original_selected_heads; F64_basis_recomposition_qualified_separately'}
            for rank in RANKS:
                stage = f'fixed_rank_{rank}'
                coefficients = contrasts @ basis[:, :rank]
                restored_private = coefficients @ basis[:, :rank].T
                residual = contrasts-restored_private
                error_energy = np.sum(residual**2, axis=(1, 2))
                retained = 1-error_energy/private_energy
                alternate = private_energy-np.sum(coefficients**2, axis=(1, 2))
                assert np.max(np.abs(error_energy-alternate)/private_energy) <= 1e-10
                predicted = (means[:, None, :]+restored_private).astype(np.float32)
                out = np.empty(native.shape, np.float32)
                paired_kl = np.empty((6, 128), np.float64)
                independent_kl = np.empty_like(paired_kl)
                js_values = []
                changed = np.zeros((6, 128), bool)
                for a in range(6):
                    i = a+18
                    if rank == 0:
                        out[a] = logits_at(i, predicted[i, 0])[None, :]
                    else:
                        for expert in range(128):
                            out[a, expert] = logits_at(i, predicted[i, expert])
                            guard()
                    q, lp, js = information(native[a])
                    js_values.append(information(out[a])[2])
                    for expert in range(128):
                        ce0 = R.numpy_loss(native[a, expert].astype(np.float64), q[expert])
                        ce1 = R.numpy_loss(out[a, expert].astype(np.float64), q[expert])
                        assert abs(ce0+np.dot(q[expert], lp[expert])) <= 1e-10
                        paired_kl[a, expert] = ce1-ce0
                        independent_kl[a, expert] = np.dot(q[expert], lp[expert]-N.log_probability(out[a, expert]))
                    changed[a] = np.argmax(out[a], axis=1) != np.argmax(native[a], axis=1)
                    guard()
                assert np.isfinite(paired_kl).all() and np.min(paired_kl) >= -1e-10
                assert np.max(np.abs(paired_kl-independent_kl)) <= 1e-10
                if rank == 0:
                    assert np.max(np.abs(js_values)) <= 1e-10
                np.save(OUT/f'validation_rank{rank}_logits.npy', out)
                weighted_dev = 1-np.sum(error_energy[:18])/np.sum(private_energy[:18])
                weighted_val = 1-np.sum(error_energy[18:])/np.sum(private_energy[18:])
                gates = {'dev_private_energy_ge0_95': bool(weighted_dev >= .95), 'val_private_energy_ge0_95': bool(weighted_val >= .95),
                    'all_function_mean_native_KL_le0_01': bool(np.mean(paired_kl) <= .01),
                    'every_anchor_mean_native_KL_le0_05': bool(np.max(np.mean(paired_kl, axis=1)) <= .05)}
                result['ranks'][str(rank)] = {'weighted_development_private_energy_retained': float(weighted_dev),
                    'weighted_validation_private_energy_retained': float(weighted_val), 'per_anchor_private_energy_retained': retained.tolist(),
                    'mean_native_KL': float(np.mean(paired_kl)), 'per_anchor_mean_KL': np.mean(paired_kl, axis=1).tolist(),
                    'per_anchor_per_function_KL': paired_kl.tolist(), 'independent_KL': independent_kl.tolist(),
                    'argmax_changed_mask': changed.tolist(), 'argmax_changes': int(np.sum(changed)),
                    'selected_original_function_KL': [float(paired_kl[a, int(anchors[a+18]['expert'])]) for a in range(6)],
                    'per_anchor_uniform_interventional_JS_nats': js_values, 'mean_JS_nats': float(np.mean(js_values)),
                    'gates': gates, 'all_geometry_gates': all(gates.values())}
                print(json.dumps({'rank': rank, 'development_energy': weighted_dev, 'validation_energy': weighted_val,
                                  'mean_native_KL': float(np.mean(paired_kl)), 'gates': gates}), flush=True)
                del out
                guard()
            result['apparatus_gates'] = {'fresh_sources_parent_output_inventories': True, 'all24_original_states_quant_full_heads_byte_exact': True,
                'all128_first_anchor_WI_WO_independent_primitive_byte_exact': True, 'all3072_functions_native_fixed_input': True,
                'mean_private_Pythagoras_and_dev_eigensystem_full_recomposition': True, 'basis_fit_only18dev_fixed_anchors': True,
                'all_counterfactual_KL_two_forms_entropy_same1e_minus10': True, 'rank0_zero_interventional_information': True}
        assert initial == (Path(artifact['payload']).stat().st_size, Path(artifact['payload']).stat().st_mtime_ns)
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource'] = {'seconds': time.monotonic()-start, 'admission_seconds': numeric_start-start,
            'numeric_and_reporting_seconds': time.monotonic()-numeric_start, 'peak_bytes': peak, 'bytes_hashed': hashed,
            'output_bytes': sum(v['bytes'] for v in result['output_inventory']), 'optimizer_updates': 0}
        passing = [r for r in RANKS if r and result['ranks'][str(r)]['all_geometry_gates']]
        result['decision'] = {'passing_fixed_ranks': passing, 'smallest_passing_fixed_rank': min(passing) if passing else None,
            'next': 'formulate_available_common_response_and_private_coefficient_cost_before_fit' if passing else
                    'fixed_dev_covariance_output_subspaces_inadequate_under_oracle_mean_reconsider_representation_before_fit'}
        result['scope'] = 'One source128 bank11;24 fixed consumed anchors/128 uniformly intervened functions, basis18dev/test6val. Oracle mean and projection coefficients require full functions offline. Optimal Frobenius dev subspace only; no global/nonlinear/downstream-optimal rank impossibility. JS is artificial uniform do(E) information, not deterministic observed routing I(E;Y|X), truth quality or useful n. No runtime compression, available-input selector, complete quality/rate/LUT/physicalDRAM/family/~100B proof. No440 rescue.'
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
