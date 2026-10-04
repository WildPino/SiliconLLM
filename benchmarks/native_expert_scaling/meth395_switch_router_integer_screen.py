"""Full-dimensional I8/A16 router plus exact candidate refinement screen."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import threadpoolctl
from threadpoolctl import threadpool_info, threadpool_limits
import meth324_switch_reference as M
import meth368_switch_bank_manifest as B

PROTOCOL = M.DOC / 'METH_395_SWITCH_ROUTER_INTEGER_SCREEN_PROTOCOL_20261004.md'
CAPTURE = M.DOC / 'meth393_switch_router_audit_result.json'
CAPTURE_SHA = 'bea3d2e1db10f03d5d9de3173d71a455142166ae71229bbb847959baf12ce07d'
EXPORT = {
    256: ('meth338_switch_tensor_recovery_result.json', '19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'),
    128: ('meth380_switch_base128_export_result.json', '6be77281f7ecca73c47eab7f318fed2cb27699f18071b3f3781ab9bce54881ee'),
}
OUT = M.ROOT / 'results/native_expert_scaling/meth395_switch_router_integer_screen'
PRIOR = M.DOC / 'meth394_switch_router_rank_screen_result.json'
PRIOR_SHA = 'b8719b4d45bf0b814499111084de5daadea5bdb8cc8990bb044735f1b47ffe11'
SHORTLISTS = (0, 1, 4, 8, 16)


def metric(probability, selected, original_probability, original_selected, n, shortlist):
    relative = np.abs(probability / original_probability - 1.)
    mismatches = int(np.sum(selected != original_selected))
    recall = 1. - mismatches / len(selected)
    weight_bytes = n * 768 + 4 * n + 4 * 768 * shortlist
    byte_ratio = weight_bytes / (4 * 768 * n)
    p95 = float(np.quantile(relative, .95)); maximum = float(np.max(relative))
    gates = {'identity_agreement_ge0p9999': recall >= .9999, 'probability_relative_p95_le0p01': p95 <= .01,
             'probability_relative_max_le0p05': maximum <= .05, 'addressed_weight_bytes_proxy_ratio_le0p5': byte_ratio <= .5}
    return {'queries': len(selected), 'selected_identity_mismatches': mismatches, 'identity_agreement': recall,
            'absolute_relative_probability_quantiles': np.quantile(relative, [0., .05, .5, .95, 1.]).tolist(),
            'addressed_coefficient_ratio': 1. + shortlist / n, 'addressed_weight_bytes_proxy_ratio': byte_ratio, 'addressed_weight_bytes_proxy': weight_bytes,
            'gates': gates, 'passed': all(gates.values())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); maximum = 0; stage = 'bindings'
    result = {'experiment': 'METH-395-full-dimensional-router-I8-A16-exact-refinement-screen', 'targets': []}

    def guard():
        nonlocal maximum
        maximum = max(maximum, psutil.Process().memory_info().rss)
        assert maximum <= 8 << 30 and time.monotonic() - start <= 1200, 'integer_screen_20min_8GiB'

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20): h.update(block); guard()
        return h.hexdigest()

    try:
        for path in (Path(__file__), PROTOCOL, CAPTURE, Path(M.__file__), Path(B.__file__), PRIOR): M.committed(path)
        assert digest(CAPTURE) == CAPTURE_SHA and digest(PRIOR) == PRIOR_SHA
        capture = json.loads(CAPTURE.read_text(encoding='utf-8')); assert all(capture['gates'].values())
        assert np.__version__ == '2.4.6'
        OUT.mkdir(parents=True)
        result.update({'controller_sha256': digest(__file__), 'protocol_sha256': digest(PROTOCOL), 'capture393_sha256': CAPTURE_SHA, 'prior394_sha256': PRIOR_SHA,
                       'numpy_version': np.__version__, 'threadpoolctl_version': threadpoolctl.__version__, 'threadpoolctl_source_sha256': digest(threadpoolctl.__file__), 'manifest_parser_sha256': digest(B.__file__)})
        with threadpool_limits(limits=1, user_api='blas'):
            result['blas_runtime'] = [{**v, 'library_sha256': digest(v['filepath'])} for v in threadpool_info()]
            assert result['blas_runtime'] and all(v['num_threads'] == 1 for v in result['blas_runtime'] if v['user_api'] == 'blas')
            for target in capture['targets']:
                n = target['n']; stage = f'source{n}'
                filename, expected = EXPORT[n]; path = M.DOC / filename
                M.committed(path); assert digest(path) == expected
                export = json.loads(path.read_text(encoding='utf-8')); assert all(export['gates'].values())
                artifact = export['artifact']; assert artifact == target['artifact']
                payload = Path(artifact['payload']); before = (payload.stat().st_size, payload.stat().st_mtime_ns)
                assert before[0] == artifact['bytes']
                assert digest(artifact['manifest']) == artifact['manifest_sha256']
                B.read_manifest(artifact['manifest'], export['original_config'], export['tensors'], payload)
                names = sorted(k for k in export['tensors'] if '.router.classifier.weight' in k)
                assert len(names) == 12
                arrays = defaultdict(list); source_files = []
                for entry in target['quality_bridges']:
                    s = len(entry['source_ids'])
                    for mode, rows in (('teacher', entry['teacher_rows']), ('natural', entry['generation_rows'])):
                        assert len(rows) == 1
                        row = rows[0]; path = Path(row['router_trace_path']); assert digest(path) == row['router_trace_sha256']
                        data = path.read_bytes(); assert data[:8] == b'SWRTA001' and struct.unpack_from('<2I', data, 8) == (n, 768)
                        dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (n,))])
                        t = len(entry['decoder_ids']) if mode == 'teacher' else len(row['generated_ids'])
                        count = 6 * (s + t); assert len(data) == 16 + count * dtype.itemsize
                        trace = np.frombuffer(data, dtype=dtype, offset=16)
                        assert np.array_equal(trace['index'], np.arange(count)) and np.all(trace['phase'][:6*s] == 0) and np.all(trace['phase'][6*s:] == 1)
                        assert np.all(np.isfinite(trace['input'])) and np.all(np.isfinite(trace['scores']))
                        source_files.append({'mode': mode, 'book': entry['book'], 'case': entry['case'], 'path': str(path), 'sha256': row['router_trace_sha256'], 'routes': count})
                        for bank in range(12):
                            selected = trace[bank*s:(bank+1)*s] if bank < 6 else trace[6*s+bank-6::6]
                            arrays[(bank, mode)].append((selected['input'].copy(), selected['scores'].copy()))
                        del data, trace; guard()
                observed = {'n': n, 'export_sha256': expected, 'source_traces': source_files, 'banks': [], 'candidates': []}
                result['targets'].append(observed); decisions = defaultdict(list)
                for bank in range(12):
                    stack = 'encoder' if bank < 6 else 'decoder'; layer = 2 * (bank % 6) + 1
                    name = f'{stack}.block.{layer}.layer.{1 if stack == "encoder" else 2}.mlp.router.classifier.weight'
                    item = export['tensors'][name]; assert item['encoding'] == 0 and item['shape'] == [n, 768] and item['bytes'] == n*768*4
                    with payload.open('rb') as stream: stream.seek(item['offset']); raw = stream.read(item['bytes'])
                    assert hashlib.sha256(raw).hexdigest() == item['sha256'] == item['source_sha256']
                    w = np.frombuffer(raw, dtype='<f4').reshape(n, 768).astype(np.float64)
                    guard()
                    baseline = {}; samples = {}
                    for mode in ('teacher', 'natural'):
                        parts = arrays.pop((bank, mode)); x = np.concatenate([v[0] for v in parts]).astype(np.float64); scores = np.concatenate([v[1] for v in parts])
                        independent = (x @ w.T).astype(np.float32)
                        error = float(np.linalg.norm(independent.astype(np.float64) - scores) / np.linalg.norm(scores.astype(np.float64)))
                        assert error <= 1e-6 and np.array_equal(np.argmax(independent, axis=1), np.argmax(scores, axis=1))
                        original_selected = np.argmax(scores, axis=1)
                        differences = scores - scores[np.arange(len(scores)), original_selected, None]
                        probability = (1. / np.exp(differences.astype(np.float64)).astype(np.float32).astype(np.float64).sum(axis=1)).astype(np.float32).astype(np.float64)
                        baseline[mode] = {'queries': len(x), 'independent_full_weight_score_relative_l2': error, 'selected_ids_exact': True}
                        samples[mode] = (x, scores, original_selected, probability)
                    maximum_weight = np.max(np.abs(w.astype(np.float32)), axis=1)
                    scales = np.where(maximum_weight == 0., np.float32(1.), maximum_weight / np.float32(127.)).astype(np.float32)
                    assert np.all(np.isfinite(scales)) and np.all(scales > 0.)
                    codes = np.clip(np.rint(w.astype(np.float32) / scales[:, None]), -127, 127).astype(np.int8)
                    saved = OUT / f'n{n}.bank{bank}.router_i8.npz'; np.savez(saved, codes=codes, scales=scales)
                    with np.load(saved) as check: assert np.array_equal(check['codes'], codes) and np.array_equal(check['scales'], scales)
                    bank_record = {'bank': bank, 'name': name, 'actual_router_weight_sha256': item['sha256'], 'baseline': baseline,
                                   'I8_codes_bytes': codes.nbytes, 'F32_scales_bytes': scales.nbytes, 'retained_full_router_F32_bytes': n*768*4,
                                   'representation_path': str(saved), 'representation_sha256': digest(saved), 'modes': {}}
                    observed['banks'].append(bank_record)
                    for mode, (x, scores, original_selected, probability) in samples.items():
                        input_maximum = np.max(np.abs(x.astype(np.float32)), axis=1)
                        input_scales = np.where(input_maximum == 0., np.float32(1.), input_maximum / np.float32(32767.)).astype(np.float32)
                        assert np.all(np.isfinite(input_scales)) and np.all(input_scales > 0.)
                        input_codes = np.clip(np.rint(x.astype(np.float32) / input_scales[:, None]), -32767, 32767).astype(np.int16)
                        # Integer-valued F64 product/sums are exact: absolute sum
                        # <=768*127*32767<2**32<2**53, independently spot-checked.
                        integer = input_codes.astype(np.float64) @ codes.astype(np.float64).T
                        assert np.all(np.isfinite(integer)) and np.array_equal(integer, np.rint(integer))
                        assert np.max(np.abs(integer)) <= 768*127*32767
                        checks = 0
                        for query in sorted(set([0, len(input_codes)//2, len(input_codes)-1])):
                            for row in sorted(set([0, n//3, 2*n//3, n-1])):
                                exact = sum(int(a)*int(b) for a,b in zip(input_codes[query], codes[row]))
                                assert integer[query,row] == exact; checks += 1
                        approximate = ((integer * scales.astype(np.float64)[None,:]) * input_scales.astype(np.float64)[:,None]).astype(np.float32)
                        assert np.all(np.isfinite(approximate))
                        order = np.argsort(-approximate, axis=1, kind='stable'); values = []
                        for shortlist in SHORTLISTS:
                            refined = approximate.copy()
                            if shortlist:
                                indices = order[:, :shortlist]; exact = np.take_along_axis(scores, indices, axis=1)
                                chosen = np.where(exact == np.max(exact, axis=1)[:,None], indices, n).min(axis=1)
                                np.put_along_axis(refined, indices, exact, axis=1)
                            else: chosen = np.argmax(refined, axis=1)
                            center = np.max(refined, axis=1); differences = refined - center[:,None]
                            denominator = np.exp(differences.astype(np.float64)).astype(np.float32).astype(np.float64).sum(axis=1)
                            chosen_score = scores[np.arange(len(scores)), chosen] if shortlist else refined[np.arange(len(scores)), chosen]
                            numerator = np.exp((chosen_score-center).astype(np.float64)).astype(np.float32).astype(np.float64)
                            estimated_probability = (numerator/denominator).astype(np.float32).astype(np.float64)
                            assert np.all(np.isfinite(estimated_probability))
                            value = {'shortlist': shortlist, **metric(estimated_probability, chosen, probability, original_selected, n, shortlist)}
                            values.append(value); decisions[shortlist].append(value['passed'])
                        bank_record['modes'][mode] = {'independent_integer_dot_controls': checks, 'input_A16_absolute_codes_max': int(np.max(np.abs(input_codes.astype(np.int32)))), 'candidates': values}
                        guard()
                    print(json.dumps({'n': n, 'bank': bank, 'completed_shortlists': len(SHORTLISTS), 'seconds': time.monotonic()-start}), flush=True)
                assert not arrays
                for shortlist, values in sorted(decisions.items()):
                    assert len(values) == 24
                    observed['candidates'].append({'shortlist': shortlist, 'passed_all12_banks_both_modes': all(values), 'passing_bank_modes': sum(values), 'bank_modes': len(values)})
                observed['eligible_candidates'] = [v for v in observed['candidates'] if v['passed_all12_banks_both_modes']]
                assert before == (payload.stat().st_size, payload.stat().st_mtime_ns)
        result['gates'] = {'all384_trace_identities_and_shapes': True, 'all24_actual_router_segments_and_manifest_verified': True,
                           'independent_full_weight_scores_selected_id_control': True, 'all_I8_code_F32_scale_roundtrips_exact': True, 'independent_integer_dot_controls_exact': True}
        result['decision'] = 'eligible_only_for_new_native_numeric_and_NEW_whole_quality_protocol' if any(t['eligible_candidates'] for t in result['targets']) else 'reject_full_dimension_I8_A16_router_refinement_grid'
        result['resource'] = {'main_seconds': time.monotonic()-start, 'maximum_checked_rss_bytes': maximum, 'representation_artifact_bytes': sum(p.stat().st_size for p in OUT.iterdir())}
        result['scope'] = 'Source row-I8 weights/A16 inputs, full dimensions/all candidates, original-F32 candidate refinement with full mixed normalization. Existing consumed cohort diagnostic identity/probability gates, weight-byte and coefficient proxies only; no native timing/whole-model quality/useful>256/LUT/physical DRAM/cross-family claim. Full original routers retained, entire payload hashes inherited393; actual source router segments freshly hashed.'
        guard(); M.write(args.out, result)
        print(json.dumps({'sha256': digest(args.out), 'decision': result['decision'], 'eligible': {t['n']: t['eligible_candidates'] for t in result['targets']}, 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start}); M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
