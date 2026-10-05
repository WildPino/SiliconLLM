"""Frozen retained-trace expert-owned INPUT data admission; no model/fit/SVD."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth462_switch_query_domain_admission'
RAW = DOC / 'meth462_switch_query_domain_admission_result.json'
PROTOCOL = DOC / 'METH_462_SWITCH_QUERY_DOMAIN_ADMISSION_PROTOCOL_20261005.md'
DIRECT = {
    'meth393_switch_router_audit_result.json': 'bea3d2e1db10f03d5d9de3173d71a455142166ae71229bbb847959baf12ce07d',
    'meth461_switch_common_input_cost_result.json': '1477dcab26918bc11d82a081ee7cc180a6d09c0c022848935f205ed9ff0d1f57',
    'RETENTION_461_20261005.json': 'ee4965b80778a7df8caedbf8a965156be4d55af9e22d819f3dbcb81bd19069d6',
}
LEDGER = struct.Struct('<H6BHIff32s32s32s')
SAMPLE_COORDS = (0, 1, 2, 3, 17, 31, 255, 511, 767)


def committed(path):
    path = Path(path); rel = path.resolve().relative_to(ROOT).as_posix()
    assert path.read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', f'--path={rel}', f'HEAD:{rel}'], cwd=ROOT), rel


def jobs():
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}; preserved = []
    for process in psutil.process_iter(['name', 'cmdline']):
        if process.pid in own:
            continue
        name, argv = (process.info['name'] or '').lower(), process.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            preserved.append(process.pid); continue
        assert not (name.startswith('python') or (name.startswith('meth') and name.endswith('.exe'))), (process.pid, name)
    return preserved


def route_bytes(data, n, source, positions):
    assert len(data) >= 36 and data[:8] == b'SWR32O01'
    s, t, d, enc, dec, vocab, count = struct.unpack_from('<7I', data, 8)
    assert (s, t, d, enc, dec, vocab) == (source, positions, 768, 12, 12, 32128)
    assert count == 6 * (s + t)
    offset = 36 + 4 * ((enc + 2) * s * d + t * (dec + 2) * d + t * vocab)
    assert len(data) == offset + 12 * count
    rows = np.frombuffer(data, dtype=[('expert', '<i4'), ('accepted', '<i4'), ('probability', '<f4')], offset=offset)
    assert len(rows) == count and np.all((rows['expert'] >= 0) & (rows['expert'] < n))
    assert np.all((rows['accepted'] == 0) | (rows['accepted'] == 1))
    p = rows['probability'].astype(np.float64)
    assert np.all(np.isfinite(p)) and np.all((p >= (1. / n) * (1. - 2e-6)) & (p <= 1.))
    return rows[:6 * s].reshape(6, s), rows[6 * s:].reshape(t, 6).T


def quantize(x):
    assert x.dtype == np.float32 and x.ndim == 2 and np.all(np.isfinite(x))
    maximum = np.max(np.abs(x), axis=1)
    with np.errstate(under='ignore', over='raise', invalid='raise', divide='raise'):
        scale = np.divide(maximum, np.float32(32767), dtype=np.float32)
    scale[maximum == 0] = np.float32(1)
    assert np.all(np.isfinite(scale)) and np.all(scale > 0)
    divided = np.divide(x, scale[:, None], dtype=np.float32)
    codes = np.clip(np.rint(divided), -32767, 32767).astype('<i2')
    return codes, scale


def exact_positive_f32(value):
    """Exact-rational nearest binary32, ties-even; adjacent candidates are checked."""
    assert value >= 0
    if value == 0:
        return 0.
    bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
    candidates = []
    for candidate in (bits - 1, bits, bits + 1):
        if 0 <= candidate <= 0x7f7fffff:
            f = struct.unpack('<f', struct.pack('<I', candidate))[0]
            candidates.append((abs(Fraction.from_float(f) - value), candidate & 1, f))
    return min(candidates)[2]


def scalar_quantize(values, coordinates=None):
    maximum = max(abs(float(value)) for value in values)
    scale = exact_positive_f32(Fraction.from_float(maximum) / 32767) if maximum else 1.
    codes = []
    for index in range(len(values)) if coordinates is None else coordinates:
        value = float(values[index]); positive = exact_positive_f32(Fraction.from_float(abs(value)) / Fraction.from_float(scale))
        code = round(positive) * (-1 if value < 0 else 1)
        codes.append(max(-32767, min(32767, code)))
    return codes, scale


def bucket():
    return {'selected': 0, 'executed': 0, 'rejected': 0, 'probability_sum': 0., 'books': set(), 'cases': set(),
            'input': set(), 'code': {}, 'pair': set(), 'scale': set()}


def code_union(buckets):
    answer = {}
    for b in buckets:
        for key, mask in b['code'].items():
            answer[key] = answer.get(key, 0) | mask
    return answer


def summary(buckets):
    codes = code_union(buckets)
    return {'selected': sum(b['selected'] for b in buckets), 'executed': sum(b['executed'] for b in buckets),
            'rejected': sum(b['rejected'] for b in buckets), 'executed_router_probability_sum': sum(b['probability_sum'] for b in buckets),
            'books': sorted(set.union(*(b['books'] for b in buckets))),
            'cases': len(set.union(*(b['cases'] for b in buckets))),
            'unique_input_SHA': len(set.union(*(b['input'] for b in buckets))), 'unique_code_SHA': len(codes),
            'unique_code_scale_SHA': len(set.union(*(b['pair'] for b in buckets))),
            'unique_scale_bytes': len(set.union(*(b['scale'] for b in buckets)))}


def write_new(path, obj):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode('utf-8'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert args.out.resolve() == RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and not OUT.exists()
    assert os.name == 'nt' and sys.flags.optimize == 0
    start = time.monotonic(); phase_start = None; peak = hashed = count = 0; stage = 'bindings'
    result = {'experiment': 'METH462-native-expert-owned-query-data-admission', 'helper_sha256': {}, 'retained_record_sha256': {},
              'reused_file_inventory': [], 'sources': {}, 'trace_checks': [], 'native_or_model_commands': 0}
    def guard():
        nonlocal peak
        info = psutil.Process().memory_info(); peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
        assert peak <= 512 << 20 and time.monotonic() - start <= 300, '300s512MiB'
        assert phase_start is not None or time.monotonic() - start <= 120, 'admission120s'
        assert not OUT.exists() or sum(p.stat().st_size for p in OUT.iterdir()) <= 32 << 20, 'output32MiB'
    def digest(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while data := stream.read(4 << 20):
                h.update(data); hashed += len(data); guard()
        return h.hexdigest()
    def reused(path, expected):
        assert digest(path) == expected
        result['reused_file_inventory'].append({'path': str(Path(path)), 'bytes': Path(path).stat().st_size, 'sha256': expected})
    try:
        result['git_head_before_execution'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip()
        assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'], cwd=ROOT)
        for path in (Path(__file__), PROTOCOL):
            committed(path); result['helper_sha256'][str(path)] = digest(path)
        for name, expected in DIRECT.items():
            committed(DOC / name); assert digest(DOC / name) == expected; result['retained_record_sha256'][name] = expected
        original = json.loads((DOC / 'meth461_switch_common_input_cost_result.json').read_text(encoding='utf-8'))
        assert all(original['apparatus_gates'].values()) and all(json.loads((DOC / 'RETENTION_461_20261005.json').read_text(encoding='utf-8'))['gates'].values())
        original_sources = {n: {key: source[key] for key in ('artifact', 'artifact_stat_before')} for n, source in original['sources'].items()}
        for path, expected in original['helper_sha256'].items():
            committed(path); assert digest(path) == expected; result['helper_sha256'][path] = expected
        engine = ROOT / 'benchmarks/phase60/engine.c'; committed(engine); assert digest(engine) == original['preserved_engine_sha256']; result['engine_sha256'] = original['preserved_engine_sha256']
        del original
        capture = json.loads((DOC / 'meth393_switch_router_audit_result.json').read_text(encoding='utf-8')); assert all(capture['gates'].values())
        for name, field in (('meth393_switch_router_audit.py', 'controller_sha256'), ('meth393_switch_router_trace.h', 'trace_helper_sha256'), ('meth393_switch_router_analysis.py', 'analysis_helper_sha256')):
            path = BASE / name; committed(path); assert digest(path) == capture[field]; result['helper_sha256'][str(path)] = capture[field]
        path = DOC / 'METH_393_SWITCH_ROUTER_AUDIT_PROTOCOL_20261004.md'; committed(path); assert digest(path) == capture['protocol_sha256']; result['helper_sha256'][str(path)] = capture['protocol_sha256']
        records = {}
        names = {389: 'meth389_switch_three_workers_contract_result.json', 356: 'meth356_switch_all_a16_contract_result.json',
                 362: 'meth362_switch_multi_span_manifest.json', 363: 'meth363_switch_all_a16_multi_span_quality_result.json',
                 374: 'meth374_switch_physical_workers_contract_result.json', 381: 'meth381_switch_base128_contract_result.json',
                 382: 'meth382_switch_multi_span_manifest.json', 387: 'meth387_switch_base128_multi_span_quality_result.json'}
        for number, expected in capture['input_sha256'].items():
            name = names[int(number)]; path = DOC / name; committed(path); assert digest(path) == expected
            result['retained_record_sha256'][name] = expected; records[int(number)] = json.loads(path.read_text(encoding='utf-8'))
        reader = BASE / 'meth392_switch_route_audit.py'; committed(reader); result['helper_sha256'][str(reader)] = digest(reader)
        old_function = reader.read_text(encoding='utf-8').split('def route_bytes(', 1)[1].split('\n\ndef summarize', 1)[0]
        own_function = Path(__file__).read_text(encoding='utf-8').split('def route_bytes(', 1)[1].split('\n\ndef quantize', 1)[0]
        assert old_function == own_function, 'route_reader_exact392_without_model_import'
        for target in capture['targets']:
            n = str(target['n']); artifact = target['artifact']; assert artifact == original_sources[n]['artifact']
            payload = Path(artifact['payload']); assert [payload.stat().st_size, payload.stat().st_mtime_ns] == original_sources[n]['artifact_stat_before']
            assert digest(artifact['manifest']) == artifact['manifest_sha256']
        result['preserved_daemons_before'] = jobs(); guard(); phase_start = time.monotonic(); OUT.mkdir(parents=True)
        stage = 'cached_native_and_exact_rational_A16_controls'
        fixture = ROOT / 'results/native_expert_scaling/meth356_switch_all_a16_contract/integer_cases.bin'
        answer = ROOT / 'results/native_expert_scaling/meth374_switch_physical_workers_contract/head-int-dot.bin'
        reused(fixture, 'e404e9a2501810532630e483450ce2bb3bcf7db7e5100d1a947b49a19ecdeb94')
        expected_answer = next(v['sha256'] for v in records[374]['integer_primitives'] if v['path'] == '--head-int-dot'); reused(answer, expected_answer)
        assert expected_answer == '0162a37a6a521c7ddd9c80097c193eaa0e7e85e39c056aa88bb5017f68559b87'
        data, output = fixture.read_bytes(), answer.read_bytes(); a = b = 12; controls = []
        assert data[:8] == b'SWI8D001' and output[:8] == b'SW16R001' and struct.unpack_from('<I', data, 8) == struct.unpack_from('<I', output, 8) == (13,)
        for ci in range(13):
            rows, cols = struct.unpack_from('<2I', data, a); a += 8
            w = np.frombuffer(data, '<i1', rows * cols, a).reshape(rows, cols); a += rows * cols
            scales = np.frombuffer(data, '<f4', rows, a); a += rows * 4
            x = np.frombuffer(data, '<f4', cols, a).reshape(1, cols); a += cols * 4
            assert struct.unpack_from('<2I', output, b) == (rows, cols); b += 8
            y = np.frombuffer(output, '<f4', rows, b); b += rows * 4
            q = np.frombuffer(output, '<i2', cols, b); b += cols * 2
            alpha = struct.unpack_from('<f', output, b)[0]; b += 4
            dots = np.frombuffer(output, '<i8', rows, b); b += rows * 8
            actual_q, actual_alpha = quantize(x); scalar_q, scalar_alpha = scalar_quantize(x[0])
            assert actual_q[0].tobytes() == q.tobytes() and actual_q[0].tolist() == scalar_q and float(actual_alpha[0]) == alpha == scalar_alpha
            actual_dot = w.astype(np.int64) @ actual_q[0].astype(np.int64)
            actual_y = ((actual_dot.astype(np.float64) * scales.astype(np.float64)) * alpha).astype('<f4')
            assert actual_dot.tobytes() == dots.tobytes() and actual_y.tobytes() == y.tobytes()
            controls.append({'case': ci, 'cols': cols, 'native_codes_scale_dot_y_and_rational_reference_exact': True}); guard()
        assert a == len(data) and b == len(output)
        for values in ([0.] * 17, [32767., .5, -.5, 1.5, -1.5, 2.5, -2.5], [.01, -.003, .005, .0007], [1e-20, -2e-20, 3e-20], [1e20, -2e20, 3e20], [2. ** -133, 2. ** -140, -2. ** -140]):
            x = np.asarray([values], dtype=np.float32); q, alpha = quantize(x); sq, sa = scalar_quantize(x[0]); assert q[0].tolist() == sq and float(alpha[0]) == sa
        result['A16_controls'] = {'cached_native_cases': controls, 'additional_exact_rational_shapes': 6, 'no_native_rerun': True}
        stage = 'retained_trace_domain_counts'; states = {n: [[bucket() for _ in range(4)] for _ in range(12 * int(n))] for n in ('128', '256')}
        ledger_path = OUT / 'query_fingerprint_ledger.bin'
        with ledger_path.open('xb') as ledger:
            ledger.write(struct.pack('<8sIQ', b'MQ462L01', LEDGER.size, 0))
            for n in ('128', '256'):
                target = next(v for v in capture['targets'] if str(v['n']) == n); cohort = records[382 if n == '128' else 362]
                assert len(target['quality_bridges']) == 96 and len(cohort['items']) == 24
                hashes = [item['whole_source_utf8_sha256'] for item in cohort['items']]; assert len(set(hashes)) == 24
                result['sources'][n] = {'artifact': target['artifact'], 'payload_stat': original_sources[n]['artifact_stat_before'], 'full_payload_SHA_refreshed_this_experiment': False,
                    'cohort_sha256': target['cohort_sha256'], 'books': [{'index': i, 'split': 'dev' if i < 12 else 'val', 'source_id': item['source_id'], 'whole_source_utf8_sha256': hashes[i]} for i, item in enumerate(cohort['items'])], 'banks': []}
                totals = {'teacher': 0, 'natural': 0}; ni = int(n)
                for case_index, entry in enumerate(target['quality_bridges']):
                    bi, ci = entry['book'], entry['case']; assert (bi, ci) == divmod(case_index, 4)
                    item = cohort['items'][bi]['cases'][ci]; assert entry['source_ids'] == item['source_ids'] and entry['decoder_ids'] == item['decoder_ids']
                    s = len(entry['source_ids']); split = int(bi >= 12); teacher_encoder = None
                    for mode_index, mode in enumerate(('teacher', 'natural')):
                        stored = entry['teacher_rows' if mode == 'teacher' else 'generation_rows'][0]
                        t = len(entry['decoder_ids']) if mode == 'teacher' else stored['actual_generated_tokens']
                        trace = Path(stored['router_trace_path']); command = next(c for c in capture['commands'] if c['trace_path'] == str(trace))
                        assert command['returncode'] == 0 and not command['negative'] and command['trace_sha256'] == stored['router_trace_sha256']
                        argv = command['argv']; prefix = argv[4]
                        whole = Path(prefix + '.0.bin'); assert stored['output_sha256'] == entry['teacher_sha256' if mode == 'teacher' else 'natural_sha256']
                        reused(trace, stored['router_trace_sha256']); reused(whole, stored['output_sha256'])
                        raw_trace = trace.read_bytes(); assert raw_trace[:8] == b'SWRTA001' and struct.unpack_from('<2I', raw_trace, 8) == (ni, 768)
                        dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (ni,))])
                        rows_count = 6 * (s + t); assert len(raw_trace) == 16 + rows_count * dtype.itemsize
                        trace_rows = np.frombuffer(raw_trace, dtype, offset=16); assert np.array_equal(trace_rows['index'], np.arange(rows_count))
                        assert np.all(trace_rows['phase'][:6*s] == 0) and np.all(trace_rows['phase'][6*s:] == 1) and np.all(np.isfinite(trace_rows['input'])) and np.all(np.isfinite(trace_rows['scores']))
                        encoder = raw_trace[16:16 + 6 * s * dtype.itemsize]
                        if mode == 'teacher':
                            teacher_encoder = encoder
                        else:
                            assert encoder == teacher_encoder, 'teacher_natural_encoder_identical_NOT_new_examples'
                        er, dr = route_bytes(whole.read_bytes(), ni, s, t); routes = np.concatenate([er.reshape(-1), dr.T.reshape(-1)])
                        selected = np.argmax(trace_rows['scores'], axis=1); assert np.array_equal(selected, routes['expert'])
                        differences = trace_rows['scores'] - trace_rows['scores'][np.arange(rows_count), selected, None]
                        probability = (1 / np.exp(differences.astype(np.float64)).astype(np.float32).astype(np.float64).sum(axis=1)).astype(np.float32)
                        error = float(np.max(np.abs(probability.astype(np.float64) / routes['probability'] - 1))); assert error <= 1e-6
                        accepted_indices = np.flatnonzero(routes['accepted']); codes, scales = quantize(trace_rows['input'][accepted_indices]); lookup = {int(index): local for local, index in enumerate(accepted_indices)}
                        for phase in (0, 1):
                            eligible = [int(i) for i in accepted_indices if (i < 6*s) == (phase == 0)]
                            if eligible:
                                index = eligible[0]; local = lookup[index]; sq, sa = scalar_quantize(trace_rows['input'][index], SAMPLE_COORDS)
                                assert codes[local, list(SAMPLE_COORDS)].tolist() == sq and float(scales[local]) == sa
                        for index, row in enumerate(trace_rows):
                            bank = index // s if index < 6*s else 6 + (index - 6*s) % 6
                            expert, accepted = int(routes[index]['expert']), int(routes[index]['accepted']); state = states[n][bank * ni + expert][2 * split + mode_index]
                            state['selected'] += 1; state['executed'] += accepted; state['rejected'] += 1 - accepted
                            input_bytes = row['input'].tobytes(); hi = hashlib.sha256(input_bytes).digest(); hq = hp = bytes(32); alpha = 0.
                            if accepted:
                                local = lookup[index]; qbytes = codes[local].tobytes(); alpha = float(scales[local]); sb = struct.pack('<f', alpha)
                                hq = hashlib.sha256(qbytes).digest(); hp = hashlib.sha256(qbytes + sb).digest()
                                state['input'].add(hi); state['pair'].add(hp); state['scale'].add(sb); state['books'].add(bi); state['cases'].add((bi, ci))
                                state['probability_sum'] += float(routes[index]['probability'])
                                state['code'][hq] = state['code'].get(hq, 0) | (1 << bi)
                            ledger.write(LEDGER.pack(ni, mode_index, bi, ci, bank, accepted, split, expert, index, alpha, float(routes[index]['probability']), hi, hq, hp)); count += 1
                        totals[mode] += rows_count; guard()
                        result['trace_checks'].append({'n': ni, 'mode': mode, 'book': bi, 'case': ci, 'source_tokens': s, 'positions': t, 'rows': rows_count,
                            'trace_path': str(trace), 'trace_sha256': stored['router_trace_sha256'], 'whole_output_path': str(whole), 'whole_output_sha256': stored['output_sha256'],
                            'maximum_probability_relative_error': error, 'per_phase_first_accepted_query_exact_rational_coordinate_check': True})
                    if ci == 3 and bi % 4 == 3:
                        print(json.dumps({'n': n, 'completed_book': bi, 'ledger_queries': count, 'seconds': time.monotonic() - start}), flush=True)
                assert all(totals[mode] == target['router_mass_diagnostics'][mode]['route_queries'] for mode in totals)
                result['sources'][n]['all_route_query_totals_exact393'] = totals
            ledger.seek(12); ledger.write(struct.pack('<Q', count))
        stage = 'fixed_domain_readiness'; fixed_ready = None
        for n, source in result['sources'].items():
            ni = int(n)
            for bank in range(12):
                experts = []; supported = covered = val_queries = 0
                for expert in range(ni):
                    bs = states[n][bank * ni + expert]; dev, val = code_union(bs[:2]), code_union(bs[2:]); novel = set(val) - set(dev)
                    novel_books = 0
                    for key in novel:
                        novel_books |= val[key]
                    masks = [i for i in range(24) if novel_books & (1 << i)]
                    dev_summary, val_summary = summary(bs[:2]), summary(bs[2:])
                    readiness = {'development_unique_codes_ge32': len(dev) >= 32, 'development_books_ge4': len(dev_summary['books']) >= 4,
                                 'validation_novel_codes_ge16': len(novel) >= 16, 'validation_novel_code_books_ge4': len(masks) >= 4}
                    ready = all(readiness.values()); supported += ready; val_queries += bs[3]['executed']; covered += bs[3]['executed'] if ready else 0
                    experts.append({'expert': expert, 'by_split_mode': {f'{split}_{mode}': summary([bs[2*si+mi]]) for si, split in enumerate(('dev', 'val')) for mi, mode in enumerate(('teacher', 'natural'))},
                        'dev_union': dev_summary, 'val_union': val_summary, 'validation_code_seen_in_dev': len(set(val) & set(dev)), 'validation_novel_codes': len(novel), 'validation_novel_code_books': masks,
                        'cross_mode_identical_input_dev_val': [len(bs[i]['input'] & bs[i+1]['input']) for i in (0, 2)],
                        'cross_mode_identical_code_dev_val': [len(set(bs[i]['code']) & set(bs[i+1]['code'])) for i in (0, 2)],
                        'sampled_dev_span_rank_UPPER_bound_NOT_measured_rank': min(768, len(dev)), 'readiness_gates': readiness, 'data_ready': ready})
                coverage = covered / val_queries if val_queries else 0.
                gates = {'data_ready_distinct_IDs_ge32': supported >= 32, 'validation_natural_executed_query_coverage_ge0p90': coverage >= .90}
                source['banks'].append({'bank': bank, 'stack': 'encoder' if bank < 6 else 'decoder', 'layer': 2 * (bank % 6) + 1,
                    'data_ready_IDs': supported, 'val_natural_executed_queries': val_queries, 'covered_val_natural_executed_queries': covered,
                    'coverage': coverage, 'readiness_gates': gates, 'passed': all(gates.values()), 'experts': experts})
            if n == '128':
                fixed_ready = source['banks'][11]['passed']
        assert count == sum(sum(source['all_route_query_totals_exact393'].values()) for source in result['sources'].values()) == 96186
        result['fixed_future_probe'] = {'source': 128, 'bank': 11, 'decoder_layer': 11, 'hypothetical_first_input_subspace_rank': 32, 'data_ready': fixed_ready}
        result['factor_bytes_algebra'] = {'original_expert_bytes': 4733952, 'F32_factor_plus_original_WO_lower_bound': '15360*r+2374656', 'any_byte_reduction_r_le': 153,
                                       'at_least20percent_expert_byte_reduction_r_le': 91, 'rank_is_not_observed_here': True, 'metadata_workspace_native_cost_quality_unqualified': True}
        result['ledger'] = {'path': str(ledger_path), 'bytes': ledger_path.stat().st_size, 'sha256': digest(ledger_path), 'record_bytes': LEDGER.size, 'records': count}
        assert ledger_path.stat().st_size == 20 + count * LEDGER.size
        result['preserved_daemons_after'] = jobs(); guard()
        result['gates'] = {'committed_controller_protocol_bound_evidence_originals_exact': True, 'ALL384_retained_trace_and_whole_output_SHA_exact': True,
            'ALL_native_trace_order_selected_ID_acceptance_probability_bridges': True, 'ALL_encoder_teacher_natural_traces_identical_and_not_independent': True,
            '13_cached_native_and6_rational_A16_controls_plus_per_trace_phase_samples': True, 'all12banks_each128_256_complete_book_domain_code_scale_provenance': True,
            'fixed_split_minimums_novelty_and_a_priori128_last_decoder_decision': True, 'exclusive_ledger_raw_no_model_fit_or_scientific_rerun_resources': True}
        result['decision'] = 'data_admitted_for_separately_frozen_ONE_rank32_private_INPUT_geometry_probe' if fixed_ready else 'existing_data_insufficient_for_fixed_private_INPUT_probe_stop_before_factors_design_new_independent_captures'
        result['scope'] = 'Consumed393 trace data admission, not fresh donor quality, rank measurement, projector fit, compressed artifact, useful-n/speed/physicalDRAM or general-family evidence. Book units are distinct source hashes, not an iid claim. Teacher/natural views correlated. Accepted consultations alone define WI domains; rejected query code fields are zero sentinels, not computed/executed expert codes. SHA256-distinct counts assume no cryptographic collision. Matrix/input nonlinear/routing composition still requires fresh same-artifact tests. Unsupported expert-ID fallback retains original weights; query-time fallback retaining BOTH encodings must be separately priced.'
        result['resource'] = {'main_seconds_excluding_imports': time.monotonic() - start, 'admission_seconds': phase_start - start, 'peak_working_set_bytes': peak, 'bytes_hashed_before_raw': hashed}
        result['resource'].update(raw_bytes=0, ledger_bytes=ledger_path.stat().st_size)
        for _ in range(5):
            encoded = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode('utf-8')
            assert len(encoded) + ledger_path.stat().st_size <= 32 << 20; guard()
            if len(encoded) == result['resource']['raw_bytes'] and peak == result['resource']['peak_working_set_bytes']:
                break
            result['resource'].update(raw_bytes=len(encoded), peak_working_set_bytes=peak)
        else:
            raise AssertionError('raw_size_resource_fixed_point')
        write_new(RAW, result)
        assert RAW.stat().st_size == result['resource']['raw_bytes'] and RAW.stat().st_size + ledger_path.stat().st_size <= 32 << 20
        print(json.dumps({'sha256': digest(RAW), 'gates': result['gates'], 'decision': result['decision'], 'fixed_future_probe': result['fixed_future_probe'],
                          'banks': {n: [{key: bank[key] for key in ('bank', 'data_ready_IDs', 'coverage', 'passed')} for bank in source['banks']] for n, source in result['sources'].items()}, 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update(stage=stage, error=repr(error), main_seconds_excluding_imports=time.monotonic() - start, peak_working_set_bytes=peak)
        write_new(RAW.with_suffix('.failure.json'), result); raise


if __name__ == '__main__':
    main()
