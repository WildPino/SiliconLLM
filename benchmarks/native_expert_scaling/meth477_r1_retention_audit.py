"""One stdlib-only scalar reporting/retention audit; no head operator replay."""
import time
START = time.monotonic()
import argparse
import ast
import base64
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import os
import struct
import subprocess
import sys
import traceback

ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth477_r1_head_observable_recovery'
SOURCE_OUT = ROOT / 'results/native_expert_scaling/meth477_switch_head_observable'
RAW = DOC / 'meth477_r1_head_observable_result.json'
RET = DOC / 'RETENTION_477_R1_20261005.json'
BIND = DOC / 'meth477_r1_prospective_bindings.json'
PROTO = DOC / 'METH_477_R1_SCALAR_RETENTION_PROTOCOL_20261005.md'
RAW_SHA = '593baadd707f5e0526e77e6d90bee637ed7bd48b432a84abf93f6b6f73b99791'
BIND_SHA = 'e304e512c361f03af06db46bbe7b0497f841e2516709aae02af2ad126e379bb9'
FAIL_SHA = 'f28571f5ee24d07ef6e786fcef3852516069305c1bdead5731b0686d4274c339'
parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
assert args.out.resolve() == RET.resolve() and not RET.exists()
assert not (OUT / 'scalar_retention_first_failure.json').exists()

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
handle = kernel.GetCurrentProcess()
kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
assert kernel.SetProcessAffinityMask(handle, 1)
kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)) and process_mask.value == 1

class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32)] + [(name, ctypes.c_size_t) for name in (
        'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
        'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]

get_memory = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
get_memory.argtypes = (ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_uint32)
peak = hashed = 0
phase = 'immutable_admission'
cache = {}

def guard():
    global peak
    value = MemoryCounters()
    value.cb = ctypes.sizeof(value)
    assert get_memory(handle, ctypes.byref(value), value.cb)
    peak = max(peak, value.PeakWorkingSetSize, value.WorkingSetSize)
    assert time.monotonic() - START <= 120 and peak <= 256 << 20, (phase, peak)

def digest(path):
    global hashed
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        value = hashlib.sha256()
        with path.open('rb') as stream:
            while data := stream.read(4 << 20):
                value.update(data)
                hashed += len(data)
                guard()
        cache[key] = value.hexdigest()
    return cache[key]

def check(path, item):
    path = Path(path)
    assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], str(path)
    if 'mtime_ns' in item:
        assert path.stat().st_mtime_ns == item['mtime_ns'], str(path)

def head(path):
    path = Path(path).resolve()
    rel = path.relative_to(ROOT).as_posix()
    expected = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel])
    assert path.read_bytes() == expected, rel

def failure(kind, value, tb):
    destination = OUT / 'scalar_retention_first_failure.json'
    retained = {'phase': phase, 'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'raw_sha256': RAW_SHA, 'seconds': time.monotonic() - START, 'OS_peak_bytes': peak,
                'traceback': ''.join(traceback.format_exception(kind, value, tb))}
    if not destination.exists():
        with destination.open('xb') as stream:
            stream.write((json.dumps(retained, indent=2) + '\n').replace('\n', '\r\n').encode())
    sys.__excepthook__(kind, value, tb)

sys.excepthook = failure
head(__file__)
head(PROTO)
assert digest(RAW) == RAW_SHA and digest(BIND) == BIND_SHA
head(BIND)
r = json.loads(RAW.read_bytes())
b = json.loads(BIND.read_bytes())
fail_path = DOC / 'meth477_switch_head_observable_result.failure.json'
assert digest(fail_path) == FAIL_SHA
f = json.loads(fail_path.read_bytes())
assert r['source_binding_sha256'] == BIND_SHA and r['first_failure_sha256'] == FAIL_SHA
assert r['source477_parent_exit'] == 1 and r['source477_native_exit'] == 2
assert 'unavailable' in r['source477_end_worker_readback']
assert [v['returncode'] for v in f['commands']] == [0, 0, 2]
assert len(f['apparatus_gates']) == 3 and all(f['apparatus_gates'].values())
assert r['native_commands'] == r['whole_model_commands'] == 0
assert r['cached_head_vectors_rederived'] == 7993 + 6649
assert len(r['gates']) == 9 and all(r['gates'].values())
assert sys.version == b['runtime']['python'] and Path(sys.executable).resolve() == Path(b['runtime']['executable']).resolve()
for path, sha in {**f['scientific_sources'], **r['scientific_sources']}.items():
    head(path)
    assert digest(path) == sha
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
for rel, item in b['helpers'].items():
    head(ROOT / rel)
    check(ROOT / rel, item)
for name, item in b['records'].items():
    head(DOC / name)
    check(DOC / name, item)
assert r['first_failure_retention_sha256'] == b['records']['RETENTION_477_FIRST_FAILURE_20261005.json']['sha256']
for path, item in b['runtime']['files'].items():
    check(path, item)
for package, item in b['runtime']['packages'].items():
    assert importlib.metadata.version(package) == item['version']
    for path, entry in item['files'].items():
        check(path, entry)
for path, item in b['native_files'].items():
    check(path, item)
for key in ('preparation_helper', 'controller_text_generation_helper'):
    check(b[key]['path'], b[key])
for item in b['predecessor_preparation_helpers']:
    check(item['path'], item)
assert len(b['retained_inventory']) == 6591 and sum(v['bytes'] for v in b['retained_inventory']) == 7754504204
for item in [*b['retained_inventory'], *b['additional_source_files'], *b['native_FFN_controls'], *b['first_failure_files']]:
    check(item['path'], item)
for path, item in b['integer_fixtures'].items():
    check(path, item)
artifact = b['original_artifact']
assert f['artifact'] == artifact and digest(artifact['payload']) == artifact['sha256']
assert digest(artifact['manifest']) == artifact['manifest_sha256']
payload_stat = Path(artifact['payload']).stat()
assert [payload_stat.st_size, payload_stat.st_mtime_ns] == f['payload_stat_before'] == [b['payload_stat_before']['bytes'], b['payload_stat_before']['mtime_ns']]
assert {v.name for v in SOURCE_OUT.iterdir()} == {Path(v['path']).name for v in b['first_failure_files']} and len(b['first_failure_files']) == 13
assert (SOURCE_OUT / 'native.stderr.log').read_bytes() == b'switch_reference_error:ALL_terminal_counts_EOF\r\n'
for rel, item in b['preserved_unrelated_files'].items():
    check(ROOT / rel, item)
head(ROOT / 'benchmarks/phase60/engine.c')
assert digest(ROOT / 'benchmarks/phase60/engine.c') == b['original_engine']['sha256']
assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines() == b['tracked_status']
gates = {'immutable6591_inputs_runtime_artifact_first_failure_and_frozen_sources': True}

phase = 'terminal_resources_and_output_wires'
event_source = ROOT / 'benchmarks/native_expert_scaling/meth477_r1_windows_terminal.ps1'
head(event_source)
event_path = ROOT / 'results/native_expert_scaling/meth477_r1_windows_terminal.json'
event = json.loads(event_path.read_bytes())
assert event['query_available'] and event['event_id'] == 1000 and not event['matching_scientific_events']
assert len(event['instances']) == 1
instance = event['instances'][0]
assert (instance['pid'], instance['create_time_unix']) == (r['process_instance']['pid'], r['process_instance']['create_time_unix'])
parse_time = datetime.fromisoformat
assert parse_time(event['query_start_utc']) <= parse_time(r['start_utc']) and parse_time(event['query_end_utc']) >= parse_time(r['end_utc'])
assert abs(parse_time(r['start_utc']).timestamp() - r['process_instance']['create_time_unix']) <= 1e-6
progress = [json.loads(line) for line in (OUT / 'progress.jsonl').read_text().splitlines()]
assert progress[-1]['terminal'] and progress[-1]['raw_sha256'] == RAW_SHA
assert {v['pid'] for v in progress} == {r['process_instance']['pid']}
assert progress[-1]['OS_peak_bytes'] == r['resource']['OS_peak_bytes'] <= 2 << 30
assert r['resource']['seconds_before_raw'] <= progress[-1]['seconds'] <= 600
assert len(progress) == 9 and progress[-2]['completed_batches'] == 576
assert {p.name for p in OUT.iterdir()} == {'fatal_native.log', 'progress.jsonl', 'metric_rederivation_absolute_error.npy'}
assert (OUT / 'fatal_native.log').stat().st_size == 0
for item in r['output_inventory']:
    check(item['path'], item)
assert {Path(v['path']).name for v in r['output_inventory']} == {'fatal_native.log', 'metric_rederivation_absolute_error.npy'}
output_bytes = sum(p.stat().st_size for p in OUT.iterdir()) + RAW.stat().st_size
assert output_bytes <= r['resource']['hard_new_bytes'] == 8 << 20
source_progress = [json.loads(line) for line in (SOURCE_OUT / 'progress.jsonl').read_text().splitlines()]
assert source_progress[-1]['terminal_failure'] and source_progress[-1]['seconds'] <= 180
assert f['resource_on_failure']['OS_peak_bytes'] <= 2 << 30
gates['actual_R1_terminal_progress_PID_creation_windows_resources_and_output_SHA'] = True

def npy(stream):
    assert stream.read(6) == b'\x93NUMPY'
    version = stream.read(2)
    assert version in (b'\x01\x00', b'\x02\x00')
    size = struct.unpack('<H' if version == b'\x01\x00' else '<I', stream.read(2 if version == b'\x01\x00' else 4))[0]
    descriptor = ast.literal_eval(stream.read(size).decode('latin1'))
    return stream.tell(), descriptor

FIELDS = ('kl', 'tv', 'margin_source', 'margin_candidate', 'delta_min', 'delta_max', 'pmax_source', 'pmax_candidate',
          'fisher_half', 'nll_source', 'nll_candidate', 'delta_nll', 'error_energy', 'reference_energy', 'route_probability', 'range_KL_bound')
wire = struct.Struct('<QIHBBHH7I16d')
assert wire.size == 176
rows = []
with (SOURCE_OUT / 'observables.bin').open('rb') as stream:
    assert stream.read(16) == struct.pack('<8sII', b'M477OBS1', 6649, 1748)
    for k in range(6649):
        value = stream.read(1748)
        assert len(value) == 1748
        entry = wire.unpack_from(value)
        assert all(math.isfinite(v) for v in entry[14:])
        rows.append(entry)
    assert not stream.read(1)
qpath = ROOT / 'results/native_expert_scaling/meth472_switch_private_input_probe/query_inputs.npy'
qmeta = struct.Struct('<QH4BHIff')
dev_codes = {e: set() for e in range(128)}
validation = {}
with qpath.open('rb') as stream:
    offset, descriptor = npy(stream)
    assert offset == 448 and descriptor['shape'] == (19962,) and not descriptor['fortran_order']
    for k in range(19962):
        value = stream.read(4732)
        assert len(value) == 4732
        m = qmeta.unpack_from(value)
        assert 0 <= m[6] < 128 and m[4] in (0, 1)
        if m[4] == 0:
            dev_codes[m[6]].add(value[4668:4700])
        else:
            validation[k] = (m, value[4668:4700])
    assert not stream.read(1)
assert len(validation) == 6649
with (ROOT / 'results/native_expert_scaling/meth476_switch_last_layer_context/context.bin').open('rb') as stream:
    assert stream.read(16) == struct.pack('<8sII', b'M476CTX1', 7993, 4628)
    contexts = []
    for k in range(7993):
        value = stream.read(4628)
        assert len(value) == 4628
        contexts.append(struct.unpack_from('<QII', value))
    assert not stream.read(1)
cohort = json.loads((DOC / 'meth467_switch_rust_query_manifest.json').read_bytes())
labels = {}
for book in range(64, 128):
    item = cohort['items'][book]
    assert item['book'] == book and item['role'] == 'diagnostic_validation'
    excerpt = base64.b64decode(item['excerpt_token_ids_base64_le_i32'], validate=True)
    assert hashlib.sha256(excerpt).hexdigest() == item['excerpt_token_ids_sha256']
    ids = struct.unpack('<' + str(len(excerpt) // 4) + 'i', excerpt)
    for case, v in enumerate(item['cases']):
        original = list(ids[v['excerpt_token_start']:v['excerpt_token_start'] + 32])
        assert original == v['original_window_ids'] and v['span_starts'] == [3, 10, 17, 24]
        target = [x for k, start in enumerate((3, 10, 17, 24)) for x in (32099 - k, *original[start:start + 2])] + [32095, 1]
        assert target == v['target_ids'] and hashlib.sha256(struct.pack('<14i', *target)).hexdigest() == v['target_ids_sha256']
        labels[book, case] = target
ready = set(b['fixed_ready_IDs'])
assert len(ready) == 107
native_progress = [json.loads(line) for line in (SOURCE_OUT / 'native.stdout.log').read_text().splitlines()]
assert len(native_progress) == len(b['last_layer_batches']) == 608
with (SOURCE_OUT / 'jobs.bin').open('rb') as stream:
    assert stream.read(44) == struct.pack('<8s3I3Q', b'M477JOB1', 608, 7993, 6649, 448, 128, 128)
    cx = ox = 0
    seen = []
    for batch, item in enumerate(b['last_layer_batches']):
        kind, book, case, mode, s, t, qstart, context_start, ledger = struct.unpack('<6I3Q', stream.read(48))
        assert (kind, book, case, mode, s, t, qstart, context_start, ledger) == (item['kind'], item['book'], item['case'], item['mode'], item['s'], item['t'], item['query_start'], cx, item['ledger_base'])
        for key in ('whole_path', 'control_path'):
            size = struct.unpack('<I', stream.read(4))[0]
            assert stream.read(size).decode() == item[key]
        for pos in range(t):
            job_label, job_flag = struct.unpack('<2I', stream.read(8))
            expected_qid = qstart + pos if kind else 2**64 - 1
            assert contexts[cx + pos] == (expected_qid, batch, pos)
            if not kind:
                assert job_label == 2**32 - 1 and job_flag == 0
                continue
            value = rows[ox + pos]
            qid, cr, rb, rc, rm, expert, flags, position, arg_s, arg_c, label, ties_s, ties_c, reserved = value[:14]
            assert (qid, cr, rb, rc, rm, position, reserved) == (qstart + pos, cx + pos, book, case, mode, pos, 0)
            qm, code_sha = validation[qid]
            assert (qm[1], qm[2], qm[3], qm[4], qm[6], qm[7], qm[0]) == (book, case, mode, 1, expert, 179 + 6 * pos, ledger + 179 + 6 * pos)
            base_flags = int(expert in ready) | (2 if code_sha not in dev_codes[expert] else 0) | (16 if expert in (25, 37) else 0)
            if mode == 0:
                base_flags |= 4 | (8 if pos in (1, 2, 4, 5, 7, 8, 10, 11) else 0)
            target = labels[book, case][pos] if mode == 0 else 2**32 - 1
            assert job_flag == base_flags and label == job_label == target
            assert 0 <= arg_s < 32128 and 0 <= arg_c < 32128 and 1 <= ties_s <= 32128 and 1 <= ties_c <= 32128
            metrics = dict(zip(FIELDS, value[14:]))
            delta_range = metrics['delta_max'] - metrics['delta_min']
            expected_flags = base_flags | (32 if ties_s == 1 else 0) | (64 if ties_c == 1 else 0) | (128 if arg_s != arg_c else 0) | (256 if ties_s == 1 and metrics['margin_source'] > delta_range else 0)
            assert flags == expected_flags and metrics['route_probability'] == qm[9]
            assert metrics['kl'] >= -1e-11 and delta_range >= 0 and 0 <= metrics['tv'] <= 1 + 1e-11
            assert abs(metrics['range_KL_bound'] - delta_range**2 / 8) <= 1e-11 + 1e-10 * abs(metrics['range_KL_bound'])
            assert metrics['kl'] <= metrics['range_KL_bound'] + 1e-11 + 1e-10 * abs(metrics['range_KL_bound'])
            assert metrics['error_energy'] >= 0 and metrics['reference_energy'] >= 0
            if mode == 1:
                assert metrics['nll_source'] == metrics['nll_candidate'] == metrics['delta_nll'] == 0
            if not flags & 1:
                assert arg_s == arg_c and metrics['kl'] == metrics['tv'] == metrics['error_energy'] == 0
            seen.append(qid)
        cx += t
        ox += t if kind else 0
        assert native_progress[batch] == {'batch': batch, 'source_positions': cx, 'candidate_positions': ox}
        guard()
    assert not stream.read(1) and (cx, ox) == (7993, 6649)
assert len(set(seen)) == 6649 and set(seen) == set(validation)
assert r['correct_head_cardinality'] == (cx + ox) * 32128 == 470418176
assert r['original_wrong_head_cardinality'] == 470422176
gates['independent_all608_jobs_context_query_labels_novelty_flags_EOF_cardinality'] = True

with (OUT / 'metric_rederivation_absolute_error.npy').open('rb') as stream:
    offset, descriptor = npy(stream)
    assert offset == 128 and descriptor == {'descr': '<f8', 'fortran_order': False, 'shape': (6649, 16)}
    maxima = [0.] * 16
    for row in rows:
        errors = struct.unpack('<16d', stream.read(128))
        for k, (error, native_value) in enumerate(zip(errors, row[14:])):
            assert math.isfinite(error) and 0 <= error <= 1e-11 + 1e-10 * abs(native_value)
            maxima[k] = max(maxima[k], error)
    assert not stream.read(1)
assert r['metric_absolute_difference_max'] == dict(zip(FIELDS, maxima))
gates['saved_all6649x16_error_matrix_tolerances_and_reported_maxima'] = True

phase = 'independent_scalar_reporting'
def summary(items):
    n = len(items)
    if not n:
        return {'count': 0}
    result = {'count': n, 'argmax_changed': sum(bool(v[6] & 128) for v in items),
              'unique_source': sum(bool(v[6] & 32) for v in items), 'margin_certified': sum(bool(v[6] & 256) for v in items),
              'negative_raw_KL': sum(v[14] < 0 for v in items)}
    for name in ('kl', 'tv', 'margin_source', 'delta_range', 'fisher_half'):
        values = [v[19] - v[18] if name == 'delta_range' else max(v[14], 0.) if name == 'kl' else v[14 + FIELDS.index(name)] for v in items]
        ordered = sorted(values)
        def quantile(p):
            index = (n - 1) * p
            lo = int(index)
            return ordered[lo] + (ordered[min(lo + 1, n - 1)] - ordered[lo]) * (index - lo)
        result[name] = {'mean': math.fsum(values) / n, 'p50': quantile(.5), 'p95': quantile(.95), 'p99': quantile(.99), 'max': ordered[-1]}
    for name, bit in (('teacher_all_label', 4), ('teacher_masked_content_label', 8)):
        labeled = [v for v in items if v[6] & bit]
        if labeled:
            result[name] = {'count': len(labeled), 'source_mean_NLL': math.fsum(v[23] for v in labeled) / len(labeled),
                            'candidate_mean_NLL': math.fsum(v[24] for v in labeled) / len(labeled),
                            'mean_delta_NLL': math.fsum(v[25] for v in labeled) / len(labeled),
                            'source_label_argmax_correct': sum(v[8] == v[10] for v in labeled),
                            'candidate_label_argmax_correct': sum(v[9] == v[10] for v in labeled)}
    den = math.fsum(v[27] for v in items)
    result['pooled_FFN_RMS'] = math.sqrt(math.fsum(v[26] for v in items) / den) if den > 0 else None
    result['reference_energy'] = den
    return result

views = {'all': rows, 'teacher': [v for v in rows if v[4] == 0], 'natural': [v for v in rows if v[4] == 1]}
for yes, no, bit in (('ready', 'fallback', 1), ('novel_code', 'seen_code', 2), ('dominant25_37', 'other_IDs', 16)):
    for flag, name in ((True, yes), (False, no)):
        views[name] = [v for v in rows if bool(v[6] & bit) == flag]
        for mode, label in ((0, 'teacher'), (1, 'natural')):
            views[name + '_' + label] = [v for v in views[name] if v[4] == mode]
expected = {'views': {k: summary(v) for k, v in views.items()},
            'books': {str(book): summary([v for v in rows if v[2] == book]) for book in range(64, 128)},
            'experts': {str(e): {name: summary([v for v in rows if v[5] == e and (mode is None or v[4] == mode)]) for mode, name in ((None, 'all'), (0, 'teacher'), (1, 'natural'))} for e in range(128)}}
statistics_checked = integers_checked = 0
max_absolute_stat_difference = 0.
def compare(expected_value, actual, trail):
    global statistics_checked, integers_checked, max_absolute_stat_difference
    if isinstance(expected_value, dict):
        assert isinstance(actual, dict) and expected_value.keys() == actual.keys(), trail
        for key in expected_value:
            compare(expected_value[key], actual[key], trail + '.' + key)
    elif isinstance(expected_value, int):
        assert type(actual) is int and expected_value == actual, trail
        integers_checked += 1
    elif expected_value is None:
        assert actual is None, trail
    else:
        assert isinstance(actual, (float, int)) and math.isfinite(actual), trail
        difference = abs(expected_value - actual)
        assert difference <= 1e-11 + 1e-10 * abs(expected_value), (trail, expected_value, actual)
        statistics_checked += 1
        max_absolute_stat_difference = max(max_absolute_stat_difference, difference)
compare(expected, r['summaries'], 'summaries')
assert (len(views['teacher']), len(views['natural'])) == (3584, 3065)
assert sum(v['count'] for v in expected['books'].values()) == sum(v['all']['count'] for v in expected['experts'].values()) == 6649
assert expected['views']['teacher']['teacher_masked_content_label']['count'] == 2048
gates['ALL_uniformquery_21views_64books_128IDs_3modes_independent_scalar_stats'] = True
assert not ({'numpy', 'psutil', 'threadpoolctl', 'torch', 'transformers', 'scipy', 'pandas', 'meth477_observable_math', 'meth477_switch_head_observable', 'meth477_r1_head_observable_recovery'} & set(sys.modules))
gates['stdlib_only_no_head_native_model_fitting_query_selection_or_promotion'] = True
guard()
inventory = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir())]
report = {'experiment': 'METH477-R1 independent scalar reporting retention', 'audit_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'raw_sha256': RAW_SHA, 'source_binding_sha256': BIND_SHA, 'first_failure_sha256': FAIL_SHA,
          'audit_source_sha256': digest(__file__), 'audit_protocol_sha256': digest(PROTO), 'windows_source_sha256': digest(event_source), 'windows_terminal_sha256': digest(event_path),
          'gates': gates, 'statistics_checked': statistics_checked, 'integers_checked': integers_checked,
          'max_absolute_stat_difference': max_absolute_stat_difference,
          'complete_denominators': {'source_contexts': cx, 'candidate_contexts': ox, 'teacher': 3584, 'masked_content': 2048, 'natural': 3065, 'books': 64, 'ID_slots': 128, 'head_logits': (cx + ox) * 32128},
          'original477_parent_exit': 1, 'original477_native_exit': 2, 'original477_end_worker_readback': r['source477_end_worker_readback'],
          'resource': {'seconds_before_retention': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed, 'hard_seconds': 120, 'hard_peak_bytes': 256 << 20, 'CPU_affinity_mask': 1, 'R1_new_bytes_raw_plus_OUT': output_bytes},
          'output_inventory': inventory, 'decision': 'COMPLETE_HEAD_OBSERVABLE_WITNESSES_RECOVERED_for_descriptive_calibration_NO_PROMOTION',
          'scope': 'Original477 remains failed; old472 remains localFAIL; no new artifact, changed own-state quality, rate, physicalDRAM, useful-n or multiple-family conclusion.'}
with RET.open('xb') as stream:
    stream.write((json.dumps(report, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n').encode())
guard()
print(json.dumps({'retention': str(RET), 'sha256': digest(RET), 'gates': gates, 'statistics_checked': statistics_checked,
                  'integers_checked': integers_checked, 'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}), flush=True)
