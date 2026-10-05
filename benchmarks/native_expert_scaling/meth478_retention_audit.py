"""Independent full-wire, geometry, ordered traversal and work audit; no C replay."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import time
START = time.monotonic()
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import faulthandler
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import traceback
import psutil

ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SOURCE = ROOT / 'results/native_expert_scaling/meth478_grouped_router_probe'
OUT = ROOT / 'results/native_expert_scaling/meth478_retention_audit'
RAW = DOC / 'meth478_grouped_router_result.json'
BIND = DOC / 'meth478_prospective_bindings.json'
PROTO = DOC / 'METH_478_RETENTION_PROTOCOL_20261005.md'
RET = DOC / 'RETENTION_478_20261005.json'
RAW_SHA = '61cae4d4218e0bd9590be7baf1f49de3941d7b0a43eb4f805236e482827c42e4'
BIND_SHA = '07dc05c3e34df0570880b6c20315008614ffeb43061a109cbcc271a8aa6d6894'
parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
assert args.out.resolve() == RET.resolve() and not RET.exists() and not OUT.exists()
OUT.mkdir()
fatal = (OUT / 'fatal_native.log').open('xb')
faulthandler.enable(file=fatal, all_threads=True)
progress = (OUT / 'progress.jsonl').open('x', encoding='utf8')
parent = psutil.Process()
parent.cpu_affinity([0])
assert parent.cpu_affinity() == [0]
instance = {'pid': parent.pid, 'create_time_unix': parent.create_time(), 'executable': sys.executable}
phase = 'immutable_admission'
peak = hashed = comparisons = numeric_fields = integer_fields = audited = 0
cache = {}

def checkpoint(**values):
    progress.write(json.dumps({'phase': phase, 'seconds': time.monotonic() - START,
                               'pid': parent.pid, **values}) + '\n')
    progress.flush()

def guard():
    global peak
    info = parent.memory_info()
    peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
    assert time.monotonic() - START <= 600 and peak <= 2 << 30, (phase, peak)
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) + (RET.stat().st_size if RET.exists() else 0) <= 16 << 20

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

def check(item):
    p = Path(item['path'])
    assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256'], str(p)
    if 'mtime_ns' in item:
        assert p.stat().st_mtime_ns == item['mtime_ns'], str(p)

def head(path):
    path = Path(path).resolve()
    rel = path.relative_to(ROOT).as_posix()
    expected = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel])
    assert path.read_bytes() == expected, rel

def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n').encode())

def failure(kind, value, tb):
    destination = OUT / 'first_failure.json'
    if not destination.exists():
        write(destination, {'experiment': 'METH478 independent audit FIRST fault', 'phase': phase,
                            'raw_sha256': RAW_SHA, 'binding_sha256': BIND_SHA,
                            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'process_instance': instance, 'queries_audited': audited,
                            'seconds': time.monotonic() - START, 'OS_peak_bytes': peak,
                            'traceback': ''.join(traceback.format_exception(kind, value, tb))})
    checkpoint(terminal_failure=True)
    sys.__excepthook__(kind, value, tb)

sys.excepthook = failure
checkpoint()
head(__file__)
head(PROTO)
assert digest(RAW) == RAW_SHA and digest(BIND) == BIND_SHA
head(BIND)
r = json.loads(RAW.read_bytes())
b = json.loads(BIND.read_bytes())
assert r['source_binding_sha256'] == BIND_SHA and len(r['gates']) == 7 and all(r['gates'].values())
assert r['native_model_commands'] == 0 and r['native_numeric_commands'] == 4
assert sys.version == b['runtime']['python'] and Path(sys.executable).resolve() == Path(b['runtime']['executable']).resolve()
assert psutil.__version__ == b['runtime']['packages']['psutil']['version']
assert str(Path(psutil.__file__).resolve()) in b['runtime']['packages']['psutil']['files']
for path, item in b['runtime']['files'].items():
    check({'path': path, **item})
for package, item in b['runtime']['packages'].items():
    assert importlib.metadata.version(package) == item['version']
    for path, value in item['files'].items():
        check({'path': path, **value})
for key in ('records', 'helpers', 'scientific_files', 'source_inventory', 'compile_assets', 'system_files'):
    for item in b[key]:
        check(item)
for key in ('compiler', 'preparation_helper'):
    check(b[key])
check(b['metadata_correction']['helper'])
assert digest(b['metadata_correction']['initial_binding_path']) == b['metadata_correction']['initial_binding_sha256']
for item in b['records'] + b['helpers'] + b['scientific_files']:
    head(item['path'])
for path, value in r['scientific_sources'].items():
    head(path)
    assert digest(path) == value
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
assert len(b['source_inventory']) == 1624 and sum(v['bytes'] for v in b['source_inventory']) == 1748802650
assert {str(p.resolve()) for p in Path(b['tasks'][0]['trace']).parent.iterdir() if p.is_file()} == {v['path'] for v in b['source_inventory']}
assert len(b['compile_assets']) == 5353
for target in b['targets']:
    artifact = target['artifact']
    assert Path(artifact['payload']).stat().st_size == artifact['bytes'] and digest(artifact['payload']) == artifact['sha256']
    assert digest(artifact['manifest']) == artifact['manifest_sha256']
for rel, item in b['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256'] and (ROOT / rel).stat().st_size == item['bytes']
assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines() == b['tracked_status']
own = {parent.pid, *(p.pid for p in parent.parents())}
daemons = []
for p in psutil.process_iter(['name', 'cmdline']):
    if p.pid in own:
        continue
    try:
        name, argv = (p.info['name'] or '').lower(), p.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            daemons.append(p.pid)
            continue
        assert not (name.startswith('python') or name.startswith('clang') or (name.startswith('meth') and name.endswith('.exe'))), (p.pid, name)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
gates = {'immutable_complete_sources_runtime_artifacts_393_477_and_freezes': True}
checkpoint(file_bytes_hashed=hashed)

phase = 'terminal_and_wires'
event_path = ROOT / 'results/native_expert_scaling/meth478_windows_terminal.json'
event = json.loads(event_path.read_bytes())
assert event['query_available'] and event['event_id'] == 1000 and not event['matching_scientific_events']
expected_instances = [r['process_instance']]
for command in r['commands']:
    expected_instances += [command['process_instance'], *command['descendant_process_peaks']]
assert {(v['pid'], v['create_time_unix']) for v in event['instances']} == {(v['pid'], v['create_time_unix']) for v in expected_instances}
assert datetime.fromisoformat(event['query_start_utc']) <= datetime.fromisoformat(r['start_utc'])
assert datetime.fromisoformat(event['query_end_utc']) >= datetime.fromisoformat(r['end_utc'])
source_progress = [json.loads(line) for line in (SOURCE / 'progress.jsonl').read_text().splitlines()]
assert source_progress[-1]['terminal'] and source_progress[-1]['raw_sha256'] == RAW_SHA
assert {v['pid'] for v in source_progress} == {r['process_instance']['pid']}
assert r['resource']['seconds_before_raw'] <= source_progress[-1]['seconds'] <= r['resource']['hard_seconds'] == 300
assert source_progress[-1]['OS_peak_combined_bytes'] == r['resource']['OS_peak_combined_bytes'] <= r['resource']['hard_peak_bytes'] == 2 << 30
assert len(r['output_inventory']) == 40
assert {p.name for p in SOURCE.iterdir()} == {Path(v['path']).name for v in r['output_inventory']} | {'progress.jsonl'}
for item in r['output_inventory']:
    check(item)
assert (SOURCE / 'fatal_native.log').stat().st_size == 0
assert sum(p.stat().st_size for p in SOURCE.iterdir()) + RAW.stat().st_size <= r['resource']['hard_new_bytes'] == 96 << 20
assert [v['label'] for v in r['commands']] == ['compile', 'controls', 'negative_weight_ledger', 'source', 'candidate']
assert [v['returncode'] for v in r['commands']] == [0, 0, 2, 0, 0]
assert all(v['returncode'] == v['expected_returncode'] for v in r['commands'])
assert r['commands'][0]['argv'][1:6] == ['-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off']
for command in r['commands']:
    pi = command['process_instance']
    assert datetime.fromisoformat(command['start_utc']).timestamp() - 1 <= pi['create_time_unix'] <= datetime.fromisoformat(command['end_utc']).timestamp()
    assert pi['executable'] == command['argv'][0]
    if command['label'] != 'negative_weight_ledger':
        assert (SOURCE / (command['label'] + '.stderr.log')).stat().st_size == 0
assert (SOURCE / 'negative_weight_ledger.stderr.log').read_bytes() == b'grouped_router_error:weight_ledger_magic\r\n'
assert (SOURCE / 'negative_weight_ledger.bin').read_bytes() == b'INVALID!' + struct.pack('<2I', 24, 768)
assert r['source_oracle'] == json.loads((SOURCE / 'source.stdout.log').read_bytes())
assert r['source_oracle'] == {'source_queries': 96186, 'source_score_values': 18438144, 'scores_BYTE_exact': True, 'probability_ID_BYTE_exact': True}
native_progress = [json.loads(v) for v in (SOURCE / 'candidate.stdout.log').read_text().splitlines()]
assert len(native_progress) == 13 and native_progress[-1] == r['native_terminal']
assert [v['jobs_completed'] for v in native_progress[:-1]] == list(range(32, 385, 32))
assert native_progress[-2]['queries_completed'] == 96186 and r['native_terminal']['terminal']
assert r['native_terminal']['CPU_affinity_mask'] == 1 and not r['native_terminal']['MXCSR'] & 0x8040

def f32(v):
    return struct.unpack('<f', struct.pack('<f', v))[0]

controls = [json.loads(v) for v in (SOURCE / 'controls.stdout.log').read_text().splitlines()]
assert len(controls) == 9
with localcontext() as context:
    context.prec = 100
    for k, values in enumerate(((0, 1, -1), (10000, 9999, -10000), (0, 0, 0), (1000, 999, -1000), (0, -1, -2), (0, 0, -1))):
        chosen = max(range(3), key=lambda j: values[j])
        terms = [f32(float((Decimal(v) - Decimal(values[chosen])).exp())) for v in values]
        assert controls[k] == {'control': k, 'chosen': chosen, 'denominator': sum(terms), 'probability': f32(1 / sum(terms))}
for kind in range(2):
    a = [(i % 7 - 3) * 2.**-10 for i in range(768)] if not kind else [2.**-100 if i % 2 else 2.**80 for i in range(768)]
    x = [(i % 5 - 2) * 2.**-9 for i in range(768)] if not kind else [2.**-20 if i % 2 else -2.**-80 if i % 4 else 2.**-80 for i in range(768)]
    lanes = [[0.] * 4 for unused in range(2)]
    for i in range(0, 768, 8):
        for lane in range(4):
            lanes[0][lane] += a[i + lane] * x[i + lane]
            lanes[1][lane] += a[i + lane + 4] * x[i + lane + 4]
    total = 0.
    for lane in range(4):
        total += lanes[0][lane] + lanes[1][lane]
    assert controls[6 + kind] == {'dot_control': kind, 'double_result': total, 'f32_result': f32(total)}
assert controls[8]['CPU_affinity_mask'] == 1 and controls[8]['rounding'] == 0 and not controls[8]['MXCSR'] & 0x8040

def read_string(stream):
    size, = struct.unpack('<I', stream.read(4))
    assert 0 < size < 4096
    data = stream.read(size)
    assert len(data) == size
    return data.decode()

with (SOURCE / 'weights.ledger.bin').open('rb') as stream:
    assert stream.read(16) == struct.pack('<8sII', b'M478WGT1', 24, 768)
    for item in b['weights']:
        assert struct.unpack('<2IQ', stream.read(16)) == (item['n'], item['bank'], item['offset'])
        assert read_string(stream) == item['payload']
    assert not stream.read(1)
with (SOURCE / 'jobs.bin').open('rb') as stream:
    assert stream.read(16) == struct.pack('<8sII', b'M478JOB1', 384, 96186)
    for item in b['tasks']:
        assert struct.unpack('<6I', stream.read(24)) == tuple(item[k] for k in ('n', 'book', 'case', 'mode', 's', 't'))
        assert (read_string(stream), read_string(stream)) == (item['trace'], item['whole'])
    assert not stream.read(1)
config_keys = ('d_model', 'd_ff', 'num_heads', 'd_kv', 'num_layers', 'num_decoder_layers', 'num_experts', 'expert_capacity', 'vocab_size', 'relative_attention_num_buckets', 'relative_attention_max_distance', 'encoder_sparse_step', 'decoder_sparse_step')
exports = {256: 'meth338_switch_tensor_recovery_result.json', 128: 'meth380_switch_base128_export_result.json'}
for target in b['targets']:
    ex = json.loads((DOC / exports[target['n']]).read_bytes())
    assert ex['artifact'] == target['artifact'] and len(ex['tensors']) == target['manifest_entries'] and all(ex['gates'].values())
    with Path(target['artifact']['manifest']).open('rb') as stream:
        assert stream.read(8) == b'SWI8A001'
        fields = struct.unpack('<13IfII', stream.read(64))
        assert fields[:13] == tuple(ex['original_config'][k] for k in config_keys)
        assert fields[14:] == (1, target['manifest_entries'])
        assert struct.pack('<f', fields[13]) == struct.pack('<f', ex['original_config']['layer_norm_epsilon'])
        assert read_string(stream) == str(Path(target['artifact']['payload']).resolve())
        for name, item in sorted(ex['tensors'].items()):
            assert read_string(stream) == name
            shape = item['shape']
            assert struct.unpack('<5I3Q', stream.read(44)) == (0, len(shape), shape[0], shape[1] if len(shape) == 2 else 1, item['encoding'], item['offset'], item['scale_offset'], item['elements'])
        assert not stream.read(1)
    for item in b['weights']:
        if item['n'] == target['n']:
            descriptor = ex['tensors'][item['name']]
            assert descriptor['shape'] == [item['n'], 768] and descriptor['encoding'] == 0
            assert [descriptor[k] for k in ('offset', 'bytes', 'sha256')] == [item[k] for k in ('offset', 'bytes', 'sha256')]
gates['actual_terminals_PID_creation_Windows_resources_controls_and_full_metadata_wires'] = True
checkpoint()

phase = 'full_source_trace_whole_observation_joins'
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
for module in (np, threadpoolctl):
    assert module.__version__ == b['runtime']['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
OBS = np.dtype([('meta', '<u4', (20,)), ('value', '<f8', (8,)), ('frontier', 'u1', (64,))])
assert OBS.itemsize == 208 and (SOURCE / 'observations.bin').stat().st_size == 24 + 96186 * 208 == b['observation_bytes']
with (SOURCE / 'observations.bin').open('rb') as stream:
    assert stream.read(24) == struct.pack('<8sIIQ', b'M478OBS1', 208, 0, 96186)
obs = np.memmap(SOURCE / 'observations.bin', dtype=OBS, mode='r', offset=24, shape=(96186,))
assert np.array_equal(obs['meta'][:, 0], np.arange(96186)) and np.isfinite(obs['value']).all()
capture = json.loads((DOC / 'meth393_switch_router_audit_result.json').read_bytes())
assert len(capture['gates']) == 8 and all(capture['gates'].values())
reconstructed_tasks = []
for target in capture['targets']:
    assert len(target['quality_bridges']) == 96
    for bridge in target['quality_bridges']:
        for mode, key in ((0, 'teacher_rows'), (1, 'generation_rows')):
            assert len(bridge[key]) == 1
            item = bridge[key][0]
            trace = Path(item['router_trace_path'])
            whole = trace.with_name(trace.name.removesuffix('.router.bin') + '.0.bin')
            assert digest(trace) == item['router_trace_sha256'] and digest(whole) == item['output_sha256']
            reconstructed_tasks.append((target['n'], bridge['book'], bridge['case'], mode, str(trace), str(whole)))
assert len(reconstructed_tasks) == len(b['tasks']) == 384
chunks = {(n, bank): [] for n in (128, 256) for bank in range(12)}
serial = 0
for ordinal, task in enumerate(b['tasks']):
    n, s, t, count = task['n'], task['s'], task['t'], task['queries']
    assert task['ordinal'] == ordinal and count == 6 * (s + t) and s == 29
    assert reconstructed_tasks[ordinal] == tuple(task[k] for k in ('n', 'book', 'case', 'mode', 'trace', 'whole'))
    assert (not task['mode'] and t == 14) or (task['mode'] and 1 <= t <= 64)
    with Path(task['whole']).open('rb') as stream:
        assert stream.read(36) == struct.pack('<8s7I', b'SWR32O01', s, t, 768, 12, 12, 32128, count)
        offset = 36 + 4 * (14 * s * 768 + t * 14 * 768 + t * 32128)
        assert Path(task['whole']).stat().st_size == offset + count * 12
        stream.seek(offset)
        routes = np.frombuffer(stream.read(count * 12), dtype=np.dtype([('id', '<i4'), ('accepted', '<i4'), ('p', '<f4')]))
        assert not stream.read(1) and np.isin(routes['accepted'], [0, 1]).all()
    dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('x', '<f4', (768,)), ('score', '<f4', (n,))])
    with Path(task['trace']).open('rb') as stream:
        assert stream.read(16) == struct.pack('<8sII', b'SWRTA001', n, 768)
        data = stream.read()
    assert len(data) == dtype.itemsize * count
    trace = np.frombuffer(data, dtype=dtype)
    index = np.arange(count)
    assert np.array_equal(trace['index'], index) and np.array_equal(trace['phase'], index >= 6 * s)
    assert np.isfinite(trace['x']).all() and np.isfinite(trace['score']).all()
    bits = trace['x'].copy().view('<u4')
    assert np.all(((bits & 0x7f800000) != 0) | ((bits & 0x7fffffff) == 0))
    bank = np.where(index < 6 * s, index // s, 6 + (index - 6 * s) % 6)
    pos = np.where(index < 6 * s, index % s, (index - 6 * s) // 6)
    rows = obs[serial:serial + count]
    expected = np.column_stack((index + serial, np.full(count, n), np.full(count, task['book']), np.full(count, task['case']), np.full(count, task['mode']), bank, pos))
    assert np.array_equal(rows['meta'][:, :7], expected)
    assert np.array_equal(rows['meta'][:, 8], routes['id']) and np.array_equal(routes['id'], np.argmax(trace['score'], axis=1))
    assert np.array_equal(rows['value'][:, 0].astype('<f4').view('<u4'), routes['p'].view('<u4'))
    for bank_id in range(12):
        mask = bank == bank_id
        chunks[n, bank_id].append((trace['x'][mask].copy(), trace['score'][mask].copy(), (index + serial)[mask]))
    serial += count
    guard()
assert serial == 96186 and sum(v['n'] * v['queries'] for v in b['tasks']) == 18438144
gates['ALL384_trace_whole_original393_source_ID_probability_phase_position_EOF_joins'] = True
checkpoint(queries_joined=serial)

def gamma(q):
    u = q * 2.**-53
    return u / (1 - u)

def close(a, z, tag):
    global comparisons
    a, z = np.asarray(a), np.asarray(z)
    assert a.shape == z.shape, tag
    good = (a == z) | (np.isfinite(a) & np.isfinite(z) & (np.abs(a - z) <= 1e-11 + 1e-10 * np.abs(z)))
    assert good.all(), (tag, np.flatnonzero(~good)[:8].tolist())
    comparisons += a.size

def parse_tree(item):
    n, bank = item['n'], item['bank']
    path = SOURCE / f'n{n}.bank{bank}.tree.bin'
    with Path(item['payload']).open('rb') as stream:
        stream.seek(item['offset'])
        original = stream.read(item['bytes'])
    assert hashlib.sha256(original).hexdigest() == item['sha256']
    w = np.frombuffer(original, '<f4').reshape(n, 768).astype('<f8')
    assert np.isfinite(w).all()
    num = 2 * n - 1
    nodes = []
    with path.open('rb') as stream:
        assert stream.read(32) == struct.pack('<8s6I', b'M478TRE1', n, bank, 768, num, 80, 8)
        assert stream.read(item['bytes']) == original
        for j in range(num):
            left, right, parent_id, m, first, res0, mode, res1, radius, maximum, cnorm, meanerror, s0, s1 = struct.unpack('<8I6d', stream.read(80))
            assert res0 == res1 == 0 and m > 0 and m <= n and m & (m - 1) == 0
            center = np.frombuffer(stream.read(768 * 8), '<f8').copy() if m > 1 else np.zeros(768)
            direction = np.frombuffer(stream.read(768 * 8), '<f8').copy() if m > 1 else np.zeros(768)
            ids = np.frombuffer(stream.read(m * 4), '<u4').copy()
            assert len(ids) == m and np.array_equal(ids, np.unique(ids)) and ids[0] == first and ids[-1] < n
            assert np.isfinite([radius, maximum, cnorm, meanerror, s0, s1]).all() and min(radius, maximum, cnorm, meanerror, s0, s1) >= 0
            padding = 8 * 768 + m + 16
            def norm(z):
                return float(np.nextafter(float(np.linalg.norm(z)) * (1 + gamma(padding)), np.inf))
            rows = w[ids]
            close(np.asarray(maximum), np.asarray(max(norm(row) for row in rows)), 'node maxnorm')
            if m > 1:
                assert left > j and right > j and left < num and right < num and mode in (0, 1, 2)
                assert np.isfinite(center).all() and np.isfinite(direction).all()
                assert np.array_equal(center, rows.mean(axis=0))
                residual = rows - center
                c = norm(center)
                rr = float(np.nextafter(float(np.linalg.norm(residual, axis=1).max()) * (1 + gamma(padding)) + gamma(padding) * (maximum + c), np.inf))
                ee = float(np.nextafter(gamma(2 * m + 2) * norm(np.abs(rows).mean(axis=0)), np.inf))
                close(np.array([radius, cnorm, meanerror]), np.array([rr, c, ee]), 'node radius/center/mean')
                assert s0 >= s1 and abs(float(direction @ direction) - 1) <= 1e-11
                if mode == 1:
                    assert s0 == 0 and np.array_equal(direction, np.eye(1, 768, 0)[0])
                elif mode == 2:
                    axis = int(np.argmax(np.sum(residual * residual, axis=0)))
                    assert s0 - s1 <= gamma(32 * m + 16) * s0 and np.array_equal(direction, np.eye(1, 768, axis)[0])
                else:
                    assert s0 > 0 and s0 - s1 > gamma(32 * m + 16) * s0
                    assert direction[int(np.argmax(np.abs(direction)))] > 0
                    eigen = residual.T @ (residual @ direction) - s0 * s0 * direction
                    assert float(np.linalg.norm(eigen)) <= 1e-10 * max(s0 * s0, float(np.sum(residual * residual)))
                ordering = np.lexsort((ids, residual @ direction))
                parts = (np.sort(ids[ordering[:m // 2]]), np.sort(ids[ordering[m // 2:]]))
            else:
                assert left == right == 2**32 - 1 and radius == cnorm == meanerror == s0 == s1 == mode == 0
                parts = None
            nodes.append({'left': left, 'right': right, 'parent': parent_id, 'm': m, 'first': first, 'ids': ids,
                          'center': center, 'radius': radius, 'maxnorm': maximum, 'centernorm': cnorm,
                          'meanerror': meanerror, 'mode': mode, 'parts': parts})
            guard()
        assert not stream.read(1)
    assert nodes[0]['parent'] == 2**32 - 1 and np.array_equal(nodes[0]['ids'], np.arange(n))
    leaf_ids = []
    for j, node in enumerate(nodes):
        if j:
            p = node['parent']
            assert p < j and j in (nodes[p]['left'], nodes[p]['right'])
        if node['m'] > 1:
            left, right = nodes[node['left']], nodes[node['right']]
            assert left['parent'] == right['parent'] == j and left['m'] == right['m'] == node['m'] // 2
            assert node['left'] == j + 1 and node['right'] == j + 2 * left['m']
            assert all(np.array_equal(child['ids'], part) for child, part in zip((left, right), node['parts']))
        else:
            leaf_ids.append(node['first'])
    assert sorted(leaf_ids) == list(range(n))
    retained = next(v for v in r['trees'] if v['n'] == n and v['bank'] == bank)
    assert retained['nodes'] == num and retained['internal'] == n - 1 and retained['sha256'] == digest(path)
    assert retained['principal_nodes'] == sum(v['m'] > 1 and v['mode'] == 0 for v in nodes)
    assert retained['zero_nodes'] == sum(v['mode'] == 1 for v in nodes)
    assert retained['degenerate_axis_nodes'] == sum(v['mode'] == 2 for v in nodes)
    return nodes

def ordered_center_dots(x, centers):
    # Explicit separate multiplication/addition and four-lane final ordering.
    left = np.zeros((len(x), len(centers), 4), '<f8')
    right = np.zeros_like(left)
    for i in range(0, 768, 8):
        left += x[:, None, i:i + 4].astype('<f8') * centers[None, :, i:i + 4]
        right += x[:, None, i + 4:i + 8].astype('<f8') * centers[None, :, i + 4:i + 8]
    values = left + right
    result = np.zeros(values.shape[:2], '<f8')
    for lane in range(4):
        result += values[:, :, lane]
    return result

def ordered_norm(x):
    left = np.zeros((len(x), 4), '<f8')
    right = np.zeros_like(left)
    xx = x.astype('<f8')
    for i in range(0, 768, 8):
        left += xx[:, i:i + 4] * xx[:, i:i + 4]
        right += xx[:, i + 4:i + 8] * xx[:, i + 4:i + 8]
    values = left + right
    result = np.zeros(len(x), '<f8')
    for lane in range(4):
        result += values[:, lane]
    return np.nextafter(np.sqrt(result) * (1 + gamma(8 * 768 + 16)), np.inf)

def audit_block(nodes, x, scores, indices, n):
    global audited, integer_fields, numeric_fields
    rows = obs[indices]
    count, num = len(x), len(nodes)
    q_all = np.arange(count)
    m = np.array([v['m'] for v in nodes], np.int32)
    first = np.array([v['first'] for v in nodes], np.int32)
    child_l = np.array([v['left'] if v['m'] > 1 else -1 for v in nodes], np.int32)
    child_r = np.array([v['right'] if v['m'] > 1 else -1 for v in nodes], np.int32)
    parents = np.array([v['parent'] if j else -1 for j, v in enumerate(nodes)], np.int32)
    internal = np.flatnonzero(m > 1)
    leaves = np.flatnonzero(m == 1)
    norm = ordered_norm(x)
    center = np.zeros((count, num), '<f8')
    center[:, internal] = ordered_center_dots(x, np.array([nodes[j]['center'] for j in internal]))
    center[:, leaves] = scores[:, first[leaves]]
    rho = norm[:, None] * np.array([v['radius'] for v in nodes])[None, :]
    maximum = np.array([v['maxnorm'] for v in nodes])[None, :]
    center_norm = np.array([v['centernorm'] for v in nodes])[None, :]
    ec = gamma(2 * 768 + 8) * center_norm * norm[:, None]
    en = (gamma(768 + 8) + 2.**-24 * (1 + gamma(768 + 8))) * maximum * norm[:, None] + 2.**-150
    eta_score = ec + en
    score_upper = center + rho + eta_score
    score_lower = center - rho - eta_score
    score_upper[:, leaves] = score_lower[:, leaves] = center[:, leaves]
    for j in internal:
        native_min = scores[:, nodes[j]['ids']].min(axis=1).astype('<f8')
        native_max = scores[:, nodes[j]['ids']].max(axis=1).astype('<f8')
        assert np.all(score_lower[:, j] <= native_min + 1e-11 + 1e-10 * np.abs(native_min))
        assert np.all(score_upper[:, j] + 1e-11 + 1e-10 * np.abs(native_max) >= native_max)

    visited = np.zeros((count, num), bool)
    expanded = np.zeros_like(visited)
    visit_order = np.empty((count, num), np.int32)
    visits = np.zeros(count, np.int32)
    counts = np.zeros((count, 11), np.uint32)  # meta9..19, derived independently
    heap = np.empty((count, num), np.int32)
    sizes = np.zeros(count, np.int32)
    key = score_upper
    mass_lower = mass_upper = None
    agg_l = np.zeros_like(center)
    agg_u = np.zeros_like(center)

    def better(q, a, z):
        counts[q, 6] += 1
        ka, kz = key[q, a], key[q, z]
        return (ka > kz) | ((ka == kz) & (first[a] < first[z]))

    def push(q, ids):
        if not len(q):
            return
        counts[q, 5] += 1
        at = sizes[q].copy()
        sizes[q] += 1
        while len(q):
            zero = at == 0
            heap[q[zero], 0] = ids[zero]
            q, ids, at = q[~zero], ids[~zero], at[~zero]
            if not len(q):
                break
            pa = (at - 1) // 2
            old = heap[q, pa].copy()
            move = better(q, ids, old)
            heap[q[~move], at[~move]] = ids[~move]
            heap[q[move], at[move]] = old[move]
            q, ids, at = q[move], ids[move], pa[move]

    def pop(q):
        answer = heap[q, 0].copy()
        sizes[q] -= 1
        last = heap[q, sizes[q]].copy()
        alive = sizes[q] > 0
        q, last = q[alive], last[alive]
        at = np.zeros(len(q), np.int32)
        while len(q):
            ch = 2 * at + 1
            has = ch < sizes[q]
            heap[q[~has], at[~has]] = last[~has]
            q, last, at, ch = q[has], last[has], at[has], ch[has]
            if not len(q):
                break
            right = ch + 1 < sizes[q]
            move_right = np.zeros(len(q), bool)
            move_right[right] = better(q[right], heap[q[right], ch[right] + 1], heap[q[right], ch[right]])
            ch += move_right.astype(np.int32)
            chosen_child = heap[q, ch].copy()
            move = better(q, chosen_child, last)
            heap[q[~move], at[~move]] = last[~move]
            heap[q[move], at[move]] = chosen_child[move]
            q, last, at = q[move], last[move], ch[move]
        return answer

    def evaluate(q, ids):
        assert np.all(~visited[q, ids])
        visited[q, ids] = True
        visit_order[q, visits[q]] = ids
        visits[q] += 1
        leaf = m[ids] == 1
        counts[q, 0] += leaf.astype(np.uint32)
        counts[q, 1] += (~leaf).astype(np.uint32)

    evaluate(q_all, np.zeros(count, np.int32))
    push(q_all, np.zeros(count, np.int32))
    while True:
        q = np.flatnonzero(m[heap[:, 0]] > 1)
        if not len(q):
            break
        ids = pop(q)
        expanded[q, ids] = True
        counts[q, 3] += 1
        left, right = child_l[ids], child_r[ids]
        evaluate(q, left)
        evaluate(q, right)
        push(q, left)
        push(q, right)
    winner = first[heap[:, 0]]
    assert np.array_equal(winner, rows['meta'][:, 8]) and np.array_equal(winner, rows['meta'][:, 7])

    chosen_score = scores[q_all, winner].astype('<f8')
    diffs = (scores - scores[q_all, winner, None]).astype('<f4')
    native_terms = np.exp(diffs.astype('<f8')).astype('<f4').astype('<f8')
    native_total = np.zeros(count, '<f8')
    for expert in range(n):
        native_total += native_terms[:, expert]
    source_p = (1 / native_total).astype('<f4')
    assert np.array_equal(source_p.view('<u4'), rows['value'][:, 0].astype('<f4').view('<u4'))
    mean = norm[:, None] * np.array([v['meanerror'] for v in nodes])[None, :]
    esub = 2.**-24 * (maximum * norm[:, None] + eta_score + np.abs(chosen_score[:, None])) + 2.**-149
    eta = eta_score + esub
    a = np.zeros_like(rho)
    np.divide(mean, rho, out=a, where=rho > 0)
    a = np.minimum(a, 1)
    def safe_exp(z):
        ans = np.exp(np.clip(z, -745., 709.))
        ans[z > 709.] = np.inf
        ans[z < -745.] = 0.
        return ans
    log_chord = rho + np.log(.5 * (1 + a) + .5 * (1 - a) * safe_exp(-2 * rho))
    log_m = np.log(m.astype('<f8'))[None, :]
    shift = center - chosen_score[:, None]
    mass_lower = np.maximum(0., safe_exp(log_m + shift - mean - eta + math.log1p(-2.**-21)) - m[None, :] * 2.**-149)
    mass_upper = safe_exp(log_m + shift + eta + log_chord + math.log1p(2.**-21)) + m[None, :] * 2.**-149
    mass_lower[:, leaves] = mass_upper[:, leaves] = native_terms[:, first[leaves]]
    for j in range(num):
        actual = np.cumsum(native_terms[:, nodes[j]['ids']], axis=1, dtype='<f8')[:, -1]
        tolerance = 1e-11 + 1e-10 * np.abs(actual)
        assert np.all(mass_lower[:, j] <= actual + tolerance) and np.all(mass_upper[:, j] + tolerance >= actual)
    key = mass_upper - mass_lower

    def mass(q, ids):
        leaf = m[ids] == 1
        counts[q, 8] += leaf.astype(np.uint32)
        counts[q, 9] += 3 * (~leaf).astype(np.uint32)
        agg_l[q, ids] = mass_lower[q, ids]
        agg_u[q, ids] = mass_upper[q, ids]

    sizes[:] = 0
    # Native mass-heap pushes follow evaluation order, rather than node order.
    for k in range(int(visits.max())):
        q = np.flatnonzero(k < visits)
        ids = visit_order[q, k]
        keep = ~expanded[q, ids]
        q, ids = q[keep], ids[keep]
        mass(q, ids)
        keep = m[ids] > 1
        push(q[keep], ids[keep])
    for j in range(num - 1, -1, -1):
        q = np.flatnonzero(expanded[:, j])
        if len(q):
            agg_l[q, j] = agg_l[q, child_l[j]] + agg_l[q, child_r[j]]
            agg_u[q, j] = agg_u[q, child_l[j]] + agg_u[q, child_r[j]]
            counts[q, 7] += 1
    while True:
        lo = np.maximum(0., agg_l[:, 0] * (1 - gamma(n + 8)))
        hi = agg_u[:, 0] * (1 + gamma(n + 8))
        assert np.all(lo > 0) and not np.isnan(hi).any()
        ratio = hi / lo
        q = np.flatnonzero((sizes > 0) & (ratio > 1.01))
        if not len(q):
            break
        ids = pop(q)
        assert np.all(m[ids] > 1)
        expanded[q, ids] = True
        counts[q, 3] += 1
        left, right = child_l[ids], child_r[ids]
        evaluate(q, left)
        evaluate(q, right)
        mass(q, left)
        mass(q, right)
        keep = m[left] > 1
        push(q[keep], left[keep])
        keep = m[right] > 1
        push(q[keep], right[keep])
        pq, pi = q, ids
        while len(pq):
            agg_l[pq, pi] = agg_l[pq, child_l[pi]] + agg_l[pq, child_r[pi]]
            agg_u[pq, pi] = agg_u[pq, child_l[pi]] + agg_u[pq, child_r[pi]]
            counts[pq, 7] += 1
            pi = parents[pi]
            keep = pi >= 0
            pq, pi = pq[keep], pi[keep]
    full = sizes == 0
    predicted = (1 / hi).astype('<f4')
    lower_p = np.nextafter(predicted, np.float32(-np.inf))
    upper_p = np.nextafter((1 / lo).astype('<f4'), np.float32(np.inf))
    predicted[full] = lower_p[full] = upper_p[full] = source_p[full]
    ratio[full] = 1.
    counts[:, 2] = counts[:, 10] = visits
    counts[:, 4] = full.astype(np.uint32)
    counts[:, 8] += full.astype(np.uint32) * n
    assert np.all(visits == 1 + 2 * counts[:, 3]) and np.all(counts[:, 0] + counts[:, 1] == visits)
    assert np.array_equal(counts, rows['meta'][:, 9:20]), ('all counters', np.argwhere(counts != rows['meta'][:, 9:20])[:12].tolist(), indices[:1].tolist())
    frontier = visited & ~expanded
    assert np.all(np.sum(frontier * m[None, :], axis=1) == n)
    padded = np.zeros((count, 512), np.uint8)
    padded[:, :num] = frontier
    assert np.array_equal(np.packbits(padded, axis=1, bitorder='little'), rows['frontier'])
    value = np.column_stack((source_p, predicted, lower_p, upper_p, (counts[:, 0].astype('<f8') + counts[:, 1] + 1) / n, norm, rho[:, 0], ratio))
    close(value, rows['value'], 'all probability/norm/rho/ratio')
    assert np.all(np.abs(predicted.astype('<f8') / source_p - 1) <= .01 + 1e-10)
    assert np.all(lower_p <= source_p + 1e-11 + 1e-10 * np.abs(source_p)) and np.all(upper_p + 1e-11 + 1e-10 * np.abs(source_p) >= source_p)
    audited += count
    integer_fields += 11 * count
    numeric_fields += 8 * count
    guard()

phase = 'all_tree_geometry_and_independent_full_traversal'
for item in b['weights']:
    nodes = parse_tree(item)
    parts = chunks[item['n'], item['bank']]
    x = np.concatenate([v[0] for v in parts])
    scores = np.concatenate([v[1] for v in parts])
    indices = np.concatenate([v[2] for v in parts])
    chunks[item['n'], item['bank']] = []
    del parts
    for at in range(0, len(x), 2048):
        audit_block(nodes, x[at:at + 2048], scores[at:at + 2048], indices[at:at + 2048], item['n'])
        checkpoint(n=item['n'], bank=item['bank'], block_end=min(at + 2048, len(x)), queries_audited=audited)
    del x, scores, indices, nodes
assert audited == 96186 and sum(v['bytes'] for v in r['trees']) == 71379840
gates['ALL24_original_weight_trees_membership_centroid_radius_mean_eigen_median_invariants'] = True
gates['ALL96186_node_enclosures_winner_exact_binary_heap_counters_cut_mass_probability'] = True
checkpoint()

phase = 'independent_reporting_and_decision'
def summary(rows):
    meta, values = rows['meta'], rows['value']
    src = meta[:, 9].astype('<f8')
    ctr = meta[:, 10].astype('<f8')
    visits = meta[:, 11].astype('<f8')
    ratio = (src + ctr + 1) / meta[:, 1]
    relative = np.abs(values[:, 1] / values[:, 0] - 1)
    ordered = np.sort(ratio)
    position = .95 * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    p95 = float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))
    return {'queries': len(rows), 'original_rows_mean': float(src.mean()), 'centroid_rows_mean': float(ctr.mean()),
            'full_fallback_count': int(np.count_nonzero(meta[:, 13])), 'winner_mismatch_count': int(np.count_nonzero(meta[:, 7] != meta[:, 8])),
            'coefficient_ratio_mean': float(ratio.mean()), 'coefficient_ratio_p95': p95, 'coefficient_ratio_max': float(ratio.max()),
            'maximum_probability_relative_error': float(relative.max()), 'mean_probability_relative_error': float(relative.mean()),
            'heap_comparisons_mean': float(meta[:, 15].mean()), 'aggregate_updates_mean': float(meta[:, 16].mean()),
            'native_exp_calls_mean': float(meta[:, 17].mean()), 'bound_exp_calls_mean': float(meta[:, 18].mean()),
            'visited_state_records_mean': float(visits.mean()), 'logical_weight_bytes_mean': float((3072 * src + 6144 * ctr).mean()),
            'logical_norm_input_bytes_per_query': 3072, 'logical_node_descriptor_bytes_mean': float((80 * visits).mean()),
            'flat_router_F32_weight_bytes_mean': float((3072 * meta[:, 1].astype('<f8')).mean())}

report = {'all': summary(obs), 'targets': []}
for n in (128, 256):
    meta = obs['meta']
    target = {'n': n, 'all': summary(obs[meta[:, 1] == n]), 'modes': {}, 'bank_modes': []}
    for mode, label in ((0, 'teacher'), (1, 'natural')):
        target['modes'][label] = summary(obs[(meta[:, 1] == n) & (meta[:, 4] == mode)])
    for bank in range(12):
        for mode, label in ((0, 'teacher'), (1, 'natural')):
            target['bank_modes'].append({'bank': bank, 'mode': label, **summary(obs[(meta[:, 1] == n) & (meta[:, 5] == bank) & (meta[:, 4] == mode)])})
    target['work_gates'] = {'mean_coefficients_le60percent': target['all']['coefficient_ratio_mean'] <= .6,
                            'ALL24bank_mode_mean_le80percent': all(v['coefficient_ratio_mean'] <= .8 for v in target['bank_modes']),
                            'BOTHmode_p95_le100percent': all(v['coefficient_ratio_p95'] <= 1 for v in target['modes'].values())}
    target['eligible_for_NEW_C_cost'] = all(target['work_gates'].values())
    report['targets'].append(target)
def compare(a, z, path='report'):
    global numeric_fields, integer_fields
    if isinstance(a, dict):
        assert a.keys() == z.keys(), path
        for k in a:
            compare(a[k], z[k], path + '/' + str(k))
    elif isinstance(a, list):
        assert len(a) == len(z), path
        for i, (v, w) in enumerate(zip(a, z)):
            compare(v, w, path + '/' + str(i))
    elif isinstance(a, float):
        close(np.asarray(a), np.asarray(z), path)
        numeric_fields += 1
    else:
        assert a == z, (path, a, z)
        integer_fields += isinstance(a, (int, bool))
compare(report, r['summaries'])
assert not any(t['eligible_for_NEW_C_cost'] for t in report['targets'])
assert r['decision'] == 'REJECT_THIS_GROUPED_RADIUS_TREE_VECTOR_WORK_pending_independent_retention'
gates['ALL_primary_counts_probabilities_55_views_vector_scalar_byte_summaries_and_gates'] = True
for rel, item in b['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
assert parent.cpu_affinity() == [0] and subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines() == b['tracked_status']
assert not ({'torch', 'transformers', 'tensorflow', 'scipy', 'sklearn', 'pandas', 'meth478_grouped_router_probe', 'meth478_grouped_router_math'} & set(sys.modules))
gates['resource_caps_no_C_model_capture_SVD_refit_or_competing_science'] = True
guard()
retained = {'experiment': 'METH478 independent complete retention', 'raw_sha256': RAW_SHA, 'binding_sha256': BIND_SHA,
            'audit_source_sha256': digest(__file__), 'audit_protocol_sha256': digest(PROTO), 'windows_terminal_sha256': digest(event_path),
            'process_instance': instance, 'start_utc': datetime.fromtimestamp(parent.create_time(), timezone.utc).isoformat(),
            'end_utc': datetime.now(timezone.utc).isoformat(), 'gates': gates,
            'queries_audited': audited, 'source_traces': 384, 'trees_audited': 24,
            'exact_derived_integer_fields': integer_fields, 'floating_fields_checked': numeric_fields,
            'enclosure_formula_comparisons': comparisons, 'independent_summaries': report,
            'decision': 'CLOSE_THIS_FIXED_GROUPED_RADIUS_TREE_VECTOR_WORK_RECIPE',
            'scope': 'ALL consumed393 queries; full native source BYTE admission retained, independently replayed heap/counters/cut using cached qualified leaf scores; eigen/partition checks do not establish global SVD optimality; no universal libm guarantee, new artifact/quality/rate/physicalDRAM/causal-n claim',
            'native_or_model_replays': 0, 'SVD_refits': 0, 'preserved_daemons': daemons,
            'resource': {'seconds_before_RET': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed,
                         'hard_seconds': 600, 'hard_peak_bytes': 2 << 30, 'hard_new_bytes': 16 << 20}}
write(RET, retained)
guard()
checkpoint(terminal=True, retention_sha256=digest(RET), OS_peak_bytes=peak)
print(json.dumps({'retention': str(RET), 'sha256': digest(RET), 'gates': gates, 'queries': audited,
                  'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()
