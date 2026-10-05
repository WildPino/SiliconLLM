"""METH479-R3 independent all-source canonical/native-target admission; no replay."""
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
from pathlib import Path
import struct
import subprocess
import sys
import traceback
import psutil

ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SOURCE_OUT = ROOT / 'results/native_expert_scaling/meth479_routing_supervision'
RECOVERY_OUT = ROOT / 'results/native_expert_scaling/meth479_r2_routing_supervision'
OUT = ROOT / 'results/native_expert_scaling/meth479_r3_retention_audit'
RAW = DOC / 'meth479_r2_routing_supervision_result.json'
FAIL = DOC / 'meth479_routing_supervision_result.failure.json'
FIRST_INVENTORY = DOC / 'meth479_first_scientific_failure_inventory.json'
FAIL_SHA = '3f4233b0d7189274b345a5be459624c40c358fc86adba682f8b52a287cd89db3'
INVENTORY_SHA = '33ecee9303981d42597a63c9fcd90546b9b9b3f7054e014d231be54ad7bf8b96'
BIND = DOC / 'meth479_prospective_bindings.json'
PROTO = DOC / 'METH_479_R3_RETENTION_PROTOCOL_20261005.md'
RET = DOC / 'RETENTION_479_R3_20261005.json'
BIND_SHA = '0523b1822b84f6e11b40025b3c677116042a06853f4094375d271df73a4016ac'
RAW_SHA = 'c03efee574a62743ac214f0c94213f41e2cbb34db0b6db86ddd18ebab53d9200'
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
peak = hashed = integer_fields = target_fields = audited_queries = audited_unique = 0
cache = {}

def checkpoint(**value):
    progress.write(json.dumps({'phase': phase, 'seconds': time.monotonic() - START, 'pid': parent.pid, **value}) + '\n')
    progress.flush()

def guard():
    global peak
    info = parent.memory_info()
    peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
    assert time.monotonic() - START <= 600 and peak <= 4 << 30, (phase, peak)
    assert sum(v.stat().st_size for v in OUT.iterdir() if v.is_file()) + (RET.stat().st_size if RET.exists() else 0) <= 32 << 20

def digest(path):
    global hashed
    p = Path(path).resolve()
    stat = p.stat()
    key = (str(p), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        value = hashlib.sha256()
        with p.open('rb') as stream:
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
    p = Path(path).resolve()
    rel = p.relative_to(ROOT).as_posix()
    assert p.read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel]), rel

def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n').encode())

def failure(kind, value, tb):
    p = OUT / 'first_failure.json'
    if not p.exists():
        write(p, {'experiment': 'METH479-R3 independent retention FIRST fault', 'phase': phase, 'raw_sha256': RAW_SHA,
                  'binding_sha256': BIND_SHA, 'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'process_instance': instance, 'queries_audited': audited_queries, 'unique_audited': audited_unique,
                  'seconds': time.monotonic() - START, 'OS_peak_bytes': peak,
                  'traceback': ''.join(traceback.format_exception(kind, value, tb))})
    checkpoint(terminal_failure=True)
    sys.__excepthook__(kind, value, tb)

sys.excepthook = failure
import threading
def deadline():
    p = OUT / 'first_failure.json'
    if not p.exists():
        write(p, {'experiment': 'METH479-R3 audit watchdog FIRST fault', 'phase': phase,
                  'raw_sha256': RAW_SHA, 'process_instance': instance, 'reason': '600 second deadline',
                  'seconds': time.monotonic() - START, 'OS_peak_bytes': peak})
    os._exit(3)
watchdog = threading.Timer(max(.001, 600 - (time.monotonic() - START)), deadline)
watchdog.daemon = True
watchdog.start()
checkpoint()
head(__file__)
head(PROTO)
assert digest(RAW) == RAW_SHA and digest(BIND) == BIND_SHA
head(BIND)
PREVIOUS_AUDIT_FAILURE = DOC / 'meth479_r2_first_audit_failure.json'
PREVIOUS_AUDIT_INVENTORY = DOC / 'meth479_r2_first_audit_failure_inventory.json'
assert digest(PREVIOUS_AUDIT_FAILURE) == '2cc9c4978230368538c02a2df5847bf3c3d8a0ce548d2fe224d39ab5c08a71a3'
assert digest(PREVIOUS_AUDIT_INVENTORY) == '1abec1bd529efc3c28a26a34fbd9c09ec7b675d4b535e78a8fa907c409edb0ca'
head(PREVIOUS_AUDIT_FAILURE)
head(PREVIOUS_AUDIT_INVENTORY)
previous = json.loads(PREVIOUS_AUDIT_FAILURE.read_bytes())
assert previous['phase'] == 'immutable_admission' and previous['queries_audited'] == previous['unique_audited'] == 0
for item in json.loads(PREVIOUS_AUDIT_INVENTORY.read_bytes())['records']:
    check(item)
previous_source = ROOT / 'benchmarks/native_expert_scaling/meth479_r2_retention_audit.py'
previous_protocol = DOC / 'METH_479_R2_RETENTION_PROTOCOL_20261005.md'
head(previous_source)
head(previous_protocol)
assert digest(previous_source) == previous['audit_source_sha256'] == 'dea6ef6fded238953d95939f85d05c1d94bd3da7cb1dbdc5a32e7f31608e8294'
assert digest(previous_protocol) == '82564a3a64faa5e58371e2f5a1746c9f610dea1f39c68005b9fe11cd90195e5b'
assert not (DOC / 'RETENTION_479_R2_20261005.json').exists()
r = json.loads(RAW.read_bytes())
b = json.loads(BIND.read_bytes())
assert r['source_binding_sha256'] == BIND_SHA and len(r['gates']) == 6 and all(r['gates'].values())
assert r['native_model_commands'] == r['training_updates'] == 0 and r['native_numeric_commands'] == 0
assert sys.version == b['runtime']['python'] and Path(sys.executable).resolve() == Path(b['runtime']['executable']).resolve()
assert psutil.__version__ == b['runtime']['packages']['psutil']['version'] and str(Path(psutil.__file__).resolve()) in b['runtime']['packages']['psutil']['files']
for path, item in b['runtime']['files'].items():
    check({'path': path, **item})
for package, item in b['runtime']['packages'].items():
    assert importlib.metadata.version(package) == item['version']
    for path, entry in item['files'].items():
        check({'path': path, **entry})
for key in ('records', 'helpers', 'scientific_files', 'source_inventory', 'compile_assets', 'system_files', 'membership_sources'):
    for item in b[key]:
        check(item)
for key in ('compiler', 'preparation_helper'):
    check(b[key])
for item in b['records'] + b['helpers'] + b['scientific_files']:
    head(item['path'])
assert digest(FAIL) == FAIL_SHA and digest(FIRST_INVENTORY) == INVENTORY_SHA
head(FAIL)
head(FIRST_INVENTORY)
f = json.loads(FAIL.read_bytes())
assert r['original_failure_sha256'] == FAIL_SHA and r['failure_inventory_sha256'] == INVENTORY_SHA
assert f['resource']['seconds'] > 600 and f['phase'] == 'all_group_support_and_native_mass_targets'
assert not (DOC / 'meth479_routing_supervision_result.json').exists()
for item in json.loads(FIRST_INVENTORY.read_bytes())['records']:
    check(item)
R1_FAILURE = DOC / 'meth479_r1_routing_supervision_result.failure.json'
R1_INVENTORY = DOC / 'meth479_r1_first_failure_inventory.json'
assert digest(R1_FAILURE) == 'ae6af9dd9c3c40144385370bb7a2f4f1abc46905b6999056c1d5e07d7d7ea92c'
assert digest(R1_INVENTORY) == '3c8361114a56308ee311b9b2220c141bd7c699d2ba6afd3195bffbfa94ca39db'
head(R1_FAILURE)
head(R1_INVENTORY)
r1_failure = json.loads(R1_FAILURE.read_bytes())
assert r1_failure['phase'] == 'immutable_admission' and not r1_failure['commands']
for item in json.loads(R1_INVENTORY.read_bytes())['records']:
    check(item)
for path, value in r1_failure['scientific_sources'].items():
    head(path)
    assert digest(path) == value
assert not (DOC / 'meth479_r1_routing_supervision_result.json').exists()
r1_event = json.loads((ROOT / 'results/native_expert_scaling/meth479_r1_windows_terminal.json').read_bytes())
assert r1_event['query_available'] and r1_event['event_id'] == 1000 and not r1_event['matching_scientific_events']
assert {(v['pid'], v['create_time_unix']) for v in r1_event['instances']} == {(r1_failure['process_instance']['pid'], r1_failure['process_instance']['create_time_unix'])}
assert datetime.fromisoformat(r1_event['query_start_utc']) <= datetime.fromisoformat(r1_failure['start_utc'])
assert datetime.fromisoformat(r1_event['query_end_utc']) >= datetime.fromisoformat(r1_failure['end_utc'])

for path, value in {**f['scientific_sources'], **r['scientific_sources']}.items():
    head(path)
    assert digest(path) == value
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
artifact = b['artifact']
assert artifact == r['artifact'] and digest(artifact['payload']) == artifact['sha256']
assert Path(artifact['payload']).stat().st_size == artifact['bytes'] == 7541946880
assert digest(artifact['manifest']) == artifact['manifest_sha256']
assert len(b['source_inventory']) == 6224 and sum(v['bytes'] for v in b['source_inventory']) == 6895456390
for rel, item in b['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256'] and (ROOT / rel).stat().st_size == item['bytes']
def status():
    rows = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines()
    assert all(v[:3] == ' M ' and v[3:] in b['preserved_unrelated_files'] for v in rows), rows
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    return rows
status()
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
gates = {'full_source6224_files_artifact_runtime_assets_metadata_fault_and_scientific_freezes': True}
checkpoint(file_bytes_hashed=hashed)

phase = 'native_terminal_controls_metadata_output_wires'
event_path = ROOT / 'results/native_expert_scaling/meth479_r2_windows_terminal.json'
event = json.loads(event_path.read_bytes())
assert event['query_available'] and event['event_id'] == 1000 and not event['matching_scientific_events']
expected_instances = [r['process_instance']]
for command in r['commands']:
    expected_instances += [command['process_instance'], *command['descendant_process_peaks']]
assert {(v['pid'], v['create_time_unix']) for v in event['instances']} == {(v['pid'], v['create_time_unix']) for v in expected_instances}
assert datetime.fromisoformat(event['query_start_utc']) <= datetime.fromisoformat(r['start_utc'])
assert datetime.fromisoformat(event['query_end_utc']) >= datetime.fromisoformat(r['end_utc'])
original_progress = [json.loads(v) for v in (SOURCE_OUT / 'progress.jsonl').read_text().splitlines()]
assert original_progress[-1]['terminal_failure']
source_progress = [json.loads(v) for v in (RECOVERY_OUT / 'progress.jsonl').read_text().splitlines()]
original_event_path = ROOT / 'results/native_expert_scaling/meth479_windows_terminal.json'
original_event = json.loads(original_event_path.read_bytes())
assert original_event['query_available'] and original_event['event_id'] == 1000 and not original_event['matching_scientific_events']
original_instances = [f['process_instance']]
for c in f['commands']:
    original_instances += [c['process_instance'], *c['descendant_process_peaks']]
assert {(v['pid'], v['create_time_unix']) for v in original_event['instances']} == {(v['pid'], v['create_time_unix']) for v in original_instances}
assert datetime.fromisoformat(original_event['query_start_utc']) <= datetime.fromisoformat(f['start_utc'])
assert datetime.fromisoformat(original_event['query_end_utc']) >= datetime.fromisoformat(f['end_utc'])
assert source_progress[-1]['terminal'] and source_progress[-1]['raw_sha256'] == RAW_SHA
assert {v['pid'] for v in source_progress} == {r['process_instance']['pid']}
assert r['resource']['seconds_before_RAW'] <= source_progress[-1]['seconds'] <= r['resource']['hard_seconds'] == 600
assert source_progress[-1]['OS_peak_combined_bytes'] == r['resource']['OS_peak_combined_bytes'] <= r['resource']['hard_peak_bytes'] == 4 << 30
assert sum(p.stat().st_size for p in RECOVERY_OUT.iterdir()) + RAW.stat().st_size <= r['resource']['hard_new_bytes'] == 2 << 30
assert {str(p.resolve()) for p in RECOVERY_OUT.iterdir()} == {v['path'] for v in r['output_inventory']} | {str((RECOVERY_OUT / 'progress.jsonl').resolve())}
for item in r['output_inventory']:
    check(item)
assert (SOURCE_OUT / 'fatal_native.log').stat().st_size == 0
assert [v['label'] for v in f['commands']] == ['compile', 'controls', 'negative_weight_ledger', 'source']
assert [v['returncode'] for v in f['commands']] == [0, 0, 2, 0]
assert all(v['returncode'] == v['expected_returncode'] for v in f['commands'])
assert f['commands'][0]['argv'][1:6] == ['-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off']
for c in f['commands']:
    pi = c['process_instance']
    assert datetime.fromisoformat(c['start_utc']).timestamp() - 1 <= pi['create_time_unix'] <= datetime.fromisoformat(c['end_utc']).timestamp()
    assert pi['executable'] == c['argv'][0]
    if c['label'] != 'negative_weight_ledger':
        assert (SOURCE_OUT / (c['label'] + '.stderr.log')).stat().st_size == 0
assert (SOURCE_OUT / 'negative_weight_ledger.stderr.log').read_bytes() == b'source_router_error:weight_ledger_magic\r\n'
assert (SOURCE_OUT / 'negative_weight_ledger.bin').read_bytes() == b'INVALID!' + struct.pack('<2I', 12, 768)
assert not (SOURCE_OUT / 'negative_source_fields.bin').exists()
native_progress = [json.loads(v) for v in (SOURCE_OUT / 'source.stdout.log').read_text().splitlines()]
assert len(native_progress) == 13 and native_progress[-1] == r['source_native_terminal']
assert [v['jobs_completed'] for v in native_progress[:-1]] == list(range(128, 1537, 128))
terminal = native_progress[-1]
assert terminal['terminal'] and terminal['source_queries'] == 387036 and terminal['source_score_values'] == 49540608
assert terminal['scores_BYTE_exact'] and terminal['ID_probability_BYTE_exact'] and terminal['CPU_affinity_mask'] == 1 and not terminal['MXCSR'] & 0x8040
controls = [json.loads(v) for v in (SOURCE_OUT / 'controls.stdout.log').read_text().splitlines()]
assert len(controls) == 9
def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]
with localcontext() as context:
    context.prec = 100
    for k, values in enumerate(((0, 1, -1), (10000, 9999, -10000), (0, 0, 0), (1000, 999, -1000), (0, -1, -2), (0, 0, -1))):
        chosen = max(range(3), key=lambda j: values[j])
        total = sum(f32(float((Decimal(v) - Decimal(values[chosen])).exp())) for v in values)
        assert controls[k] == {'control': k, 'chosen': chosen, 'denominator': total, 'probability': f32(1 / total)}
for kind in range(2):
    a = [(i % 7 - 3) * 2.**-10 for i in range(768)] if not kind else [2.**-100 if i % 2 else 2.**80 for i in range(768)]
    x = [(i % 5 - 2) * 2.**-9 for i in range(768)] if not kind else [2.**-20 if i % 2 else -2.**-80 if i % 4 else 2.**-80 for i in range(768)]
    l, z = [0.] * 4, [0.] * 4
    for at in range(0, 768, 8):
        for lane in range(4):
            l[lane] += a[at + lane] * x[at + lane]
            z[lane] += a[at + lane + 4] * x[at + lane + 4]
    value = 0.
    for lane in range(4):
        value += l[lane] + z[lane]
    assert controls[6 + kind] == {'dot_control': kind, 'double_result': value, 'f32_result': f32(value)}
assert controls[8]['CPU_affinity_mask'] == 1 and controls[8]['rounding'] == 0 and not controls[8]['MXCSR'] & 0x8040
def string(stream):
    size, = struct.unpack('<I', stream.read(4))
    assert 0 < size < 4096
    data = stream.read(size)
    assert len(data) == size
    return data.decode()
with (SOURCE_OUT / 'weights.ledger.bin').open('rb') as stream:
    assert stream.read(16) == struct.pack('<8sII', b'M479WGT1', 12, 768)
    for item in b['weights']:
        assert struct.unpack('<IQ', stream.read(12)) == (item['bank'], item['offset'])
        assert string(stream) == item['payload']
    assert not stream.read(1)
with (SOURCE_OUT / 'jobs.bin').open('rb') as stream:
    assert stream.read(16) == struct.pack('<8sII', b'M479JOB1', 1536, 387036)
    for item in b['tasks']:
        assert struct.unpack('<7I', stream.read(28)) == tuple(item[k] for k in ('book', 'case', 'mode', 's', 't', 'role', 'queries'))
        assert (string(stream), string(stream)) == (item['trace'], item['whole'])
    assert not stream.read(1)
ex = json.loads((DOC / 'meth380_switch_base128_export_result.json').read_bytes())
assert ex['artifact'] == artifact and len(ex['tensors']) == 3320 and all(ex['gates'].values())
config_keys = ('d_model', 'd_ff', 'num_heads', 'd_kv', 'num_layers', 'num_decoder_layers', 'num_experts', 'expert_capacity', 'vocab_size', 'relative_attention_num_buckets', 'relative_attention_max_distance', 'encoder_sparse_step', 'decoder_sparse_step')
with Path(artifact['manifest']).open('rb') as stream:
    assert stream.read(8) == b'SWI8A001'
    values = struct.unpack('<13IfII', stream.read(64))
    assert values[:13] == tuple(ex['original_config'][k] for k in config_keys) and values[14:] == (1, 3320)
    assert struct.pack('<f', values[13]) == struct.pack('<f', ex['original_config']['layer_norm_epsilon'])
    assert string(stream) == str(Path(artifact['payload']).resolve())
    for name, item in sorted(ex['tensors'].items()):
        assert string(stream) == name
        shape = item['shape']
        assert struct.unpack('<5I3Q', stream.read(44)) == (0, len(shape), shape[0], shape[1] if len(shape) == 2 else 1, item['encoding'], item['offset'], item['scale_offset'], item['elements'])
    assert not stream.read(1)
gates['actual_native_PID_creation_Windows_resources_controls_full_manifest_and_metadata_wires'] = True
checkpoint()

phase = 'all_original_sources_canonical_identities_ownership_and_link_joins'
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
for module in (np, threadpoolctl):
    assert module.__version__ == b['runtime']['packages'][module.__name__]['version'] and str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
source_dtype = np.dtype([('meta', '<u4', (12,)), ('value', '<f8', (3,))])
unique_dtype = np.dtype([('meta', '<u4', (12,)), ('input_sha', 'u1', (32,)), ('score_sha', 'u1', (32,)), ('value', '<f8', (3,)), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
node_dtype = np.dtype([('maximum', '<f4', (2,)), ('winner', '<u2', (2,)), ('mass', '<f8', (2,))])
target_dtype = np.dtype([('unique_id', '<u4'), ('nodes', node_dtype, (127,))])
assert (source_dtype.itemsize, unique_dtype.itemsize, node_dtype.itemsize, target_dtype.itemsize) == (72, 3720, 28, 3560)
U = r['unique_inputs']
def wire(name, magic, width, reserved, count, dtype, shape):
    path = (RECOVERY_OUT if name == 'group_targets.bin' else SOURCE_OUT) / name
    assert path.stat().st_size == 24 + width * count
    with path.open('rb') as stream:
        assert stream.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
    return np.memmap(path, dtype=dtype, mode='r', offset=24, shape=shape)
source = wire('source_fields.bin', b'M479SRC1', 72, 0, 387036, source_dtype, (387036,))
unique = wire('unique_inputs.bin', b'M479UNI1', 3720, 768, U, unique_dtype, (U,))
links = wire('query_links.bin', b'M479LNK1', 52, 0, 387036, '<u4', (387036, 13))
owner = wire('ownership.bin', b'M479OWN1', 32, 0, U, '<u4', (U, 8))
saved = wire('group_targets.bin', b'M479TGT1', 3560, 127, U, target_dtype, (U,))
assert np.array_equal(source['meta'][:, 0], np.arange(387036)) and np.array_equal(unique['meta'][:, 0], np.arange(U))
assert np.array_equal(saved['unique_id'], np.arange(U)) and np.array_equal(owner[:, 0], np.arange(U))
old = json.loads((DOC / 'meth469_switch_native_domain_capture_result.json').read_bytes())
new = json.loads((DOC / 'meth471_switch_development_capture_result.json').read_bytes())
assert old['artifact'] == new['artifact'] == artifact
expected_tasks = []
for raw in (old, new):
    for case in raw['cases']:
        book = case['book']
        assert case['role'] == b['immutable_book_roles'][book]['role']
        for mode, label in ((0, 'teacher'), (1, 'natural')):
            item = raw['commands'][case['modes'][label]]
            assert item['returncode'] == 0 and not item['negative'] and item['mode'] == label
            assert digest(item['trace_path']) == item['trace_sha256'] and digest(item['whole_output_path']) == item['whole_output_sha256']
            expected_tasks.append((book, case['case'], mode, item['trace_path'], item['whole_output_path']))
assert len(expected_tasks) == 1536
ids_seen = {}
properties = []
occurrence_ids = np.zeros((12, 3, 2, 128), '<u4')
occurrence_accept = np.zeros((12, 3, 2), '<u4')
prefix_pairs = 0
trace_dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
serial = 0
for ordinal, task in enumerate(b['tasks']):
    book, case, mode, s, t, count = (task[k] for k in ('book', 'case', 'mode', 's', 't', 'queries'))
    role = 0 if book < 64 else 1 if book < 128 else 2
    assert expected_tasks[ordinal] == (book, case, mode, task['trace'], task['whole'])
    assert task['ordinal'] == ordinal and task['role'] == role and (book, case, mode) == (ordinal // 8, ordinal // 2 % 4, ordinal % 2)
    assert count == 6 * (s + t) and s == 29 and ((mode == 0 and t == 14) or (mode == 1 and 1 <= t <= 64))
    with Path(task['trace']).open('rb') as stream:
        assert stream.read(16) == struct.pack('<8sII', b'SWRTA001', 128, 768)
        data = stream.read()
    assert len(data) == trace_dtype.itemsize * count
    trace = np.frombuffer(data, trace_dtype)
    if not mode:
        prefix = data[:180 * trace_dtype.itemsize]
    else:
        assert prefix == data[:180 * trace_dtype.itemsize]
        prefix_pairs += 1
    with Path(task['whole']).open('rb') as stream:
        assert stream.read(36) == struct.pack('<8s7I', b'SWR32O01', s, t, 768, 12, 12, 32128, count)
        offset = 36 + 4 * (14 * s * 768 + 14 * t * 768 + t * 32128)
        assert Path(task['whole']).stat().st_size == offset + count * 12
        stream.seek(offset)
        routes = np.frombuffer(stream.read(count * 12), np.dtype([('chosen', '<i4'), ('accepted', '<i4'), ('p', '<f4')]))
        assert not stream.read(1)
    idx = np.arange(count)
    assert np.array_equal(trace['index'], idx) and np.array_equal(trace['phase'], idx >= 6 * s)
    assert np.isfinite(trace['input']).all() and np.isfinite(trace['score']).all()
    bits = trace['input'].copy().view('<u4')
    assert np.all(((bits & 0x7f800000) != 0) | ((bits & 0x7fffffff) == 0))
    bank = np.where(idx < 6 * s, idx // s, 6 + (idx - 6 * s) % 6)
    position = np.where(idx < 6 * s, idx % s, (idx - 6 * s) // 6)
    native = source[serial:serial + count]
    meta = np.column_stack((idx + serial, np.full(count, 128), np.full(count, book), np.full(count, case), np.full(count, mode), np.full(count, role), bank, position, idx, routes['chosen'], routes['accepted'], np.full(count, ordinal)))
    assert np.array_equal(meta, native['meta']) and np.array_equal(meta, links[serial:serial + count, :12])
    assert np.array_equal(np.argmax(trace['score'], axis=1), routes['chosen']) and np.isin(routes['accepted'], [0, 1]).all()
    assert np.array_equal(native['value'][:, 1].astype('<f4').view('<u4'), routes['p'].view('<u4'))
    for i in range(count):
        x_bytes, s_bytes = trace['input'][i].tobytes(), trace['score'][i].tobytes()
        ih, sh = hashlib.sha256(x_bytes).digest(), hashlib.sha256(s_bytes).digest()
        k = (int(bank[i]), ih)
        first = k not in ids_seen
        if first:
            uid = len(properties)
            ids_seen[k] = uid
            properties.append([int(bank[i]), 0, 0, 0, 0, set(), set()])
            expected = (uid, 128, int(bank[i]), serial + i, book, case, mode, int(routes['chosen'][i]), int(routes['accepted'][i]), role, i, ordinal)
            assert unique[uid]['meta'].tobytes() == struct.pack('<12I', *expected)
        else:
            uid = ids_seen[k]
        record = unique[uid]
        assert record['input_sha'].tobytes() == ih and record['score_sha'].tobytes() == sh
        assert record['input'].tobytes() == x_bytes and record['score'].tobytes() == s_bytes
        assert record['value'].tobytes() == native['value'][i].tobytes()
        assert int(links[serial + i, 12]) == uid
        p = properties[uid]
        p[1] |= 1 << role
        p[2] |= 1 << mode
        p[3] |= 1 << int(routes['accepted'][i])
        p[4] += 1
        p[5 if role != 1 else 6].add(book)
        occurrence_ids[int(bank[i]), role, mode, int(routes['chosen'][i])] += 1
        occurrence_accept[int(bank[i]), role, mode] += int(routes['accepted'][i])
    serial += count
    audited_queries = serial
    if (ordinal + 1) % 128 == 0:
        checkpoint(streams_joined=ordinal + 1, queries_audited=serial, unique_inputs=len(properties))
    guard()
assert serial == 387036 and prefix_pairs == 768 and len(properties) == U <= 248796
assert r['queries'] == 387036 and r['streams'] == 1536 and r['prefix_pairs_BYTE_exact'] == 768
expected_owner = np.array([[uid, p[0], p[1], p[2], p[3], p[4], len(p[5]), len(p[6])] for uid, p in enumerate(properties)], '<u4')
assert np.array_equal(expected_owner, owner)
assert all(not p[5] & p[6] and all(v < 64 or v >= 128 for v in p[5]) and all(64 <= v < 128 for v in p[6]) for p in properties)
integer_fields += 13 * 387036 + 8 * U
dev_book_bits = np.array([[sum(1 << v for v in p[5] if v < 64), sum(1 << (v - 128) for v in p[5] if v >= 128)] for p in properties], '<u8')
del properties, ids_seen
gates['ALL1536_original_source_387036_joins_full_BYTE_identities_ownership_overlap_and_links'] = True
checkpoint(unique_inputs=U)

phase = 'all_memberships_and_independent_group_primal_targets'
memberships = []
for weight, entry in zip(b['weights'], b['membership_sources']):
    bank = weight['bank']
    descriptor = ex['tensors'][weight['name']]
    assert descriptor['shape'] == [128, 768] and descriptor['encoding'] == 0
    assert [descriptor[k] for k in ('offset', 'bytes', 'sha256')] == [weight[k] for k in ('offset', 'bytes', 'sha256')]
    with Path(weight['payload']).open('rb') as stream:
        stream.seek(weight['offset'])
        original = stream.read(weight['bytes'])
    assert hashlib.sha256(original).hexdigest() == weight['sha256']
    tree = []
    with Path(entry['path']).open('rb') as stream:
        assert stream.read(32) == struct.pack('<8s6I', b'M478TRE1', 128, bank, 768, 255, 80, 8)
        assert stream.read(128 * 768 * 4) == original
        for j in range(255):
            fields = struct.unpack('<8I6d', stream.read(80))
            left, right, parent_id, m, first, z0, mode, z1 = fields[:8]
            assert z0 == z1 == 0 and 0 < m <= 128 and m & (m - 1) == 0
            if m > 1:
                assert len(stream.read(768 * 16)) == 768 * 16
            ids = list(struct.unpack('<' + str(m) + 'I', stream.read(m * 4)))
            assert ids == sorted(set(ids)) and first == ids[0] and ids[-1] < 128
            tree.append({'node': j, 'left': left, 'right': right, 'parent': parent_id, 'count': m, 'first': first, 'ids': ids})
        assert not stream.read(1)
    assert tree[0]['ids'] == list(range(128)) and tree[0]['parent'] == 2**32 - 1
    for j, node in enumerate(tree):
        if j:
            pa = node['parent']
            assert pa < j and j in (tree[pa]['left'], tree[pa]['right'])
        if node['count'] > 1:
            l, z = tree[node['left']], tree[node['right']]
            assert l['parent'] == z['parent'] == j and l['count'] == z['count'] == node['count'] // 2
            assert node['left'] == j + 1 and node['right'] == j + 2 * l['count']
            assert not set(l['ids']) & set(z['ids']) and sorted(l['ids'] + z['ids']) == node['ids']
        else:
            assert node['left'] == node['right'] == 2**32 - 1
    assert sorted(v['first'] for v in tree if v['count'] == 1) == list(range(128))
    memberships.append({'bank': bank, 'nodes': tree})
assert json.loads((SOURCE_OUT / 'membership.json').read_bytes()) == memberships
diagnostics = {'maximum_F64_path_probability_absolute_error': 0., 'F32_path_probability_differs_from_direct_native': 0, 'zero_child_mass_fields': 0}
exposures = []
for bank in range(12):
    uid_bank = np.flatnonzero(unique['meta'][:, 2] == bank)
    tree = memberships[bank]['nodes']
    internal = [v for v in tree if v['count'] > 1]
    assert len(internal) == 127
    tables = [{'node': v['node'], 'group_size': v['count'],
               **{key: {'unique_inputs': 0, 'development_books': 0} for key in ('ALL_development', 'source_winner_path_development', 'left_ALL_development', 'right_ALL_development', 'left_path_development', 'right_path_development')},
               'development_maximum_ties': 0, 'zero_mass_pairs_ALL_unique': 0} for v in internal]
    unions = np.zeros((127, 6, 2), '<u8')
    for at in range(0, len(uid_bank), 2048):
        uid = uid_bank[at:at + 2048]
        row = unique[uid]
        score = row['score']
        q = np.arange(len(uid))
        winner = np.argmax(score, axis=1)
        assert np.array_equal(winner, row['meta'][:, 7])
        assert np.array_equal(score[q, winner].astype('<f8').view('<u8'), row['value'][:, 0].copy().view('<u8'))
        difference = (score - score[q, winner, None]).astype('<f4')
        terms = np.exp(difference.astype('<f8')).astype('<f4').astype('<f8')
        total = np.zeros(len(uid), '<f8')
        for expert in range(128):
            total += terms[:, expert]
        assert np.array_equal(total.view('<u8'), row['value'][:, 2].copy().view('<u8'))
        assert np.array_equal((1 / total).astype('<f4').view('<u4'), row['value'][:, 1].astype('<f4').view('<u4'))
        expected_target = np.zeros(len(uid), target_dtype)
        expected_target['unique_id'] = uid
        dev = (owner[uid, 2] & 5) != 0
        bits = dev_book_bits[uid]
        path_probability = np.ones(len(uid), '<f8')
        path_nodes = np.zeros(len(uid), '<u4')
        for k, node in enumerate(internal):
            group = expected_target['nodes'][:, k]
            for child in (0, 1):
                ids = tree[node['left' if not child else 'right']]['ids']
                chosen = np.full(len(uid), ids[0], '<u2')
                maximum = score[:, ids[0]].copy()
                value = np.zeros(len(uid), '<f8')
                for expert in ids:
                    take = score[:, expert] > maximum
                    maximum[take] = score[take, expert]
                    chosen[take] = expert
                    value += terms[:, expert]
                group['maximum'][:, child] = maximum
                group['winner'][:, child] = chosen
                group['mass'][:, child] = value
            right = (group['maximum'][:, 1] > group['maximum'][:, 0]) | ((group['maximum'][:, 0] == group['maximum'][:, 1]) & (group['winner'][:, 1] < group['winner'][:, 0]))
            path = np.isin(winner, node['ids'])
            assert np.array_equal(np.where(right, group['winner'][:, 1], group['winner'][:, 0])[path], winner[path])
            mass_sum = group['mass'][:, 0] + group['mass'][:, 1]
            assert np.all(mass_sum[path] > 0)
            path_probability[path] *= group['mass'][q[path], right[path].astype(int)] / mass_sum[path]
            path_nodes += path.astype('<u4')
            masks = (dev, dev & path, dev & ~right, dev & right, dev & path & ~right, dev & path & right)
            names = ('ALL_development', 'source_winner_path_development', 'left_ALL_development', 'right_ALL_development', 'left_path_development', 'right_path_development')
            for category, (name, mask) in enumerate(zip(names, masks)):
                tables[k][name]['unique_inputs'] += int(mask.sum())
                unions[k, category] |= np.bitwise_or.reduce(bits[mask], axis=0, initial=0)
            tables[k]['development_maximum_ties'] += int(np.count_nonzero(dev & (group['maximum'][:, 0] == group['maximum'][:, 1])))
            tables[k]['zero_mass_pairs_ALL_unique'] += int(np.count_nonzero(mass_sum == 0))
        assert np.array_equal(expected_target.view('u1').reshape(len(uid), 3560), saved[uid].view('u1').reshape(len(uid), 3560)), ('ALL targets BYTE', bank, at)
        assert np.all(path_nodes == 7)
        error = np.abs(path_probability - 1 / total)
        assert np.all(error <= 1e-11 + 1e-10 / total)
        diagnostics['maximum_F64_path_probability_absolute_error'] = max(diagnostics['maximum_F64_path_probability_absolute_error'], float(error.max()))
        diagnostics['F32_path_probability_differs_from_direct_native'] += int(np.count_nonzero(path_probability.astype('<f4').view('<u4') != row['value'][:, 1].astype('<f4').view('<u4')))
        diagnostics['zero_child_mass_fields'] += int(np.count_nonzero(expected_target['nodes']['mass'] == 0))
        audited_unique += len(uid)
        target_fields += len(uid) * 127 * 6
        guard()
    for k in range(127):
        for category, name in enumerate(('ALL_development', 'source_winner_path_development', 'left_ALL_development', 'right_ALL_development', 'left_path_development', 'right_path_development')):
            tables[k][name]['development_books'] = sum(int(v).bit_count() for v in unions[k, category])
    exposures.append(tables)
    checkpoint(bank=bank, unique_audited=audited_unique)
assert audited_unique == U and r['group_target_records'] == U * 127
toy = np.array([0., 0., .5, -100.], '<f4')
toy_terms = np.exp((toy - toy[2]).astype('<f4').astype('<f8')).astype('<f4')
assert int(np.argmax(toy)) == 2 and toy_terms[:2].sum(dtype='<f8') > toy_terms[2:].sum(dtype='<f8')
assert hashlib.sha256(struct.pack('<f', 0.)).digest() != hashlib.sha256(struct.pack('<f', -0.)).digest()
gates['ALL12_weight_memberships_ALL_unique_127native_max_ID_mass_targets_BYTE_path_root'] = True
checkpoint()

phase = 'independent_all_ownership_exposure_views_and_decision'
def owner_summary(rows):
    roles = rows[:, 2]
    d = (roles & 5) != 0
    v = (roles & 2) != 0
    return {'unique_inputs': len(rows), 'development_eligible': int(np.count_nonzero(d)),
            'validation_present': int(np.count_nonzero(v)), 'validation_only': int(np.count_nonzero(v & ~d)),
            'validation_overlaps_development': int(np.count_nonzero(v & d)), 'teacher_present': int(np.count_nonzero(rows[:, 3] & 1)),
            'natural_present': int(np.count_nonzero(rows[:, 3] & 2)), 'accepted_present': int(np.count_nonzero(rows[:, 4] & 2)),
            'rejected_present': int(np.count_nonzero(rows[:, 4] & 1)), 'occurrences': int(rows[:, 5].sum(dtype=np.uint64))}
report = {'ownership': owner_summary(expected_owner), 'banks': [], 'occurrences': []}
for bank in range(12):
    report['banks'].append({'bank': bank, 'ownership': owner_summary(expected_owner[expected_owner[:, 1] == bank]),
                            'branch_development_exposure': exposures[bank]})
    for role in range(3):
        for mode in range(2):
            counts = occurrence_ids[bank, role, mode]
            total = int(counts.sum(dtype=np.uint64))
            accepted = int(occurrence_accept[bank, role, mode])
            report['occurrences'].append({'bank': bank, 'role': role, 'mode': mode, 'queries': total, 'accepted': accepted,
                                          'rejected': total - accepted, 'selected_ID_counts': counts.tolist()})
def compare(a, z, path='report'):
    global integer_fields
    if isinstance(a, dict):
        assert a.keys() == z.keys(), path
        for k in a:
            compare(a[k], z[k], path + '/' + str(k))
    elif isinstance(a, list):
        assert len(a) == len(z), path
        for i, (v, w) in enumerate(zip(a, z)):
            compare(v, w, path + '/' + str(i))
    elif isinstance(a, float):
        assert abs(a - z) <= 1e-11 + 1e-10 * abs(z), path
    else:
        assert a == z, (path, a, z)
        integer_fields += isinstance(a, (int, bool))
compare(report, r['summaries'])
compare(diagnostics, r['target_diagnostics'])
assert r['decision'] == 'supervision_complete_pending_independent_retention_NOT_a_trained_candidate'
assert b['binary_upper_bound'] == (3720 + 3560 + 32) * 248796 + (52 + 72) * 387036 + 5 * 24
assert r['resource']['binary_output_upper_bound'] == 24 + U * 3560 == 850384344
assert b['binary_upper_bound'] + (8 << 20) < 2 << 30
gates['ALL72_occurrence_13_ownership_1524_development_exposure_views_and_native_diagnostics'] = True
for rel, item in b['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
last_status = status()
assert parent.cpu_affinity() == [0]
assert not ({'torch', 'transformers', 'tensorflow', 'scipy', 'sklearn', 'pandas', 'meth479_routing_supervision', 'meth479_r2_routing_supervision', 'meth479_supervision_math'} & set(sys.modules))
gates['resource_caps_no_native_model_replay_fit_refit_selection_or_competing_science'] = True
guard()
retention = {'experiment': 'METH479-R3 independent complete supervision retention', 'raw_sha256': RAW_SHA,
             'binding_sha256': BIND_SHA, 'previous_audit_failure_sha256': digest(PREVIOUS_AUDIT_FAILURE), 'previous_audit_inventory_sha256': digest(PREVIOUS_AUDIT_INVENTORY), 'audit_source_sha256': digest(__file__), 'audit_protocol_sha256': digest(PROTO),
             'windows_terminal_sha256': digest(event_path), 'original_failure_sha256': FAIL_SHA, 'first_inventory_sha256': INVENTORY_SHA, 'r1_failure_sha256': digest(R1_FAILURE), 'r1_inventory_sha256': digest(R1_INVENTORY), 'original_windows_terminal_sha256': digest(original_event_path), 'process_instance': instance,
             'start_utc': datetime.fromtimestamp(parent.create_time(), timezone.utc).isoformat(),
             'end_utc': datetime.now(timezone.utc).isoformat(), 'gates': gates, 'queries_audited': audited_queries,
             'unique_inputs_audited': audited_unique, 'target_scalar_fields_BYTE_checked': target_fields,
             'integer_fields_rederived': integer_fields, 'target_diagnostics': diagnostics,
             'ownership': report['ownership'], 'preserved_daemons': daemons, 'tracked_status': last_status,
             'decision': 'SUPERVISION_DATASET_ADMITTED_NOT_A_TRAINED_MODEL_OR_ACCURACY_RESULT',
             'scope': 'Original479 remains time-failed; ALL469+471 source128 consumed-domain joins/ownership/native support and mass targets; source full-score C BYTE admission retained, no original dot/native replay by audit; no fit/learnability/newartifact/fresh quality/rate/useful-n/LUT/DRAM/general-family claim',
             'native_model_replays': 0, 'training_updates': 0, 'SVD_refits': 0,
             'resource': {'seconds_before_RET': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed,
                          'hard_seconds': 600, 'hard_peak_bytes': 4 << 30, 'hard_new_bytes': 32 << 20}}
write(RET, retention)
guard()
checkpoint(terminal=True, retention_sha256=digest(RET), OS_peak_bytes=peak)
watchdog.cancel()
print(json.dumps({'retention': str(RET), 'sha256': digest(RET), 'gates': gates, 'unique_inputs': U,
                  'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()
