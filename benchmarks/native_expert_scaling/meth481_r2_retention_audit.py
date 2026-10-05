"""Independent complete METH481 algebra retention. No science helper imports."""
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
import threading
import traceback
import psutil

ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SOURCE_OUT = ROOT / 'results/native_expert_scaling/meth480_r1_learned_router'
SOURCE481_OUT = ROOT / 'results/native_expert_scaling/meth481_geometry'
OUT = ROOT / 'results/native_expert_scaling/meth481_r2_retention_audit'
RET = DOC / 'RETENTION_481_R2_20261005.json'
R2_PROTO = DOC / 'METH_481_R2_RETENTION_PROTOCOL_20261005.md'
R1_FAULT481 = DOC / 'meth481_r1_first_audit_fault_inventory.json'
R1_FAULT481_SHA = '9bad5d9a8aa9f5da55d82b35177416edf130fe61300c2a0dbaa291125555f563'
REPAIR481_PROTO = DOC / 'METH_481_R1_RETENTION_PROTOCOL_20261005.md'
FAULT481 = DOC / 'meth481_first_audit_fault_inventory.json'
FAULT481_SHA = '53471ad32f2d78f28ae06bbe4d47f83513902ce3896aa72d416e494cf2465da5'
GEO_RAW = DOC / 'meth481_geometry_result.json'
GEO_RAW_SHA = '73f2da6d08172d7e2e8c189cab55039d573bae27e64f3751a3fda89c62405787'
GEO_EVENT = ROOT / 'results/native_expert_scaling/meth481_windows_terminal.json'
GEO_EVENT_SHA = 'dd49377d781926c817c0c8f9a0c42b2cb3b84f44c1c01a9c7a0d0a3566cba6ee'
AUDIT481_PROTO = DOC / 'METH_481_RETENTION_PROTOCOL_20261005.md'
AUDIT481_WINDOWS = ROOT / 'benchmarks/native_expert_scaling/meth481_r2_audit_windows_terminal.ps1'
PRIOR_RET = DOC / 'RETENTION_480_R1_20261005.json'
PRIOR_RET_SHA = '059185c63d0df5f1e83795c0eb1bc965c779e7bfac08be34acc164e239d75f5f'
PROTOCOL481 = DOC / 'METH_481_GEOMETRY_PROTOCOL_20261005.md'
CALC481 = ROOT / 'benchmarks/native_expert_scaling/meth481_geometry_math.py'
WINDOWS481 = ROOT / 'benchmarks/native_expert_scaling/meth481_windows_terminal.ps1'
AUDIT_PROTO = DOC / 'METH_480_RETENTION_PROTOCOL_20261005.md'
AUDIT_REPAIR_PROTO = DOC / 'METH_480_R1_RETENTION_PROTOCOL_20261005.md'
AUDIT_FAULT_INVENTORY = DOC / 'meth480_first_audit_fault_inventory.json'
AUDIT_FAULT_INVENTORY_SHA = '65c8c2bb6e7431a398c01d42a343ef0e69f87f3287920a54cfd802c331fa3bb8'
AUDIT_WINDOWS = ROOT / 'benchmarks/native_expert_scaling/meth480_r1_audit_windows_terminal.ps1'
RAW_SHA = '9800b5635f700e18df6862afced186b644541155e87a64d1139a59f9bcc994a5'
EVENT = ROOT / 'results/native_expert_scaling/meth480_r1_windows_terminal.json'
EVENT_SHA = 'b05ba529d49d1941bb52d8f3defdbabf4d392c67e47c516ec6c72640ea654cac'
RAW = DOC / 'meth480_r1_learned_router_result.json'
BIND = DOC / 'meth480_prospective_bindings.json'
BIND_SHA = '2c33137557c5a958680633187b536dde37df75e12ce3882e0cff04330259579e'
C = ROOT / 'benchmarks/native_expert_scaling/meth480_learned_router.c'
MATH = ROOT / 'benchmarks/native_expert_scaling/meth480_transfer_math.py'
REPORTING = ROOT / 'benchmarks/native_expert_scaling/meth480_reporting.py'
WINDOWS = ROOT / 'benchmarks/native_expert_scaling/meth480_r1_windows_terminal.ps1'
PROTO = DOC / 'METH_480_LEARNED_ROUTER_PROTOCOL_20261005.md'
ARITHMETIC_NOTE = DOC / 'METH_480_STATIC_ARITHMETIC_CLARIFICATION_20261005.md'
REPAIR_PROTO = DOC / 'METH_480_R1_ADMISSION_REPAIR_PROTOCOL_20261005.md'
FIRST_INVENTORY = DOC / 'meth480_first_fault_inventory.json'
FIRST_INVENTORY_SHA = '7c3d688be6e070605fb59a5c9ade737df7748ee57906e170aa4f4803dcf2e21e'
parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
assert args.out.resolve() == RET.resolve() and not OUT.exists() and not RET.exists() and not RET.with_suffix('.failure.json').exists()
OUT.mkdir()
fatal = (OUT / 'fatal_native.log').open('xb')
faulthandler.enable(file=fatal, all_threads=True)
progress = (OUT / 'progress.jsonl').open('x', encoding='utf8')
parent = psutil.Process()
parent.cpu_affinity([0])
assert parent.cpu_affinity() == [0]
instance = {'pid': parent.pid, 'create_time_unix': parent.create_time(), 'executable': sys.executable}
start_utc = datetime.fromtimestamp(parent.create_time(), timezone.utc).isoformat()
phase = 'immutable_source_supervision_runtime_admission'
commands = []
peak = hashed = 0
cache = {}
scientific = (Path(__file__).resolve(), R2_PROTO, R1_FAULT481, DOC / 'RETENTION_481_R1_20261005.failure.json', DOC / 'METH_481_R1_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth481_r1_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth481_r1_audit_windows_terminal.ps1', REPAIR481_PROTO, FAULT481, DOC / 'RETENTION_481_20261005.failure.json', DOC / 'METH_481_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth481_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth481_audit_windows_terminal.ps1', AUDIT481_PROTO, AUDIT481_WINDOWS, ROOT / 'benchmarks/native_expert_scaling/meth481_geometry.py', GEO_RAW, DOC / 'METH_481_MAIN_TERMINAL_PENDING_20261005.md', PROTOCOL481, CALC481, WINDOWS481, PRIOR_RET, ROOT / 'benchmarks/native_expert_scaling/meth480_r1_retention_audit.py', AUDIT_PROTO, AUDIT_WINDOWS, AUDIT_REPAIR_PROTO, AUDIT_FAULT_INVENTORY, DOC / 'RETENTION_480_20261005.failure.json', DOC / 'METH_480_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth480_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth480_audit_windows_terminal.ps1', ROOT / 'benchmarks/native_expert_scaling/meth480_r1_learned_router.py', ROOT / 'benchmarks/native_expert_scaling/meth480_learned_router.py', C, MATH, REPORTING, WINDOWS, PROTO, ARITHMETIC_NOTE, REPAIR_PROTO, FIRST_INVENTORY, DOC / 'meth480_learned_router_result.failure.json', DOC / 'METH_480_FIRST_FAULT_20261005.md')

def process_gate():
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
    return daemons

def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n').encode())

def checkpoint(**value):
    progress.write(json.dumps({'phase': phase, 'seconds': time.monotonic() - START, 'pid': parent.pid, **value}) + '\n')
    progress.flush()

def guard(child=None):
    global peak
    m = parent.memory_info()
    value = max(m.rss, getattr(m, 'peak_wset', 0))
    if child is not None:
        try:
            proc = psutil.Process(child.pid)
            for p in (proc, *proc.children(recursive=True)):
                try:
                    m = p.memory_info()
                    value += max(m.rss, getattr(m, 'peak_wset', 0))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except psutil.NoSuchProcess:
            pass
    peak = max(peak, value)
    assert time.monotonic() - START <= 600 and peak <= 4 << 30, (phase, peak)
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) + (RET.stat().st_size if RET.exists() else 0) <= 32 << 20

def digest(path):
    global hashed
    p = Path(path).resolve()
    stat = p.stat()
    key = (str(p), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with p.open('rb') as stream:
            while data := stream.read(4 << 20):
                h.update(data)
                hashed += len(data)
                guard()
        cache[key] = h.hexdigest()
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

def failure(kind, value, tb):
    path = RET.with_suffix('.failure.json')
    if not path.exists():
        write(path, {'experiment': 'METH481-R2 FIRST independent retention audit fault', 'phase': phase,
              'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
              'commands': commands, 'source_binding_sha256': BIND_SHA, 'raw_sha256': GEO_RAW_SHA,
              'scientific_sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scientific},
              'traceback': ''.join(traceback.format_exception(kind, value, tb)),
              'resource': {'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak, 'file_bytes_hashed': hashed},
              'partial_output_inventory': [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()]})
    checkpoint(terminal_failure=True)
    sys.__excepthook__(kind, value, tb)

sys.excepthook = failure
def deadline():
    path = RET.with_suffix('.failure.json')
    if not path.exists():
        write(path, {'experiment': 'METH481-R2 retention watchdog FIRST fault', 'phase': phase, 'start_utc': start_utc,
                     'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
                     'commands': commands, 'source_binding_sha256': BIND_SHA, 'reason': '600 second deadline',
                     'resource': {'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak}})
    os._exit(3)
watchdog = threading.Timer(max(.001, 600 - (time.monotonic() - START)), deadline)
watchdog.daemon = True
watchdog.start()
checkpoint()
phase = 'EARLY_no_competing_process_admission'
early_daemons = process_gate()
phase = 'immutable_source_supervision_runtime_admission'
assert digest(AUDIT_FAULT_INVENTORY) == AUDIT_FAULT_INVENTORY_SHA
audit_fault = json.loads(AUDIT_FAULT_INVENTORY.read_bytes())
assert audit_fault['actual_returncode'] == 1 and audit_fault['numerical_imports_and_audited_fields'] == 0
for item in audit_fault['records']:
    check(item)
assert digest(FIRST_INVENTORY) == FIRST_INVENTORY_SHA
first_inventory = json.loads(FIRST_INVENTORY.read_bytes())
assert first_inventory['actual_returncode'] == 1 and first_inventory['numerical_imports_fit_compile_native_commands'] == 0
for item in [first_inventory['failure'], first_inventory['windows_terminal'], *first_inventory['outputs']]:
    check(item)
assert digest(BIND) == BIND_SHA
head(BIND)
assert digest(RAW) == RAW_SHA
head(RAW)
r = json.loads(RAW.read_bytes())
assert len(r['gates']) == 8 and all(r['gates'].values())
assert r['native_model_commands'] == 0 and r['native_numeric_commands'] == 3 and r['optimizer_updates'] == 24192
assert r['source_binding_sha256'] == BIND_SHA and r['steps_per_head'] == 32 and r['validation_training_examples'] == r['fallback_queries'] == 0
for path, value in r['scientific_sources'].items():
    head(path)
    assert digest(path) == value
for item in r['output_inventory']:
    check(item)
assert digest(EVENT) == EVENT_SHA
assert digest(PRIOR_RET) == PRIOR_RET_SHA
head(PRIOR_RET)
qualified = json.loads(PRIOR_RET.read_bytes())
assert qualified['raw_sha256'] == RAW_SHA and len(qualified['gates']) == 6 and all(qualified['gates'].values())
assert qualified['physical_dot_fields_BYTE_checked'] == 10032624 and qualified['saved_optimizer_updates_audited'] == 24192
for name, sha in [('benchmarks/native_expert_scaling/meth480_r1_retention_audit.py', qualified['audit_source_sha256']), ('docs/research/NATIVE_EXPERT_SCALING_20260925/METH_480_RETENTION_PROTOCOL_20261005.md', qualified['audit_protocol_sha256']), ('docs/research/NATIVE_EXPERT_SCALING_20260925/METH_480_R1_RETENTION_PROTOCOL_20261005.md', qualified['audit_repair_protocol_sha256'])]:
    head(ROOT / name)
    assert digest(ROOT / name) == sha
b = json.loads(BIND.read_bytes())
prior = b['prior']
check(b['prior_binding'])
for p in scientific:
    head(p)
for item in b['scientific_files'] + b['records'] + b['auxiliary'] + list(b['data_files'].values()):
    check(item)
for item in b['scientific_files'] + b['records']:
    head(item['path'])
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
runtime = prior['runtime']
assert sys.version == runtime['python'] and Path(sys.executable).resolve() == Path(runtime['executable']).resolve()
assert psutil.__version__ == runtime['packages']['psutil']['version'] and str(Path(psutil.__file__).resolve()) in runtime['packages']['psutil']['files']
for p, v in runtime['files'].items():
    check({'path': p, **v})
for package, v in runtime['packages'].items():
    assert importlib.metadata.version(package) == v['version']
    for p, record in v['files'].items():
        check({'path': p, **record})
for key in ('records', 'helpers', 'scientific_files', 'source_inventory', 'compile_assets', 'system_files', 'membership_sources'):
    for item in prior[key]:
        check(item)
for key in ('compiler', 'preparation_helper'):
    check(prior[key])
for item in prior['records'] + prior['helpers'] + prior['scientific_files']:
    head(item['path'])
artifact = prior['artifact']
assert digest(artifact['payload']) == artifact['sha256'] and Path(artifact['payload']).stat().st_size == artifact['bytes']
assert digest(artifact['manifest']) == artifact['manifest_sha256']
for rel, v in prior['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == v['sha256'] and (ROOT / rel).stat().st_size == v['bytes']
def status():
    rows = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines()
    assert all(v[:3] == ' M ' and v[3:] in prior['preserved_unrelated_files'] for v in rows), rows
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    return rows
status()
assert digest(R1_FAULT481) == R1_FAULT481_SHA
r1_fault481 = json.loads(R1_FAULT481.read_bytes())
assert r1_fault481['actual_returncode'] == 1 and not r1_fault481['numerical_imports'] and r1_fault481['dataset_scientific_fields_audited'] == r1_fault481['native_commands'] == 0
for item in r1_fault481['records']:
    check(item)
r1_failure481 = json.loads((DOC / 'RETENTION_481_R1_20261005.failure.json').read_bytes())
assert "assert event_instance['start_utc'] == g['start_utc']" in r1_failure481['traceback']
for path, sha in r1_failure481['scientific_sources'].items():
    head(path)
    assert digest(path) == sha
assert digest(FAULT481) == FAULT481_SHA
fault481 = json.loads(FAULT481.read_bytes())
assert fault481['actual_returncode'] == 1 and fault481['dataset_scientific_fields_audited'] == 0
assert fault481['numerical_imports'] and fault481['Decimal100_controls_completed'] == 4 and fault481['native_commands'] == 0
for item in fault481['records']:
    check(item)
old_fault481 = json.loads((DOC / 'RETENTION_481_20261005.failure.json').read_bytes())
assert "NameError: name 'g' is not defined" in old_fault481['traceback']
for path, sha in old_fault481['scientific_sources'].items():
    head(path)
    assert digest(path) == sha
marker481 = chr(10) + "phase = " + "'independent_controls_complete_wire_roles_and_ownership'"
original_suffix481 = (ROOT / 'benchmarks/native_expert_scaling/meth481_retention_audit.py').read_text(encoding='utf8').split(marker481, 1)[1]
repair_suffix481 = Path(__file__).read_text(encoding='utf8').split(marker481, 1)[1]
assert original_suffix481 == repair_suffix481
assert digest(GEO_RAW) == GEO_RAW_SHA and digest(GEO_EVENT) == GEO_EVENT_SHA
head(GEO_RAW)
g = json.loads(GEO_RAW.read_bytes())
assert len(g['gates']) == 5 and all(g['gates'].values())
assert g['source_binding_sha256'] == BIND_SHA and g['source480_RAW_sha256'] == RAW_SHA and g['source480_RET_sha256'] == PRIOR_RET_SHA
assert g['native_commands'] == g['training_updates'] == g['SVD_refits'] == 0 and g['commands'] == []
assert g['decision'] == 'SOURCE_ALGEBRA_COMPLETE_PENDING_INDEPENDENT_ADMISSION'
for path, sha in g['scientific_sources'].items():
    head(path)
    assert digest(path) == sha
for item in g['output_inventory']:
    check(item)
assert set(p.name for p in SOURCE481_OUT.iterdir()) == {'geometry.bin','controls.json','full_node_views.json','complete_reports.json','fatal_native.log','progress.jsonl'}
event = json.loads(GEO_EVENT.read_bytes())
assert event['query_available'] and event['query_error'] is None and event['event_id'] == 1000 and event['matching_scientific_events'] == []
assert len(event['instances']) == 1
event_instance = event['instances'][0]
assert event_instance['pid'] == g['process_instance']['pid'] and event_instance['create_time_unix'] == g['process_instance']['create_time_unix']
assert datetime.fromisoformat(event_instance['start_utc']) == datetime.fromisoformat(g['start_utc'])
assert datetime.fromisoformat(event_instance['end_utc']) == datetime.fromisoformat(g['end_utc'])
assert abs(datetime.fromisoformat(event['query_start_utc']).timestamp() - datetime.fromisoformat(g['start_utc']).timestamp() + 2) <= 1e-6
assert datetime.fromisoformat(event['query_end_utc']) > datetime.fromisoformat(g['end_utc'])
progress_rows = [json.loads(x) for x in (SOURCE481_OUT / 'progress.jsonl').read_text(encoding='utf8').splitlines()]
assert progress_rows[-1]['terminal'] and progress_rows[-1]['raw_sha256'] == GEO_RAW_SHA
assert all(x['pid'] == g['process_instance']['pid'] and x['seconds'] <= 600 for x in progress_rows)
assert [x['bank_complete'] for x in progress_rows if 'bank_complete' in x] == list(range(12))
assert g['resource']['hard_seconds'] == 600 and g['resource']['seconds_before_RAW'] <= 600
assert g['resource']['hard_peak_bytes'] == 4 << 30 and g['resource']['OS_peak_bytes'] <= 4 << 30
assert g['resource']['hard_new_bytes'] == 64 << 20
assert sum(p.stat().st_size for p in SOURCE481_OUT.iterdir()) + GEO_RAW.stat().st_size <= 64 << 20
assert (SOURCE481_OUT / 'fatal_native.log').stat().st_size == 0
daemons = process_gate()
ret = json.loads((DOC / 'RETENTION_479_R3_20261005.json').read_bytes())
assert all(ret['gates'].values()) and ret['unique_inputs_audited'] == 238872 and ret['queries_audited'] == 387036
assert time.monotonic() - START <= 180
gates = {'full_source6224_artifact_runtime_scientific_freezes_admitted479_supervision': True}
checkpoint(file_bytes_hashed=hashed)

phase = 'independent_controls_complete_wire_roles_and_ownership'
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
for module in (np, threadpoolctl):
    assert module.__version__ == runtime['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in runtime['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in runtime['packages']['numpy']['files']
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
U, Q = 238872, 387036
UNIQUE = np.dtype([('meta', '<u4', (12,)), ('input_sha', 'u1', (32,)), ('score_sha', 'u1', (32,)),
                   ('value', '<f8', (3,)), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
NODE = np.dtype([('maximum', '<f4', (2,)), ('winner', '<u2', (2,)), ('mass', '<f8', (2,))])
TARGET = np.dtype([('unique_id', '<u4'), ('nodes', NODE, (127,))])
PRED = np.dtype([('meta', '<u4', (12,)), ('value', '<f4', (3,)), ('root', '<f8', (4,)),
                 ('vector_ids', '<u2', (42,)), ('score', '<f4', (42,)), ('direction', '<u4', (7,))])
PATH = np.dtype([('node', '<u2'), ('left_winner', 'u1'), ('right_winner', 'u1'), ('margin', '<f8')])
GEO = np.dtype([('meta', '<u4', (4,)), ('moment', '<f8', (7,)), ('path', PATH, (7,))])
assert (UNIQUE.itemsize, TARGET.itemsize, PRED.itemsize, PATH.itemsize, GEO.itemsize) == (3720, 3560, 372, 12, 156)
differences = {}
report_integer_fields = report_float_fields = 0

def close(actual, expected, label):
    assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), label
    delta = np.abs(actual - expected)
    assert np.all(delta <= 1e-10 + 1e-9 * np.abs(expected)), (label, float(delta.max()))
    differences[label] = max(differences.get(label, 0.), float(delta.max()) if delta.size else 0.)

def compare(expected, actual, label):
    global report_integer_fields, report_float_fields
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), label
        for key in expected:
            compare(expected[key], actual[key], label + '/' + str(key))
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), label
        for i, (a, z) in enumerate(zip(expected, actual)):
            compare(a, z, label + '/' + str(i))
    elif isinstance(expected, float):
        assert isinstance(actual, (float, int)) and np.isfinite(actual), label
        delta = abs(expected - actual)
        assert delta <= 1e-10 + 1e-9 * abs(expected), (label, expected, actual)
        report_float_fields += 1
        differences['complete_reports'] = max(differences.get('complete_reports', 0.), delta)
    else:
        assert type(expected) is type(actual) and expected == actual, (label, expected, actual)
        report_integer_fields += isinstance(expected, (int, bool))

def independent_moments(scores, native):
    values = np.asarray(scores, dtype='<f8')
    mu = np.sum(values, axis=1, dtype='<f8') / 128
    centered = values - mu[:, None]
    c1 = np.sum(centered, axis=1, dtype='<f8') / 128
    variance = np.sum(np.square(centered), axis=1, dtype='<f8') / 128
    absolute = np.abs(centered)
    radius = np.max(absolute, axis=1)
    third = np.sum(np.power(absolute, 3), axis=1, dtype='<f8') / 128
    second = 1 + c1 + variance / 2
    assert np.all(second > 0)
    A2 = mu + np.log(128.) + np.log(second)
    maximum = np.max(values, axis=1)
    real_A = maximum + np.log(np.sum(np.exp(values - maximum[:, None]), axis=1, dtype='<f8'))
    native_A = native[:, 0] + np.log(native[:, 2])
    result = np.column_stack((mu, c1, variance, radius, third, A2, native_A - real_A))
    log_bound = np.full(len(values), -np.inf)
    nonzero = third > 0
    log_bound[nonzero] = radius[nonzero] + np.log(third[nonzero]) - np.log(6.) - np.minimum(c1[nonzero], np.log(second[nonzero]))
    gap = np.abs(A2 - real_A)
    assert np.all(gap[~nonzero] <= 1e-11)
    needs = nonzero & (gap > 1e-11)
    assert np.all(np.log(gap[needs] - 1e-11) <= log_bound[needs] + 1e-10)
    assert np.isfinite(result).all()
    return result, log_bound, real_A

toy = np.array([[3.] * 128, [-.01] * 64 + [.01] * 64, [20.] + [0.] * 127, [10020.] + [10000.] * 127], '<f4')
toy_native = np.zeros((4, 3), '<f8')
def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]
with localcontext() as context:
    context.prec = 100
    decimal_A = []
    for k, values in enumerate(toy):
        maximum = float(np.max(values))
        terms = [(Decimal(float(v)) - Decimal(maximum)).exp() for v in values]
        z = sum(f32(float(v)) for v in terms)
        toy_native[k] = (maximum, f32(1 / z), z)
        decimal_A.append(float(Decimal(maximum) + sum(terms).ln()))
toy_moment, toy_bound, toy_A = independent_moments(toy, toy_native)
assert np.all(np.abs(toy_A - decimal_A) <= 1e-10)
assert toy_moment[0, 2:5].tobytes() == np.zeros(3, '<f8').tobytes() and np.isneginf(toy_bound[0])
assert abs(toy_moment[1, 2] - float(toy[1, 0]) ** 2) <= 1e-15
assert abs(toy_moment[3, 5] - toy_moment[2, 5] - 10000) <= 1e-10
assert toy_native[2, 0] - toy_moment[2, 5] - np.log(toy_native[2, 1]) > np.log(1.01)
for D, eL, eR, expected in ((2., 10., 10., True), (.01, 0., -.02, False), (-2., 10., 10., True)):
    correct = (D + eR - eL > 0) == (D > 0)
    assert correct == expected and (abs(eR - eL) >= abs(D) or correct)
assert (3 < 5) != (8 < 1)
controls = {'Decimal100_centered_partition_controls': 4, 'constant_zero_remainder': True,
            'shift_invariance': True, 'symmetric_variance': True, 'large_residual_over_probability_witness': True,
            'differential_margin_controls': 3, 'source_vs_surrogate_tie_keys': True}
compare(controls, json.loads((SOURCE481_OUT / 'controls.json').read_bytes()), 'controls')
compare(controls, g['controls'], 'RAW_controls')

def wire(path, magic, width, reserved, count, dtype, shape):
    path = Path(path)
    assert path.stat().st_size == 24 + width * count
    with path.open('rb') as stream:
        assert stream.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
    return np.memmap(path, mode='r', dtype=dtype, offset=24, shape=shape)

unique = wire(b['data_files']['unique_inputs.bin']['path'], b'M479UNI1', 3720, 768, U, UNIQUE, (U,))
owner = wire(b['data_files']['ownership.bin']['path'], b'M479OWN1', 32, 0, U, '<u4', (U, 8))
links = wire(b['data_files']['query_links.bin']['path'], b'M479LNK1', 52, 0, Q, '<u4', (Q, 13))
targets = wire(b['data_files']['group_targets.bin']['path'], b'M479TGT1', 3560, 127, U, TARGET, (U,))
pred = wire(SOURCE_OUT / 'predictions.bin', b'M480PRD1', 372, 0, U, PRED, (U,))
actual_geo = wire(SOURCE481_OUT / 'geometry.bin', b'M481GEO1', 156, 12, U, GEO, (U,))
source_dtype = np.dtype([('meta', '<u4', (12,)), ('value', '<f8', (3,))])
source = wire(b['data_files']['source_fields.bin']['path'], b'M479SRC1', 72, 0, Q, source_dtype, (Q,))
assert np.array_equal(links[:, :12], source['meta']) and np.array_equal(links[:, 0], np.arange(Q))
assert np.all(links[:, 12] < U)
assert source['value'].tobytes() == unique['value'][links[:, 12]].tobytes()
assert np.array_equal(links[:, 5], np.where(links[:, 2] < 64, 0, np.where(links[:, 2] < 128, 1, 2)))
for record in (unique['meta'][:, 0], owner[:, 0], targets['unique_id'], pred['meta'][:, 0], actual_geo['meta'][:, 0]):
    assert np.array_equal(record, np.arange(U))
assert np.all(unique['meta'][:, 1] == 128)
assert np.array_equal(links[:, 6], unique['meta'][links[:, 12], 2])
assert np.array_equal(links[:, 9], unique['meta'][links[:, 12], 7])
expected_owner = np.zeros((U, 8), '<u4')
expected_owner[:, 0], expected_owner[:, 1] = np.arange(U), unique['meta'][:, 2]
for column, source_column in ((2, 5), (3, 4), (4, 10)):
    np.bitwise_or.at(expected_owner[:, column], links[:, 12], (1 << links[:, source_column]).astype('<u4'))
np.add.at(expected_owner[:, 5], links[:, 12], 1)
for book in range(192):
    ids = np.unique(links[links[:, 2] == book, 12])
    expected_owner[ids, 7 if 64 <= book < 128 else 6] += 1
assert expected_owner.tobytes() == owner.tobytes()
dev, val = (owner[:, 2] & 5) != 0, (owner[:, 2] & 2) != 0
assert np.count_nonzero(dev) == 159414 and np.count_nonzero(val) == 79458 and not np.any(dev & val)
memberships = json.loads(Path(b['data_files']['membership.json']['path']).read_bytes())
assert [x['bank'] for x in memberships] == list(range(12))
gates['independent_Decimal100_controls_ALL_wires_roles_links_and_ownership_BYTES'] = True
checkpoint()

phase = 'independent_ALL_source_maxima_ID_margins_moments_and_visited_paths'
expected_moment = np.empty((U, 7), '<f8')
all_bound = np.empty(U, '<f8')
expected_first = np.full(U, 2**32 - 1, '<u4')
metrics = {key: np.zeros((U, 7), bool if key in ('wrong', 'sufficient', 'source_tie', 'candidate_tie') else '<f8')
           for key in ('wrong', 'sufficient', 'source_tie', 'candidate_tie', 'differential', 'common')}
node_views = []
source_margins = visited_margins = path_BYTE_fields = moment_fields = 0
for bank in range(12):
    uid = np.flatnonzero(unique['meta'][:, 2] == bank)
    row, t, p, a = unique[uid], targets[uid], pred[uid], actual_geo[uid]
    count = len(uid)
    assert count == (22272 if bank < 6 else 17540)
    tree = memberships[bank]['nodes']
    assert len(tree) == 255 and tree[0]['count'] == 128
    assert [x['node'] for x in tree] == list(range(255))
    internal = [x for x in tree if x['count'] > 1]
    assert len(internal) == 127
    node_index = np.full(255, -1, '<i4')
    left_max = np.empty((count, 127), '<f4')
    right_max = np.empty_like(left_max)
    left_winner = np.empty((count, 127), '<u2')
    right_winner = np.empty_like(left_winner)
    score = row['score']
    assert np.isfinite(score).all()
    assert np.array_equal(np.argmax(score, axis=1), row['meta'][:, 7])
    assert score[np.arange(count), row['meta'][:, 7]].astype('<f8').tobytes() == row['value'][:, 0].tobytes()
    for k, node in enumerate(internal):
        node_index[node['node']] = k
        child_ids = [sorted(tree[node[side]]['ids']) for side in ('left', 'right')]
        assert set(child_ids[0]).isdisjoint(child_ids[1])
        assert sorted(child_ids[0] + child_ids[1]) == sorted(node['ids'])
        for side, ids in enumerate(child_ids):
            values = score[:, ids]
            index = np.argmax(values, axis=1)
            maxima = values[np.arange(count), index]
            winner = np.asarray(ids, '<u2')[index]
            assert maxima.tobytes() == t['nodes']['maximum'][:, k, side].tobytes()
            assert winner.tobytes() == t['nodes']['winner'][:, k, side].tobytes()
            (left_max if side == 0 else right_max)[:, k] = maxima
            (left_winner if side == 0 else right_winner)[:, k] = winner
        D = right_max[:, k].astype('<f8') - left_max[:, k].astype('<f8')
        source_right = (D > 0) | ((D == 0) & (right_winner[:, k] < left_winner[:, k]))
        winner = np.where(source_right, right_winner[:, k], left_winner[:, k])
        path = np.isin(row['meta'][:, 7], node['ids'])
        for role, mask in (('ALL', np.ones(count, bool)), ('development', dev[uid]), ('consumed_validation', val[uid])):
            absolute = np.abs(D[mask])
            n = len(absolute)
            assert n
            bins = [int(np.count_nonzero(absolute == 0))]
            edges = (0., .001, .01, .1, 1.)
            bins.extend(int(np.count_nonzero((absolute > lo) & (absolute <= hi))) for lo, hi in zip(edges, edges[1:]))
            bins.append(int(np.count_nonzero(absolute > 1)))
            node_views.append({'bank': bank, 'node': node['node'], 'role': role, 'count': n,
                               'source_right_count': int(np.count_nonzero(source_right[mask])), 'source_ties': bins[0],
                               'absolute_margin_mean': float(np.sum(absolute, dtype='<f8') / n),
                               'absolute_margin_min': float(np.min(absolute)), 'absolute_margin_max': float(np.max(absolute)),
                               'absolute_margin_p95': float(np.sort(absolute)[(95 * n + 99) // 100 - 1]),
                               'absolute_margin_bins_0_1e3_1e2_1e1_1_inf': bins,
                               'local_winner_ID_counts': np.bincount(winner[mask].astype('<i8'), minlength=128).tolist(),
                               'global_winner_path_count': int(np.count_nonzero(path & mask))})
        source_margins += count
    derived, log_bound, real_A = independent_moments(score, row['value'])
    close(a['moment'], derived, 'seven_moment_fields')
    expected_moment[uid], all_bound[uid] = derived, log_bound
    moment_fields += derived.size
    expected_path = np.zeros((count, 7), PATH)
    first = np.full(count, 2**32 - 1, '<u4')
    current = np.zeros(count, '<u4')
    slot = 0
    rows = np.arange(count)
    for depth in range(7):
        k = node_index[current]
        assert np.all(k >= 0)
        ML = left_max[rows, k].astype('<f8')
        MR = right_max[rows, k].astype('<f8')
        LW, RW = left_winner[rows, k], right_winner[rows, k]
        D = MR - ML
        source_right = (D > 0) | ((D == 0) & (RW < LW))
        forms = 1 if depth == 6 else 2
        scores_L, scores_R = p['score'][:, slot:slot + forms], p['score'][:, slot + forms:slot + 2 * forms]
        AL, AR = scores_L.max(axis=1).astype('<f8'), scores_R.max(axis=1).astype('<f8')
        next_L = np.array([tree[int(node)]['left'] for node in current], '<u4')
        next_R = np.array([tree[int(node)]['right'] for node in current], '<u4')
        if depth >= 5:
            # Last two levels use actual original coefficients and original ID keys.
            key_L = p['vector_ids'][:, slot:slot + forms][rows, np.argmax(scores_L, axis=1)]
            key_R = p['vector_ids'][:, slot + forms:slot + 2 * forms][rows, np.argmax(scores_R, axis=1)]
            assert np.all(key_L < 128) and np.all(key_R < 128)
        else:
            key_L = np.array([tree[int(node)]['first'] for node in next_L], '<u4')
            key_R = np.array([tree[int(node)]['first'] for node in next_R], '<u4')
        chosen_right = (AR > AL) | ((AR == AL) & (key_R < key_L))
        assert np.array_equal(chosen_right.astype('<u4'), p['direction'][:, depth])
        differential = AR - AL - D
        wrong = chosen_right != source_right
        sufficient = (D != 0) & (np.abs(differential) < np.abs(D))
        assert not np.any(wrong & sufficient)
        alive = first == 2**32 - 1
        contains = np.array([int(row['meta'][j, 7]) in tree[int(node)]['ids'] for j, node in enumerate(current)])
        assert np.array_equal(alive, contains)
        first[alive & wrong] = depth
        expected_path['node'][:, depth] = current
        expected_path['left_winner'][:, depth], expected_path['right_winner'][:, depth] = LW, RW
        expected_path['margin'][:, depth] = D
        for name, value in (('wrong', wrong), ('sufficient', sufficient), ('source_tie', D == 0),
                            ('candidate_tie', AL == AR), ('differential', differential), ('common', ((AL - ML) + (AR - MR)) / 2)):
            metrics[name][uid, depth] = value
        current = np.where(chosen_right, next_R, next_L)
        slot += forms * 2
        visited_margins += count
    assert slot == 26
    chosen = np.array([tree[int(node)]['first'] for node in current], '<u4')
    assert chosen.tobytes() == p['meta'][:, 2].tobytes()
    assert np.array_equal(first == 2**32 - 1, chosen == row['meta'][:, 7])
    expected_meta = np.column_stack((uid, np.full(count, bank), row['meta'][:, 7], first)).astype('<u4')
    assert expected_meta.tobytes() == a['meta'].tobytes()
    assert expected_path.tobytes() == a['path'].tobytes()
    expected_first[uid] = first
    path_BYTE_fields += count * (4 + 7 * 4)
    del row, t, p, a, derived, real_A, left_max, right_max, left_winner, right_winner, score, expected_path
    guard()
    checkpoint(bank_geometry_admitted=bank, source_margins=source_margins, visited_margins=visited_margins)
assert source_margins == U * 127 == 30336744 and visited_margins == U * 7 == 1672104
assert moment_fields == U * 7 and path_BYTE_fields == U * 32
assert len(node_views) == 4572 and sum(x['count'] for x in node_views if x['role'] == 'ALL') == source_margins
assert all(sum(x['local_winner_ID_counts']) == x['count'] and sum(x['absolute_margin_bins_0_1e3_1e2_1e1_1_inf']) == x['count'] for x in node_views)
compare(node_views, json.loads(Path(g['node_views_path']).read_bytes()), 'ALL_source_node_views')
gates['ALL_source_score_maxima_ID_BYTES_30336744_margins_4572_node_views'] = True
gates['ALL156B_UID_fields_moments_tie_paths_1672104_visited_margins'] = True

phase = 'independent_complete_occurrence_book_ID_reports_and_decisions'
native_A = unique['value'][:, 0] + np.log(unique['value'][:, 2])
log_ratio = unique['value'][:, 0] - expected_moment[:, 5] - np.log(unique['value'][:, 1])
log_low, log_high = np.log(.99), np.log(1.01)
allowed = min(-log_low, log_high) - np.abs(expected_moment[:, 6])
screen = np.zeros(U, bool)
positive = allowed > 0
screen[positive] = all_bound[positive] <= np.log(allowed[positive])
abs_error = np.abs(expected_moment[:, 5] - native_A)

def summarize(ids):
    n = len(ids)
    result = {'count': n, 'moment_log_ratio_outside_1percent': int(np.count_nonzero((log_ratio[ids] < log_low) | (log_ratio[ids] > log_high))),
              'moment_over_probability_witnesses': int(np.count_nonzero(log_ratio[ids] > log_high)),
              'moment_probability_above_1': int(np.count_nonzero(unique['value'][ids, 0] > expected_moment[ids, 5])),
              'nominal_Taylor_screen_count': int(np.count_nonzero(screen[ids])),
              'first_divergence_counts': np.bincount(expected_first[ids][expected_first[ids] < 7].astype('<i8'), minlength=7).tolist(),
              'visited_wrong_branches': int(np.count_nonzero(metrics['wrong'][ids])),
              'visited_strict_sufficient': int(np.count_nonzero(metrics['sufficient'][ids])),
              'visited_source_ties': int(np.count_nonzero(metrics['source_tie'][ids])),
              'visited_candidate_ties': int(np.count_nonzero(metrics['candidate_tie'][ids]))}
    fields = ('mean_absolute_native_A_error', 'max_absolute_native_A_error', 'max_absolute_native_rounding',
              'variance_mean', 'radius_max', 'max_log_Taylor_bound', 'max_absolute_differential_error', 'mean_absolute_common_support_error')
    if not n:
        return {**result, **{name: None for name in fields}}
    finite_bounds = all_bound[ids][np.isfinite(all_bound[ids])]
    result.update(mean_absolute_native_A_error=float(np.sum(abs_error[ids], dtype='<f8') / n),
                  max_absolute_native_A_error=float(np.max(abs_error[ids])),
                  max_absolute_native_rounding=float(np.max(np.abs(expected_moment[ids, 6]))),
                  variance_mean=float(np.sum(expected_moment[ids, 2], dtype='<f8') / n), radius_max=float(np.max(expected_moment[ids, 3])),
                  max_log_Taylor_bound=float(finite_bounds.max()) if len(finite_bounds) else None,
                  max_absolute_differential_error=float(np.max(np.abs(metrics['differential'][ids]))),
                  mean_absolute_common_support_error=float(np.sum(np.abs(metrics['common'][ids]), dtype='<f8') / (n * 7)))
    return result

reports = {'unique_views': [], 'occurrence_views': [], 'book_views': [], 'source_ID_views': []}
for bank in [-1, *range(12)]:
    eligible = np.ones(U, bool) if bank < 0 else unique['meta'][:, 2] == bank
    for role, mask in (('ALL', np.ones(U, bool)), ('development', dev), ('consumed_validation', val)):
        reports['unique_views'].append({'bank': bank, 'role': role, **summarize(np.flatnonzero(mask & eligible))})
for bank in range(12):
    for role in range(3):
        for mode in range(2):
            rows = links[(links[:, 6] == bank) & (links[:, 5] == role) & (links[:, 4] == mode)]
            reports['occurrence_views'].append({'bank': bank, 'role': role, 'mode': mode, 'accepted': int(rows[:, 10].sum(dtype='<u8')),
                                              'rejected': int(np.count_nonzero(rows[:, 10] == 0)), **summarize(rows[:, 12])})
            for expert in range(128):
                reports['source_ID_views'].append({'bank': bank, 'role': role, 'mode': mode, 'source_ID': expert,
                                                  **summarize(rows[rows[:, 9] == expert, 12])})
for book in range(192):
    for bank in range(12):
        for mode in range(2):
            rows = links[(links[:, 2] == book) & (links[:, 6] == bank) & (links[:, 4] == mode)]
            reports['book_views'].append({'book': book, 'bank': bank, 'mode': mode, 'role': 0 if book < 64 else 1 if book < 128 else 2,
                                         'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)), **summarize(rows[:, 12])})
    if book % 32 == 31:
        guard()
assert [len(reports[k]) for k in reports] == [39, 72, 4608, 9216]
assert all(sum(x['count'] for x in reports[key]) == Q for key in ('occurrence_views', 'book_views', 'source_ID_views'))
compare(reports, json.loads(Path(g['reports_path']).read_bytes()), 'complete_reports')
overall = reports['unique_views'][0]
compare(overall, g['overall'], 'RAW_overall')
assert sum(overall['first_divergence_counts']) == 201993 == r['unique_overall']['ID_differences']
decisions = {'full_score_second_moment_reference_within_1percent_every_UID': overall['moment_log_ratio_outside_1percent'] == 0,
             'nominal_Taylor_plus_observed_native_rounding_screen_every_UID': overall['nominal_Taylor_screen_count'] == U,
             'over_probability_witness_blocks_plain_PSD_reduction_in_teacher_score_coordinates': overall['moment_over_probability_witnesses'] > 0}
compare(decisions, g['decisions'], 'RAW_decisions')
assert (g['unique_inputs'], g['queries'], g['source_node_margins'], g['visited_node_margins'], g['geometry_scalar_fields'], g['geometry_wire_width']) == (U, Q, U * 127, U * 7, U * 39, 156)
compare({'forms_with_separate_affine_decision_and_branch_mass': 14, 'coefficients_at_768': 10752, 'exp_calls': 7,
         'logical_vector_bytes': 43008, 'gate_header_bytes_7x32': 224, 'complete_budget_not_measured': True}, g['inference_budget_hypothesis'], 'budget_hypothesis_only')
gates['ALL39_72_4608_9216_reports_empty_IDs_denominators_and_separate_decisions'] = True
for rel, item in prior['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256'] and (ROOT / rel).stat().st_size == item['bytes']
last_status = status()
assert parent.cpu_affinity() == [0]
assert not ({'torch', 'scipy', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'meth481_geometry', 'meth481_geometry_math',
             'meth480_learned_router', 'meth480_r1_learned_router', 'meth480_transfer_math', 'meth480_reporting',
             'meth480_retention_audit', 'meth480_r1_retention_audit'} & set(sys.modules))
gates['resource_preservation_frozen_runtime_no_imported_science_native_fit_SVD_or_sampling'] = True
guard()
retention = {'experiment': 'METH481 ONE complete independent source geometry retention', 'raw_sha256': GEO_RAW_SHA,
             'source_binding_sha256': BIND_SHA, 'source480_RAW_sha256': RAW_SHA, 'source480_RET_sha256': PRIOR_RET_SHA,
             'windows_terminal_sha256': GEO_EVENT_SHA, 'audit_source_sha256': digest(__file__),
             'audit_protocol_sha256': digest(AUDIT481_PROTO), 'audit_windows_source_sha256': digest(AUDIT481_WINDOWS),
             'process_instance': instance, 'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(),
             'commands': [], 'gates': gates, 'decisions': decisions, 'decision': 'SOURCE_ALGEBRA_COMPLETE_INDEPENDENTLY_ADMITTED',
             'unique_inputs_audited': U, 'queries_audited': Q, 'source_node_margins_rederived_from_original_scores': source_margins,
             'visited_node_margins_audited': visited_margins, 'geometry_meta_and_path_scalar_fields_BYTE_checked': path_BYTE_fields,
             'geometry_moment_F64_fields_rederived': moment_fields, 'full_geometry_scalar_fields_audited': U * 39,
             'source_maximum_and_winner_fields_BYTE_checked': U * 127 * 4,
             'report_integer_fields_rederived': report_integer_fields, 'report_float_fields_rederived': report_float_fields,
             'complete_view_counts': [39, 72, 4608, 9216, 4572], 'maximum_absolute_audit_differences': differences,
             'numerical_tolerances': {'F64_absolute': 1e-10, 'F64_relative': 1e-9, 'meta_tie_ID_and_margin': 'BYTE', 'integer_null_schema': 'EXACT'},
             'overall': overall, 'native_replays': 0, 'training_updates': 0, 'SVD_refits': 0, 'sampled_fields': 0,
             'preserved_daemons': daemons, 'tracked_status': last_status,
             'scope': 'Complete consumed source-score algebra. Oracle moment reference is not cheap inference; nominal Taylor screening is not an interval certificate. No fresh wholeartifact/quality/rate/useful-n/LUT/DRAM/family claim.',
             'resource': {'seconds_before_RET': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed,
                          'hard_seconds': 600, 'hard_peak_bytes': 4 << 30, 'hard_new_bytes': 32 << 20}}
assert len(gates) == 6 and all(gates.values())
write(RET, retention)
guard()
checkpoint(terminal=True, retention_sha256=digest(RET), OS_peak_bytes=peak)
watchdog.cancel()
print(json.dumps({'retention': str(RET), 'sha256': digest(RET), 'gates': gates, 'decisions': decisions,
                  'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()

