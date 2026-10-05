"""ONE complete original-source root affine maximum-margin inquiry."""
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
OUT = ROOT / 'results/native_expert_scaling/meth482_affine_root'
RET = DOC / 'meth482_affine_root_result.json'
PROTO482 = DOC / 'METH_482_AFFINE_ROOT_PROTOCOL_20261005.md'
CALC482 = ROOT / 'benchmarks/native_expert_scaling/meth482_affine_math.py'
WINDOWS482 = ROOT / 'benchmarks/native_expert_scaling/meth482_windows_terminal.ps1'
SCI_RUNTIME = DOC / 'meth482_runtime_binding.json'
SCI_RUNTIME_SHA = 'd506a86753ea0f1e1a16253461e07811602272dfdffafa4b45e13567381f17ad'
RET481 = DOC / 'RETENTION_481_R2_20261005.json'
RET481_SHA = 'a8573719fa6fc34fa91be6844dfa94068be47867d62a8c44d3a87ccbd2821111'
EVENT481_AUDIT = ROOT / 'results/native_expert_scaling/meth481_r2_audit_windows_terminal.json'
EVENT481_AUDIT_SHA = 'fd7066ade489b641010b438d91c282d92c10a3cb198af4b4e3036de87f2090e7'
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
scientific = (Path(__file__).resolve(), PROTO482, CALC482, WINDOWS482, SCI_RUNTIME, RET481, ROOT / 'benchmarks/native_expert_scaling/meth481_r2_retention_audit.py', R2_PROTO, R1_FAULT481, DOC / 'RETENTION_481_R1_20261005.failure.json', DOC / 'METH_481_R1_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth481_r1_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth481_r1_audit_windows_terminal.ps1', REPAIR481_PROTO, FAULT481, DOC / 'RETENTION_481_20261005.failure.json', DOC / 'METH_481_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth481_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth481_audit_windows_terminal.ps1', AUDIT481_PROTO, AUDIT481_WINDOWS, ROOT / 'benchmarks/native_expert_scaling/meth481_geometry.py', GEO_RAW, DOC / 'METH_481_MAIN_TERMINAL_PENDING_20261005.md', PROTOCOL481, CALC481, WINDOWS481, PRIOR_RET, ROOT / 'benchmarks/native_expert_scaling/meth480_r1_retention_audit.py', AUDIT_PROTO, AUDIT_WINDOWS, AUDIT_REPAIR_PROTO, AUDIT_FAULT_INVENTORY, DOC / 'RETENTION_480_20261005.failure.json', DOC / 'METH_480_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth480_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth480_audit_windows_terminal.ps1', ROOT / 'benchmarks/native_expert_scaling/meth480_r1_learned_router.py', ROOT / 'benchmarks/native_expert_scaling/meth480_learned_router.py', C, MATH, REPORTING, WINDOWS, PROTO, ARITHMETIC_NOTE, REPAIR_PROTO, FIRST_INVENTORY, DOC / 'meth480_learned_router_result.failure.json', DOC / 'METH_480_FIRST_FAULT_20261005.md')

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
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) + (RET.stat().st_size if RET.exists() else 0) <= 64 << 20

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
        write(path, {'experiment': 'METH482 FIRST affine inquiry fault', 'phase': phase,
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
        write(path, {'experiment': 'METH482 affine watchdog FIRST fault', 'phase': phase, 'start_utc': start_utc,
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
assert digest(RET481) == RET481_SHA and digest(EVENT481_AUDIT) == EVENT481_AUDIT_SHA
head(RET481)
ret481 = json.loads(RET481.read_bytes())
assert len(ret481['gates']) == 6 and all(ret481['gates'].values())
assert ret481['raw_sha256'] == GEO_RAW_SHA and ret481['full_geometry_scalar_fields_audited'] == 9316008
assert digest(ROOT / 'benchmarks/native_expert_scaling/meth481_r2_retention_audit.py') == ret481['audit_source_sha256']
audit481event = json.loads(EVENT481_AUDIT.read_bytes())
assert audit481event['query_available'] and audit481event['query_error'] is None and not audit481event['matching_scientific_events']
assert len(audit481event['instances']) == 1
audit481instance = audit481event['instances'][0]
assert audit481instance['pid'] == ret481['process_instance']['pid'] and audit481instance['create_time_unix'] == ret481['process_instance']['create_time_unix']
for key in ('start_utc','end_utc'):
    assert datetime.fromisoformat(audit481instance[key]) == datetime.fromisoformat(ret481[key])
assert digest(SCI_RUNTIME) == SCI_RUNTIME_SHA
head(SCI_RUNTIME)
sci_runtime = json.loads(SCI_RUNTIME.read_bytes())
assert sci_runtime['python'] == sys.version and Path(sci_runtime['executable']).resolve() == Path(sys.executable).resolve()
assert importlib.metadata.version('scipy') == sci_runtime['version']
for item in sci_runtime['files']:
    check(item)
daemons = process_gate()
ret = json.loads((DOC / 'RETENTION_479_R3_20261005.json').read_bytes())
assert all(ret['gates'].values()) and ret['unique_inputs_audited'] == 238872 and ret['queries_audited'] == 387036
assert time.monotonic() - START <= 180
gates = {'full_source6224_artifact_runtime_scientific_freezes_admitted479_supervision': True}
checkpoint(file_bytes_hashed=hashed)

phase = 'frozen_solver_interval_controls_and_source_wire_admission'
import numpy as np
import scipy
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
assert scipy.__version__ == sci_runtime['version']
scipy_files = {item['path'] for item in sci_runtime['files']}
assert str(Path(scipy.__file__).resolve()) in scipy_files
for module in (np, threadpoolctl):
    assert module.__version__ == runtime['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in runtime['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1
    assert str(Path(pool['filepath']).resolve()) in set(runtime['packages']['numpy']['files']) | scipy_files
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
import meth482_affine_math as M
assert Path(M.__file__).resolve() == CALC482
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1
    assert str(Path(pool['filepath']).resolve()) in set(runtime['packages']['numpy']['files']) | scipy_files

def subnormal_control():
    smallest = np.array([np.nextafter(np.float64(0.), np.float64(1.))], '<f8')
    assert smallest.tobytes() == struct.pack('<Q', 1)
    assert (smallest * np.float64(1.)).tobytes() == struct.pack('<Q', 1)
    assert (smallest + smallest).tobytes() == struct.pack('<Q', 2)

subnormal_control()
toy_outcomes = []
for values, labels, expected in (([-1., 1.], [-1, 1], 1.), ([-1., 0., 1.], [1, -1, 1], 0.), ([-1., 0., 1.], [1, 1, 1], 1.)):
    x, y = np.array(values, '<f4')[:, None], np.array(labels, '<i1')
    result, raw_x, raw_dual, theta, theta32, weights = M.solve(x, y)
    assert result['status'] == 0 and abs(-result['fun'] - expected) <= 1e-8
    lo, hi, sum_lo, sum_hi, bound = M.dual_interval(x, y, weights)
    assert bound is not None and expected <= bound + 1e-12
    assert bound - expected <= 1e-8
    physical = M.physical_intervals(x, theta32)
    if expected > 0:
        signed = np.where(y > 0, physical[2], -physical[3])
        assert np.all(signed > 0)
    toy_outcomes.append({'status': result['status'], 'expected_optimum': expected, 'dual_upper': bound, 'warnings': result['warnings']})
subnormal_control()
cancel_x = np.array([[1e20, 1., -1e20]], '<f4')
cancel_theta = np.array([1., 1., 1., 0.], '<f8')
lo, hi, _ = M.dot_intervals(cancel_x, cancel_theta)
with localcontext() as context:
    context.prec = 100
    exact = sum(Decimal(float(v)) for v in cancel_x[0])
    assert Decimal(float(lo[0])) <= exact <= Decimal(float(hi[0]))
controls = {'solver_analytic_controls': toy_outcomes, 'Decimal100_cancellation_enclosed': True,
            'IEEE_F64_subnormal_arithmetic': True, 'control_LP_calls': 3, 'physical_contract_requires_FTZ_DAZ_off': True}
write(OUT / 'controls.json', controls)
U, Q = 238872, 387036
UNIQUE = np.dtype([('meta', '<u4', (12,)), ('input_sha', 'u1', (32,)), ('score_sha', 'u1', (32,)),
                   ('value', '<f8', (3,)), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
NODE = np.dtype([('maximum', '<f4', (2,)), ('winner', '<u2', (2,)), ('mass', '<f8', (2,))])
TARGET = np.dtype([('unique_id', '<u4'), ('nodes', NODE, (127,))])

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
for value in (unique['meta'][:, 0], owner[:, 0], targets['unique_id']):
    assert np.array_equal(value, np.arange(U))
assert np.array_equal(owner[:, 1], unique['meta'][:, 2])
assert np.array_equal(links[:, 6], unique['meta'][links[:, 12], 2]) and np.array_equal(links[:, 9], unique['meta'][links[:, 12], 7])
dev, val = (owner[:, 2] & 5) != 0, (owner[:, 2] & 2) != 0
assert np.count_nonzero(dev) == 159414 and np.count_nonzero(val) == 79458 and not np.any(dev & val)
memberships = json.loads(Path(b['data_files']['membership.json']['path']).read_bytes())
assert [x['bank'] for x in memberships] == list(range(12))
gates['analytic_solver_Decimal_interval_controls_all_UID_roles_source_wire_admitted'] = True
checkpoint()

phase = 'ALL12_development_only_root_LPs_primal_dual_and_every_UID_intervals'
certificate = np.zeros(U, M.CERT)
records = []
for bank in range(12):
    uid = np.flatnonzero(unique['meta'][:, 2] == bank)
    row, target = unique[uid], targets[uid]
    assert len(uid) == (22272 if bank < 6 else 17540)
    tree = memberships[bank]['nodes']
    assert tree[0]['node'] == 0 and tree[0]['count'] == 128
    root = tree[0]
    groups = [sorted(tree[root[side]]['ids']) for side in ('left', 'right')]
    maxima, winner = [], []
    for side, ids in enumerate(groups):
        values = row['score'][:, ids]
        at = np.argmax(values, axis=1)
        maxima.append(values[np.arange(len(uid)), at])
        winner.append(np.array(ids, '<u2')[at])
        assert maxima[-1].tobytes() == target['nodes']['maximum'][:, 0, side].tobytes()
        assert winner[-1].tobytes() == target['nodes']['winner'][:, 0, side].tobytes()
    right = (maxima[1] > maxima[0]) | ((maxima[1] == maxima[0]) & (winner[1] < winner[0]))
    assert np.array_equal(right, np.isin(row['meta'][:, 7], groups[1]))
    labels = (2 * right.astype('<i1') - 1).astype('<i1')
    local_dev = np.flatnonzero(dev[uid])
    x = row['input']
    checkpoint(bank_LP_start=bank, development_examples=len(local_dev))
    guard()
    lp_start = time.monotonic()
    solver, raw_x, raw_dual, theta, theta32, weights = M.solve(x[local_dev], labels[local_dev])
    lp_seconds = time.monotonic() - lp_start
    subnormal_control()
    guard()
    dual_lo, dual_hi, sum_lo, sum_hi, dual_bound = M.dual_interval(x[local_dev], labels[local_dev], weights)
    envelope, B = M.uniform_envelope(x[local_dev])
    ideal_lo, ideal_hi, _ = M.dot_intervals(x, theta)
    exact_lo, exact_hi, physical_lo, physical_hi, absolute, error = M.physical_intervals(x, theta32)
    fields = np.column_stack((ideal_lo, ideal_hi, exact_lo, exact_hi, physical_lo, physical_hi, absolute, error))
    assert np.isfinite(fields).all()
    certificate['meta'][uid] = np.column_stack((uid, np.full(len(uid), bank), row['meta'][:, 7], right,
                                              maxima[0] == maxima[1], np.full(len(uid), solver['available']))).astype('<u4')
    certificate['value'][uid] = fields
    signed = np.where(labels > 0, physical_lo, -physical_hi)
    ideal_signed = np.where(labels > 0, ideal_lo, -ideal_hi)
    model = OUT / ('bank' + str(bank) + '_witness.npz')
    np.savez(model, development_UIDs=uid[local_dev].astype('<u4'), development_labels=labels[local_dev],
             raw_primal=raw_x, raw_dual=raw_dual, theta=theta, theta_F32=theta32, dual_weights=weights,
             dual_residual_lower=dual_lo, dual_residual_upper=dual_hi, dual_weight_sum_bounds=np.array([sum_lo, sum_hi]),
             dual_bound=np.array([dual_bound if dual_bound is not None else 0.]), dual_available=np.array([dual_bound is not None], '<u4'))
    dev_pass = solver['available'] and bool(np.all(signed[local_dev] > 0))
    all_pass = solver['available'] and bool(np.all(signed > 0))
    uniform_unattainable = dual_bound is not None and dual_bound <= envelope
    if dual_bound is not None:
        lower_candidate = max(0., float(ideal_signed[local_dev].min()))
        assert lower_candidate <= dual_bound
    record = {'bank': bank, 'unique_queries': len(uid), 'development_examples': len(local_dev),
              'consumed_validation_examples': int(np.count_nonzero(val[uid])), 'solver': solver, 'LP_seconds': lp_seconds,
              'model_path': str(model), 'dual_unit_L1_margin_upper': dual_bound, 'uniform_F32_export_and_arithmetic_envelope': envelope,
              'development_max_augmented_coordinate': B, 'candidate_unit_L1_norm_upper': M.sum_upper_nonnegative(np.abs(theta)),
              'candidate_development_min_ideal_margin_lower': float(ideal_signed[local_dev].min()) if solver['available'] else None,
              'every_development_physical_sign_sufficient': dev_pass, 'every_consumed_domain_physical_sign_sufficient': all_pass,
              'uniform_sufficient_envelope_unattainable_on_development': uniform_unattainable,
              'scope': '12 fixed original tree roots; dual margin bound, not unrestricted affine impossibility; interval model, not executed C/LUT.'}
    records.append(record)
    del row, target, fields, x, raw_x, raw_dual, values
    guard()
    checkpoint(bank_complete=bank, solver_status=solver['status'], development_physical_pass=dev_pass,
               consumed_domain_physical_pass=all_pass, uniform_envelope_unattainable=uniform_unattainable)
assert len(records) == 12 and sum(v['development_examples'] for v in records) == 159414
assert sum(v['consumed_validation_examples'] for v in records) == 79458
assert np.array_equal(certificate['meta'][:, 0], np.arange(U))
output = OUT / 'certificate.bin'
with output.open('xb') as stream:
    stream.write(struct.pack('<8sIIQ', b'M482CER1', 88, 12, U))
    stream.write(certificate.tobytes())
assert output.stat().st_size == 24 + 88 * U
write(OUT / 'bank_records.json', records)
gates['ALL12_root_labels_ID_BYTES_development_only_LPs_saved_primal_dual_every_UID_intervals'] = True

phase = 'complete_reports_and_separate_scientific_decisions'
reports = M.reports(unique, owner, links, certificate)
write(OUT / 'complete_reports.json', reports)
gates['ALL39_72_4608_9216_views_empty_ID_slots_roles_and_complete_denominators'] = True
decisions = {'all12_roots_consumed_domain_physical_sign_sufficient': all(v['every_consumed_domain_physical_sign_sufficient'] for v in records),
             'any_root_uniform_sufficient_envelope_unattainable_on_development': any(v['uniform_sufficient_envelope_unattainable_on_development'] for v in records),
             'all12_solver_calls_optimal': all(v['solver']['status'] == 0 for v in records)}
for rel, item in prior['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
last_status = status()
assert parent.cpu_affinity() == [0]
assert not ({'torch', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'meth481_geometry', 'meth481_geometry_math',
             'meth481_retention_audit', 'meth481_r1_retention_audit', 'meth481_r2_retention_audit',
             'meth480_transfer_math', 'meth480_reporting'} & set(sys.modules))
gates['resources_preservation_fixed_solver_no_native_or_previous_science_replay_no_sweep'] = True
guard()
record = {'experiment': 'METH482 ONE all-bank root affine normalized maximum-margin inquiry',
          'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance, 'commands': [],
          'source_binding_sha256': BIND_SHA, 'source481_RAW_sha256': GEO_RAW_SHA, 'source481_RET_sha256': RET481_SHA,
          'new_scipy_runtime_binding_sha256': SCI_RUNTIME_SHA, 'scientific_sources': {str(p): digest(p) for p in scientific},
          'gates': gates, 'decisions': decisions, 'decision': 'ROOT_AFFINE_INQUIRY_COMPLETE_PENDING_INDEPENDENT_ADMISSION',
          'unique_inputs': U, 'queries': Q, 'root_maximum_and_ID_fields_BYTE_checked': U * 4,
          'root_LP_calls': 12, 'control_LP_calls': 3, 'development_training_examples': 159414, 'validation_training_examples': 0,
          'banks': records, 'overall': reports['unique_views'][0], 'reports_path': str(OUT / 'complete_reports.json'), 'controls': controls,
          'native_commands': 0, 'SVD_refits': 0, 'geometry_wire_width': 88, 'interval_fields': U * 8,
          'physical_contract': 'Finite F32 coefficients/inputs,augmented bias1,exact F64 products,any summation tree <=768 F64 additions,one correctly rounded F32 cast,FTZ/DAZ off. The C gate is not executed/export-integrated.',
          'scope': 'Consumed finite original tree roots. Checked intervals of saved affine candidates and dual normalized-margin bounds. No complete tree/mass/functions/new-n/freshquality/rate/LUT/physicalDRAM/general-family claim.',
          'tracked_status': last_status, 'head_at_execution': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
          'resource': {'seconds_before_RAW': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed,
                       'hard_seconds': 600, 'hard_peak_bytes': 4 << 30, 'hard_new_bytes': 64 << 20, 'per_root_solver_time_limit': 25},
          'output_inventory': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']}
assert len(gates) == 5 and all(gates.values())
write(RET, record)
guard()
checkpoint(terminal=True, raw_sha256=digest(RET), OS_peak_bytes=peak)
watchdog.cancel()
print(json.dumps({'raw': str(RET), 'sha256': digest(RET), 'gates': gates, 'decisions': decisions,
                  'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()

