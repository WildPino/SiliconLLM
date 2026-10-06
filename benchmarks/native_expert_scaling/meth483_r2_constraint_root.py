"""ONE complete all-bank active-row root inquiry with retained witnesses."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import time
START = time.monotonic()
import argparse
import math
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
OUT = ROOT / 'results/native_expert_scaling/meth483_r2_constraint_root'
RET = DOC / 'meth483_r2_constraint_root_result.json'
PROTO483 = DOC / 'METH_483_R2_CONSTRAINT_ROOT_PROTOCOL_20261006.md'
ALGEBRA483 = DOC / 'METH_482_ALGEBRA_REASSESSMENT_AND_NEXT_20261006.md'
CALC483 = ROOT / 'benchmarks/native_expert_scaling/meth483_r1_constraint_geometry.py'
WINDOWS483 = ROOT / 'benchmarks/native_expert_scaling/meth483_r2_windows_terminal.ps1'
ORIGINAL483_OUT = ROOT / 'results/native_expert_scaling/meth483_constraint_root'
FAULT483 = DOC / 'meth483_first_fault_inventory.json'
FAULT483_SHA = 'ceea4d1cd0bcc1b27327931e48aaa041068e258e694d4c7f584c1dc6242462b5'
FAILURE483 = DOC / 'meth483_constraint_root_result.failure.json'
EVENT483 = ROOT / 'results/native_expert_scaling/meth483_windows_terminal.json'
REPAIR_PLAN483 = DOC / 'METH_483_WARNING_REPAIR_PLAN_20261006.md'
RET482 = DOC / 'RETENTION_482_R1_AUDIT_R1_20261006.json'
RET482_SHA = '0ae5102e58275dcc5894c240b467b21e3b1dcac1ed261ac8424a3d05a8365f65'
RAW482 = DOC / 'meth482_r1_affine_root_result.json'
RAW482_SHA = '506296eec5118049b5609d22226c3c3aba40ed4aa5979a8d18b8ca13437149bf'
EVENT482 = ROOT / 'results/native_expert_scaling/meth482_r1_windows_terminal.json'
EVENT482_SHA = '6d991009a4387aa27dd84ac936addc8f6a2524c97c11fae431b7339a7f6b3037'
EVENT482_AUDIT = ROOT / 'results/native_expert_scaling/meth482_r1_audit_windows_terminal.json'
EVENT482_AUDIT_SHA = '4ee05d53b8a9473545cb41656ca9f0fff86923ccd62322d9ca49ba70de0d1920'
FAULT482_AUDIT = DOC / 'meth482_first_audit_fault_inventory.json'
FAULT482_AUDIT_SHA = 'af128850a78e87b58b4a310b9cd88d3bbee0c9f7ab66950f77b41b4a9c0cbcf9'
REPAIR482_PROTO = DOC / 'METH_482_R1_RESOURCE_PROTOCOL_20261005.md'
FAULT482 = DOC / 'meth482_first_fault_inventory.json'
FAULT482_SHA = '6fb489e567dd52e1a342454d91ae41538f600397705670638997643d4dbc4deb'
PROTO482 = DOC / 'METH_482_AFFINE_ROOT_PROTOCOL_20261005.md'
CALC482 = ROOT / 'benchmarks/native_expert_scaling/meth482_affine_math.py'
WINDOWS482 = ROOT / 'benchmarks/native_expert_scaling/meth482_r1_windows_terminal.ps1'
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
R1_FAULT = DOC/'meth483_r1_first_fault_inventory.json'
R1_FAULT_SHA = 'aa708abf6cb93ca60f0785c74d9c0cc67180271977b4a669bd0c777ea1fc757a'
R1_NUMERICAL_SUFFIX_SHA = '44f6cb70097665a25674508683c15e3f12556fe540c1093df5333d337a8f4d07'
scientific = (R1_FAULT,DOC/'meth483_r1_constraint_root_result.failure.json',
              DOC/'METH_483_R1_FIRST_FAULT_20261006.md',
              ROOT/'benchmarks/native_expert_scaling/meth483_r1_constraint_root.py',
              ROOT/'benchmarks/native_expert_scaling/meth483_r1_windows_terminal.ps1',
              DOC/'METH_483_R1_CONSTRAINT_ROOT_PROTOCOL_20261006.md',
              FAULT483, FAILURE483, REPAIR_PLAN483,
              ROOT / 'benchmarks/native_expert_scaling/meth483_constraint_root.py',
              ROOT / 'benchmarks/native_expert_scaling/meth483_constraint_geometry.py',
              ROOT / 'benchmarks/native_expert_scaling/meth483_windows_terminal.ps1',
              DOC / 'METH_483_CONSTRAINT_ROOT_PROTOCOL_20261006.md',
              DOC / 'METH_483_FIRST_FAULT_20261006.md',
              DOC / 'meth483_warning_serialization_diagnosis.json',
              DOC / 'meth483_warning_serialization_diagnosis2.json',
              Path(__file__).resolve(), PROTO483, ALGEBRA483, CALC483, WINDOWS483, RET482, RAW482, DOC / 'METH_482_AFFINE_ROOT_RESULT_20261006.md', ROOT / 'benchmarks/native_expert_scaling/meth482_r1_affine_root.py', FAULT482_AUDIT, DOC / 'RETENTION_482_R1_20261006.failure.json', DOC / 'METH_482_FIRST_AUDIT_FAULT_20261006.md', ROOT / 'benchmarks/native_expert_scaling/meth482_retention_audit.py', DOC / 'METH_482_RETENTION_PROTOCOL_20261006.md', ROOT / 'benchmarks/native_expert_scaling/meth482_audit_windows_terminal.ps1', ROOT / 'benchmarks/native_expert_scaling/meth482_r1_retention_audit.py', DOC / 'METH_482_R1_RETENTION_PROTOCOL_20261006.md', ROOT / 'benchmarks/native_expert_scaling/meth482_r1_audit_windows_terminal.ps1', REPAIR482_PROTO, FAULT482, DOC / 'meth482_affine_root_result.failure.json', DOC / 'METH_482_FIRST_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth482_affine_root.py', ROOT / 'benchmarks/native_expert_scaling/meth482_windows_terminal.ps1', PROTO482, CALC482, WINDOWS482, SCI_RUNTIME, RET481, ROOT / 'benchmarks/native_expert_scaling/meth481_r2_retention_audit.py', R2_PROTO, R1_FAULT481, DOC / 'RETENTION_481_R1_20261005.failure.json', DOC / 'METH_481_R1_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth481_r1_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth481_r1_audit_windows_terminal.ps1', REPAIR481_PROTO, FAULT481, DOC / 'RETENTION_481_20261005.failure.json', DOC / 'METH_481_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth481_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth481_audit_windows_terminal.ps1', AUDIT481_PROTO, AUDIT481_WINDOWS, ROOT / 'benchmarks/native_expert_scaling/meth481_geometry.py', GEO_RAW, DOC / 'METH_481_MAIN_TERMINAL_PENDING_20261005.md', PROTOCOL481, CALC481, WINDOWS481, PRIOR_RET, ROOT / 'benchmarks/native_expert_scaling/meth480_r1_retention_audit.py', AUDIT_PROTO, AUDIT_WINDOWS, AUDIT_REPAIR_PROTO, AUDIT_FAULT_INVENTORY, DOC / 'RETENTION_480_20261005.failure.json', DOC / 'METH_480_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth480_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth480_audit_windows_terminal.ps1', ROOT / 'benchmarks/native_expert_scaling/meth480_r1_learned_router.py', ROOT / 'benchmarks/native_expert_scaling/meth480_learned_router.py', C, MATH, REPORTING, WINDOWS, PROTO, ARITHMETIC_NOTE, REPAIR_PROTO, FIRST_INVENTORY, DOC / 'meth480_learned_router_result.failure.json', DOC / 'METH_480_FIRST_FAULT_20261005.md')

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
    assert time.monotonic() - START <= 600 and peak <= 8 << 30, (phase, peak)
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
        write(path, {'experiment': 'METH483-R1 FIRST active-row inquiry fault', 'phase': phase,
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
        write(path, {'experiment': 'METH483-R1 active-row watchdog FIRST fault', 'phase': phase, 'start_utc': start_utc,
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
assert digest(FAULT482) == FAULT482_SHA
fault482 = json.loads(FAULT482.read_bytes())
assert fault482['actual_returncode'] == 1 and fault482['saved_source_primal_dual_witnesses'] == fault482['UID_interval_fields_saved'] == 0
for item in fault482['records']:
    check(item)
first482 = json.loads((DOC / 'meth482_affine_root_result.failure.json').read_bytes())
assert '5646815232' in first482['traceback']
for path, sha in first482['scientific_sources'].items():
    head(path)
    assert digest(path) == sha
assert digest(SCI_RUNTIME) == SCI_RUNTIME_SHA
head(SCI_RUNTIME)
sci_runtime = json.loads(SCI_RUNTIME.read_bytes())
assert sci_runtime['python'] == sys.version and Path(sci_runtime['executable']).resolve() == Path(sys.executable).resolve()
assert importlib.metadata.version('scipy') == sci_runtime['version']
for item in sci_runtime['files']:
    check(item)
head(ALGEBRA483)
assert digest(FAULT482_AUDIT) == FAULT482_AUDIT_SHA
fault482audit = json.loads(FAULT482_AUDIT.read_bytes())
assert fault482audit['actual_returncode'] == 1 and not fault482audit['numerical_imports']
assert fault482audit['dataset_scientific_fields_audited'] == fault482audit['native_commands'] == 0
for item in fault482audit['records']:
    check(item)
for path, sha in json.loads((DOC / 'RETENTION_482_R1_20261006.failure.json').read_bytes())['scientific_sources'].items():
    head(path)
    assert digest(path) == sha
assert digest(RAW482) == RAW482_SHA and digest(EVENT482) == EVENT482_SHA
r482 = json.loads(RAW482.read_bytes())
assert len(r482['gates']) == 5 and all(r482['gates'].values())
assert r482['source_binding_sha256'] == BIND_SHA and r482['root_LP_calls'] == 12 and r482['control_LP_calls'] == 3
for path, sha in r482['scientific_sources'].items():
    head(path)
    assert digest(path) == sha
for item in r482['output_inventory']:
    check(item)
assert r482['resource']['hard_seconds'] == 600 and r482['resource']['seconds_before_RAW'] <= 600
assert r482['resource']['hard_peak_bytes'] == 8 << 30 and r482['resource']['OS_peak_bytes'] <= 8 << 30
assert r482['resource']['hard_new_bytes'] == 64 << 20
source482out = ROOT / 'results/native_expert_scaling/meth482_r1_affine_root'
assert sum(p.stat().st_size for p in source482out.iterdir()) + RAW482.stat().st_size <= 64 << 20
source482rows = [json.loads(x) for x in (source482out / 'progress.jsonl').read_text(encoding='utf8').splitlines()]
assert source482rows[-1]['terminal'] and source482rows[-1]['raw_sha256'] == RAW482_SHA
assert [x['bank_complete'] for x in source482rows if 'bank_complete' in x] == list(range(12))
assert digest(RET482) == RET482_SHA and digest(EVENT482_AUDIT) == EVENT482_AUDIT_SHA
ret482 = json.loads(RET482.read_bytes())
assert len(ret482['gates']) == 4 and all(ret482['gates'].values()) and ret482['raw_sha256'] == RAW482_SHA
assert ret482['decision'] == 'ALL12_ROOT_LPS_BUDGET_INCONCLUSIVE_INDEPENDENTLY_ADMITTED'
assert ret482['unique_inputs_audited'] == 238872 and ret482['queries_audited'] == 387036
assert ret482['root_maximum_and_ID_fields_BYTE_checked'] == 955488 and ret482['interval_FIELDS_BYTE_checked'] == 1910976
assert ret482['saved_candidate_heads'] == ret482['saved_usable_dual_witnesses'] == ret482['native_replays'] == ret482['LP_replays'] == 0
assert digest(ROOT / 'benchmarks/native_expert_scaling/meth482_r1_retention_audit.py') == ret482['audit_source_sha256']
assert digest(DOC / 'METH_482_R1_RETENTION_PROTOCOL_20261006.md') == ret482['audit_protocol_sha256']
for event_path, actual_source in ((EVENT482, r482), (EVENT482_AUDIT, ret482)):
    event482 = json.loads(event_path.read_bytes())
    assert event482['query_available'] and event482['query_error'] is None and not event482['matching_scientific_events']
    assert len(event482['instances']) == 1
    associated = event482['instances'][0]
    assert associated['pid'] == actual_source['process_instance']['pid'] and associated['create_time_unix'] == actual_source['process_instance']['create_time_unix']
    for key in ('start_utc', 'end_utc'):
        assert datetime.fromisoformat(associated[key]) == datetime.fromisoformat(actual_source[key])
    assert datetime.fromisoformat(event482['query_end_utc']) > datetime.fromisoformat(actual_source['end_utc'])
assert ret482['resource']['hard_seconds'] == 600 and ret482['resource']['seconds_before_RET'] <= 600
assert ret482['resource']['hard_peak_bytes'] == 4 << 30 and ret482['resource']['OS_peak_bytes'] <= 4 << 30
assert ret482['resource']['hard_new_bytes'] == 32 << 20

assert digest(R1_FAULT) == R1_FAULT_SHA
r1_fault = json.loads(R1_FAULT.read_bytes())
assert r1_fault['actual_returncode'] == 1 and not r1_fault['numerical_imports']
assert r1_fault['new_control_LP_calls'] == r1_fault['new_root_LP_calls'] == r1_fault['source_rounds_replayed'] == 0
for item in r1_fault['records']: check(item)
r1_failure = json.loads((DOC/'meth483_r1_constraint_root_result.failure.json').read_bytes())
assert 'HEAD:results/native_expert_scaling/meth483_windows_terminal.json' in r1_failure['traceback']
for path, sha in r1_failure['scientific_sources'].items():
    assert digest(path) == sha
    if Path(path).resolve() != EVENT483.resolve(): head(path)
r1_event = json.loads((ROOT/'results/native_expert_scaling/meth483_r1_windows_terminal.json').read_bytes())
assert r1_event['query_available'] and r1_event['query_error'] is None and not r1_event['matching_scientific_events']
assert len(r1_event['instances']) == 1
for key in ('pid','create_time_unix'):
    assert r1_event['instances'][0][key] == r1_failure['process_instance'][key]
for key in ('start_utc','end_utc'):
    assert datetime.fromisoformat(r1_event['instances'][0][key]) == datetime.fromisoformat(r1_failure[key])
assert datetime.fromisoformat(r1_event['query_end_utc']) > datetime.fromisoformat(r1_failure['end_utc'])
old_suffix = (ROOT/'benchmarks/native_expert_scaling/meth483_r1_constraint_root.py').read_bytes().split(
    (b"phase = "+repr("frozen_solver_interval_controls_and_source_wire_admission").encode()),1)[1]
new_suffix = Path(__file__).read_bytes().split(
    (b"phase = "+repr("frozen_solver_interval_controls_and_source_wire_admission").encode()),1)[1]
assert old_suffix == new_suffix
assert hashlib.sha256((b"phase = "+repr("frozen_solver_interval_controls_and_source_wire_admission").encode())+old_suffix).hexdigest() == R1_NUMERICAL_SUFFIX_SHA
assert digest(FAULT483) == FAULT483_SHA
fault483 = json.loads(FAULT483.read_bytes())
assert fault483['actual_returncode'] == 1 and fault483['session'] == 50660
assert fault483['completed_banks'] == list(range(7)) and fault483['saved_root_rounds'] == 57
assert fault483['control_LP_calls_completed'] == 4 and fault483['fault_bank'] == 7
assert not fault483['global_certificate_saved'] and not fault483['global_reports_saved']
for item in fault483['records']: check(item)
assert len(fault483['records']) == 134
failure483 = json.loads(FAILURE483.read_bytes())
assert 'HighsStatus.kWarning' in failure483['traceback']
for path, sha in failure483['scientific_sources'].items():
    head(path); assert digest(path) == sha
old_event483 = json.loads(EVENT483.read_bytes())
assert old_event483['query_available'] and old_event483['query_error'] is None and not old_event483['matching_scientific_events']
assert len(old_event483['instances']) == 1
associated483 = old_event483['instances'][0]
for key in ('pid','create_time_unix'):
    assert associated483[key] == failure483['process_instance'][key]
for key in ('start_utc','end_utc'):
    assert datetime.fromisoformat(associated483[key]) == datetime.fromisoformat(failure483[key])
assert datetime.fromisoformat(old_event483['query_end_utc']) > datetime.fromisoformat(failure483['end_utc'])
old_progress483 = [json.loads(v) for v in (ORIGINAL483_OUT/'progress.jsonl').read_text().splitlines()]
assert [v['bank_complete'] for v in old_progress483 if 'bank_complete' in v] == list(range(7))
assert old_progress483[-1]['terminal_failure']
old_start483 = {v['bank_LP_start']:v['seconds'] for v in old_progress483 if 'bank_LP_start' in v}
old_end483 = {v['bank_complete']:v['seconds'] for v in old_progress483 if 'bank_complete' in v}
paid_bank7_seconds = math.ceil(1000*(failure483['resource']['seconds']-old_start483[7]))/1000
assert 1.828 <= paid_bank7_seconds <= 1.829
api_bytes = (ROOT/'.venv/Lib/site-packages/scipy/optimize/_highspy/_core.cp312-win_amd64.pyd').read_bytes()
for symbol in (b'getLp',b'log_file',b'small_matrix_value',b'ObjSense',b'kMinimize',b'kColwise',b'kRowwise'):
    assert symbol in api_bytes

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
import meth483_r1_constraint_geometry as M
assert Path(M.__file__).resolve() == CALC483
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1
    assert str(Path(pool['filepath']).resolve()) in set(runtime['packages']['numpy']['files']) | scipy_files

def subnormal_control():
    smallest = np.array([np.nextafter(np.float64(0.), np.float64(1.))], '<f8')
    assert smallest.tobytes() == struct.pack('<Q', 1)
    assert (smallest * np.float64(1.)).tobytes() == struct.pack('<Q', 1)
    assert (smallest + smallest).tobytes() == struct.pack('<Q', 2)

subnormal_control()
# Old four controls are retained and mathematically reconstructed, never replayed.
old_controls = json.loads((ORIGINAL483_OUT/'controls.json').read_bytes())
assert old_controls['control_LP_calls'] == 4
for control_id, (values, labels, expected, source_ids) in enumerate((([-1.,1.],[-1,1],1.,[0,1]),
       ([-1.,0.,1.],[1,-1,1],0.,[0,1,0]), ([-1.,0.,1.],[1,1,1],1.,[0,1,2]))):
    x, y = np.array(values,'<f4')[:,None], np.array(labels,'<i1')
    old = old_controls['solver_analytic_controls'][control_id]
    retained = M.saved_rounds(x,y,np.array(source_ids),ORIGINAL483_OUT,'control'+str(control_id),len(old['solver']['rounds']))
    assert retained['rounds'] == old['solver']['rounds']
    assert old['expected_optimum'] == expected and old['status'] == 0
    assert retained['best_dual']['bound'] == old['dual_upper']
    assert abs(float(retained['best_primal']['raw'][-1])-expected) <= 1e-8
original_options = old_controls['solver_analytic_controls'][0]['solver']['options_verified']
x = np.array([[-1.,0.],[1.,2.**-32]],'<f4'); y = np.array([-1,1],'<i1')
def save_warning_round(ordinal, detail, arrays):
    np.savez(OUT/('warning_control_round'+str(ordinal)+'.npz'),**arrays)
    write(OUT/('warning_control_round'+str(ordinal)+'.json'),detail); guard()
control_result = M.solve(x,y,np.array([0,1]),save_warning_round,
                         log_path=OUT/'warning_control_solver.log',control_append=True)
result, raw_x, raw_dual, theta, theta32, weights, dual_indices = control_result
assert len(result['rounds']) == 1 and result['status'] == 0 and abs(-result['fun']-1.) <= 1e-8
_, _, _, _, bound = M.dual_interval(x,y,weights)
assert bound is not None and 1. <= bound <= 1.+1e-8
physical = M.physical_intervals(x,theta32)
assert np.all(np.where(y>0,physical[2],-physical[3]) > 0.)
log = (OUT/'warning_control_solver.log').read_text()
assert 'ignored' in log and 'less than or equal' in log
subnormal_control()
cancel_x=np.array([[1e20,1.,-1e20]],'<f4');cancel_theta=np.array([1.,1.,1.,0.],'<f8')
lo,hi,_=M.dot_intervals(cancel_x,cancel_theta)
with localcontext() as context:
    context.prec=100;exact=sum(Decimal(float(v)) for v in cancel_x[0])
    assert Decimal(float(lo[0])) <= exact <= Decimal(float(hi[0]))
controls = {'retained_original_controls': old_controls, 'retained_control_LP_calls':4,
            'new_control_LP_calls':1, 'control_LP_calls':5, 'warning_control':result,
            'warning_control_original_dual_upper':bound, 'warning_control_known_optimum':1.,
            'Decimal100_cancellation_enclosed':True,'IEEE_F64_subnormal_arithmetic':True,
            'physical_contract_requires_FTZ_DAZ_off':True}
write(OUT/'controls.json',controls)
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

phase = 'reuse7_completed_resume_bank7_first_inquire8to11_ALL_UID_intervals'
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
    def save_bank_round(ordinal, detail, arrays):
        np.savez(OUT / ('bank' + str(bank) + '_round' + str(ordinal) + '.npz'), **arrays)
        write(OUT / ('bank' + str(bank) + '_round' + str(ordinal) + '.json'), detail)
        checkpoint(bank_round_saved=bank, round=ordinal, active_count=detail['active_count'], status=detail['model_status_name'])
        guard()
    seed = None
    if bank < 8:
        seed = M.saved_rounds(x[local_dev],labels[local_dev],row['meta'][local_dev,7],
                              ORIGINAL483_OUT,'bank'+str(bank),8 if bank<7 else 1)
    if bank < 7:
        last = seed['rounds'][-1]
        assert last['stop_or_add'] != 'add_complete_scan_violations'
        solved = M.finish(len(local_dev),769,seed['best_primal'],seed['best_dual'],seed['rounds'],
                         seed['initial'],last['stop_or_add'],None,original_options,[])
        lp_seconds = None
        lp_bounds = [last['wall_seconds'],old_end483[bank]-old_start483[bank]]
        assert 0. < lp_bounds[0] <= lp_bounds[1]
    else:
        solve_started = time.monotonic()
        solved = M.solve(x[local_dev],labels[local_dev],row['meta'][local_dev,7],save_bank_round,
                         log_path=OUT/('bank'+str(bank)+'_solver.log'),seed=seed,
                         paid_seconds=paid_bank7_seconds if bank==7 else 0.)
        lp_seconds = time.monotonic()-solve_started
        lp_bounds = None
    solver, raw_x, raw_dual, theta, theta32, weights, dual_indices = solved
    reconstruction_seconds = time.monotonic()-lp_start if bank<7 else None
    process_gate()
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
    model = (ORIGINAL483_OUT if bank<7 else OUT)/('bank'+str(bank)+'_witness.npz')
    witness = dict(development_UIDs=uid[local_dev].astype('<u4'),development_labels=labels[local_dev],
             raw_primal=raw_x,raw_dual=raw_dual,dual_source_active_development_indices=dual_indices,
             theta=theta,theta_F32=theta32,dual_weights=weights,dual_residual_lower=dual_lo,
             dual_residual_upper=dual_hi,dual_weight_sum_bounds=np.array([sum_lo,sum_hi]),
             dual_bound=np.array([dual_bound if dual_bound is not None else 0.]),
             dual_available=np.array([dual_bound is not None],'<u4'))
    if bank<7:
        with np.load(model,allow_pickle=False) as archive:
            assert set(archive.files) == set(witness)
            for key,wanted in witness.items():
                assert archive[key].dtype == wanted.dtype and archive[key].shape == wanted.shape
                assert archive[key].tobytes() == wanted.tobytes(),(bank,key)
    else:
        np.savez(model,**witness)
    dev_pass = solver['available'] and bool(np.all(signed[local_dev] > 0))
    all_pass = solver['available'] and bool(np.all(signed > 0))
    uniform_unattainable = dual_bound is not None and dual_bound <= envelope
    if dual_bound is not None:
        lower_candidate = max(0., float(ideal_signed[local_dev].min()))
        assert lower_candidate <= dual_bound
    record = {'bank': bank, 'unique_queries': len(uid), 'development_examples': len(local_dev),
              'consumed_validation_examples': int(np.count_nonzero(val[uid])), 'solver': solver, 'LP_seconds': lp_seconds, 'original_LP_seconds_bounds':lp_bounds,
              'new_reconstruction_seconds':reconstruction_seconds, 'retained_root_rounds':8 if bank<7 else 1 if bank==7 else 0,
              'model_path': str(model), 'dual_unit_L1_margin_upper': dual_bound, 'uniform_F32_export_and_arithmetic_envelope': envelope,
              'development_max_augmented_coordinate': B, 'candidate_unit_L1_norm_upper': M.sum_upper_nonnegative(np.abs(theta)),
              'candidate_development_min_ideal_margin_lower': float(ideal_signed[local_dev].min()) if solver['available'] else None,
              'every_development_physical_sign_sufficient': dev_pass, 'every_consumed_domain_physical_sign_sufficient': all_pass,
              'uniform_sufficient_envelope_unattainable_on_development': uniform_unattainable,
              'scope': '12 fixed roots; active-row primal/dual witnesses checked on full development. Uniform error envelope is sufficient only, not necessary for physical sign. C/LUT not executed.'}
    records.append(record)
    del row, target, fields, x, raw_x, raw_dual, values
    guard()
    checkpoint(bank_complete=bank, solver_status=solver['status'], development_physical_pass=dev_pass,
               consumed_domain_physical_pass=all_pass, uniform_envelope_unattainable=uniform_unattainable)
assert 57 <= sum(len(v['solver']['rounds']) for v in records) <= 96
assert len(records) == 12 and sum(v['development_examples'] for v in records) == 159414
assert sum(v['consumed_validation_examples'] for v in records) == 79458
assert np.array_equal(certificate['meta'][:, 0], np.arange(U))
output = OUT / 'certificate.bin'
with output.open('xb') as stream:
    stream.write(struct.pack('<8sIIQ', b'M483CER1', 88, 12, U))
    stream.write(certificate.tobytes())
assert output.stat().st_size == 24 + 88 * U
write(OUT / 'bank_records.json', records)
gates['ALL12_root_labels_ID_BYTES_development_only_active_LPs_saved_ALLrounds_every_UID_intervals'] = True

phase = 'complete_reports_and_separate_scientific_decisions'
reports = M.reports(unique, owner, links, certificate)
write(OUT / 'complete_reports.json', reports)
gates['ALL39_72_4608_9216_views_empty_ID_slots_roles_and_complete_denominators'] = True
decisions = {'all12_roots_consumed_domain_physical_sign_sufficient': all(v['every_consumed_domain_physical_sign_sufficient'] for v in records),
             'any_root_uniform_sufficient_envelope_unattainable_on_development': any(v['uniform_sufficient_envelope_unattainable_on_development'] for v in records),
             'every_retained_solver_round_optimal': bool(all(v['solver']['rounds'] for v in records) and all(round_['model_status_name'].endswith('kOptimal') for v in records for round_ in v['solver']['rounds']))}
for rel, item in prior['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
process_gate()
last_status = status()
assert parent.cpu_affinity() == [0]
assert not ({'torch', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'meth481_geometry', 'meth481_geometry_math',
             'meth481_retention_audit', 'meth481_r1_retention_audit', 'meth481_r2_retention_audit',
             'meth480_transfer_math', 'meth480_reporting'} & set(sys.modules))
gates['resources_preservation_fixed_solver_no_native_or_previous_science_replay_no_sweep'] = True
guard()
record = {'experiment': 'METH483-R1 ONE all-bank bounded active-row normalized-margin inquiry',
          'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance, 'commands': [],
          'source_binding_sha256': BIND_SHA, 'source481_RAW_sha256': GEO_RAW_SHA, 'source481_RET_sha256': RET481_SHA, 'source482_RAW_sha256': RAW482_SHA, 'source482_RET_sha256': RET482_SHA,
          'new_scipy_runtime_binding_sha256': SCI_RUNTIME_SHA, 'scientific_sources': {str(p): digest(p) for p in scientific},
          'gates': gates, 'decisions': decisions, 'decision': 'ACTIVE_ROW_INQUIRY_COMPLETE_PENDING_INDEPENDENT_ADMISSION',
          'unique_inputs': U, 'queries': Q, 'root_maximum_and_ID_fields_BYTE_checked': U * 4,
          'root_inquiries':12, 'retained_root_LP_calls':57, 'new_root_LP_calls':sum(len(v['solver']['rounds']) for v in records)-57,
          'source483_first_fault_inventory_sha256':FAULT483_SHA, 'source483_original_paid_controller_seconds':failure483['resource']['seconds'],
          'bank7_prior_search_seconds_paid':paid_bank7_seconds, 'root_LP_calls':sum(len(v['solver']['rounds']) for v in records), 'control_LP_calls': controls['control_LP_calls'], 'development_training_examples': 159414, 'validation_training_examples': 0,
          'banks': records, 'overall': reports['unique_views'][0], 'reports_path': str(OUT / 'complete_reports.json'), 'controls': controls,
          'native_commands': 0, 'SVD_refits': 0, 'geometry_wire_width': 88, 'interval_fields': U * 8,
          'physical_contract': 'Finite F32 coefficients/inputs,augmented bias1,exact F64 products,any summation tree <=768 F64 additions,one correctly rounded F32 cast,FTZ/DAZ off. The C gate is not executed/export-integrated.',
          'scope': 'Seven completed banks reused byte-exactly; bank7 saved round0 reused with reconstructed basis;8..11 first inquiries. Consumed finite original tree roots. Checked intervals of saved affine candidates and dual normalized-margin bounds. No complete tree/mass/functions/new-n/freshquality/rate/LUT/physicalDRAM/general-family claim.',
          'tracked_status': last_status, 'head_at_execution': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
          'resource': {'seconds_before_RAW': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed,
                       'hard_seconds': 600, 'hard_peak_bytes': 8 << 30, 'hard_new_bytes': 64 << 20, 'per_root_next_call_search_wall_allowance': 25, 'maximum_rounds_per_root': 8, 'maximum_added_constraints_per_round': 256},
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

