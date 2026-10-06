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
OUT = ROOT / 'results/native_expert_scaling/meth483_retention_audit'
RET = DOC / 'RETENTION_483_20261006.json'
PROTO483 = DOC / 'METH_483_RETENTION_PROTOCOL_20261006.md'
ALGEBRA483 = DOC / 'METH_482_ALGEBRA_REASSESSMENT_AND_NEXT_20261006.md'
CALC483 = ROOT / 'benchmarks/native_expert_scaling/meth483_r1_constraint_geometry.py'
WINDOWS483 = ROOT / 'benchmarks/native_expert_scaling/meth483_audit_windows_terminal.ps1'
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
R2_FAULT = DOC/'meth483_r2_first_fault_inventory.json'
R2_FAULT_SHA = '45264f5e70615cf4fa563b292c2bae5ac36b67eb404b82b21d08c76830a58cba'
SOURCE_R2 = ROOT/'results/native_expert_scaling/meth483_r2_constraint_root'
RAW483=DOC/'meth483_r3_constraint_root_result.json'
RAW483_SHA='6cfa6456192408f988546f3d577f61f254e9b27cc5eff46c1a7540caf9f232cd'
R3_OUT=ROOT/'results/native_expert_scaling/meth483_r3_constraint_root'
R3_EVENT=ROOT/'results/native_expert_scaling/meth483_r3_windows_terminal.json'
R3_EVENT_SHA='97a1105617dd4cc7b82da137c4fcd1c07d37b3ae784505c779ecf7023de01e95'
scientific = (RAW483,ROOT/'benchmarks/native_expert_scaling/meth483_r3_constraint_root.py',
              ROOT/'benchmarks/native_expert_scaling/meth483_r3_windows_terminal.ps1',
              DOC/'METH_483_R3_CONSTRAINT_ROOT_PROTOCOL_20261006.md',
              DOC/'METH_483_R3_MAIN_TERMINAL_PENDING_20261006.md',R2_FAULT,DOC/'meth483_r2_constraint_root_result.failure.json',
              DOC/'METH_483_R2_FIRST_FAULT_20261006.md',
              ROOT/'benchmarks/native_expert_scaling/meth483_r2_constraint_root.py',
              ROOT/'benchmarks/native_expert_scaling/meth483_r2_windows_terminal.ps1',
              DOC/'METH_483_R2_CONSTRAINT_ROOT_PROTOCOL_20261006.md',R1_FAULT,DOC/'meth483_r1_constraint_root_result.failure.json',
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
        write(path, {'experiment': 'METH483 FIRST independent retention fault', 'phase': phase,
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
        write(path, {'experiment': 'METH483 independent retention watchdog FIRST fault', 'phase': phase, 'start_utc': start_utc,
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
assert sum(p.stat().st_size for p in SOURCE481_OUT.iterdir()) + GEO_RAW.stat().st_size <= g['resource']['hard_new_bytes']
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
assert sum(p.stat().st_size for p in source482out.iterdir()) + RAW482.stat().st_size <= r482['resource']['hard_new_bytes']
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

assert digest(R2_FAULT) == R2_FAULT_SHA
r2_fault = json.loads(R2_FAULT.read_bytes())
assert r2_fault['actual_returncode'] == 1 and r2_fault['session'] == 25404
assert r2_fault['new_root_LP_calls'] == 0 and r2_fault['new_control_LP_calls'] == 1
assert r2_fault['new_warning_control_completed']
for item in r2_fault['records']: check(item)
r2_failure = json.loads((DOC/'meth483_r2_constraint_root_result.failure.json').read_bytes())
assert "(3644, 'python.exe')" in r2_failure['traceback']
for path,sha in r2_failure['scientific_sources'].items():
    head(path); assert digest(path) == sha
r2_event=json.loads((ROOT/'results/native_expert_scaling/meth483_r2_windows_terminal.json').read_bytes())
assert r2_event['query_available'] and r2_event['query_error'] is None and not r2_event['matching_scientific_events']
assert len(r2_event['instances']) == 1
for key in ('pid','create_time_unix'):
    assert r2_event['instances'][0][key] == r2_failure['process_instance'][key]
for key in ('start_utc','end_utc'):
    assert datetime.fromisoformat(r2_event['instances'][0][key]) == datetime.fromisoformat(r2_failure[key])
assert datetime.fromisoformat(r2_event['query_end_utc']) > datetime.fromisoformat(r2_failure['end_utc'])
root_marker=b'U, Q = '+b'238872, 387036'
assert (ROOT/'benchmarks/native_expert_scaling/meth483_r3_constraint_root.py').read_bytes().split(root_marker,1)[1] == (ROOT/'benchmarks/native_expert_scaling/meth483_r2_constraint_root.py').read_bytes().split(root_marker,1)[1]
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
new_suffix = (ROOT/'benchmarks/native_expert_scaling/meth483_r2_constraint_root.py').read_bytes().split(
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

assert digest(RAW483)==RAW483_SHA and digest(R3_EVENT)==R3_EVENT_SHA
r483=json.loads(RAW483.read_bytes())
assert all(r483['gates'].values()) and len(r483['gates'])==5
assert r483['root_LP_calls']==93 and r483['new_root_LP_calls']==36 and r483['retained_root_LP_calls']==57
assert r483['control_LP_calls']==5 and r483['controls']['new_control_LP_calls']==0
assert r483['native_commands']==0 and r483['commands']==[] and r483['validation_training_examples']==0
for path,sha in r483['scientific_sources'].items():
    head(path);assert digest(path)==sha
for item in r483['output_inventory']:check(item)
assert set(p.name for p in R3_OUT.iterdir())=={Path(v['path']).name for v in r483['output_inventory']}|{'progress.jsonl'}
event483=json.loads(R3_EVENT.read_bytes())
assert event483['query_available'] and event483['query_error'] is None and not event483['matching_scientific_events']
assert len(event483['instances'])==1
for key in ('pid','create_time_unix'):
    assert event483['instances'][0][key]==r483['process_instance'][key]
for key in ('start_utc','end_utc'):
    assert datetime.fromisoformat(event483['instances'][0][key])==datetime.fromisoformat(r483[key])
assert datetime.fromisoformat(event483['query_end_utc'])>datetime.fromisoformat(r483['end_utc'])
rows483=[json.loads(v) for v in (R3_OUT/'progress.jsonl').read_text().splitlines()]
assert rows483[-1]['terminal'] and rows483[-1]['raw_sha256']==RAW483_SHA
assert [v['bank_complete'] for v in rows483 if 'bank_complete' in v]==list(range(12))
assert len([v for v in rows483 if 'bank_round_saved' in v])==36
assert (R3_OUT/'fatal_native.log').stat().st_size==0
assert r483['resource']['hard_seconds']==600 and r483['resource']['seconds_before_RAW']<=600
assert r483['resource']['hard_peak_bytes']==8<<30 and r483['resource']['OS_peak_bytes']<=8<<30
assert r483['resource']['hard_new_bytes']==64<<20
assert sum(p.stat().st_size for p in R3_OUT.iterdir())+RAW483.stat().st_size<=64<<20
assert time.monotonic()-START<=180
process_gate()
checkpoint(complete483_source_main_admitted=True)

phase='independent_all_original_and_new_rounds_full_UID_witness_math'
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
for module in (np,threadpoolctl):
    assert module.__version__==runtime['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in runtime['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads']==1 and str(Path(pool['filepath']).resolve()) in runtime['packages']['numpy']['files']
np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
U,Q,D=238872,387036,769
UNIQUE=np.dtype([('meta','<u4',(12,)),('input_sha','u1',(32,)),('score_sha','u1',(32,)),('value','<f8',(3,)),('input','<f4',(768,)),('score','<f4',(128,))])
NODE=np.dtype([('maximum','<f4',(2,)),('winner','<u2',(2,)),('mass','<f8',(2,))])
TARGET=np.dtype([('unique_id','<u4'),('nodes',NODE,(127,))])
CERT=np.dtype([('meta','<u4',(6,)),('value','<f8',(8,))])
def wire(path,magic,width,reserved,count,dtype,shape):
    path=Path(path);assert path.stat().st_size==24+width*count
    with path.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
    return np.memmap(path,mode='r',offset=24,dtype=dtype,shape=shape)
unique=wire(b['data_files']['unique_inputs.bin']['path'],b'M479UNI1',3720,768,U,UNIQUE,(U,))
owner=wire(b['data_files']['ownership.bin']['path'],b'M479OWN1',32,0,U,'<u4',(U,8))
links=wire(b['data_files']['query_links.bin']['path'],b'M479LNK1',52,0,Q,'<u4',(Q,13))
target=wire(b['data_files']['group_targets.bin']['path'],b'M479TGT1',3560,127,U,TARGET,(U,))
cert=wire(R3_OUT/'certificate.bin',b'M483CER1',88,12,U,CERT,(U,))
for ids in (unique['meta'][:,0],owner[:,0],target['unique_id'],cert['meta'][:,0]):assert np.array_equal(ids,np.arange(U))
expected_owner=np.zeros((U,8),'<u4');expected_owner[:,0]=np.arange(U);expected_owner[:,1]=unique['meta'][:,2]
for col,src in ((2,5),(3,4),(4,10)):np.bitwise_or.at(expected_owner[:,col],links[:,12],(1<<links[:,src]).astype('<u4'))
np.add.at(expected_owner[:,5],links[:,12],1)
for book in range(192):
    ids=np.unique(links[links[:,2]==book,12]);expected_owner[ids,7 if 64<=book<128 else 6]+=1
assert expected_owner.tobytes()==owner.tobytes()
assert np.array_equal(links[:,6],unique['meta'][links[:,12],2]) and np.array_equal(links[:,9],unique['meta'][links[:,12],7])
dev,val=(owner[:,2]&5)!=0,(owner[:,2]&2)!=0
assert int(dev.sum())==159414 and int(val.sum())==79458 and not np.any(dev&val)
tree=json.loads(Path(b['data_files']['membership.json']['path']).read_bytes())
up=lambda a:np.nextafter(a,np.inf)
down=lambda a:np.nextafter(a,-np.inf)
def ieee():
    a=np.array([np.nextafter(np.float64(0.),np.float64(1.))],'<f8')
    assert a.tobytes()==struct.pack('<Q',1) and (a*1.).tobytes()==struct.pack('<Q',1) and (a+a).tobytes()==struct.pack('<Q',2)
ieee()
def norm_upper(theta):
    total=0.
    for v in theta:total=math.nextafter(total+abs(float(v)),math.inf)
    return total

def intervals(x,theta):
    lower=np.full(len(x),float(theta[-1]),'<f8');upper=lower.copy();absolute=np.full(len(x),abs(float(theta[-1])),'<f8')
    for col in range(x.shape[1]):
        v=x[:,col].astype('<f8')*float(theta[col])
        lower=down(lower+down(v));upper=up(upper+up(v));absolute=up(absolute+up(np.abs(v)))
    assert np.isfinite(lower).all() and np.isfinite(upper).all() and np.isfinite(absolute).all() and np.all(lower<=upper)
    return lower,upper,absolute

def physical(x,theta32):
    lower,upper,absolute=intervals(x,theta32.astype('<f8'))
    with localcontext() as c:
        c.prec=100;u=Decimal(2)**-53;steps=x.shape[1]
        gamma=math.nextafter(float(steps*u/(1-steps*u)),math.inf)
    e64=up(gamma*absolute);mag=up(np.maximum(np.abs(lower),np.abs(upper))+e64)
    e32=up(up(2.**-24*mag)+2.**-150);error=up(e64+e32)
    return lower,upper,down(lower-error),up(upper+error),absolute,error

def envelope(x):
    B=max(1.,float(np.abs(x).max()))
    with localcontext() as c:
        c.prec=100;u=Decimal(2)**-24;u64=Decimal(2)**-53;delta=Decimal(2)**-150
        d=x.shape[1]+1;gamma=(d-1)*u64/(1-(d-1)*u64);a=u+d*delta
        v=math.nextafter(float(Decimal(B)*(a+(gamma+u*(1+gamma))*(1+a))+delta),math.inf)
    return v,B

def dual_bound(x,y,weights):
    assert np.isfinite(weights).all() and np.all(weights>=0.)
    lower=np.zeros(x.shape[1]+1,'<f8');upper=lower.copy();sl=sh=0.
    for i in range(len(x)):
        h=np.empty(x.shape[1]+1,'<f8');h[:-1]=x[i];h[-1]=1.;h*=int(y[i]);v=float(weights[i])*h
        lower=down(lower+down(v));upper=up(upper+up(v))
        sl=math.nextafter(sl+float(weights[i]),-math.inf);sh=math.nextafter(sh+float(weights[i]),math.inf)
    bound=math.nextafter(float(np.maximum(np.abs(lower),np.abs(upper)).max())/sl,math.inf) if sl>0. else None
    return lower,upper,sl,sh,bound

def byte(a,z):
    assert a.dtype==z.dtype and a.shape==z.shape and a.tobytes()==z.tobytes()

def exact(a,z):
    assert type(a) is type(z)
    if isinstance(a,dict):
        assert a.keys()==z.keys()
        for k in a:exact(a[k],z[k])
    elif isinstance(a,list):
        assert len(a)==len(z)
        for l,r in zip(a,z):exact(l,r)
    else:assert a==z

round_count=control_count=0

def audit_rounds(x,y,source_ids,locations,expected_rounds=None):
    n,width=x.shape;d=width+1;cols=2*d+1
    active=np.sort(np.unique(source_ids,return_index=True)[1]).astype('<u4')
    initial=active.copy();selected=np.zeros(n,bool);selected[active]=True
    E,_=envelope(x);primal=dual=None;rounds=[];saved=[]
    for ordinal,(where,prefix) in enumerate(locations):
        stem=Path(where)/(prefix+'_round'+str(ordinal));r=json.loads(stem.with_suffix('.json').read_bytes())
        with np.load(stem.with_suffix('.npz'),allow_pickle=False) as archive:
            assert set(archive.files)=={'active_development_indices','added_development_indices','raw_primal','raw_dual','theta','signed_development_lower','signed_development_upper','dual_weights_active'}
            arrays={k:archive[k] for k in archive.files}
        assert r['round']==ordinal and r['active_count']==len(active)
        byte(arrays['active_development_indices'],active)
        assert type(r['value_valid']) is bool and type(r['dual_valid']) is bool
        assert r['model_status_name'] in ('HighsModelStatus.kOptimal','HighsModelStatus.kTimeLimit','HighsModelStatus.kIterationLimit')
        assert r['model_status']=={'HighsModelStatus.kOptimal':7,'HighsModelStatus.kTimeLimit':13,'HighsModelStatus.kIterationLimit':14}[r['model_status_name']]
        if r['model_status']==7:assert r['value_valid'] and r['dual_valid']
        raw=arrays['raw_primal'];rd=arrays['raw_dual']
        assert raw.dtype==rd.dtype==np.dtype('<f8') and np.isfinite(raw).all() and np.isfinite(rd).all()
        assert raw.shape==((cols,) if r['value_valid'] else (0,)) and rd.shape==((len(active)+1,) if r['dual_valid'] else (0,))
        theta=np.zeros(d,'<f8');divisor=1.;lo=np.zeros(n,'<f8');hi=lo.copy();margin=None
        if r['value_valid']:
            theta=raw[:d]-raw[d:2*d]
            divisor=max(1.,math.nextafter(math.nextafter(norm_upper(theta),math.inf)*(1.+2.**-40),math.inf))
            theta=theta/divisor;assert norm_upper(theta)<=1.
            l,h,_=intervals(x,theta);lo=np.where(y>0,l,-h);hi=np.where(y>0,h,-l);margin=float(lo.min())
            if primal is None or margin>primal['lower']:primal={'round':ordinal,'raw':raw.copy(),'theta':theta.copy(),'lower':margin}
        w=np.zeros(len(active),'<f8');bound=None
        if r['dual_valid']:
            w=np.maximum(0.,-rd[1:]);maximum=float(w.max())
            if maximum>0.:w/=maximum
            _,_,_,_,bound=dual_bound(x[active],y[active],w)
            if bound is not None and (dual is None or bound<dual['bound']):dual={'round':ordinal,'raw':rd.copy(),'active':active.copy(),'weights':w.copy(),'bound':bound}
        for k,wanted in (('theta',theta),('signed_development_lower',lo),('signed_development_upper',hi),('dual_weights_active',w)):byte(arrays[k],wanted)
        for k,v in (('candidate_divisor',divisor),('candidate_unit_L1_norm_upper',norm_upper(theta)),('complete_development_margin_lower',margin),('subset_dual_full_margin_upper',bound)):exact(r[k],v)
        if margin is not None and bound is not None:assert max(0.,margin)<=bound
        added=arrays['added_development_indices'];assert added.dtype==np.dtype('<u4') and added.shape==(r['added_count'],)
        if len(added):
            assert r['stop_or_add']=='add_complete_scan_violations'
            eligible=np.flatnonzero(~selected);eligible=eligible[lo[eligible]<=E]
            wanted=eligible[np.lexsort((eligible,lo[eligible]))[:256]].astype('<u4');byte(added,wanted)
        if primal and primal['lower']>E:assert r['stop_or_add']=='complete_development_lower_exceeds_uniform_envelope'
        elif dual and dual['bound']<=E:assert r['stop_or_add']=='verified_dual_excludes_uniform_margin_requirement'
        assert r['run_status'] in ('HighsStatus.kOk','HighsStatus.kWarning')
        assert r['simplex_nit']>=0 and 0.<r['time_limit_wall_remainder']<=25.
        assert 0.<=r['solver_run_seconds_before']<=r['solver_run_seconds_after']
        assert 0.<=r['wall_seconds']<=600.
        saved.append(arrays);rounds.append(r)
        if ordinal+1<len(locations):
            assert len(added)>0;active=np.concatenate((active,added));assert len(np.unique(active))==len(active);selected[added]=True
        guard()
    if expected_rounds is not None:exact(rounds,expected_rounds)
    theta=primal['theta'] if primal else np.zeros(d,'<f8');weights=np.zeros(n,'<f8');indices=np.empty(0,'<u4')
    if dual:indices=dual['active'];weights[indices]=dual['weights']
    return primal,dual,theta,weights,indices,rounds,initial,saved

# Mathematical controls on saved LP witnesses only, ZERO solver calls.
controls=json.loads((R3_OUT/'controls.json').read_bytes());exact(controls,r483['controls'])
old_controls=json.loads((ORIGINAL483_OUT/'controls.json').read_bytes());exact(controls['retained_original_controls'],old_controls)
for control_id,(values,labels,optimum,source_ids) in enumerate((([-1.,1.],[-1,1],1.,[0,1]),([-1.,0.,1.],[1,-1,1],0.,[0,1,0]),([-1.,0.,1.],[1,1,1],1.,[0,1,2]))):
    x=np.array(values,'<f4')[:,None];y=np.array(labels,'<i1');result=old_controls['solver_analytic_controls'][control_id]
    loc=[(ORIGINAL483_OUT,'control'+str(control_id))]*len(result['solver']['rounds'])
    pr,du,theta,w,indices,rounds,initial,saved=audit_rounds(x,y,np.array(source_ids),loc,result['solver']['rounds'])
    assert pr and du and abs(float(pr['raw'][-1])-optimum)<=1e-8 and du['bound']==result['dual_upper']
    assert optimum<=du['bound']+1e-12 and du['bound']-optimum<=1e-8
    if optimum>0.:
        v=physical(x,theta.astype('<f4'));assert np.all(np.where(y>0,v[2],-v[3])>0.)
    control_count+=len(rounds)
x=np.array([[-1.,0.],[1.,2.**-32]],'<f4');y=np.array([-1,1],'<i1')
pr,du,theta,w,indices,rounds,initial,saved=audit_rounds(x,y,np.array([0,1]),[(SOURCE_R2,'warning_control')],controls['warning_control']['rounds'])
assert pr and du and float(pr['raw'][-1])==1. and 1.<=du['bound']<=1.+1e-8
assert du['bound']==controls['warning_control_original_dual_upper']
v=physical(x,theta.astype('<f4'));assert np.all(np.where(y>0,v[2],-v[3])>0.)
control_count+=1;assert control_count==5
assert controls['new_control_LP_calls']==0 and controls['retained_R2_warning_control_LP_calls']==1

def matrix_rows(x,y):
    h=np.empty((len(x),x.shape[1]+1),'<f8');h[:,:-1]=x;h[:,-1]=1.;h*=y[:,None]
    return np.column_stack((-h,h,np.ones(len(x))))
def check_interface(event,matrix,operation):
    assert event['operation']==operation and (event['rows'],event['columns'])==matrix.shape
    assert event['status'] in ('HighsStatus.kOk','HighsStatus.kWarning')
    assert event['readback_format'] in ('MatrixFormat.kColwise','MatrixFormat.kRowwise')
    assert event['original_dense_sha256']==hashlib.sha256(matrix.tobytes()).hexdigest()
    mask=(matrix!=0.)&(np.abs(matrix)<=1e-9);canonical=matrix.copy();canonical[mask]=0.;canonical[canonical==0.]=0.
    assert event['verified_dense_sha256']==hashlib.sha256(canonical.tobytes()).hexdigest()
    assert event['readback_nonzeros']==int(np.count_nonzero(canonical))
    removed=[{'row':int(i),'column':int(j),'F64_bits':int(matrix[i,j].view('<u8')),'value':float(matrix[i,j])} for i,j in np.argwhere(mask)]
    exact(removed,event['deleted_coefficients'])
    return len(removed)
m=np.zeros((2,7),'<f8');m[0,:6]=1.;m[1:]=matrix_rows(x[:1],y[:1])
warning_events=controls['warning_control']['interface_events'];assert len(warning_events)==2
check_interface(warning_events[0],m,'passModel');m=np.vstack((m,matrix_rows(x[1:],y[1:])))
assert check_interface(warning_events[1],m,'analytic_tiny_addRows')==2 and warning_events[1]['status']=='HighsStatus.kWarning'
assert 'ignored' in (SOURCE_R2/'warning_control_solver.log').read_text()
with localcontext() as c:
    c.prec=100;x_cancel=np.array([[1e20,1.,-1e20]],'<f4');l,h,_=intervals(x_cancel,np.array([1.,1.,1.,0.],'<f8'))
    exact_cancel=sum(Decimal(float(v)) for v in x_cancel[0]);assert Decimal(float(l[0]))<=exact_cancel<=Decimal(float(h[0]))
gates['ALL5_retained_control_witnesses_model_readback_records_IEEE_Decimal_no_LP_replay']=True
checkpoint(retained_control_LPs_mathematically_checked=control_count)

physical_pass=[];uniform_excluded=[];interval_fields=root_fields=0;best_brackets=[]
assert [v['bank'] for v in r483['banks']]==list(range(12))
for bank,record in enumerate(r483['banks']):
    uid=np.flatnonzero(unique['meta'][:,2]==bank);row=unique[uid];nodes=tree[bank]['nodes'];root=nodes[0]
    groups=[sorted(nodes[root[side]]['ids']) for side in ('left','right')];maxima=[];winners=[]
    for side,ids in enumerate(groups):
        scores=row['score'][:,ids];at=np.argmax(scores,axis=1)
        maximum=scores[np.arange(len(uid)),at];winner=np.array(ids,'<u2')[at]
        byte(maximum,target['nodes']['maximum'][uid,0,side]);byte(winner,target['nodes']['winner'][uid,0,side])
        maxima.append(maximum);winners.append(winner);root_fields+=2*len(uid)
    right=(maxima[1]>maxima[0])|((maxima[1]==maxima[0])&(winners[1]<winners[0]))
    assert np.array_equal(right,np.isin(row['meta'][:,7],groups[1]))
    labels=(2*right.astype('<i1')-1).astype('<i1');local=np.flatnonzero(dev[uid]);x=row['input'];xd=x[local];yd=labels[local]
    solver=record['solver'];count=len(solver['rounds']);assert 1<=count<=8
    locations=[(ORIGINAL483_OUT if bank<7 or (bank==7 and k==0) else R3_OUT,'bank'+str(bank)) for k in range(count)]
    pr,du,theta,weights,indices,rounds,initial,saved=audit_rounds(xd,yd,row['meta'][local,7],locations,solver['rounds'])
    round_count+=count
    assert solver['primal_round']==(pr['round'] if pr else -1) and solver['dual_round']==(du['round'] if du else -1)
    assert solver['available']==bool(pr) and solver['initial_active_count']==len(initial)
    chosen=rounds[pr['round']] if pr else rounds[-1];optimal=chosen['model_status']==7
    assert solver['status']==(0 if optimal else 1) and solver['success']==optimal
    exact(solver['fun'],-float(pr['raw'][-1]) if pr else None);exact(solver['message'],chosen['message']);exact(solver['nit'],chosen['simplex_nit'])
    if bank<7:
        assert record['LP_seconds'] is None and solver['search_wall_seconds'] is None and not solver['interface_events']
        exact(record['original_LP_seconds_bounds'],[rounds[-1]['wall_seconds'],old_end483[bank]-old_start483[bank]])
        assert 0.<record['new_reconstruction_seconds']<=600.
        assert record['model_path']==str((ORIGINAL483_OUT/('bank'+str(bank)+'_witness.npz')).resolve())
        exact(solver['options_verified'],old_controls['solver_analytic_controls'][0]['solver']['options_verified'])
    else:
        assert record['LP_seconds']>0. and record['original_LP_seconds_bounds'] is None
        options={'solver':'simplex','presolve':'on','parallel':'off','threads':1,'random_seed':0,
                 'simplex_strategy':1,'simplex_dual_edge_weight_strategy':1,'primal_feasibility_tolerance':1e-9,
                 'dual_feasibility_tolerance':1e-9,'output_flag':True,'log_to_console':False,'small_matrix_value':1e-9,
                 'log_file':str((R3_OUT/('bank'+str(bank)+'_solver.log')).resolve())}
        exact(solver['options_verified'],options)
        events=solver['interface_events'];assert events
        active=saved[1]['active_development_indices'] if bank==7 and count>1 else np.concatenate((saved[0]['active_development_indices'],saved[0]['added_development_indices'])) if bank==7 else initial
        model=np.zeros((len(active)+1,1539),'<f8');model[0,:1538]=1.;model[1:]=matrix_rows(xd[active],yd[active])
        check_interface(events[0],model,'passModel')
        for event in events[1:]:
            ordinal=int(event['operation'].removeprefix('addRows_after_round'));assert 0<=ordinal<count
            added=saved[ordinal]['added_development_indices'];assert len(added)>0
            model=np.vstack((model,matrix_rows(xd[added],yd[added])));check_interface(event,model,'addRows_after_round'+str(ordinal))
        paid=paid_bank7_seconds if bank==7 else 0.
        for k in range(1 if bank==7 else 0,count):assert rounds[k]['time_limit_wall_remainder']<=25.-paid
    lower,upper,sl,sh,bound=dual_bound(xd,yd,weights);E,B=envelope(xd)
    ideal_lo,ideal_hi,_=intervals(x,theta);physical_fields=physical(x,theta.astype('<f4'))
    fields=np.column_stack((ideal_lo,ideal_hi,*physical_fields));byte(fields,cert['value'][uid]);interval_fields+=fields.size
    meta=np.column_stack((uid,np.full(len(uid),bank),row['meta'][:,7],right,maxima[0]==maxima[1],np.full(len(uid),bool(pr)))).astype('<u4')
    byte(meta,cert['meta'][uid])
    witness={'development_UIDs':uid[local].astype('<u4'),'development_labels':yd,'raw_primal':pr['raw'] if pr else np.empty(0,'<f8'),
      'raw_dual':du['raw'] if du else np.empty(0,'<f8'),'dual_source_active_development_indices':indices,'theta':theta,'theta_F32':theta.astype('<f4'),
      'dual_weights':weights,'dual_residual_lower':lower,'dual_residual_upper':upper,'dual_weight_sum_bounds':np.array([sl,sh]),
      'dual_bound':np.array([bound if bound is not None else 0.]),'dual_available':np.array([bound is not None],'<u4')}
    with np.load(record['model_path'],allow_pickle=False) as archive:
        assert set(archive.files)==set(witness)
        for k,v in witness.items():byte(archive[k],v)
    sign=np.where(labels>0,fields[:,4],-fields[:,5]);ideal_sign=np.where(labels>0,ideal_lo,-ideal_hi)
    dev_pass=bool(pr) and bool(np.all(sign[local]>0));all_pass=bool(pr) and bool(np.all(sign>0));excluded=bound is not None and bound<=E
    for k,v in (('unique_queries',len(uid)),('development_examples',len(local)),('consumed_validation_examples',int(val[uid].sum())),
       ('candidate_unit_L1_norm_upper',norm_upper(theta)),('candidate_development_min_ideal_margin_lower',float(ideal_sign[local].min()) if pr else None),
       ('dual_unit_L1_margin_upper',bound),('uniform_F32_export_and_arithmetic_envelope',E),('development_max_augmented_coordinate',B),
       ('every_development_physical_sign_sufficient',dev_pass),('every_consumed_domain_physical_sign_sufficient',all_pass),
       ('uniform_sufficient_envelope_unattainable_on_development',excluded)):exact(record[k],v)
    if bound is not None:assert max(0.,float(ideal_sign[local].min()))<=bound
    physical_pass.append(all_pass);uniform_excluded.append(excluded);best_brackets.append({'bank':bank,'lower':max(0.,float(ideal_sign[local].min())),'upper':bound,'uniform_sufficient_envelope':E})
    checkpoint(bank_ALL_rounds_witness_intervals_admitted=bank,rounds=count)
    ieee();guard();process_gate()
assert round_count==93 and root_fields==955488 and interval_fields==1910976
gates['ALL93_root_rounds_original_labels_best_witnesses_full_UID_interval_BYTES_and_dual_bounds']=True

phase='independent_ALL_views_complete_denominators_and_science_scope'
expected={k:[] for k in ('unique_views','occurrence_views','book_views','source_ID_views')}
flags=cert['meta'][:,5]!=0;labels=2*cert['meta'][:,3].astype('<i4')-1
signed_lo=np.where(labels>0,cert['value'][:,4],-cert['value'][:,5]);signed_hi=np.where(labels>0,cert['value'][:,5],-cert['value'][:,4])
def summary(ids):
    accepted=ids[flags[ids]]
    return {'count':len(ids),'available_candidate_count':len(accepted),'source_ties':int(np.count_nonzero(cert['meta'][ids,4])),
      'strict_physical_sign_sufficient':int(np.count_nonzero(flags[ids]&(signed_lo[ids]>0))),
      'strict_physical_wrong_sign':int(np.count_nonzero(flags[ids]&(signed_hi[ids]<0))),
      'physical_sign_not_proved':int(np.count_nonzero(flags[ids]&(signed_lo[ids]<=0))),
      'min_signed_physical_lower':float(signed_lo[accepted].min()) if len(accepted) else None,
      'max_arithmetic_error_upper':float(cert['value'][accepted,7].max()) if len(accepted) else None}
for bank in [-1,*range(12)]:
    eligible=np.ones(U,bool) if bank<0 else unique['meta'][:,2]==bank
    for role,mask in (('ALL',np.ones(U,bool)),('development',dev),('consumed_validation',val)):
        expected['unique_views'].append({'bank':bank,'role':role,**summary(np.flatnonzero(eligible&mask))})
for bank in range(12):
    for role in range(3):
        for mode in range(2):
            rows=links[(links[:,6]==bank)&(links[:,5]==role)&(links[:,4]==mode)]
            expected['occurrence_views'].append({'bank':bank,'role':role,'mode':mode,'accepted':int(rows[:,10].sum(dtype='<u8')),'rejected':int(np.count_nonzero(rows[:,10]==0)),**summary(rows[:,12])})
            for expert in range(128):expected['source_ID_views'].append({'bank':bank,'role':role,'mode':mode,'source_ID':expert,**summary(rows[rows[:,9]==expert,12])})
for book in range(192):
    for bank in range(12):
        for mode in range(2):
            rows=links[(links[:,2]==book)&(links[:,6]==bank)&(links[:,4]==mode)]
            expected['book_views'].append({'book':book,'bank':bank,'mode':mode,'role':0 if book<64 else 1 if book<128 else 2,'accepted':int(rows[:,10].sum(dtype='<u8')),'rejected':int(np.count_nonzero(rows[:,10]==0)),**summary(rows[:,12])})
assert [len(expected[k]) for k in expected]==[39,72,4608,9216]
exact(expected,json.loads((R3_OUT/'complete_reports.json').read_bytes()));exact(expected['unique_views'][0],r483['overall'])
for k in ('occurrence_views','book_views','source_ID_views'):assert sum(v['count'] for v in expected[k])==Q
exact(json.loads((R3_OUT/'bank_records.json').read_bytes()),r483['banks'])
decisions={'all12_roots_consumed_domain_physical_sign_sufficient':all(physical_pass),
 'any_root_uniform_sufficient_envelope_unattainable_on_development':any(uniform_excluded),
 'every_retained_solver_round_optimal':all(r['model_status']==7 for bank in r483['banks'] for r in bank['solver']['rounds'])}
exact(decisions,r483['decisions']);assert not decisions['all12_roots_consumed_domain_physical_sign_sufficient'] and not decisions['any_root_uniform_sufficient_envelope_unattainable_on_development']
gates['ALL39_72_4608_9216_views_full_roles_empty_ID_slots_decisions_exact']=True
for rel,item in prior['preserved_unrelated_files'].items():assert digest(ROOT/rel)==item['sha256']
last_status=status();ieee();process_gate()
assert parent.cpu_affinity()==[0]
assert not ({'scipy','torch','transformers','tensorflow','sklearn','pandas','meth483_constraint_geometry','meth483_r1_constraint_geometry','meth483_constraint_root','meth483_r1_constraint_root','meth483_r2_constraint_root','meth483_r3_constraint_root','meth482_affine_math'}&set(sys.modules))
gates['resources_preservation_no_solver_helper_model_control_replay_no_claim_promotion']=True
guard()
retention={'experiment':'METH483 independent complete retained active-row outcome admission','raw_sha256':RAW483_SHA,'windows_terminal_sha256':R3_EVENT_SHA,
 'audit_source_sha256':digest(__file__),'audit_protocol_sha256':digest(PROTO483),'source_binding_sha256':BIND_SHA,
 'start_utc':start_utc,'end_utc':datetime.now(timezone.utc).isoformat(),'process_instance':instance,'commands':[],
 'gates':gates,'decision':'ALL12_ROOT_PRIMAL_DUAL_BRACKETS_INCONCLUSIVE_INDEPENDENTLY_ADMITTED','decisions':decisions,
 'unique_inputs_audited':U,'queries_audited':Q,'root_rounds_audited':round_count,'control_rounds_audited':control_count,
 'root_maximum_and_ID_fields_BYTE_checked':root_fields,'interval_FIELDS_BYTE_checked':interval_fields,'meta_FIELDS_BYTE_checked':U*6,
 'saved_candidate_heads_audited':12,'complete_view_counts':[39,72,4608,9216],'brackets':best_brackets,'overall':expected['unique_views'][0],
 'LP_replays':0,'model_native_replays':0,'helper_imports':0,'tracked_status':last_status,
 'scope':'Full finite consumed roots/retained rounds. Original-input interval witnesses mathematically valid but no sufficient all-root head or dual exclusion of uniform requirement. No affine impossibility, full tree/mass/C/LUT/new-n/quality/rate/DRAM/family claim.',
 'resource':{'seconds_before_RET':time.monotonic()-START,'OS_peak_bytes':peak,'file_bytes_hashed':hashed,'hard_seconds':600,'hard_peak_bytes':4<<30,'hard_new_bytes':32<<20}}
assert len(gates)==5 and all(gates.values())
write(RET,retention);guard();checkpoint(terminal=True,retention_sha256=digest(RET),OS_peak_bytes=peak)
watchdog.cancel()
print(json.dumps({'retention':str(RET),'sha256':digest(RET),'gates':gates,'decision':retention['decision'],'seconds':time.monotonic()-START,'OS_peak_bytes':peak}),flush=True)
progress.close();faulthandler.disable();fatal.close()
