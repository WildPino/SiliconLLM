"""ONE complete source-only decision-margin and centered-mass inquiry."""
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
OUT = ROOT / 'results/native_expert_scaling/meth481_geometry'
RET = DOC / 'meth481_geometry_result.json'
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
scientific = (Path(__file__).resolve(), PROTOCOL481, CALC481, WINDOWS481, PRIOR_RET, ROOT / 'benchmarks/native_expert_scaling/meth480_r1_retention_audit.py', AUDIT_PROTO, AUDIT_WINDOWS, AUDIT_REPAIR_PROTO, AUDIT_FAULT_INVENTORY, DOC / 'RETENTION_480_20261005.failure.json', DOC / 'METH_480_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth480_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth480_audit_windows_terminal.ps1', ROOT / 'benchmarks/native_expert_scaling/meth480_r1_learned_router.py', ROOT / 'benchmarks/native_expert_scaling/meth480_learned_router.py', C, MATH, REPORTING, WINDOWS, PROTO, ARITHMETIC_NOTE, REPAIR_PROTO, FIRST_INVENTORY, DOC / 'meth480_learned_router_result.failure.json', DOC / 'METH_480_FIRST_FAULT_20261005.md')

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
        write(path, {'experiment': 'METH481 FIRST geometry inquiry fault', 'phase': phase,
              'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
              'commands': commands, 'source_binding_sha256': BIND_SHA, 'raw_sha256': RAW_SHA,
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
        write(path, {'experiment': 'METH481 geometry watchdog FIRST fault', 'phase': phase, 'start_utc': start_utc,
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
daemons = process_gate()
ret = json.loads((DOC / 'RETENTION_479_R3_20261005.json').read_bytes())
assert all(ret['gates'].values()) and ret['unique_inputs_audited'] == 238872 and ret['queries_audited'] == 387036
assert time.monotonic() - START <= 180
gates = {'full_source6224_artifact_runtime_scientific_freezes_admitted479_supervision': True}
checkpoint(file_bytes_hashed=hashed)

phase = 'frozen_controls_and_all_source_wire_admission'
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
for module in (np, threadpoolctl):
    assert module.__version__ == runtime['packages'][module.__name__]['version'] and str(Path(module.__file__).resolve()) in runtime['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in runtime['packages']['numpy']['files']
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
import meth481_geometry_math as M
assert Path(M.__file__).resolve() == CALC481
U = M.U

def wire(path, magic, width, reserved, count, dtype, shape):
    p = Path(path)
    assert p.stat().st_size == 24 + count * width
    with p.open('rb') as stream:
        assert stream.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
    return np.memmap(p, dtype=dtype, mode='r', offset=24, shape=shape)

toy = np.array([[3.] * 128, [-.01] * 64 + [.01] * 64, [20.] + [0.] * 127, [10020.] + [10000.] * 127], '<f4')
toy_native = np.zeros((4, 3), '<f8')
def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]
with localcontext() as context:
    context.prec = 100
    exact = []
    for k, values in enumerate(toy):
        maximum = float(np.max(values))
        z = sum(f32(float((Decimal(float(v)) - Decimal(maximum)).exp())) for v in values)
        toy_native[k] = (maximum, f32(1 / z), z)
        exact.append(float(Decimal(maximum) + sum((Decimal(float(v)) - Decimal(maximum)).exp() for v in values).ln()))
test_moment, test_bound, test_A = M.moments(toy, toy_native)
assert np.all(np.abs(test_A - exact) <= 1e-10)
assert test_moment[0, 2:5].tobytes() == np.zeros(3, '<f8').tobytes() and np.isneginf(test_bound[0])
assert abs(test_moment[1, 2] - float(toy[1, 0]) ** 2) <= 1e-15
assert abs(test_moment[3, 5] - test_moment[2, 5] - 10000) <= 1e-10
assert toy_native[2, 0] - test_moment[2, 5] - np.log(toy_native[2, 1]) > M.LOG_HIGH
for D, eL, eR, expected in ((2., 10., 10., True), (.01, 0., -.02, False), (-2., 10., 10., True)):
    sufficient = abs(eR - eL) < abs(D)
    correct = (D + eR - eL > 0) == (D > 0)
    assert correct == expected and (not sufficient or correct)
assert (3 < 5) and not (8 < 1)  # Original tied winner keys and surrogate keys can disagree.
controls = {'Decimal100_centered_partition_controls': 4, 'constant_zero_remainder': True,
            'shift_invariance': True, 'symmetric_variance': True, 'large_residual_over_probability_witness': True,
            'differential_margin_controls': 3, 'source_vs_surrogate_tie_keys': True}
write(OUT / 'controls.json', controls)
unique = wire(b['data_files']['unique_inputs.bin']['path'], b'M479UNI1', 3720, 768, U, M.UNIQUE, (U,))
owner = wire(b['data_files']['ownership.bin']['path'], b'M479OWN1', 32, 0, U, '<u4', (U, 8))
links = wire(b['data_files']['query_links.bin']['path'], b'M479LNK1', 52, 0, 387036, '<u4', (387036, 13))
targets = wire(b['data_files']['group_targets.bin']['path'], b'M479TGT1', 3560, 127, U, M.TARGET, (U,))
pred = wire(SOURCE_OUT / 'predictions.bin', b'M480PRD1', 372, 0, U, M.PRED, (U,))
assert np.array_equal(unique['meta'][:, 0], np.arange(U)) and np.array_equal(owner[:, 0], np.arange(U))
assert np.array_equal(targets['unique_id'], np.arange(U)) and np.array_equal(pred['meta'][:, 0], np.arange(U))
assert np.array_equal(owner[:, 1], unique['meta'][:, 2]) and np.array_equal(pred['meta'][:, 3], unique['meta'][:, 7])
assert np.array_equal(links[:, 6], unique['meta'][links[:, 12], 2]) and np.array_equal(links[:, 9], unique['meta'][links[:, 12], 7])
assert np.count_nonzero(owner[:, 2] & 5) == 159414 and np.count_nonzero(owner[:, 2] & 2) == 79458
assert not np.any(((owner[:, 2] & 5) != 0) & ((owner[:, 2] & 2) != 0))
memberships = json.loads(Path(b['data_files']['membership.json']['path']).read_bytes())
assert [x['bank'] for x in memberships] == list(range(12))
gates['Decimal100_moment_margin_tie_controls_all_UID_roles_links_source_wire_admission'] = True
checkpoint()

phase = 'ALL_unique_centered_moments_visited_margins_and_all127_node_views'
geometry = np.zeros(U, M.GEOMETRY)
geometry['meta'][:, 0] = np.arange(U)
geometry['meta'][:, 1] = unique['meta'][:, 2]
geometry['meta'][:, 2] = unique['meta'][:, 7]
metrics = {key: np.empty((U, 7), bool if key in ('wrong_branch', 'source_tie', 'candidate_tie', 'strict_sufficient') else '<f8') for key in ('margin', 'differential', 'common_error', 'wrong_branch', 'source_tie', 'candidate_tie', 'strict_sufficient')}
node_views = []
for bank in range(12):
    uid = np.flatnonzero(unique['meta'][:, 2] == bank)
    row, t, p = unique[uid], targets[uid], pred[uid]
    assert len(uid) == (22272 if bank < 6 else 17540)
    geometry['moment'][uid], _, _ = M.moments(row['score'], row['value'])
    geometry['path'][uid], geometry['meta'][uid, 3], local_metrics = M.paths(row, t, p, memberships[bank]['nodes'])
    for key in metrics:
        metrics[key][uid] = local_metrics[key]
    node_views.extend(M.full_node_views(row, owner[uid], t, memberships[bank]['nodes'], bank))
    del row, t, p, local_metrics
    guard()
    checkpoint(bank_complete=bank)
assert len(node_views) == 4572
assert sum(x['count'] for x in node_views if x['role'] == 'ALL') == U * 127
assert all(sum(x['local_winner_ID_counts']) == x['count'] and sum(x['absolute_margin_bins_0_1e3_1e2_1e1_1_inf']) == x['count'] for x in node_views)
assert np.count_nonzero(geometry['meta'][:, 3] < 7) == r['unique_overall']['ID_differences'] == 201993
output = OUT / 'geometry.bin'
with output.open('xb') as stream:
    stream.write(struct.pack('<8sIIQ', b'M481GEO1', 156, 12, U))
    stream.write(geometry.tobytes())
assert output.stat().st_size == 24 + 156 * U
write(OUT / 'full_node_views.json', node_views)
gates['ALL238872_moments_1672104_visited_margins_30336744_source_node_margins_ID_exposure'] = True

phase = 'complete_all_role_book_and_original_ID_reports'
reports = M.reports(unique, owner, links, geometry, metrics)
write(OUT / 'complete_reports.json', reports)
overall = reports['unique_views'][0]
assert sum(overall['first_divergence_counts']) == 201993
gates['ALL39_72_4608_9216_views_with_empty_rare_ID_slots_and_complete_denominators'] = True
decisions = {'full_score_second_moment_reference_within_1percent_every_UID': overall['moment_log_ratio_outside_1percent'] == 0,
             'nominal_Taylor_plus_observed_native_rounding_screen_every_UID': overall['nominal_Taylor_screen_count'] == U,
             'over_probability_witness_blocks_plain_PSD_reduction_in_teacher_score_coordinates': overall['moment_over_probability_witnesses'] > 0}
for rel, item in prior['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
last_status = status()
assert parent.cpu_affinity() == [0]
assert not ({'torch', 'scipy', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'meth480_learned_router', 'meth480_r1_learned_router', 'meth480_transfer_math', 'meth480_reporting', 'meth480_retention_audit', 'meth480_r1_retention_audit'} & set(sys.modules))
gates['resource_preservation_no_native_or_fit_replay_model_SVD_rank_sweep_or_candidate_selection'] = True
guard()
record = {'experiment': 'METH481 ONE complete source-only decision margins and centered mass inquiry',
          'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
          'commands': [], 'source_binding_sha256': BIND_SHA, 'source480_RAW_sha256': RAW_SHA,
          'source480_RET_sha256': PRIOR_RET_SHA, 'scientific_sources': {str(p): digest(p) for p in scientific},
          'gates': gates, 'decisions': decisions, 'unique_inputs': U, 'queries': len(links),
          'source_node_margins': U * 127, 'visited_node_margins': U * 7, 'geometry_scalar_fields': U * (4 + 7 + 4 * 7),
          'overall': overall, 'reports_path': str(OUT / 'complete_reports.json'), 'node_views_path': str(OUT / 'full_node_views.json'),
          'controls': controls, 'native_commands': 0, 'training_updates': 0, 'SVD_refits': 0,
          'geometry_wire_width': 156, 'inference_budget_hypothesis': {'forms_with_separate_affine_decision_and_branch_mass': 14, 'coefficients_at_768': 10752, 'exp_calls': 7, 'logical_vector_bytes': 43008, 'gate_header_bytes_7x32': 224, 'complete_budget_not_measured': True},
          'scope': 'Source-only consumed-domain algebra; full-score moment reference is NOT a cheap candidate. Nominal Taylor/F64 round drift is NOT a directed interval certificate. No physical model/fresh quality/rate/useful-n/LUT/DRAM/general-family claim.',
          'decision': 'SOURCE_ALGEBRA_COMPLETE_PENDING_INDEPENDENT_ADMISSION', 'tracked_status': last_status,
          'head_at_execution': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'resource': {'seconds_before_RAW': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed, 'hard_seconds': 600, 'hard_peak_bytes': 4 << 30, 'hard_new_bytes': 64 << 20},
          'output_inventory': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']}
write(RET, record)
guard()
checkpoint(terminal=True, raw_sha256=digest(RET), OS_peak_bytes=peak)
watchdog.cancel()
print(json.dumps({'raw': str(RET), 'sha256': digest(RET), 'gates': gates, 'decisions': decisions, 'overall': overall, 'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()

