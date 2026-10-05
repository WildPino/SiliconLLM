"""Independent ALL saved updates, physical C fields and reports; no training/native replay."""
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
OUT = ROOT / 'results/native_expert_scaling/meth480_r1_retention_audit'
RET = DOC / 'RETENTION_480_R1_20261005.json'
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
scientific = (Path(__file__).resolve(), AUDIT_PROTO, AUDIT_WINDOWS, AUDIT_REPAIR_PROTO, AUDIT_FAULT_INVENTORY, DOC / 'RETENTION_480_20261005.failure.json', DOC / 'METH_480_FIRST_AUDIT_FAULT_20261005.md', ROOT / 'benchmarks/native_expert_scaling/meth480_retention_audit.py', ROOT / 'benchmarks/native_expert_scaling/meth480_audit_windows_terminal.ps1', ROOT / 'benchmarks/native_expert_scaling/meth480_r1_learned_router.py', ROOT / 'benchmarks/native_expert_scaling/meth480_learned_router.py', C, MATH, REPORTING, WINDOWS, PROTO, ARITHMETIC_NOTE, REPAIR_PROTO, FIRST_INVENTORY, DOC / 'meth480_learned_router_result.failure.json', DOC / 'METH_480_FIRST_FAULT_20261005.md')

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
    assert time.monotonic() - START <= 1200 and peak <= 4 << 30, (phase, peak)
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
        write(path, {'experiment': 'METH480-R1 FIRST independent retention fault', 'phase': phase,
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
        write(path, {'experiment': 'METH480-R1 independent audit watchdog FIRST fault', 'phase': phase, 'start_utc': start_utc,
                     'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
                     'commands': commands, 'source_binding_sha256': BIND_SHA, 'reason': '1200 second deadline',
                     'resource': {'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak}})
    os._exit(3)
watchdog = threading.Timer(max(.001, 1200 - (time.monotonic() - START)), deadline)
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

phase = 'independent_wire_roles_book_weights_controls_and_Windows'
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
for module in (np, threadpoolctl):
    assert module.__version__ == runtime['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in runtime['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in runtime['packages']['numpy']['files']
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
U = 238872
UNIQUE = np.dtype([('meta', '<u4', (12,)), ('input_sha', 'u1', (32,)), ('score_sha', 'u1', (32,)),
                   ('value', '<f8', (3,)), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
NODE = np.dtype([('maximum', '<f4', (2,)), ('winner', '<u2', (2,)), ('mass', '<f8', (2,))])
TARGET = np.dtype([('unique_id', '<u4'), ('nodes', NODE, (127,))])
PRED = np.dtype([('meta', '<u4', (12,)), ('value', '<f4', (3,)), ('root', '<f8', (4,)),
                 ('vector_ids', '<u2', (42,)), ('score', '<f4', (42,)), ('direction', '<u4', (7,))])
assert (UNIQUE.itemsize, TARGET.itemsize, PRED.itemsize) == (3720, 3560, 372)

def wire(path, magic, width, reserved, count, dtype, shape):
    path = Path(path)
    assert path.stat().st_size == 24 + width * count
    with path.open('rb') as stream:
        assert stream.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
    return np.memmap(path, mode='r', offset=24, dtype=dtype, shape=shape)

unique = wire(b['data_files']['unique_inputs.bin']['path'], b'M479UNI1', 3720, 768, U, UNIQUE, (U,))
owner = wire(b['data_files']['ownership.bin']['path'], b'M479OWN1', 32, 0, U, '<u4', (U, 8))
links = wire(b['data_files']['query_links.bin']['path'], b'M479LNK1', 52, 0, 387036, '<u4', (387036, 13))
target = wire(b['data_files']['group_targets.bin']['path'], b'M479TGT1', 3560, 127, U, TARGET, (U,))
pred = wire(SOURCE_OUT / 'predictions.bin', b'M480PRD1', 372, 0, U, PRED, (U,))
source_dtype = np.dtype([('meta', '<u4', (12,)), ('value', '<f8', (3,))])
source = wire(b['data_files']['source_fields.bin']['path'], b'M479SRC1', 72, 0, 387036, source_dtype, (387036,))
assert np.array_equal(links[:, :12], source['meta'])
assert np.array_equal(links[:, 0], np.arange(387036)) and np.all(links[:, 12] < U)
assert np.array_equal(links[:, 5], np.where(links[:, 2] < 64, 0, np.where(links[:, 2] < 128, 1, 2)))
assert np.array_equal(unique['meta'][:, 0], np.arange(U)) and np.all(unique['meta'][:, 1] == 128)
assert np.array_equal(source['value'].view('u1'), unique['value'][links[:, 12]].view('u1'))
assert np.array_equal(links[:, 6], unique['meta'][links[:, 12], 2])
assert np.array_equal(links[:, 9], unique['meta'][links[:, 12], 7])
expected_owner = np.zeros((U, 8), '<u4')
expected_owner[:, 0] = np.arange(U)
expected_owner[:, 1] = unique['meta'][:, 2]
np.bitwise_or.at(expected_owner[:, 2], links[:, 12], (1 << links[:, 5]).astype('<u4'))
np.bitwise_or.at(expected_owner[:, 3], links[:, 12], (1 << links[:, 4]).astype('<u4'))
np.bitwise_or.at(expected_owner[:, 4], links[:, 12], (1 << links[:, 10]).astype('<u4'))
np.add.at(expected_owner[:, 5], links[:, 12], 1)
for book in range(192):
    ids = np.unique(links[links[:, 2] == book, 12])
    expected_owner[ids, 7 if 64 <= book < 128 else 6] += 1
assert np.array_equal(owner, expected_owner)
dev = (owner[:, 2] & 5) != 0
val = (owner[:, 2] & 2) != 0
assert np.count_nonzero(dev) == 159414 and np.count_nonzero(val) == 79458 and not np.any(dev & val)
memberships = json.loads(Path(b['data_files']['membership.json']['path']).read_bytes())
assert [x['bank'] for x in memberships] == list(range(12))

event = json.loads(EVENT.read_bytes())
assert event['query_available'] and event['event_id'] == 1000 and not event['matching_scientific_events']
instances = [r['process_instance']]
for command in r['commands']:
    instances.extend([command['process_instance'], *command['descendant_process_peaks']])
assert {(x['pid'], x['create_time_unix']) for x in event['instances']} == {(x['pid'], x['create_time_unix']) for x in instances}
assert datetime.fromisoformat(event['query_start_utc']) <= datetime.fromisoformat(r['start_utc'])
assert datetime.fromisoformat(event['query_end_utc']) >= datetime.fromisoformat(r['end_utc'])
assert [(x['label'], x['returncode'], x['expected_returncode']) for x in r['commands']] == [('compile', 0, 0), ('controls', 0, 0), ('negative_ledger', 2, 2), ('predict', 0, 0)]
assert all(datetime.fromisoformat(x['end_utc']) >= datetime.fromisoformat(x['start_utc']) for x in r['commands'])
assert (SOURCE_OUT / 'fatal_native.log').stat().st_size == 0
assert not (SOURCE_OUT / 'negative_predictions.bin').exists()
assert (SOURCE_OUT / 'negative_ledger.bin').read_bytes() == b'INVALID!' + struct.pack('<II', 12, 0)
assert (SOURCE_OUT / 'negative_ledger.stderr.log').read_bytes() == b'learned_router_error:model_ledger_magic\r\n'
for label in ('compile', 'controls', 'predict'):
    assert (SOURCE_OUT / (label + '.stderr.log')).stat().st_size == 0
native_controls = [json.loads(x) for x in (SOURCE_OUT / 'controls.stdout.log').read_text().splitlines()]
assert len(native_controls) == 11 and native_controls[8]['rounding'] == 0
assert native_controls[8]['CPU_affinity_mask'] == 1 and not native_controls[8]['MXCSR'] & 0x8040

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

with localcontext() as context:
    context.prec = 100
    toys = ((0, 1, -1), (10000, 9999, -10000), (0, 0, 0), (1000, 999, -1000), (0, -1, -2), (0, 0, -1))
    for k, values in enumerate(toys):
        best = max(range(3), key=lambda j: values[j])
        total = sum(f32(float((Decimal(v) - Decimal(values[best])).exp())) for v in values)
        assert native_controls[k] == {'control': k, 'chosen': best, 'denominator': total, 'probability': f32(1 / total)}
    for k, values in enumerate(toys[:2]):
        best = max(values)
        total = sum(Decimal(w) * (Decimal(v) - Decimal(best)).exp() for w, v in zip((2, 3, 5), values))
        A = float(Decimal(best) + total.ln())
        assert abs(native_controls[k + 9]['A'] - A) <= 1e-10 + 1e-12 * abs(A)
        assert native_controls[k + 9]['weighted_partition_control'] == k
        assert struct.pack('<f', native_controls[k + 9]['probability']) == struct.pack('<f', float(1 / total))
for kind in range(2):
    aa = [(i % 7 - 3) * 2.**-10 if not kind else 2.**-100 if i % 2 else 2.**80 for i in range(768)]
    xx = [(i % 5 - 2) * 2.**-9 if not kind else 2.**-20 if i % 2 else -2.**-80 if i % 4 else 2.**-80 for i in range(768)]
    accum = [0.] * 8
    for dimension in range(768):
        accum[dimension % 8] += aa[dimension] * xx[dimension]
    total = 0.
    for lane in range(4):
        total += accum[lane] + accum[lane + 4]
    assert native_controls[kind + 6] == {'dot_control': kind, 'double_result': total, 'f32_result': f32(total)}
math_controls = json.loads((SOURCE_OUT / 'math_controls.json').read_bytes())
assert math_controls == r['math_controls'] and math_controls['simplex_controls'] == 4 and math_controls['gradient_controls'] == 14
assert math_controls['maximum_gradient_absolute_difference'] <= 1e-8
assert math_controls['Adam_first_step_control'] and math_controls['root_theta_shift_invariance_gradient']
gates['ALL_UID_roles_ownership_equalbook_domains_control_logs_and_actual_Windows_instances'] = True
checkpoint()

phase = 'independent_ALL_saved_gradients_moments_updates_losses_and_physical_models'
differences = {}
optimizer_updates = optimizer_scalar_fields = model_scalar_fields = 0
physical_dots = audited_unique = path_fields = 0
model_vectors, all_groups, all_parameters = [], [], []

def close(actual, expected, label, atol=1e-12, rtol=1e-11):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), label
    error = np.abs(actual - expected)
    maximum = float(np.max(error)) if error.size else 0.
    differences[label] = max(differences.get(label, 0.), maximum)
    assert np.all(error <= atol + rtol * np.abs(expected)), (label, maximum)

def projection(rows):
    # Independent scalar cumulative threshold; no optimizer/fit module import.
    result = np.empty_like(rows)
    for j, row in enumerate(rows):
        ordered = sorted(row, reverse=True)
        accumulated, threshold = 0., None
        for count, item in enumerate(ordered, 1):
            accumulated += item
            trial = (accumulated - 1) / count
            if item > trial:
                threshold = trial
        assert threshold is not None
        result[j] = np.maximum(row - threshold, 0.)
        result[j] /= np.sum(result[j], dtype='<f8')
    return result

def support_loss_gradient(scores, a, expected_max, weight):
    values = np.einsum('ue,je->uj', scores, a, optimize=False)
    selected = (values[:, 1] > values[:, 0]).astype('<u4')
    residual = values[np.arange(len(scores)), selected] - expected_max
    derivative = weight * residual * 2.
    gradient = np.stack([np.einsum('u,ue->e', derivative * (selected == j), scores, optimize=False) for j in range(2)])
    return float(np.sum(weight * residual * residual, dtype='<f8')), gradient

def root_loss_gradient(scores, a, theta, expected_A, weight):
    bt = np.exp(theta - np.max(theta))
    bt /= np.sum(bt, dtype='<f8')
    log_b = np.log(128.) + theta - (np.max(theta) + np.log(np.sum(np.exp(theta - np.max(theta)), dtype='<f8')))
    values = np.einsum('ue,je->uj', scores, a, optimize=False) + log_b
    top = np.max(values, axis=1)
    terms = np.exp(values - top[:, None])
    denominator = np.sum(terms, axis=1, dtype='<f8')
    residual = top + np.log(denominator) - expected_A
    derivative = weight * residual * 2.
    weighted = terms / denominator[:, None] * derivative[:, None]
    ga = np.einsum('uj,ue->je', weighted, scores, optimize=False)
    gt = np.sum(weighted, axis=0, dtype='<f8') - bt * np.sum(derivative, dtype='<f8')
    return float(np.sum(weight * residual * residual, dtype='<f8')), ga, gt, log_b

for bank in range(12):
    record = r['models'][bank]
    assert record['bank'] == bank
    tree = memberships[bank]['nodes']
    assert len(tree) == 255 and tree[0]['ids'] == list(range(128))
    uid = np.flatnonzero((unique['meta'][:, 2] == bank) & dev)
    weight = np.zeros(U, '<f8')
    book_counts = np.zeros(U, '<u4')
    book_rows = []
    for book in [*range(64), *range(128, 192)]:
        ids = np.unique(links[(links[:, 2] == book) & (links[:, 6] == bank), 12])
        assert len(ids) > 0 and np.all(dev[ids])
        weight[ids] += 1 / (128 * len(ids))
        book_counts[ids] += 1
        book_rows.append({'book': book, 'unique_inputs': len(ids)})
    assert np.array_equal(book_counts, owner[:, 6] * (owner[:, 1] == bank))
    assert not np.any(weight[val])
    weight = weight[uid]
    assert len(uid) == (14848 if bank < 6 else 11721) and abs(weight.sum() - 1) <= 1e-12
    assert record['books'] == book_rows and record['training_unique'] == len(uid)
    close(record['training_weight_sum'], weight.sum(), 'book_weight_sum')
    training_dtype = np.dtype([('unique_id', '<u4'), ('omega', '<f8')])
    training = wire(record['training_path'], b'M480TRN1', 12, bank, len(uid), training_dtype, (len(uid),))
    assert np.array_equal(training['unique_id'], uid)
    assert training['omega'].tobytes() == weight.tobytes()
    scores = unique['score'][uid].astype('<f8')
    expected_A = unique['value'][uid, 0] + np.log(unique['value'][uid, 2])
    targets = target[uid]
    assert np.array_equal(targets['unique_id'], uid)
    with np.load(record['history_path'], allow_pickle=False) as archive:
        assert archive.files == ['initial', 'parameter', 'gradient', 'first', 'second', 'loss_before', 'loss_final', 'log_b']
        h = {key: archive[key] for key in archive.files}
    expected_shapes = {'initial': (3344,), 'parameter': (32, 3344), 'gradient': (32, 3344), 'first': (32, 3344), 'second': (32, 3344), 'loss_before': (32, 63), 'loss_final': (63,), 'log_b': (16,)}
    assert all(h[k].shape == expected_shapes[k] and h[k].dtype == np.dtype('<f8') and np.isfinite(h[k]).all() for k in h)
    groups, offset = [], 0
    for ti, node in enumerate([x for x in tree if x['count'] > 1]):
        for child, name in enumerate(('left', 'right')):
            group = tree[node[name]]
            if group['count'] >= 4:
                groups.append({'group': group['node'], 'target_index': ti, 'child': child, 'ids': group['ids'], 'offset': offset, 'size': 2 * group['count'], 'vector_offset': 128 + 2 * len(groups)})
                offset += 2 * group['count']
    assert len(groups) == 62 and offset == 1280 and groups == record['groups']
    initial = np.zeros(3344, '<f8')
    for k, group in enumerate(groups):
        ids, at, size = group['ids'], group['offset'], group['size']
        a0 = np.zeros((2, len(ids)), '<f8')
        a0[0, 0] = a0[1, -1] = 1.
        initial[at:at + size] = a0.ravel()
        s = np.ascontiguousarray(scores[:, ids])
        maximum = np.max(s, axis=1)
        assert maximum.astype('<f4').tobytes() == targets['nodes']['maximum'][:, group['target_index'], group['child']].tobytes()
        for step in range(32):
            previous = a0 if step == 0 else h['parameter'][step - 1, at:at + size].reshape(a0.shape)
            loss, gradient = support_loss_gradient(s, previous, maximum, weight)
            close(h['loss_before'][step, k], loss, 'support_losses', 1e-10, 1e-9)
            close(h['gradient'][step, at:at + size].reshape(a0.shape), gradient, 'support_gradients', 1e-10, 1e-9)
            # Audit each identity from its saved, independently admitted inputs.
            saved_g = h['gradient'][step, at:at + size]
            first = .9 * (np.zeros(size) if step == 0 else h['first'][step - 1, at:at + size]) + .1 * saved_g
            second = .999 * (np.zeros(size) if step == 0 else h['second'][step - 1, at:at + size]) + .001 * saved_g * saved_g
            close(h['first'][step, at:at + size], first, 'support_first_moments')
            close(h['second'][step, at:at + size], second, 'support_second_moments')
            update = .03 * (first / (1 - .9 ** (step + 1))) / (np.sqrt(second / (1 - .999 ** (step + 1))) + 1e-8)
            projected = projection((previous.ravel() - update).reshape(a0.shape))
            close(h['parameter'][step, at:at + size].reshape(a0.shape), projected, 'support_projected_updates')
            assert np.min(h['parameter'][step, at:at + size]) >= 0
            close(h['parameter'][step, at:at + size].reshape(a0.shape).sum(axis=1), np.ones(2), 'support_simplex')
            optimizer_updates += 1
            optimizer_scalar_fields += 4 * size + 1
            guard()
        final_loss, _ = support_loss_gradient(s, h['parameter'][-1, at:at + size].reshape(a0.shape), maximum, weight)
        close(h['loss_final'][k], final_loss, 'support_final_losses', 1e-10, 1e-9)
    c0 = np.zeros((16, 128), '<f8')
    for k, node in enumerate([x for x in tree if x['count'] == 8]):
        c0[k, node['ids']] = 1 / 8
    initial[1280:3328] = c0.ravel()
    assert initial.tobytes() == h['initial'].tobytes()
    for step in range(32):
        previous = initial[1280:] if step == 0 else h['parameter'][step - 1, 1280:]
        a, theta = previous[:2048].reshape(16, 128), previous[2048:]
        loss, ga, gt, _ = root_loss_gradient(scores, a, theta, expected_A, weight)
        close(h['loss_before'][step, 62], loss, 'root_losses', 1e-10, 1e-9)
        close(h['gradient'][step, 1280:], np.r_[ga.ravel(), gt], 'root_gradients', 1e-10, 1e-9)
        saved_g = h['gradient'][step, 1280:]
        first = .9 * (np.zeros(2064) if step == 0 else h['first'][step - 1, 1280:]) + .1 * saved_g
        second = .999 * (np.zeros(2064) if step == 0 else h['second'][step - 1, 1280:]) + .001 * saved_g * saved_g
        close(h['first'][step, 1280:], first, 'root_first_moments')
        close(h['second'][step, 1280:], second, 'root_second_moments')
        update = .03 * (first / (1 - .9 ** (step + 1))) / (np.sqrt(second / (1 - .999 ** (step + 1))) + 1e-8)
        theta_next = theta - update[2048:]
        theta_next -= theta_next.mean(dtype='<f8')
        projected = np.r_[projection((a.ravel() - update[:2048]).reshape(16, 128)).ravel(), theta_next]
        close(h['parameter'][step, 1280:], projected, 'root_projected_updates')
        assert np.min(h['parameter'][step, 1280:3328]) >= 0
        close(h['parameter'][step, 1280:3328].reshape(16, 128).sum(axis=1), np.ones(16), 'root_simplex')
        close(h['parameter'][step, 3328:].mean(), 0., 'theta_centering')
        optimizer_updates += 1
        optimizer_scalar_fields += 4 * 2064 + 1
        guard()
    last = h['parameter'][-1]
    final_loss, _, _, log_b = root_loss_gradient(scores, last[1280:3328].reshape(16, 128), last[3328:], expected_A, weight)
    close(h['loss_final'][62], final_loss, 'root_final_losses', 1e-10, 1e-9)
    close(h['log_b'], log_b, 'log_b_export')
    close(np.exp(log_b).sum(), 128., 'root_mass_sum')
    close(record['support_initial_losses'], h['loss_before'][0, :62], 'RAW_initial_losses')
    close(record['support_final_losses'], h['loss_final'][:62], 'RAW_final_losses')
    close(record['partition_initial_loss'], h['loss_before'][0, 62], 'RAW_root_initial_loss')
    close(record['partition_final_loss'], h['loss_final'][62], 'RAW_root_final_loss')
    source_weight = prior['weights'][bank]
    with Path(source_weight['payload']).open('rb') as stream:
        stream.seek(source_weight['offset'])
        weight_bytes = stream.read(source_weight['bytes'])
    assert hashlib.sha256(weight_bytes).hexdigest() == source_weight['sha256']
    original = np.frombuffer(weight_bytes, '<f4').reshape(128, 768)
    vectors = np.empty((268, 768), '<f4')
    vectors[:128] = original
    for group in [*groups, {'ids': list(range(128)), 'offset': 1280, 'size': 2048, 'vector_offset': 252}]:
        ids, at, size, start = group['ids'], group['offset'], group['size'], group['vector_offset']
        mixing = last[at:at + size].reshape(-1, len(ids))
        accum = np.zeros((len(mixing), 768), '<f8')
        for j, expert in enumerate(ids):
            accum += mixing[:, j, None] * original[expert].astype('<f8')
        vectors[start:start + len(mixing)] = accum.astype('<f4')
    with Path(record['model_path']).open('rb') as stream:
        assert stream.read(32) == struct.pack('<8s6I', b'M480MOD1', 128, bank, 768, 255, 268, 16)
        assert stream.read(vectors.nbytes) == vectors.tobytes()
        assert stream.read(128) == h['log_b'].tobytes()
        mapping = {x['group']: x for x in groups}
        for node in tree:
            count = node['count']
            k, at = (0, 2**32 - 1) if node['node'] == 0 else (count, 2**32 - 1) if count <= 2 else (2, mapping[node['node']]['vector_offset'])
            assert stream.read(32) == struct.pack('<8I', *(node[x] for x in ('left', 'right', 'parent', 'count', 'first')), k, at, 0)
            assert stream.read(count * 4) == struct.pack('<' + str(count) + 'I', *node['ids'])
        assert not stream.read(1)
    model_scalar_fields += vectors.size + 16
    model_vectors.append(vectors)
    all_groups.append(groups)
    all_parameters.append(last.copy())
    del scores, targets, h, training, expected_A
    checkpoint(bank_optimizer_and_export_admitted=bank, optimizer_updates=optimizer_updates)
    guard()
assert optimizer_updates == 24192
gates['ALL159414_development_book_weights_24192_saved_updates_losses_and_physical_model_BYTES'] = True

phase = 'independent_ALL10032624_ordered_physical_dots_paths_roots_probabilities_counters'
with (SOURCE_OUT / 'models.ledger.bin').open('rb') as stream:
    assert stream.read(16) == struct.pack('<8sII', b'M480LED1', 12, 0)
    for bank, record in enumerate(r['models']):
        data = record['model_path'].encode()
        assert stream.read(8) == struct.pack('<II', bank, len(data)) and stream.read(len(data)) == data
    assert not stream.read(1)
diagnostic_views, divergence_views = [], []
for bank in range(12):
    uid = np.flatnonzero(unique['meta'][:, 2] == bank)
    vectors = model_vectors[bank]
    groups, parameter = all_groups[bank], all_parameters[bank]
    tree = memberships[bank]['nodes']
    offsets = {g['group']: g['vector_offset'] for g in groups}
    with np.load(r['models'][bank]['history_path'], allow_pickle=False) as archive:
        log_b = archive['log_b']
    first_divergence = np.zeros(7, '<u4')
    for begin in range(0, len(uid), 2048):
        ids = uid[begin:begin + 2048]
        p = pred[ids]
        row = unique[ids]
        v = p['vector_ids'].astype('<u4')
        assert np.all(v < 268)
        x = row['input'].astype('<f8')
        # Every physical field, using eight distinct ordered F64 lane sums.
        left = np.zeros((len(ids), 42, 4), '<f8')
        right = np.zeros_like(left)
        for at in range(0, 768, 8):
            first_dimensions = np.arange(at, at + 4)
            second_dimensions = np.arange(at + 4, at + 8)
            left += vectors[v[:, :, None], first_dimensions].astype('<f8') * x[:, None, first_dimensions]
            right += vectors[v[:, :, None], second_dimensions].astype('<f8') * x[:, None, second_dimensions]
        total = np.zeros((len(ids), 42), '<f8')
        lanes = left + right
        for lane in range(4):
            total += lanes[:, :, lane]
        actual = total.astype('<f4')
        assert actual.tobytes() == p['score'].tobytes(), ('physical_dot_BYTE', bank, begin)
        physical_dots += actual.size
        source_mask = v < 128
        expected_source = row['score'][np.arange(len(ids))[:, None], np.minimum(v, 127)]
        assert actual[source_mask].tobytes() == expected_source[source_mask].tobytes()
        expected_meta = np.zeros((len(ids), 12), '<u4')
        expected_meta[:, 0] = ids
        expected_meta[:, 1] = bank
        expected_meta[:, 3] = row['meta'][:, 7]
        expected_meta[:, 4:11] = np.array([42, 7, 16, 32256, 129152, 672, 24])
        nodes = np.zeros(len(ids), '<u4')
        alive = np.ones(len(ids), bool)
        slot = 0
        for depth in range(7):
            next_nodes = np.empty_like(nodes)
            used = 1 if depth == 6 else 2
            for node_index in np.unique(nodes):
                local = np.flatnonzero(nodes == node_index)
                node = tree[int(node_index)]
                children = [tree[node['left']], tree[node['right']]]
                best, keys = [], []
                for side, child in enumerate(children):
                    originals = child['ids']
                    vector_ids = originals if child['count'] <= 2 else [offsets[child['node']], offsets[child['node']] + 1]
                    assert len(vector_ids) == used
                    at = slot + side * used
                    assert np.all(v[local, at:at + used] == np.array(vector_ids))
                    values = actual[local, at:at + used]
                    winner = np.argmax(values, axis=1)
                    best.append(values[np.arange(len(local)), winner])
                    keys.append(np.array(originals)[winner] if child['count'] <= 2 else np.full(len(local), child['first']))
                right_choice = (best[1] > best[0]) | ((best[1] == best[0]) & (keys[1] < keys[0]))
                expected_meta[local, 11] += (best[1] == best[0])
                assert np.array_equal(p['direction'][local, depth], right_choice.astype('<u4'))
                in_node = np.isin(row['meta'][local, 7], node['ids'])
                assert np.array_equal(alive[local], in_node)
                source_right = np.isin(row['meta'][local, 7], children[1]['ids'])
                diverged = alive[local] & (source_right != right_choice)
                first_divergence[depth] += int(np.count_nonzero(diverged))
                alive[local[diverged]] = False
                next_nodes[local] = np.where(right_choice, node['right'], node['left'])
            nodes = next_nodes
            slot += 2 * used
        assert slot == 26 and np.all(v[:, 26:] == np.arange(252, 268))
        chosen = np.array([tree[int(node)]['first'] for node in nodes], '<u4')
        expected_meta[:, 2] = chosen
        assert expected_meta.tobytes() == p['meta'].tobytes()
        chosen_score = row['score'][np.arange(len(ids)), chosen]
        assert chosen_score.tobytes() == p['value'][:, 0].tobytes()
        assert row['value'][:, 1].astype('<f4').tobytes() == p['value'][:, 2].tobytes()
        root_terms = actual[:, 26:].astype('<f8') + log_b
        top = np.max(root_terms, axis=1)
        shifted_mass = np.zeros(len(ids), '<f8')
        for j in range(16):
            shifted_mass += np.exp(root_terms[:, j] - top)
        A = top + np.log(shifted_mass)
        probability = np.exp(chosen_score.astype('<f8') - A)
        assert top.tobytes() == p['root'][:, 0].tobytes()
        close(p['root'][:, 1], shifted_mass, 'native_shifted_mass', 2e-14, 4e-14)
        close(p['root'][:, 2], A, 'native_log_partition', 2e-13, 4e-14)
        close(p['root'][:, 3], probability, 'native_probability_F64', 1e-12, 1e-12)
        assert probability.astype('<f4').tobytes() == p['value'][:, 1].tobytes(), ('native_probability_F32_BYTE', bank, begin)
        path_fields += len(ids) * (12 + 7 + 42)
        audited_unique += len(ids)
        guard()
    assert int(first_divergence.sum()) == int(np.count_nonzero(pred['meta'][uid, 2] != unique['meta'][uid, 7]))
    divergence_views.append({'bank': bank, 'unique_queries': len(uid), 'first_original_winner_divergence_by_depth': first_divergence.tolist()})
    scores = unique['score'][uid].astype('<f8')
    visited = pred['vector_ids'][uid]
    observed = pred['score'][uid]
    for vector in range(128, 268):
        if vector < 252:
            group = groups[(vector - 128) // 2]
            ids = group['ids']
            a = parameter[group['offset']:group['offset'] + group['size']].reshape(2, len(ids))[(vector - 128) % 2]
        else:
            ids, a = list(range(128)), parameter[1280:3328].reshape(16, 128)[vector - 252]
        where = visited == vector
        assert np.all(where.sum(axis=1) <= 1)
        local = np.flatnonzero(np.any(where, axis=1))
        if len(local):
            position = np.argmax(where[local], axis=1)
            cached = np.einsum('ue,e->u', scores[local][:, ids], a, optimize=False)
            error = np.abs(cached - observed[local, position].astype('<f8'))
            maximum, mean = float(error.max()), float(error.mean(dtype='<f8'))
        else:
            maximum = mean = None
        diagnostic_views.append({'bank': bank, 'vector': vector, 'visited_queries': len(local), 'cached_teacher_combo_minus_physical_dot_absolute_max': maximum, 'cached_teacher_combo_minus_physical_dot_absolute_mean': mean})
    checkpoint(bank_physical_prediction_admitted=bank, physical_dots=physical_dots, unique_inputs=audited_unique)
assert physical_dots == 10032624 and audited_unique == U
assert len(diagnostic_views) == 1680 and len(divergence_views) == 12
terminal = json.loads((SOURCE_OUT / 'predict.stdout.log').read_text().splitlines()[-1])
assert terminal == r['native_terminal']
assert terminal['terminal'] and terminal['unique_queries'] == U and terminal['visited_dot_fields'] == physical_dots
assert terminal['coefficient_values'] == U * 32256 and terminal['source_visited_scores_BYTE']
assert terminal['CPU_affinity_mask'] == 1 and not terminal['MXCSR'] & 0x8040
assert r['inference_arithmetic_per_unique'] == {'ordered_dot_forms': 42, 'coefficient_values': 32256, 'root_exp_calls_meta6': 16, 'probability_exp_calls': 1, 'total_exp_calls': 17, 'root_log_calls': 1, 'query_dot_read_bytes': 129024, 'weights_including_log_b_bytes': 129152, 'node_bytes': 672, 'member_ID_bytes': 24, 'complete_charged_weights_metadata_bytes': 129848}
gates['ALL10032624_physical_dot_BYTES_238872_paths_mass_p_BYTES_complete_paid_work'] = True

phase = 'independent_complete_reports_diagnostics_and_scientific_decision'
integer_fields = report_float_fields = 0

def compare(expected, actual, path='report'):
    global integer_fields, report_float_fields
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), path
        for key in expected:
            compare(expected[key], actual[key], path + '/' + str(key))
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), path
        for i, (a, z) in enumerate(zip(expected, actual)):
            compare(a, z, path + '/' + str(i))
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)) and abs(expected - actual) <= 1e-11 + 1e-10 * abs(expected), (path, expected, actual)
        report_float_fields += 1
    else:
        assert expected == actual, (path, expected, actual)
        integer_fields += isinstance(expected, (int, bool))

different = pred['meta'][:, 2] != unique['meta'][:, 7]
probability = pred['value'][:, 1].astype('<f8')
relative = np.abs(probability - unique['value'][:, 1]) / unique['value'][:, 1]
invalid = ~np.isfinite(probability) | (probability < 0) | (probability > 1)
log_error = np.abs(pred['root'][:, 2] - (unique['value'][:, 0] + np.log(unique['value'][:, 2])))
coefficient = pred['meta'][:, 7].astype('<f8') / (128 * 768)
charged = (pred['meta'][:, 8].astype('<f8') + pred['meta'][:, 9] + pred['meta'][:, 10]) / (128 * 768 * 4)
fields = ('p_relative_mean', 'p_relative_max', 'p_relative_p95', 'same_ID_p_relative_max', 'log_partition_absolute_mean', 'log_partition_absolute_max', 'coefficient_ratio_mean', 'coefficient_ratio_max', 'coefficient_ratio_p95', 'charged_byte_ratio_mean', 'charged_byte_ratio_max', 'charged_byte_ratio_p95')

def summary(ids):
    count = len(ids)
    result = {'count': count, 'ID_differences': int(np.count_nonzero(different[ids])), 'invalid_probability_count': int(np.count_nonzero(invalid[ids])), 'p_relative_over_1percent': int(np.count_nonzero(relative[ids] > .01)), 'same_ID_count': int(np.count_nonzero(~different[ids]))}
    if not count:
        return {**result, **{key: None for key in fields}}
    errors = relative[ids]
    same = ids[~different[ids]]
    index = (95 * count + 99) // 100 - 1
    result.update(p_relative_mean=float(np.mean(errors, dtype='<f8')), p_relative_max=float(np.max(errors)), p_relative_p95=float(np.sort(errors)[index]), same_ID_p_relative_max=float(relative[same].max()) if len(same) else None,
                  log_partition_absolute_mean=float(np.mean(log_error[ids], dtype='<f8')), log_partition_absolute_max=float(np.max(log_error[ids])), coefficient_ratio_mean=float(np.mean(coefficient[ids], dtype='<f8')), coefficient_ratio_max=float(np.max(coefficient[ids])), coefficient_ratio_p95=float(np.sort(coefficient[ids])[index]), charged_byte_ratio_mean=float(np.mean(charged[ids], dtype='<f8')), charged_byte_ratio_max=float(np.max(charged[ids])), charged_byte_ratio_p95=float(np.sort(charged[ids])[index]))
    return result

report = {'unique_views': [], 'occurrence_views': [], 'book_views': [], 'source_ID_views': []}
for bank in [-1, *range(12)]:
    mask = np.ones(U, bool) if bank < 0 else unique['meta'][:, 2] == bank
    for role, eligible in (('ALL', np.ones(U, bool)), ('development', dev), ('consumed_validation', val)):
        report['unique_views'].append({'bank': bank, 'role': role, **summary(np.flatnonzero(mask & eligible))})
for bank in range(12):
    for role in range(3):
        for mode in range(2):
            rows = links[(links[:, 6] == bank) & (links[:, 5] == role) & (links[:, 4] == mode)]
            report['occurrence_views'].append({'bank': bank, 'role': role, 'mode': mode, 'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)), **summary(rows[:, 12])})
            for expert in range(128):
                report['source_ID_views'].append({'bank': bank, 'role': role, 'mode': mode, 'source_ID': expert, **summary(rows[rows[:, 9] == expert, 12])})
for book in range(192):
    role = 0 if book < 64 else 1 if book < 128 else 2
    for bank in range(12):
        for mode in range(2):
            rows = links[(links[:, 2] == book) & (links[:, 6] == bank) & (links[:, 4] == mode)]
            report['book_views'].append({'book': book, 'bank': bank, 'role': role, 'mode': mode, 'accepted': int(rows[:, 10].sum(dtype='<u8')), 'rejected': int(np.count_nonzero(rows[:, 10] == 0)), **summary(rows[:, 12])})
    if book % 32 == 31:
        guard()
assert [len(report[k]) for k in ('unique_views', 'occurrence_views', 'book_views', 'source_ID_views')] == [39, 72, 4608, 9216]
assert all(sum(x['count'] for x in report[key]) == 387036 for key in ('occurrence_views', 'book_views', 'source_ID_views'))
compare(report, json.loads(Path(r['reports_path']).read_bytes()))
compare(report['unique_views'][0], r['unique_overall'])
compare({'surrogate_to_physical_visited_dot_views': diagnostic_views, 'first_source_path_divergence': divergence_views}, json.loads(Path(r['physical_diagnostics_path']).read_bytes()), 'diagnostics')
scientific_gates = {'ALL_original_source_ID_tie_conserved': not bool(np.any(different)), 'EVERY_native_selected_probability_relative_within_1percent': bool(np.all(relative <= .01)), 'EVERY_native_probability_in_0_1': not bool(np.any(invalid)), 'EVERY_coefficient_work_within_80percent_flat': bool(np.all(coefficient <= .8)), 'EVERY_charged_weight_metadata_bytes_within_80percent_flat': bool(np.all(charged <= .8))}
assert scientific_gates == r['scientific_gates']
decision = 'LOCAL_SOURCE_TRANSFER_PASS' if all(scientific_gates.values()) else 'THIS_FIXED_LEARNED_SUPPORT2_PARTITION16_RECIPE_FAIL'
assert r['decision'] == decision + '_PENDING_INDEPENDENT_ADMISSION'
gates['ALL39_72_4608_9216_reports_1680_dotviews_12_divergences_and_separate_scientific_decision'] = True
resource = r['resource']
assert resource['hard_seconds'] == 1200 and resource['seconds_before_RAW'] <= 1200
assert resource['hard_peak_bytes'] == 4 << 30 and resource['OS_peak_combined_bytes'] <= 4 << 30
assert resource['hard_new_bytes'] == 512 << 20
assert sum(x['bytes'] for x in r['output_inventory']) + RAW.stat().st_size + (SOURCE_OUT / 'progress.jsonl').stat().st_size <= 512 << 20
progress_rows = [json.loads(x) for x in (SOURCE_OUT / 'progress.jsonl').read_text().splitlines()]
assert progress_rows[-1]['terminal'] and progress_rows[-1]['raw_sha256'] == RAW_SHA
assert progress_rows[-1]['seconds'] <= 1200
assert all(x['pid'] == r['process_instance']['pid'] for x in progress_rows)
assert [x['bank_fitted_and_exported'] for x in progress_rows if 'bank_fitted_and_exported' in x] == list(range(12))
predict_terminal = next(i for i, x in enumerate(progress_rows) if x.get('command_terminal') == 'predict')
assert predict_terminal > max(i for i, x in enumerate(progress_rows) if 'bank_fitted_and_exported' in x)
for rel, item in prior['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256'] and (ROOT / rel).stat().st_size == item['bytes']
last_status = status()
assert parent.cpu_affinity() == [0]
assert not ({'torch', 'scipy', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'meth480_learned_router', 'meth480_r1_learned_router', 'meth480_transfer_math', 'meth480_reporting'} & set(sys.modules))
gates['resource_caps_preservation_no_native_replay_fit_checkpoint_selection_or_imported_fit_report_module'] = True
guard()
retention = {'experiment': 'METH480-R1 independent complete learned transfer retention', 'raw_sha256': RAW_SHA, 'source_binding_sha256': BIND_SHA, 'first_fault_inventory_sha256': FIRST_INVENTORY_SHA, 'audit_source_sha256': digest(__file__), 'audit_protocol_sha256': digest(AUDIT_PROTO), 'audit_repair_protocol_sha256': digest(AUDIT_REPAIR_PROTO), 'audit_fault_inventory_sha256': AUDIT_FAULT_INVENTORY_SHA, 'windows_terminal_sha256': EVENT_SHA, 'process_instance': instance,
             'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(), 'commands': [], 'gates': gates, 'scientific_gates': scientific_gates, 'decision': decision + '_INDEPENDENTLY_ADMITTED', 'unique_inputs_audited': audited_unique, 'queries_audited': len(links), 'physical_dot_fields_BYTE_checked': physical_dots, 'saved_optimizer_updates_audited': optimizer_updates, 'optimizer_scalar_fields_audited': optimizer_scalar_fields, 'model_scalar_fields_BYTE_checked': model_scalar_fields, 'path_integer_fields_BYTE_checked': path_fields, 'report_integer_fields_rederived': integer_fields, 'report_float_fields_rederived': report_float_fields, 'maximum_absolute_audit_differences': differences, 'unique_overall': report['unique_views'][0], 'first_source_path_divergence': divergence_views,
             'numerical_tolerances': {'gradient_loss_absolute': 1e-10, 'gradient_loss_relative': 1e-9, 'update_moment_absolute': 1e-12, 'update_moment_relative': 1e-11, 'report_absolute': 1e-11, 'report_relative': 1e-10, 'physical_model_dot_p_F32': 'BYTE'}, 'native_replays': 0, 'training_updates': 0, 'preserved_daemons': daemons, 'tracked_status': last_status,
             'scope': 'Complete consumed-domain original128 router transfer; original480 remains admission-failed. All saved update identities and new physical fields admitted; no wholeartifact/fresh ownstate quality/rate/useful-n/LUT/DRAM/general-family claim.', 'resource': {'seconds_before_RET': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed, 'hard_seconds': 1200, 'hard_peak_bytes': 4 << 30, 'hard_new_bytes': 32 << 20}}
write(RET, retention)
guard()
checkpoint(terminal=True, retention_sha256=digest(RET), OS_peak_bytes=peak)
watchdog.cancel()
print(json.dumps({'retention': str(RET), 'sha256': digest(RET), 'gates': gates, 'scientific_gates': scientific_gates, 'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()

