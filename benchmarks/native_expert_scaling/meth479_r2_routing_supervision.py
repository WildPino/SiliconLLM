"""METH479-R2 sequential complete target recovery, no original native replay."""
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
OUT = ROOT / 'results/native_expert_scaling/meth479_r2_routing_supervision'
RAW = DOC / 'meth479_r2_routing_supervision_result.json'
FAIL = DOC / 'meth479_routing_supervision_result.failure.json'
INVENTORY = DOC / 'meth479_first_scientific_failure_inventory.json'
FAIL_SHA = '3f4233b0d7189274b345a5be459624c40c358fc86adba682f8b52a287cd89db3'
INVENTORY_SHA = '33ecee9303981d42597a63c9fcd90546b9b9b3f7054e014d231be54ad7bf8b96'
BIND = DOC / 'meth479_prospective_bindings.json'
BIND_SHA = '0523b1822b84f6e11b40025b3c677116042a06853f4094375d271df73a4016ac'
C = ROOT / 'benchmarks/native_expert_scaling/meth479_source_router.c'
MATH = ROOT / 'benchmarks/native_expert_scaling/meth479_supervision_math.py'
WINDOWS = ROOT / 'benchmarks/native_expert_scaling/meth479_r2_windows_terminal.ps1'
PROTO = DOC / 'METH_479_R2_ROUTING_SUPERVISION_PROTOCOL_20261005.md'
parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
assert args.out.resolve() == RAW.resolve() and not OUT.exists() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists()
OUT.mkdir()
fatal = (OUT / 'fatal_native.log').open('xb')
faulthandler.enable(file=fatal, all_threads=True)
progress = (OUT / 'progress.jsonl').open('x', encoding='utf8')
parent = psutil.Process()
parent.cpu_affinity([0])
assert parent.cpu_affinity() == [0]
instance = {'pid': parent.pid, 'create_time_unix': parent.create_time(), 'executable': sys.executable}
start_utc = datetime.fromtimestamp(parent.create_time(), timezone.utc).isoformat()
phase = 'immutable_admission'
commands = []
peak = hashed = 0
cache = {}

def checkpoint(**values):
    progress.write(json.dumps({'phase': phase, 'seconds': time.monotonic() - START, 'pid': parent.pid, **values}) + '\n')
    progress.flush()

def guard(child=None):
    global peak
    info = parent.memory_info()
    value = max(info.rss, getattr(info, 'peak_wset', 0))
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
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) + (RAW.stat().st_size if RAW.exists() else 0) <= 2 << 30

def digest(path):
    global hashed
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with path.open('rb') as stream:
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
    path = Path(path).resolve()
    rel = path.relative_to(ROOT).as_posix()
    assert path.read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel]), rel

def write(path, record):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(record, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n').encode())

def failure(kind, value, tb):
    destination = RAW.with_suffix('.failure.json')
    if not destination.exists():
        write(destination, {'experiment': 'METH479-R2 FIRST sequential target recovery fault', 'phase': phase,
                            'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(),
                            'process_instance': instance, 'commands': commands, 'binding_sha256': BIND_SHA,
                            'scientific_sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), C, MATH, WINDOWS, PROTO)},
                            'traceback': ''.join(traceback.format_exception(kind, value, tb)),
                            'resource': {'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak, 'file_bytes_hashed': hashed},
                            'partial_output_inventory': [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()]})
    checkpoint(terminal_failure=True)
    sys.__excepthook__(kind, value, tb)

sys.excepthook = failure
import threading
def deadline():
    destination = RAW.with_suffix('.failure.json')
    if not destination.exists():
        write(destination, {'experiment': 'METH479-R2 watchdog FIRST fault', 'phase': phase,
                            'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(),
                            'process_instance': instance, 'commands': [], 'binding_sha256': BIND_SHA,
                            'original_failure_sha256': FAIL_SHA, 'reason': '600 second deadline',
                            'resource': {'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak}})
    os._exit(3)
watchdog = threading.Timer(max(.001, 600 - (time.monotonic() - START)), deadline)
watchdog.daemon = True
watchdog.start()
checkpoint()
assert digest(BIND) == BIND_SHA
head(BIND)
b = json.loads(BIND.read_bytes())
for p in (Path(__file__), C, MATH, WINDOWS, PROTO):
    head(p)
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
assert sys.version == b['runtime']['python'] and Path(sys.executable).resolve() == Path(b['runtime']['executable']).resolve()
assert psutil.__version__ == b['runtime']['packages']['psutil']['version'] and str(Path(psutil.__file__).resolve()) in b['runtime']['packages']['psutil']['files']
for path, item in b['runtime']['files'].items():
    check({'path': path, **item})
for package, item in b['runtime']['packages'].items():
    assert importlib.metadata.version(package) == item['version']
    for path, value in item['files'].items():
        check({'path': path, **value})
for key in ('records', 'helpers', 'scientific_files', 'source_inventory', 'compile_assets', 'system_files', 'membership_sources'):
    for item in b[key]:
        check(item)
for key in ('compiler', 'preparation_helper'):
    check(b[key])
for item in b['records'] + b['helpers'] + b['scientific_files']:
    head(item['path'])
artifact = b['artifact']
assert Path(artifact['payload']).stat().st_size == artifact['bytes'] and digest(artifact['payload']) == artifact['sha256']
assert digest(artifact['manifest']) == artifact['manifest_sha256']
for rel, item in b['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
def preserved_status():
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines()
    assert all(v[:3] == ' M ' and v[3:] in b['preserved_unrelated_files'] for v in status), status
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    return status
preserved_status()
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
assert len(b['source_inventory']) == 6224 and b['query_count'] == 387036 and b['unique_upper_bound'] == 248796
assert b['binary_upper_bound'] + (8 << 20) <= 2 << 30
assert digest(FAIL) == FAIL_SHA and digest(INVENTORY) == INVENTORY_SHA
head(FAIL)
head(INVENTORY)
for item in json.loads(INVENTORY.read_bytes())['records']:
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

assert time.monotonic() - START <= 120
gates = {'complete_469_471_source_artifact_memberships_roles_runtime_actual_assets_frozen': True}
checkpoint(file_bytes_hashed=hashed)


phase = 'retained_original_native_admission_and_canonical_wires'
f = json.loads(FAIL.read_bytes())
assert f['phase'] == 'all_group_support_and_native_mass_targets' and f['resource']['seconds'] > 600
assert f['resource']['OS_peak_combined_bytes'] <= 4 << 30
assert [v['returncode'] for v in f['commands']] == [0, 0, 2, 0]
for path, value in f['scientific_sources'].items():
    head(path)
    assert digest(path) == value
original_event = json.loads((ROOT / 'results/native_expert_scaling/meth479_windows_terminal.json').read_bytes())
assert original_event['query_available'] and not original_event['matching_scientific_events']
original_progress = [json.loads(v) for v in (SOURCE_OUT / 'progress.jsonl').read_text().splitlines()]
assert original_progress[-1]['terminal_failure'] and not (DOC / 'meth479_routing_supervision_result.json').exists()
assert any(v.get('unique_inputs') == 238872 and v['phase'] == 'exact_input_identities_source_joins_and_all_occurrence_links' for v in original_progress)
native_lines = [json.loads(v) for v in (SOURCE_OUT / 'source.stdout.log').read_text().splitlines()]
terminal = native_lines[-1]
assert terminal['terminal'] and terminal['source_queries'] == 387036 and terminal['source_score_values'] == 49540608
assert terminal['scores_BYTE_exact'] and terminal['ID_probability_BYTE_exact'] and terminal['CPU_affinity_mask'] == 1 and not terminal['MXCSR'] & 0x8040
assert (SOURCE_OUT / 'fatal_native.log').stat().st_size == 0
assert all((SOURCE_OUT / (v + '.stderr.log')).stat().st_size == 0 for v in ('compile', 'controls', 'source'))
assert (SOURCE_OUT / 'negative_weight_ledger.stderr.log').read_bytes() == b'source_router_error:weight_ledger_magic\r\n'
gates['completed_original_native_full_score_ID_probability_admission_retained_NOT_original_main_PASS'] = True
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
for module in (np, threadpoolctl):
    assert module.__version__ == b['runtime']['packages'][module.__name__]['version'] and str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
import meth479_supervision_math as M
assert Path(M.__file__).resolve() == MATH.resolve()
U = 238872

def array(name, magic, width, reserved, count, dtype, shape):
    path = SOURCE_OUT / name
    assert path.stat().st_size == 24 + width * count
    with path.open('rb') as stream:
        assert stream.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
        value = np.fromfile(stream, dtype=dtype, count=count if len(shape) == 1 else count * shape[1]).reshape(shape)
        assert value.shape == shape and not stream.read(1)
    guard()
    return value
unique = array('unique_inputs.bin', b'M479UNI1', 3720, 768, U, M.UNIQUE, (U,))
owner = array('ownership.bin', b'M479OWN1', 32, 0, U, M.OWNER, (U,))
links = array('query_links.bin', b'M479LNK1', 52, 0, 387036, '<u4', (387036, 13))
source = array('source_fields.bin', b'M479SRC1', 72, 0, 387036, M.SOURCE, (387036,))
assert np.array_equal(source['meta'], links[:, :12]) and np.all(links[:, 12] < U)
assert np.array_equal(unique['meta'][:, 0], np.arange(U)) and np.array_equal(owner['meta'][:, 0], np.arange(U))
assert np.array_equal(owner['meta'][:, 1], unique['meta'][:, 2]) and np.array_equal(source['meta'][:, 0], np.arange(387036))
assert np.isfinite(unique['input']).all() and np.isfinite(unique['score']).all() and np.isfinite(unique['value']).all()
assert np.array_equal(np.argmax(unique['score'], axis=1), unique['meta'][:, 7])
assert np.array_equal(np.bincount(links[:, 12], minlength=U), owner['meta'][:, 5])
assert M.owner_summary(owner)['occurrences'] == 387036
# Recover development book sets from ALL occurrences, rather than first-owner metadata.
development_book_bits = np.zeros((U, 2), '<u8')
for book in list(range(64)) + list(range(128, 192)):
    ids = links[links[:, 2] == book, 12]
    column, bit = (0, book) if book < 64 else (1, book - 128)
    np.bitwise_or.at(development_book_bits[:, column], ids, np.uint64(1 << bit))
assert np.array_equal(np.array([int(v[0]).bit_count() + int(v[1]).bit_count() for v in development_book_bits]), owner['meta'][:, 6])
gates['full_retained_canonical_wires_all_occurrence_book_ownership_and_shape_joins'] = True
checkpoint(unique_inputs=U)
phase = 'sequential_UID_all_native_group_targets'
memberships = []
for weight in b['weights']:
    with Path(weight['payload']).open('rb') as stream:
        stream.seek(weight['offset'])
        original = stream.read(weight['bytes'])
    assert hashlib.sha256(original).hexdigest() == weight['sha256']
    nodes = M.membership(b['membership_sources'][weight['bank']]['path'], weight['bank'], original)
    memberships.append({'bank': weight['bank'], 'nodes': nodes})
assert memberships == json.loads((SOURCE_OUT / 'membership.json').read_bytes())
target_path = OUT / 'group_targets.bin'
target_diagnostics = {'maximum_F64_path_probability_absolute_error': 0., 'F32_path_probability_differs_from_direct_native': 0, 'zero_child_mass_fields': 0}
with target_path.open('xb', buffering=4 << 20) as stream:
    stream.write(struct.pack('<8sIIQ', b'M479TGT1', 3560, 127, U))
    for at in range(0, U, 8192):
        end = min(U, at + 8192)
        block = np.zeros(end - at, M.TARGET)
        for bank in range(12):
            local = np.flatnonzero(unique['meta'][at:end, 2] == bank)
            if not len(local):
                continue
            uid = local + at
            row = unique[uid]
            target, values = M.targets(row['score'], row['value'], memberships[bank]['nodes'], uid)
            block[local] = target
            for key, value in values.items():
                target_diagnostics[key] = max(target_diagnostics[key], value) if key.startswith('maximum') else target_diagnostics[key] + value
            guard()
        assert np.array_equal(block['unique_id'], np.arange(at, end))
        stream.write(block.tobytes())
        checkpoint(unique_targets_written=end)
        guard()
assert target_path.stat().st_size == 24 + U * 3560
# Read-only RAM array; no scattered writable mappings and no partial-original targets.
with target_path.open('rb') as stream:
    stream.seek(24)
    saved_targets = np.fromfile(stream, M.TARGET, U)
assert len(saved_targets) == U and np.array_equal(saved_targets['unique_id'], np.arange(U))
assert np.isfinite(saved_targets['nodes']['mass']).all() and (saved_targets['nodes']['mass'] >= 0).all()
gates['ALL238872_unique_127_targets_sequential_BYTE_schema_native_root_path_tie_mass'] = True
phase = 'complete_ownership_and_development_exposure_reports'
summary = {'ownership': M.owner_summary(owner), 'banks': [], 'occurrences': M.occurrence_tables(links)}
for bank in range(12):
    ids = np.flatnonzero(unique['meta'][:, 2] == bank)
    summary['banks'].append({'bank': bank, 'ownership': M.owner_summary(owner[ids]),
                           'branch_development_exposure': M.exposure(bank, unique, owner, saved_targets, memberships[bank]['nodes'], development_book_bits)})
    checkpoint(exposure_bank=bank)
    guard()
assert len(summary['occurrences']) == 72 and sum(v['queries'] for v in summary['occurrences']) == 387036
assert sum(len(v['branch_development_exposure']) for v in summary['banks']) == 1524
gates['ALL72_occurrence_13ownership_1524unconditional_and_path_exposure_views'] = True
del saved_targets, unique, owner, links, source, development_book_bits
for rel, item in b['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
last_status = preserved_status()
assert not ({'torch', 'transformers', 'tensorflow', 'scipy', 'sklearn', 'pandas'} & set(sys.modules))
gates['600s_watchdog_4GiB_2GiB_no_native_replay_training_or_partial_target_reuse'] = True
guard()
report = {'experiment': 'METH479-R2 complete sequential target recovery', 'start_utc': start_utc,
          'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
          'source_binding_sha256': BIND_SHA, 'original_failure_sha256': FAIL_SHA, 'failure_inventory_sha256': INVENTORY_SHA,
          'r1_failure_sha256': digest(R1_FAILURE), 'r1_inventory_sha256': digest(R1_INVENTORY),
          'scientific_sources': {str(p): digest(p) for p in (Path(__file__), C, MATH, WINDOWS, PROTO)},
          'commands': [], 'original_native_commands': f['commands'], 'gates': gates, 'artifact': artifact,
          'source_native_terminal': terminal, 'queries': 387036, 'streams': 1536, 'unique_inputs': U,
          'unique_upper_bound': 248796, 'group_target_records': U * 127, 'prefix_pairs_BYTE_exact': 768,
          'target_diagnostics': target_diagnostics, 'summaries': summary, 'preserved_daemons': daemons,
          'native_model_commands': 0, 'native_numeric_commands': 0, 'training_updates': 0,
          'decision': 'supervision_complete_pending_independent_retention_NOT_a_trained_candidate',
          'head_at_execution': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'tracked_status_at_terminal': last_status,
          'scope': 'Original479 remains time-failed; R1 reuses completed native/canonical wires, independently audits original joins before dataset admission; no partial targets/native replay/fit/newartifact/quality/rate/useful-n/LUT/DRAM/generalization',
          'resource': {'seconds_before_RAW': time.monotonic() - START, 'OS_peak_combined_bytes': peak, 'file_bytes_hashed': hashed,
                       'hard_seconds': 600, 'hard_peak_bytes': 4 << 30, 'hard_new_bytes': 2 << 30,
                       'binary_output_upper_bound': 24 + U * 3560},
          'output_inventory': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']}
write(RAW, report)
guard()
checkpoint(terminal=True, raw_sha256=digest(RAW), OS_peak_combined_bytes=peak)
watchdog.cancel()
print(json.dumps({'raw': str(RAW), 'sha256': digest(RAW), 'gates': gates, 'unique_inputs': U,
                  'ownership': summary['ownership'], 'target_diagnostics': target_diagnostics,
                  'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()
