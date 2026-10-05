"""ONE fixed support2/root16 fit, physical export and all-source native inquiry."""
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
OUT = ROOT / 'results/native_expert_scaling/meth480_learned_router'
RAW = DOC / 'meth480_learned_router_result.json'
BIND = DOC / 'meth480_prospective_bindings.json'
BIND_SHA = '2c33137557c5a958680633187b536dde37df75e12ce3882e0cff04330259579e'
C = ROOT / 'benchmarks/native_expert_scaling/meth480_learned_router.c'
MATH = ROOT / 'benchmarks/native_expert_scaling/meth480_transfer_math.py'
REPORTING = ROOT / 'benchmarks/native_expert_scaling/meth480_reporting.py'
WINDOWS = ROOT / 'benchmarks/native_expert_scaling/meth480_windows_terminal.ps1'
PROTO = DOC / 'METH_480_LEARNED_ROUTER_PROTOCOL_20261005.md'
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
phase = 'immutable_source_supervision_runtime_admission'
commands = []
peak = hashed = 0
cache = {}
scientific = (Path(__file__).resolve(), C, MATH, REPORTING, WINDOWS, PROTO)

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
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) + (RAW.stat().st_size if RAW.exists() else 0) <= 512 << 20

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
    path = RAW.with_suffix('.failure.json')
    if not path.exists():
        write(path, {'experiment': 'METH480 FIRST learned transfer fault', 'phase': phase,
              'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
              'commands': commands, 'source_binding_sha256': BIND_SHA,
              'scientific_sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scientific},
              'traceback': ''.join(traceback.format_exception(kind, value, tb)),
              'resource': {'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak, 'file_bytes_hashed': hashed},
              'partial_output_inventory': [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()]})
    checkpoint(terminal_failure=True)
    sys.__excepthook__(kind, value, tb)

sys.excepthook = failure
def deadline():
    path = RAW.with_suffix('.failure.json')
    if not path.exists():
        write(path, {'experiment': 'METH480 watchdog FIRST fault', 'phase': phase, 'start_utc': start_utc,
                     'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
                     'commands': commands, 'source_binding_sha256': BIND_SHA, 'reason': '1200 second deadline',
                     'resource': {'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak}})
    os._exit(3)
watchdog = threading.Timer(max(.001, 1200 - (time.monotonic() - START)), deadline)
watchdog.daemon = True
watchdog.start()
checkpoint()
assert digest(BIND) == BIND_SHA
head(BIND)
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
ret = json.loads((DOC / 'RETENTION_479_R3_20261005.json').read_bytes())
assert all(ret['gates'].values()) and ret['unique_inputs_audited'] == 238872 and ret['queries_audited'] == 387036
assert time.monotonic() - START <= 180
gates = {'full_source6224_artifact_runtime_scientific_freezes_admitted479_supervision': True}
checkpoint(file_bytes_hashed=hashed)

phase = 'all_canonical_native_roots_links_ownership_and_memberships'
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
for module in (np, threadpoolctl):
    assert module.__version__ == runtime['packages'][module.__name__]['version'] and str(Path(module.__file__).resolve()) in runtime['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in runtime['packages']['numpy']['files']
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
import meth480_transfer_math as M
import meth480_reporting as R
assert Path(M.__file__).resolve() == MATH.resolve() and Path(R.__file__).resolve() == REPORTING.resolve()
U = 238872
def wire(name, magic, width, reserved, count, dtype, shape, mapped=False):
    p = Path(b['data_files'][name]['path'])
    assert p.stat().st_size == 24 + width * count
    with p.open('rb') as stream:
        assert stream.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
        if mapped:
            return np.memmap(p, dtype=dtype, mode='r', offset=24, shape=shape)
        value = np.fromfile(stream, dtype=dtype, count=count if len(shape) == 1 else count * shape[1]).reshape(shape)
        assert not stream.read(1)
    guard()
    return value
unique = wire('unique_inputs.bin', b'M479UNI1', 3720, 768, U, M.UNIQUE, (U,))
owner = wire('ownership.bin', b'M479OWN1', 32, 0, U, '<u4', (U, 8))
links = wire('query_links.bin', b'M479LNK1', 52, 0, 387036, '<u4', (387036, 13))
source_dtype = np.dtype([('meta', '<u4', (12,)), ('value', '<f8', (3,))])
source = wire('source_fields.bin', b'M479SRC1', 72, 0, 387036, source_dtype, (387036,))
target = wire('group_targets.bin', b'M479TGT1', 3560, 127, U, M.TARGET, (U,), mapped=True)
assert np.array_equal(unique['meta'][:, 0], np.arange(U)) and np.array_equal(owner[:, 0], np.arange(U))
assert np.array_equal(target['unique_id'], np.arange(U)) and np.array_equal(source['meta'], links[:, :12])
assert np.all(links[:, 12] < U) and np.array_equal(owner[:, 1], unique['meta'][:, 2])
assert np.array_equal(source['value'].copy().view('<u8'), unique['value'][links[:, 12]].copy().view('<u8'))
assert np.array_equal(np.bincount(links[:, 12], minlength=U), owner[:, 5])
assert np.array_equal(np.argmax(unique['score'], axis=1), unique['meta'][:, 7])
assert np.array_equal(unique['score'][np.arange(U), unique['meta'][:, 7]].astype('<f8').view('<u8'), unique['value'][:, 0].copy().view('<u8'))
assert np.isfinite(unique['input']).all() and np.isfinite(unique['score']).all() and np.isfinite(unique['value']).all()
assert np.all((unique['value'][:, 2] >= 1) & (unique['value'][:, 2] <= 128))
assert np.array_equal((1 / unique['value'][:, 2]).astype('<f4').view('<u4'), unique['value'][:, 1].astype('<f4').view('<u4'))
assert np.count_nonzero(owner[:, 2] & 5) == 159414 and np.count_nonzero(owner[:, 2] & 2) == 79458
assert not np.any(((owner[:, 2] & 5) != 0) & ((owner[:, 2] & 2) != 0))
memberships = json.loads(Path(b['data_files']['membership.json']['path']).read_bytes())
assert [v['bank'] for v in memberships] == list(range(12))
del source
gates['ALL_canonical_source_native_roots_links_roles_ownership_and_target_indices'] = True
checkpoint()

def run(argv, label, expected=0):
    before = datetime.now(timezone.utc).isoformat()
    maximum, children = 0, {}
    with (OUT / (label + '.stdout.log')).open('xb') as stdout, (OUT / (label + '.stderr.log')).open('xb') as stderr:
        child = subprocess.Popen([str(v) for v in argv], stdout=stdout, stderr=stderr)
        proc = psutil.Process(child.pid)
        pi = {'pid': child.pid, 'create_time_unix': proc.create_time(), 'executable': str(argv[0])}
        proc.cpu_affinity([0])
        assert proc.cpu_affinity() == [0]
        try:
            while child.poll() is None:
                guard(child)
                try:
                    mi = proc.memory_info()
                    maximum = max(maximum, mi.rss, getattr(mi, 'peak_wset', 0))
                    for p in proc.children(recursive=True):
                        try:
                            key = (p.pid, p.create_time())
                            mi = p.memory_info()
                            children[key] = {'pid': p.pid, 'create_time_unix': p.create_time(), 'executable': p.exe(),
                                'OS_peak_bytes': max(mi.rss, getattr(mi, 'peak_wset', 0), children.get(key, {}).get('OS_peak_bytes', 0))}
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.025)
        except BaseException:
            if child.poll() is None:
                child.kill()
                child.wait()
            raise
    commands.append({'label': label, 'argv': [str(v) for v in argv], 'returncode': child.returncode,
                     'expected_returncode': expected, 'process_instance': pi, 'start_utc': before,
                     'end_utc': datetime.now(timezone.utc).isoformat(), 'OS_peak_bytes': maximum,
                     'descendant_process_peaks': list(children.values())})
    checkpoint(command_terminal=label, returncode=child.returncode)
    assert child.returncode == expected, (label, child.returncode, (OUT / (label + '.stderr.log')).read_bytes())
    return [json.loads(v) for v in (OUT / (label + '.stdout.log')).read_text().splitlines()]

phase = 'new_optimizer_native_codec_FPU_and_negative_controls'
math_controls = M.controls()
write(OUT / 'math_controls.json', math_controls)
binary = OUT / 'meth480_learned_router.exe'
run([prior['compiler']['path'], '-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off', C, '-o', binary], 'compile')
native_controls = run([binary, 'controls'], 'controls')
assert len(native_controls) == 11
def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]
with localcontext() as context:
    context.prec = 100
    toys = ((0, 1, -1), (10000, 9999, -10000), (0, 0, 0), (1000, 999, -1000), (0, -1, -2), (0, 0, -1))
    for k, values in enumerate(toys):
        chosen = max(range(3), key=lambda j: values[j])
        total = sum(f32(float((Decimal(v) - Decimal(values[chosen])).exp())) for v in values)
        assert native_controls[k] == {'control': k, 'chosen': chosen, 'denominator': total, 'probability': f32(1 / total)}
    for k in range(2):
        values = toys[k]
        chosen = max(values)
        total = sum(Decimal(w) * (Decimal(v) - Decimal(chosen)).exp() for w, v in zip((2, 3, 5), values))
        A = float(Decimal(chosen) + total.ln())
        assert native_controls[9 + k]['weighted_partition_control'] == k
        assert abs(native_controls[9 + k]['A'] - A) <= 1e-10 + 1e-12 * abs(A)
        assert struct.pack('<f', native_controls[9 + k]['probability']) == struct.pack('<f', float(1 / total))
for kind in range(2):
    a = [(i % 7 - 3) * 2.**-10 for i in range(768)] if not kind else [2.**-100 if i % 2 else 2.**80 for i in range(768)]
    x = [(i % 5 - 2) * 2.**-9 for i in range(768)] if not kind else [2.**-20 if i % 2 else -2.**-80 if i % 4 else 2.**-80 for i in range(768)]
    l, r = [0.] * 4, [0.] * 4
    for at in range(0, 768, 8):
        for j in range(4):
            l[j] += a[at + j] * x[at + j]
            r[j] += a[at + j + 4] * x[at + j + 4]
    value = 0.
    for j in range(4):
        value += l[j] + r[j]
    assert native_controls[6 + kind] == {'dot_control': kind, 'double_result': value, 'f32_result': f32(value)}
assert native_controls[8]['rounding'] == 0 and native_controls[8]['CPU_affinity_mask'] == 1 and not native_controls[8]['MXCSR'] & 0x8040
bad = OUT / 'negative_ledger.bin'
bad.write_bytes(b'INVALID!' + struct.pack('<2I', 12, 0))
negative_output = OUT / 'negative_predictions.bin'
run([binary, 'predict', bad, b['data_files']['unique_inputs.bin']['path'], negative_output], 'negative_ledger', 2)
assert not negative_output.exists() and (OUT / 'negative_ledger.stderr.log').read_bytes() == b'learned_router_error:model_ledger_magic\r\n'
gates['simplex_gradients_Adam_Decimal100_ordered_dot_weighted_partition_FPU_negative_controls'] = True

phase = 'ALL12_development_only_fixed32_support_and_partition_fit'
models = []
for bank in range(12):
    tree = memberships[bank]['nodes']
    assert len(tree) == 255 and tree[0]['ids'] == list(range(128))
    uid = np.flatnonzero((unique['meta'][:, 2] == bank) & ((owner[:, 2] & 5) != 0))
    omega_global = np.zeros(U, '<f8')
    book_count = np.zeros(U, '<u4')
    book_rows = []
    for book in [*range(64), *range(128, 192)]:
        selected = np.unique(links[(links[:, 2] == book) & (links[:, 6] == bank), 12])
        assert len(selected) and np.all((owner[selected, 2] & 5) != 0)
        omega_global[selected] += 1 / (128 * len(selected))
        book_count[selected] += 1
        book_rows.append({'book': book, 'unique_inputs': len(selected)})
    assert np.array_equal(book_count, owner[:, 6] * (owner[:, 1] == bank))
    assert np.all(omega_global[owner[:, 2] & 2 != 0] == 0)
    omega = omega_global[uid]
    assert len(uid) == (14848 if bank < 6 else 11721) and abs(omega.sum() - 1) <= 1e-12
    train_dtype = np.dtype([('unique_id', '<u4'), ('omega', '<f8')])
    training = np.empty(len(uid), train_dtype)
    training['unique_id'], training['omega'] = uid, omega
    training_path = OUT / f'bank{bank:02}_training.bin'
    with training_path.open('xb') as stream:
        stream.write(struct.pack('<8sIIQ', b'M480TRN1', 12, bank, len(uid)))
        stream.write(training.tobytes())
    row = unique[uid]
    score = row['score'].astype('<f8')
    targets = target[uid]
    assert np.array_equal(targets['unique_id'], uid)
    history = M.fit(score, targets, omega, tree, guard, lambda **v: checkpoint(bank=bank, **v))
    root_target = row['value'][:, 0] + np.log(row['value'][:, 2])
    log_b = M.fit_mass(score, root_target, omega, history, guard)
    weight = prior['weights'][bank]
    with Path(weight['payload']).open('rb') as stream:
        stream.seek(weight['offset'])
        weight_bytes = stream.read(weight['bytes'])
    assert hashlib.sha256(weight_bytes).hexdigest() == weight['sha256']
    weights = np.frombuffer(weight_bytes, '<f4').reshape(128, 768)
    vectors = M.exported_vectors(weights, history)
    assert vectors[:128].tobytes() == weight_bytes and abs(np.exp(log_b).sum() - 128) <= 1e-10
    history_path = OUT / f'bank{bank:02}_history.npz'
    with history_path.open('xb') as stream:
        np.savez(stream, **{k: v for k, v in history.items() if isinstance(v, np.ndarray)}, log_b=log_b)
    model_path = OUT / f'bank{bank:02}_model.bin'
    mapping = {v['group']: v for v in history['groups']}
    with model_path.open('xb') as stream:
        stream.write(struct.pack('<8s6I', b'M480MOD1', 128, bank, 768, 255, 268, 16))
        stream.write(vectors.tobytes())
        stream.write(log_b.astype('<f8').tobytes())
        for node in tree:
            count = node['count']
            k, offset = (0, 2**32 - 1) if node['node'] == 0 else (count, 2**32 - 1) if count <= 2 else (2, mapping[node['node']]['vector_offset'])
            stream.write(struct.pack('<8I', *(node[v] for v in ('left', 'right', 'parent', 'count', 'first')), k, offset, 0))
            stream.write(struct.pack('<' + str(count) + 'I', *node['ids']))
    assert model_path.stat().st_size == 835712
    saved = np.load(history_path, allow_pickle=False)
    assert saved.files == [k for k, v in history.items() if isinstance(v, np.ndarray)] + ['log_b']
    assert all(np.array_equal(saved[k].view('u1'), history[k].view('u1')) for k in saved.files if k != 'log_b')
    saved.close()
    record = {'bank': bank, 'groups': history['groups'], 'books': book_rows, 'training_unique': len(uid),
              'training_weight_sum': float(omega.sum()), 'model_path': str(model_path),
              'history_path': str(history_path), 'training_path': str(training_path),
              'support_initial_losses': history['loss_before'][0, :62].tolist(),
              'support_final_losses': history['loss_final'][:62].tolist(),
              'partition_initial_loss': float(history['loss_before'][0, 62]), 'partition_final_loss': float(history['loss_final'][62])}
    write(OUT / f'bank{bank:02}_fit.json', record)
    models.append(record)
    del row, score, targets, history, weights, vectors, training, omega_global, book_count
    checkpoint(bank_fitted_and_exported=bank)
    guard()
assert len(models) == 12
gates['ALL12_full_development_only_62support_root_fixed32_complete_optimizer_histories'] = True
gates['ALL12_physical_models_original128_vectors_new140_forms_memberships_mass_export_and_EOF'] = True

phase = 'ALL238872_physical_C_predictions_no_flat_scan'
ledger = OUT / 'models.ledger.bin'
with ledger.open('xb') as stream:
    stream.write(struct.pack('<8sII', b'M480LED1', 12, 0))
    for record in models:
        data = record['model_path'].encode()
        stream.write(struct.pack('<II', record['bank'], len(data)))
        stream.write(data)
prediction_path = OUT / 'predictions.bin'
native = run([binary, 'predict', ledger, b['data_files']['unique_inputs.bin']['path'], prediction_path], 'predict')
terminal = native[-1]
assert terminal['terminal'] and terminal['unique_queries'] == U and terminal['visited_dot_fields'] == U * 42
assert terminal['coefficient_values'] == U * 32256 and terminal['source_visited_scores_BYTE'] and terminal['CPU_affinity_mask'] == 1 and not terminal['MXCSR'] & 0x8040
assert prediction_path.stat().st_size == 24 + U * 372
with prediction_path.open('rb') as stream:
    assert stream.read(24) == struct.pack('<8sIIQ', b'M480PRD1', 372, 0, U)
    pred = np.fromfile(stream, M.PRED, U)
assert np.array_equal(pred['meta'][:, 0], np.arange(U)) and np.array_equal(pred['meta'][:, 1], unique['meta'][:, 2])
assert np.array_equal(pred['meta'][:, 3], unique['meta'][:, 7]) and np.all(pred['meta'][:, 2] < 128)
assert np.array_equal(pred['value'][:, 2].view('<u4'), unique['value'][:, 1].astype('<f4').view('<u4'))
assert np.array_equal(pred['value'][:, 0].view('<u4'), unique['score'][np.arange(U), pred['meta'][:, 2]].view('<u4'))
assert np.all(pred['meta'][:, 4:11] == np.array([42, 7, 16, 32256, 129152, 672, 24]))
assert np.isfinite(pred['value']).all() and np.isfinite(pred['root']).all()
gates['ALL238872_native_physical_predictions_visited_original_scores_BYTE_paid_counters_and_schema'] = True

phase = 'ALL_source_and_rare_ID_reports_scientific_decision'
summaries, scientific_gates = R.reports(unique, owner, links, pred)
write(OUT / 'complete_reports.json', summaries)
diagnostics = R.physical_diagnostics(unique, pred, models, memberships)
write(OUT / 'physical_diagnostics.json', diagnostics)
gates['ALL39_unique_72occurrence_4608book_9216source_ID_views_complete_without_filter'] = True
del target, unique, owner, links, pred
for rel, item in prior['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
last_status = status()
assert not ({'torch', 'scipy', 'transformers', 'tensorflow', 'sklearn', 'pandas'} & set(sys.modules))
gates['resource_caps_no_native_model_capture_GPU_T4_refit_or_validation_selection'] = True
guard()
report = {'experiment': 'METH480 ONE learned convex support2/root16 native transfer', 'start_utc': start_utc,
          'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance,
          'source_binding_sha256': BIND_SHA, 'artifact': artifact, 'scientific_sources': {str(p): digest(p) for p in scientific},
          'commands': commands, 'gates': gates, 'scientific_gates': scientific_gates, 'models': models,
          'unique_inputs': U, 'queries': 387036, 'native_terminal': terminal, 'math_controls': math_controls,
          'reports_path': str(OUT / 'complete_reports.json'), 'unique_overall': summaries['unique_views'][0],
          'physical_diagnostics_path': str(OUT / 'physical_diagnostics.json'),
          'native_model_commands': 0, 'native_numeric_commands': 3, 'optimizer_updates': 12 * 63 * 32,
          'steps_per_head': 32, 'validation_training_examples': 0, 'fallback_queries': 0, 'preserved_daemons': daemons,
          'decision': ('LOCAL_SOURCE_TRANSFER_PASS' if all(scientific_gates.values()) else 'THIS_FIXED_LEARNED_SUPPORT2_PARTITION16_RECIPE_FAIL') + '_PENDING_INDEPENDENT_ADMISSION',
          'scope': 'Pretrained128 complete consumed-domain local router transfer only; 32 fixed development steps/all12banks/new physical C dot predictions; no complete artifact/fresh ownstate quality/rate/useful-n/LUT/DRAM/other-family promotion',
          'head_at_execution': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'tracked_status_at_terminal': last_status,
          'resource': {'seconds_before_RAW': time.monotonic() - START, 'OS_peak_combined_bytes': peak, 'file_bytes_hashed': hashed,
                       'hard_seconds': 1200, 'hard_peak_bytes': 4 << 30, 'hard_new_bytes': 512 << 20},
          'output_inventory': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']}
write(RAW, report)
guard()
checkpoint(terminal=True, raw_sha256=digest(RAW), OS_peak_combined_bytes=peak)
watchdog.cancel()
print(json.dumps({'raw': str(RAW), 'sha256': digest(RAW), 'gates': gates, 'scientific_gates': scientific_gates,
                  'overall': summaries['unique_views'][0], 'seconds': time.monotonic() - START,
                  'OS_peak_combined_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()
