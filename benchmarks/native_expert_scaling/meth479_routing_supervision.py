"""ONE full-domain native-source admission and exact-input supervised compiler."""
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
OUT = ROOT / 'results/native_expert_scaling/meth479_routing_supervision'
RAW = DOC / 'meth479_routing_supervision_result.json'
BIND = DOC / 'meth479_prospective_bindings.json'
BIND_SHA = '0523b1822b84f6e11b40025b3c677116042a06853f4094375d271df73a4016ac'
C = ROOT / 'benchmarks/native_expert_scaling/meth479_source_router.c'
MATH = ROOT / 'benchmarks/native_expert_scaling/meth479_supervision_math.py'
WINDOWS = ROOT / 'benchmarks/native_expert_scaling/meth479_windows_terminal.ps1'
PROTO = DOC / 'METH_479_ROUTING_SUPERVISION_PROTOCOL_20261005.md'
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
        write(destination, {'experiment': 'METH479 FIRST full-domain source/supervision fault', 'phase': phase,
                            'start_utc': start_utc, 'end_utc': datetime.now(timezone.utc).isoformat(),
                            'process_instance': instance, 'commands': commands, 'binding_sha256': BIND_SHA,
                            'scientific_sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), C, MATH, WINDOWS, PROTO)},
                            'traceback': ''.join(traceback.format_exception(kind, value, tb)),
                            'resource': {'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak, 'file_bytes_hashed': hashed},
                            'partial_output_inventory': [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()]})
    checkpoint(terminal_failure=True)
    sys.__excepthook__(kind, value, tb)

sys.excepthook = failure
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
assert time.monotonic() - START <= 120
gates = {'complete_469_471_source_artifact_memberships_roles_runtime_actual_assets_frozen': True}
checkpoint(file_bytes_hashed=hashed)

weights_path, jobs_path = OUT / 'weights.ledger.bin', OUT / 'jobs.bin'
def string(stream, value):
    data = value.encode()
    stream.write(struct.pack('<I', len(data)) + data)
with weights_path.open('xb') as stream:
    stream.write(struct.pack('<8sII', b'M479WGT1', 12, 768))
    for item in b['weights']:
        stream.write(struct.pack('<IQ', item['bank'], item['offset']))
        string(stream, item['payload'])
with jobs_path.open('xb') as stream:
    stream.write(struct.pack('<8sII', b'M479JOB1', 1536, 387036))
    for item in b['tasks']:
        stream.write(struct.pack('<7I', *(item[k] for k in ('book', 'case', 'mode', 's', 't', 'role', 'queries'))))
        string(stream, item['trace'])
        string(stream, item['whole'])

def run(argv, label, expected=0):
    before = datetime.now(timezone.utc).isoformat()
    maximum = 0
    descendants = {}
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
                    info = proc.memory_info()
                    maximum = max(maximum, info.rss, getattr(info, 'peak_wset', 0))
                    for p in proc.children(recursive=True):
                        try:
                            key = (p.pid, p.create_time())
                            info = p.memory_info()
                            descendants[key] = {'pid': p.pid, 'create_time_unix': p.create_time(), 'executable': p.exe(),
                                                'OS_peak_bytes': max(info.rss, getattr(info, 'peak_wset', 0), descendants.get(key, {}).get('OS_peak_bytes', 0))}
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
    commands.append({'label': label, 'argv': [str(v) for v in argv], 'returncode': child.returncode, 'expected_returncode': expected,
                     'process_instance': pi, 'start_utc': before, 'end_utc': datetime.now(timezone.utc).isoformat(),
                     'OS_peak_bytes': maximum, 'descendant_process_peaks': list(descendants.values())})
    checkpoint(command_terminal=label, returncode=child.returncode)
    assert child.returncode == expected, (label, child.returncode, (OUT / (label + '.stderr.log')).read_bytes())
    return [json.loads(v) for v in (OUT / (label + '.stdout.log')).read_text().splitlines()]

phase = 'compile_controls_and_complete_original_native_source_admission'
binary = OUT / 'meth479_source_router.exe'
run([b['compiler']['path'], '-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off', C, '-o', binary], 'compile')
controls = run([binary, 'controls'], 'controls')
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
    left, right = [0.] * 4, [0.] * 4
    for i in range(0, 768, 8):
        for lane in range(4):
            left[lane] += a[i + lane] * x[i + lane]
            right[lane] += a[i + lane + 4] * x[i + lane + 4]
    result = 0.
    for lane in range(4):
        result += left[lane] + right[lane]
    assert controls[6 + kind] == {'dot_control': kind, 'double_result': result, 'f32_result': f32(result)}
assert controls[8]['CPU_affinity_mask'] == 1 and controls[8]['rounding'] == 0 and not controls[8]['MXCSR'] & 0x8040
bad = OUT / 'negative_weight_ledger.bin'
bad.write_bytes(b'INVALID!' + struct.pack('<2I', 12, 768))
negative_out = OUT / 'negative_source_fields.bin'
run([binary, 'source', bad, jobs_path, negative_out], 'negative_weight_ledger', 2)
assert not negative_out.exists() and (OUT / 'negative_weight_ledger.stderr.log').read_bytes() == b'source_router_error:weight_ledger_magic\r\n'
source_path = OUT / 'source_fields.bin'
source_terminal = run([binary, 'source', weights_path, jobs_path, source_path], 'source')
assert len(source_terminal) == 13
terminal = source_terminal[-1]
assert terminal['terminal'] and terminal['source_queries'] == 387036 and terminal['source_score_values'] == 49540608
assert terminal['scores_BYTE_exact'] and terminal['ID_probability_BYTE_exact'] and terminal['CPU_affinity_mask'] == 1 and not terminal['MXCSR'] & 0x8040
gates['Decimal100digits6_ordered_dot2_FPU_CPU0_and_negative_ledger_before_source'] = True
gates['ALL387036_original128_scores_winner_probability_BYTE_before_any_targets'] = True
checkpoint()

phase = 'exact_input_identities_source_joins_and_all_occurrence_links'
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
assert source_path.stat().st_size == 24 + 387036 * 72
with source_path.open('rb') as stream:
    assert stream.read(24) == struct.pack('<8sIIQ', b'M479SRC1', 72, 0, 387036)
source = np.memmap(source_path, dtype=M.SOURCE, mode='r', offset=24, shape=(387036,))
assert np.array_equal(source['meta'][:, 0], np.arange(387036)) and np.isfinite(source['value']).all()
assert {v['book']: v['role'] for v in b['tasks']}[128] == 2 and {v['book']: v['role'] for v in b['tasks']}[64] == 1
assert 2 != int(128 >= 64), 'high-book validation cutoff negative'
trace_dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
identities = {}
owners = []
serial = 0
paired_prefixes = 0
unique_path, links_path, owner_path, target_path = [OUT / v for v in ('unique_inputs.bin', 'query_links.bin', 'ownership.bin', 'group_targets.bin')]
with unique_path.open('x+b') as unique_stream, links_path.open('xb') as links_stream:
    unique_stream.write(struct.pack('<8sIIQ', b'M479UNI1', 3720, 768, 0))
    links_stream.write(struct.pack('<8sIIQ', b'M479LNK1', 52, 0, 387036))
    for task in b['tasks']:
        with Path(task['trace']).open('rb') as stream:
            assert stream.read(16) == struct.pack('<8sII', b'SWRTA001', 128, 768)
            data = stream.read()
        count, s, t = task['queries'], task['s'], task['t']
        assert len(data) == count * trace_dtype.itemsize
        trace = np.frombuffer(data, trace_dtype)
        idx = np.arange(count)
        assert np.array_equal(trace['index'], idx) and np.array_equal(trace['phase'], idx >= 6 * s)
        prefix = data[:180 * trace_dtype.itemsize]
        if task['mode'] == 0:
            teacher_prefix = prefix
        else:
            assert teacher_prefix == prefix
            paired_prefixes += 1
        with Path(task['whole']).open('rb') as stream:
            assert stream.read(36) == struct.pack('<8s7I', b'SWR32O01', s, t, 768, 12, 12, 32128, count)
            offset = 36 + 4 * (14 * s * 768 + 14 * t * 768 + t * 32128)
            assert Path(task['whole']).stat().st_size == offset + 12 * count
            stream.seek(offset)
            routes = np.frombuffer(stream.read(count * 12), np.dtype([('chosen', '<i4'), ('accepted', '<i4'), ('p', '<f4')]))
            assert not stream.read(1)
        native = source[serial:serial + count]
        bank = np.where(idx < 6 * s, idx // s, 6 + (idx - 6 * s) % 6)
        position = np.where(idx < 6 * s, idx % s, (idx - 6 * s) // 6)
        expected = np.column_stack((idx + serial, np.full(count, 128), np.full(count, task['book']), np.full(count, task['case']),
                                    np.full(count, task['mode']), np.full(count, task['role']), bank, position, idx,
                                    routes['chosen'], routes['accepted'], np.full(count, task['ordinal'])))
        assert np.array_equal(native['meta'], expected) and np.array_equal(routes['chosen'], np.argmax(trace['score'], axis=1))
        assert np.array_equal(native['value'][:, 1].astype('<f4').view('<u4'), routes['p'].view('<u4'))
        link_rows = np.empty((count, 13), '<u4')
        link_rows[:, :12] = native['meta']
        for i, row in enumerate(trace):
            meta, root_fields = native['meta'][i], native['value'][i]
            input_bytes, score_bytes = row['input'].tobytes(), row['score'].tobytes()
            ih, sh = hashlib.sha256(input_bytes).digest(), hashlib.sha256(score_bytes).digest()
            key = (int(bank[i]), ih)
            if key not in identities:
                uid = len(owners)
                identities[key] = uid
                first = (uid, 128, int(bank[i]), serial + i, task['book'], task['case'], task['mode'], int(meta[9]), int(meta[10]), task['role'], i, task['ordinal'])
                unique_stream.seek(0, 2)
                unique_stream.write(struct.pack('<12I', *first) + ih + sh + root_fields.tobytes() + input_bytes + score_bytes)
                owners.append([int(bank[i]), 0, 0, 0, 0, 0, 0])
            else:
                uid = identities[key]
                unique_stream.seek(24 + uid * 3720 + 48)
                recorded = unique_stream.read(3720 - 48)
                assert recorded == ih + sh + root_fields.tobytes() + input_bytes + score_bytes, 'full-BYTE input/hash/score/root collision'
            value = owners[uid]
            value[1] |= 1 << task['role']
            value[2] |= 1 << task['mode']
            value[3] |= 1 << int(meta[10])
            value[4] += 1
            value[5 if task['role'] != 1 else 6] |= 1 << task['book']
            link_rows[i, 12] = uid
        links_stream.write(link_rows.tobytes())
        serial += count
        if (task['ordinal'] + 1) % 128 == 0:
            unique_stream.flush()
            links_stream.flush()
            checkpoint(jobs_joined=task['ordinal'] + 1, queries_joined=serial, unique_inputs=len(owners))
        guard()
    unique_stream.seek(0)
    unique_stream.write(struct.pack('<8sIIQ', b'M479UNI1', 3720, 768, len(owners)))
assert serial == 387036 and paired_prefixes == 768 and len(owners) <= 248796
unique_count = len(owners)
assert unique_path.stat().st_size == 24 + 3720 * unique_count and links_path.stat().st_size == 24 + 52 * 387036
with owner_path.open('xb') as stream:
    stream.write(struct.pack('<8sIIQ', b'M479OWN1', 32, 0, unique_count))
    for uid, row in enumerate(owners):
        bank, role, mode, accepted, occurrence, dev_books, val_books = row
        assert dev_books & val_books == 0
        stream.write(struct.pack('<8I', uid, bank, role, mode, accepted, occurrence, dev_books.bit_count(), val_books.bit_count()))
development_book_bits = np.array([[v[5] & (2**64 - 1), v[5] >> 128] for v in owners], '<u8')
assert all(v[5] >> 64 & (2**64 - 1) == 0 for v in owners)
del owners, identities
unique = np.memmap(unique_path, dtype=M.UNIQUE, mode='r', offset=24, shape=(unique_count,))
owner = np.memmap(owner_path, dtype=M.OWNER, mode='r', offset=24, shape=(unique_count,))
links = np.memmap(links_path, dtype='<u4', mode='r', offset=24, shape=(387036, 13))
assert np.array_equal(unique['meta'][:, 0], np.arange(unique_count)) and np.array_equal(owner['meta'][:, 0], np.arange(unique_count))
gates['ALL1536_stream_joins_768_prefixes_full_BYTE_identities_all_links_roles_ownership_novelty'] = True
checkpoint(unique_inputs=unique_count)

phase = 'all_group_support_and_native_mass_targets'
memberships = []
for weight in b['weights']:
    with Path(weight['payload']).open('rb') as stream:
        stream.seek(weight['offset'])
        original = stream.read(weight['bytes'])
    assert hashlib.sha256(original).hexdigest() == weight['sha256']
    nodes = M.membership(b['membership_sources'][weight['bank']]['path'], weight['bank'], original)
    memberships.append({'bank': weight['bank'], 'nodes': nodes})
write(OUT / 'membership.json', memberships)
with target_path.open('xb') as stream:
    stream.write(struct.pack('<8sIIQ', b'M479TGT1', 3560, 127, unique_count))
    stream.truncate(24 + 3560 * unique_count)
saved_targets = np.memmap(target_path, dtype=M.TARGET, mode='r+', offset=24, shape=(unique_count,))
target_diagnostics = {'maximum_F64_path_probability_absolute_error': 0., 'F32_path_probability_differs_from_direct_native': 0, 'zero_child_mass_fields': 0}
for bank in range(12):
    ids = np.flatnonzero(unique['meta'][:, 2] == bank)
    for at in range(0, len(ids), 2048):
        uid = ids[at:at + 2048]
        records = unique[uid]
        target, values = M.targets(records['score'], records['value'], memberships[bank]['nodes'], uid)
        saved_targets[uid] = target
        for key, value in values.items():
            target_diagnostics[key] = max(target_diagnostics[key], value) if key.startswith('maximum') else target_diagnostics[key] + value
        guard()
    saved_targets.flush()
    checkpoint(target_bank=bank, bank_unique_inputs=len(ids))
assert np.array_equal(saved_targets['unique_id'], np.arange(unique_count))
assert np.isfinite(saved_targets['nodes']['mass']).all() and (saved_targets['nodes']['mass'] >= 0).all()
toy = np.array([0., 0., .5, -100.], '<f4')
toy_terms = np.exp((toy - toy[2]).astype('<f4').astype('<f8')).astype('<f4').astype('<f8')
assert int(np.argmax(toy)) == 2 and toy_terms[:2].sum() > toy_terms[2:].sum(), 'mass-argmax counterexample'
assert hashlib.sha256(struct.pack('<f', 0.)).digest() != hashlib.sha256(struct.pack('<f', -0.)).digest()
gates['ALL12_memberships_127_targets_per_unique_max_tie_native_mass_path_and_root_byte_qualified'] = True
checkpoint(target_unique_inputs=unique_count)

phase = 'all_exposure_and_ownership_reporting'
summary = {'ownership': M.owner_summary(owner), 'banks': [], 'occurrences': M.occurrence_tables(links)}
for bank in range(12):
    ids = np.flatnonzero(unique['meta'][:, 2] == bank)
    summary['banks'].append({'bank': bank, 'ownership': M.owner_summary(owner[ids]),
                             'branch_development_exposure': M.exposure(bank, unique, owner, saved_targets, memberships[bank]['nodes'], development_book_bits)})
    guard()
assert summary['ownership']['occurrences'] == 387036
gates['ALL72_occurrence_ID_views_13_ownership_views_1524_branch_exposure_records_complete'] = True
saved_targets.flush()
del saved_targets, unique, source, owner, links, development_book_bits
for rel, item in b['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == item['sha256']
last_status = preserved_status()
assert not ({'torch', 'transformers', 'tensorflow', 'scipy', 'sklearn', 'pandas'} & set(sys.modules))
gates['resource_caps_no_model_capture_training_SVD_query_filter_or_competing_science'] = True
guard()
report = {'experiment': 'METH479 complete pretrained support/native-mass supervision', 'start_utc': start_utc,
          'end_utc': datetime.now(timezone.utc).isoformat(), 'process_instance': instance, 'source_binding_sha256': BIND_SHA,
          'scientific_sources': {str(p): digest(p) for p in (Path(__file__), C, MATH, WINDOWS, PROTO)},
          'commands': commands, 'gates': gates, 'artifact': artifact, 'source_native_terminal': terminal,
          'queries': 387036, 'streams': 1536, 'unique_inputs': unique_count, 'unique_upper_bound': 248796,
          'group_target_records': unique_count * 127, 'prefix_pairs_BYTE_exact': paired_prefixes,
          'target_diagnostics': target_diagnostics, 'summaries': summary, 'preserved_daemons': daemons,
          'native_model_commands': 0, 'native_numeric_commands': 3, 'training_updates': 0,
          'decision': 'supervision_complete_pending_independent_retention_NOT_a_trained_candidate',
          'head_at_execution': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'tracked_status_at_terminal': last_status,
          'scope': 'Source128 complete original469+471 consumed-domain native targets and exact-input ownership; no fit/accuracy/wholecandidate/fresh quality/rate/useful-n/LUT/physicalDRAM/other-family claim',
          'resource': {'seconds_before_RAW': time.monotonic() - START, 'OS_peak_combined_bytes': peak, 'file_bytes_hashed': hashed,
                       'hard_seconds': 600, 'hard_peak_bytes': 4 << 30, 'hard_new_bytes': 2 << 30,
                       'binary_output_upper_bound': b['binary_upper_bound']},
          'output_inventory': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']}
write(RAW, report)
guard()
checkpoint(terminal=True, raw_sha256=digest(RAW), OS_peak_combined_bytes=peak)
print(json.dumps({'raw': str(RAW), 'sha256': digest(RAW), 'gates': gates, 'unique_inputs': unique_count,
                  'ownership': summary['ownership'], 'target_diagnostics': target_diagnostics,
                  'seconds': time.monotonic() - START, 'OS_peak_combined_bytes': peak}), flush=True)
progress.close()
faulthandler.disable()
fatal.close()
