"""Independent stored-span Gram-solve audit; no SVD, QR, refit or forward replay."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import time
START = time.monotonic()
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import traceback
import numpy as np
import psutil
import threadpoolctl

ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth474_switch_private_output_floor'
RAW = DOC / 'meth474_switch_private_output_floor_result.json'
RET = DOC / 'RETENTION_474_20261005.json'
EVENTS = ROOT / 'results/native_expert_scaling/meth474_windows_terminal.json'
parser = argparse.ArgumentParser()
parser.add_argument('--expected-raw-sha', required=True)
args = parser.parse_args()
assert not RET.exists()
peak = hashed = 0
psutil.Process().cpu_affinity([0])
limits = threadpoolctl.threadpool_limits(limits=1)

def guard():
    global peak
    info = psutil.Process().memory_info()
    peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
    assert time.monotonic() - START <= 300 and peak <= 2 << 30

def digest(path):
    global hashed
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(4 << 20):
            h.update(data)
            hashed += len(data)
            guard()
    return h.hexdigest()

def head(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    assert Path(path).read_bytes() == subprocess.check_output([
        'git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel])

def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def fail_hook(kind, value, tb):
    path = OUT / 'retention_audit_first_failure.json'
    record = {'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'raw_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
              'traceback': ''.join(traceback.format_exception(kind, value, tb)),
              'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}
    if not path.exists():
        with path.open('xb') as stream:
            stream.write((json.dumps(record, indent=2) + '\n').replace('\n', '\r\n').encode())
    sys.__excepthook__(kind, value, tb)

sys.excepthook = fail_hook
assert digest(RAW) == args.expected_raw_sha
j = json.loads(RAW.read_bytes())
assert len(j['apparatus_gates']) == 6 and all(j['apparatus_gates'].values())
assert j['native_or_model_commands'] == j['updates'] == 0
bpath = DOC / 'meth474_prospective_bindings.json'
head(bpath)
assert digest(bpath) == j['source_binding_sha256']
b = json.loads(bpath.read_bytes())
for path, h in j['scientific_sources'].items():
    head(path)
    assert digest(path) == h
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
for rel, row in b['helpers'].items():
    head(ROOT / rel)
    assert digest(ROOT / rel) == row['sha256']
for name, row in b['records'].items():
    head(DOC / name)
    assert (DOC / name).stat().st_size == row['bytes'] and digest(DOC / name) == row['sha256']
for path, row in b['runtime']['files'].items():
    assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
for item in b['runtime']['packages'].values():
    for path, row in item['files'].items():
        assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
for module in (np, psutil, threadpoolctl):
    assert module.__version__ == b['runtime']['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
for key in ('preparation_helper', 'controller_text_generation_helper'):
    assert digest(b[key]['path']) == b[key]['sha256']
for path, row in b['native_files'].items():
    assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
assert len(b['retained_inventory']) == 6447
for row in b['retained_inventory']:
    p = Path(row['path'])
    assert p.stat().st_size == row['bytes'] and p.stat().st_mtime_ns == row['mtime_ns'] and digest(p) == row['sha256']
for row in b['native_FFN_controls']:
    assert Path(row['path']).stat().st_size == row['bytes'] and digest(row['path']) == row['sha256']
for path, row in b['integer_fixtures'].items():
    assert digest(path) == row['sha256']
payload = Path(j['artifact']['payload'])
assert j['artifact'] == b['original_artifact'] and digest(payload) == j['artifact']['sha256']
assert [payload.stat().st_size, payload.stat().st_mtime_ns] == j['payload_stat_before'] == [b['payload_stat_before']['bytes'], b['payload_stat_before']['mtime_ns']]
export = json.loads((DOC / 'meth380_switch_base128_export_result.json').read_bytes())
manifest = Path(j['artifact']['manifest']).read_bytes()
assert manifest[:8] == b'SWI8A001'
fields = ('d_model', 'd_ff', 'num_heads', 'd_kv', 'num_layers', 'num_decoder_layers', 'num_experts', 'expert_capacity', 'vocab_size', 'relative_attention_num_buckets', 'relative_attention_max_distance', 'encoder_sparse_step', 'decoder_sparse_step')
assert struct.unpack_from('<13I', manifest, 8) == tuple(export['original_config'][k] for k in fields)
assert manifest[60:64] == struct.pack('<f', export['original_config']['layer_norm_epsilon'])
assert struct.unpack_from('<2I', manifest, 64) == (1, 3320)
cursor = 72
def read_text():
    global cursor
    size = struct.unpack_from('<I', manifest, cursor)[0]
    cursor += 4
    value = manifest[cursor:cursor + size].decode()
    cursor += size
    return value
assert Path(read_text()).resolve() == payload.resolve()
for name, e in sorted(export['tensors'].items()):
    assert read_text() == name
    assert struct.unpack_from('<5I3Q', manifest, cursor) == (0, len(e['shape']), e['shape'][0], e['shape'][1] if len(e['shape']) == 2 else 1, e['encoding'], e['offset'], e['scale_offset'], e['elements'])
    cursor += 44
assert cursor == len(manifest)
gates = {'frozen_runtime_sources_all6447_inputs_original_payload_manifest_immutable': True}

assert len(j['output_inventory']) == 109
assert {Path(row['path']).resolve() for row in j['output_inventory']} == {p.resolve() for p in OUT.iterdir() if p.is_file() and p.name != 'progress.jsonl'}
for row in j['output_inventory']:
    assert Path(row['path']).stat().st_size == row['bytes'] and digest(row['path']) == row['sha256']
assert (OUT / 'fatal_native.log').stat().st_size == 0 and not RAW.with_suffix('.failure.json').exists()
old_out = ROOT / 'results/native_expert_scaling/meth472_switch_private_input_probe'
q = np.load(old_out / 'query_inputs.npy', mmap_mode='r', allow_pickle=False)
f = np.load(old_out / 'reference_functions.npy', mmap_mode='r', allow_pickle=False)
saved_metrics = np.load(OUT / 'per_query_floor_metrics.npy', allow_pickle=False)
old = json.loads((DOC / 'meth472_switch_private_input_probe_result.json').read_bytes())
assert q.shape == (19962,) and q.dtype.itemsize == 4732
assert json.loads(json.dumps(q.dtype.descr)) == old['domain']['query_dtype_descr']
assert f.shape == (19962, 768) and f.dtype == np.dtype('<f4') and np.isfinite(f).all()
assert saved_metrics.shape == (19962, 4) and saved_metrics.dtype == np.dtype('<f8')
assert np.count_nonzero(q['role'] == 0) == 13313 and np.all(q['accepted'] == 1)
assert np.all(q['book'] < 192) and np.all(q['role'] == np.where((q['book'] >= 64) & (q['book'] < 128), 1, 0))
assert [e['expert'] for e in j['experts']] == b['fixed_ready_IDs'] == old['domain']['ready_IDs']
assert j['domain']['fallback_IDs'] == old['domain']['fallback_IDs'] == sorted(set(range(128)) - set(b['fixed_ready_IDs']))
assert len(j['experts']) == 107
values = np.zeros((19962, 4), dtype='<f8')
values[:, 0] = np.sum(f.astype(np.float64)**2, axis=1)
u64, u32 = 2.**-53, 2.**-24
gamma = 64 * u64 / (1 - 64 * u64)
eta = gamma * math.sqrt(33) / .99999
lam, beta = eta + u32 * (1 + eta), math.sqrt(768) * 2.**-126
assert lam == j['rounding_scope']['relative_lambda'] and beta == j['rounding_scope']['absolute_beta']
assert j['rounding_scope']['rank'] == 32 and j['rounding_scope']['numerical_distance_guard'] == 1e-8
def reps(indices):
    first = {}
    for i in indices:
        first.setdefault((int(q['book'][i]), q['codes'][i].tobytes(), q['alpha'][i].tobytes()), int(i))
    ids = np.asarray(sorted(first.values()), dtype='<u4')
    books = q['book'][ids]
    distinct, counts = np.unique(books, return_counts=True)
    sizes = dict(zip(map(int, distinct), map(int, counts)))
    weights = np.asarray([1 / (len(distinct) * sizes[int(book)]) for book in books], dtype='<f8')
    assert len(ids) and abs(float(weights.sum()) - 1) <= 1e-12
    assert all(abs(float(weights[books == book].sum()) - 1 / len(distinct)) <= 1e-12 for book in distinct)
    return ids, weights

def summarize(ids, threshold, weights=None):
    v = values[ids]
    energy = np.sum(v if weights is None else v * weights[:, None], axis=0)
    rms = [math.sqrt(float(x) / energy[0]) if energy[0] else (0. if x == 0 else None) for x in energy[1:]]
    status = 'FAIL' if rms[1] is None or rms[1] > threshold else ('PASS' if rms[2] is not None and rms[2] <= threshold else 'INDETERMINATE')
    return dict(count=len(ids), threshold=threshold, reference_energy=float(energy[0]),
                ideal_projection_error_energy=float(energy[1]), conservative_lower_error_energy=float(energy[2]),
                constructive_upper_error_energy=float(energy[3]), ideal_RMS=rms[0], lower_RMS=rms[1], upper_RMS=rms[2], status=status)

def compare_summary(current, previous):
    assert current.keys() == previous.keys()
    for key, value in current.items():
        if key.endswith('_energy'):
            assert abs(value - previous[key]) <= 1e-10 * max(current['reference_energy'], 1e-20)
        elif key.endswith('_RMS'):
            assert (value is None and previous[key] is None) or abs(value - previous[key]) <= 1e-9
        else:
            assert value == previous[key]

derived = []
total_dev = total_val = 0
for e, dc, vc in zip(j['experts'], b['development_representative_counts'], b['novel_validation_representative_counts']):
    expert = e['expert']
    all_ids = np.flatnonzero(q['expert'] == expert)
    dev = all_ids[q['role'][all_ids] == 0]
    codes = {q['code_sha'][i:i + 1].tobytes() for i in dev}
    novel = [i for i in all_ids if q['role'][i] == 1 and q['code_sha'][i:i + 1].tobytes() not in codes]
    di, dw = reps(dev)
    vi, vw = reps(novel)
    assert len(di) == dc == e['development_representatives'] and len(vi) == vc == e['novel_validation_representatives']
    assert np.unique(q['book'][di]).tolist() == e['development_books']
    path = Path(e['witness'])
    assert path.resolve() == (OUT / f'output_witness_e{expert:03d}.npz').resolve() and digest(path) == e['witness_sha256']
    with np.load(path, allow_pickle=False) as z, np.load(old_out / f'svd_witness_e{expert:03d}.npz', allow_pickle=False) as previous:
        assert set(z.files) == {'U64_top32', 'P32', 'Q64', 'R64', 'singular_values', 'Vt', 'development_representatives', 'development_weights', 'validation_novel_representatives', 'validation_novel_weights'}
        for current, key in ((di, 'development_representatives'), (dw, 'development_weights'), (vi, 'validation_novel_representatives'), (vw, 'validation_novel_weights')):
            exact(current, z[key]); exact(current, previous[key])
        top, p32, orth, triangular, s, vt = (z[k] for k in ('U64_top32', 'P32', 'Q64', 'R64', 'singular_values', 'Vt'))
    for a, shape, dtype in ((top, (768, 32), '<f8'), (p32, (768, 32), '<f4'), (orth, (768, 32), '<f8'), (triangular, (32, 32), '<f8'), (s, (dc,), '<f8'), (vt, (dc, dc), '<f8')):
        assert a.shape == shape and a.dtype == np.dtype(dtype) and np.isfinite(a).all()
    exact(top.astype('<f4'), p32)
    assert np.all(s >= 0) and np.all(s[:-1] >= s[1:])
    assert all(top[int(np.argmax(np.abs(top[:, k]))), k] >= 0 for k in range(32))
    y = (f[di].astype(np.float64) * np.sqrt(dw)[:, None]).T
    energy = float(np.sum(y * y)); denom = energy if energy else 1.
    gram = y.T @ y
    eigen = float(np.linalg.norm(gram @ vt.T - vt.T * (s * s)) / denom)
    reconstruction = float(np.linalg.norm(gram - (vt.T * (s * s)) @ vt) / denom)
    right_orth = float(np.linalg.norm(vt @ vt.T - np.eye(dc)))
    top_relation = float(np.linalg.norm(y @ vt[:32].T - top * s[:32]) / math.sqrt(denom))
    assert eigen <= 1e-10 and reconstruction <= 1e-10 and abs(float(np.sum(s * s)) - energy) / denom <= 1e-10
    assert right_orth <= 1e-10 * math.sqrt(dc) and top_relation <= 1e-10
    assert np.linalg.norm(top.T @ top - np.eye(32)) <= 1e-10 * math.sqrt(dc)
    p = p32.astype(np.float64); g = p.T @ p
    gram_orth = float(np.linalg.norm(g - np.eye(32)))
    assert gram_orth <= 1e-6 and float(np.sum(p * p)) < 33
    assert np.linalg.norm(orth @ triangular - p) <= 1e-12 and np.linalg.norm(orth.T @ orth - np.eye(32)) <= 1e-12
    assert np.max(np.abs(np.tril(triangular, -1))) == 0
    tol = float(s[0] * max(y.shape) * np.finfo(np.float64).eps)
    rank = int(np.count_nonzero(s > tol))
    assert e['spectrum']['numerical_rank'] == rank and e['spectrum']['included_numerical_null_axes'] == max(0, 32 - rank)
    assert abs(e['spectrum']['energy'] - energy) <= 1e-10 * denom
    for first in range(0, len(all_ids), 64):
        ids = all_ids[first:first + 64]
        fv = f[ids].astype(np.float64)
        # Independent normal equations of the actual stored F32 span, no Q projection.
        coordinates = np.linalg.solve(g, p.T @ fv.T)
        projection = (p @ coordinates).T
        distance = np.linalg.norm(fv - projection, axis=1)
        norm = np.sqrt(values[ids, 0]); assert np.all(norm < np.finfo(np.float32).max / 8)
        lower = np.maximum((1 - lam) * np.maximum(distance - 1e-8 * norm, 0) - lam * norm - beta, 0)
        upper = distance + 1e-8 * norm + lam * norm + beta
        lower[norm == 0] = upper[norm == 0] = 0
        values[ids, 1:] = np.column_stack((distance**2, lower**2, upper**2))
        guard()
    result = summarize(vi, .05, vw)
    compare_summary(result, e['novel_validation_function_floor'])
    derived.append({'expert': expert, 'novel_validation_function_floor': result, 'development_representatives': dc,
                    'novel_validation_representatives': vc, 'right_Gram_reconstruction_relative': reconstruction,
                    'right_Gram_eigen_relative': eigen, 'right_orthogonal_residual': right_orth,
                    'top_left_singular_relation_relative': top_relation, 'stored_Gram_orthogonal_residual': gram_orth,
                    'numerical_rank': rank, 'included_numerical_null_axes': max(0, 32 - rank)})
    total_dev += dc; total_val += vc
    guard()
assert total_dev == 11331 and total_val == 5653
fallback = np.flatnonzero(~np.isin(q['expert'], b['fixed_ready_IDs']))
assert np.all(values[fallback, 1:] == 0) and np.all(saved_metrics[fallback, 1:] == 0)
assert np.isfinite(values).all() and np.all(values >= 0)
exact(values[:, 0], saved_metrics[:, 0])
discrepancy = np.abs(values - saved_metrics)
assert np.all(discrepancy <= 1e-10 * values[:, :1] + 1e-20)
gates['all107_exact_dev_novel_dedup_equal_book_weights_and_stored_span_spectrum_witnesses'] = True
gates['all19962_primary_floor_metrics_independent_Gram_solve_rounding_envelopes_fallback_zero'] = True
natural_ids = np.flatnonzero((q['role'] == 1) & (q['mode'] == 1))
assert len(natural_ids) == 3065
natural = summarize(natural_ids, .05)
compare_summary(natural, j['natural_validation_function_floor'])
books = []
for book, previous in zip(range(64, 128), j['natural_validation_books']):
    ids = natural_ids[q['book'][natural_ids] == book]
    result = {'book': book, **summarize(ids, .10)}
    compare_summary(result, previous); books.append(result)
assert len(books) == len(j['natural_validation_books']) == 64
states = [e['novel_validation_function_floor']['status'] for e in derived] + [natural['status']] + [e['status'] for e in books]
verdict = 'FAIL' if 'FAIL' in states else ('PASS' if all(s == 'PASS' for s in states) else 'INDETERMINATE')
decision = {'fixed_rank32_private_output_space_floor': verdict,
            'failed_novel_expert_IDs': [e['expert'] for e in derived if e['novel_validation_function_floor']['status'] == 'FAIL'],
            'indeterminate_novel_expert_IDs': [e['expert'] for e in derived if e['novel_validation_function_floor']['status'] == 'INDETERMINATE'],
            'next': 'separately_freeze_actual_WO_coefficients_full_function_candidate' if verdict == 'PASS' else ('close_this_learned_private_rank32_output_span' if verdict == 'FAIL' else 'retain_numerical_indeterminacy_no_candidate_or_rank_grid')}
assert decision == j['decision']
bank = 64 + 128 * 128 + 107 * (3072 * 768 + 4 * 3072 + 4 * 32 * (3072 + 768)) + 21 * (2 * 3072 * 768 + 4 * (3072 + 768))
assert bank == 405781568 and j['nominal_storage'] == {'complete_bank_bytes': bank, 'original_bank_bytes': 605945856, 'le70percent': 10 * bank <= 7 * 605945856, 'actual_bank_exported': False}
gates['all107_novel_3065_natural_64book_statuses_storage_complete_decision_rederived'] = True

terminal = json.loads((OUT / 'progress.jsonl').read_bytes().splitlines()[-1])
assert terminal['terminal'] and terminal['raw_sha256'] == args.expected_raw_sha and terminal['raw_bytes'] == RAW.stat().st_size
assert terminal['pid'] == j['main_process_instance']['pid'] and terminal['floor_decision'] == verdict
assert terminal['seconds'] <= 180 and terminal['OS_peak_bytes_after_raw'] <= 2 << 30
reused = []
try:
    process = psutil.Process(j['main_process_instance']['pid'])
    assert abs(process.create_time() - j['main_process_instance']['create_time_unix']) > .001, 'same main still live'
    reused.append({'pid': process.pid, 'create_time': process.create_time(), 'name': process.name()})
except psutil.NoSuchProcess:
    pass
events = json.loads(EVENTS.read_bytes())
assert events['main_pid'] == j['main_process_instance']['pid'] and events['query_available'] and events['query_error'] is None
assert not events['matching_main_events']
assert datetime.fromisoformat(events['query_start_utc']) <= datetime.fromisoformat(j['start_utc']) <= datetime.fromisoformat(j['end_utc']) <= datetime.fromisoformat(events['query_end_utc'])
files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
assert len(files) == 110
outbytes = sum(v['bytes'] for v in files)
assert outbytes + RAW.stat().st_size <= b['output_upper_bytes'] <= 96 << 20
assert sum(p.stat().st_size for p in OUT.glob('output_witness*.npz')) <= b['witness_upper_bytes']
gates['actual_tool50935_exit0_instance_terminal_fatal_windows_all110_outputs_resources'] = True
daemons = []
own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
for process in psutil.process_iter(['name', 'cmdline']):
    if process.pid in own: continue
    try:
        name, argv = (process.info['name'] or '').lower(), process.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            daemons.append(process.pid); continue
        assert not (name.startswith('python') or name == 'clang.exe' or (name.startswith('meth') and name.endswith('.exe')))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
for rel, row in b['preserved_unrelated_files'].items(): assert digest(ROOT / rel) == row['sha256']
assert digest(ROOT / 'benchmarks/phase60/engine.c') == b['original_engine']['sha256']
assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines() == b['tracked_status']
assert not {'torch', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'pyarrow', 'tokenizers', 'scipy'}.intersection(sys.modules)
gates['original_engine_unrelated3_no_concurrent_scientific_job_preserved'] = True
helper_sha, event_sha = digest(__file__), digest(EVENTS)
query_sha = digest(ROOT / 'benchmarks/native_expert_scaling/meth474_windows_terminal.ps1')
guard()
r = {'experiment': 'METH474 independent stored-F32 output-span Gram-solve retention', 'raw_sha256': args.expected_raw_sha,
     'helper_sha256': helper_sha, 'windows_query_helper_sha256': query_sha, 'gates': gates,
     'derived_experts': derived, 'natural_validation_function_floor': natural, 'natural_validation_books': books,
     'decision': decision, 'nominal_storage': j['nominal_storage'],
     'maximum_per_query_energy_discrepancy_over_reference': float(np.max(discrepancy / np.maximum(values[:, :1], 1e-20))),
     'main_terminal': {'tool_session': 50935, 'tool_exit_code': 0, 'actual_instance': j['main_process_instance'], 'PID_reuse_observations': reused},
     'windows_events': events, 'windows_events_sha256': event_sha, 'terminal_progress': terminal,
     'files': files, 'OUT_bytes': outbytes, 'raw_bytes': RAW.stat().st_size, 'all_main_output_bytes': outbytes + RAW.stat().st_size,
     'preserved_daemons': daemons, 'audit_seconds_before_RET_write': time.monotonic() - START,
     'audit_OS_peak_bytes_before_RET_write': peak, 'audit_file_bytes_hashed_before_RET_write': hashed,
     'scope': 'Primary full spectra/right-Gram/top-left relations/stored-F32 conditioning, exact dev/novel representatives and all function floor metrics/statuses independently rederived without SVD/QR/refit/forward replay. Full left-SVD reconstruction/orthogonality, tiny controls and cached original1344 native qualification remain source-recorded and freshly byte-bound. Numerical guards are empirical numeric guards, not formal interval certificates. Closes THIS learned fixed private rank32 stored output span under declared arithmetic/gates, not all rank32 or nonlinear representations.'}
assert len(gates) == 6 and all(gates.values())
with RET.open('xb') as stream:
    stream.write((json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode())
print(json.dumps({'RET': str(RET), 'RET_sha256': hashlib.sha256(RET.read_bytes()).hexdigest(), 'gates': gates,
                  'decision': decision, 'seconds': r['audit_seconds_before_RET_write'], 'OS_peak_bytes': peak}))
