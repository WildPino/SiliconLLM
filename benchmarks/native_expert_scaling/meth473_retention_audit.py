"""Independent original-WI/right-Gram spectral retention, no SVD/refit/replay."""
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
import subprocess
import numpy as np
import psutil
import threadpoolctl

ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth473_switch_operator_spectrum'
RAW = DOC / 'meth473_switch_operator_spectrum_result.json'
RET = DOC / 'RETENTION_473_20261005.json'
EVENTS = ROOT / 'results/native_expert_scaling/meth473_windows_terminal.json'
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
            h.update(data); hashed += len(data); guard()
    return h.hexdigest()
def head(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    assert Path(path).read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel])
def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

assert digest(RAW) == args.expected_raw_sha
j = json.loads(RAW.read_bytes())
assert len(j['apparatus_gates']) == 5 and all(j['apparatus_gates'].values())
assert j['native_or_model_commands'] == j['updates'] == 0
bpath = DOC / 'meth473_prospective_bindings.json'
head(bpath); assert digest(bpath) == j['source_binding_sha256']
b = json.loads(bpath.read_bytes())
for path, h in j['scientific_sources'].items(): head(path); assert digest(path) == h
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
for rel, row in b['helpers'].items(): head(ROOT / rel); assert digest(ROOT / rel) == row['sha256']
for name, row in b['records'].items(): head(DOC / name); assert (DOC / name).stat().st_size == row['bytes'] and digest(DOC / name) == row['sha256']
for path, row in b['runtime']['files'].items(): assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
for item in b['runtime']['packages'].values():
    for path, row in item['files'].items(): assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
for module in (np, psutil, threadpoolctl):
    assert module.__version__ == b['runtime']['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():
    assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
assert digest(b['preparation_helper']['path']) == b['preparation_helper']['sha256']
for path, row in b['native_files'].items(): assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
assert len(b['retained_inventory']) == 6338
for row in b['retained_inventory']:
    p = Path(row['path'])
    assert p.stat().st_size == row['bytes'] and p.stat().st_mtime_ns == row['mtime_ns'] and digest(p) == row['sha256']
for row in b['native_FFN_controls']: assert Path(row['path']).stat().st_size == row['bytes'] and digest(row['path']) == row['sha256']
for path, row in b['integer_fixtures'].items(): assert digest(path) == row['sha256']
payload = Path(j['artifact']['payload'])
assert j['artifact'] == b['original_artifact'] and digest(payload) == j['artifact']['sha256']
assert [payload.stat().st_size, payload.stat().st_mtime_ns] == j['payload_stat_before'] == [b['payload_stat_before']['bytes'], b['payload_stat_before']['mtime_ns']]
gates = {'frozen_sources_runtime_all6338_prior_inputs_original_payload_immutable': True}

assert len(j['output_inventory']) == 108
assert {Path(row['path']).resolve() for row in j['output_inventory']} == {p.resolve() for p in OUT.iterdir() if p.is_file() and p.name != 'progress.jsonl'}
for row in j['output_inventory']:
    assert Path(row['path']).stat().st_size == row['bytes'] and digest(row['path']) == row['sha256']
assert (OUT / 'fatal_native.log').stat().st_size == 0 and not RAW.with_suffix('.failure.json').exists()

old_out = ROOT / 'results/native_expert_scaling/meth472_switch_private_input_probe'
q = np.load(old_out / 'query_inputs.npy', mmap_mode='r', allow_pickle=False)
old = json.loads((DOC / 'meth472_switch_private_input_probe_result.json').read_bytes())
export = json.loads((DOC / 'meth380_switch_base128_export_result.json').read_bytes())
assert q.shape == (19962,) and q.dtype.itemsize == 4732
assert json.loads(json.dumps(q.dtype.descr)) == old['domain']['query_dtype_descr']
assert np.count_nonzero(q['role'] == 0) == 13313 and np.all(q['accepted'] == 1)
assert np.all(q['book'] < 192) and np.all(q['role'] == np.where((q['book'] >= 64) & (q['book'] < 128), 1, 0))
assert [e['expert'] for e in j['experts']] == b['fixed_ready_IDs'] == old['domain']['ready_IDs']
assert old['domain']['fallback_IDs'] == sorted(set(range(128)) - set(b['fixed_ready_IDs']))
assert len(j['experts']) == 107
derived = []
total = 0
for e, count in zip(j['experts'], b['development_representative_counts']):
    expert = e['expert']
    first = {}
    for i in np.flatnonzero((q['role'] == 0) & (q['expert'] == expert)):
        first.setdefault((int(q['book'][i]), q['codes'][i].tobytes(), q['alpha'][i].tobytes()), int(i))
    ids = np.asarray(sorted(first.values()), dtype='<u4')
    books = q['book'][ids]
    distinct, counts = np.unique(books, return_counts=True)
    sizes = dict(zip(map(int, distinct), map(int, counts)))
    weights = np.asarray([1 / (len(distinct) * sizes[int(book)]) for book in books], dtype='<f8')
    assert len(ids) == count == e['development_representatives'] and 34 <= count <= 308
    assert distinct.tolist() == e['development_books'] and abs(float(weights.sum()) - 1) <= 1e-12
    assert all(abs(float(weights[books == book].sum()) - 1 / len(distinct)) <= 1e-12 for book in distinct)
    with np.load(old_out / f'svd_witness_e{expert:03d}.npz', allow_pickle=False) as previous:
        exact(ids, previous['development_representatives']); exact(weights, previous['development_weights'])
    p = Path(e['witness'])
    assert p.resolve() == (OUT / f'operator_witness_e{expert:03d}.npz').resolve() and digest(p) == e['witness_sha256']
    with np.load(p, allow_pickle=False) as saved:
        assert set(saved.files) == {'singular_values', 'Vt', 'development_representatives', 'development_weights', 'tail_relative_squared'}
        exact(ids, saved['development_representatives']); exact(weights, saved['development_weights'])
        s, vt, saved_tail = saved['singular_values'], saved['Vt'], saved['tail_relative_squared']
    assert s.shape == (count,) and vt.shape == (count, count) and saved_tail.shape == (count + 1,)
    assert s.dtype == vt.dtype == saved_tail.dtype == np.dtype('<f8')
    assert np.isfinite(s).all() and np.isfinite(vt).all() and np.isfinite(saved_tail).all()
    assert np.all(s >= 0) and np.all(s[:-1] >= s[1:])
    assert all(vt[k, int(np.argmax(np.abs(vt[k])))] >= 0 for k in range(count))
    t = export['tensors'][f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.wi.weight']
    assert t['shape'] == [3072, 768] and t['encoding'] == 1
    with payload.open('rb') as stream:
        stream.seek(t['offset']); wbytes = stream.read(t['bytes'])
        stream.seek(t['scale_offset']); sbytes = stream.read(t['scale_bytes'])
    assert hashlib.sha256(wbytes).hexdigest() == t['sha256'] and hashlib.sha256(sbytes).hexdigest() == t['scale_sha256']
    w = np.frombuffer(wbytes, '<i1').reshape(3072, 768).astype(np.float64)
    scale = np.frombuffer(sbytes, '<f4').astype(np.float64)
    # Independent weighted-input-first association; no main/math import or SVD.
    weighted = (q['codes'][ids].astype(np.float64) * q['alpha'][ids].astype(np.float64)[:, None]) * np.sqrt(weights)[:, None]
    y = (w * scale[:, None]) @ weighted.T
    assert np.isfinite(y).all()
    energy = float(np.sum(y * y))
    denom = energy if energy else 1.
    gram = y.T @ y
    orth = float(np.linalg.norm(vt @ vt.T - np.eye(count)))
    eigen = float(np.linalg.norm(gram @ vt.T - vt.T * (s * s)) / denom)
    reconstruction = float(np.linalg.norm(gram - (vt.T * (s * s)) @ vt) / denom)
    balance = abs(float(np.sum(s * s)) - energy) / denom
    assert orth <= 1e-10 * math.sqrt(count) and eigen <= 1e-10 and reconstruction <= 1e-10 and balance <= 1e-10
    assert abs(energy - e['spectrum']['energy']) <= 1e-10 * denom
    tail = np.concatenate((np.cumsum((s * s)[::-1])[::-1], np.zeros(1)))
    relative = tail / energy if energy else np.zeros_like(tail)
    assert np.allclose(relative, saved_tail, rtol=1e-10, atol=1e-12)
    margin, threshold = (1e-6 if energy else 0.), 0.05**2
    lower, upper = np.maximum(relative - margin, 0), relative + margin
    rank = {name: int(np.flatnonzero(curve <= threshold)[0]) for name, curve in [('lower', lower), ('nominal', relative), ('upper', upper)]}
    for name, r in rank.items(): assert r == e['spectrum']['required_rank_' + name]
    def status(r):
        idx = min(r, count)
        return 'PASS' if upper[idx] <= threshold else ('FAIL' if lower[idx] > threshold else 'INDETERMINATE')
    for r in (32, 43):
        assert status(r) == e['spectrum'][f'rank{r}_status']
        assert abs(math.sqrt(float(relative[min(r, count)])) - e['spectrum'][f'rank{r}_RMS']) <= 1e-10
    derived.append({'expert': expert, 'ranks': rank, 'rank32': status(32), 'rank43': status(43),
                    'energy': energy, 'Gram_reconstruction_relative': reconstruction,
                    'eigen_relative': eigen, 'orthogonal_residual': orth, 'energy_relative_discrepancy': balance})
    total += count
    guard()
assert total == 11331
gates['all107_exact_development_dedup_book_weights_original_WI_full_response_Gram_witnesses'] = True
gates['all107_spectrum_energy_orthogonality_eigen_full_tail_status_minimum_rank_rederived'] = True

base, stride, original = 352202816, 15360, 605945856
cap = (7 * original - 10 * base) // (10 * stride)
assert cap == 4684 and j['budget']['base_bytes'] == base and j['budget']['bytes_per_rank'] == stride
assert j['budget']['original_bank_bytes'] == original and j['budget']['total_rank_cap'] == cap
sums = {name: sum(e['ranks'][name] for e in derived) for name in ('lower', 'nominal', 'upper')}
assert sums == j['budget']['sum_required_ranks']
assert base + stride * sums['nominal'] == j['budget']['ideal_nominal_rank_bank_bytes']
variable = 'INFEASIBLE' if sums['lower'] > cap else ('IDEAL_BUDGET_FEASIBLE' if sums['upper'] <= cap else 'INDETERMINATE')
all32 = all(e['rank32'] == 'PASS' for e in derived)
decision = {'ALL107_rank32': all32, 'uniform43_failed_IDs': [e['expert'] for e in derived if e['rank43'] == 'FAIL'],
            'variable_rank_ideal_development_budget': variable,
            'next': 'separately_freeze_rank32_full_function' if all32 else ('close_this_preactivation_preserving_F32_factor_family' if variable == 'INFEASIBLE' else 'review_all_required_ranks_and_whole_format_economics_before_candidate')}
assert decision == j['decision']
gates['complete107_21_policy_uniform_and_variable_storage_decision_rederived'] = True

terminal = json.loads((OUT / 'progress.jsonl').read_bytes().splitlines()[-1])
assert terminal['terminal'] and terminal['raw_sha256'] == args.expected_raw_sha and terminal['raw_bytes'] == RAW.stat().st_size
assert terminal['pid'] == j['main_process_instance']['pid'] and terminal['variable_rank_decision'] == variable
assert terminal['seconds'] <= 600 and terminal['OS_peak_bytes_after_raw'] <= 4 << 30
reused = []
try:
    p = psutil.Process(j['main_process_instance']['pid'])
    assert abs(p.create_time() - j['main_process_instance']['create_time_unix']) > .001, 'same main still live'
    reused.append({'pid': p.pid, 'create_time': p.create_time(), 'name': p.name()})
except psutil.NoSuchProcess: pass
events = json.loads(EVENTS.read_bytes())
assert events['main_pid'] == j['main_process_instance']['pid'] and events['query_available'] and events['query_error'] is None
assert not events['matching_main_events']
assert datetime.fromisoformat(events['query_start_utc']) <= datetime.fromisoformat(j['start_utc']) <= datetime.fromisoformat(j['end_utc']) <= datetime.fromisoformat(events['query_end_utc'])
files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
assert len(files) == 109
outbytes = sum(v['bytes'] for v in files)
assert outbytes + RAW.stat().st_size <= 128 << 20
assert sum(p.stat().st_size for p in OUT.glob('operator_witness*.npz')) <= b['witness_upper_bytes']
gates['actual_tool33007_exit0_main_instance_terminal_fatal_windows_bytes_resource_bounds'] = True
daemons = []
own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
for p in psutil.process_iter(['name', 'cmdline']):
    if p.pid in own: continue
    try:
        name, argv = (p.info['name'] or '').lower(), p.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): daemons.append(p.pid); continue
        assert not (name.startswith('python') or name == 'clang.exe' or (name.startswith('meth') and name.endswith('.exe')))
    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
for rel, row in b['preserved_unrelated_files'].items(): assert digest(ROOT / rel) == row['sha256']
assert digest(ROOT / 'benchmarks/phase60/engine.c') == b['original_engine']['sha256']
assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines() == b['tracked_status']
assert not {'torch', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'pyarrow', 'tokenizers', 'scipy'}.intersection(__import__('sys').modules)
gates['original_engine_unrelated3_actual_no_concurrent_scientific_job_preserved'] = True
helper_sha, event_sha = digest(__file__), digest(EVENTS)
query_sha = digest(ROOT / 'benchmarks/native_expert_scaling/meth473_windows_terminal.ps1')
guard()
r = {'experiment': 'METH473 independent right-Gram/rank/full-budget retention', 'raw_sha256': args.expected_raw_sha,
     'helper_sha256': helper_sha, 'windows_query_helper_sha256': query_sha, 'gates': gates,
     'derived_experts': derived, 'sum_required_ranks': sums, 'decision': decision,
     'main_terminal': {'tool_session': 33007, 'tool_exit_code': 0, 'actual_instance': j['main_process_instance'], 'PID_reuse_observations': reused},
     'windows_events': events, 'windows_events_sha256': event_sha, 'terminal_progress': terminal,
     'files': files, 'OUT_bytes': outbytes, 'raw_bytes': RAW.stat().st_size, 'all_main_output_bytes': outbytes + RAW.stat().st_size,
     'preserved_daemons': daemons, 'audit_seconds_before_RET_write': time.monotonic() - START,
     'audit_OS_peak_bytes_before_RET_write': peak, 'audit_file_bytes_hashed_before_RET_write': hashed,
     'scope': 'All107 primary spectra/energy/ranks/guarded statuses/budget decisions independently rederived from original WI and full development weighted inputs using right-Gram witnesses, no SVD/refit/main/native/model/tokenizer replay. Cached native integer1344WI qualification and main decoded/native64 rounding-envelope fields remain source-recorded, with fresh byte-bound controls. Numerical guard band is not a formal interval certificate.'}
assert len(gates) == 6 and all(gates.values())
with RET.open('xb') as s: s.write((json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode())
print(json.dumps({'RET': str(RET), 'RET_sha256': hashlib.sha256(RET.read_bytes()).hexdigest(),
                  'gates': gates, 'decision': decision, 'sum_required_ranks': sums,
                  'seconds': r['audit_seconds_before_RET_write'], 'OS_peak_bytes': peak}))
