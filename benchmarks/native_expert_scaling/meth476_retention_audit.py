"""Independent full context->norm/residual/A16 head/logit retention audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import time
START = time.monotonic()
import argparse
from datetime import datetime
import hashlib
import importlib.metadata as metadata
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

ROOT = Path.cwd(); DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth476_switch_last_layer_context'
RAW = DOC / 'meth476_switch_last_layer_context_result.json'; RET = DOC / 'RETENTION_476_20261005.json'
parser = argparse.ArgumentParser(); parser.add_argument('--expected-raw-sha', required=True); args = parser.parse_args()
assert not RET.exists()
peak = hashed = 0
parent = psutil.Process(); parent.cpu_affinity([0])
limits = threadpoolctl.threadpool_limits(limits=1)
np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
def guard():
    global peak
    info = parent.memory_info(); peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
    assert time.monotonic() - START <= 600 and peak <= 2 << 30
def digest(path):
    global hashed
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(4 << 20): h.update(data); hashed += len(data); guard()
    return h.hexdigest()
def head(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    assert Path(path).read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel])
def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (a.shape, b.shape)
def fail_hook(kind, value, tb):
    path = OUT / 'retention_audit_first_failure.json'
    record = {'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'raw_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
              'traceback': ''.join(traceback.format_exception(kind, value, tb)), 'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}
    if not path.exists():
        with path.open('xb') as stream: stream.write((json.dumps(record, indent=2) + '\n').replace('\n', '\r\n').encode())
    sys.__excepthook__(kind, value, tb)
sys.excepthook = fail_hook
assert digest(RAW) == args.expected_raw_sha
j = json.loads(RAW.read_bytes()); assert len(j['apparatus_gates']) == 6 and all(j['apparatus_gates'].values())
assert j['native_or_model_commands'] == 1 and j['updates'] == 0
bpath = DOC / 'meth476_prospective_bindings.json'; head(bpath); assert digest(bpath) == j['source_binding_sha256']
b = json.loads(bpath.read_bytes())
assert sys.version == b['runtime']['python'] and Path(sys.executable).resolve() == Path(b['runtime']['executable']).resolve()
for path, h in j['scientific_sources'].items(): head(path); assert digest(path) == h
head(__file__)
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
for rel, r in b['helpers'].items(): head(ROOT / rel); assert digest(ROOT / rel) == r['sha256']
for name, r in b['records'].items(): head(DOC / name); assert (DOC / name).stat().st_size == r['bytes'] and digest(DOC / name) == r['sha256']
for path, r in b['runtime']['files'].items(): assert Path(path).stat().st_size == r['bytes'] and digest(path) == r['sha256']
for package, item in b['runtime']['packages'].items():
    assert metadata.version(package) == item['version']
    for path, r in item['files'].items(): assert Path(path).stat().st_size == r['bytes'] and digest(path) == r['sha256']
for module in (np, psutil, threadpoolctl):
    assert module.__version__ == b['runtime']['packages'][module.__name__]['version']
    assert str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info(): assert pool['num_threads'] == 1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
for key in ('preparation_helper', 'controller_text_generation_helper'): assert digest(b[key]['path']) == b[key]['sha256']
for r in b['predecessor_preparation_helpers']: assert digest(r['path']) == r['sha256']
for path, r in b['native_files'].items(): assert Path(path).stat().st_size == r['bytes'] and digest(path) == r['sha256']
assert len(b['retained_inventory']) == 6567
for r in b['retained_inventory']:
    p = Path(r['path']); assert p.stat().st_size == r['bytes'] and p.stat().st_mtime_ns == r['mtime_ns'] and digest(p) == r['sha256']
for r in [*b['additional_source_files'], *b['native_FFN_controls']]: assert Path(r['path']).stat().st_size == r['bytes'] and digest(r['path']) == r['sha256']
for path, r in b['integer_fixtures'].items(): assert digest(path) == r['sha256']
payload = Path(j['artifact']['payload']); assert j['artifact'] == b['original_artifact'] and digest(payload) == j['artifact']['sha256']
assert [payload.stat().st_size, payload.stat().st_mtime_ns] == j['payload_stat_before'] == [b['payload_stat_before']['bytes'], b['payload_stat_before']['mtime_ns']]
assert digest(j['artifact']['manifest']) == j['artifact']['manifest_sha256']
for r in j['output_inventory']: assert Path(r['path']).stat().st_size == r['bytes'] and digest(r['path']) == r['sha256']
assert (OUT / 'fatal_native.log').stat().st_size == 0
assert all(c['returncode'] == 0 for c in j['commands']) and len(j['commands']) == 2
own = {os.getpid(), *(p.pid for p in parent.parents())}; daemons = []
for p in psutil.process_iter(['name', 'cmdline']):
    if p.pid in own: continue
    try:
        name, argv = (p.info['name'] or '').lower(), p.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): daemons.append(p.pid); continue
        assert not (name.startswith('python') or name.startswith('clang') or (name.startswith('meth') and name.endswith('.exe'))), (p.pid, name)
    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
events_path = ROOT / 'results/native_expert_scaling/meth476_windows_terminal.json'
events = json.loads(events_path.read_bytes()); assert events['query_available'] and not events['matching_scientific_events']
expected_instances = {(j['main_process_instance']['pid'], j['main_process_instance']['create_time_unix'])}
for cmd in j['commands']:
    expected_instances.add((cmd['process_instance']['pid'], cmd['process_instance']['create_time_unix']))
    expected_instances.update((r['pid'], r['create_time_unix']) for r in cmd['descendant_process_peaks'])
assert {(r['pid'], r['create_time_unix']) for r in events['instances']} == expected_instances
progress = [json.loads(line) for line in (OUT / 'progress.jsonl').read_text().splitlines()]
assert progress[-1]['terminal'] and progress[-1]['raw_sha256'] == args.expected_raw_sha and progress[-1]['seconds'] <= 600
assert progress[-1]['OS_peak_bytes_after_raw'] <= 2 << 30 and j['resource']['output_upper_bytes'] == 47690884
gates = {'all_frozen_actual_sources_runtime_payload_prior6567_extra193_outputs_terminal_instances_bound': True}
print(json.dumps({'stage': 'independent_context_primal', 'seconds': time.monotonic() - START}), flush=True)

def quant(x):
    maximum = np.max(np.abs(x), axis=1)
    alpha = np.where(maximum == 0, np.float32(1), maximum / np.float32(32767)).astype('<f4')
    codes = np.clip(np.rint(x / alpha[:, None]), -32767, 32767).astype('<i2')
    return codes, alpha
def norm(x, w, epsilon):
    total = np.zeros(len(x), '<f8')
    for coordinate in range(768): total += (x[:, coordinate] * x[:, coordinate]).astype('<f8')
    mean = (total / np.float64(768)).astype('<f4')
    scale = np.float32(1) / np.sqrt(mean + epsilon)
    return ((x * scale[:, None]) * w).astype('<f4')
def scaled(codes, w, row_scales, alpha):
    dots = codes.astype('<f8') @ w.T
    # Integer terms and every absolute partial sum <=768*128*32767<2^53.
    assert np.isfinite(dots).all() and np.equal(dots, np.rint(dots)).all()
    return ((dots * row_scales.astype('<f8')[None, :]) * alpha.astype('<f8')[:, None]).astype('<f4')
fixture = ROOT / 'results/native_expert_scaling/meth356_switch_all_a16_contract/integer_cases.bin'
answers = ROOT / 'results/native_expert_scaling/meth374_switch_physical_workers_contract/head-int-dot.bin'
fd, ad = fixture.read_bytes(), answers.read_bytes(); assert fd[:12] == struct.pack('<8sI', b'SWI8D001', 13) and ad[:12] == struct.pack('<8sI', b'SW16R001', 13)
assert (OUT / 'head-int-dot.bin').read_bytes() == ad
fc = ac = 12
for i in range(13):
    nr, nc = struct.unpack_from('<2I', fd, fc); fc += 8
    w = np.frombuffer(fd, '<i1', nr * nc, fc).reshape(nr, nc).astype('<f8'); fc += nr * nc
    si = np.frombuffer(fd, '<f4', nr, fc); fc += nr * 4
    x = np.frombuffer(fd, '<f4', nc, fc).reshape(1, nc); fc += nc * 4
    assert struct.unpack_from('<2I', ad, ac) == (nr, nc); ac += 8
    y = np.frombuffer(ad, '<f4', nr, ac); ac += nr * 4
    q = np.frombuffer(ad, '<i2', nc, ac); ac += nc * 2
    alpha = np.frombuffer(ad, '<f4', 1, ac); ac += 4
    dots = np.frombuffer(ad, '<i8', nr, ac); ac += nr * 8
    q1, a1 = quant(x); exact(q1[0], q); exact(a1, alpha)
    exact((q1.astype('<f8') @ w.T).astype('<i8')[0], dots); exact(scaled(q1, w, si, a1)[0], y)
assert fc == len(fd) and ac == len(ad)
gates['independent_exact_I8_A16_codes_dot_scales13_native_fixtures'] = True
export = json.loads((DOC / 'meth380_switch_base128_export_result.json').read_bytes())
def tensor(name):
    item = export['tensors'][name]
    with payload.open('rb') as stream:
        stream.seek(item['offset']); data = stream.read(item['bytes'])
        if item['encoding']:
            stream.seek(item['scale_offset']); sb = stream.read(item['scale_bytes'])
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    if item['encoding']:
        assert hashlib.sha256(sb).hexdigest() == item['scale_sha256']
        return np.frombuffer(data, '<i1').reshape(item['shape']).astype('<f8'), np.frombuffer(sb, '<f4')
    return np.frombuffer(data, '<f4').reshape(item['shape'])
wi_norm = tensor('decoder.block.11.layer.2.layer_norm.weight')
final_norm = tensor('decoder.final_layer_norm.weight')
hw, hs = tensor('lm_head.weight'); assert hw.shape == (32128, 768) and len(hs) == 32128
epsilon = np.float32(export['original_config']['layer_norm_epsilon'])
ctype = np.dtype([('qid', '<u8'), ('batch', '<u4'), ('position', '<u4'), ('pre', '<f4', (768,)), ('head_codes', '<i2', (768,)), ('head_alpha', '<f4')])
assert ctype.itemsize == 4628
assert (OUT / 'context.bin').stat().st_size == 16 + 7993 * 4628
with (OUT / 'context.bin').open('rb') as stream: assert stream.read(16) == struct.pack('<8sII', b'M476CTX1', 7993, 4628)
contexts = np.memmap(OUT / 'context.bin', dtype=ctype, mode='r', offset=16, shape=(7993,))
qpath = ROOT / 'results/native_expert_scaling/meth472_switch_private_input_probe/query_inputs.npy'
queries = np.load(qpath, mmap_mode='r', allow_pickle=False); reference = np.load(qpath.with_name('reference_functions.npy'), mmap_mode='r', allow_pickle=False)
assert queries.shape == (19962,) and queries.dtype.itemsize == 4732 and reference.shape == (19962, 768)
capture = json.loads((DOC / 'meth471_switch_development_capture_result.json').read_bytes()); ledger = Path(capture['ledger']['path']).read_bytes()
wire = struct.Struct('<H6BHIff32s32s32s'); assert ledger[:20] == struct.pack('<8sIQ', b'MQ471L01', 118, 387036)
trace_type = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (128,))])
cursor = 0; selected = []; gold_count = val_count = logit_count = 0; by_mode = [0, 0]
rescale = np.float32(1 / math.sqrt(768))
native_rows = [json.loads(line) for line in (OUT / 'native.stdout.log').read_text().splitlines()]
assert len(native_rows) == 609
for batch, item in enumerate(b['last_layer_batches']):
    t, s, kind = item['t'], item['s'], item['kind']; ctx = contexts[cursor:cursor + t]
    assert ctx['batch'].tolist() == [batch] * t and ctx['position'].tolist() == list(range(t))
    whole = Path(item['whole_path']).read_bytes(); trace = Path(item['trace_path']).read_bytes(); nr = 6 * (s + t)
    assert whole[:36] == struct.pack('<8s7I', b'SWR32O01', s, t, 768, 12, 12, 32128, nr)
    assert trace[:16] == struct.pack('<8sII', b'SWRTA001', 128, 768) and len(trace) == 16 + nr * 3592
    de_offset = 36 + 4 * 14 * s * 768; logit_offset = de_offset + 4 * t * 14 * 768; route_offset = logit_offset + 4 * t * 32128
    assert len(whole) == route_offset + nr * 12
    snapshots = np.frombuffer(whole, '<f4', t * 14 * 768, de_offset).reshape(t, 14, 768)
    expected_logits = np.frombuffer(whole, '<f4', t * 32128, logit_offset).reshape(t, 32128)
    rr = np.frombuffer(whole, np.dtype([('expert', '<i4'), ('accepted', '<i4'), ('p', '<f4')]), nr, route_offset)
    tr = np.frombuffer(trace, trace_type, offset=16); ix = 179 + 6 * np.arange(t)
    norm_input = norm(ctx['pre'], wi_norm, epsilon); exact(norm_input, tr['input'][ix]); probability = rr['p'][ix]
    if kind:
        ids = np.arange(item['query_start'], item['query_start'] + t); assert ctx['qid'].tolist() == ids.tolist()
        qr = queries[ids]; exact(norm_input, qr['input']); codes, alpha = quant(norm_input); exact(codes, qr['codes']); exact(alpha, qr['alpha']); exact(probability, qr['probability'])
        assert np.all(qr['role'] == 1) and np.all(qr['book'] == item['book']) and np.all(qr['case'] == item['case']) and np.all(qr['mode'] == item['mode']) and np.all(qr['accepted'] == 1)
        assert qr['ledger_record'].tolist() == (item['ledger_base'] + ix).tolist() and qr['index'].tolist() == ix.tolist()
        assert qr['expert'].tolist() == rr['expert'][ix].tolist()
        for k, row in enumerate(qr):
            hi, hq, hp = (qr[key][k:k + 1].tobytes() for key in ('input_sha', 'code_sha', 'pair_sha'))
            inp, qc, sb = norm_input[k].tobytes(), codes[k].tobytes(), alpha[k:k + 1].tobytes()
            assert (hi, hq, hp) == (hashlib.sha256(inp).digest(), hashlib.sha256(qc).digest(), hashlib.sha256(qc + sb).digest())
            li = int(row['ledger_record']); expected = wire.pack(128, item['mode'], item['book'], item['case'], 11, 1, 1, int(row['expert']), int(row['index']), float(row['alpha']), float(row['probability']), hi, hq, hp)
            assert ledger[20 + li * 118:20 + (li + 1) * 118] == expected
        source_down = reference[ids]; selected.extend(ids.tolist()); val_count += t; by_mode[item['mode']] += t
    else:
        assert np.all(ctx['qid'] == np.uint64((1 << 64) - 1))
        gd = Path(item['control_path']).read_bytes(); assert gd[:32] == struct.pack('<8s6I', b'SWFUN001', 768, 3072, 32128, 128, 11, 1) and len(gd) == 32 + t * 52264
        def golden(offset, dtype, width):
            return np.stack([np.frombuffer(gd, dtype, width, 32 + k * 52264 + offset) for k in range(t)])
        exact(ctx['pre'], golden(16, '<f4', 768)); exact(norm_input, golden(3088, '<f4', 768)); source_down = golden(38424, '<f4', 768)
        exact(ctx['head_codes'], golden(50728, '<i2', 768)); exact(ctx['head_alpha'], golden(50724, '<f4', 1)[:, 0]); gold_count += t
    post = (ctx['pre'] + probability[:, None] * source_down).astype('<f4'); exact(post, snapshots[:, 12, :])
    final = norm(post, final_norm, epsilon); exact(final, snapshots[:, 13, :])
    head_input = (final * rescale).astype('<f4'); hc, ha = quant(head_input); exact(hc, ctx['head_codes']); exact(ha, ctx['head_alpha'])
    if not kind: exact(head_input, golden(47652, '<f4', 768))
    actual_logits = scaled(hc, hw, hs, ha); exact(actual_logits, expected_logits)
    logit_count += t * 32128; cursor += t
    assert native_rows[batch] == {'batch': batch, 'kind': kind, 'book': item['book'], 'case': item['case'], 'mode': item['mode'], 'positions': t, 'cumulative_positions': cursor, 'byte_exact': True}
    guard()
    if batch % 96 == 95: print(json.dumps({'completed_batches': batch + 1, 'positions': cursor, 'seconds': time.monotonic() - START}), flush=True)
assert cursor == 7993 and gold_count == 1344 and val_count == 6649 and by_mode == [3584, 3065] and logit_count == 256799104
assert selected == np.flatnonzero(queries['role'] == 1).tolist() and len(set(selected)) == 6649
assert hashlib.sha256(struct.pack('<6649Q', *selected)).hexdigest() == j['query_admission']['all_original_query_indices_sha256']
assert set(queries['book'][selected].tolist()) == set(range(64, 128))
gates['ALL1344_known_true_pre_norm_residual_final_head_codes_ALL_logits_independent_BYTE_exact'] = True
gates['ALL6649_query_ledger_digests_roles_original_prefix_order_independent_exact'] = True
gates['ALL6649_pre_norm_source_residual_final_A16_head_ALL_logits_independent_BYTE_exact'] = True
terminal = native_rows[-1]; assert terminal == j['native_terminal'] and terminal['checked_logits'] == logit_count
assert terminal['terminal'] and terminal['goldens'] == 1344 and terminal['validation'] == 6649 and terminal['worker_physical_cores'] == 6
assert [r['actual_mask'] for r in terminal['worker_affinity']] == [1, 4, 16]
for rel, r in b['preserved_unrelated_files'].items(): assert digest(ROOT / rel) == r['sha256']
assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines() == b['tracked_status']
engine = ROOT / 'benchmarks/phase60/engine.c'; head(engine); assert digest(engine) == b['original_engine']['sha256']
assert not ({'torch', 'transformers', 'tensorflow', 'scipy', 'sklearn', 'pandas', 'pyarrow', 'tokenizers'} & set(sys.modules))
files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
total = sum(r['bytes'] for r in files) + RAW.stat().st_size
assert len(files) == 11 and total <= 64 << 20 and total <= b['output_upper_bytes']
gates['whole_count_scope_source_recorded_controls_preserved_budget_and_decision_exact'] = True
record = {'experiment': 'METH476 independent retention audit', 'raw_sha256': args.expected_raw_sha,
          'helper_sha256': digest(__file__), 'source_binding_sha256': j['source_binding_sha256'],
          'gates': gates, 'files': files, 'window_event_metadata': {'path': str(events_path), 'sha256': digest(events_path)},
          'independent_positions': 7993, 'independent_head_logits': logit_count, 'validation_positions': 6649,
          'native_recorded_only': 'last-layer attention reconstruction and original full WI/WO comparisons; cached native records independently byte bound',
          'decision': 'CONTEXT_ADMITTED_for_separately_frozen_observable_diagnostic_only', 'resource': {'seconds': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed, 'scientific_new_OUT_plus_RAW_bytes': total},
          'audit_process_instance': {'pid': os.getpid(), 'create_time_unix': parent.create_time()}, 'preserved_daemons': daemons}
with RET.open('xb') as stream: stream.write((json.dumps(record, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n').encode())
print(json.dumps({'retention': str(RET), 'sha256': digest(RET), 'gates': gates, 'seconds': time.monotonic() - START, 'OS_peak_bytes': peak}), flush=True)
