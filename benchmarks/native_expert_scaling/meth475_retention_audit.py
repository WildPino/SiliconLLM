"""Independent nibble/margin/full-primal audit using chunked scalar I64 sums."""
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
OUT = ROOT / 'results/native_expert_scaling/meth475_switch_coarse_certificate'
RAW = DOC / 'meth475_switch_coarse_certificate_result.json'
RET = DOC / 'RETENTION_475_20261005.json'
EVENTS = ROOT / 'results/native_expert_scaling/meth475_windows_terminal.json'
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
bpath = DOC / 'meth475_prospective_bindings.json'
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
assert len(b['retained_inventory']) == 6557
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
gates = {'frozen_runtime_sources_all6557_inputs_original_payload_manifest_immutable': True}

for item in b['predecessor_preparation_helpers']: assert digest(item['path']) == item['sha256']
assert len(j['output_inventory']) == 9
assert {Path(v['path']).resolve() for v in j['output_inventory']} == {p.resolve() for p in OUT.iterdir() if p.is_file() and p.name != 'progress.jsonl'}
for v in j['output_inventory']:
    assert Path(v['path']).stat().st_size == v['bytes'] and digest(v['path']) == v['sha256']
assert (OUT / 'fatal_native.log').stat().st_size == 0 and not RAW.with_suffix('.failure.json').exists()
arrays = {name: np.load(OUT / (name + '.npy'), allow_pickle=False) for name in (
    'coarse_div8', 'certificate_bits', 'residual_norms', 'query_norms', 'query_metadata',
    'candidate_hidden_codes', 'candidate_hidden_scales', 'candidate_down')}
for name, shape, dtype in (('coarse_div8', (1344, 3072), '<i4'), ('certificate_bits', (1344, 384), 'u1'),
                         ('residual_norms', (128, 3072), '<u4'), ('query_norms', (1344,), '<u8'),
                         ('candidate_hidden_codes', (1344, 3072), '<i2'), ('candidate_hidden_scales', (1344,), '<f4'),
                         ('candidate_down', (1344, 768), '<f4')):
    assert arrays[name].shape == shape and arrays[name].dtype == np.dtype(dtype)
meta = arrays['query_metadata']
assert meta.shape == (1344,) and meta.dtype.itemsize == 16
assert json.loads(json.dumps(meta.dtype.descr)) == j['domain']['query_metadata_dtype_descr']
control_type = np.dtype([('position', '<u4'), ('id', '<u4'), ('source_tokens', '<u4'), ('route_index', '<u4'),
    ('pre', '<f4', (768,)), ('input', '<f4', (768,)), ('wi_scale', '<f4'), ('wi_codes', '<i2', (768,)),
    ('up_raw', '<f4', (3072,)), ('up', '<f4', (3072,)), ('wo_scale', '<f4'), ('wo_codes', '<i2', (3072,)),
    ('down', '<f4', (768,)), ('expert', '<i4'), ('accepted', '<i4'), ('probability', '<f4'),
    ('post', '<f4', (768,)), ('final', '<f4', (768,)), ('head_input', '<f4', (768,)), ('head_scale', '<f4'), ('head_codes', '<i2', (768,))])
assert control_type.itemsize == 52264 and len(b['native_FFN_controls']) == 96
captures = []
for index, item in enumerate(b['native_FFN_controls']):
    data = Path(item['path']).read_bytes()
    assert len(data) == 731728 and data[:32] == struct.pack('<8s6I', b'SWFUN001', 768, 3072, 32128, 128, 11, 1)
    rows = np.frombuffer(data, control_type, offset=32).copy()
    assert len(rows) == 14 and np.all(rows['accepted'] == 1)
    assert rows['position'].tolist() == list(range(14)) and rows['id'].tolist() == item['decoder_ids']
    assert np.all(rows['source_tokens'] == len(item['source_ids']))
    label = item['label'].split('.')
    book, case = int(label[2][4:]), int(label[3][4:])
    assert label[:2] == ['teacher', 'n128']
    actual = meta[index * 14:(index + 1) * 14]
    assert np.all(actual['capture'] == index) and np.all(actual['book'] == book) and np.all(actual['case'] == case)
    assert np.array_equal(actual['position'], rows['position']) and np.array_equal(actual['expert'], rows['expert'])
    assert np.array_equal(actual['decoder_id'], rows['id']) and np.array_equal(actual['source_tokens'], rows['source_tokens'])
    captures.append(rows)
controls = np.concatenate(captures)
assert len(controls) == 1344 and np.all((controls['expert'] >= 0) & (controls['expert'] < 128))
assert j['domain']['observed_IDs'] == sorted(set(controls['expert'].tolist()))

def scalar_chunked_dots(q, w):
    assert q.dtype == np.dtype('<i2') and w.dtype == np.dtype('i1') and q.shape[1] == w.shape[1]
    assert np.all(q != -32768) and q.shape[1] <= 3072
    total = np.zeros((len(q), len(w)), '<i8')
    # Exact I64 products/reductions; no F64 BLAS or main/math helper association.
    for first in range(0, q.shape[1], 64):
        x = q[:, first:first + 64].astype(np.int64)
        y = w[:, first:first + 64].astype(np.int64)
        total += np.einsum('ij,kj->ik', x, y, optimize=False, dtype=np.int64)
        guard()
    return total

def quant(values):
    assert values.dtype == np.dtype('<f4') and np.isfinite(values).all()
    maximum = np.amax(np.abs(values), axis=1)
    alpha = np.divide(maximum, np.float32(32767), dtype=np.float32)
    alpha[maximum == 0] = np.float32(1)
    assert np.all(alpha > 0)
    codes = np.clip(np.rint(np.divide(values, alpha[:, None], dtype=np.float32)), -32767, 32767).astype('<i2')
    return codes, alpha

def scaled(dots, alpha, row_scales):
    assert np.all(alpha > 0) and np.all(row_scales > 0)
    return np.multiply(np.multiply(dots.astype(np.float64), row_scales.astype(np.float64)),
                       alpha.astype(np.float64)[:, None]).astype('<f4')

def tensor(expert, kind):
    t = export['tensors'][f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.{kind}.weight']
    shape = [3072, 768] if kind == 'wi' else [768, 3072]
    assert t['shape'] == shape and t['encoding'] == 1
    with payload.open('rb') as stream:
        stream.seek(t['offset']); wb = stream.read(t['bytes'])
        stream.seek(t['scale_offset']); sb = stream.read(t['scale_bytes'])
    assert hashlib.sha256(wb).hexdigest() == t['sha256'] and hashlib.sha256(sb).hexdigest() == t['scale_sha256']
    w = np.frombuffer(wb, 'i1').reshape(shape); row_scales = np.frombuffer(sb, '<f4')
    assert np.isfinite(row_scales).all() and np.all(row_scales > 0)
    return w, row_scales

def packed(a):
    n = a.astype(np.uint8) & np.uint8(15)
    return (n[:, 0::2] + np.left_shift(n[:, 1::2], 4)).astype('u1')

max_t, max_c = 15 * 32767 * 768, 120 * 32767 * 768
assert max_t < 2**31 and max_c**2 < 2**63 and (768 * 64) * (768 * 32767**2) < 2**63
counts, nonzero = np.zeros(1344, '<u4'), np.zeros(1344, '<u4')
source_nonpositive = 0
derived = []
for e in j['experts']:
    expert = e['expert']
    assert expert == len(derived)
    wi, si = tensor(expert, 'wi'); wo, so = tensor(expert, 'wo')
    # Independent unsigned byte extraction, rather than signed floor division.
    u = wi.view('u1')
    h = (((u >> 4).astype(np.int16) + 8) % 16 - 8).astype('i1')
    residual = ((u & 15).astype(np.int16) - 8).astype('i1')
    exact((16 * h.astype(np.int16) + 8 + residual.astype(np.int16)).astype('i1'), wi)
    hp, bp = packed(h), packed(residual)
    r2 = np.sum(residual.astype(np.int64)**2, axis=1, dtype=np.int64).astype('<u4')
    exact(r2, arrays['residual_norms'][expert])
    assert hashlib.sha256(hp.tobytes()).hexdigest() == e['coarse_packed_sha256']
    assert hashlib.sha256(bp.tobytes()).hexdigest() == e['fine_packed_sha256']
    assert hashlib.sha256(r2.tobytes()).hexdigest() == e['residual_norm_sha256']
    ids = np.flatnonzero(controls['expert'] == expert)
    assert len(ids) == e['query_count']
    row = {'expert': expert, 'queries': len(ids)}
    if len(ids):
        selected = controls[ids]
        q, alpha = quant(selected['input'])
        exact(q, selected['wi_codes']); exact(alpha, selected['wi_scale'])
        t64 = 2 * scalar_chunked_dots(q, h) + np.sum(q.astype(np.int64), axis=1)[:, None]
        assert np.all(np.abs(t64) <= max_t)
        exact(t64.astype('<i4'), arrays['coarse_div8'][ids])
        q2 = np.sum(q.astype(np.int64)**2, axis=1, dtype=np.int64).astype('<u8')
        exact(q2, arrays['query_norms'][ids])
        c = 8 * t64
        margin = c * c - q2.astype(np.int64)[:, None] * r2.astype(np.int64)[None, :]
        cert = (c < 0) & (margin > 0)
        exact(np.packbits(cert, axis=1, bitorder='little'), arrays['certificate_bits'][ids])
        full_integer = scalar_chunked_dots(q, wi)
        assert np.all(full_integer[cert] < 0)
        raw = scaled(full_integer, alpha, si)
        exact(raw, selected['up_raw'])
        up = np.maximum(raw, np.float32(0)).astype('<f4')
        # Signed-zero up is not an exact-array gate, and source zero signs remain declared.
        assert np.array_equal(up, selected['up'])
        candidate_raw = raw.copy(); candidate_raw[cert] = 0
        hidden, ha = quant(np.maximum(candidate_raw, np.float32(0)).astype('<f4'))
        exact(hidden, selected['wo_codes']); exact(ha, selected['wo_scale'])
        exact(hidden, arrays['candidate_hidden_codes'][ids]); exact(ha, arrays['candidate_hidden_scales'][ids])
        complete = scaled(scalar_chunked_dots(hidden, wo), ha, so)
        exact(complete, selected['down']); exact(complete, arrays['candidate_down'][ids])
        counts[ids] = np.sum(cert, axis=1, dtype=np.uint32)
        nonzero[ids] = np.count_nonzero(hidden, axis=1).astype('<u4')
        assert int(np.sum(counts[ids], dtype=np.uint64)) == e['certified_rows']
        assert int(np.sum(nonzero[ids], dtype=np.uint64)) == e['candidate_nonzero_WO_codes']
        nneg = int(np.count_nonzero(raw <= 0)); assert nneg == e['source_negative_or_zero_rows']
        source_nonpositive += nneg
        row.update(certified_rows=int(np.sum(counts[ids], dtype=np.uint64)), source_nonpositive_rows=nneg)
    derived.append(row); guard()
assert len(derived) == 128 and sum(v['queries'] for v in derived) == 1344
gates['all128_full_codec_inverse_norms_and_all1344_independent_chunked_I64_coarse_margin_masks'] = True
gates['all1344_scalar_I64_full_source_WI_quantizer_hidden_and_complete_WO_primal_byte_exact'] = True

for i, previous in enumerate(j['per_query']):
    cert = np.unpackbits(arrays['certificate_bits'][i], bitorder='little').astype(bool)
    margin = 64 * arrays['coarse_div8'][i].astype(np.int64)**2 - arrays['query_norms'][i].astype(np.int64) * arrays['residual_norms'][int(meta['expert'][i])].astype(np.int64)
    result = {'query': i, 'book': int(meta['book'][i]), 'expert': int(meta['expert'][i]),
              'certified_rows': int(counts[i]), 'nonzero_hidden_codes': int(nonzero[i]),
              'coarse_WI_coefficient_bytes': 1179648, 'fine_WI_coefficient_bytes': (3072 - int(counts[i])) * 384,
              'WI_norm_bytes': 12288, 'WI_scale_bytes': 12288,
              'planned64aligned_WI_coefficient_lines': 18432 + (3072 - int(counts[i])) * 6,
              'planned64aligned_WI_norm_scale_lines': 384, 'planned64aligned_WO_column_lines': int(nonzero[i]) * 12,
              'min_certified_integer_margin': int(np.min(margin[cert])) if np.any(cert) else None}
    assert result == previous
assert len(j['per_query']) == 1344
books = []
total = 1344 * 3072; certified = int(np.sum(counts, dtype=np.uint64))
for book in range(24):
    ids = np.flatnonzero(meta['book'] == book)
    assert len(ids) == 56
    k = int(np.sum(counts[ids], dtype=np.uint64)); denom = len(ids) * 3072
    books.append({'book': book, 'queries': len(ids), 'certified_rows': k, 'total_rows': denom,
                  'certified_fraction': k / denom, 'ge0p40': 10 * k >= 4 * denom})
screen = dict(certified_rows=certified, total_rows=total, certified_fraction=certified / total,
              mean_WI_coefficient_read_fraction=1 - certified / (2 * total), books=books,
              planned_header_bytes=4096, planned_per_expert_stride=4746240,
              planned_complete_bank_bytes=4096 + 128 * 4746240, original_bank_coefficients_scales_bytes=605945856,
              actual_bank_exported=False, physical_DRAM_bytes_measured=False)
assert screen == j['certificate_screen']
recipe = {'mean_certified_rows_ge0p60': 10 * certified >= 6 * total, 'every_book_certified_rows_ge0p40': all(v['ge0p40'] for v in books)}
assert recipe == j['recipe_gates']
verdict = 'PASS' if all(recipe.values()) else 'FAIL'
decision = {'coarse_fine_integer_certificate_admission': verdict,
            'next': 'separately_freeze_one_complete_C_cost_primal' if verdict == 'PASS' else 'close_this_fixed_nibble_center_Cauchy_certificate_no_C_or_grid'}
assert decision == j['decision']
assert 2 * 1179648 + 2 * 12288 + 2359296 + 3072 == 4746240
gates['all1344_query24_book_economic_gates_lines_storage_complete_decision_rederived'] = True

terminal = json.loads((OUT / 'progress.jsonl').read_bytes().splitlines()[-1])
assert terminal['terminal'] and terminal['raw_sha256'] == args.expected_raw_sha and terminal['raw_bytes'] == RAW.stat().st_size
assert terminal['pid'] == j['main_process_instance']['pid'] and terminal['certificate_decision'] == verdict
assert terminal['seconds'] <= 180 and terminal['OS_peak_bytes_after_raw'] <= 2 << 30
reused = []
try:
    process = psutil.Process(j['main_process_instance']['pid'])
    assert abs(process.create_time() - j['main_process_instance']['create_time_unix']) > .001, 'same main still live'
    reused.append({'pid': process.pid, 'create_time': process.create_time(), 'name': process.name()})
except psutil.NoSuchProcess: pass
events = json.loads(EVENTS.read_bytes())
assert events['main_pid'] == j['main_process_instance']['pid'] and events['query_available'] and events['query_error'] is None
assert not events['matching_main_events']
assert datetime.fromisoformat(events['query_start_utc']) <= datetime.fromisoformat(j['start_utc']) <= datetime.fromisoformat(j['end_utc']) <= datetime.fromisoformat(events['query_end_utc'])
files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
assert len(files) == 10
outbytes = sum(v['bytes'] for v in files)
assert outbytes + RAW.stat().st_size <= b['output_upper_bytes'] <= 32 << 20
gates['actual_tool96814_exit0_instance_terminal_fatal_windows_all10_outputs_resource_bounds'] = True
daemons = []
own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
for process in psutil.process_iter(['name', 'cmdline']):
    if process.pid in own: continue
    try:
        name, argv = (process.info['name'] or '').lower(), process.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            daemons.append(process.pid); continue
        assert not (name.startswith('python') or name == 'clang.exe' or (name.startswith('meth') and name.endswith('.exe')))
    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
for rel, row in b['preserved_unrelated_files'].items(): assert digest(ROOT / rel) == row['sha256']
assert digest(ROOT / 'benchmarks/phase60/engine.c') == b['original_engine']['sha256']
assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines() == b['tracked_status']
assert not {'torch', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'pyarrow', 'tokenizers', 'scipy'}.intersection(sys.modules)
gates['original_engine_unrelated3_no_concurrent_scientific_job_preserved'] = True
helper_sha, event_sha = digest(__file__), digest(EVENTS)
query_sha = digest(ROOT / 'benchmarks/native_expert_scaling/meth475_windows_terminal.ps1')
guard()
r = {'experiment': 'METH475 independent scalar-chunked-I64 codec/full-primal/margin/count retention',
     'raw_sha256': args.expected_raw_sha, 'helper_sha256': helper_sha, 'windows_query_helper_sha256': query_sha,
     'gates': gates, 'derived_experts': derived, 'certificate_screen': screen, 'recipe_gates': recipe, 'decision': decision,
     'source_nonpositive_rows': source_nonpositive, 'source_nonpositive_fraction': source_nonpositive / total,
     'certified_fraction_among_source_nonpositive': certified / source_nonpositive if source_nonpositive else 0.,
     'main_terminal': {'tool_session': 96814, 'tool_exit_code': 0, 'actual_instance': j['main_process_instance'], 'PID_reuse_observations': reused},
     'windows_events': events, 'windows_events_sha256': event_sha, 'terminal_progress': terminal,
     'files': files, 'OUT_bytes': outbytes, 'raw_bytes': RAW.stat().st_size, 'all_main_output_bytes': outbytes + RAW.stat().st_size,
     'preserved_daemons': daemons, 'audit_seconds_before_RET_write': time.monotonic() - START,
     'audit_OS_peak_bytes_before_RET_write': peak, 'audit_file_bytes_hashed_before_RET_write': hashed,
     'scope': 'All128 codecs/norms and all1344 exact coarse dots/margins/masks/native WI/quantizers/hidden/full-WO functions independently rederived using chunked scalar I64 products/reductions, no F64 BLAS dot/main/math import/refit/native/model replay. Exact scalar bounds and logical counts/book gates/storage decisions verified. Tiny main controls and13fixtures remain source-recorded with fresh byte-bound assets. Consumed source prefixes, not fresh quality or packed C execution/physicalDRAM/rate/useful-n.'}
assert len(gates) == 6 and all(gates.values())
with RET.open('xb') as stream:
    stream.write((json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode())
print(json.dumps({'RET': str(RET), 'RET_sha256': hashlib.sha256(RET.read_bytes()).hexdigest(), 'gates': gates,
                  'decision': decision, 'certified_fraction': screen['certified_fraction'],
                  'seconds': r['audit_seconds_before_RET_write'], 'OS_peak_bytes': peak}))
