"""Frozen original128 capture and expert-owned INPUT admission on cohort467."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth468_switch_native_domain_capture'
RAW = DOC / 'meth468_switch_native_domain_capture_result.json'
PROTOCOL = DOC / 'METH_468_SWITCH_NATIVE_DOMAIN_CAPTURE_PROTOCOL_20261005.md'
BINDINGS = DOC / 'meth468_prospective_bindings.json'
BINDINGS_SHA = 'a15510850296a09dc32c305873b50d99cbd6722ed0863402195e2283c5214394'
FORBIDDEN = {'torch', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'pyarrow', 'tokenizers'}
LEDGER = struct.Struct('<H6BHIff32s32s')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def committed(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    assert Path(path).read_bytes() == subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel], cwd=ROOT), rel


def write_new(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode()
    assert len(data) <= 64 << 20, ('raw64MiB', len(data))
    with Path(path).open('xb') as stream:
        stream.write(data)


def health(ids):
    # Original387 grammar/terminal/empty-field/repeated-trigram health contract.
    fields = [[] for _ in range(4)]
    expected = 0; current = -1; malformed = closed = False
    for token in ids:
        if token == 1: break
        if token == 0: malformed = True; continue
        if token >= 32000:
            if expected <= 4 and token == 32099 - expected:
                if expected == 4: closed = True; current = -1
                else: current = expected
                expected += 1
            else: malformed = True
        elif 0 <= current < 4: fields[current].append(token)
        else: malformed = True
    if malformed: fields = [[] for _ in range(4)]
    trigrams = [[tuple(field[i:i + 3]) for i in range(max(len(field) - 2, 0))] for field in fields]
    total = sum(map(len, trigrams))
    repeated = (total - sum(len(set(v)) for v in trigrams)) / max(total, 1)
    good = closed and expected == 5 and not malformed and all(fields) and ids[-1] == 32095 and repeated <= .50
    return {'healthy_complete_nonempty_fields': bool(good), 'fields': fields,
            'within_field_repeated_trigram_fraction': repeated,
            'cap_reached_without_terminal': len(ids) == 64 and ids[-1] not in (1, 32095)}


def native_bytes(s, t):
    whole = 36 + 4 * (14 * s * 768 + 14 * t * 768 + t * 32128) + 72 * (s + t)
    trace = 16 + 6 * (s + t) * (8 + 4 * (768 + 128))
    return whole + trace


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert args.out.resolve() == RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and not OUT.exists()
    assert os.name == 'nt' and sys.flags.optimize == 0
    start = time.monotonic(); stage = 'bindings_before_numpy_import'
    root_peak = child_peak = aggregate = hashed = count = output_bytes = 0
    last_scan = -10.; active = None
    result = {'experiment': 'METH468-original128-cohort467-native-private-INPUT-domain',
              'start_utc': utc(), 'argv': sys.argv.copy(), 'commands': [], 'cases': [],
              'reused_files': [], 'gates': {}, 'source_binding_sha256': BINDINGS_SHA}
    OUT.mkdir()
    progress = (OUT / 'progress.jsonl').open('x', encoding='utf-8')
    fault = (OUT / 'fatal_native.log').open('xb'); faulthandler.enable(file=fault, all_threads=True)
    def checkpoint(**values):
        progress.write(json.dumps({'stage': stage, 'pid': os.getpid(), 'utc': utc(), 'seconds': time.monotonic() - start, **values}) + '\n')
        progress.flush()
    checkpoint()
    try:
        import psutil
        parent = psutil.Process()
        def guard(child=None, force_scan=False):
            nonlocal root_peak, child_peak, aggregate, last_scan, output_bytes
            info = parent.memory_info(); root_peak = max(root_peak, info.rss, getattr(info, 'peak_wset', 0))
            rss = info.rss
            for process in parent.children(recursive=True):
                try:
                    mem = process.memory_info(); rss += mem.rss
                    child_peak = max(child_peak, mem.rss, getattr(mem, 'peak_wset', 0))
                except psutil.NoSuchProcess: pass
            aggregate = max(aggregate, rss)
            if force_scan or time.monotonic() - last_scan >= 2:
                output_bytes = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
                assert output_bytes + (RAW.stat().st_size if RAW.exists() else 0) <= 12 << 30, 'all_outputs12GiB'
                assert (OUT / 'progress.jsonl').stat().st_size <= 2 << 20, 'progress2MiB'
                last_scan = time.monotonic()
            assert root_peak + child_peak <= 16 << 30 and aggregate <= 16 << 30, 'conservative_parent_child16GiB'
            assert time.monotonic() - start <= 2100, 'main35min'
        def digest(path):
            nonlocal hashed
            assert active is None, 'no_hashing_while_native_child_live'
            h = hashlib.sha256()
            with Path(path).open('rb') as stream:
                while data := stream.read(4 << 20): h.update(data); hashed += len(data); guard()
            return h.hexdigest()
        def jobs():
            own = {os.getpid(), *(p.pid for p in parent.parents())}; preserved = []
            for p in psutil.process_iter(['name', 'cmdline']):
                if p.pid in own: continue
                name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
                if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                    preserved.append(p.pid); continue
                assert not (name.startswith('python') or name == 'clang.exe' or (name.startswith('meth') and name.endswith('.exe'))), (p.pid, name)
            return preserved
        result['preserved_daemons_before'] = jobs()
        for p in (Path(__file__), PROTOCOL, BINDINGS, ROOT / 'benchmarks/phase60/engine.c'): committed(p)
        assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'], cwd=ROOT)
        assert digest(BINDINGS) == BINDINGS_SHA
        bindings = json.loads(BINDINGS.read_text(encoding='utf-8'))
        result.update(controller_sha256=digest(__file__), protocol_sha256=digest(PROTOCOL),
                      head_at_execution=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip())
        assert str(Path(sys.executable).resolve()) == bindings['runtime']['executable'] and sys.version == bindings['runtime']['python']
        for path, row in bindings['runtime']['files'].items():
            assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
        for name, package in bindings['runtime']['packages'].items():
            assert metadata.version(name) == package['version']
            for path, row in package['files'].items(): assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
        for rel, row in bindings['helpers'].items():
            committed(ROOT / rel); assert digest(ROOT / rel) == row['sha256']
        assert digest(ROOT / 'benchmarks/phase60/engine.c') == bindings['original_engine']['sha256']
        result['preserved_engine_sha256'] = bindings['original_engine']['sha256']
        records = {}
        for name, row in bindings['records'].items():
            committed(DOC / name); assert digest(DOC / name) == row['sha256']
            records[name] = json.loads((DOC / name).read_text(encoding='utf-8'))
        assert bindings['records']['meth467_switch_rust_query_manifest.json']['sha256'] == 'd90553a6b4454fae41978d7a1221f7389ea4c52a9517eb0a0ea079b0840bddb5'
        assert bindings['records']['RETENTION_467_20261005.json']['sha256'] == '651f724f2221ddd759a92f8eb9aa9cfda87bb5a0e9f7a190d1e2a5f120c77117'
        cohort = records['meth467_switch_rust_query_manifest.json']; capture = records['meth393_switch_router_audit_result.json']
        assert all(cohort['gates'].values()) and all(records['RETENTION_467_20261005.json']['gates'].values()) and all(capture['gates'].values())
        assert all(records['meth462_switch_query_domain_admission_result.json']['gates'].values())
        for rel, row in bindings['preserved_unrelated_files'].items(): assert digest(ROOT / rel) == row['sha256']
        assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True, cwd=ROOT).splitlines() == bindings['tracked_status']
        for path, row in bindings['native_files'].items(): assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
        target = next(v for v in capture['targets'] if v['n'] == 128); artifact = target['artifact']
        assert artifact == bindings['original_artifact']
        payload = Path(artifact['payload']); before = [payload.stat().st_size, payload.stat().st_mtime_ns]
        assert before == [bindings['payload_stat_before']['bytes'], bindings['payload_stat_before']['mtime_ns']]
        assert before[0] == artifact['bytes'] and digest(payload) == artifact['sha256']
        assert digest(artifact['manifest']) == artifact['manifest_sha256']
        manifest = Path(artifact['manifest']).read_bytes()
        assert manifest[:8] == b'SWI8A001'
        config = struct.unpack_from('<13I', manifest, 8)
        assert (config[0], config[1], config[4], config[5], config[6], config[7], config[8], config[11], config[12]) == (768, 3072, 12, 12, 128, 64, 32128, 2, 2)
        binary = Path(capture['compile']['argv'][-1]); dll = binary.parent / 'libomp.dll'
        assert digest(binary) == capture['compile']['binary_sha256'] == '67cc62872ea51748f2f92818567dbfcc54c94a152606dfde674ad65a5f3b4afd'
        assert digest(dll) == capture['compile']['runtime_sha256']
        result.update(artifact=artifact, payload_stat_before=before, native_binary={'path': str(binary), 'sha256': capture['compile']['binary_sha256']},
                      native_runtime_sha256=capture['compile']['runtime_sha256'])
        assert len(cohort['items']) == 128 and all(len(book['cases']) == 4 for book in cohort['items'])
        assert len({book['whole_source_utf8_sha256'] for book in cohort['items']}) == 128
        for i, book in enumerate(cohort['items']):
            assert book['book'] == i and book['role'] == ('development' if i < 64 else 'diagnostic_validation')
            for ci, case in enumerate(book['cases']):
                assert case['index'] == ci and len(case['source_ids']) == 29 and len(case['decoder_ids']) == len(case['target_ids']) == 14
                assert case['source_ids_sha256'] == sha(struct.pack('<29i', *case['source_ids']))
                assert case['target_ids_sha256'] == sha(struct.pack('<14i', *case['target_ids'])) and case['decoder_ids'] == [0] + case['target_ids'][:-1]
        bound = 516 * (native_bytes(29, 14) + native_bytes(29, 64)) + (256 << 20)
        max_queries = 512 * 6 * (29 + 14 + 29 + 64)
        assert bound < 12 << 30 and 20 + max_queries * LEDGER.size < 64 << 20
        assert shutil.disk_usage(ROOT).free >= 16 << 30
        result['prospective_byte_bounds'] = {'native_and_metadata_upper_bound': bound, 'ledger_max_records': max_queries,
                                             'ledger_upper_bytes': 20 + max_queries * LEDGER.size, 'metadata_reserve_bytes': 256 << 20}
        result['gates']['strict_frozen_source_runtime_original_payload_binary_records_roles'] = True
        checkpoint()

        stage = 'qualified_helpers_A16_controls_and_original_golden_files'
        import numpy as np
        import meth462_switch_query_domain_admission as D
        import meth359_switch_cost_topology as A
        assert np.__version__ == bindings['runtime']['packages']['numpy']['version']
        assert psutil.__version__ == bindings['runtime']['packages']['psutil']['version']
        assert str(Path(np.__file__).resolve()) in bindings['runtime']['packages']['numpy']['files']
        assert str(Path(psutil.__file__).resolve()) in bindings['runtime']['packages']['psutil']['files']
        result['actual_loaded_package_paths'] = {'numpy': str(Path(np.__file__).resolve()), 'psutil': str(Path(psutil.__file__).resolve())}
        assert not sorted(name for name in sys.modules if name.partition('.')[0] in FORBIDDEN)
        assert A.physical_topology() == capture['fresh_topology']
        result['fresh_topology'] = A.physical_topology()
        fixture = ROOT / 'results/native_expert_scaling/meth356_switch_all_a16_contract/integer_cases.bin'
        answer = ROOT / 'results/native_expert_scaling/meth374_switch_physical_workers_contract/head-int-dot.bin'
        assert digest(fixture) == 'e404e9a2501810532630e483450ce2bb3bcf7db7e5100d1a947b49a19ecdeb94'
        assert digest(answer) == '0162a37a6a521c7ddd9c80097c193eaa0e7e85e39c056aa88bb5017f68559b87'
        data, output = fixture.read_bytes(), answer.read_bytes(); a = b = 12
        assert data[:8] == b'SWI8D001' and output[:8] == b'SW16R001'
        assert struct.unpack_from('<I', data, 8) == struct.unpack_from('<I', output, 8) == (13,)
        for ci in range(13):
            rows, cols = struct.unpack_from('<2I', data, a); a += 8
            w = np.frombuffer(data, '<i1', rows * cols, a).reshape(rows, cols); a += rows * cols
            scales = np.frombuffer(data, '<f4', rows, a); a += rows * 4
            x = np.frombuffer(data, '<f4', cols, a).reshape(1, cols); a += cols * 4
            assert struct.unpack_from('<2I', output, b) == (rows, cols); b += 8
            y = np.frombuffer(output, '<f4', rows, b); b += rows * 4
            q = np.frombuffer(output, '<i2', cols, b); b += cols * 2
            alpha = struct.unpack_from('<f', output, b)[0]; b += 4
            dots = np.frombuffer(output, '<i8', rows, b); b += rows * 8
            aq, aa = D.quantize(x); sq, sa = D.scalar_quantize(x[0])
            actual_dot = w.astype(np.int64) @ aq[0].astype(np.int64)
            actual_y = ((actual_dot.astype(np.float64) * scales.astype(np.float64)) * alpha).astype('<f4')
            assert aq[0].tobytes() == q.tobytes() and aq[0].tolist() == sq and float(aa[0]) == alpha == sa
            assert actual_dot.tobytes() == dots.tobytes() and actual_y.tobytes() == y.tobytes(); guard()
        assert a == len(data) and b == len(output)
        for values in ([0.] * 17, [32767., .5, -.5, 1.5, -1.5, 2.5, -2.5], [.01, -.003, .005, .0007],
                       [1e-20, -2e-20, 3e-20], [1e20, -2e20, 3e20], [2. ** -133, 2. ** -140, -2. ** -140]):
            x = np.asarray([values], dtype=np.float32); q, alpha = D.quantize(x); sq, sa = D.scalar_quantize(x[0])
            assert q[0].tolist() == sq and float(alpha[0]) == sa
        assert len(target['quality_bridges']) == 96
        golden = {}
        for entry in target['quality_bridges']:
            for mode in ('teacher', 'natural'):
                stored = entry['teacher_rows' if mode == 'teacher' else 'generation_rows'][0]
                command = next(v for v in capture['commands'] if v['trace_path'] == stored['router_trace_path'])
                whole = Path(command['argv'][4] + '.0.bin'); trace = Path(stored['router_trace_path'])
                assert command['returncode'] == 0 and not command['negative']
                assert digest(whole) == stored['output_sha256'] and digest(trace) == stored['router_trace_sha256']
                result['reused_files'].append({'book': entry['book'], 'case': entry['case'], 'mode': mode,
                    'whole': str(whole), 'whole_sha256': stored['output_sha256'], 'trace': str(trace), 'trace_sha256': stored['router_trace_sha256']})
            golden[(entry['book'], entry['case'])] = entry
        assert len(result['reused_files']) == 192
        result['gates']['ALL192_original128_retained_whole_trace_SHA_and13plus6_A16_controls'] = True
        assert time.monotonic() - start <= 180, 'admission180s'

        class MemoryCounters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
                        ('peak_rss', 'rss', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
        get_memory = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]; get_memory.restype = wintypes.BOOL
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_WORKER_BINDING_', 'SILICON_ROUTER_'))}
        env.update(capture['runtime_environment']); env['OMP_NUM_THREADS'] = '3'
        def invoke(source, decoder, label, mode, negative=False):
            nonlocal active, child_peak
            prefix = OUT / label; trace = OUT / (label + '.router.bin')
            argv = ([str(binary), str(artifact['manifest']), ','.join(map(str, source)), ','.join(map(str, decoder)), str(prefix), '3', '0', '0', '1', '0']
                    if mode == 'teacher' else [str(binary), '--generate', str(artifact['manifest']), ','.join(map(str, source)), str(prefix), '3', '64', '32095', '0', '0', '1', '0'])
            row = {'argv': argv.copy(), 'label': label, 'mode': mode, 'negative': negative, 'start_utc': utc()}
            result['commands'].append(row)
            assert not prefix.exists() and not trace.exists() and not Path(str(prefix) + '.0.bin').exists()
            child_env = dict(env); child_env['SILICON_ROUTER_AUDIT_PATH'] = str(trace)
            if negative: child_env['SILICON_WORKER_BINDING_FAULT'] = '1'
            child_start = time.monotonic()
            with (OUT / (label + '.stdout.log')).open('xb') as stdout, (OUT / (label + '.stderr.log')).open('xb') as stderr:
                child = subprocess.Popen(argv.copy(), stdout=stdout, stderr=stderr, env=child_env); active = child
                row['pid'] = child.pid; checkpoint(child_started=child.pid, label=label)
                try:
                    if not negative:
                        process = psutil.Process(child.pid); process.cpu_affinity([0, 2, 4]); row['actual_affinity'] = process.cpu_affinity()
                        assert row['actual_affinity'] == [0, 2, 4]
                    while child.poll() is None:
                        guard(child); assert time.monotonic() - child_start <= 120, 'native_child120s'; time.sleep(.05)
                    memory = MemoryCounters(); memory.cb = ctypes.sizeof(memory)
                    assert get_memory(wintypes.HANDLE(int(child._handle)), ctypes.byref(memory), ctypes.sizeof(memory)), 'terminal_native_OS_memory_query'
                    row['OS_peak_bytes'] = int(memory.peak_rss); child_peak = max(child_peak, row['OS_peak_bytes'])
                    assert root_peak + child_peak <= 16 << 30
                except BaseException:
                    if child.poll() is None: child.kill(); child.wait()
                    raise
                finally:
                    row.update(returncode=child.poll(), end_utc=utc(), wall_seconds=time.monotonic() - child_start)
                    active = None
            checkpoint(child_terminal=child.pid, returncode=child.returncode, label=label)
            row['stdout'] = {'path': str(OUT / (label + '.stdout.log')), 'sha256': digest(OUT / (label + '.stdout.log'))}
            row['stderr'] = {'path': str(OUT / (label + '.stderr.log')), 'sha256': digest(OUT / (label + '.stderr.log'))}
            if negative:
                assert child.returncode == 2 and b'worker_affinity_readback' in (OUT / (label + '.stderr.log')).read_bytes()
                assert not trace.exists() and not Path(str(prefix) + '.0.bin').exists(); return None
            assert child.returncode == 0 and (OUT / (label + '.stderr.log')).stat().st_size == 0, (label, row['returncode'])
            rows = [json.loads(line) for line in (OUT / (label + '.stdout.log')).read_text(encoding='utf-8').splitlines()]
            assert len(rows) == 1; native = rows[0]
            assert (native['repetition'], native['threads'], native['profile'], native['source_tokens']) == (0, 3, 0, len(source))
            assert native['worker_physical_cores'] == 6 and [v['slot'] for v in native['worker_affinity']] == [0, 1, 2]
            assert [v['actual_mask'] for v in native['worker_affinity']] == [1, 4, 16]
            assert all(v['group'] == 0 for v in native['worker_affinity']) and len({v['windows_thread_id'] for v in native['worker_affinity']}) == 3
            assert all(v['group'] == 0 and v['actual_mask'] == 1 << [0, 2, 4][v['slot']] for v in native['worker_binding_events'])
            assert {v['slot'] for v in native['worker_binding_events']} == {0, 1, 2}
            positions = len(decoder) if mode == 'teacher' else native['actual_generated_tokens']
            counts = native['counters'][1]
            assert sum(v['code_bytes'] for v in counts) == 123764736 * positions
            assert sum(v['scale_bytes'] for v in counts) == 534016 * positions
            assert sum(v['f32_bytes'] for v in counts) == 18432 * 128 * positions
            whole = Path(str(prefix) + '.0.bin')
            row.update(native_row=native, trace_path=str(trace), trace_sha256=digest(trace),
                       whole_output_path=str(whole), whole_output_sha256=digest(whole))
            guard(force_scan=True); return row

        dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (128,))])
        def validate(trace, whole, s, t, mode, native=None):
            raw_trace = Path(trace).read_bytes(); data = Path(whole).read_bytes()
            assert raw_trace[:8] == b'SWRTA001' and struct.unpack_from('<2I', raw_trace, 8) == (128, 768)
            total = 6 * (s + t); assert len(raw_trace) == 16 + total * dtype.itemsize
            traced = np.frombuffer(raw_trace, dtype, offset=16)
            assert np.array_equal(traced['index'], np.arange(total))
            assert np.all(traced['phase'][:6*s] == 0) and np.all(traced['phase'][6*s:] == 1)
            assert np.all(np.isfinite(traced['input'])) and np.all(np.isfinite(traced['scores']))
            er, dr = D.route_bytes(data, 128, s, t); routes = np.concatenate([er.reshape(-1), dr.T.reshape(-1)])
            assert np.all(routes['accepted'] == 1), 'capacity64_all_queries_expected_accepted_S29_and_single_token_decode'
            floats = np.frombuffer(data, '<f4', count=(len(data)-36-12*total)//4, offset=36)
            assert np.all(np.isfinite(floats)), 'all_complete_states_logits_finite'
            selected = np.argmax(traced['scores'], axis=1); assert np.array_equal(selected, routes['expert'])
            differences = traced['scores'] - traced['scores'][np.arange(total), selected, None]
            probability = (1 / np.exp(differences.astype(np.float64)).astype(np.float32).astype(np.float64).sum(axis=1)).astype(np.float32)
            error = float(np.max(np.abs(probability.astype(np.float64) / routes['probability'] - 1)))
            assert error <= 1e-6
            logits_offset = 36 + 4 * (14*s*768 + 14*t*768)
            logits = np.frombuffer(data, '<f4', t*32128, logits_offset).reshape(t, 32128)
            if mode == 'natural':
                ids = np.argmax(logits, axis=1).tolist(); assert native is not None and ids == native['generated_ids']
                assert 1 <= t <= 64 and len(ids) == native['actual_generated_tokens'] == t
                assert all(v not in (1, 32095) for v in ids[:-1])
                assert native['closing_id'] == 32095 and native['stopped_on_EOS'] == (ids[-1] == 1)
                assert native['stopped_on_closing'] == (ids[-1] == 32095)
                assert native['stopped_at_cap'] == (ids[-1] not in (1, 32095))
                assert ids[-1] in (1, 32095) or t == 64
            enc_bytes = data[36:36 + 4*14*s*768]
            first_dec = data[36 + 4*14*s*768:36 + 4*14*s*768 + 4*14*768]
            first_logits = data[logits_offset:logits_offset + 4*32128]
            guard()
            return traced, routes, raw_trace[16:16 + 6*(s+1)*dtype.itemsize], (enc_bytes, first_dec, first_logits), error

        stage = 'fresh_negative_and_eight_original_golden_native_controls'
        invoke([2, 1], [0], 'negative_worker', 'teacher', True)
        for bi in (0, 11, 12, 23):
            entry = golden[(bi, 0)]
            assert len(entry['source_ids']) == 29 and len(entry['decoder_ids']) == 14
            for mode in ('teacher', 'natural'):
                row = invoke(entry['source_ids'], entry['decoder_ids'], f'control.book{bi}.{mode}', mode)
                expected = entry['teacher_rows' if mode == 'teacher' else 'generation_rows'][0]
                assert row['whole_output_sha256'] == expected['output_sha256'] and row['trace_sha256'] == expected['router_trace_sha256']
                t = 14 if mode == 'teacher' else row['native_row']['actual_generated_tokens']
                validate(row['trace_path'], row['whole_output_path'], 29, t, mode, row['native_row'])
        first = result['commands'][1]; raw_trace = Path(first['trace_path']).read_bytes()
        for kind, offset, value in (('index', 16, struct.pack('<I', 1)), ('score', 16+8+768*4, struct.pack('<f', float('nan')))):
            invalid = bytearray(raw_trace); invalid[offset:offset+4] = value
            path = OUT / ('negative_' + kind + '.bin'); path.write_bytes(invalid)
            try: validate(path, first['whole_output_path'], 29, 14, 'teacher')
            except AssertionError: pass
            else: raise AssertionError('negative_trace_undetected_' + kind)
        invalid = bytearray(Path(first['whole_output_path']).read_bytes())
        route_offset = 36 + 4 * (14*29*768 + 14*14*768 + 14*32128)
        invalid[route_offset+4:route_offset+8] = struct.pack('<i', 2)
        bad = OUT / 'negative_accept.bin'; bad.write_bytes(invalid)
        try: validate(first['trace_path'], bad, 29, 14, 'teacher')
        except AssertionError: pass
        else: raise AssertionError('negative_accept_undetected')
        valid_ids = [32099, 7, 32098, 8, 32097, 9, 32096, 10, 32095]
        assert health(valid_ids)['healthy_complete_nonempty_fields'] and not health([0]+valid_ids)['healthy_complete_nonempty_fields']
        assert not health(valid_ids[:-1])['healthy_complete_nonempty_fields'] and not health([32099,32098,8,32097,9,32096,10,32095])['healthy_complete_nonempty_fields']
        result['gates']['fresh_worker_fault_8_full_golden_bytes_and3_trace_route_negative_controls'] = True
        checkpoint(native_controls_complete=9)

        stage = 'ALL512_teacher_natural_capture_and_domain_ledger'
        states = [[D.bucket() for _ in range(4)] for _ in range(12*128)]
        ledger_path = OUT / 'query_fingerprint_ledger.bin'
        totals = {'teacher': 0, 'natural': 0}; healthy_count = 0
        with ledger_path.open('xb') as ledger:
            ledger.write(struct.pack('<8sIQ', b'MQ468L01', LEDGER.size, 0))
            for bi, book in enumerate(cohort['items']):
                split = int(bi >= 64)
                for ci, case in enumerate(book['cases']):
                    entry = {'book': bi, 'case': ci, 'role': book['role'], 'source_ids_sha256': case['source_ids_sha256'], 'modes': {}}
                    result['cases'].append(entry); teacher_bridge = None
                    for mi, mode in enumerate(('teacher', 'natural')):
                        row = invoke(case['source_ids'], case['decoder_ids'], f'book{bi}.case{ci}.{mode}', mode)
                        entry['modes'][mode] = len(result['commands']) - 1
                        t = 14 if mode == 'teacher' else row['native_row']['actual_generated_tokens']
                        traced, routes, trace_bridge, whole_bridge, error = validate(row['trace_path'], row['whole_output_path'], 29, t, mode, row['native_row'])
                        if mode == 'teacher': teacher_bridge = (trace_bridge, whole_bridge)
                        else:
                            assert teacher_bridge == (trace_bridge, whole_bridge), 'encoder_and_first_decoder_teacher_natural_byte_identity'
                            row['natural_health'] = health(row['native_row']['generated_ids']); healthy_count += row['natural_health']['healthy_complete_nonempty_fields']
                        accepted_indices = np.flatnonzero(routes['accepted']); codes, scales = D.quantize(traced['input'][accepted_indices])
                        lookup = {int(index): local for local, index in enumerate(accepted_indices)}
                        for phase in (0, 1):
                            eligible = [int(i) for i in accepted_indices if (i < 6*29) == (phase == 0)]
                            if eligible:
                                index = eligible[0]; local = lookup[index]; sq, sa = D.scalar_quantize(traced['input'][index], D.SAMPLE_COORDS)
                                assert codes[local, list(D.SAMPLE_COORDS)].tolist() == sq and float(scales[local]) == sa
                        for index, traced_row in enumerate(traced):
                            bank = index // 29 if index < 6*29 else 6 + (index-6*29) % 6
                            expert = int(routes[index]['expert']); accepted = int(routes[index]['accepted']); state = states[bank*128+expert][2*split+mi]
                            state['selected'] += 1; state['executed'] += accepted; state['rejected'] += 1-accepted
                            hi = hashlib.sha256(traced_row['input'].tobytes()).digest(); hq = hp = bytes(32); alpha = 0.
                            if accepted:
                                local = lookup[index]; qbytes = codes[local].tobytes(); alpha = float(scales[local]); sb = struct.pack('<f', alpha)
                                hq = hashlib.sha256(qbytes).digest(); hp = hashlib.sha256(qbytes+sb).digest()
                                state['input'].add(hi); state['pair'].add(hp); state['scale'].add(sb); state['books'].add(bi); state['cases'].add((bi,ci))
                                state['probability_sum'] += float(routes[index]['probability'])
                                state['code'][hq] = state['code'].get(hq,0) | (1 << bi)
                            ledger.write(LEDGER.pack(128, mi, bi, ci, bank, accepted, split, expert, index, alpha, float(routes[index]['probability']), hi, hq, hp)); count += 1
                        totals[mode] += len(traced)
                        row['trace_checks'] = {'source_tokens':29,'positions':t,'queries':len(traced),'maximum_probability_relative_error':error,
                                               'per_phase_first_accepted_exact_rational_coordinates':True}
                        guard()
                ledger.flush(); assert ledger_path.stat().st_size <= 64 << 20
                result['preserved_daemons_last_book'] = jobs()
                checkpoint(completed_book=bi, completed_cases=len(result['cases']), ledger_queries=count, healthy_natural_cases=healthy_count)
            ledger.seek(12); ledger.write(struct.pack('<Q', count))
        assert len(result['cases']) == 512 and len(result['commands']) == 1033 and count == sum(totals.values()) <= max_queries
        stage = 'fixed_bank11_data_readiness'
        banks = []
        for bank in range(12):
            experts = []; supported = covered = val_queries = 0
            for expert in range(128):
                bs = states[bank*128+expert]; dev = D.code_union(bs[:2]); val = D.code_union(bs[2:]); novel = set(val)-set(dev)
                novel_books = 0
                for key in novel: novel_books |= val[key]
                masks = [i for i in range(128) if novel_books & (1 << i)]
                ds, vs = D.summary(bs[:2]), D.summary(bs[2:])
                gates = {'development_unique_codes_ge32':len(dev)>=32,'development_books_ge4':len(ds['books'])>=4,
                         'validation_novel_codes_ge16':len(novel)>=16,'validation_novel_code_books_ge4':len(masks)>=4}
                ready = all(gates.values()); supported += ready; val_queries += bs[3]['executed']; covered += bs[3]['executed'] if ready else 0
                experts.append({'expert':expert,'by_split_mode':{f'{split}_{mode}':D.summary([bs[2*si+mi]]) for si,split in enumerate(('dev','val')) for mi,mode in enumerate(('teacher','natural'))},
                    'dev_union':ds,'val_union':vs,'validation_code_seen_in_dev':len(set(val)&set(dev)),'validation_novel_codes':len(novel),'validation_novel_code_books':masks,
                    'cross_mode_identical_input_dev_val':[len(bs[i]['input']&bs[i+1]['input']) for i in (0,2)],
                    'cross_mode_identical_code_dev_val':[len(set(bs[i]['code'])&set(bs[i+1]['code'])) for i in (0,2)],
                    'sampled_dev_span_rank_UPPER_bound_NOT_measured_rank':min(768,len(dev)),'readiness_gates':gates,'data_ready':ready})
            coverage = covered/val_queries if val_queries else 0.
            gates = {'data_ready_distinct_IDs_ge32':supported>=32,'validation_natural_executed_query_coverage_ge0p90':coverage>=.90}
            banks.append({'bank':bank,'stack':'encoder' if bank<6 else 'decoder','layer':2*(bank%6)+1,
                          'data_ready_IDs':supported,'val_natural_executed_queries':val_queries,'covered_val_natural_executed_queries':covered,
                          'coverage':coverage,'readiness_gates':gates,'passed':all(gates.values()),'experts':experts})
            guard()
        result.update(banks=banks, route_query_totals=totals, natural_health={'healthy':healthy_count,'all_cases':512,'unhealthy':512-healthy_count,'no_health_filter':True},
                      fixed_future_probe={'source':128,'bank':11,'decoder_layer':11,'hypothetical_rank':32,'data_ready':banks[11]['passed']})
        result['ledger'] = {'path':str(ledger_path),'bytes':ledger_path.stat().st_size,'sha256':digest(ledger_path),'record_bytes':LEDGER.size,'records':count}
        assert ledger_path.stat().st_size == 20 + count*LEDGER.size
        assert [payload.stat().st_size,payload.stat().st_mtime_ns] == before
        for rel,row in bindings['preserved_unrelated_files'].items(): assert digest(ROOT/rel)==row['sha256']
        result['preserved_daemons_after'] = jobs()
        fault.flush(); assert (OUT/'fatal_native.log').stat().st_size == 0
        result['forbidden_modules_after_all_cases'] = sorted(name for name in sys.modules if name.partition('.')[0] in FORBIDDEN)
        assert not result['forbidden_modules_after_all_cases']
        result['gates'].update(ALL1024_native_streams_actual_worker_order_shapes_states_logits_route_probability_exact=True,
            ALL512_encoder_and_first_decoder_correlated_views_byte_identical_not_independent=True,
            ALL_actual_executed_A16_input_code_scale_ledger_book_role_and_novelty=True,
            fixed_source128_bank11_SAME462_data_floors_all1536_IDs_reported=True,
            every_unhealthy_natural_case_included_no_case_or_bank_selection=True,
            original_engine_payload_stat_unrelated_work_and_empty_native_fault_log_preserved=True)
        result['decision'] = ('data_admitted_for_separately_frozen_ONE_rank32_private_INPUT_probe' if banks[11]['passed'] else
                              'new_domain_insufficient_fixed_bank11_stop_before_factors_reassess_information_or_geometry')
        result['scope'] = 'Actual new expert conditional-input data admission; no rank/SVD/fit/compressed artifact, donor-relative quality, final speed, useful-n/DRAM or other-family claim. Source128 original trained functions preserved. Same467 source-only cohort counted once; dev/diagnostic-validation roles immutable. Finite unique codes bound rank only above. Encoder/first decoder teacher/natural views correlated. Native timers include instrumentation/I/O; no rate qualification. Terminal Windows APPCRASH audit still required after real completion.'
        guard(force_scan=True)
        result['resource'] = {'main_seconds_including_heavy_imports_through_admission':time.monotonic()-start,'parent_OS_peak_bytes':root_peak,
            'maximum_terminal_native_OS_peak_bytes':child_peak,'conservative_parent_plus_largest_child_peak_bytes':root_peak+child_peak,
            'maximum_sampled_parent_all_descendant_RSS_sum':aggregate,'bytes_hashed_before_raw':hashed,'output_bytes_before_raw':output_bytes,
            'hard_seconds':2100,'hard_memory_bytes':16<<30,'hard_all_output_bytes':12<<30}
        result['end_utc'] = utc(); write_new(RAW,result); guard(force_scan=True)
        checkpoint(admitted=True,data_ready=banks[11]['passed'],raw_sha256=digest(RAW),native_commands=len(result['commands']),
                   parent_OS_peak_bytes_after_raw=root_peak,maximum_terminal_native_OS_peak_bytes=child_peak,
                   conservative_parent_plus_largest_child_peak_bytes=root_peak+child_peak,
                   maximum_sampled_parent_all_descendant_RSS_sum=aggregate,all_output_bytes_including_raw=output_bytes)
        print(json.dumps({'sha256':digest(RAW),'gates':result['gates'],'decision':result['decision'],
                          'bank11':{k:banks[11][k] for k in ('data_ready_IDs','val_natural_executed_queries','covered_val_natural_executed_queries','coverage','passed')},
                          'native_commands':len(result['commands']),'ledger_records':count,'natural_health':result['natural_health'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        if active is not None and active.poll() is None: active.kill(); active.wait()
        result.update(stage=stage,error=repr(error),end_utc=utc(),main_seconds=time.monotonic()-start,ledger_queries_written=count)
        write_new(RAW.with_suffix('.failure.json'),result); raise
    finally:
        faulthandler.disable(); progress.close(); fault.close()


if __name__ == '__main__':
    main()
