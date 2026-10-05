"""Complete original source-prefix last-layer context reconstruction."""
import time
START = time.monotonic()
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth476_switch_last_layer_context'
RAW = DOC / 'meth476_switch_last_layer_context_result.json'
PROTO = DOC / 'METH_476_SWITCH_LAST_LAYER_CONTEXT_PROTOCOL_20261005.md'
BIND = DOC / 'meth476_prospective_bindings.json'
BIND_SHA = '2d3d5b225a0126c293ed716c99cc33125e36f8c7e64b628bf9b9c02db4991dc2'
FORBIDDEN = {'torch', 'transformers', 'tensorflow', 'sklearn', 'pandas', 'pyarrow', 'tokenizers', 'scipy'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def committed(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    assert Path(path).read_bytes() == subprocess.check_output([
        'git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel], cwd=ROOT), rel


def write_new(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode()
    assert len(data) <= 4 << 20
    with Path(path).open('xb') as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert os.name == 'nt' and sys.flags.optimize == 0
    assert args.out.resolve() == RAW.resolve()
    assert not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and not OUT.exists()
    stage = 'prospective_byte_admission'
    peak = hashed = 0
    last_job_scan = -10.
    psutil = None
    active_child = None
    result = {'experiment': 'METH476 all6649 validation original128 partial last-layer context reconstruction',
              'start_utc': utc(), 'argv': sys.argv.copy(), 'source_binding_sha256': BIND_SHA,
              'apparatus_gates': {}, 'experts': [], 'native_or_model_commands': 0, 'updates': 0}
    OUT.mkdir()
    progress = (OUT / 'progress.jsonl').open('x', encoding='utf-8')
    fatal = (OUT / 'fatal_native.log').open('xb')
    faulthandler.enable(file=fatal, all_threads=True)
    def checkpoint(**values):
        progress.write(json.dumps({'stage': stage, 'utc': utc(), 'pid': os.getpid(),
                                   'seconds': time.monotonic() - START, **values}) + '\n')
        progress.flush()
    def guard():
        nonlocal peak, last_job_scan
        assert time.monotonic() - START <= 600, 'hard600s'
        if psutil is not None:
            info = psutil.Process().memory_info()
            peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
            assert peak <= 2 << 30, 'hard2GiB'
            if time.monotonic() - last_job_scan >= 5:
                jobs()
                last_job_scan = time.monotonic()
        count = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
        for p in (RAW, RAW.with_suffix('.failure.json')):
            if p.exists(): count += p.stat().st_size
        assert count <= 64 << 20, 'new64MiB'
    def digest(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while data := stream.read(4 << 20):
                h.update(data)
                hashed += len(data)
                guard()
        return h.hexdigest()
    def jobs():
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        if active_child is not None:
            own.add(active_child)
            try: own.update(p.pid for p in psutil.Process(active_child).children(recursive=True))
            except psutil.NoSuchProcess: pass
        daemons = []
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.pid in own: continue
            try:
                name, argv = (p.info['name'] or '').lower(), p.info['cmdline'] or []
                if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                    daemons.append(p.pid)
                    continue
                assert not (name.startswith('python') or name == 'clang.exe' or (name.startswith('meth') and name.endswith('.exe'))), (p.pid, name)
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        return daemons
    checkpoint()
    try:
        assert digest(BIND) == BIND_SHA
        committed(BIND)
        b = json.loads(BIND.read_bytes())
        assert sys.version == b['runtime']['python'] and Path(sys.executable).resolve() == Path(b['runtime']['executable']).resolve()
        for path, row in b['runtime']['files'].items():
            assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
        for package, item in b['runtime']['packages'].items():
            assert metadata.version(package) == item['version']
            for path, row in item['files'].items():
                assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
        import psutil as ps
        psutil = ps
        parent = psutil.Process()
        parent.cpu_affinity([0])
        result['main_process_instance'] = {'pid': os.getpid(), 'create_time_unix': parent.create_time(),
                                           'name': parent.name(), 'executable': parent.exe()}
        result['preserved_daemons_before'] = jobs()
        result['head_at_execution'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        result['scientific_sources'] = {}
        for p in (Path(__file__), BASE / 'meth476_switch_last_layer_context.c', PROTO):
            committed(p)
            result['scientific_sources'][str(p)] = digest(p)
        assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'], cwd=ROOT)
        for rel, row in b['helpers'].items():
            committed(ROOT / rel)
            assert digest(ROOT / rel) == row['sha256']
        records = {}
        for name, row in b['records'].items():
            p = DOC / name
            committed(p)
            assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256']
            if name in ('meth472_switch_private_input_probe_result.json', 'RETENTION_472_20261005.json', 'RETENTION_473_20261005.json', 'RETENTION_474_20261005.json', 'meth380_switch_base128_export_result.json'):
                records[name] = json.loads(p.read_bytes())
        for rel, row in b['preserved_unrelated_files'].items():
            assert digest(ROOT / rel) == row['sha256']
        assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines() == b['tracked_status']
        engine = ROOT / 'benchmarks/phase60/engine.c'
        committed(engine)
        assert digest(engine) == b['original_engine']['sha256']
        result['preserved_engine_sha256'] = b['original_engine']['sha256']
        for path, row in b['native_files'].items():
            assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256']
        for row in b['additional_source_files']:
            assert Path(row['path']).stat().st_size == row['bytes'] and digest(row['path']) == row['sha256']
        assert digest(b['preparation_helper']['path']) == b['preparation_helper']['sha256']
        assert digest(b['controller_text_generation_helper']['path']) == b['controller_text_generation_helper']['sha256']
        for row in b['retained_inventory']:
            p = Path(row['path'])
            assert p.stat().st_size == row['bytes'] and p.stat().st_mtime_ns == row['mtime_ns'] and digest(p) == row['sha256']
        assert len(b['retained_inventory']) == 6567
        for row in b['native_FFN_controls']:
            assert Path(row['path']).stat().st_size == row['bytes'] and digest(row['path']) == row['sha256']
        for path, row in b['integer_fixtures'].items():
            assert digest(path) == row['sha256']
        artifact = b['original_artifact']
        payload = Path(artifact['payload'])
        initial = [payload.stat().st_size, payload.stat().st_mtime_ns]
        assert initial == [b['payload_stat_before']['bytes'], b['payload_stat_before']['mtime_ns']]
        assert digest(payload) == artifact['sha256'] and digest(artifact['manifest']) == artifact['manifest_sha256']
        result['artifact'], result['payload_stat_before'] = artifact, initial
        previous = records['meth472_switch_private_input_probe_result.json']
        assert len(previous['apparatus_gates']) == 8 and all(previous['apparatus_gates'].values())
        assert len(records['RETENTION_472_20261005.json']['gates']) == 9 and all(records['RETENTION_472_20261005.json']['gates'].values())
        assert len(records['RETENTION_473_20261005.json']['gates']) == 6 and all(records['RETENTION_473_20261005.json']['gates'].values())
        assert previous['domain']['ready_IDs'] == b['fixed_ready_IDs'] and len(b['fixed_ready_IDs']) == 107
        assert len(records['RETENTION_474_20261005.json']['gates']) == 6 and all(records['RETENTION_474_20261005.json']['gates'].values())
        for item in b['predecessor_preparation_helpers']:
            assert digest(item['path']) == item['sha256']
        result['apparatus_gates']['frozen_actual_runtime_sources_all6567_inputs_original_payload'] = True
        result['admission'] = {'seconds': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed}
        assert time.monotonic() - START <= 150
        checkpoint(file_bytes_hashed=hashed)

        stage = 'ALL6649_static_query_ledger_native_prefix_joins'
        import ast
        import shutil
        for row in b['additional_source_files']:
            p = Path(row['path'])
            assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256']
        capture = json.loads((DOC / 'meth471_switch_development_capture_result.json').read_bytes())
        ledger = Path(capture['ledger']['path']).read_bytes()
        assert ledger[:20] == struct.pack('<8sIQ', b'MQ471L01', 118, 387036)
        def npy_header(path):
            with Path(path).open('rb') as stream:
                assert stream.read(8) == b'\x93NUMPY\x01\x00'
                size = struct.unpack('<H', stream.read(2))[0]
                description = ast.literal_eval(stream.read(size).decode('ascii'))
                return stream.tell(), description
        qpath = ROOT / 'results/native_expert_scaling/meth472_switch_private_input_probe/query_inputs.npy'
        fpath = qpath.with_name('reference_functions.npy')
        qoffset, qdesc = npy_header(qpath)
        foffset, fdesc = npy_header(fpath)
        assert json.loads(json.dumps([qoffset, qdesc])) == b['query_npy_header'] and json.loads(json.dumps([foffset, fdesc])) == b['reference_npy_header']
        assert qdesc['shape'] == (19962,) and not qdesc['fortran_order']
        assert fdesc == {'descr': '<f4', 'fortran_order': False, 'shape': (19962, 768)}
        qdata = qpath.read_bytes()
        assert len(qdata) == qoffset + 19962 * 4732
        wire = struct.Struct('<H6BHIff32s32s32s')
        qwire = struct.Struct('<QH4BHIff3072s1536s32s32s32s')
        assert qwire.size == 4732 and wire.size == 118
        admitted, by_mode, by_book = [], [0, 0], {book: 0 for book in range(64, 128)}
        jobs_path = OUT / 'jobs.bin'
        with jobs_path.open('xb') as stream:
            stream.write(struct.pack('<8s4I2Q', b'M476JOB1', 608, 7993, 4732, 4628, qoffset, foffset))
            for index, item in enumerate(b['last_layer_batches']):
                assert index == item['batch']
                kind, book, case, mode, s, t = (item[k] for k in ('kind', 'book', 'case', 'mode', 's', 't'))
                qs, lb = item['query_start'], item['ledger_base']
                stream.write(struct.pack('<6I2Q', kind, book, case, mode, s, t, qs, lb))
                for k in ('whole_path', 'trace_path', 'control_path'):
                    encoded = item[k].encode('utf-8'); assert 0 < len(encoded) < 1024
                    stream.write(struct.pack('<I', len(encoded)) + encoded)
                stream.write(struct.pack('<' + 'I' * s, *item['source_ids']))
                stream.write(struct.pack('<' + 'I' * t, *item['decoder_ids']))
                with Path(item['whole_path']).open('rb') as src:
                    assert src.read(36) == struct.pack('<8s7I', b'SWR32O01', s, t, 768, 12, 12, 32128, 6 * (s + t))
                with Path(item['trace_path']).open('rb') as src:
                    assert src.read(16) == struct.pack('<8sII', b'SWRTA001', 128, 768)
                if kind:
                    assert 64 <= book < 128 and s == 29 and 1 <= t <= 64
                    for pos in range(t):
                        qid = qs + pos; v = qwire.unpack_from(qdata, qoffset + qid * 4732)
                        li, qb, qc, qm, role, accepted, expert, ix, alpha, prob, inp, codes, hi, hq, hp = v
                        assert (qb, qc, qm, role, accepted, ix, li) == (book, case, mode, 1, 1, 179 + 6 * pos, lb + 179 + 6 * pos)
                        sb = struct.pack('<f', alpha)
                        assert hi == hashlib.sha256(inp).digest() and hq == hashlib.sha256(codes).digest() and hp == hashlib.sha256(codes + sb).digest()
                        expected = wire.pack(128, mode, book, case, 11, 1, 1, expert, ix, alpha, prob, hi, hq, hp)
                        assert ledger[20 + li * 118:20 + (li + 1) * 118] == expected
                        assert all(-32767 <= c <= 32767 for c in struct.unpack('<768h', codes))
                        admitted.append(qid); by_mode[mode] += 1; by_book[book] += 1
                guard()
        all_val = [i for i in range(19962) if qdata[qoffset + i * 4732 + 12] == 1]
        assert admitted == all_val and len(admitted) == 6649 and by_mode == [3584, 3065]
        assert all(v > 0 for v in by_book.values()) and len(set(admitted)) == 6649
        result['query_admission'] = {'count': 6649, 'teacher': 3584, 'natural': 3065,
                                   'books': by_book, 'all_original_query_indices_sha256': hashlib.sha256(struct.pack('<6649Q', *admitted)).hexdigest()}
        result['apparatus_gates']['ALL6649_immutable_validation_queries_ledger_prefixes_no_selection'] = True
        del qdata, ledger, all_val, admitted
        stage = 'ONE_frozen_compile_and_ONE_partial_last_layer_source_primal'
        compiler = next(Path(p) for p in b['native_files'] if Path(p).name == 'clang.exe')
        dll = next(Path(p) for p in b['native_files'] if Path(p).name == 'libomp.dll')
        shutil.copyfile(dll, OUT / 'libomp.dll')
        binary = OUT / 'meth476_switch_last_layer_context.exe'
        argv = [str(compiler), '-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off', '-fopenmp',
                str(BASE / 'meth476_switch_last_layer_context.c'), '-o', str(binary)]
        active_child = None
        commands = []
        def invoke(argv, label, env, deadline):
            nonlocal active_child, peak
            entry = {'argv': argv, 'label': label, 'start_utc': utc()}
            commands.append(entry)
            start = time.monotonic()
            stdout, stderr = OUT / (label + '.stdout.log'), OUT / (label + '.stderr.log')
            with stdout.open('xb') as so, stderr.open('xb') as se:
                child = subprocess.Popen(argv, stdout=so, stderr=se, env=env)
                active_child = child.pid
                try:
                    proc = psutil.Process(child.pid)
                    entry['process_instance'] = {'pid': child.pid, 'create_time_unix': proc.create_time(), 'executable': proc.exe()}
                    child_peak = 0
                    descendant_peaks = {}
                    while child.poll() is None:
                        guard()
                        assert time.monotonic() - start <= deadline, label + '_deadline'
                        try:
                            info = proc.memory_info()
                            for dp in proc.children(recursive=True):
                                try:
                                    di = dp.memory_info(); key = (dp.pid, dp.create_time())
                                    descendant_peaks[key] = max(descendant_peaks.get(key, 0), di.rss, getattr(di, 'peak_wset', 0))
                                except psutil.NoSuchProcess: pass
                            child_peak = max(child_peak, info.rss + sum(descendant_peaks.values()), getattr(info, 'peak_wset', 0) + sum(descendant_peaks.values()))
                            own_info = parent.memory_info()
                            total = child_peak + max(own_info.rss, getattr(own_info, 'peak_wset', 0))
                            peak = max(peak, total); assert total <= 2 << 30, 'conservative_parent_child_peak2GiB'
                        except psutil.NoSuchProcess: pass
                        time.sleep(.1)
                except BaseException:
                    if child.poll() is None: child.kill(); child.wait()
                    entry['returncode'] = child.returncode
                    raise
                finally:
                    active_child = None
            entry.update(returncode=child.returncode, end_utc=utc(), wall_seconds=time.monotonic() - start,
                         OS_child_peak_bytes=child_peak, descendant_process_peaks=[{'pid': k[0], 'create_time_unix': k[1], 'peak_bytes': v} for k, v in descendant_peaks.items()], stdout={'path': str(stdout), 'sha256': digest(stdout)},
                         stderr={'path': str(stderr), 'sha256': digest(stderr)})
            assert child.returncode == 0 and stderr.stat().st_size == 0, entry
            return entry
        result['commands'] = commands
        invoke(argv, 'compile', dict(os.environ), 120)
        result['compile'] = {'argv': argv, 'binary_sha256': digest(binary), 'runtime_sha256': digest(OUT / 'libomp.dll')}
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_'))}
        env.update(b['native_runtime_environment']); env['OMP_NUM_THREADS'] = '3'
        context = OUT / 'context.bin'
        native = [str(binary), artifact['manifest'], str(jobs_path), str(qpath), str(fpath), str(context),
                  str(ROOT / 'results/native_expert_scaling/meth356_switch_all_a16_contract/integer_cases.bin'), str(OUT / 'head-int-dot.bin')]
        result['native_or_model_commands'] = 1
        invoke(native, 'native', env, 420)
        result['native_runtime_environment'] = {k: env[k] for k in (*b['native_runtime_environment'], 'OMP_NUM_THREADS')}
        rows = [json.loads(line) for line in (OUT / 'native.stdout.log').read_text().splitlines()]
        assert len(rows) == 609
        cumulative = 0
        for item, row in zip(b['last_layer_batches'], rows[:-1]):
            cumulative += item['t']
            assert row == {'batch': item['batch'], 'kind': item['kind'], 'book': item['book'], 'case': item['case'],
                           'mode': item['mode'], 'positions': item['t'], 'cumulative_positions': cumulative, 'byte_exact': True}
        terminal = rows[-1]
        assert terminal['terminal'] and terminal['positions'] == 7993 and terminal['goldens'] == 1344 and terminal['validation'] == 6649
        assert terminal['checked_logits'] == 7993 * 32128 and terminal['worker_physical_cores'] == 6
        assert [r['actual_mask'] for r in terminal['worker_affinity']] == [1, 4, 16]
        assert [r['slot'] for r in terminal['worker_affinity']] == [0, 1, 2]
        assert all(r['actual_mask'] == [1, 4, 16][r['slot']] and r['group'] == 0 for r in terminal['worker_binding_events'])
        assert context.stat().st_size == 16 + 7993 * 4628
        with context.open('rb') as stream: assert stream.read(16) == struct.pack('<8sII', b'M476CTX1', 7993, 4628)
        answers = ROOT / 'results/native_expert_scaling/meth374_switch_physical_workers_contract/head-int-dot.bin'
        assert (OUT / 'head-int-dot.bin').read_bytes() == answers.read_bytes()
        result['apparatus_gates']['same_source_primitives_compiler_flags_actual3_workers_integer13_BYTE_exact'] = True
        result['apparatus_gates']['ALL1344_known_true_pre_complete_FFN_norm_codes_down_head_and_logits_BYTE_exact'] = True
        result['apparatus_gates']['ALL6649_source_pre_FFN_input_scores_winner_mass_complete_function_post_final_ALL_logits_BYTE_exact'] = True
        result['native_terminal'] = terminal
        assert [payload.stat().st_size, payload.stat().st_mtime_ns] == initial
        assert parent.cpu_affinity() == [0]
        result['preserved_daemons_after'] = jobs()
        assert not (FORBIDDEN & set(sys.modules))
        result['apparatus_gates']['complete_output_counts_bound_budget_inputs_no_candidate_or_whole_recapture'] = True
        result['decision'] = 'CONTEXT_ADMITTED_pending_independent_retention_audit'
        result['scope'] = 'Source-prefix conditional last-layer reconstruction only. All validation472 prefixes retained; no candidate forward, changed states, fitting, global quality, speed, DRAM, useful-n or model promotion.'
        result['end_utc'] = utc()
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']
        result['resource'] = {'seconds_before_raw_write': time.monotonic() - START, 'OS_peak_bytes_before_raw_write': peak,
                              'file_bytes_hashed_before_raw_write': hashed, 'hard_seconds': 600, 'hard_peak_bytes': 2 << 30,
                              'hard_new_bytes': 64 << 20, 'output_upper_bytes': b['output_upper_bytes']}
        write_new(RAW, result); guard()
        raw_sha = digest(RAW)
        checkpoint(terminal=True, raw_sha256=raw_sha, raw_bytes=RAW.stat().st_size, OS_peak_bytes_after_raw=peak, file_bytes_hashed_after_raw=hashed)
        print(json.dumps({'raw': str(RAW), 'raw_sha256': raw_sha, 'decision': result['decision']}))
    except BaseException as exc:
        result['failure'] = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        result['end_utc'] = utc()
        result['resource_on_failure'] = {'seconds': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed}
        failure = RAW.with_suffix('.failure.json')
        if not failure.exists(): write_new(failure, result)
        checkpoint(terminal_failure=True, failure_path=str(failure))
        raise
    finally:
        progress.close(); faulthandler.disable(); fatal.close()


if __name__ == '__main__':
    main()
