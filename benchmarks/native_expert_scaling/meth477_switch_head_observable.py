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
OUT = ROOT / 'results/native_expert_scaling/meth477_switch_head_observable'
RAW = DOC / 'meth477_switch_head_observable_result.json'
PROTO = DOC / 'METH_477_SWITCH_HEAD_OBSERVABLE_PROTOCOL_20261005.md'
BIND = DOC / 'meth477_prospective_bindings.json'
BIND_SHA = '630af85674071dd453595dc5d1a9c810072c0328f9edfaa4987f5ca1ebe742f8'
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
    result = {'experiment': 'METH477 all6649 saved472 candidate finite head-observable diagnostic',
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
        assert time.monotonic() - START <= 180, 'hard180s'
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
        assert count <= 24 << 20, 'new24MiB'
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
        for p in (Path(__file__), BASE / 'meth477_switch_head_observable.c', BASE / 'meth477_observable_math.py', PROTO):
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
        assert len(b['retained_inventory']) == 6578
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
        result['apparatus_gates']['frozen_actual_runtime_sources_all6578_inputs_original_payload'] = True
        result['admission'] = {'seconds': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed}
        assert time.monotonic() - START <= 90
        checkpoint(file_bytes_hashed=hashed)

        stage = 'immutable_labels_query_ready_novel_flags_complete_job_admission'
        import base64
        import shutil
        import meth477_observable_math as M
        assert Path(M.__file__).resolve() == (BASE / 'meth477_observable_math.py').resolve()
        cohort = json.loads((DOC / 'meth467_switch_rust_query_manifest.json').read_bytes())
        labels = {}
        for book in range(64, 128):
            item = cohort['items'][book]; assert item['book'] == book and item['role'] == 'diagnostic_validation'
            raw_ids = base64.b64decode(item['excerpt_token_ids_base64_le_i32'], validate=True)
            assert hashlib.sha256(raw_ids).hexdigest() == item['excerpt_token_ids_sha256']
            ids = struct.unpack('<' + 'i' * (len(raw_ids)//4), raw_ids)
            for case, c in enumerate(item['cases']):
                assert c['index'] == case and c['span_starts'] == [3, 10, 17, 24]
                original = list(ids[c['excerpt_token_start']:c['excerpt_token_start']+32])
                assert original == c['original_window_ids']
                source_ids, target_ids, prior = [], [], 0
                for k, start in enumerate((3, 10, 17, 24)):
                    span = original[start:start+2]; assert span == c['masked_spans_ids'][k]
                    source_ids.extend(original[prior:start]); source_ids.append(32099-k); prior = start+2
                    target_ids.extend([32099-k, *span])
                source_ids.extend(original[prior:]); source_ids.append(1); target_ids.extend([32095, 1])
                assert source_ids == c['source_ids'] and target_ids == c['target_ids'] and c['decoder_ids'] == [0] + target_ids[:-1]
                assert hashlib.sha256(struct.pack('<14i', *target_ids)).hexdigest() == c['target_ids_sha256']
                assert hashlib.sha256(struct.pack('<29i', *source_ids)).hexdigest() == c['source_ids_sha256']
                labels[(book, case)] = target_ids
        result['label_admission'] = {'cases': 256, 'teacher_labels': 3584, 'masked_content_labels': 2048,
                                     'natural_true_labels': 0, 'source_record': 'meth467_switch_rust_query_manifest.json'}
        qpath = ROOT / 'results/native_expert_scaling/meth472_switch_private_input_probe/query_inputs.npy'
        fpath, cpath = qpath.with_name('reference_functions.npy'), qpath.with_name('candidate_functions.npy')
        def npy_header(path):
            import ast
            with Path(path).open('rb') as stream:
                assert stream.read(8) == b'\x93NUMPY\x01\x00'; size = struct.unpack('<H', stream.read(2))[0]
                desc = ast.literal_eval(stream.read(size).decode('ascii')); return stream.tell(), desc
        qo, qdesc = npy_header(qpath); fo, fdesc = npy_header(fpath); co, cdesc = npy_header(cpath)
        assert json.loads(json.dumps([qo, qdesc])) == b['query_npy_header']
        assert json.loads(json.dumps([fo, fdesc])) == b['reference_npy_header'] and json.loads(json.dumps([co, cdesc])) == b['candidate_npy_header']
        assert fdesc == cdesc == {'descr': '<f4', 'fortran_order': False, 'shape': (19962, 768)}
        qdata = qpath.read_bytes(); qwire = struct.Struct('<QH4BHIff3072s1536s32s32s32s'); assert qwire.size == 4732
        assert len(qdata) == qo + 19962 * 4732
        dev_codes = {expert: set() for expert in range(128)}
        for qid in range(19962):
            v = qwire.unpack_from(qdata, qo + qid*4732)
            if v[4] == 0: dev_codes[v[6]].add(v[13])
        ready = set(b['fixed_ready_IDs']); assert len(ready) == 107
        jobs_path = OUT / 'jobs.bin'; all_val = []; query_flags = {}; context_indices = {}
        context_cursor = teacher_count = natural_count = masked_count = 0
        with jobs_path.open('xb') as stream:
            stream.write(struct.pack('<8s3I3Q', b'M477JOB1', 608, 7993, 6649, qo, fo, co))
            for index, item in enumerate(b['last_layer_batches']):
                assert item['batch'] == index
                kind, book, case, mode, s, t = (item[k] for k in ('kind', 'book', 'case', 'mode', 's', 't'))
                stream.write(struct.pack('<6I3Q', kind, book, case, mode, s, t, item['query_start'], context_cursor, item['ledger_base']))
                for key in ('whole_path', 'control_path'):
                    encoded = item[key].encode(); assert len(encoded) < 1024
                    stream.write(struct.pack('<I', len(encoded)) + encoded)
                if kind and mode == 0: assert item['decoder_ids'] == [0] + labels[(book, case)][:-1]
                for pos in range(t):
                    label, flags = (1 << 32)-1, 0
                    if kind:
                        qid = item['query_start'] + pos; v = qwire.unpack_from(qdata, qo + qid*4732)
                        assert (v[1],v[2],v[3],v[4],v[5],v[7],v[0]) == (book,case,mode,1,1,179+6*pos,item['ledger_base']+179+6*pos)
                        assert v[12] == hashlib.sha256(v[10]).digest() and v[13] == hashlib.sha256(v[11]).digest() and v[14] == hashlib.sha256(v[11]+struct.pack('<f',v[8])).digest()
                        flags = int(v[6] in ready) | (2 if v[13] not in dev_codes[v[6]] else 0) | (16 if v[6] in (25,37) else 0)
                        if mode == 0:
                            flags |= 4; label = labels[(book,case)][pos]; teacher_count += 1
                            if pos in (1,2,4,5,7,8,10,11): flags |= 8; masked_count += 1
                        else: natural_count += 1
                        query_flags[qid] = flags; context_indices[qid] = context_cursor + pos; all_val.append(qid)
                    stream.write(struct.pack('<2I', label, flags))
                context_cursor += t; guard()
        assert context_cursor == 7993 and len(all_val) == 6649 and (teacher_count,natural_count,masked_count) == (3584,3065,2048)
        assert all_val == [qid for qid in range(19962) if qdata[qo+qid*4732+12] == 1]
        result['query_admission'] = {'count': 6649, 'teacher': 3584, 'natural': 3065,
                                    'all_original_query_indices_sha256': hashlib.sha256(struct.pack('<6649Q', *all_val)).hexdigest(),
                                    'ready_IDs': sorted(ready), 'novelty_against_ALL13313_development_codes': True}
        result['apparatus_gates']['ALL6649_original_queries_roles_ready_novel_masks_and_true_teacher_label_provenance'] = True
        del qdata, dev_codes
        stage = 'ONE_frozen_compile_source_primal_and_complete_candidate_observable'
        compiler = next(Path(p) for p in b['native_files'] if Path(p).name == 'clang.exe')
        dll = next(Path(p) for p in b['native_files'] if Path(p).name == 'libomp.dll')
        shutil.copyfile(dll, OUT / 'libomp.dll'); binary = OUT / 'meth477_switch_head_observable.exe'
        argv = [str(compiler), '-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off', '-fopenmp',
                str(BASE / 'meth477_switch_head_observable.c'), '-o', str(binary), '-lbcrypt']
        active_child = None; commands = []
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
        invoke(argv, 'compile', dict(os.environ), 60)
        result['compile'] = {'argv': argv, 'binary_sha256': digest(binary), 'runtime_sha256': digest(OUT / 'libomp.dll')}
        env = {k:v for k,v in os.environ.items() if not k.startswith(('OMP_','KMP_','GOMP_','SILICON_'))}
        env.update(b['native_runtime_environment']); env['OMP_NUM_THREADS'] = '3'
        controls = [str(binary),'--controls',str(ROOT / 'results/native_expert_scaling/meth356_switch_all_a16_contract/integer_cases.bin'),str(OUT / 'head-int-dot.bin')]
        result['native_or_model_commands'] = 1
        invoke(controls,'controls',env,30)
        control_rows = [json.loads(line) for line in (OUT / 'controls.stdout.log').read_text().splitlines()]
        assert len(control_rows) == 8
        for k,toy in enumerate(M.TOYS):
            assert control_rows[k]['control'] == k and len(control_rows[k]['metrics']) == 16
            for actual,expected in zip(control_rows[k]['metrics'],M.decimal_oracle(*toy)): M.close(actual,expected)
        assert control_rows[6] == {'SHA_controls':[hashlib.sha256(b'').hexdigest(),hashlib.sha256(b'abc').hexdigest()]}
        assert control_rows[-1]['control_terminal'] and control_rows[-1]['integer_fixtures'] == 13
        assert control_rows[-1]['worker_physical_cores'] == 6 and [r['actual_mask'] for r in control_rows[-1]['worker_affinity']] == [1,4,16]
        assert (OUT / 'head-int-dot.bin').read_bytes() == (ROOT / 'results/native_expert_scaling/meth374_switch_physical_workers_contract/head-int-dot.bin').read_bytes()
        result['apparatus_gates']['actual3_workers_integer13_and6high_precision_softmax_margin_shift_controls_and_SHA'] = True
        obs = OUT / 'observables.bin'; context_path = ROOT / 'results/native_expert_scaling/meth476_switch_last_layer_context/context.bin'
        native = [str(binary), artifact['manifest'], str(jobs_path), str(qpath), str(fpath), str(cpath), str(context_path), str(obs)]
        result['native_or_model_commands'] = 2
        invoke(native, 'native', env, 90)
        result['native_runtime_environment'] = {key: env[key] for key in (*b['native_runtime_environment'], 'OMP_NUM_THREADS')}
        rows = [json.loads(line) for line in (OUT / 'native.stdout.log').read_text().splitlines()]
        assert len(rows) == 609
        source_count = candidate_count = 0
        for item, row in zip(b['last_layer_batches'], rows[:-1]):
            source_count += item['t']; candidate_count += item['t'] if item['kind'] else 0
            assert row == {'batch': item['batch'], 'source_positions': source_count, 'candidate_positions': candidate_count}
        terminal = rows[-1]
        assert terminal['terminal'] and terminal['source_positions'] == 7993 and terminal['candidate_positions'] == 6649
        assert terminal['teacher_labels'] == 3584 and terminal['masked_content_labels'] == 2048 and terminal['head_rows'] == 470422176
        assert terminal['worker_physical_cores'] == 6 and [r['actual_mask'] for r in terminal['worker_affinity']] == [1,4,16]
        assert all(r['actual_mask'] == [1,4,16][r['slot']] and r['group'] == 0 for r in terminal['worker_binding_events'])
        assert (OUT / 'head-int-dot.bin').read_bytes() == (ROOT / 'results/native_expert_scaling/meth374_switch_physical_workers_contract/head-int-dot.bin').read_bytes()
        result['apparatus_gates']['ALL7993_original_source_context_finalnorm_head_codes_ALL32128_logits_exact'] = True
        assert obs.stat().st_size == 16 + 6649*1748
        records = []; base_struct = struct.Struct('<QIHBBHH7I16d'); assert base_struct.size == 176
        with obs.open('rb') as stream:
            assert stream.read(16) == struct.pack('<8sII',b'M477OBS1',6649,1748)
            for k, qid in enumerate(all_val):
                packet = stream.read(1748); assert len(packet) == 1748
                v = base_struct.unpack_from(packet); meta = dict(zip(('qid','context_row','book','case','mode','expert','flags','position','argmax_source','argmax_candidate','label','source_ties','candidate_ties','reserved'),v[:14]))
                assert meta['qid'] == qid and meta['context_row'] == context_indices[qid] and meta['reserved'] == 0 and meta['flags'] & 31 == query_flags[qid]
                assert meta['source_ties'] >= 1 and meta['candidate_ties'] >= 1
                record = {**meta, **dict(zip(M.FIELDS,v[14:]))}
                assert all(__import__('math').isfinite(record[f]) for f in M.FIELDS)
                if not meta['flags'] & 4: assert record['nll_source'] == record['nll_candidate'] == record['delta_nll'] == 0
                records.append(record)
            assert not stream.read(1)
        assert terminal['argmax_changed'] == sum(bool(r['flags'] & 128) for r in records)
        assert terminal['fallback'] == sum(not bool(r['flags'] & 1) for r in records)
        result['native_terminal'] = terminal
        result['apparatus_gates']['ALL6649_candidate_full_vocabulary_metrics_head_witnesses_denominators_and_fallback_identity'] = True
        result['summaries'] = M.all_summaries(records)
        result['decision'] = 'DESCRIPTIVE_OBSERVABLE_DIAGNOSTIC_COMPLETE_pending_independent_audit_NO_PROMOTION'
        result['scope'] = 'One saved472 candidate at fixed original prefixes. No fitting, threshold selected from results, original472gate changes, candidate whole generation, global quality, actual speed, DRAM, causal useful-n or other family claim.'
        assert [payload.stat().st_size,payload.stat().st_mtime_ns] == initial and parent.cpu_affinity() == [0]
        result['preserved_daemons_after'] = jobs(); assert not (FORBIDDEN & set(sys.modules))
        result['apparatus_gates']['complete_all_cohorts_immutable_inputs_no_fitting_budget_and_scope'] = True
        result['end_utc'] = utc()
        result['output_inventory'] = [{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']
        result['resource'] = {'seconds_before_raw_write':time.monotonic()-START,'OS_peak_bytes_before_raw_write':peak,'file_bytes_hashed_before_raw_write':hashed,
                              'hard_seconds':180,'hard_peak_bytes':2<<30,'hard_new_bytes':24<<20,'output_upper_bytes':b['output_upper_bytes']}
        write_new(RAW,result);guard();raw_sha=digest(RAW)
        checkpoint(terminal=True,raw_sha256=raw_sha,raw_bytes=RAW.stat().st_size,OS_peak_bytes_after_raw=peak,file_bytes_hashed_after_raw=hashed)
        print(json.dumps({'raw':str(RAW),'raw_sha256':raw_sha,'decision':result['decision'],'all':result['summaries']['views']['all'],'natural':result['summaries']['views']['natural']}))
    except BaseException as exc:
        result['failure']={'stage':stage,'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};result['end_utc']=utc()
        result['resource_on_failure']={'seconds':time.monotonic()-START,'OS_peak_bytes':peak,'file_bytes_hashed':hashed}
        failure=RAW.with_suffix('.failure.json')
        if not failure.exists():write_new(failure,result)
        checkpoint(terminal_failure=True,failure_path=str(failure));raise
    finally:
        progress.close();faulthandler.disable();fatal.close()

if __name__ == '__main__':
    main()
