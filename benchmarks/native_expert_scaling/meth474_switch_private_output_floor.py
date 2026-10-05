"""One frozen private rank32 OUTPUT floor on complete source functions, all107 IDs."""
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
OUT = ROOT / 'results/native_expert_scaling/meth474_switch_private_output_floor'
RAW = DOC / 'meth474_switch_private_output_floor_result.json'
PROTO = DOC / 'METH_474_SWITCH_PRIVATE_OUTPUT_PROTOCOL_20261005.md'
BIND = DOC / 'meth474_prospective_bindings.json'
BIND_SHA = '53999bd7023770320a77a5c7698e7c624738ac7b76ff81be2f7e8cf4236b4618'
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
    result = {'experiment': 'METH474 original128 bank11 private rank32 complete-function output-space floor',
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
        assert count <= 96 << 20, 'new96MiB'
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
        for p in (Path(__file__), BASE / 'meth474_output_math.py', PROTO):
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
            if name in ('meth472_switch_private_input_probe_result.json', 'RETENTION_472_20261005.json', 'RETENTION_473_20261005.json', 'meth380_switch_base128_export_result.json'):
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
        assert digest(b['preparation_helper']['path']) == b['preparation_helper']['sha256']
        assert digest(b['controller_text_generation_helper']['path']) == b['controller_text_generation_helper']['sha256']
        for row in b['retained_inventory']:
            p = Path(row['path'])
            assert p.stat().st_size == row['bytes'] and p.stat().st_mtime_ns == row['mtime_ns'] and digest(p) == row['sha256']
        assert len(b['retained_inventory']) == 6447
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
        result['apparatus_gates']['frozen_actual_runtime_sources_all6447_inputs_original_payload'] = True
        result['admission'] = {'seconds': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed}
        assert time.monotonic() - START <= 120
        checkpoint(file_bytes_hashed=hashed)

        stage = 'runtime_manifest_and_geometry_controls'
        import numpy as np
        import threadpoolctl as T
        import meth474_output_math as M
        assert Path(M.__file__).resolve() == (BASE / 'meth474_output_math.py').resolve()
        for module in (np, psutil, T):
            assert module.__version__ == b['runtime']['packages'][module.__name__]['version']
            assert str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
        np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
        limits = T.threadpool_limits(limits=1)
        pools = T.threadpool_info()
        assert pools and all(p['num_threads'] == 1 and p['user_api'] == 'blas' for p in pools)
        for p in pools: assert str(Path(p['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
        assert parent.cpu_affinity() == [0]
        result['runtime'] = {'BLAS': pools, 'affinity': [0], 'GPU': False,
                             'packages': {k: metadata.version(k) for k in ('numpy', 'psutil', 'threadpoolctl')}}
        export = records['meth380_switch_base128_export_result.json']
        assert export['artifact'] == artifact and all(export['gates'].values())
        manifest = Path(artifact['manifest']).read_bytes()
        assert manifest[:8] == b'SWI8A001'
        fields = ('d_model', 'd_ff', 'num_heads', 'd_kv', 'num_layers', 'num_decoder_layers', 'num_experts', 'expert_capacity', 'vocab_size', 'relative_attention_num_buckets', 'relative_attention_max_distance', 'encoder_sparse_step', 'decoder_sparse_step')
        assert struct.unpack_from('<13I', manifest, 8) == tuple(export['original_config'][k] for k in fields)
        assert manifest[60:64] == struct.pack('<f', export['original_config']['layer_norm_epsilon'])
        assert struct.unpack_from('<2I', manifest, 64) == (1, 3320)
        cursor = 72
        def text():
            nonlocal cursor
            size = struct.unpack_from('<I', manifest, cursor)[0]; cursor += 4
            value = manifest[cursor:cursor + size].decode(); cursor += size
            return value
        assert Path(text()).resolve() == payload.resolve()
        for name, e in sorted(export['tensors'].items()):
            assert text() == name
            assert struct.unpack_from('<5I3Q', manifest, cursor) == (0, len(e['shape']), e['shape'][0], e['shape'][1] if len(e['shape']) == 2 else 1, e['encoding'], e['offset'], e['scale_offset'], e['elements'])
            cursor += 44
        assert cursor == len(manifest)
        result['tiny_controls'] = M.controls(guard)
        result['rounding_scope'] = {'rank': 32, 'relative_lambda': M.rounding_constants()[0],
                                    'absolute_beta': M.rounding_constants()[1], 'numerical_distance_guard': M.NUMERICAL_DISTANCE_GUARD,
                                    'scope': 'storedF32_P, finite F64 final32term factor dot and final RNE F32 cast; beta also permits final-cast flush; coordinate map arbitrary. Numerical projection guard is not a formal interval certificate.'}
        result['apparatus_gates']['actual_singleCPU_BLAS_manifest_and_known_projection_rounding_controls'] = True

        stage = 'all107_development_private_output_bases_and_function_floors'
        old_out = ROOT / 'results/native_expert_scaling/meth472_switch_private_input_probe'
        queries = np.load(old_out / 'query_inputs.npy', mmap_mode='r', allow_pickle=False)
        functions = np.load(old_out / 'reference_functions.npy', mmap_mode='r', allow_pickle=False)
        assert queries.shape == (19962,) and queries.dtype.itemsize == 4732
        assert json.loads(json.dumps(queries.dtype.descr)) == previous['domain']['query_dtype_descr']
        assert functions.shape == (19962, 768) and functions.dtype == np.dtype('<f4') and np.isfinite(functions).all()
        assert np.all(queries['book'] < 192) and np.all(queries['accepted'] == 1)
        assert np.all(queries['role'] == np.where((queries['book'] >= 64) & (queries['book'] < 128), 1, 0))
        assert np.count_nonzero(queries['role'] == 0) == 13313
        ready = set(b['fixed_ready_IDs'])
        values = np.zeros((19962, 4), dtype='<f8')
        values[:, 0] = np.sum(functions.astype(np.float64)**2, axis=1)
        total_dev = total_val = 0
        for expert, expected_dev, expected_val in zip(b['fixed_ready_IDs'], b['development_representative_counts'], b['novel_validation_representative_counts']):
            all_ids = np.flatnonzero(queries['expert'] == expert)
            dev = all_ids[queries['role'][all_ids] == 0]
            dev_codes = {queries['code_sha'][i:i + 1].tobytes() for i in dev}
            novel = [i for i in all_ids if queries['role'][i] == 1 and queries['code_sha'][i:i + 1].tobytes() not in dev_codes]
            di, dw = M.representatives(queries, dev)
            vi, vw = M.representatives(queries, novel)
            assert len(di) == expected_dev and len(vi) == expected_val
            with np.load(old_out / f'svd_witness_e{expert:03d}.npz', allow_pickle=False) as old:
                for current, key in [(di, 'development_representatives'), (dw, 'development_weights'),
                                     (vi, 'validation_novel_representatives'), (vw, 'validation_novel_weights')]: M.exact(current, old[key])
            y = (functions[di].astype(np.float64) * np.sqrt(dw)[:, None]).T.copy()
            top, stored, orth, triangular, singular, vt, stats = M.basis(y, guard)
            p = OUT / f'output_witness_e{expert:03d}.npz'
            parts = {'U64_top32': top, 'P32': stored, 'Q64': orth, 'R64': triangular, 'singular_values': singular, 'Vt': vt,
                     'development_representatives': di, 'development_weights': dw,
                     'validation_novel_representatives': vi, 'validation_novel_weights': vw}
            np.savez(p, **parts)
            with np.load(p, allow_pickle=False) as saved:
                for key, value in parts.items(): M.exact(value, saved[key])
            for first in range(0, len(all_ids), 64):
                ids = all_ids[first:first + 64]
                values[ids] = M.metrics(functions[ids], orth)
                guard()
            novel_summary = M.summary(values[vi], .05, vw)
            result['experts'].append({'expert': expert, 'development_representatives': len(di),
                                      'novel_validation_representatives': len(vi), 'development_books': np.unique(queries['book'][di]).tolist(),
                                      'spectrum': stats, 'novel_validation_function_floor': novel_summary,
                                      'witness': str(p), 'witness_sha256': digest(p)})
            total_dev += len(di); total_val += len(vi)
            checkpoint(expert=expert, novel_floor_status=novel_summary['status'], lower_RMS=novel_summary['lower_RMS'])
            del y, top, stored, orth, triangular, singular, vt, parts
        assert total_dev == 11331 and total_val == sum(b['novel_validation_representative_counts']) and len(result['experts']) == 107
        fallback = np.flatnonzero(~np.isin(queries['expert'], list(ready)))
        assert np.all(values[fallback, 1:] == 0)
        np.save(OUT / 'per_query_floor_metrics.npy', values, allow_pickle=False)
        M.exact(values, np.load(OUT / 'per_query_floor_metrics.npy', allow_pickle=False))
        result['domain'] = {'total_queries': 19962, 'development_queries': 13313, 'development_representatives': total_dev,
                            'novel_validation_representatives': total_val, 'ready_IDs': b['fixed_ready_IDs'],
                            'fallback_IDs': sorted(set(range(128)) - ready), 'metric_columns': ['reference_energy', 'ideal_projection_error_energy', 'conservative_lower_error_energy', 'constructive_upper_error_energy']}
        result['apparatus_gates']['all107_exact_development_novelty_book_weights_fixed107_21_policy'] = True
        result['apparatus_gates']['all107_full_output_spectra_stored_F32_spans_QR_and_witnesses_qualified'] = True
        result['apparatus_gates']['all19962_function_energy_lower_upper_metrics_saved_reloaded_exact'] = True
        natural_ids = np.flatnonzero((queries['role'] == 1) & (queries['mode'] == 1))
        assert len(natural_ids) == 3065
        result['natural_validation_function_floor'] = M.summary(values[natural_ids], .05)
        result['natural_validation_books'] = []
        for book in range(64, 128):
            ids = natural_ids[queries['book'][natural_ids] == book]
            result['natural_validation_books'].append({'book': book, **M.summary(values[ids], .10)})
        states = [e['novel_validation_function_floor']['status'] for e in result['experts']] + [result['natural_validation_function_floor']['status']] + [e['status'] for e in result['natural_validation_books']]
        verdict = 'FAIL' if 'FAIL' in states else ('PASS' if all(s == 'PASS' for s in states) else 'INDETERMINATE')
        result['nominal_storage'] = {'complete_bank_bytes': M.NOMINAL_BANK_BYTES, 'original_bank_bytes': M.ORIGINAL_BANK_BYTES,
                                    'le70percent': 10 * M.NOMINAL_BANK_BYTES <= 7 * M.ORIGINAL_BANK_BYTES, 'actual_bank_exported': False}
        result['decision'] = {'fixed_rank32_private_output_space_floor': verdict,
                              'failed_novel_expert_IDs': [e['expert'] for e in result['experts'] if e['novel_validation_function_floor']['status'] == 'FAIL'],
                              'indeterminate_novel_expert_IDs': [e['expert'] for e in result['experts'] if e['novel_validation_function_floor']['status'] == 'INDETERMINATE'],
                              'next': 'separately_freeze_actual_WO_coefficients_full_function_candidate' if verdict == 'PASS' else ('close_this_learned_private_rank32_output_span' if verdict == 'FAIL' else 'retain_numerical_indeterminacy_no_candidate_or_rank_grid')}
        result['preserved_daemons_after'] = jobs()
        assert [payload.stat().st_size, payload.stat().st_mtime_ns] == initial
        for rel, row in b['preserved_unrelated_files'].items(): assert digest(ROOT / rel) == row['sha256']
        assert digest(engine) == b['original_engine']['sha256']
        assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines() == b['tracked_status']
        assert not FORBIDDEN.intersection(sys.modules) and fatal.tell() == 0
        result['apparatus_gates']['original_engine_policy_resources_terminal_module_scope_preserved'] = True
        assert len(result['apparatus_gates']) == 6 and all(result['apparatus_gates'].values())
        result['scope'] = 'Optimistic complete source-function range screen for THIS learned storedF32 private rank32 span. No actual coordinates/WO coefficients/bank/Cmodel/native/head/generation/task/rate/DRAM/useful-n result. Numerical projection guards are not formal interval certificates.'
        result['end_utc'] = utc()
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']
        result['resource'] = {'seconds_before_raw_write': time.monotonic() - START, 'OS_peak_bytes_before_raw_write': peak,
                               'file_bytes_hashed_before_raw_write': hashed, 'hard_seconds': 180, 'hard_peak_bytes': 2 << 30,
                               'hard_new_bytes': 96 << 20, 'output_upper_bytes': b['output_upper_bytes']}
        write_new(RAW, result); guard()
        raw_sha = digest(RAW)
        checkpoint(terminal=True, raw_sha256=raw_sha, raw_bytes=RAW.stat().st_size, OS_peak_bytes_after_raw=peak,
                   file_bytes_hashed_after_raw=hashed, floor_decision=verdict)
        print(json.dumps({'raw': str(RAW), 'raw_sha256': raw_sha, 'decision': result['decision'],
                          'natural_floor': result['natural_validation_function_floor']}))
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
