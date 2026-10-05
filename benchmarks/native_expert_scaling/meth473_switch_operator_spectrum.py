"""One frozen development-only decoded-WI feasibility inquiry, all107 IDs."""
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
OUT = ROOT / 'results/native_expert_scaling/meth473_switch_operator_spectrum'
RAW = DOC / 'meth473_switch_operator_spectrum_result.json'
PROTO = DOC / 'METH_473_SWITCH_OPERATOR_SPECTRUM_PROTOCOL_20261005.md'
BIND = DOC / 'meth473_prospective_bindings.json'
BIND_SHA = 'd2e8fe7566ebbfd8d6bcc70f2cc62729f189ba403d520b5a50863b12ebf05476'
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
    result = {'experiment': 'METH473 original128 bank11 learned-WI full spectrum feasibility',
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
            assert peak <= 4 << 30, 'hard4GiB'
            if time.monotonic() - last_job_scan >= 5:
                jobs()
                last_job_scan = time.monotonic()
        count = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
        for p in (RAW, RAW.with_suffix('.failure.json')):
            if p.exists(): count += p.stat().st_size
        assert count <= 128 << 20, 'new128MiB'
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
        for p in (Path(__file__), BASE / 'meth473_operator_math.py', PROTO):
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
            if name in ('meth472_switch_private_input_probe_result.json', 'RETENTION_472_20261005.json', 'meth380_switch_base128_export_result.json'):
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
        for row in b['retained_inventory']:
            p = Path(row['path'])
            assert p.stat().st_size == row['bytes'] and p.stat().st_mtime_ns == row['mtime_ns'] and digest(p) == row['sha256']
        assert len(b['retained_inventory']) == 6338
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
        retained = records['RETENTION_472_20261005.json']
        assert retained['raw_sha256'] == digest(DOC / 'meth472_switch_private_input_probe_result.json')
        assert len(previous['apparatus_gates']) == 8 and all(previous['apparatus_gates'].values())
        assert len(retained['gates']) == 9 and all(retained['gates'].values())
        assert previous['domain']['ready_IDs'] == b['fixed_ready_IDs'] and len(b['fixed_ready_IDs']) == 107
        result['apparatus_gates']['frozen_runtime_sources_records_all6338_input_bytes_original_payload'] = True
        result['admission'] = {'seconds': time.monotonic() - START, 'file_bytes_hashed': hashed, 'OS_peak_bytes': peak}
        assert time.monotonic() - START <= 300
        checkpoint(file_bytes_hashed=hashed)

        stage = 'actual_runtime_manifest_and_native_order'
        import numpy as np
        import threadpoolctl as T
        import meth473_operator_math as M
        assert Path(M.__file__).resolve() == (BASE / 'meth473_operator_math.py').resolve()
        for module in (np, psutil, T):
            assert str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
            assert module.__version__ == b['runtime']['packages'][module.__name__]['version']
        np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
        limits = T.threadpool_limits(limits=1)
        pools = T.threadpool_info()
        assert pools and all(p['num_threads'] == 1 and p['user_api'] == 'blas' for p in pools)
        for p in pools:
            assert str(Path(p['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
        result['runtime'] = {'BLAS': pools, 'affinity': parent.cpu_affinity(), 'GPU': False,
                             'packages': {k: metadata.version(k) for k in ('numpy', 'psutil', 'threadpoolctl')}}
        assert parent.cpu_affinity() == [0]
        export = records['meth380_switch_base128_export_result.json']
        assert export['artifact'] == artifact and all(export['gates'].values())
        entries = export['tensors']
        manifest = Path(artifact['manifest']).read_bytes()
        assert manifest[:8] == b'SWI8A001'
        config = struct.unpack_from('<13IfII', manifest, 8)
        fields = ('d_model', 'd_ff', 'num_heads', 'd_kv', 'num_layers', 'num_decoder_layers', 'num_experts', 'expert_capacity', 'vocab_size', 'relative_attention_num_buckets', 'relative_attention_max_distance', 'encoder_sparse_step', 'decoder_sparse_step')
        assert config[:13] == tuple(export['original_config'][k] for k in fields)
        assert config[13] == np.float32(export['original_config']['layer_norm_epsilon']) and config[14:] == (1, 3320)
        cursor = 72
        def text():
            nonlocal cursor
            size = struct.unpack_from('<I', manifest, cursor)[0]
            cursor += 4
            value = manifest[cursor:cursor + size].decode()
            cursor += size
            return value
        assert Path(text()).resolve() == payload.resolve()
        for name, e in sorted(entries.items()):
            assert text() == name
            assert struct.unpack_from('<5I3Q', manifest, cursor) == (0, len(e['shape']), e['shape'][0], e['shape'][1] if len(e['shape']) == 2 else 1, e['encoding'], e['offset'], e['scale_offset'], e['elements'])
            cursor += 44
        assert cursor == len(manifest)
        def wi(expert):
            e = entries[f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.wi.weight']
            assert e['shape'] == [3072, 768] and e['encoding'] == 1 and e['bytes'] == 2359296 and e['scale_bytes'] == 12288
            with payload.open('rb') as stream:
                stream.seek(e['offset']); codes = stream.read(e['bytes'])
                stream.seek(e['scale_offset']); scales = stream.read(e['scale_bytes'])
            assert hashlib.sha256(codes).hexdigest() == e['sha256'] and hashlib.sha256(scales).hexdigest() == e['scale_sha256']
            w, s = np.frombuffer(codes, '<i1').reshape(3072, 768), np.frombuffer(scales, '<f4')
            assert np.isfinite(s).all() and np.all(s >= 0)
            return w, s
        result['tiny_controls'] = M.controls(guard)
        a = c = 12
        fixtures = {Path(p).name: Path(p).read_bytes() for p in b['integer_fixtures']}
        data, answers = fixtures['integer_cases.bin'], fixtures['head-int-dot.bin']
        assert data[:12] == struct.pack('<8sI', b'SWI8D001', 13) and answers[:12] == struct.pack('<8sI', b'SW16R001', 13)
        for _ in range(13):
            rows, cols = struct.unpack_from('<2I', data, a); a += 8
            w = np.frombuffer(data, '<i1', rows * cols, a).reshape(rows, cols); a += rows * cols
            si = np.frombuffer(data, '<f4', rows, a); a += rows * 4
            x = np.frombuffer(data, '<f4', cols, a).reshape(1, cols); a += cols * 4
            assert struct.unpack_from('<2I', answers, c) == (rows, cols); c += 8
            expected = np.frombuffer(answers, '<f4', rows, c); c += rows * 4
            q0 = np.frombuffer(answers, '<i2', cols, c); c += cols * 2
            alpha0 = np.frombuffer(answers, '<f4', 1, c); c += 4
            dots0 = np.frombuffer(answers, '<i8', rows, c); c += rows * 8
            q, alpha = M.quant(x); M.exact(q[0], q0); M.exact(alpha, alpha0)
            assert np.array_equal(q.astype(np.float64) @ w.astype(np.float64).T, dots0.astype(np.float64)[None, :])
            M.exact(M.response(q, alpha, w, si)[1][0], expected)
        assert a == len(data) and c == len(answers)
        control_type = np.dtype([('position', '<u4'), ('id', '<u4'), ('source_tokens', '<u4'), ('route_index', '<u4'),
            ('pre', '<f4', (768,)), ('input', '<f4', (768,)), ('wi_scale', '<f4'), ('wi_codes', '<i2', (768,)),
            ('up_raw', '<f4', (3072,)), ('up', '<f4', (3072,)), ('wo_scale', '<f4'), ('wo_codes', '<i2', (3072,)),
            ('down', '<f4', (768,)), ('expert', '<i4'), ('accepted', '<i4'), ('probability', '<f4'),
            ('post', '<f4', (768,)), ('final', '<f4', (768,)), ('head_input', '<f4', (768,)), ('head_scale', '<f4'), ('head_codes', '<i2', (768,))])
        controls = []
        for row in b['native_FFN_controls']:
            d = Path(row['path']).read_bytes()
            assert d[:32] == struct.pack('<8s6I', b'SWFUN001', 768, 3072, 32128, 128, 11, 1)
            assert len(d) == 731728
            controls.append(np.frombuffer(d, control_type, offset=32).copy())
        controls = np.concatenate(controls)
        assert len(controls) == 1344
        for expert in range(128):
            selected = controls[controls['expert'] == expert]
            if not len(selected): continue
            w, si = wi(expert)
            q, alpha = M.quant(selected['input'])
            M.exact(q, selected['wi_codes']); M.exact(alpha, selected['wi_scale'])
            M.exact(M.response(q, alpha, w, si)[1], selected['up_raw'])
            guard()
        del controls
        result['apparatus_gates']['actual_singleCPU_BLAS_manifest13_native_integer1344_WI_controls'] = True

        stage = 'all107_development_operator_spectra'
        old_out = ROOT / 'results/native_expert_scaling/meth472_switch_private_input_probe'
        queries = np.load(old_out / 'query_inputs.npy', mmap_mode='r', allow_pickle=False)
        assert queries.shape == (19962,) and queries.dtype.itemsize == 4732
        assert json.loads(json.dumps(queries.dtype.descr)) == previous['domain']['query_dtype_descr']
        assert np.count_nonzero(queries['role'] == 0) == 13313
        assert np.all(queries['accepted'] == 1) and np.all(queries['book'] < 192)
        assert np.all(queries['role'] == np.where((queries['book'] >= 64) & (queries['book'] < 128), 1, 0))
        total = 0
        for expert, count in zip(b['fixed_ready_IDs'], b['development_representative_counts']):
            ids, weights = M.representatives(queries, expert)
            assert len(ids) == count and count <= 308
            with np.load(old_out / f'svd_witness_e{expert:03d}.npz', allow_pickle=False) as old:
                M.exact(ids, old['development_representatives']); M.exact(weights, old['development_weights'])
            selected = queries[ids]
            q, alpha = M.quant(selected['input'])
            M.exact(q, selected['codes']); M.exact(alpha, selected['alpha'])
            w, si = wi(expert)
            decoded, _, rounding = M.response(q, alpha, w, si)
            y = (decoded * np.sqrt(weights)[:, None]).T.copy()
            singular, vt, tail, stats = M.spectrum(y, guard)
            assert len(singular) == count
            p = OUT / f'operator_witness_e{expert:03d}.npz'
            np.savez(p, singular_values=singular, Vt=vt, development_representatives=ids,
                     development_weights=weights, tail_relative_squared=tail)
            with np.load(p, allow_pickle=False) as saved:
                for key, value in [('singular_values', singular), ('Vt', vt), ('development_representatives', ids),
                                   ('development_weights', weights), ('tail_relative_squared', tail)]:
                    M.exact(value, saved[key])
            result['experts'].append({'expert': expert, 'development_representatives': count,
                                      'development_books': np.unique(selected['book']).tolist(),
                                      'spectrum': stats, 'rounding_qualifier': rounding,
                                      'witness': str(p), 'witness_sha256': digest(p)})
            total += count
            checkpoint(expert=expert, required_rank_lower=stats['required_rank_lower'], required_rank_upper=stats['required_rank_upper'])
            assert not FORBIDDEN.intersection(sys.modules)
            guard()
            del decoded, y, singular, vt, tail, w, si, selected
        assert total == 11331 and len(result['experts']) == 107
        result['apparatus_gates']['all107_exact_development_roles_book_weights_dedup_response_envelopes'] = True
        result['apparatus_gates']['all107_full_spectra_reconstruction_eigen_orthogonality_energy_witnesses'] = True
        ranks = {name: sum(e['spectrum']['required_rank_' + name] for e in result['experts']) for name in ('lower', 'nominal', 'upper')}
        result['budget'] = {'base_bytes': M.BASE_BYTES, 'bytes_per_rank': M.RANK_BYTES,
                             'original_bank_bytes': M.ORIGINAL_BYTES, 'total_rank_cap': M.TOTAL_RANK_CAP,
                             'sum_required_ranks': ranks, 'ideal_nominal_rank_bank_bytes': M.BASE_BYTES + M.RANK_BYTES * ranks['nominal'],
                             'numerical_guard_band_squared_relative': M.DECISION_MARGIN}
        all32 = all(e['spectrum']['rank32_status'] == 'PASS' for e in result['experts'])
        variable = 'INFEASIBLE' if ranks['lower'] > M.TOTAL_RANK_CAP else ('IDEAL_BUDGET_FEASIBLE' if ranks['upper'] <= M.TOTAL_RANK_CAP else 'INDETERMINATE')
        result['decision'] = {'ALL107_rank32': all32, 'uniform43_failed_IDs': [e['expert'] for e in result['experts'] if e['spectrum']['rank43_status'] == 'FAIL'],
                              'variable_rank_ideal_development_budget': variable,
                              'next': 'separately_freeze_rank32_full_function' if all32 else ('close_this_preactivation_preserving_F32_factor_family' if variable == 'INFEASIBLE' else 'review_all_required_ranks_and_whole_format_economics_before_candidate')}
        result['preserved_daemons_after'] = jobs()
        assert [payload.stat().st_size, payload.stat().st_mtime_ns] == initial
        for rel, row in b['preserved_unrelated_files'].items(): assert digest(ROOT / rel) == row['sha256']
        assert digest(engine) == b['original_engine']['sha256']
        assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines() == b['tracked_status']
        assert not FORBIDDEN.intersection(sys.modules) and fatal.tell() == 0
        result['apparatus_gates']['storage_policy_resources_original_bytes_terminal_module_scope'] = True
        assert len(result['apparatus_gates']) == 5 and all(result['apparatus_gates'].values())
        result['scope'] = 'ONE numerical ideal-real development preactivation feasibility screen, all107 fixed IDs; no factors/candidate/validation fit/function/head/model/generation/task/rate/DRAM/useful-n claim. Numerical guard band is not a statistical interval or formal interval-arithmetic certificate.'
        result['end_utc'] = utc()
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']
        result['resource'] = {'seconds_before_raw_write': time.monotonic() - START, 'OS_peak_bytes_before_raw_write': peak,
                               'file_bytes_hashed_before_raw_write': hashed, 'hard_seconds': 600, 'hard_peak_bytes': 4 << 30,
                               'hard_new_bytes': 128 << 20, 'witness_upper_bytes': b['witness_upper_bytes']}
        write_new(RAW, result)
        guard()
        raw_sha = digest(RAW)
        checkpoint(terminal=True, raw_sha256=raw_sha, raw_bytes=RAW.stat().st_size,
                   OS_peak_bytes_after_raw=peak, file_bytes_hashed_after_raw=hashed,
                   variable_rank_decision=variable)
        print(json.dumps({'raw': str(RAW), 'raw_sha256': raw_sha, 'decision': result['decision'], 'ranks': ranks}))
    except BaseException as exc:
        result['failure'] = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        result['end_utc'] = utc()
        result['resource_on_failure'] = {'seconds': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed}
        failure = RAW.with_suffix('.failure.json')
        if not failure.exists(): write_new(failure, result)
        checkpoint(terminal_failure=True, failure_path=str(failure))
        raise
    finally:
        progress.close()
        faulthandler.disable()
        fatal.close()


if __name__ == '__main__':
    main()
