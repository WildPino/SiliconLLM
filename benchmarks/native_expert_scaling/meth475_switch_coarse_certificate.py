"""One lossless coarse/fine integer certificate admission; all cached1344 queries."""
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
OUT = ROOT / 'results/native_expert_scaling/meth475_switch_coarse_certificate'
RAW = DOC / 'meth475_switch_coarse_certificate_result.json'
PROTO = DOC / 'METH_475_SWITCH_COARSE_CERTIFICATE_PROTOCOL_20261005.md'
BIND = DOC / 'meth475_prospective_bindings.json'
BIND_SHA = 'c80c00c80f5baf4b57e643ef14520d83ab635076fff3749cfd023298b9f10c03'
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
    result = {'experiment': 'METH475 original128 exact signed nibble planes and integer negative-ReLU certificate',
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
        assert count <= 32 << 20, 'new32MiB'
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
        for p in (Path(__file__), BASE / 'meth475_coarse_math.py', PROTO):
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
        assert digest(b['preparation_helper']['path']) == b['preparation_helper']['sha256']
        assert digest(b['controller_text_generation_helper']['path']) == b['controller_text_generation_helper']['sha256']
        for row in b['retained_inventory']:
            p = Path(row['path'])
            assert p.stat().st_size == row['bytes'] and p.stat().st_mtime_ns == row['mtime_ns'] and digest(p) == row['sha256']
        assert len(b['retained_inventory']) == 6557
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
        result['apparatus_gates']['frozen_actual_runtime_sources_all6557_inputs_original_payload'] = True
        result['admission'] = {'seconds': time.monotonic() - START, 'OS_peak_bytes': peak, 'file_bytes_hashed': hashed}
        assert time.monotonic() - START <= 120
        checkpoint(file_bytes_hashed=hashed)

        stage = 'runtime_manifest_and_integer_controls'
        import numpy as np
        import threadpoolctl as T
        import meth475_coarse_math as M
        assert Path(M.__file__).resolve() == (BASE / 'meth475_coarse_math.py').resolve()
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
        stage = 'source_integer_fixtures_and_cached_native_controls'
        fixture = ROOT / 'results/native_expert_scaling/meth356_switch_all_a16_contract'
        data = (fixture / 'integer_cases.bin').read_bytes()
        answers = (ROOT / 'results/native_expert_scaling/meth374_switch_physical_workers_contract/head-int-dot.bin').read_bytes()
        assert data[:12] == struct.pack('<8sI', b'SWI8D001', 13) and answers[:12] == struct.pack('<8sI', b'SW16R001', 13)
        a = c = 12
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
            dots = M.integer_dot(q, w); M.exact(dots[0], dots0)
            M.exact(M.scaled(dots, alpha, si)[0], expected)
        assert a == len(data) and c == len(answers)
        control_type = np.dtype([('position', '<u4'), ('id', '<u4'), ('source_tokens', '<u4'), ('route_index', '<u4'),
            ('pre', '<f4', (768,)), ('input', '<f4', (768,)), ('wi_scale', '<f4'), ('wi_codes', '<i2', (768,)),
            ('up_raw', '<f4', (3072,)), ('up', '<f4', (3072,)), ('wo_scale', '<f4'), ('wo_codes', '<i2', (3072,)),
            ('down', '<f4', (768,)), ('expert', '<i4'), ('accepted', '<i4'), ('probability', '<f4'),
            ('post', '<f4', (768,)), ('final', '<f4', (768,)), ('head_input', '<f4', (768,)), ('head_scale', '<f4'), ('head_codes', '<i2', (768,))])
        meta_type = np.dtype([('capture', '<u2'), ('book', '<u2'), ('case', 'u1'), ('position', 'u1'),
                              ('expert', '<u2'), ('decoder_id', '<u4'), ('source_tokens', '<u4')])
        assert control_type.itemsize == 52264 and meta_type.itemsize == 16
        controls, metadata_rows = [], np.empty(1344, meta_type)
        assert len(b['native_FFN_controls']) == 96
        for index, item in enumerate(b['native_FFN_controls']):
            p = Path(item['path']); d = p.read_bytes()
            assert d[:32] == struct.pack('<8s6I', b'SWFUN001', 768, 3072, 32128, 128, 11, 1) and len(d) == 731728
            selected = np.frombuffer(d, control_type, offset=32).copy()
            assert len(selected) == 14 and np.all(selected['accepted'] == 1)
            assert selected['position'].tolist() == list(range(14))
            assert selected['id'].tolist() == item['decoder_ids'] and np.all(selected['source_tokens'] == len(item['source_ids']))
            label = item['label'].split('.')
            assert label[:2] == ['teacher', 'n128']
            book, case = int(label[2][4:]), int(label[3][4:])
            assert 0 <= book < 24 and 0 <= case < 4
            controls.append(selected)
            for pos in range(14):
                metadata_rows[index * 14 + pos] = (index, book, case, pos, int(selected['expert'][pos]),
                                                   int(selected['id'][pos]), len(item['source_ids']))
        controls = np.concatenate(controls)
        assert len(controls) == 1344 and np.all((controls['expert'] >= 0) & (controls['expert'] < 128))
        assert set(metadata_rows['book'].tolist()) == set(range(24))
        assert all(np.count_nonzero(metadata_rows['book'] == book) == 56 for book in range(24))
        result['apparatus_gates']['actual_singleCPU_BLAS_manifest_tiny_integer13_native_fixture_controls'] = True

        stage = 'all128_lossless_planes_all1344_complete_native_primal_certificate'
        coarse = np.empty((1344, 3072), '<i4')
        bits = np.empty((1344, 384), 'u1')
        norms = np.empty((128, 3072), '<u4')
        qnorm = np.empty(1344, '<u8')
        hidden_codes = np.empty((1344, 3072), '<i2')
        hidden_scales = np.empty(1344, '<f4')
        down = np.empty((1344, 768), '<f4')
        counts = np.empty(1344, '<u4')
        nonzero = np.empty(1344, '<u4')
        def tensor(expert, kind):
            name = f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.{kind}.weight'
            t = export['tensors'][name]
            shape = [3072, 768] if kind == 'wi' else [768, 3072]
            assert t['encoding'] == 1 and t['shape'] == shape and t['bytes'] == 2359296
            with payload.open('rb') as stream:
                stream.seek(t['offset']); wb = stream.read(t['bytes'])
                stream.seek(t['scale_offset']); sb = stream.read(t['scale_bytes'])
            assert hashlib.sha256(wb).hexdigest() == t['sha256'] and hashlib.sha256(sb).hexdigest() == t['scale_sha256']
            w = np.frombuffer(wb, '<i1').reshape(shape)
            s = np.frombuffer(sb, '<f4')
            assert len(s) == shape[0] and np.isfinite(s).all() and np.all(s > 0)
            return w, s
        for expert in range(128):
            wi, si = tensor(expert, 'wi'); wo, so = tensor(expert, 'wo')
            hp, bp, r2 = M.planes(wi); norms[expert] = r2
            ids = np.flatnonzero(controls['expert'] == expert)
            summary = {'expert': expert, 'query_count': len(ids),
                       'coarse_packed_sha256': hashlib.sha256(hp.tobytes()).hexdigest(),
                       'fine_packed_sha256': hashlib.sha256(bp.tobytes()).hexdigest(),
                       'residual_norm_sha256': hashlib.sha256(r2.tobytes()).hexdigest()}
            if len(ids):
                selected = controls[ids]
                q, alpha = M.quant(selected['input'])
                M.exact(q, selected['wi_codes']); M.exact(alpha, selected['wi_scale'])
                original_dots = M.integer_dot(q, wi)
                original_raw = M.scaled(original_dots, alpha, si)
                M.exact(original_raw, selected['up_raw'])
                original_up = np.where(original_raw < 0, np.float32(0), original_raw).astype('<f4')
                M.exact(original_up, selected['up'])
                h0, a0 = M.quant(original_up)
                M.exact(h0, selected['wo_codes']); M.exact(a0, selected['wo_scale'])
                source_down = M.scaled(M.integer_dot(h0, wo), a0, so)
                M.exact(source_down, selected['down'])
                t, q2, cert, raw, h1, a1 = M.candidate(q, alpha, hp, bp, r2, si, guard)
                assert np.all(original_dots[cert] < 0) and np.all(original_raw[cert] <= 0)
                M.exact(raw[~cert], original_raw[~cert])
                M.exact(h1, h0); M.exact(a1, a0)
                candidate_down, nz = M.sparse_down(h1, a1, wo, so, guard)
                M.exact(candidate_down, source_down)
                coarse[ids] = t; qnorm[ids] = q2; bits[ids] = np.packbits(cert, axis=1, bitorder='little')
                hidden_codes[ids] = h1; hidden_scales[ids] = a1; down[ids] = candidate_down
                counts[ids] = np.sum(cert, axis=1, dtype=np.uint32); nonzero[ids] = np.asarray(nz, '<u4')
                summary['certified_rows'] = int(np.sum(cert))
                summary['candidate_nonzero_WO_codes'] = int(sum(nz))
                summary['source_negative_or_zero_rows'] = int(np.count_nonzero(original_raw <= 0))
            result['experts'].append(summary)
            checkpoint(expert=expert, query_count=len(ids), certified_rows=summary.get('certified_rows', 0))
            guard()
        assert sum(e['query_count'] for e in result['experts']) == 1344
        result['apparatus_gates']['all128_complete_I8_bijective_packed_plane_inverse_and_norms'] = True
        result['apparatus_gates']['all1344_original_WI_WO_quantizers_and_candidate_hidden_codes_scales_complete_down_byte_exact'] = True
        parts = {'coarse_div8': coarse, 'certificate_bits': bits, 'residual_norms': norms, 'query_norms': qnorm,
                 'query_metadata': metadata_rows, 'candidate_hidden_codes': hidden_codes,
                 'candidate_hidden_scales': hidden_scales, 'candidate_down': down}
        for name, value in parts.items():
            np.save(OUT / (name + '.npy'), value, allow_pickle=False)
            M.exact(value, np.load(OUT / (name + '.npy'), allow_pickle=False)); guard()
        result['apparatus_gates']['all1344_exact_margin_witness_masks_complete_outputs_saved_reloaded_exact'] = True
        perquery = []
        for i in range(1344):
            cert = np.unpackbits(bits[i], bitorder='little').astype(bool)
            assert len(cert) == 3072 and np.count_nonzero(cert) == counts[i]
            margin = 64 * coarse[i].astype(np.int64)**2 - qnorm[i].astype(np.int64) * norms[int(metadata_rows['expert'][i])].astype(np.int64)
            assert np.array_equal(cert, (coarse[i] < 0) & (margin > 0))
            fine = (3072 - int(counts[i])) * 384
            perquery.append({'query': i, 'book': int(metadata_rows['book'][i]), 'expert': int(metadata_rows['expert'][i]),
                             'certified_rows': int(counts[i]), 'nonzero_hidden_codes': int(nonzero[i]),
                             'coarse_WI_coefficient_bytes': 1179648, 'fine_WI_coefficient_bytes': fine,
                             'WI_norm_bytes': 12288, 'WI_scale_bytes': 12288,
                             'planned64aligned_WI_coefficient_lines': 18432 + (3072 - int(counts[i])) * 6,
                             'planned64aligned_WI_norm_scale_lines': 384,
                             'planned64aligned_WO_column_lines': int(nonzero[i]) * 12,
                             'min_certified_integer_margin': int(np.min(margin[cert])) if np.any(cert) else None})
        total_rows = 1344 * 3072
        certified = int(np.sum(counts, dtype=np.uint64))
        books = []
        for book in range(24):
            ids = np.flatnonzero(metadata_rows['book'] == book)
            k = int(np.sum(counts[ids], dtype=np.uint64)); denominator = len(ids) * 3072
            books.append({'book': book, 'queries': len(ids), 'certified_rows': k, 'total_rows': denominator,
                          'certified_fraction': k / denominator, 'ge0p40': 10 * k >= 4 * denominator})
        result['per_query'] = perquery
        result['domain'] = {'captures': 96, 'queries': 1344, 'books': 24, 'source': 'all cached original128 teacher bank11 native positions; consumed evidence, not fresh quality',
                            'query_metadata_dtype_descr': metadata_rows.dtype.descr, 'all128_codec_IDs': True,
                            'observed_IDs': sorted(set(metadata_rows['expert'].tolist())), 'no_ID_or_input_fit_selection': True}
        result['certificate_screen'] = {'certified_rows': certified, 'total_rows': total_rows, 'certified_fraction': certified / total_rows,
                                        'mean_WI_coefficient_read_fraction': 1 - certified / (2 * total_rows), 'books': books,
                                        'planned_header_bytes': 4096, 'planned_per_expert_stride': 4746240,
                                        'planned_complete_bank_bytes': 4096 + 128 * 4746240,
                                        'original_bank_coefficients_scales_bytes': 605945856,
                                        'actual_bank_exported': False, 'physical_DRAM_bytes_measured': False}
        result['recipe_gates'] = {'mean_certified_rows_ge0p60': 10 * certified >= 6 * total_rows,
                                  'every_book_certified_rows_ge0p40': all(v['ge0p40'] for v in books)}
        verdict = 'PASS' if all(result['recipe_gates'].values()) else 'FAIL'
        result['decision'] = {'coarse_fine_integer_certificate_admission': verdict,
                              'next': 'separately_freeze_one_complete_C_cost_primal' if verdict == 'PASS' else 'close_this_fixed_nibble_center_Cauchy_certificate_no_C_or_grid'}
        result['preserved_daemons_after'] = jobs()
        assert [payload.stat().st_size, payload.stat().st_mtime_ns] == initial
        for rel, row in b['preserved_unrelated_files'].items(): assert digest(ROOT / rel) == row['sha256']
        assert digest(engine) == b['original_engine']['sha256']
        assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines() == b['tracked_status']
        assert not FORBIDDEN.intersection(sys.modules) and fatal.tell() == 0
        result['apparatus_gates']['original_engine_policy_resources_terminal_module_scope_preserved'] = True
        assert len(result['apparatus_gates']) == 6 and all(result['apparatus_gates'].values())
        result['scope'] = 'Exact cached-original complete-FFN primal plus ONE fixed lossless bitplane/sign-certificate logical-read screen. No new bank/Cmodel/native/head/fresh generation/task/rate/physicalDRAM/LUT/useful-n result. Signedzero intermediate raw/up need not match; hidden A16 codes/scales and complete down must be byte exact. All integer margins retained losslessly through T=C/8, residual row norms and query norms. Python unpacked arithmetic is not physical packed execution.'
        result['end_utc'] = utc()
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'progress.jsonl']
        result['resource'] = {'seconds_before_raw_write': time.monotonic() - START, 'OS_peak_bytes_before_raw_write': peak,
                               'file_bytes_hashed_before_raw_write': hashed, 'hard_seconds': 180, 'hard_peak_bytes': 2 << 30,
                               'hard_new_bytes': 32 << 20, 'output_upper_bytes': b['output_upper_bytes']}
        write_new(RAW, result); guard()
        raw_sha = digest(RAW)
        checkpoint(terminal=True, raw_sha256=raw_sha, raw_bytes=RAW.stat().st_size, OS_peak_bytes_after_raw=peak,
                   file_bytes_hashed_after_raw=hashed, certificate_decision=verdict)
        print(json.dumps({'raw': str(RAW), 'raw_sha256': raw_sha, 'decision': result['decision'],
                          'certified_fraction': result['certificate_screen']['certified_fraction']}))
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
