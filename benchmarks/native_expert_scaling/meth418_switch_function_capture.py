"""Qualify immutable final-bank observations; no fitting or new model artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth359_switch_cost_topology as A
import meth368_switch_bank_manifest as B
import meth393_switch_router_analysis as R
import meth401_pretrained_bank_union_applicability as U
import meth405_switch_full_basis_inputs as P

BASE = M.ROOT / 'benchmarks/native_expert_scaling'
ENGINE = M.ROOT / 'benchmarks/phase60/engine.c'
OUT = M.ROOT / 'results/native_expert_scaling/meth418_switch_function_capture'
PROTOCOL = M.DOC / 'METH_418_SWITCH_FUNCTION_CAPTURE_PROTOCOL_20261004.md'
INPUTS = {
    374: ('meth374_switch_physical_workers_contract_result.json', '4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'),
    393: ('meth393_switch_router_audit_result.json', 'bea3d2e1db10f03d5d9de3173d71a455142166ae71229bbb847959baf12ce07d'),
    405: ('meth405_switch_full_basis_inputs_result.json', '1fb5c222807da2ad8f8e52e8f3d2d5fc64664431edc2332070305fa46cb64788'),
    326: ('meth326_switch_acquisition_result.json', '39bac2bda18cc6660da77bba8256d1e21fcad5b6fa70796d343b7e139d480711'),
    378: ('meth378_switch_base128_acquisition_result.json', '89add8d24541423dfa61827f785bf125c2062cf1aa6976214f03da94b7c582b7'),
}
DTYPE = np.dtype([
    ('position', '<u4'), ('id', '<u4'), ('source_tokens', '<u4'), ('route_index', '<u4'),
    ('pre', '<f4', (768,)), ('input', '<f4', (768,)), ('wi_scale', '<f4'), ('wi_codes', '<i2', (768,)),
    ('up_raw', '<f4', (3072,)), ('up', '<f4', (3072,)), ('wo_scale', '<f4'), ('wo_codes', '<i2', (3072,)),
    ('down', '<f4', (768,)), ('expert', '<i4'), ('accepted', '<i4'), ('probability', '<f4'),
    ('post', '<f4', (768,)), ('final', '<f4', (768,)), ('head_input', '<f4', (768,)),
    ('head_scale', '<f4'), ('head_codes', '<i2', (768,)),
])


def exact(a, b):
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), 'component_bytes'


def quant(x):
    maximum = np.max(np.abs(x))
    scale = np.float32(1) if maximum == 0 else np.float32(maximum / np.float32(32767))
    return scale, np.clip(np.rint(x / scale), -32767, 32767).astype('<i2')


def norm(x, w):
    # Qualified Switch squares are F32, summed in F64; this is not Granite RMS.
    mean = np.float32(np.sum((x*x).astype(np.float64)) / 768)
    scale = np.float32(1) / np.sqrt(np.float32(mean + np.float32(1e-6)))
    return (x * scale) * w


def tensor(mapped, entries, name, kind='value'):
    e = entries[name]
    if kind == 'scales': return np.frombuffer(mapped, dtype='<f4', count=e['shape'][0], offset=e['scale_offset'])
    return np.frombuffer(mapped, dtype='<i1' if e['encoding'] == 1 else '<f4', count=e['elements'], offset=e['offset']).reshape(e['shape'])


def matrix(mapped, entries, name, codes, scale, rows=None):
    weights = tensor(mapped, entries, name)
    scales = tensor(mapped, entries, name, 'scales')
    if rows is not None: weights = weights[rows]; scales = scales[rows]
    sums = weights.astype(np.int64) @ codes.astype(np.int64)
    return ((sums.astype(np.float64) * scales.astype(np.float64)) * np.float64(scale)).astype('<f4')


def analyze(data, output, trace, n, source, decoder, mapped, entries, full_head=False):
    s, t = len(source), len(decoder)
    assert data[:8] == b'SWFUN001' and struct.unpack_from('<6I', data, 8) == (768, 3072, 32128, n, 11, 1)
    assert len(data) == 32 + t * DTYPE.itemsize, 'capture_exact_EOF'
    rows = np.frombuffer(data, dtype=DTYPE, offset=32)
    assert np.array_equal(rows['position'], np.arange(t)) and np.array_equal(rows['id'], decoder)
    assert np.all(rows['source_tokens'] == s) and np.array_equal(rows['route_index'], 6*s + 6*np.arange(t) + 5)
    for name in DTYPE.names:
        if name.endswith('codes'): assert np.all(rows[name] != -32768)
        elif DTYPE[name].base.kind == 'f': assert np.isfinite(rows[name]).all(), name
    enc, dec = R.R.route_bytes(output, n, s, t)
    off = 36 + 14*s*768*4
    snapshots = np.frombuffer(output, dtype='<f4', count=t*14*768, offset=off).reshape(t, 14, 768)
    logits = np.frombuffer(output, dtype='<f4', count=t*32128, offset=off+t*14*768*4).reshape(t, 32128)
    dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (n,))])
    assert trace[:8] == b'SWRTA001' and struct.unpack_from('<2I', trace, 8) == (n, 768)
    assert len(trace) == 16 + 6*(s+t)*dtype.itemsize
    rt = np.frombuffer(trace, dtype=dtype, offset=16)
    assert np.array_equal(rt['index'], np.arange(6*(s+t)))
    assert np.all(rt['phase'][:6*s] == 0) and np.all(rt['phase'][6*s:] == 1)
    assert np.isfinite(rt['input']).all() and np.isfinite(rt['scores']).all()
    last = rt[6*s+5::6]
    exact(rows['input'], last['input'])
    for name in ('expert', 'accepted', 'probability'): exact(rows[name], dec[5][name])
    assert np.all(rows['accepted'] == 1) and np.array_equal(rows['expert'], np.argmax(last['scores'], axis=1))
    exact(rows['post'], snapshots[:, 12]); exact(rows['final'], snapshots[:, 13])
    exact(rows['up'], np.where(rows['up_raw'] < 0, np.float32(0), rows['up_raw']))
    exact(rows['post'], rows['pre'] + rows['probability'][:, None] * rows['down'])
    exact(rows['head_input'], rows['final'] * np.float32(1/np.sqrt(768.0)))
    prefix = 'decoder.block.11.layer.2.'
    ffnorm = tensor(mapped, entries, prefix+'layer_norm.weight')
    finalnorm = tensor(mapped, entries, 'decoder.final_layer_norm.weight')
    norm_error = 0.; head_rows = 0
    for i, row in enumerate(rows):
        for field, expected in (('input', norm(row['pre'], ffnorm)), ('final', norm(row['post'], finalnorm))):
            error = float(np.linalg.norm(expected.astype(np.float64)-row[field]) / max(np.linalg.norm(row[field].astype(np.float64)), 1e-30))
            norm_error = max(norm_error, error); assert error <= 1e-6, ('norm', field, error)
        for field, stem in (('input', 'wi'), ('up', 'wo'), ('head_input', 'head')):
            scale, codes = quant(row[field]); exact(scale, row[stem+'_scale']); exact(codes, row[stem+'_codes'])
        ep = prefix+f'mlp.experts.expert_{int(row["expert"])}.'
        exact(matrix(mapped, entries, ep+'wi.weight', row['wi_codes'], row['wi_scale']), row['up_raw'])
        exact(matrix(mapped, entries, ep+'wo.weight', row['wo_codes'], row['wo_scale']), row['down'])
        selected = None if full_head and i == 0 else sorted({0, 1, 257, 4096, 16383, 32127, int(np.argmax(logits[i]))})
        expected = matrix(mapped, entries, 'lm_head.weight', row['head_codes'], row['head_scale'], selected)
        exact(expected, logits[i] if selected is None else logits[i, selected])
        head_rows += 32128 if selected is None else len(selected)
    return {'positions': t, 'full_WI_rows_exact': t*3072, 'full_WO_rows_exact': t*768,
            'head_rows_exact': head_rows, 'maximum_norm_relative_error': norm_error,
            'all_observed_routes_states_relu_quant_residual_and_head_input_bound': True}


def source_identity():
    for suffix in ('.c', '_entry.c', '_cost_entry.c'):
        old = BASE / ('meth393_switch_router_audit'+suffix); new = BASE / ('meth417_switch_function_capture'+suffix)
        M.committed(old); M.committed(new)
        s = new.read_text(encoding='utf-8')
        s = re.sub(r'/\* M417 observer begin \*/\n.*?/\* M417 observer end \*/\n', '', s, flags=re.S)
        s = re.sub(r'/\* M417 observer begin \*/.*?/\* M417 observer end \*/', '', s, flags=re.S)
        s = s.replace('meth417_switch_function_capture', 'meth393_switch_router_audit').replace('meth417_contract_entry', 'meth393_contract_entry').replace('meth417_forced_entry', 'meth393_forced_entry')
        assert s == old.read_text(encoding='utf-8'), suffix
    s = ENGINE.read_text(encoding='utf-8')
    prefix = '#ifdef SILICON_SWITCH_FUNCTION_CAPTURE\n#include "../native_expert_scaling/meth417_switch_function_capture_entry.c"\n#elif defined(SILICON_GRANITE_I8_FOUR_ROWS_PREFLIGHT)'
    assert s.startswith(prefix)
    old = subprocess.check_output(['git', 'show', '3513e8b:benchmarks/phase60/engine.c'], cwd=M.ROOT).decode('utf-8')
    assert s.replace(prefix, '#ifdef SILICON_GRANITE_I8_FOUR_ROWS_PREFLIGHT', 1) == old
    marker = '#elif defined(SILICON_COMPLETE_I16_NATIVE_GENERATE)'
    old = subprocess.check_output(['git', 'show', '0ff9705:benchmarks/phase60/engine.c'], cwd=M.ROOT).decode('utf-8')
    assert s[s.index(marker):].replace(marker, '#ifdef SILICON_COMPLETE_I16_NATIVE_GENERATE', 1) == old


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; files = []; bytes_out = 0
    result = {'experiment': 'METH-418-final-decoder-bank-native-observation-contract-schema-repair', 'commands': [], 'captures': []}

    def guard(child=None):
        nonlocal peak
        rss = psutil.Process().memory_info().rss
        if child is not None and child.poll() is None:
            try: rss += psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess: pass
        peak = max(peak, rss)
        assert peak <= 4 << 30 and time.monotonic()-start <= 1200 and bytes_out <= 2 << 30, 'capture_20min_4GiB_2GiB'

    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()

    def inventory():
        nonlocal files, bytes_out
        files = sorted(OUT.glob('*')); bytes_out = sum(p.stat().st_size for p in files); guard()

    try:
        helpers = [Path(__file__), PROTOCOL, ENGINE, Path(M.__file__), Path(A.__file__), Path(B.__file__), Path(R.__file__), Path(R.R.__file__), Path(U.__file__), Path(P.__file__), Path(P.F.__file__)]
        helpers += [BASE / ('meth417_switch_function_capture'+s) for s in ('.c', '_entry.c', '_cost_entry.c', '.h')]
        helpers += [BASE/'meth388_switch_thread_binding.h', BASE/'meth393_switch_router_trace.h']
        result['helper_sha256'] = {}
        for path in helpers: M.committed(path); result['helper_sha256'][str(path)] = sha(path)
        source_identity(); result['source_reversal_and_old_engine_paths_exact'] = True
        first_failure = M.DOC/'meth417_switch_function_capture_result.failure.json'; M.committed(first_failure)
        assert sha(first_failure) == '5eea9bdbd4767cc9310f74ff695b52132b8c1e7c813f272d27462b2f8c4d84ca'
        result['retained_first_failure417_sha256'] = sha(first_failure)
        records = {}
        for k, (name, expected) in INPUTS.items():
            p = M.DOC/name; M.committed(p); assert sha(p) == expected
            records[k] = json.loads(p.read_text(encoding='utf-8'))
            if k in (326, 378): assert records[k]['passed'] is True
            else: assert all(records[k]['gates'].values())
        result['input_sha256'] = {str(k): v[1] for k, v in INPUTS.items()}
        tok = []
        for k in (326, 378):
            side = {v['name']: v for v in records[k]['files']}; values = {}
            for name in ('spiece.model', 'tokenizer.json', 'special_tokens_map.json'):
                values[name] = sha(side[name]['path']); assert values[name] == side[name]['sha256']
            tok.append(values)
        assert tok[0] == tok[1]; result['fresh_tokenizer_sha256'] = tok
        topology = A.physical_topology(); assert topology == records[374]['fresh_topology']
        affinity = topology['selected_one_logical_per_physical_core'][:3]; assert affinity == [0, 2, 4]
        result['fresh_topology'] = topology
        ancestors = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.pid in ancestors: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('unrelated_background_processes', []).append({'pid': p.pid, 'argv': argv}); continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_model_job', p.pid, name)
        exports = {}; mapped = {}; initial = {}; artifacts = {}
        for n, (name, expected) in U.EXPORT.items():
            stage = f'fresh_payload{n}'; p = M.DOC/name; M.committed(p); assert sha(p) == expected
            e = json.loads(p.read_text(encoding='utf-8')); assert all(e['gates'].values()); exports[n] = e
            a = e['artifact']; artifacts[n] = a; payload = Path(a['payload'])
            initial[n] = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert initial[n][0] == a['bytes'] and sha(payload) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']
            B.read_manifest(a['manifest'], e['original_config'], e['tensors'], payload)
            mapped[n] = np.memmap(payload, dtype='u1', mode='r')
            print(json.dumps({'stage': stage, 'seconds': time.monotonic()-start}), flush=True)
        result['artifacts'] = {str(n): a for n, a in artifacts.items()}
        compiler = Path(records[374]['compile']['argv'][0]); dll = Path(records[374]['compile']['argv'][-1]).parent/'libomp.dll'
        assert sha(compiler) == records[374]['compile']['compiler_sha256'] and sha(dll) == records[374]['compile']['runtime_sha256']
        assert shutil.disk_usage(M.ROOT).free >= 4 << 30
        OUT.mkdir(parents=True); shutil.copyfile(dll, OUT/'libomp.dll'); binary = OUT/'meth418_switch_function_capture.exe'
        argv = [str(compiler), '-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off', '-fopenmp', '-DSILICON_SWITCH_FUNCTION_CAPTURE', str(ENGINE), '-o', str(binary)]
        stage = 'compile'; completed = subprocess.run(argv, capture_output=True, timeout=120)
        (OUT/'compile.stdout.log').write_bytes(completed.stdout); (OUT/'compile.stderr.log').write_bytes(completed.stderr)
        result['compile'] = {'argv': argv, 'returncode': completed.returncode, 'compiler_sha256': sha(compiler), 'runtime_sha256': sha(OUT/'libomp.dll')}
        assert completed.returncode == 0, completed.stderr.decode(errors='replace')
        result['compile']['binary_sha256'] = sha(binary); inventory()
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_WORKER_BINDING_', 'SILICON_ROUTER_AUDIT_', 'SILICON_FUNCTION_CAPTURE_'))}
        env.update(records[374]['runtime_environment']); env['OMP_NUM_THREADS'] = '3'; result['runtime_environment'] = records[374]['runtime_environment']

        def invoke(n, source, decoder, label, old_output=None, old_trace=None, old_output_sha=None, old_trace_sha=None, generated=None, negative=False):
            nonlocal stage
            stage = label; prefix = OUT/label; trace = OUT/(label+'.router.bin'); capture = OUT/(label+'.function.bin')
            child_env = dict(env); child_env.update({'SILICON_ROUTER_AUDIT_PATH': str(trace), 'SILICON_FUNCTION_CAPTURE_PATH': str(capture)})
            if negative: child_env['SILICON_WORKER_BINDING_FAULT'] = '1'
            if generated is None:
                argv = [str(binary), artifacts[n]['manifest'], ','.join(map(str, source)), ','.join(map(str, decoder)), str(prefix), '3', '0', '0', '1', '0']
            else:
                argv = [str(binary), '--generate', artifacts[n]['manifest'], ','.join(map(str, source)), str(prefix), '3', '64', '32095', '0', '0', '1', '0']
            with (OUT/(label+'.stdout.log')).open('xb') as out, (OUT/(label+'.stderr.log')).open('xb') as err:
                child = subprocess.Popen(argv, stdout=out, stderr=err, env=child_env)
                try:
                    proc = psutil.Process(child.pid); proc.cpu_affinity(affinity); assert proc.cpu_affinity() == affinity
                    while child.poll() is None: guard(child); time.sleep(.05)
                    child.wait()
                except BaseException:
                    if child.poll() is None: child.kill(); child.wait()
                    raise
            result['commands'].append({'argv': argv, 'returncode': child.returncode, 'negative': negative,
                'stdout_sha256': sha(OUT/(label+'.stdout.log')), 'stderr_sha256': sha(OUT/(label+'.stderr.log'))})
            inventory()
            if negative:
                assert child.returncode == 2 and b'worker_affinity_readback' in (OUT/(label+'.stderr.log')).read_bytes(); return
            assert child.returncode == 0, (label, child.returncode)
            logs = [json.loads(s) for s in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()]
            P.placement(logs)
            if generated is not None: assert logs[0]['generated_ids'] == generated
            else: assert logs[0]['experts_per_bank'] == n
            output = Path(str(prefix)+'.0.bin')
            assert sha(old_output) == old_output_sha == sha(output)
            assert sha(old_trace) == old_trace_sha == sha(trace)
            output_data = output.read_bytes(); trace_data = trace.read_bytes(); data = capture.read_bytes()
            assert output_data == Path(old_output).read_bytes() and trace_data == Path(old_trace).read_bytes()
            if generated is not None: decoder = [0, *generated[:-1]]
            analysis = analyze(data, output_data, trace_data, n, source, decoder, mapped[n], exports[n]['tensors'], label.endswith('book0.case0'))
            R.analyze(trace, output, n, len(source), len(decoder))
            item = {'n': n, 'label': label, 'source_ids': source, 'decoder_ids': decoder,
                'pairing_sha256': P.pairing_key(source, decoder), 'output_path': str(output), 'output_sha256': sha(output),
                'trace_path': str(trace), 'trace_sha256': sha(trace), 'capture_path': str(capture), 'capture_sha256': sha(capture),
                'prior_output_path': str(old_output), 'prior_trace_path': str(old_trace), 'rows': logs, 'analysis': analysis}
            result['captures'].append(item); guard()
            if label == 'teacher.n256.book0.case0':
                result['negative_capture_controls'] = []
                mutations = [('magic', 0, b'BADMAGIC'), ('index', 32, struct.pack('<I', 1)),
                    ('source_count', 40, struct.pack('<I', 30)), ('up_nonfinite', 32+DTYPE.fields['up_raw'][1], struct.pack('<f', float('nan'))),
                    ('wi_code_bit', 32+DTYPE.fields['wi_codes'][1], bytes([data[32+DTYPE.fields['wi_codes'][1]] ^ 1])),
                    ('up_finite_bit', 32+DTYPE.fields['up_raw'][1], bytes([data[32+DTYPE.fields['up_raw'][1]] ^ 1])),
                    ('post_bit', 32+DTYPE.fields['post'][1], bytes([data[32+DTYPE.fields['post'][1]] ^ 1])),
                    ('head_code_bit', 32+DTYPE.fields['head_codes'][1], bytes([data[32+DTYPE.fields['head_codes'][1]] ^ 1]))]
                for tag, offset, value in mutations:
                    wrong = bytearray(data); wrong[offset:offset+len(value)] = value
                    path = OUT/('negative_'+tag+'.bin'); path.write_bytes(wrong)
                    try: analyze(wrong, output_data, trace_data, n, source, decoder, mapped[n], exports[n]['tensors'])
                    except AssertionError: pass
                    else: raise AssertionError('undetected_'+tag)
                    result['negative_capture_controls'].append({'kind': tag, 'path': str(path), 'sha256': sha(path), 'detected': True})
                for tag, wrong in (('truncated', data[:-1]), ('trailing', data+b'\0')):
                    path = OUT/('negative_'+tag+'.bin'); path.write_bytes(wrong)
                    try: analyze(wrong, output_data, trace_data, n, source, decoder, mapped[n], exports[n]['tensors'])
                    except AssertionError: pass
                    else: raise AssertionError('undetected_'+tag)
                    result['negative_capture_controls'].append({'kind': tag, 'path': str(path), 'sha256': sha(path), 'detected': True})
                inventory()

        stage = 'negative_worker'; invoke(256, [2, 1], [0], 'negative_worker', negative=True)
        with threadpool_limits(limits=1):
            result['replay_runtime'] = {'numpy': np.__version__, 'BLAS': threadpool_info(), 'integer_replay': 'all WI/WO rows I64; F64 scales to F32; sampled vocabulary plus four full-head positions'}
            assert all(v['num_threads'] == 1 for v in threadpool_info())
            pairs = records[405]['pairs']; assert len(pairs) == 96
            for index, pair in enumerate(pairs):
                bi, ci = divmod(index, 4); assert (pair['book'], pair['case']) == (bi, ci)
                assert pair['prospective_split'] == ('development' if bi < 18 else 'validation')
                s, d = pair['source_ids'], pair['decoder_ids']; assert (len(s), len(d)) == (29, 14)
                assert P.pairing_key(s, d) == pair['pairing_sha256']
                for n in (256, 128):
                    old = pair['source128_capture'] if n == 128 else {'output_path': pair['source256_output_path'], 'output_sha256': pair['source256_output_sha256'], 'trace_path': pair['source256_trace_path'], 'trace_sha256': pair['source256_trace_sha256']}
                    invoke(n, s, d, f'teacher.n{n}.book{bi}.case{ci}', old['output_path'], old['trace_path'], old['output_sha256'], old['trace_sha256'])
                    result['captures'][-1]['prospective_split'] = pair['prospective_split']
                if ci == 3 and bi % 4 == 3: print(json.dumps({'teacher_paired_cases': index+1, 'seconds': time.monotonic()-start, 'output_bytes': bytes_out}), flush=True)
            for target in records[393]['targets']:
                n = target['n']; assert len(target['quality_bridges']) == 96
                for i, bridge in enumerate(target['quality_bridges']):
                    bi, ci = divmod(i, 4); assert (bridge['book'], bridge['case']) == (bi, ci)
                    old = bridge['generation_rows'][0]; tr = old['router_trace_path']; op = tr.replace('.router.bin', '.0.bin')
                    invoke(n, bridge['source_ids'], None, f'natural.n{n}.book{bi}.case{ci}', op, tr, bridge['natural_sha256'], old['router_trace_sha256'], old['generated_ids'])
                    result['captures'][-1]['qualification_only_not_paired_training'] = True
                    if ci == 3 and bi % 4 == 3: print(json.dumps({'natural_n': n, 'cases': i+1, 'seconds': time.monotonic()-start, 'output_bytes': bytes_out}), flush=True)
        assert len(result['captures']) == 384
        for n, a in artifacts.items(): assert initial[n] == (Path(a['payload']).stat().st_size, Path(a['payload']).stat().st_mtime_ns)
        inventory(); result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
        result['gates'] = {'committed_source_reversal_and_engine_exact': True, 'fresh_payload_manifest_tokenizer_identity': True,
            'ALL192_shared_context_complete_outputs_and_routes_exact': True, 'ALL192_prior_natural_complete_outputs_routes_and_greedy_exact': True,
            'actual_three_physical_workers_and_negative_fault': True, 'ALL_full_selected_WI_WO_integer_replay_exact': True,
            'ALL_quant_relu_residual_norm_head_input_and_vocabulary_controls': True, 'ALL10_capture_negative_controls_detected': True,
            'budget_20min_4GiB_2GiB': True}
        result['resource'] = {'seconds': time.monotonic()-start, 'maximum_checked_combined_rss_bytes': peak,
            'actual_file_bytes_read_for_hashes': hashed, 'output_bytes': bytes_out,
            'captured_positions': sum(v['analysis']['positions'] for v in result['captures'])}
        result['decision'] = 'eligible_for_new_bounded_final_bank_function_objective_and_gradient_protocol'
        result['scope'] = 'Observation prerequisite only, on already-consumed calibration/quality contexts. ALL192 teacher pairs use shared405 contexts; natural128 uses its own qualified393 contexts and is excluded from paired training. All original outputs/routes/greedy bytes preserved. ALL selected WI/WO rows replayed, head sampled plus four complete positions. No fit, added expert artifact, gradient/native surrogate, new quality, rate, LUT/physical DRAM, larger-n usefulness, GPU/T4/network or additional-family claim.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': M.digest(args.out), 'gates': result['gates'], 'resource': result['resource'], 'decision': result['decision']}), flush=True)
    except BaseException as error:
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'bytes_hashed': hashed, 'maximum_checked_combined_rss_bytes': peak})
        if OUT.exists(): result['partial_output_bytes'] = sum(p.stat().st_size for p in OUT.glob('*'))
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
