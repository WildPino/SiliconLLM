"""Paired source-native inputs on consumed calibration cases; no fitted model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
import psutil
import meth324_switch_reference as M
import meth359_switch_cost_topology as A
import meth393_switch_router_analysis as R

PROTOCOL = M.DOC / 'METH_403_SWITCH_CROSS_SOURCE_INPUTS_PROTOCOL_20261004.md'
OUT = M.ROOT / 'results/native_expert_scaling/meth403_switch_cross_source_inputs'
INPUTS = {
    362: ('meth362_switch_multi_span_manifest.json', 'c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'),
    374: ('meth374_switch_physical_workers_contract_result.json', '4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'),
    389: ('meth389_switch_three_workers_contract_result.json', 'fd12cfc2809fb42f52faef0b442781154c30ba27576f264c5f693bc30ec91dd2'),
    393: ('meth393_switch_router_audit_result.json', 'bea3d2e1db10f03d5d9de3173d71a455142166ae71229bbb847959baf12ce07d'),
    401: ('meth401_pretrained_bank_union_applicability_result.json', 'd35bf7f82b064593ff8568379fc8106b100d8337c1b860395e1bc6bd83d4e160'),
    326: ('meth326_switch_acquisition_result.json', '39bac2bda18cc6660da77bba8256d1e21fcad5b6fa70796d343b7e139d480711'),
    378: ('meth378_switch_base128_acquisition_result.json', '89add8d24541423dfa61827f785bf125c2062cf1aa6976214f03da94b7c582b7'),
}


def pairing_key(source, decoder):
    # Explicit lengths prevent ambiguous concatenation of the two sequences.
    return hashlib.sha256(struct.pack('<2I', len(source), len(decoder)) +
                          struct.pack('<' + 'I' * (len(source) + len(decoder)), *source, *decoder)).hexdigest()


def read_trace(path, n):
    data = Path(path).read_bytes()
    assert data[:8] == b'SWRTA001' and struct.unpack_from('<2I', data, 8) == (n, 768)
    dtype = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (n,))])
    assert len(data) == 16 + 258 * dtype.itemsize
    rows = np.frombuffer(data, dtype=dtype, offset=16)
    assert np.array_equal(rows['index'], np.arange(258))
    assert np.all(rows['phase'][:174] == 0) and np.all(rows['phase'][174:] == 1)
    assert np.isfinite(rows['input']).all() and np.isfinite(rows['scores']).all()
    # Encoder: bank then source position; decoder: position then bank.
    return [rows['input'][b*29:(b+1)*29].copy() for b in range(6)] + [rows['input'][174+b::6].copy() for b in range(6)]


def placement(rows):
    assert len(rows) == 1 and rows[0]['repetition'] == 0
    for row in rows:
        assert row['threads'] == 3 and row['profile'] == 0 and row['worker_physical_cores'] == 6
        assert [v['slot'] for v in row['worker_affinity']] == [0, 1, 2]
        assert [v['actual_mask'] for v in row['worker_affinity']] == [1, 4, 16]
        assert all(v['group'] == 0 for v in row['worker_affinity'])
        assert len({v['windows_thread_id'] for v in row['worker_affinity']}) == 3
        assert {v['slot'] for v in row['worker_binding_events']} == {0, 1, 2}
        assert all(v['group'] == 0 and v['actual_mask'] == 1 << (2*v['slot']) for v in row['worker_binding_events'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'
    result = {'experiment': 'METH-403-paired-consumed-cross-source-normalized-inputs', 'commands': [], 'pairs': [], 'banks': []}

    def guard(child=None):
        nonlocal peak
        rss = psutil.Process().memory_info().rss
        if child is not None and child.poll() is None:
            try: rss += psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess: pass
        peak = max(peak, rss)
        assert peak <= 4 << 30 and time.monotonic() - start <= 1200, 'paired_20min_4GiB'

    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()

    try:
        helpers = [Path(__file__), PROTOCOL, Path(M.__file__), Path(A.__file__), Path(R.__file__), Path(R.R.__file__)]
        result['helper_sha256'] = {}
        for path in helpers: M.committed(path); result['helper_sha256'][str(path)] = sha(path)
        records = {}
        for number, (name, expected) in INPUTS.items():
            path = M.DOC / name; M.committed(path); assert sha(path) == expected
            records[number] = json.loads(path.read_text(encoding='utf-8'))
            if 'gates' in records[number]: assert all(records[number]['gates'].values())
        result['input_sha256'] = {str(k): v[1] for k, v in INPUTS.items()}
        # Token IDs represent the same tokenizer vocabulary/control tokens.
        tokenizers = []
        for acquisition in (326, 378):
            side = {v['name']: v for v in records[acquisition]['files']}
            hashes = {}
            for name in ('spiece.model', 'tokenizer.json', 'special_tokens_map.json'):
                hashes[name] = sha(side[name]['path']); assert hashes[name] == side[name]['sha256']
            tokenizers.append(hashes)
        assert tokenizers[0] == tokenizers[1]; result['fresh_tokenizer_sha256'] = tokenizers
        binaries = {}
        for number in (389, 393):
            compiled = records[number]['compile']; binary = Path(compiled['argv'][-1])
            assert sha(binary) == compiled['binary_sha256']
            assert sha(binary.parent / 'libomp.dll') == compiled['runtime_sha256']
            binaries[number] = binary
        result['qualified_binaries'] = {str(k): records[k]['compile'] for k in binaries}
        topology = A.physical_topology(); assert topology == records[374]['fresh_topology']
        affinity = topology['selected_one_logical_per_physical_core'][:3]; assert affinity == [0, 2, 4]
        result['fresh_topology'] = topology
        ancestors = {v.pid for v in psutil.Process().parents()} | {os.getpid()}
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            if process.pid in ancestors: continue
            name = (process.info['name'] or '').lower(); command = ' '.join(process.info['cmdline'] or [])
            unrelated_daemon = (name == 'pythonw.exe' and len(process.info['cmdline'] or []) == 2
                and Path(process.info['cmdline'][1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve())
            if unrelated_daemon:
                result.setdefault('unrelated_background_processes', []).append({'pid': process.pid, 'name': name,
                    'argv': process.info['cmdline'], 'cpu_user_system_seconds': list(process.cpu_times()[:2])})
                continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_model_job', process.pid, name, command)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_WORKER_BINDING_', 'SILICON_ROUTER_AUDIT_'))}
        env.update(records[374]['runtime_environment']); env['OMP_NUM_THREADS'] = '3'
        result['runtime_environment'] = {k: v for k, v in env.items() if k.startswith(('OMP_', 'KMP_'))}
        artifacts = {v['n']: v['artifact'] for v in records[401]['sources']}; initial = {}
        for n, artifact in artifacts.items():
            stage = f'fresh_payload_{n}'; payload = Path(artifact['payload'])
            initial[n] = [payload.stat().st_size, payload.stat().st_mtime_ns]
            assert initial[n][0] == artifact['bytes'] and sha(payload) == artifact['sha256']
            assert sha(artifact['manifest']) == artifact['manifest_sha256']
            print(json.dumps({'stage': stage, 'seconds': time.monotonic()-start}), flush=True)
        result['artifacts'] = {str(n): {'artifact': a, 'initial_size_mtime_ns': initial[n]} for n, a in artifacts.items()}
        OUT.mkdir(parents=True)

        def invoke(number, source, decoder, label, negative=False):
            child_env = dict(env); prefix = OUT / label; trace = OUT / (label + '.router.bin')
            if number == 393: child_env['SILICON_ROUTER_AUDIT_PATH'] = str(trace)
            if negative: child_env['SILICON_WORKER_BINDING_FAULT'] = '1'
            argv = [str(binaries[number]), artifacts[128]['manifest'], ','.join(map(str, source)), ','.join(map(str, decoder)), str(prefix), '3', '0', '0', '1', '0']
            with (OUT / (label + '.stdout.log')).open('xb') as stdout, (OUT / (label + '.stderr.log')).open('xb') as stderr:
                child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=child_env)
                try:
                    process = psutil.Process(child.pid); process.cpu_affinity(affinity); assert process.cpu_affinity() == affinity
                    while child.poll() is None: guard(child); time.sleep(.1)
                    child.wait()
                except BaseException:
                    if child.poll() is None: child.kill(); child.wait()
                    raise
            result['commands'].append({'argv': argv, 'returncode': child.returncode, 'negative': negative,
                                       'stdout_sha256': sha(OUT / (label + '.stdout.log')), 'stderr_sha256': sha(OUT / (label + '.stderr.log'))})
            if negative:
                assert child.returncode == 2 and b'worker_affinity_readback' in (OUT / (label + '.stderr.log')).read_bytes()
                return None
            assert child.returncode == 0, (label, child.returncode)
            rows = [json.loads(line) for line in (OUT / (label + '.stdout.log')).read_text(encoding='utf-8').splitlines()]
            placement(rows); assert rows[0]['experts_per_bank'] == 128
            output = Path(str(prefix) + '.0.bin'); output_sha = sha(output)
            analysis = None
            if number == 393: analysis = R.analyze(trace, output, 128, len(source), len(decoder))[0]
            return {'rows': rows, 'output_path': str(output), 'output_sha256': output_sha,
                    'trace_path': str(trace) if number == 393 else None, 'trace_sha256': sha(trace) if number == 393 else None, 'analysis': analysis}

        stage = 'negative_placement'; invoke(393, [2, 1], [0], 'negative_worker', True)
        source_bridges = {(v['book'], v['case']): v for v in records[393]['targets'][0]['quality_bridges']}
        assert records[393]['targets'][0]['n'] == 256
        collected = {n: [[] for _ in range(12)] for n in (256, 128)}; keys = set()
        for bi, book in enumerate(records[362]['items']):
            stage = f'paired_book_{bi}'; case = book['cases'][0]; source = case['source_ids']; decoder = case['decoder_ids']
            assert len(source) == 29 and len(decoder) == 14
            old = source_bridges[(bi, 0)]; assert old['source_ids'] == source and old['decoder_ids'] == decoder and old['profile'] == 0
            key = pairing_key(source, decoder); assert key not in keys; keys.add(key)
            assert key != pairing_key(source, decoder[:-1] + [(decoder[-1] + 1) % 32128])
            old_row = old['teacher_rows'][0]; placement(old['teacher_rows'])
            old_output = Path(old_row['router_trace_path'].replace('.router.bin', '.0.bin'))
            assert sha(old_row['router_trace_path']) == old_row['router_trace_sha256']
            assert sha(old_output) == old['teacher_sha256'] == old_row['output_sha256']
            old_analysis = R.analyze(old_row['router_trace_path'], old_output, 256, 29, 14)[0]
            plain = invoke(389, source, decoder, f'book{bi}.plain128')
            captured = invoke(393, source, decoder, f'book{bi}.trace128')
            assert plain['output_sha256'] == captured['output_sha256']
            for n, path in ((256, old_row['router_trace_path']), (128, captured['trace_path'])):
                arrays = read_trace(path, n)
                for bank, values in enumerate(arrays): collected[n][bank].append(values)
            result['pairs'].append({'book': bi, 'case': 0, 'source_id': book['source_id'], 'whole_source_utf8_sha256': book['whole_source_utf8_sha256'],
                'source_ids': source, 'decoder_ids': decoder, 'pairing_sha256': key, 'prospective_split': 'development' if bi < 18 else 'validation',
                'source256_trace_path': old_row['router_trace_path'], 'source256_trace_sha256': old_row['router_trace_sha256'],
                'source256_output_path': str(old_output), 'source256_output_sha256': old['teacher_sha256'], 'source256_analysis': old_analysis,
                'source128_plain': plain, 'source128_capture': captured})
            if bi == 0:
                result['trace_negative_controls'] = []
                original = Path(captured['trace_path']).read_bytes()
                for tag, offset, value in (('index', 16, struct.pack('<I', 1)), ('score', 16+8+768*4, struct.pack('<f', float('nan')))):
                    invalid = bytearray(original); invalid[offset:offset+4] = value; path = OUT / ('negative_' + tag + '.bin'); path.write_bytes(invalid)
                    try: R.analyze(path, captured['output_path'], 128, 29, 14)
                    except AssertionError: pass
                    else: raise AssertionError('undetected_' + tag)
                    result['trace_negative_controls'].append({'kind': tag, 'sha256': sha(path), 'detected': True})
            if bi % 4 == 3: print(json.dumps({'paired_cases': bi+1, 'seconds': time.monotonic()-start}), flush=True)
        assert len(result['pairs']) == 24 and len({v['source_id'] for v in result['pairs']}) == 24
        stage = 'input_diagnostics'; all_identity = True
        for bank in range(12):
            x = np.concatenate(collected[256][bank]).astype(np.float64); y = np.concatenate(collected[128][bank]).astype(np.float64)
            nx = np.linalg.norm(x, axis=1); ny = np.linalg.norm(y, axis=1); assert np.all(nx > 0) and np.all(ny > 0)
            relative = np.linalg.norm(x-y, axis=1) / ny; cosine = np.sum(x*y, axis=1) / (nx*ny)
            maximum = float(np.max(np.abs(x-y)))
            identity = bool(relative.max() <= 1e-6 and maximum <= 1e-5); all_identity &= identity
            rows_per_book = 29 if bank < 6 else 14
            def quantiles(values): return np.quantile(values, [0, .05, .5, .95, 1]).tolist()
            result['banks'].append({'stack': 'encoder' if bank < 6 else 'decoder', 'bank': bank % 6, 'dimension': 768,
                'paired_positions': len(x), 'development_positions': 18*rows_per_book, 'validation_positions': 6*rows_per_book,
                'relative_L2_to_source128_quantiles': quantiles(relative), 'cosine_quantiles': quantiles(cosine),
                'source256_norm_quantiles': quantiles(nx), 'source128_norm_quantiles': quantiles(ny), 'maximum_absolute_difference': maximum,
                'identity_interface_gate': identity, 'full768_map_identifiable_from_development_sample_count': 18*rows_per_book >= 768})
        for n, artifact in artifacts.items():
            path = Path(artifact['payload']); final = [path.stat().st_size, path.stat().st_mtime_ns]; assert initial[n] == final
            result['artifacts'][str(n)]['final_size_mtime_ns'] = final
        result['gates'] = {'both_complete_payload_and_manifest_fresh_hashes_exact': True, 'same_tokenizer_vocabulary_and_special_tokens_fresh_exact': True,
            'qualified393_and389_binaries_DLLs_exact': True, 'ALL24_same_token_and_forced_prefix_pairs': True,
            'ALL24_source128_complete_output_bytes_SHA_exact_plain389': True, 'ALL24_source256_trace_output_SHA_exact393': True,
            'all_selected_router_ID_probability_reconstructions': True, 'all_actual_three_worker_readbacks': True,
            'negative_worker_index_score_and_prefix_controls_detected': True}
        result['identity_interface_ALL12_banks'] = all_identity
        result['decision'] = 'identity_inputs_eligible_only_for_new_function_response_screen' if all_identity else 'reject_unaligned_identity_interface_require_new_input_and_output_function_alignment_screen'
        size = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()); assert size <= 1 << 30
        guard(); result['resource'] = {'main_seconds': time.monotonic()-start, 'maximum_checked_combined_rss_bytes': peak, 'file_bytes_hashed': hashed, 'new_output_bytes': size}
        result['scope'] = 'Consumed362 case0 of all24 books, calibration only; no new held-out quality. Paired normalized router inputs in own complete source-native targets.256 trace reused with fresh full hash;128 new393 capture complete output SHA equals new389 plain. Prospective18/6 split does not become held-out quality; no fitting here. Identity thresholds are local numeric compatibility, not quality bounds. No full768 fit, output-space alignment, transplanted function, new384 selector/artifact, usefulness, accepted rate, physical DRAM or other-family proof.'
        M.write(args.out, result); print(json.dumps({'sha256': sha(args.out), 'decision': result['decision'], 'gates': result['gates'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'maximum_checked_combined_rss_bytes': peak, 'file_bytes_hashed': hashed})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
