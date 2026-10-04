"""Both real original-source artifacts, NEW three-worker fixed CPU cost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import psutil
import meth324_switch_reference as M
import meth359_switch_cost_topology as A
import meth389_switch_three_workers_contract as C

NUMERIC = M.DOC / 'meth389_switch_three_workers_contract_result.json'
NUMERIC_SHA = 'fd12cfc2809fb42f52faef0b442781154c30ba27576f264c5f693bc30ec91dd2'
PROTOCOL = M.DOC / 'METH_390_SWITCH_THREE_WORKERS_COST_PROTOCOL_20261004.md'
OUT = M.ROOT / 'results/native_expert_scaling/meth390_switch_three_workers_cost'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); maximum = 0; stage = 'bindings'
    result = {'experiment': 'METH-390-both-originals-three-workers-actual-CPU-cost', 'targets': [], 'commands': []}

    def guard(child=None):
        nonlocal maximum
        rss = psutil.Process().memory_info().rss
        if child is not None:
            try: rss += psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess: pass
        maximum = max(maximum, rss)
        if rss > 16 << 30 or time.monotonic() - start > 1800:
            if child is not None and child.poll() is None: child.kill(); child.wait()
            raise RuntimeError('three_workers_cost_30min_16GiB')

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20): h.update(block); guard()
        return h.hexdigest()

    try:
        for path in (Path(__file__), PROTOCOL, NUMERIC, Path(M.__file__), Path(A.__file__), Path(C.__file__), C.ENGINE): M.committed(path)
        C.source_identity(); assert digest(NUMERIC) == NUMERIC_SHA
        numeric = json.loads(NUMERIC.read_text(encoding='utf-8')); assert all(numeric['gates'].values())
        assert numeric['primary_threads'] == 3 and numeric['primary_affinity'] == [0, 2, 4]
        assert A.physical_topology() == numeric['fresh_topology']; affinity = numeric['primary_affinity']
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            command = ' '.join(process.info['cmdline'] or []).replace('\\', '/')
            if process.pid not in own and (process.info['name'] or '').lower() in ('python.exe', 'meth374_switch_physical_workers.exe', 'meth389_switch_three_workers.exe'):
                assert not ('benchmarks/native_expert_scaling/' in command or 'switch_' in process.info['name']), 'concurrent_model_or_timing'
        binary = Path(numeric['compile']['argv'][-1]); compiler = Path(numeric['compile']['argv'][0]); dll = binary.parent / 'libomp.dll'
        assert digest(binary) == numeric['compile']['binary_sha256'] and digest(compiler) == numeric['compile']['compiler_sha256'] and digest(dll) == numeric['compile']['runtime_sha256']
        result.update({'controller_sha256': digest(__file__), 'protocol_sha256': digest(PROTOCOL), 'numeric389_sha256': NUMERIC_SHA,
                       'compile': numeric['compile'], 'runtime_environment': numeric['runtime_environment'], 'primary_native_threads': 3,
                       'native_process_affinity': affinity, 'fresh_topology': numeric['fresh_topology']})
        OUT.mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_WORKER_BINDING_'))}
        env.update(numeric['runtime_environment']); env['OMP_NUM_THREADS'] = '3'
        for candidate in numeric['targets']:
            n = candidate['n']; stage = f'fresh_artifact{n}'; artifact = candidate['artifact']; payload = Path(artifact['payload'])
            before = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert before[0] == artifact['bytes'] and digest(payload) == artifact['sha256'] and digest(artifact['manifest']) == artifact['manifest_sha256']
            target = {'n': n, 'artifact': artifact, 'quality_sha256': candidate['quality_sha256'], 'bridges': [], 'measurements': [], 'profiles': []}
            result['targets'].append(target)

            def run(control, profile, warm, reps, label):
                prefix = OUT / label
                argv = [str(binary), artifact['manifest'], ','.join(map(str, control['source_ids'])), ','.join(map(str, control['decoder_ids'])), str(prefix), '3', str(profile), str(warm), str(reps), '0']
                with (OUT / (label + '.stdout.log')).open('wb') as stdout, (OUT / (label + '.stderr.log')).open('wb') as stderr:
                    child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env)
                    try:
                        process = psutil.Process(child.pid); process.cpu_affinity(affinity); actual = process.cpu_affinity(); assert actual == affinity
                        while child.poll() is None: guard(child); time.sleep(.1)
                    except BaseException:
                        if child.poll() is None: child.kill(); child.wait()
                        raise
                assert child.returncode == 0, (label, child.returncode)
                result['commands'].append({'argv': argv, 'returncode': child.returncode, 'actual_affinity': actual, 'stdout_sha256': digest(OUT / (label + '.stdout.log')), 'stderr_sha256': digest(OUT / (label + '.stderr.log'))})
                rows = [json.loads(line) for line in (OUT / (label + '.stdout.log')).read_text(encoding='utf-8').splitlines()]
                assert [row['repetition'] for row in rows] == list(range(-warm, reps))
                for row in rows:
                    assert row['worker_physical_cores'] == 6 and [v['slot'] for v in row['worker_affinity']] == [0, 1, 2]
                    assert [v['actual_mask'] for v in row['worker_affinity']] == [1, 4, 16]
                    assert all(v['group'] == 0 for v in row['worker_affinity']) and len({v['windows_thread_id'] for v in row['worker_affinity']}) == 3
                    assert all(v['group'] == 0 and v['actual_mask'] == 1 << affinity[v['slot']] for v in row['worker_binding_events'])
                    assert {v['slot'] for v in row['worker_binding_events']} == {0, 1, 2}
                    row['output_sha256'] = digest(Path(str(prefix) + f'.{row["repetition"]}.bin')); assert row['output_sha256'] == control['teacher_sha256']
                    counts = row['counters'][1]; t = len(control['decoder_ids'])
                    assert row['experts_per_bank'] == n and row['decoder_positions'] == t
                    assert sum(v['code_bytes'] for v in counts) == 123764736 * t and sum(v['scale_bytes'] for v in counts) == 534016 * t and sum(v['f32_bytes'] for v in counts) == 18432 * n * t
                    assert counts[2]['calls'] == 6 * t and counts[3]['calls'] == t
                return rows

            for index, control in enumerate(candidate['controls']):
                rows = run(control, control['profile'], 0, 1, f'n{n}.bridge{index}')
                target['bridges'].append({'full_reference_bytes_exact389': True, 'output_sha256': rows[0]['output_sha256']})
            fixtures = [control for control in candidate['controls'] if len(control['decoder_ids']) == 32]
            assert [len(v['source_ids']) for v in fixtures] == [9, 64]
            for fixture in fixtures:
                source_length = len(fixture['source_ids']); stage = f'n{n}.source{source_length}.timing'
                rows = run(fixture, 0, 1, 3, stage); observed = [v for v in rows if v['repetition'] >= 0]
                decode = [v['decode_seconds'] / 32 * 1000 for v in observed]; full = [v['full_seconds'] / 32 * 1000 for v in observed]
                measurement = {'source_length': source_length, 'rows': rows, 'median_decode_ms_per_position': float(np.median(decode)),
                               'median_full_ms_per_position': float(np.median(full)), 'repeat_ratio_max_min_decode': max(decode) / min(decode),
                               'logical_matrix_bytes_per_decode_position': 123764736 + 534016 + 18432 * n}
                target['measurements'].append(measurement); print(json.dumps({'n': n, **{k: v for k, v in measurement.items() if k != 'rows'}}), flush=True)
            for fixture in fixtures:
                source_length = len(fixture['source_ids']); stage = f'n{n}.source{source_length}.profile'
                target['profiles'].append({'source_length': source_length, 'rows': run(fixture, 1, 1, 1, stage), 'complete_bytes_exact_primary': True})
            target['gates'] = {'all_actual_worker_and_process_readbacks': True, 'all_complete_control_counter_bytes_exact389': True, 'all_profile_bytes_exact_primary': True,
                               'both_median_decode_le20ms': all(v['median_decode_ms_per_position'] <= 20 for v in target['measurements']),
                               'both_median_FULL_le20ms': all(v['median_full_ms_per_position'] <= 20 for v in target['measurements']),
                               'both_individual_decode_repeat_le1p10': all(v['repeat_ratio_max_min_decode'] <= 1.10 for v in target['measurements'])}
            assert before == (payload.stat().st_size, payload.stat().st_mtime_ns)
            target['passed'] = all(target['gates'].values())
        result['gates'] = {'both_original_artifact_costs_qualified': all(v['passed'] for v in result['targets']), 'all_same_binary_worker_outputs_exact': True}
        result['resource'] = {'main_seconds_excluding_imports': time.monotonic() - start, 'maximum_checked_combined_rss_bytes': maximum}
        result['decision'] = 'only_individually_cost_qualified_artifacts_eligible_for_separately_frozen_SAME_accepted_rate'
        result['scope'] = 'NEW three-worker profile, two complete useful source artifacts, unchanged per-fixture medians/repeat rubric; no optional retry. First failures retained per artifact. Apparatus positions not accepted tokens, logical bytes not physical DRAM.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': digest(args.out), 'gates': result['gates'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'stage': stage, 'error': repr(error), 'seconds': time.monotonic() - start, 'maximum_checked_combined_rss_bytes': maximum})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
