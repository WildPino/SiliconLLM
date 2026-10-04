"""Exact three-worker execution, both qualified original-source whole outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil
import meth324_switch_reference as M
import meth359_switch_cost_topology as A

BASE = M.ROOT / 'benchmarks/native_expert_scaling'
ENGINE = M.ROOT / 'benchmarks/phase60/engine.c'
OUT = M.ROOT / 'results/native_expert_scaling/meth388_switch_three_workers_contract'
PROTOCOL = M.DOC / 'METH_388_SWITCH_THREE_WORKERS_CONTRACT_PROTOCOL_20261004.md'
INPUTS = {
    356: ('meth356_switch_all_a16_contract_result.json', 'ec51e76c08277e4f874cacac9a49d5b60477c0abb6a651cbe648f9bdad3f66d3'),
    362: ('meth362_switch_multi_span_manifest.json', 'c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'),
    363: ('meth363_switch_all_a16_multi_span_quality_result.json', 'ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'),
    374: ('meth374_switch_physical_workers_contract_result.json', '4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'),
    381: ('meth381_switch_base128_contract_result.json', '6ede7a90fbf2716381d31456897eb41b01a0c6ed9010e7ba3b3af91064831a6e'),
    382: ('meth382_switch_multi_span_manifest.json', '96b3ffd09b0bdd42d7c990006e8744e39b5e1f32b7ae40aaf3b3dca98bf4db78'),
    387: ('meth387_switch_base128_multi_span_quality_result.json', '539275a209c8b8a5680c388ec9130012157a681ed382ba1a4bf49ab8cccb9d4c'),
}


def source_identity():
    for suffix in ('.c', '_entry.c', '_cost_entry.c'):
        old = BASE / f'meth374_switch_physical_workers{suffix}'
        new = BASE / f'meth388_switch_three_workers{suffix}'
        M.committed(old); M.committed(new)
        reversed_source = new.read_text(encoding='utf-8').replace('meth388_switch_three_workers', 'meth374_switch_physical_workers').replace('meth388_contract_entry', 'meth374_contract_entry').replace('meth388_forced_entry', 'meth374_forced_entry').replace('meth388_switch_thread_binding.h', 'meth374_switch_thread_binding.h').replace('threads==1||threads==3||threads==6', 'threads==1||threads==6')
        assert reversed_source == old.read_text(encoding='utf-8'), suffix
    old = BASE / 'meth374_switch_thread_binding.h'
    new = BASE / 'meth388_switch_thread_binding.h'
    M.committed(old); M.committed(new)
    assert new.read_text(encoding='utf-8').replace('threads==1||threads==3||threads==6', 'threads==1||threads==6') == old.read_text(encoding='utf-8')
    data = ENGINE.read_bytes()
    prefix = b'#ifdef SILICON_SWITCH_THREE_WORKERS\n#include "../native_expert_scaling/meth388_switch_three_workers_entry.c"\n#elif defined(SILICON_SWITCH_PHYSICAL_WORKERS)'
    assert data.startswith(prefix)
    assert data.replace(prefix, b'#ifdef SILICON_SWITCH_PHYSICAL_WORKERS', 1) == subprocess.check_output(['git', 'show', '4a22f08:benchmarks/phase60/engine.c'], cwd=M.ROOT)
    marker = b'#elif defined(SILICON_COMPLETE_I16_NATIVE_GENERATE)'
    assert data[data.index(marker):].replace(marker, b'#ifdef SILICON_COMPLETE_I16_NATIVE_GENERATE', 1) == subprocess.check_output(['git', 'show', '0ff9705:benchmarks/phase60/engine.c'], cwd=M.ROOT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); maximum = 0; stage = 'bindings'
    result = {'experiment': 'METH-388-both-originals-three-workers-exact-whole-contract', 'commands': [], 'targets': []}

    def guard(child=None):
        nonlocal maximum
        rss = psutil.Process().memory_info().rss
        if child is not None:
            try: rss += psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess: pass
        maximum = max(maximum, rss)
        if rss > 16 << 30 or time.monotonic() - start > 1800:
            if child is not None and child.poll() is None: child.kill(); child.wait()
            raise RuntimeError('three_workers_contract_30min_16GiB')

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20): h.update(block); guard()
        return h.hexdigest()

    try:
        for path in (Path(__file__), PROTOCOL, ENGINE, Path(M.__file__), Path(A.__file__)): M.committed(path)
        source_identity(); records = {}
        for number, (name, expected) in INPUTS.items():
            path = M.DOC / name; M.committed(path); assert digest(path) == expected
            records[number] = json.loads(path.read_text(encoding='utf-8'))
            assert all(records[number]['gates'].values()), number
        execution = records[374]; topology = A.physical_topology()
        assert topology == execution['fresh_topology']
        affinity = topology['selected_one_logical_per_physical_core'][:3]; assert affinity == [0, 2, 4]
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            command = ' '.join(process.info['cmdline'] or []).replace('\\', '/')
            if process.pid not in own and (process.info['name'] or '').lower() in ('python.exe', 'meth374_switch_physical_workers.exe', 'meth388_switch_three_workers.exe'):
                assert not ('benchmarks/native_expert_scaling/' in command or 'switch_' in process.info['name']), 'concurrent_model_or_timing'
        compiler = Path(execution['compile']['argv'][0]); dll = Path(execution['compile']['argv'][-1]).parent / 'libomp.dll'
        assert digest(compiler) == execution['compile']['compiler_sha256'] and digest(dll) == execution['compile']['runtime_sha256']
        OUT.mkdir(parents=True); shutil.copyfile(dll, OUT / 'libomp.dll'); binary = OUT / 'meth388_switch_three_workers.exe'
        argv = [str(compiler), '-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off', '-fopenmp', '-DSILICON_SWITCH_THREE_WORKERS', str(ENGINE), '-o', str(binary)]
        compiled = subprocess.run(argv, capture_output=True, timeout=120)
        (OUT / 'compile.stdout.log').write_bytes(compiled.stdout); (OUT / 'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode == 0, compiled.stderr.decode(errors='replace')
        result.update({'controller_sha256': digest(__file__), 'protocol_sha256': digest(PROTOCOL), 'engine_sha256': digest(ENGINE),
                       'input_sha256': {str(n): v[1] for n, v in INPUTS.items()}, 'source_math_reversal_exact374': True,
                       'compile': {'argv': argv, 'binary_sha256': digest(binary), 'compiler_sha256': digest(compiler), 'runtime_sha256': digest(OUT / 'libomp.dll')},
                       'runtime_environment': execution['runtime_environment'], 'fresh_topology': topology,
                       'primary_threads': 3, 'primary_affinity': affinity, 'unchanged374_primitive_Tiny_nine_fault_evidence_reused': True})
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_WORKER_BINDING_'))}
        env.update(execution['runtime_environment']); env['OMP_NUM_THREADS'] = '3'

        def invoke(argv, label, negative=False):
            child_env = dict(env)
            if negative: child_env['SILICON_WORKER_BINDING_FAULT'] = '1'
            with (OUT / (label + '.stdout.log')).open('wb') as stdout, (OUT / (label + '.stderr.log')).open('wb') as stderr:
                child = subprocess.Popen([str(binary), *map(str, argv)], stdout=stdout, stderr=stderr, env=child_env)
                try:
                    process = psutil.Process(child.pid); process.cpu_affinity(affinity); actual = process.cpu_affinity(); assert actual == affinity
                    while child.poll() is None: guard(child); time.sleep(.1)
                except BaseException:
                    if child.poll() is None: child.kill(); child.wait()
                    raise
            result['commands'].append({'argv': [str(binary), *map(str, argv)], 'returncode': child.returncode, 'actual_affinity': actual,
                                       'negative': negative, 'stdout_sha256': digest(OUT / (label + '.stdout.log')), 'stderr_sha256': digest(OUT / (label + '.stderr.log'))})
            if negative:
                assert child.returncode == 2 and b'worker_affinity_readback' in (OUT / (label + '.stderr.log')).read_bytes(); return []
            assert child.returncode == 0, (label, child.returncode)
            rows = [json.loads(line) for line in (OUT / (label + '.stdout.log')).read_text(encoding='utf-8').splitlines()]
            for row in rows:
                assert row['worker_physical_cores'] == 6 and [v['slot'] for v in row['worker_affinity']] == [0, 1, 2]
                assert [v['actual_mask'] for v in row['worker_affinity']] == [1, 4, 16]
                assert all(v['group'] == 0 for v in row['worker_affinity']) and len({v['windows_thread_id'] for v in row['worker_affinity']}) == 3
                assert all(v['group'] == 0 and v['actual_mask'] == 1 << affinity[v['slot']] for v in row['worker_binding_events'])
                assert {v['slot'] for v in row['worker_binding_events']} == {0, 1, 2}
            return rows

        for n, qn, mn in ((256, 363, 362), (128, 387, 382)):
            stage = f'fresh_artifact{n}'; quality = records[qn]; cohort = records[mn]; artifact = quality['artifact']
            payload = Path(artifact['payload']); before = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert before[0] == artifact['bytes'] and digest(payload) == artifact['sha256'] and digest(artifact['manifest']) == artifact['manifest_sha256']
            target = {'n': n, 'artifact': artifact, 'quality_sha256': INPUTS[qn][1], 'cohort_sha256': INPUTS[mn][1], 'controls': [], 'quality_bridges': []}
            result['targets'].append(target)
            if n == 256:
                invoke([artifact['manifest'], '2,1', '0', OUT / 'negative', 3, 0, 0, 1, 0], 'negative', True)
                result['negative_actual_worker_fault_detected'] = True

            def complete(source, decoder, label, teacher_sha, natural_sha=None, ids=None, cap=64, closing=32095, profile=0):
                prefix = OUT / label
                rows = invoke([artifact['manifest'], ','.join(map(str, source)), ','.join(map(str, decoder)), prefix, 3, profile, 1, 1, 0], label)
                assert [row['repetition'] for row in rows] == [-1, 0]
                for row in rows:
                    row['output_sha256'] = digest(Path(str(prefix) + f'.{row["repetition"]}.bin')); assert row['output_sha256'] == teacher_sha
                    counts = row['counters'][1]; t = len(decoder)
                    assert sum(v['code_bytes'] for v in counts) == 123764736 * t and sum(v['scale_bytes'] for v in counts) == 534016 * t
                    assert sum(v['f32_bytes'] for v in counts) == 18432 * n * t
                gen_rows = []
                if natural_sha is not None:
                    gp = Path(str(prefix) + '.generation')
                    gen_rows = invoke(['--generate', artifact['manifest'], ','.join(map(str, source)), gp, 3, cap, closing, profile, 1, 1, 0], label + '.generation')
                    assert [row['repetition'] for row in gen_rows] == [-1, 0]
                    for row in gen_rows:
                        row['output_sha256'] = digest(Path(str(gp) + f'.{row["repetition"]}.bin'))
                        assert row['output_sha256'] == natural_sha and row['generated_ids'] == ids
                return {'source_ids': source, 'decoder_ids': decoder, 'profile': profile, 'teacher_sha256': teacher_sha,
                        'natural_sha256': natural_sha, 'teacher_rows': rows, 'generation_rows': gen_rows, 'complete_bytes_exact_quality_or_independent_reference': True}

            engineering = records[356]['cases'] if n == 256 else records[381]['targets'][0]['engineering']
            long_controls = records[356]['long_cases'] if n == 256 else records[381]['targets'][0]['long']
            for ci, control in enumerate(engineering):
                natural_sha = control['generation_rows'][0]['output_sha256'] if n == 256 else control['natural_sha256']
                ids = control['generated_ids']
                for profile in (0, 1):
                    target['controls'].append(complete(control['source_ids'], control['decoder_ids'], f'n{n}.engineering{ci}.p{profile}', control['native_sha256'], natural_sha, ids, 16, 32098, profile))
            for ci, control in enumerate(long_controls):
                target['controls'].append(complete(control['source_ids'], control['decoder_ids'], f'n{n}.long{ci}', control['native_sha256']))
            for bi, book in enumerate(cohort['items']):
                for ci, case in enumerate(book['cases']):
                    stage = f'n{n}.book{bi}.case{ci}'; observed = quality['books'][bi]['cases'][ci]
                    entry = complete(case['source_ids'], case['decoder_ids'], stage, observed['native_output_sha256'], observed['generation']['native_generation_sha256'], observed['generation']['native']['generated_ids'])
                    entry.update({'book': bi, 'case': ci}); target['quality_bridges'].append(entry)
                if bi % 4 == 3: print(json.dumps({'n': n, 'completed_cases': len(target['quality_bridges']), 'seconds': time.monotonic() - start}), flush=True)
            assert len(target['quality_bridges']) == 96 and before == (payload.stat().st_size, payload.stat().st_mtime_ns)
            target['passed'] = True
        result['gates'] = {'exact374_math_and_earlier_engine_paths': True, 'negative_actual_placement_fault_detected': True,
                           'both_full_artifact_spec_identities': True, 'both_engineering_profile_and_long_complete_reference_bytes': True,
                           'ALL192_teacher_and_own_natural_complete_quality_bytes_exact': True, 'all_actual_three_worker_process_event_readbacks': True}
        result['resource'] = {'main_seconds_excluding_imports': time.monotonic() - start, 'maximum_checked_combined_rss_bytes': maximum}
        result['decision'] = 'both_source_scoped_quality_inherited_three_workers_requires_separate_cost_and_accepted_rate'
        result['scope'] = 'NEW team size3 on0,2,4; SAME weights/math, both qualified original sources full quality outputs byte-exact. No unchanged383 retry or performance claim; three-worker cost/rate separate. No physical cache/DRAM causality or additional useful n/family proof.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': digest(args.out), 'gates': result['gates'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'stage': stage, 'error': repr(error), 'seconds': time.monotonic() - start, 'maximum_checked_combined_rss_bytes': maximum})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
