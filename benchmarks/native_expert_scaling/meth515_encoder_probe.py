"""One changed515 implementation probe, exact wires and matched six-core clocks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import meth515_operations as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-sha', required=True)
    args = ap.parse_args()
    ctx = O.Context('main')
    try:
        b = ctx.admit(args.binding_sha)
        env = os.environ.copy()
        env.update(b['native_environment'], OMP_NUM_THREADS='6')
        env.pop('OMP_PROC_BIND', None)
        binary = ctx.out / 'meth515_encoder_reuse.exe'
        argv = [b['compiler']['path'], *b['compiler_flags'], str(O.ROOT / 'benchmarks/native_expert_scaling/meth515_encoder_reuse_entry.c'), '-o', str(binary)]
        ctx.run(argv, 'compile', env, native=False)
        binary_sha = ctx.digest(binary)
        shutil.copyfile(b['libomp']['path'], ctx.out / 'libomp.dll')
        assert ctx.digest(ctx.out / 'libomp.dll') == b['libomp']['sha256']
        snapshots = []
        for p in psutil_process_snapshot():
            if any(v in p['command'].lower() for v in ['meth51', 'meth48', 'meth36', 'llama-server', 'ollama']):
                snapshots.append(p)
        ctx.r['observed_processes_before_native'] = snapshots
        results = []
        all_wires = []
        for ordinal, case in enumerate(b['cases']):
            label = f"book{case['book']}.case{case['index']}"
            source = ','.join(map(str, case['source_ids']))
            prefix = ctx.out / (label + '.candidate.teacher')
            argv = [str(binary), b['manifest']['path'], source, ','.join(map(str, case['decoder_ids'])), str(prefix), '6', '0', '0', '1', '0']
            ctx.run(argv, label + '.candidate.teacher', env)
            wire = Path(str(prefix) + '.0.bin')
            assert ctx.digest(wire) == case['expected_teacher']['sha256']
            all_wires.append(str(wire))
            arms = {}
            for arm in (['source', 'candidate'] if ordinal == 0 else ['candidate', 'source']):
                path = b['baseline']['path'] if arm == 'source' else str(binary)
                prefix = ctx.out / (label + '.' + arm + '.generation')
                argv = [path, '--generate', b['manifest']['path'], source, str(prefix), '6', '64', '32095', '0', '1', '3', '0']
                log = ctx.run(argv, label + '.' + arm + '.generation', env)
                rows = [json.loads(line) for line in log.read_text(encoding='utf8').splitlines()]
                assert [v['repetition'] for v in rows] == [-1, 0, 1, 2]
                for v in rows:
                    wire = Path(str(prefix) + f".{v['repetition']}.bin")
                    assert ctx.digest(wire) == case['expected_generation']['sha256'] and v['generated_ids'] == case['expected_ids']
                    assert v['threads'] == v['worker_physical_cores'] == 6
                    assert [v['actual_mask'] for v in v['worker_affinity']] == [1, 4, 16, 64, 256, 1024]
                    all_wires.append(str(wire))
                arms[arm] = rows
            # Logical addressed bytes/calls remain unchanged despite physical reuse.
            for source_row, candidate_row in zip(arms['source'], arms['candidate']):
                assert source_row['counters'] == candidate_row['counters']
            result = {'book': case['book'], 'index': case['index'], 'arms': arms, 'ratio': {}}
            for metric in ['encoder_seconds', 'cross_kv_seconds', 'decode_greedy_seconds', 'full_generation_seconds']:
                s = statistics.mean(v[metric] for v in arms['source'] if v['repetition'] >= 0)
                c = statistics.mean(v[metric] for v in arms['candidate'] if v['repetition'] >= 0)
                result['ratio'][metric] = c / s
            results.append(result)
        encoder_source = sum(statistics.mean(v['encoder_seconds'] for v in c['arms']['source'][1:]) for c in results)
        encoder_candidate = sum(statistics.mean(v['encoder_seconds'] for v in c['arms']['candidate'][1:]) for c in results)
        ratio = encoder_candidate / encoder_source
        repeat_ratios = [max(v['full_generation_seconds'] for v in c['arms'][arm][1:]) / min(v['full_generation_seconds'] for v in c['arms'][arm][1:]) for c in results for arm in ['source', 'candidate']]
        feasibility = {'pilot_encoder_ratio_supplies_retained489_necessary_gap': ratio <= b['encoder_ratio_necessary_on_retained489'],
                       'ALL2_candidate_whole_mean_not_slower': all(c['ratio']['full_generation_seconds'] <= 1 for c in results),
                       'ALL4_three_repeat_whole_ratio_le1p10': all(v <= 1.10 for v in repeat_ratios)}
        summary = {'cases': 2, 'native_calls': 6, 'compile_calls': 1, 'exact_full_wires': len(all_wires), 'encoder_mean_ratio': ratio,
                   'required489_encoder_ratio': b['encoder_ratio_necessary_on_retained489'], 'repeat_ratios': repeat_ratios,
                   'feasibility': feasibility, 'eligible_larger_same_recipe_cost_stage': all(feasibility.values()),
                   'scope': 'Two consumed source29 probes, exact original wires; pilot relative clocks only. No accepted-rate/fresh quality/DRAM/useful-n claim.'}
        ctx.r['gates'].update(literal365_integration_build_and_runtime_identity=True, ALL2_teacher_AND16_natural_complete_wires_exact363=True,
                              ALL16_measured_and_warm_worker_masks_AND_IDs_exact=True, ALL_logical_counters_exact_and_all_children_exit0=True)
        ctx.finish({'binary_sha256': binary_sha, 'cases': results, 'summary': summary, 'goal_complete': False})
    except BaseException:
        ctx.fail()
        raise


def psutil_process_snapshot():
    import psutil
    rows = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            rows.append({'pid': p.pid, 'name': p.info['name'], 'create_time_unix': p.info['create_time'], 'command': ' '.join(p.info['cmdline'] or [])})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return rows


if __name__ == '__main__':
    main()
