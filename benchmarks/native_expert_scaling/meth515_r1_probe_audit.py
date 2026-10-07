"""Independent515 literal-port/full-byte/control-clock audit; no C replay."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import meth515_r1_operations as O


def extract(text, signature):
    start = text.index(signature)
    cursor = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[cursor] == '{') - (text[cursor] == '}')
        cursor += 1
    return text[start:cursor]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-sha', required=True)
    ap.add_argument('--main-sha', required=True)
    args = ap.parse_args()
    ctx = O.Context('audit')
    try:
        b = ctx.admit(args.binding_sha)
        path = O.DOC / 'meth515_main_result.json'
        assert ctx.digest(path) == args.main_sha
        raw = json.loads(path.read_bytes())
        assert all(raw['gates'].values()) and raw['binding_sha256'] == args.binding_sha
        base = O.ROOT / 'benchmarks/native_expert_scaling'
        old = (base / 'meth365_switch_encoder_batches.c').read_text(encoding='utf8')
        port = (base / 'meth515_encoder_reuse.c').read_text(encoding='utf8')
        pieces = [extract(old, 'static void integer_dot4('), extract(old, 'static void mv_batch(').replace('static void mv_batch(', 'static void encoder_reuse_mv_batch('),
                  extract(old, 'static void feed_encoder(').replace('feed_encoder(', 'encoder_reuse_feed(').replace('mv_batch(', 'encoder_reuse_mv_batch('),
                  extract(old, 'static float *encode(').replace('mv_batch(b->self.o,', 'encoder_reuse_mv_batch(b->self.o,').replace('feed_encoder(', 'encoder_reuse_feed(')]
        expected = '/* METH515: literal365 missing encoder batches, qualified374 physical workers. */\n#define encode meth515_original_encode\n#include "meth374_switch_physical_workers.c"\n#undef encode\n\n' + '\n\n'.join(pieces) + '\n'
        assert port.replace('\r\n', '\n') == expected.replace('\r\n', '\n')
        for name in ['entry', 'cost_entry']:
            source = (base / f'meth374_switch_physical_workers_{name}.c').read_text(encoding='utf8')
            expected = source.replace('"meth374_switch_physical_workers.c"', '"meth515_encoder_reuse.c"').replace('"meth374_switch_physical_workers_cost_entry.c"', '"meth515_encoder_reuse_cost_entry.c"').replace('meth374_forced_entry', 'meth515_forced_entry')
            assert (base / f'meth515_encoder_reuse_{name}.c').read_text(encoding='utf8') == expected
        for row in raw['output_inventory']:
            assert Path(row['path']).stat().st_size == row['bytes'] and ctx.digest(row['path']) == row['sha256']
        assert ctx.digest(O.ROOT / 'results/native_expert_scaling/meth515_main/meth515_encoder_reuse.exe') == raw['binary_sha256']
        commands = raw['commands']
        assert len(commands) == 7 and sum(v['native'] for v in commands) == 6 and all(v['returncode'] == 0 for v in commands)
        import psutil
        for v in commands:
            assert v['affinity_readback'] == ([0, 2, 4, 6, 8, 10] if v['native'] else [10])
            assert v['terminal_memory']['peak_working_set_bytes'] > 0
            try:
                p = psutil.Process(v['process_instance']['pid'])
                assert abs(p.create_time() - v['process_instance']['create_time_unix']) > .002
            except psutil.NoSuchProcess:
                pass
        ratios = []
        means = []
        wires = 0
        for expected_case, observed in zip(b['cases'], raw['cases']):
            book, index = expected_case['book'], expected_case['index']
            assert (book, index) == (observed['book'], observed['index'])
            prefix = O.ROOT / f'results/native_expert_scaling/meth515_main/book{book}.case{index}'
            teacher = Path(str(prefix) + '.candidate.teacher.0.bin')
            assert teacher.read_bytes() == Path(expected_case['expected_teacher']['path']).read_bytes()
            wires += 1
            arms = {}
            for arm in ['source', 'candidate']:
                stdout = Path(str(prefix) + '.' + arm + '.generation.stdout')
                rows = [json.loads(v) for v in stdout.read_text(encoding='utf8').splitlines()]
                assert rows == observed['arms'][arm] and [v['repetition'] for v in rows] == [-1, 0, 1, 2]
                arms[arm] = rows
                for v in rows:
                    path = Path(str(prefix) + f".{arm}.generation.{v['repetition']}.bin")
                    data = path.read_bytes()
                    assert data == Path(expected_case['expected_generation']['path']).read_bytes()
                    magic, s, t, d, el, dl, vocab, nr = struct.unpack_from('<8s7I', data)
                    assert (magic, s, d, el, dl, vocab, nr) == (b'SWR32O01', 29, 768, 12, 12, 32128, 174 + 6 * t)
                    assert t == len(v['generated_ids']) == len(expected_case['expected_ids']) and v['generated_ids'] == expected_case['expected_ids']
                    assert [x['actual_mask'] for x in v['worker_affinity']] == [1, 4, 16, 64, 256, 1024]
                    assert v['threads'] == v['worker_physical_cores'] == 6
                    for key in ['load_seconds', 'encoder_seconds', 'cross_kv_seconds', 'decode_greedy_seconds', 'full_generation_seconds', 'first_token_seconds']:
                        assert math.isfinite(v[key]) and v[key] >= 0
                    assert abs(v['full_generation_seconds'] - v['encoder_seconds'] - v['cross_kv_seconds'] - v['decode_greedy_seconds']) <= 2e-9
                    wires += 1
                full = [v['full_generation_seconds'] for v in rows[1:]]
                ratios.append(max(full) / min(full))
            for s, c in zip(arms['source'], arms['candidate']):
                assert s['counters'] == c['counters']
            mean = {arm: {key: math.fsum(v[key] for v in arms[arm][1:]) / 3 for key in observed['ratio']} for arm in arms}
            for key, r in observed['ratio'].items():
                assert abs(r - mean['candidate'][key] / mean['source'][key]) <= 2e-14
            means.append(mean)
        encoder_ratio = math.fsum(v['candidate']['encoder_seconds'] for v in means) / math.fsum(v['source']['encoder_seconds'] for v in means)
        assert abs(encoder_ratio - raw['summary']['encoder_mean_ratio']) <= 2e-14 and ratios == raw['summary']['repeat_ratios']
        feasible = {'pilot_encoder_ratio_supplies_retained489_necessary_gap': encoder_ratio <= b['encoder_ratio_necessary_on_retained489'],
                    'ALL2_candidate_whole_mean_not_slower': all(v['candidate']['full_generation_seconds'] <= v['source']['full_generation_seconds'] for v in means),
                    'ALL4_three_repeat_whole_ratio_le1p10': all(v <= 1.1 for v in ratios)}
        assert feasible == raw['summary']['feasibility'] and all(feasible.values()) == raw['summary']['eligible_larger_same_recipe_cost_stage'] and wires == 18
        ctx.r['gates'].update(literal365_functions_and_exact374_inclusions_and_entries=True, ALL18_complete_wires_original363_byte_identical=True,
                              ALL7_closed_children_terminal_peaks_AND_affinities=True, ALL_timing_counter_arithmetic_AND_prospective_decision_exact=True)
        ctx.finish({'main_sha256': args.main_sha, 'summary': {'encoder_mean_ratio': encoder_ratio, 'feasibility': feasible,
                    'eligible_larger_same_recipe_cost_stage': all(feasible.values()), 'full_wires_audited': wires, 'new_native_calls': 0,
                    'scope': 'Independent literal-source/full-byte and retained clock audit; no C or original model replay.'}, 'goal_complete': False})
    except BaseException:
        ctx.fail()
        raise


if __name__ == '__main__':
    main()
