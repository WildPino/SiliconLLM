"""One frozen matched original128/256 whole-generation component diagnosis."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
BASE = ROOT / 'benchmarks/native_expert_scaling'
OUT = ROOT / 'results/native_expert_scaling/meth458_switch_matched_whole_cost'
PROTOCOL = DOC / 'METH_458_SWITCH_MATCHED_WHOLE_COST_PROTOCOL_20261005.md'
RAW = DOC / 'meth458_switch_matched_whole_cost_result.json'
CONFIG = {
    128: {'baseline': (391, 'b738cd25449a4903595d1c33e6257b8de8267367cf6d685418a29db186b8f5d8'),
          'quality': (387, '539275a209c8b8a5680c388ec9130012157a681ed382ba1a4bf49ab8cccb9d4c'),
          'manifest': (382, '96b3ffd09b0bdd42d7c990006e8744e39b5e1f32b7ae40aaf3b3dca98bf4db78'),
          'numeric': (389, 'fd12cfc2809fb42f52faef0b442781154c30ba27576f264c5f693bc30ec91dd2'),
          'cost': (390, '7a09d4aaa54af2a1455ef4153fa21461f661dec8e0d22b252c944c0d34f7e44b'),
          'threads': 3, 'affinity': [0, 2, 4], 'totals': [96, 1142, 662]},
    256: {'baseline': (376, '5986695ed0966f110c72129b058109d028b9f4bb90c23b763d57d7a62507bea5'),
          'quality': (363, 'ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'),
          'manifest': (362, 'c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'),
          'numeric': (374, '4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'),
          'cost': (375, '28b535b6a01ba67fcf45aaa008d8cf033d4f4cbc1aefcdc17c261a8749244375'),
          'threads': 6, 'affinity': [0, 2, 4, 6, 8, 10], 'totals': [81, 895, 490]},
}
NAMES = {391: 'base128_accepted_rate', 387: 'base128_multi_span_quality',
         382: 'multi_span_manifest', 389: 'three_workers_contract', 390: 'three_workers_cost',
         376: 'physical_workers_accepted_rate', 363: 'all_a16_multi_span_quality',
         362: 'multi_span_manifest', 374: 'physical_workers_contract', 375: 'physical_workers_cost'}
KINDS = ['dense_control_matrices', 'sparse_expert_WI_WO_matrices', 'router_score_matrices', 'head_matrix']
PHASES = ['encoder_plus_cross_KV', 'decoder_greedy_stop']
TIMES = ['encoder_seconds', 'cross_kv_seconds', 'decode_greedy_seconds', 'full_generation_seconds']
LOGICAL = ['calls', 'code_bytes', 'f32_bytes', 'scale_bytes']
STOPS = ['stopped_on_EOS', 'stopped_on_closing', 'stopped_at_cap']


def committed(path):
    path = Path(path); rel = path.resolve().relative_to(ROOT).as_posix()
    expected = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters',
                                        f'--path={rel}', f'HEAD:{rel}'], cwd=ROOT)
    assert path.read_bytes() == expected, ('physical_HEAD', rel)


def write_new(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(data)


def jobs():
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}; preserved = []
    for process in psutil.process_iter(['name', 'cmdline']):
        if process.pid in own:
            continue
        name = (process.info['name'] or '').lower(); argv = process.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            preserved.append(process.pid); continue
        assert not (name.startswith('python') or (name.startswith('meth') and name.endswith('.exe'))), ('concurrent_job', process.pid, name)
    return preserved


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
        'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert args.out.resolve() == RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and not OUT.exists()
    assert sys.flags.optimize == 0 and os.name == 'nt' and ctypes.sizeof(MemoryCounters) == 72
    start = time.monotonic(); numeric_start = None; stage = 'fresh_bindings'; completed_bytes = 0
    parent_peak = native_peak = hashed = 0; child = None; source_records = {}
    result = {'experiment': 'METH458-matched-whole-original128-256-cost', 'sources': {}, 'commands': [],
              'retained_record_sha256': {}, 'helper_sha256': {}, 'reused_output_inventory': [], 'output_inventory': []}
    memory_api = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    memory_api.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]; memory_api.restype = wintypes.BOOL
    def native_memory(process):
        counter = MemoryCounters(); counter.cb = ctypes.sizeof(counter)
        assert memory_api(wintypes.HANDLE(int(process._handle)), ctypes.byref(counter), counter.cb), ('GetProcessMemoryInfo', ctypes.get_last_error())
        return {'working_set_bytes': int(counter.WorkingSetSize), 'peak_working_set_bytes': int(counter.PeakWorkingSetSize)}
    def guard(process=None, prefix=None, child_started=None):
        nonlocal parent_peak, native_peak
        observation = None
        info = psutil.Process().memory_info(); parent_peak = max(parent_peak, info.rss, getattr(info, 'peak_wset', 0))
        if process is not None:
            observation = native_memory(process)
            native_peak = max(native_peak, observation['peak_working_set_bytes'])
        elapsed = time.monotonic() - start
        assert parent_peak <= 1 << 30 and parent_peak + native_peak <= 16 << 30, 'parent1GiB_conservative_sum16GiB'
        assert elapsed <= 1800 and (numeric_start is not None or elapsed <= 300), 'admission300_total1800_seconds'
        assert numeric_start is None or time.monotonic() - numeric_start <= 1500, 'numeric1500_seconds'
        if child_started is not None:
            assert time.monotonic() - child_started <= 60, 'one_child60_seconds'
        live_bytes = sum(p.stat().st_size for p in OUT.glob(prefix.name + '.*') if p.is_file()) if prefix is not None else 0
        assert live_bytes <= 128 << 20 and completed_bytes + live_bytes <= 8 << 30, 'child128MiB_all_outputs8GiB'
        return observation
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20):
                h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    def bound_record(number, expected):
        suffix = '' if number in (362, 382) else '_result'
        path = DOC / f'meth{number}_switch_{NAMES[number]}{suffix}.json'
        committed(path); assert sha(path) == expected, ('bound_record', number)
        result['retained_record_sha256'][path.name] = expected
        return json.loads(path.read_text(encoding='utf-8'))
    def inventory(path, reused=False):
        path = Path(path); item = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
        result['reused_output_inventory' if reused else 'output_inventory'].append(item)
        return item['sha256']
    try:
        result['git_head_before_execution'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'], cwd=ROOT)
        for path in (Path(__file__), PROTOCOL):
            committed(path); result['helper_sha256'][str(path)] = sha(path)
        for name, expected in [
            ('meth457_switch_matched_whole_cost_result.failure.json', 'd5923940e18e7a127b99959e30694ebd7ebfa470c563a09204f023ade5c7ba9a'),
            ('meth456_switch_output_tile_cost_result.json', 'cb7b9be5af6da6ce7fcbcbc058142ae6e8af2dda97c162727d44e9b591020977'),
            ('RETENTION_456_20261005.json', '37580c7af58d09b82d41c361bc4b0e4c1b7dc300131b8d03f8fd52f9e93d6eb9')]:
            committed(DOC / name); assert sha(DOC / name) == expected; result['retained_record_sha256'][name] = expected
        prior = json.loads((DOC / 'meth456_switch_output_tile_cost_result.json').read_text(encoding='utf-8'))
        assert all(prior['apparatus_gates'].values()) and not all(prior['feasibility_gates'].values())
        for item in prior['output_inventory']:
            assert Path(item['path']).stat().st_size == item['bytes'] and inventory(item['path'], True) == item['sha256']
        engine = ROOT / 'benchmarks/phase60/engine.c'; committed(engine)
        result['preserved_engine_sha256'] = sha(engine)
        assert result['preserved_engine_sha256'] == '54194c36b62571df014e8ebc251147ddd37436b5fb81c75db76f6fd4d8da0414'
        for n, config in CONFIG.items():
            records = {key: bound_record(*config[key]) for key in ('baseline', 'quality', 'manifest', 'numeric', 'cost')}
            source_records[n] = records; baseline = records['baseline']; quality = records['quality']; manifest = records['manifest']; numeric = records['numeric']
            assert all(all(record['gates'].values()) for record in (baseline, quality, manifest, numeric))
            assert baseline['native_threads'] == config['threads'] and baseline['native_process_affinity'] == config['affinity']
            assert baseline['artifact'] == quality['artifact'] and len(baseline['cases']) == len(baseline['commands']) == 96 and len(manifest['items']) == 24
            for key in ('quality', 'manifest', 'numeric', 'cost'):
                number, expected = config[key]
                assert baseline[f'{key}{number}_sha256'] == expected
            artifact = baseline['artifact']; payload = Path(artifact['payload']); spec = Path(artifact['manifest'])
            assert payload.stat().st_size == artifact['bytes'] and sha(payload) == artifact['sha256'] and sha(spec) == artifact['manifest_sha256']
            binary = Path(baseline['compile']['argv'][-1]); compiler = Path(baseline['compile']['argv'][0]); dll = binary.parent / 'libomp.dll'
            assert sha(binary) == baseline['compile']['binary_sha256'] == numeric['compile']['binary_sha256']
            assert sha(compiler) == baseline['compile']['compiler_sha256'] and sha(dll) == baseline['compile']['runtime_sha256']
            assert baseline['runtime_environment'] == numeric['runtime_environment']
            if n == 128:
                assert baseline['runtime_environment'] == quality['runtime_environment']
            else:
                assert 'runtime_environment' not in quality, 'historical363_schema_absence_must_be_explicit'
            artifact_stat = [payload.stat().st_size, payload.stat().st_mtime_ns]
            totals = [0, 0, 0]
            for index, (case, command) in enumerate(zip(baseline['cases'], baseline['commands'], strict=True)):
                bi, ci = divmod(index, 4); accepted = quality['books'][bi]['cases'][ci]['generation']['native']
                expected = quality['books'][bi]['cases'][ci]['generation']['native_generation_sha256']
                ids = manifest['items'][bi]['cases'][ci]['source_ids']; assert len(ids) == 29 and case['book'] == bi and case['case'] == ci
                assert command['argv'][:4] == [str(binary), '--generate', str(spec), ','.join(map(str, ids))]
                assert command['argv'][5:] == [str(config['threads']), '64', '32095', '0', '1', '3', '0'] and command['returncode'] == 0 and command['actual_affinity'] == config['affinity']
                assert [r['repetition'] for r in case['rows']] == [-1, 0, 1, 2]
                assert case['healthy_accepted'] == accepted['healthy_complete_nonempty_fields'] and case['generated_tokens'] == accepted['generated_tokens'] and case['prose_tokens'] == accepted['prose_tokens']
                for row in case['rows']:
                    path = Path(command['argv'][4] + f'.{row["repetition"]}.bin')
                    assert inventory(path, True) == row['output_sha256'] == expected and row['generated_ids'] == accepted['generated_ids']
                    assert row['profile'] == 0 and row['threads'] == config['threads'] and row['source_tokens'] == 29
                for kind in ('stdout', 'stderr'):
                    assert inventory(command['argv'][4] + f'.{kind}.log', True) == command[f'{kind}_sha256']
                if case['healthy_accepted']:
                    totals = [totals[0] + 1, totals[1] + case['generated_tokens'], totals[2] + case['prose_tokens']]
            assert totals == config['totals']
            result['sources'][str(n)] = {'artifact': artifact, 'artifact_stat_before': artifact_stat, 'compile': baseline['compile'],
                'runtime_environment': baseline['runtime_environment'], 'threads': config['threads'], 'affinity': config['affinity'],
                'accepted_cases': totals[0], 'accepted_generated_tokens': totals[1], 'accepted_prose_tokens': totals[2],
                'rejected_cases_time_charged': 96 - totals[0], 'historical_accepted_full_rate': baseline['accepted_full_rate'], 'cases': []}
        old_numeric = source_records[256]['numeric']
        for name, expected in old_numeric['source_sha256'].items():
            path = BASE / name; committed(path); assert sha(path) == expected; result['helper_sha256'][str(path)] = expected
        for suffix in ('.c', '_entry.c', '_cost_entry.c'):
            old = BASE / f'meth374_switch_physical_workers{suffix}'; new = BASE / f'meth388_switch_three_workers{suffix}'; committed(new)
            reversed_source = new.read_text(encoding='utf-8').replace('meth388_switch_three_workers', 'meth374_switch_physical_workers').replace('meth388_contract_entry', 'meth374_contract_entry').replace('meth388_forced_entry', 'meth374_forced_entry').replace('meth388_switch_thread_binding.h', 'meth374_switch_thread_binding.h').replace('threads==1||threads==3||threads==6', 'threads==1||threads==6')
            assert reversed_source == old.read_text(encoding='utf-8'); result['helper_sha256'][str(new)] = sha(new)
        path = BASE / 'meth388_switch_thread_binding.h'; committed(path)
        assert path.read_text(encoding='utf-8').replace('threads==1||threads==3||threads==6', 'threads==1||threads==6') == (BASE / 'meth374_switch_thread_binding.h').read_text(encoding='utf-8')
        result['helper_sha256'][str(path)] = sha(path)
        for name in ('meth359_switch_cost_topology.py', 'meth324_switch_reference.py'):
            path = BASE / name; committed(path); result['helper_sha256'][str(path)] = sha(path)
        import meth359_switch_cost_topology as topology
        result['fresh_topology'] = topology.physical_topology()
        assert result['fresh_topology']['process_allowed_logical_processors'] == list(range(12)) and result['fresh_topology']['selected_one_logical_per_physical_core'] == CONFIG[256]['affinity']
        result['preserved_daemons_before'] = jobs()
        result['host'] = {'physical_cpus': psutil.cpu_count(logical=False), 'logical_cpus': psutil.cpu_count(),
                          'RAM_bytes': psutil.virtual_memory().total, 'available_RAM_before_bytes': psutil.virtual_memory().available,
                          'disk_free_before_bytes': shutil.disk_usage(ROOT).free}
        assert result['host']['available_RAM_before_bytes'] >= 16 << 30 and result['host']['disk_free_before_bytes'] >= 10 << 30
        guard(); OUT.mkdir(parents=True); numeric_start = time.monotonic()
        print(json.dumps({'admission_seconds': numeric_start - start, 'next': '384_sequential_native_commands'}), flush=True)
        for n, config in CONFIG.items():
            baseline = source_records[n]['baseline']; source = result['sources'][str(n)]; binary = Path(source['compile']['argv'][-1])
            env = {k: v for k, v in os.environ.items() if not k.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_WORKER_BINDING_'))}
            env.update(source['runtime_environment']); env['OMP_NUM_THREADS'] = str(config['threads'])
            for index, case in enumerate(baseline['cases']):
                bi, ci = case['book'], case['case']; manifest_case = source_records[n]['manifest']['items'][bi]['cases'][ci]
                observed = {'book': bi, 'case': ci, 'source_id': case['source_id'], 'healthy_accepted': case['healthy_accepted'],
                            'generated_tokens': case['generated_tokens'], 'prose_tokens': case['prose_tokens'], 'mode_order': [index % 2, 1 - index % 2], 'modes': {}}
                source['cases'].append(observed)
                for mode in observed['mode_order']:
                    stage = f'n{n}_book{bi}_case{ci}_profile{mode}'; label = f'n{n}.book{bi}.case{ci}.profile{mode}'; prefix = OUT / label
                    argv = [str(binary), '--generate', source['artifact']['manifest'], ','.join(map(str, manifest_case['source_ids'])), str(prefix), str(config['threads']), '64', '32095', str(mode), '1', '3', '0']
                    command = {'n': n, 'book': bi, 'case': ci, 'profile': mode, 'argv': list(argv), 'maximum_polled_peak_working_set_bytes': 0}; result['commands'].append(command)
                    child_started = time.monotonic()
                    with Path(str(prefix) + '.stdout.log').open('xb') as stdout, Path(str(prefix) + '.stderr.log').open('xb') as stderr:
                        child = subprocess.Popen(list(argv), stdout=stdout, stderr=stderr, env=env, cwd=ROOT)
                        command['pid'] = child.pid; process = psutil.Process(child.pid); process.cpu_affinity(config['affinity'])
                        command['actual_affinity'] = process.cpu_affinity(); assert command['actual_affinity'] == config['affinity']
                        while child.poll() is None:
                            sample = guard(child, prefix, child_started)
                            command['maximum_polled_peak_working_set_bytes'] = max(command['maximum_polled_peak_working_set_bytes'], sample['peak_working_set_bytes'])
                            time.sleep(.25)
                        command['returncode'] = child.returncode; command['terminal_memory'] = native_memory(child)
                        assert command['terminal_memory']['peak_working_set_bytes'] > 0 and command['terminal_memory']['peak_working_set_bytes'] >= command['maximum_polled_peak_working_set_bytes']
                        native_peak = max(native_peak, command['terminal_memory']['peak_working_set_bytes']); guard(None, prefix, child_started)
                    assert child.returncode == 0, (stage, child.returncode)
                    child = None
                    rows = [json.loads(line) for line in Path(str(prefix) + '.stdout.log').read_text(encoding='utf-8').splitlines()]
                    assert [row['repetition'] for row in rows] == [-1, 0, 1, 2]
                    observed['modes'][str(mode)] = rows
                    for row, expected_row in zip(rows, case['rows'], strict=True):
                        assert row['threads'] == config['threads'] and row['profile'] == mode and row['source_tokens'] == 29 and row['closing_id'] == 32095
                        assert row['generated_ids'] == expected_row['generated_ids'] and row['actual_generated_tokens'] == case['generated_tokens']
                        assert all(row[key] == expected_row[key] for key in STOPS)
                        assert row['worker_physical_cores'] == 6 and [v['slot'] for v in row['worker_affinity']] == list(range(config['threads']))
                        assert [v['actual_mask'] for v in row['worker_affinity']] == [1 << cpu for cpu in config['affinity']]
                        assert len({v['windows_thread_id'] for v in row['worker_affinity']}) == config['threads'] and all(v['group'] == 0 for v in row['worker_affinity'])
                        assert set(v['slot'] for v in row['worker_binding_events']) == set(range(config['threads']))
                        assert all(v['actual_mask'] == 1 << config['affinity'][v['slot']] and v['group'] == 0 for v in row['worker_binding_events'])
                        assert all(math.isfinite(row[key]) and row[key] > 0 for key in TIMES + ['load_seconds', 'first_token_seconds'])
                        assert len(row['step_seconds']) == case['generated_tokens'] and all(math.isfinite(v) and v > 0 for v in row['step_seconds'])
                        assert abs(sum(row[key] for key in TIMES[:3]) - row['full_generation_seconds']) <= 1e-7
                        assert len(row['counters']) == 2 and all(len(v) == 4 for v in row['counters'])
                        for phase in range(2):
                            for kind in range(4):
                                counter = row['counters'][phase][kind]; expected_counter = expected_row['counters'][phase][kind]
                                assert all(counter[key] == expected_counter[key] for key in LOGICAL)
                                assert math.isfinite(counter['matrix_seconds']) and counter['matrix_seconds'] >= 0
                                assert mode == 1 or counter['matrix_seconds'] == 0
                            phase_time = row['encoder_seconds'] + row['cross_kv_seconds'] if phase == 0 else row['decode_greedy_seconds']
                            assert sum(c['matrix_seconds'] for c in row['counters'][phase]) <= phase_time + 1e-7
                        row['output_sha256'] = inventory(str(prefix) + f'.{row["repetition"]}.bin')
                        assert row['output_sha256'] == expected_row['output_sha256'], 'complete_encoder_decoder_logits_routes_BYTES_changed'
                    for kind in ('stdout', 'stderr'):
                        command[f'{kind}_sha256'] = inventory(str(prefix) + f'.{kind}.log')
                    completed_bytes = sum(item['bytes'] for item in result['output_inventory']); guard()
                if ci == 3:
                    print(json.dumps({'completed_source': n, 'completed_book': bi, 'commands': len(result['commands']), 'seconds': time.monotonic() - start}), flush=True)
            assert len(source['cases']) == 96
            assert source['artifact_stat_before'] == [Path(source['artifact']['payload']).stat().st_size, Path(source['artifact']['payload']).stat().st_mtime_ns]
        stage = 'fixed_aggregation'; admitted_all = True
        for n, source in result['sources'].items():
            mode_summaries = {}; book_ratios = []
            for mode in (0, 1):
                measured = [r for c in source['cases'] for r in c['modes'][str(mode)] if r['repetition'] >= 0]
                book_times = [sum(r['full_generation_seconds'] for c in source['cases'] if c['book'] == bi for r in c['modes'][str(mode)] if r['repetition'] >= 0) / 3 for bi in range(24)]
                total = sum(book_times); median_total = sum(statistics.median(r['full_generation_seconds'] for r in c['modes'][str(mode)] if r['repetition'] >= 0) for c in source['cases'])
                cube = [[sum(r['counters'][phase][kind]['matrix_seconds'] for r in measured) / 3 for kind in range(4)] for phase in range(2)]
                phases = {key: sum(r[key] for r in measured) / 3 for key in TIMES}
                kind_total = [sum(cube[phase][kind] for phase in range(2)) for kind in range(4)]
                phase_residual = [phases[TIMES[0]] + phases[TIMES[1]] - sum(cube[0]), phases[TIMES[2]] - sum(cube[1])]
                mode_summaries[str(mode)] = {'ALL_case_mean_full_seconds': total, 'ALL_case_median_full_seconds': median_total,
                    'book_ALL_case_mean_full_seconds': book_times, 'accepted_ordinary_rate_mean': source['accepted_generated_tokens'] / total,
                    'accepted_prose_rate_mean': source['accepted_prose_tokens'] / total,
                    'accepted_ordinary_rate_case_median': source['accepted_generated_tokens'] / median_total,
                    'accepted_prose_rate_case_median': source['accepted_prose_tokens'] / median_total,
                    'phase_mean_seconds': phases, 'matrix_mean_seconds_by_phase_kind': cube, 'matrix_mean_seconds_by_kind': dict(zip(KINDS, kind_total, strict=True)),
                    'residual_mean_seconds_by_phase': dict(zip(PHASES, phase_residual, strict=True)),
                    'residual_mean_seconds': total - sum(kind_total),
                    'repetition_ALL_case_full_seconds': [sum(c['modes'][str(mode)][rep + 1]['full_generation_seconds'] for c in source['cases']) for rep in range(3)]}
            ratio = mode_summaries['1']['ALL_case_mean_full_seconds'] / mode_summaries['0']['ALL_case_mean_full_seconds']
            book_ratios = [b / a for a, b in zip(mode_summaries['0']['book_ALL_case_mean_full_seconds'], mode_summaries['1']['book_ALL_case_mean_full_seconds'], strict=True)]
            gates = {'aggregate_mean_ratio_in_0p90_1p10': .90 <= ratio <= 1.10, 'ALL24_book_mean_ratios_in_0p85_1p15': all(.85 <= r <= 1.15 for r in book_ratios)}
            admitted = all(gates.values()); admitted_all &= admitted
            source['mode_summaries'] = mode_summaries; source['profiler_admissibility'] = {'aggregate_mean_ratio': ratio, 'book_mean_ratios': book_ratios, 'gates': gates, 'admitted': admitted}
            source['admitted_decomposition'] = None
            if admitted:
                profile = mode_summaries['1']; full = profile['ALL_case_mean_full_seconds']; expert = profile['matrix_mean_seconds_by_kind'][KINDS[1]]; f = expert / full
                assert 0 < f < 1
                fractions = {key: value / full for key, value in profile['matrix_mean_seconds_by_kind'].items()}; fractions['residual'] = profile['residual_mean_seconds'] / full
                targets = []
                for metric, key in [('ordinary', 'accepted_ordinary_rate_mean'), ('prose', 'accepted_prose_rate_mean')]:
                    rate = profile[key]
                    for target in (50, 100):
                        required = (rate / target - (1 - f)) / f
                        targets.append({'metric': metric, 'target_rate': target, 'same_profile1_mean_rate': rate,
                            'maximum_expert_matrix_cost_multiplier_r': required, 'feasible_by_expert_matrices_alone': required >= 0,
                            'expert_elimination_rate_ceiling': rate / (1 - f)})
                source['admitted_decomposition'] = {'whole_cost_fractions': fractions, 'expert_matrix_fraction_f': f,
                    'expert_elimination_speedup_ceiling': 1 / (1 - f), 'target_constraints': targets,
                    'decision': 'prioritize_useful_selective_expert_response_geometry' if f >= .5 else 'prioritize_dominant_whole_component_before_another_WI_layout'}
        result['preserved_daemons_after'] = jobs(); guard()
        actual_files = {str(p) for p in OUT.iterdir() if p.is_file()}; assert actual_files == {item['path'] for item in result['output_inventory']}
        assert len(result['commands']) == 384 and len(result['output_inventory']) == 2304 and len(actual_files) == 2304
        result['apparatus_gates'] = {'all_fresh_bindings_source_math_runtime_originals_exact': True, 'ALL1536_warm_measured_complete_output_hashes_and_ids_exact_original': True,
            'all_actual_worker_process_affinity_readbacks_exact': True, 'all_logical_matrix_counters_and_stop_context_exact_baseline': True,
            'all_phase_counter_time_partitions_finite_nonoverlap': True, 'accepted_totals_and_ALL_rejected_time_charged': True,
            'all384_terminal_peak_queries_and_resources_pass': True, 'all2304_outputs_retained_in_inventory': True}
        result['phase_labels'] = PHASES; result['matrix_kind_labels'] = KINDS
        result['resource'] = {'main_seconds_excluding_imports': time.monotonic() - start, 'admission_seconds': numeric_start - start,
            'numeric_and_aggregation_seconds': time.monotonic() - numeric_start, 'parent_peak_working_set_bytes': parent_peak,
            'native_peak_working_set_bytes': native_peak, 'conservative_sum_of_nonconcurrent_peaks_bytes': parent_peak + native_peak,
            'output_bytes': completed_bytes, 'bytes_hashed': hashed, 'available_RAM_end_bytes': psutil.virtual_memory().available,
            'native_peak_source': 'GetProcessMemoryInfo PeakWorkingSetSize on retained Popen handle including terminal query'}
        result['decision'] = 'matched_whole_economic_constraints_admitted_each_source' if admitted_all else 'profiler_perturbation_failure_no_fraction_inference_for_failed_source_no_retry'
        result['scope'] = 'Consumed original same-quality short English infilling cost diagnosis ONLY. No new quality/data/model/kernel/compiler/rounding/fit/GPU/DRAM counters. Different original cores/workers are separate within-source contrasts, not causal E-only scaling. Full timer includes encoder/crossKV/cached decode/head/greedy/stop; excludes startup/team setup/load/ID tokenization/serialization/cleanup. Phase0 matrix cube includes encoder AND crossKV. Matrix timers include A16/OpenMP, exclude ReLU/probability combination/norm/allocator/router softmax. Residual aggregates those and other nonmatrix/control costs plus unallocated timer overhead. Conditional bounds use profile1 arithmetic mean ALL-case denominator and unchanged healthy numerators; median and historical rates never enter this algebra. Admitted profile perturbation permits diagnosis, not exact uninstrumented fraction or measured improved speed. No456 local multiplier substituted.'
        write_new(RAW, result)
        print(json.dumps({'sha256': sha(RAW), 'apparatus_gates': result['apparatus_gates'], 'decision': result['decision'],
            'sources': {n: {'admission': s['profiler_admissibility'], 'decomposition': s['admitted_decomposition']} for n, s in result['sources'].items()}, 'resource': result['resource']}), flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.kill(); child.wait()
        result.update({'stage': stage, 'error': repr(error), 'main_seconds_excluding_imports': time.monotonic() - start,
                       'parent_peak_working_set_bytes': parent_peak, 'native_peak_working_set_bytes': native_peak})
        write_new(RAW.with_suffix('.failure.json'), result)
        raise


if __name__ == '__main__':
    main()
