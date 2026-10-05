"""One frozen exact common-input fanout primal and whole-cost comparison."""
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
import struct
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]; BASE = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT / 'results/native_expert_scaling/meth461_switch_common_input_cost'
RAW = DOC / 'meth461_switch_common_input_cost_result.json'
PROTOCOL = DOC / 'METH_461_SWITCH_COMMON_INPUT_COST_PROTOCOL_20261005.md'
PAIRS = (
    ('mv_batch(b->self.q,x,q,c.d,c.d,tokens);mv_batch(b->self.k,x,k,c.d,c.d,tokens);mv_batch(b->self.v,x,v,c.d,c.d,tokens);',
     '{Matrix fan[3]={b->self.q,b->self.k,b->self.v};float *outputs[3]={q,k,v};mv_fanout(fan,outputs,3,x,c.d,c.d,tokens);}'),
    ('mv_batch(m->dec[l].cross.k,encoder,a->crossk,c.d,c.d,tokens);mv_batch(m->dec[l].cross.v,encoder,a->crossv,c.d,c.d,tokens);',
     '{Matrix fan[2]={m->dec[l].cross.k,m->dec[l].cross.v};float *outputs[2]={a->crossk,a->crossv};mv_fanout(fan,outputs,2,encoder,c.d,c.d,tokens);}'),
    ('mv(b->self.q,x,q,c.d,c.d);mv(b->self.k,x,a->selfk+(size_t)position*c.d,c.d,c.d);mv(b->self.v,x,a->selfv+(size_t)position*c.d,c.d,c.d);',
     '{Matrix fan[3]={b->self.q,b->self.k,b->self.v};float *outputs[3]={q,a->selfk+(size_t)position*c.d,a->selfv+(size_t)position*c.d};mv_fanout(fan,outputs,3,x,c.d,c.d,1);}'),
)
DIRECT = {
    'meth458_switch_matched_whole_cost_result.json': '3762fd4e6c86a6ea7e64f54d4350cc2bd91761f2f856a06bb0a0a442f12d285f',
    'RETENTION_458_20261005.json': '65ec6408bc1de79d0a3bad869b57de7298285901c5c91ca502330250ea9ea8b0',
    'meth460_switch_common_input_admission_result.json': '6e1f921d05f08eba48e59c05d167420fc9160732ef3060617e23a4cdbf4646b0',
    'RETENTION_460_20261005.json': '51b658380511c9fb654c7f30e1b493ba95288feec234d69c24ef4227128f1e36',
}
TIMES = ('encoder_seconds', 'cross_kv_seconds', 'decode_greedy_seconds', 'full_generation_seconds')
STOPS = ('stopped_on_EOS', 'stopped_on_closing', 'stopped_at_cap')
LOGICAL = ('calls', 'code_bytes', 'scale_bytes', 'f32_bytes')


def committed(path):
    path = Path(path); rel = path.resolve().relative_to(ROOT).as_posix()
    assert path.read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', f'--path={rel}', f'HEAD:{rel}'], cwd=ROOT), ('physical_HEAD', rel)


def write_new(path, data):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode('utf-8'))


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
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
        'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def p95(values):
    values = sorted(values); position = .95 * (len(values) - 1); lower = math.floor(position); upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def control_fixture():
    shapes = ((2, 2, 17, 1), (3, 65, 17, 3), (3, 65, 768, 3), (2, 2, 4096, 1), (3, 65, 4096, 1))
    data = bytearray(b'MF461I01' + struct.pack('<I', 5)); expected = bytearray(b'MF461O01' + struct.pack('<I', 5))
    base_values = (32767., -32767., .5, -.5, 1.5, -1.5, 2.5, -2.5, 0.)
    for ci, (maps, rows, cols, tokens) in enumerate(shapes):
        data.extend(struct.pack('<4I', maps, rows, cols, tokens)); expected.extend(struct.pack('<4I', maps, rows, cols, tokens))
        x = []
        for token in range(tokens):
            if ci == 0:
                values = [0.] * cols
            elif ci == 3:
                values = [32767.] * cols
            elif ci == 4:
                values = [-32767.] * cols
            elif token == 0:
                values = [0.] * cols
            else:
                values = [(1 if token == 1 else 4) * base_values[j % len(base_values)] for j in range(cols)]
            x.append(values)
        data.extend(struct.pack(f'<{tokens * cols}f', *(value for row in x for value in row)))
        query_codes = []; query_scales = []
        for values in x:
            maximum = max(abs(v) for v in values); alpha = maximum / 32767 if maximum else 1.
            assert alpha in (1., 4.)
            # Fixture divisions are EXACT binary halves/integers, independently RNE.
            query_codes.append([max(-32767, min(32767, round(value / alpha))) for value in values]); query_scales.append(alpha)
        for map_index in range(maps):
            row_scales = [2. ** ((row + map_index) % 7 - 5) for row in range(rows)]
            weights = []
            for row in range(rows):
                weights.append([-128 if (row + map_index) % 3 == 0 else 127 if (row + map_index) % 3 == 1 else ((j * 37 + row * 11 + map_index) % 256 - 128) for j in range(cols)])
            data.extend(bytes(value & 255 for row in weights for value in row)); data.extend(struct.pack(f'<{rows}f', *row_scales))
            outputs = []
            for token in range(tokens):
                for row in range(rows):
                    dot = sum(w * code for w, code in zip(weights[row], query_codes[token], strict=True))
                    # Powers-of-two scales make F64 intermediates exact; final F32 RNE.
                    outputs.append((float(dot) * row_scales[row]) * query_scales[token])
            expected.extend(struct.pack(f'<{tokens * rows}f', *outputs))
    return bytes(data), bytes(expected)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert args.out.resolve() == RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and not OUT.exists()
    assert sys.flags.optimize == 0 and os.name == 'nt' and ctypes.sizeof(MemoryCounters) == 72
    start = time.monotonic(); numeric_start = None; stage = 'fresh_bindings'; child = None
    parent_peak = child_peak = compiler_descendant_sample_peak = hashed = completed_bytes = 0
    result = {'experiment': 'METH461-exact-shared-A16-fanout-primal-and-WHOLE-cost', 'helper_sha256': {}, 'retained_record_sha256': {},
              'commands': [], 'sources': {}, 'output_inventory': [], 'reused_output_inventory': []}
    memory_api = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    memory_api.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]; memory_api.restype = wintypes.BOOL
    def memory(process):
        counter = MemoryCounters(); counter.cb = ctypes.sizeof(counter)
        assert memory_api(wintypes.HANDLE(int(process._handle)), ctypes.byref(counter), counter.cb), ('memory_api', ctypes.get_last_error())
        return {'peak_working_set_bytes': int(counter.PeakWorkingSetSize), 'working_set_bytes': int(counter.WorkingSetSize)}
    def guard(process=None, prefix=None, launched=None, timeout=None, compile_root=False):
        nonlocal parent_peak, child_peak, compiler_descendant_sample_peak
        info = psutil.Process().memory_info(); parent_peak = max(parent_peak, info.rss, getattr(info, 'peak_wset', 0)); observation = None
        if process is not None:
            observation = memory(process); child_peak = max(child_peak, observation['peak_working_set_bytes'])
        descendant_rss = 0
        if compile_root and process is not None:
            try:
                for descendant in psutil.Process(process.pid).children(recursive=True):
                    try:
                        descendant_rss += descendant.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            compiler_descendant_sample_peak = max(compiler_descendant_sample_peak, descendant_rss)
        elapsed = time.monotonic() - start
        assert parent_peak <= 1 << 30 and parent_peak + child_peak + descendant_rss <= 16 << 30, 'parent1GiB_sum16GiB'
        assert elapsed <= 1800 and (numeric_start is not None or elapsed <= 300), 'admission300_total1800_seconds'
        assert numeric_start is None or time.monotonic() - numeric_start <= 1500, 'numeric1500_seconds'
        if launched is not None:
            assert time.monotonic() - launched <= timeout, 'child_timeout'
        live_bytes = sum(p.stat().st_size for p in OUT.glob(prefix.name + '.*') if p.is_file()) if prefix is not None else 0
        assert live_bytes <= 128 << 20 and completed_bytes + live_bytes <= 8 << 30, 'child128MiB_all8GiB'
        return observation
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20):
                h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    def inventory(path, reused=False):
        path = Path(path); item = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
        result['reused_output_inventory' if reused else 'output_inventory'].append(item)
        return item['sha256']
    def run(label, argv, env, timeout, affinity=None, kind='generation'):
        nonlocal child, child_peak, completed_bytes
        prefix = OUT / label; record = {'kind': kind, 'argv': list(argv), 'timeout_seconds': timeout, 'maximum_polled_peak_working_set_bytes': 0}
        result['commands'].append(record); launched = time.monotonic()
        with Path(str(prefix) + '.stdout.log').open('xb') as stdout, Path(str(prefix) + '.stderr.log').open('xb') as stderr:
            child = subprocess.Popen(list(argv), cwd=ROOT, env=env, stdout=stdout, stderr=stderr); record['pid'] = child.pid
            if affinity is not None:
                process = psutil.Process(child.pid); process.cpu_affinity(affinity); record['actual_affinity'] = process.cpu_affinity(); assert record['actual_affinity'] == affinity
            while child.poll() is None:
                sample = guard(child, prefix, launched, timeout, kind == 'compile')
                record['maximum_polled_peak_working_set_bytes'] = max(record['maximum_polled_peak_working_set_bytes'], sample['peak_working_set_bytes']); time.sleep(.25)
            record['returncode'] = child.returncode; record['terminal_memory'] = memory(child); child_peak = max(child_peak, record['terminal_memory']['peak_working_set_bytes'])
            assert record['terminal_memory']['peak_working_set_bytes'] >= record['maximum_polled_peak_working_set_bytes'] and record['terminal_memory']['peak_working_set_bytes'] > 0
            guard(None, prefix, launched, timeout)
        child = None; record['wrapper_seconds'] = time.monotonic() - launched
        for kind_name in ('stdout', 'stderr'):
            record[kind_name + '_sha256'] = inventory(str(prefix) + f'.{kind_name}.log')
        completed_bytes = sum(item['bytes'] for item in result['output_inventory']); guard()
        return record
    def worker_rows(row, threads, affinity):
        assert row['worker_physical_cores'] == 6 and [v['slot'] for v in row['worker_affinity']] == list(range(threads))
        assert [v['actual_mask'] for v in row['worker_affinity']] == [1 << cpu for cpu in affinity]
        assert len({v['windows_thread_id'] for v in row['worker_affinity']}) == threads and all(v['group'] == 0 for v in row['worker_affinity'])
        assert set(v['slot'] for v in row['worker_binding_events']) == set(range(threads))
        assert all(v['actual_mask'] == 1 << affinity[v['slot']] and v['group'] == 0 for v in row['worker_binding_events'])
    try:
        result['git_head_before_execution'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'], cwd=ROOT)
        fresh_files = (Path(__file__), PROTOCOL, *(BASE / name for name in ('meth461_switch_common_input.c', 'meth461_switch_common_input_cost_entry.c',
                      'meth461_switch_common_input_entry.c', 'meth461_switch_fanout.h', 'meth461_switch_fanout_controls.h')))
        for path in fresh_files:
            committed(path); result['helper_sha256'][str(path)] = sha(path)
        for name, expected in DIRECT.items():
            path = DOC / name; committed(path); assert sha(path) == expected; result['retained_record_sha256'][name] = expected
        parent = json.loads((DOC / 'meth458_switch_matched_whole_cost_result.json').read_text(encoding='utf-8'))
        admission = json.loads((DOC / 'meth460_switch_common_input_admission_result.json').read_text(encoding='utf-8'))
        assert all(parent['apparatus_gates'].values()) and all(admission['gates'].values())
        assert all(all(json.loads((DOC / name).read_text(encoding='utf-8'))['gates'].values()) for name in ('RETENTION_458_20261005.json', 'RETENTION_460_20261005.json'))
        for path, expected in parent['helper_sha256'].items():
            committed(path); assert sha(path) == expected; result['helper_sha256'][path] = expected
        for name, expected in parent['retained_record_sha256'].items():
            committed(DOC / name); assert sha(DOC / name) == expected; result['retained_record_sha256'][name] = expected
        for item in parent['output_inventory']:
            assert Path(item['path']).stat().st_size == item['bytes'] and inventory(item['path'], True) == item['sha256']
        old_cpu = (BASE / 'meth388_switch_three_workers.c').read_text(encoding='utf-8')
        new_cpu = (BASE / 'meth461_switch_common_input.c').read_text(encoding='utf-8')
        restored = new_cpu.replace('#include "meth461_switch_fanout.h"\n', '')
        for old_call, new_call in PAIRS:
            assert restored.count(new_call) == 1; restored = restored.replace(new_call, old_call)
        assert restored == old_cpu
        assert (BASE / 'meth461_switch_common_input_cost_entry.c').read_text(encoding='utf-8').replace('meth461_switch_common_input.c', 'meth388_switch_three_workers.c') == (BASE / 'meth388_switch_three_workers_cost_entry.c').read_text(encoding='utf-8')
        entry = (BASE / 'meth461_switch_common_input_entry.c').read_text(encoding='utf-8'); marker = '#include "meth461_switch_fanout_controls.h"'
        reversed_entry = entry[:entry.index(marker)].rstrip() + '\n'
        assert reversed_entry.replace('meth461_switch_common_input_cost_entry.c', 'meth388_switch_three_workers_cost_entry.c').replace('int meth461_generation_main(', 'int main(') == (BASE / 'meth388_switch_three_workers_entry.c').read_text(encoding='utf-8').rstrip() + '\n'
        result['source_math_reversal_exact388'] = True
        engine = ROOT / 'benchmarks/phase60/engine.c'; committed(engine); assert sha(engine) == parent['preserved_engine_sha256']; result['preserved_engine_sha256'] = parent['preserved_engine_sha256']
        for n, source in parent['sources'].items():
            assert source['profiler_admissibility']['admitted']; artifact = source['artifact']; payload = Path(artifact['payload'])
            assert [payload.stat().st_size, payload.stat().st_mtime_ns] == source['artifact_stat_before'] and sha(payload) == artifact['sha256'] and sha(artifact['manifest']) == artifact['manifest_sha256']
            compile_record = source['compile']; binary = Path(compile_record['argv'][-1]); runtime = binary.parent / 'libomp.dll'; compiler = Path(compile_record['argv'][0])
            assert sha(binary) == compile_record['binary_sha256'] and sha(runtime) == compile_record['runtime_sha256'] and sha(compiler) == compile_record['compiler_sha256']
            result['sources'][n] = {key: source[key] for key in ('artifact', 'compile', 'runtime_environment', 'threads', 'affinity', 'accepted_cases', 'accepted_generated_tokens', 'accepted_prose_tokens', 'rejected_cases_time_charged')}
            result['sources'][n].update(artifact_stat_before=source['artifact_stat_before'], cases=[])
            assert admission['sources'][n]['artifact'] == artifact
        topology_path = BASE / 'meth359_switch_cost_topology.py'; assert sha(topology_path) == parent['helper_sha256'][str(topology_path)]
        import meth359_switch_cost_topology as topology
        result['fresh_topology'] = topology.physical_topology(); assert result['fresh_topology']['process_allowed_logical_processors'] == list(range(12)) and result['fresh_topology']['selected_one_logical_per_physical_core'] == [0, 2, 4, 6, 8, 10]
        result['preserved_daemons_before'] = jobs(); assert psutil.virtual_memory().available >= 16 << 30 and shutil.disk_usage(ROOT).free >= 10 << 30
        guard(); OUT.mkdir(parents=True); numeric_start = time.monotonic(); print(json.dumps({'admission_seconds': numeric_start - start}), flush=True)
        common_env = {key: value for key, value in os.environ.items() if not key.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_WORKER_BINDING_'))}
        common_env.update(parent['sources']['128']['runtime_environment']); common_env['PATH'] = str(compiler.parent) + os.pathsep + common_env.get('PATH', '')
        stage = 'compile'; candidate = OUT / 'meth461_switch_common_input.exe'
        argv = [str(compiler), '-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off', '-fopenmp', str(BASE / 'meth461_switch_common_input_entry.c'), '-o', str(candidate)]
        record = run('compile', argv, common_env, 120, kind='compile'); assert record['returncode'] == 0
        shutil.copyfile(runtime, OUT / 'libomp.dll'); inventory(candidate); inventory(OUT / 'libomp.dll')
        result['candidate_compile'] = {'argv': list(argv), 'binary_sha256': sha(candidate), 'compiler_sha256': sha(compiler), 'runtime_sha256': sha(OUT / 'libomp.dll')}
        stage = 'arithmetic_fixture'; fixture_data, expected_data = control_fixture()
        fixture = OUT / 'fixture.bin'; expected_file = OUT / 'control_scalar_expected.bin'; fixture.write_bytes(fixture_data); expected_file.write_bytes(expected_data)
        inventory(fixture); expected_sha = inventory(expected_file); result['control_expected_sha256'] = expected_sha
        for threads in (3, 6):
            affinity = [0, 2, 4, 6, 8, 10][:threads]; env = common_env | {'OMP_NUM_THREADS': str(threads)}
            output = OUT / f'control{threads}.bin'; record = run(f'control{threads}', [str(candidate), '--fanout-controls', str(fixture), str(output), str(threads)], env, 60, affinity, 'control')
            assert record['returncode'] == 0 and output.read_bytes() == expected_data and inventory(output) == expected_sha
            row = json.loads((OUT / f'control{threads}.stdout.log').read_text(encoding='utf-8')); assert row['cases'] == 5 and row['threads'] == threads and row['original_candidate_bytes_exact']; worker_rows(row, threads, affinity)
        record = run('negative_alias', [str(candidate), '--fanout-alias'], common_env, 30, kind='negative')
        assert record['returncode'] == 2 and (OUT / 'negative_alias.stderr.log').read_bytes().splitlines() == [b'switch_reference_error:fanout_input_output_alias']
        completed_bytes = sum(item['bytes'] for item in result['output_inventory']); guard()
        for n, source in result['sources'].items():
            threads, affinity = source['threads'], source['affinity']; env = common_env | source['runtime_environment'] | {'OMP_NUM_THREADS': str(threads)}
            original = Path(source['compile']['argv'][-1]); previous = parent['sources'][n]
            for index, expected_case in enumerate(previous['cases']):
                bi, ci = expected_case['book'], expected_case['case']
                previous_command = next(command for command in parent['commands'] if str(command['n']) == n and command['book'] == bi and command['case'] == ci and command['profile'] == 0)
                observed = {'book': bi, 'case': ci, 'source_id': expected_case['source_id'], 'healthy_accepted': expected_case['healthy_accepted'],
                            'generated_tokens': expected_case['generated_tokens'], 'prose_tokens': expected_case['prose_tokens'], 'arm_order': [index % 2, 1 - index % 2], 'arms': {}}
                source['cases'].append(observed)
                for arm in observed['arm_order']:
                    stage = f'n{n}_book{bi}_case{ci}_arm{arm}'; label = f'n{n}.book{bi}.case{ci}.arm{arm}'; prefix = OUT / label
                    binary = original if arm == 0 else candidate
                    argv = [str(binary), '--generate', source['artifact']['manifest'], previous_command['argv'][3], str(prefix), str(threads), '64', '32095', '0', '1', '3', '0']
                    record = run(label, argv, env, 60, affinity); record.update(n=int(n), book=bi, case=ci, arm=arm)
                    assert record['returncode'] == 0
                    rows = [json.loads(line) for line in Path(str(prefix) + '.stdout.log').read_text(encoding='utf-8').splitlines()]
                    assert [row['repetition'] for row in rows] == [-1, 0, 1, 2]; observed['arms'][str(arm)] = rows
                    for row, expected in zip(rows, expected_case['modes']['0'], strict=True):
                        worker_rows(row, threads, affinity)
                        assert row['threads'] == threads and row['profile'] == 0 and row['source_tokens'] == 29 and row['closing_id'] == 32095
                        assert row['generated_ids'] == expected['generated_ids'] and row['actual_generated_tokens'] == expected['actual_generated_tokens']
                        assert all(row[key] == expected[key] for key in STOPS)
                        assert all(math.isfinite(row[key]) and row[key] > 0 for key in TIMES + ('load_seconds', 'first_token_seconds'))
                        assert len(row['step_seconds']) == expected_case['generated_tokens'] and all(math.isfinite(value) and value > 0 for value in row['step_seconds'])
                        assert abs(sum(row[key] for key in TIMES[:3]) - row['full_generation_seconds']) <= 1e-7
                        assert len(row['counters']) == 2 and all(len(counters) == 4 for counters in row['counters'])
                        for phase in range(2):
                            for kind in range(4):
                                assert all(row['counters'][phase][kind][key] == expected['counters'][phase][kind][key] for key in LOGICAL)
                                assert row['counters'][phase][kind]['matrix_seconds'] == 0
                        row['output_sha256'] = inventory(str(prefix) + f'.{row["repetition"]}.bin'); assert row['output_sha256'] == expected['output_sha256']
                    completed_bytes = sum(item['bytes'] for item in result['output_inventory']); guard()
                if ci == 3:
                    print(json.dumps({'source': n, 'completed_book': bi, 'generation_commands': sum(c['kind'] == 'generation' for c in result['commands']), 'seconds': time.monotonic() - start}), flush=True)
            payload = Path(source['artifact']['payload']); assert source['artifact_stat_before'] == [payload.stat().st_size, payload.stat().st_mtime_ns]
        stage = 'fixed_whole_aggregation'; both_pass = True
        for n, source in result['sources'].items():
            summaries = {}
            for arm in ('0', '1'):
                measured = [row for case in source['cases'] for row in case['arms'][arm] if row['repetition'] >= 0]
                books = [sum(row['full_generation_seconds'] for case in source['cases'] if case['book'] == bi for row in case['arms'][arm] if row['repetition'] >= 0) / 3 for bi in range(24)]
                full = sum(books); median_total = sum(statistics.median(row['full_generation_seconds'] for row in case['arms'][arm] if row['repetition'] >= 0) for case in source['cases'])
                summaries[arm] = {'ALL_case_mean_full_seconds': full, 'book_mean_full_seconds': books, 'pooled_measured_full_p95': p95([row['full_generation_seconds'] for row in measured]),
                                  'ALL_case_median_full_seconds': median_total, 'accepted_ordinary_mean_rate': source['accepted_generated_tokens'] / full,
                                  'accepted_prose_mean_rate': source['accepted_prose_tokens'] / full,
                                  'phase_mean_seconds': {key: sum(row[key] for row in measured) / 3 for key in TIMES},
                                  'repetition_ALL_case_full_seconds': [sum(case['arms'][arm][rep + 1]['full_generation_seconds'] for case in source['cases']) for rep in range(3)]}
            mean_ratio = summaries['1']['ALL_case_mean_full_seconds'] / summaries['0']['ALL_case_mean_full_seconds']
            books = [candidate_time / baseline for candidate_time, baseline in zip(summaries['1']['book_mean_full_seconds'], summaries['0']['book_mean_full_seconds'], strict=True)]
            tail_ratio = summaries['1']['pooled_measured_full_p95'] / summaries['0']['pooled_measured_full_p95']
            gates = {'whole_mean_ratio_le0p95': mean_ratio <= .95, 'ALL24_book_mean_ratios_le1p05': all(value <= 1.05 for value in books), 'pooled_measured_full_p95_ratio_le1p00': tail_ratio <= 1.00}
            passed = all(gates.values()); both_pass &= passed
            source.update(arm_summaries=summaries, whole_cost={'mean_ratio': mean_ratio, 'book_mean_ratios': books, 'p95_ratio': tail_ratio, 'gates': gates, 'passed': passed})
            accepted = [case for case in source['cases'] if case['healthy_accepted']]
            assert [len(accepted), sum(c['generated_tokens'] for c in accepted), sum(c['prose_tokens'] for c in accepted)] == [source['accepted_cases'], source['accepted_generated_tokens'], source['accepted_prose_tokens']]
        result['preserved_daemons_after'] = jobs(); guard()
        actual_files = {str(path) for path in OUT.iterdir() if path.is_file()}; assert actual_files == {item['path'] for item in result['output_inventory']}
        assert len(result['commands']) == 388 and len(result['output_inventory']) == 2318
        result['apparatus_gates'] = {'fresh_original_payload_quality_runtime_source_bindings_exact': True, 'source_math_exact388_except_ONE_primitive_three_callsites_dispatch': True,
            'five_primal_shapes_native_original_and_independent_scalar_bytes_exact_CPU3_CPU6': True, 'expected_alias_negative_exit2': True,
            'ALL1536_whole_warm_measured_states_logits_routes_ids_exact_original': True, 'all_worker_process_affinity_counter_context_readbacks_exact': True,
            'all_accepted_totals_and_rejected_case_time_charged': True, 'all388_terminal_peak_resource_queries_and2318_files_retained': True}
        result['resource'] = {'main_seconds_excluding_imports': time.monotonic() - start, 'admission_seconds': numeric_start - start, 'compile_control_native_aggregation_seconds': time.monotonic() - numeric_start,
            'parent_peak_working_set_bytes': parent_peak, 'command_root_peak_working_set_bytes': child_peak, 'conservative_parent_plus_root_peaks_bytes': parent_peak + child_peak,
            'compiler_descendant_maximum_checked_RSS_sum_bytes': compiler_descendant_sample_peak, 'output_bytes': completed_bytes, 'bytes_hashed_before_raw': hashed}
        result['decision'] = 'ONE_exact_fanout_whole_cost_candidate_admitted_both_sources_prepare_fresh_same_artifact_quality_rate' if both_pass else 'ONE_exact_fanout_execution_recipe_closed_by_whole_cost_gates_no_tuning_return_to_transfer_capacity'
        result['scope'] = 'ONE common-input execution transformation on same original128/256 payloads and consumed short infilling contexts. Complete-state/ID equivalence and arithmetic controls, not new held-out donor quality or learned capacity. Ordinary/prose accepted mean rates descriptive, all rejected time charged; no confidence interval specified here. Same respective CPU3/6 worker settings, two original different cores not causal E-only scaling. Full timer includes encoder/crossKV/cached decoder/head/greedy/stop, excludes startup/load/team setup/fixed-ID tokenization/serialization/cleanup. No coefficient reformat/rounding/fit/GPU/engine edits. Integer products/logical coefficient bytes/storage and E unchanged. Working set is not hardwareDRAM traffic. Separate source cost gates fixed before compile; no median substitution/outlier trim/retry. Return to full transfer/useful-n composition after this one cycle regardless of outcome.'
        write_new(RAW, result)
        print(json.dumps({'sha256': sha(RAW), 'apparatus_gates': result['apparatus_gates'], 'decision': result['decision'],
                          'sources': {n: source['whole_cost'] for n, source in result['sources'].items()}, 'resource': result['resource']}), flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.kill(); child.wait()
        result.update(stage=stage, error=repr(error), main_seconds_excluding_imports=time.monotonic() - start, parent_peak_working_set_bytes=parent_peak, command_root_peak_working_set_bytes=child_peak)
        write_new(RAW.with_suffix('.failure.json'), result)
        raise


if __name__ == '__main__':
    main()
