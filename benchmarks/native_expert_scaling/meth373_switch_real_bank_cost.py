"""Equal-length native CPU cost across actual nested learned 64/128/256 banks."""
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
import meth368_switch_bank_manifest as I
import meth372_switch_nested_bank_usefulness as U

OUT = M.ROOT / 'results/native_expert_scaling/meth373_switch_real_bank_cost'
PROTOCOL = M.DOC / 'METH_373_SWITCH_REAL_BANK_COST_PROTOCOL_20261004.md'
BINDINGS = {
    338: ('meth338_switch_tensor_recovery_result.json', '19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'),
    356: ('meth356_switch_all_a16_contract_result.json', 'ec51e76c08277e4f874cacac9a49d5b60477c0abb6a651cbe648f9bdad3f66d3'),
    362: ('meth362_switch_multi_span_manifest.json', 'c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'),
    363: ('meth363_switch_all_a16_multi_span_quality_result.json', 'ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'),
    371: ('meth371_switch_nested_bank_contract_result.json', '2d534664bd715c771e62c33fa50f4d85211dc994efefbddaecef0d7e0c80614d'),
    372: ('meth372_switch_nested_bank_usefulness_result.json', '34861eb0fc78bb2c126d15c5544fb03525d77df2c69fd9d99de0a9c5619ddd23'),
}
ENGINE = M.ROOT / 'benchmarks/phase60/engine.c'


def paired_ratio(high, low):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    assert high.shape == low.shape == (24,) and np.all(low > 0)
    indices = np.random.default_rng(373373).integers(0, 24, size=(10000, 24))
    draws = high[indices].sum(axis=1) / low[indices].sum(axis=1)
    return {'ratio_of_sums': float(high.sum() / low.sum()),
            'one_sided_lower95': float(np.quantile(draws, .05)),
            'one_sided_upper95': float(np.quantile(draws, .95)),
            'bootstrap_unit': 'book', 'draws': 10000, 'seed': 373373}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    started = time.monotonic()
    maximum = 0
    stage = 'bindings'
    result = {'experiment': 'METH-373-equal-length-real-bank-CPU-cost', 'commands': [], 'cases': [], 'profiles': []}

    def guard(child=None):
        nonlocal maximum
        rss = psutil.Process().memory_info().rss
        child_rss = 0
        if child is not None:
            try:
                child_rss = psutil.Process(child.pid).memory_info().rss
                rss += child_rss
            except psutil.NoSuchProcess:
                pass
        maximum = max(maximum, rss)
        if rss > 16 << 30 or time.monotonic() - started > 1800:
            if child is not None and child.poll() is None:
                child.kill(); child.wait()
            raise RuntimeError('real_bank_cost_30min_16GiB')
        return child_rss

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20):
                h.update(block); guard()
        return h.hexdigest()

    try:
        for path in (Path(__file__), PROTOCOL, ENGINE, Path(M.__file__), Path(A.__file__), Path(I.__file__), Path(U.__file__)):
            M.committed(path)
        records = {}
        for number, (name, sha) in BINDINGS.items():
            path = M.DOC / name
            M.committed(path); assert digest(path) == sha
            records[number] = json.loads(path.read_text(encoding='utf-8'))
        assert all(records[356]['gates'].values()) and all(records[371]['gates'].values())
        assert all(records[363]['gates'].values()) and not all(records[372]['gates'].values())
        assert records[372]['gates']['both_actual64_128_ALL96_completed_with_retained_function_ids']
        fresh = A.physical_topology()
        assert fresh == records[371]['fresh_topology']
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmd = ' '.join(process.info['cmdline'] or []).replace('\\', '/')
            if process.pid not in own and process.info['name'].lower() in ('python.exe', 'meth356_switch_all_a16.exe', 'meth365_switch_encoder_batches.exe'):
                assert not ('benchmarks/native_expert_scaling/' in cmd or 'meth356_switch_all_a16.exe' in cmd or 'meth365_switch_encoder_batches.exe' in cmd), 'concurrent_model_or_native_worker'
        binary = Path(records[356]['compile']['argv'][-1])
        assert digest(binary) == records[356]['compile']['binary_sha256'] == records[371]['compile']['binary_sha256']
        assert digest(binary.parent / 'libomp.dll') == records[356]['compile']['runtime_sha256']
        assert digest(records[356]['compile']['argv'][0]) == records[356]['compile']['compiler_sha256']
        artifacts = {256: records[338]['artifact']}
        configs = {256: records[338]['original_config']}
        entries = {256: records[338]['tensors']}
        for target in records[371]['targets']:
            n = target['n']; artifacts[n] = target['artifact']; configs[n] = target['artifact']['original_config']
            metadata = Path(target['artifact']['metadata'])
            assert digest(metadata) == target['artifact']['metadata_sha256']
            entries[n] = json.loads(metadata.read_text(encoding='utf-8'))['tensors']
        before = {}
        for n in (64, 128, 256):
            artifact = artifacts[n]; payload = Path(artifact['payload'])
            before[n] = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert before[n][0] == artifact['bytes'] and digest(payload) == artifact['sha256']
            assert digest(artifact['manifest']) == artifact['manifest_sha256']
            assert configs[n]['num_experts'] == n and len(entries[n]) == 24*n+248
            I.read_manifest(artifact['manifest'], configs[n], entries[n], payload)
        expected = {}
        for n in (64, 128):
            control = next(v for v in records[372]['controls'] if v['n'] == n)
            for case in control['cases']:
                bi, ci = case['book'], case['case']
                path = U.OUT / f'n{n}.book{bi}.case{ci}.0.bin'
                assert digest(path) == case['forced_sha256']
                expected[n, bi, ci] = case['forced_sha256']
        base_dir = M.ROOT / 'results/native_expert_scaling/meth363_switch_all_a16_multi_span_quality'
        for bi, book in enumerate(records[363]['books']):
            for ci, case in enumerate(book['cases']):
                path = base_dir / f'book{bi}.case{ci}.0.bin'
                assert digest(path) == case['native_output_sha256']
                expected[256, bi, ci] = case['native_output_sha256']
        result.update({'input_sha256': {str(k): v[1] for k,v in BINDINGS.items()},
                       'controller_sha256': digest(__file__), 'protocol_sha256': digest(PROTOCOL), 'engine_sha256': digest(ENGINE),
                       'compile': records[356]['compile'], 'artifacts': artifacts, 'fresh_topology': fresh,
                       'native_threads': 1, 'native_process_affinity': [0],
                       'physical_RAM_bytes': psutil.virtual_memory().total, 'available_RAM_before_bytes': psutil.virtual_memory().available,
                       'primary_is_forced_equal_length_cost_not_accepted_generation_rate': True})
        OUT.mkdir(parents=True)
        env = os.environ.copy(); env.pop('OMP_PROC_BIND', None)
        env.update({'OMP_NUM_THREADS':'1', 'OMP_WAIT_POLICY':'PASSIVE', 'KMP_AFFINITY':'none'})

        def run(n, bi, ci, fixture, profile):
            label = f'n{n}.book{bi}.case{ci}.p{profile}'
            prefix = OUT / label
            reps = 3 if profile == 0 else 1
            argv = [str(binary), str(artifacts[n]['manifest']), ','.join(map(str, fixture['source_ids'])),
                    ','.join(map(str, fixture['decoder_ids'])), str(prefix), '1', str(profile), '1', str(reps), '0']
            sampled_peak = 0
            with (OUT / (label+'.stdout.log')).open('wb') as stdout, (OUT / (label+'.stderr.log')).open('wb') as stderr:
                child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env)
                try:
                    process = psutil.Process(child.pid); process.cpu_affinity([0]); affinity = process.cpu_affinity()
                    assert affinity == [0]
                    while child.poll() is None:
                        sampled_peak = max(sampled_peak, guard(child)); time.sleep(.10)
                except BaseException:
                    if child.poll() is None: child.kill(); child.wait()
                    raise
            assert child.returncode == 0, (label, child.returncode)
            result['commands'].append({'argv': argv, 'returncode': child.returncode, 'actual_affinity': affinity,
                                       'stdout_sha256': digest(OUT / (label+'.stdout.log')), 'stderr_sha256': digest(OUT / (label+'.stderr.log'))})
            rows = [json.loads(v) for v in (OUT / (label+'.stdout.log')).read_text(encoding='utf-8').splitlines()]
            assert [r['repetition'] for r in rows] == [-1, *range(reps)]
            for row in rows:
                path = Path(str(prefix)+'.'+str(row['repetition'])+'.bin')
                row['output_sha256'] = digest(path)
                assert row['output_sha256'] == expected[n,bi,ci], 'warm_repeat_profile_complete_output_bridge'
                assert (row['source_tokens'], row['decoder_positions'], row['experts_per_bank'], row['threads'], row['profile']) == (29,14,n,1,profile)
                for phase, positions in enumerate((29,14)):
                    counters = row['counters'][phase]
                    assert counters[2]['calls'] == 6*positions
                    assert counters[2]['f32_bytes'] == 18432*n*positions
                    assert counters[1]['calls'] == 12*positions
                    assert counters[1]['code_bytes'] == 4718592*6*positions
                    if phase == 1:
                        assert sum(v['code_bytes'] for v in counters) == 123764736*positions
                        assert sum(v['scale_bytes'] for v in counters) == 534016*positions
                        assert sum(v['f32_bytes'] for v in counters) == 18432*n*positions
            arrays = U.read_output(Path(str(prefix)+'.0.bin'))
            consultations = I.consultations(configs[n],29,14,arrays[3],0)
            groups = {}
            for call in consultations:
                bank = f"{call['stack']}.{call['layer']}"
                if call['accepted']: groups.setdefault(bank,set()).add(call['source_expert'])
            assert len(groups) == 12 and all(call['accepted'] for call in consultations)
            unique_names = {call[key] for call in consultations for key in ('consulted_wi','consulted_wo') if call[key]}
            unique_bytes = sum(entries[n][name]['bytes'] + entries[n][name]['scale_bytes'] for name in unique_names)
            return {'n':n, 'book':bi, 'case':ci, 'rows':rows, 'sampled_child_peak_RSS_bytes':sampled_peak,
                    'mapped_payload_file_bytes_not_RSS':artifacts[n]['bytes'],
                    'accepted_expert_union_per_bank':{key:sorted(value) for key,value in sorted(groups.items())},
                    'unique_consulted_expert_code_and_scale_bytes':unique_bytes,
                    'accepted_routes':len(consultations), 'exact_prior_teacher_SHA':expected[n,bi,ci]}

        # Three cyclic orders, equally represented, fixed before observations.
        for bi, book in enumerate(records[362]['items']):
            for ci, fixture in enumerate(book['cases']):
                assert len(fixture['source_ids']) == 29 and len(fixture['decoder_ids']) == 14
                order = (64,128,256); shift = (bi*4+ci)%3; order = order[shift:]+order[:shift]
                for n in order:
                    stage = f'primary.n{n}.book{bi}.case{ci}'
                    result['cases'].append(run(n,bi,ci,fixture,0))
            print(json.dumps({'stage':'primary','book':bi,'completed':len(result['cases'])}),flush=True)
        for bi, book in enumerate(records[362]['items']):
            for ci, fixture in enumerate(book['cases']):
                order = (64,128,256); shift = (bi*4+ci)%3; order = order[shift:]+order[:shift]
                for n in order:
                    stage = f'profile.n{n}.book{bi}.case{ci}'
                    result['profiles'].append(run(n,bi,ci,fixture,1))
            print(json.dumps({'stage':'profile','book':bi,'completed':len(result['profiles'])}),flush=True)
        summary = {}
        vectors = {}
        for n in (64,128,256):
            cases = [c for c in result['cases'] if c['n']==n]
            profiles = [c for c in result['profiles'] if c['n']==n]
            assert len(cases)==len(profiles)==96
            measured = [[row for row in c['rows'] if row['repetition']>=0] for c in cases]
            totals = [sum(rows[rep]['full_seconds'] for rows in measured) for rep in range(3)]
            decode_totals = [sum(rows[rep]['decode_seconds'] for rows in measured) for rep in range(3)]
            medians = {phase:[float(np.median([row[phase+'_seconds'] for row in rows])) for rows in measured]
                       for phase in ('encode','cross_kv','decode','full')}
            vectors[n] = {phase:[sum(medians[phase][bi*4:bi*4+4]) for bi in range(24)] for phase in medians}
            matrix = np.asarray([[row['matrix_seconds'] for phase in c['rows'][1]['counters'] for row in phase] for c in profiles])
            unions = {}
            for case in cases:
                for bank, experts in case['accepted_expert_union_per_bank'].items(): unions.setdefault(bank,set()).update(experts)
            summary[n] = {'sum_case_median_seconds':{phase:sum(values) for phase,values in medians.items()},
                          'full_totals_by_repetition':totals,'decode_totals_by_repetition':decode_totals,
                          'full_aggregate_repeat_ratio':max(totals)/min(totals),'decode_aggregate_repeat_ratio':max(decode_totals)/min(decode_totals),
                          'profile_matrix_seconds_encoder_core_expert_router_head_decoder_core_expert_router_head':matrix.sum(axis=0).tolist(),
                          'cohort_consulted_union_per_bank':{bank:len(ids) for bank,ids in unions.items()},
                          'sampled_child_peak_RSS_bytes':max(c['sampled_child_peak_RSS_bytes'] for c in cases),
                          'mean_unique_expert_code_and_scale_bytes_per_case':float(np.mean([c['unique_consulted_expert_code_and_scale_bytes'] for c in cases])),
                          'decode_logical_matrix_bytes_per_position':123764736+534016+18432*n}
        result['summary_by_n'] = summary
        result['paired_256_over64'] = {phase:paired_ratio(vectors[256][phase],vectors[64][phase]) for phase in ('encode','cross_kv','decode','full')}
        result['paired_128_over64'] = {phase:paired_ratio(vectors[128][phase],vectors[64][phase]) for phase in ('encode','cross_kv','decode','full')}
        for n, old in before.items():
            payload = Path(artifacts[n]['payload']); assert old==(payload.stat().st_size,payload.stat().st_mtime_ns)
        result['gates'] = {'all_288_primary_and288_profile_cases_complete':len(result['cases'])==len(result['profiles'])==288,
                           'all_warm_repeats_profiles_complete_bytes_exact363_or372':True,
                           'all_affinity_readbacks_exact':all(c['actual_affinity']==[0] for c in result['commands']),
                           'all_dynamic_n_lookup_routes_and_logical_descriptors_exact':True,
                           'all_full_and_decode_aggregate_repeat_ratios_le1p10':all(max(s['full_aggregate_repeat_ratio'],s['decode_aggregate_repeat_ratio'])<=1.10 for s in summary.values()),
                           '256_over64_full_cost_upper95_le1p20':result['paired_256_over64']['full']['one_sided_upper95']<=1.20,
                           '256_over64_decode_cost_upper95_le1p20':result['paired_256_over64']['decode']['one_sided_upper95']<=1.20}
        result['resource'] = {'main_seconds_excluding_imports':time.monotonic()-started,'maximum_checked_combined_RSS_bytes':maximum,'available_RAM_end_bytes':psutil.virtual_memory().available}
        result['decision'] = 'bounded_fourfold_actual_bank_CPU_cost_supported' if all(result['gates'].values()) else 'frozen_fourfold_actual_bank_CPU_cost_claim_not_all_established'
        result['scope'] = 'Equal forced source29/decoder14 consumed cohort; same356 executable/precision/core/top1, actual physically compact64/128/256 learned banks. Warm/hash-primed pretokenized CPU1 affinity[0], full timer includes encoder/crossKV/cached decode, excludes startup/load/serialization. Direct expert lookup, not a new hierarchical or ternary LUT design. Profile attribution includes per-matrix clocks. Sampled process RSS is not exact expert residency. Logical addressed bytes and unique consulted regions are not physical DRAM traffic. No natural accepted-rate/50tps/new original-quality or monotonic quality claim;372 mixed quality retained. No >256/100B/cross-family extrapolation.'
        M.write(args.out,result)
        print(json.dumps({'sha256':digest(args.out),'gates':result['gates'],'resource':result['resource'],'summary':summary,'paired_256_over64':result['paired_256_over64']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-started,'maximum_checked_combined_RSS_bytes':maximum})
        M.write(args.out.with_suffix('.failure.json'),result)
        raise


if __name__=='__main__':
    main()
