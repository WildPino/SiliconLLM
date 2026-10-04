"""NEW374 exact worker execution, PRIMARY CPU6 physical workers/active waiting cost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import psutil
import meth324_switch_reference as M
import meth359_switch_cost_topology as A
import meth374_switch_physical_workers_contract as N

OUT=M.ROOT/'results/native_expert_scaling/meth375_switch_physical_workers_cost'
PROTOCOL=M.DOC/'METH_375_SWITCH_PHYSICAL_WORKERS_COST_PROTOCOL_20261004.md'
QUALIFIED=M.DOC/'meth356_switch_all_a16_contract_result.json'
QUALIFIED_SHA='ec51e76c08277e4f874cacac9a49d5b60477c0abb6a651cbe648f9bdad3f66d3'
NUMERIC=M.DOC/'meth374_switch_physical_workers_contract_result.json'
NUMERIC_SHA='4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'
RECOVERED=M.DOC/'meth338_switch_tensor_recovery_result.json'
RECOVERED_SHA='19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'
BASELINE=M.DOC/'meth341_switch_batched_prefill_result.json'
BASELINE_SHA='a73b41e01867915f0b0c5b2e1b1add2000bd97acdda329c1edecf98f9db8607c'
SOURCE=M.ROOT/'benchmarks/native_expert_scaling/meth374_switch_physical_workers.c'
ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth374_switch_physical_workers_entry.c'
COST_ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth374_switch_physical_workers_cost_entry.c'
ENGINE=M.ROOT/'benchmarks/phase60/engine.c'
TOPOLOGY=M.DOC/'meth359_switch_cost_topology_result.json'
TOPOLOGY_SHA='45fd854eb7032a2c6a8804079410f0ecf197b70eabc02fc9d5b6f756d1424381'


def stream_digest(path,start):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4<<20):
            h.update(block);assert time.monotonic()-start<=1800,'preflight_time_guard'
    return h.hexdigest()


def output_routes(path):
    raw=Path(path).read_bytes();assert raw[:8]==b'SWR32O01'
    s,t,d,enc,dec,vocab,count=np.frombuffer(raw,dtype='<u4',offset=8,count=7).tolist()
    offset=36+4*((enc+2)*s*d+t*(dec+2)*d+t*vocab)
    assert len(raw)==offset+count*12
    dtype=np.dtype([('expert','<i4'),('accepted','<i4'),('probability','<f4')])
    route=np.frombuffer(raw,dtype=dtype,offset=offset,count=count)
    assert np.isfinite(route['probability']).all() and set(route['accepted'].tolist())<={0,1}
    encoder=route[:6*s].reshape(6,s);decoder=route[6*s:].reshape(t,6)
    return {'encoder_union_per_bank':[len(set(row.tolist())) for row in encoder['expert']],
            'decoder_union_per_bank':[len(set(row.tolist())) for row in decoder['expert'].T],
            'decoder_choices_by_position':decoder['expert'].tolist(),
            'accepted_routes':int(route['accepted'].sum()),'total_routes':count}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-375-actual-all-A16-same-binary-complete-CPU-cost','commands':[],'bridges':[],'measurements':[],'profiles':[]}
    try:
        for path in (Path(__file__),SOURCE,ENTRY,COST_ENTRY,PROTOCOL,QUALIFIED,NUMERIC,RECOVERED,BASELINE,ENGINE,TOPOLOGY,Path(N.__file__),Path(A.__file__),Path(M.__file__)):M.committed(path)
        qualified=json.loads(QUALIFIED.read_text(encoding='utf-8'));recovered=json.loads(RECOVERED.read_text(encoding='utf-8'))
        assert M.digest(NUMERIC)==NUMERIC_SHA
        numeric=json.loads(NUMERIC.read_text(encoding='utf-8'));assert all(numeric['gates'].values());N.source_identity()
        for name,sha in numeric['source_sha256'].items():
            path=M.ROOT/'benchmarks/native_expert_scaling'/name;M.committed(path);assert M.digest(path)==sha
        assert M.digest(BASELINE)==BASELINE_SHA
        baseline_record=json.loads(BASELINE.read_text(encoding='utf-8'))
        assert M.digest(QUALIFIED)==QUALIFIED_SHA
        assert all(qualified['gates'].values()) and M.digest(RECOVERED)==RECOVERED_SHA
        assert M.digest(TOPOLOGY)==TOPOLOGY_SHA
        topology=json.loads(TOPOLOGY.read_text(encoding='utf-8'));fresh=A.physical_topology()
        assert all(topology['gates'].values()) and fresh==topology['topology']
        affinity=fresh['selected_one_logical_per_physical_core'];assert affinity==[0,2,4,6,8,10]
        assert qualified['input_sha256']['338']==RECOVERED_SHA
        own_process_tree={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            command=' '.join(process.info['cmdline'] or [])
            if process.pid not in own_process_tree and process.info['name'].lower() in ('python.exe','meth356_switch_all_a16.exe','meth365_switch_encoder_batches.exe','meth374_switch_physical_workers.exe'):
                assert not ('benchmarks/native_expert_scaling/' in command.replace('\\','/') or 'meth374_switch_physical_workers.exe' in command or 'meth356_switch_all_a16.exe' in command or 'meth365_switch_encoder_batches.exe' in command),'another_model_worker_before_timing'
        compiler=Path(qualified['compile']['argv'][0]);dll=OUT.parent/'meth374_switch_physical_workers_contract/libomp.dll'
        assert M.digest(compiler)==qualified['compile']['compiler_sha256'] and M.digest(dll)==qualified['compile']['runtime_sha256']
        result.update({'controller_sha256':M.digest(__file__),'source_sha256':M.digest(SOURCE),'entry_sha256':M.digest(ENTRY),
                       'numeric374_sha256':NUMERIC_SHA,'runtime_environment':numeric['runtime_environment'],'engine_sha256':M.digest(ENGINE),'protocol_sha256':M.digest(PROTOCOL),'qualified356_sha256':M.digest(QUALIFIED),
                       'recovered338_sha256':RECOVERED_SHA,'preserved341_baseline_sha256':BASELINE_SHA,'host':{'platform':platform.platform(),'logical_cpus':psutil.cpu_count(),
                       'physical_cpus':psutil.cpu_count(logical=False),'physical_RAM_bytes':psutil.virtual_memory().total,
                       'available_RAM_before_bytes':psutil.virtual_memory().available}})
        stage='fresh_full_target_hash';artifact=recovered['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest'])
        before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert before[0]==artifact['bytes'] and stream_digest(payload,start)==artifact['sha256'] and M.digest(spec)==artifact['manifest_sha256']
        result['artifact']=artifact
        OUT.mkdir(parents=True);binary=Path(numeric['compile']['argv'][-1])
        assert M.digest(binary)==numeric['compile']['binary_sha256']
        result['compile']=numeric['compile'];result['same_qualified374_executable']=True
        result['topology359_sha256']=TOPOLOGY_SHA;result['fresh_topology']=fresh;result['native_process_affinity']=affinity;result['primary_native_threads']=6
        env={k:v for k,v in os.environ.items() if not k.startswith(('OMP_','KMP_','GOMP_','SILICON_WORKER_BINDING_'))};env.update(numeric['runtime_environment'])
        def run(control,threads,profile,warmups,repetitions,label):
            nonlocal maximum
            prefix=OUT/label;argv=[str(binary),str(spec),','.join(map(str,control['source_ids'])),','.join(map(str,control['decoder_ids'])),
                str(prefix),str(threads),str(profile),str(warmups),str(repetitions),'0']
            env['OMP_NUM_THREADS']=str(threads)
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env)
                try:
                    native_process=psutil.Process(child.pid);native_process.cpu_affinity(affinity);actual_affinity=native_process.cpu_affinity();assert actual_affinity==affinity
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    raise
                while child.poll() is None:
                    try:
                        rss=psutil.Process(child.pid).memory_info().rss+psutil.Process().memory_info().rss;maximum=max(maximum,rss)
                        if rss>16<<30 or time.monotonic()-start>1800:child.kill();child.wait();raise RuntimeError('cost_resource_guard')
                    except psutil.NoSuchProcess:pass
                    time.sleep(.25)
            result['commands'].append({'argv':argv,'returncode':child.returncode,'actual_affinity':actual_affinity,'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),
                                       'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode)
            rows=[json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()]
            assert len(rows)==warmups+repetitions
            for row in rows:
                assert row['worker_physical_cores']==6
                assert [v['slot'] for v in row['worker_affinity']]==list(range(6))
                assert [v['actual_mask'] for v in row['worker_affinity']]==[1<<cpu for cpu in affinity]
                assert len({v['windows_thread_id'] for v in row['worker_affinity']})==6
                assert all(v['group']==0 for v in row['worker_affinity'])
                assert all(v['actual_mask']==1<<affinity[v['slot']] and v['group']==0 for v in row['worker_binding_events'])
                assert set(v['slot'] for v in row['worker_binding_events'])==set(range(6))
                output=Path(str(prefix)+'.'+str(row['repetition'])+'.bin')
                row['output_sha256']=M.digest(output);row['route_summary']=output_routes(output)
                assert row['experts_per_bank']==256 and row['decoder_positions']==len(control['decoder_ids'])
                positions=len(control['decoder_ids']);counters=row['counters'][1]
                assert sum(v['code_bytes'] for v in counters)==123764736*positions
                assert sum(v['scale_bytes'] for v in counters)==534016*positions
                assert sum(v['f32_bytes'] for v in counters)==4718592*positions
                assert counters[2]['calls']==6*positions and counters[3]['calls']==positions
                row['exact_actual_decode_matrix_descriptor']=True
                if positions==32:
                    previous=next(v for v in qualified['long_cases'] if v['source_length']==len(control['source_ids']))
                    assert row['output_sha256']==previous['native_sha256'],'complete_all_A16_exact356'
                    row['complete_output_exact356']=True
            assert len({r['output_sha256'] for r in rows})==1,'repeat_numeric_identity'
            return rows
        stage='exact_complete356_bridges'
        for i,control in enumerate(qualified['cases']):
            for threads in (6,):
                for profile in (0,1):
                    rows=run(control,threads,profile,0,1,f'bridge{i}.t{threads}.p{profile}')
                    exact=rows[0]['output_sha256']==control['native_sha256'];assert exact,'same_binary_thread_bridge_exact356'
                    result['bridges'].append({'case':i,'threads':threads,'profile':profile,'exact356':exact,'output_sha256':rows[0]['output_sha256']})
        fixtures=[]
        for i,source_length in enumerate((9,64)):
            base=qualified['cases'][i];source=(base['source_ids'][:-1]*8)[:source_length-1]+[1]
            decoder=(base['decoder_ids']*8)[:32]
            fixtures.append({'source_ids':source,'decoder_ids':decoder,'source_length':source_length})
        result['fixtures']=fixtures
        for i,fixture in enumerate(fixtures):
            for threads in (6,):
                rows=run(fixture,threads,0,0,1,f'long_bridge.s{fixture["source_length"]}.t{threads}')
                result['bridges'].append({'case':f'long_source{fixture["source_length"]}','threads':threads,'profile':0,'exact356':True,'exact356_long':rows[0]['complete_output_exact356'],'output_sha256':rows[0]['output_sha256']})
        order=((0,6),(1,6))
        for i,threads in order:
            stage=f'actual_cost_source{fixtures[i]["source_length"]}_threads{threads}'
            rows=run(fixtures[i],threads,0,1,3,f'cost.s{fixtures[i]["source_length"]}.t{threads}')
            observed=[r for r in rows if r['repetition']>=0]
            decode_ms=[r['decode_seconds']/32*1000 for r in observed];full_ms=[r['full_seconds']/32*1000 for r in observed]
            entry={'source_length':fixtures[i]['source_length'],'threads':threads,'rows':rows,'median_decode_ms_per_position':float(np.median(decode_ms)),
                   'median_full_ms_per_position':float(np.median(full_ms)),'repeat_ratio_max_min_decode':max(decode_ms)/min(decode_ms),
                   'logical_matrix_bytes_per_decode_position':sum(v['code_bytes']+v['f32_bytes']+v['scale_bytes'] for v in observed[0]['counters'][1])/32}
            result['measurements'].append(entry);print(json.dumps({k:v for k,v in entry.items() if k!='rows'}),flush=True)
        for i,threads in order:
            stage=f'cost_attribution_source{fixtures[i]["source_length"]}_threads{threads}'
            rows=run(fixtures[i],threads,1,1,1,f'profile.s{fixtures[i]["source_length"]}.t{threads}')
            baseline=next(v for v in result['measurements'] if v['source_length']==fixtures[i]['source_length'] and v['threads']==threads)
            assert rows[0]['output_sha256']==baseline['rows'][0]['output_sha256'],'profile_identity'
            result['profiles'].append({'source_length':fixtures[i]['source_length'],'threads':threads,'rows':rows,'exact_primary_outputs':True})
        primaries=[v for v in result['measurements'] if v['threads']==6]
        result['gates']={'all_actual_worker_affinity_readbacks_exact':True,'all_native_process_affinity_readbacks_exact':all(v['actual_affinity']==affinity for v in result['commands']),
                         'all_complete356_thread_and_counter_bridges_exact':all(v['exact356'] for v in result['bridges']),
                         'all_profile_outputs_exact_primary':all(v['exact_primary_outputs'] for v in result['profiles']),
                         'all_threads6_median_decode_le20ms':all(v['median_decode_ms_per_position']<=20 for v in primaries),
                         'all_threads6_median_full_including_prefill_le20ms':all(v['median_full_ms_per_position']<=20 for v in primaries),
                         'all_threads6_repeat_ratio_le1p10':all(v['repeat_ratio_max_min_decode']<=1.10 for v in primaries)}
        assert before==(payload.stat().st_size,payload.stat().st_mtime_ns),'payload_changed_during_cost'
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,
                            'available_RAM_end_bytes':psutil.virtual_memory().available}
        result['decision']='exact_worker_cost_margin_licenses376_SAME_scoped_quality_accepted_rate' if all(result['gates'].values()) else 'unchanged_CPU_candidate_not_promoted_diagnose_measured_bottleneck_before_quality'
        result['scope']='NEW374 exact Windows physical worker placement plus ACTIVE/infinite waiting; SAME qualified374 executable/338 full256 weights, primary6 workers on0,2,4,6,8,10 with actual per-worker masks/TIDs/readbacks. Forced32 on consumed source9/64, all full outputs exact356 and profiles exact. Logical reads not physical DRAM. Apparatus cost includes encoder/crossKV/forced decode, not accepted natural generation/rate. No unchanged358/360/366 rerun, new original quality, larger-n or cross-family claim.'
        M.write(args.out,result);print(json.dumps({'gates':result['gates'],'resource':result['resource'],'sha256':M.digest(args.out),'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
