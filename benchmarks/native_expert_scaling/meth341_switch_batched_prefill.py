"""Actual all-bank forced-decoder cost, after immutable336 qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
import psutil
import meth324_switch_reference as M

OUT=M.ROOT/'results/native_expert_scaling/meth341_switch_batched_prefill'
PROTOCOL=M.DOC/'METH_341_SWITCH_BATCHED_PREFILL_PROTOCOL_20261003.md'
QUALIFIED=M.DOC/'meth336_switch_w8a8_contract_result.json'
QUALIFIED_SHA='9ad429178abce3a8451162dff788480e5d87eb2b5fcc8537ed84edffcccbe046'
RECOVERED=M.DOC/'meth338_switch_tensor_recovery_result.json'
RECOVERED_SHA='19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'
BASELINE=M.DOC/'meth340_switch_w8a8_cost_resume_result.json'
BASELINE_SHA='a93863da56b79cc2d4ed94ed8712d712a42456c1ac2cced27f0091e53830ee32'
SOURCE=M.ROOT/'benchmarks/native_expert_scaling/meth341_switch_batched_prefill.c'
ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth341_switch_batched_prefill_entry.c'
ENGINE=M.ROOT/'benchmarks/phase60/engine.c'


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
    result={'experiment':'METH-341-exact-batched-prefill-complete-CPU-cost','commands':[],'bridges':[],'measurements':[],'profiles':[]}
    try:
        for path in (Path(__file__),SOURCE,ENTRY,PROTOCOL,QUALIFIED,RECOVERED,BASELINE,ENGINE,Path(M.__file__)):M.committed(path)
        qualified=json.loads(QUALIFIED.read_text());recovered=json.loads(RECOVERED.read_text())
        assert M.digest(BASELINE)==BASELINE_SHA
        baseline_record=json.loads(BASELINE.read_text())
        assert M.digest(QUALIFIED)==QUALIFIED_SHA
        assert all(qualified['gates'].values()) and M.digest(RECOVERED)==RECOVERED_SHA
        assert qualified['recovered338_sha256']==RECOVERED_SHA
        own_process_tree={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            command=' '.join(process.info['cmdline'] or [])
            if process.pid not in own_process_tree and process.info['name'].lower() in ('python.exe','meth336_switch_w8a8.exe'):
                assert not ('benchmarks/native_expert_scaling/' in command.replace('\\','/')),'another_model_worker_before_timing'
        compiler=Path(qualified['compile']['argv'][0]);dll=OUT.parent/'meth336_switch_w8a8_contract/libomp.dll'
        assert M.digest(compiler)==qualified['compile']['compiler_sha256'] and M.digest(dll)==qualified['compile']['runtime_sha256']
        result.update({'controller_sha256':M.digest(__file__),'source_sha256':M.digest(SOURCE),'entry_sha256':M.digest(ENTRY),
                       'engine_sha256':M.digest(ENGINE),'protocol_sha256':M.digest(PROTOCOL),'qualified336_sha256':M.digest(QUALIFIED),
                       'recovered338_sha256':RECOVERED_SHA,'preserved340_baseline_sha256':BASELINE_SHA,'host':{'platform':platform.platform(),'logical_cpus':psutil.cpu_count(),
                       'physical_cpus':psutil.cpu_count(logical=False),'physical_RAM_bytes':psutil.virtual_memory().total,
                       'available_RAM_before_bytes':psutil.virtual_memory().available}})
        stage='fresh_full_target_hash';artifact=recovered['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest'])
        before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert before[0]==artifact['bytes'] and stream_digest(payload,start)==artifact['sha256'] and M.digest(spec)==artifact['manifest_sha256']
        result['artifact']=artifact
        OUT.mkdir(parents=True);shutil.copyfile(dll,OUT/'libomp.dll');binary=OUT/'meth341_switch_batched_prefill.exe'
        command=[str(compiler),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',
                 '-DSILICON_SWITCH_BATCHED_PREFILL',str(ENGINE),'-o',str(binary)]
        stage='compile';compiled=subprocess.run(command,capture_output=True,timeout=120)
        (OUT/'compile.stdout.log').write_bytes(compiled.stdout);(OUT/'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode==0,compiled.stderr.decode(errors='replace')
        result['compile']={'argv':command,'binary_sha256':M.digest(binary),'compiler_sha256':M.digest(compiler),'runtime_sha256':M.digest(dll)}
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def run(control,threads,profile,warmups,repetitions,label):
            nonlocal maximum
            prefix=OUT/label;argv=[str(binary),str(spec),','.join(map(str,control['source_ids'])),','.join(map(str,control['decoder_ids'])),
                str(prefix),str(threads),str(profile),str(warmups),str(repetitions),'0']
            env['OMP_NUM_THREADS']=str(threads)
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env)
                while child.poll() is None:
                    try:
                        rss=psutil.Process(child.pid).memory_info().rss+psutil.Process().memory_info().rss;maximum=max(maximum,rss)
                        if rss>32<<30 or time.monotonic()-start>1800:child.kill();child.wait();raise RuntimeError('cost_resource_guard')
                    except psutil.NoSuchProcess:pass
                    time.sleep(.25)
            result['commands'].append({'argv':argv,'returncode':child.returncode,'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),
                                       'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode)
            rows=[json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text().splitlines()]
            assert len(rows)==warmups+repetitions
            for row in rows:
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
                    previous=next(v for v in baseline_record['measurements'] if v['source_length']==len(control['source_ids']) and v['threads']==threads)
                    assert row['output_sha256']==previous['rows'][0]['output_sha256'],'complete_batched_prefill_exact340'
                    row['complete_output_exact340']=True
            assert len({r['output_sha256'] for r in rows})==1,'repeat_numeric_identity'
            return rows
        stage='exact_complete336_bridges'
        for i,control in enumerate(qualified['cases']):
            for threads in (1,6):
                for profile in (0,1):
                    rows=run(control,threads,profile,0,1,f'bridge{i}.t{threads}.p{profile}')
                    exact=rows[0]['output_sha256']==control['native_sha256'];assert exact,'new_counter_thread_bridge_exact336'
                    result['bridges'].append({'case':i,'threads':threads,'profile':profile,'exact336':exact,'output_sha256':rows[0]['output_sha256']})
        fixtures=[]
        for i,source_length in enumerate((9,64)):
            base=qualified['cases'][i];source=(base['source_ids'][:-1]*8)[:source_length-1]+[1]
            decoder=(base['decoder_ids']*8)[:32]
            fixtures.append({'source_ids':source,'decoder_ids':decoder,'source_length':source_length})
        result['fixtures']=fixtures
        for i,fixture in enumerate(fixtures):
            for threads in (1,6):
                rows=run(fixture,threads,0,0,1,f'long_bridge.s{fixture["source_length"]}.t{threads}')
                result['bridges'].append({'case':f'long_source{fixture["source_length"]}','threads':threads,'profile':0,'exact336':True,'exact340':rows[0]['complete_output_exact340'],'output_sha256':rows[0]['output_sha256']})
        order=((0,1),(1,6),(0,6),(1,1))
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
        result['gates']={'all_complete336_thread_and_counter_bridges_exact':all(v['exact336'] for v in result['bridges']),
                         'all_profile_outputs_exact_primary':all(v['exact_primary_outputs'] for v in result['profiles']),
                         'all_threads6_median_decode_le20ms':all(v['median_decode_ms_per_position']<=20 for v in primaries),
                         'all_threads6_median_full_including_prefill_le20ms':all(v['median_full_ms_per_position']<=20 for v in primaries),
                         'all_threads6_repeat_ratio_le1p10':all(v['repeat_ratio_max_min_decode']<=1.10 for v in primaries)}
        assert before==(payload.stat().st_size,payload.stat().st_mtime_ns),'payload_changed_during_cost'
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,
                            'available_RAM_end_bytes':psutil.virtual_memory().available}
        result['decision']='cost_margin_licenses_NEW_untouched_original_donor_quality_protocol' if all(result['gates'].values()) else 'unchanged_CPU_candidate_not_promoted_diagnose_measured_bottleneck_before_quality'
        result['scope']='Complete actual14.818GB all-bank target, forced32 decoder positions on repeated consumed9/64 source fixtures. Logical addressed bytes are not physical DRAM traffic. Matrix-only profile includes clock overhead and is not primary latency. No accepted-generation rate, untouched quality, useful larger-n/LUT or cross-family claim.'
        M.write(args.out,result);print(json.dumps({'gates':result['gates'],'resource':result['resource'],'sha256':M.digest(args.out),'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
