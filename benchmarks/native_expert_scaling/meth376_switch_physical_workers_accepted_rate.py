"""Exact374 SAME363 output artifact, physical CPU6 active wait accepted FULL model rate."""
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

COST=M.DOC/'meth375_switch_physical_workers_cost_result.json'
COST_SHA='28b535b6a01ba67fcf45aaa008d8cf033d4f4cbc1aefcdc17c261a8749244375'
QUALITY=M.DOC/'meth363_switch_all_a16_multi_span_quality_result.json'
QUALITY_SHA='ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'
MANIFEST=M.DOC/'meth362_switch_multi_span_manifest.json'
MANIFEST_SHA='c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'
NUMERIC=M.DOC/'meth374_switch_physical_workers_contract_result.json'
NUMERIC_SHA='4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'
PROTOCOL=M.DOC/'METH_376_SWITCH_PHYSICAL_WORKERS_ACCEPTED_RATE_PROTOCOL_20261004.md'
OUT=M.ROOT/'results/native_expert_scaling/meth376_switch_physical_workers_accepted_rate'
ENGINE=M.ROOT/'benchmarks/phase60/engine.c'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-376-SAME-quality-artifact-accepted-FULL-generation-rate','cases':[],'commands':[]}
    def guard(child=None):
        nonlocal maximum
        rss=psutil.Process().memory_info().rss
        if child is not None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        maximum=max(maximum,rss)
        if rss>16<<30 or time.monotonic()-start>1800:
            if child is not None and child.poll() is None:child.kill();child.wait()
            raise RuntimeError('accepted_rate_resource_guard')
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while block:=f.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),QUALITY,MANIFEST,NUMERIC,COST,PROTOCOL,ENGINE,Path(A.__file__),Path(M.__file__)):M.committed(path)
        assert digest(QUALITY)==QUALITY_SHA and digest(MANIFEST)==MANIFEST_SHA and digest(NUMERIC)==NUMERIC_SHA
        quality=json.loads(QUALITY.read_text(encoding='utf-8'));manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));numeric=json.loads(NUMERIC.read_text(encoding='utf-8'))
        for name,sha in numeric['source_sha256'].items():
            path=M.ROOT/'benchmarks/native_expert_scaling'/name;M.committed(path);assert M.digest(path)==sha
        assert all(quality['gates'].values()) and all(numeric['gates'].values()) and all(manifest['gates'].values())
        assert numeric['input_sha256']['363']==QUALITY_SHA and numeric['input_sha256']['362']==MANIFEST_SHA
        assert quality['manifest362_sha256']==MANIFEST_SHA and numeric['gates']['ALL96_teacher_and_own_natural_complete_bytes_exact363']
        assert M.digest(COST)==COST_SHA
        cost=json.loads(COST.read_text(encoding='utf-8'));assert all(cost['gates'].values()) and cost['numeric374_sha256']==NUMERIC_SHA
        assert numeric['primary_threads']==cost['primary_native_threads']==6 and numeric['primary_affinity']==cost['native_process_affinity']==[0,2,4,6,8,10]
        assert numeric['runtime_environment']==cost['runtime_environment']
        assert A.physical_topology()==cost['fresh_topology'];affinity=cost['native_process_affinity']
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            cmd=' '.join(process.info['cmdline'] or []).replace('\\','/')
            if process.pid not in own and process.info['name'].lower() in ('python.exe','meth356_switch_all_a16.exe','meth365_switch_encoder_batches.exe','meth374_switch_physical_workers.exe'):
                assert not ('benchmarks/native_expert_scaling/' in cmd or 'meth356_switch_all_a16.exe' in cmd or 'meth374_switch_physical_workers.exe' in cmd or 'meth365_switch_encoder_batches.exe' in cmd),'concurrent_model_or_native_timing_worker'
        binary=Path(numeric['compile']['argv'][-1]);dll=binary.parent/'libomp.dll';compiler=Path(numeric['compile']['argv'][0])
        assert digest(binary)==numeric['compile']['binary_sha256']==cost['compile']['binary_sha256']
        assert digest(dll)==numeric['compile']['runtime_sha256'] and digest(compiler)==numeric['compile']['compiler_sha256']
        artifact=numeric['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest']);before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert before[0]==artifact['bytes'] and digest(payload)==artifact['sha256'] and digest(spec)==artifact['manifest_sha256']
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'quality363_sha256':QUALITY_SHA,'manifest362_sha256':MANIFEST_SHA,
                       'numeric374_sha256':NUMERIC_SHA,'cost375_sha256':COST_SHA,'runtime_environment':numeric['runtime_environment'],'native_threads':6,'native_process_affinity':affinity,'artifact':artifact,'compile':numeric['compile'],
                       'timing_boundary':{'included':['encoder','cross_kv','cached_decoder','greedy_argmax','terminal_checks'],
                                          'excluded':['process_startup','manifest_load','tokenization_of_already_fixed_ids','output_serialization','cleanup'],
                                          'manifest_load_seconds_reported_per_row':True,'startup_and_tokenization_not_timed':True},
                       'host':{'physical_cpus':psutil.cpu_count(logical=False),'logical_cpus':psutil.cpu_count(),'RAM_bytes':psutil.virtual_memory().total,'available_RAM_before_bytes':psutil.virtual_memory().available}})
        OUT.mkdir(parents=True);env={k:v for k,v in os.environ.items() if not k.startswith(('OMP_','KMP_','GOMP_','SILICON_WORKER_BINDING_'))};env.update(numeric['runtime_environment']);env['OMP_NUM_THREADS']='6'
        for bi,book in enumerate(manifest['items']):
            for ci,case in enumerate(book['cases']):
                stage=f'book{bi}_case{ci}';label=f'book{bi}.case{ci}';prefix=OUT/label
                argv=[str(binary),'--generate',str(spec),','.join(map(str,case['source_ids'])),str(prefix),'6','64','32095','0','1','3','0']
                with (OUT/(label+'.stdout.log')).open('wb') as out,(OUT/(label+'.stderr.log')).open('wb') as err:
                    child=subprocess.Popen(argv,stdout=out,stderr=err,env=env)
                    try:
                        native_process=psutil.Process(child.pid);native_process.cpu_affinity(affinity);actual_affinity=native_process.cpu_affinity();assert actual_affinity==affinity
                    except BaseException:
                        if child.poll() is None:child.kill();child.wait()
                        raise
                    while child.poll() is None:guard(child);time.sleep(.25)
                assert child.returncode==0,(label,child.returncode)
                result['commands'].append({'argv':argv,'returncode':child.returncode,'actual_affinity':actual_affinity,'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
                rows=[json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()];assert [r['repetition'] for r in rows]==[-1,0,1,2]
                accepted=quality['books'][bi]['cases'][ci]['generation']['native']
                expected_sha=quality['books'][bi]['cases'][ci]['generation']['native_generation_sha256']
                for row in rows:
                    assert row['worker_physical_cores']==6
                    assert [v['slot'] for v in row['worker_affinity']]==list(range(6))
                    assert [v['actual_mask'] for v in row['worker_affinity']]==[1<<cpu for cpu in affinity]
                    assert len({v['windows_thread_id'] for v in row['worker_affinity']})==6
                    assert all(v['group']==0 for v in row['worker_affinity'])
                    assert all(v['actual_mask']==1<<affinity[v['slot']] and v['group']==0 for v in row['worker_binding_events'])
                    assert set(v['slot'] for v in row['worker_binding_events'])==set(range(6))
                    row['output_sha256']=M.digest(Path(str(prefix)+f'.{row["repetition"]}.bin'))
                    assert row['output_sha256']==expected_sha and row['generated_ids']==accepted['generated_ids'],'SAME_quality_generated_output_changed'
                    n=row['actual_generated_tokens'];assert n==accepted['generated_tokens'] and row['source_tokens']==29
                    counts=row['counters'][1]
                    assert sum(v['code_bytes'] for v in counts)==123764736*n and sum(v['scale_bytes'] for v in counts)==534016*n
                    assert sum(v['f32_bytes'] for v in counts)==4718592*n and counts[2]['calls']==6*n and counts[3]['calls']==n
                result['cases'].append({'book':bi,'case':ci,'source_id':book['source_id'],'healthy_accepted':accepted['healthy_complete_nonempty_fields'],
                                        'generated_tokens':accepted['generated_tokens'],'prose_tokens':accepted['prose_tokens'],'rows':rows,'exact_quality_output':True})
            print(json.dumps({'completed_book':bi,'cases':len(result['cases'])}),flush=True)
        assert len(result['cases'])==96 and before==(payload.stat().st_size,payload.stat().st_mtime_ns)
        token_books=np.zeros(24);prose_books=np.zeros(24);time_books=np.zeros(24);repeat_times=np.zeros(3)
        for case in result['cases']:
            rows=[r for r in case['rows'] if r['repetition']>=0];full=[r['full_generation_seconds'] for r in rows]
            time_books[case['book']]+=float(np.median(full));repeat_times+=full
            if case['healthy_accepted']:
                token_books[case['book']]+=case['generated_tokens'];prose_books[case['book']]+=case['prose_tokens']
        rng=np.random.default_rng(364364);indices=rng.integers(0,24,size=(10000,24))
        rates=token_books[indices].sum(-1)/time_books[indices].sum(-1)
        prose_rates=prose_books[indices].sum(-1)/time_books[indices].sum(-1)
        rate=float(token_books.sum()/time_books.sum());prose_rate=float(prose_books.sum()/time_books.sum())
        result['accepted_full_rate']={'accepted_generated_tokens':int(token_books.sum()),'accepted_prose_tokens':int(prose_books.sum()),
            'ALL_case_median_full_seconds':float(time_books.sum()),'tokens_per_second':rate,'one_sided_lower95':float(np.quantile(rates,.05)),
            'prose_tokens_per_second':prose_rate,'prose_one_sided_lower95':float(np.quantile(prose_rates,.05)),
            'accepted_cases':sum(c['healthy_accepted'] for c in result['cases']),'rejected_case_time_charged':True,
            'repetition_ALL_case_full_seconds':repeat_times.tolist(),'aggregate_repeat_ratio_max_min':float(repeat_times.max()/repeat_times.min()),
            'bootstrap_unit':'book','draws':10000,'seed':364364,'context_source_tokens':29}
        result['gates']={'all_actual_worker_affinity_readbacks_exact':True,'all_native_process_affinity_readbacks_exact':all(v['actual_affinity']==affinity for v in result['commands']),'all96_SAME_quality_outputs_exact':True,'accepted_full_lower95_ge50':result['accepted_full_rate']['one_sided_lower95']>=50,
                         'aggregate_ALL_case_repeat_ratio_le1p10':result['accepted_full_rate']['aggregate_repeat_ratio_max_min']<=1.10}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'available_RAM_end_bytes':psutil.virtual_memory().available}
        result['decision']='single_family_declared_multispan_quality_and_accepted_full_rate_qualified' if all(result['gates'].values()) else 'unchanged_full_rate_not_qualified_preserve_then_diagnose_phases'
        result['scope']='Quality363 full256 outputs inherited ONLY via374 ALL96 full teacher/own-natural byte equivalence; SAME374 binary/338 weights/375qualified physical CPU6 workers0,2,4,6,8,10 ACTIVE/infinite waiting. Complete encoder/crossKV/decoder/greedy/stop timed; startup/initial worker setup/load/pre-fixed tokenization/serialization/cleanup excluded; any lazy worker bind during inference included. Ordinary generated IDs explicitly include structural sentinels; separate prose rate. One warmup/hash priming, no physical DRAM/cold-service/long-context/larger-n/multiple-family/100B extrapolation. Old364 failed rate retained; this is a new execution variable.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'gates':result['gates'],'accepted_full_rate':result['accepted_full_rate'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
