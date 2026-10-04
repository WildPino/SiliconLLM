"""Paired consumed actualnested learnedbank-size intervention; no original reload/rate."""
import argparse
import ast
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
import meth369_switch_bank_metrics as G

OUT=M.ROOT/'results/native_expert_scaling/meth372_switch_nested_bank_usefulness'
PROTOCOL=M.DOC/'METH_372_SWITCH_NESTED_BANK_USEFULNESS_PROTOCOL_20261004.md'
NUMERIC=M.DOC/'meth371_switch_nested_bank_contract_result.json'
NUMERIC_SHA='2d534664bd715c771e62c33fa50f4d85211dc994efefbddaecef0d7e0c80614d'
QUALITY=M.DOC/'meth363_switch_all_a16_multi_span_quality_result.json'
QUALITY_SHA='ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'
MANIFEST=M.DOC/'meth362_switch_multi_span_manifest.json'
MANIFEST_SHA='c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'
RECOVERED=M.DOC/'meth338_switch_tensor_recovery_result.json'
RECOVERED_SHA='19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'
OLD_METRICS=M.ROOT/'benchmarks/native_expert_scaling/meth351_switch_multi_span_quality.py'
ENGINE=M.ROOT/'benchmarks/phase60/engine.c'


def read_output(path):
    data=Path(path).read_bytes();assert data[:8]==b'SWR32O01'
    s,t,d,enc,dec,vocab,count=np.frombuffer(data,dtype='<u4',offset=8,count=7).tolist();offset=36
    def read(shape):
        nonlocal offset
        n=int(np.prod(shape));a=np.frombuffer(data,dtype='<f4',count=n,offset=offset).reshape(shape);offset+=4*n
        assert np.isfinite(a).all();return a
    e=read((enc+2,s,d));h=read((t,dec+2,d));logits=read((t,vocab))
    assert len(data)==offset+count*12
    routes=np.frombuffer(data,dtype=[('e','<i4'),('a','<i4'),('p','<f4')],offset=offset,count=count)
    r=np.column_stack([routes['e'],routes['a'],routes['p']]);assert np.isfinite(r).all()
    return e,h,logits,r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-372-paired-consumed-learned-bank-identity-usefulness','commands':[],'baseline_cases':[],'controls':[]}
    def guard(child=None):
        nonlocal maximum
        rss=psutil.Process().memory_info().rss
        if child is not None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        maximum=max(maximum,rss)
        if rss>16<<30 or time.monotonic()-start>1800:
            if child is not None and child.poll() is None:child.kill();child.wait()
            raise RuntimeError('bank_usefulness_30min_16GiB')
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,NUMERIC,QUALITY,MANIFEST,RECOVERED,OLD_METRICS,ENGINE,Path(M.__file__),Path(A.__file__),Path(I.__file__),Path(G.__file__)):M.committed(path)
        for path,sha in ((NUMERIC,NUMERIC_SHA),(QUALITY,QUALITY_SHA),(MANIFEST,MANIFEST_SHA),(RECOVERED,RECOVERED_SHA)):assert digest(path)==sha
        numeric=json.loads(NUMERIC.read_text(encoding='utf-8'));quality=json.loads(QUALITY.read_text(encoding='utf-8'));cohort=json.loads(MANIFEST.read_text(encoding='utf-8'));recovered=json.loads(RECOVERED.read_text(encoding='utf-8'))
        assert all(numeric['gates'].values()) and all(quality['gates'].values()) and all(cohort['gates'].values()) and all(recovered['gates'].values())
        assert numeric['input_sha256']['363']==QUALITY_SHA and numeric['input_sha256']['362']==MANIFEST_SHA
        # ALL inherited metrics must remain exact351 function AST, separate
        # prospectively declared primary harm bounds are the only new rubric.
        old=ast.parse(OLD_METRICS.read_text(encoding='utf-8'));new=ast.parse(Path(G.__file__).read_text(encoding='utf-8'))
        for name in ('metrics','edit_distance','lcs','task','paired_interval'):
            a=next(v for v in old.body if isinstance(v,ast.FunctionDef) and v.name==name);b=next(v for v in new.body if isinstance(v,ast.FunctionDef) and v.name==name)
            assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
        assert numeric['native_threads']==quality['native_threads']==1 and numeric['native_process_affinity']==quality['native_process_affinity']==[0]
        assert A.physical_topology()==numeric['fresh_topology'];affinity=[0]
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            cmd=' '.join(process.info['cmdline'] or []).replace('\\','/')
            if process.pid not in own and process.info['name'].lower() in ('python.exe','meth356_switch_all_a16.exe','meth365_switch_encoder_batches.exe'):
                assert not ('benchmarks/native_expert_scaling/' in cmd or 'meth356_switch_all_a16.exe' in cmd or 'meth365_switch_encoder_batches.exe' in cmd),'concurrent_model_or_native_worker'
        binary=Path(numeric['compile']['argv'][-1]);assert digest(binary)==numeric['compile']['binary_sha256']==quality['native_binary_sha256']
        assert digest(binary.parent/'libomp.dll')==numeric['compile']['runtime_sha256'] and digest(numeric['compile']['argv'][0])==numeric['compile']['compiler_sha256']
        artifact=recovered['artifact'];payload=Path(artifact['payload']);before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert before[0]==artifact['bytes'] and digest(payload)==artifact['sha256'] and digest(artifact['manifest'])==artifact['manifest_sha256']
        specs={};configs={};target_artifacts=[];subset_before={}
        for row in numeric['targets']:
            target=row['artifact'];n=row['n'];assert n in (64,128) and row['passed'] and n==target['n']
            assert digest(target['payload'])==target['sha256'] and digest(target['manifest'])==target['manifest_sha256'] and digest(target['metadata'])==target['metadata_sha256']
            metadata=json.loads(Path(target['metadata']).read_text(encoding='utf-8'));config=target['original_config'];entries=metadata['tensors']
            assert config==metadata['original_config'] and config['num_experts']==n and len(entries)==24*n+248==target['namespace']
            I.read_manifest(target['manifest'],config,entries,target['payload']);specs[n]=target['manifest'];configs[n]=config;target_artifacts.append(target)
            path=Path(target['payload']);subset_before[n]=(path.stat().st_size,path.stat().st_mtime_ns)
        assert set(specs)=={64,128}
        result.update({'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'numeric371_sha256':NUMERIC_SHA,'quality363_sha256':QUALITY_SHA,
                       'manifest362_sha256':MANIFEST_SHA,'recovered338_sha256':RECOVERED_SHA,'artifact':artifact,'compile':numeric['compile'],
                       'native_threads':1,'native_process_affinity':affinity,'fixed_actual_n':[64,128,256],'target_artifacts':target_artifacts,'paired_unit':'book','metrics351_AST_exact':True})
        OUT.mkdir(parents=True);baseline_dir=M.ROOT/'results/native_expert_scaling/meth363_switch_all_a16_multi_span_quality'
        for bi,book in enumerate(cohort['items']):
            for ci,case in enumerate(book['cases']):
                stage=f'baseline{bi}_{ci}';oldcase=quality['books'][bi]['cases'][ci];teacher=baseline_dir/f'book{bi}.case{ci}.0.bin';gen=baseline_dir/f'book{bi}.case{ci}.generation.0.bin'
                assert digest(teacher)==oldcase['native_output_sha256'] and digest(gen)==oldcase['generation']['native_generation_sha256']
                arrays=read_output(teacher);natural=read_output(gen);m=G.metrics(arrays[2],case['target_ids'],case['masked_spans_ids']);task=G.task(natural[2].argmax(-1).tolist(),case['masked_spans_ids'])
                assert m==oldcase['native'] and task==oldcase['generation']['native']
                result['baseline_cases'].append({'book':bi,'case':ci,'teacher_sha256':digest(teacher),'natural_sha256':digest(gen),'metrics':m,'task':task})
        assert len(result['baseline_cases'])==96
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def run(spec,source,decoder,label,generate=False):
            prefix=OUT/label
            if generate:argv=[str(binary),'--generate',str(spec),','.join(map(str,source)),str(prefix),'1','64','32095','0','0','1','0']
            else:argv=[str(binary),str(spec),','.join(map(str,source)),','.join(map(str,decoder)),str(prefix),'1','0','0','1','0']
            with (OUT/(label+'.stdout.log')).open('wb') as out,(OUT/(label+'.stderr.log')).open('wb') as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env)
                try:process=psutil.Process(child.pid);process.cpu_affinity(affinity);actual_affinity=process.cpu_affinity();assert actual_affinity==affinity
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    raise
                while child.poll() is None:guard(child);time.sleep(.25)
            assert child.returncode==0,(label,child.returncode)
            result['commands'].append({'argv':argv,'returncode':child.returncode,'actual_affinity':actual_affinity,'stdout_sha256':digest(OUT/(label+'.stdout.log')),'stderr_sha256':digest(OUT/(label+'.stderr.log'))})
            path=Path(str(prefix)+'.0.bin');arrays=read_output(path);rows=[json.loads(v) for v in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()]
            assert len(rows)==1 and rows[0]['repetition']==0
            if generate:assert rows[0]['generated_ids']==arrays[2].argmax(-1).tolist()
            return arrays,digest(path),rows[0]
        for n in (64,128):
            record={'n':n,'cases':[]};result['controls'].append(record)
            for bi,book in enumerate(cohort['items']):
                for ci,case in enumerate(book['cases']):
                    stage=f'n{n}.book{bi}.case{ci}';label=f'n{n}.book{bi}.case{ci}'
                    arrays,sha,row=run(specs[n],case['source_ids'],case['decoder_ids'],label)
                    natural,gsha,grow=run(specs[n],case['source_ids'],[],label+'.gen',True)
                    qualified=next(v for v in numeric['targets'] if v['n']==n)
                    bridge=next((v for v in qualified['consumed'] if v['book']==bi and v['case']==ci),None)
                    if bridge:assert sha==bridge['native_sha256'] and gsha==bridge['natural_sha256'],'same371_independent_consumed_bridge_changed'
                    m=G.metrics(arrays[2],case['target_ids'],case['masked_spans_ids']);task=G.task(grow['generated_ids'],case['masked_spans_ids'])
                    consulted=I.consultations(configs[n],29,len(grow['generated_ids']),natural[3],0)
                    sidecar=OUT/(label+'.consultations.json');M.write(sidecar,consulted)
                    original=quality['books'][bi]['cases'][ci]['original'];original_choices=np.asarray(original['greedy_teacher_forced_ids'])
                    original_agreement=float(np.mean(arrays[2].argmax(-1)==original_choices))
                    original_mask_agreement=float(np.mean(arrays[2].argmax(-1)[G.MASK]==original_choices[G.MASK]))
                    record['cases'].append({'book':bi,'case':ci,'source_id':book['source_id'],'metrics':m,'task':task,'original_top1_agreement':original_agreement,'masked_original_top1_agreement':original_mask_agreement,'forced_sha256':sha,'natural_sha256':gsha,
                                            'consultations_sha256':digest(sidecar),'apparatus_generation_row_not_rate':grow})
                print(json.dumps({'n':n,'book':bi,'cases':len(record['cases'])}),flush=True)
            assert len(record['cases'])==96
            deltas={key:[] for key in ('all_NLL','masked_NLL','teacher_mask_accuracy','teacher_field_exact','generated_known_token_accuracy','generated_field_exact','generated_LCS_F1','generated_health')}
            for bi in range(24):
                pairs=[(result['baseline_cases'][bi*4+ci],record['cases'][bi*4+ci]) for ci in range(4)]
                for key,calc in (
                    ('all_NLL',lambda a,b:b['metrics']['mean_nll']-a['metrics']['mean_nll']),
                    ('masked_NLL',lambda a,b:b['metrics']['mean_span_nll']-a['metrics']['mean_span_nll']),
                    ('teacher_mask_accuracy',lambda a,b:(b['metrics']['correct_span_tokens']-a['metrics']['correct_span_tokens'])/8),
                    ('teacher_field_exact',lambda a,b:(b['metrics']['correct_fields']-a['metrics']['correct_fields'])/4),
                    ('generated_known_token_accuracy',lambda a,b:b['task']['known_token_accuracy']-a['task']['known_token_accuracy']),
                    ('generated_field_exact',lambda a,b:b['task']['known_field_exact_accuracy']-a['task']['known_field_exact_accuracy']),
                    ('generated_LCS_F1',lambda a,b:b['task']['known_field_lcs_f1']-a['task']['known_field_lcs_f1']),
                    ('generated_health',lambda a,b:int(b['task']['healthy_complete_nonempty_fields'])-int(a['task']['healthy_complete_nonempty_fields']))):
                    deltas[key].append(float(np.mean([calc(a,b) for a,b in pairs])))
            record['paired_diagnostic95']={key:G.paired_interval(values) for key,values in deltas.items()}
            record['primary_masked_NLL_harm']=G.primary_harm_interval(deltas['masked_NLL'])
            record['primary_generated_field_harm']=G.primary_harm_interval(deltas['generated_field_exact'])
        assert before==(payload.stat().st_size,payload.stat().st_mtime_ns)
        for n in (64,128):
            path=Path(next(v['payload'] for v in target_artifacts if v['n']==n));assert subset_before[n]==(path.stat().st_size,path.stat().st_mtime_ns)
        result['gates']={'ALL96_baseline_bytes_and351_metrics_exact363':True,'both_actual64_128_ALL96_completed_with_retained_function_ids':True,
                         'all_native_process_affinity_readbacks_exact':all(v['actual_affinity']==affinity for v in result['commands'])}
        for record in result['controls']:
            n=record['n'];result['gates'][f'n{n}_masked_NLL_harm_lower98p75_ge0p05']=record['primary_masked_NLL_harm']['one_sided_lower98p75']>=.05
            result['gates'][f'n{n}_generated_field_loss_upper98p75_le_minus0p02']=record['primary_generated_field_harm']['one_sided_upper98p75']<=-.02
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'available_RAM_end_bytes':psutil.virtual_memory().available}
        result['decision']='paired_nested_bank_removal_harm_supports_usefulness_of_additional_real_source_functions' if all(result['gates'].values()) else 'frozen_additional_real_bank_usefulness_criteria_not_all_established'
        result['scope']='Diagnostic paired consumed362/363 cohort, actualphysicallysmaller64/128 nestedrealexpert subsets versusfull256 SAME14.664B donor/precision/core/head/top1/capacity/runtime. Retainfirstnoriginalexpertpairs/F32classifierrows, sourceexposure/trainingoriginally256; winneravailability/normalization/ownstates/routes change underthis intervention, not independentlytrainedsmall-nmodels or universalmore-noptimality. Four primary book-bootstrap98.75 bounds Bonferroni familyalpha.05; no new untouched original-quality claim, no each-expert proof, no larger-than256 or universaln extrapolation, no accepted-rate/physicalDRAM extrapolation. Own control routes change naturally; no baseline-route replay.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':digest(args.out),'gates':result['gates'],'resource':result['resource'],'primary':[{'n':r['n'],'masked_NLL':r['primary_masked_NLL_harm'],'generated_field':r['primary_generated_field_harm']} for r in result['controls']]}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
