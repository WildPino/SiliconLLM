"""NEW multispan prediction/free-generation/known-answer task, ORIGINAL primary."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
import argparse
import ctypes
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import psutil
import torch
import transformers
from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
import meth324_switch_reference as M
import meth379_switch_tensor_binding as T
import meth328_switch_native_contract as B
import meth351_switch_generation_reference as H
import meth359_switch_cost_topology as A
import meth387_switch_teacher_reference as J

MANIFEST=M.DOC/'meth382_switch_multi_span_manifest.json'
MANIFEST_SHA='96b3ffd09b0bdd42d7c990006e8744e39b5e1f32b7ae40aaf3b3dca98bf4db78'
COST=M.DOC/'meth383_switch_base128_cost_result.json'
COST_SHA='8e9a17fbaceed0b694a0ba77acb5041bb5f2c1ba3455c5959302b675a97c366a'
BOUND=M.DOC/'meth379_switch_tensor_binding_result.json'
BOUND_SHA='e40710575189b2c635226200a83f56c3242e68915bd0901926bbb9efb728dbe2'
CONTROL=M.DOC/'meth381_switch_base128_contract_result.json'
CONTROL_SHA='6ede7a90fbf2716381d31456897eb41b01a0c6ed9010e7ba3b3af91064831a6e'
RECOVERED=M.DOC/'meth380_switch_base128_export_result.json'
RECOVERED_SHA='6be77281f7ecca73c47eab7f318fed2cb27699f18071b3f3781ab9bce54881ee'
PROTOCOL=M.DOC/'METH_387_SWITCH_BASE128_MULTI_SPAN_QUALITY_PROTOCOL_20261004.md'
FAILURE=M.DOC/'meth386_switch_base128_multi_span_quality_result.failure.json'
FAILURE_SHA='2375874a5bcb9da939595ed2f080063f3554b0800749eeea5b45a6caeab94aeb'
OUT=M.ROOT/'results/native_expert_scaling/meth387_switch_base128_multi_span_quality'


def trim():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetProcessWorkingSetSize.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
    assert kernel.SetProcessWorkingSetSize(kernel.GetCurrentProcess(),ctypes.c_size_t(-1).value,ctypes.c_size_t(-1).value)


MASK=np.array([1,2,4,5,7,8,10,11])

def metrics(logits,target,spans):
    assert logits.shape==(14,32128) and len(target)==14 and np.asarray(spans).shape==(4,2)
    values=logits.astype(np.float64);shift=values.max(-1,keepdims=True)
    nll=np.log(np.exp(values-shift).sum(-1))+shift[:,0]-values[np.arange(14),target]
    choice=logits.argmax(-1);truth=np.asarray(spans).reshape(8)
    return {'mean_nll':float(nll.mean()),'mean_span_nll':float(nll[MASK].mean()),
            'correct_span_tokens':int(np.sum(choice[MASK]==truth)),
            'correct_fields':int(np.all((choice[MASK]==truth).reshape(4,2),axis=1).sum()),
            'all_span_teacher_forced_correct':bool(np.array_equal(choice[MASK],truth)),
            'greedy_teacher_forced_ids':choice.tolist(),'per_target_nll':nll.tolist()}

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a):
        new=[i+1]
        for j,y in enumerate(b):new.append(min(new[-1]+1,row[j+1]+1,row[j]+(x!=y)))
        row=new
    return row[-1]

def lcs(a,b):
    row=[0]*(len(b)+1)
    for x in a:
        new=[0]
        for j,y in enumerate(b):new.append(row[j]+1 if x==y else max(new[-1],row[j+1]))
        row=new
    return row[-1]

def task(ids,truth):
    fields=[[] for _ in range(4)];expected=0;current=-1;malformed=False;closed=False
    for token in ids:
        if token==1:break
        if token==0:malformed=True;continue
        if token>=32000:
            if expected<=4 and token==32099-expected:
                if expected==4:closed=True;current=-1
                else:current=expected
                expected+=1
            else:malformed=True
        elif 0<=current<4:fields[current].append(token)
        else:malformed=True
    if malformed:fields=[[] for _ in range(4)]
    trigrams=[[(field[i],field[i+1],field[i+2]) for i in range(max(len(field)-2,0))] for field in fields]
    total=sum(len(values) for values in trigrams)
    repeated=(total-sum(len(set(values)) for values in trigrams))/max(total,1)
    healthy=closed and expected==5 and not malformed and all(fields) and ids[-1]==32095 and repeated<=.50
    exact=sum(field==gold for field,gold in zip(fields,truth))
    correct=sum(sum(i<len(field) and field[i]==gold[i] for i in range(2)) for field,gold in zip(fields,truth))
    f1=np.mean([2*lcs(field,gold)/(len(field)+len(gold)) for field,gold in zip(fields,truth)])
    prose=[token for token in ids if 1<token<32000]
    return {'healthy_complete_nonempty_fields':bool(healthy),'fields':fields,'correct_fields':exact,
            'known_token_accuracy':correct/8,'known_field_exact_accuracy':exact/4,'known_field_lcs_f1':float(f1),
            'generated_ids':ids,'generated_tokens':len(ids),'prose_tokens':len(prose),'prose_ids':prose,
            'within_field_repeated_trigram_fraction':repeated,
            'cap_reached_without_terminal':len(ids)==64 and ids[-1] not in (1,32095)}

def paired_interval(values):
    rng=np.random.default_rng(351351);values=np.asarray(values,dtype=np.float64);assert values.shape==(24,)
    draws=values[rng.integers(0,24,size=(10000,24))].mean(-1)
    return {'mean':float(values.mean()),'one_sided_lower95':float(np.quantile(draws,.05)),
            'one_sided_upper95':float(np.quantile(draws,.95)),'bootstrap_unit':'book','draws':10000,'seed':351351}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-387-NEW-original128-multispan-original-primary-whole-quality','source_shards':[],'original_bridges':[],'native_bridges':[],'books':[],'commands':[]}
    def guard():
        nonlocal maximum
        rss=psutil.Process().memory_info().rss;maximum=max(maximum,rss)
        assert rss<=48<<30 and time.monotonic()-start<=5400,'quality_resource_guard'
    def stream_digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,MANIFEST,COST,BOUND,CONTROL,RECOVERED,T.ACQUISITION,T.HEADERS,B.ENGINE,
                     Path(M.__file__),Path(T.__file__),Path(B.__file__),Path(H.__file__),Path(A.__file__),Path(J.__file__),FAILURE):M.committed(path)
        for path,sha in ((MANIFEST,MANIFEST_SHA),(COST,COST_SHA),(BOUND,BOUND_SHA),(CONTROL,CONTROL_SHA),(RECOVERED,RECOVERED_SHA)):
            assert M.digest(path)==sha,str(path)
        assert M.digest(FAILURE)==FAILURE_SHA;result['preserved386_failure_sha256']=FAILURE_SHA
        cohort=json.loads(MANIFEST.read_text(encoding='utf-8'));cost=json.loads(COST.read_text(encoding='utf-8'));bound=json.loads(BOUND.read_text(encoding='utf-8'))
        numeric=json.loads(CONTROL.read_text(encoding='utf-8'));control={'gates':numeric['gates'],'cases':numeric['targets'][0]['engineering']};recovered=json.loads(RECOVERED.read_text(encoding='utf-8'))
        assert all(cohort['gates'].values()) and all(control['gates'].values()) and bound['passed']
        assert cost['gates']['all_threads6_repeat_ratio_le1p10'] is False
        assert all(v for k,v in cost['gates'].items() if k!='all_threads6_repeat_ratio_le1p10'), 'require_exact383_numeric_placement_and_median_gates'
        result['preserved383_stability_failure_sha256']=COST_SHA;result['performance_qualified']=False
        assert cost['primary_native_threads']==6 and cost['native_process_affinity']==[0,2,4,6,8,10]
        assert A.physical_topology()==cost['fresh_topology']
        affinity=cost['native_process_affinity']
        assert len({book['whole_source_utf8_sha256'] for book in cohort['items']})==24
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        binary=Path(cost['compile']['argv'][-1]);assert M.digest(binary)==cost['compile']['binary_sha256']
        assert M.digest(binary.parent/'libomp.dll')==cost['compile']['runtime_sha256']
        artifact=cost['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest']);payload_stat=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert payload_stat[0]==artifact['bytes'] and stream_digest(payload)==artifact['sha256'] and M.digest(spec)==artifact['manifest_sha256']
        for row in cohort['source_tokenizer_files']:assert stream_digest(row['path'])==row['sha256']
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'manifest382_sha256':MANIFEST_SHA,
                       'cost383_sha256':COST_SHA,'numeric381_sha256':CONTROL_SHA,'original379_binding_sha256':BOUND_SHA,
                       'artifact':artifact,'native_binary_sha256':cost['compile']['binary_sha256'],'native_threads':6,'runtime_environment':cost['runtime_environment'],'native_process_affinity':affinity,'original_threads':1,
                       'installed_model_source_sha256':M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))})
        OUT.mkdir(parents=True);torch.set_num_threads(1)
        result['teacher_reference387_sha256']=M.digest(J.__file__)
        result['official_cached_teacher_Tiny_controls']=J.tiny_controls(B,SwitchTransformersConfig,SwitchTransformersForConditionalGeneration)
        assert all(v['passed'] for v in result['official_cached_teacher_Tiny_controls']);guard()
        stage='fresh_all_original_coefficients_and_reference_model'
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**recovered['original_config']))
        namespace=model.state_dict();assert set(namespace)==set(bound['tensors']);seen=set()
        for shard in sorted(T.SOURCE.glob('pytorch_model-*-of-*.bin')):
            before=time.monotonic();state=torch.load(shard,weights_only=True,mmap=True,map_location='cpu')
            assert not seen.intersection(state)
            for ordinal,(name,value) in enumerate(sorted(state.items(),key=lambda item:item[1].data_ptr())):
                expected=bound['tensors'][name]
                assert value.dtype==torch.float32 and value.is_contiguous() and list(value.shape)==expected['shape']==list(namespace[name].shape)
                assert hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()==expected['sha256'],name
                seen.add(name)
                if ordinal%32==31:trim()
                guard()
            missing=model.load_state_dict(state,strict=False,assign=True);assert not missing.unexpected_keys
            del state;gc.collect();trim();guard();result['source_shards'].append({'name':shard.name,'tensor_names_so_far':len(seen),'seconds':time.monotonic()-before})
        assert len(seen)==3320 and seen==set(bound['tensors'])
        model.tie_weights();model.requires_grad_(False);model.eval()
        assert sum(p.numel() for p in model.parameters())==7415217408
        assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in model.parameters())
        result['original_source_identity']='Fresh ALL3320 canonical original128 F32 coefficient bytes379 plus config/tokenizer; no new ZIP envelope claim. Unmodified official CPU1 forward.'
        env={k:v for k,v in os.environ.items() if not k.startswith(('OMP_','KMP_','GOMP_','SILICON_WORKER_BINDING_'))};env.update(cost['runtime_environment']);env['OMP_NUM_THREADS']='6'
        def worker_checks(row):
            assert row['worker_physical_cores']==6 and [v['slot'] for v in row['worker_affinity']]==list(range(6))
            assert [v['actual_mask'] for v in row['worker_affinity']]==[1<<cpu for cpu in affinity]
            assert all(v['group']==0 for v in row['worker_affinity']) and len({v['windows_thread_id'] for v in row['worker_affinity']})==6
            assert all(v['actual_mask']==1<<affinity[v['slot']] and v['group']==0 for v in row['worker_binding_events'])
            assert set(v['slot'] for v in row['worker_binding_events'])==set(range(6))
        def native(source_ids,decoder_ids,label):
            nonlocal maximum
            prefix=OUT/label;argv=[str(binary),str(spec),','.join(map(str,source_ids)),','.join(map(str,decoder_ids)),str(prefix),'6','0','0','1','0']
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env)
                try:
                    native_process=psutil.Process(child.pid);native_process.cpu_affinity(affinity);actual_affinity=native_process.cpu_affinity();assert actual_affinity==affinity
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    raise
                while child.poll() is None:
                    try:
                        rss=psutil.Process().memory_info().rss+psutil.Process(child.pid).memory_info().rss;maximum=max(maximum,rss)
                        if rss>48<<30 or time.monotonic()-start>5400:child.kill();child.wait();raise RuntimeError('native_quality_resource_guard')
                    except psutil.NoSuchProcess:pass
                    time.sleep(.1)
            result['commands'].append({'argv':argv,'returncode':child.returncode,'actual_affinity':actual_affinity,'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode)
            row=json.loads((OUT/(label+'.stdout.log')).read_text(encoding='utf-8'));worker_checks(row)
            output=Path(str(prefix)+'.0.bin');return B.read_output(output),M.digest(output)
        def native_generation(source_ids,label,cap=64,closing=32095):
            nonlocal maximum
            prefix=OUT/label;argv=[str(binary),'--generate',str(spec),','.join(map(str,source_ids)),str(prefix),'6',str(cap),str(closing),'0','0','1','0']
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env)
                try:
                    native_process=psutil.Process(child.pid);native_process.cpu_affinity(affinity);actual_affinity=native_process.cpu_affinity();assert actual_affinity==affinity
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    raise
                while child.poll() is None:
                    try:
                        rss=psutil.Process().memory_info().rss+psutil.Process(child.pid).memory_info().rss;maximum=max(maximum,rss)
                        if rss>48<<30 or time.monotonic()-start>5400:child.kill();child.wait();raise RuntimeError('native_generation_resource_guard')
                    except psutil.NoSuchProcess:pass
                    time.sleep(.1)
            result['commands'].append({'argv':argv,'returncode':child.returncode,'actual_affinity':actual_affinity,'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode)
            row=json.loads((OUT/(label+'.stdout.log')).read_text(encoding='utf-8'));worker_checks(row);output=Path(str(prefix)+'.0.bin');arrays=B.read_output(output)
            assert row['generated_ids']==arrays[2].argmax(-1).tolist() and row['actual_generated_tokens']==len(row['generated_ids'])
            return row['generated_ids'],arrays,M.digest(output),row
        stage='fresh_original_and_native_consumed_bridges_before_NEW_scores'
        for index,case in enumerate(control['cases']):
            source=torch.tensor([case['source_ids']]);decoder=torch.tensor([case['decoder_ids']]);reference=B.torch_reference(model,source,decoder)
            official_teacher=J.official_teacher_cached(model,source,case['decoder_ids'])
            teacher_error=B.relative(official_teacher,reference[2]);assert teacher_error<=1e-6 and np.array_equal(official_teacher.argmax(-1),reference[2].argmax(-1)),('original_official_teacher_cache_bridge',index,teacher_error)
            saved=OUT/f'bridge{index}.original_reference.npz';np.savez(saved,encoder=reference[0],decoder=reference[1],logits=reference[2],routes=reference[3],official_teacher_logits=official_teacher)
            result['original_bridges'].append({'case':index,'official_cached_teacher_generate_relative_l2':teacher_error,'official_teacher_top1_exact':True,'reference_sha256':M.digest(saved)})
            actual,output_sha=native(case['source_ids'],case['decoder_ids'],f'bridge{index}')
            assert output_sha==case['native_sha256'],'all_A16_native381_bridge_changed'
            result['native_bridges'].append({'case':index,'exact381_native':True,'sha256':output_sha})
            original_ids,original_arrays=H.cached_greedy(model,source,16,32098)
            official_ids,official_scores=H.official_greedy(model,source,16,32098)
            original_error=B.relative(official_scores,original_arrays[2]);assert original_ids==official_ids and original_error<=1e-6
            ids,generated,sha,row=native_generation(case['source_ids'],f'gen_bridge{index}',16,32098)
            assert sha==control['cases'][index]['natural_sha256'],'native381_greedy_bridge_changed'
            result['original_bridges'][-1]['official_cached_greedy_choices_exact']=True
            result['original_bridges'][-1]['official_cached_greedy_relative_l2']=original_error
            result['native_bridges'][-1]['exact381_greedy_sha256']=sha
            trim();guard()
        stage='NEW_all96_paired_multispan_prediction_and_generation'
        for ordinal,book in enumerate(cohort['items']):
            record={'source_id':book['source_id'],'cases':[]}
            for case in book['cases']:
                label=f'book{ordinal}.case{case["index"]}';source=torch.tensor([case['source_ids']]);decoder=torch.tensor([case['decoder_ids']])
                reference=B.torch_reference(model,source,decoder);guard()
                official_teacher=J.official_teacher_cached(model,source,case['decoder_ids'])
                teacher_error=B.relative(official_teacher,reference[2]);assert teacher_error<=1e-6 and np.array_equal(official_teacher.argmax(-1),reference[2].argmax(-1)),('NEW_original_cached_teacher_generate_bridge',ordinal,case['index'],teacher_error)
                saved=OUT/(label+'.original_reference.npz');np.savez(saved,encoder=reference[0],decoder=reference[1],logits=reference[2],routes=reference[3],official_teacher_logits=official_teacher)
                original_ids,original_gen=H.cached_greedy(model,source,64,32095);guard()
                assert np.array_equal(original_gen[0],reference[0]),'original_encoder_changed_between_pred_gen'
                gen_saved=OUT/(label+'.original_generation.npz');np.savez(gen_saved,encoder=original_gen[0],decoder=original_gen[1],logits=original_gen[2],routes=original_gen[3])
                actual,native_sha=native(case['source_ids'],case['decoder_ids'],label)
                target_ids,target_gen,gen_sha,gen_row=native_generation(case['source_ids'],label+'.generation')
                assert np.array_equal(target_gen[0],actual[0]),'native_encoder_changed_between_pred_gen'
                om=metrics(reference[2],case['target_ids'],case['masked_spans_ids']);nm=metrics(actual[2],case['target_ids'],case['masked_spans_ids'])
                ot=task(original_ids,case['masked_spans_ids']);nt=task(target_ids,case['masked_spans_ids'])
                normalized_edit=edit_distance(ot['prose_ids'],nt['prose_ids'])/max(len(ot['prose_ids']),len(nt['prose_ids']),1)
                record['cases'].append({'index':case['index'],'original':om,'native':nm,
                    'original_top1_agreement':float(np.mean(actual[2].argmax(-1)==reference[2].argmax(-1))),
                    'masked_original_top1_agreement':float(np.mean(actual[2].argmax(-1)[MASK]==reference[2].argmax(-1)[MASK])),
                    'changed_route_choices':int(np.sum(actual[3][:,0]!=reference[3][:,0])),
                    'original_reference_sha256':M.digest(saved),'original_cached_teacher_generate_relative_l2':teacher_error,'original_cached_teacher_top1_exact':True,'native_output_sha256':native_sha,
                    'generation':{'original':ot,'native':nt,'normalized_prose_edit_to_original':normalized_edit,
                                  'original_generation_sha256':M.digest(gen_saved),'native_generation_sha256':gen_sha,
                                  'native_apparatus_row_not_accepted_timing':gen_row}})
                del reference,actual,original_gen,target_gen;gc.collect();trim();guard()
            result['books'].append(record);print(json.dumps({'book':ordinal,'source_id':book['source_id'],'elapsed_seconds':time.monotonic()-start,
                'original_gen_lengths':[c['generation']['original']['generated_tokens'] for c in record['cases']],
                'native_gen_lengths':[c['generation']['native']['generated_tokens'] for c in record['cases']]}),flush=True)
        def book_delta(original,native):
            return [np.mean([native(c)-original(c) for c in b['cases']]) for b in result['books']]
        all_cases=[c for b in result['books'] for c in b['cases']]
        paired={}
        for key in ('mean_nll','mean_span_nll'):
            paired[key]=paired_interval(book_delta(lambda c:c['original'][key],lambda c:c['native'][key]))
        paired['teacher_masked_token_accuracy']=paired_interval(book_delta(lambda c:c['original']['correct_span_tokens']/8,lambda c:c['native']['correct_span_tokens']/8))
        paired['teacher_field_exact_accuracy']=paired_interval(book_delta(lambda c:c['original']['correct_fields']/4,lambda c:c['native']['correct_fields']/4))
        for key in ('known_token_accuracy','known_field_exact_accuracy','known_field_lcs_f1','healthy_complete_nonempty_fields','within_field_repeated_trigram_fraction'):
            paired['generation_'+key]=paired_interval(book_delta(lambda c:c['generation']['original'][key],lambda c:c['generation']['native'][key]))
        paired['normalized_prose_edit_to_original']=paired_interval([np.mean([c['generation']['normalized_prose_edit_to_original'] for c in b['cases']]) for b in result['books']])
        paired['original_top1_agreement']=float(np.mean([c['original_top1_agreement'] for c in all_cases]))
        paired['masked_original_top1_agreement']=float(np.mean([c['masked_original_top1_agreement'] for c in all_cases]))
        result['paired']=paired
        result['original_task_signal']={'generated_field_exact_accuracy':float(np.mean([c['generation']['original']['known_field_exact_accuracy'] for c in all_cases])),
            'healthy_complete_output_fraction':float(np.mean([c['generation']['original']['healthy_complete_nonempty_fields'] for c in all_cases]))}
        result['native_complete_output_fraction']=float(np.mean([c['generation']['native']['healthy_complete_nonempty_fields'] for c in all_cases]))
        result['gates']={'all_original_source_and_official_cache_bridges':True,'all_same381_native_prediction_generation_bridges':True,'all96_prediction_generation_tasks':len(all_cases)==96,
            'all_token_nll_upper95_le0p05':paired['mean_nll']['one_sided_upper95']<=.05,
            'masked_token_nll_upper95_le0p05':paired['mean_span_nll']['one_sided_upper95']<=.05,
            'teacher_masked_token_accuracy_lower95_ge_minus0p02':paired['teacher_masked_token_accuracy']['one_sided_lower95']>=-.02,
            'teacher_field_exact_lower95_ge_minus0p05':paired['teacher_field_exact_accuracy']['one_sided_lower95']>=-.05,
            'original_top1_agreement_ge0p95':paired['original_top1_agreement']>=.95,
            'masked_original_top1_agreement_ge0p95':paired['masked_original_top1_agreement']>=.95,
            'original_generated_field_exact_signal_ge0p10':result['original_task_signal']['generated_field_exact_accuracy']>=.10,
            'original_complete_task_signal_ge0p80':result['original_task_signal']['healthy_complete_output_fraction']>=.80,
            'native_complete_task_fraction_ge0p80':result['native_complete_output_fraction']>=.80,
            'generation_known_token_accuracy_lower95_ge_minus0p02':paired['generation_known_token_accuracy']['one_sided_lower95']>=-.02,
            'generation_field_exact_lower95_ge_minus0p05':paired['generation_known_field_exact_accuracy']['one_sided_lower95']>=-.05,
            'generation_field_lcs_f1_lower95_ge_minus0p02':paired['generation_known_field_lcs_f1']['one_sided_lower95']>=-.02,
            'generation_complete_health_lower95_ge_minus0p05':paired['generation_healthy_complete_nonempty_fields']['one_sided_lower95']>=-.05,
            'generation_repeated_trigram_upper95_le0p05':paired['generation_within_field_repeated_trigram_fraction']['one_sided_upper95']<=.05,
            'generation_prose_edit_to_original_upper95_le0p10':paired['normalized_prose_edit_to_original']['one_sided_upper95']<=.10}
        assert payload_stat==(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert all(command['actual_affinity']==affinity for command in result['commands'])
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='independent_original128_NEW_whole_quality_pass_performance_unqualified' if all(result['gates'].values()) else 'fixed_original128_multispan_quality_not_qualified_preserve_all_sources_before_diagnostics'
        result['scope']='96 NEW project source-disjoint four-two-token-mask cases/source29/target14/cap64 from24 PG19 books. Independently pretrained original128 unmodified F32 PRIMARY CPU1 with official cached teacher/generate bridges (raw logits captured before teacher-ID forcing, all96 teacher cache bridges); complete380 target/SAME374 binary/CPU6 exact actual masks ACTIVE wait. Every original coefficient freshly verified379; complete target numerical bridges381. Teacher AND natural known-answer/prose/health gates unchanged363, no inherited363 source quality. Quality-only experiment after preserved383 stability FAIL: no accepted-rate promotion, performance unqualified. Apparatus timing not accepted rate; greater useful n/LUT/physical DRAM/cross-family/general contexts remain open.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'gates':result['gates'],'paired':result['paired'],'resource':result['resource'],'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
