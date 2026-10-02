#!/usr/bin/env python3
"""Prospectively bound actual native-primary held-out prediction; no timing claim."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import gc
import json
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
import psutil
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
import meth280_complete_i16_fresh_prediction as Q
import meth294_native_scorer_bridge as B

P,L,R,F=Q.P,Q.L,Q.R,Q.F
DOC,ART,CORE,CORE_SHA=B.DOC,B.ART,B.CORE,B.CORE_SHA
MANIFEST=DOC/'meth292_native_primary_manifest.json'
MANIFEST_SHA='99fdb09fe13e8abf8c401718911d08137851aa90c043415dd92eeb7d619f3847'
ANSWER=DOC/'meth293_native_source_answerability_result.json'
ANSWER_SHA='a6b34f60541efd02d21ae3e458c17c0b52050c192b22c6aad4e526ca3fe29109'
ANNOTATIONS=DOC/'meth293_native_source_answerability_annotations.json'
ANNOTATIONS_SHA='9ceb199045da48152eae40784139fd78007550578f2e1e3d070298f317dad21e'
POLICY=DOC/'METH_292_NATIVE_PRIMARY_EVALUATION_POLICY_20261002.md'
POLICY_SHA='0a3aab8f4733523b3af0b9d6bf919de406b1e94da249d5a3125176a246c57753'
BRIDGE=DOC/'meth294_native_scorer_bridge_result.json'
BRIDGE_SHA='fd94fe0fa4568090e22f2e07ee84755bc556ccb3750654c933765dcc507a7da3'
ARMS=('bf16_donor','bf16_e1280','native_i16_private128_e1280')
GPU_STORED='gpu_i16_private128_e1280_descriptive'
V,EOS=151936,151645
CPU_SECONDS=75*60

def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')

def windows(ids):
    prefix=[EOS]+ids
    for first in range(0,len(ids),512):
        end=min(first+512,len(ids));lo=max(0,first-512)
        yield {'ids':prefix[lo:end],'targets':ids[first:end],'offset':lo,
               'first':first-lo,'document_first':first}

def window_fixture():
    ids=list(range(1200));w=list(windows(ids))
    assert [x['offset'] for x in w]==[0,0,512]
    assert [x['first'] for x in w]==[0,512,512]
    assert [len(x['ids']) for x in w]==[512,1024,688]
    assert w[0]['ids']==[EOS]+ids[:511]
    assert w[2]['ids']==ids[511:1199] and w[2]['targets']==ids[1024:1200]
    assert sum(len(x['targets']) for x in w)==1200
    assert P.M17.STRIDE==P.M17.CONTEXT==512

def bind():
    window_fixture();B.proof()
    for path,sha in ((MANIFEST,MANIFEST_SHA),(ANSWER,ANSWER_SHA),(ANNOTATIONS,ANNOTATIONS_SHA),
                     (POLICY,POLICY_SHA),(BRIDGE,BRIDGE_SHA),(CORE,CORE_SHA),
                     (Q.EXPORT,Q.EXPORT_SHA),(P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),
                     (P.M122.TRAINING,P.M57.TRAINING_SHA)):
        assert P.digest(path)==sha,str(path)
    bridge=json.loads(BRIDGE.read_text(encoding='utf-8'));assert all(bridge['gates'].values())
    for name,sha in bridge['sha256'].items():
        if name.startswith('benchmarks/native_expert_scaling/'):
            assert P.digest(P.ROOT/name)==sha,name
    original=(P.ROOT/'benchmarks/native_expert_scaling/meth294_native_prediction_cpu.c').read_text(encoding='utf-8')
    expected=original.replace('// Full-head native quality entry; original285 operators with absolute-window RoPE.',
        '// Native-primary complete-bank quality; same frozen294 operators/scorer, one arm.')
    expected=expected.replace('uint32_t dims[2]={2,count}','uint32_t dims[2]={1,count}')
    expected=expected.replace('for(int arm=0;arm<2;arm++)','for(int arm=1;arm<2;arm++)')
    expected=expected.replace('        }\n    }\n    if(fclose(out))',
        '        }\n        if(fflush(out))die("output flush");\n        printf("{\\"case_complete\\":%u,\\"cases\\":%u,\\"scored_rows\\":%llu,\\"prefill_rows\\":%llu}\\n",c+1,count,(unsigned long long)scored,(unsigned long long)prefill);fflush(stdout);\n    }\n    if(fclose(out))')
    expected=expected.replace('\\"arms\\":2','\\"arms\\":1')
    assert expected==(P.ROOT/'benchmarks/native_expert_scaling/meth295_native_primary_cpu.c').read_text(encoding='utf-8')
    export=json.loads(Q.EXPORT.read_text(encoding='utf-8'));assert all(export['gates'].values())
    assert P.digest(Path(R.__file__))==export['script_sha256']
    for name,sha in export['helper_sha256'].items():assert P.digest(Path(name))==sha,name
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));answer=json.loads(ANSWER.read_text(encoding='utf-8'))
    assert answer['answerable_count']==24 and all(answer['gates'].values())
    assert answer['model_outputs_consulted'] is False and answer['annotations_sha256']==ANNOTATIONS_SHA
    assert answer['manifest_sha256']==MANIFEST_SHA and manifest['artifact_sha256']==CORE_SHA
    assert manifest['selected_counts']==dict.fromkeys(P.CATEGORIES,8)
    items=manifest['items'];assert len(items)==24
    for item,row in zip(items,answer['rows']):
        assert item['source_id']==row['source_id'] and row['anchor'] in item['excerpt']
        assert P.M17.sha(item['text'].encode())==item['text_sha256']
        assert item['bytes']==len(item['text'].encode())
        for key in ('document_ids','prompt_ids'):
            assert P.M17.sha(np.asarray(item[key],dtype='<i4').tobytes())==item[key+'_sha256']
            assert all(0<=i<V for i in item[key])
    parent=json.loads(P.M122.TRAINING.read_text(encoding='utf-8'))['checkpoints']['512']
    assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
    source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
    assert P.digest(source)==P.M57.MODEL_SHA
    return items,parent

def cases_for(items):
    cases=[]
    for i,item in enumerate(items):
        for w in windows(item['document_ids']):cases.append({'kind':'document','item':i,**w})
        ids=item['prompt_ids'];cases.append({'kind':'prompt','item':i,'ids':ids,'targets':ids[1:]+[-1],
            'offset':0,'first':0,'document_first':0})
    return cases

def write_bundle(path,cases):
    with path.open('wb') as file:
        file.write(struct.pack('<8sI',b'M294IN01',len(cases)))
        for c in cases:
            n=len(c['ids']);assert len(c['targets'])==n-c['first']
            file.write(struct.pack('<3I',n,c['offset'],c['first']))
            file.write(np.asarray(c['ids'],dtype='<i4').tobytes());file.write(np.asarray(c['targets'],dtype='<i4').tobytes())
    # Read actual bytes back, including every nonzero-offset target/position binding.
    with path.open('rb') as file:
        assert struct.unpack('<8sI',file.read(12))==(b'M294IN01',len(cases))
        for c in cases:
            n=len(c['ids']);assert struct.unpack('<3I',file.read(12))==(n,c['offset'],c['first'])
            assert np.array_equal(np.frombuffer(file.read(n*4),dtype='<i4'),c['ids'])
            assert np.array_equal(np.frombuffer(file.read((n-c['first'])*4),dtype='<i4'),c['targets'])
        assert not file.read(1)

def score_gpu(model,wrappers,items,enabled,device,start,donor_top):
    P.M44.set_experts(wrappers,enabled);docs=[];prompts=[]
    with torch.inference_mode():
        for item in items:
            losses=[];top=[];nats=0.0
            for c in windows(item['document_ids']):
                ids=torch.as_tensor(c['ids'],device=device)[None]
                positions=torch.arange(c['offset'],c['offset']+len(c['ids']),device=device)[None]
                logits=model(ids,position_ids=positions,use_cache=False).logits[0,c['first']:].float()
                lp=torch.nn.functional.log_softmax(logits,dim=-1)
                target=torch.as_tensor(c['targets'],device=device)
                nll=-lp.gather(1,target[:,None]);nats+=float(nll.sum())
                losses.extend(nll[:,0].cpu().tolist());top.extend(logits.argmax(-1).cpu().tolist());P.budget(start,device)
            # Canonical independently called helper must give the identical scalar.
            canonical=P.M17.score_doc(model,item['document_ids'],wrappers,enabled,device,start)
            assert canonical==nats,(item['source_id'],canonical,nats)
            docs.append({'source_id':item['source_id'],'category':item['category'],'bytes':item['bytes'],
                         'nats':nats,'token_nll':losses,'token_top1':top,'canonical_m17_nats_exact':True})
        for item in items:
            top=model(torch.as_tensor(item['prompt_ids'],device=device)[None],use_cache=False).logits.argmax(-1)[0].cpu().numpy()
            if item['source_id'] not in donor_top:donor_top[item['source_id']]=top
            prompts.append({'source_id':item['source_id'],'category':item['category'],'positions':len(top),
                'matching':int((top==donor_top[item['source_id']]).sum()),'token_top1':top.tolist()});P.budget(start,device)
    return {'document_rows':docs,'prompt_rows':prompts}

def run_cpu(exe,bundle,native,log,err):
    start=time.monotonic();peak=0
    with log.open('w',encoding='utf-8') as stdout,err.open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen([str(exe),str(CORE),str(bundle),str(native)],stdout=stdout,stderr=stderr)
        try:
            process=psutil.Process(child.pid)
            while child.poll() is None:
                if time.monotonic()-start>CPU_SECONDS:raise TimeoutError('native CPU 75-minute stop')
                try:peak=max(peak,process.memory_info().rss)
                except psutil.NoSuchProcess:pass
                if peak>20*(1<<30):raise MemoryError('native CPU RSS 20GiB stop')
                time.sleep(5)
            if child.returncode:raise RuntimeError('native CPU exit '+str(child.returncode))
        except BaseException:
            if child.poll() is None:child.kill();child.wait()
            raise
    loader=json.loads(err.read_text(encoding='utf-8'))
    assert loader['fields_bound']==725 and loader['fallback'] is False and loader['archive_sha256']==CORE_SHA
    reports=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    return {'seconds':time.monotonic()-start,'peak_rss_bytes':peak,'loader':loader,'reports':reports}

def parse_native(path,cases,items,donor_top):
    docs=[{'source_id':i['source_id'],'category':i['category'],'bytes':i['bytes'],'nats':0.0,'token_nll':[],'token_top1':[]} for i in items]
    prompts=[None]*24;oracles=[]
    with path.open('rb') as file:
        assert struct.unpack('<8s2I',file.read(16))==(b'M294OUT1',1,len(cases))
        for index,c in enumerate(cases):
            n=len(c['ids']);assert struct.unpack('<5I',file.read(20))==(1,index,n,c['offset'],c['first'])
            logits=np.frombuffer(file.read(V*4),dtype='<f4').copy().astype(np.float64)
            assert len(logits)==V and np.isfinite(logits).all()
            losses=[];top=[]
            for j,target in enumerate(c['targets']):
                pos,choice,stored_target,loss=struct.unpack('<iiid',file.read(20))
                assert pos==c['offset']+c['first']+j and stored_target==target
                assert 0<=choice<V and np.isfinite(loss) and loss>=0
                if j==0:
                    maximum=logits.max();oracle=0.0 if target<0 else float(np.log(np.exp(logits-maximum).sum())+maximum-logits[target])
                    assert choice==int(logits.argmax()) and abs(loss-oracle)<=1e-9
                    oracles.append({'case':index,'offset':c['offset'],'top1_exact':True,'nll_abs_error':abs(loss-oracle)})
                if target<0:assert loss==0.0
                losses.append(loss);top.append(choice)
            if c['kind']=='document':
                row=docs[c['item']];assert len(row['token_nll'])==c['document_first']
                row['nats']+=sum(losses);row['token_nll'].extend(losses);row['token_top1'].extend(top)
            else:
                item=items[c['item']];assert prompts[c['item']] is None
                prompts[c['item']]={'source_id':item['source_id'],'category':item['category'],'positions':n,
                    'matching':int((np.asarray(top)==donor_top[item['source_id']]).sum()),'token_top1':top,'token_nll':losses}
        assert not file.read(1)
    for item,row in zip(items,docs):assert len(row['token_nll'])==len(item['document_ids'])
    assert all(row is not None for row in prompts)
    return {'document_rows':docs,'prompt_rows':prompts},oracles

def bootstrap(arms):
    rng=np.random.default_rng(295295);draws=rng.integers(0,24,size=(10000,24));out={}
    candidate=arms[ARMS[2]]['document_rows'];bytes_=np.asarray([r['bytes'] for r in candidate])
    for control in ARMS[:2]:
        delta=np.asarray([a['nats']-b['nats'] for a,b in zip(candidate,arms[control]['document_rows'])])
        values=delta[draws].sum(1)/(np.log(2)*bytes_[draws].sum(1))
        out[control]={'bpb_delta_p05':float(np.quantile(values,.05)),'bpb_delta_p95':float(np.quantile(values,.95)),
            'seed':295295,'draws':10000,'unit':'source','decision_gate':False}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    bundle=ART/'meth295_native_primary_ids.bin';native=ART/'meth295_native_primary.bin'
    log=ART/'meth295_native_primary_stdout.log';err=ART/'meth295_native_primary_stderr.log';exe=ART/'meth295_native_primary_cpu.exe'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json'),bundle,native,log,err))
    start=time.monotonic();stage='bindings';arms={};cpu=None
    def partial():dump(args.out.with_suffix('.partial.json'),{'stage':stage,'arms':arms,'seconds':time.monotonic()-start})
    try:
        items,parent=bind();cases=cases_for(items);write_bundle(bundle,cases)
        command=['clang','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11','-DSILICON_COMPLETE_I16_NATIVE_PRIMARY',
            str(P.ROOT/'benchmarks/phase60/engine.c'),'-o',str(exe),'-lm','-lpsapi','-lbcrypt']
        subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        print(json.dumps({'stage':'bound_and_compiled','cases':len(cases),'scored_rows':sum(len(c['targets']) for c in cases),
            'prefill_rows':sum(c['first'] for c in cases),'nonzero_offset_cases':sum(c['offset']>0 for c in cases)}),flush=True)
        device=R.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        stage='original_controls'
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval();model.config.use_cache=False
        assert model.config.eos_token_id==EOS
        L.D.H.load_centered(model,device,parent['path']);wrappers=[l.mlp for l in model.model.layers];donor_top={}
        for arm,enabled in ((ARMS[0],False),(ARMS[1],True)):
            stage=arm;arms[arm]=score_gpu(model,wrappers,items,enabled,device,start,donor_top);partial()
            print(json.dumps({'stage':stage,'runtime':P.budget(start,device)}),flush=True)
        del model,wrappers;gc.collect();torch.cuda.empty_cache()
        stage=GPU_STORED;model,wrappers,proposal,load_record=R.load_stored(CORE,device,CORE_SHA)
        arms[GPU_STORED]=score_gpu(model,wrappers,items,True,device,start,donor_top);partial()
        gpu_runtime=P.budget(start,device);print(json.dumps({'stage':stage,'runtime':gpu_runtime}),flush=True)
        del model,wrappers,proposal;gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated(device)==0
        stage='native_CPU_full_head';partial()
        print(json.dumps({'stage':stage,'log':str(log),'hard_seconds':CPU_SECONDS,'performance_measurement':False}),flush=True)
        cpu=run_cpu(exe,bundle,native,log,err)
        assert len(cpu['reports'])==len(cases)+1
        assert [r['case_complete'] for r in cpu['reports'][:-1]]==list(range(1,len(cases)+1))
        report=cpu['reports'][-1]
        assert report['arms']==1 and report['cases']==len(cases) and report['timing_qualification'] is False
        assert report['scored_rows']==sum(len(c['targets']) for c in cases) and report['prefill_rows']==sum(c['first'] for c in cases)
        stage='parse_native';arms[ARMS[2]],oracles=parse_native(native,cases,items,donor_top)
        F.ARMS=ARMS;summary=F.summarize({k:arms[k] for k in ARMS})
        F.ARMS=(*ARMS[:2],GPU_STORED)
        descriptive=F.summarize({k:arms[k] for k in F.ARMS});F.ARMS=ARMS
        gates={'fixed_manifest_source_annotations_and_original_artifact':True,'same_native_operator_body_and_all725_no_fallback':True,
            'all_windows_absolute_positions_targets_and_scored_tokens_verified':True,'all_case_first_row_full_head_score_oracles':True,
            'all_GPU_document_scalars_exact_independent_canonical_M17':True,
            'pooled_bpb_vs_both':all(summary['pooled'][k]<=.01 for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'category_bpb_vs_both':all(summary[c][k]<=.02 for c in P.CATEGORIES for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'pooled_top1':summary['pooled']['candidate_minus_bf16_e1280_top1']>=-.01,
            'category_top1':all(summary[c]['candidate_minus_bf16_e1280_top1']>=-.02 for c in P.CATEGORIES)}
        paths=[Path(__file__),P.ROOT/'benchmarks/native_expert_scaling/meth295_native_primary_cpu.c',
            P.ROOT/'benchmarks/phase60/engine.c',P.ROOT/'benchmarks/native_expert_scaling/meth285_model_operator.h',
            P.ROOT/'benchmarks/native_expert_scaling/meth284_source_operator.h',
            P.ROOT/'benchmarks/native_expert_scaling/meth284_archive_catalog.h',
            P.ROOT/'benchmarks/native_expert_scaling/meth294_window_forward.h',exe,bundle,native,log,err]
        result={'experiment':'METH-295-actual-native-primary-new-source-prediction','artifact_sha256':CORE_SHA,
            'manifest_sha256':MANIFEST_SHA,'answerability_sha256':ANSWER_SHA,'annotations_sha256':ANNOTATIONS_SHA,
            'policy_sha256':POLICY_SHA,'bridge_sha256':BRIDGE_SHA,'source_sha256':P.M57.MODEL_SHA,
            'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,
            'arms':arms,'summary':summary,'GPU_stored_descriptive_summary':descriptive,'bootstrap':bootstrap(arms),
            'gates':gates,'full_head_oracles':oracles,'cpu_runtime':cpu,'gpu_runtime':gpu_runtime,'gpu_load_record':load_record,
            'cases':[{'case':i,'kind':c['kind'],'source_id':items[c['item']]['source_id'],'n':len(c['ids']),
                'offset':c['offset'],'first':c['first'],'document_first':c['document_first']} for i,c in enumerate(cases)],
            'compile_command':command,'sha256':{str(p.relative_to(P.ROOT)):P.digest(p) for p in paths},
            'seconds':time.monotonic()-start,'diagnostic_only':True,'native_promotion_qualified':False,
            'decision':'native_primary_prediction_pass_freeze_generation' if all(gates.values()) else 'native_primary_prediction_fail_close_fixed_native_profile',
            'scope':'All24 new292 sources, actual native original285 complete-bank profile, full head/M17 windows/prompt top1; unchanged donor/E1280 quality margins. GPU same-archive descriptive. Not generation/semantics/PIQA/CPU K64/accepted rate/useful new n/RAM/DRAM/family proof. Prior5% numerical failures retained.'}
        dump(args.out,result);print(json.dumps({k:result[k] for k in ('decision','summary','gates','seconds')}),flush=True)
    except BaseException as error:
        partial();dump(args.out.with_suffix('.failure.json'),{'stage':stage,'error':type(error).__name__+': '+str(error),
            'seconds':time.monotonic()-start,'cpu_runtime':cpu});raise

if __name__=='__main__':main()
