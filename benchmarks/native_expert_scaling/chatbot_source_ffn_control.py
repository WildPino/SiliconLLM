"""Full source arithmetic control from the already converted FFN sector."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write,normalized
sys.path.insert(0,str(SITE))


def bind(a):
    preflight_path=DOC/'chatbot_source_ffn_preflight_result_20261009.json';pre=json.loads(preflight_path.read_bytes())
    receipt_path=preflight_path.with_suffix('.terminal.json');receipt=json.loads(receipt_path.read_bytes())
    assert pre['decision']=='SOURCE_FFN_COST_ARITHMETIC_PASS' and receipt['exit_code']==0 and receipt['resource_gates']
    assert receipt['result_sha256']==sha(preflight_path)
    original_path=DOC/'chatbot_source_ffn_preflight_binding_20261009.json';original=json.loads(original_path.read_bytes())
    assert sha(original_path)==pre['binding_sha256']
    for item in original['inputs']:
        p=Path(item['path'])
        if p.resolve()!=(B/'chatbot_falcon_usability_launch.py').resolve():assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    directory=Path(pre['observations'][0]['logits']['path']).parent
    manifest_path=directory/'ffn.manifest.json';manifest=json.loads(manifest_path.read_bytes());payload=Path(manifest['file']['path'])
    assert sha(payload)==manifest['file']['sha256']=='9aa6f319363294aac6ee611def4384e9c63354929c43c6219d8144f698a2a33b'
    screen_path=DOC/'chatbot_hybrid_usability_result_repair1_20261008.json';screen=json.loads(screen_path.read_bytes())
    screen_receipt=screen_path.with_suffix('.terminal.json');sr=json.loads(screen_receipt.read_bytes())
    assert screen['correct']==14 and screen['total']==16 and screen['decision']=='TEACHER_SCREEN_PASS'
    assert sr['exit_code']==0 and sr['result_sha256']==sha(screen_path)
    corpus=Path(original['corpus']);records=json.loads(corpus.read_bytes())['records'];assert len(records)==48
    files=[Path(v['path']) for v in original['inputs']]
    files += [Path(__file__),preflight_path,receipt_path,original_path,manifest_path,payload,
              screen_path,screen_receipt,B/'chatbot_falcon_usability_cases_v1.json',DOC/'CHATBOT_SOURCE_FFN_CONTROL_PROTOCOL_20261009.md']
    files += [Path(r['logits']['path']) for r in records]
    files += [directory/(r['id']+'.json') for r in pre['observations']]+[Path(r['logits']['path']) for r in pre['observations']]
    files += [SITE/'transformers'/p for p in ('generation/utils.py','generation/configuration_utils.py','tokenization_utils_base.py','tokenization_utils_tokenizers.py')]
    files += [SITE/'tokenizers/tokenizers.pyd']
    files=list(dict.fromkeys(p.resolve() for p in files))
    write(a.out,dict(schema='SOURCE_FFN_CONTROL_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
        source=original['source'],corpus=str(corpus),manifest=str(manifest_path),payload=str(payload),preflight=str(preflight_path),
        screen_cases=str(B/'chatbot_falcon_usability_cases_v1.json'),source_screen_correct=14,
        criteria=dict(DEV_KL=1.,DEV_disagreement=.2,domain_DEV_KL=2.,domain_DEV_disagreement=.35,screen_correct_min=13),
        limits=dict(seconds=2100,reserve_seconds=60,OS_bytes=8<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=2<<30),
        runtime_binding_scope='Original source/complete adopted48/actual packed FFN/preflight reuse/known16 screen/new4 actual-history probes/code/Python/selected runtime/foreign hashes;not full DLL-tree.',
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());assert b['schema']=='SOURCE_FFN_CONTROL_BINDING_V1'
    a.directory.mkdir(exist_ok=False);phase='startup';completed=[];forwards=0;generations=0;torch=None
    try:
        assert sys.version_info[:3]==(3,12,10) and Path(sys.executable).resolve()==Path(b['python']).resolve()
        assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers import AutoTokenizer,FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        from chatbot_falcon_ssd_tiles import install
        from chatbot_source_ternary_ffn import EngineFFN,packed_projection
        assert (torch.__version__,transformers.__version__,np.__version__)==('2.6.0+cu124','5.13.1','2.4.6')
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def guard():
            lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds'],'time reserve'
            assert proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True),'worker child'
            assert sum(v.stat().st_size for v in a.directory.iterdir() if v.is_file())<=lim['output_bytes'],'output cap'
        def event(stage,**fields):print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True);guard()
        write(a.directory/'source_method.json',install(code))
        phase='load_source'
        tokenizer=AutoTokenizer.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False)
        assert tokenizer.bos_token_id==17 and len(tokenizer)==65537
        model=FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        assert sum(p.numel() for p in model.parameters())==1554863488 and not code.is_fast_path_available
        assert model.generation_config.eos_token_id==[11,228] and model.config.mamba_chunk_size==128
        unchanged={n:id(p) for n,p in model.named_parameters() if '.feed_forward.' not in n}
        manifest=json.loads(Path(b['manifest']).read_bytes());fields=manifest['fields'];assert len(fields)==144
        cursor=0
        for field in fields:assert field['offset']==cursor;cursor+=field['bytes']
        assert cursor==340819968
        phase='load_packed_FFN';t0=time.monotonic()
        for site,layer in enumerate(model.model.layers):
            original=layer.feed_forward;replacement=EngineFFN.__new__(EngineFFN);torch.nn.Module.__init__(replacement)
            replacement.site=site;replacement.observer=None;replacement.weight_metrics=[]
            replacement.gate_multiplier=original.gate_multiplier;replacement.down_multiplier=original.down_multiplier
            for name in ('gate','up','down'):
                field=next(v for v in fields if v['site']==site and v['projection']==name and v['field']=='pair_codes')
                raw=np.fromfile(b['payload'],dtype=np.int8,count=field['bytes'],offset=field['offset']).reshape(field['shape'])
                pairs=torch.from_numpy(raw).to('cuda');assert pairs.min().item()>=0 and pairs.max().item()<=8
                rows=field['shape'][1];columns=field['shape'][0]*2
                assert (rows,columns)==tuple(getattr(original,name+'_proj').weight.shape)
                codes=torch.empty((rows,columns),dtype=torch.int8,device='cuda')
                codes[:,0::2]=(pairs//3-1).T;codes[:,1::2]=(pairs%3-1).T
                sf=next(v for v in fields if v['site']==site and v['projection']==name and v['field']=='row_scale')
                scales=torch.from_numpy(np.fromfile(b['payload'],dtype='<f4',count=rows,offset=sf['offset'])).to('cuda')
                assert list(scales.shape)==sf['shape'] and torch.isfinite(scales).all().item() and (scales>=1e-8).all().item()
                replacement.register_buffer(name+'_codes',codes);replacement.register_buffer(name+'_scale',scales)
                encoded,_=packed_projection(replacement,name);assert torch.equal(encoded,pairs)
                del raw,pairs,codes,scales,encoded
            layer.feed_forward=replacement;del original,replacement
        assert {n:id(p) for n,p in model.named_parameters()}==unchanged and len(unchanged)==339
        assert sum(p.numel() for p in model.parameters())==875386240
        event('packed_ready',loader_seconds=time.monotonic()-t0,recalibration_calls=0)
        pre=json.loads(Path(b['preflight']).read_bytes());reuse={r['id']:r for r in pre['observations']}
        records=json.loads(Path(b['corpus']).read_bytes())['records'];observations=[]
        def lp64(x):x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
        for rec in records:
            phase=rec['id'];guard()
            if rec['id'] in reuse:
                row=dict(reuse[rec['id']],origin='Completed preflight reused;zero new forward')
            else:
                t0=time.monotonic();prompt=rec['input_ids'];outputs=rec['output_ids'];past=None;actual=[]
                with torch.inference_mode():
                    for index in range(len(outputs)):
                        ids=prompt if index==0 else [outputs[index-1]]
                        result=model(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(prompt)+index),dtype=torch.long,device='cuda'),
                                     past_key_values=past,use_cache=True,logits_to_keep=1)
                        past=result.past_key_values;logits=result.logits[0,-1];forwards+=1
                        assert logits.dtype==torch.bfloat16 and torch.isfinite(logits).all().item()
                        actual.append(logits.view(torch.uint16).cpu().numpy().copy());del result,logits;guard()
                values=np.stack(actual);path=a.directory/(rec['id']+'.logits.bf16')
                with path.open('xb') as f:f.write(values.astype('<u2',copy=False).tobytes());f.flush();os.fsync(f.fileno())
                source_bits=np.fromfile(rec['logits']['path'],dtype='<u2').reshape(rec['logits']['shape'])
                source=(source_bits.astype('<u4')<<16).view('<f4');changed=(values.astype('<u4')<<16).view('<f4')
                labels=[]
                for offset in range(0,len(outputs),16):
                    p=lp64(source[offset:offset+16].astype(np.float64));q=lp64(changed[offset:offset+16].astype(np.float64))
                    labels.extend((np.exp(p)*(p-q)).sum(-1).tolist())
                assert all(np.isfinite(v) and v>=-1e-10 for v in labels)
                winners=changed.argmax(-1).tolist()
                row=dict(id=rec['id'],labels=len(outputs),case_KL_F64=sum(labels)/len(labels),label_KL_F64=labels,predicted_ids=winners,
                    disagreements=sum(x!=y for x,y in zip(winners,outputs,strict=True)),stop=rec['stop'],origin='New changed-FFN forced trajectory',
                    seconds=time.monotonic()-t0,logits=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)))
                del past,actual,values,source_bits,source,changed
            row.update(domain=rec['domain'],split=rec['split']);write(a.directory/(rec['id']+'.json'),row)
            observations.append(row);completed.append(rec['id'])
            event('forced_case',id=rec['id'],origin=row['origin'],KL=row['case_KL_F64'],disagreements=row['disagreements'],labels=row['labels'])
        assert len(observations)==48 and forwards==8534
        def aggregate(rows):
            n=sum(r['labels'] for r in rows)
            return dict(cases=len(rows),labels=n,case_KL=sum(r['case_KL_F64'] for r in rows)/len(rows),
                label_KL=sum(sum(r['label_KL_F64']) for r in rows)/n,disagreements=sum(r['disagreements'] for r in rows),
                label_disagreement=sum(r['disagreements'] for r in rows)/n)
        summary={s:aggregate([r for r in observations if r['split']==s]) for s in ('FIT','DEV')}
        summary['DEV_domains']={d:aggregate([r for r in observations if r['domain']==d and r['split']=='DEV']) for d in sorted({r['domain'] for r in observations})}
        write(a.directory/'forced_summary.json',summary);event('forced_complete',summary=summary)
        phase='generation';spec=json.loads(Path(b['screen_cases']).read_bytes());assert spec['max_new_tokens']==64 and len(spec['cases'])==16
        generated_rows=[];followups=[]
        def generate(case,messages,identifier):
            nonlocal generations
            t0=time.monotonic();text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            expected=tokenizer.bos_token+''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
            context_cap=256 if identifier.startswith('screen.') else 384
            assert text==expected and ids==tokenizer.encode(text,add_special_tokens=False) and ids[0]==17 and len(ids)+64<=context_cap
            with torch.inference_mode():
                output=model.generate(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long),
                    max_new_tokens=64,do_sample=False,use_cache=True,return_dict_in_generate=True,output_scores=True)
                generations+=1;new_ids=output.sequences[0,len(ids):].tolist();scores=torch.stack(output.scores).squeeze(1).float().cpu().numpy()
            assert scores.shape==(len(new_ids),65537) and np.isfinite(scores).all() and scores.argmax(-1).tolist()==new_ids
            path=a.directory/(identifier+'.scores.f32')
            with path.open('xb') as f:f.write(scores.astype('<f4',copy=False).tobytes());f.flush();os.fsync(f.fileno())
            decoded=tokenizer.decode(new_ids,skip_special_tokens=True)
            row=dict(id=identifier,category=case['category'],expected=case['expected'],messages=messages,serialized=text,input_ids=ids,output_ids=new_ids,
                output_text=decoded,normalized=normalized(decoded),correct=normalized(decoded)==case['expected'],blank=not decoded.strip(),
                special_leak=any(v in set(tokenizer.all_special_ids)-{11,228} for v in new_ids),
                stop_policy=not any(v in (11,228) for v in new_ids[:-1]) and (new_ids[-1] in (11,228) or len(new_ids)==64),
                eos_stop=new_ids[-1] in (11,228),scores=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)),seconds=time.monotonic()-t0)
            write(a.directory/(identifier+'.json'),row);completed.append(identifier)
            event('generation',id=identifier,correct=row['correct'],answer=decoded,labels=len(new_ids));return row
        for case in spec['cases']:
            messages=case.get('history',[])+[dict(role='user',content=case['prompt'])]
            row=generate(case,messages,'screen.'+case['id']);generated_rows.append(row)
            if case['category']=='history':
                follow=messages+[dict(role='assistant',content=row['output_text']),
                    dict(role='user',content='Answer the previous question again. Return only the requested value.')]
                followups.append(generate(case,follow,'own_history.'+case['id']))
        assert generations==20 and len(generated_rows)==16 and len(followups)==4
        c=b['criteria'];gates=dict(DEV_KL=summary['DEV']['case_KL']<=c['DEV_KL'],DEV_disagreement=summary['DEV']['label_disagreement']<=c['DEV_disagreement'],
            every_domain_KL=all(v['case_KL']<=c['domain_DEV_KL'] for v in summary['DEV_domains'].values()),
            every_domain_disagreement=all(v['label_disagreement']<=c['domain_DEV_disagreement'] for v in summary['DEV_domains'].values()),
            screen_correct=sum(r['correct'] for r in generated_rows)>=c['screen_correct_min'],
            every_screen_category=all(sum(r['correct'] for r in generated_rows if r['category']==category)>=2 for category in ('arithmetic','extraction','instruction','history')),
            screen_blank=sum(r['blank'] for r in generated_rows)<=1,no_special_leak=not any(r['special_leak'] for r in generated_rows+followups),
            generation_stop=all(r['stop_policy'] for r in generated_rows+followups),own_history_correct=all(r['correct'] for r in followups))
        assert {n:id(p) for n,p in model.named_parameters()}==unchanged
        phase='result';guard();decision='SOURCE_FFN_CONTROL_PASS' if all(gates.values()) else 'SOURCE_FFN_CONTROL_FAIL'
        write(a.out,dict(schema='SOURCE_FFN_CONTROL_RESULT_V1',decision=decision,freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),cases=48,reused_cases=2,new_cases=46,summary=summary,observations=observations,
            generations=20,screen_correct=sum(r['correct'] for r in generated_rows),source_screen_correct=14,generated_rows=generated_rows,followups=followups,gates=gates,
            new_forced_forwards=forwards,original_source_generations=0,weight_calibrations=0,optimizer_updates=0,native_runs=0,reserved_queries=0,
            packed_FFN_bytes=340819968,unchanged_parameter_object_identity=True,GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            quality_admission=False,native_admission=False,scope='Full source-width FFN arithmetic control:original48 forced prefixes,known16 screen andfour actual generated-history fact followups. Offline full source organs,not compact final engine/fresh broad quality/accepted50/useful-n/DRAM.'))
        event('complete',decision=decision,gates=gates)
    except BaseException as error:
        failure=dict(fault=repr(error),phase=phase,completed=completed,new_forced_forwards=forwards,generations=generations,elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
