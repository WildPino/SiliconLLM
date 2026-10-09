"""Source FFN F32 positive control; no ternary weights or AQ63."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write,normalized
sys.path.insert(0,str(SITE))


def bind(a):
    prepath=DOC/'chatbot_source_ffn_preflight_binding_20261009.json';pre=json.loads(prepath.read_bytes())
    pre_result_path=DOC/'chatbot_source_ffn_preflight_result_20261009.json';pr=json.loads(pre_result_path.read_bytes())
    assert sha(prepath)==pr['binding_sha256']
    for item in pre['inputs']:
        path=Path(item['path'])
        if path.resolve()!=(B/'chatbot_falcon_usability_launch.py').resolve():assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    result_path=DOC/'chatbot_source_ffn_control_result_20261009.json';result=json.loads(result_path.read_bytes())
    terminal_path=result_path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['exit_code']==0 and terminal['resource_gates'] and terminal['result_sha256']==sha(result_path)
    assert result['cases']==48 and result['generations']==20 and result['decision']=='SOURCE_FFN_CONTROL_FAIL'
    corpus=Path(pre['corpus']);records=json.loads(corpus.read_bytes())['records']
    selected=[r for r in records if r['id'] in pre['selected_ids']];assert len(selected)==2 and sum(len(r['output_ids']) for r in selected)==274
    full_binding_path=DOC/'chatbot_source_ffn_control_binding_20261009.json'
    assert sha(full_binding_path)==result['binding_sha256']
    files=[Path(v['path']) for v in pre['inputs']]
    files += [Path(__file__),prepath,pre_result_path,result_path,terminal_path,full_binding_path,
              DOC/'CHATBOT_SOURCE_FFN_CAST_CONTROL_PROTOCOL_20261009.md',B/'chatbot_falcon_usability_cases_v1.json']
    full_binding=json.loads(full_binding_path.read_bytes())
    files += [Path(v['path']) for v in full_binding['inputs'] if 'generation/' in v['path'].replace('\\','/') or 'tokeniz' in v['path']]
    files=list(dict.fromkeys(p.resolve() for p in files))
    write(a.out,dict(schema='SOURCE_FFN_CAST_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
        source=pre['source'],corpus=str(corpus),selected_ids=pre['selected_ids'],screen_cases=str(B/'chatbot_falcon_usability_cases_v1.json'),
        criteria=dict(case_KL_max=.01,label_disagreement_max=.05,screen_correct_min=13),
        limits=dict(seconds=420,reserve_seconds=30,OS_bytes=8<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=512<<20),
        runtime_binding_scope='Original source andtwo adopted FIT packets,completed FFN control,screen/code/Python/selected runtime/foreign hashes;not full DLL-tree.',
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());assert b['schema']=='SOURCE_FFN_CAST_BINDING_V1'
    a.directory.mkdir(exist_ok=False);phase='startup';completed=[];forwards=0;generations=0;torch=None
    try:
        assert sys.version_info[:3]==(3,12,10) and Path(sys.executable).resolve()==Path(b['python']).resolve()
        assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        from torch.nn import functional as F
        import transformers
        from transformers import AutoTokenizer,FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        from chatbot_falcon_ssd_tiles import install
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
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'output cap'
        def event(stage,**fields):print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True);guard()
        write(a.directory/'source_method.json',install(code))
        phase='load'
        tokenizer=AutoTokenizer.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False)
        model=FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        assert len(tokenizer)==65537 and tokenizer.bos_token_id==17 and model.generation_config.eos_token_id==[11,228]
        identities={n:id(p) for n,p in model.named_parameters()}
        assert len(identities)==411 and sum(p.numel() for p in model.parameters())==1554863488
        def f32_ffn(self,x):
            dtype=x.dtype;x=x.float()
            u=F.linear(x,self.up_proj.weight.float())
            g=F.linear(x,self.gate_proj.weight.float())*self.gate_multiplier
            z=u*F.silu(g)
            return (F.linear(z,self.down_proj.weight.float())*self.down_multiplier).to(dtype)
        for layer in model.model.layers:
            ffn=layer.feed_forward
            assert ffn.hidden_size==2048 and ffn.intermediate_size==4608 and ffn.config.hidden_act=='silu'
            assert all(getattr(ffn,n+'_proj').bias is None for n in ('gate','up','down'))
            ffn.forward=types.MethodType(f32_ffn,ffn)
        assert {n:id(p) for n,p in model.named_parameters()}==identities
        event('F32_FFN_ready',changed_weight_values=0,quantizations=0)
        records={r['id']:r for r in json.loads(Path(b['corpus']).read_bytes())['records']};observations=[]
        def lp(x):x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
        for identifier in b['selected_ids']:
            phase=identifier;t0=time.monotonic();rec=records[identifier];prompt=rec['input_ids'];outputs=rec['output_ids'];past=None;actual=[]
            with torch.inference_mode():
                for index in range(len(outputs)):
                    ids=prompt if index==0 else [outputs[index-1]]
                    result=model(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(prompt)+index),dtype=torch.long,device='cuda'),
                        past_key_values=past,use_cache=True,logits_to_keep=1)
                    past=result.past_key_values;logits=result.logits[0,-1];forwards+=1
                    assert logits.dtype==torch.bfloat16 and torch.isfinite(logits).all().item()
                    actual.append(logits.view(torch.uint16).cpu().numpy().copy());del result,logits;guard()
            bits=np.stack(actual);path=a.directory/(identifier+'.logits.bf16')
            with path.open('xb') as stream:stream.write(bits.astype('<u2',copy=False).tobytes());stream.flush();os.fsync(stream.fileno())
            sbits=np.fromfile(rec['logits']['path'],dtype='<u2').reshape(rec['logits']['shape'])
            source=(sbits.astype('<u4')<<16).view('<f4');changed=(bits.astype('<u4')<<16).view('<f4');labels=[]
            for offset in range(0,len(outputs),16):
                p=lp(source[offset:offset+16].astype(np.float64));q=lp(changed[offset:offset+16].astype(np.float64))
                labels.extend((np.exp(p)*(p-q)).sum(-1).tolist())
            winners=changed.argmax(-1).tolist();assert all(np.isfinite(v) and v>=-1e-10 for v in labels)
            row=dict(id=identifier,labels=len(outputs),case_KL_F64=sum(labels)/len(labels),label_KL_F64=labels,predicted_ids=winners,
                disagreements=sum(x!=y for x,y in zip(winners,outputs,strict=True)),stop=rec['stop'],seconds=time.monotonic()-t0,
                logits=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)))
            write(a.directory/(identifier+'.json'),row);observations.append(row);completed.append(identifier)
            event('forced_case',id=identifier,KL=row['case_KL_F64'],disagreements=row['disagreements'],labels=row['labels'])
            del past,actual,bits,sbits,source,changed
        assert forwards==274
        phase='generation';spec=json.loads(Path(b['screen_cases']).read_bytes());assert len(spec['cases'])==16 and spec['max_new_tokens']==64
        generated=[]
        for case in spec['cases']:
            guard();messages=case.get('history',[])+[dict(role='user',content=case['prompt'])]
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            expected=tokenizer.bos_token+''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
            assert text==expected and ids==tokenizer.encode(text,add_special_tokens=False) and len(ids)+64<=256
            with torch.inference_mode():
                output=model.generate(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long),
                    max_new_tokens=64,do_sample=False,use_cache=True,return_dict_in_generate=True,output_scores=True)
                generations+=1;new_ids=output.sequences[0,len(ids):].tolist();scores=torch.stack(output.scores).squeeze(1).float().cpu().numpy()
            assert scores.shape==(len(new_ids),65537) and np.isfinite(scores).all() and scores.argmax(-1).tolist()==new_ids
            identifier='screen.'+case['id'];path=a.directory/(identifier+'.scores.f32')
            with path.open('xb') as stream:stream.write(scores.astype('<f4',copy=False).tobytes());stream.flush();os.fsync(stream.fileno())
            decoded=tokenizer.decode(new_ids,skip_special_tokens=True)
            row=dict(id=identifier,category=case['category'],expected=case['expected'],messages=messages,serialized=text,input_ids=ids,output_ids=new_ids,
                output_text=decoded,correct=normalized(decoded)==case['expected'],blank=not decoded.strip(),
                special_leak=any(v in set(tokenizer.all_special_ids)-{11,228} for v in new_ids),
                stop_policy=not any(v in (11,228) for v in new_ids[:-1]) and (new_ids[-1] in (11,228) or len(new_ids)==64),
                eos_stop=new_ids[-1] in (11,228),scores=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)))
            write(a.directory/(identifier+'.json'),row);generated.append(row);completed.append(identifier)
            event('generation',id=identifier,correct=row['correct'],answer=decoded,labels=len(new_ids))
        assert generations==16 and {n:id(p) for n,p in model.named_parameters()}==identities
        c=b['criteria'];gates=dict(both_FIT_KL=all(r['case_KL_F64']<=c['case_KL_max'] for r in observations),
            FIT_disagreement=sum(r['disagreements'] for r in observations)/274<=c['label_disagreement_max'],
            screen_correct=sum(r['correct'] for r in generated)>=c['screen_correct_min'],
            every_category=all(sum(r['correct'] for r in generated if r['category']==category)>=2 for category in ('arithmetic','extraction','instruction','history')),
            screen_blank=sum(r['blank'] for r in generated)<=1,no_special_leak=not any(r['special_leak'] for r in generated),
            generation_stop=all(r['stop_policy'] for r in generated))
        phase='result';guard();decision='SOURCE_FFN_F32_CAST_PASS' if all(gates.values()) else 'SOURCE_FFN_F32_CAST_FAIL'
        write(a.out,dict(schema='SOURCE_FFN_F32_CAST_RESULT_V1',decision=decision,freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),cases=2,observations=observations,generations=16,generated_rows=generated,
            screen_correct=sum(r['correct'] for r in generated),source_screen_correct=14,gates=gates,new_forced_forwards=forwards,
            original_source_generations=0,weight_quantizations=0,activation_quantizations=0,optimizer_updates=0,native_runs=0,reserved_queries=0,
            unchanged_parameter_objects=411,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start,quality_admission=False,native_admission=False,
            scope='Two consumed FIT trajectories andknown16 screen;unquantized original FFN coefficients withF32 internal arithmetic/final BF16 cast. No heldout broad/fresh chat/native/rate admission.'))
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
