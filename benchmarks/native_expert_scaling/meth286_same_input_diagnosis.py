#!/usr/bin/env python3
"""Same actual GPU inputs for frozen285 norm/projection/RoPE/attention probes."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
import psutil
import torch
from torch.nn import functional as TF
from torch.nn.attention import SDPBackend, sdpa_kernel
import meth280_complete_i16_fresh_prediction as Q

P,R=Q.P,Q.R
ROOT=P.ROOT;DOC=P.DOC;ART=ROOT/'results/native_expert_scaling'
PRIOR=DOC/'meth285_whole_prefix_qualification_result.json'
PRIOR_SHA='c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda'
SOURCE=ROOT/'benchmarks/native_expert_scaling/meth286_same_input_cpu.c'
WIDTHS={'norm':896,'q':896,'k':128,'v':128,'rope_q':896,'rope_k':128,'attention_current':896,'attention_bf16_probability':896}
LOCAL={'qwen':(ROOT/'.venv/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py','99fa98c5676604cf6ef505892b70fda5c1c4cd835971459f38d61090cccab1e4'),
       'sdpa':(ROOT/'.venv/Lib/site-packages/transformers/integrations/sdpa_attention.py','fdf62fb0eb9b5dc0a6796a7e526015865267f271a0d5ac8377968ad05d1454b8')}

def metric(actual,reference):
    assert actual.shape==reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
    rel=np.linalg.norm(actual-reference,axis=-1)/np.maximum(np.linalg.norm(reference,axis=-1),1e-12)
    return {'relative_l2_median':float(np.median(rel)),'relative_l2_max':float(rel.max()),
            'unequal_coordinates':int(np.count_nonzero(actual!=reference)),
            'per_layer_max':rel.max(axis=1).tolist(),'per_layer_median':np.median(rel,axis=1).tolist()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    fixture=ART/'meth286_same_input.bin';native=ART/'meth286_operators.bin';refpath=ART/'meth286_reference.npz';exe=ART/'meth286_same_input_cpu.exe'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.failure.json'),fixture,native,refpath))
    start=time.monotonic();stage='binding'
    try:
        for path,sha in ((PRIOR,PRIOR_SHA),(Q.CORE,Q.CORE_SHA),(Q.EXPORT,Q.EXPORT_SHA),(Q.MANIFEST,Q.MANIFEST_SHA),*LOCAL.values()):assert P.digest(path)==sha,str(path)
        prior=json.loads(PRIOR.read_text(encoding='utf-8'));assert prior['decision']=='whole_native_prefix_smoke_fail_stop_before_quality_and_rate'
        for path,sha in prior['source_sha256'].items():assert P.digest(ROOT/path)==sha,path
        for path,sha in json.loads(Q.EXPORT.read_text(encoding='utf-8'))['helper_sha256'].items():assert P.digest(Path(path))==sha
        item=json.loads(Q.MANIFEST.read_text(encoding='utf-8'))['items'][0];assert len(item['prompt_ids'])==147
        device=R.G.Q.M.D.Q.setup();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);assert torch.get_num_threads()==6
        model,wrappers,proposal,load_record=R.load_stored(Q.CORE,device,Q.CORE_SHA);P.M44.set_experts(wrappers,False)
        captures=[{} for _ in range(24)];handles=[];sdpa_calls=[];original=TF.scaled_dot_product_attention
        def pre(index,key):
            def hook(module,inputs):captures[index][key]=inputs[0].detach().clone()
            return hook
        def post(index,key):
            def hook(module,inputs,output):captures[index][key]=output.detach().clone()
            return hook
        for index,layer in enumerate(model.model.layers):
            handles.append(layer.input_layernorm.register_forward_pre_hook(pre(index,'x')))
            handles.append(layer.input_layernorm.register_forward_hook(post(index,'norm')))
            for name in ('q','k','v'):handles.append(getattr(layer.self_attn,name+'_proj').register_forward_hook(post(index,name)))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(pre(index,'attention')))
        def observe(q,k,v,*positional,**kwargs):
            assert not positional and kwargs.get('attn_mask') is None and kwargs['is_causal'] and kwargs['dropout_p']==0 and kwargs['scale']==.125
            assert q.dtype==k.dtype==v.dtype==torch.bfloat16 and q.shape==(1,14,147,64)
            out=original(q,k,v,**kwargs);sdpa_calls.append((q.detach().clone(),k.detach().clone(),v.detach().clone(),out.detach().clone(),kwargs.copy()));return out
        stage='actual_GPU_capture';TF.scaled_dot_product_attention=observe
        try:
            with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                hidden=model.model(torch.tensor(item['prompt_ids'],device=device)[None],use_cache=False).last_hidden_state
        finally:
            TF.scaled_dot_product_attention=original
            for handle in handles:handle.remove()
        backend=[{'name':event.key,'count':event.count} for event in prof.key_averages() if 'scaled_dot_product' in event.key]
        assert len(sdpa_calls)==24
        def array(value):return value.squeeze(0).float().cpu().numpy().astype('<f4',copy=False)
        reference={key:[] for key in ('norm','q','k','v','rope_q','rope_k','attention','attention_math')}
        with fixture.open('wb') as file,torch.inference_mode():
            file.write(struct.pack('<8s2I',b'M286IN01',24,147))
            for index,(q,k,v,out,kwargs) in enumerate(sdpa_calls):
                cap=captures[index]
                assert torch.equal(out.transpose(1,2).reshape(1,147,896),cap['attention'])
                if k.shape[1]==14:
                    assert torch.equal(k,k[:,::7].repeat_interleave(7,dim=1)) and torch.equal(v,v[:,::7].repeat_interleave(7,dim=1))
                    smallk,smallv=k[:,::7],v[:,::7]
                else:
                    assert k.shape==(1,2,147,64);smallk,smallv=k,v
                postq=q.transpose(1,2).reshape(1,147,896);postk=smallk.transpose(1,2).reshape(1,147,128)
                assert torch.equal(smallv.transpose(1,2).reshape(1,147,128),cap['v'])
                with sdpa_kernel([SDPBackend.MATH]):math=original(q,k,v,**kwargs)
                for value in (cap['x'],cap['norm'],cap['q'],cap['k'],postq,postk,cap['v']):file.write(array(value).tobytes())
                for key in ('norm','q','k','v'):reference[key].append(array(cap[key]))
                reference['rope_q'].append(array(postq));reference['rope_k'].append(array(postk))
                reference['attention'].append(array(cap['attention']));reference['attention_math'].append(array(math.transpose(1,2).reshape(1,147,896)))
                if time.monotonic()-start>720 or psutil.Process().memory_info().rss>20*(1<<30) or torch.cuda.max_memory_allocated(device)>10.5*(1<<30):raise RuntimeError('M286 GPU resource stop')
        reference={k:np.stack(v) for k,v in reference.items()};np.savez(refpath,**reference);torch.cuda.synchronize(device)
        gpu={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,'peak_cuda_bytes':torch.cuda.max_memory_allocated(device)}
        del model,wrappers,proposal,hidden,captures,sdpa_calls,q,k,v,out,cap,postq,postk,smallk,smallv,math;torch.cuda.empty_cache()
        stage='isolated_CPU_operators';cpu_start=time.monotonic()
        command=['clang','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11',str(SOURCE),'-o',str(exe),'-lm','-lpsapi','-lbcrypt']
        build=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        run=subprocess.run([str(exe),str(Q.CORE),str(fixture),str(native)],capture_output=True,text=True,check=True,timeout=180-(time.monotonic()-cpu_start))
        loader=json.loads(run.stderr);assert loader['fields_bound']==725 and not loader['fallback'] and loader['archive_sha256']==Q.CORE_SHA
        actual={k:[] for k in WIDTHS}
        with native.open('rb') as file:
            assert struct.unpack('<8s2I',file.read(16))==(b'M286OP01',24,147)
            for index in range(24):
                for key,width in WIDTHS.items():actual[key].append(np.frombuffer(file.read(147*width*4),dtype='<f4').copy().reshape(147,width))
            assert not file.read(1)
        actual={k:np.stack(v) for k,v in actual.items()}
        summaries={key:metric(actual[key],reference[key]) for key in ('norm','q','k','v','rope_q','rope_k')}
        for key in ('attention_current','attention_bf16_probability'):
            summaries[key+'_vs_actual_default']=metric(actual[key],reference['attention'])
            summaries[key+'_vs_GPU_math']=metric(actual[key],reference['attention_math'])
        summaries['GPU_math_vs_actual_default']=metric(reference['attention_math'],reference['attention'])
        c=summaries['attention_current_vs_actual_default'];b=summaries['attention_bf16_probability_vs_actual_default']
        reduction=b['relative_l2_median']<=.5*c['relative_l2_median'] and b['relative_l2_max']<=.5*c['relative_l2_max'] and c['relative_l2_max']>0
        assert fixture.stat().st_size+native.stat().st_size+refpath.stat().st_size<256*(1<<20)
        result={'experiment':'METH-286-same-input-numerical-localization','prior_sha256':PRIOR_SHA,'artifact_sha256':Q.CORE_SHA,
                'source_id':item['source_id'],'prompt_ids_sha256':item['prompt_ids_sha256'],'arm':'stored_compact_core_only',
                'positions':147,'layers':24,'actual_SDPA_backend_events':backend,'summary':summaries,
                'BF16_normalized_probability_halves_median_and_max':reduction,'GPU_runtime':gpu,'CPU_seconds':time.monotonic()-cpu_start,'seconds':time.monotonic()-start,
                'loader':loader,'load_record':load_record,'compile_command':command,'compile_stderr':build.stderr,'native_report':json.loads(run.stdout),
                'sha256':{str(p.relative_to(ROOT)):P.digest(p) for p in (SOURCE,Path(__file__),fixture,native,refpath,exe,*[p for p,s in LOCAL.values()])},
                'diagnostic_only':True,'native_promotion_qualified':False,
                'decision':'same_input_diagnosis_recorded_requires_new_execution_recipe_and_unchanged285_gates',
                'scope':'Consumed one147-token prefix/all24 layers;actual GPU intermediate oracle inputs. No complete changed forward/quality/rate/new useful capacity demonstrated. Original285 failure unchanged.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps({'decision':result['decision'],'backend':backend,'summary':summaries,'seconds':result['seconds']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8');raise

if __name__=='__main__':main()
