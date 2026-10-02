#!/usr/bin/env python3
"""Frozen285 propagation trace and remaining same-input numeric controls."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
import psutil
import torch
import meth286_same_input_diagnosis as A

Q,P,R=A.Q,A.P,A.R
ROOT,DOC,ART=A.ROOT,A.DOC,A.ART
PRIOR=DOC/'meth286_same_input_diagnosis_result.json'
PRIOR_SHA='1c73b9fa032318f820de796b3e7743708846fcd4888bb414d4e75fd1750202bc'
SOURCE=ROOT/'benchmarks/native_expert_scaling/meth287_layer_trace_cpu.c'
TRACE_SOURCE=ROOT/'benchmarks/native_expert_scaling/meth287_trace_forward.h'
TRACE_KEYS=('x','postatt','postnorm','ffn_fp32','final_x')
ISOLATED=('o','postnorm','ffn_fp32','postatt','final_x')

def prove_trace():
    original=(ROOT/'benchmarks/native_expert_scaling/meth285_model_operator.h').read_text(encoding='utf-8')
    original=original[original.index('static void cc_forward('):original.index('static int cc_argmax')]
    trace=TRACE_SOURCE.read_text(encoding='utf-8').split('static float *cc_trace;\n')[1]
    for copy in ('memcpy(cc_trace+((size_t)pos*24+li)*5*D,s->x,D*4);\n        ',
                 'memcpy(cc_trace+(((size_t)pos*24+li)*5+1)*D,s->x,D*4);\n        ',
                 'memcpy(cc_trace+(((size_t)pos*24+li)*5+2)*D,s->norm,D*4);',
                 'memcpy(cc_trace+(((size_t)pos*24+li)*5+3)*D,s->tmp,D*4);',
                 '\n        memcpy(cc_trace+(((size_t)pos*24+li)*5+4)*D,s->x,D*4);'):
        assert trace.count(copy)==1;trace=trace.replace(copy,'',1)
    assert trace.replace('cc_forward_trace(','cc_forward(',1)==original

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    fixture=ART/'meth287_same_input.bin';idsfile=ART/'meth287_ids.bin';native=ART/'meth287_operators.bin'
    tracefile=ART/'meth287_trace.bin';refpath=ART/'meth287_reference.npz';exe=ART/'meth287_layer_trace_cpu.exe'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.failure.json'),fixture,idsfile,native,tracefile,refpath))
    start=time.monotonic();stage='bindings';prove_trace()
    try:
        for path,sha in ((PRIOR,PRIOR_SHA),(A.PRIOR,A.PRIOR_SHA),(Q.CORE,Q.CORE_SHA),(Q.EXPORT,Q.EXPORT_SHA),(Q.MANIFEST,Q.MANIFEST_SHA),*A.LOCAL.values()):assert P.digest(path)==sha,str(path)
        previous=json.loads(A.PRIOR.read_text(encoding='utf-8'));local=json.loads(PRIOR.read_text(encoding='utf-8'))
        for path,sha in previous['source_sha256'].items():assert P.digest(ROOT/path)==sha
        for path,sha in json.loads(Q.EXPORT.read_text(encoding='utf-8'))['helper_sha256'].items():assert P.digest(Path(path))==sha
        oldnative=ART/'meth285_native_prefixes.bin';oldreference=ART/'meth285_gpu_prefixes.npz'
        assert P.digest(oldnative)==previous['native_sha256'] and P.digest(oldreference)==previous['reference_sha256']
        assert local['summary']['GPU_math_vs_actual_default']['unequal_coordinates']==0
        item=json.loads(Q.MANIFEST.read_text(encoding='utf-8'))['items'][0];assert len(item['prompt_ids'])==147
        idsfile.write_bytes(np.asarray(item['prompt_ids'],dtype='<i4').tobytes())
        device=R.G.Q.M.D.Q.setup();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);assert torch.get_num_threads()==6
        model,wrappers,proposal,load_record=R.load_stored(Q.CORE,device,Q.CORE_SHA);P.M44.set_experts(wrappers,False)
        captures=[{} for _ in range(24)];handles=[]
        def pre(index,key):
            def hook(module,inputs):captures[index][key]=inputs[0].detach().clone()
            return hook
        def post(index,key):
            def hook(module,inputs,output):captures[index][key]=output.detach().clone()
            return hook
        for index,layer in enumerate(model.model.layers):
            handles.append(layer.input_layernorm.register_forward_pre_hook(pre(index,'x')))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(pre(index,'attention')))
            handles.append(layer.self_attn.o_proj.register_forward_hook(post(index,'o')))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(pre(index,'postatt')))
            handles.append(layer.post_attention_layernorm.register_forward_hook(post(index,'postnorm')))
            handles.append(layer.mlp.base.register_forward_hook(post(index,'ffn_bf16')))
            handles.append(layer.register_forward_hook(post(index,'final_x')))
        stage='GPU_trace'
        try:
            with torch.inference_mode():hidden=model.model(torch.tensor(item['prompt_ids'],device=device)[None],use_cache=False).last_hidden_state
        finally:
            for handle in handles:handle.remove()
        def array(value):return value.squeeze(0).float().cpu().numpy().astype('<f4',copy=False)
        old=np.load(oldreference);assert np.array_equal(array(hidden)[-8:],old['stored_compact_core_only_hidden'][0]);old.close()
        reference={key:[] for key in ('x','attention','o','postatt','postnorm','ffn_fp32','ffn_bf16','final_x')}
        with fixture.open('wb') as file,torch.inference_mode():
            file.write(b'M287IN01')
            for index,cap in enumerate(captures):
                fp32=wrappers[index].base.fp32(cap['postnorm']);assert torch.equal(fp32.bfloat16(),cap['ffn_bf16'])
                for key in ('x','attention','o','postatt','postnorm','ffn_bf16'):file.write(array(cap[key]).tobytes())
                for key in reference:reference[key].append(array(fp32 if key=='ffn_fp32' else cap[key]))
                if time.monotonic()-start>720 or psutil.Process().memory_info().rss>20*(1<<30) or torch.cuda.max_memory_allocated(device)>10.5*(1<<30):raise RuntimeError('M287 GPU resource stop')
        reference={k:np.stack(v) for k,v in reference.items()};np.savez(refpath,**reference);torch.cuda.synchronize(device)
        gpu={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,'peak_cuda_bytes':torch.cuda.max_memory_allocated(device)}
        del model,wrappers,proposal,hidden,captures,cap,fp32;torch.cuda.empty_cache()
        stage='CPU_trace_and_same_input';cpu_start=time.monotonic()
        command=['clang','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11',str(SOURCE),'-o',str(exe),'-lm','-lpsapi','-lbcrypt']
        build=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        run=subprocess.run([str(exe),str(Q.CORE),str(idsfile),str(fixture),str(tracefile),str(native)],capture_output=True,text=True,check=True,timeout=180-(time.monotonic()-cpu_start))
        loader=json.loads(run.stderr);assert loader['fields_bound']==725 and not loader['fallback'] and loader['archive_sha256']==Q.CORE_SHA
        with tracefile.open('rb') as file:
            assert file.read(8)==b'M287TR01'
            trace=np.frombuffer(file.read(147*24*5*896*4),dtype='<f4').copy().reshape(147,24,5,896).transpose(1,0,2,3)
            final=file.read((896+151936)*4);assert not file.read(1)
        with oldnative.open('rb') as file:
            file.seek(28+20+7*(896+151936)*4);assert final==file.read((896+151936)*4),'Instrumented CPU must reproduce original285 final bytes'
        isolated={key:[] for key in ISOLATED}
        with native.open('rb') as file:
            assert file.read(8)==b'M287OP01'
            for index in range(24):
                for key in ISOLATED:isolated[key].append(np.frombuffer(file.read(147*896*4),dtype='<f4').copy().reshape(147,896))
            assert not file.read(1)
        isolated={k:np.stack(v) for k,v in isolated.items()}
        propagation={key:A.metric(trace[:,:,index,:],reference[key]) for index,key in enumerate(TRACE_KEYS)}
        localmetrics={key:A.metric(isolated[key],reference[key]) for key in ISOLATED}
        rounded=torch.from_numpy(isolated['ffn_fp32']).bfloat16().float().numpy()
        localmetrics['ffn_bf16']=A.metric(rounded,reference['ffn_bf16'])
        assert sum(p.stat().st_size for p in (fixture,native,tracefile,refpath))<384*(1<<20)
        result={'experiment':'METH-287-propagated-layer-trace-and-same-input-remaining-operators',
                'prior_sha256':PRIOR_SHA,'artifact_sha256':Q.CORE_SHA,'source_id':item['source_id'],'prompt_ids_sha256':item['prompt_ids_sha256'],
                'positions':147,'layers':24,'arm':'stored_compact_core_only','propagated':propagation,'same_input':localmetrics,
                'GPU_tail8_exact_original285_reference':True,'CPU_final_hidden_logits_bytes_exact_original285':True,'trace_source_exact_285_after_removing_copies':True,
                'GPU_runtime':gpu,'CPU_seconds':time.monotonic()-cpu_start,'seconds':time.monotonic()-start,'loader':loader,'load_record':load_record,
                'compile_command':command,'compile_stderr':build.stderr,'native_report':json.loads(run.stdout),
                'sha256':{str(p.relative_to(ROOT)):P.digest(p) for p in (SOURCE,TRACE_SOURCE,Path(__file__),idsfile,fixture,native,tracefile,refpath,exe)},
                'diagnostic_only':True,'native_promotion_qualified':False,
                'decision':'layer_drift_recorded_original285_failure_unchanged',
                'scope':'One consumed full147-token prefix/24 layers, actual GPU oracles. No corrected equation/quality/rate/new learned capacity.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps({'decision':result['decision'],'propagated':propagation,'same_input':localmetrics,'seconds':result['seconds']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8');raise

if __name__=='__main__':main()
