#!/usr/bin/env python3
"""Actual intervened inputs: restore one GPU operator group in an all-C hybrid."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import psutil
import torch
from torch.nn import functional as TF
import meth290_reference_prefix_equivalence as A

Q,P,R=A.Q,A.P,A.R
ROOT,DOC,ART=A.ROOT,A.DOC,A.ART
PRIOR=DOC/'meth290_reference_prefix_equivalence_result.json'
PRIOR_SHA='797b8ef913cea4b8eab2754ab37e66c0c75026854eec05a3c0ca9f8bfa2abbcb'
SOURCE=ROOT/'benchmarks/native_expert_scaling/meth291_operator_surgery.c'
GROUPS=('norm','projection','attention','ffn')
ARMS=('all_cpu','norm_gpu','projection_gpu','attention_gpu','ffn_gpu')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',type=Path,default=ART/'meth291_operator_surgery.npz');args=ap.parse_args()
    dllpath=ART/'meth291_operator_surgery.dll'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json'),args.reference))
    start=time.monotonic();stage='bindings';summary={};reference={};counts={}
    def partial():args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'summary':summary,'callback_counts':counts,'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    try:
        for path,sha in ((PRIOR,PRIOR_SHA),(A.ORIGINAL,A.ORIGINAL_SHA),(Q.CORE,Q.CORE_SHA),(Q.EXPORT,Q.EXPORT_SHA),(Q.MANIFEST,Q.MANIFEST_SHA)):assert P.digest(path)==sha,str(path)
        original=json.loads(A.ORIGINAL.read_text(encoding='utf-8'));prior=json.loads(PRIOR.read_text(encoding='utf-8'));assert all(prior['gates'].values())
        model_header=ROOT/'benchmarks/native_expert_scaling/meth285_model_operator.h'
        assert P.digest(model_header)==original['source_sha256'][str(model_header.relative_to(ROOT))]
        for path,sha in json.loads(Q.EXPORT.read_text(encoding='utf-8'))['helper_sha256'].items():assert P.digest(Path(path))==sha
        oldnative=ART/'meth285_native_prefixes.bin';oldref=ART/'meth285_gpu_prefixes.npz'
        assert P.digest(oldnative)==original['native_sha256'] and P.digest(oldref)==original['reference_sha256']
        with oldnative.open('rb') as file:
            file.seek(28+20);cpu=np.frombuffer(file.read(8*(896+151936)*4),dtype='<f4').copy().reshape(8,896+151936)
        with np.load(oldref) as file:rh=file['stored_compact_core_only_hidden'][0].copy();rl=file['stored_compact_core_only_logits'][0].copy()
        item=json.loads(Q.MANIFEST.read_text(encoding='utf-8'))['items'][0];assert len(item['prompt_ids'])==147
        command=['clang','-shared','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11',str(SOURCE),'-o',str(dllpath),'-lm','-lpsapi','-lbcrypt']
        build=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        dirs=[os.add_dll_directory(str(Path(shutil.which('clang')).parent)),os.add_dll_directory(str(ART))]
        lib=ctypes.CDLL(str(dllpath));ptr=ctypes.POINTER(ctypes.c_float)
        lib.m291_load.argtypes=[ctypes.c_char_p];lib.m291_load.restype=ctypes.c_int
        for name,nints in (('norm',3),('mat',3),('ffn',2)):
            fn=getattr(lib,'m291_'+name);fn.argtypes=[ctypes.c_int]*nints+[ptr,ptr];fn.restype=None
        lib.m291_attention.argtypes=[ctypes.c_int,ptr,ptr,ptr,ptr];lib.m291_attention.restype=None
        assert lib.m291_load(os.fsencode(Q.CORE))==725
        device=R.G.Q.M.D.Q.setup();assert torch.get_num_threads()==6
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        model,wrappers,proposal,load_record=R.load_stored(Q.CORE,device,Q.CORE_SHA);P.M44.set_experts(wrappers,False)
        ids=torch.tensor(item['prompt_ids'],device=device)[None];saved=[]
        for li,layer in enumerate(model.model.layers):
            saved.extend([(layer.input_layernorm,'norm',li,0,layer.input_layernorm.forward),
                          (layer.post_attention_layernorm,'norm',li,1,layer.post_attention_layernorm.forward),
                          (layer.mlp.base,'ffn',li,0,layer.mlp.base.forward)])
            for kind,name in enumerate(('q','k','v','o')):
                mod=getattr(layer.self_attn,name+'_proj');saved.append((mod,'projection',li,kind,mod.forward))
        saved.append((model.model.norm,'norm',0,2,model.model.norm.forward));original_attention=TF.scaled_dot_product_attention
        def budget():
            if time.monotonic()-start>720 or psutil.Process().memory_info().rss>20*(1<<30) or torch.cuda.max_memory_allocated(device)>10.5*(1<<30):raise RuntimeError('M291 resource stop')
        def arr(value):
            assert value.dtype==torch.bfloat16 and value.shape[-1]==896 and np.isfinite(value.float().cpu().numpy()).all()
            return np.ascontiguousarray(value.reshape(-1,896).float().cpu().numpy())
        def pointer(value):return value.ctypes.data_as(ptr)
        def cpu_operation(group,li,kind,value):
            data=arr(value);n=len(data);width=128 if group=='projection' and kind in (1,2) else 151936 if group=='projection' and kind==4 else 896
            out=np.empty((n,width),dtype=np.float32)
            if group=='norm':lib.m291_norm(li,kind,n,pointer(data),pointer(out))
            elif group=='projection':lib.m291_mat(li,kind,n,pointer(data),pointer(out))
            else:lib.m291_ffn(li,n,pointer(data),pointer(out))
            counts[group]=counts.get(group,0)+1
            result=torch.from_numpy(out).to(device=device,dtype=torch.bfloat16).reshape(*value.shape[:-1],width);budget();return result
        def callback(group,li,kind):
            def forward(value):return cpu_operation(group,li,kind,value)
            return forward
        def cpu_attention(q,k,v,*positional,**kwargs):
            assert not positional and kwargs['attn_mask'] is None and kwargs['is_causal'] and kwargs['dropout_p']==0 and kwargs['scale']==.125
            assert q.shape==(1,14,147,64) and q.dtype==k.dtype==v.dtype==torch.bfloat16
            if k.shape[1]==14:
                assert torch.equal(k,k[:,::7].repeat_interleave(7,dim=1)) and torch.equal(v,v[:,::7].repeat_interleave(7,dim=1));k=k[:,::7];v=v[:,::7]
            assert k.shape==v.shape==(1,2,147,64)
            qa=np.ascontiguousarray(q.transpose(1,2).reshape(147,896).float().cpu().numpy())
            ka=np.ascontiguousarray(k.transpose(1,2).reshape(147,128).float().cpu().numpy());va=np.ascontiguousarray(v.transpose(1,2).reshape(147,128).float().cpu().numpy());out=np.empty_like(qa)
            lib.m291_attention(147,pointer(qa),pointer(ka),pointer(va),pointer(out));counts['attention']=counts.get('attention',0)+1;budget()
            return torch.from_numpy(out).to(device=device,dtype=torch.bfloat16).reshape(1,147,14,64).transpose(1,2)
        gates={};closure=False
        with torch.inference_mode():
            stage='original_GPU_repeat';hidden=model.model(ids,use_cache=False).last_hidden_state[0,-8:];logits=TF.linear(hidden,model.lm_head.weight)
            assert np.array_equal(hidden.float().cpu().numpy(),rh) and np.array_equal(logits.float().cpu().numpy(),rl)
            for arm in ARMS:
                active=set(GROUPS) if arm=='all_cpu' else set(GROUPS)-{arm.removesuffix('_gpu')};stage=arm;counts={}
                try:
                    for module,group,li,kind,old in saved:module.forward=callback(group,li,kind) if group in active else old
                    TF.scaled_dot_product_attention=cpu_attention if 'attention' in active else original_attention
                    hidden=model.model(ids,use_cache=False).last_hidden_state[0,-8:]
                    logits=cpu_operation('projection',0,4,hidden) if 'projection' in active else TF.linear(hidden,model.lm_head.weight)
                finally:
                    for module,group,li,kind,old in saved:module.forward=old
                    TF.scaled_dot_product_attention=original_attention
                nh=hidden.float().cpu().numpy();nl=logits.float().cpu().numpy();reference[arm+'_hidden']=nh;reference[arm+'_logits']=nl
                h=A.metrics(nh,rh);l=A.metrics(nl,rl);matches=int(np.count_nonzero(nl.argmax(-1)==rl.argmax(-1)))
                summary[arm]={'hidden':h,'logits':l,'top1_matches':matches,'callback_counts':counts.copy()}
                if arm=='all_cpu':
                    closure=np.array_equal(nh,cpu[:,:896]) and np.array_equal(nl,cpu[:,896:]);gates['all_CPU_hybrid_exact_original285_tail8']=closure
                    summary[arm]['vs_original285_CPU_hidden']=A.metrics(nh,cpu[:,:896]);summary[arm]['vs_original285_CPU_logits']=A.metrics(nl,cpu[:,896:])
                    if not closure:partial();break
                else:
                    base=summary['all_cpu'];gates[arm+'_halves_hidden_and_logit_median_and_max']=all(summary[arm][key]['relative_l2_'+stat]<=.5*base[key]['relative_l2_'+stat] for key in ('hidden','logits') for stat in ('median','max'))
                partial();budget()
        np.savez(args.reference,**reference);torch.cuda.synchronize(device);budget();assert args.reference.stat().st_size<32*(1<<20)
        result={'experiment':'METH-291-actual-input-operator-group-causal-surgery','artifact_sha256':Q.CORE_SHA,'prior290_sha256':PRIOR_SHA,
                'original285_sha256':A.ORIGINAL_SHA,'source_id':item['source_id'],'prompt_ids_sha256':item['prompt_ids_sha256'],'positions':8,
                'arm':'compact_core_only','summary':summary,'gates':gates,'load_record':load_record,'C_fields_bound':725,'compile_command':command,'compile_stderr':build.stderr,
                'sha256':{str(p.relative_to(ROOT)):P.digest(p) for p in (SOURCE,Path(__file__),model_header,dllpath,args.reference)},
                'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,'peak_cuda_bytes':torch.cuda.max_memory_allocated(device),
                'diagnostic_only':True,'native_promotion_qualified':False,
                'decision':'causal_group_surgery_recorded_requires_new_deployable_equation' if closure else 'hybrid_CPU_closure_fail_stop_before_group_interpretation',
                'scope':'One consumed147-token prefix/tail8/bank-off; all-C hybrid must exactly reproduce285. One group GPU restored on actual intervened inputs,not teacher-state injection. Nondeployable GPU/CPU assay;no native quality/rate/new capacity.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','gates','summary','seconds')}),flush=True)
    except BaseException as error:
        partial();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8');raise

if __name__=='__main__':main()
