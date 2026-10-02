#!/usr/bin/env python3
"""Prospective changed norm accumulation; unchanged285 full prefix/cache gates."""
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
import meth280_complete_i16_fresh_prediction as Q

P,R=Q.P,Q.R
ROOT=P.ROOT;DOC=P.DOC;ART=ROOT/'results/native_expert_scaling'
OPERATORS=DOC/'meth284_native_archive_qualification_result.json'
OPERATORS_SHA='615e4854b8e9300f660c66f513152e4ed63b78e41f6d521400f784bede0ca18a'
INDICES=(0,1,8,9,16,17)
SOURCE=ROOT/'benchmarks/native_expert_scaling/meth288_norm_reduction_cpu.c'
MODEL=ROOT/'benchmarks/native_expert_scaling/meth288_model_operator.h'
ARMS=('stored_compact_core_only','stored_complete_e1280')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--exe',type=Path,default=ART/'meth288_norm_reduction_cpu.exe')
    ap.add_argument('--bundle',type=Path,default=ART/'meth288_prefix_ids.bin')
    ap.add_argument('--native',type=Path,default=ART/'meth288_native_prefixes.bin')
    ap.add_argument('--reference',type=Path,default=ART/'meth285_gpu_prefixes.npz')
    args=ap.parse_args()
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json'),args.bundle,args.native))
    start=time.monotonic();stage='bindings';sources=[];summaries={}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'sources':sources,
            'summary':summaries,'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    try:
        for path,sha in ((OPERATORS,OPERATORS_SHA),(Q.CORE,Q.CORE_SHA),(Q.MANIFEST,Q.MANIFEST_SHA),(Q.EXPORT,Q.EXPORT_SHA)):
            assert P.digest(path)==sha,str(path)
        prior=json.loads(OPERATORS.read_text(encoding='utf-8'))
        assert all(prior['gates'].values()) and prior['artifact_sha256']==Q.CORE_SHA
        assert prior['decision']=='native_archive_operators_pass_requires_whole_model_qualification'
        old_model=(ROOT/'benchmarks/native_expert_scaling/meth285_model_operator.h').read_text(encoding='utf-8')
        old_norm='float square=0;for(int i=0;i<D;i++)square+=x[i]*x[i];float inv=1.0f/sqrtf(square/(float)D+1e-6f);'
        new_norm='double square=0;for(int i=0;i<D;i++)square+=(double)x[i]*(double)x[i];float mean=(float)(square/(double)D);float inv=1.0f/sqrtf(mean+1e-6f);'
        assert old_model.count(old_norm)==1 and MODEL.read_text(encoding='utf-8')==old_model.replace(old_norm,new_norm,1)
        old_main=(ROOT/'benchmarks/native_expert_scaling/meth285_complete_core_cpu.c').read_text(encoding='utf-8').split('int main',1)[1]
        assert SOURCE.read_text(encoding='utf-8').split('int main',1)[1]==old_main
        manifest=json.loads(Q.MANIFEST.read_text(encoding='utf-8'));items=[manifest['items'][i] for i in INDICES]
        assert [i['category'] for i in items]==['code']*2+['prose']*2+['technical_general']*2
        with args.bundle.open('wb') as f:
            f.write(struct.pack('<8sI',b'M285ID01',6))
            for index,item in zip(INDICES,items):
                ids=np.asarray(item['prompt_ids'],dtype='<i4');assert 8<=len(ids)<=4096
                assert P.M17.sha(ids.tobytes())==item['prompt_ids_sha256']
                f.write(struct.pack('<I',len(ids)));f.write(ids.tobytes())
                sources.append({'index':index,'source_id':item['source_id'],'category':item['category'],
                                'prefix_tokens':len(ids),'prompt_ids_sha256':item['prompt_ids_sha256']})
        composition=json.loads(Q.EXPORT.read_text(encoding='utf-8'))
        for path,sha in composition['helper_sha256'].items():assert P.digest(Path(path))==sha
        original_path=DOC/'meth285_whole_prefix_qualification_result.json'
        assert P.digest(original_path)=='c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda'
        original=json.loads(original_path.read_text(encoding='utf-8'))
        for path,sha in original['source_sha256'].items():
            if path!='benchmarks/phase60/engine.c':assert P.digest(ROOT/path)==sha,path
        trace_result=DOC/'meth287_layer_trace_result.json'
        assert P.digest(trace_result)=='32adcf57d582270510bd689726b1e34c217310a213f19635a5f15056e99f1650'
        assert P.digest(args.reference)==original['reference_sha256']
        with np.load(args.reference) as saved:reference={k:saved[k].copy() for k in saved.files}
        GPU={'reused_original285_reference_sha256':original['reference_sha256'],'new_GPU_execution':False}
        load_record=original['load_record']
        stage='actual_CPU_prefixes_and_cache';native_start=time.monotonic();partial()
        command=['clang','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11',
                 '-DSILICON_COMPLETE_I16_NORM64',str(ROOT/'benchmarks/phase60/engine.c'),'-o',str(args.exe),'-lm','-lpsapi','-lbcrypt']
        compiled=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        run=subprocess.run([str(args.exe),str(Q.CORE),str(args.bundle),str(args.native)],
                           capture_output=True,text=True,check=True,timeout=600-(time.monotonic()-native_start))
        native_report=json.loads(run.stdout);loader=json.loads(run.stderr)
        assert loader['fields_bound']==725 and not loader['fallback'] and loader['archive_sha256']==Q.CORE_SHA
        native={};cache_checks=[];row=896+151936
        with args.native.open('rb') as f:
            assert struct.unpack('<8s5I',f.read(28))==(b'M285LG01',2,6,8,896,151936)
            for arm_index,arm in enumerate(ARMS):
                hidden_rows=[];logit_rows=[]
                for source_index,item in enumerate(items):
                    meta=struct.unpack('<5I',f.read(20));assert meta[:3]==(arm_index,source_index,len(item['prompt_ids']))
                    values=np.frombuffer(f.read(8*row*4),dtype='<f4').copy().reshape(8,row)
                    wrong=np.frombuffer(f.read(row*4),dtype='<f4').copy()
                    hidden_rows.append(values[:,:896]);logit_rows.append(values[:,896:])
                    cache_checks.append({'arm':arm,'source_id':item['source_id'],'cached_and_fresh_bytes_exact':bool(meta[3]),
                                         'erased_history_detected':bool(meta[4]),
                                         'erased_history_logit_relative_l2':float(np.linalg.norm(wrong[896:]-values[-1,896:])/np.linalg.norm(values[-1,896:]))})
                native[arm+'_hidden']=np.stack(hidden_rows);native[arm+'_logits']=np.stack(logit_rows)
            assert not f.read(1)
        gates={'all725_same_archive_no_fallback':True,'all12_CPU_cache_rebuilds_byte_exact':all(r['cached_and_fresh_bytes_exact'] for r in cache_checks),
               'all12_erased_history_controls_detected':all(r['erased_history_detected'] for r in cache_checks)}
        for arm in ARMS:
            nh,nl=native[arm+'_hidden'],native[arm+'_logits'];rh,rl=reference[arm+'_hidden'],reference[arm+'_logits']
            hrel=np.linalg.norm(nh-rh,axis=-1)/np.maximum(np.linalg.norm(rh,axis=-1),1e-12)
            lrel=np.linalg.norm(nl-rl,axis=-1)/np.maximum(np.linalg.norm(rl,axis=-1),1e-12)
            ntop=nl.argmax(-1);rtop=rl.argmax(-1);matches=int(np.count_nonzero(ntop==rtop))
            summaries[arm]={'positions':48,'matching_top1':matches,'top1_fraction':matches/48,
                            'hidden_relative_l2':hrel.tolist(),'logit_relative_l2':lrel.tolist(),
                            'max_hidden_relative_l2':float(hrel.max()),'max_logit_relative_l2':float(lrel.max()),
                            'native_top1':ntop.tolist(),'GPU_top1':rtop.tolist()}
            gates[arm+'_finite']=bool(np.isfinite(nh).all() and np.isfinite(nl).all())
            gates[arm+'_hidden_max']=bool(hrel.max()<=.05)
            gates[arm+'_logit_max']=bool(lrel.max()<=.05)
            gates[arm+'_top1_smoke']=matches/48>=.95
        assert args.native.stat().st_size+args.reference.stat().st_size<160*(1<<20)
        result={'experiment':'METH-288-norm64-whole-complete-archive-prefix-cache-smoke',
                'artifact_sha256':Q.CORE_SHA,'operator_result_sha256':OPERATORS_SHA,'manifest_sha256':Q.MANIFEST_SHA,
                'original285_result_sha256':P.digest(original_path),'trace287_result_sha256':P.digest(trace_result),
                'sources':sources,'summary':summaries,'cache_checks':cache_checks,'gates':gates,
                'GPU_reference_runtime':GPU,'CPU_qualification_seconds':time.monotonic()-native_start,'seconds':time.monotonic()-start,
                'native_report':native_report,'loader':loader,'load_record':load_record,
                'source_sha256':{str(p.relative_to(ROOT)):P.digest(p) for p in (SOURCE,MODEL,ROOT/'benchmarks/phase60/engine.c')},
                'script_sha256':P.digest(Path(__file__)),'compile_command':command,'compile_stderr':compiled.stderr,
                'executable_sha256':P.digest(args.exe),'bundle_sha256':P.digest(args.bundle),
                'native_sha256':P.digest(args.native),'reference_sha256':P.digest(args.reference),
                'diagnostic_only':True,'native_promotion_qualified':False,
                'decision':'whole_native_prefix_smoke_pass_requires_full_quality_and_rate' if all(gates.values()) else 'whole_native_prefix_smoke_fail_stop_before_quality_and_rate',
                'changed_coordinate':'Only norm sum-of-squares uses F64, mean cast to F32 before original inverse/BF16 boundaries.',
                'scope':'Reuses frozen285 GPU reference. Six fixed consumed full prompts/tail48 positions per arm;stored compact-core-only is an ablation,not donor. GPU reference has no cache. CPU fresh/incremental exact cache control does not reopen264. No native BPB/generation/PIQA/K64/accepted rate/useful n/family qualification.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','gates','summary','seconds','CPU_qualification_seconds')}),flush=True)
    except BaseException as error:
        partial();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
