#!/usr/bin/env python3
"""Prospective same-archive no-cache causal equivalence and head geometry."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from torch.nn import functional as TF
import meth280_complete_i16_fresh_prediction as Q

P,R=Q.P,Q.R
ROOT,DOC=P.ROOT,P.DOC
ART=ROOT/'results/native_expert_scaling'
ORIGINAL=DOC/'meth285_whole_prefix_qualification_result.json'
ORIGINAL_SHA='c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda'
LATEST=DOC/'meth289_residual32_qualification_result.json'
LATEST_SHA='091f9894325fe4f24e4808585964a322c615112238fb12ffbc56463d76d3e1b1'
INDICES=(0,1,8,9,16,17)
ARMS=('stored_compact_core_only','stored_complete_e1280')

def metrics(actual,reference):
    assert actual.shape==reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
    rel=np.linalg.norm(actual-reference,axis=-1)/np.maximum(np.linalg.norm(reference,axis=-1),1e-12)
    return {'relative_l2':rel.tolist(),'relative_l2_median':float(np.median(rel)),
            'relative_l2_max':float(rel.max()),'unequal_coordinates':int(np.count_nonzero(actual!=reference))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',type=Path,default=ART/'meth290_prefix_equivalence.npz');args=ap.parse_args()
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json'),args.reference))
    start=time.monotonic();stage='bindings';records=[];summary={};gates={};reference={}
    def partial():args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'records':records,'summary':summary,'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    try:
        for path,sha in ((ORIGINAL,ORIGINAL_SHA),(LATEST,LATEST_SHA),(Q.CORE,Q.CORE_SHA),(Q.EXPORT,Q.EXPORT_SHA),(Q.MANIFEST,Q.MANIFEST_SHA)):assert P.digest(path)==sha,str(path)
        original=json.loads(ORIGINAL.read_text(encoding='utf-8'));assert original['artifact_sha256']==Q.CORE_SHA
        assert original['decision']=='whole_native_prefix_smoke_fail_stop_before_quality_and_rate'
        saved=ART/'meth285_gpu_prefixes.npz';assert P.digest(saved)==original['reference_sha256']
        with np.load(saved) as file:baseline={k:file[k].copy() for k in file.files}
        for path,sha in json.loads(Q.EXPORT.read_text(encoding='utf-8'))['helper_sha256'].items():assert P.digest(Path(path))==sha
        items=[json.loads(Q.MANIFEST.read_text(encoding='utf-8'))['items'][i] for i in INDICES]
        assert [len(i['prompt_ids']) for i in items]==[147,152,166,161,209,184]
        device=R.G.Q.M.D.Q.setup();assert torch.get_num_threads()==6
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        model,wrappers,proposal,load_record=R.load_stored(Q.CORE,device,Q.CORE_SHA)
        def budget():
            rss=psutil.Process().memory_info().rss;peak=torch.cuda.max_memory_allocated(device)
            if time.monotonic()-start>720 or rss>20*(1<<30) or peak>10.5*(1<<30):raise RuntimeError('METH-290 resource stop')
        def array(value):assert value.dtype==torch.bfloat16;return value.float().cpu().numpy()
        with torch.inference_mode():
            # Exact repeatability of the ORIGINAL full-prefix/M8 head is a prerequisite.
            stage='same_geometry_original_reference_repeat'
            for enabled,arm in enumerate(ARMS):
                P.M44.set_experts(wrappers,bool(enabled));rh=[];rl=[]
                for item in items:
                    ids=torch.as_tensor(item['prompt_ids'],device=device)[None]
                    hidden=model.model(ids,use_cache=False).last_hidden_state[0,-8:]
                    logits=TF.linear(hidden,model.lm_head.weight);rh.append(array(hidden));rl.append(array(logits));budget()
                reference[arm+'_repeat_hidden']=np.stack(rh);reference[arm+'_repeat_logits']=np.stack(rl)
                gates[arm+'_original_repeat_byte_exact']=bool(np.array_equal(reference[arm+'_repeat_hidden'],baseline[arm+'_hidden']) and np.array_equal(reference[arm+'_repeat_logits'],baseline[arm+'_logits']))
            if all(gates.values()):
                stage='causal_truncated_prefixes_and_head_M1'
                for enabled,arm in enumerate(ARMS):
                    P.M44.set_experts(wrappers,bool(enabled));hs=[];ls=[];head_m1=[]
                    for source_index,item in enumerate(items):
                        ids=torch.as_tensor(item['prompt_ids'],device=device)[None];n=ids.shape[1];hidden_rows=[];logit_rows=[];head_rows=[]
                        full_hidden=torch.as_tensor(reference[arm+'_repeat_hidden'][source_index],device=device).bfloat16()
                        for tail in range(8):
                            end=n-7+tail
                            hidden=model.model(ids[:,:end],use_cache=False).last_hidden_state[0,-1:]
                            logits=TF.linear(hidden,model.lm_head.weight)
                            single=TF.linear(full_hidden[tail:tail+1],model.lm_head.weight)
                            hidden_rows.append(array(hidden)[0]);logit_rows.append(array(logits)[0]);head_rows.append(array(single)[0]);budget()
                            records.append({'arm':arm,'source_index':INDICES[source_index],'tail_index':tail,'last_position':end-1,'prefix_tokens':end})
                        hs.append(np.stack(hidden_rows));ls.append(np.stack(logit_rows));head_m1.append(np.stack(head_rows));partial()
                    reference[arm+'_truncated_hidden']=np.stack(hs);reference[arm+'_truncated_logits']=np.stack(ls);reference[arm+'_full_hidden_head_M1']=np.stack(head_m1)
                    h=metrics(reference[arm+'_truncated_hidden'],baseline[arm+'_hidden'])
                    l=metrics(reference[arm+'_truncated_logits'],baseline[arm+'_logits'])
                    hm=metrics(reference[arm+'_full_hidden_head_M1'],baseline[arm+'_logits'])
                    top=reference[arm+'_truncated_logits'].argmax(-1);base_top=baseline[arm+'_logits'].argmax(-1);match=int(np.count_nonzero(top==base_top))
                    summary[arm]={'positions':48,'hidden':h,'logits':l,'isolated_head_M1_vs_M8':hm,
                                  'top1_matches':match,'top1_fraction':match/48,'truncated_top1':top.tolist(),'original_top1':base_top.tolist()}
                    gates[arm+'_causal_hidden_max_unchanged285']=h['relative_l2_max']<=.05
                    gates[arm+'_causal_logits_max_unchanged285']=l['relative_l2_max']<=.05
                    gates[arm+'_causal_top1_unchanged285']=match/48>=.95
                    gates[arm+'_finite']=True
        np.savez(args.reference,**reference);torch.cuda.synchronize(device);budget()
        assert args.reference.stat().st_size+saved.stat().st_size<256*(1<<20)
        repeat=all(gates[arm+'_original_repeat_byte_exact'] for arm in ARMS)
        decision=('reference_same_geometry_not_repeatable_stop' if not repeat else
                  'reference_causal_equivalence_pass_diagnostic_only' if all(gates.values()) else 'reference_causal_equivalence_fail_diagnostic_only')
        result={'experiment':'METH-290-reference-no-cache-causal-equivalence','artifact_sha256':Q.CORE_SHA,
                'original285_sha256':ORIGINAL_SHA,'latest289_sha256':LATEST_SHA,'manifest_sha256':Q.MANIFEST_SHA,
                'original_reference_sha256':original['reference_sha256'],'new_reference_sha256':P.digest(args.reference),
                'script_sha256':P.digest(Path(__file__)),'sources':[{'index':i,'source_id':item['source_id'],'prompt_ids_sha256':item['prompt_ids_sha256']} for i,item in zip(INDICES,items)],
                'records':records,'summary':summary,'gates':gates,'load_record':load_record,'seconds':time.monotonic()-start,
                'rss_bytes':psutil.Process().memory_info().rss,'peak_cuda_bytes':torch.cuda.max_memory_allocated(device),
                'torch_version':torch.__version__,'GPU':torch.cuda.get_device_name(device),'diagnostic_only':True,'native_promotion_qualified':False,
                'decision':decision,'scope':'Same archive/six consumed prefixes/two arms/no cache both sides. Only causal batch geometry and M1/M8 head geometry differ. Original native285/288/289 stops unchanged. No native quality/rate/new useful capacity/family claim.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','summary','gates','seconds')}),flush=True)
    except BaseException as error:
        partial();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8');raise

if __name__=='__main__':main()
