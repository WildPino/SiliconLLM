#!/usr/bin/env python3
"""Frozen consumed development regression of the diagnostic M276 archive."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
import meth260_complete_core_development as D
import meth276_diagnostic_complete_core as C

P,L,F=D.P,D.L,D.F
COMPOSITION=P.DOC/'meth276_diagnostic_complete_core_repair1_result.json'
COMPOSITION_SHA='ff4694b426725f3de2e500739ba11fb06ad280dc7a5ef04515ff5b4a592fe553'
PREVIOUS=P.DOC/'meth260_complete_core_development_repair1_result.json'
PREVIOUS_SHA='f8e102ce698410e43df7a5b1c14a2620a80df185ce65f0d535ba0bf0c4d046f9'
ARMS=('bf16_donor','bf16_e1280','stored_i16_private128_e1280')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--composition-sha',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();start=time.monotonic();stage='bindings';arms={}
    assert args.composition_sha==COMPOSITION_SHA
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json')))
    def partial():
        C.I.A.dump(args.out.with_suffix('.partial.json'),{'stage':stage,'arms':arms,'seconds':time.monotonic()-start})
    try:
        for path,sha in ((COMPOSITION,args.composition_sha),(PREVIOUS,PREVIOUS_SHA),(D.PRIOR,D.PRIOR_SHA),
                         (P.M122.MANIFEST,P.M122.MANIFEST_SHA),(P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),
                         (P.M122.TRAINING,P.M57.TRAINING_SHA)):
            assert P.digest(path)==sha,str(path)
        composition=json.loads(COMPOSITION.read_text(encoding='utf-8'))
        previous=json.loads(PREVIOUS.read_text(encoding='utf-8'))
        assert all(composition['gates'].values()) and composition['diagnostic_only'] is True
        assert composition['native_promotion_qualified'] is False
        assert composition['decision']=='diagnostic_complete_archive_pass_freeze_consumed_whole_model_regression'
        assert P.digest(Path(C.__file__))==composition['script_sha256']
        for path,sha in composition['helper_sha256'].items():assert P.digest(Path(path))==sha
        core=Path(composition['artifact']['path']);core_sha=composition['artifact']['sha256']
        assert P.digest(core)==core_sha
        assert previous['manifest_sha256']==P.M122.MANIFEST_SHA and all(previous['gates'].values())
        prior=json.loads(D.PRIOR.read_text(encoding='utf-8'))
        items=json.loads(P.M122.MANIFEST.read_text(encoding='utf-8'))['items']
        assert len(items)==24 and prior['manifest_sha256']==P.M122.MANIFEST_SHA
        for item in items:
            assert P.M17.sha(item['text'].encode())==item['text_sha256']
            for key in ('document_ids','prompt_ids'):
                assert P.M17.sha(np.asarray(item[key],dtype=np.int32).tobytes())==item[key+'_sha256']
        parent=json.loads(P.M122.TRAINING.read_text(encoding='utf-8'))['checkpoints']['512']
        assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        device=C.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        stage='original_BF16_control_load'
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval();model.config.use_cache=False
        L.D.H.load_centered(model,device,parent['path']);wrappers=[layer.mlp for layer in model.model.layers];donor_top={}
        with torch.inference_mode():
            for arm,enabled in ((ARMS[0],False),(ARMS[1],True)):
                stage=arm;documents=[];prompts=[]
                for index,item in enumerate(items):
                    nats=P.M17.score_doc(model,item['document_ids'],wrappers,enabled,device,start)
                    assert item['source_id']==prior['document_rows'][index]['source_id']
                    assert nats==prior['document_rows'][index]['nats'][arm]
                    documents.append({'source_id':item['source_id'],'category':item['category'],'bytes':item['bytes'],'nats':nats})
                P.M44.set_experts(wrappers,enabled)
                for index,item in enumerate(items):
                    ids=torch.as_tensor(item['prompt_ids'],device=device)[None]
                    top=model(ids,use_cache=False).logits.argmax(-1)[0].cpu()
                    if not enabled:donor_top[item['source_id']]=top
                    matching=int((top==donor_top[item['source_id']]).sum())
                    old=prior['prompt_rows'][index]
                    assert item['source_id']==old['source_id'] and matching==old['matching'][arm]
                    prompts.append({'source_id':item['source_id'],'category':item['category'],
                        'positions':len(item['prompt_ids']),'matching':matching});P.budget(start,device)
                arms[arm]={'document_rows':documents,'prompt_rows':prompts}
                assert arms[arm]==previous['arms'][arm]
                partial();print(json.dumps({'arm':arm,'runtime':P.budget(start,device)}),flush=True)
        del model,wrappers;gc.collect();torch.cuda.empty_cache();stage='diagnostic_complete_archive_load'
        model,wrappers,proposal,load_record=C.load_stored(core,device,core_sha)
        L.D.E.P=P;stage=ARMS[2]
        arms[ARMS[2]]=L.D.E.evaluate(model,wrappers,items,donor_top,ARMS[2],device,start);partial()
        stage='finite_prompt_shortlist_controls'
        approx=(proposal['codes'].float()*proposal['scales'].float()[:,None]).bfloat16()
        L.D.H.KS=(64,);shortlist=[]
        with torch.inference_mode():
            for item in items:
                ids=torch.as_tensor(item['prompt_ids'],device=device)[None]
                hidden=model.model(ids,use_cache=False).last_hidden_state[0]
                shortlist.append({'source_id':item['source_id'],'category':item['category'],
                    **L.D.H.score_item(hidden,model.lm_head.weight,approx)});P.budget(start,device)
        F.ARMS=ARMS;summary=F.summarize(arms)
        gates={'all_original194_and260_control_document_and_prompt_scores_exact':True,
            'complete_candidate_load_from_same_diagnostic_archive':True,
            'pooled_bpb_vs_both':all(summary['pooled'][k]<=.01 for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'category_bpb_vs_both':all(summary[c][k]<=.02 for c in P.CATEGORIES for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'pooled_top1':summary['pooled']['candidate_minus_bf16_e1280_top1']>=-.01,
            'category_top1':all(summary[c]['candidate_minus_bf16_e1280_top1']>=-.02 for c in P.CATEGORIES),
            'prompt_k64_inclusion':all(r['misses']['64']==0 for r in shortlist),
            'prompt_exact_rerank':all(r['rerank_mismatches']['64']==0 for r in shortlist)}
        changes={category:{'candidate_bpb_minus_previous259':summary[category]['arms'][ARMS[2]]['bpb']-previous['summary'][category]['arms'][D.ARMS[2]]['bpb'],
            'candidate_agreement_minus_previous259':summary[category]['arms'][ARMS[2]]['top1']-previous['summary'][category]['arms'][D.ARMS[2]]['top1']}
            for category in ('pooled',*P.CATEGORIES)}
        result={'experiment':'METH-277-diagnostic-complete-I16-private128-consumed-development',
            'composition_result_sha256':args.composition_sha,'artifact_sha256':core_sha,
            'manifest_sha256':P.M122.MANIFEST_SHA,'previous259_development_sha256':PREVIOUS_SHA,
            'original_control_result_sha256':D.PRIOR_SHA,'source_sha256':P.M57.MODEL_SHA,
            'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,
            'load_record':load_record,'arms':arms,'summary':summary,'changes_vs_previous259':changes,
            'bootstrap':F.bootstraps(arms),'shortlist':shortlist,'gates':gates,
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'diagnostic_only':True,'native_promotion_qualified':False,
            'decision':'diagnostic_development_pass_requires_fresh_complete_quality' if all(gates.values()) else 'stop_diagnostic_complete_I16_private128_at_consumed_development',
            'scope':'Consumed121/122 cohort; exact original194/260 controls. No new independent source/generation/task/semantic/native rate/useful-n/DRAM/family evidence. Prior fixed259 semantic/component-cost stops retained.'}
        C.I.A.dump(args.out,result)
        print(json.dumps({key:result[key] for key in ('decision','summary','changes_vs_previous259','gates','runtime')}),flush=True)
    except BaseException as failure:
        partial();C.I.A.dump(args.out.with_suffix('.failure.json'),{'stage':stage,
            'error':type(failure).__name__+': '+str(failure),'seconds':time.monotonic()-start,
            'script_sha256':P.digest(Path(__file__))});raise


if __name__=='__main__':main()
