#!/usr/bin/env python3
"""Frozen fresh-source prediction of the diagnostic complete I16/private128 archive."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
import meth277_complete_i16_development as D
import meth276_diagnostic_complete_core as C
import meth279_i16_source_answerability as A

P,L,R,F=D.P,D.L,C,D.F
MANIFEST=A.MANIFEST;MANIFEST_SHA=A.MANIFEST_SHA
ANSWERABILITY=P.DOC/'meth279_i16_source_answerability_result.json'
ANSWERABILITY_SHA='98f380821f0b2dc145dac4c75cfa27a056ac1ca233035ce4c02d07f769fdaa4a'
ANNOTATIONS=P.DOC/'meth279_i16_source_answerability_annotations.json'
ANNOTATIONS_SHA='3123f83260d833b505fed5f60b7f68537012a2f7fb4c577adcaca5f54fa5fcd0'
EXPORT=D.COMPOSITION;EXPORT_SHA=D.COMPOSITION_SHA
CORE=P.ROOT/'results/native_expert_scaling/meth276_qwen05b_i16_private128_diagnostic_repair1.safetensors'
CORE_SHA='4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9'
DEVELOPMENT=P.DOC/'meth277_complete_i16_development_result.json'
DEVELOPMENT_SHA='3d997545fc690a5ebb3681e60fb0d3dc190a2a292e94056e5cf0abcb9ed88b35'
ARMS=D.ARMS


def bootstrap(arms):
    rng=np.random.default_rng(280280);draws=rng.integers(0,24,size=(10000,24));out={}
    candidate=arms[ARMS[2]]['document_rows'];bytes_=np.asarray([r['bytes'] for r in candidate])
    for control in ARMS[:2]:
        previous=arms[control]['document_rows'];assert [r['source_id'] for r in candidate]==[r['source_id'] for r in previous]
        delta=np.asarray([r['nats']-p['nats'] for r,p in zip(candidate,previous)])
        values=delta[draws].sum(1)/(np.log(2)*bytes_[draws].sum(1))
        out[control]={'candidate_minus_control_bpb_p05':float(np.quantile(values,.05)),
            'candidate_minus_control_bpb_p95':float(np.quantile(values,.95)),'seed':280280,'draws':10000,'unit':'source','decision_gate':False}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--answerability-sha',required=True);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();assert args.answerability_sha==ANSWERABILITY_SHA
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json')))
    start=time.monotonic();stage='bindings';arms={}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'arms':arms,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        for path,sha in ((MANIFEST,MANIFEST_SHA),(ANSWERABILITY,args.answerability_sha),(EXPORT,EXPORT_SHA),
                         (CORE,CORE_SHA),(DEVELOPMENT,DEVELOPMENT_SHA),(ANNOTATIONS,ANNOTATIONS_SHA),(P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),(P.M122.TRAINING,P.M57.TRAINING_SHA)):
            assert P.digest(path)==sha
        manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));answer=json.loads(ANSWERABILITY.read_text(encoding='utf-8'))
        assert answer['manifest_sha256']==MANIFEST_SHA and answer['answerable_count']==24 and all(answer['gates'].values())
        assert answer['model_outputs_consulted'] is False and answer['annotations_sha256']==ANNOTATIONS_SHA
        development=json.loads(DEVELOPMENT.read_text(encoding='utf-8'))
        assert all(development['gates'].values()) and development['decision']=='diagnostic_development_pass_requires_fresh_complete_quality'
        assert development['artifact_sha256']==CORE_SHA
        items=manifest['items'];assert len(items)==24 and manifest['artifact_sha256']==CORE_SHA
        assert manifest['selected_counts']==dict.fromkeys(P.CATEGORIES,8)
        for item,row in zip(items,answer['rows']):
            assert row['source_id']==item['source_id'] and row['anchor'] in item['excerpt']
            assert P.M17.sha(item['text'].encode())==item['text_sha256']
            for key in ('document_ids','prompt_ids'):assert P.M17.sha(np.asarray(item[key],dtype=np.int32).tobytes())==item[key+'_sha256']
        export=json.loads(EXPORT.read_text(encoding='utf-8'));assert all(export['gates'].values())
        assert export['diagnostic_only'] is True and export['native_promotion_qualified'] is False
        assert P.digest(Path(R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        parent=json.loads(P.M122.TRAINING.read_text(encoding='utf-8'))['checkpoints']['512'];assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True));assert P.digest(source)==P.M57.MODEL_SHA
        device=R.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        stage='original_BF16_control_load'
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval();model.config.use_cache=False
        L.D.H.load_centered(model,device,parent['path']);wrappers=[layer.mlp for layer in model.model.layers];donor_top={}
        with torch.inference_mode():
            for arm,enabled in ((ARMS[0],False),(ARMS[1],True)):
                stage=arm;documents=[];prompts=[]
                for item in items:
                    nats=P.M17.score_doc(model,item['document_ids'],wrappers,enabled,device,start)
                    documents.append({'source_id':item['source_id'],'category':item['category'],'bytes':item['bytes'],'nats':nats})
                P.M44.set_experts(wrappers,enabled)
                for item in items:
                    ids=torch.as_tensor(item['prompt_ids'],device=device)[None];top=model(ids,use_cache=False).logits.argmax(-1)[0].cpu()
                    if not enabled:donor_top[item['source_id']]=top
                    prompts.append({'source_id':item['source_id'],'category':item['category'],'positions':len(item['prompt_ids']),
                        'matching':int((top==donor_top[item['source_id']]).sum())});P.budget(start,device)
                arms[arm]={'document_rows':documents,'prompt_rows':prompts};partial()
                print(json.dumps({'arm':arm,'runtime':P.budget(start,device)}),flush=True)
        del model,wrappers;gc.collect();torch.cuda.empty_cache();stage='complete_stored_candidate_load'
        model,wrappers,proposal,load_record=R.load_stored(CORE,device,CORE_SHA)
        L.D.E.P=P;stage=ARMS[2];arms[ARMS[2]]=L.D.E.evaluate(model,wrappers,items,donor_top,ARMS[2],device,start);partial()
        stage='finite_prompt_shortlist_controls';approx=(proposal['codes'].float()*proposal['scales'].float()[:,None]).bfloat16()
        L.D.H.KS=(64,);shortlist=[]
        with torch.inference_mode():
            for item in items:
                ids=torch.as_tensor(item['prompt_ids'],device=device)[None];hidden=model.model(ids,use_cache=False).last_hidden_state[0]
                shortlist.append({'source_id':item['source_id'],'category':item['category'],
                    **L.D.H.score_item(hidden,model.lm_head.weight,approx)});P.budget(start,device)
        F.ARMS=ARMS;summary=F.summarize(arms)
        gates={'fixed_independent_manifest_answerability_and_actual_artifact':True,'complete_same_archive_loader':True,
            'pooled_bpb_vs_both':all(summary['pooled'][k]<=.01 for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'category_bpb_vs_both':all(summary[c][k]<=.02 for c in P.CATEGORIES for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'pooled_top1':summary['pooled']['candidate_minus_bf16_e1280_top1']>=-.01,
            'category_top1':all(summary[c]['candidate_minus_bf16_e1280_top1']>=-.02 for c in P.CATEGORIES),
            'prompt_k64_inclusion':all(r['misses']['64']==0 for r in shortlist),
            'prompt_exact_rerank':all(r['rerank_mismatches']['64']==0 for r in shortlist)}
        result={'experiment':'METH-280-independent-diagnostic-complete-I16-private128-prediction','manifest_sha256':MANIFEST_SHA,
            'answerability_sha256':args.answerability_sha,'annotations_sha256':ANNOTATIONS_SHA,'development_sha256':DEVELOPMENT_SHA,'export_sha256':EXPORT_SHA,'artifact_sha256':CORE_SHA,
            'source_sha256':P.M57.MODEL_SHA,'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,
            'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,'arms':arms,'summary':summary,'bootstrap':bootstrap(arms),
            'shortlist':shortlist,'gates':gates,'load_record':load_record,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),
            'diagnostic_only':True,'native_promotion_qualified':False,
            'decision':'fresh_complete_I16_prediction_pass_freeze_generation_tasks' if all(gates.values()) else 'fresh_complete_I16_prediction_fail_close_fixed_candidate',
            'scope':'One new24-source prediction screen on unchanged actual276 diagnostic archive. New project-heldout IDs,not a new corpus. No generation/task/semantic/native rate/useful n/DRAM/family promotion;prior259 semantic/component-cost failures retained.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        partial();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
