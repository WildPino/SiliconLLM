#!/usr/bin/env python3
"""Fixed complete archive and two original controls on the full PIQA regression set."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import meth265_full_prefix_generation as E

P,L,R,Q=E.P,E.L,E.R,E.Q
T=P.M122.M21
S=P.M122.M90
GENERATION=P.DOC/'meth265_full_prefix_generation_result.json'
ARMS=E.ARMS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--generation-sha',required=True)
    ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists();start=time.monotonic();stage='bindings';arms={};gates={};task_manifest=None
    def save():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'arms':arms,
            'gates':gates,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        assert P.digest(GENERATION)==args.generation_sha
        gen=json.loads(GENERATION.read_text(encoding='utf-8'))
        assert gen['decision']=='full_prefix_generation_health_pass_freeze_task_and_blind' and all(gen['gates'].values())
        assert gen['artifact_sha256']==Q.D.CORE_SHA and gen['manifest_sha256']==Q.MANIFEST_SHA
        for path,sha in ((Q.D.EXPORT,Q.D.EXPORT_SHA),(Q.D.CORE,Q.D.CORE_SHA),
                         (P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),(P.M122.TRAINING,P.M57.TRAINING_SHA)):
            assert P.digest(path)==sha
        export=json.loads(Q.D.EXPORT.read_text(encoding='utf-8'));assert P.digest(Path(R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        parent=json.loads(P.M122.TRAINING.read_text(encoding='utf-8'))['checkpoints']['512']
        assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        tokenizer=AutoTokenizer.from_pretrained(P.M42.MODEL,revision=P.M42.REV,local_files_only=True)
        assert P.M15.M13.C.tok_fingerprint(tokenizer)==P.M15.M13.TOK_FP
        task_manifest,items=T.bind_data(tokenizer);assert len(items)==1838
        device=R.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=70*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval();model.config.use_cache=False
        L.D.H.load_centered(model,device,parent['path']);wrappers=[layer.mlp for layer in model.model.layers]
        for arm,enabled in ((ARMS[0],False),(ARMS[1],True),(ARMS[2],True)):
            if arm==ARMS[2]:
                del model,wrappers;gc.collect();torch.cuda.empty_cache()
                model,wrappers,proposal,load_record=R.load_stored(Q.D.CORE,device,Q.D.CORE_SHA)
            P.M44.set_experts(wrappers,enabled);stage=arm+'_piqa';rows=[];arms[arm]=rows
            for item in items:
                scores=[T.score_option(model,item['prefix'],suffix,device) for suffix in item['suffixes']]
                rows.append({'index':item['index'],'label':item['label'],
                    'choice_mean':0 if scores[0][1]<=scores[1][1] else 1,
                    'choice_total':0 if scores[0][0]<=scores[1][0] else 1,
                    'options':[{'total_nll':total,'mean_nll':mean,'tokens':len(suffix)}
                        for (total,mean),suffix in zip(scores,item['suffixes'])]})
                P.budget(start,device)
                if len(rows)%128==0:
                    save();print(json.dumps({'arm':arm,'items':len(rows),'total':1838,'runtime':P.budget(start,device)}),flush=True)
            assert len(rows)==1838;save()
        summary={}
        for control in ARMS[:2]:
            paired=[{'label':a['label'],'donor':a,'student':b} for a,b in zip(arms[control],arms[ARMS[2]])]
            assert all(a['index']==b['index'] and a['label']==b['label'] for a,b in zip(arms[control],arms[ARMS[2]]))
            summary[control]=S.task_summary(paired)
            gates[control+'_accuracy']=summary[control]['accuracy_delta']>=-.02
            gates[control+'_paired_lower95']=summary[control]['paired_bootstrap_lower95']>=-.05
        result={'experiment':'METH-266-complete-core-PIQA-regression',
            'generation_sha256':args.generation_sha,'artifact_sha256':Q.D.CORE_SHA,
            'source_sha256':P.M57.MODEL_SHA,'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,
            'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,'task_manifest':task_manifest,
            'arms':arms,'summary':summary,'gates':gates,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),
            'task_helper_sha256':{str(Path(module.__file__)):P.digest(Path(module.__file__)) for module in (T,S)},
            'decision':'complete_core_task_pass_blind_semantics_pending' if all(gates.values()) else 'complete_core_task_fail_close_fixed_candidate',
            'scope':'Repeated full1838-item PIQA regression,no fresh independent task/general capability proof. Same saved archive/full head,no production-cache/native accepted-rate/large-RAM n/other-family proof.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({key:result[key] for key in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        save();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,
            'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
