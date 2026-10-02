#!/usr/bin/env python3
"""Consumed-prefix core/route diagnosis with nondeployable source-route replay."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as TF
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import meth266_complete_core_piqa as E

P,L,R,Q=E.P,E.L,E.R,E.Q
ARMS=E.ARMS
GEN=E.GENERATION
GEN_SHA='f3581cefc4a7d758b141fc4b4fa62ca45f43d00fe6cc00a10af57895b4a5ec93'
TASK=P.DOC/'meth266_complete_core_piqa_result.json'
TASK_SHA='da571182fd1190c5d3cf634c634d63810a1f33c3f302a6848f13085202c0385e'
SEM=P.DOC/'meth267_unblinded_semantic_result.json'
SEM_SHA='bbcc22e31d599f1a6f4eb0212e4836f804161603504699053747cd0bf0f2d790'
GRID=(0,1,2,4,8,16,32,64,96,127)


class Trace:
    def __init__(self,wrappers):
        self.wrappers=wrappers;self.rows={};self.replay=None;self.original=[];self.handles=[]
        for li,w in enumerate(wrappers):
            routes,child=w.routes,w.child_route;self.original.append((routes,child))
            def pre(module,inputs,li=li):
                self.rows.setdefault(li,{})['x']=inputs[0].reshape(-1,896)[-1:].detach().clone()
            def routed(flat,li=li,fn=routes):
                if self.replay is None:p,s=fn(flat)
                else:
                    p,s=self.replay.rows[li]['parents'],self.replay.rows[li]['scores']
                    assert p.shape==(len(flat),4) and s.shape==p.shape
                self.rows.setdefault(li,{})['parents']=p.detach().clone()
                self.rows[li]['scores']=s.detach().clone();return p,s
            def childed(flat,parents,li=li,fn=child):
                if self.replay is None:c=fn(flat,parents)
                else:
                    assert torch.equal(parents,self.replay.rows[li]['parents'])
                    c=self.replay.rows[li]['children'];assert c.shape==parents.shape
                self.rows[li]['children']=c.detach().clone();return c
            w.routes=routed;w.child_route=childed;self.handles.append(w.register_forward_pre_hook(pre))

    def forward(self,model,ids):
        self.rows={};output=model.model(ids,use_cache=False)
        hidden=output.last_hidden_state[0,-1:]
        logits=TF.linear(hidden,model.lm_head.weight).float()[0]
        assert len(self.rows)==24 and torch.isfinite(logits).all()
        return int(logits.argmax()),logits


def error(a,b):
    a=a.float();b=b.float();return float((a-b).square().sum()/a.square().sum().clamp_min(1e-30))


def schedule(items,gen):
    byarm={a:{r['source_id']:r for r in gen['arms'][a]} for a in ARMS[1:]};cases=[]
    for item in items:
        sid=item['source_id'];a=byarm[ARMS[1]][sid]['continuation_ids'];b=byarm[ARMS[2]][sid]['continuation_ids']
        first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
        if first is None and len(a)!=len(b):first=min(len(a),len(b))
        for arm,tokens in ((ARMS[1],a),(ARMS[2],b)):
            steps=set(GRID)
            if first is not None:steps.update((max(0,first-1),first,first+1))
            for step in sorted(s for s in steps if s<len(tokens)):
                cases.append({'source_id':sid,'category':item['category'],'trajectory':arm,'step':step,
                    'first_generation_divergence':first,'own_recorded_choice':tokens[step],
                    'prefix_ids':item['prompt_ids']+tokens[:step]})
    return cases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists();start=time.monotonic();stage='bindings';rows=[];gates={}
    def save():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'rows':rows,'gates':gates,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        for path,sha in ((GEN,GEN_SHA),(TASK,TASK_SHA),(SEM,SEM_SHA),(Q.MANIFEST,Q.MANIFEST_SHA),
                         (Q.D.CORE,Q.D.CORE_SHA),(Q.D.EXPORT,Q.D.EXPORT_SHA),
                         (P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),(P.M122.TRAINING,P.M57.TRAINING_SHA)):
            assert P.digest(path)==sha
        gen=json.loads(GEN.read_text(encoding='utf-8'));task=json.loads(TASK.read_text(encoding='utf-8'))
        sem=json.loads(SEM.read_text(encoding='utf-8'))
        assert all(gen['gates'].values()) and all(task['gates'].values())
        assert sem['decision']=='anonymous_semantic_fail_close_fixed_candidate' and not all(sem['gates'].values())
        export=json.loads(Q.D.EXPORT.read_text(encoding='utf-8'));assert P.digest(Path(R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        parent=json.loads(P.M122.TRAINING.read_text(encoding='utf-8'))['checkpoints']['512']
        assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        tok=AutoTokenizer.from_pretrained(P.M42.MODEL,revision=P.M42.REV,local_files_only=True)
        assert P.M15.M13.C.tok_fingerprint(tok)==P.M15.M13.TOK_FP
        items=json.loads(Q.MANIFEST.read_text(encoding='utf-8'))['items'];assert len(items)==24
        cases=schedule(items,gen);assert cases and len(cases)<=24*2*13
        device=R.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        reference=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval();reference.config.use_cache=False
        L.D.H.load_centered(reference,device,parent['path']);rw=[layer.mlp for layer in reference.model.layers]
        candidate,cw,proposal,load_record=R.load_stored(Q.D.CORE,device,Q.D.CORE_SHA)
        P.M44.set_experts(rw,True);P.M44.set_experts(cw,True);rt=Trace(rw);ct=Trace(cw)
        stage='fixed_consumed_prefix_trace_and_source_route_replay'
        with torch.inference_mode():
            for index,case in enumerate(cases):
                ids=torch.as_tensor(case['prefix_ids'],device=device)[None]
                ref_choice,ref_logits=rt.forward(reference,ids)
                ct.replay=None;choice,logits=ct.forward(candidate,ids)
                own=ref_choice if case['trajectory']==ARMS[1] else choice
                assert own==case['own_recorded_choice'],('original265_choice_parity',case['source_id'],case['step'],own,case['own_recorded_choice'])
                normal_rows=ct.rows;layer_rows=[]
                for li,(wr,wc) in enumerate(zip(rw,cw)):
                    a=rt.rows[li];b=normal_rows[li];x=a['x']
                    rp,rs=rt.original[li][0](x);cp,cs=ct.original[li][0](x)
                    rc=rt.original[li][1](x,rp);cc=ct.original[li][1](x,cp)
                    assert torch.equal(rp,cp) and torch.equal(rs,cs) and torch.equal(rc,cc)
                    assert torch.equal(wr.a[rc].bfloat16(),wc.a[cc//10])
                    assert torch.equal(wr.b[rc].bfloat16(),wc.b[wc.leaf_map[cc].long()])
                    base=wr.base(x);approx=wc.base(x);assert torch.isfinite(base).all() and torch.isfinite(approx).all()
                    layer_rows.append({'layer':li,'input_relative_squared_error':error(a['x'],b['x']),
                        'same_reference_input_FFN_relative_squared_error':error(base,approx),
                        'reference_parents':a['parents'][-1].tolist(),'candidate_parents':b['parents'][-1].tolist(),
                        'reference_children':a['children'][-1].tolist(),'candidate_children':b['children'][-1].tolist(),
                        'parent_order_differs':not torch.equal(a['parents'][-1],b['parents'][-1]),
                        'child_order_differs':not torch.equal(a['children'][-1],b['children'][-1]),
                        'same_input_route_and_selected_BF16_values_exact':True})
                ct.replay=rt;locked_choice,locked_logits=ct.forward(candidate,ids);ct.replay=None
                assert all(torch.equal(ct.rows[li]['parents'],rt.rows[li]['parents'])
                    and torch.equal(ct.rows[li]['scores'],rt.rows[li]['scores'])
                    and torch.equal(ct.rows[li]['children'],rt.rows[li]['children']) for li in range(24))
                rows.append({k:v for k,v in case.items() if k!='prefix_ids'} | {
                    'prefix_ids_sha256':P.M17.sha(np.asarray(case['prefix_ids'],dtype=np.int32).tobytes()),
                    'prefix_tokens':len(case['prefix_ids']),'reference_choice':ref_choice,'candidate_choice':choice,
                    'source_route_replay_choice':locked_choice,'candidate_logit_relative_squared_error':error(ref_logits,logits),
                    'replay_logit_relative_squared_error':error(ref_logits,locked_logits),
                    'candidate_reference_choice_deficit':float(logits.max()-logits[ref_choice]),
                    'replay_reference_choice_deficit':float(locked_logits.max()-locked_logits[ref_choice]),'layers':layer_rows})
                P.budget(start,device)
                if (index+1)%24==0:
                    save();print(json.dumps({'cases':index+1,'total':len(cases),'runtime':P.budget(start,device)}),flush=True)
        gates={'all_original265_own_trajectory_next_choices_exact':True,
            'all_same_input_routes_and_selected_BF16_dictionary_values_exact':True,
            'all_replay_parent_scores_and_children_exact':True,'all_logits_and_local_FFN_values_finite':True}
        summary={}
        for category in ('pooled',*P.CATEGORIES):
            group=[r for r in rows if category=='pooled' or r['category']==category]
            n=len(group);assert n
            summary[category]={'cases':n,'reference_candidate_choices_differ':sum(r['reference_choice']!=r['candidate_choice'] for r in group),
                'replay_choices_differ':sum(r['reference_choice']!=r['source_route_replay_choice'] for r in group),
                'replay_recovers_reference_choice':sum(r['candidate_choice']!=r['reference_choice']==r['source_route_replay_choice'] for r in group),
                'replay_loses_previously_matching_reference_choice':sum(r['candidate_choice']==r['reference_choice']!=r['source_route_replay_choice'] for r in group),
                'cases_with_any_parent_order_change':sum(any(q['parent_order_differs'] for q in r['layers']) for r in group),
                'cases_with_any_child_order_change':sum(any(q['child_order_differs'] for q in r['layers']) for r in group),
                'mean_candidate_logit_relative_squared_error':float(np.mean([r['candidate_logit_relative_squared_error'] for r in group])),
                'mean_replay_logit_relative_squared_error':float(np.mean([r['replay_logit_relative_squared_error'] for r in group])),
                'mean_same_input_FFN_relative_squared_error':float(np.mean([q['same_reference_input_FFN_relative_squared_error'] for r in group for q in r['layers']]))}
        result={'experiment':'METH-268-fixed-consumed-generation-core-route-replay-diagnostic',
            'generation_sha256':GEN_SHA,'task_sha256':TASK_SHA,'semantic_stop_sha256':SEM_SHA,
            'artifact_sha256':Q.D.CORE_SHA,'manifest_sha256':Q.MANIFEST_SHA,'grid':GRID,
            'sample_rule':'Both fixed BF16 E1280/saved265 trajectories,grid plus first ID-divergence-1/0/+1,valid steps only',
            'rows':rows,'summary':summary,'gates':gates,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),
            'decision':'consumed_route_replay_diagnosis_complete_fixed_candidate_stays_closed',
            'scope':'No fit/new artifact/free generation/reclassified semantic gate. Nondeployable replay injects all-position reference parentIDs,scores,childIDs; conditional inputs/core remain candidate. Finite consumed diagnosis,no independent quality/native rate/large-n/family proof.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({key:result[key] for key in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        save();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),
            'completed_cases':len(rows),'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
