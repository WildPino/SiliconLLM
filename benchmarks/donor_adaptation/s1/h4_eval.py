#!/usr/bin/env python3
"""Frozen CPU-fp32 H4 evaluator; emits independent preregistered bands only."""
import argparse,hashlib,json,os,sys,time
import numpy as np
import torch
HERE=os.path.dirname(os.path.abspath(__file__)); DENSDIR=os.path.abspath(os.path.join(HERE,"..","density"));ENGDIR=os.path.abspath(os.path.join(HERE,"..","engine"));sys.path[:0]=[DENSDIR,ENGDIR]
import common as C
from e6_generate import PROMPTS,N_NEW
import h4_qat as H4
torch.set_num_threads(int(os.environ.get("D_THREADS","6")))
OUTDIR=os.path.join(HERE,"results","h4");E6REF=os.path.join(ENGDIR,"results","e6","ref.json");E6ENG=os.path.join(ENGDIR,"results","e6","engine.json");EXPECT_IDS_SHA="a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65";LN2=.6931471805599453;CHANCE=4.069819
ANCHORS={"dense_E65":{"bpb":.7675949641196624,"free":160,"tf":160,"mean_rank":1},"QO192_posthoc_fp32":{"bpb":1.8563784310382423,"free":4,"tf":56},"QO96_posthoc_fp32":{"bpb":2.0972750188750418,"free":1,"tf":42},"QO48_posthoc_fp32_E65":{"bpb":2.473430292224694,"free":4,"tf":33,"mean_rank":3476.46875,"median_rank":30.5,"rank_le5":55}}
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()
def greedy(model,p,n):
    ids=list(p)
    with torch.no_grad():
        for _ in range(n):ids.append(int(torch.argmax(model(torch.tensor([ids])).logits[0,-1])))
    return ids
def _read_init():
    p=os.path.join(OUTDIR,"h4_eval_init.json")
    if not os.path.exists(p):raise SystemExit("h4_eval_init.json missing: trained bands cannot be read")
    r=json.load(open(p));assert r.get("stage")=="H4" and r.get("mode")=="factored_init" and r.get("rank")==48 and r.get("model")==C.MODEL_ID and r.get("revision")==C.REVISION and r.get("eval_slice",{}).get("ids_sha256")==EXPECT_IDS_SHA and r.get("factors_sha256");return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--factors");ap.add_argument("--tag");ap.add_argument("--emit-probe",action="store_true");ap.add_argument("--out");a=ap.parse_args();os.makedirs(OUTDIR,exist_ok=True)
    if a.emit_probe and (a.factors or a.tag or a.out): ap.error("--emit-probe takes no factors, tag or out")
    if not a.emit_probe and (a.tag or "").lower()=="init" and not a.factors: ap.error("init requires --factors")
    mode="intact" if not a.factors else ("factored_init" if (a.tag or "").lower()=="init" else "factored_trained")
    canonical=os.path.join(OUTDIR,"h4_eval_init.json" if mode=="factored_init" else "h4_eval_intact.json")
    if mode in ("factored_init","intact"):
        if a.out and os.path.abspath(a.out)!=os.path.abspath(canonical): ap.error("intact/init must use canonical output: "+canonical)
        if os.path.exists(canonical): raise SystemExit("canonical eval already exists; preserve it. Repair: archive it manually with provenance, then rerun explicitly")
        if mode=="intact" and a.tag and a.tag.lower()!="intact": ap.error("intact tag must be intact")
        if mode=="factored_init":
            intact_path=os.path.join(OUTDIR,"h4_eval_intact.json")
            if not os.path.isfile(intact_path):
                raise SystemExit("run passing H4 intact control before freezing init")
            intact=json.load(open(intact_path,encoding="utf-8"))
            if not(intact.get("stage")=="H4" and intact.get("mode")=="intact" and intact.get("model")==C.MODEL_ID and intact.get("revision")==C.REVISION and intact.get("eval_slice",{}).get("ids_sha256")==EXPECT_IDS_SHA and intact.get("intact_control",{}).get("passes") is True):
                raise SystemExit("H4 intact control provenance or result failed")
    elif (a.tag or "").lower() in ("init","intact") or not a.tag:
        ap.error("trained eval requires a distinct --tag, excluding init/intact")
    if mode=="factored_trained":
        out=a.out or os.path.join(OUTDIR,"h4_eval_%s.json"%a.tag.replace(os.sep,"_"))
        if os.path.abspath(out) in (os.path.abspath(os.path.join(OUTDIR,"h4_eval_init.json")),os.path.abspath(os.path.join(OUTDIR,"h4_eval_intact.json"))): ap.error("trained output cannot replace a frozen control")
        if os.path.exists(out):raise SystemExit("trained eval output already exists: "+out)
        trained_meta=os.path.splitext(a.factors)[0]+".json"
        if not os.path.isfile(trained_meta):raise SystemExit("terminal checkpoint metadata missing: "+trained_meta)
        tm=json.load(open(trained_meta,encoding="utf-8"))
        if not(tm.get("stage")=="H4" and tm.get("model")==C.MODEL_ID and tm.get("revision")==C.REVISION and tm.get("rank")==48 and tm.get("terminal_checkpoint") is True and tm.get("status") in ("TIME_CAP","STEPS_COMPLETE") and tm.get("actual_steps",0)>0 and tm.get("applied_steps",0)>0 and tm.get("checkpoint_sha256")==sha256_file(a.factors)):raise SystemExit("not a valid H4 terminal checkpoint")
    if mode=="factored_trained": init=_read_init()
    else: init=None
    factors_sha=sha256_file(a.factors) if a.factors else None
    if init is not None:
        initial=os.path.join(OUTDIR,"h4_factors.npz")
        if not os.path.isfile(initial) or sha256_file(initial)!=init["factors_sha256"]:
            raise SystemExit("init factor hash no longer matches canonical H4 factors")
        if tm.get("origin_factors_sha256")!=init["factors_sha256"]:
            raise SystemExit("trained checkpoint does not originate from the frozen H4 init factors")
    started=time.time();model,tok=C.load_model(dtype=torch.float32);model.eval()
    ref=json.load(open(E6REF))[C.MODEL_ID]; e6=json.load(open(E6ENG));pids=[]
    for i,p in enumerate(PROMPTS):
        q=tok(p)["input_ids"];assert q==e6["prompt_ids"][i];pids.append(q)
    tgts=[x["ids"][-N_NEW:] for x in ref]
    if a.emit_probe:
        with open(os.path.join(OUTDIR,"h4_probe.json"),"x",encoding="utf-8") as f:
            json.dump({"stage":"H4","model":C.MODEL_ID,"revision":C.REVISION,"n_new":N_NEW,"prompt_ids":pids,"target_ids":tgts,"counted":160},f,indent=1)
        return 0
    rank=None
    if a.factors:
        z=np.load(a.factors);layers=H4.bundle_layers(a.factors);assert layers==list(range(28)) and all(z["L%02d.%s.A"%(li,nm)].shape==(1536,48) and z["L%02d.%s.B"%(li,nm)].shape==(48,1536) for li in layers for nm in H4.ORGANS);assert H4.build(model,a.factors,layers,"cpu")==56;rank=48
        if mode=="factored_init" and (os.path.abspath(a.factors)!=os.path.abspath(os.path.join(OUTDIR,"h4_factors.npz"))): raise SystemExit("init must evaluate canonical h4_factors.npz")
    ids,byts,evalmeta=C.get_slice(tok,"heldout",24,512,1234);assert evalmeta["ids_sha256"]==EXPECT_IDS_SHA
    nats=[]
    with torch.no_grad():
        for ch in ids:
            ch=ch[None];lg=model(ch).logits;lp=torch.nn.functional.log_softmax(lg[:,:-1],-1);nats.append(-lp.gather(-1,ch[:,1:].unsqueeze(-1)).squeeze(-1)[0].double())
    bpb=float(torch.stack(nats).sum()/(LN2*float(byts.sum())));free=0;tf=0;ranks=[];per=[];texts=[]
    for i in range(5):
        new=greedy(model,pids[i],N_NEW)[-N_NEW:];texts.append(tok.decode(new));f=sum(x==y for x,y in zip(new,tgts[i]));free+=f;h=0
        with torch.no_grad():lg=model(torch.tensor([pids[i]+tgts[i]])).logits[0].float()
        for k,t in enumerate(tgts[i]):
            row=lg[len(pids[i])-1+k];h+=int(int(torch.argmax(row))==t);ranks.append(int((row>row[t]).sum())+1)
        tf+=h;per.append({"prompt":i,"free":f,"teacher_forced":h})
    rec={"stage":"H4","tag":a.tag or mode,"mode":mode,"factors":a.factors,"factors_sha256":factors_sha,"rank":rank,"model":C.MODEL_ID,"revision":C.REVISION,"eval_slice":evalmeta,"anchors_external_not_step0_equivalence":ANCHORS,"bpb":bpb,"free":free,"counted":160,"teacher_forced":tf,"mean_rank":sum(ranks)/160.,"median_rank":float(np.median(ranks)),"rank_le5":sum(x<=5 for x in ranks),"per_prompt":per,"text":texts,"seconds":time.time()-started}
    if mode=="intact":
        rec["intact_control"]={"passes":tf==160 and free==160 and abs(bpb-ANCHORS["dense_E65"]["bpb"])<=1e-7}
    elif mode=="factored_trained":
        rec["init_eval_sha256"]=sha256_file(os.path.join(OUTDIR,"h4_eval_init.json"));rec["init_factors_sha256"]=init["factors_sha256"]
        rec["bands"]={"TRAINING_HELPS":bpb<init["bpb"] and tf>init["teacher_forced"] and rec["mean_rank"]<init["mean_rank"] and rec["rank_le5"]>init["rank_le5"],"BEATS_QO96":bpb<ANCHORS["QO96_posthoc_fp32"]["bpb"] and tf>42 and free>1,"BEATS_QO192":bpb<ANCHORS["QO192_posthoc_fp32"]["bpb"] and tf>56 and free>4,"GENERATOR_PARTIAL":free>14}
    if a.factors and sha256_file(a.factors)!=factors_sha:raise SystemExit("factors changed during H4 evaluation")
    out=canonical if mode in ("factored_init","intact") else out
    with open(out,"x",encoding="utf-8") as f:json.dump(rec,f,indent=1)
    print("wrote",out,flush=True)
    return 2 if mode=="intact" and not rec["intact_control"]["passes"] else 0
if __name__=="__main__":sys.exit(main())
