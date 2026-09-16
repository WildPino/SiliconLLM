#!/usr/bin/env python3
"""Self-contained H4 T4 trainer.  CPU fp32 h4_eval.py alone adjudicates outcomes."""
import argparse,hashlib,json,math,os,sys,time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
D_GRID=[.3,.4,.5,.6,.7,.8,.9,1.,1.1,1.2]; ORGANS=("q_proj","o_proj"); G_H4E_WINDOW=25
MODEL_ID="Qwen/Qwen2.5-1.5B";REVISION="8faed761d45a263340a0528343f099c05c9a4323"
CALIB_SHA="c5509846cdc3aa44e45e77895b59a4638c49eb03790030d851e7bf1357ca4c0c"
TRAIN_IDS_SHA="9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da"
EVAL_IDS_SHA="a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
def log(x): print(x,flush=True)
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def checked_inputs(a):
    """Reject partial, smoke or changed input before loading donor/GPU."""
    if a.smoke:raise SystemExit("H4 Stage A refuses --smoke")
    root=os.path.dirname(os.path.abspath(a.factors))
    fac_meta=os.path.join(root,"h4_factors.json") if os.path.basename(a.factors)=="h4_factors.npz" else os.path.splitext(a.factors)[0]+".json"
    train_meta=os.path.splitext(a.train)[0]+".json"
    for path in (fac_meta,train_meta):
        if not os.path.isfile(path):raise SystemExit("missing input metadata: "+path)
    fm=json.load(open(fac_meta,encoding="utf-8"));tm=json.load(open(train_meta,encoding="utf-8"));pr=json.load(open(a.probe,encoding="utf-8"))
    if fm.get("stage")=="H4" and "G_H4a" in fm:
        valid_factor=(fm.get("G_H4a",{}).get("fires") is True and fm.get("smoke") is False and fm.get("calib_slice",{}).get("ids_sha256")==CALIB_SHA and fm.get("calib_slice",{}).get("seed")==42424 and fm.get("calib_slice",{}).get("n_seq")==32)
    else:
        valid_factor=(fm.get("stage")=="H4" and fm.get("status") in ("TIME_CAP","STEPS_COMPLETE") and fm.get("rank")==48 and fm.get("actual_steps",0)>0 and fm.get("terminal_checkpoint") is True and fm.get("checkpoint_sha256")==sha256_file(a.factors))
    if not(valid_factor and fm.get("model")==MODEL_ID and fm.get("revision")==REVISION):raise SystemExit("H4 factor metadata/revision/calibration mismatch")
    with np.load(a.factors) as z:
        layers=sorted({int(k[1:3]) for k in z.files if k.startswith("L")})
        if layers!=list(range(28)) or len(z.files)!=280:raise SystemExit("H4 requires exactly 56 organs / 280 arrays")
        for li in layers:
            for nm in ORGANS:
                p="L%02d.%s."%(li,nm)
                if any(z[p+k].shape!=shape for k,shape in (("A",(1536,48)),("s",(48,)),("B",(48,1536)),("rms_in",(1536,)),("rms_A",(48,)))):raise SystemExit("invalid rank48 factor shape: "+p)
    if tm.get("model")!=MODEL_ID or tm.get("revision")!=REVISION or tm.get("smoke") is not False or tm.get("part")!="calib" or tm.get("seed")!=90011 or tm.get("seq_len")!=512 or tm.get("n_seq")!=31250 or tm.get("ids_sha256")!=TRAIN_IDS_SHA or tm.get("eval_slice_sha256_NOT_THIS")!=EVAL_IDS_SHA:raise SystemExit("H4 train stream metadata mismatch")
    if pr.get("model")!=MODEL_ID or pr.get("revision")!=REVISION or pr.get("n_new")!=32 or pr.get("counted")!=160 or len(pr.get("prompt_ids",[]))!=5 or len(pr.get("target_ids",[]))!=5 or any(len(x)!=32 for x in pr["target_ids"]):raise SystemExit("H4 probe metadata/shape mismatch")
    with np.load(a.train) as z:
        ids=z["ids"]
        if ids.shape!=(31250,512) or hashlib.sha256(ids.tobytes()).hexdigest()!=TRAIN_IDS_SHA:raise SystemExit("H4 train IDs mismatch")
    manifest=os.path.join(root,"MANIFEST.json")
    if os.path.isfile(manifest):
        files=json.load(open(manifest,encoding="utf-8")).get("files",{})
        paths=(a.train,train_meta,a.probe,__file__)
        if os.path.basename(a.factors)=="h4_factors.npz":paths=(a.factors,fac_meta)+paths
        for path in paths:
            name=os.path.basename(path)
            if files.get(name,{}).get("sha256")!=sha256_file(path):raise SystemExit("MANIFEST input hash mismatch: "+name)
        if os.path.basename(a.factors)!="h4_factors.npz" and fm.get("origin_factors_sha256")!=files.get("h4_factors.npz",{}).get("sha256"):
            raise SystemExit("resume checkpoint is not linked to the packaged H4 init factors")
    current_sha=sha256_file(a.factors)
    origin_sha=current_sha if os.path.basename(a.factors)=="h4_factors.npz" else fm.get("origin_factors_sha256")
    if not origin_sha:raise SystemExit("resume checkpoint lacks H4 init factor provenance")
    return layers,ids,pr,{"factors_sha256":current_sha,"origin_factors_sha256":origin_sha,"train_sha256":sha256_file(a.train),"probe_sha256":sha256_file(a.probe)}
def applied_steps(opt):
    return max([int((s["step"].item() if hasattr(s["step"],"item") else s["step"])) for p,s in opt.state.items() if "step" in s] or [0])
def r3_actsearch(w,act_rms):
    m=w.abs().mean(1,keepdim=True); ww=act_rms.view(1,-1); best=None
    for f in D_GRID:
        q=torch.sign(w)*(w.abs()>f*m).to(w.dtype); kept=(q!=0).to(w.dtype); a=((w*q*ww*ww).sum(1,keepdim=True)/(kept*ww*ww).sum(1,keepdim=True).clamp_min(1e-12)).clamp_min(1e-5); err=((w-a*q)**2*ww*ww).sum(1,keepdim=True)
        if best is None: best=(err,q,a)
        else:
            take=err<best[0]; best=(torch.where(take,err,best[0]),torch.where(take,q,best[1]),torch.where(take,a,best[2]))
    return best[1],best[2]
def ste(w,rms):
    with torch.no_grad(): q,a=r3_actsearch(w.detach().float(),rms); q=q*a
    return w+(q-w).detach()
class TernaryLowRank(nn.Module):
    def __init__(self,A,s,B,rms_in,rms_A,bias):
        super().__init__(); self.A=nn.Parameter(A.float()); self.s=nn.Parameter(s.float()); self.B=nn.Parameter(B.float()); self.register_buffer("rms_in",rms_in.float()); self.register_buffer("rms_A",rms_A.float()); self._ck=self._cv=None
        if bias is None:self.bias=None
        else:self.register_buffer("bias_buf",bias.clone());self.bias="buf"
    def _quant(self):
        key=(self.A._version,self.B._version,self.s._version)
        if self._ck==key and not torch.is_grad_enabled(): return self._cv
        Bt=ste(self.B,self.rms_in); At=ste(self.A,self.rms_A*self.s.detach().abs().clamp_min(1e-8))
        if not torch.is_grad_enabled(): self._ck,self._cv=key,(At,Bt)
        return At,Bt
    def forward(self,x):
        with torch.autocast(device_type="cuda" if x.is_cuda else "cpu",enabled=False): At,Bt=self._quant()
        h=F.linear(x,Bt.to(x.dtype))*self.s.to(x.dtype); y=F.linear(h,At.to(x.dtype)); return y if self.bias is None else y+self.bias_buf.to(y.dtype)
def bundle_layers(bundle): return sorted({int(k[1:3]) for k in np.load(bundle).files if k.startswith("L")})
def build(model,bundle,layer_ids,device):
    z=np.load(bundle); have=set(bundle_layers(bundle)); n=0
    for li in layer_ids:
        if li not in have:continue
        attn=model.model.layers[li].self_attn
        for nm in ORGANS:
            p="L%02d.%s"%(li,nm); old=getattr(attn,nm); setattr(attn,nm,TernaryLowRank(torch.from_numpy(z[p+".A"]),torch.from_numpy(z[p+".s"]),torch.from_numpy(z[p+".B"]),torch.from_numpy(z[p+".rms_in"]),torch.from_numpy(z[p+".rms_A"]),None if old.bias is None else old.bias.data.detach().clone()).to(device));n+=1
    return n
@torch.no_grad()
def tf_count(model,pids,tgts,device):
    model.eval(); n=0
    for p,t in zip(pids,tgts):
        lg=model(torch.tensor([p+t],device=device)).logits[0].float()
        n+=sum(int(int(torch.argmax(lg[len(p)-1+k]))==x) for k,x in enumerate(t))
    model.train();return n
def save(model,layers,a,hist,tf0,nonfinite,elapsed,status,gate,declined,actual_steps,applied_count,input_hashes):
    st={}
    for li in layers:
        for nm in ORGANS:
            m=getattr(model.model.layers[li].self_attn,nm);p="L%02d.%s"%(li,nm)
            for x in ("A","s","B","rms_in","rms_A"):st[p+"."+x]=getattr(m,x).detach().float().cpu().numpy()
    if os.path.exists(a.out) or os.path.exists(os.path.splitext(a.out)[0]+".json"):raise SystemExit("terminal output exists; refusing overwrite")
    np.savez(a.out,**st)
    rec={"stage":"H4","model":MODEL_ID,"revision":REVISION,"rank":48,"status":status,"terminal_checkpoint":True,"checkpoint_sha256":sha256_file(a.out),"complete":status=="STEPS_COMPLETE","actual_steps":actual_steps,"applied_steps":applied_count,"steps_requested":a.steps,"bs":a.bs,"accum":a.accum,"lr":a.lr,"seed":a.seed,"resumed_from":a.factors,"origin_factors_sha256":input_hashes["origin_factors_sha256"],"adam_state_restarted":True,"seconds":elapsed,"nonfinite_microbatches":nonfinite,"tf_fp16_gpu_step0":tf0,"history":hist,"G_H4e":gate,"scaler_declined_before_first_applied":declined,"input_hashes":input_hashes,"adjudication":"h4_eval.py CPU fp32 only; no best-checkpoint selection"}
    with open(os.path.splitext(a.out)[0]+".json","x",encoding="utf-8") as f:json.dump(rec,f,indent=1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--factors",required=True);ap.add_argument("--train",required=True);ap.add_argument("--probe",required=True);ap.add_argument("--out",default="h4_trained.npz");ap.add_argument("--steps",type=int,default=int(os.environ.get("H4_STEPS","4000")));ap.add_argument("--bs",type=int,default=int(os.environ.get("H4_BS","2")));ap.add_argument("--accum",type=int,default=int(os.environ.get("H4_ACCUM","8")));ap.add_argument("--lr",type=float,default=float(os.environ.get("H4_LR","2e-4")));ap.add_argument("--every",type=int,default=int(os.environ.get("H4_EVERY","250")));ap.add_argument("--max-hours",type=float,default=2.8);ap.add_argument("--seed",type=int,default=1717);ap.add_argument("--smoke",action="store_true",default=os.environ.get("H4_SMOKE")=="1");a=ap.parse_args()
    if a.steps<=0 or a.every<=0 or a.accum<=0 or a.bs<=0 or not (0<a.max_hours<=2.8):raise SystemExit("invalid H4 training CLI or wall cap above 2.8 h")
    if os.path.abspath(a.out)==os.path.abspath(a.factors):raise SystemExit("output cannot overwrite resume factors")
    if os.path.exists(a.out) or os.path.exists(os.path.splitext(a.out)[0]+".json"):raise SystemExit("terminal output exists; refusing overwrite")
    layers,ids_np,probe,input_hashes=checked_inputs(a)
    torch.set_grad_enabled(True);assert torch.is_grad_enabled(),"AUTOGRAD DISABLED"
    dev=os.environ.get("H4_DEV","cuda" if torch.cuda.is_available() else "cpu");cuda=dev.startswith("cuda")
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REVISION,dtype=torch.float16 if cuda else torch.float32,attn_implementation="sdpa")
    assert getattr(model.config,"_attn_implementation",None)=="sdpa","sdpa required";model.to(dev)
    for p in model.parameters():p.requires_grad_(False)
    if model.config.num_hidden_layers!=28 or model.config.hidden_size!=1536:raise SystemExit("donor architecture mismatch")
    nmod=build(model,a.factors,layers,dev);params=[p for p in model.parameters() if p.requires_grad];assert nmod==56 and sum(p.numel() for p in params)==8_260_224
    model.gradient_checkpointing_enable();model.enable_input_require_grads();model.train();ids=torch.from_numpy(ids_np).long();pids,tgts=probe["prompt_ids"],probe["target_ids"]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.,betas=(.9,.95));sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.steps,pct_start=.05,anneal_strategy="cos");scaler=torch.amp.GradScaler("cuda",enabled=cuda);tf0=tf_count(model,pids,tgts,dev);hist=[{"step":0,"tf_fp16_gpu":tf0,"loss":None,"seconds":0.}];watch=[("A",getattr(model.model.layers[layers[0]].self_attn.q_proj,"A")),("B",getattr(model.model.layers[layers[0]].self_attn.o_proj,"B")),("s",getattr(model.model.layers[layers[-1]].self_attn.q_proj,"s"))];snap=[x.detach().clone() for _,x in watch];gate=None;declined=nonfinite=0;rng=np.random.default_rng(a.seed);started=time.time();runloss=nb=0
    last_step=0
    for step in range(1,a.steps+1):
        last_step=step
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            batch=ids[torch.from_numpy(rng.integers(0,ids.shape[0],size=a.bs))].to(dev)
            with torch.autocast("cuda",dtype=torch.float16,enabled=cuda):loss=model(batch,labels=batch).loss/a.accum
            if not torch.isfinite(loss):nonfinite+=1;continue
            scaler.scale(loss).backward();runloss+=float(loss.detach())*a.accum;nb+=1
        scaler.unscale_(opt);gnorm=float(torch.nn.utils.clip_grad_norm_(params,1.));before=applied_steps(opt);scale=scaler.get_scale();scaler.step(opt);scaler.update();sched.step();applied=applied_steps(opt)>before
        if gate is None:
            if applied:
                moved={k:float((v.detach()-s).abs().max()) for (k,v),s in zip(watch,snap)};assert all(x>0 for x in moved.values()),"G-H4e master did not move after applied update";gate={"moved":moved,"first_applied_step":step,"declined_before":declined,"scale_at_first_applied":scale,"grad_norm_at_first_applied":gnorm}
            else:
                declined+=1
                if declined>=G_H4E_WINDOW:raise SystemExit("G-H4e CANNOT BE READ: GradScaler declined all first updates")
        elapsed=time.time()-started
        if step%a.every==0 or step==a.steps:
            tf=tf_count(model,pids,tgts,dev);hist.append({"step":step,"tf_fp16_gpu":tf,"loss":runloss/max(1,nb),"seconds":elapsed});runloss=nb=0
        if elapsed>a.max_hours*3600:break
    status="STEPS_COMPLETE" if last_step==a.steps else "TIME_CAP"
    save(model,layers,a,hist,tf0,nonfinite,time.time()-started,status,gate,declined,last_step,applied_steps(opt),input_hashes)
    log("terminal checkpoint step %d (%s); CPU fp32 h4_eval.py alone adjudicates H4"%(last_step,status));return 0
if __name__=="__main__":sys.exit(main())
