#!/usr/bin/env python3
"""CPU H4 bundle controls G-H4b/c/d/f/g plus the H4 interim-checkpoint contract."""
import hashlib,json,os,sys,tempfile,traceback
from types import SimpleNamespace
import numpy as np
import torch
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[os.path.abspath(os.path.join(HERE,"..","ternary"))]
import t2_rules as T2
torch.set_grad_enabled(True)
import h4_qat as H4
torch.set_num_threads(int(os.environ.get("D_THREADS","6")));BUNDLE=os.environ.get("H4_BUNDLE",os.path.join(HERE,"results","h4","h4_factors_smoke.npz"));OUT=os.environ.get("H4_SELFTEST_OUT",os.path.join(HERE,"results","h4","h4_selftest.json"))
def main():
    if os.path.exists(OUT):raise SystemExit("H4 self-test record exists; archive with provenance or set H4_SELFTEST_OUT")
    rec={"stage":"H4","bundle":BUNDLE,"bundle_sha256":None,"smoke":"_smoke" in os.path.basename(BUNDLE),"rank":48,"passes":False}
    try:
        rec.update(run_checks(rec))
    except Exception as exc:
        rec["error"]="%s: %s"%(type(exc).__name__,exc)
        rec["traceback"]=traceback.format_exc()
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    with open(OUT,"x",encoding="utf-8") as f:json.dump(rec,f,indent=1)
    print("wrote",OUT,"PASS" if rec["passes"] else "FAIL",flush=True)
    return 0 if rec["passes"] else 2

def run_checks(rec):
    h=hashlib.sha256()
    with open(BUNDLE,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    rec["bundle_sha256"]=h.hexdigest()
    meta_path=os.path.splitext(BUNDLE)[0]+".json"
    if not os.path.isfile(meta_path):raise ValueError("factor metadata missing: "+meta_path)
    fm=json.load(open(meta_path,encoding="utf-8"))
    rec["factor_meta_sha256"]=hashlib.sha256(open(meta_path,"rb").read()).hexdigest()
    if not(fm.get("stage")=="H4" and fm.get("rank")==48 and fm.get("smoke")==rec["smoke"] and fm.get("model")==H4.MODEL_ID and fm.get("revision")==H4.REVISION):raise ValueError("H4 factor metadata mismatch")
    z=np.load(BUNDLE);organs=sorted({k.rsplit(".",1)[0] for k in z.files});smoke=rec["smoke"];full=len(organs)==56
    rec.update({"organs":len(organs),"full_bundle":full})
    expected={"L%02d.%s"%(li,nm) for li in range(28) for nm in H4.ORGANS}
    if any(p not in expected for p in organs) or (full and set(organs)!=expected):raise ValueError("unexpected or missing H4 organs")
    if not smoke and not full:raise ValueError("H4 full bundle must contain 56 q/o organs")
    if not organs:raise ValueError("empty factor bundle")
    shape_ok=all(z[p+".A"].shape==(1536,48) and z[p+".B"].shape==(48,1536) and z[p+".s"].shape==(48,) and z[p+".rms_in"].shape==(1536,) and z[p+".rms_A"].shape==(48,) for p in organs)
    rec["shape_assert"]=shape_ok
    if not shape_ok:raise ValueError("H4 rank must be 48 from factor shapes")
    bad=[];worst=0.
    for p in organs:
        for fac,rms in (("A","rms_A"),("B","rms_in")):
            q0,a0=T2.r3_actsearch(torch.from_numpy(z[p+"."+fac]),torch.from_numpy(z[p+"."+rms]));q1,a1=H4.r3_actsearch(torch.from_numpy(z[p+"."+fac]),torch.from_numpy(z[p+"."+rms]));
            if not(torch.equal(q0,q1) and torch.equal(a0,a1)):bad.append(p+"."+fac)
        A,s,B,ri,ra=(torch.from_numpy(z[p+"."+x]) for x in ("A","s","B","rms_in","rms_A"));m=H4.TernaryLowRank(A,s,B,ri,ra,None)
        with torch.no_grad():got=m(torch.eye(B.shape[1])).T
        qa,aa=T2.r3_actsearch(A,ra);qb,ab=T2.r3_actsearch(B,ri);worst=max(worst,float((got-(qa*aa).mm(qb*ab)).abs().max()/((qa*aa).mm(qb*ab)).abs().max().clamp_min(1e-30)))
    contract=checkpoint_contract_selftest()
    w=torch.randn(8,32,requires_grad=True);H4.ste(w,torch.rand(32)+.5).sum().backward();gmax=float((w.grad-1).abs().max())
    lin=torch.nn.Linear(8,8);opt=torch.optim.AdamW(lin.parameters(),lr=1e-3);sc=torch.amp.GradScaler("cpu",enabled=True);x=torch.randn(4,8);snap=lin.weight.detach().clone();opt.zero_grad();sc.scale(lin(x).sum()).backward()
    for p in lin.parameters():p.grad.fill_(float("inf"))
    before=sc.get_scale();sc.unscale_(opt);sc.step(opt);sc.update();declined=(H4.applied_steps(opt)==0 and float((lin.weight.detach()-snap).abs().max())==0. and sc.get_scale()<before);opt.zero_grad();sc.scale(lin(x).sum()).backward();sc.unscale_(opt);sc.step(opt);sc.update();applied=(H4.applied_steps(opt)==1 and float((lin.weight.detach()-snap).abs().max())>0.)
    p=organs[0];m=H4.TernaryLowRank(*(torch.from_numpy(z[p+"."+x]) for x in ("A","s","B","rms_in","rms_A")),None);x=torch.randn(2,m.B.shape[1])
    with torch.no_grad():m(x)
    m(x).sum().backward();cache=(m.A.grad is not None and float(m.A.grad.abs().sum())>0 and m.B.grad is not None and float(m.B.grad.abs().sum())>0 and m.s.grad is not None);ok=shape_ok and not bad and worst<=1e-6 and gmax==0. and declined and applied and cache and contract["passes"]
    return {"G_H4b":{"passes":not bad,"bad":bad},"G_H4c":{"passes":worst<=1e-6,"worst_relative":worst},"G_H4d":{"passes":gmax==0.,"max_gradient_deviation":gmax},"G_H4f":{"passes":declined and applied,"declined_control":declined,"applied_control":applied},"G_H4g":{"passes":cache},"G_H4h":contract,"passes":ok}
def checkpoint_contract_selftest():
    """Synthetic only: verifies durable interim names, metadata, hashes, and refusal."""
    with tempfile.TemporaryDirectory() as td:
        a=SimpleNamespace(out=os.path.join(td,"h4_trained.npz"),steps=4000,bs=2,accum=8,lr=2e-4,seed=1717,factors="h4_factors.npz")
        state={"synthetic":np.arange(6,dtype=np.float32).reshape(2,3)};hashes={"factors_sha256":"factor","origin_factors_sha256":"origin","train_sha256":"train","probe_sha256":"probe"}
        npz_path,json_path=H4.save_interim(state,a,[{"step":250,"tf_fp16_gpu":7,"loss":1.,"seconds":2.}],3,0,2.,{"first_applied_step":1},0,250,250,hashes)
        rec=json.load(open(json_path,encoding="utf-8"));stored=np.load(npz_path)["synthetic"]
        try:H4.save_interim(state,a,[],3,0,2.,None,0,250,250,hashes)
        except SystemExit:collision_refused=True
        else:collision_refused=False
        names=(os.path.basename(npz_path)=="h4_trained.step0250.npz" and os.path.basename(json_path)=="h4_trained.step0250.json")
        metadata=(rec.get("status")=="INTERIM_NONTERMINAL" and rec.get("terminal_checkpoint") is False and rec.get("nonterminal_checkpoint") is True and rec.get("checkpoint_step")==250 and rec.get("checkpoint_interval_steps")==H4.CHECKPOINT_EVERY and rec.get("checkpoint_sha256")==H4.sha256_file(npz_path) and rec.get("input_hashes")==hashes and rec.get("no_best_checkpoint_selection") is True and "terminal output alone adjudicates" in rec.get("adjudication","") and np.array_equal(stored,state["synthetic"]))
        return {"passes":names and metadata and collision_refused,"synthetic_only":True,"names":names,"metadata":metadata,"collision_refused":collision_refused}
if __name__=="__main__":sys.exit(main())
