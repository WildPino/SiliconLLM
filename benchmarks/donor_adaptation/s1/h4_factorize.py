#!/usr/bin/env python3
"""H4 factorisation: isolated rank-48 ternary Q/O masters for Qwen2.5-1.5B.

This is deliberately not an E22-rank512 or E65-QO48 numerical anchor.  H4
uses E22C's canonical activation-weighted, balanced rank-48 construction, but
then ternarizes both factors. G-H4a compares separately constructed H4
masters with the canonical E22C balanced ternary path at rank 48.
"""
import json, os, sys, time
import numpy as np
import torch

HERE=os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(HERE,"..","density")),os.path.abspath(os.path.join(HERE,"..","ternary")),os.path.abspath(os.path.join(HERE,"..","engine"))): sys.path.insert(0,p)
import common as C
import t2_rules as T2
import t2b_organs as T2B
import e22_compose as E22C

THREADS=int(os.environ.get("D_THREADS","6")); torch.set_num_threads(THREADS)
SMOKE=os.environ.get("H4_SMOKE","0")=="1"; RANK=48; D_MODEL=1536; EXPECTED_PARAMS=8_260_224
EXPECTED_CALIB_SHA="c5509846cdc3aa44e45e77895b59a4638c49eb03790030d851e7bf1357ca4c0c"
NCAL,SEQCAL,SEEDCAL=(4 if SMOKE else 32),512,42424
ORGANS=("q_proj","o_proj"); OUTDIR=os.path.join(HERE,"results","h4")
BUNDLE=os.path.join(OUTDIR,"h4_factors%s.npz"%("_smoke" if SMOKE else "")); META=os.path.splitext(BUNDLE)[0]+".json"
def log(x): print(x,flush=True)

def reparam(W,H,r,rms_in,ncal):
    """H0 construction at rank r: balanced fp32 masters and learned identity s."""
    A,B,d=E22C.factors(W,H,r)
    c=A.norm(dim=0).clamp_min(1e-30)
    A0=(A/c).float()
    B0=(c.unsqueeze(1)*B).float()
    s=torch.ones(r,dtype=torch.float32)
    qB,aB=T2.r3_actsearch(B0,rms_in)
    Bt0=qB*aB
    Bd=(s.unsqueeze(1)*Bt0).double()
    rmsA=(torch.diagonal(Bd.mm(H.double()).mm(Bd.T))/float(ncal)).clamp_min(1e-16).sqrt().float()
    qA,aA=T2.r3_actsearch(A0,rmsA)
    Ahat=((qA*aA)*s.unsqueeze(0)).mm(Bt0)
    d.update({"zero_frac_A":float((qA==0).float().mean()),"zero_frac_B":float((qB==0).float().mean()),"c_spread":float(c.max()/c.min().clamp_min(1e-30))})
    return A0,s,B0,rmsA,Ahat,d,(qA,aA,qB,aB),c.float()

def e22_tb_path(W,H,r,rms_in,ncal):
    """Separate canonical E22C balanced ternary path, recomputed at rank r."""
    A,B,_=E22C.factors(W,H,r)
    c=A.norm(dim=0).clamp_min(1e-30)
    Af,Bf=(A/c).float(),(c.unsqueeze(1)*B).float()
    qB,aB=T2.r3_actsearch(Bf,rms_in)
    Bt=qB*aB
    Bd=Bt.double()
    rmsA=(torch.diagonal(Bd.mm(H.double()).mm(Bd.T))/float(ncal)).clamp_min(1e-16).sqrt().float()
    qA,aA=T2.r3_actsearch(Af,rmsA)
    return (qA,aA,qB,aB),(qA*aA).mm(Bt)

def main():
    if os.path.exists(BUNDLE) or os.path.exists(META):raise SystemExit("H4 factors/meta already exist; archive with provenance before rerun")
    os.makedirs(OUTDIR,exist_ok=True); started=time.time(); model,tok=C.load_model(); arch=C.arch(model)
    assert arch["d_model"]==D_MODEL and arch["n_layers"]==28
    layers=[0,27] if SMOKE else list(range(28))
    if os.environ.get("H4_LAYERS"): layers=[int(x) for x in os.environ["H4_LAYERS"].split(",")]
    ids,_,calmeta=C.get_slice(tok,"calib",NCAL,SEQCAL,SEEDCAL)
    if not SMOKE and calmeta.get("ids_sha256")!=EXPECTED_CALIB_SHA:raise SystemExit("H4 calibration slice does not match frozen H0/E22 slice")
    act=T2B.capture(model,ids,arch["n_layers"]); Hs={}; counts={}; hooks=[]
    def hook(key):
        def f(mod,inp,out):
            x=inp[0].detach().reshape(-1,inp[0].shape[-1]).float(); Hs[key]=x.T@x if key not in Hs else Hs[key]+x.T@x; counts[key]=counts.get(key,0)+x.shape[0]
        return f
    for li in layers:
        for nm in ORGANS: hooks.append(getattr(model.model.layers[li].self_attn,nm).register_forward_hook(hook((li,nm))))
    with torch.no_grad():
        for i in range(ids.shape[0]): model(ids[i:i+1])
    for h in hooks: h.remove()
    store={}; rows=[]; bad=[]; worst_scale=worst_prod=0.0
    for li in layers:
        for nm in ORGANS:
            key=(li,nm); W=C.get_linear(model,li,nm).weight.data; ncal=counts[key]
            A,s,B,rmsA,gotprod,d,got,cvec=reparam(W,Hs[key],RANK,act[key],ncal)
            ref,refprod=e22_tb_path(W,Hs[key],RANK,act[key],ncal)
            qA,aA,qB,aB=got; rqA,raA,rqB,raB=ref; codes=bool(torch.equal(qA,rqA) and torch.equal(qB,rqB))
            scale=max(float(((aA-raA).abs()/raA.abs().clamp_min(1e-30)).max()),float(((aB-raB).abs()/raB.abs().clamp_min(1e-30)).max()))
            prod=float((gotprod-refprod).abs().max()/refprod.abs().max().clamp_min(1e-30))
            worst_scale=max(worst_scale,scale); worst_prod=max(worst_prod,prod)
            if not codes or scale!=0.0 or prod!=0.0: bad.append("L%d.%s"%(li,nm))
            p="L%02d.%s"%(li,nm); store.update({p+".A":A.numpy(),p+".s":s.numpy(),p+".B":B.numpy(),p+".rms_in":act[key].numpy(),p+".rms_A":rmsA.numpy()})
            Bunf=B/cvec.unsqueeze(1);fol,uns=B.abs()@act[key],Bunf.abs()@act[key];nf,nu=B.norm(dim=1),Bunf.norm(dim=1)
            rows.append({"layer":li,"organ":nm,"codes_identical":codes,"scale_relmax":scale,"prod_relmax":prod,"rank":RANK,"rel_err":float((gotprod-W).norm()/W.norm()),"zero_A":d["zero_frac_A"],"zero_B":d["zero_frac_B"],"c_spread":d["c_spread"],"range_Bx_folded_KEPT":float(fol.max()/fol.min().clamp_min(1e-30)),"range_Bx_unfolded":float(uns.max()/uns.min().clamp_min(1e-30)),"rownorm_spread_folded_KEPT":float(nf.max()/nf.min().clamp_min(1e-30)),"rownorm_spread_unfolded":float(nu.max()/nu.min().clamp_min(1e-30))})
    npar=int(sum(v.size for k,v in store.items() if k.endswith((".A",".B",".s"))))
    if not SMOKE: assert npar==EXPECTED_PARAMS,(npar,EXPECTED_PARAMS)
    fires=not bad and worst_scale==0.0 and worst_prod==0.0
    meta={"stage":"H4","what":"fp32 masters for ternarize(A)*diag(s)*ternarize(B), q_proj/o_proj only, all 28 layers, rank 48 (D=1536, r/D=1/32); all other donor weights frozen fp32","model":C.MODEL_ID,"revision":C.REVISION,"smoke":SMOKE,"rank":RANK,"d_model":D_MODEL,"rank_fraction":"1/32","organs":list(ORGANS),"layers":layers,"calib_slice":calmeta,"calib_tokens":counts[(layers[0],ORGANS[0])],"G_H4a":{"claim":"separate H4 reparam and canonical E22C balanced ternary path at the SAME rank 48; E22 rank512 and E65 fp32 QO48 are not step-zero anchors","codes_tolerance":0.0,"scale_relative_tolerance":0.0,"product_relative_tolerance":0.0,"product_note":"different multiplication order can create fp32 roundoff; report the measured difference and fail rather than silently accepting it","bad_organs":bad,"worst_scale_rel":worst_scale,"worst_prod_rel":worst_prod,"fires":fires},"trainable_params":npar,"expected_full_trainable_params":EXPECTED_PARAMS,"rows":rows,"seconds":time.time()-started}
    np.savez(BUNDLE,**store)
    with open(META,"x",encoding="utf-8") as f:json.dump(meta,f,indent=1)
    log("G-H4a %s; params %d; wrote %s"%("FIRES" if fires else "FAILS",npar,BUNDLE)); return 0 if fires else 2
if __name__=="__main__": sys.exit(main())
