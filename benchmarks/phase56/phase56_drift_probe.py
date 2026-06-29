#!/usr/bin/env python3
# Phase 56 - DRIFT make-or-break APPARATUS probe (R-B), EVAL-ONLY. Decides the index architecture: does the LEARNED
#   coarse partition go stale under the SSM query/key drift at length, while a DATA-INDEPENDENT partition (+ l2-query)
#   cures it while KEEPING the InfoNCE representation? This script validates apparatus + mechanism (NOT the verdict;
#   the verdict needs the long-ctx train the Capo launches). Three readouts:
#     (1) MECHANISM GATE  : ||h_t|| per-layer vs absolute position on a long input. R-B predicts SSM state norm grows
#                           ~10x at long range. If it does NOT grow on OUR model -> drift mechanism differs -> STOP/re-scope.
#     (2) RECALL vs POSITION: arm A (learned partition) vs arm B (Hadamard data-independent), encoder=InfoNCE in BOTH.
#                           At SHORT distance (no drift) sanity: A ~= B ~= ceiling. The gap, if any, emerges at length.
#     (3) l2-QUERY ABLATION: arm A with qnorm ON vs OFF across position -> confirms it removes the radial drift.
#
# Run (after a drift train that produced <dir>/simvq/ and <dir>/simvq-fixed/):
#   python benchmarks/phase56/phase56_drift_probe.py --run-dir <dir> --hnorm-ctx 8192
#   (mechanism gate alone, on any existing moderate-ctx ckpt as an OOD-length proxy:)
#   python benchmarks/phase56/phase56_drift_probe.py --hnorm-only --hnorm-ckpt <path.pt> --hnorm-ctx 8192
import argparse, os, sys
import torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from phase56_mqar import Model, eval_strat, make_batch, POSBINS, BINS

def find_ckpt(run_dir,arm):                               # accept both layouts: <dir>/<arm>/ckpt_<arm>.pt OR flat <dir>/ckpt_<arm>.pt
    for p in (os.path.join(run_dir,arm,f"ckpt_{arm}.pt"), os.path.join(run_dir,f"ckpt_{arm}.pt")):
        if os.path.exists(p): return p
    return ""

def load(path,dev):
    ck=torch.load(path,map_location=dev,weights_only=False); c=ck['cfg']
    NK,NV=c['keys'],c['vals']; vocab=1+NK+NV
    m=Model(vocab,c['dim'],c['layers'],c['arm'],c['heads'],c['window'],c['topk'],c['nstate'],
            c['vatoms'],c['nprobe'],c['route_tau'],c.get('qnorm',False)).to(dev)
    m.load_state_dict(ck['model']); m.eval(); return m,c

def hnorm(m,c,dev,ctx,grok_thr=0.6):                      # (1) ||h_t|| per layer vs position on a long spread input
    from argparse import Namespace; a=Namespace(**c); a.ctx=ctx
    # GROK-GATE: ||h_t|| is meaningful ONLY if the model actually WORKS at this ctx (else the query is not a query -> norm = noise)
    _,_,_,_,ov=eval_strat(m,a,c['keys'],c['vals'],dev,seed=777,spread=True)
    grok=ov>=grok_thr
    print(f"  GROK-GATE @ctx={ctx}: recall={ov*100:.1f}%  -> {'GROKKED (||h|| reading VALID)' if grok else f'NOT GROKKED (<{grok_thr*100:.0f}%) -> ||h|| below is NOISE (training problem: more steps/warm-start), NOT a mechanism finding'}",flush=True)
    gen=torch.Generator().manual_seed(777)
    x,_,_,_=make_batch(min(a.batch,8),ctx,a.pairs,a.queries,c['keys'],c['vals'],gen,dev,spread=True)
    m._cap_h=True; m._hbuf=None
    with torch.no_grad():
        with torch.autocast(dev,dtype=torch.bfloat16,enabled=(c.get('bf16') and dev=='cuda')): m(x)
    hb=m._hbuf; m._cap_h=False; m._hbuf=None
    T=hb[0].shape[0]; wins=[(0,T//8),(T//8,T//4),(T//4,T//2),(T//2,3*T//4),(3*T//4,T)]
    print(f"  (1) MECHANISM GATE | ||h_t|| vs position (ctx={ctx}) | layers={len(hb)} (0=emb){'' if grok else '  [NOISE: not grokked]'}",flush=True)
    hdr="    layer | "+" | ".join(f"[{lo}-{hi})".rjust(12) for lo,hi in wins)+" |   tail/head"
    print(hdr,flush=True)
    for li,h in enumerate(hb):
        vals=[float(h[lo:hi].float().mean()) for lo,hi in wins]
        ratio=vals[-1]/max(vals[0],1e-9)
        print(f"    {('emb' if li==0 else f'L{li-1}'):>5} | "+" | ".join(f"{v:12.3f}" for v in vals)+f" | {ratio:9.2f}x",flush=True)
    # verdict on the SSM-driven layers (exclude the final routed/attn slot)
    ssm_ratios=[hb[li+1][3*T//4:T].float().mean()/max(hb[li+1][0:T//8].float().mean(),1e-9) for li in range(len(hb)-1)]
    mx=float(max(ssm_ratios)); print(f"    -> max layer tail/head norm ratio = {mx:.2f}x  ({'GROWS (drift mechanism present)' if mx>=3 else 'FLAT (<3x: mechanism NOT Mamba-2-like -> RE-SCOPE before long train)'})",flush=True)
    return mx

def recall_pos(m,c,dev,ctx,qnorm=None):
    from argparse import Namespace; a=Namespace(**c); a.ctx=ctx
    if qnorm is not None and getattr(m,'routed',None) is not None: m.routed.qnorm=qnorm
    ph,pt,dh,dt,ov=eval_strat(m,a,c['keys'],c['vals'],dev,seed=777,spread=True)
    return ph,pt,dh,dt,ov

def fmt_pos(ph,pt):
    ab=[bi for bi in range(len(POSBINS)) if pt[bi]>0]
    return ab," ".join(f"{POSBINS[bi][0]}-{POSBINS[bi][1] if POSBINS[bi][1]<(1<<20) else 'inf'}:{ph[bi]/pt[bi]*100:4.0f}%" for bi in ab)
def fmt_dist(dh,dt):
    ab=[bi for bi in range(len(BINS)) if dt[bi]>0]
    return " ".join(f"{BINS[bi][0]}-{BINS[bi][1] if BINS[bi][1]<(1<<20) else 'inf'}:{dh[bi]/dt[bi]*100:4.0f}%" for bi in ab)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",default="",help="dir with simvq/ (arm A) and simvq-fixed/ (arm B)")
    ap.add_argument("--arms",default="simvq,simvq-fixed")
    ap.add_argument("--ctx",type=int,default=0,help="recall-vs-position eval ctx (0=trained ctx)")
    ap.add_argument("--hnorm-ctx",type=int,default=8192,help="long ctx for the ||h_t|| mechanism gate (OOD-length proxy)")
    ap.add_argument("--hnorm-ckpt",default="",help="ckpt for the mechanism gate (default: run-dir/simvq)")
    ap.add_argument("--hnorm-only",action="store_true")
    ap.add_argument("--grok-thr",type=float,default=0.6,help="GROK-GATE: below this overall recall the model isn't working -> ||h|| and A/B are NOISE (a training problem), not findings")
    ap.add_argument("--device",default="auto")
    a=ap.parse_args()
    dev=a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DRIFT apparatus probe (R-B) | dev={dev}",flush=True)

    # (1) mechanism gate
    hck=a.hnorm_ckpt or (find_ckpt(a.run_dir,"simvq") if a.run_dir else "")
    if hck and os.path.exists(hck):
        mh,ch=load(hck,dev); hnorm(mh,ch,dev,a.hnorm_ctx,a.grok_thr)
    else:
        print(f"  (1) MECHANISM GATE skipped (no ckpt at {hck})",flush=True)
    if a.hnorm_only:
        print("STOP. eval-only, mechanism gate only, no commit.",flush=True); return
    print("",flush=True)

    arms=[s for s in a.arms.split(",") if s]
    loaded={}
    for arm in arms:
        p=find_ckpt(a.run_dir,arm)
        if p: loaded[arm]=load(p,dev)
        else: print(f"  [{arm}] MISSING ckpt_{arm}.pt under {a.run_dir} -> skip",flush=True)
    if not loaded:
        print("  no arm ckpts -> only mechanism gate available. STOP.",flush=True); return
    ctx=a.ctx or list(loaded.values())[0][1]['ctx']

    # (2) recall vs POSITION overlay (A vs B), + distance secondary
    print(f"  (2) RECALL vs POSITION (ctx={ctx}, spread queries) | arm A=learned vs arm B=data-independent (encoder=InfoNCE both)",flush=True)
    res={}; ungrokked=[]
    for arm,(m,c) in loaded.items():
        ph,pt,dh,dt,ov=recall_pos(m,c,dev,ctx)
        res[arm]=(ph,pt,dh,dt,ov); ab,line=fmt_pos(ph,pt)
        gflag='' if ov>=a.grok_thr else f'  <-- NOT GROKKED (<{a.grok_thr*100:.0f}%): NOISE, train more'
        if ov<a.grok_thr: ungrokked.append(arm)
        print(f"    {arm:12s} overall={ov*100:5.1f}% | by POS:  {line}{gflag}",flush=True)
        print(f"    {'':12s}               | by DIST: {fmt_dist(dh,dt)}",flush=True)
    if ungrokked:
        print(f"    !! GROK-GATE: {ungrokked} below {a.grok_thr*100:.0f}% -> A/B comparison INVALID (training problem, not partition-drift). Extend: --resume <out-dir> --steps <more>.",flush=True)
    else:
        print("    read: SHORT-position sanity -> A ~= B ~= ceiling (B not broken). Gap opening at HIGH position = learned partition drift-brittle.",flush=True)

    # (3) l2-query ablation on arm A (drift is radial -> qnorm should flatten the position curve)
    if 'simvq' in loaded:
        print(f"\n  (3) l2-QUERY ABLATION on arm A (qnorm ON vs OFF) vs position",flush=True)
        m,c=loaded['simvq']
        for q in (True,False):
            ph,pt,dh,dt,ov=recall_pos(m,c,dev,ctx,qnorm=q); ab,line=fmt_pos(ph,pt)
            print(f"    qnorm={'ON ' if q else 'OFF'} overall={ov*100:5.1f}% | by POS: {line}",flush=True)
        m.routed.qnorm=c.get('qnorm',False)  # restore
        print("    read: qnorm flattens the position curve (radial drift removed) -> bake it in. (NB: post-hoc toggle on a qnorm-trained model is indicative; the trained ablation is --no-qnorm arm.)",flush=True)
    print("\nSTOP. eval-only, reused drift ckpts, no retrain, no commit.",flush=True)

if __name__=="__main__": main()
