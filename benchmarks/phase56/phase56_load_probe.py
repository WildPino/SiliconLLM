#!/usr/bin/env python3
# Phase 56 — INFERENCE-LOAD probe (EVAL-ONLY, no re-training). The from-scratch pairs=32/64 trains hit the
#   GROKKING-TIME wall (even the exact sparse ceiling didn't grok in budget) -> can't measure the index wall there.
#   Cheap surrogate: take the GROKKED pairs=16 checkpoints and evaluate them at higher INFERENCE load (more
#   key-value pairs in the same ctx), no training. Does the learned index (simvq) degrade FASTER than the exact
#   ceiling (sparse) as load grows? sparse = generalization ceiling at each load; simvq tracked vs it at nprobe in {4,16,64}.
#   Caveat: this is GENERALIZATION of a p16-trained model, NOT learned-at-scale. If sparse itself collapses ->
#   inconclusive (model doesn't generalize -> need warm-start path). If sparse holds & simvq falls -> index load-fragile.
#
# Run: python benchmarks/phase56/phase56_load_probe.py --run-dir benchmarks/phase56/outputs_run6/phase56_ramp16_complete
import argparse, os, sys, json
import torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from phase56_mqar import Model, eval_recall, BINS

def load(path,dev):
    ck=torch.load(path,map_location=dev,weights_only=False); c=ck['cfg']
    NK,NV=c['keys'],c['vals']; vocab=1+NK+NV
    m=Model(vocab,c['dim'],c['layers'],c['arm'],c['heads'],c['window'],c['topk'],c['nstate'],c['vatoms'],c['nprobe'],c['route_tau']).to(dev)
    m.load_state_dict(ck['model']); m.eval(); return m,c

def evalat(m,c,dev,pairs,queries,topk=None,nprobe=None):
    from argparse import Namespace
    a=Namespace(**c); a.pairs=pairs; a.queries=queries
    slot=m.layers[m.slot].mix
    if topk is not None and hasattr(slot,'topk'): slot.topk=topk        # un-starve the sparse ceiling at higher load
    if nprobe is not None and getattr(m,'routed',None) is not None: m.routed.nprobe=nprobe
    return eval_recall(m,a,c['keys'],c['vals'],dev,seed=777)[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--pairs",default="16,24,32,48,64")
    ap.add_argument("--nprobes",default="4,16,64")
    ap.add_argument("--device",default="auto")
    a=ap.parse_args()
    dev=a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    pl=[int(x) for x in a.pairs.split(",")]; nps=[int(x) for x in a.nprobes.split(",")]
    sm,sc=load(os.path.join(a.run_dir,"sparse","ckpt_sparse.pt"),dev)
    qm,qc=load(os.path.join(a.run_dir,"simvq","ckpt_simvq.pt"),dev)
    print(f"INFERENCE-LOAD probe (eval-only, p16-grokked ckpts) | dev={dev} ({'bf16' if dev=='cuda' else 'fp32'}) | ctx={sc['ctx']}",flush=True)
    print(f"  trained @pairs=16; evaluating @pairs={pl} (queries=pairs, sparse topk un-starved to >=pairs)\n",flush=True)
    hdr=f"  {'pairs':>5} | {'sparse(ceiling)':>15} | "+" | ".join(f"simvq np{n}".rjust(11) for n in nps)
    print(hdr,flush=True)
    rows={}
    for P in pl:
        Q=P
        sp=evalat(sm,sc,dev,P,Q,topk=max(P,16))
        sv={n:evalat(qm,qc,dev,P,Q,nprobe=n) for n in nps}
        rows[P]={"sparse":sp,"simvq":sv}
        print(f"  {P:>5} | {sp*100:14.1f}% | "+" | ".join(f"{sv[n]*100:10.1f}%" for n in nps),flush=True)
    print("\n  reading:",flush=True)
    print("   - sparse holds across load & simvq(best np) tracks it -> index ROBUST to load (generalizes).",flush=True)
    print("   - sparse holds but simvq falls / needs ever-higher np -> index LOAD-FRAGILE -> hierarchy needed at scale.",flush=True)
    print("   - sparse ALSO collapses -> p16 model doesn't generalize to load -> inconclusive, need warm-start (learned-at-scale).",flush=True)
    print("STOP. eval-only, reused p16 ckpts, no retrain, no commit.",flush=True)

if __name__=="__main__": main()
