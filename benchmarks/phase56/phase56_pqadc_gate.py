#!/usr/bin/env python3
# Phase 56 — PQ-ADC GATE (EVAL-ONLY, reuse p64 ckpt). Closes the cost-quality consistency hole: the recall
#   numbers were measured with EXACT refine, but the cheap cost assumes PQ-ADC. Here we swap the recall-path
#   refine EXACT -> PQ-ADC 4-bit (asymmetric: query exact, KEY reconstructed from 4-bit PQ codes, DS=2 -> M=64
#   subquantizers over d=128 = the config the 8us cost-estimate assumed) and measure the approximation TAX on p64.
#   PQ codebook is FIT (k-means) on the model's own keys (eval-only, no retraining of the network).
#
# Run: python benchmarks/phase56/phase56_pqadc_gate.py --run-dir benchmarks/phase56/outputs_run8/phase56_scale_ws/sz64
import argparse, os, sys
import torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from phase56_mqar import Model, eval_recall, make_batch, BINS

def load(path,dev):
    ck=torch.load(path,map_location=dev,weights_only=False); c=ck['cfg']
    NK,NV=c['keys'],c['vals']; vocab=1+NK+NV
    m=Model(vocab,c['dim'],c['layers'],c['arm'],c['heads'],c['window'],c['topk'],c['nstate'],c['vatoms'],c['nprobe'],c['route_tau']).to(dev)
    m.load_state_dict(ck['model']); m.eval(); return m,c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True,help="dir with simvq/ckpt_simvq.pt (the p64 run)")
    ap.add_argument("--nprobes",default="4,16"); ap.add_argument("--fit-batches",type=int,default=6)
    ap.add_argument("--configs",default="4:2,8:4",help="comma list bits:DS (4:2=4-bit/2dim, 8:4=8-bit/4dim; both 32B/key at d=128)")
    ap.add_argument("--device",default="auto")
    a=ap.parse_args()
    dev=a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    nps=[int(x) for x in a.nprobes.split(",")]
    cfgs=[(int(b),int(d)) for b,d in (kv.split(":") for kv in a.configs.split(","))]
    m,c=load(os.path.join(a.run_dir,"simvq","ckpt_simvq.pt"),dev); rt=m.routed
    from argparse import Namespace; aa=Namespace(**c)
    print(f"PQ-ADC gate | p{c['pairs']} simvq | dev={dev} | hd={rt.hd} | configs(bits:DS)={cfgs}",flush=True)
    # --- collect keys ONCE (k-means PQ fit per-config below) ---
    rt._capture=True; rt._kbuf=[]; gen=torch.Generator().manual_seed(777)
    with torch.no_grad():
        for _ in range(a.fit_batches):
            x,_,_,_=make_batch(c['batch'],c['ctx'],c['pairs'],c['queries'],c['keys'],c['vals'],gen,dev); m(x)
    rt._capture=False; kbuf=rt._kbuf; rt._kbuf=None
    ab=[bi for bi,(lo,hi) in enumerate(BINS) if hi<=512]
    # exact baseline (PQ-independent)
    rt.refine='exact'; base={}
    for npb in nps:
        rt.nprobe=npb; hit,tot,ov=eval_recall(m,aa,c['keys'],c['vals'],dev,seed=777); base[npb]=ov
        print(f"  [exact]  nprobe={npb:>3} recall={ov*100:.1f}%",flush=True)
    print("",flush=True)
    for bits,ds in cfgs:
        rt.pq_ds=ds; rt.fit_pq(kbuf,ncodes=2**bits); rt.refine='pqadc'
        Bpk=(c['dim']//ds)*bits//8
        print(f"  == PQ {bits}-bit DS={ds} -> M={c['dim']//ds} subq, {Bpk}B/key ==",flush=True)
        for npb in nps:
            rt.nprobe=npb; hit,tot,ov=eval_recall(m,aa,c['keys'],c['vals'],dev,seed=777)
            tax=(base[npb]-ov)*100
            print(f"    nprobe={npb:>3} pqadc={ov*100:5.1f}% vs exact {base[npb]*100:5.1f}%  TAX {tax:+.1f} pts",flush=True)
        print("",flush=True)
    # --- 2-stage: cheap 4-bit ADC shortlist (pshufb-native) + EXACT rerank of top-kshort ---
    rt.pq_ds=2; rt.fit_pq(kbuf,ncodes=16); rt.refine='rerank'
    print("  == 2-STAGE: 4-bit ADC shortlist + EXACT rerank of top-kshort (keeps cheap pshufb scan) ==",flush=True)
    for ks in (16,32):
        rt.kshort=ks
        for npb in nps:
            rt.nprobe=npb; hit,tot,ov=eval_recall(m,aa,c['keys'],c['vals'],dev,seed=777)
            tax=(base[npb]-ov)*100
            print(f"    kshort={ks:>2} nprobe={npb:>3} recall={ov*100:5.1f}% vs exact {base[npb]*100:5.1f}%  TAX {tax:+.1f} pts",flush=True)
    print("",flush=True)
    print("  reading: tax ~0 & >=~99% -> recall-path validated. 4-bit pqadc large tax -> 2-stage rerank (cheap) OR anisotropic/score-aware PQ (ScaNN) close it; pure-bits doesn't (and 8-bit breaks pshufb).",flush=True)
    print("STOP. eval-only, reused p64 ckpt, PQ fit on keys, no network retrain, no commit.",flush=True)

if __name__=="__main__": main()
