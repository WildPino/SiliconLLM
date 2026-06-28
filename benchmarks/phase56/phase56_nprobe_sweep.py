#!/usr/bin/env python3
# Phase 56 — nprobe RECALL-SWEEP (EVAL-ONLY, no re-training). Overlays recall(nprobe) across one or more runs
#   (e.g. pairs=16 anchor from RUN-6 + pairs=32 + pairs=64) for a chosen routed arm, vs each run's sparse ceiling.
#   Reuses SAVED checkpoints; re-evaluates the HARD IVF path at nprobe in {4,8,16,32,64}, same eval-set/seed (777).
#   Gate-0: each run's nprobe=4 must reproduce its recorded number (else the ckpt mis-reloads -> STOP, debug THAT).
#   NOTE: index COST @128K is pairs-INDEPENDENT (n_cand=nprobe/V*128K); cost is measured separately (ivfpq_profile).
#   (Local torch may be CPU/fp32 while runs were cuda/bf16 -> reproduction within tolerance, not bit-exact.)
#
# Run: python benchmarks/phase56/phase56_nprobe_sweep.py \
#        --runs "16=benchmarks/phase56/outputs_run6/phase56_ramp16_complete,32=<dir32>,64=<dir64>" --arm simvq
import argparse, os, sys, json
import torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from phase56_mqar import Model, eval_recall, BINS

def load(path,dev):
    ck=torch.load(path,map_location=dev,weights_only=False); c=ck['cfg']
    NK,NV=c['keys'],c['vals']; vocab=1+NK+NV
    m=Model(vocab,c['dim'],c['layers'],c['arm'],c['heads'],c['window'],c['topk'],c['nstate'],c['vatoms'],c['nprobe'],c['route_tau']).to(dev)
    m.load_state_dict(ck['model']); m.eval()
    return m,c

def recorded(d,arm):
    p=os.path.join(d,arm,"mqar_results.json")
    return json.load(open(p))["results"][arm]["ov"] if os.path.exists(p) else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",required=True,help='comma list label=dir (label usually = pairs); each dir has <arm>/ and sparse/')
    ap.add_argument("--arm",default="simvq",help="routed arm to sweep (ckpt_<arm>.pt in <dir>/<arm>/)")
    ap.add_argument("--nprobes",default="4,8,16,32,64")
    ap.add_argument("--device",default="auto")
    ap.add_argument("--out",default="")
    a=ap.parse_args()
    dev = a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    nprobes=[int(x) for x in a.nprobes.split(",")]
    runs=[(lbl,d) for lbl,d in (kv.split("=",1) for kv in a.runs.split(","))]
    ab=[bi for bi,(lo,hi) in enumerate(BINS) if hi<=512]
    print(f"nprobe RECALL-SWEEP overlay | arm={a.arm} | dev={dev} ({'bf16' if dev=='cuda' else 'fp32'}) | nprobes={nprobes}",flush=True)
    res={}
    for lbl,d in runs:
        ckp=os.path.join(d,a.arm,f"ckpt_{a.arm}.pt")
        if not os.path.exists(ckp): print(f"  [{lbl}] MISSING {ckp} -> skip",flush=True); continue
        m,c=load(ckp,dev); from argparse import Namespace; aa=Namespace(**c)
        sref=recorded(d,"sparse"); rec4=recorded(d,a.arm)
        res[lbl]={"pairs":c['pairs'],"sparse":sref,"rec4":rec4,"curve":{}}
        print(f"\n  == run {lbl} (pairs={c['pairs']}, topk={c['topk']}) | sparse ceiling={sref*100 if sref else float('nan'):.1f}% ==",flush=True)
        for npb in nprobes:
            m.routed.nprobe=npb
            hit,tot,ov=eval_recall(m,aa,c['keys'],c['vals'],dev,seed=777)
            bins={bi:(hit[bi]/tot[bi] if tot[bi]>0 else 0.0) for bi in ab}
            res[lbl]["curve"][npb]={"ov":ov,"bins":bins}
            print(f"    nprobe={npb:3d} recall={ov*100:5.1f}%  "+" ".join(f"{BINS[bi][0]}-{BINS[bi][1]}:{bins[bi]*100:4.1f}%" for bi in ab),flush=True)
        if rec4 is not None:
            got=res[lbl]["curve"][4]["ov"]*100 if 4 in res[lbl]["curve"] else float('nan')
            d4=abs(got-rec4*100); print(f"    GATE-0 @nprobe4: got {got:.1f}% vs recorded {rec4*100:.1f}% (|d|={d4:.1f}) -> {'OK' if d4<=2 else 'MISMATCH-DEBUG'}",flush=True)
    # overlay table
    print(f"\n  ==== recall(nprobe) overlay across runs | arm={a.arm} ====",flush=True)
    print("  nprobe | "+" | ".join(f"pairs{res[l]['pairs']}".rjust(11) for l in res),flush=True)
    for npb in nprobes:
        print(f"  {npb:6d} | "+" | ".join(f"{res[l]['curve'][npb]['ov']*100:10.1f}%" for l in res),flush=True)
    print("  ceil   | "+" | ".join(f"{(res[l]['sparse'] or 0)*100:10.1f}%" for l in res)+"   (sparse exact ceiling per pairs)",flush=True)
    # in-budget read: cost gate caps nprobe<=~16 (24us) at V=256 dim128 (cost is pairs-independent)
    print(f"\n  in-budget (<=30us ~ nprobe<=16) recall per pairs:  "+
          "  ".join(f"p{res[l]['pairs']}={res[l]['curve'][16]['ov']*100:.1f}%" for l in res),flush=True)
    print(f"  ceiling-recall nprobe (where curve meets sparse):  "+
          "  ".join(f"p{res[l]['pairs']}=~{next((n for n in nprobes if res[l]['curve'][n]['ov']>=(res[l]['sparse'] or 1)-0.01),'>64')}" for l in res),flush=True)
    if a.out: json.dump(res,open(a.out,"w"),indent=1); print(f"  [saved -> {a.out}]",flush=True)
    print("STOP. eval-only, reused ckpts, no retrain, no commit.",flush=True)

if __name__=="__main__": main()
