#!/usr/bin/env python3
# Phase 62 / Task 3 - FOUNDATION on Cat-A + honest baselines + property TRANSFER. magic: "P62T" 0x50363254
#   Trains the SAME 5M Arch-A foundation recipe (dReLU-gated-ternary MLP, fp32 SSM projections -- the P61 verdict)
#   on a Cat-A domain (code|log), then answers the PRE-REGISTERED characterization questions (no hard gate):
#     Q1  stable train?                    -> loss/gnorm curve + final val BPB (bits/byte, unit-invariant).
#     (a) BPB vs COMPRESSOR FLOOR          -> zstd-19 + brotli-q11 on the SAME val bytes (our honest heritage).
#     (b) property transfer:
#         sparsity per-layer (anchor TinyStories: hidden ~92% zero / gate ~79% zero)
#         in-place predictability (ridge x_t -> gate support, recall@k held-out; anchor 86-92%)
#         persistence  A_t vs A_{t-1}      (anchor ~nil / base-rate)
#     (c) greedy generation sample (qualitative).
#   Unit = PER-DOMAIN BPE-1024 relearned on TRAIN split (results/phase62/bpe1024_{dom}.bin). Corpus = corpus.py.
#
# Smoke (CPU): .venv/Scripts/python.exe benchmarks/phase62/task3_catA.py --smoke --domain code
# Real (3060): .venv/Scripts/python.exe benchmarks/phase62/task3_catA.py --domain code --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase62/catA_code.pt
#         ... : .venv/Scripts/python.exe benchmarks/phase62/task3_catA.py --domain log  --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase62/catA_log.pt
import argparse, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55")); sys.path.insert(0, os.path.join(HERE, "..", "phase57")); sys.path.insert(0, HERE)
from phase55_ssm import ArchA, load_meta
from phase57_sparse import sparsify_mlp
from phase58_predict import ridge_fit, ridge_apply, recall_at_k
import zstandard, brotli
ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); P62 = os.path.join(ROOT, "results", "phase62")

def compressor_floor(vbytes):
    z = zstandard.ZstdCompressor(level=19).compress(vbytes)
    b = brotli.compress(vbytes, quality=11)
    n = len(vbytes)
    return dict(zstd19_bpb=len(z)*8/n, brotli11_bpb=len(b)*8/n, raw_bytes=n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["code","log"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--bf16", action="store_true"); ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--eval-tok", type=int, default=200000); ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--save", type=str, default=""); ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--probe-nbatch", type=int, default=24); ap.add_argument("--ridge", type=float, default=1.0)
    a = ap.parse_args()
    if a.smoke: a.steps=60; a.seq=64; a.batch=4; a.eval_tok=20000; a.probe_nbatch=12
    dev = a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    amp = torch.float16 if a.fp16 else (torch.bfloat16 if a.bf16 else None)
    torch.manual_seed(0); np.random.seed(0)
    if dev_type=="cuda": torch.cuda.manual_seed_all(0)

    dom = a.domain
    V, exp_len, id2b = load_meta(os.path.join(P62, f"{dom}.meta"))
    train = np.fromfile(os.path.join(P62, f"{dom}_train.u16"), dtype=np.uint16).astype(np.int64)
    val   = np.fromfile(os.path.join(P62, f"{dom}_val.u16"),   dtype=np.uint16).astype(np.int64)
    el = torch.tensor(exp_len)
    D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16); hid = 4*D
    model = ArchA(V, **AC).to(dev); model.use_ckpt = not a.no_ckpt
    nmlp = sparsify_mlp(model, D, hid, act="drelu", gated=True, topk=0.0, ternary=True)  # foundation MLP (fp32 proj)
    model = model.to(dev); npar = sum(p.numel() for p in model.parameters())
    print(f"Phase62 Task3 Cat-A | domain={dom} V={V} | Arch-A params={npar/1e6:.3f}M | dev={dev} amp={amp}")
    print(f"  corpus: train={len(train)}tok val={len(val)}tok | MLP={nmlp} drelu-gated-ternary, proj fp32 (foundation)")
    print(f"  anchors(TinyStories): hidden~92%/gate~79% zero, recall_inplace 86-92%, persist~base-rate")

    def actx(): return torch.autocast(dev_type, dtype=amp) if amp is not None else torch.autocast(dev_type, enabled=False)
    def val_bpb(cap):
        model.eval(); bits=0.0; nb=0
        with torch.no_grad():
            W=a.seq; lim=min(cap, len(val)-1); pos=0
            while pos+W+1<=lim:
                x=torch.from_numpy(val[pos:pos+W][None,:]).to(dev); y=torch.from_numpy(val[pos+1:pos+1+W][None,:]).to(dev)
                bits+=F.cross_entropy(model(x).reshape(-1,V), y.reshape(-1), reduction="sum").item()/math.log(2)
                nb+=int(el[y.reshape(-1).cpu()].sum().item()); pos+=W
        model.train(); return bits/max(nb,1)

    opt=torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9,0.95), weight_decay=0.1)
    def get_batch(src):
        ix=np.random.randint(0, len(src)-a.seq-1, size=a.batch)
        return (torch.from_numpy(np.stack([src[i:i+a.seq] for i in ix])).to(dev),
                torch.from_numpy(np.stack([src[i+1:i+1+a.seq] for i in ix])).to(dev))
    print("== training (CE only) == [Q1: stable?]")
    model.train()
    for step in range(a.steps):
        ts=time.time(); opt.zero_grad(); lsum=0.0
        for _ in range(a.accum):
            x,y=get_batch(train)
            with actx(): loss=F.cross_entropy(model(x).reshape(-1,V), y.reshape(-1))/a.accum
            loss.backward(); lsum+=loss.item()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==0 or (step+1)%max(1,a.steps//20)==0:
            if dev_type=="cuda": torch.cuda.synchronize()
            ms=(time.time()-ts)*1000; vmsg=""
            if (step+1)%max(1,a.steps//10)==0 or step==0: vmsg=f"  | val BPB={val_bpb(min(a.eval_tok,16000)):.4f}"
            print(f"  step {step+1:5d}/{a.steps} loss={lsum:.4f} gnorm={gn:.2f} ({ms:.0f} ms){vmsg}")
    bpb=val_bpb(a.eval_tok)

    # ---- (a) compressor floor on the SAME val bytes ----
    vbytes=b"".join(id2b[int(t)] for t in val[:min(len(val), a.eval_tok)])
    comp=compressor_floor(vbytes)

    # ---- (b) transfer probes ----
    #   sparsity: SparseMLP built-ins _s_h (hidden=relu(g)*relu(u)==0, anchor 92%) and _s_g (gate g<=0, anchor 79%).
    #   recall/persistence: hook x_t (mlp input) + gate support (g>0) tensors.
    cap_x={i:[] for i in range(AC["L"])}; cap_g={i:[] for i in range(AC["L"])}
    sh={i:[] for i in range(AC["L"])}; sg={i:[] for i in range(AC["L"])}; handles=[]; mlp_layers=[]
    for i,blk in enumerate(model.blocks):
        m=getattr(blk,"mlp",None)
        if m is None: continue
        mlp_layers.append(i); m._cap=True
        def mk_pre(i):
            def h(mod,inp): cap_x[i].append(inp[0].detach().float().cpu().numpy())
            return h
        def mk_gate(i):
            def h(mod,inp,out): cap_g[i].append((out.detach()>0).cpu().numpy())
            return h
        handles.append(m.register_forward_pre_hook(mk_pre(i)))
        gate_mod = m.gate if getattr(m,"gated",False) and hasattr(m,"gate") else m.up
        handles.append(gate_mod.register_forward_hook(mk_gate(i)))
    model.eval()
    with torch.no_grad():
        for bi in range(a.probe_nbatch):
            pos=bi*a.seq
            if pos+a.seq+1>len(val): break
            model(torch.from_numpy(val[pos:pos+a.seq][None,:]).to(dev))
            for i in mlp_layers:
                m=model.blocks[i].mlp; sh[i].append(m._s_h); sg[i].append(m._s_g)
    for hd in handles: hd.remove()
    for i in mlp_layers: model.blocks[i].mlp._cap=False
    model.train()

    print("\n== RESULTS ==")
    print(f"  [Q1] domain={dom} FINAL val BPB={bpb:.4f} bits/byte  (over ~{comp['raw_bytes']} val bytes)")
    print(f"  [a ] compressor floor SAME bytes: zstd-19={comp['zstd19_bpb']:.4f}  brotli-q11={comp['brotli11_bpb']:.4f}  "
          f"-> model {'BELOW (beats)' if bpb<min(comp['zstd19_bpb'],comp['brotli11_bpb']) else 'ABOVE'} best floor")
    print(f"  [b ] property transfer per MLP layer (anchor hidden~92%/gate~79% zero | recall 86-92% | persist~base):")
    print(f"       {'layer':>5} {'hid_zero%':>9} {'gate_zero%':>10} {'recall_inplace%':>15} {'persist%':>9} {'baserate%':>9}")
    hid_z=[]; gate_z=[]; rec=[]
    for i in mlp_layers:
        if not cap_g[i]: continue
        G=np.concatenate([g.reshape(-1,g.shape[-1]) for g in cap_g[i]],0)      # (M,hid) gate>0
        X=np.concatenate([x.reshape(-1,x.shape[-1]) for x in cap_x[i]],0)      # (M,D)
        active=G; act_frac=active.mean()
        hz=float(np.mean(sh[i]))*100; gz=float(np.mean(sg[i]))*100             # anchor defs (built-in capture)
        # in-place recall: ridge x->gate support on FIT half, recall on held-out half
        M=X.shape[0]; h=M//2; rr=float("nan")
        if h>D+2 and active[:h].sum()>0:
            W=ridge_fit(X[:h].astype(np.float64), active[:h].astype(np.float64), a.ridge)
            sc=ridge_apply(X[h:].astype(np.float64), W); rr=100*recall_at_k(sc, active[h:])
        At=active[1:]; Atm1=active[:-1]                                        # persistence (approx: consecutive rows)
        persist=100*np.mean((At&Atm1).sum(1)/np.maximum(At.sum(1),1))
        hid_z.append(hz); gate_z.append(gz); rec.append(rr)
        print(f"       {i:>5} {hz:>9.1f} {gz:>10.1f} {rr:>15.1f} {persist:>9.1f} {100*act_frac:>9.1f}")
    print(f"  [Q2] mean gate_zero={np.nanmean(gate_z):.1f}% (anchor 79)  mean recall_inplace={np.nanmean(rec):.1f}% (anchor 86-92)")

    # ---- (c) greedy sample ----
    def greedy(n=400, rep=1.2, win=128):
        model.eval(); ctx=[int(t) for t in val[:16]]; out=bytearray()
        with torch.no_grad():
            while len(out)<n:
                x=torch.tensor(ctx[-a.seq:])[None,:].to(dev); lg=model(x)[0,-1].float().cpu().clone()
                recent=sorted(set(ctx[-win:])); idx=torch.tensor(recent); sv=lg[idx]
                lg[idx]=torch.where(sv>0, sv/rep, sv*rep); tok=int(lg.argmax()); ctx.append(tok); out+=id2b[tok]
        model.train(); return bytes(out[:n])
    print(f"  [c ] greedy sample: {greedy()[:200]!r}")

    if a.save:
        torch.save({"model":model.state_dict(), "cfg":dict(V=V,**AC,win=128,expand=2,conv=4,seq=a.seq,steps=a.steps,
                    bpb=bpb,domain=dom,act="drelu",gated=True,mlp_precision="ternary",proj_precision="fp32",
                    zstd19_bpb=comp['zstd19_bpb'],brotli11_bpb=comp['brotli11_bpb'])}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. Cat-A foundation characterization (no gate). No commit.")

if __name__ == "__main__":
    main()
