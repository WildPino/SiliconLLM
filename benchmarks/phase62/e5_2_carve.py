#!/usr/bin/env python3
# Phase 62 -> E5.2 - CARVE / SET-BLOCK training (E5 PRIMARY lane). magic: "E52C" 0x45353243
#   E5.1 verdict: frozen MTP = 1.96 tok/pass < 2.0 -> spec-AR out, carve is primary. The frozen hidden carries
#   ~1 reliable step of look-ahead; CO-TRAINING is the lever it lacked. Carve co-trains BY CONSTRUCTION.
#
#   Block factorization: from the state h_t the model emits a BLOCK of B tokens (offset 0 = the base head t+1,
#   offsets 1..B-1 = per-offset heads t+2..t+B), all co-trained with the backbone. At generation the state
#   advances by RE-COMPUTING the scan over the B committed tokens (R-H: recompute dissolves rollback). The
#   weight-stream amortization is B, DETERMINISTIC -- no acceptance stochasticity (unlike spec-AR).
#
#   val BPB is scored UNDER ITS OWN FACTORIZATION: prod_k P(t+k | state at block start). B=1 reduces to the
#   standard AR model (anchor catA_code 1.242). Per-offset CE shows where the price is paid.
#
#   PRE-REGISTERED GATE (Architect, fixed before results): carve = viable E5 lane IFF BPB(B=2) <= 1.242 + 0.015
#   (~3 sigma_seed; the price buys a guaranteed 2x weight-stream amortization). B=4 reported as a curve (interest
#   <= +0.040). If even B=2 fails -> multi-token closed at this scale; E5 falls back to block-verify chassis + threads.
#
# Smoke (CPU): .venv/Scripts/python.exe benchmarks/phase62/e5_2_carve.py --smoke --B 2
# B=2 (3060) : .venv/Scripts/python.exe benchmarks/phase62/e5_2_carve.py --B 2 --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase62/carve_b2.pt
# B=4 (3060) : .venv/Scripts/python.exe benchmarks/phase62/e5_2_carve.py --B 4 --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase62/carve_b4.pt
import argparse, os, sys, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55")); sys.path.insert(0, os.path.join(HERE, "..", "phase57")); sys.path.insert(0, HERE)
from phase55_ssm import ArchA, load_meta
from phase57_sparse import sparsify_mlp
from phase58_predict import ridge_fit, ridge_apply, recall_at_k
from e5_1_mtp import MedusaHeads, capture_hidden
ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); P62 = os.path.join(ROOT, "results", "phase62")
ANCHOR = 1.242; GATE_B2 = ANCHOR + 0.015

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2, help="block size (tokens emitted per state pass)")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--bf16", action="store_true"); ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--eval-tok", type=int, default=200000); ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--save", default=""); ap.add_argument("--device", default="auto")
    ap.add_argument("--probe-nbatch", type=int, default=24); ap.add_argument("--ridge", type=float, default=1.0)
    a = ap.parse_args()
    if a.smoke: a.steps=60; a.seq=64; a.batch=4; a.eval_tok=20000; a.probe_nbatch=12
    dev = a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    amp = torch.float16 if a.fp16 else (torch.bfloat16 if a.bf16 else None)
    torch.manual_seed(0); np.random.seed(0)
    if dev_type=="cuda": torch.cuda.manual_seed_all(0)
    B = a.B

    V, exp_len, id2b = load_meta(os.path.join(P62, "code.meta"))
    train = np.fromfile(os.path.join(P62, "code_train.u16"), dtype=np.uint16).astype(np.int64)
    val   = np.fromfile(os.path.join(P62, "code_val.u16"),   dtype=np.uint16).astype(np.int64)
    el = torch.tensor(exp_len)
    D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16); hid = 4*D
    model = ArchA(V, **AC).to(dev); model.use_ckpt = not a.no_ckpt
    nmlp = sparsify_mlp(model, D, hid, act="drelu", gated=True, topk=0.0, ternary=True)   # foundation recipe
    model = model.to(dev)
    heads = MedusaHeads(D, model.head, max(B-1, 1)).to(dev)                                 # offsets 1..B-1 (co-trained)
    # heads.lm_head IS model.head (tied) -> exclude it from the optimizer list to avoid a duplicate (double-step) param
    head_params = [p for n, p in heads.named_parameters() if not n.startswith("lm_head")] if B > 1 else []
    params = list(model.parameters()) + head_params
    npar = sum(p.numel() for p in params if p.requires_grad)
    print(f"E5.2 CARVE | domain=code B={B} V={V} | params(model+heads)={npar/1e6:.3f}M | dev={dev} amp={amp}")
    print(f"  offsets: 0=base head (t+1) + {max(B-1,0)} co-trained heads (t+2..t+{B}) | GATE: BPB(B=2) <= {GATE_B2:.3f} (anchor {ANCHOR})")

    hook, store = capture_hidden(model)
    def actx(): return torch.autocast(dev_type, dtype=amp) if amp is not None else torch.autocast(dev_type, enabled=False)
    def get_batch():
        ix = np.random.randint(0, len(train)-a.seq-1, size=a.batch)
        return torch.from_numpy(np.stack([train[i:i+a.seq] for i in ix])).long().to(dev)

    def block_bpb(cap):
        """BPB under the block factorization: at every B-th position, predict t+1..t+B from that state. Per-offset.
           Vectorized: one CE per (window, offset) over the block-start positions."""
        model.eval(); bits=[0.0]*B; nb=[0]*B
        with torch.no_grad():
            W=a.seq; lim=min(cap,len(val)-1); pos=0
            while pos+W+1<=lim:
                store.clear(); xt=torch.from_numpy(val[pos:pos+W][None,:]).to(dev)
                lg0=model(xt); h=store[-1]; hl=heads(h) if B>1 else []
                starts=np.arange((-pos) % B, W, B)                    # local i with (pos+i)%B==0
                for j in range(B):
                    valid=[int(i) for i in starts if pos+int(i)+1+j<=lim]
                    if not valid: continue
                    idx=torch.tensor(valid, device=dev)
                    logit=(lg0 if j==0 else hl[j-1])[0].index_select(0, idx)     # (nv,V)
                    tgtids=[int(val[pos+i+1+j]) for i in valid]
                    tgt=torch.tensor(tgtids, device=dev)
                    bits[j]+=F.cross_entropy(logit, tgt, reduction="sum").item()/math.log(2)
                    nb[j]+=int(el[torch.tensor(tgtids)].sum().item())
                pos+=W
        model.train()
        per=[bits[j]/max(nb[j],1) for j in range(B)]
        return sum(bits)/max(sum(nb),1), per

    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9,0.95), weight_decay=0.1)
    print("== training (block-MTP CE, backbone CO-TRAINED) ==")
    model.train()
    for step in range(a.steps):
        ts=time.time(); x=get_batch(); store.clear()
        with actx():
            lg0=model(x); h=store[-1]; hl=heads(h) if B>1 else []
            loss=F.cross_entropy(lg0[:, :a.seq-1, :].reshape(-1,V), x[:, 1:].reshape(-1))
            for j in range(1, B):
                lj=hl[j-1][:, :a.seq-1-j, :]; loss=loss+F.cross_entropy(lj.reshape(-1,V), x[:, 1+j:].reshape(-1))
            loss=loss/B
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()
        if step==0 or (step+1)%max(1,a.steps//10)==0:
            if dev_type=="cuda": torch.cuda.synchronize()
            print(f"  step {step+1:5d}/{a.steps} loss={loss.item():.4f} ({(time.time()-ts)*1000:.0f} ms)")

    bpb, per = block_bpb(a.eval_tok)

    # ---- transfer probes (cheap): sparsity built-ins + in-place ridge recall ----
    cap_x={}; cap_g={}; sh={}; sg={}
    hooks=[]; mlp_layers=[]
    for i,blk in enumerate(model.blocks):
        m=getattr(blk,"mlp",None)
        if m is None: continue
        mlp_layers.append(i); m._cap=True; cap_x[i]=[]; cap_g[i]=[]; sh[i]=[]; sg[i]=[]
        hooks.append(m.register_forward_pre_hook((lambda ii: (lambda mod,inp: cap_x[ii].append(inp[0].detach().float().cpu().numpy())))(i)))
        gm=m.gate if getattr(m,"gated",False) and hasattr(m,"gate") else m.up
        hooks.append(gm.register_forward_hook((lambda ii: (lambda mod,inp,out: cap_g[ii].append((out.detach()>0).cpu().numpy())))(i)))
    model.eval()
    with torch.no_grad():
        for bi in range(a.probe_nbatch):
            p=bi*a.seq
            if p+a.seq+1>len(val): break
            store.clear(); model(torch.from_numpy(val[p:p+a.seq][None,:]).to(dev))
            for i in mlp_layers:
                m=model.blocks[i].mlp; sh[i].append(m._s_h); sg[i].append(m._s_g)
    for hh in hooks: hh.remove()
    for i in mlp_layers: model.blocks[i].mlp._cap=False
    model.train()
    hz=[]; gz=[]; rec=[]
    for i in mlp_layers:
        if not cap_g[i]: continue
        Gm=np.concatenate([g.reshape(-1,g.shape[-1]) for g in cap_g[i]],0)
        Xm=np.concatenate([x.reshape(-1,x.shape[-1]) for x in cap_x[i]],0)
        M=Xm.shape[0]; h=M//2; rr=float("nan")
        if h>D+2 and Gm[:h].sum()>0:
            W=ridge_fit(Xm[:h].astype(np.float64), Gm[:h].astype(np.float64), a.ridge)
            rr=100*recall_at_k(ridge_apply(Xm[h:].astype(np.float64), W), Gm[h:])
        hz.append(float(np.mean(sh[i]))*100); gz.append(float(np.mean(sg[i]))*100); rec.append(rr)

    # ---- greedy block-commit B sample ----
    def greedy_block(n=300, rep=1.2, win=128):
        model.eval(); ctx=[int(t) for t in val[:16]]; out=bytearray()
        with torch.no_grad():
            while len(out)<n:
                store.clear(); xt=torch.tensor(ctx[-a.seq:])[None,:].to(dev)
                lg0=model(xt); hh=store[-1][:, -1, :]; hl=heads(hh) if B>1 else []
                blk=[lg0[0,-1].float()]+[hl[j-1][0].float() for j in range(1,B)]
                for j,logit in enumerate(blk):                       # commit B tokens from THIS state
                    l2=logit.clone(); recent=sorted(set(ctx[-win:])); idx=torch.tensor(recent)
                    sv=l2[idx]; l2[idx]=torch.where(sv>0, sv/rep, sv*rep)
                    tok=int(l2.argmax()); ctx.append(tok); out+=id2b[tok]
        model.train(); return bytes(out[:n])

    print("\n== RESULTS ==")
    print(f"  [BPB] block-factorized (B={B}) val BPB = {bpb:.4f} bits/byte  | anchor(B=1)={ANCHOR}")
    print(f"  [per-offset BPB] " + " ".join(f"t+{j+1}={per[j]:.3f}" for j in range(B)))
    if B==2:
        print(f"  >>> GATE BPB(B=2) <= {GATE_B2:.3f} -> {bpb:.4f} {'PASS (carve VIABLE)' if bpb<=GATE_B2 else 'FAIL'}")
    else:
        print(f"  (B={B} curve; interest B=4 <= {ANCHOR+0.040:.3f})")
    print(f"  [transfer] hidden_zero={np.nanmean(hz):.1f}% gate_zero={np.nanmean(gz):.1f}% (anchor 92/79) | recall_inplace={np.nanmean(rec):.1f}% (anchor 86-92)")
    print(f"  [greedy block-commit B={B}] {greedy_block()[:180]!r}")
    hook.remove()
    if a.save:
        torch.save({"model":model.state_dict(),"heads":heads.state_dict(),"cfg":dict(V=V,**AC,B=B,bpb=bpb,per_offset=per,
                    win=128,expand=2,conv=4,act="drelu",gated=True,mlp_precision="ternary",proj_precision="fp32")}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. E5.2 carve/set-block (design-gate, rule pre-registered). No commit.")

if __name__ == "__main__":
    main()
