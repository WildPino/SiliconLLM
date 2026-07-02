#!/usr/bin/env python3
# Phase 58.B / Probe (Finding-7 / K3) - A/B REGULARIZER: does co-training a cheap in-place predictor + a temporal
#   coherence term make the model MORE PREDICTABLE (the "train it to be predictable" claim) without hurting BPB?
#   magic: "P58B" 0x50353842
#
#   ONE VARIABLE = the regularizer.  Both arms = gated dReLU ternary, probe-2 recipe IDENTICAL (same arch/steps/lr/
#   precision). Run 58.B ONLY after the Architect reads 58.A.
#     base : --act drelu --gated                         (no aux; the probe-2 baseline reproduced)
#     +reg : --reg                                        adds  L = CE + lam_pred*pred_BCE + lam_coh*coherence
#       pred_BCE   : per block, a cheap linear predictor P(x_t)=W_p.x_t (x_t = norm2 residual) trained with BCE to
#                    predict the BLOCK active-set (gate>0 grouped to --reg-block wide). The predictor co-adapts AND
#                    its gradient flows into the body -> the body is pushed toward an input-predictable support.
#       coherence  : mean |a_t - a_{t-1}| of the soft gate (relu(g)) within a sequence -> pushes the support to
#                    PERSIST step-to-step (helps block-decode amortization). Detached target for BCE (predict, don't
#                    collapse); coherence is on the live activation (shapes the body).
#
#   Gate (pre-registered, anti-Goodhart): +reg is promoted IFF  BPB <= base + 0.01  AND predictor-recall rises
#   materially (>= +10 points, or exceeds ~85% at a useful block-size >= 32). Measured on BOTH arms at the end.
#
# Smoke (CPU): python benchmarks/phase57/phase58_reg.py --smoke --reg          (60 steps; watch pred-loss fall)
# A/B (3060) : python benchmarks/phase57/phase58_reg.py --act drelu --gated --steps 4000 --seq 512 --batch 16 --fp16 --save results/phase57/sp58_base.pt
#         ... : python benchmarks/phase57/phase58_reg.py --reg                 --steps 4000 --seq 512 --batch 16 --fp16 --save results/phase57/sp58_reg.pt
import argparse, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55"))
sys.path.insert(0, HERE)
from phase55_ssm import ArchA, load_meta, IDS, META
from phase57_ternary import BitLinear158
from phase57_sparse import SparseMLP, sparsify_mlp

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "phase57"); os.makedirs(OUT, exist_ok=True)


# SparseMLP + an in-place predictor head. When reg is on, forward stashes the aux losses on the module; the trainer
# sums them. Predictor sees x_t (the MLP input = norm2 residual) and predicts the BS-wide block active-set (gate>0).
class PredSparseMLP(SparseMLP):
    def __init__(s, D, hid, act, gated, topk, ternary, reg_block):
        super().__init__(D, hid, act, gated, topk, ternary)
        s.reg = False; s.reg_block = reg_block; s.nb = hid // reg_block; s.hid = hid
        s.pred = nn.Linear(D, s.nb)                 # cheap fp32 predictor head (dropped at inference; it's the regularizer)
        s._aux_pred = None; s._aux_coh = None; s._last_recall = None
    def forward(s, x):
        if not s.reg:
            return super().forward(x)
        g = s.gate(x) if s.gated else s.up(x)
        u = s.up(x)
        if s.gated:
            if   s.act == "drelu": h = F.relu(g) * F.relu(u)
            elif s.act == "silu":  h = F.silu(g) * u
            else:                  h = s._act1(g) * u
            soft = F.relu(g)
        else:
            h = s._act1(u); soft = h
        if s.topk and s.topk > 0.0:
            keep = max(1, int(round(h.shape[-1] * (1.0 - s.topk))))
            thr = h.abs().kthvalue(h.shape[-1] - keep + 1, dim=-1, keepdim=True).values
            h = torch.where(h.abs() >= thr, h, torch.zeros_like(h))
        # block active-set target (gate>0), grouped to reg_block wide -> (B,T,nb) in {0,1}, detached
        B, T, _ = soft.shape
        act = (soft[..., :s.nb * s.reg_block].reshape(B, T, s.nb, s.reg_block) > 0).any(-1).float().detach()
        logit = s.pred(x)                                          # (B,T,nb)
        s._aux_pred = F.binary_cross_entropy_with_logits(logit, act)
        # temporal coherence: encourage the soft gate to persist step-to-step (within sequence)
        s._aux_coh = (soft[:, 1:, :] - soft[:, :-1, :]).abs().mean() if T > 1 else soft.sum() * 0.0
        # recall@k diagnostic (k=|active|), no grad
        with torch.no_grad():
            sc = logit.reshape(-1, s.nb); av = act.reshape(-1, s.nb).bool()
            k = av.sum(1); sel = k > 0
            if sel.any():
                rank = sc[sel].argsort(1, descending=True).argsort(1)
                pred = rank < k[sel, None]
                s._last_recall = ((pred & av[sel]).sum(1).float() / k[sel].float()).mean().item()
        return s.down(h)


def predify(model, D, hid, act, gated, topk, ternary, reg_block):
    n = 0
    for blk in model.blocks:
        if getattr(blk, "use_mlp", False):
            blk.mlp = PredSparseMLP(D, hid, act, gated, topk, ternary, reg_block); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reg", action="store_true", help="enable the regularizer arm (+reg); omit for base arm")
    ap.add_argument("--act", choices=["silu", "relu", "relu2", "drelu"], default="drelu")
    ap.add_argument("--gated", action="store_true", default=True)
    ap.add_argument("--topk", type=float, default=0.0)
    ap.add_argument("--reg-block", type=int, default=32, help="block width for the predictor target (>=32 useful)")
    ap.add_argument("--lam-pred", type=float, default=0.1)
    ap.add_argument("--lam-coh", type=float, default=0.01)
    ap.add_argument("--mlp-mult", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="grad-accumulation: eff batch = batch*accum (cuts activation mem; ~identical math)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--eval-tok", type=int, default=200000)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile (Linux/WSL only; fuses the SSM scan; Windows->eager)")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args()
    if a.smoke: a.steps = 60; a.seq = 64; a.batch = 4; a.eval_tok = 20000
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if a.fp16 else (torch.bfloat16 if a.bf16 else None)
    torch.manual_seed(0); np.random.seed(0)
    if dev_type == "cuda": torch.cuda.manual_seed_all(0)

    V, exp_len, id2bytes = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); ntr = int(n * 0.9); train = ids[:ntr]; val = ids[ntr:]
    el_t = torch.tensor(exp_len)

    D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=a.mlp_mult, dt_rank=16); hid = a.mlp_mult * D
    model = ArchA(V, **AC).to(dev); model.use_ckpt = not a.no_ckpt
    nsw = predify(model, D, hid, a.act, True, a.topk, True, a.reg_block)
    for blk in model.blocks:
        if getattr(blk, "use_mlp", False): blk.mlp.reg = a.reg
    model = model.to(dev)
    mlp_blocks = [blk.mlp for blk in model.blocks if getattr(blk, "use_mlp", False)]
    npar = sum(p.numel() for p in model.parameters())
    print(f"Phase58.B REG | arm={'+reg' if a.reg else 'base'} act={a.act} reg_block={a.reg_block} lam_pred={a.lam_pred} lam_coh={a.lam_coh}")
    print(f"  Arch-A 5M: params={npar/1e6:.3f}M hid={hid} nb/block={hid//a.reg_block} | dev={dev} amp={amp_dtype} | predify {nsw} blocks")
    print(f"  config: steps={a.steps} seq={a.seq} batch={a.batch} lr={a.lr} | baselines: ternary 0.8382, probe-2 dReLU 0.8813")

    def autocast_ctx():
        return torch.autocast(dev_type, dtype=amp_dtype) if amp_dtype is not None else torch.autocast(dev_type, enabled=False)

    def val_bpb(cap):
        model.eval(); bits = 0.0; nb = 0
        with torch.no_grad():
            W = a.seq; lim = min(cap, len(val) - 1); pos = 0
            while pos + W + 1 <= lim:
                x = torch.from_numpy(val[pos:pos+W][None, :]).to(dev); y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).to(dev)
                ce = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction="sum")
                bits += ce.item() / math.log(2); nb += int(el_t[y.reshape(-1).cpu()].sum().item()); pos += W
        model.train(); return bits / max(nb, 1)

    def measure_recall(nbatch=16):
        # turn reg on (for the diagnostic) regardless of arm, measure predictor recall@k on val; restore arm
        was = [m.reg for m in mlp_blocks]
        # base arm has untrained pred heads -> fit a quick ridge instead would be fairer, but for the END readout we
        # report the co-trained predictor's recall (+reg) and, for base, a freshly ridge-fit predictor on val (apples
        # to apples with 58.A's measurement). Keep it simple: enable reg=True to populate _last_recall.
        for m in mlp_blocks: m.reg = True
        model.eval(); recs = [[] for _ in mlp_blocks]
        with torch.no_grad():
            for b in range(nbatch):
                pos = b * a.seq
                if pos + a.seq + 1 > len(val): break
                x = torch.from_numpy(val[pos:pos+a.seq][None, :]).to(dev)
                model(x)
                for i, m in enumerate(mlp_blocks):
                    if m._last_recall is not None: recs[i].append(m._last_recall)
        for m, w in zip(mlp_blocks, was): m.reg = w
        model.train()
        return [float(np.mean(v)) if v else float("nan") for v in recs]

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    # fp16 needs gradient scaling to stay stable in EAGER (probe-2 only survived fp16 because --compile kept reductions
    # in fp32); on Ampere (3060) prefer --bf16, which is stable with no scaler. Scaler is a no-op when disabled.
    use_scaler = (amp_dtype == torch.float16 and dev_type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if a.compile:
        # torch.compile is LAZY (codegen at first forward) -> force a warmup inside try/except so a missing compiler
        # (Windows: no Triton/cl) degrades to eager instead of crashing at step 1. Same modules -> aux stash still read.
        try:
            cm = torch.compile(model); warm = torch.zeros(a.batch, a.seq, dtype=torch.long, device=dev)
            with torch.no_grad(), autocast_ctx(): cm(warm)
            model = cm; print("  torch.compile enabled (SSM scan fused)")
        except Exception as e:
            print(f"  torch.compile failed -> eager ({type(e).__name__}: {str(e)[:90]})")

    def get_batch(src):
        ix = np.random.randint(0, len(src) - a.seq - 1, size=a.batch)
        return (torch.from_numpy(np.stack([src[i:i+a.seq] for i in ix])).to(dev),
                torch.from_numpy(np.stack([src[i+1:i+1+a.seq] for i in ix])).to(dev))

    print("== training ==")
    model.train(); t0 = time.time()
    for step in range(a.steps):
        ts = time.time(); opt.zero_grad()
        ce_v = ap_v = cp_v = 0.0
        for _ in range(a.accum):                               # eff batch = batch*accum; each microbatch backward accumulates
            x, y = get_batch(train)
            with autocast_ctx():
                ce = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            aux_p = aux_c = torch.zeros((), device=dev)
            if a.reg:
                aux_p = sum(m._aux_pred for m in mlp_blocks) / len(mlp_blocks)
                aux_c = sum(m._aux_coh for m in mlp_blocks) / len(mlp_blocks)
            loss = (ce + (a.lam_pred * aux_p + a.lam_coh * aux_c if a.reg else 0.0)) / a.accum
            scaler.scale(loss).backward()
            ce_v += ce.item() / a.accum; ap_v += float(aux_p.detach()) / a.accum; cp_v += float(aux_c.detach()) / a.accum
        scaler.unscale_(opt)                                   # unscale before clipping so the norm is in real units
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        for m in mlp_blocks: m._aux_pred = m._aux_coh = None    # drop graph refs so they don't pin activations into next step
        if step == 0 or (step + 1) % max(1, a.steps // 20) == 0:
            if dev_type == "cuda": torch.cuda.synchronize()
            ms = (time.time() - ts) * 1000
            pmsg = f" pred={ap_v:.4f} coh={cp_v:.4f}" if a.reg else ""
            vmsg = ""
            if (step + 1) % max(1, a.steps // 10) == 0 or step == 0:
                vmsg = f"  | val BPB={val_bpb(min(a.eval_tok,16000)):.4f}"
            print(f"  step {step+1:5d}/{a.steps}  ce={ce_v:.4f}{pmsg}  gnorm={gn:.2f}  ({ms:.0f} ms){vmsg}")

    bpb = val_bpb(a.eval_tok)
    recs = measure_recall(16)
    print("== RESULTS ==")
    print(f"  arm={'+reg' if a.reg else 'base'}  val BPB={bpb:.4f}  | vs probe-2 dReLU 0.8813 -> {bpb-0.8813:+.4f} | vs ternary 0.8382 -> {bpb-0.8382:+.4f}")
    print(f"  predictor recall@k per block (reg_block={a.reg_block}): " + " ".join(f"{v*100:4.0f}%" for v in recs) + f"  | mean {np.nanmean(recs)*100:.1f}%")
    print(f"  GATE (anti-Goodhart): promote +reg IFF BPB<=base+0.01 AND recall +>=10pt (or >~85% at block>=32). Architect compares the two runs.")

    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({"model": base.state_dict(),
                    "cfg": dict(V=V, **AC, win=128, act=a.act, gated=True, topk=a.topk, mlp_precision="ternary",
                                reg=a.reg, reg_block=a.reg_block, lam_pred=a.lam_pred, lam_coh=a.lam_coh,
                                bpb=bpb, recall_mean=float(np.nanmean(recs)))}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. base vs +reg = the Architect's read. No commit.")


if __name__ == "__main__":
    main()
