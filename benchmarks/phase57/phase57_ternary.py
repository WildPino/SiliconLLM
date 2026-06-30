#!/usr/bin/env python3
# Phase 57 / Probe-1 (1a) - QUALITA: does NATIVE ternary (1.58-bit) MLP keep Arch-A's quality?
#   magic: "TERN" 0x5445524E
#
#   BitNet b1.58 style, FROM SCRATCH (NOT post-training quant): the MLP linears are ternarized {-1,0,+1} with a
#   straight-through estimator + per-row (or per-tensor) absmean scale. From-scratch dodges the ~2.4bpw PTQ floor.
#   MIXED PRECISION: SSM / SWA / head / embedding / norms stay fp32 (the sensitive parts); ONLY the per-block MLP
#   (the byte-sink) is ternarized. ONE VARIABLE = MLP precision (fp32 vs ternary).
#
#   Target = the 5M Arch-A (the only Arch-A variant WITH an MLP). fp32 reference = results/phase55/archA_5m.pt
#   (recorded val BPB 0.8104). For a clean one-variable A/B, train BOTH precisions with THIS script/recipe
#   (--mlp-precision fp32  and  --mlp-precision ternary), same steps/seq/lr.
#
#   Activations are kept fp here (weight-only ternary): isolates "MLP weight precision" as the single variable,
#   and matches the 1b kernel (ternary W x fp/int8 activations). No commit; smoke + STOP; the Capo launches the GPU train.
#
# Smoke (CPU): python benchmarks/phase57/phase57_ternary.py --smoke --mlp-precision ternary
# Real  (GPU): python benchmarks/phase57/phase57_ternary.py --mlp-precision ternary --steps 10785 --seq 512 --batch 16 --bf16 --save results/phase57/archA_5m_ternmlp.pt
import argparse, math, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase55"))
from phase55_ssm import ArchA, load_meta, byte_guard, word_metrics, IDS, META, OUT as P55OUT  # reuse frozen baseline pieces

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT  = os.path.join(ROOT, "results", "phase57"); os.makedirs(OUT, exist_ok=True)

# ---------------- ternary MLP linear (BitNet b1.58, weight-only, STE) ----------------
class BitLinear158(nn.Module):
    def __init__(s, fin, fout, per_row=True):
        super().__init__()
        s.in_features, s.out_features = fin, fout
        s.weight = nn.Parameter(torch.empty(fout, fin)); nn.init.kaiming_uniform_(s.weight, a=math.sqrt(5))
        s.per_row = per_row
    def forward(s, x):
        w = s.weight
        scale = (w.abs().mean(dim=1, keepdim=True) if s.per_row else w.abs().mean()).clamp_min(1e-5)
        wq = (w / scale).round().clamp(-1, 1)            # ternary {-1,0,+1}
        w_ste = w + (wq * scale - w).detach()            # forward=wq*scale, backward=identity (STE)
        return F.linear(x, w_ste)
    @torch.no_grad()
    def ternary_stats(s):
        w = s.weight; scale = (w.abs().mean(dim=1, keepdim=True) if s.per_row else w.abs().mean()).clamp_min(1e-5)
        wq = (w / scale).round().clamp(-1, 1)
        return float((wq == 0).float().mean())           # sparsity (fraction of zeros)

def ternarize_mlp(model, per_row=True):
    swapped = 0
    for blk in model.blocks:
        if getattr(blk, "use_mlp", False):
            for name in ("fc1", "fc2"):
                lin = getattr(blk.mlp, name)
                setattr(blk.mlp, name, BitLinear158(lin.in_features, lin.out_features, per_row))
                swapped += 1
    return swapped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mlp-precision", choices=["fp32", "ternary"], default="ternary")
    ap.add_argument("--scale", choices=["per-row", "per-tensor"], default="per-row")
    ap.add_argument("--steps", type=int, default=10785)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--eval-tok", type=int, default=200000)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (fuses the SSM scan; first step compiles -> slow)")
    ap.add_argument("--time-budget-min", type=float, default=0.0, help="stop training when wall-clock exceeds this (T4 6h guard)")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args()
    if a.smoke: a.steps = 40; a.seq = 64; a.batch = 4; a.eval_tok = 20000
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    if dev_type == "cuda": torch.cuda.manual_seed_all(0)

    V, exp_len, id2bytes = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); ntr = int(n * 0.9); train = ids[:ntr]; val = ids[ntr:]
    el_t = torch.tensor(exp_len)

    AC = dict(D=256, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16)  # the 5M Arch-A
    model = ArchA(V, **AC).to(dev); model.use_ckpt = not a.no_ckpt
    nsw = ternarize_mlp(model, per_row=(a.scale == "per-row")) if a.mlp_precision == "ternary" else 0
    model = model.to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    nmlp = sum(p.numel() for blk in model.blocks if getattr(blk, "use_mlp", False) for p in blk.mlp.parameters())
    print(f"Phase57 ternary-MLP | mlp_precision={a.mlp_precision} scale={a.scale} | swapped {nsw} MLP linears -> BitLinear158")
    print(f"  Arch-A 5M: params={nparam/1e6:.3f}M (MLP={nmlp/1e6:.3f}M = {100*nmlp/nparam:.0f}% of weights, the byte-sink) | V={V} dev={dev} ckpt={model.use_ckpt} bf16={a.bf16}")
    print(f"  config: steps={a.steps} seq={a.seq} batch={a.batch} accum={a.accum} lr={a.lr} | fp32 ref = archA_5m.pt BPB 0.8104")

    def val_bpb(cap):
        model.eval(); bits = 0.0; nb = 0; nt = 0
        with torch.no_grad():
            W = a.seq; lim = min(cap, len(val) - 1); pos = 0
            while pos + W + 1 <= lim:
                x = torch.from_numpy(val[pos:pos+W][None, :]).to(dev); y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).to(dev)
                ce = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction="sum")
                bits += ce.item() / math.log(2); nb += int(el_t[y.reshape(-1).cpu()].sum().item()); nt += W; pos += W
        model.train(); return bits / max(nb, 1), nt, nb

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)  # opt on raw params (before compile)
    base = model                                # underlying module (clean keys for save / stats), even if compiled
    if a.compile:
        try:
            cm = torch.compile(model)
            warm = torch.zeros(a.batch, a.seq, dtype=torch.long, device=dev)  # force codegen NOW (compile is lazy) so failures are catchable
            with torch.no_grad():
                if a.bf16:
                    with torch.autocast(dev_type, dtype=torch.bfloat16): cm(warm)
                else: cm(warm)
            model = cm; print("  torch.compile enabled (compiled at warmup)")
        except Exception as e:
            print(f"  torch.compile failed -> running eager ({type(e).__name__}: {str(e)[:120]})")
    def get_batch(src):
        ix = np.random.randint(0, len(src) - a.seq - 1, size=a.batch)
        x = np.stack([src[i:i+a.seq] for i in ix]); y = np.stack([src[i+1:i+1+a.seq] for i in ix])
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    print("== training (CE only) ==")
    model.train(); t_start = time.time(); stopped = a.steps
    for step in range(a.steps):
        ts = time.time(); opt.zero_grad(); lsum = 0.0
        for _ in range(a.accum):
            x, y = get_batch(train)
            if a.bf16:
                with torch.autocast(dev_type, dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)) / a.accum
            else:
                loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)) / a.accum
            loss.backward(); lsum += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0 or (step + 1) % max(1, a.steps // 20) == 0:
            if dev_type == "cuda": torch.cuda.synchronize()
            ms = (time.time() - ts) * 1000; vmsg = ""
            if (step + 1) % max(1, a.steps // 10) == 0 or step == 0:
                vb, _, _ = val_bpb(min(a.eval_tok, 16000)); vmsg = f"  | val BPB={vb:.4f}"
            print(f"  step {step+1:5d}/{a.steps}  loss={lsum:.4f}  gnorm={gn:.2f}  bits/tok={lsum/math.log(2):.4f}  ({ms:.0f} ms){vmsg}")
        if a.time_budget_min > 0 and (time.time() - t_start) / 60.0 > a.time_budget_min:
            print(f"  [time-budget {a.time_budget_min:.0f}min hit at step {step+1}] stopping early"); stopped = step + 1; break

    print("== val BPB (held-out, bits/byte) ==")
    bpb, ntok, nby = val_bpb(a.eval_tok)
    print(f"  val BPB={bpb:.4f}  (over {ntok} tok / {nby} bytes)  | fp32 ref 0.8104 -> delta {bpb-0.8104:+.4f}")
    if a.mlp_precision == "ternary":
        sps = [blk.mlp.fc1.ternary_stats() for blk in base.blocks if getattr(blk, "use_mlp", False)]
        print(f"  ternary MLP fc1 sparsity (frac zeros) per block: " + " ".join(f"{s:.2f}" for s in sps))

    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"model": base.state_dict(),
                    "cfg": dict(V=V, **AC, win=128, expand=2, conv=4, seq=a.seq, steps=stopped, bpb=bpb,
                                mlp_precision=a.mlp_precision, scale=a.scale)}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. val BPB above (read fp32 vs ternary delta). No commit.")

if __name__ == "__main__":
    main()
