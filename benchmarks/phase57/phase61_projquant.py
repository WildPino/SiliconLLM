#!/usr/bin/env python3
# Phase 61 / probe (model-side) - TERNARIZE THE SSM PROJECTIONS. magic: "P61P" 0x50363150
#   Motive: in the C engine (Phase 60 E3.5) the dense fp32 SSM-projection GEMVs (in_proj/x_proj/dt_proj/out_proj) are
#   52.7% of per-token time, and at scale-up they are a large resident-pool byte share. The probe-1 mixed-precision
#   recipe kept them fp32 out of caution -- NEVER measured whether that caution is needed. This A/B measures it.
#
#   ONE VARIABLE = the SSM-projection precision. Everything else = the validated sp58_base recipe (gated-dReLU-TERNARY
#   MLP via SparseMLP; SWA/head/emb/norms fp32). Base arm (proj fp32) reproduces the sp58_base anchor (BPB ~0.8799 @4k).
#     --proj-precision fp32           : base (SSM projections fp32)                 [anchor]
#     --proj-precision ternary-all    : ARM A - all 4 projections ternary (BitLinear158, STE, per-row absmean)
#     --proj-precision ternary-inout  : ARM B - in_proj + out_proj ternary, x_proj + dt_proj fp32 (77% of proj bytes;
#                                       hypothesis: the dt/x path IS the SSM selectivity = the sensitive part)
#   dt_proj carries a bias -> ternarize the WEIGHT (STE), keep the bias fp32 (TernLinear handles it).
#
#   Pre-registered gate (Architect): BPB(arm) <= 0.8799 + 0.010 = 0.8899 @4k matched. If A passes -> next engine stage
#   brings the projections onto the LUT path (kernel-exact + END-TO-END parity, the E4 law). If both A and B fail ->
#   projections stay fp32 and the engine lever becomes GEMV vectorization (pure engineering, no quality question).
#
# Smoke (CPU): python benchmarks/phase57/phase61_projquant.py --smoke --proj-precision ternary-all
# A/B  (3060): python benchmarks/phase57/phase61_projquant.py --proj-precision fp32          --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase57/p61_base.pt
#         ... : python benchmarks/phase57/phase61_projquant.py --proj-precision ternary-all   --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase57/p61_all.pt
#         ... : python benchmarks/phase57/phase61_projquant.py --proj-precision ternary-inout --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase57/p61_inout.pt
import argparse, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55")); sys.path.insert(0, HERE)
from phase55_ssm import ArchA, load_meta, IDS, META
from phase57_ternary import BitLinear158
from phase57_sparse import sparsify_mlp

ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); OUT = os.path.join(ROOT, "results", "phase57"); os.makedirs(OUT, exist_ok=True)


class TernLinear(nn.Module):
    """BitLinear158-style ternary weight (STE, per-row absmean scale) + OPTIONAL fp32 bias (for dt_proj)."""
    def __init__(s, lin):
        super().__init__()
        s.weight = nn.Parameter(lin.weight.data.clone())
        s.bias = nn.Parameter(lin.bias.data.clone()) if lin.bias is not None else None
        s.in_features, s.out_features = lin.in_features, lin.out_features
    def forward(s, x):
        w = s.weight
        scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
        wq = (w / scale).round().clamp(-1, 1)
        w_ste = w + (wq * scale - w).detach()
        return F.linear(x, w_ste, s.bias)
    @torch.no_grad()
    def zero_frac(s):
        w = s.weight; scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
        return float(((w / scale).round().clamp(-1, 1) == 0).float().mean())


# byte weights of each projection (D=256, Dn=512, dt_rank=16, N=96): in 262144, x 106496, dt 8192, out 131072
PROJ_ORDER = ["in_proj", "x_proj", "dt_proj", "out_proj"]
def ternarize_ssm_proj(model, mode):
    which = {"fp32": [], "ternary-all": PROJ_ORDER, "ternary-inout": ["in_proj", "out_proj"]}[mode]
    n = 0
    for blk in model.blocks:
        mix = blk.mix
        if not hasattr(mix, "in_proj"):  # SWA block: skip
            continue
        for name in which:
            setattr(mix, name, TernLinear(getattr(mix, name))); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--proj-precision", choices=["fp32", "ternary-all", "ternary-inout"], default="ternary-all")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--bf16", action="store_true"); ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--eval-tok", type=int, default=200000); ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--save", type=str, default=""); ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args()
    if a.smoke: a.steps = 60; a.seq = 64; a.batch = 4; a.eval_tok = 20000
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    amp = torch.float16 if a.fp16 else (torch.bfloat16 if a.bf16 else None)
    torch.manual_seed(0); np.random.seed(0)
    if dev_type == "cuda": torch.cuda.manual_seed_all(0)

    V, exp_len, id2b = load_meta(META); ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); ntr = int(n * 0.9); train = ids[:ntr]; val = ids[ntr:]; el = torch.tensor(exp_len)
    D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16); hid = 4 * D
    model = ArchA(V, **AC).to(dev); model.use_ckpt = not a.no_ckpt
    nmlp = sparsify_mlp(model, D, hid, act="drelu", gated=True, topk=0.0, ternary=True)   # sp58_base MLP recipe
    nproj = ternarize_ssm_proj(model, a.proj_precision)
    model = model.to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"Phase61 PROJ-QUANT | proj_precision={a.proj_precision} | ternarized {nproj} SSM projections | MLP={nmlp} drelu-gated-ternary")
    print(f"  Arch-A 5M: params={npar/1e6:.3f}M | dev={dev} amp={amp} | anchor: sp58_base fp32-proj BPB 0.8799 | GATE <= 0.8899")
    print(f"  config: steps={a.steps} seq={a.seq} batch={a.batch} accum={a.accum} lr={a.lr}")

    def actx(): return torch.autocast(dev_type, dtype=amp) if amp is not None else torch.autocast(dev_type, enabled=False)
    def val_bpb(cap):
        model.eval(); bits = 0.0; nb = 0
        with torch.no_grad():
            W = a.seq; lim = min(cap, len(val) - 1); pos = 0
            while pos + W + 1 <= lim:
                x = torch.from_numpy(val[pos:pos+W][None, :]).to(dev); y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).to(dev)
                bits += F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction="sum").item()/math.log(2)
                nb += int(el[y.reshape(-1).cpu()].sum().item()); pos += W
        model.train(); return bits / max(nb, 1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    def get_batch(src):
        ix = np.random.randint(0, len(src)-a.seq-1, size=a.batch)
        return (torch.from_numpy(np.stack([src[i:i+a.seq] for i in ix])).to(dev),
                torch.from_numpy(np.stack([src[i+1:i+1+a.seq] for i in ix])).to(dev))
    print("== training (CE only) ==")
    model.train()
    for step in range(a.steps):
        ts = time.time(); opt.zero_grad(); lsum = 0.0
        for _ in range(a.accum):
            x, y = get_batch(train)
            with actx(): loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))/a.accum
            loss.backward(); lsum += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 0 or (step+1) % max(1, a.steps//20) == 0:
            if dev_type == "cuda": torch.cuda.synchronize()
            ms = (time.time()-ts)*1000; vmsg = ""
            if (step+1) % max(1, a.steps//10) == 0 or step == 0: vmsg = f"  | val BPB={val_bpb(min(a.eval_tok,16000)):.4f}"
            print(f"  step {step+1:5d}/{a.steps} loss={lsum:.4f} gnorm={gn:.2f} ({ms:.0f} ms){vmsg}")

    bpb = val_bpb(a.eval_tok)
    print("== RESULTS ==")
    print(f"  proj_precision={a.proj_precision} | val BPB={bpb:.4f} | vs anchor 0.8799 -> {bpb-0.8799:+.4f} | GATE(<=0.8899) {'PASS' if bpb<=0.8899 else 'FAIL'}")
    if a.proj_precision != "fp32":
        zf = {n: [] for n in PROJ_ORDER}
        for blk in model.blocks:
            for name in PROJ_ORDER:
                m = getattr(blk.mix, name, None)
                if isinstance(m, TernLinear): zf[name].append(m.zero_frac())
        print("  ternary zero-frac per proj: " + " ".join(f"{n}={np.mean(v):.2f}" for n, v in zf.items() if v))
    if a.save:
        torch.save({"model": model.state_dict(), "cfg": dict(V=V, **AC, win=128, expand=2, conv=4, seq=a.seq, steps=a.steps,
                    bpb=bpb, proj_precision=a.proj_precision, mlp_precision="ternary", act="drelu", gated=True)}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. BPB above (proj-precision A/B). No commit.")

if __name__ == "__main__":
    main()
