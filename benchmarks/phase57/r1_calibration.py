#!/usr/bin/env python3
# R1 / CALIBRATION (not an experiment with a gate). magic: "R1CL" 0x5231434C
#   Two questions, both answered WITHIN THIS ONE SCRIPT (all runs same code path):
#     (1) sigma_seed : how much does the foundation recipe's final BPB move when ONLY the seed changes?
#                      (prices every past/future delta in the project; prior hint ~0.004 cross-script.)
#     (2) combined-cost anchor : in one clean A/B, what does the foundation recipe
#                      (ternary MLP + dReLU) cost vs an fp32+SiLU baseline of identical structure/size?
#
#   NO GATE. This is calibration: we report the numbers, we do not adjust them, we draw no conclusion.
#
#   --recipe foundation : sp58_base EXACTLY. MLP gated (gate/up/down, hid=1024), act dReLU, weights TERNARY
#                         (BitLinear158, STE, per-row absmean). SSM/SWA/head/emb/PROJECTIONS fp32.
#   --recipe fp32silu   : SAME gated structure, SAME dims (param-identical), but act SiLU and MLP fp32.
#                         DELIBERATE: two variables move together (precision + activation) because this is the
#                         COMBINED-cost anchor; the structure stays constant.
#   --seed N            : seeds torch / cuda / numpy AND the data order (get_batch draws). Foundation seed was
#                         hardcoded 0 historically; this exposes it.
#
#   Config identical to the historical runs: --steps 4000 --seq 512 --batch 16 --lr 0.003 --bf16.
#
# Smoke (CPU): python benchmarks/phase57/r1_calibration.py --smoke --recipe foundation
#         ...  python benchmarks/phase57/r1_calibration.py --smoke --recipe fp32silu
# R1 (3060), 4 sequential runs (the Capo launches):
#   python benchmarks/phase57/r1_calibration.py --recipe foundation --seed 0 --steps 4000 --seq 512 --batch 16 --lr 0.003 --bf16 --save results/phase57/r1_found_s0.pt
#   python benchmarks/phase57/r1_calibration.py --recipe foundation --seed 1 --steps 4000 --seq 512 --batch 16 --lr 0.003 --bf16 --save results/phase57/r1_found_s1.pt
#   python benchmarks/phase57/r1_calibration.py --recipe foundation --seed 2 --steps 4000 --seq 512 --batch 16 --lr 0.003 --bf16 --save results/phase57/r1_found_s2.pt
#   python benchmarks/phase57/r1_calibration.py --recipe fp32silu   --seed 0 --steps 4000 --seq 512 --batch 16 --lr 0.003 --bf16 --save results/phase57/r1_fp32silu_s0.pt
import argparse, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55")); sys.path.insert(0, HERE)
from phase55_ssm import ArchA, load_meta, IDS, META
from phase57_sparse import sparsify_mlp                # sp58_base MLP recipe (gated dReLU, STE ternary)

ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); OUT = os.path.join(ROOT, "results", "phase57"); os.makedirs(OUT, exist_ok=True)

# recipe -> (act, gated, ternary). Projections stay fp32 for BOTH (foundation keeps SSM proj fp32).
RECIPE = {"foundation": dict(act="drelu", gated=True, ternary=True),
          "fp32silu":   dict(act="silu",  gated=True, ternary=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--recipe", choices=["foundation", "fp32silu"], default="foundation")
    ap.add_argument("--seed", type=int, default=0, help="seeds torch/cuda/numpy AND data order")
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
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if dev_type == "cuda": torch.cuda.manual_seed_all(a.seed)
    rng = np.random.default_rng(a.seed)                # data-order RNG, seed-scoped (independent of any global use)

    V, exp_len, id2b = load_meta(META); ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); ntr = int(n * 0.9); train = ids[:ntr]; val = ids[ntr:]; el = torch.tensor(exp_len)
    D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16); hid = 4 * D
    model = ArchA(V, **AC).to(dev); model.use_ckpt = not a.no_ckpt
    r = RECIPE[a.recipe]
    nmlp = sparsify_mlp(model, D, hid, act=r["act"], gated=r["gated"], topk=0.0, ternary=r["ternary"])
    model = model.to(dev)
    npar = sum(p.numel() for p in model.parameters())
    mlp_prec = "ternary" if r["ternary"] else "fp32"
    print(f"R1-CALIB | recipe={a.recipe} seed={a.seed} | MLP={nmlp} gated act={r['act']} precision={mlp_prec} | proj=fp32 (both)")
    print(f"  Arch-A 5M: params={npar/1e6:.3f}M hid={hid} | dev={dev} amp={amp} | CALIBRATION: no gate, report raw")
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
        ix = rng.integers(0, len(src)-a.seq-1, size=a.batch)
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
    print(f"  recipe={a.recipe} seed={a.seed} | val BPB={bpb:.4f}   (calibration: raw, no gate, no adjustment)")
    if a.save:
        torch.save({"model": model.state_dict(), "cfg": dict(V=V, **AC, win=128, expand=2, conv=4, seq=a.seq, steps=a.steps,
                    bpb=bpb, recipe=a.recipe, seed=a.seed, act=r["act"], gated=r["gated"], mlp_precision=mlp_prec, proj_precision="fp32")}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. BPB above. No conclusion, no commit (that is the Architect / Media Manager).")

if __name__ == "__main__":
    main()
