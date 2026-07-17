#!/usr/bin/env python3
# Inventor / S1c - APPARATUS for the pre-registered from-scratch structured-x_proj A/B
# (brief: docs/in_research/INVENTORE_02_S1_PREREG_BRIEF.md - DRAFT until owner seals it).
#
# ONE VARIABLE = the x_proj parameterization: dense (208x512, control) vs low-rank U.V at r in {52, 26}.
# Everything else = the sp58-class foundation recipe (8.3M Arch-A, gated-dReLU ternary MLP, fp32 organs),
# phase57 training recipe, same seed/data order for every arm -> within-script matched comparisons (R1 rule).
#
# Gates (sealed in the brief, BEFORE numbers): G1 BPB(r52) <= BPB(ctl)+0.005; gray +0.005..0.010 -> 3 seeds.
# r26 = curve only. Secondary recorded: sparsity bands, top-1 vs control on a fixed val slice.
#
# Smoke (CPU):  python benchmarks/in_research/s1c_structured_xproj.py --smoke --xproj-rank 52
# Real (3060):  see the brief's ready commands (owner launches; Builder STOPs here).
import argparse, math, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase57"))
from phase55_ssm import ArchA, load_meta, IDS, META      # noqa: E402
from phase57_sparse import sparsify_mlp                  # noqa: E402

OUT = os.path.join(ROOT, "results", "phase57")


class LowRankLinear(nn.Module):
    """y = U(V(x)), bias-free (matches x_proj). Params r*(fin+fout) vs fin*fout."""
    def __init__(s, fin, fout, r):
        super().__init__()
        s.Vl = nn.Linear(fin, r, bias=False)
        s.Ul = nn.Linear(r, fout, bias=False)
    def forward(s, x):
        return s.Ul(s.Vl(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--xproj-rank", type=int, default=0, help="0 = dense control; 52 / 26 = low-rank arms")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-tok", type=int, default=200000)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args()
    if a.smoke: a.steps, a.seq, a.batch, a.eval_tok = 40, 64, 4, 20000
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if dev.startswith("cuda") else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if dev_type == "cuda": torch.cuda.manual_seed_all(a.seed)

    V, exp_len, _ = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    ntr = int(len(ids) * 0.9); train, val = ids[:ntr], ids[ntr:]
    el_t = torch.tensor(exp_len)

    AC = dict(D=256, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16)
    model = ArchA(V, **AC)
    sparsify_mlp(model, AC["D"], 4 * AC["D"], act="drelu", gated=True, topk=0.0, ternary=True)
    nsw = 0
    if a.xproj_rank > 0:
        for blk in model.blocks:
            mix = blk.mix
            if hasattr(mix, "x_proj"):
                fin, fout = mix.x_proj.in_features, mix.x_proj.out_features
                mix.x_proj = LowRankLinear(fin, fout, a.xproj_rank); nsw += 1
    model = model.to(dev); model.use_ckpt = True
    npar = sum(p.numel() for p in model.parameters())
    dense_x, lr_x = 208 * 512, a.xproj_rank * (208 + 512)
    print(f"S1c | arm={'ctl-dense' if a.xproj_rank == 0 else f'lowrank-r{a.xproj_rank}'} | swapped {nsw} x_proj"
          f" | params={npar/1e6:.3f}M | xproj bytes {100*(lr_x/dense_x if a.xproj_rank else 1):.1f}% of dense"
          f" | seed={a.seed} dev={dev} bf16={a.bf16}")
    print(f"  recipe: steps={a.steps} seq={a.seq} batch={a.batch} lr={a.lr} (phase57 recipe, matched across arms)")

    def val_bpb(cap):
        model.eval(); bits = 0.0; nb = 0; tops = []
        with torch.no_grad():
            W = a.seq; pos = 0
            while pos + W + 1 <= min(cap, len(val) - 1):
                x = torch.from_numpy(val[pos:pos+W][None, :]).to(dev)
                y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).to(dev)
                lg = model(x)
                bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
                nb += int(el_t[y.reshape(-1).cpu()].sum().item())
                if pos < 10 * W: tops.append(lg.reshape(-1, V).argmax(-1).cpu().numpy())  # fixed 10-window top1 slice
                pos += W
        model.train(); return bits / max(nb, 1), (np.concatenate(tops) if tops else np.zeros(0, np.int64))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    def get_batch():
        ix = np.random.randint(0, len(train) - a.seq - 1, size=a.batch)
        return (torch.from_numpy(np.stack([train[i:i+a.seq] for i in ix])).to(dev),
                torch.from_numpy(np.stack([train[i+1:i+1+a.seq] for i in ix])).to(dev))

    print("== training (CE only) ==")
    model.train(); t0 = time.time()
    for step in range(a.steps):
        ts = time.time(); opt.zero_grad(); lsum = 0.0
        for _ in range(a.accum):
            x, y = get_batch()
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
            vmsg = ""
            if (step + 1) % max(1, a.steps // 10) == 0 or step == 0:
                vb, _ = val_bpb(min(a.eval_tok, 16000)); vmsg = f"  | val BPB={vb:.4f}"
            print(f"  step {step+1:5d}/{a.steps}  loss={lsum:.4f}  gnorm={gn:.2f}  ({(time.time()-ts)*1000:.0f} ms){vmsg}")

    bpb, top1 = val_bpb(a.eval_tok)
    print(f"== RESULT ==  arm={'ctl' if a.xproj_rank == 0 else f'r{a.xproj_rank}'}  val BPB={bpb:.4f}  "
          f"({(time.time()-t0)/60:.1f} min)")
    print("  gate read (Architect, vs same-script ctl): G1 pass iff BPB(r52) <= BPB(ctl)+0.005; gray to 0.010 -> 3 seeds")
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "cfg": dict(V=V, **AC, xproj_rank=a.xproj_rank, seed=a.seed, steps=a.steps, bpb=bpb),
                    "top1_slice": top1}, a.save)
        print(f"  saved -> {a.save}")
    print("STOP. Owner launches the real arms per the brief; no gate is read before the brief is sealed+pushed.")


if __name__ == "__main__":
    main()
