#!/usr/bin/env python3
# Inventor / S1d - the SECONDARY reads declared in the sealed brief (recorded, no gates):
#   (a) top-1 agreement between arms on the fixed val slice (saved in each ckpt at train time);
#   (b) spectra: does the fresh dense ctl reproduce sp58's x_proj PR~53? do the trained low-rank
#       factors USE their full rank r, or concentrate further (a lower r was available)?
#   (c) foundation intact: ternary MLP weight zero-frac (~31% expected from S2) and dReLU activation
#       sparsity on val windows (anchors: gate ~79%, hidden ~92%) - the one-variable check.
# Read-only; CPU. Output: tee to docs/in_research/s1d_secondary_out.txt
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase57"))
from phase55_ssm import ArchA, load_meta, IDS, META      # noqa: E402
from phase57_sparse import sparsify_mlp, SparseMLP       # noqa: E402
from s1c_structured_xproj import LowRankLinear           # noqa: E402

R57 = os.path.join(ROOT, "results", "phase57")
ARMS = {"ctl": 0, "r52": 52, "r26": 26}


def pr_rank(sv):
    s2 = sv ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def build(rank, sd):
    AC = dict(D=256, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16)
    m = ArchA(1024, **AC)
    sparsify_mlp(m, 256, 1024, act="drelu", gated=True, topk=0.0, ternary=True)
    if rank > 0:
        for blk in m.blocks:
            if hasattr(blk.mix, "x_proj"):
                blk.mix.x_proj = LowRankLinear(blk.mix.x_proj.in_features, blk.mix.x_proj.out_features, rank)
    m.load_state_dict(sd["model"], strict=True)
    m.eval(); m.use_ckpt = False
    return m


def main():
    print("=" * 100)
    print("S1d - secondary reads of the seed-0 A/B (declared in the sealed brief; recorded, no gates)")
    print("=" * 100)
    cks = {n: torch.load(os.path.join(R57, f"s1c_{n}.pt"), map_location="cpu", weights_only=False) for n in ARMS}

    # (a) top-1 agreement between arms (fixed 10-window val slice saved at train time)
    print("\n-- (a) top-1 agreement on the fixed val slice --")
    tops = {n: cks[n]["top1_slice"] for n in ARMS}
    for a in ("ctl",):
        for b in ("r52", "r26"):
            ag = float((tops[a] == tops[b]).mean())
            print(f"  {a} vs {b}: {ag*100:5.2f}%  ({len(tops[a])} preds)")
    print(f"  r52 vs r26: {float((tops['r52'] == tops['r26']).mean())*100:5.2f}%")

    # (b) spectra
    print("\n-- (b) x_proj spectra of the TRAINED arms --")
    sd0 = cks["ctl"]["model"]
    prs = []
    for l in range(6):
        k = f"blocks.{l}.mix.x_proj.weight"
        if k in sd0:
            prs.append(pr_rank(np.linalg.svd(sd0[k].float().numpy(), compute_uv=False)))
    print(f"  ctl dense: PR-rank per layer = {[f'{p:.0f}' for p in prs]}  mean {np.mean(prs):.1f}"
          f"  (sp58_base read: ~53 -> reproducibility of the concentration finding)")
    for n in ("r52", "r26"):
        sd = cks[n]["model"]; r = ARMS[n]
        prs, active = [], []
        for l in range(6):
            kV, kU = f"blocks.{l}.mix.x_proj.Vl.weight", f"blocks.{l}.mix.x_proj.Ul.weight"
            if kV not in sd: continue
            W = (sd[kU].float() @ sd[kV].float()).numpy()          # (208,512), rank <= r
            sv = np.linalg.svd(W, compute_uv=False)
            prs.append(pr_rank(sv))
            active.append(float((sv > sv[0] * 1e-3).sum()))
        print(f"  {n}: product PR-rank = {[f'{p:.0f}' for p in prs]}  mean {np.mean(prs):.1f} of r={r}"
              f"  | numerically-active dims (sv > 1e-3*sv0): {[f'{a:.0f}' for a in active]}")

    # (c) foundation intact
    print("\n-- (c) foundation properties (one-variable check) --")
    for n in ARMS:
        sd = cks[n]["model"]
        zf = []
        for k, v in sd.items():
            if ".mlp." in k and k.endswith(".weight") and ("gate" in k or "up" in k or "down" in k):
                w = v.float()
                sc = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
                zf.append(float(((w / sc).round().clamp(-1, 1) == 0).float().mean()))
        print(f"  {n}: ternary MLP weight zero-frac mean = {np.mean(zf)*100:.1f}%  (S2 read on moe_gran: ~31%)")

    V, exp_len, _ = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    val = ids[int(len(ids) * 0.9):]
    print("  activation sparsity on 5 val windows (anchors: gate ~79%, hidden ~92%):")
    for n in ARMS:
        m = build(ARMS[n], cks[n])
        mlps = [b.mlp for b in m.blocks if isinstance(getattr(b, "mlp", None), SparseMLP)]
        for mm in mlps: mm._cap = True
        with torch.no_grad():
            for w in range(5):
                x = torch.from_numpy(val[w*512:(w+1)*512][None, :])
                m(x)
        sg = np.mean([mm._s_g for mm in mlps]); sh = np.mean([mm._s_h for mm in mlps])
        print(f"    {n}: gate {sg*100:.1f}%  hidden {sh*100:.1f}%")

    print("\nreading: (a) how different the low-rank models' outputs are from ctl (context for the BPB deltas);")
    print("(b) if the r=52 product's PR << 52, the trained factor concentrates further -> r was generous;")
    print("(c) sparsity bands ~= anchors on every arm -> the x_proj variable did not perturb the foundation.")


if __name__ == "__main__":
    main()
