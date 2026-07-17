#!/usr/bin/env python3
# Inventor / S2 - empirical entropy of the ternary weights: how many bits/weight does the streamed
# expert pool REALLY need, vs the engine's current 4-bit-effective pack and the queued 5-trits/byte pack?
#
# Read-only probe: loads results/phase57/moe_gran.pt (probe-4 promoted MoE, the streamed-pool worst case:
# trained from scratch) and results/phase57/sp58_base.pt (dense-1024 reference). Ternarization = the EXACT
# engine/export semantics (e4_export.py deq_pack: per-row absmean scale, round, clamp) - we measure the
# trits the engine actually streams, not an approximation.
#
# Coding bounds reported (bits/weight):
#   flat        log2(3) = 1.585                      (no model)
#   H0          order-0 entropy of the trit marginal (a static arithmetic/rANS coder, one table)
#   H0/row      mean per-row order-0                 (per-row tables; side info ~= scales already shipped)
#   Hpair/2     entropy of adjacent-trit PAIRS / 2   (the 9-symbol alphabet the g=2 engine codes already use:
#                                                     a fixed 9-symbol entropy code needs exactly this)
#   H1          order-1 Markov cond. entropy along rows (upper-bound of cheap context coding)
# Storage lines: current int8-code pack (8 bits/pair = 4.0 b/w), 5-trits/byte (1.6 b/w), and each bound.
#
# Output: prints a full table; run with  > docs/in_research/s2_weight_entropy_out.txt
import os, sys, math
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT_MOE = os.path.join(ROOT, "results", "phase57", "moe_gran.pt")
CKPT_DEN = os.path.join(ROOT, "results", "phase57", "sp58_base.pt")


def ternarize(w: torch.Tensor, dim: int) -> np.ndarray:
    """EXACT e4_export.py / BitLinear158 semantics: per-row absmean over `dim`, round, clamp."""
    scale = w.abs().mean(dim=dim, keepdim=True).clamp_min(1e-5)
    return (w / scale).round().clamp(-1, 1).to(torch.int8).numpy()


def H_of_counts(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def stats(trits: np.ndarray, pair_axis: int):
    """trits: int8 array; pair_axis: the axis the engine pairs along (the per-row input dim)."""
    t = np.moveaxis(trits, pair_axis, -1)          # rows x len
    t = t.reshape(-1, t.shape[-1])
    n = t.size
    # marginals + H0
    cnt = np.array([(t == -1).sum(), (t == 0).sum(), (t == 1).sum()], dtype=np.int64)
    h0 = H_of_counts(cnt)
    # per-row H0 (weighted mean = the per-row-adaptive bound)
    hr = 0.0
    for v in (-1, 0, 1):
        pv = (t == v).mean(axis=1)
        nz = pv > 0
        hrow = np.zeros(t.shape[0])
        hrow[nz] = -pv[nz] * np.log2(pv[nz])
        hr += hrow
    h0_row = float((hr * t.shape[1]).sum() / n)
    # pair entropy (g=2 pairing along the row dim, as the engine codes)
    L2 = t.shape[1] - (t.shape[1] % 2)
    pairs = (t[:, 0:L2:2].astype(np.int32) + 1) * 3 + (t[:, 1:L2:2] + 1)   # 9 symbols
    pc = np.bincount(pairs.ravel(), minlength=9)
    h_pair = H_of_counts(pc) / 2.0
    # order-1 Markov along rows
    a, b = t[:, :-1].ravel() + 1, t[:, 1:].ravel() + 1
    joint = np.bincount(a * 3 + b, minlength=9).astype(np.float64).reshape(3, 3)
    pj = joint / joint.sum()
    pa = pj.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        hcond = -(pj * np.log2(np.where(pj > 0, pj / pa, 1))).sum()
    return dict(n=n, p_m1=cnt[0] / n, p_0=cnt[1] / n, p_p1=cnt[2] / n,
                H0=h0, H0_row=h0_row, Hpair=h_pair, H1=float(hcond))


def fmt(name, s):
    return (f"  {name:<18} n={s['n']/1e6:6.2f}M  p(-1/0/+1)={s['p_m1']:.3f}/{s['p_0']:.3f}/{s['p_p1']:.3f}"
            f"  H0={s['H0']:.4f}  H0/row={s['H0_row']:.4f}  Hpair/2={s['Hpair']:.4f}  H1={s['H1']:.4f}")


def agg(list_of_stats):
    n = sum(s["n"] for s in list_of_stats)
    out = {"n": n}
    for k in ("p_m1", "p_0", "p_p1", "H0", "H0_row", "Hpair", "H1"):
        out[k] = sum(s[k] * s["n"] for s in list_of_stats) / n
    return out


def main():
    print("=" * 100)
    print("S2 - empirical entropy of the ternary weights (engine-exact trits)")
    print("=" * 100)

    # ---------------- MoE pool (the streamed class) ----------------
    sd = torch.load(CKPT_MOE, map_location="cpu")
    msd, cfg = sd["model"], sd["cfg"]
    L, E, hid_e = cfg["L"], cfg["E"], cfg["hid_e"]
    print(f"\n[moe_gran.pt]  L={L} E={E} hid_e={hid_e} top-{cfg['topk']}  bpb={cfg.get('bpb')}  (from-scratch pool = worst case for structure)")
    per_class = {"gate": [], "up": [], "Wd": []}
    for l in range(L):
        for name, dim in (("gate", 1), ("up", 1)):
            w = msd[f"blocks.{l}.mlp.{name}.weight"].float()
            per_class[name].append(stats(ternarize(w, dim), pair_axis=1))
        wd = msd[f"blocks.{l}.mlp.Wd"].float()
        per_class["Wd"].append(stats(ternarize(wd, 2), pair_axis=2))
    print("\nper matrix class (aggregated over layers):")
    for name in ("gate", "up", "Wd"):
        print(fmt(name, agg(per_class[name])))
    pool = agg(per_class["gate"] + per_class["up"] + per_class["Wd"])
    print(fmt("POOL (all)", pool))

    # per-layer drift (does entropy change with depth?)
    print("\nper-layer POOL H0 (drift check):")
    for l in range(L):
        sl = agg([per_class["gate"][l], per_class["up"][l], per_class["Wd"][l]])
        print(f"  L{l}: H0={sl['H0']:.4f}  Hpair/2={sl['Hpair']:.4f}  p0={sl['p_0']:.3f}")

    # ---------------- storage table for the streamed pool ----------------
    nW = pool["n"]                                   # ternary weights in the pool
    schemes = [
        ("current engine pack (int8 code/pair)", 4.000),
        ("dense trit-pack (5 trits/byte)",       1.600),
        ("flat log2(3)",                         math.log2(3)),
        ("H0 (one static table)",                pool["H0"]),
        ("H0/row (per-row tables)",              pool["H0_row"]),
        ("Hpair/2 (9-symbol pair code)",         pool["Hpair"]),
        ("H1 (order-1 context)",                 pool["H1"]),
    ]
    base = 4.000
    print(f"\nstreamed-pool storage ({nW/1e6:.2f}M ternary weights, {L} layers x {E} experts):")
    print(f"  {'scheme':<38}{'bits/w':>8}{'pool MB':>10}{'KB/expert':>11}{'vs current':>11}")
    for name, b in schemes:
        mb = nW * b / 8 / 1e6
        kbe = nW * b / 8 / 1024 / (L * E)
        print(f"  {name:<38}{b:>8.3f}{mb:>10.2f}{kbe:>11.1f}{base/b:>10.2f}x")

    # ---------------- dense reference (resident MLP, for context) ----------------
    sdd = torch.load(CKPT_DEN, map_location="cpu")
    dsd = sdd["model"] if "model" in sdd else sdd
    dl = []
    for kk in sorted(dsd.keys()):
        if kk.endswith(".weight") and (".mlp.gate" in kk or ".mlp.up" in kk or ".mlp.down" in kk):
            dl.append(stats(ternarize(dsd[kk].float(), 1), pair_axis=1))
    if dl:
        print(f"\n[sp58_base.pt] dense-1024 ternary MLP (resident class, context only):")
        print(fmt("dense MLP", agg(dl)))

    print("\nreading: if H* << 1.6, entropy coding beats even the dense trit-pack; the decode-side cost")
    print("is the open engineering question (S2 report). Numbers here are coding BOUNDS, not a codec.")


if __name__ == "__main__":
    main()
