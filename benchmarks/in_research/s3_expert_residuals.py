#!/usr/bin/env python3
# Inventor / S3 - cross-expert redundancy of the MoE pool: is there shared structure that delta-coding
# (experts = centroid + sparse ternary residual) could exploit?
#
# Read-only probe on results/phase57/moe_gran.pt. Engine-exact trits (e4_export.py semantics).
#
# Honest design notes (declared before the numbers):
#  * PERMUTATION SYMMETRY: experts trained from scratch have arbitrary hidden-unit ordering, so ALIGNED
#    elementwise agreement across experts is expected ~= chance. We measure it anyway (cheap, falsifiable)
#    with a shuffled-control baseline.
#  * The honest redundancy question is POOL-LEVEL: for every ternary row in the pool, how close is its
#    nearest OTHER row (any expert, any position)? If near-duplicate rows exist, "match-id + residual"
#    coding beats direct coding regardless of alignment. Chance baseline = same statistic on rows with
#    independently shuffled entries (preserves per-row marginals, destroys structure).
#  * fp32-domain check: nearest-neighbour cosine BEFORE ternarization - structure can exist in fp32 and
#    be destroyed by ternary rounding (or vice versa: ternary can alias rows together).
#  * SCOPE: moe_gran is the from-scratch WORST case. The ladder's models are upcycled with 4x replica
#    seeding (8 slices x 4 copies) -> reruns on MVE/S0 checkpoints are the real product-relevant read.
#
# Output: prints tables; run with  > docs/in_research/s3_expert_residuals_out.txt
import os, math
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(ROOT, "results", "phase57", "moe_gran.pt")
RNG = np.random.default_rng(0)


def ternarize(w: torch.Tensor, dim: int) -> np.ndarray:
    scale = w.abs().mean(dim=dim, keepdim=True).clamp_min(1e-5)
    return (w / scale).round().clamp(-1, 1).to(torch.int8).numpy()


def H_of_probs(p):
    p = np.asarray([x for x in p if x > 0], dtype=np.float64)
    return float(-(p * np.log2(p)).sum())


def mode_agreement(T3):
    """T3: (E, R, C) trits. Mode over experts at aligned (r,c); return mean agreement with the mode."""
    onehot = np.stack([(T3 == v) for v in (-1, 0, 1)], axis=0)          # (3,E,R,C)
    counts = onehot.sum(axis=1)                                          # (3,R,C)
    mode = counts.argmax(axis=0)                                         # (R,C) in {0,1,2}
    agree = (T3 + 1 == mode[None]).mean()
    return float(agree)


def nn_match(pool, sample=None):
    """pool: (Nrows, Ln) trits. For each row, best agreement with any OTHER row.
    Returns per-row best-agreement array. O(N^2 * Ln) via 3 one-hot matmuls (float32)."""
    N, Ln = pool.shape
    idx = np.arange(N)
    if sample is not None and N > sample:                                # cap the query side, keys = full pool
        idx = RNG.choice(N, size=sample, replace=False)
    q = pool[idx]
    agree = np.zeros((len(idx), N), dtype=np.float32)
    for v in (-1, 0, 1):
        Iv_q = (q == v).astype(np.float32)
        Iv_k = (pool == v).astype(np.float32)
        agree += Iv_q @ Iv_k.T
    agree /= Ln
    agree[np.arange(len(idx)), idx] = -1.0                               # exclude self
    best = agree.max(axis=1)
    barg = agree.argmax(axis=1)
    return idx, best, barg


def nn_cosine_fp32(pool_f, sample=None):
    N = pool_f.shape[0]
    idx = np.arange(N)
    if sample is not None and N > sample:
        idx = RNG.choice(N, size=sample, replace=False)
    X = pool_f / (np.linalg.norm(pool_f, axis=1, keepdims=True) + 1e-12)
    sims = np.abs(X[idx] @ X.T)                                          # |cos|: sign-flipped twins count too
    sims[np.arange(len(idx)), idx] = 0.0                                 # exclude self AFTER abs
    return sims.max(axis=1)


def shuffled_copy(pool):
    """Independently permute each row's entries: same per-row marginals, no structure."""
    out = pool.copy()
    for i in range(out.shape[0]):
        RNG.shuffle(out[i])
    return out


def residual_bits(pool, idx, best, barg):
    """Delta-coding estimate: row ~ (match-id, ternary residual vs matched row).
    Residual symbol = (self trit, match trit) disagreement pattern; bound it by the entropy of the
    disagreement indicator + the trit value where it disagrees (<= 1 bit given disagreement is binary-ish;
    we compute the exact empirical symbol entropy)."""
    N, Ln = pool.shape
    q = pool[idx]
    m = pool[barg]
    diff_mask = (q != m)
    p_d = diff_mask.mean()
    # empirical entropy of the residual stream: symbol = 0 (agree) or the actual trit (2 options when disagree)
    sym = np.where(diff_mask, q + 2, 0).ravel()                          # 0=agree, {1,2,3} minus one impossible
    cnt = np.bincount(sym, minlength=4).astype(np.float64)
    h_res = H_of_probs(cnt / cnt.sum())
    bits_per_row = math.log2(max(N, 2)) + Ln * h_res
    return p_d, h_res, bits_per_row


def main():
    print("=" * 100)
    print("S3 - cross-expert redundancy / delta-coding potential (engine-exact trits)")
    print("=" * 100)
    sd = torch.load(CKPT, map_location="cpu")
    msd, cfg = sd["model"], sd["cfg"]
    L, E, hid_e, D = cfg["L"], cfg["E"], cfg["hid_e"], cfg["D"]
    print(f"[moe_gran.pt] L={L} E={E} hid_e={hid_e} D={D}  (from-scratch = worst case; upcycled ladder ckpts = the real read)\n")

    tot_direct = tot_delta = 0.0
    for l in range(L):
        for name, is_wd in (("gate", False), ("up", False), ("Wd", True)):
            if is_wd:
                w = msd[f"blocks.{l}.mlp.Wd"].float()                    # (E, D, hid_e)
                trits = ternarize(w, 2)
                T3 = trits                                               # aligned axis: output dim d (cols permuted per expert)
                pool = trits.reshape(E * D, hid_e)
                poolf = w.numpy().reshape(E * D, hid_e)
            else:
                w = msd[f"blocks.{l}.mlp.{name}.weight"].float()         # (E*hid_e, D)
                trits = ternarize(w, 1)
                T3 = trits.reshape(E, hid_e, D)
                pool = trits
                poolf = w.numpy()
            # (1) aligned mode-centroid agreement vs shuffled control
            agree = mode_agreement(T3)
            T3s = T3.copy().reshape(E, -1)
            for e in range(E):
                RNG.shuffle(T3s[e])                                      # destroy alignment within each expert
            agree_ctl = mode_agreement(T3s.reshape(T3.shape))
            # (2) pool NN match vs shuffled control
            idx, best, barg = nn_match(pool, sample=2048)
            pool_ctl = shuffled_copy(pool)
            _, best_ctl, _ = nn_match(pool_ctl, sample=1024)
            # intra vs inter expert matches
            rows_per_e = pool.shape[0] // E
            same_e = ((idx // rows_per_e) == (barg // rows_per_e)).mean()
            # (3) fp32 cosine NN
            bcos = nn_cosine_fp32(poolf, sample=1024)
            # (4) delta-coding estimate vs direct H0
            cnt = np.array([(pool == -1).sum(), (pool == 0).sum(), (pool == 1).sum()], dtype=np.float64)
            h0 = H_of_probs(cnt / cnt.sum())
            p_d, h_res, bits_row = residual_bits(pool, idx, best, barg)
            direct_row = pool.shape[1] * h0
            tot_direct += direct_row * pool.shape[0]
            tot_delta += min(bits_row, direct_row) * pool.shape[0]       # coder picks the cheaper per row class
            print(f"L{l} {name:<4} pool={pool.shape[0]}x{pool.shape[1]}  "
                  f"mode-agree={agree*100:5.1f}% (ctl {agree_ctl*100:5.1f}%)  "
                  f"NN-agree med={np.median(best)*100:5.1f}% p90={np.quantile(best,0.9)*100:5.1f}% (ctl med {np.median(best_ctl)*100:5.1f}%)  "
                  f"intra-e={same_e*100:4.1f}%  |cos|fp32 med={np.median(bcos):.3f}  "
                  f"delta: p_diff={p_d*100:4.1f}% Hres={h_res:.3f} -> {bits_row:6.1f} vs direct {direct_row:6.1f} b/row")
    print(f"\nTOTAL pool: direct-H0 {tot_direct/8/1e6:.2f} MB  vs  delta-NN {tot_delta/8/1e6:.2f} MB  "
          f"({(1 - tot_delta/max(tot_direct,1e-9))*100:+.1f}% saving; negative saving = no structure)")
    print("\nreading: NN-agree ~= ctl -> the pool has NO exploitable near-duplicate structure (S3 dies on")
    print("from-scratch pools); NN-agree >> ctl -> delta-coding is alive. Rerun on upcycled MVE/S0 ckpts")
    print("(4x replica seeding) before any verdict on the ladder models.")


if __name__ == "__main__":
    main()
