#!/usr/bin/env python3
"""D3 -- does the project's own low-rank win transfer to a donor?

THE CLAIM BEING TESTED, STATED HONESTLY: the Inventor measured `x_proj` low-rank r=26
beating the dense baseline on 3/3 seeds (~3.5 sigma) at 17.6% of the bytes.  That was
measured on OUR architecture, at 8.3M parameters, on one SSM organ.  It does not transfer
by assumption to a 1.5B transformer donor.  This probe tests it; it does not apply it.

Method: replace organ W (in ALL layers at once) by its rank-r truncation and read BPB off
the pinned held-out slice.  W_r = W V_r V_r^T (or U_r U_r^T W, whichever side is smaller),
with V from the eigendecomposition of the Gram matrix -- exactly the top-r SVD subspace,
computed on the small side so 28 layers x 7 organs is minutes, not hours.

Bytes retained for a rank-r factorisation of an m x n matrix is r(m+n)/(mn); that ratio,
not r, is the axis the engine cares about, so every row reports it.

PLANTED CONTROLS:
  * r = full rank must reproduce the baseline (delta ~ 0 to float precision) -- if the
    reconstruction path is wrong this is where it shows;
  * r = 1 must be catastrophic.
Both directions logged.
"""
from __future__ import annotations

import sys, os, json, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from d1_pruning import Snapshot, paired_se

torch.set_num_threads(int(os.environ.get("D_THREADS", "10")))

N_SEQ, SEQ_LEN, SEED = 24, 512, 1234
ORGANS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RANKS = [26, 64, 128, 256, 512, 768, 1024]


def rank_subspace(W: torch.Tensor):
    """Return (side, B) such that W_r = W @ B[:, :r] @ B[:, :r].T  (side='right')
    or W_r = B[:, :r] @ B[:, :r].T @ W  (side='left').  B holds singular vectors."""
    m, n = W.shape
    Wd = W.double()
    if n <= m:
        G = Wd.T @ Wd
        evals, evecs = torch.linalg.eigh(G)
        return "right", evecs.flip(-1).float()          # descending singular value order
    G = Wd @ Wd.T
    evals, evecs = torch.linalg.eigh(G)
    return "left", evecs.flip(-1).float()


def apply_rank(W: torch.Tensor, side: str, B: torch.Tensor, r: int):
    if side == "right":
        Br = B[:, :r]
        W.copy_((W @ Br) @ Br.T)
    else:
        Br = B[:, :r]
        W.copy_(Br @ (Br.T @ W))


def bytes_fraction(m, n, r):
    return r * (m + n) / (m * n)


def main():
    m_, tok = C.load_model()
    A = C.arch(m_)
    ids, byts, meta = C.get_slice(tok, "heldout", N_SEQ, SEQ_LEN, SEED)
    L_all = list(range(A["n_layers"]))

    base, base_per = C.bpb(m_, ids, byts, return_per_seq=True)
    print(f"baseline BPB {base:.6f}", flush=True)
    log = {"arch": A, "slice": meta, "baseline_bpb": base, "ranks": RANKS,
           "provenance_note": ("the r=26 win is OURS, measured on x_proj of an 8.3M SSM; "
                               "it is under test here, not assumed"),
           "controls": [], "sweep": []}

    def evaluate(tag, **extra):
        b, per = C.bpb(m_, ids, byts, return_per_seq=True)
        se = paired_se(base_per, per, byts)
        rec = {"tag": tag, "bpb": b, "delta": b - base, "paired_se": se, **extra}
        print(f"  {tag:34s} BPB {b:.5f}  d={b-base:+.5f} +-{se:.5f} "
              f"bytes={extra.get('bytes_frac', float('nan')):.4f}", flush=True)
        return rec

    snap = Snapshot()
    for organ in ORGANS:
        subs = {}
        t0 = time.time()
        for L in L_all:
            W = C.get_linear(m_, L, organ).weight
            snap.take((L, organ), W)
            subs[L] = rank_subspace(W)
        shape = tuple(C.get_linear(m_, 0, organ).weight.shape)
        full = min(shape)
        print(f"== {organ} {shape} full_rank={full} subspaces in {time.time()-t0:.0f}s ==",
              flush=True)

        ranks = [r for r in RANKS if r < full] + [full]
        for r in ranks:
            for L in L_all:
                side, B = subs[L]
                apply_rank(C.get_linear(m_, L, organ).weight, side, B, r)
            bf = bytes_fraction(shape[0], shape[1], r)
            tag = f"{organ}/r={r}"
            rec = evaluate(tag, organ=organ, rank=r, full_rank=full,
                           bytes_frac=bf if r < full else 1.0, shape=list(shape))
            if r == full:
                rec["control"] = "full_rank_must_reproduce_baseline"
                rec["fired"] = abs(rec["delta"]) < 1e-3
                log["controls"].append(rec)
            else:
                log["sweep"].append(rec)
            # restore before the next rank (truncation is not idempotent)
            for L in L_all:
                C.get_linear(m_, L, organ).weight.copy_(snap.saved[(L, organ)])
            C.dump("d3_lowrank.json", log)

        # r = 1 catastrophic control, once per organ family (attn + mlp representative)
        if organ in ("o_proj", "down_proj"):
            for L in L_all:
                side, B = subs[L]
                apply_rank(C.get_linear(m_, L, organ).weight, side, B, 1)
            rec = evaluate(f"CTRL {organ}/r=1", organ=organ, rank=1,
                           bytes_frac=bytes_fraction(shape[0], shape[1], 1))
            rec["control"] = "r1_must_be_catastrophic"
            rec["fired"] = rec["delta"] > 0.5
            log["controls"].append(rec)
            for L in L_all:
                C.get_linear(m_, L, organ).weight.copy_(snap.saved[(L, organ)])
            C.dump("d3_lowrank.json", log)

        snap.restore(m_)
        del subs

    print("wrote", C.dump("d3_lowrank.json", log))


if __name__ == "__main__":
    main()
