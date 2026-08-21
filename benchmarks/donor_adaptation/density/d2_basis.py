#!/usr/bin/env python3
"""D2 -- is there a basis in which a trained LLM's weights are sparse?

Two questions, one cheap and one open.

(1) Singular spectrum, per organ: how much Frobenius energy sits in the top r?  This prices
    axis C directly and needs no theory.

(2) The Owner's question -- "sono matrici, come le mando in zero e che basi servono?".
    A weight matrix is dense only in the basis it is stored in.  For orthogonal U, V the map
    W -> U^T W V preserves ||W||_F and the operator, so it is free in information terms; the
    only question is whether some U, V concentrates the ENTRIES.  Candidates tested:

      identity            -- the baseline, i.e. "the weights as shipped"
      permutation         -- a no-op control: must reproduce identity's metric EXACTLY
      block-Hadamard      -- structured, essentially free to apply (the engine could afford it)
      block-random-orth   -- structured null at the same block size
      dense random orth   -- the generic-rotation null (Haar)
      own singular basis  -- the trivial upper bound: W becomes diagonal, and the cost is U,V

    Metric: `frac99`, the fraction of matrix entries needed to carry 99% of the Frobenius
    energy (lower = sparser), plus the normalised participation ratio.

PLANTED CONTROL: a synthetic matrix that IS sparse in a Hadamard basis is run through the
same instrument.  If the instrument cannot find a rotation that is *known* to be there, its
"no" on real weights would be worthless.
"""
from __future__ import annotations

import sys, os, json, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

torch.set_num_threads(int(os.environ.get("D_THREADS", "3")))

LAYERS = [0, 13, 27]
ORGANS = ["q_proj", "o_proj", "gate_proj", "down_proj"]
SEED = 4242


# ------------------------------------------------------------------ concentration metrics
def frac_for_energy(W: torch.Tensor, target: float) -> float:
    v = (W.double() ** 2).flatten()
    tot = float(v.sum())
    s, _ = torch.sort(v, descending=True)
    c = torch.cumsum(s, 0)
    k = int(torch.searchsorted(c, torch.tensor(target * tot)).item()) + 1
    return k / v.numel()


def participation_ratio(W: torch.Tensor) -> float:
    v = (W.double() ** 2).flatten()
    return float((v.sum() ** 2) / (v.numel() * (v ** 2).sum()))


def metrics(W: torch.Tensor) -> dict:
    return {"frac90": frac_for_energy(W, 0.90),
            "frac99": frac_for_energy(W, 0.99),
            "frac999": frac_for_energy(W, 0.999),
            "pr": participation_ratio(W),
            "kurtosis": float(((W.double() ** 4).mean() / (W.double() ** 2).mean() ** 2))}


# ------------------------------------------------------------------ orthogonal families
def hadamard_blocks(n: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """Block-diagonal Hadamard: the largest power-of-2 block that divides n, tiled."""
    from scipy.linalg import hadamard
    b = 1
    while b * 2 <= n and n % (b * 2) == 0:
        b *= 2
    H = torch.from_numpy(hadamard(b).astype(np.float32)) / math.sqrt(b)
    M = torch.zeros(n, n)
    for i in range(n // b):
        M[i * b:(i + 1) * b, i * b:(i + 1) * b] = H
    return M


def block_random_orth(n: int, block: int, gen: torch.Generator) -> torch.Tensor:
    b = block
    while n % b != 0:
        b //= 2
    M = torch.zeros(n, n)
    for i in range(n // b):
        A = torch.randn(b, b, generator=gen)
        Q, R = torch.linalg.qr(A)
        Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)     # Haar on O(b)
        M[i * b:(i + 1) * b, i * b:(i + 1) * b] = Q
    return M


def dense_random_orth(n: int, gen: torch.Generator) -> torch.Tensor:
    A = torch.randn(n, n, generator=gen)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def permutation(n: int, gen: torch.Generator) -> torch.Tensor:
    p = torch.randperm(n, generator=gen)
    M = torch.zeros(n, n)
    M[torch.arange(n), p] = 1.0
    return M


# ------------------------------------------------------------------ the probe
def probe_matrix(W: torch.Tensor, name: str, gen: torch.Generator,
                 do_dense_random: bool = True) -> dict:
    """Run every candidate basis over one matrix.  W is [out, in]."""
    W = W.detach().float().clone()
    m, n = W.shape
    out = {"name": name, "shape": [m, n], "bases": {}}

    out["bases"]["identity"] = metrics(W)

    Pl, Pr = permutation(m, gen), permutation(n, gen)
    out["bases"]["permutation"] = metrics(Pl.T @ W @ Pr)

    Hl, Hr = hadamard_blocks(m), hadamard_blocks(n)
    out["bases"]["hadamard_block"] = metrics(Hl.T @ W @ Hr)
    out["hadamard_block_sizes"] = [int((Hl[0] != 0).sum()), int((Hr[0] != 0).sum())]

    Bl, Br = block_random_orth(m, 256, gen), block_random_orth(n, 256, gen)
    out["bases"]["block_random_orth256"] = metrics(Bl.T @ W @ Br)

    if do_dense_random:
        t = time.time()
        Ql = dense_random_orth(m, gen)
        Qr = Ql if n == m else dense_random_orth(n, gen)
        out["bases"]["dense_random_orth"] = metrics(Ql.T @ W @ Qr)
        out["dense_random_secs"] = time.time() - t

    # own singular basis: the upper bound.  W -> diag(sigma)
    t = time.time()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    D = torch.zeros(m, n)
    r = S.numel()
    D[torch.arange(r), torch.arange(r)] = S
    out["bases"]["own_singular"] = metrics(D)
    out["svd_secs"] = time.time() - t

    # spectrum -> axis C pricing
    e = (S.double() ** 2)
    ce = torch.cumsum(e, 0) / e.sum()
    out["spectrum"] = {
        "full_rank": r,
        "r_for_90pct_energy": int(torch.searchsorted(ce, torch.tensor(0.90)).item()) + 1,
        "r_for_95pct_energy": int(torch.searchsorted(ce, torch.tensor(0.95)).item()) + 1,
        "r_for_99pct_energy": int(torch.searchsorted(ce, torch.tensor(0.99)).item()) + 1,
        "energy_top_1pct_rank": float(ce[max(0, r // 100 - 1)]),
        "energy_top_10pct_rank": float(ce[max(0, r // 10 - 1)]),
        "stable_rank": float(e.sum() / e.max()),
        "sigma_max": float(S.max()), "sigma_min": float(S.min()),
    }
    return out


# ------------------------------------------------------------------ planted control
def planted_control(gen: torch.Generator) -> dict:
    """A matrix that IS Hadamard-basis-sparse.  The instrument must see it."""
    m = n = 1536
    Hl, Hr = hadamard_blocks(m), hadamard_blocks(n)
    S = torch.zeros(m, n)
    k = int(0.01 * m * n)                       # 1% of entries carry everything
    idx = torch.randperm(m * n, generator=gen)[:k]
    S.view(-1)[idx] = torch.randn(k, generator=gen)
    W = Hl @ S @ Hr.T                           # dense in the stored basis, sparse under H
    res = probe_matrix(W, "PLANTED_hadamard_sparse", gen, do_dense_random=False)
    res["expect"] = "hadamard_block.frac99 << identity.frac99"
    res["fired"] = res["bases"]["hadamard_block"]["frac99"] < 0.5 * res["bases"]["identity"]["frac99"]
    return res


def planted_negative_control(gen: torch.Generator) -> dict:
    """The other direction: an i.i.d. Gaussian matrix has NO sparsifying basis except its own
    singular one.  Every structured basis must come back ~equal to identity."""
    W = torch.randn(1536, 1536, generator=gen)
    res = probe_matrix(W, "PLANTED_iid_gaussian", gen, do_dense_random=False)
    i99 = res["bases"]["identity"]["frac99"]
    res["expect"] = "hadamard_block.frac99 ~= identity.frac99 (within 5%)"
    res["fired"] = abs(res["bases"]["hadamard_block"]["frac99"] - i99) / i99 < 0.05
    return res


def main():
    gen = torch.Generator().manual_seed(SEED)
    m, tok = C.load_model()
    A = C.arch(m)
    log = {"arch": A, "seed": SEED, "controls": [], "matrices": []}

    print("== D2 planted controls ==", flush=True)
    for fn in (planted_control, planted_negative_control):
        r = fn(gen)
        log["controls"].append(r)
        print(f"  {r['name']}: identity frac99={r['bases']['identity']['frac99']:.4f} "
              f"hadamard frac99={r['bases']['hadamard_block']['frac99']:.4f} "
              f"own_singular frac99={r['bases']['own_singular']['frac99']:.6f} "
              f"-> {'FIRED' if r['fired'] else 'DID NOT FIRE'}", flush=True)
        C.dump("d2_basis.json", log)

    print("== D2 real donor organs ==", flush=True)
    for L in LAYERS:
        for organ in ORGANS:
            W = C.get_linear(m, L, organ).weight
            t = time.time()
            r = probe_matrix(W, f"L{L:02d}.{organ}", gen,
                             do_dense_random=(max(W.shape) <= 1536))
            r["layer"], r["organ"] = L, organ
            log["matrices"].append(r)
            b = r["bases"]
            print(f"  L{L:02d}.{organ:10s} {tuple(W.shape)}  "
                  f"frac99: id={b['identity']['frac99']:.4f} "
                  f"perm={b['permutation']['frac99']:.4f} "
                  f"had={b['hadamard_block']['frac99']:.4f} "
                  f"blkrand={b['block_random_orth256']['frac99']:.4f} "
                  f"{'dense=' + format(b['dense_random_orth']['frac99'], '.4f') if 'dense_random_orth' in b else ''} "
                  f"svd={b['own_singular']['frac99']:.5f}  |  "
                  f"r99={r['spectrum']['r_for_99pct_energy']}/{r['spectrum']['full_rank']} "
                  f"stable_rank={r['spectrum']['stable_rank']:.1f}  [{time.time()-t:.0f}s]",
                  flush=True)
            C.dump("d2_basis.json", log)

    print("wrote", C.dump("d2_basis.json", log))


if __name__ == "__main__":
    main()
