#!/usr/bin/env python3
"""D2b -- is there a basis in which the donor's ACTIVATIONS are sparse?

The coordinator's prior-art note retired rotate-for-weight-sparsity as an open question
(DenoiseRotator / ProcrustesGPT / RotPruner, 2025-26) and redirected the search here: our
engine's gate-first skip path consumes ACTIVATION sparsity, not weight magnitude.  So the
question is whether an orthogonal Q exists with Qh markedly sparser than h.

THE COST, COMPUTED FROM THE ARTEFACT BEFORE ANY RESULT (this is what the rotation must beat):
  FFN cost per token = 3 * d_model * d_ffn MACs.
  A dense d_ffn x d_ffn rotation costs d_ffn^2 MACs.
  A block-diagonal rotation with block b costs d_ffn * b MACs.
Those ratios are printed and stored; a rotation that does not pay for itself in skipped
bytes is dead on arrival regardless of how well it concentrates.

Also note, and it is not a detail: Qh cannot be folded into the weights.  h is an elementwise
product of two projections, so Qh can only be formed AFTER h exists.  A rotation therefore
never saves the gate/up read; at best it saves the down read.  That halves any upside and
is stated here, not discovered later.

Metric: the fidelity that matters is of y = down(h), not of h.  For each basis Q and each
kept fraction f, we keep the top-f |coefficients| of Qh per token, map back, and measure the
relative L2 error of the reconstructed y.  Identity is the baseline; block-random-orthogonal
and dense-random-orthogonal are the nulls; the activation-covariance eigenbasis is the
data-optimal upper bound.

PLANTED CONTROLS: activations that ARE block-Hadamard-sparse must be found; i.i.d. Gaussian
activations must show every basis equal to identity.
"""
from __future__ import annotations

import sys, os, json, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from d0_layout import capture_mlp_inputs, hidden
from d2_basis import hadamard_blocks, block_random_orth

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

LAYERS = [1, 7, 14, 21, 27]
KEEP_FRACS = [0.02, 0.05, 0.10, 0.25, 0.50]
N_CAL, N_EVAL, SEQ_LEN = 12, 6, 512
SEED = 3131


def keep_topf(Z: torch.Tensor, f: float) -> torch.Tensor:
    k = max(1, int(round(f * Z.shape[1])))
    idx = Z.abs().topk(k, dim=1).indices
    M = torch.zeros_like(Z)
    M.scatter_(1, idx, Z.gather(1, idx))
    return M


def concentration(Z: torch.Tensor, target=0.99) -> float:
    """Mean over tokens of the fraction of coefficients carrying `target` of the energy."""
    v, _ = torch.sort(Z ** 2, dim=1, descending=True)
    c = torch.cumsum(v, 1)
    tot = c[:, -1:].clamp_min(1e-30)
    k = (c / tot < target).sum(1) + 1
    return float(k.float().mean() / Z.shape[1])


def eval_basis(Q, h: torch.Tensor, down, name: str) -> dict:
    """Q may be None (identity)."""
    Z = h if Q is None else h @ Q                    # rows are tokens: z = Q^T h  (Q columns)
    y = down(h)
    den = float((y ** 2).sum())
    out = {"basis": name, "concentration99": concentration(Z), "relerr": {}}
    for f in KEEP_FRACS:
        Zk = keep_topf(Z, f)
        hk = Zk if Q is None else Zk @ Q.T
        yk = down(hk)
        out["relerr"][str(f)] = math.sqrt(float(((yk - y) ** 2).sum()) / den)
    return out


def bases_for(N: int, gen: torch.Generator, dense: bool, hcov: torch.Tensor | None):
    B = {"identity": None,
         "hadamard_block": hadamard_blocks(N),
         "block_random_orth256": block_random_orth(N, 256, gen)}
    if dense:
        B["dense_random_orth"] = _dense_orth(N, gen)
    if hcov is not None:
        evals, evecs = torch.linalg.eigh(hcov.double())
        B["activation_pca"] = evecs.flip(-1).float()
    return B


def _dense_orth(n, gen):
    A = torch.randn(n, n, generator=gen)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


# ---------------------------------------------------------------- planted controls
def planted_controls(gen):
    N, T = 1024, 512
    Hb = hadamard_blocks(N)
    out = []

    # (+) activations that ARE sparse in a Hadamard basis
    Z = torch.zeros(T, N)
    k = int(0.02 * N)
    for t in range(T):
        idx = torch.randperm(N, generator=gen)[:k]
        Z[t, idx] = torch.randn(k, generator=gen)
    h = Z @ Hb.T                                     # dense in the native basis
    down = torch.nn.Linear(N, 256, bias=False)
    with torch.no_grad():
        down.weight.copy_(torch.randn(256, N, generator=gen) / math.sqrt(N))
    idr = eval_basis(None, h, down, "identity")
    hdr = eval_basis(Hb, h, down, "hadamard_block")
    out.append({"name": "planted_hadamard_sparse_activations",
                "expect": "hadamard relerr@0.02 << identity relerr@0.02",
                "identity_relerr_002": idr["relerr"]["0.02"],
                "hadamard_relerr_002": hdr["relerr"]["0.02"],
                "fired": hdr["relerr"]["0.02"] < 0.1 * idr["relerr"]["0.02"]})

    # (-) i.i.d. Gaussian activations: no basis may beat identity
    h2 = torch.randn(T, N, generator=gen)
    a = eval_basis(None, h2, down, "identity")["relerr"]["0.10"]
    b = eval_basis(Hb, h2, down, "hadamard_block")["relerr"]["0.10"]
    out.append({"name": "iid_gaussian_activations",
                "expect": "hadamard relerr@0.10 ~= identity relerr@0.10 (within 5%)",
                "identity_relerr_010": a, "hadamard_relerr_010": b,
                "fired": abs(b - a) / a < 0.05})
    return out


def main():
    gen = torch.Generator().manual_seed(SEED)
    log = {"keep_fracs": KEEP_FRACS, "controls": [], "layers": []}

    print("== D2b planted controls ==", flush=True)
    for c in planted_controls(gen):
        log["controls"].append(c)
        print(f"  {c['name']:38s} -> {'FIRED' if c['fired'] else 'DID NOT FIRE'}  {c}", flush=True)
    C.dump("d2b_actbasis.json", log)

    m, tok = C.load_model()
    A = C.arch(m); log["arch"] = A
    d, N = A["d_model"], A["d_ffn"]
    cost = {"ffn_macs_per_token": 3 * d * N,
            "dense_rotation_macs": N * N,
            "dense_rotation_vs_ffn": (N * N) / (3 * d * N),
            "block256_rotation_macs": N * 256,
            "block256_rotation_vs_ffn": (N * 256) / (3 * d * N),
            "note": "Qh is not foldable: h is an elementwise product, so a rotation can "
                    "never save the gate/up read -- at best it saves the down read (1/3)."}
    log["cost_accounting"] = cost
    print("cost:", json.dumps(cost, indent=1), flush=True)

    cal_ids, _, cm = C.get_slice(tok, "calib", N_CAL, SEQ_LEN, SEED)
    ev_ids, _, em = C.get_slice(tok, "heldout", N_EVAL, SEQ_LEN, SEED + 1)
    log["cal_slice"], log["eval_slice"] = cm, em
    with C.Timer("capture"):
        Xcal = capture_mlp_inputs(m, cal_ids, LAYERS)
        Xev = capture_mlp_inputs(m, ev_ids, LAYERS)

    dense_Q = None
    for L in LAYERS:
        t0 = time.time()
        hc = hidden(m, L, Xcal[L])
        he = hidden(m, L, Xev[L])
        cov = (hc.T @ hc) / hc.shape[0]
        rec = {"layer": L, "bases": []}

        if dense_Q is None:
            with C.Timer("dense random orthogonal 8960x8960 (built once)"):
                dense_Q = _dense_orth(N, gen)
        B = bases_for(N, gen, dense=False, hcov=cov)
        B["dense_random_orth"] = dense_Q

        down = m.model.layers[L].mlp.down_proj
        for name, Q in B.items():
            r = eval_basis(Q, he, down, name)
            rec["bases"].append(r)
            print(f"  L{L:02d} {name:22s} conc99={r['concentration99']:.4f}  "
                  + "  ".join(f"f={f}:{r['relerr'][str(f)]:.4f}" for f in KEEP_FRACS),
                  flush=True)

        # activation subspace: how many PCA directions carry the energy at all
        ev = torch.linalg.eigvalsh(cov.double()).flip(-1).clamp_min(0)
        ce = torch.cumsum(ev, 0) / ev.sum()
        rec["activation_spectrum"] = {
            "r_for_90pct": int(torch.searchsorted(ce, torch.tensor(0.90)).item()) + 1,
            "r_for_99pct": int(torch.searchsorted(ce, torch.tensor(0.99)).item()) + 1,
            "r_for_999pct": int(torch.searchsorted(ce, torch.tensor(0.999)).item()) + 1,
            "dim": int(N)}
        print(f"  L{L:02d} activation spectrum: r90={rec['activation_spectrum']['r_for_90pct']} "
              f"r99={rec['activation_spectrum']['r_for_99pct']} "
              f"r999={rec['activation_spectrum']['r_for_999pct']} of {N}   "
              f"[{time.time()-t0:.0f}s]", flush=True)
        log["layers"].append(rec)
        C.dump("d2b_actbasis.json", log)

    print("wrote", C.dump("d2b_actbasis.json", log))


if __name__ == "__main__":
    main()
