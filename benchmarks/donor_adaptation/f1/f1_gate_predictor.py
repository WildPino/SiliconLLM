#!/usr/bin/env python3
"""F1 -- is the 33.3% FFN floor structural?  A closed-form low-rank gate predictor.

Implements `docs/research/donor_adaptation/briefs/BRIEF_F1_GATE_SIGN_PREDICTOR.md`
INCLUDING AMENDMENT 1 (the IVF-PQ / ANN arm, stage `ann`).

--------------------------------------------------------------------------------------------
THE CONSTRUCTION (brief section 1)

    SwiGLU:   h = SiLU(W_g x) (*) (W_u x),   y = W_d h
    factor:   W_g ~= A B,  A: [F, r], B: [r, D],  ghat = A (B x)
    use ghat ONLY to choose S = { i : ghat_i > tau }; compute the EXACT gate on S alone.

    FFN_active = [ r(D+F)/(D*F)  +  3|S|/F ] / 3

THE ASYMMETRY (brief section 1.1) -- never summed into an "accuracy":
    false positive  (predicted active, actually inactive) -> speed cost only, zero quality loss
    false negative  (predicted inactive, actually active) -> quality loss
    => the measured quantity is RECALL vs |S|, swept over tau, at each rank.

--------------------------------------------------------------------------------------------
CLOSED FORM (and why it is cheap)

    min_{A,B} ||(W_g - AB) X||_F   with  H = X X^T  (D x D, activation second moment)

Write M = W_g H^{1/2}.  The minimiser is the rank-r truncation of M, i.e.

    A B = U_r U_r^T W_g          with U_r = top-r LEFT singular vectors of M.

So BOTH registered arms are "project the gate onto an r-dimensional subspace of R^F":
  * activation-weighted arm : U_r from  M = W_g H^{1/2}
  * plain-SVD arm           : U_r from  M = W_g          (H = I)
and every projector control (C1/C2b/C3) is the same expression with a different subspace.
That makes the arms exactly comparable -- they differ ONLY in which subspace is chosen --
and it removes the H^{-1/2} inverse from the predictor entirely (H^{1/2} is used to CHOOSE
the subspace, never to apply it).  U_r is obtained from eigh(M^T M) [D x D], not from an
[F x D] SVD; a direct-SVD cross-check is run on one layer and reported.

C2 (the brief's random-projection floor) is NOT of that family -- B is random and A is
solved -- so it is implemented separately, exactly as registered.

--------------------------------------------------------------------------------------------
STAGES (separate processes on purpose; a probe on this project was silently evicted once)

    selfcheck   no model.  peak < 0.6 GB.  Planted-positive C4 end-to-end on synthetic data,
                                           plus algebraic checks of the closed form and of
                                           the FFN_active formula.  MUST PASS FIRST.
    capture     donor resident fp32.  peak ~8.0 GB.  One forward pass; writes per-layer
                                           H_fit (fp64), X_theta, X_eval to CACHE_DIR.
    analyse     no model.  peak ~1.6 GB.  All arms + C1..C4, per layer, per rank.
    ann         no model.  peak ~1.6 GB.  Amendment 1: IVF-PQ over W_g's rows.

Weights are read one layer at a time straight out of the pinned safetensors snapshot, so
`analyse` and `ann` never hold the donor.

--------------------------------------------------------------------------------------------
ALLOCATION INVENTORY -- every O(input) allocation, including the ones that fit.
See `_ALLOCATION_INVENTORY` at the bottom; it is emitted into the JSON so the write-up
cannot drift from the code.

Usage:
    python f1_gate_predictor.py selfcheck
    python f1_gate_predictor.py capture
    python f1_gate_predictor.py analyse
    python f1_gate_predictor.py ann
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY = os.path.abspath(os.path.join(HERE, "..", "density"))
sys.path.insert(0, DENSITY)

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Multi-GB intermediates live OUTSIDE the repo on purpose: `results/*` is a repo-root-anchored
# gitignore rule, so a nested results/ dir is NOT ignored, and 1.6 GB of activations must never
# become a candidate for `git add`.  Nothing here is an input to any committed number except
# through the JSON that `analyse` writes.
CACHE_DIR = os.environ.get("F1_CACHE", r"D:\_THINGS\_scratch_f1")

# ---------------------------------------------------------------- pre-registered constants
MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"     # pinned; same donor as D0-D4 and R2a

SEQ_LEN = 1024
N_FIT = 16            # calib sequences -> T_fit = 16384 tokens, T/D = 10.67
N_EVAL = 4            # HELD-OUT sequences -> T_eval = 4096 tokens (out-of-sample recall)
N_THETA = 2           # first 2 fit sequences -> T_theta = 2048 tokens, for the threshold rule
FIT_SEED = 20260822   # the seed D4/R2a used for calib slices (slice is already on disk)
EVAL_SEED = 20260828
RANKS = (8, 16, 32, 64, 128, 256)
EPS_LADDER = (0.001, 0.01, 0.05)     # FFN-output relative-error budgets for the threshold rule
EPS_PRIMARY = 0.01
RECALL_TARGETS = (0.90, 0.95, 0.99, 0.999, 1.0)
H_RIDGE_REL = 1e-10                  # eigenvalue floor on H, relative to lambda_max
RNG_C2 = 20260901                    # random-projection floor seed  (STATED)
RNG_C2B = 20260902                   # random-output-subspace floor seed (Builder addition)
RNG_C4 = 20260903                    # planted-positive seed
C4_TRUE_RANK = 32                    # ON the rank grid, so "recovers at r and not before" is testable
C3_LAYER_OFFSET = 14                 # wrong-layer control: layer L borrows from (L+14) mod n_layers

# Target width the brief prices the construction at (Llama-3-70B-class FFN geometry).
TARGET_D, TARGET_F = 8192, 28672

# ANN arm (Amendment 1)
ANN_NLIST_RULE = "round(sqrt(F))"    # the banked nlist ~ sqrt(N) law
ANN_M_SUBQ = 48                      # PQ subquantizers; dsub = D/m = 32 at D=1536
ANN_NBITS = 8                        # 256 codewords per subquantizer
ANN_NPROBE = (1, 2, 4, 8, 16, 32, 64, 95)
ANN_KMEANS_ITERS = 25
RNG_ANN = 20260904

ARMS_PROJECTOR = ("fit_actw", "fit_plain", "C2b_random_out",
                  "C3_wrong_layer_actw", "C3_wrong_layer_plain")


def log(*a):
    print(*a, flush=True)


def sha_arr(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


# ================================================================= donor plumbing (weights only)
def snapshot_dir() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)


class WeightReader:
    """One layer's MLP weights at a time, straight from the pinned safetensors file.

    Peak cost is 3 x [F, D] fp32 = 165 MB at this donor -- the donor itself is never
    materialised in `analyse` / `ann`.
    """

    def __init__(self):
        from safetensors import safe_open
        self.dir = snapshot_dir()
        self.path = os.path.join(self.dir, "model.safetensors")
        self._open = safe_open
        with safe_open(self.path, framework="np") as h:
            keys = list(h.keys())
        self.n_layers = 1 + max(
            int(k.split(".")[2]) for k in keys if k.startswith("model.layers."))
        g0 = "model.layers.0.mlp.gate_proj.weight"
        with safe_open(self.path, framework="np") as h:
            sl = h.get_slice(g0)
            self.F, self.D = [int(v) for v in sl.get_shape()]
            self.stored_dtype = sl.get_dtype()

    def mlp(self, layer: int):
        """(W_g, W_u, W_d) as fp32.  W_g,W_u: [F, D]; W_d: [D, F]."""
        import torch
        out = []
        with self._open(self.path, framework="pt") as h:
            for organ in ("gate_proj", "up_proj", "down_proj"):
                t = h.get_tensor(f"model.layers.{layer}.mlp.{organ}.weight")
                out.append(t.float().numpy())
        del torch
        return out

    def meta(self) -> dict:
        cfg = json.load(open(os.path.join(self.dir, "config.json")))
        return {
            "repo_id": MODEL_ID,
            "revision_pinned": REVISION,
            "snapshot_dir": self.dir,
            "snapshot_leaf": os.path.basename(self.dir),
            "config_sha256": hashlib.sha256(
                open(os.path.join(self.dir, "config.json"), "rb").read()).hexdigest(),
            "safetensors_bytes": os.path.getsize(self.path),
            # ACHIEVED, read off the tensor headers -- not off config prose:
            "n_layers_achieved": self.n_layers,
            "D_achieved": self.D,
            "F_achieved": self.F,
            "stored_dtype_achieved": self.stored_dtype,
            "hidden_act_config": cfg.get("hidden_act"),
            "hidden_size_config": cfg.get("hidden_size"),
            "intermediate_size_config": cfg.get("intermediate_size"),
        }


# ================================================================= closed-form core
def sym_sqrt(H: np.ndarray, ridge_rel: float = H_RIDGE_REL):
    """H^{1/2} for a PSD H, with an eigenvalue floor.  Returns (Hh, diagnostics)."""
    w, Q = np.linalg.eigh(H)
    wmax = float(w[-1])
    floor = ridge_rel * wmax
    n_floored = int((w < floor).sum())
    w = np.clip(w, floor, None)
    Hh = (Q * np.sqrt(w)) @ Q.T
    diag = {
        "eig_max": wmax,
        "eig_min_raw": float(np.linalg.eigvalsh(H)[0]) if H.shape[0] <= 2048 else None,
        "cond_after_floor_achieved": float(w[-1] / w[0]),
        "n_eigs_floored_achieved": n_floored,
        "ridge_rel_requested": ridge_rel,
        "ridge_abs_achieved": floor,
    }
    return Hh, diag


def left_subspace(M: np.ndarray, k: int):
    """Top-k LEFT singular vectors of M [F, D] via eigh(M^T M) [D, D].

    Returns (U [F,k] fp32, sigma_ALL [D] fp64 descending, diagnostics).  Squaring costs
    conditioning at the TAIL of the spectrum; only the head is used, and orthonormality is
    verified and reported.  The FULL sigma vector comes free from the same eigh, so no
    separate [F, D] SVD is ever run for the spectrum diagnostics.
    """
    K = M.T @ M                                  # [D, D] fp64
    w, V = np.linalg.eigh(K)                     # ascending
    ordd = np.argsort(w)[::-1]
    sig_all = np.sqrt(np.clip(w[ordd], 0.0, None))
    idx = ordd[:k]
    sig = sig_all[:k]
    V = V[:, idx]
    tol = max(sig[0], 1e-300) * 1e-12
    bad = int((sig <= tol).sum())
    U = (M @ V) / np.where(sig > tol, sig, 1.0)[None, :]
    orth = float(np.abs(U.T @ U - np.eye(k)).max())
    return U.astype(np.float32), sig_all, {"orthonormality_err_achieved": orth,
                                           "n_degenerate_sigma": bad,
                                           "sigma_head": float(sig_all[0]),
                                           "sigma_tail": float(sig_all[-1])}


def solve_random_B(Wg: np.ndarray, H: np.ndarray, B: np.ndarray):
    """Brief's C2: B fixed (random), A solved to minimise ||(W_g - AB)X||_F.

        A = W_g H B^T (B H B^T)^{-1}
    """
    HBt = H @ B.T                                 # [D, r]
    G = B @ HBt                                   # [r, r]
    G = G + np.eye(G.shape[0]) * (np.trace(G) / G.shape[0]) * 1e-10
    A = (Wg @ HBt) @ np.linalg.inv(G)             # [F, r]
    return A


# ================================================================= threshold rule
def silu(x):
    return x / (1.0 + np.exp(-x, dtype=x.dtype))


def theta_rule(g: np.ndarray, u: np.ndarray, Wd: np.ndarray, eps_list, n_refine: int = 14,
               n_tok_theta: int = 1024, n_tok_oracle: int = 512):
    """Define the donor's OPERATING THRESHOLD theta on the gate, and print it ACHIEVED.

    This donor is SwiGLU, not dReLU: there is no natural zero, so the threshold is a CHOICE
    and it is made explicit here.  The rule:

        theta(eps) = the LARGEST theta such that skipping every neuron with g_i <= theta
                     costs at most `eps` of the FFN block's output energy, measured as
                     ||Y - Y(theta)||_F / ||Y||_F on the calibration tokens.

    That is the rule an engine would actually implement (a one-sided compare on the gate),
    and it is exactly the rule the brief's cost model prices.  Its known imperfection is
    reported alongside: SiLU has a negative lobe (min -0.2785 at g = -1.2785), so a one-sided
    gate threshold cannot be a perfect proxy for |h_i| -- the `oracle_*` fields below bound
    how much that costs.

    g, u : [T, F] fp32.   Wd : [D, F] fp32.
    """
    g = g[:n_tok_theta]
    u = u[:n_tok_theta]
    T, F = g.shape
    h = (silu(g) * u).astype(np.float32)
    Y = h @ Wd.T
    denom = float(np.linalg.norm(Y))

    def rel_err(th):
        m = (g > th)
        drop = np.where(m, np.float32(0.0), h)
        return float(np.linalg.norm(drop @ Wd.T) / denom), float(m.mean())

    # coarse grid over quantiles of the pooled gate, then bisection refinement
    qs = np.linspace(0.0, 0.98, 13)
    grid = np.quantile(g.astype(np.float64), qs)
    curve = []
    for th in grid:
        e, a = rel_err(float(th))
        curve.append({"theta": float(th), "rel_err": e, "active_frac": a})

    out = {"grid_curve": curve, "T_theta_achieved": int(T), "F_achieved": int(F),
           "silu_negative_lobe_min": -0.2784645427610738}

    # theta = 0 : the dReLU-analogue "sign of the gate" rule, reported because it is the
    # obvious default and because its active fraction decides whether the idea can work at all.
    e0, a0 = rel_err(0.0)
    out["theta_zero"] = {"theta": 0.0, "rel_err_achieved": e0, "active_frac_achieved": a0}

    # oracle: what fraction of neurons an ORACLE on |h_i| would need for the same eps.  A
    # ceiling on any gate-only rule; the gap is the price of the SiLU negative lobe.
    ho = h[:n_tok_oracle]
    To = ho.shape[0]
    deno = float(np.linalg.norm(ho @ Wd.T))
    order = np.argsort(np.abs(ho), axis=1)        # ascending |h_i|, per token
    oracle = {}
    for eps in eps_list:
        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            kdrop = int(round(mid * F))
            mask = np.zeros((To, F), dtype=bool)
            if kdrop > 0:
                np.put_along_axis(mask, order[:, :kdrop], True, axis=1)
            e = float(np.linalg.norm(np.where(mask, ho, np.float32(0.0)) @ Wd.T) / deno)
            if e <= eps:
                lo = mid
            else:
                hi = mid
        oracle[str(eps)] = {"oracle_drop_frac": lo, "oracle_active_frac": 1.0 - lo,
                            "n_tokens_achieved": To}
    del order, ho
    out["oracle_magnitude"] = oracle
    out["note_oracle"] = ("An ORACLE on |h_i| = |SiLU(g_i) u_i|.  It is a CEILING on any "
                          "gate-only rule: the gap between oracle_active_frac and "
                          "active_frac_achieved is the price of SiLU's negative lobe plus the "
                          "price of ignoring W_u.  It is NOT reachable by the construction "
                          "under test, which reads only the gate.")

    # theta(eps) by bisection on the gate threshold
    sel = {}
    lo_th = float(grid[0]) - 1.0
    hi_th = float(g.max())
    for eps in eps_list:
        lo, hi = lo_th, hi_th                     # rel_err(lo) ~ 0 ; rel_err(hi) large
        e_lo, a_lo = rel_err(lo)
        if e_lo > eps:
            sel[str(eps)] = {"theta_achieved": None, "rel_err_achieved": e_lo,
                             "active_frac_achieved": a_lo,
                             "note": "budget unreachable even at theta = -inf"}
            continue
        best = (lo, e_lo, a_lo)
        for _ in range(n_refine + 8):
            mid = 0.5 * (lo + hi)
            e, a = rel_err(mid)
            if e <= eps:
                lo, best = mid, (mid, e, a)
            else:
                hi = mid
        sel[str(eps)] = {"theta_achieved": best[0], "rel_err_achieved": best[1],
                         "active_frac_achieved": best[2], "eps_requested": eps}
    out["theta_by_eps"] = sel
    del h, Y
    return out


# ================================================================= cost model
def predictor_term(r, D, F):
    return r * (D + F) / (D * F)


def ffn_active(r_or_pred_cost, s_frac, D, F, predictor_is_rank=True, n_organ_terms=3):
    """The brief's section 1 formula.  `s_frac` = |S|/F.  Fraction of dense-FFN bytes.

        FFN_active = [ r(D+F)/(D*F)  +  n * |S|/F ] / 3

    n = 3 (DEFAULT, and what section 1 registers): the predictor picks S, then the EXACT gate
        is recomputed on S -- so gate, up and down are each read on S.
    n = 2: the predictor's ghat is USED as the gate and never corrected, so only up and down
        are read on S.  This is the accounting behind section 4's identity
        FFN_active = (0.01 + 2(1-s))/3, and it is a DIFFERENT construction: it inherits the
        predictor's error into the activation values, not just into the choice of S.

    BOTH are reported everywhere.  See `BRIEF_TABLE_DISCREPANCY` below: the brief's own
    section 1 table mixes them -- rows a=0.10 and a=0.05 are n=3, row a=0.02 is n=2.
    """
    pred = predictor_term(r_or_pred_cost, D, F) if predictor_is_rank else r_or_pred_cost
    return (pred + float(n_organ_terms) * s_frac) / 3.0


BRIEF_TABLE_DISCREPANCY = {
    "what": ("BRIEF_F1 section 1's cost table is internally inconsistent, and the Builder has "
             "NOT altered the brief."),
    "detail": {
        "a=0.10": {"brief": 0.103, "n3_from_sec1_formula": 0.10335, "n2": 0.07002},
        "a=0.05": {"brief": 0.053, "n3_from_sec1_formula": 0.05335, "n2": 0.03668},
        "a=0.02": {"brief": 0.0167, "n3_from_sec1_formula": 0.02335, "n2": 0.01668},
    },
    "reading": ("Rows 1 and 2 are the n=3 accounting that section 1's own formula and prose "
                "specify (exact gate recomputed on S).  Row 3 is the n=2 accounting.  Under "
                "the registered n=3 formula the 2% target row is 2.33%, not 1.67% -- 40% "
                "higher, at exactly the operating point the brief cares about."),
    "knock_on": ("Section 4's 's = 0.97 reaches 2%' is likewise 2.33% under n=3.  Reaching a "
                 "true 0.02 needs s = 0.975 under n=2, or s = 0.9833 under n=3."),
    "builder_choice": ("n=3 is PRIMARY throughout, because it is what section 1's construction "
                       "actually does and it is the conservative one.  n=2 is reported beside "
                       "it everywhere as FFN_active_n2."),
}


# ================================================================= recall / |S| curve
def recall_curve(Ghat: np.ndarray, true_mask: np.ndarray, r: int, D: int, F: int,
                 recall_targets=RECALL_TARGETS, n_curve: int = 16):
    """The brief's section 3.1 measurement.  NEVER an accuracy.

    micro-recall(tau) = #{ ghat_i > tau AND g_i > theta } / #{ g_i > theta }
    |S|(tau)/F        = mean over tokens of #{ ghat_i > tau } / F
    """
    T = Ghat.shape[0]
    pos = Ghat[true_mask]                                  # predicted score AT true actives
    n_pos = int(pos.size)
    if n_pos == 0:
        return {"n_true_active": 0, "note": "no true actives"}
    pos_sorted = np.sort(pos)
    true_per_tok = true_mask.sum(axis=1)

    def size_at(tau):
        return float((Ghat > tau).sum()) / (T * F)

    def stats_at(tau, want_pertok=False):
        hit = (Ghat > tau) & true_mask
        rec = float(hit.sum()) / n_pos
        d = {"tau": float(tau), "recall_micro": rec, "S_frac": size_at(tau)}
        d["FFN_active"] = ffn_active(r, d["S_frac"], D, F)
        d["FFN_active_n2"] = ffn_active(r, d["S_frac"], D, F, n_organ_terms=2)
        d["FFN_active_target_width"] = ffn_active(r, d["S_frac"], TARGET_D, TARGET_F)
        d["FFN_active_target_width_n2"] = ffn_active(
            r, d["S_frac"], TARGET_D, TARGET_F, n_organ_terms=2)
        if want_pertok:
            with np.errstate(invalid="ignore", divide="ignore"):
                pr = hit.sum(axis=1) / np.maximum(true_per_tok, 1)
            pr = pr[true_per_tok > 0]
            d["recall_per_token_min"] = float(pr.min())
            d["recall_per_token_p01"] = float(np.quantile(pr, 0.01))
            d["recall_per_token_median"] = float(np.median(pr))
        return d

    at_target = {}
    for rho in recall_targets:
        # smallest tau achieving micro-recall >= rho is just below the (1-rho) quantile of pos
        k = int(math.floor((1.0 - rho) * n_pos))
        k = min(max(k, 0), n_pos - 1)
        tau = float(np.nextafter(pos_sorted[k], -np.inf))
        at_target[str(rho)] = stats_at(tau, want_pertok=(rho in (0.99, 0.999)))

    qs = np.linspace(0.02, 0.999, n_curve)
    flat = Ghat.ravel()[::7].astype(np.float64)                    # stride-7 subsample: GRID ONLY
    flat = flat[np.isfinite(flat)]
    curve = ([stats_at(float(t)) for t in np.unique(np.quantile(flat, qs))]
             if flat.size else [])
    return {"n_true_active": n_pos, "at_recall_target": at_target, "curve": curve}


# ================================================================= stage: selfcheck
def stage_selfcheck():
    t0 = time.time()
    rng = np.random.default_rng(RNG_C4)
    out = {"stage": "selfcheck", "checks": {}}
    D, F, T, r0 = 96, 512, 3000, C4_TRUE_RANK

    X = rng.standard_normal((T, D)).astype(np.float64)
    # give the data anisotropy, so H matters and the two arms can differ
    scale = np.exp(-np.arange(D) / 20.0)
    X = X * scale[None, :]
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    X = X @ Q.T
    H = X.T @ X
    Hh, hdiag = sym_sqrt(H)
    out["checks"]["H_sqrt"] = hdiag
    err = float(np.abs(Hh @ Hh - H).max() / np.abs(H).max())
    out["checks"]["H_sqrt_roundtrip_rel_err"] = err
    assert err < 1e-8, err

    # ---- algebraic check: U_r U_r^T W_g really is the minimiser of ||(W_g - AB)X||_F
    Wg = rng.standard_normal((F, D))
    M = Wg @ Hh
    for k in (8, 32):
        U, sig, d = left_subspace(M, k)
        approx = (U.astype(np.float64) @ (U.astype(np.float64).T @ Wg))
        got = np.linalg.norm((Wg - approx) @ X.T)
        # Eckart-Young: the optimum is sqrt(sum of the discarded squared singular values)
        allsig = np.linalg.svd(M, compute_uv=False)
        want = float(np.sqrt((allsig[k:] ** 2).sum()))
        rel = abs(got - want) / want
        out["checks"][f"eckart_young_rel_err_k{k}"] = rel
        out["checks"][f"orthonormality_k{k}"] = d["orthonormality_err_achieved"]
        assert rel < 1e-6, (k, got, want)
        # 200 random rank-k factorisations must all be WORSE -- the optimum is an optimum
        worse = 0
        for _ in range(200):
            Br = rng.standard_normal((k, D))
            Ar = solve_random_B(Wg, H, Br)
            worse += float(np.linalg.norm((Wg - Ar @ Br) @ X.T)) > got - 1e-9
        out["checks"][f"random_factorisations_worse_k{k}"] = f"{worse}/200"
        assert worse == 200

    # ---- direct-SVD cross-check of left_subspace (the eigh(M^T M) shortcut)
    Ud, sd, _ = np.linalg.svd(M, full_matrices=False)
    U32, sig32, _ = left_subspace(M, 32)
    align = float(np.abs(np.abs(Ud[:, :32].T @ U32.astype(np.float64)) - np.eye(32)).max())
    out["checks"]["direct_svd_subspace_alignment_err"] = align
    out["checks"]["direct_svd_sigma_rel_err"] = float(np.abs(sd[:32] - sig32[:32]).max() / sd[0])
    assert align < 1e-6

    # ---- C4 PLANTED POSITIVE, end to end through the real measurement path
    P = np.linalg.qr(rng.standard_normal((F, r0)))[0]
    Qp = rng.standard_normal((r0, D))
    Wg_plant = P @ Qp                                     # EXACTLY rank r0
    rank_ach = int(np.linalg.matrix_rank(Wg_plant, tol=1e-8 * np.abs(Wg_plant).max()))
    out["checks"]["C4_planted_rank_achieved"] = rank_ach
    out["checks"]["C4_planted_rank_requested"] = r0
    assert rank_ach == r0

    Mp = Wg_plant @ Hh
    Gp = (X @ Wg_plant.T).astype(np.float32)
    theta = float(np.quantile(Gp, 0.95))                  # 5% active, stated
    tmask = Gp > theta
    fired = {}
    for k in (8, 16, 32, 64):
        U, _, _ = left_subspace(Mp, k)
        Gh = (Gp @ U) @ U.T
        rc = recall_curve(Gh, tmask, k, D, F)
        fired[str(k)] = {
            "recall_at_tau_theta": float(((Gh > theta) & tmask).sum() / tmask.sum()),
            "S_frac_at_tau_theta": float((Gh > theta).mean()),
            "max_abs_ghat_minus_g": float(np.abs(Gh - Gp).max()),
            "S_frac_at_recall_1.0": rc["at_recall_target"]["1.0"]["S_frac"],
        }
    out["checks"]["C4_planted_firing"] = fired
    # the instrument must recover EXACTLY at the planted rank and NOT before
    assert fired["32"]["max_abs_ghat_minus_g"] < 1e-6 * float(np.abs(Gp).max())
    assert fired["64"]["max_abs_ghat_minus_g"] < 1e-6 * float(np.abs(Gp).max())
    assert fired["16"]["recall_at_tau_theta"] < 0.999
    assert fired["8"]["recall_at_tau_theta"] < fired["16"]["recall_at_tau_theta"] + 1e-9
    out["checks"]["C4_verdict"] = ("FIRES: exact at r=32=planted rank, degraded at r=16 and "
                                   "worse at r=8")

    # ---- C1 full-rank tautology
    Ufull, _, _ = left_subspace(Wg @ Hh, D)
    Ghf = ((X @ Wg.T).astype(np.float32) @ Ufull) @ Ufull.T
    Gf = (X @ Wg.T).astype(np.float32)
    out["checks"]["C1_fullrank_max_abs_dev"] = float(np.abs(Ghf - Gf).max())
    out["checks"]["C1_fullrank_rel_dev"] = float(np.abs(Ghf - Gf).max() / np.abs(Gf).max())
    assert out["checks"]["C1_fullrank_rel_dev"] < 1e-5

    # ---- cost formula, checked against the brief's own printed table (section 1)
    tbl = {}
    for a in (0.10, 0.05, 0.02):
        tbl[str(a)] = {"n3": ffn_active(64, a, TARGET_D, TARGET_F),
                       "n2": ffn_active(64, a, TARGET_D, TARGET_F, n_organ_terms=2)}
    out["checks"]["brief_table_reproduced"] = tbl
    out["checks"]["brief_table_as_printed"] = {"0.1": 0.103, "0.05": 0.053, "0.02": 0.0167}
    out["checks"]["BRIEF_TABLE_DISCREPANCY"] = BRIEF_TABLE_DISCREPANCY
    # rows 1 and 2 must reproduce under n=3 ...
    for a, want in ((0.10, 0.103), (0.05, 0.053)):
        assert abs(tbl[str(a)]["n3"] - want) < 6e-4, (a, tbl[str(a)], want)
    # ... and row 3 must reproduce only under n=2, which is the discrepancy itself
    assert abs(tbl["0.02"]["n2"] - 0.0167) < 6e-4
    assert abs(tbl["0.02"]["n3"] - 0.0167) > 5e-3
    out["checks"]["predictor_term_r64_target_width"] = predictor_term(64, TARGET_D, TARGET_F)
    out["checks"]["predictor_term_r64_achieved_width_1p5B"] = predictor_term(64, 1536, 8960)

    # ---- theta_rule sanity on a synthetic SwiGLU block
    gg = (X @ Wg.T).astype(np.float32)
    Wu = rng.standard_normal((F, D)).astype(np.float32)
    Wd = rng.standard_normal((D, F)).astype(np.float32)
    uu = (X.astype(np.float32) @ Wu.T)
    th = theta_rule(gg[:512], uu[:512], Wd, (0.01,))
    out["checks"]["theta_rule_synthetic"] = {
        "theta_zero": th["theta_zero"],
        "theta_by_eps": th["theta_by_eps"],
        "oracle_magnitude": th["oracle_magnitude"],
    }
    mono = [c["rel_err"] for c in th["grid_curve"]]
    out["checks"]["theta_grid_monotone"] = bool(all(
        mono[i] <= mono[i + 1] + 1e-6 for i in range(len(mono) - 1)))
    assert out["checks"]["theta_grid_monotone"]

    out["elapsed_s"] = time.time() - t0
    out["env"] = env_block()
    out["allocation_inventory"] = _ALLOCATION_INVENTORY
    p = os.path.join(RESULTS, "f1_selfcheck.json")
    json.dump(out, open(p, "w"), indent=2, default=float)
    log(f"SELFCHECK PASS  ({out['elapsed_s']:.1f}s) -> {p}")
    log(json.dumps(out["checks"]["C4_planted_firing"], indent=2))
    return out


def env_block():
    import numpy
    b = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        b["torch"] = torch.__version__
        b["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    try:
        import transformers
        b["transformers"] = transformers.__version__
    except Exception:
        pass
    return b


# ================================================================= stage: capture
def stage_capture():
    """One forward pass over the donor.  Writes per-layer H_fit / X_theta / X_eval."""
    import torch
    import common as C

    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    torch.set_grad_enabled(False)
    model, tok = C.load_model(dtype=torch.float32)
    arch = C.arch(model)
    log(f"donor loaded: {arch}")

    ids_fit, byts_fit, meta_fit = C.get_slice(tok, "calib", N_FIT, SEQ_LEN, FIT_SEED)
    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", N_EVAL, SEQ_LEN, EVAL_SEED)
    L = len(model.model.layers)
    D = arch["d_model"]

    manifest = {"stage": "capture", "arch_achieved": arch,
                "slice_fit": meta_fit, "slice_eval": meta_ev,
                "T_fit_achieved": int(ids_fit.numel()), "T_eval_achieved": int(ids_ev.numel()),
                "T_theta_achieved": int(N_THETA * SEQ_LEN),
                "cache_dir": CACHE_DIR, "env": env_block()}

    def run(ids, want_H, want_store, tag, batch=2):
        acc = {l: (np.zeros((D, D), dtype=np.float64) if want_H else None) for l in range(L)}
        store = {l: [] for l in range(L)} if want_store else None
        n_tok = 0
        hooks = []

        def mk(l):
            def hook(mod, args):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float().numpy()
                if want_H:
                    acc[l] += x.astype(np.float64).T @ x.astype(np.float64)
                if want_store is not None and store is not None:
                    if want_store == "all" or len(store[l]) * x.shape[0] < want_store:
                        store[l].append(x.copy())
            return model.model.layers[l].mlp.gate_proj.register_forward_pre_hook(hook)

        for l in range(L):
            hooks.append(mk(l))
        for i in range(0, ids.shape[0], batch):
            model.model(ids[i:i + batch])
            n_tok += int(ids[i:i + batch].numel())
            log(f"  [{tag}] {n_tok}/{ids.numel()} tokens  {time.time()-t0:.0f}s")
        for h in hooks:
            h.remove()
        return acc, store, n_tok

    # FIT pass: accumulate H online (X_fit is never materialised in full), keep the first
    # N_THETA*SEQ_LEN rows for the threshold rule.
    accH, store_theta, n_fit = run(ids_fit, True, N_THETA * SEQ_LEN, "fit")
    for l in range(L):
        Xt = np.concatenate(store_theta[l], 0)[:N_THETA * SEQ_LEN]
        np.save(os.path.join(CACHE_DIR, f"H_fit_L{l:02d}.npy"), accH[l])
        np.save(os.path.join(CACHE_DIR, f"X_theta_L{l:02d}.npy"), Xt.astype(np.float32))
        store_theta[l] = None
        accH[l] = None
    del accH, store_theta

    # EVAL pass: HELD-OUT corpus part, stored in full (recall must be out-of-sample).
    _, store_ev, n_ev = run(ids_ev, False, "all", "eval")
    for l in range(L):
        Xe = np.concatenate(store_ev[l], 0)
        np.save(os.path.join(CACHE_DIR, f"X_eval_L{l:02d}.npy"), Xe.astype(np.float32))
        store_ev[l] = None
    del store_ev, model

    manifest["n_tokens_fit_achieved"] = n_fit
    manifest["n_tokens_eval_achieved"] = n_ev
    manifest["elapsed_s"] = time.time() - t0
    manifest["allocation_inventory"] = _ALLOCATION_INVENTORY
    p = os.path.join(RESULTS, "f1_capture.json")
    json.dump(manifest, open(p, "w"), indent=2, default=float)
    log(f"CAPTURE done ({manifest['elapsed_s']:.0f}s) -> {p}")
    return manifest


def load_cached(l):
    H = np.load(os.path.join(CACHE_DIR, f"H_fit_L{l:02d}.npy"))
    Xt = np.load(os.path.join(CACHE_DIR, f"X_theta_L{l:02d}.npy"))
    Xe = np.load(os.path.join(CACHE_DIR, f"X_eval_L{l:02d}.npy"))
    return H, Xt, Xe


# ================================================================= stage: analyse
def spectrum_facts(sig):
    """Distributional facts about W_g's spectrum, reported in their own right."""
    e = sig ** 2
    tot = float(e.sum())
    c = np.cumsum(e) / tot
    return {
        "n_singular_values": int(sig.size),
        "sigma_max": float(sig[0]),
        "sigma_min": float(sig[-1]),
        "condition_number": float(sig[0] / max(sig[-1], 1e-300)),
        "participation_ratio_effective_rank": float(tot ** 2 / float((e ** 2).sum())),
        "stable_rank": float(tot / float(e[0])),
        "rank_for_energy_50pct": int(np.searchsorted(c, 0.50) + 1),
        "rank_for_energy_90pct": int(np.searchsorted(c, 0.90) + 1),
        "rank_for_energy_99pct": int(np.searchsorted(c, 0.99) + 1),
        "sigma_decimated_every_16": [float(x) for x in sig[::16]],
    }


def stage_analyse(layers=None):
    t0 = time.time()
    wr = WeightReader()
    D, F = wr.D, wr.F
    L = wr.n_layers
    layers = list(range(L)) if layers is None else layers
    rmax = max(RANKS)

    out = {"stage": "analyse", "donor": wr.meta(), "env": env_block(),
           "pre_registered": {
               "ranks": list(RANKS), "eps_ladder": list(EPS_LADDER),
               "eps_primary": EPS_PRIMARY, "recall_targets": list(RECALL_TARGETS),
               "H_ridge_rel": H_RIDGE_REL, "C4_true_rank": C4_TRUE_RANK,
               "C3_layer_offset": C3_LAYER_OFFSET,
               "seeds": {"C2_base": RNG_C2, "C2b_base": RNG_C2B, "C4_base": RNG_C4,
                         "fit": FIT_SEED, "eval": EVAL_SEED}},
           "null_construction_policy": (
               "Every null is rebuilt PER LAYER and PER RANK, and matched per-arm: C2 draws a "
               "fresh B for each (layer, rank); C2b draws a fresh random output subspace for "
               "each (layer, rank); C3 exists in BOTH weightings, so the activation-weighted "
               "arm is nulled against a wrong-layer ACTIVATION-WEIGHTED subspace and the plain "
               "arm against a wrong-layer PLAIN one.  A null borrowed from a neighbouring "
               "configuration is not a null."),
           "cost_model": {
               "formula": "FFN_active = [ r(D+F)/(D*F) + 3|S|/F ] / 3",
               "D_achieved": D, "F_achieved": F,
               "predictor_term_by_rank_achieved_width": {
                   str(r): r * (D + F) / (D * F) for r in RANKS},
               "predictor_term_by_rank_target_width": {
                   str(r): r * (TARGET_D + TARGET_F) / (TARGET_D * TARGET_F) for r in RANKS},
               "target_width": {"D": TARGET_D, "F": TARGET_F},
               "n_organ_terms_primary": 3,
               "BRIEF_TABLE_DISCREPANCY": BRIEF_TABLE_DISCREPANCY},
           "spectra": {}, "layers": {}}

    # ---------------------------------------------------------------- PASS A: subspace bank
    # Both weightings for EVERY layer, so C3 can pair layer L with (L+14) mod n_layers in the
    # MATCHING weighting.  28 x 2 x [F, 256] fp32 = 0.51 GB.
    bankA, bankP = {}, {}
    for l in range(L):
        H, _, _ = load_cached(l)
        Wg, _, _ = wr.mlp(l)
        Wg64 = Wg.astype(np.float64)
        Hh, hd = sym_sqrt(H)
        Ua, sig_aw, da = left_subspace(Wg64 @ Hh, rmax)
        Up, sig_pl, dp = left_subspace(Wg64, rmax)
        bankA[l], bankP[l] = Ua, Up
        out["spectra"][str(l)] = {
            "H_diag": hd,
            "actw": spectrum_facts(sig_aw), "plain": spectrum_facts(sig_pl),
            "actw_orthonormality_err": da["orthonormality_err_achieved"],
            "plain_orthonormality_err": dp["orthonormality_err_achieved"],
            "energy_captured_actw": {
                str(r): float((sig_aw[:r] ** 2).sum() / (sig_aw ** 2).sum()) for r in RANKS},
            "energy_captured_plain": {
                str(r): float((sig_pl[:r] ** 2).sum() / (sig_pl ** 2).sum()) for r in RANKS},
        }
        if l == 0:      # one direct-SVD cross-check of the eigh(M^T M) shortcut
            Ud = np.linalg.svd(Wg64 @ Hh, full_matrices=False)[0]
            k = 64
            out["direct_svd_crosscheck"] = {
                "layer": 0, "k": k,
                "subspace_alignment_err": float(np.abs(
                    np.abs(Ud[:, :k].T @ Ua[:, :k].astype(np.float64)) - np.eye(k)).max())}
            del Ud
        del H, Wg, Wg64, Hh
        log("  [bank] layer %2d  eff_rank_actw=%.1f  eff_rank_plain=%.1f  (%.0fs)" % (
            l, out["spectra"][str(l)]["actw"]["participation_ratio_effective_rank"],
            out["spectra"][str(l)]["plain"]["participation_ratio_effective_rank"],
            time.time() - t0))
    json.dump(out, open(os.path.join(RESULTS, "f1_analyse.json"), "w"), indent=2, default=float)

    # ---------------------------------------------------------------- PASS B: measurement
    for l in layers:
        tl = time.time()
        H, Xt, Xe = load_cached(l)
        Wg, Wu, Wd = wr.mlp(l)
        Wg64 = Wg.astype(np.float64)
        Hh, _ = sym_sqrt(H)
        rec = {"layer": l, "T_theta_stored": int(Xt.shape[0]),
               "T_eval_achieved": int(Xe.shape[0]),
               "X_eval_sha16": sha_arr(Xe), "H_sha16": sha_arr(H),
               "Wg_sha16": sha_arr(Wg)}

        # ---- 1. the donor's TRUE active set: the operating threshold, ACHIEVED
        # only the first 1024 stored theta-tokens are used; slice BEFORE the matmul
        Xt1 = Xt[:1024]
        rec["threshold_rule"] = theta_rule(Xt1 @ Wg.T, Xt1 @ Wu.T, Wd, EPS_LADDER)
        th_p = rec["threshold_rule"]["theta_by_eps"][str(EPS_PRIMARY)]
        theta = th_p["theta_achieved"]
        rec["theta_primary_achieved"] = theta
        rec["a_calib_achieved"] = th_p["active_frac_achieved"]
        rec["rel_err_at_theta_achieved"] = th_p["rel_err_achieved"]

        # ---- 2. exact gate on the HELD-OUT slice
        G = (Xe @ Wg.T).astype(np.float32)
        true_mask = G > np.float32(theta)
        a_eval = float(true_mask.mean())
        rec["a_eval_achieved"] = a_eval
        pt = true_mask.mean(axis=1)
        rec["a_eval_per_token"] = {"min": float(pt.min()), "p01": float(np.quantile(pt, 0.01)),
                                   "median": float(np.median(pt)), "max": float(pt.max())}
        # a distributional fact about this donor's gate, reported in its own right
        qs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
        rec["gate_distribution_eval"] = {
            "quantiles": dict((str(q), float(v)) for q, v in
                              zip(qs, np.quantile(G.astype(np.float64), qs))),
            "mean": float(G.mean()), "std": float(G.std()),
            "frac_gate_positive": float((G > 0).mean())}
        # the three reference points every recall number must be read against
        rec["baseline_dense_gate_FFN_active"] = (1.0 + 3.0 * a_eval) / 3.0
        rec["floor_if_a_zero_dense_gate"] = 1.0 / 3.0
        rec["ideal_FFN_active_free_perfect_predictor"] = a_eval

        # ---- 3. arms.  Every projector arm is  ghat = G U_r U_r^T .
        arms = {}
        subs_static = {"fit_actw": bankA[l], "fit_plain": bankP[l],
                       "C3_wrong_layer_actw": bankA[(l + C3_LAYER_OFFSET) % L],
                       "C3_wrong_layer_plain": bankP[(l + C3_LAYER_OFFSET) % L]}
        rec["C3_borrowed_from_layer"] = (l + C3_LAYER_OFFSET) % L
        for name, U in subs_static.items():
            GU = G @ U
            arms[name] = dict(
                (str(r), recall_curve(GU[:, :r] @ U[:, :r].T, true_mask, r, D, F))
                for r in RANKS)
            del GU

        # C2b -- Builder addition, DRAWN FRESH PER (layer, rank).  The matched null for the
        # fitted arms: same family, same cost, random subspace instead of a fitted one.  Without
        # it, a win over C2 could be a win of FAMILY rather than of FIT.
        c2b = {}
        for r in RANKS:
            gg = np.random.default_rng(RNG_C2B + 1000 * l + r)
            Ur = np.linalg.qr(gg.standard_normal((F, r)))[0].astype(np.float32)
            c2b[str(r)] = recall_curve((G @ Ur) @ Ur.T, true_mask, r, D, F)
            del Ur
        arms["C2b_random_out"] = c2b

        # C2 -- the brief's registered floor: B random, A solved.  FRESH PER (layer, rank).
        c2 = {}
        for r in RANKS:
            gg = np.random.default_rng(RNG_C2 + 1000 * l + r)
            B = gg.standard_normal((r, D))
            A = solve_random_B(Wg64, H, B)
            Ghat = ((Xe.astype(np.float64) @ B.T) @ A.T).astype(np.float32)
            c2[str(r)] = recall_curve(Ghat, true_mask, r, D, F)
            del Ghat, A, B
        arms["C2_random_proj"] = c2

        # C1 -- full rank.  A TAUTOLOGY.
        Ufull, _, _ = left_subspace(Wg64 @ Hh, D)
        Ghat = (G @ Ufull) @ Ufull.T
        c1 = recall_curve(Ghat, true_mask, D, D, F)
        c1["max_abs_ghat_minus_g"] = float(np.abs(Ghat - G).max())
        c1["rel_dev"] = float(np.abs(Ghat - G).max() / np.abs(G).max())
        c1["IS_A_TAUTOLOGY"] = ("r = D = rank(W_g); U_D spans colspace(W_g), so ghat == g "
                                "identically.  Tests the code path, not the science.")
        arms["C1_full_rank"] = {str(D): c1}
        del Ufull, Ghat

        # C4 -- planted positive, on THIS layer's real activations and real H.
        g4 = np.random.default_rng(RNG_C4 + l)
        Pp = np.linalg.qr(g4.standard_normal((F, C4_TRUE_RANK)))[0]
        Wg_p = Pp @ g4.standard_normal((C4_TRUE_RANK, D))
        rank_ach = int(np.linalg.matrix_rank(Wg_p, tol=1e-8 * np.abs(Wg_p).max()))
        Gp = (Xe.astype(np.float64) @ Wg_p.T).astype(np.float32)
        theta_p = float(np.quantile(Gp, 1.0 - a_eval))   # SAME active fraction as the real layer
        tmask_p = Gp > np.float32(theta_p)
        Up4, _, _ = left_subspace(Wg_p @ Hh, rmax)
        c4 = {"planted_rank_requested": C4_TRUE_RANK, "planted_rank_achieved": rank_ach,
              "theta_planted_achieved": theta_p,
              "active_frac_planted_achieved": float(tmask_p.mean()),
              "matched_to_real_layer_a": a_eval, "by_rank": {}}
        for r in RANKS:
            Gh = (Gp @ Up4[:, :r]) @ Up4[:, :r].T
            e = recall_curve(Gh, tmask_p, r, D, F)
            e["max_abs_ghat_minus_g"] = float(np.abs(Gh - Gp).max())
            e["rel_dev"] = float(np.abs(Gh - Gp).max() / np.abs(Gp).max())
            e["recall_at_tau_eq_theta"] = float(
                ((Gh > np.float32(theta_p)) & tmask_p).sum() / max(1, tmask_p.sum()))
            c4["by_rank"][str(r)] = e
            del Gh
        c4["FIRES"] = bool(c4["by_rank"][str(C4_TRUE_RANK)]["rel_dev"] < 1e-4
                           and c4["by_rank"]["16"]["rel_dev"] > 1e-3)
        arms["C4_planted"] = c4
        del Pp, Wg_p, Gp, tmask_p, Up4

        rec["arms"] = arms
        rec["elapsed_s"] = time.time() - tl
        out["layers"][str(l)] = rec
        del G, true_mask, Wg, Wu, Wd, Wg64, Hh, H, Xt, Xe

        def hi(arm, r, tgt="0.99"):
            d = arms[arm][str(r)].get("at_recall_target", {}).get(tgt)
            return d["S_frac"] if d else float("nan")
        log("  layer %2d  a=%.4f theta=%.3f | S_frac@rec.99 r=64: fit=%.3f plain=%.3f "
            "C2=%.3f C2b=%.3f C3=%.3f  (%.0fs)" % (
                l, a_eval, theta, hi("fit_actw", 64), hi("fit_plain", 64),
                hi("C2_random_proj", 64), hi("C2b_random_out", 64),
                hi("C3_wrong_layer_actw", 64), rec["elapsed_s"]))
        out["elapsed_s"] = time.time() - t0
        out["allocation_inventory"] = _ALLOCATION_INVENTORY
        json.dump(out, open(os.path.join(RESULTS, "f1_analyse.json"), "w"),
                  indent=2, default=float)      # checkpoint every layer

    log("ANALYSE done (%.0fs)" % (time.time() - t0))
    return out


# ================================================================= stage: ann (Amendment 1)
def kmeans(Xn, k, iters, rng, minit=None):
    n = Xn.shape[0]
    C = Xn[rng.choice(n, size=k, replace=False)].copy() if minit is None else minit.copy()
    for _ in range(iters):
        d = (Xn @ C.T)
        d = (Xn * Xn).sum(1)[:, None] - 2 * d + (C * C).sum(1)[None, :]
        a = np.argmin(d, axis=1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = Xn[m].mean(0)
            else:
                C[j] = Xn[rng.integers(n)]
    return C, a


def stage_ann(layers=None):
    """Amendment 1: IVF-PQ over W_g's rows, treated as keys.

    Recall is measured against the TRUE active set from the EXACT gate (A1.4 point 2), never
    against the index's own ranking.  The index's own bytes and per-token work are counted
    into FFN_active (A1.4 point 1), not into a footnote.  Build cost is stated (point 3).
    """
    t0 = time.time()
    wr = WeightReader()
    D, F = wr.D, wr.F
    rng = np.random.default_rng(RNG_ANN)
    nlist = int(round(math.sqrt(F)))
    dsub = D // ANN_M_SUBQ
    assert dsub * ANN_M_SUBQ == D, (D, ANN_M_SUBQ)

    an = json.load(open(os.path.join(RESULTS, "f1_analyse.json")))
    layers = ([int(k) for k in an["layers"].keys()] if layers is None else layers)

    out = {"stage": "ann", "donor": wr.meta(), "env": env_block(),
           "pre_registered": {"nlist_rule": ANN_NLIST_RULE, "nlist_achieved": nlist,
                              "m_subq": ANN_M_SUBQ, "dsub_achieved": dsub,
                              "nbits": ANN_NBITS, "nprobe": list(ANN_NPROBE),
                              "kmeans_iters": ANN_KMEANS_ITERS, "seed": RNG_ANN},
           "layers": {}}

    # ---- index cost model, stated once, in BOTH widths
    def index_cost(D_, F_, nlist_, m_, nprobe_, s_frac_, code_bytes=1, w_bytes=0.25,
                   cb_bytes=2):
        """Per-token cost of the IVF-PQ path as a fraction of the dense FFN.

        w_bytes   : bytes per weight element assumed for W_g/W_u/W_d (0.25 = ternary-packed)
        cb_bytes  : bytes per centroid/codebook element (2 = fp16)
        code_bytes: bytes per PQ code (1 = 8-bit)
        """
        cand = nprobe_ / nlist_ * F_
        macs_dense = 3.0 * D_ * F_
        macs = (nlist_ * D_                    # coarse: <c_j, x> for every list
                + 256 * D_                     # PQ lookup-table build (256 codewords x D)
                + cand * m_                    # table lookups over candidates
                + 3.0 * s_frac_ * F_ * D_)     # exact gate/up/down on S
        by_dense = 3.0 * D_ * F_ * w_bytes
        by = (nlist_ * D_ * cb_bytes           # coarse centroids
              + 256 * D_ * cb_bytes            # PQ codebooks
              + cand * m_ * code_bytes         # codes of the probed lists
              + 3.0 * s_frac_ * F_ * D_ * w_bytes)
        return {"macs_frac": macs / macs_dense, "bytes_frac": by / by_dense,
                "index_resident_bytes": (nlist_ * D_ + 256 * D_) * cb_bytes + F_ * m_ * code_bytes,
                "candidates": cand}

    out["index_cost_model"] = {
        "note": ("Codebooks and inverted lists are weights that must be READ or HELD; they are "
                 "inside FFN_active, per A1.4 point 1.  bytes_frac assumes W_* at 0.25 B/elem "
                 "(ternary-packed, the engine's target) and centroids/codebooks at fp16."),
        "achieved_width": {"D": D, "F": F, "nlist": nlist, "m": ANN_M_SUBQ},
        "target_width": {"D": TARGET_D, "F": TARGET_F,
                         "nlist": int(round(math.sqrt(TARGET_F))), "m": ANN_M_SUBQ},
        "L3_reference_bytes": 16 * 1024 * 1024,
    }
    for tag, (D_, F_, nl_) in {
            "achieved_width": (D, F, nlist),
            "target_width": (TARGET_D, TARGET_F, int(round(math.sqrt(TARGET_F))))}.items():
        ic = index_cost(D_, F_, nl_, ANN_M_SUBQ, 8, 0.05)
        out["index_cost_model"][tag + "_example_nprobe8_s0.05"] = ic
        out["index_cost_model"][tag + "_index_bytes_vs_L3"] = (
            ic["index_resident_bytes"] / (16 * 1024 * 1024))

    for l in layers:
        tl = time.time()
        H, Xt, Xe = load_cached(l)
        Wg, Wu, Wd = wr.mlp(l)
        theta = an["layers"][str(l)]["theta_primary_achieved"]
        G = (Xe @ Wg.T).astype(np.float32)
        true_mask = G > np.float32(theta)
        a_eval = float(true_mask.mean())

        # ---- build (offline, once per layer -- A1.4 point 3)
        tb = time.time()
        K = Wg.astype(np.float64)                       # the keys
        cent, assign = kmeans(K, nlist, ANN_KMEANS_ITERS, rng)
        resid = K - cent[assign]
        codebooks = np.zeros((ANN_M_SUBQ, 256, dsub))
        codes = np.zeros((F, ANN_M_SUBQ), dtype=np.uint8)
        for m in range(ANN_M_SUBQ):
            sub = resid[:, m * dsub:(m + 1) * dsub]
            cb, aa = kmeans(sub, 256, 10, rng)
            codebooks[m] = cb
            codes[:, m] = aa.astype(np.uint8)
        build_s = time.time() - tb

        # reconstruction fidelity of the index's own scores (PQ is lossy -- A1.4 point 2)
        Krec = cent[assign] + np.concatenate(
            [codebooks[m][codes[:, m]] for m in range(ANN_M_SUBQ)], axis=1)
        rec_err = float(np.linalg.norm(K - Krec) / np.linalg.norm(K))

        # ---- query: coarse scores, then PQ-approximate scores on probed lists
        Xe64 = Xe.astype(np.float64)
        coarse = Xe64 @ cent.T                          # [T, nlist]
        order = np.argsort(-coarse, axis=1)
        # exact approximate-score matrix, computed once at full nprobe, then masked per nprobe
        approx_all = (Xe64 @ Krec.T).astype(np.float32)  # what the index would score
        listof = assign                                  # [F]
        per_nprobe = {}
        for npb in ANN_NPROBE:
            if npb > nlist:
                continue
            probed = order[:, :npb]                     # [T, npb]
            sel = np.zeros((Xe.shape[0], nlist), dtype=bool)
            np.put_along_axis(sel, probed, True, axis=1)
            reach = sel[:, listof]                      # [T, F] retrievable this token
            Gh = np.where(reach, approx_all, np.float32(-np.inf))
            rc = recall_curve(Gh, true_mask, 0, D, F)
            # replace the rank-based predictor term with the index's own cost
            for key, d in list(rc.get("at_recall_target", {}).items()):
                ic = index_cost(D, F, nlist, ANN_M_SUBQ, npb, d["S_frac"])
                ict = index_cost(TARGET_D, TARGET_F, int(round(math.sqrt(TARGET_F))),
                                 ANN_M_SUBQ, npb, d["S_frac"])
                d["FFN_active"] = ic["macs_frac"]
                d["FFN_active_bytes"] = ic["bytes_frac"]
                d["FFN_active_target_width"] = ict["macs_frac"]
                d["FFN_active_bytes_target_width"] = ict["bytes_frac"]
            rc["reachable_frac"] = float(reach.mean())
            rc["ceiling_recall_if_all_reached"] = float(
                (reach & true_mask).sum() / max(1, true_mask.sum()))
            per_nprobe[str(npb)] = rc
            del Gh, sel, reach

        out["layers"][str(l)] = {
            "layer": l, "a_eval_achieved": a_eval, "theta_used": theta,
            "nlist_achieved": nlist, "list_size_mean": float(F / nlist),
            "list_size_max_achieved": int(np.bincount(assign, minlength=nlist).max()),
            "list_size_min_achieved": int(np.bincount(assign, minlength=nlist).min()),
            "pq_reconstruction_rel_err_achieved": rec_err,
            "build_seconds_achieved": build_s,
            "by_nprobe": per_nprobe,
        }
        log(f"  ANN layer {l:2d}  build={build_s:.0f}s  pq_err={rec_err:.3f}  "
            f"({time.time()-tl:.0f}s, total {time.time()-t0:.0f}s)")
        out["elapsed_s"] = time.time() - t0
        out["allocation_inventory"] = _ALLOCATION_INVENTORY
        json.dump(out, open(os.path.join(RESULTS, "f1_ann.json"), "w"), indent=2, default=float)
        del G, true_mask, Wg, Wu, Wd, K, Krec, approx_all, coarse, order, Xe64, Xe, Xt, H

    log(f"ANN done ({time.time()-t0:.0f}s)")
    return out


# ================================================================= allocation inventory
_ALLOCATION_INVENTORY = [
    {"stage": "capture", "object": "donor fp32 (transformers)", "shape": "1.54e9 params",
     "bytes": 6.2e9, "note": "the dominant term; freed before analyse runs at all"},
    {"stage": "capture", "object": "H_fit accumulator", "shape": "L x [D, D] fp64",
     "bytes": 28 * 1536 * 1536 * 8, "note": "0.53 GB; X_fit is NEVER materialised -- H is "
                                            "accumulated online, batch by batch"},
    {"stage": "capture", "object": "X_theta", "shape": "L x [2048, D] fp32",
     "bytes": 28 * 2048 * 1536 * 4, "note": "0.35 GB"},
    {"stage": "capture", "object": "X_eval", "shape": "L x [4096, D] fp32",
     "bytes": 28 * 4096 * 1536 * 4, "note": "0.70 GB"},
    {"stage": "capture", "object": "forward activations, batch=2 x 1024", "shape": "transient",
     "bytes": 3e8, "note": "eager attention, 2 x 12 x 1024^2 fp32 scores per layer, freed"},
    {"stage": "capture", "object": "PEAK", "bytes": 8.0e9, "note": "6.2 + 1.6 + transients"},

    {"stage": "analyse", "object": "W_g, W_u, W_d fp32 (one layer)", "shape": "3 x [F, D]",
     "bytes": 3 * 8960 * 1536 * 4, "note": "0.165 GB; read straight from safetensors, one "
                                           "layer at a time -- the donor is never resident"},
    {"stage": "analyse", "object": "H, H^{1/2}, K, V fp64", "shape": "4 x [D, D]",
     "bytes": 4 * 1536 * 1536 * 8, "note": "0.075 GB"},
    {"stage": "analyse", "object": "M = W_g H^{1/2} fp64", "shape": "[F, D]",
     "bytes": 8960 * 1536 * 8, "note": "0.11 GB, transient per subspace"},
    {"stage": "analyse", "object": "subspace bank (pass A): U_actw + U_plain, ALL layers",
     "shape": "2 x L x [F, 256] fp32", "bytes": 2 * 28 * 8960 * 256 * 4,
     "note": "0.51 GB; held for the whole stage because C3 pairs layer L with (L+14) mod L in the MATCHING weighting -- the largest persistent allocation here"},
    {"stage": "analyse", "object": "per-(layer,rank) random subspaces (C2, C2b)",
     "shape": "[F, r] / [r, D]", "bytes": 8960 * 256 * 8,
     "note": "0.018 GB, freed each rank; drawn fresh per (layer, rank)"},
    {"stage": "analyse", "object": "U_full for C1", "shape": "[F, D] fp32",
     "bytes": 8960 * 1536 * 4, "note": "0.055 GB, transient"},
    {"stage": "analyse", "object": "G (exact gate, eval)", "shape": "[T_eval, F] fp32",
     "bytes": 4096 * 8960 * 4, "note": "0.147 GB"},
    {"stage": "analyse", "object": "Ghat (per arm, per rank)", "shape": "[T_eval, F] fp32",
     "bytes": 4096 * 8960 * 4, "note": "0.147 GB, ONE live at a time (freed inside the loop)"},
    {"stage": "analyse", "object": "true_mask", "shape": "[T_eval, F] bool",
     "bytes": 4096 * 8960, "note": "0.037 GB"},
    {"stage": "analyse", "object": "theta rule: g, u, h, Y on X_theta",
     "shape": "3 x [T_theta, F] fp32 + [T_theta, D]",
     "bytes": 4 * 2048 * 8960 * 4, "note": "0.29 GB; the oracle-magnitude arm adds an argsort "
                                           "index [T_theta, F] int64 = 0.15 GB -- the largest "
                                           "single allocation in this stage"},
    {"stage": "analyse", "object": "PEAK", "bytes": 1.6e9, "note": "no donor resident"},

    {"stage": "ann", "object": "keys K + reconstruction Krec fp64", "shape": "2 x [F, D]",
     "bytes": 2 * 8960 * 1536 * 8, "note": "0.22 GB"},
    {"stage": "ann", "object": "approx_all (index's own scores)", "shape": "[T_eval, F] fp32",
     "bytes": 4096 * 8960 * 4, "note": "0.147 GB"},
    {"stage": "ann", "object": "reach / Gh per nprobe", "shape": "2 x [T_eval, F]",
     "bytes": 4096 * 8960 * 5, "note": "0.18 GB, freed each nprobe"},
    {"stage": "ann", "object": "kmeans distance matrix", "shape": "[F, nlist] fp64",
     "bytes": 8960 * 95 * 8, "note": "0.007 GB"},
    {"stage": "ann", "object": "PEAK", "bytes": 1.6e9, "note": "no donor resident"},

    {"stage": "disk", "object": "CACHE_DIR", "bytes": 1.6e9,
     "note": "H_fit + X_theta + X_eval for 28 layers, OUTSIDE the repo tree on purpose"},
]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    only = None
    if len(sys.argv) > 2:
        only = [int(x) for x in sys.argv[2].split(",")]
    if cmd == "selfcheck":
        stage_selfcheck()
    elif cmd == "capture":
        stage_capture()
    elif cmd == "analyse":
        stage_analyse(only)
    elif cmd == "ann":
        stage_ann(only)
    else:
        raise SystemExit(f"unknown stage {cmd}")
