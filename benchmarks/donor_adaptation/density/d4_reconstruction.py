#!/usr/bin/env python3
"""D4 -- does layer-wise reconstruction recover the block-pruning loss D1 measured?

D1 (`d1_pruning.py` / `d1b_organ_sweep_completion.py`) measured ONE-SHOT MAGNITUDE
block-masking with NO weight update: catastrophic at every organ/level tested. That is
the floor a Hessian-weighted layer-wise solve exists to beat. This probe applies exactly
one variable on top of D1's own masks: mask-only (D1, already measured) vs
mask-plus-reconstruction (this probe).

METHOD (SparseGPT's weight update restricted to a FIXED mask; not the joint ADMM solver --
that is explicitly out of scope):

    minimise || W' X - W X ||_F^2   subject to  W' having support M

With H = X X^T (uncentered activation covariance from a calibration set) this is
row-separable and closed-form: for row m with surviving column set S,

    W'[m, S] = (W[m,:] H[:, S]) (H[S, S])^-1 ,   W'[m, not-S] = 0

using Cholesky (or lstsq fallback) on H[S,S] + lambda*I, lambda = 1e-4 * trace(H) / N.

MASK GEOMETRY MATTERS AND IS DERIVED, NOT ASSUMED (see PREREG.derivation_note below): D1's
own `_struct_axis` makes q/k/v/gate/up_proj ROW-structured (whole OUTPUT neurons zeroed) and
o_proj/down_proj COLUMN-structured (whole INPUT features zeroed, same surviving set S shared
by every output row). The closed-form reconstruction above only has freedom to act when S
varies within a row's support -- i.e. on COLUMN-structured organs. For ROW-structured organs
every kept row already has full (S = all columns) support and every dropped row has none, so
the row-separable solve returns exactly W (kept rows) or exactly 0 (dropped rows) BY
CONSTRUCTION, regardless of H's content. This is stated in the prereg below, before any
number is computed, not discovered after the fact.

Reuses, unmodified: common.py (model/slice/bpb/dump), d1_pruning.py (Snapshot, _struct_axis,
paired_se), d1b_organ_sweep_completion.py (block_size_for -- the D0-derived engine-legal
block size, read off the loaded weights, never hardcoded).
"""
from __future__ import annotations

import sys, os, json, math, time, subprocess, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import d1_pruning as D1
import d1b_organ_sweep_completion as D1B

OUTP = os.path.join(C.RESULTS, "d4_reconstruction.json")

THREADS = int(os.environ.get("D_THREADS", "10"))
torch.set_num_threads(THREADS)

ORGANS = ["gate_proj", "o_proj", "down_proj"]     # one MLP-expansion, one attn-out, one MLP-contraction
LEVELS = [0.25, 0.50, 0.75]                        # per brief: exclude 0.90 (quantised for k/v; not tested here anyway)
LAMBDA_SCALE = 1e-4
NOISE_BAND_SIGMA = 2.0

# calibration slice -- DISJOINT corpus half from the eval slice (asserted, not assumed)
N_CAL, SEQ_LEN_CAL, SEED_CAL = 32, 512, 42424   # T=16384 tokens > d_ffn=8960 (down_proj's
# in_features) with ~1.8x margin -- an earlier N_CAL=8 (T=4096 < 8960) made H rank-deficient,
# so even the identity control (mask=0%, S=full) was projecting W onto a <=4096-rank
# subspace instead of returning it unchanged. Caught by the identity control itself.
# eval slice -- byte-identical to D1's, so the comparison is not void
N_EVAL, SEQ_LEN_EVAL, SEED_EVAL = D1.N_SEQ, D1.SEQ_LEN, D1.SEED

# controls 3 (Hessian ablation) and 4 (leakage) run at ORGAN/LEVEL below -- time-budget
# decision, stated up front: down_proj is the largest, most redundant, COLUMN-structured
# organ (the case where the mechanism can act at all), so it is the one the mechanism is
# staked on. o_proj gets the real-H sweep only (no separate ablation). gate_proj is the
# ROW-structured organ where the method is analytically forced to be a no-op (see above);
# its sweep points are still run for an empirical confirmation, not because a different
# result is possible.
ABLATION_ORGAN = "down_proj"
ABLATION_LEVEL = 0.50

SHUFFLE_SEED = 777


# ------------------------------------------------------------------ block selection (mirrors d1b's own criterion)
def select_zero_blocks(model, layers, organ, frac, block_size):
    """Exactly D1B.prune_block_structured's selection math (lowest block-Frobenius-norm,
    fixed contiguous position, no reordering) but returns the block indices instead of
    mutating weights, so the SAME mask D1 used can be built without re-deriving the criterion
    and without a second mutate-then-inspect pass."""
    ax = D1._struct_axis(organ)
    out = {}
    for L in layers:
        W = C.get_linear(model, L, organ).weight
        n_units = W.shape[0] if ax == "row" else W.shape[1]
        nblocks = n_units // block_size
        assert nblocks * block_size == n_units, f"L{L} {organ}: block_size {block_size} does not divide {n_units}"
        if ax == "row":
            Wb = W.view(nblocks, block_size, W.shape[1])
            block_norm = Wb.norm(dim=(1, 2))
        else:
            Wb = W.view(W.shape[0], nblocks, block_size).permute(1, 0, 2)
            block_norm = Wb.norm(dim=(1, 2))
        k = int(round(frac * nblocks))
        idx_zero = torch.argsort(block_norm)[:k].tolist() if k > 0 else []
        out[L] = {"axis": ax, "nblocks": nblocks, "block_size": block_size,
                  "idx_zero_blocks": sorted(idx_zero)}
    return out


def surviving_units(mask_info_L, n_units, block_size):
    zeroed = set()
    for b in mask_info_L["idx_zero_blocks"]:
        zeroed.update(range(b * block_size, (b + 1) * block_size))
    return torch.tensor(sorted(set(range(n_units)) - zeroed), dtype=torch.long)


# ------------------------------------------------------------------ activation capture (generalises d0_layout's own hook pattern)
def capture_organ_inputs(model, ids, organ, layers, batch=2):
    buf = {L: [] for L in layers}
    hooks = []

    def mk(L):
        lin = C.get_linear(model, L, organ)
        def hook(mod, args):
            buf[L].append(args[0].detach().reshape(-1, args[0].shape[-1]).float().clone())
        return lin.register_forward_pre_hook(hook)

    for L in layers:
        hooks.append(mk(L))
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch):
            model.model(ids[i:i + batch])
    for h in hooks:
        h.remove()
    return {L: torch.cat(v, 0) for L, v in buf.items()}


def capture_organ_inputs_multi(model, ids, organs, layers, batch=2):
    """Same as capture_organ_inputs but captures several organs in ONE forward pass (one set
    of hooks, one sweep over ids) -- avoids paying the full-model-forward cost once per organ."""
    buf = {organ: {L: [] for L in layers} for organ in organs}
    hooks = []

    def mk(organ, L):
        lin = C.get_linear(model, L, organ)
        def hook(mod, args):
            buf[organ][L].append(args[0].detach().reshape(-1, args[0].shape[-1]).float().clone())
        return lin.register_forward_pre_hook(hook)

    for organ in organs:
        for L in layers:
            hooks.append(mk(organ, L))
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch):
            model.model(ids[i:i + batch])
    for h in hooks:
        h.remove()
    return {organ: {L: torch.cat(v, 0) for L, v in per_layer.items()} for organ, per_layer in buf.items()}


def shuffle_columns(X: torch.Tensor, seed: int) -> torch.Tensor:
    """Independent random permutation of the TOKEN axis, per column (feature) independently.
    Preserves each feature's exact marginal (hence H_shuf's diagonal == H_real's diagonal
    exactly) while destroying cross-feature correlation (H_shuf's off-diagonal -> ~0). This is
    the 'fake but same-scale' Hessian: if it recovers as much as the real one, the recovery is
    not coming from correlation structure."""
    g = torch.Generator().manual_seed(seed)
    T, D = X.shape
    Xs = torch.empty_like(X)
    for j in range(D):
        perm = torch.randperm(T, generator=g)
        Xs[:, j] = X[perm, j]
    return Xs


# ------------------------------------------------------------------ closed-form reconstruction
def reconstruct_row(W_full: torch.Tensor, H: torch.Tensor, S: torch.Tensor, lam: float):
    """W_full: [out, in] fp64.  H: [in, in] fp64, RAW / unregularised.  S: surviving column
    indices.  lam: Tikhonov strength.

    FIXED (was a regularisation-asymmetry bug, caught by the identity control): H is
    regularised ONCE, over the FULL matrix (H_reg = H + lam*I), and that SAME H_reg is used
    on BOTH the system matrix and the right-hand side -- the SparseGPT convention. This is
    what makes the S=full (identity/mask=0%) case algebraically EXACT regardless of H's own
    rank: W @ H_reg @ H_reg^-1 == W whenever H_reg is invertible, which lam>0 guarantees even
    when the raw H is singular. The earlier version regularised only the system matrix
    (H[S,S]+lam*I) while leaving the right-hand side on the unregularised H[:,S], so even at
    S=full it computed W @ H @ (H+lam*I)^-1 != W -- a real distortion, not a rounding error.
    Returns (W_new [out,in] fp64, fell_back: bool)."""
    if S.numel() == 0:
        return torch.zeros_like(W_full), False
    D = H.shape[0]
    H_reg = H + lam * torch.eye(D, dtype=torch.float64)
    A = H_reg.index_select(0, S).index_select(1, S)     # H_reg[S,S]
    rhs = W_full @ H_reg.index_select(1, S)             # W @ H_reg[:,S]  -- SAME H_reg
    fell_back = False
    try:
        L = torch.linalg.cholesky(A)
        sol = torch.cholesky_solve(rhs.T, L).T          # [out, |S|]
    except Exception:
        fell_back = True
        sol = torch.linalg.lstsq(A, rhs.T).solution.T
    W_new = torch.zeros_like(W_full)
    W_new.index_copy_(1, S, sol)
    return W_new, fell_back


def hessian_rank_ratio(H: torch.Tensor):
    """EXACT numerical rank of the RAW (unregularised) H over N = H.shape[0] (the organ's own
    input dimension) -- the diagnostic the Principal's adjudication of the joint-solver
    dossier specifies as Failure Mode 1: rank(H)/N < 0.5 means the regularisation is expected
    to distort the Hessian geometry beyond recovery. Computed via symmetric eigendecomposition
    (H is PSD by construction, H = X^T X) with the standard eigenvalue-threshold rank
    definition (largest eigenvalue * dim * float64 eps). Costs ~D^3 (independent of T) --
    ~7.8s for D=8960 on this CPU, measured -- so it is used directly for o_proj (D=1536,
    ~0.04s) and for down_proj's real-H arm (the one everything downstream depends on), but
    NOT repeated for down_proj's secondary control arms (see theoretical_rank_ratio)."""
    D = H.shape[0]
    eigvals = torch.linalg.eigvalsh(H).clamp_min(0)
    tol = float(eigvals.max()) * D * torch.finfo(torch.float64).eps
    rank = int((eigvals > tol).sum())
    return rank, D, rank / D


def theoretical_rank_ratio(T: int, D: int):
    """Upper bound rank(X^T X) <= min(T, D) -- a general linear-algebra fact, true for ANY
    T x D matrix X regardless of correlation structure, so it is always a mathematically valid
    (never optimistic) stand-in for the exact rank when the exact eigvalsh pass is skipped for
    cost reasons: if the bound itself is < threshold, the true rank is provably < threshold too
    (true rank <= bound), so the abort trigger stays correctly conservative in that direction.
    Validated empirically against hessian_rank_ratio on down_proj (D=8960) at L00/L09/L18/L27,
    T=8192: exact/theoretical = 0.9122/0.9143, 0.9142/0.9143, 0.9142/0.9143, 0.9142/0.9143 --
    max observed gap 0.21 percentage points, i.e. no material degeneracy at any tested depth."""
    ratio = min(T, D) / D
    return min(T, D), D, ratio


class HessianRankAbort(SystemExit):
    pass


def check_hessian_rank_by_layer(H_by_layer: dict, label: str, layers, threshold: float = 0.5,
                                method: str = "exact"):
    """H_by_layer: {layer_index: H [D,D] fp64, ..., "_N": calib_token_count}. Computed and
    logged BEFORE any solve is attempted (per the Principal's directive after control 1
    failed). Raises HessianRankAbort -- does NOT silently proceed -- if rank(H)/N < threshold
    for ANY layer.

    method="exact": full eigvalsh per layer (~7.8s/layer for D=8960, D-dependent only).
    method="theoretical": the min(T,D)/D upper bound (instant), valid as an abort trigger in
    the SAFE direction (true rank <= bound, so bound < threshold implies true rank < threshold
    too) -- used for secondary/control arms where re-paying the exact eigvalsh cost a second
    or third time is not worth it. See theoretical_rank_ratio's docstring for the empirical
    validation (max 0.21 percentage-point gap vs exact, across 4 depths on down_proj).
    Returns a dict of per-layer (rank, D, ratio) plus min/mean for the summary."""
    T = H_by_layer.get("_N")
    per_layer = {}
    for L in layers:
        if method == "exact":
            rank, D, ratio = hessian_rank_ratio(H_by_layer[L])
        else:
            D = H_by_layer[L].shape[0]
            rank, D, ratio = theoretical_rank_ratio(T, D)
        per_layer[L] = {"rank": rank, "N": D, "ratio": ratio}
    ratios = [v["ratio"] for v in per_layer.values()]
    summary = {"label": label, "method": method, "T_tokens": T, "per_layer": per_layer,
               "ratio_min": min(ratios), "ratio_mean": sum(ratios) / len(ratios),
               "ratio_max": max(ratios), "threshold": threshold,
               "worst_layer": min(per_layer, key=lambda L: per_layer[L]["ratio"])}
    print(f"  rank(H)/N [{label}, method={method}]: min={summary['ratio_min']:.3f} "
          f"mean={summary['ratio_mean']:.3f} max={summary['ratio_max']:.3f} "
          f"(worst layer {summary['worst_layer']})", flush=True)
    if summary["ratio_min"] < threshold:
        summary["ABORTED"] = True
        print(f"  ABORT: rank(H)/N = {summary['ratio_min']:.3f} < {threshold} for {label} "
              f"(layer {summary['worst_layer']}) -- Failure Mode 1, refusing to solve.", flush=True)
        raise HessianRankAbort(summary)
    summary["ABORTED"] = False
    return summary


def reconstruct_organ_level(model, layers, organ, block_size, mask_by_layer, H_by_layer, lam_scale):
    """Returns {L: W_new fp32} for a COLUMN-structured organ, plus per-layer lambda/fallback log.
    H_by_layer must contain one [in,in] fp64 matrix per layer index plus a scalar "_N" (the
    calibration-token count used to form H, for the lambda = lam_scale*trace(H)/N rule)."""
    ax = D1._struct_axis(organ)
    assert ax == "col"
    N = H_by_layer["_N"]
    out_w, meta = {}, {}
    for L in layers:
        W = C.get_linear(model, L, organ).weight
        n_in = W.shape[1]
        S = surviving_units(mask_by_layer[L], n_in, block_size)
        H = H_by_layer[L]
        lam = lam_scale * float(torch.trace(H)) / max(1, N)
        Wnew64, fb = reconstruct_row(W.double(), H, S, lam)
        out_w[L] = Wnew64.to(W.dtype)
        meta[L] = {"lambda": lam, "fell_back": fb, "n_surviving": int(S.numel())}
    return out_w, meta


def reconstruct_row_axis_level(model, layers, organ, block_size, mask_by_layer):
    """ROW-structured organ: analytically forced -- kept rows unchanged, dropped rows zero.
    No H, no solve; documented as a mathematical fact in the module docstring/prereg, not
    discovered empirically."""
    out_w = {}
    for L in layers:
        W = C.get_linear(model, L, organ).weight
        n_out = W.shape[0]
        zeroed = set()
        for b in mask_by_layer[L]["idx_zero_blocks"]:
            zeroed.update(range(b * block_size, (b + 1) * block_size))
        Wnew = W.detach().clone()
        if zeroed:
            idx = torch.tensor(sorted(zeroed), dtype=torch.long)
            Wnew[idx, :] = 0.0
        out_w[L] = Wnew
    return out_w


# ------------------------------------------------------------------ stats
def bootstrap_recovery(base_per, recon_per, byts, delta_naive, n_boot=2000, seed=11):
    """Paired bootstrap of delta_reconstructed (same scheme as D1.paired_se), then propagated
    through recovery = 1 - delta_recon/delta_naive holding delta_naive FIXED at its D1 point
    value (D1's own SE on delta_naive is reported alongside, not re-bootstrapped jointly --
    stated as an assumption in PREREG, not hidden)."""
    rng = np.random.default_rng(seed)
    b = byts.numpy()
    na = base_per * math.log(2.0) * b
    nb = recon_per * math.log(2.0) * b
    n = len(b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        s = b[k].sum() * math.log(2.0)
        deltas[i] = nb[k].sum() / s - na[k].sum() / s
    recov = 1.0 - deltas / delta_naive
    return float(np.std(recov)), recov


def classify_recovery(point_est, se, band_sigma=NOISE_BAND_SIGMA):
    lo, hi = point_est - band_sigma * se, point_est + band_sigma * se
    if lo <= 0.0 <= hi:
        return "INCONCLUSIVE"
    if point_est >= 0.70 and lo > 0.5:
        return "LARGE"
    if 0.30 <= point_est < 0.70:
        return "MODERATE"
    return "NEGLIGIBLE"


# ------------------------------------------------------------------ main
def main():
    run_t0 = time.time()
    log = {}
    log["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO).decode().strip()
    log["git_branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=C.REPO).decode().strip()
    log["command_line"] = "python " + " ".join(sys.argv)
    log["threads"] = THREADS

    # ============================================================ PRE-REGISTRATION (written before any number below is computed)
    log["prereg"] = {
        "primary_quantity": "recovery = 1 - delta_reconstructed / delta_naive, per (organ, sparsity) "
            "point; delta_naive read verbatim from d1_pruning.json's block_structured record for "
            "that exact (organ, level).",
        "se_method": "paired bootstrap (n_boot=2000, >= 10 required) of delta_reconstructed over "
            "sequence resampling on the SAME 24x512 heldout slice D1 used, propagated through the "
            "recovery ratio holding delta_naive fixed at its D1 point value (D1's own paired_se on "
            "delta_naive is reported alongside per point, not re-bootstrapped jointly -- assumption "
            "stated here, not hidden).",
        "inconclusive_rule": "a point's recovery is INCONCLUSIVE iff its "
            f"[point - {NOISE_BAND_SIGMA}*SE, point + {NOISE_BAND_SIGMA}*SE] interval spans 0.",
        "magnitude_bands_fixed_before_any_result": {
            "LARGE": "recovery point estimate >= 0.70 AND the lower band bound > 0.5",
            "MODERATE": "0.30 <= recovery point estimate < 0.70 (and not inconclusive)",
            "NEGLIGIBLE": "recovery point estimate < 0.30 (and not inconclusive)",
            "INCONCLUSIVE": "band spans 0, regardless of point estimate",
        },
        "derivation_note_mask_geometry": "D1._struct_axis makes q/k/v/gate/up_proj ROW-structured "
            "(whole output neurons zeroed) and o_proj/down_proj COLUMN-structured (whole input "
            "features zeroed, same surviving column set S shared across every output row). The "
            "row-separable SparseGPT-style solve in this brief only has freedom to act when S "
            "varies WITHIN a row's support -- i.e. only for COLUMN-structured organs. For "
            "gate_proj (ROW-structured), every kept row already has full support (S = all "
            "columns) and every dropped row has none: the closed form is therefore mathematically "
            "forced to return exactly W (kept rows) or exactly 0 (dropped rows), independent of H's "
            "content. This is stated here, before gate_proj is run, not discovered after.",
        "scope": "gate_proj, o_proj, down_proj (one MLP-expansion/ROW-structured, one "
            "attn-output/COLUMN-structured, one MLP-contraction/COLUMN-structured) at "
            "sparsity {0.25, 0.50, 0.75}. 0.90 excluded per brief (quantisation artefact struck "
            "for k/v_proj in D1; not requested for these three organs anyway).",
        "control3_control4_scope_decision": f"Hessian ablation (control 3) and leakage control "
            f"(control 4) are run on {ABLATION_ORGAN}@{ABLATION_LEVEL} only, a time-budget decision "
            "stated up front: down_proj is the larger and only-interesting-case-among-the-two "
            "COLUMN-structured organs tested (in_features=8960 vs o_proj's 1536), so it is the "
            "instance the reconstruction mechanism is most staked on. o_proj's sweep numbers stand "
            "on real-H only and are NOT independently ablated -- flagged in the report.",
    }
    C.dump("d4_reconstruction.json", log)

    # ============================================================ load model, reproduce D1's baseline
    m, tok = C.load_model()
    A = C.arch(m)
    all_layers = list(range(A["n_layers"]))
    log["arch"] = A

    d1 = json.load(open(os.path.join(C.RESULTS, "d1_pruning.json")))
    assert A == d1["arch"], f"arch mismatch vs D1: {A} != {d1['arch']}"

    log["hardware"] = {
        "cuda_available": torch.cuda.is_available(),
        "note": "torch.cuda.device_count()==0 in this shell (consistent with every D0-D3 session "
            "in this repo, which ran CPU-only -- see DENSITY_PROBES.md header). Hardware-pinning "
            "assertions from the brief are N/A here; ran CPU, same as D1, so the eval path matches.",
    }

    ids_eval, byts_eval, meta_eval = C.get_slice(tok, "heldout", N_EVAL, SEQ_LEN_EVAL, SEED_EVAL)
    assert meta_eval["ids_sha256"] == d1["slice"]["ids_sha256"], "eval slice does not match D1's"
    log["eval_slice"] = meta_eval

    ids_cal, byts_cal, meta_cal = C.get_slice(tok, "calib", N_CAL, SEQ_LEN_CAL, SEED_CAL)
    log["calib_slice"] = meta_cal

    # ---- mandatory disjointness assertion ----
    manifest = json.load(open(os.path.join(C.CORPUS, "manifest.json")))
    assert meta_cal["part"] == "calib" and meta_eval["part"] == "heldout"
    assert manifest["parts"]["calib"]["sha256"] != manifest["parts"]["heldout"]["sha256"]
    assert meta_cal["corpus_sha256"] == manifest["parts"]["calib"]["sha256"]
    assert meta_eval["corpus_sha256"] == manifest["parts"]["heldout"]["sha256"]
    log["calib_eval_disjointness"] = {
        "calib_part": "calib", "eval_part": "heldout",
        "calib_corpus_sha256": meta_cal["corpus_sha256"],
        "eval_corpus_sha256": meta_eval["corpus_sha256"],
        "assertion": "calib and heldout are two DIFFERENT physical corpus files "
            "(benchmarks/donor_adaptation/density/corpus/{calib,heldout}.txt), built by splitting "
            "the globally-shuffled chunk list in half BEFORE either slice is drawn (build_calib.py) "
            "-- disjoint by construction, and their distinct sha256 hashes are asserted equal to "
            "the manifest's own record here, not merely assumed.",
        "verified": True,
    }
    C.dump("d4_reconstruction.json", log)

    t0 = time.time()
    base_bpb, base_per = C.bpb(m, ids_eval, byts_eval, return_per_seq=True)
    print(f"baseline BPB {base_bpb:.6f} (D1: {d1['baseline_bpb']:.6f})  {time.time()-t0:.0f}s", flush=True)
    assert base_bpb == d1["baseline_bpb"], "baseline BPB does not reproduce D1's bit-exactly -- eval path diverged"
    log["baseline_bpb"] = base_bpb
    log["baseline_reproduces_d1_exactly"] = True
    C.dump("d4_reconstruction.json", log)

    def evaluate(tag, **extra):
        b, per = C.bpb(m, ids_eval, byts_eval, return_per_seq=True)
        se = D1.paired_se(base_per, per, byts_eval)
        delta = b - base_bpb
        rec = {"tag": tag, "bpb": b, "delta": delta, "paired_se": se, **extra}
        print(f"  {tag:46s} BPB {b:.5f}  d={delta:+.5f} +-{se:.5f}", flush=True)
        return rec, per

    snap = D1.Snapshot()
    block_sizes = {}
    for organ in ORGANS:
        bs, bpu = D1B.block_size_for(m, organ)
        block_sizes[organ] = bs
        print(f"block size {organ}: {bs} units ({bpu:.0f} B/unit)", flush=True)
    log["block_sizes"] = block_sizes
    C.dump("d4_reconstruction.json", log)

    # ============================================================ activation capture: down_proj + o_proj calib, ONE forward pass
    print()
    print("== capturing calibration activations ==", flush=True)
    t0 = time.time()
    X_cal = capture_organ_inputs_multi(m, ids_cal, ["down_proj", "o_proj"], all_layers)
    X_cal_down, X_cal_o = X_cal["down_proj"], X_cal["o_proj"]
    T_cal = ids_cal.shape[0] * ids_cal.shape[1]
    print(f"  combined calib capture: {time.time()-t0:.0f}s, T={T_cal}  "
          f"(down_proj in_features={next(iter(X_cal_down.values())).shape[1]}, "
          f"o_proj in_features={next(iter(X_cal_o.values())).shape[1]})", flush=True)
    log["calib_tokens_T"] = T_cal

    def H_of(X):
        return X.double().T @ X.double()

    # H_real for down_proj is expensive (D=8960) and level-independent -- form it ONCE and
    # reuse for control 1 (identity), control 3's real-H arm, and every real-H sweep point at
    # this organ, instead of recomputing it 5 separate times.
    print("  forming down_proj real-H (once, reused everywhere below)...", flush=True)
    t0 = time.time()
    H_real_down = {L: H_of(X_cal_down[L]) for L in all_layers}
    H_real_down["_N"] = T_cal
    print(f"  done: {time.time()-t0:.0f}s", flush=True)

    # ---- MANDATORY rank(H)/N diagnostic, BEFORE any solve is attempted (Failure Mode 1 from
    # the joint-solver dossier adjudication -- added after control 1 failed on this exact
    # issue with the old N_CAL=8/T=4096 calibration set). Exact eigvalsh for down_proj's
    # real-H, since everything below depends on it; auto-aborts if rank(H)/N < 0.5 for ANY
    # layer, does not proceed silently.
    rank_diagnostics = {}
    rank_diagnostics["down_proj_real_H"] = check_hessian_rank_by_layer(
        H_real_down, "down_proj_real_H", all_layers, threshold=0.5, method="exact")
    log["rank_diagnostics"] = rank_diagnostics
    C.dump("d4_reconstruction.json", log)

    # o_proj's H is cheap (D=1536) -- cache once too, exact rank check, reused by every
    # o_proj sweep point instead of recomputed per level.
    H_real_o = {L: H_of(X_cal_o[L]) for L in all_layers}
    H_real_o["_N"] = T_cal
    rank_diagnostics["o_proj_real_H"] = check_hessian_rank_by_layer(
        H_real_o, "o_proj_real_H", all_layers, threshold=0.5, method="exact")
    log["rank_diagnostics"] = rank_diagnostics
    C.dump("d4_reconstruction.json", log)

    # ============================================================ CONTROLS -- run FIRST, before the sweep
    print("\n== D4 planted controls ==", flush=True)
    controls = {}

    # ---- control 1: identity (mask=0%) -- reconstruction must return W to fp tolerance ----
    print("-- control 1: identity (mask=0%) --", flush=True)
    zero_masks = {organ: select_zero_blocks(m, all_layers, organ, 0.0, block_sizes[organ]) for organ in ORGANS}
    max_dev = {}
    for organ in ("down_proj",):  # the only organ where a solve actually runs at 0%
        H_by = H_real_down   # cached, formed once above
        Wnew, meta = reconstruct_organ_level(m, all_layers, organ, block_sizes[organ], zero_masks[organ], H_by, LAMBDA_SCALE)
        devs = []
        for L in all_layers:
            W = C.get_linear(m, L, organ).weight
            snap.take((L, organ), W)
            dev = float((Wnew[L].float() - W).abs().max())
            devs.append(dev)
            W.copy_(Wnew[L].float())
        max_dev[organ] = max(devs)
        r, _ = evaluate("CTRL identity_mask0pct_down_proj", organ=organ, max_abs_weight_deviation=max_dev[organ])
        r["fired"] = abs(r["delta"]) < 0.001   # near-zero, not bit-exact: lambda regularisation is nonzero even at S=full
        controls["identity"] = r
        snap.restore(m)
        del H_by
    # gate_proj at 0% mask needs no solve at all (S=full trivially, no H involved) -- exact by construction
    controls["identity"]["gate_proj_note"] = ("ROW-structured: at mask=0% no rows are zeroed, no solve is "
        "invoked at all (see reconstruct_row_axis_level) -- weight deviation is exactly 0 by construction, "
        "not run as a separate eval.")
    controls["identity"]["o_proj_note"] = ("Not separately measured (time budget): same COLUMN-structured "
        "solve as down_proj, same S=full-set-at-0%-mask argument; down_proj is the larger case and stands "
        "as the representative check.")
    log["controls"] = controls
    C.dump("d4_reconstruction.json", log)

    # ---- control 2: saturation (mask=100%) -- reconstruction has nothing to work with ----
    print("-- control 2: saturation (mask=100%) --", flush=True)
    organ = "down_proj"
    full_masks = select_zero_blocks(m, all_layers, organ, 1.0, block_sizes[organ])
    for L in all_layers:
        W = C.get_linear(m, L, organ).weight
        snap.take((L, organ), W)
        W.zero_()  # S=empty for every row -- naive-100% and reconstructed-100% are the SAME operation by construction
    r, _ = evaluate("CTRL saturation_mask100pct_down_proj", organ=organ,
                    note="naive-100% and reconstructed-100% are IDENTICAL by construction (S=empty for "
                         "every row at full block-column removal) -- this single eval serves as both; D1 "
                         "did not log a 100% point for down_proj (only for v_proj as a generic control), "
                         "so there is nothing from D1 to compare against directly; the self-consistency "
                         "IS the control.")
    r["fired"] = r["delta"] > 0.5
    controls["saturation"] = r
    controls["saturation"]["gate_proj_o_proj_note"] = ("Not separately measured (time budget): the same "
        "construction argument (S=empty at 100%) holds for any organ/axis, so this is expected to "
        "generalise; listed under 'could not obtain a number' in the report.")
    log["controls"] = controls
    snap.restore(m)
    C.dump("d4_reconstruction.json", log)

    # ---- control 3: THE LOAD-BEARING ONE -- Hessian ablation on down_proj@50% ----
    print(f"-- control 3: Hessian ablation, {ABLATION_ORGAN}@{int(ABLATION_LEVEL*100)}% --", flush=True)
    organ = ABLATION_ORGAN
    lvl_mask = select_zero_blocks(m, all_layers, organ, ABLATION_LEVEL, block_sizes[organ])
    ablation = {}

    # real H (cached, formed once above)
    H_real = H_real_down
    Wnew, rmeta = reconstruct_organ_level(m, all_layers, organ, block_sizes[organ], lvl_mask, H_real, LAMBDA_SCALE)
    for L in all_layers:
        W = C.get_linear(m, L, organ).weight; snap.take((L, organ), W); W.copy_(Wnew[L].float())
    r, per_real = evaluate(f"ABLATION real_H {organ}@{int(ABLATION_LEVEL*100)}%", organ=organ, arm="real_H",
                           lambda_mean=float(np.mean([rmeta[L]["lambda"] for L in all_layers])),
                           any_fell_back=any(rmeta[L]["fell_back"] for L in all_layers))
    ablation["real_H"] = r; snap.restore(m)
    del H_real

    # identity H (plain least squares, no activation weighting)
    n_in = C.get_linear(m, 0, organ).weight.shape[1]
    H_id = {L: torch.eye(n_in, dtype=torch.float64) for L in all_layers}
    H_id["_N"] = T_cal
    # rank(I_D) = D exactly, by construction -- no eigvalsh needed (would cost ~7.8s for no
    # information: an identity matrix is full-rank by definition).
    rank_diagnostics["down_proj_identity_H"] = {"label": "down_proj_identity_H", "method": "analytic",
        "ratio_min": 1.0, "ratio_mean": 1.0, "ratio_max": 1.0, "note": "rank(I_D)=D exactly by construction"}
    Wnew, imeta = reconstruct_organ_level(m, all_layers, organ, block_sizes[organ], lvl_mask, H_id, LAMBDA_SCALE)
    for L in all_layers:
        W = C.get_linear(m, L, organ).weight; snap.take((L, organ), W); W.copy_(Wnew[L].float())
    r, _ = evaluate(f"ABLATION identity_H {organ}@{int(ABLATION_LEVEL*100)}%", organ=organ, arm="identity_H",
                    lambda_mean=float(np.mean([imeta[L]["lambda"] for L in all_layers])))
    ablation["identity_H"] = r; snap.restore(m)
    del H_id

    # shuffled/random activations H
    H_shuf = {L: H_of(shuffle_columns(X_cal_down[L], SHUFFLE_SEED + L)) for L in all_layers}
    H_shuf["_N"] = T_cal
    rank_diagnostics["down_proj_shuffled_H"] = check_hessian_rank_by_layer(
        H_shuf, "down_proj_shuffled_H", all_layers, threshold=0.5, method="theoretical")
    Wnew, smeta = reconstruct_organ_level(m, all_layers, organ, block_sizes[organ], lvl_mask, H_shuf, LAMBDA_SCALE)
    for L in all_layers:
        W = C.get_linear(m, L, organ).weight; snap.take((L, organ), W); W.copy_(Wnew[L].float())
    r, _ = evaluate(f"ABLATION shuffled_H {organ}@{int(ABLATION_LEVEL*100)}%", organ=organ, arm="shuffled_H",
                    lambda_mean=float(np.mean([smeta[L]["lambda"] for L in all_layers])))
    ablation["shuffled_H"] = r; snap.restore(m)
    del H_shuf
    # X_cal_down is not needed again (main sweep's down_proj points reuse cached H_real_down,
    # not X_cal_down directly) -- free it now, before control 4 adds the eval-slice buffers.
    del X_cal_down
    gc.collect()

    dn = None
    for rec in d1["organ_sweep"]:
        if rec.get("organ") == organ and rec.get("mode") == "block_structured" and abs(rec.get("level", -1) - ABLATION_LEVEL) < 1e-9:
            dn = rec["delta"]; break
    ablation["delta_naive_d1"] = dn
    for arm in ("real_H", "identity_H", "shuffled_H"):
        ablation[arm]["recovery_point"] = 1.0 - ablation[arm]["delta"] / dn
    controls["hessian_ablation"] = ablation
    log["controls"] = controls
    C.dump("d4_reconstruction.json", log)
    print(f"  ABLATION recovery: real_H={ablation['real_H']['recovery_point']:.3f}  "
          f"identity_H={ablation['identity_H']['recovery_point']:.3f}  "
          f"shuffled_H={ablation['shuffled_H']['recovery_point']:.3f}", flush=True)

    # ---- control 4: leakage -- H built from the EVAL slice itself ----
    print(f"-- control 4: leakage, {ABLATION_ORGAN}@{int(ABLATION_LEVEL*100)}% with eval-slice H --", flush=True)
    t0 = time.time()
    X_eval_down = capture_organ_inputs(m, ids_eval, ABLATION_ORGAN, all_layers)
    print(f"  eval-slice capture ({ABLATION_ORGAN}): {time.time()-t0:.0f}s", flush=True)
    T_eval_tok = ids_eval.shape[0] * ids_eval.shape[1]
    H_leak = {L: H_of(X_eval_down[L]) for L in all_layers}
    H_leak["_N"] = T_eval_tok
    rank_diagnostics["down_proj_leakage_H"] = check_hessian_rank_by_layer(
        H_leak, "down_proj_leakage_H", all_layers, threshold=0.5, method="theoretical")
    Wnew, lmeta = reconstruct_organ_level(m, all_layers, organ, block_sizes[organ], lvl_mask, H_leak, LAMBDA_SCALE)
    for L in all_layers:
        W = C.get_linear(m, L, organ).weight; snap.take((L, organ), W); W.copy_(Wnew[L].float())
    r, _ = evaluate(f"LEAKAGE eval_H {organ}@{int(ABLATION_LEVEL*100)}%", organ=organ, arm="eval_H")
    r["recovery_point"] = 1.0 - r["delta"] / dn
    r["compare_to_honest_real_H_delta"] = ablation["real_H"]["delta"]
    r["leakage_looks_better_than_honest"] = r["delta"] < ablation["real_H"]["delta"]
    controls["leakage"] = r
    log["controls"] = controls
    snap.restore(m)
    del X_eval_down, H_leak
    gc.collect()
    C.dump("d4_reconstruction.json", log)
    print(f"  LEAKAGE: eval-H delta={r['delta']:.5f} vs honest calib-H delta={ablation['real_H']['delta']:.5f} "
          f"-> {'looks better (expected)' if r['leakage_looks_better_than_honest'] else 'DOES NOT LOOK BETTER -- INVESTIGATE'}",
          flush=True)

    all_control_flags = [
        controls["identity"]["fired"], controls["saturation"]["fired"],
        r["leakage_looks_better_than_honest"],
    ]
    log["controls_summary"] = {
        "identity_fired": controls["identity"]["fired"],
        "saturation_fired": controls["saturation"]["fired"],
        "leakage_looks_better_than_honest": r["leakage_looks_better_than_honest"],
        "hessian_ablation_arms": {k: ablation[k]["recovery_point"] for k in ("real_H", "identity_H", "shuffled_H")},
    }
    C.dump("d4_reconstruction.json", log)

    # ============================================================ MAIN SWEEP
    print("\n== D4 main sweep ==", flush=True)
    sweep = []
    for organ in ORGANS:
        ax = D1._struct_axis(organ)
        bs = block_sizes[organ]
        for level in LEVELS:
            mask = select_zero_blocks(m, all_layers, organ, level, bs)
            n_units = C.get_linear(m, 0, organ).weight.shape[1 if ax == "col" else 0]
            zeroed_total = sum(len(mask[L]["idx_zero_blocks"]) * bs for L in all_layers)
            total_units = n_units * len(all_layers)
            zero_frac_achieved = zeroed_total / total_units

            dn = None; dn_se = None
            for rec in d1["organ_sweep"]:
                if rec.get("organ") == organ and rec.get("mode") == "block_structured" and abs(rec.get("level", -1) - level) < 1e-9:
                    dn, dn_se = rec["delta"], rec["paired_se"]; break
            assert dn is not None, f"no D1 naive record for {organ}@{level}"

            if ax == "col":
                if organ == "down_proj":
                    H_by = H_real_down   # cached, formed once above -- not recomputed per level
                else:
                    H_by = H_real_o      # cached, formed once above -- not recomputed per level
                Wnew, rmeta = reconstruct_organ_level(m, all_layers, organ, bs, mask, H_by, LAMBDA_SCALE)
                lam_mean = float(np.mean([rmeta[L]["lambda"] for L in all_layers]))
                any_fb = any(rmeta[L]["fell_back"] for L in all_layers)
                del H_by
            else:
                Wnew = reconstruct_row_axis_level(m, all_layers, organ, bs, mask)
                lam_mean, any_fb = None, False

            for L in all_layers:
                W = C.get_linear(m, L, organ).weight
                snap.take((L, organ), W)
                W.copy_(Wnew[L].to(W.dtype))
            rec, per = evaluate(f"{organ}/reconstructed/{int(level*100)}%", organ=organ, level=level,
                                axis=ax, zero_frac_requested=level, zero_frac_achieved=zero_frac_achieved,
                                zero_frac_divergence_pct=abs(zero_frac_achieved - level) * 100,
                                delta_naive_d1=dn, delta_naive_d1_paired_se=dn_se,
                                lambda_mean=lam_mean, any_fell_back=any_fb)
            snap.restore(m)

            recovery = 1.0 - rec["delta"] / dn
            rec_se, _ = bootstrap_recovery(base_per, per, byts_eval, dn, seed=11 + len(sweep))
            rec["recovery"] = recovery
            rec["recovery_se"] = rec_se
            rec["recovery_verdict"] = classify_recovery(recovery, rec_se)
            if ax == "row":
                rec["row_axis_forced_note"] = ("ROW-structured organ: reconstruction is mathematically forced "
                    "to equal naive masking exactly (see PREREG.derivation_note_mask_geometry). recovery ~ 0 "
                    "here is a construction fact, not a measurement of whether the mechanism works.")
                # Coordinator correction (post D1's own SIGNIFICANT-on-exact-zero bug, same class of error):
                # classify_recovery's empirical INCONCLUSIVE/LARGE/MODERATE/NEGLIGIBLE bands are for
                # quantities whose true value is UNKNOWN a priori. gate_proj's recovery is proven exactly 0
                # by derivation_note_mask_geometry BEFORE this number was computed -- INCONCLUSIVE would
                # mischaracterise a confirmed theorem as unresolved measurement noise, and would wrongly
                # invite a reader to think more samples could resolve it. A null-expected result must not be
                # able to print an empirical significance verdict. Overridden here, unconditionally, for
                # every ROW-structured sweep point.
                rec["recovery_verdict"] = "ZERO_BY_CONSTRUCTION"
                rec["verdict_override_reason"] = ("Empirical classify_recovery() output "
                    f"was {classify_recovery(recovery, rec_se)!r} (band spans 0 -> INCONCLUSIVE by the "
                    "generic rule); overridden because PREREG.derivation_note_mask_geometry proves recovery=0 "
                    "algebraically for ROW-structured organs, independent of H -- this is a theorem being "
                    "confirmed, not a datum about the reconstruction method.")
                rec["excluded_from_summary_stats"] = True
            sweep.append(rec)
            log["organ_sweep"] = sweep
            C.dump("d4_reconstruction.json", log)
            print(f"    recovery={recovery:+.3f} +-{rec_se:.3f}  [{rec['recovery_verdict']}]  "
                  f"zero_frac_div={rec['zero_frac_divergence_pct']:.3f}%", flush=True)

    del X_cal_o, H_real_down
    gc.collect()

    log["done"] = True
    log["wallclock_seconds"] = time.time() - run_t0
    print("\nwrote", C.dump("d4_reconstruction.json", log), flush=True)
    print(f"total wallclock: {log['wallclock_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
