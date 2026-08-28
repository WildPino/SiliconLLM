#!/usr/bin/env python3
"""R2a -- the principal-angle pre-screen.  BRIEF_R2A_PRINCIPAL_ANGLE_PRESCREEN.md.

Computes, per donor layer / per head / per candidate feature map `phi`:

    residual = || W_v^donor Z* (I - P_Z) ||_F  /  || W_v^donor Z* ||_F

with  Z  = X A_lin^T ,  Z* = X A_soft^T ,  P_Z = orthogonal projector onto rowspace(Z) in R^T.

`residual` is an UPPER BOUND on what any W_v can achieve in R2's solve; it needs no solver.

Non-negotiables carried from the brief and the Controller audit:
  * X is the POST-RMSNorm activation entering v_proj (audit finding 2), read via a
    forward-pre-hook on q_proj (same tensor, positional arg).
  * every A_lin is ROW-NORMALISED over the causal prefix (audit finding 3); Qwen2 has
    q/k/v biases, and the bias term only cancels when row sums are exactly 1.
  * the instrument REFUSES to emit a residual when T <= D (brief section 1).
  * rank(Z), rank(Z*) exact, at a stated tolerance; sensitivity to that tolerance reported.
  * every reported number is stratified by A_soft row entropy.
  * ACHIEVED parameters are printed from the constructed objects, never from a config dict.

Usage:
    python r2a_principal_angle.py selfcheck
    python r2a_principal_angle.py main
    python r2a_principal_angle.py curve
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
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY = os.path.abspath(os.path.join(HERE, "..", "density"))
sys.path.insert(0, DENSITY)
import common  # noqa: E402  (D-probe harness: pinned donor, pinned corpus, cached slices)

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

torch.set_grad_enabled(False)

# ---------------------------------------------------------------- pre-registered constants
SEQ_LEN = 1024                 # real in-distribution context; A is block-diagonal over sequences
CALIB_SEED = 20260822          # slice seed for the arm sequences
ALT_SEED = 20260823            # slice seed for the C5' unrelated sequences
FAVOR_SEED = 424242            # Performer FAVOR+ random-feature seed -- STATED, per brief section 2
FAVOR_M = 256                  # number of random features
RANK_TOL = 1e-6                # primary numerical-rank tolerance, on singular values / sigma_max
RANK_TOLS = (1e-5, 1e-6, 1e-7)  # sensitivity ladder
SHORT_PREFIX = 32              # positions t < 32 in a sequence are a separate degenerate bucket
PHIS = ("elu1", "taylor2", "favor", "favor_d14", "hedgehog0")
PHIS_EXTRA = ("elu1_noscale", "hedgehog0_noscale")   # scaling-sensitivity spot check
C5P_C4_PHI = "elu1"          # the phi used to build the C5' / C4 mixings
CONTROLS = ("C1_identity", "C6_causal_uniform", "C5p_wrong_sequence", "C4_wrong_layer")
# Builder-added floors, BEYOND the brief's table -- they separate "residual earned by
# content" from "residual explained by dimension alone" and from "residual explained by
# Z merely sharing rowspace(X)".  Additions, not alterations.
FLOORS = ("C7_random_gauss", "C8_no_mixing")


# ================================================================= donor plumbing
def load_donor():
    model, tok = common.load_model(dtype=torch.float32)
    return model, tok


def achieved_arch(model) -> dict:
    """Every field read off the LOADED OBJECT, not off config prose."""
    l0 = model.model.layers[0]
    sa = l0.self_attn
    wq, wk, wv, wo = sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj
    return {
        "repo_id": common.MODEL_ID,
        "revision_pinned": common.REVISION,
        "n_layers_achieved": len(model.model.layers),
        "D_achieved": int(wq.in_features),
        "q_out_achieved": int(wq.out_features),
        "k_out_achieved": int(wk.out_features),
        "v_out_achieved": int(wv.out_features),
        "o_in_achieved": int(wo.in_features),
        "head_dim_achieved": int(sa.head_dim),
        "n_q_heads_achieved": int(wq.out_features // sa.head_dim),
        "n_kv_heads_achieved": int(wk.out_features // sa.head_dim),
        "kv_groups_achieved": int(sa.num_key_value_groups),
        "scaling_achieved": float(sa.scaling),
        "q_bias_achieved": bool(wq.bias is not None),
        "k_bias_achieved": bool(wk.bias is not None),
        "v_bias_achieved": bool(wv.bias is not None),
        "o_bias_achieved": bool(wo.bias is not None),
        "sliding_window_achieved": sa.sliding_window,
        "layer_types_unique_achieved": sorted(set(model.config.layer_types)),
        "param_dtype_achieved": str(wq.weight.dtype),
        "rms_norm_eps_achieved": float(l0.input_layernorm.variance_epsilon),
        "rope_theta_achieved": float(model.model.rotary_emb.config.rope_theta),
        "n_params_achieved": int(sum(p.numel() for p in model.parameters())),
    }


def revision_hash_on_disk() -> dict:
    """The donor's exact revision, read off the local snapshot dir + its config sha256."""
    from huggingface_hub import snapshot_download
    d = snapshot_download(common.MODEL_ID, revision=common.REVISION, local_files_only=True)
    cfg = os.path.join(d, "config.json")
    h = hashlib.sha256(open(cfg, "rb").read()).hexdigest()
    return {"snapshot_dir": d, "snapshot_leaf": os.path.basename(d), "config_sha256": h}


def capture(model, ids: torch.Tensor, layers) -> dict:
    """Post-RMSNorm activations entering v_proj, per requested layer.

    Hooked on q_proj's *positional* input, which is the exact tensor v_proj also receives.
    Returns {layer: float32 ndarray [n_seq, L, D]}.
    """
    store = {l: [] for l in layers}
    hooks = []

    def mk(l):
        def h(mod, args):
            store[l].append(args[0].detach().to(torch.float32).numpy().copy())
        return h

    for l in layers:
        hooks.append(model.model.layers[l].self_attn.q_proj.register_forward_pre_hook(mk(l)))
    try:
        for i in range(ids.shape[0]):
            model.model(ids[i:i + 1])
    finally:
        for h in hooks:
            h.remove()
    return {l: np.concatenate(store[l], axis=0) for l in layers}


def rope_cos_sin(model, L: int):
    pos = torch.arange(L).unsqueeze(0)
    dummy = torch.zeros(1, L, model.config.hidden_size, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(dummy, pos)
    return cos[0].numpy().astype(np.float64), sin[0].numpy().astype(np.float64)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def qk_for_layer(model, layer: int, xn: np.ndarray, cos, sin):
    """q,k AFTER RoPE for one layer.  xn: [n_seq, L, D] float32.  Returns fp64
    q [n_seq, Hq, L, hd] and k [n_seq, Hkv, L, hd]."""
    sa = model.model.layers[layer].self_attn
    hd = sa.head_dim
    Wq = sa.q_proj.weight.numpy().astype(np.float64); bq = sa.q_proj.bias.numpy().astype(np.float64)
    Wk = sa.k_proj.weight.numpy().astype(np.float64); bk = sa.k_proj.bias.numpy().astype(np.float64)
    X = xn.astype(np.float64)
    q = X @ Wq.T + bq
    k = X @ Wk.T + bk
    n, L, _ = X.shape
    q = q.reshape(n, L, -1, hd).transpose(0, 2, 1, 3)
    k = k.reshape(n, L, -1, hd).transpose(0, 2, 1, 3)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


def wv_for_layer(model, layer: int):
    sa = model.model.layers[layer].self_attn
    hd = sa.head_dim
    W = sa.v_proj.weight.numpy().astype(np.float64)      # [n_kv*hd, D]
    b = sa.v_proj.bias.numpy().astype(np.float64)
    n_kv = W.shape[0] // hd
    return W.reshape(n_kv, hd, -1), b.reshape(n_kv, hd)


# ================================================================= mixing matrices
def causal_softmax(q, k, scaling):
    """q,k: [L, hd] fp64.  Returns row-stochastic causal A [L, L]."""
    L = q.shape[0]
    S = (q @ k.T) * scaling
    S = np.where(np.tril(np.ones((L, L), dtype=bool)), S, -np.inf)
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


_TRILCACHE = {}


def _tril(L):
    if L not in _TRILCACHE:
        _TRILCACHE[L] = np.tril(np.ones((L, L)))
    return _TRILCACHE[L]


def _rownorm_causal(Araw, already_causal=False):
    A = Araw if already_causal else Araw * _tril(Araw.shape[0])
    return A / np.maximum(A.sum(axis=1, keepdims=True), 1e-300)


def _elu1(x):
    return np.where(x > 0, x + 1.0, np.exp(np.clip(x, -60, 0)))


def _favor_features(x, W, stab):
    """FAVOR+ positive random features.  phi(x) = exp(w.x - ||x||^2/2 - stab)/sqrt(m).
    `stab` is a per-row vector (queries: row max) or a scalar (keys: global max) --
    a per-QUERY constant cancels in the row normalisation, a per-KEY constant does not,
    so keys must use one global constant."""
    p = x @ W.T - 0.5 * (x * x).sum(-1, keepdims=True)
    return np.exp(p - stab) / math.sqrt(W.shape[0])


def _hedgehog0(x):
    """Hedgehog-SHAPED elementwise map used WITHOUT fitting: the learned projection is
    replaced by the identity, phi(x) = softmax([x, -x]) over the 2*hd feature axis."""
    z = np.concatenate([x, -x], axis=-1)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


DECAY_GAMMAS = (0.1, 0.25, 0.35, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999)   # requested; ACHIEVED gamma is read back off A
WINDOWS = (32, 128, 512)
_LAGCACHE = {}


def _decay_matrix(L, gamma):
    key = (L, gamma)
    if key not in _LAGCACHE:
        lag = np.arange(L)[:, None] - np.arange(L)[None, :]
        _LAGCACHE[key] = np.where(lag >= 0, np.power(float(gamma), np.maximum(lag, 0.0)), 0.0)
    return _LAGCACHE[key]


_WINCACHE = {}


def _window_matrix(L, W):
    if (L, W) not in _WINCACHE:
        lag = np.arange(L)[:, None] - np.arange(L)[None, :]
        _WINCACHE[(L, W)] = ((lag >= 0) & (lag < W)).astype(np.float64)
    return _WINCACHE[(L, W)]


def make_phi(arm, hd, scaling):
    """Phi factory.  Returns None for arms that need no feature map."""
    if arm in PHIS or arm in PHIS_EXTRA or arm.startswith(("decay", "win")):
        return Phi(arm, hd, scaling)
    if arm in ("C5p_wrong_sequence", "C4_wrong_layer"):
        return Phi(C5P_C4_PHI, hd, scaling)
    if arm.startswith("C5p_") or arm.startswith("C4_"):
        # MATCHED null: the null is built with the SAME feature map as the arm it tests.
        return Phi(arm.split("_", 1)[1], hd, scaling)
    return None


class Phi:
    """Builds A_lin for one (sequence, head).  All maps consume the donor's OWN RoPE'd
    q,k and apply the donor's OWN scaling; no learned parameters anywhere."""

    def __init__(self, name, hd, scaling):
        self.name, self.hd, self.scaling = name, hd, scaling
        self.gamma = self.window = None
        if name.startswith("decay"):
            self.gamma = float(name.split("_g")[1])
        if name.startswith("win"):
            self.window = int(name.split("_w")[1])
        if name.startswith("favor"):
            rng = np.random.default_rng(FAVOR_SEED)
            G = rng.standard_normal((FAVOR_M, hd))
            # orthogonal blocks, renormalised to chi-distributed row norms (FAVOR+ standard)
            blocks = []
            for i in range(0, FAVOR_M, hd):
                m = min(hd, FAVOR_M - i)
                Q, _ = np.linalg.qr(rng.standard_normal((hd, hd)))
                blocks.append(Q[:m])
            Wo = np.concatenate(blocks, axis=0)
            self.W = Wo * np.linalg.norm(G, axis=1, keepdims=True)

    def A(self, q, k):
        s = self.scaling
        # ---- BRIEF AMENDMENT 1: mixings that carry a temporal envelope --------------
        if self.gamma is not None or self.window is not None:
            L = q.shape[0]
            env = (_decay_matrix(L, self.gamma) if self.gamma is not None
                   else _window_matrix(L, self.window))
            if self.name.startswith(("decayonly", "winonly")):
                return _rownorm_causal(env, already_causal=True)   # NO content whatsoever
            K = _elu1(q * s) @ _elu1(k).T                 # same kernel as the elu1 arm
            return _rownorm_causal(env * K, already_causal=True)
        if self.name == "elu1":
            return _rownorm_causal(_elu1(q * s) @ _elu1(k).T)
        if self.name == "elu1_noscale":
            return _rownorm_causal(_elu1(q) @ _elu1(k).T)
        if self.name == "hedgehog0_noscale":
            return _rownorm_causal(_hedgehog0(q) @ _hedgehog0(k).T)
        if self.name == "taylor2":
            u = (q @ k.T) * s
            return _rownorm_causal(1.0 + u + 0.5 * u * u)
        if self.name == "favor_d14":
            # FAITHFUL FAVOR+: the softmax-kernel approximation requires the d^{-1/4}
            # normaliser on BOTH q and k, so that phi(q').phi(k') ~ exp(q.k/sqrt(d)).
            # Applying the donor's d^{-1/2} to q alone (arm "favor") leaves ||k||^2/2
            # unnormalised and the map collapses -- see the probe doc.
            n14 = self.hd ** -0.25
            qs, ks = q * n14, k * n14
            pq = qs @ self.W.T - 0.5 * (qs * qs).sum(-1, keepdims=True)
            pk = ks @ self.W.T - 0.5 * (ks * ks).sum(-1, keepdims=True)
            Fq = np.exp(pq - pq.max(axis=1, keepdims=True)) / math.sqrt(FAVOR_M)
            Fk = np.exp(pk - pk.max()) / math.sqrt(FAVOR_M)
            return _rownorm_causal(Fq @ Fk.T)
        if self.name == "favor":
            qs = q * s
            pq = qs @ self.W.T - 0.5 * (qs * qs).sum(-1, keepdims=True)
            pk = k @ self.W.T - 0.5 * (k * k).sum(-1, keepdims=True)
            Fq = np.exp(pq - pq.max(axis=1, keepdims=True)) / math.sqrt(FAVOR_M)
            Fk = np.exp(pk - pk.max()) / math.sqrt(FAVOR_M)
            return _rownorm_causal(Fq @ Fk.T)
        if self.name == "hedgehog0":
            return _rownorm_causal(_hedgehog0(q * s) @ _hedgehog0(k).T)
        raise KeyError(self.name)


def causal_uniform(L):
    A = np.tril(np.ones((L, L)))
    return A / A.sum(axis=1, keepdims=True)


# ================================================================= the residual
def residual_from_Z(Z, M, tols=RANK_TOLS, tol_primary=RANK_TOL, want_cols=True):
    """residual = ||M (I - P_Z)||_F / ||M||_F, P_Z onto rowspace(Z).

    Route: G = Z Z^T (D x D, fp64), eigh, truncate at sigma > tol*sigma_max.
    An orthonormal basis of rowspace(Z) is Z^T V_r Lam_r^{-1/2}; only its action on M
    is ever needed, so no T x D basis is materialised.
    Returns dict incl. per-column residual norms at the primary tolerance.
    """
    D, T = Z.shape
    G = Z @ Z.T
    w, V = np.linalg.eigh(G)
    w = w[::-1]; V = V[:, ::-1]
    sig = np.sqrt(np.clip(w, 0.0, None))
    smax = sig[0] if sig.size else 0.0
    C = M @ Z.T                                   # d_v x D
    normM2 = float((M * M).sum())
    out = {"sigma_max": float(smax), "normM": math.sqrt(normM2)}
    for tol in tols:
        r = int((sig > tol * smax).sum())
        if r == 0:
            out[f"residual@{tol:g}"] = 1.0
            out[f"rank_Z@{tol:g}"] = 0
            continue
        B = (C @ V[:, :r]) / sig[:r]              # d_v x r
        proj2 = float((B * B).sum())
        out[f"residual@{tol:g}"] = math.sqrt(max(normM2 - proj2, 0.0) / normM2)
        out[f"rank_Z@{tol:g}"] = r
    r = int((sig > tol_primary * smax).sum())
    out["rank_Z"] = r
    out["residual"] = out[f"residual@{tol_primary:g}"]
    out["cond_ZZt"] = float((sig[0] / sig[r - 1]) ** 2) if r > 0 else float("inf")
    out["spectrum_decile_sigma_over_max"] = [
        float(sig[min(int(f * (D - 1)), D - 1)] / smax) for f in np.linspace(0, 1, 11)
    ] if smax > 0 else []
    if want_cols and r > 0:
        K = ((C @ V[:, :r]) / sig[:r]) @ (V[:, :r] / sig[:r]).T      # d_v x D
        R = M - K @ Z
        out["col_res2"] = (R * R).sum(axis=0)                        # length T
        out["col_M2"] = (M * M).sum(axis=0)
    return out


def exact_rank_qr(Z, tol=RANK_TOL):
    """Reference route that does NOT square the condition number: QR of Z^T, SVD of R."""
    _, R = np.linalg.qr(Z.T, mode="reduced")
    s = np.linalg.svd(R, compute_uv=False)
    return int((s > tol * s[0]).sum()), s


def residual_qr(Z, M, tol=RANK_TOL):
    """Cross-check route: explicit orthonormal basis via QR (T x D materialised)."""
    Q, R = np.linalg.qr(Z.T, mode="reduced")
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    r = int((s > tol * s[0]).sum())
    B = (M @ Q) @ U[:, :r]
    n2 = float((M * M).sum())
    return math.sqrt(max(n2 - float((B * B).sum()), 0.0) / n2), r


# ================================================================= entropy strata
def row_entropy(A):
    """Shannon entropy (nats) of each row of a row-stochastic causal A."""
    P = np.clip(A, 1e-300, None)
    return -(A * np.log(P)).sum(axis=1)


def strata_masks(ent_norm, n_seq, L):
    """Pre-registered strata.  Positions t < SHORT_PREFIX inside a sequence are a
    separate degenerate bucket (a 3-token prefix is trivially peaked).  The rest are
    split by quantile of NORMALISED entropy H_t / log(t+1)."""
    T = n_seq * L
    t_in_seq = np.tile(np.arange(L), n_seq)
    short = t_in_seq < SHORT_PREFIX
    long_ = ~short
    e = ent_norm[long_]
    qs = np.quantile(e, [0.1, 0.5, 0.9]) if e.size else [0, 0, 0]
    m = {}
    m["short_prefix"] = short
    m["peaked_d1"] = long_ & (ent_norm <= qs[0])
    m["mid"] = long_ & (ent_norm > qs[0]) & (ent_norm <= qs[2])
    m["diffuse_d10"] = long_ & (ent_norm > qs[2])
    m["all_long"] = long_
    m["all"] = np.ones(T, dtype=bool)
    return m, qs


def pool(col_res2, col_M2, mask):
    a = float(col_res2[mask].sum()); b = float(col_M2[mask].sum())
    return math.sqrt(a / b) if b > 0 else float("nan")


# ================================================================= the T <= D gate
class Degenerate(Exception):
    pass


def gate_TD(T, D, where=""):
    if T <= D:
        raise Degenerate(
            f"REFUSED [{where}]: T={T} <= D={D}. rowspace(Z) is generically all of R^T, "
            f"P_Z = I and the residual is exactly 0 for EVERY phi including a nonsensical "
            f"one. No residual is emitted. (brief section 1)")
    return True


# ================================================================= arm construction
def build_Z(model, layer, xn, q, k, arm, phi_cache, alt=None, wrong=None,
            transpose_bug=False):
    """Z = X A_lin^T, accumulated sequence-by-sequence.  A_lin is materialised only as an
    L x L block (8 MB fp64 at L=1024) -- never as T x T."""
    n_seq, L, D = xn.shape
    sa = model.model.layers[layer].self_attn
    Z = np.empty((D, n_seq * L), dtype=np.float64)
    for i in range(n_seq):
        Xi = xn[i].astype(np.float64).T                       # D x L
        if arm == "C8_no_mixing":
            Z[:, i * L:(i + 1) * L] = Xi
            continue
        if arm == "C7_random_gauss":
            Z[:, i * L:(i + 1) * L] = np.random.default_rng(9000 + i).standard_normal((D, L))
            continue
        if arm == "C6_causal_uniform":
            A = causal_uniform(L)
        elif arm == "C1_identity":
            A = causal_softmax(q[i], k[i], sa.scaling)
        else:
            qi, ki = q[i], k[i]
            if arm.startswith("C5p"):
                qi, ki = alt[0][i], alt[1][i]
            elif arm.startswith("C4"):
                qi, ki = wrong[0][i], wrong[1][i]
            A = phi_cache.A(qi, ki)
        Z[:, i * L:(i + 1) * L] = Xi @ (A if transpose_bug else A.T)
    return Z


# ================================================================= per-layer driver
def layer_pass(model, layer, xn, xn_alt, xn_wrong, cos, sin, arms, tol=RANK_TOL,
               heads=None, want_qr_check=False):
    n_seq, L, D = xn.shape
    T = n_seq * L
    gate_TD(T, D, f"layer {layer}")
    sa = model.model.layers[layer].self_attn
    hd, scaling = sa.head_dim, sa.scaling
    Wv, bv = wv_for_layer(model, layer)
    grp = sa.num_key_value_groups

    q, k = qk_for_layer(model, layer, xn, cos, sin)
    q_alt, k_alt = qk_for_layer(model, layer, xn_alt, cos, sin) if xn_alt is not None else (None, None)
    q_wr, k_wr = qk_for_layer(model, layer, xn_wrong, cos, sin) if xn_wrong is not None else (None, None)

    Hq = q.shape[1]
    heads = range(Hq) if heads is None else heads
    rows = []
    for h in heads:
        g = h // grp
        Wg = Wv[g]                                            # d_v x D
        M = np.empty((Wg.shape[0], T), dtype=np.float64)
        Zs = np.empty((D, T), dtype=np.float64)
        ent = np.empty(T)
        rowsum_lo, rowsum_hi = np.inf, -np.inf
        for i in range(n_seq):
            Xi = xn[i].astype(np.float64).T
            A = causal_softmax(q[i, h], k[i, h // grp], scaling)
            rowsum_lo = min(rowsum_lo, A.sum(1).min()); rowsum_hi = max(rowsum_hi, A.sum(1).max())
            M[:, i * L:(i + 1) * L] = (Wg @ Xi) @ A.T
            Zs[:, i * L:(i + 1) * L] = Xi @ A.T
            ent[i * L:(i + 1) * L] = row_entropy(A)
        t_in_seq = np.tile(np.arange(L), n_seq)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent_norm = np.where(t_in_seq > 0, ent / np.log(np.maximum(t_in_seq + 1, 2)), 0.0)
        masks, qs = strata_masks(ent_norm, n_seq, L)
        _ev = np.clip(np.linalg.eigvalsh(Zs @ Zs.T), 0.0, None)
        _sg = np.sqrt(_ev)
        rk_star = int((_sg > tol * _sg.max()).sum())

        for arm in arms:
            phi = make_phi(arm, hd, scaling)
            alt = (q_alt[:, h], k_alt[:, h // grp]) if q_alt is not None else None
            wrong = (q_wr[:, h], k_wr[:, h // grp]) if q_wr is not None else None
            Z = build_Z(model, layer, xn, q[:, h], k[:, h // grp], arm, phi, alt, wrong)
            r = residual_from_Z(Z, M, tol_primary=tol)
            rec = {
                "layer": int(layer), "head": int(h), "kv_group": int(g), "arm": arm,
                "T_achieved": int(T), "D_achieved": int(D), "TD_achieved": T / D,
                "n_seq_achieved": int(n_seq), "L_seq_achieved": int(L),
                "rank_Z": r["rank_Z"], "rank_Zstar": rk_star,
                "cond_ZZt": r["cond_ZZt"],
                "A_soft_rowsum_min": float(rowsum_lo), "A_soft_rowsum_max": float(rowsum_hi),
                "residual": r["residual"],
                "gamma_requested": (phi.gamma if phi is not None else None),
                "window_requested": (phi.window if phi is not None else None),
                "residual_random_subspace_ref": math.sqrt(max(1.0 - r["rank_Z"] / T, 0.0)),
                "ent_q10_norm": float(qs[0]), "ent_q50_norm": float(qs[1]), "ent_q90_norm": float(qs[2]),
            }
            for t2 in RANK_TOLS:
                rec[f"residual@{t2:g}"] = r[f"residual@{t2:g}"]
                rec[f"rank_Z@{t2:g}"] = r[f"rank_Z@{t2:g}"]
            for nm, mk in masks.items():
                rec[f"res_{nm}"] = pool(r["col_res2"], r["col_M2"], mk)
                rec[f"n_{nm}"] = int(mk.sum())
            if want_qr_check:
                rq, rr = residual_qr(Z, M, tol)
                rec["residual_qr_route"] = rq
                rec["rank_Z_qr_route"] = rr
            rows.append(rec)
            del Z
        del Zs, M
    return rows


# ================================================================= slices
def get_slices(tok, n_seq):
    ids, byts, meta = common.get_slice(tok, "calib", n_seq, SEQ_LEN, CALIB_SEED)
    ids2, _, meta2 = common.get_slice(tok, "calib", n_seq, SEQ_LEN, ALT_SEED)
    return ids, meta, ids2, meta2


def env_manifest():
    import transformers
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__, "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "seq_len": SEQ_LEN, "calib_seed": CALIB_SEED, "alt_seed": ALT_SEED,
        "favor_seed": FAVOR_SEED, "favor_m": FAVOR_M,
        "rank_tol_primary": RANK_TOL, "rank_tols": list(RANK_TOLS),
        "short_prefix": SHORT_PREFIX,
    }


# ================================================================= STAGE: selfcheck
def stage_selfcheck():
    t0 = time.time()
    out = {"env": env_manifest()}
    model, tok = load_donor()
    arch = achieved_arch(model)
    out["arch_achieved"] = arch
    out["donor_revision"] = revision_hash_on_disk()
    D = arch["D_achieved"]
    print(json.dumps(arch, indent=2))
    print(json.dumps(out["donor_revision"], indent=2))

    # ---- S1. my A_soft vs the model's own attention probabilities -------------------
    ids, meta, ids2, meta2 = get_slices(tok, 2)
    small = ids[:1, :256]
    o = model.model(small, output_attentions=True, attn_implementation="eager")
    A_model = o.attentions[3][0].numpy().astype(np.float64)          # [Hq, 256, 256]
    xn = capture(model, small, [3])[3]
    cos, sin = rope_cos_sin(model, 256)
    q, k = qk_for_layer(model, 3, xn, cos, sin)
    sa = model.model.layers[3].self_attn
    dmax = 0.0
    for h in range(A_model.shape[0]):
        A_mine = causal_softmax(q[0, h], k[0, h // sa.num_key_value_groups], sa.scaling)
        dmax = max(dmax, float(np.abs(A_mine - A_model[h]).max()))
    out["S1_A_soft_vs_model_max_abs"] = dmax
    print(f"S1  my A_soft vs model output_attentions : max_abs = {dmax:.3e}")

    # ---- S2. commuting step on the real donor ---------------------------------------
    Wv, bv = wv_for_layer(model, 3)
    X = xn[0].astype(np.float64).T
    A0 = causal_softmax(q[0, 0], k[0, 0], sa.scaling)
    lhs = (Wv[0] @ X + bv[0][:, None]) @ A0.T
    rhs = Wv[0] @ (X @ A0.T) + bv[0][:, None]
    out["S2_commute_with_bias_max_abs"] = float(np.abs(lhs - rhs).max())
    out["S2_A_soft_rowsum_dev"] = float(np.abs(A0.sum(1) - 1).max())
    print(f"S2  |W_v(XA^T)+b  -  (W_vX+b)A^T| = {out['S2_commute_with_bias_max_abs']:.3e}"
          f"   rowsum dev {out['S2_A_soft_rowsum_dev']:.3e}")

    # ---- S3. every phi is row-normalised (audit finding 3) ---------------------------
    out["S3_phi_rowsum"] = {}
    for name in PHIS:
        p = Phi(name, sa.head_dim, sa.scaling)
        A = p.A(q[0, 0], k[0, 0])
        out["S3_phi_rowsum"][name] = [float(A.sum(1).min()), float(A.sum(1).max()),
                                      float(A.min()), float(np.isfinite(A).all())]
        print(f"S3  phi={name:10s} rowsum [{A.sum(1).min():.6f},{A.sum(1).max():.6f}] "
              f"min_entry={A.min():.3e} finite={np.isfinite(A).all()}")

    # ---- S4. the T <= D REFUSAL, demonstrated deliberately ---------------------------
    print(f"S4  deliberately asking for T <= D  (n_seq=1 -> T={SEQ_LEN}, D={D})")
    try:
        ids1, _, _ = common.get_slice(tok, "calib", 1, SEQ_LEN, CALIB_SEED)
        xn1 = capture(model, ids1, [3])[3]
        c1, s1 = rope_cos_sin(model, SEQ_LEN)
        layer_pass(model, 3, xn1, None, None, c1, s1, ["elu1"], heads=[0])
        out["S4_refusal"] = "FAILED TO REFUSE"
        print("S4  *** THE INSTRUMENT DID NOT REFUSE -- everything below is void ***")
    except Degenerate as e:
        out["S4_refusal"] = str(e)
        print(f"S4  {e}")

    # ---- S4b. and it would have emitted a flattering ZERO ----------------------------
    xn1 = capture(model, ids[:1], [3])[3]
    c1, s1 = rope_cos_sin(model, SEQ_LEN)
    q1, k1 = qk_for_layer(model, 3, xn1, c1, s1)
    Wg = Wv[0]
    Msm = np.empty((Wg.shape[0], SEQ_LEN)); Xi = xn1[0].astype(np.float64).T
    A = causal_softmax(q1[0, 0], k1[0, 0], sa.scaling)
    Msm[:] = (Wg @ Xi) @ A.T
    demo = {}
    for name in ("elu1", "hedgehog0"):
        p = Phi(name, sa.head_dim, sa.scaling)
        Zb = Xi @ p.A(q1[0, 0], k1[0, 0]).T
        demo[name] = residual_from_Z(Zb, Msm, want_cols=False)["residual"]
    # a deliberately NONSENSICAL A: constant 1/L on the FULL square, not causal at all
    Anon = np.full((SEQ_LEN, SEQ_LEN), 1.0 / SEQ_LEN)
    demo["nonsense_noncausal_uniform"] = residual_from_Z(Xi @ Anon.T, Msm, want_cols=False)["residual"]
    out["S4b_residual_when_T_le_D"] = demo
    print("S4b at T<=D the residual is a flattering artefact:", json.dumps(demo))

    # ---- S5. C1 IDENTITY must fire (residual exactly 0) ------------------------------
    ids8, meta8, ids8b, meta8b = get_slices(tok, 2)
    xn2 = capture(model, ids8, [3])[3]
    cos, sin = rope_cos_sin(model, SEQ_LEN)
    rows = layer_pass(model, 3, xn2, None, None, cos, sin, ["C1_identity"], heads=[0],
                      want_qr_check=True)
    out["S5_C1_identity"] = rows[0]
    print(f"S5  C1 IDENTITY residual = {rows[0]['residual']:.3e} "
          f"(qr route {rows[0]['residual_qr_route']:.3e})  rank_Z={rows[0]['rank_Z']} "
          f"T={rows[0]['T_achieved']} D={D}   [TAUTOLOGY of the algebra: Z = Z*]")

    # ---- S6. PLANTED POSITIVE on the phi -> A_lin -> Z path --------------------------
    q2, k2 = qk_for_layer(model, 3, xn2, cos, sin)
    Zpl = build_Z(model, 3, xn2, q2[:, 0], k2[:, 0], "elu1", Phi("elu1", sa.head_dim, sa.scaling))
    rng = np.random.default_rng(11)
    Wplant = rng.standard_normal((128, D))
    Mplant = Wplant @ Zpl
    rp = residual_from_Z(Zpl, Mplant, want_cols=False)
    out["S6_planted_positive_residual"] = rp["residual"]
    print(f"S6  PLANTED POSITIVE (target built from the LINEAR side) residual = "
          f"{rp['residual']:.3e}   <- fires the whole phi->A_lin->Z path")

    # ---- S7. transpose trap: Z = X A_lin (no transpose) -- same shape, silent bug ----
    Msoft = np.empty((128, xn2.shape[0] * SEQ_LEN))
    for i in range(xn2.shape[0]):
        Xi = xn2[i].astype(np.float64).T
        A = causal_softmax(q2[i, 0], k2[i, 0], sa.scaling)
        Msoft[:, i * SEQ_LEN:(i + 1) * SEQ_LEN] = (Wv[0] @ Xi) @ A.T
    r_ok = residual_from_Z(Zpl, Msoft, want_cols=False)["residual"]
    Zbug = build_Z(model, 3, xn2, q2[:, 0], k2[:, 0], "elu1",
                   Phi("elu1", sa.head_dim, sa.scaling), transpose_bug=True)
    r_bug = residual_from_Z(Zbug, Msoft, want_cols=False)["residual"]
    out["S7_transpose_trap"] = {"correct": r_ok, "A_lin_untransposed": r_bug}
    print(f"S7  TRANSPOSE TRAP  correct={r_ok:.4f}   A_lin untransposed={r_bug:.4f}")

    # ---- S8. O(1)-state recurrence vs the L x L block form ---------------------------
    Xi = xn2[0].astype(np.float64).T
    qh, kh = q2[0, 0] * sa.scaling, k2[0, 0]
    Fq, Fk = _elu1(qh), _elu1(kh)
    L = SEQ_LEN
    Zrec = np.empty((D, L))
    S = np.zeros((D, sa.head_dim)); n = np.zeros(sa.head_dim)
    for t in range(L):
        S += np.outer(Xi[:, t], Fk[t]); n += Fk[t]
        Zrec[:, t] = (S @ Fq[t]) / max(float(Fq[t] @ n), 1e-300)
    Zmat = Xi @ Phi("elu1", sa.head_dim, sa.scaling).A(q2[0, 0], k2[0, 0]).T
    d = float(np.abs(Zrec - Zmat).max()); rel = d / float(np.abs(Zmat).max())
    out["S8_recurrence_vs_matrix"] = {"max_abs": d, "rel": rel,
                                      "state_shape_D_by_m": [int(D), int(sa.head_dim)],
                                      "state_bytes_fp32_mix_then_project": int(D * sa.head_dim * 4),
                                      "state_bytes_fp32_project_then_mix": int(128 * sa.head_dim * 4)}
    print(f"S8  O(1)-recurrence vs L x L block: max_abs={d:.3e} rel={rel:.3e}")

    # ---- S9. eigh route vs QR route (normal-equation squaring) -----------------------
    rq, rr = residual_qr(Zpl, Msoft)
    out["S9_route_cross_check"] = {"eigh_route": r_ok, "qr_route": rq,
                                   "rank_eigh": residual_from_Z(Zpl, Msoft, want_cols=False)["rank_Z"],
                                   "rank_qr": rr}
    print(f"S9  eigh route {r_ok:.6f}  vs  QR route {rq:.6f}   (rank {out['S9_route_cross_check']['rank_eigh']} vs {rr})")

    out["wall_s"] = time.time() - t0
    p = os.path.join(RESULTS, "r2a_selfcheck.json")
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}   [{out['wall_s']:.0f}s]")


# ================================================================= STAGE: main
MAIN_NSEQ = 8                                   # T = 8192, T/D = 5.333
MAIN_ARMS_ALL_LAYERS = list(PHIS) + ["C1_identity", "C6_causal_uniform", "C8_no_mixing"]
MAIN_ARMS_SUBSET = list(PHIS) + list(CONTROLS) + list(FLOORS)
SUBSET_LAYERS = (0, 7, 14, 21, 27)


def stage_main():
    t0 = time.time()
    model, tok = load_donor()
    arch = achieved_arch(model)
    D = arch["D_achieved"]; Lyr = arch["n_layers_achieved"]
    ids, meta, ids2, meta2 = get_slices(tok, MAIN_NSEQ)
    T = MAIN_NSEQ * SEQ_LEN
    gate_TD(T, D, "main")
    print(f"main: T={T} D={D} T/D={T/D:.3f}  layers=0..{Lyr-1}")
    rev = revision_hash_on_disk()
    cos, sin = rope_cos_sin(model, SEQ_LEN)
    all_layers = list(range(Lyr))
    print("capturing activations ...", flush=True)
    xn_all = capture(model, ids, all_layers)
    xn_alt_all = capture(model, ids2, list(SUBSET_LAYERS))
    print(f"captured, {time.time()-t0:.0f}s", flush=True)

    rows = []
    outp = os.path.join(RESULTS, "r2a_main.json")
    for l in all_layers:
        arms = MAIN_ARMS_SUBSET if l in SUBSET_LAYERS else MAIN_ARMS_ALL_LAYERS
        wrong_l = (l + Lyr // 2) % Lyr
        xn_w = xn_all[wrong_l] if any(a.startswith("C4") for a in arms) else None
        xn_a = xn_alt_all.get(l) if any(a.startswith("C5p") for a in arms) else None
        tl = time.time()
        rows += layer_pass(model, l, xn_all[l], xn_a, xn_w, cos, sin, arms,
                           want_qr_check=(l in (0, 14)))
        rr = [r for r in rows if r["layer"] == l]
        summ = {a: float(np.median([r["residual"] for r in rr if r["arm"] == a])) for a in arms}
        print(f"layer {l:2d} [{time.time()-tl:.0f}s] median residual: " +
              "  ".join(f"{a}={v:.4f}" for a, v in summ.items()), flush=True)
        json.dump({"env": env_manifest(), "arch_achieved": arch,
                   "donor_revision": rev,
                   "slice_meta": meta, "alt_slice_meta": meta2,
                   "rows": rows, "wall_s": time.time() - t0},
                  open(outp, "w"), indent=2, default=float)
    print(f"wrote {outp}  [{time.time()-t0:.0f}s]")


# ================================================================= STAGE: curve
CURVE_NSEQ = (1, 2, 4, 8, 16, 32)   # T/D = 0.667(REFUSED), 1.33, 2.67, 5.33, 10.67, 21.33
CURVE_LAYERS = (0, 14, 27)
CURVE_ARMS = ("elu1", "taylor2", "favor", "hedgehog0", "C6_causal_uniform",
              "C8_no_mixing", "C7_random_gauss", "C1_identity")
CURVE_HEADS = (0, 5, 11)


def stage_curve():
    t0 = time.time()
    model, tok = load_donor()
    arch = achieved_arch(model); D = arch["D_achieved"]
    cos, sin = rope_cos_sin(model, SEQ_LEN)
    rows, refusals = [], []
    outp = os.path.join(RESULTS, "r2a_curve.json")
    nmax = max(CURVE_NSEQ)
    ids_max, meta_max, _, _ = get_slices(tok, nmax)
    # the curve is NESTED: slice(n) is the prefix of slice(nmax), verified, so one capture serves all
    for n in CURVE_NSEQ:
        idn, _, _ = common.get_slice(tok, "calib", n, SEQ_LEN, CALIB_SEED)
        assert torch.equal(idn, ids_max[:n]), f"slice nesting broken at n={n}"
    print("slice nesting verified: slice(n) == slice(nmax)[:n] for all n", flush=True)
    xn_max = capture(model, ids_max, list(CURVE_LAYERS))
    for n in CURVE_NSEQ:
        T = n * SEQ_LEN
        try:
            gate_TD(T, D, f"curve n_seq={n}")
        except Degenerate as e:
            refusals.append({"n_seq": n, "T": T, "D": D, "message": str(e)})
            print(f"n_seq={n} T={T}: {e}", flush=True)
            continue
        xn = {l: xn_max[l][:n] for l in CURVE_LAYERS}
        for l in CURVE_LAYERS:
            tl = time.time()
            rows += layer_pass(model, l, xn[l], None, None, cos, sin, list(CURVE_ARMS),
                               heads=list(CURVE_HEADS))
            print(f"  n_seq={n:2d} T/D={T/D:5.2f} layer {l:2d} [{time.time()-tl:.0f}s]", flush=True)
        json.dump({"env": env_manifest(), "arch_achieved": arch, "refusals": refusals,
                   "curve_layers": list(CURVE_LAYERS), "curve_heads": list(CURVE_HEADS),
                   "slice_meta": meta_max,
                   "rows": rows, "wall_s": time.time() - t0}, open(outp, "w"), indent=2, default=float)
    print(f"wrote {outp}  [{time.time()-t0:.0f}s]")


def stage_extra():
    """The arms added after the main sweep: the faithful FAVOR+ (`favor_d14`) on every
    layer, and the phi-scaling sensitivity spot check."""
    t0 = time.time()
    model, tok = load_donor(); arch = achieved_arch(model)
    D = arch["D_achieved"]; Lyr = arch["n_layers_achieved"]
    ids, meta, _, _ = get_slices(tok, MAIN_NSEQ)
    gate_TD(MAIN_NSEQ * SEQ_LEN, D, "extra")
    cos, sin = rope_cos_sin(model, SEQ_LEN)
    all_layers = list(range(Lyr))
    xn_all = capture(model, ids, all_layers)
    rows = []
    outp = os.path.join(RESULTS, "r2a_extra.json")
    for l in all_layers:
        arms = ["favor_d14"] + (list(PHIS_EXTRA) if l in SUBSET_LAYERS else [])
        tl = time.time()
        rows += layer_pass(model, l, xn_all[l], None, None, cos, sin, arms)
        rr = [r for r in rows if r["layer"] == l]
        print(f"layer {l:2d} [{time.time()-tl:.0f}s] " + "  ".join(
            f"{a}={np.median([r['residual'] for r in rr if r['arm']==a]):.4f}" for a in arms), flush=True)
        json.dump({"env": env_manifest(), "arch_achieved": arch, "slice_meta": meta,
                   "rows": rows, "wall_s": time.time() - t0}, open(outp, "w"), indent=2, default=float)
    print(f"wrote {outp}  [{time.time()-t0:.0f}s]")


DECAY_ALL_LAYERS = ["favor_d14", "decay_g0.99", "decayonly_g0.99",
                    "win_w128", "winonly_w128"]
DECAY_SUBSET_EXTRA = ["elu1_noscale", "hedgehog0_noscale",
                      "decay_g0.5", "decayonly_g0.5", "decay_g0.9", "decayonly_g0.9",
                      "decay_g0.95", "decayonly_g0.95", "decay_g0.999", "decayonly_g0.999",
                      "win_w32", "winonly_w32", "win_w512", "winonly_w512"]


def achieved_envelope(model, layer, xn, cos, sin):
    """gamma and window read back off the CONSTRUCTED A, never off the requested name."""
    sa = model.model.layers[layer].self_attn
    q, k = qk_for_layer(model, layer, xn[:1], cos, sin)
    out = {}
    for g in DECAY_GAMMAS:
        A = Phi(f"decayonly_g{g}", sa.head_dim, sa.scaling).A(q[0, 0], k[0, 0])
        lag2 = np.array([A[t, t - 2] for t in range(600, 900)])
        lag1 = np.array([A[t, t - 1] for t in range(600, 900)])
        out[f"decayonly_g{g}"] = {"gamma_requested": g,
                                  "gamma_achieved": float(np.median(lag2 / lag1))}
    for w in WINDOWS:
        A = Phi(f"winonly_w{w}", sa.head_dim, sa.scaling).A(q[0, 0], k[0, 0])
        t = 900
        nz = np.nonzero(A[t])[0]
        out[f"winonly_w{w}"] = {"window_requested": w,
                                "window_achieved": int(t - nz.min() + 1),
                                "n_nonzero_in_row": int(nz.size)}
    return out


def stage_decay():
    """BRIEF AMENDMENT 1 (decay / window mixings) + the faithful FAVOR+ + phi-scaling
    sensitivity.  Same donor, same slice, same T/D, same layers, same heads as `main`."""
    t0 = time.time()
    model, tok = load_donor(); arch = achieved_arch(model)
    D = arch["D_achieved"]; Lyr = arch["n_layers_achieved"]
    ids, meta, _, _ = get_slices(tok, MAIN_NSEQ)
    gate_TD(MAIN_NSEQ * SEQ_LEN, D, "decay")
    print(f"decay stage: T={MAIN_NSEQ*SEQ_LEN} D={D} T/D={MAIN_NSEQ*SEQ_LEN/D:.4f} "
          f"(IDENTICAL to the main sweep)", flush=True)
    cos, sin = rope_cos_sin(model, SEQ_LEN)
    xn_all = capture(model, ids, list(range(Lyr)))
    env = achieved_envelope(model, 14, xn_all[14], cos, sin)
    print("ACHIEVED envelope, read back off the constructed A:")
    for k_, v in env.items():
        print("   ", k_, v, flush=True)
    rows = []
    outp = os.path.join(RESULTS, "r2a_decay.json")
    for l in range(Lyr):
        arms = list(DECAY_ALL_LAYERS) + (DECAY_SUBSET_EXTRA if l in SUBSET_LAYERS else [])
        tl = time.time()
        rows += layer_pass(model, l, xn_all[l], None, None, cos, sin, arms)
        rr = [r for r in rows if r["layer"] == l]
        print(f"layer {l:2d} [{time.time()-tl:.0f}s] " + "  ".join(
            f"{a.replace('decay','d').replace('only','O')}="
            f"{np.median([r['residual'] for r in rr if r['arm']==a]):.4f}" for a in arms), flush=True)
        json.dump({"env": env_manifest(), "arch_achieved": arch, "slice_meta": meta,
                   "achieved_envelope": env, "rows": rows, "wall_s": time.time() - t0},
                  open(outp, "w"), indent=2, default=float)
    print(f"wrote {outp}  [{time.time()-t0:.0f}s]")


GAMMA_ALL = ["decay_g0.1", "decay_g0.25", "decay_g0.35", "decay_g0.5", "decay_g0.7"]
GAMMA_SUBSET = ["decayonly_g0.1", "decayonly_g0.25", "decayonly_g0.35", "decayonly_g0.7"]


def mixing_geometry(model, layer, xn, cos, sin, heads=(0, 5, 11)):
    """MECHANISM probe: is `decay == decayonly` because the two MATRICES are nearly
    identical (an artefact of how content and envelope combine), or because they are
    genuinely different matrices whose ROW SPACES land in the same place (a real null)?

    Reports, per pair of mixings, the relative Frobenius distance between the A matrices
    themselves and the mean row-wise cosine similarity -- neither of which involves the
    projector at all.
    """
    sa = model.model.layers[layer].self_attn
    q, k = qk_for_layer(model, layer, xn[:1], cos, sin)
    L = xn.shape[1]
    out = {}

    def A_of(arm):
        if arm == "C6_causal_uniform":
            return causal_uniform(L)
        if arm == "C8_no_mixing":
            return np.eye(L)
        if arm == "A_soft":
            return causal_softmax(q[0, h], k[0, h // sa.num_key_value_groups], sa.scaling)
        return Phi(arm, sa.head_dim, sa.scaling).A(q[0, h], k[0, h // sa.num_key_value_groups])

    pairs = [("decay_g0.99", "decayonly_g0.99"), ("decay_g0.5", "decayonly_g0.5"),
             ("decay_g0.9", "decayonly_g0.9"), ("win_w128", "winonly_w128"),
             ("elu1", "C6_causal_uniform"), ("taylor2", "C6_causal_uniform"),
             ("elu1", "C8_no_mixing"), ("A_soft", "C6_causal_uniform"),
             ("A_soft", "elu1"), ("A_soft", "decay_g0.5")]
    for a, b in pairs:
        rel, cos_ = [], []
        for h in heads:
            A, B = A_of(a), A_of(b)
            rel.append(float(np.linalg.norm(A - B) / np.linalg.norm(B)))
            num = (A * B).sum(1)
            den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
            cos_.append(float(np.median(num / np.maximum(den, 1e-300))))
        out[f"{a} vs {b}"] = {"rel_fro_dist": float(np.median(rel)),
                              "median_row_cosine": float(np.median(cos_))}
    return out


def stage_gamma():
    """Locate the minimum of the gamma curve. C8_no_mixing is the gamma->0 limit and
    C6_causal_uniform the gamma->1 limit of this same family, so both endpoints already
    exist; this fills the interior on ALL 28 layers.  Same T/D, slice, layers, heads."""
    t0 = time.time()
    model, tok = load_donor(); arch = achieved_arch(model)
    D = arch["D_achieved"]; Lyr = arch["n_layers_achieved"]
    ids, meta, _, _ = get_slices(tok, MAIN_NSEQ)
    gate_TD(MAIN_NSEQ * SEQ_LEN, D, "gamma")
    print(f"gamma stage: T={MAIN_NSEQ*SEQ_LEN} D={D} T/D={MAIN_NSEQ*SEQ_LEN/D:.4f} "
          f"(IDENTICAL to main and decay)", flush=True)
    cos, sin = rope_cos_sin(model, SEQ_LEN)
    xn_all = capture(model, ids, list(range(Lyr)))
    env = achieved_envelope(model, 14, xn_all[14], cos, sin)
    print("ACHIEVED envelope:")
    for k_, v in env.items():
        print("   ", k_, v, flush=True)
    geo = {l: mixing_geometry(model, l, xn_all[l], cos, sin) for l in (0, 14, 27)}
    print("")
    print("MIXING GEOMETRY (are the matrices themselves different?)")
    for l, g in geo.items():
        print(f"  layer {l}:")
        for k_, v in g.items():
            print(f"    {k_:36s} rel_fro={v['rel_fro_dist']:.4f}  row_cos={v['median_row_cosine']:.4f}",
                  flush=True)
    rows = []
    outp = os.path.join(RESULTS, "r2a_gamma.json")
    for l in range(Lyr):
        arms = list(GAMMA_ALL) + (GAMMA_SUBSET if l in SUBSET_LAYERS else [])
        tl = time.time()
        rows += layer_pass(model, l, xn_all[l], None, None, cos, sin, arms)
        rr = [r for r in rows if r["layer"] == l]
        print(f"layer {l:2d} [{time.time()-tl:.0f}s] " + "  ".join(
            f"{a}={np.median([r['residual'] for r in rr if r['arm']==a]):.4f}" for a in arms), flush=True)
        json.dump({"env": env_manifest(), "arch_achieved": arch, "slice_meta": meta,
                   "achieved_envelope": env, "mixing_geometry": geo,
                   "rows": rows, "wall_s": time.time() - t0}, open(outp, "w"), indent=2, default=float)
    print(f"wrote {outp}  [{time.time()-t0:.0f}s]")


NULL_ALL_LAYERS = ["C5p_taylor2", "C4_taylor2"]
NULL_SUBSET = ["C5p_hedgehog0", "C4_hedgehog0", "C5p_favor_d14", "C4_favor_d14",
               "C5p_elu1", "C4_elu1"]


def qk_logit_range(model, layer, xn, cos, sin, heads=(0, 5, 11)):
    """The range of z = scaling * q.k over CAUSAL pairs, per head.

    `taylor2` uses 1 + z + z^2/2, which is NON-MONOTONE below z = -1: it up-weights
    increasingly DISSIMILAR query-key pairs.  If a material fraction of z sits below -1,
    taylor2's separation could be an artefact of spreading mass onto unrelated positions
    rather than of modelling the donor.  Nobody has published a decoder-only donor's
    q.k range, so it is measured here.
    """
    sa = model.model.layers[layer].self_attn
    q, k = qk_for_layer(model, layer, xn[:1], cos, sin)
    L = q.shape[2]
    m = np.tril(np.ones((L, L), dtype=bool))
    out = {}
    for h in heads:
        z = (q[0, h] @ k[0, h // sa.num_key_value_groups].T) * sa.scaling
        v = z[m]
        out[f"head{h}"] = {
            "min": float(v.min()), "max": float(v.max()),
            "p0.1": float(np.percentile(v, 0.1)), "p1": float(np.percentile(v, 1)),
            "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)),
            "frac_below_-1": float((v < -1).mean()),
            "frac_below_-1_massweighted": float(
                (np.maximum(1 + v[v < -1] + 0.5 * v[v < -1] ** 2, 0).sum())
                / np.maximum(1 + v + 0.5 * v ** 2, 0).sum()),
        }
    return out


def stage_nulls():
    """MATCHED nulls: C5' (wrong sequence) and C4 (wrong layer) rebuilt with the SAME
    feature map as the arm they are testing.  The originals used the elu1 kernel for
    every arm, which is not a matched control for taylor2.  Same T/D, slice, layers,
    heads as `main`."""
    t0 = time.time()
    model, tok = load_donor(); arch = achieved_arch(model)
    D = arch["D_achieved"]; Lyr = arch["n_layers_achieved"]
    ids, meta, ids2, meta2 = get_slices(tok, MAIN_NSEQ)
    gate_TD(MAIN_NSEQ * SEQ_LEN, D, "nulls")
    print(f"nulls stage: T={MAIN_NSEQ*SEQ_LEN} D={D} T/D={MAIN_NSEQ*SEQ_LEN/D:.4f} "
          f"(IDENTICAL to main, decay, gamma)", flush=True)
    cos, sin = rope_cos_sin(model, SEQ_LEN)
    xn_all = capture(model, ids, list(range(Lyr)))
    xn_alt_all = capture(model, ids2, list(range(Lyr)))
    zr = {l: qk_logit_range(model, l, xn_all[l], cos, sin) for l in (0, 7, 14, 21, 27)}
    print("q.k LOGIT RANGE (causal pairs; taylor2 is non-monotone below z = -1)")
    for l, g in zr.items():
        for h, v in g.items():
            print(f"  layer {l:2d} {h}: min={v['min']:8.2f} p0.1={v['p0.1']:7.2f} "
                  f"p50={v['p50']:6.2f} max={v['max']:7.2f}  frac(z<-1)={v['frac_below_-1']:.4f} "
                  f"massfrac={v['frac_below_-1_massweighted']:.4f}", flush=True)
    rows = []
    outp = os.path.join(RESULTS, "r2a_nulls.json")
    for l in range(Lyr):
        arms = list(NULL_ALL_LAYERS) + (NULL_SUBSET if l in SUBSET_LAYERS else [])
        wrong_l = (l + Lyr // 2) % Lyr
        tl = time.time()
        rows += layer_pass(model, l, xn_all[l], xn_alt_all[l], xn_all[wrong_l],
                           cos, sin, arms)
        rr = [r for r in rows if r["layer"] == l]
        print(f"layer {l:2d} [{time.time()-tl:.0f}s] " + "  ".join(
            f"{a}={np.median([r['residual'] for r in rr if r['arm']==a]):.4f}" for a in arms),
            flush=True)
        json.dump({"env": env_manifest(), "arch_achieved": arch, "slice_meta": meta,
                   "alt_slice_meta": meta2, "qk_logit_range": zr,
                   "wrong_layer_rule": "(l + 14) mod 28",
                   "rows": rows, "wall_s": time.time() - t0}, open(outp, "w"), indent=2,
                  default=float)
    print(f"wrote {outp}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    {"selfcheck": stage_selfcheck, "main": stage_main, "curve": stage_curve,
     "extra": stage_extra, "decay": stage_decay, "gamma": stage_gamma,
     "nulls": stage_nulls}[sys.argv[1]]()
