#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T3 -- does rotating the residual basis before ternarizing reduce the damage R3 leaves?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_T3_ROTATION.md, pushed at 7cdeca8.
Section 4's thresholds are hard-coded below and the label is read off them mechanically.

THE CONSTRUCTION.  For orthogonal R, y = Wx = (WR)(R^T x).  Rotating the residual stream is a
change of basis, exact in fp terms, and free at inference because R^T x is what the previous
organ already produces.  Offline it is:

  0. UNTIE lm_head from the embedding.  They are the same tensor on this donor, and the final
     norm's gain has to be folded into the head WITHOUT touching the input lookup.
  1. FOLD every RMSNorm gain into the linears that read that norm.  After folding the norm is a
     pure rescale by ||x||, and ||Rx|| = ||x||, so the norm commutes with R.
       input_layernorm.gamma          -> q, k, v
       post_attention_layernorm.gamma -> gate, up
       model.norm.gamma               -> lm_head
  2. ROTATE once.  Everything that READS the stream gets W @ R (embed rows included, since an
     embedding row is a stream vector); everything that WRITES to the stream gets R^T @ W.
       readers: embed_tokens, q, k, v, gate, up, lm_head
       writers: o_proj, down_proj
     o_proj's and down_proj's INPUTS are head-concat and FFN-hidden, not the stream, so their
     input side is untouched -- which is exactly why this rotation needs no runtime multiply.

  The engine sees the same shapes, the same format, one fp32 scale per output row and no new
  kernel.  donor_engine.c is not modified by this probe at all.

TWO DEPARTURES FROM THE BRIEF, both ADDITIVE -- controls added, nothing removed or relabelled.

  (a) THE BRIEF'S ARM P IS ANALYTICALLY AN EXACT NULL, not merely a weak one.  The brief's
      reasoning was that "any rotation reshuffles which weight lands in which row".  For a
      RESIDUAL-STREAM rotation that is false for a permutation: permuting the stream permutes
      the COLUMNS of every reader and the ROWS of every writer, and R3 searches one scale per
      output row over (weight, act_rms) pairs.  Permuting a row's columns leaves that set
      unchanged; permuting whole rows relabels them.  So R3 on a permuted model must reproduce
      R3 on the unpermuted one to fp round-off.  P is therefore an INSTRUMENT control, and the
      brief's PERMUTATION-ARTEFACT branch is unreachable by construction.  P is still run and
      still gates: a P that DIFFERS is a bug in the rotation machinery, not a finding.

  (b) THE FOLD IS A SEPARATE VARIABLE FROM THE ROTATION.  Arm Q (the T2 replication gate) does
      not fold, because it has to reproduce T2's +1.709372 exactly.  Every rotated arm does
      fold.  So Q vs a rotated arm confounds fold with rotation.  Arm N -- fold only, identity
      rotation, R3 -- separates them, and it is N, not Q, that H and O should be read against
      for a mechanism claim.  The brief's LABEL is still computed against Q, verbatim.

  Cost: the brief said "six arms plus one calibration pass". It needs one capture PER BASIS --
  R3 weights by the activation RMS of its actual input, and the rotated stream has different
  per-channel RMS. That is a correctness requirement, not a design change.

Env: D_THREADS (6), T3_ONLY (comma list), T3_SMOKE (1 = 2 seq / 4 layers / 4 calib seq)
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
sys.path.insert(0, HERE)
import common as C                                                        # noqa: E402
import t2_rules as T2                                                     # noqa: E402
from d2_basis import hadamard_blocks, dense_random_orth, permutation      # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("T3_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("T3_ONLY", "").split(",") if x.strip()]

SIGMA_SEED = 0.005
BASELINE_STANDING = 0.7675949641196624
T2_R3_STANDING = 1.709372          # T2's winning FFN Delta, results/t2_rules.json
D2_SEED = 4242                     # d2_basis.py's SEED constant, same constructions

# ---------------------------------------------------------------- brief section 4, verbatim
WINS_MARGIN = 0.30                 # best <= q - 0.30
MARGINAL_MARGIN = 0.05             # q - 0.30 < best <= q - 0.05
PLANTED_MARGIN = 0.20              # Delta(P) >= best + 0.20 for the mechanism to stand
X_EXACTNESS_TOL = 1e-4             # arm X must reproduce base to better than this
Q_REPLICATION_TOL = 0.01           # arm Q must land within this of T2's Delta

N_CAL, SEQ_CAL, SEED_CAL = 32, 512, 42424      # identical to T2, T2b and D4

SUFFIX = "_smoke" if SMOKE else ""
OUT = os.path.join(C.RESULTS, "t3_rotation%s.json" % SUFFIX)
ARMDIR = os.path.join(C.RESULTS, "t3_arms%s" % SUFFIX)
os.makedirs(ARMDIR, exist_ok=True)

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")
ORGANS = FFN + ATTN                # brief s3: FFN + attention. The head is NOT converted.

#            arm  -> (basis, fold, quantize)
ARM_SPEC = {
    "XN":  (None,        True,  False),   # fold only, no rotation, no quant -- is the FOLD exact?
    "XP":  ("perm",      True,  False),   # fold + permutation, no quant
    "XH":  ("hadamard",  True,  False),   # fold + block Hadamard, no quant
    "XO":  ("orth",      True,  False),   # fold + dense random orthogonal, no quant
    "Q":   (None,        False, True),    # R3, no fold, no rotation -- T2 replication gate
    "N":   (None,        True,  True),    # R3 after folding only -- isolates the fold
    "P":   ("perm",      True,  True),    # R3 after a permutation -- PLANTED NULL
    "H":   ("hadamard",  True,  True),    # R3 after block Hadamard
    "O":   ("orth",      True,  True),    # R3 after dense random orthogonal
}
ARMS = ["base", "XN", "XP", "XH", "XO", "Q", "N", "P", "H", "O"]
SMOKE_LAYERS = 4


def truncate(model, k):
    """Smoke mode drops layers from the MODEL, not from the transformation.

    A rotation is only exact if EVERY organ on the residual stream is rotated. Applying it to
    the first 4 of 28 layers and running all 28 would make arm X fail for a reason that has
    nothing to do with the algebra being tested. So the model itself is cut instead.
    """
    model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:k])
    model.config.num_hidden_layers = k
    return model


# ===================================================================== the basis
def build_R(kind, d):
    """The same constructions d2_basis.py measured kurtosis with, BUILT IN FLOAT64.

    d2 built its bases in float32 because it only needed them to compute a kurtosis. Here they
    multiply every weight in the model and arm X asks whether the result is exact, so a float32
    R is not good enough: 1/sqrt(512) is not representable, and a float32 block Hadamard has
    ||R^T R - I|| = 3.4e-08. That is 300x the tolerance arm X is judged at.

    Same family, same seed constant. Not the identical random draw -- d2 drew from one running
    generator across many matrices -- and the kurtosis claim is about the family, not the draw.
    The hadamard and permutation cases are cross-checked against d2's own functions below.
    """
    if kind is None:
        return None
    gen = torch.Generator().manual_seed(D2_SEED)
    if kind == "hadamard":
        from scipy.linalg import hadamard as _had
        b = 1
        while b * 2 <= d and d % (b * 2) == 0:
            b *= 2
        H = torch.from_numpy(_had(b).astype(np.float64)) / math.sqrt(b)
        R = torch.zeros(d, d, dtype=torch.float64)
        for i in range(d // b):
            R[i * b:(i + 1) * b, i * b:(i + 1) * b] = H
        # control: this must be d2's own matrix, to float32 precision
        assert float((R - hadamard_blocks(d).double()).abs().max()) < 1e-6, "hadamard drifted from d2"
        return R
    if kind == "orth":
        A = torch.randn(d, d, generator=gen, dtype=torch.float64)
        Q, Rq = torch.linalg.qr(A)
        return Q * torch.sign(torch.diagonal(Rq)).unsqueeze(0)          # Haar on O(d)
    if kind == "perm":
        R = permutation(d, gen).double()
        assert float((R.sum(0) - 1).abs().max()) == 0.0, "permutation is not a permutation"
        return R
    raise ValueError(kind)


def _chunked(W, R, left):
    """W @ R (left=False) or R.T @ W (left=True), in float64, in row chunks.

    Chunked because embed/head are [151936, 1536] and a float64 copy of the whole thing is
    1.9 GB. The chunking is over OUTPUT rows and changes nothing numerically.
    """
    if left:
        return (R.T @ W.double()).to(W.dtype)
    out = torch.empty_like(W)
    step = max(1, (1 << 24) // max(1, W.shape[1]))
    for i in range(0, W.shape[0], step):
        out[i:i + step] = (W[i:i + step].double() @ R).to(W.dtype)
    return out


# ===================================================================== fold + rotate
def fold_norms(model, n_layers=None):
    """Fold every RMSNorm gain into the linears that read it, and set the gain to 1.

    Returns the number of gains folded. Unties lm_head first if it shares storage with the
    embedding -- the final norm's gain belongs to the head only.
    """
    untied = False
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
        untied = True

    n = 0
    for li, lay in enumerate(model.model.layers):
        if n_layers is not None and li >= n_layers:
            break
        g = lay.input_layernorm.weight.data
        for organ in ("q_proj", "k_proj", "v_proj"):
            m = getattr(lay.self_attn, organ)
            m.weight.data = m.weight.data * g.unsqueeze(0)
        lay.input_layernorm.weight.data = torch.ones_like(g)
        g = lay.post_attention_layernorm.weight.data
        for organ in ("gate_proj", "up_proj"):
            m = getattr(lay.mlp, organ)
            m.weight.data = m.weight.data * g.unsqueeze(0)
        lay.post_attention_layernorm.weight.data = torch.ones_like(g)
        n += 2
    if n_layers is None:
        g = model.model.norm.weight.data
        model.lm_head.weight.data = model.lm_head.weight.data * g.unsqueeze(0)
        model.model.norm.weight.data = torch.ones_like(g)
        n += 1
    return n, untied


def rotate_model(model, R, n_layers=None):
    """Rotate the residual stream once. READERS get W @ R, WRITERS get R^T @ W."""
    E = model.model.embed_tokens
    E.weight.data = _chunked(E.weight.data, R, left=False)
    if n_layers is None:
        model.lm_head.weight.data = _chunked(model.lm_head.weight.data, R, left=False)
    n = 2
    for li, lay in enumerate(model.model.layers):
        if n_layers is not None and li >= n_layers:
            break
        for organ in ("q_proj", "k_proj", "v_proj"):
            m = getattr(lay.self_attn, organ)
            m.weight.data = _chunked(m.weight.data, R, left=False)
        for organ in ("gate_proj", "up_proj"):
            m = getattr(lay.mlp, organ)
            m.weight.data = _chunked(m.weight.data, R, left=False)
        lay.self_attn.o_proj.weight.data = _chunked(lay.self_attn.o_proj.weight.data, R, left=True)
        lay.mlp.down_proj.weight.data = _chunked(lay.mlp.down_proj.weight.data, R, left=True)
        n += 7
    return n


# ===================================================================== calibration capture
def capture(model, ids, n_layers=None):
    """Per-input activation RMS for every organ R3 touches, IN THE BASIS THE ARM RUNS IN.

    Inputs are shared inside a layer and the sharing is exact: q/k/v see the input-norm output,
    gate/up see the post-attention-norm output. Hook one of each and reuse, exactly as
    t2b_organs.py and qwen_export.py do, so the three cannot drift.
    """
    sums, cnts, hooks = {}, {}, []

    def mk(key):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            sums[key] = (x * x).sum(0) if key not in sums else sums[key] + (x * x).sum(0)
            cnts[key] = cnts.get(key, 0) + x.shape[0]
        return f

    for li, lay in enumerate(model.model.layers):
        if n_layers is not None and li >= n_layers:
            break
        for nm, mod in (("q_proj", lay.self_attn.q_proj), ("o_proj", lay.self_attn.o_proj),
                        ("gate_proj", lay.mlp.gate_proj), ("down_proj", lay.mlp.down_proj)):
            hooks.append(mod.register_forward_hook(mk((li, nm))))
    with torch.no_grad():
        for i in range(ids.shape[0]):
            model(ids[i:i + 1])
    for h in hooks:
        h.remove()

    rms = {k: torch.sqrt(s / cnts[k]).clamp_min(1e-8) for k, s in sums.items()}
    for (li, nm) in list(rms.keys()):
        if nm == "q_proj":
            rms[(li, "k_proj")] = rms[(li, "q_proj")]
            rms[(li, "v_proj")] = rms[(li, "q_proj")]
        elif nm == "gate_proj":
            rms[(li, "up_proj")] = rms[(li, "gate_proj")]
    return rms


def quantize_r3(model, act_rms, n_layers=None):
    stats = {"n": 0, "zero_frac": []}
    for li, lay in enumerate(model.model.layers):
        if n_layers is not None and li >= n_layers:
            break
        for organ in ORGANS:
            mod = getattr(lay.self_attn, organ, None) or getattr(lay.mlp, organ)
            q, a = T2.r3_actsearch(mod.weight.data, act_rms[(li, organ)])
            mod.weight.data = q * a
            stats["n"] += 1
            stats["zero_frac"].append(float((q == 0).float().mean()))
    return stats


# ===================================================================== the run
def main():
    t_all = time.time()
    out = {"brief": "docs/research/donor_adaptation/briefs/BRIEF_T3_ROTATION.md @ 7cdeca8",
           "rule": "R3 (T2's pre-registered winner)",
           "organs": list(ORGANS),
           "smoke": SMOKE, "arms": {},
           "thresholds_from_brief_s4": {
               "WINS_MARGIN": WINS_MARGIN, "MARGINAL_MARGIN": MARGINAL_MARGIN,
               "PLANTED_MARGIN": PLANTED_MARGIN, "X_EXACTNESS_TOL": X_EXACTNESS_TOL,
               "Q_REPLICATION_TOL": Q_REPLICATION_TOL,
               "T2_R3_standing": T2_R3_STANDING, "sigma_seed": SIGMA_SEED}}

    arms = [a for a in ARMS if not ONLY or a in ONLY or a == "base"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.REVISION)
    A, d = None, None

    ids, byts, meta = C.get_slice(tok, "heldout", 24 if not SMOKE else 2, 512, 1234)
    out["slice"] = {"ids_sha256": meta["ids_sha256"], "n_seq": int(ids.shape[0]),
                    "bytes": int(byts.sum())}
    ids_cal, _, meta_cal = C.get_slice(tok, "calib", 4 if SMOKE else N_CAL, SEQ_CAL, SEED_CAL)
    assert meta_cal["corpus_sha256"] != meta["corpus_sha256"], "calib and heldout must be disjoint"
    out["calib_slice"] = {"ids_sha256": meta_cal["ids_sha256"],
                          "tokens_T": int(ids_cal.shape[0] * ids_cal.shape[1])}
    print("slice %s (%d bytes)  calib T=%d%s"
          % (meta["ids_sha256"][:16], int(byts.sum()), out["calib_slice"]["tokens_T"],
             "  SMOKE: model truncated to %d layers" % SMOKE_LAYERS if SMOKE else ""),
          flush=True)

    R_cache = {}
    per_seq = {}
    for tag in arms:
        f = os.path.join(ARMDIR, tag + ".npy")
        if os.path.exists(f) and tag in out["arms"]:
            per_seq[tag] = np.load(f)
            continue
        t0 = time.time()
        model, _ = C.load_model()
        if SMOKE:
            truncate(model, SMOKE_LAYERS)
        if A is None:
            A = C.arch(model)
            out["arch"] = A
            d = A["d_model"]
        info = {"basis": None, "folded": 0, "rotated": 0, "n_substituted": 0}
        if tag != "base":
            kind, do_fold, do_quant = ARM_SPEC[tag]
            info["basis"] = kind
            if do_fold:
                nf, untied = fold_norms(model)
                info["folded"], info["untied"] = nf, untied
            if kind is not None:
                if kind not in R_cache:
                    R_cache[kind] = build_R(kind, d)
                    RR = R_cache[kind]
                    orth = float((RR.T @ RR - torch.eye(d, dtype=torch.float64)).abs().max())
                    info["R_orthogonality_maxdev"] = orth
                    assert orth < 1e-11, "R is not orthogonal: %g" % orth
                info["rotated"] = rotate_model(model, R_cache[kind])
            if do_quant:
                act = capture(model, ids_cal)
                st = quantize_r3(model, act)
                info["n_substituted"] = st["n"]
                info["mean_zero_frac"] = float(np.mean(st["zero_frac"]))
        val, ps = C.bpb(model, ids, byts, return_per_seq=True)
        del model
        np.save(f, ps)
        per_seq[tag] = ps
        info["bpb"], info["seconds"] = val, time.time() - t0
        out["arms"][tag] = info
        print("  %-3s BPB = %.9f   basis=%-9s fold=%-3d rot=%-4d quant=%-4d %.0fs"
              % (tag, val, str(info["basis"]), info["folded"], info["rotated"],
                 info["n_substituted"], info["seconds"]), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    # ---------------------------------------------------------------- deltas and contrasts
    w = byts.numpy().astype(np.float64)
    rng = np.random.default_rng(7)
    IDX = [rng.integers(0, len(w), len(w)) for _ in range(2000)]

    def paired(a, b="base"):
        if a not in per_seq or b not in per_seq:
            return None
        dd = per_seq[a] - per_seq[b]
        pt = float((dd * w).sum() / w.sum())
        bs = np.array([float((dd[i] * w[i]).sum() / w[i].sum()) for i in IDX])
        return {"delta_bpb": pt, "paired_se_sequence_bootstrap": float(bs.std(ddof=1)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    out["delta_vs_base"] = {a: paired(a) for a in arms if a != "base"}
    # the contrasts a Delta-against-base cannot support (T2 s4)
    out["contrasts"] = {k: paired(*k.split("_minus_")) for k in
                        ("H_minus_P", "O_minus_P", "H_minus_N", "O_minus_N",
                         "P_minus_N", "N_minus_Q", "H_minus_Q", "O_minus_Q")
                        if k.split("_minus_")[0] in per_seq and k.split("_minus_")[1] in per_seq}

    # ---------------------------------------------------------------- brief s4, mechanically
    D = {a: (out["delta_vs_base"].get(a) or {}).get("delta_bpb") for a in ARMS if a != "base"}
    dec = {"rule": "brief section 4, thresholds fixed before the run"}

    x_arms = {a: D[a] for a in ("XN", "XP", "XH", "XO") if D.get(a) is not None}
    dec["exactness_controls_X"] = {
        "deltas": x_arms, "tol": X_EXACTNESS_TOL,
        "passes": all(abs(v) < X_EXACTNESS_TOL for v in x_arms.values()) if x_arms else None}
    dec["Q_replication"] = {"measured": D.get("Q"), "T2_standing": T2_R3_STANDING,
                            "ok": (D.get("Q") is not None
                                   and abs(D["Q"] - T2_R3_STANDING) <= Q_REPLICATION_TOL)}
    # departure (a): P is an analytic null against N, not against Q -- the fold differs
    dec["P_is_analytic_null_of_N"] = {
        "P_minus_N": (out["contrasts"].get("P_minus_N") or {}).get("delta_bpb"),
        "tol": X_EXACTNESS_TOL,
        "passes": (abs((out["contrasts"].get("P_minus_N") or {}).get("delta_bpb", 9.9))
                   < X_EXACTNESS_TOL) if "P_minus_N" in out.get("contrasts", {}) else None,
        "why": ("a residual-stream permutation permutes a reader's columns and a writer's rows; "
                "R3 searches one scale per output row over (weight, act_rms) pairs, so both are "
                "invariant. A nonzero value here is a bug in the rotation machinery.")}

    q = D.get("Q")
    cand = {a: D[a] for a in ("H", "O") if D.get(a) is not None}
    best = min(cand.values()) if cand else None
    dec["q"], dec["best"], dec["best_arm"] = q, best, (min(cand, key=cand.get) if cand else None)

    if not dec["exactness_controls_X"]["passes"] or not dec["Q_replication"]["ok"]:
        dec["OUTCOME_LABEL"] = "VOID"
        dec["meaning"] = "the fold or the harness is wrong; report nothing else"
    elif best is None:
        dec["OUTCOME_LABEL"] = "INCOMPLETE"
        dec["meaning"] = "H and O were not both run"
    else:
        planted_ok = D.get("P") is not None and D["P"] >= best + PLANTED_MARGIN
        if best <= q - WINS_MARGIN and planted_ok:
            dec["OUTCOME_LABEL"] = "ROTATION-WINS"
        elif best <= q - WINS_MARGIN and not planted_ok:
            dec["OUTCOME_LABEL"] = "PERMUTATION-ARTEFACT"
        elif best <= q - MARGINAL_MARGIN and planted_ok:
            dec["OUTCOME_LABEL"] = "ROTATION-MARGINAL"
        elif best > q - MARGINAL_MARGIN:
            dec["OUTCOME_LABEL"] = "NULL"
        else:
            dec["OUTCOME_LABEL"] = "ROTATION-MARGINAL-PLANTED-FAILED"
        dec["planted_null_P_clears_margin"] = planted_ok

    out["decision"] = dec
    out["total_seconds"] = time.time() - t_all
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n%-4s %14s %12s %10s %9s" % ("arm", "BPB", "delta", "SE", "sigma"))
    for a in ARMS:
        if a == "base":
            print("%-4s %14.9f %12s" % (a, out["arms"]["base"]["bpb"], "[baseline]"))
        elif a in out["arms"]:
            dd = out["delta_vs_base"][a]
            print("%-4s %14.9f %+12.6f %10.6f %9.0f"
                  % (a, out["arms"][a]["bpb"], dd["delta_bpb"],
                     dd["paired_se_sequence_bootstrap"], dd["delta_bpb"] / SIGMA_SEED))
    print()
    for k, v in out["contrasts"].items():
        if v:
            print("  %-12s %+.6f +/- %.6f  ci95 [%+.4f, %+.4f]"
                  % (k, v["delta_bpb"], v["paired_se_sequence_bootstrap"],
                     v["ci95"][0], v["ci95"][1]))
    print("\nX controls pass: %s | Q replicates: %s | P is N's null: %s"
          % (dec["exactness_controls_X"]["passes"],
             ("n/a (smoke)" if SMOKE else dec["Q_replication"]["ok"]),
             dec["P_is_analytic_null_of_N"]["passes"]))
    print("q = %s   best = %s (%s)" % (q, best, dec["best_arm"]))
    print("OUTCOME: %s" % dec["OUTCOME_LABEL"])
    print("wrote %s  (%.0f s)" % (OUT, out["total_seconds"]))


if __name__ == "__main__":
    main()
