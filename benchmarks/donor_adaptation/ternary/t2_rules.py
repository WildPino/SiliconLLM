#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 -- was T1's +4.74 BPB the ternary FORMAT, or one naive RULE for reaching it?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_T2_TERNARIZATION_RULE.md (1286faf).
Section 4's thresholds are hard-coded below and the label is read off them mechanically.

EVERY arm lands in the SAME format donor_engine.c consumes -- codes in {-1,0,+1} with one fp32
scale per output row -- so every arm is executable by the runtime unchanged. Only the choice of
codes and scale differs. A rule that needed a different format would be out of scope, because
the format is the architecture.

  R0  BitLinear158    alpha = mean|w|, round-to-nearest            <- T1's rule, replication gate
  R1  TWN             Delta = 0.7*mean|w|, alpha = mean(|w|>Delta)
  R2  alpha-search    per row, grid over Delta/alpha minimising ||w - alpha*q||^2
  R3  act-weighted    same search, weighted by per-input activation RMS
  R4  GPTQ / OBQ      column-by-column with inverse-Hessian error compensation   (--r4)
  Z   random signs    descriptive reference only, gates nothing
  I   identity        INSTRUMENT CONTROL -- must reproduce base bit-exactly

Env: D_THREADS (6), T2_ONLY (comma list of arms), T2_SMOKE (1 = 2 seq / 4 layers)
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
import common as C  # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("T2_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("T2_ONLY", "").split(",") if x.strip()]
WANT_R4 = "--r4" in sys.argv or "R4" in ONLY

SIGMA_SEED = 0.005
BASELINE_STANDING = 0.7675949641196624
R0_STANDING = 3.309099          # T1's Delta for the FFN arm, results/t1_ternarize.json
FFN = ("gate_proj", "up_proj", "down_proj")

# ---------------------------------------------------------------- brief section 4, verbatim
RULE_BOUND_MAX = 0.50
HELPS_MIN_RECOVERY = 1.00
MARGINAL_MIN_RECOVERY = 0.20

SUFFIX = "_smoke" if SMOKE else ""
OUT = os.path.join(C.RESULTS, "t2_rules%s.json" % SUFFIX)
ARMDIR = os.path.join(C.RESULTS, "t2_arms%s" % SUFFIX)
os.makedirs(ARMDIR, exist_ok=True)

# calibration slice: D4's registered operating point, same disjoint corpus half
N_CAL, SEQ_CAL, SEED_CAL = 32, 512, 42424


# ===================================================================== the rules
def _codes_from(w, alpha, delta):
    """q in {-1,0,+1}: keep sign where |w| > delta, else zero. alpha is the per-row scale."""
    q = torch.zeros_like(w)
    q[w > delta] = 1.0
    q[w < -delta] = -1.0
    return q


def r0_bitlinear(w):
    a = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
    q = (w / a).round().clamp(-1, 1)
    return q, a


def r1_twn(w):
    """Ternary Weight Networks: Delta = 0.7*E|w|, alpha = E(|w| : |w|>Delta)."""
    m = w.abs().mean(dim=1, keepdim=True)
    d = 0.7 * m
    q = _codes_from(w, None, d)
    kept = (q != 0).float()
    n = kept.sum(dim=1, keepdim=True).clamp_min(1.0)
    a = ((w.abs() * kept).sum(dim=1, keepdim=True) / n).clamp_min(1e-5)
    return q, a


def _search(w, d_grid, weight=None):
    """Per row, pick the threshold minimising ||(w - alpha q) * weight||^2 with the optimal
    alpha for that threshold. `weight` is a per-INPUT vector (broadcast over rows) or None."""
    m = w.abs().mean(dim=1, keepdim=True)
    best_err = None
    best_q = None
    best_a = None
    ww = None if weight is None else weight.view(1, -1)
    for f in d_grid:
        d = f * m
        q = _codes_from(w, None, d)
        kept = (q != 0).float()
        if ww is None:
            num = (w * q).sum(dim=1, keepdim=True)
            den = kept.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            num = (w * q * ww * ww).sum(dim=1, keepdim=True)
            den = (kept * ww * ww).sum(dim=1, keepdim=True).clamp_min(1e-12)
        a = (num / den).clamp_min(1e-5)
        r = w - a * q
        err = ((r * r) if ww is None else (r * r * ww * ww)).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_q, best_a = err, q, a
        else:
            take = (err < best_err)
            best_err = torch.where(take, err, best_err)
            best_a = torch.where(take, a, best_a)
            best_q = torch.where(take, q, best_q)
    return best_q, best_a


D_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


def r2_search(w):
    return _search(w, D_GRID, None)


def r3_actsearch(w, act_rms):
    return _search(w, D_GRID, act_rms)


def r4_gptq(w, H, percdamp=0.01, blocksize=128):
    """GPTQ / OBQ: quantize column by column, pushing each column's rounding error onto the
    columns not yet quantized through the inverse Hessian of the layer inputs.

    w: [out, in] fp32.  H: [in, in] fp64 = X^T X over the calibration set.
    The per-row scale is fixed UP FRONT from the original weights (standard GPTQ): letting it
    float as W is updated would make the quantizer chase its own error.
    """
    W = w.clone().double()
    n_out, n_in = W.shape
    a = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5).double()      # fixed scale, R0's

    Hd = H.clone()
    dead = torch.diag(Hd) == 0
    Hd[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(Hd))
    Hd += torch.eye(n_in, dtype=Hd.dtype) * damp

    # Hinv upper-Cholesky, the standard GPTQ construction
    Hc = torch.linalg.cholesky(Hd)
    Hinv = torch.cholesky_inverse(Hc)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    for i1 in range(0, n_in, blocksize):
        i2 = min(i1 + blocksize, n_in)
        Wb = W[:, i1:i2].clone()
        Qb = torch.zeros_like(Wb)
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            wj = Wb[:, j]
            d = Hb[j, j]
            qj = (wj.unsqueeze(1) / a).round().clamp(-1, 1).squeeze(1)
            Qb[:, j] = qj
            err = (wj - qj * a.squeeze(1)) / d
            Wb[:, j:] -= err.unsqueeze(1) * Hb[j, j:].unsqueeze(0)
            Eb[:, j] = err
        Q[:, i1:i2] = Qb
        W[:, i2:] -= Eb @ Hinv[i1:i2, i2:]
    return Q.float(), a.float()


# ===================================================================== apply / restore
def apply_rule(model, rule, n_layers=None, act_rms=None, H_by=None):
    saved = []
    stats = {"n": 0, "zero_frac": []}
    for li, lay in enumerate(model.model.layers):
        if n_layers is not None and li >= n_layers:
            break
        for organ in FFN:
            mod = getattr(lay.mlp, organ)
            w = mod.weight.data
            saved.append((mod, w.clone()))
            if rule == "I":
                new = w.clone()
                q = None
            elif rule == "R0":
                q, a = r0_bitlinear(w); new = q * a
            elif rule == "R1":
                q, a = r1_twn(w); new = q * a
            elif rule == "R2":
                q, a = r2_search(w); new = q * a
            elif rule == "R3":
                q, a = r3_actsearch(w, act_rms[(li, organ)]); new = q * a
            elif rule == "R4":
                q, a = r4_gptq(w, H_by[(li, organ)]); new = q * a
            elif rule == "Z":
                q, a = r0_bitlinear(w)
                g = torch.Generator().manual_seed(1000 + stats["n"])
                nz = q != 0
                rs = (torch.randint(0, 2, q.shape, generator=g, dtype=torch.int8) * 2 - 1).float()
                q = torch.where(nz, rs, torch.zeros_like(q)); new = q * a
            else:
                raise ValueError(rule)
            mod.weight.data = new
            stats["n"] += 1
            if q is not None:
                stats["zero_frac"].append(float((q == 0).float().mean()))

    def restore():
        for m, w in saved:
            m.weight.data = w
    return restore, stats


# ===================================================================== calibration capture
def capture(model, ids, want_H, n_layers=None):
    """One forward pass over the calibration slice, accumulating per-(layer, organ):
       - act_rms: sqrt(mean(x^2)) per input feature      (R3)
       - H = X^T X                                       (R4, only if want_H)
    gate_proj and up_proj share their input, so it is captured once and reused.
    H is accumulated in fp32 and promoted per-layer inside the solve: fp64 for down_proj
    (in=8960) across 28 layers would be 18 GB, which is not a memory budget worth spending
    to carry precision the Cholesky re-establishes anyway.
    """
    sums, cnts, Hs = {}, {}, {}
    hooks = []

    def mk(key, keepH):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            s = sums.get(key)
            sums[key] = (x * x).sum(0) if s is None else s + (x * x).sum(0)
            cnts[key] = cnts.get(key, 0) + x.shape[0]
            if keepH:
                h = Hs.get(key)
                Hs[key] = (x.T @ x) if h is None else h + (x.T @ x)
        return f

    for li, lay in enumerate(model.model.layers):
        if n_layers is not None and li >= n_layers:
            break
        hooks.append(lay.mlp.gate_proj.register_forward_hook(mk((li, "gate_proj"), want_H)))
        hooks.append(lay.mlp.down_proj.register_forward_hook(mk((li, "down_proj"), want_H)))
    with torch.no_grad():
        for i in range(ids.shape[0]):
            model(ids[i:i + 1])
    for h in hooks:
        h.remove()

    act_rms, H_by = {}, {}
    for (li, organ), s in sums.items():
        rms = torch.sqrt(s / cnts[(li, organ)]).clamp_min(1e-8)
        act_rms[(li, organ)] = rms
        if organ == "gate_proj":
            act_rms[(li, "up_proj")] = rms                    # identical input
    for k, h in Hs.items():
        H_by[k] = h.double()
        if k[1] == "gate_proj":
            H_by[(k[0], "up_proj")] = H_by[k]
    return act_rms, H_by


ARMS = ["base", "I", "R0", "R1", "R2", "R3", "Z"] + (["R4"] if WANT_R4 else [])


def main():
    t_start = time.time()
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    arch = C.arch(model)
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    n_layers = None
    if SMOKE:
        ids, byts, n_layers = ids[:2], byts[:2], 4
    if not SMOKE:
        assert meta["ids_sha256"] == \
            "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65", "wrong eval slice"

    arms = [a for a in ARMS if (not ONLY or a in ONLY or a == "base")]
    need_cal = any(a in ("R3", "R4") for a in arms)

    out = {"brief": "briefs/BRIEF_T2_TERNARIZATION_RULE.md @ 1286faf",
           "arch_achieved": arch, "slice": meta, "smoke": SMOKE,
           "thresholds_from_brief_s4": {"RULE_BOUND_MAX": RULE_BOUND_MAX,
                                        "HELPS_MIN_RECOVERY": HELPS_MIN_RECOVERY,
                                        "MARGINAL_MIN_RECOVERY": MARGINAL_MIN_RECOVERY,
                                        "R0_standing": R0_STANDING, "sigma_seed": SIGMA_SEED},
           "arms": {}, "d_grid": D_GRID}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT)).get("arms", {})
        except Exception:
            pass

    act_rms = H_by = None
    if need_cal:
        ids_cal, _, meta_cal = C.get_slice(tok, "calib", N_CAL, SEQ_CAL, SEED_CAL)
        if SMOKE:
            ids_cal = ids_cal[:2]
        assert meta_cal["corpus_sha256"] != meta["corpus_sha256"], \
            "calib and eval must be DIFFERENT corpus halves"
        out["calib_slice"] = meta_cal
        out["calib_eval_disjointness"] = {
            "calib_corpus_sha256": meta_cal["corpus_sha256"],
            "eval_corpus_sha256": meta["corpus_sha256"], "verified": True}
        out["calib_tokens_T"] = int(ids_cal.shape[0] * ids_cal.shape[1])
        print("== capturing calibration activations (T=%d, H=%s) ==" %
              (out["calib_tokens_T"], WANT_R4), flush=True)
        t0 = time.time()
        act_rms, H_by = capture(model, ids_cal, WANT_R4, n_layers)
        print("   done in %.0fs" % (time.time() - t0), flush=True)

    per_seq = {}
    for tag in arms:
        f = os.path.join(ARMDIR, tag + ".npy")
        if os.path.exists(f) and tag in out["arms"]:
            per_seq[tag] = np.load(f)
            print("  %-4s CACHED  BPB = %.9f" % (tag, out["arms"][tag]["bpb"]), flush=True)
            continue
        t0 = time.time()
        if tag == "base":
            restore, st = (lambda: None), {"n": 0, "zero_frac": []}
        else:
            restore, st = apply_rule(model, tag, n_layers, act_rms, H_by)
        val, ps = C.bpb(model, ids, byts, return_per_seq=True)
        restore()
        np.save(f, ps)
        per_seq[tag] = ps
        out["arms"][tag] = {"bpb": val, "n_substituted": st["n"],
                            "mean_zero_frac": (float(np.mean(st["zero_frac"]))
                                               if st["zero_frac"] else None),
                            "seconds": time.time() - t0}
        print("  %-4s BPB = %.9f   (%d tensors, zero_frac %.4f, %.0fs)"
              % (tag, val, st["n"],
                 float(np.mean(st["zero_frac"])) if st["zero_frac"] else 0.0,
                 time.time() - t0), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    w = byts.numpy().astype(np.float64)

    def paired(a, b="base"):
        if a not in per_seq or b not in per_seq:
            return None
        d = per_seq[a] - per_seq[b]
        delta = float((d * w).sum() / w.sum())
        rng = np.random.default_rng(7)
        bs = [float((d[i] * w[i]).sum() / w[i].sum())
              for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
        return {"delta_bpb": delta, "paired_se_sequence_bootstrap": float(np.std(bs, ddof=1)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    out["delta_vs_base"] = {a: paired(a) for a in arms if a != "base"}

    dec = {"rule": "brief section 4, thresholds fixed before the run"}
    dI = (out["delta_vs_base"].get("I") or {}).get("delta_bpb")
    dec["instrument_control_I"] = {"delta": dI, "passes": (dI is not None and abs(dI) < 1e-12)}
    dR0 = (out["delta_vs_base"].get("R0") or {}).get("delta_bpb")
    dec["R0_replication"] = {"measured": dR0, "T1_standing": R0_STANDING,
                             "ok": (dR0 is not None and abs(dR0 - R0_STANDING) < 0.01)}
    cand = {k: v["delta_bpb"] for k, v in out["delta_vs_base"].items()
            if v and k in ("R1", "R2", "R3", "R4")}
    best_tag = min(cand, key=cand.get) if cand else None
    best = cand.get(best_tag) if best_tag else None
    dec["best_rule"] = best_tag
    dec["best_delta"] = best
    dec["recovery_vs_R0"] = (dR0 - best) if (best is not None and dR0 is not None) else None

    if not dec["instrument_control_I"]["passes"]:
        dec["OUTCOME_LABEL"] = "VOID"
        dec["meaning"] = "identity substitution did not reproduce the baseline; instrument broken"
    elif best is None:
        dec["OUTCOME_LABEL"] = "INCOMPLETE"
    elif best <= RULE_BOUND_MAX:
        dec["OUTCOME_LABEL"] = "RULE-BOUND"
        dec["meaning"] = ("the format is fine and T1's number was the rule; the conversion is "
                          "essentially solved without training")
    elif (dR0 - best) >= HELPS_MIN_RECOVERY:
        dec["OUTCOME_LABEL"] = "RULE-HELPS"
        dec["meaning"] = ("a better rule recovers a large part but not enough; healing is still "
                          "needed but starts from a much better place")
    elif (dR0 - best) >= MARGINAL_MIN_RECOVERY:
        dec["OUTCOME_LABEL"] = "RULE-MARGINAL"
        dec["meaning"] = "rules move it a little; the format is the wall and QAT is the only route"
    else:
        dec["OUTCOME_LABEL"] = "FORMAT-BOUND"
        dec["meaning"] = ("no rule helps; T1's negative hardens into a statement about ternary "
                          "itself at this width, and the next question is width, not rule")
    out["decision"] = dec
    out["total_seconds"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n" + "=" * 76)
    for a in arms:
        if a in out["arms"]:
            d = out["delta_vs_base"].get(a)
            print("  %-4s BPB %.6f%s" % (a, out["arms"][a]["bpb"],
                  ("   delta %+.6f +/- %.6f  (%.0f sigma_seed)"
                   % (d["delta_bpb"], d["paired_se_sequence_bootstrap"],
                      d["delta_bpb"] / SIGMA_SEED)) if d else "   [baseline]"))
    print("  instrument control I passes:", dec["instrument_control_I"]["passes"])
    print("  R0 replicates T1:", dec["R0_replication"]["ok"])
    print("  best rule: %s  delta %s  recovery vs R0 %s"
          % (best_tag, best, dec["recovery_vs_R0"]))
    print("  OUTCOME:", dec["OUTCOME_LABEL"])
    print("  " + str(dec.get("meaning", ""))[:200])
    print("=" * 76)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
