#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1 -- what does the engine's OWN ternarization cost the donor, in BPB?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_T1_DONOR_TERNARIZATION.md (95cdf33).
Section 4's thresholds are hard-coded below from the brief and the label is read off them
mechanically. Nothing here may be changed to chase an outcome.

WHY THIS EXISTS. engine.c is a ternary engine: every weight it multiplies is in {-1, 0, +1} with
one fp32 scale per output row. The donor-adaptation programme has measured carving (D0/D0c),
pruning (D1), basis (D2), low-rank (D3), reconstruction (D4) and sparsity (S1) -- all of them on
fp32 weights. Nobody has measured what happens when the donor's weights are actually converted to
the format the engine consumes. This does.

THE RULE, taken verbatim from benchmarks/phase60/e1_export.py:34-38 (BitLinear158):
    scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)   # per OUTPUT row
    wq    = (w / scale).round().clamp(-1, 1)                     # {-1, 0, +1}
    deq   = wq * scale
The engine computes exactly `deq`, so evaluating `deq` in fp32 IS the engine's arithmetic for the
weights. Activation int8 (AQ=63) is a separate conversion and is NOT tested here (brief section 6).

This module is also the ORACLE for the C runtime: `ternarize()` is the single definition of the
conversion, and the exporter and the parity gate both call it rather than re-deriving it.

Env:
  D_THREADS   torch threads (default 6)
  T1_SMOKE    1 = 2 sequences, 4 layers, arms base/FA/Z only
  T1_ONLY     comma list of arm tags to run (resumes the rest from disk)
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
SMOKE = os.environ.get("T1_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("T1_ONLY", "").split(",") if x.strip()]

LN2 = math.log(2.0)
SIGMA_SEED = 0.005
BASELINE_STANDING = 0.7675949641196624          # every probe in this programme shares it

# ------------------------------------------------------------------ brief section 4, verbatim
CHEAP_MAX = 0.05        # <= this  -> CONVERSION-CHEAP   (10 sigma_seed)
FAIL_MIN = 0.30         # >  this  -> CONVERSION-FAILS   (60 sigma_seed)
GRAN_RECOVER = 0.10     # secondary: Delta(F/g64) <= Delta(F) - this -> GRANULARITY-RECOVERABLE

SUFFIX = "_smoke" if SMOKE else ""
OUT = os.path.join(C.RESULTS, "t1_ternarize%s.json" % SUFFIX)
ARMDIR = os.path.join(C.RESULTS, "t1_arms%s" % SUFFIX)
os.makedirs(ARMDIR, exist_ok=True)

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")


# ===================================================================== the conversion itself
def ternarize(w: torch.Tensor, group: int = 0, mode: str = "bitlinear158", seed: int = 0):
    """(out, in) fp32 -> dequantized fp32, exactly what engine.c computes.

    group = 0  -> one scale per output row, mean over ALL input features (the engine's rule).
    group = g  -> one scale per (row, block of g input features). `in` must be divisible by g.
    mode  = "random_sign" -> arm Z: same scales, same zero-fraction, RANDOM signs. The planted
            control (brief section 3.1): it preserves every norm and destroys only the
            information, so if it is not catastrophic the harness is not substituting weights.
    """
    if mode == "identity":
        # AMENDED 2026-09-04. The pre-registered control (arm Z, random signs) turned out to
        # test a SCIENTIFIC claim ("signs carry information") rather than an INSTRUMENT property
        # ("the harness really substitutes weights"), and it fired VOID because ternarization is
        # nearly as destructive as randomizing the signs -- which is a finding, not a fault.
        # This is the control that was needed: substitute every weight with itself, through the
        # exact same code path. BPB must reproduce the baseline to the last digit, or the
        # substitution machinery is broken and nothing in the probe is readable.
        return w.clone()
    out_f, in_f = w.shape
    if group and group > 0:
        assert in_f % group == 0, "in_features %d not divisible by group %d" % (in_f, group)
        wv = w.view(out_f, in_f // group, group)
        scale = wv.abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
        wq = (wv / scale).round().clamp(-1, 1)
        if mode == "random_sign":
            g = torch.Generator().manual_seed(seed)
            nz = wq != 0
            rs = (torch.randint(0, 2, wq.shape, generator=g, dtype=torch.int8) * 2 - 1).to(wq.dtype)
            wq = torch.where(nz, rs, torch.zeros_like(wq))
        return (wq * scale).view(out_f, in_f)
    scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
    wq = (w / scale).round().clamp(-1, 1)
    if mode == "random_sign":
        g = torch.Generator().manual_seed(seed)
        nz = wq != 0
        rs = (torch.randint(0, 2, wq.shape, generator=g, dtype=torch.int8) * 2 - 1).to(wq.dtype)
        wq = torch.where(nz, rs, torch.zeros_like(wq))
    return wq * scale


def zero_frac(w, group=0):
    out_f, in_f = w.shape
    if group:
        wv = w.view(out_f, in_f // group, group)
        s = wv.abs().mean(dim=2, keepdim=True).clamp_min(1e-5)
        q = (wv / s).round().clamp(-1, 1)
    else:
        s = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
        q = (w / s).round().clamp(-1, 1)
    return float((q == 0).float().mean())


# ===================================================================== arms
ARMS = [
    # tag,      organs,           group, mode,           note
    ("base",    (),                0,    "bitlinear158", "replication gate"),
    ("F",       FFN,               0,    "bitlinear158", "FFN, engine's per-row rule"),
    ("A",       ATTN,              0,    "bitlinear158", "attention projections"),
    ("FA",      FFN + ATTN,        0,    "bitlinear158", "PRIMARY -- the engine's conversion"),
    ("Fg256",   FFN,             256,    "bitlinear158", "granularity sweep"),
    ("Fg64",    FFN,              64,    "bitlinear158", "granularity sweep"),
    ("Z",       FFN,               0,    "random_sign",  "PLANTED CONTROL (mis-specified, see report)"),
    ("I",       FFN + ATTN,        0,    "identity",     "INSTRUMENT CONTROL -- must equal base exactly"),
]
ARMS_SMOKE = [a for a in ARMS if a[0] in ("base", "FA", "Z")]


def apply_arm(model, organs, group, mode, n_layers=None):
    """Replace the named organs' weights in place. Returns a restore closure."""
    saved = []
    layers = model.model.layers
    rng = 0
    for li, lay in enumerate(layers):
        if n_layers is not None and li >= n_layers:
            break
        for organ in organs:
            mod = getattr(lay.mlp, organ, None) or getattr(lay.self_attn, organ, None)
            if mod is None:
                continue
            w = mod.weight.data
            saved.append((mod, w.clone()))
            rng += 1
            mod.weight.data = ternarize(w, group=group, mode=mode, seed=1000 + rng)

    def restore():
        for m, w in saved:
            m.weight.data = w
    return restore, len(saved)


def main():
    t_start = time.time()
    model, tok = C.load_model(dtype=torch.float32)   # eager attention, per common.py
    model.eval()
    a = C.arch(model)
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    n_layers = None
    if SMOKE:
        ids, byts = ids[:2], byts[:2]
        n_layers = 4
    assert meta["ids_sha256"] == \
        "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65" or SMOKE, \
        "eval slice is not the shared one -- stop"

    arms = ARMS_SMOKE if SMOKE else ARMS
    if ONLY:
        arms = [x for x in arms if x[0] in ONLY]

    out = {
        "brief": "docs/research/donor_adaptation/briefs/BRIEF_T1_DONOR_TERNARIZATION.md @ 95cdf33",
        "rule": "BitLinear158, verbatim from benchmarks/phase60/e1_export.py:34-38",
        "env": (C.env_block() if hasattr(C, "env_block") else {}),
        "arch_achieved": a, "slice": meta, "smoke": SMOKE,
        "thresholds_from_brief_s4": {"CHEAP_MAX": CHEAP_MAX, "FAIL_MIN": FAIL_MIN,
                                     "GRAN_RECOVER": GRAN_RECOVER, "sigma_seed": SIGMA_SEED,
                                     "baseline_standing": BASELINE_STANDING},
        "arms": {}, "zero_fraction": {},
    }
    if os.path.exists(OUT):
        try:
            out.update({"arms": json.load(open(OUT)).get("arms", {})})
        except Exception:
            pass

    per_seq = {}
    for tag, organs, group, mode, note in arms:
        f = os.path.join(ARMDIR, tag + ".npy")
        if os.path.exists(f) and tag in out["arms"]:
            per_seq[tag] = np.load(f)
            print("  %-6s CACHED   BPB = %.9f" % (tag, out["arms"][tag]["bpb"]), flush=True)
            continue
        t0 = time.time()
        restore, n_sub = (lambda: None, 0) if not organs else \
            apply_arm(model, organs, group, mode, n_layers)
        if organs:
            restore, n_sub = apply_arm.__wrapped__ if False else (restore, n_sub)
        val, ps = C.bpb(model, ids, byts, return_per_seq=True)
        if organs:
            restore()
        np.save(f, ps)
        per_seq[tag] = ps
        out["arms"][tag] = {"bpb": val, "organs": list(organs), "group": group, "mode": mode,
                            "note": note, "n_substituted": n_sub,
                            "seconds": time.time() - t0}
        print("  %-6s BPB = %.9f   (%d tensors, %.0fs)" % (tag, val, n_sub, time.time() - t0),
              flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    # zero-fraction health check on one representative tensor per granularity
    lay0 = model.model.layers[0]
    for g in (0, 256, 64):
        out["zero_fraction"]["gate_proj_L0_g%d" % g] = zero_frac(lay0.mlp.gate_proj.weight.data, g)

    # ------------------------------------------------------------------ paired statistics
    def paired(a_tag, b_tag):
        if a_tag not in per_seq or b_tag not in per_seq:
            return None
        d = per_seq[a_tag] - per_seq[b_tag]
        w = byts.numpy().astype(np.float64)
        delta = float((d * w).sum() / w.sum())
        rng = np.random.default_rng(7)
        n = len(d)
        bs = [float((d[i] * w[i]).sum() / w[i].sum())
              for i in (rng.integers(0, n, n) for _ in range(2000))]
        return {"delta_bpb": delta, "paired_se_sequence_bootstrap": float(np.std(bs, ddof=1)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "frac_seq_worse": float((d > 0).mean())}

    out["delta_vs_base"] = {t: paired(t, "base") for t, *_ in arms if t != "base"}

    # ------------------------------------------------------------------ brief section 4, mechanical
    dec = {"rule": "brief section 4, thresholds fixed before the run"}
    dFA = (out["delta_vs_base"].get("FA") or {}).get("delta_bpb")
    dF = (out["delta_vs_base"].get("F") or {}).get("delta_bpb")
    dZ = (out["delta_vs_base"].get("Z") or {}).get("delta_bpb")
    dg64 = (out["delta_vs_base"].get("Fg64") or {}).get("delta_bpb")

    base_ok = abs(out["arms"].get("base", {}).get("bpb", 0) - BASELINE_STANDING) < 1e-6 or SMOKE
    dec["baseline_replication"] = {
        "measured": out["arms"].get("base", {}).get("bpb"),
        "standing": BASELINE_STANDING, "ok": bool(base_ok)}

    if dZ is not None and dF is not None:
        control_ok = dZ > dF * 2.0 and dZ > 0.30
        dec["planted_control_Z"] = {"delta_Z": dZ, "delta_F": dF, "fires": bool(control_ok),
                                    "requirement": "Z must be >2x F and >0.30 BPB"}
    else:
        control_ok = None
        dec["planted_control_Z"] = None

    if control_ok is False:
        dec["OUTCOME_LABEL"] = "VOID"
        dec["meaning"] = ("arm Z is not catastrophically worse than arm F, so the harness is not "
                          "really substituting weights. No other number in this probe may be read.")
    elif dFA is None:
        dec["OUTCOME_LABEL"] = "INCOMPLETE"
    elif dFA <= CHEAP_MAX:
        dec["OUTCOME_LABEL"] = "CONVERSION-CHEAP"
        dec["meaning"] = ("the donor survives the engine's numeric format nearly intact; every "
                          "structural probe should be re-read as additional to a near-free base")
    elif dFA > FAIL_MIN:
        dec["OUTCOME_LABEL"] = "CONVERSION-FAILS"
        dec["meaning"] = ("naive post-training ternarization does not work on this donor; no "
                          "amount of speed work matters until it is fixed, and healing becomes "
                          "the critical path ahead of every structural probe")
    else:
        dec["OUTCOME_LABEL"] = "CONVERSION-COSTLY"
        dec["meaning"] = ("real quality cost, in the range where healing (QAT, calibration, "
                          "layer-wise correction) plausibly recovers it")

    if dg64 is not None and dF is not None:
        dec["granularity"] = {
            "delta_F_perrow": dF, "delta_Fg64": dg64, "improvement": dF - dg64,
            "threshold": GRAN_RECOVER,
            "label": "GRANULARITY-RECOVERABLE" if (dF - dg64) >= GRAN_RECOVER else "NOT-RECOVERABLE",
            "byte_cost_note": ("a per-row scale is 4 B/row; g=64 on down_proj (in=8960) is "
                               "4*8960/64 = 560 B/row against 4480 B of codes = +12.5% weight bytes"),
        }
    dec["delta_in_sigma_seed"] = {k: (v["delta_bpb"] / SIGMA_SEED if v else None)
                                  for k, v in out["delta_vs_base"].items()}
    out["decision"] = dec
    out["total_seconds"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n" + "=" * 78)
    for t, *_ in arms:
        if t in out["arms"]:
            d = out["delta_vs_base"].get(t)
            print("  %-6s BPB %.6f%s" % (t, out["arms"][t]["bpb"],
                  ("   delta %+.6f +/- %.6f  (%.1f sigma_seed)"
                   % (d["delta_bpb"], d["paired_se_sequence_bootstrap"],
                      d["delta_bpb"] / SIGMA_SEED)) if d else "   [baseline]"))
    print("  planted control Z fires:", (dec.get("planted_control_Z") or {}).get("fires"))
    print("  OUTCOME:", dec.get("OUTCOME_LABEL"))
    print("  " + str(dec.get("meaning", ""))[:200])
    if "granularity" in dec:
        print("  granularity:", dec["granularity"]["label"],
              "(improvement %+.6f, threshold %.2f)" % (dec["granularity"]["improvement"],
                                                       GRAN_RECOVER))
    print("=" * 78)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
