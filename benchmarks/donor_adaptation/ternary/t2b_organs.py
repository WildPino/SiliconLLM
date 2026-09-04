#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2b -- does T2's winning rule survive OUTSIDE the FFN?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_T2B_ORGAN_COVERAGE.md.
Section 4's thresholds are hard-coded below and the label is read off them mechanically.

T2 measured 84 tensors: 28 layers x {gate, up, down}. A model that RUNS also converts
q/k/v/o and, with --head-ternary, lm_head. Their cost has never been measured, and T2 s4
explicitly forbids extrapolating to them. So T2's +1.709 is a LOWER BOUND on what the
runnable model carries, not an estimate of it.

  base    nothing converted                      replication gate: 0.767594958
  I       identity through the FAH code path     INSTRUMENT CONTROL, must be bit-exact
  F       gate, up, down                         replication gate: must reproduce T2's R3
  FA      F + q, k, v, o                         the attention increment
  FAH     FA + lm_head                           what the exporter actually writes
  A       q, k, v, o only                        isolates attention
  H       lm_head only                           isolates the head

RULE. The brief says "T2's winner". Under T2's PRE-REGISTERED decision rule (min over R1-R4)
that is R3, and R3 is what runs here. R5 scored better (+1.260 vs +1.709) but is post-hoc and
gates nothing, and swapping the rule after seeing its number is exactly the drift this
programme keeps guarding against. R3 also needs only per-input activation RMS, not the
Hessians, so this run is minutes rather than an hour.

Env: D_THREADS (6), T2B_ONLY (comma list), T2B_SMOKE (1 = 2 seq / 4 layers)
"""
import json
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

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("T2B_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("T2B_ONLY", "").split(",") if x.strip()]

SIGMA_SEED = 0.005
BASELINE_STANDING = 0.7675949641196624
T2_R3_STANDING = 1.709372        # T2's winning Delta on the FFN, results/t2_rules.json

# ---------------------------------------------------------------- brief section 4, verbatim
UNIFORM_MAX_RATIO = 1.60         # derived from the organ shares, see the brief
HEAD_BOUND_FRACTION = 0.50
F_REPLICATION_TOL = 0.01

N_CAL, SEQ_CAL, SEED_CAL = 32, 512, 42424      # identical to T2 and D4

SUFFIX = "_smoke" if SMOKE else ""
OUT = os.path.join(C.RESULTS, "t2b_organs%s.json" % SUFFIX)
ARMDIR = os.path.join(C.RESULTS, "t2b_arms%s" % SUFFIX)
os.makedirs(ARMDIR, exist_ok=True)

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")
ARM_ORGANS = {
    "I":   ("identity", FFN + ATTN, True),
    "F":   ("R3", FFN, False),
    "FA":  ("R3", FFN + ATTN, False),
    "FAH": ("R3", FFN + ATTN, True),
    "A":   ("R3", ATTN, False),
    "H":   ("R3", (), True),
}
ARMS = ["base", "I", "F", "A", "H", "FA", "FAH"]


# ===================================================================== calibration capture
def capture(model, ids, n_layers=None):
    """Per-input activation RMS for every organ R3 may touch.

    Inputs are SHARED inside a layer and the sharing is exact, not approximate:
    q/k/v see the post-attention-norm output, gate/up see the post-MLP-norm output. Hooking
    one of each and reusing is what the exporter does, so the two cannot drift. lm_head sees
    the final norm output and is hooked directly.
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
    hooks.append(model.lm_head.register_forward_hook(mk(("head", "lm_head"))))

    with torch.no_grad():
        for i in range(ids.shape[0]):
            model(ids[i:i + 1])
    for h in hooks:
        h.remove()

    rms = {}
    for k, s in sums.items():
        rms[k] = torch.sqrt(s / cnts[k]).clamp_min(1e-8)
    for (li, nm) in list(rms.keys()):
        if nm == "q_proj":
            rms[(li, "k_proj")] = rms[(li, "q_proj")]
            rms[(li, "v_proj")] = rms[(li, "q_proj")]
        elif nm == "gate_proj":
            rms[(li, "up_proj")] = rms[(li, "gate_proj")]
    return rms


# ===================================================================== apply / restore
def apply_arm(model, tag, act_rms, n_layers=None):
    rule, organs, do_head = ARM_ORGANS[tag]
    saved, stats = [], {"n": 0, "zero_frac": []}

    def convert(mod, key):
        w = mod.weight.data
        saved.append((mod, w.clone()))
        if rule == "identity":
            new = w.clone()
            q = None
        else:
            q, a = T2.r3_actsearch(w, act_rms[key])
            new = q * a
        mod.weight.data = new
        stats["n"] += 1
        if q is not None:
            stats["zero_frac"].append(float((q == 0).float().mean()))

    for li, lay in enumerate(model.model.layers):
        if n_layers is not None and li >= n_layers:
            break
        for organ in organs:
            mod = getattr(lay.self_attn, organ, None) or getattr(lay.mlp, organ)
            convert(mod, (li, organ))
    if do_head:
        # tied embeddings: lm_head.weight IS the embedding, so converting it in place would
        # also destroy the input lookup. Untie first, exactly as qwen_export.py does.
        if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
            stats["untied"] = True
        convert(model.lm_head, ("head", "lm_head"))

    def restore():
        for mod, w in saved:
            mod.weight.data = w
    return restore, stats


def main():
    t_start = time.time()
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    n_layers = None
    if SMOKE:
        ids, byts, n_layers = ids[:2], byts[:2], 4
    else:
        assert meta["ids_sha256"] == \
            "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65", "wrong eval slice"

    arms = [a for a in ARMS if (not ONLY or a in ONLY or a == "base")]
    out = {"brief": "briefs/BRIEF_T2B_ORGAN_COVERAGE.md",
           "rule": "R3 (T2's pre-registered winner; R5 is post-hoc and not used here)",
           "slice": meta, "smoke": SMOKE,
           "thresholds_from_brief_s4": {"UNIFORM_MAX_RATIO": UNIFORM_MAX_RATIO,
                                        "HEAD_BOUND_FRACTION": HEAD_BOUND_FRACTION,
                                        "F_REPLICATION_TOL": F_REPLICATION_TOL,
                                        "T2_R3_standing": T2_R3_STANDING,
                                        "sigma_seed": SIGMA_SEED},
           "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT)).get("arms", {})
        except Exception:
            pass

    ids_cal, _, meta_cal = C.get_slice(tok, "calib", N_CAL, SEQ_CAL, SEED_CAL)
    if SMOKE:
        ids_cal = ids_cal[:2]
    assert meta_cal["corpus_sha256"] != meta["corpus_sha256"], \
        "calib and eval must be DIFFERENT corpus halves"
    out["calib_slice"] = meta_cal
    out["calib_eval_disjointness"] = {"calib_corpus_sha256": meta_cal["corpus_sha256"],
                                      "eval_corpus_sha256": meta["corpus_sha256"],
                                      "verified": True}
    out["calib_tokens_T"] = int(ids_cal.shape[0] * ids_cal.shape[1])
    print("== capturing activation RMS (T=%d, all organs incl. head) ==" % out["calib_tokens_T"],
          flush=True)
    t0 = time.time()
    act_rms = capture(model, ids_cal, n_layers)
    print("   done in %.0fs, %d organ keys" % (time.time() - t0, len(act_rms)), flush=True)

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
            restore, st = apply_arm(model, tag, act_rms, n_layers)
        val, ps = C.bpb(model, ids, byts, return_per_seq=True)
        restore()
        np.save(f, ps)
        per_seq[tag] = ps
        out["arms"][tag] = {"bpb": val, "n_substituted": st["n"],
                            "mean_zero_frac": (float(np.mean(st["zero_frac"]))
                                               if st["zero_frac"] else None),
                            "untied": st.get("untied", False),
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
    # the increments, paired BETWEEN arms -- a delta against base cannot compare two arms
    out["increments"] = {"FA_minus_F": paired("FA", "F"), "FAH_minus_FA": paired("FAH", "FA"),
                         "FAH_minus_F": paired("FAH", "F")}

    dec = {"rule": "brief section 4, thresholds fixed before the run"}
    dI = (out["delta_vs_base"].get("I") or {}).get("delta_bpb")
    dec["instrument_control_I"] = {"delta": dI, "passes": (dI is not None and abs(dI) < 1e-12)}
    dF = (out["delta_vs_base"].get("F") or {}).get("delta_bpb")
    dec["F_replication"] = {"measured": dF, "T2_standing": T2_R3_STANDING,
                            "ok": (dF is not None and abs(dF - T2_R3_STANDING) < F_REPLICATION_TOL)}
    dFAH = (out["delta_vs_base"].get("FAH") or {}).get("delta_bpb")
    dA = (out["delta_vs_base"].get("A") or {}).get("delta_bpb")
    dH = (out["delta_vs_base"].get("H") or {}).get("delta_bpb")
    dec["ratio_FAH_over_F"] = (dFAH / dF) if (dFAH is not None and dF) else None

    if not dec["instrument_control_I"]["passes"] or not dec["F_replication"]["ok"]:
        dec["OUTCOME_LABEL"] = "VOID"
        dec["meaning"] = ("arm I did not reproduce base exactly, or arm F missed T2's Delta by "
                          "more than %.2f; instrument broken, report nothing else"
                          % F_REPLICATION_TOL)
    elif dFAH is None or dA is None or dH is None:
        dec["OUTCOME_LABEL"] = "INCOMPLETE"
    elif dA > dF:
        dec["OUTCOME_LABEL"] = "ATTENTION-BOUND"
        dec["meaning"] = ("the attention projections are the precision-hungry organ, as P61 found "
                          "for the SSM; they come out of the ternary set and the ledger is "
                          "re-priced without them")
    elif dH > HEAD_BOUND_FRACTION * dF:
        dec["OUTCOME_LABEL"] = "HEAD-BOUND"
        dec["meaning"] = ("the head cannot be ternarized at this rule; R1 s4.1's 23.5 -> 38.0 "
                          "tok/s is withdrawn and the head needs its own treatment")
    elif dFAH <= UNIFORM_MAX_RATIO * dF:
        dec["OUTCOME_LABEL"] = "UNIFORM"
        dec["meaning"] = ("the rule transfers; the runnable model costs what the FFN measurement "
                          "said, and T2's number may be quoted for the whole model")
    else:
        dec["OUTCOME_LABEL"] = "DIFFUSE"
        dec["meaning"] = ("the damage is spread; no single organ is the culprit and the next "
                          "lever is width or healing, not organ selection")
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
    for k, v in out["increments"].items():
        if v:
            print("  increment %-14s %+.6f +/- %.6f  ci95 [%+.4f, %+.4f]"
                  % (k, v["delta_bpb"], v["paired_se_sequence_bootstrap"],
                     v["ci95"][0], v["ci95"][1]))
    print("  instrument control I passes:", dec["instrument_control_I"]["passes"])
    print("  F replicates T2's R3:", dec["F_replication"]["ok"],
          "(measured %s vs standing %s)" % (dF, T2_R3_STANDING))
    print("  ratio FAH/F: %s   (UNIFORM bar %.2f)" % (dec["ratio_FAH_over_F"], UNIFORM_MAX_RATIO))
    print("  OUTCOME:", dec["OUTCOME_LABEL"])
    print("  " + str(dec.get("meaning", ""))[:200])
    print("=" * 76)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
