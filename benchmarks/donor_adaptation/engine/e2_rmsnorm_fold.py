#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 -- the RMSNorm fold, in the exporter, measured through donor_engine.c.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E2_RMSNORM_FOLD.md, pre-registered at
338d187, amended at edbe956 (s3.1) BEFORE any run.

WHAT THIS ASKS.  T3 measured the fold at -0.220001 +/- 0.052861 BPB as a CONTROL inside a run
whose headline question died, in a transformers module in memory, by the runner that produced
it.  This asks whether it survives the trip to the artifact: qwen_export.py --fold, scored
through the engine, on weights Gate A proves identical.

THE ARMS, one variable each (brief s3 + s3.1):

    F32   fold none    fp32     the reference
    XF    fold layers  fp32     exactness control: the 2L fold is a re-parameterization
    XA    fold all     fp32     ... and so is the final gain, which also unties the head
    TQ    fold none    packed   the control the fold is measured against
    NL    fold layers  packed   THE FOLD, as T3 measured it minus the neutral part
    TQH   fold none    packed + head-ternary   the runnable model, unfolded
    NLH   fold layers  packed + head-ternary   the fold on the runnable model
    NAH   fold all     packed + head-ternary   the only arm that can answer brief s2

WHY XA REPLACED THE PRE-REGISTERED ARM NA.  Brief s3 asked for "fold all, packed, fp32 head".
That cannot exist: a quantized file carries ONE quant flag and donor_engine.c:397 reads the
untied head with it, so an untied head in a packed file is necessarily packed.  The exporter
refuses the combination rather than silently ternarizing a head nobody asked to ternarize.
NA's job was an algebra null -- with an fp32 head, folding model.norm into lm_head leaves the
196 quantized tensors and their calibration untouched -- and that check is SHARPER at fp32,
where it must hold exactly rather than approximately.  Amendment written into the brief.

WHAT THIS RUNNER DOES NOT RE-DERIVE.  layout/read_header/read_codes/build_reference/run_engine
are imported from e1_bpb_through_engine, and the fold itself from t3_rotation.  Three probes
now share one definition of the fold and one definition of the file layout; a second copy of
either is how two arms drift apart without anyone noticing.

STATISTICS.  Paired between-arm bootstrap on the same sequence draws, 2000 resamples, seed 7,
byte-weighted (brief s4).  A delta against a baseline cannot compare two arms -- T2 s4 is where
that was paid for.  The per-sequence PyTorch nats are exact; the per-sequence ENGINE nats are
recovered from a running mean printed at 6 decimals and are a diagnostic only, so every
bootstrapped contrast below is computed on the PyTorch side, where the numbers are exact, and
the engine's job is to agree with it arm by arm.

    python e2_rmsnorm_fold.py --arms F32,XF,XA,TQ,NL,TQH,NLH,NAH --seqs 24 --calib-seqs 32
    python e2_rmsnorm_fold.py --model Qwen/Qwen2.5-1.5B \
      --revision 8faed761d45a263340a0528343f099c05c9a4323 \
      --arms TQ,NL,TQH,NLH,NAH --seqs 24 --calib-seqs 32
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ternary")))
sys.path.insert(0, HERE)
import common as C                    # noqa: E402
import e1_bpb_through_engine as E1    # noqa: E402

EXE = E1.EXE
THREADS = E1.THREADS
SIGMA_SEED = 0.005

# ---- thresholds, brief s5, fixed before the run ------------------------------------------
GATE_F_FP32_TOL = 0.002       # |XF - F32| and |XA - F32|; Gate B's tolerance, 0.4 sigma_seed
FOLD_CONFIRMED = -0.10        # NL - TQ at or below this, ci95 excluding 0
FOLD_SHRINKS = -0.02          # ... between the two, ci95 excluding 0
BOOT_N = 2000
BOOT_SEED = 7

BASE_1P5B = 0.7675949584171732
T3_ARM_N = 2.4966555196149862   # T3 arm N, 196 tensors, density/results/t3_rotation.json
T3_ARM_Q = 2.7166563086672921   # T3 arm Q == T2b arm FA, same file / t2b_organs.json
T3_FOLD = -0.220001             # T3's N - Q, paired SE 0.052861

#            arm -> (fold, quant, head_ternary)
ARMS = {
    "F32": ("none",   "fp32",   False),
    "XF":  ("layers", "fp32",   False),
    "XA":  ("all",    "fp32",   False),
    "TQ":  ("none",   "packed", False),
    "NL":  ("layers", "packed", False),
    "TQH": ("none",   "packed", True),
    "NLH": ("layers", "packed", True),
    "NAH": ("all",    "packed", True),
}


def paired_bootstrap(nats_a, nats_b, byts, n=BOOT_N, seed=BOOT_SEED):
    """ci95 of BPB(a) - BPB(b) resampling SEQUENCES, byte-weighted, paired.

    The arms are correlated across sequences -- they are the same text -- so the resample has
    to draw the same sequence indices for both arms.  Returns (point, se, lo, hi).
    """
    rng = np.random.default_rng(seed)
    k = len(byts)
    ln2 = math.log(2.0)
    point = (nats_a.sum() - nats_b.sum()) / (ln2 * byts.sum())
    draws = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        draws[i] = (nats_a[idx].sum() - nats_b[idx].sum()) / (ln2 * byts[idx].sum())
    return float(point), float(draws.std(ddof=1)), \
        float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arms", default="F32,XF,XA,TQ,NL,TQH,NLH,NAH")
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--calib-seqs", type=int, default=32)
    ap.add_argument("--workdir", default="D:/_ktmp/e2")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for x in arms:
        if x not in ARMS:
            sys.exit("unknown arm %r; known: %s" % (x, ",".join(ARMS)))
    os.makedirs(a.workdir, exist_ok=True)
    if not os.path.exists(EXE):
        sys.exit("no donor_engine.exe -- build it first")
    tag = a.model.split("/")[-1].replace(".", "").lower()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    ids_all, byts_all, meta = C.get_slice(tk, "heldout", 24, 512, 1234)
    assert meta["ids_sha256"].startswith("a1a48dc9"), meta["ids_sha256"]
    assert int(meta["total_scored_bytes"]) == 51870, meta["total_scored_bytes"]
    for i in range(ids_all.shape[0]):
        assert len(tk.decode(ids_all[i, 1:].tolist()).encode("utf-8")) == int(byts_all[i]), \
            "byte count differs under this donor's tokenizer at seq %d" % i

    ids, byts = ids_all[: a.seqs], byts_all[: a.seqs]
    SL = ids.shape[1]
    nb = float(byts.sum())
    bnp = byts.numpy().astype(np.float64)
    print("E2  %s  arms=%s  seqs=%d/%d  bytes=%d  slice %s"
          % (a.model, ",".join(arms), ids.shape[0], ids_all.shape[0], int(nb),
             meta["ids_sha256"][:16]), flush=True)

    out = {"brief": "docs/research/donor_adaptation/briefs/BRIEF_E2_RMSNORM_FOLD.md "
                    "@ 338d187, amended s3.1 @ edbe956",
           "model": a.model, "revision": a.revision, "smoke": bool(a.smoke),
           "slice": {"ids_sha256": meta["ids_sha256"], "n_seq": int(ids.shape[0]),
                     "seq_len": SL, "bytes": int(nb),
                     "subset_of_pinned_24": int(ids.shape[0]) != 24},
           "calib_seqs": a.calib_seqs, "threads": THREADS,
           "thresholds_from_brief_s5": {
               "GATE_F_FP32_TOL": GATE_F_FP32_TOL, "FOLD_CONFIRMED": FOLD_CONFIRMED,
               "FOLD_SHRINKS": FOLD_SHRINKS, "sigma_seed": SIGMA_SEED,
               "boot_n": BOOT_N, "boot_seed": BOOT_SEED},
           "replication_constants": {
               "T3_arm_N_delta": T3_ARM_N, "T3_arm_Q_delta": T3_ARM_Q, "T3_fold": T3_FOLD,
               "file": "density/results/t3_rotation.json / t2b_organs.json",
               "note": "arm N folded 2L+1 gains; this probe's NL folds 2L. Brief s2 argues "
                       "the difference is neutral for an fp32 head; XA is the arm that tests "
                       "that claim, exactly, at fp32."},
           "arms": {}}

    per_seq_nats = {}       # arm -> exact PyTorch per-sequence nats, for the paired bootstrap
    t_all = time.time()

    for arm in arms:
        fold, quant, head_tern = ARMS[arm]
        wpath = os.path.join(a.workdir, "%s_%s.bin" % (tag, arm.lower()))
        t0 = time.time()
        if not (a.reuse and os.path.exists(wpath)):
            cmd = [sys.executable, os.path.join(HERE, "qwen_export.py"),
                   "--model", a.model, "--quant", quant, "--out", wpath,
                   "--calib-seqs", str(a.calib_seqs), "--threads", str(THREADS),
                   "--fold", fold]
            if a.revision:
                cmd += ["--revision", a.revision]
            if quant != "fp32":
                cmd += ["--rule", "R3"]
            if head_tern:
                cmd += ["--head-ternary"]
            print("\n[%s] exporting: %s" % (arm, " ".join(cmd[2:])), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("export failed:\n" + r.stdout[-3000:] + "\n" + r.stderr[-3000:])
            print("   " + r.stdout.strip().splitlines()[-1], flush=True)

        hdr = E1.read_header(wpath)
        side = json.load(open(wpath + ".json", encoding="utf-8"))
        # the sidecar is the artifact's own account of itself: check it says what was asked
        assert side.get("fold") == fold, (side.get("fold"), fold)
        exp_gains = None
        if fold != "none":
            L = hdr["L"]
            exp_gains = 2 * L + (1 if fold == "all" else 0)
            assert side.get("n_gains_folded") == exp_gains, \
                "sidecar says %s gains folded, layout says %s" % (side.get("n_gains_folded"),
                                                                  exp_gains)

        print("[%s] building the PyTorch reference by reading the weights back out of the "
              "exported file..." % arm, flush=True)
        m, ga = E1.build_reference(a.model, a.revision, arm, a.calib_seqs, wpath, hdr,
                                   spec=(quant, head_tern), fold=fold)

        if quant == "fp32":
            print("[%s] GATE A (fp32 round-trip): %d tensors, worst |diff| %.3e -> %s"
                  % (arm, ga["tensors"], ga["worst_product_absdiff"],
                     "PASS" if ga["passes"] else "FAIL"), flush=True)
        else:
            print("[%s] GATE A1 codes: %d differing of %d -> %s"
                  % (arm, ga["codes_differing"], ga["n_codes"],
                     "PASS" if ga["passes_A1_codes"] else "FAIL"), flush=True)
            print("[%s] GATE A2 scales: %d differing, worst %.2f ulp -> %s"
                  % (arm, ga["scales_differing"], ga["worst_scale_ulp"],
                     "PASS" if ga["passes_A2_scales"] else "FAIL"), flush=True)

        rec = {"weights": wpath, "fold": fold, "quant": quant, "head_ternary": head_tern,
               "n_gains_folded": side.get("n_gains_folded"),
               "n_gains_expected": exp_gains,
               "untied_by_fold": side.get("untied_by_fold"),
               "sidecar_rule": side.get("rule"), "sidecar_sha256": side.get("sha256"),
               "mean_ternary_zero_fraction": side.get("mean_ternary_zero_fraction"),
               "gate_a": ga, "export_seconds": time.time() - t0}

        if not ga["passes"]:
            print("[%s] Gate A FAILED -- no BPB is compared for this arm." % arm, flush=True)
            rec["bpb_torch"] = rec["bpb_engine"] = rec["delta"] = None
            out["arms"][arm] = rec
            del m
            continue

        with torch.no_grad():
            bt, per_t = C.bpb(m, ids, byts, batch=1, return_per_seq=True)
        del m
        # per_t is per-sequence BPB; recover exact nats so contrasts are byte-weighted properly
        per_seq_nats[arm] = np.asarray(per_t, dtype=np.float64) * math.log(2.0) * bnp
        print("[%s] PyTorch BPB %.9f" % (arm, bt), flush=True)

        tot, npred, per_e_nats, secs = E1.run_engine(wpath, ids, SL,
                                                     "%s_%s" % (tag, arm.lower()), a.workdir)
        be = tot / (math.log(2.0) * nb)
        assert npred == ids.shape[0] * (SL - 1), (npred, ids.shape[0] * (SL - 1))
        per_e = per_e_nats / (math.log(2.0) * bnp) if len(per_e_nats) == len(per_t) else None
        pmax = float(np.abs(per_e - np.asarray(per_t)).max()) if per_e is not None else None

        rec.update({"bpb_torch": bt, "bpb_engine": be, "delta": be - bt,
                    "n_predicted": npred, "nats_total_engine": tot,
                    "per_seq_max_abs_delta": pmax, "engine_seconds": secs})
        print("[%s] engine  BPB %.9f   delta %+.9f   per-seq max |delta| %s   (%.0fs)"
              % (arm, be, be - bt, ("%.2e" % pmax) if pmax is not None else "n/a", secs),
              flush=True)
        out["arms"][arm] = rec

    # ------------------------------------------------------- brief s5, mechanically
    A = out["arms"]

    def bpb(k, side="bpb_torch"):
        return A[k][side] if k in A and A[k].get(side) is not None else None

    # Gate F -- the fold must be exact where it must be exact. Reported on BOTH sides.
    gate_f = {}
    for k in ("XF", "XA"):
        for side in ("bpb_torch", "bpb_engine"):
            v, ref = bpb(k, side), bpb("F32", side)
            if v is not None and ref is not None:
                gate_f["%s_vs_F32_%s" % (k, side.split("_")[1])] = {
                    "diff": v - ref, "tol": GATE_F_FP32_TOL, "ok": abs(v - ref) <= GATE_F_FP32_TOL}
    gate_f_measured = bool(gate_f)
    gate_f_ok = gate_f_measured and all(v["ok"] for v in gate_f.values())

    # the paired contrasts, on the exact PyTorch per-sequence nats
    contrasts = {}
    for name, x, y in (("NL_minus_TQ", "NL", "TQ"),
                       ("NLH_minus_TQH", "NLH", "TQH"),
                       ("NAH_minus_NLH", "NAH", "NLH"),
                       ("XA_minus_XF", "XA", "XF")):
        if x in per_seq_nats and y in per_seq_nats:
            p, se, lo, hi = paired_bootstrap(per_seq_nats[x], per_seq_nats[y], bnp)
            contrasts[name] = {"delta": p, "paired_se": se, "ci95": [lo, hi],
                               "sigma_seed": abs(p) / SIGMA_SEED,
                               "excludes_zero": bool(lo > 0 or hi < 0)}

    # the fold term of the rule, on its own
    fold_c = contrasts.get("NL_minus_TQ")
    if fold_c is None:
        fold_label = "INCOMPLETE (arms NL and TQ are both required for the decision)"
    elif not fold_c["excludes_zero"]:
        fold_label = "FOLD-NULL"
    elif fold_c["ci95"][0] > 0:
        fold_label = "FOLD-HURTS"
    elif fold_c["delta"] <= FOLD_CONFIRMED:
        fold_label = "FOLD-CONFIRMED"
    elif fold_c["delta"] <= FOLD_SHRINKS:
        fold_label = "FOLD-SHRINKS"
    else:
        fold_label = "FOLD-NULL"

    # brief s3.3(a): a gate nobody evaluated has not failed.  VOID is reserved for a Gate F
    # that was measured and came back bad; an invocation carrying no fp32 arm says so instead
    # and names the run that owes the gate, rather than asserting the fold is inexact.
    if gate_f_measured and not gate_f_ok:
        label = "VOID (Gate F failed: the fold is not exact where it must be)"
    elif not gate_f_measured:
        label = ("%s / GATE-F-NOT-MEASURED-HERE (no F32/XF/XA arm in this invocation; "
                 "brief s3.2 gives Gate F on this donor to run 3 -- not final until it passes)"
                 % fold_label)
    else:
        label = fold_label

    print("\n%-5s %-7s %14s %14s %14s %8s" % ("arm", "fold", "BPB torch", "BPB engine",
                                              "delta", "gate A"))
    for k in arms:
        if k not in A:
            continue
        r = A[k]
        print("%-5s %-7s %14s %14s %14s %8s"
              % (k, r["fold"],
                 "%.9f" % r["bpb_torch"] if r.get("bpb_torch") is not None else "-",
                 "%.9f" % r["bpb_engine"] if r.get("bpb_engine") is not None else "-",
                 "%+.9f" % r["delta"] if r.get("delta") is not None else "-",
                 "PASS" if r["gate_a"]["passes"] else "FAIL"))

    if contrasts:
        print("\npaired between-arm contrasts (PyTorch side, exact per-sequence nats, "
              "%d resamples seed %d)" % (BOOT_N, BOOT_SEED))
        for k, v in contrasts.items():
            print("  %-14s %+.6f  SE %.6f  ci95 [%+.6f, %+.6f]  %5.0f sigma_seed  %s"
                  % (k, v["delta"], v["paired_se"], v["ci95"][0], v["ci95"][1],
                     v["sigma_seed"], "excludes 0" if v["excludes_zero"] else "contains 0"))

    # replication, only where it applies
    repl = {}
    if a.model.endswith("1.5B") and ids.shape[0] == 24:
        for k, const, nm in (("TQ", T3_ARM_Q, "T3 arm Q / T2b arm FA (196 tensors, no fold)"),
                             ("NL", T3_ARM_N, "T3 arm N (196 tensors, fold) -- NEAR, not exact: "
                                              "T3 folded 2L+1 gains, NL folds 2L")):
            if bpb(k) is not None:
                d = bpb(k) - BASE_1P5B
                repl[k] = {"delta_vs_base": d, "standing": const, "which": nm,
                           "residue": d - const}
                print("  %-3s vs %s\n      %+.9f vs %+.9f  residue %+.3e"
                      % (k, nm, d, const, d - const))

    out["replication"] = repl
    out["gate_F"] = gate_f
    out["contrasts"] = contrasts
    out["decision"] = {"rule": "brief section 5 as amended in s3.1, fixed before the run",
                       "gate_F_measured": gate_f_measured,
                       "gate_F_ok": gate_f_ok, "fold_term_label": fold_label,
                       "OUTCOME_LABEL": label}
    print("\nOUTCOME: %s" % label)

    out["total_seconds"] = time.time() - t_all
    canonical = (arms == list(ARMS) and a.seqs == 24)
    sub = "" if canonical else "_%s_s%d" % ("".join(x.lower() for x in arms), a.seqs)
    name = "e2_rmsnorm_fold%s_%s%s.json" % ("_smoke" if a.smoke else "", tag, sub)
    p = os.path.join(os.path.abspath(os.path.join(HERE, "..", "density", "results")), name)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print("wrote %s  (%.0f s)" % (p, out["total_seconds"]))


if __name__ == "__main__":
    main()
