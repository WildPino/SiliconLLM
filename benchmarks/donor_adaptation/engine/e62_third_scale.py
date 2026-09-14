#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E62 -- does the one-byte rung stay faithful as the donor grows?

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E62_DOES_THE_ONE_BYTE_RUNG_SURVIVE_SCALE.md`
Pre-registered and PUSHED (4c91b1c) before this file existed.

Every headline this programme currently carries -- 36.6-37.6 tok/s at 10 B, 73-75% of the bar
(`SPEED_LEDGER` section 60.4) -- assumes the one-byte rung is still FAITHFUL at 10 B.  Two points
exist (dBPB 6.64e-05 at 0.4941 B, 1.2525e-03 at 1.5437 B) and they read either as a power law
(~0.15 BPB at 10 B, the headline void) or as linear in N (~0.008, the rung fine): a ~19x spread
at the target.  One measurement at a third scale separates them.

WHAT THIS RUNNER IS CAREFUL ABOUT.

  * It carries NO speed claim.  E61 measured this box at 4.44-19.57% foreign occupancy with nine
    of ten cell-groups over OCC_BAR.  A BPB pass does not care -- E61 reproduced 05b F32 to
    |d| = 0.00e+00 across two runs 710 s apart -- and E62 is deliberately scoped to the only axis
    a dirty box cannot corrupt.  Occupancy is still RECORDED, as conduct, and gates nothing.

  * It asks nothing of the operator's memory.  E61 found `CONFIG` printing `mvacc=4` on a kernel
    that read 1: a configuration defect does not look like an error, it looks like a plausible
    number.  So every export has its SIDECAR checked against what was asked, and every engine run
    has its `CONFIG  ... quant=` checked against the arm it claims to be.  A cell whose witness
    disagrees is refused, not recorded.

  * It plants first.  G-E62a re-measures two cells this programme already knows to 1e-08 before
    any 3 B number is read.

  * It writes results after EVERY phase.
"""
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e44_interval as E44                                          # noqa: E402
import e51_math_errno as E51                                        # noqa: E402

OUTDIR = os.path.join(HERE, "results")
OUTFILE = os.path.join(OUTDIR, "e62_third_scale.json")
BRIEF = "BRIEF_E62_DOES_THE_ONE_BYTE_RUNG_SURVIVE_SCALE.md"
EXPORTER = os.path.join(HERE, "qwen_export.py")

ENGINE = os.path.join(HERE, "donor_engine_e61.exe")     # E61's binary; --mvacc 4 is the default
THREADS = 6

E1 = os.path.join("D:", os.sep, "_ktmp", "e1")
E60 = os.path.join("D:", os.sep, "_ktmp", "e60")
E62 = os.path.join("D:", os.sep, "_ktmp", "e62")

# ---------------------------------------------------------------- the standard slice
# 24 x 512, 12264 predicted positions, 51870 scored bytes, chance 4.069819.  The ids file is
# byte-identical across the Qwen2.5 family (one tokenizer, ids_sha256 a1a48dc9...), which is the
# only reason a 3 B cell is comparable to the 0.5 B and 1.5 B ones at all.
BPB_IDS = os.path.join(E1, "ids_qwen25-05b_tqh.bin")
BPB_SEQLEN = 512
BPB_BYTES = 51870
BPB_NPRED = 12264
CHANCE_BPB = 4.069819
SIGMA_SEED = 0.005

# ---------------------------------------------------------------- what is already known
# E60's measured engine cells (results/e60_one_byte.json).  G-E62a's planted references.
E60_BPB = {("05b", "F32"): 0.871810465558026,
           ("05b", "I8"): 0.8718768840274047,
           ("15b", "F32"): 0.767606372511151,
           ("15b", "I8"): 0.7688588381536873}
G62A_TOL = 1.0e-08

# E12's PyTorch fp32 baselines, from density/results/e12_scale.json, arm "base".  A DIFFERENT
# implementation on the same slice: the cross-path control, not a baseline.
E12_PT_FP32 = {"05b": 0.8717951206093644,
               "15b": 0.7675949584171732,
               "3b": 0.7244497971214012}
G62B_TOL = 5.0e-05

# E1's packed R3 cells against the chance line, for G-E62e's trend.
E1_PACKED = {"05b": 4.531233733626962, "15b": 3.47570669163278}

# Total parameters, from each donor's own config (embeddings included; all three are tied).
NPARAM = {"05b": 0.4941e9, "15b": 1.5437e9, "3b": 3.0859e9}

# ---------------------------------------------------------------- the 3 B donor
# The local snapshot IS E12's revision -- checked on disk, not assumed -- so the cross-path
# control compares two implementations of the same weights and nothing else.
MODEL_3B = "Qwen/Qwen2.5-3B"
REV_3B = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

# E1's protocol, verbatim (`e1_bpb_through_engine.py` ARM_SPEC + the pinned `--fold none`).
# NOTE on --head-ternary: E1 applies it to the QUANTISED arms only.  On an fp32 arm it would
# ternarise lm_head, and the fp32 arm is precisely the thing G-E62b compares against a PyTorch
# fp32 reference -- so passing it there would void the control it exists to serve.
ARMS_3B = [
    ("F32",    {"quant": "fp32",   "rule": None, "head_ternary": False}, "fp32"),
    ("I8",     {"quant": "int8",   "rule": "R8", "head_ternary": True},  "int8"),
    ("PACKED", {"quant": "packed", "rule": "R3", "head_ternary": True},  "packed"),
]
CALIB_SEQS = 32
FOLD = "none"

# quant string the engine's CONFIG line must print for each arm.
CONFIG_QUANT = {"F32": "fp32", "I8": "int8", "PACKED": "packed", "T1": "ternary"}


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


# ============================================================ the decision functions
def g_e62a(obs, tol=G62A_TOL):
    """PLANTED: the instrument reproduces what it already knows.

    `obs` = {name: (measured, reference)}.  E61 read |d| = 0.00e+00 on 05b F32 twice, so a
    tolerance of 1e-08 is tight ON PURPOSE: this axis has no dispersion to respect.
    """
    if not obs:
        return "DEAD -- nothing re-measured"
    bad = ["%s |d|=%.2e" % (k, abs(m - r)) for k, (m, r) in sorted(obs.items())
           if abs(m - r) > tol]
    if bad:
        return "DEAD -- " + "; ".join(bad)
    return "FIRES"


def g_e62b(measured, ref=E12_PT_FP32["3b"], tol=G62B_TOL):
    """The cross-path control at the new scale: engine fp32 vs E12's PyTorch fp32."""
    if measured is None:
        return "DEAD -- no 3B fp32 cell", None
    d = abs(measured - ref)
    return ("ADMISSIBLE" if d <= tol else "VOID"), d


def g_e62c(dbpb):
    """THE HEADLINE.  Bands fixed in the brief, section 4, before anything ran.

    <= 0.0025           AT-MOST-LINEAR           (0.0025 is exactly linear-in-N from 1.5 B)
    0.0025 .. 0.0060    SUPERLINEAR-BUT-BOUNDED
    >= 0.0060           COST-EXPLODES            (section 60.4's 10 B headline void at target)
    """
    if dbpb is None:
        return "DEAD -- no 3B int8 cell"
    if dbpb <= 0.0025:
        return "AT-MOST-LINEAR"
    if dbpb < 0.0060:
        return "SUPERLINEAR-BUT-BOUNDED"
    return "COST-EXPLODES"


def g_e62d(by_n, within):
    """The RANK partner (E14 section 3).  A SCORE without it carries no trend claim.

    `by_n`   = [(N, dBPB), ...] -- clause 1: dBPB strictly increasing in N.
    `within` = {"F32":bpb, "I8":bpb, "PACKED":bpb} -- clause 2: F32 < I8 << PACKED.
    """
    if len(by_n) < 3 or len(within) < 3:
        return "DEAD -- needs three scales and three arms"
    s = sorted(by_n)
    c1 = all(s[i][1] < s[i + 1][1] for i in range(len(s) - 1))
    c2 = within["F32"] < within["I8"] < within["PACKED"]
    if c1 and c2:
        return "HOLDS"
    if c1:
        return "CLAUSE2-BREAKS"
    if c2:
        return "CLAUSE1-BREAKS"
    return "BOTH-BREAK"


def g_e62e(bpb_packed, chance=CHANCE_BPB):
    """E12's owed 3 B cell at the rule this programme actually ships.  Descriptive."""
    if bpb_packed is None:
        return "DEAD -- no 3B packed cell"
    return "BELOW-CHANCE" if bpb_packed < chance else "ABOVE-CHANCE"


def power_fit(points):
    """(exponent, coefficient) of dBPB = c * N^p through >=2 points, least squares in log-log."""
    pts = [(n, d) for n, d in points if n > 0 and d > 0]
    if len(pts) < 2:
        return None, None
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(d) for _, d in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None, None
    p = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return p, math.exp(my - p * mx)


# ============================================================ self-test
def selftest():
    n = [0]

    def chk(what, ok):
        n[0] += 1
        log("  %-78s %s" % (what, "ok" if ok else "FAIL"))
        if not ok:
            raise SystemExit("self-test failed: " + what)

    chk("A-1 G-E62a FIRES on an exact reproduction",
        g_e62a({"x": (1.0, 1.0)}) == "FIRES")
    chk("A-2 FIRES just inside the tolerance",
        g_e62a({"x": (1.0, 1.0 + 9e-09)}) == "FIRES")
    chk("A-3 DEAD just outside it",
        g_e62a({"x": (1.0, 1.0 + 2e-08)}).startswith("DEAD"))
    chk("A-4 DEAD names which cell moved",
        "b " in g_e62a({"a": (1.0, 1.0), "b": (1.0, 1.1)}))
    chk("A-5 DEAD with nothing to plant", g_e62a({}).startswith("DEAD"))

    chk("B-1 G-E62b ADMISSIBLE in family with the other two scales",
        g_e62b(E12_PT_FP32["3b"] + 1.5e-05)[0] == "ADMISSIBLE")
    chk("B-2 ADMISSIBLE at the boundary",
        g_e62b(E12_PT_FP32["3b"] + G62B_TOL)[0] == "ADMISSIBLE")
    chk("B-3 VOID beyond it -- the export is wrong, the cell is not adverse",
        g_e62b(E12_PT_FP32["3b"] + 1e-04)[0] == "VOID")
    chk("B-4 DEAD without a cell", g_e62b(None)[0].startswith("DEAD"))

    chk("C-1 G-E62c AT-MOST-LINEAR at exactly linear-in-N", g_e62c(0.0025) == "AT-MOST-LINEAR")
    chk("C-2 AT-MOST-LINEAR below it", g_e62c(0.0019) == "AT-MOST-LINEAR")
    chk("C-3 SUPERLINEAR-BUT-BOUNDED in the middle band",
        g_e62c(0.0040) == "SUPERLINEAR-BUT-BOUNDED")
    chk("C-4 SUPERLINEAR-BUT-BOUNDED just under the upper edge",
        g_e62c(0.0059999) == "SUPERLINEAR-BUT-BOUNDED")
    chk("C-5 COST-EXPLODES at the upper edge -- the 10 B headline dies there",
        g_e62c(0.0060) == "COST-EXPLODES")
    chk("C-6 COST-EXPLODES at the power-law point", g_e62c(0.00747) == "COST-EXPLODES")
    chk("C-7 DEAD without a cell", g_e62c(None).startswith("DEAD"))

    rank_ok = [(0.4941e9, 6.6e-05), (1.5437e9, 1.25e-03), (3.0859e9, 3.0e-03)]
    rank_no = [(0.4941e9, 6.6e-05), (1.5437e9, 1.25e-03), (3.0859e9, 9.0e-04)]
    w_ok = {"F32": 0.7244, "I8": 0.7280, "PACKED": 3.20}
    w_no = {"F32": 0.7244, "I8": 3.20, "PACKED": 0.7280}
    chk("D-1 G-E62d HOLDS when both orderings hold", g_e62d(rank_ok, w_ok) == "HOLDS")
    chk("D-2 CLAUSE1-BREAKS when dBPB is not increasing in N",
        g_e62d(rank_no, w_ok) == "CLAUSE1-BREAKS")
    chk("D-3 CLAUSE2-BREAKS when the arms are out of order",
        g_e62d(rank_ok, w_no) == "CLAUSE2-BREAKS")
    chk("D-4 BOTH-BREAK when neither holds", g_e62d(rank_no, w_no) == "BOTH-BREAK")
    chk("D-5 DEAD on two scales -- a rank claim needs the third",
        g_e62d(rank_ok[:2], w_ok).startswith("DEAD"))

    chk("E-1 G-E62e BELOW-CHANCE under 4.069819", g_e62e(3.2) == "BELOW-CHANCE")
    chk("E-2 ABOVE-CHANCE over it", g_e62e(4.53) == "ABOVE-CHANCE")
    chk("E-3 DEAD without a cell", g_e62e(None).startswith("DEAD"))

    p, c = power_fit([(1.0, 1.0), (2.0, 4.0)])
    chk("F-1 power_fit recovers an exponent of 2 exactly", abs(p - 2.0) < 1e-12)
    chk("F-2 and its coefficient", abs(c - 1.0) < 1e-12)
    p2, _ = power_fit([(0.4941e9, 6.64185e-05), (1.5437e9, 1.2524656e-03)])
    chk("F-3 the two known int8 points fit N^2.578 as the brief says", abs(p2 - 2.578) < 0.002)
    chk("F-4 power_fit refuses a single point", power_fit([(1.0, 1.0)])[0] is None)
    chk("F-5 power_fit drops a non-positive dBPB rather than taking log(0)",
        power_fit([(1.0, 0.0), (2.0, 4.0)])[0] is None)

    chk("G-1 the chance line is the standard slice's", abs(CHANCE_BPB - 4.069819) < 1e-09)
    chk("G-2 E12's 0.5 B cross-path disagreement is ~1.5e-05, inside G-E62b's tolerance",
        abs(E60_BPB[("05b", "F32")] - E12_PT_FP32["05b"]) < G62B_TOL)
    chk("G-3 and the 1.5 B one too",
        abs(E60_BPB[("15b", "F32")] - E12_PT_FP32["15b"]) < G62B_TOL)
    chk("G-4 E60's two int8 deltas are the brief's two anchor points",
        abs((E60_BPB[("05b", "I8")] - E60_BPB[("05b", "F32")]) - 6.64185e-05) < 1e-09
        and abs((E60_BPB[("15b", "I8")] - E60_BPB[("15b", "F32")]) - 1.2524656e-03) < 1e-09)
    log("")
    log("  %d of %d fire." % (n[0], n[0]))


# ============================================================ measurement primitives
def export_arm(arm, spec, outpath, reuse=True):
    """Export one 3 B arm with E1's protocol, then CHECK the sidecar says what was asked."""
    side = outpath + ".json"
    if reuse and os.path.exists(outpath) and os.path.exists(side):
        log("   %-7s reuse %s" % (arm, outpath))
    else:
        cmd = [sys.executable, EXPORTER, "--model", MODEL_3B, "--revision", REV_3B,
               "--quant", spec["quant"], "--out", outpath,
               "--calib-seqs", str(CALIB_SEQS), "--threads", str(THREADS), "--fold", FOLD]
        if spec["rule"]:
            cmd += ["--rule", spec["rule"]]
        if spec["head_ternary"]:
            cmd += ["--head-ternary"]
        log("   %-7s exporting: %s" % (arm, " ".join(cmd[2:])))
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write((r.stdout or "")[-3000:] + "\n" + (r.stderr or "")[-3000:])
            raise SystemExit("export failed: " + arm)
        log("   %-7s exported in %.0f s" % (arm, time.time() - t0))

    s = json.load(open(side, encoding="utf-8"))
    # E61's lesson, applied to the exporter: the sidecar is CHECKED, not trusted.  A wrong
    # --fold would move the packed arm by ~0.22 BPB (T3) and nothing downstream would notice.
    want = {"model": MODEL_3B, "revision": REV_3B, "quant": spec["quant"],
            "rule": spec["rule"] if spec["rule"] else "R0",
            "head_ternary": spec["head_ternary"], "fold": FOLD}
    for k, v in want.items():
        if s.get(k) != v:
            raise SystemExit("sidecar disagrees for %s: %s = %r, asked %r"
                             % (arm, k, s.get(k), v))
    if s.get("d_model") != 2048 or s.get("n_layers") != 36 or s.get("vocab") != 151936:
        raise SystemExit("sidecar is not the 3 B architecture: " + json.dumps(
            {k: s.get(k) for k in ("d_model", "n_layers", "vocab")}))
    return {"path": outpath, "bytes": s["bytes"], "sha256": s["sha256"],
            "tied": s.get("tied"), "calib_seqs": s.get("calib_seqs"),
            "mean_ternary_zero_fraction": s.get("mean_ternary_zero_fraction"),
            "active_weights_per_token": s.get("active_weights_per_token")}


def bpb_cell(weights, arm):
    """(bpb, nats, seconds, occupancy record).  The engine reports NATS; BPB is ours.

    Occupancy is recorded because E61's conduct section had to be written from it -- but this
    experiment gates nothing on speed, so a dirty reading here is reported, not acted on.
    """
    sp = E44.Split()
    cmd = [ENGINE, "--weights", weights, "--threads", str(THREADS),
           "--seqlen", str(BPB_SEQLEN), "--bpb", BPB_IDS]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sout, serr = p.communicate()
    child = E44._process_times(int(p._handle))
    dt = time.time() - t0
    if p.returncode != 0:
        sys.stderr.write(serr.decode("utf-8", "replace")[-1500:])
        raise SystemExit("engine failed on " + weights)
    occ, foc = sp.close(child)
    txt = sout.decode("utf-8", "replace")

    cfg = [l for l in txt.splitlines() if l.startswith("CONFIG")]
    cfg = cfg[0] if cfg else ""
    # The witness, checked: E61 exists because a CONFIG line named a kernel it was not running.
    if ("quant=%s" % CONFIG_QUANT[arm]) not in cfg:
        raise SystemExit("CONFIG does not report quant=%s for arm %s: %s"
                         % (CONFIG_QUANT[arm], arm, cfg))
    m = E51.RE_NATS.search(txt)
    if not m:
        raise SystemExit("no NATS_TOTAL for " + weights)
    tot, npred = float(m.group(1)), int(m.group(2))
    if npred != BPB_NPRED:
        raise SystemExit("slice changed: %d predicted, expected %d" % (npred, BPB_NPRED))
    rec = {"bpb": tot / (math.log(2.0) * BPB_BYTES), "nats": tot, "n_predicted": npred,
           "seconds": dt, "occ": occ, "foreign": foc, "config": cfg}
    rec.update(sp.record())
    return rec


def save(out):
    os.makedirs(OUTDIR, exist_ok=True)
    out["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return OUTFILE


# ============================================================
def main():
    log("E62 -- does the one-byte rung stay faithful as the donor grows?")
    log("brief: " + BRIEF)
    log("")
    log("== self-test ==")
    selftest()
    if "--selftest" in sys.argv:
        return
    reuse = "--no-reuse" not in sys.argv

    for p in (ENGINE, BPB_IDS, EXPORTER):
        if not os.path.exists(p):
            raise SystemExit("missing: " + p)
    os.makedirs(E62, exist_ok=True)

    t_all = time.time()
    out = {"brief": BRIEF, "engine": os.path.basename(ENGINE), "threads": THREADS,
           "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "donor_3b": {"model": MODEL_3B, "revision": REV_3B, "n_params": NPARAM["3b"]},
           "protocol": {"fold": FOLD, "calib_seqs": CALIB_SEQS,
                        "head_ternary_on": ["I8", "PACKED"],
                        "note": "E1's protocol; --head-ternary is NOT applied to the fp32 arm, "
                                "which is the arm G-E62b compares to a PyTorch fp32 reference"},
           "slice": {"n_seq": 24, "seq_len": BPB_SEQLEN, "bytes": BPB_BYTES,
                     "n_predicted": BPB_NPRED, "chance_bpb": CHANCE_BPB,
                     "ids": BPB_IDS},
           "references": {"E60_engine": {"%s/%s" % k: v for k, v in E60_BPB.items()},
                          "E12_pytorch_fp32": E12_PT_FP32,
                          "E1_packed_R3": E1_PACKED},
           "cells": {}}
    save(out)

    # ------------------------------------------------------------ phase 1: G-E62a (PLANTED)
    log("")
    log("== G-E62a -- PLANTED: the instrument reproduces what it already knows ==")
    log("   05b F32 and 05b I8, against E60/E61, |d| <= %.1e.  FAIL => nothing below is read."
        % G62A_TOL)
    planted = {}
    obs = {}
    for arm, wp in (("F32", os.path.join(E1, "qwen25-05b_f32.bin")),
                    ("I8", os.path.join(E60, "qwen25-05b_i8h.bin"))):
        if not os.path.exists(wp):
            raise SystemExit("missing planted weights: " + wp)
        c = bpb_cell(wp, arm)
        ref = E60_BPB[("05b", arm)]
        d = c["bpb"] - ref
        c["reference"] = ref
        c["delta_vs_reference"] = d
        planted["05b_" + arm] = c
        obs["05b " + arm] = (c["bpb"], ref)
        log("   05b %-7s BPB %.15f   ref %.15f   d = %+.2e   (%.0f s, foreign %.2f%%)"
            % (arm, c["bpb"], ref, d, c["seconds"], c["foreign"] * 100.0))
    va = g_e62a(obs)
    log("   G-E62a : %s" % va)
    out["cells"]["planted"] = planted
    out["G_E62a"] = {"tol": G62A_TOL, "verdict": va,
                     "observed": {k: {"measured": m, "reference": r, "delta": m - r}
                                  for k, (m, r) in obs.items()}}
    save(out)
    if va != "FIRES":
        raise SystemExit("the planted control did not fire.  STOP.")

    # ------------------------------------------------------------ phase 2: the three exports
    log("")
    log("== the 3 B exports -- E1's protocol, sidecars checked against what was asked ==")
    files = {}
    for arm, spec, _ in ARMS_3B:
        outp = os.path.join(E62, "qwen25-3b_%s.bin" % arm.lower())
        files[arm] = export_arm(arm, spec, outp, reuse=reuse)
        log("   %-7s %s  %.3f GB  sha %s"
            % (arm, os.path.basename(outp), files[arm]["bytes"] / 1e9,
               files[arm]["sha256"][:16]))
        out["exports"] = files
        save(out)

    # ------------------------------------------------------------ phase 3: the 3 B cells
    log("")
    log("== the 3 B BPB cells ==")
    cells3 = {}
    for arm, _, _ in ARMS_3B:
        c = bpb_cell(files[arm]["path"], arm)
        cells3[arm] = c
        log("   3b  %-7s BPB %.15f   (%.0f s, foreign %.2f%%)"
            % (arm, c["bpb"], c["seconds"], c["foreign"] * 100.0))
        out["cells"]["3b"] = cells3
        save(out)

    b32 = cells3["F32"]["bpb"]
    bi8 = cells3["I8"]["bpb"]
    bpk = cells3["PACKED"]["bpb"]

    # ------------------------------------------------------------ phase 4: G-E62b
    log("")
    log("== G-E62b -- the cross-path control at the new scale ==")
    vb, db = g_e62b(b32)
    log("   engine fp32 %.15f  vs E12 PyTorch %.15f   |d| = %.2e  (tol %.1e)"
        % (b32, E12_PT_FP32["3b"], db, G62B_TOL))
    log("   for scale: the same comparison reads %.2e at 0.5 B and %.2e at 1.5 B"
        % (abs(E60_BPB[("05b", "F32")] - E12_PT_FP32["05b"]),
           abs(E60_BPB[("15b", "F32")] - E12_PT_FP32["15b"])))
    log("   G-E62b : %s" % vb)
    out["G_E62b"] = {"engine_fp32": b32, "pytorch_fp32": E12_PT_FP32["3b"],
                     "abs_delta": db, "tol": G62B_TOL, "verdict": vb}
    save(out)

    # ------------------------------------------------------------ phase 5: G-E62c, THE HEADLINE
    log("")
    log("== G-E62c -- THE HEADLINE: does the int8 cost explode with scale? ==")
    d3 = bi8 - b32
    d05 = E60_BPB[("05b", "I8")] - E60_BPB[("05b", "F32")]
    d15 = E60_BPB[("15b", "I8")] - E60_BPB[("15b", "F32")]
    vc = g_e62c(d3)
    pts = [(NPARAM["05b"], d05), (NPARAM["15b"], d15), (NPARAM["3b"], d3)]
    p3, c3 = power_fit(pts)
    p2, c2 = power_fit(pts[:2])
    lin10 = d3 * (10.0e9 / NPARAM["3b"]) if d3 is not None else None
    pow10 = (c3 * (10.0e9 ** p3)) if p3 else None
    for nm, n, d, base in (("0.5 B", NPARAM["05b"], d05, E60_BPB[("05b", "F32")]),
                           ("1.5 B", NPARAM["15b"], d15, E60_BPB[("15b", "F32")]),
                           ("3.0 B", NPARAM["3b"], d3, b32)):
        log("   %-6s N = %.4f B   dBPB = %.9f   = %.3f sigma_seed   rel = %.3e"
            % (nm, n / 1e9, d, d / SIGMA_SEED, d / base))
    log("")
    if p3:
        log("   power fit on all three : dBPB ~ N^%.3f   -> %.5f at 10 B (%.1f sigma_seed)"
            % (p3, pow10, pow10 / SIGMA_SEED))
    else:
        log("   power fit on all three : not available (a non-positive dBPB)")
    if p2:
        log("   the two-point fit was  : dBPB ~ N^%.3f" % p2)
    log("   linear-in-N from 3 B   : %.5f at 10 B (%.1f sigma_seed)"
        % (lin10, lin10 / SIGMA_SEED))
    log("   G-E62c : %s" % vc)
    out["G_E62c"] = {"dbpb": {"05b": d05, "15b": d15, "3b": d3},
                     "dbpb_relative_3b": d3 / b32,
                     "sigma_seed": SIGMA_SEED, "dbpb_3b_in_sigma": d3 / SIGMA_SEED,
                     "power_fit_three_points": {"exponent": p3, "coefficient": c3,
                                                "implied_10B": pow10},
                     "power_fit_two_points": {"exponent": p2, "coefficient": c2},
                     "linear_from_3b_implied_10B": lin10,
                     "bands": {"AT-MOST-LINEAR": "<= 0.0025",
                               "SUPERLINEAR-BUT-BOUNDED": "0.0025 .. 0.0060",
                               "COST-EXPLODES": ">= 0.0060"},
                     "verdict": vc}
    save(out)

    # ------------------------------------------------------------ phase 6: G-E62d (RANK)
    log("")
    log("== G-E62d -- the RANK partner (E14 section 3) ==")
    vd = g_e62d([(NPARAM["05b"], d05), (NPARAM["15b"], d15), (NPARAM["3b"], d3)],
                {"F32": b32, "I8": bi8, "PACKED": bpk})
    log("   clause 1  dBPB strictly increasing in N : %.3e -> %.3e -> %.3e" % (d05, d15, d3))
    log("   clause 2  within 3 B  F32 %.6f < I8 %.6f << PACKED %.6f" % (b32, bi8, bpk))
    log("             gaps: I8-F32 = %.6f, PACKED-I8 = %.6f (ratio %.0fx)"
        % (bi8 - b32, bpk - bi8, (bpk - bi8) / (bi8 - b32) if bi8 > b32 else float("nan")))
    log("   G-E62d : %s" % vd)
    out["G_E62d"] = {"clause1_points": [[NPARAM["05b"], d05], [NPARAM["15b"], d15],
                                        [NPARAM["3b"], d3]],
                     "clause2": {"F32": b32, "I8": bi8, "PACKED": bpk},
                     "verdict": vd}
    save(out)

    # ------------------------------------------------------------ phase 7: G-E62e
    log("")
    log("== G-E62e -- E12's owed 3 B cell, at the rule this programme actually ships ==")
    ve = g_e62e(bpk)
    log("   packed R3 across the family: 0.5 B %.6f -> 1.5 B %.6f -> 3 B %.6f   (chance %.6f)"
        % (E1_PACKED["05b"], E1_PACKED["15b"], bpk, CHANCE_BPB))
    log("   step 0.5->1.5 = %+.6f, step 1.5->3 = %+.6f"
        % (E1_PACKED["15b"] - E1_PACKED["05b"], bpk - E1_PACKED["15b"]))
    log("   G-E62e : %s   (descriptive -- no verdict rides on it)" % ve)
    out["G_E62e"] = {"packed_R3": {"05b": E1_PACKED["05b"], "15b": E1_PACKED["15b"],
                                   "3b": bpk},
                     "chance_bpb": CHANCE_BPB, "verdict": ve}
    save(out)

    # ------------------------------------------------------------ conduct + summary
    focs = [c["foreign"] for c in list(planted.values()) + list(cells3.values())]
    out["conduct"] = {"foreign_min": min(focs), "foreign_max": max(focs),
                      "occ_bar": 4.39,
                      "note": "recorded, gates nothing: E62 carries no speed claim"}
    out["seconds_total"] = time.time() - t_all
    save(out)
    log("")
    log("== summary ==")
    log("   G-E62a %s | G-E62b %s | G-E62c %s | G-E62d %s | G-E62e %s"
        % (va, vb, vc, vd, ve))
    log("   foreign occupancy over the run: %.2f%% .. %.2f%%   (conduct only)"
        % (min(focs) * 100.0, max(focs) * 100.0))
    log("   %.1f min total.  %s" % (out["seconds_total"] / 60.0, OUTFILE))


if __name__ == "__main__":
    main()
