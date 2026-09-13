#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E55 -- the demonstration table, with an interval that survives.

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E55_THE_DEMONSTRATION_TABLE_WITH_AN_INTERVAL_THAT_SURVIVES.md`
Pre-registered and pushed before this file ran a single cell.

SHAPE REAL, WEIGHTS SYNTHETIC, QUALITY NOT DEMONSTRATED.  This is a speed claim about the 10B
target shape and not a working-model claim, and the runner prints that on every table it emits
so the sentence cannot be separated from the numbers.
"""
import json
import os
import random
import statistics
import subprocess
import sys

import e44_interval as E44

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results")
ENGINE = os.path.join(HERE, "donor_engine_e53.exe")

R128 = "D:/_ktmp/e40/e40_r128.bin"
ARM = ["--carve-k", "3", "--fexp", "libm"]
THREADS = 6
OCC_BAR = E44.OCC_BAR

# k allocated by COST, not uniformly: every k is at or above where E54's resampling shows the
# interquartile width converged, and each window costs roughly the same wall time.
PLAN = [(160, 40), (640, 25), (1280, 25), (2560, 15)]

IQR_TOL = 0.20      # G-E55a: half-sample IQR must match full-sample IQR within 20% of the full
TARGET = 50.0
EXCELLENT = 100.0
SEED = 5501

SCOPE = [
    "SHAPE REAL      e40_r128.bin, 5.11 GB, E=256 experts, k=3 active = 1.17% of the FFN, GQA",
    "WEIGHTS SYNTHETIC   these are not a trained model's parameters (E45: rate does not depend",
    "                    on weight VALUES, bounded under 5% -- supported, not certified)",
    "QUALITY NOT DEMONSTRATED AT THIS SHAPE   E37's verdict on post-hoc conversion was NO;",
    "                    the trained-carve attempt is H1 and is not finished",
    "=> a SPEED claim about the SHAPE.  NOT a working-model claim.",
]


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


# ======================================================================================
def quantile(v, q):
    """Linear-interpolated quantile on a sorted copy.  No numpy dependency."""
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    p = q * (len(s) - 1)
    lo = int(p)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (p - lo)


def iqr_pct(v):
    """Interquartile width as a percent of the median -- the statistic E54 B.5 showed converges."""
    m = statistics.median(v)
    return 100.0 * (quantile(v, 0.75) - quantile(v, 0.25)) / m if m else float("nan")


def g_e55a(vals, rng):
    """The interval must be STABLE, or it is not an interval.

    Full-sample IQR against the IQR of a random half, required to agree within IQR_TOL of the
    FULL-SAMPLE width.  Derived in brief section 5: E54's resampling puts the k=20 -> k=40 IQR
    change at 7% and the min-max change at 30%, so a converged statistic passes and the
    statistic this programme has been quoting would fail.
    """
    if len(vals) < 6:
        return ("UNSTABLE", float("nan"), "only %d reps -- too few to split" % len(vals))
    full = iqr_pct(vals)
    half = iqr_pct(rng.sample(vals, len(vals) // 2))
    if full <= 0:
        return ("UNSTABLE", float("nan"), "full-sample interquartile width is zero")
    rel = abs(half - full) / full
    if rel <= IQR_TOL:
        return ("PASS", rel, "IQR %.2f%% on %d reps vs %.2f%% on a random half = %.0f%% apart"
                % (full, len(vals), half, 100 * rel))
    return ("UNSTABLE", rel, "IQR %.2f%% on %d reps vs %.2f%% on a random half = %.0f%% apart,"
            " past the %.0f%% bar -- this is not an interval, it is a sample"
            % (full, len(vals), half, 100 * rel, 100 * IQR_TOL))


def g_e55b(p25):
    """The claim rides the 25th percentile, not the median."""
    if p25 >= EXCELLENT:
        return "EXCELLENT HELD"
    if p25 >= TARGET:
        return "TARGET HELD"
    return "BELOW TARGET"


def fit(pts):
    """ms/tok = a + b*pos, least squares.  Returns (a, b, crossing at 20 ms/tok)."""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    if b <= 0:
        return {"a": a, "b": b, "cross": float("inf")}
    return {"a": a, "b": b, "cross": (1000.0 / TARGET - a) / b}


def g_e55c(rows):
    """How far does it hold?  A RANGE between the median fit and the 25th-percentile fit."""
    ok = [r for r in rows if r["stable"] == "PASS"]
    if len(ok) < 3:
        return ("REFUSED", "only %d of %d windows passed G-E55a; a fit needs three"
                % (len(ok), len(rows)), None)
    fm = fit([(r["mean_pos"], 1000.0 / r["median"]) for r in ok])
    fp = fit([(r["mean_pos"], 1000.0 / r["p25"]) for r in ok])
    if not fm or not fp:
        return ("REFUSED", "the fit is degenerate", None)
    lo, hi = sorted((fm["cross"], fp["cross"]))
    width = 100.0 * (hi - lo) / lo if lo else float("nan")
    return ("BAND", "50 tok/s is crossed between context %.0f (25th pct) and %.0f (median)"
            " -- a %.0f%% wide band, NOT a single C50"
            % (lo, hi, width), {"median_fit": fm, "p25_fit": fp,
                                "cross_lo": lo, "cross_hi": hi, "band_pct": width})


# ======================================================================================
def selftest():
    ok = [0]
    rng = random.Random(1)

    def chk(name, cond):
        ok[0] += 1
        log("  %-52s %s" % (name, "fires" if cond else "*** DEAD ***"))
        if not cond:
            raise SystemExit("a decision function did not fire on a known-positive.  STOP.")

    chk("Q-1 the quantile of a known set is exact", abs(quantile([1, 2, 3, 4, 5], 0.5) - 3) < 1e-9)
    chk("Q-2 and it interpolates", abs(quantile([0.0, 10.0], 0.25) - 2.5) < 1e-9)
    chk("Q-3 IQR of a uniform ramp is half its range",
        abs(iqr_pct([float(i) for i in range(101)]) - 100.0 * 50.0 / 50.0) < 1.0)

    tight = [100.0 + 0.01 * ((i * 37) % 20) for i in range(40)]
    chk("A-1 a converged sample is STABLE", g_e55a(tight, rng)[0] == "PASS")
    spiky = [100.0] * 38 + [10.0, 400.0]
    chk("A-2 a sample dominated by two outliers is UNSTABLE",
        g_e55a(spiky, random.Random(3))[0] == "UNSTABLE")
    chk("A-3 too few reps is UNSTABLE, not PASS", g_e55a([1.0, 2.0, 3.0], rng)[0] == "UNSTABLE")

    chk("B-1 p25 above 100 is EXCELLENT", g_e55b(117.0) == "EXCELLENT HELD")
    chk("B-2 p25 between 50 and 100 is TARGET", g_e55b(76.0) == "TARGET HELD")
    chk("B-3 p25 below 50 is BELOW TARGET", g_e55b(49.9) == "BELOW TARGET")
    chk("B-4 exactly 50 HOLDS (the bar is inclusive)", g_e55b(50.0) == "TARGET HELD")

    # a planted linear cost: ms/tok = 8 + 0.008*pos  ->  50 tok/s at pos (20-8)/0.008 = 1500
    planted = [{"mean_pos": p, "median": 1000.0 / (8 + 0.008 * p),
                "p25": 1000.0 / (8 + 0.008 * p), "stable": "PASS"}
               for p in (80, 320, 640, 1280)]
    v, why, f = g_e55c(planted)
    chk("C-1 a planted crossing is recovered exactly",
        v == "BAND" and abs(f["cross_lo"] - 1500.0) < 1e-6)
    chk("C-2 identical fits give a zero-width band", abs(f["band_pct"]) < 1e-6)

    split = [dict(r, p25=1000.0 / (8.5 + 0.009 * r["mean_pos"])) for r in planted]
    v, why, f = g_e55c(split)
    chk("C-3 a pessimistic band is narrower on the p25 side",
        v == "BAND" and f["cross_lo"] < f["cross_hi"] and f["band_pct"] > 0)

    chk("C-4 fewer than three stable windows is REFUSED",
        g_e55c([dict(r, stable="UNSTABLE") for r in planted[:2]] + planted[2:3])[0] == "REFUSED")

    log("")
    log("  %d of %d fire." % (ok[0], ok[0]))


# ======================================================================================
def one_cell(n):
    sp = E44.Split()
    cmd = [ENGINE, "--weights", R128, "--threads", str(THREADS)] + ARM + ["--bench", str(n)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sout, serr = p.communicate()
    child = E44._process_times(int(p._handle))
    if p.returncode != 0:
        sys.stderr.write(serr.decode("utf-8", "replace")[-1500:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    occ, foc = sp.close(child)
    txt = sout.decode("utf-8", "replace")
    m = E44.RE_BENCH.search(txt)
    if not m:
        raise SystemExit("no BENCH line: " + " ".join(cmd))
    rec = sp.record()
    cfg = [l for l in txt.splitlines() if l.startswith("CONFIG")]
    rec.update({"n": n, "rate": float(m.group(3)), "occ": occ, "foreign": foc,
                "config": cfg[0] if cfg else ""})
    return rec


def main():
    log("E55 -- the demonstration table, with an interval that survives")
    log("brief: BRIEF_E55_THE_DEMONSTRATION_TABLE_WITH_AN_INTERVAL_THAT_SURVIVES.md")
    log("")
    for s in SCOPE:
        log("  " + s)
    log("")
    log("== self-test.  Every decision function must fire on a known-positive. ==")
    selftest()
    if "--selftest" in sys.argv:
        return

    rng = random.Random(SEED)
    log("")
    log("== warm-up cell, DISCARDED (E54 B.4: a run's own launch sits inside its first cells) ==")
    w = one_cell(160)
    log("   discarded: %.2f tok/s  foreign %.1f%%  clock %.2f%%" %
        (w["rate"], w["foreign"], w["clock_pct"]))
    log("   %s" % w["config"])
    log("")

    plan = dict(PLAN)
    maxrep = max(k for _, k in PLAN)
    cells = []
    log("== sweep.  Window order ROTATES each repetition (E54 s2: an ascending ladder makes")
    log("   'time elapsed' and 'heavy work just finished' the same observation). ==")
    log("")
    for rep in range(maxrep):
        order = [n for n, k in PLAN if rep < k]
        rng.shuffle(order)
        for n in order:
            c = one_cell(n)
            c["rep"] = rep + 1
            cells.append(c)
        if (rep + 1) % 5 == 0 or rep == maxrep - 1:
            log("   rep %2d/%2d done  (t+%6.1fs, %d cells)"
                % (rep + 1, maxrep, cells[-1]["since_start_s"], len(cells)))
    log("")

    rows = []
    for n, k in PLAN:
        allc = [c for c in cells if c["n"] == n]
        clean = [c for c in allc if not (c["foreign"] == c["foreign"] and c["foreign"] > OCC_BAR)]
        use = clean if len(clean) >= 6 else allc
        v = [c["rate"] for c in use]
        st, rel, why = g_e55a(v, rng)
        r = {"n": n, "mean_pos": n / 2.0, "reps": len(allc), "reps_clean": len(clean),
             "used": len(use), "median": statistics.median(v),
             "p25": quantile(v, 0.25), "p75": quantile(v, 0.75),
             "iqr_pct": iqr_pct(v), "minmax_pct": 100.0 * (max(v) - min(v)) / statistics.median(v),
             "stable": st, "stable_why": why,
             "median_allcells": statistics.median([c["rate"] for c in allc]),
             "breached": len(allc) - len(clean)}
        r["verdict"] = g_e55b(r["p25"]) if st == "PASS" else "NOT CLAIMED (interval unstable)"
        rows.append(r)

    log("== G-E55a / G-E55b -- the table ==")
    log("")
    log("   %-6s %-9s %5s %8s %8s %8s %7s %7s  %s"
        % ("N", "mean pos", "reps", "p25", "median", "p75", "IQR", "min-max", "verdict"))
    for r in rows:
        log("   %-6d %-9.0f %5d %8.2f %8.2f %8.2f %6.2f%% %6.2f%%  %s"
            % (r["n"], r["mean_pos"], r["used"], r["p25"], r["median"], r["p75"],
               r["iqr_pct"], r["minmax_pct"], r["verdict"]))
    log("")
    for r in rows:
        log("   N=%-5d G-E55a %-8s  %s" % (r["n"], r["stable"], r["stable_why"]))
    log("")
    log("   (IQR is the statistic that converges; min-max is printed beside it only to show")
    log("    how much larger the number this programme used to quote would have been.)")
    log("")

    cv, cwhy, cfit = g_e55c(rows)
    log("== G-E55c -- how far does it hold? ==")
    log("   %s : %s" % (cv, cwhy))
    log("")

    nb = sum(r["breached"] for r in rows)
    log("== G-E55d -- occupancy ==")
    for r in rows:
        log("   N=%-5d %2d of %2d cells above the %.2f%% bar, excluded from the quantiles;"
            " median with them = %.2f (without = %.2f)"
            % (r["n"], r["breached"], r["reps"], OCC_BAR, r["median_allcells"], r["median"]))
    log("")
    log("   RUN: %.1f minutes, %d cells (+1 discarded warm-up)"
        % (cells[-1]["since_start_s"] / 60.0, len(cells)))
    log("")
    for s in SCOPE:
        log("  " + s)

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    path = os.path.join(OUTDIR, "e55_table.json")
    with open(path, "w") as f:
        json.dump({"brief": "BRIEF_E55_THE_DEMONSTRATION_TABLE_WITH_AN_INTERVAL_THAT_SURVIVES.md",
                   "scope": SCOPE, "arm": ARM, "threads": THREADS, "plan": PLAN,
                   "seed": SEED, "occ_bar": OCC_BAR, "iqr_tol": IQR_TOL,
                   "warmup_discarded": w, "rows": rows, "cells": cells,
                   "G_E55c": {"verdict": cv, "why": cwhy, "fits": cfit},
                   "n_breached": nb,
                   "run_minutes": cells[-1]["since_start_s"] / 60.0}, f, indent=1)
    log("wrote %s" % path)


if __name__ == "__main__":
    main()
