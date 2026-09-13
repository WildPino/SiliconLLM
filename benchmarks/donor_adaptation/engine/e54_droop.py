#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E54 -- is the droop TIME, or is it the WORK that ran before it?

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E54_IS_THE_DROOP_TIME_OR_IS_IT_THE_WORK_BEFORE_IT.md`
Pre-registered and pushed before this file ran a single cell.

E53's ladder ascends, so its closing drift witness is preceded by ten n=1280 cells and its
opening one by an idle box: "9.7 minutes elapsed" and "ten heavy cells just finished" are the
same observation there, and addendum C.3 read it as the first.  This separates them by placing
an identical witness after blocks of EQUAL WALL TIME and different weight.

    W0 | HEAVY 180s | W1 | LIGHT 180s | W2 | HEAVY 180s | W3

Nothing here produces or improves a rate.  It decides whether the drift refusal can be replaced
by a correction, which every remaining speed experiment is currently blocked behind.
"""
import json
import os
import statistics
import subprocess
import sys
import time

import e44_interval as E44

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results")
E53 = os.path.join(HERE, "donor_engine_e53.exe")

R128 = "D:/_ktmp/e40/e40_r128.bin"
R128_FLAGS = ["--carve-k", "3"]
THREADS = 6
ARM = ["--fexp", "libm"]          # the control arm; E54 is about the box, not the kernel

W_N = 160                          # the witness cell -- identical to G-E53e's
W_REPS = 3                         # so each W carries its OWN dispersion (brief section 4)
HEAVY_N = 1280
LIGHT_N = 160
BLOCK_SECONDS = 180.0              # blocks bounded by TIME, never by cell count
OCC_BAR = E44.OCC_BAR


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


# ======================================================================================
def one_cell(n):
    """One engine invocation, with occupancy, timestamp and the clock witness."""
    sp = E44.Split()
    cmd = [E53, "--weights", R128, "--threads", str(THREADS)] + R128_FLAGS + ARM \
        + ["--bench", str(n)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sout, serr = p.communicate()
    child = E44._process_times(int(p._handle))
    if p.returncode != 0:
        sys.stderr.write(serr.decode("utf-8", "replace")[-1500:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    occ, foc = sp.close(child)
    m = E44.RE_BENCH.search(sout.decode("utf-8", "replace"))
    if not m:
        raise SystemExit("no BENCH line: " + " ".join(cmd))
    rec = sp.record()
    rec.update({"n": n, "rate": float(m.group(3)), "occ": occ, "foreign": foc})
    return rec


# ======================================================================================
def g_e54b(ws):
    """TIME or WORK -- the discriminator, registered in brief section 5.

    `ws` is four dicts, each {"med":, "spread":, "all":}.  The yardstick is each W's OWN
    min-max spread, never a tolerance of mine: the standing law is that a gate may not be
    tighter than the dispersion on the axis it watches.

      WORK          W2 > W1 by more than the larger of their two spreads
      TIME          W0 > W1 > W2 > W3 monotone
      UNRESOLVABLE  any W's spread exceeds the total W0 - W3 range
      NEITHER       anything else

    REPAIRED BEFORE ANY CELL RAN.  The brief registered TIME as "monotone AND W2 - W1 inside
    the spreads", which is self-contradictory: under a monotone decline W2 - W1 is negative and
    as large as the decline, so no genuinely monotone series could ever satisfy it -- the
    planted control B-1 caught it on the first run.  MALFORMED under the E4 precedent, not
    failed.  Monotonicity ALREADY excludes a recovery, so the clause was redundant at best;
    dropping it makes TIME strictly easier to reach and WORK unchanged, i.e. it relaxes the
    branch that CONTRADICTS my section 6 prediction and tightens nothing in its favour.  No
    measurement existed when this was changed.  See E54 addendum A.
    """
    med = [w["med"] for w in ws]
    sp_abs = [w["spread"] / 100.0 * w["med"] for w in ws]
    total = med[0] - med[3]

    if max(sp_abs) > abs(total):
        return ("UNRESOLVABLE", "the widest witness spread (%.2f tok/s) exceeds the whole W0-W3"
                " range (%.2f tok/s) -- this instrument cannot see an effect smaller than its"
                " own noise and will not guess" % (max(sp_abs), abs(total)))

    recov = med[2] - med[1]
    bar = max(sp_abs[1], sp_abs[2])
    if recov > bar:
        return ("WORK", "W2 - W1 = +%.2f tok/s (+%.2f%%), MORE than the larger of their spreads"
                " (%.2f tok/s): the witness RECOVERED across a light block of the same duration"
                " a heavy block costs, so the droop follows the WORKLOAD, not the clock"
                % (recov, 100.0 * recov / med[1], bar))

    monotone = med[0] > med[1] > med[2] > med[3]
    if monotone:
        return ("TIME", "W0 > W1 > W2 > W3 monotone (%.2f > %.2f > %.2f > %.2f); the light"
                " block did not bring the level back (W2 - W1 = %+.2f tok/s against a %.2f"
                " bar), so the droop followed elapsed time and not the workload"
                % (med[0], med[1], med[2], med[3], recov, bar))

    return ("NEITHER", "W0..W3 = %.2f %.2f %.2f %.2f, recovery %+.2f tok/s against a %.2f bar:"
            " neither the monotone pattern nor a recovery beyond the spreads"
            % (med[0], med[1], med[2], med[3], recov, bar))


def g_e54c(cells):
    """Does the CLOCK explain the droop?  SCORE, with g_e54b as its RANK partner (E14 s3).

    Refused when the predictor is noisier than the target: a clock reading that disperses more
    than the rate it is meant to predict explains nothing, however good the fit looks.
    """
    pts = [(c["clock_pct"], c["rate"]) for c in cells
           if c["clock_pct"] == c["clock_pct"] and c["n"] == W_N]
    if len(pts) < 6:
        return ("NOT COMPUTABLE", "only %d witness cells carried a clock reading" % len(pts),
                None)
    cl = [p[0] for p in pts]
    rt = [p[1] for p in pts]
    disp_cl = (max(cl) - min(cl)) / statistics.median(cl)
    disp_rt = (max(rt) - min(rt)) / statistics.median(rt)
    if disp_cl > disp_rt:
        return ("REFUSED", "the clock disperses %.2f%% and the rate it must predict only %.2f%%"
                " -- a predictor noisier than its target explains nothing"
                % (100 * disp_cl, 100 * disp_rt), None)
    mx, my = statistics.mean(cl), statistics.mean(rt)
    sxx = sum((x - mx) ** 2 for x in cl)
    if sxx <= 0:
        return ("NOT COMPUTABLE", "the clock did not vary across the witness cells", None)
    b = sum((x - mx) * (y - my) for x, y in zip(cl, rt)) / sxx
    a = my - b * mx
    pred = b * (max(cl) - min(cl))
    obs = max(rt) - min(rt)
    frac = pred / obs if obs else float("nan")
    return ("FIT", "rate = %.3f + %.3f * clock%%;  the clock's own %.2f-point range predicts"
            " %.2f tok/s of the %.2f tok/s observed = %.0f%% of the droop"
            % (a, b, max(cl) - min(cl), pred, obs, 100 * frac),
            {"a": a, "b": b, "explained_fraction": frac})


# ======================================================================================
def _w(vals):
    med = statistics.median(vals)
    return {"med": med, "spread": 100.0 * (max(vals) - min(vals)) / med, "all": vals}


def selftest():
    """Every decision function must FIRE on a known-positive before its nulls count."""
    ok = [0]

    def chk(name, cond):
        ok[0] += 1
        log("  %-48s %s" % (name, "fires" if cond else "*** DEAD ***"))
        if not cond:
            raise SystemExit("a decision function did not fire on a known-positive.  STOP.")

    # -- G-E54b ------------------------------------------------------------------------
    tight = lambda v: [v * 0.999, v, v * 1.001]          # ~0.2% spread
    mono = [_w(tight(v)) for v in (120.0, 117.0, 114.0, 111.0)]
    chk("B-1 a monotone series reads TIME", g_e54b(mono)[0] == "TIME")

    saw = [_w(tight(v)) for v in (120.0, 114.0, 119.0, 113.0)]
    chk("B-2 a sawtooth reads WORK", g_e54b(saw)[0] == "WORK")

    wide = [_w([v * 0.90, v, v * 1.10]) for v in (120.0, 119.5, 119.0, 118.5)]
    chk("B-3 noise wider than the effect is UNRESOLVABLE",
        g_e54b(wide)[0] == "UNRESOLVABLE")

    # a recovery of EXACTLY the spread must not tip the verdict to WORK
    w1 = _w([113.0, 114.0, 115.0])                        # spread = 2.0 tok/s
    w2 = _w([115.0, 116.0, 117.0])                        # med 116 = 114 + 2.0, exactly the bar
    edge = [_w(tight(120.0)), w1, w2, _w(tight(111.0))]
    chk("B-4 a recovery of exactly the spread does not tip", g_e54b(edge)[0] != "WORK")

    chk("B-5 a rise at the end is NEITHER, not TIME",
        g_e54b([_w(tight(v)) for v in (120.0, 117.0, 116.0, 119.0)])[0] == "NEITHER")

    flat = [_w(tight(120.0)) for _ in range(4)]
    chk("B-6 a flat series is UNRESOLVABLE, not TIME", g_e54b(flat)[0] == "UNRESOLVABLE")

    # -- G-E54c ------------------------------------------------------------------------
    planted = [{"n": W_N, "clock_pct": 100.0 + i, "rate": 50.0 + 2.0 * i} for i in range(8)]
    v, why, f = g_e54c(planted)
    chk("C-1 a planted slope is recovered", v == "FIT" and abs(f["b"] - 2.0) < 1e-6)
    chk("C-2 and it is scored as fully explained", abs(f["explained_fraction"] - 1.0) < 1e-6)

    noisy = [{"n": W_N, "clock_pct": 100.0 + 5.0 * (i % 2), "rate": 50.0 + 0.01 * (i % 2)}
             for i in range(8)]
    chk("C-3 a predictor noisier than its target is REFUSED",
        g_e54c(noisy)[0] == "REFUSED")

    chk("C-4 too few cells is NOT COMPUTABLE",
        g_e54c(planted[:3])[0] == "NOT COMPUTABLE")

    half = [{"n": W_N, "clock_pct": 100.0 + i, "rate": 50.0 + 1.0 * i + (3.0 if i == 7 else 0)}
            for i in range(8)]
    v, why, f = g_e54c(half)
    chk("C-5 partial explanation reports a fraction below 1",
        v == "FIT" and f["explained_fraction"] < 1.0)

    log("")
    log("  %d of %d fire." % (ok[0], ok[0]))


# ======================================================================================
def witness(tag, cells):
    vals = []
    for r in range(W_REPS):
        c = one_cell(W_N)
        c["block"] = tag
        cells.append(c)
        vals.append(c["rate"])
        log("    %-3s rep %d/%d  %7.2f tok/s  foreign %5.1f%%  clock %6.2f%%  t+%6.1fs%s"
            % (tag, r + 1, W_REPS, c["rate"], c["foreign"], c["clock_pct"], c["since_start_s"],
               "   <-- ABOVE THE BAR" if (c["foreign"] == c["foreign"]
                                          and c["foreign"] > OCC_BAR) else ""))
    w = _w(vals)
    log("    %-3s  median %7.2f tok/s   spread %5.2f%%" % (tag, w["med"], w["spread"]))
    log("")
    return w


def block(tag, n, cells):
    log("    %s block: --bench %d until %.0f s of wall time" % (tag, n, BLOCK_SECONDS))
    t0 = time.time()
    k = 0
    rates = []
    while time.time() - t0 < BLOCK_SECONDS:
        c = one_cell(n)
        c["block"] = tag
        cells.append(c)
        rates.append(c["rate"])
        k += 1
    log("    %s block done: %d cells, %.1f s, rate %.2f .. %.2f tok/s, clock %.2f .. %.2f%%"
        % (tag, k, time.time() - t0, min(rates), max(rates),
           min(c["clock_pct"] for c in cells if c["block"] == tag),
           max(c["clock_pct"] for c in cells if c["block"] == tag)))
    log("")
    return k


def main():
    log("E54 -- is the droop TIME, or is it the WORK that ran before it?")
    log("brief: BRIEF_E54_IS_THE_DROOP_TIME_OR_IS_IT_THE_WORK_BEFORE_IT.md")
    log("")
    log("== self-test.  Every decision function must fire on a known-positive. ==")
    selftest()
    if "--selftest" in sys.argv:
        return
    log("")
    log("== W0 | HEAVY %.0fs | W1 | LIGHT %.0fs | W2 | HEAVY %.0fs | W3 =="
        % (BLOCK_SECONDS, BLOCK_SECONDS, BLOCK_SECONDS))
    log("  Blocks are bounded by TIME, not by cell count: if the light block were shorter,")
    log("  elapsed time and block weight would be confounded again -- the defect being fixed.")
    log("")

    cells = []
    ws = []
    ws.append(witness("W0", cells))
    block("HEAVY-1", HEAVY_N, cells)
    ws.append(witness("W1", cells))
    block("LIGHT", LIGHT_N, cells)
    ws.append(witness("W2", cells))
    block("HEAVY-2", HEAVY_N, cells)
    ws.append(witness("W3", cells))

    bv, bwhy = g_e54b(ws)
    cv, cwhy, cfit = g_e54c(cells)
    breached = [c for c in cells if c["foreign"] == c["foreign"] and c["foreign"] > OCC_BAR]

    log("  G-E54b (TIME or WORK) : %s" % bv)
    log("     %s" % bwhy)
    log("")
    log("  G-E54c (does the clock explain it?) : %s" % cv)
    log("     %s" % cwhy)
    log("")
    log("  OCCUPANCY: %d of %d cells above the %.2f%% bar" % (len(breached), len(cells), OCC_BAR))
    log("  RUN       : %.1f minutes, %d cells" % (cells[-1]["since_start_s"] / 60.0, len(cells)))
    log("")

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    path = os.path.join(OUTDIR, "e54_droop.json")
    with open(path, "w") as f:
        json.dump({"brief": "BRIEF_E54_IS_THE_DROOP_TIME_OR_IS_IT_THE_WORK_BEFORE_IT.md",
                   "arm": R128_FLAGS + ARM, "threads": THREADS,
                   "W_n": W_N, "W_reps": W_REPS, "block_seconds": BLOCK_SECONDS,
                   "heavy_n": HEAVY_N, "light_n": LIGHT_N, "occ_bar": OCC_BAR,
                   "W": ws, "cells": cells,
                   "G_E54b": {"verdict": bv, "why": bwhy},
                   "G_E54c": {"verdict": cv, "why": cwhy, "fit": cfit},
                   "n_breached": len(breached),
                   "run_minutes": cells[-1]["since_start_s"] / 60.0}, f, indent=1)
    log("wrote %s" % path)


if __name__ == "__main__":
    main()
