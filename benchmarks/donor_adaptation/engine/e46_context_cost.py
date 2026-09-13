#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E46 -- what does a token cost at a REAL context?

Brief: BRIEF_E46_WHAT_DOES_A_TOKEN_COST_AT_A_REAL_CONTEXT.md  (faf1828)
Pushed before this file existed.  The four gates are transcribed from section 4 and
nothing here decides anything the brief did not register.

THE GAP.  `--bench N` decodes at positions 1..N, so the timing window and the context
length are the same knob.  NTOK = 40 in e40_levers_exhausted.py -- the file the 10B
headline comes from -- makes that headline a MEAN-POSITION-20 number, and so are
E36's, E37's, E39's, E43's and E45's.  This engine has never been measured at a
context a user would type.

  python e46_context_cost.py --selftest
  python e46_context_cost.py --phase a     # 1.5B, the out-of-sample test
  python e46_context_cost.py --phase b     # the target shape
  python e46_context_cost.py               # a then b then the table
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e44_interval import Occupancy, one_rep, log                     # noqa: E402
from e28_kernel_transfer import winpath                              # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e46_context_cost.json")
ENGINE_15B = os.path.join(HERE, "donor_engine_e26.exe")   # E37/E45's engine
ENGINE_10B = os.path.join(HERE, "donor_engine_e26.exe")   # E40's engine too
# ADDENDUM A: the 112.7 tok/s headline is R128 at --carve-k 3, read straight out
# of results/e40_levers_exhausted.json (speed_arms/R128_k3, mean 112.732, median
# 116.59).  It is NOT "the FFN fully on" -- k=3 of E=256 is 1.17% of it.  Phase B
# must therefore run the headline's OWN flags or it prices a different arm.
B_FLAGS = ["--carve-k", "3"]
E40_R128_K3_MEDIAN = 116.59
THREADS = 6

A_ARTIFACT = "D:/_ktmp/e37/e37_carved_nf.bin"
B_ARTIFACT = "D:/_ktmp/e40/e40_r128.bin"

# brief section 3
FIT_WINDOWS = [40, 160, 640]
TEST_WINDOW = 1280
REPS = 5
# brief section 4
G46A_RISE = 1.05          # dt(640)/640 must exceed dt(40)/40 by this
G46B_TOL = 0.10           # out-of-sample prediction of dt(1280)
G46C_TOL = 0.30           # slope at A10B vs b(S15) * 16/28
G46C_LAYER_RATIO = 16.0 / 28.0
G46D_TARGET = 50.0        # tok/s
G46D_CONTEXT = 2048       # the context the decision is read at
REPORT_CONTEXTS = [40, 512, 2048, 4096]


def fit(points):
    """Least squares of per-token cost against MEAN context position.  `points` is a
    list of (mean_position, seconds_per_token).  Returns (a, b) for s/tok = a + b*pos."""
    n = len(points)
    sx = sum(p for p, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(p * p for p, _ in points)
    sxy = sum(p * y for p, y in points)
    den = n * sxx - sx * sx
    if den == 0:
        raise SystemExit("degenerate fit -- every window at the same position.  STOP.")
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return a, b


def predict_dt(a, b, n):
    """dt(n) = sum over i=1..n of (a + b*i) = a*n + b*n*(n+1)/2."""
    return a * n + b * n * (n + 1) / 2.0


def rate_at_context(a, b, ctx):
    """The SUSTAINED rate at a fixed context: one token decoded with `ctx` behind it."""
    return 1.0 / (a + b * ctx)


def g_e46a(sec_per_tok_40, sec_per_tok_640):
    got = sec_per_tok_640 / sec_per_tok_40
    return {"ratio": got, "bar": G46A_RISE, "fires": bool(got > G46A_RISE)}


def g_e46b(pred, obs):
    rel = abs(obs - pred) / pred
    return {"predicted_dt": pred, "observed_dt": obs, "rel_error": rel, "tol": G46B_TOL,
            "survives": bool(rel <= G46B_TOL)}


def g_e46c(b_s15, b_a10b):
    want = b_s15 * G46C_LAYER_RATIO
    rel = abs(b_a10b - want) / want
    return {"b_s15": b_s15, "b_a10b": b_a10b, "expected": want, "rel_error": rel,
            "tol": G46C_TOL, "fires": bool(rel <= G46C_TOL)}


def g_e46d(a, b):
    """C50: the largest context at which the sustained rate is still >= 50 tok/s.
    a + b*ctx <= 1/50  ->  ctx <= (1/50 - a)/b."""
    if b <= 0:
        return {"c50": None, "verdict": "INCONCLUSIVE",
                "reason": "the fitted slope is not positive -- the law does not hold"}
    c50 = (1.0 / G46D_TARGET - a) / b
    if c50 < 0:
        return {"c50": 0.0, "verdict": "TARGET IS CONTEXT-LIMITED",
                "reason": "the target is not met even at context 0"}
    v = ("TARGET HOLDS AT REALISTIC CONTEXT" if c50 >= G46D_CONTEXT
         else "TARGET IS CONTEXT-LIMITED")
    return {"c50": c50, "target_tok_s": G46D_TARGET, "context_bar": G46D_CONTEXT,
            "verdict": v, "reason": "C50 = %.0f tokens against a bar of %d"
                                    % (c50, G46D_CONTEXT)}


def selftest():
    """Every gate must be shown to return each outcome it can return, on built data,
    before it is allowed near a cell.  Nothing here touches the engine."""
    log("== E46 gate self-test -- every gate in every direction ==")
    ok = [True]

    def check(name, got, want, note):
        good = (got == want)
        ok[0] = ok[0] and good
        log("  %-4s %-34s expected %-34s %s   %s"
            % (name, got, want, "FIRES" if good else "*** FAILS ***", note))

    # the fit itself, on data with a known answer
    a, b = fit([(20.0, 0.01 + 20 * 1e-5), (80.0, 0.01 + 80 * 1e-5),
                (320.0, 0.01 + 320 * 1e-5)])
    check("X1", "%.6f/%.2e" % (a, b), "%.6f/%.2e" % (0.01, 1e-5),
          "the fit recovers a planted line exactly")
    check("X2", "%.4f" % predict_dt(0.01, 0.0, 100), "%.4f" % 1.0,
          "dt of a flat cost is a*n")
    check("X3", "%.4f" % predict_dt(0.0, 2.0, 3), "%.4f" % 12.0,
          "dt of a pure slope is b*n*(n+1)/2")
    # G-E46a
    check("X4", g_e46a(0.0387, 0.0455)["fires"], True, "run 3's own rise, 17.6%")
    check("X5", g_e46a(0.0387, 0.0390)["fires"], False, "a flat engine -> refuses, as it must")
    # G-E46b
    check("X6", g_e46b(100.0, 105.0)["survives"], True, "5% out of sample")
    check("X7", g_e46b(100.0, 130.0)["survives"], False, "30% out -> the law is wrong")
    # G-E46c
    check("X8", g_e46c(1.454e-05, 1.454e-05 * 16 / 28)["fires"], True,
          "the slope scales exactly with layer count")
    check("X9", g_e46c(1.454e-05, 1.454e-05)["fires"], False,
          "no scaling at all -> the mechanism is not KV depth")
    # G-E46d, both directions
    check("X10", g_e46d(0.00885, 8.3e-06)["verdict"], "TARGET IS CONTEXT-LIMITED",
          "the brief's own desk arithmetic: C50 ~ 1343")
    check("X11", g_e46d(0.00885, 2.0e-06)["verdict"], "TARGET HOLDS AT REALISTIC CONTEXT",
          "a shallower slope clears 2048")
    check("X12", g_e46d(0.03, 8.3e-06)["verdict"], "TARGET IS CONTEXT-LIMITED",
          "below target even at context 0")
    log("  E46 gate self-test : %s" % ("ALL FIRE" if ok[0] else "*** SOMETHING FAILED ***"))
    if not ok[0]:
        raise SystemExit("a decision function does not behave as registered.  STOP.")


def sweep(engine, weights, flags, windows, reps, occ, label):
    """Round-robin over `windows`, reps outermost, so no window owns the end of the
    session.  Returns a cell dict per window."""
    cells = dict((n, {"ntok": n, "rates": [], "dt_s": [], "wall_s": [], "occupancy_pct": []})
                 for n in windows)
    log("  %s -- windows %s, %d reps, round robin"
        % (label, ", ".join(str(n) for n in windows), reps))
    for i in range(reps):
        for n in windows:
            occ.sample()
            v, dt, wall = one_rep(engine, weights, n, THREADS, flags)
            o = occ.sample()
            c = cells[n]
            c["rates"].append(v)
            c["dt_s"].append(dt)
            c["wall_s"].append(wall)
            c["occupancy_pct"].append(o)
            log("    n=%-5d rep %d/%d  %7.2f tok/s  dt %8.3f s  occ %4.1f%%  load %5.1fs"
                % (n, i + 1, reps, v, dt, o, wall - dt))
    for n in windows:
        c = cells[n]
        c["median_tok_s"] = statistics.median(c["rates"])
        c["median_dt_s"] = statistics.median(c["dt_s"])
        c["sec_per_tok"] = c["median_dt_s"] / n
        c["mean_pos"] = n / 2.0
        c["cv"] = (max(c["rates"]) - min(c["rates"])) / (sum(c["rates"]) / len(c["rates"]))
    return cells


def phase(engine, weights, flags, reps, occ, label, out):
    """Fit windows, then the out-of-sample window.  Returns the record."""
    if not os.path.exists(weights):
        raise SystemExit("missing artifact %s.  STOP." % weights)
    if not os.path.exists(engine):
        raise SystemExit("missing engine %s.  STOP." % engine)
    cells = sweep(engine, weights, flags, FIT_WINDOWS + [TEST_WINDOW], reps, occ, label)
    pts = [(cells[n]["mean_pos"], cells[n]["sec_per_tok"]) for n in FIT_WINDOWS]
    a, b = fit(pts)
    pred = predict_dt(a, b, TEST_WINDOW)
    rec = {"weights": weights, "engine": os.path.basename(engine), "flags": list(flags),
           "cells": dict((str(n), cells[n]) for n in cells),
           "fit_windows": FIT_WINDOWS, "a_sec_per_tok": a, "b_sec_per_tok_per_pos": b,
           "rate_at_pos0": (1.0 / a) if a > 0 else None,
           "G_E46a": g_e46a(cells[FIT_WINDOWS[0]]["sec_per_tok"],
                            cells[FIT_WINDOWS[-1]]["sec_per_tok"]),
           "G_E46b": g_e46b(pred, cells[TEST_WINDOW]["median_dt_s"])}
    log("  fit on %s:  s/tok = %.6f + %.4e * pos   ->  %.2f tok/s at position 0"
        % (FIT_WINDOWS, a, b, rec["rate_at_pos0"]))
    for n in FIT_WINDOWS:
        f = a + b * cells[n]["mean_pos"]
        log("    pos %6.0f  obs %.6f  fit %.6f  resid %+.1f%%"
            % (cells[n]["mean_pos"], cells[n]["sec_per_tok"], f,
               100.0 * (cells[n]["sec_per_tok"] - f) / f))
    ga, gb = rec["G_E46a"], rec["G_E46b"]
    log("  G-E46a  s/tok(%d)/s/tok(%d) = %.4f  bar %.2f : %s"
        % (FIT_WINDOWS[-1], FIT_WINDOWS[0], ga["ratio"], G46A_RISE,
           "FIRES" if ga["fires"] else "*** FAILS ***"))
    log("  G-E46b  dt(%d) predicted %.3f s, observed %.3f s, rel err %.4f (tol %.2f) : %s"
        % (TEST_WINDOW, gb["predicted_dt"], gb["observed_dt"], gb["rel_error"], G46B_TOL,
           "SURVIVES" if gb["survives"] else "*** THE LAW IS WRONG ***"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="ab", choices=["a", "b", "ab"])
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--a-weights", default=A_ARTIFACT)
    ap.add_argument("--b-weights", default=B_ARTIFACT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.reps < REPS:
        raise SystemExit("the brief registers >= %d reps.  %d is below it and the design is "
                         "not tunable downward.  STOP." % (REPS, a.reps))

    log("=" * 78)
    log("E46 -- what does a token cost at a REAL context?")
    log("  Every tok/s this programme has published was measured over 40 tokens, at a")
    log("  mean context position of 20.  This measures the cost as context grows.")
    log("=" * 78)
    selftest()

    out = {"brief": "BRIEF_E46_WHAT_DOES_A_TOKEN_COST_AT_A_REAL_CONTEXT.md",
           "brief_commit": "faf1828", "threads": THREADS, "reps": a.reps,
           "fit_windows": FIT_WINDOWS, "test_window": TEST_WINDOW,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    occ = Occupancy()
    occ.sample()

    if "a" in a.phase:
        log("")
        log("== PHASE A -- the linear law, out of sample, on REAL 1.5B weights ==")
        out["phase_a"] = phase(ENGINE_15B, winpath(a.a_weights), ["--carve-k", "256"],
                               a.reps, occ, "phase A (S15 carved, k=256)", out)
        if not out["phase_a"]["G_E46a"]["fires"]:
            json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
            raise SystemExit("G-E46a did NOT fire: the context rise run 3 saw is not there, so "
                             "run 3 was contention and E46 stops.  No null from an instrument "
                             "that has not fired.  STOP.")

    if "b" in a.phase:
        log("")
        log("== PHASE B -- the slope at the TARGET shape (A10B R128, k=3, synthetic) ==")
        log("  The headline arm itself: E40 read median %.2f tok/s here, at context 20."
            % E40_R128_K3_MEDIAN)
        log("  This is a SPEED measurement on noise.  E45's bridge is bounded under 5% at")
        log("  1.5B and is still not certified at 10B -- nothing here changes that.")
        out["phase_b"] = phase(ENGINE_10B, winpath(a.b_weights), B_FLAGS, a.reps, occ,
                               "phase B (A10B R128, FFN on)", out)

    if "a" in a.phase and "b" in a.phase:
        log("")
        log("== G-E46c -- does the slope scale with layer count? ==")
        gc = g_e46c(out["phase_a"]["b_sec_per_tok_per_pos"],
                    out["phase_b"]["b_sec_per_tok_per_pos"])
        out["G_E46c"] = gc
        log("  b(S15) %.4e  *16/28 = %.4e   b(A10B) %.4e   rel err %.4f (tol %.2f) : %s"
            % (gc["b_s15"], gc["expected"], gc["b_a10b"], gc["rel_error"], G46C_TOL,
               "FIRES" if gc["fires"] else "*** FAILS -- the mechanism is not KV depth ***"))

    if "b" in a.phase:
        pa = out["phase_b"]["a_sec_per_tok"]
        pb = out["phase_b"]["b_sec_per_tok_per_pos"]
        log("")
        log("== PHASE C -- the sustained rate at a real context (A10B R128) ==")
        log("  %10s %14s %14s" % ("context", "s/token", "tok/s"))
        tbl = []
        for ctx in REPORT_CONTEXTS:
            r = rate_at_context(pa, pb, ctx)
            tbl.append({"context": ctx, "sec_per_tok": pa + pb * ctx, "tok_s": r})
            log("  %10d %14.6f %14.1f" % (ctx, pa + pb * ctx, r))
        out["phase_c_table"] = tbl
        gd = g_e46d(pa, pb)
        out["G_E46d"] = gd
        log("")
        log("== G-E46d -- THE DECISION ==")
        log("  C50 = %s tokens (the largest context still at %.0f tok/s), bar %d"
            % ("%.0f" % gd["c50"] if gd["c50"] is not None else "n/a",
               G46D_TARGET, G46D_CONTEXT))
        log("  %s" % gd["verdict"])
        log("")
        log("  Phase C is an EXTRAPOLATION of phase B's fit, licensed only by G-E46b")
        log("  surviving out of sample at n=%d.  It is not a measured cell at 2048 or" % TEST_WINDOW)
        log("  4096 and must never be quoted as one.")

    log("")
    log("=" * 78)
    log("  E46 may not revise any published rate: those were measured, at context 20.")
    log("  The repair is to quote the context beside them.  Phase B is synthetic")
    log("  weights, so nothing here says anything about quality.")
    log("=" * 78)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.isdir(RES):
        os.makedirs(RES)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
