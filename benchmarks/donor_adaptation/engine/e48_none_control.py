#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E48 B.5 -- the adjudication: E46 says C50 = 575, E48 says 1454.  Which is it?

Brief: BRIEF_E48_WHERE_DOES_THE_CONTEXT_SLOPE_LIVE.md, ADDENDUM B section B.5  (9de5fe1)
Pushed before this file existed.  G-E48e is transcribed from B.5's table verbatim.

THE DISAGREEMENT.  Same engine, same weights, same --carve-k 3, same machine:

    E46 phase B   a = 8.327 ms   b = 0.02031 ms/pos   C50 =  575
    E48 sweep6    a = 8.107 ms   b = 0.00818 ms/pos   C50 = 1454

The intercepts agree to 2.6% and the slopes differ by 2.5x, so it is not machine
speed and not a level drift -- it is only the context term, which is the one thing
C50 is made of.

THE METHOD.  Pure `none`: no --sweep6, no --profile, nothing but --carve-k 3 and
--bench N, which is exactly what E46 ran.  ONE dataset over windows that contain
E46's three fit windows AND E48's four, then TWO fits of that one dataset.  If a
single straight line described the whole range, the two fits would agree; if they
do not, the disagreement was never about the instrument.

  python e48_none_control.py --selftest
  python e48_none_control.py
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
from e46_context_cost import fit                                     # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e48_none_control.json")
ENGINE = os.path.join(HERE, "donor_engine_e26.exe")
WEIGHTS = "D:/_ktmp/e40/e40_r128.bin"
FLAGS = ["--carve-k", "3"]
THREADS = 6

# ---- B.5 -------------------------------------------------------------------------
WINDOWS = [40, 160, 640, 1280, 2560]
SHORT_WINDOWS = [40, 160, 640]              # E46's fit windows
LONG_WINDOWS = [160, 640, 1280, 2560]       # E48's fit windows
REPS = 5
E46_B_MS_PER_POS = 0.020307                 # E46 phase B, s/tok/pos * 1000
E48_B_MS_PER_POS = 0.0081821                # E48 sweep6, whole token
G48E_TOL = 0.25                             # B.5, fixed before the numbers exist
TARGET_TOK_S = 50.0


def near(got, want, tol=G48E_TOL):
    return abs(got - want) / abs(want) <= tol


def g_e48e(b_short, b_long):
    """B.5's four-way table, verbatim.  The runner does not choose: every branch is
    written out and the data selects one."""
    s46, s48 = near(b_short, E46_B_MS_PER_POS), near(b_short, E48_B_MS_PER_POS)
    l46, l48 = near(b_long, E46_B_MS_PER_POS), near(b_long, E48_B_MS_PER_POS)
    if s46 and l48:
        v = ("THE LAW IS NOT LINEAR",
             "b_short reproduces E46 and b_long reproduces E48 from ONE dataset: both "
             "prior fits are locally right, C50 sits in the bend, and E46's linearity "
             "claim is scoped to short context")
    elif s46 and l46:
        v = ("THE SWEEP INSTRUMENT IS BIASED",
             "pure `none` reproduces E46 at BOTH fit ranges, so E48's shares are measured "
             "in a regime the engine never runs in and B.1 is withdrawn")
    elif s48 and l48:
        v = ("E46's PHASE B IS THE OUTLIER",
             "pure `none` reproduces E48 at BOTH fit ranges: C50 = 575 is not reproduced "
             "and must be re-measured")
    else:
        v = ("INCONCLUSIVE",
             "no branch of B.5's table matches; C50 carries both numbers and no verdict "
             "is drawn")
    return {"b_short_ms_per_pos": b_short, "b_long_ms_per_pos": b_long,
            "e46_b": E46_B_MS_PER_POS, "e48_b": E48_B_MS_PER_POS, "tol": G48E_TOL,
            "short_matches_e46": bool(s46), "short_matches_e48": bool(s48),
            "long_matches_e46": bool(l46), "long_matches_e48": bool(l48),
            "verdict": v[0], "reason": v[1]}


def c50(a_ms, b_ms):
    """The largest context at which the sustained rate is still >= 50 tok/s."""
    a, b = a_ms / 1000.0, b_ms / 1000.0
    if b <= 0:
        return None
    return (1.0 / TARGET_TOK_S - a) / b


def selftest():
    log("== E48 B.5 gate self-test -- every branch of G-E48e, on built data ==")
    ok = [True]

    def check(name, got, want, note):
        good = (got == want)
        ok[0] = ok[0] and good
        log("  %-4s %-28s expected %-28s %s   %s"
            % (name, got, want, "FIRES" if good else "*** FAILS ***", note))

    check("E1", g_e48e(E46_B_MS_PER_POS, E48_B_MS_PER_POS)["verdict"],
          "THE LAW IS NOT LINEAR", "the two prior answers, reproduced from one dataset")
    check("E2", g_e48e(E46_B_MS_PER_POS, E46_B_MS_PER_POS * 1.10)["verdict"],
          "THE SWEEP INSTRUMENT IS BIASED", "pure none is steep at BOTH ranges")
    check("E3", g_e48e(E48_B_MS_PER_POS * 0.95, E48_B_MS_PER_POS)["verdict"],
          "E46's PHASE B IS THE OUTLIER", "pure none is shallow at BOTH ranges")
    check("E4", g_e48e(E46_B_MS_PER_POS * 3.0, E48_B_MS_PER_POS * 0.2)["verdict"],
          "INCONCLUSIVE", "neither number is reproduced -> no verdict")
    check("E5", g_e48e(E48_B_MS_PER_POS, E46_B_MS_PER_POS)["verdict"],
          "INCONCLUSIVE", "the BACKWARDS shape -- shallow short, steep long -- has no "
                          "branch in B.5 and must not be forced into one")
    # the tolerance is a band, not a point: it must accept the edge and refuse past it
    check("E6", near(E46_B_MS_PER_POS * 1.24, E46_B_MS_PER_POS), True,
          "24% off is inside the registered 25%")
    check("E7", near(E46_B_MS_PER_POS * 1.26, E46_B_MS_PER_POS), False,
          "26% off is outside it")
    # the two reference numbers must not be within tolerance of EACH OTHER, or no branch
    # could ever be distinguished -- the gate would be unable to separate its own cases
    check("E8", near(E46_B_MS_PER_POS, E48_B_MS_PER_POS), False,
          "E46's b and E48's b must be DISTINGUISHABLE at this tolerance, or the gate "
          "cannot separate its own branches")
    # C50 must reproduce both published numbers from the published fits
    check("E9", "%.0f" % c50(8.327, E46_B_MS_PER_POS), "575", "E46's own C50")
    check("E10", "%.0f" % c50(8.107, E48_B_MS_PER_POS), "1454", "E48's own C50")
    log("  self-test : %s" % ("ALL FIRE" if ok[0] else "*** SOMETHING FAILED ***"))
    if not ok[0]:
        raise SystemExit("a decision function does not behave as registered.  STOP.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.reps < REPS:
        raise SystemExit("B.5 registers %d reps.  STOP." % REPS)

    log("=" * 82)
    log("E48 B.5 -- pure `none`, one dataset, two fits.  E46 says C50 = 575, E48 says 1454.")
    log("=" * 82)
    selftest()

    if not os.path.exists(WEIGHTS):
        raise SystemExit("missing artifact %s.  STOP." % WEIGHTS)
    out = {"brief": "BRIEF_E48_WHERE_DOES_THE_CONTEXT_SLOPE_LIVE.md", "addendum": "B",
           "addendum_commit": "9de5fe1", "engine": os.path.basename(ENGINE),
           "weights": WEIGHTS, "flags": FLAGS, "threads": THREADS, "reps": a.reps,
           "windows": WINDOWS, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    occ = Occupancy()
    occ.sample()

    cells = dict((w, {"window": w, "mean_pos": w / 2.0, "dt_s": [], "occ": []})
                 for w in WINDOWS)
    log("")
    log("  pure none -- no --sweep6, no --profile.  windows %s, %d reps, round robin"
        % (", ".join(str(w) for w in WINDOWS), a.reps))
    for i in range(a.reps):
        for w in WINDOWS:
            occ.sample()
            v, dt, wall = one_rep(ENGINE, WEIGHTS, w, THREADS, FLAGS)
            o = occ.sample()
            cells[w]["dt_s"].append(dt)
            cells[w]["occ"].append(o)
            log("    n=%-5d rep %d/%d  %7.2f tok/s  %8.4f ms/tok  dt %7.2fs  occ %4.1f%%"
                % (w, i + 1, a.reps, v, dt / w * 1e3, dt, o))
    for w in WINDOWS:
        c = cells[w]
        c["median_dt_s"] = statistics.median(c["dt_s"])
        c["ms_per_tok"] = c["median_dt_s"] / w * 1e3
        c["spread"] = (max(c["dt_s"]) - min(c["dt_s"])) / statistics.median(c["dt_s"])
        c["median_occ"] = statistics.median(c["occ"])

    pts = lambda ws: [(cells[w]["mean_pos"], cells[w]["ms_per_tok"]) for w in ws]
    a_s, b_s = fit(pts(SHORT_WINDOWS))
    a_l, b_l = fit(pts(LONG_WINDOWS))
    a_all, b_all = fit(pts(WINDOWS))
    gv = g_e48e(b_s, b_l)

    out["cells"] = dict((str(w), cells[w]) for w in WINDOWS)
    out["fit_short"] = {"windows": SHORT_WINDOWS, "a_ms": a_s, "b_ms_per_pos": b_s,
                        "c50": c50(a_s, b_s)}
    out["fit_long"] = {"windows": LONG_WINDOWS, "a_ms": a_l, "b_ms_per_pos": b_l,
                       "c50": c50(a_l, b_l)}
    out["fit_all"] = {"windows": WINDOWS, "a_ms": a_all, "b_ms_per_pos": b_all,
                      "c50": c50(a_all, b_all)}
    out["G_E48e"] = gv

    log("")
    log("  %-8s %10s %12s %14s %10s" % ("window", "mean pos", "ms/tok", "spread", "occ"))
    for w in WINDOWS:
        c = cells[w]
        log("  %-8d %10.1f %12.4f %13.1f%% %9.1f%%"
            % (w, c["mean_pos"], c["ms_per_tok"], 100 * c["spread"], c["median_occ"]))
    log("")
    log("  fit on E46's windows %s :  ms/tok = %.4f + %.6f*pos   -> C50 = %.0f"
        % (SHORT_WINDOWS, a_s, b_s, out["fit_short"]["c50"]))
    log("  fit on E48's windows %s :  ms/tok = %.4f + %.6f*pos   -> C50 = %.0f"
        % (LONG_WINDOWS, a_l, b_l, out["fit_long"]["c50"]))
    log("  fit on ALL windows      :  ms/tok = %.4f + %.6f*pos   -> C50 = %.0f"
        % (a_all, b_all, out["fit_all"]["c50"]))
    log("")
    log("  references:  E46 b = %.6f   E48 b = %.6f   tolerance +-%.0f%%"
        % (E46_B_MS_PER_POS, E48_B_MS_PER_POS, 100 * G48E_TOL))
    log("  b_short matches: E46 %s, E48 %s      b_long matches: E46 %s, E48 %s"
        % (gv["short_matches_e46"], gv["short_matches_e48"],
           gv["long_matches_e46"], gv["long_matches_e48"]))
    log("")
    log("=" * 82)
    log("  G-E48e : %s" % gv["verdict"])
    log("  %s" % gv["reason"])
    log("=" * 82)

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.isdir(RES):
        os.makedirs(RES)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
