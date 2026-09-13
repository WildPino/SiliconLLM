#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 run 3 -- is the engine's dispersion, and its central value, an artifact of a
40-token measurement window?

Brief: BRIEF_E45_DOES_SPEED_DEPEND_ON_THE_WEIGHTS.md, ADDENDUM D  (f6a173a)
Pushed before this file existed.  The four gates below are transcribed from D.4 and
nothing here decides anything the addendum did not register.

WHY THIS EXISTS.  Run 2 came back INCONCLUSIVE on both arms with the occupancy
exclusion dropping almost nothing, so the noise is not contention and not drift --
it is E43's intrinsic per-cell oscillation, landing inside each cell and entering
the paired ratio twice.  And every cell in E45, and NTOK = 40 in
e40_levers_exhausted.py where the 10B headline comes from, measures a decode window
of 40 tokens: about 0.35 s at the R128 rate, squarely inside Zen 2 boost.

H-WINDOW: the 9-22% dispersion, and possibly the central value, are artifacts of a
0.35-1.6 s window.

  python e45_window.py --selftest     # the four gates against planted data
  python e45_window.py                # ~20-25 min, CPU only, no user action
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
OUT = os.path.join(RES, "e45_window.json")
ENGINE = os.path.join(HERE, "donor_engine_e26.exe")
THREADS = 6

# addendum D, D.4
WINDOWS_K256 = [40, 160, 640, 2560]
WINDOWS_K16 = [40, 2560]
REPS = 7
SHORT, LONG = 40, 2560
G45I_ARTIFACT, G45I_SCALEFREE = 0.5, 0.8       # cv(LONG) vs cv(SHORT)
G45J_BOOST = 0.95                              # m(LONG)/m(SHORT) below this is inflated
G45H_TOL = 0.15                                # dt(LONG)/dt(SHORT) around LONG/SHORT
# D.5, patched BEFORE any cell existed: without this cap G-E45j has exactly the
# defect addendum C found in G-E45c -- "inside my own dispersion" is passed by
# being noisy.  A long window whose cv exceeds this cannot answer the question.
G45J_CV_CAP = 0.10


def cv(v):
    """The relative spread this programme reports everywhere: (max - min) / mean."""
    return (max(v) - min(v)) / (sum(v) / len(v)) if v else float("nan")


def g_e45g(a_rates, z_rates):
    """DRIFT CONTROL.  The NTOK=40 block runs first and last; the two must agree to
    within the wider of their two cv's, or the sweep moved under itself."""
    ma, mz = statistics.median(a_rates), statistics.median(z_rates)
    rel = abs(mz - ma) / ((ma + mz) / 2.0)
    bar = max(cv(a_rates), cv(z_rates))
    return {"median_first": ma, "median_last": mz, "rel_gap": rel, "bar": bar,
            "fires": bool(rel <= bar)}


def g_e45h(dt_short, dt_long, short=SHORT, long_=LONG):
    """LINEARITY CONTROL.  --bench N must actually decode N tokens."""
    ms, ml = statistics.median(dt_short), statistics.median(dt_long)
    got = ml / ms if ms > 0 else float("inf")
    want = float(long_) / float(short)
    rel = abs(got - want) / want
    return {"dt_short_median": ms, "dt_long_median": ml, "ratio": got, "expected": want,
            "rel_error": rel, "tol": G45H_TOL, "fires": bool(rel <= G45H_TOL)}


def g_e45i(cv_short, cv_long):
    """DISPERSION.  Does measuring longer buy a tighter number?"""
    if cv_long <= G45I_ARTIFACT * cv_short:
        v = "SHORT-WINDOW ARTIFACT"
    elif cv_long >= G45I_SCALEFREE * cv_short:
        v = "SCALE-FREE"
    else:
        v = "PARTIAL"
    return {"cv_short": cv_short, "cv_long": cv_long,
            "ratio": (cv_long / cv_short) if cv_short else float("nan"), "verdict": v}


def g_e45j(m_short, m_long, cv_long):
    """CENTRAL VALUE.  Is the published rate a boost-clock rate?"""
    w = m_long / m_short
    d = abs(1.0 - w)
    if cv_long > G45J_CV_CAP:
        v = "INCONCLUSIVE"          # D.5: no verdict from a long window that is itself noisy
    elif w < G45J_BOOST and d > cv_long:
        v = "BOOST-INFLATED"
    elif d <= cv_long:
        v = "NO WINDOW EFFECT"
    else:
        v = "INCONCLUSIVE"
    return {"m_short": m_short, "m_long": m_long, "w": w, "abs_1_minus_w": d,
            "cv_long": cv_long, "cv_cap": G45J_CV_CAP, "verdict": v}


def selftest():
    """The planted-control law applies to the decision functions.  Each gate must be
    shown to return every outcome it can return, on data built to force it, before it
    is allowed near a cell.  Nothing here touches the engine."""
    log("== run-3 gate self-test -- every gate must fire in every direction ==")
    ok = [True]

    def check(name, got, want, note):
        good = (got == want)
        ok[0] = ok[0] and good
        log("  %-4s %-22s expected %-22s %s   %s"
            % (name, got, want, "FIRES" if good else "*** FAILS ***", note))

    # G-E45g
    check("W1", g_e45g([100, 101, 99], [100, 99, 101])["fires"], True,
          "first and last agree -> the drift control passes")
    check("W2", g_e45g([100, 101, 99], [70, 71, 69])["fires"], False,
          "the box drifted 30% across the sweep -> VOID, as it must")
    # G-E45h
    check("W3", g_e45h([1.0, 1.0, 1.0], [64.0, 64.0, 64.0])["fires"], True,
          "dt scales with the window")
    check("W4", g_e45h([1.0, 1.0, 1.0], [8.0, 8.0, 8.0])["fires"], False,
          "--bench N decoding a different N -> refuses")
    # G-E45i
    check("W5", g_e45i(0.20, 0.05)["verdict"], "SHORT-WINDOW ARTIFACT",
          "cv falls 4x with the window")
    check("W6", g_e45i(0.20, 0.19)["verdict"], "SCALE-FREE",
          "cv does not fall -- the BIGGER outcome")
    check("W7", g_e45i(0.20, 0.13)["verdict"], "PARTIAL",
          "between the two, reported as the curve")
    # G-E45j
    check("W8", g_e45j(113.0, 90.0, 0.03)["verdict"], "BOOST-INFLATED",
          "the long window reads 20% slower than the short one")
    check("W9", g_e45j(113.0, 112.0, 0.05)["verdict"], "NO WINDOW EFFECT",
          "the gap is inside the long window's own dispersion")
    # W10's first shape was mine and it was wrong, not the gate's: 113 -> 105 is
    # w = 0.929, below 0.95 and outside cv, which IS boost-inflated by the registered
    # rule.  INCONCLUSIVE needs a gap bigger than cv but shallower than the 5% bar,
    # or a LONG window that reads FASTER -- which is not "inflated" in either
    # direction and must not be called so.
    check("W10", g_e45j(113.0, 110.0, 0.02)["verdict"], "INCONCLUSIVE",
          "2.7% slower: outside cv but inside the 5% bar")
    check("W10b", g_e45j(113.0, 120.0, 0.02)["verdict"], "INCONCLUSIVE",
          "the LONG window reads FASTER -- resolved, but not inflation")
    # THE ONE THAT MATTERS, and it caught me repeating addendum C's own defect: as
    # first written, G-E45j called a 47% gap "NO WINDOW EFFECT" when cv was 0.90 --
    # passed by being noisy, exactly what C condemned.  D.5 caps it, before any cell.
    check("W11", g_e45j(113.0, 60.0, 0.90)["verdict"], "INCONCLUSIVE",
          "a 47% gap may NOT be swallowed by a cv of 0.90 (D.5's cap)")
    check("W12", g_e45j(113.0, 112.0, 0.11)["verdict"], "INCONCLUSIVE",
          "and the cap bites just above 0.10 even when the arms agree")
    log("  run-3 gate self-test : %s" % ("ALL FIRE" if ok[0] else "*** SOMETHING FAILED ***"))
    if not ok[0]:
        raise SystemExit("a decision function does not behave as registered.  STOP.")


def block(occ, weights, flags, ntok, reps, label):
    """`reps` invocations of one cell.  Returns rates, engine dt's, walls, occupancies."""
    rates, dts, walls, occs = [], [], [], []
    for i in range(reps):
        occ.sample()
        v, dt, wall = one_rep(ENGINE, weights, ntok, THREADS, flags)
        o = occ.sample()
        rates.append(v)
        dts.append(dt)
        walls.append(wall)
        occs.append(o)
        log("    %-14s rep %d/%d  %7.2f tok/s  dt %7.3f s  occ %4.1f%%  load %4.1fs"
            % (label, i + 1, reps, v, dt, o, wall - dt))
    return {"ntok": ntok, "rates": rates, "dt_s": dts, "wall_s": walls,
            "occupancy_pct": occs, "median_tok_s": statistics.median(rates),
            "min_tok_s": min(rates), "max_tok_s": max(rates), "cv": cv(rates)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e37")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.reps < REPS:
        raise SystemExit("addendum D D.4 registers >= %d reps per cell.  %d is below the "
                         "gate's own number and it is not tunable downward.  STOP."
                         % (REPS, a.reps))

    nf = os.path.join(winpath(a.dir), "e37_carved_nf.bin")
    if not os.path.exists(nf):
        raise SystemExit("missing %s.  STOP." % nf)

    log("=" * 78)
    log("E45 run 3 -- H-WINDOW: is the dispersion an artifact of a 40-token window?")
    log("  engine %s   threads %d   reps %d" % (os.path.basename(ENGINE), THREADS, a.reps))
    log("  arm NF (real weights) only.  This run does not compare NF against SYN and")
    log("  may not revise run 1 or run 2 (E36's run-2 rule).  It may not mint a rate.")
    log("=" * 78)
    selftest()

    out = {"brief": "BRIEF_E45_DOES_SPEED_DEPEND_ON_THE_WEIGHTS.md", "addendum": "D",
           "addendum_commit": "f6a173a", "run": 3, "weights": nf,
           "engine": os.path.basename(ENGINE), "threads": THREADS, "reps": a.reps,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": {}}

    occ = Occupancy()
    occ.sample()
    K256, K16 = ["--carve-k", "256"], ["--carve-k", "16"]

    # --- the SHORT block, first.  Its twin at the end is G-E45g's drift control.
    log("")
    log("== block A -- NTOK %d, first (G-E45g's first half) ==" % SHORT)
    out["cells"]["K256_40_first"] = block(occ, nf, K256, SHORT, a.reps, "K256 n40 A")
    out["cells"]["K16_40_first"] = block(occ, nf, K16, SHORT, a.reps, "K16  n40 A")

    # --- round robin over the remaining windows, so no window owns the end of the
    #     session.  Reps outermost, windows innermost.
    rr = [("K256_%d" % n, K256, n) for n in WINDOWS_K256 if n != SHORT] + \
         [("K16_%d" % n, K16, n) for n in WINDOWS_K16 if n != SHORT]
    for key, flags, n in rr:
        out["cells"][key] = {"ntok": n, "rates": [], "dt_s": [], "wall_s": [],
                             "occupancy_pct": []}
    log("")
    log("== round robin over %s, reps outermost ==" % ", ".join(k for k, _, _ in rr))
    for i in range(a.reps):
        for key, flags, n in rr:
            occ.sample()
            v, dt, wall = one_rep(ENGINE, nf, n, THREADS, flags)
            o = occ.sample()
            c = out["cells"][key]
            c["rates"].append(v)
            c["dt_s"].append(dt)
            c["wall_s"].append(wall)
            c["occupancy_pct"].append(o)
            log("    %-14s rep %d/%d  %7.2f tok/s  dt %7.3f s  occ %4.1f%%  load %4.1fs"
                % (key, i + 1, a.reps, v, dt, o, wall - dt))
    for key, _, _ in rr:
        c = out["cells"][key]
        c["median_tok_s"] = statistics.median(c["rates"])
        c["min_tok_s"], c["max_tok_s"] = min(c["rates"]), max(c["rates"])
        c["cv"] = cv(c["rates"])

    # --- the SHORT block again, last.
    log("")
    log("== block Z -- NTOK %d, last (G-E45g's second half) ==" % SHORT)
    out["cells"]["K256_40_last"] = block(occ, nf, K256, SHORT, a.reps, "K256 n40 Z")
    out["cells"]["K16_40_last"] = block(occ, nf, K16, SHORT, a.reps, "K16  n40 Z")

    # ------------------------------------------------------------------ gates
    log("")
    log("== G-E45g -- DRIFT CONTROL.  Nothing below counts until it fires ==")
    gg = {}
    for arm in ("K256", "K16"):
        g = g_e45g(out["cells"]["%s_40_first" % arm]["rates"],
                   out["cells"]["%s_40_last" % arm]["rates"])
        gg[arm] = g
        log("  %-5s first %7.2f  last %7.2f  gap %.4f  bar %.4f  : %s"
            % (arm, g["median_first"], g["median_last"], g["rel_gap"], g["bar"],
               "FIRES" if g["fires"] else "*** FAILS ***"))
    out["G_E45g"] = gg
    if not all(g["fires"] for g in gg.values()):
        out["verdict"] = {"run3": "VOID -- the sweep drifted under itself"}
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        log("")
        log("RUN 3 IS VOID: the NTOK=%d block does not reproduce across the sweep, so the" % SHORT)
        log("window curve is confounded by session-long drift.  No window conclusion is")
        log("drawn and none may be quoted.  (addendum D, G-E45g)")
        raise SystemExit(3)

    # pooled short block: 2 x reps samples of the same cell, first and last
    pooled = {}
    for arm in ("K256", "K16"):
        r = out["cells"]["%s_40_first" % arm]["rates"] + out["cells"]["%s_40_last" % arm]["rates"]
        d = out["cells"]["%s_40_first" % arm]["dt_s"] + out["cells"]["%s_40_last" % arm]["dt_s"]
        pooled[arm] = {"rates": r, "dt_s": d, "median": statistics.median(r), "cv": cv(r)}
    out["pooled_short"] = pooled

    log("")
    log("== G-E45h -- LINEARITY CONTROL ==")
    gh = {}
    for arm in ("K256", "K16"):
        h = g_e45h(pooled[arm]["dt_s"], out["cells"]["%s_%d" % (arm, LONG)]["dt_s"])
        gh[arm] = h
        log("  %-5s dt(%d)/dt(%d) = %.2f  expected %.0f  rel err %.4f (tol %.2f) : %s"
            % (arm, LONG, SHORT, h["ratio"], h["expected"], h["rel_error"], G45H_TOL,
               "FIRES" if h["fires"] else "*** FAILS ***"))
    out["G_E45h"] = gh
    if not all(h["fires"] for h in gh.values()):
        out["verdict"] = {"run3": "VOID -- --bench N does not decode N"}
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E45h did not fire: --bench N is not measuring what its name says, "
                         "so no window conclusion may be drawn from it.  STOP.")

    log("")
    log("== the curve ==")
    log("  %-6s %6s %10s %10s %10s %8s" % ("arm", "ntok", "median", "min", "max", "cv"))
    curve = {}
    for arm, wins in (("K256", WINDOWS_K256), ("K16", WINDOWS_K16)):
        curve[arm] = []
        for n in wins:
            if n == SHORT:
                row = {"ntok": n, "median": pooled[arm]["median"],
                       "min": min(pooled[arm]["rates"]), "max": max(pooled[arm]["rates"]),
                       "cv": pooled[arm]["cv"], "n_reps": len(pooled[arm]["rates"])}
            else:
                c = out["cells"]["%s_%d" % (arm, n)]
                row = {"ntok": n, "median": c["median_tok_s"], "min": c["min_tok_s"],
                       "max": c["max_tok_s"], "cv": c["cv"], "n_reps": len(c["rates"])}
            curve[arm].append(row)
            log("  %-6s %6d %10.2f %10.2f %10.2f %8.4f"
                % (arm, n, row["median"], row["min"], row["max"], row["cv"]))
    out["curve"] = curve

    log("")
    log("== G-E45i -- does measuring longer buy a tighter number? ==")
    gi, gj = {}, {}
    for arm in ("K256", "K16"):
        cs = pooled[arm]["cv"]
        cl = out["cells"]["%s_%d" % (arm, LONG)]["cv"]
        i_ = g_e45i(cs, cl)
        gi[arm] = i_
        log("  %-5s cv(%d) %.4f  cv(%d) %.4f  ratio %.3f  : %s"
            % (arm, SHORT, cs, LONG, cl, i_["ratio"], i_["verdict"]))
    out["G_E45i"] = gi

    log("")
    log("== G-E45j -- is the published rate a boost-clock rate? ==")
    for arm in ("K256", "K16"):
        j = g_e45j(pooled[arm]["median"],
                   out["cells"]["%s_%d" % (arm, LONG)]["median_tok_s"],
                   out["cells"]["%s_%d" % (arm, LONG)]["cv"])
        gj[arm] = j
        log("  %-5s m(%d) %7.2f  m(%d) %7.2f  w %.4f  |1-w| %.4f  cv(long) %.4f : %s"
            % (arm, SHORT, j["m_short"], LONG, j["m_long"], j["w"], j["abs_1_minus_w"],
               j["cv_long"], j["verdict"]))
    out["G_E45j"] = gj

    log("")
    log("=" * 78)
    log("E45 RUN 3 -- registered outcomes (addendum D, D.4)")
    for arm in ("K256", "K16"):
        log("  %-5s dispersion %-22s   central value %s"
            % (arm, gi[arm]["verdict"], gj[arm]["verdict"]))
    log("")
    log("  Run 3 does not answer E45's own question.  A favourable G-E45i only licenses")
    log("  a run 4 of the NF/SYN pairing at the window it identifies, with G-E45d")
    log("  unchanged.  No rate here may be quoted as a headline (brief section 5).")
    log("=" * 78)
    out["verdict"] = {"G_E45i": dict((k, v["verdict"]) for k, v in gi.items()),
                      "G_E45j": dict((k, v["verdict"]) for k, v in gj.items())}
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.isdir(RES):
        os.makedirs(RES)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
