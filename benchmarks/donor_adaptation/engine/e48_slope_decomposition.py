#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E48 part 2 -- where, inside the engine, does the context slope live?

Brief: BRIEF_E48_WHERE_DOES_THE_CONTEXT_SLOPE_LIVE.md  (275538f)
       ADDENDUM A -- instrument, shapes and the replacement for a vacuous gate  (2b88546)
Both pushed before this file existed.  Every gate here is transcribed from A.3 and
section 3.2; nothing in this file decides anything the brief did not register.

THE INSTRUMENT (addendum A.1).  `--sweep6 --profile` runs ten arms -- none, sm1/2/3,
av1/2/3, qk1/2/3 -- INSIDE ONE PROCESS, rotating by position on a palindrome of period
20 so every arm gets the same mean context position, and prints per-arm per-organ
ms/token.  The sub-organ terms are differences between the 2x and the 1x wrapper,
never against `none`:

    X = attention(qk2) - attention(qk1)      the Q.K loop
    S = attention(sm2) - attention(sm1)      the softmax pass
    Y = attention(av2) - attention(av1)      the A.V loop
    Rem = attention(none) - X - S - Y        everything else in the organ

Rem is a RESIDUAL BY DEFINITION, which is why the closure gate I first registered was
vacuous and was replaced (addendum A.3).  What can fail is the 3x test -- the second
increment must equal the first -- and the claim that no organ but attention has a
context slope.

  python e48_slope_decomposition.py --selftest
  python e48_slope_decomposition.py --shape r128
  python e48_slope_decomposition.py --shape s15
  python e48_slope_decomposition.py                 # both, then the table
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e44_interval import Occupancy, log                              # noqa: E402
from e46_context_cost import fit                                     # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e48_slope_decomposition.json")
ENGINE = os.path.join(HERE, "donor_engine_e26.exe")

# ---- addendum A.2 ----------------------------------------------------------------
WINDOWS = [160, 640, 1280, 2560]          # every one divisible by the period, 20
REPS = 5
THREADS = 6
SHAPES = {
    "r128": {"weights": "D:/_ktmp/e40/e40_r128.bin", "flags": ["--carve-k", "3"],
             "e46_b": 2.0307e-05, "label": "A10B R128 --carve-k 3 (the headline arm)"},
    "s15":  {"weights": "D:/_ktmp/e37/e37_carved_nf.bin", "flags": [],
             "e46_b": 1.6276e-05, "label": "S15, the real 1.5B"},
}
JUDGING_WINDOWS = WINDOWS[-2:]            # A.3: the 3x test is gated at the two largest

# ---- the gates, transcribed ------------------------------------------------------
G48A_RISE = 1.05                          # section 3.2
G48B1_TOL = 0.15                          # addendum A.3
G48B2_TOL = 0.15                          # addendum A.3

ORGANS = ["qkv_proj", "rope", "attention", "o_proj", "ffn", "head", "norm+glue"]
TERMS = ["X", "S", "Y", "Rem"]
RE_SWEEP = re.compile(r"^SWEEP (\S+) n=(\d+) (.*)$", re.M)
RE_BENCH = re.compile(r"^BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s", re.M)


# ---- measurement -----------------------------------------------------------------
def one_sweep(engine, weights, flags, ntok, threads):
    """One --sweep6 --profile process.  Returns {arm: {organ: ms_per_token}}, dt, wall."""
    if ntok % 20:
        raise SystemExit("window %d is not divisible by the schedule period 20.  STOP." % ntok)
    cmd = [engine, "--weights", weights, "--threads", str(threads)] + list(flags) + \
          ["--sweep6", "--profile", "--bench", str(ntok)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    wall = time.time() - t0
    if r.returncode != 0:
        log(r.stderr.decode(errors="replace")[-1200:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    txt = r.stdout.decode(errors="replace")
    m = RE_BENCH.search(txt)
    if not m:
        log(txt[-1200:])
        raise SystemExit("could not parse the BENCH line")
    arms = {}
    for name, n, rest in RE_SWEEP.findall(txt):
        d = {}
        for kv in rest.split():
            k, _, v = kv.partition("=")
            d[k] = float(v)
        d["_n"] = int(n)
        d["_token_ms"] = sum(d[o] for o in ORGANS if o in d)
        arms[name] = d
    missing = [a for a in ("none", "qk1", "qk2", "qk3", "sm1", "sm2", "sm3",
                           "av1", "av2", "av3") if a not in arms]
    if missing:
        raise SystemExit("the sweep did not report %s -- wrong engine build?  STOP."
                         % ", ".join(missing))
    return arms, float(m.group(2)), wall


def terms_from(arms):
    """A.1's differences.  Rem is the residual and is not independent."""
    X = arms["qk2"]["attention"] - arms["qk1"]["attention"]
    S = arms["sm2"]["attention"] - arms["sm1"]["attention"]
    Y = arms["av2"]["attention"] - arms["av1"]["attention"]
    organ = arms["none"]["attention"]
    return {"X": X, "S": S, "Y": Y, "Rem": organ - X - S - Y, "organ": organ,
            "token_ms": arms["none"]["_token_ms"],
            "X2": arms["qk3"]["attention"] - arms["qk2"]["attention"],
            "S2": arms["sm3"]["attention"] - arms["sm2"]["attention"],
            "Y2": arms["av3"]["attention"] - arms["av2"]["attention"]}


def measure(shape_key, reps, occ):
    sh = SHAPES[shape_key]
    if not os.path.exists(sh["weights"]):
        raise SystemExit("missing artifact %s.  STOP." % sh["weights"])
    log("")
    log("== %s ==" % sh["label"])
    log("  windows %s, %d reps, round robin with reps outermost"
        % (", ".join(str(w) for w in WINDOWS), reps))
    per = dict((w, []) for w in WINDOWS)
    organ_per = dict((w, dict((o, dict((a, []) for a in
                                       ("none", "qk1", "sm1", "av1"))) for o in ORGANS))
                     for w in WINDOWS)
    for i in range(reps):
        for w in WINDOWS:
            occ.sample()
            arms, dt, wall = one_sweep(ENGINE, sh["weights"], sh["flags"], w, THREADS)
            o = occ.sample()
            t = terms_from(arms)
            per[w].append(t)
            for org in ORGANS:
                for a in ("none", "qk1", "sm1", "av1"):
                    organ_per[w][org][a].append(arms[a].get(org, 0.0))
            log("    n=%-5d rep %d/%d  organ %7.3f  X %6.3f  S %6.3f  Y %6.3f  Rem %6.3f"
                "  tok %7.3f ms  dt %7.2fs occ %4.1f%%"
                % (w, i + 1, reps, t["organ"], t["X"], t["S"], t["Y"], t["Rem"],
                   t["token_ms"], dt, o))
    cells = {}
    for w in WINDOWS:
        c = {"window": w, "mean_pos": w / 2.0, "reps": len(per[w])}
        for k in ("X", "S", "Y", "Rem", "organ", "token_ms", "X2", "S2", "Y2"):
            vals = [t[k] for t in per[w]]
            c[k] = statistics.median(vals)
            c[k + "_spread"] = (max(vals) - min(vals)) / abs(statistics.median(vals)) \
                if statistics.median(vals) else float("inf")
        c["organs_none"] = dict((o, statistics.median(organ_per[w][o]["none"]))
                                for o in ORGANS)
        cells[w] = c
    return cells


# ---- the gates -------------------------------------------------------------------
def g_e48a(cells):
    """Planted control: each priced term must be POSITIVE everywhere and must RISE with
    context.  A sub-organ probe that cannot see context in an O(pos) loop is not
    measuring the loop."""
    out, fires = {}, True
    lo, hi = WINDOWS[0], WINDOWS[-1]
    for t in ("X", "S", "Y"):
        pos = all(cells[w][t] > 0 for w in WINDOWS)
        ratio = (cells[hi][t] / cells[lo][t]) if cells[lo][t] > 0 else 0.0
        ok = bool(pos and ratio > G48A_RISE)
        out[t] = {"positive_everywhere": bool(pos), "rise": ratio, "bar": G48A_RISE,
                  "fires": ok}
        fires = fires and ok
    out["fires"] = bool(fires)
    return out


def g_e48b1(cells):
    """The 3x test.  Both increments are ONE extra pass of the same loop, so they must be
    equal.  Three predictions per window; gated at the two largest windows only."""
    out, fires = {"tol": G48B1_TOL, "gated_windows": JUDGING_WINDOWS}, True
    for w in WINDOWS:
        row = {}
        for t in ("X", "S", "Y"):
            i1, i2 = cells[w][t], cells[w][t + "2"]
            rel = abs(i2 - i1) / abs(i1) if i1 else float("inf")
            ok = bool(rel <= G48B1_TOL)
            row[t] = {"inc1": i1, "inc2": i2, "ratio": (i2 / i1) if i1 else None,
                      "rel_error": rel, "passes": ok}
            if w in JUDGING_WINDOWS:
                fires = fires and ok
        out[str(w)] = row
    out["fires"] = bool(fires)
    return out


def g_e48b2(slope_token, slope_organ):
    """Nothing but attention may have a context slope.  If this fails, the structural
    reading of donor_engine.c in section 1 is false and part 1 goes with it."""
    rel = abs(slope_token - slope_organ) / abs(slope_organ) if slope_organ else float("inf")
    return {"slope_token_ms_per_pos": slope_token, "slope_organ_ms_per_pos": slope_organ,
            "rel_error": rel, "tol": G48B2_TOL, "fires": bool(rel <= G48B2_TOL)}


def g_e48d(shares_a, shares_b):
    """The RANK partner E14 s3 requires for the SCORE in G-E48c.  If the ordering is not
    stable across shapes it is a property of the shape, and no lever may be prioritised
    from it."""
    oa = sorted(TERMS, key=lambda t: -shares_a[t])
    ob = sorted(TERMS, key=lambda t: -shares_b[t])
    return {"order_a": oa, "order_b": ob, "fires": bool(oa == ob)}


def analyse(shape_key, cells):
    sh = SHAPES[shape_key]
    pts = lambda k: [(cells[w]["mean_pos"], cells[w][k]) for w in WINDOWS]
    slopes, intercepts = {}, {}
    for k in TERMS + ["organ", "token_ms"]:
        a, b = fit(pts(k))
        slopes[k], intercepts[k] = b, a
    tot = slopes["organ"]
    shares = dict((t, slopes[t] / tot) for t in TERMS)
    rec = {"shape": shape_key, "label": sh["label"], "weights": sh["weights"],
           "flags": sh["flags"], "windows": WINDOWS, "reps": REPS,
           "cells": dict((str(w), cells[w]) for w in WINDOWS),
           "slopes_ms_per_pos": slopes, "intercepts_ms": intercepts,
           "shares_of_organ_slope": shares,
           "G_E48a": g_e48a(cells), "G_E48b1": g_e48b1(cells),
           "G_E48b2": g_e48b2(slopes["token_ms"], slopes["organ"])}
    # A.4 -- a DRIFT LINE, never a gate and never a correction to E46
    rec["drift_vs_e46"] = {
        "e46_b_sec_per_tok_per_pos": sh["e46_b"],
        "e48_b_sec_per_tok_per_pos": slopes["token_ms"] / 1000.0,
        "ratio_e48_over_e46": (slopes["token_ms"] / 1000.0) / sh["e46_b"],
        "note": "recorded under A.4.  A share travels between sweeps; a millisecond "
                "does not.  This is NOT a gate and NOT a correction to E46."}
    return rec


def report(rec):
    s, sl = rec["shares_of_organ_slope"], rec["slopes_ms_per_pos"]
    log("")
    log("  -- %s --" % rec["label"])
    log("  %-22s %14s %14s %10s" % ("term", "slope ms/pos", "intercept ms", "share"))
    for t in TERMS:
        log("  %-22s %14.4e %14.4f %9.1f%%"
            % ({"X": "X   Q.K loop", "S": "S   softmax", "Y": "Y   A.V loop",
                "Rem": "Rem non-loop resid"}[t], sl[t], rec["intercepts_ms"][t],
               100 * s[t]))
    log("  %-22s %14.4e %14.4f %9.1f%%"
        % ("attention organ", sl["organ"], rec["intercepts_ms"]["organ"], 100.0))
    log("  %-22s %14.4e %14.4f" % ("whole token", sl["token_ms"],
                                   rec["intercepts_ms"]["token_ms"]))
    ga, g1, g2 = rec["G_E48a"], rec["G_E48b1"], rec["G_E48b2"]
    log("")
    log("  G-E48a  planted control : %s" % ("FIRES" if ga["fires"] else "*** FAILS ***"))
    for t in ("X", "S", "Y"):
        log("     %-3s positive %-5s  rise %6.2fx (bar %.2f)  %s"
            % (t, ga[t]["positive_everywhere"], ga[t]["rise"], G48A_RISE,
               "fires" if ga[t]["fires"] else "*** FAILS ***"))
    log("  G-E48b1 the 3x test     : %s   (gated at %s)"
        % ("PASSES" if g1["fires"] else "*** FAILS ***",
           ", ".join(str(w) for w in JUDGING_WINDOWS)))
    for w in WINDOWS:
        log("     n=%-5d %s" % (w, "  ".join(
            "%s inc %6.3f/%6.3f = %5.2fx %s" % (t, g1[str(w)][t]["inc1"],
                                                g1[str(w)][t]["inc2"],
                                                g1[str(w)][t]["ratio"] or 0.0,
                                                "ok" if g1[str(w)][t]["passes"] else "MISS")
            for t in ("X", "S", "Y"))))
    log("  G-E48b2 only attention has a slope : %s   token %.4e vs organ %.4e, %.1f%% apart"
        % ("FIRES" if g2["fires"] else "*** FAILS ***", g2["slope_token_ms_per_pos"],
           g2["slope_organ_ms_per_pos"], 100 * g2["rel_error"]))
    d = rec["drift_vs_e46"]
    log("  drift line (A.4, NOT a gate) : E48 b = %.4e vs E46 b = %.4e   ratio %.3f"
        % (d["e48_b_sec_per_tok_per_pos"], d["e46_b_sec_per_tok_per_pos"],
           d["ratio_e48_over_e46"]))
    log("  organs of `none` at the largest window (ms/token):")
    log("     " + "  ".join("%s %.3f" % (o, rec["cells"][str(WINDOWS[-1])]["organs_none"][o])
                            for o in ORGANS))


# ---- self-test: every gate, in every direction, before any cell ------------------
def selftest():
    log("== E48 gate self-test -- every decision function in every direction ==")
    ok = [True]

    def check(name, got, want, note):
        good = (got == want)
        ok[0] = ok[0] and good
        log("  %-4s %-14s expected %-14s %s   %s"
            % (name, str(got), str(want), "FIRES" if good else "*** FAILS ***", note))

    def cellset(x, s, y, rise=4.0, x2=None, s2=None, y2=None):
        """Four windows with a planted linear rise; the 3x increments default to equal."""
        c = {}
        for i, w in enumerate(WINDOWS):
            f = 1.0 + (rise - 1.0) * i / (len(WINDOWS) - 1.0)
            c[w] = {"window": w, "mean_pos": w / 2.0,
                    "X": x * f, "S": s * f, "Y": y * f, "Rem": 1.0 * f,
                    "organ": (x + s + y + 1.0) * f, "token_ms": (x + s + y + 1.0) * f,
                    "X2": (x2 if x2 is not None else x) * f,
                    "S2": (s2 if s2 is not None else s) * f,
                    "Y2": (y2 if y2 is not None else y) * f}
        return c

    # G-E48a, both directions
    check("A1", g_e48a(cellset(1.0, 2.0, 1.5))["fires"], True,
          "three positive terms rising 4x with context")
    check("A2", g_e48a(cellset(1.0, 2.0, 1.5, rise=1.0))["fires"], False,
          "a flat probe cannot see an O(pos) loop -> must refuse")
    check("A3", g_e48a(cellset(-0.2, 2.0, 1.5))["fires"], False,
          "a NEGATIVE term means the 2x arm was faster than the 1x -- refuse")
    # G-E48b1, both directions
    check("B1", g_e48b1(cellset(1.0, 2.0, 1.5))["fires"], True,
          "equal increments -- the wrappers are doing one extra pass each")
    check("B2", g_e48b1(cellset(1.0, 2.0, 1.5, s2=3.0))["fires"], False,
          "sm's second increment 50% off its first -> the arm is not one extra pass")
    check("B3", g_e48b1(cellset(1.0, 2.0, 1.5, y2=1.6))["fires"], True,
          "6.7% off is inside the registered 15%")
    # the window-160 carve-out: a miss there must NOT sink the gate
    cs = cellset(1.0, 2.0, 1.5)
    cs[WINDOWS[0]]["S2"] = 99.0
    check("B4", g_e48b1(cs)["fires"], True,
          "a miss at the SMALLEST window is reported, not gated (A.3)")
    check("B5", g_e48b1(cs)[str(WINDOWS[0])]["S"]["passes"], False,
          "...and it is still recorded as a miss")
    # G-E48b2, both directions
    check("C1", g_e48b2(1.00e-2, 1.05e-2)["fires"], True,
          "token and organ slopes agree to 4.8% -- attention owns the slope")
    check("C2", g_e48b2(2.00e-2, 1.00e-2)["fires"], False,
          "the token rises twice as fast as attention -> something else has a slope")
    # G-E48d, both directions
    check("D1", g_e48d({"X": .2, "S": .3, "Y": .2, "Rem": .3},
                       {"X": .1, "S": .4, "Y": .1, "Rem": .4})["fires"], True,
          "same ordering at different magnitudes -> the rank travels")
    check("D2", g_e48d({"X": .4, "S": .2, "Y": .2, "Rem": .2},
                       {"X": .1, "S": .5, "Y": .2, "Rem": .2})["fires"], False,
          "the largest term changes with the shape -> no lever may be ranked")
    # the fit, on a planted line
    a, b = fit([(80.0, 1.0 + 80 * 2e-3), (320.0, 1.0 + 320 * 2e-3),
                (640.0, 1.0 + 640 * 2e-3), (1280.0, 1.0 + 1280 * 2e-3)])
    check("F1", "%.5f/%.3e" % (a, b), "%.5f/%.3e" % (1.0, 2e-3),
          "the fit recovers a planted line over E48's own windows")
    log("  E48 gate self-test : %s" % ("ALL FIRE" if ok[0] else "*** SOMETHING FAILED ***"))
    if not ok[0]:
        raise SystemExit("a decision function does not behave as registered.  STOP.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=sorted(SHAPES) + ["both"], default="both")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.reps < REPS:
        raise SystemExit("addendum A.2 registers %d reps.  STOP." % REPS)

    log("=" * 82)
    log("E48 part 2 -- where, inside the engine, does the context slope live?")
    log("  --sweep6 --profile: ten bit-identical arms in ONE process, palindrome period 20.")
    log("  Shares are the result; milliseconds are the arithmetic (addendum A.4).")
    log("=" * 82)
    selftest()

    out = {"brief": "BRIEF_E48_WHERE_DOES_THE_CONTEXT_SLOPE_LIVE.md",
           "brief_commit": "275538f", "addendum": "A", "addendum_commit": "2b88546",
           "engine": os.path.basename(ENGINE), "threads": THREADS, "reps": a.reps,
           "windows": WINDOWS, "shapes": {},
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    occ = Occupancy()
    occ.sample()

    keys = sorted(SHAPES) if a.shape == "both" else [a.shape]
    for k in keys:
        rec = analyse(k, measure(k, a.reps, occ))
        report(rec)
        out["shapes"][k] = rec
        if not os.path.isdir(RES):
            os.makedirs(RES)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)

    if len(keys) == 2:
        ra, rb = out["shapes"][keys[0]], out["shapes"][keys[1]]
        gd = g_e48d(ra["shares_of_organ_slope"], rb["shares_of_organ_slope"])
        out["G_E48d"] = gd
        log("")
        log("=" * 82)
        log("  G-E48c -- the shares (a SCORE; its RANK partner is G-E48d)")
        log("  %-22s %12s %12s" % ("term", keys[0], keys[1]))
        for t in TERMS:
            log("  %-22s %11.1f%% %11.1f%%"
                % (t, 100 * ra["shares_of_organ_slope"][t],
                   100 * rb["shares_of_organ_slope"][t]))
        log("")
        log("  G-E48d -- does the ordering survive the change of shape?")
        log("     %-6s %s" % (keys[0], " > ".join(gd["order_a"])))
        log("     %-6s %s" % (keys[1], " > ".join(gd["order_b"])))
        log("     %s" % ("FIRES -- the rank is a property of the ENGINE" if gd["fires"]
                         else "*** FAILS *** -- the rank is a property of the SHAPE, "
                              "and no lever may be prioritised from it"))
        log("=" * 82)

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
