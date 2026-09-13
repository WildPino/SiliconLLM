#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E47 -- is the cost of CONTEXT query FLOPs or KV bytes?

Brief: BRIEF_E46_WHAT_DOES_A_TOKEN_COST_AT_A_REAL_CONTEXT.md, ADDENDUM B  (f218682)
Pushed before these artifacts existed.  The gates are transcribed from B.3.

WHY.  E46 measured per-token cost linear in context at both scales and `G-E46c`
FAILED: A10B has FEWER KV bytes per position than S15 (NKV*HD*L 4,096 against
7,168) and a STEEPER slope (2.0307e-05 against 1.6276e-05).  Query FLOPs
(NH*HD*L, ratio 1.524) predicts the measured 1.248 far better than KV bytes
(0.571) -- but a two-term fit to two shapes has ZERO degrees of freedom and is
not a test.  E47 is the third, fourth and fifth shape.

  E47-BASE   NH 16  NKV 2     NH*HD*L 12288   NKV*HD*L 1536
  E47-QUERY  NH 32  NKV 2     NH*HD*L 24576   NKV*HD*L 1536    query x2, KV x1
  E47-KV     NH 16  NKV 8     NH*HD*L 12288   NKV*HD*L 6144    query x1, KV x4

D, F, L, HD, V and tied are identical across the three.  The position-0 term is
allowed to differ (the q/o projections change size with NH); E47 reads only the
SLOPE.

  python e47_head_counts.py --selftest
  python e47_head_counts.py
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e44_interval import Occupancy, log                              # noqa: E402
from e46_context_cost import (sweep, fit, predict_dt, g_e46a, g_e46b,  # noqa: E402
                              FIT_WINDOWS, TEST_WINDOW, REPS, THREADS)

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e47_head_counts.json")
ENGINE = os.path.join(HERE, "donor_engine_e26.exe")

ARMS = [("BASE", "D:/_ktmp/e47/e47_e47_base.bin", 16, 2),
        ("QUERY", "D:/_ktmp/e47/e47_e47_query.bin", 32, 2),
        ("KV", "D:/_ktmp/e47/e47_e47_kv.bin", 16, 8)]
HD, L = 64, 12

# addendum B, B.3 -- G-E47b
Q_QUERY_BOUND, V_QUERY_BOUND = 1.5, 1.5
V_KV_BOUND, Q_KV_BOUND = 2.5, 1.3
Q_MIXED, V_MIXED = 1.3, 1.5


def g_e47b(q, v):
    """B.3 verbatim.  q = b(QUERY)/b(BASE), v = b(KV)/b(BASE)."""
    if q >= Q_QUERY_BOUND and v <= V_QUERY_BOUND:
        r = ("QUERY-BOUND", "query heads doubled -> slope x%.2f, KV quadrupled -> x%.2f"
             % (q, v))
    elif v >= V_KV_BOUND and q <= Q_KV_BOUND:
        r = ("KV-BOUND", "KV quadrupled -> slope x%.2f, query doubled -> only x%.2f" % (v, q))
    elif q >= Q_MIXED and v >= V_MIXED:
        r = ("MIXED", "both terms move (q %.2f, v %.2f) -- the two-term model is now TESTED"
             % (q, v))
    else:
        r = ("NEITHER", "q %.2f and v %.2f fit no registered pattern -- B.1 is withdrawn"
             % (q, v))
    return {"q": q, "v": v, "verdict": r[0], "reason": r[1]}


def selftest():
    log("== E47 gate self-test -- G-E47b in all four directions ==")
    ok = [True]

    def check(name, got, want, note):
        good = (got == want)
        ok[0] = ok[0] and good
        log("  %-4s %-12s expected %-12s %s   %s"
            % (name, got, want, "FIRES" if good else "*** FAILS ***", note))

    check("Y1", g_e47b(1.95, 1.03)["verdict"], "QUERY-BOUND",
          "slope tracks query FLOPs and ignores KV")
    check("Y2", g_e47b(1.02, 3.80)["verdict"], "KV-BOUND",
          "slope tracks KV bytes and ignores query FLOPs")
    check("Y3", g_e47b(1.40, 2.00)["verdict"], "MIXED", "both terms real")
    check("Y4", g_e47b(1.01, 1.02)["verdict"], "NEITHER",
          "nothing moves -- the marginal cost is something else entirely")
    check("Y5", g_e47b(1.60, 2.60)["verdict"], "MIXED",
          "high on both is MIXED, not two verdicts at once")
    check("Y6", g_e47b(1.20, 1.20)["verdict"], "NEITHER",
          "a small rise on both is not evidence for either")
    log("  E47 gate self-test : %s" % ("ALL FIRE" if ok[0] else "*** SOMETHING FAILED ***"))
    if not ok[0]:
        raise SystemExit("the decision function does not behave as registered.  STOP.")


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
        raise SystemExit("E46's method registers >= %d reps.  STOP." % REPS)

    log("=" * 78)
    log("E47 -- is the cost of CONTEXT query FLOPs or KV bytes?")
    log("  Three shapes, identical but for the head counts.  E47 reads the SLOPE of")
    log("  per-token cost against context position, never a rate.")
    log("=" * 78)
    selftest()

    out = {"brief": "BRIEF_E46_WHAT_DOES_A_TOKEN_COST_AT_A_REAL_CONTEXT.md",
           "addendum": "B", "addendum_commit": "f218682", "engine": os.path.basename(ENGINE),
           "threads": THREADS, "reps": a.reps, "arms": {},
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    occ = Occupancy()
    occ.sample()

    for tag, path, nh, nkv in ARMS:
        if not os.path.exists(path):
            raise SystemExit("missing artifact %s.  STOP." % path)
        log("")
        log("== %s  (NH %d, NKV %d -- query %d, KV %d) =="
            % (tag, nh, nkv, nh * HD * L, nkv * HD * L))
        cells = sweep(ENGINE, path, [], FIT_WINDOWS + [TEST_WINDOW], a.reps, occ, tag)
        pts = [(cells[n]["mean_pos"], cells[n]["sec_per_tok"]) for n in FIT_WINDOWS]
        aa, bb = fit(pts)
        ga = g_e46a(cells[FIT_WINDOWS[0]]["sec_per_tok"], cells[FIT_WINDOWS[-1]]["sec_per_tok"])
        gb = g_e46b(predict_dt(aa, bb, TEST_WINDOW), cells[TEST_WINDOW]["median_dt_s"])
        log("  fit  s/tok = %.6f + %.4e * pos   ->  %.2f tok/s at position 0"
            % (aa, bb, 1.0 / aa))
        log("  G-E46a rise %.4f : %s   G-E46b dt(%d) pred %.3f obs %.3f rel %.4f : %s"
            % (ga["ratio"], "FIRES" if ga["fires"] else "*** FAILS ***", TEST_WINDOW,
               gb["predicted_dt"], gb["observed_dt"], gb["rel_error"],
               "SURVIVES" if gb["survives"] else "*** THE LAW IS WRONG ***"))
        out["arms"][tag] = {"weights": path, "NH": nh, "NKV": nkv,
                            "query_units": nh * HD * L, "kv_units": nkv * HD * L,
                            "cells": dict((str(n), cells[n]) for n in cells),
                            "a_sec_per_tok": aa, "b_sec_per_tok_per_pos": bb,
                            "G_E46a": ga, "G_E46b": gb}

    log("")
    log("== G-E47a -- the law must hold on EVERY arm, or no slope means anything ==")
    fires = all(out["arms"][t]["G_E46b"]["survives"] and out["arms"][t]["G_E46a"]["fires"]
                for t, _, _, _ in ARMS)
    for t, _, _, _ in ARMS:
        r = out["arms"][t]
        log("  %-6s G-E46a %s   G-E46b rel err %.4f %s"
            % (t, "FIRES" if r["G_E46a"]["fires"] else "FAILS", r["G_E46b"]["rel_error"],
               "SURVIVES" if r["G_E46b"]["survives"] else "FAILS"))
    log("  G-E47a : %s" % ("FIRES" if fires else "*** FAILS ***"))
    out["G_E47a"] = {"fires": bool(fires)}
    if not fires:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E47a did not fire: the linear law does not hold on every arm, so "
                         "no slope from them means anything.  STOP.")

    b0 = out["arms"]["BASE"]["b_sec_per_tok_per_pos"]
    q = out["arms"]["QUERY"]["b_sec_per_tok_per_pos"] / b0
    v = out["arms"]["KV"]["b_sec_per_tok_per_pos"] / b0
    gv = g_e47b(q, v)
    out["G_E47b"] = gv

    log("")
    log("== the slopes ==")
    log("  %-6s %6s %6s %12s %12s %14s" % ("arm", "NH", "NKV", "query", "KV", "b (s/tok/pos)"))
    for t, _, nh, nkv in ARMS:
        r = out["arms"][t]
        log("  %-6s %6d %6d %12d %12d %14.4e"
            % (t, nh, nkv, r["query_units"], r["kv_units"], r["b_sec_per_tok_per_pos"]))
    log("")
    log("  query heads x2 -> slope x %.3f   (query FLOPs predict 2.00, KV bytes predict 1.00)"
        % q)
    log("  KV heads    x4 -> slope x %.3f   (query FLOPs predict 1.00, KV bytes predict 4.00)"
        % v)
    log("")
    log("=" * 78)
    log("  G-E47b : %s" % gv["verdict"])
    log("  %s" % gv["reason"])
    log("")
    log("  E46's C50 = 575 does NOT depend on this answer -- it was measured, not derived.")
    log("  What this decides is WHICH lever moves it.")
    log("=" * 78)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.isdir(RES):
        os.makedirs(RES)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
