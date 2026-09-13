#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E51 -- the softmax was paying for errno.

Brief: BRIEF_E51_THE_SOFTMAX_WAS_PAYING_FOR_ERRNO.md  (c67daee)
Pushed before this file existed.  Every gate is transcribed from section 4.

One flag, -fno-math-errno, and nothing else.  It is NOT -ffast-math: no
reassociation, no FTZ/DAZ, no reciprocal substitution, no change to
-ffp-contract=on.  It removes the promise to set errno, which is what makes
`exp` a memory-writing call and blocks hoisting, unrolling and vectorisation.

  BASE  donor_engine_e50.exe   -O3 -mavx2 -mfma -ffp-contract=on -fopenmp
  E51   donor_engine_e51.exe   the same, plus -fno-math-errno

expf(x) and (float)exp((double)x) are DIFFERENT functions, so this is not
bit-exact and is not gated as if it were.

  python e51_math_errno.py --selftest
  python e51_math_errno.py --phase parity     # G-E51a      (~20 min)
  python e51_math_errno.py --phase greedy     # G-E51b      (~10 min)
  python e51_math_errno.py --phase speed      # G-E51c      (~35 min)
  python e51_math_errno.py --phase attrib     # G-E51d
  python e51_math_errno.py                    # all four, in that order
"""
import argparse
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e44_interval import Occupancy, log                              # noqa: E402
from e46_context_cost import fit                                     # noqa: E402

RES = os.path.join(HERE, "results")
E6_RES = os.path.join(RES, "e6")
E51_RES = os.path.join(RES, "e51")
OUT = os.path.join(RES, "e51_math_errno.json")
TMP = r"D:\_ktmp\e51"

BASE_ENGINE = os.path.join(HERE, "donor_engine_e50.exe")     # the current default build
E51_ENGINE = os.path.join(HERE, "donor_engine_e51.exe")      # + -fno-math-errno
BUILDS = [("base", BASE_ENGINE), ("e51", E51_ENGINE)]
THREADS = 6

# ---- G-E51a ----------------------------------------------------------------------
S15 = "D:/_ktmp/e37/e37_carved_nf.bin"
PARITY_K = ["--carve-k", "256"]
CONTROL_K = ["--carve-k", "3"]
IDS = "D:/_ktmp/e1/ids_qwen25-15b_tq.bin"
SEQLEN = 512
G51A_BPB_BAR, G51A_BPB_UPPER_BOUND = 1e-4, 5.0
G51A_REL_BAR = G51A_BPB_BAR / G51A_BPB_UPPER_BOUND           # 2e-05
G51A_CONTROL_MIN = 1e-3

# ---- G-E51b ----------------------------------------------------------------------
E6_ARMS = [
    ("A1", r"D:\_ktmp\e1\qwen25-05b_f32.bin", "Qwen/Qwen2.5-0.5B", "known-positive (fp32)"),
    ("A2", r"D:\_ktmp\e1\qwen25-05b_tqh.bin", "Qwen/Qwen2.5-0.5B", "planted control (ternary+head)"),
    ("A3", r"D:\_ktmp\e1\qwen25-15b_tqh.bin", "Qwen/Qwen2.5-1.5B", "planted control at 1.5B"),
]
G51B_A1_REQUIRED = 160
G51B_CONTROL_MAX = 0.90
G51B_CONTROL_TOL = 2
E6_CONTROL_MATCHED = {"A2": 3, "A3": 10}

# ---- G-E51c ----------------------------------------------------------------------
R128 = "D:/_ktmp/e40/e40_r128.bin"
R128_FLAGS = ["--carve-k", "3"]
WINDOWS = [40, 160, 640, 1280, 2560]
REPS = 5
# E49 ADDENDUM C.  A gate on a rate may only judge a window whose dispersion allows it.
# This is the tolerance the gate itself carries; any window whose OBSERVED spread (this
# run's, not E49's) exceeds it is printed UNRESOLVABLE and takes no verdict.
G51C_SPREAD_TOL = 0.06

# ---- G-E51d ----------------------------------------------------------------------
ATTRIB_WINDOW = 1280
ATTRIB_REPS = 3

RE_BENCH = re.compile(r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s")
RE_NATS = re.compile(r"NATS_TOTAL ([0-9.eE+-]+)\s*\nN_PREDICTED (\d+)")
RE_ORGAN = re.compile(r"^\s+(\w+)\s+([\d.]+)\s+([\d.]+)%", re.M)


# ==================================================================================
# decision functions -- all exercised in both directions before they judge anything
# ==================================================================================

def g_e51a(nats_base, nats_e51, nats_control):
    """Parity, and the null does not count unless the control FIRES."""
    ctl = abs(nats_control - nats_base) / nats_base
    if ctl < G51A_CONTROL_MIN:
        return ("VOID", ctl, None,
                "the planted control did not fire (%.3e, bar %.0e) -- an instrument not shown "
                "able to see a difference says nothing when it is silent" % (ctl, G51A_CONTROL_MIN))
    rel = abs(nats_e51 - nats_base) / nats_base
    if rel <= G51A_REL_BAR:
        return ("PARITY HOLDS", ctl, rel,
                "%.3e relative -> |dBPB| <= %.2e, inside the %.0e the 1e-4 bar implies"
                % (rel, rel * G51A_BPB_UPPER_BOUND, G51A_REL_BAR))
    return ("PARITY FAILS", ctl, rel,
            "%.3e relative -> |dBPB| <= %.2e, outside the bar"
            % (rel, rel * G51A_BPB_UPPER_BOUND))


def g_e51b(a1, controls):
    """E50's lesson: a scalar BPB average does not settle a discrete argmax."""
    for k, m in sorted(controls.items()):
        if m / 160.0 >= G51B_CONTROL_MAX:
            return ("VOID", "planted control %s did NOT fire: %d/160 -- the scorer has not been "
                            "shown to see a disagreement" % (k, m))
        if abs(m - E6_CONTROL_MATCHED[k]) > G51B_CONTROL_TOL:
            return ("VOID", "planted control %s moved %d -> %d, beyond the +-%d allowed"
                            % (k, E6_CONTROL_MATCHED[k], m, G51B_CONTROL_TOL))
    if a1 == G51B_A1_REQUIRED:
        return ("PASS", "A1 is %d/160 -- the greedy trajectory is unchanged" % a1)
    return ("FAIL", "A1 falls to %d/160 -- the flag does not ship" % a1)


def spread(v):
    return (max(v) - min(v)) / statistics.median(v)


def judgeable(sp_base, sp_e51):
    """E49 addendum C, in code rather than in my reading of a table."""
    return sp_base <= G51C_SPREAD_TOL and sp_e51 <= G51C_SPREAD_TOL


def g_e51c(cells):
    """cells: {n: {"base": [tok/s...], "e51": [tok/s...]}}.  Returns per-window verdicts.

    A window whose observed spread exceeds the gate's own tolerance is UNRESOLVABLE and
    takes no verdict -- it is still printed.
    """
    rows = []
    for n in sorted(cells):
        b, e = cells[n]["base"], cells[n]["e51"]
        sb, se = spread(b), spread(e)
        mb, me = statistics.median(b), statistics.median(e)
        delta = (me - mb) / mb
        if not judgeable(sb, se):
            rows.append({"n": n, "verdict": "UNRESOLVABLE", "base": mb, "e51": me,
                         "delta": delta, "spread_base": sb, "spread_e51": se})
        else:
            sep = (max(b) < min(e)) or (max(e) < min(b))
            rows.append({"n": n, "verdict": "JUDGED", "base": mb, "e51": me, "delta": delta,
                         "spread_base": sb, "spread_e51": se, "separable": sep})
    return rows


def g_e51c_summary(rows):
    """One line for the whole gate, DESCRIPTIVE only.

    It introduces no threshold the brief did not register: `C50` needs a line and a line needs
    two points, so "fewer than two windows survived the dispersion rule" is a statement about
    the instrument, not about the flag.  E51 run 1 crashed here -- `fit()` raised on a single
    judged window and G-E51c never printed a verdict at all, which is worse than any verdict
    it could have printed.
    """
    judged = [r for r in rows if r["verdict"] == "JUDGED"]
    npos = len({r["n"] for r in judged})
    if npos == 0:
        return {"verdict": "UNRESOLVABLE THROUGHOUT",
                "why": "no window's observed spread was inside the %.0f%% tolerance, so the "
                       "run measures the machine, not the flag" % (100 * G51C_SPREAD_TOL),
                "n_judged": 0}
    sep = [r for r in judged if r.get("separable")]
    if npos < 2:
        return {"verdict": "NO FIT -- ONE RESOLVABLE WINDOW",
                "why": "the single judged window (n=%d) reads %+.1f%% and the arms are %s; "
                       "C50 needs two points and cannot be computed"
                       % (judged[0]["n"], 100 * judged[0]["delta"],
                          "SEPARABLE" if sep else "NOT separable"),
                "n_judged": len(judged)}
    if not sep:
        return {"verdict": "NOT SEPARABLE",
                "why": "%d windows are judgeable and in none of them do the two arms' "
                       "repetitions separate" % len(judged),
                "n_judged": len(judged)}
    return {"verdict": "SEPARABLE AT %d OF %d JUDGED WINDOWS" % (len(sep), len(judged)),
            "why": "windows %s separate" % ",".join(str(r["n"]) for r in sep),
            "n_judged": len(judged)}


def g_e51d(s_base, s_e51, total_base, total_e51):
    """Attribution: a faster engine is not evidence that the SOFTMAX got faster."""
    d_total = total_base - total_e51
    d_s = s_base - s_e51
    if d_total <= 0:
        return ("NO IMPROVEMENT", d_s, d_total, "the total did not improve; nothing to attribute")
    share = d_s / d_total
    if d_s <= 0:
        return ("STORY IS WRONG", d_s, d_total,
                "the total improved by %.3f ms but S did NOT fall (%.3f ms) -- the win is not "
                "the softmax, and section 1's account does not hold" % (d_total, d_s))
    if share >= 0.5:
        return ("ATTRIBUTED TO S", d_s, d_total,
                "S falls %.3f ms of a %.3f ms total = %.0f%% -- the bulk of it"
                % (d_s, d_total, 100 * share))
    return ("PARTLY ELSEWHERE", d_s, d_total,
            "S falls %.3f ms but the total falls %.3f ms = only %.0f%% -- most of the win is "
            "somewhere else and E51 must say where" % (d_s, d_total, 100 * share))


# ==================================================================================
def selftest():
    log("== E51 gate self-test -- every decision function in every direction ==")
    n = [0]

    def chk(tag, got, want, why):
        n[0] += 1
        ok = (got == want)
        log("  %-5s %-20s expected %-20s %s   %s"
            % (tag, got, want, "FIRES" if ok else "**DEAD**", why))
        if not ok:
            raise SystemExit("self-test %s did not fire" % tag)

    B = 124963.95
    chk("A1", g_e51a(B, B * (1 + 1e-7), B * 1.2)[0], "PARITY HOLDS", "1e-7 is inside 2e-5")
    chk("A2", g_e51a(B, B * (1 + 1e-3), B * 1.2)[0], "PARITY FAILS", "1e-3 is 50x the bar")
    chk("A3", g_e51a(B, B * (1 + 1e-7), B * (1 + 1e-9))[0], "VOID",
        "a control that does not fire voids the null however good it looks")
    chk("A4", g_e51a(B, B * (1 + 2.1e-5), B * 1.2)[0], "PARITY FAILS", "the bar is a bar")
    chk("A5", g_e51a(B, B * (1 + 1.9e-5), B * 1.2)[0], "PARITY HOLDS", "and so is its other side")

    chk("B1", g_e51b(160, {"A2": 3, "A3": 10})[0], "PASS", "E6's own reading")
    chk("B2", g_e51b(159, {"A2": 3, "A3": 10})[0], "FAIL", "one token is one token")
    chk("B3", g_e51b(160, {"A2": 160, "A3": 10})[0], "VOID", "a control that agrees is not one")
    chk("B4", g_e51b(160, {"A2": 9, "A3": 10})[0], "VOID", "a control that moved is its own finding")

    # G-E51c: the addendum C rule, in both directions
    tight = {1280: {"base": [48.0, 48.5, 49.0], "e51": [54.0, 54.5, 55.0]}}
    loose = {40: {"base": [98.8, 121.4, 112.2], "e51": [117.3, 121.7, 126.8]}}
    chk("C1", g_e51c(tight)[0]["verdict"], "JUDGED", "spreads 2.0%/1.8% are inside the tolerance")
    chk("C2", g_e51c(loose)[0]["verdict"], "UNRESOLVABLE",
        "spreads 20.2%/7.8% exceed it -- THIS is the E49c defect, now refused in code")
    chk("C3", g_e51c(tight)[0]["separable"], True, "the ranges do not overlap")
    olap = {1280: {"base": [48.0, 50.0, 49.0], "e51": [49.5, 50.5, 50.0]}}
    chk("C4", g_e51c(olap)[0]["separable"], False,
        "judgeable by spread and STILL not separable -- the two tests are not the same test")

    # C5-C7: the crash run 1 died on.  One judged window is not a line, and a gate that
    # raises prints no verdict at all -- which is worse than any verdict it could print.
    one = dict(tight); one.update(loose)
    chk("C5", g_e51c_summary(g_e51c(one))["verdict"], "NO FIT -- ONE RESOLVABLE WINDOW",
        "1280 is judgeable and 40 is not; C50 needs two points")
    chk("C6", g_e51c_summary(g_e51c(loose))["verdict"], "UNRESOLVABLE THROUGHOUT",
        "nothing survived the dispersion rule -- the run measured the machine")
    two = {640: {"base": [96.0, 97.0, 98.0], "e51": [104.0, 105.0, 106.0]},
           1280: {"base": [48.0, 48.5, 49.0], "e51": [54.0, 54.5, 55.0]}}
    chk("C7", g_e51c_summary(g_e51c(two))["verdict"], "SEPARABLE AT 2 OF 2 JUDGED WINDOWS",
        "two resolvable windows, both separating -- the case C50 is computable in")

    chk("D1", g_e51d(4.0, 2.0, 13.5, 11.0)[0], "ATTRIBUTED TO S", "S falls 2.0 of 2.5 = 80%")
    chk("D2", g_e51d(4.0, 3.9, 13.5, 11.0)[0], "PARTLY ELSEWHERE", "S falls 0.1 of 2.5 = 4%")
    chk("D3", g_e51d(4.0, 4.3, 13.5, 11.0)[0], "STORY IS WRONG",
        "the engine got faster and the softmax got SLOWER -- section 1 would be wrong")
    chk("D4", g_e51d(4.0, 2.0, 13.5, 13.5)[0], "NO IMPROVEMENT", "nothing to attribute")

    log("  E51 gate self-test : ALL FIRE  (%d checks)" % n[0])
    log("")


# ==================================================================================
def run(cmd, what):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True)
    dt = time.time() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode("utf-8", "replace")[-1500:])
        raise SystemExit("engine failed (%s): %s" % (what, " ".join(cmd)))
    return p.stdout.decode("utf-8", "replace"), dt


def nats(engine, flags):
    cmd = ([engine, "--weights", S15, "--threads", str(THREADS), "--seqlen", str(SEQLEN)]
           + list(flags) + ["--bpb", IDS])
    txt, dt = run(cmd, "bpb")
    m = RE_NATS.search(txt)
    if not m:
        raise SystemExit("could not parse NATS_TOTAL")
    return float(m.group(1)), int(m.group(2)), dt


def phase_parity(out):
    log("== G-E51a -- BPB parity.  Nothing in E51 counts until this and G-E51b pass. ==")
    nb, n1, t1 = nats(BASE_ENGINE, PARITY_K)
    log("  base  k=256   NATS_TOTAL %.10f  n=%d  (%.0f s)" % (nb, n1, t1))
    ne, _, t2 = nats(E51_ENGINE, PARITY_K)
    log("  e51   k=256   NATS_TOTAL %.10f         (%.0f s)" % (ne, t2))
    nc, _, t3 = nats(BASE_ENGINE, CONTROL_K)
    log("  CTRL  k=3     NATS_TOTAL %.10f         (%.0f s)" % (nc, t3))
    v, ctl, rel, why = g_e51a(nb, ne, nc)
    log("")
    log("  planted control : %s   (%.3e apart, bar %.0e)"
        % ("FIRES" if ctl >= G51A_CONTROL_MIN else "DOES NOT FIRE", ctl, G51A_CONTROL_MIN))
    if rel is not None:
        log("  e51 vs base     : %.3e relative" % rel)
    log("  G-E51a : %s" % v)
    log("  %s" % why)
    log("")
    out["G_E51a"] = {"verdict": v, "why": why, "nats_base": repr(nb), "nats_e51": repr(ne),
                     "nats_control": repr(nc), "control_rel": ctl, "pair_rel": rel}
    return v == "PARITY HOLDS"


# ---- G-E51b ----------------------------------------------------------------------
def write_prompt_ids(prompt_ids):
    os.makedirs(TMP, exist_ok=True)
    for i, ids in enumerate(prompt_ids):
        with open(os.path.join(TMP, "p%d.bin" % i), "wb") as f:
            f.write(struct.pack("<%di" % len(ids), *ids))


def parse_gen(txt):
    d = {}
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "GEN_IDS":
            d["ids"] = [int(x) for x in f[1:]]
        elif f[0].startswith("GEN_"):
            d[f[0][4:].lower()] = float(f[1])
    return d


def e6_engine_stage(engine, n_new, prompt_ids, tag):
    arms = {}
    for name, wp, hf, role in E6_ARMS:
        if not os.path.exists(wp):
            log("  SKIP %s -- %s not on disk" % (name, wp))
            continue
        rec = {"weights": wp, "hf": hf, "role": role, "runs": []}
        for i in range(len(prompt_ids)):
            pfx = os.path.join(TMP, "%s_%s_p%d" % (tag, name, i))
            ids = os.path.join(TMP, "p%d.bin" % i)
            base = [engine, "--weights", wp, "--threads", str(THREADS)]
            r1 = parse_gen(run(base + ["--generate", ids, str(n_new), pfx], "gen")[0])
            r2 = parse_gen(run(base + ["--generate", ids, str(n_new), pfx + "_b"], "gen b")[0])
            gd = (open(pfx + ".ids.bin", "rb").read() ==
                  open(pfx + "_b.ids.bin", "rb").read())
            for f in (pfx + "_b.ids.bin", pfx + "_b.prefill.bin", pfx + ".prefill.bin"):
                if os.path.exists(f):
                    os.remove(f)
            rec["runs"].append({"prompt": i, "ids": r1["ids"], "G_D": bool(gd),
                                "decode_toks": r1["decode_toks"]})
            log("    %-5s %s p%d  G-D %s  decode %.2f tok/s"
                % (tag, name, i, "PASS" if gd else "FAIL", r1["decode_toks"]))
        arms[name] = rec
    return arms


def score_against_ref(arms, ref, n_new):
    summary = {}
    for name, rec in sorted(arms.items()):
        rr = {x["prompt"]: x for x in ref[rec["hf"]]}
        tot = match = 0
        for r in rec["runs"]:
            ours = r["ids"][-n_new:]
            theirs = rr[r["prompt"]]["ids"][-n_new:]
            for k in range(n_new):
                tot += 1
                if ours[k] == theirs[k]:
                    match += 1
        summary[name] = {"matched": match, "counted": tot,
                         "agree": match / float(tot) if tot else 0.0}
    return summary


def phase_greedy(out):
    log("== G-E51b -- the DISCRETE test.  E50 showed a BPB average does not settle an argmax. ==")
    e6 = json.load(open(os.path.join(E6_RES, "engine.json")))
    ref = json.load(open(os.path.join(E6_RES, "ref.json")))
    n_new = e6["n_new"]
    write_prompt_ids(e6["prompt_ids"])
    os.makedirs(E51_RES, exist_ok=True)
    arms = e6_engine_stage(E51_ENGINE, n_new, e6["prompt_ids"], "e51")
    json.dump({"n_new": n_new, "threads": THREADS, "arms": arms},
              open(os.path.join(E51_RES, "engine_e51.json"), "w"), indent=1)
    res = score_against_ref(arms, ref, n_new)
    log("")
    log("  %-4s %-12s %-12s" % ("arm", "E6", "E51"))
    for name in ("A1", "A2", "A3"):
        if name in res:
            e6m = 160 if name == "A1" else E6_CONTROL_MATCHED[name]
            log("  %-4s %-12s %-12s" % (name, "%d/160" % e6m, "%d/160" % res[name]["matched"]))
    controls = {k: res[k]["matched"] for k in res if k != "A1"}
    v, why = g_e51b(res["A1"]["matched"], controls)
    log("")
    log("  G-E51b : %s" % v)
    log("  %s" % why)
    log("")
    out["G_E51b"] = {"verdict": v, "why": why, "summary": res}
    return v == "PASS"


# ---- G-E51c ----------------------------------------------------------------------
def one_bench(engine, n):
    occ = Occupancy()
    occ.sample()
    txt, _ = run([engine, "--weights", R128, "--threads", str(THREADS)]
                 + R128_FLAGS + ["--bench", str(n)], "bench")
    o = occ.sample()
    m = RE_BENCH.search(txt)
    return float(m.group(3)), o


def phase_speed(out):
    log("== G-E51c -- the speed, judged ONLY where the dispersion permits (E49 addendum C) ==")
    log("  tolerance %.0f%%; any window whose OBSERVED spread exceeds it takes NO verdict."
        % (100 * G51C_SPREAD_TOL))
    cells = {n: {"base": [], "e51": []} for n in WINDOWS}
    occs = {n: [] for n in WINDOWS}
    for rep in range(1, REPS + 1):
        for n in WINDOWS:
            for tag, eng in BUILDS:
                r, o = one_bench(eng, n)
                cells[n][tag].append(r)
                occs[n].append(o)
                log("    %-5s n=%-5d rep %d/%d  %7.2f tok/s   occ %.1f%%"
                    % (tag, n, rep, REPS, r, o))
    log("")
    rows = g_e51c(cells)
    log("  %-6s %9s %9s %8s %8s %8s  %s"
        % ("n", "base", "e51", "delta", "sp_base", "sp_e51", "verdict"))
    for r in rows:
        log("  %-6d %8.2f  %8.2f  %+7.1f%% %7.1f%% %7.1f%%  %s%s"
            % (r["n"], r["base"], r["e51"], 100 * r["delta"], 100 * r["spread_base"],
               100 * r["spread_e51"], r["verdict"],
               "" if r["verdict"] == "UNRESOLVABLE"
               else ("  separable" if r["separable"] else "  NOT separable")))
    log("")
    summary = g_e51c_summary(rows)
    log("  G-E51c : %s" % summary["verdict"])
    log("  %s" % summary["why"])
    log("")
    judged = [r for r in rows if r["verdict"] == "JUDGED"]
    fits = {}
    if len({r["n"] for r in judged}) < 2:
        log("  C50 NOT COMPUTABLE -- %d window(s) survived the dispersion rule and a straight"
            " line needs two." % len(judged))
        log("  That is the instrument declining to extrapolate, NOT a measurement of the flag.")
    else:
        for tag in ("base", "e51"):
            # fit() takes [(mean_pos, per_token)] and returns (a, b) in whatever unit it is
            # given; E49 feeds it ms/tok, so C50 = (20 ms - a) / b.  Same convention here.
            a, b = fit([(r["n"] / 2.0, 1000.0 / r[tag]) for r in judged])
            c50 = (20.0 - a) / b if b > 0 else float("inf")
            fits[tag] = {"a": a, "b": b, "C50": c50}
            log("  %-5s ms/tok = %.4f + %.6f*pos   ->  C50 = %.0f   (fit on the %d JUDGED "
                "windows)" % (tag, a, b, c50, len(judged)))
        if fits["e51"]["b"] > 0:
            log("  b(base)/b(e51) = %.3f      C50 %.0f -> %.0f"
                % (fits["base"]["b"] / fits["e51"]["b"],
                   fits["base"]["C50"], fits["e51"]["C50"]))
    log("")
    out["G_E51c"] = {"rows": rows, "fits": fits, "cells": cells,
                     "summary": summary,
                     "tolerance": G51C_SPREAD_TOL, "reps": REPS}


# ---- G-E51d ----------------------------------------------------------------------
def organ_ms(engine, attnr):
    txt, _ = run([engine, "--weights", R128, "--threads", str(THREADS)] + R128_FLAGS
                 + ["--profile", "--attnr", attnr, "--bench", str(ATTRIB_WINDOW)], "profile")
    d = {}
    for m in RE_ORGAN.finditer(txt):
        d[m.group(1)] = float(m.group(2))
    tot = RE_BENCH.search(txt)
    return d, 1000.0 / float(tot.group(3))


def phase_attrib(out):
    log("== G-E51d -- attribution.  A faster engine is not evidence the SOFTMAX got faster. ==")
    rec = {}
    for tag, eng in BUILDS:
        s_vals, tot_vals = [], []
        for rep in range(ATTRIB_REPS):
            d1, _ = organ_ms(eng, "sm1")
            d2, t2 = organ_ms(eng, "sm2")
            s = d2.get("attention", 0.0) - d1.get("attention", 0.0)
            s_vals.append(s)
            tot_vals.append(t2)
            log("    %-5s rep %d/%d   S = (sm2 - sm1) = %.4f ms   attention sm1 %.4f ms"
                % (tag, rep + 1, ATTRIB_REPS, s, d1.get("attention", 0.0)))
        rec[tag] = {"S": statistics.median(s_vals), "S_all": s_vals}
        d0, t0 = organ_ms(eng, "none")
        rec[tag]["total_ms"] = t0
        rec[tag]["attention_none"] = d0.get("attention", 0.0)
        log("    %-5s  S %.4f ms   attention(none) %.4f ms   token %.4f ms"
            % (tag, rec[tag]["S"], rec[tag]["attention_none"], t0))
    v, d_s, d_tot, why = g_e51d(rec["base"]["S"], rec["e51"]["S"],
                                rec["base"]["total_ms"], rec["e51"]["total_ms"])
    log("")
    log("  G-E51d : %s" % v)
    log("  %s" % why)
    log("")
    out["G_E51d"] = {"verdict": v, "why": why, "dS": d_s, "dTotal": d_tot, "arms": rec}


# ==================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--phase", choices=["parity", "greedy", "speed", "attrib"])
    a = ap.parse_args()

    log("=" * 84)
    log("E51 -- the softmax was paying for errno")
    log("  -fno-math-errno and nothing else.  NOT -ffast-math: no reassociation, no FTZ,")
    log("  no reciprocal substitution, -ffp-contract=on unchanged.  expf(x) and")
    log("  (float)exp((double)x) are different functions, so this is NOT bit-exact.")
    log("=" * 84)
    selftest()
    if a.selftest:
        return

    for tag, eng in BUILDS:
        if not os.path.exists(eng):
            raise SystemExit("missing build: " + eng)
    os.makedirs(TMP, exist_ok=True)
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    out.update({"brief": "BRIEF_E51_THE_SOFTMAX_WAS_PAYING_FOR_ERRNO.md",
                "base": BASE_ENGINE, "e51": E51_ENGINE, "threads": THREADS})

    phases = [a.phase] if a.phase else ["parity", "greedy", "speed", "attrib"]
    for ph in phases:
        if ph == "parity":
            ok = phase_parity(out)
            json.dump(out, open(OUT, "w"), indent=1)
            if not ok and not a.phase:
                raise SystemExit("G-E51a did not pass -- no speed number in E51 counts")
        elif ph == "greedy":
            ok = phase_greedy(out)
            json.dump(out, open(OUT, "w"), indent=1)
            if not ok and not a.phase:
                raise SystemExit("G-E51b did not pass -- the flag does not ship, section 4")
        elif ph == "speed":
            phase_speed(out)
        elif ph == "attrib":
            phase_attrib(out)
        json.dump(out, open(OUT, "w"), indent=1)
    log("wrote " + OUT)


if __name__ == "__main__":
    main()
