#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E57 -- what does a TRAINED LLM actually cost on this engine, and at what quality?

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E57_WHAT_DOES_A_TRAINED_LLM_ACTUALLY_COST_ON_THIS_ENGINE.md`
Pre-registered and pushed before the gated runs.

Every tok/s in this programme is measured on SYNTHETIC weights.  A real trained model has run on
this engine since E6, used only as a correctness control, and its speed was never measured.
This reads what is already on the disk.

Quality reuses E51.score_against_ref UNCHANGED against E6's stored HuggingFace references, so
E57 cannot drift away from E6/E50/E51/E53.  No new scoring code is written here.
"""
import json
import os
import random
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e44_interval as E44                                          # noqa: E402
import e51_math_errno as E51                                        # noqa: E402

OUTDIR = os.path.join(HERE, "results")
ENGINE = os.path.join(HERE, "donor_engine_e53.exe")
THREADS = 6
OCC_BAR = E44.OCC_BAR

W = r"D:\_ktmp\e1"
# (tag, weights, hf model, role, is a stored E6 control?)
ARMS = [
    ("A1_05b_f32",  os.path.join(W, "qwen25-05b_f32.bin"),  "Qwen/Qwen2.5-0.5B",
     "known-positive (fp32)", 160),
    ("05b_tq",      os.path.join(W, "qwen25-05b_tq.bin"),   "Qwen/Qwen2.5-0.5B",
     "UNSCORED -- ternary body, fp32 head", None),
    ("A2_05b_tqh",  os.path.join(W, "qwen25-05b_tqh.bin"),  "Qwen/Qwen2.5-0.5B",
     "planted control (ternary+head)", 3),
    ("15b_f32",     os.path.join(W, "qwen25-15b_f32.bin"),  "Qwen/Qwen2.5-1.5B",
     "UNSCORED (fp32)", None),
    ("15b_tq",      os.path.join(W, "qwen25-15b_tq.bin"),   "Qwen/Qwen2.5-1.5B",
     "UNSCORED -- ternary body, fp32 head", None),
    ("A3_15b_tqh",  os.path.join(W, "qwen25-15b_tqh.bin"),  "Qwen/Qwen2.5-1.5B",
     "planted control at 1.5B", 10),
]

BENCH_N = 160
REPS = 15
SPLITS = 200          # G-E55a2: the median over many splits, not one draw
IQR_TOL = 0.20
TARGET = 50.0
EXCELLENT = 100.0
SEED = 5701


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


def quantile(v, q):
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    p = q * (len(s) - 1)
    lo = int(p)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (p - lo)


def iqr_pct(v):
    m = statistics.median(v)
    return 100.0 * (quantile(v, 0.75) - quantile(v, 0.25)) / m if m else float("nan")


# ======================================================================================
def g_e55a2(vals, rng, splits=SPLITS):
    """E55 A.2's repaired stability estimator: the MEDIAN of |half-full|/full over many
    random splits, not one draw.  A variance reduction on the same statistic -- registered
    post-hoc in E55 A.2 and permitted on FRESH cells only, which these are."""
    if len(vals) < 6:
        return ("UNSTABLE", float("nan"), "only %d reps -- too few to split" % len(vals))
    full = iqr_pct(vals)
    if full <= 0:
        return ("UNSTABLE", float("nan"), "full-sample interquartile width is zero")
    rels = []
    for _ in range(splits):
        rels.append(abs(iqr_pct(rng.sample(vals, len(vals) // 2)) - full) / full)
    med = statistics.median(rels)
    if med <= IQR_TOL:
        return ("PASS", med, "IQR %.2f%% on %d reps; median |half-full|/full over %d splits ="
                " %.0f%%" % (full, len(vals), splits, 100 * med))
    return ("UNSTABLE", med, "IQR %.2f%% on %d reps; median |half-full|/full over %d splits ="
            " %.0f%%, past the %.0f%% bar" % (full, len(vals), splits, 100 * med,
                                              100 * IQR_TOL))


def g_e57b(matched, counted):
    """FAITHFUL only at a perfect match -- the same bar G-E53c2 had to clear."""
    return "FAITHFUL" if matched == counted and counted > 0 else "NOT FAITHFUL"


def g_e57d(faithful, p25):
    """Speed from a non-faithful arm may not be combined with quality from a slow one."""
    if faithful != "FAITHFUL":
        return "NO CLAIM (not faithful)"
    if p25 >= EXCELLENT:
        return "DEMONSTRATES EXCELLENT"
    if p25 >= TARGET:
        return "DEMONSTRATES THE TARGET"
    return "FAITHFUL BUT BELOW TARGET"


def first_divergence(ours, theirs):
    for i, (a, b) in enumerate(zip(ours, theirs)):
        if a != b:
            return i
    return -1


# ======================================================================================
def selftest():
    ok = [0]
    rng = random.Random(1)

    def chk(name, cond):
        ok[0] += 1
        log("  %-54s %s" % (name, "fires" if cond else "*** DEAD ***"))
        if not cond:
            raise SystemExit("a decision function did not fire on a known-positive.  STOP.")

    chk("A2-1 a converged sample is STABLE",
        g_e55a2([100.0 + 0.01 * ((i * 37) % 20) for i in range(30)], rng, 50)[0] == "PASS")
    chk("A2-2 an outlier-dominated sample is UNSTABLE",
        g_e55a2([100.0] * 28 + [10.0, 400.0], rng, 50)[0] == "UNSTABLE")
    chk("A2-3 too few reps is UNSTABLE", g_e55a2([1.0, 2.0], rng, 50)[0] == "UNSTABLE")
    r1 = g_e55a2([100.0 + (i % 7) for i in range(20)], random.Random(1), 200)[1]
    r2 = g_e55a2([100.0 + (i % 7) for i in range(20)], random.Random(2), 200)[1]
    chk("A2-4 and it is SEED-STABLE, which G-E55a was not",
        abs(r1 - r2) / max(r1, 1e-9) < 0.10)

    chk("B-1 a perfect match is FAITHFUL", g_e57b(160, 160) == "FAITHFUL")
    chk("B-2 159 of 160 is NOT faithful", g_e57b(159, 160) == "NOT FAITHFUL")
    chk("B-3 zero counted is NOT faithful", g_e57b(0, 0) == "NOT FAITHFUL")
    chk("B-4 first divergence is located", first_divergence([1, 2, 9, 4], [1, 2, 3, 4]) == 2)
    chk("B-5 identical sequences report no divergence",
        first_divergence([1, 2, 3], [1, 2, 3]) == -1)

    chk("D-1 fast but unfaithful makes NO CLAIM",
        g_e57d("NOT FAITHFUL", 120.0) == "NO CLAIM (not faithful)")
    chk("D-2 faithful and fast DEMONSTRATES EXCELLENT",
        g_e57d("FAITHFUL", 120.0) == "DEMONSTRATES EXCELLENT")
    chk("D-3 faithful at 50 DEMONSTRATES THE TARGET",
        g_e57d("FAITHFUL", 50.0) == "DEMONSTRATES THE TARGET")
    chk("D-4 faithful but slow says so", g_e57d("FAITHFUL", 43.0) == "FAITHFUL BUT BELOW TARGET")

    log("")
    log("  %d of %d fire." % (ok[0], ok[0]))


# ======================================================================================
def one_cell(weights):
    sp = E44.Split()
    cmd = [ENGINE, "--weights", weights, "--threads", str(THREADS), "--bench", str(BENCH_N)]
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
    rec.update({"rate": float(m.group(3)), "occ": occ, "foreign": foc})
    return rec


def main():
    log("E57 -- what does a TRAINED LLM actually cost on this engine, and at what quality?")
    log("brief: BRIEF_E57_WHAT_DOES_A_TRAINED_LLM_ACTUALLY_COST_ON_THIS_ENGINE.md")
    log("")
    log("  These are REAL TRAINED Qwen2.5 weights, not the synthetic e40_r128 shape every")
    log("  other tok/s in this programme is measured on.")
    log("")
    log("== self-test.  Every decision function must fire on a known-positive. ==")
    selftest()
    if "--selftest" in sys.argv:
        return

    e6 = json.load(open(os.path.join(E51.E6_RES, "engine.json")))
    ref = json.load(open(os.path.join(E51.E6_RES, "ref.json")))
    n_new = e6["n_new"]
    E51.write_prompt_ids(e6["prompt_ids"])
    os.makedirs(E51.TMP, exist_ok=True)

    # ---------------- quality ----------------
    log("")
    log("== G-E57a / G-E57b -- quality, scored by E51.score_against_ref against E6's")
    log("   stored HuggingFace references.  %d prompts x %d new tokens, greedy. =="
        % (len(e6["prompt_ids"]), n_new))
    log("")
    qual = {}
    for tag, wp, hf, role, stored in ARMS:
        if not os.path.exists(wp):
            log("  SKIP %s -- not on disk" % tag)
            continue
        rec = {"weights": wp, "hf": hf, "role": role, "runs": []}
        for i in range(len(e6["prompt_ids"])):
            pfx = os.path.join(E51.TMP, "e57_%s_p%d" % (tag, i))
            ids = os.path.join(E51.TMP, "p%d.bin" % i)
            base = [ENGINE, "--weights", wp, "--threads", str(THREADS)]
            r1 = E51.parse_gen(E51.run(base + ["--generate", ids, str(n_new), pfx], "gen")[0])
            # score_against_ref keys the reference by PROMPT INDEX, so the record must carry
            # it.  Appending parse_gen's dict raw would have thrown a KeyError -- or worse,
            # scored the wrong reference against the wrong prompt.
            rec["runs"].append({"prompt": i, "ids": r1["ids"],
                                "decode_toks": r1.get("decode_toks")})
            for f in (pfx + ".prefill.bin", pfx + ".ids.bin"):
                if os.path.exists(f):
                    os.remove(f)
        arms = {tag: rec}
        s = E51.score_against_ref(arms, ref, n_new)[tag]
        rr = {x["prompt"]: x for x in ref[hf]}
        divs = [first_divergence(r["ids"][-n_new:], rr[r["prompt"]]["ids"][-n_new:])
                for r in rec["runs"]]
        v = g_e57b(s["matched"], s["counted"])
        qual[tag] = {"matched": s["matched"], "counted": s["counted"], "agree": s["agree"],
                     "verdict": v, "first_divergence": divs, "stored": stored, "role": role}
        ctl = ""
        if stored is not None:
            ctl = ("   CONTROL: E6 says %d -> %s" %
                   (stored, "REPRODUCED" if s["matched"] == stored else "*** DEVIATED ***"))
        log("  %-12s %3d/%-3d  %-13s  first divergence per prompt %s%s"
            % (tag, s["matched"], s["counted"], v, divs, ctl))

    bad = [t for t, q in qual.items()
           if q["stored"] is not None and q["matched"] != q["stored"]]
    log("")
    if bad:
        log("  G-E57a : *** DEVIATED *** -- %s did not reproduce E6.  STOP." % bad)
        raise SystemExit("G-E57a failed; nothing below counts.")
    log("  G-E57a : FIRES -- all three stored controls reproduce E6 exactly")
    log("")

    # ---------------- speed ----------------
    log("== G-E57c -- speed, %d reps per arm at --bench %d, arm order rotated,"
        % (REPS, BENCH_N))
    log("   judged by G-E55a2 (median over %d splits), on FRESH cells. ==" % SPLITS)
    log("")
    rng = random.Random(SEED)
    live = [a for a in ARMS if os.path.exists(a[1])]
    log("   warm-up cell, DISCARDED: %.2f tok/s" % one_cell(live[0][1])["rate"])
    cells = {a[0]: [] for a in live}
    for rep in range(REPS):
        order = list(live)
        rng.shuffle(order)
        for tag, wp, _, _, _ in order:
            c = one_cell(wp)
            c["rep"] = rep + 1
            cells[tag].append(c)
        if (rep + 1) % 5 == 0:
            log("   rep %2d/%2d done" % (rep + 1, REPS))
    log("")

    # E55 A.3 measured that OCC_BAR rejects ~40%% of cells and that the rejected ones are NOT
    # slower, and the idle floor read before this run was 3.27%% -- most of the bar.  The bar
    # is therefore still applied as the registered analysis (E55's), and the ALL-CELL
    # counterfactual is printed beside it, because E52's law is that a refusal must never
    # launder a number.
    rows = []
    # The 'cells' column USED to be ambiguous: four of six arms fell back to the all-cell set
    # because fewer than six of their cells cleared OCC_BAR, and the table printed the count
    # without saying which set it came from -- a column labelled one thing holding another,
    # the same defect class as G-E53f's 'p = nan'.  The analysis is now named per row.
    log("   %-12s %5s %-6s %8s %8s %8s %7s  %-9s %s"
        % ("arm", "cells", "set", "p25", "median", "p75", "IQR", "G-E55a2", "G-E57d"))
    for tag, wp, hf, role, stored in live:
        allc = cells[tag]
        clean = [c for c in allc if not (c["foreign"] == c["foreign"]
                                         and c["foreign"] > OCC_BAR)]
        use = clean if len(clean) >= 6 else allc
        v = [c["rate"] for c in use]
        st, rel, why = g_e55a2(v, rng)
        p25 = quantile(v, 0.25)
        d = g_e57d(qual[tag]["verdict"], p25) if st == "PASS" else "NO CLAIM (interval unstable)"
        va = [c["rate"] for c in allc]
        rows.append({"arm": tag, "weights": wp, "hf": hf, "role": role,
                     "cells": len(allc), "used": len(use), "p25": p25,
                     "median": statistics.median(v), "p75": quantile(v, 0.75),
                     "iqr_pct": iqr_pct(v), "stable": st, "stable_why": why,
                     "quality": qual[tag]["verdict"],
                     "matched": qual[tag]["matched"], "counted": qual[tag]["counted"],
                     "verdict": d,
                     "all_p25": quantile(va, 0.25), "all_median": statistics.median(va),
                     "all_iqr_pct": iqr_pct(va), "all_cells": len(va),
                     "all_verdict": g_e57d(qual[tag]["verdict"], quantile(va, 0.25)),
                     "foreign_median": statistics.median([c["foreign"] for c in allc]),
                     "clock_median": statistics.median(
                         [c["clock_pct"] for c in allc
                          if c["clock_pct"] == c["clock_pct"]] or [float("nan")])})
        rows[-1]["set"] = "clean" if use is clean else "ALL"
        log("   %-12s %5d %-6s %8.2f %8.2f %8.2f %6.2f%%  %-9s %s"
            % (tag, len(use), rows[-1]["set"], p25, statistics.median(v),
               quantile(v, 0.75), iqr_pct(v), st, d))
    log("")
    log("   'set' = ALL means fewer than six of that arm's cells cleared OCC_BAR = %.2f and"
        % OCC_BAR)
    log("   the row fell back to every cell; it is not a clean-cell number.")
    log("")
    log("   counterfactual, ALL cells including those over OCC_BAR = %.2f (E52's law: a"
        % OCC_BAR)
    log("   refusal must not launder a number; E55 A.3 found the rejected cells are not slower)")
    log("   %-12s %5s %8s %8s %7s %9s %8s  %s"
        % ("arm", "cells", "p25", "median", "IQR", "foreign", "clock", "G-E57d"))
    for r in rows:
        log("   %-12s %5d %8.2f %8.2f %6.2f%% %8.2f%% %7.1f%%  %s"
            % (r["arm"], r["all_cells"], r["all_p25"], r["all_median"], r["all_iqr_pct"],
               r["foreign_median"], r["clock_median"], r["all_verdict"]))
    log("")
    for r in rows:
        log("   %-12s %s" % (r["arm"], r["stable_why"]))
    log("")

    winners = [r for r in rows if r["verdict"].startswith("DEMONSTRATES")]
    log("== G-E57d -- the joint claim ==")
    if winners:
        for r in winners:
            log("   %s : %s at p25 %.2f tok/s, %d/%d against HuggingFace"
                % (r["arm"], r["verdict"], r["p25"], r["matched"], r["counted"]))
    else:
        log("   NO ARM DEMONSTRATES THE TARGET.")
        fa = [r for r in rows if r["quality"] == "FAITHFUL"]
        if fa:
            best = max(fa, key=lambda r: r["p25"])
            log("   The fastest FAITHFUL arm is %s at p25 %.2f tok/s (%d/%d), %.0f%% of the"
                " 50 tok/s bar." % (best["arm"], best["p25"], best["matched"],
                                    best["counted"], 100 * best["p25"] / TARGET))
        fast = max(rows, key=lambda r: r["p25"])
        log("   The fastest arm overall is %s at p25 %.2f tok/s and it is %s."
            % (fast["arm"], fast["p25"], fast["quality"]))
    dis = [r for r in rows if r["all_verdict"] != r["verdict"]
           and not r["verdict"].startswith("NO CLAIM (interval")]
    if dis:
        log("")
        log("   THE TWO ANALYSES DISAGREE, which is itself the finding:")
        for r in dis:
            log("     %-12s clean %s   /   all cells %s"
                % (r["arm"], r["verdict"], r["all_verdict"]))
    log("")

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    path = os.path.join(OUTDIR, "e57_trained.json")
    with open(path, "w") as f:
        json.dump({"brief": "BRIEF_E57_WHAT_DOES_A_TRAINED_LLM_ACTUALLY_COST_ON_THIS_ENGINE.md",
                   "bench_n": BENCH_N, "reps": REPS, "threads": THREADS, "splits": SPLITS,
                   "occ_bar": OCC_BAR, "n_new": n_new, "quality": qual, "rows": rows,
                   "cells": {k: v for k, v in cells.items()}}, f, indent=1)
    log("wrote %s" % path)


if __name__ == "__main__":
    main()
