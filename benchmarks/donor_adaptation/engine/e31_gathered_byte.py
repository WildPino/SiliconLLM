#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E31 -- what does a GATHERED byte cost, and at what granularity does the penalty go away?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E31_THE_GATHERED_BYTE.md, pushed at 8003fa3
BEFORE this file existed and before e31_gather.exe had ever been TIMED.  Nothing here may
contradict it; the result document scores it as written.

E30 left one variable -- bytes per token, against 1.45 G moved weights for 50 tok/s -- and that
number was derived from a DENSE stream.  Every route to it reads a SUBSET.  This measures what
a subset actually delivers, in USEFUL bytes per second, as a function of how contiguous it is.

RUN 1 WAS VOIDED BY ITS OWN PLANTED CONTROL, and the defect was mine: `contig` was implemented
as a walk through the group-offset array, so it carried the gathered arms' per-group bookkeeping
and sagged from 39.96 GB/s at 2048 B groups to 34.34 at 64 B -- it moved with the variable it was
controlling for.  Run 1's numbers are kept at results/e31_gathered_byte_run1_VOID.json.  The fix
is the STRICTER reading of the brief ("one unbroken run"), not a looser gate, and the old
behaviour survives as `contig_ind`, a diagnostic that prices the bookkeeping instead of hiding it.

  python e31_gathered_byte.py --reps 3
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e26_carve_cost import cpu_busy, IDLE_BAR                   # noqa: E402

OUT = os.path.join(HERE, "results", "e31_gathered_byte.json")
EXE = os.path.join(HERE, "e31_gather.exe")
SRC = os.path.join(HERE, "e31_gather.c")

E30_256MB_GB_S = 37.16       # E30's own 256 MB median -- the anchor G-E31A is judged against
E30_CEIL = 36.299            # E30's BW-CEIL, for the budget arithmetic
CTRL_TOL = 0.15
S15_ROW, T10_ROW = 768, 2048
CROSSOVER = 0.90
GOAL_TOK_S = 50.0
T10_MOVED_GB = 5.31275776    # G-E30B, to the byte
BANDS = [(0.85, "GATHER-IS-FREE"), (0.50, "GATHER-COSTS"), (0.0, "GATHER-DOMINATES")]


def log(*a):
    print(*a, flush=True)


def run_exe(args):
    r = subprocess.run([EXE] + [str(x) for x in args], capture_output=True)
    txt = r.stdout.decode(errors="replace")
    if r.returncode != 0:
        log(txt[-1200:]); log(r.stderr.decode(errors="replace")[-1200:])
        raise SystemExit("e31_gather failed")
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--buf-mb", type=float, default=2048.0)
    ap.add_argument("--div", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()

    if not os.path.exists(EXE):
        raise SystemExit("%s missing -- build with the engine's own flags:\n"
                         "  clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp %s -o %s -lm"
                         % (EXE, os.path.basename(SRC), os.path.basename(EXE)))

    log("== E31: what does a gathered byte cost? ==")
    busy, peak = cpu_busy(4)
    log("  contention witness: %.1f%% mean / %.0f%% peak (bar %.0f%% on the MEAN)"
        % (busy, peak, IDLE_BAR))
    if busy > IDLE_BAR:
        raise SystemExit("the box is at %.1f%% -- a contended timing is not a timing.  STOP."
                         % busy)

    out = {"brief": "briefs/BRIEF_E31_THE_GATHERED_BYTE.md (8003fa3)",
           "threads": a.threads, "reps": a.reps, "buffer_MB": a.buf_mb, "select_1_in": a.div,
           "cpu_busy_mean_pct": busy, "cpu_busy_peak_pct": peak, "bands": BANDS,
           "e30_anchor_256MB_GB_s": E30_256MB_GB_S, "e30_ceiling_GB_s": E30_CEIL,
           "run": 2,
           "run1": "VOID on G-E31A -- the control walked the offset array and so carried the "
                   "gathered arms' own per-group bookkeeping; results/e31_gathered_byte_run1_"
                   "VOID.json"}

    # ---- G-E31C: correctness, in this session, before any timing is read
    st = run_exe(["--selftest"]).strip()
    log("  " + st)
    out["G_E31C"] = {"line": st, "fires": st.endswith("OK")}
    if not out["G_E31C"]["fires"]:
        raise SystemExit("G-E31C: the gather does not read what it claims to read.  VOID.")

    # ---- the sweep
    log("")
    log("  -- sweep: %.0f MB buffer, 1 group in %d, %d threads, %d reps"
        % (a.buf_mb, a.div, a.threads, a.reps))
    txt = run_exe([a.threads, a.reps, a.buf_mb, a.div])
    rows = []
    for ln in txt.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        g, order, umb, best, med = ln.split(",")
        rows.append({"group_bytes": int(g), "order": order, "useful_MB": float(umb),
                     "best": float(best), "median": float(med)})
    out["sweep"] = rows
    tab = {(r["group_bytes"], r["order"]): r["median"] for r in rows}
    groups = sorted({r["group_bytes"] for r in rows})

    log("")
    log("   group      sorted    random    contig  contig_ind   s/contig  r/contig  "
        "bookkeeping")
    for g in groups:
        s, rd, c = tab.get((g, "sorted")), tab.get((g, "random")), tab.get((g, "contig"))
        ci = tab.get((g, "contig_ind"))
        if None in (s, rd, c, ci):
            continue
        log("  %8d  %8.2f  %8.2f  %8.2f  %10.2f  %9.3f  %8.3f  %10.3f"
            % (g, s, rd, c, ci, s / c, rd / c, ci / c))

    # ---- G-E31A: the same-volume planted control
    ctl = [(g, tab[(g, "contig")]) for g in groups if (g, "contig") in tab]
    devs = [(g, v / E30_256MB_GB_S - 1.0) for g, v in ctl]
    worst = max(devs, key=lambda kv: abs(kv[1]))
    vals = [v for _, v in ctl]
    spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
    g_a = {"contig_GB_s": dict((str(g), v) for g, v in ctl),
           "anchor_GB_s": E30_256MB_GB_S, "tol": CTRL_TOL,
           "worst_group_bytes": worst[0], "worst_rel_dev": worst[1],
           "contig_spread": spread,
           "fires": bool(abs(worst[1]) <= CTRL_TOL and spread <= CTRL_TOL)}
    out["G_E31A"] = g_a
    log("")
    log("  G-E31A  contig vs E30's 37.16 GB/s: worst %+.1f%% (at %d B), spread %.1f%% "
        "(bar +-%.0f%% on both) -> %s"
        % (100 * worst[1], worst[0], 100 * spread, 100 * CTRL_TOL,
           "FIRES" if g_a["fires"] else "VOID"))

    # ---- G-E31B: the known positive, from E26
    s768, s2048 = tab.get((S15_ROW, "sorted")), tab.get((T10_ROW, "sorted"))
    g_b = {"sorted_768_GB_s": s768, "sorted_2048_GB_s": s2048,
           "ratio_768_over_2048": (s768 / s2048) if (s768 and s2048) else None,
           "source": "E26 measured S15 (768 B rows) suffering more than T10 (2048 B rows), "
                     "bands 18.00% vs 12.54%, disjoint",
           "fires": bool(s768 is not None and s2048 is not None and s768 < s2048)}
    out["G_E31B"] = g_b
    log("  G-E31B  sorted 768 B %.2f vs sorted 2048 B %.2f GB/s -> %s  (E26's known positive: "
        "the S15 row must be the worse one)"
        % (s768, s2048, "FIRES" if g_b["fires"] else "VOID -- E26 saw a difference this "
           "instrument cannot"))

    if not (g_a["fires"] and g_b["fires"]):
        out["VOID"] = ("G-E31A" if not g_a["fires"] else "G-E31B") + " did not fire"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("VOID -- see " + OUT)

    # ---- the verdict cell and the crossover
    out["bookkeeping_diagnostic"] = {
        "WHAT": "contig_ind / contig -- the SAME addresses read through the group loop vs read "
                "flat.  It prices this harness's per-group overhead at each granularity, and "
                "it is the defect that voided run 1 when it was sitting in the DENOMINATOR.  "
                "Reported, never folded into a verdict.",
        "ratio": dict((str(g), tab[(g, "contig_ind")] / tab[(g, "contig")])
                      for g in groups if (g, "contig_ind") in tab and (g, "contig") in tab)}
    r_t10 = tab[(T10_ROW, "sorted")] / tab[(T10_ROW, "contig")]
    name = next(n for lo, n in BANDS if r_t10 >= lo)
    cross = None
    for g in groups:
        if (g, "sorted") in tab and tab[(g, "sorted")] / tab[(g, "contig")] >= CROSSOVER:
            cross = g
            break
    conv = None
    for g in groups:
        if (g, "random") in tab and abs(tab[(g, "random")] / tab[(g, "sorted")] - 1.0) <= 0.05:
            conv = g
            break
    r64 = (tab[(64, "sorted")] / tab[(64, "random")]) if (64, "random") in tab else None

    sparse_budget = r_t10 * (E30_CEIL / GOAL_TOK_S) * 2      # G ternary weights/token
    out["verdict"] = {
        "name": name, "cell": "sorted(2048) / contig(2048)", "r_T10": r_t10,
        "r_S15": tab[(S15_ROW, "sorted")] / tab[(S15_ROW, "contig")],
        "crossover_bytes_at_0.90": cross,
        "crossover_in_T10_rows": (cross / T10_ROW) if cross else None,
        "sorted_over_random_at_64B": r64,
        "sorted_random_converge_at_bytes": conv,
        "dense_budget_G_weights_per_token": E30_CEIL / GOAL_TOK_S * 2,
        "sparse_budget_G_weights_per_token": sparse_budget,
        "T10_dense_G_weights_per_token": 10.6032005,
        "activation_fraction_needed": sparse_budget / 10.6032005}
    v = out["verdict"]
    log("")
    log("  VERDICT CELL  sorted(2048)/contig(2048) = %.3f  ->  %s" % (r_t10, name))
    log("     the S15 row (768 B) reads %.3f of the same control" % v["r_S15"])
    log("     crossover to %.0f%% of dense: %s  (%s T10 rows)"
        % (100 * CROSSOVER, ("%d B" % cross) if cross else "NEVER, inside this sweep",
           ("%.0f" % v["crossover_in_T10_rows"]) if cross else "-"))
    log("     sorted/random at 64 B = %s; they converge within 5%% at %s"
        % (("%.2fx" % r64) if r64 else "-",
           ("%d B" % conv) if conv else "no granularity in this sweep"))
    log("")
    log("     E30's DENSE budget for 50 tok/s:  %.3f G weights/token" % v["dense_budget_G_weights_per_token"])
    log("     the SPARSE budget at row granularity: %.3f G weights/token" % sparse_budget)
    log("     = %.2f%% of T10's dense token -- that is the activation fraction 50 tok/s permits"
        % (100 * v["activation_fraction_needed"]))

    b2, p2 = cpu_busy(2)
    out["cpu_busy_after"] = [b2, p2]
    log("")
    log("  contention witness after: %.1f%% mean / %.0f%% peak" % (b2, p2))
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
