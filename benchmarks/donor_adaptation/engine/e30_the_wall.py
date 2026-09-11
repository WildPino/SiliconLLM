#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E30 -- is the engine still core-bound, or has the fast kernel put it against the memory wall?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E30_THE_WALL.md, pushed at 63515b4 BEFORE this
file existed.  Nothing here may contradict it; the result document scores it as written.

THE QUESTION.  E28 broke the constant this programme priced the goal against -- the numerator is
61.64 G active weights/s at T10 on --lutblk, not the 49.9 that has stood since E10.  The goal
needs 530.  Is the remaining 8.60x available to kernel work at all?

EVERYTHING HERE IS IN THE MOVED-BYTE CONVENTION, on both sides of every ratio, because the
standing law says charged weights and moved bytes never meet inside a fraction.  The engine's
byte rate is divided by a ceiling MEASURED ON THIS BOX (e30_bandwidth.exe), never by a spec sheet.

  python e30_the_wall.py --dir D:/_ktmp/e28 --engine ./donor_engine_e26.exe --reps 3
  python e30_the_wall.py --bw-only            # the sweep and its planted control, nothing else
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e26_carve_cost import cpu_busy, IDLE_BAR                        # noqa: E402
from e28_kernel_transfer import bench, build_t10_packed, winpath, RE_BENCH  # noqa: E402

OUT = os.path.join(HERE, "results", "e30_the_wall.json")
BW_EXE = os.path.join(HERE, "e30_bandwidth.exe")
BW_SRC = os.path.join(HERE, "e30_bandwidth.c")

# E28's own published rates, the anchors this run must reproduce before its ratio is licensed.
E28_T10_PACKED = 4.3533333333333335
E28_T10_LUTBLK = 5.813333333333333
ANCHOR_TOL = 0.10

# T10, the goal's shape.  Same numbers E25/E27/E28 use.
T10 = dict(D=4096, QD=4096, KD=1024, F=14336, L=48, V=32768)
GOAL_TOK_S = 50.0

# The bands, named in the brief before the run.
BANDS = [(0.85, "AT-THE-WALL"), (0.60, "PARTIALLY-BOUND"), (0.0, "STILL-CORE-BOUND")]
CLIFF_MIN = 3.0          # G-E30A: L3-resident must read at least this many times the DRAM plateau


def log(*a):
    print(*a, flush=True)
    sys.stdout.flush()


def moved_bytes_per_token(s):
    """Weight BYTES the engine reads to produce one token, from the shape.

    Packed ternary is 2 trits per byte, every matrix carries one fp32 scale per output row,
    q/k/v each carry an fp32 bias vector, and each layer has two fp32 norm vectors.  The head is
    read in full; the embedding TABLE is not -- exactly one row of it is.

    THE BRIEF'S SS1 TABLE WAS SHORT.  It omitted the q/k/v biases -- (QD + 2*KD) * 4 = 24,576
    bytes per layer at T10 -- and so read 5.3116 GB/token where the file says 5.31276.  That is
    +0.022% and changes no conclusion, but it is exactly what G-E30B exists to catch, and the
    gate is stated in the brief as reproducing the file TO THE BYTE.  It now does: header +
    embedding table + layers + final norm + head == os.path.getsize(), difference zero.
    """
    D, QD, KD, F, L, V = s["D"], s["QD"], s["KD"], s["F"], s["L"], s["V"]

    def mat(out_, in_):
        return out_ * in_ // 2 + out_ * 4          # codes + fp32 row scales

    per_layer = (mat(QD, D) + mat(D, QD) +         # q, o
                 2 * mat(KD, D) +                  # k, v
                 (QD + 2 * KD) * 4 +               # q, k, v fp32 biases
                 2 * mat(F, D) + mat(D, F) +       # gate, up, down
                 2 * D * 4)                        # input_norm, post_norm, fp32
    head = mat(V, D)
    return {"per_layer": per_layer, "layers": per_layer * L, "head": head,
            "embed_row": D * 4, "total": per_layer * L + head + D * 4,
            "embed_table_fp32": V * D * 4, "header": 8 + 9 * 4 + 2 * 4, "final_norm": D * 4}


def run_bw(threads, reps):
    if not os.path.exists(BW_EXE):
        raise SystemExit("%s is missing -- build it with the engine's own flags:\n"
                         "  clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp %s -o %s -lm"
                         % (BW_EXE, os.path.basename(BW_SRC), os.path.basename(BW_EXE)))
    r = subprocess.run([BW_EXE, str(threads), str(reps)], capture_output=True)
    if r.returncode != 0:
        log(r.stderr.decode(errors="replace")[-1500:])
        raise SystemExit("the bandwidth bench failed")
    rows = []
    for ln in r.stdout.decode(errors="replace").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        b, mb, best, med = ln.split(",")
        rows.append({"bytes": int(b), "MB": float(mb), "best": float(best),
                     "median": float(med)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e28")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ntok", type=int, default=40)
    ap.add_argument("--bw-only", action="store_true")
    a = ap.parse_args()
    d = winpath(a.dir)
    engine = a.engine if os.path.isabs(a.engine) else os.path.join(HERE, a.engine)
    t_start = time.time()

    log("== E30: is the engine still core-bound, or against the memory wall? ==")
    log("  engine %s   threads %d   reps %d" % (os.path.basename(engine), a.threads, a.reps))

    busy, peak = cpu_busy(4)
    log("")
    log("  contention witness: %.1f%% mean / %.0f%% peak before the run (bar %.0f%% on the MEAN)"
        % (busy, peak, IDLE_BAR))
    if busy > IDLE_BAR and not a.bw_only:
        raise SystemExit("the box is at %.1f%% -- a contended timing is not a timing.  STOP."
                         % busy)

    out = {"brief": "briefs/BRIEF_E30_THE_WALL.md (63515b4)",
           "question": "E28 broke the 49.9 constant.  How much of the remaining 8.60x to the "
                       "goal is available to KERNEL work at all?  Everything is in the "
                       "MOVED-BYTE convention and the ceiling is measured on this box.",
           "threads": a.threads, "reps": a.reps, "ntok": a.ntok, "T10": T10,
           "goal_tok_s": GOAL_TOK_S, "idle_bar_pct": IDLE_BAR,
           "cpu_busy_mean_pct": [busy], "cpu_busy_peak_pct": [peak], "bands": BANDS}

    # ---------------------------------------------------------------- the ceiling
    log("")
    log("  -- BW-SWEEP: read bandwidth, 4 MB -> 8 GB, %d threads, %d reps" % (a.threads, a.reps))
    rows = run_bw(a.threads, a.reps)
    for r in rows:
        log("     %7.0f MB   best %8.2f GB/s   median %8.2f GB/s"
            % (r["MB"], r["best"], r["median"]))
    out["bw_sweep"] = rows

    big = [r["median"] for r in rows if r["bytes"] >= (1 << 30)]
    if not big:
        raise SystemExit("the sweep never reached 1 GB -- no DRAM plateau, no ceiling")
    big.sort()
    ceil_gbs = big[len(big) // 2]
    # L3-resident: the best any size at or below 16 MB delivered.  BEST and not median on
    # purpose -- at 4-16 MB a pass takes tens of microseconds and the median is dominated by
    # OpenMP fork cost, which is a property of the harness and not of the cache.
    resident = max(r["best"] for r in rows if r["bytes"] <= 16 * 1048576)
    cliff = resident / ceil_gbs
    g_a = {"resident_best_GB_s": resident, "dram_plateau_GB_s": ceil_gbs,
           "ratio": cliff, "bar": CLIFF_MIN, "fires": bool(cliff >= CLIFF_MIN)}
    out["G_E30A"] = g_a
    out["bw_ceiling_GB_s"] = ceil_gbs
    log("")
    log("     G-E30A  L3-resident %.1f GB/s vs DRAM plateau %.2f GB/s = %.1fx (bar %.1fx) -> %s"
        % (resident, ceil_gbs, cliff, CLIFF_MIN, "FIRES" if g_a["fires"] else "VOID"))
    if not g_a["fires"]:
        out["VOID"] = "G-E30A did not fire: the bench cannot tell L3 from DRAM, so it is not " \
                      "measuring memory bandwidth and reports no ceiling"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E30A VOID -- see " + OUT)

    if a.bw_only:
        out["seconds"] = time.time() - t_start
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("")
        log("  --bw-only: ceiling %.2f GB/s recorded, no engine arm run" % ceil_gbs)
        return 0

    # ---------------------------------------------------------------- the accounting
    w = build_t10_packed(d)
    mb = moved_bytes_per_token(T10)
    fsize = os.path.getsize(w)
    predicted_file = (mb["header"] + mb["embed_table_fp32"] + mb["layers"]
                      + mb["final_norm"] + mb["head"])
    slack = fsize - predicted_file
    g_b = {"per_token_bytes": mb["total"], "per_token_GB": mb["total"] / 1e9,
           "file_bytes": fsize, "accounted_file_bytes": predicted_file,
           "unaccounted_bytes": slack, "tol_bytes": 0,
           "fires": bool(slack == 0), "parts": mb,
           "brief_table_GB": 5.3116,
           "note": "the brief's SS1 table omitted the q/k/v fp32 biases, 24576 B/layer"}
    out["G_E30B"] = g_b
    log("")
    log("  -- G-E30B: the byte accounting must reproduce the artifact")
    log("     per token   %s bytes  = %.4f GB" % (f"{mb['total']:,}", mb["total"] / 1e9))
    log("     file        %s bytes, accounted %s, unaccounted %+d (must be zero)"
        % (f"{fsize:,}", f"{predicted_file:,}", slack))
    log("     -> %s" % ("FIRES" if g_b["fires"] else "MISMATCH -- the accounting is wrong"))
    if not g_b["fires"]:
        out["VOID"] = "G-E30B: the byte accounting does not reproduce the artifact"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E30B MISMATCH -- see " + OUT)

    # ---------------------------------------------------------------- the engine arms
    arms = [("T10-PACKED", []), ("T10-LUTBLK", ["--lutblk"])]
    rates = {t: [] for t, _ in arms}
    log("")
    for r in range(a.reps):
        for tag, flags in arms:                      # interleaved, so drift is common-mode
            v = bench(engine, w, a.ntok, a.threads, flags)
            rates[tag].append(v)
            log("  rep %d  %-12s %7.2f tok/s   %7.2f GB/s"
                % (r + 1, tag, v, v * mb["total"] / 1e9))
        b2, p2 = cpu_busy(2)
        out["cpu_busy_mean_pct"].append(b2)
        out["cpu_busy_peak_pct"].append(p2)
        log("  rep %d  contention witness: %.1f%% mean / %.0f%% peak" % (r + 1, b2, p2))

    out["arms"] = {}
    for tag, _ in arms:
        v = rates[tag]
        mean = sum(v) / len(v)
        out["arms"][tag] = {"rates": v, "mean_tok_s": mean,
                            "spread": (max(v) - min(v)) / mean,
                            "GB_per_s": mean * mb["total"] / 1e9}

    anch = {}
    for tag, pub in (("T10-PACKED", E28_T10_PACKED), ("T10-LUTBLK", E28_T10_LUTBLK)):
        m = out["arms"][tag]["mean_tok_s"]
        dev = m / pub - 1.0
        anch[tag] = {"measured": m, "e28_published": pub, "rel_dev": dev,
                     "tol": ANCHOR_TOL, "passes": bool(abs(dev) <= ANCHOR_TOL)}
        log("  ANCHOR  %-12s reads %.2f against E28's %.2f (%+.2f%%, bar +-%.0f%%) -> %s"
            % (tag, m, pub, 100 * dev, 100 * ANCHOR_TOL,
               "PASS" if anch[tag]["passes"] else "FAIL"))
    out["G_E30C"] = anch
    if not all(x["passes"] for x in anch.values()):
        out["VOID_AS_A_TIMING"] = True
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E30C: an anchor missed -- this session is not comparable to E28")
    out["VOID_AS_A_TIMING"] = False

    # ---------------------------------------------------------------- the verdict
    eng = out["arms"]["T10-LUTBLK"]["GB_per_s"]
    ratio = eng / ceil_gbs
    name = next(n for lo, n in BANDS if ratio >= lo)
    goal_gbs = GOAL_TOK_S * mb["total"] / 1e9
    kernel_ceiling_tok_s = ceil_gbs / (mb["total"] / 1e9)
    out["verdict"] = {
        "name": name, "cell": "T10-LUTBLK GB/s / BW-CEIL",
        "engine_GB_s": eng, "ceiling_GB_s": ceil_gbs, "ratio": ratio,
        "packed_ratio": out["arms"]["T10-PACKED"]["GB_per_s"] / ceil_gbs,
        "headroom_left_x": ceil_gbs / eng,
        "tok_s_if_perfect_kernel": kernel_ceiling_tok_s,
        "goal_needs_GB_s": goal_gbs,
        "goal_over_ceiling_x": goal_gbs / ceil_gbs,
        "bytes_per_token_for_goal_GB": ceil_gbs / GOAL_TOK_S,
        "ternary_G_weights_per_token_for_goal": ceil_gbs / GOAL_TOK_S * 2}
    log("")
    log("  VERDICT CELL  %.2f / %.2f GB/s = %.3f  ->  %s" % (eng, ceil_gbs, ratio, name))
    log("     packed sits at %.3f of the same ceiling (E10 called that core-bound)"
        % out["verdict"]["packed_ratio"])
    log("     an INFINITELY good kernel at this shape reads %.2f tok/s -- the hard ceiling of"
        % kernel_ceiling_tok_s)
    log("     all future kernel work on this box, %.2fx from here."
        % out["verdict"]["headroom_left_x"])
    log("")
    log("     50 tok/s at this shape needs %.1f GB/s = %.2fx THE WHOLE MACHINE'S BANDWIDTH."
        % (goal_gbs, out["verdict"]["goal_over_ceiling_x"]))
    log("     At the wall, 50 tok/s permits %.3f GB/token = %.2f G ternary weights/token."
        % (out["verdict"]["bytes_per_token_for_goal_GB"],
           out["verdict"]["bytes_per_token_for_goal_GB"] * 2))

    out["seconds"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
