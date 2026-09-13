#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E58 -- where is a trained token's time actually spent, on the engine as it is today?

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E58_WHERE_DID_THE_1_75x_GO.md`
Pre-registered and pushed before this file existed.

E57 addendum D measured 05b_tqh at 21.1 GB/s against a 37.0 GB/s streamed floor and called the
difference headroom.  E58 asks which organ it is in, using `--profile`, which has carried
T_QKV/T_ROPE/T_ATTN/T_O/T_FFN/T_HEAD/T_NORM since E4 and E8's FFN sub-timers under them.

A BOUND TRAP THIS RUNNER EXISTS TO AVOID.  37.0 GB/s is the proj-GEMV STREAMED floor and it was
measured on fp32 reads.  The packed path is a different kernel with a different ceiling --
ledger 23.2/23.3 measures it at ~25.5 GB/s / 51-53 G-weights/s -- so charging a packed organ
against 37.0 compares a rate to the wrong bound, which is the byte-convention law one level up.
Every organ here is therefore scored against the bound for ITS OWN KIND, and both readings are
printed so the substitution is visible rather than assumed.
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

OUTDIR = os.path.join(HERE, "results")
ENGINE = os.path.join(HERE, "donor_engine_e53.exe")
WEIGHTS = r"D:\_ktmp\e1\qwen25-05b_tqh.bin"
THREADS = 6
BENCH_N = 160
REPS = 9
CLOSE_TOL = 0.08                    # G-E58a, registered
SEED = 5801

# ledger 1 (fp32 stream) and 23.2/23.3 (packed kernel), both measured on this box
FP32_FLOOR = 37.0
PACKED_CEIL = 25.5

# shape of qwen25-05b_tqh, read from the engine's own footer line
D, L, F, V = 896, 24, 4864, 151936
KV, HD = 2, 64
W_QKV = L * (D * D + 2 * (D * KV * HD))
W_O = L * (D * D)
W_FFN = L * (3 * D * F)
W_HEAD = V * D
E1_NCODES = 493961216               # E1's stored code count for this artifact

ORGANS = [("qkv_proj", W_QKV, "packed"), ("o_proj", W_O, "packed"),
          ("ffn", W_FFN, "packed"), ("head", W_HEAD, "packed")]
NONWEIGHT = ("rope", "attention", "norm+glue")
ORGAN_NAMES = ("qkv_proj", "rope", "attention", "o_proj", "ffn", "head", "norm+glue")
FFN_NAMES = ("gate+up", "glue(silu)", "down", "residual", "router+select")


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


def boot_ci(v, rng, n=2000):
    b = []
    for _ in range(n):
        s = [v[rng.randrange(len(v))] for _ in v]
        b.append(statistics.median(s))
    b.sort()
    return b[int(0.025 * n)], b[int(0.975 * n) - 1]


# ======================================================================================
def g_e58a(organ_sum_ms, wall_ms):
    """The profile must sum to the thing it decomposes, or no attribution may be read."""
    rel = (organ_sum_ms - wall_ms) / wall_ms
    if abs(rel) <= CLOSE_TOL:
        return "CLOSES", rel
    return "DOES NOT CLOSE", rel


def g_e58b(rows):
    ok = [r for r in rows if r["GB_s"] == r["GB_s"]]
    if not ok:
        return None, None
    return (min(ok, key=lambda r: r["GB_s"]), max(ok, key=lambda r: r["GB_s"]))


def selftest():
    n = [0]

    def chk(name, cond):
        n[0] += 1
        log("  %-56s %s" % (name, "fires" if cond else "*** DEAD ***"))
        if not cond:
            raise SystemExit("a decision function did not fire on a known-positive.  STOP.")

    chk("A-1 a profile that closes exactly CLOSES", g_e58a(10.0, 10.0)[0] == "CLOSES")
    chk("A-2 +7.9% still CLOSES (inside the registered bar)", g_e58a(10.79, 10.0)[0] == "CLOSES")
    chk("A-3 +8.1% DOES NOT CLOSE", g_e58a(10.81, 10.0)[0] == "DOES NOT CLOSE")
    chk("A-4 and it fires on the LOW side too", g_e58a(9.0, 10.0)[0] == "DOES NOT CLOSE")
    lo, hi = g_e58b([{"organ": "a", "GB_s": 10.0}, {"organ": "b", "GB_s": 30.0}])
    chk("B-1 the laggard is the lowest GB/s", lo["organ"] == "a" and hi["organ"] == "b")
    chk("C-1 the shape arithmetic reproduces E1's stored code count",
        W_QKV + W_O + W_FFN + W_HEAD == E1_NCODES)
    chk("C-2 and the parts are the published ones (12.4 / 9.6 / 156.9 / 68.1 MB)",
        abs(W_QKV * 0.5 / 1e6 - 12.4) < 0.05 and abs(W_O * 0.5 / 1e6 - 9.6) < 0.05
        and abs(W_FFN * 0.5 / 1e6 - 156.9) < 0.05 and abs(W_HEAD * 0.5 / 1e6 - 68.1) < 0.05)
    log("")
    log("  %d of %d fire." % (n[0], n[0]))


# ======================================================================================
def one_cell():
    sp = E44.Split()
    cmd = [ENGINE, "--weights", WEIGHTS, "--threads", str(THREADS), "--profile",
           "--bench", str(BENCH_N)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sout, serr = p.communicate()
    child = E44._process_times(int(p._handle))
    if p.returncode != 0:
        sys.stderr.write(serr.decode("utf-8", "replace")[-1200:])
        raise SystemExit("engine failed")
    occ, foc = sp.close(child)
    txt = sout.decode("utf-8", "replace")
    m = E44.RE_BENCH.search(txt)
    if not m:
        raise SystemExit("no BENCH line")
    cell = {"rate": float(m.group(3)), "occ": occ, "foreign": foc, "organs": {}, "ffn": {}}
    cell.update(sp.record())
    sec = 0
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "TOTAL":
            cell["organ_sum_ms"] = float(f[1])
            cell["wall_ms"] = float(f[-2])
            continue
        if line.strip().startswith("-- ffn decomposed"):
            sec = 1
            continue
        if f[0] == "FFN-SUM":
            cell["ffn_sum_ms"] = float(f[1])
            cell["ffn_over_organ"] = float(f[-1].rstrip(")"))
            continue
        # A whitelist, not a shape test: `BENCH  32 tokens ...` also has a number in
        # field 1, and letting it through would have added a fake eighth "organ".
        tgt = FFN_NAMES if sec else ORGAN_NAMES
        if len(f) >= 2 and f[0] in tgt:
            try:
                v = float(f[1])
            except ValueError:
                continue
            (cell["ffn"] if sec else cell["organs"])[f[0]] = v
    if "wall_ms" not in cell or len(cell["organs"]) < 7:
        raise SystemExit("could not parse the profile table")
    return cell


def main():
    log("E58 -- where did the 1.75x go?  The organ table, on the engine as it is today.")
    log("brief: BRIEF_E58_WHERE_DID_THE_1_75x_GO.md")
    log("")
    log("== self-test ==")
    selftest()
    if "--selftest" in sys.argv:
        return

    log("")
    log("== %d reps, %s, --bench %d, --profile, idle box =="
        % (REPS, os.path.basename(WEIGHTS), BENCH_N))
    log("   warm-up cell, DISCARDED: %.2f tok/s" % one_cell()["rate"])
    cells = [one_cell() for _ in range(REPS)]
    rng = random.Random(SEED)

    rate = [c["rate"] for c in cells]
    wall = [c["wall_ms"] for c in cells]
    osum = [c["organ_sum_ms"] for c in cells]
    lo, hi = boot_ci(rate, rng)
    log("")
    log("   rate    median %.2f tok/s  [%.2f, %.2f]  IQR %.2f%%  (%d cells)"
        % (statistics.median(rate), lo, hi, iqr_pct(rate), len(rate)))
    log("   foreign median %.2f%%   clock median %.1f%%"
        % (statistics.median([c["foreign"] for c in cells]),
           statistics.median([c["clock_pct"] for c in cells
                              if c["clock_pct"] == c["clock_pct"]] or [float("nan")])))
    log("")

    v, rel = g_e58a(statistics.median(osum), statistics.median(wall))
    log("== G-E58a -- the profile must sum to the token ==")
    log("   organs summed %.3f ms   wall %.3f ms   %+.2f%%   bar +-%.0f%%"
        % (statistics.median(osum), statistics.median(wall), 100 * rel, 100 * CLOSE_TOL))
    log("   G-E58a : %s" % v)
    if v != "CLOSES":
        log("   No organ attribution may be read.  STOP.")
        raise SystemExit("G-E58a did not close")
    log("")

    rows = []
    for name, w, kind in ORGANS:
        ms = [c["organs"][name] for c in cells]
        mb = w * 0.5 / 1e6
        gbs = mb * 1e6 / (statistics.median(ms) / 1e3) / 1e9
        rows.append({"organ": name, "weights": w, "MB": mb, "ms": statistics.median(ms),
                     "ms_iqr_pct": iqr_pct(ms), "GB_s": gbs, "kind": kind,
                     "vs_packed": PACKED_CEIL / gbs, "vs_fp32": FP32_FLOOR / gbs})
    nw = {}
    for name in NONWEIGHT:
        nw[name] = statistics.median([c["organs"][name] for c in cells])

    log("== G-E58b -- the laggard, named on bytes ==")
    log("   %-10s %10s %9s %7s %9s  %8s %8s"
        % ("organ", "MB/token", "ms", "IQR", "GB/s", "x to 25.5", "x to 37"))
    for r in rows:
        log("   %-10s %10.1f %9.3f %6.1f%% %9.1f  %8.2f %8.2f"
            % (r["organ"], r["MB"], r["ms"], r["ms_iqr_pct"], r["GB_s"],
               r["vs_packed"], r["vs_fp32"]))
    lag, fast = g_e58b(rows)
    tot_mb = sum(r["MB"] for r in rows)
    tot_ms = sum(r["ms"] for r in rows)
    log("   %-10s %10.1f %9.3f %6s %9.1f" % ("TOTAL", tot_mb, tot_ms, "",
                                             tot_mb * 1e6 / (tot_ms / 1e3) / 1e9))
    log("")
    log("   laggard : %s at %.1f GB/s      fastest : %s at %.1f GB/s"
        % (lag["organ"], lag["GB_s"], fast["organ"], fast["GB_s"]))
    ceil_ms = tot_mb * 1e6 / (fast["GB_s"] * 1e9) * 1e3
    rem_ms = statistics.median(wall) - tot_ms
    log("   if every organ reached %s's %.1f GB/s: %.3f ms of weights + %.3f ms of the rest"
        % (fast["organ"], fast["GB_s"], ceil_ms, max(rem_ms, 0.0)))
    log("   = %.3f ms/token = %.2f tok/s  (x%.2f on the measured %.2f)"
        % (ceil_ms + max(rem_ms, 0.0), 1000.0 / (ceil_ms + max(rem_ms, 0.0)),
           (statistics.median(wall)) / (ceil_ms + max(rem_ms, 0.0)),
           statistics.median(rate)))
    log("")

    log("== G-E58c -- Rem, the non-weight remainder ==")
    for name in NONWEIGHT:
        log("   %-10s %9.3f ms   %5.1f%% of the token"
            % (name, nw[name], 100 * nw[name] / statistics.median(wall)))
    log("   %-10s %9.3f ms   %5.1f%% of the token   (wall minus the four weight organs)"
        % ("Rem", rem_ms, 100 * rem_ms / statistics.median(wall)))
    log("")
    w_at_floor = tot_mb * 1e6 / (FP32_FLOOR * 1e9) * 1e3
    w_at_packed = tot_mb * 1e6 / (PACKED_CEIL * 1e9) * 1e3
    log("   the brief registered: at the 37.0 GB/s fp32 floor the weights would be %.2f ms of an"
        % w_at_floor)
    log("   %.2f ms token, leaving %.2f ms for Rem to explain.  Measured Rem is %.3f ms."
        % (statistics.median(wall), statistics.median(wall) - w_at_floor, rem_ms))
    log("   Against the PACKED ceiling of %.1f GB/s the weights would be %.2f ms, i.e. the token"
        % (PACKED_CEIL, w_at_packed))
    log("   would be %.2f ms = %.1f tok/s -- x%.2f, not x%.2f."
        % (w_at_packed + max(rem_ms, 0.0), 1000.0 / (w_at_packed + max(rem_ms, 0.0)),
           statistics.median(wall) / (w_at_packed + max(rem_ms, 0.0)),
           statistics.median(wall) / (w_at_floor + max(rem_ms, 0.0))))
    log("")

    ffn = {k: statistics.median([c["ffn"].get(k, float("nan")) for c in cells])
           for k in cells[0]["ffn"]}
    log("== the FFN, decomposed (E8's sub-timers; %.1f%% of the token) =="
        % (100 * statistics.median([c["organs"]["ffn"] for c in cells])
           / statistics.median(wall)))
    for k, vv in sorted(ffn.items(), key=lambda x: -x[1]):
        log("   %-14s %8.3f ms" % (k, vv))
    log("   sum/ffn = %.4f" % statistics.median([c["ffn_over_organ"] for c in cells]))
    log("")
    log("== G-E58d -- nothing is promoted.  This is a MAP, not a change. ==")

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    path = os.path.join(OUTDIR, "e58_organs.json")
    json.dump({"brief": "BRIEF_E58_WHERE_DID_THE_1_75x_GO.md", "weights": WEIGHTS,
               "bench_n": BENCH_N, "reps": REPS, "threads": THREADS,
               "fp32_floor": FP32_FLOOR, "packed_ceiling": PACKED_CEIL,
               "rate_median": statistics.median(rate), "rate_ci": [lo, hi],
               "rate_iqr_pct": iqr_pct(rate), "wall_ms": statistics.median(wall),
               "organ_sum_ms": statistics.median(osum), "g_e58a": v, "close_rel": rel,
               "rows": rows, "nonweight_ms": nw, "rem_ms": rem_ms, "ffn": ffn,
               "cells": cells}, open(path, "w"), indent=1)
    log("wrote %s" % path)


if __name__ == "__main__":
    main()
