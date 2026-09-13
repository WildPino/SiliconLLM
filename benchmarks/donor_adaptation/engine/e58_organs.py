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
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e44_interval as E44                                          # noqa: E402

OUTDIR = os.path.join(HERE, "results")
ENGINE = os.path.join(HERE, "donor_engine_e53.exe")
THREADS = 6
BENCH_N = 160
REPS = 9
CLOSE_TOL = 0.08                    # G-E58a, registered
SEED = 5801

# ledger 1 (fp32 stream) and 23.2/23.3 (packed kernel), both measured on this box
FP32_FLOOR = 37.0
PACKED_CEIL = 25.5

# E1's stored code count per artifact -- the file the shape arithmetic is checked against
E1_NCODES = {"qwen25-05b_tqh.bin": 493961216, "qwen25-15b_tqh.bin": 1543569408}

# the registered arm (brief sec.3).  A second arm may only appear in a clearly marked
# UNREGISTERED EXTENSION; the sec.4 gates are scored on this one.
WEIGHTS = os.path.join("D:", os.sep, "_ktmp", "e1", "qwen25-05b_tqh.bin")
WEIGHTS_15B = os.path.join("D:", os.sep, "_ktmp", "e1", "qwen25-15b_tqh.bin")

RE_SHAPE = re.compile(r"D=(\d+)\s+F=(\d+)\s+L=(\d+)\s+heads=(\d+)/(\d+)\s+hd=(\d+)\s+V=(\d+)")

ORGAN_NAMES = ("qkv_proj", "rope", "attention", "o_proj", "ffn", "head", "norm+glue")
FFN_NAMES = ("gate+up", "glue(silu)", "down", "residual", "router+select")
NONWEIGHT = ("rope", "attention", "norm+glue")


def organs_from_shape(sh):
    """Weights streamed per decoded token, per organ.

    Derived from the shape the ENGINE prints, never from a table in a document: the
    stale sec.12 organ table is exactly what this experiment exists to replace, and a
    hardcoded shape would have carried its assumptions in through the back door.
    """
    D, F, L, H, KV, HD, V = sh
    return [("qkv_proj", L * (D * D + 2 * (D * KV * HD))),
            ("o_proj",   L * (D * D)),
            ("ffn",      L * (3 * D * F)),
            ("head",     V * D)]


# the sub-matrices inside T_FFN, so the FFN can be read in G-weights/s against E10
FFN_SHARE = {"gate+up": 2.0 / 3.0, "down": 1.0 / 3.0}


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
    o05 = dict(organs_from_shape((896, 4864, 24, 14, 2, 64, 151936)))
    o15 = dict(organs_from_shape((1536, 8960, 28, 12, 2, 128, 151936)))
    chk("C-1 the 0.5B shape arithmetic reproduces E1's stored code count",
        sum(o05.values()) == E1_NCODES["qwen25-05b_tqh.bin"])
    chk("C-2 and the parts are the published ones (12.4 / 9.6 / 156.9 / 68.1 MB)",
        abs(o05["qkv_proj"] * .5 / 1e6 - 12.4) < .05 and abs(o05["o_proj"] * .5 / 1e6 - 9.6) < .05
        and abs(o05["ffn"] * .5 / 1e6 - 156.9) < .05 and abs(o05["head"] * .5 / 1e6 - 68.1) < .05)
    chk("C-3 and the SAME arithmetic closes on the 1.5B artifact too",
        sum(o15.values()) == E1_NCODES["qwen25-15b_tqh.bin"])
    log("")
    log("  %d of %d fire." % (n[0], n[0]))


# ======================================================================================
def one_cell(weights):
    sp = E44.Split()
    cmd = [ENGINE, "--weights", weights, "--threads", str(THREADS), "--profile",
           "--bench", str(BENCH_N)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sout, serr = p.communicate()
    child = E44._process_times(int(p._handle))
    if p.returncode != 0:
        sys.stderr.write(serr.decode("utf-8", "replace")[-1200:])
        raise SystemExit("engine failed")
    occ, foc = sp.close(child)
    # The profile table goes to STDOUT and the shape footer the engine prints about ITSELF
    # goes to STDERR.  A runner that captured only stdout would have timed an artifact
    # without ever seeing which one it was -- the same defect class as
    # feedback_config_must_appear_in_output, one stream over.  Parse both.
    txt = sout.decode("utf-8", "replace") + os.linesep + serr.decode("utf-8", "replace")
    m = E44.RE_BENCH.search(txt)
    if not m:
        raise SystemExit("no BENCH line")
    ms = RE_SHAPE.search(txt)
    if not ms:
        raise SystemExit("no shape footer -- the byte table cannot be derived.  STOP.")
    cell = {"rate": float(m.group(3)), "occ": occ, "foreign": foc,
            "shape": tuple(int(x) for x in ms.groups()), "organs": {}, "ffn": {}}
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
        # A whitelist, not a shape test: `BENCH  160 tokens ...` also has a number in
        # field 1, and letting it through would have added a fake eighth "organ".
        tgt = FFN_NAMES if sec else ORGAN_NAMES
        if len(f) >= 2 and f[0] in tgt:
            try:
                (cell["ffn"] if sec else cell["organs"])[f[0]] = float(f[1])
            except ValueError:
                continue
    if "wall_ms" not in cell or len(cell["organs"]) < 7:
        raise SystemExit("could not parse the profile table")
    return cell


def measure(weights, label, registered):
    base = os.path.basename(weights)
    log("")
    log("=" * 86)
    log("== %s -- %s, %d reps, --bench %d, --profile, idle box ==" % (label, base, REPS, BENCH_N))
    if not registered:
        log("   UNREGISTERED EXTENSION.  Brief sec.3 registers ONE artifact; the sec.4 gates are")
        log("   scored on that one.  This arm exists only to repair E57 D.3's published transfer")
        log("   number, which was computed against a bound sec.23.3 had already withdrawn.")
    log("=" * 86)
    log("   warm-up cell, DISCARDED: %.2f tok/s" % one_cell(weights)["rate"])
    cells = [one_cell(weights) for _ in range(REPS)]
    rng = random.Random(SEED)
    shape = cells[0]["shape"]
    for c in cells:
        if c["shape"] != shape:
            raise SystemExit("the engine printed two different shapes.  STOP.")
    organs = organs_from_shape(shape)
    n_codes = sum(w for _, w in organs)
    exp = E1_NCODES.get(base)
    log("   shape D=%d F=%d L=%d heads=%d/%d hd=%d V=%d" % shape)
    log("   derived weights/token %d   E1 n_codes %s   %s"
        % (n_codes, exp, "MATCH" if exp == n_codes else "*** MISMATCH ***"))
    if exp is not None and exp != n_codes:
        raise SystemExit("the shape arithmetic does not reproduce E1's file.  STOP.")

    rate = [c["rate"] for c in cells]
    wall = [c["wall_ms"] for c in cells]
    osum = [c["organ_sum_ms"] for c in cells]
    lo, hi = boot_ci(rate, rng)
    med_wall = statistics.median(wall)
    log("")
    log("   rate    median %.2f tok/s  [%.2f, %.2f]  IQR %.2f%%  (%d cells)"
        % (statistics.median(rate), lo, hi, iqr_pct(rate), len(rate)))
    log("   foreign median %.2f%%   clock median %.1f%%"
        % (statistics.median([c["foreign"] for c in cells]),
           statistics.median([c["clock_pct"] for c in cells])))
    log("")

    v, rel = g_e58a(statistics.median(osum), med_wall)
    log("== G-E58a -- the profile must sum to the token ==")
    log("   organs summed %.3f ms   wall %.3f ms   %+.2f%%   bar +-%.0f%%"
        % (statistics.median(osum), med_wall, 100 * rel, 100 * CLOSE_TOL))
    log("   G-E58a : %s" % v)
    if v != "CLOSES":
        log("   No organ attribution may be read.  STOP.")
        raise SystemExit("G-E58a did not close")
    log("")

    rows = []
    for name, w in organs:
        ms = [c["organs"][name] for c in cells]
        mms = statistics.median(ms)
        mb = w * 0.5 / 1e6
        rows.append({"organ": name, "weights": w, "MB": mb, "ms": mms,
                     "ms_iqr_pct": iqr_pct(ms), "GB_s": mb * 1e6 / (mms / 1e3) / 1e9,
                     "Gw_s": w / (mms / 1e3) / 1e9})
    nw = {k: statistics.median([c["organs"][k] for c in cells]) for k in NONWEIGHT}

    log("== G-E58b -- the laggard, named on bytes ==")
    log("   Both bounds are printed.  37.0 GB/s is the fp32 proj-GEMV floor; ledger sec.23.3")
    log("   WITHDREW it for the packed path, which E10 measures CORE-BOUND at 25.5 GB/s /")
    log("   49.1-53.0 G-weights/s flat across a 512x footprint change.  G-w/s is the column")
    log("   that compares to E10; GB/s is the column E57 D.3 used.")
    log("")
    log("   %-10s %9s %8s %6s %8s %8s  %8s %7s"
        % ("organ", "MB/tok", "ms", "IQR", "GB/s", "G-w/s", "x to 25.5", "x to 37"))
    for r in rows:
        log("   %-10s %9.1f %8.3f %5.1f%% %8.1f %8.2f  %8.2f %7.2f"
            % (r["organ"], r["MB"], r["ms"], r["ms_iqr_pct"], r["GB_s"], r["Gw_s"],
               PACKED_CEIL / r["GB_s"], FP32_FLOOR / r["GB_s"]))
    tot_mb = sum(r["MB"] for r in rows)
    tot_ms = sum(r["ms"] for r in rows)
    log("   %-10s %9.1f %8.3f %5s %8.1f %8.2f"
        % ("TOTAL", tot_mb, tot_ms, "", tot_mb * 1e6 / (tot_ms / 1e3) / 1e9,
           n_codes / (tot_ms / 1e3) / 1e9))
    lag, fast = g_e58b(rows)
    rem_ms = med_wall - tot_ms
    log("")
    log("   laggard : %s at %.1f GB/s / %.2f G-w/s      fastest : %s at %.1f / %.2f"
        % (lag["organ"], lag["GB_s"], lag["Gw_s"], fast["organ"], fast["GB_s"], fast["Gw_s"]))
    for nm, gw in (("the fastest organ here", fast["Gw_s"]), ("E10's best kernel cell", 52.95)):
        cms = n_codes / (gw * 1e9) * 1e3
        tot = cms + max(rem_ms, 0.0)
        log("   ceiling if every organ reached %-22s %5.2f G-w/s : %6.3f ms = %6.2f tok/s  x%.2f"
            % (nm, gw, tot, 1000.0 / tot, med_wall / tot))
    log("")

    log("== G-E58c -- Rem, the non-weight remainder ==")
    for k in NONWEIGHT:
        log("   %-10s %8.3f ms   %5.1f%% of the token" % (k, nw[k], 100 * nw[k] / med_wall))
    log("   %-10s %8.3f ms   %5.1f%% of the token   (wall minus the four weight organs)"
        % ("Rem", rem_ms, 100 * rem_ms / med_wall))
    w37 = tot_mb * 1e6 / (FP32_FLOOR * 1e9) * 1e3
    log("   brief sec.4: at the 37.0 GB/s floor the weights would be %.2f ms of a %.2f ms token,"
        % (w37, med_wall))
    log("   leaving %.2f ms for Rem to explain.  Measured Rem is %.3f ms -- %.0fx smaller."
        % (med_wall - w37, rem_ms, (med_wall - w37) / rem_ms))
    log("")

    ffn = {k: statistics.median([c["ffn"].get(k, float("nan")) for c in cells])
           for k in cells[0]["ffn"]}
    ffn_ms = statistics.median([c["organs"]["ffn"] for c in cells])
    ffn_w = dict(organs)["ffn"]
    log("== the FFN, decomposed (E8's sub-timers; %.1f%% of the token) =="
        % (100 * ffn_ms / med_wall))
    for k, vv in sorted(ffn.items(), key=lambda x: -x[1]):
        sh = FFN_SHARE.get(k)
        extra = ("  %6.2f G-w/s" % (ffn_w * sh / (vv / 1e3) / 1e9)) if sh else ""
        log("   %-14s %8.3f ms%s" % (k, vv, extra))
    log("   sum/ffn = %.4f" % statistics.median([c["ffn_over_organ"] for c in cells]))

    return {"label": label, "weights": weights, "registered": registered, "shape": shape,
            "n_codes": n_codes, "rate_median": statistics.median(rate), "rate_ci": [lo, hi],
            "rate_iqr_pct": iqr_pct(rate), "wall_ms": med_wall,
            "organ_sum_ms": statistics.median(osum), "g_e58a": v, "close_rel": rel,
            "rows": rows, "nonweight_ms": nw, "rem_ms": rem_ms, "ffn": ffn, "cells": cells}


def main():
    log("E58 -- where did the 1.75x go?  The organ table, on the engine as it is today.")
    log("brief: BRIEF_E58_WHERE_DID_THE_1_75x_GO.md")
    log("")
    log("== self-test ==")
    selftest()
    if "--selftest" in sys.argv:
        return

    out = [measure(WEIGHTS, "REGISTERED ARM", True)]
    if "--with-15b" in sys.argv:
        out.append(measure(WEIGHTS_15B, "EXTENSION: 1.5B", False))

    log("")
    log("== G-E58d -- nothing is promoted.  This is a MAP, not a change. ==")
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    path = os.path.join(OUTDIR, "e58_organs.json")
    json.dump({"brief": "BRIEF_E58_WHERE_DID_THE_1_75x_GO.md", "bench_n": BENCH_N,
               "reps": REPS, "threads": THREADS, "engine": os.path.basename(ENGINE),
               "fp32_floor": FP32_FLOOR, "packed_ceiling": PACKED_CEIL,
               "e10_kernel_Gw_s": [49.11, 52.95], "arms": out},
              open(path, "w"), indent=1)
    log("wrote %s" % path)


if __name__ == "__main__":
    main()
