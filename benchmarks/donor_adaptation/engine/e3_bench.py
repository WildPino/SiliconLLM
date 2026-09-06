#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3 -- run donor_engine --bench with repetitions, and report median + IQR.

Brief: BRIEF_E3_ENGINE_AT_TARGET_SCALE.md (d4937a2) s4: "Idle machine, >=3 reps per point,
median reported with IQR".  This file exists so that no timing in E3 is a single sample.
The organ table is taken from the MEDIAN rep, not averaged across reps: averaging ms/token
across runs of different total speed would mix two different denominators.
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "donor_engine.exe")
ORGANS = ("qkv_proj", "rope", "attention", "o_proj", "ffn", "head", "norm+glue")


def one(weights, bench, threads, extra, exe=None):
    cmd = [exe or EXE, "--bench", str(bench), "--weights", weights, "--threads", str(threads), "--profile"]
    cmd += list(extra)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("engine failed (%d):\n%s\n%s" % (p.returncode, p.stdout[-2000:], p.stderr[-2000:]))
    out = p.stdout
    m = re.search(r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s", out)
    if not m:
        raise SystemExit("no BENCH line:\n" + out[-2000:])
    toks = float(m.group(3))
    if not (toks > 0 and toks == toks and toks < 1e6):
        raise SystemExit("BENCH line is not a finite rate: %r" % m.group(0))
    org = {}
    for k in ORGANS:
        mm = re.search(r"^\s+%s\s+([\d.]+)\s" % re.escape(k), out, re.M)
        if mm:
            org[k] = float(mm.group(1))
    mw = re.search(r"TOTAL\s+([\d.]+)\s+\(organs summed; wall ([\d.]+) ms/token\)", out)
    return {"tok_s": toks, "wall_s": float(m.group(2)), "organs": org,
            "organs_total_ms": float(mw.group(1)) if mw else None,
            "wall_ms_token": float(mw.group(2)) if mw else None,
            "elapsed_s": time.time() - t0}


def quart(v):
    """Median and IQR of a small sample, linear interpolation (numpy's default)."""
    import numpy as np
    a = np.asarray(v, dtype=float)
    q1, q2, q3 = (float(np.percentile(a, p)) for p in (25, 50, 75))
    return q2, q3 - q1, q1, q3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bench", type=int, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--extra", nargs="*", default=[])
    # E4: the arm is a value, not a free-form passthrough -- argparse cannot carry "--attn avx4"
    # inside --extra (it reads the leading "--" as an option of its own).
    ap.add_argument("--attn", default="", help="E4 arm passed to donor_engine --attn")
    # E5: the R-decomposition arm, orthogonal to --attn and composed with it on every point.
    ap.add_argument("--attnr", default="", help="E5 arm passed to donor_engine --attnr")
    # E4 s8.2: to tell a code change from a machine that has drifted, the OLD binary must be
    # runnable in the SAME session as the new one.
    ap.add_argument("--exe", default="", help="binary to run instead of donor_engine.exe")
    a = ap.parse_args()

    extra = (list(a.extra) + (["--attn", a.attn] if a.attn else [])
             + (["--attnr", a.attnr] if a.attnr else []))
    reps = [one(a.weights, a.bench, a.threads, extra, a.exe or None) for _ in range(a.reps)]
    rates = [r["tok_s"] for r in reps]
    med, iqr, q1, q3 = quart(rates)
    # organ table of the rep whose rate is closest to the median
    mid = min(reps, key=lambda r: abs(r["tok_s"] - med))
    rec = {"label": a.label or os.path.basename(a.weights), "weights": a.weights,
           "bench": a.bench, "threads": a.threads, "reps": a.reps, "attn": a.attn or "serial", "attnr": a.attnr or "none", "exe": a.exe or "donor_engine.exe",
           "tok_s": rates, "median_tok_s": med, "iqr_tok_s": iqr, "q1": q1, "q3": q3,
           "median_rep_organs_ms": mid["organs"],
           "median_rep_wall_ms_token": mid["wall_ms_token"],
           "median_rep_organs_total_ms": mid["organs_total_ms"]}
    print("%-22s bench=%-4d median %8.3f tok/s  IQR %7.3f   reps %s"
          % (rec["label"], a.bench, med, iqr, " ".join("%.2f" % x for x in rates)), flush=True)
    if mid["organs"]:
        print("   organs ms/tok: " + "  ".join("%s=%.3f" % (k, mid["organs"][k])
                                               for k in ORGANS if k in mid["organs"]), flush=True)
    if a.out:
        prev = []
        if os.path.exists(a.out):
            prev = json.load(open(a.out, encoding="utf-8"))
        prev.append(rec)
        json.dump(prev, open(a.out, "w", encoding="utf-8"), indent=1)
    return rec


if __name__ == "__main__":
    main()
