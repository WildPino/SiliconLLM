#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_matrix.py -- WHY is this runtime at 12.4 GB/s when DRAM is 40-44?

Two hypotheses, crossed:

  H1  OPENMP REGION COUNT. A token opens 169 parallel regions. qkv delivers 4.1 GB/s against
      the head's 17.6 on the same kernel; the excess is 2.32 ms over 72 calls = 32 us/call.
      --fuse cuts 169 -> 97 and removes the two 128-row matvecs. Bit-identical, verified.
      If H1 is right, the drop shows up IN qkv_proj's own ms, not merely in the total.

  H2  OPENMP WAIT POLICY. 32 us is far too much for a barrier (1-5 us). If the regions are
      short enough that idle threads SLEEP between them, each region pays a wake-up.
      OMP_WAIT_POLICY=active keeps them spinning.

WHY THIS REPLACED bench_matrix.sh. The .sh ran each config's three reps back to back and
compared medians ACROSS blocks. Its first run produced within-config spreads of 3-5 tok/s on
effects of 1-5%, and put --lut at 0.955x where an earlier idle measurement had it at 1.06x --
i.e. the harness was measuring the machine, not the flags. SPEED_LEDGER.md s11 exists because a
contended timing is not a timing. Three changes:

  * ROUNDS, not blocks. Every config is measured once per round; the comparison that is read is
    the WITHIN-ROUND ratio to that round's baseline, so drift between rounds cancels the way a
    paired bootstrap cancels sequence difficulty in the BPB probes.
  * ROTATED ORDER. Round r starts at config r, so no config keeps a fixed position in the
    sequence and a periodic background task cannot land on the same arm every time.
  * --profile. Per-organ ms is what H1 actually predicts. A total-throughput number cannot say
    whether a change hit qkv or something else.

Dispersion is REPORTED, not hidden: the baseline's own round-to-round spread is printed first
and is the witness for whether anything below may be read at all.

Usage: python bench_matrix.py [--weights W] [--tokens N] [--rounds R] [--out FILE]
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
EXE = os.path.join(HERE, "donor_engine.exe")

FLAGS = {
    "packed":       [],
    "fuse":         ["--fuse"],
    "lut g32":      ["--lut", "--lut-group", "32"],
    "fuse lut g32": ["--fuse", "--lut", "--lut-group", "32"],
}
CONFIGS = ["packed", "fuse", "lut g32", "fuse lut g32"]
POLICIES = ["default", "active"]
CELLS = [(c, p) for p in POLICIES for c in CONFIGS]      # 8 cells, hashable
BASELINE = ("packed", "default")

ORGANS = ["qkv_proj", "attention", "o_proj", "ffn", "head", "norm+glue", "TOTAL"]
RE_TOK = re.compile(r"([0-9.]+) tok/s")
RE_ORG = re.compile(r"^\s+(\S+)\s+([0-9.]+)\s")


def run_one(weights, flags, policy, tokens):
    env = dict(os.environ)
    env.pop("OMP_WAIT_POLICY", None)
    if policy == "active":
        env["OMP_WAIT_POLICY"] = "active"
    cmd = [EXE, "--weights", weights, "--threads", "6", "--profile"] + flags + \
          ["--bench", str(tokens)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    m = RE_TOK.search(out)
    if not m:
        raise RuntimeError("no tok/s in output:\n" + out)
    organs = {}
    for line in out.splitlines():
        mo = RE_ORG.match(line)
        if mo and mo.group(1) in ORGANS:
            organs[mo.group(1)] = float(mo.group(2))
    return float(m.group(1)), organs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="D:/_ktmp/qwen05b_packed.bin")
    ap.add_argument("--tokens", type=int, default=800)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "bench_matrix.json"))
    a = ap.parse_args()

    if not os.path.exists(EXE):
        sys.exit("no donor_engine.exe -- build it first")

    print("bench_matrix: %d cells x %d rounds x %d tokens, rotated order, --profile"
          % (len(CELLS), a.rounds, a.tokens), flush=True)
    print("  paired within round against %s\n" % (BASELINE,), flush=True)

    rec = {c: [] for c in CELLS}                 # cell -> [(round, tok/s, organs)]
    t0 = time.time()
    for r in range(a.rounds):
        order = CELLS[r % len(CELLS):] + CELLS[:r % len(CELLS)]
        line = []
        for cell in order:
            name, pol = cell
            tps, organs = run_one(a.weights, FLAGS[name], pol, a.tokens)
            rec[cell].append((r, tps, organs))
            line.append("%s/%s %.2f" % (name, pol[0], tps))
        print("round %d  (%5.0fs)  %s" % (r, time.time() - t0, "  ".join(line)), flush=True)

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    base_by_round = {r: t for r, t, _ in rec[BASELINE]}
    bt = sorted(base_by_round.values())
    print("\n== idleness witness: the baseline's own round-to-round spread ==")
    print("  packed/default tok/s: %s" % "  ".join("%.2f" % x for x in bt))
    spread = (max(bt) - min(bt)) / med(bt) * 100.0
    print("  median %.2f   min %.2f   max %.2f   spread %.1f%% of median" % (med(bt), bt[0], bt[-1], spread))
    print("  reference must be at the SAME --bench length: attention is O(position), so a")
    print("  longer bench has a genuinely lower tok/s. At 300 tokens, idle, post-rope-hoist:")
    print("  56.41 / 55.75 / 56.14  (spread 1.2%).  Pre-hoist at 300: 50.91 / 50.54 / 50.86.")
    verdict = ("USABLE for effects larger than the spread"
               if spread < 2.0 else
               "CONTENDED -- spread swamps the 1-5% effects this matrix exists to resolve")
    print("  -> %s" % verdict)

    print("\n== tok/s, and the within-round ratio to that round's baseline ==")
    print("%-16s %-8s %8s %8s %8s   %9s %9s %9s"
          % ("config", "policy", "median", "min", "max", "ratio med", "ratio min", "ratio max"))
    table = {}
    for cell in CELLS:
        name, pol = cell
        ts = [t for _, t, _ in rec[cell]]
        ratios = [t / base_by_round[r] for r, t, _ in rec[cell] if r in base_by_round]
        table["%s|%s" % (name, pol)] = {
            "tok_s": ts, "median_tok_s": med(ts),
            "within_round_ratio": ratios, "median_ratio": med(ratios),
            "organs_ms": [o for _, _, o in rec[cell]],
        }
        print("%-16s %-8s %8.2f %8.2f %8.2f   %9.3f %9.3f %9.3f"
              % (name, pol, med(ts), min(ts), max(ts),
                 med(ratios), min(ratios), max(ratios)))

    print("\n== per-organ ms/token (median over rounds) -- what H1 actually predicts ==")
    print("%-16s %-8s %s" % ("config", "policy", " ".join("%9s" % o for o in ORGANS)))
    for cell in CELLS:
        name, pol = cell
        oms = [o for _, _, o in rec[cell]]
        print("%-16s %-8s %s" % (name, pol,
              " ".join("%9.3f" % med([o.get(k, float("nan")) for o in oms]) for k in ORGANS)))

    out = {"weights": a.weights, "tokens": a.tokens, "rounds": a.rounds,
           "baseline_cell": "packed|default",
           "idleness": {"baseline_tok_s": bt, "iqr_pct_of_median": spread,
                        "verdict": verdict,
                        "idle_reference_at_300_tokens": [56.41, 55.75, 56.14],
                        "note": ("a tok/s number is only comparable to one taken at the same "
                                 "--bench length: attention grows with position, so the bench "
                                 "average does too. 800 tokens costs ~1.7 ms/token more than "
                                 "300 on this donor, entirely in the attention bucket.")},
           "cells": table, "seconds": time.time() - t0}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s  (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
