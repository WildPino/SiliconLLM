"""C50-A1 verdict: applies the pre-registered gates to a c50_a1_run.py output file.

G-A1a  per (class, shape) cell: median over admissible processes of the process
       medians >= 30 GB/s.
G-A1b  L1: median over processes of the process-median layer time
       <= charged bytes / 30 GB/s + 80 us.
Reported without a gate: T20 time and the non-streaming residual.

Usage:
    python -I benchmarks/chatbot50/c50_a1_verdict.py <run.json> [--json verdict.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GATE_GBS = 30.0
LAYER_BW = 30e9
LAYER_FIXED_S = 80e-6
STOP_GBS = 25.0
PLAN_BW = 32e9          # desk BW for organs without a measured cell (router F32, KV F16)
MATVEC_CELLS = ["S1_Q4_K", "S1_Q8_0", "S1_IQ4_XS", "S2_Q4_K", "S2_Q8_0", "S2_IQ4_XS",
                "S3_Q4_K", "S3_IQ4_XS", "S3_IQ3_S", "S4_Q6_K", "S4_Q4_K", "S5_Q4_K"]
Q4K_STOP_CELLS = ["S1_Q4_K", "S2_Q4_K", "S3_Q4_K"]

# Ling-mini-2.0 P0 organ bytes of the synthetic token (Q4_K = 144 B / 256 weights)
Q4K = 144 / 256
QKV_B = 3072 * 2048 * Q4K
O_B = 2048 * 2048 * Q4K
EXPERT_B = (1024 * 2048 + 2048 * 512) * Q4K
DENSE_B = (10240 * 2048 + 2048 * 5120) * Q4K
ROUTER_B = 256 * 2048 * 4
KV_B = 2 * 2048 * 4 * 128 * 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json")
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    procs = [p for p in run.get("processes", []) if not p["void"]]
    out = {"run": args.run, "exe_sha256": run.get("exe_sha256"), "admissible_processes": len(procs),
           "void_processes": sum(p["void"] for p in run.get("processes", [])), "cells": {}}
    if run.get("verdict_input") != "COMPLETE" or len(procs) < 5:
        out["verdict"] = f"VOID ({run.get('verdict_input')})"
        print(out["verdict"])
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 1
    by = {}
    for p in procs:
        for c in p["cells"]:
            by.setdefault(c["cell"], []).append(c)
    print(f"admissible processes {len(procs)}; foreign occupancy "
          f"{min(p['foreign_pct'] for p in procs):.2f}..{max(p['foreign_pct'] for p in procs):.2f}%")
    fails = []
    for name in MATVEC_CELLS:
        meds = [c["median_gbs"] for c in by[name]]
        m = statistics.median(meds)
        ok = m >= GATE_GBS
        out["cells"][name] = {"process_median_gbs": meds, "median_gbs": m, "min": min(meds), "max": max(meds),
                              "max_over_min": max(meds) / min(meds), "class": by[name][0]["class"],
                              "shape": by[name][0]["shape"], "bytes": by[name][0]["bytes"], "G-A1a": ok}
        fails += [] if ok else [name]
        print(f"G-A1a {name:10s} {m:7.2f} GB/s  [{min(meds):.2f}..{max(meds):.2f}]  {'PASS' if ok else 'FAIL'}")
    l1 = by["L1"]
    l1_med = statistics.median(c["median_s"] for c in l1)
    l1_bytes = l1[0]["bytes"]
    l1_limit = l1_bytes / LAYER_BW + LAYER_FIXED_S
    out["L1"] = {"process_median_s": [c["median_s"] for c in l1], "median_s": l1_med, "bytes": l1_bytes,
                 "limit_s": l1_limit, "G-A1b": l1_med <= l1_limit,
                 "phases_mean_s": [c["mean_s_per_call"] for c in by.get("L1_phases", [])]}
    print(f"G-A1b L1 {l1_med * 1e6:.1f} us vs limit {l1_limit * 1e6:.1f} us "
          f"({l1_bytes / 1e6:.2f} MB)  {'PASS' if out['L1']['G-A1b'] else 'FAIL'}")
    t20 = by["T20"]
    t20_med = statistics.median(c["median_s"] for c in t20)
    bw = {k: out["cells"][k]["median_gbs"] * 1e9 for k in ("S1_Q4_K", "S2_Q4_K", "S3_Q4_K", "S5_Q4_K")}
    streamed = (20 * QKV_B / bw["S1_Q4_K"] + 20 * O_B / bw["S2_Q4_K"] + 19 * 9 * EXPERT_B / bw["S3_Q4_K"]
                + DENSE_B / bw["S5_Q4_K"] + (19 * ROUTER_B + 20 * KV_B) / PLAN_BW)
    out["T20"] = {"process_median_s": [c["median_s"] for c in t20], "median_s": t20_med,
                  "bytes": t20[0]["bytes"], "streamed_time_s_at_measured_bw": streamed,
                  "non_streaming_residual_s": t20_med - streamed,
                  "phases_mean_s": [c["mean_s_per_call"] for c in by.get("T20_phases", [])]}
    print(f"T20 {t20_med * 1e3:.2f} ms for {t20[0]['bytes'] / 1e6:.1f} MB; streamed at measured BW "
          f"{streamed * 1e3:.2f} ms; non-streaming residual {(t20_med - streamed) * 1e3:.2f} ms (desk T_FIXED 2 ms)")
    stop = [n for n in Q4K_STOP_CELLS if out["cells"][n]["median_gbs"] < STOP_GBS]
    out["G-A1a_failures"] = fails
    out["q4k_below_stop"] = stop
    out["verdict"] = ("PASS" if not fails and out["L1"]["G-A1b"] else
                      "FAIL: " + ", ".join(fails + ([] if out["L1"]["G-A1b"] else ["G-A1b"])))
    if any(n in fails for n in Q4K_STOP_CELLS):
        out["consequence"] = "Q4_K below 30 GB/s: one tuning iteration allowed; stop rule if still < 25 GB/s"
    print("VERDICT", out["verdict"])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
