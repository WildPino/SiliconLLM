#!/usr/bin/env python3
"""Summarize a throughput-only 10x expert-pool stress against real E128."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from score_nes01_greedy import sha256
from score_nes01_routes import read_routes, route_stats
from summarize_nes01_timing import summarize


SYNTHETIC_SHA256 = "25f99a300a26ed90948d04990ed75df9f94ab9a288ef48e6ea0c46fc0a3dcf2e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e128-log", type=Path, required=True)
    parser.add_argument("--e1280-log", type=Path, required=True)
    parser.add_argument("--e1280-routes", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    if args.synthetic.stat().st_size != 3_812_869_184 or sha256(args.synthetic) != SYNTHETIC_SHA256:
        parser.error("synthetic E1280 artifact differs from pinned throughput probe")
    e128 = summarize(args.e128_log, 37_748_736)
    e1280 = summarize(args.e1280_log, 377_487_360)
    a = e128["median_of_last_three"]
    b = e1280["median_of_last_three"]
    routes = read_routes(args.e1280_routes, 1280)
    stacked = np.stack([route for _, route in routes])
    route_coverage = []
    for layer in range(6):
        stats = route_stats(stacked[:, :, layer, :].reshape(-1, 8), 1280)
        route_coverage.append({key: stats[key] for key in ("distinct_experts", "dead_experts", "max_to_mean_load")})
    result = {
        "schema": "nes02_capacity_stress_baseline_v1",
        "scope": "throughput-only: E128 is trained; E1280 repeats experts and randomizes router; no E1280 quality or distinct-capacity inference",
        "conditions": "same dynamic-E C executable, Ryzen 5 3600X, clang 21.1.8 -O3 -mavx2 -mfma -march=znver2 -fopenmp, 6 threads, byte code, LUT+fast, 3000 validation input tokens, first of four invocations discarded",
        "synthetic_file_bytes": args.synthetic.stat().st_size,
        "synthetic_file_sha256": SYNTHETIC_SHA256,
        "active_selected_code_bytes_per_token_both_arms": 2_359_296,
        "arms": {"e128": e128, "e1280": e1280},
        "median_ratios_e1280_over_e128": {
            key: b[key] / a[key] for key in ("total_us_per_token", "router_selection_us_per_token", "selected_expert_us_per_token")},
        "synthetic_generated_route_scope": "16 validation prefixes x 128 autoregressive tokens; separate from timed teacher-forced input slice",
        "synthetic_generated_routes_sha256": sha256(args.e1280_routes),
        "synthetic_generated_route_coverage_by_layer": route_coverage,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("E128",a,"E1280",b,"ratios",result["median_ratios_e1280_over_e128"])


if __name__ == "__main__":
    main()
