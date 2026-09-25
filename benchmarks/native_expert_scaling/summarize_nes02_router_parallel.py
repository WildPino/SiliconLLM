#!/usr/bin/env python3
"""Check frozen NES-02 parallel-router cost and numerical gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_nes01_greedy import sha256
from summarize_nes01_timing import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--e128-log", type=Path, required=True)
    parser.add_argument("--e1280-log", type=Path, required=True)
    parser.add_argument("--baseline-e32-logits", type=Path, required=True)
    parser.add_argument("--parallel-e32-logits", type=Path, required=True)
    parser.add_argument("--baseline-e128-logits", type=Path, required=True)
    parser.add_argument("--parallel-e128-logits", type=Path, required=True)
    parser.add_argument("--baseline-e1280-logits", type=Path, required=True)
    parser.add_argument("--parallel-e1280-logits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("schema") != "nes02_capacity_stress_baseline_v1":
        parser.error("unexpected baseline schema")
    e128 = summarize(args.e128_log, 37_748_736)
    e1280 = summarize(args.e1280_log, 377_487_360)
    prior128 = baseline["arms"]["e128"]["median_of_last_three"]
    prior1280 = baseline["arms"]["e1280"]["median_of_last_three"]
    next128 = e128["median_of_last_three"]
    next1280 = e1280["median_of_last_three"]
    parity = {}
    for arm in ("e32", "e128", "e1280"):
        before = getattr(args, f"baseline_{arm}_logits")
        after = getattr(args, f"parallel_{arm}_logits")
        parity[arm] = {
            "baseline_sha256": sha256(before),
            "parallel_sha256": sha256(after),
            "bit_identical": before.read_bytes() == after.read_bytes(),
        }
    ratios = {
        "e1280_router_vs_baseline": next1280["router_selection_us_per_token"] / prior1280["router_selection_us_per_token"],
        "e1280_total_vs_baseline": next1280["total_us_per_token"] / prior1280["total_us_per_token"],
        "e128_total_vs_baseline": next128["total_us_per_token"] / prior128["total_us_per_token"],
    }
    gates = {
        "bit_identical_all_three": all(row["bit_identical"] for row in parity.values()),
        "e1280_router_at_most_half": ratios["e1280_router_vs_baseline"] <= 0.5,
        "e1280_total_at_most_0_85": ratios["e1280_total_vs_baseline"] <= 0.85,
        "e128_no_more_than_5pct_regression": ratios["e128_total_vs_baseline"] <= 1.05,
    }
    result = {
        "schema": "nes02_parallel_router_result_v1",
        "scope": "conditional OpenMP across router rows at E>=512; synthetic E1280 cost only; not a quality result",
        "baseline_report_sha256": sha256(args.baseline),
        "parallel_timing": {"e128": e128, "e1280": e1280},
        "fp32_exact_64_token_parity": parity,
        "median_ratios": ratios,
        "frozen_gates": gates,
        "advance_candidate": all(gates.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("ratios", ratios, "gates", gates, "advance", result["advance_candidate"])


if __name__ == "__main__":
    main()
