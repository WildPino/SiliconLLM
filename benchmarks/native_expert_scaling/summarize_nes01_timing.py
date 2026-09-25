#!/usr/bin/env python3
"""Summarize the frozen NES-01 E32/E128 CPU timing repetitions."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from score_nes01_greedy import sha256


TIMING = re.compile(r"==== timing \(MoE, mlp=lut skip=1 exp=fast, 3000 tok\): ([\d.]+) tok/s \| ([\d.]+) us/tok")
BREAKDOWN = re.compile(r"MoE router\+selection ([\d.]+)  selected-expert path ([\d.]+)")
CODE_BYTES = re.compile(r"ternary weight-code bytes .* = (\d+) B")


def summarize(path: Path, expected_code_bytes: int) -> dict:
    content = path.read_text(encoding="utf-8")
    times = [(float(rate), float(latency)) for rate, latency in TIMING.findall(content)]
    components = [(float(router), float(expert)) for router, expert in BREAKDOWN.findall(content)]
    byte_counts = [int(count) for count in CODE_BYTES.findall(content)]
    if len(times) != 4 or len(components) != 4 or byte_counts != [expected_code_bytes] * 4:
        raise ValueError(f"unexpected repeat count, mode or code footprint: {path}")
    runs = [{"tokens_per_second": rate, "total_us_per_token": latency,
             "router_selection_us_per_token": router,
             "selected_expert_us_per_token": expert}
            for (rate, latency), (router, expert) in zip(times, components)]
    warm = runs[1:]
    return {"raw_log_sha256": sha256(path), "ternary_code_bytes": expected_code_bytes,
            "runs": runs, "first_run_excluded_as_warmup": True,
            "median_of_last_three": {key: statistics.median(run[key] for run in warm)
                                     for key in runs[0]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e32", type=Path, required=True)
    parser.add_argument("--e128", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    arms = {"e32": summarize(args.e32, 9_437_184),
            "e128": summarize(args.e128, 37_748_736)}
    med32 = arms["e32"]["median_of_last_three"]
    med128 = arms["e128"]["median_of_last_three"]
    ratio = med128["total_us_per_token"] / med32["total_us_per_token"]
    result = {
        "schema": "nes01_cpu_timing_v1",
        "conditions": "Ryzen 5 3600X; clang 21.1.8 -O3 -mavx2 -mfma -march=znver2 -fopenmp; 6 threads; byte-packed code; LUT+fast; 3000 validation-slice input tokens per run; one warmup plus three timed invocations per arm",
        "scope": "cache-resident small pilot; not a 10B/100B or accepted-autoregressive-rate measurement",
        "arms": arms,
        "median_latency_ratio_e128_over_e32": ratio,
        "pilot_gate_e128_latency_at_most_1_25_e32": ratio <= 1.25,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"E32 {med32['tokens_per_second']} tok/s, E128 {med128['tokens_per_second']} tok/s, "
          f"latency ratio {ratio:.3f}, gate {'PASS' if ratio <= 1.25 else 'FAIL'}")


if __name__ == "__main__":
    main()
