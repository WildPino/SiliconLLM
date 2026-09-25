#!/usr/bin/env python3
"""Apply the frozen NES-03 numerical, quality and CPU timing gates."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from score_nes01_greedy import max_ngram_count, read_generations, sha256
from summarize_nes01_timing import summarize


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "native_expert_scaling"
DOCS = ROOT / "docs" / "research" / "NATIVE_EXPERT_SCALING_20260925"
AUDIT = re.compile(r"router audit layer (\d+): positions=(\d+) missed_exact_top8=(\d+) changed_route_positions=(\d+)")
BPB = re.compile(r"==== BPB \((\d+) tok\): ([\d.]+) ====")
TOP1 = re.compile(r"agreement=([\d.]+)% \((\d+)/(\d+)\)")


def quality_log(name: str, pattern: re.Pattern[str]) -> tuple[list[str], str]:
    path = RUNS / name
    text = path.read_text(encoding="utf-8-sig")
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one quality measurement in {path}, found {len(matches)}")
    return list(matches[0]), sha256(path)


def audit_log(name: str) -> dict:
    path = RUNS / name
    rows = [tuple(map(int, row)) for row in AUDIT.findall(path.read_text(encoding="utf-8-sig"))]
    if len(rows) != 6 or [row[0] for row in rows] != list(range(6)) or any(row[1] != 8192 for row in rows):
        raise ValueError(f"invalid six-layer frozen audit: {path}")
    return {"raw_log_sha256": sha256(path), "layers": [
        {"layer": l, "positions": n, "missed_exact_top8": miss, "changed_route_positions": changed}
        for l, n, miss, changed in rows]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    probe_path = RUNS / "nes03_frozen16" / "probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe["schema"] != "nes03_int8_router_probe_v1" or probe["chosen_shortlist"] != 32:
        raise ValueError("unexpected frozen numerical result")
    baseline_path = DOCS / "nes02_capacity_stress_baseline_20260925.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["schema"] != "nes02_capacity_stress_baseline_v1":
        raise ValueError("unexpected NES-02 baseline")
    timings = {
        "int8_e128_serial": summarize(RUNS / "nes03_int8_e128_serial_timing_6threads.log", 37_748_736),
        "int8_e1280_serial": summarize(RUNS / "nes03_int8_e1280_serial_timing_6threads.log", 377_487_360),
        "int8_e128_parallel": summarize(RUNS / "nes03_int8_e128_parallel_timing_6threads.log", 37_748_736),
        "int8_e1280_parallel": summarize(RUNS / "nes03_int8_e1280_parallel_timing_6threads.log", 377_487_360),
        "matched_exact_e128": summarize(RUNS / "nes03_matched_e128_exact_timing_6threads.log", 37_748_736),
        "matched_exact_e1280": summarize(RUNS / "nes03_matched_e1280_exact_timing_6threads.log", 377_487_360),
    }
    exact_bpb_row, exact_bpb_hash = quality_log("nes03_quality_exact_bpb.log", BPB)
    int8_bpb_row, int8_bpb_hash = quality_log("nes03_quality_int8_bpb.log", BPB)
    top1_row, top1_hash = quality_log("nes03_quality_top1.log", TOP1)
    exact_n, exact_bpb = int(exact_bpb_row[0]), float(exact_bpb_row[1])
    int8_n, int8_bpb = int(int8_bpb_row[0]), float(int8_bpb_row[1])
    top1_count, top1_total = int(top1_row[1]), int(top1_row[2])
    if exact_n != 20480 or int8_n != 20480 or top1_total != 20480:
        raise ValueError("quality token count differs from frozen 20,480")
    greedy_exact = RUNS / "nes01_e128_fp32_greedy.u16"
    greedy_int8 = RUNS / "nes03_e128_int8_fp32_greedy.u16"
    exact_rows = read_generations(greedy_exact, 1024)
    int8_rows = read_generations(greedy_int8, 1024)
    if [offset for offset, _ in exact_rows] != [offset for offset, _ in int8_rows]:
        raise ValueError("greedy prefix mismatch")
    exact_loops = sum(max_ngram_count(tokens) >= 3 for _, tokens in exact_rows)
    int8_loops = sum(max_ngram_count(tokens) >= 3 for _, tokens in int8_rows)
    audits = {
        "e128": audit_log("nes03_c_e128_route_audit.log"),
        "e1280": audit_log("nes03_c_e1280_route_audit.log"),
    }
    old128 = baseline["arms"]["e128"]["median_of_last_three"]
    old1280 = baseline["arms"]["e1280"]["median_of_last_three"]
    configuration_gates = {}
    for mode in ("serial", "parallel"):
        med128 = timings[f"int8_e128_{mode}"]["median_of_last_three"]
        med1280 = timings[f"int8_e1280_{mode}"]["median_of_last_three"]
        configuration_gates[mode] = {
            "e128_total_ratio_to_nes02_baseline": med128["total_us_per_token"] / old128["total_us_per_token"],
            "e1280_router_ratio_to_nes02_baseline": med1280["router_selection_us_per_token"] / old1280["router_selection_us_per_token"],
            "e1280_total_ratio_to_nes02_baseline": med1280["total_us_per_token"] / old1280["total_us_per_token"],
            "e128_no_more_than_5pct_regression": med128["total_us_per_token"] <= 1.05 * old128["total_us_per_token"],
            "e1280_router_at_most_half": med1280["router_selection_us_per_token"] <= 0.5 * old1280["router_selection_us_per_token"],
            "e1280_total_at_most_0_85": med1280["total_us_per_token"] <= 0.85 * old1280["total_us_per_token"],
        }
    numerical_pass = probe["advance_to_native"] and all(
        all(layer["missed_exact_top8"] == 0 for layer in arm["layers"])
        for arm in audits.values())
    quality_pass = (int8_bpb - exact_bpb <= .001 and top1_count / top1_total >= .99 and
                    int8_loops <= exact_loops + 1)
    for values in configuration_gates.values():
        values["all_frozen_cost_gates"] = all(values[k] for k in (
            "e128_no_more_than_5pct_regression", "e1280_router_at_most_half", "e1280_total_at_most_0_85"))
    result = {
        "schema": "nes03_int8_router_result_v1",
        "scope": "router fidelity and CPU cost at trained E128 and synthetic E1280; no donor transfer or large-E quality claim",
        "frozen_numerical_probe_sha256": sha256(probe_path),
        "nes02_baseline_sha256": sha256(baseline_path),
        "engine_source_sha256": sha256(ROOT / "benchmarks" / "phase60" / "engine.c"),
        "engine_binary_sha256": sha256(RUNS / "engine_nes03_int8.exe"),
        "router_int8_sketch_bytes": {"e128": 202_752, "e1280": 2_027_520},
        "numerical_pass": numerical_pass,
        "c_route_audits": audits,
        "quality": {
            "exact_bpb": exact_bpb, "int8_bpb": int8_bpb, "delta_bpb": int8_bpb - exact_bpb,
            "exact_bpb_log_sha256": exact_bpb_hash, "int8_bpb_log_sha256": int8_bpb_hash,
            "top1_agreement": top1_count / top1_total, "top1_count": top1_count,
            "top1_total": top1_total, "top1_log_sha256": top1_hash,
            "exact_greedy_sha256": sha256(greedy_exact), "int8_greedy_sha256": sha256(greedy_int8),
            "exact_triple_8gram_continuations": exact_loops,
            "int8_triple_8gram_continuations": int8_loops,
            "greedy_byte_identical": greedy_exact.read_bytes() == greedy_int8.read_bytes(),
            "nonregression_gate": quality_pass,
        },
        "timings": timings,
        "configuration_gates": configuration_gates,
        "advance_parallel_cost_candidate": bool(numerical_pass and quality_pass and
                                                configuration_gates["parallel"]["all_frozen_cost_gates"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("numerical", numerical_pass, "quality", quality_pass,
          "serial", configuration_gates["serial"], "parallel", configuration_gates["parallel"])


if __name__ == "__main__":
    main()
