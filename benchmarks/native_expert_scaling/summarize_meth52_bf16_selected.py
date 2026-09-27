#!/usr/bin/env python3
"""Validate the six METH-52 CPU arms and compute the frozen decision."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
SEED_REPORT = DOCS / "meth52_bf16_seed_export.json"
SOURCE = ROOT / "benchmarks/native_expert_scaling/meth52_bf16_selected_cpu.c"
EXPECTED_SEED_SOURCE = "53f2849da55f0da7f91f7d097ab3357308abfd85174cdf8a82b7e226cd738473"
LAYERS, DIM, RANK, TOPK = 24, 896, 8, 4
BYTES_PER_EXPERT_LAYER = 2 * RANK * DIM * 2
SELECTED_BYTES = LAYERS * TOPK * BYTES_PER_EXPERT_LAYER


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    return abs(actual - expected) <= 1e-6 * max(1.0, abs(expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(SEED_REPORT.read_text(encoding="utf-8"))
    assert report["source_sha256"] == EXPECTED_SEED_SOURCE
    assert sha(args.seed) == report["raw_seed_sha256"]
    assert args.seed.stat().st_size == report["raw_seed_bytes"]
    arms = {}
    raw_hashes = {}
    for experts in (128, 2735, 27355):
        arms[str(experts)] = {}
        for threads in (1, 6):
            path = DOCS / f"meth52_e{experts}_t{threads}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["experiment"] == "METH-52"
            assert (data["experts"], data["threads"], data["layers"],
                    data["dimension"], data["rank"], data["selected_experts"]) == (
                        experts, threads, LAYERS, DIM, RANK, TOPK)
            assert (data["warmup_tokens"], data["measured_tokens_per_rep"],
                    data["repetitions"]) == (64, 256, 4)
            assert data["pool_bytes"] == experts * LAYERS * BYTES_PER_EXPERT_LAYER
            assert data["selected_factor_bytes_per_token"] == SELECTED_BYTES
            assert data["pool_bytes"] <= data["rss_end_bytes"] <= 32 * (1 << 30)
            assert data["selftest_max_abs_error"] < 0.001
            assert data["process_seconds"] < 15 * 60
            assert len(data["repetition_checksums"]) == 4
            for name in ("repetitions", "dispatch", "projection", "output"):
                key = ("repetitions_ms_per_token" if name == "repetitions"
                       else f"{name}_ms_per_token")
                values = data[key]
                assert len(values) == 4 and all(0 <= value < 20 for value in values)
                median_key = ("median_total_ms_per_token" if name == "repetitions"
                              else f"median_{name}_ms_per_token")
                assert close(data[median_key], median(values))
            for total, dispatch, projection, output in zip(
                    data["repetitions_ms_per_token"], data["dispatch_ms_per_token"],
                    data["projection_ms_per_token"], data["output_ms_per_token"]):
                assert close(total, dispatch + projection + output)
            arms[str(experts)][str(threads)] = data
            raw_hashes[path.name] = sha(path)
        assert (arms[str(experts)]["1"]["repetition_checksums"] ==
                arms[str(experts)]["6"]["repetition_checksums"])
    fastest_thread = min((1, 6), key=lambda t: arms["27355"][str(t)]["median_total_ms_per_token"])
    large = arms["27355"][str(fastest_thread)]["median_total_ms_per_token"]
    small = arms["2735"][str(fastest_thread)]["median_total_ms_per_token"]
    ratio = large / small
    gates = {"large_selected_path_under_3ms": large <= 3.0,
             "tenfold_pool_ratio_under_1p5": ratio <= 1.5}
    gates["joint"] = all(gates.values())
    summary = {
        "experiment": "METH-52-bf16-selected-CPU-summary",
        "source_artifact_sha256": EXPECTED_SEED_SOURCE,
        "seed_report_sha256": sha(SEED_REPORT),
        "seed_sha256": sha(args.seed),
        "source_c_sha256": sha(SOURCE),
        "binary_sha256": sha(args.binary),
        "raw_arm_sha256": raw_hashes,
        "shape": {"layers": LAYERS, "dimension": DIM, "rank": RANK,
                  "selected_experts_per_layer": TOPK},
        "selected_factor_bytes_per_token": SELECTED_BYTES,
        "projected_E273547_factor_bytes": 273547 * LAYERS * BYTES_PER_EXPERT_LAYER,
        "arms": {experts: {threads: {
            "pool_bytes": data["pool_bytes"],
            "rss_end_bytes": data["rss_end_bytes"],
            "median_total_ms_per_token": data["median_total_ms_per_token"],
            "median_dispatch_ms_per_token": data["median_dispatch_ms_per_token"],
            "median_projection_ms_per_token": data["median_projection_ms_per_token"],
            "median_output_ms_per_token": data["median_output_ms_per_token"],
            "selected_addressed_gbps": SELECTED_BYTES /
                (data["median_total_ms_per_token"] / 1000) / 1e9,
            "process_seconds": data["process_seconds"],
        } for threads, data in group.items()} for experts, group in arms.items()},
        "decision": {"fastest_large_threads": fastest_thread,
                     "large_ms_per_token": large,
                     "tenfold_ratio_same_threads": ratio,
                     "remaining_within_20ms": 20.0 - large,
                     "gates": gates,
                     "status": "component_candidate" if gates["joint"]
                               else "reject_bf16_selected_component"},
        "scope": "E>128 repeats trained E128 rows; cost only, no router/core/quality/accepted-token rate",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": summary["arms"], "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
