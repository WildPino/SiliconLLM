#!/usr/bin/env python3
"""Validate and summarize the four METH-40 CPU router-scan arms."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
SEED = ROOT / "results/native_expert_scaling/meth40_e128_router_seed.bin"
SEED_SHA = "e0bca061fb9f42b17b22ddf10d17f4da37599e9995cbe698ee19d380aa489262"
BINARY = ROOT / "results/native_expert_scaling/meth40_rank64_router_scan.exe"
BINARY_SHA = "53710db22afb013355ee9b7befde83de65647f3a0afa9cbe1ea09c6cbb1c5515"
SOURCE = ROOT / "benchmarks/native_expert_scaling/meth40_rank64_router_scan.c"
SOURCE_SHA = "3b1ac5c8119239e44938c73110dd1dc56f96352ddf1d022f54a08a9c3e00d2ef"
SHAPE = {"layers": 24, "dimension": 896, "rank": 64, "candidates": 96,
         "warmup_tokens": 8, "repetitions": 4}
EXPERTS = (27355, 273547)
THREADS = (1, 6)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_repository_text(path):
    """Use Git's LF form so Windows CRLF checkouts verify identically."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def close(a, b, tolerance=1e-6):
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert sha(SEED) == SEED_SHA
    assert sha(BINARY) == BINARY_SHA
    assert sha_repository_text(SOURCE) == SOURCE_SHA
    arms = {}
    raw_hashes = {}
    for experts in EXPERTS:
        for threads in THREADS:
            path = DOCS / f"meth40_e{experts}_t{threads}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["experiment"] == "METH-40"
            assert data["experts"] == experts and data["threads"] == threads
            assert all(data[k] == v for k, v in SHAPE.items())
            assert data["measured_tokens_per_rep"] == (64 if experts == 27355 else 16)
            codes = 24 * experts * 64
            scales = 24 * experts * 4
            basis = 24 * 896 * 64 * 4
            assert data["code_bytes_per_token"] == codes
            assert data["scale_bytes_per_token"] == scales
            assert data["basis_bytes_per_token"] == basis
            assert data["pool_bytes"] == codes + scales + basis
            assert 0 <= data["selftest_max_abs_error"] <= 0.001
            assert 0 < data["process_seconds"] < 15 * 60
            for key, median_key in (
                ("repetitions_ms_per_token", "median_total_ms_per_token"),
                ("projection_ms_per_token", "median_projection_ms_per_token"),
                ("scan_ms_per_token", "median_scan_ms_per_token"),
                ("merge_ms_per_token", "median_merge_ms_per_token"),
            ):
                assert len(data[key]) == 4 and all(x > 0 for x in data[key])
                assert close(statistics.median(data[key]), data[median_key])
            for total, proj, scan, merge in zip(
                data["repetitions_ms_per_token"],
                data["projection_ms_per_token"],
                data["scan_ms_per_token"],
                data["merge_ms_per_token"],
            ):
                assert close(total, proj + scan + merge)
            assert len(data["checksum"]) == 16
            arms[f"e{experts}_t{threads}"] = data
            raw_hashes[path.name] = sha_repository_text(path)
        assert arms[f"e{experts}_t1"]["checksum"] == arms[f"e{experts}_t6"]["checksum"]
    fast_small = min(arms[f"e{EXPERTS[0]}_t{t}"]["median_total_ms_per_token"]
                     for t in THREADS)
    fast_large = min(arms[f"e{EXPERTS[1]}_t{t}"]["median_total_ms_per_token"]
                     for t in THREADS)
    summary = {"experiment": "METH-40-summary",
               "source_sha256": SOURCE_SHA, "binary_sha256": BINARY_SHA,
               "seed_sha256": SEED_SHA, "raw_result_sha256": raw_hashes,
               "host": {"cpu": "AMD Ryzen 5 3600X 6-Core Processor",
                        "physical_ram_bytes": 85845118976,
                        "os": "Windows"},
               "arms": arms,
               "fastest_small_ms_per_token": fast_small,
               "fastest_large_ms_per_token": fast_large,
               "fastest_10x_latency_ratio": fast_large / fast_small,
               "large_remaining_within_20ms": 20.0 - fast_large,
               "large_code_only_40GBps_ms":
               arms["e273547_t6"]["code_bytes_per_token"] / 40e9 * 1000,
               "hard_component_under_20ms": fast_large <= 20.0,
               "decision": "router_scan_under_20ms_but_no_joint_speed_or_quality_claim"
               if fast_large <= 20.0 else "router_scan_alone_exceeds_20ms"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "fastest_small_ms_per_token", "fastest_large_ms_per_token",
        "fastest_10x_latency_ratio", "large_remaining_within_20ms",
        "large_code_only_40GBps_ms", "hard_component_under_20ms", "decision")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
