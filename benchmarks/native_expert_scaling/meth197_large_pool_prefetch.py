#!/usr/bin/env python3
"""Same-binary selected LUT baseline/prefetch scaling at 10x expert count."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import psutil

import meth177_large_lut_pool as M177


ROOT = M177.ROOT
SOURCE = M177.SOURCE
EXPERTS = (12800, 128000)
THREADS = (1, 6)
MODES = ("baseline", "prefetch")
MAX_DISK = 1_000_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.binary.exists() and not args.out.exists()
    args.binary.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build = ["clang", "-O3", "-mavx2", "-mfma", "-march=znver2", "-fopenmp",
             str(SOURCE), "-o", str(args.binary), "-lm"]
    subprocess.run(build, cwd=ROOT, check=True, timeout=120)
    selftest = subprocess.run([str(args.binary), "--kselftest"], cwd=ROOT,
                              check=True, text=True, capture_output=True, timeout=120)
    assert "73024 checks | worst |S - S_ref| = 0  PASS" in selftest.stdout
    assert "exp256_ps vs libm" in selftest.stdout and selftest.stdout.count("PASS") == 2
    result = {"experiment": "METH-197-large-pool-prefetch",
              "protocol": "METH_197_LARGE_POOL_PREFETCH_PROTOCOL_20260930.md",
              "source_sha256": M177.digest(SOURCE),
              "binary_sha256": M177.digest(args.binary),
              "build_command": build, "selftest": selftest.stdout.strip(),
              "host": {"processor": platform.processor(),
                       "physical_cpus": psutil.cpu_count(logical=False),
                       "total_ram_bytes": psutil.virtual_memory().total},
              "arms": {}, "decision": "incomplete"}
    try:
        for experts in EXPERTS:
            result["arms"][str(experts)] = {}
            for threads in THREADS:
                result["arms"][str(experts)][str(threads)] = {}
                for mode in MODES:
                    available = psutil.virtual_memory().available
                    if experts == 128000 and available < M177.MIN_AVAILABLE:
                        result["decision"] = "capacity_stop"
                        result["stop"] = {"experts": experts, "threads": threads,
                                          "mode": mode,
                                          "available_ram_bytes": available,
                                          "required_available_bytes": M177.MIN_AVAILABLE}
                        return
                    command = [str(args.binary), "--rank8-lut-pool", str(experts),
                               "--threads", str(threads)]
                    if mode == "prefetch":
                        command.append("--rank8-prefetch")
                    start = time.monotonic()
                    process = subprocess.run(command, cwd=ROOT, text=True,
                                             capture_output=True, timeout=900)
                    log = args.out.with_name(args.out.stem +
                                             f".e{experts}.t{threads}.{mode}.log")
                    assert not log.exists()
                    log.write_text(process.stdout + process.stderr,
                                   encoding="utf-8")
                    assert process.returncode == 0, (command, process.returncode)
                    arm = M177.parse_arm(process.stdout, experts, threads)
                    arm.update({"command": command,
                                "elapsed_seconds": time.monotonic() - start,
                                "available_ram_before_bytes": available,
                                "log_sha256": M177.digest(log),
                                "log_path": str(log.resolve())})
                    result["arms"][str(experts)][str(threads)][mode] = arm
                    print(json.dumps({"completed": [experts, threads, mode],
                                      "median_us_token":
                                          arm["median_last_three_us_token"],
                                      "available_ram_before_bytes": available}),
                          flush=True)
        for experts in EXPERTS:
            for threads in THREADS:
                arms = result["arms"][str(experts)][str(threads)]
                a = [row["checksum"] for row in arms["baseline"]["repetitions"]]
                b = [row["checksum"] for row in arms["prefetch"]["repetitions"]]
                assert a == b, (experts, threads, a, b)
            for mode in MODES:
                t1 = result["arms"][str(experts)]["1"][mode]
                t6 = result["arms"][str(experts)]["6"][mode]
                assert ([row["checksum"] for row in t1["repetitions"]] ==
                        [row["checksum"] for row in t6["repetitions"]])
        baseline_small = result["arms"]["12800"]["6"]["baseline"][
            "median_last_three_us_token"]
        baseline_large = result["arms"]["128000"]["6"]["baseline"][
            "median_last_three_us_token"]
        prefetch_small = result["arms"]["12800"]["6"]["prefetch"][
            "median_last_three_us_token"]
        prefetch_large = result["arms"]["128000"]["6"]["prefetch"][
            "median_last_three_us_token"]
        result["six_thread_medians_us_token"] = {
            "baseline_small": baseline_small, "baseline_large": baseline_large,
            "prefetch_small": prefetch_small, "prefetch_large": prefetch_large}
        result["six_thread_ratios"] = {
            "baseline_large_over_small": baseline_large / baseline_small,
            "prefetch_large_over_small": prefetch_large / prefetch_small,
            "small_prefetch_over_baseline": prefetch_small / baseline_small,
            "large_prefetch_over_baseline": prefetch_large / baseline_large}
        result["gates"] = {
            "prefetch_scaling_ratio_at_most_1_25":
                prefetch_large / prefetch_small <= 1.25,
            "prefetch_large_us_at_most_5000": prefetch_large <= 5000,
            "small_prefetch_regression_at_most_10pct":
                prefetch_small / baseline_small <= 1.10,
            "all_checksums_equal": True}
        result["decision"] = "prefetch_selected_lut_scaling_pass" if all(
            result["gates"].values()) else "prefetch_selected_lut_scaling_fail"
    except BaseException as error:
        result["decision"] = "apparatus_or_resource_failure"
        result["error"] = repr(error)
        raise
    finally:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size + sum(
            Path(arm["log_path"]).stat().st_size
            for e in result["arms"].values() for t in e.values()
            for arm in t.values()) < MAX_DISK
        print(json.dumps({"decision": result["decision"],
                          "result_sha256": M177.digest(args.out)}), flush=True)


if __name__ == "__main__":
    main()
