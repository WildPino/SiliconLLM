#!/usr/bin/env python3
"""Repeat six-thread large/small selected LUT pools in paired order."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import psutil

import meth177_large_lut_pool as M177


ROOT = M177.ROOT
SOURCE = M177.SOURCE
PRIOR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth197_large_pool_prefetch_result.json"
PRIOR_SHA = "e9435e4ad58c4dc92420b395b1876edfbabf9e48328c34db6581b39abfec644a"
ORDER = (12800, 128000, 128000, 12800, 12800, 128000)
MAX_DISK = 1_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert args.binary.is_file() and not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert M177.digest(PRIOR) == PRIOR_SHA
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    assert M177.digest(SOURCE) == prior["source_sha256"]
    assert M177.digest(args.binary) == prior["binary_sha256"]
    selftest = subprocess.run([str(args.binary), "--kselftest"], cwd=ROOT,
                              check=True, text=True, capture_output=True, timeout=120)
    assert selftest.stdout.strip() == prior["selftest"]
    result = {"experiment": "METH-198-paired-LUT-scaling",
              "protocol": "METH_198_PAIRED_LUT_SCALING_PROTOCOL_20260930.md",
              "prior_sha256": PRIOR_SHA,
              "source_sha256": prior["source_sha256"],
              "binary_sha256": prior["binary_sha256"],
              "selftest": selftest.stdout.strip(),
              "order": ORDER, "runs": [], "decision": "incomplete"}
    try:
        for number, experts in enumerate(ORDER, 1):
            available = psutil.virtual_memory().available
            if experts == 128000 and available < M177.MIN_AVAILABLE:
                result["decision"] = "capacity_stop"
                result["stop"] = {"run": number, "experts": experts,
                                  "available_ram_bytes": available,
                                  "required_available_bytes": M177.MIN_AVAILABLE}
                return
            command = [str(args.binary), "--rank8-lut-pool", str(experts),
                       "--threads", "6"]
            start = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, text=True,
                                     capture_output=True, timeout=900)
            log = args.out.with_name(args.out.stem + f".run{number}.e{experts}.log")
            assert not log.exists()
            log.write_text(process.stdout + process.stderr, encoding="utf-8")
            assert process.returncode == 0, (command, process.returncode)
            arm = M177.parse_arm(process.stdout, experts, 6)
            arm.update({"run": number, "experts": experts, "command": command,
                        "elapsed_seconds": time.monotonic() - start,
                        "available_ram_before_bytes": available,
                        "log_sha256": M177.digest(log),
                        "log_path": str(log.resolve())})
            result["runs"].append(arm)
            print(json.dumps({"completed_run": number, "experts": experts,
                              "median_us_token": arm["median_last_three_us_token"],
                              "available_ram_before_bytes": available}), flush=True)
        for experts in (12800, 128000):
            runs = [run for run in result["runs"] if run["experts"] == experts]
            assert len(runs) == 3
            checksums = [[row["checksum"] for row in run["repetitions"]]
                         for run in runs]
            assert checksums[0] == checksums[1] == checksums[2], (experts, checksums)
        pairs = []
        for i in (0, 2, 4):
            arm_a, arm_b = result["runs"][i:i + 2]
            small, large = ((arm_a, arm_b) if arm_a["experts"] == 12800
                            else (arm_b, arm_a))
            assert small["experts"] == 12800 and large["experts"] == 128000
            pairs.append({"pair": i // 2 + 1,
                          "small_run": small["run"], "large_run": large["run"],
                          "small_us_token": small["median_last_three_us_token"],
                          "large_us_token": large["median_last_three_us_token"],
                          "large_over_small":
                              large["median_last_three_us_token"] /
                              small["median_last_three_us_token"]})
        variation = {}
        for experts in (12800, 128000):
            medians = [run["median_last_three_us_token"]
                       for run in result["runs"] if run["experts"] == experts]
            variation[str(experts)] = {"medians_us_token": medians,
                                       "max_over_min": max(medians) / min(medians)}
        repeatable = all(row["max_over_min"] <= 1.10 for row in variation.values())
        ratios = [pair["large_over_small"] for pair in pairs]
        result["pairs"] = pairs
        result["variation"] = variation
        result["median_paired_ratio"] = statistics.median(ratios)
        result["gates"] = {
            "checksum_sequences_equal": True,
            "small_repeatability_at_most_1_10": variation["12800"]["max_over_min"] <= 1.10,
            "large_repeatability_at_most_1_10": variation["128000"]["max_over_min"] <= 1.10,
            "all_paired_ratios_at_most_1_25": all(ratio <= 1.25 for ratio in ratios),
            "all_large_us_at_most_5000": all(
                pair["large_us_token"] <= 5000 for pair in pairs)}
        result["decision"] = ("inconclusive_variation" if not repeatable else
                              "selected_lut_scaling_pass" if all(result["gates"].values())
                              else "selected_lut_scaling_fail")
    except BaseException as error:
        result["decision"] = "apparatus_or_resource_failure"
        result["error"] = repr(error)
        raise
    finally:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size + sum(Path(row["log_path"]).stat().st_size
                                              for row in result["runs"]) < MAX_DISK
        print(json.dumps({"decision": result["decision"],
                          "result_sha256": M177.digest(args.out)}), flush=True)


if __name__ == "__main__":
    main()
