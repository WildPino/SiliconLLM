#!/usr/bin/env python3
"""Measure METH-31's selected LUT kernel with a 10x larger resident pool."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import psutil

from summarize_meth31_rank8_lut_pool import META, REP


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks/phase60/engine.c"
MIN_AVAILABLE = 64 * (1 << 30)
EXPERTS = (12800, 128000)
THREADS = (1, 6)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_arm(raw, experts, threads):
    meta = None
    reps = []
    for line in raw.splitlines():
        match = META.fullmatch(line)
        if match:
            assert meta is None
            e, t, pool, selected, warm, timed, count, init = match.groups()
            assert (int(e), int(t)) == (experts, threads)
            assert int(pool) == 24 * experts * (32 * 448 + 896 * 4)
            assert int(selected) == 1720320
            assert (int(warm), int(timed), int(count)) == (128, 512, 4)
            meta = {"pool_bytes": int(pool), "selected_code_bytes_per_token": int(selected),
                    "init_seconds": float(init)}
            continue
        match = REP.fullmatch(line)
        if match:
            assert meta is not None
            e, t, rep, seconds, us, gbps, checksum = match.groups()
            assert (int(e), int(t), int(rep)) == (experts, threads, len(reps) + 1)
            seconds, us, gbps = float(seconds), float(us), float(gbps)
            assert abs(us - seconds * 1e6 / 512) < 0.002
            assert abs(gbps - 512 * 1720320 / 1e9 / seconds) < 0.002
            reps.append({"rep": int(rep), "seconds": seconds, "us_token": us,
                         "selected_gbps": gbps, "checksum": int(checksum)})
            continue
        raise AssertionError(f"Unexpected native output: {line}")
    assert meta is not None and len(reps) == 4
    return {**meta, "repetitions": reps,
            "median_last_three_us_token": statistics.median(r["us_token"] for r in reps[1:]),
            "median_last_three_selected_gbps": statistics.median(
                r["selected_gbps"] for r in reps[1:])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.binary.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.binary.parent.mkdir(parents=True, exist_ok=True)
    build = ["clang", "-O3", "-mavx2", "-mfma", "-march=znver2", "-fopenmp",
             str(SOURCE), "-o", str(args.binary), "-lm"]
    subprocess.run(build, cwd=ROOT, check=True, timeout=120)
    selftest = subprocess.run([str(args.binary), "--kselftest"], cwd=ROOT,
                              check=True, text=True, capture_output=True, timeout=120)
    assert "73024 checks | worst |S - S_ref| = 0  PASS" in selftest.stdout
    assert "exp256_ps vs libm" in selftest.stdout and selftest.stdout.count("PASS") == 2
    result = {"experiment": "METH-177-large-RAM-LUT-pool",
              "protocol": "METH_177_LARGE_RAM_LUT_POOL_PROTOCOL_20260930.md",
              "source_sha256": digest(SOURCE), "binary_sha256": digest(args.binary),
              "build_command": build, "selftest": selftest.stdout.strip(),
              "host": {"processor": platform.processor(),
                       "physical_cpus": psutil.cpu_count(logical=False),
                       "total_ram_bytes": psutil.virtual_memory().total},
              "arms": {}, "decision": "incomplete"}
    try:
        for experts in EXPERTS:
            result["arms"][str(experts)] = {}
            for threads in THREADS:
                available = psutil.virtual_memory().available
                if experts == 128000 and available < MIN_AVAILABLE:
                    result["decision"] = "capacity_stop"
                    result["stop"] = {"experts": experts, "threads": threads,
                                      "available_ram_bytes": available,
                                      "required_available_bytes": MIN_AVAILABLE}
                    return
                command = [str(args.binary), "--rank8-lut-pool", str(experts),
                           "--threads", str(threads)]
                start = time.monotonic()
                process = subprocess.run(command, cwd=ROOT, text=True,
                                         capture_output=True, timeout=900)
                path = args.out.with_name(args.out.stem + f".e{experts}.t{threads}.log")
                assert not path.exists()
                path.write_text(process.stdout + process.stderr, encoding="utf-8")
                assert process.returncode == 0, (command, process.returncode)
                arm = parse_arm(process.stdout, experts, threads)
                arm.update({"elapsed_seconds": time.monotonic() - start,
                            "available_ram_before_bytes": available,
                            "log_sha256": digest(path), "log_path": str(path.resolve())})
                result["arms"][str(experts)][str(threads)] = arm
                print(json.dumps({"completed": [experts, threads],
                                  "median_us_token": arm["median_last_three_us_token"],
                                  "available_ram_before_bytes": available}), flush=True)
        for experts in EXPERTS:
            a = result["arms"][str(experts)]
            assert [r["checksum"] for r in a["1"]["repetitions"]] == [
                r["checksum"] for r in a["6"]["repetitions"]]
        base = result["arms"]["12800"]["6"]["median_last_three_us_token"]
        large = result["arms"]["128000"]["6"]["median_last_three_us_token"]
        ratio = large / base
        result["median_10x_ratio_threads6"] = ratio
        result["gates"] = {"ratio_at_most_1_25": ratio <= 1.25,
                           "large_us_at_most_5000": large <= 5000,
                           "thread_checksums_equal": True}
        result["decision"] = "selected_lut_large_pool_pass" if all(
            result["gates"].values()) else "selected_lut_large_pool_fail"
    except BaseException as error:
        result["decision"] = "apparatus_or_resource_failure"
        result["error"] = repr(error)
        raise
    finally:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"decision": result["decision"],
                          "result_sha256": digest(args.out)}), flush=True)


if __name__ == "__main__":
    main()
