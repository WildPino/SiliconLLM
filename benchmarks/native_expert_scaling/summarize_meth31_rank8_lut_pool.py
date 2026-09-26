#!/usr/bin/env python3
"""Validate and summarize the fixed METH-31 native CPU LUT sweep log."""
import argparse
import hashlib
import json
import platform
import re
import statistics
from pathlib import Path

import psutil


META = re.compile(
    r"^METH31 meta E=(\d+) threads=(\d+) pool_bytes=(\d+) "
    r"selected_code_bytes_per_token=(\d+) warm=(\d+) timed=(\d+) "
    r"reps=(\d+) init_seconds=([0-9.]+)$"
)
REP = re.compile(
    r"^METH31 rep E=(\d+) threads=(\d+) rep=(\d+) "
    r"seconds=([0-9.]+) us_token=([0-9.]+) "
    r"selected_gbps=([0-9.]+) checksum=(-?\d+)$"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lines = args.raw.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].startswith("METH-31 clang ")
    meta, runs = {}, {}
    for line in lines[1:]:
        m = META.fullmatch(line)
        if m:
            e, threads, pool, addressed, warm, timed, reps, init = m.groups()
            e, threads = int(e), int(threads)
            assert (e, threads) not in meta
            assert int(pool) == 24 * e * (32 * 448 + 896 * 4)
            assert int(addressed) == 24 * 4 * (32 * 448 + 896 * 4)
            assert (int(warm), int(timed), int(reps)) == (128, 512, 4)
            meta[e, threads] = {"pool_bytes": int(pool),
                                "selected_code_bytes_per_token": int(addressed),
                                "init_seconds": float(init)}
            continue
        m = REP.fullmatch(line)
        if m:
            e, threads, rep, seconds, us, gbps, checksum = m.groups()
            key = int(e), int(threads)
            assert key in meta
            assert int(rep) == len(runs.setdefault(key, [])) + 1
            seconds, us, gbps = float(seconds), float(us), float(gbps)
            assert abs(us - seconds * 1e6 / 512) < 0.002
            assert abs(gbps - 512 * 1720320 / 1e9 / seconds) < 0.002
            runs[key].append({"rep": int(rep), "seconds": seconds,
                              "us_token": us, "selected_gbps": gbps,
                              "checksum": int(checksum)})
            continue
        assert line.startswith("METH31 overall_elapsed_seconds=")
    assert set(meta) == {(e, t) for e in (128, 1280, 12800) for t in (1, 6)}
    assert all(len(v) == 4 for v in runs.values())
    summary = {}
    for e in (128, 1280, 12800):
        assert [x["checksum"] for x in runs[e, 1]] == [
            x["checksum"] for x in runs[e, 6]]
        summary[str(e)] = {}
        for t in (1, 6):
            arm = runs[e, t]
            summary[str(e)][str(t)] = {
                **meta[e, t], "repetitions": arm,
                "median_last_three_us_token": statistics.median(
                    x["us_token"] for x in arm[1:]),
                "median_last_three_selected_gbps": statistics.median(
                    x["selected_gbps"] for x in arm[1:]),
            }
    ratio = (summary["12800"]["6"]["median_last_three_us_token"] /
             summary["1280"]["6"]["median_last_three_us_token"])
    t6 = summary["12800"]["6"]["median_last_three_us_token"]
    decision = "selected_lut_component_gate_pass" if ratio <= 1.25 and t6 <= 5000 else "selected_lut_component_gate_fail"
    result = {"experiment": "METH-31", "source_sha256": sha(args.source),
              "binary_sha256": sha(args.binary), "raw_log_sha256": sha(args.raw),
              "host": {"processor": platform.processor(),
                       "logical_cpus": psutil.cpu_count(logical=True),
                       "physical_cpus": psutil.cpu_count(logical=False),
                       "memory_total_bytes": psutil.virtual_memory().total},
              "median_10x_ratio_threads6": ratio,
              "selected_path_gate": {"max_ratio": 1.25,
                                     "max_us_token": 5000,
                                     "pass": decision.endswith("pass")},
              "decision": decision,
              "arms": summary,
              "scope": "synthetic ternary LUT selected path only; no router, donor quality, or accepted-token measurement"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ratio": ratio, "e12800_threads6_us": t6,
                      "decision": decision,
                      "raw_sha256": result["raw_log_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
