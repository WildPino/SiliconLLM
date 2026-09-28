#!/usr/bin/env python3
"""Build, parity-check and time the conditional METH-174 CPU route."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess

import numpy as np
import psutil

import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
ART = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling"
SOURCE = Path(__file__).with_name("meth174_native_low_recurrence_cost.c")
EXE = ART / "meth174_native_low_recurrence_cost.exe"
ROUTE = DOC / "meth173_fresh_recurrence_route_result.json"
TABLE = DOC / "meth172_threshold8_shared_table.json"
BASE = DOC / "meth162_training_derived_shared_table.json"
FIXTURE_REPORT = DOC / "meth174_native_low_recurrence_fixtures.json"
BANK = (ART / "meth126_shared_a_factor_bank.bin",
        "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1")
VECTORS = (ART / "meth125_e1280_vectors.bin",
           "f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699")
BASE_BINARY = (ART / "meth166_shared_table.bin",
               "e7a270e0e77fb2c7bca711a0932c24224dc7f6e060611cc3a942ed7593cf38d8")
BASE_HIT = (ART / "meth151_context_hit.bin",
            "c211bde3459b3a1df27250ee0e785cde412cba07fb763aeb519281588f54a629")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def parse(line):
    _, *parts = line.split()
    return dict(part.split("=", 1) for part in parts)


def first_context(path):
    data = path.read_bytes()
    assert data[:8] == b"M151TX01"
    assert struct.unpack_from("<I", data, 8)[0] == 256
    return struct.unpack_from("<III", data, 12)


def golden(raw, fixture_paths):
    old_json = json.loads(BASE.read_text(encoding="utf-8"))
    new_json = json.loads(TABLE.read_text(encoding="utf-8"))
    old = {(r["token"], r["previous"], r["position"])
           for r in old_json["tuples"]}
    new = {(r["token"], r["previous"], r["position"])
           for r in new_json["tuples"]}
    assert len(old) == 138 and len(new) == 1685 and old <= new
    found = []
    for line in raw.splitlines():
        if not line.startswith("M174_GOLDEN "):
            continue
        item = parse(line)
        name = item["cell"]
        token, previous, position = first_context(fixture_paths[name])
        assert (int(item["token"]), int(item["previous"]),
                int(item["position"])) == (token, previous, position)
        ids = np.asarray([token], dtype=np.int64)
        prev = np.asarray([previous], dtype=np.int64)
        pos = np.asarray([position], dtype=np.int64)
        child = np.asarray([[89, 89, 89, 89]], dtype=np.int64)
        old_grand, _ = R150.route(ids, prev, pos, child, 3, old)
        new_grand, _ = R150.route(ids, prev, pos, child, 3, new)
        if token in (151644, 151645):
            old_grand[0, :] = 890
            new_grand[0, :] = 890
        assert int(item["old_grand"]) == int(old_grand[0, 0])
        assert int(item["new_grand"]) == int(new_grand[0, 0])
        found.append({"cell": name, "context": [token, previous, position],
                      "baseline_grandchild": int(old_grand[0, 0]),
                      "candidate_grandchild": int(new_grand[0, 0])})
    assert [r["cell"] for r in found] == [
        "base_hit", "candidate_hit", "delimiter", "miss"]
    return found


def ensure_no_teacher_or_route_running():
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(proc.info["cmdline"] or []).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if ("meth173_fresh_recurrence_teacher.py" in command or
                "meth173_fresh_recurrence_route_screen.py" in command):
            raise RuntimeError(f"METH-173 GPU process still active: {proc.pid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-result-sha", required=True)
    ap.add_argument("--fixture-report-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    ensure_no_teacher_or_route_running()
    assert digest(ROUTE) == args.route_result_sha
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    assert route["decision"] == "prospective_low_recurrence_route_pass_native_cpu_pending"
    assert digest(FIXTURE_REPORT) == args.fixture_report_sha
    report = json.loads(FIXTURE_REPORT.read_text(encoding="utf-8"))
    assert report["route_result_sha256"] == args.route_result_sha
    inputs = {"bank": BANK, "vectors": VECTORS,
              "base_table": BASE_BINARY,
              "candidate_table": (Path(report["candidate_table_binary"]["path"]),
                                  report["candidate_table_binary"]["sha256"]),
              "base_hit": BASE_HIT,
              **{name: (Path(report[name]["path"]), report[name]["sha256"])
                 for name in ("new_hit", "delimiter", "miss")}}
    for name, (path, sha) in inputs.items():
        assert digest(path) == sha, name
    build = ["clang", "-O3", "-mavx2", "-mfma", "-std=c11", "-Wall", "-Wextra",
             str(SOURCE), "-o", str(EXE), "-lm", "-lpsapi"]
    subprocess.run(build, cwd=ROOT, check=True)
    order = ("bank", "vectors", "base_table", "candidate_table",
             "base_hit", "new_hit", "delimiter", "miss")
    command = [str(EXE), *[str(inputs[k][0]) for k in order]]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log = args.out.with_suffix(".log")
    assert not log.exists()
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    fixture_paths = {"base_hit": inputs["base_hit"][0],
                     "candidate_hit": inputs["new_hit"][0],
                     "delimiter": inputs["delimiter"][0],
                     "miss": inputs["miss"][0]}
    goldens = golden(process.stdout, fixture_paths)
    reps = [parse(line) for line in process.stdout.splitlines()
            if line.startswith("M174_REP ")]
    cells = [parse(line) for line in process.stdout.splitlines()
             if line.startswith("M174_CELL ")]
    summaries = [parse(line) for line in process.stdout.splitlines()
                 if line.startswith("M174_SUMMARY ")]
    assert len(reps) == 20 and len(cells) == 4 and len(summaries) == 1
    assert [cell["cell"] for cell in cells] == list(fixture_paths)
    assert all(float(cell["new_median_ms"]) <= 3.5 or cell["gate"] == "FAIL"
               for cell in cells)
    assert all(float(cell["ratio"]) <= 1.25 or cell["gate"] == "FAIL"
               for cell in cells)
    passed = all(cell["gate"] == "PASS" for cell in cells)
    assert (process.returncode == 0) == (summaries[0]["gate"] == "PASS") == passed
    result = {"experiment": "METH-174-native-low-recurrence-route-cost",
              "protocol": "METH_174_NATIVE_LOW_RECURRENCE_ROUTE_PROTOCOL_20260928.md",
              "route_result_sha256": args.route_result_sha,
              "fixture_report_sha256": args.fixture_report_sha,
              "source_sha256": digest(SOURCE), "executable_sha256": digest(EXE),
              "input_sha256": {name: sha for name, (_, sha) in inputs.items()},
              "build_command": build, "command": command,
              "cpu": platform.processor(), "goldens": goldens,
              "repetitions": reps, "cells": cells,
              "summary": summaries[0], "log_sha256": digest(log),
              "decision": "native_low_recurrence_route_cost_pass_quality_and_full_rate_open"
                          if passed else "native_low_recurrence_route_cost_fail"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "result_sha256": digest(args.out),
                      "cells": cells, "summary": summaries[0]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
