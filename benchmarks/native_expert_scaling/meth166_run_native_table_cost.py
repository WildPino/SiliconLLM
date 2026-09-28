#!/usr/bin/env python3
"""Build, SHA-bind and verify paired native CPU lookup timings."""

import argparse
from datetime import datetime
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
SOURCE = Path(__file__).with_name("meth166_native_table_cost.c")
EXE = ART / "meth166_native_table_cost.exe"
FIXTURE_REPORT = DOC / "meth166_native_fixtures.json"
FIXTURE_REPORT_SHA = "9d9d0311b8fc33f9e458516c71ec6dfe3736612db32011cdca1c401e9fe10145"
INPUTS = {
    "exact_bank": (ART / "meth126_shared_a_factor_bank.bin",
                   "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1"),
    "vectors": (ART / "meth125_e1280_vectors.bin",
                "f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699"),
    "old_table": (ART / "meth151_shared_table.bin",
                  "926362655bcc71bf7bd1ceee944e56ba8f35ef124c8f60b6e0052d6525ddfc27"),
    "new_table": (ART / "meth166_shared_table.bin",
                  "e7a270e0e77fb2c7bca711a0932c24224dc7f6e060611cc3a942ed7593cf38d8"),
    "old_hit": (ART / "meth151_context_hit.bin",
                "c211bde3459b3a1df27250ee0e785cde412cba07fb763aeb519281588f54a629"),
    "new_hit": (ART / "meth166_context_new_hit.bin",
                "bc8658b854f9093a7795b186aee67b2e10138a7f041c077e98a5e28a3af94941"),
    "miss": (ART / "meth151_context_miss.bin",
             "ecce0701d71201f9a48495adbc9814632c9f5b0929675a30daf2ffc92ac8682e"),
}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def active_process(path):
    if not path.exists():
        return False
    meta = json.loads(path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(meta["started_utc"].replace("Z", "+00:00")).timestamp()
    try:
        process = psutil.Process(meta["pid"])
        return (abs(process.create_time() - started) < 2.0 and
                process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def parse_line(line):
    _, *parts = line.split()
    return dict(part.split("=", 1) for part in parts)


def first_context(path):
    data = path.read_bytes()
    assert data[:8] == b"M151TX01" and struct.unpack_from("<I", data, 8)[0] == 256
    return struct.unpack_from("<III", data, 12)


def check_golden(raw):
    old = R150.load_table()
    record = json.loads((DOC / "meth162_training_derived_shared_table.json").read_text())
    new = {(r["token"], r["previous"], r["position"]) for r in record["tuples"]}
    assert len(old) == 75 and len(new) == 138
    fixtures = {name: INPUTS[name][0] for name in ("old_hit", "new_hit", "miss")}
    for line in raw.splitlines():
        if not line.startswith("M166_GOLDEN "):
            continue
        row = parse_line(line)
        name = row["cell"]
        token, previous, position = first_context(fixtures[name])
        assert (int(row["token"]), int(row["previous"]), int(row["position"])) == (
            token, previous, position)
        tokens = np.array([token], dtype=np.int64)
        predecessors = np.array([previous], dtype=np.int64)
        positions = np.array([position], dtype=np.int64)
        children = np.array([[89, 89, 89, 89]], dtype=np.int64)
        old_grand, _ = R150.route(tokens, predecessors, positions, children, 3, old)
        new_grand, _ = R150.route(tokens, predecessors, positions, children, 3, new)
        assert int(row["old_grand"]) == int(old_grand[0, 0])
        assert int(row["new_grand"]) == int(new_grand[0, 0])
        yield {"cell": name, "context": [token, previous, position],
               "old_grand": int(old_grand[0, 0]),
               "new_grand": int(new_grand[0, 0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    for name in ("meth165_teacher_process.json", "meth165_route_process.json"):
        assert not active_process(DOC / name), f"GPU work still active: {name}"
    assert digest(FIXTURE_REPORT) == FIXTURE_REPORT_SHA
    for name, (path, sha) in INPUTS.items():
        assert digest(path) == sha, name
    build = ["clang", "-O3", "-mavx2", "-mfma", "-std=c11", "-Wall", "-Wextra",
             str(SOURCE), "-o", str(EXE), "-lm", "-lpsapi"]
    subprocess.run(build, cwd=ROOT, check=True)
    command = [str(EXE), *[str(INPUTS[key][0]) for key in
                          ("exact_bank", "vectors", "old_table", "new_table",
                           "old_hit", "new_hit", "miss")]]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log = args.out.with_suffix(".log")
    assert not log.exists()
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    goldens = list(check_golden(process.stdout))
    assert [g["cell"] for g in goldens] == ["old_hit", "new_hit", "miss"]
    reps = [parse_line(line) for line in process.stdout.splitlines()
            if line.startswith("M166_REP ")]
    cells = [parse_line(line) for line in process.stdout.splitlines()
             if line.startswith("M166_CELL ")]
    summaries = [parse_line(line) for line in process.stdout.splitlines()
                 if line.startswith("M166_SUMMARY ")]
    assert len(reps) == 15 and len(cells) == 3 and len(summaries) == 1
    assert [c["cell"] for c in cells] == ["old_hit", "new_hit", "miss"]
    for c in cells:
        assert float(c["new_median_ms"]) <= 3.5 or c["gate"] == "FAIL"
        assert (float(c["ratio"]) <= 1.25) or c["gate"] == "FAIL"
    passed = all(c["gate"] == "PASS" for c in cells)
    assert (process.returncode == 0) == (summaries[0]["gate"] == "PASS") == passed
    result = {"experiment": "METH-166-native-138-tuple-route-cost",
              "protocol": "METH_166_NATIVE_TABLE_COST_PROTOCOL_20260928.md",
              "fixture_report_sha256": FIXTURE_REPORT_SHA,
              "source_sha256": digest(SOURCE), "executable_sha256": digest(EXE),
              "input_sha256": {name: sha for name, (_, sha) in INPUTS.items()},
              "build_command": build, "command": command,
              "cpu": platform.processor(), "goldens": goldens,
              "repetitions": reps, "cells": cells,
              "summary": summaries[0], "log_sha256": digest(log),
              "decision": "native_table_cost_pass_quality_and_full_rate_open"
                          if passed else "native_table_cost_fail"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "result_sha256": digest(args.out),
                      "cells": cells, "summary": summaries[0]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
