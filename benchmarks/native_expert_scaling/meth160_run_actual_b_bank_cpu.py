#!/usr/bin/env python3
"""SHA-bind, build and report the frozen METH-160 paired CPU cells."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import psutil


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
ART = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling"
SOURCE = Path(__file__).with_name("meth160_actual_b_bank_cpu.c")
EXE = ART / "meth160_actual_b_bank_cpu.exe"
INPUTS = {
    "exact_bank": (ART / "meth126_shared_a_factor_bank.bin",
                   "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1"),
    "vectors": (ART / "meth125_e1280_vectors.bin",
                "f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699"),
    "table": (ART / "meth151_shared_table.bin",
              "926362655bcc71bf7bd1ceee944e56ba8f35ef124c8f60b6e0052d6525ddfc27"),
    "hit": (ART / "meth151_context_hit.bin",
            "c211bde3459b3a1df27250ee0e785cde412cba07fb763aeb519281588f54a629"),
    "miss": (ART / "meth151_context_miss.bin",
             "ecce0701d71201f9a48495adbc9814632c9f5b0929675a30daf2ffc92ac8682e"),
    "control": (ART / "meth136_control_bf16.bin",
                "d5d9e6b3753ab55ecbb20a8f60d1ba11a6d9bdd41f66bb34909207b9a29ba299"),
    "candidate": (ART / "meth155_factorized_candidate_bf16.bin",
                  "c335d425c97fac5ec7284eea97ce3f3dc9462264ec79e7f46ce8598465d3e01d"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def active_process(meta_path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(meta["started_utc"].replace("Z", "+00:00")).timestamp()
    try:
        process = psutil.Process(meta["pid"])
        return (abs(process.create_time() - started) < 2.0 and
                process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def parse_output(raw):
    reps = []
    layers = []
    summary = None
    for line in raw.splitlines():
        if line.startswith(("M160_REP ", "M160_LAYER ", "M160_SUMMARY ")):
            tag, *parts = line.split()
            row = dict(part.split("=", 1) for part in parts)
            if tag == "M160_REP":
                reps.append(row)
            elif tag == "M160_LAYER":
                layers.append(row)
            else:
                assert summary is None
                summary = row
    assert len(reps) == 5 and len(layers) == 24 and summary is not None
    assert summary["tokens"] == "256" and summary["states"] == "6144"
    assert summary["selected_B_bytes_per_token"] == str(4 * 24 * 896 * 8 * 2)
    assert summary["control_bank_bytes"] == str(INPUTS["control"][0].stat().st_size)
    assert summary["candidate_bank_bytes"] == str(INPUTS["candidate"][0].stat().st_size)
    return {"repetitions": reps, "layers": layers, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    status = DOC / "meth158_finalize_preflight_status.json"
    assert json.loads(status.read_text(encoding="utf-8"))["stage"] == "preflight_complete"
    for name in ("meth158_teacher_runner_process.json", "meth158_finalize_preflight_process.json"):
        assert not active_process(DOC / name), f"GPU work still active: {name}"
    for name, (path, expected) in INPUTS.items():
        assert digest(path) == expected, name

    build = ["clang", "-O3", "-mavx2", "-mfma", "-std=c11", "-Wall", "-Wextra",
             str(SOURCE), "-o", str(EXE), "-lm", "-lpsapi"]
    subprocess.run(build, cwd=ROOT, check=True)
    smoke = subprocess.run([str(EXE), "--selftest"], cwd=ROOT,
                           capture_output=True, text=True, check=True)
    assert "exact_residual=1" in smoke.stdout
    cells = {}
    for name in ("hit", "miss"):
        command = [str(EXE), str(INPUTS["exact_bank"][0]),
                   str(INPUTS["vectors"][0]), str(INPUTS[name][0]),
                   str(INPUTS["table"][0]), str(INPUTS["control"][0]),
                   str(INPUTS["candidate"][0])]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log = args.out.with_name(args.out.stem + f".{name}.log")
        assert not log.exists()
        log.write_text(process.stdout + process.stderr, encoding="utf-8")
        parsed = parse_output(process.stdout)
        assert (process.returncode == 0) == (parsed["summary"]["gate"] == "PASS")
        cells[name] = {"command": command, "exit_code": process.returncode,
                       "log_path": str(log.resolve()), "log_sha256": digest(log), **parsed}
    gates = {name: cell["summary"]["gate"] == "PASS" for name, cell in cells.items()}
    result = {"experiment": "METH-160-actual-B-bank-CPU-selected-path",
              "protocol": "METH_160_ACTUAL_B_BANK_CPU_COST_PROTOCOL_20260928.md",
              "source_sha256": digest(SOURCE), "executable_sha256": digest(EXE),
              "input_sha256": {name: expected for name, (_, expected) in INPUTS.items()},
              "preflight_status_sha256": digest(status),
              "build_command": build, "selftest": smoke.stdout.strip(),
              "cpu": platform.processor(), "cells": cells, "gates": gates,
              "decision": ("selected_B_component_pass_quality_and_full_rate_open"
                           if all(gates.values()) else "selected_B_component_fail")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "result_sha256": digest(args.out),
                      "hit": cells["hit"]["summary"],
                      "miss": cells["miss"]["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
