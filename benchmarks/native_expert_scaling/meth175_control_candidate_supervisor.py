#!/usr/bin/env python3
"""Start the frozen METH-175 candidate only after a valid control result."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
ART = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling"
CONTROL_PROCESS = DOC / "meth175_control_process.json"
CONTROL_RESULT = DOC / "meth175_control_long_result.json"
CANDIDATE_RESULT = DOC / "meth175_candidate_long_result.json"
CANDIDATE_BANK = ART / "meth175_candidate_long_bf16.bin"
CANDIDATE_STDOUT = DOC / "meth175_candidate_stdout.log"
CANDIDATE_STDERR = DOC / "meth175_candidate_stderr.log"
CANDIDATE_PROCESS = DOC / "meth175_candidate_process.json"
STATE = DOC / "meth175_sequence_state.json"
TRAINER = ROOT / "benchmarks/native_expert_scaling/meth175_matched_long_train.py"
DRAWS_SHA = "9d4e06eced3cc1c51bac4e6219013f37f4495acd10c87827b567f0966f5c490c"
TABLE_SHA = "e89ff2cecbf887b85b5e361d47a8b918d8ccb93bd5c385a580ba993b93ac72db"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(stage, **details):
    payload = {"experiment": "METH-175-separate-arm-sequence",
               "stage": stage, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **details}
    write_json(STATE, payload)
    print(json.dumps(payload), flush=True)


def wait_for_control(process_info):
    pid = process_info["pid"]
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        process = None
    if process is not None:
        # A reused PID must never authorize candidate training.
        started = time.strptime(process_info["started_utc"][:19], "%Y-%m-%dT%H:%M:%S")
        expected = __import__("calendar").timegm(started)
        assert abs(process.create_time() - expected) < 120
        status("waiting_control", control_pid=pid)
        while True:
            try:
                process.wait(timeout=30)
                break
            except psutil.TimeoutExpired:
                continue
    if not CONTROL_RESULT.is_file():
        raise RuntimeError("control exited without complete result")


def checked_control(process_info):
    result_sha = digest(CONTROL_RESULT)
    result = json.loads(CONTROL_RESULT.read_text(encoding="utf-8"))
    assert result["decision"] == "matched_long_control_complete"
    assert result["arm"] == "control" and result["updates"] == 3840
    assert len(result["training"]["records"]) == 3840
    assert all(result["gates"].values())
    assert result["draws_sha256"] == DRAWS_SHA
    assert result["table_sha256"] == TABLE_SHA
    assert result["training"]["initial_parity"]["logit_max_abs_error"] == 0
    bank = Path(result["artifact"]["path"])
    assert bank.resolve() == Path(process_info["bank"]).resolve()
    assert bank.stat().st_size == result["artifact"]["bytes"]
    bank_sha = digest(bank)
    assert bank_sha == result["artifact"]["sha256"]
    assert result["artifact"]["readback_exact"]
    return result_sha, bank_sha


def run_candidate(control_result_sha, control_bank_sha):
    for path in (CANDIDATE_RESULT, CANDIDATE_BANK, CANDIDATE_STDOUT,
                 CANDIDATE_STDERR, CANDIDATE_PROCESS):
        assert not path.exists(), path
    command = [sys.executable, "-u", str(TRAINER), "--arm", "candidate",
               "--out", str(CANDIDATE_RESULT), "--bank", str(CANDIDATE_BANK),
               "--control-result-sha", control_result_sha,
               "--control-bank-sha", control_bank_sha]
    with CANDIDATE_STDOUT.open("w", encoding="utf-8") as stdout, \
            CANDIDATE_STDERR.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        write_json(CANDIDATE_PROCESS, {
            "experiment": "METH-175-candidate-long",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": process.pid, "out": str(CANDIDATE_RESULT),
            "bank": str(CANDIDATE_BANK), "stdout": str(CANDIDATE_STDOUT),
            "stderr": str(CANDIDATE_STDERR),
            "control_result_sha256": control_result_sha,
            "control_bank_sha256": control_bank_sha,
            "draws_sha256": DRAWS_SHA, "max_hours": 12,
            "max_rss_gib": 52, "max_gpu_allocated_gib": 10.5})
        status("candidate_running", control_result_sha256=control_result_sha,
               control_bank_sha256=control_bank_sha, candidate_pid=process.pid)
        exit_code = process.wait()
    if exit_code != 0 or not CANDIDATE_RESULT.is_file():
        raise RuntimeError(f"candidate exited without complete result: {exit_code}")
    candidate = json.loads(CANDIDATE_RESULT.read_text(encoding="utf-8"))
    if candidate["decision"] != "matched_long_candidate_pass_fresh_quality_pending":
        raise RuntimeError(f"candidate gate failed: {candidate['decision']}")
    assert candidate["updates"] == 3840 and all(candidate["gates"].values())
    assert digest(CANDIDATE_BANK) == candidate["artifact"]["sha256"]
    status("training_complete_quality_pending",
           control_result_sha256=control_result_sha,
           control_bank_sha256=control_bank_sha,
           candidate_result_sha256=digest(CANDIDATE_RESULT),
           candidate_bank_sha256=candidate["artifact"]["sha256"])


def main():
    try:
        process_info = json.loads(CONTROL_PROCESS.read_text(encoding="utf-8"))
        assert process_info["experiment"] == "METH-175-control-long"
        assert process_info["draws_sha256"] == DRAWS_SHA
        assert Path(process_info["out"]).resolve() == CONTROL_RESULT.resolve()
        wait_for_control(process_info)
        result_sha, bank_sha = checked_control(process_info)
        status("control_verified", control_result_sha256=result_sha,
               control_bank_sha256=bank_sha)
        run_candidate(result_sha, bank_sha)
    except BaseException as error:
        status("stopped", error=repr(error))
        raise


if __name__ == "__main__":
    main()
