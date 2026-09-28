#!/usr/bin/env python3
"""Wait for frozen METH-165 teacher, then run support and native CPU gates."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

import psutil


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
PYTHON = ROOT / ".venv/Scripts/python.exe"
PROCESS = DOC / "meth165_teacher_process.json"
MANIFEST = DOC / "meth165_expanded_training_manifest.json"
MANIFEST_SHA = "2875ac50a4de005b56ad6403c6bd30e358b87eb21d7dbb1918738957855068ba"
TEACHER = DOC / "meth165_expanded_teacher_merged.json"
ROUTE = DOC / "meth165_expanded_route_support_result.json"
CPU = DOC / "meth166_native_table_cost_result.json"
STATUS = DOC / "meth165_finalize_status.json"
SHARDS = DOC / "meth165_teacher_shards"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(stage, **fields):
    record = {"experiment": "METH-165-finalize-support-and-cpu",
              "stage": stage, "updated_utc": datetime.utcnow().isoformat() + "Z",
              **fields}
    STATUS.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)


def run(command, stdout_path, stderr_path, timeout):
    assert not stdout_path.exists() and not stderr_path.exists()
    with stdout_path.open("w", encoding="utf-8") as stdout, \
         stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                timeout=timeout)
    return result.returncode


def main():
    assert not STATUS.exists()
    assert sha(MANIFEST) == MANIFEST_SHA
    meta = json.loads(PROCESS.read_text(encoding="utf-8"))
    assert meta["manifest_sha256"] == MANIFEST_SHA
    started = datetime.fromisoformat(meta["started_utc"].replace("Z", "+00:00")).timestamp()
    try:
        process = psutil.Process(meta["pid"])
        assert abs(process.create_time() - started) < 2.0
        save("waiting_for_teacher", teacher_pid=meta["pid"])
        process.wait(timeout=75 * 60)
    except psutil.NoSuchProcess:
        pass
    assert TEACHER.exists(), "teacher process exited without merged artifact"
    merged = json.loads(TEACHER.read_text(encoding="utf-8"))
    assert merged["experiment"] == "METH-165-expanded-teacher-merged"
    assert merged["manifest_sha256"] == MANIFEST_SHA
    assert len(merged["rows"]) == 1280 and len(merged["shards"]) == 20
    for first, item in zip(range(0, 1280, 64), merged["shards"]):
        path = SHARDS / f"shard_{first:04d}_{first + 64:04d}.json"
        assert path.exists() and sha(path) == item["sha256"]
        assert item["start"] == first and item["stop"] == first + 64
        shard = json.loads(path.read_text(encoding="utf-8"))
        assert shard["manifest_sha256"] == MANIFEST_SHA
        assert len(shard["rows"]) == 64
    teacher_sha = sha(TEACHER)
    save("teacher_complete", teacher_sha256=teacher_sha,
         summary=merged["summary"])
    assert not ROUTE.exists()
    command = [str(PYTHON), str(ROOT / "benchmarks/native_expert_scaling/"
                                "meth165_expanded_route_support.py"),
               "--teacher-sha", teacher_sha, "--out", str(ROUTE)]
    save("route_running", teacher_sha256=teacher_sha, route_command=command)
    code = run(command, DOC / "meth165_route.stdout.log",
               DOC / "meth165_route.stderr.log", 45 * 60)
    if code != 0 or not ROUTE.exists():
        save("route_failed", teacher_sha256=teacher_sha, route_exit_code=code)
        return
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    route_sha = sha(ROUTE)
    save("route_complete", teacher_sha256=teacher_sha,
         route_sha256=route_sha, route_decision=route["decision"],
         route_gates=route["gates"])
    assert not CPU.exists()
    command = [str(PYTHON), str(ROOT / "benchmarks/native_expert_scaling/"
                                "meth166_run_native_table_cost.py"),
               "--out", str(CPU)]
    save("cpu_running", teacher_sha256=teacher_sha,
         route_sha256=route_sha, cpu_command=command)
    code = run(command, DOC / "meth166_cpu.stdout.log",
               DOC / "meth166_cpu.stderr.log", 10 * 60)
    if code != 0 or not CPU.exists():
        save("cpu_failed", teacher_sha256=teacher_sha,
             route_sha256=route_sha, cpu_exit_code=code)
        return
    cpu = json.loads(CPU.read_text(encoding="utf-8"))
    save("complete", teacher_sha256=teacher_sha,
         route_sha256=route_sha, route_decision=route["decision"],
         cpu_sha256=sha(CPU), cpu_decision=cpu["decision"])


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save("failed", error=repr(error))
        raise
