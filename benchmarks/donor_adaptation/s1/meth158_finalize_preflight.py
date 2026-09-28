#!/usr/bin/env python3
"""After the live teacher runner exits, merge and run the frozen preflight.

This deliberately stops if the acquisition is incomplete. It does not
restart teacher generation or alter the frozen manifest or route gates.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
RUNNER_PROCESS = DOC / "meth158_teacher_runner_process.json"
PROGRESS = DOC / "meth158_teacher_progress.json"
SHARDS = DOC / "meth158_teacher_shards"
MERGED = DOC / "meth158_independent_teacher_merged.json"
PREFLIGHT = DOC / "meth159_tenfold_route_support_result.json"
STATUS = DOC / "meth158_finalize_preflight_status.json"
MERGER = Path(__file__).with_name("meth158_merge_teacher_shards.py")
PREFLIGHT_SCRIPT = ROOT / "benchmarks/native_expert_scaling/meth159_tenfold_route_support_screen.py"
MANIFEST_SHA = "09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(stage, **items):
    state = {"experiment": "METH-158-finalize-and-METH-159-preflight",
             "stage": stage, "updated_utc": datetime.now().astimezone().isoformat(),
             **items}
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)
    print(json.dumps(state), flush=True)


def runner_alive(pid, started):
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - started) > 2.0:
            return False
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=int, default=30)
    args = ap.parse_args()
    assert 10 <= args.poll_seconds <= 60
    meta = json.loads(RUNNER_PROCESS.read_text(encoding="utf-8"))
    pid = meta["pid"]
    started = datetime.fromisoformat(meta["started_utc"].replace("Z", "+00:00")).timestamp()
    report("waiting_for_teacher_runner", runner_pid=pid)
    while runner_alive(pid, started):
        time.sleep(args.poll_seconds)

    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    assert progress["prompt_manifest_sha256"] == MANIFEST_SHA
    assert progress["completed_shards"] == progress["total_shards"] == 40
    assert progress["completed_rows"] == 2560
    stdout = Path(meta["stdout"]).read_text(encoding="utf-8")
    final = json.loads(stdout.splitlines()[-1])
    assert final["complete"] is True and final["completed_shards"] == 40
    assert final["progress_sha256"] == digest(PROGRESS)
    report("teacher_complete", runner_pid=pid,
           progress_sha256=digest(PROGRESS), response_tokens=progress["response_tokens"])

    if not MERGED.exists():
        subprocess.run([sys.executable, str(MERGER), "--shard-dir", str(SHARDS),
                        "--out", str(MERGED)], cwd=ROOT, check=True)
    merged_sha = digest(MERGED)
    merged = json.loads(MERGED.read_text(encoding="utf-8"))
    assert merged["prompt_manifest_sha256"] == MANIFEST_SHA
    assert len(merged["shards"]) == 40 and len(merged["rows"]) == 2560
    report("teacher_merged", teacher_merged_sha256=merged_sha,
           teacher_summary=merged["summary"])

    assert not PREFLIGHT.exists(), "METH-159 result already exists; do not rerun it"
    subprocess.run([sys.executable, str(PREFLIGHT_SCRIPT),
                    "--teacher-sha", merged_sha, "--out", str(PREFLIGHT)],
                   cwd=ROOT, check=True)
    result = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert result["teacher_merged_sha256"] == merged_sha
    report("preflight_complete", teacher_merged_sha256=merged_sha,
           preflight_sha256=digest(PREFLIGHT), decision=result["decision"],
           gates=result["gates"])


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        report("failed", error=repr(error))
        raise
