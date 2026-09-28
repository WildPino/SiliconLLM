#!/usr/bin/env python3
"""Run the frozen CPU factor screen only after the GPU preflight exits."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
META = DOC / "meth158_finalize_preflight_process.json"
STATUS = DOC / "meth158_finalize_preflight_status.json"
OUT = DOC / "meth160_actual_b_bank_cpu_result.json"
RUNNER = Path(__file__).with_name("meth160_run_actual_b_bank_cpu.py")
WAIT_STATUS = DOC / "meth160_wait_status.json"


def report(stage, **items):
    state = {"experiment": "METH-160-deferred-CPU-component-screen",
             "stage": stage, "updated_utc": datetime.now().astimezone().isoformat(),
             **items}
    temporary = WAIT_STATUS.with_suffix(WAIT_STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(WAIT_STATUS)
    print(json.dumps(state), flush=True)


def alive(pid, started):
    try:
        process = psutil.Process(pid)
        return (abs(process.create_time() - started) < 2.0 and
                process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def main():
    meta = json.loads(META.read_text(encoding="utf-8"))
    pid = meta["pid"]
    started = datetime.fromisoformat(meta["started_utc"].replace("Z", "+00:00")).timestamp()
    report("waiting_for_preflight", finalizer_pid=pid)
    while alive(pid, started):
        time.sleep(30)
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    assert status["stage"] == "preflight_complete", status
    assert not OUT.exists()
    report("running_cpu_screen", preflight_decision=status["decision"])
    subprocess.run([sys.executable, str(RUNNER), "--out", str(OUT)],
                   cwd=ROOT, check=True)
    result = json.loads(OUT.read_text(encoding="utf-8"))
    report("cpu_screen_complete", preflight_decision=status["decision"],
           cpu_decision=result["decision"], cpu_gates=result["gates"])


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        report("failed", error=repr(error))
        raise
