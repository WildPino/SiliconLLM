#!/usr/bin/env python3
"""Qualify and supervise the verified directory-size migration continuation."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path("D:/_THINGS/Progetti/SiliconLLM")
CONTROL = Path("E:/SiliconLLM_Artifacts/migrations/20261015")
QUALIFICATION = CONTROL / "resume_qualification.json"
HISTORY = CONTROL / "history/initial_failure_20261015"
PLAN = ROOT / "docs/maintenance/storage_migration_plan_20261015.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True)
        stream.write("\n")


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def guarded_control_file(path):
    path = Path(path).absolute()
    check(path.resolve().is_relative_to(CONTROL.resolve()), "control-file boundary violation")
    check(not path.is_symlink() and not os.path.isjunction(path), "control-file link forbidden")
    return path


def prepare():
    check(not QUALIFICATION.exists() and not HISTORY.exists(), "continuation already prepared")
    spec = importlib.util.spec_from_file_location("storage_resume_impl", ROOT / "scripts/migrate_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_roots()
    state = load(CONTROL / "state.json")
    failure = load(CONTROL / "failure.json")
    check(state["phase"] == "FAILED", "only failed pre-cutover migration qualifies")
    expected_error = "source metadata changed: " + str(ROOT / "kaggle_rung1/_rttmp/.kaggle/uploads")
    check(failure["error_type"] == "RuntimeError" and failure["error"] == expected_error,
          "unexpected first fault; no automatic qualification")
    check(state["units"]["kaggle-rung1-staging"] == "COPYING" and
          set(state["units"].values()) <= {"COPYING", "PREPARED", "BACKUP_REMOVED"},
          "partial cutover requires separate review")
    inventory = load(CONTROL / "inventory.json")
    unit = inventory["units"]["kaggle-rung1-staging"]
    source = ROOT / unit["relative_path"]
    actual = module.inventory_unit(source)
    old = {r["relative_path"]: r["metadata"] for r in unit["entries"]}
    now = {r["relative_path"]: r["metadata"] for r in actual}
    check(set(old) == set(now), "current source child membership changed")
    differences = []
    for rel, expected in old.items():
        current = now[rel]
        keys = module.unchanged_metadata_keys(expected)
        check(all(current.get(k) == expected.get(k) for k in keys),
              "source has a change beyond ordinary-directory length: " + rel)
        if current["size"] != expected["size"]:
            check(expected["kind"] == "directory", "file length drift is forbidden")
            differences.append({"relative_path": rel, "old_size": expected["size"],
                                "current_size": current["size"], "kind": "directory"})
    check(differences, "observed directory-size fault was not reproduced")
    receipts = module.receipts()
    files = [r for r in unit["entries"] if r["metadata"]["kind"] == "file"]
    check(all(("kaggle-rung1-staging", r["relative_path"]) in receipts for r in files),
          "unreceipted current-unit files need separate review")
    module.verify_tree(module.EROOT / unit["relative_path"], unit["entries"], receipts,
                       "kaggle-rung1-staging", hash_files=False)
    HISTORY.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name in ("state.json", "status.json", "failure.json", "started.json", "receipts.jsonl",
                 "controller_launch.json", "finalization_status.json", "finalization_ownership.json"):
        original = guarded_control_file(CONTROL / name)
        if original.is_file():
            target = guarded_control_file(HISTORY / name)
            with original.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            check(sha(original) == sha(target), "history snapshot mismatch")
            archived[name] = {"path": str(target), "sha256": sha(target)}
    # Move only the two exact terminal output files after their byte-exact copy
    # has been verified. The immutable history copies retain every old receipt.
    for name in ("finalization_status.json", "finalization_ownership.json"):
        original = guarded_control_file(CONTROL / name)
        if original.exists():
            destination = guarded_control_file(HISTORY / ("original_" + name))
            check(not destination.exists(), "archive destination occupied")
            original.rename(destination)
    qualification = {"schema": "STORAGE_DIRECTORY_SIZE_RESUME_V1", "prepared_wall_time": time.time(),
        "failure_sha256": sha(CONTROL / "failure.json"),
        "inventory_sha256": sha(CONTROL / "inventory.json"),
        "plan_sha256": sha(PLAN), "original_started_sha256": sha(CONTROL / "started.json"),
        "migration_script_sha256": sha(ROOT / "scripts/migrate_artifacts.py"),
        "finalizer_script_sha256": sha(ROOT / "scripts/finalize_storage_maintenance.py"),
        "directory_size_differences": differences, "current_unit_files_receipted": len(files),
        "current_unit_payload_rehash": "Pending explicit --resume-verified; all accepted file receipts are rehashed before cutover",
        "archived_original_receipts": archived,
        "free_bytes_at_resume": {d: shutil.disk_usage(d + ":/").free for d in ("D", "E")},
        "policy": "Only ordinary-directory st_size excluded; exact child inventory, identity, mtime, attributes, file lengths, file SHA256, streams and links remain checked"}
    save(QUALIFICATION, qualification)
    print(json.dumps({"qualified": True, "directory_size_differences": differences,
                      "files_to_revalidate": len(files)}), flush=True)


def run():
    qualification = load(QUALIFICATION)
    check(qualification["schema"] == "STORAGE_DIRECTORY_SIZE_RESUME_V1", "qualification schema")
    for path, key in ((PLAN, "plan_sha256"), (CONTROL / "inventory.json", "inventory_sha256"),
                      (CONTROL / "started.json", "original_started_sha256"),
                      (ROOT / "scripts/migrate_artifacts.py", "migration_script_sha256"),
                      (ROOT / "scripts/finalize_storage_maintenance.py", "finalizer_script_sha256")):
        check(sha(path) == qualification[key], "qualified input changed: " + str(path))
    with (CONTROL / "resume_controller.lock").open("x", encoding="ascii") as lock:
        lock.write(str(os.getpid()))
    status = {"schema": "STORAGE_RESUME_CONTROLLER_V1", "status": "RUNNING",
              "pid": os.getpid(), "started_wall_time": time.time(),
              "qualification_sha256": sha(QUALIFICATION)}
    save(CONTROL / "resume_controller_started.json", status)
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    status["system_sleep_inhibition_requested"] = bool(awake)
    try:
        argv = [sys.executable, "-I", "-S", "-B", "-X", "utf8",
                str(ROOT / "scripts/migrate_artifacts.py"), "run", "--plan", str(PLAN), "--resume-verified"]
        with (CONTROL / "resume_copy.stdout.log").open("xb") as stdout, (CONTROL / "resume_copy.stderr.log").open("xb") as stderr:
            worker = subprocess.Popen(argv, cwd=ROOT, stdout=stdout, stderr=stderr,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
            launched = time.time()
            receipt = {"schema": "STORAGE_RESUME_LAUNCH_V1", "pid": worker.pid, "wall_time": launched,
                       "controller_pid": os.getpid(), "argv": argv,
                       "plan_sha256": sha(PLAN), "inventory_sha256": sha(CONTROL / "inventory.json"),
                       "original_started_sha256": sha(CONTROL / "started.json"),
                       "migration_script_sha256": sha(ROOT / "scripts/migrate_artifacts.py"),
                       "qualification_sha256": sha(QUALIFICATION)}
            launch_path = CONTROL / "resume_launch.json"
            save(launch_path, receipt)
            finalizer_argv = [sys.executable, "-I", "-S", "-B", "-X", "utf8",
                             str(ROOT / "scripts/finalize_storage_maintenance.py"), "--pid", str(worker.pid),
                             "--resume-receipt", str(launch_path)]
            with (CONTROL / "resume_finalizer.stdout.log").open("xb") as fout, (CONTROL / "resume_finalizer.stderr.log").open("xb") as ferr:
                finalizer = subprocess.Popen(finalizer_argv, cwd=ROOT, stdout=fout, stderr=ferr,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                save(CONTROL / "resume_finalizer_launch.json", {"pid": finalizer.pid,
                     "wall_time": time.time(), "argv": finalizer_argv,
                     "script_sha256": sha(ROOT / "scripts/finalize_storage_maintenance.py")})
                status["migration_exit_code"] = worker.wait()
                status["finalizer_exit_code"] = finalizer.wait()
        status["status"] = "PASS" if status["migration_exit_code"] == status["finalizer_exit_code"] == 0 else "FAIL"
    except BaseException as exc:
        status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        status["finished_wall_time"] = time.time()
        save(CONTROL / "resume_controller_terminal.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.command == "prepare" else run()
