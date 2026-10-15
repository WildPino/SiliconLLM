#!/usr/bin/env python3
"""Adopt the owned migration process, then verify its completed relocation.

This controller adopts the running migration and may make one verified
continuation after its qualified time-cap fault. It never starts a model.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path("D:/_THINGS/Progetti/SiliconLLM")
CONTROL = Path("E:/SiliconLLM_Artifacts/migrations/20261015")
PLAN = ROOT / "docs/maintenance/storage_migration_plan_20261015.json"
BASELINE = ROOT / "docs/maintenance/storage_baseline_20261015.json"
RESULT_DOC = ROOT / "docs/maintenance/STORAGE_RESULT_20261015.md"
VERIFICATION = ROOT / "docs/maintenance/storage_verification_20261015.json"
STATUS = CONTROL / "finalization_status.json"
OWNERSHIP = CONTROL / "finalization_ownership.json"
AUDIT = CONTROL / "relocation_audit.json"
I8_READBACK = CONTROL / "recipient_int8_relocation_readback.json"
MIXED_READBACK = CONTROL / "recipient_mixed_relocation_readback.json"
I8_INPUT = ROOT / "results/native_expert_scaling/recipient_int8_baseline_repair2_20261015"
MIXED_INPUT = ROOT / "results/native_expert_scaling/recipient_mixed_scalar_20261015"
MIXED_READBACK_SCRIPT = ROOT / "benchmarks/native_expert_scaling/recipient_mixed_scalar_readback.py"
LAUNCH_IMPLEMENTATION = CONTROL / "migration_implementation_at_launch.py"
IMPLEMENTATION_CUSTODY = CONTROL / "implementation_custody.json"
INVENTORY = CONTROL / "inventory.json"
RESUME_QUALIFICATION = CONTROL / "resume_qualification.json"
RESUME_PROCESS_IDENTITY = CONTROL / "resume_process_identity.json"
LAUNCH_IMPLEMENTATION_SHA256 = "acabcfad7fa34dc0823bb9433a27b4ec6d665dbe239653c0da4893520460dc81"
MAX_WAIT_SECONDS = 72 * 3600
WAIT_SLICE_MS = 30_000
STEP_TIMEOUT = 3600
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
FILETIME_UNIX_EPOCH = 116444736000000000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_resume_receipt(path: Path, pid: int, started: dict,
                            plan_sha: str) -> dict:
    """Bind an explicitly launched replacement to the preserved original custody."""
    receipt = load(path)
    require(receipt.get("schema") == "STORAGE_RESUME_LAUNCH_V1",
            "unrecognized storage resume launch receipt")
    require(type(receipt.get("pid")) is int and receipt["pid"] == pid and pid > 0,
            "resume PID differs from launch receipt")
    wall_time = receipt.get("wall_time")
    require(type(wall_time) in (int, float) and math.isfinite(wall_time) and
            wall_time > 0, "invalid resume launch wall time")
    expected = {"plan_sha256": plan_sha,
                "original_started_sha256": sha(CONTROL / "started.json"),
                "inventory_sha256": sha(INVENTORY),
                "migration_script_sha256": sha(ROOT / "scripts/migrate_artifacts.py"),
                "qualification_sha256": sha(RESUME_QUALIFICATION)}
    require(all(receipt.get(key) == value for key, value in expected.items()),
            "resume launch plan/inventory/implementation/original custody mismatch")
    qualification = load(RESUME_QUALIFICATION)
    require(qualification.get("schema") == "STORAGE_DIRECTORY_SIZE_RESUME_V1" and
            qualification.get("failure_sha256") == sha(CONTROL / "failure.json"),
            "directory-size resume qualification or original failure custody mismatch")
    if "inventory_sha256" in qualification:
        require(qualification["inventory_sha256"] == expected["inventory_sha256"],
                "qualified inventory changed")
    return receipt


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_exclusive(path: Path, value) -> None:
    require(path.parent.is_dir(), f"output parent missing: {path.parent}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_text_exclusive(path: Path, content: str) -> None:
    require(path.parent.is_dir(), f"output parent missing: {path.parent}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


if os.name == "nt":
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(FILETIME),
                                       ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
                                       ctypes.POINTER(FILETIME)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL


def filetime_ticks(value: FILETIME) -> int:
    return (int(value.high) << 32) | int(value.low)


def filetime_seconds(value: FILETIME) -> float:
    return (filetime_ticks(value) - FILETIME_UNIX_EPOCH) / 10_000_000


def adopt_and_wait(pid: int, started_wall_time: float,
                   creation_filetime: int | None = None,
                   identity_receipt: dict | None = None) -> dict:
    require(os.name == "nt", "Windows process adoption required")
    handle = kernel.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
                                False, pid)
    require(bool(handle), f"owned migration PID cannot be opened: {pid}"
            f" (Windows error {ctypes.get_last_error()})")
    try:
        created, exited, kernel_time, user_time = (FILETIME() for _ in range(4))
        require(kernel.GetProcessTimes(handle, ctypes.byref(created),
                                       ctypes.byref(exited), ctypes.byref(kernel_time),
                                       ctypes.byref(user_time)),
                f"GetProcessTimes failed for PID {pid}: {ctypes.get_last_error()}")
        created_wall_time = filetime_seconds(created)
        created_ticks = filetime_ticks(created)
        if creation_filetime is not None:
            require(created_ticks == creation_filetime,
                    f"PID creation FILETIME differs from exact process identity receipt: {pid}")
            require(abs(started_wall_time - created_wall_time) <= 60,
                    f"PID creation time is outside the retained launch window: {pid}")
        else:
            require(0 <= started_wall_time - created_wall_time <= 60,
                    f"PID creation time does not match retained started receipt: {pid}")
        ownership = {"controller_pid": os.getpid(),
                                   "observed_utc": utc_now(),
                                   "migration_pid": pid,
                                   "migration_creation_filetime": created_ticks,
                                   "migration_creation_wall_time": created_wall_time,
                                   "migration_started_wall_time": started_wall_time,
                                   "controller_sha256": sha(Path(__file__))}
        if identity_receipt is not None:
            ownership.update(identity_receipt)
        write_exclusive(OWNERSHIP, ownership)
        waited = 0.0
        while True:
            remaining = MAX_WAIT_SECONDS - waited
            require(remaining > 0, "migration process exceeded 72-hour wait cap")
            slice_ms = min(WAIT_SLICE_MS, max(1, int(remaining * 1000)))
            before = time.monotonic()
            code = kernel.WaitForSingleObject(handle, slice_ms)
            waited += time.monotonic() - before
            if code == WAIT_OBJECT_0:
                break
            require(code == WAIT_TIMEOUT, f"WaitForSingleObject failed: {code}")
        exit_code = wintypes.DWORD()
        require(kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)),
                f"GetExitCodeProcess failed: {ctypes.get_last_error()}")
        require(kernel.GetProcessTimes(handle, ctypes.byref(created),
                                       ctypes.byref(exited), ctypes.byref(kernel_time),
                                       ctypes.byref(user_time)), "final GetProcessTimes failed")
        return {"pid": pid, "creation_wall_time": created_wall_time,
                "creation_filetime": created_ticks,
                "started_wall_time": started_wall_time,
                "exit_wall_time": filetime_seconds(exited),
                "exit_code": int(exit_code.value), "waited_seconds": waited}
    finally:
        kernel.CloseHandle(handle)


def run_step(name: str, argv: list[str], *, timeout: int = STEP_TIMEOUT) -> dict:
    out = CONTROL / f"finalize_{name}.stdout"
    err = CONTROL / f"finalize_{name}.stderr"
    require(not out.exists() and not err.exists(), f"step output exists: {name}")
    began = time.monotonic()
    with out.open("xb") as stdout, err.open("xb") as stderr:
        child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            code = -1
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    receipt = {"name": name, "argv": argv, "pid": child.pid,
               "exit_code": code, "seconds": time.monotonic() - began,
               "stdout": str(out), "stdout_sha256": sha(out),
               "stderr": str(err), "stderr_sha256": sha(err)}
    write_exclusive(CONTROL / f"finalize_{name}.terminal.json", receipt)
    return receipt


def checked_step(status: dict, name: str, argv: list[str],
                 *, timeout: int = STEP_TIMEOUT) -> dict:
    receipt = run_step(name, argv, timeout=timeout)
    status["steps"].append(receipt)
    require(receipt["exit_code"] == 0,
            f"finalization step failed: {name}; exit {receipt['exit_code']}; "
            f"see {receipt['stderr']}")
    return receipt


def qualified_time_cap_failure() -> dict:
    failure_path = CONTROL / "failure.json"
    require(failure_path.is_file(), "nonzero migration exit lacks retained failure receipt")
    failure = load(failure_path)
    require(failure.get("error_type") == "RuntimeError" and
            failure.get("error") == "copy time cap",
            "migration failure is not the one qualified copy time cap")
    state = load(CONTROL / "state.json")
    allowed = {"PREPARED", "COPYING", "VERIFIED", "ROLLED_BACK", "BACKUP_REMOVED"}
    planned_names = {unit["name"] for unit in load(PLAN)["units"]}
    require(state.get("phase") == "FAILED" and
            set(state.get("units", {})) == planned_names and
            set(state["units"].values()) <= allowed and
            failure.get("state", {}).get("units") == state["units"],
            "copy time cap state is not pre-cutover/completed only")
    require(not (CONTROL / "manifest.json").exists(),
            "copy time cap unexpectedly has a final manifest")
    return {"failure_path": str(failure_path), "failure_sha256": sha(failure_path),
            "state_sha256": sha(CONTROL / "state.json"),
            "state_units": state["units"]}


def free_bytes() -> dict[str, int]:
    return {drive: shutil.disk_usage(drive + ":/").free for drive in ("D", "E")}


def check_git_deletions(status: dict, plan: dict) -> int:
    receipt = checked_step(status, "git_deleted_paths",
                           ["git", "-C", str(ROOT), "-c", "core.quotePath=false",
                            "diff", "--name-only", "--diff-filter=D"])
    deleted = (Path(receipt["stdout"]).read_text(encoding="utf-8")
               .splitlines())
    prefixes = [unit["relative_path"].replace("\\", "/").casefold()
                for unit in plan["units"]]
    in_scope = [path for path in deleted
                if any(path.replace("\\", "/").casefold() == prefix or
                       path.replace("\\", "/").casefold().startswith(prefix + "/")
                       for prefix in prefixes)]
    require(not in_scope, f"tracked files deleted under migrated units: {in_scope[:8]}")
    status["git_deleted_tracked_paths_in_scope"] = 0
    status["git_deleted_tracked_paths_outside_scope"] = len(deleted)
    return len(deleted)


def report_markdown(report: dict) -> str:
    base, final = report["baseline_free_bytes"], report["final_free_bytes"]
    control_uri = "E:/SiliconLLM_Artifacts/migrations/20261015/"
    resume = (f"The explicitly launched verified continuation is bound by resume receipt "
              f"`{report['resume_receipt_path']}`, SHA256 `{report['resume_receipt_sha256']}`. "
              f"Its original started receipt remains preserved for PID "
              f"{report['original_migration_pid']}.\n\n"
              if report.get("resume_receipt_path") else "")
    identity = (f"Process identity is bound by receipt `{report['process_identity_receipt_path']}`, "
                f"SHA256 `{report['process_identity_receipt_sha256']}`, and the exact Windows "
                f"creation FILETIME `{report['migration_process']['creation_filetime']}`.\n\n"
                if report.get("process_identity_receipt_path") else "")
    return ("# Storage migration result (2026-10-15)\n\n"
            f"The {report['units']} planned directory packages completed their verified "
            "D-to-E relocation. Their original D paths now resolve through junctions. "
            "The independent relocation audit and both retained recipient readbacks passed.\n\n"
            "Git reported no deleted tracked paths inside the migrated units.\n\n"
            + resume + identity +
            f"The migration manifest is SHA256 `{report['manifest_sha256']}`. "
            f"It records {report['files']:,} file receipts and "
            f"{report['logical_file_bytes']:,} logical bytes; hardlinks can make the "
            "physical footprint smaller.\n\n"
            "| Volume | Free before (bytes) | Free after (bytes) | Change (bytes) |\n"
            "| --- | ---: | ---: | ---: |\n"
            f"| D | {base['D']:,} | {final['D']:,} | {final['D']-base['D']:+,} |\n"
            f"| E | {base['E']:,} | {final['E']:,} | {final['E']-base['E']:+,} |\n\n"
            "Retained checks: [relocation audit](" + control_uri + "relocation_audit.json), "
            "[INT8 recipient readback](" + control_uri + "recipient_int8_relocation_readback.json), "
            "and [mixed recipient readback](" + control_uri +
            "recipient_mixed_relocation_readback.json). "
            "The [verification receipt](storage_verification_20261015.json) binds their hashes.\n\n"
            "The readbacks verify stored receipts and path continuity. They do not rerun "
            "models, remeasure inference throughput, or change the scientific admission state.\n")


def internal_mixed_readback() -> None:
    parser = argparse.ArgumentParser(description="Preserve the retained logical D directory in the mixed readback")
    parser.add_argument("--internal-mixed-readback", action="store_true", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logical = args.directory.absolute()  # deliberately do not resolve the junction
    require(logical == MIXED_INPUT and logical.is_dir() and
            args.out == MIXED_READBACK and not args.out.exists(),
            "mixed adapter inputs/output differ from fixed logical paths")
    spec = importlib.util.spec_from_file_location("retained_mixed_readback", MIXED_READBACK_SCRIPT)
    require(spec is not None and spec.loader is not None, "frozen mixed readback import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run(logical)
    write_exclusive(args.out, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--resume-receipt", type=Path,
                        help="Explicit replacement launch receipt; original started.json remains unchanged")
    parser.add_argument("--process-identity-receipt", type=Path,
                        help="Exact Windows creation FILETIME receipt bound to the resume launch")
    args = parser.parse_args()
    require(args.pid > 0, "positive owned PID required")
    require(ROOT.is_dir() and CONTROL.is_dir(), "fixed roots missing")
    outputs = [STATUS, OWNERSHIP, AUDIT, I8_READBACK, MIXED_READBACK, RESULT_DOC, VERIFICATION]
    outputs += [CONTROL / f"finalize_{step}.{suffix}" for step in
                ("resume_verified", "git_deleted_paths", "relocation_audit",
                 "int8_readback", "mixed_readback",
                 "catalog_build", "catalog_check") for suffix in
                ("stdout", "stderr", "terminal.json")]
    require(all(not path.exists() for path in outputs),
            "finalization output already exists; no automatic retry")
    status = {"schema": "STORAGE_FINALIZATION_V1", "status": "RUNNING",
              "started_utc": utc_now(), "pid": args.pid,
              "controller_sha256": sha(Path(__file__)),
              "migration_script_sha256_at_finalization": sha(ROOT / "scripts/migrate_artifacts.py"),
              "steps": []}
    try:
        started = load(CONTROL / "started.json")
        plan_sha = sha(PLAN)
        require(started["plan_sha256"] == plan_sha and
                load(CONTROL / "frozen_plan.json")["raw_sha256"] == plan_sha,
                "plan changed since migration started")
        baseline_sha = sha(BASELINE)
        baseline = load(BASELINE)
        baseline_free = {x["DriveLetter"]: x["SizeRemaining"] for x in baseline["volumes"]}
        require(set(baseline_free) == {"D", "E"}, "baseline volume keys")
        custody = load(IMPLEMENTATION_CUSTODY)
        require(sha(LAUNCH_IMPLEMENTATION) == LAUNCH_IMPLEMENTATION_SHA256 and
                custody["sha256"] == LAUNCH_IMPLEMENTATION_SHA256 and
                custody["bytes"] == LAUNCH_IMPLEMENTATION.stat().st_size and
                custody["initial_process_pid"] == started["pid"],
                "original migration implementation custody")
        status["original_migration_pid"] = started["pid"]
        process_started_wall_time = started["wall_time"]
        creation_filetime = None
        identity_binding = None
        if args.resume_receipt is not None:
            resume_path = args.resume_receipt.absolute()
            status["resume_receipt_path"] = str(resume_path)
            status["resume_receipt_sha256"] = sha(resume_path)
            resume = validate_resume_receipt(resume_path, args.pid, started, plan_sha)
            process_started_wall_time = resume["wall_time"]
            status["resume_launch"] = resume
            status["resume_qualification_path"] = str(RESUME_QUALIFICATION)
            status["resume_qualification_sha256"] = sha(RESUME_QUALIFICATION)
        else:
            require(started["pid"] == args.pid,
                    "requested PID differs from retained migration PID")
        if args.process_identity_receipt is not None:
            require(args.resume_receipt is not None,
                    "exact process identity receipt requires a qualified resume launch")
            identity_path = args.process_identity_receipt.absolute()
            require(identity_path == RESUME_PROCESS_IDENTITY.absolute(),
                    "process identity receipt differs from the fixed control path")
            status["process_identity_receipt_path"] = str(identity_path)
            status["process_identity_receipt_sha256"] = sha(identity_path)
            identity = load(identity_path)
            require(identity.get("schema") == "STORAGE_RESUME_PROCESS_IDENTITY_V1" and
                    type(identity.get("pid")) is int and identity["pid"] == args.pid and
                    identity.get("resume_launch_sha256") == status["resume_receipt_sha256"],
                    "exact process identity is not bound to this resume PID and launch")
            creation_filetime = identity.get("creation_filetime")
            require(type(creation_filetime) is int and creation_filetime > 0,
                    "positive integer Windows creation FILETIME required")
            identity_binding = {key: status[key] for key in
                                ("process_identity_receipt_path", "process_identity_receipt_sha256")}
        status["plan_sha256"] = plan_sha
        status["baseline_sha256"] = baseline_sha
        status["launch_implementation_sha256"] = LAUNCH_IMPLEMENTATION_SHA256
        status["implementation_custody_sha256"] = sha(IMPLEMENTATION_CUSTODY)
        status["migration_process"] = adopt_and_wait(args.pid, process_started_wall_time,
                                                      creation_filetime, identity_binding)
        if status["migration_process"]["exit_code"] != 0:
            status["qualified_resume"] = qualified_time_cap_failure()
            remaining = MAX_WAIT_SECONDS - status["migration_process"]["waited_seconds"]
            require(remaining > 0, "no time remains for one qualified resume")
            status["qualified_resume"]["attempts"] = 1
            checked_step(status, "resume_verified",
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", str(ROOT / "scripts/migrate_artifacts.py"),
                 "run", "--plan", str(PLAN), "--resume-verified"],
                timeout=int(remaining))
        state = load(CONTROL / "state.json")
        manifest_path = CONTROL / "manifest.json"
        require(manifest_path.is_file() and state["phase"] == "COMPLETE" and
                all(value == "BACKUP_REMOVED" for value in state["units"].values()),
                "migration lacks complete manifest/state; no readbacks run")
        manifest = load(manifest_path)
        require(manifest["status"] == "COMPLETE" and
                manifest["plan_sha256"] == plan_sha, "manifest status or plan")
        manifest_sha = sha(manifest_path)
        python = sys.executable
        check_git_deletions(status, load(PLAN))
        checked_step(status, "relocation_audit",
            [python, "-I", "-S", "-B", "-X", "utf8", str(ROOT / "scripts/verify_storage_migration.py"),
             "--out", str(AUDIT)])
        require(load(AUDIT)["status"] == "PASS", "relocation audit status")
        checked_step(status, "int8_readback",
            [python, "-I", "-S", "-B", "-X", "utf8", str(ROOT / "benchmarks/native_expert_scaling/recipient_int8_readback.py"),
             str(I8_INPUT), str(I8_READBACK)])
        require(load(I8_READBACK)["status"] == "PASS_RETAINED_COST_READBACK",
                "INT8 readback status")
        checked_step(status, "mixed_readback",
            [python, "-I", "-S", "-B", "-X", "utf8", str(Path(__file__).absolute()),
             "--internal-mixed-readback", "--directory", str(MIXED_INPUT),
             "--out", str(MIXED_READBACK)])
        require(load(MIXED_READBACK)["status"] == "READBACK_PASS",
                "mixed readback status")
        status["mixed_readback_adapter"] = {
            "frozen_script_sha256": sha(MIXED_READBACK_SCRIPT),
            "logical_directory": str(MIXED_INPUT),
            "reason": "Retained command argv contains the original D path; resolving the junction would change the historical comparison."}
        final_free = free_bytes()
        report = {"schema": "STORAGE_VERIFICATION_V1", "status": "PASS",
                  "observed_utc": utc_now(), "migration_pid": args.pid,
                  "migration_process": status["migration_process"],
                  "plan_sha256": plan_sha, "baseline_sha256": baseline_sha,
                  "manifest_sha256": manifest_sha,
                  "launch_implementation_sha256": LAUNCH_IMPLEMENTATION_SHA256,
                  "implementation_custody_sha256": sha(IMPLEMENTATION_CUSTODY),
                  "git_deleted_tracked_paths_in_scope": 0,
                  "units": len(state["units"]), "files": manifest["files"],
                  "logical_file_bytes": manifest["logical_file_bytes"],
                  "baseline_free_bytes": baseline_free, "final_free_bytes": final_free,
                  "free_delta_bytes": {key: final_free[key] - baseline_free[key]
                                       for key in ("D", "E")},
                  "evidence": {"relocation_audit": {"path": str(AUDIT), "sha256": sha(AUDIT)},
                               "int8_readback": {"path": str(I8_READBACK), "sha256": sha(I8_READBACK)},
                               "mixed_readback": {"path": str(MIXED_READBACK),
                                                  "sha256": sha(MIXED_READBACK)}},
                  "scope": "Storage custody and retained readback only; no model execution or new rate claim."}
        report["original_migration_pid"] = started["pid"]
        if args.resume_receipt is not None:
            for key in ("resume_receipt_path", "resume_receipt_sha256",
                        "resume_qualification_path", "resume_qualification_sha256"):
                report[key] = status[key]
        if args.process_identity_receipt is not None:
            for key in ("process_identity_receipt_path", "process_identity_receipt_sha256"):
                report[key] = status[key]
        write_exclusive(VERIFICATION, report)
        write_text_exclusive(RESULT_DOC, report_markdown(report))
        checked_step(status, "catalog_build",
            [python, "-I", "-S", "-B", "-X", "utf8", str(ROOT / "scripts/build_docs_catalog.py"), "build"])
        checked_step(status, "catalog_check",
            [python, "-I", "-S", "-B", "-X", "utf8", str(ROOT / "scripts/build_docs_catalog.py"), "check"])
        status["status"] = "PASS"
        status["verification_sha256"] = sha(VERIFICATION)
        status["result_document_sha256"] = sha(RESULT_DOC)
        status["catalog_sha256"] = sha(ROOT / "docs/navigation/catalog.json")
        status["catalog_index_sha256"] = sha(ROOT / "docs/navigation/index.html")
    except BaseException as error:
        status["status"] = "FAIL"
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        raise
    finally:
        status["finished_utc"] = utc_now()
        write_exclusive(STATUS, status)


if __name__ == "__main__":
    if "--internal-mixed-readback" in sys.argv[1:]:
        internal_mixed_readback()
    else:
        main()
