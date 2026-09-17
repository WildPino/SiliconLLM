#!/usr/bin/env python3
"""Bounded, write-once STRAT-02 first-document teacher smoke supervisor.

This is an apparatus-safety smoke only.  It scores exactly calibration JSONL
row zero, fixed before execution; its bits and BPB are observations, never a
quality gate or a rate claim.  Raw document text is never written to a report.

Usage (the only command that can perform a real forward)::

    python benchmarks/donor_adaptation/density/strat02_bounded_smoke.py \
        --run --output-dir D:\\runs\\strat02-smoke [--snapshot E:\\...\\<revision>]

The output directory contains write-once ``supervisor_manifest.json``,
``supervisor_log.jsonl``, ``worker_result.json``, and
``supervisor_result.json``.  Existing files are never overwritten.  Use
``--selftest`` for tiny mock-supervisor controls; it never opens donor weights,
downloads, or runs a model.  ``--worker`` is an internal child mode.

The parent starts a direct Python interpreter (bypassing the Windows venv
redirector) and samples only that worker at five-second intervals.  On a
safety breach it terminates only the worker it created; it never enumerates
or kills a process tree.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus" / "strat02_document_holdout_v1"
REVISION = "d2a4949c9d4ad6cf47fbac131f7e020077332b21"
EXPECTED_MANIFEST_SHA256 = "56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749"
EXPECTED_CALIB_SHA256 = "f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f"
PREREGISTERED_ROW = {
    "source_document_id": "file:data/external/the_stack_python/cpython/Lib/test/test_sqlite3/test_userfunctions.py",
    "item_sha256": "17020c6b2147d0d33555bb96dbde6302b9e4859523e8d0bd1a9d73bb0177d11e",
    "category": "code",
    "span_start_byte": 14904,
    "span_byte_count": 8192,
}
EXPECTED_RUNTIME = {
    "torch": "2.6.0+cu124",
    "transformers": "4.57.1",
    "tokenizers": "0.22.2",
    "safetensors": "0.8.0",
}
MIN_LAUNCH_RAM_BYTES = 55 * 1024 ** 3
MIN_LAUNCH_DISK_BYTES = 65 * 1024 ** 3
MIN_RUNTIME_RAM_BYTES = 8 * 1024 ** 3
MAX_CHILD_MEMORY_BYTES = 70 * 1024 ** 3
MAX_WALL_SECONDS = 60 * 60
MONITOR_INTERVAL_SECONDS = 5.0


class GateError(RuntimeError):
    """A preregistered safety, provenance, or runtime gate failed."""


@dataclass(frozen=True)
class MonitorOutcome:
    status: str
    exit_code: int | None
    reason: str | None
    elapsed_seconds: float
    samples: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _load_local_module(filename: str, module_name: str) -> Any:
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot load local module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _pinned_environment() -> dict[str, str]:
    """Check package metadata only; this does not import model code or use a network."""
    if os.environ.get("USE_HUB_KERNELS") != "NO":
        raise GateError("USE_HUB_KERNELS must already be NO before any model import")
    observed: dict[str, str] = {}
    for package, expected in EXPECTED_RUNTIME.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise GateError(f"pinned runtime lacks {package}=={expected}") from exc
        observed[package] = actual
        if actual != expected:
            raise GateError(f"pinned runtime requires {package}=={expected}; found {actual}")
    if sys.version_info[:3] != (3, 12, 10):
        raise GateError(f"pinned runtime requires Python 3.12.10; found {sys.version.split()[0]}")
    return observed


def _psutil() -> Any:
    try:
        import psutil
    except ImportError as exc:
        raise GateError("psutil is required in the target .venv for Windows resource monitoring") from exc
    return psutil


def _direct_worker_python(child_env: dict[str, str]) -> str:
    """Bypass the Windows venv redirector so Popen.pid is the real worker.

    Windows ``.venv/Scripts/python.exe`` can spawn the base interpreter as a
    child while retaining a tiny launcher PID.  Monitoring that PID would not
    measure or stop the model.  The base interpreter gets the same venv site
    packages via PYTHONPATH, after any pinned isolated package target.
    """
    if os.name != "nt" or sys.prefix == sys.base_prefix:
        return sys.executable
    base = Path(sys._base_executable).resolve()
    site = Path(sys.prefix) / "Lib" / "site-packages"
    if not base.is_file() or not site.is_dir():
        raise GateError("cannot resolve direct base Python and venv site-packages for monitored worker")
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(part for part in (existing, str(site)) if part)
    return str(base)


def _launch_preflight(snapshot: Path | None) -> tuple[Any, dict[str, Any]]:
    """Run only existing local metadata checks plus parent resource gates."""
    versions = _pinned_environment()
    teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_smoke_teacher")
    report = teacher.preflight(snapshot=snapshot)
    status = report.source_status
    if not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO":
        raise GateError("local teacher metadata does not match the pinned offline runtime")
    if not status.get("shards_present"):
        raise GateError("all eleven local teacher shards must be present; no download is attempted")
    psutil = _psutil()
    available_ram = int(psutil.virtual_memory().available)
    available_disk = int(psutil.disk_usage(str(report.snapshot)).free)
    if available_ram < MIN_LAUNCH_RAM_BYTES:
        raise GateError(f"launch requires >=55 GiB available physical RAM; observed {available_ram} bytes")
    if available_disk < MIN_LAUNCH_DISK_BYTES:
        raise GateError(f"launch requires >=65 GiB free on the snapshot volume; observed {available_disk} bytes")
    return report, {
        "utc_timestamp": _utc_now(),
        "runtime_versions": versions,
        "python": sys.version.split()[0],
        "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
        "offline_policy": {
            "hub_download_api": "not used",
            "model_loader": "local mmap loader only",
            "tokenizer_local_files_only": True,
        },
        "snapshot": str(report.snapshot),
        "metadata": dict(status),
        "available_physical_ram_bytes": available_ram,
        "snapshot_volume_free_bytes": available_disk,
    }


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise GateError(f"write-once output already exists: {path}") from exc


def _prepare_output(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "supervisor_manifest.json",
        "log": output_dir / "supervisor_log.jsonl",
        "worker": output_dir / "worker_result.json",
        "result": output_dir / "supervisor_result.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise GateError("refusing to overwrite write-once output: " + "; ".join(existing))
    try:
        with paths["log"].open("x", encoding="utf-8", newline="\n"):
            pass
    except FileExistsError as exc:  # defensive against a concurrent creator.
        raise GateError(f"write-once log already exists: {paths['log']}") from exc
    return paths


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    record = {"utc_timestamp": _utc_now(), **event}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _process_sample(process: Any, psutil: Any) -> dict[str, int | None]:
    """Read Windows working set/private commit and page-fault counters once."""
    try:
        basic = process.memory_info()
        full = process.memory_full_info()
    except Exception as exc:
        # A monitor that cannot establish the cap must fail closed.  This is
        # deliberately broad because psutil uses platform-specific subclasses.
        raise GateError(f"cannot read child resource sample: {type(exc).__name__}: {exc}") from exc
    private = getattr(full, "private", None)
    if private is None:
        # Windows psutil exposes ``private`` as private committed bytes.  A VMS
        # fallback would not prove the requested cap, so it is intentionally refused.
        raise GateError("psutil does not expose Windows private commit for the child")
    return {
        "child_working_set_bytes": int(basic.rss),
        "child_private_commit_bytes": int(private),
        "child_pagefile_bytes": _optional_int(getattr(basic, "pagefile", None)),
        "child_peak_working_set_bytes": _optional_int(getattr(basic, "peak_wset", None)),
        "child_peak_pagefile_bytes": _optional_int(getattr(basic, "peak_pagefile", None)),
        # This is diagnostic only; no safety decision is made from it.
        "child_page_faults": _optional_int(getattr(basic, "num_page_faults", None)),
    }


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _terminate_only_child(process: Any) -> int | None:
    """Terminate exactly the Popen child, never a process group or process tree."""
    if process.poll() is not None:
        return process.poll()
    try:
        process.terminate()
    except OSError:
        return process.poll()
    try:
        return process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            return process.wait(timeout=20)
        except OSError:
            return process.poll()


def _monitor_child(
    process: Any,
    *,
    psutil: Any,
    on_sample: Callable[[Mapping[str, Any]], None],
    resource_process: Any | None = None,
    expected_executable: Path | None = None,
    now: Callable[[], float] = time.monotonic,
    wall_limit_seconds: float = MAX_WALL_SECONDS,
    interval_seconds: float = MONITOR_INTERVAL_SECONDS,
) -> MonitorOutcome:
    """Bounded wait loop: an initial sample, then at most one sample per interval."""
    started = now()
    samples = 0
    previous_faults: int | None = None
    previous_elapsed: float | None = None
    executable_checked = False
    while True:
        elapsed = now() - started
        exit_code = process.poll()
        if exit_code is not None:
            return MonitorOutcome("CHILD_EXITED", int(exit_code), None, elapsed, samples)
        if elapsed >= wall_limit_seconds:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "wall_clock_exceeded", elapsed, samples)
        try:
            if resource_process is None:
                resource_process = psutil.Process(process.pid)
            if expected_executable is not None and not executable_checked:
                observed_executable = Path(resource_process.exe()).resolve()
                if observed_executable != expected_executable.resolve():
                    return MonitorOutcome(
                        "VOID_APPARATUS", _terminate_only_child(process),
                        f"monitored_executable_mismatch:{observed_executable}", elapsed, samples,
                    )
                executable_checked = True
            sample = _process_sample(resource_process, psutil)
            available = int(psutil.virtual_memory().available)
        except Exception as exc:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), str(exc), elapsed, samples)
        sample = {"elapsed_seconds": elapsed, "available_physical_ram_bytes": available, **sample}
        faults = sample["child_page_faults"]
        if isinstance(faults, int) and previous_faults is not None and previous_elapsed is not None:
            delta_seconds = elapsed - previous_elapsed
            sample["page_fault_delta"] = faults - previous_faults
            sample["page_faults_per_second"] = (faults - previous_faults) / delta_seconds if delta_seconds > 0 else None
        else:
            sample["page_fault_delta"] = None
            sample["page_faults_per_second"] = None
        on_sample(sample)
        samples += 1
        previous_faults = faults if isinstance(faults, int) else None
        previous_elapsed = elapsed
        if available < MIN_RUNTIME_RAM_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "available_physical_ram_below_8_gib", elapsed, samples)
        if sample["child_working_set_bytes"] > MAX_CHILD_MEMORY_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "child_working_set_above_70_gib", elapsed, samples)
        if sample["child_private_commit_bytes"] > MAX_CHILD_MEMORY_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "child_private_commit_above_70_gib", elapsed, samples)
        remaining = max(0.0, wall_limit_seconds - elapsed)
        try:
            process.wait(timeout=min(interval_seconds, remaining))
        except subprocess.TimeoutExpired:
            pass


def _monitor_or_terminate(process: Any, **kwargs: Any) -> MonitorOutcome:
    """Never leave the direct child running if supervision itself raises."""
    try:
        return _monitor_child(process, **kwargs)
    except BaseException:
        _terminate_only_child(process)
        raise


def _read_row_zero(calib_path: Path, manifest_path: Path, audit: Any) -> dict[str, Any]:
    """Hash both pinned corpus files before reading and validate the fixed row."""
    if _sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise GateError("corpus manifest SHA-256 differs from the preregistered pin")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot parse pinned corpus manifest: {exc}") from exc
    parts = manifest.get("parts") if isinstance(manifest, dict) else None
    calib_part = parts.get("calib") if isinstance(parts, dict) else None
    if not isinstance(calib_part, dict) or calib_part.get("jsonl_sha256") != EXPECTED_CALIB_SHA256:
        raise GateError("pinned corpus manifest does not bind calib.jsonl to the preregistered SHA-256")
    # This hash is intentionally completed before opening/parsing the JSONL row.
    if _sha256_file(calib_path) != EXPECTED_CALIB_SHA256:
        raise GateError("calib.jsonl SHA-256 differs from the preregistered pin")
    try:
        with calib_path.open("rb") as handle:
            line = handle.readline()
        row = json.loads(line)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot parse preregistered calibration row zero: {exc}") from exc
    if not isinstance(row, dict):
        raise GateError("preregistered calibration row zero is not a JSON object")
    for field, expected in PREREGISTERED_ROW.items():
        if row.get(field) != expected:
            raise GateError(f"preregistered calibration row zero {field} differs from its fixed identity")
    if row.get("split") != "calib":
        raise GateError("preregistered calibration row zero is not in the calib split")
    text = row.get("text")
    if not isinstance(text, str) or len(text.encode("utf-8")) != PREREGISTERED_ROW["span_byte_count"]:
        raise GateError("preregistered calibration row zero text/byte count is invalid")
    unsigned = dict(row)
    observed = unsigned.pop("item_sha256", None)
    if observed != audit.sha(audit.canonical(unsigned)):
        raise GateError("preregistered calibration row zero item SHA-256 is invalid")
    return row


def _verify_exact_tokenizer(audit: Any, snapshot: Path) -> Any:
    observed = {name: _sha256_file(snapshot / name) for name in audit.ASSETS}
    if observed != audit.EXPECTED_ASSET_SHA256:
        raise GateError("pinned tokenizer asset SHA-256 differs from STRAT-02 pin")
    tokenizer, _, transformers_version, tokenizers_version = audit.load_tokenizer(snapshot)
    if transformers_version != "4.57.1" or tokenizers_version != "0.22.2":
        raise GateError("pinned tokenizer runtime versions differ from STRAT-02")
    return tokenizer


def _worker(args: argparse.Namespace) -> int:
    """Child-only real forward; all results are written once and contain no text."""
    result_path = args.result.resolve()
    try:
        _pinned_environment()
        teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_smoke_worker_teacher")
        audit = _load_local_module("strat02_token_audit.py", "_strat02_smoke_worker_audit")
        score = _load_local_module("strat02_score.py", "_strat02_smoke_worker_score")
        report = teacher.preflight(snapshot=args.snapshot)
        verified = teacher.verify_shards(report)  # hashes all 11 local shards; never downloads.
        row = _read_row_zero(args.calib, args.manifest, audit)
        tokenizer = _verify_exact_tokenizer(audit, report.snapshot)
        score.validate_strat02_tokenizer(tokenizer)
        with teacher.load_reference_model(verified, pointer_check="all") as model:
            model.eval()
            # The existing bounded scorer also enters inference_mode itself;
            # this outer guard keeps the smoke's forward explicitly grad-free.
            import torch
            with torch.inference_mode():
                document_score = score.score_document(model, tokenizer, row, chunk_size=score.CHUNK_SIZE)
        bpb = document_score.bits / document_score.bytes
        if not math.isfinite(document_score.bits) or not math.isfinite(bpb):
            raise GateError("non-finite apparatus observation")
        observation = {
            "source_document_id": row["source_document_id"],
            "item_sha256": row["item_sha256"],
            "category": row["category"],
            "bytes": document_score.bytes,
            "tokens": document_score.tokens,
            "bits": document_score.bits,
            "bpb": bpb,
        }
        _write_json_once(result_path, {
            "ok": True,
            "status": "APPARATUS_OBSERVATION",
            "utc_timestamp": _utc_now(),
            "observation": observation,
            "notes": [
                "Fixed calibration row zero only; not selected from outputs.",
                "Apparatus observation only; not a quality gate or a rate claim.",
                "No raw document text is included.",
            ],
        })
        return 0
    except Exception as exc:
        try:
            _write_json_once(result_path, {
                "ok": False,
                "status": "WORKER_ERROR",
                "utc_timestamp": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        except (GateError, OSError):
            pass
        return 1


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _worker_looks_oom(worker_result: Mapping[str, Any] | None) -> bool:
    if not worker_result:
        return False
    detail = " ".join(str(worker_result.get(field, "")) for field in ("error_type", "error")).lower()
    return any(marker in detail for marker in ("outofmemory", "out of memory", "cannot allocate memory", "cuda oom"))


def _run_parent(args: argparse.Namespace) -> int:
    report, preflight = _launch_preflight(args.snapshot)
    paths = _prepare_output(args.output_dir.resolve())
    manifest = {
        "schema": "strat02_bounded_smoke_v1",
        "purpose": "first real teacher forward apparatus-safety smoke; no quality/rate conclusion",
        "preregistered_calibration": {
            "jsonl_row_zero_index": 0,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "calib_jsonl_sha256": EXPECTED_CALIB_SHA256,
            **PREREGISTERED_ROW,
            "selection": "fixed before execution; never selected from outputs",
            "raw_text_reported": False,
        },
        "limits": {
            "launch_available_physical_ram_bytes_min": MIN_LAUNCH_RAM_BYTES,
            "launch_snapshot_disk_free_bytes_min": MIN_LAUNCH_DISK_BYTES,
            "runtime_available_physical_ram_bytes_min": MIN_RUNTIME_RAM_BYTES,
            "child_working_set_bytes_max": MAX_CHILD_MEMORY_BYTES,
            "child_private_commit_bytes_max": MAX_CHILD_MEMORY_BYTES,
            "wall_seconds_max": MAX_WALL_SECONDS,
            "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
            "page_faults": "diagnostic only; never a stop threshold",
        },
        "parent_preflight": preflight,
        "child_environment_additions": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    }
    _write_json_once(paths["manifest"], manifest)
    _append_log(paths["log"], {"event": "parent_preflight_passed"})
    # Start from the user's environment, adding only standard offline guards.
    # The parent has already rejected an unsafe USE_HUB_KERNELS value.
    child_env = os.environ.copy()
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["TRANSFORMERS_OFFLINE"] = "1"
    worker_python = _direct_worker_python(child_env)
    command = [
        worker_python, str(Path(__file__).resolve()), "--worker",
        "--snapshot", str(report.snapshot), "--manifest", str(args.manifest.resolve()),
        "--calib", str(args.calib.resolve()), "--result", str(paths["worker"]),
    ]
    worker_token = secrets.token_urlsafe(32)
    child_env["STRAT02_BOUNDED_SMOKE_PARENT_TOKEN"] = worker_token
    command.extend(("--worker-token", worker_token))
    _append_log(paths["log"], {"event": "child_launch", "python": worker_python, "termination_scope": "direct_worker_pid_only"})
    psutil = _psutil()
    process = subprocess.Popen(
        command,
        cwd=str(HERE),
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    outcome = _monitor_or_terminate(
        process,
        psutil=psutil,
        on_sample=lambda sample: _append_log(paths["log"], {"event": "resource_sample", **sample}),
        expected_executable=Path(worker_python),
    )
    _append_log(paths["log"], {"event": "monitor_finished", **asdict(outcome)})
    worker_result = _load_json_object(paths["worker"])
    if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
        status = outcome.status
    elif outcome.exit_code != 0 and _worker_looks_oom(worker_result):
        status = "VOID_RESOURCE"
    elif outcome.exit_code != 0:
        status = "WORKER_EXIT_ERROR"
    elif not worker_result:
        status = "WORKER_PROTOCOL_ERROR"
    elif worker_result.get("ok") is True and worker_result.get("status") == "APPARATUS_OBSERVATION":
        status = "APPARATUS_OBSERVATION"
    else:
        status = "WORKER_ERROR"
    final = {
        "ok": status == "APPARATUS_OBSERVATION",
        "status": status,
        "utc_timestamp": _utc_now(),
        "child_pid": process.pid,
        "monitor": asdict(outcome),
        "worker_result": worker_result,
        "notes": [
            "No quality conclusion or throughput/rate claim is produced by this smoke.",
            "Page faults are recorded only as diagnostics.",
            "Only the direct child PID may have been terminated for a resource breach.",
        ],
    }
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": final["ok"], "status": status, "result": str(paths["result"])}, sort_keys=True))
    return 0 if final["ok"] else 1


def _selftest() -> None:
    """Tiny fake-process controls for exit, timeout, and resource-breach gates."""
    class Clock:
        value = 0.0

        def now(self) -> float:
            return self.value

    class FakePsutil:
        class NoSuchProcess(Exception):
            pass

        def __init__(self, available: int) -> None:
            self.available = available

        def virtual_memory(self) -> Any:
            return type("Memory", (), {"available": self.available})()

    class FakeProcess:
        def __init__(self, clock: Clock, exit_code: int | None = None, memory: int = 1) -> None:
            self.clock, self.returncode, self.memory = clock, exit_code, memory
            self.terminated = False
            self.waits: list[float] = []

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            if self.returncode is not None:
                return self.returncode
            assert timeout is not None
            self.waits.append(timeout)
            self.clock.value += timeout
            raise subprocess.TimeoutExpired("fake", timeout)

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:  # pragma: no cover - fake termination always completes.
            self.returncode = -9

        def memory_info(self) -> Any:
            return type("MemoryInfo", (), {"rss": self.memory, "pagefile": self.memory, "peak_wset": self.memory, "peak_pagefile": self.memory, "num_page_faults": 2})()

        def memory_full_info(self) -> Any:
            return type("MemoryFull", (), {"private": self.memory})()

    clock = Clock()
    psutil = FakePsutil(MIN_RUNTIME_RAM_BYTES)
    exited = FakeProcess(clock, exit_code=7)
    outcome = _monitor_child(exited, psutil=psutil, on_sample=lambda _: None, resource_process=exited, now=clock.now)
    assert outcome.status == "CHILD_EXITED" and outcome.exit_code == 7 and not exited.terminated

    clock = Clock()
    low_ram = FakePsutil(MIN_RUNTIME_RAM_BYTES - 1)
    breach = FakeProcess(clock)
    outcome = _monitor_child(breach, psutil=low_ram, on_sample=lambda _: None, resource_process=breach, now=clock.now)
    assert outcome.status == "VOID_RESOURCE" and outcome.reason == "available_physical_ram_below_8_gib" and breach.terminated

    clock = Clock()
    timeout = FakeProcess(clock)
    outcome = _monitor_child(timeout, psutil=psutil, on_sample=lambda _: None, resource_process=timeout, now=clock.now, wall_limit_seconds=10, interval_seconds=5)
    assert outcome.status == "VOID_RESOURCE" and outcome.reason == "wall_clock_exceeded" and timeout.terminated
    assert timeout.waits == [5, 5]
    orphan_guard = FakeProcess(Clock())
    try:
        _monitor_or_terminate(
            orphan_guard,
            psutil=psutil,
            on_sample=lambda _: (_ for _ in ()).throw(OSError("planted log failure")),
            resource_process=orphan_guard,
            now=orphan_guard.clock.now,
        )
    except OSError:
        assert orphan_guard.terminated
    else:
        raise AssertionError("planted log failure did not reach orphan guard")
    wrong_exe = FakeProcess(Clock())
    wrong_exe.exe = lambda: str(Path(sys.executable).with_name("not-the-worker.exe"))
    mismatched = _monitor_child(
        wrong_exe, psutil=psutil, on_sample=lambda _: None,
        resource_process=wrong_exe, expected_executable=Path(sys.executable),
        now=wrong_exe.clock.now,
    )
    assert mismatched.status == "VOID_APPARATUS" and wrong_exe.terminated
    assert _worker_looks_oom({"error_type": "OutOfMemoryError", "error": "allocator failed"})
    assert not _worker_looks_oom({"error_type": "ValueError", "error": "bad metadata"})

    # Real Popen has no memory_info; exercise the psutil.Process(pid) adapter.
    import psutil as real_psutil
    tiny_env = os.environ.copy()
    tiny_python = _direct_worker_python(tiny_env)
    tiny = subprocess.Popen(
        [tiny_python, "-c", "import time; time.sleep(0.4)"],
        env=tiny_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert Path(real_psutil.Process(tiny.pid).exe()).resolve() == Path(tiny_python).resolve()
        real_samples: list[Mapping[str, Any]] = []
        real_outcome = _monitor_or_terminate(
            tiny, psutil=real_psutil, on_sample=real_samples.append,
            expected_executable=Path(tiny_python),
            interval_seconds=1.0, wall_limit_seconds=10.0,
        )
        assert real_outcome.status == "CHILD_EXITED" and real_outcome.exit_code == 0
        assert real_samples and real_samples[0]["child_private_commit_bytes"] > 0
    finally:
        _terminate_only_child(tiny)

    with tempfile.TemporaryDirectory(prefix="strat02-bounded-smoke-selftest-") as temporary:
        target = Path(temporary) / "once.json"
        _write_json_once(target, {"ok": True})
        try:
            _write_json_once(target, {"ok": False})
        except GateError:
            pass
        else:  # pragma: no cover - explicit write-once control.
            raise AssertionError("write-once output was overwritten")
    print(json.dumps({"ok": True, "selftest": True}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="perform the one real child-worker smoke")
    action.add_argument("--selftest", action="store_true", help="run tiny fake-worker/mock-monitor tests only")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help="new or empty directory for write-once run outputs")
    parser.add_argument("--snapshot", type=Path, help="pinned local snapshot directory; never a Hub identifier")
    parser.add_argument("--manifest", type=Path, default=CORPUS / "manifest.json")
    parser.add_argument("--calib", type=Path, default=CORPUS / "calib.jsonl")
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.worker:
            if args.result is None:
                raise GateError("internal worker requires --result")
            inherited_token = os.environ.get("STRAT02_BOUNDED_SMOKE_PARENT_TOKEN", "")
            if not args.worker_token or not secrets.compare_digest(args.worker_token, inherited_token):
                raise GateError("internal worker may run only when launched by its --run parent")
            return _worker(args)
        if args.output_dir is None:
            raise GateError("--run requires an explicit --output-dir; refusing implicit writes")
        return _run_parent(args)
    except (GateError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
