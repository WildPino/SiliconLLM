#!/usr/bin/env python3
"""Bounded, read-only STRAT-02 W4-v1 format-underflow census.

This apparatus implements the preregistered census in
``BRIEF_STRAT_02_W4_FORMAT_CENSUS.md``.  It is deliberately *not* a W4
conversion, donor forward, quality evaluation, rate benchmark, download, or
T4 job.  The real worker verifies each already-local shard once, then scans
all pinned ``nn.Linear`` weights through safetensors row slices no larger than
256 rows.  It writes only counts and invalid group coordinates: never a raw
weight, packed code, corpus item, token, or model output.

Modes::

    python strat02_w4_format_census.py --selftest
    python strat02_w4_format_census.py --preflight [--snapshot PATH]
    python strat02_w4_format_census.py --run --output-dir NEW_PATH [--snapshot PATH]

``--preflight`` is metadata/meta-model only.  ``--run`` starts one direct base
Python worker and monitors only the PID it created.  Interrupted or failed
runs preserve append-only partial files and are reported as ``INCOMPLETE``;
there is intentionally no resume mechanism.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
WORKER_TOKEN_ENV = "STRAT02_W4_FORMAT_CENSUS_WORKER_TOKEN"
SCHEMA = "strat02_w4_format_census_v1"
GROUP_SIZE = 128
MAX_ROWS_PER_SLICE = 256
EXPECTED_KEY_COUNT = 6259
EXPECTED_LINEAR_COUNT = 6225
EXPECTED_LINEAR_WEIGHTS = 13_363_052_544
EXPECTED_LINEAR_GROUPS = 104_398_848
POSITIVE_CONTROL_KEYS = frozenset(
    f"model.layers.{layer}.mlp.gate.weight" for layer in range(16)
)
POSITIVE_CONTROL_ROW = 127
POSITIVE_CONTROL_EXPECTED_GROUPS = 256
MIN_LAUNCH_RAM_BYTES = 8 * 1024 ** 3
MIN_LAUNCH_OUTPUT_FREE_BYTES = 1 * 1024 ** 3
MIN_RUNTIME_RAM_BYTES = 4 * 1024 ** 3
MAX_PRIVATE_COMMIT_BYTES = 8 * 1024 ** 3
MAX_WALL_SECONDS = 30 * 60
MONITOR_INTERVAL_SECONDS = 5.0
# This is deliberately lower than the one-GiB launch reservation.  It bounds
# pathological all-invalid checkpoints before their coordinate stream can
# consume the run volume, and fails closed rather than omitting coordinates.
MAX_INVALID_COORDINATE_BYTES = 512 * 1024 ** 2
BRIEF_PATH = HERE.parents[2] / "docs" / "research" / "donor_adaptation" / "briefs" / "BRIEF_STRAT_02_W4_FORMAT_CENSUS.md"
SOURCE_PATHS = {
    "runner": Path(__file__).resolve(),
    "brief": BRIEF_PATH,
    "teacher_loader": HERE / "strat02_mmap_teacher.py",
    "bounded_supervisor": HERE / "strat02_bounded_smoke.py",
    "w4_codec": HERE / "strat02_weight_codec.py",
}


class GateError(RuntimeError):
    """A pinned provenance, resource, or census invariant failed."""


@dataclass(frozen=True)
class MonitorOutcome:
    status: str
    exit_code: int | None
    reason: str | None
    elapsed_seconds: float
    samples: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
    except FileExistsError as exc:
        raise GateError(f"write-once output already exists: {path}") from exc


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> int:
    line = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    handle.write(line)
    handle.flush()
    return len(line.encode("utf-8"))


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        _append_jsonl(handle, {"utc_timestamp": _utc_now(), **event})


def _prepare_output(output_dir: Path) -> dict[str, Path]:
    root = output_dir.resolve()
    if root.exists():
        raise GateError(f"--output-dir must be a fresh path, not an existing directory: {root}")
    root.mkdir(parents=True, exist_ok=False)
    paths = {
        "root": root,
        "manifest": root / "census_manifest.json",
        "log": root / "supervisor_log.jsonl",
        "counts": root / "tensor_counts.jsonl",
        "invalid": root / "invalid_groups.jsonl",
        "worker": root / "worker_result.json",
        "result": root / "supervisor_result.json",
        "stdout": root / "worker_stdout.log",
        "stderr": root / "worker_stderr.log",
    }
    with paths["log"].open("x", encoding="utf-8", newline="\n"):
        pass
    return paths


def _psutil() -> Any:
    try:
        import psutil
    except ImportError as exc:
        raise GateError("psutil is required for Windows resource monitoring") from exc
    return psutil


def _inventory_linears(teacher: Any, report: Any) -> dict[str, Any]:
    """Inventory exactly the pinned meta model without opening a shard payload."""
    teacher._require_target_environment()
    import torch

    EmoConfig, EmoForCausalLM = teacher._import_pinned_emo(report.snapshot)
    config = teacher._read_json(report.snapshot / "config.json", "pinned model config")
    with torch.device("meta"):
        model = EmoForCausalLM(EmoConfig(**config))
    index_keys = set(report.index["weight_map"])
    state = model.state_dict()
    state_keys = set(state)
    if len(index_keys) != EXPECTED_KEY_COUNT:
        raise GateError(f"pinned index has {len(index_keys)} keys, expected {EXPECTED_KEY_COUNT}")
    if state_keys != index_keys:
        missing, unexpected = sorted(index_keys - state_keys), sorted(state_keys - index_keys)
        raise GateError(f"meta model/index keyset mismatch; missing={missing[:5]} unexpected={unexpected[:5]}")

    linear_keys: list[str] = []
    linear_weights = 0
    linear_groups = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not module_name or module.weight is None:
            raise GateError(f"invalid nn.Linear inventory entry: {module_name!r}")
        key = module_name + ".weight"
        shape = tuple(int(value) for value in module.weight.shape)
        if key not in index_keys or tuple(int(value) for value in state[key].shape) != shape:
            raise GateError(f"linear inventory/index mismatch for {key}")
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 or shape[1] % GROUP_SIZE:
            raise GateError(f"linear W4-g128 shape is invalid for {key}: {shape}")
        linear_keys.append(key)
        linear_weights += int(module.weight.numel())
        linear_groups += shape[0] * (shape[1] // GROUP_SIZE)

    sorted_keys = sorted(linear_keys)
    if len(set(linear_keys)) != len(linear_keys):
        raise GateError("meta model exposes duplicate nn.Linear weight keys")
    if len(sorted_keys) != EXPECTED_LINEAR_COUNT or linear_weights != EXPECTED_LINEAR_WEIGHTS:
        raise GateError(
            f"unexpected meta linear inventory: {len(sorted_keys)} matrices/{linear_weights} weights "
            f"!= {EXPECTED_LINEAR_COUNT}/{EXPECTED_LINEAR_WEIGHTS}"
        )
    if linear_groups != EXPECTED_LINEAR_GROUPS:
        raise GateError(f"unexpected g128 group total: {linear_groups} != {EXPECTED_LINEAR_GROUPS}")
    return {
        "index_key_count": len(index_keys),
        "meta_model_key_count": len(state_keys),
        "linear_count": len(sorted_keys),
        "linear_weight_count": linear_weights,
        "linear_group_count": linear_groups,
        "linear_keys_sha256": hashlib.sha256("\n".join(sorted_keys).encode("utf-8")).hexdigest(),
        "linear_keys": sorted_keys,
    }


def _metadata_preflight(snapshot: Path | None) -> tuple[Any, Any, dict[str, Any]]:
    """Pinned local metadata and meta-model checks; no safetensors payload read."""
    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_census_bounded")
    teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_census_teacher")
    versions = bounded._pinned_environment()
    report = teacher.preflight(snapshot=snapshot)
    status = report.source_status
    if not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO":
        raise GateError("pinned local model code/runtime proof failed")
    if not status.get("shards_present"):
        raise GateError("all 11 pinned local shards must be present; no download is attempted")
    inventory = _inventory_linears(teacher, report)
    return bounded, teacher, {
        "utc_timestamp": _utc_now(),
        "runtime_versions": versions,
        "python": sys.version.split()[0],
        "snapshot": str(report.snapshot),
        "revision": report.source_status.get("revision"),
        "metadata": dict(status),
        "inventory": {key: value for key, value in inventory.items() if key != "linear_keys"},
        "apparatus_sha256": {name: _sha256_file(path) for name, path in SOURCE_PATHS.items()},
        "offline_policy": {
            "hub_download_api": "not used",
            "weight_payload_access": "not used by --preflight",
            "heldout_or_corpus": "not used",
            "donor_forward": "not used",
            "t4": "not used",
        },
    }


def _launch_preflight(snapshot: Path | None, output_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    bounded, teacher, preflight = _metadata_preflight(snapshot)
    psutil = _psutil()
    available = int(psutil.virtual_memory().available)
    if available < MIN_LAUNCH_RAM_BYTES:
        raise GateError(f"launch requires >=8 GiB available physical RAM; observed {available} bytes")
    free = int(psutil.disk_usage(str(output_dir.resolve())).free)
    if free < MIN_LAUNCH_OUTPUT_FREE_BYTES:
        raise GateError(f"launch requires >=1 GiB free on output volume; observed {free} bytes")
    return bounded, teacher, {
        **preflight,
        "available_physical_ram_bytes": available,
        "output_volume_free_bytes": free,
    }


def _endpoint_scale(absmax: np.float32, c: np.float32) -> np.float16:
    """The exact W4-v1 candidate scale arithmetic before its F16 cast."""
    # Overflow is the observation being censused, not a warning-worthy error.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.float16(np.float32(absmax) * np.float32(c) / np.float32(7))


def _classify_group(values: np.ndarray) -> str | None:
    """Return the census invalid reason, or ``None`` for a valid g128 group."""
    if values.shape != (GROUP_SIZE,) or values.dtype != np.float32:
        raise ValueError("census group must be exactly 128 float32 values")
    if not bool(np.isfinite(values).all()):
        return "source_nonfinite"
    absmax = np.max(np.abs(values), initial=np.float32(0))
    if absmax == 0:
        return None
    if _endpoint_scale(absmax, np.float32(0.50)) == 0:
        return "scale_underflow_c_0.50"
    if not bool(np.isfinite(_endpoint_scale(absmax, np.float32(1.00)))):
        return "scale_nonfinite_c_1.00"
    return None


def _classify_tile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized finite-F32 g128 census for a ``[rows, groups, 128]`` tile."""
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[2] != GROUP_SIZE:
        raise ValueError("census tile must have dtype float32 and shape [rows, groups, 128]")
    finite = np.isfinite(values).all(axis=2)
    # abs is applied only to finite groups: it prevents NaN propagation from
    # obscuring their separately recorded source_nonfinite reason.
    absmax = np.zeros(values.shape[:2], dtype=np.float32)
    if bool(finite.any()):
        absmax[finite] = np.max(np.abs(values[finite]), axis=1)
    true_zero = finite & (absmax == 0)
    nonzero = finite & ~true_zero
    scale_low = np.zeros_like(absmax, dtype=np.float16)
    scale_high = np.zeros_like(absmax, dtype=np.float16)
    if bool(nonzero.any()):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale_low[nonzero] = np.float16(np.float32(absmax[nonzero]) * np.float32(0.50) / np.float32(7))
            scale_high[nonzero] = np.float16(np.float32(absmax[nonzero]) * np.float32(1.00) / np.float32(7))
    underflow = nonzero & (scale_low == 0)
    overflow = nonzero & ~np.isfinite(scale_high)
    return finite, true_zero, underflow, overflow


def _all_candidate_invalid(values: np.ndarray, candidates: Iterable[np.float32]) -> bool:
    """Synthetic-only reference loop over all frozen W4-v1 candidate factors."""
    if values.shape != (GROUP_SIZE,) or values.dtype != np.float32:
        raise ValueError("reference group must be exactly 128 float32 values")
    if not bool(np.isfinite(values).all()):
        return True
    absmax = np.max(np.abs(values), initial=np.float32(0))
    if absmax == 0:
        return False
    for c in candidates:
        scale = _endpoint_scale(absmax, np.float32(c))
        if scale == 0 or not bool(np.isfinite(scale)):
            return True
    return False


def _validate_endpoint_shortcut(codec: Any) -> None:
    """Prove planted synthetic endpoint cases agree with the all-11 v1 loop."""
    tiny = np.zeros(GROUP_SIZE, dtype=np.float32)
    tiny[0] = np.nextafter(np.float32(0), np.float32(1))
    normal = np.zeros(GROUP_SIZE, dtype=np.float32)
    normal[0] = np.float32(1.0)
    huge = np.zeros(GROUP_SIZE, dtype=np.float32)
    huge[0] = np.float32(np.finfo(np.float32).max)
    nonfinite = normal.copy()
    nonfinite[3] = np.float32(np.nan)
    for group, expected in ((tiny, "scale_underflow_c_0.50"), (normal, None),
                            (huge, "scale_nonfinite_c_1.00"), (nonfinite, "source_nonfinite")):
        observed = _classify_group(group)
        if observed != expected:
            raise AssertionError(f"planted endpoint classification differs: {observed!r} != {expected!r}")
        if (observed is not None) != _all_candidate_invalid(group, codec.W4_CANDIDATE_C):
            raise AssertionError("endpoint shortcut disagrees with all-candidate v1 loop")


def _open_append_only(path: Path) -> Any:
    try:
        return path.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise GateError(f"append-only output already exists: {path}") from exc


def _emit_invalids(handle: Any, *, key: str, row_offset: int, finite: np.ndarray,
                   underflow: np.ndarray, overflow: np.ndarray, current_bytes: int) -> int:
    for mask, reason in ((~finite, "source_nonfinite"), (underflow, "scale_underflow_c_0.50"),
                         (overflow, "scale_nonfinite_c_1.00")):
        for local_row, group in np.argwhere(mask):
            record = {"tensor": key, "row": int(row_offset + int(local_row)), "group": int(group), "reason": reason}
            encoded = (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            if current_bytes + len(encoded) > MAX_INVALID_COORDINATE_BYTES:
                raise GateError(
                    f"invalid coordinate output would exceed declared cap {MAX_INVALID_COORDINATE_BYTES} bytes; refusing truncation"
                )
            handle.write(encoded.decode("utf-8"))
            handle.flush()
            current_bytes += len(encoded)
    return current_bytes


def _positive_control_counts(key: str, row_start: int, finite: np.ndarray,
                             underflow: np.ndarray, overflow: np.ndarray) -> dict[str, int]:
    """Count, but never exempt, the preregistered router-row control groups."""
    if key not in POSITIVE_CONTROL_KEYS or not row_start <= POSITIVE_CONTROL_ROW < row_start + finite.shape[0]:
        return {"groups": 0, "underflow": 0, "other_invalid": 0}
    local_row = POSITIVE_CONTROL_ROW - row_start
    groups = int(finite.shape[1])
    underflow_count = int(np.count_nonzero(underflow[local_row]))
    other_invalid = int(np.count_nonzero(~finite[local_row])) + int(np.count_nonzero(overflow[local_row]))
    return {"groups": groups, "underflow": underflow_count, "other_invalid": other_invalid}


def _scan_linear_tensor(handle: Any, *, teacher: Any, verified: Any, report: Any, key: str,
                        invalid_handle: Any, invalid_bytes: int) -> tuple[dict[str, Any], int]:
    """Scan one tensor through safe_open/get_slice and bounded row slices only."""
    from safetensors import safe_open
    import torch

    shard = report.index["weight_map"].get(key)
    if not isinstance(shard, str):
        raise GateError(f"linear key is absent from pinned index: {key}")
    teacher._assert_unchanged(verified)
    totals = {"groups": 0, "true_zero": 0, "underflow": 0, "scale_nonfinite": 0, "source_nonfinite": 0,
              "positive_control_groups": 0, "positive_control_underflow": 0, "positive_control_other_invalid": 0}
    with safe_open(str(report.snapshot / shard), framework="pt", device="cpu") as archive:
        tensor_slice = archive.get_slice(key)
        shape = tuple(int(value) for value in tensor_slice.get_shape())
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 or shape[1] % GROUP_SIZE:
            raise GateError(f"linear tensor violates W4-g128 matrix contract: {key} {shape}")
        rows, columns = shape
        groups_per_row = columns // GROUP_SIZE
        for row_start in range(0, rows, MAX_ROWS_PER_SLICE):
            row_stop = min(rows, row_start + MAX_ROWS_PER_SLICE)
            source = tensor_slice[row_start:row_stop]
            if source.dtype != torch.float32 or source.device.type != "cpu" or source.ndim != 2:
                raise GateError(f"safetensors slice is not CPU F32 matrix: {key}")
            if tuple(int(value) for value in source.shape) != (row_stop - row_start, columns):
                raise GateError(f"safetensors slice shape mismatch: {key}")
            array = source.numpy()
            if array.dtype != np.float32 or array.shape != (row_stop - row_start, columns):
                raise GateError(f"cannot obtain bounded F32 view: {key}")
            finite, true_zero, underflow, overflow = _classify_tile(array.reshape(row_stop - row_start, groups_per_row, GROUP_SIZE))
            control = _positive_control_counts(key, row_start, finite, underflow, overflow)
            invalid_bytes = _emit_invalids(
                invalid_handle, key=key, row_offset=row_start, finite=finite, underflow=underflow,
                overflow=overflow, current_bytes=invalid_bytes,
            )
            totals["groups"] += int(finite.size)
            totals["true_zero"] += int(np.count_nonzero(true_zero))
            totals["underflow"] += int(np.count_nonzero(underflow))
            totals["scale_nonfinite"] += int(np.count_nonzero(overflow))
            totals["source_nonfinite"] += int(np.count_nonzero(~finite))
            totals["positive_control_groups"] += control["groups"]
            totals["positive_control_underflow"] += control["underflow"]
            totals["positive_control_other_invalid"] += control["other_invalid"]
    expected_groups = rows * groups_per_row
    if totals["groups"] != expected_groups:
        raise GateError(f"tensor group accounting mismatch for {key}")
    invalid = totals["underflow"] + totals["scale_nonfinite"] + totals["source_nonfinite"]
    record = {
        "tensor": key,
        "source_shard": shard,
        "shape": [rows, columns],
        "groups": totals["groups"],
        "true_zero_groups": totals["true_zero"],
        "underflow_groups": totals["underflow"],
        "scale_nonfinite_groups": totals["scale_nonfinite"],
        "source_nonfinite_groups": totals["source_nonfinite"],
        "invalid_groups": invalid,
        "positive_control_groups": totals["positive_control_groups"],
        "positive_control_underflow_groups": totals["positive_control_underflow"],
        "positive_control_other_invalid_groups": totals["positive_control_other_invalid"],
    }
    _append_jsonl(handle, record)
    return record, invalid_bytes


def _worker(args: argparse.Namespace) -> int:
    result_path = args.result.resolve()
    completed_tensors = 0
    aggregate = {"groups": 0, "true_zero_groups": 0, "underflow_groups": 0,
                 "scale_nonfinite_groups": 0, "source_nonfinite_groups": 0, "invalid_groups": 0,
                 "positive_control_groups": 0, "positive_control_underflow_groups": 0,
                 "positive_control_other_invalid_groups": 0}
    try:
        parent_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected_hashes = parent_manifest.get("parent_preflight", {}).get("apparatus_sha256")
        actual_hashes = {name: _sha256_file(path) for name, path in SOURCE_PATHS.items()}
        if expected_hashes != actual_hashes:
            raise GateError("runner/brief/loader/supervisor/codec changed after parent preflight")
        teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_census_worker_teacher")
        codec = _load_local_module("strat02_weight_codec.py", "_strat02_census_worker_codec")
        report = teacher.preflight(snapshot=args.snapshot)
        inventory = _inventory_linears(teacher, report)
        _validate_endpoint_shortcut(codec)
        # Exactly one full 11-shard SHA pass, before the first tensor is opened.
        verified = teacher.verify_shards(report)
        keys = inventory["linear_keys"]
        invalid_bytes = 0
        with _open_append_only(args.counts) as counts_handle, _open_append_only(args.invalid) as invalid_handle:
            for key in keys:
                record, invalid_bytes = _scan_linear_tensor(
                    counts_handle, teacher=teacher, verified=verified, report=report, key=key,
                    invalid_handle=invalid_handle, invalid_bytes=invalid_bytes,
                )
                completed_tensors += 1
                aggregate["groups"] += int(record["groups"])
                aggregate["true_zero_groups"] += int(record["true_zero_groups"])
                aggregate["underflow_groups"] += int(record["underflow_groups"])
                aggregate["scale_nonfinite_groups"] += int(record["scale_nonfinite_groups"])
                aggregate["source_nonfinite_groups"] += int(record["source_nonfinite_groups"])
                aggregate["invalid_groups"] += int(record["invalid_groups"])
                aggregate["positive_control_groups"] += int(record["positive_control_groups"])
                aggregate["positive_control_underflow_groups"] += int(record["positive_control_underflow_groups"])
                aggregate["positive_control_other_invalid_groups"] += int(record["positive_control_other_invalid_groups"])
                _append_log(args.log, {"event": "tensor_complete", "tensor": key, "completed_tensors": completed_tensors})
        if completed_tensors != EXPECTED_LINEAR_COUNT or aggregate["groups"] != EXPECTED_LINEAR_GROUPS:
            raise GateError("census did not account for the complete frozen linear inventory")
        if (aggregate["positive_control_groups"] != POSITIVE_CONTROL_EXPECTED_GROUPS
                or aggregate["positive_control_underflow_groups"] != POSITIVE_CONTROL_EXPECTED_GROUPS
                or aggregate["positive_control_other_invalid_groups"] != 0):
            raise GateError("preregistered 256-group router shared-row positive control did not reproduce")
        _write_json_once(result_path, {
            "ok": True,
            "status": "COMPLETE",
            "utc_timestamp": _utc_now(),
            "completed_tensors": completed_tensors,
            "aggregate": aggregate,
            "tensor_counts_sha256": _sha256_file(args.counts),
            "invalid_groups_sha256": _sha256_file(args.invalid),
            "invalid_groups_bytes": invalid_bytes,
            "inventory": {key: value for key, value in inventory.items() if key != "linear_keys"},
            "notes": [
                "Read-only W4-v1 format census; no candidate conversion, inference, heldout, corpus, download, or T4 use.",
                "All invalid coordinates were appended with flush; a declared cap fails closed rather than truncating them.",
            ],
        })
        return 0
    except Exception as exc:
        try:
            _write_json_once(result_path, {
                "ok": False,
                "status": "INCOMPLETE",
                "utc_timestamp": _utc_now(),
                "completed_tensors": completed_tensors,
                "partial_aggregate": aggregate,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "partial_files_preserved": True,
            })
        except (GateError, OSError):
            pass
        return 1


def _process_sample(process: Any, psutil: Any) -> dict[str, int]:
    try:
        basic, full = process.memory_info(), process.memory_full_info()
    except Exception as exc:
        raise GateError(f"cannot read worker resource sample: {type(exc).__name__}: {exc}") from exc
    private = getattr(full, "private", None)
    if private is None:
        raise GateError("psutil does not expose Windows private commit for the worker")
    return {"child_working_set_bytes": int(basic.rss), "child_private_commit_bytes": int(private)}


def _terminate_only_child(process: Any) -> int | None:
    if process.poll() is not None:
        return process.poll()
    try:
        process.terminate()
        return process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=20)
    except OSError:
        return process.poll()


def _monitor_child(process: Any, *, psutil: Any, on_sample: Callable[[Mapping[str, Any]], None],
                   expected_executable: Path, now: Callable[[], float] = time.monotonic,
                   interval_seconds: float = MONITOR_INTERVAL_SECONDS,
                   wall_limit_seconds: float = MAX_WALL_SECONDS) -> MonitorOutcome:
    started, samples, checked = now(), 0, False
    resource_process: Any | None = None
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
            if not checked:
                if Path(resource_process.exe()).resolve() != expected_executable.resolve():
                    return MonitorOutcome("VOID_APPARATUS", _terminate_only_child(process), "monitored_executable_mismatch", elapsed, samples)
                checked = True
            sample = _process_sample(resource_process, psutil)
            available = int(psutil.virtual_memory().available)
        except Exception as exc:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), str(exc), elapsed, samples)
        sample = {"elapsed_seconds": elapsed, "available_physical_ram_bytes": available, **sample}
        on_sample(sample)
        samples += 1
        if available < MIN_RUNTIME_RAM_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "available_physical_ram_below_4_gib", elapsed, samples)
        if sample["child_private_commit_bytes"] > MAX_PRIVATE_COMMIT_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "child_private_commit_above_8_gib", elapsed, samples)
        try:
            process.wait(timeout=min(interval_seconds, max(0.0, wall_limit_seconds - elapsed)))
        except subprocess.TimeoutExpired:
            pass


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_parent(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    try:
        bounded, _teacher, preflight = _launch_preflight(args.snapshot, paths["root"])
        manifest = {
            "schema": SCHEMA,
            "purpose": "bounded, read-only full W4-v1 scale-validity census; not conversion, quality, rate, inference, corpus, download, or T4",
            "algorithm": {
                "linear_order": "lexicographic safetensors index key order",
                "group_size": GROUP_SIZE,
                "max_rows_per_safetensors_slice": MAX_ROWS_PER_SLICE,
                "endpoints": ["0.50", "1.00"],
                "scale_expression": "float16(float32(absmax) * float32(c) / float32(7))",
                "all_11_candidate_control": "synthetic planted endpoint equivalence only",
                "positive_control": "all 16 g128 groups of router row 127 across layers 0..15; never exempt from scan",
            },
            "limits": {
                "launch_available_physical_ram_bytes_min": MIN_LAUNCH_RAM_BYTES,
                "launch_output_volume_free_bytes_min": MIN_LAUNCH_OUTPUT_FREE_BYTES,
                "runtime_available_physical_ram_bytes_min": MIN_RUNTIME_RAM_BYTES,
                "worker_private_commit_bytes_max": MAX_PRIVATE_COMMIT_BYTES,
                "wall_seconds_max": MAX_WALL_SECONDS,
                "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
                "invalid_coordinate_bytes_max": MAX_INVALID_COORDINATE_BYTES,
                "overflow_policy": "fail_closed_no_coordinate_truncation",
            },
            "parent_preflight": preflight,
            "output": {
                "tensor_counts": paths["counts"].name,
                "invalid_groups": paths["invalid"].name,
                "raw_weights_reported": False,
                "corpus_or_heldout_access": False,
                "auto_resume": False,
            },
        }
        _write_json_once(paths["manifest"], manifest)
        _append_log(paths["log"], {"event": "parent_preflight_passed"})
    except Exception as exc:
        _write_json_once(paths["manifest"], {"schema": SCHEMA, "preflight": "failed"})
        _append_log(paths["log"], {"event": "parent_preflight_failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json_once(paths["result"], {"ok": False, "status": "INCOMPLETE", "failure_classification": "PREFLIGHT_ERROR", "error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps({"ok": False, "status": "INCOMPLETE", "result": str(paths["result"])}, sort_keys=True))
        return 1

    process: Any | None = None
    try:
        child_env = os.environ.copy()
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        worker_python = bounded._direct_worker_python(child_env)
        token = secrets.token_urlsafe(32)
        child_env[WORKER_TOKEN_ENV] = token
        command = [worker_python, str(Path(__file__).resolve()), "--worker", "--snapshot", preflight["snapshot"],
                   "--output-dir", str(paths["root"]), "--manifest", str(paths["manifest"]), "--log", str(paths["log"]),
                   "--counts", str(paths["counts"]), "--invalid", str(paths["invalid"]), "--result", str(paths["worker"]),
                   "--worker-token", token]
        _append_log(paths["log"], {"event": "child_launch", "python": worker_python, "termination_scope": "direct_worker_pid_only"})
        with paths["stdout"].open("x", encoding="utf-8") as stdout, paths["stderr"].open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=str(HERE), env=child_env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            outcome = _monitor_child(process, psutil=_psutil(), expected_executable=Path(worker_python),
                                     on_sample=lambda sample: _append_log(paths["log"], {"event": "resource_sample", **sample}))
        _append_log(paths["log"], {"event": "monitor_finished", **asdict(outcome)})
        worker = _load_json_object(paths["worker"])
        failure = None
        if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
            failure = outcome.status
        elif outcome.exit_code != 0:
            failure = "WORKER_ERROR"
        elif not worker or worker.get("ok") is not True or worker.get("status") != "COMPLETE":
            failure = "WORKER_PROTOCOL_ERROR"
        final = {"ok": failure is None, "status": "COMPLETE" if failure is None else "INCOMPLETE",
                 "failure_classification": failure, "utc_timestamp": _utc_now(), "child_pid": process.pid,
                 "monitor": asdict(outcome), "worker_result": worker}
        if failure is None:
            final["tensor_counts_sha256"] = _sha256_file(paths["counts"])
            final["invalid_groups_sha256"] = _sha256_file(paths["invalid"])
    except Exception as exc:
        if process is not None:
            _terminate_only_child(process)
        _append_log(paths["log"], {"event": "parent_supervision_error", "error_type": type(exc).__name__, "error": str(exc)})
        final = {"ok": False, "status": "INCOMPLETE", "failure_classification": "PARENT_SUPERVISION_ERROR",
                 "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": final["ok"], "status": final["status"], "result": str(paths["result"])}, sort_keys=True))
    return 0 if final["ok"] else 1


def _selftest() -> None:
    """Synthetic-only arithmetic, append-only, and resource-cap controls."""
    codec = _load_local_module("strat02_weight_codec.py", "_strat02_census_selftest_codec")
    _validate_endpoint_shortcut(codec)
    tile = np.zeros((2, 3, GROUP_SIZE), dtype=np.float32)
    tile[0, 1, 0] = np.nextafter(np.float32(0), np.float32(1))
    tile[0, 2, 0] = np.float32(np.finfo(np.float32).max)
    tile[1, 0, 1] = np.float32(np.nan)
    finite, zero, underflow, overflow = _classify_tile(tile)
    assert int(np.count_nonzero(zero)) == 3
    assert int(np.count_nonzero(underflow)) == 1
    assert int(np.count_nonzero(overflow)) == 1
    assert int(np.count_nonzero(~finite)) == 1
    with tempfile.TemporaryDirectory(prefix="strat02-w4-format-census-selftest-") as temporary:
        path = Path(temporary) / "invalid.jsonl"
        with _open_append_only(path) as handle:
            used = _emit_invalids(handle, key="synthetic.weight", row_offset=0, finite=finite,
                                  underflow=underflow, overflow=overflow, current_bytes=0)
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert used == path.stat().st_size and len(records) == 3
        assert {record["reason"] for record in records} == {
            "source_nonfinite", "scale_underflow_c_0.50", "scale_nonfinite_c_1.00",
        }
        try:
            _open_append_only(path)
        except GateError:
            pass
        else:
            raise AssertionError("append-only invalid coordinate file was reopened")
    print(json.dumps({"ok": True, "selftest": True, "donor_weights_opened": False}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selftest", action="store_true", help="synthetic-only; never opens donor weights")
    action.add_argument("--preflight", action="store_true", help="metadata and meta-model only; no safetensors payload")
    action.add_argument("--run", action="store_true", help="launch one bounded full local read-only census worker")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path, help="explicit pinned local snapshot; never a Hub identifier")
    parser.add_argument("--output-dir", type=Path, help="fresh output path required by --run")
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--counts", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--invalid", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.preflight:
            _bounded, _teacher, preflight = _metadata_preflight(args.snapshot)
            print(json.dumps({"ok": True, "status": "PREFLIGHT_COMPLETE", "preflight": preflight}, sort_keys=True))
            return 0
        if args.worker:
            inherited = os.environ.get(WORKER_TOKEN_ENV, "")
            if not args.worker_token or not secrets.compare_digest(args.worker_token, inherited):
                raise GateError("internal worker token mismatch")
            if any(value is None for value in (args.output_dir, args.manifest, args.log, args.counts, args.invalid, args.result)):
                raise GateError("internal worker requires output paths")
            return _worker(args)
        if args.output_dir is None:
            raise GateError("--run requires --output-dir pointing to a fresh path")
        return _run_parent(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "INCOMPLETE", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
