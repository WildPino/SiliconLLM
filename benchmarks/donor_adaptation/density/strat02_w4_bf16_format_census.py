#!/usr/bin/env python3
"""Count-only full-format census for STRAT-02 W4 g128 BF16-scale v2.

The parent fixes the meta-model inventory and supervises eleven fresh,
sequential, direct-Python workers.  Each worker hashes only its assigned
local pinned shard before reading F32 slices of at most 128 rows.  No weights,
coordinates list, model output, or packed blobs are written.  A first invalid
group is retained as a single decisive counterexample in the worker result.
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
SCHEMA = "strat02_w4_bf16_full_format_census_v1"
WORKER_TOKEN_ENV = "STRAT02_W4_BF16_CENSUS_WORKER_TOKEN"
GROUP_SIZE = 128
MAX_ROWS_PER_SLICE = 128
EXPECTED_KEYS = 6259
EXPECTED_LINEARS = 6225
EXPECTED_WEIGHTS = 13_363_052_544
EXPECTED_GROUPS = 104_398_848
EXPECTED_SHARDS = 11
ROUTER_KEYS = tuple(f"model.layers.{layer}.mlp.gate.weight" for layer in range(16))
ROUTER_ROW = 127
ROUTER_GROUPS = 256
EXPERT_CONTROL_KEY = "model.layers.0.mlp.experts.0.down_proj.weight"
EXPERT_CONTROL_ROW, EXPERT_CONTROL_GROUP = 0, 3
MIN_LAUNCH_RAM_BYTES = 8 * 1024 ** 3
MIN_LAUNCH_OUTPUT_FREE_BYTES = 1 * 1024 ** 3
MIN_RUNTIME_RAM_BYTES = 4 * 1024 ** 3
MAX_PRIVATE_COMMIT_BYTES = 8 * 1024 ** 3
MAX_SHARD_SECONDS = 15 * 60
MAX_TOTAL_SECONDS = 60 * 60
MONITOR_INTERVAL_SECONDS = 5.0
MAX_COUNTS_BYTES = 16 * 1024 ** 2
BRIEF_PATH = HERE.parents[2] / "docs" / "research" / "donor_adaptation" / "briefs" / "BRIEF_STRAT_02_W4_BF16_FULL_FORMAT_CENSUS.md"
SOURCES = {"census": Path(__file__).resolve(), "brief": BRIEF_PATH,
           "codec": HERE / "strat02_w4_bf16_codec.py", "teacher": HERE / "strat02_mmap_teacher.py",
           "supervisor": HERE / "strat02_bounded_smoke.py"}
COUNT_FIELDS = ("groups", "true_zero", "source_nonfinite", "scale_underflow", "scale_overflow_nonfinite")


class GateError(RuntimeError):
    """Apparatus, pinned-source, or accounting invariant failure."""


class OutputCapError(RuntimeError):
    """The count JSONL would exceed the preregistered 16 MiB cap."""


class ResourceError(RuntimeError):
    """A preregistered launch resource threshold was not met."""


class SourceNonfinite(RuntimeError):
    def __init__(self, record: dict[str, Any]):
        self.record = record
        super().__init__("nonfinite source F32 group")


class FormatCounterexample(RuntimeError):
    def __init__(self, record: dict[str, Any]):
        self.record = record
        super().__init__("first invalid nonzero BF16-scale group")


@dataclass(frozen=True)
class MonitorOutcome:
    status: str
    exit_code: int | None
    reason: str | None
    elapsed_seconds: float
    samples: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _load_local_module(filename: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot import local module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"utc_timestamp": _utc_now(), **event}, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _prepare_output(directory: Path) -> dict[str, Path]:
    root = directory.resolve()
    if root.exists():
        raise GateError(f"output must be a fresh directory: {root}")
    root.mkdir(parents=True, exist_ok=False)
    paths = {"root": root, "manifest": root / "census_manifest.json", "log": root / "supervisor_log.jsonl",
             "result": root / "supervisor_result.json"}
    with paths["log"].open("x", encoding="utf-8", newline="\n"):
        pass
    return paths


def _psutil() -> Any:
    import psutil
    return psutil


def _inventory(teacher: Any, report: Any) -> dict[str, Any]:
    """Plan every linear from actual nn.Linear modules on the pinned meta model."""
    teacher._require_target_environment()
    import torch
    EmoConfig, EmoForCausalLM = teacher._import_pinned_emo(report.snapshot)
    config = teacher._read_json(report.snapshot / "config.json", "pinned model config")
    with torch.device("meta"):
        model = EmoForCausalLM(EmoConfig(**config))
    state = model.state_dict()
    index_map = report.index["weight_map"]
    if len(state) != EXPECTED_KEYS or set(state) != set(index_map):
        raise GateError("pinned model/index 6,259-key inventory mismatch")
    by_shard: dict[str, list[dict[str, Any]]] = {entry["name"]: [] for entry in report.manifest["shards"]}
    names: set[str] = set()
    weights = groups = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = module_name + ".weight"
        if not module_name or key in names or key not in index_map or module.weight is None:
            raise GateError(f"invalid/duplicate linear key {key}")
        shape = tuple(int(dimension) for dimension in module.weight.shape)
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 or shape[1] % GROUP_SIZE or tuple(state[key].shape) != shape:
            raise GateError(f"invalid g128 linear shape/state for {key}")
        shard = index_map[key]
        if shard not in by_shard:
            raise GateError(f"linear maps to an unpinned shard: {key}")
        by_shard[shard].append({"key": key, "shape": list(shape)})
        names.add(key)
        weights += shape[0] * shape[1]
        groups += shape[0] * (shape[1] // GROUP_SIZE)
    if (len(names), weights, groups) != (EXPECTED_LINEARS, EXPECTED_WEIGHTS, EXPECTED_GROUPS):
        raise GateError("pinned linear count/weight/group inventory mismatch")
    for entries in by_shard.values():
        entries.sort(key=lambda entry: entry["key"])
    if not set(ROUTER_KEYS).issubset(names) or EXPERT_CONTROL_KEY not in names:
        raise GateError("preregistered positive controls absent from structural inventory")
    for key in ROUTER_KEYS:
        entry = next(item for entries in by_shard.values() for item in entries if item["key"] == key)
        if entry["shape"][0] <= ROUTER_ROW or entry["shape"][1] // GROUP_SIZE != 16:
            raise GateError(f"router control shape mismatch: {key}")
    expert = next(item for entries in by_shard.values() for item in entries if item["key"] == EXPERT_CONTROL_KEY)
    if expert["shape"][0] <= EXPERT_CONTROL_ROW or expert["shape"][1] // GROUP_SIZE <= EXPERT_CONTROL_GROUP:
        raise GateError("expert positive control shape mismatch")
    return {"index_key_count": len(index_map), "linear_count": len(names), "linear_weight_count": weights,
            "linear_group_count": groups, "by_shard": by_shard}


def _metadata_preflight(snapshot: Path | None) -> tuple[Any, Any, Any, dict[str, Any]]:
    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_bf16_census_bounded")
    teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_bf16_census_teacher")
    codec = _load_local_module("strat02_w4_bf16_codec.py", "_strat02_bf16_census_codec")
    versions = bounded._pinned_environment()
    codec.selftest()
    report = teacher.preflight(snapshot=snapshot)
    if not report.source_status.get("transformers_ok") or report.source_status.get("use_hub_kernels") != "NO" or not report.source_status.get("shards_present"):
        raise GateError("pinned local metadata/runtime preflight failed")
    inventory = _inventory(teacher, report)
    _validate_endpoint_shortcut(codec)
    return bounded, teacher, report, {"utc_timestamp": _utc_now(), "runtime_versions": versions,
        "snapshot": str(report.snapshot), "source_status": dict(report.source_status),
        "inventory": inventory, "apparatus_sha256": {name: _sha256_file(path) for name, path in SOURCES.items()},
        "offline_policy": {"download": False, "heldout": False, "donor_forward": False, "t4": False,
                           "weight_payload_access_in_preflight": False}}


def _launch_preflight(snapshot: Path | None, output_root: Path) -> tuple[Any, dict[str, Any]]:
    bounded, _teacher, _report, preflight = _metadata_preflight(snapshot)
    psutil = _psutil()
    available, free = int(psutil.virtual_memory().available), int(psutil.disk_usage(str(output_root)).free)
    if available < MIN_LAUNCH_RAM_BYTES or free < MIN_LAUNCH_OUTPUT_FREE_BYTES:
        raise ResourceError(f"launch resource preflight failed: available_ram={available} free_disk={free}")
    return bounded, {**preflight, "available_physical_ram_bytes": available, "output_volume_free_bytes": free}


def _candidate_scale(maxima: np.ndarray, c: np.float32, codec: Any) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        candidate = np.float32(np.float32(np.float32(maxima) * np.float32(c)) / np.float32(7))
        return np.asarray(codec.bf16_bits_to_float32(codec.bf16_rne_bits(candidate)), dtype=np.float32)


def _classify_tile(values: np.ndarray, codec: Any) -> dict[str, np.ndarray]:
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[2] != GROUP_SIZE or values.shape[0] > MAX_ROWS_PER_SLICE:
        raise GateError("census tile must be F32 [<=128 rows, groups, 128]")
    finite = np.isfinite(values).all(axis=2)
    maxima = np.zeros(values.shape[:2], dtype=np.float32)
    if finite.any():
        maxima[finite] = np.max(np.abs(values[finite]), axis=1)
    zero = finite & (maxima == 0)
    nonzero = finite & ~zero
    low = np.zeros_like(maxima)
    high = np.zeros_like(maxima)
    if nonzero.any():
        low[nonzero] = _candidate_scale(maxima[nonzero], codec.W4_CANDIDATE_C[0], codec)
        high[nonzero] = _candidate_scale(maxima[nonzero], codec.W4_CANDIDATE_C[-1], codec)
    underflow = nonzero & (low <= 0)
    overflow = nonzero & ~np.isfinite(high)
    return {"finite": finite, "zero": zero, "nonzero": nonzero, "underflow": underflow,
            "overflow": overflow, "maxima": maxima, "low": low, "high": high}


def _all_candidate_reason(group: np.ndarray, codec: Any) -> str | None:
    if group.dtype != np.float32 or group.shape != (GROUP_SIZE,):
        raise GateError("scalar check requires a 128-value F32 group")
    if not np.isfinite(group).all():
        return "source_nonfinite"
    maximum = np.max(np.abs(group), initial=np.float32(0))
    if maximum == 0:
        return None
    scales = [_candidate_scale(np.asarray([maximum], dtype=np.float32), c, codec)[0] for c in codec.W4_CANDIDATE_C]
    if any(scale <= 0 for scale in scales):
        return "scale_underflow"
    if any(not np.isfinite(scale) for scale in scales):
        return "scale_overflow_nonfinite"
    return None


def _validate_endpoint_shortcut(codec: Any) -> None:
    zero = np.zeros(GROUP_SIZE, dtype=np.float32)
    smallest = zero.copy(); smallest[0] = np.nextafter(np.float32(0), np.float32(1))
    halfway = zero.copy(); halfway[0] = np.float32(14.0546875)
    largest = zero.copy(); largest[0] = np.finfo(np.float32).max
    cases = [zero, smallest, halfway, largest]
    rng = np.random.default_rng(20260917)
    exponents = rng.uniform(-43, 37, size=256)
    for exponent in exponents:
        local_exponents = np.clip(exponent + rng.uniform(-2, 0, size=GROUP_SIZE), -44, 37)
        case = np.asarray(10.0 ** local_exponents, dtype=np.float32)
        case *= rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=GROUP_SIZE)
        cases.append(case)
    for group in cases:
        endpoint = _classify_tile(group.reshape(1, 1, GROUP_SIZE), codec)
        reason = "scale_underflow" if endpoint["underflow"][0, 0] else "scale_overflow_nonfinite" if endpoint["overflow"][0, 0] else None
        if reason != _all_candidate_reason(group, codec):
            raise GateError("endpoint shortcut disagrees with all eleven BF16 candidates")
    if _all_candidate_reason(smallest, codec) != "scale_underflow" or _all_candidate_reason(zero, codec) is not None:
        raise GateError("planted underflow/zero control failed")
    halfway_scale = _candidate_scale(np.asarray([np.float32(14.0546875)], dtype=np.float32), codec.W4_CANDIDATE_C[0], codec)[0]
    if halfway_scale != np.float32(1.0) or int(codec.bf16_rne_bits(np.asarray([np.finfo(np.float32).max], dtype=np.float32))[0]) != 0x7F80:
        raise GateError("planted BF16 halfway/finite-to-overflow control failed")


def _first_problem(classification: dict[str, np.ndarray]) -> tuple[int, int, str] | None:
    mask = ~classification["finite"] | classification["underflow"] | classification["overflow"]
    positions = np.argwhere(mask)
    if not len(positions):
        return None
    row, group = map(int, positions[0])
    reason = ("source_nonfinite" if not classification["finite"][row, group] else
              "scale_underflow" if classification["underflow"][row, group] else "scale_overflow_nonfinite")
    return row, group, reason


def _json_f32(value: np.float32) -> float | str:
    """Keep decisive nonfinite endpoint observations JSON compliant."""
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return float(value)


def _confirm_problem(values: np.ndarray, classification: dict[str, np.ndarray], position: tuple[int, int, str],
                     *, key: str, shard: str, row_offset: int, codec: Any) -> dict[str, Any]:
    row, group, reason = position
    source = values[row, group]
    record = {"tensor": key, "shard": shard, "row": row_offset + row, "group": group,
              "classification": reason, "absmax": _json_f32(classification["maxima"][row, group]) if reason != "source_nonfinite" else None,
              "scale_c_0_50": _json_f32(classification["low"][row, group]) if reason != "source_nonfinite" else None,
              "scale_c_1_00": _json_f32(classification["high"][row, group]) if reason != "source_nonfinite" else None}
    if reason == "source_nonfinite":
        if np.isfinite(source).all():
            raise GateError("source-nonfinite tile/scalar mismatch")
        return record
    if _all_candidate_reason(source, codec) != reason:
        raise GateError("endpoint/all-eleven mismatch on first invalid real group")
    try:
        codec._w4_bf16_group_scalar_reference(source)
    except ValueError as exc:
        if "invalid BF16 candidate scale" not in str(exc):
            raise GateError("scalar codec rejected group for another reason") from exc
        record["scalar_codec_confirmation"] = "invalid BF16 candidate scale"
    else:
        raise GateError("scalar v2 codec accepted endpoint-invalid group")
    return record


def _real_valid_reference(group: np.ndarray, codec: Any) -> None:
    if _all_candidate_reason(group, codec) is not None:
        raise GateError("endpoint/all-eleven mismatch on sampled real valid group")
    bits, _codes = codec._w4_bf16_group_scalar_reference(group)
    if int(bits) == 0 or not np.isfinite(codec.bf16_bits_to_float32(bits)):
        raise GateError("scalar v2 codec returned invalid scale for valid real group")


def _append_count(handle: Any, record: Mapping[str, Any], current_bytes: int, remaining_cap: int) -> int:
    raw = (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if current_bytes + len(raw) > remaining_cap:
        raise OutputCapError(f"count JSONL would exceed the 16 MiB aggregate cap ({MAX_COUNTS_BYTES} bytes)")
    handle.write(raw.decode("utf-8"))
    handle.flush()
    return current_bytes + len(raw)


def _scan_tensor(handle: Any, *, key: str, shape: Sequence[int], shard: str, codec: Any,
                 counts_handle: Any, count_bytes: int, remaining_cap: int) -> tuple[int, dict[str, int]]:
    import torch
    tensor_slice = handle.get_slice(key)
    if tuple(tensor_slice.get_shape()) != tuple(shape):
        raise GateError(f"safetensors/meta shape mismatch for {key}")
    rows, columns = map(int, shape)
    groups = columns // GROUP_SIZE
    totals = {field: 0 for field in COUNT_FIELDS}
    router_control = expert_control = 0
    sampled_real_valid = 0
    for row_start in range(0, rows, MAX_ROWS_PER_SLICE):
        row_end = min(row_start + MAX_ROWS_PER_SLICE, rows)
        tensor = tensor_slice[row_start:row_end]
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tuple(tensor.shape) != (row_end - row_start, columns):
            raise GateError(f"non-F32 or malformed source slice: {key}")
        source = tensor.numpy()
        if source.dtype != np.float32:
            raise GateError(f"failed F32 numpy view for {key}")
        values = source.reshape(row_end - row_start, groups, GROUP_SIZE)
        classified = _classify_tile(values, codec)
        problem = _first_problem(classified)
        if problem is not None:
            record = _confirm_problem(values, classified, problem, key=key, shard=shard, row_offset=row_start, codec=codec)
            if problem[2] == "source_nonfinite":
                raise SourceNonfinite(record)
            raise FormatCounterexample(record)
        current_groups = (row_end - row_start) * groups
        totals["groups"] += current_groups
        totals["true_zero"] += int(np.count_nonzero(classified["zero"]))
        if totals["groups"] != (row_end * groups):
            raise GateError("slice group accounting mismatch")
        if key in ROUTER_KEYS and row_start <= ROUTER_ROW < row_end:
            control = classified["nonzero"][ROUTER_ROW - row_start]
            if not np.all(control) or control.size != 16:
                raise GateError(f"router shared-row control is not nonzero and BF16-valid: {key}")
            router_control += int(control.size)
        if key == EXPERT_CONTROL_KEY and row_start <= EXPERT_CONTROL_ROW < row_end:
            if not classified["nonzero"][EXPERT_CONTROL_ROW - row_start, EXPERT_CONTROL_GROUP]:
                raise GateError("expert [0,3] control is not nonzero and BF16-valid")
            expert_control += 1
        if sampled_real_valid < 2:
            positions = np.argwhere(classified["nonzero"])
            for local_row, group in positions[:2 - sampled_real_valid]:
                _real_valid_reference(values[int(local_row), int(group)], codec)
                sampled_real_valid += 1
        del tensor, source, values, classified
    if totals["groups"] != rows * groups or sum(totals[field] for field in COUNT_FIELDS[1:]) > totals["groups"]:
        raise GateError(f"tensor group accounting mismatch: {key}")
    record = {"tensor": key, "shard": shard, "shape": [rows, columns], **totals,
              "router_shared_nonzero_valid": router_control, "expert_0_3_nonzero_valid": expert_control,
              "real_valid_scalar_checks": sampled_real_valid}
    return _append_count(counts_handle, record, count_bytes, remaining_cap), record


def _metadata_fingerprint(report: Any) -> dict[str, str]:
    return {"manifest": _sha256_file(report.manifest_path), "index": _sha256_file(report.snapshot / report.manifest["index_file"]),
            **{name: _sha256_file(report.snapshot / name) for name in ("config.json", "configuration_emo.py", "modeling_emo.py")}}


def _same_snapshot_directory(left: Path | str, right: Path | str) -> bool:
    """Compare the filesystem object, not C: cache-junction versus E: target text."""
    try:
        return Path(left).samefile(Path(right))
    except OSError as exc:
        raise GateError("cannot resolve pinned snapshot directory identity") from exc


def _verify_assigned_shard(report: Any, entry: Mapping[str, Any]) -> tuple[dict[str, int], str]:
    path = report.snapshot / entry["name"]
    before = path.stat()
    if before.st_size != entry["size"]:
        raise GateError(f"assigned pinned shard size mismatch: {entry['name']}")
    digest = _sha256_file(path)  # exactly one full shard hash pass in this worker
    after = path.stat()
    if digest != entry["sha256"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise GateError(f"assigned pinned shard SHA/stat mismatch: {entry['name']}")
    return {"size": after.st_size, "mtime_ns": after.st_mtime_ns}, digest


def _assert_shard_stat(report: Any, shard: str, verified: Mapping[str, int]) -> None:
    stat = (report.snapshot / shard).stat()
    if (stat.st_size, stat.st_mtime_ns) != (verified["size"], verified["mtime_ns"]):
        raise GateError(f"assigned shard changed after SHA verification: {shard}")


def _worker(args: argparse.Namespace) -> int:
    completed = 0
    total = {field: 0 for field in COUNT_FIELDS}
    control_totals = {"router_shared_nonzero_valid": 0, "expert_0_3_nonzero_valid": 0, "real_valid_scalar_checks": 0}
    used = 0
    report = shard_stat = fingerprint = digest = shard = teacher = None
    try:
        manifest = json.loads((args.output_dir / "census_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("parent_preflight", {}).get("apparatus_sha256") != {name: _sha256_file(path) for name, path in SOURCES.items()}:
            raise GateError("census apparatus changed after parent preflight")
        teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_bf16_census_worker_teacher")
        codec = _load_local_module("strat02_w4_bf16_codec.py", "_strat02_bf16_census_worker_codec")
        codec.selftest()
        _validate_endpoint_shortcut(codec)
        report = teacher.preflight(snapshot=args.snapshot)
        if not _same_snapshot_directory(report.snapshot, manifest["parent_preflight"]["snapshot"]):
            raise GateError("worker snapshot differs from parent preflight")
        shard_entries = report.manifest["shards"]
        if not 0 <= args.shard_index < len(shard_entries):
            raise GateError("worker assigned invalid shard index")
        entry = shard_entries[args.shard_index]
        shard = entry["name"]
        plan = manifest["parent_preflight"]["inventory"]["by_shard"][shard]
        if [item["key"] for item in plan] != sorted(item["key"] for item in plan):
            raise GateError("worker tensor plan is not lexicographic")
        if any(report.index["weight_map"].get(item["key"]) != shard for item in plan):
            raise GateError("worker tensor plan/index mismatch")
        fingerprint = _metadata_fingerprint(report)
        shard_stat, digest = _verify_assigned_shard(report, entry)
        _assert_shard_stat(report, shard, shard_stat)
        from safetensors import safe_open
        with args.counts.open("x", encoding="utf-8", newline="\n") as counts_handle:
            with safe_open(str(report.snapshot / shard), framework="pt", device="cpu") as handle:
                for item in plan:
                    _assert_shard_stat(report, shard, shard_stat)
                    used, record = _scan_tensor(handle, key=item["key"], shape=item["shape"], shard=shard, codec=codec,
                                                counts_handle=counts_handle, count_bytes=used, remaining_cap=args.remaining_bytes)
                    completed += 1
                    for field in COUNT_FIELDS:
                        total[field] += record[field]
                    for field in control_totals:
                        control_totals[field] += record[field]
            counts_handle.flush()
            os.fsync(counts_handle.fileno())
        _assert_shard_stat(report, shard, shard_stat)
        if _metadata_fingerprint(report) != fingerprint:
            raise GateError("pinned metadata changed during assigned shard reads")
        if completed != len(plan) or used != args.counts.stat().st_size:
            raise GateError("worker count file/accounting mismatch")
        _write_json_once(args.result, {"ok": True, "status": "SHARD_COMPLETE", "shard": shard, "shard_index": args.shard_index,
            "shard_sha256": digest, "shard_stat": shard_stat, "completed_tensors": completed,
            "counts": total, "controls": control_totals, "counts_bytes": used, "counts_sha256": _sha256_file(args.counts),
            "metadata_fingerprint": fingerprint, "utc_timestamp": _utc_now()})
        return 0
    except Exception as exc:
        integrity_error = getattr(teacher, "IntegrityError", ()) if teacher is not None else ()
        status = ("FORMAT_INVALID_COUNTEREXAMPLE" if isinstance(exc, FormatCounterexample) else
                  "VOID_SOURCE" if isinstance(exc, SourceNonfinite) else
                  "INCOMPLETE" if isinstance(exc, OutputCapError) else
                  "VOID_APPARATUS" if isinstance(exc, (GateError, OSError, ValueError, integrity_error)) else "INCOMPLETE")
        # A decisive early stop still checks that its verified source and
        # pinned metadata stayed unchanged throughout the value reads.
        if report is not None and shard_stat is not None and shard is not None and fingerprint is not None:
            try:
                _assert_shard_stat(report, shard, shard_stat)
                if _metadata_fingerprint(report) != fingerprint:
                    raise GateError("pinned metadata changed during assigned shard reads")
            except Exception as mutation_exc:
                status = "VOID_APPARATUS"
                exc = mutation_exc
        partial_size = args.counts.stat().st_size if args.counts.exists() else None
        partial_sha = _sha256_file(args.counts) if args.counts.exists() else None
        try:
            _write_json_once(args.result, {"ok": False, "status": status, "shard_index": args.shard_index,
                "shard": shard, "verified_shard_sha256": digest,
                "completed_tensors": completed, "partial_counts": total, "partial_controls": control_totals,
                "partial_counts_file": args.counts.name if args.counts.exists() else None,
                "partial_counts_bytes": partial_size, "partial_counts_sha256": partial_sha,
                "counterexample": exc.record if status in {"FORMAT_INVALID_COUNTEREXAMPLE", "VOID_SOURCE"} and isinstance(exc, (FormatCounterexample, SourceNonfinite)) else None,
                "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                "partial_files_preserved": True, "utc_timestamp": _utc_now()})
        except (OSError, FileExistsError):
            pass
        return 1


def _process_sample(process: Any) -> dict[str, int]:
    info, full = process.memory_info(), process.memory_full_info()
    private = getattr(full, "private", None)
    if private is None:
        raise GateError("private commit unavailable for direct worker PID")
    return {"worker_working_set_bytes": int(info.rss), "worker_private_commit_bytes": int(private)}


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


def _monitor_child(process: Any, *, psutil: Any, expected_executable: Path, on_sample: Callable[[Mapping[str, Any]], None],
                   now: Callable[[], float] = time.monotonic, wall_limit_seconds: float = MAX_SHARD_SECONDS,
                   interval_seconds: float = MONITOR_INTERVAL_SECONDS) -> MonitorOutcome:
    started, samples, checked, resource = now(), 0, False, None
    while True:
        elapsed = now() - started
        if elapsed >= wall_limit_seconds:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "wall_clock_exceeded", elapsed, samples)
        exit_code = process.poll()
        if exit_code is not None:
            return MonitorOutcome("CHILD_EXITED", int(exit_code), None, elapsed, samples)
        try:
            resource = resource or psutil.Process(process.pid)
            if not checked:
                if Path(resource.exe()).resolve() != expected_executable.resolve():
                    return MonitorOutcome("VOID_APPARATUS", _terminate_only_child(process), "monitored_executable_mismatch", elapsed, samples)
                checked = True
            sample = _process_sample(resource)
            available = int(psutil.virtual_memory().available)
        except Exception as exc:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), f"resource_sample_failed:{exc}", elapsed, samples)
        on_sample({"elapsed_seconds": elapsed, "available_physical_ram_bytes": available, **sample})
        samples += 1
        if available < MIN_RUNTIME_RAM_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "available_physical_ram_below_4_gib", elapsed, samples)
        if sample["worker_private_commit_bytes"] > MAX_PRIVATE_COMMIT_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "worker_private_commit_above_8_gib", elapsed, samples)
        try:
            process.wait(timeout=min(interval_seconds, max(0.0, wall_limit_seconds - elapsed)))
        except subprocess.TimeoutExpired:
            pass


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _validate_shard_output(path: Path, worker: Mapping[str, Any], plan: Sequence[Mapping[str, Any]], shard: str,
                           expected_sha: str, remaining_cap: int) -> tuple[dict[str, int], dict[str, int], int]:
    try:
        size, digest = path.stat().st_size, _sha256_file(path)
    except OSError as exc:
        raise GateError("worker count file is absent or unreadable") from exc
    if size > remaining_cap or size != worker.get("counts_bytes") or digest != worker.get("counts_sha256"):
        raise GateError("worker count file size/SHA/cap mismatch")
    if worker.get("shard") != shard or worker.get("shard_sha256") != expected_sha or worker.get("completed_tensors") != len(plan):
        raise GateError("worker assigned shard identity/count mismatch")
    totals = {field: 0 for field in COUNT_FIELDS}
    controls = {"router_shared_nonzero_valid": 0, "expert_0_3_nonzero_valid": 0, "real_valid_scalar_checks": 0}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for expected in plan:
                line = handle.readline()
                if not line:
                    raise GateError("worker JSONL ended before planned tensors")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GateError("worker JSONL is malformed") from exc
                if not isinstance(record, dict):
                    raise GateError("worker JSONL record is not an object")
                if record.get("tensor") != expected["key"] or record.get("shape") != expected["shape"] or record.get("shard") != shard:
                    raise GateError("worker JSONL tensor plan mismatch")
                expected_groups = expected["shape"][0] * (expected["shape"][1] // GROUP_SIZE)
                if any(type(record.get(field)) is not int or record[field] < 0 for field in (*COUNT_FIELDS, *controls)):
                    raise GateError("worker JSONL contains invalid count fields")
                if (record["groups"] != expected_groups or record["source_nonfinite"] or record["scale_underflow"] or
                    record["scale_overflow_nonfinite"] or record["true_zero"] > record["groups"]):
                    raise GateError("worker JSONL count invariant failed")
                expected_router = 16 if expected["key"] in ROUTER_KEYS else 0
                expected_expert = 1 if expected["key"] == EXPERT_CONTROL_KEY else 0
                if record["router_shared_nonzero_valid"] != expected_router or record["expert_0_3_nonzero_valid"] != expected_expert or record["real_valid_scalar_checks"] > 2:
                    raise GateError("worker JSONL positive-control/reference-count invariant failed")
                for field in totals:
                    totals[field] += record[field]
                for field in controls:
                    controls[field] += record[field]
            if handle.readline():
                raise GateError("worker JSONL has records beyond planned tensors")
    except (OSError, UnicodeDecodeError) as exc:
        raise GateError("worker count JSONL is unreadable") from exc
    if totals != worker.get("counts") or controls != worker.get("controls"):
        raise GateError("parent/worker shard count summaries disagree")
    return totals, controls, size


def _valid_counterexample_record(record: object, plan: Sequence[Mapping[str, Any]], shard: str,
                                 classification: str) -> bool:
    if not isinstance(record, dict) or record.get("shard") != shard:
        return False
    matching = [item for item in plan if item["key"] == record.get("tensor")]
    if len(matching) != 1 or type(record.get("row")) is not int or type(record.get("group")) is not int:
        return False
    rows, columns = matching[0]["shape"]
    if not (0 <= record["row"] < rows and 0 <= record["group"] < columns // GROUP_SIZE):
        return False
    if classification == "VOID_SOURCE":
        return record.get("classification") == "source_nonfinite"
    return (record.get("classification") in {"scale_underflow", "scale_overflow_nonfinite"} and
            record.get("scalar_codec_confirmation") == "invalid BF16 candidate scale")


def _run_parent(args: argparse.Namespace) -> int:
    started = time.monotonic()
    paths = _prepare_output(args.output_dir)
    try:
        bounded, preflight = _launch_preflight(args.snapshot, paths["root"])
        _write_json_once(paths["manifest"], {"schema": SCHEMA, "purpose": "count-only W4 BF16-scale v2 full-format census",
            "algorithm": {"shard_order": "manifest order", "tensor_order": "lexicographic within shard", "slice_rows_max": MAX_ROWS_PER_SLICE,
                          "groups": GROUP_SIZE, "endpoints": [0.50, 1.00], "candidate_count": 11,
                          "positive_controls": {"router_shared_row_127_groups": ROUTER_GROUPS, "expert_down_0_3": 1}},
            "limits": {"launch_ram_min": MIN_LAUNCH_RAM_BYTES, "launch_output_free_min": MIN_LAUNCH_OUTPUT_FREE_BYTES,
                       "runtime_ram_min": MIN_RUNTIME_RAM_BYTES, "worker_private_commit_max": MAX_PRIVATE_COMMIT_BYTES,
                       "shard_wall_max_seconds": MAX_SHARD_SECONDS, "total_wall_max_seconds": MAX_TOTAL_SECONDS,
                       "sample_interval_seconds": MONITOR_INTERVAL_SECONDS, "counts_bytes_max": MAX_COUNTS_BYTES},
            "output": {"auto_resume": False, "raw_weights": False, "heldout": False, "download": False},
            "parent_preflight": preflight})
        _append_log(paths["log"], {"event": "preflight_complete"})
    except Exception as exc:
        _write_json_once(paths["manifest"], {"schema": SCHEMA, "preflight": "failed"})
        _write_json_once(paths["result"], {"ok": False, "status": "INCOMPLETE", "failure_classification": "VOID_RESOURCE" if isinstance(exc, ResourceError) else "VOID_APPARATUS",
            "error_type": type(exc).__name__, "error": str(exc), "utc_timestamp": _utc_now()})
        return 1
    totals = {field: 0 for field in COUNT_FIELDS}
    controls = {"router_shared_nonzero_valid": 0, "expert_0_3_nonzero_valid": 0, "real_valid_scalar_checks": 0}
    completed_shards: list[dict[str, Any]] = []
    used = 0
    classification: str | None = None
    counterexample: dict[str, Any] | None = None
    reason: str | None = None
    try:
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        worker_python = bounded._direct_worker_python(environment)
        token = secrets.token_urlsafe(32)
        environment[WORKER_TOKEN_ENV] = token
        # The parent plan is frozen before any worker reads a source value.
        teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_bf16_census_parent_teacher")
        report = teacher.preflight(snapshot=Path(preflight["snapshot"]))
        if report.source_status.get("manifest_sha256") != preflight["source_status"].get("manifest_sha256") or report.source_status.get("index_sha256") != preflight["source_status"].get("index_sha256"):
            raise GateError("parent metadata changed after inventory preflight")
        entries = report.manifest["shards"]
        if len(entries) != EXPECTED_SHARDS:
            raise GateError("manifest shard count changed")
        for shard_index, entry in enumerate(entries):
            remaining_wall = MAX_TOTAL_SECONDS - (time.monotonic() - started)
            if remaining_wall <= 0:
                classification, reason = "VOID_RESOURCE", "total_wall_clock_exceeded"
                break
            shard = entry["name"]
            stem = f"shard_{shard_index + 1:02d}"
            counts = paths["root"] / f"{stem}_tensor_counts.jsonl"
            worker_result = paths["root"] / f"{stem}_worker_result.json"
            stdout_path, stderr_path = paths["root"] / f"{stem}_stdout.log", paths["root"] / f"{stem}_stderr.log"
            command = [worker_python, str(Path(__file__).resolve()), "--worker", "--snapshot", preflight["snapshot"],
                       "--output-dir", str(paths["root"]), "--shard-index", str(shard_index), "--counts", str(counts),
                       "--result", str(worker_result), "--remaining-bytes", str(MAX_COUNTS_BYTES - used), "--worker-token", token]
            _append_log(paths["log"], {"event": "shard_launch", "shard": shard, "index": shard_index,
                                       "termination_scope": "direct_worker_pid_only", "remaining_count_bytes": MAX_COUNTS_BYTES - used})
            process = None
            try:
                with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
                    process = subprocess.Popen(command, cwd=str(HERE), env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                    outcome = _monitor_child(process, psutil=_psutil(), expected_executable=Path(worker_python),
                        wall_limit_seconds=min(MAX_SHARD_SECONDS, remaining_wall),
                        on_sample=lambda sample: _append_log(paths["log"], {"event": "resource_sample", "shard": shard, **sample}))
            except Exception:
                if process is not None:
                    _terminate_only_child(process)
                raise
            worker = _load_json(worker_result)
            _append_log(paths["log"], {"event": "shard_finished", "shard": shard, "monitor": asdict(outcome),
                                       "worker_status": worker.get("status") if worker else None})
            if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
                classification, reason = outcome.status, outcome.reason
                break
            if time.monotonic() - started >= MAX_TOTAL_SECONDS:
                classification, reason = "VOID_RESOURCE", "total_wall_clock_exceeded"
                break
            if worker and worker.get("status") in {"FORMAT_INVALID_COUNTEREXAMPLE", "VOID_SOURCE"}:
                classification = worker["status"]
                counterexample = worker.get("counterexample")
                partial_size = counts.stat().st_size if counts.exists() else None
                partial_sha = _sha256_file(counts) if counts.exists() else None
                if (not _valid_counterexample_record(counterexample, preflight["inventory"]["by_shard"][shard], shard, classification) or
                    outcome.exit_code != 1 or
                    worker.get("shard") != shard or worker.get("verified_shard_sha256") != entry["sha256"] or
                    partial_size is None or partial_size > MAX_COUNTS_BYTES - used or
                    worker.get("partial_counts_bytes") != partial_size or worker.get("partial_counts_sha256") != partial_sha or
                    worker.get("ok") is not False or worker.get("shard_index") != shard_index):
                    classification, reason = "VOID_APPARATUS", "malformed_counterexample_record"
                break
            if outcome.exit_code != 0 or not worker or worker.get("ok") is not True or worker.get("status") != "SHARD_COMPLETE":
                classification = worker.get("status") if worker and worker.get("status") in {"VOID_APPARATUS", "INCOMPLETE"} else "INCOMPLETE"
                reason = worker.get("error") if worker else "worker_result_missing_or_bad"
                break
            shard_totals, shard_controls, size = _validate_shard_output(counts, worker,
                preflight["inventory"]["by_shard"][shard], shard, entry["sha256"], MAX_COUNTS_BYTES - used)
            used += size
            for field in totals:
                totals[field] += shard_totals[field]
            for field in controls:
                controls[field] += shard_controls[field]
            completed_shards.append({"name": shard, "sha256": worker["shard_sha256"], "counts_file": counts.name,
                                     "counts_bytes": size, "counts_sha256": worker["counts_sha256"], "tensors": worker["completed_tensors"]})
        if classification is None:
            if (len(completed_shards) != EXPECTED_SHARDS or sum(item["tensors"] for item in completed_shards) != EXPECTED_LINEARS or
                totals["groups"] != EXPECTED_GROUPS or any(totals[field] for field in COUNT_FIELDS[2:]) or
                controls["router_shared_nonzero_valid"] != ROUTER_GROUPS or controls["expert_0_3_nonzero_valid"] != 1 or
                controls["real_valid_scalar_checks"] == 0 or used > MAX_COUNTS_BYTES):
                raise GateError("full census count, control, or shard-completeness invariant failed")
            verdict = "FORMAT_VALID_FULL"
            status = "COMPLETE"
        else:
            verdict, status = None, "INCOMPLETE"
    except GateError as exc:
        classification, reason, verdict, status = "VOID_APPARATUS", str(exc), None, "INCOMPLETE"
    except Exception as exc:
        classification, reason, verdict, status = "INCOMPLETE", f"{type(exc).__name__}: {exc}", None, "INCOMPLETE"
        _append_log(paths["log"], {"event": "parent_error", "traceback": traceback.format_exc()})
    final = {"ok": status == "COMPLETE", "status": status, "verdict": verdict, "failure_classification": classification,
             "reason": reason, "counterexample": counterexample, "completed_shards": completed_shards,
             "aggregate": totals, "controls": controls, "counts_bytes": used, "utc_timestamp": _utc_now(),
             "partial_files_preserved": True}
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": final["ok"], "status": status, "failure_classification": classification, "result": str(paths["result"])}, sort_keys=True))
    return 0 if status == "COMPLETE" else 1


def _selftest() -> None:
    codec = _load_local_module("strat02_w4_bf16_codec.py", "_strat02_bf16_census_selftest_codec")
    codec.selftest()
    _validate_endpoint_shortcut(codec)
    tile = np.zeros((2, 3, GROUP_SIZE), dtype=np.float32)
    tile[0, 1, 0] = np.nextafter(np.float32(0), np.float32(1))
    tile[0, 2, 0] = np.finfo(np.float32).max
    tile[1, 0, 4] = np.nan
    classified = _classify_tile(tile, codec)
    if (_first_problem(classified) != (0, 1, "scale_underflow") or
        int(np.count_nonzero(classified["zero"])) != 3 or
        not classified["nonzero"][0, 2] or
        int(np.count_nonzero(classified["overflow"])) != 0):
        raise AssertionError("planted tile control failed")
    record = _confirm_problem(tile, classified, (0, 1, "scale_underflow"), key="synthetic", shard="synthetic", row_offset=4, codec=codec)
    if record["row"] != 4 or record["group"] != 1 or record["scalar_codec_confirmation"] != "invalid BF16 candidate scale":
        raise AssertionError("first-invalid scalar codec confirmation failed")
    with tempfile.TemporaryDirectory(prefix="strat02-bf16-census-selftest-") as temporary:
        path = Path(temporary) / "counts.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            _append_count(handle, {"synthetic": True}, 0, MAX_COUNTS_BYTES)
            try:
                _append_count(handle, {"synthetic": True}, handle.tell(), handle.tell())
            except OutputCapError:
                pass
            else:
                raise AssertionError("count cap did not fail closed")
    print(json.dumps({"ok": True, "selftest": True, "donor_weights_opened": False}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selftest", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--counts", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--remaining-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.preflight:
            _bounded, _teacher, _report, preflight = _metadata_preflight(args.snapshot)
            print(json.dumps({"ok": True, "status": "PREFLIGHT_COMPLETE", "preflight": preflight}, sort_keys=True))
            return 0
        if args.worker:
            if (not args.worker_token or not secrets.compare_digest(args.worker_token, os.environ.get(WORKER_TOKEN_ENV, "")) or
                args.snapshot is None or args.output_dir is None or args.shard_index is None or args.counts is None or
                args.result is None or args.remaining_bytes is None or not 0 <= args.remaining_bytes <= MAX_COUNTS_BYTES):
                raise GateError("internal worker arguments/token invalid")
            return _worker(args)
        if args.output_dir is None:
            raise GateError("--run requires a fresh --output-dir")
        return _run_parent(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "INCOMPLETE", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
