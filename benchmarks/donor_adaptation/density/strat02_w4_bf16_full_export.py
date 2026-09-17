#!/usr/bin/env python3
"""Bounded full donor export to STRAT-02 W4 g128 BF16-scale v2 raw shards.

Modes: ``--selftest`` (synthetic), ``--preflight`` (pinned metadata/meta
model only), ``--shard-smoke`` (first shard in a fresh result directory), and
``--run`` (eleven fresh sequential shard workers).  No mode downloads or
opens heldout; this apparatus never performs a donor forward.
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
SCHEMA = "strat02_w4_bf16_full_export_v1"
WORKER_TOKEN_ENV = "STRAT02_W4_BF16_EXPORT_WORKER_TOKEN"
MAX_ROWS = MAX_TILE_ROWS = MAX_TILE_GROUPS = 16
MIN_LAUNCH_RAM = 8 * 1024 ** 3
MIN_LAUNCH_DISK = 20 * 1024 ** 3
MIN_RUNTIME_RAM = 4 * 1024 ** 3
MAX_PRIVATE_COMMIT = 8 * 1024 ** 3
MAX_SHARD_WALL = 90 * 60
MAX_TOTAL_WALL = 12 * 60 * 60
SAMPLE_INTERVAL = 5.0
BRIEF = HERE.parents[2] / "docs" / "research" / "donor_adaptation" / "briefs" / "BRIEF_STRAT_02_W4_BF16_FULL_EXPORT.md"
SOURCES = {"runner": Path(__file__).resolve(), "verifier": HERE / "strat02_w4_bf16_export_verify.py",
           "codec": HERE / "strat02_w4_bf16_codec.py", "teacher": HERE / "strat02_mmap_teacher.py",
           "supervisor": HERE / "strat02_bounded_smoke.py", "token_audit": HERE / "strat02_token_audit.py",
           "brief": BRIEF}


class GateError(RuntimeError):
    """Pinned-source, codec, byte-layout, or accounting failure."""


class ResourceError(RuntimeError):
    """Preflight or monitored resource limit failure."""


class FormatError(RuntimeError):
    def __init__(self, coordinate: Mapping[str, Any]):
        self.coordinate = dict(coordinate)
        super().__init__("nonzero source group has invalid BF16 candidate scale")


@dataclass(frozen=True)
class MonitorOutcome:
    status: str
    exit_code: int | None
    reason: str | None
    elapsed_seconds: float
    samples: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _load_module(filename: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot load local module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_segment(handle: Any, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    handle.seek(offset)
    remaining = length
    while remaining:
        data = handle.read(min(8 * 1024 * 1024, remaining))
        if not data:
            raise GateError("raw record truncated before hash completed")
        digest.update(data)
        remaining -= len(data)
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def _append_log(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        _append_jsonl(handle, {"utc_timestamp": _utc_now(), **value})


def _prepare_output(output: Path) -> dict[str, Path]:
    root = output.resolve()
    if root.exists():
        raise GateError(f"output directory must be new: {root}")
    root.mkdir(parents=True, exist_ok=False)
    paths = {"root": root, "plan": root / "plan_manifest.json", "log": root / "supervisor_log.jsonl",
             "artifact": root / "artifact_manifest.json", "result": root / "supervisor_result.json"}
    with paths["log"].open("x", encoding="utf-8", newline="\n"):
        pass
    return paths


def _tokenizer_hashes(snapshot: Path, token_audit: Any) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name in token_audit.ASSETS:
        path = snapshot / name
        if not path.is_file():
            raise GateError(f"pinned tokenizer asset missing: {name}")
        digest = _sha_file(path)
        if digest != token_audit.EXPECTED_ASSET_SHA256[name]:
            raise GateError(f"pinned tokenizer asset hash mismatch: {name}")
        observed[name] = digest
    return observed


def _inventory(teacher: Any, report: Any, verifier: Any) -> dict[str, Any]:
    """Build all 6,259 records from the exact pinned model on meta."""
    teacher._require_target_environment()
    import torch
    EmoConfig, EmoForCausalLM = teacher._import_pinned_emo(report.snapshot)
    config = teacher._read_json(report.snapshot / "config.json", "pinned model config")
    with torch.device("meta"):
        model = EmoForCausalLM(EmoConfig(**config))
    state = model.state_dict()
    index_map = report.index["weight_map"]
    if len(state) != verifier.EXPECTED_KEYS or set(state) != set(index_map):
        raise GateError("meta model/index tensor keyset mismatch")
    linear_organs: dict[str, str] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            key = module_name + ".weight"
            if not module_name or key in linear_organs or key not in state or tuple(module.weight.shape) != tuple(state[key].shape):
                raise GateError(f"invalid or duplicate nn.Linear weight: {key}")
            parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            grandparent_name = parent_name.rsplit(".", 1)[0] if "." in parent_name else ""
            grandparent = model.get_submodule(grandparent_name) if grandparent_name else model
            if module_name == "lm_head" and getattr(model, "lm_head", None) is module:
                organ = "lm_head"
            elif all(hasattr(parent, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")) and module in {parent.q_proj, parent.k_proj, parent.v_proj, parent.o_proj}:
                organ = "attention"
            elif hasattr(parent, "gate") and hasattr(parent, "experts") and getattr(parent, "gate") is module:
                organ = "router"
            elif isinstance(grandparent, torch.nn.ModuleList) and any(candidate is parent for candidate in grandparent):
                owner_name = grandparent_name.rsplit(".", 1)[0] if "." in grandparent_name else ""
                owner = model.get_submodule(owner_name) if owner_name else model
                if getattr(owner, "experts", None) is not grandparent or not hasattr(owner, "gate"):
                    raise GateError(f"expert parent topology mismatch: {key}")
                organ = "expert"
            else:
                raise GateError(f"linear organ cannot be derived structurally: {key}")
            linear_organs[key] = organ
    if len(linear_organs) != verifier.EXPECTED_LINEARS:
        raise GateError("meta nn.Linear count mismatch")
    organ_counts = {name: sum(organ == name for organ in linear_organs.values()) for name in ("attention", "router", "expert", "lm_head")}
    if organ_counts != verifier.EXPECTED_ORGAN_COUNTS:
        raise GateError(f"meta linear organ taxonomy mismatch: {organ_counts}")
    source_order = [entry["name"] for entry in report.manifest["shards"]]
    if len(source_order) != 11 or len(set(source_order)) != 11:
        raise GateError("source shard order is not the pinned eleven-shard manifest")
    plan: dict[str, list[dict[str, Any]]] = {name: [] for name in source_order}
    ledger = {"records": 0, "linears": 0, "groups": 0, "linear_params": 0, "other_params": 0,
              "linear_bytes": 0, "f32_bytes": 0, "payload_bytes": 0}
    for key in sorted(state):
        tensor = state[key]
        if tensor.dtype != torch.float32:
            raise GateError(f"non-F32 pinned tensor: {key}")
        shape = [int(dimension) for dimension in tensor.shape]
        if any(dimension <= 0 for dimension in shape):
            raise GateError(f"invalid pinned tensor shape: {key}")
        numel = int(tensor.numel())
        encoding = verifier.LINEAR_ENCODING if key in linear_organs else verifier.F32_ENCODING
        length = verifier._record_length(shape, encoding)
        shard = index_map[key]
        if shard not in plan:
            raise GateError(f"unlisted source shard for {key}")
        item = {"name": key, "source_shard": shard, "shape": shape,
                "source_dtype": "float32", "encoding": encoding, "offset": 0, "length": length}
        if key in linear_organs:
            item["organ"] = linear_organs[key]
        plan[shard].append(item)
        ledger["records"] += 1
        if encoding == verifier.LINEAR_ENCODING:
            ledger["linears"] += 1
            ledger["groups"] += numel // 128
            ledger["linear_params"] += numel
            ledger["linear_bytes"] += length
        else:
            ledger["other_params"] += numel
            ledger["f32_bytes"] += length
        ledger["payload_bytes"] += length
    for shard, entries in plan.items():
        offset = 0
        for entry in entries:
            entry["offset"] = offset
            offset += entry["length"]
    frozen = (verifier.EXPECTED_KEYS, verifier.EXPECTED_LINEARS, verifier.EXPECTED_GROUPS,
              verifier.EXPECTED_LINEAR_PARAMS, verifier.EXPECTED_OTHER_PARAMS,
              verifier.EXPECTED_LINEAR_BYTES, verifier.EXPECTED_F32_BYTES, verifier.EXPECTED_PAYLOAD_BYTES)
    if tuple(ledger.values()) != frozen or any(len(plan[name]) == 0 for name in source_order):
        raise GateError(f"meta inventory byte ledger mismatch: {ledger}")
    return {"source_shard_order": source_order, "plan": plan, "ledger": ledger, "organ_counts": organ_counts}


def _preflight(snapshot: Path | None) -> tuple[Any, Any, Any, dict[str, Any]]:
    bounded = _load_module("strat02_bounded_smoke.py", "_strat02_export_bounded")
    teacher = _load_module("strat02_mmap_teacher.py", "_strat02_export_teacher")
    codec = _load_module("strat02_w4_bf16_codec.py", "_strat02_export_codec")
    verifier = _load_module("strat02_w4_bf16_export_verify.py", "_strat02_export_verifier")
    token_audit = _load_module("strat02_token_audit.py", "_strat02_export_token_audit")
    versions = bounded._pinned_environment()
    codec.selftest()
    report = teacher.preflight(snapshot=snapshot)
    if not report.source_status.get("transformers_ok") or report.source_status.get("use_hub_kernels") != "NO" or not report.source_status.get("shards_present"):
        raise GateError("pinned local source/runtime metadata failed")
    tokenizer = _tokenizer_hashes(report.snapshot, token_audit)
    inventory = _inventory(teacher, report, verifier)
    metadata = {"snapshot": str(report.snapshot), "repository": teacher.REPOSITORY, "revision": teacher.REVISION,
                "runtime_versions": versions, "source_status": dict(report.source_status),
                "tokenizer_sha256": tokenizer, "apparatus_sha256": {name: _sha_file(path) for name, path in SOURCES.items()},
                "source_manifest_sha256": _sha_file(report.manifest_path), "source_index_sha256": report.manifest["index_sha256"],
                "source_shards": list(report.manifest["shards"])}
    return bounded, teacher, report, {"metadata": metadata, **inventory}


def _launch_preflight(snapshot: Path | None, output_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    bounded, teacher, _report, preflight = _preflight(snapshot)
    import psutil
    available = int(psutil.virtual_memory().available)
    free = int(psutil.disk_usage(str(output_root)).free)
    if available < MIN_LAUNCH_RAM or free < MIN_LAUNCH_DISK:
        raise ResourceError(f"launch cap failed: available_ram={available} free_output={free}")
    preflight["metadata"].update({"launch_available_ram_bytes": available, "launch_output_free_bytes": free})
    return bounded, teacher, preflight


def _metadata_fingerprint(report: Any, token_audit: Any) -> dict[str, str]:
    paths = {"manifest": report.manifest_path, "index": report.snapshot / report.manifest["index_file"]}
    paths.update({name: report.snapshot / name for name in ("config.json", "configuration_emo.py", "modeling_emo.py", *token_audit.ASSETS)})
    return {name: _sha_file(path) for name, path in paths.items()}


def _verify_source_shard(report: Any, entry: Mapping[str, Any]) -> tuple[tuple[int, int], str]:
    path = report.snapshot / entry["name"]
    before = path.stat()
    if before.st_size != entry["size"]:
        raise GateError(f"source shard size mismatch: {entry['name']}")
    digest = _sha_file(path)  # one full assigned-shard hash before value reads
    after = path.stat()
    stat = (after.st_size, after.st_mtime_ns)
    if digest != entry["sha256"] or stat != (before.st_size, before.st_mtime_ns):
        raise GateError(f"source shard SHA/stat mismatch: {entry['name']}")
    return stat, digest


def _assert_source_stat(report: Any, entry: Mapping[str, Any], stat: tuple[int, int]) -> None:
    current = (report.snapshot / entry["name"]).stat()
    if (current.st_size, current.st_mtime_ns) != stat:
        raise GateError(f"verified source shard changed: {entry['name']}")


def _source_rows(handle: Any, key: str, shape: Sequence[int]):
    """Yield C-contiguous CPU F32 chunks with at most sixteen first-axis rows."""
    import torch
    if not shape:
        tensor = handle.get_tensor(key)
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tensor.ndim != 0:
            raise GateError(f"invalid scalar passthrough source: {key}")
        yield 0, np.asarray(tensor.numpy(), dtype=np.float32).reshape(1)
        return
    source_slice = handle.get_slice(key)
    if tuple(source_slice.get_shape()) != tuple(shape):
        raise GateError(f"source/meta tensor shape mismatch: {key}")
    for start in range(0, shape[0], MAX_ROWS):
        stop = min(start + MAX_ROWS, shape[0])
        tensor = source_slice[start:stop]
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tuple(tensor.shape) != (stop - start, *shape[1:]):
            raise GateError(f"source tensor is not an F32 CPU slice: {key}")
        array = tensor.numpy()
        if array.dtype != np.float32:
            raise GateError(f"source numpy dtype mismatch: {key}")
        yield start, array


def _find_invalid_group(source: np.ndarray, row_start: int, key: str, shard: str, codec: Any) -> FormatError | None:
    rows, columns = source.shape
    for local_row in range(rows):
        for group in range(columns // 128):
            values = source[local_row, group * 128:(group + 1) * 128]
            if not np.isfinite(values).all():
                raise GateError(f"source contains nonfinite F32 values at {key}[{row_start + local_row},{group}]")
            try:
                codec._w4_bf16_group_scalar_reference(values)
            except ValueError as exc:
                if "invalid BF16 candidate scale" not in str(exc):
                    raise GateError("scalar codec rejected group for a different reason") from exc
                return FormatError({"tensor": key, "shard": shard, "row": row_start + local_row,
                                    "group": group, "reason": "invalid BF16 candidate scale"})
    return None


def _check_fixed_group(payload: Any, plan: Mapping[str, Any], row: int, group: int, values: np.ndarray, codec: Any) -> None:
    rows, columns = plan["shape"]
    groups = columns // 128
    index = row * groups + group
    base = plan["offset"]
    payload.seek(base + index * 2)
    scale_raw = payload.read(2)
    payload.seek(base + rows * groups * 2 + index * 64)
    code_raw = payload.read(64)
    if len(scale_raw) != 2 or len(code_raw) != 64:
        raise GateError("fixed-group blob read is truncated")
    expected_bits, expected_codes = codec._w4_bf16_group_scalar_reference(values)
    if scale_raw != np.asarray([expected_bits], dtype="<u2").tobytes() or code_raw != codec.pack_w4_codes_scalar(expected_codes).tobytes():
        raise GateError(f"tile/scalar bytes disagree at {plan['name']}[{row},{group}]")
    stored_bits = np.frombuffer(scale_raw, dtype="<u2")[0]
    decoded_codes = codec.unpack_w4_codes_scalar(np.frombuffer(code_raw, dtype=np.uint8))
    scalar = codec.dequantize_w4_bf16_group_scalar(stored_bits, decoded_codes)
    vector = np.float32(np.float32(codec.bf16_bits_to_float32(stored_bits)) * decoded_codes.astype(np.float32))
    if not np.array_equal(scalar.view("<u4"), vector.view("<u4")):
        raise GateError("scalar/vector decode differs for fixed group")


def _write_linear(payload: Any, source_handle: Any, plan: Mapping[str, Any], codec: Any) -> int:
    rows, columns = plan["shape"]
    groups = columns // 128
    base = plan["offset"]
    code_base = base + rows * groups * 2
    selected: dict[tuple[int, int], np.ndarray] = {}
    for row_start, source in _source_rows(source_handle, plan["name"], plan["shape"]):
        if source.ndim != 2 or source.shape[1] != columns:
            raise GateError("linear source slice shape mismatch")
        for row, group in ((0, 0), (rows - 1, groups - 1)):
            if row_start <= row < row_start + source.shape[0] and (row, group) not in selected:
                selected[(row, group)] = source[row - row_start, group * 128:(group + 1) * 128].copy()
        try:
            tiles = codec.iter_encode_w4_bf16_tiles((source,), columns, max_rows_per_tile=MAX_TILE_ROWS,
                                                     max_groups_per_tile=MAX_TILE_GROUPS)
            for tile in tiles:
                tile_rows, tile_groups = tile.scale_bits.shape
                for local_row in range(tile_rows):
                    global_row = row_start + tile.row_offset + local_row
                    offset = global_row * groups + tile.group_offset
                    payload.seek(base + offset * 2)
                    payload.write(tile.scale_bits[local_row].astype("<u2", copy=False).tobytes(order="C"))
                    payload.seek(code_base + offset * 64)
                    payload.write(tile.packed_codes[local_row].tobytes(order="C"))
        except ValueError as exc:
            invalid = _find_invalid_group(source, row_start, plan["name"], plan["source_shard"], codec)
            if invalid is not None:
                raise invalid from exc
            raise GateError("tile codec failed but scalar reference found no invalid group") from exc
    payload.flush()
    if not selected or len(selected) != (1 if rows * groups == 1 else 2):
        raise GateError("fixed first/last group selection failed")
    for (row, group), values in selected.items():
        _check_fixed_group(payload, plan, row, group, values, codec)
    return len(selected)


def _write_passthrough(payload: Any, source_handle: Any, plan: Mapping[str, Any]) -> str:
    cursor = plan["offset"]
    digest = hashlib.sha256()
    for _row_start, source in _source_rows(source_handle, plan["name"], plan["shape"]):
        raw = np.ascontiguousarray(source).astype("<f4", copy=False).tobytes(order="C")
        payload.seek(cursor)
        payload.write(raw)
        digest.update(raw)
        cursor += len(raw)
    if cursor != plan["offset"] + plan["length"]:
        raise GateError("F32 passthrough length mismatch")
    payload.flush()
    return digest.hexdigest()


def _write_record(payload: Any, source_handle: Any, plan: Mapping[str, Any], codec: Any, verifier: Any) -> tuple[dict[str, Any], int]:
    if plan["encoding"] == verifier.LINEAR_ENCODING:
        checks = _write_linear(payload, source_handle, plan, codec)
        source_sha = None
    elif plan["encoding"] == verifier.F32_ENCODING:
        checks = 0
        source_sha = _write_passthrough(payload, source_handle, plan)
    else:
        raise GateError("unknown planned encoding")
    digest = _sha_segment(payload, plan["offset"], plan["length"])
    if source_sha is not None and digest != source_sha:
        raise GateError("F32 passthrough changed source bits")
    record = {field: plan[field] for field in ("source_shard", "name", "shape", "source_dtype", "encoding", "offset", "length")}
    record["sha256"] = digest
    return record, checks


def _worker(args: argparse.Namespace) -> int:
    completed = checks = 0
    source_sha = source_stat = fingerprint = report = entry = teacher = None
    try:
        plan_file = json.loads((args.output_dir / "plan_manifest.json").read_text(encoding="utf-8"))
        if plan_file.get("schema") != SCHEMA or _sha_file(args.output_dir / "plan_manifest.json") != args.plan_sha256:
            raise GateError("parent plan changed after launch")
        if plan_file.get("metadata", {}).get("apparatus_sha256") != {name: _sha_file(path) for name, path in SOURCES.items()}:
            raise GateError("protocol/codec/loader/verifier changed after preflight")
        teacher = _load_module("strat02_mmap_teacher.py", "_strat02_export_worker_teacher")
        codec = _load_module("strat02_w4_bf16_codec.py", "_strat02_export_worker_codec")
        verifier = _load_module("strat02_w4_bf16_export_verify.py", "_strat02_export_worker_verifier")
        token_audit = _load_module("strat02_token_audit.py", "_strat02_export_worker_tokens")
        codec.selftest()
        report = teacher.preflight(snapshot=args.snapshot)
        if not report.snapshot.samefile(Path(plan_file["metadata"]["snapshot"])) or _tokenizer_hashes(report.snapshot, token_audit) != plan_file["metadata"]["tokenizer_sha256"]:
            raise GateError("worker pinned snapshot/tokenizer differs from parent")
        if not 0 <= args.shard_index < len(report.manifest["shards"]):
            raise GateError("invalid assigned shard index")
        entry = report.manifest["shards"][args.shard_index]
        shard = entry["name"]
        if plan_file["source_shard_order"][args.shard_index] != shard or plan_file["metadata"]["source_shards"][args.shard_index] != entry:
            raise GateError("worker shard order differs from parent plan")
        planned = plan_file["plan"][shard]
        if [item["name"] for item in planned] != sorted(item["name"] for item in planned) or any(report.index["weight_map"].get(item["name"]) != shard for item in planned):
            raise GateError("worker tensor plan/index mismatch")
        fingerprint = _metadata_fingerprint(report, token_audit)
        source_stat, source_sha = _verify_source_shard(report, entry)
        from safetensors import safe_open
        expected_size = sum(item["length"] for item in planned)
        with args.payload.open("x+b") as payload, args.records.open("x", encoding="utf-8", newline="\n") as records:
            payload.truncate(expected_size)
            with safe_open(str(report.snapshot / shard), framework="pt", device="cpu") as source_handle:
                for item in planned:
                    _assert_source_stat(report, entry, source_stat)
                    record, fixed_checks = _write_record(payload, source_handle, item, codec, verifier)
                    checks += fixed_checks
                    _append_jsonl(records, record)
                    completed += 1
            payload.flush()
            os.fsync(payload.fileno())
            records.flush()
            os.fsync(records.fileno())
        _assert_source_stat(report, entry, source_stat)
        if _metadata_fingerprint(report, token_audit) != fingerprint:
            raise GateError("pinned metadata/tokenizer changed during source reads")
        if args.payload.stat().st_size != expected_size or completed != len(planned):
            raise GateError("worker payload size/record count mismatch")
        descriptor = {"source_shard": shard, "source_sha256": source_sha, "payload_file": args.payload.name,
                      "payload_bytes": expected_size, "payload_sha256": _sha_file(args.payload),
                      "records_file": args.records.name, "records_sha256": _sha_file(args.records),
                      "record_count": completed, "scalar_group_checks": checks}
        verifier.verify_shard(args.output_dir, descriptor, planned)
        _write_json_once(args.result, {"ok": True, "status": "SHARD_COMPLETE", "descriptor": descriptor,
            "shard_index": args.shard_index, "metadata_fingerprint": fingerprint, "utc_timestamp": _utc_now()})
        return 0
    except Exception as exc:
        integrity_type = getattr(teacher, "IntegrityError", ()) if teacher is not None else ()
        status = "VOID_FORMAT" if isinstance(exc, FormatError) else "VOID_APPARATUS" if isinstance(exc, (GateError, ValueError, OSError, integrity_type)) else "INCOMPLETE"
        if report is not None and entry is not None and source_stat is not None and fingerprint is not None:
            try:
                _assert_source_stat(report, entry, source_stat)
                token_audit = _load_module("strat02_token_audit.py", "_strat02_export_failure_tokens")
                if _metadata_fingerprint(report, token_audit) != fingerprint:
                    raise GateError("pinned metadata/tokenizer changed during source reads")
            except Exception as mutation:
                status, exc = "VOID_APPARATUS", mutation
        try:
            _write_json_once(args.result, {"ok": False, "status": status, "shard_index": args.shard_index,
                "source_shard": entry["name"] if entry is not None else None, "source_sha256": source_sha,
                "completed_records": completed, "coordinate": exc.coordinate if status == "VOID_FORMAT" and isinstance(exc, FormatError) else None,
                "payload_file": args.payload.name if args.payload.exists() else None,
                "payload_bytes_partial": args.payload.stat().st_size if args.payload.exists() else None,
                "records_file": args.records.name if args.records.exists() else None,
                "records_sha256_partial": _sha_file(args.records) if args.records.exists() else None,
                "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                "partial_preserved": True, "utc_timestamp": _utc_now()})
        except (OSError, FileExistsError):
            pass
        return 1


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
                   now: Callable[[], float] = time.monotonic, wall_limit: float = MAX_SHARD_WALL,
                   interval: float = SAMPLE_INTERVAL) -> MonitorOutcome:
    start, samples, resource, checked = now(), 0, None, False
    while True:
        elapsed = now() - start
        if elapsed >= wall_limit:
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
            basic, full = resource.memory_info(), resource.memory_full_info()
            private = getattr(full, "private", None)
            if private is None:
                raise GateError("worker private commit unavailable")
            available = int(psutil.virtual_memory().available)
        except Exception as exc:
            exit_code = process.poll()
            if exit_code is not None:
                return MonitorOutcome("CHILD_EXITED", int(exit_code), None, now() - start, samples)
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), f"resource_sample_failed:{exc}", elapsed, samples)
        on_sample({"elapsed_seconds": elapsed, "available_physical_ram_bytes": available,
                   "worker_working_set_bytes": int(basic.rss), "worker_private_commit_bytes": int(private)})
        samples += 1
        if available < MIN_RUNTIME_RAM:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "available_physical_ram_below_4_gib", elapsed, samples)
        if int(private) > MAX_PRIVATE_COMMIT:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "worker_private_commit_above_8_gib", elapsed, samples)
        try:
            process.wait(timeout=min(interval, max(0.0, wall_limit - elapsed)))
        except subprocess.TimeoutExpired:
            pass


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _valid_format_coordinate(coordinate: object, shard: str, planned: Sequence[Mapping[str, Any]]) -> bool:
    if not isinstance(coordinate, dict) or coordinate.get("shard") != shard or coordinate.get("reason") != "invalid BF16 candidate scale":
        return False
    matches = [item for item in planned if item["name"] == coordinate.get("tensor") and item["encoding"] == "strat02_w4_g128_bf16scale_v2"]
    if len(matches) != 1 or type(coordinate.get("row")) is not int or type(coordinate.get("group")) is not int:
        return False
    rows, columns = matches[0]["shape"]
    return 0 <= coordinate["row"] < rows and 0 <= coordinate["group"] < columns // 128


def _run_parent(args: argparse.Namespace, *, smoke: bool) -> int:
    start = time.monotonic()
    paths = _prepare_output(args.output_dir)
    try:
        bounded, _teacher, preflight = _launch_preflight(args.snapshot, paths["root"])
        plan_document = {"schema": SCHEMA, "mode": "shard-smoke" if smoke else "run",
            "metadata": preflight["metadata"], "source_shard_order": preflight["source_shard_order"],
            "plan": preflight["plan"], "expected_ledger": preflight["ledger"], "organ_counts": preflight["organ_counts"],
            "caps": {"launch_ram_min": MIN_LAUNCH_RAM, "launch_disk_min": MIN_LAUNCH_DISK,
                     "runtime_ram_min": MIN_RUNTIME_RAM, "private_commit_max": MAX_PRIVATE_COMMIT,
                     "shard_wall_seconds": MAX_SHARD_WALL, "total_wall_seconds": MAX_TOTAL_WALL,
                     "monitor_seconds": SAMPLE_INTERVAL, "max_rows": MAX_ROWS,
                     "max_tile_rows": MAX_TILE_ROWS, "max_tile_groups": MAX_TILE_GROUPS},
            "auto_resume": False, "heldout_access": False, "t4": False}
        _write_json_once(paths["plan"], plan_document)
        plan_sha = _sha_file(paths["plan"])
        _append_log(paths["log"], {"event": "preflight_complete", "plan_sha256": plan_sha})
    except Exception as exc:
        _write_json_once(paths["plan"], {"schema": SCHEMA, "preflight": "failed"})
        _write_json_once(paths["result"], {"ok": False, "status": "INCOMPLETE",
            "failure_classification": "VOID_RESOURCE" if isinstance(exc, ResourceError) else "VOID_APPARATUS",
            "error_type": type(exc).__name__, "error": str(exc), "utc_timestamp": _utc_now()})
        return 1
    completed: list[dict[str, Any]] = []
    classification = reason = coordinate = None
    try:
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        worker_python = bounded._direct_worker_python(environment)
        token = secrets.token_urlsafe(32)
        environment[WORKER_TOKEN_ENV] = token
        source_order = preflight["source_shard_order"][:1 if smoke else 11]
        verifier = _load_module("strat02_w4_bf16_export_verify.py", "_strat02_export_parent_verifier")
        for shard_index, shard in enumerate(source_order):
            remaining = MAX_TOTAL_WALL - (time.monotonic() - start)
            if remaining <= 0:
                classification, reason = "VOID_RESOURCE", "total_wall_clock_exceeded"
                break
            stem = f"shard_{shard_index + 1:02d}"
            payload = paths["root"] / f"{stem}.payload"
            records = paths["root"] / f"{stem}.records.jsonl"
            worker_result = paths["root"] / f"{stem}.worker.json"
            stdout_path, stderr_path = paths["root"] / f"{stem}.stdout.log", paths["root"] / f"{stem}.stderr.log"
            command = [worker_python, str(Path(__file__).resolve()), "--worker", "--snapshot", preflight["metadata"]["snapshot"],
                "--output-dir", str(paths["root"]), "--shard-index", str(shard_index), "--payload", str(payload),
                "--records", str(records), "--result", str(worker_result), "--plan-sha256", plan_sha, "--worker-token", token]
            _append_log(paths["log"], {"event": "shard_launch", "shard": shard, "termination_scope": "direct_worker_pid_only"})
            process = None
            try:
                with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
                    process = subprocess.Popen(command, cwd=str(HERE), env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                    import psutil
                    outcome = _monitor_child(process, psutil=psutil, expected_executable=Path(worker_python),
                        wall_limit=min(MAX_SHARD_WALL, remaining),
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
            if time.monotonic() - start >= MAX_TOTAL_WALL:
                classification, reason = "VOID_RESOURCE", "total_wall_clock_exceeded"
                break
            if worker and worker.get("status") == "VOID_FORMAT":
                coordinate = worker.get("coordinate")
                if (outcome.exit_code != 1 or not _valid_format_coordinate(coordinate, shard, preflight["plan"][shard]) or
                    worker.get("source_sha256") != preflight["metadata"]["source_shards"][shard_index]["sha256"] or
                    worker.get("shard_index") != shard_index or worker.get("ok") is not False):
                    classification, reason = "VOID_APPARATUS", "malformed format counterexample record"
                else:
                    classification, reason = "VOID_FORMAT", "invalid BF16 scale at fixed source coordinate"
                break
            if outcome.exit_code != 0 or not worker or worker.get("ok") is not True or worker.get("status") != "SHARD_COMPLETE":
                classification = worker.get("status") if worker and worker.get("status") in {"VOID_APPARATUS", "INCOMPLETE"} else "INCOMPLETE"
                reason = worker.get("error") if worker else "worker_result_missing_or_bad"
                break
            descriptor = worker.get("descriptor")
            if not isinstance(descriptor, dict) or worker.get("shard_index") != shard_index or descriptor.get("source_shard") != shard:
                raise GateError("worker shard identity/result mismatch")
            if descriptor.get("source_sha256") != preflight["metadata"]["source_shards"][shard_index]["sha256"]:
                raise GateError("worker source shard SHA differs from pinned manifest")
            if descriptor.get("record_count") != len(preflight["plan"][shard]) or descriptor.get("scalar_group_checks") != sum(
                    1 if item["shape"][0] * (item["shape"][1] // 128) == 1 else 2 for item in preflight["plan"][shard]
                    if item["encoding"] == verifier.LINEAR_ENCODING):
                raise GateError("worker record/fixed-group check count mismatch")
            verifier.verify_shard(paths["root"], descriptor, preflight["plan"][shard])
            completed.append(descriptor)
        if classification is None:
            _write_json_once(paths["artifact"], {"schema": SCHEMA, "mode": "shard-smoke" if smoke else "run",
                "plan_file": paths["plan"].name, "plan_sha256": plan_sha, "shards": completed,
                "source_revision": preflight["metadata"]["revision"]})
            summary = verifier.verify_artifact(paths["root"], paths["artifact"], expected_plan=preflight["plan"], require_full=not smoke)
            status = "PARTIAL_APPARATUS" if smoke else "COMPLETE"
            verdict = None if smoke else "EXPORT_VERIFIED"
        else:
            status, verdict, summary = "INCOMPLETE", None, None
    except (GateError, ValueError, OSError) as exc:
        status, verdict, summary, classification, reason = "INCOMPLETE", None, None, "VOID_APPARATUS", str(exc)
        _append_log(paths["log"], {"event": "parent_apparatus_error", "traceback": traceback.format_exc()})
    except Exception as exc:
        status, verdict, summary, classification, reason = "INCOMPLETE", None, None, "INCOMPLETE", f"{type(exc).__name__}: {exc}"
        _append_log(paths["log"], {"event": "parent_error", "traceback": traceback.format_exc()})
    final = {"ok": status in {"PARTIAL_APPARATUS", "COMPLETE"}, "status": status, "verdict": verdict,
             "failure_classification": classification, "reason": reason, "coordinate": coordinate,
             "completed_shards": completed, "verification": summary, "artifact_manifest": paths["artifact"].name if paths["artifact"].exists() else None,
             "partial_preserved": True, "utc_timestamp": _utc_now()}
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": final["ok"], "status": status, "failure_classification": classification,
                      "result": str(paths["result"])}, sort_keys=True))
    return 0 if final["ok"] else 1


def _selftest() -> None:
    codec = _load_module("strat02_w4_bf16_codec.py", "_strat02_export_selftest_codec")
    codec.selftest()
    verifier = _load_module("strat02_w4_bf16_export_verify.py", "_strat02_export_selftest_verifier")
    import torch
    class Slice:
        def __init__(self, value: Any): self.value = value
        def get_shape(self): return tuple(self.value.shape)
        def __getitem__(self, rows): return self.value[rows]
    class Handle:
        def __init__(self, values: Mapping[str, Any]): self.values = values
        def get_slice(self, key): return Slice(self.values[key])
        def get_tensor(self, key): return self.values[key]
    rng = np.random.default_rng(20260917)
    weights = torch.from_numpy(rng.normal(size=(3, 256)).astype(np.float32))
    passthrough = torch.tensor([-0.0, 1.0, -2.0], dtype=torch.float32)
    with tempfile.TemporaryDirectory(prefix="strat02-bf16-export-selftest-") as temporary:
        root = Path(temporary)
        source = Handle({"a.weight": weights, "b.bias": passthrough})
        entries = [{"source_shard": "synthetic", "name": "a.weight", "shape": [3, 256], "source_dtype": "float32",
                    "encoding": verifier.LINEAR_ENCODING, "offset": 0, "length": 3 * 2 * 66},
                   {"source_shard": "synthetic", "name": "b.bias", "shape": [3], "source_dtype": "float32",
                    "encoding": verifier.F32_ENCODING, "offset": 3 * 2 * 66, "length": 12}]
        path = root / "shard_01.payload"
        with path.open("x+b") as payload:
            payload.truncate(sum(item["length"] for item in entries))
            records = [_write_record(payload, source, item, codec, verifier)[0] for item in entries]
            payload.flush()
        records_path = root / "shard_01.records.jsonl"
        with records_path.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                _append_jsonl(handle, record)
        descriptor = {"source_shard": "synthetic", "payload_file": path.name, "payload_bytes": path.stat().st_size,
                      "payload_sha256": _sha_file(path), "records_file": records_path.name,
                      "records_sha256": _sha_file(records_path)}
        summary = verifier.verify_shard(root, descriptor, entries)
        if summary["records"] != 2 or summary["groups"] != 6 or summary["f32_bytes"] != 12:
            raise AssertionError("synthetic raw container accounting failed")
    print(json.dumps({"ok": True, "selftest": True, "donor_weights_opened": False}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selftest", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--shard-smoke", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--records", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.preflight:
            _bounded, _teacher, _report, preflight = _preflight(args.snapshot)
            print(json.dumps({"ok": True, "status": "PREFLIGHT_COMPLETE", "snapshot": preflight["metadata"]["snapshot"],
                              "revision": preflight["metadata"]["revision"], "runtime_versions": preflight["metadata"]["runtime_versions"],
                              "tokenizer_sha256": preflight["metadata"]["tokenizer_sha256"], "ledger": preflight["ledger"]}, sort_keys=True))
            return 0
        if args.worker:
            if (not args.worker_token or not secrets.compare_digest(args.worker_token, os.environ.get(WORKER_TOKEN_ENV, "")) or
                args.snapshot is None or args.output_dir is None or args.shard_index is None or args.payload is None or
                args.records is None or args.result is None or args.plan_sha256 is None):
                raise GateError("internal worker token/arguments invalid")
            root = args.output_dir.resolve()
            if args.payload.parent.resolve() != root or args.records.parent.resolve() != root or args.result.parent.resolve() != root:
                raise GateError("internal worker output paths escape result directory")
            return _worker(args)
        if args.output_dir is None:
            raise GateError("--shard-smoke/--run requires a fresh --output-dir")
        return _run_parent(args, smoke=args.shard_smoke)
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "INCOMPLETE", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
