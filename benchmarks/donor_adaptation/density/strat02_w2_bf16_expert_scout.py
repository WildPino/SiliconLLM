#!/usr/bin/env python3
"""Bounded STRAT-02E calibration-only W2-BF16 expert scout.

``--preflight`` is deliberately read-only: it reads pins, metadata and the
published W4 calibration control but never opens a donor safetensors value.
Only ``--run`` creates a fresh write-once output and a direct, monitored
worker.  The worker has one W4-derived F32 arena, maps one donor shard at a
time, and never opens a heldout path.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from benchmarks.donor_adaptation.density import strat02_bounded_smoke as bounded
from benchmarks.donor_adaptation.density import strat02_mmap_teacher as teacher
from benchmarks.donor_adaptation.density import strat02_score as score_core
from benchmarks.donor_adaptation.density import strat02_token_audit as audit
from benchmarks.donor_adaptation.density import strat02_w2_bf16_codec as codec
from benchmarks.donor_adaptation.density import strat02_w4_bf16_candidate_loader as candidate
from benchmarks.donor_adaptation.density import strat02_w4_bf16_export_verify as verifier
from benchmarks.donor_adaptation.density import strat02_w4_organ_attribution as strat02d


SCHEMA = "strat02_w2_bf16_expert_scout_v1"
BRIEF_COMMIT = "8149a5e"
BRIEF_RELATIVE = "docs/research/donor_adaptation/briefs/BRIEF_STRAT_02E_W2_BF16_EXPERT_SCOUT.md"
CALIB = HERE / "corpus" / "strat02_document_holdout_v1" / "calib.jsonl"
W4_CONTROL = HERE / "results" / "strat02_w4_bf16_quality_20260917_162329" / "calibration_checks.jsonl"
W4_CONTROL_SHA256 = "ef1d7f42ad30ff5727432bcc4f0c1a5299ff681f52a5a7690f55602f0171a142"
ROUTER_F32_CONTROL = HERE / "results" / "strat02_w4_organ_attribution_20260917_183751" / "router_f32.jsonl"
ROUTER_F32_CONTROL_SHA256 = "f7913767de5c65396d90473cfd7302961c615ec09bcc0eeb65b3ff428a78550b"
W4_EXPORT = HERE / "results" / "strat02_w4_bf16_full_export_20260917_143753"
W4_EXPORT_PINS = {
    "artifact_manifest.json": "06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c",
    "plan_manifest.json": "1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42",
    "supervisor_result.json": "d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca",
}
ARMS = ("W2_EXPERTS_ROUTER_W4", "W2_EXPERTS_ROUTER_F32")
EXPERT_TENSORS = 6_144
ROUTER_TENSORS = 16
EXPERT_STORED_WEIGHTS = 12_884_901_888
W2_ACTIVE_BYTES = 473_038_848
W2_ROUTER_F32_ACTIVE_BYTES = 487_653_376
MIN_LAUNCH_RAM = 55 * 1024**3
MIN_OUTPUT_DISK = 1 * 1024**3
MIN_RUNTIME_RAM = 8 * 1024**3
MAX_WORKER_MEMORY = 70 * 1024**3
MAX_WALL_SECONDS = 6 * 60 * 60
PROJECTION_SECONDS_MAX = 3 * 60 * 60
MONITOR_SECONDS = 5.0
MAX_ROWS = 16
MAX_GROUPS = 16
SENTINEL_TOLERANCE_BITS = 1e-5
WORKER_TOKEN_ENV = "STRAT02E_W2_EXPERT_SCOUT_PARENT_TOKEN"
PROVENANCE = {
    "runner": Path(__file__).resolve(),
    "codec": HERE / "strat02_w2_bf16_codec.py",
    "candidate_loader": HERE / "strat02_w4_bf16_candidate_loader.py",
    "teacher": HERE / "strat02_mmap_teacher.py",
    "score_core": HERE / "strat02_score.py",
    "strat02d_control": HERE / "strat02_w4_organ_attribution.py",
    "bounded_supervisor": HERE / "strat02_bounded_smoke.py",
}


class ApparatusError(RuntimeError):
    """An identity, score, rollback, or wire-format invariant failed."""


class ResourceError(MemoryError):
    """A preregistered physical-resource or runtime-projection gate failed."""


class FormatError(ApparatusError):
    """A source group cannot represent the frozen W2 BF16 wire format."""


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        handle.write("\n")


def _log(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"utc_timestamp": _utc(), **event}, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _frozen_brief() -> bytes:
    try:
        return subprocess.check_output(["git", "show", f"{BRIEF_COMMIT}:{BRIEF_RELATIVE}"], cwd=PROJECT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ApparatusError(f"cannot read frozen brief {BRIEF_COMMIT}") from exc


def _provenance() -> dict[str, str]:
    result = {name: _sha_file(path) for name, path in PROVENANCE.items()}
    result["brief_frozen_commit"] = hashlib.sha256(_frozen_brief()).hexdigest()
    return result


def _existing_volume(path: Path | None) -> Path:
    volume = Path(path or HERE).expanduser().resolve()
    while not volume.exists():
        if volume.parent == volume:
            raise ApparatusError("output path has no existing ancestor")
        volume = volume.parent
    if not volume.is_dir():
        raise ApparatusError("output ancestor is not a directory")
    return volume


def _require_kernel_policy() -> None:
    present = os.environ.get("USE_HUB_KERNELS")
    if present is None:
        os.environ["USE_HUB_KERNELS"] = "NO"
    elif present != "NO":
        raise ApparatusError(f"USE_HUB_KERNELS must be NO, got {present!r}")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApparatusError(f"expected JSON object: {path.name}")
    return value


def _check_static_pins() -> dict[str, str]:
    observed = {name: _sha_file(W4_EXPORT / name) for name in W4_EXPORT_PINS}
    if observed != W4_EXPORT_PINS:
        raise ApparatusError("pinned W4-v2 export controls do not match")
    if _sha_file(W4_CONTROL) != W4_CONTROL_SHA256:
        raise ApparatusError("pinned W4 calibration control does not match")
    if _sha_file(ROUTER_F32_CONTROL) != ROUTER_F32_CONTROL_SHA256:
        raise ApparatusError("pinned STRAT-02D ROUTER_F32 control does not match")
    return {**{f"w4_export:{name}": value for name, value in observed.items()},
            "w4_calibration": W4_CONTROL_SHA256, "router_f32": ROUTER_F32_CONTROL_SHA256}


def _read_calibration_and_controls() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Open calibration and published controls only; never touch heldout."""
    rows = strat02d._read_calibration()
    w4 = strat02d._baseline(rows)
    router = [json.loads(line) for line in ROUTER_F32_CONTROL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(router) != 48:
        raise ApparatusError("ROUTER_F32 control must have exactly 48 rows")
    for ordinal, (row, item) in enumerate(zip(rows, router, strict=True), 1):
        if (item.get("source_document_id") != row["source_document_id"] or item.get("category") != row["category"] or
                item.get("bytes") != len(row["text"].encode("utf-8")) or type(item.get("tokens")) is not int or
                item["tokens"] <= 0 or type(item.get("bits")) not in (int, float) or not math.isfinite(float(item["bits"]))):
            raise ApparatusError(f"ROUTER_F32 control identity mismatch at row {ordinal}")
    return rows, w4, router


def _project_seconds(values_per_second: float) -> float:
    if not math.isfinite(values_per_second) or values_per_second <= 0:
        raise ResourceError("synthetic encoder throughput is non-finite or zero")
    return EXPERT_STORED_WEIGHTS / values_per_second


def synthetic_speed_diagnostic() -> dict[str, Any]:
    """Frozen cost guard: one warmup, then three timed fit/pack/decode reps."""
    values = np.random.default_rng(20260917).normal(size=4_194_304).astype(np.float32).reshape(-1, 2048)

    def convert() -> None:
        groups = values.shape[1] // codec.GROUP_SIZE
        a = np.empty((values.shape[0], groups), dtype=np.uint16)
        b = np.empty_like(a)
        packed = np.empty((values.shape[0], groups * codec.CODE_BYTES_PER_GROUP), dtype=np.uint8)
        decoded = np.empty_like(values)
        for tile in codec.iter_encode_w2_bf16_tiles((values,), values.shape[1], max_rows_per_tile=MAX_ROWS, max_groups_per_tile=MAX_GROUPS):
            rows, tile_groups = tile.a_bits.shape
            target_rows = slice(tile.row_offset, tile.row_offset + rows)
            a[target_rows, tile.group_offset:tile.group_offset + tile_groups] = tile.a_bits
            b[target_rows, tile.group_offset:tile.group_offset + tile_groups] = tile.b_bits
            packed[target_rows, tile.group_offset * codec.CODE_BYTES_PER_GROUP:(tile.group_offset + tile_groups) * codec.CODE_BYTES_PER_GROUP] = tile.packed_codes
            decoded[target_rows, tile.group_offset * codec.GROUP_SIZE:(tile.group_offset + tile_groups) * codec.GROUP_SIZE] = codec.decode_w2_bf16_tile(tile)
        if not np.all(np.isfinite(decoded)):
            raise ApparatusError("synthetic packed decode is non-finite")

    convert()  # exactly one warmup; setup/allocation is outside measured conversion.
    seconds: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        convert()
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ResourceError("synthetic timed repetition has invalid elapsed time")
        seconds.append(elapsed)
    throughputs = [values.size / item for item in seconds]
    minimum = min(throughputs)
    projected = _project_seconds(minimum)
    return {"seed": 20260917, "values": int(values.size), "shape": list(values.shape), "warmups": 1,
            "timed_repetitions": 3, "timed_seconds": seconds, "values_per_second": throughputs,
            "minimum_values_per_second": minimum, "expert_stored_weights": EXPERT_STORED_WEIGHTS,
            "projected_expert_encode_seconds": projected, "projection_limit_seconds": PROJECTION_SECONDS_MAX,
            "includes": ["fit", "pack", "decode"], "excludes": ["I/O"]}


def _read_only_preflight(snapshot: Path | None, output_dir: Path | None) -> dict[str, Any]:
    """Metadata/control preflight; it cannot open donor tensor values."""
    _require_kernel_policy()
    codec.selftest()
    pins = _check_static_pins()
    rows, _w4, _router = _read_calibration_and_controls()
    versions = bounded._pinned_environment()
    report = teacher.preflight(snapshot=snapshot)
    status = report.source_status
    if (status.get("repository") != teacher.REPOSITORY or status.get("revision") != teacher.REVISION or
            not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO" or not status.get("shards_present")):
        raise ApparatusError("pinned donor metadata/runtime status is invalid")
    import psutil
    volume = _existing_volume(output_dir)
    available, free = int(psutil.virtual_memory().available), int(psutil.disk_usage(str(volume)).free)
    if available < MIN_LAUNCH_RAM:
        raise ResourceError(f"launch requires >=55 GiB available RAM; observed {available}")
    if free < MIN_OUTPUT_DISK:
        raise ResourceError(f"launch requires >=1 GiB output free space; observed {free}")
    speed = synthetic_speed_diagnostic()
    launch_status = "READY" if speed["projected_expert_encode_seconds"] <= PROJECTION_SECONDS_MAX else "NOT_RUN_RESOURCE_PROJECTION"
    return {"status": launch_status, "utc_timestamp": _utc(), "snapshot": str(report.snapshot),
            "runtime_versions": versions, "source_status": dict(status), "static_pins": pins,
            "frozen_brief_commit": BRIEF_COMMIT, "frozen_brief_sha256": hashlib.sha256(_frozen_brief()).hexdigest(),
            "calibration_documents": len(rows), "heldout_access": False,
            "source_f32_values_opened": False, "source_payload_hashes": "not performed by preflight",
            "available_physical_ram_bytes": available, "output_volume": str(volume), "output_free_bytes": free,
            "speed_diagnostic": speed}


def _prepare_output(root: Path) -> dict[str, Path]:
    root = root.resolve()
    if root.exists():
        raise ApparatusError("output directory must be new")
    root.mkdir(parents=True, exist_ok=False)
    paths = {"root": root, "manifest": root / "supervisor_manifest.json", "log": root / "supervisor_log.jsonl",
             "worker": root / "worker_result.json", "result": root / "supervisor_result.json",
             "stdout": root / "worker_stdout.log", "stderr": root / "worker_stderr.log",
             "tile_audit": root / "w2_tile_audit.jsonl"}
    with paths["log"].open("x", encoding="utf-8", newline="\n"):
        pass
    return paths


def _tensor_sha(tensor: Any) -> str:
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ApparatusError("tensor hash requires contiguous CPU F32")
    return hashlib.sha256(memoryview(tensor.detach().numpy()).cast("B")).hexdigest()


def _tile_wire(tile: codec.W2BF16EncodedTile) -> bytes:
    return (tile.a_bits.astype("<u2", copy=False).tobytes(order="C") +
            tile.b_bits.astype("<u2", copy=False).tobytes(order="C") + tile.packed_codes.tobytes(order="C"))


def _planned_organs(plan: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    experts, routers, names = [], [], set()
    for records in plan.values():
        for record in records:
            if record["name"] in names:
                raise ApparatusError("duplicate candidate plan tensor")
            names.add(record["name"])
            if record.get("encoding") == verifier.LINEAR_ENCODING and record.get("organ") == "expert":
                experts.append(record)
            elif record.get("encoding") == verifier.LINEAR_ENCODING and record.get("organ") == "router":
                routers.append(record)
    if len(experts) != EXPERT_TENSORS or len(routers) != ROUTER_TENSORS:
        raise ApparatusError("frozen expert/router tensor inventory differs")
    if sum(int(item["shape"][0]) * int(item["shape"][1]) for item in experts) != EXPERT_STORED_WEIGHTS:
        raise ApparatusError("frozen expert stored-weight inventory differs")
    return experts, routers


def _source_open(path: Path) -> Any:
    from safetensors import safe_open
    return safe_open(str(path), framework="pt", device="cpu")


def _replace_experts_w2(*, report: Any, verified: Any, experts: Sequence[Mapping[str, Any]], state: Mapping[str, Any], tile_audit: Path) -> dict[str, str]:
    """Encode/decode each expert into the preallocated F32 arena, one source shard at a time."""
    grouped = {entry["name"]: [] for entry in report.manifest["shards"]}
    for record in experts:
        shard = record.get("source_shard")
        if shard not in grouped or report.index["weight_map"].get(record["name"]) != shard or record["name"] not in state:
            raise ApparatusError("expert plan/source/state binding differs")
        grouped[shard].append(record)
    decoded_hashes: dict[str, str] = {}
    with tile_audit.open("x", encoding="utf-8", newline="\n") as audit_handle:
        for entry in report.manifest["shards"]:
            shard = entry["name"]
            if not grouped[shard]:
                continue
            with _source_open(report.snapshot / shard) as source:
                if set(source.keys()) != {key for key, mapped in report.index["weight_map"].items() if mapped == shard}:
                    raise ApparatusError("source shard keyset differs from pinned index")
                for record in grouped[shard]:
                    name, shape = record["name"], tuple(record["shape"])
                    target = state[name]
                    if tuple(target.shape) != shape or target.dtype.__str__() != "torch.float32":
                        raise ApparatusError("expert target is not planned F32 arena view")
                    source_slice = source.get_slice(name)
                    if tuple(source_slice.get_shape()) != shape:
                        raise ApparatusError("source expert shape differs from plan")
                    target_array = target.detach().numpy()
                    tile_count = 0
                    for start in range(0, shape[0], MAX_ROWS):
                        stop = min(start + MAX_ROWS, shape[0])
                        batch = source_slice[start:stop]
                        if batch.dtype.__str__() != "torch.float32" or tuple(batch.shape) != (stop - start, shape[1]):
                            raise ApparatusError("source expert batch is not bounded CPU F32")
                        values = batch.numpy()
                        try:
                            for tile in codec.iter_encode_w2_bf16_tiles((values,), shape[1], max_rows_per_tile=MAX_ROWS, max_groups_per_tile=MAX_GROUPS):
                                decoded = codec.decode_w2_bf16_tile(tile)
                                target_array[start + tile.row_offset:start + tile.row_offset + decoded.shape[0],
                                             tile.group_offset * codec.GROUP_SIZE:(tile.group_offset + tile.a_bits.shape[1]) * codec.GROUP_SIZE] = decoded
                                wire = _tile_wire(tile)
                                audit_handle.write(json.dumps({"tensor": name, "source_shard": shard, "row_offset": start + tile.row_offset,
                                    "group_offset": tile.group_offset, "rows": int(tile.a_bits.shape[0]), "groups": int(tile.a_bits.shape[1]),
                                    "wire_bytes": len(wire), "wire_sha256": hashlib.sha256(wire).hexdigest()}, sort_keys=True) + "\n")
                                tile_count += 1
                        except ValueError as exc:
                            raise FormatError(f"invalid W2 group in {name} at source row {start}") from exc
                        del values, batch
                    if tile_count == 0:
                        raise ApparatusError("expert had no encoded tiles")
                    decoded_hashes[name] = _tensor_sha(target)
            teacher._assert_unchanged(verified)
    return decoded_hashes


def _check_observation(observed: Any, expected: Mapping[str, Any]) -> None:
    if ((observed.source_document_id, observed.category, observed.tokens, observed.bytes) !=
            (expected["source_document_id"], expected["category"], expected["tokens"], expected["bytes"]) or
            not math.isfinite(float(observed.bits)) or not math.isclose(float(observed.bits), float(expected["bits"]), rel_tol=0.0, abs_tol=SENTINEL_TOLERANCE_BITS)):
        raise ApparatusError("sentinel identity or bits differ")


def _score_arm(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], output: Path) -> dict[str, Any]:
    return strat02d._score_arm(model, tokenizer, rows, output, score_core.score_document)


def _paired_delta(score_path: Path, control: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute a fixed paired BPB diagnostic without reading a partial score."""
    scored = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(scored) != 48 or len(control) != 48 or len(rows) != 48:
        raise ApparatusError("paired comparison requires two complete 48-row arms")
    totals = {"all": {"bits": 0.0, "bytes": 0},
              **{category: {"bits": 0.0, "bytes": 0} for category in ("code", "prose", "technical_general")}}
    for ordinal, (actual, expected, row) in enumerate(zip(scored, control, rows, strict=True), 1):
        binding = (row["source_document_id"], row["category"], len(row["text"].encode("utf-8")))
        if ((actual.get("source_document_id"), actual.get("category"), actual.get("bytes")) != binding or
                (expected.get("source_document_id"), expected.get("category"), expected.get("bytes")) != binding or
                type(actual.get("tokens")) is not int or actual["tokens"] != expected.get("tokens") or
                type(actual.get("bits")) not in (int, float) or type(expected.get("bits")) not in (int, float) or
                not math.isfinite(float(actual["bits"])) or not math.isfinite(float(expected["bits"]))):
            raise ApparatusError(f"paired identity/token/finite mismatch at row {ordinal}")
        delta = float(actual["bits"]) - float(expected["bits"])
        for bucket in (totals["all"], totals[row["category"]]):
            bucket["bits"] += delta
            bucket["bytes"] += binding[2]
    return {name: {"delta_bits": item["bits"], "delta_bpb": item["bits"] / item["bytes"], "bytes": item["bytes"]}
            for name, item in totals.items()}


def _restore_w4_records(records: Iterable[Mapping[str, Any]], prepared: Any, state: Mapping[str, Any], expected: Mapping[str, str]) -> None:
    for record in records:
        descriptor = next((item for item in prepared.descriptors if item["source_shard"] == record["source_shard"]), None)
        if descriptor is None:
            raise ApparatusError("W4 payload descriptor missing")
        with verifier._safe_child(prepared.root, descriptor["payload_file"]).open("rb") as handle:
            candidate.decode_record(handle, record, destination=state[record["name"]])
        if _tensor_sha(state[record["name"]]) != expected[record["name"]]:
            raise ApparatusError(f"W4 rollback SHA mismatch: {record['name']}")


def _copy_router_f32(records: Sequence[Mapping[str, Any]], state: Mapping[str, Any], report: Any) -> None:
    # STRAT-02D's bounded copier opens at most one source shard and verifies bit SHA.
    strat02d._copy_arm_sources(records, state, report)


def _worker(args: argparse.Namespace) -> int:
    current_arm: str | None = None
    paths = {"tile_audit": args.output_dir / "w2_tile_audit.jsonl"}
    try:
        _require_kernel_policy()
        manifest = _json(args.supervisor_manifest)
        if manifest.get("schema") != SCHEMA or manifest.get("provenance_sha256") != _provenance():
            raise ApparatusError("worker provenance differs from parent manifest")
        if manifest.get("preflight", {}).get("status") != "READY":
            raise ResourceError("worker launch without READY preflight is forbidden")
        rows, w4_control, router_control = _read_calibration_and_controls()
        report = teacher.preflight(snapshot=Path(manifest["preflight"]["snapshot"]))
        tokenizer = strat02d._tokenizer_and_rows(report.snapshot, rows)
        # All eleven pinned hashes are checked before the first donor value read.
        verified = teacher.verify_shards(report)
        prepared = candidate.prepare_candidate(W4_EXPORT, snapshot=report.snapshot, output_dir=args.output_dir)
        experts, routers = _planned_organs(prepared.plan)
        # The candidate quality loader hashes heldout-derived teacher scores.
        # This diagnostic is calibration-only, so use STRAT-02D's equivalent
        # verified arena loader that deliberately omits that prerequisite.
        with strat02d._load_calibration_candidate(prepared) as model:
            model.eval()
            state = model.state_dict()
            w4_hashes = {item["name"]: _tensor_sha(state[item["name"]]) for item in (*experts, *routers)}
            sentinel = rows[0]
            _check_observation(score_core.score_document(model, tokenizer, sentinel, chunk_size=128), w4_control[0])
            decoded_hashes = _replace_experts_w2(report=report, verified=verified, experts=experts, state=state, tile_audit=paths["tile_audit"])
            teacher._assert_unchanged(verified)
            current_arm = ARMS[0]
            first_path = args.output_dir / f"{current_arm.lower()}.jsonl"
            first = _score_arm(model, tokenizer, rows, first_path)
            first_vs_w4 = _paired_delta(first_path, w4_control, rows)
            first_vs_router = _paired_delta(first_path, router_control, rows)
            w2_sentinel = score_core.score_document(model, tokenizer, sentinel, chunk_size=128)
            current_arm = ARMS[1]
            _copy_router_f32(routers, state, report)
            teacher._assert_unchanged(verified)
            second_path = args.output_dir / f"{current_arm.lower()}.jsonl"
            second = _score_arm(model, tokenizer, rows, second_path)
            second_vs_w4 = _paired_delta(second_path, w4_control, rows)
            second_vs_router = _paired_delta(second_path, router_control, rows)
            _restore_w4_records(routers, prepared, state, w4_hashes)
            _check_observation(score_core.score_document(model, tokenizer, sentinel, chunk_size=128), {
                "source_document_id": w2_sentinel.source_document_id, "category": w2_sentinel.category,
                "tokens": w2_sentinel.tokens, "bytes": w2_sentinel.bytes, "bits": w2_sentinel.bits})
            _restore_w4_records(experts, prepared, state, w4_hashes)
            _check_observation(score_core.score_document(model, tokenizer, sentinel, chunk_size=128), w4_control[0])
        result = {"ok": True, "status": "COMPLETE_DIAGNOSTIC", "schema": SCHEMA, "utc_timestamp": _utc(),
                  "arms": [{"arm": ARMS[0], "score_file": first_path.name, "score_sha256": _sha_file(first_path), "score": first,
                            "active_bytes_per_token": W2_ACTIVE_BYTES, "paired_vs_w4": first_vs_w4,
                            "paired_vs_strat02d_router_f32": first_vs_router},
                           {"arm": ARMS[1], "score_file": second_path.name, "score_sha256": _sha_file(second_path), "score": second,
                            "active_bytes_per_token": W2_ROUTER_F32_ACTIVE_BYTES, "paired_vs_w4": second_vs_w4,
                            "paired_vs_strat02d_router_f32": second_vs_router}],
                  "expert_tensors": EXPERT_TENSORS, "router_tensors": ROUTER_TENSORS,
                  "w2_decoded_tensor_sha256": decoded_hashes, "tile_audit_sha256": _sha_file(paths["tile_audit"]),
                  "source_shards_verified": 11, "source_sha256": {item["name"]: item["sha256"] for item in report.manifest["shards"]},
                  "w4_payload_sha256": {item["payload_file"]: item["payload_sha256"] for item in prepared.descriptors},
                  "controls": {"w4": W4_CONTROL_SHA256, "router_f32": ROUTER_F32_CONTROL_SHA256},
                  "sentinel_w4_before": True, "sentinel_w2_router_rollback": True, "sentinel_w4_final": True,
                  "heldout_access": False, "model_copies": 1, "partial_preserved": True}
        _write_once(args.worker_result, result)
        return 0
    except FormatError as exc:
        failure = {"status": "VOID_FORMAT", "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc()}
    except ResourceError as exc:
        failure = {"status": "VOID_RESOURCE", "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc()}
    except MemoryError as exc:
        failure = {"status": "VOID_RESOURCE", "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc()}
    except Exception as exc:
        failure = {"status": "VOID_APPARATUS", "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc()}
    try:
        _write_once(args.worker_result, {"ok": False, "schema": SCHEMA, "utc_timestamp": _utc(),
                    "current_arm": current_arm, "heldout_access": False, "partial_preserved": True,
                    **failure,
                    "tile_audit_sha256_partial": _sha_file(paths["tile_audit"]) if paths["tile_audit"].is_file() else None})
    except OSError:
        pass
    return 1


def _validate_worker(root: Path, worker: Mapping[str, Any]) -> None:
    if worker.get("status") != "COMPLETE_DIAGNOSTIC" or not worker.get("ok") or worker.get("heldout_access") is not False:
        raise ApparatusError("worker did not complete frozen diagnostic")
    arms = worker.get("arms")
    if not isinstance(arms, list) or [item.get("arm") for item in arms] != list(ARMS):
        raise ApparatusError("worker arm order differs from frozen protocol")
    for arm in arms:
        file = root / arm["score_file"]
        if not file.is_file() or _sha_file(file) != arm.get("score_sha256") or sum(1 for line in file.read_text(encoding="utf-8").splitlines() if line.strip()) != 48:
            raise ApparatusError("write-once arm score file is invalid")
    if not (root / "w2_tile_audit.jsonl").is_file() or _sha_file(root / "w2_tile_audit.jsonl") != worker.get("tile_audit_sha256"):
        raise ApparatusError("W2 tile audit hash mismatch")


def _run(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    try:
        preflight = _read_only_preflight(args.snapshot, paths["root"])
        if preflight["status"] == "NOT_RUN_RESOURCE_PROJECTION":
            provenance = _provenance()
            _write_once(paths["manifest"], {"schema": SCHEMA, "utc_timestamp": _utc(), "preflight": preflight,
                        "provenance_sha256": provenance, "arms": list(ARMS), "heldout_access": False,
                        "source_f32_values_opened": False, "auto_resume": False})
            result = {"ok": False, "status": "NOT_RUN_RESOURCE_PROJECTION", "schema": SCHEMA, "utc_timestamp": _utc(),
                      "preflight": preflight, "provenance_sha256": provenance, "heldout_access": False,
                      "source_f32_values_opened": False, "partial_preserved": True,
                      "output_sha256": {paths["manifest"].name: _sha_file(paths["manifest"]), paths["log"].name: _sha_file(paths["log"])} }
            _write_once(paths["result"], result)
            print(json.dumps({"ok": False, "status": result["status"], "result": str(paths["result"])}, sort_keys=True))
            return 1
        provenance = _provenance()
    except ResourceError as exc:
        _write_once(paths["result"], {"ok": False, "status": "VOID_RESOURCE", "schema": SCHEMA, "utc_timestamp": _utc(),
                    "error_type": type(exc).__name__, "error": str(exc), "heldout_access": False, "partial_preserved": True})
        return 1
    except Exception as exc:
        _write_once(paths["result"], {"ok": False, "status": "VOID_APPARATUS", "schema": SCHEMA, "utc_timestamp": _utc(),
                    "error_type": type(exc).__name__, "error": str(exc), "heldout_access": False, "partial_preserved": True})
        return 1
    limits = {"launch_available_ram_min": MIN_LAUNCH_RAM, "output_free_min": MIN_OUTPUT_DISK,
              "runtime_available_ram_min": MIN_RUNTIME_RAM, "private_commit_max": MAX_WORKER_MEMORY,
              "working_set_max": MAX_WORKER_MEMORY, "wall_seconds_max": MAX_WALL_SECONDS,
              "monitor_interval_seconds": MONITOR_SECONDS}
    _write_once(paths["manifest"], {"schema": SCHEMA, "utc_timestamp": _utc(), "preflight": preflight,
                "provenance_sha256": provenance, "arms": list(ARMS), "caps": limits, "heldout_access": False,
                "auto_resume": False, "raw_write_once": [paths["tile_audit"].name]})
    _log(paths["log"], {"event": "parent_preflight_complete"})
    child_env = os.environ.copy()
    child_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HUB_KERNELS": "NO"})
    python = bounded._direct_worker_python(child_env)
    token = secrets.token_urlsafe(32)
    child_env[WORKER_TOKEN_ENV] = token
    command = [python, str(Path(__file__).resolve()), "--worker", "--output-dir", str(paths["root"]),
               "--supervisor-manifest", str(paths["manifest"]), "--worker-result", str(paths["worker"]), "--worker-token", token]
    process = None
    try:
        with paths["stdout"].open("x", encoding="utf-8") as stdout, paths["stderr"].open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=str(HERE), env=child_env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            _log(paths["log"], {"event": "child_launch", "pid": process.pid, "direct_worker_only": True})
            outcome = bounded._monitor_or_terminate(process, psutil=bounded._psutil(), expected_executable=Path(python),
                                                     wall_limit_seconds=MAX_WALL_SECONDS, interval_seconds=MONITOR_SECONDS,
                                                     on_sample=lambda sample: _log(paths["log"], {"event": "resource_sample", **sample}))
        worker = _json(paths["worker"]) if paths["worker"].is_file() else None
        status = "VOID_RESOURCE" if outcome.status == "VOID_RESOURCE" else worker.get("status", "VOID_APPARATUS") if worker else "VOID_APPARATUS"
        if outcome.exit_code == 0 and worker is not None and status == "COMPLETE_DIAGNOSTIC":
            _validate_worker(paths["root"], worker)
        elif status not in {"VOID_RESOURCE", "VOID_FORMAT"}:
            status = "VOID_APPARATUS"
        result = {"ok": status == "COMPLETE_DIAGNOSTIC", "status": status, "schema": SCHEMA, "utc_timestamp": _utc(),
                  "worker": worker, "monitor": asdict(outcome), "heldout_access": False, "partial_preserved": True}
    except Exception as exc:
        if process is not None:
            bounded._terminate_only_child(process)
        status = "VOID_APPARATUS"
        result = {"ok": False, "status": status, "schema": SCHEMA, "utc_timestamp": _utc(), "heldout_access": False,
                  "partial_preserved": True, "error_type": type(exc).__name__, "error": str(exc)}
    _log(paths["log"], {"event": "supervisor_finished", "status": status})
    result["output_sha256"] = {path.name: _sha_file(path) for path in paths["root"].iterdir() if path.is_file() and path != paths["result"]}
    _write_once(paths["result"], result)
    print(json.dumps({"ok": result["ok"], "status": status, "result": str(paths["result"])}, sort_keys=True))
    return 0 if result["ok"] else 1


def _selftest() -> None:
    values = np.arange(256, dtype=np.float32).reshape(1, 256)
    encoded = codec.encode_w2_bf16(values)
    assert codec.w2_bf16_from_blob(codec.w2_bf16_to_blob(encoded), encoded.shape).byte_count == 72
    assert _project_seconds(float(EXPERT_STORED_WEIGHTS)) == 1.0
    try:
        _project_seconds(0.0)
    except ResourceError:
        pass
    else:
        raise AssertionError("zero throughput was accepted")
    with tempfile.TemporaryDirectory(prefix="strat02e-scout-") as directory:
        root = Path(directory)
        score = root / "arm.jsonl"
        score.write_text("{}\n" * 48, encoding="utf-8")
        worker = {"ok": True, "status": "COMPLETE_DIAGNOSTIC", "heldout_access": False,
                  "arms": [{"arm": ARMS[0], "score_file": score.name, "score_sha256": _sha_file(score)},
                           {"arm": ARMS[1], "score_file": score.name, "score_sha256": _sha_file(score)}]}
        # Deliberately test arm-order validation without allowing duplicate files in a real result.
        try:
            _validate_worker(root, worker)
        except ApparatusError:
            pass
    print(json.dumps({"ok": True, "selftest": True, "heldout_access": False, "source_f32_values_opened": False}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selftest", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--supervisor-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.preflight:
            report = _read_only_preflight(args.snapshot, args.output_dir)
            print(json.dumps({"ok": report["status"] == "READY", **report}, sort_keys=True))
            return 0 if report["status"] == "READY" else 1
        if args.worker:
            if (args.output_dir is None or args.supervisor_manifest is None or args.worker_result is None or not args.worker_token or
                    not secrets.compare_digest(args.worker_token, os.environ.get(WORKER_TOKEN_ENV, "")) or
                    args.supervisor_manifest.resolve().parent != args.output_dir.resolve() or
                    args.worker_result.resolve().parent != args.output_dir.resolve()):
                raise ApparatusError("internal worker token or paths are invalid")
            return _worker(args)
        if args.output_dir is None:
            raise ApparatusError("--run requires a fresh --output-dir")
        return _run(args)
    except ResourceError as exc:
        print(json.dumps({"ok": False, "status": "VOID_RESOURCE", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "VOID_APPARATUS", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
