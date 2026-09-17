"""Offline, source-weight-free loader for the verified STRAT-02 W4-BF16-v2 export.

This is a candidate construction gate, not a scoring runner.  It never opens
source safetensors or heldout.  The caller must supervise its process for the
quality brief's wall/RAM caps before any calibration or forward pass.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from benchmarks.donor_adaptation.density import strat02_mmap_teacher as teacher
from benchmarks.donor_adaptation.density import strat02_token_audit as token_audit
from benchmarks.donor_adaptation.density import strat02_w4_bf16_export_verify as verifier


HERE = Path(__file__).resolve().parent
BOUND_EXPORT = HERE / "results" / "strat02_w4_bf16_full_export_20260917_143753"
BOUND_CONTROL_SHA256 = {
    "artifact_manifest.json": "06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c",
    "plan_manifest.json": "1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42",
    "supervisor_result.json": "d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca",
}
QUALITY_PREREQUISITE_SHA256 = {
    HERE / "results" / "strat02_teacher_baseline_20260917_105941" / "teacher_scores.jsonl":
        "96d284da8f223d613039cb9e2a87b24e7b6130701c0873eea0854c3e8290b224",
    HERE / "results" / "strat02_teacher_repeatability_20260917_142728" / "supervisor_result.json":
        "89390013e926fddfad5c3bd651b29f1ae861de2bb98996d297300343cf65d41f",
}
MAX_TILE_ROWS = 16
MAX_TILE_GROUPS = 16
MIN_LAUNCH_RAM = 55 * 1024 ** 3
MIN_LAUNCH_DISK = 1 * 1024 ** 3
MIN_RUNTIME_RAM = 8 * 1024 ** 3
MAX_WORKER_MEMORY = 70 * 1024 ** 3
SAMPLE_INTERVAL_SECONDS = 5
MAX_SCORE_WALL_SECONDS = 6 * 60 * 60


class CandidateIntegrityError(RuntimeError):
    """Pinned identity, artifact, or decoded tensor validation failed."""


class CandidateResourceError(RuntimeError):
    """The candidate cannot safely enter or continue a bounded scoring run."""


@dataclass(frozen=True)
class ResourcePlan:
    launch_available_ram_min: int = MIN_LAUNCH_RAM
    launch_output_free_min: int = MIN_LAUNCH_DISK
    runtime_available_ram_min: int = MIN_RUNTIME_RAM
    worker_private_commit_max: int = MAX_WORKER_MEMORY
    worker_working_set_max: int = MAX_WORKER_MEMORY
    sample_interval_seconds: int = SAMPLE_INTERVAL_SECONDS
    score_wall_seconds_max: int = MAX_SCORE_WALL_SECONDS


@dataclass
class PreparedCandidate:
    root: Path
    manifest_sha256: str
    plan: Mapping[str, list[dict[str, Any]]]
    descriptors: tuple[Mapping[str, Any], ...]
    model: Any  # still on meta; no output values have been decoded
    summary: Mapping[str, int]
    resource_plan: ResourcePlan


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateIntegrityError(f"expected JSON object: {path.name}")
    return value


def _check_control_hashes(root: Path, expected: Mapping[str, str]) -> None:
    for name, pinned_sha in expected.items():
        if _sha_file(root / name) != pinned_sha:
            raise CandidateIntegrityError(f"bound export control SHA mismatch: {name}")


def check_bound_controls(root: Path) -> dict[str, str]:
    """Metadata-only quality-arm binding; never opens any raw payload."""
    root = Path(root).resolve()
    if not root.samefile(BOUND_EXPORT):
        raise CandidateIntegrityError("candidate is not the preregistered full export directory")
    _check_control_hashes(root, BOUND_CONTROL_SHA256)
    result = _read_json(root / "supervisor_result.json")
    if result.get("status") != "COMPLETE" or result.get("verdict") != "EXPORT_VERIFIED" or result.get("ok") is not True:
        raise CandidateIntegrityError("bound export does not carry COMPLETE/EXPORT_VERIFIED")
    return dict(BOUND_CONTROL_SHA256)


def check_quality_prerequisites() -> None:
    """Hash-only binding of teacher denominator and repeatability before forward."""
    for path, expected in QUALITY_PREREQUISITE_SHA256.items():
        if _sha_file(path) != expected:
            raise CandidateIntegrityError(f"quality prerequisite SHA mismatch: {path.name}")
    repeatability = _read_json(next(path for path in QUALITY_PREREQUISITE_SHA256 if path.name == "supervisor_result.json"))
    if repeatability.get("status") != "PASS_REPEATABILITY":
        raise CandidateIntegrityError("teacher repeatability prerequisite did not pass")


def _require_tokenizer(snapshot: Path, expected: Mapping[str, str]) -> None:
    if set(expected) != set(token_audit.ASSETS):
        raise CandidateIntegrityError("export tokenizer asset keyset differs from pin")
    for name in token_audit.ASSETS:
        actual = _sha_file(snapshot / name)
        if actual != expected[name] or actual != token_audit.EXPECTED_ASSET_SHA256[name]:
            raise CandidateIntegrityError(f"pinned tokenizer SHA mismatch: {name}")


def bind_plan_to_meta(model: Any, plan: Mapping[str, list[dict[str, Any]]],
                      weight_map: Mapping[str, str]) -> None:
    """Bind every exported record to a meta tensor before opening payload bytes."""
    import torch

    state = model.state_dict()
    names = [item["name"] for entries in plan.values() for item in entries]
    if len(names) != len(set(names)) or set(names) != set(state) or set(names) != set(weight_map):
        raise CandidateIntegrityError("candidate/meta/index keyset mismatch")
    linear_keys = {name + ".weight" for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    for shard, entries in plan.items():
        for item in entries:
            key = item["name"]
            tensor = state[key]
            if (weight_map[key] != shard or item.get("source_shard") != shard or
                item.get("shape") != list(tensor.shape) or item.get("source_dtype") != "float32" or
                tensor.dtype != torch.float32 or not tensor.is_meta):
                raise CandidateIntegrityError(f"candidate metadata binding mismatch: {key}")
            expected_encoding = verifier.LINEAR_ENCODING if key in linear_keys else verifier.F32_ENCODING
            if item.get("encoding") != expected_encoding or item.get("length") != verifier._record_length(item["shape"], expected_encoding):
                raise CandidateIntegrityError(f"candidate structural encoding mismatch: {key}")


def check_launch_resources(output_dir: Path) -> ResourcePlan:
    """Read-only preflight for a later single-worker quality run."""
    import psutil

    plan = ResourcePlan()
    available = int(psutil.virtual_memory().available)
    free = int(psutil.disk_usage(str(output_dir)).free)
    if available < plan.launch_available_ram_min or free < plan.launch_output_free_min:
        raise CandidateResourceError(f"quality launch cap: available_ram={available}, output_free={free}")
    return plan


def check_runtime_resources(plan: ResourcePlan, pid: int) -> dict[str, int]:
    """Sample only the supplied direct worker PID; a supervisor calls every 5 s."""
    import psutil

    process = psutil.Process(pid)
    private = getattr(process.memory_full_info(), "private", None)
    if private is None:
        raise CandidateResourceError("worker private commit unavailable")
    sample = {"available_ram": int(psutil.virtual_memory().available),
              "worker_private_commit": int(private), "worker_working_set": int(process.memory_info().rss)}
    if (sample["available_ram"] < plan.runtime_available_ram_min or
        sample["worker_private_commit"] > plan.worker_private_commit_max or
        sample["worker_working_set"] > plan.worker_working_set_max):
        raise CandidateResourceError(f"quality runtime cap: {sample}")
    return sample


def prepare_candidate(root: Path, *, snapshot: Path | None = None,
                      output_dir: Path | None = None) -> PreparedCandidate:
    """Bind pinned meta topology, then verify the *entire* raw export.

    This step reads all exported payloads for hashes/format but no source
    weight values.  Do not call it during a metadata-only preflight.
    """
    import torch

    root = Path(root).resolve()
    check_bound_controls(root)
    if output_dir is not None:
        resource_plan = check_launch_resources(Path(output_dir))
    else:
        resource_plan = ResourcePlan()
    teacher._require_target_environment()
    report = teacher.preflight(snapshot=snapshot)
    if not report.source_status.get("transformers_ok") or report.source_status.get("use_hub_kernels") != "NO":
        raise CandidateIntegrityError("pinned model runtime metadata invalid")
    manifest_path = root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    result = _read_json(root / "supervisor_result.json")
    if (manifest.get("schema") != verifier.SCHEMA or manifest.get("mode") != "run" or
        manifest.get("source_revision") != teacher.REVISION or
        result.get("status") != "COMPLETE" or result.get("verdict") != "EXPORT_VERIFIED" or result.get("ok") is not True or
        result.get("artifact_manifest") != manifest_path.name):
        raise CandidateIntegrityError("artifact is not a completed full export")
    plan_name = manifest.get("plan_file")
    if plan_name != "plan_manifest.json":
        raise CandidateIntegrityError("unexpected export plan filename")
    plan_doc = _read_json(root / plan_name)
    metadata = plan_doc.get("metadata", {})
    if (plan_doc.get("schema") != verifier.SCHEMA or plan_doc.get("mode") != "run" or
        metadata.get("repository") != teacher.REPOSITORY or metadata.get("revision") != teacher.REVISION or
        metadata.get("source_manifest_sha256") != teacher.EXPECTED_MANIFEST_SHA256 or
        metadata.get("source_index_sha256") != report.manifest["index_sha256"] or
        metadata.get("source_shards") != report.manifest["shards"] or
        plan_doc.get("source_shard_order") != [entry["name"] for entry in report.manifest["shards"]]):
        raise CandidateIntegrityError("export plan differs from pinned metadata")
    _require_tokenizer(report.snapshot, metadata.get("tokenizer_sha256", {}))
    EmoConfig, EmoForCausalLM = teacher._import_pinned_emo(report.snapshot)
    config = teacher._read_json(report.snapshot / "config.json", "pinned model config")
    with torch.device("meta"):
        model = EmoForCausalLM(EmoConfig(**config))
    plan = plan_doc.get("plan")
    if not isinstance(plan, dict) or set(plan) != set(plan_doc["source_shard_order"]):
        raise CandidateIntegrityError("export plan shard keyset mismatch")
    bind_plan_to_meta(model, plan, report.index["weight_map"])
    summary = verifier.verify_artifact(root, manifest_path, expected_plan=plan, require_full=True)
    if summary != plan_doc.get("expected_ledger") or summary != result.get("verification"):
        raise CandidateIntegrityError("export ledger differs from completed result")
    return PreparedCandidate(root, _sha_file(manifest_path), plan, tuple(manifest["shards"]), model, summary, resource_plan)


def _read_exact(handle: Any, offset: int, size: int, limit: int) -> bytes:
    if offset < 0 or size < 0 or offset + size > limit:
        raise CandidateIntegrityError("record read escapes its byte boundary")
    handle.seek(offset)
    value = handle.read(size)
    if len(value) != size:
        raise CandidateIntegrityError("payload truncated during decode")
    return value


def decode_record(handle: Any, record: Mapping[str, Any]) -> Any:
    """Allocate one CPU F32 tensor and fill it from one bounded raw record."""
    import torch

    shape = record["shape"]
    encoding = record["encoding"]
    length = verifier._record_length(shape, encoding)
    base = record["offset"]
    if type(base) is not int or base < 0 or record["length"] != length:
        raise CandidateIntegrityError("record offset/length invalid")
    end = base + length
    tensor = torch.empty(tuple(shape), dtype=torch.float32, device="cpu")
    output = tensor.numpy()
    if encoding == verifier.F32_ENCODING:
        words = output.view(np.uint32).reshape(-1)
        for first in range(0, words.size, 2 * 1024 * 1024):
            count = min(2 * 1024 * 1024, words.size - first)
            raw = _read_exact(handle, base + first * 4, count * 4, end)
            words[first:first + count] = np.frombuffer(raw, dtype="<u4")
        return tensor
    rows, columns = shape
    groups = columns // 128
    codes_base = base + rows * groups * 2
    for row in range(0, rows, MAX_TILE_ROWS):
        tile_rows = min(MAX_TILE_ROWS, rows - row)
        for group in range(0, groups, MAX_TILE_GROUPS):
            tile_groups = min(MAX_TILE_GROUPS, groups - group)
            ordinal = row * groups + group
            # Rows are not contiguous when this tile spans fewer than all groups.
            for local_row in range(tile_rows):
                position = ordinal + local_row * groups
                scale_raw = _read_exact(handle, base + position * 2, tile_groups * 2, end)
                code_raw = _read_exact(handle, codes_base + position * 64, tile_groups * 64, end)
                bits = np.frombuffer(scale_raw, dtype="<u2")
                packed = np.frombuffer(code_raw, dtype=np.uint8).reshape(tile_groups, 64)
                if np.any((bits == 0) | ((bits & 0x8000) != 0) | ((bits & 0x7F80) == 0x7F80)):
                    raise CandidateIntegrityError("invalid BF16 scale in candidate payload")
                low, high = packed & 0x0F, packed >> 4
                if np.any((low == 8) | (high == 8)):
                    raise CandidateIntegrityError("reserved W4 -8 code in candidate payload")
                codes = np.empty((tile_groups, 128), dtype=np.int8)
                codes[:, 0::2] = np.where(low >= 8, low.astype(np.int16) - 16, low).astype(np.int8)
                codes[:, 1::2] = np.where(high >= 8, high.astype(np.int16) - 16, high).astype(np.int8)
                scales = (bits.astype("<u4") << np.uint32(16)).view("<f4")
                destination = output[row + local_row, group * 128:(group + tile_groups) * 128].reshape(tile_groups, 128)
                np.multiply(codes.astype(np.float32), scales[:, None], out=destination)
    return tensor


def _assert_no_meta_and_shared(model: Any, assigned: Mapping[str, Any]) -> None:
    actual = model.state_dict()
    if set(actual) != set(assigned):
        raise CandidateIntegrityError("assigned model keyset changed")
    for key, source in assigned.items():
        target = actual[key]
        if source.is_meta or target.is_meta or target.data_ptr() != source.data_ptr():
            raise CandidateIntegrityError(f"meta/pointer mismatch after assign=True: {key}")
    if any(buffer.is_meta for buffer in model.buffers()):
        raise CandidateIntegrityError("meta buffer remains after RoPE restoration")


@contextlib.contextmanager
def load_candidate_model(prepared: PreparedCandidate) -> Iterator[Any]:
    """Yield one F32 CPU model built exclusively from raw payloads, once.

    The context clears its own state references on exit.  Callers must not
    retain the yielded model beyond the context if they expect RAM release.
    """
    import psutil

    if not isinstance(prepared, PreparedCandidate):
        raise TypeError("load_candidate_model requires a verified PreparedCandidate")
    if prepared.model is None:
        raise CandidateIntegrityError("prepared candidate was already consumed")
    if int(psutil.virtual_memory().available) < prepared.resource_plan.launch_available_ram_min:
        raise CandidateResourceError("available RAM below 55 GiB candidate launch cap")
    check_bound_controls(prepared.root)
    check_quality_prerequisites()
    manifest_path = prepared.root / "artifact_manifest.json"
    if _sha_file(manifest_path) != prepared.manifest_sha256:
        raise CandidateIntegrityError("artifact manifest changed after verification")
    # Recheck all hashes immediately before allocation.  This makes the
    # verification-to-use boundary explicit, without touching source shards.
    verifier.verify_artifact(prepared.root, manifest_path, expected_plan=prepared.plan, require_full=True)
    state: dict[str, Any] = {}
    model = prepared.model
    try:
        for descriptor in prepared.descriptors:
            check_runtime_resources(prepared.resource_plan, os.getpid())
            shard = descriptor["source_shard"]
            payload = verifier._safe_child(prepared.root, descriptor["payload_file"])
            with payload.open("rb") as handle:
                for record in prepared.plan[shard]:
                    key = record["name"]
                    if key in state:
                        raise CandidateIntegrityError("duplicate candidate tensor key")
                    state[key] = decode_record(handle, record)
            if _sha_file(payload) != descriptor["payload_sha256"]:
                raise CandidateIntegrityError(f"payload changed during candidate decode: {payload.name}")
        result = model.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise CandidateIntegrityError("strict candidate state assignment failed")
        teacher._restore_nonpersistent_rope_buffer(model)
        _assert_no_meta_and_shared(model, state)
        check_bound_controls(prepared.root)
        model.eval()
        yield model
    finally:
        state.clear()
        prepared.model = None
