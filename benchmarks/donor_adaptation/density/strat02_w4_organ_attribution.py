#!/usr/bin/env python3
"""CPU-only, calibration-only STRAT-02D W4 organ counterfactual apparatus.

``--selftest`` is synthetic; ``--preflight`` opens metadata/control files only.
Only ``--run`` launches a monitored worker that hashes source shards and
constructs the single verified candidate arena.  No code path opens heldout.
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
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from benchmarks.donor_adaptation.density import strat02_bounded_smoke as bounded
from benchmarks.donor_adaptation.density import strat02_mmap_teacher as teacher
from benchmarks.donor_adaptation.density import strat02_score as score_core
from benchmarks.donor_adaptation.density import strat02_token_audit as audit
from benchmarks.donor_adaptation.density import strat02_w4_bf16_candidate_loader as candidate
from benchmarks.donor_adaptation.density import strat02_w4_bf16_export_verify as verifier


SCHEMA = "strat02_w4_organ_attribution_v1"
BRIEF = PROJECT / "docs/research/donor_adaptation/briefs/BRIEF_STRAT_02_W4_ORGAN_ATTRIBUTION.md"
CALIB = HERE / "corpus/strat02_document_holdout_v1/calib.jsonl"
CORPUS_MANIFEST = CALIB.with_name("manifest.json")
W4_CALIBRATION = HERE / "results/strat02_w4_bf16_quality_20260917_162329/calibration_checks.jsonl"
W4_CALIBRATION_SHA256 = "ef1d7f42ad30ff5727432bcc4f0c1a5299ff681f52a5a7690f55602f0171a142"
EXPECTED_CALIB_SHA256 = "f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f"
EXPECTED_MANIFEST_SHA256 = audit.EXPECTED_MANIFEST_SHA256
EXPECTED_CALIB_IDS_SHA256 = audit.EXPECTED_IDS_SHA256["calib"]
EXPECTED_CALIB_TOKENS = audit.EXPECTED_TOTALS["calib"]
EXPECTED_CALIB_BYTES = 362_405
SENTINEL_TOLERANCE_BITS = 1e-5
MAX_COPY_ROWS = 16
BASE_W4_ACTIVE_BYTES = 661_782_528
ARMS = (("ROUTER_F32", ("router",)), ("HEAD_F32", ("lm_head",)),
        ("ATTENTION_F32", ("attention",)),
        ("NONEXPERT_F32", ("router", "lm_head", "attention")))
ORGAN_COUNTS = {"router": 16, "lm_head": 1, "attention": 64, "expert": 6144}
ARM_COUNTS = {"ROUTER_F32": 16, "HEAD_F32": 1, "ATTENTION_F32": 64, "NONEXPERT_F32": 81}
ARM_EXTRA_BYTES = {"ROUTER_F32": 14_614_528, "HEAD_F32": 716_111_872,
                   "ATTENTION_F32": 935_329_792, "NONEXPERT_F32": 1_666_056_192}
MIN_LAUNCH_RAM = 55 * 1024**3
MIN_OUTPUT_DISK = 1 * 1024**3
MIN_RUNTIME_RAM = 8 * 1024**3
MAX_WORKER_MEMORY = 70 * 1024**3
MAX_WALL_SECONDS = 6 * 60 * 60
MONITOR_SECONDS = 5.0
WORKER_TOKEN_ENV = "STRAT02D_ORGAN_PARENT_TOKEN"
PROVENANCE = {"runner": Path(__file__).resolve(), "brief": BRIEF,
              "candidate_loader": HERE / "strat02_w4_bf16_candidate_loader.py",
              "teacher": HERE / "strat02_mmap_teacher.py", "export_verifier": HERE / "strat02_w4_bf16_export_verify.py",
              "score_core": HERE / "strat02_score.py", "token_audit": HERE / "strat02_token_audit.py",
              "bounded_supervisor": HERE / "strat02_bounded_smoke.py"}


class ApparatusError(RuntimeError):
    """A frozen identity, protocol, tensor, score, or rollback check failed."""


class ResourceError(MemoryError):
    """The preregistered resource cap was crossed."""


def _require_safe_kernel_policy() -> None:
    """Default only an absent policy; an explicit conflicting value is fatal."""
    current = os.environ.get("USE_HUB_KERNELS")
    if current is None:
        os.environ["USE_HUB_KERNELS"] = "NO"
    elif current != "NO":
        raise ApparatusError(f"USE_HUB_KERNELS must be NO; found {current!r}")


def _existing_output_volume(path: Path | None) -> Path:
    """Measure the destination filesystem even before a fresh output exists."""
    volume = Path(path or HERE).expanduser().resolve()
    while not volume.exists():
        parent = volume.parent
        if parent == volume:
            raise ApparatusError("no existing ancestor for output volume preflight")
        volume = parent
    if not volume.is_dir():
        raise ApparatusError("output volume ancestor is not a directory")
    return volume


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


def _provenance() -> dict[str, str]:
    return {name: _sha_file(path) for name, path in PROVENANCE.items()}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApparatusError(f"expected JSON object: {path.name}")
    return value


def _read_calibration() -> list[dict[str, Any]]:
    """Read only frozen calibration, never the heldout partition."""
    if _sha_file(CORPUS_MANIFEST) != EXPECTED_MANIFEST_SHA256 or _sha_file(CALIB) != EXPECTED_CALIB_SHA256:
        raise ApparatusError("calibration corpus/manifest SHA differs from pin")
    manifest = _json(CORPUS_MANIFEST)
    part = manifest.get("parts", {}).get("calib", {})
    if part.get("jsonl_sha256") != EXPECTED_CALIB_SHA256 or part.get("item_count") != 48:
        raise ApparatusError("calibration partition not bound by pinned manifest")
    errors: list[str] = []
    rows = audit.read_rows(CALIB, "calib", errors)
    if errors or len(rows) != 48:
        raise ApparatusError(f"frozen calibration rows invalid: {errors}")
    counts = Counter(row.get("category") for row in rows)
    identifiers = [row.get("source_document_id") for row in rows]
    if counts != {"code": 16, "prose": 16, "technical_general": 16} or len(set(identifiers)) != 48:
        raise ApparatusError("calibration category/ID inventory differs from pin")
    if sum(len(row["text"].encode("utf-8")) for row in rows) != EXPECTED_CALIB_BYTES:
        raise ApparatusError("calibration UTF-8 byte total differs from pin")
    for ordinal, row in enumerate(rows, 1):
        unsigned = dict(row)
        digest = unsigned.pop("item_sha256", None)
        if (row.get("split") != "calib" or row.get("schema") != manifest.get("schema") or
            not isinstance(row.get("text"), str) or not row["text"] or
            row.get("span_byte_count") != len(row["text"].encode("utf-8")) or
            digest != audit.sha(audit.canonical(unsigned))):
            raise ApparatusError(f"calibration row {ordinal} failed frozen integrity")
    if audit.aggregate(rows) != part.get("items_aggregate_sha256"):
        raise ApparatusError("calibration item aggregate SHA mismatch")
    return rows


def _tokenizer_and_rows(snapshot: Path, rows: list[dict[str, Any]]) -> Any:
    for name, digest in audit.EXPECTED_ASSET_SHA256.items():
        if _sha_file(snapshot / name) != digest:
            raise ApparatusError(f"tokenizer asset SHA mismatch: {name}")
    tokenizer, _, _, _ = audit.load_tokenizer(snapshot)
    score_core.validate_strat02_tokenizer(tokenizer)
    ids_by_doc: list[list[int]] = []
    for row in rows:
        ids = score_core._encode_payload(tokenizer, row["text"])
        if len(ids) + 1 > score_core.MAX_CONTEXT:
            raise ApparatusError("calibration document exceeds frozen context")
        ids_by_doc.append(ids)
    digest = hashlib.sha256(json.dumps(ids_by_doc, separators=(",", ":")).encode("ascii")).hexdigest()
    if digest != EXPECTED_CALIB_IDS_SHA256 or sum(map(len, ids_by_doc)) != EXPECTED_CALIB_TOKENS:
        raise ApparatusError("calibration token identity/total differs from pin")
    return tokenizer


def _baseline(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if _sha_file(W4_CALIBRATION) != W4_CALIBRATION_SHA256:
        raise ApparatusError("write-once W4 calibration SHA mismatch")
    records = [json.loads(line) for line in W4_CALIBRATION.read_text(encoding="utf-8").splitlines()]
    if len(records) != 48:
        raise ApparatusError("W4 calibration control must contain 48 rows")
    for ordinal, (row, item) in enumerate(zip(rows, records, strict=True), 1):
        if (item.get("source_document_id") != row["source_document_id"] or
            item.get("category") != row["category"] or
            item.get("bytes") != len(row["text"].encode("utf-8")) or
            type(item.get("tokens")) is not int or item["tokens"] <= 0 or
            type(item.get("bits")) not in (int, float) or not math.isfinite(item["bits"]) or item["bits"] < 0):
            raise ApparatusError(f"W4 calibration control row {ordinal} differs from frozen document")
    if sum(item["tokens"] for item in records) != EXPECTED_CALIB_TOKENS or sum(item["bytes"] for item in records) != EXPECTED_CALIB_BYTES:
        raise ApparatusError("W4 calibration control totals differ from pin")
    return records


def _baseline_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets = {name: {"documents": 0, "bits": 0.0, "bytes": 0} for name in ("code", "prose", "technical_general")}
    for item in records:
        bucket = buckets[item["category"]]
        bucket["documents"] += 1
        bucket["bits"] += float(item["bits"])
        bucket["bytes"] += item["bytes"]
    for bucket in buckets.values():
        bucket["bpb"] = bucket["bits"] / bucket["bytes"]
    bits = sum(float(item["bits"]) for item in records)
    return {"documents": len(records), "bits": bits, "bytes": EXPECTED_CALIB_BYTES,
            "bpb": bits / EXPECTED_CALIB_BYTES, "categories": buckets}


def build_arms(plan: Mapping[str, Sequence[Mapping[str, Any]]], *, frozen: bool = True) -> dict[str, dict[str, Any]]:
    """Select exact structural organs, independently of scores or key-name regex."""
    organs: dict[str, list[Mapping[str, Any]]] = {name: [] for name in ORGAN_COUNTS}
    passthrough = 0
    seen: set[str] = set()
    for entries in plan.values():
        for record in entries:
            key = record["name"]
            if key in seen:
                raise ApparatusError("duplicate planned tensor key")
            seen.add(key)
            if record["encoding"] == verifier.LINEAR_ENCODING:
                organ = record.get("organ")
                if organ not in organs or len(record["shape"]) != 2 or record["shape"][1] % 128:
                    raise ApparatusError(f"unclassified W4 linear: {key}")
                organs[organ].append(record)
            elif record["encoding"] == verifier.F32_ENCODING and "organ" not in record:
                passthrough += 1
            else:
                raise ApparatusError(f"unexpected record encoding/organ: {key}")
    if frozen and (dict((name, len(items)) for name, items in organs.items()) != ORGAN_COUNTS or passthrough != 34 or len(seen) != 6259):
        raise ApparatusError("full donor organ/key inventory differs from frozen plan")
    output: dict[str, dict[str, Any]] = {}
    for arm, chosen in ARMS:
        records = [item for organ in chosen for item in organs[organ]]
        active_weights = sum(item["shape"][0] * item["shape"][1] for item in records)
        numerator = active_weights * 223
        if numerator % 64:
            raise ApparatusError("active F32 delta is not integral bytes")
        extra = numerator // 64  # 4 - (66/128) bytes per restored weight
        if frozen and (len(records) != ARM_COUNTS[arm] or extra != ARM_EXTRA_BYTES[arm]):
            raise ApparatusError(f"frozen organ count/active-byte delta mismatch: {arm}")
        output[arm] = {"records": tuple(records), "count": len(records),
                       "active_weights": active_weights, "extra_active_bytes_per_token": extra,
                       "total_active_bytes_per_token": BASE_W4_ACTIVE_BYTES + extra}
    if frozen and set(item["name"] for item in output["NONEXPERT_F32"]["records"]) != set().union(
            *(set(item["name"] for item in output[arm]["records"]) for arm, _ in ARMS[:3])):
        raise ApparatusError("NONEXPERT arm is not exact union of three frozen organs")
    return output


def _tensor_sha(tensor: Any) -> str:
    """Hash raw F32 bits in chunks without creating a tensor-sized byte copy."""
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ApparatusError("tensor hash requires contiguous CPU F32")
    data = memoryview(tensor.detach().numpy()).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(data), 8 * 1024 * 1024):
        digest.update(data[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def _copy_f32_record(handle: Any, record: Mapping[str, Any], destination: Any) -> str:
    """Copy one verified safetensors linear into the existing F32 arena view."""
    import numpy as np
    import torch

    shape = tuple(record["shape"])
    if (record["encoding"] != verifier.LINEAR_ENCODING or len(shape) != 2 or
        destination.device.type != "cpu" or destination.dtype != torch.float32 or
        tuple(destination.shape) != shape or not destination.is_contiguous()):
        raise ApparatusError("F32 copy target/plan mismatch")
    source = handle.get_slice(record["name"])
    if tuple(source.get_shape()) != shape:
        raise ApparatusError("source slice shape differs from verified plan")
    source_hash = hashlib.sha256()
    target = destination.detach().numpy()
    for first in range(0, shape[0], MAX_COPY_ROWS):
        stop = min(first + MAX_COPY_ROWS, shape[0])
        chunk = source[first:stop]
        if (chunk.device.type != "cpu" or chunk.dtype != torch.float32 or
            tuple(chunk.shape) != (stop - first, shape[1])):
            raise ApparatusError("source slice is not bounded CPU F32")
        values = chunk.numpy()
        np.copyto(target[first:stop], values, casting="no")
        source_hash.update(memoryview(values).cast("B"))
    digest = source_hash.hexdigest()
    if _tensor_sha(destination) != digest:
        raise ApparatusError("F32 source/destination bitwise SHA mismatch")
    return digest


def _restore_w4_record(handle: Any, record: Mapping[str, Any], destination: Any,
                       original_sha: str) -> None:
    candidate.decode_record(handle, record, destination=destination)
    if _tensor_sha(destination) != original_sha:
        raise ApparatusError(f"W4 rollback SHA mismatch: {record['name']}")


def _check_sentinel(observed: Any, expected: Mapping[str, Any]) -> None:
    if ((observed.source_document_id, observed.category, observed.tokens, observed.bytes) !=
        (expected["source_document_id"], expected["category"], expected["tokens"], expected["bytes"]) or
        not math.isfinite(float(observed.bits)) or
        not math.isclose(float(observed.bits), float(expected["bits"]), rel_tol=0.0, abs_tol=SENTINEL_TOLERANCE_BITS)):
        raise ApparatusError("W4 sentinel differs from published calibration bits")


def _score_arm(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], output: Path,
               scorer: Callable[..., Any]) -> dict[str, Any]:
    """Persist exactly 48 calibration observations in original order, once."""
    totals = {"documents": 0, "tokens": 0, "bytes": 0, "bits": 0.0}
    categories = {name: {"documents": 0, "tokens": 0, "bytes": 0, "bits": 0.0}
                  for name in ("code", "prose", "technical_general")}
    if len(rows) != 48:
        raise ApparatusError("arm must score exactly 48 calibration rows")
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            observation = scorer(model, tokenizer, row, chunk_size=score_core.CHUNK_SIZE)
            expected_tokens = len(score_core._encode_payload(tokenizer, row["text"]))
            expected_bytes = len(row["text"].encode("utf-8"))
            if ((observation.source_document_id, observation.category) !=
                (row["source_document_id"], row["category"]) or
                type(observation.tokens) is not int or observation.tokens != expected_tokens or
                type(observation.bytes) is not int or observation.bytes != expected_bytes or
                not math.isfinite(float(observation.bits)) or observation.bits < 0):
                raise ApparatusError("arm score identity/token/byte/finite check failed")
            item = {"source_document_id": observation.source_document_id, "category": observation.category,
                    "tokens": observation.tokens, "bytes": observation.bytes, "bits": float(observation.bits)}
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            for bucket in (totals, categories[observation.category]):
                bucket["documents"] += 1
                bucket["tokens"] += observation.tokens
                bucket["bytes"] += observation.bytes
                bucket["bits"] += float(observation.bits)
    if totals["documents"] != 48 or totals["tokens"] != EXPECTED_CALIB_TOKENS or totals["bytes"] != EXPECTED_CALIB_BYTES:
        raise ApparatusError("arm score totals differ from frozen calibration")
    for bucket in categories.values():
        if bucket["documents"] != 16:
            raise ApparatusError("arm category count differs from frozen calibration")
        bucket["bpb"] = bucket["bits"] / bucket["bytes"]
    totals["bpb"] = totals["bits"] / totals["bytes"]
    totals["categories"] = categories
    return totals


def _read_only_preflight(snapshot: Path | None, output_volume: Path | None) -> dict[str, Any]:
    _require_safe_kernel_policy()
    versions = bounded._pinned_environment()
    report = teacher.preflight(snapshot=snapshot)
    if (report.source_status.get("repository") != teacher.REPOSITORY or
        report.source_status.get("revision") != teacher.REVISION or
        not report.source_status.get("transformers_ok") or
        report.source_status.get("use_hub_kernels") != "NO" or
        not report.source_status.get("shards_present")):
        raise ApparatusError("pinned F32 metadata/runtime not present")
    candidate.check_bound_controls(candidate.BOUND_EXPORT)
    tokenizer_hashes = {name: _sha_file(report.snapshot / name) for name in audit.ASSETS}
    for name, digest in audit.EXPECTED_ASSET_SHA256.items():
        if tokenizer_hashes[name] != digest:
            raise ApparatusError(f"tokenizer asset SHA mismatch: {name}")
    if _sha_file(W4_CALIBRATION) != W4_CALIBRATION_SHA256:
        raise ApparatusError("frozen W4 calibration control SHA mismatch")
    rows = _read_calibration()
    import psutil
    available = int(psutil.virtual_memory().available)
    volume = _existing_output_volume(output_volume)
    free = int(psutil.disk_usage(str(volume)).free)
    if available < MIN_LAUNCH_RAM or free < MIN_OUTPUT_DISK:
        raise ResourceError(f"launch cap: available_ram={available}, output_free={free}")
    return {"snapshot": str(report.snapshot), "source_manifest_sha256": teacher.EXPECTED_MANIFEST_SHA256,
            "revision": teacher.REVISION, "runtime_versions": versions,
            "artifact_control_sha256": dict(candidate.BOUND_CONTROL_SHA256),
            "tokenizer_sha256": tokenizer_hashes,
            "w4_calibration_sha256": W4_CALIBRATION_SHA256,
            "calib_sha256": EXPECTED_CALIB_SHA256, "calib_rows": len(rows),
            "calib_tokens_expected": EXPECTED_CALIB_TOKENS, "calib_bytes_expected": EXPECTED_CALIB_BYTES,
            "available_ram_bytes": available, "output_free_bytes": free, "output_volume_checked": str(volume),
            "heldout_access": False, "source_weight_values_opened": False}


@contextlib.contextmanager
def _load_calibration_candidate(prepared: candidate.PreparedCandidate):
    """Use the candidate's verified arena/decoder without its heldout-score pin.

    The quality loader binds the heldout-derived teacher_scores.jsonl before
    loading. STRAT-02D must not open that file, so this diagnostic reproduces
    the same source-free arena assignment from the candidate's core helpers.
    """
    import torch

    candidate.check_bound_controls(prepared.root)
    manifest_path = prepared.root / "artifact_manifest.json"
    if _sha_file(manifest_path) != prepared.manifest_sha256:
        raise ApparatusError("bound export manifest changed after candidate preparation")
    verifier.verify_artifact(prepared.root, manifest_path, expected_plan=prepared.plan, require_full=True)
    if (prepared.summary.get("linear_params") != verifier.EXPECTED_LINEAR_PARAMS or
        prepared.summary.get("other_params") != verifier.EXPECTED_OTHER_PARAMS):
        raise ApparatusError("candidate arena parameter ledger differs from frozen export")
    order = [entry["source_shard"] for entry in prepared.descriptors]
    arena, state = candidate.make_arena_views(prepared.plan, order, prepared.summary)
    if arena.numel() != 13_568_641_024:
        raise ApparatusError("candidate arena is not exact pinned F32 size")
    model = prepared.model
    try:
        for descriptor in prepared.descriptors:
            candidate.check_runtime_resources(prepared.resource_plan, os.getpid())
            payload = verifier._safe_child(prepared.root, descriptor["payload_file"])
            with payload.open("rb") as handle:
                for record in prepared.plan[descriptor["source_shard"]]:
                    target = state[record["name"]]
                    if candidate.decode_record(handle, record, destination=target) is not target:
                        raise ApparatusError("candidate decoder replaced arena tensor view")
            if _sha_file(payload) != descriptor["payload_sha256"]:
                raise ApparatusError("candidate payload changed during decode")
        result = model.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise ApparatusError("strict candidate state assignment failed")
        teacher._restore_nonpersistent_rope_buffer(model)
        candidate._assert_no_meta_and_shared(model, state, arena, prepared.plan, order)
        candidate.check_bound_controls(prepared.root)
        model.eval()
        yield model
    finally:
        state.clear()
        prepared.model = None


def _worker(args: argparse.Namespace) -> int:
    root = args.output_dir.resolve()
    results: list[dict[str, Any]] = []
    current_arm: str | None = None
    try:
        _require_safe_kernel_policy()
        manifest = _json(args.supervisor_manifest)
        if manifest.get("schema") != SCHEMA or manifest.get("provenance_sha256") != _provenance():
            raise ApparatusError("worker provenance differs from parent")
        bounded._pinned_environment()
        rows = _read_calibration()
        baseline = _baseline(rows)
        report = teacher.preflight(snapshot=args.snapshot)
        tokenizer = _tokenizer_and_rows(report.snapshot, rows)
        verified = teacher.verify_shards(report)  # hash all 11 source files before any weight-value read
        prepared = candidate.prepare_candidate(candidate.BOUND_EXPORT, snapshot=report.snapshot, output_dir=root)
        arms = build_arms(prepared.plan)
        index = report.index["weight_map"]
        # Refuse an organ label whose key is not the exact pinned index partition.
        for arm, _ in ARMS:
            for record in arms[arm]["records"]:
                if index.get(record["name"]) != record["source_shard"]:
                    raise ApparatusError("organ record/index shard mismatch")
        from safetensors import safe_open
        with contextlib.ExitStack() as stack:
            source_handles = {entry["name"]: stack.enter_context(safe_open(str(report.snapshot / entry["name"]),
                              framework="pt", device="cpu")) for entry in report.manifest["shards"]}
            teacher._assert_index_key_mapping(index, source_handles)
            artifact_handles = {descriptor["source_shard"]: stack.enter_context(
                verifier._safe_child(prepared.root, descriptor["payload_file"]).open("rb"))
                for descriptor in prepared.descriptors}
            with _load_calibration_candidate(prepared) as model:
                import torch
                model.eval()
                state = model.state_dict()
                originals = {record["name"]: _tensor_sha(state[record["name"]])
                             for record in arms["NONEXPERT_F32"]["records"]}
                sentinel = rows[0]
                _check_sentinel(score_core.score_document(model, tokenizer, sentinel, chunk_size=128), baseline[0])
                for arm, _ in ARMS:
                    current_arm = arm
                    selected = arms[arm]["records"]
                    teacher._assert_unchanged(verified)
                    for record in selected:
                        _copy_f32_record(source_handles[record["source_shard"]], record, state[record["name"]])
                    path = root / f"{arm.lower()}.jsonl"
                    summary = _score_arm(model, tokenizer, rows, path, score_core.score_document)
                    for record in selected:
                        _restore_w4_record(artifact_handles[record["source_shard"]], record,
                                           state[record["name"]], originals[record["name"]])
                    _check_sentinel(score_core.score_document(model, tokenizer, sentinel, chunk_size=128), baseline[0])
                    summary["gain_cal_BPB"] = (sum(item["bits"] for item in baseline) - summary["bits"]) / EXPECTED_CALIB_BYTES
                    results.append({"arm": arm, "selected_linears": arms[arm]["count"],
                                    "active_weights": arms[arm]["active_weights"],
                                    "extra_active_bytes_per_token": arms[arm]["extra_active_bytes_per_token"],
                                    "total_active_bytes_per_token": arms[arm]["total_active_bytes_per_token"],
                                    "score_file": path.name, "score_sha256": _sha_file(path), "score": summary,
                                    "rollback_sha256_verified": True, "sentinel_bits_verified": True})
                    print(json.dumps({"event": "arm_complete", "arm": arm, "rows": 48}, sort_keys=True), flush=True)
        teacher._assert_unchanged(verified)
        import torch
        _write_once(args.worker_result, {"ok": True, "status": "COMPLETE", "schema": SCHEMA,
                    "utc_timestamp": _utc(), "arms": results, "source_shards_verified": 11,
                    "source_sha256": {entry["name"]: entry["sha256"] for entry in report.manifest["shards"]},
                    "candidate_control_sha256": dict(candidate.BOUND_CONTROL_SHA256),
                    "w4_control_sha256": W4_CALIBRATION_SHA256,
                    "w4_control_score": _baseline_summary(baseline),
                    "calibration_control_bits": baseline[0]["bits"],
                    "torch_threads": {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()},
                    "heldout_access": False, "model_copies": 1})
        return 0
    except MemoryError as exc:
        status = "VOID_RESOURCE"
        error_type, error_message, error_traceback = type(exc).__name__, str(exc), traceback.format_exc()
    except Exception as exc:
        status = "VOID_APPARATUS"
        error_type, error_message, error_traceback = type(exc).__name__, str(exc), traceback.format_exc()
    try:
        _write_once(args.worker_result, {"ok": False, "status": status, "schema": SCHEMA,
                    "utc_timestamp": _utc(), "current_arm": current_arm,
                    "completed_arms": [item["arm"] for item in results],
                    "error_type": error_type, "error": error_message,
                    "traceback": error_traceback, "partial_preserved": True,
                    "heldout_access": False})
    except OSError:
        pass
    return 1


def _prepare_output(root: Path) -> dict[str, Path]:
    root = root.resolve()
    if root.exists():
        raise ApparatusError(f"output directory must be new: {root}")
    root.mkdir(parents=True, exist_ok=False)
    paths = {"root": root, "manifest": root / "supervisor_manifest.json", "log": root / "supervisor_log.jsonl",
             "worker": root / "worker_result.json", "result": root / "supervisor_result.json",
             "stdout": root / "worker_stdout.log", "stderr": root / "worker_stderr.log"}
    with paths["log"].open("x", encoding="utf-8", newline="\n"):
        pass
    return paths


def _validate_completed_arms(root: Path, worker: Mapping[str, Any]) -> None:
    arms = worker.get("arms")
    if not isinstance(arms, list) or [item.get("arm") for item in arms] != [name for name, _ in ARMS]:
        raise ApparatusError("worker did not complete four frozen arms in order")
    rows = _read_calibration()
    for item in arms:
        name = item["arm"]
        filename = f"{name.lower()}.jsonl"
        path = root / filename
        if (item.get("score_file") != filename or item.get("selected_linears") != ARM_COUNTS[name] or
            item.get("extra_active_bytes_per_token") != ARM_EXTRA_BYTES[name] or
            item.get("rollback_sha256_verified") is not True or
            item.get("sentinel_bits_verified") is not True or
            not path.is_file() or _sha_file(path) != item.get("score_sha256")):
            raise ApparatusError(f"arm output/control mismatch: {name}")
        with path.open("r", encoding="utf-8") as handle:
            scored = [json.loads(line) for line in handle]
        if len(scored) != 48:
            raise ApparatusError(f"arm output does not contain exactly 48 rows: {name}")
        for ordinal, (source, score) in enumerate(zip(rows, scored, strict=True), 1):
            if (score.get("source_document_id") != source["source_document_id"] or
                score.get("category") != source["category"] or
                score.get("bytes") != len(source["text"].encode("utf-8")) or
                type(score.get("tokens")) is not int or score["tokens"] <= 0 or
                type(score.get("bits")) not in (int, float) or not math.isfinite(score["bits"])):
                raise ApparatusError(f"arm row binding mismatch: {name}:{ordinal}")


def _run(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    try:
        preflight = _read_only_preflight(args.snapshot, paths["root"])
        hashes = _provenance()
    except Exception as exc:
        status = "VOID_RESOURCE" if isinstance(exc, (MemoryError, ResourceError)) else "VOID_APPARATUS"
        _write_once(paths["result"], {"ok": False, "status": status, "error_type": type(exc).__name__,
                                      "error": str(exc), "utc_timestamp": _utc(), "partial_preserved": True})
        return 1
    limits = {"launch_ram_min": MIN_LAUNCH_RAM, "launch_disk_min": MIN_OUTPUT_DISK,
              "runtime_ram_min": MIN_RUNTIME_RAM, "worker_private_max": MAX_WORKER_MEMORY,
              "worker_working_set_max": MAX_WORKER_MEMORY, "wall_seconds_max": MAX_WALL_SECONDS,
              "monitor_interval_seconds": MONITOR_SECONDS}
    _write_once(paths["manifest"], {"schema": SCHEMA, "utc_timestamp": _utc(), "preflight": preflight,
                                    "provenance_sha256": hashes, "arms": [arm for arm, _ in ARMS],
                                    "arm_counts": ARM_COUNTS, "arm_extra_active_bytes_per_token": ARM_EXTRA_BYTES,
                                    "baseline_w4_active_bytes_per_token": BASE_W4_ACTIVE_BYTES,
                                    "source_manifest_sha256": teacher.EXPECTED_MANIFEST_SHA256,
                                    "calibration_control_sha256": W4_CALIBRATION_SHA256,
                                    "caps": limits, "heldout_access": False, "auto_resume": False})
    _log(paths["log"], {"event": "preflight_complete"})
    child_env = os.environ.copy()
    child_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HUB_KERNELS": "NO"})
    python = bounded._direct_worker_python(child_env)
    token = secrets.token_urlsafe(32)
    child_env[WORKER_TOKEN_ENV] = token
    command = [python, str(Path(__file__).resolve()), "--worker", "--snapshot", preflight["snapshot"],
               "--output-dir", str(paths["root"]), "--supervisor-manifest", str(paths["manifest"]),
               "--worker-result", str(paths["worker"]), "--worker-token", token]
    process = None
    try:
        with paths["stdout"].open("x", encoding="utf-8") as stdout, paths["stderr"].open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=str(HERE), env=child_env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr)
            _log(paths["log"], {"event": "child_launch", "pid": process.pid, "direct_worker_only": True})
            outcome = bounded._monitor_or_terminate(process, psutil=bounded._psutil(), expected_executable=Path(python),
                                                     wall_limit_seconds=MAX_WALL_SECONDS, interval_seconds=MONITOR_SECONDS,
                                                     on_sample=lambda sample: _log(paths["log"], {"event": "resource_sample", **sample}))
        worker = _json(paths["worker"]) if paths["worker"].is_file() else None
        if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
            status = outcome.status
        elif worker and worker.get("status") in {"VOID_RESOURCE", "VOID_APPARATUS"}:
            status = worker["status"]
        elif outcome.exit_code == 0 and worker and worker.get("ok") is True and worker.get("status") == "COMPLETE":
            status = "COMPLETE"
            try:
                _validate_completed_arms(paths["root"], worker)
            except (ApparatusError, OSError, ValueError, KeyError, TypeError):
                _log(paths["log"], {"event": "parent_arm_artifact_validation_failed", "traceback": traceback.format_exc()})
                status = "VOID_APPARATUS"
        else:
            status = "VOID_APPARATUS"
        result = {"ok": status == "COMPLETE", "status": status, "schema": SCHEMA,
                  "utc_timestamp": _utc(), "worker": worker, "monitor": asdict(outcome),
                  "provenance_sha256": hashes, "heldout_access": False, "partial_preserved": True}
    except Exception as exc:
        if process is not None:
            bounded._terminate_only_child(process)
        status = "VOID_APPARATUS"
        result = {"ok": False, "status": status, "schema": SCHEMA, "utc_timestamp": _utc(),
                  "error_type": type(exc).__name__, "error": str(exc), "partial_preserved": True,
                  "heldout_access": False}
    _log(paths["log"], {"event": "supervisor_finished", "status": status})
    result["output_sha256"] = {path.name: _sha_file(path) for path in paths["root"].iterdir() if path.is_file() and path != paths["result"]}
    _write_once(paths["result"], result)
    print(json.dumps({"ok": result["ok"], "status": status, "result": str(paths["result"])}, sort_keys=True))
    return 0 if result["ok"] else 1


def _selftest() -> None:
    import numpy as np
    import torch
    from types import SimpleNamespace
    from io import BytesIO
    record = {"name": "router.weight", "source_shard": "synthetic", "shape": [1, 128],
              "encoding": verifier.LINEAR_ENCODING, "organ": "router"}
    arms = build_arms({"synthetic": [record]}, frozen=False)
    assert [name for name, _ in ARMS] == list(arms)
    assert arms["ROUTER_F32"]["count"] == 1
    source = np.arange(128, dtype=np.float32).reshape(1, 128)
    class Slice:
        def get_shape(self): return (1, 128)
        def __getitem__(self, index): return torch.from_numpy(source[index])
    target = torch.empty((1, 128), dtype=torch.float32)
    _copy_f32_record(SimpleNamespace(get_slice=lambda _key: Slice()), record, target)
    assert _tensor_sha(target) == hashlib.sha256(source.tobytes()).hexdigest()
    _check_sentinel(SimpleNamespace(source_document_id="s", category="code", tokens=1, bytes=1, bits=2.0),
                    {"source_document_id": "s", "category": "code", "tokens": 1, "bytes": 1, "bits": 2.0})
    print(json.dumps({"ok": True, "selftest": True, "source_weights_opened": False,
                      "heldout_access": False}, sort_keys=True))


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
            print(json.dumps({"ok": True, "status": "PREFLIGHT_COMPLETE", **report}, sort_keys=True))
            return 0
        if args.worker:
            if (not args.worker_token or not secrets.compare_digest(args.worker_token, os.environ.get(WORKER_TOKEN_ENV, "")) or
                args.output_dir is None or args.supervisor_manifest is None or args.worker_result is None or args.snapshot is None or
                args.supervisor_manifest.resolve().parent != args.output_dir.resolve() or
                args.worker_result.resolve().parent != args.output_dir.resolve()):
                raise ApparatusError("internal worker token/paths invalid")
            return _worker(args)
        if args.output_dir is None:
            raise ApparatusError("--run requires a fresh --output-dir")
        return _run(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "VOID_RESOURCE" if isinstance(exc, MemoryError) else "VOID_APPARATUS",
                          "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
