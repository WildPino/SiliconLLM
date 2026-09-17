"""Source-independent verifier for STRAT-02 W4 BF16 v2 raw shard exports.

Only the exported plan, record streams, and raw payloads are opened.  This
module never opens a donor shard, corpus, tokenizer, or model.  It checks the
wire layout and accounting; model fidelity belongs to a later gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "strat02_w4_bf16_full_export_v1"
LINEAR_ENCODING = "strat02_w4_g128_bf16scale_v2"
F32_ENCODING = "f32_le_passthrough"
EXPECTED_KEYS = 6259
EXPECTED_LINEARS = 6225
EXPECTED_GROUPS = 104_398_848
EXPECTED_LINEAR_PARAMS = 13_363_052_544
EXPECTED_OTHER_PARAMS = 205_588_480
EXPECTED_LINEAR_BYTES = 6_890_323_968
EXPECTED_F32_BYTES = 822_353_920
EXPECTED_PAYLOAD_BYTES = 7_712_677_888
EXPECTED_ORGAN_COUNTS = {"attention": 64, "router": 16, "expert": 6144, "lm_head": 1}
CHUNK_BYTES = 8 * 1024 * 1024
GROUP_CHUNK = 8192
RECORD_FIELDS = {"source_shard", "name", "shape", "source_dtype", "encoding", "offset", "length", "sha256"}


class VerificationError(ValueError):
    """Exported bytes, layout, or frozen accounting are inconsistent."""


def _exact_int(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise VerificationError(f"expected integer >= {minimum}")
    return value


def _safe_child(root: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or any(c in filename for c in ("/", "\\")):
        raise VerificationError("artifact filename must be a bare local name")
    return root / filename


def _sha_segment(handle: Any, offset: int, length: int) -> str:
    handle.seek(offset)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        block = handle.read(min(CHUNK_BYTES, remaining))
        if not block:
            raise VerificationError("payload truncated during record hash")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def _sha_file(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_length(shape: Sequence[int], encoding: str) -> int:
    if not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape):
        raise VerificationError("record shape must contain positive integer dimensions")
    numel = int(np.prod(shape, dtype=np.int64))
    if encoding == LINEAR_ENCODING:
        if len(shape) != 2 or shape[1] % 128:
            raise VerificationError("W4 record is not an eligible [out,in] linear")
        return shape[0] * (shape[1] // 128) * 66
    if encoding == F32_ENCODING:
        return numel * 4
    raise VerificationError("unknown record encoding")


def _validate_w4_stream(handle: Any, offset: int, shape: Sequence[int]) -> None:
    groups = shape[0] * (shape[1] // 128)
    code_start = offset + groups * 2
    for start in range(0, groups, GROUP_CHUNK):
        count = min(GROUP_CHUNK, groups - start)
        handle.seek(offset + start * 2)
        scales_raw = handle.read(count * 2)
        handle.seek(code_start + start * 64)
        codes_raw = handle.read(count * 64)
        if len(scales_raw) != count * 2 or len(codes_raw) != count * 64:
            raise VerificationError("W4 record truncated inside scale/code region")
        bits = np.frombuffer(scales_raw, dtype="<u2")
        codes = np.frombuffer(codes_raw, dtype=np.uint8).reshape(count, 64)
        invalid_scale = ((bits & 0x7F80) == 0x7F80) | ((bits & 0x8000) != 0)
        if bool(np.any(invalid_scale)):
            raise VerificationError("nonfinite or negative BF16 scale")
        if bool(np.any(((codes & 0x0F) == 8) | ((codes >> 4) == 8))):
            raise VerificationError("reserved W4 -8 nibble")
        if bool(np.any(bits == 0)):
            raise VerificationError("zero BF16 scale is forbidden for pinned donor")


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError("record JSONL is malformed") from exc
            if not isinstance(value, dict) or set(value) != RECORD_FIELDS:
                raise VerificationError("record schema differs from frozen export")
            records.append(value)
    return records


def verify_shard(root: Path, descriptor: Mapping[str, Any], plan: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Verify one exported shard without opening its source shard."""
    root = Path(root)
    shard = descriptor.get("source_shard")
    if not isinstance(shard, str) or not isinstance(plan, list):
        raise VerificationError("invalid shard descriptor/plan")
    payload = _safe_child(root, descriptor.get("payload_file"))
    record_path = _safe_child(root, descriptor.get("records_file"))
    expected_payload_bytes = _exact_int(descriptor.get("payload_bytes"))
    if not isinstance(descriptor.get("payload_sha256"), str) or not isinstance(descriptor.get("records_sha256"), str):
        raise VerificationError("missing shard/record digest")
    if record_path.stat().st_size == 0 and plan:
        raise VerificationError("nonempty shard has empty record stream")
    if _sha_file(record_path) != descriptor["records_sha256"]:
        raise VerificationError("record JSONL SHA-256 mismatch")
    records = _read_records(record_path)
    if len(records) != len(plan):
        raise VerificationError("record count differs from meta inventory plan")
    if "record_count" in descriptor and descriptor["record_count"] != len(records):
        raise VerificationError("worker descriptor record count mismatch")
    if payload.stat().st_size != expected_payload_bytes:
        raise VerificationError("payload truncated or has extra bytes")
    if _sha_file(payload) != descriptor["payload_sha256"]:
        raise VerificationError("payload container SHA-256 mismatch")
    position = linears = groups = linear_params = other_params = linear_bytes = f32_bytes = 0
    with payload.open("rb") as handle:
        for expected, record in zip(plan, records, strict=True):
            if record["name"] != expected.get("name") or record["source_shard"] != shard:
                raise VerificationError("record key/shard differs from meta inventory plan")
            if record["shape"] != expected.get("shape") or record["source_dtype"] != "float32":
                raise VerificationError("record shape/source dtype differs from plan")
            if record["encoding"] != expected.get("encoding") or record["offset"] != position or record["offset"] != expected.get("offset"):
                raise VerificationError("record offset/order/encoding mismatch")
            length = _record_length(record["shape"], record["encoding"])
            if _exact_int(record["length"]) != length or length != expected.get("length"):
                raise VerificationError("record length differs from frozen layout")
            if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or _sha_segment(handle, position, length) != record["sha256"]:
                raise VerificationError("record SHA-256 mismatch")
            numel = int(np.prod(record["shape"], dtype=np.int64))
            if record["encoding"] == LINEAR_ENCODING:
                _validate_w4_stream(handle, position, record["shape"])
                linears += 1
                groups += numel // 128
                linear_params += numel
                linear_bytes += length
            else:
                if record.get("source_dtype") != "float32":
                    raise VerificationError("passthrough source dtype mismatch")
                other_params += numel
                f32_bytes += length
            position += length
    if position != expected_payload_bytes:
        raise VerificationError("record offsets do not exactly cover shard payload")
    return {"records": len(records), "linears": linears, "groups": groups,
            "linear_params": linear_params, "other_params": other_params,
            "linear_bytes": linear_bytes, "f32_bytes": f32_bytes, "payload_bytes": position}


def verify_artifact(root: Path, manifest_path: Path, *, expected_plan: Mapping[str, Any] | None = None,
                    require_full: bool = True) -> dict[str, int]:
    """Verify all declared output shards against the frozen metadata-only plan."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.parent != root:
        raise VerificationError("manifest must live in the result directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise VerificationError("unexpected export manifest schema")
    plan_path = _safe_child(root, manifest.get("plan_file"))
    if _sha_file(plan_path) != manifest.get("plan_sha256"):
        raise VerificationError("metadata plan SHA-256 mismatch")
    plan_file = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = plan_file.get("plan")
    if not isinstance(plan, dict) or plan_file.get("schema") != SCHEMA:
        raise VerificationError("invalid metadata plan")
    if expected_plan is not None and plan != expected_plan:
        raise VerificationError("export plan differs from parent meta-model inventory")
    if "expected_ledger" in plan_file:
        if plan_file.get("organ_counts") != EXPECTED_ORGAN_COUNTS:
            raise VerificationError("planned linear organ counts differ from pinned taxonomy")
        counted = {name: 0 for name in EXPECTED_ORGAN_COUNTS}
        for entries in plan.values():
            if not isinstance(entries, list):
                raise VerificationError("invalid planned shard records")
            for entry in entries:
                if entry.get("encoding") == LINEAR_ENCODING:
                    organ = entry.get("organ")
                    if organ not in counted:
                        raise VerificationError("linear plan has unknown organ")
                    counted[organ] += 1
                elif "organ" in entry:
                    raise VerificationError("F32 passthrough has linear organ")
        if counted != EXPECTED_ORGAN_COUNTS:
            raise VerificationError("planned linear organ taxonomy mismatch")
    shards = manifest.get("shards")
    source_order = plan_file.get("source_shard_order")
    if not isinstance(shards, list) or not isinstance(source_order, list):
        raise VerificationError("missing output/source shard list")
    if require_full and (len(shards) != 11 or len(source_order) != 11):
        raise VerificationError("full export requires eleven shards")
    if not require_full and (len(shards) != 1 or len(source_order) != 11):
        raise VerificationError("smoke requires exactly the first of eleven source shards")
    if [item.get("source_shard") for item in shards] != source_order[:len(shards)]:
        raise VerificationError("output shard order differs from pinned manifest order")
    if set(plan) != set(source_order) or len(set(source_order)) != len(source_order):
        raise VerificationError("plan source shard keyset mismatch")
    pinned_shards = plan_file.get("metadata", {}).get("source_shards")
    if not isinstance(pinned_shards, list) or [item.get("name") for item in pinned_shards] != source_order:
        raise VerificationError("plan pinned source shard list mismatch")
    aggregate = {key: 0 for key in ("records", "linears", "groups", "linear_params", "other_params", "linear_bytes", "f32_bytes", "payload_bytes")}
    names: set[str] = set()
    for index, descriptor in enumerate(shards):
        shard = descriptor["source_shard"]
        if descriptor.get("source_sha256") != pinned_shards[index].get("sha256"):
            raise VerificationError("worker source SHA differs from pinned manifest")
        shard_plan = plan[shard]
        if [item.get("name") for item in shard_plan] != sorted(item.get("name") for item in shard_plan):
            raise VerificationError("planned tensor names are not lexicographic")
        for item in shard_plan:
            if item["name"] in names:
                raise VerificationError("duplicate tensor key across shards")
            names.add(item["name"])
        summary = verify_shard(root, descriptor, shard_plan)
        for key in aggregate:
            aggregate[key] += summary[key]
    if require_full and (aggregate["records"], aggregate["linears"], aggregate["groups"],
                         aggregate["linear_params"], aggregate["other_params"], aggregate["linear_bytes"],
                         aggregate["f32_bytes"], aggregate["payload_bytes"]) != (
                             EXPECTED_KEYS, EXPECTED_LINEARS, EXPECTED_GROUPS, EXPECTED_LINEAR_PARAMS,
                             EXPECTED_OTHER_PARAMS, EXPECTED_LINEAR_BYTES, EXPECTED_F32_BYTES,
                             EXPECTED_PAYLOAD_BYTES):
        raise VerificationError("full export ledger differs from frozen expected totals")
    return aggregate
