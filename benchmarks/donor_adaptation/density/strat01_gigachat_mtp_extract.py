"""Reproducibly extract the pinned GigaChat 3.1 BF16 MTP sidecar.

This tool deliberately has no import-time I/O.  Its default ``--plan`` mode
fetches only the immutable safetensors index and its 103,408-byte prefix,
using HTTP Range and rejecting anything other than a fully validated 206
response.  ``--execute`` is the explicit opt-in for the 1,614,430,336-byte
MTP payload range.  It never fetches the 4.99 GB source shard as a whole.

The output is a normal safetensors container containing precisely the 210
``model.layers.26.*`` BF16 tensors.  The source shard's published SHA-256 is
recorded as an identity assertion, not claimed as locally verified: proving
that digest would require reading the entire source blob.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence
from urllib.request import Request, urlopen


REPOSITORY = "ai-sage/GigaChat3.1-10B-A1.8B-bf16"
REVISION = "189fff27a1dee68473960c3d5bca53e0e07a3191"
SHARD = "model-00000-of-00005.safetensors"
SOURCE_FILE_BYTES = 4_986_065_392
SOURCE_LFS_SHA256 = "612f84d11aca544694ef674e58781d0c39ae4b2046f9cadeada79f1d37e581b6"
SOURCE_HEADER_BYTES = 103_400
SOURCE_PREFIX_BYTES = 8 + SOURCE_HEADER_BYTES
MTP_KEY_PREFIX = "model.layers.26."
MTP_KEY_COUNT = 210
MTP_ABSOLUTE_FIRST = 2_554_652_912
MTP_ABSOLUTE_LAST = 4_169_083_247
MTP_PAYLOAD_BYTES = 1_614_430_336
ALIASED_TENSOR_BYTES = 394_002_432
ALIAS_CANDIDATES = (
    ("model.layers.26.shared_head.head.weight", "lm_head.weight"),
    ("model.layers.26.embed_tokens.weight", "model.embed_tokens.weight"),
)

METADATA_MAX_BYTES = 1 * 1024 * 1024
TRANSFER_CHUNK_BYTES = 16 * 1024 * 1024
READ_BLOCK_BYTES = 1 * 1024 * 1024


class ExtractionError(RuntimeError):
    """The pinned source or a local extraction invariant was not proved."""


@dataclass(frozen=True)
class SourceSpec:
    repository: str
    revision: str
    shard: str
    source_file_bytes: int
    source_lfs_sha256: str
    source_header_bytes: int
    mtp_key_prefix: str
    mtp_key_count: int
    mtp_absolute_first: int
    mtp_absolute_last: int
    mtp_payload_bytes: int


PINNED = SourceSpec(
    repository=REPOSITORY,
    revision=REVISION,
    shard=SHARD,
    source_file_bytes=SOURCE_FILE_BYTES,
    source_lfs_sha256=SOURCE_LFS_SHA256,
    source_header_bytes=SOURCE_HEADER_BYTES,
    mtp_key_prefix=MTP_KEY_PREFIX,
    mtp_key_count=MTP_KEY_COUNT,
    mtp_absolute_first=MTP_ABSOLUTE_FIRST,
    mtp_absolute_last=MTP_ABSOLUTE_LAST,
    mtp_payload_bytes=MTP_PAYLOAD_BYTES,
)


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_start: int
    source_end: int
    sidecar_start: int
    sidecar_end: int


@dataclass(frozen=True)
class ExtractionPlan:
    spec: SourceSpec
    index_url: str
    source_url: str
    source_header_sha256: str
    source_header_bytes: int
    sidecar_header: bytes
    records: tuple[TensorRecord, ...]

    @property
    def sidecar_payload_bytes(self) -> int:
        return sum(record.sidecar_end - record.sidecar_start for record in self.records)

    @property
    def sidecar_file_bytes(self) -> int:
        return 8 + len(self.sidecar_header) + self.sidecar_payload_bytes


_CONTENT_RANGE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")


def _resolve_url(spec: SourceSpec, filename: str) -> str:
    return "https://huggingface.co/{}/resolve/{}/{}?download=true".format(
        spec.repository, spec.revision, filename
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_content_range(value: str | None) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ExtractionError("HTTP Range response has no Content-Range")
    match = _CONTENT_RANGE.fullmatch(value.strip())
    if not match:
        raise ExtractionError(f"malformed Content-Range: {value!r}")
    start, end, total = (int(part) for part in match.groups())
    if total <= 0 or start > end or end >= total:
        raise ExtractionError(f"invalid Content-Range bounds: {value!r}")
    return start, end, total


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    try:
        return int(status)
    except (TypeError, ValueError) as exc:
        raise ExtractionError(f"HTTP response has invalid status: {status!r}") from exc


def _range_response(
    url: str,
    start: int,
    end: int,
    *,
    expected_total: int | None = None,
    allow_short_final: bool = False,
    opener: Callable[..., Any] = urlopen,
) -> tuple[Any, int, int, int]:
    if start < 0 or end < start:
        raise ExtractionError(f"invalid requested byte range {start}-{end}")
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    try:
        response = opener(request, timeout=60)
    except Exception as exc:
        raise ExtractionError(f"HTTP Range request failed for {start}-{end}: {exc}") from exc
    status = _response_status(response)
    if status != 206:
        try:
            response.close()
        finally:
            raise ExtractionError(f"HTTP Range request must return 206, got {status} for {start}-{end}")
    try:
        actual_start, actual_end, total = _parse_content_range(response.headers.get("Content-Range"))
        if actual_start != start:
            raise ExtractionError(
                f"Content-Range starts at {actual_start}, expected requested {start}"
            )
        if (not allow_short_final and actual_end != end) or (allow_short_final and actual_end > end):
            raise ExtractionError(
                f"Content-Range ends at {actual_end}, incompatible with requested {end}"
            )
        if expected_total is not None and total != expected_total:
            raise ExtractionError(f"Content-Range total {total} != pinned source size {expected_total}")
        expected_length = actual_end - actual_start + 1
        content_length = response.headers.get("Content-Length")
        if content_length is not None and content_length != str(expected_length):
            raise ExtractionError(
                f"Content-Length {content_length!r} != Content-Range length {expected_length}"
            )
        return response, actual_start, actual_end, total
    except Exception:
        response.close()
        raise


def _read_range(
    url: str,
    start: int,
    end: int,
    *,
    expected_total: int | None = None,
    allow_short_final: bool = False,
    byte_cap: int | None = None,
    opener: Callable[..., Any] = urlopen,
) -> tuple[bytes, int]:
    response, actual_start, actual_end, total = _range_response(
        url, start, end, expected_total=expected_total, allow_short_final=allow_short_final, opener=opener
    )
    expected_length = actual_end - actual_start + 1
    if byte_cap is not None and expected_length > byte_cap:
        response.close()
        raise ExtractionError(f"metadata response {expected_length} bytes exceeds {byte_cap}-byte cap")
    try:
        chunks: list[bytes] = []
        remaining = expected_length
        while remaining:
            block = response.read(min(READ_BLOCK_BYTES, remaining))
            if not block:
                raise ExtractionError(f"truncated HTTP Range body: {remaining} bytes missing")
            if len(block) > remaining:
                raise ExtractionError("HTTP Range body exceeds its Content-Range")
            chunks.append(block)
            remaining -= len(block)
        if response.read(1):
            raise ExtractionError("HTTP Range body exceeds its Content-Range")
        return b"".join(chunks), total
    finally:
        response.close()


def _parse_header(prefix: bytes, spec: SourceSpec) -> Mapping[str, Any]:
    if len(prefix) != 8 + spec.source_header_bytes:
        raise ExtractionError(f"source header prefix has {len(prefix)} bytes, expected {8 + spec.source_header_bytes}")
    encoded_length = int.from_bytes(prefix[:8], "little", signed=False)
    if encoded_length != spec.source_header_bytes:
        raise ExtractionError(f"safetensors header says {encoded_length} bytes, expected {spec.source_header_bytes}")
    try:
        value = json.loads(prefix[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"invalid source safetensors JSON header: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtractionError("source safetensors header is not an object")
    return value


def _positive_shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ExtractionError(f"{name}: BF16 tensor shape must be a non-empty list")
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ExtractionError(f"{name}: invalid shape dimension {dimension!r}")
        shape.append(dimension)
    return tuple(shape)


def _tensor_bytes(shape: Iterable[int]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * 2  # BF16 is exactly two bytes.


def _plan_records(index: Mapping[str, Any], header: Mapping[str, Any], spec: SourceSpec) -> tuple[TensorRecord, ...]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ExtractionError("fresh index has no object weight_map")
    index_names = {name for name, shard in weight_map.items() if isinstance(name, str) and name.startswith(spec.mtp_key_prefix)}
    if len(index_names) != spec.mtp_key_count:
        raise ExtractionError(f"index has {len(index_names)} MTP keys, expected {spec.mtp_key_count}")
    if any(weight_map[name] != spec.shard for name in index_names):
        raise ExtractionError("an MTP index key is not mapped to the pinned source shard")
    header_names = {name for name in header if isinstance(name, str) and name.startswith(spec.mtp_key_prefix)}
    if index_names != header_names:
        missing = sorted(index_names - header_names)
        extra = sorted(header_names - index_names)
        raise ExtractionError(f"MTP key set differs between index and header; missing={missing[:3]} extra={extra[:3]}")

    unsorted: list[tuple[int, str, str, tuple[int, ...], int]] = []
    for name in header_names:
        item = header[name]
        if not isinstance(item, dict):
            raise ExtractionError(f"{name}: safetensors entry is not an object")
        if item.get("dtype") != "BF16":
            raise ExtractionError(f"{name}: expected BF16, got {item.get('dtype')!r}")
        shape = _positive_shape(item.get("shape"), name)
        offsets = item.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
        ):
            raise ExtractionError(f"{name}: invalid data_offsets")
        source_start, source_end = offsets
        if source_start < 0 or source_end <= source_start:
            raise ExtractionError(f"{name}: invalid data offset ordering")
        if source_end - source_start != _tensor_bytes(shape):
            raise ExtractionError(f"{name}: BF16 shape and data_offsets disagree")
        unsorted.append((source_start, name, "BF16", shape, source_end))

    unsorted.sort()
    expected_source_start = spec.mtp_absolute_first - 8 - spec.source_header_bytes
    expected_source_end = spec.mtp_absolute_last + 1 - 8 - spec.source_header_bytes
    if not unsorted or unsorted[0][0] != expected_source_start or unsorted[-1][4] != expected_source_end:
        raise ExtractionError("MTP payload endpoints disagree with the immutable range manifest")

    records: list[TensorRecord] = []
    next_source = expected_source_start
    next_sidecar = 0
    for source_start, name, dtype, shape, source_end in unsorted:
        if source_start != next_source:
            raise ExtractionError(f"MTP tensors are not contiguous before {name}")
        length = source_end - source_start
        records.append(TensorRecord(name, dtype, shape, source_start, source_end, next_sidecar, next_sidecar + length))
        next_source = source_end
        next_sidecar += length
    if next_source != expected_source_end or next_sidecar != spec.mtp_payload_bytes:
        raise ExtractionError("MTP contiguous interval byte count disagrees with immutable manifest")
    return tuple(records)


def _sidecar_header(records: Sequence[TensorRecord], spec: SourceSpec) -> bytes:
    metadata = {
        "format": "strat01_gigachat_mtp_sidecar_v1",
        "source_repository": spec.repository,
        "source_revision": spec.revision,
        "source_shard": spec.shard,
        "source_lfs_sha256_expected": spec.source_lfs_sha256,
        "source_lfs_sha256_status": "UNVERIFIED_FULL_SOURCE_NOT_READ",
        "source_absolute_interval": f"{spec.mtp_absolute_first}-{spec.mtp_absolute_last}",
        "source_payload_bytes": str(spec.mtp_payload_bytes),
    }
    value: dict[str, Any] = {"__metadata__": metadata}
    for record in sorted(records, key=lambda item: item.name):
        value[record.name] = {
            "dtype": record.dtype,
            "shape": list(record.shape),
            "data_offsets": [record.sidecar_start, record.sidecar_end],
        }
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def fetch_plan(
    spec: SourceSpec = PINNED,
    *,
    index_url: str | None = None,
    source_url: str | None = None,
    opener: Callable[..., Any] = urlopen,
) -> ExtractionPlan:
    """Fetch fresh, capped index/header metadata and prove the extraction plan."""
    chosen_index_url = index_url or _resolve_url(spec, "model.safetensors.index.json")
    chosen_source_url = source_url or _resolve_url(spec, spec.shard)
    index_bytes, index_total = _read_range(
        chosen_index_url,
        0,
        METADATA_MAX_BYTES - 1,
        allow_short_final=True,
        byte_cap=METADATA_MAX_BYTES,
        opener=opener,
    )
    if index_total != len(index_bytes):
        raise ExtractionError("index Range response is not the complete capped index file")
    try:
        index = json.loads(index_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"invalid fresh source index: {exc}") from exc
    if not isinstance(index, dict):
        raise ExtractionError("fresh source index is not an object")
    prefix, _ = _read_range(
        chosen_source_url,
        0,
        8 + spec.source_header_bytes - 1,
        expected_total=spec.source_file_bytes,
        byte_cap=8 + spec.source_header_bytes,
        opener=opener,
    )
    header = _parse_header(prefix, spec)
    records = _plan_records(index, header, spec)
    sidecar_header = _sidecar_header(records, spec)
    return ExtractionPlan(
        spec=spec,
        index_url=chosen_index_url,
        source_url=chosen_source_url,
        source_header_sha256=_sha256_bytes(prefix),
        source_header_bytes=len(prefix),
        sidecar_header=sidecar_header,
        records=records,
    )


def plan_summary(plan: ExtractionPlan) -> Mapping[str, Any]:
    names = {record.name for record in plan.records}
    aliases = [
        {
            "sidecar_tensor": sidecar,
            "candidate_base_tensor": base,
            "bytes": ALIASED_TENSOR_BYTES,
            "evidence_status": "EXTERNALLY_REPORTED_SHA256_EQUALITY_NOT_REVERIFIED_BY_PLAN",
            "extraction_behavior": "MATERIALIZED_IN_SIDECAR_NO_ALIASING",
        }
        for sidecar, base in ALIAS_CANDIDATES
        if sidecar in names
    ]
    return {
        "schema": "strat01_gigachat_mtp_extract_plan_v1",
        "mode": "PLAN_ONLY_NO_MTP_PAYLOAD_DOWNLOADED",
        "source": {
            "repository": plan.spec.repository,
            "revision": plan.spec.revision,
            "shard": plan.spec.shard,
            "url": plan.source_url,
            "full_file_bytes": plan.spec.source_file_bytes,
            "lfs_sha256_expected": plan.spec.source_lfs_sha256,
            "lfs_sha256_status": "UNVERIFIED_FULL_SOURCE_NOT_READ",
            "header_bytes": plan.source_header_bytes,
            "header_sha256": plan.source_header_sha256,
        },
        "mtp": {
            "key_count": len(plan.records),
            "absolute_first": plan.spec.mtp_absolute_first,
            "absolute_last": plan.spec.mtp_absolute_last,
            "payload_bytes": plan.sidecar_payload_bytes,
            "sidecar_header_bytes": len(plan.sidecar_header),
            "sidecar_file_bytes": plan.sidecar_file_bytes,
        },
        "candidate_aliases": {
            "aliases": aliases,
            "theoretical_payload_bytes_saved_if_a_future_loader_manifest_explicitly_allows_aliases": sum(
                int(alias["bytes"]) for alias in aliases
            ),
            "current_extractor_behavior": "COMPLETE_210_TENSOR_SIDECAR; CANDIDATES_ARE_NOT_APPLIED",
        },
    }


def _part_path(output: Path) -> Path:
    return output.with_name(output.name + ".part")


def _resume_path(output: Path) -> Path:
    return output.with_name(output.name + ".part.resume.json")


def _manifest_path(output: Path) -> Path:
    return output.with_name(output.name + ".manifest.json")


def _require_safe_output_paths(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ExtractionError(f"refusing to overwrite existing sidecar: {output}")
    manifest = _manifest_path(output)
    if manifest.exists() or manifest.is_symlink():
        raise ExtractionError(f"refusing to overwrite existing sidecar manifest: {manifest}")
    if not output.parent.is_dir():
        raise ExtractionError(f"output parent does not exist: {output.parent}")


def _safe_output_path(output: Path) -> Path:
    """Resolve the directory, never an output leaf which might be a dangling link."""
    raw = output.expanduser()
    if raw.name in {"", ".", ".."}:
        raise ExtractionError(f"output must name a file, got {output!s}")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise ExtractionError(f"cannot resolve output parent {raw.parent}: {exc}") from exc
    if not parent.is_dir():
        raise ExtractionError(f"output parent is not a directory: {parent}")
    candidate = parent / raw.name
    # Path.resolve() on the whole pathname would dereference a dangling leaf
    # and silently redirect a new extraction to its target.  Keep the leaf
    # literal and reject both live and dangling symlinks instead.
    if candidate.is_symlink():
        raise ExtractionError(f"refusing symlink as sidecar output leaf: {candidate}")
    return candidate


def _preflight_disk(output: Path, plan: ExtractionPlan) -> int:
    try:
        free = int(shutil.disk_usage(output.parent).free)
    except OSError as exc:
        raise ExtractionError(f"cannot inspect output filesystem {output.parent}: {exc}") from exc
    required = plan.sidecar_file_bytes + TRANSFER_CHUNK_BYTES
    if free < required:
        raise ExtractionError(f"need at least {required} free bytes for sidecar extraction; found {free}")
    return free


def _resume_payload(plan: ExtractionPlan) -> Mapping[str, Any]:
    return {
        "schema": "strat01_gigachat_mtp_resume_v1",
        "repository": plan.spec.repository,
        "revision": plan.spec.revision,
        "shard": plan.spec.shard,
        "source_url": plan.source_url,
        "source_lfs_sha256_expected": plan.spec.source_lfs_sha256,
        "source_header_sha256": plan.source_header_sha256,
        "sidecar_header_sha256": _sha256_bytes(plan.sidecar_header),
        "sidecar_header_bytes": len(plan.sidecar_header),
        "mtp_payload_bytes": plan.sidecar_payload_bytes,
    }


def _atomic_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace our private resume state, never a user output."""
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _validated_chunk_records(value: Any, plan: ExtractionPlan, downloaded_payload_bytes: int) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ExtractionError("resumable extraction state has no committed chunk list")
    expected_offset = 0
    records: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"payload_offset", "bytes", "sha256"}:
            raise ExtractionError("resumable extraction state has an invalid committed chunk record")
        offset, length, digest = item["payload_offset"], item["bytes"], item["sha256"]
        expected_length = min(TRANSFER_CHUNK_BYTES, plan.sidecar_payload_bytes - expected_offset)
        if (
            expected_length <= 0
            or isinstance(offset, bool) or not isinstance(offset, int) or offset != expected_offset
            or isinstance(length, bool) or not isinstance(length, int) or length != expected_length
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ExtractionError("resumable extraction state committed chunks are not an exact prefix")
        expected_offset += length
        records.append({"payload_offset": offset, "bytes": length, "sha256": digest})
    if expected_offset != downloaded_payload_bytes:
        raise ExtractionError("resumable extraction byte count disagrees with committed chunks")
    return tuple(records)


def _write_resume(
    path: Path,
    plan: ExtractionPlan,
    downloaded_payload_bytes: int,
    committed_chunks: Sequence[Mapping[str, Any]],
) -> None:
    if downloaded_payload_bytes < 0 or downloaded_payload_bytes > plan.sidecar_payload_bytes:
        raise ExtractionError("invalid resumable payload byte count")
    chunks = _validated_chunk_records(list(committed_chunks), plan, downloaded_payload_bytes)
    value = dict(_resume_payload(plan))
    value["downloaded_payload_bytes"] = downloaded_payload_bytes
    value["committed_chunks"] = list(chunks)
    _atomic_json_replace(path, value)


def _verify_committed_chunks(
    part: Path,
    header_bytes: int,
    chunks: Sequence[Mapping[str, Any]],
) -> None:
    """Authenticate every committed payload chunk in one forward file pass."""
    try:
        with part.open("rb") as handle:
            handle.seek(header_bytes)
            for item in chunks:
                digest = hashlib.sha256()
                remaining = int(item["bytes"])
                while remaining:
                    block = handle.read(min(READ_BLOCK_BYTES, remaining))
                    if not block:
                        raise ExtractionError("partial sidecar ends inside a committed chunk")
                    digest.update(block)
                    remaining -= len(block)
                if digest.hexdigest() != item["sha256"]:
                    raise ExtractionError(
                        f"partial sidecar committed chunk SHA-256 mismatch at payload offset {item['payload_offset']}"
                    )
    except OSError as exc:
        raise ExtractionError(f"cannot authenticate committed partial sidecar chunks: {exc}") from exc


def _load_safe_resume(part: Path, resume: Path, plan: ExtractionPlan) -> tuple[int, tuple[Mapping[str, Any], ...]]:
    if part.is_symlink() or resume.is_symlink():
        raise ExtractionError("resumable extraction artifacts must not be symlinks")
    try:
        value = json.loads(resume.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"cannot read resumable extraction state: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtractionError("resumable extraction state is not an object")
    expected = _resume_payload(plan)
    if any(value.get(key) != item for key, item in expected.items()):
        raise ExtractionError("resumable extraction state does not match the fresh immutable plan")
    done = value.get("downloaded_payload_bytes")
    if isinstance(done, bool) or not isinstance(done, int) or not 0 <= done <= plan.sidecar_payload_bytes:
        raise ExtractionError("resumable extraction state has invalid payload byte count")
    chunks = _validated_chunk_records(value.get("committed_chunks"), plan, done)
    try:
        stat = part.stat()
        if not part.is_file() or stat.st_size != 8 + len(plan.sidecar_header) + done:
            raise ExtractionError("partial sidecar size does not match resumable extraction state")
        with part.open("rb") as handle:
            encoded = handle.read(8)
            if len(encoded) != 8 or int.from_bytes(encoded, "little") != len(plan.sidecar_header):
                raise ExtractionError("partial sidecar has a different safetensors header length")
            if handle.read(len(plan.sidecar_header)) != plan.sidecar_header:
                raise ExtractionError("partial sidecar header does not match fresh immutable plan")
    except OSError as exc:
        raise ExtractionError(f"cannot validate partial sidecar: {exc}") from exc
    _verify_committed_chunks(part, 8 + len(plan.sidecar_header), chunks)
    return done, chunks


def _start_or_resume(
    output: Path,
    plan: ExtractionPlan,
    no_resume: bool,
) -> tuple[Path, Path, int, tuple[Mapping[str, Any], ...]]:
    part, resume = _part_path(output), _resume_path(output)
    have_part = part.exists() or part.is_symlink()
    have_resume = resume.exists() or resume.is_symlink()
    if part.is_symlink() or resume.is_symlink():
        raise ExtractionError("incomplete extraction artifacts must not be symlinks")
    if no_resume and (have_part or have_resume):
        if have_part != have_resume:
            raise ExtractionError("incomplete extraction artifacts are inconsistent; refusing --no-resume deletion")
        # Destructive cleanup is allowed only after the artifacts prove they
        # belong to this exact fresh immutable plan, including chunk hashes.
        _load_safe_resume(part, resume, plan)
        if part.is_symlink() or resume.is_symlink():
            raise ExtractionError("incomplete extraction artifacts became symlinks during validation")
        part.unlink()
        resume.unlink()
        have_part = have_resume = False
    if have_part != have_resume:
        raise ExtractionError("incomplete extraction artifacts are inconsistent; use --no-resume to discard them")
    if have_part:
        done, chunks = _load_safe_resume(part, resume, plan)
        return part, resume, done, chunks
    try:
        with part.open("xb") as handle:
            handle.write(len(plan.sidecar_header).to_bytes(8, "little"))
            handle.write(plan.sidecar_header)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ExtractionError(f"cannot create exclusive partial sidecar: {exc}") from exc
    _write_resume(resume, plan, 0, ())
    return part, resume, 0, ()


def _stream_range_to_handle(
    url: str,
    start: int,
    end: int,
    handle: BinaryIO,
    *,
    expected_total: int,
    opener: Callable[..., Any] = urlopen,
) -> tuple[int, str]:
    response, actual_start, actual_end, _ = _range_response(
        url, start, end, expected_total=expected_total, opener=opener
    )
    expected_length = actual_end - actual_start + 1
    remaining = expected_length
    written = 0
    digest = hashlib.sha256()
    try:
        while remaining:
            block = response.read(min(READ_BLOCK_BYTES, remaining))
            if not block:
                raise ExtractionError(f"truncated source body while reading {start}-{end}")
            if len(block) > remaining:
                raise ExtractionError("source body exceeds validated Content-Range")
            handle.write(block)
            digest.update(block)
            written += len(block)
            remaining -= len(block)
        if response.read(1):
            raise ExtractionError("source body exceeds validated Content-Range")
        return written, digest.hexdigest()
    finally:
        response.close()


def _read_sidecar(path: Path, plan: ExtractionPlan) -> Mapping[str, Any]:
    try:
        with path.open("rb") as handle:
            encoded = handle.read(8)
            if len(encoded) != 8:
                raise ExtractionError("sidecar is missing its safetensors header length")
            header_length = int.from_bytes(encoded, "little")
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ExtractionError("sidecar safetensors header is truncated")
        header = json.loads(header_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"cannot read sidecar safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ExtractionError("sidecar safetensors header is not an object")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or metadata.get("format") != "strat01_gigachat_mtp_sidecar_v1":
        raise ExtractionError("sidecar provenance metadata is absent or invalid")
    actual_names = set(header) - {"__metadata__"}
    expected_names = {record.name for record in plan.records}
    if actual_names != expected_names:
        raise ExtractionError("sidecar tensor names do not equal the validated source MTP key set")
    expected_size = 8 + len(plan.sidecar_header) + plan.sidecar_payload_bytes
    if path.stat().st_size != expected_size:
        raise ExtractionError("sidecar file size disagrees with validated header and payload length")
    by_name = {record.name: record for record in plan.records}
    for name in actual_names:
        item, record = header[name], by_name[name]
        if not isinstance(item, dict) or item.get("dtype") != record.dtype:
            raise ExtractionError(f"sidecar {name}: dtype parity failure")
        if item.get("shape") != list(record.shape):
            raise ExtractionError(f"sidecar {name}: shape parity failure")
        if item.get("data_offsets") != [record.sidecar_start, record.sidecar_end]:
            raise ExtractionError(f"sidecar {name}: offset parity failure")
    return header


def _link_finalize(temp: Path, final: Path) -> None:
    """Atomically create a final path without an overwrite race."""
    if final.exists() or final.is_symlink():
        raise ExtractionError(f"refusing to overwrite existing output during finalization: {final}")
    try:
        os.link(temp, final)
        temp.unlink()
    except FileExistsError as exc:
        raise ExtractionError(f"output appeared during finalization: {final}") from exc
    except OSError as exc:
        raise ExtractionError(f"cannot atomically finalize output {final}: {exc}") from exc


def _write_manifest(output: Path, plan: ExtractionPlan) -> Path:
    manifest = _manifest_path(output)
    if manifest.exists() or manifest.is_symlink():
        raise ExtractionError(f"refusing to overwrite existing manifest: {manifest}")
    value = {
        "schema": "strat01_gigachat_mtp_sidecar_manifest_v1",
        "sidecar": {
            "name": output.name,
            "bytes": output.stat().st_size,
            "sha256": _sha256_file(output),
            "self_read_parity": "PASS_names_shapes_dtypes_offsets",
        },
        "source": {
            "repository": plan.spec.repository,
            "revision": plan.spec.revision,
            "shard": plan.spec.shard,
            "url": plan.source_url,
            "full_blob_bytes": plan.spec.source_file_bytes,
            "lfs_sha256_expected": plan.spec.source_lfs_sha256,
            "lfs_sha256_status": "UNVERIFIED_FULL_SOURCE_NOT_READ",
            "header_sha256": plan.source_header_sha256,
            "absolute_interval": [plan.spec.mtp_absolute_first, plan.spec.mtp_absolute_last],
        },
        "tensors": [
            {
                "name": record.name,
                "dtype": record.dtype,
                "shape": list(record.shape),
                "source_data_offsets": [record.source_start, record.source_end],
                "sidecar_data_offsets": [record.sidecar_start, record.sidecar_end],
            }
            for record in plan.records
        ],
    }
    temp = manifest.with_name(manifest.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _link_finalize(temp, manifest)
    finally:
        if temp.exists():
            temp.unlink()
    return manifest


def execute_plan(
    plan: ExtractionPlan,
    output: Path,
    *,
    no_resume: bool = False,
    opener: Callable[..., Any] = urlopen,
) -> Mapping[str, Any]:
    """Download only the proven interval and atomically produce its sidecar."""
    output = _safe_output_path(output)
    _require_safe_output_paths(output)
    free = _preflight_disk(output, plan)
    part, resume, done, committed_chunks = _start_or_resume(output, plan, no_resume)
    header_bytes = 8 + len(plan.sidecar_header)
    with part.open("r+b") as handle:
        handle.seek(header_bytes + done)
        while done < plan.sidecar_payload_bytes:
            count = min(TRANSFER_CHUNK_BYTES, plan.sidecar_payload_bytes - done)
            source_start = plan.spec.mtp_absolute_first + done
            source_end = source_start + count - 1
            before = done
            try:
                copied, digest = _stream_range_to_handle(
                    plan.source_url,
                    source_start,
                    source_end,
                    handle,
                    expected_total=plan.spec.source_file_bytes,
                    opener=opener,
                )
                if copied != count:
                    raise ExtractionError(f"copied {copied} bytes, expected {count}")
                handle.flush()
                os.fsync(handle.fileno())
                done += copied
                new_committed_chunks = committed_chunks + ({
                    "payload_offset": before,
                    "bytes": copied,
                    "sha256": digest,
                },)
                _write_resume(resume, plan, done, new_committed_chunks)
                committed_chunks = new_committed_chunks
            except Exception:
                handle.truncate(header_bytes + before)
                handle.flush()
                os.fsync(handle.fileno())
                _write_resume(resume, plan, before, committed_chunks)
                raise
    _read_sidecar(part, plan)
    _link_finalize(part, output)
    try:
        resume.unlink()
    except OSError as exc:
        raise ExtractionError(f"sidecar finalized but cannot remove private resume state: {exc}") from exc
    _read_sidecar(output, plan)
    manifest = _write_manifest(output, plan)
    return {
        "sidecar": str(output),
        "manifest": str(manifest),
        "sidecar_bytes": output.stat().st_size,
        "sidecar_sha256": _sha256_file(output),
        "tensors": len(plan.records),
        "free_disk_bytes_preflight": free,
        "source_full_blob_sha256": "UNVERIFIED_FULL_SOURCE_NOT_READ",
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="extract only the pinned GigaChat 3.1 MTP safetensors sidecar")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="default: fresh capped index/header proof, no MTP payload")
    action.add_argument("--execute", action="store_true", help="explicitly download the 1,614,430,336-byte MTP interval")
    parser.add_argument("--out", default="gigachat31_10b_a18b_mtp.safetensors", help="new sidecar output path")
    parser.add_argument("--no-resume", action="store_true", help="discard this output's private incomplete artifacts before --execute")
    args = parser.parse_args(argv)
    if args.no_resume and not args.execute:
        parser.error("--no-resume requires --execute")
    try:
        plan = fetch_plan()
        if not args.execute:
            print(json.dumps(plan_summary(plan), sort_keys=True, indent=2))
            return 0
        print("MTP extraction started: one validated Range request per 16 MiB chunk; no per-chunk progress output.", flush=True)
        result = execute_plan(plan, Path(args.out), no_resume=args.no_resume)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except ExtractionError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
