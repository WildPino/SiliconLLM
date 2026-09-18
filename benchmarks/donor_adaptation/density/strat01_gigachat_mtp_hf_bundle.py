"""Build a bounded, provenance-checked HF input bundle for GigaChat 3.1 MTP.

The bundle is deliberately only the 26th (MTP) layer, its three required
root aliases, and four inert Hugging Face JSON files.  It is an input for the
official llama.cpp converter, *not* a runnable full checkpoint and it never
imports model code.  The default ``--plan`` mode performs no network I/O,
writes nothing, and does not read the 1.6 GiB sidecar payload.  ``--execute``
is the explicit opt-in for the bounded copy/link and the four pinned downloads.

The existing sidecar is never changed.  On Windows a hardlink is preferred
when it is safe and on the same volume; otherwise the file is copied into a
new output directory.  A symlink is intentionally never produced.
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

from safetensors import safe_open

try:  # Support both ``python -m`` and direct invocation from this directory.
    from benchmarks.donor_adaptation.density import strat01_gigachat_mtp_extract as extract
    from benchmarks.donor_adaptation.density import strat01_gigachat_mtp_gguf_contract as contract
except ModuleNotFoundError:  # pragma: no cover - direct-script convenience.
    import strat01_gigachat_mtp_extract as extract
    import strat01_gigachat_mtp_gguf_contract as contract


SIDECAR_SHA256 = contract.SIDECAR_SHA256
SIDECAR_BYTES = contract.SIDECAR_BYTES
ROOT_NORM_NAME = "model.norm.weight"
ROOT_NORM_SHAPE = (contract.HIDDEN,)
ROOT_NORM_BYTES = contract.HIDDEN * 2
ROOT_NORM_SHA256 = contract.ROOT_NORM_BF16_SHA256
SOURCE_HEADER_SHA256 = "e970752e3764fc1be2a71b800c96e7d580f2b5763ae772eb453b9b2b386cd67b"

ROOT_FILE = "model-root-aliases.safetensors"
SIDECAR_FILE = "model-mtp-sidecar.safetensors"
INDEX_FILE = "model.safetensors.index.json"
BUNDLE_MANIFEST_FILE = "strat01_mtp_hf_bundle.manifest.json"

METADATA_FILES: tuple[tuple[str, int], ...] = (
    ("config.json", 1 * 1024 * 1024),
    ("tokenizer.json", 16 * 1024 * 1024),
    ("tokenizer_config.json", 1 * 1024 * 1024),
    ("special_tokens_map.json", 1 * 1024 * 1024),
)
METADATA_SHA256 = {
    "config.json": "6a6b8260f08791c4968f70934903e9aa892a53ee3f2e2e05b61a68ea1ff33503",
    "tokenizer.json": "b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe",
    "tokenizer_config.json": "ded167f2cd1a755cebfce85d678e94bf8197114b0e8eaf89265fbd9b3d9f3150",
    "special_tokens_map.json": "5d4a6b397eb900972536128341670e9a499823c21291ffc3bf7f29f7a69dae0a",
}
HEADER_CAP = 2 * 1024 * 1024
COPY_BLOCK_BYTES = 1024 * 1024

DEFAULT_SIDECAR = Path(__file__).resolve().parent / "results" / contract.SIDECAR_NAME
DEFAULT_MANIFEST = DEFAULT_SIDECAR.with_name(DEFAULT_SIDECAR.name + ".manifest.json")


class BundleError(RuntimeError):
    """Raised when a pinned bundle invariant cannot be proved."""


@dataclass(frozen=True)
class TensorLocation:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def byte_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SidecarInspection:
    sidecar: Path
    manifest: Path
    tensors: Mapping[str, TensorLocation]
    payload_start: int
    expected_sha256: str
    expected_bytes: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(COPY_BLOCK_BYTES), b""):
                digest.update(block)
    except OSError as exc:
        raise BundleError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _safe_file(path: Path, label: str) -> Path:
    """Resolve the parent, keep the leaf literal, and reject links/special files."""
    raw = path.expanduser()
    if raw.name in {"", ".", ".."}:
        raise BundleError(f"{label} must name a file, got {path}")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"cannot resolve {label} parent {raw.parent}: {exc}") from exc
    candidate = parent / raw.name
    if candidate.is_symlink():
        raise BundleError(f"{label} must not be a symlink: {candidate}")
    if not candidate.is_file():
        raise BundleError(f"{label} is not a regular file: {candidate}")
    return candidate


def _safe_new_directory(path: Path) -> Path:
    raw = path.expanduser()
    if raw.name in {"", ".", ".."}:
        raise BundleError(f"output directory must name a new leaf, got {path}")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"cannot resolve output parent {raw.parent}: {exc}") from exc
    if not parent.is_dir():
        raise BundleError(f"output parent is not a directory: {parent}")
    candidate = parent / raw.name
    if candidate.exists() or candidate.is_symlink():
        raise BundleError(f"refusing to overwrite existing output directory: {candidate}")
    return candidate


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be a JSON object")
    return value


def _manifest_tensor_locations(manifest: Mapping[str, Any]) -> dict[str, TensorLocation]:
    records = manifest.get("tensors")
    if not isinstance(records, list):
        raise BundleError("sidecar manifest has no tensor list")
    result: dict[str, TensorLocation] = {}
    for number, item in enumerate(records):
        if not isinstance(item, dict):
            raise BundleError(f"sidecar manifest tensor {number} is not an object")
        name, dtype, shape, offsets = (
            item.get("name"), item.get("dtype"), item.get("shape"), item.get("sidecar_data_offsets")
        )
        if not isinstance(name, str) or name in result:
            raise BundleError(f"sidecar manifest has invalid or duplicate tensor name at {number}")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape
        ):
            raise BundleError(f"sidecar manifest tensor {name} has invalid dtype or shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
        ):
            raise BundleError(f"sidecar manifest tensor {name} has invalid offsets")
        result[name] = TensorLocation(name, dtype, tuple(shape), offsets[0], offsets[1])
    if len(result) != extract.MTP_KEY_COUNT:
        raise BundleError(f"sidecar manifest has {len(result)} tensors, expected {extract.MTP_KEY_COUNT}")
    return result


def _read_safetensors_header(path: Path) -> tuple[Mapping[str, Any], int]:
    try:
        with path.open("rb") as handle:
            encoded = handle.read(8)
            if len(encoded) != 8:
                raise BundleError("safetensors file is missing its header length")
            size = int.from_bytes(encoded, "little", signed=False)
            if size == 0 or size > HEADER_CAP:
                raise BundleError(f"safetensors header has unsafe length {size}")
            raw = handle.read(size)
            if len(raw) != size:
                raise BundleError("safetensors header is truncated")
    except OSError as exc:
        raise BundleError(f"cannot read safetensors header {path}: {exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid safetensors header in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError("safetensors header is not an object")
    return value, 8 + size


def inspect_sidecar(sidecar: Path, manifest_path: Path) -> SidecarInspection:
    """Validate manifest, safe_open metadata, and every manifest header record.

    This reads only the JSON manifest and safetensors metadata.  It does not
    map or materialize any MTP tensor payload.
    """
    safe_sidecar = _safe_file(sidecar, "sidecar")
    safe_manifest = _safe_file(manifest_path, "sidecar manifest")
    manifest = _load_object(safe_manifest, "sidecar manifest")
    report = contract.preflight_manifest(manifest)
    if report.get("metadata_contract") != "PASS":
        errors = report.get("errors")
        raise BundleError(f"sidecar manifest fails the pinned MTP contract: {errors}")
    declared = manifest.get("sidecar")
    if not isinstance(declared, dict):
        raise BundleError("sidecar manifest has no sidecar object")
    if declared.get("name") != contract.SIDECAR_NAME:
        raise BundleError("sidecar manifest names a different sidecar")
    if declared.get("bytes") != SIDECAR_BYTES or declared.get("sha256") != SIDECAR_SHA256:
        raise BundleError("sidecar manifest does not carry the pinned byte size and SHA-256")
    try:
        actual_size = safe_sidecar.stat().st_size
    except OSError as exc:
        raise BundleError(f"cannot stat sidecar {safe_sidecar}: {exc}") from exc
    if actual_size != SIDECAR_BYTES:
        raise BundleError(f"sidecar size {actual_size} != pinned {SIDECAR_BYTES}")

    locations = _manifest_tensor_locations(manifest)
    header, payload_start = _read_safetensors_header(safe_sidecar)
    header_names = set(header) - {"__metadata__"}
    if header_names != set(locations):
        raise BundleError("sidecar safetensors header keys disagree with its manifest")
    for name, location in locations.items():
        item = header.get(name)
        if not isinstance(item, dict):
            raise BundleError(f"sidecar header lacks object record for {name}")
        if item.get("dtype") != location.dtype or item.get("shape") != list(location.shape):
            raise BundleError(f"sidecar header dtype/shape differs from manifest for {name}")
        if item.get("data_offsets") != [location.start, location.end]:
            raise BundleError(f"sidecar header offsets differ from manifest for {name}")
    try:
        with safe_open(str(safe_sidecar), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(locations):
                raise BundleError("safe_open key set differs from sidecar manifest")
            for name, location in locations.items():
                slice_ = handle.get_slice(name)
                if tuple(slice_.get_shape()) != location.shape or slice_.get_dtype() != location.dtype:
                    raise BundleError(f"safe_open metadata differs from manifest for {name}")
    except BundleError:
        raise
    except Exception as exc:
        raise BundleError(f"safe_open metadata inspection failed: {exc}") from exc
    return SidecarInspection(safe_sidecar, safe_manifest, locations, payload_start, SIDECAR_SHA256, SIDECAR_BYTES)


def _https_url(url: str, label: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise BundleError(f"{label} URL must be absolute HTTPS, got {url!r}")
    return url


def _read_capped_https_range(
    url: str,
    cap: int,
    *,
    expected_total: int | None = None,
    opener: Callable[..., Any] = urlopen,
) -> tuple[bytes, int]:
    if cap <= 0:
        raise BundleError("HTTP cap must be positive")
    _https_url(url, "source")
    try:
        data, total = extract._read_range(  # Existing strict 206/Content-Range helper.
            url,
            0,
            cap - 1,
            expected_total=expected_total,
            allow_short_final=True,
            byte_cap=cap,
            opener=opener,
        )
    except extract.ExtractionError as exc:
        raise BundleError(str(exc)) from exc
    if total > cap:
        raise BundleError(f"remote file declares {total} bytes, exceeding cap {cap}")
    if len(data) != total:
        raise BundleError(f"remote Range body length {len(data)} != declared file size {total}")
    return data, total


def fetch_root_tensor(
    source_url: str,
    *,
    source_file_bytes: int,
    source_header_bytes: int,
    source_header_sha256: str,
    tensor_name: str,
    expected_shape: tuple[int, ...],
    expected_sha256: str,
    opener: Callable[..., Any] = urlopen,
) -> tuple[bytes, Mapping[str, Any]]:
    """Fetch exactly one BF16 tensor after proving its shard header and Range."""
    prefix_size = 8 + source_header_bytes
    try:
        prefix, total = extract._read_range(
            _https_url(source_url, "source shard"),
            0,
            prefix_size - 1,
            expected_total=source_file_bytes,
            byte_cap=prefix_size,
            opener=opener,
        )
        header = extract._parse_header(prefix, replace(extract.PINNED, source_header_bytes=source_header_bytes))
    except extract.ExtractionError as exc:
        raise BundleError(str(exc)) from exc
    if total != source_file_bytes or _sha256_bytes(prefix) != source_header_sha256:
        raise BundleError("source shard header does not match its pinned size/SHA-256")
    entry = header.get(tensor_name)
    if not isinstance(entry, dict):
        raise BundleError(f"source shard lacks required root tensor {tensor_name}")
    if entry.get("dtype") != "BF16" or entry.get("shape") != list(expected_shape):
        raise BundleError(f"source root tensor {tensor_name} has unexpected dtype or shape")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
        or offsets[0] < 0
        or offsets[1] <= offsets[0]
    ):
        raise BundleError(f"source root tensor {tensor_name} has invalid offsets")
    expected_bytes = 2
    for dimension in expected_shape:
        expected_bytes *= dimension
    if offsets[1] - offsets[0] != expected_bytes:
        raise BundleError(f"source root tensor {tensor_name} byte size disagrees with BF16 shape")
    absolute_start = prefix_size + offsets[0]
    absolute_end = prefix_size + offsets[1] - 1
    try:
        data, total = extract._read_range(
            source_url,
            absolute_start,
            absolute_end,
            expected_total=source_file_bytes,
            byte_cap=expected_bytes,
            opener=opener,
        )
    except extract.ExtractionError as exc:
        raise BundleError(str(exc)) from exc
    if total != source_file_bytes or len(data) != expected_bytes:
        raise BundleError(f"source root tensor {tensor_name} Range length is invalid")
    digest = _sha256_bytes(data)
    if digest != expected_sha256:
        raise BundleError(f"source root tensor {tensor_name} SHA-256 mismatch: {digest}")
    return data, {"source_data_offsets": offsets, "absolute_range": [absolute_start, absolute_end], "sha256": digest}


def fetch_pinned_root_norm(*, opener: Callable[..., Any] = urlopen) -> tuple[bytes, Mapping[str, Any]]:
    return fetch_root_tensor(
        extract._resolve_url(extract.PINNED, extract.PINNED.shard),
        source_file_bytes=extract.PINNED.source_file_bytes,
        source_header_bytes=extract.PINNED.source_header_bytes,
        source_header_sha256=SOURCE_HEADER_SHA256,
        tensor_name=ROOT_NORM_NAME,
        expected_shape=ROOT_NORM_SHAPE,
        expected_sha256=ROOT_NORM_SHA256,
        opener=opener,
    )


def fetch_pinned_metadata(*, opener: Callable[..., Any] = urlopen) -> Mapping[str, Mapping[str, Any]]:
    """Fetch inert source-revision JSON assets under explicit byte caps."""
    result: dict[str, Mapping[str, Any]] = {}
    for filename, cap in METADATA_FILES:
        url = extract._resolve_url(extract.PINNED, filename)
        data, total = _read_capped_https_range(url, cap, opener=opener)
        source_sha256 = _sha256_bytes(data)
        if source_sha256 != METADATA_SHA256[filename]:
            raise BundleError(f"pinned metadata SHA-256 mismatch for {filename}: {source_sha256}")
        normalization = None
        if filename == "config.json":
            data, normalization = normalize_config_for_transformers(data)
        result[filename] = {
            "bytes": data, "size": total, "sha256": source_sha256,
            "materialized_bytes": len(data), "materialized_sha256": _sha256_bytes(data),
            "normalization": normalization, "url": url, "cap": cap,
        }
    return result


def normalize_config_for_transformers(source: bytes) -> tuple[bytes, Mapping[str, Any]]:
    """Preserve the pinned JSON byte-for-byte except for integer 1 -> float 1.0.

    This is required by the installed Transformers strict config validator;
    the numerical value and all model weights are unchanged.
    """
    try:
        config = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"pinned config.json is invalid JSON: {exc}") from exc
    if not isinstance(config, dict) or type(config.get("routed_scaling_factor")) is not int or config["routed_scaling_factor"] != 1:
        raise BundleError("pinned config.json routed_scaling_factor is not integer 1")
    local, replacements = re.subn(
        rb'("routed_scaling_factor"\s*:\s*)1(?=\s*[,}])',
        lambda match: match.group(1) + b"1.0",
        source,
    )
    if replacements != 1 or json.loads(local)["routed_scaling_factor"] != 1.0:
        raise BundleError("could not normalize exactly one routed_scaling_factor field")
    return local, {"field": "routed_scaling_factor", "source_value": 1, "local_value": 1.0}


def _write_bytes_new(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise BundleError(f"cannot write {path}: {exc}") from exc


def _copy_exact_region(
    source: BinaryIO, destination: BinaryIO, start: int, length: int, aggregate: Any | None = None
) -> str:
    try:
        source.seek(start)
    except OSError as exc:
        raise BundleError(f"cannot seek sidecar source: {exc}") from exc
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        try:
            block = source.read(min(COPY_BLOCK_BYTES, remaining))
        except OSError as exc:
            raise BundleError(f"cannot read sidecar source: {exc}") from exc
        if not block:
            raise BundleError("sidecar ends within a required root-alias tensor")
        destination.write(block)
        digest.update(block)
        if aggregate is not None:
            aggregate.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def _root_alias_header(locations: Sequence[tuple[str, tuple[int, ...], int]]) -> bytes:
    position = 0
    value: dict[str, Any] = {"__metadata__": {"format": "pt", "bundle": "strat01_gigachat_mtp_hf_bundle_v1"}}
    for name, shape, byte_count in locations:
        value[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [position, position + byte_count]}
        position += byte_count
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def write_root_aliases(destination: Path, inspection: SidecarInspection, root_norm: bytes) -> Mapping[str, Any]:
    """Stream the two MTP aliases and write the verified 3 KiB source norm."""
    source_names = {
        "model.embed_tokens.weight": "model.layers.26.embed_tokens.weight",
        "lm_head.weight": "model.layers.26.shared_head.head.weight",
    }
    ordered: list[tuple[str, tuple[int, ...], int]] = []
    for target, source in source_names.items():
        location = inspection.tensors.get(source)
        if location is None or location.dtype != "BF16":
            raise BundleError(f"sidecar lacks required alias source {source}")
        ordered.append((target, location.shape, location.byte_count))
    if len(root_norm) != ROOT_NORM_BYTES or _sha256_bytes(root_norm) != ROOT_NORM_SHA256:
        raise BundleError("root norm bytes are not the pinned BF16 tensor")
    ordered.append((ROOT_NORM_NAME, ROOT_NORM_SHAPE, len(root_norm)))
    header = _root_alias_header(ordered)
    file_digest = hashlib.sha256()
    tensor_digests: dict[str, str] = {}
    try:
        with inspection.sidecar.open("rb") as source, destination.open("xb") as target:
            prefix = len(header).to_bytes(8, "little") + header
            target.write(prefix)
            file_digest.update(prefix)
            for name, _, size in ordered:
                if name == ROOT_NORM_NAME:
                    target.write(root_norm)
                    tensor_digests[name] = _sha256_bytes(root_norm)
                    file_digest.update(root_norm)
                    continue
                source_name = source_names[name]
                location = inspection.tensors[source_name]
                digest = _copy_exact_region(source, target, inspection.payload_start + location.start, size, file_digest)
                tensor_digests[name] = digest
            target.flush()
            os.fsync(target.fileno())
    except OSError as exc:
        raise BundleError(f"cannot write root aliases {destination}: {exc}") from exc
    return {"bytes": destination.stat().st_size, "sha256": file_digest.hexdigest(), "tensor_sha256": tensor_digests}


def build_index(entries: Iterable[tuple[str, str, int]]) -> Mapping[str, Any]:
    """Build the exact 213-key index and reject duplicate names/unsafe files."""
    weight_map: dict[str, str] = {}
    total_size = 0
    for name, filename, byte_count in entries:
        if not isinstance(name, str) or not name or name in weight_map:
            raise BundleError(f"index has invalid or duplicate tensor key {name!r}")
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise BundleError(f"index has unsafe filename {filename!r}")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise BundleError(f"index tensor {name} has invalid byte size")
        weight_map[name] = filename
        total_size += byte_count
    if len(weight_map) != extract.MTP_KEY_COUNT + 3:
        raise BundleError(f"index has {len(weight_map)} keys, expected {extract.MTP_KEY_COUNT + 3}")
    return {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}


def _materialize_sidecar(
    source: Path, destination: Path, expected_sha256: str, expected_bytes: int, *, verify_sha256: bool
) -> Mapping[str, Any]:
    """Hardlink same-volume sources; otherwise copy without touching the source."""
    if destination.exists() or destination.is_symlink():
        raise BundleError(f"refusing to overwrite sidecar destination: {destination}")
    mode = "copy"
    try:
        same_volume = source.stat().st_dev == destination.parent.stat().st_dev
    except OSError as exc:
        raise BundleError(f"cannot inspect sidecar filesystem: {exc}") from exc
    if same_volume:
        try:
            os.link(source, destination)
            mode = "hardlink"
        except OSError:
            mode = "copy"
    digest: str | None = None
    if mode == "copy":
        copied_digest = hashlib.sha256() if verify_sha256 else None
        try:
            with source.open("rb") as input_handle, destination.open("xb") as output_handle:
                for block in iter(lambda: input_handle.read(COPY_BLOCK_BYTES), b""):
                    output_handle.write(block)
                    if copied_digest is not None:
                        copied_digest.update(block)
                output_handle.flush()
                os.fsync(output_handle.fileno())
        except OSError as exc:
            raise BundleError(f"cannot copy MTP sidecar: {exc}") from exc
        digest = copied_digest.hexdigest() if copied_digest is not None else None
    elif verify_sha256:
        digest = _sha256_file(destination)
    if verify_sha256 and digest != expected_sha256:
        raise BundleError(f"MTP sidecar SHA-256 mismatch: {digest}")
    try:
        byte_count = destination.stat().st_size
    except OSError as exc:
        raise BundleError(f"cannot stat materialized sidecar: {exc}") from exc
    if byte_count != expected_bytes:
        raise BundleError(f"materialized sidecar size {byte_count} != expected {expected_bytes}")
    return {"filename": destination.name, "mode": mode, "bytes": byte_count, "sha256": digest, "sha256_verified": verify_sha256}


def _create_temp_directory(final: Path) -> Path:
    temporary = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(mode=0o700)
    except OSError as exc:
        raise BundleError(f"cannot create private bundle temporary directory {temporary}: {exc}") from exc
    return temporary


def _finalize_directory(temporary: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise BundleError(f"refusing to overwrite output directory during finalization: {final}")
    try:
        os.rename(temporary, final)
    except FileExistsError as exc:
        raise BundleError(f"output directory appeared during finalization: {final}") from exc
    except OSError as exc:
        raise BundleError(f"cannot atomically finalize output directory {final}: {exc}") from exc


def bundle_plan(sidecar: Path, manifest: Path, output_dir: Path | None) -> Mapping[str, Any]:
    """No-I/O plan: describe the bounded operation without reading payloads or network."""
    return {
        "schema": "strat01_gigachat_mtp_hf_bundle_plan_v1",
        "mode": "PLAN_ONLY_NO_NETWORK_NO_SIDECAR_PAYLOAD_READ_NO_WRITES",
        "input_sidecar": str(sidecar),
        "input_manifest": str(manifest),
        "output_directory": None if output_dir is None else str(output_dir),
        "expected_sidecar": {"bytes": SIDECAR_BYTES, "sha256": SIDECAR_SHA256, "tensors": extract.MTP_KEY_COUNT},
        "root_aliases": ["model.embed_tokens.weight", "lm_head.weight", ROOT_NORM_NAME],
        "source_revision": extract.PINNED.revision,
        "metadata_files": [{"name": name, "cap": cap, "source_sha256": METADATA_SHA256[name]} for name, cap in METADATA_FILES],
        "config_normalization": "EXECUTE_ONLY: routed_scaling_factor JSON integer 1 -> float 1.0 for installed Transformers",
        "network": "EXECUTE_ONLY: 4 capped HTTPS JSON Range requests + source header + exact 3072-byte root norm Range",
        "sidecar_materialization": "EXECUTE_ONLY: hardlink same-volume when possible, otherwise copy; never symlink",
    }


def execute_bundle(
    sidecar: Path,
    manifest: Path,
    output_dir: Path,
    *,
    verify_sidecar_sha256: bool = True,
    opener: Callable[..., Any] = urlopen,
) -> Mapping[str, Any]:
    """Build a new, complete, bounded HF directory and never overwrite it."""
    final = _safe_new_directory(output_dir)
    inspection = inspect_sidecar(sidecar, manifest)
    root_norm, root_norm_provenance = fetch_pinned_root_norm(opener=opener)
    metadata = fetch_pinned_metadata(opener=opener)
    temporary = _create_temp_directory(final)
    finalized = False
    try:
        sidecar_info = _materialize_sidecar(
            inspection.sidecar, temporary / SIDECAR_FILE, inspection.expected_sha256, inspection.expected_bytes,
            verify_sha256=verify_sidecar_sha256,
        )
        aliases_info = write_root_aliases(temporary / ROOT_FILE, inspection, root_norm)
        for filename, record in metadata.items():
            _write_bytes_new(temporary / filename, record["bytes"])
        index_entries = [(name, SIDECAR_FILE, location.byte_count) for name, location in inspection.tensors.items()]
        index_entries.extend(
            [
                ("model.embed_tokens.weight", ROOT_FILE, inspection.tensors["model.layers.26.embed_tokens.weight"].byte_count),
                ("lm_head.weight", ROOT_FILE, inspection.tensors["model.layers.26.shared_head.head.weight"].byte_count),
                (ROOT_NORM_NAME, ROOT_FILE, ROOT_NORM_BYTES),
            ]
        )
        index = build_index(index_entries)
        _write_bytes_new(temporary / INDEX_FILE, (json.dumps(index, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        record = {
            "schema": "strat01_gigachat_mtp_hf_bundle_manifest_v1",
            "limitations": [
                "MTP_ONLY_INPUT_BUNDLE_NOT_A_FULL_HF_CHECKPOINT",
                "NO_GGUF_WRITTEN",
                "NO_MODEL_CODE_EXECUTED",
                "NO_LOGITS_ACCEPTANCE_SPEED_OR_T4_MEASUREMENT",
            ],
            "source": {"repository": extract.PINNED.repository, "revision": extract.PINNED.revision, "shard": extract.PINNED.shard},
            "input_sidecar": {
                "path": str(inspection.sidecar), "manifest": str(inspection.manifest), "expected_sha256": inspection.expected_sha256,
                "expected_bytes": inspection.expected_bytes, **sidecar_info,
            },
            "root_aliases": {"filename": ROOT_FILE, **aliases_info, "root_norm_source": root_norm_provenance},
            "remote_json": {name: {key: value for key, value in item.items() if key != "bytes"} for name, item in metadata.items()},
            "index": {"filename": INDEX_FILE, "keys": len(index["weight_map"]), "total_size": index["metadata"]["total_size"]},
        }
        _write_bytes_new(temporary / BUNDLE_MANIFEST_FILE, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        _finalize_directory(temporary, final)
        finalized = True
        return {"output_directory": str(final), "manifest": str(final / BUNDLE_MANIFEST_FILE), **record}
    finally:
        if not finalized and temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build a bounded pinned GigaChat 3.1 MTP-only HF converter bundle")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="default: print plan only; no network, payload read, or writes")
    action.add_argument("--execute", action="store_true", help="explicitly create a new bounded HF bundle directory")
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR, help="existing pinned MTP sidecar")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="existing sidecar manifest")
    parser.add_argument("--out-dir", type=Path, help="required new output directory for --execute")
    parser.add_argument("--skip-sidecar-sha256", action="store_true", help="explicitly skip the one full 1.6 GiB SHA-256 pass (records unverified status)")
    args = parser.parse_args(argv)
    if args.skip_sidecar_sha256 and not args.execute:
        parser.error("--skip-sidecar-sha256 requires --execute")
    if args.execute and args.out_dir is None:
        parser.error("--execute requires an explicit --out-dir")
    try:
        if not args.execute:
            print(json.dumps(bundle_plan(args.sidecar, args.manifest, args.out_dir), sort_keys=True, indent=2))
            return 0
        result = execute_bundle(
            args.sidecar, args.manifest, args.out_dir, verify_sidecar_sha256=not args.skip_sidecar_sha256
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except BundleError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
