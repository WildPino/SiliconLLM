"""Safe, opt-in acquisition for the pinned STRAT-02 safetensors shards.

This module deliberately has no import-time network action.  ``--preflight``
is read-only and offline: it proves the local metadata snapshot using
``strat02_mmap_teacher.py`` before a caller may use ``--download``.  Download
mode asks ``hf_hub_download`` for *only* the eleven allowlisted shard names,
at the immutable commit below, in Hugging Face's ordinary cache (never in a
``local_dir`` copy).  The cap is on final, hash-verified payload bytes in the
cache; it is not a claim about wire bytes.  The Hub client can retry or add
protocol overhead, so on-wire bytes are reported as ``unknown`` unless a
separate transfer instrumenter is supplied.

The audit JSON is intentionally content-free: it records filenames, sizes,
and digests, but never config, tokenizer, or model source text.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


REPOSITORY = "allenai/StdMoE_1b14b_1T_Preanneal"
REVISION = "d2a4949c9d4ad6cf47fbac131f7e020077332b21"
MANIFEST_PATH = Path(__file__).with_name("strat02_weight_sources.json")
EXPECTED_MANIFEST_SHA256 = "7aab6424b2390d5b1956793df58aa6be56a279e2d7ec01568d79a29a525a179e"
SHARD_NAMES = frozenset(
    f"model-{number:05d}-of-00011.safetensors" for number in range(1, 12)
)
MIN_FREE_RAM_BYTES = 55 * 1024**3
MIN_FREE_CACHE_BYTES = 65 * 1024**3
FINAL_VALID_PAYLOAD_CAP_BYTES = 54_275_336_216
TRANSFORMERS_VERSION = "4.57.1"
TOKENIZERS_VERSION = "0.22.2"
USE_HUB_KERNELS_VALUE = "NO"
TOKENIZER_ASSET_SHA256 = {
    "tokenizer.json": "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca",
    "tokenizer_config.json": "733b2ec0f743cf206dd20335eacbd4427c85487fd6f5216bb361f92a85314315",
    "special_tokens_map.json": "4cb2b52960bababa8ec27153831e405ef7811e3fad1226106e09c28e715cfb21",
    "vocab.json": "9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195",
    "merges.txt": "b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502",
}
HASH_BLOCK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 256 * 1024 * 1024
AUDIT_BASENAME = "strat02_acquire_weights.audit.json"


class AcquisitionError(RuntimeError):
    """Raised when acquisition cannot be proved safe or complete."""


@dataclass(frozen=True)
class AcquisitionPreflight:
    """Read-only proof required before shard acquisition can begin."""

    snapshot: Path
    cache_dir: Path
    manifest: Mapping[str, Any]
    source_status: Mapping[str, Any]
    runtime: Mapping[str, str]
    tokenizer_identity: Mapping[str, Any]
    ram_available_bytes: int
    cache_free_bytes: int
    byte_cap: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcquisitionError(f"{label} must be a JSON object: {path}")
    return value


def _default_hf_cache_dir() -> Path:
    """Match Hugging Face's cache convention without importing its client.

    Keeping this pure is important: ``--preflight`` must not even need the
    Hub package, much less contact it.  ``--download`` passes this same value
    as ``cache_dir`` and never passes ``local_dir``.
    """

    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser() / "hub").resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return (base / "huggingface" / "hub").resolve()


def _snapshot_path(cache_dir: Path) -> Path:
    return cache_dir / ("models--" + REPOSITORY.replace("/", "--")) / "snapshots" / REVISION


def _available_ram_bytes() -> int | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _import_teacher() -> Any:
    """Load the local, offline verifier only when preflight is requested."""

    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    try:
        return importlib.import_module("strat02_mmap_teacher")
    except ImportError as exc:
        raise AcquisitionError("cannot import local strat02_mmap_teacher.py") from exc


def _require_pinned_runtime(
    *,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    environ: Mapping[str, str] = os.environ,
) -> Mapping[str, str]:
    """Require the target runtime without importing model or Hub code."""

    try:
        transformers_version = version_getter("transformers")
        tokenizers_version = version_getter("tokenizers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise AcquisitionError(f"pinned runtime package is missing: {exc.name}") from exc
    if transformers_version != TRANSFORMERS_VERSION:
        raise AcquisitionError(
            f"require transformers=={TRANSFORMERS_VERSION}; found {transformers_version!r}"
        )
    if tokenizers_version != TOKENIZERS_VERSION:
        raise AcquisitionError(
            f"require tokenizers=={TOKENIZERS_VERSION}; found {tokenizers_version!r}"
        )
    hub_kernels = environ.get("USE_HUB_KERNELS")
    if hub_kernels != USE_HUB_KERNELS_VALUE:
        raise AcquisitionError(
            f"USE_HUB_KERNELS must be exactly {USE_HUB_KERNELS_VALUE!r}; found {hub_kernels!r}"
        )
    return {
        "transformers": transformers_version,
        "tokenizers": tokenizers_version,
        "USE_HUB_KERNELS": hub_kernels,
    }


def _validate_shard_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_payload_cap: int = FINAL_VALID_PAYLOAD_CAP_BYTES,
) -> int:
    """Independently constrain acquisition to the exact eleven safe names."""

    if manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION:
        raise AcquisitionError("manifest repository/revision is not the STRAT-02 immutable pin")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != len(SHARD_NAMES):
        raise AcquisitionError("manifest must contain exactly eleven shard records")
    names: set[str] = set()
    total = 0
    for entry in shards:
        if not isinstance(entry, dict) or set(entry) != {"name", "size", "sha256"}:
            raise AcquisitionError("each shard record must contain only name, size, and sha256")
        name, size, digest = entry["name"], entry["size"], entry["sha256"]
        if not isinstance(name, str) or name not in SHARD_NAMES or Path(name).name != name:
            raise AcquisitionError(f"manifest contains disallowed shard name: {name!r}")
        if name in names:
            raise AcquisitionError(f"manifest contains duplicate shard name: {name}")
        if not isinstance(size, int) or size <= 0:
            raise AcquisitionError(f"invalid shard size for {name}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise AcquisitionError(f"invalid shard SHA-256 for {name}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise AcquisitionError(f"invalid shard SHA-256 for {name}") from exc
        names.add(name)
        total += size
    if names != SHARD_NAMES:
        raise AcquisitionError("manifest shard names do not equal the fixed eleven-file allowlist")
    if manifest.get("total_shard_file_bytes") != total:
        raise AcquisitionError("manifest total_shard_file_bytes disagrees with its shard records")
    if total != expected_payload_cap:
        raise AcquisitionError(
            f"manifest valid-payload bytes do not equal the required {expected_payload_cap}-byte cap"
        )
    return total


def _load_pinned_gpt2_tokenizer(snapshot: Path) -> tuple[Any, type[Any]]:
    """Load only installed GPT-2 tokenizer code with local assets, never Hub code."""

    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained(str(snapshot), local_files_only=True), GPT2TokenizerFast


def _tokenizer_identity(
    snapshot: Path,
    *,
    expected_asset_sha256: Mapping[str, str] = TOKENIZER_ASSET_SHA256,
    tokenizer_loader: Callable[[Path], tuple[Any, type[Any]]] = _load_pinned_gpt2_tokenizer,
) -> Mapping[str, Any]:
    """Prove all pinned tokenizer assets and exact GPT2TokenizerFast identity.

    ``tokenizer_config.json`` is one of the exact-hashed assets, so its bytes
    are pinned.  Loading it through the installed, pinned runtime and requiring
    an exact ``GPT2TokenizerFast`` type prevents a look-alike configuration
    from being accepted.  No remote tokenizer code is executed.
    """

    asset_sha256: dict[str, str] = {}
    for name, expected in expected_asset_sha256.items():
        path = snapshot / name
        if not path.is_file():
            raise AcquisitionError(f"pinned tokenizer asset is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise AcquisitionError(f"pinned tokenizer asset SHA-256 mismatch for {name}")
        asset_sha256[name] = actual
    try:
        tokenizer, fast_class = tokenizer_loader(snapshot)
    except Exception as exc:
        raise AcquisitionError(f"cannot load pinned GPT2TokenizerFast locally: {exc}") from exc
    if type(tokenizer) is not fast_class:
        raise AcquisitionError("tokenizer configuration did not instantiate exact GPT2TokenizerFast")
    records = [{"name": name, "sha256": asset_sha256[name]} for name in sorted(asset_sha256)]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "asset_sha256": asset_sha256,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "tokenizer_class": fast_class.__name__,
    }


def _stage_preflight(
    snapshot: Path | None = None,
    *,
    cache_dir: Path | None = None,
    manifest_path: Path = MANIFEST_PATH,
    source_preflight: Callable[..., Any] | None = None,
    ram_getter: Callable[[], int | None] = _available_ram_bytes,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    expected_payload_cap: int = FINAL_VALID_PAYLOAD_CAP_BYTES,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
    runtime_check: Callable[[], Mapping[str, str]] = _require_pinned_runtime,
    expected_tokenizer_asset_sha256: Mapping[str, str] = TOKENIZER_ASSET_SHA256,
    tokenizer_loader: Callable[[Path], tuple[Any, type[Any]]] = _load_pinned_gpt2_tokenizer,
) -> AcquisitionPreflight:
    """Run the offline, read-only gate.  No Hub client is imported here."""

    chosen_cache = (cache_dir or _default_hf_cache_dir()).resolve()
    expected_snapshot = _snapshot_path(chosen_cache).resolve()
    chosen_snapshot = (snapshot or expected_snapshot).expanduser().resolve()
    if chosen_snapshot != expected_snapshot:
        raise AcquisitionError(
            "snapshot must be the pinned snapshot in Hugging Face's default cache; "
            "local_dir copies are intentionally unsupported"
        )
    if not chosen_snapshot.is_dir():
        raise AcquisitionError(f"pinned metadata snapshot is absent: {chosen_snapshot}")
    resolved_manifest = manifest_path.expanduser().resolve()
    actual_manifest_sha256 = _sha256_file(resolved_manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise AcquisitionError(
            "weight-source manifest SHA-256 mismatch: "
            f"{actual_manifest_sha256} != {expected_manifest_sha256}"
        )
    runtime = runtime_check()
    verifier = source_preflight or _import_teacher().preflight
    try:
        report = verifier(snapshot=chosen_snapshot, manifest_path=manifest_path)
    except Exception as exc:
        raise AcquisitionError(f"offline teacher metadata proof failed: {exc}") from exc
    if Path(report.snapshot).resolve() != chosen_snapshot:
        raise AcquisitionError("teacher verifier returned a different snapshot")
    if report.source_status.get("transformers_ok") is not True:
        raise AcquisitionError("teacher verifier did not confirm transformers==4.57.1")
    if report.source_status.get("use_hub_kernels") != USE_HUB_KERNELS_VALUE:
        raise AcquisitionError("teacher verifier did not observe USE_HUB_KERNELS=NO")
    manifest = report.manifest
    cap = _validate_shard_manifest(manifest, expected_payload_cap=expected_payload_cap)
    tokenizer = _tokenizer_identity(
        chosen_snapshot,
        expected_asset_sha256=expected_tokenizer_asset_sha256,
        tokenizer_loader=tokenizer_loader,
    )
    available_ram = ram_getter()
    if available_ram is None or available_ram < MIN_FREE_RAM_BYTES:
        raise AcquisitionError(
            f"available RAM must be at least 55 GiB; observed {available_ram!r} bytes"
        )
    try:
        free_cache = int(disk_usage(chosen_cache).free)
    except OSError as exc:
        raise AcquisitionError(f"cannot inspect Hugging Face cache filesystem: {chosen_cache}: {exc}") from exc
    if free_cache < MIN_FREE_CACHE_BYTES:
        raise AcquisitionError(
            f"free cache disk must be at least 65 GiB; observed {free_cache} bytes"
        )
    return AcquisitionPreflight(
        snapshot=chosen_snapshot,
        cache_dir=chosen_cache,
        manifest=manifest,
        source_status=report.source_status,
        runtime=runtime,
        tokenizer_identity=tokenizer,
        ram_available_bytes=available_ram,
        cache_free_bytes=free_cache,
        byte_cap=cap,
    )


def _verify_shard(path: Path, entry: Mapping[str, Any]) -> None:
    try:
        stat = path.stat()
    except OSError as exc:
        raise AcquisitionError(f"cannot stat shard {path}: {exc}") from exc
    if not path.is_file():
        raise AcquisitionError(f"shard path is not a regular file: {path}")
    if stat.st_size != entry["size"]:
        raise AcquisitionError(
            f"shard size mismatch for {path.name}: {stat.st_size} != {entry['size']}"
        )
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        raise AcquisitionError(f"cannot hash shard {path}: {exc}") from exc
    if actual != entry["sha256"]:
        raise AcquisitionError(
            f"shard SHA-256 mismatch for {path.name}: {actual} != {entry['sha256']}"
        )


def _periodic_tqdm() -> type[Any]:
    """A small HF-compatible progress adapter; it prints at bounded intervals."""

    class PeriodicProgress:
        def __init__(self, total: int | None = None, desc: str | None = None, disable: bool = False, **_: Any) -> None:
            self.total = total
            self.desc = desc or "download"
            self.disable = disable
            self.count = 0
            self.next_report = PROGRESS_BYTES

        def __enter__(self) -> "PeriodicProgress":
            return self

        def __exit__(self, *_: Any) -> None:
            self.close()

        def update(self, amount: int = 1) -> None:
            self.count += amount
            if not self.disable and self.count >= self.next_report:
                total = "?" if self.total is None else str(self.total)
                print(f"  transfer progress: {self.desc}: {self.count}/{total} bytes", flush=True)
                self.next_report = self.count + PROGRESS_BYTES

        def close(self) -> None:
            return None

    return PeriodicProgress


def _audit_payload(
    preflight: AcquisitionPreflight,
    outcomes: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    by_name = {outcome["name"]: outcome for outcome in outcomes}
    if set(by_name) != SHARD_NAMES:
        raise AcquisitionError("audit requires one verified outcome for every allowlisted shard")
    return {
        "schema": "strat02_acquisition_audit_v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "final_valid_payload_cap_bytes": preflight.byte_cap,
        "on_wire_bytes": "unknown",
        "shards": [
            {
                "name": entry["name"],
                "size": entry["size"],
                "sha256": entry["sha256"],
                "status": by_name[entry["name"]]["status"],
            }
            for entry in preflight.manifest["shards"]
        ],
        "tokenizer_identity_sha256": preflight.tokenizer_identity["identity_sha256"],
        "tokenizer_asset_sha256": preflight.tokenizer_identity["asset_sha256"],
        "runtime": preflight.runtime,
        "source_code_sha256": preflight.source_status.get("code_sha256", {}),
    }


def _validate_existing_audit(existing: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Accept a prior audit only if its immutable proof remains equivalent.

    A resumed invocation naturally sees every completed file as cached, while
    the first audit must preserve which files it actually downloaded.  Thus
    existing audit status is validated, not rewritten to reflect a later run.
    """

    static_keys = {
        "schema", "repository", "revision", "manifest_sha256",
        "final_valid_payload_cap_bytes", "on_wire_bytes", "tokenizer_identity_sha256",
        "tokenizer_asset_sha256", "runtime", "source_code_sha256",
    }
    if {key: existing.get(key) for key in static_keys} != {key: payload.get(key) for key in static_keys}:
        raise AcquisitionError("write-once audit has different immutable metadata")
    expected = {entry["name"]: entry for entry in payload["shards"]}
    actual = existing.get("shards")
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise AcquisitionError("write-once audit has an invalid shard list")
    seen: set[str] = set()
    for entry in actual:
        if not isinstance(entry, dict) or entry.get("status") not in {"cached", "downloaded"}:
            raise AcquisitionError("write-once audit has an invalid cached/downloaded status")
        name = entry.get("name")
        if name not in expected or {
            "name": entry.get("name"), "size": entry.get("size"), "sha256": entry.get("sha256"),
        } != {
            "name": expected[name]["name"], "size": expected[name]["size"], "sha256": expected[name]["sha256"],
        }:
            raise AcquisitionError("write-once audit has different final shard evidence")
        seen.add(name)
    if seen != SHARD_NAMES:
        raise AcquisitionError("write-once audit does not cover every allowlisted shard")


def _write_or_validate_audit(path: Path, payload: Mapping[str, Any]) -> str:
    """Write exactly once, or validate a compatible immutable prior audit."""

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = _read_json(path, "existing acquisition audit")
        _validate_existing_audit(existing, payload)
        return "existing"
    if not path.parent.is_dir():
        raise AcquisitionError(f"audit parent directory does not exist: {path.parent}")
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except FileExistsError:
        existing = _read_json(path, "existing acquisition audit")
        _validate_existing_audit(existing, payload)
        return "existing"
    return "written"


def _download_shards(
    preflight: AcquisitionPreflight,
    *,
    downloader: Callable[..., str] | None = None,
) -> list[Mapping[str, Any]]:
    """Download only missing shards, sequentially, after all existing files pass.

    There is deliberately no outer retry loop and no ``force_download``.  A
    failure leaves the Hub cache untouched by this module; rerunning verifies
    any completed files and resumes only the still-missing allowlisted name.
    """

    expected_total = sum(entry["size"] for entry in preflight.manifest["shards"])
    if expected_total != preflight.byte_cap:
        raise AcquisitionError("final valid-payload cap is not exactly the manifest shard-byte sum")
    entries = list(preflight.manifest["shards"])
    # Fail before contacting the Hub if *any* user-preexisting shard is bad.
    existing: dict[str, Path] = {}
    for entry in entries:
        path = preflight.snapshot / entry["name"]
        # ``exists`` is false for a dangling symlink.  ``lexists`` makes that
        # a corruption error instead of giving the Hub a chance to replace it.
        if os.path.lexists(path):
            _verify_shard(path, entry)
            existing[entry["name"]] = path

    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise AcquisitionError("--download requires huggingface_hub") from exc
        downloader = hf_hub_download
    parameters = inspect.signature(downloader).parameters
    outcomes: list[Mapping[str, Any]] = []
    accounted = 0
    for position, entry in enumerate(entries, start=1):
        name = entry["name"]
        if accounted + entry["size"] > preflight.byte_cap:
            raise AcquisitionError("refusing shard: final valid-payload cap would be exceeded")
        path = existing.get(name)
        source = "cached"
        if path is None:
            print(
                f"[{position}/{len(entries)}] downloading {name} "
                f"({entry['size']} bytes; pinned revision)",
                flush=True,
            )
            kwargs: dict[str, Any] = {
                "repo_id": REPOSITORY,
                "filename": name,
                "repo_type": "model",
                "revision": REVISION,
                "cache_dir": str(preflight.cache_dir),
                "force_download": False,
            }
            if "tqdm_class" in parameters:
                kwargs["tqdm_class"] = _periodic_tqdm()
            returned = Path(downloader(**kwargs)).resolve()
            expected_path = (preflight.snapshot / name).resolve()
            if returned != expected_path:
                raise AcquisitionError(
                    f"Hub returned an unexpected path for {name}: {returned}; expected {expected_path}"
                )
            path = returned
            source = "downloaded"
        _verify_shard(path, entry)
        accounted += entry["size"]
        outcomes.append({"name": name, "status": source, "size": entry["size"]})
        print(f"[{position}/{len(entries)}] verified {name} ({source})", flush=True)
    if accounted != preflight.byte_cap:
        raise AcquisitionError("verified shard bytes do not equal the exact final valid-payload cap")
    return outcomes


def _public_preflight(preflight: AcquisitionPreflight) -> Mapping[str, Any]:
    """Report only hashes and metadata; never emit config/tokenizer source text."""

    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "snapshot": str(preflight.snapshot),
        "cache_dir": str(preflight.cache_dir),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "shard_count": len(preflight.manifest["shards"]),
        "final_valid_payload_cap_bytes": preflight.byte_cap,
        "on_wire_bytes": "unknown",
        "ram_available_bytes": preflight.ram_available_bytes,
        "cache_free_bytes": preflight.cache_free_bytes,
        "runtime": preflight.runtime,
        "source_code_sha256": preflight.source_status.get("code_sha256", {}),
        "tokenizer_identity": preflight.tokenizer_identity,
        "note": "Read-only preflight; no Hub request and no shard payload was opened.",
    }


def selftest() -> None:
    """Exercise mocked, tiny downloads and all failure gates without network access."""

    with tempfile.TemporaryDirectory(prefix="strat02-acquire-selftest-") as temporary:
        root = Path(temporary)
        cache = root / "hub"
        snapshot = _snapshot_path(cache)
        snapshot.mkdir(parents=True)
        # Tiny stand-ins for the five exact-pinned tokenizer assets.
        tokenizer_bytes = {
            "tokenizer.json": b'{"version":"1.0"}',
            "tokenizer_config.json": b'{"tokenizer_class":"Tiny"}',
            "special_tokens_map.json": b"{}",
            "vocab.json": b'{"tiny":0}',
            "merges.txt": b"#version: 0.2\n",
        }
        fixture_tokenizer_sha256 = {
            name: hashlib.sha256(content).hexdigest() for name, content in tokenizer_bytes.items()
        }
        for name, content in tokenizer_bytes.items():
            (snapshot / name).write_bytes(content)
        FakeGPT2TokenizerFast = type("GPT2TokenizerFast", (), {})

        def fixture_tokenizer_loader(_: Path) -> tuple[Any, type[Any]]:
            return FakeGPT2TokenizerFast(), FakeGPT2TokenizerFast

        runtime = {
            "transformers": TRANSFORMERS_VERSION,
            "tokenizers": TOKENIZERS_VERSION,
            "USE_HUB_KERNELS": USE_HUB_KERNELS_VALUE,
        }

        def expect_error(callback: Callable[[], Any], message: str) -> None:
            try:
                callback()
            except AcquisitionError:
                return
            raise AssertionError(message)

        # Planted runtime negatives do not import packages or contact the Hub.
        versions = {"transformers": TRANSFORMERS_VERSION, "tokenizers": TOKENIZERS_VERSION}
        expect_error(
            lambda: _require_pinned_runtime(
                version_getter=lambda package: versions[package],
                environ={"USE_HUB_KERNELS": "YES"},
            ),
            "unsafe USE_HUB_KERNELS did not fail closed",
        )
        expect_error(
            lambda: _require_pinned_runtime(
                version_getter=lambda package: "wrong" if package == "transformers" else versions[package],
                environ={"USE_HUB_KERNELS": USE_HUB_KERNELS_VALUE},
            ),
            "wrong transformers version did not fail closed",
        )
        shards = []
        fixture_bytes: dict[str, bytes] = {}
        for number, name in enumerate(sorted(SHARD_NAMES), start=1):
            content = bytes([number]) * (number + 2)
            fixture_bytes[name] = content
            shards.append({"name": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        manifest = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "total_shard_file_bytes": sum(item["size"] for item in shards),
            "shards": shards,
        }
        manifest_path = root / "tiny_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        # A verified cache hit proves the no-copy, existing-path branch.
        first = shards[0]
        (snapshot / first["name"]).write_bytes(fixture_bytes[first["name"]])
        network_calls: list[str] = []

        def no_network_source_preflight(**_: Any) -> Any:
            network_calls.append("source-preflight")
            return SimpleNamespace(
                snapshot=snapshot,
                manifest=manifest,
                source_status={
                    "code_sha256": {}, "transformers_ok": True,
                    "use_hub_kernels": USE_HUB_KERNELS_VALUE,
                },
            )

        # The source preflight mock stands in for the existing offline verifier;
        # it never invokes a network function.
        report = _stage_preflight(
            snapshot,
            cache_dir=cache,
            manifest_path=manifest_path,
            source_preflight=no_network_source_preflight,
            ram_getter=lambda: MIN_FREE_RAM_BYTES,
            disk_usage=lambda _: SimpleNamespace(free=MIN_FREE_CACHE_BYTES),
            expected_payload_cap=manifest["total_shard_file_bytes"],
            expected_manifest_sha256=_sha256_file(manifest_path),
            runtime_check=lambda: runtime,
            expected_tokenizer_asset_sha256=fixture_tokenizer_sha256,
            tokenizer_loader=fixture_tokenizer_loader,
        )
        assert network_calls == ["source-preflight"]

        # A single altered tokenizer byte must fail before the Hub is imported.
        altered_name = "vocab.json"
        (snapshot / altered_name).write_bytes(b"corrupt")
        expect_error(
            lambda: _tokenizer_identity(
                snapshot,
                expected_asset_sha256=fixture_tokenizer_sha256,
                tokenizer_loader=fixture_tokenizer_loader,
            ),
            "mismatched tokenizer asset did not fail closed",
        )
        (snapshot / altered_name).write_bytes(tokenizer_bytes[altered_name])

        def bad_teacher_report(**_: Any) -> Any:
            return SimpleNamespace(
                snapshot=snapshot,
                manifest=manifest,
                source_status={
                    "code_sha256": {}, "transformers_ok": False,
                    "use_hub_kernels": USE_HUB_KERNELS_VALUE,
                },
            )

        expect_error(
            lambda: _stage_preflight(
                snapshot,
                cache_dir=cache,
                manifest_path=manifest_path,
                source_preflight=bad_teacher_report,
                ram_getter=lambda: MIN_FREE_RAM_BYTES,
                disk_usage=lambda _: SimpleNamespace(free=MIN_FREE_CACHE_BYTES),
                expected_payload_cap=manifest["total_shard_file_bytes"],
                expected_manifest_sha256=_sha256_file(manifest_path),
                runtime_check=lambda: runtime,
                expected_tokenizer_asset_sha256=fixture_tokenizer_sha256,
                tokenizer_loader=fixture_tokenizer_loader,
            ),
            "teacher transformers_ok=false did not fail closed",
        )

        downloaded: list[str] = []

        def mocked_download(**kwargs: Any) -> str:
            name = kwargs["filename"]
            assert kwargs["repo_id"] == REPOSITORY and kwargs["revision"] == REVISION
            assert kwargs["cache_dir"] == str(cache) and kwargs["force_download"] is False
            assert "local_dir" not in kwargs
            downloaded.append(name)
            target = snapshot / name
            target.write_bytes(fixture_bytes[name])
            return str(target)

        outcomes = _download_shards(report, downloader=mocked_download)
        assert outcomes[0]["status"] == "cached"
        assert downloaded == [entry["name"] for entry in shards[1:]]
        audit = snapshot / AUDIT_BASENAME
        payload = _audit_payload(report, outcomes)
        assert _write_or_validate_audit(audit, payload) == "written"
        assert _write_or_validate_audit(audit, payload) == "existing"

        # Existing corruption must abort before the downloader is called.
        corrupted = snapshot / shards[3]["name"]
        corrupted.write_bytes(b"bad")
        calls_before_corruption = len(downloaded)
        try:
            _download_shards(report, downloader=mocked_download)
        except AcquisitionError:
            pass
        else:
            raise AssertionError("preexisting corruption did not fail closed")
        assert len(downloaded) == calls_before_corruption
        corrupted.write_bytes(fixture_bytes[shards[3]["name"]])

        # Simulate a dangling snapshot symlink: lexists is true although stat
        # fails.  This must not become a Hub repair/download attempt.
        from unittest.mock import patch

        broken = snapshot / shards[-1]["name"]
        broken.unlink()
        real_lexists = os.path.lexists
        calls_before_broken_link = len(downloaded)
        with patch(
            "os.path.lexists",
            side_effect=lambda candidate: Path(candidate) == broken or real_lexists(candidate),
        ):
            expect_error(
                lambda: _download_shards(report, downloader=mocked_download),
                "broken snapshot symlink did not fail closed",
            )
        assert len(downloaded) == calls_before_broken_link
        broken.write_bytes(fixture_bytes[broken.name])

        # A cap below the exact manifest sum must fail before any hub action.
        limited = replace(report, byte_cap=report.byte_cap - 1)
        try:
            _download_shards(limited, downloader=mocked_download)
        except AcquisitionError:
            pass
        else:
            raise AssertionError("non-exact cap did not fail closed")

        # Drive the CLI's --preflight branch with a mocked offline gate.
        original = _stage_preflight
        try:
            globals()["_stage_preflight"] = lambda *_, **__: report
            with contextlib.redirect_stdout(io.StringIO()):
                assert _main(["--preflight"]) == 0
        finally:
            globals()["_stage_preflight"] = original
        assert network_calls == ["source-preflight"]


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="safe, pinned STRAT-02 safetensors acquisition")
    parser.add_argument("--preflight", action="store_true", help="offline/read-only metadata and host gate")
    parser.add_argument("--download", action="store_true", help="opt in to sequential Hub download of the 11 shards")
    parser.add_argument("--selftest", action="store_true", help="run tiny mocked offline controls")
    parser.add_argument("--snapshot", help="pinned metadata snapshot; must be in the default HF cache")
    parser.add_argument("--audit-out", help="write-once audit path (default: pinned snapshot)")
    args = parser.parse_args(argv)
    if sum((args.preflight, args.download, args.selftest)) != 1:
        parser.error("select exactly one of --preflight, --download, or --selftest")
    if args.selftest:
        selftest()
        print("STRAT-02 acquisition selftest: PASS (mocked, offline, no donor weights)")
        return 0
    try:
        snapshot = Path(args.snapshot).expanduser() if args.snapshot else None
        preflight = _stage_preflight(snapshot)
        if args.preflight:
            print(json.dumps(_public_preflight(preflight), indent=2, sort_keys=True))
            return 0
        audit_path = Path(args.audit_out).expanduser() if args.audit_out else preflight.snapshot / AUDIT_BASENAME
        if not audit_path.parent.is_dir():
            raise AcquisitionError(f"audit parent directory does not exist: {audit_path.parent}")
        if audit_path.exists():
            provisional = _audit_payload(
                preflight,
                [{"name": entry["name"], "status": "cached"} for entry in preflight.manifest["shards"]],
            )
            _validate_existing_audit(_read_json(audit_path, "existing acquisition audit"), provisional)
        outcomes = _download_shards(preflight)
        audit_status = _write_or_validate_audit(audit_path, _audit_payload(preflight, outcomes))
        print(json.dumps({
            "repository": REPOSITORY,
            "revision": REVISION,
            "shard_count": len(outcomes),
            "final_valid_payload_cap_bytes": preflight.byte_cap,
            "on_wire_bytes": "unknown",
            "audit": {"path": str(audit_path), "status": audit_status},
            "shards": outcomes,
        }, indent=2, sort_keys=True))
        return 0
    except AcquisitionError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
