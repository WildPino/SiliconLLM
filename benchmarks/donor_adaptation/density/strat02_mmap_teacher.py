"""Offline, mmap-backed STRAT-02 F32 teacher loader.

This module deliberately never calls ``from_pretrained`` or any Hub download API.
Weights remain safetensors CPU mappings for the lifetime of the context manager:

    report = preflight()
    verified = verify_shards(report)       # hashes local files; no download
    with load_reference_model(verified) as teacher:
        logits = teacher(input_ids).logits

The small self-test verifies the mechanics of ``assign=True``.  It does *not*
establish the peak-RAM behaviour of the 54 GB teacher; that remains unverified
until a target-environment run with the real local shards.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence


REPOSITORY = "allenai/StdMoE_1b14b_1T_Preanneal"
REVISION = "d2a4949c9d4ad6cf47fbac131f7e020077332b21"
TRANSFORMERS_VERSION = "4.57.1"
EXPECTED_SOURCE_SHA256 = {
    "config.json": "508fb6d7ae0419a27af8e9d776ba546152e89d259a80c35a9c5e3f8cfe8ee66e",
    "configuration_emo.py": "f1bd419d8dd926cf7d15131b8192938a0feda1b003586320fa56de20360e4bf2",
    "modeling_emo.py": "26f57354940655db673b35d47e9c6b1a85900068f41d91cb08afed4ec53219d6",
}
MANIFEST_PATH = Path(__file__).with_name("strat02_weight_sources.json")
EXPECTED_MANIFEST_SHA256 = "7aab6424b2390d5b1956793df58aa6be56a279e2d7ec01568d79a29a525a179e"
_CHUNK_BYTES = 8 * 1024 * 1024


class IntegrityError(RuntimeError):
    """Raised when a local source cannot be proved to be the pinned source."""


@dataclass(frozen=True)
class PreflightReport:
    """Validated metadata; creating this object never opens a weight shard."""

    snapshot: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    index: Mapping[str, Any]
    source_status: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedSnapshot:
    """Capability token produced only after every listed shard was hashed."""

    report: PreflightReport
    shard_stats: Mapping[str, tuple[int, int]]  # size, mtime_ns at verification


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} must be a JSON object: {path}")
    return value


def _cache_roots() -> list[Path]:
    """Return possible local Hub cache roots without importing huggingface_hub."""
    roots: list[Path] = []
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]).expanduser())
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    # HF's cross-platform default and the common Windows cache used by this repo.
    roots.extend((Path.home() / ".cache" / "huggingface" / "hub",
                  Path(os.environ.get("LOCALAPPDATA", Path.home())) / "huggingface" / "hub"))
    return list(dict.fromkeys(roots))


def find_local_snapshot(snapshot: str | os.PathLike[str] | None = None) -> Path:
    """Locate exactly the pinned snapshot locally; this function never contacts HF."""
    if snapshot is not None:
        candidate = Path(snapshot).expanduser().resolve()
        if not candidate.is_dir():
            raise IntegrityError(f"snapshot directory does not exist: {candidate}")
        candidates = [candidate]
    else:
        encoded_repo = "models--" + REPOSITORY.replace("/", "--")
        candidates = [root / encoded_repo / "snapshots" / REVISION for root in _cache_roots()]
    present = [path for path in candidates if path.is_dir()]
    if not present:
        searched = "; ".join(str(path) for path in candidates)
        raise IntegrityError(f"pinned local snapshot is absent; searched: {searched}")
    if len(present) != 1:
        raise IntegrityError("ambiguous pinned local snapshots: " + "; ".join(map(str, present)))
    chosen = present[0]
    if chosen.name != REVISION:
        raise IntegrityError(f"snapshot is not the pinned revision {REVISION}: {chosen}")
    return chosen


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema", "repository", "revision", "index_file", "index_sha256",
        "tensor_count", "parameter_count", "tensor_payload_bytes",
        "total_shard_file_bytes", "shards",
    }
    missing = required.difference(manifest)
    if missing:
        raise IntegrityError(f"manifest lacks required fields: {sorted(missing)}")
    if manifest["schema"] != "strat02_weight_sources_v1":
        raise IntegrityError("unexpected weight-source manifest schema")
    if manifest["repository"] != REPOSITORY or manifest["revision"] != REVISION:
        raise IntegrityError("manifest donor repository/revision is not STRAT-02's pin")
    if not isinstance(manifest["shards"], list) or len(manifest["shards"]) != 11:
        raise IntegrityError("STRAT-02 manifest must describe exactly 11 shards")
    names: set[str] = set()
    total = 0
    for entry in manifest["shards"]:
        if not isinstance(entry, dict) or set(entry) != {"name", "size", "sha256"}:
            raise IntegrityError("each shard record must contain only name, size, sha256")
        name, size, digest = entry["name"], entry["size"], entry["sha256"]
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise IntegrityError(f"invalid or duplicate shard name: {name!r}")
        if not isinstance(size, int) or size <= 0 or not isinstance(digest, str) or len(digest) != 64:
            raise IntegrityError(f"invalid shard size or SHA-256 for {name!r}")
        int(digest, 16)
        names.add(name)
        total += size
    if total != manifest["total_shard_file_bytes"]:
        raise IntegrityError("manifest total_shard_file_bytes disagrees with shard records")
    if manifest["tensor_payload_bytes"] != manifest["parameter_count"] * 4:
        raise IntegrityError("STRAT-02 F32 payload bytes must equal parameter_count * 4")
    if manifest["tensor_count"] != 6259 or manifest["parameter_count"] != 13_568_641_024:
        raise IntegrityError("unexpected STRAT-02 tensor or parameter count")


def _code_hashes(snapshot: Path) -> Mapping[str, Mapping[str, str]]:
    hashes: dict[str, Mapping[str, str]] = {}
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        path = snapshot / name
        if not path.is_file():
            raise IntegrityError(f"pinned local model code/config is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise IntegrityError(f"pinned model source hash mismatch for {name}: {actual} != {expected}")
        hashes[name] = {"sha256": actual}
    return hashes


def _require_sha256(path: Path, expected: str, label: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise IntegrityError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _ram_bytes() -> int | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                       ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                       ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                       ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                       ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        return int(status.ullAvailPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def preflight(snapshot: str | os.PathLike[str] | None = None,
              manifest_path: str | os.PathLike[str] = MANIFEST_PATH) -> PreflightReport:
    """Check manifest, local index and local code, but never open or hash a shard."""
    manifest_file = Path(manifest_path).resolve()
    _require_sha256(manifest_file, EXPECTED_MANIFEST_SHA256, "weight-source manifest")
    manifest = _read_json(manifest_file, "weight-source manifest")
    _validate_manifest(manifest)
    local_snapshot = find_local_snapshot(snapshot)
    index_path = local_snapshot / manifest["index_file"]
    if not index_path.is_file():
        raise IntegrityError(f"pinned snapshot index is missing: {index_path}")
    _require_sha256(index_path, manifest["index_sha256"], "index")
    index_hash = manifest["index_sha256"]
    index = _read_json(index_path, "safetensors index")
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise IntegrityError("index must contain object-valued weight_map and metadata")
    if len(weight_map) != manifest["tensor_count"] or len(set(weight_map)) != len(weight_map):
        raise IntegrityError("index tensor count/key set does not match manifest")
    if metadata.get("total_parameters") != manifest["parameter_count"]:
        raise IntegrityError("index total_parameters disagrees with manifest")
    if metadata.get("total_size") != manifest["tensor_payload_bytes"]:
        raise IntegrityError("index total_size disagrees with manifest's F32 payload")
    shard_names = {entry["name"] for entry in manifest["shards"]}
    if set(weight_map.values()) != shard_names or not all(isinstance(k, str) for k in weight_map):
        raise IntegrityError("index does not map exactly onto the manifest's 11 shards")
    try:
        installed_transformers = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        installed_transformers = None
    disk = shutil.disk_usage(local_snapshot)
    missing = [entry["name"] for entry in manifest["shards"] if not (local_snapshot / entry["name"]).is_file()]
    status: dict[str, Any] = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "snapshot": str(local_snapshot),
        "manifest_sha256": _sha256_file(manifest_file),
        "index_sha256": index_hash,
        "code_sha256": _code_hashes(local_snapshot),
        "transformers_installed": installed_transformers,
        "transformers_required": TRANSFORMERS_VERSION,
        "transformers_ok": installed_transformers == TRANSFORMERS_VERSION,
        "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
        "shards_present": len(missing) == 0,
        "missing_shards": missing,
        "required_shard_bytes": manifest["total_shard_file_bytes"],
        "disk_free_bytes": disk.free,
        "ram_available_bytes": _ram_bytes(),
        "note": "Preflight did not open, hash, download, or load any weight shard.",
    }
    return PreflightReport(local_snapshot, manifest_file, manifest, index, status)


def verify_shards(report: PreflightReport) -> VerifiedSnapshot:
    """Explicitly hash every expected local shard. Missing shards are never fetched."""
    if not isinstance(report, PreflightReport):
        raise TypeError("verify_shards requires the result of preflight()")
    missing = [entry["name"] for entry in report.manifest["shards"]
               if not (report.snapshot / entry["name"]).is_file()]
    if missing:
        raise IntegrityError("cannot verify: local shards missing (no download attempted): " + ", ".join(missing))
    stats: dict[str, tuple[int, int]] = {}
    for entry in report.manifest["shards"]:
        path = report.snapshot / entry["name"]
        stat = path.stat()
        if stat.st_size != entry["size"]:
            raise IntegrityError(f"shard size mismatch for {path.name}: {stat.st_size} != {entry['size']}")
        actual = _sha256_file(path)
        if actual != entry["sha256"]:
            raise IntegrityError(f"shard SHA-256 mismatch for {path.name}: {actual} != {entry['sha256']}")
        stats[path.name] = (stat.st_size, stat.st_mtime_ns)
    return VerifiedSnapshot(report, stats)


def _require_target_environment() -> None:
    try:
        installed = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise IntegrityError(f"transformers=={TRANSFORMERS_VERSION} is required but not installed") from exc
    if installed != TRANSFORMERS_VERSION:
        raise IntegrityError(f"transformers=={TRANSFORMERS_VERSION} is required; found {installed}")
    # This must be set before the pinned modeling module is imported: its decorator
    # otherwise may resolve Hub kernels. Do not silently override an explicit unsafe value.
    current = os.environ.get("USE_HUB_KERNELS")
    if current is None:
        os.environ["USE_HUB_KERNELS"] = "NO"
    elif current != "NO":
        raise IntegrityError(f"USE_HUB_KERNELS must be NO before model import; found {current!r}")


def _assert_unchanged(verified: VerifiedSnapshot) -> None:
    if not isinstance(verified, VerifiedSnapshot):
        raise TypeError("load_reference_model requires VerifiedSnapshot from verify_shards(); it will not hash/download")
    for name, expected in verified.shard_stats.items():
        path = verified.report.snapshot / name
        try:
            stat = path.stat()
        except OSError as exc:
            raise IntegrityError(f"verified shard disappeared before load: {path}") from exc
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise IntegrityError(f"verified shard changed after verification: {path.name}; verify again")


def _import_pinned_emo(snapshot: Path) -> tuple[Any, Any]:
    """Import only configuration_emo.py/modeling_emo.py from this local snapshot."""
    package_name = "_strat02_pinned_emo_" + hashlib.sha256(str(snapshot).encode()).hexdigest()[:16]
    package = types.ModuleType(package_name)
    package.__path__ = [str(snapshot)]  # relative import in modeling_emo.py
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        config_spec = importlib.util.spec_from_file_location(package_name + ".configuration_emo", snapshot / "configuration_emo.py")
        model_spec = importlib.util.spec_from_file_location(package_name + ".modeling_emo", snapshot / "modeling_emo.py")
        if config_spec is None or config_spec.loader is None or model_spec is None or model_spec.loader is None:
            raise IntegrityError("cannot create local import specs for pinned Emo code")
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules[config_spec.name] = config_module
        config_spec.loader.exec_module(config_module)
        model_module = importlib.util.module_from_spec(model_spec)
        sys.modules[model_spec.name] = model_module
        model_spec.loader.exec_module(model_module)
        return config_module.EmoConfig, model_module.EmoForCausalLM
    except Exception:
        for name in (package_name + ".modeling_emo", package_name + ".configuration_emo", package_name):
            sys.modules.pop(name, None)
        raise


def _open_tensor_state(verified: VerifiedSnapshot, stack: contextlib.ExitStack) -> dict[str, Any]:
    from safetensors import safe_open
    import torch

    report = verified.report
    handles: dict[str, Any] = {}
    for entry in report.manifest["shards"]:
        path = report.snapshot / entry["name"]
        handles[entry["name"]] = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
    index_map = report.index["weight_map"]
    _assert_index_key_mapping(index_map, handles)
    state: dict[str, Any] = {}
    count = 0
    for key, shard_name in index_map.items():
        tensor = handles[shard_name].get_tensor(key)
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise IntegrityError(f"{key} is not a CPU float32 tensor")
        count += tensor.numel()
        state[key] = tensor
    if count != report.manifest["parameter_count"]:
        raise IntegrityError(f"F32 element count mismatch: {count} != {report.manifest['parameter_count']}")
    return state


def _assert_index_key_mapping(index_map: Mapping[str, str], handles: Mapping[str, Any]) -> None:
    """Assert that each opened shard's header is exactly the index partition."""
    actual_keys: set[str] = set()
    for shard_name, handle in handles.items():
        keys = set(handle.keys())
        indexed = {key for key, mapped in index_map.items() if mapped == shard_name}
        if keys != indexed:
            raise IntegrityError(f"safetensors/index key mismatch in {shard_name}")
        actual_keys.update(keys)
    if actual_keys != set(index_map):
        raise IntegrityError("safetensors key union does not equal the index key set")


def _assert_pointer_sharing(model: Any, source: Mapping[str, Any], mode: Literal["sample", "all"]) -> None:
    keys = sorted(source)
    if mode == "sample" and len(keys) > 96:
        stride = max(1, len(keys) // 96)
        keys = sorted(set(keys[:16] + keys[-16:] + keys[::stride]))
    model_state = model.state_dict()
    for key in keys:
        expected, actual = source[key].data_ptr(), model_state[key].data_ptr()
        if actual != expected:
            raise IntegrityError(f"assign=True pointer/copy mismatch for {key}: {actual} != {expected}")


def _restore_nonpersistent_rope_buffer(model: Any) -> None:
    """Construct the sole non-state RoPE buffer on CPU after meta instantiation.

    ``inv_freq`` is registered with ``persistent=False`` in the pinned Emo
    source, so ``load_state_dict(assign=True)`` cannot replace its meta tensor.
    Recreating only the weightless rotary module with the same pinned class
    and config preserves all mmap-backed parameter storage.
    """
    import torch

    meta_buffers = [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if meta_buffers != ["model.rotary_emb.inv_freq"]:
        raise IntegrityError(f"unexpected meta buffers after weight assignment: {meta_buffers}")
    rotary = model.model.rotary_emb
    if rotary.state_dict():
        raise IntegrityError("rotary module unexpectedly has persistent state")
    replacement = type(rotary)(config=model.config, device=torch.device("cpu"))
    if replacement.state_dict() or replacement.inv_freq.is_meta:
        raise IntegrityError("CPU rotary reconstruction did not produce only a real nonpersistent buffer")
    model.model.rotary_emb = replacement
    remaining = [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if remaining:
        raise IntegrityError(f"meta buffers remain after RoPE restoration: {remaining}")


@contextlib.contextmanager
def load_reference_model(verified: VerifiedSnapshot,
                         pointer_check: Literal["sample", "all"] = "sample") -> Iterator[Any]:
    """Yield a meta-instantiated, mmap-backed EmoForCausalLM after explicit verification.

    The returned model is valid only inside this ``with`` block.  Its safetensors
    mappings are intentionally held open by the ExitStack for the entire lifetime.
    """
    if pointer_check not in {"sample", "all"}:
        raise ValueError("pointer_check must be 'sample' or 'all'")
    _assert_unchanged(verified)
    _require_sha256(verified.report.manifest_path, EXPECTED_MANIFEST_SHA256, "weight-source manifest")
    _require_sha256(verified.report.snapshot / verified.report.manifest["index_file"],
                    verified.report.manifest["index_sha256"], "index")
    _code_hashes(verified.report.snapshot)
    _require_target_environment()
    import torch

    with contextlib.ExitStack() as stack:
        source = _open_tensor_state(verified, stack)
        EmoConfig, EmoForCausalLM = _import_pinned_emo(verified.report.snapshot)
        config_dict = _read_json(verified.report.snapshot / "config.json", "pinned model config")
        with torch.device("meta"):
            model = EmoForCausalLM(EmoConfig(**config_dict))
        model_keys = set(model.state_dict())
        if model_keys != set(source):
            missing, unexpected = sorted(model_keys - set(source)), sorted(set(source) - model_keys)
            raise IntegrityError(f"model/index keyset mismatch; missing={missing[:5]} unexpected={unexpected[:5]}")
        result = model.load_state_dict(source, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise IntegrityError(f"strict state load failed: {result}")
        meta = [name for name, tensor in model.state_dict().items() if tensor.is_meta]
        if meta:
            raise IntegrityError(f"meta tensors remain after assign=True load: {meta[:5]}")
        _restore_nonpersistent_rope_buffer(model)
        _assert_pointer_sharing(model, source, pointer_check)
        yield model


def _expect_error(function: Any, *args: Any) -> None:
    try:
        function(*args)
    except IntegrityError:
        return
    raise AssertionError("planted integrity control did not fail")


def selftest() -> None:
    """Run only tiny temporary-file controls; no donor weights are touched or fetched."""
    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory(prefix="strat02-mmap-selftest-") as temporary:
        root = Path(temporary)
        snapshot = root / "models--allenai--StdMoE_1b14b_1T_Preanneal" / "snapshots" / REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "configuration_emo.py").write_text("# tiny selftest fixture\n", encoding="utf-8")
        (snapshot / "modeling_emo.py").write_text("# tiny selftest fixture\n", encoding="utf-8")
        tensors = {"weight": torch.tensor([[1.0, -2.0], [3.0, 4.0]]), "bias": torch.tensor([0.5, -1.5])}
        first = "model-00001-of-00011.safetensors"
        save_file(tensors, snapshot / first)
        for number in range(2, 12):
            shutil.copyfile(snapshot / first, snapshot / f"model-{number:05d}-of-00011.safetensors")
        index = {"metadata": {"total_parameters": 6, "total_size": 24},
                 "weight_map": {"weight": first, "bias": first}}
        index_path = snapshot / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        shards = []
        for number in range(1, 12):
            name = f"model-{number:05d}-of-00011.safetensors"
            path = snapshot / name
            shards.append({"name": name, "size": path.stat().st_size, "sha256": _sha256_file(path)})
        manifest = {"schema": "strat02_weight_sources_v1", "repository": REPOSITORY, "revision": REVISION,
                    "index_file": index_path.name, "index_sha256": _sha256_file(index_path), "tensor_count": 6259,
                    "parameter_count": 13_568_641_024, "tensor_payload_bytes": 54_274_564_096,
                    "total_shard_file_bytes": sum(item["size"] for item in shards), "shards": shards}
        # The real-manifest cardinality invariant is intentional.  Exercise the mechanics below with a
        # direct tiny source map, then exercise bad manifest/index controls without invoking preflight.
        from safetensors import safe_open
        with safe_open(str(snapshot / first), framework="pt", device="cpu") as mapped_file:
            mapped = {key: mapped_file.get_tensor(key) for key in ("weight", "bias")}
            with torch.device("meta"):
                linear = torch.nn.Linear(2, 2)
            result = linear.load_state_dict(mapped, strict=True, assign=True)
            assert not result.missing_keys and not result.unexpected_keys
            _assert_pointer_sharing(linear, mapped, "all")
            x = torch.tensor([[2.0, -1.0]])
            assert torch.equal(linear(x), torch.nn.functional.linear(x, tensors["weight"], tensors["bias"]))
        bad_manifest = dict(manifest)
        bad_manifest["tensor_count"] = 2
        _expect_error(_validate_manifest, bad_manifest)
        _expect_error(_code_hashes, snapshot)
        bad_index = dict(index)
        bad_index["weight_map"] = {"missing": first, "bias": first}
        with safe_open(str(snapshot / first), framework="pt", device="cpu") as handle:
            _expect_error(_assert_index_key_mapping, bad_index["weight_map"], {first: handle})
        bad_hash_manifest = dict(manifest)
        bad_hash_manifest["index_sha256"] = "0" * 64
        _expect_error(_require_sha256, index_path, bad_hash_manifest["index_sha256"], "planted index")


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="offline STRAT-02 mmap F32 teacher source checks")
    parser.add_argument("--snapshot", help="explicit local pinned HF snapshot directory")
    parser.add_argument("--preflight", action="store_true", help="validate local source metadata; never touch shard payloads")
    parser.add_argument("--verify-shards", action="store_true", help="explicitly SHA-256 all 11 local shards; never downloads")
    parser.add_argument("--selftest", action="store_true", help="run tiny temporary safetensors/assign controls")
    args = parser.parse_args(argv)
    if sum((args.preflight, args.verify_shards, args.selftest)) != 1:
        parser.error("select exactly one of --preflight, --verify-shards, or --selftest")
    if args.selftest:
        selftest()
        print("STRAT-02 mmap teacher selftest: PASS (tiny fixture only; 54 GB peak RAM unverified)")
        return 0
    try:
        report = preflight(args.snapshot)
        if args.preflight:
            print(json.dumps(report.source_status, indent=2, sort_keys=True))
            return 0
        verified = verify_shards(report)
        print(json.dumps({"verified": True, "snapshot": str(verified.report.snapshot),
                          "shards": len(verified.shard_stats), "downloaded": False}, indent=2))
        return 0
    except IntegrityError as exc:
        print(f"STRAT-02 source check: FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
