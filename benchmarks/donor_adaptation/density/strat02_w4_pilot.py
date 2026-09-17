#!/usr/bin/env python3
"""Bounded STRAT-02 W4 real-weight pilot apparatus.

This is deliberately not a model conversion, quality measurement, inference
benchmark, donor forward, or a T4 job.  ``--run`` hashes the already-local,
pinned donor shards once, samples at most the first 256 complete rows from the
four frozen linear matrices, and writes four W4 g128 blobs.  It never opens a
corpus or tokenizer and it cannot download or resume.

The only modes are::

    python strat02_w4_pilot.py --selftest
    python strat02_w4_pilot.py --preflight [--snapshot PATH]
    python strat02_w4_pilot.py --run --output-dir NEW_EMPTY_PATH [--snapshot PATH]

The parent supervises exactly the direct base-Python child it creates.  A
resource breach produces ``VOID_RESOURCE``; codec/inventory/provenance failure
produces ``VOID_APPARATUS``; all other failures remain ``INCOMPLETE``.  Partial
blobs are deliberately retained and are never resumed automatically.
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
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
WORKER_TOKEN_ENV = "STRAT02_W4_PILOT_WORKER_TOKEN"
PILOT_SCHEMA = "strat02_w4_real_weight_pilot_v1"
MIN_LAUNCH_RAM_BYTES = 8 * 1024 ** 3
MIN_LAUNCH_OUTPUT_FREE_BYTES = 2 * 1024 ** 3
MIN_RUNTIME_RAM_BYTES = 4 * 1024 ** 3
MAX_PRIVATE_COMMIT_BYTES = 8 * 1024 ** 3
MAX_WALL_SECONDS = 30 * 60
MONITOR_INTERVAL_SECONDS = 5.0
MAX_ROWS = 256
TILE_ROWS = 16
TILE_GROUPS = 16
EXPECTED_LINEAR_COUNT = 6225
EXPECTED_LINEAR_WEIGHTS = 13_363_052_544
EXPECTED_CATEGORY_COUNTS = {"attention": 64, "router": 16, "expert": 6144, "lm_head": 1}
FROZEN_KEYS = {
    "attention": "model.layers.0.self_attn.k_proj.weight",
    "router": "model.layers.0.mlp.gate.weight",
    "expert": "model.layers.0.mlp.experts.0.down_proj.weight",
    "lm_head": "lm_head.weight",
}
BRIEF_PATH = HERE.parents[2] / "docs" / "research" / "donor_adaptation" / "briefs" / "BRIEF_STRAT_02_W4_CONVERSION_APPARATUS.md"
SOURCE_PATHS = {
    "pilot": Path(__file__).resolve(),
    "codec": HERE / "strat02_weight_codec.py",
    "teacher_loader": HERE / "strat02_mmap_teacher.py",
    "bounded_supervisor": HERE / "strat02_bounded_smoke.py",
    "brief": BRIEF_PATH,
}


class GateError(RuntimeError):
    """A frozen provenance, inventory, resource, or codec invariant failed."""


@dataclass(frozen=True)
class MonitorOutcome:
    status: str
    exit_code: int | None
    reason: str | None
    elapsed_seconds: float
    samples: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _load_local_module(filename: str, module_name: str) -> Any:
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot load local module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise GateError(f"write-once output already exists: {path}") from exc


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    record = {"utc_timestamp": _utc_now(), **event}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _prepare_output(output_dir: Path) -> dict[str, Path]:
    """Create a genuinely fresh result directory and its write-once control files."""
    resolved = output_dir.resolve()
    if resolved.exists():
        raise GateError(f"--output-dir must be a fresh path, not an existing directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    paths = {
        "root": resolved,
        "manifest": resolved / "pilot_manifest.json",
        "log": resolved / "supervisor_log.jsonl",
        "worker": resolved / "worker_result.json",
        "result": resolved / "supervisor_result.json",
        "worker_stdout": resolved / "worker_stdout.log",
        "worker_stderr": resolved / "worker_stderr.log",
    }
    with paths["log"].open("x", encoding="utf-8", newline="\n"):
        pass
    return paths


def _psutil() -> Any:
    try:
        import psutil
    except ImportError as exc:
        raise GateError("psutil is required for Windows resource monitoring") from exc
    return psutil


def _inventory_linear_weights(teacher: Any, report: Any) -> dict[str, Any]:
    """Instantiate the exactly-pinned remote code on meta and inventory linears.

    This does not call ``safe_open``, touch a shard payload, load a state dict,
    or run a model forward.  Categories are derived from real parent module
    relationships, rather than looking for words such as ``router`` in keys.
    """
    teacher._require_target_environment()
    import torch

    EmoConfig, EmoForCausalLM = teacher._import_pinned_emo(report.snapshot)
    config_dict = teacher._read_json(report.snapshot / "config.json", "pinned model config")
    with torch.device("meta"):
        model = EmoForCausalLM(EmoConfig(**config_dict))
    index_keys = set(report.index["weight_map"])
    model_state = model.state_dict()
    model_keys = set(model_state)
    if model_keys != index_keys:
        missing, unexpected = sorted(index_keys - model_keys), sorted(model_keys - index_keys)
        raise GateError(f"meta model/index keyset mismatch; missing={missing[:5]} unexpected={unexpected[:5]}")

    categories: dict[str, list[str]] = {"attention": [], "router": [], "expert": [], "lm_head": []}
    linear_keys: list[str] = []
    linear_weight_total = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not module_name or module.weight is None:
            raise GateError(f"invalid nn.Linear inventory entry: {module_name!r}")
        key = module_name + ".weight"
        if key not in index_keys or tuple(module.weight.shape) != tuple(model_state[key].shape):
            raise GateError(f"linear inventory/index mismatch for {key}")
        linear_keys.append(key)
        linear_weight_total += int(module.weight.numel())
        parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        parent = model.get_submodule(parent_name) if parent_name else model
        grandparent_name = parent_name.rsplit(".", 1)[0] if "." in parent_name else ""
        grandparent = model.get_submodule(grandparent_name) if grandparent_name else model
        if module_name == "lm_head" and getattr(model, "lm_head", None) is module:
            categories["lm_head"].append(key)
        elif (all(hasattr(parent, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
              and module in {parent.q_proj, parent.k_proj, parent.v_proj, parent.o_proj}):
            categories["attention"].append(key)
        elif (hasattr(parent, "gate") and hasattr(parent, "experts")
              and getattr(parent, "gate") is module):
            categories["router"].append(key)
        elif isinstance(grandparent, torch.nn.ModuleList) and any(
            candidate is parent for candidate in grandparent
        ):
            owner_name = grandparent_name.rsplit(".", 1)[0] if "." in grandparent_name else ""
            owner = model.get_submodule(owner_name) if owner_name else model
            if getattr(owner, "experts", None) is grandparent and hasattr(owner, "gate"):
                categories["expert"].append(key)

    if len(linear_keys) != EXPECTED_LINEAR_COUNT or linear_weight_total != EXPECTED_LINEAR_WEIGHTS:
        raise GateError(
            f"unexpected meta linear inventory: {len(linear_keys)} matrices/{linear_weight_total} weights "
            f"!= {EXPECTED_LINEAR_COUNT}/{EXPECTED_LINEAR_WEIGHTS}"
        )
    category_counts = {name: len(keys) for name, keys in categories.items()}
    if category_counts != EXPECTED_CATEGORY_COUNTS:
        raise GateError(f"linear organ taxonomy differs from pin: {category_counts}")
    selected = {organ: min(keys) if keys else None for organ, keys in categories.items()}
    if selected != FROZEN_KEYS:
        raise GateError(f"frozen pilot keys disagree with structural inventory: {selected}")
    if len(set(selected.values())) != len(selected) or any(key not in index_keys for key in selected.values()):
        raise GateError("selected pilot linear keys are not distinct index keys")
    return {
        "linear_count": len(linear_keys),
        "linear_weight_count": linear_weight_total,
        "selected": selected,
        "category_counts": category_counts,
        "model_key_count": len(model_keys),
        "index_key_count": len(index_keys),
    }


def _preflight(snapshot: Path | None, *, output_dir: Path | None = None) -> tuple[Any, Any, dict[str, Any]]:
    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_w4_pilot_bounded")
    teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_w4_pilot_teacher")
    versions = bounded._pinned_environment()
    report = teacher.preflight(snapshot=snapshot)
    status = report.source_status
    if not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO":
        raise GateError("pinned local model code/runtime proof failed")
    if not status.get("shards_present"):
        raise GateError("all 11 pinned local shards must be present; no download is attempted")
    inventory = _inventory_linear_weights(teacher, report)
    psutil = _psutil()
    available_ram = int(psutil.virtual_memory().available)
    if available_ram < MIN_LAUNCH_RAM_BYTES:
        raise GateError(f"launch requires >=8 GiB available physical RAM; observed {available_ram} bytes")
    volume = output_dir.resolve() if output_dir is not None else report.snapshot
    available_disk = int(psutil.disk_usage(str(volume)).free)
    if available_disk < MIN_LAUNCH_OUTPUT_FREE_BYTES:
        raise GateError(f"launch requires >=2 GiB free on output volume; observed {available_disk} bytes")
    return bounded, teacher, {
        "utc_timestamp": _utc_now(),
        "runtime_versions": versions,
        "python": sys.version.split()[0],
        "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
        "offline_policy": {
            "hub_download_api": "not used",
            "hf_hub_offline": True,
            "transformers_offline": True,
            "weight_access": "pinned local safetensors only in --run worker",
            "heldout_or_text_access": "not used",
            "donor_forward": "not used",
            "t4": "not used",
        },
        "snapshot": str(report.snapshot),
        "metadata": dict(status),
        "available_physical_ram_bytes": available_ram,
        "output_volume_free_bytes": available_disk,
        "inventory": inventory,
        "apparatus_sha256": {name: _sha256_file(path) for name, path in SOURCE_PATHS.items()},
    }


def _write_tile_blob(
    path: Path, source: np.ndarray, codec: Any, *, max_rows: int = TILE_ROWS,
    max_groups: int = TILE_GROUPS,
) -> None:
    """Write one sampled matrix in frozen scale-then-code layout without an output array."""
    if source.dtype != np.float32 or source.ndim != 2:
        raise GateError("sampled source must be a two-dimensional float32 array")
    rows, in_features = source.shape
    groups = in_features // codec.GROUP_SIZE
    expected_size = codec.w4_byte_count(rows, in_features)
    scale_bytes = rows * groups * np.dtype(np.float16).itemsize
    with path.open("xb") as handle:
        handle.truncate(expected_size)
        for tile in codec.iter_encode_w4_tiles(
            (source,), in_features, max_rows_per_tile=max_rows, max_groups_per_tile=max_groups
        ):
            tile_rows, tile_groups = tile.scales.shape
            for local_row in range(tile_rows):
                row = tile.row_offset + local_row
                scale_offset = (row * groups + tile.group_offset) * np.dtype(np.float16).itemsize
                code_offset = scale_bytes + (row * groups + tile.group_offset) * 64
                handle.seek(scale_offset)
                handle.write(tile.scales[local_row].tobytes(order="C"))
                handle.seek(code_offset)
                handle.write(tile.packed_codes[local_row].tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    if path.stat().st_size != expected_size:
        raise GateError(f"tiled W4 blob size mismatch for {path.name}")


def _rowwise_reference_blob(source: np.ndarray, codec: Any) -> bytes:
    rows, in_features = source.shape
    groups = in_features // codec.GROUP_SIZE
    scales = np.empty((rows, groups), dtype=np.float16)
    packed = np.empty((rows, groups * 64), dtype=np.uint8)
    for row, (scale_row, packed_row) in enumerate(codec.iter_encode_w4_rows((source[i] for i in range(rows)), in_features)):
        scales[row], packed[row] = scale_row, packed_row
    return codec.w4_to_blob(codec.W4Encoded((rows, in_features), scales, packed))


def _scalar_decode_from_blob(blob: bytes, shape: tuple[int, int], codec: Any) -> np.ndarray:
    encoded = codec.w4_from_blob(blob, shape)
    rows, in_features = shape
    groups = in_features // codec.GROUP_SIZE
    decoded = np.empty(shape, dtype=np.float32)
    for row in range(rows):
        for group in range(groups):
            codes = codec.unpack_w4_codes_scalar(encoded.packed_codes[row, group * 64:(group + 1) * 64])
            decoded[row, group * codec.GROUP_SIZE:(group + 1) * codec.GROUP_SIZE] = (
                codec.dequantize_w4_group_scalar(encoded.scales[row, group], codes)
            )
    return decoded


def _assert_packed_roundtrip(blob: bytes, source: np.ndarray, codec: Any) -> dict[str, Any]:
    """Validate only bytes we just produced; donor F32 is never used for output parity."""
    reference = _rowwise_reference_blob(source, codec)
    if blob != reference:
        raise GateError("tiled W4 bytes differ from row-wise reference")
    encoded = codec.w4_from_blob(blob, tuple(source.shape))
    vector = codec.dequantize_w4(encoded)
    scalar = _scalar_decode_from_blob(blob, tuple(source.shape), codec)
    if not np.array_equal(scalar, vector):
        raise GateError("scalar and vector W4 decoding from packed blob differ")
    import torch

    # Both sides below use only reconstructed weight values.  The original
    # sampled donor array is intentionally absent from this output-parity test.
    reconstructed = torch.from_numpy(vector.copy())
    synthetic = torch.linspace(-1.0, 1.0, steps=3 * source.shape[1], dtype=torch.float32).reshape(3, source.shape[1])
    linear = torch.nn.Linear(source.shape[1], source.shape[0], bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(reconstructed)
        observed = linear(synthetic)
        expected = torch.nn.functional.linear(synthetic, torch.from_numpy(scalar.copy()))
    if not torch.equal(observed, expected):
        raise GateError("synthetic nn.Linear output differs for reconstructed packed weights")
    return {
        "decoded_f32_exact": True,
        "synthetic_linear_parity": True,
        "reference_blob_sha256": hashlib.sha256(reference).hexdigest(),
    }


def _read_sampled_tensor(report: Any, key: str) -> np.ndarray:
    """Open exactly one local shard after prior full SHA verification, then take prefix rows."""
    from safetensors import safe_open
    import torch

    shard_name = report.index["weight_map"].get(key)
    if not isinstance(shard_name, str):
        raise GateError(f"selected key missing from safetensors index: {key}")
    with safe_open(str(report.snapshot / shard_name), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(key)
        shape = tuple(int(value) for value in tensor_slice.get_shape())
        if len(shape) != 2:
            raise GateError(f"selected linear tensor is not a matrix: {key}")
        rows = min(MAX_ROWS, shape[0])
        if rows <= 0 or shape[1] % 128:
            raise GateError(f"selected matrix shape violates frozen W4 pilot constraints: {key} {shape}")
        tensor = tensor_slice[:rows]
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tensor.ndim != 2:
            raise GateError(f"selected linear tensor is not a CPU F32 matrix: {key}")
        if tuple(tensor.shape) != (rows, shape[1]):
            raise GateError(f"selected tensor slice/header shape mismatch: {key}")
        view = tensor.numpy()
        if view.dtype != np.float32 or view.shape != (rows, shape[1]):
            raise GateError(f"failed to obtain sampled F32 rows for {key}")
        return view.copy()


def _worker(args: argparse.Namespace) -> int:
    result_path = args.result.resolve()
    completed: list[dict[str, Any]] = []
    try:
        parent_manifest = json.loads((args.output_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
        expected_hashes = parent_manifest.get("parent_preflight", {}).get("apparatus_sha256")
        actual_hashes = {name: _sha256_file(path) for name, path in SOURCE_PATHS.items()}
        if expected_hashes != actual_hashes:
            raise GateError("pilot/codec/loader/brief changed after parent preflight")
        teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_w4_pilot_worker_teacher")
        codec = _load_local_module("strat02_weight_codec.py", "_strat02_w4_pilot_worker_codec")
        report = teacher.preflight(snapshot=args.snapshot)
        inventory = _inventory_linear_weights(teacher, report)
        if inventory["selected"] != FROZEN_KEYS:
            raise GateError("worker structural inventory changed from frozen pilot selection")
        # This is the only full shard pass, and it happens before any selected tensor opens.
        verified = teacher.verify_shards(report)
        if not isinstance(verified, teacher.VerifiedSnapshot):
            raise GateError("pinned shard verification did not produce a capability token")
        for organ in ("attention", "router", "expert", "lm_head"):
            key = FROZEN_KEYS[organ]
            # Record provenance before reading the tensor values into a sample array.
            planned = {"organ": organ, "key": key, "selected_before_tensor_read": True}
            _append_log(args.log, {"event": "matrix_selected", **planned})
            shard_name = report.index["weight_map"][key]
            # Stat-check all verified shards immediately before each selected
            # read; this is not a second hash pass and fails closed on mutation.
            teacher._assert_unchanged(verified)
            # Header-only shape lookup is intentionally avoided: the selected tensor is read once.
            source = _read_sampled_tensor(report, key)
            rows, in_features = source.shape
            blob_path = args.output_dir / f"{organ}.w4g128.bin"
            _write_tile_blob(blob_path, source, codec)
            blob = blob_path.read_bytes()
            validation = _assert_packed_roundtrip(blob, source, codec)
            record = {
                **planned,
                "source_shard": shard_name,
                "shape": [rows, in_features],
                "source_dtype": "float32",
                "format": "STRAT-02-W4-g128-scale-then-codes",
                "blob_file": blob_path.name,
                "blob_bytes": len(blob),
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                **validation,
            }
            completed.append(record)
            _append_log(args.log, {"event": "matrix_complete", "organ": organ, "blob_bytes": len(blob)})
        _write_json_once(result_path, {
            "ok": True,
            "status": "COMPLETE",
            "utc_timestamp": _utc_now(),
            "matrices": completed,
            "notes": [
                "Bounded W4 apparatus only; no full conversion, donor forward, heldout, text, download, or T4 use.",
                "F32 decoding and nn.Linear parity were derived from each packed blob, never from original weights.",
            ],
        })
        return 0
    except Exception as exc:
        status = "VOID_APPARATUS" if isinstance(exc, GateError) else "INCOMPLETE"
        try:
            _write_json_once(result_path, {
                "ok": False,
                "status": status,
                "utc_timestamp": _utc_now(),
                "completed_matrices": completed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        except (GateError, OSError):
            pass
        return 1


def _process_sample(process: Any, psutil: Any) -> dict[str, int | None]:
    try:
        basic = process.memory_info()
        full = process.memory_full_info()
    except Exception as exc:
        raise GateError(f"cannot read worker resource sample: {type(exc).__name__}: {exc}") from exc
    private = getattr(full, "private", None)
    if private is None:
        raise GateError("psutil does not expose Windows private commit for the worker")
    return {"child_working_set_bytes": int(basic.rss), "child_private_commit_bytes": int(private)}


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


def _monitor_child(process: Any, *, psutil: Any, on_sample: Callable[[Mapping[str, Any]], None],
                   expected_executable: Path, now: Callable[[], float] = time.monotonic,
                   interval_seconds: float = MONITOR_INTERVAL_SECONDS,
                   wall_limit_seconds: float = MAX_WALL_SECONDS) -> MonitorOutcome:
    started, samples, checked = now(), 0, False
    resource_process: Any | None = None
    while True:
        elapsed = now() - started
        exit_code = process.poll()
        if exit_code is not None:
            return MonitorOutcome("CHILD_EXITED", int(exit_code), None, elapsed, samples)
        if elapsed >= wall_limit_seconds:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "wall_clock_exceeded", elapsed, samples)
        try:
            if resource_process is None:
                resource_process = psutil.Process(process.pid)
            if not checked:
                if Path(resource_process.exe()).resolve() != expected_executable.resolve():
                    return MonitorOutcome("VOID_APPARATUS", _terminate_only_child(process), "monitored_executable_mismatch", elapsed, samples)
                checked = True
            sample = _process_sample(resource_process, psutil)
            available = int(psutil.virtual_memory().available)
        except Exception as exc:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), str(exc), elapsed, samples)
        sample = {"elapsed_seconds": elapsed, "available_physical_ram_bytes": available, **sample}
        on_sample(sample)
        samples += 1
        if available < MIN_RUNTIME_RAM_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "available_physical_ram_below_4_gib", elapsed, samples)
        if sample["child_private_commit_bytes"] > MAX_PRIVATE_COMMIT_BYTES:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "child_private_commit_above_8_gib", elapsed, samples)
        try:
            process.wait(timeout=min(interval_seconds, max(0.0, wall_limit_seconds - elapsed)))
        except subprocess.TimeoutExpired:
            pass


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_parent(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    try:
        bounded, _teacher, preflight = _preflight(args.snapshot, output_dir=paths["root"])
        _write_json_once(paths["manifest"], {
            "schema": PILOT_SCHEMA,
            "purpose": "bounded real-weight W4 g128 apparatus; not a full conversion or quality/rate run",
            "selected_before_tensor_read": FROZEN_KEYS,
            "limits": {
                "launch_available_physical_ram_bytes_min": MIN_LAUNCH_RAM_BYTES,
                "output_volume_free_bytes_min": MIN_LAUNCH_OUTPUT_FREE_BYTES,
                "runtime_available_physical_ram_bytes_min": MIN_RUNTIME_RAM_BYTES,
                "worker_private_commit_bytes_max": MAX_PRIVATE_COMMIT_BYTES,
                "wall_seconds_max": MAX_WALL_SECONDS,
                "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
                "max_rows_per_matrix": MAX_ROWS,
            },
            "parent_preflight": preflight,
            "output": {"auto_resume": False, "heldout_or_text": False, "full_conversion": False,
                       "worker_stdout": paths["worker_stdout"].name,
                       "worker_stderr": paths["worker_stderr"].name},
        })
        _append_log(paths["log"], {"event": "parent_preflight_passed"})
    except Exception as exc:
        _write_json_once(paths["manifest"], {"schema": PILOT_SCHEMA, "preflight": "failed"})
        _append_log(paths["log"], {"event": "parent_preflight_failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json_once(paths["result"], {"ok": False, "status": "INCOMPLETE", "failure_classification": "PREFLIGHT_ERROR", "error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps({"ok": False, "status": "INCOMPLETE", "result": str(paths["result"])}, sort_keys=True))
        return 1

    process: Any | None = None
    try:
        child_env = os.environ.copy()
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        worker_python = bounded._direct_worker_python(child_env)
        token = secrets.token_urlsafe(32)
        child_env[WORKER_TOKEN_ENV] = token
        command = [worker_python, str(Path(__file__).resolve()), "--worker", "--snapshot", preflight["snapshot"],
                   "--output-dir", str(paths["root"]), "--log", str(paths["log"]), "--result", str(paths["worker"]),
                   "--worker-token", token]
        _append_log(paths["log"], {"event": "child_launch", "python": worker_python, "termination_scope": "direct_worker_pid_only"})
        with paths["worker_stdout"].open("x", encoding="utf-8") as stdout, \
             paths["worker_stderr"].open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=str(HERE), env=child_env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr)
            outcome = _monitor_child(process, psutil=_psutil(), expected_executable=Path(worker_python),
                                     on_sample=lambda sample: _append_log(paths["log"], {"event": "resource_sample", **sample}))
        _append_log(paths["log"], {"event": "monitor_finished", **asdict(outcome)})
        worker = _load_json(paths["worker"])
        if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
            classification = outcome.status
        elif worker and worker.get("status") == "VOID_APPARATUS":
            classification = "VOID_APPARATUS"
        elif outcome.exit_code != 0:
            classification = "WORKER_ERROR"
        elif not worker or worker.get("ok") is not True or worker.get("status") != "COMPLETE":
            classification = "WORKER_PROTOCOL_ERROR"
        else:
            classification = None
        final = {"ok": classification is None, "status": "COMPLETE" if classification is None else "INCOMPLETE",
                 "failure_classification": classification, "utc_timestamp": _utc_now(), "child_pid": process.pid,
                 "monitor": asdict(outcome), "worker_result": worker}
    except Exception as exc:
        if process is not None:
            _terminate_only_child(process)
        _append_log(paths["log"], {"event": "parent_supervision_error", "error_type": type(exc).__name__, "error": str(exc)})
        final = {"ok": False, "status": "INCOMPLETE", "failure_classification": "PARENT_SUPERVISION_ERROR",
                 "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": final["ok"], "status": final["status"], "result": str(paths["result"])}, sort_keys=True))
    return 0 if final["ok"] else 1


def _selftest() -> None:
    """Synthetic only: exact bytes, decode-from-packed, output parity, and frozen key controls."""
    codec = _load_local_module("strat02_weight_codec.py", "_strat02_w4_pilot_selftest_codec")
    rng = np.random.default_rng(20260917)
    source = rng.normal(size=(7, 384)).astype(np.float32)
    source[0, :128] = 0
    with tempfile.TemporaryDirectory(prefix="strat02-w4-pilot-selftest-") as temporary:
        blob_path = Path(temporary) / "synthetic.w4g128.bin"
        _write_tile_blob(blob_path, source, codec, max_rows=2, max_groups=2)
        blob = blob_path.read_bytes()
        result = _assert_packed_roundtrip(blob, source, codec)
        assert result["decoded_f32_exact"] and result["synthetic_linear_parity"]
        assert len(blob) == codec.w4_byte_count(*source.shape)
        try:
            _write_tile_blob(blob_path, source, codec)
        except FileExistsError:
            pass
        else:
            raise AssertionError("write-once blob path was overwritten")
    if FROZEN_KEYS != {
        "attention": "model.layers.0.self_attn.k_proj.weight",
        "router": "model.layers.0.mlp.gate.weight",
        "expert": "model.layers.0.mlp.experts.0.down_proj.weight",
        "lm_head": "lm_head.weight",
    }:
        raise AssertionError("frozen key regression")
    print(json.dumps({"ok": True, "selftest": True, "donor_weights_opened": False}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--selftest", action="store_true", help="synthetic-only apparatus test; no donor weights")
    actions.add_argument("--preflight", action="store_true", help="read-only pinned metadata/meta-model inventory")
    actions.add_argument("--run", action="store_true", help="launch one bounded local real-weight pilot worker")
    actions.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path, help="explicit local pinned snapshot; never a Hub identifier")
    parser.add_argument("--output-dir", type=Path, help="fresh output path required by --run")
    parser.add_argument("--log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.preflight:
            _bounded, _teacher, report = _preflight(args.snapshot)
            print(json.dumps({"ok": True, "status": "PREFLIGHT_COMPLETE", "preflight": report}, sort_keys=True))
            return 0
        if args.worker:
            inherited = os.environ.get(WORKER_TOKEN_ENV, "")
            if not args.worker_token or not secrets.compare_digest(args.worker_token, inherited):
                raise GateError("internal worker token mismatch")
            if args.output_dir is None or args.log is None or args.result is None:
                raise GateError("internal worker requires output, log, and result paths")
            return _worker(args)
        if args.output_dir is None:
            raise GateError("--run requires --output-dir pointing to a fresh path")
        return _run_parent(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "INCOMPLETE", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
