#!/usr/bin/env python3
"""Bounded STRAT-02 W4 g128 BF16-scale v2 real-weight pilot apparatus.

``--selftest`` is synthetic only.  ``--preflight`` reads pinned metadata and a
meta model only.  ``--run`` is the separately preregistered five-matrix pilot:
it hashes the eleven already-local shards once, samples at most 256 rows per
matrix, and never downloads, opens heldout/text, runs a donor forward, uses a
T4, or converts a full model.  Result directories and control JSON are
write-once; an interrupted run remains INCOMPLETE and is never resumed.
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
WORKER_TOKEN_ENV = "STRAT02_W4_BF16_V2_WORKER_TOKEN"
PILOT_SCHEMA = "strat02_w4_bf16_scale_v2_pilot"
MIN_LAUNCH_RAM_BYTES = 8 * 1024 ** 3
MIN_LAUNCH_OUTPUT_FREE_BYTES = 2 * 1024 ** 3
MIN_RUNTIME_RAM_BYTES = 4 * 1024 ** 3
MAX_PRIVATE_COMMIT_BYTES = 8 * 1024 ** 3
MAX_WALL_SECONDS = 30 * 60
MONITOR_INTERVAL_SECONDS = 5.0
MAX_ROWS, TILE_ROWS, TILE_GROUPS = 256, 16, 16
LINEAR_ATOL, LINEAR_RTOL = 1e-5, 1e-6
EXPECTED_LINEAR_COUNT = 6225
EXPECTED_LINEAR_WEIGHTS = 13_363_052_544
EXPECTED_CATEGORY_COUNTS = {"attention": 64, "router": 16, "expert": 6144, "lm_head": 1}
FROZEN_KEYS = {
    "attention": "model.layers.0.self_attn.k_proj.weight",
    "router": "model.layers.0.mlp.gate.weight",
    "expert_down": "model.layers.0.mlp.experts.0.down_proj.weight",
    "expert_gate": "model.layers.0.mlp.experts.0.gate_proj.weight",
    "lm_head": "lm_head.weight",
}
BRIEF_PATH = HERE.parents[2] / "docs" / "research" / "donor_adaptation" / "briefs" / "BRIEF_STRAT_02_W4_BF16_SCALE_V2_PILOT.md"
SOURCE_PATHS = {
    "pilot": Path(__file__).resolve(),
    "codec": HERE / "strat02_w4_bf16_codec.py",
    "teacher_loader": HERE / "strat02_mmap_teacher.py",
    "bounded_supervisor": HERE / "strat02_bounded_smoke.py",
    "brief": BRIEF_PATH,
}


class GateError(RuntimeError):
    """A frozen apparatus, inventory, provenance, or resource invariant failed."""


class FormatError(GateError):
    """A nonzero group cannot be represented by the frozen v2 BF16 format."""


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
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"utc_timestamp": _utc_now(), **event}, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _prepare_output(output_dir: Path) -> dict[str, Path]:
    root = output_dir.resolve()
    if root.exists():
        raise GateError(f"--output-dir must be a fresh path, not an existing directory: {root}")
    root.mkdir(parents=True, exist_ok=False)
    paths = {"root": root, "manifest": root / "pilot_manifest.json", "log": root / "supervisor_log.jsonl",
             "worker": root / "worker_result.json", "result": root / "supervisor_result.json",
             "worker_stdout": root / "worker_stdout.log", "worker_stderr": root / "worker_stderr.log"}
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
    """Derive the pilot inventory from the pinned model instantiated on meta."""
    teacher._require_target_environment()
    import torch

    EmoConfig, EmoForCausalLM = teacher._import_pinned_emo(report.snapshot)
    config_dict = teacher._read_json(report.snapshot / "config.json", "pinned model config")
    with torch.device("meta"):
        model = EmoForCausalLM(EmoConfig(**config_dict))
    index_keys, state = set(report.index["weight_map"]), model.state_dict()
    if set(state) != index_keys:
        raise GateError("meta model/index keyset mismatch")
    categories: dict[str, list[str]] = {key: [] for key in EXPECTED_CATEGORY_COUNTS}
    linear_count = linear_weights = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = module_name + ".weight"
        if not module_name or module.weight is None or key not in index_keys or tuple(module.weight.shape) != tuple(state[key].shape):
            raise GateError(f"linear inventory/index mismatch for {key}")
        linear_count += 1
        linear_weights += int(module.weight.numel())
        parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        parent = model.get_submodule(parent_name) if parent_name else model
        grandparent_name = parent_name.rsplit(".", 1)[0] if "." in parent_name else ""
        grandparent = model.get_submodule(grandparent_name) if grandparent_name else model
        if module_name == "lm_head" and getattr(model, "lm_head", None) is module:
            categories["lm_head"].append(key)
        elif all(hasattr(parent, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")) and module in {parent.q_proj, parent.k_proj, parent.v_proj, parent.o_proj}:
            categories["attention"].append(key)
        elif hasattr(parent, "gate") and hasattr(parent, "experts") and getattr(parent, "gate") is module:
            categories["router"].append(key)
        elif isinstance(grandparent, torch.nn.ModuleList) and any(candidate is parent for candidate in grandparent):
            owner_name = grandparent_name.rsplit(".", 1)[0] if "." in grandparent_name else ""
            owner = model.get_submodule(owner_name) if owner_name else model
            if getattr(owner, "experts", None) is grandparent and hasattr(owner, "gate"):
                categories["expert"].append(key)
    if (linear_count, linear_weights) != (EXPECTED_LINEAR_COUNT, EXPECTED_LINEAR_WEIGHTS):
        raise GateError("unexpected pinned meta linear inventory")
    counts = {name: len(values) for name, values in categories.items()}
    if counts != EXPECTED_CATEGORY_COUNTS:
        raise GateError(f"linear organ taxonomy differs from pin: {counts}")
    memberships = {key: ("expert" if key.startswith("expert_") else key) for key in FROZEN_KEYS}
    if any(FROZEN_KEYS[name] not in categories[category] for name, category in memberships.items()):
        raise GateError("frozen v2 pilot key disagrees with structural inventory")
    if len(set(FROZEN_KEYS.values())) != len(FROZEN_KEYS):
        raise GateError("frozen v2 pilot keys are not distinct")
    return {"linear_count": linear_count, "linear_weight_count": linear_weights, "selected": dict(FROZEN_KEYS),
            "category_counts": counts, "model_key_count": len(state), "index_key_count": len(index_keys)}


def _preflight(snapshot: Path | None, *, output_dir: Path | None = None) -> tuple[Any, Any, dict[str, Any]]:
    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_w4_bf16_v2_bounded")
    teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_w4_bf16_v2_teacher")
    codec = _load_local_module("strat02_w4_bf16_codec.py", "_strat02_w4_bf16_v2_codec_preflight")
    codec.selftest()  # pinned CPU torch BF16 RNE reference; still synthetic only.
    versions, report = bounded._pinned_environment(), teacher.preflight(snapshot=snapshot)
    status = report.source_status
    if not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO" or not status.get("shards_present"):
        raise GateError("pinned local model/runtime proof failed; no download is attempted")
    available_ram = int(_psutil().virtual_memory().available)
    if available_ram < MIN_LAUNCH_RAM_BYTES:
        raise GateError(f"launch requires >=8 GiB available physical RAM; observed {available_ram} bytes")
    volume = output_dir.resolve() if output_dir is not None else report.snapshot
    free_disk = int(_psutil().disk_usage(str(volume)).free)
    if free_disk < MIN_LAUNCH_OUTPUT_FREE_BYTES:
        raise GateError(f"launch requires >=2 GiB free on output volume; observed {free_disk} bytes")
    return bounded, teacher, {"utc_timestamp": _utc_now(), "runtime_versions": versions, "python": sys.version.split()[0],
        "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"), "snapshot": str(report.snapshot), "metadata": dict(status),
        "available_physical_ram_bytes": available_ram, "output_volume_free_bytes": free_disk,
        "inventory": _inventory_linear_weights(teacher, report),
        "offline_policy": {"hub_download_api": "not used", "hf_hub_offline": True, "transformers_offline": True,
          "weight_access": "pinned local safetensors only in --run worker", "heldout_or_text_access": "not used",
          "donor_forward": "not used", "t4": "not used"},
        "apparatus_sha256": {name: _sha256_file(path) for name, path in SOURCE_PATHS.items()}}


def _write_tile_blob(path: Path, source: np.ndarray, codec: Any, *, max_rows: int = TILE_ROWS, max_groups: int = TILE_GROUPS) -> None:
    if source.dtype != np.float32 or source.ndim != 2:
        raise GateError("sampled source must be a two-dimensional float32 array")
    rows, columns = source.shape
    groups = columns // codec.GROUP_SIZE
    expected_size, scale_bytes = codec.w4_bf16_byte_count(rows, columns), rows * groups * 2
    try:
        with path.open("xb") as handle:
            handle.truncate(expected_size)
            for tile in codec.iter_encode_w4_bf16_tiles((source,), columns, max_rows_per_tile=max_rows, max_groups_per_tile=max_groups):
                tile_rows, tile_groups = tile.scale_bits.shape
                for local_row in range(tile_rows):
                    row = tile.row_offset + local_row
                    handle.seek((row * groups + tile.group_offset) * 2)
                    handle.write(tile.scale_bits[local_row].astype("<u2", copy=False).tobytes(order="C"))
                    handle.seek(scale_bytes + (row * groups + tile.group_offset) * 64)
                    handle.write(tile.packed_codes[local_row].tobytes(order="C"))
            handle.flush()
            os.fsync(handle.fileno())
    except ValueError as exc:
        if "invalid BF16 candidate scale" in str(exc):
            raise FormatError(str(exc)) from exc
        raise GateError(str(exc)) from exc
    if path.stat().st_size != expected_size:
        raise GateError(f"tiled W4 BF16 blob size mismatch for {path.name}")


def _rowwise_reference_blob(source: np.ndarray, codec: Any) -> bytes:
    rows, columns = source.shape
    groups = columns // codec.GROUP_SIZE
    scales, packed = np.empty((rows, groups), dtype=np.uint16), np.empty((rows, groups * 64), dtype=np.uint8)
    for row, (scale_row, packed_row) in enumerate(codec.iter_encode_w4_bf16_rows((source[i] for i in range(rows)), columns)):
        scales[row], packed[row] = scale_row, packed_row
    return codec.w4_bf16_to_blob(codec.W4BF16Encoded((rows, columns), scales, packed))


def _scalar_decode_from_blob(blob: bytes, shape: tuple[int, int], codec: Any) -> np.ndarray:
    encoded = codec.w4_bf16_from_blob(blob, shape)
    rows, columns = shape
    decoded = np.empty(shape, dtype=np.float32)
    for row in range(rows):
        for group in range(columns // codec.GROUP_SIZE):
            codes = codec.unpack_w4_codes_scalar(encoded.packed_codes[row, group * 64:(group + 1) * 64])
            decoded[row, group * 128:(group + 1) * 128] = codec.dequantize_w4_bf16_group_scalar(encoded.scale_bits[row, group], codes)
    return decoded


def _assert_packed_roundtrip(blob: bytes, source: np.ndarray, codec: Any) -> dict[str, Any]:
    try:
        reference = _rowwise_reference_blob(source, codec)
    except ValueError as exc:
        # The tile path has already accepted this same source. A scalar-only
        # rejection is an implementation disagreement, not a format verdict.
        raise GateError("scalar BF16 reference rejected a tile-accepted source") from exc
    if blob != reference:
        raise GateError("tiled W4 BF16 bytes differ from scalar row-wise reference")
    try:
        vector = codec.dequantize_w4_bf16(codec.w4_bf16_from_blob(blob, tuple(source.shape)))
        scalar = _scalar_decode_from_blob(blob, tuple(source.shape), codec)
    except ValueError as exc:
        raise GateError("BF16 blob decoder rejected tile-accepted bytes") from exc
    if not np.array_equal(scalar.view("<u4"), vector.view("<u4")):
        raise GateError("scalar and vector W4 BF16 decoding from blob differ")
    import torch
    reconstructed = torch.from_numpy(vector.copy())
    synthetic = torch.linspace(-1.0, 1.0, steps=3 * source.shape[1], dtype=torch.float32).reshape(3, source.shape[1])
    linear = torch.nn.Linear(source.shape[1], source.shape[0], bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(reconstructed)
        observed = linear(synthetic)
        expected = synthetic @ torch.from_numpy(scalar.copy()).T
    if not torch.allclose(observed, expected, atol=LINEAR_ATOL, rtol=LINEAR_RTOL):
        raise GateError("synthetic nn.Linear differs from X @ W_hat.T reconstructed from blob")
    return {"decoded_f32_exact": True, "synthetic_linear_parity": True,
            "reference_blob_sha256": hashlib.sha256(reference).hexdigest()}


def _read_sampled_tensor(report: Any, key: str) -> tuple[np.ndarray, str]:
    from safetensors import safe_open
    import torch
    shard = report.index["weight_map"].get(key)
    if not isinstance(shard, str):
        raise GateError(f"selected key missing from safetensors index: {key}")
    with safe_open(str(report.snapshot / shard), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(key)
        shape = tuple(int(value) for value in tensor_slice.get_shape())
        if len(shape) != 2 or shape[1] % 128:
            raise GateError(f"selected matrix shape violates v2 constraints: {key} {shape}")
        rows = min(MAX_ROWS, shape[0])
        tensor = tensor_slice[:rows]
        if rows <= 0 or tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tuple(tensor.shape) != (rows, shape[1]):
            raise GateError(f"selected linear tensor is not a CPU F32 matrix: {key}")
        source = tensor.numpy()
        if source.dtype != np.float32:
            raise GateError(f"failed to obtain sampled F32 rows for {key}")
        return source.copy(), shard


def _worker_failure_status(exc: BaseException) -> str:
    """Keep the preregistered BF16 representability verdict distinct."""
    if isinstance(exc, FormatError):
        return "VOID_FORMAT"
    if isinstance(exc, GateError):
        return "VOID_APPARATUS"
    return "INCOMPLETE"


def _worker(args: argparse.Namespace) -> int:
    completed: list[dict[str, Any]] = []
    try:
        manifest = json.loads((args.output_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("parent_preflight", {}).get("apparatus_sha256") != {name: _sha256_file(path) for name, path in SOURCE_PATHS.items()}:
            raise GateError("pilot/codec/loader/brief changed after parent preflight")
        teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_w4_bf16_v2_worker_teacher")
        codec = _load_local_module("strat02_w4_bf16_codec.py", "_strat02_w4_bf16_v2_worker_codec")
        codec.selftest()
        report = teacher.preflight(snapshot=args.snapshot)
        if _inventory_linear_weights(teacher, report)["selected"] != FROZEN_KEYS:
            raise GateError("worker structural inventory changed from frozen v2 selection")
        verified = teacher.verify_shards(report)  # the one full local shard hash pass
        if not isinstance(verified, teacher.VerifiedSnapshot):
            raise GateError("pinned shard verification did not produce a capability token")
        for organ, key in FROZEN_KEYS.items():
            planned = {"organ": organ, "key": key, "selected_before_tensor_read": True}
            _append_log(args.log, {"event": "matrix_selected", **planned})
            teacher._assert_unchanged(verified)
            source, shard = _read_sampled_tensor(report, key)
            blob_path = args.output_dir / f"{organ}.w4g128.bf16scale-v2.bin"
            _write_tile_blob(blob_path, source, codec)
            blob = blob_path.read_bytes()
            encoded = codec.w4_bf16_from_blob(blob, source.shape)
            validation = _assert_packed_roundtrip(blob, source, codec)
            completed.append({**planned, "source_shard": shard, "shape": list(source.shape), "source_dtype": "float32",
                "source_f32_slice_sha256": hashlib.sha256(source.tobytes(order="C")).hexdigest(),
                "format": "strat02_w4_g128_bf16scale_v2", "blob_file": blob_path.name, "blob_bytes": len(blob),
                "blob_sha256": hashlib.sha256(blob).hexdigest(), "zero_groups": int(np.count_nonzero(encoded.scale_bits == 0)),
                "invalid_groups": 0, **validation})
            _append_log(args.log, {"event": "matrix_complete", "organ": organ, "blob_bytes": len(blob)})
        _write_json_once(args.result.resolve(), {"ok": True, "status": "COMPLETE", "utc_timestamp": _utc_now(), "matrices": completed,
            "notes": ["Bounded v2 apparatus only; no full conversion, donor forward, heldout, text, download, or T4 use.",
                      "All decodes and operators use reconstructed packed blob values, not original donor weights."]})
        return 0
    except Exception as exc:
        status = _worker_failure_status(exc)
        try:
            _write_json_once(args.result.resolve(), {"ok": False, "status": status, "utc_timestamp": _utc_now(), "completed_matrices": completed,
                "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        except (GateError, OSError):
            pass
        return 1


def _process_sample(process: Any) -> dict[str, int]:
    try:
        basic, full = process.memory_info(), process.memory_full_info()
        private = getattr(full, "private", None)
    except Exception as exc:
        raise GateError(f"cannot read worker resource sample: {type(exc).__name__}: {exc}") from exc
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


def _monitor_child(process: Any, *, psutil: Any, on_sample: Callable[[Mapping[str, Any]], None], expected_executable: Path,
                   now: Callable[[], float] = time.monotonic, interval_seconds: float = MONITOR_INTERVAL_SECONDS,
                   wall_limit_seconds: float = MAX_WALL_SECONDS) -> MonitorOutcome:
    started, samples, checked, resource_process = now(), 0, False, None
    while True:
        elapsed, exit_code = now() - started, process.poll()
        if exit_code is not None:
            return MonitorOutcome("CHILD_EXITED", int(exit_code), None, elapsed, samples)
        if elapsed >= wall_limit_seconds:
            return MonitorOutcome("VOID_RESOURCE", _terminate_only_child(process), "wall_clock_exceeded", elapsed, samples)
        try:
            resource_process = resource_process or psutil.Process(process.pid)
            if not checked:
                if Path(resource_process.exe()).resolve() != expected_executable.resolve():
                    return MonitorOutcome("VOID_APPARATUS", _terminate_only_child(process), "monitored_executable_mismatch", elapsed, samples)
                checked = True
            sample, available = _process_sample(resource_process), int(psutil.virtual_memory().available)
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
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _run_parent(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    try:
        bounded, _teacher, preflight = _preflight(args.snapshot, output_dir=paths["root"])
        _write_json_once(paths["manifest"], {"schema": PILOT_SCHEMA, "purpose": "bounded real-weight W4 g128 BF16-scale v2 apparatus; not a quality/rate run",
            "selected_before_tensor_read": FROZEN_KEYS, "limits": {"launch_available_physical_ram_bytes_min": MIN_LAUNCH_RAM_BYTES,
              "output_volume_free_bytes_min": MIN_LAUNCH_OUTPUT_FREE_BYTES, "runtime_available_physical_ram_bytes_min": MIN_RUNTIME_RAM_BYTES,
              "worker_private_commit_bytes_max": MAX_PRIVATE_COMMIT_BYTES, "wall_seconds_max": MAX_WALL_SECONDS,
              "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS, "max_rows_per_matrix": MAX_ROWS}, "parent_preflight": preflight,
            "operator_parity": {"atol": LINEAR_ATOL, "rtol": LINEAR_RTOL},
            "output": {"auto_resume": False, "heldout_or_text": False, "full_conversion": False, "worker_stdout": paths["worker_stdout"].name,
                       "worker_stderr": paths["worker_stderr"].name}})
        _append_log(paths["log"], {"event": "parent_preflight_passed"})
    except Exception as exc:
        _write_json_once(paths["manifest"], {"schema": PILOT_SCHEMA, "preflight": "failed"})
        _append_log(paths["log"], {"event": "parent_preflight_failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json_once(paths["result"], {"ok": False, "status": "INCOMPLETE", "failure_classification": "PREFLIGHT_ERROR", "error": str(exc)})
        return 1
    process: Any | None = None
    try:
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        worker_python, token = bounded._direct_worker_python(environment), secrets.token_urlsafe(32)
        environment[WORKER_TOKEN_ENV] = token
        command = [worker_python, str(Path(__file__).resolve()), "--worker", "--snapshot", preflight["snapshot"], "--output-dir", str(paths["root"]),
                   "--log", str(paths["log"]), "--result", str(paths["worker"]), "--worker-token", token]
        _append_log(paths["log"], {"event": "child_launch", "python": worker_python, "termination_scope": "direct_worker_pid_only"})
        with paths["worker_stdout"].open("x", encoding="utf-8") as stdout, paths["worker_stderr"].open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=str(HERE), env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            outcome = _monitor_child(process, psutil=_psutil(), expected_executable=Path(worker_python), on_sample=lambda sample: _append_log(paths["log"], {"event": "resource_sample", **sample}))
        _append_log(paths["log"], {"event": "monitor_finished", **asdict(outcome)})
        worker = _load_json(paths["worker"])
        if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
            classification = outcome.status
        elif worker and worker.get("status") in {"VOID_APPARATUS", "VOID_FORMAT"}:
            classification = worker["status"]
        elif outcome.exit_code != 0:
            classification = "WORKER_ERROR"
        elif not worker or worker.get("ok") is not True or worker.get("status") != "COMPLETE":
            classification = "WORKER_PROTOCOL_ERROR"
        else:
            classification = None
        final = {"ok": classification is None, "status": "COMPLETE" if classification is None else "INCOMPLETE", "failure_classification": classification,
                 "utc_timestamp": _utc_now(), "child_pid": process.pid, "monitor": asdict(outcome), "worker_result": worker}
    except Exception as exc:
        if process is not None:
            _terminate_only_child(process)
        _append_log(paths["log"], {"event": "parent_supervision_error", "error_type": type(exc).__name__, "error": str(exc)})
        final = {"ok": False, "status": "INCOMPLETE", "failure_classification": "PARENT_SUPERVISION_ERROR", "error": str(exc), "traceback": traceback.format_exc()}
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": final["ok"], "status": final["status"], "result": str(paths["result"])}, sort_keys=True))
    return 0 if final["ok"] else 1


def _selftest() -> None:
    codec = _load_local_module("strat02_w4_bf16_codec.py", "_strat02_w4_bf16_v2_selftest_codec")
    codec.selftest()
    rng = np.random.default_rng(20260917)
    source = rng.normal(size=(7, 384)).astype(np.float32)
    source[0, :128] = 0
    with tempfile.TemporaryDirectory(prefix="strat02-w4-bf16-v2-selftest-") as temporary:
        path = Path(temporary) / "synthetic.w4g128.bf16.bin"
        _write_tile_blob(path, source, codec, max_rows=2, max_groups=2)
        report = _assert_packed_roundtrip(path.read_bytes(), source, codec)
        if not report["decoded_f32_exact"] or not report["synthetic_linear_parity"]:
            raise AssertionError("packed synthetic v2 round-trip regression")
        try:
            _write_tile_blob(path, source, codec)
        except FileExistsError:
            pass
        else:
            raise AssertionError("write-once blob path was overwritten")
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
            if not args.worker_token or not secrets.compare_digest(args.worker_token, os.environ.get(WORKER_TOKEN_ENV, "")):
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
