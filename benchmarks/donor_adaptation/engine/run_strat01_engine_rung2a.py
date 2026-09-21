"""Build, execute once, and independently adjudicate STRAT-01 engine rung 2A.

This is a correctness experiment, not a throughput benchmark.  It validates
the complete apparatus before either implementation sees the accepted donor,
then runs the C engine once and the pinned llama.cpp reference once.  Both
processes execute the paired prefill8 and cached7p1 arms in that invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ENGINE = ROOT / "benchmarks" / "phase60" / "engine.c"
RUNG2A_HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2a.h"
REFERENCE_SOURCE = HERE / "strat01_engine_rung2a_reference.cpp"
REFERENCE_BUILDER = HERE / "build_strat01_engine_rung2a_reference.py"
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROTOCOL_20260921.md"
DEFAULT_MODEL = ROOT / "benchmarks" / "donor_adaptation" / "density" / "results" / "strat01_gigachat_q4_97045b2" / "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_rung2a_20260921"
PINNED_LLAMA = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4")
LLAMA_COMMIT = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
EXPECTED_BYTES = 6_474_702_976
EXPECTED_SHA256 = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
TOKENS = [1, 72, 14, 14129, 14, 2135, 1512, 2015]
POSITIONS = list(range(8))
ARMS = ("prefill8", "cached7p1")
GENERAL_LIMITS = (2e-3, 1e-2)
TERMINAL_LIMITS = (1e-3, 5e-3)
CONTINUITY_LIMITS = (2e-6, 1e-5)
COMPILE_FLAGS = ["-std=c11", "-O3", "-mavx2", "-mfma"]
EXPECTED_C_CONFIG = (
    "reference=llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2;cpu_threads=1;"
    "n_ctx=8;n_batch=8;n_ubatch=8;flash_attn=false;offload_kqv=false;"
    "type_k=f16;type_v=f16-no-allocation-mla;cache=layer-slot-576-f16-k-only-latent512-rope64;"
    "tokens=1,72,14,14129,14,2135,1512,2015;positions=0,1,2,3,4,5,6,7;"
    "rms_eps=1e-6;rope=deepseek2-normal-yarn;rope_base=100000;rope_factor=64;"
    "rope_orig_ctx=4096;beta_fast=32;beta_slow=1;mscale=1;mscale_all_dim=1;"
    "build=clang-c11-O3-mavx2-mfma-no-fast-math;fp_contract=off-c11-pragma;"
    "payload=f32le-token-major;adjudication=external-reference-only"
)
EXPECTED_REFERENCE_CONFIG = {
    "n_gpu_layers": 0,
    "n_threads": 1,
    "n_threads_batch": 1,
    "n_ctx": 8,
    "n_batch": 8,
    "n_ubatch": 8,
    "flash_attn": "disabled",
    "offload_kqv": False,
    "op_offload": False,
    "type_k": "F16",
    "type_v": "F16",
}
SHAPES = {
    "attn_norm-0": [1536, 8],
    "q-0": [192, 32, 8],
    "kv_cmpr_pe-0": [576, 8],
    "k_pe-0": [64, 1, 8],
    "kv_cmpr-0": [512, 8],
    "q_pe-0": [64, 32, 8],
    "q_nope_absorbed_perm-0": [512, 32, 8],
    "Qcur-0": [576, 32, 8],
    "Kcur-0": [576, 1, 8],
    "Vcur-0": [512, 1, 8],
    "kqv_out-0": [6144, 8],
    "ffn_inp-0": [1536, 8],
}
EXPECTED_OPS = {
    "attn_norm-0": "MUL",
    "q-0": "RESHAPE",
    "kv_cmpr_pe-0": "MUL_MAT",
    "k_pe-0": "ROPE",
    "kv_cmpr-0": "MUL",
    "q_pe-0": "ROPE",
    "q_nope_absorbed_perm-0": "PERMUTE",
    "Qcur-0": "CONCAT",
    "Kcur-0": "CONCAT",
    "Vcur-0": "RESHAPE",
    "kqv_out-0": "CONT",
    "ffn_inp-0": "ADD",
}
EXPECTED_ORDINALS = {name: 0 for name in SHAPES}
EXPECTED_ORDINALS.update({"q-0": 1, "k_pe-0": 1, "kv_cmpr-0": 1, "q_pe-0": 1})
TEST_MODULES = (
    "benchmarks.donor_adaptation.engine.test_strat01_engine_rung0",
    "benchmarks.donor_adaptation.engine.test_strat01_engine_rung1",
    "benchmarks.donor_adaptation.engine.test_strat01_engine_rung2a",
    "benchmarks.donor_adaptation.engine.test_strat01_engine_rung2a_reference",
    "benchmarks.donor_adaptation.engine.test_strat01_engine_rung2a_runner",
)
NON_CLAIMS = [
    "full-model or later-layer parity",
    "tokenizer, logits, sampling, or generation parity",
    "language-model quality through the C engine",
    "RAM fit or accepted-token throughput",
    "any speed or SPEED_LEDGER claim",
]


class RunnerError(RuntimeError):
    def __init__(self, message: str, *, status: str = "VOID_ENGINE_RUNG2A") -> None:
        super().__init__(message)
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_value(args: list[str], cwd: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_command(command: list[str], *, cwd: Path, output: Path, label: str, timeout: int) -> dict[str, Any]:
    started_utc = utc_now()
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=timeout)
        record = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        record = {
            "returncode": None,
            "stdout": str(getattr(exc, "stdout", "") or ""),
            "stderr": str(getattr(exc, "stderr", "") or ""),
            "spawn_or_timeout_error": str(exc),
        }
    (output / f"{label}.stdout.log").write_text(record["stdout"], encoding="utf-8")
    (output / f"{label}.stderr.log").write_text(record["stderr"], encoding="utf-8")
    record.update({
        "command": command,
        "cwd": str(cwd),
        "started_utc": started_utc,
        "seconds": time.perf_counter() - started,
        "stdout_path": str(output / f"{label}.stdout.log"),
        "stderr_path": str(output / f"{label}.stderr.log"),
    })
    return record


def require_ok(record: dict[str, Any], label: str) -> None:
    if record.get("returncode") != 0:
        detail = record.get("spawn_or_timeout_error") or record.get("stderr") or record.get("stdout") or ""
        raise RunnerError(f"{label} failed (rc={record.get('returncode')}): {str(detail)[-1600:]}")


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{label} is missing or malformed: {exc}") from exc


def contained_file(root: Path, reported: str, expected_bytes: int, label: str) -> Path:
    candidate = Path(reported)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RunnerError(f"{label} path is absent or escapes its raw directory") from exc
    if not resolved.is_file() or resolved.stat().st_size != expected_bytes:
        raise RunnerError(f"{label} payload byte count mismatch")
    return resolved


def load_payload(path: Path, expected_count: int, expected_hash: str, label: str) -> np.ndarray:
    if sha256_file(path) != expected_hash:
        raise RunnerError(f"{label} payload SHA-256 mismatch")
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != expected_count or not bool(np.isfinite(values).all()):
        raise RunnerError(f"{label} payload count/finiteness mismatch")
    return values


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if c.shape != r.shape or c.size == 0 or not bool(np.isfinite(c).all() and np.isfinite(r).all()):
        raise RunnerError("metric inputs differ in shape, are empty, or are non-finite")
    delta = c - r
    nrmse = math.sqrt(float(np.dot(delta, delta)) / max(float(np.dot(r, r)), 1e-30))
    normalized_max = float(np.max(np.abs(delta))) / max(float(np.max(np.abs(r))), 1e-6)
    return {"nrmse": nrmse, "normalized_max": normalized_max}


def judged_metrics(candidate: np.ndarray, reference: np.ndarray, limits: tuple[float, float]) -> dict[str, Any]:
    result: dict[str, Any] = metrics(candidate, reference)
    result.update({"nrmse_limit": limits[0], "normalized_max_limit": limits[1]})
    result["pass"] = result["nrmse"] <= limits[0] and result["normalized_max"] <= limits[1]
    return result


def token7(values: np.ndarray, shape: list[int]) -> np.ndarray:
    per_token = math.prod(shape[:-1])
    if shape[-1] != 8 or values.size != per_token * 8:
        raise RunnerError("cannot extract frozen token-7 slice")
    return values[7 * per_token : 8 * per_token]


def source_inventory() -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "engine": ENGINE,
        "rung2a_header": RUNG2A_HEADER,
        "reference_source": REFERENCE_SOURCE,
        "reference_builder": REFERENCE_BUILDER,
        "protocol": PROTOCOL,
        "rung0_test": HERE / "test_strat01_engine_rung0.py",
        "rung1_test": HERE / "test_strat01_engine_rung1.py",
        "rung2a_c_test": HERE / "test_strat01_engine_rung2a.py",
        "rung2a_reference_test": HERE / "test_strat01_engine_rung2a_reference.py",
        "rung2a_runner_test": HERE / "test_strat01_engine_rung2a_runner.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunnerError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def validate_c_outputs(c_root: Path, source_hashes: dict[str, Any], model: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    report = read_json(c_root / "strat01_rung2a.json", "C report")
    required = {"command", "c_state", "self_certifies_pass", "input_path", "byte_size", "sha256", "reference_revision", "CONFIG", "compiler_family", "compiler_embedded_version", "compiler_resolved_path_and_full_version", "engine_source_sha256", "rung2a_source_sha256", "token_ids", "positions", "arms", "timing_or_rate_claim"}
    if not isinstance(report, dict) or set(report) != required:
        raise RunnerError("C report schema mismatch")
    if report["command"] != "--strat01-gguf-rung2a" or report["c_state"] != "ENGINE_OUTPUT_READY_PENDING_REFERENCE" or report["self_certifies_pass"] is not False:
        raise RunnerError("C report state/command/self-certification mismatch")
    if Path(report["input_path"]).resolve(strict=True) != model.resolve(strict=True) or report["byte_size"] != EXPECTED_BYTES or report["sha256"] != EXPECTED_SHA256:
        raise RunnerError("C report artifact identity mismatch")
    if report["reference_revision"] != f"llama.cpp {LLAMA_COMMIT}" or report["CONFIG"] != EXPECTED_C_CONFIG:
        raise RunnerError("C report reference/config mismatch")
    if report["compiler_family"] != "clang" or report["compiler_resolved_path_and_full_version"] != "EXTERNAL_RUNNER_REQUIRED":
        raise RunnerError("C report compiler contract mismatch")
    if report["engine_source_sha256"] != source_hashes["engine"]["sha256"] or report["rung2a_source_sha256"] != source_hashes["rung2a_header"]["sha256"]:
        raise RunnerError("C report source hash mismatch")
    if report["token_ids"] != TOKENS or report["positions"] != POSITIONS or report["timing_or_rate_claim"] is not None:
        raise RunnerError("C report fixed input or non-speed contract mismatch")
    if report["arms"] != [{"name": "prefill8", "manifest": "prefill8_manifest.json"}, {"name": "cached7p1", "manifest": "cached7p1_manifest.json"}]:
        raise RunnerError("C report paired-arm contract mismatch")

    tensors: dict[str, np.ndarray] = {}
    caches: dict[str, np.ndarray] = {}
    manifests: dict[str, Any] = {}
    for arm in ARMS:
        manifest = read_json(c_root / f"{arm}_manifest.json", f"C {arm} manifest")
        if not isinstance(manifest, dict) or set(manifest) != {"arm", "payload_encoding", "shape_order", "tensors", "cache"} or manifest["arm"] != arm:
            raise RunnerError(f"C {arm} manifest schema mismatch")
        if manifest["payload_encoding"] != "IEEE-754 binary32 little-endian" or len(manifest["tensors"]) != len(SHAPES):
            raise RunnerError(f"C {arm} payload contract mismatch")
        by_name = {item.get("name"): item for item in manifest["tensors"] if isinstance(item, dict)}
        if set(by_name) != set(SHAPES) or len(by_name) != len(manifest["tensors"]):
            raise RunnerError(f"C {arm} tensor set is incomplete or duplicated")
        for name, shape in SHAPES.items():
            item = by_name[name]
            if set(item) != {"name", "ordinal", "selection", "op", "type", "logical_shape", "payload_order", "byte_count", "path", "sha256"}:
                raise RunnerError(f"C {arm}/{name} manifest entry schema mismatch")
            count = math.prod(shape)
            if item["logical_shape"] != shape or item["op"] != EXPECTED_OPS[name] or item["ordinal"] != EXPECTED_ORDINALS[name] or item["type"] != "F32" or item["byte_count"] != 4 * count:
                raise RunnerError(f"C {arm}/{name} identity/shape/type mismatch")
            path = contained_file(c_root, item["path"], 4 * count, f"C {arm}/{name}")
            tensors[f"{arm}/{name}"] = load_payload(path, count, item["sha256"], f"C {arm}/{name}")
        cache = manifest["cache"]
        if cache.get("storage") != "F16" or cache.get("layout") != "layer,slot,576[latent512,rope64]" or cache.get("layer") != 0 or cache.get("row_length") != 576 or cache.get("no_separate_v_cache") is not True:
            raise RunnerError(f"C {arm} cache layout invariant mismatch")
        checkpoints = [("final", 8)] + ([("prefix7", 7)] if arm == "cached7p1" else [])
        for checkpoint, rows in checkpoints:
            item = cache.get(checkpoint)
            expected_slots = list(range(rows))
            if not isinstance(item, dict) or item.get("occupied_slots") != expected_slots or item.get("absolute_positions") != expected_slots or item.get("payload_type") != "dequantized-F32LE":
                raise RunnerError(f"C {arm}/{checkpoint} cache checkpoint mismatch")
            path = contained_file(c_root, item["path"], rows * 576 * 4, f"C {arm}/{checkpoint} cache")
            caches[f"{arm}/{checkpoint}"] = load_payload(path, rows * 576, item["sha256"], f"C {arm}/{checkpoint} cache")
        manifests[arm] = manifest
    return tensors, {"report": report, "manifests": manifests}, caches


def reference_source_identity(selection: dict[str, Any], arm: str, name: str) -> tuple[dict[str, Any], list[int]]:
    if selection.get("logical_shape") != SHAPES[name]:
        raise RunnerError(f"reference {arm}/{name} logical shape mismatch")
    if arm == "prefill8":
        if selection.get("composition") != "single_prefill8_callback" or not isinstance(selection.get("source"), dict):
            raise RunnerError(f"reference {arm}/{name} selection composition mismatch")
        sources = [selection["source"]]
    else:
        if selection.get("composition") != "prefix7_then_final1" or not isinstance(selection.get("prefix_source"), dict) or not isinstance(selection.get("final_source"), dict):
            raise RunnerError(f"reference {arm}/{name} selection composition mismatch")
        sources = [selection["prefix_source"], selection["final_source"]]
    token_lengths: list[int] = []
    for source in sources:
        if source.get("name") != name or source.get("op") != EXPECTED_OPS[name] or source.get("ordinal") != EXPECTED_ORDINALS[name] or source.get("type") not in {"F32", "F16"}:
            raise RunnerError(f"reference {arm}/{name} callback identity mismatch")
        shape = source.get("shape")
        if not isinstance(shape, list) or len(shape) != len(SHAPES[name]) or shape[:-1] != SHAPES[name][:-1]:
            raise RunnerError(f"reference {arm}/{name} callback shape mismatch")
        token_lengths.append(shape[-1])
    return sources[-1], token_lengths


def validate_reference_outputs(ref_root: Path, model: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    root_manifest = read_json(ref_root / "manifest.json", "reference root manifest")
    required = {"schema", "state", "llama_cpp_commit", "model", "config", "fixed_tokens", "fixed_positions", "arms"}
    if not isinstance(root_manifest, dict) or set(root_manifest) != required:
        raise RunnerError("reference root manifest schema mismatch")
    if root_manifest["schema"] != "strat01_engine_rung2a_reference_manifest_v1" or root_manifest["state"] != "REFERENCE_TRACE_READY_PENDING_C_ENGINE" or root_manifest["llama_cpp_commit"] != LLAMA_COMMIT:
        raise RunnerError("reference root state/revision mismatch")
    expected_model = {"path": model.as_posix(), "bytes": EXPECTED_BYTES, "sha256": EXPECTED_SHA256}
    actual_model = root_manifest["model"]
    if Path(actual_model.get("path", "")).resolve(strict=True) != model.resolve(strict=True) or actual_model.get("bytes") != EXPECTED_BYTES or actual_model.get("sha256") != EXPECTED_SHA256:
        raise RunnerError("reference artifact identity mismatch")
    if root_manifest["config"] != EXPECTED_REFERENCE_CONFIG or root_manifest["fixed_tokens"] != TOKENS or root_manifest["fixed_positions"] != POSITIONS:
        raise RunnerError("reference config or fixed input mismatch")
    expected_arms = [{"arm": arm, "manifest": f"{arm}/manifest.json"} for arm in ARMS]
    if root_manifest["arms"] != expected_arms:
        raise RunnerError("reference did not execute exactly both paired arms")

    tensors: dict[str, np.ndarray] = {}
    logical_cache: dict[str, np.ndarray] = {}
    manifests: dict[str, Any] = {}
    for arm in ARMS:
        manifest = read_json(ref_root / arm / "manifest.json", f"reference {arm} manifest")
        if not isinstance(manifest, dict) or set(manifest) != {"schema", "arm", "logical_cache_contract", "callback_records", "logical_selection", "payloads"} or manifest["arm"] != arm:
            raise RunnerError(f"reference {arm} manifest schema mismatch")
        if manifest["schema"] != root_manifest["schema"] or not isinstance(manifest["callback_records"], list):
            raise RunnerError(f"reference {arm} callback manifest mismatch")
        selection = manifest["logical_selection"]
        if not isinstance(selection, dict) or set(selection) != set(SHAPES):
            raise RunnerError(f"reference {arm} logical selection is incomplete")
        for name in SHAPES:
            _, token_lengths = reference_source_identity(selection[name], arm, name)
            if token_lengths != ([8] if arm == "prefill8" else [7, 1]):
                raise RunnerError(f"reference {arm}/{name} source token decomposition mismatch")
        payload_entries = manifest["payloads"]
        if not isinstance(payload_entries, list):
            raise RunnerError(f"reference {arm} payload list malformed")
        keyed: dict[tuple[str, str], dict[str, Any]] = {}
        for item in payload_entries:
            key = (item.get("logical"), item.get("kind")) if isinstance(item, dict) else (None, None)
            if key in keyed:
                raise RunnerError(f"reference {arm} duplicate payload entry")
            keyed[key] = item
        expected_keys = {(name, kind) for name in SHAPES for kind in ("full", "token7")}
        if arm == "cached7p1":
            expected_keys.add(("Kcur-0", "prefix_logical_rows"))
        if set(keyed) != expected_keys:
            raise RunnerError(f"reference {arm} payload set mismatch")
        for name, shape in SHAPES.items():
            count = math.prod(shape)
            full = keyed[(name, "full")]
            if set(full) != {"logical", "kind", "path", "byte_count", "sha256"} or full["byte_count"] != count * 4:
                raise RunnerError(f"reference {arm}/{name} full payload metadata mismatch")
            path = contained_file(ref_root, full["path"], count * 4, f"reference {arm}/{name}")
            tensors[f"{arm}/{name}"] = load_payload(path, count, full["sha256"], f"reference {arm}/{name}")
            token_item = keyed[(name, "token7")]
            token_count = math.prod(shape[:-1])
            token_path = contained_file(ref_root, token_item["path"], token_count * 4, f"reference {arm}/{name} token7")
            token_values = load_payload(token_path, token_count, token_item["sha256"], f"reference {arm}/{name} token7")
            if not np.array_equal(token_values, token7(tensors[f"{arm}/{name}"], shape)):
                raise RunnerError(f"reference {arm}/{name} token7 payload is inconsistent with full payload")
        cache = manifest["logical_cache_contract"]
        if cache.get("extraction") != "logical_Kcur_callback_values_only" or cache.get("physical_bytes_claimed") is not False or cache.get("storage_type") != "F16" or cache.get("row_length") != 576 or cache.get("separate_v_cache") is not False:
            raise RunnerError(f"reference {arm} cache contract mismatch")
        checkpoints = cache.get("checkpoints")
        if not isinstance(checkpoints, list) or checkpoints[-1].get("occupied_positions") != POSITIONS or checkpoints[-1].get("token7_position") != 7:
            raise RunnerError(f"reference {arm} final cache checkpoint mismatch")
        logical_cache[f"{arm}/final"] = tensors[f"{arm}/Kcur-0"]
        if arm == "cached7p1":
            if len(checkpoints) != 2 or checkpoints[0].get("occupied_positions") != list(range(7)):
                raise RunnerError("reference cached7p1 prefix checkpoint mismatch")
            item = keyed[("Kcur-0", "prefix_logical_rows")]
            path = contained_file(ref_root, item["path"], 7 * 576 * 4, "reference cached7p1 prefix Kcur")
            logical_cache[f"{arm}/prefix7"] = load_payload(path, 7 * 576, item["sha256"], "reference cached7p1 prefix Kcur")
        manifests[arm] = manifest
    return tensors, {"root": root_manifest, "manifests": manifests}, logical_cache


def adjudicate(c_tensors: dict[str, np.ndarray], ref_tensors: dict[str, np.ndarray], c_cache: dict[str, np.ndarray], ref_cache: dict[str, np.ndarray]) -> dict[str, Any]:
    tensor_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for arm in ARMS:
        for name, shape in SHAPES.items():
            limits = TERMINAL_LIMITS if name == "ffn_inp-0" else GENERAL_LIMITS
            result = judged_metrics(c_tensors[f"{arm}/{name}"], ref_tensors[f"{arm}/{name}"], limits)
            result.update({"arm": arm, "tensor": name, "logical_shape": shape})
            tensor_results.append(result)
            if not result["pass"]:
                failures.append(f"{arm}/{name} numerical parity")

    continuity: list[dict[str, Any]] = []
    for implementation, tensors in (("c_engine", c_tensors), ("pinned_reference", ref_tensors)):
        prefill = token7(tensors["prefill8/ffn_inp-0"], SHAPES["ffn_inp-0"])
        cached = token7(tensors["cached7p1/ffn_inp-0"], SHAPES["ffn_inp-0"])
        result = judged_metrics(cached, prefill, CONTINUITY_LIMITS)
        result["implementation"] = implementation
        continuity.append(result)
        if not result["pass"]:
            failures.append(f"{implementation} prefill/cache continuity")

    cache_results: list[dict[str, Any]] = []
    for key in ("prefill8/final", "cached7p1/prefix7", "cached7p1/final"):
        result = judged_metrics(c_cache[key], ref_cache[key], GENERAL_LIMITS)
        result["checkpoint"] = key
        cache_results.append(result)
        if not result["pass"]:
            failures.append(f"{key} compact-cache parity")
    return {
        "status": "PASS_ENGINE_RUNG2A" if not failures else "FAIL_ENGINE_RUNG2A",
        "tensor_results": tensor_results,
        "continuity_results": continuity,
        "cache_results": cache_results,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only", action="store_true", help="build and run model-free gates, but do not read or execute the donor")
    args = parser.parse_args()
    model = args.model.resolve()
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started_utc = utc_now()
    started = time.perf_counter()
    status = "VOID_ENGINE_RUNG2A"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    source_hashes: dict[str, Any] = {}
    adjudication: dict[str, Any] = {"status": "NOT_RUN", "tensor_results": [], "continuity_results": [], "cache_results": [], "failures": []}
    c_metadata: dict[str, Any] = {}
    reference_metadata: dict[str, Any] = {}
    artifact = {"path": str(model), "expected_bytes": EXPECTED_BYTES, "expected_sha256": EXPECTED_SHA256, "bytes": None, "sha256": None}
    binary: Path | None = None
    reference_binary: Path | None = None
    compiler_path = shutil.which("clang")
    environment = {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler_path}

    try:
        source_hashes = source_inventory()
        if compiler_path is None:
            raise RunnerError("clang is unavailable")
        if not args.apparatus_only and (not model.is_file() or model.stat().st_size != EXPECTED_BYTES):
            raise RunnerError("frozen accepted artifact is absent or has the wrong byte size")
        if git_value(["git", "rev-parse", "HEAD"], PINNED_LLAMA) != LLAMA_COMMIT or git_value(["git", "status", "--porcelain"], PINNED_LLAMA):
            raise RunnerError("llama.cpp checkout is not the clean pinned revision")

        clang_version = run_command([compiler_path, "--version"], cwd=ROOT, output=output, label="clang_version", timeout=30)
        commands["clang_version"] = clang_version
        require_ok(clang_version, "clang --version")
        environment["clang_version"] = clang_version["stdout"]

        binary = output / "engine_rung2a.exe"
        compile_record = run_command([compiler_path, *COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], cwd=ROOT, output=output, label="compile_c", timeout=600)
        commands["compile_c"] = compile_record
        require_ok(compile_record, "C engine build")
        if not binary.is_file():
            raise RunnerError("C engine build produced no binary")

        reference_build_dir = output / "reference_build"
        build_record = run_command([sys.executable, "-B", str(REFERENCE_BUILDER), "--build-dir", str(reference_build_dir), "--config", "Release"], cwd=ROOT, output=output, label="build_reference", timeout=3600)
        commands["build_reference"] = build_record
        require_ok(build_record, "pinned reference build")
        build_lines = [line.strip() for line in build_record["stdout"].splitlines() if line.strip()]
        if not build_lines:
            raise RunnerError("reference builder did not report its binary")
        reference_binary = Path(build_lines[-1]).resolve(strict=True)
        reference_binary.relative_to(reference_build_dir.resolve(strict=True))

        tests = run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], cwd=ROOT, output=output, label="model_free_unittests", timeout=3600)
        commands["model_free_unittests"] = tests
        require_ok(tests, "model-free regression suite")
        c_selftest = run_command([str(binary), "--strat01-gguf-rung2a-selftest"], cwd=ROOT, output=output, label="c_rung2a_selftest", timeout=300)
        commands["c_rung2a_selftest"] = c_selftest
        require_ok(c_selftest, "C rung-2A selftest")
        legacy_selftest = run_command([str(binary), "--kselftest"], cwd=ROOT, output=output, label="legacy_kselftest", timeout=300)
        commands["legacy_kselftest"] = legacy_selftest
        require_ok(legacy_selftest, "legacy kselftest")
        reference_selftest = run_command([str(reference_binary), "--self-test"], cwd=ROOT, output=output, label="reference_selftest", timeout=300)
        commands["reference_selftest"] = reference_selftest
        require_ok(reference_selftest, "reference selftest")

        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            artifact_sha = sha256_file(model)
            artifact.update({"bytes": model.stat().st_size, "sha256": artifact_sha})
            if artifact_sha != EXPECTED_SHA256:
                raise RunnerError("frozen accepted artifact SHA-256 mismatch")

            c_root = output / "c_engine"
            c_root.mkdir()
            c_run = run_command([str(binary), "--strat01-gguf-rung2a", str(model), "--out-dir", str(c_root)], cwd=ROOT, output=output, label="accepted_artifact_c_engine", timeout=21_600)
            commands["accepted_artifact_c_engine"] = c_run
            require_ok(c_run, "accepted-artifact C engine execution")
            source_hashes = source_inventory()
            c_tensors, c_metadata, c_cache = validate_c_outputs(c_root, source_hashes, model)

            reference_root = output / "pinned_reference"
            reference_run = run_command([str(reference_binary), "--model", str(model), "--out-dir", str(reference_root), "--all"], cwd=ROOT, output=output, label="accepted_artifact_pinned_reference", timeout=21_600)
            commands["accepted_artifact_pinned_reference"] = reference_run
            require_ok(reference_run, "accepted-artifact pinned reference execution")
            ref_tensors, reference_metadata, ref_cache = validate_reference_outputs(reference_root, model)

            adjudication = adjudicate(c_tensors, ref_tensors, c_cache, ref_cache)
            status = adjudication["status"]
    except RunnerError as exc:
        status = exc.status
        errors.append(str(exc))
    except Exception as exc:
        status = "VOID_ENGINE_RUNG2A"
        errors.append(f"unexpected {type(exc).__name__}: {exc}")

    finished_utc = utc_now()
    provenance = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "seconds": time.perf_counter() - started,
        "git_head": git_value(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": git_value(["git", "status", "--porcelain"]),
        "source_hashes": source_hashes,
        "artifact": artifact,
        "environment": environment,
        "binaries": {
            "c_engine": {"path": str(binary) if binary else None, "sha256": sha256_file(binary) if binary and binary.is_file() else None},
            "pinned_reference": {"path": str(reference_binary) if reference_binary else None, "sha256": sha256_file(reference_binary) if reference_binary and reference_binary.is_file() else None},
        },
        "commands": commands,
    }
    final_record = {
        "schema": "strat01_gigachat_engine_rung2a_adjudication_v1",
        "status": status,
        "scope": "paired block-0 MLA semantic parity through phase60/engine.c",
        "errors": errors,
        "adjudication": adjudication,
        "c_metadata": c_metadata,
        "reference_metadata": reference_metadata,
        "non_claims": NON_CLAIMS,
        "provenance": provenance,
    }
    manifest = {
        "schema": "strat01_gigachat_engine_rung2a_run_manifest_v1",
        "status": status,
        "gate_summary": {
            "apparatus_and_regressions_pass": all(commands.get(key, {}).get("returncode") == 0 for key in ("compile_c", "build_reference", "model_free_unittests", "c_rung2a_selftest", "legacy_kselftest", "reference_selftest")),
            "c_output_validated": bool(c_metadata),
            "reference_output_validated": bool(reference_metadata),
            "all_numerical_gates_pass": adjudication.get("status") == "PASS_ENGINE_RUNG2A",
        },
        "errors": errors,
        "provenance": provenance,
    }
    write_json(output / "adjudication.json", final_record)
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {"PASS_ENGINE_RUNG2A", "APPARATUS_READY_NO_DONOR_EXECUTION"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
