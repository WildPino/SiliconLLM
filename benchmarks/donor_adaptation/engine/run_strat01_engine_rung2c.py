#!/usr/bin/env python3
"""Build, execute once per producer, and adjudicate GigaChat engine Rung 2C."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2b as base

HERE = Path(__file__).resolve().parent
ENGINE = base.ENGINE
RUNG2A_HEADER = base.RUNG2A_HEADER
RUNG2B_HEADER = base.RUNG2B_HEADER
RUNG2C_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_rung2c.h"
REFERENCE_SOURCE = HERE / "strat01_engine_rung2a_reference.cpp"
REFERENCE_BUILDER = HERE / "build_strat01_engine_rung2c_reference.py"
TESTS = HERE / "test_strat01_engine_rung2c.py"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_PROTOCOL_20260923.md"
MODEL = base.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_rung2c_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_rung2c_apparatus_20260923"
BLOCK0_RUN = HERE / "results/strat01_gigachat_engine_block0_production_integration_20260922"
BLOCK0_ADJ_SHA = "58f661d9461f3dc17b2a52f830dcc9f36423621528941d475e23ee2aed8e0bf2"
BLOCK0_MANIFEST_SHA = "7b8de95a3766e4f76bde37f4038ece2986b9067ad9ac7a0e4cc11855a3062497"
C_START_SHA = "258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44"
REFERENCE_START_SHA = "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"
REFERENCE_SCHEMA = "strat01_engine_rung2c_reference_manifest_v1"
GRAPH_MARKER = "STRAT01_RUNG2C_GRAPH_COMPLETE arm="

SHAPES: dict[str, list[int]] = {
    "l_out-0": [1536, 8],
    "attn_norm-1": [1536, 8], "q-1": [192, 32, 8],
    "kv_cmpr_pe-1": [576, 8], "k_pe-1": [64, 1, 8], "kv_cmpr-1": [512, 8],
    "q_pe-1": [64, 32, 8], "q_nope_absorbed_perm-1": [512, 32, 8],
    "Qcur-1": [576, 32, 8], "Kcur-1": [576, 1, 8], "Vcur-1": [512, 1, 8],
    "kqv_out-1": [6144, 8], "ffn_inp-1": [1536, 8], "ffn_norm-1": [1536, 8],
    "ffn_moe_logits-1": [64, 8], "ffn_moe_probs-1": [64, 8],
    "ffn_moe_probs_biased-1": [64, 8], "ffn_moe_topk-1": [4, 8],
    "ffn_moe_weights-1": [4, 8], "ffn_moe_weights_norm-1": [4, 8],
    "ffn_moe_up-1": [1280, 4, 8], "ffn_moe_gate-1": [1280, 4, 8],
    "ffn_moe_swiglu-1": [1280, 4, 8], "ffn_moe_down-1": [1536, 4, 8],
    "ffn_moe_weighted-1": [1536, 4, 8], "ffn_moe_out-1": [1536, 8],
    "ffn_up-1": [1280, 8], "ffn_gate-1": [1280, 8], "ffn_swiglu-1": [1280, 8],
    "ffn_shexp-1": [1536, 8], "ffn_out-1": [1536, 8], "l_out-1": [1536, 8],
}
I32_NAMES = {"ffn_moe_topk-1"}
TERMINALS = {"ffn_inp-1", "l_out-1"}
EXPECTED_OPS = {
    "l_out-0": "ADD", "attn_norm-1": "MUL", "q-1": "RESHAPE", "kv_cmpr_pe-1": "MUL_MAT",
    "k_pe-1": "ROPE", "kv_cmpr-1": "MUL", "q_pe-1": "ROPE", "q_nope_absorbed_perm-1": "PERMUTE",
    "Qcur-1": "CONCAT", "Kcur-1": "CONCAT", "Vcur-1": "RESHAPE", "kqv_out-1": "CONT",
    "ffn_inp-1": "ADD", "ffn_norm-1": "MUL", "ffn_moe_logits-1": "MUL_MAT",
    "ffn_moe_probs-1": "UNARY", "ffn_moe_probs_biased-1": "ADD", "ffn_moe_topk-1": "VIEW",
    "ffn_moe_weights-1": "GET_ROWS", "ffn_moe_weights_norm-1": "DIV", "ffn_moe_up-1": "MUL_MAT_ID",
    "ffn_moe_gate-1": "MUL_MAT_ID", "ffn_moe_swiglu-1": "GLU", "ffn_moe_down-1": "MUL_MAT_ID",
    "ffn_moe_weighted-1": "MUL", "ffn_moe_out-1": "ADD", "ffn_up-1": "MUL_MAT",
    "ffn_gate-1": "MUL_MAT", "ffn_swiglu-1": "GLU", "ffn_shexp-1": "MUL_MAT",
    "ffn_out-1": "ADD", "l_out-1": "ADD",
}
PAYLOAD_ORDER = {
    name: ("token,slot,feature" if len(shape) == 3 and "moe_" in name
           else "token,head,feature" if len(shape) == 3
           else "token,slot" if shape[0] == 4
           else "token,expert" if shape[0] == 64
           else "token,feature")
    for name, shape in SHAPES.items()
}
EXPECTED_C_CONFIG = (
    "reference=llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2;cpu_threads=1;"
    "n_ctx=8;n_batch=8;n_ubatch=8;flash_attn=false;offload_kqv=false;"
    "type_k=f16;type_v=f16-no-allocation-mla;f16_dot=pinned-generic-f64;cache=layers0-1-slot-576-f16-k-only-latent512-rope64;"
    "tokens=1,72,14,14129,14,2135,1512,2015;positions=0,1,2,3,4,5,6,7;"
    "rms_eps=1e-6;rope=deepseek2-normal-yarn;rope_base=100000;rope_factor=64;"
    "rope_orig_ctx=4096;beta_fast=32;beta_slow=1;mscale=1;mscale_all_dim=1;"
    "rms_accum=double;kb=q5_0xq8_0;"
    "block0=dense-accepted;block1=mla-sigmoid-bias-select-top4-unbiased-normalized-q4k-q6k-shared-residual;"
    "build=clang-c11-O3-mavx2-mfma-no-fast-math;fp_contract=off-c11-pragma;"
    "payload=f32le-or-i32le-token-major;adjudication=external-reference-only"
)


class RunnerError(RuntimeError):
    pass


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {"runner": Path(__file__).resolve(), "engine": ENGINE, "rung2a_header": RUNG2A_HEADER,
             "rung2b_header": RUNG2B_HEADER, "rung2c_header": RUNG2C_HEADER,
             "reference_source": REFERENCE_SOURCE, "reference_builder": REFERENCE_BUILDER,
             "tests": TESTS, "protocol": PROTOCOL}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunnerError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def validate_predecessor() -> dict[str, Any]:
    adj_path, manifest_path = BLOCK0_RUN / "adjudication.json", BLOCK0_RUN / "run_manifest.json"
    if base.sha256_file(adj_path) != BLOCK0_ADJ_SHA or base.sha256_file(manifest_path) != BLOCK0_MANIFEST_SHA:
        raise RunnerError("block-0 predecessor hash mismatch")
    adj = read_json(adj_path, "block-0 adjudication")
    if adj.get("status") != "PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION" or adj.get("errors") != [] or adj.get("donor_graph_executions") != 1 or adj.get("reference_graph_executions") != 0:
        raise RunnerError("block-0 predecessor state mismatch")
    return {"adjudication_sha256": BLOCK0_ADJ_SHA, "run_manifest_sha256": BLOCK0_MANIFEST_SHA}


def source_controls() -> dict[str, bool]:
    r2a, r2b, r2c = (path.read_text(encoding="utf-8") for path in (RUNG2A_HEADER, RUNG2B_HEADER, RUNG2C_HEADER))
    ref = REFERENCE_SOURCE.read_text(encoding="utf-8")
    return {
        "q4_kernel_reused": "strat01_r2a_matmul_batch(path" in r2c and "static float strat01_q4k_q8k_dot(" not in r2c,
        "q5_kernel_reused": "strat01_r2a_kb_batch(path" in r2c and "static float strat01_kb_q5q8_dot(" not in r2c,
        "q6_kernel_reused": "strat01_q6k_q8k_dot(raw" in r2c and r2b.count("static float strat01_q6k_q8k_dot(") == 1 and "static float strat01_q6k_q8k_dot(" not in r2c,
        "rmsnorm_reused": r2c.count("strat01_r2a_rmsnorm_pinned(") >= 2 and r2a.count("static void strat01_r2a_rmsnorm_pinned(") == 1,
        "reference_callbacks_not_python_recompute": "ffn_moe_topk-1" in ref and "tensor_i32" in ref and "write_i32_payload" in ref,
        "reference_cache_f16_witness": "write_cache_f16_roundtrip_payload" in ref,
    }


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise RunnerError("Rung-2C implementation/protocol differs from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"Rung-2C source is untracked: {path}")


def load_typed(path: Path, count: int, digest: str, kind: str, label: str) -> np.ndarray:
    if base.sha256_file(path) != digest:
        raise RunnerError(f"{label} SHA-256 mismatch")
    dtype = np.dtype("<i4") if kind == "I32" else np.dtype("<f4")
    values = np.fromfile(path, dtype=dtype)
    if values.size != count or (kind == "F32" and not bool(np.isfinite(values).all())):
        raise RunnerError(f"{label} count/finiteness mismatch")
    return values


def validate_cache_records(root: Path, records: Any, arm: str, label: str) -> dict[str, np.ndarray]:
    if not isinstance(records, list) or len(records) != 2:
        raise RunnerError(f"{label} cache layer set mismatch")
    result: dict[str, np.ndarray] = {}
    for layer, record in enumerate(records):
        required = {"layer", "storage", "row_length", "no_separate_v_cache", "final"} | ({"prefix7"} if arm == "cached7p1" else set())
        if not isinstance(record, dict) or set(record) != required or record.get("layer") != layer or record.get("storage") != "F16" or record.get("row_length") != 576 or record.get("no_separate_v_cache") is not True:
            raise RunnerError(f"{label} layer-{layer} cache contract mismatch")
        for phase, rows in (("final", 8), *(([("prefix7", 7)] if arm == "cached7p1" else []))):
            item = record[phase]
            expected = list(range(rows))
            if item.get("occupied_slots") != expected or item.get("absolute_positions") != expected or item.get("payload_type") != "dequantized-F32LE":
                raise RunnerError(f"{label} layer-{layer}/{phase} occupancy mismatch")
            path = base.contained_file(root, item["path"], rows * 576 * 4, f"{label} cache")
            result[f"layer{layer}/{phase}"] = load_typed(path, rows * 576, item["sha256"], "F32", f"{label} cache")
    return result


def validate_c(c_root: Path, sources: dict[str, dict[str, str]], model: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report = read_json(c_root / "strat01_rung2c.json", "C report")
    required = {"command", "c_state", "self_certifies_pass", "input_path", "byte_size", "sha256", "reference_revision", "CONFIG", "compiler_family", "compiler_embedded_version", "compiler_resolved_path_and_full_version", "engine_source_sha256", "rung2c_source_sha256", "token_ids", "positions", "arms", "timing_or_rate_claim"}
    if not isinstance(report, dict) or set(report) != required or report["command"] != "--strat01-gguf-rung2c" or report["c_state"] != "ENGINE_RUNG2C_OUTPUT_READY_PENDING_REFERENCE" or report["self_certifies_pass"] is not False:
        raise RunnerError("C report schema/state mismatch")
    if Path(report["input_path"]).resolve(strict=True) != model.resolve(strict=True) or report["byte_size"] != base.EXPECTED_BYTES or report["sha256"] != base.EXPECTED_SHA256:
        raise RunnerError("C artifact identity mismatch")
    if report["reference_revision"] != f"llama.cpp {base.LLAMA_COMMIT}" or report["CONFIG"] != EXPECTED_C_CONFIG or report["compiler_family"] != "clang" or report["compiler_resolved_path_and_full_version"] != "EXTERNAL_RUNNER_REQUIRED":
        raise RunnerError("C reference/config/compiler mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["rung2c_source_sha256"] != sources["rung2c_header"]["sha256"] or report["token_ids"] != base.TOKENS or report["positions"] != base.POSITIONS or report["timing_or_rate_claim"] is not None:
        raise RunnerError("C source/input/non-speed contract mismatch")
    tensors: dict[str, np.ndarray] = {}; manifests: dict[str, Any] = {}; caches: dict[str, Any] = {}
    for arm in base.ARMS:
        manifest = read_json(c_root / f"{arm}_manifest.json", f"C {arm} manifest")
        if not isinstance(manifest, dict) or set(manifest) != {"arm", "payload_encoding", "shape_order", "tensors", "caches"} or manifest["arm"] != arm:
            raise RunnerError(f"C {arm} manifest schema mismatch")
        by_name = {item.get("name"): item for item in manifest["tensors"] if isinstance(item, dict)}
        if set(by_name) != set(SHAPES) or len(by_name) != len(manifest["tensors"]):
            raise RunnerError(f"C {arm} checkpoint set mismatch")
        for name, shape in SHAPES.items():
            item, kind, count = by_name[name], "I32" if name in I32_NAMES else "F32", math.prod(shape)
            if item.get("logical_shape") != shape or item.get("op") != EXPECTED_OPS[name] or item.get("type") != kind or item.get("payload_order") != PAYLOAD_ORDER[name] or item.get("byte_count") != 4 * count:
                raise RunnerError(f"C {arm}/{name} identity mismatch")
            path = base.contained_file(c_root, item["path"], 4 * count, f"C {arm}/{name}")
            tensors[f"{arm}/{name}"] = load_typed(path, count, item["sha256"], kind, f"C {arm}/{name}")
        caches[arm] = validate_cache_records(c_root, manifest["caches"], arm, f"C {arm}")
        manifests[arm] = manifest
    return tensors, {"report": report, "manifests": manifests, "caches": caches}


def validate_reference(ref_root: Path, model: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = read_json(ref_root / "manifest.json", "reference root manifest")
    required = {"schema", "state", "llama_cpp_commit", "model", "config", "fixed_tokens", "fixed_positions", "arms"}
    if not isinstance(root, dict) or set(root) != required or root["schema"] != REFERENCE_SCHEMA or root["state"] != "REFERENCE_TRACE_READY_PENDING_C_ENGINE" or root["llama_cpp_commit"] != base.LLAMA_COMMIT:
        raise RunnerError("reference root schema/state mismatch")
    model_record = root["model"]
    if Path(model_record.get("path", "")).resolve(strict=True) != model.resolve(strict=True) or model_record.get("bytes") != base.EXPECTED_BYTES or model_record.get("sha256") != base.EXPECTED_SHA256 or root["config"] != base.EXPECTED_REFERENCE_CONFIG or root["fixed_tokens"] != base.TOKENS or root["fixed_positions"] != base.POSITIONS:
        raise RunnerError("reference identity/config mismatch")
    tensors: dict[str, np.ndarray] = {}; manifests: dict[str, Any] = {}; caches: dict[str, Any] = {}
    for arm in base.ARMS:
        manifest = read_json(ref_root / arm / "manifest.json", f"reference {arm} manifest")
        if not isinstance(manifest, dict) or set(manifest) != {"schema", "arm", "logical_cache_contract", "callback_records", "logical_selection", "payloads"} or manifest["schema"] != REFERENCE_SCHEMA or manifest["arm"] != arm:
            raise RunnerError(f"reference {arm} manifest schema mismatch")
        selection = manifest["logical_selection"]
        payloads = {(item.get("logical"), item.get("kind")): item for item in manifest["payloads"] if isinstance(item, dict)}
        for name, shape in SHAPES.items():
            selected, kind, count = selection.get(name), "I32" if name in I32_NAMES else "F32", math.prod(shape)
            if not isinstance(selected, dict) or selected.get("logical_shape") != shape or selected.get("composition") != ("single_prefill8_callback" if arm == "prefill8" else "prefix7_then_final1"):
                raise RunnerError(f"reference {arm}/{name} logical selection mismatch")
            sources = [selected.get("source")] if arm == "prefill8" else [selected.get("prefix_source"), selected.get("final_source")]
            for source, ntok in zip(sources, [8] if arm == "prefill8" else [7, 1]):
                if not isinstance(source, dict) or source.get("name") != name or source.get("op") != EXPECTED_OPS[name] or source.get("shape") != shape[:-1] + [ntok] or str(source.get("type", "")).upper() != kind:
                    raise RunnerError(f"reference {arm}/{name} callback identity mismatch")
            item = payloads.get((name, "full"))
            if not isinstance(item, dict) or item.get("byte_count") != 4 * count:
                raise RunnerError(f"reference {arm}/{name} full payload missing")
            path = base.contained_file(ref_root, item["path"], 4 * count, f"reference {arm}/{name}")
            tensors[f"{arm}/{name}"] = load_typed(path, count, item["sha256"], kind, f"reference {arm}/{name}")
        arm_cache: dict[str, np.ndarray] = {}
        for layer in (0, 1):
            logical = f"Kcur-{layer}"
            for phase, kind_name, rows in (("final", "cache_f16_roundtrip", 8), *(([("prefix7", "prefix_cache_f16_roundtrip", 7)] if arm == "cached7p1" else []))):
                item = payloads.get((logical, kind_name))
                if not isinstance(item, dict) or item.get("byte_count") != rows * 576 * 4:
                    raise RunnerError(f"reference {arm}/{logical}/{phase} cache witness missing")
                path = base.contained_file(ref_root, item["path"], rows * 576 * 4, f"reference cache {logical}/{phase}")
                arm_cache[f"layer{layer}/{phase}"] = load_typed(path, rows * 576, item["sha256"], "F32", f"reference cache {logical}/{phase}")
        caches[arm] = arm_cache; manifests[arm] = manifest
    return tensors, {"root": root, "manifests": manifests, "caches": caches}


def negative_controls(reference: dict[str, np.ndarray]) -> dict[str, Any]:
    pfx = "prefill8/"
    ids = reference[pfx + "ffn_moe_topk-1"].reshape(8, 4)
    probs = reference[pfx + "ffn_moe_probs-1"].reshape(8, 64)
    biased = reference[pfx + "ffn_moe_probs_biased-1"].reshape(8, 64)
    logits = reference[pfx + "ffn_moe_logits-1"].reshape(8, 64).astype(np.float64)
    ref_norm = reference[pfx + "ffn_moe_weights_norm-1"].reshape(8, 4)
    selected_biased = np.take_along_axis(biased, ids, axis=1); selected_biased /= selected_biased.sum(axis=1, keepdims=True)
    softmax = np.exp(logits - logits.max(axis=1, keepdims=True)); softmax /= softmax.sum(axis=1, keepdims=True)
    selected_softmax = np.take_along_axis(softmax, ids, axis=1); selected_softmax /= selected_softmax.sum(axis=1, keepdims=True)
    gate = reference[pfx + "ffn_moe_gate-1"].astype(np.float64); up = reference[pfx + "ffn_moe_up-1"].astype(np.float64)
    controls = {
        "biased_route_weights": base.judged(selected_biased.astype(np.float32).ravel(), ref_norm.ravel(), base.GENERAL_LIMITS),
        "softmax_router": base.judged(selected_softmax.astype(np.float32).ravel(), ref_norm.ravel(), base.GENERAL_LIMITS),
        "omit_weight_normalization": base.judged(reference[pfx + "ffn_moe_weights-1"], reference[pfx + "ffn_moe_weights_norm-1"], base.GENERAL_LIMITS),
        "permute_ids_only": {"pass": bool(np.array_equal(np.roll(ids, 1, axis=1), ids))},
        "wrong_expert_slice": base.judged(np.roll(reference[pfx + "ffn_moe_up-1"], 1280), reference[pfx + "ffn_moe_up-1"], base.GENERAL_LIMITS),
        "swap_routed_gate_up": base.judged((up / (1.0 + np.exp(-up)) * gate).astype(np.float32), reference[pfx + "ffn_moe_swiglu-1"], base.GENERAL_LIMITS),
        "omit_shared_expert": base.judged(reference[pfx + "ffn_moe_out-1"], reference[pfx + "ffn_out-1"], base.GENERAL_LIMITS),
        "omit_routed_shared_add": base.judged(reference[pfx + "ffn_shexp-1"], reference[pfx + "ffn_out-1"], base.GENERAL_LIMITS),
        "omit_final_residual": base.judged(reference[pfx + "ffn_out-1"], reference[pfx + "l_out-1"], base.TERMINAL_LIMITS),
        "mutate_start_or_cache_index": {"pass": C_START_SHA == ("0" + C_START_SHA[1:]) or (1, 7) == (0, 7)},
    }
    return controls


def adjudicate(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray], c_meta: dict[str, Any], r_meta: dict[str, Any], controls: dict[str, bool]) -> dict[str, Any]:
    failures: list[str] = []; checkpoints: list[dict[str, Any]] = []; continuity: list[dict[str, Any]] = []; cache_results: list[dict[str, Any]] = []
    for arm in base.ARMS:
        for name in SHAPES:
            if name in I32_NAMES:
                passed = bool(np.array_equal(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"]))
                result = {"arm": arm, "checkpoint": name, "exact": passed, "pass": passed}
            else:
                limits = base.TERMINAL_LIMITS if name in TERMINALS else base.GENERAL_LIMITS
                result = base.judged(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"], limits); result.update({"arm": arm, "checkpoint": name})
            checkpoints.append(result)
            if not result["pass"]: failures.append(f"checkpoint/{arm}/{name}")
    for implementation, values in (("c_engine", candidate), ("pinned_reference", reference)):
        for name, shape in SHAPES.items():
            a, b = base.token7(values[f"prefill8/{name}"], shape), base.token7(values[f"cached7p1/{name}"], shape)
            if name in I32_NAMES: result = {"implementation": implementation, "checkpoint": name, "exact": bool(np.array_equal(a, b)), "pass": bool(np.array_equal(a, b))}
            else: result = base.judged(a, b, base.CONTINUITY_LIMITS); result.update({"implementation": implementation, "checkpoint": name})
            continuity.append(result)
            if not result["pass"]: failures.append(f"continuity/{implementation}/{name}")
    for arm in base.ARMS:
        for key, c_values in c_meta["caches"][arm].items():
            result = base.judged(c_values, r_meta["caches"][arm][key], base.GENERAL_LIMITS); result.update({"arm": arm, "cache": key})
            cache_results.append(result)
            if not result["pass"]: failures.append(f"cache/{arm}/{key}")
    start_hashes = {arm: {"c_engine": c_meta["manifests"][arm]["tensors"][0]["sha256"], "pinned_reference": next(item["sha256"] for item in r_meta["manifests"][arm]["payloads"] if item.get("logical") == "l_out-0" and item.get("kind") == "full")} for arm in base.ARMS}
    for arm, hashes in start_hashes.items():
        if hashes != {"c_engine": C_START_SHA, "pinned_reference": REFERENCE_START_SHA}: failures.append(f"start_state/{arm}")
    neg = negative_controls(reference); neg_pass = all(not item["pass"] for item in neg.values())
    if not neg_pass: failures.append("causal_negative_controls")
    if not all(controls.values()): failures.append("source_controls")
    return {"status": "PASS_ENGINE_RUNG2C" if not failures else "FAIL_ENGINE_RUNG2C", "failures": failures,
            "start_hashes": start_hashes, "checkpoint_results": checkpoints, "continuity_results": continuity,
            "cache_results": cache_results, "negative_controls": neg, "negative_controls_pass": neg_pass, "source_controls": controls}


def completed_graph_count(record: dict[str, Any]) -> int:
    text = f"{record.get('stdout', '')}\n{record.get('stderr', '')}"
    arms = re.findall(r"^STRAT01_RUNG2C_GRAPH_COMPLETE arm=(prefill8|cached7p1)$", text, flags=re.MULTILINE)
    if len(arms) != len(set(arms)):
        raise RunnerError("duplicate Rung-2C graph-completion marker")
    return len(arms)


def metadata_for_record(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep validated manifests, but not their in-memory ndarray cache copies."""
    return {key: value for key, value in metadata.items() if key != "caches"}


def recover_existing(source: Path, output: Path, model: Path, execution_head: str) -> int:
    source = source.resolve(strict=True); output = output.resolve(); model = model.resolve(strict=True)
    if not source.is_dir() or (source / "adjudication.json").exists() or (source / "run_manifest.json").exists():
        raise SystemExit("recovery requires an unfinalized Rung-2C source directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory recovery output path: {output}")
    resolved_head = base.git_value(["git", "rev-parse", f"{execution_head}^{{commit}}"])
    if resolved_head != execution_head:
        raise SystemExit("recovery execution HEAD is not an exact commit")
    if model.stat().st_size != base.EXPECTED_BYTES or base.sha256_file(model) != base.EXPECTED_SHA256:
        raise SystemExit("recovery model identity mismatch")

    producer_records: dict[str, Any] = {}
    for producer, label in (("pinned_reference", "accepted_artifact_pinned_reference"), ("c_engine", "accepted_artifact_c_engine")):
        stdout_path, stderr_path = source / f"{label}.stdout.log", source / f"{label}.stderr.log"
        if not stdout_path.is_file() or not stderr_path.is_file():
            raise SystemExit(f"recovery is missing {producer} command logs")
        record = {"stdout": stdout_path.read_text(encoding="utf-8"), "stderr": stderr_path.read_text(encoding="utf-8")}
        completed = completed_graph_count(record)
        if completed != 2:
            raise SystemExit(f"recovery {producer} graph count is {completed}, expected 2")
        producer_records[producer] = {
            "invocations": 1, "graph_executions": completed,
            "stdout": {"path": str(stdout_path), "bytes": stdout_path.stat().st_size, "sha256": base.sha256_file(stdout_path)},
            "stderr": {"path": str(stderr_path), "bytes": stderr_path.stat().st_size, "sha256": base.sha256_file(stderr_path)},
        }

    sources = source_inventory(); controls = source_controls()
    if not all(controls.values()):
        raise SystemExit("recovery source controls failed")
    reference, r_meta = validate_reference(source / "pinned_reference", model)
    candidate, c_meta = validate_c(source / "c_engine", sources, model)
    adjudication = adjudicate(candidate, reference, c_meta, r_meta, controls)

    key_paths = [
        source / "pinned_reference" / "manifest.json",
        *(source / "pinned_reference" / arm / "manifest.json" for arm in base.ARMS),
        source / "c_engine" / "strat01_rung2c.json",
        *(source / "c_engine" / f"{arm}_manifest.json" for arm in base.ARMS),
    ]
    key_files: dict[str, Any] = {}
    for path in key_paths:
        if not path.is_file():
            raise SystemExit(f"recovery key file is missing: {path}")
        relative = path.relative_to(source).as_posix()
        key_files[relative] = {"bytes": path.stat().st_size, "sha256": base.sha256_file(path)}

    output.mkdir(parents=True, exist_ok=True)
    recovered_utc = datetime.now(timezone.utc).isoformat()
    provenance = {
        "recovery_reason": "both producers completed and validated; original runner failed only while JSON-encoding ndarray cache copies",
        "source_run_directory": str(source), "execution_git_head": execution_head,
        "recovery_git_head": base.git_value(["git", "rev-parse", "HEAD"]), "recovered_utc": recovered_utc,
        "artifact": {"path": str(model), "bytes": model.stat().st_size, "sha256": base.EXPECTED_SHA256},
        "source_hashes_at_recovery": sources, "producer_records": producer_records, "key_files": key_files,
    }
    record = {
        "schema": "strat01_gigachat_engine_rung2c_recovered_adjudication_v1", "status": adjudication["status"], "errors": [],
        "reference_producer_invocations": 1, "reference_graph_executions": 2,
        "donor_producer_invocations": 1, "donor_graph_executions": 2,
        "adjudication": adjudication, "c_metadata": metadata_for_record(c_meta),
        "reference_metadata": metadata_for_record(r_meta), "recovery": provenance,
        "non_claims": ["later MoE layers", "tokenizer/logits/generation", "C-path language-model quality", "RAM", "rate", "SPEED_LEDGER"],
    }
    manifest = {
        "schema": "strat01_gigachat_engine_rung2c_recovered_run_manifest_v1", "status": adjudication["status"], "errors": [],
        "reference_producer_invocations": 1, "reference_graph_executions": 2,
        "donor_producer_invocations": 1, "donor_graph_executions": 2, "recovery": provenance,
    }
    json.dumps(record); json.dumps(manifest)
    base.write_json(output / "adjudication.json", record); base.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": adjudication["status"], "output": str(output), "recovered_without_producer_execution": True}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument("--execution-head")
    args = parser.parse_args(); model = args.model.resolve()
    if args.recover_from:
        if args.apparatus_only or not args.output_dir or not args.execution_head:
            raise SystemExit("--recover-from requires --output-dir and --execution-head, and forbids --apparatus-only")
        return recover_existing(args.recover_from, args.output_dir, model, args.execution_head)
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter(); status = "VOID_ENGINE_RUNG2C"; errors: list[str] = []
    commands: dict[str, Any] = {}; sources: dict[str, Any] = {}; predecessor: dict[str, Any] = {}; c_meta: dict[str, Any] = {}; r_meta: dict[str, Any] = {}; controls: dict[str, bool] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}
    compiler = shutil.which("clang"); binary: Path | None = None; reference_binary: Path | None = None
    reference_producer_invocations = 0; donor_producer_invocations = 0; reference_graph_executions = 0; donor_graph_executions = 0
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_BYTES, "expected_sha256": base.EXPECTED_SHA256, "bytes": None, "sha256": None}
    try:
        sources = source_inventory(); predecessor = validate_predecessor(); controls = source_controls()
        if not all(controls.values()): raise RunnerError("Rung-2C source controls failed")
        if not compiler: raise RunnerError("clang is unavailable")
        if base.git_value(["git", "rev-parse", "HEAD"], base.PINNED_LLAMA) != base.LLAMA_COMMIT or base.git_value(["git", "status", "--porcelain"], base.PINNED_LLAMA): raise RunnerError("llama.cpp checkout is not the clean pinned revision")
        if not model.is_file() or model.stat().st_size != base.EXPECTED_BYTES: raise RunnerError("accepted artifact is absent or has wrong size")
        artifact.update({"bytes": model.stat().st_size, "sha256": base.sha256_file(model)})
        if artifact["sha256"] != base.EXPECTED_SHA256: raise RunnerError("accepted artifact SHA-256 mismatch")
        if not args.apparatus_only: clean_sources_at_head(sources)
        commands["clang_version"] = base.run_command([compiler, "--version"], output=output, label="clang_version", timeout=30); base.require_ok(commands["clang_version"], "clang version")
        binary = output / "engine_rung2c.exe"
        commands["compile_c"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile_c", timeout=600); base.require_ok(commands["compile_c"], "C build")
        for label, flag in (("rung2c_selftest", "--strat01-gguf-rung2c-selftest"), ("combined_selftest", "--strat01-combined-rms-q5q8-selftest"), ("legacy_selftest", "--kselftest")):
            commands[label] = base.run_command([str(binary), flag], output=output, label=label, timeout=300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_engine_rung2c"], output=output, label="python_tests", timeout=300); base.require_ok(commands["python_tests"], "Rung-2C Python tests")
        build_dir = output / "reference_build"
        commands["build_reference"] = base.run_command([sys.executable, "-B", str(REFERENCE_BUILDER), "--build-dir", str(build_dir), "--config", "Release"], output=output, label="build_reference", timeout=3600); base.require_ok(commands["build_reference"], "reference build")
        lines = [line.strip() for line in commands["build_reference"]["stdout"].splitlines() if line.strip()]
        if not lines: raise RunnerError("reference builder reported no binary")
        reference_binary = Path(lines[-1]).resolve(strict=True); reference_binary.relative_to(build_dir.resolve(strict=True))
        commands["reference_selftest"] = base.run_command([str(reference_binary), "--self-test"], output=output, label="reference_selftest", timeout=300); base.require_ok(commands["reference_selftest"], "reference selftest")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            ref_root = output / "pinned_reference"; reference_producer_invocations = 1
            commands["accepted_artifact_pinned_reference"] = base.run_command([str(reference_binary), "--model", str(model), "--out-dir", str(ref_root), "--all"], output=output, label="accepted_artifact_pinned_reference", timeout=21600)
            reference_graph_executions = completed_graph_count(commands["accepted_artifact_pinned_reference"])
            base.require_ok(commands["accepted_artifact_pinned_reference"], "pinned reference producer")
            if reference_graph_executions != 2: raise RunnerError("pinned reference producer omitted a graph-completion marker")
            reference, r_meta = validate_reference(ref_root, model)
            for arm in base.ARMS:
                item = next(x for x in r_meta["manifests"][arm]["payloads"] if x.get("logical") == "l_out-0" and x.get("kind") == "full")
                if item["sha256"] != REFERENCE_START_SHA: raise RunnerError(f"reference block-0 start hash mismatch: {arm}")
            c_root = output / "c_engine"; c_root.mkdir(); donor_producer_invocations = 1
            commands["accepted_artifact_c_engine"] = base.run_command([str(binary), "--strat01-gguf-rung2c", str(model), "--out-dir", str(c_root)], output=output, label="accepted_artifact_c_engine", timeout=21600)
            donor_graph_executions = completed_graph_count(commands["accepted_artifact_c_engine"])
            base.require_ok(commands["accepted_artifact_c_engine"], "C producer")
            if donor_graph_executions != 2: raise RunnerError("C producer omitted a graph-completion marker")
            sources = source_inventory(); candidate, c_meta = validate_c(c_root, sources, model); adjudication = adjudicate(candidate, reference, c_meta, r_meta, controls); status = adjudication["status"]
    except (RunnerError, base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started,
                  "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": base.git_value(["git", "status", "--porcelain"]),
                  "source_hashes": sources, "artifact": artifact, "predecessor": predecessor,
                  "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler},
                  "binaries": {"c_engine": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "pinned_reference": {"path": str(reference_binary) if reference_binary else None, "sha256": base.sha256_file(reference_binary) if reference_binary and reference_binary.is_file() else None}}, "commands": commands}
    record = {"schema": "strat01_gigachat_engine_rung2c_adjudication_v1", "status": status, "errors": errors,
              "reference_producer_invocations": reference_producer_invocations, "donor_producer_invocations": donor_producer_invocations,
              "reference_graph_executions": reference_graph_executions, "donor_graph_executions": donor_graph_executions,
              "adjudication": adjudication, "c_metadata": metadata_for_record(c_meta), "reference_metadata": metadata_for_record(r_meta),
              "non_claims": ["later MoE layers", "tokenizer/logits/generation", "C-path language-model quality", "RAM", "rate", "SPEED_LEDGER"], "provenance": provenance}
    manifest = {"schema": "strat01_gigachat_engine_rung2c_run_manifest_v1", "status": status, "errors": errors,
                "reference_producer_invocations": reference_producer_invocations, "donor_producer_invocations": donor_producer_invocations,
                "reference_graph_executions": reference_graph_executions, "donor_graph_executions": donor_graph_executions, "provenance": provenance}
    base.write_json(output / "adjudication.json", record); base.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION", "PASS_ENGINE_RUNG2C", "FAIL_ENGINE_RUNG2C"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
