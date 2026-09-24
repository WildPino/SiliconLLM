#!/usr/bin/env python3
"""Qualify the frozen STRAT-01 layer-2 depth-extension apparatus."""
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

from benchmarks.donor_adaptation.engine import build_strat01_engine_rung2a_reference as reference_build
from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2c as r2c
from benchmarks.donor_adaptation.engine import run_strat01_post_f16_swiglu_production_integration as production
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as q4base

HERE = Path(__file__).resolve().parent
ENGINE = r2c.ENGINE
RUNG2A = r2c.RUNG2A_HEADER
RUNG2B = r2c.RUNG2B_HEADER
RUNG2C = r2c.RUNG2C_HEADER
RUNG2D = ROOT / "benchmarks/phase60/strat01_gguf_rung2d.h"
REFERENCE_SOURCE = HERE / "strat01_engine_rung2a_reference.cpp"
REFERENCE_BUILD = HERE / "build_strat01_engine_rung2a_reference.py"
REFERENCE_WRAPPER = HERE / "build_strat01_engine_rung2d_reference.py"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_PROTOCOL_20260924.md"
AUDIT = ROOT / "docs/research/donor_adaptation/audits/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_AUDIT_20260924.md"
TESTS = HERE / "test_strat01_engine_layer2_depth_extension.py"
PREDECESSOR = HERE / "results/strat01_gigachat_engine_post_f16_swiglu_production_integration_20260924/adjudication.json"
PREDECESSOR_SHA = "d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer2_depth_extension_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer2_depth_extension_apparatus_repair1_20260924"
PREVIOUS_REFERENCE = HERE / "results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference"
EXPECTED_COUNTS = {"mode": "pinned-generic-f64", "qk_invocations": 6_912, "value_invocations": 786_432}
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + path.stem for path in sorted(HERE.glob("test_strat01_*.py")))
SELFTESTS = (*production.SELFTESTS, "--strat01-gguf-rung2d-selftest")
REFERENCE_SCHEMA = "strat01_engine_rung2d_reference_manifest_v1"
GRAPH_MARKER = "STRAT01_RUNG2D_GRAPH_COMPLETE arm="
REFERENCE_GRAPH_MARKER = "STRAT01_RUNG2D_GRAPH_COMPLETE arm="
REFERENCE_START_SHA = "40d5a0f07fbb77c1ef73ca24f81cb35ea0df32457faa8d04d6c5cd33cd9f1d5f"
C_START_SHA = "9af8cec3f42781e9f4cac6af1ea13f6a63b751a5323201a8a18f91b3f7b7bbcb"
SHAPES = {"l_out-1": [1536, 8], **{name[:-1] + "2": shape for name, shape in r2c.SHAPES.items() if name != "l_out-0"}}
I32_NAMES = {"ffn_moe_topk-2"}
TERMINALS = {"ffn_inp-2", "l_out-2"}
EXPECTED_OPS = {"l_out-1": "ADD", **{name[:-1] + "2": op for name, op in r2c.EXPECTED_OPS.items() if name != "l_out-0"}}
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
    "type_k=f16;type_v=f16-no-allocation-mla;f16_dot=pinned-generic-f64;cache=layers0-2-slot-576-f16-k-only-latent512-rope64;"
    "tokens=1,72,14,14129,14,2135,1512,2015;positions=0,1,2,3,4,5,6,7;"
    "rms_eps=1e-6;rope=deepseek2-normal-yarn;rope_base=100000;rope_factor=64;"
    "rope_orig_ctx=4096;beta_fast=32;beta_slow=1;mscale=1;mscale_all_dim=1;"
    "rms_accum=double;kb=q5_0xq8_0;"
    "block0=dense-swiglu-sse2-nofma4;block1=accepted-rung2c-unchanged;"
    "block2=mla-sigmoid-bias-select-top4-unbiased-normalized-q4k-q6k-shared-residual;"
    "build=clang-c11-O3-mavx2-mfma-no-fast-math;fp_contract=off-c11-pragma;"
    "payload=f32le-or-i32le-token-major;adjudication=external-reference-only"
)


class DepthExtensionError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return r2c.base.sha256_file(path)


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthExtensionError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL, "audit": AUDIT,
        "engine": ENGINE, "rung2a": RUNG2A, "rung2b": RUNG2B, "rung2c": RUNG2C, "rung2d": RUNG2D,
        "reference_source": REFERENCE_SOURCE, "reference_build": REFERENCE_BUILD,
        "reference_wrapper": REFERENCE_WRAPPER, "rung2c_runner": Path(r2c.__file__).resolve(),
        "production_runner": Path(production.__file__).resolve(), "base_runner": Path(r2c.base.__file__).resolve(),
        "q4_parity_runner": Path(q4base.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DepthExtensionError("missing layer-2 source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def validate_predecessor() -> dict[str, str]:
    if not PREDECESSOR.is_file() or sha(PREDECESSOR) != PREDECESSOR_SHA:
        raise DepthExtensionError("production predecessor binding mismatch")
    record = read_json(PREDECESSOR, "production predecessor")
    if (record.get("status") != "PASS_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION" or
            record.get("errors") != [] or record.get("production_invocations") != 1 or
            record.get("donor_graph_executions") != 2 or record.get("reference_graph_executions") != 0):
        raise DepthExtensionError("production predecessor state mismatch")
    return {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8")
    r2c_text = RUNG2C.read_text(encoding="utf-8")
    r2d = RUNG2D.read_text(encoding="utf-8")
    ref = REFERENCE_SOURCE.read_text(encoding="utf-8")
    build = REFERENCE_BUILD.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    return {
        "separate_engine_command": '#include "strat01_gguf_rung2d.h"' in engine and "--strat01-gguf-rung2d" in engine,
        "exact_layer2_inventory": r2d.count('{"blk.2.') == 16 and '"blk.2.attn_norm.weight"' in r2d and '"blk.2.ffn_down_shexp.weight"' in r2d,
        "accepted_layer1_reused": "strat01_r2c_build_attention_range(path,a1" in r2d and "strat01_r2c_run_moe(path,m1" in r2d,
        "layer2_reuses_same_primitives": "strat01_r2c_build_attention_range(path,a2" in r2d and "strat01_r2c_run_moe(path,m2" in r2d,
        "layer1_swiglu_unchanged": "strat01_sse2_swiglu_compute" not in r2c_text and "expf(-g[i])" in r2c_text and "expf(-a->gate[i])" in r2c_text,
        "complete_layer2_surface": r2d.count('F("') == 31 and '"ffn_moe_topk-2"' in r2d and "STRAT01_R2C_DUMPS" in r2d,
        "three_layer_cache_schema": "layer<3U" in r2d and "cache_paths[6]" in r2d and "cache_sha[6][65]" in r2d,
        "exact_helper_totals": "3U*2304U==6912U" in r2d and "3U*262144U==786432U" in r2d,
        "separate_reference_variant": "STRAT01_RUNG2D" in ref and "strat01_engine_rung2d_reference_manifest_v1" in ref and "rung2d" in build,
        "reference_layer2_surface": ref.count('"Kcur-2"') >= 3 and '"l_out-2"' in ref and '\\"layers\\":[0,1,2]' in ref,
        "protocol_frozen": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol and PREDECESSOR_SHA in protocol,
        "audit_names_changed_coordinate": "LAYER2_IS_THE_NEXT_NON_DUPLICATE_DEPTH_BOUNDARY" in audit,
    }


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise DepthExtensionError("layer-2 sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise DepthExtensionError(f"untracked layer-2 source: {path}")


def validate_cache_records(root: Path, records: Any, arm: str, label: str) -> dict[str, np.ndarray]:
    if not isinstance(records, list) or len(records) != 3:
        raise DepthExtensionError(f"{label} cache layer set mismatch")
    result: dict[str, np.ndarray] = {}
    for layer, record in enumerate(records):
        required = {"layer", "storage", "row_length", "no_separate_v_cache", "final"} | ({"prefix7"} if arm == "cached7p1" else set())
        if (not isinstance(record, dict) or set(record) != required or record.get("layer") != layer or
                record.get("storage") != "F16" or record.get("row_length") != 576 or record.get("no_separate_v_cache") is not True):
            raise DepthExtensionError(f"{label} layer-{layer} cache contract mismatch")
        phases = (("final", 8), ("prefix7", 7)) if arm == "cached7p1" else (("final", 8),)
        for phase, rows in phases:
            item = record[phase]
            expected = list(range(rows))
            if item.get("occupied_slots") != expected or item.get("absolute_positions") != expected or item.get("payload_type") != "dequantized-F32LE":
                raise DepthExtensionError(f"{label} layer-{layer}/{phase} occupancy mismatch")
            path = r2c.base.contained_file(root, item["path"], rows * 576 * 4, f"{label} cache")
            result[f"layer{layer}/{phase}"] = r2c.load_typed(path, rows * 576, item["sha256"], "F32", f"{label} cache")
    return result


def validate_c(c_root: Path, sources: dict[str, dict[str, str]], model: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report = read_json(c_root / "strat01_rung2d.json", "C report")
    required = {"command", "c_state", "self_certifies_pass", "input_path", "byte_size", "sha256", "reference_revision", "CONFIG", "compiler_family", "compiler_embedded_version", "compiler_resolved_path_and_full_version", "engine_source_sha256", "rung2d_source_sha256", "token_ids", "positions", "arms", "timing_or_rate_claim"}
    if (not isinstance(report, dict) or set(report) != required or report["command"] != "--strat01-gguf-rung2d" or
            report["c_state"] != "ENGINE_RUNG2D_OUTPUT_READY_PENDING_REFERENCE" or report["self_certifies_pass"] is not False):
        raise DepthExtensionError("C report schema/state mismatch")
    if Path(report["input_path"]).resolve(strict=True) != model.resolve(strict=True) or report["byte_size"] != r2c.base.EXPECTED_BYTES or report["sha256"] != r2c.base.EXPECTED_SHA256:
        raise DepthExtensionError("C artifact identity mismatch")
    if report["reference_revision"] != f"llama.cpp {r2c.base.LLAMA_COMMIT}" or report["CONFIG"] != EXPECTED_C_CONFIG or report["compiler_family"] != "clang" or report["compiler_resolved_path_and_full_version"] != "EXTERNAL_RUNNER_REQUIRED":
        raise DepthExtensionError("C reference/config/compiler mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["rung2d_source_sha256"] != sources["rung2d"]["sha256"] or report["token_ids"] != r2c.base.TOKENS or report["positions"] != r2c.base.POSITIONS or report["timing_or_rate_claim"] is not None:
        raise DepthExtensionError("C source/input/non-speed contract mismatch")
    tensors: dict[str, np.ndarray] = {}; manifests: dict[str, Any] = {}; caches: dict[str, Any] = {}
    for arm in r2c.base.ARMS:
        manifest = read_json(c_root / f"{arm}_manifest.json", f"C {arm} manifest")
        if not isinstance(manifest, dict) or set(manifest) != {"arm", "payload_encoding", "shape_order", "tensors", "caches"} or manifest["arm"] != arm:
            raise DepthExtensionError(f"C {arm} manifest schema mismatch")
        by_name = {item.get("name"): item for item in manifest["tensors"] if isinstance(item, dict)}
        if set(by_name) != set(SHAPES) or len(by_name) != len(manifest["tensors"]):
            raise DepthExtensionError(f"C {arm} checkpoint set mismatch")
        for name, shape in SHAPES.items():
            item, kind, count = by_name[name], "I32" if name in I32_NAMES else "F32", math.prod(shape)
            if item.get("logical_shape") != shape or item.get("op") != EXPECTED_OPS[name] or item.get("type") != kind or item.get("payload_order") != PAYLOAD_ORDER[name] or item.get("byte_count") != 4 * count:
                raise DepthExtensionError(f"C {arm}/{name} identity mismatch")
            path = r2c.base.contained_file(c_root, item["path"], 4 * count, f"C {arm}/{name}")
            tensors[f"{arm}/{name}"] = r2c.load_typed(path, count, item["sha256"], kind, f"C {arm}/{name}")
        caches[arm] = validate_cache_records(c_root, manifest["caches"], arm, f"C {arm}")
        manifests[arm] = manifest
    return tensors, {"report": report, "manifests": manifests, "caches": caches}


def validate_reference(ref_root: Path, model: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = read_json(ref_root / "manifest.json", "reference root manifest")
    required = {"schema", "state", "llama_cpp_commit", "model", "config", "fixed_tokens", "fixed_positions", "arms"}
    if not isinstance(root, dict) or set(root) != required or root["schema"] != REFERENCE_SCHEMA or root["state"] != "REFERENCE_TRACE_READY_PENDING_C_ENGINE" or root["llama_cpp_commit"] != r2c.base.LLAMA_COMMIT:
        raise DepthExtensionError("reference root schema/state mismatch")
    model_record = root["model"]
    if Path(model_record.get("path", "")).resolve(strict=True) != model.resolve(strict=True) or model_record.get("bytes") != r2c.base.EXPECTED_BYTES or model_record.get("sha256") != r2c.base.EXPECTED_SHA256 or root["config"] != r2c.base.EXPECTED_REFERENCE_CONFIG or root["fixed_tokens"] != r2c.base.TOKENS or root["fixed_positions"] != r2c.base.POSITIONS:
        raise DepthExtensionError("reference identity/config mismatch")
    tensors: dict[str, np.ndarray] = {}; manifests: dict[str, Any] = {}; caches: dict[str, Any] = {}
    for arm in r2c.base.ARMS:
        manifest = read_json(ref_root / arm / "manifest.json", f"reference {arm} manifest")
        if not isinstance(manifest, dict) or set(manifest) != {"schema", "arm", "logical_cache_contract", "callback_records", "logical_selection", "payloads"} or manifest["schema"] != REFERENCE_SCHEMA or manifest["arm"] != arm:
            raise DepthExtensionError(f"reference {arm} manifest schema mismatch")
        contract = manifest["logical_cache_contract"]
        if contract != {"extraction": "Kcur callback values rounded through F16 by producer", "physical_bytes_claimed": False, "storage_type": "F16", "row_length": 576, "layers": [0, 1, 2], "separate_v_cache": False}:
            raise DepthExtensionError(f"reference {arm} cache contract mismatch")
        selection = manifest["logical_selection"]
        payloads = {(item.get("logical"), item.get("kind")): item for item in manifest["payloads"] if isinstance(item, dict)}
        for name, shape in SHAPES.items():
            selected, kind, count = selection.get(name), "I32" if name in I32_NAMES else "F32", math.prod(shape)
            if not isinstance(selected, dict) or selected.get("logical_shape") != shape or selected.get("composition") != ("single_prefill8_callback" if arm == "prefill8" else "prefix7_then_final1"):
                raise DepthExtensionError(f"reference {arm}/{name} logical selection mismatch")
            events = [selected.get("source")] if arm == "prefill8" else [selected.get("prefix_source"), selected.get("final_source")]
            for event, ntok in zip(events, [8] if arm == "prefill8" else [7, 1]):
                if not isinstance(event, dict) or event.get("name") != name or event.get("op") != EXPECTED_OPS[name] or event.get("shape") != shape[:-1] + [ntok] or str(event.get("type", "")).upper() != kind:
                    raise DepthExtensionError(f"reference {arm}/{name} callback identity mismatch")
            item = payloads.get((name, "full"))
            if not isinstance(item, dict) or item.get("byte_count") != 4 * count:
                raise DepthExtensionError(f"reference {arm}/{name} full payload missing")
            path = r2c.base.contained_file(ref_root, item["path"], 4 * count, f"reference {arm}/{name}")
            tensors[f"{arm}/{name}"] = r2c.load_typed(path, count, item["sha256"], kind, f"reference {arm}/{name}")
        arm_cache: dict[str, np.ndarray] = {}
        for layer in (0, 1, 2):
            logical = f"Kcur-{layer}"
            phases = (("final", "cache_f16_roundtrip", 8), ("prefix7", "prefix_cache_f16_roundtrip", 7)) if arm == "cached7p1" else (("final", "cache_f16_roundtrip", 8),)
            for phase, kind_name, rows in phases:
                item = payloads.get((logical, kind_name))
                if not isinstance(item, dict) or item.get("byte_count") != rows * 576 * 4:
                    raise DepthExtensionError(f"reference {arm}/{logical}/{phase} cache witness missing")
                path = r2c.base.contained_file(ref_root, item["path"], rows * 576 * 4, f"reference cache {logical}/{phase}")
                arm_cache[f"layer{layer}/{phase}"] = r2c.load_typed(path, rows * 576, item["sha256"], "F32", f"reference cache {logical}/{phase}")
        caches[arm] = arm_cache; manifests[arm] = manifest
    return tensors, {"root": root, "manifests": manifests, "caches": caches}


def validate_counts(c_root: Path) -> dict[str, Any]:
    counts = read_json(c_root / "strat01_f16_vector_counts.json", "layer-2 helper counts")
    if counts != EXPECTED_COUNTS:
        raise DepthExtensionError(f"layer-2 helper count mismatch: {counts!r}")
    return counts


def completed_graph_count(record: dict[str, Any], marker: str) -> int:
    text = f"{record.get('stdout', '')}\n{record.get('stderr', '')}"
    arms = re.findall(rf"^{re.escape(marker)}(prefill8|cached7p1)$", text, flags=re.MULTILINE)
    return len(arms) if sorted(arms) == ["cached7p1", "prefill8"] else -1


def layer2_negative_controls(reference: dict[str, np.ndarray]) -> dict[str, Any]:
    translated: dict[str, np.ndarray] = {}
    for arm in r2c.base.ARMS:
        translated[f"{arm}/l_out-0"] = reference[f"{arm}/l_out-1"]
        for name in r2c.SHAPES:
            if name != "l_out-0":
                translated[f"{arm}/{name}"] = reference[f"{arm}/{name[:-1]}2"]
    return r2c.negative_controls(translated)


def adjudicate(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray], c_meta: dict[str, Any], r_meta: dict[str, Any], previous_reference: dict[str, np.ndarray], previous_meta: dict[str, Any], controls: dict[str, bool], counts: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []; checkpoints: list[dict[str, Any]] = []; continuity: list[dict[str, Any]] = []; cache_results: list[dict[str, Any]] = []; predecessor_results: list[dict[str, Any]] = []
    for arm in r2c.base.ARMS:
        for name in SHAPES:
            if name in I32_NAMES:
                passed = bool(np.array_equal(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"]))
                result = {"arm": arm, "checkpoint": name, "exact": passed, "pass": passed}
            else:
                limits = r2c.base.TERMINAL_LIMITS if name in TERMINALS else r2c.base.GENERAL_LIMITS
                result = r2c.base.judged(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"], limits); result.update({"arm": arm, "checkpoint": name})
            checkpoints.append(result)
            if not result["pass"]: failures.append(f"checkpoint/{arm}/{name}")
    for implementation, values in (("c_engine", candidate), ("pinned_reference", reference)):
        for name, shape in SHAPES.items():
            a, b = r2c.base.token7(values[f"prefill8/{name}"], shape), r2c.base.token7(values[f"cached7p1/{name}"], shape)
            if name in I32_NAMES: result = {"implementation": implementation, "checkpoint": name, "exact": bool(np.array_equal(a, b)), "pass": bool(np.array_equal(a, b))}
            else: result = r2c.base.judged(a, b, r2c.base.CONTINUITY_LIMITS); result.update({"implementation": implementation, "checkpoint": name})
            continuity.append(result)
            if not result["pass"]: failures.append(f"continuity/{implementation}/{name}")
    for arm in r2c.base.ARMS:
        for key, c_values in c_meta["caches"][arm].items():
            result = r2c.base.judged(c_values, r_meta["caches"][arm][key], r2c.base.GENERAL_LIMITS); result.update({"arm": arm, "cache": key})
            cache_results.append(result)
            if not result["pass"]: failures.append(f"cache/{arm}/{key}")
        old_ref_start = previous_reference[f"{arm}/l_out-1"]
        start_exact = bool(np.array_equal(reference[f"{arm}/l_out-1"], old_ref_start))
        start_result = {"arm": arm, "surface": "reference/l_out-1", "exact": start_exact, "pass": start_exact}
        predecessor_results.append(start_result)
        if not start_exact: failures.append(f"predecessor/reference_start/{arm}")
        for layer in (0, 1):
            for phase in (("final",) if arm == "prefill8" else ("final", "prefix7")):
                key = f"layer{layer}/{phase}"
                exact = bool(np.array_equal(r_meta["caches"][arm][key], previous_meta["caches"][arm][key]))
                item = {"arm": arm, "surface": f"reference/{key}", "exact": exact, "pass": exact}
                predecessor_results.append(item)
                if not exact: failures.append(f"predecessor/reference_cache/{arm}/{key}")
        c_start_hash = c_meta["manifests"][arm]["tensors"][0]["sha256"]
        r_start_hash = next(item["sha256"] for item in r_meta["manifests"][arm]["payloads"] if item.get("logical") == "l_out-1" and item.get("kind") == "full")
        if c_start_hash != C_START_SHA or r_start_hash != REFERENCE_START_SHA: failures.append(f"start_hash/{arm}")
    negative = layer2_negative_controls(reference); negative_pass = all(not item["pass"] for item in negative.values())
    if not negative_pass: failures.append("causal_negative_controls")
    if not all(controls.values()): failures.append("source_controls")
    if counts != EXPECTED_COUNTS: failures.append("helper_counts")
    return {"status": "PASS_ENGINE_LAYER2_DEPTH_EXTENSION" if not failures else "FAIL_ENGINE_LAYER2_DEPTH_EXTENSION", "failures": failures, "checkpoint_results": checkpoints, "continuity_results": continuity, "cache_results": cache_results, "predecessor_consistency": predecessor_results, "negative_controls": negative, "negative_controls_pass": negative_pass, "source_controls": controls, "helper_counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    status = "VOID_ENGINE_LAYER2_DEPTH_EXTENSION"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    predecessor: dict[str, str] = {}
    controls: dict[str, bool] = {}
    compiler = shutil.which("clang")
    binary: Path | None = None
    reference_binary: Path | None = None
    scientific: dict[str, Any] = {}
    c_report: dict[str, Any] = {}
    production_invocations = reference_invocations = donor_graphs = reference_graphs = 0
    try:
        sources = source_inventory()
        predecessor = validate_predecessor()
        controls = source_controls()
        if not all(controls.values()):
            raise DepthExtensionError("layer-2 source controls failed: " + ", ".join(k for k, v in controls.items() if not v))
        if not compiler:
            raise DepthExtensionError("clang unavailable")
        binary = output / "engine_layer2_depth_extension.exe"
        commands["compile"] = r2c.base.run_command([compiler, *r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile", timeout=600)
        r2c.base.require_ok(commands["compile"], "layer-2 engine compile")
        reference_binary = reference_build.build(output / "reference-build", rung2d=True)
        commands["reference_build"] = {"returncode": 0, "binary": str(reference_binary)}
        commands["reference_selftest"] = r2c.base.run_command([str(reference_binary), "--self-test"], output=output, label="reference_selftest", timeout=300)
        r2c.base.require_ok(commands["reference_selftest"], "layer-2 reference selftest")
        commands["python_tests"] = r2c.base.run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], output=output, label="all_strat01_unittests", timeout=1800)
        r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
        for index, option in enumerate(SELFTESTS):
            label = f"selftest_{index:02d}"
            commands[label] = r2c.base.run_command([str(binary), option], output=output, label=label, timeout=300)
            r2c.base.require_ok(commands[label], option)
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            model = r2c.MODEL.resolve()
            if not model.is_file() or model.stat().st_size != r2c.base.EXPECTED_BYTES or sha(model) != r2c.base.EXPECTED_SHA256:
                raise DepthExtensionError("accepted artifact identity mismatch")
            previous_reference, previous_meta = r2c.validate_reference(PREVIOUS_REFERENCE, model)
            reference_root = output / "pinned_reference"
            reference_invocations = 1
            commands["reference"] = r2c.base.run_command([str(reference_binary), "--model", str(model), "--out-dir", str(reference_root), "--all"], output=output, label="reference", timeout=21600)
            r2c.base.require_ok(commands["reference"], "layer-2 reference producer")
            reference_graphs = completed_graph_count(commands["reference"], REFERENCE_GRAPH_MARKER)
            if reference_graphs != 2: raise DepthExtensionError(f"reference graph count is {reference_graphs}, expected 2")
            reference, r_meta = validate_reference(reference_root, model)
            c_root = output / "c_engine"; c_root.mkdir()
            production_invocations = 1
            commands["production"] = r2c.base.run_command([str(binary), "--strat01-gguf-rung2d", str(model), "--out-dir", str(c_root)], output=output, label="production", timeout=21600)
            r2c.base.require_ok(commands["production"], "layer-2 C producer")
            donor_graphs = completed_graph_count(commands["production"], GRAPH_MARKER)
            if donor_graphs != 2: raise DepthExtensionError(f"C graph count is {donor_graphs}, expected 2")
            sources = source_inventory()
            candidate, c_meta = validate_c(c_root, sources, model)
            counts = validate_counts(c_root)
            scientific = adjudicate(candidate, reference, c_meta, r_meta, previous_reference, previous_meta, controls, counts)
            c_report = c_meta["report"]
            status = scientific["status"]
    except (DepthExtensionError, r2c.RunnerError, r2c.base.RunnerError, reference_build.BuildError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record = {
        "schema": "strat01_engine_layer2_depth_extension_apparatus_v1",
        "status": status, "errors": errors,
        "production_invocations": production_invocations, "reference_invocations": reference_invocations,
        "donor_graph_executions": donor_graphs, "reference_graph_executions": reference_graphs,
        "expected_scientific_helper_counts": EXPECTED_COUNTS,
        "source_controls": controls,
        "provenance": {
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": time.perf_counter() - started,
            "git_head_observed" if args.apparatus_only else "git_head": r2c.base.git_value(["git", "rev-parse", "HEAD"]),
            "source_hashes": sources, "predecessor": predecessor,
            "artifact": {"path_recorded_not_opened": str(r2c.MODEL), "expected_bytes": r2c.base.EXPECTED_BYTES, "expected_sha256": r2c.base.EXPECTED_SHA256},
            "environment": {"platform": platform.platform(), "python": sys.version, "cwd": os.getcwd(), "clang_path": compiler},
            "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None},
            "reference_binary": {"path": str(reference_binary) if reference_binary else None, "sha256": sha(reference_binary) if reference_binary and reference_binary.is_file() else None},
            "commands": commands,
        },
        "adjudication": scientific, "c_report": c_report,
        "non_claims": ["layer-2 numerical fidelity", "layers 3-25", "tokenizer/logits/generation", "quality", "RAM", "rate"],
    }
    r2c.base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status.startswith("PASS_ENGINE_") or status.startswith("FAIL_ENGINE_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
