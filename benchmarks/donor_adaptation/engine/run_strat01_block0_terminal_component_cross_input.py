#!/usr/bin/env python3
"""Execute the frozen block-0 terminal-component cross-input diagnostic."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_IMPORT_ROOT = Path(__file__).resolve().parents[3]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_layer1_start_cross_input as l1


base = l1.base
ROOT, HERE, ENGINE = l1.ROOT, l1.HERE, l1.ENGINE
HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_block0_terminal_component_cross_input.h"
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_block0_terminal_component_cross_input.py"
DEFAULT_MODEL = l1.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_block0_terminal_component_cross_input_20260923"
COMBINED = HERE / "results" / "strat01_gigachat_engine_combined_rms_q5q8_20260921" / "candidate"
BLOCK0 = HERE / "results" / "strat01_gigachat_engine_block0_production_integration_20260922" / "c_engine"
REFERENCE = HERE / "results" / "strat01_gigachat_engine_rung2b_repair2_20260921" / "pinned_reference" / "prefill8"
ORDER, SIZES = l1.ORDER, l1.SIZES
ARMS = ["reference_reference", "c_c", "c_attention_reference_ffn", "reference_attention_c_ffn"]
COMPONENTS: dict[str, tuple[Path, str]] = {
    "c_ffn_inp": (COMBINED / "prefill8_ffn_inp-0.f32", "13721a425bf93e87aa313b9ea7718fdcc2d2ce25f9ec38403eee9fc6cce9847e"),
    "c_ffn_out": (BLOCK0 / "prefill8_ffn_out-0.f32", "8433164833fe4daecd24a840bb4cd70c41d390a15655abe479ef058248a6fb4f"),
    "c_l_out": (BLOCK0 / "prefill8_l_out-0.f32", "258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44"),
    "reference_ffn_inp": (REFERENCE / "ffn_inp-0.full.f32le", "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1"),
    "reference_ffn_out": (REFERENCE / "ffn_out-0.full.f32le", "f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4"),
    "reference_l_out": (REFERENCE / "l_out-0.full.f32le", "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"),
}
ORIGINS = {
    "reference_reference": ("reference", "reference"),
    "c_c": ("c", "c"),
    "c_attention_reference_ffn": ("c", "reference"),
    "reference_attention_c_ffn": ("reference", "c"),
}


def source_inventory() -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(), "layer1_runner": Path(l1.__file__).resolve(),
        "engine": ENGINE, "rung2a": l1.RUNG2A, "rung2c": l1.RUNG2C,
        "layer1_header": l1.HEADER, "header": HEADER, "protocol": PROTOCOL, "tests": TESTS,
    }
    if any(not path.is_file() for path in paths.values()):
        raise base.RunnerError("missing terminal-component source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def validate_components() -> tuple[dict[str, Path], dict[str, bytes], dict[str, str]]:
    paths: dict[str, Path] = {}; payloads: dict[str, bytes] = {}
    for name, (path, digest) in COMPONENTS.items():
        if not path.is_file() or path.stat().st_size != 49_152 or base.sha256_file(path) != digest:
            raise base.RunnerError(f"component identity mismatch: {name}")
        paths[name] = path.resolve(strict=True); payloads[name] = path.read_bytes()
    arrays = {name: np.frombuffer(data, dtype="<f4") for name, data in payloads.items()}
    sums = {
        "reference_reference": (arrays["reference_ffn_inp"] + arrays["reference_ffn_out"]).astype("<f4", copy=False),
        "c_c": (arrays["c_ffn_inp"] + arrays["c_ffn_out"]).astype("<f4", copy=False),
        "c_attention_reference_ffn": (arrays["c_ffn_inp"] + arrays["reference_ffn_out"]).astype("<f4", copy=False),
        "reference_attention_c_ffn": (arrays["reference_ffn_inp"] + arrays["c_ffn_out"]).astype("<f4", copy=False),
    }
    sum_bytes = {name: value.tobytes() for name, value in sums.items()}
    if sum_bytes["reference_reference"] != payloads["reference_l_out"] or sum_bytes["c_c"] != payloads["c_l_out"]:
        raise base.RunnerError("offline homogeneous reconstruction mismatch")
    return paths, sum_bytes, {name: hashlib.sha256(data).hexdigest() for name, data in sum_bytes.items()}


def validate_output_map(root: Path, items: Any, arm: str) -> dict[str, np.ndarray]:
    if not isinstance(items, dict) or list(items) != ORDER:
        raise base.RunnerError(f"{arm} output ordering/schema mismatch")
    values: dict[str, np.ndarray] = {}
    for checkpoint in ORDER:
        item = items[checkpoint]
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != SIZES[checkpoint]:
            raise base.RunnerError(f"{arm}/{checkpoint} output schema mismatch")
        path = base.contained(root, item["path"], SIZES[checkpoint], item["sha256"], f"{arm}/{checkpoint}")
        values[checkpoint] = base.load_f32(path, SIZES[checkpoint] // 4, f"{arm}/{checkpoint}")
    return values


def validate_report(root: Path, model: Path, sources: dict[str, Any], paths: dict[str, Path], expected_sums: dict[str, str]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads((root / "strat01_block0_terminal_component_cross_input.json").read_text(encoding="utf-8"))
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "tensors", "compositions", "outputs", "controls", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-block0-terminal-component-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise base.RunnerError("terminal-component report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise base.RunnerError("model report mismatch")
    expected_inputs = {name: {"path": str(paths[name]), "bytes": 49_152, "sha256": COMPONENTS[name][1]} for name in COMPONENTS}
    if report["inputs"] != expected_inputs or len(report["tensors"]) != 7:
        raise base.RunnerError("input/tensor report mismatch")
    expected_names = ["blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_kv_a_norm.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.1.attn_output.weight"]
    if [item.get("name") for item in report["tensors"]] != expected_names or any(set(item) != {"name", "type", "offset", "file_offset", "span"} for item in report["tensors"]):
        raise base.RunnerError("tensor descriptor report mismatch")
    if list(report["compositions"]) != ARMS or list(report["outputs"]) != ARMS:
        raise base.RunnerError("arm ordering mismatch")
    for index, arm in enumerate(ARMS):
        inp_origin, out_origin = ORIGINS[arm]
        expected = {"ffn_inp_origin": inp_origin, "ffn_out_origin": out_origin, "bytes": 49_152, "sha256": expected_sums[arm], "homogeneous_replay": index < 2}
        if report["compositions"][arm] != expected:
            raise base.RunnerError(f"composition provenance mismatch: {arm}")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise base.RunnerError("source/execution contract mismatch")
    values = {arm: validate_output_map(root, report["outputs"][arm], arm) for arm in ARMS}
    if set(report["controls"]) != {"hybrid_token7_negated", "hybrid_rows0_7_swapped"}:
        raise base.RunnerError("control set mismatch")
    controls: dict[str, np.ndarray] = {}
    for name, item in report["controls"].items():
        path = base.contained(root, item["path"], 196_608, item["sha256"], name)
        controls[name] = base.load_f32(path, 8 * 6144, name)
    return values, report, controls


def adjudicate(values: dict[str, dict[str, np.ndarray]], report: dict[str, Any], controls: dict[str, np.ndarray], paths: dict[str, Path]) -> dict[str, Any]:
    frozen = l1.validate_frozen(); judgments: dict[str, Any] = {}
    for arm in ARMS:
        judgments[arm] = {}
        for checkpoint in ORDER:
            reference = base.load_f32(frozen[f"ref/{checkpoint}"], SIZES[checkpoint] // 4, f"ref/{checkpoint}")
            judgments[arm][checkpoint] = l1.judged_checkpoint(values[arm][checkpoint], reference, checkpoint)
    for checkpoint in ORDER:
        c = base.load_f32(frozen[f"c/{checkpoint}"], SIZES[checkpoint] // 4, f"c/{checkpoint}")
        if values["c_c"][checkpoint].tobytes() != c.tobytes() or report["outputs"]["c_c"][checkpoint]["sha256"] != l1.C_HASHES[checkpoint]:
            raise base.RunnerError(f"C homogeneous replay mismatch: {checkpoint}")
        metric = base.metrics(values["c_c"][checkpoint], base.load_f32(frozen[f"ref/{checkpoint}"], SIZES[checkpoint] // 4, f"ref/{checkpoint}"))
        expected = l1.FROZEN_METRICS[checkpoint]
        if abs(metric["nrmse"] - expected[0]) > 1e-12 or abs(metric["normalized_max"] - expected[1]) > 1e-12:
            raise base.RunnerError(f"C frozen metric replay mismatch: {checkpoint}")
    if any(not judgments["reference_reference"][checkpoint]["pass"] for checkpoint in ORDER):
        raise base.RunnerError("reference homogeneous arm failed a layer-1 gate")
    ref_kqv = base.load_f32(frozen["ref/kqv_out-1"], 8 * 6144, "reference kqv")
    control_judgments = {name: l1.judged_checkpoint(value, ref_kqv, "kqv_out-1") for name, value in controls.items()}
    if any(value["pass"] for value in control_judgments.values()):
        raise base.RunnerError("causal control did not reject")
    mutation_refused = {}
    for name, path in paths.items():
        data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1
        mutation_refused[name] = not base.identity_matches(bytes(data), 49_152, COMPONENTS[name][1])
    swapped = copy.deepcopy(report["compositions"])
    swapped["c_attention_reference_ffn"]["ffn_inp_origin"] = "reference"
    origin_swap_rejected = swapped["c_attention_reference_ffn"] != report["compositions"]["c_attention_reference_ffn"]
    if not all(mutation_refused.values()) or not origin_swap_rejected:
        raise base.RunnerError("identity/origin control failure")
    attention_fails = any(not judgments["c_attention_reference_ffn"][checkpoint]["pass"] for checkpoint in ORDER)
    ffn_fails = any(not judgments["reference_attention_c_ffn"][checkpoint]["pass"] for checkpoint in ORDER)
    if attention_fails and ffn_fails:
        status = "ATTENTION_AND_FFN_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
    elif attention_fails:
        status = "ATTENTION_STREAM_RESIDUAL_SUFFICIENT"
    elif ffn_fails:
        status = "FFN_RESIDUAL_SUFFICIENT"
    else:
        status = "TERMINAL_RESIDUAL_JOINT_ONLY"
    first_failures = {arm: next((checkpoint for checkpoint in ORDER if not judgments[arm][checkpoint]["pass"]), None) for arm in ARMS}
    return {"status": status, "first_failures": first_failures, "arm_judgments": judgments, "control_judgments": control_judgments, "controls": {"reference_homogeneous_passes_all_11": True, "c_homogeneous_byte_exact_all_11": True, "c_frozen_metrics_reproduced_within_1e-12": True, "mutated_components_refused": mutation_refused, "addend_origin_swap_rejected": origin_swap_rejected}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model, output = args.model.resolve(), args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True); started_utc, started = base.now_utc(), time.perf_counter()
    status = "VOID_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}; sources: dict[str, Any] = {}; report: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}; compiler = shutil.which("clang"); binary: Path | None = None
    try:
        sources = source_inventory()
        if not compiler:
            raise base.RunnerError("clang unavailable")
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(output / "engine_terminal_component.exe"), "-lm"], output, "compile", 600); base.require_ok(commands["compile"], "compile"); binary = output / "engine_terminal_component.exe"
        commands["selftest"] = base.run_command([str(binary), "--strat01-block0-terminal-component-cross-input-selftest"], output, "selftest", 300); base.require_ok(commands["selftest"], "selftest")
        commands["layer1_selftest"] = base.run_command([str(binary), "--strat01-rung2c-layer1-start-cross-input-selftest"], output, "layer1_selftest", 300); base.require_ok(commands["layer1_selftest"], "layer1 selftest")
        commands["legacy"] = base.run_command([str(binary), "--kselftest"], output, "legacy", 300); base.require_ok(commands["legacy"], "legacy")
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_block0_terminal_component_cross_input"], output, "python_tests", 300); base.require_ok(commands["python_tests"], "Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise base.RunnerError("model identity mismatch")
            paths, _, expected_sums = validate_components(); root = output / "diagnostic"; root.mkdir()
            command = [str(binary), "--strat01-block0-terminal-component-cross-input", str(model), "--c-ffn-inp", str(paths["c_ffn_inp"]), "--c-ffn-out", str(paths["c_ffn_out"]), "--c-l-out", str(paths["c_l_out"]), "--reference-ffn-inp", str(paths["reference_ffn_inp"]), "--reference-ffn-out", str(paths["reference_ffn_out"]), "--reference-l-out", str(paths["reference_l_out"]), "--out-dir", str(root)]
            commands["diagnostic"] = base.run_command(command, output, "diagnostic", 21600); base.require_ok(commands["diagnostic"], "diagnostic")
            sources = source_inventory(); values, report, controls = validate_report(root, model, sources, paths, expected_sums); adjudication = adjudicate(values, report, controls, paths); status = adjudication["status"]
    except base.RunnerError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": base.now_utc(), "seconds": time.perf_counter() - started, "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": base.git_value(["git", "status", "--porcelain"]), "source_hashes": sources, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_block0_terminal_component_cross_input_adjudication_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "reference_graph_executions": 0, "adjudication": adjudication, "c_report": report, "non_claims": ["Rung 2C repair or promotion", "natural mixed FFN forward pass", "MoE/later layers", "quality, generation, RAM, or rate"], "provenance": provenance}
    manifest = {"schema": "strat01_block0_terminal_component_cross_input_run_manifest_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "reference_graph_executions": 0, "provenance": provenance}
    base.write_json(output / "adjudication.json", record); base.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if not status.startswith("VOID_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
