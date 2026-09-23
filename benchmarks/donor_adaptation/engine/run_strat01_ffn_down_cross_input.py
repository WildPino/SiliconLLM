#!/usr/bin/env python3
"""Execute the frozen FFN down-projection cross-input diagnostic."""
from __future__ import annotations

import argparse
import copy
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
from benchmarks.donor_adaptation.engine import run_strat01_block0_terminal_component_cross_input as term


l1, base = term.l1, term.base
ROOT, HERE, ENGINE = term.ROOT, term.HERE, term.ENGINE
HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_ffn_down_cross_input.h"
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_FFN_DOWN_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_ffn_down_cross_input.py"
DEFAULT_MODEL = term.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_ffn_down_cross_input_20260923"
BLOCK0 = term.BLOCK0; REFERENCE = term.REFERENCE
PREVIOUS = HERE / "results" / "strat01_gigachat_engine_block0_terminal_component_cross_input_20260923"
PREVIOUS_ADJUDICATION_SHA = "edbb82a814a92be9f822d37f6683375dfa437b5b0774cbae13b86632d2a9a6f5"
ORDER, SIZES = l1.ORDER, l1.SIZES
ARMS = ["reference_captured_control", "reference_swiglu_current_q6", "c_swiglu_current_q6"]
INPUTS: dict[str, tuple[Path, str, int]] = {
    "c_swiglu": (BLOCK0 / "prefill8_ffn_swiglu-0.f32", "e4073a39ca0d6f904dab90b22f8fac9c9516c219b265611a102da8dd3597bc8d", 286_720),
    "reference_swiglu": (REFERENCE / "ffn_swiglu-0.full.f32le", "de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef", 286_720),
    "c_ffn_out": (BLOCK0 / "prefill8_ffn_out-0.f32", "8433164833fe4daecd24a840bb4cd70c41d390a15655abe479ef058248a6fb4f", 49_152),
    "reference_ffn_out": (REFERENCE / "ffn_out-0.full.f32le", "f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4", 49_152),
    "reference_ffn_inp": (REFERENCE / "ffn_inp-0.full.f32le", "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1", 49_152),
}
SOURCES = {"reference_captured_control": "captured_reference", "reference_swiglu_current_q6": "reference_swiglu_current_q6", "c_swiglu_current_q6": "c_swiglu_current_q6"}


def source_inventory() -> dict[str, Any]:
    paths = {"runner": Path(__file__).resolve(), "terminal_runner": Path(term.__file__).resolve(), "layer1_runner": Path(l1.__file__).resolve(), "engine": ENGINE, "rung2a": l1.RUNG2A, "rung2b": ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2b.h", "rung2c": l1.RUNG2C, "layer1_header": l1.HEADER, "terminal_header": term.HEADER, "header": HEADER, "protocol": PROTOCOL, "tests": TESTS}
    if any(not path.is_file() for path in paths.values()):
        raise base.RunnerError("missing FFN-down source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def validate_inputs() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, (path, digest, size) in INPUTS.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise base.RunnerError(f"input identity mismatch: {name}")
        paths[name] = path.resolve(strict=True)
    if base.sha256_file(PREVIOUS / "adjudication.json") != PREVIOUS_ADJUDICATION_SHA:
        raise base.RunnerError("previous terminal-component adjudication mismatch")
    previous = json.loads((PREVIOUS / "adjudication.json").read_text(encoding="utf-8"))
    if previous.get("status") != "FFN_RESIDUAL_SUFFICIENT" or previous.get("donor_graph_executions") != 0 or previous.get("reference_graph_executions") != 0:
        raise base.RunnerError("previous terminal-component status/accounting mismatch")
    return paths


def previous_record() -> dict[str, Any]:
    return json.loads((PREVIOUS / "adjudication.json").read_text(encoding="utf-8"))


def direct_judgment(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = base.metrics(candidate, reference)
    result.update({"nrmse_limit": base.LIMITS[0], "normalized_max_limit": base.LIMITS[1]})
    result["pass"] = result["nrmse"] <= base.LIMITS[0] and result["normalized_max"] <= base.LIMITS[1]
    result["per_token"] = [dict(base.metrics(candidate.reshape(8, 1536)[i], reference.reshape(8, 1536)[i]), token=i) for i in range(8)]
    return result


def load_output(root: Path, item: Any, size: int, label: str) -> np.ndarray:
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"} or item["bytes"] != size:
        raise base.RunnerError(f"{label} output schema mismatch")
    path = base.contained(root, item["path"], size, item["sha256"], label)
    return base.load_f32(path, size // 4, label)


def validate_report(root: Path, model: Path, sources: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads((root / "strat01_ffn_down_cross_input.json").read_text(encoding="utf-8"))
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "down_tensor", "layer1_tensors", "arms", "controls", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-ffn-down-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise base.RunnerError("FFN-down report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise base.RunnerError("model report mismatch")
    expected_inputs = {name: {"path": str(paths[name]), "bytes": size, "sha256": digest} for name, (_, digest, size) in INPUTS.items()}
    if report["inputs"] != expected_inputs:
        raise base.RunnerError("input report mismatch")
    down = report["down_tensor"]
    if set(down) != {"name", "type", "offset", "file_offset", "span"} or down["name"] != "blk.0.ffn_down.weight" or down["type"] != 14:
        raise base.RunnerError("down descriptor mismatch")
    expected_l1 = ["blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_kv_a_norm.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.1.attn_output.weight"]
    if [item.get("name") for item in report["layer1_tensors"]] != expected_l1 or any(set(item) != {"name", "type", "offset", "file_offset", "span"} for item in report["layer1_tensors"]):
        raise base.RunnerError("layer-1 descriptor mismatch")
    if list(report["arms"]) != ARMS or any(report["arms"][arm].get("source") != SOURCES[arm] for arm in ARMS):
        raise base.RunnerError("arm/source mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise base.RunnerError("source/execution contract mismatch")
    values: dict[str, Any] = {}
    for arm in ARMS:
        item = report["arms"][arm]
        if set(item) != {"source", "ffn_out", "l_out_sha256", "checkpoints"}:
            raise base.RunnerError(f"arm schema mismatch: {arm}")
        if not isinstance(item["l_out_sha256"], str) or len(item["l_out_sha256"]) != 64:
            raise base.RunnerError(f"sum hash mismatch: {arm}")
        values[arm] = {"ffn_out": load_output(root, item["ffn_out"], 49_152, f"{arm}/ffn_out")}
        values[arm]["checkpoints"] = l1.validate_output_map(root, item["checkpoints"], arm)
    if set(report["controls"]) != {"reference_swiglu_negated", "reference_swiglu_rows0_7_swapped"}:
        raise base.RunnerError("control set mismatch")
    controls: dict[str, Any] = {}
    for name, item in report["controls"].items():
        if set(item) != {"ffn_out", "kqv_out-1"}:
            raise base.RunnerError(f"control schema mismatch: {name}")
        controls[name] = {"ffn_out": load_output(root, item["ffn_out"], 49_152, f"{name}/ffn_out"), "kqv_out-1": load_output(root, item["kqv_out-1"], 196_608, f"{name}/kqv")}
    return values, {"report": report, "controls": controls}


def adjudicate(values: dict[str, Any], validated: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    report, controls = validated["report"], validated["controls"]; previous = previous_record(); frozen = l1.validate_frozen()
    ref_out = base.load_f32(paths["reference_ffn_out"], 8 * 1536, "reference ffn_out")
    direct = {arm: direct_judgment(values[arm]["ffn_out"], ref_out) for arm in ARMS}
    downstream: dict[str, Any] = {}
    for arm in ARMS:
        downstream[arm] = {}
        for checkpoint in ORDER:
            ref = base.load_f32(frozen[f"ref/{checkpoint}"], SIZES[checkpoint] // 4, f"ref/{checkpoint}")
            downstream[arm][checkpoint] = l1.judged_checkpoint(values[arm]["checkpoints"][checkpoint], ref, checkpoint)
    expected_rr = previous["c_report"]["outputs"]["reference_reference"]
    expected_c = previous["c_report"]["outputs"]["reference_attention_c_ffn"]
    for checkpoint in ORDER:
        if report["arms"]["reference_captured_control"]["checkpoints"][checkpoint]["sha256"] != expected_rr[checkpoint]["sha256"]:
            raise base.RunnerError(f"reference captured replay mismatch: {checkpoint}")
        if report["arms"]["c_swiglu_current_q6"]["checkpoints"][checkpoint]["sha256"] != expected_c[checkpoint]["sha256"]:
            raise base.RunnerError(f"C SwiGLU propagated replay mismatch: {checkpoint}")
        old = previous["adjudication"]["arm_judgments"]["reference_attention_c_ffn"][checkpoint]
        new = downstream["c_swiglu_current_q6"][checkpoint]
        if abs(new["nrmse"] - old["nrmse"]) > 1e-12 or abs(new["normalized_max"] - old["normalized_max"]) > 1e-12:
            raise base.RunnerError(f"C SwiGLU metric replay mismatch: {checkpoint}")
    if report["arms"]["c_swiglu_current_q6"]["ffn_out"]["sha256"] != INPUTS["c_ffn_out"][1] or report["arms"]["c_swiglu_current_q6"]["l_out_sha256"] != "b822e425fb6c88faa62cc59dec8011a7869ebb9d64c3731e54beb585f66cb541":
        raise base.RunnerError("C Q6/output terminal replay mismatch")
    if not all(downstream["reference_captured_control"][checkpoint]["pass"] for checkpoint in ORDER):
        raise base.RunnerError("reference captured control failed")
    control_judgments = {}
    ref_kqv = base.load_f32(frozen["ref/kqv_out-1"], 8 * 6144, "reference kqv")
    for name, item in controls.items():
        control_judgments[name] = {"ffn_out": direct_judgment(item["ffn_out"], ref_out), "kqv_out-1": l1.judged_checkpoint(item["kqv_out-1"], ref_kqv, "kqv_out-1")}
        if control_judgments[name]["ffn_out"]["pass"] and control_judgments[name]["kqv_out-1"]["pass"]:
            raise base.RunnerError(f"causal control did not reject: {name}")
    mutation_refused = {}
    for name, path in paths.items():
        data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1
        mutation_refused[name] = not base.identity_matches(bytes(data), INPUTS[name][2], INPUTS[name][1])
    swapped = copy.deepcopy(report["arms"]); swapped["reference_swiglu_current_q6"]["source"] = "c_swiglu_current_q6"
    origin_swap_rejected = swapped["reference_swiglu_current_q6"] != report["arms"]["reference_swiglu_current_q6"]
    if not all(mutation_refused.values()) or not origin_swap_rejected:
        raise base.RunnerError("identity/origin control failure")
    exact_arm_pass = direct["reference_swiglu_current_q6"]["pass"] and all(downstream["reference_swiglu_current_q6"][checkpoint]["pass"] for checkpoint in ORDER)
    c_arm_fails = (not direct["c_swiglu_current_q6"]["pass"]) or any(not downstream["c_swiglu_current_q6"][checkpoint]["pass"] for checkpoint in ORDER)
    if not c_arm_fails:
        raise base.RunnerError("C replay unexpectedly passes")
    status = "SWIGLU_INPUT_RESIDUAL_SUFFICIENT" if exact_arm_pass else "Q6_DOWN_OPERATOR_RESIDUAL_SUFFICIENT"
    first_failures = {arm: {"direct": None if direct[arm]["pass"] else "ffn_out-0", "downstream": next((checkpoint for checkpoint in ORDER if not downstream[arm][checkpoint]["pass"]), None)} for arm in ARMS}
    return {"status": status, "first_failures": first_failures, "direct_ffn_out_vs_reference": direct, "downstream_vs_reference": downstream, "control_judgments": control_judgments, "controls": {"captured_reference_replay_exact_all_11": True, "c_q6_output_replay_exact": True, "c_terminal_and_propagation_replay_exact": True, "c_metrics_reproduced_within_1e-12": True, "mutated_inputs_refused": mutation_refused, "input_origin_swap_rejected": origin_swap_rejected}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--model", type=Path, default=DEFAULT_MODEL); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--apparatus-only", action="store_true"); args = parser.parse_args(); model, output = args.model.resolve(), args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True); started_utc, started = base.now_utc(), time.perf_counter(); status = "VOID_FFN_DOWN_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}; sources: dict[str, Any] = {}; report: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}; compiler = shutil.which("clang"); binary: Path | None = None
    try:
        sources = source_inventory()
        if not compiler: raise base.RunnerError("clang unavailable")
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(output / "engine_ffn_down.exe"), "-lm"], output, "compile", 600); base.require_ok(commands["compile"], "compile"); binary = output / "engine_ffn_down.exe"
        commands["selftest"] = base.run_command([str(binary), "--strat01-ffn-down-cross-input-selftest"], output, "selftest", 300); base.require_ok(commands["selftest"], "selftest")
        commands["terminal_selftest"] = base.run_command([str(binary), "--strat01-block0-terminal-component-cross-input-selftest"], output, "terminal_selftest", 300); base.require_ok(commands["terminal_selftest"], "terminal selftest")
        commands["legacy"] = base.run_command([str(binary), "--kselftest"], output, "legacy", 300); base.require_ok(commands["legacy"], "legacy")
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_ffn_down_cross_input"], output, "python_tests", 300); base.require_ok(commands["python_tests"], "Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA: raise base.RunnerError("model identity mismatch")
            paths = validate_inputs(); root = output / "diagnostic"; root.mkdir(); command = [str(binary), "--strat01-ffn-down-cross-input", str(model), "--c-swiglu", str(paths["c_swiglu"]), "--reference-swiglu", str(paths["reference_swiglu"]), "--c-ffn-out", str(paths["c_ffn_out"]), "--reference-ffn-out", str(paths["reference_ffn_out"]), "--reference-ffn-inp", str(paths["reference_ffn_inp"]), "--out-dir", str(root)]
            commands["diagnostic"] = base.run_command(command, output, "diagnostic", 21600); base.require_ok(commands["diagnostic"], "diagnostic"); sources = source_inventory(); values, validated = validate_report(root, model, sources, paths); report = validated["report"]; adjudication = adjudicate(values, validated, paths); status = adjudication["status"]
    except base.RunnerError as exc: errors.append(str(exc))
    except Exception as exc: errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": base.now_utc(), "seconds": time.perf_counter()-started, "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": base.git_value(["git", "status", "--porcelain"]), "source_hashes": sources, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_ffn_down_cross_input_adjudication_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "reference_graph_executions": 0, "adjudication": adjudication, "c_report": report, "non_claims": ["Rung 2C repair or promotion", "gate/up/SwiGLU split", "MoE/later layers", "quality, generation, RAM, or rate"], "provenance": provenance}; manifest = {"schema": "strat01_ffn_down_cross_input_run_manifest_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "reference_graph_executions": 0, "provenance": provenance}
    base.write_json(output / "adjudication.json", record); base.write_json(output / "run_manifest.json", manifest); print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2)); return 0 if not status.startswith("VOID_") else 2


if __name__ == "__main__": raise SystemExit(main())
