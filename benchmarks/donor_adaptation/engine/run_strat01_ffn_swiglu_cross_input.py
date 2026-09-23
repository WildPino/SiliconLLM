#!/usr/bin/env python3
"""Execute the frozen FFN SwiGLU cross-input diagnostic."""
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
from benchmarks.donor_adaptation.engine import run_strat01_ffn_down_cross_input as down


term, l1, base = down.term, down.l1, down.base
ROOT, HERE, ENGINE = down.ROOT, down.HERE, down.ENGINE
HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_ffn_swiglu_cross_input.h"
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_ffn_swiglu_cross_input.py"
DEFAULT_MODEL = down.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_ffn_swiglu_cross_input_20260923"
BLOCK0, REFERENCE = down.BLOCK0, down.REFERENCE
PREVIOUS = HERE / "results" / "strat01_gigachat_engine_ffn_down_cross_input_20260923"
PREVIOUS_SHA = "6d3ab2b6a1097502060380350238940d434d38b2b41bcdff8d7c586f5f8d05d0"
ORDER, SIZES = l1.ORDER, l1.SIZES
ARMS = ["reference_captured_control", "reference_gate_reference_up", "c_gate_reference_up", "reference_gate_c_up", "c_gate_c_up"]
ORIGINS = {"reference_captured_control": ("captured", "captured"), "reference_gate_reference_up": ("reference", "reference"), "c_gate_reference_up": ("c", "reference"), "reference_gate_c_up": ("reference", "c"), "c_gate_c_up": ("c", "c")}
INPUTS: dict[str, tuple[Path, str, int]] = {
    "c_gate": (BLOCK0 / "prefill8_ffn_gate-0.f32", "bb52399fa69bc0f9295c0b2b2fc9588ec7689e440b7306f6df2e43f53c3710fa", 286_720),
    "reference_gate": (REFERENCE / "ffn_gate-0.full.f32le", "5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a", 286_720),
    "c_up": (BLOCK0 / "prefill8_ffn_up-0.f32", "34a98ab5d44a7c1282901db0be2e152ea1a37f0cdf88366e65acd0597dbc390f", 286_720),
    "reference_up": (REFERENCE / "ffn_up-0.full.f32le", "2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4", 286_720),
    "c_swiglu": (BLOCK0 / "prefill8_ffn_swiglu-0.f32", "e4073a39ca0d6f904dab90b22f8fac9c9516c219b265611a102da8dd3597bc8d", 286_720),
    "reference_swiglu": (REFERENCE / "ffn_swiglu-0.full.f32le", "de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef", 286_720),
    "reference_ffn_inp": (REFERENCE / "ffn_inp-0.full.f32le", "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1", 49_152),
}
REFERENCE_FFN_OUT = down.INPUTS["reference_ffn_out"]


def source_inventory() -> dict[str, Any]:
    paths = {"runner": Path(__file__).resolve(), "down_runner": Path(down.__file__).resolve(), "terminal_runner": Path(term.__file__).resolve(), "layer1_runner": Path(l1.__file__).resolve(), "engine": ENGINE, "rung2a": l1.RUNG2A, "rung2b": ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2b.h", "rung2c": l1.RUNG2C, "down_header": down.HEADER, "header": HEADER, "protocol": PROTOCOL, "tests": TESTS}
    if any(not path.is_file() for path in paths.values()): raise base.RunnerError("missing SwiGLU source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def validate_inputs() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, (path, digest, size) in INPUTS.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest: raise base.RunnerError(f"input identity mismatch: {name}")
        paths[name] = path.resolve(strict=True)
    ref_out_path, ref_out_digest, ref_out_size = REFERENCE_FFN_OUT
    if not ref_out_path.is_file() or ref_out_path.stat().st_size != ref_out_size or base.sha256_file(ref_out_path) != ref_out_digest:
        raise base.RunnerError("input identity mismatch: reference_ffn_out")
    if base.sha256_file(PREVIOUS / "adjudication.json") != PREVIOUS_SHA: raise base.RunnerError("previous FFN-down adjudication mismatch")
    record = json.loads((PREVIOUS / "adjudication.json").read_text(encoding="utf-8"))
    if record.get("status") != "SWIGLU_INPUT_RESIDUAL_SUFFICIENT" or record.get("donor_graph_executions") != 0 or record.get("reference_graph_executions") != 0: raise base.RunnerError("previous FFN-down status/accounting mismatch")
    return paths


def load_item(root: Path, item: Any, size: int, label: str) -> np.ndarray:
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"} or item["bytes"] != size: raise base.RunnerError(f"{label} schema mismatch")
    return base.load_f32(base.contained(root, item["path"], size, item["sha256"], label), size // 4, label)


def wide_judgment(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = base.metrics(candidate, reference); result.update({"nrmse_limit": base.LIMITS[0], "normalized_max_limit": base.LIMITS[1]}); result["pass"] = result["nrmse"] <= base.LIMITS[0] and result["normalized_max"] <= base.LIMITS[1]
    result["per_token"] = [dict(base.metrics(candidate.reshape(8, 8960)[i], reference.reshape(8, 8960)[i]), token=i) for i in range(8)]; return result


def validate_report(root: Path, model: Path, sources: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads((root / "strat01_ffn_swiglu_cross_input.json").read_text(encoding="utf-8")); required = {"command", "state", "self_certifies_pass", "model", "inputs", "down_tensor", "layer1_tensors", "arms", "controls", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-ffn-swiglu-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False: raise base.RunnerError("SwiGLU report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}: raise base.RunnerError("model report mismatch")
    expected_inputs = {name: {"path": str(paths[name]), "bytes": size, "sha256": digest} for name, (_, digest, size) in INPUTS.items()}
    if report["inputs"] != expected_inputs: raise base.RunnerError("input report mismatch")
    if report["down_tensor"].get("name") != "blk.0.ffn_down.weight" or report["down_tensor"].get("type") != 14 or set(report["down_tensor"]) != {"name", "type", "offset", "file_offset", "span"}: raise base.RunnerError("down descriptor mismatch")
    expected_l1 = ["blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_kv_a_norm.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.1.attn_output.weight"]
    descriptor_keys = {"name", "type", "offset", "file_offset", "span"}
    if [item.get("name") for item in report["layer1_tensors"]] != expected_l1 or any(set(item) != descriptor_keys for item in report["layer1_tensors"]): raise base.RunnerError("layer-1 descriptor mismatch")
    if list(report["arms"]) != ARMS: raise base.RunnerError("arm ordering mismatch")
    values: dict[str, Any] = {}
    for arm in ARMS:
        item = report["arms"][arm]
        if set(item) != {"gate_origin", "up_origin", "swiglu", "ffn_out", "l_out_sha256", "checkpoints"} or (item["gate_origin"], item["up_origin"]) != ORIGINS[arm]: raise base.RunnerError(f"arm provenance/schema mismatch: {arm}")
        values[arm] = {"swiglu": load_item(root, item["swiglu"], 286_720, f"{arm}/swiglu"), "ffn_out": load_item(root, item["ffn_out"], 49_152, f"{arm}/ffn_out"), "checkpoints": l1.validate_output_map(root, item["checkpoints"], arm)}
    if set(report["controls"]) != {"reference_gate_negated", "reference_up_rows0_7_swapped"}: raise base.RunnerError("control set mismatch")
    if any(not isinstance(item, dict) or set(item) != {"swiglu", "kqv_out-1"} for item in report["controls"].values()): raise base.RunnerError("control item schema mismatch")
    controls = {name: {"swiglu": load_item(root, item["swiglu"], 286_720, f"{name}/swiglu"), "kqv_out-1": load_item(root, item["kqv_out-1"], 196_608, f"{name}/kqv")} for name, item in report["controls"].items()}
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None: raise base.RunnerError("source/execution contract mismatch")
    return values, {"report": report, "controls": controls}


def adjudicate(values: dict[str, Any], validated: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    report, controls = validated["report"], validated["controls"]; previous = json.loads((PREVIOUS / "adjudication.json").read_text(encoding="utf-8")); frozen = l1.validate_frozen(); ref_sw = base.load_f32(paths["reference_swiglu"], 8 * 8960, "reference swiglu")
    swiglu_judgments = {arm: wide_judgment(values[arm]["swiglu"], ref_sw) for arm in ARMS}; reference_ffn_out = base.load_f32(REFERENCE_FFN_OUT[0], 8 * 1536, "reference ffn_out"); out_judgments = {arm: down.direct_judgment(values[arm]["ffn_out"], reference_ffn_out) for arm in ARMS}; downstream: dict[str, Any] = {}
    for arm in ARMS:
        downstream[arm] = {checkpoint: l1.judged_checkpoint(values[arm]["checkpoints"][checkpoint], base.load_f32(frozen[f"ref/{checkpoint}"], SIZES[checkpoint] // 4, f"ref/{checkpoint}"), checkpoint) for checkpoint in ORDER}
    prev_ref = previous["c_report"]["arms"]["reference_swiglu_current_q6"]; prev_c = previous["c_report"]["arms"]["c_swiglu_current_q6"]
    for checkpoint in ORDER:
        if report["arms"]["reference_captured_control"]["checkpoints"][checkpoint]["sha256"] != prev_ref["checkpoints"][checkpoint]["sha256"]: raise base.RunnerError(f"captured reference replay mismatch: {checkpoint}")
        if report["arms"]["c_gate_c_up"]["checkpoints"][checkpoint]["sha256"] != prev_c["checkpoints"][checkpoint]["sha256"]: raise base.RunnerError(f"C/C propagated replay mismatch: {checkpoint}")
        old = previous["adjudication"]["downstream_vs_reference"]["c_swiglu_current_q6"][checkpoint]; new = downstream["c_gate_c_up"][checkpoint]
        if abs(new["nrmse"] - old["nrmse"]) > 1e-12 or abs(new["normalized_max"] - old["normalized_max"]) > 1e-12: raise base.RunnerError(f"C/C metric replay mismatch: {checkpoint}")
    if report["arms"]["c_gate_c_up"]["swiglu"]["sha256"] != INPUTS["c_swiglu"][1] or report["arms"]["c_gate_c_up"]["ffn_out"]["sha256"] != down.INPUTS["c_ffn_out"][1]: raise base.RunnerError("C/C direct replay mismatch")
    control_judgments = {}; ref_kqv = base.load_f32(frozen["ref/kqv_out-1"], 8 * 6144, "reference kqv")
    for name, item in controls.items():
        control_judgments[name] = {"swiglu": wide_judgment(item["swiglu"], ref_sw), "kqv_out-1": l1.judged_checkpoint(item["kqv_out-1"], ref_kqv, "kqv_out-1")}
        if control_judgments[name]["swiglu"]["pass"] and control_judgments[name]["kqv_out-1"]["pass"]: raise base.RunnerError(f"causal control did not reject: {name}")
    mutation_refused = {}
    for name, path in paths.items(): data = bytearray(path.read_bytes()); data[len(data)//2] ^= 1; mutation_refused[name] = not base.identity_matches(bytes(data), INPUTS[name][2], INPUTS[name][1])
    swapped = copy.deepcopy(report["arms"]); swapped["c_gate_reference_up"]["gate_origin"] = "reference"; origin_swap_rejected = swapped["c_gate_reference_up"] != report["arms"]["c_gate_reference_up"]
    if not all(mutation_refused.values()) or not origin_swap_rejected: raise base.RunnerError("identity/origin control failure")
    arm_fails = {arm: any(not downstream[arm][checkpoint]["pass"] for checkpoint in ORDER) for arm in ARMS}
    if not arm_fails["c_gate_c_up"]: raise base.RunnerError("C/C replay unexpectedly passes")
    if arm_fails["reference_gate_reference_up"]: status = "SWIGLU_OPERATOR_RESIDUAL_SUFFICIENT"
    else:
        gate_fails, up_fails = arm_fails["c_gate_reference_up"], arm_fails["reference_gate_c_up"]
        if gate_fails and up_fails: status = "GATE_AND_UP_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
        elif gate_fails: status = "GATE_RESIDUAL_SUFFICIENT"
        elif up_fails: status = "UP_RESIDUAL_SUFFICIENT"
        else: status = "GATE_UP_RESIDUAL_JOINT_ONLY"
    first_failures = {arm: next((checkpoint for checkpoint in ORDER if not downstream[arm][checkpoint]["pass"]), None) for arm in ARMS}
    return {"status": status, "first_failures": first_failures, "swiglu_vs_reference": swiglu_judgments, "ffn_out_vs_reference": out_judgments, "downstream_vs_reference": downstream, "control_judgments": control_judgments, "controls": {"captured_reference_replay_exact": True, "c_c_direct_and_propagated_replay_exact": True, "c_c_metrics_reproduced_within_1e-12": True, "mutated_inputs_refused": mutation_refused, "operand_origin_swap_rejected": origin_swap_rejected}}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--model",type=Path,default=DEFAULT_MODEL);parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);parser.add_argument("--apparatus-only",action="store_true");args=parser.parse_args();model,output=args.model.resolve(),args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True,exist_ok=True);started_utc,started=base.now_utc(),time.perf_counter();status="VOID_FFN_SWIGLU_CROSS_INPUT";errors:list[str]=[];commands:dict[str,Any]={};sources:dict[str,Any]={};report:dict[str,Any]={};adjudication:dict[str,Any]={"status":"NOT_RUN"};compiler=shutil.which("clang");binary:Path|None=None
    try:
        sources=source_inventory()
        if not compiler: raise base.RunnerError("clang unavailable")
        commands["compile"]=base.run_command([compiler,*base.COMPILE_FLAGS,str(ENGINE),"-o",str(output/"engine_ffn_swiglu.exe"),"-lm"],output,"compile",600);base.require_ok(commands["compile"],"compile");binary=output/"engine_ffn_swiglu.exe"
        commands["selftest"]=base.run_command([str(binary),"--strat01-ffn-swiglu-cross-input-selftest"],output,"selftest",300);base.require_ok(commands["selftest"],"selftest");commands["down_selftest"]=base.run_command([str(binary),"--strat01-ffn-down-cross-input-selftest"],output,"down_selftest",300);base.require_ok(commands["down_selftest"],"down selftest");commands["legacy"]=base.run_command([str(binary),"--kselftest"],output,"legacy",300);base.require_ok(commands["legacy"],"legacy");commands["python_tests"]=base.run_command([sys.executable,"-B","-m","unittest","-v","benchmarks.donor_adaptation.engine.test_strat01_ffn_swiglu_cross_input"],output,"python_tests",300);base.require_ok(commands["python_tests"],"Python tests")
        if args.apparatus_only: status="APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            if not model.is_file() or model.stat().st_size!=base.EXPECTED_MODEL_BYTES or base.sha256_file(model)!=base.EXPECTED_MODEL_SHA: raise base.RunnerError("model identity mismatch")
            paths=validate_inputs();root=output/"diagnostic";root.mkdir();command=[str(binary),"--strat01-ffn-swiglu-cross-input",str(model),"--c-gate",str(paths["c_gate"]),"--reference-gate",str(paths["reference_gate"]),"--c-up",str(paths["c_up"]),"--reference-up",str(paths["reference_up"]),"--c-swiglu",str(paths["c_swiglu"]),"--reference-swiglu",str(paths["reference_swiglu"]),"--reference-ffn-inp",str(paths["reference_ffn_inp"]),"--out-dir",str(root)];commands["diagnostic"]=base.run_command(command,output,"diagnostic",21600);base.require_ok(commands["diagnostic"],"diagnostic");sources=source_inventory();values,validated=validate_report(root,model,sources,paths);report=validated["report"];adjudication=adjudicate(values,validated,paths);status=adjudication["status"]
    except base.RunnerError as exc: errors.append(str(exc))
    except Exception as exc: errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance={"started_utc":started_utc,"finished_utc":base.now_utc(),"seconds":time.perf_counter()-started,"git_head":base.git_value(["git","rev-parse","HEAD"]),"git_status_porcelain":base.git_value(["git","status","--porcelain"]),"source_hashes":sources,"environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd(),"clang_path":compiler},"binary":{"path":str(binary) if binary else None,"sha256":base.sha256_file(binary) if binary and binary.is_file() else None},"commands":commands};record={"schema":"strat01_ffn_swiglu_cross_input_adjudication_v1","status":status,"errors":errors,"donor_graph_executions":0,"reference_graph_executions":0,"adjudication":adjudication,"c_report":report,"non_claims":["Rung 2C repair or promotion","projection/operator attribution","MoE/later layers","quality, generation, RAM, or rate"],"provenance":provenance};manifest={"schema":"strat01_ffn_swiglu_cross_input_run_manifest_v1","status":status,"errors":errors,"donor_graph_executions":0,"reference_graph_executions":0,"provenance":provenance};base.write_json(output/"adjudication.json",record);base.write_json(output/"run_manifest.json",manifest);print(json.dumps({"status":status,"output":str(output),"errors":errors},indent=2));return 0 if not status.startswith("VOID_") else 2


if __name__=="__main__":raise SystemExit(main())
