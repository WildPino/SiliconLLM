#!/usr/bin/env python3
"""Run the frozen layer-1 routed-SwiGLU component cross-input diagnostic."""
from __future__ import annotations

import argparse, copy, hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_q6_residual_propagation as prior
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE, DEFAULT_MODEL = prior.ENGINE, prior.DEFAULT_MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer1_routed_swiglu_component_cross_input.h"
SHARED = ROOT / "benchmarks/phase60/strat01_swiglu_sse2.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_SWIGLU_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer1_routed_swiglu_component_cross_input.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_routed_swiglu_component_cross_input_repair1_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_routed_swiglu_component_cross_input_apparatus_repair1_20260924"
PREDECESSOR = prior.DEFAULT_OUTPUT / "adjudication.json"
PREDECESSOR_SHA = "2eee409b8f007b9e53136e7a3d14d250af93a4d951b1ce859d4080fcfe6a9bc2"
PREDECESSOR_C = prior.DEFAULT_OUTPUT / "diagnostic/captured_c_down.f32le"
PREDECESSOR_C_SHA = "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"
REF_ROOT, C_ROOT = prior.rm.REF_ROOT, prior.rm.C_ROOT

ARM_NAMES = (
    "captured_reference_swiglu", "captured_production_c_swiglu",
    "scalar_reference_gate_reference_up", "scalar_c_gate_c_up",
    "sse2_reference_gate_reference_up", "sse2_c_gate_reference_up",
    "sse2_reference_gate_c_up", "sse2_c_gate_c_up",
    "control_reference_token7_negated",
)
ARM_KINDS = ("capture", "capture", "scalar", "scalar", "sse2", "sse2", "sse2", "sse2", "control")
VALID_STATUSES = {
    "LAYER1_ROUTED_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT",
    "LAYER1_ROUTED_GATE_INPUT_RESIDUAL_SUFFICIENT",
    "LAYER1_ROUTED_UP_INPUT_RESIDUAL_SUFFICIENT",
    "LAYER1_ROUTED_GATE_AND_UP_INPUT_RESIDUALS_SUFFICIENT",
    "LAYER1_ROUTED_GATE_UP_COMPOSITION_RESIDUAL_SUFFICIENT",
}
TOPK = REF_ROOT / "prefill8/ffn_moe_topk-1.full.i32le"
TOPK_SHA = "557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95"
INPUTS = {
    "reference_gate": (REF_ROOT / "prefill8/ffn_moe_gate-1.full.f32le", "341d79877218695fe5d26ab66fb6fb8a25737107c5199e8177eca466deed1a23", 163840),
    "reference_up": (REF_ROOT / "prefill8/ffn_moe_up-1.full.f32le", "e14f712c83cfd96404dd832ac851d06b6ab81d882567af8cb6590cc858182c9b", 163840),
    "c_gate": (C_ROOT / "prefill8_ffn_moe_gate-1.f32", "2261171cad3033adc659e1f70348d1c5ea9ac8d1adb57f40aba679f853db1293", 163840),
    "c_up": (C_ROOT / "prefill8_ffn_moe_up-1.f32", "7c2f80efe17e22ac2ef5bbf3059d67dcb45ff593cb36e9f4b54c639817a51859", 163840),
    "reference_swiglu": (REF_ROOT / "prefill8/ffn_moe_swiglu-1.full.f32le", "5097dc8599d0635477ad83f63f7e2de21b659775abc345dc933ce1e173be29e8", 163840),
    "c_swiglu": (C_ROOT / "prefill8_ffn_moe_swiglu-1.f32", "68fbf31260eacf458be0032e63fffa4f5ea33280ecdb81a53c88a815b2326a8c", 163840),
    "ref_q": prior.PINNED["ref_q"], "ref_k": prior.PINNED["ref_k"],
    "ref_ffn_inp": prior.PINNED["ref_ffn_inp"], "ref_shared_out": prior.PINNED["ref_shared_out"],
    "ref_weights": prior.PINNED["ref_weights"], "ref_target": prior.PINNED["ref_target"],
}
TWINS = {
    "reference_gate": REF_ROOT / "cached7p1/ffn_moe_gate-1.full.f32le",
    "reference_up": REF_ROOT / "cached7p1/ffn_moe_up-1.full.f32le",
    "reference_swiglu": REF_ROOT / "cached7p1/ffn_moe_swiglu-1.full.f32le",
    "c_gate": C_ROOT / "cached7p1_ffn_moe_gate-1.f32", "c_up": C_ROOT / "cached7p1_ffn_moe_up-1.f32",
    "c_swiglu": C_ROOT / "cached7p1_ffn_moe_swiglu-1.f32",
    "ref_q": prior.TWINS["ref_q"], "ref_k": prior.TWINS["ref_k"], "ref_ffn_inp": prior.TWINS["ref_ffn_inp"],
    "ref_shared_out": prior.TWINS["ref_shared_out"], "ref_weights": prior.TWINS["ref_weights"], "ref_target": prior.TWINS["ref_target"],
}


class RunnerError(RuntimeError):
    pass


def sources():
    paths = {"runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL, "engine": ENGINE, "header": HEADER, "shared_sse2": SHARED, "q6_header": ROOT / "benchmarks/phase60/strat01_gguf_layer1_q6_cross_input.h", "propagation_header": prior.HEADER, "prior_runner": Path(prior.__file__).resolve()}
    if any(not path.is_file() for path in paths.values()):
        raise RunnerError("missing routed-SwiGLU component source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def clean(source_map):
    rel = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in source_map.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, rel)], cwd=ROOT, check=False).returncode:
        raise RunnerError("routed-SwiGLU sources differ from HEAD")
    for path in rel:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"untracked routed-SwiGLU source: {path}")


def evidence():
    found = {}
    for name, (path, digest, size) in INPUTS.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise RunnerError(f"routed-SwiGLU input mismatch: {name}")
        twin = TWINS[name]
        if not twin.is_file() or twin.stat().st_size != size or base.sha256_file(twin) != digest:
            raise RunnerError(f"routed-SwiGLU twin mismatch: {name}")
        found[name] = path.resolve(strict=True)
    for path, digest, size, label in ((TOPK, TOPK_SHA, 128, "top-k"), (PREDECESSOR, PREDECESSOR_SHA, None, "predecessor"), (PREDECESSOR_C, PREDECESSOR_C_SHA, 196608, "predecessor C output")):
        if not path.is_file() or (size is not None and path.stat().st_size != size) or base.sha256_file(path) != digest:
            raise RunnerError(label + " mismatch")
    found["topk"] = TOPK.resolve(strict=True); found["prior_c"] = PREDECESSOR_C.resolve(strict=True)
    return found


def arm_manifest(outputs):
    if not isinstance(outputs, dict) or tuple(outputs) != ARM_NAMES:
        raise RunnerError("routed-SwiGLU arm labels/order mismatch")
    sizes = {"swiglu": 163840, "down": 196608, "moe_out": 49152, "downstream": 196608}
    for i, (name, item) in enumerate(outputs.items()):
        if set(item) != {"kind", *sizes} or item["kind"] != ARM_KINDS[i]:
            raise RunnerError("routed-SwiGLU arm metadata mismatch")
        for key, size in sizes.items():
            if set(item[key]) != {"path", "bytes", "sha256"} or item[key]["bytes"] != size:
                raise RunnerError("routed-SwiGLU payload schema mismatch")


def validate_report(root, model, source_map):
    report = json.loads((root / "strat01_layer1_routed_swiglu_component_cross_input.json").read_text(encoding="utf-8"))
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "outputs", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "q6_arms", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-layer1-routed-swiglu-component-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise RunnerError("routed-SwiGLU report schema mismatch")
    expected = {name: {"path": str(INPUTS[name][0].resolve()), "bytes": INPUTS[name][2], "sha256": INPUTS[name][1]} for name in tuple(INPUTS)[:-1]}
    expected["topk"] = {"path": str(TOPK.resolve()), "bytes": 128, "sha256": TOPK_SHA}
    if report["inputs"] != expected or report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise RunnerError("routed-SwiGLU identity mismatch")
    if report["q6_arms"] != 9 or report["donor_graph_executions"] or report["reference_graph_executions"] or report["timing_or_rate_claim"] is not None:
        raise RunnerError("routed-SwiGLU execution-count mismatch")
    if report["engine_source_sha256"] != source_map["engine"]["sha256"] or report["diagnostic_source_sha256"] != source_map["header"]["sha256"] or report["compiler_family"] != "clang":
        raise RunnerError("routed-SwiGLU source provenance mismatch")
    arm_manifest(report["outputs"]); values = {}
    for name, item in report["outputs"].items():
        values[name] = {key: base.load_f32(base.contained(root, item[key]["path"], item[key]["bytes"], item[key]["sha256"], name + " " + key), item[key]["bytes"] // 4, name + " " + key) for key in ("swiglu", "down", "moe_out", "downstream")}
    return values, report


def descriptive(candidate, reference, shape):
    result = base.metrics(candidate, reference); a, b = np.asarray(candidate).reshape(shape), np.asarray(reference).reshape(shape)
    result["per_token"] = [dict(base.metrics(a[i].ravel(), b[i].ravel()), token=i) for i in range(shape[0])]; return result


def adjudicate(values, report, frozen):
    target = base.load_f32(frozen["ref_target"], 49152, "reference target"); prior_c = base.load_f32(frozen["prior_c"], 49152, "predecessor C output")
    if values[ARM_NAMES[0]]["downstream"].tobytes() != target.tobytes() or values[ARM_NAMES[1]]["downstream"].tobytes() != prior_c.tobytes():
        raise RunnerError("routed-SwiGLU anchor replay mismatch")
    if values[ARM_NAMES[3]]["swiglu"].tobytes() != values[ARM_NAMES[1]]["swiglu"].tobytes():
        raise RunnerError("routed-SwiGLU scalar C replay mismatch")
    judgments = {name: base.judged(values[name]["downstream"], target) for name in ARM_NAMES}
    if not judgments[ARM_NAMES[0]]["pass"] or judgments[ARM_NAMES[1]]["pass"] or judgments[ARM_NAMES[8]]["pass"] or not judgments[ARM_NAMES[4]]["pass"]:
        raise RunnerError("routed-SwiGLU precondition contradiction")
    if not judgments[ARM_NAMES[2]]["pass"]:
        status = "LAYER1_ROUTED_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT"
    else:
        gate_fail, up_fail, both_fail = not judgments[ARM_NAMES[5]]["pass"], not judgments[ARM_NAMES[6]]["pass"], not judgments[ARM_NAMES[7]]["pass"]
        if gate_fail and up_fail: status = "LAYER1_ROUTED_GATE_AND_UP_INPUT_RESIDUALS_SUFFICIENT"
        elif gate_fail: status = "LAYER1_ROUTED_GATE_INPUT_RESIDUAL_SUFFICIENT"
        elif up_fail: status = "LAYER1_ROUTED_UP_INPUT_RESIDUAL_SUFFICIENT"
        elif both_fail: status = "LAYER1_ROUTED_GATE_UP_COMPOSITION_RESIDUAL_SUFFICIENT"
        else: raise RunnerError("no arm accounts for captured routed-SwiGLU failure")
    mutations = {}
    for name, (path, digest, size) in INPUTS.items():
        data = bytearray(path.read_bytes()); data[len(data)//2] ^= 1; mutations[name] = not base.identity_matches(bytes(data), size, digest)
    data = bytearray(TOPK.read_bytes()); data[len(data)//2] ^= 1; mutations["topk"] = not base.identity_matches(bytes(data), 128, TOPK_SHA)
    if not all(mutations.values()): raise RunnerError("routed-SwiGLU mutation control failed")
    swapped = copy.deepcopy(report["outputs"]); items = list(swapped.items()); items[5], items[6] = items[6], items[5]
    try: arm_manifest(dict(items)); label_swap = False
    except RunnerError: label_swap = True
    if not label_swap: raise RunnerError("routed-SwiGLU label swap accepted")
    ref = values[ARM_NAMES[0]]
    return {"status": status, "secondary": {"SSE2_FULL_C_REPAIR_PASSES": judgments[ARM_NAMES[7]]["pass"]}, "downstream": judgments,
            "swiglu_metrics": {name: descriptive(values[name]["swiglu"], ref["swiglu"], (8,4,1280)) for name in ARM_NAMES[:8]},
            "down_metrics": {name: descriptive(values[name]["down"], ref["down"], (8,4,1536)) for name in ARM_NAMES[:8]},
            "routed_output_metrics": {name: descriptive(values[name]["moe_out"], ref["moe_out"], (8,1536)) for name in ARM_NAMES[:8]},
            "controls": {"reference_anchor_byte_exact": True, "production_c_replay_byte_exact": True, "scalar_c_replay_byte_exact": True, "schedule_twins_byte_exact": True, "mutated_inputs_refused": mutations, "label_swap_rejected": label_swap}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, default=DEFAULT_MODEL); parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true"); args = parser.parse_args()
    model = args.model.resolve(); out = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if out.exists(): raise SystemExit(f"output already exists: {out}")
    out.mkdir(parents=True); started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter(); status = "VOID_LAYER1_ROUTED_SWIGLU_COMPONENT"; errors = []; commands = {}; report = {}; result = {"status": "NOT_RUN"}; source_map = {}; compiler = shutil.which("clang"); binary = None; invocations = 0
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None, "opened": False}; q6_completed = 0
    try:
        source_map = sources()
        if not compiler: raise RunnerError("clang unavailable")
        commands["clang_version"] = base.run_command([compiler, "--version"], out, "clang_version", 30); base.require_ok(commands["clang_version"], "clang")
        binary = out / "engine_layer1_routed_swiglu_component.exe"; commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], out, "compile", 600); base.require_ok(commands["compile"], "compile")
        selftests = (("component_selftest", "--strat01-layer1-routed-swiglu-component-cross-input-selftest"), ("q6_selftest", "--strat01-layer1-q6-cross-input-selftest"), ("propagation_selftest", "--strat01-layer1-routed-q6-residual-propagation-selftest"), ("routed_selftest", "--strat01-layer1-routed-moe-component-cross-input-selftest"), ("sse2_selftest", "--strat01-post-f16-block0-swiglu-sse2-semantics-selftest"), ("legacy_selftest", "--kselftest"))
        for label, flag in selftests:
            commands[label] = base.run_command([str(binary), flag], out, label, 300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer1_routed_swiglu_component_cross_input"], out, "python_tests", 300); base.require_ok(commands["python_tests"], "tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean(source_map)
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA: raise RunnerError("artifact mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA, "opened": True}); frozen = evidence(); root = out / "diagnostic"; root.mkdir()
            command = [str(binary), "--strat01-layer1-routed-swiglu-component-cross-input", str(model), "--topk", str(frozen["topk"]), "--ref-gate", str(frozen["reference_gate"]), "--ref-up", str(frozen["reference_up"]), "--c-gate", str(frozen["c_gate"]), "--c-up", str(frozen["c_up"]), "--ref-swiglu", str(frozen["reference_swiglu"]), "--c-swiglu", str(frozen["c_swiglu"]), "--ref-q", str(frozen["ref_q"]), "--ref-k", str(frozen["ref_k"]), "--ref-ffn-inp", str(frozen["ref_ffn_inp"]), "--ref-shared-out", str(frozen["ref_shared_out"]), "--ref-weights", str(frozen["ref_weights"]), "--out-dir", str(root)]
            invocations = 1; commands["diagnostic"] = base.run_command(command, out, "diagnostic", 21600)
            if commands["diagnostic"]["returncode"]:
                failure_path = root / "strat01_layer1_routed_swiglu_component_cross_input.json"
                if failure_path.is_file():
                    try: q6_completed = int(json.loads(failure_path.read_text(encoding="utf-8")).get("q6_arms_completed", 0))
                    except (OSError, ValueError, json.JSONDecodeError): q6_completed = 0
            base.require_ok(commands["diagnostic"], "diagnostic"); q6_completed = 9; source_map = sources(); values, report = validate_report(root, model, source_map); result = adjudicate(values, report, frozen); status = result["status"]
    except (RunnerError, base.RunnerError) as exc: errors.append(str(exc))
    except Exception as exc: errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record = {"schema": "strat01_layer1_routed_swiglu_component_cross_input_v1", "status": status, "errors": errors, "diagnostic_invocations": invocations, "q6_arms_completed": q6_completed, "donor_graph_executions": 0, "reference_graph_executions": 0, "adjudication": result, "c_report": report, "provenance": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-tick, "git_head_observed" if args.apparatus_only else "git_head": base.git_value(["git","rev-parse","HEAD"]), "source_hashes": source_map, "predecessor": {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}, "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd()}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}, "non_claims": ["graph execution", "production repair", "later layers", "quality/RAM/rate"]}
    base.write_json(out / "adjudication.json", record); print(json.dumps({"status": status, "output": str(out), "errors": errors}, indent=2)); return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
