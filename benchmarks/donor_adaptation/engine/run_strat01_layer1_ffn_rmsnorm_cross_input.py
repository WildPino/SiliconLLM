#!/usr/bin/env python3
"""Execute the frozen layer-1 FFN RMSNorm cross-input diagnostic."""
from __future__ import annotations

import argparse, copy, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_up_projection_cross_input as up
from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_swiglu_component_cross_input as sw
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE, DEFAULT_MODEL = up.ENGINE, up.DEFAULT_MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer1_ffn_rmsnorm_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_RMSNORM_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer1_ffn_rmsnorm_cross_input.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_ffn_rmsnorm_cross_input_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_ffn_rmsnorm_cross_input_apparatus_repair4_20260924"
PREDECESSOR = up.DEFAULT_OUTPUT / "adjudication.json"
PREDECESSOR_SHA = "2de61cdc02deeb9639044cb75cc614705a41eb3e133b00f3ad2746b80bc7c71d"
PREDECESSOR_C = up.DEFAULT_OUTPUT / "diagnostic/current_up_on_c_norm.downstream.f32le"
PREDECESSOR_C_SHA = "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"

ARM_NAMES = (
    "captured_reference_norm", "captured_production_c_norm",
    "computed_norm_on_reference_input", "computed_norm_on_c_input",
    "control_reference_input_token7_negated", "control_norm_weight_negated",
)
ARM_META = {
    "captured_reference_norm": ("captured", "reference", False, False),
    "captured_production_c_norm": ("captured", "c", False, False),
    "computed_norm_on_reference_input": ("computed", "reference", False, False),
    "computed_norm_on_c_input": ("computed", "c", False, False),
    "control_reference_input_token7_negated": ("computed", "reference", True, False),
    "control_norm_weight_negated": ("computed", "reference", False, True),
}
VALID_STATUSES = {
    "LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT",
    "LAYER1_FFN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT",
}
REF_ROOT, C_ROOT = up.REF_ROOT, up.C_ROOT
TOPK, TOPK_SHA = up.TOPK, up.TOPK_SHA
REFERENCE_TARGET = up.INPUTS["ref_target"]
REFERENCE_TARGET_TWIN = up.TWINS["ref_target"]
INPUTS = {
    "reference_input": (REF_ROOT / "prefill8/ffn_inp-1.full.f32le", "99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8", 49152),
    "c_input": (C_ROOT / "prefill8_ffn_inp-1.f32", "c506fcc0e51d6b80c90958374527f48c1dd25150bf91b9ff83851c059ec890b2", 49152),
    "reference_norm": up.INPUTS["reference_norm"], "c_norm": up.INPUTS["c_norm"],
    "reference_up": up.INPUTS["reference_up"], "c_up": up.INPUTS["c_up"],
    "reference_gate": up.INPUTS["reference_gate"], "ref_q": up.INPUTS["ref_q"],
    "ref_k": up.INPUTS["ref_k"], "ref_ffn_inp": up.INPUTS["ref_ffn_inp"],
    "ref_shared_out": up.INPUTS["ref_shared_out"], "ref_weights": up.INPUTS["ref_weights"],
}
TWINS = {
    "reference_input": REF_ROOT / "cached7p1/ffn_inp-1.full.f32le",
    "c_input": C_ROOT / "cached7p1_ffn_inp-1.f32",
    "reference_norm": up.TWINS["reference_norm"], "c_norm": up.TWINS["c_norm"],
    "reference_up": up.TWINS["reference_up"], "c_up": up.TWINS["c_up"],
    "reference_gate": up.TWINS["reference_gate"], "ref_q": up.TWINS["ref_q"],
    "ref_k": up.TWINS["ref_k"], "ref_ffn_inp": up.TWINS["ref_ffn_inp"],
    "ref_shared_out": up.TWINS["ref_shared_out"], "ref_weights": up.TWINS["ref_weights"],
}


class RunnerError(RuntimeError):
    pass


def sources():
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "engine": ENGINE, "header": HEADER, "routed_up_header": up.HEADER,
        "routed_up_runner": Path(up.__file__).resolve(), "swiglu_header": sw.HEADER,
        "swiglu_runner": Path(sw.__file__).resolve(), "base_runner": Path(base.__file__).resolve(),
        "shared_sse2": sw.SHARED,
    }
    if any(not path.is_file() for path in paths.values()):
        raise RunnerError("missing layer-1 FFN RMSNorm source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def clean(source_map):
    rel = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in source_map.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, rel)], cwd=ROOT, check=False).returncode:
        raise RunnerError("layer-1 FFN RMSNorm sources differ from HEAD")
    for path in rel:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"untracked layer-1 FFN RMSNorm source: {path}")


def evidence():
    found = {}
    for name, (path, digest, size) in INPUTS.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise RunnerError(f"layer-1 FFN RMSNorm input mismatch: {name}")
        twin = TWINS[name]
        if not twin.is_file() or twin.stat().st_size != size or base.sha256_file(twin) != digest:
            raise RunnerError(f"layer-1 FFN RMSNorm twin mismatch: {name}")
        found[name] = path.resolve(strict=True)
    checks = (
        (TOPK, TOPK_SHA, 128, "top-k"),
        (PREDECESSOR, PREDECESSOR_SHA, None, "routed-up predecessor"),
        (PREDECESSOR_C, PREDECESSOR_C_SHA, 196608, "routed-up C output"),
    )
    for path, digest, size, label in checks:
        if not path.is_file() or (size is not None and path.stat().st_size != size) or base.sha256_file(path) != digest:
            raise RunnerError(f"layer-1 FFN RMSNorm {label} mismatch")
    found["topk"] = TOPK.resolve(strict=True)
    found["prior_c"] = PREDECESSOR_C.resolve(strict=True)
    target_path, target_digest, target_size = REFERENCE_TARGET
    if (not target_path.is_file() or target_path.stat().st_size != target_size or
            base.sha256_file(target_path) != target_digest or not REFERENCE_TARGET_TWIN.is_file() or
            REFERENCE_TARGET_TWIN.stat().st_size != target_size or
            base.sha256_file(REFERENCE_TARGET_TWIN) != target_digest):
        raise RunnerError("layer-1 FFN RMSNorm reference target mismatch")
    found["ref_target"] = target_path.resolve(strict=True)
    return found


def arm_manifest(outputs):
    if not isinstance(outputs, dict) or tuple(outputs) != ARM_NAMES:
        raise RunnerError("layer-1 FFN RMSNorm arm labels/order mismatch")
    sizes = {"norm": 49152, "up": 163840, "swiglu": 163840, "down": 196608, "moe_out": 49152, "downstream": 196608}
    for name, item in outputs.items():
        if set(item) != {"kind", "source", "input_control", "weight_control", *sizes}:
            raise RunnerError("layer-1 FFN RMSNorm arm schema mismatch")
        if (item["kind"], item["source"], item["input_control"], item["weight_control"]) != ARM_META[name]:
            raise RunnerError("layer-1 FFN RMSNorm arm metadata mismatch")
        for key, size in sizes.items():
            if set(item[key]) != {"path", "bytes", "sha256"} or item[key]["bytes"] != size:
                raise RunnerError("layer-1 FFN RMSNorm payload schema mismatch")


def validate_report(root, model, source_map):
    report = json.loads((root / "strat01_layer1_ffn_rmsnorm_cross_input.json").read_text(encoding="utf-8"))
    required = {
        "command", "state", "self_certifies_pass", "model", "inputs", "matrices", "outputs",
        "engine_source_sha256", "diagnostic_source_sha256", "compiler_family",
        "rmsnorm_arms_completed", "up_projection_arms_completed", "propagated_arms_completed",
        "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim",
    }
    if set(report) != required or report["command"] != "--strat01-layer1-ffn-rmsnorm-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise RunnerError("layer-1 FFN RMSNorm report schema mismatch")
    expected = {name: {"path": str(path.resolve()), "bytes": size, "sha256": digest} for name, (path, digest, size) in INPUTS.items()}
    expected["topk"] = {"path": str(TOPK.resolve()), "bytes": 128, "sha256": TOPK_SHA}
    matrices = {
        "ffn_norm": {"name": "blk.1.ffn_norm.weight", "type": "F32", "shape": [1536], "offset": 504978688, "file_offset": 511081600, "span": 6144},
        "up_experts": {"name": "blk.1.ffn_up_exps.weight", "type": "Q4_K", "shape": [1536, 1280, 64]},
        "down_experts": {"name": "blk.1.ffn_down_exps.weight", "type": "Q6_K", "shape": [1280, 1536, 64]},
    }
    if report["inputs"] != expected or report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA} or report["matrices"] != matrices:
        raise RunnerError("layer-1 FFN RMSNorm identity mismatch")
    if (report["engine_source_sha256"] != source_map["engine"]["sha256"] or report["diagnostic_source_sha256"] != source_map["header"]["sha256"] or report["compiler_family"] != "clang" or report["rmsnorm_arms_completed"] != 4 or report["up_projection_arms_completed"] != 4 or report["propagated_arms_completed"] != 6 or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None):
        raise RunnerError("layer-1 FFN RMSNorm execution accounting mismatch")
    arm_manifest(report["outputs"])
    values = {}
    for name, item in report["outputs"].items():
        values[name] = {key: base.load_f32(base.contained(root, item[key]["path"], item[key]["bytes"], item[key]["sha256"], f"{name} {key}"), item[key]["bytes"] // 4, f"{name} {key}") for key in ("norm", "up", "swiglu", "down", "moe_out", "downstream")}
    return values, report


def descriptive(candidate, reference, shape):
    result = base.metrics(candidate, reference)
    candidate_view = np.asarray(candidate).reshape(shape)
    reference_view = np.asarray(reference).reshape(shape)
    result["per_token"] = [dict(base.metrics(candidate_view[token].ravel(), reference_view[token].ravel()), token=token) for token in range(shape[0])]
    return result


def judged_downstream(candidate, reference):
    result = base.judged(candidate, reference)
    result["aggregate_pass"] = result["pass"]
    for item in result["per_token"]:
        item.update({"nrmse_limit": base.LIMITS[0], "normalized_max_limit": base.LIMITS[1]})
        item["pass"] = item["nrmse"] <= base.LIMITS[0] and item["normalized_max"] <= base.LIMITS[1]
    result["pass"] = result["aggregate_pass"] and all(item["pass"] for item in result["per_token"])
    return result


def classify(judgments):
    if not judgments["captured_reference_norm"]["pass"] or judgments["captured_production_c_norm"]["pass"]:
        raise RunnerError("layer-1 FFN RMSNorm anchor contradiction")
    if judgments["computed_norm_on_reference_input"]["pass"] and not judgments["computed_norm_on_c_input"]["pass"]:
        return "LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT"
    if not judgments["computed_norm_on_reference_input"]["pass"] and not judgments["computed_norm_on_c_input"]["pass"]:
        return "LAYER1_FFN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT"
    return "VOID_LAYER1_FFN_RMSNORM_CROSS_INPUT"


def adjudicate(values, report, frozen):
    target = base.load_f32(frozen["ref_target"], 49152, "reference target")
    prior_c = base.load_f32(frozen["prior_c"], 49152, "routed-up C output")
    ref = values["captured_reference_norm"]
    c = values["captured_production_c_norm"]
    if ref["norm"].tobytes() != base.load_f32(frozen["reference_norm"], 12288, "reference norm").tobytes() or c["norm"].tobytes() != base.load_f32(frozen["c_norm"], 12288, "C norm").tobytes():
        raise RunnerError("layer-1 FFN RMSNorm captured norm mismatch")
    if ref["downstream"].tobytes() != target.tobytes() or c["downstream"].tobytes() != prior_c.tobytes():
        raise RunnerError("layer-1 FFN RMSNorm captured downstream mismatch")
    for stage in ("norm", "up", "swiglu", "down", "moe_out", "downstream"):
        if values["computed_norm_on_c_input"][stage].tobytes() != c[stage].tobytes():
            raise RunnerError(f"layer-1 FFN RMSNorm computed-C {stage} replay mismatch")
    judgments = {name: judged_downstream(values[name]["downstream"], target) for name in ARM_NAMES[:4]}
    controls = {name: judged_downstream(values[name]["downstream"], target) for name in ARM_NAMES[4:]}
    if any(item["pass"] for item in controls.values()):
        raise RunnerError("layer-1 FFN RMSNorm planted control passed")
    mutations = {}
    for name, (path, digest, size) in INPUTS.items():
        data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1
        mutations[name] = not base.identity_matches(bytes(data), size, digest)
    data = bytearray(TOPK.read_bytes()); data[len(data) // 2] ^= 1
    mutations["topk"] = not base.identity_matches(bytes(data), 128, TOPK_SHA)
    target_path, target_digest, target_size = REFERENCE_TARGET
    data = bytearray(target_path.read_bytes()); data[len(data) // 2] ^= 1
    mutations["ref_target"] = not base.identity_matches(bytes(data), target_size, target_digest)
    if not all(mutations.values()):
        raise RunnerError("layer-1 FFN RMSNorm mutation control failed")
    swapped = copy.deepcopy(report["outputs"]); items = list(swapped.items()); items[2], items[3] = items[3], items[2]
    try:
        arm_manifest(dict(items)); label_swap = False
    except RunnerError:
        label_swap = True
    if not label_swap:
        raise RunnerError("layer-1 FFN RMSNorm label swap accepted")
    shapes = {"norm": (8, 1536), "up": (8, 4, 1280), "swiglu": (8, 4, 1280), "down": (8, 4, 1536), "moe_out": (8, 1536), "downstream": (8, 6144)}
    reference_replay = {
        stage: values["computed_norm_on_reference_input"][stage].tobytes() == ref[stage].tobytes()
        for stage in ("norm", "up", "swiglu", "down", "moe_out", "downstream")
    }
    return {
        "status": classify(judgments), "downstream": judgments, "controls_vs_reference": controls,
        "stage_metrics": {stage: {name: descriptive(values[name][stage], ref[stage], shape) for name in ARM_NAMES} for stage, shape in shapes.items()},
        "controls": {"captured_anchors_byte_exact": True, "computed_reference_stage_replay": reference_replay, "computed_c_all_stages_byte_exact": True, "schedule_twins_byte_exact": True, "mutated_inputs_refused": mutations, "label_swap_rejected": label_swap},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    out = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    out.mkdir(parents=True)
    started, tick = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status, errors, commands, report, result, source_map = "VOID_LAYER1_FFN_RMSNORM_CROSS_INPUT", [], {}, {}, {"status": "NOT_RUN"}, {}
    compiler, binary, invocations = shutil.which("clang"), None, 0
    counters = {"rmsnorm_arms_completed": 0, "up_projection_arms_completed": 0, "propagated_arms_completed": 0}
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None, "opened": False}
    try:
        source_map = sources()
        if not compiler:
            raise RunnerError("clang unavailable")
        commands["clang_version"] = base.run_command([compiler, "--version"], out, "clang_version", 30); base.require_ok(commands["clang_version"], "clang")
        binary = out / "engine_layer1_ffn_rmsnorm_cross_input.exe"
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], out, "compile", 600); base.require_ok(commands["compile"], "compile")
        selftests = (
            ("ffn_rmsnorm_selftest", "--strat01-layer1-ffn-rmsnorm-cross-input-selftest"),
            ("routed_up_selftest", "--strat01-layer1-routed-up-projection-cross-input-selftest"),
            ("swiglu_selftest", "--strat01-layer1-routed-swiglu-component-cross-input-selftest"),
            ("q6_selftest", "--strat01-layer1-q6-cross-input-selftest"),
            ("propagation_selftest", "--strat01-layer1-routed-q6-residual-propagation-selftest"),
            ("legacy_selftest", "--kselftest"),
        )
        for label, flag in selftests:
            commands[label] = base.run_command([str(binary), flag], out, label, 300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer1_ffn_rmsnorm_cross_input"], out, "python_tests", 300); base.require_ok(commands["python_tests"], "tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean(source_map)
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise RunnerError("artifact mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA, "opened": True})
            frozen = evidence(); root = out / "diagnostic"; root.mkdir()
            command = [str(binary), "--strat01-layer1-ffn-rmsnorm-cross-input", str(model), "--topk", str(frozen["topk"]), "--ref-input", str(frozen["reference_input"]), "--c-input", str(frozen["c_input"]), "--ref-norm", str(frozen["reference_norm"]), "--c-norm", str(frozen["c_norm"]), "--ref-up", str(frozen["reference_up"]), "--c-up", str(frozen["c_up"]), "--ref-gate", str(frozen["reference_gate"]), "--ref-q", str(frozen["ref_q"]), "--ref-k", str(frozen["ref_k"]), "--ref-ffn-inp", str(frozen["ref_ffn_inp"]), "--ref-shared-out", str(frozen["ref_shared_out"]), "--ref-weights", str(frozen["ref_weights"]), "--out-dir", str(root)]
            invocations = 1; commands["diagnostic"] = base.run_command(command, out, "diagnostic", 21600)
            failure = root / "strat01_layer1_ffn_rmsnorm_cross_input.json"
            if commands["diagnostic"]["returncode"] and failure.is_file():
                try:
                    data = json.loads(failure.read_text(encoding="utf-8")); counters = {key: int(data.get(key, 0)) for key in counters}
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            base.require_ok(commands["diagnostic"], "diagnostic")
            counters = {"rmsnorm_arms_completed": 4, "up_projection_arms_completed": 4, "propagated_arms_completed": 6}
            source_map = sources(); values, report = validate_report(root, model, source_map); result = adjudicate(values, report, frozen); status = result["status"]
    except (RunnerError, base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record = {
        "schema": "strat01_layer1_ffn_rmsnorm_cross_input_v1", "status": status, "errors": errors,
        "diagnostic_invocations": invocations, **counters, "donor_graph_executions": 0, "reference_graph_executions": 0,
        "adjudication": result, "c_report": report,
        "provenance": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - tick, "git_head_observed" if args.apparatus_only else "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "source_hashes": source_map, "predecessor": {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}, "predecessor_output": {"path": str(PREDECESSOR_C), "sha256": PREDECESSOR_C_SHA}, "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd()}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands},
        "non_claims": ["graph execution", "production repair", "later layers", "quality/RAM/rate"],
    }
    base.write_json(out / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(out), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
