#!/usr/bin/env python3
"""Execute the frozen layer-1 attention-output residual cross-input diagnostic."""
from __future__ import annotations

import argparse, copy, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_layer1_ffn_rmsnorm_cross_input as fr
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE, DEFAULT_MODEL = fr.ENGINE, fr.DEFAULT_MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer1_attention_output_residual_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer1_attention_output_residual_cross_input.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_attention_output_residual_cross_input_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_attention_output_residual_cross_input_apparatus_20260924"
PREDECESSOR = fr.DEFAULT_OUTPUT / "adjudication.json"
PREDECESSOR_SHA = "831fe805bdffeee26693543bed60e96f9f0fe42945e315b16108677736589943"
PREDECESSOR_C = fr.DEFAULT_OUTPUT / "diagnostic/captured_production_c_norm.downstream.f32le"
PREDECESSOR_C_SHA = "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"

ARM_NAMES = (
    "captured_reference_ffn_input", "captured_production_c_ffn_input",
    "computed_projection_plus_reference_residual", "computed_projection_plus_c_residual",
    "control_reference_residual_token6_negated", "control_kqv_token7_negated",
)
ARM_META = {
    ARM_NAMES[0]: ("captured", "reference", False, False),
    ARM_NAMES[1]: ("captured", "c", False, False),
    ARM_NAMES[2]: ("computed", "reference", False, False),
    ARM_NAMES[3]: ("computed", "c", False, False),
    ARM_NAMES[4]: ("computed", "reference", True, False),
    ARM_NAMES[5]: ("computed", "reference", False, True),
}
VALID_STATUSES = {
    "LAYER1_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT",
    "LAYER1_ATTN_OUTPUT_PROJECTION_FAILS_EXACT_KQV",
}
TOPK, TOPK_SHA = fr.TOPK, fr.TOPK_SHA
REFERENCE_TARGET, REFERENCE_TARGET_TWIN = fr.REFERENCE_TARGET, fr.REFERENCE_TARGET_TWIN
REF_ROOT, C_ROOT = fr.REF_ROOT, fr.C_ROOT
INPUTS = {
    "reference_residual": (REF_ROOT / "prefill8/l_out-0.full.f32le", "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa", 49152),
    "c_residual": (C_ROOT / "prefill8_l_out-0.f32", "a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11", 49152),
    "reference_kqv": (REF_ROOT / "prefill8/kqv_out-1.full.f32le", "fa1006c4e2c365d3a5540baebbe4c88180f524d5f69c610fcc3b4205a856d489", 196608),
    "c_kqv": (C_ROOT / "prefill8_kqv_out-1.f32", "fa1006c4e2c365d3a5540baebbe4c88180f524d5f69c610fcc3b4205a856d489", 196608),
    "reference_ffn_input": (REF_ROOT / "prefill8/ffn_inp-1.full.f32le", "99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8", 49152),
    "c_ffn_input": (C_ROOT / "prefill8_ffn_inp-1.f32", "c506fcc0e51d6b80c90958374527f48c1dd25150bf91b9ff83851c059ec890b2", 49152),
    "reference_gate": fr.INPUTS["reference_gate"], "ref_q": fr.INPUTS["ref_q"],
    "ref_k": fr.INPUTS["ref_k"], "ref_shared_out": fr.INPUTS["ref_shared_out"],
    "ref_weights": fr.INPUTS["ref_weights"],
}
TWINS = {
    "reference_residual": REF_ROOT / "cached7p1/l_out-0.full.f32le",
    "c_residual": C_ROOT / "cached7p1_l_out-0.f32",
    "reference_kqv": REF_ROOT / "cached7p1/kqv_out-1.full.f32le",
    "c_kqv": C_ROOT / "cached7p1_kqv_out-1.f32",
    "reference_ffn_input": REF_ROOT / "cached7p1/ffn_inp-1.full.f32le",
    "c_ffn_input": C_ROOT / "cached7p1_ffn_inp-1.f32",
    "reference_gate": fr.TWINS["reference_gate"], "ref_q": fr.TWINS["ref_q"],
    "ref_k": fr.TWINS["ref_k"], "ref_shared_out": fr.TWINS["ref_shared_out"],
    "ref_weights": fr.TWINS["ref_weights"],
}


class RunnerError(RuntimeError):
    pass


def sources():
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "engine": ENGINE, "header": HEADER, "ffn_rmsnorm_header": fr.HEADER,
        "ffn_rmsnorm_runner": Path(fr.__file__).resolve(),
        "routed_up_header": fr.up.HEADER, "routed_up_runner": Path(fr.up.__file__).resolve(),
        "swiglu_header": fr.sw.HEADER, "swiglu_runner": Path(fr.sw.__file__).resolve(),
        "base_runner": Path(base.__file__).resolve(), "shared_sse2": fr.sw.SHARED,
    }
    if any(not path.is_file() for path in paths.values()):
        raise RunnerError("missing layer-1 attention-output residual source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def clean(source_map):
    rel = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in source_map.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, rel)], cwd=ROOT, check=False).returncode:
        raise RunnerError("layer-1 attention-output residual sources differ from HEAD")
    for path in rel:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"untracked layer-1 attention-output residual source: {path}")


def evidence():
    found = {}
    for name, (path, digest, size) in INPUTS.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise RunnerError(f"layer-1 attention-output residual input mismatch: {name}")
        twin = TWINS[name]
        if not twin.is_file() or twin.stat().st_size != size or base.sha256_file(twin) != digest:
            raise RunnerError(f"layer-1 attention-output residual twin mismatch: {name}")
        found[name] = path.resolve(strict=True)
    for path, digest, size, label in (
        (TOPK, TOPK_SHA, 128, "top-k"), (PREDECESSOR, PREDECESSOR_SHA, None, "predecessor"),
        (PREDECESSOR_C, PREDECESSOR_C_SHA, 196608, "predecessor C output"),
    ):
        if not path.is_file() or (size is not None and path.stat().st_size != size) or base.sha256_file(path) != digest:
            raise RunnerError(f"layer-1 attention-output residual {label} mismatch")
    found["topk"] = TOPK.resolve(strict=True); found["prior_c"] = PREDECESSOR_C.resolve(strict=True)
    target, digest, size = REFERENCE_TARGET
    if (not target.is_file() or target.stat().st_size != size or base.sha256_file(target) != digest or
            not REFERENCE_TARGET_TWIN.is_file() or REFERENCE_TARGET_TWIN.stat().st_size != size or
            base.sha256_file(REFERENCE_TARGET_TWIN) != digest):
        raise RunnerError("layer-1 attention-output residual reference target mismatch")
    found["ref_target"] = target.resolve(strict=True)
    return found


def arm_manifest(outputs):
    if not isinstance(outputs, dict) or tuple(outputs) != ARM_NAMES:
        raise RunnerError("layer-1 attention-output residual arm labels/order mismatch")
    sizes = {"projection": 49152, "ffn_input": 49152, "norm": 49152, "up": 163840,
             "swiglu": 163840, "down": 196608, "moe_out": 49152, "downstream": 196608}
    for name, item in outputs.items():
        if set(item) != {"kind", "source", "residual_control", "kqv_control", *sizes}:
            raise RunnerError("layer-1 attention-output residual arm schema mismatch")
        if (item["kind"], item["source"], item["residual_control"], item["kqv_control"]) != ARM_META[name]:
            raise RunnerError("layer-1 attention-output residual arm metadata mismatch")
        for key, size in sizes.items():
            if set(item[key]) != {"path", "bytes", "sha256"} or item[key]["bytes"] != size:
                raise RunnerError("layer-1 attention-output residual payload schema mismatch")


def validate_report(root, model, source_map):
    report = json.loads((root / "strat01_layer1_attention_output_residual_cross_input.json").read_text(encoding="utf-8"))
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "matrices", "outputs",
                "engine_source_sha256", "diagnostic_source_sha256", "compiler_family",
                "projection_arms_completed", "ffn_input_arms_completed", "rmsnorm_arms_completed",
                "up_projection_arms_completed", "propagated_arms_completed", "donor_graph_executions",
                "reference_graph_executions", "timing_or_rate_claim"}
    if (set(report) != required or report["command"] != "--strat01-layer1-attention-output-residual-cross-input" or
            report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False):
        raise RunnerError("layer-1 attention-output residual report schema mismatch")
    expected = {name: {"path": str(path.resolve()), "bytes": size, "sha256": digest} for name, (path, digest, size) in INPUTS.items()}
    expected["topk"] = {"path": str(TOPK.resolve()), "bytes": 128, "sha256": TOPK_SHA}
    matrices = {
        "attention_output": {"name": "blk.1.attn_output.weight", "type": "Q4_K", "shape": [6144, 1536], "offset": 315482112, "file_offset": 321585024, "span": 5308416},
        "ffn_norm": {"name": "blk.1.ffn_norm.weight", "type": "F32", "shape": [1536]},
        "up_experts": {"name": "blk.1.ffn_up_exps.weight", "type": "Q4_K"},
        "down_experts": {"name": "blk.1.ffn_down_exps.weight", "type": "Q6_K"},
    }
    if report["inputs"] != expected or report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA} or report["matrices"] != matrices:
        raise RunnerError("layer-1 attention-output residual identity mismatch")
    counts = (report["projection_arms_completed"], report["ffn_input_arms_completed"], report["rmsnorm_arms_completed"], report["up_projection_arms_completed"], report["propagated_arms_completed"])
    if (report["engine_source_sha256"] != source_map["engine"]["sha256"] or report["diagnostic_source_sha256"] != source_map["header"]["sha256"] or
            report["compiler_family"] != "clang" or counts != (2, 4, 6, 6, 6) or report["donor_graph_executions"] != 0 or
            report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None):
        raise RunnerError("layer-1 attention-output residual execution accounting mismatch")
    arm_manifest(report["outputs"]); values = {}
    for name, item in report["outputs"].items():
        values[name] = {key: base.load_f32(base.contained(root, item[key]["path"], item[key]["bytes"], item[key]["sha256"], f"{name} {key}"), item[key]["bytes"] // 4, f"{name} {key}") for key in ("projection", "ffn_input", "norm", "up", "swiglu", "down", "moe_out", "downstream")}
    return values, report


def classify(judgments, reference_replay, c_replay):
    if not judgments[ARM_NAMES[0]]["pass"] or judgments[ARM_NAMES[1]]["pass"]:
        raise RunnerError("layer-1 attention-output residual anchor contradiction")
    if judgments[ARM_NAMES[2]]["pass"] and not judgments[ARM_NAMES[3]]["pass"] and all(reference_replay.values()) and all(c_replay.values()):
        return "LAYER1_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT"
    if not judgments[ARM_NAMES[2]]["pass"]:
        return "LAYER1_ATTN_OUTPUT_PROJECTION_FAILS_EXACT_KQV"
    return "VOID_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT"


def adjudicate(values, report, frozen):
    target = base.load_f32(frozen["ref_target"], 49152, "reference target")
    prior_c = base.load_f32(frozen["prior_c"], 49152, "predecessor C output")
    ref_ffn = base.load_f32(frozen["reference_ffn_input"], 12288, "reference ffn input")
    c_ffn = base.load_f32(frozen["c_ffn_input"], 12288, "C ffn input")
    if values[ARM_NAMES[0]]["ffn_input"].tobytes() != ref_ffn.tobytes() or values[ARM_NAMES[1]]["ffn_input"].tobytes() != c_ffn.tobytes():
        raise RunnerError("layer-1 attention-output residual captured input mismatch")
    if values[ARM_NAMES[0]]["downstream"].tobytes() != target.tobytes() or values[ARM_NAMES[1]]["downstream"].tobytes() != prior_c.tobytes():
        raise RunnerError("layer-1 attention-output residual captured downstream mismatch")
    judgments = {name: fr.judged_downstream(values[name]["downstream"], target) for name in ARM_NAMES[:4]}
    controls = {name: fr.judged_downstream(values[name]["downstream"], target) for name in ARM_NAMES[4:]}
    if any(item["pass"] for item in controls.values()):
        raise RunnerError("layer-1 attention-output residual planted control passed")
    stages = ("ffn_input", "norm", "up", "swiglu", "down", "moe_out", "downstream")
    reference_replay = {stage: values[ARM_NAMES[2]][stage].tobytes() == values[ARM_NAMES[0]][stage].tobytes() for stage in stages}
    c_replay = {stage: values[ARM_NAMES[3]][stage].tobytes() == values[ARM_NAMES[1]][stage].tobytes() for stage in stages}
    normal_projection = values[ARM_NAMES[0]]["projection"].tobytes()
    if any(values[name]["projection"].tobytes() != normal_projection for name in ARM_NAMES[:5]):
        raise RunnerError("layer-1 attention-output residual normal projection inconsistency")
    if values[ARM_NAMES[5]]["projection"].tobytes() == normal_projection:
        raise RunnerError("layer-1 attention-output residual projection control inert")
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
        raise RunnerError("layer-1 attention-output residual mutation control failed")
    swapped = copy.deepcopy(report["outputs"]); items = list(swapped.items()); items[2], items[3] = items[3], items[2]
    try:
        arm_manifest(dict(items)); label_swap = False
    except RunnerError:
        label_swap = True
    if not label_swap:
        raise RunnerError("layer-1 attention-output residual label swap accepted")
    shapes = {"projection": (8, 1536), "ffn_input": (8, 1536), "norm": (8, 1536), "up": (8, 4, 1280),
              "swiglu": (8, 4, 1280), "down": (8, 4, 1536), "moe_out": (8, 1536), "downstream": (8, 6144)}
    return {
        "status": classify(judgments, reference_replay, c_replay), "downstream": judgments,
        "controls_vs_reference": controls,
        "stage_metrics": {stage: {name: fr.descriptive(values[name][stage], values[ARM_NAMES[0]][stage], shape) for name in ARM_NAMES} for stage, shape in shapes.items()},
        "controls": {"captured_anchors_byte_exact": True, "reference_stage_replay": reference_replay,
                     "c_stage_replay": c_replay, "kqv_inputs_byte_exact": True,
                     "normal_projection_consistent": True, "projection_control_changed": True,
                     "schedule_twins_byte_exact": True, "mutated_inputs_refused": mutations,
                     "label_swap_rejected": label_swap},
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve()
    out = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    out.mkdir(parents=True); started, tick = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status, errors, commands, report, result, source_map = "VOID_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT", [], {}, {}, {"status": "NOT_RUN"}, {}
    compiler, binary, invocations = shutil.which("clang"), None, 0
    counters = {"projection_arms_completed": 0, "ffn_input_arms_completed": 0, "rmsnorm_arms_completed": 0, "up_projection_arms_completed": 0, "propagated_arms_completed": 0}
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None, "opened": False}
    try:
        source_map = sources()
        if not compiler:
            raise RunnerError("clang unavailable")
        commands["clang_version"] = base.run_command([compiler, "--version"], out, "clang_version", 30); base.require_ok(commands["clang_version"], "clang")
        binary = out / "engine_layer1_attention_output_residual_cross_input.exe"
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], out, "compile", 600); base.require_ok(commands["compile"], "compile")
        selftests = (
            ("attention_output_residual_selftest", "--strat01-layer1-attention-output-residual-cross-input-selftest"),
            ("ffn_rmsnorm_selftest", "--strat01-layer1-ffn-rmsnorm-cross-input-selftest"),
            ("routed_up_selftest", "--strat01-layer1-routed-up-projection-cross-input-selftest"),
            ("swiglu_selftest", "--strat01-layer1-routed-swiglu-component-cross-input-selftest"),
            ("q6_selftest", "--strat01-layer1-q6-cross-input-selftest"),
            ("propagation_selftest", "--strat01-layer1-routed-q6-residual-propagation-selftest"),
            ("legacy_selftest", "--kselftest"),
        )
        for label, flag in selftests:
            commands[label] = base.run_command([str(binary), flag], out, label, 300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer1_attention_output_residual_cross_input"], out, "python_tests", 300); base.require_ok(commands["python_tests"], "tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean(source_map)
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise RunnerError("artifact mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA, "opened": True})
            frozen = evidence(); root = out / "diagnostic"; root.mkdir()
            command = [str(binary), "--strat01-layer1-attention-output-residual-cross-input", str(model), "--topk", str(frozen["topk"]), "--ref-residual", str(frozen["reference_residual"]), "--c-residual", str(frozen["c_residual"]), "--ref-kqv", str(frozen["reference_kqv"]), "--c-kqv", str(frozen["c_kqv"]), "--ref-ffn", str(frozen["reference_ffn_input"]), "--c-ffn", str(frozen["c_ffn_input"]), "--ref-gate", str(frozen["reference_gate"]), "--ref-q", str(frozen["ref_q"]), "--ref-k", str(frozen["ref_k"]), "--ref-shared", str(frozen["ref_shared_out"]), "--ref-weights", str(frozen["ref_weights"]), "--out-dir", str(root)]
            invocations = 1; commands["diagnostic"] = base.run_command(command, out, "diagnostic", 21600)
            failure = root / "strat01_layer1_attention_output_residual_cross_input.json"
            if commands["diagnostic"]["returncode"] and failure.is_file():
                try:
                    data = json.loads(failure.read_text(encoding="utf-8")); counters = {key: int(data.get(key, 0)) for key in counters}
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            base.require_ok(commands["diagnostic"], "diagnostic")
            counters = {"projection_arms_completed": 2, "ffn_input_arms_completed": 4, "rmsnorm_arms_completed": 6, "up_projection_arms_completed": 6, "propagated_arms_completed": 6}
            source_map = sources(); values, report = validate_report(root, model, source_map); result = adjudicate(values, report, frozen); status = result["status"]
    except (RunnerError, base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record = {
        "schema": "strat01_layer1_attention_output_residual_cross_input_v1", "status": status, "errors": errors,
        "diagnostic_invocations": invocations, **counters, "donor_graph_executions": 0, "reference_graph_executions": 0,
        "adjudication": result, "c_report": report,
        "provenance": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - tick,
                       "git_head_observed" if args.apparatus_only else "git_head": base.git_value(["git", "rev-parse", "HEAD"]),
                       "source_hashes": source_map, "predecessor": {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA},
                       "predecessor_output": {"path": str(PREDECESSOR_C), "sha256": PREDECESSOR_C_SHA}, "artifact": artifact,
                       "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd()},
                       "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands},
        "non_claims": ["graph execution", "production repair", "attention producer rerun", "later layers", "quality/RAM/rate"],
    }
    base.write_json(out / "adjudication.json", record); print(json.dumps({"status": status, "output": str(out), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
