#!/usr/bin/env python3
"""Execute the frozen layer-1 routed-MoE component cross-input diagnostic."""
from __future__ import annotations

import argparse, copy, hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_layer1_ffn_output_component_cross_input as fc
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE = fc.ENGINE
DEFAULT_MODEL = fc.DEFAULT_MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer1_routed_moe_component_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer1_routed_moe_component_cross_input.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_routed_moe_component_cross_input_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_routed_moe_component_cross_input_apparatus_20260924"
PREDECESSOR = fc.DEFAULT_OUTPUT / "adjudication.json"
PREDECESSOR_SHA = "4d03a081ab06cf695c4569c4d8ab615d2e2b88a2bfc520874da89e32685e984a"
PREDECESSOR_OUTPUT = fc.DEFAULT_OUTPUT / "diagnostic/cross_c_moe_ref_shared.f32le"
PREDECESSOR_OUTPUT_SHA = "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"
REF_ROOT, C_ROOT = fc.REF_ROOT, fc.C_ROOT

ARM_NAMES = (
    "captured_ref_moe_out", "captured_c_moe_out",
    "computed_ref_down_ref_weights", "computed_c_down_c_weights",
    "cross_c_down_ref_weights", "cross_ref_down_c_weights",
    "control_ref_down_token7_negated",
)
ARM_META = {
    ARM_NAMES[0]: ("captured", "reference", "reference", False),
    ARM_NAMES[1]: ("captured", "c", "c", False),
    ARM_NAMES[2]: ("computed", "reference", "reference", False),
    ARM_NAMES[3]: ("computed", "c", "c", False),
    ARM_NAMES[4]: ("computed", "c", "reference", False),
    ARM_NAMES[5]: ("computed", "reference", "c", False),
    ARM_NAMES[6]: ("computed", "reference", "reference", True),
}
VALID_STATUSES = {
    "LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT",
    "LAYER1_NORMALIZED_ROUTER_WEIGHT_RESIDUAL_SUFFICIENT",
    "LAYER1_ROUTED_MOE_INPUT_RESIDUALS_INDEPENDENTLY_SUFFICIENT",
    "LAYER1_ROUTED_MOE_INPUT_RESIDUALS_JOINTLY_SUFFICIENT",
}
PINNED = {
    "ref_q": fc.PINNED["ref_q"], "ref_k": fc.PINNED["ref_k"], "ref_target": fc.PINNED["ref_target"],
    "ref_ffn_inp": fc.PINNED["ref_ffn_inp"], "ref_shared_out": fc.PINNED["ref_shared_out"],
    "ref_down": (REF_ROOT / "prefill8/ffn_moe_down-1.full.f32le", "d19ce37968e1c2c6b04dffbc56ebba7ed25d704db2447e45031786b4710bb0af", 196608),
    "ref_weights": (REF_ROOT / "prefill8/ffn_moe_weights_norm-1.full.f32le", "075f2321d029c8ca30385a5cc0d8555b8a70cf0e1fec1e4c0108c11447572f03", 128),
    "ref_weighted": (REF_ROOT / "prefill8/ffn_moe_weighted-1.full.f32le", "875d7a321cfb4c28343edc04bdd1f9940ce014fcfa2aa1306922291c014009b2", 196608),
    "ref_moe_out": fc.PINNED["ref_moe_out"],
    "c_down": (C_ROOT / "prefill8_ffn_moe_down-1.f32", "c30893f3dd0a751f9312c16fc09638e4f0b55265a5dccaba7205edf3caccc2ce", 196608),
    "c_weights": (C_ROOT / "prefill8_ffn_moe_weights_norm-1.f32", "30cabc5a31c66444fa63734e94c4072a99868920320954b0b466f182f185707a", 128),
    "c_weighted": (C_ROOT / "prefill8_ffn_moe_weighted-1.f32", "9288f02aad131b78eff128ddbd64a4933c659ccffb80f2978d211c34c7c26a94", 196608),
    "c_moe_out": fc.PINNED["c_moe_out"],
}
TWINS = {
    "ref_q": fc.TWINS["ref_q"], "ref_k": fc.TWINS["ref_k"], "ref_target": fc.TWINS["ref_target"],
    "ref_ffn_inp": fc.TWINS["ref_ffn_inp"], "ref_shared_out": fc.TWINS["ref_shared_out"],
    "ref_down": REF_ROOT / "cached7p1/ffn_moe_down-1.full.f32le", "ref_weights": REF_ROOT / "cached7p1/ffn_moe_weights_norm-1.full.f32le",
    "ref_weighted": REF_ROOT / "cached7p1/ffn_moe_weighted-1.full.f32le", "ref_moe_out": fc.TWINS["ref_moe_out"],
    "c_down": C_ROOT / "cached7p1_ffn_moe_down-1.f32", "c_weights": C_ROOT / "cached7p1_ffn_moe_weights_norm-1.f32",
    "c_weighted": C_ROOT / "cached7p1_ffn_moe_weighted-1.f32", "c_moe_out": fc.TWINS["c_moe_out"],
}
TOPK_SHA = "557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95"
TOPK = (
    REF_ROOT / "prefill8/ffn_moe_topk-1.full.i32le", REF_ROOT / "cached7p1/ffn_moe_topk-1.full.i32le",
    C_ROOT / "prefill8_ffn_moe_topk-1.i32", C_ROOT / "cached7p1_ffn_moe_topk-1.i32",
)
EXPECTED_WEIGHTED_SHA = {
    ARM_NAMES[2]: PINNED["ref_weighted"][1], ARM_NAMES[3]: PINNED["c_weighted"][1],
    ARM_NAMES[4]: "c26abad90be611b33824373d48b3d016248099884400d499ffe6bdcfdd582de5",
    ARM_NAMES[5]: "1fb97c90a161cbc09968177efcb29efe61a30598618893347ca7405fa61760e9",
}
EXPECTED_MOE_SHA = {
    ARM_NAMES[2]: PINNED["ref_moe_out"][1], ARM_NAMES[3]: PINNED["c_moe_out"][1],
    ARM_NAMES[4]: "a63d05e6532b6a6c71aa6b09520c4fb8cf076546e6bd293bc97c3aa06c205d81",
    ARM_NAMES[5]: "f48dfa00abf5fdeaf0b5cb06459102bc81e00e0543ccdb7ace96c69271e392e6",
}
FROZEN = {"nrmse": 0.002642086799405445, "normalized_max": 0.004687597394884091}


class RunnerError(RuntimeError):
    pass


def sources():
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL, "engine": ENGINE, "header": HEADER,
        "ffn_component_header": fc.HEADER, "ffn_component_runner": Path(fc.__file__).resolve(),
        "terminal_header": fc.tc.HEADER, "attn_norm_header": fc.tc.an.CROSS_HEADER,
        "kva_header": fc.tc.an.KVA_HEADER, "rms_header": fc.tc.an.RMS_HEADER,
        "rung2a": fc.tc.an.RUNG2A, "rung2c": fc.tc.an.RUNG2C, "base_runner": Path(base.__file__).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise RunnerError("missing routed-MoE component source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def clean(source_map):
    rel = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in source_map.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, rel)], cwd=ROOT, check=False).returncode:
        raise RunnerError("routed-MoE component sources differ from HEAD")
    for path in rel:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"untracked routed-MoE component source: {path}")


def frozen_evidence():
    evidence = {}
    for name, (path, digest, size) in PINNED.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise RunnerError(f"routed-MoE component input mismatch: {name}")
        twin = TWINS[name]
        if not twin.is_file() or twin.stat().st_size != size or base.sha256_file(twin) != digest:
            raise RunnerError(f"routed-MoE component twin mismatch: {name}")
        evidence[name] = path.resolve(strict=True)
    for path in TOPK:
        if not path.is_file() or path.stat().st_size != 128 or base.sha256_file(path) != TOPK_SHA:
            raise RunnerError("routed-MoE top-k identity mismatch")
    for path, digest, label in ((PREDECESSOR, PREDECESSOR_SHA, "FFN-component predecessor"), (PREDECESSOR_OUTPUT, PREDECESSOR_OUTPUT_SHA, "predecessor downstream output")):
        if not path.is_file() or base.sha256_file(path) != digest:
            raise RunnerError(label + " mismatch")
    evidence["prior"] = PREDECESSOR_OUTPUT.resolve(strict=True)
    return evidence


def arm_manifest(outputs):
    if not isinstance(outputs, dict) or tuple(outputs) != ARM_NAMES:
        raise RunnerError("routed-MoE component arm labels/order mismatch")
    sizes = (("weighted", 196608), ("moe_out", 49152), ("ffn_out", 49152), ("l_out", 49152), ("norm", 49152), ("projection", 18432), ("prefix", 16384), ("downstream", 196608))
    for name, item in outputs.items():
        if set(item) != {"kind", "down_origin", "weights_origin", "down_control", *(key for key, _ in sizes)}:
            raise RunnerError("routed-MoE component arm schema mismatch")
        if (item["kind"], item["down_origin"], item["weights_origin"], item["down_control"]) != ARM_META[name]:
            raise RunnerError("routed-MoE component arm metadata mismatch")
        for key, size in sizes:
            if set(item[key]) != {"path", "bytes", "sha256"} or item[key]["bytes"] != size:
                raise RunnerError("routed-MoE component payload schema mismatch")


def validate_report(root, model, source_map):
    report = json.loads((root / "strat01_layer1_routed_moe_component_cross_input.json").read_text(encoding="utf-8"))
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "matrices", "outputs", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-layer1-routed-moe-component-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise RunnerError("routed-MoE component report schema mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise RunnerError("routed-MoE component model mismatch")
    order = ("ref_q", "ref_k", "ref_ffn_inp", "ref_shared_out", "ref_down", "ref_weights", "ref_weighted", "ref_moe_out", "c_down", "c_weights", "c_weighted", "c_moe_out")
    expected = {name: {"path": str(PINNED[name][0].resolve()), "bytes": PINNED[name][2], "sha256": PINNED[name][1]} for name in order}
    if report["inputs"] != expected:
        raise RunnerError("routed-MoE component inputs mismatch")
    matrices = {"attn_norm": {"name": "blk.2.attn_norm.weight", "type": "F32", "shape": [1536], "offset": 578811136, "file_offset": 584914048, "span": 6144}, "projection": {"name": "blk.2.attn_kv_a_mqa.weight", "type": "Q4_K", "shape": [1536, 576]}, "kv_norm": {"name": "blk.2.attn_kv_a_norm.weight", "type": "F32", "shape": [512]}, "v_b": {"name": "blk.2.attn_v_b.weight", "type": "Q4_K", "shape": [512, 192, 32]}}
    if report["matrices"] != matrices:
        raise RunnerError("routed-MoE component descriptors mismatch")
    if report["engine_source_sha256"] != source_map["engine"]["sha256"] or report["diagnostic_source_sha256"] != source_map["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise RunnerError("routed-MoE component provenance mismatch")
    arm_manifest(report["outputs"]); values = {}
    for name, item in report["outputs"].items():
        values[name] = {key: base.load_f32(base.contained(root, item[key]["path"], item[key]["bytes"], item[key]["sha256"], name + " " + key), item[key]["bytes"] // 4, name + " " + key) for key in ("weighted", "moe_out", "ffn_out", "l_out", "norm", "projection", "prefix", "downstream")}
    return values, report


def classify(judgments):
    if not judgments[ARM_NAMES[0]]["pass"] or judgments[ARM_NAMES[1]]["pass"] or judgments[ARM_NAMES[3]]["pass"]:
        raise RunnerError("routed-MoE component anchor contradiction")
    down_fails, weights_fail = not judgments[ARM_NAMES[4]]["pass"], not judgments[ARM_NAMES[5]]["pass"]
    if down_fails and weights_fail:
        return "LAYER1_ROUTED_MOE_INPUT_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
    if down_fails:
        return "LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT"
    if weights_fail:
        return "LAYER1_NORMALIZED_ROUTER_WEIGHT_RESIDUAL_SUFFICIENT"
    return "LAYER1_ROUTED_MOE_INPUT_RESIDUALS_JOINTLY_SUFFICIENT"


def descriptive(candidate, reference, shape):
    result = base.metrics(candidate, reference); a, b = np.asarray(candidate).reshape(shape), np.asarray(reference).reshape(shape)
    result["per_token"] = [dict(base.metrics(a[i].ravel(), b[i].ravel()), token=i) for i in range(shape[0])]
    return result


def adjudicate(values, report, evidence):
    target = base.load_f32(evidence["ref_target"], 49152, "reference target"); prior = base.load_f32(evidence["prior"], 49152, "predecessor output")
    ref_moe = base.load_f32(evidence["ref_moe_out"], 12288, "reference routed output"); c_moe = base.load_f32(evidence["c_moe_out"], 12288, "C routed output")
    if values[ARM_NAMES[0]]["moe_out"].tobytes() != ref_moe.tobytes() or values[ARM_NAMES[1]]["moe_out"].tobytes() != c_moe.tobytes():
        raise RunnerError("routed-MoE component captured anchor mismatch")
    for key in ("weighted", "moe_out", "ffn_out", "l_out", "norm", "projection", "prefix", "downstream"):
        if values[ARM_NAMES[2]][key].tobytes() != values[ARM_NAMES[0]][key].tobytes():
            raise RunnerError("routed-MoE reference replay mismatch")
        if values[ARM_NAMES[3]][key].tobytes() != values[ARM_NAMES[1]][key].tobytes():
            raise RunnerError("routed-MoE C replay mismatch")
    for name, digest in EXPECTED_WEIGHTED_SHA.items():
        if hashlib.sha256(values[name]["weighted"].astype("<f4", copy=False).tobytes()).hexdigest() != digest:
            raise RunnerError("routed-MoE weighted mismatch: " + name)
    for name, digest in EXPECTED_MOE_SHA.items():
        if hashlib.sha256(values[name]["moe_out"].astype("<f4", copy=False).tobytes()).hexdigest() != digest:
            raise RunnerError("routed-MoE sum mismatch: " + name)
    if values[ARM_NAMES[0]]["downstream"].tobytes() != target.tobytes() or values[ARM_NAMES[1]]["downstream"].tobytes() != prior.tobytes():
        raise RunnerError("routed-MoE downstream anchor mismatch")
    judgments = {name: base.judged(values[name]["downstream"], target) for name in ARM_NAMES[:6]}
    for metric, expected in FROZEN.items():
        if abs(judgments[ARM_NAMES[1]][metric] - expected) > 1e-12:
            raise RunnerError("routed-MoE frozen metric mismatch")
    control = base.judged(values[ARM_NAMES[6]]["downstream"], target)
    if control["pass"]:
        raise RunnerError("routed-MoE planted control failed")
    mutations = {}
    for name, (_, digest, size) in PINNED.items():
        data = bytearray(evidence[name].read_bytes()); data[len(data) // 2] ^= 1; mutations[name] = not base.identity_matches(bytes(data), size, digest)
    topk_data = bytearray(TOPK[0].read_bytes()); topk_data[len(topk_data) // 2] ^= 1; mutations["topk"] = not base.identity_matches(bytes(topk_data), 128, TOPK_SHA)
    if not all(mutations.values()):
        raise RunnerError("routed-MoE mutation failed")
    swapped = copy.deepcopy(report["outputs"]); items = list(swapped.items()); items[4], items[5] = items[5], items[4]
    try:
        arm_manifest(dict(items)); label_swap_rejected = False
    except RunnerError:
        label_swap_rejected = True
    if not label_swap_rejected:
        raise RunnerError("routed-MoE label swap failed")
    component_metrics = {
        "c_down_vs_ref": descriptive(base.load_f32(evidence["c_down"], 49152, "C expert down"), base.load_f32(evidence["ref_down"], 49152, "reference expert down"), (8, 4, 1536)),
        "c_weights_vs_ref": descriptive(base.load_f32(evidence["c_weights"], 32, "C normalized weights"), base.load_f32(evidence["ref_weights"], 32, "reference normalized weights"), (8, 4)),
        "c_moe_out_vs_ref": descriptive(c_moe, ref_moe, (8, 1536)),
    }
    return {"status": classify(judgments), "arms_vs_reference": judgments, "routed_output_metrics": {name: descriptive(values[name]["moe_out"], ref_moe, (8, 1536)) for name in ARM_NAMES[:6]}, "component_metrics": component_metrics, "control_vs_reference": control, "controls": {"reference_replay_all_stages_byte_exact": True, "c_replay_all_stages_byte_exact": True, "captured_anchors_byte_exact": True, "schedule_twins_byte_exact": True, "topk_all_schedules_byte_exact": True, "mutated_inputs_refused": mutations, "label_swap_rejected": label_swap_rejected}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, default=DEFAULT_MODEL); parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve(); out = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    out.mkdir(parents=True); started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter(); status = "VOID_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT"; errors = []; commands = {}; report = {}; adjudication = {"status": "NOT_RUN"}; source_map = {}; compiler = shutil.which("clang"); binary = None; invocations = 0
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None, "opened": False}
    try:
        source_map = sources()
        if not compiler:
            raise RunnerError("clang unavailable")
        commands["clang_version"] = base.run_command([compiler, "--version"], out, "clang_version", 30); base.require_ok(commands["clang_version"], "clang")
        binary = out / "engine_layer1_routed_moe_component.exe"; commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], out, "compile", 600); base.require_ok(commands["compile"], "compile")
        tests = (("routed_moe_component_selftest", "--strat01-layer1-routed-moe-component-cross-input-selftest"), ("ffn_output_component_selftest", "--strat01-layer1-ffn-output-component-cross-input-selftest"), ("terminal_component_selftest", "--strat01-layer1-terminal-component-cross-input-selftest"), ("attn_norm_selftest", "--strat01-layer2-attn-rmsnorm-cross-input-selftest"), ("kva_selftest", "--strat01-layer2-kv-a-projection-cross-input-selftest"), ("rms_selftest", "--strat01-layer2-kv-rmsnorm-cross-input-selftest"), ("partition_selftest", "--strat01-layer2-kv-partition-cross-input-selftest"), ("legacy_selftest", "--kselftest"))
        for label, flag in tests:
            commands[label] = base.run_command([str(binary), flag], out, label, 300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer1_routed_moe_component_cross_input"], out, "python_tests", 300); base.require_ok(commands["python_tests"], "tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean(source_map)
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise RunnerError("artifact mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA, "opened": True}); evidence = frozen_evidence(); root = out / "diagnostic"; root.mkdir()
            command = [str(binary), "--strat01-layer1-routed-moe-component-cross-input", str(model), "--ref-q", str(evidence["ref_q"]), "--ref-k", str(evidence["ref_k"]), "--ref-ffn-inp", str(evidence["ref_ffn_inp"]), "--ref-shared-out", str(evidence["ref_shared_out"]), "--ref-down", str(evidence["ref_down"]), "--ref-weights", str(evidence["ref_weights"]), "--ref-weighted", str(evidence["ref_weighted"]), "--ref-moe-out", str(evidence["ref_moe_out"]), "--c-down", str(evidence["c_down"]), "--c-weights", str(evidence["c_weights"]), "--c-weighted", str(evidence["c_weighted"]), "--c-moe-out", str(evidence["c_moe_out"]), "--out-dir", str(root)]
            invocations = 1; commands["diagnostic"] = base.run_command(command, out, "diagnostic", 21600); base.require_ok(commands["diagnostic"], "diagnostic"); source_map = sources(); values, report = validate_report(root, model, source_map); adjudication = adjudicate(values, report, evidence); status = adjudication["status"]
    except (RunnerError, base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record = {"schema": "strat01_layer1_routed_moe_component_cross_input_v1", "status": status, "errors": errors, "diagnostic_invocations": invocations, "donor_graph_executions": 0, "reference_graph_executions": 0, "adjudication": adjudication, "c_report": report, "provenance": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - tick, "git_head_observed" if args.apparatus_only else "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "source_hashes": source_map, "predecessor": {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}, "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd()}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}, "non_claims": ["repair", "producer/router/expert rerun", "uncaptured operator", "later layers", "quality/RAM/rate"]}
    base.write_json(out / "adjudication.json", record); print(json.dumps({"status": status, "output": str(out), "errors": errors}, indent=2)); return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
