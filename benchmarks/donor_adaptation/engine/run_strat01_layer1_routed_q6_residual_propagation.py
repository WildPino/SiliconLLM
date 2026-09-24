#!/usr/bin/env python3
"""Propagate the frozen layer-1 routed-Q6 residual through closed downstream."""
from __future__ import annotations

import argparse, copy, hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_moe_component_cross_input as rm
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE, DEFAULT_MODEL = rm.ENGINE, rm.DEFAULT_MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer1_routed_q6_residual_propagation.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_Q6_RESIDUAL_PROPAGATION_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer1_routed_q6_residual_propagation.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_routed_q6_residual_propagation_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_routed_q6_residual_propagation_apparatus_20260924"
PREDECESSOR = rm.DEFAULT_OUTPUT / "adjudication.json"
PREDECESSOR_SHA = "ac78794d24467b2ca4c3fce092b969bfd9fa9a2698d2227d5ea5e5c67640ab59"
PREDECESSOR_OUTPUT = rm.DEFAULT_OUTPUT / "diagnostic/cross_c_down_ref_weights.f32le"
PREDECESSOR_OUTPUT_SHA = "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"
OLD_Q6_ADJ = HERE / "results/strat01_gigachat_engine_layer1_q6_cross_input_20260923/adjudication.json"
OLD_Q6_ADJ_SHA = "db9b478e208c35460d4bd1ae54eb56cfb2a8106361b19f0b7f1f024699509148"
Q6_REF_DOWN = HERE / "results/strat01_gigachat_engine_layer1_q6_cross_input_20260923/diagnostic/reference_routed_current_q6.f32le"
Q6_REF_DOWN_SHA = "609aa170ac5acf0cec49e403e28e1dde831d14cb76f1de35d1ce51e39e9df91f"

ARM_NAMES = ("captured_ref_down", "captured_q6_ref_swiglu_down", "captured_c_down", "control_q6_ref_down_token7_negated")
ARM_META = {ARM_NAMES[0]: ("reference", False), ARM_NAMES[1]: ("q6_reference_swiglu", False), ARM_NAMES[2]: ("c", False), ARM_NAMES[3]: ("q6_reference_swiglu", True)}
VALID_STATUSES = {"LAYER1_ROUTED_SWIGLU_RESIDUAL_SUFFICIENT", "LAYER1_ROUTED_Q6_RESIDUAL_SUFFICIENT"}
PINNED = {
    "ref_q": rm.PINNED["ref_q"], "ref_k": rm.PINNED["ref_k"], "ref_target": rm.PINNED["ref_target"],
    "ref_ffn_inp": rm.PINNED["ref_ffn_inp"], "ref_shared_out": rm.PINNED["ref_shared_out"],
    "ref_weights": rm.PINNED["ref_weights"], "ref_down": rm.PINNED["ref_down"], "c_down": rm.PINNED["c_down"],
}
TWINS = {name: rm.TWINS[name] for name in PINNED}
EXPECTED_WEIGHTED_SHA = {ARM_NAMES[0]: "875d7a321cfb4c28343edc04bdd1f9940ce014fcfa2aa1306922291c014009b2", ARM_NAMES[1]: "143a7a89241470098ffb63e19aa4fdde330a95b488530001d7c02b16b31752d3", ARM_NAMES[2]: "c26abad90be611b33824373d48b3d016248099884400d499ffe6bdcfdd582de5"}
EXPECTED_MOE_SHA = {ARM_NAMES[0]: "9d09e5582483a471ec57712ad42b5b868f2ee1715f186e1324f9904ebebb08c8", ARM_NAMES[1]: "57106eb8eb23cf38da05165b522b5c0cc00b6fbc899bc9d306dc1e665093f4b6", ARM_NAMES[2]: "a63d05e6532b6a6c71aa6b09520c4fb8cf076546e6bd293bc97c3aa06c205d81"}
FROZEN = {"nrmse": 0.002642086799405445, "normalized_max": 0.004687597394884091}


class RunnerError(RuntimeError):
    pass


def sources():
    paths = {"runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL, "engine": ENGINE, "header": HEADER, "routed_component_header": rm.HEADER, "routed_component_runner": Path(rm.__file__).resolve(), "ffn_component_header": rm.fc.HEADER, "terminal_header": rm.fc.tc.HEADER, "attn_norm_header": rm.fc.tc.an.CROSS_HEADER, "kva_header": rm.fc.tc.an.KVA_HEADER, "rms_header": rm.fc.tc.an.RMS_HEADER, "rung2a": rm.fc.tc.an.RUNG2A, "rung2c": rm.fc.tc.an.RUNG2C, "base_runner": Path(base.__file__).resolve()}
    if any(not path.is_file() for path in paths.values()):
        raise RunnerError("missing routed-Q6 propagation source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def clean(source_map):
    rel = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in source_map.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, rel)], cwd=ROOT, check=False).returncode:
        raise RunnerError("routed-Q6 propagation sources differ from HEAD")
    for path in rel:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"untracked routed-Q6 propagation source: {path}")


def frozen_evidence():
    evidence = {}
    for name, (path, digest, size) in PINNED.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise RunnerError(f"routed-Q6 propagation input mismatch: {name}")
        twin = TWINS[name]
        if not twin.is_file() or twin.stat().st_size != size or base.sha256_file(twin) != digest:
            raise RunnerError(f"routed-Q6 propagation twin mismatch: {name}")
        evidence[name] = path.resolve(strict=True)
    for path, digest, size, label in ((Q6_REF_DOWN, Q6_REF_DOWN_SHA, 196608, "old Q6 reference-input output"), (PREDECESSOR, PREDECESSOR_SHA, None, "predecessor"), (PREDECESSOR_OUTPUT, PREDECESSOR_OUTPUT_SHA, 196608, "predecessor output"), (OLD_Q6_ADJ, OLD_Q6_ADJ_SHA, None, "old Q6 adjudication")):
        if not path.is_file() or (size is not None and path.stat().st_size != size) or base.sha256_file(path) != digest:
            raise RunnerError(label + " mismatch")
    evidence["q6_ref_down"] = Q6_REF_DOWN.resolve(strict=True); evidence["prior"] = PREDECESSOR_OUTPUT.resolve(strict=True)
    return evidence


def arm_manifest(outputs):
    if not isinstance(outputs, dict) or tuple(outputs) != ARM_NAMES:
        raise RunnerError("routed-Q6 propagation arm labels/order mismatch")
    sizes = (("weighted", 196608), ("moe_out", 49152), ("ffn_out", 49152), ("l_out", 49152), ("norm", 49152), ("projection", 18432), ("prefix", 16384), ("downstream", 196608))
    for name, item in outputs.items():
        if set(item) != {"origin", "down_control", *(key for key, _ in sizes)} or (item["origin"], item["down_control"]) != ARM_META[name]:
            raise RunnerError("routed-Q6 propagation arm schema/metadata mismatch")
        for key, size in sizes:
            if set(item[key]) != {"path", "bytes", "sha256"} or item[key]["bytes"] != size:
                raise RunnerError("routed-Q6 propagation payload schema mismatch")


def validate_report(root, model, source_map):
    report = json.loads((root / "strat01_layer1_routed_q6_residual_propagation.json").read_text(encoding="utf-8"))
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "matrices", "outputs", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "q6_executions", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-layer1-routed-q6-residual-propagation" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise RunnerError("routed-Q6 propagation report schema mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise RunnerError("routed-Q6 propagation model mismatch")
    order = ("ref_q", "ref_k", "ref_ffn_inp", "ref_shared_out", "ref_weights", "ref_down", "q6_ref_down", "c_down")
    spec = {**PINNED, "q6_ref_down": (Q6_REF_DOWN, Q6_REF_DOWN_SHA, 196608)}
    expected = {name: {"path": str(spec[name][0].resolve()), "bytes": spec[name][2], "sha256": spec[name][1]} for name in order}
    if report["inputs"] != expected:
        raise RunnerError("routed-Q6 propagation inputs mismatch")
    matrices = {"attn_norm": {"name": "blk.2.attn_norm.weight", "type": "F32", "shape": [1536], "offset": 578811136, "file_offset": 584914048, "span": 6144}, "projection": {"name": "blk.2.attn_kv_a_mqa.weight", "type": "Q4_K", "shape": [1536, 576]}, "kv_norm": {"name": "blk.2.attn_kv_a_norm.weight", "type": "F32", "shape": [512]}, "v_b": {"name": "blk.2.attn_v_b.weight", "type": "Q4_K", "shape": [512, 192, 32]}}
    if report["matrices"] != matrices or report["q6_executions"] != 0 or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise RunnerError("routed-Q6 propagation execution/provenance mismatch")
    if report["engine_source_sha256"] != source_map["engine"]["sha256"] or report["diagnostic_source_sha256"] != source_map["header"]["sha256"] or report["compiler_family"] != "clang":
        raise RunnerError("routed-Q6 propagation source provenance mismatch")
    arm_manifest(report["outputs"]); values = {}
    for name, item in report["outputs"].items():
        values[name] = {key: base.load_f32(base.contained(root, item[key]["path"], item[key]["bytes"], item[key]["sha256"], name + " " + key), item[key]["bytes"] // 4, name + " " + key) for key in ("weighted", "moe_out", "ffn_out", "l_out", "norm", "projection", "prefix", "downstream")}
    return values, report


def descriptive(candidate, reference, shape):
    result = base.metrics(candidate, reference); a, b = np.asarray(candidate).reshape(shape), np.asarray(reference).reshape(shape)
    result["per_token"] = [dict(base.metrics(a[i].ravel(), b[i].ravel()), token=i) for i in range(shape[0])]; return result


def adjudicate(values, report, evidence):
    target = base.load_f32(evidence["ref_target"], 49152, "reference target"); prior = base.load_f32(evidence["prior"], 49152, "predecessor output")
    ref_moe = values[ARM_NAMES[0]]["moe_out"]
    for name, digest in EXPECTED_WEIGHTED_SHA.items():
        if hashlib.sha256(values[name]["weighted"].astype("<f4", copy=False).tobytes()).hexdigest() != digest:
            raise RunnerError("routed-Q6 weighted mismatch: " + name)
    for name, digest in EXPECTED_MOE_SHA.items():
        if hashlib.sha256(values[name]["moe_out"].astype("<f4", copy=False).tobytes()).hexdigest() != digest:
            raise RunnerError("routed-Q6 sum mismatch: " + name)
    if values[ARM_NAMES[0]]["downstream"].tobytes() != target.tobytes() or values[ARM_NAMES[2]]["downstream"].tobytes() != prior.tobytes():
        raise RunnerError("routed-Q6 downstream anchor mismatch")
    judgments = {name: base.judged(values[name]["downstream"], target) for name in ARM_NAMES[:3]}
    if not judgments[ARM_NAMES[0]]["pass"] or judgments[ARM_NAMES[2]]["pass"]:
        raise RunnerError("routed-Q6 anchor contradiction")
    for metric, expected in FROZEN.items():
        if abs(judgments[ARM_NAMES[2]][metric] - expected) > 1e-12:
            raise RunnerError("routed-Q6 frozen metric mismatch")
    status = "LAYER1_ROUTED_SWIGLU_RESIDUAL_SUFFICIENT" if judgments[ARM_NAMES[1]]["pass"] else "LAYER1_ROUTED_Q6_RESIDUAL_SUFFICIENT"
    control = base.judged(values[ARM_NAMES[3]]["downstream"], target)
    if control["pass"]:
        raise RunnerError("routed-Q6 planted control failed")
    mutations = {}
    for name, (_, digest, size) in PINNED.items():
        data = bytearray(evidence[name].read_bytes()); data[len(data) // 2] ^= 1; mutations[name] = not base.identity_matches(bytes(data), size, digest)
    data = bytearray(evidence["q6_ref_down"].read_bytes()); data[len(data) // 2] ^= 1; mutations["q6_ref_down"] = not base.identity_matches(bytes(data), 196608, Q6_REF_DOWN_SHA)
    if not all(mutations.values()):
        raise RunnerError("routed-Q6 mutation failed")
    swapped = copy.deepcopy(report["outputs"]); items = list(swapped.items()); items[1], items[2] = items[2], items[1]
    try:
        arm_manifest(dict(items)); label_swap_rejected = False
    except RunnerError:
        label_swap_rejected = True
    if not label_swap_rejected:
        raise RunnerError("routed-Q6 label swap failed")
    ref_down = base.load_f32(evidence["ref_down"], 49152, "reference down")
    return {"status": status, "arms_vs_reference": judgments, "routed_output_metrics": {name: descriptive(values[name]["moe_out"], ref_moe, (8, 1536)) for name in ARM_NAMES[:3]}, "down_payload_metrics": {"q6_reference_swiglu_vs_ref": descriptive(base.load_f32(evidence["q6_ref_down"], 49152, "Q6 reference-input down"), ref_down, (8, 4, 1536)), "c_vs_ref": descriptive(base.load_f32(evidence["c_down"], 49152, "C down"), ref_down, (8, 4, 1536))}, "control_vs_reference": control, "controls": {"reference_anchor_byte_exact": True, "c_downstream_replay_byte_exact": True, "available_schedule_twins_byte_exact": True, "old_q6_payload_bound": True, "q6_executions_zero": True, "mutated_inputs_refused": mutations, "label_swap_rejected": label_swap_rejected}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, default=DEFAULT_MODEL); parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve(); out = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    out.mkdir(parents=True); started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter(); status = "VOID_LAYER1_ROUTED_Q6_RESIDUAL_PROPAGATION"; errors = []; commands = {}; report = {}; adjudication = {"status": "NOT_RUN"}; source_map = {}; compiler = shutil.which("clang"); binary = None; invocations = 0
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None, "opened": False}
    try:
        source_map = sources()
        if not compiler:
            raise RunnerError("clang unavailable")
        commands["clang_version"] = base.run_command([compiler, "--version"], out, "clang_version", 30); base.require_ok(commands["clang_version"], "clang")
        binary = out / "engine_layer1_routed_q6_propagation.exe"; commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], out, "compile", 600); base.require_ok(commands["compile"], "compile")
        tests = (("routed_q6_propagation_selftest", "--strat01-layer1-routed-q6-residual-propagation-selftest"), ("routed_moe_component_selftest", "--strat01-layer1-routed-moe-component-cross-input-selftest"), ("ffn_output_component_selftest", "--strat01-layer1-ffn-output-component-cross-input-selftest"), ("terminal_component_selftest", "--strat01-layer1-terminal-component-cross-input-selftest"), ("attn_norm_selftest", "--strat01-layer2-attn-rmsnorm-cross-input-selftest"), ("kva_selftest", "--strat01-layer2-kv-a-projection-cross-input-selftest"), ("rms_selftest", "--strat01-layer2-kv-rmsnorm-cross-input-selftest"), ("partition_selftest", "--strat01-layer2-kv-partition-cross-input-selftest"), ("legacy_selftest", "--kselftest"))
        for label, flag in tests:
            commands[label] = base.run_command([str(binary), flag], out, label, 300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer1_routed_q6_residual_propagation"], out, "python_tests", 300); base.require_ok(commands["python_tests"], "tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean(source_map)
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise RunnerError("artifact mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA, "opened": True}); evidence = frozen_evidence(); root = out / "diagnostic"; root.mkdir()
            command = [str(binary), "--strat01-layer1-routed-q6-residual-propagation", str(model), "--ref-q", str(evidence["ref_q"]), "--ref-k", str(evidence["ref_k"]), "--ref-ffn-inp", str(evidence["ref_ffn_inp"]), "--ref-shared-out", str(evidence["ref_shared_out"]), "--ref-weights", str(evidence["ref_weights"]), "--ref-down", str(evidence["ref_down"]), "--q6-ref-down", str(evidence["q6_ref_down"]), "--c-down", str(evidence["c_down"]), "--out-dir", str(root)]
            invocations = 1; commands["diagnostic"] = base.run_command(command, out, "diagnostic", 21600); base.require_ok(commands["diagnostic"], "diagnostic"); source_map = sources(); values, report = validate_report(root, model, source_map); adjudication = adjudicate(values, report, evidence); status = adjudication["status"]
    except (RunnerError, base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record = {"schema": "strat01_layer1_routed_q6_residual_propagation_v1", "status": status, "errors": errors, "diagnostic_invocations": invocations, "q6_executions": 0, "donor_graph_executions": 0, "reference_graph_executions": 0, "adjudication": adjudication, "c_report": report, "provenance": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - tick, "git_head_observed" if args.apparatus_only else "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "source_hashes": source_map, "predecessors": {"routed_component": {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}, "old_q6": {"path": str(OLD_Q6_ADJ), "sha256": OLD_Q6_ADJ_SHA}}, "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd()}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}, "non_claims": ["Q6 execution/repair", "producer/router/expert rerun", "later layers", "quality/RAM/rate"]}
    base.write_json(out / "adjudication.json", record); print(json.dumps({"status": status, "output": str(out), "errors": errors}, indent=2)); return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
