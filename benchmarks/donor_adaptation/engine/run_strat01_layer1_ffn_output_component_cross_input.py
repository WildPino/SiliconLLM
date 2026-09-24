#!/usr/bin/env python3
"""Execute the frozen layer-1 FFN-output component cross-input diagnostic."""
from __future__ import annotations

import argparse, copy, hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_layer1_terminal_component_cross_input as tc
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE = tc.ENGINE
DEFAULT_MODEL = tc.DEFAULT_MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer1_ffn_output_component_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer1_ffn_output_component_cross_input.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_ffn_output_component_cross_input_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_ffn_output_component_cross_input_apparatus_20260924"
PREDECESSOR = tc.DEFAULT_OUTPUT / "adjudication.json"
PREDECESSOR_SHA = "9bdda63cabb7bfe00d7e9dbf76cbead7216aa03064e816cdfa62748fe314ff98"
PREDECESSOR_OUTPUT = tc.DEFAULT_OUTPUT / "diagnostic/cross_ref_inp_c_out.f32le"
PREDECESSOR_OUTPUT_SHA = "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"
REF_ROOT = HERE / "results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference"
C_ROOT = HERE / "results/strat01_gigachat_engine_post_f16_swiglu_production_integration_20260924/c_engine"

ARM_NAMES = (
    "captured_ref_ffn_out", "captured_c_ffn_out",
    "computed_ref_moe_ref_shared", "computed_c_moe_c_shared",
    "cross_c_moe_ref_shared", "cross_ref_moe_c_shared",
    "control_ref_moe_token7_negated",
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
    "LAYER1_ROUTED_MOE_OUTPUT_RESIDUAL_SUFFICIENT",
    "LAYER1_SHARED_EXPERT_OUTPUT_RESIDUAL_SUFFICIENT",
    "LAYER1_FFN_OUTPUT_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT",
    "LAYER1_FFN_OUTPUT_COMPONENT_RESIDUALS_JOINTLY_SUFFICIENT",
}
PINNED = {
    "ref_q": tc.PINNED["ref_q"],
    "ref_k": tc.PINNED["ref_k"],
    "ref_target": tc.PINNED["ref_target"],
    "ref_ffn_inp": (REF_ROOT / "prefill8/ffn_inp-1.full.f32le", "99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8", 49152),
    "ref_moe_out": (REF_ROOT / "prefill8/ffn_moe_out-1.full.f32le", "9d09e5582483a471ec57712ad42b5b868f2ee1715f186e1324f9904ebebb08c8", 49152),
    "ref_shared_out": (REF_ROOT / "prefill8/ffn_shexp-1.full.f32le", "7bc8b2104cacb7742e57df0c34a1af64fadbff8cc61006ed1aedc034a030ddae", 49152),
    "ref_ffn_out": (REF_ROOT / "prefill8/ffn_out-1.full.f32le", "e8cdbbf3154447c90fa2dbddff5a454bd37200091fa2c5aaf7021104e76c104e", 49152),
    "c_moe_out": (C_ROOT / "prefill8_ffn_moe_out-1.f32", "9d0c6f49f7f6c8ffc68f442d2138e0b30b43a9791e8f861088e02cce882887dc", 49152),
    "c_shared_out": (C_ROOT / "prefill8_ffn_shexp-1.f32", "2fc5d9d6248452269bb6850571559a246bd9703ab984210cffdf0f15e92bbe10", 49152),
    "c_ffn_out": (C_ROOT / "prefill8_ffn_out-1.f32", "2932f1d3b23fc439ebdbee791a031a93e0681724f62935967e324b19a09196f5", 49152),
}
TWINS = {
    "ref_q": tc.TWINS["ref_q"], "ref_k": tc.TWINS["ref_k"], "ref_target": tc.TWINS["ref_target"],
    "ref_ffn_inp": REF_ROOT / "cached7p1/ffn_inp-1.full.f32le",
    "ref_moe_out": REF_ROOT / "cached7p1/ffn_moe_out-1.full.f32le",
    "ref_shared_out": REF_ROOT / "cached7p1/ffn_shexp-1.full.f32le",
    "ref_ffn_out": REF_ROOT / "cached7p1/ffn_out-1.full.f32le",
    "c_moe_out": C_ROOT / "cached7p1_ffn_moe_out-1.f32",
    "c_shared_out": C_ROOT / "cached7p1_ffn_shexp-1.f32",
    "c_ffn_out": C_ROOT / "cached7p1_ffn_out-1.f32",
}
EXPECTED_CANDIDATE_SHA = {
    ARM_NAMES[2]: PINNED["ref_ffn_out"][1],
    ARM_NAMES[3]: PINNED["c_ffn_out"][1],
    ARM_NAMES[4]: "2d0b1e880f91423cfe59444ba89972e8943a3f1d61a8f50bdb89b2fe1a57ee1c",
    ARM_NAMES[5]: "c79eb03d686e492968bec34321f918709131e8f37b4ea4b314930228b5779a7f",
}
FROZEN = {"nrmse": 0.002642086799405445, "normalized_max": 0.004687597394884091}


class RunnerError(RuntimeError):
    pass


def sources():
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "engine": ENGINE, "header": HEADER,
        "terminal_header": tc.HEADER, "terminal_runner": Path(tc.__file__).resolve(),
        "attn_norm_header": tc.an.CROSS_HEADER, "kva_header": tc.an.KVA_HEADER,
        "rms_header": tc.an.RMS_HEADER, "rung2a": tc.an.RUNG2A,
        "rung2c": tc.an.RUNG2C, "base_runner": Path(base.__file__).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise RunnerError("missing FFN-output component source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def clean(source_map):
    rel = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in source_map.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, rel)], cwd=ROOT, check=False).returncode:
        raise RunnerError("FFN-output component sources differ from HEAD")
    for path in rel:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"untracked FFN-output component source: {path}")


def frozen_evidence():
    evidence = {}
    for name, (path, digest, size) in PINNED.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise RunnerError(f"FFN-output component input mismatch: {name}")
        twin = TWINS[name]
        if not twin.is_file() or twin.stat().st_size != size or base.sha256_file(twin) != digest:
            raise RunnerError(f"FFN-output component twin mismatch: {name}")
        evidence[name] = path.resolve(strict=True)
    for path, digest, label in (
        (PREDECESSOR, PREDECESSOR_SHA, "terminal-component predecessor"),
        (PREDECESSOR_OUTPUT, PREDECESSOR_OUTPUT_SHA, "predecessor downstream output"),
    ):
        if not path.is_file() or base.sha256_file(path) != digest:
            raise RunnerError(label + " mismatch")
    evidence["prior"] = PREDECESSOR_OUTPUT.resolve(strict=True)
    return evidence


def arm_manifest(outputs):
    if not isinstance(outputs, dict) or tuple(outputs) != ARM_NAMES:
        raise RunnerError("FFN-output component arm labels/order mismatch")
    for name, item in outputs.items():
        expected = {"kind", "moe_origin", "shared_origin", "moe_control", "ffn_out", "l_out", "norm", "projection", "prefix", "downstream"}
        if set(item) != expected:
            raise RunnerError("FFN-output component arm schema mismatch")
        if (item["kind"], item["moe_origin"], item["shared_origin"], item["moe_control"]) != ARM_META[name]:
            raise RunnerError("FFN-output component arm metadata mismatch")
        for key, size in (("ffn_out", 49152), ("l_out", 49152), ("norm", 49152), ("projection", 18432), ("prefix", 16384), ("downstream", 196608)):
            if set(item[key]) != {"path", "bytes", "sha256"} or item[key]["bytes"] != size:
                raise RunnerError("FFN-output component payload schema mismatch")


def validate_report(root, model, source_map):
    report = json.loads((root / "strat01_layer1_ffn_output_component_cross_input.json").read_text(encoding="utf-8"))
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "matrices", "outputs", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-layer1-ffn-output-component-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise RunnerError("FFN-output component report schema mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise RunnerError("FFN-output component model mismatch")
    order = ("ref_q", "ref_k", "ref_ffn_inp", "ref_moe_out", "ref_shared_out", "ref_ffn_out", "c_moe_out", "c_shared_out", "c_ffn_out")
    expected_inputs = {name: {"path": str(PINNED[name][0].resolve()), "bytes": PINNED[name][2], "sha256": PINNED[name][1]} for name in order}
    if report["inputs"] != expected_inputs:
        raise RunnerError("FFN-output component inputs mismatch")
    matrices = {
        "attn_norm": {"name": "blk.2.attn_norm.weight", "type": "F32", "shape": [1536], "offset": 578811136, "file_offset": 584914048, "span": 6144},
        "projection": {"name": "blk.2.attn_kv_a_mqa.weight", "type": "Q4_K", "shape": [1536, 576]},
        "kv_norm": {"name": "blk.2.attn_kv_a_norm.weight", "type": "F32", "shape": [512]},
        "v_b": {"name": "blk.2.attn_v_b.weight", "type": "Q4_K", "shape": [512, 192, 32]},
    }
    if report["matrices"] != matrices:
        raise RunnerError("FFN-output component descriptors mismatch")
    if report["engine_source_sha256"] != source_map["engine"]["sha256"] or report["diagnostic_source_sha256"] != source_map["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise RunnerError("FFN-output component provenance mismatch")
    arm_manifest(report["outputs"])
    values = {}
    for name, item in report["outputs"].items():
        values[name] = {key: base.load_f32(base.contained(root, item[key]["path"], item[key]["bytes"], item[key]["sha256"], name + " " + key), item[key]["bytes"] // 4, name + " " + key) for key in ("ffn_out", "l_out", "norm", "projection", "prefix", "downstream")}
    return values, report


def classify(judgments):
    if not judgments[ARM_NAMES[0]]["pass"] or judgments[ARM_NAMES[1]]["pass"] or judgments[ARM_NAMES[3]]["pass"]:
        raise RunnerError("FFN-output component anchor contradiction")
    routed_fails = not judgments[ARM_NAMES[4]]["pass"]
    shared_fails = not judgments[ARM_NAMES[5]]["pass"]
    if routed_fails and shared_fails:
        return "LAYER1_FFN_OUTPUT_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
    if routed_fails:
        return "LAYER1_ROUTED_MOE_OUTPUT_RESIDUAL_SUFFICIENT"
    if shared_fails:
        return "LAYER1_SHARED_EXPERT_OUTPUT_RESIDUAL_SUFFICIENT"
    return "LAYER1_FFN_OUTPUT_COMPONENT_RESIDUALS_JOINTLY_SUFFICIENT"


def descriptive(candidate, reference, shape):
    result = base.metrics(candidate, reference)
    a, b = np.asarray(candidate).reshape(shape), np.asarray(reference).reshape(shape)
    result["per_token"] = [dict(base.metrics(a[i], b[i]), token=i) for i in range(shape[0])]
    return result


def adjudicate(values, report, evidence):
    target = base.load_f32(evidence["ref_target"], 49152, "reference target")
    prior = base.load_f32(evidence["prior"], 49152, "predecessor output")
    ref_out = base.load_f32(evidence["ref_ffn_out"], 12288, "reference FFN output")
    c_out = base.load_f32(evidence["c_ffn_out"], 12288, "C FFN output")
    if values[ARM_NAMES[0]]["ffn_out"].tobytes() != ref_out.tobytes() or values[ARM_NAMES[1]]["ffn_out"].tobytes() != c_out.tobytes():
        raise RunnerError("FFN-output component captured anchor mismatch")
    for key in ("ffn_out", "l_out", "norm", "projection", "prefix", "downstream"):
        if values[ARM_NAMES[2]][key].tobytes() != values[ARM_NAMES[0]][key].tobytes():
            raise RunnerError("FFN-output component reference replay mismatch")
        if values[ARM_NAMES[3]][key].tobytes() != values[ARM_NAMES[1]][key].tobytes():
            raise RunnerError("FFN-output component C replay mismatch")
    for name, digest in EXPECTED_CANDIDATE_SHA.items():
        observed = hashlib.sha256(values[name]["ffn_out"].astype("<f4", copy=False).tobytes()).hexdigest()
        if observed != digest:
            raise RunnerError("FFN-output component candidate mismatch: " + name)
    if values[ARM_NAMES[0]]["downstream"].tobytes() != target.tobytes() or values[ARM_NAMES[1]]["downstream"].tobytes() != prior.tobytes():
        raise RunnerError("FFN-output component downstream anchor mismatch")
    judgments = {name: base.judged(values[name]["downstream"], target) for name in ARM_NAMES[:6]}
    for metric, expected in FROZEN.items():
        if abs(judgments[ARM_NAMES[1]][metric] - expected) > 1e-12:
            raise RunnerError("FFN-output component frozen metric mismatch")
    control = base.judged(values[ARM_NAMES[6]]["downstream"], target)
    if control["pass"]:
        raise RunnerError("FFN-output component planted control failed")
    mutations = {}
    for name, (_, digest, size) in PINNED.items():
        data = bytearray(evidence[name].read_bytes()); data[len(data) // 2] ^= 1
        mutations[name] = not base.identity_matches(bytes(data), size, digest)
    if not all(mutations.values()):
        raise RunnerError("FFN-output component mutation failed")
    swapped = copy.deepcopy(report["outputs"]); items = list(swapped.items()); items[4], items[5] = items[5], items[4]
    try:
        arm_manifest(dict(items)); label_swap_rejected = False
    except RunnerError:
        label_swap_rejected = True
    if not label_swap_rejected:
        raise RunnerError("FFN-output component label swap failed")
    component_metrics = {
        "c_moe_vs_ref": descriptive(base.load_f32(evidence["c_moe_out"], 12288, "C routed output"), base.load_f32(evidence["ref_moe_out"], 12288, "reference routed output"), (8, 1536)),
        "c_shared_vs_ref": descriptive(base.load_f32(evidence["c_shared_out"], 12288, "C shared output"), base.load_f32(evidence["ref_shared_out"], 12288, "reference shared output"), (8, 1536)),
        "c_ffn_out_vs_ref": descriptive(c_out, ref_out, (8, 1536)),
    }
    return {
        "status": classify(judgments), "arms_vs_reference": judgments,
        "candidate_metrics": {name: descriptive(values[name]["ffn_out"], ref_out, (8, 1536)) for name in ARM_NAMES[:6]},
        "component_metrics": component_metrics, "control_vs_reference": control,
        "controls": {"reference_replay_all_stages_byte_exact": True, "c_replay_all_stages_byte_exact": True, "captured_anchors_byte_exact": True, "schedule_twins_byte_exact": True, "mutated_inputs_refused": mutations, "label_swap_rejected": label_swap_rejected},
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, default=DEFAULT_MODEL); parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve(); out = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    out.mkdir(parents=True); started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter()
    status = "VOID_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT"; errors = []; commands = {}; report = {}; adjudication = {"status": "NOT_RUN"}; source_map = {}; compiler = shutil.which("clang"); binary = None; invocations = 0
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None, "opened": False}
    try:
        source_map = sources()
        if not compiler:
            raise RunnerError("clang unavailable")
        commands["clang_version"] = base.run_command([compiler, "--version"], out, "clang_version", 30); base.require_ok(commands["clang_version"], "clang")
        binary = out / "engine_layer1_ffn_output_component.exe"
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], out, "compile", 600); base.require_ok(commands["compile"], "compile")
        tests = (
            ("ffn_output_component_selftest", "--strat01-layer1-ffn-output-component-cross-input-selftest"),
            ("terminal_component_selftest", "--strat01-layer1-terminal-component-cross-input-selftest"),
            ("attn_norm_selftest", "--strat01-layer2-attn-rmsnorm-cross-input-selftest"),
            ("kva_selftest", "--strat01-layer2-kv-a-projection-cross-input-selftest"),
            ("rms_selftest", "--strat01-layer2-kv-rmsnorm-cross-input-selftest"),
            ("partition_selftest", "--strat01-layer2-kv-partition-cross-input-selftest"),
            ("legacy_selftest", "--kselftest"),
        )
        for label, flag in tests:
            commands[label] = base.run_command([str(binary), flag], out, label, 300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer1_ffn_output_component_cross_input"], out, "python_tests", 300); base.require_ok(commands["python_tests"], "tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean(source_map)
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise RunnerError("artifact mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA, "opened": True})
            evidence = frozen_evidence(); root = out / "diagnostic"; root.mkdir()
            command = [str(binary), "--strat01-layer1-ffn-output-component-cross-input", str(model), "--ref-q", str(evidence["ref_q"]), "--ref-k", str(evidence["ref_k"]), "--ref-ffn-inp", str(evidence["ref_ffn_inp"]), "--ref-moe-out", str(evidence["ref_moe_out"]), "--ref-shared-out", str(evidence["ref_shared_out"]), "--ref-ffn-out", str(evidence["ref_ffn_out"]), "--c-moe-out", str(evidence["c_moe_out"]), "--c-shared-out", str(evidence["c_shared_out"]), "--c-ffn-out", str(evidence["c_ffn_out"]), "--out-dir", str(root)]
            invocations = 1; commands["diagnostic"] = base.run_command(command, out, "diagnostic", 21600); base.require_ok(commands["diagnostic"], "diagnostic")
            source_map = sources(); values, report = validate_report(root, model, source_map); adjudication = adjudicate(values, report, evidence); status = adjudication["status"]
    except (RunnerError, base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record = {
        "schema": "strat01_layer1_ffn_output_component_cross_input_v1", "status": status, "errors": errors,
        "diagnostic_invocations": invocations, "donor_graph_executions": 0, "reference_graph_executions": 0,
        "adjudication": adjudication, "c_report": report,
        "provenance": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - tick, "git_head_observed" if args.apparatus_only else "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "source_hashes": source_map, "predecessor": {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}, "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd()}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands},
        "non_claims": ["repair", "producer rerun", "uncaptured operator", "later layers", "quality/RAM/rate"],
    }
    base.write_json(out / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(out), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
