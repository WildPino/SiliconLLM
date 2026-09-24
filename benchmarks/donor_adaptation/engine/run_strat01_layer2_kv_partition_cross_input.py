#!/usr/bin/env python3
"""Execute the frozen layer-2 compact-KV partition cross-input diagnostic."""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
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

from benchmarks.donor_adaptation.engine import run_strat01_engine_layer2_depth_extension as layer2
from benchmarks.donor_adaptation.engine import run_strat01_layer2_attention_cross_input as predecessor
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE = layer2.ENGINE
RUNG2A = layer2.RUNG2A
RUNG2C = layer2.RUNG2C
INHERITED_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_rung2c_cross_input.h"
PREDECESSOR_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer2_attention_cross_input.h"
CROSS_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer2_kv_partition_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_PARTITION_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer2_kv_partition_cross_input.py"
RAW = predecessor.RAW
DEFAULT_MODEL = predecessor.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer2_kv_partition_cross_input_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer2_kv_partition_cross_input_apparatus_20260924"
PREDECESSOR_DIR = predecessor.DEFAULT_OUTPUT
PREDECESSOR_ADJUDICATION_SHA = "266eba9376ea338db89b6caf9cb86b956d88d25c48e517b9c1102874b61575c9"
PREDECESSOR_OUTPUT_SHA = "a127b01e4c8f06a7e9a37dcc4dabe307a7a8733c6a6a84c8810d81125a884a92"
FROZEN_PREDECESSOR_METRICS = {"nrmse": 0.0027922348709553107, "normalized_max": 0.004362653758957389}
ARM_NAMES = (
    "ref_latent__ref_positional", "c_latent__c_positional",
    "c_latent__ref_positional", "ref_latent__c_positional",
    "control_ref_latent_token7_negated", "control_ref_positional_rows0_7_swapped",
)
ARM_META = {
    "ref_latent__ref_positional": ("reference", "reference", False, False),
    "c_latent__c_positional": ("c", "c", False, False),
    "c_latent__ref_positional": ("c", "reference", False, False),
    "ref_latent__c_positional": ("reference", "c", False, False),
    "control_ref_latent_token7_negated": ("reference", "reference", True, False),
    "control_ref_positional_rows0_7_swapped": ("reference", "reference", False, True),
}
VALID_STATUSES = {
    "LAYER2_EXACT_REFERENCE_FAILS_KV_PARTITION_OPERATOR",
    "LAYER2_LATENT_VALUE_RESIDUAL_SUFFICIENT",
    "LAYER2_POSITIONAL_TAIL_RESIDUAL_SUFFICIENT",
    "LAYER2_LATENT_AND_POSITIONAL_RESIDUALS_INDEPENDENTLY_SUFFICIENT",
    "LAYER2_JOINT_KV_PARTITION_INTERACTION_SUFFICIENT",
}
PINNED_FILES = {name: predecessor.PINNED_FILES[name] for name in ("ref_q", "ref_k", "ref_v", "ref_target", "c_k", "c_v")}
SCHEDULE_TWINS = {name: predecessor.SCHEDULE_TWINS[name] for name in PINNED_FILES}
EVIDENCE_FILES: dict[str, tuple[Path, str]] = {
    "canonical_recovery": (predecessor.RECOVERY, predecessor.RECOVERY_SHA),
    "raw_layer2_adjudication": (RAW / "adjudication.json", predecessor.RAW_ADJUDICATION_SHA),
    "reference_prefill_manifest": predecessor.EVIDENCE_FILES["reference_prefill_manifest"],
    "c_prefill_manifest": predecessor.EVIDENCE_FILES["c_prefill_manifest"],
    "whole_kv_predecessor": (PREDECESSOR_DIR / "adjudication.json", PREDECESSOR_ADJUDICATION_SHA),
}


class RunnerError(RuntimeError):
    pass


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "engine": ENGINE, "rung2a": RUNG2A, "rung2c": RUNG2C,
        "inherited_cross_header": INHERITED_HEADER,
        "predecessor_cross_header": PREDECESSOR_HEADER,
        "kv_partition_header": CROSS_HEADER,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunnerError("missing layer-2 KV-partition source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise RunnerError("layer-2 KV-partition sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"untracked layer-2 KV-partition source: {path}")


def validate_frozen_evidence() -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, (path, digest, size) in PINNED_FILES.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise RunnerError(f"pinned layer-2 {name} identity mismatch")
        twin = SCHEDULE_TWINS[name]
        if not twin.is_file() or twin.stat().st_size != size or base.sha256_file(twin) != digest:
            raise RunnerError(f"cached schedule twin mismatch for {name}")
        resolved[name] = path.resolve(strict=True)
    for name, (path, digest) in EVIDENCE_FILES.items():
        if not path.is_file() or base.sha256_file(path) != digest:
            raise RunnerError(f"layer-2 evidence {name} identity mismatch")
        resolved[name] = path.resolve(strict=True)
    base.validate_v_k_prefix(resolved["ref_k"], resolved["ref_v"], "reference layer-2")
    base.validate_v_k_prefix(resolved["c_k"], resolved["c_v"], "C layer-2")
    prior = PREDECESSOR_DIR / "diagnostic/ref_q__c_kv.f32le"
    if not prior.is_file() or prior.stat().st_size != 196_608 or base.sha256_file(prior) != PREDECESSOR_OUTPUT_SHA:
        raise RunnerError("whole-KV predecessor output identity mismatch")
    resolved["predecessor_output"] = prior.resolve(strict=True)
    return resolved


def validate_partition(value: Any) -> None:
    if value != {"latent_value": [0, 512], "positional": [512, 576]}:
        raise RunnerError("layer-2 KV partition boundary mismatch")


def validate_arm_manifest(outputs: Any) -> None:
    if not isinstance(outputs, dict) or tuple(outputs) != ARM_NAMES:
        raise RunnerError("layer-2 KV-partition arm labels/order mismatch")
    required = {"path", "bytes", "sha256", "q_source", "latent_source", "positional_source", "latent_control", "positional_control"}
    for name, item in outputs.items():
        if not isinstance(item, dict) or set(item) != required:
            raise RunnerError(f"layer-2 KV-partition arm schema mismatch: {name}")
        latent, positional, lc, pc = ARM_META[name]
        if (item["bytes"] != 196_608 or item["q_source"] != "reference" or
                item["latent_source"] != latent or item["positional_source"] != positional or
                item["latent_control"] is not lc or item["positional_control"] is not pc):
            raise RunnerError(f"layer-2 KV-partition arm metadata mismatch: {name}")


def validate_report(root: Path, model: Path, sources: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        report = json.loads((root / "strat01_layer2_kv_partition_cross_input.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"layer-2 KV-partition report missing or malformed: {exc}") from exc
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "v_k_prefix_equal", "partition", "matrix", "outputs", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "timing_or_rate_claim"}
    if (not isinstance(report, dict) or set(report) != required or
            report["command"] != "--strat01-layer2-kv-partition-cross-input" or
            report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or
            report["self_certifies_pass"] is not False):
        raise RunnerError("layer-2 KV-partition report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise RunnerError("layer-2 KV-partition model identity mismatch")
    expected_inputs = {name: {"path": str(PINNED_FILES[name][0].resolve()), "bytes": PINNED_FILES[name][2], "sha256": PINNED_FILES[name][1]} for name in ("ref_q", "ref_k", "ref_v", "c_k", "c_v")}
    if report["inputs"] != expected_inputs or report["v_k_prefix_equal"] != {"reference": True, "c": True}:
        raise RunnerError("layer-2 KV-partition input/V-K manifest mismatch")
    validate_partition(report["partition"])
    matrix = report["matrix"]
    if (not isinstance(matrix, dict) or set(matrix) != {"name", "type", "shape", "offset", "file_offset", "span"} or
            matrix != {"name": "blk.2.attn_v_b.weight", "type": "Q4_K", "shape": [512, 192, 32],
                       "offset": 589_434_112, "file_offset": 595_537_024, "span": 1_769_472}):
        raise RunnerError("layer-2 KV-partition matrix descriptor mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["kv_partition_header"]["sha256"]:
        raise RunnerError("layer-2 KV-partition source identity mismatch")
    if report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise RunnerError("layer-2 KV-partition execution/non-rate contract mismatch")
    validate_arm_manifest(report["outputs"])
    values: dict[str, np.ndarray] = {}
    for name, item in report["outputs"].items():
        path = base.contained(root, item["path"], 196_608, item["sha256"], name)
        values[name] = base.load_f32(path, 8 * 6144, name)
    if report["outputs"]["ref_latent__ref_positional"]["sha256"] != PINNED_FILES["ref_target"][1]:
        raise RunnerError("exact-reference partition replay is not byte-exact")
    if report["outputs"]["c_latent__c_positional"]["sha256"] != PREDECESSOR_OUTPUT_SHA:
        raise RunnerError("complete-C-K predecessor replay is not byte-exact")
    return values, report


def classify(judgments: dict[str, dict[str, Any]]) -> str:
    if not judgments["ref_latent__ref_positional"]["pass"]:
        return "LAYER2_EXACT_REFERENCE_FAILS_KV_PARTITION_OPERATOR"
    if judgments["c_latent__c_positional"]["pass"]:
        raise RunnerError("complete C KV unexpectedly passes frozen target")
    latent_fails = not judgments["c_latent__ref_positional"]["pass"]
    positional_fails = not judgments["ref_latent__c_positional"]["pass"]
    if latent_fails and positional_fails:
        return "LAYER2_LATENT_AND_POSITIONAL_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
    if latent_fails:
        return "LAYER2_LATENT_VALUE_RESIDUAL_SUFFICIENT"
    if positional_fails:
        return "LAYER2_POSITIONAL_TAIL_RESIDUAL_SUFFICIENT"
    return "LAYER2_JOINT_KV_PARTITION_INTERACTION_SUFFICIENT"


def adjudicate(values: dict[str, np.ndarray], report: dict[str, Any], frozen: dict[str, Path]) -> dict[str, Any]:
    reference = base.load_f32(frozen["ref_target"], 8 * 6144, "layer-2 reference target")
    predecessor_output = base.load_f32(frozen["predecessor_output"], 8 * 6144, "whole-KV predecessor output")
    if values["c_latent__c_positional"].tobytes() != predecessor_output.tobytes():
        raise RunnerError("complete-C-K replay differs from predecessor bytes")
    scientific = ARM_NAMES[:4]
    judgments = {name: base.judged(values[name], reference) for name in scientific}
    for metric, expected in FROZEN_PREDECESSOR_METRICS.items():
        if abs(judgments["c_latent__c_positional"][metric] - expected) > 1e-12:
            raise RunnerError(f"frozen whole-KV predecessor {metric} mismatch")
    planted = {name: base.judged(values[name], reference) for name in ARM_NAMES[4:]}
    if any(item["pass"] for item in planted.values()):
        raise RunnerError("layer-2 KV-partition planted control did not reject")
    mutation_refusals: dict[str, bool] = {}
    for name, (_, digest, size) in PINNED_FILES.items():
        mutated = bytearray(frozen[name].read_bytes()); mutated[len(mutated) // 2] ^= 1
        mutation_refusals[name] = not base.identity_matches(bytes(mutated), size, digest)
    if not all(mutation_refusals.values()):
        raise RunnerError("layer-2 KV-partition mutated-input control did not reject")
    swapped = copy.deepcopy(report["outputs"])
    items = list(swapped.items()); items[2], items[3] = items[3], items[2]
    label_swap_rejected = False
    try:
        validate_arm_manifest(dict(items))
    except RunnerError:
        label_swap_rejected = True
    if not label_swap_rejected:
        raise RunnerError("layer-2 KV-partition mixed-label swap did not reject")
    boundary_rejected = False
    try:
        validate_partition({"latent_value": [0, 511], "positional": [511, 576]})
    except RunnerError:
        boundary_rejected = True
    if not boundary_rejected:
        raise RunnerError("layer-2 KV partition-boundary mutation did not reject")
    return {
        "status": classify(judgments), "arms_vs_reference": judgments, "controls_vs_reference": planted,
        "controls": {"exact_reference_byte_exact": True, "whole_kv_predecessor_byte_exact": True,
                     "frozen_metrics_reproduced_within_1e-12": True, "schedule_twins_byte_exact": True,
                     "v_k_prefix_byte_exact": True, "mutated_inputs_refused": mutation_refusals,
                     "mixed_label_swap_rejected": label_swap_rejected, "partition_boundary_mutation_rejected": boundary_rejected},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_LAYER2_KV_PARTITION_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}
    report: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}; sources: dict[str, Any] = {}
    compiler = shutil.which("clang"); binary: Path | None = None; diagnostic_invocations = 0
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None, "opened": False}
    try:
        sources = source_inventory()
        if not compiler:
            raise RunnerError("clang unavailable")
        commands["clang_version"] = base.run_command([compiler, "--version"], output, "clang_version", 30); base.require_ok(commands["clang_version"], "clang version")
        binary = output / "engine_layer2_kv_partition_cross_input.exe"
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output, "compile", 600); base.require_ok(commands["compile"], "layer-2 KV-partition build")
        for label, flag in (("partition_selftest", "--strat01-layer2-kv-partition-cross-input-selftest"), ("whole_kv_selftest", "--strat01-layer2-attention-cross-input-selftest"), ("inherited_selftest", "--strat01-rung2c-cross-input-selftest"), ("legacy_selftest", "--kselftest")):
            commands[label] = base.run_command([str(binary), flag], output, label, 300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer2_kv_partition_cross_input"], output, "python_tests", 300); base.require_ok(commands["python_tests"], "layer-2 KV-partition Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise RunnerError("accepted artifact identity mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA, "opened": True})
            frozen = validate_frozen_evidence(); diagnostic_root = output / "diagnostic"; diagnostic_root.mkdir()
            command = [str(binary), "--strat01-layer2-kv-partition-cross-input", str(model), "--ref-q", str(frozen["ref_q"]), "--ref-k", str(frozen["ref_k"]), "--ref-v", str(frozen["ref_v"]), "--c-k", str(frozen["c_k"]), "--c-v", str(frozen["c_v"]), "--out-dir", str(diagnostic_root)]
            diagnostic_invocations = 1
            commands["boundary_diagnostic"] = base.run_command(command, output, "boundary_diagnostic", 21600); base.require_ok(commands["boundary_diagnostic"], "layer-2 KV-partition diagnostic")
            sources = source_inventory(); values, report = validate_report(diagnostic_root, model, sources)
            adjudication = adjudicate(values, report, frozen); status = adjudication["status"]
    except (RunnerError, base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started,
        "git_head_observed" if args.apparatus_only else "git_head": base.git_value(["git", "rev-parse", "HEAD"]),
        "source_hashes": sources, "evidence_hashes": {name: {"path": str(path), "sha256": digest} for name, (path, digest) in EVIDENCE_FILES.items()},
        "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler},
        "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands,
    }
    record = {
        "schema": "strat01_layer2_kv_partition_cross_input_adjudication_v1", "status": status, "errors": errors,
        "diagnostic_invocations": diagnostic_invocations, "donor_graph_executions": 0, "reference_graph_executions": 0,
        "adjudication": adjudication, "c_report": report, "provenance": provenance,
        "non_claims": ["repaired layer 2", "Q/KV projection or output projection/FFN/later layers", "tokenizer/logits/generation", "quality/RAM/rate"],
    }
    base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
