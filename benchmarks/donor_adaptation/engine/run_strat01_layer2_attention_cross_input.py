#!/usr/bin/env python3
"""Execute the frozen layer-2 Q/KV cross-input boundary diagnostic."""
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
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE = layer2.ENGINE
RUNG2A = layer2.RUNG2A
RUNG2C = layer2.RUNG2C
INHERITED_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_rung2c_cross_input.h"
CROSS_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer2_attention_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTENTION_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer2_attention_cross_input.py"
RAW = HERE / "results/strat01_gigachat_engine_layer2_depth_extension_20260924"
RECOVERY = HERE / "results/strat01_gigachat_engine_layer2_depth_extension_offline_recovery1_20260924/adjudication.json"
DEFAULT_MODEL = layer2.r2c.MODEL
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer2_attention_cross_input_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer2_attention_cross_input_apparatus_20260924"
RECOVERY_SHA = "0b5d34ea652676c8ec04b8a87f89ff1e08757c370ebcfdab939871ca6558b46e"
RAW_ADJUDICATION_SHA = "65cf9f30bdf4c2028f5c13f91cf4dfc44d28b3c04d07bc254c395e0c74e0693c"
FROZEN_C_METRICS = {"nrmse": 0.003014631769821054, "normalized_max": 0.004362653758957389}
VALID_STATUSES = {
    "LAYER2_EXACT_REFERENCE_FAILS_ATTENTION_VB_OPERATOR",
    "LAYER2_QUERY_RESIDUAL_SUFFICIENT",
    "LAYER2_KV_RESIDUAL_SUFFICIENT",
    "LAYER2_QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT",
    "LAYER2_JOINT_QUERY_KV_INTERACTION_SUFFICIENT",
}

PINNED_FILES: dict[str, tuple[Path, str, int]] = {
    "ref_q": (RAW / "pinned_reference/prefill8/Qcur-2.full.f32le", "46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea", 589_824),
    "ref_k": (RAW / "pinned_reference/prefill8/Kcur-2.full.f32le", "b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b", 18_432),
    "ref_v": (RAW / "pinned_reference/prefill8/Vcur-2.full.f32le", "0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91", 16_384),
    "ref_target": (RAW / "pinned_reference/prefill8/kqv_out-2.full.f32le", "d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e", 196_608),
    "c_q": (RAW / "c_engine/prefill8_Qcur-2.f32", "a1bcd7529bae496e48db28964123f8f28bfc26fd2559c54ada51fb9ae49464d5", 589_824),
    "c_k": (RAW / "c_engine/prefill8_Kcur-2.f32", "163c0eb7f03892c0ff86cfd1ad455382aa9f59a245c2f849918a92d5537a41f9", 18_432),
    "c_v": (RAW / "c_engine/prefill8_Vcur-2.f32", "e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9", 16_384),
    "c_target": (RAW / "c_engine/prefill8_kqv_out-2.f32", "47c8b7db2e7a7a01886033870c0806987f804a0ac5ec01e5e8b1db8a285a2ae2", 196_608),
}
SCHEDULE_TWINS: dict[str, Path] = {
    "ref_q": RAW / "pinned_reference/cached7p1/Qcur-2.full.f32le",
    "ref_k": RAW / "pinned_reference/cached7p1/Kcur-2.full.f32le",
    "ref_v": RAW / "pinned_reference/cached7p1/Vcur-2.full.f32le",
    "ref_target": RAW / "pinned_reference/cached7p1/kqv_out-2.full.f32le",
    "c_q": RAW / "c_engine/cached7p1_Qcur-2.f32",
    "c_k": RAW / "c_engine/cached7p1_Kcur-2.f32",
    "c_v": RAW / "c_engine/cached7p1_Vcur-2.f32",
    "c_target": RAW / "c_engine/cached7p1_kqv_out-2.f32",
}
EVIDENCE_FILES: dict[str, tuple[Path, str]] = {
    "canonical_recovery": (RECOVERY, RECOVERY_SHA),
    "raw_adjudication": (RAW / "adjudication.json", RAW_ADJUDICATION_SHA),
    "reference_prefill_manifest": (RAW / "pinned_reference/prefill8/manifest.json", "088c75d7be93865abfcea1f22d34080477c360bfd2cfd06738cf86b02cc6b035"),
    "c_prefill_manifest": (RAW / "c_engine/prefill8_manifest.json", "b0f7bddb442ba89da0edb129ff20387b3c4016a661a3e7858ebe458c09558989"),
}


class RunnerError(RuntimeError):
    pass


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "engine": ENGINE, "rung2a": RUNG2A, "rung2c": RUNG2C,
        "inherited_cross_header": INHERITED_HEADER, "layer2_cross_header": CROSS_HEADER,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunnerError("missing layer-2 cross-input source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise RunnerError("layer-2 cross-input sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RunnerError(f"untracked layer-2 cross-input source: {path}")


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
    return resolved


def validate_report(root: Path, model: Path, sources: dict[str, Any], frozen: dict[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        report = json.loads((root / "strat01_layer2_attention_cross_input.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"layer-2 diagnostic report missing or malformed: {exc}") from exc
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "v_k_prefix_equal", "matrix", "outputs", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "timing_or_rate_claim"}
    if (not isinstance(report, dict) or set(report) != required or
            report["command"] != "--strat01-layer2-attention-cross-input" or
            report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or
            report["self_certifies_pass"] is not False):
        raise RunnerError("layer-2 diagnostic report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise RunnerError("layer-2 diagnostic model identity mismatch")
    expected_inputs = {
        name: {"path": str(PINNED_FILES[name][0].resolve()), "bytes": PINNED_FILES[name][2], "sha256": PINNED_FILES[name][1]}
        for name in ("ref_q", "ref_k", "ref_v", "c_q", "c_k", "c_v")
    }
    if report["inputs"] != expected_inputs or report["v_k_prefix_equal"] != {"reference": True, "c": True}:
        raise RunnerError("layer-2 diagnostic input/V-K manifest mismatch")
    matrix = report["matrix"]
    if (not isinstance(matrix, dict) or set(matrix) != {"name", "type", "shape", "offset", "file_offset", "span"} or
            matrix["name"] != "blk.2.attn_v_b.weight" or matrix["type"] != "Q4_K" or
            matrix["shape"] != [512, 192, 32] or matrix["span"] != 1_769_472 or
            not all(isinstance(matrix[key], int) and matrix[key] >= 0 for key in ("offset", "file_offset"))):
        raise RunnerError("layer-2 diagnostic matrix descriptor mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["layer2_cross_header"]["sha256"]:
        raise RunnerError("layer-2 diagnostic source identity mismatch")
    if report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise RunnerError("layer-2 diagnostic execution/non-rate contract mismatch")
    base.validate_arm_manifest(report["outputs"])
    values: dict[str, np.ndarray] = {}
    for name, item in report["outputs"].items():
        path = base.contained(root, item["path"], 196_608, item["sha256"], name)
        values[name] = base.load_f32(path, 8 * 6144, name)
    if report["outputs"]["c_q__c_kv"]["sha256"] != PINNED_FILES["c_target"][1]:
        raise RunnerError("layer-2 native C/C byte replay mismatch")
    return values, report


def classify(judgments: dict[str, dict[str, Any]]) -> str:
    if judgments["c_q__c_kv"]["pass"]:
        raise RunnerError("native layer-2 C/C unexpectedly passes frozen target")
    if not judgments["ref_q__ref_kv"]["pass"]:
        return "LAYER2_EXACT_REFERENCE_FAILS_ATTENTION_VB_OPERATOR"
    query_fails = not judgments["c_q__ref_kv"]["pass"]
    kv_fails = not judgments["ref_q__c_kv"]["pass"]
    if query_fails and kv_fails:
        return "LAYER2_QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
    if query_fails:
        return "LAYER2_QUERY_RESIDUAL_SUFFICIENT"
    if kv_fails:
        return "LAYER2_KV_RESIDUAL_SUFFICIENT"
    return "LAYER2_JOINT_QUERY_KV_INTERACTION_SUFFICIENT"


def adjudicate(values: dict[str, np.ndarray], report: dict[str, Any], frozen: dict[str, Path]) -> dict[str, Any]:
    reference = base.load_f32(frozen["ref_target"], 8 * 6144, "layer-2 reference target")
    c_target = base.load_f32(frozen["c_target"], 8 * 6144, "layer-2 C target")
    if values["c_q__c_kv"].tobytes() != c_target.tobytes():
        raise RunnerError("layer-2 native replay differs from captured C target")
    judgments = {name: base.judged(values[name], reference) for name in ("c_q__c_kv", "ref_q__ref_kv", "c_q__ref_kv", "ref_q__c_kv")}
    for metric, expected in FROZEN_C_METRICS.items():
        if abs(judgments["c_q__c_kv"][metric] - expected) > 1e-12:
            raise RunnerError(f"frozen layer-2 C/reference {metric} replay mismatch")
    planted = {name: base.judged(values[name], reference) for name in ("control_ref_q_token7_negated", "control_ref_kv_rows0_7_swapped")}
    if any(item["pass"] for item in planted.values()):
        raise RunnerError("layer-2 planted query/KV control did not reject")
    mutation_refusals: dict[str, bool] = {}
    for name in ("ref_q", "ref_k", "ref_v", "c_q", "c_k", "c_v"):
        _, digest, size = PINNED_FILES[name]
        mutated = bytearray(frozen[name].read_bytes()); mutated[len(mutated) // 2] ^= 1
        mutation_refusals[name] = not base.identity_matches(bytes(mutated), size, digest)
    if not all(mutation_refusals.values()):
        raise RunnerError("layer-2 mutated-input identity control did not reject")
    swapped = copy.deepcopy(report["outputs"])
    swapped["c_q__ref_kv"], swapped["ref_q__c_kv"] = swapped["ref_q__c_kv"], swapped["c_q__ref_kv"]
    label_swap_rejected = False
    try:
        base.validate_arm_manifest(swapped)
    except base.RunnerError:
        label_swap_rejected = True
    if not label_swap_rejected:
        raise RunnerError("layer-2 mixed-arm label-swap control did not reject")
    status = classify(judgments)
    return {
        "status": status, "arms_vs_reference": judgments, "controls_vs_reference": planted,
        "controls": {"native_replay_byte_exact": True, "frozen_metrics_reproduced_within_1e-12": True,
                     "schedule_twins_byte_exact": True, "v_k_prefix_byte_exact": True,
                     "mutated_inputs_refused": mutation_refusals, "mixed_label_swap_rejected": label_swap_rejected},
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
    status = "VOID_LAYER2_ATTENTION_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}
    report: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}; sources: dict[str, Any] = {}
    compiler = shutil.which("clang"); binary: Path | None = None; diagnostic_invocations = 0
    artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None, "opened": False}
    try:
        sources = source_inventory()
        if not compiler:
            raise RunnerError("clang unavailable")
        commands["clang_version"] = base.run_command([compiler, "--version"], output, "clang_version", 30); base.require_ok(commands["clang_version"], "clang version")
        binary = output / "engine_layer2_attention_cross_input.exe"
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output, "compile", 600); base.require_ok(commands["compile"], "layer-2 cross-input build")
        for label, flag in (("layer2_selftest", "--strat01-layer2-attention-cross-input-selftest"), ("inherited_selftest", "--strat01-rung2c-cross-input-selftest"), ("legacy_selftest", "--kselftest")):
            commands[label] = base.run_command([str(binary), flag], output, label, 300); base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer2_attention_cross_input"], output, "python_tests", 300); base.require_ok(commands["python_tests"], "layer-2 cross-input Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise RunnerError("accepted artifact identity mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA, "opened": True})
            frozen = validate_frozen_evidence(); diagnostic_root = output / "diagnostic"; diagnostic_root.mkdir()
            command = [str(binary), "--strat01-layer2-attention-cross-input", str(model), "--ref-q", str(frozen["ref_q"]), "--ref-k", str(frozen["ref_k"]), "--ref-v", str(frozen["ref_v"]), "--c-q", str(frozen["c_q"]), "--c-k", str(frozen["c_k"]), "--c-v", str(frozen["c_v"]), "--out-dir", str(diagnostic_root)]
            diagnostic_invocations = 1
            commands["boundary_diagnostic"] = base.run_command(command, output, "boundary_diagnostic", 21600); base.require_ok(commands["boundary_diagnostic"], "layer-2 boundary diagnostic")
            sources = source_inventory(); values, report = validate_report(diagnostic_root, model, sources, frozen)
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
        "schema": "strat01_layer2_attention_cross_input_adjudication_v1", "status": status, "errors": errors,
        "diagnostic_invocations": diagnostic_invocations, "donor_graph_executions": 0, "reference_graph_executions": 0,
        "adjudication": adjudication, "c_report": report, "provenance": provenance,
        "non_claims": ["repaired layer 2", "output projection/FFN/later layers", "tokenizer/logits/generation", "quality/RAM/rate"],
    }
    base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
