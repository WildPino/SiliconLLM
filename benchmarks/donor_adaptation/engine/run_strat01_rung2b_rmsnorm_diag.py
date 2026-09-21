#!/usr/bin/env python3
"""Execute the frozen Rung-2B float-vs-double RMSNorm diagnostic."""
from __future__ import annotations

import argparse
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

MODULE_ROOT = Path(__file__).resolve().parents[3]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_rung2b_cross_input as base


ROOT = base.ROOT
HERE = base.HERE
ENGINE = base.ENGINE
RMS_HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2b_rmsnorm_diag.h"
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RMSNORM_ACCUMULATOR_DIAGNOSTIC_PROTOCOL_20260921.md"
REF_INPUT = base.ACCEPTED_RUN / "pinned_reference" / "prefill8" / "ffn_inp-0.full.f32le"
C_INPUT = HERE / "results" / "strat01_gigachat_engine_attention_vb_repair_20260921" / "c_engine" / "prefill8_ffn_inp-0.f32"
REF_NORM = base.REF_INPUT
C_NORM = base.C_INPUT
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_rung2b_rmsnorm_diag_20260921"
PINNED = {
    "reference_input": (REF_INPUT, "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1", base.INPUT_BYTES),
    "c_input": (C_INPUT, "bd00c9c5b01726980ab0865e27860f73b7c4186940362919d16dee7e75ee8d82", base.INPUT_BYTES),
    "reference_norm": (REF_NORM, base.PINNED["reference_input"][1], base.INPUT_BYTES),
    "c_norm": (C_NORM, base.PINNED["c_input"][1], base.INPUT_BYTES),
    "reference_up": base.PINNED["reference_up"], "reference_gate": base.PINNED["reference_gate"],
    "c_up": base.PINNED["c_up"], "c_gate": base.PINNED["c_gate"],
}
OUTPUT_SIZES = {
    "reference_float_norm.f32le": base.INPUT_BYTES, "reference_double_norm.f32le": base.INPUT_BYTES,
    "c_float_norm.f32le": base.INPUT_BYTES, "c_double_norm.f32le": base.INPUT_BYTES,
    "c_float_up.f32le": base.OUTPUT_BYTES, "c_float_gate.f32le": base.OUTPUT_BYTES,
    "c_double_up.f32le": base.OUTPUT_BYTES, "c_double_gate.f32le": base.OUTPUT_BYTES,
}
FROZEN_NORM_METRIC = {"nrmse": 0.0007914143404266998, "normalized_max": 0.0006736163184616577}


def source_inventory() -> dict[str, Any]:
    paths = {"runner": Path(__file__).resolve(), "engine": ENGINE, "rung2a": base.RUNG2A, "rung2b": base.RUNG2B, "cross_header": base.CROSS_HEADER, "rms_header": RMS_HEADER, "protocol": PROTOCOL}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise base.RunnerError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def validate_pinned() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, (path, digest, size) in PINNED.items():
        if not path.is_file() or path.stat().st_size != size or base.sha256_file(path) != digest:
            raise base.RunnerError(f"pinned {name} identity mismatch")
        result[name] = path.resolve(strict=True)
    return result


def read_report(root: Path, model: Path, sources: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        report = json.loads((root / "strat01_rung2b_rmsnorm_diag.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise base.RunnerError(f"RMSNorm report missing or malformed: {exc}") from exc
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "outputs", "q8", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "timing_or_rate_claim"}
    if not isinstance(report, dict) or set(report) != required or report["command"] != "--strat01-rung2b-rmsnorm-diagnostic" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise base.RunnerError("RMSNorm report schema/state mismatch")
    model_record = report["model"]
    if Path(model_record.get("path", "")).resolve(strict=True) != model.resolve(strict=True) or model_record.get("bytes") != base.EXPECTED_MODEL_BYTES or model_record.get("sha256") != base.EXPECTED_MODEL_SHA:
        raise base.RunnerError("RMSNorm model identity mismatch")
    expected_inputs = {"reference": {"path": str(REF_INPUT.resolve()), "bytes": base.INPUT_BYTES, "sha256": PINNED["reference_input"][1]}, "c_engine": {"path": str(C_INPUT.resolve()), "bytes": base.INPUT_BYTES, "sha256": PINNED["c_input"][1]}}
    if report["inputs"] != expected_inputs or report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["rms_header"]["sha256"]:
        raise base.RunnerError("RMSNorm input/source identity mismatch")
    if report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise base.RunnerError("RMSNorm execution/non-rate contract mismatch")
    if set(report["outputs"]) != set(OUTPUT_SIZES):
        raise base.RunnerError("RMSNorm output set mismatch")
    values: dict[str, np.ndarray] = {}
    for key, size in OUTPUT_SIZES.items():
        item = report["outputs"][key]
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != size:
            raise base.RunnerError(f"RMSNorm {key} schema mismatch")
        path = base.contained(root, item["path"], size, item["sha256"], key)
        values[key] = base.load_f32(path, item["sha256"], size // 4, key)
    q8 = report["q8"]
    if set(q8) != {"float", "double", "changed_blocks", "total_blocks", "changed_bytes", "total_bytes"} or q8["total_blocks"] != 48 or q8["total_bytes"] != base.Q8_BYTES:
        raise base.RunnerError("RMSNorm Q8 schema mismatch")
    paths = []
    for key in ("float", "double"):
        item = q8[key]
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != base.Q8_BYTES:
            raise base.RunnerError("RMSNorm Q8 payload schema mismatch")
        paths.append(base.contained(root, item["path"], base.Q8_BYTES, item["sha256"], f"q8/{key}"))
    a, b = paths[0].read_bytes(), paths[1].read_bytes()
    if q8["changed_bytes"] != sum(x != y for x, y in zip(a, b)) or q8["changed_blocks"] != sum(a[i:i+292] != b[i:i+292] for i in range(0, base.Q8_BYTES, 292)):
        raise base.RunnerError("RMSNorm Q8 census mismatch")
    return values, report


def adjudicate(values: dict[str, np.ndarray], pinned: dict[str, Path], report: dict[str, Any]) -> dict[str, Any]:
    targets = {name: base.load_f32(path, PINNED[name][1], PINNED[name][2] // 4, name) for name, path in pinned.items() if name in {"reference_norm", "c_norm", "reference_up", "reference_gate", "c_up", "c_gate"}}
    output_meta = report["outputs"]
    exact = {
        "c_float_norm": output_meta["c_float_norm.f32le"]["sha256"] == PINNED["c_norm"][1],
        "c_float_up": output_meta["c_float_up.f32le"]["sha256"] == PINNED["c_up"][1],
        "c_float_gate": output_meta["c_float_gate.f32le"]["sha256"] == PINNED["c_gate"][1],
    }
    if not all(exact.values()):
        raise base.RunnerError("float-semantics byte replay mismatch")
    baseline = {
        "norm": base.metrics(values["c_float_norm.f32le"], targets["reference_norm"]),
        "up": base.judged(values["c_float_up.f32le"], targets["reference_up"]),
        "gate": base.judged(values["c_float_gate.f32le"], targets["reference_gate"]),
    }
    for metric in ("nrmse", "normalized_max"):
        if abs(baseline["norm"][metric] - FROZEN_NORM_METRIC[metric]) > 1e-12 or abs(baseline["up"][metric] - base.FROZEN_C_METRICS["up"][metric]) > 1e-12 or abs(baseline["gate"][metric] - base.FROZEN_C_METRICS["gate"][metric]) > 1e-12:
            raise base.RunnerError("frozen baseline metric replay mismatch")
    primary = {"up": base.judged(values["c_double_up.f32le"], targets["reference_up"]), "gate": base.judged(values["c_double_gate.f32le"], targets["reference_gate"])}
    norms = {
        "reference_float_vs_reference": base.metrics(values["reference_float_norm.f32le"], targets["reference_norm"]),
        "reference_double_vs_reference": base.metrics(values["reference_double_norm.f32le"], targets["reference_norm"]),
        "c_float_vs_reference": baseline["norm"],
        "c_double_vs_reference": base.metrics(values["c_double_norm.f32le"], targets["reference_norm"]),
        "c_double_vs_c_float": base.metrics(values["c_double_norm.f32le"], values["c_float_norm.f32le"]),
    }
    swapped = {"up_as_gate": base.judged(values["c_double_up.f32le"], targets["reference_gate"]), "gate_as_up": base.judged(values["c_double_gate.f32le"], targets["reference_up"])}
    if swapped["up_as_gate"]["pass"] or swapped["gate_as_up"]["pass"]:
        raise base.RunnerError("double candidate swapped-target control did not reject")
    data = pinned["c_input"].read_bytes(); mutated = bytearray(data); mutated[len(mutated)//2] ^= 1
    mutation_refused = not base.identity_matches(bytes(mutated), base.INPUT_BYTES, PINNED["c_input"][1])
    if not mutation_refused:
        raise base.RunnerError("mutated RMSNorm input was not refused")
    status = "DOUBLE_RMSNORM_SUFFICIENT_FOR_PROJECTION_GATES" if all(item["pass"] for item in primary.values()) else "DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES"
    return {"status": status, "primary_double_c_projections": primary, "norm_comparisons": norms, "frozen_float_baseline": baseline, "q8_census_float_vs_double_c": {key: report["q8"][key] for key in ("changed_blocks", "total_blocks", "changed_bytes", "total_bytes")}, "controls": {"float_payloads_byte_exact": exact, "frozen_metrics_reproduced_within_1e-12": True, "mutated_input_refused": mutation_refused, "swapped_targets": swapped}}


def run_command(command: list[str], output: Path, label: str, timeout: int) -> dict[str, Any]:
    return base.run_command(command, output=output, label=label, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve(); output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter(); status = "VOID_RMSNORM_ACCUMULATOR_DIAGNOSTIC"; errors: list[str] = []; commands: dict[str, Any] = {}; report: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}; sources: dict[str, Any] = {}; binary: Path | None = None
    compiler = shutil.which("clang"); artifact = {"path": str(model), "expected_bytes": base.EXPECTED_MODEL_BYTES, "expected_sha256": base.EXPECTED_MODEL_SHA, "bytes": None, "sha256": None}
    try:
        sources = source_inventory()
        if not compiler:
            raise base.RunnerError("clang is unavailable")
        commands["clang_version"] = run_command([compiler, "--version"], output, "clang_version", 30); base.require_ok(commands["clang_version"], "clang --version")
        binary = output / "engine_rmsnorm_diag.exe"
        commands["compile"] = run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output, "compile", 600); base.require_ok(commands["compile"], "C build")
        commands["rmsnorm_selftest"] = run_command([str(binary), "--strat01-rung2b-rmsnorm-diagnostic-selftest"], output, "rmsnorm_selftest", 300); base.require_ok(commands["rmsnorm_selftest"], "RMSNorm selftest")
        commands["legacy_selftest"] = run_command([str(binary), "--kselftest"], output, "legacy_selftest", 300); base.require_ok(commands["legacy_selftest"], "legacy selftest")
        commands["python_tests"] = run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_rung2b_rmsnorm_diag"], output, "python_tests", 300); base.require_ok(commands["python_tests"], "RMSNorm Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA:
                raise base.RunnerError("accepted model identity mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": base.EXPECTED_MODEL_SHA}); pinned = validate_pinned(); raw = output / "diagnostic"; raw.mkdir()
            commands["rmsnorm_diagnostic"] = run_command([str(binary), "--strat01-rung2b-rmsnorm-diagnostic", str(model), "--reference-input", str(pinned["reference_input"]), "--c-input", str(pinned["c_input"]), "--out-dir", str(raw)], output, "rmsnorm_diagnostic", 21600); base.require_ok(commands["rmsnorm_diagnostic"], "RMSNorm diagnostic")
            sources = source_inventory(); values, report = read_report(raw, model, sources); adjudication = adjudicate(values, pinned, report); status = adjudication["status"]
    except base.RunnerError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-started, "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": base.git_value(["git", "status", "--porcelain"]), "source_hashes": sources, "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_rung2b_rmsnorm_accumulator_adjudication_v1", "status": status, "errors": errors, "scope": "captured ffn_inp float-vs-double RMSNorm and unchanged block-0 Q4_K up/gate", "donor_graph_executions": 0, "adjudication": adjudication, "c_report": report, "non_claims": ["production repair", "complete Rung 2B", "Rung 2C", "quality, generation, RAM, or rate"], "provenance": provenance}
    manifest = {"schema": "strat01_rung2b_rmsnorm_accumulator_run_manifest_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "provenance": provenance}
    base.write_json(output / "adjudication.json", record); base.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION", "DOUBLE_RMSNORM_SUFFICIENT_FOR_PROJECTION_GATES", "DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
