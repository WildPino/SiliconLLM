#!/usr/bin/env python3
"""Confirm the integrated STRAT-01 block-0 production path without rerunning reference."""
from __future__ import annotations

import argparse
import json
import math
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

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2b as base

HERE = Path(__file__).resolve().parent
ENGINE = base.ENGINE
RUNG2A_HEADER = base.RUNG2A_HEADER
RUNG2B_HEADER = base.RUNG2B_HEADER
KB_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_kb_q5q8_diag.h"
RMS_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_rung2b_rmsnorm_diag.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_PRODUCTION_INTEGRATION_PROTOCOL_20260922.md"
MODEL = base.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_block0_production_integration_20260922"
REFERENCE_RUN = HERE / "results/strat01_gigachat_engine_rung2b_repair2_20260921"
REFERENCE_ROOT = REFERENCE_RUN / "pinned_reference"
REFERENCE_RUN_MANIFEST_SHA = "5fa1ca6317c3afb466df937f4a9f2e622c38f7d3787d84d2874b2e8afb6ca0b7"
REFERENCE_MANIFEST_SHA = "dc856554fef8b538a9a699363f78989c9dbfc9c2e7263ad0509a683aa97254b5"
COMBINED_ADJUDICATION = HERE / "results/strat01_gigachat_engine_combined_rms_q5q8_offline_adjudication_repair1_20260922/adjudication.json"
COMBINED_ADJUDICATION_SHA = "0a0aac6a58fe4bbf37d414d80d5681afdf357c7fda846596bea10685092c1b77"
EXPECTED_HASHES = {
    "ffn_norm-0": "4b17c45fcfc6573f9a5e1461e4f2c232a9536d6c8cf689884a6c8050ee0b2632",
    "ffn_up-0": "34a98ab5d44a7c1282901db0be2e152ea1a37f0cdf88366e65acd0597dbc390f",
    "ffn_gate-0": "bb52399fa69bc0f9295c0b2b2fc9588ec7689e440b7306f6df2e43f53c3710fa",
}
PRODUCTION_CONFIG = base.EXPECTED_C_CONFIG.replace(
    "ffn=block0-rmsnorm-q4kq8k-gate-up-silu-q6kq8k-down-residual;",
    "rms_accum=double;kb=q5_0xq8_0;ffn=block0-rmsnorm-q4kq8k-gate-up-silu-q6kq8k-down-residual;",
)


class IntegrationError(RuntimeError):
    pass


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "base_runner": Path(base.__file__).resolve(),
        "engine": ENGINE, "rung2a_header": RUNG2A_HEADER, "rung2b_header": RUNG2B_HEADER,
        "kb_header": KB_HEADER, "rms_header": RMS_HEADER,
        "tests": HERE / "test_strat01_block0_production_integration.py", "protocol": PROTOCOL,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise IntegrationError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def validate_bindings() -> dict[str, Any]:
    if not COMBINED_ADJUDICATION.is_file() or base.sha256_file(COMBINED_ADJUDICATION) != COMBINED_ADJUDICATION_SHA:
        raise IntegrationError("combined adjudication binding mismatch")
    combined = read_json(COMBINED_ADJUDICATION, "combined adjudication")
    if combined.get("status") != "COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES" or combined.get("errors") != [] or combined.get("new_donor_graph_executions") != 0:
        raise IntegrationError("combined adjudication state mismatch")
    run_manifest = REFERENCE_RUN / "run_manifest.json"
    if not run_manifest.is_file() or base.sha256_file(run_manifest) != REFERENCE_RUN_MANIFEST_SHA:
        raise IntegrationError("Rung-2B source run manifest binding mismatch")
    manifest = REFERENCE_ROOT / "manifest.json"
    if not manifest.is_file() or base.sha256_file(manifest) != REFERENCE_MANIFEST_SHA:
        raise IntegrationError("Rung-2B reference manifest binding mismatch")
    return combined


def validate_intermediate_hashes(metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for arm in base.ARMS:
        by_name = {item["name"]: item for item in metadata["manifests"][arm]["tensors"]}
        observed[arm] = {name: by_name[name]["sha256"] for name in EXPECTED_HASHES}
        if observed[arm] != EXPECTED_HASHES:
            raise IntegrationError(f"production intermediate hash mismatch: {arm}")
    return observed


def source_controls() -> dict[str, bool]:
    r2a = RUNG2A_HEADER.read_text(encoding="utf-8")
    r2b = RUNG2B_HEADER.read_text(encoding="utf-8")
    kb = KB_HEADER.read_text(encoding="utf-8")
    rms = RMS_HEADER.read_text(encoding="utf-8")
    return {
        "shared_q8_type_defined_once": r2a.count("typedef struct { uint16_t d; int8_t qs[32]; } strat01_kb_q8_0_block;") == 1 and "typedef struct { uint16_t d; int8_t qs[32]; } strat01_kb_q8_0_block;" not in kb,
        "shared_q8_quantizer_defined_once": r2a.count("static void strat01_kb_quantize_q8_0(") == 1 and "static void strat01_kb_quantize_q8_0(" not in kb,
        "shared_q5q8_dot_defined_once": r2a.count("static float strat01_kb_q5q8_dot(") == 1 and "static float strat01_kb_q5q8_dot(" not in kb,
        "production_rms_sites": r2a.count("strat01_r2a_rmsnorm_pinned(") == 3 and r2b.count("strat01_r2a_rmsnorm_pinned(") == 1,
        "diagnostic_rms_delegates": "strat01_r2a_rmsnorm_pinned(x,weight,y,rows,n,eps);" in rms,
        "production_kb_uses_q8q5": "strat01_kb_quantize_q8_0(q+(size_t)tok*6144U" in r2a and "strat01_kb_q5q8_dot(raw" in r2a,
        "production_config_declares_semantics": "rms_accum=double;kb=q5_0xq8_0;" in r2a and "rms_accum=double;kb=q5_0xq8_0;" in r2b,
    }


def adjudicate(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray], controls: dict[str, bool]) -> dict[str, Any]:
    failures: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    for arm in base.ARMS:
        for name in base.SHAPES:
            limits = base.TERMINAL_LIMITS if name == "l_out-0" else base.GENERAL_LIMITS
            result = base.judged(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"], limits)
            result.update({"arm": arm, "checkpoint": name})
            checkpoints.append(result)
            if not result["pass"]:
                failures.append(f"checkpoint/{arm}/{name}")
    continuity: list[dict[str, Any]] = []
    for implementation, values in (("production", candidate), ("pinned_reference", reference)):
        for name, shape in base.SHAPES.items():
            result = base.judged(base.token7(values[f"prefill8/{name}"], shape), base.token7(values[f"cached7p1/{name}"], shape), base.CONTINUITY_LIMITS)
            result.update({"implementation": implementation, "checkpoint": name})
            continuity.append(result)
            if not result["pass"]:
                failures.append(f"continuity/{implementation}/{name}")
    prior = base.adjudicate(candidate, reference)
    negative_controls = prior["negative_controls"]
    if not prior["negative_controls_pass"]:
        failures.append("causal_negative_controls")
    if not all(controls.values()):
        failures.append("source_controls")
    status = "PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION" if not failures else "FAIL_ENGINE_BLOCK0_PRODUCTION_INTEGRATION"
    return {
        "status": status, "failures": failures, "checkpoint_results": checkpoints,
        "continuity_results": continuity, "negative_controls": negative_controls,
        "negative_controls_pass": prior["negative_controls_pass"], "source_controls": controls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    model, output = args.model.resolve(), args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_ENGINE_BLOCK0_PRODUCTION_INTEGRATION"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, dict[str, str]] = {}
    candidate_metadata: dict[str, Any] = {}
    reference_metadata: dict[str, Any] = {}
    intermediate_hashes: dict[str, dict[str, str]] = {}
    controls: dict[str, bool] = {}
    adjudication: dict[str, Any] = {"status": "NOT_RUN"}
    donor_graph_executions = 0
    compiler = shutil.which("clang")
    binary: Path | None = None
    try:
        sources = source_inventory()
        validate_bindings()
        controls = source_controls()
        if not all(controls.values()):
            raise IntegrationError("production source controls failed")
        if not compiler:
            raise IntegrationError("clang is unavailable")
        if not args.apparatus_only:
            relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
            if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode or any(subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode for path in relative):
                raise IntegrationError("production implementation/protocol differs from HEAD")
            if not model.is_file() or model.stat().st_size != base.EXPECTED_BYTES or base.sha256_file(model) != base.EXPECTED_SHA256:
                raise IntegrationError("accepted model identity mismatch")
        commands["clang_version"] = base.run_command([compiler, "--version"], output=output, label="clang_version", timeout=30)
        base.require_ok(commands["clang_version"], "clang version")
        binary = output / "engine_block0_production.exe"
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile", timeout=600)
        base.require_ok(commands["compile"], "C build")
        for label, flag in (
            ("rung2b_selftest", "--strat01-gguf-rung2b-selftest"),
            ("combined_selftest", "--strat01-combined-rms-q5q8-selftest"),
            ("kb_selftest", "--strat01-kb-q5q8-diagnostic-selftest"),
            ("rms_selftest", "--strat01-rung2b-rmsnorm-diagnostic-selftest"),
            ("legacy_selftest", "--kselftest"),
        ):
            commands[label] = base.run_command([str(binary), flag], output=output, label=label, timeout=300)
            base.require_ok(commands[label], label)
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_block0_production_integration"], output=output, label="python_tests", timeout=300)
        base.require_ok(commands["python_tests"], "production integration Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            reference, reference_metadata = base.validate_reference(REFERENCE_ROOT, model)
            candidate_root = output / "c_engine"
            candidate_root.mkdir()
            donor_graph_executions = 1
            commands["production"] = base.run_command([str(binary), "--strat01-gguf-rung2b", str(model), "--out-dir", str(candidate_root)], output=output, label="production", timeout=21600)
            base.require_ok(commands["production"], "block-0 production")
            sources = source_inventory()
            candidate, candidate_metadata = base.validate_c(candidate_root, sources, model, expected_config=PRODUCTION_CONFIG)
            intermediate_hashes = validate_intermediate_hashes(candidate_metadata)
            adjudication = adjudicate(candidate, reference, controls)
            status = adjudication["status"]
    except (IntegrationError, base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started, "git_head": base.git_value(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": base.git_value(["git", "status", "--porcelain"]),
        "source_hashes": sources,
        "artifact": {"path": str(model), "expected_bytes": base.EXPECTED_BYTES, "expected_sha256": base.EXPECTED_SHA256},
        "reference": {"path": str(REFERENCE_ROOT), "source_run_manifest_sha256": REFERENCE_RUN_MANIFEST_SHA, "manifest_sha256": REFERENCE_MANIFEST_SHA, "executions": 0},
        "combined_adjudication_sha256": COMBINED_ADJUDICATION_SHA,
        "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler},
        "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None},
        "commands": commands,
    }
    record = {
        "schema": "strat01_block0_production_integration_adjudication_v1", "status": status,
        "errors": errors, "donor_graph_executions": donor_graph_executions,
        "reference_graph_executions": 0, "intermediate_hashes": intermediate_hashes,
        "source_controls": controls, "adjudication": adjudication,
        "candidate_metadata": candidate_metadata, "reference_metadata": reference_metadata,
        "non_claims": ["Rung 2C", "later layers", "tokenizer/logits", "generation", "quality", "RAM", "rate"],
        "provenance": provenance,
    }
    manifest = {
        "schema": "strat01_block0_production_integration_run_manifest_v1", "status": status,
        "errors": errors, "donor_graph_executions": donor_graph_executions,
        "reference_graph_executions": 0, "provenance": provenance,
    }
    base.write_json(output / "adjudication.json", record)
    base.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION", "PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION", "FAIL_ENGINE_BLOCK0_PRODUCTION_INTEGRATION"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
