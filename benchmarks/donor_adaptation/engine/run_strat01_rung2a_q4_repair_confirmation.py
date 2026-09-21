#!/usr/bin/env python3
"""Confirm the integrated Q4_K/Q8_K repair and localize the next Rung-2A mismatch."""
from __future__ import annotations

import argparse
import json
import math
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

from benchmarks.donor_adaptation.engine.run_strat01_engine_rung2a import (
    ARMS,
    GENERAL_LIMITS,
    SHAPES,
    TERMINAL_LIMITS,
    adjudicate,
    judged_metrics,
    read_json,
    token7,
    validate_c_outputs,
    validate_reference_outputs,
)
from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_repair import (
    ENGINE,
    MODEL,
    MODEL_BYTES,
    MODEL_SHA,
    OPERATOR,
    RUNG2A,
    TEST_MODULES,
    require_ok,
    run_command,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION_PROTOCOL_20260921.md"
SOURCE_RUN = HERE / "results/strat01_gigachat_engine_rung2a_repair2_20260921"
SOURCE_MANIFEST_SHA = "0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f"
REPAIR_ADJUDICATION = HERE / "results/strat01_gigachat_engine_q4k_q8k_repair1_20260921/adjudication.json"
REPAIR_ADJUDICATION_SHA = "53931d8e141c9e6faff2fce04d1665ff71fd8976e6a3c763721ddcd5435fdc39"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_rung2a_q4_repair_confirmation_20260921"
TIGHT_LIMITS = (2e-6, 1e-5)
REQUIRED_TENSORS = ("attn_norm-0", "q-0", "kv_cmpr_pe-0")

CRITICAL_PATHS = (
    Path(__file__).resolve(),
    ENGINE,
    OPERATOR,
    RUNG2A,
    HERE / "test_strat01_q4k_q8k_operator.py",
    HERE / "test_strat01_q4k_q8k_repair_runner.py",
    HERE / "test_strat01_rung2a_q4_repair_confirmation.py",
    PROTOCOL,
)


class ConfirmationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_source_hashes() -> dict[str, dict[str, str]]:
    return {
        "engine": {"path": str(ENGINE), "sha256": sha256_file(ENGINE)},
        "rung2a_header": {"path": str(RUNG2A), "sha256": sha256_file(RUNG2A)},
    }


def old_source_hashes() -> dict[str, dict[str, str]]:
    report = read_json(SOURCE_RUN / "c_engine/strat01_rung2a.json", "old C report")
    return {
        "engine": {"path": "frozen old source", "sha256": report["engine_source_sha256"]},
        "rung2a_header": {"path": "frozen old source", "sha256": report["rung2a_source_sha256"]},
    }


def primary_results(
    c_tensors: dict[str, np.ndarray], reference_tensors: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for arm in ARMS:
        for name in REQUIRED_TENSORS:
            result = judged_metrics(c_tensors[f"{arm}/{name}"], reference_tensors[f"{arm}/{name}"], TIGHT_LIMITS)
            result.update({"arm": arm, "tensor": name, "logical_shape": SHAPES[name]})
            results.append(result)
    return results


def continuity_result(c_tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    shape = SHAPES["ffn_inp-0"]
    prefill = token7(c_tensors["prefill8/ffn_inp-0"], shape)
    cached = token7(c_tensors["cached7p1/ffn_inp-0"], shape)
    result = judged_metrics(cached, prefill, TIGHT_LIMITS)
    result["implementation"] = "repaired_c_engine"
    return result


def old_negative_controls(
    old_c: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for arm in ARMS:
        for name in ("q-0", "kv_cmpr_pe-0"):
            result = judged_metrics(old_c[f"{arm}/{name}"], reference[f"{arm}/{name}"], GENERAL_LIMITS)
            result.update({"arm": arm, "tensor": name, "must_fail_old_gate": not result["pass"]})
            controls.append(result)
    return controls


def first_downstream_failures(full: dict[str, Any]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for arm in ARMS:
        first = next(
            (entry["tensor"] for entry in full["tensor_results"] if entry["arm"] == arm and not entry["pass"]),
            None,
        )
        if first is None:
            first = next(
                ("cache:" + entry["checkpoint"] for entry in full["cache_results"] if entry["checkpoint"].startswith(arm) and not entry["pass"]),
                None,
            )
        result[arm] = first
    return result


def classify(primary: list[dict[str, Any]], continuity: dict[str, Any], negatives: list[dict[str, Any]]) -> str:
    if not all(bool(item["must_fail_old_gate"]) for item in negatives):
        return "VOID_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION"
    if all(bool(item["pass"]) for item in primary) and bool(continuity["pass"]):
        return "PASS_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION"
    return "FAIL_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ConfirmationError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": "strat01_rung2a_q4_repair_confirmation_v1",
        "status": "VOID_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION",
        "started_utc": utc_now(),
        "c_donor_executions": 0,
        "reference_donor_executions": 0,
        "commands": [],
        "errors": [],
    }
    try:
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)],
            cwd=ROOT, check=False,
        )
        if dirty.returncode:
            raise ConfirmationError("confirmation implementation or protocol differs from HEAD")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        if sha256_file(SOURCE_RUN / "run_manifest.json") != SOURCE_MANIFEST_SHA:
            raise ConfirmationError("frozen source-run manifest identity mismatch")
        if sha256_file(REPAIR_ADJUDICATION) != REPAIR_ADJUDICATION_SHA:
            raise ConfirmationError("accepted repair adjudication identity mismatch")
        repair = read_json(REPAIR_ADJUDICATION, "accepted repair adjudication")
        if repair.get("status") != "PASS_ENGINE_Q4K_Q8K_REPAIR":
            raise ConfirmationError("accepted repair did not pass")
        if not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES:
            raise ConfirmationError("accepted GGUF is absent or has wrong size")

        reference_tensors, reference_metadata, reference_cache = validate_reference_outputs(
            SOURCE_RUN / "pinned_reference", MODEL
        )
        old_c_tensors, _, _ = validate_c_outputs(SOURCE_RUN / "c_engine", old_source_hashes(), MODEL)
        negatives = old_negative_controls(old_c_tensors, reference_tensors)
        if not all(bool(item["must_fail_old_gate"]) for item in negatives):
            raise ConfirmationError("old captured Q/KV negative control did not reproduce")

        clang = shutil.which("clang")
        if not clang:
            raise ConfirmationError("clang is unavailable")
        binary = output / "engine_rung2a_q4_repair.exe"
        compile_record = run_command(
            [clang, "-std=c11", "-O3", "-mavx2", "-mfma", str(ENGINE), "-o", str(binary), "-lm"],
            output, "compile",
        )
        record["commands"].append(compile_record)
        require_ok(compile_record, "engine build")
        tests = run_command(
            [sys.executable, "-m", "unittest", "-v", *TEST_MODULES,
             "benchmarks.donor_adaptation.engine.test_strat01_rung2a_q4_repair_confirmation"],
            output, "model_free_oracle_and_legacy_tests",
        )
        record["commands"].append(tests)
        require_ok(tests, "model-free oracle and legacy tests")
        for label, command in (
            ("rung2a_selftest", [str(binary), "--strat01-gguf-rung2a-selftest"]),
            ("kernel_selftest_73024", [str(binary), "--kselftest"]),
        ):
            command_record = run_command(command, output, label)
            record["commands"].append(command_record)
            require_ok(command_record, label)

        c_root = output / "c_engine"
        c_root.mkdir()
        c_run = run_command(
            [str(binary), "--strat01-gguf-rung2a", str(MODEL), "--out-dir", str(c_root)],
            output, "accepted_artifact_repaired_c_engine", timeout=21_600,
        )
        record["commands"].append(c_run)
        record["c_donor_executions"] = 1
        require_ok(c_run, "accepted-artifact repaired C execution")
        c_tensors, c_metadata, c_cache = validate_c_outputs(c_root, current_source_hashes(), MODEL)
        primary = primary_results(c_tensors, reference_tensors)
        continuity = continuity_result(c_tensors)
        full = adjudicate(c_tensors, reference_tensors, c_cache, reference_cache)
        first_failures = first_downstream_failures(full)
        status = classify(primary, continuity, negatives)
        record.update({
            "status": status,
            "finished_utc": utc_now(),
            "seconds": time.perf_counter() - started,
            "identity": {
                "git_head": head,
                "model": {"path": str(MODEL), "bytes": MODEL_BYTES, "sha256": MODEL_SHA},
                "source_run_manifest_sha256": SOURCE_MANIFEST_SHA,
                "repair_adjudication_sha256": REPAIR_ADJUDICATION_SHA,
                "binary": {"path": str(binary), "sha256": sha256_file(binary)},
                "critical_source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS},
            },
            "controls": {
                "critical_sources_committed": True,
                "immutable_reference_validated": True,
                "old_c_q_kv_negative_controls_fire": True,
                "model_free_oracle_and_legacy_tests": True,
                "kernel_selftest_73024": True,
                "reference_process_not_rerun": True,
            },
            "primary_results": primary,
            "c_continuity": continuity,
            "old_negative_controls": negatives,
            "full_old_gate_adjudication": full,
            "first_remaining_failure": first_failures,
            "c_metadata": c_metadata,
            "reference_metadata": reference_metadata,
            "non_claims": ["historical Rung-2A rewrite", "Rung 2B", "quality", "RAM", "speed"],
        })
    except Exception as error:
        record["errors"].append(str(error))
        record["finished_utc"] = utc_now()
        record["seconds"] = time.perf_counter() - started
    (output / "adjudication.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(record["status"])
    if record["errors"]:
        print(record["errors"][0], file=sys.stderr)
    return 0 if record["status"] in {
        "PASS_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION",
        "FAIL_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION",
    } else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfirmationError as error:
        raise SystemExit(f"run_strat01_rung2a_q4_repair_confirmation: {error}")
