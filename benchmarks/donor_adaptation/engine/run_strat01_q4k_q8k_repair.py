#!/usr/bin/env python3
"""Build and adjudicate the frozen STRAT-01 Q4_K/Q8_K engine repair cell."""
from __future__ import annotations

import argparse
import hashlib
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
HERE = Path(__file__).resolve().parent
ENGINE = ROOT / "benchmarks/phase60/engine.c"
OPERATOR = ROOT / "benchmarks/phase60/strat01_q4k_q8k.h"
RUNG1 = ROOT / "benchmarks/phase60/strat01_gguf_rung1.h"
RUNG2A = ROOT / "benchmarks/phase60/strat01_gguf_rung2a.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REPAIR_PROTOCOL_20260921.md"
MODEL = ROOT / "benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
SOURCE_RUN = HERE / "results/strat01_gigachat_engine_rung2a_repair2_20260921"
SOURCE_MANIFEST = SOURCE_RUN / "run_manifest.json"
SOURCE_MANIFEST_SHA = "0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f"
INPUT = SOURCE_RUN / "pinned_reference/prefill8/attn_norm-0.full.f32le"
Q_REFERENCE = SOURCE_RUN / "pinned_reference/prefill8/q-0.full.f32le"
KV_REFERENCE = SOURCE_RUN / "pinned_reference/prefill8/kv_cmpr_pe-0.full.f32le"
MODEL_BYTES = 6_474_702_976
MODEL_SHA = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
INPUT_SHA = "c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd"
Q_REFERENCE_SHA = "4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b"
KV_REFERENCE_SHA = "6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c"
TIGHT_NRMSE = 2e-6
TIGHT_MAX = 1e-5
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q4k_q8k_repair1_20260921"

CRITICAL_PATHS = (
    ENGINE, OPERATOR, RUNG2A,
    HERE / "strat01_q4k_q8k_project_probe.c",
    HERE / "strat01_q4k_q8k_oracle.cpp",
    HERE / "build_strat01_q4k_q8k_oracle.py",
    HERE / "test_strat01_q4k_q8k_operator.py",
    Path(__file__).resolve(), PROTOCOL,
)

TEST_MODULES = (
    "benchmarks.donor_adaptation.engine.test_strat01_engine_rung0",
    "benchmarks.donor_adaptation.engine.test_strat01_engine_rung1",
    "benchmarks.donor_adaptation.engine.test_strat01_engine_rung2a",
    "benchmarks.donor_adaptation.engine.test_strat01_q4k_q8k_operator",
    "benchmarks.donor_adaptation.engine.test_strat01_q4k_q8k_repair_runner",
)


class RepairError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], output: Path, label: str, timeout: int = 1800) -> dict[str, Any]:
    started_utc = utc_now()
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False, timeout=timeout,
        )
        record: dict[str, Any] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        record = {
            "returncode": None,
            "stdout": str(getattr(error, "stdout", "") or ""),
            "stderr": str(getattr(error, "stderr", "") or ""),
            "spawn_or_timeout_error": str(error),
        }
    stdout_path, stderr_path = output / f"{label}.stdout.log", output / f"{label}.stderr.log"
    stdout_path.write_text(record["stdout"], encoding="utf-8", newline="\n")
    stderr_path.write_text(record["stderr"], encoding="utf-8", newline="\n")
    record.update({
        "command": command,
        "started_utc": started_utc,
        "seconds": time.perf_counter() - started,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    })
    record.pop("stdout", None)
    record.pop("stderr", None)
    return record


def require_ok(record: dict[str, Any], label: str) -> None:
    if record.get("returncode") != 0:
        raise RepairError(f"{label} failed with return code {record.get('returncode')}")


def load_payload(path: Path, count: int, expected_sha: str | None = None) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * 4:
        raise RepairError(f"payload size mismatch: {path}")
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RepairError(f"payload identity mismatch: {path}")
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != count or not bool(np.isfinite(values).all()):
        raise RepairError(f"payload count/finiteness mismatch: {path}")
    return values


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    if candidate.shape != reference.shape or candidate.size == 0:
        raise RepairError("metric inputs differ in shape or are empty")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    rms_delta = math.sqrt(float(np.mean(delta * delta)))
    rms_reference = math.sqrt(float(np.mean(reference.astype(np.float64) ** 2)))
    nrmse = rms_delta / max(rms_reference, 1e-12)
    normalized_max = float(np.max(np.abs(delta))) / max(float(np.max(np.abs(reference))), 1e-6)
    return {
        "nrmse": nrmse,
        "normalized_max": normalized_max,
        "pass": nrmse <= TIGHT_NRMSE and normalized_max <= TIGHT_MAX,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RepairError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "strat01_engine_q4k_q8k_repair_v1",
        "status": "VOID_ENGINE_Q4K_Q8K_REPAIR",
        "started_utc": utc_now(),
        "donor_executions": 0,
        "commands": [],
        "errors": [],
    }
    try:
        clean = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)],
            cwd=ROOT, check=False,
        )
        if clean.returncode:
            raise RepairError("repair implementation or frozen protocol differs from HEAD")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA:
            raise RepairError("source Rung-2A manifest identity mismatch")
        if not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES:
            raise RepairError("accepted GGUF is absent or has wrong size")
        load_payload(INPUT, 8 * 1536, INPUT_SHA)
        q_reference = load_payload(Q_REFERENCE, 8 * 6144, Q_REFERENCE_SHA)
        kv_reference = load_payload(KV_REFERENCE, 8 * 576, KV_REFERENCE_SHA)

        clang = shutil.which("clang")
        if not clang:
            raise RepairError("clang is unavailable")
        binary = output / "engine_q4k_q8k.exe"
        compile_record = run_command(
            [clang, "-std=c11", "-O3", "-mavx2", "-mfma", str(ENGINE), "-o", str(binary), "-lm"],
            output, "compile",
        )
        manifest["commands"].append(compile_record)
        require_ok(compile_record, "engine build")

        test_record = run_command(
            [sys.executable, "-m", "unittest", "-v", *TEST_MODULES],
            output, "model_free_oracle_and_legacy_tests",
        )
        manifest["commands"].append(test_record)
        require_ok(test_record, "model-free oracle and legacy tests")

        selftests = (
            ("q4k_q8k_selftest", [str(binary), "--strat01-q4k-q8k-selftest"]),
            ("rung1_selftest", [str(binary), "--strat01-gguf-rung1-selftest"]),
            ("rung2a_selftest", [str(binary), "--strat01-gguf-rung2a-selftest"]),
            ("kernel_selftest_73024", [str(binary), "--kselftest"]),
        )
        for label, command in selftests:
            record = run_command(command, output, label)
            manifest["commands"].append(record)
            require_ok(record, label)

        project_output = output / "project_output"
        project_output.mkdir()
        projection_record = run_command(
            [str(binary), "--strat01-q4k-q8k-projection", str(MODEL),
             "--input", str(INPUT), "--out-dir", str(project_output)],
            output, "accepted_projection",
        )
        manifest["commands"].append(projection_record)
        require_ok(projection_record, "accepted projection")
        report = json.loads((project_output / "projection.json").read_text(encoding="utf-8"))
        if report.get("state") != "ENGINE_Q4K_Q8K_OUTPUT_READY" or report.get("donor_executions") != 0:
            raise RepairError("project projection report contract mismatch")
        if report.get("model_sha256") != MODEL_SHA or report.get("input_sha256") != INPUT_SHA:
            raise RepairError("project projection report identity mismatch")
        q_candidate = load_payload(project_output / "q.f32le", 8 * 6144)
        kv_candidate = load_payload(project_output / "kv.f32le", 8 * 576)
        results = {"q": metrics(q_candidate, q_reference), "kv": metrics(kv_candidate, kv_reference)}
        controls = {
            "critical_sources_committed": True,
            "source_manifest_identity": True,
            "input_and_reference_identity": True,
            "model_free_oracle_population": True,
            "negative_controls": True,
            "legacy_rung0_rung1_rung2a_tests": True,
            "kernel_selftest_73024": True,
            "project_report_identity": True,
        }
        manifest.update({
            "status": "PASS_ENGINE_Q4K_Q8K_REPAIR" if all(bool(value["pass"]) for value in results.values()) else "FAIL_ENGINE_Q4K_Q8K_REPAIR",
            "finished_utc": utc_now(),
            "identity": {
                "git_head": head,
                "model": {"path": str(MODEL), "bytes": MODEL_BYTES, "sha256": MODEL_SHA},
                "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": SOURCE_MANIFEST_SHA},
                "input": {"path": str(INPUT), "sha256": INPUT_SHA},
                "binary": {"path": str(binary), "sha256": sha256_file(binary)},
                "critical_source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS},
            },
            "controls": controls,
            "results": results,
            "outputs": {
                "q": {"path": str(project_output / "q.f32le"), "sha256": sha256_file(project_output / "q.f32le")},
                "kv": {"path": str(project_output / "kv.f32le"), "sha256": sha256_file(project_output / "kv.f32le")},
                "project_report": {"path": str(project_output / "projection.json"), "sha256": sha256_file(project_output / "projection.json")},
            },
            "non_claims": ["downstream attention parity", "Rung-2A acceptance", "quality", "RAM", "speed"],
            "environment": {"platform": platform.platform(), "python": sys.version, "cwd": os.getcwd()},
        })
    except Exception as error:
        manifest["errors"].append(str(error))
        manifest["finished_utc"] = utc_now()
    adjudication = output / "adjudication.json"
    adjudication.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(manifest["status"])
    if manifest["errors"]:
        print(manifest["errors"][0], file=sys.stderr)
    return 0 if manifest["status"] in {"PASS_ENGINE_Q4K_Q8K_REPAIR", "FAIL_ENGINE_Q4K_Q8K_REPAIR"} else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairError as error:
        raise SystemExit(f"run_strat01_q4k_q8k_repair: {error}")
