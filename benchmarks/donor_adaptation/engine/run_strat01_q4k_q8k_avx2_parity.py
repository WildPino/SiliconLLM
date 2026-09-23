#!/usr/bin/env python3
"""Build and adjudicate the frozen STRAT-01 Q4_K/Q8_K AVX2 parity cell."""
from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ENGINE = ROOT / "benchmarks/phase60/engine.c"
OPERATOR = ROOT / "benchmarks/phase60/strat01_q4k_q8k.h"
RUNG2A = ROOT / "benchmarks/phase60/strat01_gguf_rung2a.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_PROTOCOL_20260923.md"
MODEL = ROOT / "benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
SOURCE_RUN = HERE / "results/strat01_gigachat_engine_rung2a_repair2_20260921"
SOURCE_MANIFEST = SOURCE_RUN / "run_manifest.json"
INPUT = SOURCE_RUN / "pinned_reference/prefill8/attn_norm-0.full.f32le"
Q_REFERENCE = SOURCE_RUN / "pinned_reference/prefill8/q-0.full.f32le"
KV_REFERENCE = SOURCE_RUN / "pinned_reference/prefill8/kv_cmpr_pe-0.full.f32le"
PINNED_X86_QUANTS = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4\ggml\src\ggml-cpu\arch\x86\quants.c")

MODEL_BYTES = 6_474_702_976
MODEL_SHA = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
SOURCE_MANIFEST_SHA = "0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f"
INPUT_SHA = "c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd"
Q_REFERENCE_SHA = "4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b"
KV_REFERENCE_SHA = "6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c"
Q_GENERIC_SHA = "6255f40d5a717af2a742c484b0ab75ff549b37acea70f73556698c794c8b3366"
KV_GENERIC_SHA = "5c3fdab029c1660bae7c4f4d5d256991def1ccce39468ccdb5d9b1346d5be9ce"
PINNED_X86_QUANTS_SHA = "99a98747c1ac84ec40e2d1a31227b947aeb0a05bf1d8e79ec630a6d783b89f86"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q4k_q8k_avx2_parity_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_q4k_q8k_avx2_parity_apparatus_20260923"

CRITICAL_PATHS = (
    ENGINE, OPERATOR, RUNG2A, PROTOCOL,
    HERE / "strat01_q4k_q8k_avx2_probe.c",
    HERE / "strat01_q4k_q8k_avx2_oracle.cpp",
    HERE / "build_strat01_q4k_q8k_avx2_oracle.py",
    HERE / "build_strat01_rung2a_projection_diagnostic.py",
    HERE / "test_strat01_q4k_q8k_avx2_parity.py",
    HERE / "test_strat01_q4k_q8k_avx2_parity_runner.py",
    Path(__file__).resolve(),
)

TEST_MODULES = tuple(
    "benchmarks.donor_adaptation.engine." + path.stem
    for path in sorted(HERE.glob("test_strat01_*.py"))
)

SELFTESTS = (
    "--strat01-q4k-q8k-selftest",
    "--strat01-gguf-rung1-selftest",
    "--strat01-gguf-rung2a-selftest",
    "--strat01-gguf-rung2b-selftest",
    "--strat01-gguf-rung2c-selftest",
    "--strat01-rung2b-cross-input-selftest",
    "--strat01-rung2c-cross-input-selftest",
    "--strat01-rung2c-layer1-start-cross-input-selftest",
    "--strat01-block0-terminal-component-cross-input-selftest",
    "--strat01-ffn-down-cross-input-selftest",
    "--strat01-ffn-swiglu-cross-input-selftest",
    "--strat01-rung2b-rmsnorm-diagnostic-selftest",
    "--strat01-upstream-rmsnorm-diagnostic-selftest",
    "--strat01-kb-q5q8-diagnostic-selftest",
    "--strat01-combined-rms-q5q8-selftest",
    "--kselftest",
)


class ParityError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, size: int | None = None, sha: str | None = None) -> None:
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        raise ParityError(f"file identity/size failure: {path}")
    if sha is not None and sha256_file(path) != sha:
        raise ParityError(f"file hash failure: {path}")


def run_command(command: list[str], output: Path, label: str, timeout: int = 3600) -> dict[str, Any]:
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
        raise ParityError(f"{label} failed with return code {record.get('returncode')}")


def exact_projection_hashes(directory: Path) -> dict[str, str]:
    report = json.loads((directory / "projection.json").read_text(encoding="utf-8"))
    if report.get("state") != "ENGINE_Q4K_Q8K_OUTPUT_READY" or report.get("donor_executions") != 0:
        raise ParityError("projection report contract mismatch")
    if report.get("model_sha256") != MODEL_SHA or report.get("input_sha256") != INPUT_SHA:
        raise ParityError("projection report identity mismatch")
    return {
        "q": sha256_file(directory / "q.f32le"),
        "kv": sha256_file(directory / "kv.f32le"),
        "report": sha256_file(directory / "projection.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists():
        raise ParityError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "strat01_engine_q4k_q8k_avx2_reduction_parity_v1",
        "status": "VOID_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY",
        "started_utc": utc_now(),
        "donor_graph_executions": 0,
        "reference_graph_executions": 0,
        "model_reads": 0,
        "commands": [],
        "errors": [],
    }
    try:
        require_file(PINNED_X86_QUANTS, sha=PINNED_X86_QUANTS_SHA)
        require_file(SOURCE_MANIFEST, sha=SOURCE_MANIFEST_SHA)
        require_file(INPUT, 8 * 1536 * 4, INPUT_SHA)
        require_file(Q_REFERENCE, 8 * 6144 * 4, Q_REFERENCE_SHA)
        require_file(KV_REFERENCE, 8 * 576 * 4, KV_REFERENCE_SHA)
        if not args.apparatus_only:
            require_file(MODEL, MODEL_BYTES, MODEL_SHA)
            clean = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)],
                cwd=ROOT, check=False,
            )
            if clean.returncode:
                raise ParityError("implementation or frozen protocol differs from HEAD")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        clang = shutil.which("clang")
        if not clang:
            raise ParityError("clang is unavailable")
        active_binary = output / "engine_active_avx2.exe"
        generic_binary = output / "engine_generic_control.exe"
        for label, binary, extra in (
            ("compile_active_avx2", active_binary, []),
            ("compile_generic_control", generic_binary, ["-DSTRAT01_Q4K_Q8K_DIAGNOSTIC_GENERIC_REDUCTION=1"]),
        ):
            record = run_command(
                [clang, "-std=c11", "-O3", "-mavx2", "-mfma", *extra,
                 str(ENGINE), "-o", str(binary), "-lm"],
                output, label,
            )
            manifest["commands"].append(record)
            require_ok(record, label)

        tests = run_command(
            [sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES],
            output, "all_strat01_unittests",
        )
        manifest["commands"].append(tests)
        require_ok(tests, "all STRAT-01 unit tests")
        for index, option in enumerate(SELFTESTS):
            record = run_command([str(active_binary), option], output, f"selftest_{index:02d}")
            manifest["commands"].append(record)
            require_ok(record, option)

        controls: dict[str, bool] = {
            "pinned_x86_source_hash": True,
            "source_manifest_identity": True,
            "input_and_reference_identity": True,
            "active_oracle_one_and_multiblock_exact": True,
            "generic_multiblock_negative_control": True,
            "all_strat01_unit_tests": True,
            "all_registered_c_selftests": True,
            "kernel_selftest_73024": True,
            "zero_graph_executions": True,
        }
        if args.apparatus_only:
            manifest.update({
                "status": "APPARATUS_READY_NO_DONOR_EXECUTION",
                "finished_utc": utc_now(),
                "identity": {
                    "git_head_observed": head,
                    "critical_source_hashes": {
                        str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS
                    },
                    "active_binary_sha256": sha256_file(active_binary),
                    "generic_control_binary_sha256": sha256_file(generic_binary),
                },
                "controls": controls,
                "non_claims": ["full projection parity", "downstream parity", "quality", "RAM", "speed"],
            })
        else:
            active_output, generic_output = output / "active_projection", output / "generic_projection"
            active_output.mkdir()
            generic_output.mkdir()
            for label, binary, destination in (
                ("active_projection", active_binary, active_output),
                ("generic_projection", generic_binary, generic_output),
            ):
                record = run_command(
                    [str(binary), "--strat01-q4k-q8k-projection", str(MODEL),
                     "--input", str(INPUT), "--out-dir", str(destination)],
                    output, label,
                )
                manifest["commands"].append(record)
                require_ok(record, label)
                manifest["model_reads"] += 1
            active_hashes = exact_projection_hashes(active_output)
            generic_hashes = exact_projection_hashes(generic_output)
            exact = {
                "active_q_reference_exact": active_hashes["q"] == Q_REFERENCE_SHA,
                "active_kv_reference_exact": active_hashes["kv"] == KV_REFERENCE_SHA,
                "generic_q_history_exact": generic_hashes["q"] == Q_GENERIC_SHA,
                "generic_kv_history_exact": generic_hashes["kv"] == KV_GENERIC_SHA,
            }
            controls.update(exact)
            status = (
                "PASS_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY"
                if all(controls.values()) else "FAIL_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY"
            )
            manifest.update({
                "status": status,
                "finished_utc": utc_now(),
                "identity": {
                    "git_head": head,
                    "model": {"path": str(MODEL), "bytes": MODEL_BYTES, "sha256": MODEL_SHA},
                    "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": SOURCE_MANIFEST_SHA},
                    "input": {"path": str(INPUT), "sha256": INPUT_SHA},
                    "pinned_x86_quants": {"path": str(PINNED_X86_QUANTS), "sha256": PINNED_X86_QUANTS_SHA},
                    "critical_source_hashes": {
                        str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS
                    },
                    "active_binary_sha256": sha256_file(active_binary),
                    "generic_control_binary_sha256": sha256_file(generic_binary),
                },
                "controls": controls,
                "outputs": {"active": active_hashes, "generic_control": generic_hashes},
                "non_claims": ["downstream propagation parity", "Rung 2C", "quality", "generation", "RAM", "rate"],
            })
        manifest["environment"] = {
            "platform": platform.platform(), "python": sys.version, "cwd": os.getcwd()
        }
    except Exception as error:
        manifest["errors"].append(str(error))
        manifest["finished_utc"] = utc_now()
    adjudication = output / "adjudication.json"
    adjudication.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(manifest["status"])
    if manifest["errors"]:
        print(manifest["errors"][0], file=sys.stderr)
    return 0 if manifest["status"] in {
        "APPARATUS_READY_NO_DONOR_EXECUTION",
        "PASS_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY",
        "FAIL_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY",
    } else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParityError as error:
        raise SystemExit(f"run_strat01_q4k_q8k_avx2_parity: {error}")
