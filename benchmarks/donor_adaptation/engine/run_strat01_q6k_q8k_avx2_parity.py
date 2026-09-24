#!/usr/bin/env python3
"""Qualify the model-free STRAT-01 Q6_K/Q8_K AVX2 parity apparatus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_AVX2_PARITY_PROTOCOL_20260924.md"
PINNED_X86_QUANTS = Path(
    r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4\ggml\src\ggml-cpu\arch\x86\quants.c"
)
PINNED_X86_QUANTS_SHA = "99a98747c1ac84ec40e2d1a31227b947aeb0a05bf1d8e79ec630a6d783b89f86"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q6k_q8k_avx2_parity_apparatus_20260924"
TEST_MODULE = "benchmarks.donor_adaptation.engine.test_strat01_q6k_q8k_avx2_parity"

CRITICAL_PATHS = (
    ROOT / "benchmarks/phase60/strat01_q6k_q8k_avx2.h",
    HERE / "strat01_q6k_q8k_avx2_probe.c",
    HERE / "strat01_q6k_q8k_avx2_oracle.cpp",
    HERE / "build_strat01_q6k_q8k_avx2_oracle.py",
    HERE / "test_strat01_q6k_q8k_avx2_parity.py",
    PROTOCOL,
    Path(__file__).resolve(),
)


class ApparatusError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], output: Path, label: str) -> dict[str, Any]:
    started_utc = utc_now()
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False, timeout=1800,
    )
    stdout_path = output / f"{label}.stdout.log"
    stderr_path = output / f"{label}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
    return {
        "command": command,
        "returncode": completed.returncode,
        "started_utc": started_utc,
        "seconds": time.perf_counter() - started,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ApparatusError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "strat01_engine_q6k_q8k_avx2_parity_apparatus_v1",
        "status": "VOID_BLOCK0_Q6K_Q8K_AVX2_PARITY_APPARATUS",
        "started_utc": utc_now(),
        "counters": {
            "model_reads": 0,
            "scientific_payload_reads": 0,
            "scientific_q8_population_emissions": 0,
            "scientific_q6_matrix_reads": 0,
            "diagnostic_executions": 0,
            "donor_graph_executions": 0,
            "reference_graph_executions": 0,
        },
        "commands": [],
        "errors": [],
    }
    try:
        for path in CRITICAL_PATHS:
            if not path.is_file():
                raise ApparatusError(f"missing critical source: {path}")
        if sha256_file(PINNED_X86_QUANTS) != PINNED_X86_QUANTS_SHA:
            raise ApparatusError("pinned x86 quants.c identity mismatch")
        test = run_command(
            [sys.executable, "-B", "-m", "unittest", "-v", TEST_MODULE],
            output, "q6k_q8k_avx2_model_free_tests",
        )
        manifest["commands"].append(test)
        if test["returncode"] != 0:
            raise ApparatusError("model-free AVX2 parity tests failed")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        manifest.update({
            "status": "APPARATUS_READY_NO_SCIENTIFIC_EXECUTION",
            "finished_utc": utc_now(),
            "identity": {
                "git_head_observed": head,
                "pinned_x86_quants_sha256": PINNED_X86_QUANTS_SHA,
                "critical_source_hashes": {
                    str(path.relative_to(ROOT)): sha256_file(path)
                    for path in CRITICAL_PATHS
                },
            },
            "controls": {
                "active_matches_pinned_oracle_bit_exact": True,
                "generic_negative_control_fires": True,
                "one_byte_q6_mutation_fires": True,
                "short_read_rejected": True,
                "invalid_length_rejected": True,
                "pinned_x86_source_hash": True,
                "zero_scientific_and_graph_counts": True,
            },
            "non_claims": [
                "scientific Q8_K parity", "full matrix parity", "downstream parity",
                "quality", "generation", "RAM", "speed",
            ],
        })
    except Exception as error:
        manifest["errors"].append(str(error))
        manifest["finished_utc"] = utc_now()
    manifest["environment"] = {
        "platform": platform.platform(), "python": sys.version, "cwd": os.getcwd(),
    }
    adjudication = output / "adjudication.json"
    adjudication.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(manifest["status"])
    if manifest["errors"]:
        print(manifest["errors"][0], file=sys.stderr)
    return 0 if manifest["status"] == "APPARATUS_READY_NO_SCIENTIFIC_EXECUTION" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApparatusError as error:
        raise SystemExit(f"run_strat01_q6k_q8k_avx2_parity: {error}")
