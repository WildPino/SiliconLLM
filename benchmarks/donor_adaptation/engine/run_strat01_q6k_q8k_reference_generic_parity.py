#!/usr/bin/env python3
"""Qualify the model-free Q6_K/Q8_K reference-generic compile apparatus."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_q6k_q8k_avx2_parity as base

ROOT = base.ROOT
HERE = base.HERE
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260924.md"
PINNED_GENERIC_QUANTS = Path(
    r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4\ggml\src\ggml-cpu\quants.c"
)
PINNED_GENERIC_QUANTS_SHA = "459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q6k_q8k_reference_generic_parity_apparatus_repair2_20260924"
TEST_MODULE = "benchmarks.donor_adaptation.engine.test_strat01_q6k_q8k_reference_generic_parity"

CRITICAL_PATHS = (
    ROOT / "benchmarks/phase60/engine.c",
    ROOT / "benchmarks/phase60/strat01_q6k_q8k_avx2.h",
    ROOT / "benchmarks/phase60/strat01_gguf_block0_q6k_q8k_avx2_parity.h",
    HERE / "strat01_q6k_q8k_reference_generic_probe.c",
    HERE / "strat01_q6k_q8k_reference_generic_oracle.cpp",
    HERE / "build_strat01_q6k_q8k_reference_generic_oracle.py",
    HERE / "test_strat01_q6k_q8k_reference_generic_parity.py",
    PROTOCOL,
    Path(__file__).resolve(),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise base.ApparatusError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "strat01_engine_q6k_q8k_reference_generic_parity_apparatus_v1",
        "status": "VOID_BLOCK0_Q6_REFERENCE_GENERIC_APPARATUS",
        "started_utc": base.utc_now(),
        "counters": {
            "model_reads": 0, "scientific_payload_reads": 0,
            "scientific_q8_population_emissions": 0,
            "scientific_q6_matrix_reads": 0, "diagnostic_executions": 0,
            "donor_graph_executions": 0, "reference_graph_executions": 0,
        },
        "commands": [], "errors": [],
    }
    try:
        for path in CRITICAL_PATHS:
            if not path.is_file():
                raise base.ApparatusError(f"missing critical source: {path}")
        if base.sha256_file(PINNED_GENERIC_QUANTS) != PINNED_GENERIC_QUANTS_SHA:
            raise base.ApparatusError("pinned generic quants.c identity mismatch")
        clang = shutil.which("clang")
        if not clang:
            raise base.ApparatusError("clang is unavailable")
        engine = output / "engine_q6_reference_generic.exe"
        compile_engine = base.run_command(
            [clang, "-std=c11", "-O3", "-mavx2", "-mfma",
             str(ROOT / "benchmarks/phase60/engine.c"), "-o", str(engine), "-lm"],
            output, "compile_engine",
        )
        manifest["commands"].append(compile_engine)
        if compile_engine["returncode"]:
            raise base.ApparatusError("full engine diagnostic compile failed")
        selftest = base.run_command(
            [str(engine), "--strat01-block0-q6k-q8k-reference-generic-parity-selftest"],
            output, "engine_q6_reference_generic_selftest",
        )
        manifest["commands"].append(selftest)
        if selftest["returncode"]:
            raise base.ApparatusError("full engine diagnostic selftest failed")
        test = base.run_command(
            [sys.executable, "-B", "-m", "unittest", "-v", TEST_MODULE],
            output, "q6k_q8k_reference_generic_model_free_tests",
        )
        manifest["commands"].append(test)
        if test["returncode"]:
            raise base.ApparatusError("model-free reference-generic tests failed")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        source_text = (ROOT / "benchmarks/phase60/strat01_q6k_q8k_avx2.h").read_text(encoding="utf-8")
        engine_text = (ROOT / "benchmarks/phase60/engine.c").read_text(encoding="utf-8")
        if ('target("no-avx,no-avx2,no-fma")' not in source_text or
                "strat01_q6k_q8k_dot_reference_generic" not in source_text or
                "--strat01-block0-q6k-q8k-reference-generic-parity" not in engine_text):
            raise base.ApparatusError("reference-generic compile/dispatch contract missing")
        manifest.update({
            "status": "APPARATUS_READY_NO_SCIENTIFIC_EXECUTION",
            "finished_utc": base.utc_now(),
            "identity": {
                "git_head_observed": head,
                "pinned_generic_quants_sha256": PINNED_GENERIC_QUANTS_SHA,
                "critical_source_hashes": {
                    str(path.relative_to(ROOT)): base.sha256_file(path)
                    for path in CRITICAL_PATHS
                },
            },
            "controls": {
                "candidate_matches_baseline_generic_oracle_at_all_lengths": True,
                "synthetic_full_matrix_q8_and_output_exact": True,
                "candidate_avx_tu_generic_active_avx2_pairwise_distinct": True,
                "short_read_rejected": True, "invalid_mode_rejected": True,
                "pinned_generic_source_hash": True,
                "isolated_no_avx_no_fma_target_present": True,
                "full_engine_diagnostic_compiles_and_selftests": True,
                "zero_scientific_and_graph_counts": True,
            },
            "non_claims": [
                "scientific Q8_K parity", "full matrix parity", "downstream parity",
                "quality", "generation", "RAM", "speed",
            ],
        })
    except Exception as error:
        manifest["errors"].append(str(error))
        manifest["finished_utc"] = base.utc_now()
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
    except base.ApparatusError as error:
        raise SystemExit(f"run_strat01_q6k_q8k_reference_generic_parity: {error}")
