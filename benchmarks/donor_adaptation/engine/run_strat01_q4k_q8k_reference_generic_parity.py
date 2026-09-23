#!/usr/bin/env python3
"""Adjudicate frozen Q4_K/Q8_K reference-generic compile parity."""
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

from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as base

ROOT = base.ROOT
HERE = base.HERE
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260923.md"
REFERENCE_BUILD = HERE / "results/strat01_gigachat_engine_rung2a_repair2_20260921/reference_build/cmake-build"
REFERENCE_COMPILE_COMMANDS = REFERENCE_BUILD / "compile_commands.json"
REFERENCE_BINARY = REFERENCE_BUILD / "strat01_engine_rung2a_reference.exe"
REFERENCE_COMPILE_COMMANDS_SHA = "de27906dc35179b5efd47ffab98adb33476425ef215f9b0513ae9229454ecb80"
REFERENCE_BINARY_SHA = "abdadf3fdac1bed75e24b1353d1fbff85159c1f6e4c3e8d81e9efe5e1e246627"
Q_ACTIVE_AVX2_SHA = "b1d355be461cf30f42cc0acc88c4cf08963678a252be74761031a41e44978cd3"
KV_ACTIVE_AVX2_SHA = "a2c177598c5ea5e7a9b5cb5707a6a738b711edd548ce6d9430eb3f66be2222f2"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q4k_q8k_reference_generic_parity_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_q4k_q8k_reference_generic_parity_apparatus_20260923"

CRITICAL_PATHS = (
    base.ENGINE, base.OPERATOR, base.RUNG2A, PROTOCOL,
    HERE / "strat01_q4k_q8k_reference_generic_probe.c",
    HERE / "strat01_q4k_q8k_reference_generic_oracle.cpp",
    HERE / "build_strat01_q4k_q8k_reference_generic_oracle.py",
    HERE / "test_strat01_q4k_q8k_reference_generic_parity.py",
    HERE / "test_strat01_q4k_q8k_reference_generic_parity_runner.py",
    HERE / "build_strat01_rung2a_projection_diagnostic.py",
    Path(__file__).resolve(),
)
TEST_MODULES = tuple(
    "benchmarks.donor_adaptation.engine." + path.stem
    for path in sorted(HERE.glob("test_strat01_*.py"))
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists():
        raise base.ParityError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "strat01_engine_q4k_q8k_reference_generic_compile_parity_v1",
        "status": "VOID_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY",
        "started_utc": base.utc_now(),
        "donor_graph_executions": 0,
        "reference_graph_executions": 0,
        "model_reads": 0,
        "commands": [],
        "errors": [],
    }
    try:
        base.require_file(base.PINNED_X86_QUANTS, sha=base.PINNED_X86_QUANTS_SHA)
        base.require_file(base.SOURCE_MANIFEST, sha=base.SOURCE_MANIFEST_SHA)
        base.require_file(base.INPUT, 8 * 1536 * 4, base.INPUT_SHA)
        base.require_file(base.Q_REFERENCE, 8 * 6144 * 4, base.Q_REFERENCE_SHA)
        base.require_file(base.KV_REFERENCE, 8 * 576 * 4, base.KV_REFERENCE_SHA)
        base.require_file(REFERENCE_COMPILE_COMMANDS, sha=REFERENCE_COMPILE_COMMANDS_SHA)
        base.require_file(REFERENCE_BINARY, sha=REFERENCE_BINARY_SHA)
        compile_commands = json.loads(REFERENCE_COMPILE_COMMANDS.read_text(encoding="utf-8"))
        generic_entries = [
            entry for entry in compile_commands
            if entry["file"].replace("\\", "/").endswith("/ggml-cpu/quants.c")
        ]
        if len(generic_entries) != 1:
            raise base.ParityError("preserved reference compile inventory lacks one generic quants.c")
        reference_command = generic_entries[0]["command"]
        if "-DGGML_CPU_GENERIC" not in reference_command or "-mavx" in reference_command or "-mfma" in reference_command:
            raise base.ParityError("preserved reference compile mode is not baseline GGML_CPU_GENERIC")
        if not args.apparatus_only:
            base.require_file(base.MODEL, base.MODEL_BYTES, base.MODEL_SHA)
            clean = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)],
                cwd=ROOT, check=False,
            )
            if clean.returncode:
                raise base.ParityError("implementation or frozen protocol differs from HEAD")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        clang = shutil.which("clang")
        if not clang:
            raise base.ParityError("clang is unavailable")
        binaries = {
            "candidate": output / "engine_reference_generic.exe",
            "avx_translation_generic": output / "engine_avx_translation_generic.exe",
            "active_avx2": output / "engine_active_avx2.exe",
        }
        definitions = {
            "candidate": [],
            "avx_translation_generic": ["-DSTRAT01_Q4K_Q8K_DIAGNOSTIC_GENERIC_REDUCTION=1"],
            "active_avx2": ["-DSTRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2=1"],
        }
        for label, binary in binaries.items():
            record = base.run_command(
                [clang, "-std=c11", "-O3", "-mavx2", "-mfma", *definitions[label],
                 str(base.ENGINE), "-o", str(binary), "-lm"],
                output, f"compile_{label}",
            )
            manifest["commands"].append(record)
            base.require_ok(record, f"compile {label}")

        tests = base.run_command(
            [sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES],
            output, "all_strat01_unittests",
        )
        manifest["commands"].append(tests)
        base.require_ok(tests, "all STRAT-01 unit tests")
        for index, option in enumerate(base.SELFTESTS):
            record = base.run_command([str(binaries["candidate"]), option], output, f"selftest_{index:02d}")
            manifest["commands"].append(record)
            base.require_ok(record, option)

        controls: dict[str, bool] = {
            "preserved_reference_compile_commands_hash": True,
            "preserved_reference_binary_hash": True,
            "reference_backend_is_baseline_generic": True,
            "source_manifest_identity": True,
            "input_and_reference_identity": True,
            "candidate_oracle_one_and_multiblock_exact": True,
            "avx_translation_generic_control_fires": True,
            "active_avx2_control_fires": True,
            "all_strat01_unit_tests": True,
            "all_registered_c_selftests": True,
            "kernel_selftest_73024": True,
            "zero_graph_executions": True,
        }
        identity = {
            "git_head_observed" if args.apparatus_only else "git_head": head,
            "reference_compile_commands": {"path": str(REFERENCE_COMPILE_COMMANDS), "sha256": REFERENCE_COMPILE_COMMANDS_SHA},
            "reference_binary": {"path": str(REFERENCE_BINARY), "sha256": REFERENCE_BINARY_SHA},
            "critical_source_hashes": {str(path.relative_to(ROOT)): base.sha256_file(path) for path in CRITICAL_PATHS},
            "binary_hashes": {label: base.sha256_file(path) for label, path in binaries.items()},
        }
        if args.apparatus_only:
            manifest.update({
                "status": "APPARATUS_READY_NO_DONOR_EXECUTION",
                "finished_utc": base.utc_now(),
                "identity": identity,
                "controls": controls,
                "non_claims": ["full projection parity", "downstream parity", "quality", "RAM", "speed"],
            })
        else:
            outputs: dict[str, dict[str, str]] = {}
            for label, binary in binaries.items():
                destination = output / f"{label}_projection"
                destination.mkdir()
                record = base.run_command(
                    [str(binary), "--strat01-q4k-q8k-projection", str(base.MODEL),
                     "--input", str(base.INPUT), "--out-dir", str(destination)],
                    output, f"projection_{label}",
                )
                manifest["commands"].append(record)
                base.require_ok(record, f"projection {label}")
                manifest["model_reads"] += 1
                outputs[label] = base.exact_projection_hashes(destination)
            exact = {
                "candidate_q_reference_exact": outputs["candidate"]["q"] == base.Q_REFERENCE_SHA,
                "candidate_kv_reference_exact": outputs["candidate"]["kv"] == base.KV_REFERENCE_SHA,
                "avx_generic_q_history_exact": outputs["avx_translation_generic"]["q"] == base.Q_GENERIC_SHA,
                "avx_generic_kv_history_exact": outputs["avx_translation_generic"]["kv"] == base.KV_GENERIC_SHA,
                "active_avx2_q_history_exact": outputs["active_avx2"]["q"] == Q_ACTIVE_AVX2_SHA,
                "active_avx2_kv_history_exact": outputs["active_avx2"]["kv"] == KV_ACTIVE_AVX2_SHA,
            }
            controls.update(exact)
            manifest.update({
                "status": (
                    "PASS_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY"
                    if all(controls.values()) else "FAIL_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY"
                ),
                "finished_utc": base.utc_now(),
                "identity": {
                    **identity,
                    "model": {"path": str(base.MODEL), "bytes": base.MODEL_BYTES, "sha256": base.MODEL_SHA},
                    "input": {"path": str(base.INPUT), "sha256": base.INPUT_SHA},
                },
                "controls": controls,
                "outputs": outputs,
                "non_claims": ["downstream propagation parity", "quality", "generation", "RAM", "rate"],
            })
        manifest["environment"] = {"platform": platform.platform(), "python": sys.version, "cwd": os.getcwd()}
    except Exception as error:
        manifest["errors"].append(str(error))
        manifest["finished_utc"] = base.utc_now()
    adjudication = output / "adjudication.json"
    adjudication.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(manifest["status"])
    if manifest["errors"]:
        print(manifest["errors"][0], file=sys.stderr)
    return 0 if manifest["status"] in {
        "APPARATUS_READY_NO_DONOR_EXECUTION",
        "PASS_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY",
        "FAIL_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY",
    } else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.ParityError as error:
        raise SystemExit(f"run_strat01_q4k_q8k_reference_generic_parity: {error}")
