#!/usr/bin/env python3
"""Run the frozen block-0 Q6_K/Q8_K AVX2 full-matrix parity cell once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import (  # noqa: E402
    run_strat01_layer1_attention_output_residual_cross_input as ao,
)
from benchmarks.donor_adaptation.engine.build_strat01_q6k_q8k_avx2_oracle import (  # noqa: E402
    build as build_oracle,
)

HERE = Path(__file__).resolve().parent
ENGINE = ROOT / "benchmarks/phase60/engine.c"
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_block0_q6k_q8k_avx2_parity.h"
PRIMITIVE = ROOT / "benchmarks/phase60/strat01_q6k_q8k_avx2.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_AVX2_PARITY_PROTOCOL_20260924.md"
MODEL = ao.DEFAULT_MODEL
BLOCK0_REFERENCE = HERE / "results/strat01_gigachat_engine_rung2b_repair1_20260921/pinned_reference"
SWIGLU = BLOCK0_REFERENCE / "prefill8/ffn_swiglu-0.full.f32le"
SWIGLU_TWIN = BLOCK0_REFERENCE / "cached7p1/ffn_swiglu-0.full.f32le"
FFN_INPUT = BLOCK0_REFERENCE / "prefill8/ffn_inp-0.full.f32le"
FFN_INPUT_TWIN = BLOCK0_REFERENCE / "cached7p1/ffn_inp-0.full.f32le"
REFERENCE_FFN_OUT = BLOCK0_REFERENCE / "prefill8/ffn_out-0.full.f32le"
REFERENCE_LOUT = BLOCK0_REFERENCE / "prefill8/l_out-0.full.f32le"
APPARATUS = HERE / "results/strat01_gigachat_engine_q6k_q8k_avx2_parity_apparatus_repair1_20260924/adjudication.json"
APPARATUS_SHA = "c0771eb574a281c5a98ec858694f1b88fe3b0ef13f2d2d193a47592c9261a584"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q6k_q8k_avx2_parity_20260924"

MODEL_BYTES = 6_474_702_976
MODEL_SHA = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
SWIGLU_SHA = "de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef"
FFN_INPUT_SHA = "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1"
REFERENCE_FFN_OUT_SHA = "f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4"
GENERIC_FFN_OUT_SHA = "f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce"
REFERENCE_LOUT_SHA = "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"
GENERIC_LOUT_SHA = "a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11"
Q8_BYTES = 8 * 35 * 292
OUTPUT_BYTES = 8 * 1536 * 4

EXTRA_INPUTS = {
    "topk": (ao.TOPK, ao.TOPK_SHA, 128),
    "ref_kqv": ao.INPUTS["reference_kqv"],
    "ref_layer1_ffn": ao.INPUTS["reference_ffn_input"],
    "ref_gate": ao.INPUTS["reference_gate"],
    "ref_q": ao.INPUTS["ref_q"],
    "ref_k": ao.INPUTS["ref_k"],
    "ref_shared": ao.INPUTS["ref_shared_out"],
    "ref_weights": ao.INPUTS["ref_weights"],
}

CRITICAL_PATHS = (
    ENGINE, HEADER, PRIMITIVE, PROTOCOL,
    HERE / "strat01_q6k_q8k_avx2_oracle.cpp",
    HERE / "build_strat01_q6k_q8k_avx2_oracle.py",
    HERE / "run_strat01_q6k_q8k_avx2_parity.py",
    HERE / "test_strat01_q6k_q8k_avx2_parity.py",
    Path(__file__).resolve(),
)


class ScientificError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        raise ScientificError(f"file identity/size failure: {path}")
    if digest is not None and sha256_file(path) != digest:
        raise ScientificError(f"file hash failure: {path}")


def clean_sources() -> str:
    for path in CRITICAL_PATHS:
        require_file(path)
    relative = [path.resolve().relative_to(ROOT.resolve()) for path in CRITICAL_PATHS]
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)],
        cwd=ROOT, check=False,
    ).returncode:
        raise ScientificError("scientific sources differ from HEAD")
    for path in relative:
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT,
            capture_output=True, check=False,
        ).returncode:
            raise ScientificError(f"untracked scientific source: {path}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()


def validate_apparatus() -> dict[str, Any]:
    require_file(APPARATUS, digest=APPARATUS_SHA)
    record = json.loads(APPARATUS.read_text(encoding="utf-8"))
    if record.get("status") != "APPARATUS_READY_NO_SCIENTIFIC_EXECUTION" or record.get("errors"):
        raise ScientificError("Q6 parity apparatus is not qualified")
    if any(record.get("counters", {}).values()):
        raise ScientificError("Q6 parity apparatus counters are nonzero")
    hashes = record.get("identity", {}).get("critical_source_hashes", {})
    for path in (ENGINE, HEADER, PRIMITIVE):
        key = str(path.relative_to(ROOT))
        if hashes.get(key) != sha256_file(path):
            raise ScientificError(f"apparatus/source binding mismatch: {key}")
    return record


def validate_evidence() -> None:
    require_file(MODEL, MODEL_BYTES, MODEL_SHA)
    for path, twin, size, digest in (
        (SWIGLU, SWIGLU_TWIN, 286720, SWIGLU_SHA),
        (FFN_INPUT, FFN_INPUT_TWIN, 49152, FFN_INPUT_SHA),
    ):
        require_file(path, size, digest)
        require_file(twin, size, digest)
    require_file(REFERENCE_FFN_OUT, OUTPUT_BYTES, REFERENCE_FFN_OUT_SHA)
    require_file(REFERENCE_LOUT, OUTPUT_BYTES, REFERENCE_LOUT_SHA)
    for name, (path, digest, size) in EXTRA_INPUTS.items():
        require_file(path, size, digest)
        if name != "topk":
            twin = ao.TWINS[{"ref_kqv": "reference_kqv", "ref_layer1_ffn": "reference_ffn_input",
                             "ref_gate": "reference_gate", "ref_q": "ref_q", "ref_k": "ref_k",
                             "ref_shared": "ref_shared_out", "ref_weights": "ref_weights"}[name]]
            require_file(twin, size, digest)
    target, digest, size = ao.REFERENCE_TARGET
    require_file(target, size, digest)
    require_file(ao.REFERENCE_TARGET_TWIN, size, digest)


def run_command(command: list[str], output: Path, label: str, timeout: int = 3600) -> dict[str, Any]:
    started_utc = utc_now()
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False, timeout=timeout,
    )
    stdout_path, stderr_path = output / f"{label}.stdout.log", output / f"{label}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
    return {
        "command": command, "returncode": completed.returncode,
        "started_utc": started_utc, "seconds": time.perf_counter() - started,
        "stdout_sha256": sha256_file(stdout_path), "stderr_sha256": sha256_file(stderr_path),
    }


def output_path(root: Path, item: dict[str, Any], expected_bytes: int) -> Path:
    if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != expected_bytes:
        raise ScientificError("C report output schema mismatch")
    path = Path(item["path"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ScientificError("C report output escaped result directory") from error
    require_file(path, expected_bytes, item["sha256"])
    return path


def validate_c_report(root: Path, source_hashes: dict[str, str]) -> dict[str, Path]:
    report_path = root / "strat01_block0_q6k_q8k_avx2_parity.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("command") != "--strat01-block0-q6k-q8k-avx2-parity" or
            report.get("state") != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or
            report.get("self_certifies_pass") is not False or
            report.get("compiler_family") != "clang"):
        raise ScientificError("C report execution contract mismatch")
    if (report.get("model") != {"path": str(MODEL), "bytes": MODEL_BYTES, "sha256": MODEL_SHA} or
            report.get("matrix") != {"name": "blk.0.ffn_down.weight", "type": "Q6_K",
                                     "shape": [8960, 1536], "offset": 286755840,
                                     "file_offset": 292858752, "span": 11289600}):
        raise ScientificError("C report model/matrix identity mismatch")
    if (report.get("engine_source_sha256") != source_hashes[str(ENGINE.relative_to(ROOT))] or
            report.get("diagnostic_source_sha256") != source_hashes[str(HEADER.relative_to(ROOT))] or
            report.get("q8_populations_completed") != 1 or report.get("q6_matrix_reads") != 1 or
            report.get("diagnostic_executions") != 1 or report.get("donor_graph_executions") != 0 or
            report.get("reference_graph_executions") != 0 or report.get("timing_or_rate_claim") is not None):
        raise ScientificError("C report source/accounting mismatch")
    expected = {
        "q8_population": Q8_BYTES,
        "current_generic_replay": OUTPUT_BYTES,
        "pinned_avx2_candidate": OUTPUT_BYTES,
        "current_generic_lout": OUTPUT_BYTES,
        "pinned_avx2_candidate_lout": OUTPUT_BYTES,
        "control_mutated_q6_row": 32,
        "pinned_avx2_candidate_downstream": 8 * 6144 * 4,
    }
    if set(report.get("outputs", {})) != set(expected):
        raise ScientificError("C report output labels mismatch")
    return {name: output_path(root, report["outputs"][name], size) for name, size in expected.items()}


def classify(q8_exact: bool, candidate_oracle_exact: bool, exact_controls: bool) -> str:
    if not q8_exact:
        return "BLOCK0_Q6_Q8K_QUANTIZER_MISMATCH"
    if not candidate_oracle_exact:
        return "BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT"
    if exact_controls:
        return "BLOCK0_Q6_AVX2_EXACT_REPAIR"
    return "VOID_BLOCK0_Q6K_Q8K_AVX2_PARITY"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ScientificError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "strat01_engine_block0_q6k_q8k_avx2_parity_v1",
        "status": "VOID_BLOCK0_Q6K_Q8K_AVX2_PARITY",
        "started_utc": utc_now(), "commands": [], "errors": [],
        "counters": {"model_reads": 0, "q8_populations": 0, "q6_matrix_reads": 0,
                     "diagnostic_executions": 0, "donor_graph_executions": 0,
                     "reference_graph_executions": 0},
    }
    try:
        head = clean_sources()
        apparatus = validate_apparatus()
        validate_evidence()
        source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS}
        clang = shutil.which("clang")
        if not clang:
            raise ScientificError("clang is unavailable")
        engine = output / "engine_q6_parity.exe"
        compile_record = run_command(
            [clang, "-std=c11", "-O3", "-mavx2", "-mfma", str(ENGINE),
             "-o", str(engine), "-lm"], output, "compile_engine",
        )
        manifest["commands"].append(compile_record)
        if compile_record["returncode"]:
            raise ScientificError("engine compile failed")
        selftest = run_command(
            [str(engine), "--strat01-block0-q6k-q8k-avx2-parity-selftest"],
            output, "engine_selftest",
        )
        manifest["commands"].append(selftest)
        if selftest["returncode"]:
            raise ScientificError("engine Q6 parity selftest failed")
        oracle = build_oracle(output / "oracle-build")
        c_root = output / "c_diagnostic"
        c_root.mkdir()
        command = [
            str(engine), "--strat01-block0-q6k-q8k-avx2-parity", str(MODEL),
            "--input", str(SWIGLU), "--ffn-input", str(FFN_INPUT),
            "--topk", str(EXTRA_INPUTS["topk"][0]),
            "--ref-kqv", str(EXTRA_INPUTS["ref_kqv"][0]),
            "--ref-layer1-ffn", str(EXTRA_INPUTS["ref_layer1_ffn"][0]),
            "--ref-gate", str(EXTRA_INPUTS["ref_gate"][0]),
            "--ref-q", str(EXTRA_INPUTS["ref_q"][0]),
            "--ref-k", str(EXTRA_INPUTS["ref_k"][0]),
            "--ref-shared", str(EXTRA_INPUTS["ref_shared"][0]),
            "--ref-weights", str(EXTRA_INPUTS["ref_weights"][0]),
            "--out-dir", str(c_root),
        ]
        diagnostic = run_command(command, output, "c_diagnostic")
        manifest["commands"].append(diagnostic)
        manifest["counters"].update({"model_reads": 1, "q8_populations": 1,
                                     "q6_matrix_reads": 1, "diagnostic_executions": 1})
        if diagnostic["returncode"]:
            raise ScientificError("C full-matrix diagnostic failed")
        c_outputs = validate_c_report(c_root, source_hashes)
        oracle_q8 = output / "oracle_q8_population.bin"
        oracle_output = output / "oracle_ffn_out.f32le"
        oracle_record = run_command(
            [str(oracle), "matrix", "8960", "1536", "8", "292858752",
             str(MODEL), str(SWIGLU), str(oracle_q8), str(oracle_output)],
            output, "pinned_oracle",
        )
        manifest["commands"].append(oracle_record)
        manifest["counters"].update({"model_reads": 2, "q8_populations": 2,
                                     "q6_matrix_reads": 2})
        if oracle_record["returncode"]:
            raise ScientificError("pinned full-matrix oracle failed")
        require_file(oracle_q8, Q8_BYTES)
        require_file(oracle_output, OUTPUT_BYTES)
        q8_exact = c_outputs["q8_population"].read_bytes() == oracle_q8.read_bytes()
        candidate_oracle_exact = (
            c_outputs["pinned_avx2_candidate"].read_bytes() == oracle_output.read_bytes()
        )
        candidate_reference_exact = sha256_file(c_outputs["pinned_avx2_candidate"]) == REFERENCE_FFN_OUT_SHA
        oracle_reference_exact = sha256_file(oracle_output) == REFERENCE_FFN_OUT_SHA
        generic_history_exact = sha256_file(c_outputs["current_generic_replay"]) == GENERIC_FFN_OUT_SHA
        generic_negative = c_outputs["current_generic_replay"].read_bytes() != oracle_output.read_bytes()
        candidate_lout_exact = sha256_file(c_outputs["pinned_avx2_candidate_lout"]) == REFERENCE_LOUT_SHA
        generic_lout_history_exact = sha256_file(c_outputs["current_generic_lout"]) == GENERIC_LOUT_SHA
        target, target_sha, _ = ao.REFERENCE_TARGET
        downstream_exact = sha256_file(c_outputs["pinned_avx2_candidate_downstream"]) == target_sha
        candidate_bytes = c_outputs["pinned_avx2_candidate"].read_bytes()
        mutation = struct.unpack("<8f", c_outputs["control_mutated_q6_row"].read_bytes())
        baseline = tuple(struct.unpack_from("<f", candidate_bytes, item * 1536 * 4)[0] for item in range(8))
        mutation_fires = any(struct.pack("<f", left) != struct.pack("<f", right)
                             for left, right in zip(mutation, baseline))
        controls = {
            "q8_population_exact": q8_exact,
            "candidate_oracle_exact": candidate_oracle_exact,
            "candidate_reference_exact": candidate_reference_exact,
            "oracle_reference_exact": oracle_reference_exact,
            "generic_history_exact": generic_history_exact,
            "generic_negative_control_fires": generic_negative,
            "candidate_lout_reference_exact": candidate_lout_exact,
            "generic_lout_history_exact": generic_lout_history_exact,
            "candidate_downstream_reference_exact": downstream_exact,
            "one_byte_q6_mutation_fires": mutation_fires,
            "schedule_twins_exact": True,
            "zero_graph_executions": True,
        }
        exact_controls = all(controls.values())
        status = classify(q8_exact, candidate_oracle_exact, exact_controls)
        manifest.update({
            "status": status, "finished_utc": utc_now(), "controls": controls,
            "identity": {
                "git_head": head, "model": {"path": str(MODEL), "bytes": MODEL_BYTES, "sha256": MODEL_SHA},
                "apparatus": {"path": str(APPARATUS), "sha256": APPARATUS_SHA,
                              "observed_head": apparatus["identity"]["git_head_observed"]},
                "critical_source_hashes": source_hashes,
                "engine_binary_sha256": sha256_file(engine),
                "oracle_binary_sha256": sha256_file(oracle),
            },
            "outputs": {
                "c_q8_sha256": sha256_file(c_outputs["q8_population"]),
                "oracle_q8_sha256": sha256_file(oracle_q8),
                "candidate_sha256": sha256_file(c_outputs["pinned_avx2_candidate"]),
                "oracle_sha256": sha256_file(oracle_output),
                "generic_sha256": sha256_file(c_outputs["current_generic_replay"]),
                "candidate_lout_sha256": sha256_file(c_outputs["pinned_avx2_candidate_lout"]),
                "candidate_downstream_sha256": sha256_file(c_outputs["pinned_avx2_candidate_downstream"]),
            },
            "non_claims": ["quality", "generation", "RAM", "throughput", "full-model repair"],
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
    return 0 if manifest["status"] in {
        "BLOCK0_Q6_AVX2_EXACT_REPAIR", "BLOCK0_Q6_Q8K_QUANTIZER_MISMATCH",
        "BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT",
    } else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScientificError as error:
        raise SystemExit(f"run_strat01_q6k_q8k_avx2_scientific: {error}")
