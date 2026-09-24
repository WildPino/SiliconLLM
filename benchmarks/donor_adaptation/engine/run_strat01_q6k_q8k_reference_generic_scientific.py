#!/usr/bin/env python3
"""Run the sole zero-graph Q6_K/Q8_K reference-generic scientific cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import (  # noqa: E402
    run_strat01_q6k_q8k_avx2_scientific as base,
)
from benchmarks.donor_adaptation.engine.build_strat01_q6k_q8k_reference_generic_oracle import (  # noqa: E402
    build as build_oracle,
)

HERE = Path(__file__).resolve().parent
ENGINE = base.ENGINE
HEADER = base.HEADER
PRIMITIVE = base.PRIMITIVE
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260924.md"
MODEL = base.MODEL
APPARATUS = HERE / "results/strat01_gigachat_engine_q6k_q8k_reference_generic_parity_apparatus_repair2_20260924/adjudication.json"
APPARATUS_SHA = "60675f36971f9fd67982ecfcdd1eac8aa5ffa9a3f0c174cd63efc8f7b867e838"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q6k_q8k_reference_generic_parity_20260924"

ACTIVE_FFN_OUT_SHA = "3d0da87b33d531458bd0ca6529e1be8a715b67ff88b03ab82541f4d47fa445ec"

CRITICAL_PATHS = (
    ENGINE, HEADER, PRIMITIVE, PROTOCOL,
    HERE / "strat01_q6k_q8k_reference_generic_probe.c",
    HERE / "strat01_q6k_q8k_reference_generic_oracle.cpp",
    HERE / "build_strat01_q6k_q8k_reference_generic_oracle.py",
    HERE / "test_strat01_q6k_q8k_reference_generic_parity.py",
    HERE / "run_strat01_q6k_q8k_reference_generic_parity.py",
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
        raise ScientificError("reference-generic apparatus is not qualified")
    if any(record.get("counters", {}).values()):
        raise ScientificError("reference-generic apparatus counters are nonzero")
    for relative, digest in record.get("identity", {}).get("critical_source_hashes", {}).items():
        path = ROOT / Path(relative)
        require_file(path, digest=digest)
    return record


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
    report = json.loads(
        (root / "strat01_block0_q6k_q8k_reference_generic_parity.json").read_text(
            encoding="utf-8"
        )
    )
    if (report.get("command") != "--strat01-block0-q6k-q8k-reference-generic-parity" or
            report.get("state") != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or
            report.get("self_certifies_pass") is not False or
            report.get("compiler_family") != "clang"):
        raise ScientificError("C report execution contract mismatch")
    if (report.get("model") != {
            "path": str(MODEL), "bytes": base.MODEL_BYTES, "sha256": base.MODEL_SHA,
        } or report.get("matrix") != {
            "name": "blk.0.ffn_down.weight", "type": "Q6_K", "shape": [8960, 1536],
            "offset": 286755840, "file_offset": 292858752, "span": 11289600,
        }):
        raise ScientificError("C report model/matrix identity mismatch")
    if (report.get("engine_source_sha256") != source_hashes[str(ENGINE.relative_to(ROOT))] or
            report.get("diagnostic_source_sha256") != source_hashes[str(HEADER.relative_to(ROOT))] or
            report.get("q8_populations_completed") != 1 or
            report.get("q6_matrix_reads") != 1 or
            report.get("diagnostic_executions") != 1 or
            report.get("donor_graph_executions") != 0 or
            report.get("reference_graph_executions") != 0 or
            report.get("timing_or_rate_claim") is not None):
        raise ScientificError("C report source/accounting mismatch")
    expected = {
        "q8_population": base.Q8_BYTES,
        "historical_avx_tu_generic": base.OUTPUT_BYTES,
        "reference_generic_candidate": base.OUTPUT_BYTES,
        "historical_avx_tu_generic_lout": base.OUTPUT_BYTES,
        "reference_generic_candidate_lout": base.OUTPUT_BYTES,
        "control_mutated_q6_row": 32,
        "closed_active_avx2_control": base.OUTPUT_BYTES,
        "reference_generic_candidate_downstream": 8 * 6144 * 4,
    }
    if set(report.get("outputs", {})) != set(expected):
        raise ScientificError("C report output labels mismatch")
    return {
        name: output_path(root, report["outputs"][name], size)
        for name, size in expected.items()
    }


def compare_f32(left_path: Path, right_path: Path) -> dict[str, float | int]:
    left_bytes, right_bytes = left_path.read_bytes(), right_path.read_bytes()
    if len(left_bytes) != len(right_bytes) or len(left_bytes) % 4:
        raise ScientificError("comparison byte count mismatch")
    left = struct.unpack(f"<{len(left_bytes) // 4}f", left_bytes)
    right = struct.unpack(f"<{len(right_bytes) // 4}f", right_bytes)
    squared_delta = squared_right = 0.0
    changed = 0
    maximum = reference_maximum = 0.0
    for a, b in zip(left, right):
        delta = abs(float(a) - float(b))
        squared_delta += delta * delta
        squared_right += float(b) * float(b)
        maximum = max(maximum, delta)
        reference_maximum = max(reference_maximum, abs(float(b)))
        changed += struct.pack("<f", a) != struct.pack("<f", b)
    return {
        "changed": changed,
        "count": len(left),
        "nrmse": math.sqrt(squared_delta / len(left)) /
                 math.sqrt(squared_right / len(left)) if squared_right else 0.0,
        "normalized_max": maximum / reference_maximum if reference_maximum else 0.0,
        "max_abs": maximum,
    }


def classify(
    q8_exact: bool,
    candidate_oracle_exact: bool,
    candidate_reference_exact: bool,
    lout_exact: bool,
    downstream_exact: bool,
    ancillary_controls: bool,
) -> str:
    if not q8_exact or not ancillary_controls:
        return "VOID_BLOCK0_Q6_REFERENCE_GENERIC_COMPILE_PARITY"
    if not candidate_oracle_exact:
        return "BLOCK0_Q6_REFERENCE_GENERIC_IMPLEMENTATION_MISMATCH"
    if not candidate_reference_exact or not lout_exact or not downstream_exact:
        return "BLOCK0_Q6_REFERENCE_GENERIC_COMPILE_INSUFFICIENT"
    return "BLOCK0_Q6_REFERENCE_GENERIC_EXACT_REPAIR"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ScientificError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "strat01_engine_block0_q6k_q8k_reference_generic_parity_v1",
        "status": "VOID_BLOCK0_Q6_REFERENCE_GENERIC_COMPILE_PARITY",
        "started_utc": utc_now(), "commands": [], "errors": [],
        "counters": {
            "model_reads": 0, "q8_populations": 0, "q6_matrix_reads": 0,
            "diagnostic_executions": 0, "donor_graph_executions": 0,
            "reference_graph_executions": 0,
        },
    }
    try:
        head = clean_sources()
        apparatus = validate_apparatus()
        base.validate_evidence()
        source_hashes = {
            str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS
        }
        clang = shutil.which("clang")
        if not clang:
            raise ScientificError("clang is unavailable")
        engine = output / "engine_q6_reference_generic.exe"
        compile_record = base.run_command(
            [clang, "-std=c11", "-O3", "-mavx2", "-mfma", str(ENGINE),
             "-o", str(engine), "-lm"], output, "compile_engine",
        )
        manifest["commands"].append(compile_record)
        if compile_record["returncode"]:
            raise ScientificError("engine compile failed")
        selftest = base.run_command(
            [str(engine), "--strat01-block0-q6k-q8k-reference-generic-parity-selftest"],
            output, "engine_selftest",
        )
        manifest["commands"].append(selftest)
        if selftest["returncode"]:
            raise ScientificError("engine reference-generic selftest failed")
        oracle = build_oracle(output / "oracle-build")
        c_root = output / "c_diagnostic"
        c_root.mkdir()
        command = [
            str(engine), "--strat01-block0-q6k-q8k-reference-generic-parity", str(MODEL),
            "--input", str(base.SWIGLU), "--ffn-input", str(base.FFN_INPUT),
            "--topk", str(base.EXTRA_INPUTS["topk"][0]),
            "--ref-kqv", str(base.EXTRA_INPUTS["ref_kqv"][0]),
            "--ref-layer1-ffn", str(base.EXTRA_INPUTS["ref_layer1_ffn"][0]),
            "--ref-gate", str(base.EXTRA_INPUTS["ref_gate"][0]),
            "--ref-q", str(base.EXTRA_INPUTS["ref_q"][0]),
            "--ref-k", str(base.EXTRA_INPUTS["ref_k"][0]),
            "--ref-shared", str(base.EXTRA_INPUTS["ref_shared"][0]),
            "--ref-weights", str(base.EXTRA_INPUTS["ref_weights"][0]),
            "--out-dir", str(c_root),
        ]
        diagnostic = base.run_command(command, output, "c_diagnostic")
        manifest["commands"].append(diagnostic)
        manifest["counters"].update({
            "model_reads": 1, "q8_populations": 1,
            "q6_matrix_reads": 1, "diagnostic_executions": 1,
        })
        if diagnostic["returncode"]:
            raise ScientificError("C full-matrix diagnostic failed")
        c_outputs = validate_c_report(c_root, source_hashes)
        oracle_q8 = output / "oracle_q8_population.bin"
        oracle_output = output / "oracle_ffn_out.f32le"
        oracle_record = base.run_command(
            [str(oracle), "matrix", "8960", "1536", "8", "292858752",
             str(MODEL), str(base.SWIGLU), str(oracle_q8), str(oracle_output)],
            output, "pinned_generic_oracle",
        )
        manifest["commands"].append(oracle_record)
        manifest["counters"].update({
            "model_reads": 2, "q8_populations": 2, "q6_matrix_reads": 2,
        })
        if oracle_record["returncode"]:
            raise ScientificError("pinned full-matrix generic oracle failed")
        require_file(oracle_q8, base.Q8_BYTES)
        require_file(oracle_output, base.OUTPUT_BYTES)
        candidate = c_outputs["reference_generic_candidate"]
        generic = c_outputs["historical_avx_tu_generic"]
        active = c_outputs["closed_active_avx2_control"]
        candidate_lout = c_outputs["reference_generic_candidate_lout"]
        downstream = c_outputs["reference_generic_candidate_downstream"]
        q8_exact = c_outputs["q8_population"].read_bytes() == oracle_q8.read_bytes()
        candidate_oracle_exact = candidate.read_bytes() == oracle_output.read_bytes()
        candidate_reference_exact = sha256_file(candidate) == base.REFERENCE_FFN_OUT_SHA
        oracle_reference_exact = sha256_file(oracle_output) == base.REFERENCE_FFN_OUT_SHA
        generic_history_exact = sha256_file(generic) == base.GENERIC_FFN_OUT_SHA
        active_history_exact = sha256_file(active) == ACTIVE_FFN_OUT_SHA
        candidate_lout_exact = sha256_file(candidate_lout) == base.REFERENCE_LOUT_SHA
        generic_lout_history_exact = (
            sha256_file(c_outputs["historical_avx_tu_generic_lout"]) == base.GENERIC_LOUT_SHA
        )
        target, target_sha, _ = base.ao.REFERENCE_TARGET
        downstream_exact = sha256_file(downstream) == target_sha
        distinct_arms = len({candidate.read_bytes(), generic.read_bytes(), active.read_bytes()}) == 3
        candidate_bytes = candidate.read_bytes()
        mutation = struct.unpack("<8f", c_outputs["control_mutated_q6_row"].read_bytes())
        baseline = tuple(
            struct.unpack_from("<f", candidate_bytes, item * 1536 * 4)[0]
            for item in range(8)
        )
        mutation_fires = any(
            struct.pack("<f", left) != struct.pack("<f", right)
            for left, right in zip(mutation, baseline)
        )
        controls = {
            "q8_population_exact": q8_exact,
            "candidate_oracle_exact": candidate_oracle_exact,
            "candidate_reference_exact": candidate_reference_exact,
            "oracle_reference_exact": oracle_reference_exact,
            "historical_avx_tu_generic_exact": generic_history_exact,
            "closed_active_avx2_exact": active_history_exact,
            "three_arms_pairwise_distinct": distinct_arms,
            "candidate_lout_reference_exact": candidate_lout_exact,
            "historical_generic_lout_exact": generic_lout_history_exact,
            "candidate_downstream_reference_exact": downstream_exact,
            "one_byte_q6_mutation_fires": mutation_fires,
            "schedule_twins_exact": True,
            "zero_graph_executions": True,
        }
        ancillary = all(value for name, value in controls.items() if name not in {
            "q8_population_exact", "candidate_oracle_exact", "candidate_reference_exact",
            "oracle_reference_exact", "candidate_lout_reference_exact",
            "candidate_downstream_reference_exact",
        }) and oracle_reference_exact
        status = classify(
            q8_exact, candidate_oracle_exact, candidate_reference_exact,
            candidate_lout_exact, downstream_exact, ancillary,
        )
        manifest.update({
            "status": status, "finished_utc": utc_now(), "controls": controls,
            "metrics": {
                "candidate_vs_reference": compare_f32(candidate, base.REFERENCE_FFN_OUT),
                "oracle_vs_reference": compare_f32(oracle_output, base.REFERENCE_FFN_OUT),
                "candidate_lout_vs_reference": compare_f32(candidate_lout, base.REFERENCE_LOUT),
                "candidate_downstream_vs_reference": compare_f32(downstream, target),
            },
            "identity": {
                "git_head": head,
                "model": {"path": str(MODEL), "bytes": base.MODEL_BYTES, "sha256": base.MODEL_SHA},
                "apparatus": {
                    "path": str(APPARATUS), "sha256": APPARATUS_SHA,
                    "observed_head": apparatus["identity"]["git_head_observed"],
                },
                "critical_source_hashes": source_hashes,
                "engine_binary_sha256": sha256_file(engine),
                "oracle_binary_sha256": sha256_file(oracle),
            },
            "outputs": {
                "c_q8_sha256": sha256_file(c_outputs["q8_population"]),
                "oracle_q8_sha256": sha256_file(oracle_q8),
                "candidate_sha256": sha256_file(candidate),
                "oracle_sha256": sha256_file(oracle_output),
                "historical_generic_sha256": sha256_file(generic),
                "closed_active_avx2_sha256": sha256_file(active),
                "candidate_lout_sha256": sha256_file(candidate_lout),
                "candidate_downstream_sha256": sha256_file(downstream),
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
        "BLOCK0_Q6_REFERENCE_GENERIC_EXACT_REPAIR",
        "BLOCK0_Q6_REFERENCE_GENERIC_COMPILE_INSUFFICIENT",
        "BLOCK0_Q6_REFERENCE_GENERIC_IMPLEMENTATION_MISMATCH",
    } else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScientificError as error:
        raise SystemExit(f"run_strat01_q6k_q8k_reference_generic_scientific: {error}")
