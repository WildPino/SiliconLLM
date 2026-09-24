#!/usr/bin/env python3
"""Offline repair-1 adjudication of the valid Q6 AVX2 scientific evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine.run_strat01_q6k_q8k_avx2_scientific import (
    GENERIC_FFN_OUT_SHA,
    GENERIC_LOUT_SHA,
    REFERENCE_FFN_OUT,
    REFERENCE_FFN_OUT_SHA,
    REFERENCE_LOUT,
    REFERENCE_LOUT_SHA,
    classify,
)
from benchmarks.donor_adaptation.engine import (
    run_strat01_layer1_attention_output_residual_cross_input as ao,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/strat01_gigachat_engine_q6k_q8k_avx2_parity_20260924"
SOURCE_ADJUDICATION = SOURCE / "adjudication.json"
SOURCE_ADJUDICATION_SHA = "c38fad6017cb1524bc92025410cd16c70398c5387a2cb9357a9fd801aa42cad2"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q6k_q8k_avx2_parity_readjudication1_20260924"


class RejudgeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path, size: int, digest: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise RejudgeError(f"immutable evidence mismatch: {path}")


def metrics(actual: Path, reference: Path) -> dict[str, float | int]:
    left = np.fromfile(actual, dtype="<f4").astype(np.float64)
    right = np.fromfile(reference, dtype="<f4").astype(np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RejudgeError("metric payload shape/finiteness mismatch")
    delta = left - right
    return {
        "elements": int(left.size),
        "changed": int(np.count_nonzero(left != right)),
        "nrmse": float(np.sqrt(np.mean(delta * delta)) / np.sqrt(np.mean(right * right))),
        "nmax": float(np.max(np.abs(delta)) / np.max(np.abs(right))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RejudgeError("output directory already exists; offline adjudication is immutable")
    output.mkdir(parents=True)
    result: dict[str, object] = {
        "schema": "strat01_engine_block0_q6k_q8k_avx2_readjudication_v1",
        "status": "VOID_BLOCK0_Q6K_Q8K_AVX2_PARITY",
        "started_utc": utc_now(), "errors": [],
        "counters": {"new_model_reads": 0, "new_q8_populations": 0,
                     "new_q6_matrix_reads": 0, "new_diagnostic_executions": 0,
                     "new_donor_graph_executions": 0, "new_reference_graph_executions": 0},
    }
    try:
        require(SOURCE_ADJUDICATION, SOURCE_ADJUDICATION.stat().st_size, SOURCE_ADJUDICATION_SHA)
        source = json.loads(SOURCE_ADJUDICATION.read_text(encoding="utf-8"))
        if (source.get("status") != "VOID_BLOCK0_Q6K_Q8K_AVX2_PARITY" or source.get("errors") or
                source.get("identity", {}).get("git_head") != "e46205390b2e4117ae8a63b1946f5b2c2773d6e1"):
            raise RejudgeError("source scientific record contract mismatch")
        expected_counters = {"model_reads": 2, "q8_populations": 2, "q6_matrix_reads": 2,
                             "diagnostic_executions": 1, "donor_graph_executions": 0,
                             "reference_graph_executions": 0}
        if source.get("counters") != expected_counters:
            raise RejudgeError("source scientific execution accounting mismatch")
        c_root = SOURCE / "c_diagnostic"
        paths = {
            "candidate": c_root / "pinned_avx2_candidate.f32le",
            "oracle": SOURCE / "oracle_ffn_out.f32le",
            "generic": c_root / "current_generic_replay.f32le",
            "candidate_lout": c_root / "pinned_avx2_candidate_lout.f32le",
            "generic_lout": c_root / "current_generic_lout.f32le",
            "downstream": c_root / "pinned_avx2_candidate_downstream.f32le",
            "c_q8": c_root / "q8_population.bin",
            "oracle_q8": SOURCE / "oracle_q8_population.bin",
            "mutation": c_root / "control_mutated_q6_row.f32le",
        }
        outputs = source["outputs"]
        require(paths["candidate"], 49152, outputs["candidate_sha256"])
        require(paths["oracle"], 49152, outputs["oracle_sha256"])
        require(paths["generic"], 49152, outputs["generic_sha256"])
        require(paths["candidate_lout"], 49152, outputs["candidate_lout_sha256"])
        require(paths["downstream"], 196608, outputs["candidate_downstream_sha256"])
        require(paths["c_q8"], 81760, outputs["c_q8_sha256"])
        require(paths["oracle_q8"], 81760, outputs["oracle_q8_sha256"])
        require(paths["generic_lout"], 49152, GENERIC_LOUT_SHA)
        if not paths["mutation"].is_file() or paths["mutation"].stat().st_size != 32:
            raise RejudgeError("mutation control payload mismatch")
        controls = dict(source["controls"])
        expected_controls = {
            "q8_population_exact": True, "candidate_oracle_exact": True,
            "candidate_reference_exact": False, "oracle_reference_exact": False,
            "generic_history_exact": True, "generic_negative_control_fires": True,
            "candidate_lout_reference_exact": False, "generic_lout_history_exact": True,
            "candidate_downstream_reference_exact": False,
            "one_byte_q6_mutation_fires": True, "schedule_twins_exact": True,
            "zero_graph_executions": True,
        }
        if controls != expected_controls:
            raise RejudgeError("source scientific controls mismatch")
        if (sha256_file(paths["candidate"]) == REFERENCE_FFN_OUT_SHA or
                sha256_file(paths["oracle"]) == REFERENCE_FFN_OUT_SHA or
                sha256_file(paths["generic"]) != GENERIC_FFN_OUT_SHA or
                sha256_file(paths["candidate_lout"]) == REFERENCE_LOUT_SHA):
            raise RejudgeError("source scientific hash relation mismatch")
        status = classify(True, True, False, False)
        if status != "BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT":
            raise RejudgeError("repaired decision table did not close the frozen branch")
        target, target_sha, target_size = ao.REFERENCE_TARGET
        require(target, target_size, target_sha)
        require(REFERENCE_FFN_OUT, 49152, REFERENCE_FFN_OUT_SHA)
        require(REFERENCE_LOUT, 49152, REFERENCE_LOUT_SHA)
        result.update({
            "status": status, "finished_utc": utc_now(),
            "source_scientific_record": {
                "path": str(SOURCE_ADJUDICATION), "sha256": SOURCE_ADJUDICATION_SHA,
                "status": source["status"], "git_head": source["identity"]["git_head"],
                "reused_counters": source["counters"],
            },
            "decision": {
                "q8_population_exact": True,
                "candidate_oracle_exact": True,
                "candidate_reference_exact": False,
                "meaning": "pinned active AVX2 vec-dot semantics do not reproduce the immutable graph output",
            },
            "metrics": {
                "candidate_vs_reference_ffn_out": metrics(paths["candidate"], REFERENCE_FFN_OUT),
                "candidate_vs_reference_lout": metrics(paths["candidate_lout"], REFERENCE_LOUT),
                "candidate_vs_reference_downstream": metrics(paths["downstream"], target),
            },
            "controls": controls,
            "non_claims": ["quality", "generation", "RAM", "throughput", "full-model repair"],
        })
    except Exception as error:
        result["errors"].append(str(error))
        result["finished_utc"] = utc_now()
    result["environment"] = {
        "platform": platform.platform(), "python": sys.version, "cwd": os.getcwd(),
    }
    adjudication = output / "adjudication.json"
    adjudication.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(result["status"])
    if result["errors"]:
        print(result["errors"][0], file=sys.stderr)
    return 0 if result["status"] == "BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RejudgeError as error:
        raise SystemExit(f"rejudge_strat01_q6k_q8k_avx2_scientific: {error}")
