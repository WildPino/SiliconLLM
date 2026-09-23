#!/usr/bin/env python3
"""Adjudicate reference-generic Q4 propagation with frozen Rung-2C references."""
from __future__ import annotations

import argparse
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2c as r2c
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as q4base

HERE = Path(__file__).resolve().parent
ENGINE = r2c.ENGINE
Q4_HEADER = ROOT / "benchmarks/phase60/strat01_q4k_q8k.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_REFERENCE_GENERIC_PROPAGATION_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_reference_generic_propagation.py"
MODEL = r2c.MODEL
REFERENCE_SOURCE_RUN = HERE / "results/strat01_gigachat_engine_rung2c_repair1_20260923"
REFERENCE_ROOT = REFERENCE_SOURCE_RUN / "pinned_reference"
RUNG2C_ADJUDICATION = HERE / "results/strat01_gigachat_engine_rung2c_repair1_recovery1_20260923/adjudication.json"
RUNG2C_ADJUDICATION_SHA = "3742bc5982dd47be36a8e422f7b79c9e6716dc628153e44e6f9e90553c9ba62c"
Q4_ADJUDICATION = HERE / "results/strat01_gigachat_engine_q4k_q8k_reference_generic_parity_20260923/adjudication.json"
Q4_ADJUDICATION_SHA = "f0371d9c98e762e6434907ea5c386a7b622adac30dae9939a101fdc22df995f1"
REFERENCE_MANIFEST_SHAS = {
    "manifest.json": "9a00ea08a47482323055a7fa8fcfa6c0e079ea2ed4957665bd491c1abfec21eb",
    "prefill8/manifest.json": "d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451",
    "cached7p1/manifest.json": "ad214e8d38ee51a3b0f0629bdadf7cc692f6806499e1347c90d46aba937a39b9",
}
OLD_C_START_SHA = r2c.C_START_SHA
REFERENCE_START_SHA = r2c.REFERENCE_START_SHA
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_reference_generic_propagation_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_reference_generic_propagation_apparatus_20260923"
TEST_MODULES = tuple(
    "benchmarks.donor_adaptation.engine." + path.stem
    for path in sorted(HERE.glob("test_strat01_*.py"))
)


class PropagationError(RuntimeError):
    pass


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PropagationError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "tests": TESTS,
        "protocol": PROTOCOL,
        "engine": ENGINE,
        "q4_header": Q4_HEADER,
        "rung2a_header": r2c.RUNG2A_HEADER,
        "rung2b_header": r2c.RUNG2B_HEADER,
        "rung2c_header": r2c.RUNG2C_HEADER,
        "rung2c_runner": Path(r2c.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise PropagationError("missing source(s): " + ", ".join(missing))
    return {
        name: {"path": str(path), "sha256": r2c.base.sha256_file(path)}
        for name, path in paths.items()
    }


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)],
        cwd=ROOT, check=False,
    ).returncode:
        raise PropagationError("propagation implementation/protocol differs from HEAD")
    for path in relative:
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=ROOT, capture_output=True, check=False,
        ).returncode:
            raise PropagationError(f"propagation source is untracked: {path}")


def validate_predecessors() -> dict[str, Any]:
    if r2c.base.sha256_file(Q4_ADJUDICATION) != Q4_ADJUDICATION_SHA:
        raise PropagationError("reference-generic Q4 adjudication hash mismatch")
    q4 = read_json(Q4_ADJUDICATION, "reference-generic Q4 adjudication")
    if (
        q4.get("status") != "PASS_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY"
        or q4.get("errors") != []
        or not q4.get("controls")
        or not all(q4["controls"].values())
        or q4.get("identity", {}).get("model", {}).get("sha256") != r2c.base.EXPECTED_SHA256
    ):
        raise PropagationError("reference-generic Q4 predecessor state mismatch")

    if r2c.base.sha256_file(RUNG2C_ADJUDICATION) != RUNG2C_ADJUDICATION_SHA:
        raise PropagationError("canonical Rung-2C adjudication hash mismatch")
    rung2c = read_json(RUNG2C_ADJUDICATION, "canonical Rung-2C adjudication")
    inner = rung2c.get("adjudication", {})
    if (
        rung2c.get("status") != "FAIL_ENGINE_RUNG2C"
        or rung2c.get("errors") != []
        or rung2c.get("donor_graph_executions") != 2
        or rung2c.get("reference_graph_executions") != 2
        or "checkpoint/prefill8/kqv_out-1" not in inner.get("failures", [])
        or "checkpoint/cached7p1/kqv_out-1" not in inner.get("failures", [])
    ):
        raise PropagationError("canonical Rung-2C predecessor state mismatch")

    observed: dict[str, str] = {}
    for relative, expected in REFERENCE_MANIFEST_SHAS.items():
        path = REFERENCE_ROOT / relative
        observed[relative] = r2c.base.sha256_file(path)
        if observed[relative] != expected:
            raise PropagationError(f"immutable reference manifest hash mismatch: {relative}")
    return {
        "reference_generic_q4": {"path": str(Q4_ADJUDICATION), "sha256": Q4_ADJUDICATION_SHA},
        "canonical_rung2c": {"path": str(RUNG2C_ADJUDICATION), "sha256": RUNG2C_ADJUDICATION_SHA},
        "reference_manifests": observed,
    }


def source_controls() -> dict[str, bool]:
    q4 = Q4_HEADER.read_text(encoding="utf-8")
    base_controls = r2c.source_controls()
    return {
        "rung2c_source_controls": all(base_controls.values()),
        "reference_generic_target_isolated": (
            '__attribute__((noinline, target("no-avx,no-avx2,no-fma")))' in q4
            and "strat01_q4k_q8k_dot_reference_generic" in q4
        ),
        "production_dispatches_reference_generic": (
            "#elif defined(STRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2)" in q4
            and "return strat01_q4k_q8k_dot_reference_generic(q4_blocks, q8_blocks, count);" in q4
        ),
        "historical_generic_control_retained": (
            "STRAT01_Q4K_Q8K_DIAGNOSTIC_GENERIC_REDUCTION" in q4
            and "return strat01_q4k_q8k_dot_generic(q4_blocks, q8_blocks, count);" in q4
        ),
        "active_avx2_control_retained": (
            "STRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2" in q4
            and "return strat01_q4k_q8k_dot_avx2(q4_blocks, q8_blocks, count);" in q4
        ),
    }


def intervention_controls(candidate_meta: dict[str, Any], reference_meta: dict[str, Any]) -> dict[str, bool]:
    candidate_hashes = {
        arm: candidate_meta["manifests"][arm]["tensors"][0]["sha256"]
        for arm in r2c.base.ARMS
    }
    reference_hashes = {
        arm: next(
            item["sha256"]
            for item in reference_meta["manifests"][arm]["payloads"]
            if item.get("logical") == "l_out-0" and item.get("kind") == "full"
        )
        for arm in r2c.base.ARMS
    }
    return {
        "candidate_schedules_share_start": len(set(candidate_hashes.values())) == 1,
        "changed_coordinate_reaches_graph": all(value != OLD_C_START_SHA for value in candidate_hashes.values()),
        "reference_start_identity": all(value == REFERENCE_START_SHA for value in reference_hashes.values()),
    }


def adjudicate(
    candidate: dict[str, Any], reference: dict[str, Any],
    candidate_meta: dict[str, Any], reference_meta: dict[str, Any],
    controls: dict[str, bool],
) -> dict[str, Any]:
    inherited = r2c.adjudicate(candidate, reference, candidate_meta, reference_meta, controls)
    inherited_failures = list(inherited["failures"])
    unexpected = [item for item in inherited_failures if item.startswith("start_state/")]
    failures = [item for item in inherited_failures if not item.startswith("start_state/")]
    intervention = intervention_controls(candidate_meta, reference_meta)
    if len(unexpected) != len(r2c.base.ARMS):
        raise PropagationError("inherited Rung-2C start-state control did not fire for both changed schedules")
    if not all(intervention.values()):
        raise PropagationError("changed-coordinate intervention control failed")
    inherited.update({
        "status": "PASS_ENGINE_REFERENCE_GENERIC_PROPAGATION" if not failures else "FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION",
        "failures": failures,
        "superseded_old_start_failures": unexpected,
        "intervention_controls": intervention,
        "original_c_start_sha256": OLD_C_START_SHA,
    })
    return inherited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    status = "VOID_ENGINE_REFERENCE_GENERIC_PROPAGATION"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    predecessors: dict[str, Any] = {}
    controls: dict[str, bool] = {}
    reference_meta: dict[str, Any] = {}
    candidate_meta: dict[str, Any] = {}
    result: dict[str, Any] = {"status": "NOT_RUN"}
    compiler = shutil.which("clang")
    binary: Path | None = None
    donor_invocations = 0
    donor_graphs = 0
    try:
        sources = source_inventory()
        predecessors = validate_predecessors()
        controls = source_controls()
        if not all(controls.values()):
            raise PropagationError("source controls failed")
        if not compiler:
            raise PropagationError("clang is unavailable")
        reference, reference_meta = r2c.validate_reference(REFERENCE_ROOT, model)

        binary = output / "engine_reference_generic_propagation.exe"
        commands["compile_c"] = r2c.base.run_command(
            [compiler, *r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"],
            output=output, label="compile_c", timeout=600,
        )
        r2c.base.require_ok(commands["compile_c"], "C build")
        commands["python_tests"] = r2c.base.run_command(
            [sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES],
            output=output, label="all_strat01_unittests", timeout=1800,
        )
        r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
        for index, option in enumerate(q4base.SELFTESTS):
            label = f"selftest_{index:02d}"
            commands[label] = r2c.base.run_command(
                [str(binary), option], output=output, label=label, timeout=300,
            )
            r2c.base.require_ok(commands[label], option)

        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            if (
                not model.is_file()
                or model.stat().st_size != r2c.base.EXPECTED_BYTES
                or r2c.base.sha256_file(model) != r2c.base.EXPECTED_SHA256
            ):
                raise PropagationError("accepted artifact identity mismatch")
            c_root = output / "c_engine"
            c_root.mkdir()
            donor_invocations = 1
            commands["accepted_artifact_c_engine"] = r2c.base.run_command(
                [str(binary), "--strat01-gguf-rung2c", str(model), "--out-dir", str(c_root)],
                output=output, label="accepted_artifact_c_engine", timeout=21600,
            )
            donor_graphs = r2c.completed_graph_count(commands["accepted_artifact_c_engine"])
            r2c.base.require_ok(commands["accepted_artifact_c_engine"], "C producer")
            if donor_graphs != 2:
                raise PropagationError("C producer did not emit exactly two graph-completion markers")
            candidate, candidate_meta = r2c.validate_c(c_root, sources, model)
            result = adjudicate(candidate, reference, candidate_meta, reference_meta, controls)
            status = result["status"]
    except (PropagationError, r2c.RunnerError, r2c.base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")

    provenance = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "git_head_observed" if args.apparatus_only else "git_head": r2c.base.git_value(["git", "rev-parse", "HEAD"]),
        "source_hashes": sources,
        "predecessors": predecessors,
        "artifact": {
            "path": str(model),
            "expected_bytes": r2c.base.EXPECTED_BYTES,
            "expected_sha256": r2c.base.EXPECTED_SHA256,
        },
        "environment": {
            "platform": platform.platform(), "python": sys.version,
            "cwd": os.getcwd(), "clang_path": compiler,
        },
        "binary": {
            "path": str(binary) if binary else None,
            "sha256": r2c.base.sha256_file(binary) if binary and binary.is_file() else None,
        },
        "commands": commands,
    }
    record = {
        "schema": "strat01_engine_reference_generic_propagation_v1",
        "status": status,
        "errors": errors,
        "donor_producer_invocations": donor_invocations,
        "donor_graph_executions": donor_graphs,
        "reference_producer_invocations": 0,
        "reference_graph_executions": 0,
        "source_controls": controls,
        "adjudication": result,
        "candidate_metadata": r2c.metadata_for_record(candidate_meta),
        "reference_metadata": r2c.metadata_for_record(reference_meta),
        "non_claims": [
            "later layers", "tokenizer/logits/generation", "C-path language-model quality",
            "RAM", "rate", "SPEED_LEDGER",
        ],
        "provenance": provenance,
    }
    r2c.base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {
        "APPARATUS_READY_NO_DONOR_EXECUTION",
        "PASS_ENGINE_REFERENCE_GENERIC_PROPAGATION",
        "FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
