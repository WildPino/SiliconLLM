#!/usr/bin/env python3
"""Recover the frozen layer-2 verdict from immutable producer outputs only."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_engine_layer2_depth_extension as layer2

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/strat01_gigachat_engine_layer2_depth_extension_20260924"
SOURCE_ADJUDICATION = SOURCE / "adjudication.json"
OUTPUT = HERE / "results/strat01_gigachat_engine_layer2_depth_extension_offline_recovery1_20260924"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_OFFLINE_RECOVERY_PROTOCOL_20260924.md"
TESTS = HERE / "test_recover_strat01_engine_layer2_depth_extension.py"
SOURCE_ADJUDICATION_BYTES = 168_044
SOURCE_ADJUDICATION_SHA = "65cf9f30bdf4c2028f5c13f91cf4dfc44d28b3c04d07bc254c395e0c74e0693c"
SOURCE_EXECUTION_HEAD = "9360378657a74f764b0374424f3cb325c61fd764"
SOURCE_RUNNER_SHA = "b27d9929148c07d67289c55050945ce0bfbaf1e07c117898d1c235d6a6c8007d"
RAW_VOID_STATUS = "VOID_ENGINE_LAYER2_DEPTH_EXTENSION_CAUSAL_CONTROL"


class RecoveryError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return layer2.sha(path)


def read_json(path: Path, label: str) -> Any:
    return layer2.read_json(path, label)


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "recovery": Path(__file__).resolve(),
        "tests": TESTS,
        "protocol": PROTOCOL,
        "qualified_runner": Path(layer2.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RecoveryError("missing recovery source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> str:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise RecoveryError("offline-recovery sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise RecoveryError(f"untracked offline-recovery source: {path}")
    return layer2.r2c.base.git_value(["git", "rev-parse", "HEAD"])


def validate_raw_record() -> dict[str, Any]:
    if not SOURCE_ADJUDICATION.is_file():
        raise RecoveryError("raw adjudication is missing")
    if SOURCE_ADJUDICATION.stat().st_size != SOURCE_ADJUDICATION_BYTES or sha(SOURCE_ADJUDICATION) != SOURCE_ADJUDICATION_SHA:
        raise RecoveryError("raw adjudication identity mismatch")
    raw = read_json(SOURCE_ADJUDICATION, "raw layer-2 adjudication")
    if (raw.get("status") != "FAIL_ENGINE_LAYER2_DEPTH_EXTENSION" or raw.get("errors") != [] or
            raw.get("production_invocations") != 1 or raw.get("reference_invocations") != 1 or
            raw.get("donor_graph_executions") != 2 or raw.get("reference_graph_executions") != 2):
        raise RecoveryError("raw execution accounting/state mismatch")
    provenance = raw.get("provenance", {})
    if provenance.get("git_head") != SOURCE_EXECUTION_HEAD:
        raise RecoveryError("raw execution HEAD mismatch")
    recorded_sources = provenance.get("source_hashes", {})
    if recorded_sources.get("runner", {}).get("sha256") != SOURCE_RUNNER_SHA:
        raise RecoveryError("raw qualified-runner hash mismatch")
    commands = provenance.get("commands", {})
    for name, marker in (("reference", layer2.REFERENCE_GRAPH_MARKER), ("production", layer2.GRAPH_MARKER)):
        command = commands.get(name)
        if not isinstance(command, dict) or command.get("returncode") != 0:
            raise RecoveryError(f"raw {name} command is absent or failed")
        if layer2.completed_graph_count(command, marker) != 2:
            raise RecoveryError(f"raw {name} graph-marker accounting mismatch")
    raw_adjudication = raw.get("adjudication", {})
    controls = raw_adjudication.get("negative_controls", {})
    passing_mutations = sorted(name for name, result in controls.items() if result.get("pass") is True)
    if raw_adjudication.get("negative_controls_pass") is not False or passing_mutations != ["omit_shared_expert"]:
        raise RecoveryError("raw causal-control failure is not the frozen single-control trigger")
    if "causal_negative_controls" not in raw_adjudication.get("failures", []):
        raise RecoveryError("raw adjudication omitted its causal-control failure")
    return raw


def branch_local_omit_shared(reference: dict[str, np.ndarray]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in layer2.r2c.base.ARMS:
        shared = reference[f"{arm}/ffn_shexp-2"]
        arms[arm] = layer2.r2c.base.judged(
            np.zeros_like(shared), shared, layer2.r2c.base.GENERAL_LIMITS
        )
    return {
        "definition": "zero versus immutable ffn_shexp-2 branch checkpoint",
        "limits": {
            "nrmse": layer2.r2c.base.GENERAL_LIMITS[0],
            "normalized_max": layer2.r2c.base.GENERAL_LIMITS[1],
        },
        "arms": arms,
        "pass": any(result["pass"] for result in arms.values()),
        "rejects_in_both_schedules": all(not result["pass"] for result in arms.values()),
    }


def classify(validity_pass: bool, numerical_failures: list[str]) -> str:
    if not validity_pass:
        return "VOID_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERY"
    if numerical_failures:
        return "FAIL_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED"
    return "PASS_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED"


def key_file_inventory() -> dict[str, dict[str, Any]]:
    paths = [
        SOURCE_ADJUDICATION,
        SOURCE / "pinned_reference/manifest.json",
        *(SOURCE / "pinned_reference" / arm / "manifest.json" for arm in layer2.r2c.base.ARMS),
        SOURCE / "c_engine/strat01_rung2d.json",
        SOURCE / "c_engine/strat01_f16_vector_counts.json",
        *(SOURCE / "c_engine" / f"{arm}_manifest.json" for arm in layer2.r2c.base.ARMS),
    ]
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise RecoveryError(f"missing immutable key file: {path}")
        result[path.relative_to(SOURCE).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    return result


def recover() -> dict[str, Any]:
    raw = validate_raw_record()
    qualified_sources = layer2.source_inventory()
    if qualified_sources["runner"]["sha256"] != SOURCE_RUNNER_SHA:
        raise RecoveryError("current qualified runner no longer matches the producer run")
    controls = layer2.source_controls()
    if not all(controls.values()):
        raise RecoveryError("qualified source controls no longer pass")

    model = layer2.r2c.MODEL.resolve(strict=True)
    reference, reference_meta = layer2.validate_reference(SOURCE / "pinned_reference", model)
    candidate, c_meta = layer2.validate_c(SOURCE / "c_engine", qualified_sources, model)
    counts = layer2.validate_counts(SOURCE / "c_engine")
    previous_reference, previous_meta = layer2.r2c.validate_reference(layer2.PREVIOUS_REFERENCE, model)
    reproduced = layer2.adjudicate(
        candidate, reference, c_meta, reference_meta,
        previous_reference, previous_meta, controls, counts,
    )
    if reproduced != raw["adjudication"]:
        raise RecoveryError("raw adjudication is not exactly reproducible from immutable payloads")

    original_negative = copy.deepcopy(reproduced["negative_controls"])
    repaired_negative = copy.deepcopy(original_negative)
    repaired_negative["omit_shared_expert"] = branch_local_omit_shared(reference)
    repaired_negative_pass = all(not result["pass"] for result in repaired_negative.values())
    if not repaired_negative_pass:
        raise RecoveryError("repaired causal-control set does not reject every mutation")

    numerical_failures = [
        failure for failure in reproduced["failures"]
        if failure.startswith(("checkpoint/", "continuity/", "cache/"))
    ]
    recovered_status = classify(True, numerical_failures)
    recovered = copy.deepcopy(reproduced)
    recovered.update({
        "status": recovered_status,
        "failures": numerical_failures,
        "negative_controls": repaired_negative,
        "negative_controls_pass": True,
        "original_aggregate_omit_shared_diagnostic": original_negative["omit_shared_expert"],
        "raw_noncanonical_status": raw["status"],
        "raw_protocol_classification": RAW_VOID_STATUS,
    })
    first_failure = next(
        (item for item in recovered["checkpoint_results"] if not item["pass"]), None
    )
    return {
        "status": recovered_status,
        "raw": raw,
        "recovered": recovered,
        "first_failed_checkpoint": first_failure,
        "qualified_source_hashes": qualified_sources,
        "key_files": key_file_inventory(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    status = "VOID_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERY"
    errors: list[str] = []
    payload: dict[str, Any] = {}
    recovery_sources: dict[str, Any] = {}
    execution_head = ""
    try:
        recovery_sources = source_inventory()
        execution_head = clean_sources_at_head(recovery_sources)
        payload = recover()
        status = payload["status"]
    except (RecoveryError, layer2.DepthExtensionError, layer2.r2c.RunnerError, layer2.r2c.base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")

    raw = payload.get("raw", {})
    recovered = payload.get("recovered", {})
    record = {
        "schema": "strat01_engine_layer2_depth_extension_offline_recovery_v1",
        "status": status,
        "errors": errors,
        "recovery_invocations": 1,
        "new_reference_invocations": 0,
        "new_production_invocations": 0,
        "new_reference_graph_executions": 0,
        "new_donor_graph_executions": 0,
        "inherited_execution": {
            "raw_adjudication_path": str(SOURCE_ADJUDICATION),
            "raw_adjudication_bytes": SOURCE_ADJUDICATION_BYTES,
            "raw_adjudication_sha256": SOURCE_ADJUDICATION_SHA,
            "raw_emitted_status": raw.get("status"),
            "raw_protocol_classification": RAW_VOID_STATUS,
            "reference_invocations": raw.get("reference_invocations"),
            "production_invocations": raw.get("production_invocations"),
            "reference_graph_executions": raw.get("reference_graph_executions"),
            "donor_graph_executions": raw.get("donor_graph_executions"),
        },
        "adjudication": recovered,
        "first_failed_checkpoint": payload.get("first_failed_checkpoint"),
        "provenance": {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": execution_head,
            "recovery_source_hashes": recovery_sources,
            "qualified_source_hashes": payload.get("qualified_source_hashes", {}),
            "immutable_key_files": payload.get("key_files", {}),
        },
        "non_claims": ["layers 3-25", "tokenizer/logits/generation", "quality", "RAM", "rate"],
    }
    layer2.r2c.base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {
        "PASS_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED",
        "FAIL_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
