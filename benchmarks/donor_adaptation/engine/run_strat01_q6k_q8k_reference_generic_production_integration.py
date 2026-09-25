#!/usr/bin/env python3
"""Qualify and run the frozen Q6 reference-generic production integration."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2c as r2c
from benchmarks.donor_adaptation.engine import run_strat01_post_f16_swiglu_production_integration as prior
from benchmarks.donor_adaptation.engine import run_strat01_q6k_q8k_avx2_parity as q6base

HERE = Path(__file__).resolve().parent
ENGINE = r2c.ENGINE
MODEL = r2c.MODEL
RUNG2B_HEADER = r2c.RUNG2B_HEADER
RUNG2C_HEADER = r2c.RUNG2C_HEADER
Q6_HEADER = ROOT / "benchmarks/phase60/strat01_q6k_q8k_avx2.h"
Q4_HEADER = ROOT / "benchmarks/phase60/strat01_q4k_q8k.h"
F16_HEADER = ROOT / "benchmarks/phase60/strat01_f16_vector_dot.h"
SWIGLU_HEADER = ROOT / "benchmarks/phase60/strat01_swiglu_sse2.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_Q6K_Q8K_REFERENCE_GENERIC_PRODUCTION_INTEGRATION_PROTOCOL_20260924.md"
VOID_RESULT = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_Q6K_Q8K_REFERENCE_GENERIC_PRODUCTION_INTEGRATION_VOID1_20260925.md"
TESTS = HERE / "test_strat01_q6k_q8k_reference_generic_production_integration.py"
Q6_RESULT = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_RESULT_20260924.md"
Q6_ADJUDICATION = HERE / "results/strat01_gigachat_engine_q6k_q8k_reference_generic_parity_20260924/adjudication.json"
Q6_ADJUDICATION_SHA = "cb811b19c58d3a31688967c18dc9119451ed01faee0e56098166efd4f46ff066"
REFERENCE_ROOT = prior.REFERENCE_ROOT
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_q6k_q8k_reference_generic_production_integration_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_q6k_q8k_reference_generic_production_integration_apparatus_repair1_20260925"
DEFAULT_RECOVERY = HERE / "results/strat01_gigachat_engine_q6k_q8k_reference_generic_production_integration_offline_recovery1_20260925"
OLD_L_OUT0_SHA = "a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11"
EXACT_L_OUT0_SHA = "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"
EXPECTED_COUNTS = prior.EXPECTED_COUNTS
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + path.stem for path in sorted(HERE.glob("test_strat01_*.py")))
SELFTESTS = (*prior.SELFTESTS, "--strat01-block0-q6k-q8k-reference-generic-parity-selftest")
Q6_CONFIG_MARKER = "q6kq8k=reference-generic-noavx-noavx2-nofma-noinline;"
EXPECTED_C_CONFIG = r2c.EXPECTED_C_CONFIG.replace(
    "kb=q5_0xq8_0;", "kb=q5_0xq8_0;" + Q6_CONFIG_MARKER,
)


class IntegrationError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return r2c.base.sha256_file(path)


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    sources = prior.source_inventory()
    sources["prior_runner"] = sources.pop("runner")
    sources["prior_tests"] = sources.pop("tests")
    sources["prior_protocol"] = sources.pop("protocol")
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "void_result": VOID_RESULT,
        "q6_result": Q6_RESULT, "q6_kernel": Q6_HEADER,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise IntegrationError("missing Q6 production-integration source(s): " + ", ".join(missing))
    sources.update({name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()})
    return sources


def validate_bindings() -> dict[str, Any]:
    bindings = prior.validate_bindings()
    if not Q6_ADJUDICATION.is_file() or sha(Q6_ADJUDICATION) != Q6_ADJUDICATION_SHA:
        raise IntegrationError("Q6 reference-generic adjudication binding mismatch")
    record = read_json(Q6_ADJUDICATION, "Q6 reference-generic adjudication")
    counters = record.get("counters", {})
    if (record.get("status") != "BLOCK0_Q6_REFERENCE_GENERIC_EXACT_REPAIR" or
            record.get("errors") != [] or counters.get("diagnostic_executions") != 1 or
            counters.get("donor_graph_executions") != 0 or
            counters.get("reference_graph_executions") != 0 or
            not all(record.get("controls", {}).values())):
        raise IntegrationError("Q6 reference-generic adjudication state mismatch")
    bindings["q6_reference_generic"] = {
        "path": str(Q6_ADJUDICATION), "sha256": Q6_ADJUDICATION_SHA,
        "status": record["status"],
    }
    return bindings


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8")
    q6 = Q6_HEADER.read_text(encoding="utf-8")
    r2b = RUNG2B_HEADER.read_text(encoding="utf-8")
    r2c_text = RUNG2C_HEADER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    marker = "q6kq8k=reference-generic-noavx-noavx2-nofma-noinline"
    wrapper = re.search(
        r"static float strat01_q6k_q8k_dot\([^{}]+\)\s*\{\s*"
        r"return strat01_q6k_q8k_dot_reference_generic\(q6_blocks,q8_blocks,count\);\s*\}",
        r2b,
    )
    return {
        "prior_production_controls_still_hold": all(prior.source_controls().values()),
        "q6_header_precedes_rung2b": engine.index('#include "strat01_q6k_q8k_avx2.h"') < engine.index('#include "strat01_gguf_rung2b.h"'),
        "single_target_isolated_reference_helper": q6.count("static float strat01_q6k_q8k_dot_reference_generic(") == 1 and 'target("no-avx,no-avx2,no-fma")' in q6 and "noinline" in q6,
        "production_wrapper_delegates_exactly": wrapper is not None,
        "old_production_arithmetic_removed": "float sums[8]" not in r2b,
        "rung2c_uses_shared_wrapper": "float v=strat01_q6k_q8k_dot(raw,q8+(size_t)b*blocks,in)" in r2c_text,
        "both_configs_declare_reference_generic": marker in r2b and marker in r2c_text,
        "residual_order_unchanged": "out->l_out[i]=out->out[i]+input->ffn_inp[i]" in r2b,
        "q4_default_unchanged": "return strat01_q4k_q8k_dot_reference_generic(q4_blocks, q8_blocks, count);" in Q4_HEADER.read_text(encoding="utf-8"),
        "f16_default_unchanged": "double out=0.0" in F16_HEADER.read_text(encoding="utf-8"),
        "swiglu_default_unchanged": "strat01_sse2_swiglu_compute" in SWIGLU_HEADER.read_text(encoding="utf-8"),
        "protocol_and_observability_addendum_frozen": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol and "Pre-implementation observability addendum" in protocol,
        "offline_recovery_addendum_frozen": "Post-execution offline-recovery addendum" in protocol,
    }


def validate_candidate(c_root: Path, sources: dict[str, dict[str, str]], model: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Use the historical validator with this cell's exact declared CONFIG."""
    report = read_json(c_root / "strat01_rung2c.json", "C report")
    if report.get("CONFIG") != EXPECTED_C_CONFIG:
        raise IntegrationError("Q6 production CONFIG mismatch")
    historical = r2c.EXPECTED_C_CONFIG
    r2c.EXPECTED_C_CONFIG = EXPECTED_C_CONFIG
    try:
        return r2c.validate_c(c_root, sources, model)
    finally:
        r2c.EXPECTED_C_CONFIG = historical


def metadata_for_record(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "caches"}


def classify(failures: list[str]) -> str:
    return ("PASS_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION" if not failures
            else "FAIL_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION")


def exact_result(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    exact = bool(np.array_equal(left, right))
    return {"count": int(left.size), "changed": int(np.count_nonzero(left != right)), "exact": exact, "pass": exact}


def adjudicate(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray],
               c_meta: dict[str, Any], r_meta: dict[str, Any],
               controls: dict[str, bool], counts: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    continuity: list[dict[str, Any]] = []
    caches: list[dict[str, Any]] = []
    for arm in r2c.base.ARMS:
        for name in r2c.SHAPES:
            result = exact_result(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"])
            result.update({"arm": arm, "checkpoint": name})
            checkpoints.append(result)
            if not result["pass"]:
                failures.append(f"checkpoint/{arm}/{name}")
    for implementation, values in (("c_engine", candidate), ("pinned_reference", reference)):
        for name, shape in r2c.SHAPES.items():
            result = exact_result(
                r2c.base.token7(values[f"prefill8/{name}"], shape),
                r2c.base.token7(values[f"cached7p1/{name}"], shape),
            )
            result.update({"implementation": implementation, "checkpoint": name})
            continuity.append(result)
            if not result["pass"]:
                failures.append(f"continuity/{implementation}/{name}")
    for arm in r2c.base.ARMS:
        for name, values in c_meta["caches"][arm].items():
            result = exact_result(values, r_meta["caches"][arm][name])
            result.update({"arm": arm, "cache": name})
            caches.append(result)
            if not result["pass"]:
                failures.append(f"cache/{arm}/{name}")
    l_out0_hashes = {
        arm: next(item["sha256"] for item in c_meta["manifests"][arm]["tensors"] if item.get("name") == "l_out-0")
        for arm in r2c.base.ARMS
    }
    if any(value != EXACT_L_OUT0_SHA for value in l_out0_hashes.values()):
        failures.append("block0_l_out0_exact_hash")
    if any(value == OLD_L_OUT0_SHA for value in l_out0_hashes.values()):
        failures.append("old_block0_terminal_not_displaced")
    negative = r2c.negative_controls(reference)
    negative_pass = all(not item["pass"] for item in negative.values())
    if not negative_pass:
        failures.append("causal_negative_controls")
    if not all(controls.values()):
        failures.append("source_controls")
    if counts != EXPECTED_COUNTS:
        failures.append("helper_counts")
    return {
        "status": classify(failures), "failures": failures,
        "l_out0_hashes": l_out0_hashes, "old_l_out0_sha256": OLD_L_OUT0_SHA,
        "exact_l_out0_sha256": EXACT_L_OUT0_SHA,
        "checkpoint_results": checkpoints, "continuity_results": continuity,
        "cache_results": caches, "negative_controls": negative,
        "negative_controls_pass": negative_pass, "source_controls": controls,
        "helper_counts": counts,
    }


def recover_existing(source: Path, output: Path, model: Path, execution_head: str) -> int:
    source = source.resolve(strict=True)
    output = output.resolve()
    model = model.resolve(strict=True)
    if not source.is_dir() or (source / "adjudication.json").exists() or (source / "run_manifest.json").exists():
        raise SystemExit("recovery requires the preserved unfinalized Q6 production-integration directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory recovery output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION"
    errors: list[str] = []
    result: dict[str, Any] = {"status": "NOT_RUN"}
    c_meta: dict[str, Any] = {}
    r_meta: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    controls: dict[str, bool] = {}
    bindings: dict[str, Any] = {}
    producer_record: dict[str, Any] = {}
    key_files: dict[str, Any] = {}
    report: dict[str, Any] = {}
    try:
        resolved_head = r2c.base.git_value(["git", "rev-parse", f"{execution_head}^{{commit}}"])
        if resolved_head != execution_head:
            raise IntegrationError("recovery execution HEAD is not an exact commit")
        if not model.is_file() or model.stat().st_size != r2c.base.EXPECTED_BYTES or sha(model) != r2c.base.EXPECTED_SHA256:
            raise IntegrationError("recovery artifact identity mismatch")
        stdout_path = source / "production.stdout.log"
        stderr_path = source / "production.stderr.log"
        if not stdout_path.is_file() or not stderr_path.is_file():
            raise IntegrationError("recovery production logs are missing")
        command_record = {
            "stdout": stdout_path.read_text(encoding="utf-8"),
            "stderr": stderr_path.read_text(encoding="utf-8"),
        }
        completed = r2c.completed_graph_count(command_record)
        if completed != 2:
            raise IntegrationError(f"recovery donor graph count is {completed}, expected 2")
        producer_record = {
            "invocations": 1,
            "graph_executions": completed,
            "stdout": {"path": str(stdout_path), "bytes": stdout_path.stat().st_size, "sha256": sha(stdout_path)},
            "stderr": {"path": str(stderr_path), "bytes": stderr_path.stat().st_size, "sha256": sha(stderr_path)},
        }
        sources = source_inventory()
        prior.clean_sources_at_head(sources)
        bindings = validate_bindings()
        controls = source_controls()
        if not all(controls.values()):
            raise IntegrationError("offline-recovery source controls failed")
        reference, r_meta = r2c.validate_reference(REFERENCE_ROOT, model)
        candidate, c_meta = validate_candidate(source / "c_engine", sources, model)
        report = c_meta["report"]
        counts = prior.validate_counts(source / "c_engine")
        result = adjudicate(candidate, reference, c_meta, r_meta, controls, counts)
        status = result["status"]
        key_paths = [
            source / "c_engine" / "strat01_rung2c.json",
            source / "c_engine" / "strat01_f16_vector_counts.json",
            *(source / "c_engine" / f"{arm}_manifest.json" for arm in r2c.base.ARMS),
            REFERENCE_ROOT / "manifest.json",
            *(REFERENCE_ROOT / arm / "manifest.json" for arm in r2c.base.ARMS),
        ]
        for path in key_paths:
            if not path.is_file():
                raise IntegrationError(f"recovery key file is missing: {path}")
            key_files[str(path)] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    except (IntegrationError, prior.IntegrationError, r2c.RunnerError, r2c.base.RunnerError, q6base.ApparatusError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "mode": "offline-recovery",
        "source_run_directory": str(source),
        "execution_git_head": execution_head,
        "recovery_git_head": r2c.base.git_value(["git", "rev-parse", "HEAD"]),
        "source_hashes_at_recovery": sources,
        "bindings": bindings,
        "producer_record": producer_record,
        "key_files": key_files,
        "artifact": {"path": str(model), "bytes": model.stat().st_size, "sha256": r2c.base.EXPECTED_SHA256},
        "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd()},
    }
    record = {
        "schema": "strat01_q6k_q8k_reference_generic_production_integration_offline_recovery_v1",
        "status": status,
        "errors": errors,
        "mode": "offline-recovery",
        "production_invocations": 1,
        "donor_graph_executions": producer_record.get("graph_executions", 0),
        "new_production_invocations": 0,
        "new_donor_graph_executions": 0,
        "reference_graph_executions": 0,
        "source_controls": controls,
        "adjudication": result,
        "c_report": report,
        "c_metadata": metadata_for_record(c_meta),
        "reference_metadata": metadata_for_record(r_meta),
        "non_claims": ["producer rerun", "later layers", "tokenizer/logits/generation", "quality", "RAM", "rate"],
        "provenance": provenance,
    }
    manifest = {
        "schema": "strat01_q6k_q8k_reference_generic_production_integration_offline_recovery_manifest_v1",
        "status": status,
        "errors": errors,
        "source_production_invocations": 1,
        "source_donor_graph_executions": producer_record.get("graph_executions", 0),
        "new_production_invocations": 0,
        "new_donor_graph_executions": 0,
        "new_reference_graph_executions": 0,
        "provenance": provenance,
    }
    json.dumps(record)
    json.dumps(manifest)
    r2c.base.write_json(output / "adjudication.json", record)
    r2c.base.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors, "recovered_without_producer_execution": True}, indent=2))
    return 0 if status.startswith("PASS_ENGINE_") or status.startswith("FAIL_ENGINE_") else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument("--execution-head")
    args = parser.parse_args()
    model = args.model.resolve()
    if args.recover_from:
        if args.apparatus_only or not args.execution_head:
            raise SystemExit("--recover-from requires --execution-head and forbids --apparatus-only")
        return recover_existing(args.recover_from, args.output_dir or DEFAULT_RECOVERY, model, args.execution_head)
    if args.execution_head:
        raise SystemExit("--execution-head is valid only with --recover-from")
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    result: dict[str, Any] = {"status": "NOT_RUN"}
    report: dict[str, Any] = {}
    compiler = shutil.which("clang")
    binary: Path | None = None
    production_invocations = 0
    donor_graph_executions = 0
    try:
        sources = source_inventory()
        bindings = validate_bindings()
        controls = source_controls()
        if not all(controls.values()):
            raise IntegrationError("Q6 production-integration source controls failed")
        if not compiler:
            raise IntegrationError("clang unavailable")
        binary = output / "engine_q6_reference_generic_production_integration.exe"
        commands["compile"] = r2c.base.run_command([compiler, *r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile", timeout=600)
        r2c.base.require_ok(commands["compile"], "compile")
        commands["python_tests"] = r2c.base.run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], output=output, label="all_strat01_unittests", timeout=1800)
        r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
        for index, option in enumerate(SELFTESTS):
            label = f"selftest_{index:02d}"
            commands[label] = r2c.base.run_command([str(binary), option], output=output, label=label, timeout=300)
            r2c.base.require_ok(commands[label], option)
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            prior.clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size != r2c.base.EXPECTED_BYTES or sha(model) != r2c.base.EXPECTED_SHA256:
                raise IntegrationError("accepted artifact identity mismatch")
            reference, r_meta = r2c.validate_reference(REFERENCE_ROOT, model)
            candidate_root = output / "c_engine"
            candidate_root.mkdir()
            production_invocations = 1
            commands["production"] = r2c.base.run_command([str(binary), "--strat01-gguf-rung2c", str(model), "--out-dir", str(candidate_root)], output=output, label="production", timeout=21600)
            r2c.base.require_ok(commands["production"], "standard Rung-2C production")
            donor_graph_executions = r2c.completed_graph_count(commands["production"])
            if donor_graph_executions != 2:
                raise IntegrationError(f"production graph count is {donor_graph_executions}, expected 2")
            sources = source_inventory()
            candidate, c_meta = validate_candidate(candidate_root, sources, model)
            counts = prior.validate_counts(candidate_root)
            result = adjudicate(candidate, reference, c_meta, r_meta, controls, counts)
            report = c_meta["report"]
            status = result["status"]
    except (IntegrationError, prior.IntegrationError, r2c.RunnerError, r2c.base.RunnerError, q6base.ApparatusError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "git_head_observed" if args.apparatus_only else "git_head": r2c.base.git_value(["git", "rev-parse", "HEAD"]),
        "source_hashes": sources, "bindings": bindings,
        "artifact": {"path": str(model), "expected_bytes": r2c.base.EXPECTED_BYTES, "expected_sha256": r2c.base.EXPECTED_SHA256},
        "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler},
        "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None},
        "commands": commands,
    }
    record = {
        "schema": "strat01_q6k_q8k_reference_generic_production_integration_v1",
        "status": status, "errors": errors,
        "production_invocations": production_invocations,
        "donor_graph_executions": donor_graph_executions,
        "reference_graph_executions": 0,
        "source_controls": source_controls() if sources else {},
        "adjudication": result, "c_report": report,
        "non_claims": ["later layers", "tokenizer/logits/generation", "quality", "RAM", "rate"],
        "provenance": provenance,
    }
    r2c.base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status.startswith("PASS_ENGINE_") or status.startswith("FAIL_ENGINE_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
