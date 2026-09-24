#!/usr/bin/env python3
"""Qualify and run the frozen post-F16 SwiGLU production integration."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
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
from benchmarks.donor_adaptation.engine import run_strat01_post_f16_block0_swiglu_sse2_semantics as sse2
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as q4base

HERE = Path(__file__).resolve().parent
ENGINE = r2c.ENGINE
MODEL = r2c.MODEL
RUNG2A_HEADER = r2c.RUNG2A_HEADER
RUNG2B_HEADER = r2c.RUNG2B_HEADER
RUNG2C_HEADER = r2c.RUNG2C_HEADER
SHARED_HEADER = ROOT / "benchmarks/phase60/strat01_swiglu_sse2.h"
DIAGNOSTIC_HEADER = sse2.HEADER
F16_DOT_HEADER = ROOT / "benchmarks/phase60/strat01_f16_vector_dot.h"
Q4_HEADER = ROOT / "benchmarks/phase60/strat01_q4k_q8k.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_post_f16_swiglu_production_integration.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_post_f16_swiglu_production_integration_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_post_f16_swiglu_production_integration_apparatus_repair1_20260924"

SSE2_ADJUDICATION = HERE / "results/strat01_gigachat_engine_post_f16_block0_swiglu_sse2_semantics_20260924/adjudication.json"
SSE2_ADJUDICATION_SHA = "46f127215e4bfc00b80ba4eb151d20c294482d2cb5f3c976616e953d21852c0e"
F16_ADJUDICATION = HERE / "results/strat01_gigachat_engine_f16_vector_propagation_20260923/adjudication.json"
F16_ADJUDICATION_SHA = "512e3ea7754c5e8fa067dab5692a8a192879be26895547cbedbb6023a896edcf"
REFERENCE_ROOT = HERE / "results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference"
REFERENCE_MANIFESTS = {
    "root": (REFERENCE_ROOT / "manifest.json", "9a00ea08a47482323055a7fa8fcfa6c0e079ea2ed4957665bd491c1abfec21eb", 855),
    "prefill8": (REFERENCE_ROOT / "prefill8/manifest.json", "d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451", 22_382),
    "cached7p1": (REFERENCE_ROOT / "cached7p1/manifest.json", "ad214e8d38ee51a3b0f0629bdadf7cc692f6806499e1347c90d46aba937a39b9", 31_305),
}
SCALAR_START_SHA = "7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4"
EXPECTED_COUNTS = {"mode": "pinned-generic-f64", "qk_invocations": 4_608, "value_invocations": 524_288}
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + path.stem for path in sorted(HERE.glob("test_strat01_*.py")))
SELFTESTS = (*q4base.SELFTESTS, "--strat01-f16-vector-parity-selftest", "--strat01-post-f16-layer1-start-cross-input-selftest", "--strat01-post-f16-block0-terminal-component-cross-input-selftest", "--strat01-post-f16-block0-ffn-operator-cross-input-selftest", "--strat01-post-f16-block0-swiglu-sse2-semantics-selftest")


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
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "rung2c_runner": Path(r2c.__file__).resolve(),
        "sse2_runner": Path(sse2.__file__).resolve(),
        "q4_parity_runner": Path(q4base.__file__).resolve(),
        "base_runner": Path(r2c.base.__file__).resolve(),
        "engine": ENGINE, "shared_swiglu": SHARED_HEADER,
        "diagnostic_swiglu": DIAGNOSTIC_HEADER, "rung2a_header": RUNG2A_HEADER,
        "rung2b_header": RUNG2B_HEADER, "rung2c_header": RUNG2C_HEADER,
        "f16_dot": F16_DOT_HEADER, "q4_kernel": Q4_HEADER,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise IntegrationError("missing production-integration source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def validate_bindings() -> dict[str, Any]:
    if not SSE2_ADJUDICATION.is_file() or sha(SSE2_ADJUDICATION) != SSE2_ADJUDICATION_SHA:
        raise IntegrationError("SSE2 adjudication binding mismatch")
    sse2_record = read_json(SSE2_ADJUDICATION, "SSE2 adjudication")
    if (sse2_record.get("status") != "POST_F16_BLOCK0_SWIGLU_SSE2_EXACT_REPAIR" or
            sse2_record.get("errors") != [] or sse2_record.get("diagnostic_invocations") != 1 or
            sse2_record.get("donor_graph_executions") != 0 or sse2_record.get("reference_graph_executions") != 0):
        raise IntegrationError("SSE2 adjudication state mismatch")
    if not F16_ADJUDICATION.is_file() or sha(F16_ADJUDICATION) != F16_ADJUDICATION_SHA:
        raise IntegrationError("post-F16 adjudication binding mismatch")
    f16_record = read_json(F16_ADJUDICATION, "post-F16 adjudication")
    if f16_record.get("status") != "FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION" or f16_record.get("errors") != []:
        raise IntegrationError("post-F16 adjudication state mismatch")
    manifests: dict[str, Any] = {}
    for name, (path, digest, size) in REFERENCE_MANIFESTS.items():
        if not path.is_file() or path.stat().st_size != size or sha(path) != digest:
            raise IntegrationError(f"reference manifest binding mismatch: {name}")
        manifests[name] = {"path": str(path), "bytes": size, "sha256": digest}
    return {"sse2": {"path": str(SSE2_ADJUDICATION), "sha256": SSE2_ADJUDICATION_SHA},
            "post_f16": {"path": str(F16_ADJUDICATION), "sha256": F16_ADJUDICATION_SHA},
            "reference_manifests": manifests}


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8")
    shared = SHARED_HEADER.read_text(encoding="utf-8")
    diagnostic = DIAGNOSTIC_HEADER.read_text(encoding="utf-8")
    r2b = RUNG2B_HEADER.read_text(encoding="utf-8")
    r2c_text = RUNG2C_HEADER.read_text(encoding="utf-8")
    q4 = Q4_HEADER.read_text(encoding="utf-8")
    f16 = F16_DOT_HEADER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    return {
        "shared_header_precedes_rung2b": engine.index('#include "strat01_swiglu_sse2.h"') < engine.index('#include "strat01_gguf_rung2b.h"'),
        "single_shared_definition": shared.count("static __m128 strat01_sse2_expf_no_fma(") == 1 and "static __m128 strat01_sse2_expf_no_fma(" not in diagnostic,
        "no_fma_operation_order": "_mm_add_ps" in shared and "_mm_mul_ps" in shared and "_mm_fmadd" not in shared and "_mm_fnmadd" not in shared,
        "four_lane_tail_refusal": "count & 3U" in shared and "i += 4U" in shared,
        "production_delegates_shared": "strat01_sse2_swiglu_compute(out->gate,out->up,out->swiglu,8U*STRAT01_R2B_FFN,error)" in r2b,
        "diagnostic_delegates_shared": diagnostic.count("strat01_sse2_swiglu_compute(") >= 3,
        "layer1_swiglu_unchanged": (
            "sw[i]=(g[i]/(1.0f+expf(-g[i])))*u[i]" in r2c_text and
            "a->swiglu[i]=(a->gate[i]/(1.0f+expf(-a->gate[i])))*a->up[i]" in r2c_text and
            "strat01_sse2_swiglu_compute" not in r2c_text),
        "reference_generic_q4_default": "return strat01_q4k_q8k_dot_reference_generic(q4_blocks, q8_blocks, count);" in q4,
        "pinned_f64_f16_default": "double out=0.0" in f16 and "strat01_f16vec_dot" in f16,
        "production_configs_declare_sse2": "swiglu-sse2-nofma4" in r2b and "block0=dense-swiglu-sse2-nofma4" in r2c_text,
        "protocol_frozen_before_implementation": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol,
    }


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise IntegrationError("production-integration sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise IntegrationError(f"untracked production-integration source: {path}")


def validate_counts(root: Path) -> dict[str, Any]:
    counts = read_json(root / "strat01_f16_vector_counts.json", "production helper counts")
    if counts != EXPECTED_COUNTS:
        raise IntegrationError(f"production helper count mismatch: {counts!r}")
    return counts


def classify(failures: list[str]) -> str:
    return ("PASS_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION" if not failures
            else "FAIL_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION")


def adjudicate(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray],
               c_meta: dict[str, Any], r_meta: dict[str, Any],
               controls: dict[str, bool], counts: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    continuity: list[dict[str, Any]] = []
    cache_results: list[dict[str, Any]] = []
    for arm in r2c.base.ARMS:
        for name in r2c.SHAPES:
            if name in r2c.I32_NAMES:
                passed = bool(np.array_equal(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"]))
                result = {"arm": arm, "checkpoint": name, "exact": passed, "pass": passed}
            else:
                limits = r2c.base.TERMINAL_LIMITS if name in r2c.TERMINALS else r2c.base.GENERAL_LIMITS
                result = r2c.base.judged(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"], limits)
                result.update({"arm": arm, "checkpoint": name})
            checkpoints.append(result)
            if not result["pass"]:
                failures.append(f"checkpoint/{arm}/{name}")
    for implementation, values in (("c_engine", candidate), ("pinned_reference", reference)):
        for name, shape in r2c.SHAPES.items():
            left = r2c.base.token7(values[f"prefill8/{name}"], shape)
            right = r2c.base.token7(values[f"cached7p1/{name}"], shape)
            if name in r2c.I32_NAMES:
                passed = bool(np.array_equal(left, right))
                result = {"implementation": implementation, "checkpoint": name, "exact": passed, "pass": passed}
            else:
                result = r2c.base.judged(left, right, r2c.base.CONTINUITY_LIMITS)
                result.update({"implementation": implementation, "checkpoint": name})
            continuity.append(result)
            if not result["pass"]:
                failures.append(f"continuity/{implementation}/{name}")
    for arm in r2c.base.ARMS:
        for key, values in c_meta["caches"][arm].items():
            result = r2c.base.judged(values, r_meta["caches"][arm][key], r2c.base.GENERAL_LIMITS)
            result.update({"arm": arm, "cache": key})
            cache_results.append(result)
            if not result["pass"]:
                failures.append(f"cache/{arm}/{key}")
    start_hashes = {
        arm: next(item["sha256"] for item in c_meta["manifests"][arm]["tensors"] if item.get("name") == "l_out-0")
        for arm in r2c.base.ARMS
    }
    if len(set(start_hashes.values())) != 1:
        failures.append("production_start_schedule_identity")
    if any(digest == SCALAR_START_SHA for digest in start_hashes.values()):
        failures.append("scalar_predecessor_not_displaced")
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
        "start_hashes": start_hashes, "scalar_predecessor_sha256": SCALAR_START_SHA,
        "checkpoint_results": checkpoints, "continuity_results": continuity,
        "cache_results": cache_results, "negative_controls": negative,
        "negative_controls_pass": negative_pass, "source_controls": controls,
        "helper_counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION"
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
            raise IntegrationError("production-integration source controls failed")
        if not compiler:
            raise IntegrationError("clang unavailable")
        binary = output / "engine_post_f16_swiglu_production_integration.exe"
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
            clean_sources_at_head(sources)
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
            candidate, c_meta = r2c.validate_c(candidate_root, sources, model)
            counts = validate_counts(candidate_root)
            result = adjudicate(candidate, reference, c_meta, r_meta, controls, counts)
            report = c_meta["report"]
            status = result["status"]
    except (IntegrationError, r2c.RunnerError, r2c.base.RunnerError, sse2.DiagnosticError, q4base.ParityError) as exc:
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
        "schema": "strat01_post_f16_swiglu_production_integration_v1",
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
