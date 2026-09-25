#!/usr/bin/env python3
"""Qualify and run layer-1 router/SwiGLU primitive production integration."""
from __future__ import annotations

import argparse
import hashlib
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
from benchmarks.donor_adaptation.engine import run_strat01_q6k_q8k_reference_generic_production_integration as prior

HERE = Path(__file__).resolve().parent
ENGINE = r2c.ENGINE
MODEL = r2c.MODEL
RUNG2C_HEADER = r2c.RUNG2C_HEADER
F32_HEADER = ROOT / "benchmarks/phase60/strat01_f32_dot_reference_generic.h"
SWIGLU_HEADER = ROOT / "benchmarks/phase60/strat01_swiglu_sse2.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION_PROTOCOL_20260925.md"
PRIMITIVE_RESULT = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_RESULT_20260925.md"
PRIMITIVE_ADJUDICATION = HERE / "results/strat01_gigachat_engine_layer1_numerical_primitives_compile_parity_repair1_20260925/adjudication.json"
PRIMITIVE_ADJUDICATION_SHA = "ac04b6daeadb0ae6e71d19746585cf04fffeb74620217ed211e56484578b045a"
TESTS = HERE / "test_strat01_layer1_numerical_primitives_production_integration.py"
REFERENCE_ROOT = prior.REFERENCE_ROOT
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_numerical_primitives_production_integration_20260925"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_numerical_primitives_production_integration_apparatus_repair1_20260925"
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + path.stem for path in sorted(HERE.glob("test_strat01_*.py")))
SELFTESTS = prior.SELFTESTS
ROUTER_MARKER = "routerdot=f32prod-f64accum-reference-generic;"
SWIGLU_MARKER = "layer1swiglu=sse2-nofma4;"
EXPECTED_C_CONFIG = prior.EXPECTED_C_CONFIG.replace(
    prior.Q6_CONFIG_MARKER, prior.Q6_CONFIG_MARKER + ROUTER_MARKER + SWIGLU_MARKER,
)
EXPECTED_PRIMITIVE_HASHES = {
    "ffn_moe_logits-1": "8188ff0d608387dc8ce6295018c0f54a751f8a57c348d3a45bf0446770a0bbf5",
    "ffn_moe_swiglu-1": "5097dc8599d0635477ad83f63f7e2de21b659775abc345dc933ce1e173be29e8",
    "ffn_swiglu-1": "fa0d1b4fe5da3f561ac30b8b8cb95520f359232528ea66c1d60e9244642c6003",
}


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
    for old, new in (("runner", "q6_runner"), ("tests", "q6_tests"), ("protocol", "q6_protocol")):
        sources[new] = sources.pop(old)
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "primitive_result": PRIMITIVE_RESULT, "f32_helper": F32_HEADER,
        "swiglu_helper": SWIGLU_HEADER,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise IntegrationError("missing primitive-integration source(s): " + ", ".join(missing))
    sources.update({name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()})
    return sources


def validate_bindings() -> dict[str, Any]:
    bindings = prior.validate_bindings()
    if sha(PRIMITIVE_ADJUDICATION) != PRIMITIVE_ADJUDICATION_SHA:
        raise IntegrationError("local-primitives adjudication binding mismatch")
    record = read_json(PRIMITIVE_ADJUDICATION, "local-primitives adjudication")
    if (record.get("status") != "LAYER1_ROUTER_AND_SWIGLU_EXACT_PRIMITIVES" or
            record.get("errors") != [] or record.get("diagnostic_invocations") != 1 or
            record.get("producer_invocations") != 0 or
            record.get("donor_graph_executions") != 0 or
            record.get("reference_graph_executions") != 0 or
            not all(record.get("adjudication", {}).get("exact", {}).values()) or
            not all(record.get("adjudication", {}).get("controls", {}).values())):
        raise IntegrationError("local-primitives adjudication state mismatch")
    bindings["layer1_numerical_primitives"] = {
        "path": str(PRIMITIVE_ADJUDICATION), "sha256": PRIMITIVE_ADJUDICATION_SHA,
        "status": record["status"],
    }
    return bindings


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8")
    r2c_text = RUNG2C_HEADER.read_text(encoding="utf-8")
    f32 = F32_HEADER.read_text(encoding="utf-8")
    swiglu = SWIGLU_HEADER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    run_moe = r2c_text[r2c_text.index("static int strat01_r2c_run_moe"):r2c_text.index("static void strat01_r2c_set_dump")]
    f32_matmul = r2c_text[r2c_text.index("static int strat01_r2c_f32_matmul_batch"):r2c_text.index("static uint64_t strat01_r2c_quant_row_bytes")]
    q6_controls = prior.source_controls()
    swiglu_predecessor_controls = prior.prior.source_controls()
    return {
        "q6_production_controls_still_hold": all(value for name, value in q6_controls.items() if name != "prior_production_controls_still_hold"),
        "block0_swiglu_controls_still_hold": all(value for name, value in swiglu_predecessor_controls.items() if name != "layer1_swiglu_unchanged"),
        "f32_header_precedes_rung2c": engine.index('#include "strat01_f32_dot_reference_generic.h"') < engine.index('#include "strat01_gguf_rung2c.h"'),
        "single_isolated_f32_helper": f32.count("static float strat01_f32_dot_reference_generic(") == 1 and 'target("no-avx,no-avx2,no-fma")' in f32 and "noinline" in f32,
        "router_delegates_exactly": f32_matmul.count("strat01_f32_dot_reference_generic(w,xb,in)") == 1 and "sum+=w[i]*xb[i]" not in f32_matmul,
        "routed_swiglu_delegates_exactly": run_moe.count("strat01_sse2_swiglu_compute(g,u,sw,1280U,error)") == 1,
        "shared_swiglu_delegates_exactly": run_moe.count("strat01_sse2_swiglu_compute(a->gate,a->up,a->swiglu,8U*1280U,error)") == 1,
        "old_layer1_scalar_swiglu_removed": "sw[i]=(g[i]/(1.0f+expf(-g[i])))*u[i]" not in run_moe and "a->swiglu[i]=(a->gate[i]/(1.0f+expf(-a->gate[i])))*a->up[i]" not in run_moe,
        "qualified_swiglu_helper_unchanged": "strat01_sse2_expf_no_fma" in swiglu and "strat01_sse2_swiglu_compute" in swiglu,
        "config_declares_both_primitives": ROUTER_MARKER.rstrip(";") in r2c_text and SWIGLU_MARKER.rstrip(";") in r2c_text,
        "protocol_frozen_preimplementation": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol and "Exactly one non-VOID producer invocation is allowed" in protocol,
        "apparatus_repair_frozen": "Apparatus repair addendum" in protocol and "four historical assertions" in protocol,
        "local_result_is_bound": PRIMITIVE_ADJUDICATION_SHA in protocol,
    }


def validate_candidate(c_root: Path, sources: dict[str, dict[str, str]], model: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report = read_json(c_root / "strat01_rung2c.json", "C report")
    if report.get("CONFIG") != EXPECTED_C_CONFIG:
        raise IntegrationError("layer-1 primitive production CONFIG mismatch")
    historical = r2c.EXPECTED_C_CONFIG
    r2c.EXPECTED_C_CONFIG = EXPECTED_C_CONFIG
    try:
        return r2c.validate_c(c_root, sources, model)
    finally:
        r2c.EXPECTED_C_CONFIG = historical


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def adjudicate(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray],
               c_meta: dict[str, Any], r_meta: dict[str, Any], controls: dict[str, bool],
               counts: dict[str, Any]) -> dict[str, Any]:
    result = prior.adjudicate(candidate, reference, c_meta, r_meta, controls, counts)
    primitive_hashes: dict[str, dict[str, str]] = {}
    for arm in r2c.base.ARMS:
        primitive_hashes[arm] = {}
        for name, expected in EXPECTED_PRIMITIVE_HASHES.items():
            observed = array_sha(candidate[f"{arm}/{name}"])
            primitive_hashes[arm][name] = observed
            if observed != expected:
                result["failures"].append(f"primitive_hash/{arm}/{name}")
    result["failures"] = list(dict.fromkeys(result["failures"]))
    result["primitive_hashes"] = primitive_hashes
    result["expected_primitive_hashes"] = EXPECTED_PRIMITIVE_HASHES
    result["status"] = ("PASS_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION"
                        if not result["failures"] else
                        "FAIL_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION")
    return result


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
    status = "VOID_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION"
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
        if not all(controls.values()) or not compiler:
            raise IntegrationError("production-integration apparatus source/compiler gate failed")
        binary = output / "engine_layer1_numerical_primitives.exe"
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
            prior.prior.clean_sources_at_head(sources)
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
            counts = prior.prior.validate_counts(candidate_root)
            result = adjudicate(candidate, reference, c_meta, r_meta, controls, counts)
            report = c_meta["report"]
            status = result["status"]
    except (IntegrationError, prior.IntegrationError, r2c.RunnerError, r2c.base.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "git_head_observed" if args.apparatus_only else "git_head": r2c.base.git_value(["git", "rev-parse", "HEAD"]),
        "source_hashes": sources, "bindings": bindings,
        "artifact": {"path": str(model), "expected_bytes": r2c.base.EXPECTED_BYTES, "expected_sha256": r2c.base.EXPECTED_SHA256},
        "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang": compiler},
        "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None},
        "commands": commands,
    }
    record = {
        "schema": "strat01_layer1_numerical_primitives_production_integration_v1",
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
