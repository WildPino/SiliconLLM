#!/usr/bin/env python3
"""Qualify the frozen STRAT-01 layer-2 depth-extension apparatus."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import build_strat01_engine_rung2a_reference as reference_build
from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2c as r2c
from benchmarks.donor_adaptation.engine import run_strat01_post_f16_swiglu_production_integration as production
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as q4base

HERE = Path(__file__).resolve().parent
ENGINE = r2c.ENGINE
RUNG2A = r2c.RUNG2A_HEADER
RUNG2B = r2c.RUNG2B_HEADER
RUNG2C = r2c.RUNG2C_HEADER
RUNG2D = ROOT / "benchmarks/phase60/strat01_gguf_rung2d.h"
REFERENCE_SOURCE = HERE / "strat01_engine_rung2a_reference.cpp"
REFERENCE_BUILD = HERE / "build_strat01_engine_rung2a_reference.py"
REFERENCE_WRAPPER = HERE / "build_strat01_engine_rung2d_reference.py"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_PROTOCOL_20260924.md"
AUDIT = ROOT / "docs/research/donor_adaptation/audits/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_AUDIT_20260924.md"
TESTS = HERE / "test_strat01_engine_layer2_depth_extension.py"
PREDECESSOR = HERE / "results/strat01_gigachat_engine_post_f16_swiglu_production_integration_20260924/adjudication.json"
PREDECESSOR_SHA = "d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer2_depth_extension_apparatus_20260924"
EXPECTED_COUNTS = {"mode": "pinned-generic-f64", "qk_invocations": 6_912, "value_invocations": 786_432}
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + path.stem for path in sorted(HERE.glob("test_strat01_*.py")))
SELFTESTS = (*production.SELFTESTS, "--strat01-gguf-rung2d-selftest")


class DepthExtensionError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return r2c.base.sha256_file(path)


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthExtensionError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL, "audit": AUDIT,
        "engine": ENGINE, "rung2a": RUNG2A, "rung2b": RUNG2B, "rung2c": RUNG2C, "rung2d": RUNG2D,
        "reference_source": REFERENCE_SOURCE, "reference_build": REFERENCE_BUILD,
        "reference_wrapper": REFERENCE_WRAPPER, "rung2c_runner": Path(r2c.__file__).resolve(),
        "production_runner": Path(production.__file__).resolve(), "base_runner": Path(r2c.base.__file__).resolve(),
        "q4_parity_runner": Path(q4base.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DepthExtensionError("missing layer-2 source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def validate_predecessor() -> dict[str, str]:
    if not PREDECESSOR.is_file() or sha(PREDECESSOR) != PREDECESSOR_SHA:
        raise DepthExtensionError("production predecessor binding mismatch")
    record = read_json(PREDECESSOR, "production predecessor")
    if (record.get("status") != "PASS_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION" or
            record.get("errors") != [] or record.get("production_invocations") != 1 or
            record.get("donor_graph_executions") != 2 or record.get("reference_graph_executions") != 0):
        raise DepthExtensionError("production predecessor state mismatch")
    return {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8")
    r2c_text = RUNG2C.read_text(encoding="utf-8")
    r2d = RUNG2D.read_text(encoding="utf-8")
    ref = REFERENCE_SOURCE.read_text(encoding="utf-8")
    build = REFERENCE_BUILD.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    return {
        "separate_engine_command": '#include "strat01_gguf_rung2d.h"' in engine and "--strat01-gguf-rung2d" in engine,
        "exact_layer2_inventory": r2d.count('{"blk.2.') == 16 and '"blk.2.attn_norm.weight"' in r2d and '"blk.2.ffn_down_shexp.weight"' in r2d,
        "accepted_layer1_reused": "strat01_r2c_build_attention_range(path,a1" in r2d and "strat01_r2c_run_moe(path,m1" in r2d,
        "layer2_reuses_same_primitives": "strat01_r2c_build_attention_range(path,a2" in r2d and "strat01_r2c_run_moe(path,m2" in r2d,
        "layer1_swiglu_unchanged": "strat01_sse2_swiglu_compute" not in r2c_text and "expf(-g[i])" in r2c_text and "expf(-a->gate[i])" in r2c_text,
        "complete_layer2_surface": r2d.count('F("') == 31 and '"ffn_moe_topk-2"' in r2d and "STRAT01_R2C_DUMPS" in r2d,
        "three_layer_cache_schema": "layer<3U" in r2d and "cache_paths[6]" in r2d and "cache_sha[6][65]" in r2d,
        "exact_helper_totals": "3U*2304U==6912U" in r2d and "3U*262144U==786432U" in r2d,
        "separate_reference_variant": "STRAT01_RUNG2D" in ref and "strat01_engine_rung2d_reference_manifest_v1" in ref and "rung2d" in build,
        "reference_layer2_surface": ref.count('"Kcur-2"') >= 3 and '"l_out-2"' in ref and '\\"layers\\":[0,1,2]' in ref,
        "protocol_frozen": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol and PREDECESSOR_SHA in protocol,
        "audit_names_changed_coordinate": "LAYER2_IS_THE_NEXT_NON_DUPLICATE_DEPTH_BOUNDARY" in audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    if not args.apparatus_only:
        raise SystemExit("layer-2 scientific execution is not implemented; use --apparatus-only")
    output = (args.output_dir or DEFAULT_OUTPUT).resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    status = "VOID_ENGINE_LAYER2_DEPTH_EXTENSION"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    predecessor: dict[str, str] = {}
    controls: dict[str, bool] = {}
    compiler = shutil.which("clang")
    binary: Path | None = None
    reference_binary: Path | None = None
    try:
        sources = source_inventory()
        predecessor = validate_predecessor()
        controls = source_controls()
        if not all(controls.values()):
            raise DepthExtensionError("layer-2 source controls failed: " + ", ".join(k for k, v in controls.items() if not v))
        if not compiler:
            raise DepthExtensionError("clang unavailable")
        binary = output / "engine_layer2_depth_extension.exe"
        commands["compile"] = r2c.base.run_command([compiler, *r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile", timeout=600)
        r2c.base.require_ok(commands["compile"], "layer-2 engine compile")
        reference_binary = reference_build.build(output / "reference-build", rung2d=True)
        commands["reference_build"] = {"returncode": 0, "binary": str(reference_binary)}
        commands["reference_selftest"] = r2c.base.run_command([str(reference_binary), "--self-test"], output=output, label="reference_selftest", timeout=300)
        r2c.base.require_ok(commands["reference_selftest"], "layer-2 reference selftest")
        commands["python_tests"] = r2c.base.run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], output=output, label="all_strat01_unittests", timeout=1800)
        r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
        for index, option in enumerate(SELFTESTS):
            label = f"selftest_{index:02d}"
            commands[label] = r2c.base.run_command([str(binary), option], output=output, label=label, timeout=300)
            r2c.base.require_ok(commands[label], option)
        status = "APPARATUS_READY_NO_DONOR_EXECUTION"
    except (DepthExtensionError, r2c.RunnerError, r2c.base.RunnerError, reference_build.BuildError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record = {
        "schema": "strat01_engine_layer2_depth_extension_apparatus_v1",
        "status": status, "errors": errors,
        "production_invocations": 0, "reference_invocations": 0,
        "donor_graph_executions": 0, "reference_graph_executions": 0,
        "expected_scientific_helper_counts": EXPECTED_COUNTS,
        "source_controls": controls,
        "provenance": {
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": time.perf_counter() - started,
            "git_head_observed": r2c.base.git_value(["git", "rev-parse", "HEAD"]),
            "source_hashes": sources, "predecessor": predecessor,
            "artifact": {"path_recorded_not_opened": str(r2c.MODEL), "expected_bytes": r2c.base.EXPECTED_BYTES, "expected_sha256": r2c.base.EXPECTED_SHA256},
            "environment": {"platform": platform.platform(), "python": sys.version, "cwd": os.getcwd(), "clang_path": compiler},
            "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None},
            "reference_binary": {"path": str(reference_binary) if reference_binary else None, "sha256": sha(reference_binary) if reference_binary and reference_binary.is_file() else None},
            "commands": commands,
        },
        "non_claims": ["layer-2 numerical fidelity", "layers 3-25", "tokenizer/logits/generation", "quality", "RAM", "rate"],
    }
    r2c.base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
