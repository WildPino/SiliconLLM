#!/usr/bin/env python3
"""Run the frozen layer-1 router/SwiGLU local compile-parity cell."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2a as r2a
from benchmarks.donor_adaptation.engine import run_strat01_q6k_q8k_reference_generic_production_integration as prior

HERE = Path(__file__).resolve().parent
PROBE = HERE / "strat01_layer1_numerical_primitives_probe.c"
ORACLE = HERE / "strat01_layer1_f32_router_oracle.cpp"
HEADER = ROOT / "benchmarks/phase60/strat01_f32_dot_reference_generic.h"
SWIGLU = ROOT / "benchmarks/phase60/strat01_swiglu_sse2.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_PROTOCOL_20260925.md"
ADDENDUM = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_ADDENDUM_A_20260925.md"
TESTS = HERE / "test_strat01_layer1_numerical_primitives_compile_parity.py"
MODEL = prior.MODEL
SOURCE = prior.DEFAULT_OUTPUT
REFERENCE = prior.REFERENCE_ROOT
RECOVERY = HERE / "results/strat01_gigachat_engine_q6k_q8k_reference_generic_production_integration_offline_recovery1_20260925/adjudication.json"
RECOVERY_SHA = "cb2a2d9cbfc80d8eb08dfc32d0f850fae717cf9a76525a6f81a9f600269a3b93"
SSE2_ADJ = HERE / "results/strat01_gigachat_engine_post_f16_block0_swiglu_sse2_semantics_20260924/adjudication.json"
SSE2_ADJ_SHA = "46f127215e4bfc00b80ba4eb151d20c294482d2cb5f3c976616e953d21852c0e"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_numerical_primitives_compile_parity_repair1_20260925"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_numerical_primitives_compile_parity_apparatus_repair2_20260925"
ROUTER_OFFSET = 509_582_464
ROUTER_SPAN = 393_216
OUTPUT_COUNTS = {
    "router_float_replay.f32": 512, "router_candidate.f32": 512,
    "router_oracle.f32": 512, "router_promoted_product.f32": 512,
    "router_input_mutated.f32": 512, "router_weight_mutated.f32": 512,
    "routed_scalar_replay.f32": 40_960, "routed_sse2.f32": 40_960,
    "routed_gate_mutated.f32": 40_960, "routed_swapped.f32": 40_960,
    "shared_scalar_replay.f32": 10_240, "shared_sse2.f32": 10_240,
}


class DiagnosticError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return r2a.sha256_file(path)


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {"runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
             "addendum": ADDENDUM,
             "probe": PROBE, "oracle": ORACLE, "candidate_header": HEADER, "swiglu": SWIGLU}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing numerical-primitives source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise DiagnosticError("numerical-primitives sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise DiagnosticError(f"untracked numerical-primitives source: {path}")


def source_controls() -> dict[str, bool]:
    header = HEADER.read_text(encoding="utf-8")
    oracle = ORACLE.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    addendum = ADDENDUM.read_text(encoding="utf-8")
    return {
        "protocol_frozen_and_corrected_preimplementation": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol and "Pre-implementation compiler-path correction" in protocol,
        "repair_authorized_after_void": "REPAIR 1 AUTHORIZED" in addendum and "VOID_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY" in addendum and "scientific claim" in addendum,
        "candidate_target_isolated": 'target("no-avx,no-avx2,no-fma")' in header and "noinline" in header,
        "candidate_float_product_double_accumulator": "const float product = x[i] * y[i]" in header and "sum += (double) product" in header,
        "oracle_independent_translation_unit": "strat01_f32_dot_reference_generic" not in oracle and "(double) (x[i] * y[i])" in oracle,
        "production_float_replay_present": "#pragma STDC FP_CONTRACT OFF" in probe and "static float dot_float" in probe,
        "promoted_product_control_present": "static float dot_promoted_product" in probe,
        "shared_swiglu_reused": "strat01_sse2_swiglu_compute" in probe and "strat01_sse2_expf_no_fma" not in probe,
        "no_engine_or_graph_entrypoint": "engine.c" not in probe and "GRAPH_COMPLETE" not in probe,
        "exact_router_descriptor": "ROUTER_FILE_OFFSET 509582464LL" in probe and "ROUTER_IN 1536U" in probe and "ROUTER_ROWS 64U" in probe,
        "material_one_value_mutations": probe.count("+ 1.0f") == 3,
    }


def validate_bindings() -> dict[str, Any]:
    if sha(RECOVERY) != RECOVERY_SHA or sha(SSE2_ADJ) != SSE2_ADJ_SHA:
        raise DiagnosticError("predecessor binding mismatch")
    recovery = read_json(RECOVERY, "Q6 recovery")
    sse2 = read_json(SSE2_ADJ, "SSE2 predecessor")
    if recovery.get("status") != "FAIL_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION" or recovery.get("errors") != [] or recovery.get("new_production_invocations") != 0:
        raise DiagnosticError("Q6 recovery state mismatch")
    descriptor = next(item for item in sse2["c_report"]["layer1_tensors"] if item["name"] == "blk.1.ffn_gate_inp.weight")
    if descriptor != {"name": "blk.1.ffn_gate_inp.weight", "type": 0, "offset": 503_479_552, "file_offset": ROUTER_OFFSET, "span": ROUTER_SPAN}:
        raise DiagnosticError("router descriptor mismatch")
    return {"recovery": {"path": str(RECOVERY), "sha256": RECOVERY_SHA}, "sse2": {"path": str(SSE2_ADJ), "sha256": SSE2_ADJ_SHA}, "router_descriptor": descriptor}


def capture_paths() -> dict[str, Path]:
    c = SOURCE / "c_engine"
    ref = REFERENCE / "prefill8"
    return {
        "norm": c / "prefill8_ffn_norm-1.f32", "router_current": c / "prefill8_ffn_moe_logits-1.f32",
        "routed_gate": c / "prefill8_ffn_moe_gate-1.f32", "routed_up": c / "prefill8_ffn_moe_up-1.f32",
        "routed_current": c / "prefill8_ffn_moe_swiglu-1.f32", "shared_gate": c / "prefill8_ffn_gate-1.f32",
        "shared_up": c / "prefill8_ffn_up-1.f32", "shared_current": c / "prefill8_ffn_swiglu-1.f32",
        "router_reference": ref / "ffn_moe_logits-1.full.f32le", "routed_reference": ref / "ffn_moe_swiglu-1.full.f32le",
        "shared_reference": ref / "ffn_swiglu-1.full.f32le",
    }


def load(path: Path, count: int, label: str) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * 4:
        raise DiagnosticError(f"{label} size mismatch")
    values = np.fromfile(path, dtype="<f4")
    if values.size != count or not bool(np.isfinite(values).all()):
        raise DiagnosticError(f"{label} payload invalid")
    return values


def validate_twins(paths: dict[str, Path]) -> dict[str, str]:
    counts = {"norm": 12_288, "router_current": 512, "routed_gate": 40_960, "routed_up": 40_960,
              "routed_current": 40_960, "shared_gate": 10_240, "shared_up": 10_240, "shared_current": 10_240}
    hashes: dict[str, str] = {}
    for name, count in counts.items():
        prefill = paths[name]
        cached = Path(str(prefill).replace("prefill8_", "cached7p1_"))
        if not np.array_equal(load(prefill, count, name), load(cached, count, name + " cached")):
            raise DiagnosticError(f"candidate schedule twin mismatch: {name}")
        hashes[name] = sha(prefill)
    for name, count in (("router_reference", 512), ("routed_reference", 40_960), ("shared_reference", 10_240)):
        prefill = paths[name]
        cached = Path(str(prefill).replace("prefill8", "cached7p1"))
        if not np.array_equal(load(prefill, count, name), load(cached, count, name + " cached")):
            raise DiagnosticError(f"reference schedule twin mismatch: {name}")
        hashes[name] = sha(prefill)
    return hashes


def classify(router: bool, routed: bool, shared: bool) -> str:
    if router and routed and shared:
        return "LAYER1_ROUTER_AND_SWIGLU_EXACT_PRIMITIVES"
    if router:
        return "LAYER1_ROUTER_EXACT_SWIGLU_INSUFFICIENT"
    if routed and shared:
        return "LAYER1_SWIGLU_EXACT_ROUTER_INSUFFICIENT"
    return "LAYER1_NUMERICAL_PRIMITIVES_INSUFFICIENT"


def adjudicate(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    values = {name: load(root / name, count, name) for name, count in OUTPUT_COUNTS.items()}
    expected = {
        "router_current": load(paths["router_current"], 512, "router current"),
        "router_reference": load(paths["router_reference"], 512, "router reference"),
        "routed_current": load(paths["routed_current"], 40_960, "routed current"),
        "routed_reference": load(paths["routed_reference"], 40_960, "routed reference"),
        "shared_current": load(paths["shared_current"], 10_240, "shared current"),
        "shared_reference": load(paths["shared_reference"], 10_240, "shared reference"),
    }
    exact = {
        "router_scalar_replay": bool(np.array_equal(values["router_float_replay.f32"], expected["router_current"])),
        "router_candidate_oracle": bool(np.array_equal(values["router_candidate.f32"], values["router_oracle.f32"])),
        "router_candidate_reference": bool(np.array_equal(values["router_candidate.f32"], expected["router_reference"])),
        "routed_scalar_replay": bool(np.array_equal(values["routed_scalar_replay.f32"], expected["routed_current"])),
        "routed_sse2_reference": bool(np.array_equal(values["routed_sse2.f32"], expected["routed_reference"])),
        "shared_scalar_replay": bool(np.array_equal(values["shared_scalar_replay.f32"], expected["shared_current"])),
        "shared_sse2_reference": bool(np.array_equal(values["shared_sse2.f32"], expected["shared_reference"])),
    }
    controls = {
        "float_accumulator_rejects": not np.array_equal(values["router_float_replay.f32"], expected["router_reference"]),
        "promoted_product_rejects": not np.array_equal(values["router_promoted_product.f32"], expected["router_reference"]),
        "router_input_mutation_rejects": not np.array_equal(values["router_input_mutated.f32"], expected["router_reference"]),
        "router_weight_mutation_rejects": not np.array_equal(values["router_weight_mutated.f32"], expected["router_reference"]),
        "routed_scalar_distinct": not np.array_equal(values["routed_scalar_replay.f32"], expected["routed_reference"]),
        "shared_scalar_distinct": not np.array_equal(values["shared_scalar_replay.f32"], expected["shared_reference"]),
        "routed_gate_mutation_rejects": not np.array_equal(values["routed_gate_mutated.f32"], expected["routed_reference"]),
        "routed_swap_rejects": not np.array_equal(values["routed_swapped.f32"], expected["routed_reference"]),
    }
    apparatus_exact = {
        name: exact[name] for name in (
            "router_scalar_replay", "router_candidate_oracle",
            "routed_scalar_replay", "shared_scalar_replay",
        )
    }
    if not all(apparatus_exact.values()):
        failed = [name for name, passed in apparatus_exact.items() if not passed]
        raise DiagnosticError("replay/oracle exact gate failed: " + ", ".join(failed))
    if not all(controls.values()):
        failed = [name for name, passed in controls.items() if not passed]
        raise DiagnosticError("negative-control gate failed: " + ", ".join(failed))
    component = classify(exact["router_candidate_reference"], exact["routed_sse2_reference"], exact["shared_sse2_reference"])
    metrics = {
        "router_float": r2a.metrics(values["router_float_replay.f32"], expected["router_reference"]),
        "router_candidate": r2a.metrics(values["router_candidate.f32"], expected["router_reference"]),
        "routed_scalar": r2a.metrics(values["routed_scalar_replay.f32"], expected["routed_reference"]),
        "routed_sse2": r2a.metrics(values["routed_sse2.f32"], expected["routed_reference"]),
        "shared_scalar": r2a.metrics(values["shared_scalar_replay.f32"], expected["shared_reference"]),
        "shared_sse2": r2a.metrics(values["shared_sse2.f32"], expected["shared_reference"]),
    }
    return {"status": component, "exact": exact, "controls": controls, "metrics": metrics,
            "output_hashes": {name: sha(root / name) for name in OUTPUT_COUNTS}}


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
    status = "VOID_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    captures: dict[str, str] = {}
    result: dict[str, Any] = {"status": "NOT_RUN"}
    compiler = shutil.which("clang")
    cxx = shutil.which("clang++")
    binary: Path | None = None
    diagnostic_invocations = 0
    model_reads = 0
    try:
        sources = source_inventory()
        bindings = validate_bindings()
        controls = source_controls()
        if not all(controls.values()) or not compiler or not cxx:
            raise DiagnosticError("apparatus source/compiler gate failed")
        probe_obj, oracle_obj = output / "probe.o", output / "oracle.o"
        binary = output / "layer1_numerical_primitives.exe"
        commands["compile_probe"] = r2a.run_command([compiler, "-std=c11", "-O3", "-mavx2", "-mfma", "-c", str(PROBE), "-o", str(probe_obj)], cwd=ROOT, output=output, label="compile_probe", timeout=600)
        r2a.require_ok(commands["compile_probe"], "probe compile")
        commands["compile_oracle"] = r2a.run_command([cxx, "-std=c++17", "-O3", "-DGGML_CPU_GENERIC", "-c", str(ORACLE), "-o", str(oracle_obj)], cwd=ROOT, output=output, label="compile_oracle", timeout=600)
        r2a.require_ok(commands["compile_oracle"], "oracle compile")
        commands["link"] = r2a.run_command([cxx, str(probe_obj), str(oracle_obj), "-o", str(binary), "-lm"], cwd=ROOT, output=output, label="link", timeout=600)
        r2a.require_ok(commands["link"], "probe link")
        commands["tests"] = r2a.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_layer1_numerical_primitives_compile_parity"], cwd=ROOT, output=output, label="tests", timeout=600)
        r2a.require_ok(commands["tests"], "Python tests")
        commands["selftest"] = r2a.run_command([str(binary), "--selftest"], cwd=ROOT, output=output, label="selftest", timeout=300)
        r2a.require_ok(commands["selftest"], "probe selftest")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size != prior.r2c.base.EXPECTED_BYTES or sha(model) != prior.r2c.base.EXPECTED_SHA256:
                raise DiagnosticError("accepted artifact identity mismatch")
            paths = capture_paths()
            captures = validate_twins(paths)
            root = output / "diagnostic"
            root.mkdir()
            diagnostic_invocations = 1
            model_reads = 1
            command = [str(binary), "--run", str(model), str(paths["norm"]), str(paths["routed_gate"]), str(paths["routed_up"]), str(paths["shared_gate"]), str(paths["shared_up"]), str(root)]
            commands["diagnostic"] = r2a.run_command(command, cwd=ROOT, output=output, label="diagnostic", timeout=1800)
            r2a.require_ok(commands["diagnostic"], "local diagnostic")
            result = adjudicate(root, paths)
            status = result["status"]
    except (DiagnosticError, r2a.RunnerError, prior.IntegrationError, prior.r2c.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started,
                  "git_head_observed" if args.apparatus_only else "git_head": r2a.git_value(["git", "rev-parse", "HEAD"]),
                  "source_hashes": sources, "bindings": bindings, "capture_hashes": captures,
                  "artifact": {"path": str(model), "expected_bytes": prior.r2c.base.EXPECTED_BYTES, "expected_sha256": prior.r2c.base.EXPECTED_SHA256},
                  "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang": compiler, "clangxx": cxx},
                  "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_layer1_numerical_primitives_compile_parity_v1", "status": status, "errors": errors,
              "diagnostic_invocations": diagnostic_invocations, "model_reads": model_reads, "producer_invocations": 0,
              "donor_graph_executions": 0, "reference_graph_executions": 0, "source_controls": source_controls() if sources else {},
              "adjudication": result, "non_claims": ["production integration", "Q6 propagation", "later layers", "quality", "generation", "RAM", "rate"], "provenance": provenance}
    r2a.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status.startswith("LAYER1_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
