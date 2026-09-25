#!/usr/bin/env python3
"""Run the frozen selected-router-weight normalization compile-parity cell."""
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
from benchmarks.donor_adaptation.engine import run_strat01_layer1_numerical_primitives_production_integration as production

HERE = Path(__file__).resolve().parent
PROBE = HERE / "strat01_router_weight_norm_probe.c"
ORACLE = HERE / "strat01_router_weight_norm_oracle.cpp"
HEADER = ROOT / "benchmarks/phase60/strat01_router_weight_norm_reference_generic.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTER_WEIGHT_NORMALIZATION_COMPILE_PARITY_PROTOCOL_20260925.md"
TESTS = HERE / "test_strat01_router_weight_normalization_compile_parity.py"
PREDECESSOR = HERE / "results/strat01_gigachat_engine_layer1_numerical_primitives_production_integration_20260925/adjudication.json"
PREDECESSOR_SHA = "3cec456bc7b1d2798976cb9544cd91d36d45451b94911efa1f0ec1ac97437816"
SOURCE = production.DEFAULT_OUTPUT / "c_engine"
REFERENCE = production.REFERENCE_ROOT / "prefill8"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_router_weight_normalization_compile_parity_20260925"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_router_weight_normalization_compile_parity_apparatus_repair1_20260925"
OUTPUTS = {name: 32 for name in (
    "production_replay.f32", "candidate.f32", "oracle.f32", "float_pairwise.f32",
    "unrounded_f64_denominator.f32", "mutated.f32", "permuted.f32", "wrong_origin.f32",
)}


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
             "probe": PROBE, "oracle": ORACLE, "candidate_header": HEADER}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing normalization source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise DiagnosticError("normalization sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise DiagnosticError(f"untracked normalization source: {path}")


def validate_binding() -> dict[str, Any]:
    if sha(PREDECESSOR) != PREDECESSOR_SHA:
        raise DiagnosticError("production-integration binding mismatch")
    record = read_json(PREDECESSOR, "production-integration adjudication")
    failures = record.get("adjudication", {}).get("failures", [])
    expected = {
        f"checkpoint/{arm}/{name}" for arm in r2a.ARMS
        for name in ("ffn_moe_weights_norm-1", "ffn_moe_weighted-1", "ffn_moe_out-1", "ffn_out-1", "l_out-1")
    }
    if (record.get("status") != "FAIL_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION" or
            record.get("errors") != [] or set(failures) != expected or record.get("production_invocations") != 1 or
            record.get("donor_graph_executions") != 2 or record.get("reference_graph_executions") != 0):
        raise DiagnosticError("production-integration state mismatch")
    return {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA, "status": record["status"]}


def source_controls() -> dict[str, bool]:
    header = HEADER.read_text(encoding="utf-8")
    oracle = ORACLE.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    return {
        "protocol_frozen": "FROZEN BEFORE IMPLEMENTATION OR SCIENTIFIC EXECUTION" in protocol,
        "candidate_isolated": 'target("no-avx,no-avx2,no-fma")' in header and "noinline" in header,
        "candidate_double_sum_final_f32": "sum += (double) weights[i]" in header and "float denominator = (float) sum" in header,
        "oracle_independent_translation_unit": "strat01_router_weight_norm_reference_generic" not in oracle and "static_cast<double>" in oracle,
        "production_replay_present": "static void production_norm" in probe and "#pragma STDC FP_CONTRACT OFF" in probe,
        "pairwise_and_unrounded_controls": "static void pairwise_norm" in probe and "static void unrounded_norm" in probe,
        "mutation_permutation_wrong_origin_controls": all(term in probe for term in ("changed[0] += 1.0f", "permuted", "wrong_origin")),
        "exact_clamp_constant": "6.103515625e-5f" in header and "6.103515625e-5f" in probe,
        "no_model_or_graph_entrypoint": "gguf" not in probe.lower() and "GRAPH_COMPLETE" not in probe,
    }


def paths() -> dict[str, Path]:
    return {
        "weights": SOURCE / "prefill8_ffn_moe_weights-1.f32",
        "current": SOURCE / "prefill8_ffn_moe_weights_norm-1.f32",
        "reference": REFERENCE / "ffn_moe_weights_norm-1.full.f32le",
    }


def load(path: Path, label: str) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != 128:
        raise DiagnosticError(f"{label} size mismatch")
    values = np.fromfile(path, dtype="<f4")
    if values.size != 32 or not bool(np.isfinite(values).all()):
        raise DiagnosticError(f"{label} payload invalid")
    return values


def validate_payloads(bound: dict[str, Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in bound.items():
        cached = Path(str(path).replace("prefill8_", "cached7p1_")) if name != "reference" else Path(str(path).replace("prefill8", "cached7p1"))
        if not np.array_equal(load(path, name), load(cached, name + " cached")):
            raise DiagnosticError(f"schedule twin mismatch: {name}")
        hashes[name] = sha(path)
    reference_weights = production.REFERENCE_ROOT / "prefill8/ffn_moe_weights-1.full.f32le"
    if not np.array_equal(load(bound["weights"], "candidate weights"), load(reference_weights, "reference weights")):
        raise DiagnosticError("selected weights are not exact to reference")
    hashes["reference_weights"] = sha(reference_weights)
    return hashes


def adjudicate(root: Path, bound: dict[str, Path]) -> dict[str, Any]:
    values = {name: load(root / name, name) for name in OUTPUTS}
    current, reference = load(bound["current"], "current"), load(bound["reference"], "reference")
    exact = {
        "production_replay": bool(np.array_equal(values["production_replay.f32"], current)),
        "candidate_oracle": bool(np.array_equal(values["candidate.f32"], values["oracle.f32"])),
        "candidate_reference": bool(np.array_equal(values["candidate.f32"], reference)),
    }
    controls = {
        "production_float_rejects": not np.array_equal(values["production_replay.f32"], reference),
        "pairwise_float_rejects": not np.array_equal(values["float_pairwise.f32"], reference),
        "unrounded_f64_denominator_rejects": not np.array_equal(values["unrounded_f64_denominator.f32"], reference),
        "mutation_rejects": not np.array_equal(values["mutated.f32"], reference),
        "permutation_rejects": not np.array_equal(values["permuted.f32"], reference),
        "wrong_origin_rejects": not np.array_equal(values["wrong_origin.f32"], reference),
    }
    if not exact["production_replay"] or not exact["candidate_oracle"]:
        raise DiagnosticError("replay/oracle exact gate failed")
    if not all(controls.values()):
        raise DiagnosticError("negative control gate failed: " + ", ".join(k for k, v in controls.items() if not v))
    status = ("LAYER1_ROUTER_WEIGHT_NORMALIZATION_EXACT" if exact["candidate_reference"]
              else "LAYER1_ROUTER_WEIGHT_NORMALIZATION_INSUFFICIENT")
    metrics = {name: r2a.metrics(value, reference) for name, value in values.items()}
    changed = {name: int(np.count_nonzero(value.view(np.uint32) != reference.view(np.uint32))) for name, value in values.items()}
    return {"status": status, "exact": exact, "controls": controls, "metrics": metrics, "changed": changed,
            "output_hashes": {name: sha(root / name) for name in OUTPUTS}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_LAYER1_ROUTER_WEIGHT_NORMALIZATION_COMPILE_PARITY"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    binding: dict[str, Any] = {}
    captures: dict[str, str] = {}
    result: dict[str, Any] = {"status": "NOT_RUN"}
    compiler, cxx = shutil.which("clang"), shutil.which("clang++")
    binary: Path | None = None
    diagnostic_invocations = payload_reads = 0
    try:
        sources = source_inventory(); binding = validate_binding(); controls = source_controls()
        if not all(controls.values()) or not compiler or not cxx:
            raise DiagnosticError("apparatus source/compiler gate failed")
        probe_obj, oracle_obj = output / "probe.o", output / "oracle.o"
        binary = output / "router_weight_norm.exe"
        commands["compile_probe"] = r2a.run_command([compiler, "-std=c11", "-O3", "-mavx2", "-mfma", "-c", str(PROBE), "-o", str(probe_obj)], cwd=ROOT, output=output, label="compile_probe", timeout=600)
        r2a.require_ok(commands["compile_probe"], "probe compile")
        commands["compile_oracle"] = r2a.run_command([cxx, "-std=c++17", "-O3", "-DGGML_CPU_GENERIC", "-c", str(ORACLE), "-o", str(oracle_obj)], cwd=ROOT, output=output, label="compile_oracle", timeout=600)
        r2a.require_ok(commands["compile_oracle"], "oracle compile")
        commands["link"] = r2a.run_command([cxx, str(probe_obj), str(oracle_obj), "-o", str(binary), "-lm"], cwd=ROOT, output=output, label="link", timeout=600)
        r2a.require_ok(commands["link"], "link")
        commands["tests"] = r2a.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_router_weight_normalization_compile_parity"], cwd=ROOT, output=output, label="tests", timeout=600)
        r2a.require_ok(commands["tests"], "tests")
        commands["selftest"] = r2a.run_command([str(binary), "--selftest"], cwd=ROOT, output=output, label="selftest", timeout=300)
        r2a.require_ok(commands["selftest"], "selftest")
        truncated = output / "truncated.f32"; np.zeros(31, dtype="<f4").tofile(truncated)
        rejected = output / "truncated_reject"; rejected.mkdir()
        commands["truncated_reject"] = r2a.run_command([str(binary), "--run", str(truncated), str(truncated), str(rejected)], cwd=ROOT, output=output, label="truncated_reject", timeout=300)
        if commands["truncated_reject"].get("returncode") == 0:
            raise DiagnosticError("truncated-file control accepted")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_PAYLOAD_EXECUTION"
        else:
            clean_sources_at_head(sources)
            bound = paths(); captures = validate_payloads(bound); payload_reads = 1
            root = output / "diagnostic"; root.mkdir(); diagnostic_invocations = 1
            commands["diagnostic"] = r2a.run_command([str(binary), "--run", str(bound["weights"]), str(bound["current"]), str(root)], cwd=ROOT, output=output, label="diagnostic", timeout=300)
            r2a.require_ok(commands["diagnostic"], "diagnostic")
            result = adjudicate(root, bound); status = result["status"]
    except (DiagnosticError, r2a.RunnerError, production.IntegrationError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-started,
                  "git_head_observed" if args.apparatus_only else "git_head": r2a.git_value(["git", "rev-parse", "HEAD"]),
                  "source_hashes": sources, "binding": binding, "capture_hashes": captures,
                  "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang": compiler, "clangxx": cxx},
                  "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_router_weight_normalization_compile_parity_v1", "status": status, "errors": errors,
              "diagnostic_invocations": diagnostic_invocations, "payload_reads": payload_reads, "model_reads": 0,
              "producer_invocations": 0, "donor_graph_executions": 0, "reference_graph_executions": 0,
              "source_controls": source_controls() if sources else {}, "adjudication": result,
              "non_claims": ["production integration", "expert propagation", "later layers", "quality", "RAM", "rate"], "provenance": provenance}
    r2a.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_PAYLOAD_EXECUTION" or status.startswith("LAYER1_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
