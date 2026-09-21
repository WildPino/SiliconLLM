#!/usr/bin/env python3
"""Run the frozen no-donor K-B Q5_0/Q8_0 operator diagnostic."""
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

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2a as r2a
from benchmarks.donor_adaptation.engine.build_strat01_kb_q5q8_diagnostic import build, verify_pinned_llama

HERE = Path(__file__).resolve().parent
ENGINE = ROOT / "benchmarks/phase60/engine.c"
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_kb_q5q8_diag.h"
HELPER_SOURCE = HERE / "strat01_kb_q5q8_diagnostic.cpp"
HELPER_BUILD = HERE / "build_strat01_kb_q5q8_diagnostic.py"
TEST_SOURCE = HERE / "test_strat01_kb_q5q8_diagnostic.py"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_KB_Q5_0_Q8_0_DIAGNOSTIC_PROTOCOL_20260921.md"
MODEL = r2a.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_kb_q5q8_diag_20260921"

R2A_RUN = HERE / "results/strat01_gigachat_engine_rung2a_repair2_20260921"
UPSTREAM_RUN = HERE / "results/strat01_gigachat_engine_upstream_rmsnorm_diag_20260921"
FLOAT_RUN = HERE / "results/strat01_gigachat_engine_attention_vb_repair_20260921"
PAYLOADS = {
    "reference_q": (R2A_RUN / "pinned_reference/prefill8/q-0.full.f32le", 196_608, "4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b"),
    "upstream_q": (UPSTREAM_RUN / "candidate/prefill8_q-0.f32", 196_608, "6255f40d5a717af2a742c484b0ab75ff549b37acea70f73556698c794c8b3366"),
    "float_q": (FLOAT_RUN / "c_engine/prefill8_q-0.f32", 196_608, "cbc263ed903c9a7d992ebd2f14795ea60f60b999efc3e5365be6b7d84fe1252b"),
    "reference_target": (R2A_RUN / "pinned_reference/prefill8/q_nope_absorbed_perm-0.full.f32le", 524_288, "94ddcff3f90faca65dc7237222939f407751f5c29e1c3f145e817de1e9c45ee9"),
    "upstream_target": (UPSTREAM_RUN / "candidate/prefill8_q_nope_absorbed_perm-0.f32", 524_288, "11adb3548a0b69320cc4a38c45e1350763a35ebd298a8f1bd0e6efe7c5ee1dee"),
    "float_target": (FLOAT_RUN / "c_engine/prefill8_q_nope_absorbed_perm-0.f32", 524_288, "715e12661c6934f82ae9eec437da5713a83405edac64cb4bb04f86482d3f3a99"),
}
OUTPUT_COUNT = 8 * 32 * 512
Q8_BYTES = 8 * 32 * 4 * 34
TIGHT = (2e-6, 1e-5)
GENERAL = (2e-3, 1e-2)


class DiagnosticError(RuntimeError):
    pass


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {"runner": Path(__file__).resolve(), "engine": ENGINE, "rung2a_header": r2a.RUNG2A_HEADER, "diagnostic_header": HEADER, "helper_source": HELPER_SOURCE, "helper_build": HELPER_BUILD, "tests": TEST_SOURCE, "protocol": PROTOCOL}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": r2a.sha256_file(path)} for name, path in paths.items()}


def validate_payloads() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, (path, byte_count, digest) in PAYLOADS.items():
        if not path.is_file() or path.stat().st_size != byte_count or r2a.sha256_file(path) != digest:
            raise DiagnosticError(f"immutable payload mismatch: {name}")
        values = np.fromfile(path, dtype="<f4")
        if values.size * 4 != byte_count or not bool(np.isfinite(values).all()):
            raise DiagnosticError(f"invalid float payload: {name}")
        arrays[name] = values
    return arrays


def metrics(candidate: np.ndarray, target: np.ndarray, limits: tuple[float, float]) -> dict[str, float | bool]:
    if candidate.shape != target.shape or not bool(np.isfinite(candidate).all() and np.isfinite(target).all()):
        raise DiagnosticError("invalid metric operands")
    delta = candidate.astype(np.float64) - target.astype(np.float64)
    rms_target = math.sqrt(float(np.mean(target.astype(np.float64) ** 2)))
    nrmse = math.sqrt(float(np.mean(delta * delta))) / max(rms_target, 1e-12)
    normalized_max = float(np.max(np.abs(delta))) / max(float(np.max(np.abs(target))), 1e-6)
    return {"nrmse": nrmse, "normalized_max": normalized_max, "limits": list(limits), "pass": nrmse <= limits[0] and normalized_max <= limits[1]}


def classify(reference: dict[str, Any], upstream: dict[str, Any], controls: dict[str, bool]) -> str:
    if not all(controls.values()):
        return "VOID_KB_Q5_0_Q8_0_DIAGNOSTIC"
    if bool(reference["pass"]):
        return "Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY" if bool(upstream["pass"]) else "Q5_0_Q8_0_REFERENCE_ONLY"
    return "Q5_0_Q8_0_FAILS_REFERENCE_INPUT"


def run_command(command: list[str], output: Path, label: str, timeout: int) -> dict[str, Any]:
    return r2a.run_command(command, cwd=ROOT, output=output, label=label, timeout=timeout)


def contained(root: Path, reported: str, byte_count: int, label: str) -> Path:
    try:
        path = Path(reported).resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DiagnosticError(f"{label} escapes output root or is absent") from exc
    if not path.is_file() or path.stat().st_size != byte_count:
        raise DiagnosticError(f"{label} byte count mismatch")
    return path


def load_output(path: Path, byte_count: int, digest: str | None = None) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != byte_count or (digest is not None and r2a.sha256_file(path) != digest):
        raise DiagnosticError(f"output identity mismatch: {path}")
    values = np.fromfile(path, dtype="<f4")
    if not bool(np.isfinite(values).all()):
        raise DiagnosticError(f"non-finite output: {path}")
    return values


def q8_census(a: bytes, b: bytes) -> dict[str, int]:
    if len(a) != Q8_BYTES or len(b) != Q8_BYTES:
        raise DiagnosticError("Q8 census size mismatch")
    return {"changed_blocks": sum(a[i:i+34] != b[i:i+34] for i in range(0, Q8_BYTES, 34)), "total_blocks": Q8_BYTES // 34, "changed_bytes": sum(x != y for x, y in zip(a, b)), "total_bytes": Q8_BYTES}


def validate_c_outputs(root: Path, model: Path, sources: dict[str, dict[str, str]]) -> tuple[dict[str, Path], dict[str, Any]]:
    try:
        report = json.loads((root / "strat01_kb_q5q8_diag.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"C report missing/malformed: {exc}") from exc
    if report.get("command") != "--strat01-kb-q5q8-diagnostic" or report.get("state") != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report.get("self_certifies_pass") is not False or report.get("donor_graph_executions") != 0 or report.get("timing_or_rate_claim") is not None:
        raise DiagnosticError("C report state contract mismatch")
    identity = report.get("model", {})
    if Path(identity.get("path", "")).resolve(strict=True) != model.resolve(strict=True) or identity.get("bytes") != r2a.EXPECTED_BYTES or identity.get("sha256") != r2a.EXPECTED_SHA256:
        raise DiagnosticError("C artifact identity mismatch")
    descriptor = report.get("tensor", {})
    if descriptor != {"name": "blk.0.attn_k_b.weight", "type": "Q5_0", "dims": [128, 512, 32], "offset": 272421888, "span": 1441792, "file_offset": 278524800}:
        raise DiagnosticError("C tensor descriptor mismatch")
    if report.get("engine_source_sha256") != sources["engine"]["sha256"] or report.get("diagnostic_source_sha256") != sources["diagnostic_header"]["sha256"]:
        raise DiagnosticError("C source identity mismatch")
    paths: dict[str, Path] = {}
    arms = report.get("arms", {})
    if set(arms) != {"reference", "upstream_double", "accepted_float"}:
        raise DiagnosticError("C arm set mismatch")
    for arm in ("reference", "upstream_double"):
        if set(arms[arm]) != {"current", "q5q8", "q8"}:
            raise DiagnosticError(f"C {arm} schema mismatch")
        for kind, size in (("current", OUTPUT_COUNT * 4), ("q5q8", OUTPUT_COUNT * 4), ("q8", Q8_BYTES)):
            item = arms[arm][kind]; path = contained(root, item.get("path", ""), size, f"C {arm}/{kind}")
            if r2a.sha256_file(path) != item.get("sha256"):
                raise DiagnosticError(f"C {arm}/{kind} hash mismatch")
            paths[f"{arm}_{kind}"] = path
    if set(arms["accepted_float"]) != {"current", "q8"}:
        raise DiagnosticError("C accepted-float schema mismatch")
    for kind, size in (("current", OUTPUT_COUNT * 4), ("q8", Q8_BYTES)):
        item = arms["accepted_float"][kind]; path = contained(root, item.get("path", ""), size, f"C accepted_float/{kind}")
        if r2a.sha256_file(path) != item.get("sha256"):
            raise DiagnosticError(f"C accepted_float/{kind} hash mismatch")
        paths[f"accepted_float_{kind}"] = path
    controls = report.get("controls", {})
    if set(controls) != {"one_byte_input_mutation", "wrong_head_stride", "corrupt_qh"}:
        raise DiagnosticError("C control schema mismatch")
    for name, reported in controls.items():
        paths[f"control_{name}"] = contained(root, reported, OUTPUT_COUNT * 4, f"C control/{name}")
    return paths, report


def run_actual(model: Path, output: Path, c_binary: Path, helper: Path, sources: dict[str, dict[str, str]], arrays: dict[str, np.ndarray], commands: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    c_root = output / "c_engine"; c_root.mkdir()
    command = [str(c_binary), "--strat01-kb-q5q8-diagnostic", str(model), "--reference-q", str(PAYLOADS["reference_q"][0]), "--upstream-q", str(PAYLOADS["upstream_q"][0]), "--float-q", str(PAYLOADS["float_q"][0]), "--out-dir", str(c_root)]
    commands["c_diagnostic"] = run_command(command, output, "c_diagnostic", 7200); r2a.require_ok(commands["c_diagnostic"], "C diagnostic")
    c_paths, c_report = validate_c_outputs(c_root, model, sources)
    helper_paths: dict[str, Path] = {}
    for arm, key in (("reference", "reference_q"), ("upstream_double", "upstream_q")):
        arm_root = output / f"pinned_helper_{arm}"; arm_root.mkdir()
        commands[f"helper_{arm}"] = run_command([str(helper), "--model", str(model), "--input", str(PAYLOADS[key][0]), "--output-dir", str(arm_root)], output, f"helper_{arm}", 1800); r2a.require_ok(commands[f"helper_{arm}"], f"helper {arm}")
        helper_paths[f"{arm}_q8"] = contained(arm_root, str(arm_root / "q8.bin"), Q8_BYTES, f"helper {arm}/q8")
        helper_paths[f"{arm}_q5q8"] = contained(arm_root, str(arm_root / "q5q8.f32le"), OUTPUT_COUNT * 4, f"helper {arm}/q5q8")

    controls: dict[str, bool] = {"current_upstream_replay": r2a.sha256_file(c_paths["upstream_double_current"]) == PAYLOADS["upstream_target"][2], "current_float_replay": r2a.sha256_file(c_paths["accepted_float_current"]) == PAYLOADS["float_target"][2]}
    comparisons: dict[str, Any] = {}
    for arm, target_key in (("reference", "reference_target"), ("upstream_double", "reference_target")):
        c_q8 = c_paths[f"{arm}_q8"].read_bytes(); h_q8 = helper_paths[f"{arm}_q8"].read_bytes()
        controls[f"q8_exact_{arm}"] = c_q8 == h_q8
        c_out = load_output(c_paths[f"{arm}_q5q8"], OUTPUT_COUNT * 4); helper_out = load_output(helper_paths[f"{arm}_q5q8"], OUTPUT_COUNT * 4)
        comparisons[f"c_vs_helper_{arm}"] = metrics(c_out, helper_out, TIGHT)
        controls[f"helper_agreement_{arm}"] = bool(comparisons[f"c_vs_helper_{arm}"]["pass"])
    reference_out = load_output(c_paths["reference_q5q8"], OUTPUT_COUNT * 4)
    upstream_out = load_output(c_paths["upstream_double_q5q8"], OUTPUT_COUNT * 4)
    comparisons["reference_primary"] = metrics(reference_out, arrays["reference_target"], TIGHT)
    comparisons["upstream_primary"] = metrics(upstream_out, arrays["reference_target"], GENERAL)
    comparisons["reference_current_vs_reference"] = metrics(load_output(c_paths["reference_current"], OUTPUT_COUNT * 4), arrays["reference_target"], TIGHT)
    comparisons["upstream_current_vs_frozen"] = metrics(load_output(c_paths["upstream_double_current"], OUTPUT_COUNT * 4), arrays["upstream_target"], TIGHT)
    comparisons["float_current_vs_frozen"] = metrics(load_output(c_paths["accepted_float_current"], OUTPUT_COUNT * 4), arrays["float_target"], TIGHT)
    for name in ("one_byte_input_mutation", "wrong_head_stride", "corrupt_qh"):
        result = metrics(load_output(c_paths[f"control_{name}"], OUTPUT_COUNT * 4), arrays["reference_target"], TIGHT)
        comparisons[f"control_{name}"] = result; controls[f"{name}_fires"] = not bool(result["pass"])
    controls["omitted_q8_quantization_fires"] = not bool(comparisons["reference_current_vs_reference"]["pass"])
    q8 = {name: c_paths[f"{name}_q8"].read_bytes() for name in ("reference", "upstream_double", "accepted_float")}
    census = {"reference_vs_upstream_double": q8_census(q8["reference"], q8["upstream_double"]), "reference_vs_accepted_float": q8_census(q8["reference"], q8["accepted_float"]), "upstream_double_vs_accepted_float": q8_census(q8["upstream_double"], q8["accepted_float"])}
    status = classify(comparisons["reference_primary"], comparisons["upstream_primary"], controls)
    return status, {"comparisons": comparisons, "controls": controls, "q8_census": census, "c_report": c_report, "outputs": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": r2a.sha256_file(path)} for name, path in {**c_paths, **helper_paths}.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve(); output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter(); status = "VOID_KB_Q5_0_Q8_0_DIAGNOSTIC"; errors: list[str] = []; commands: dict[str, Any] = {}; sources: dict[str, dict[str, str]] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}; c_binary: Path | None = None; helper: Path | None = None
    compiler = shutil.which("clang")
    try:
        sources = source_inventory(); validate_payloads(); verify_pinned_llama()
        if not compiler:
            raise DiagnosticError("clang is unavailable")
        if not args.apparatus_only:
            critical = [Path(item["path"]) for item in sources.values()]
            relative = [path.resolve().relative_to(ROOT.resolve()) for path in critical]
            dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False)
            untracked = [path for path in relative if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode]
            if dirty.returncode or untracked:
                raise DiagnosticError("diagnostic implementation/protocol differs from HEAD")
            if not model.is_file() or model.stat().st_size != r2a.EXPECTED_BYTES or r2a.sha256_file(model) != r2a.EXPECTED_SHA256:
                raise DiagnosticError("accepted model identity mismatch")
        commands["clang_version"] = run_command([compiler, "--version"], output, "clang_version", 30); r2a.require_ok(commands["clang_version"], "clang version")
        c_binary = output / "engine_kb_q5q8_diag.exe"
        commands["compile_c"] = run_command([compiler, *r2a.COMPILE_FLAGS, str(ENGINE), "-o", str(c_binary), "-lm"], output, "compile_c", 600); r2a.require_ok(commands["compile_c"], "C build")
        commands["c_selftest"] = run_command([str(c_binary), "--strat01-kb-q5q8-diagnostic-selftest"], output, "c_selftest", 300); r2a.require_ok(commands["c_selftest"], "C selftest")
        commands["python_tests"] = run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_kb_q5q8_diagnostic"], output, "python_tests", 300); r2a.require_ok(commands["python_tests"], "Python tests")
        helper = build(output / "helper_build")
        commands["helper_selftest"] = run_command([str(helper), "--selftest"], output, "helper_selftest", 300); r2a.require_ok(commands["helper_selftest"], "helper selftest")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            arrays = validate_payloads(); status, adjudication = run_actual(model, output, c_binary, helper, sources, arrays, commands); adjudication["status"] = status
    except (DiagnosticError, r2a.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-started, "git_head": r2a.git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": r2a.git_value(["git", "status", "--porcelain"]), "source_hashes": sources, "artifact": {"path": str(model), "expected_bytes": r2a.EXPECTED_BYTES, "expected_sha256": r2a.EXPECTED_SHA256}, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binaries": {"c": {"path": str(c_binary) if c_binary else None, "sha256": r2a.sha256_file(c_binary) if c_binary and c_binary.is_file() else None}, "helper": {"path": str(helper) if helper else None, "sha256": r2a.sha256_file(helper) if helper and helper.is_file() else None}}, "commands": commands}
    record = {"schema": "strat01_kb_q5q8_diagnostic_adjudication_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "adjudication": adjudication, "non_claims": ["production repair", "Rung 2C", "full-model parity", "quality", "generation", "RAM", "rate"], "provenance": provenance}
    r2a.write_json(output / "adjudication.json", record); r2a.write_json(output / "run_manifest.json", {"schema": "strat01_kb_q5q8_diagnostic_run_manifest_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "provenance": provenance})
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    valid = {"APPARATUS_READY_NO_DONOR_EXECUTION", "Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY", "Q5_0_Q8_0_REFERENCE_ONLY", "Q5_0_Q8_0_FAILS_REFERENCE_INPUT"}
    return 0 if status in valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
