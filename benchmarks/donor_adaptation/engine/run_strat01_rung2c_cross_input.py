#!/usr/bin/env python3
"""Execute and adjudicate the frozen Rung-2C attention cross-input diagnostic."""
from __future__ import annotations

import argparse
import copy
import hashlib
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
HERE = Path(__file__).resolve().parent
ENGINE = ROOT / "benchmarks" / "phase60" / "engine.c"
RUNG2A = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2a.h"
RUNG2C = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2c.h"
CROSS_HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2c_cross_input.h"
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_RUNG2C_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_rung2c_cross_input.py"
RAW = HERE / "results" / "strat01_gigachat_engine_rung2c_repair1_20260923"
RECOVERY = HERE / "results" / "strat01_gigachat_engine_rung2c_repair1_recovery1_20260923"
DEFAULT_MODEL = ROOT / "benchmarks" / "donor_adaptation" / "density" / "results" / "strat01_gigachat_q4_97045b2" / "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_rung2c_cross_input_20260923"
EXPECTED_MODEL_BYTES = 6_474_702_976
EXPECTED_MODEL_SHA = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
LIMITS = (2e-3, 1e-2)
FROZEN_C_METRICS = {"nrmse": 0.006136167591602251, "normalized_max": 0.0030491102772673647}
COMPILE_FLAGS = ["-std=c11", "-O3", "-mavx2", "-mfma"]

PINNED_FILES: dict[str, tuple[Path, str, int]] = {
    "ref_q": (RAW / "pinned_reference" / "prefill8" / "Qcur-1.full.f32le", "9cb50fb41e0d33549eb53bd4ae19605705331562d36f975bb2ac82cc4c41cb16", 589_824),
    "ref_k": (RAW / "pinned_reference" / "prefill8" / "Kcur-1.full.f32le", "73febde321c2ea45a56a5d27a3605b820e08f8b038420a379afc4b9c3f0c72b4", 18_432),
    "ref_v": (RAW / "pinned_reference" / "prefill8" / "Vcur-1.full.f32le", "3f3c00b3fe008aecda5e1164b5da2b295d349270b1e339831da1544a849a9387", 16_384),
    "ref_target": (RAW / "pinned_reference" / "prefill8" / "kqv_out-1.full.f32le", "fa1006c4e2c365d3a5540baebbe4c88180f524d5f69c610fcc3b4205a856d489", 196_608),
    "c_q": (RAW / "c_engine" / "prefill8_Qcur-1.f32", "3faa0a37833c9d9b9cf92d3bf985d13bf6f09b9c278f81e8813258e86e7d2b2f", 589_824),
    "c_k": (RAW / "c_engine" / "prefill8_Kcur-1.f32", "febfcaa2ca7192a71b7ffca1af26252c9e81b010068478928ed6d56fa78fb0d4", 18_432),
    "c_v": (RAW / "c_engine" / "prefill8_Vcur-1.f32", "58fa280015b838d47df9b1a0ba9064d6cc8eb25000e454330d988ec48d537b57", 16_384),
    "c_target": (RAW / "c_engine" / "prefill8_kqv_out-1.f32", "7e6d44661f21dee62b449b44e1939b59d8ad0799c4ccf6779d77a01a48e3ee41", 196_608),
}
EVIDENCE_FILES: dict[str, tuple[Path, str]] = {
    "c_prefill_manifest": (RAW / "c_engine" / "prefill8_manifest.json", "bdaa49fb9493f891f66fa65c48dd703bae7636b35fd173f54c135326355923c3"),
    "reference_prefill_manifest": (RAW / "pinned_reference" / "prefill8" / "manifest.json", "d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451"),
    "recovery_adjudication": (RECOVERY / "adjudication.json", "3742bc5982dd47be36a8e422f7b79c9e6716dc628153e44e6f9e90553c9ba62c"),
    "recovery_run_manifest": (RECOVERY / "run_manifest.json", "898f614a8baaf40716a7aab66e4e31ba51c92da84e133ff24a55e7ccafa60dd4"),
}
ARM_SPECS = {
    "c_q__c_kv": ("c", "c", False, False),
    "ref_q__ref_kv": ("reference", "reference", False, False),
    "c_q__ref_kv": ("c", "reference", False, False),
    "ref_q__c_kv": ("reference", "c", False, False),
    "control_ref_q_token7_negated": ("reference", "reference", True, False),
    "control_ref_kv_rows0_7_swapped": ("reference", "reference", False, True),
}


class RunnerError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_matches(data: bytes, expected_bytes: int, expected_sha: str) -> bool:
    return len(data) == expected_bytes and hashlib.sha256(data).hexdigest() == expected_sha


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_value(args: list[str]) -> str | None:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_command(command: list[str], output: Path, label: str, timeout: int) -> dict[str, Any]:
    started_utc, started = now_utc(), time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=timeout)
        record: dict[str, Any] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        record = {"returncode": None, "stdout": str(getattr(exc, "stdout", "") or ""), "stderr": str(getattr(exc, "stderr", "") or ""), "spawn_or_timeout_error": str(exc)}
    (output / f"{label}.stdout.log").write_text(record["stdout"], encoding="utf-8")
    (output / f"{label}.stderr.log").write_text(record["stderr"], encoding="utf-8")
    record.update({"command": command, "started_utc": started_utc, "seconds": time.perf_counter() - started})
    return record


def require_ok(record: dict[str, Any], label: str) -> None:
    if record.get("returncode") != 0:
        detail = record.get("spawn_or_timeout_error") or record.get("stderr") or record.get("stdout") or ""
        raise RunnerError(f"{label} failed (rc={record.get('returncode')}): {str(detail)[-1600:]}")


def source_inventory() -> dict[str, Any]:
    paths = {"runner": Path(__file__).resolve(), "engine": ENGINE, "rung2a": RUNG2A, "rung2c": RUNG2C, "cross_header": CROSS_HEADER, "protocol": PROTOCOL, "tests": TESTS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunnerError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def validate_frozen_evidence() -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, (path, digest, size) in PINNED_FILES.items():
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise RunnerError(f"pinned {name} identity mismatch")
        resolved[name] = path.resolve(strict=True)
    for name, (path, digest) in EVIDENCE_FILES.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RunnerError(f"evidence {name} identity mismatch")
        resolved[name] = path.resolve(strict=True)
    validate_v_k_prefix(resolved["ref_k"], resolved["ref_v"], "reference")
    validate_v_k_prefix(resolved["c_k"], resolved["c_v"], "C")
    return resolved


def load_f32(path: Path, count: int, label: str) -> np.ndarray:
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != count or not bool(np.isfinite(values).all()):
        raise RunnerError(f"{label} payload count/finiteness mismatch")
    return values


def validate_v_k_prefix(k_path: Path, v_path: Path, label: str) -> None:
    k = load_f32(k_path, 8 * 576, f"{label} K").reshape(8, 576)
    v = load_f32(v_path, 8 * 512, f"{label} V").reshape(8, 512)
    if k[:, :512].tobytes(order="C") != v.tobytes(order="C"):
        raise RunnerError(f"{label} V is not the byte-exact K prefix")


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    c, r = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if c.shape != r.shape or not c.size or not bool(np.isfinite(c).all() and np.isfinite(r).all()):
        raise RunnerError("metric input mismatch")
    delta = c - r
    return {"nrmse": math.sqrt(float(np.dot(delta, delta)) / max(float(np.dot(r, r)), 1e-30)), "normalized_max": float(np.max(np.abs(delta))) / max(float(np.max(np.abs(r))), 1e-6)}


def judged(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = metrics(candidate, reference)
    result.update({"nrmse_limit": LIMITS[0], "normalized_max_limit": LIMITS[1]})
    result["pass"] = result["nrmse"] <= LIMITS[0] and result["normalized_max"] <= LIMITS[1]
    result["per_token"] = [dict(metrics(candidate.reshape(8, 6144)[i], reference.reshape(8, 6144)[i]), token=i) for i in range(8)]
    return result


def contained(root: Path, reported: str, size: int, digest: str, label: str) -> Path:
    candidate = Path(reported)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RunnerError(f"{label} path escapes diagnostic directory") from exc
    if not resolved.is_file() or resolved.stat().st_size != size or sha256_file(resolved) != digest:
        raise RunnerError(f"{label} output identity mismatch")
    return resolved


def validate_arm_manifest(outputs: Any) -> None:
    if not isinstance(outputs, dict) or set(outputs) != set(ARM_SPECS):
        raise RunnerError("diagnostic arm set mismatch")
    for name, expected in ARM_SPECS.items():
        item = outputs[name]
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256", "q_source", "kv_source", "query_control", "kv_control"}:
            raise RunnerError(f"{name} arm schema mismatch")
        actual = (item["q_source"], item["kv_source"], item["query_control"], item["kv_control"])
        if actual != expected or item["bytes"] != 196_608 or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise RunnerError(f"{name} arm provenance mismatch")


def validate_report(root: Path, model: Path, sources: dict[str, Any], frozen: dict[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        report = json.loads((root / "strat01_rung2c_cross_input.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"diagnostic report missing or malformed: {exc}") from exc
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "v_k_prefix_equal", "matrix", "outputs", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "timing_or_rate_claim"}
    if not isinstance(report, dict) or set(report) != required or report["command"] != "--strat01-rung2c-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise RunnerError("diagnostic report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": EXPECTED_MODEL_BYTES, "sha256": EXPECTED_MODEL_SHA}:
        raise RunnerError("diagnostic model identity mismatch")
    expected_inputs = {}
    for name in ("ref_q", "ref_k", "ref_v", "c_q", "c_k", "c_v"):
        path, digest, size = PINNED_FILES[name]
        expected_inputs[name] = {"path": str(path.resolve()), "bytes": size, "sha256": digest}
    if report["inputs"] != expected_inputs or report["v_k_prefix_equal"] != {"reference": True, "c": True}:
        raise RunnerError("diagnostic input/V-K manifest mismatch")
    matrix = report["matrix"]
    if set(matrix) != {"name", "type", "shape", "offset", "file_offset", "span"} or matrix["name"] != "blk.1.attn_v_b.weight" or matrix["type"] != "Q4_K" or matrix["shape"] != [512, 192, 32] or matrix["span"] != 1_769_472 or not all(isinstance(matrix[x], int) and matrix[x] >= 0 for x in ("offset", "file_offset")):
        raise RunnerError("diagnostic matrix descriptor mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["cross_header"]["sha256"]:
        raise RunnerError("diagnostic source identity mismatch")
    if report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise RunnerError("diagnostic execution/non-rate contract mismatch")
    validate_arm_manifest(report["outputs"])
    values: dict[str, np.ndarray] = {}
    for name, item in report["outputs"].items():
        path = contained(root, item["path"], 196_608, item["sha256"], name)
        values[name] = load_f32(path, 8 * 6144, name)
    if report["outputs"]["c_q__c_kv"]["sha256"] != PINNED_FILES["c_target"][1] or report["outputs"]["c_q__c_kv"]["sha256"] != sha256_file(frozen["c_target"]):
        raise RunnerError("native C/C byte replay mismatch")
    return values, report


def classify(judgments: dict[str, dict[str, Any]]) -> str:
    if not judgments["c_q__c_kv"]["pass"]:
        pass
    else:
        raise RunnerError("native C/C unexpectedly passes frozen reference target")
    if not judgments["ref_q__ref_kv"]["pass"]:
        return "EXACT_REFERENCE_FAILS_ATTENTION_VB_OPERATOR"
    query_fails = not judgments["c_q__ref_kv"]["pass"]
    kv_fails = not judgments["ref_q__c_kv"]["pass"]
    if query_fails and kv_fails:
        return "QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
    if query_fails:
        return "QUERY_RESIDUAL_SUFFICIENT"
    if kv_fails:
        return "KV_RESIDUAL_SUFFICIENT"
    return "JOINT_QUERY_KV_INTERACTION_SUFFICIENT"


def adjudicate(values: dict[str, np.ndarray], report: dict[str, Any], frozen: dict[str, Path]) -> dict[str, Any]:
    reference = load_f32(frozen["ref_target"], 8 * 6144, "reference target")
    c_target = load_f32(frozen["c_target"], 8 * 6144, "C target")
    if values["c_q__c_kv"].tobytes() != c_target.tobytes():
        raise RunnerError("native replay differs from captured C target")
    judgments = {name: judged(values[name], reference) for name in ("c_q__c_kv", "ref_q__ref_kv", "c_q__ref_kv", "ref_q__c_kv")}
    for metric, expected in FROZEN_C_METRICS.items():
        if abs(judgments["c_q__c_kv"][metric] - expected) > 1e-12:
            raise RunnerError(f"frozen C/reference {metric} replay mismatch")
    controls = {name: judged(values[name], reference) for name in ("control_ref_q_token7_negated", "control_ref_kv_rows0_7_swapped")}
    if any(item["pass"] for item in controls.values()):
        raise RunnerError("planted query/KV control did not reject")
    mutation_refusals = {}
    for name in ("ref_q", "ref_k", "ref_v", "c_q", "c_k", "c_v"):
        _, digest, size = PINNED_FILES[name]
        mutated = bytearray(frozen[name].read_bytes()); mutated[len(mutated) // 2] ^= 1
        mutation_refusals[name] = not identity_matches(bytes(mutated), size, digest)
    if not all(mutation_refusals.values()):
        raise RunnerError("mutated-input identity control did not reject")
    swapped = copy.deepcopy(report["outputs"])
    swapped["c_q__ref_kv"], swapped["ref_q__c_kv"] = swapped["ref_q__c_kv"], swapped["c_q__ref_kv"]
    label_swap_rejected = False
    try:
        validate_arm_manifest(swapped)
    except RunnerError:
        label_swap_rejected = True
    if not label_swap_rejected:
        raise RunnerError("mixed-arm label-swap control did not reject")
    status = classify(judgments)
    return {"status": status, "arms_vs_reference": judgments, "controls_vs_reference": controls, "controls": {"native_replay_byte_exact": True, "frozen_metrics_reproduced_within_1e-12": True, "v_k_prefix_byte_exact": True, "mutated_inputs_refused": mutation_refusals, "mixed_label_swap_rejected": label_swap_rejected}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve(); output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = now_utc(), time.perf_counter(); status = "VOID_RUNG2C_ATTENTION_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}; report: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}
    compiler = shutil.which("clang"); binary: Path | None = None; sources: dict[str, Any] = {}; artifact = {"path": str(model), "expected_bytes": EXPECTED_MODEL_BYTES, "expected_sha256": EXPECTED_MODEL_SHA, "bytes": None, "sha256": None}
    try:
        sources = source_inventory()
        if not compiler:
            raise RunnerError("clang is unavailable")
        commands["clang_version"] = run_command([compiler, "--version"], output, "clang_version", 30); require_ok(commands["clang_version"], "clang --version")
        binary = output / "engine_rung2c_cross_input.exe"
        commands["compile"] = run_command([compiler, *COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output, "compile", 600); require_ok(commands["compile"], "C build")
        commands["cross_selftest"] = run_command([str(binary), "--strat01-rung2c-cross-input-selftest"], output, "cross_selftest", 300); require_ok(commands["cross_selftest"], "cross-input selftest")
        commands["legacy_selftest"] = run_command([str(binary), "--kselftest"], output, "legacy_selftest", 300); require_ok(commands["legacy_selftest"], "legacy selftest")
        commands["python_tests"] = run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_rung2c_cross_input"], output, "python_tests", 300); require_ok(commands["python_tests"], "cross-input Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            if not model.is_file() or model.stat().st_size != EXPECTED_MODEL_BYTES or sha256_file(model) != EXPECTED_MODEL_SHA:
                raise RunnerError("accepted model identity mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": EXPECTED_MODEL_SHA})
            frozen = validate_frozen_evidence(); diagnostic_root = output / "diagnostic"; diagnostic_root.mkdir()
            command = [str(binary), "--strat01-rung2c-cross-input", str(model), "--ref-q", str(frozen["ref_q"]), "--ref-k", str(frozen["ref_k"]), "--ref-v", str(frozen["ref_v"]), "--c-q", str(frozen["c_q"]), "--c-k", str(frozen["c_k"]), "--c-v", str(frozen["c_v"]), "--out-dir", str(diagnostic_root)]
            commands["cross_input_diagnostic"] = run_command(command, output, "cross_input_diagnostic", 21600); require_ok(commands["cross_input_diagnostic"], "cross-input diagnostic")
            sources = source_inventory(); values, report = validate_report(diagnostic_root, model, sources, frozen); adjudication = adjudicate(values, report, frozen); status = adjudication["status"]
    except RunnerError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": now_utc(), "seconds": time.perf_counter() - started, "git_head": git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": git_value(["git", "status", "--porcelain"]), "source_hashes": sources, "evidence_hashes": {name: {"path": str(path), "sha256": digest} for name, (path, digest) in EVIDENCE_FILES.items()}, "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_rung2c_cross_input_adjudication_v1", "status": status, "errors": errors, "scope": "captured layer-1 Qcur/Kcur/Vcur through accepted causal-attention plus V-B operator only", "donor_graph_executions": 0, "adjudication": adjudication, "c_report": report, "non_claims": ["repaired or promoted Rung 2C", "output projection, residual, MoE, later layers", "quality, logits, generation, RAM, or rate"], "provenance": provenance}
    manifest = {"schema": "strat01_rung2c_cross_input_run_manifest_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "provenance": provenance}
    write_json(output / "adjudication.json", record); write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    valid = {"APPARATUS_READY_NO_DONOR_EXECUTION", "EXACT_REFERENCE_FAILS_ATTENTION_VB_OPERATOR", "QUERY_RESIDUAL_SUFFICIENT", "KV_RESIDUAL_SUFFICIENT", "QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT", "JOINT_QUERY_KV_INTERACTION_SUFFICIENT"}
    return 0 if status in valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
