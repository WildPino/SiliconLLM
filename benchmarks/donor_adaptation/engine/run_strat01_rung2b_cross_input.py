#!/usr/bin/env python3
"""Execute and adjudicate the frozen Rung-2B cross-input Q4_K diagnostic."""
from __future__ import annotations

import argparse
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
RUNG2B = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2b.h"
CROSS_HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2b_cross_input.h"
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_RUNG2B_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260921.md"
ACCEPTED_RUN = HERE / "results" / "strat01_gigachat_engine_rung2b_repair2_20260921"
DEFAULT_MODEL = ROOT / "benchmarks" / "donor_adaptation" / "density" / "results" / "strat01_gigachat_q4_97045b2" / "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_rung2b_cross_input_20260921"
REF_INPUT = ACCEPTED_RUN / "pinned_reference" / "prefill8" / "ffn_norm-0.full.f32le"
C_INPUT = ACCEPTED_RUN / "c_engine" / "prefill8_ffn_norm-0.f32"
REF_UP = ACCEPTED_RUN / "pinned_reference" / "prefill8" / "ffn_up-0.full.f32le"
REF_GATE = ACCEPTED_RUN / "pinned_reference" / "prefill8" / "ffn_gate-0.full.f32le"
C_UP = ACCEPTED_RUN / "c_engine" / "prefill8_ffn_up-0.f32"
C_GATE = ACCEPTED_RUN / "c_engine" / "prefill8_ffn_gate-0.f32"
EXPECTED_MODEL_BYTES = 6_474_702_976
EXPECTED_MODEL_SHA = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
INPUT_BYTES = 49_152
OUTPUT_BYTES = 286_720
Q8_BYTES = 14_016
INPUT_COUNT = 8 * 1536
OUTPUT_COUNT = 8 * 8960
LIMITS = (2e-3, 1e-2)
COMPILE_FLAGS = ["-std=c11", "-O3", "-mavx2", "-mfma"]
PINNED = {
    "reference_input": (REF_INPUT, "7bbfd9c2f0ed17c12a429b5dff83f75d47a45dd97008ad88251a7ec8d64dc588", INPUT_BYTES),
    "c_input": (C_INPUT, "ea5d60791c4404fbb98e7839fa73ff8112c54abba6060a7fbc3d5e83b951e3e0", INPUT_BYTES),
    "reference_up": (REF_UP, "2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4", OUTPUT_BYTES),
    "reference_gate": (REF_GATE, "5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a", OUTPUT_BYTES),
    "c_up": (C_UP, "7ebb2858834cd390e6563333b4e0da3f4f95098180ea64e1c0be9be79c4e74b3", OUTPUT_BYTES),
    "c_gate": (C_GATE, "83d26be14a2349a74f25bd5c56fd3499fe62e20daf41ed7fdedb0ea6a70c354f", OUTPUT_BYTES),
}
FROZEN_C_METRICS = {
    "up": {"nrmse": 0.0030612619849613985, "normalized_max": 0.0017181935481242877},
    "gate": {"nrmse": 0.0026914051074363363, "normalized_max": 0.001061259056391299},
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
    paths = {"runner": Path(__file__).resolve(), "engine": ENGINE, "rung2a": RUNG2A, "rung2b": RUNG2B, "cross_header": CROSS_HEADER, "protocol": PROTOCOL}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunnerError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def validate_pinned() -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, (path, digest, size) in PINNED.items():
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise RunnerError(f"pinned {name} identity mismatch")
        resolved[name] = path.resolve(strict=True)
    return resolved


def load_f32(path: Path, expected_sha: str, count: int, label: str) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * 4 or sha256_file(path) != expected_sha:
        raise RunnerError(f"{label} payload identity mismatch")
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != count or not bool(np.isfinite(values).all()):
        raise RunnerError(f"{label} payload count/finiteness mismatch")
    return values


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
        raise RunnerError(f"{label} payload identity mismatch")
    return resolved


def validate_report(root: Path, model: Path, sources: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        report = json.loads((root / "strat01_rung2b_cross_input.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"diagnostic report missing or malformed: {exc}") from exc
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "matrices", "outputs", "q8", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "timing_or_rate_claim"}
    if not isinstance(report, dict) or set(report) != required or report["command"] != "--strat01-rung2b-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise RunnerError("diagnostic report schema/state mismatch")
    model_record = report["model"]
    if Path(model_record.get("path", "")).resolve(strict=True) != model.resolve(strict=True) or model_record.get("bytes") != EXPECTED_MODEL_BYTES or model_record.get("sha256") != EXPECTED_MODEL_SHA:
        raise RunnerError("diagnostic model identity mismatch")
    if report["inputs"] != {"reference": {"path": str(REF_INPUT.resolve()), "bytes": INPUT_BYTES, "sha256": PINNED["reference_input"][1]}, "c_engine": {"path": str(C_INPUT.resolve()), "bytes": INPUT_BYTES, "sha256": PINNED["c_input"][1]}}:
        raise RunnerError("diagnostic input manifest mismatch")
    expected_matrices = [{"name": "blk.0.ffn_up.weight", "offset": 305793024, "file_offset": 311895936}, {"name": "blk.0.ffn_gate.weight", "offset": 298045440, "file_offset": 304148352}]
    if report["matrices"] != expected_matrices or report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["cross_header"]["sha256"]:
        raise RunnerError("diagnostic matrix/source identity mismatch")
    if report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise RunnerError("diagnostic execution/non-rate contract mismatch")
    values: dict[str, np.ndarray] = {}
    for key, item in report["outputs"].items():
        if key not in {"reference_input_up", "reference_input_gate", "c_input_up", "c_input_gate"} or set(item) != {"path", "bytes", "sha256"} or item["bytes"] != OUTPUT_BYTES:
            raise RunnerError("diagnostic output schema mismatch")
        path = contained(root, item["path"], OUTPUT_BYTES, item["sha256"], key)
        values[key] = load_f32(path, item["sha256"], OUTPUT_COUNT, key)
    if set(values) != {"reference_input_up", "reference_input_gate", "c_input_up", "c_input_gate"}:
        raise RunnerError("diagnostic output set mismatch")
    q8 = report["q8"]
    if set(q8) != {"reference", "c_engine", "changed_blocks", "total_blocks", "changed_bytes", "total_bytes"} or q8["total_blocks"] != 48 or q8["total_bytes"] != Q8_BYTES:
        raise RunnerError("diagnostic Q8 schema mismatch")
    q8_paths = []
    for key in ("reference", "c_engine"):
        item = q8[key]
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != Q8_BYTES:
            raise RunnerError("diagnostic Q8 payload schema mismatch")
        q8_paths.append(contained(root, item["path"], Q8_BYTES, item["sha256"], f"q8/{key}"))
    a, b = q8_paths[0].read_bytes(), q8_paths[1].read_bytes()
    changed_bytes = sum(x != y for x, y in zip(a, b))
    changed_blocks = sum(a[i:i+292] != b[i:i+292] for i in range(0, Q8_BYTES, 292))
    if q8["changed_bytes"] != changed_bytes or q8["changed_blocks"] != changed_blocks:
        raise RunnerError("diagnostic Q8 census mismatch")
    return values, report


def adjudicate(values: dict[str, np.ndarray], pinned: dict[str, Path], report: dict[str, Any]) -> dict[str, Any]:
    ref_input = load_f32(pinned["reference_input"], PINNED["reference_input"][1], INPUT_COUNT, "reference input")
    c_input = load_f32(pinned["c_input"], PINNED["c_input"][1], INPUT_COUNT, "C input")
    ref_up = load_f32(pinned["reference_up"], PINNED["reference_up"][1], OUTPUT_COUNT, "reference up")
    ref_gate = load_f32(pinned["reference_gate"], PINNED["reference_gate"][1], OUTPUT_COUNT, "reference gate")
    primary = {"up": judged(values["reference_input_up"], ref_up), "gate": judged(values["reference_input_gate"], ref_gate)}
    frozen_replay = {"up": judged(values["c_input_up"], ref_up), "gate": judged(values["c_input_gate"], ref_gate)}
    for name in ("up", "gate"):
        for metric in ("nrmse", "normalized_max"):
            if abs(frozen_replay[name][metric] - FROZEN_C_METRICS[name][metric]) > 1e-12:
                raise RunnerError(f"frozen {name} metric replay mismatch")
    output_items = report["outputs"]
    if output_items["c_input_up"]["sha256"] != PINNED["c_up"][1] or output_items["c_input_gate"]["sha256"] != PINNED["c_gate"][1]:
        raise RunnerError("accepted C output byte replay mismatch")
    swapped = {"up_as_gate": judged(values["reference_input_up"], ref_gate), "gate_as_up": judged(values["reference_input_gate"], ref_up)}
    if swapped["up_as_gate"]["pass"] or swapped["gate_as_up"]["pass"]:
        raise RunnerError("gate/up swapped-target control did not reject")
    ref_bytes = pinned["reference_input"].read_bytes()
    mutated = bytearray(ref_bytes); mutated[len(mutated) // 2] ^= 1
    mutation_refused = not identity_matches(bytes(mutated), INPUT_BYTES, PINNED["reference_input"][1])
    if not mutation_refused:
        raise RunnerError("mutated-input identity control did not reject")
    status = "REFERENCE_INPUT_PASSES_Q4K_PATH" if all(item["pass"] for item in primary.values()) else "REFERENCE_INPUT_FAILS_Q4K_PATH"
    return {
        "status": status,
        "primary": primary,
        "accepted_c_input_replay": frozen_replay,
        "input_difference": metrics(c_input, ref_input),
        "c_path_output_difference": {"up": metrics(values["c_input_up"], values["reference_input_up"]), "gate": metrics(values["c_input_gate"], values["reference_input_gate"])},
        "q8_census": {key: report["q8"][key] for key in ("changed_blocks", "total_blocks", "changed_bytes", "total_bytes")},
        "controls": {"accepted_c_outputs_byte_exact": True, "frozen_metrics_reproduced_within_1e-12": True, "mutated_input_refused": mutation_refused, "swapped_targets": swapped},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve(); output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = now_utc(), time.perf_counter(); status = "VOID_CROSS_INPUT_DIAGNOSTIC"; errors: list[str] = []; commands: dict[str, Any] = {}; report: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}
    compiler = shutil.which("clang"); binary: Path | None = None; sources: dict[str, Any] = {}; artifact = {"path": str(model), "expected_bytes": EXPECTED_MODEL_BYTES, "expected_sha256": EXPECTED_MODEL_SHA, "bytes": None, "sha256": None}
    try:
        sources = source_inventory()
        if not compiler:
            raise RunnerError("clang is unavailable")
        commands["clang_version"] = run_command([compiler, "--version"], output, "clang_version", 30); require_ok(commands["clang_version"], "clang --version")
        binary = output / "engine_cross_input.exe"
        commands["compile"] = run_command([compiler, *COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output, "compile", 600); require_ok(commands["compile"], "C build")
        commands["cross_selftest"] = run_command([str(binary), "--strat01-rung2b-cross-input-selftest"], output, "cross_selftest", 300); require_ok(commands["cross_selftest"], "cross-input selftest")
        commands["legacy_selftest"] = run_command([str(binary), "--kselftest"], output, "legacy_selftest", 300); require_ok(commands["legacy_selftest"], "legacy selftest")
        commands["python_tests"] = run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_rung2b_cross_input"], output, "python_tests", 300); require_ok(commands["python_tests"], "cross-input Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            if not model.is_file() or model.stat().st_size != EXPECTED_MODEL_BYTES or sha256_file(model) != EXPECTED_MODEL_SHA:
                raise RunnerError("accepted model identity mismatch")
            artifact.update({"bytes": model.stat().st_size, "sha256": EXPECTED_MODEL_SHA})
            pinned = validate_pinned(); diagnostic_root = output / "diagnostic"; diagnostic_root.mkdir()
            commands["cross_input_diagnostic"] = run_command([str(binary), "--strat01-rung2b-cross-input", str(model), "--reference-input", str(pinned["reference_input"]), "--c-input", str(pinned["c_input"]), "--out-dir", str(diagnostic_root)], output, "cross_input_diagnostic", 21600); require_ok(commands["cross_input_diagnostic"], "cross-input diagnostic")
            sources = source_inventory(); values, report = validate_report(diagnostic_root, model, sources); adjudication = adjudicate(values, pinned, report); status = adjudication["status"]
    except RunnerError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": now_utc(), "seconds": time.perf_counter() - started, "git_head": git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": git_value(["git", "status", "--porcelain"]), "source_hashes": sources, "artifact": artifact, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_rung2b_cross_input_adjudication_v1", "status": status, "errors": errors, "scope": "captured ffn_norm cross-input through accepted block-0 Q4_K up/gate only", "donor_graph_executions": 0, "adjudication": adjudication, "c_report": report, "non_claims": ["repaired Rung 2B", "Rung 2C or MoE", "quality, logits, generation, RAM, or rate"], "provenance": provenance}
    manifest = {"schema": "strat01_rung2b_cross_input_run_manifest_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "provenance": provenance}
    write_json(output / "adjudication.json", record); write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION", "REFERENCE_INPUT_PASSES_Q4K_PATH", "REFERENCE_INPUT_FAILS_Q4K_PATH"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

