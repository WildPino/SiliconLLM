"""Run and independently adjudicate STRAT-01 C-engine numerical rung 1.

This is a correctness experiment, not a throughput benchmark.  The accepted
GGUF is consumed once by the C command; pinned gguf-py independently decodes
the three frozen tensor spans and compares every stored-row result.
"""

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


ROOT = Path(__file__).resolve().parents[3]
ENGINE_SOURCE = ROOT / "benchmarks" / "phase60" / "engine.c"
INSPECTOR_SOURCE = ROOT / "benchmarks" / "phase60" / "strat01_gguf_inspect.h"
RUNG1_SOURCE = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung1.h"
RUNG0_RUNNER = ROOT / "benchmarks" / "donor_adaptation" / "engine" / "run_strat01_engine_rung0.py"
RUNG0_VERIFIER = ROOT / "benchmarks" / "donor_adaptation" / "engine" / "verify_strat01_engine_rung0_reference.py"
RUNG0_TEST = ROOT / "benchmarks" / "donor_adaptation" / "engine" / "test_strat01_engine_rung0.py"
RUNG1_TEST = ROOT / "benchmarks" / "donor_adaptation" / "engine" / "test_strat01_engine_rung1.py"
DEFAULT_MODEL = (
    ROOT / "benchmarks" / "donor_adaptation" / "density" / "results"
    / "strat01_gigachat_q4_97045b2" / "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "donor_adaptation" / "engine" / "results"
    / "strat01_gigachat_engine_rung1_20260921"
)
DEFAULT_GGUF_PY = (
    Path.home() / "AppData" / "Local" / "Temp"
    / "siliconllm-llama-bind-5b335f4" / "gguf-py"
)
LLAMA_COMMIT = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
EXPECTED_SIZE = 6_474_702_976
EXPECTED_SHA256 = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
MAX_NORMALIZED_RESIDUAL = 2e-6
COMPILE_FLAGS = ["-std=c11", "-O3", "-mavx2", "-mfma"]
NON_CLAIMS = [
    "optimized AVX2 kernels",
    "full attention, MLA, MoE, or transformer operator semantics",
    "tokenizer or generation parity",
    "language-model quality through the C engine",
    "RAM fit or accepted-token throughput",
    "any speed or SPEED_LEDGER claim",
]

CELLS: tuple[dict[str, Any], ...] = (
    {
        "cell": "G-R1-Q4",
        "tensor": "blk.0.attn_q.weight",
        "type": "Q4_K",
        "rank": 2,
        "dims": [1536, 6144],
        "offset": 279_677_952,
        "byte_span": 5_308_416,
        "file_offset": 285_780_864,
        "block_values": 256,
        "block_bytes": 144,
        "output_leaf": "strat01_rung1_Q4.f32",
    },
    {
        "cell": "G-R1-Q5",
        "tensor": "blk.0.attn_k_b.weight",
        "type": "Q5_0",
        "rank": 3,
        "dims": [128, 512, 32],
        "offset": 272_421_888,
        "byte_span": 1_441_792,
        "file_offset": 278_524_800,
        "block_values": 32,
        "block_bytes": 22,
        "output_leaf": "strat01_rung1_Q5.f32",
    },
    {
        "cell": "G-R1-Q6",
        "tensor": "blk.0.ffn_down.weight",
        "type": "Q6_K",
        "rank": 2,
        "dims": [8960, 1536],
        "offset": 286_755_840,
        "byte_span": 11_289_600,
        "file_offset": 292_858_752,
        "block_values": 256,
        "block_bytes": 210,
        "output_leaf": "strat01_rung1_Q6.f32",
    },
)


class RunnerError(RuntimeError):
    def __init__(self, message: str, *, status: str = "VOID_APPARATUS") -> None:
        super().__init__(message)
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> dict[str, Any]:
    started_utc = utc_now()
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        elapsed = time.perf_counter() - started
        stdout = getattr(exc, "stdout", "") or ""
        stderr = getattr(exc, "stderr", "") or ""
        stdout_path.write_text(str(stdout), encoding="utf-8")
        stderr_path.write_text(str(stderr), encoding="utf-8")
        return {
            "command": command,
            "cwd": str(cwd),
            "started_utc": started_utc,
            "seconds": elapsed,
            "returncode": None,
            "timed_out_or_spawn_error": str(exc),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout": str(stdout),
            "stderr": str(stderr),
        }
    elapsed = time.perf_counter() - started
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "command": command,
        "cwd": str(cwd),
        "started_utc": started_utc,
        "seconds": elapsed,
        "returncode": result.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def require_command_ok(record: dict[str, Any], label: str, *, status: str = "VOID_APPARATUS") -> None:
    if record.get("returncode") != 0:
        details = record.get("timed_out_or_spawn_error") or record.get("stderr") or ""
        raise RunnerError(f"{label} failed (rc={record.get('returncode')}): {details[-1200:]}", status=status)


def source_inventory() -> dict[str, Any]:
    files = {
        "runner": Path(__file__).resolve(),
        "engine": ENGINE_SOURCE,
        "rung0_inspector": INSPECTOR_SOURCE,
        "rung1_codec": RUNG1_SOURCE,
        "rung0_runner": RUNG0_RUNNER,
        "rung0_verifier": RUNG0_VERIFIER,
        "rung0_tests": RUNG0_TEST,
        "rung1_tests": RUNG1_TEST,
    }
    result: dict[str, Any] = {}
    for name, path in files.items():
        result[name] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "exists": path.is_file(),
        }
    missing = [name for name, item in result.items() if not item["exists"]]
    if missing:
        raise RunnerError(f"required source file(s) missing: {', '.join(missing)}")
    return result


def git_value(command: list[str], *, cwd: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def expected_descriptor(cell: dict[str, Any]) -> dict[str, Any]:
    return {key: cell[key] for key in ("tensor", "type", "rank", "dims", "offset", "byte_span", "file_offset")}


def validate_c_report(report_path: Path, output_dir: Path, model: Path, source_hashes: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"C rung-1 JSON missing or malformed: {exc}", status="FAIL_ENGINE_RUNG1") from exc

    expected_report_keys = {
        "command", "c_state", "input_path", "byte_size", "sha256",
        "engine_source_sha256", "rung1_source_sha256", "reference_revision", "cells",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected_report_keys
        or report.get("command") != "--strat01-gguf-rung1"
    ):
        raise RunnerError("C rung-1 report schema/command mismatch", status="FAIL_ENGINE_RUNG1")
    if report.get("c_state") != "ENGINE_OUTPUT_READY_PENDING_REFERENCE":
        raise RunnerError(f"unexpected C state: {report.get('c_state')!r}", status="FAIL_ENGINE_RUNG1")
    try:
        report_model = Path(report["input_path"]).resolve(strict=True)
        model_resolved = model.resolve(strict=True)
    except (KeyError, OSError, TypeError) as exc:
        raise RunnerError("C report artifact path is missing or invalid", status="FAIL_ENGINE_RUNG1") from exc
    if report_model != model_resolved:
        raise RunnerError("C report input path does not match the frozen artifact path", status="FAIL_ENGINE_RUNG1")
    if report.get("byte_size") != EXPECTED_SIZE or report.get("sha256") != EXPECTED_SHA256:
        raise RunnerError("C report artifact size/hash mismatch", status="FAIL_ENGINE_RUNG1")
    if report.get("engine_source_sha256") != source_hashes["engine"]["sha256"]:
        raise RunnerError("C report engine source hash mismatch", status="FAIL_ENGINE_RUNG1")
    if report.get("rung1_source_sha256") != source_hashes["rung1_codec"]["sha256"]:
        raise RunnerError("C report rung-1 source hash mismatch", status="FAIL_ENGINE_RUNG1")

    raw_cells = report.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != len(CELLS):
        raise RunnerError("C report selected-cell count mismatch", status="FAIL_ENGINE_RUNG1")
    output_paths: list[Path] = []
    for actual, expected in zip(raw_cells, CELLS, strict=True):
        expected_cell_keys = {
            "cell", "tensor", "type", "rank", "dims", "offset", "byte_span", "file_offset",
            "block_count", "decoded_value_count", "decoded_finite_count", "decoded_min", "decoded_max",
            "dequant_f32le_sha256", "output_path", "output_f32le_sha256", "output_finite_count",
            "output_min", "output_max",
        }
        if not isinstance(actual, dict) or set(actual) != expected_cell_keys:
            raise RunnerError("C report cell schema mismatch", status="FAIL_ENGINE_RUNG1")
        if actual.get("cell") != expected["cell"] or actual.get("tensor") != expected["tensor"]:
            raise RunnerError("C report cell order/name mismatch", status="FAIL_ENGINE_RUNG1")
        for key, value in expected_descriptor(expected).items():
            if actual.get(key) != value:
                raise RunnerError(f"C descriptor mismatch for {expected['cell']}: {key}", status="FAIL_ENGINE_RUNG1")

        elements = math.prod(expected["dims"])
        row_length = expected["dims"][0]
        rows = elements // row_length
        expected_blocks = elements // expected["block_values"]
        count_fields = {
            "block_count": expected_blocks,
            "decoded_value_count": elements,
            "decoded_finite_count": elements,
            "output_finite_count": rows,
        }
        for key, value in count_fields.items():
            if actual.get(key) != value:
                raise RunnerError(f"C {key} mismatch for {expected['cell']}", status="FAIL_ENGINE_RUNG1")
        for key in ("dequant_f32le_sha256", "output_f32le_sha256"):
            digest = actual.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise RunnerError(f"C {key} malformed for {expected['cell']}", status="FAIL_ENGINE_RUNG1")
        for key in ("decoded_min", "decoded_max", "output_min", "output_max"):
            if not isinstance(actual.get(key), (int, float)) or not math.isfinite(actual[key]):
                raise RunnerError(f"C {key} non-finite/missing for {expected['cell']}", status="FAIL_ENGINE_RUNG1")

        expected_path = (output_dir / expected["output_leaf"]).resolve()
        try:
            reported_path = Path(actual["output_path"]).resolve(strict=True)
        except (KeyError, OSError, TypeError) as exc:
            raise RunnerError(f"C output path missing for {expected['cell']}", status="FAIL_ENGINE_RUNG1") from exc
        if reported_path != expected_path or not reported_path.is_file():
            raise RunnerError(f"C output path mismatch for {expected['cell']}", status="FAIL_ENGINE_RUNG1")
        if reported_path.stat().st_size != rows * 4:
            raise RunnerError(f"C output byte count mismatch for {expected['cell']}", status="FAIL_ENGINE_RUNG1")
        if sha256_file(reported_path) != actual["output_f32le_sha256"]:
            raise RunnerError(f"C output hash mismatch for {expected['cell']}", status="FAIL_ENGINE_RUNG1")
        output_paths.append(reported_path)
    return report, output_paths


def load_pinned_reference(gguf_py: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    gguf_py = gguf_py.resolve(strict=True)
    checkout_root = gguf_py.parent
    if not (gguf_py / "gguf" / "gguf_reader.py").is_file() or not (gguf_py / "gguf" / "quants.py").is_file():
        raise RunnerError(f"pinned gguf-py source is incomplete: {gguf_py}")
    head = git_value(["git", "rev-parse", "HEAD"], cwd=checkout_root)
    if head != LLAMA_COMMIT:
        raise RunnerError(f"llama.cpp checkout revision mismatch: {head!r}")
    dirty = git_value(["git", "status", "--porcelain", "--", "gguf-py"], cwd=checkout_root)
    if dirty:
        raise RunnerError("pinned gguf-py source tree has local modifications")
    sys.path.insert(0, str(gguf_py))
    try:
        import numpy as np
        import gguf
        from gguf import GGUFReader
        from gguf.quants import dequantize
    except Exception as exc:  # import errors make the apparatus unavailable
        raise RunnerError(f"cannot import pinned gguf-py/NumPy: {exc}") from exc
    versions = {
        "llama_cpp_commit": head,
        "gguf_py_path": str(gguf_py),
        "gguf_module_path": str(Path(gguf.__file__).resolve()),
        "numpy_version": np.__version__,
        "python_version": sys.version,
    }
    return np, GGUFReader, dequantize, versions


def reference_adjudicate(
    *,
    np: Any,
    GGUFReader: Any,
    dequantize: Any,
    model: Path,
    output_paths: list[Path],
    c_report: dict[str, Any],
    gguf_py: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    reader = GGUFReader(model, "r")
    if getattr(reader, "byte_order", None) != "I":
        raise RunnerError("pinned GGUFReader reports non-native byte order", status="FAIL_ENGINE_RUNG1")
    if int(reader.data_offset) != 6_102_912:
        raise RunnerError("pinned GGUFReader data offset mismatch", status="FAIL_ENGINE_RUNG1")
    tensors_by_name: dict[str, list[Any]] = {}
    for tensor in reader.tensors:
        tensors_by_name.setdefault(tensor.name, []).append(tensor)

    cell_results: list[dict[str, Any]] = []
    gate_errors: list[str] = []
    c_cells = c_report["cells"]
    for spec, c_cell, c_output_path in zip(CELLS, c_cells, output_paths, strict=True):
        matches = tensors_by_name.get(spec["tensor"], [])
        if len(matches) != 1:
            raise RunnerError(f"GGUFReader tensor multiplicity mismatch: {spec['tensor']}", status="FAIL_ENGINE_RUNG1")
        tensor = matches[0]
        actual_desc = {
            "tensor": tensor.name,
            "type": tensor.tensor_type.name,
            "rank": len(tensor.shape),
            "dims": [int(value) for value in tensor.shape],
            "offset": int(tensor.data_offset) - int(reader.data_offset),
            "byte_span": int(tensor.n_bytes),
            "file_offset": int(tensor.data_offset),
        }
        frozen_desc = expected_descriptor(spec)
        if actual_desc != frozen_desc:
            raise RunnerError(f"pinned GGUFReader descriptor mismatch for {spec['cell']}: {actual_desc}", status="FAIL_ENGINE_RUNG1")
        raw = tensor.data
        if int(raw.nbytes) != spec["byte_span"]:
            raise RunnerError(f"GGUFReader raw byte count mismatch for {spec['cell']}", status="FAIL_ENGINE_RUNG1")

        weights = np.asarray(dequantize(raw, tensor.tensor_type), dtype=np.float32)
        elements = math.prod(spec["dims"])
        rows = elements // spec["dims"][0]
        row_length = spec["dims"][0]
        if weights.size != elements or weights.size != int(tensor.n_elements):
            raise RunnerError(f"reference decoded-value count mismatch for {spec['cell']}", status="FAIL_ENGINE_RUNG1")
        weights = weights.reshape((rows, row_length), order="C")
        if not bool(np.isfinite(weights).all()):
            raise RunnerError(f"non-finite reference decoded weights for {spec['cell']}", status="FAIL_ENGINE_RUNG1")
        canonical_weights = np.asarray(weights, dtype=np.dtype("<f4"), order="C")
        reference_dequant_hash = hashlib.sha256(canonical_weights.tobytes(order="C")).hexdigest()

        indices = np.arange(row_length, dtype=np.int64)
        inputs = (((indices * 73 + 19) % 257) - 128).astype(np.float32) / np.float32(128.0)
        reference_outputs = np.empty(rows, dtype=np.float32)
        denominator = np.empty(rows, dtype=np.float64)
        chunk_rows = max(1, min(128, 1_000_000 // row_length))
        for first in range(0, rows, chunk_rows):
            last = min(rows, first + chunk_rows)
            row_weights = weights[first:last]
            products = np.multiply(row_weights, inputs, dtype=np.float32)
            reference_outputs[first:last] = np.sum(products, axis=1, dtype=np.float32)
            abs_products = np.abs(row_weights.astype(np.float64) * inputs.astype(np.float64))
            denominator[first:last] = 1.0 + np.sum(abs_products, axis=1, dtype=np.float64)
        if not bool(np.isfinite(reference_outputs).all()):
            raise RunnerError(f"non-finite NumPy row outputs for {spec['cell']}", status="FAIL_ENGINE_RUNG1")

        c_outputs = np.fromfile(c_output_path, dtype=np.dtype("<f4"))
        if c_outputs.size != rows or not bool(np.isfinite(c_outputs).all()):
            raise RunnerError(f"C output count/finiteness mismatch for {spec['cell']}", status="FAIL_ENGINE_RUNG1")
        difference = c_outputs.astype(np.float64) - reference_outputs.astype(np.float64)
        absolute = np.abs(difference)
        normalized = absolute / denominator
        worst_row = int(np.argmax(normalized))
        max_abs = float(np.max(absolute))
        rmse = float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
        max_normalized = float(normalized[worst_row])
        reference_output_bytes = np.asarray(reference_outputs, dtype=np.dtype("<f4"), order="C").tobytes(order="C")
        reference_output_hash = hashlib.sha256(reference_output_bytes).hexdigest()
        c_output_hash = sha256_file(c_output_path)
        digest_match = reference_dequant_hash == c_cell["dequant_f32le_sha256"]
        residual_pass = max_normalized <= MAX_NORMALIZED_RESIDUAL
        cell_pass = digest_match and residual_pass
        if not digest_match:
            gate_errors.append(f"{spec['cell']}: full-stream dequant SHA-256 mismatch")
        if not residual_pass:
            gate_errors.append(f"{spec['cell']}: max normalized residual {max_normalized:.9g} exceeds {MAX_NORMALIZED_RESIDUAL}")
        cell_results.append(
            {
                "cell": spec["cell"],
                "tensor": spec["tensor"],
                "type": spec["type"],
                "descriptor": actual_desc,
                "counts": {
                    "reference_blocks": int(elements // spec["block_values"]),
                    "c_blocks": int(c_cell["block_count"]),
                    "reference_decoded_values": int(weights.size),
                    "c_decoded_values": int(c_cell["decoded_value_count"]),
                    "reference_rows": int(reference_outputs.size),
                    "c_rows": int(c_outputs.size),
                },
                "dequant_sha256": {
                    "reference_f32le": reference_dequant_hash,
                    "c_f32le": c_cell["dequant_f32le_sha256"],
                    "exact_match": digest_match,
                },
                "row_matvec": {
                    "input_formula": "(((i * 73 + 19) mod 257) - 128) / 128.0f",
                    "max_absolute_error": max_abs,
                    "rmse": rmse,
                    "max_normalized_residual": max_normalized,
                    "normalized_residual_limit": MAX_NORMALIZED_RESIDUAL,
                    "worst_row": worst_row,
                    "worst_row_c": float(c_outputs[worst_row]),
                    "worst_row_reference": float(reference_outputs[worst_row]),
                    "worst_row_denominator": float(denominator[worst_row]),
                    "c_output_sha256": c_output_hash,
                    "c_reported_output_sha256": c_cell["output_f32le_sha256"],
                    "reference_output_f32le_sha256": reference_output_hash,
                    "c_output_hash_matches_report": c_output_hash == c_cell["output_f32le_sha256"],
                },
                "gate_pass": cell_pass,
            }
        )

    return {
        "status": "PASS_REFERENCE_COMPARE" if not gate_errors else "FAIL_REFERENCE_COMPARE",
        "gguf_reader_revision": LLAMA_COMMIT,
        "gguf_py_path": str(gguf_py.resolve()),
        "reader_data_offset": int(reader.data_offset),
        "cells": cell_results,
        "errors": gate_errors,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = DEFAULT_MODEL.resolve()
    output = args.output_dir.resolve()
    gguf_py = DEFAULT_GGUF_PY.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started_utc = utc_now()
    total_started = time.perf_counter()
    status = "VOID_APPARATUS"
    errors: list[str] = []
    timings: dict[str, Any] = {}
    commands: dict[str, Any] = {}
    c_report: dict[str, Any] | None = None
    reference_result: dict[str, Any] = {"status": "NOT_RUN", "cells": [], "errors": []}
    source_hashes: dict[str, Any] = {}
    binary: Path | None = None
    artifact_record: dict[str, Any] = {
        "path": str(model),
        "expected_bytes": EXPECTED_SIZE,
        "expected_sha256": EXPECTED_SHA256,
    }
    environment: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "clang_path": shutil.which("clang"),
        "llama_cpp_commit_expected": LLAMA_COMMIT,
    }

    try:
        source_hashes = source_inventory()
        if shutil.which("clang") is None:
            raise RunnerError("clang not found")
        if not model.is_file() or model.stat().st_size != EXPECTED_SIZE:
            raise RunnerError("frozen accepted artifact is missing or has the wrong byte size")
        compiler = run_command(
            [shutil.which("clang") or "clang", "--version"],
            cwd=ROOT,
            stdout_path=output / "clang_version.stdout.log",
            stderr_path=output / "clang_version.stderr.log",
            timeout=30,
        )
        require_command_ok(compiler, "clang --version")
        commands["clang_version"] = compiler
        environment["clang_version"] = compiler["stdout"].splitlines()[0] if compiler["stdout"] else "unknown"
        environment["numpy_version"] = None

        np, GGUFReader, dequantize, ref_versions = load_pinned_reference(gguf_py)
        environment.update(ref_versions)

        binary = output / "engine_rung1.exe"
        compile_command = [
            shutil.which("clang") or "clang",
            *COMPILE_FLAGS,
            str(ENGINE_SOURCE),
            "-o",
            str(binary),
            "-lm",
        ]
        compile_record = run_command(
            compile_command,
            cwd=ROOT,
            stdout_path=output / "compile.stdout.log",
            stderr_path=output / "compile.stderr.log",
            timeout=300,
        )
        commands["compile"] = compile_record
        require_command_ok(compile_record, "registered clang build")
        if not binary.is_file():
            raise RunnerError("registered build returned success but produced no binary")
        timings["compile_seconds"] = compile_record["seconds"]

        for label, module in (("rung0_unittest", "benchmarks.donor_adaptation.engine.test_strat01_engine_rung0"),
                              ("rung1_unittest", "benchmarks.donor_adaptation.engine.test_strat01_engine_rung1")):
            command = [sys.executable, "-B", "-m", "unittest", "-v", module]
            record = run_command(
                command,
                cwd=ROOT,
                stdout_path=output / f"{label}.stdout.log",
                stderr_path=output / f"{label}.stderr.log",
                timeout=1800,
            )
            commands[label] = record
            require_command_ok(record, label)
            if "OK" not in record["stderr"]:
                raise RunnerError(f"{label} did not report the unittest success summary")

        kselftest_command = [str(binary), "--kselftest"]
        kselftest = run_command(
            kselftest_command,
            cwd=ROOT,
            stdout_path=output / "kselftest.stdout.log",
            stderr_path=output / "kselftest.stderr.log",
            timeout=300,
        )
        commands["legacy_kselftest"] = kselftest
        require_command_ok(kselftest, "legacy --kselftest")
        if "PASS" not in (kselftest["stdout"] + kselftest["stderr"]).upper():
            raise RunnerError("legacy --kselftest returned zero without a PASS marker")

        codec_selftest_command = [str(binary), "--strat01-gguf-rung1-selftest"]
        codec_selftest = run_command(
            codec_selftest_command,
            cwd=ROOT,
            stdout_path=output / "rung1_codec_selftest.stdout.log",
            stderr_path=output / "rung1_codec_selftest.stderr.log",
            timeout=300,
        )
        commands["rung1_codec_selftest"] = codec_selftest
        require_command_ok(codec_selftest, "rung-1 codec selftest")
        if "PASS" not in (codec_selftest["stdout"] + codec_selftest["stderr"]).upper():
            raise RunnerError("rung-1 codec selftest returned zero without a PASS marker")

        artifact_command = [str(binary), "--strat01-gguf-rung1", str(model), "--out-dir", str(output)]
        artifact_run = run_command(
            artifact_command,
            cwd=ROOT,
            stdout_path=output / "rung1_c.stdout.log",
            stderr_path=output / "rung1_c.stderr.log",
            timeout=14_400,
        )
        commands["accepted_artifact_c_run"] = artifact_run
        timings["accepted_artifact_c_seconds"] = artifact_run["seconds"]
        if artifact_run.get("returncode") != 0:
            detail = artifact_run.get("stderr", "")
            lowered = detail.lower()
            gate_failure = any(
                token in lowered
                for token in (
                    "descriptor mismatch", "dimensions mismatch", "identity mismatch",
                    "frozen rung-1", "decode failure", "non-finite", "count mismatch",
                    "unsupported rung-1 tensor type", "row block divisibility",
                )
            )
            raise RunnerError(
                f"C accepted-artifact rung-1 command failed: {detail[-1200:]}",
                status="FAIL_ENGINE_RUNG1" if gate_failure else "VOID_APPARATUS",
            )

        source_hashes = source_inventory()
        c_report, output_paths = validate_c_report(output / "strat01_rung1.json", output, model, source_hashes)
        artifact_record.update({"bytes": c_report["byte_size"], "sha256": c_report["sha256"]})
        if c_report.get("reference_revision") != f"llama.cpp {LLAMA_COMMIT} (gguf-py/gguf/quants.py)":
            raise RunnerError("C report reference revision mismatch", status="FAIL_ENGINE_RUNG1")

        ref_started = time.perf_counter()
        try:
            reference_result = reference_adjudicate(
                np=np,
                GGUFReader=GGUFReader,
                dequantize=dequantize,
                model=model,
                output_paths=output_paths,
                c_report=c_report,
                gguf_py=gguf_py,
            )
        except RunnerError:
            raise
        timings["reference_adjudication_seconds"] = time.perf_counter() - ref_started
        commands["reference_adjudication"] = {
            "execution": "in-process pinned gguf-py and NumPy adjudication",
            "seconds": timings["reference_adjudication_seconds"],
            "returncode": 0,
        }
        if reference_result["status"] != "PASS_REFERENCE_COMPARE":
            raise RunnerError("one or more independent reference gates failed", status="FAIL_ENGINE_RUNG1")
        status = "PASS_ENGINE_RUNG1"
    except RunnerError as exc:
        status = exc.status
        errors.append(str(exc))
    except Exception as exc:  # persist a fail-closed record for unexpected apparatus errors
        status = "VOID_APPARATUS"
        errors.append(f"unexpected {type(exc).__name__}: {exc}")

    source_hashes = source_hashes or {
        key: {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None, "exists": path.is_file()}
        for key, path in {
            "runner": Path(__file__).resolve(),
            "engine": ENGINE_SOURCE,
            "rung0_inspector": INSPECTOR_SOURCE,
            "rung1_codec": RUNG1_SOURCE,
            "rung0_runner": RUNG0_RUNNER,
            "rung0_verifier": RUNG0_VERIFIER,
            "rung0_tests": RUNG0_TEST,
            "rung1_tests": RUNG1_TEST,
        }.items()
    }
    binary_record = {
        "path": str(binary) if binary else None,
        "sha256": sha256_file(binary) if binary and binary.is_file() else None,
        "bytes": binary.stat().st_size if binary and binary.is_file() else None,
    }
    if binary_record["sha256"] is None and status == "PASS_ENGINE_RUNG1":
        status = "VOID_APPARATUS"
        errors.append("binary disappeared before manifest finalization")

    finished_utc = utc_now()
    timings["total_seconds"] = time.perf_counter() - total_started
    provenance = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "git_head": git_value(["git", "rev-parse", "HEAD"]),
        "source_hashes": source_hashes,
        "binary": binary_record,
        "artifact": artifact_record,
        "environment": environment,
        "timings_seconds": timings,
        "commands": commands,
        "non_claims": NON_CLAIMS,
    }
    reference_document = {
        "schema": "strat01_gigachat_engine_rung1_reference_compare_v1",
        "status": reference_result.get("status", "NOT_RUN"),
        "combined_status": status,
        "reference_method": "pinned llama.cpp gguf-py dequantize + independently generated NumPy float32 row sums",
        "reference_result": reference_result,
        "errors": errors + list(reference_result.get("errors", [])),
        "provenance": provenance,
    }
    manifest = {
        "schema": "strat01_gigachat_engine_rung1_run_manifest_v1",
        "status": status,
        "gate_summary": {
            "registered_build_and_regressions": all(
                commands.get(key, {}).get("returncode") == 0
                for key in (
                    "compile", "rung0_unittest", "rung1_unittest",
                    "legacy_kselftest", "rung1_codec_selftest",
                )
            ),
            "c_output_ready": c_report is not None,
            "all_three_reference_cells_pass": reference_result.get("status") == "PASS_REFERENCE_COMPARE",
        },
        "errors": errors,
        "scope": "whole-stream quant decode and scalar stored-row matvec parity for three frozen tensors",
        "provenance": provenance,
    }
    write_json(output / "reference_compare.json", reference_document)
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "PASS_ENGINE_RUNG1" else 2


if __name__ == "__main__":
    raise SystemExit(main())
