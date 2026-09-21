#!/usr/bin/env python3
"""Run the frozen upstream RMSNorm propagation diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[3]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2a as r2a
from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2b as r2b


ROOT = r2a.ROOT
HERE = r2a.HERE
ENGINE = r2a.ENGINE
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_upstream_rmsnorm_diag.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_UPSTREAM_RMSNORM_PROPAGATION_DIAGNOSTIC_PROTOCOL_20260921.md"
DEFAULT_MODEL = r2a.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_upstream_rmsnorm_diag_20260921"
ATTENTION_RUN = HERE / "results/strat01_gigachat_engine_attention_vb_repair_20260921"
RUNG2A_SOURCE_RUN = HERE / "results/strat01_gigachat_engine_rung2a_repair2_20260921"
RUNG2B_SOURCE_RUN = HERE / "results/strat01_gigachat_engine_rung2b_repair2_20260921"
RMS_SOURCE_RUN = HERE / "results/strat01_gigachat_engine_rung2b_rmsnorm_diag_20260921"
RUNG2A_REFERENCE = RUNG2A_SOURCE_RUN / "pinned_reference"
RUNG2B_REFERENCE = RUNG2B_SOURCE_RUN / "pinned_reference"
BASELINE_ATTN = ATTENTION_RUN / "c_engine/prefill8_attn_norm-0.f32"
BASELINE_FFN = RUNG2B_SOURCE_RUN / "c_engine/prefill8_ffn_norm-0.f32"
BASELINE_ATTN_SHA = "f746d41ff1d909d70d09b241c2a3f2d66fd837b51dbeec5ca95ba618d3456d9e"
BASELINE_FFN_SHA = "ea5d60791c4404fbb98e7839fa73ff8112c54abba6060a7fbc3d5e83b951e3e0"
PRIOR_HASHES = {
    ATTENTION_RUN / "adjudication.json": "e99f0b3e05286b02546a9d9c83f1a8745ba486d0c05087db313f23f9d87f0627",
    RUNG2A_SOURCE_RUN / "run_manifest.json": "0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f",
    RUNG2B_SOURCE_RUN / "run_manifest.json": "5fa1ca6317c3afb466df937f4a9f2e622c38f7d3787d84d2874b2e8afb6ca0b7",
    RMS_SOURCE_RUN / "adjudication.json": "88bb7f63cd832328c1a4043e457f905304ef520a6144093811d5ebbbe98e8960",
}
ARMS = r2a.ARMS
FFN_NAMES = ("ffn_norm-0", "ffn_up-0", "ffn_gate-0")
FFN_SHAPES = {"ffn_norm-0": [1536, 8], "ffn_up-0": [8960, 8], "ffn_gate-0": [8960, 8]}
CONTINUITY_LIMITS = (1e-7, 1e-6)
Q8_BLOCK_BYTES = 292


class DiagnosticError(RuntimeError):
    pass


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(), "engine": ENGINE,
        "rung2a_header": r2a.RUNG2A_HEADER, "rung2b_header": r2b.RUNG2B_HEADER,
        "diagnostic_header": HEADER, "protocol": PROTOCOL,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": r2a.sha256_file(path)} for name, path in paths.items()}


def validate_prior_bindings() -> None:
    for path, digest in PRIOR_HASHES.items():
        if not path.is_file() or r2a.sha256_file(path) != digest:
            raise DiagnosticError(f"immutable prior binding mismatch: {path}")
    for path, digest, size in ((BASELINE_ATTN, BASELINE_ATTN_SHA, 49152), (BASELINE_FFN, BASELINE_FFN_SHA, 49152)):
        if not path.is_file() or path.stat().st_size != size or r2a.sha256_file(path) != digest:
            raise DiagnosticError(f"immutable baseline mismatch: {path}")


def contained(root: Path, reported: str, size: int, label: str) -> Path:
    try:
        candidate = Path(reported)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DiagnosticError(f"{label} escapes or is absent") from exc
    if not resolved.is_file() or resolved.stat().st_size != size:
        raise DiagnosticError(f"{label} size mismatch")
    return resolved


def load_f32(path: Path, count: int, digest: str, label: str) -> np.ndarray:
    if r2a.sha256_file(path) != digest:
        raise DiagnosticError(f"{label} hash mismatch")
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != count or not bool(np.isfinite(values).all()):
        raise DiagnosticError(f"{label} count/finiteness mismatch")
    return values


def validate_candidate(root: Path, model: Path, sources: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    report = read_json(root / "strat01_upstream_rmsnorm_diag.json", "candidate report")
    required = {"command", "state", "self_certifies_pass", "model", "baseline_inputs", "engine_source_sha256", "diagnostic_source_sha256", "ffn_output_sha256", "q8_census", "compiler_family", "donor_graph_executions", "timing_or_rate_claim"}
    if not isinstance(report, dict) or set(report) != required:
        raise DiagnosticError("candidate report schema mismatch")
    if report["command"] != "--strat01-upstream-rmsnorm-diagnostic" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise DiagnosticError("candidate report state mismatch")
    model_item = report["model"]
    if Path(model_item.get("path", "")).resolve(strict=True) != model.resolve(strict=True) or model_item.get("bytes") != r2a.EXPECTED_BYTES or model_item.get("sha256") != r2a.EXPECTED_SHA256:
        raise DiagnosticError("candidate artifact identity mismatch")
    expected_inputs = {"attn_norm": {"path": str(BASELINE_ATTN.resolve()), "sha256": BASELINE_ATTN_SHA}, "ffn_norm": {"path": str(BASELINE_FFN.resolve()), "sha256": BASELINE_FFN_SHA}}
    if report["baseline_inputs"] != expected_inputs:
        raise DiagnosticError("candidate baseline identity mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["diagnostic_header"]["sha256"]:
        raise DiagnosticError("candidate source hash mismatch")
    if report["compiler_family"] != "clang" or report["donor_graph_executions"] != 1 or report["timing_or_rate_claim"] is not None:
        raise DiagnosticError("candidate execution/non-rate contract mismatch")

    tensors: dict[str, np.ndarray] = {}
    caches: dict[str, np.ndarray] = {}
    for arm in ARMS:
        manifest = read_json(root / f"{arm}_manifest.json", f"candidate {arm} manifest")
        if not isinstance(manifest, dict) or set(manifest) != {"arm", "payload_encoding", "shape_order", "tensors", "cache"} or manifest["arm"] != arm or manifest["payload_encoding"] != "IEEE-754 binary32 little-endian":
            raise DiagnosticError(f"candidate {arm} manifest contract mismatch")
        entries = {item.get("name"): item for item in manifest["tensors"] if isinstance(item, dict)}
        if set(entries) != set(r2a.SHAPES) or len(entries) != len(manifest["tensors"]):
            raise DiagnosticError(f"candidate {arm} tensor set mismatch")
        for name, shape in r2a.SHAPES.items():
            item = entries[name]; count = math.prod(shape)
            if set(item) != {"name", "ordinal", "selection", "op", "type", "logical_shape", "payload_order", "byte_count", "path", "sha256"} or item["logical_shape"] != shape or item["op"] != r2a.EXPECTED_OPS[name] or item["ordinal"] != r2a.EXPECTED_ORDINALS[name] or item["type"] != "F32" or item["byte_count"] != 4 * count:
                raise DiagnosticError(f"candidate {arm}/{name} metadata mismatch")
            path = contained(root, item["path"], 4 * count, f"candidate {arm}/{name}")
            tensors[f"{arm}/{name}"] = load_f32(path, count, item["sha256"], f"candidate {arm}/{name}")
        cache = manifest["cache"]
        if cache.get("storage") != "F16" or cache.get("layout") != "layer,slot,576[latent512,rope64]" or cache.get("row_length") != 576 or cache.get("no_separate_v_cache") is not True:
            raise DiagnosticError(f"candidate {arm} cache contract mismatch")
        checkpoints = (("final", 8), ("prefix7", 7)) if arm == "cached7p1" else (("final", 8),)
        for checkpoint, rows in checkpoints:
            item = cache.get(checkpoint)
            if not isinstance(item, dict) or item.get("occupied_slots") != list(range(rows)) or item.get("absolute_positions") != list(range(rows)) or item.get("payload_type") != "dequantized-F32LE":
                raise DiagnosticError(f"candidate {arm}/{checkpoint} cache metadata mismatch")
            path = contained(root, item["path"], rows * 576 * 4, f"candidate {arm}/{checkpoint} cache")
            caches[f"{arm}/{checkpoint}"] = load_f32(path, rows * 576, item["sha256"], f"candidate {arm}/{checkpoint} cache")

    ffn: dict[str, np.ndarray] = {}
    hashes = report["ffn_output_sha256"]
    if set(hashes) != set(ARMS):
        raise DiagnosticError("candidate FFN arm hash set mismatch")
    for arm in ARMS:
        if set(hashes[arm]) != {"norm", "up", "gate"}:
            raise DiagnosticError(f"candidate {arm} FFN hash schema mismatch")
        for name, short in (("ffn_norm-0", "norm"), ("ffn_up-0", "up"), ("ffn_gate-0", "gate")):
            count = math.prod(FFN_SHAPES[name]); path = root / f"{arm}_double_{name}.f32"
            if not path.is_file() or path.stat().st_size != 4 * count:
                raise DiagnosticError(f"candidate {arm}/{name} absent or wrong size")
            ffn[f"{arm}/{name}"] = load_f32(path, count, hashes[arm][short], f"candidate {arm}/{name}")

    census: dict[str, Any] = {}
    for key in ("attention", "ffn"):
        before = root / f"baseline_{'attn' if key == 'attention' else 'ffn'}_norm.q8k"
        after = root / f"candidate_{'attn' if key == 'attention' else 'ffn'}_norm.q8k"
        if not before.is_file() or not after.is_file() or before.stat().st_size != after.stat().st_size or before.stat().st_size != 48 * Q8_BLOCK_BYTES:
            raise DiagnosticError(f"candidate {key} Q8 payload contract mismatch")
        a, b = before.read_bytes(), after.read_bytes()
        measured = {"changed_blocks": sum(a[i:i+Q8_BLOCK_BYTES] != b[i:i+Q8_BLOCK_BYTES] for i in range(0, len(a), Q8_BLOCK_BYTES)), "total_blocks": 48, "changed_bytes": sum(x != y for x, y in zip(a, b)), "total_bytes": len(a)}
        if report["q8_census"].get(key) != measured:
            raise DiagnosticError(f"candidate {key} Q8 census mismatch")
        census[key] = measured
    return tensors, caches, ffn, {"report": report, "q8_census": census}


def adjudicate(candidate: dict[str, np.ndarray], caches: dict[str, np.ndarray], ffn: dict[str, np.ndarray], ref: dict[str, np.ndarray], ref_cache: dict[str, np.ndarray], ref_ffn: dict[str, np.ndarray], baseline: dict[str, np.ndarray], census: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    tensor_results: list[dict[str, Any]] = []
    for arm in ARMS:
        for name in r2a.SHAPES:
            limits = r2a.TERMINAL_LIMITS if name == "ffn_inp-0" else r2a.GENERAL_LIMITS
            result = r2a.judged_metrics(candidate[f"{arm}/{name}"], ref[f"{arm}/{name}"], limits)
            result.update({"arm": arm, "tensor": name, "baseline": r2a.metrics(baseline[f"{arm}/{name}"], ref[f"{arm}/{name}"]), "delta_vs_float_baseline": r2a.metrics(candidate[f"{arm}/{name}"], baseline[f"{arm}/{name}"])})
            tensor_results.append(result)
            if not result["pass"]:
                failures.append(f"{arm}/{name}")
    cache_results: list[dict[str, Any]] = []
    for key in sorted(ref_cache):
        result = r2a.judged_metrics(caches[key], ref_cache[key], r2a.GENERAL_LIMITS); result["checkpoint"] = key; cache_results.append(result)
        if not result["pass"]:
            failures.append(f"cache/{key}")
    continuity: list[dict[str, Any]] = []
    for name, shape in r2a.SHAPES.items():
        result = r2a.judged_metrics(r2a.token7(candidate[f"cached7p1/{name}"], shape), r2a.token7(candidate[f"prefill8/{name}"], shape), CONTINUITY_LIMITS); result["tensor"] = name; continuity.append(result)
        if not result["pass"]:
            failures.append(f"continuity/{name}")
    ffn_results: list[dict[str, Any]] = []
    for arm in ARMS:
        for name in FFN_NAMES:
            result = r2a.judged_metrics(ffn[f"{arm}/{name}"], ref_ffn[f"{arm}/{name}"], r2a.GENERAL_LIMITS); result.update({"arm": arm, "tensor": name}); ffn_results.append(result)
            if name in {"ffn_up-0", "ffn_gate-0"} and not result["pass"]:
                failures.append(f"primary/{arm}/{name}")
    swapped = {
        "up_as_gate": r2a.judged_metrics(ffn["prefill8/ffn_up-0"], ref_ffn["prefill8/ffn_gate-0"], r2a.GENERAL_LIMITS),
        "gate_as_up": r2a.judged_metrics(ffn["prefill8/ffn_gate-0"], ref_ffn["prefill8/ffn_up-0"], r2a.GENERAL_LIMITS),
    }
    if swapped["up_as_gate"]["pass"] or swapped["gate_as_up"]["pass"]:
        raise DiagnosticError("swapped FFN targets did not reject")
    source = HEADER.read_text(encoding="utf-8")
    double_site_count = source.count("strat01_r2b_rmsnorm_double_sum(")
    if double_site_count != 4:  # definition is in another header; three calls plus self-test call here
        raise DiagnosticError("upstream double-site source control mismatch")
    mutated = bytearray(BASELINE_ATTN.read_bytes()); mutated[len(mutated)//2] ^= 1
    if r2a.sha256_file(BASELINE_ATTN) != BASELINE_ATTN_SHA or __import__("hashlib").sha256(mutated).hexdigest() == BASELINE_ATTN_SHA:
        raise DiagnosticError("baseline mutation control failed")
    status = "UPSTREAM_DOUBLE_RMSNORM_SUFFICIENT_FOR_PROJECTION_GATES" if not failures else "UPSTREAM_DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES"
    return {
        "status": status, "failures": failures, "rung2a_tensor_results": tensor_results,
        "cache_results": cache_results, "continuity_results": continuity,
        "ffn_results": ffn_results, "q8_census": census,
        "controls": {"prior_hashes_valid": True, "baseline_hashes_valid": True, "mutated_baseline_refused": True, "swapped_targets": swapped, "double_site_count": double_site_count},
    }


def run_command(command: list[str], output: Path, label: str, timeout: int) -> dict[str, Any]:
    return r2a.run_command(command, cwd=ROOT, output=output, label=label, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args(); model = args.model.resolve(); output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter(); status = "VOID_UPSTREAM_RMSNORM_PROPAGATION_DIAGNOSTIC"; errors: list[str] = []; commands: dict[str, Any] = {}; sources: dict[str, Any] = {}; candidate_record: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}; binary: Path | None = None; donor_graph_executions = 0
    compiler = shutil.which("clang")
    try:
        sources = source_inventory(); validate_prior_bindings()
        if not compiler:
            raise DiagnosticError("clang is unavailable")
        commands["clang_version"] = run_command([compiler, "--version"], output, "clang_version", 30); r2a.require_ok(commands["clang_version"], "clang --version")
        binary = output / "engine_upstream_rmsnorm_diag.exe"
        commands["compile"] = run_command([compiler, *r2a.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output, "compile", 600); r2a.require_ok(commands["compile"], "C build")
        commands["diagnostic_selftest"] = run_command([str(binary), "--strat01-upstream-rmsnorm-diagnostic-selftest"], output, "diagnostic_selftest", 300); r2a.require_ok(commands["diagnostic_selftest"], "diagnostic selftest")
        commands["legacy_selftest"] = run_command([str(binary), "--kselftest"], output, "legacy_selftest", 300); r2a.require_ok(commands["legacy_selftest"], "legacy selftest")
        commands["python_tests"] = run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_upstream_rmsnorm_diag"], output, "python_tests", 300); r2a.require_ok(commands["python_tests"], "diagnostic Python tests")
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            if not model.is_file() or model.stat().st_size != r2a.EXPECTED_BYTES or r2a.sha256_file(model) != r2a.EXPECTED_SHA256:
                raise DiagnosticError("accepted model identity mismatch")
            c_root = output / "candidate"; c_root.mkdir()
            donor_graph_executions = 1
            commands["candidate"] = run_command([str(binary), "--strat01-upstream-rmsnorm-diagnostic", str(model), "--baseline-attn-norm", str(BASELINE_ATTN.resolve()), "--baseline-ffn-norm", str(BASELINE_FFN.resolve()), "--out-dir", str(c_root)], output, "candidate", 21600); r2a.require_ok(commands["candidate"], "candidate diagnostic")
            sources = source_inventory(); candidate, caches, ffn, candidate_record = validate_candidate(c_root, model, sources)
            ref, ref_record, ref_cache = r2a.validate_reference_outputs(RUNG2A_REFERENCE, model)
            ref_ffn, ref_ffn_record = r2b.validate_reference(RUNG2B_REFERENCE, model)
            baseline_report = read_json(ATTENTION_RUN / "c_engine/strat01_rung2a.json", "accepted attention C report")
            baseline_sources = {"engine": {"sha256": baseline_report.get("engine_source_sha256")}, "rung2a_header": {"sha256": baseline_report.get("rung2a_source_sha256")}}
            baseline, _, _ = r2a.validate_c_outputs(ATTENTION_RUN / "c_engine", baseline_sources, model)
            adjudication = adjudicate(candidate, caches, ffn, ref, ref_cache, ref_ffn, baseline, candidate_record["q8_census"])
            adjudication["reference_records"] = {"rung2a": ref_record, "rung2b": ref_ffn_record}
            status = adjudication["status"]
    except (DiagnosticError, r2a.RunnerError, r2b.RunnerError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-started, "git_head": r2a.git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": r2a.git_value(["git", "status", "--porcelain"]), "source_hashes": sources, "artifact": {"path": str(model), "expected_bytes": r2a.EXPECTED_BYTES, "expected_sha256": r2a.EXPECTED_SHA256}, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": r2a.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_upstream_rmsnorm_propagation_adjudication_v1", "status": status, "errors": errors, "donor_graph_executions": donor_graph_executions, "adjudication": adjudication, "candidate_record": candidate_record, "non_claims": ["production repair", "complete Rung 2B", "Rung 2C", "quality, generation, RAM, or rate"], "provenance": provenance}
    manifest = {"schema": "strat01_upstream_rmsnorm_propagation_run_manifest_v1", "status": status, "errors": errors, "donor_graph_executions": record["donor_graph_executions"], "provenance": provenance}
    r2a.write_json(output / "adjudication.json", record); r2a.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION", "UPSTREAM_DOUBLE_RMSNORM_SUFFICIENT_FOR_PROJECTION_GATES", "UPSTREAM_DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
