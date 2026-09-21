#!/usr/bin/env python3
"""Execute and adjudicate the frozen offline kqv_out attribution diagnostic."""
from __future__ import annotations

import argparse
import json
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

from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_repair import (
    MODEL,
    MODEL_BYTES,
    MODEL_SHA,
    OPERATOR,
    metrics,
    require_ok,
    run_command,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
SOURCE_RUN = HERE / "results/strat01_gigachat_engine_rung2a_repair2_20260921/pinned_reference"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_KQV_OUT_DIAGNOSTIC_PROTOCOL_20260921.md"
BUILDER = HERE / "build_strat01_kqv_out_diagnostic.py"
HELPER_SOURCE = HERE / "strat01_kqv_out_diagnostic.cpp"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_kqv_out_diagnostic_20260921"
COUNT_OUTPUT = 8 * 6144
COUNT_LATENT = 8 * 32 * 512
Q8_BYTES = 8 * 32 * 2 * 292
TIGHT_NRMSE, TIGHT_MAX = 2e-6, 1e-5
OLD_NRMSE, OLD_MAX = 2e-3, 1e-2

PAYLOADS = {
    "qcur": ("Qcur-0", 8 * 32 * 576, "4aca21f044acf71404ef0a7a000ed7b1bfe894efa82c5c1e49f7a5764cc6314b"),
    "kcur": ("Kcur-0", 8 * 576, "2860d9791b620d19788b8112e5366424be21669153eb1ec3255fd5a6c1b167f3"),
    "vcur": ("Vcur-0", 8 * 512, "8b775afa6fedd04f3c99bca0700f365cd13bbf235cef82e8f21809fbf252d2e9"),
    "target": ("kqv_out-0", COUNT_OUTPUT, "bb73ca15e48df5de663c5fd90f9104a9a6e78652d5322aac12fba660d33177f3"),
}

CRITICAL_PATHS = (
    Path(__file__).resolve(), BUILDER, HELPER_SOURCE, OPERATOR,
    HERE / "test_strat01_kqv_out_diagnostic.py", PROTOCOL,
)


class DiagnosticError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_payload(arm: str, logical: str) -> Path:
    root = SOURCE_RUN / arm
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest["payloads"] if item.get("logical") == logical and item.get("kind") == "full"]
    if len(matches) != 1:
        raise DiagnosticError(f"missing or duplicate frozen payload {arm}/{logical}")
    item = matches[0]
    path = SOURCE_RUN / item["path"]
    if not path.is_file() or path.stat().st_size != item["byte_count"] or sha256_file(path) != item["sha256"]:
        raise DiagnosticError(f"frozen manifest payload validation failed: {arm}/{logical}")
    return path


def load_f32(path: Path, count: int, expected_sha: str | None = None) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * 4:
        raise DiagnosticError(f"float payload size mismatch: {path}")
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise DiagnosticError(f"float payload hash mismatch: {path}")
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != count or not bool(np.isfinite(values).all()):
        raise DiagnosticError(f"float payload invalid: {path}")
    return values


def gate(candidate: np.ndarray, reference: np.ndarray, nrmse: float, maximum: float) -> dict[str, Any]:
    result = metrics(candidate, reference)
    result.update({"nrmse_limit": nrmse, "normalized_max_limit": maximum})
    result["pass"] = result["nrmse"] <= nrmse and result["normalized_max"] <= maximum
    return result


def classify(results: dict[str, dict[str, Any]], controls: dict[str, bool]) -> str:
    if not all(controls.values()):
        return "VOID_KQV_OUT_DIAGNOSTIC"
    q8_agree = bool(results["project_vs_pinned"]["pass"])
    target_q8 = bool(results["project_vs_target"]["pass"] and results["pinned_vs_target"]["pass"])
    d32_fails = not bool(results["d32_vs_target"]["pass"])
    if q8_agree and target_q8 and d32_fails:
        return "ATTRIBUTED_KQV_OUT_TO_VB_Q8K"
    if bool(results["pinned_vs_target"]["pass"]):
        return "PARTIAL_VB_Q8K_ATTRIBUTION"
    return "ATTENTION_RECONSTRUCTION_OR_OTHER_VB_SEMANTICS_REMAIN"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise DiagnosticError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": "strat01_kqv_out_diagnostic_v1",
        "status": "VOID_KQV_OUT_DIAGNOSTIC",
        "started_utc": utc_now(),
        "donor_executions": 0,
        "commands": [],
        "errors": [],
    }
    try:
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)],
            cwd=ROOT, check=False,
        )
        if dirty.returncode:
            raise DiagnosticError("diagnostic implementation or frozen protocol differs from HEAD")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        if not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES or sha256_file(MODEL) != MODEL_SHA:
            raise DiagnosticError("accepted GGUF identity mismatch")

        paths: dict[str, Path] = {}
        arrays: dict[str, np.ndarray] = {}
        for key, (logical, count, expected_sha) in PAYLOADS.items():
            prefill = manifest_payload("prefill8", logical)
            cached = manifest_payload("cached7p1", logical)
            if sha256_file(prefill) != expected_sha or sha256_file(cached) != expected_sha:
                raise DiagnosticError(f"frozen payload identity mismatch: {logical}")
            if prefill.read_bytes() != cached.read_bytes():
                raise DiagnosticError(f"prefill/cached full payloads differ: {logical}")
            paths[key] = prefill
            arrays[key] = load_f32(prefill, count, expected_sha)

        build_dir = output / "build"
        build_record = run_command(
            [sys.executable, str(BUILDER), "--build-dir", str(build_dir)],
            output, "build_helper", timeout=3600,
        )
        record["commands"].append(build_record)
        require_ok(build_record, "kqv helper build")
        candidates = list((build_dir / "cmake-build").rglob("strat01_kqv_out_diagnostic.exe"))
        if len(candidates) != 1:
            raise DiagnosticError("cannot resolve built kqv helper")
        helper = candidates[0]
        tests = run_command(
            [sys.executable, "-m", "unittest", "-v",
             "benchmarks.donor_adaptation.engine.test_strat01_q4k_q8k_operator",
             "benchmarks.donor_adaptation.engine.test_strat01_q4k_q8k_repair_runner",
             "benchmarks.donor_adaptation.engine.test_strat01_rung2a_q4_repair_confirmation",
             "benchmarks.donor_adaptation.engine.test_strat01_kqv_out_diagnostic"],
            output, "model_free_tests", timeout=3600,
        )
        record["commands"].append(tests)
        require_ok(tests, "model-free tests")
        helper_selftest = run_command([str(helper), "--selftest"], output, "helper_selftest")
        record["commands"].append(helper_selftest)
        require_ok(helper_selftest, "helper selftest")

        products = output / "products"
        products.mkdir()
        diagnostic = run_command(
            [str(helper), "--model", str(MODEL), "--qcur", str(paths["qcur"]),
             "--kcur", str(paths["kcur"]), "--vcur", str(paths["vcur"]),
             "--output-dir", str(products)],
            output, "offline_diagnostic", timeout=3600,
        )
        record["commands"].append(diagnostic)
        require_ok(diagnostic, "offline diagnostic")

        produced = {
            name: load_f32(products / filename, COUNT_OUTPUT)
            for name, filename in {
                "d32": "d32.f32le", "project": "project_q8.f32le", "pinned": "pinned_q8.f32le",
                "mutated": "mutated_q8.f32le", "transposed": "transposed_head.f32le",
                "wrong_scales": "wrong_scales.f32le",
            }.items()
        }
        latent = load_f32(products / "latent.f32le", COUNT_LATENT)
        project_q8_path, pinned_q8_path = products / "project_q8.bin", products / "pinned_q8.bin"
        if project_q8_path.stat().st_size != Q8_BYTES or pinned_q8_path.stat().st_size != Q8_BYTES:
            raise DiagnosticError("Q8_K audit payload size mismatch")
        q8_exact = project_q8_path.read_bytes() == pinned_q8_path.read_bytes()
        results = {
            "project_vs_pinned": gate(produced["project"], produced["pinned"], TIGHT_NRMSE, TIGHT_MAX),
            "project_vs_target": gate(produced["project"], arrays["target"], TIGHT_NRMSE, TIGHT_MAX),
            "pinned_vs_target": gate(produced["pinned"], arrays["target"], TIGHT_NRMSE, TIGHT_MAX),
            "d32_vs_target": gate(produced["d32"], arrays["target"], OLD_NRMSE, OLD_MAX),
            "mutated_vs_target": gate(produced["mutated"], arrays["target"], TIGHT_NRMSE, TIGHT_MAX),
            "transposed_vs_target": gate(produced["transposed"], arrays["target"], TIGHT_NRMSE, TIGHT_MAX),
            "wrong_scales_vs_target": gate(produced["wrong_scales"], arrays["target"], TIGHT_NRMSE, TIGHT_MAX),
        }
        controls = {
            "identity_and_schedule_equality": True,
            "model_free_tests": True,
            "helper_selftest": True,
            "q8_bytes_exact": q8_exact,
            "q8_mutation_fires": not bool(results["mutated_vs_target"]["pass"]),
            "head_transpose_fires": not bool(results["transposed_vs_target"]["pass"]),
            "packed_scale_error_fires": not bool(results["wrong_scales_vs_target"]["pass"]),
        }
        record.update({
            "status": classify(results, controls),
            "finished_utc": utc_now(),
            "seconds": time.perf_counter() - started,
            "identity": {
                "git_head": head,
                "model": {"path": str(MODEL), "bytes": MODEL_BYTES, "sha256": MODEL_SHA},
                "pinned_payloads": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()},
                "helper": {"path": str(helper), "sha256": sha256_file(helper)},
                "critical_source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS},
            },
            "controls": controls,
            "results": results,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in products.iterdir() if path.is_file()
            },
            "latent": {"count": int(latent.size), "sha256": sha256_file(products / "latent.f32le")},
            "non_claims": ["production V-B repair", "Rung 2B", "quality", "RAM", "speed"],
        })
    except Exception as error:
        record["errors"].append(str(error))
        record["finished_utc"] = utc_now()
        record["seconds"] = time.perf_counter() - started
    (output / "adjudication.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(record["status"])
    if record["errors"]:
        print(record["errors"][0], file=sys.stderr)
    return 0 if record["status"] != "VOID_KQV_OUT_DIAGNOSTIC" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        raise SystemExit(f"run_strat01_kqv_out_diagnostic: {error}")
