#!/usr/bin/env python3
"""Capture and adjudicate the frozen STRAT-01 attention-stage boundary."""
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

from benchmarks.donor_adaptation.engine.run_strat01_pre_vb_latent_capture import (
    MODEL,
    MODEL_BYTES,
    MODEL_SHA,
    PINNED_GRAPH_SHA,
    PINNED_HEAD,
    PINNED_LLAMA,
    callback_to_token_head,
)
from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_repair import (
    metrics,
    require_ok,
    run_command,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_ATTENTION_STAGE_DIAGNOSTIC_PROTOCOL_20260921.md"
REFERENCE_SOURCE = HERE / "strat01_engine_rung2a_reference.cpp"
REFERENCE_BUILDER = HERE / "build_strat01_engine_rung2a_reference.py"
HELPER_SOURCE = HERE / "strat01_attention_stage_diagnostic.cpp"
HELPER_BUILDER = HERE / "build_strat01_attention_stage_diagnostic.py"
TEST_SOURCE = HERE / "test_strat01_attention_stage_diagnostic.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_attention_stage_diagnostic_20260921"
TIGHT_NRMSE, TIGHT_MAX = 2e-6, 1e-5
COUNT_SCORES = 8 * 32 * 8
COUNT_LATENT = 8 * 32 * 512
EXPECTED = {
    "Qcur-0": ([576, 32, 8], 8 * 32 * 576, "4aca21f044acf71404ef0a7a000ed7b1bfe894efa82c5c1e49f7a5764cc6314b", "CONCAT"),
    "Kcur-0": ([576, 1, 8], 8 * 576, "2860d9791b620d19788b8112e5366424be21669153eb1ec3255fd5a6c1b167f3", "CONCAT"),
    "Vcur-0": ([512, 1, 8], 8 * 512, "8b775afa6fedd04f3c99bca0700f365cd13bbf235cef82e8f21809fbf252d2e9", "RESHAPE"),
    "kq-0": ([8, 8, 32], COUNT_SCORES, None, "MUL_MAT"),
    "kq_soft_max-0": ([8, 8, 32], COUNT_SCORES, None, "SOFT_MAX"),
    "kqv-0": ([512, 8, 32], COUNT_LATENT, "3922f34159f499dd26788acb7d9600d72fded004425392946bc09b8098fd88df", "MUL_MAT"),
}
CRITICAL_PATHS = (
    Path(__file__).resolve(), PROTOCOL, REFERENCE_SOURCE, REFERENCE_BUILDER,
    HELPER_SOURCE, HELPER_BUILDER, TEST_SOURCE,
)


class DiagnosticError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_f32(path: Path, count: int, expected_sha: str | None = None) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * 4:
        raise DiagnosticError(f"float payload size mismatch: {path}")
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise DiagnosticError(f"float payload hash mismatch: {path}")
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != count or not bool(np.isfinite(values).all()):
        raise DiagnosticError(f"invalid float payload: {path}")
    return values


def gate(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result = metrics(candidate, reference)
    result.update({"nrmse_limit": TIGHT_NRMSE, "normalized_max_limit": TIGHT_MAX})
    result["pass"] = result["nrmse"] <= TIGHT_NRMSE and result["normalized_max"] <= TIGHT_MAX
    return result


def classify(results: dict[str, dict[str, Any]], controls: dict[str, bool]) -> str:
    if not all(controls.values()):
        return "VOID_ATTENTION_STAGE_DIAGNOSTIC"
    raw = bool(results["raw_qk"]["pass"])
    softmax = bool(results["captured_qk_softmax"]["pass"])
    reduction = bool(results["captured_softmax_value_reduction"]["pass"])
    if raw and softmax and reduction:
        return "ATTENTION_STAGE_DIAGNOSTIC_PASS"
    if not raw and softmax and reduction:
        return "ATTRIBUTED_ATTENTION_RESIDUAL_TO_QK_DOT"
    if raw and not softmax and reduction:
        return "ATTRIBUTED_ATTENTION_RESIDUAL_TO_SOFTMAX"
    if raw and softmax and not reduction:
        return "ATTRIBUTED_ATTENTION_RESIDUAL_TO_VALUE_REDUCTION"
    return "MIXED_ATTENTION_STAGE_RESIDUAL"


def capture_payloads(root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    manifest = json.loads((root / "prefill8/manifest.json").read_text(encoding="utf-8"))
    selected = manifest.get("logical_selection", {})
    paths: dict[str, Path] = {}
    metadata: dict[str, Any] = {}
    for logical, (shape, count, expected_sha, expected_op) in EXPECTED.items():
        entry = selected.get(logical)
        source = entry.get("source") if isinstance(entry, dict) else None
        if not isinstance(entry, dict) or entry.get("logical_shape") != shape or not isinstance(source, dict):
            raise DiagnosticError(f"callback selection mismatch: {logical}")
        if source.get("name") != logical or source.get("op") != expected_op or source.get("ordinal") != 0 or str(source.get("type", "")).lower() != "f32":
            raise DiagnosticError(f"callback identity/op/type mismatch: {logical}")
        matches = [item for item in manifest.get("payloads", []) if item.get("logical") == logical and item.get("kind") == "full"]
        if len(matches) != 1:
            raise DiagnosticError(f"missing or duplicate callback payload: {logical}")
        item = matches[0]
        path = root / item["path"]
        load_f32(path, count, expected_sha or item.get("sha256"))
        if item.get("byte_count") != count * 4:
            raise DiagnosticError(f"manifest byte count mismatch: {logical}")
        paths[logical] = path
        metadata[logical] = {"shape": shape, "source": source, "sha256": item["sha256"]}
    return paths, metadata


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
        "schema": "strat01_attention_stage_diagnostic_v1",
        "status": "VOID_ATTENTION_STAGE_DIAGNOSTIC",
        "started_utc": utc_now(),
        "donor_executions": 0,
        "commands": [],
        "errors": [],
    }
    try:
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)], cwd=ROOT, check=False)
        if dirty.returncode:
            raise DiagnosticError("diagnostic implementation or frozen protocol differs from HEAD")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        if not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES or sha256_file(MODEL) != MODEL_SHA:
            raise DiagnosticError("accepted GGUF identity mismatch")
        if sha256_file(PINNED_LLAMA / "src/llama-graph.cpp") != PINNED_GRAPH_SHA:
            raise DiagnosticError("pinned graph-source hash mismatch")
        llama_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PINNED_LLAMA, text=True, capture_output=True, check=True).stdout.strip()
        llama_dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PINNED_LLAMA, text=True, capture_output=True, check=True).stdout.strip()
        if llama_head != PINNED_HEAD or llama_dirty:
            raise DiagnosticError("pinned llama.cpp checkout is not clean at the frozen revision")

        tests = run_command([sys.executable, "-m", "unittest", "-v",
            "benchmarks.donor_adaptation.engine.test_strat01_pre_vb_latent_capture",
            "benchmarks.donor_adaptation.engine.test_strat01_attention_stage_diagnostic"], output, "model_free_tests", timeout=3600)
        record["commands"].append(tests)
        require_ok(tests, "model-free tests")

        reference_build = output / "reference_build"
        command = run_command([sys.executable, str(REFERENCE_BUILDER), "--build-dir", str(reference_build)], output, "build_reference", timeout=3600)
        record["commands"].append(command)
        require_ok(command, "reference build")
        references = list((reference_build / "cmake-build").rglob("strat01_engine_rung2a_reference.exe"))
        if len(references) != 1:
            raise DiagnosticError("cannot resolve reference executable")
        reference = references[0]
        command = run_command([str(reference), "--self-test"], output, "reference_selftest")
        record["commands"].append(command)
        require_ok(command, "reference self-test")

        helper_build = output / "helper_build"
        command = run_command([sys.executable, str(HELPER_BUILDER), "--build-dir", str(helper_build)], output, "build_helper", timeout=3600)
        record["commands"].append(command)
        require_ok(command, "helper build")
        helpers = list((helper_build / "cmake-build").rglob("strat01_attention_stage_diagnostic.exe"))
        if len(helpers) != 1:
            raise DiagnosticError("cannot resolve attention-stage helper")
        helper = helpers[0]
        command = run_command([str(helper), "--selftest"], output, "helper_selftest")
        record["commands"].append(command)
        require_ok(command, "helper self-test")

        trace = output / "pinned_reference"
        record["donor_executions"] = 1
        command = run_command([str(reference), "--model", str(MODEL), "--out-dir", str(trace), "--arm", "prefill8"], output, "capture_reference", timeout=3600)
        record["commands"].append(command)
        require_ok(command, "pinned attention-stage capture")
        paths, callback_metadata = capture_payloads(trace)

        mapped = output / "mapped"
        mapped.mkdir()
        captured_kq = callback_to_token_head(load_f32(paths["kq-0"], COUNT_SCORES), 8)
        captured_softmax = callback_to_token_head(load_f32(paths["kq_soft_max-0"], COUNT_SCORES), 8)
        true_kqv = callback_to_token_head(load_f32(paths["kqv-0"], COUNT_LATENT), 512)
        mapped_kq = mapped / "kq-0.token_head_slot.f32le"
        mapped_softmax = mapped / "kq_soft_max-0.token_head_slot.f32le"
        mapped_kqv = mapped / "kqv-0.token_head.f32le"
        captured_kq.astype("<f4", copy=False).tofile(mapped_kq)
        captured_softmax.astype("<f4", copy=False).tofile(mapped_softmax)
        true_kqv.astype("<f4", copy=False).tofile(mapped_kqv)
        if sha256_file(mapped_kqv) != "541183b4eb5992cfcfc711fe3c9f179ab41685106ba9cfe2aa7099f45f821312":
            raise DiagnosticError("mapped true kqv differs from prior frozen capture")

        products = output / "products"
        products.mkdir()
        command = run_command([str(helper), "--qcur", str(paths["Qcur-0"]), "--kcur", str(paths["Kcur-0"]),
            "--vcur", str(paths["Vcur-0"]), "--captured-kq", str(mapped_kq),
            "--captured-softmax", str(mapped_softmax), "--output-dir", str(products)],
            output, "offline_attention_stages", timeout=3600)
        record["commands"].append(command)
        require_ok(command, "offline attention-stage diagnostic")

        scores = {name: load_f32(products / filename, COUNT_SCORES) for name, filename in {
            "raw": "raw_qk.f32le", "mutated": "mutated_qk.f32le", "f32": "f32_cache_qk.f32le",
            "project_softmax": "project_softmax.f32le", "captured_qk_softmax": "captured_qk_softmax.f32le",
        }.items()}
        latents = {name: load_f32(products / filename, COUNT_LATENT) for name, filename in {
            "project": "project_latent.f32le", "captured_softmax": "captured_softmax_latent.f32le",
            "swapped": "swapped_probability_latent.f32le", "f32": "f32_cache_latent.f32le",
        }.items()}
        results = {
            "raw_qk": gate(scores["raw"], captured_kq),
            "captured_qk_softmax": gate(scores["captured_qk_softmax"], captured_softmax),
            "captured_softmax_value_reduction": gate(latents["captured_softmax"], true_kqv),
            "fully_project_composed_latent": gate(latents["project"], true_kqv),
            "mutated_q_vs_kq": gate(scores["mutated"], captured_kq),
            "swapped_probability_vs_kqv": gate(latents["swapped"], true_kqv),
            "f32_cache_qk": gate(scores["f32"], captured_kq),
            "f32_cache_value_reduction": gate(latents["f32"], true_kqv),
        }
        controls = {
            "identity_clean_source_and_prior_payloads": True,
            "model_free_tests_and_selftests": True,
            "callback_completeness_shapes_ops_and_finiteness": True,
            "q_mutation_fires": not bool(results["mutated_q_vs_kq"]["pass"]),
            "probability_swap_fires": not bool(results["swapped_probability_vs_kqv"]["pass"]),
            "unrounded_f32_cache_fires": not bool(results["f32_cache_qk"]["pass"] and results["f32_cache_value_reduction"]["pass"]),
        }
        record.update({
            "status": classify(results, controls),
            "finished_utc": utc_now(),
            "seconds": time.perf_counter() - started,
            "identity": {
                "git_head": head,
                "model": {"path": str(MODEL), "bytes": MODEL_BYTES, "sha256": MODEL_SHA},
                "llama_cpp": {"path": str(PINNED_LLAMA), "head": llama_head, "graph_sha256": PINNED_GRAPH_SHA},
                "reference_binary": {"path": str(reference), "sha256": sha256_file(reference)},
                "helper_binary": {"path": str(helper), "sha256": sha256_file(helper)},
                "critical_source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS},
            },
            "callback_metadata": callback_metadata,
            "axis_mapping": "raw [head,query,slot-or-width] transposed to [query,head,slot-or-width]",
            "controls": controls,
            "results": results,
            "outputs": {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in list(paths.values()) + [mapped_kq, mapped_softmax, mapped_kqv] + [item for item in products.iterdir() if item.is_file()]},
            "non_claims": ["production attention repair", "Rung 2B", "quality", "RAM", "speed"],
        })
    except Exception as error:
        record["errors"].append(str(error))
        record["finished_utc"] = utc_now()
        record["seconds"] = time.perf_counter() - started
    (output / "adjudication.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(record["status"])
    if record["errors"]:
        print(record["errors"][0], file=sys.stderr)
    return 0 if record["status"] != "VOID_ATTENTION_STAGE_DIAGNOSTIC" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        raise SystemExit(f"run_strat01_attention_stage_diagnostic: {error}")
