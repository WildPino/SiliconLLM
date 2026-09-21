#!/usr/bin/env python3
"""Capture pinned pre/post-V-B tensors once and adjudicate the frozen boundary."""
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
    metrics,
    require_ok,
    run_command,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_PRE_VB_LATENT_CAPTURE_PROTOCOL_20260921.md"
REFERENCE_SOURCE = HERE / "strat01_engine_rung2a_reference.cpp"
REFERENCE_BUILDER = HERE / "build_strat01_engine_rung2a_reference.py"
HELPER_SOURCE = HERE / "strat01_kqv_out_diagnostic.cpp"
HELPER_BUILDER = HERE / "build_strat01_kqv_out_diagnostic.py"
OPERATOR = ROOT / "benchmarks/phase60/strat01_q4k_q8k.h"
TEST_SOURCE = HERE / "test_strat01_pre_vb_latent_capture.py"
PRIOR_LATENT = HERE / "results/strat01_gigachat_engine_kqv_out_diagnostic_20260921/products/latent.f32le"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_pre_vb_latent_capture_20260921"
PINNED_LLAMA = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4")
PINNED_HEAD = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
PINNED_GRAPH_SHA = "000c88afa5ebd4f1d20821054ef9dc4d10a0c9cea4f440d179eaea8723ad8dee"
PRIOR_LATENT_SHA = "88990b6ba96f21a001fa591a0fe63365531f31672fbb43360467df3a2407465d"
TIGHT_NRMSE, TIGHT_MAX = 2e-6, 1e-5
COUNT_LATENT = 8 * 32 * 512
COUNT_OUTPUT = 8 * 32 * 192
Q8_BYTES = 8 * 32 * 2 * 292
EXPECTED = {
    "kqv-0": ([512, 8, 32], COUNT_LATENT),
    "kqv_mla-0": ([192, 8, 32], COUNT_OUTPUT),
    "kqv_out-0": ([6144, 8], COUNT_OUTPUT),
}
CRITICAL_PATHS = (
    Path(__file__).resolve(), PROTOCOL, REFERENCE_SOURCE, REFERENCE_BUILDER,
    HELPER_SOURCE, HELPER_BUILDER, OPERATOR, TEST_SOURCE,
)


class CaptureError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_f32(path: Path, count: int, expected_sha: str | None = None) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * 4:
        raise CaptureError(f"float payload size mismatch: {path}")
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise CaptureError(f"float payload hash mismatch: {path}")
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != count or not bool(np.isfinite(values).all()):
        raise CaptureError(f"invalid float payload: {path}")
    return values


def gate(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result = metrics(candidate, reference)
    result.update({"nrmse_limit": TIGHT_NRMSE, "normalized_max_limit": TIGHT_MAX})
    result["pass"] = result["nrmse"] <= TIGHT_NRMSE and result["normalized_max"] <= TIGHT_MAX
    return result


def callback_to_token_head(values: np.ndarray, width: int) -> np.ndarray:
    """Map GGML [width, token, head] canonical order to [token, head, width]."""
    expected = 32 * 8 * width
    if values.size != expected:
        raise CaptureError("callback axis mapping received the wrong element count")
    return values.reshape(32, 8, width).transpose(1, 0, 2).copy().reshape(expected)


def classify(results: dict[str, dict[str, Any]], controls: dict[str, bool]) -> str:
    if not all(controls.values()):
        return "VOID_PRE_VB_LATENT_CAPTURE"
    latent = bool(results["reconstruction_vs_kqv"]["pass"])
    q8_agrees = bool(results["project_vs_pinned"]["pass"])
    vb = q8_agrees and bool(results["project_vs_kqv_mla"]["pass"] and results["pinned_vs_kqv_mla"]["pass"])
    layout = bool(results["kqv_mla_vs_kqv_out"]["pass"])
    if latent and vb and layout:
        return "PRE_VB_LATENT_CAPTURE_PASS"
    if not latent and vb and layout:
        return "ATTRIBUTED_RESIDUAL_TO_ATTENTION_RECONSTRUCTION"
    if latent and q8_agrees and not vb and layout:
        return "ATTRIBUTED_RESIDUAL_TO_VB_GRAPH_SEMANTICS"
    if latent and vb and not layout:
        return "ATTRIBUTED_RESIDUAL_TO_FINAL_LAYOUT"
    return "MIXED_PRE_VB_RESIDUAL"


def capture_payloads(root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    manifest_path = root / "prefill8/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest.get("logical_selection", {})
    paths: dict[str, Path] = {}
    metadata: dict[str, Any] = {}
    for logical, (shape, count) in EXPECTED.items():
        entry = selected.get(logical)
        if not isinstance(entry, dict) or entry.get("logical_shape") != shape:
            raise CaptureError(f"callback logical shape mismatch: {logical}")
        source = entry.get("source")
        if not isinstance(source, dict) or source.get("name") != logical:
            raise CaptureError(f"callback identity mismatch: {logical}")
        matches = [item for item in manifest.get("payloads", []) if item.get("logical") == logical and item.get("kind") == "full"]
        if len(matches) != 1:
            raise CaptureError(f"missing or duplicate callback payload: {logical}")
        item = matches[0]
        path = root / item["path"]
        load_f32(path, count, item.get("sha256"))
        if item.get("byte_count") != count * 4:
            raise CaptureError(f"manifest byte count mismatch: {logical}")
        paths[logical] = path
        metadata[logical] = {"shape": shape, "source": source, "sha256": item["sha256"]}
    return paths, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise CaptureError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": "strat01_pre_vb_latent_capture_v1",
        "status": "VOID_PRE_VB_LATENT_CAPTURE",
        "started_utc": utc_now(),
        "donor_executions": 0,
        "commands": [],
        "errors": [],
    }
    try:
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)], cwd=ROOT, check=False)
        if dirty.returncode:
            raise CaptureError("capture implementation or frozen protocol differs from HEAD")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        if not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES or sha256_file(MODEL) != MODEL_SHA:
            raise CaptureError("accepted GGUF identity mismatch")
        if sha256_file(PINNED_LLAMA / "src/llama-graph.cpp") != PINNED_GRAPH_SHA:
            raise CaptureError("pinned llama.cpp graph source hash mismatch")
        llama_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PINNED_LLAMA, text=True, capture_output=True, check=True).stdout.strip()
        llama_dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PINNED_LLAMA, text=True, capture_output=True, check=True).stdout.strip()
        if llama_head != PINNED_HEAD or llama_dirty:
            raise CaptureError("pinned llama.cpp checkout is not clean at the frozen revision")
        prior_latent = load_f32(PRIOR_LATENT, COUNT_LATENT, PRIOR_LATENT_SHA)

        tests = run_command(
            [sys.executable, "-m", "unittest", "-v",
             "benchmarks.donor_adaptation.engine.test_strat01_engine_rung2a",
             "benchmarks.donor_adaptation.engine.test_strat01_kqv_out_diagnostic",
             "benchmarks.donor_adaptation.engine.test_strat01_pre_vb_latent_capture"],
            output, "model_free_tests", timeout=3600,
        )
        record["commands"].append(tests)
        require_ok(tests, "model-free tests")

        reference_build = output / "reference_build"
        command = run_command([sys.executable, str(REFERENCE_BUILDER), "--build-dir", str(reference_build)], output, "build_reference", timeout=3600)
        record["commands"].append(command)
        require_ok(command, "reference build")
        reference_candidates = list((reference_build / "cmake-build").rglob("strat01_engine_rung2a_reference.exe"))
        if len(reference_candidates) != 1:
            raise CaptureError("cannot resolve reference executable")
        reference = reference_candidates[0]
        selftest = run_command([str(reference), "--self-test"], output, "reference_selftest")
        record["commands"].append(selftest)
        require_ok(selftest, "reference self-test")

        helper_build = output / "helper_build"
        command = run_command([sys.executable, str(HELPER_BUILDER), "--build-dir", str(helper_build)], output, "build_helper", timeout=3600)
        record["commands"].append(command)
        require_ok(command, "helper build")
        helper_candidates = list((helper_build / "cmake-build").rglob("strat01_kqv_out_diagnostic.exe"))
        if len(helper_candidates) != 1:
            raise CaptureError("cannot resolve V-B helper executable")
        helper = helper_candidates[0]
        helper_selftest = run_command([str(helper), "--selftest"], output, "helper_selftest")
        record["commands"].append(helper_selftest)
        require_ok(helper_selftest, "helper self-test")

        trace = output / "pinned_reference"
        record["donor_executions"] = 1
        command = run_command([str(reference), "--model", str(MODEL), "--out-dir", str(trace), "--arm", "prefill8"], output, "capture_reference", timeout=3600)
        record["commands"].append(command)
        require_ok(command, "pinned pre-V-B capture")
        paths, callback_metadata = capture_payloads(trace)
        kqv_raw = load_f32(paths["kqv-0"], COUNT_LATENT)
        kqv_mla_raw = load_f32(paths["kqv_mla-0"], COUNT_OUTPUT)
        kqv_out = load_f32(paths["kqv_out-0"], COUNT_OUTPUT)

        mapped = output / "mapped"
        mapped.mkdir()
        kqv = callback_to_token_head(kqv_raw, 512)
        kqv_mla = callback_to_token_head(kqv_mla_raw, 192)
        mapped_kqv_path = mapped / "kqv-0.token_head.f32le"
        mapped_kqv_mla_path = mapped / "kqv_mla-0.token_head.f32le"
        kqv.astype("<f4", copy=False).tofile(mapped_kqv_path)
        kqv_mla.astype("<f4", copy=False).tofile(mapped_kqv_mla_path)

        products = output / "true_latent_products"
        products.mkdir()
        command = run_command([str(helper), "--model", str(MODEL), "--latent", str(mapped_kqv_path), "--output-dir", str(products)], output, "evaluate_true_latent", timeout=3600)
        record["commands"].append(command)
        require_ok(command, "true-latent V-B evaluation")
        produced = {
            name: load_f32(products / filename, COUNT_OUTPUT)
            for name, filename in {
                "project": "project_q8.f32le", "pinned": "pinned_q8.f32le",
                "mutated": "mutated_q8.f32le", "transposed": "transposed_head.f32le",
                "wrong_scales": "wrong_scales.f32le",
            }.items()
        }
        project_q8, pinned_q8 = products / "project_q8.bin", products / "pinned_q8.bin"
        q8_exact = project_q8.stat().st_size == Q8_BYTES and pinned_q8.stat().st_size == Q8_BYTES and project_q8.read_bytes() == pinned_q8.read_bytes()
        results = {
            "reconstruction_vs_kqv": gate(prior_latent, kqv),
            "project_vs_pinned": gate(produced["project"], produced["pinned"]),
            "project_vs_kqv_mla": gate(produced["project"], kqv_mla),
            "pinned_vs_kqv_mla": gate(produced["pinned"], kqv_mla),
            # Source-derived mapping: [192,8,32] is PERMUTE(0,2,1,3) then
            # CONT_2D into the canonical [6144,8] token-major sequence.
            "kqv_mla_vs_kqv_out": gate(kqv_mla, kqv_out),
            "mutated_vs_kqv_mla": gate(produced["mutated"], kqv_mla),
            "transposed_vs_kqv_mla": gate(produced["transposed"], kqv_mla),
            "wrong_scales_vs_kqv_mla": gate(produced["wrong_scales"], kqv_mla),
        }
        controls = {
            "identity_and_clean_pinned_source": True,
            "model_free_tests": True,
            "reference_and_helper_selftests": True,
            "callback_completeness_shapes_and_finiteness": True,
            "q8_bytes_exact": q8_exact,
            "q8_mutation_fires": not bool(results["mutated_vs_kqv_mla"]["pass"]),
            "head_transpose_fires": not bool(results["transposed_vs_kqv_mla"]["pass"]),
            "packed_scale_error_fires": not bool(results["wrong_scales_vs_kqv_mla"]["pass"]),
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
                "prior_reconstructed_latent": {"path": str(PRIOR_LATENT), "sha256": PRIOR_LATENT_SHA},
                "critical_source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS},
            },
            "callback_metadata": callback_metadata,
            "layout_mapping": {
                "source": "PERMUTE(0,2,1,3) then CONT_2D",
                "input_callback_shapes": {"kqv": [512, 8, 32], "kqv_mla": [192, 8, 32]},
                "mapping": "raw reshape [head,token,width], transpose to [token,head,width]",
                "comparison": "mapped [token,head,192] canonical sequence versus [6144,8]",
            },
            "controls": controls,
            "results": results,
            "outputs": {
                str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in list(paths.values()) + [mapped_kqv_path, mapped_kqv_mla_path] + [item for item in products.iterdir() if item.is_file()]
            },
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
    return 0 if record["status"] != "VOID_PRE_VB_LATENT_CAPTURE" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CaptureError as error:
        raise SystemExit(f"run_strat01_pre_vb_latent_capture: {error}")
