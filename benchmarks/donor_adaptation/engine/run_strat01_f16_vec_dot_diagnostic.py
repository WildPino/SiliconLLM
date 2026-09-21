#!/usr/bin/env python3
"""Execute the frozen zero-donor F16 vector-dot attribution cell."""
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

from benchmarks.donor_adaptation.engine.run_strat01_pre_vb_latent_capture import PINNED_GRAPH_SHA, PINNED_HEAD, PINNED_LLAMA, callback_to_token_head
from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_repair import metrics, require_ok, run_command, sha256_file

HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_F16_VEC_DOT_DIAGNOSTIC_PROTOCOL_20260921.md"
SOURCE_TRACE = HERE / "results/strat01_gigachat_engine_attention_stage_diagnostic_20260921/pinned_reference/prefill8"
PRIOR = HERE / "results/strat01_gigachat_engine_attention_stage_offline_adjudication_20260921/adjudication.json"
PRIOR_SHA = "9292a15db752cfe63cead091d7d8a4f7f3c4f7295c13c68bcb5981ebd87541d8"
BUILDER = HERE / "build_strat01_f16_vec_dot_diagnostic.py"
HELPER_SOURCE = HERE / "strat01_f16_vec_dot_diagnostic.cpp"
TEST_SOURCE = HERE / "test_strat01_f16_vec_dot_diagnostic.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_f16_vec_dot_diagnostic_20260921"
TIGHT_NRMSE, TIGHT_MAX = 2e-6, 1e-5
COUNT_SCORES = 8 * 32 * 8
COUNT_PADDED = 8 * 32 * 256
COUNT_LATENT = 8 * 32 * 512
CRITICAL_PATHS = (Path(__file__).resolve(), PROTOCOL, BUILDER, HELPER_SOURCE, TEST_SOURCE)


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


def classify(results: dict[str, dict[str, Any]], controls: dict[str, bool], old: dict[str, dict[str, Any]]) -> str:
    if not all(controls.values()):
        return "VOID_F16_VEC_DOT_DIAGNOSTIC"
    q_scalar = bool(results["qk_f16_scalar"]["pass"])
    v_scalar = bool(results["value_f16_scalar"]["pass"])
    q_vec = bool(results["qk_f16_vec_dot"]["pass"])
    v_vec = bool(results["value_f16_vec_dot"]["pass"])
    if q_scalar and v_scalar:
        return "F16_CONVERSION_ONLY_SUFFICIENT"
    if q_vec and v_vec and not old["qk"]["pass"] and not old["value"]["pass"]:
        return "ATTRIBUTED_BOTH_ATTENTION_DOTS_TO_F16_VEC_DOT"
    if q_vec and not v_vec:
        return "ATTRIBUTED_QK_TO_F16_VEC_DOT_ONLY"
    if v_vec and not q_vec:
        return "ATTRIBUTED_VALUE_REDUCTION_TO_F16_VEC_DOT_ONLY"
    improved = (
        results["qk_f16_vec_dot"]["nrmse"] < old["qk"]["nrmse"]
        or results["value_f16_vec_dot"]["nrmse"] < old["value"]["nrmse"]
    )
    return "PARTIAL_F16_VEC_DOT_ATTRIBUTION" if improved else "F16_VEC_DOT_HYPOTHESIS_REJECTED"


def manifest_payload(logical: str) -> Path:
    manifest = json.loads((SOURCE_TRACE / "manifest.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest["payloads"] if item.get("logical") == logical and item.get("kind") == "full"]
    if len(matches) != 1:
        raise DiagnosticError(f"missing or duplicate source payload: {logical}")
    item = matches[0]; path = SOURCE_TRACE.parent / item["path"]
    if not path.is_file() or path.stat().st_size != item["byte_count"] or sha256_file(path) != item["sha256"]:
        raise DiagnosticError(f"source payload validation failed: {logical}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists(): raise DiagnosticError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True); started = time.perf_counter()
    record: dict[str, Any] = {"schema": "strat01_f16_vec_dot_diagnostic_v1", "status": "VOID_F16_VEC_DOT_DIAGNOSTIC", "started_utc": utc_now(), "donor_executions": 0, "commands": [], "errors": []}
    try:
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)], cwd=ROOT, check=False)
        if dirty.returncode: raise DiagnosticError("diagnostic implementation or protocol differs from HEAD")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        if sha256_file(PRIOR) != PRIOR_SHA: raise DiagnosticError("prior attention-stage adjudication identity mismatch")
        if sha256_file(PINNED_LLAMA / "src/llama-graph.cpp") != PINNED_GRAPH_SHA: raise DiagnosticError("pinned graph source mismatch")
        llama_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PINNED_LLAMA, text=True, capture_output=True, check=True).stdout.strip()
        llama_dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PINNED_LLAMA, text=True, capture_output=True, check=True).stdout.strip()
        if llama_head != PINNED_HEAD or llama_dirty: raise DiagnosticError("pinned llama.cpp checkout mismatch")
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        old = {"qk": prior["results"]["raw_qk"], "value": prior["results"]["captured_softmax_value_reduction"]}

        qcur_path, kcur_path = manifest_payload("Qcur-0"), manifest_payload("Kcur-0")
        kq_path, softmax_path, kqv_path = manifest_payload("kq-0"), manifest_payload("kq_soft_max-0"), manifest_payload("kqv-0")
        if sha256_file(qcur_path) != "4aca21f044acf71404ef0a7a000ed7b1bfe894efa82c5c1e49f7a5764cc6314b" or sha256_file(kcur_path) != "2860d9791b620d19788b8112e5366424be21669153eb1ec3255fd5a6c1b167f3": raise DiagnosticError("Q/K identity mismatch")
        captured_kq = callback_to_token_head(load_f32(kq_path, COUNT_PADDED, "9c5d5e73b21b5b92482c0ba396a49a0dba1be3d5bb99be332911f56787e2f74c"), 256).reshape(8, 32, 256)[:, :, :8].copy().reshape(COUNT_SCORES)
        padded_softmax = callback_to_token_head(load_f32(softmax_path, COUNT_PADDED, "8def8f8d6969dab39987e915084a74cb0c766c9d842db8b272886c248d60092e"), 256)
        true_kqv = callback_to_token_head(load_f32(kqv_path, COUNT_LATENT, "3922f34159f499dd26788acb7d9600d72fded004425392946bc09b8098fd88df"), 512)
        mapped = output / "mapped"; mapped.mkdir()
        mapped_kq = mapped / "kq-0.occupied.token_head_slot.f32le"; captured_kq.astype("<f4", copy=False).tofile(mapped_kq)
        mapped_softmax = mapped / "kq_soft_max-0.padded.token_head_slot.f32le"; padded_softmax.astype("<f4", copy=False).tofile(mapped_softmax)
        mapped_kqv = mapped / "kqv-0.token_head.f32le"; true_kqv.astype("<f4", copy=False).tofile(mapped_kqv)
        if sha256_file(mapped_kq) != "121c689214a7bcf2e709b8de896df14aabcb8fb00580bab297231b8887a3b009" or sha256_file(mapped_kqv) != "541183b4eb5992cfcfc711fe3c9f179ab41685106ba9cfe2aa7099f45f821312": raise DiagnosticError("mapped target identity mismatch")

        tests = run_command([sys.executable, "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_f16_vec_dot_diagnostic"], output, "model_free_tests")
        record["commands"].append(tests); require_ok(tests, "model-free tests")
        build = output / "build"; command = run_command([sys.executable, str(BUILDER), "--build-dir", str(build)], output, "build_helper", timeout=3600)
        record["commands"].append(command); require_ok(command, "helper build")
        helpers = list((build / "cmake-build").rglob("strat01_f16_vec_dot_diagnostic.exe"))
        if len(helpers) != 1: raise DiagnosticError("cannot resolve F16 vec-dot helper")
        helper = helpers[0]; command = run_command([str(helper), "--selftest"], output, "helper_selftest")
        record["commands"].append(command); require_ok(command, "helper self-test")
        products = output / "products"; products.mkdir()
        command = run_command([str(helper), "--qcur", str(qcur_path), "--kcur", str(kcur_path), "--padded-softmax", str(mapped_softmax), "--output-dir", str(products)], output, "offline_f16_vec_dot", timeout=3600)
        record["commands"].append(command); require_ok(command, "F16 vec-dot diagnostic")
        q_outputs = {name: load_f32(products / file, COUNT_SCORES) for name, file in {"scalar": "qk_f16_scalar.f32le", "vec": "qk_f16_vec_dot.f32le", "mutated": "qk_mutated.f32le"}.items()}
        v_outputs = {name: load_f32(products / file, COUNT_LATENT) for name, file in {"scalar": "value_f16_scalar.f32le", "vec": "value_f16_vec_dot.f32le", "mutated": "value_mutated.f32le"}.items()}
        project_f16, pinned_f16 = products / "project_f16.bin", products / "pinned_f16.bin"
        f16_exact = project_f16.stat().st_size == pinned_f16.stat().st_size and project_f16.read_bytes() == pinned_f16.read_bytes()
        results = {
            "qk_f16_scalar": gate(q_outputs["scalar"], captured_kq), "qk_f16_vec_dot": gate(q_outputs["vec"], captured_kq),
            "value_f16_scalar": gate(v_outputs["scalar"], true_kqv), "value_f16_vec_dot": gate(v_outputs["vec"], true_kqv),
            "qk_mutated": gate(q_outputs["mutated"], captured_kq), "value_mutated": gate(v_outputs["mutated"], true_kqv),
        }
        controls = {"source_identity_and_zero_donor": True, "model_free_tests_and_selftest": True, "f16_conversion_bytes_exact": f16_exact,
            "q_mutation_fires": not bool(results["qk_mutated"]["pass"]), "probability_mutation_fires": not bool(results["value_mutated"]["pass"])}
        record.update({"status": classify(results, controls, old), "finished_utc": utc_now(), "seconds": time.perf_counter() - started,
            "identity": {"git_head": head, "source_adjudication": {"path": str(PRIOR), "sha256": PRIOR_SHA}, "llama_cpp": {"head": llama_head, "graph_sha256": PINNED_GRAPH_SHA},
                "helper": {"path": str(helper), "sha256": sha256_file(helper)}, "critical_source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS}},
            "controls": controls, "old_f16_by_f32_controls": old, "results": results,
            "outputs": {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in [mapped_kq, mapped_softmax, mapped_kqv] + [item for item in products.iterdir() if item.is_file()]},
            "non_claims": ["production attention repair", "Rung 2B", "quality", "RAM", "speed"]})
    except Exception as error:
        record["errors"].append(str(error)); record["finished_utc"] = utc_now(); record["seconds"] = time.perf_counter() - started
    (output / "adjudication.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(record["status"])
    if record["errors"]: print(record["errors"][0], file=sys.stderr)
    return 0 if record["status"] != "VOID_F16_VEC_DOT_DIAGNOSTIC" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except DiagnosticError as error: raise SystemExit(f"run_strat01_f16_vec_dot_diagnostic: {error}")
