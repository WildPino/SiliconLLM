#!/usr/bin/env python3
"""Qualify and propagate the pinned generic-F64 F16-dot semantics."""
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

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2c as r2c
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as q4base

HERE = Path(__file__).resolve().parent
ENGINE = r2c.ENGINE
F16_DOT_HEADER = ROOT / "benchmarks/phase60/strat01_f16_vector_dot.h"
PARITY_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_f16_vector_parity.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_f16_vector_propagation.py"
MODEL = r2c.MODEL
REFERENCE_ROOT = HERE / "results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference"
OLD_C_ROOT = HERE / "results/strat01_gigachat_engine_reference_generic_propagation_20260923/c_engine"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_f16_vector_propagation_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_f16_vector_propagation_apparatus_20260923"

F16_ADJUDICATION = HERE / "results/strat01_gigachat_engine_f16_vec_dot_repair2_20260921/adjudication.json"
PROPAGATION_ADJUDICATION = HERE / "results/strat01_gigachat_engine_reference_generic_propagation_20260923/adjudication.json"
Q6_ADJUDICATION = HERE / "results/strat01_gigachat_engine_layer1_q6_cross_input_20260923/adjudication.json"
PREDECESSORS = {
    "f16_vector_dot": (F16_ADJUDICATION, "d9691cbcd569681f970e5e4af553cec75a7f86ec47f66acbec2a111288239e0a"),
    "reference_generic_propagation": (PROPAGATION_ADJUDICATION, "79a19806fcf80df3e4d8a84335eaaecfc7beb9aef543f021dd1da01f6558b9c3"),
    "layer1_q6_cross_input": (Q6_ADJUDICATION, "db9b478e208c35460d4bd1ae54eb56cfb2a8106361b19f0b7f1f024699509148"),
}
MANIFESTS = {
    "old_c_prefill8": (OLD_C_ROOT / "prefill8_manifest.json", "517470cef63b1dd640d24c3fc0743a327d7dc160fad4ba3917148cb5c554190c"),
    "old_c_cached7p1": (OLD_C_ROOT / "cached7p1_manifest.json", "fb48cdef9e764dd8aa672ee7d372751b90a7716d556ffa5c7209eb21148238c0"),
    "reference_prefill8": (REFERENCE_ROOT / "prefill8/manifest.json", "d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451"),
    "reference_cached7p1": (REFERENCE_ROOT / "cached7p1/manifest.json", "ad214e8d38ee51a3b0f0629bdadf7cc692f6806499e1347c90d46aba937a39b9"),
}

HISTORICAL_ROOT = HERE / "results/strat01_gigachat_engine_f16_vec_dot_repair2_20260921"
STAGE_A_INPUTS = {
    "qcur": (HERE / "results/strat01_gigachat_engine_attention_stage_diagnostic_20260921/pinned_reference/prefill8/Qcur-0.full.f32le", 589824, "4aca21f044acf71404ef0a7a000ed7b1bfe894efa82c5c1e49f7a5764cc6314b"),
    "kcur": (HERE / "results/strat01_gigachat_engine_attention_stage_diagnostic_20260921/pinned_reference/prefill8/Kcur-0.full.f32le", 18432, "2860d9791b620d19788b8112e5366424be21669153eb1ec3255fd5a6c1b167f3"),
    "padded_softmax": (HISTORICAL_ROOT / "mapped/kq_soft_max-0.padded.token_head_slot.f32le", 262144, "6ececd173c490ac5b0d099021850724c43dce1d1bb00c46be28500cb4397efb2"),
}
STAGE_A_EXACT = {
    "qk_vector": ("qk_vector.f32le", 8192, "121c689214a7bcf2e709b8de896df14aabcb8fb00580bab297231b8887a3b009"),
    "qk_scalar": ("qk_scalar.f32le", 8192, "5b4caf1ebab46a5f6340dc695a72f8f08145f5f8817081e7cec10bceedaaae1d"),
    "value_vector": ("value_vector.f32le", 524288, "541183b4eb5992cfcfc711fe3c9f179ab41685106ba9cfe2aa7099f45f821312"),
    "value_scalar": ("value_scalar.f32le", 524288, "b77004cfaeb0c96588df64d79e9df091e7a3b75273475f610105d106f9d56bbb"),
    "conversion_stream": ("conversion_stream.f16le", 435240, "7cdec0744fb5c9dcc6225dd70a0c952283128d4f8668d9861131b868379defbc"),
}
HISTORICAL_EXACT = {
    "qk_vector": HISTORICAL_ROOT / "products/qk_f16_vec_dot.f32le",
    "qk_scalar": HISTORICAL_ROOT / "products/qk_f16_scalar.f32le",
    "value_vector": HISTORICAL_ROOT / "products/value_f16_vec_dot.f32le",
    "value_scalar": HISTORICAL_ROOT / "products/value_f16_scalar.f32le",
    "conversion_stream": HISTORICAL_ROOT / "products/pinned_f16.bin",
}
STAGE_A_ALL = {
    **{name: spec[:2] for name, spec in STAGE_A_EXACT.items()},
    "qk_mutated": ("qk_mutated.f32le", 8192),
    "value_mutated": ("value_mutated.f32le", 524288),
}
EXPECTED_COUNTS = {"mode": "pinned-generic-f64", "qk_invocations": 4608, "value_invocations": 524288}
REPORT_NAMES = (
    "kqv_out-1", "ffn_inp-1", "ffn_norm-1",
    "ffn_moe_up-1", "ffn_moe_gate-1", "ffn_moe_swiglu-1", "ffn_moe_down-1",
    "ffn_up-1", "ffn_gate-1", "ffn_swiglu-1", "ffn_shexp-1",
    "ffn_out-1", "l_out-1",
)
TEST_MODULES = tuple(
    "benchmarks.donor_adaptation.engine." + path.stem
    for path in sorted(HERE.glob("test_strat01_*.py"))
)


class PropagationError(RuntimeError):
    pass


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PropagationError(f"{label} missing or malformed: {exc}") from exc


def sha(path: Path) -> str:
    return r2c.base.sha256_file(path)


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "engine": ENGINE, "f16_dot_header": F16_DOT_HEADER, "parity_header": PARITY_HEADER,
        "q4_header": ROOT / "benchmarks/phase60/strat01_q4k_q8k.h",
        "rung2a_header": r2c.RUNG2A_HEADER, "rung2b_header": r2c.RUNG2B_HEADER,
        "rung2c_header": r2c.RUNG2C_HEADER, "rung2c_runner": Path(r2c.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise PropagationError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise PropagationError("F16 propagation implementation/protocol differs from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise PropagationError(f"untracked F16 propagation source: {path}")


def validate_predecessors() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, (path, expected) in PREDECESSORS.items():
        if sha(path) != expected:
            raise PropagationError(f"predecessor hash mismatch: {name}")
        records[name] = read_json(path, name)
    f16, propagation, q6 = (records[name] for name in PREDECESSORS)
    if f16.get("status") != "F16_CONVERSION_ONLY_SUFFICIENT" or f16.get("errors") != [] or f16.get("donor_executions") != 0:
        raise PropagationError("F16-vector predecessor state mismatch")
    if propagation.get("status") != "FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION" or propagation.get("errors") != [] or propagation.get("donor_graph_executions") != 2 or propagation.get("reference_graph_executions") != 0:
        raise PropagationError("reference-generic propagation predecessor state mismatch")
    if q6.get("status") != "LAYER1_SWIGLU_RESIDUAL_SUFFICIENT" or q6.get("errors") != [] or q6.get("donor_graph_executions") != 0 or q6.get("reference_graph_executions") != 0:
        raise PropagationError("layer-1 Q6 predecessor state mismatch")
    manifest_hashes: dict[str, str] = {}
    for name, (path, expected) in MANIFESTS.items():
        manifest_hashes[name] = sha(path)
        if manifest_hashes[name] != expected:
            raise PropagationError(f"manifest hash mismatch: {name}")
    for name, (_, size, expected) in STAGE_A_INPUTS.items():
        path = STAGE_A_INPUTS[name][0]
        if not path.is_file() or path.stat().st_size != size or sha(path) != expected:
            raise PropagationError(f"Stage-A input identity mismatch: {name}")
    for name, path in HISTORICAL_EXACT.items():
        _, size, expected = STAGE_A_EXACT[name]
        if not path.is_file() or path.stat().st_size != size or sha(path) != expected:
            raise PropagationError(f"historical Stage-A output identity mismatch: {name}")
    return {
        "adjudications": {name: {"path": str(path), "sha256": expected} for name, (path, expected) in PREDECESSORS.items()},
        "manifests": manifest_hashes,
        "stage_a_inputs": {name: {"path": str(spec[0]), "bytes": spec[1], "sha256": spec[2]} for name, spec in STAGE_A_INPUTS.items()},
        "historical_stage_a_outputs": {name: {"path": str(HISTORICAL_EXACT[name]), "bytes": STAGE_A_EXACT[name][1], "sha256": STAGE_A_EXACT[name][2]} for name in STAGE_A_EXACT},
    }


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8")
    dot = F16_DOT_HEADER.read_text(encoding="utf-8")
    parity = PARITY_HEADER.read_text(encoding="utf-8")
    r2a = r2c.RUNG2A_HEADER.read_text(encoding="utf-8")
    r2c_text = r2c.RUNG2C_HEADER.read_text(encoding="utf-8")
    return {
        "rung2c_controls_retained": all(r2c.source_controls().values()),
        "pinned_generic_f64_semantics": "double out=0.0" in dot and "return (float)out" in dot,
        "scalar_control_retained": "strat01_f16vec_scalar_dot" in dot and "qks[o]=strat01_f16vec_scalar_dot" in parity,
        "qk_production_dispatch": "strat01_f16vec_dot(576U,q16,a->cache[s])" in r2a and "++strat01_f16vec_qk_invocations" in r2a,
        "value_production_dispatch": "strat01_f16vec_dot(256U,prob16,values)" in r2a and "++strat01_f16vec_value_invocations" in r2a,
        "counts_reset_and_emitted": "strat01_f16vec_reset_counts()" in r2c_text and "strat01_f16v_write_counts(out_dir,error)" in r2c_text,
        "config_names_exact_mode": "f16_dot=pinned-generic-f64" in r2c_text and "f16_dot=pinned-generic-f64" in r2c.EXPECTED_C_CONFIG,
        "model_free_cli_registered": "--strat01-f16-vector-parity" in engine and "--strat01-f16-vector-parity-selftest" in engine,
    }


def load_f32(path: Path, count: int, label: str) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != count or not bool(np.isfinite(values).all()):
        raise PropagationError(f"invalid F32 payload: {label}")
    return values


def validate_stage_a(root: Path, sources: dict[str, dict[str, str]]) -> dict[str, Any]:
    report = read_json(root / "strat01_f16_vector_parity.json", "Stage-A report")
    required = {"command", "state", "self_certifies_pass", "outputs", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if not isinstance(report, dict) or set(report) != required or report["command"] != "--strat01-f16-vector-parity" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise PropagationError("Stage-A report schema/state mismatch")
    if set(report["outputs"]) != set(STAGE_A_ALL):
        raise PropagationError("Stage-A output set mismatch")
    resolved: dict[str, Path] = {}
    for name, (leaf, size) in STAGE_A_ALL.items():
        item = report["outputs"].get(name)
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"} or item["bytes"] != size:
            raise PropagationError(f"Stage-A output schema mismatch: {name}")
        path = r2c.base.contained_file(root, item["path"], size, f"Stage-A {name}")
        if path.name != leaf or sha(path) != item["sha256"]:
            raise PropagationError(f"Stage-A output identity mismatch: {name}")
        resolved[name] = path
    exact = {name: sha(resolved[name]) == expected for name, (_, _, expected) in STAGE_A_EXACT.items()}
    if not all(exact.values()):
        raise PropagationError("Stage-A exact replay mismatch")
    qk_vector = load_f32(resolved["qk_vector"], 2048, "qk_vector")
    value_vector = load_f32(resolved["value_vector"], 131072, "value_vector")
    mutation_metrics = {
        "qk": r2c.base.judged(load_f32(resolved["qk_mutated"], 2048, "qk_mutated"), qk_vector, r2c.base.CONTINUITY_LIMITS),
        "value": r2c.base.judged(load_f32(resolved["value_mutated"], 131072, "value_mutated"), value_vector, r2c.base.CONTINUITY_LIMITS),
    }
    controls = {
        "all_frozen_outputs_exact": all(exact.values()),
        "qk_vector_distinct_from_scalar": sha(resolved["qk_vector"]) != sha(resolved["qk_scalar"]),
        "value_vector_distinct_from_scalar": sha(resolved["value_vector"]) != sha(resolved["value_scalar"]),
        "qk_mutation_rejects": not mutation_metrics["qk"]["pass"],
        "value_mutation_rejects": not mutation_metrics["value"]["pass"],
        "source_binding": report["engine_source_sha256"] == sources["engine"]["sha256"] and report["diagnostic_source_sha256"] == sources["parity_header"]["sha256"],
        "zero_graphs": report["donor_graph_executions"] == 0 and report["reference_graph_executions"] == 0,
        "not_a_speed_claim": report["timing_or_rate_claim"] is None,
    }
    if report["compiler_family"] != "clang" or not all(controls.values()):
        raise PropagationError("Stage-A causal/source control failure")
    return {"report": report, "exact_replay": exact, "controls": controls, "mutation_metrics": mutation_metrics}


def validate_counts(c_root: Path) -> dict[str, Any]:
    counts = read_json(c_root / "strat01_f16_vector_counts.json", "F16 helper counts")
    if counts != EXPECTED_COUNTS:
        raise PropagationError(f"F16 helper invocation mismatch: {counts!r}")
    return counts


def tensor_hash(manifest: dict[str, Any], name: str) -> str:
    matches = [item.get("sha256") for item in manifest.get("tensors", []) if item.get("name") == name]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise PropagationError(f"manifest tensor hash missing: {name}")
    return matches[0]


def intervention_controls(candidate_meta: dict[str, Any], counts: dict[str, Any]) -> dict[str, bool]:
    old_manifests = {arm: read_json(OLD_C_ROOT / f"{arm}_manifest.json", f"old C {arm} manifest") for arm in r2c.base.ARMS}
    changed = {
        arm: tensor_hash(candidate_meta["manifests"][arm], "kqv_out-1") != tensor_hash(old_manifests[arm], "kqv_out-1")
        for arm in r2c.base.ARMS
    }
    return {
        "exact_helper_counts": counts == EXPECTED_COUNTS,
        "prefill_attention_hash_changed": changed["prefill8"],
        "cached_attention_hash_changed": changed["cached7p1"],
    }


def adjudicate(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray], candidate_meta: dict[str, Any], reference_meta: dict[str, Any], controls: dict[str, bool], counts: dict[str, Any]) -> dict[str, Any]:
    inherited = r2c.adjudicate(candidate, reference, candidate_meta, reference_meta, controls)
    start_failures = [item for item in inherited["failures"] if item.startswith("start_state/")]
    if len(start_failures) not in (0, len(r2c.base.ARMS)):
        raise PropagationError("asymmetric inherited start-state failure")
    failures = [item for item in inherited["failures"] if not item.startswith("start_state/")]
    intervention = intervention_controls(candidate_meta, counts)
    if not all(intervention.values()):
        raise PropagationError("changed-coordinate intervention control failed")
    inherited.update({
        "status": "PASS_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION" if not failures else "FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION",
        "failures": failures,
        "superseded_start_state_failures": start_failures,
        "intervention_controls": intervention,
        "helper_invocation_counts": counts,
    })
    return inherited


def checkpoint_detail(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    old = read_json(PROPAGATION_ADJUDICATION, "old propagation adjudication")["adjudication"]["checkpoint_results"]
    old_by_key = {(item["arm"], item["checkpoint"]): item for item in old}
    result: dict[str, Any] = {}
    for arm in r2c.base.ARMS:
        arm_result: dict[str, Any] = {}
        for name in REPORT_NAMES:
            limits = r2c.base.TERMINAL_LIMITS if name in r2c.TERMINALS else r2c.base.GENERAL_LIMITS
            c = candidate[f"{arm}/{name}"].reshape(8, -1)
            r = reference[f"{arm}/{name}"].reshape(8, -1)
            new = r2c.base.judged(c.ravel(), r.ravel(), limits)
            per_token = [dict(r2c.base.judged(c[token], r[token], limits), token=token) for token in range(8)]
            old_item = old_by_key.get((arm, name))
            if old_item is None:
                raise PropagationError(f"old checkpoint result missing: {arm}/{name}")
            arm_result[name] = {
                "old": {key: old_item[key] for key in ("nrmse", "normalized_max", "pass")},
                "new": new,
                "delta_nrmse": new["nrmse"] - old_item["nrmse"],
                "delta_normalized_max": new["normalized_max"] - old_item["normalized_max"],
                "per_token": per_token,
            }
        result[arm] = arm_result
    return result


def route_tables(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for arm in r2c.base.ARMS:
        c_ids = candidate[f"{arm}/ffn_moe_topk-1"].reshape(8, 4)
        r_ids = reference[f"{arm}/ffn_moe_topk-1"].reshape(8, 4)
        c_weights = candidate[f"{arm}/ffn_moe_weights_norm-1"].reshape(8, 4)
        r_weights = reference[f"{arm}/ffn_moe_weights_norm-1"].reshape(8, 4)
        tables[arm] = {
            "candidate_ids": c_ids.tolist(), "reference_ids": r_ids.tolist(),
            "ids_exact": bool(np.array_equal(c_ids, r_ids)),
            "candidate_normalized_weights": c_weights.tolist(),
            "reference_normalized_weights": r_weights.tolist(),
            "weight_metrics": r2c.base.judged(c_weights.ravel(), r_weights.ravel(), r2c.base.GENERAL_LIMITS),
        }
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    predecessors: dict[str, Any] = {}
    controls: dict[str, bool] = {}
    stage_a: dict[str, Any] = {"status": "NOT_RUN"}
    result: dict[str, Any] = {"status": "NOT_RUN"}
    detail: dict[str, Any] = {}
    routes: dict[str, Any] = {}
    candidate_meta: dict[str, Any] = {}
    reference_meta: dict[str, Any] = {}
    compiler = shutil.which("clang")
    binary: Path | None = None
    donor_invocations = donor_graphs = 0
    try:
        sources = source_inventory()
        predecessors = validate_predecessors()
        controls = source_controls()
        if not all(controls.values()):
            raise PropagationError("source controls failed")
        if not compiler:
            raise PropagationError("clang is unavailable")
        binary = output / "engine_f16_vector_propagation.exe"
        commands["compile_c"] = r2c.base.run_command([compiler, *r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile_c", timeout=600)
        r2c.base.require_ok(commands["compile_c"], "C build")
        commands["python_tests"] = r2c.base.run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], output=output, label="all_strat01_unittests", timeout=1800)
        r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
        for index, option in enumerate((*q4base.SELFTESTS, "--strat01-f16-vector-parity-selftest")):
            label = f"selftest_{index:02d}"
            commands[label] = r2c.base.run_command([str(binary), option], output=output, label=label, timeout=300)
            r2c.base.require_ok(commands[label], option)
        stage_a_root = output / "stage_a"
        stage_a_root.mkdir()
        stage_a_command = [str(binary), "--strat01-f16-vector-parity", "--qcur", str(STAGE_A_INPUTS["qcur"][0]), "--kcur", str(STAGE_A_INPUTS["kcur"][0]), "--padded-softmax", str(STAGE_A_INPUTS["padded_softmax"][0]), "--out-dir", str(stage_a_root)]
        commands["stage_a_model_free"] = r2c.base.run_command(stage_a_command, output=output, label="stage_a_model_free", timeout=600)
        r2c.base.require_ok(commands["stage_a_model_free"], "Stage-A model-free identity gate")
        stage_a = validate_stage_a(stage_a_root, sources)
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size != r2c.base.EXPECTED_BYTES or sha(model) != r2c.base.EXPECTED_SHA256:
                raise PropagationError("accepted artifact identity mismatch")
            reference, reference_meta = r2c.validate_reference(REFERENCE_ROOT, model)
            c_root = output / "c_engine"
            c_root.mkdir()
            donor_invocations = 1
            commands["accepted_artifact_c_engine"] = r2c.base.run_command([str(binary), "--strat01-gguf-rung2c", str(model), "--out-dir", str(c_root)], output=output, label="accepted_artifact_c_engine", timeout=21600)
            donor_graphs = r2c.completed_graph_count(commands["accepted_artifact_c_engine"])
            r2c.base.require_ok(commands["accepted_artifact_c_engine"], "C producer")
            if donor_graphs != 2:
                raise PropagationError("C producer did not emit exactly two graph-completion markers")
            counts = validate_counts(c_root)
            candidate, candidate_meta = r2c.validate_c(c_root, sources, model)
            result = adjudicate(candidate, reference, candidate_meta, reference_meta, controls, counts)
            detail = checkpoint_detail(candidate, reference)
            routes = route_tables(candidate, reference)
            status = result["status"]
    except (PropagationError, r2c.RunnerError, r2c.base.RunnerError, q4base.ParityError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")

    provenance = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "git_head_observed" if args.apparatus_only else "git_head": r2c.base.git_value(["git", "rev-parse", "HEAD"]),
        "source_hashes": sources, "predecessors": predecessors,
        "artifact": {"path": str(model), "expected_bytes": r2c.base.EXPECTED_BYTES, "expected_sha256": r2c.base.EXPECTED_SHA256},
        "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler},
        "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None},
        "commands": commands,
    }
    record = {
        "schema": "strat01_engine_f16_vector_reduction_propagation_v1", "status": status, "errors": errors,
        "donor_producer_invocations": donor_invocations, "donor_graph_executions": donor_graphs,
        "reference_producer_invocations": 0, "reference_graph_executions": 0,
        "source_controls": controls, "stage_a": stage_a, "adjudication": result,
        "old_vs_new_checkpoint_detail": detail, "route_tables": routes,
        "candidate_metadata": r2c.metadata_for_record(candidate_meta), "reference_metadata": r2c.metadata_for_record(reference_meta),
        "non_claims": ["later layers", "tokenizer/logits/generation", "C-path language-model quality", "RAM", "rate", "SPEED_LEDGER"],
        "provenance": provenance,
    }
    r2c.base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION", "PASS_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION", "FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
