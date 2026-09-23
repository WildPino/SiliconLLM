#!/usr/bin/env python3
"""Qualify and run the frozen post-F16 complete layer-1-start cross-input."""
from __future__ import annotations

import argparse
import copy
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
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_post_f16_layer1_start_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_LAYER1_START_CROSS_INPUT_PROTOCOL_20260923.md"
RECOVERY_PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_LAYER1_START_CROSS_INPUT_RECOVERY_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_post_f16_layer1_start_cross_input.py"
MODEL = r2c.MODEL
REFERENCE_ROOT = HERE / "results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference"
CURRENT_RUN = HERE / "results/strat01_gigachat_engine_f16_vector_propagation_20260923"
CURRENT_ROOT = CURRENT_RUN / "c_engine"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_apparatus_repair1_20260923"
DEFAULT_RECOVERY = HERE / "results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_recovery1_20260923"
RESULT_SHA = "512e3ea7754c5e8fa067dab5692a8a192879be26895547cbedbb6023a896edcf"
REFERENCE_MANIFEST_SHA = "d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451"
CURRENT_MANIFEST_SHA = "1b79c30281107937845a09499cc13b505e49721fdc00ab40e9518c762a22fb68"
CURRENT_CACHED_MANIFEST_SHA = "0b98c6ca454b260eb4dde27b9341bb489a93ea5d57d8a9fde8333cbdae475026"
REFERENCE_START_SHA = "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"
CURRENT_START_SHA = "7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4"
EXPECTED_COUNTS = {"mode": "pinned-generic-f64", "qk_invocations": 4_608, "value_invocations": 524_288}
RAW_VOID_SHA = "6ed2d017ca36c166929f658b4a79a77a242adffb5afb39ee986bdda10494e6bf"
RAW_REPORT_SHA = "7c49afcf18272575124455d70ed82979e931ee7c29b8b6c02119d41b78538a3f"
RAW_COUNTS_SHA = "19892ac3af937f601854246657f91b9ff0042c168fa7ed7a8761b4a24a8553da"
RAW_HEAD = "8f03fa3963723fc2083cc2d608437d1a487c9666"
RAW_BINARY_SHA = "6dccd8941166f55beed2a4020ed0e87046f07a0857e5b8e8aae3c4fc81561629"
ORDER = list(r2c.SHAPES)
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + p.stem for p in sorted(HERE.glob("test_strat01_*.py")))


class DiagnosticError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return r2c.base.sha256_file(path)


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "recovery_protocol": RECOVERY_PROTOCOL,
        "engine": ENGINE, "header": HEADER, "rung2a": r2c.RUNG2A_HEADER,
        "rung2c": r2c.RUNG2C_HEADER,
        "f16_dot": ROOT / "benchmarks/phase60/strat01_f16_vector_dot.h",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise DiagnosticError("post-F16 implementation/protocol differs from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise DiagnosticError(f"untracked post-F16 source: {path}")


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    recovery_protocol = RECOVERY_PROTOCOL.read_text(encoding="utf-8")
    return {
        "cli_registered": "--strat01-post-f16-layer1-start-cross-input" in engine,
        "production_attention_reused": "strat01_r2c_build_attention_range" in header and "strat01_r2a_run_schedule" in header,
        "production_moe_reused": "strat01_r2c_run_moe" in header,
        "no_local_q4_q6_reimplementation": "static float strat01_q4k_q8k_dot" not in header and "static float strat01_q6k_q8k_dot" not in header,
        "exact_helper_accounted": "strat01_f16vec_reset_counts" in header and "strat01_f16v_write_counts" in header,
        "zero_graph_contract": "donor_graph_executions\\\":0" in header and "reference_graph_executions\\\":0" in header,
        "protocol_frozen_before_implementation": "before implementation or execution" in protocol,
        "recovery_frozen_before_execution": "FROZEN BEFORE RECOVERY EXECUTION" in recovery_protocol,
    }


def _load(path: Path, name: str) -> np.ndarray:
    kind = "<i4" if name in r2c.I32_NAMES else "<f4"
    values = np.fromfile(path, dtype=kind)
    count = math.prod(r2c.SHAPES[name])
    if values.size != count or (name not in r2c.I32_NAMES and not bool(np.isfinite(values).all())):
        raise DiagnosticError(f"invalid payload: {name}")
    return values


def validate_frozen() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    result_path = CURRENT_RUN / "adjudication.json"
    if sha(result_path) != RESULT_SHA:
        raise DiagnosticError("F16 propagation adjudication hash mismatch")
    result = read_json(result_path, "F16 propagation result")
    if result.get("status") != "FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION" or result.get("errors") != [] or result.get("donor_graph_executions") != 2 or result.get("reference_graph_executions") != 0:
        raise DiagnosticError("F16 propagation predecessor state mismatch")

    ref_manifest_path = REFERENCE_ROOT / "prefill8/manifest.json"
    cur_manifest_path = CURRENT_ROOT / "prefill8_manifest.json"
    cached_manifest_path = CURRENT_ROOT / "cached7p1_manifest.json"
    if sha(ref_manifest_path) != REFERENCE_MANIFEST_SHA or sha(cur_manifest_path) != CURRENT_MANIFEST_SHA or sha(cached_manifest_path) != CURRENT_CACHED_MANIFEST_SHA:
        raise DiagnosticError("frozen manifest hash mismatch")
    ref_manifest = read_json(ref_manifest_path, "reference manifest")
    cur_manifest = read_json(cur_manifest_path, "current manifest")
    cached_manifest = read_json(cached_manifest_path, "current cached manifest")
    ref_items = {(item.get("logical"), item.get("kind")): item for item in ref_manifest.get("payloads", []) if isinstance(item, dict)}
    cur_items = {item.get("name"): item for item in cur_manifest.get("tensors", []) if isinstance(item, dict)}
    cached_items = {item.get("name"): item for item in cached_manifest.get("tensors", []) if isinstance(item, dict)}
    if set(cur_items) != set(ORDER) or set(cached_items) != set(ORDER):
        raise DiagnosticError("current checkpoint set mismatch")
    reference: dict[str, np.ndarray] = {}
    current: dict[str, np.ndarray] = {}
    paths: dict[str, str] = {}
    for name in ORDER:
        count = math.prod(r2c.SHAPES[name]); size = count * 4
        ref_item = ref_items.get((name, "full")); cur_item = cur_items[name]; cached_item = cached_items[name]
        if not isinstance(ref_item, dict) or ref_item.get("byte_count") != size or cur_item.get("byte_count") != size:
            raise DiagnosticError(f"frozen payload metadata mismatch: {name}")
        if cur_item.get("sha256") != cached_item.get("sha256"):
            raise DiagnosticError(f"post-F16 schedules differ: {name}")
        ref_path = r2c.base.contained_file(REFERENCE_ROOT, ref_item["path"], size, f"reference/{name}")
        cur_path = r2c.base.contained_file(CURRENT_ROOT, cur_item["path"], size, f"current/{name}")
        if sha(ref_path) != ref_item["sha256"] or sha(cur_path) != cur_item["sha256"]:
            raise DiagnosticError(f"frozen payload hash mismatch: {name}")
        reference[name] = _load(ref_path, name); current[name] = _load(cur_path, name)
        paths[f"reference/{name}"] = str(ref_path); paths[f"current/{name}"] = str(cur_path)
    if sha(Path(paths["reference/l_out-0"])) != REFERENCE_START_SHA or sha(Path(paths["current/l_out-0"])) != CURRENT_START_SHA:
        raise DiagnosticError("start-state identity mismatch")
    return reference, current, {"result": {"path": str(result_path), "sha256": RESULT_SHA}, "manifests": {"reference": REFERENCE_MANIFEST_SHA, "current_prefill": CURRENT_MANIFEST_SHA, "current_cached": CURRENT_CACHED_MANIFEST_SHA}, "paths": paths}


def validate_outputs(root: Path, items: Any, arm: str) -> dict[str, np.ndarray]:
    if not isinstance(items, dict) or list(items) != ORDER:
        raise DiagnosticError(f"{arm} output ordering mismatch")
    values: dict[str, np.ndarray] = {}
    for name in ORDER:
        item = items[name]; size = math.prod(r2c.SHAPES[name]) * 4
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != size:
            raise DiagnosticError(f"{arm}/{name} output schema mismatch")
        path = r2c.base.contained_file(root, item["path"], size, f"{arm}/{name}")
        if sha(path) != item["sha256"]:
            raise DiagnosticError(f"{arm}/{name} output hash mismatch")
        values[name] = _load(path, name)
    return values


def validate_report(root: Path, model: Path, sources: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    report = read_json(root / "strat01_post_f16_layer1_start_cross_input.json", "post-F16 C report")
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "tensors", "outputs", "controls", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-post-f16-layer1-start-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise DiagnosticError("post-F16 report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": r2c.base.EXPECTED_BYTES, "sha256": r2c.base.EXPECTED_SHA256}:
        raise DiagnosticError("post-F16 model report mismatch")
    if report["inputs"]["reference"]["sha256"] != REFERENCE_START_SHA or report["inputs"]["current"]["sha256"] != CURRENT_START_SHA or any(item["bytes"] != 49_152 for item in report["inputs"].values()):
        raise DiagnosticError("post-F16 input report mismatch")
    expected_names = ["blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_kv_a_norm.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.1.attn_output.weight", "blk.1.ffn_norm.weight", "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias", "blk.1.ffn_up_exps.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight", "blk.1.ffn_up_shexp.weight", "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_down_shexp.weight"]
    if [item.get("name") for item in report["tensors"]] != expected_names or any(set(item) != {"name", "type", "offset", "file_offset", "span"} for item in report["tensors"]):
        raise DiagnosticError("post-F16 tensor descriptors mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise DiagnosticError("post-F16 source/execution contract mismatch")
    if set(report["outputs"]) != {"reference_start", "current_start"}:
        raise DiagnosticError("post-F16 arm set mismatch")
    values = {arm: validate_outputs(root, report["outputs"][arm], arm) for arm in ("reference_start", "current_start")}
    if set(report["controls"]) != {"reference_token6_negated", "reference_rows0_7_swapped"}:
        raise DiagnosticError("post-F16 control set mismatch")
    controls: dict[str, np.ndarray] = {}
    for name, item in report["controls"].items():
        path = r2c.base.contained_file(root, item["path"], 196_608, name)
        if sha(path) != item["sha256"]:
            raise DiagnosticError(f"control hash mismatch: {name}")
        controls[name] = np.fromfile(path, dtype="<f4")
    counts = read_json(root / "strat01_f16_vector_counts.json", "post-F16 helper counts")
    if counts != EXPECTED_COUNTS:
        raise DiagnosticError(f"post-F16 helper count mismatch: {counts!r}")
    return values, controls, counts, report


def validate_recovery_source(root: Path) -> dict[str, Any]:
    adjudication_path = root / "adjudication.json"
    report_path = root / "diagnostic/strat01_post_f16_layer1_start_cross_input.json"
    counts_path = root / "diagnostic/strat01_f16_vector_counts.json"
    if sha(adjudication_path) != RAW_VOID_SHA or sha(report_path) != RAW_REPORT_SHA or sha(counts_path) != RAW_COUNTS_SHA:
        raise DiagnosticError("post-F16 recovery source hash mismatch")
    raw = read_json(adjudication_path, "post-F16 raw VOID")
    expected_error = "post-F16 helper count mismatch: {'mode': 'pinned-generic-f64', 'qk_invocations': 4608, 'value_invocations': 524288}"
    diagnostic = raw.get("provenance", {}).get("commands", {}).get("diagnostic", {})
    binary = raw.get("provenance", {}).get("binary", {})
    if (
        raw.get("status") != "VOID_POST_F16_LAYER1_START_CROSS_INPUT"
        or raw.get("errors") != [expected_error]
        or raw.get("diagnostic_invocations") != 1
        or raw.get("donor_graph_executions") != 0
        or raw.get("reference_graph_executions") != 0
        or raw.get("provenance", {}).get("git_head") != RAW_HEAD
        or binary.get("sha256") != RAW_BINARY_SHA
        or diagnostic.get("returncode") != 0
    ):
        raise DiagnosticError("post-F16 recovery source state mismatch")
    return {
        "root": str(root),
        "adjudication_sha256": RAW_VOID_SHA,
        "c_report_sha256": RAW_REPORT_SHA,
        "counts_sha256": RAW_COUNTS_SHA,
        "producer_git_head": RAW_HEAD,
        "producer_binary_sha256": RAW_BINARY_SHA,
        "inherited_diagnostic_invocations": 1,
    }


def judged(candidate: np.ndarray, reference: np.ndarray, name: str) -> dict[str, Any]:
    if name in r2c.I32_NAMES:
        exact = bool(np.array_equal(candidate, reference)); return {"exact": exact, "pass": exact}
    limits = r2c.base.TERMINAL_LIMITS if name in r2c.TERMINALS else r2c.base.GENERAL_LIMITS
    result = r2c.base.judged(candidate, reference, limits)
    rows = candidate.reshape(8, -1); refs = reference.reshape(8, -1)
    result["per_token"] = [dict(r2c.base.judged(rows[i], refs[i], limits), token=i) for i in range(8)]
    return result


def adjudicate(values: dict[str, dict[str, np.ndarray]], controls: dict[str, np.ndarray], reference: dict[str, np.ndarray], current: dict[str, np.ndarray], counts: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    replay: dict[str, bool] = {}; judgments: dict[str, Any] = {}
    for name in ORDER:
        replay[name] = values["current_start"][name].tobytes() == current[name].tobytes()
        if not replay[name] or report["outputs"]["current_start"][name]["sha256"] != sha(Path(CURRENT_ROOT / f"prefill8_{name}.{'i32' if name in r2c.I32_NAMES else 'f32'}")):
            raise DiagnosticError(f"post-F16 current replay mismatch: {name}")
        judgments[name] = judged(values["reference_start"][name], reference[name], name)
    control_judgments = {name: r2c.base.judged(value, reference["kqv_out-1"], r2c.base.GENERAL_LIMITS) for name, value in controls.items()}
    if any(item["pass"] for item in control_judgments.values()):
        raise DiagnosticError("post-F16 causal control did not reject")
    mutated = {}
    for label, digest in (("reference", REFERENCE_START_SHA), ("current", CURRENT_START_SHA)):
        data = bytearray((Path(REFERENCE_ROOT / "prefill8/l_out-0.full.f32le") if label == "reference" else Path(CURRENT_ROOT / "prefill8_l_out-0.f32")).read_bytes()); data[len(data)//2] ^= 1
        mutated[label] = not (len(data) == 49_152 and __import__("hashlib").sha256(data).hexdigest() == digest)
    swapped = copy.deepcopy(report["outputs"]); swapped["reference_start"], swapped["current_start"] = swapped["current_start"], swapped["reference_start"]
    label_swap_rejected = swapped["current_start"]["l_out-0"]["sha256"] != CURRENT_START_SHA
    if not all(mutated.values()) or not label_swap_rejected:
        raise DiagnosticError("post-F16 identity control failed")
    first_failure = next((name for name in ORDER[1:] if not judgments[name]["pass"]), None)
    status = "POST_F16_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT" if first_failure is None else "POST_F16_LAYER1_LOCAL_RESIDUAL_AT_" + first_failure.upper().replace("-", "_")
    return {"status": status, "first_failure": first_failure, "reference_start_vs_reference": judgments, "current_replay_byte_exact": replay, "control_judgments": control_judgments, "controls": {"mutated_inputs_refused": mutated, "arm_label_swap_rejected": label_swap_rejected, "helper_counts_exact": counts == EXPECTED_COUNTS}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL); parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true"); parser.add_argument("--recover-existing", action="store_true"); args = parser.parse_args()
    if args.apparatus_only and args.recover_existing:
        raise SystemExit("--apparatus-only and --recover-existing are mutually exclusive")
    default_output = DEFAULT_RECOVERY if args.recover_existing else (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)
    model = args.model.resolve(); output = (args.output_dir or default_output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True); started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_POST_F16_LAYER1_START_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}; sources: dict[str, Any] = {}; frozen_meta: dict[str, Any] = {}; recovery_meta: dict[str, Any] = {}; result: dict[str, Any] = {"status": "NOT_RUN"}; report: dict[str, Any] = {}; compiler = shutil.which("clang"); binary: Path | None = None; diagnostic_invocations = 0
    try:
        sources = source_inventory(); reference, current, frozen_meta = validate_frozen(); controls = source_controls()
        if not all(controls.values()): raise DiagnosticError("post-F16 source controls failed")
        if args.recover_existing:
            clean_sources_at_head(sources)
            recovery_meta = validate_recovery_source(DEFAULT_OUTPUT)
            values, causal, counts, report = validate_report(DEFAULT_OUTPUT / "diagnostic", model, sources)
            result = adjudicate(values, causal, reference, current, counts, report); status = result["status"]
        else:
            if not compiler: raise DiagnosticError("clang unavailable")
            binary = output / "engine_post_f16_layer1_start.exe"
            commands["compile"] = r2c.base.run_command([compiler, *r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile", timeout=600); r2c.base.require_ok(commands["compile"], "compile")
            commands["python_tests"] = r2c.base.run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], output=output, label="all_strat01_unittests", timeout=1800); r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
            for index, option in enumerate((*q4base.SELFTESTS, "--strat01-f16-vector-parity-selftest", "--strat01-post-f16-layer1-start-cross-input-selftest")):
                label=f"selftest_{index:02d}"; commands[label]=r2c.base.run_command([str(binary), option], output=output, label=label, timeout=300); r2c.base.require_ok(commands[label], option)
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        elif not args.recover_existing:
            clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size != r2c.base.EXPECTED_BYTES or sha(model) != r2c.base.EXPECTED_SHA256: raise DiagnosticError("accepted artifact identity mismatch")
            root = output / "diagnostic"; root.mkdir(); diagnostic_invocations = 1
            command = [str(binary), "--strat01-post-f16-layer1-start-cross-input", str(model), "--reference-input", frozen_meta["paths"]["reference/l_out-0"], "--current-input", frozen_meta["paths"]["current/l_out-0"], "--out-dir", str(root)]
            commands["diagnostic"] = r2c.base.run_command(command, output=output, label="diagnostic", timeout=21600); r2c.base.require_ok(commands["diagnostic"], "post-F16 diagnostic")
            sources = source_inventory(); values, causal, counts, report = validate_report(root, model, sources); result = adjudicate(values, causal, reference, current, counts, report); status = result["status"]
    except (DiagnosticError, r2c.RunnerError, r2c.base.RunnerError, q4base.ParityError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-started, "git_head_observed" if args.apparatus_only else "git_head": r2c.base.git_value(["git","rev-parse","HEAD"]), "source_hashes": sources, "frozen_inputs": frozen_meta, "recovery_source": recovery_meta, "artifact": {"path": str(model), "expected_bytes": r2c.base.EXPECTED_BYTES, "expected_sha256": r2c.base.EXPECTED_SHA256}, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_post_f16_layer1_start_cross_input_v1", "status": status, "errors": errors, "mode": "offline-recovery" if args.recover_existing else ("apparatus" if args.apparatus_only else "scientific"), "diagnostic_invocations": diagnostic_invocations, "donor_graph_executions": 0, "reference_graph_executions": 0, "source_controls": source_controls() if sources else {}, "adjudication": result, "c_report": report, "non_claims": ["graph rerun", "later layers", "tokenizer/logits/generation", "quality", "RAM", "rate"], "provenance": provenance}
    r2c.base.write_json(output / "adjudication.json", record); print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2)); return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION", "POST_F16_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT"} or status.startswith("POST_F16_LAYER1_LOCAL_RESIDUAL_AT_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
