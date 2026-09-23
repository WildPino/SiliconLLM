#!/usr/bin/env python3
"""Qualify and run the frozen post-F16 block-0 FFN operator-chain split."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_ffn_swiglu_cross_input as sw
from benchmarks.donor_adaptation.engine import run_strat01_post_f16_block0_terminal_component_cross_input as predecessor
from benchmarks.donor_adaptation.engine import run_strat01_post_f16_layer1_start_cross_input as post
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as q4base

HERE, ENGINE, MODEL = post.HERE, post.ENGINE, post.MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_post_f16_block0_ffn_operator_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_post_f16_block0_ffn_operator_cross_input.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_post_f16_block0_ffn_operator_cross_input_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_post_f16_block0_ffn_operator_cross_input_apparatus_repair1_20260923"
PREDECESSOR = HERE / "results/strat01_gigachat_engine_post_f16_block0_terminal_component_cross_input_20260923/adjudication.json"
PREDECESSOR_SHA = "4d30ffb5d79d0682b1522157caf27bded8edd655bfe4e7bdf8c5353a59d515fb"
OLD_SW = HERE / "results/strat01_gigachat_engine_ffn_swiglu_cross_input_repair1_20260923/adjudication.json"
OLD_SW_SHA = "0f897c815c1f5d5dc3ede8f0aacca687df325f10a2d4b504be679f65b42922f5"
OLD_DOWN = HERE / "results/strat01_gigachat_engine_ffn_down_cross_input_20260923/adjudication.json"
OLD_DOWN_SHA = "6d3ab2b6a1097502060380350238940d434d38b2b41bcdff8d7c586f5f8d05d0"

INPUTS: dict[str, tuple[Path, str, int]] = {
    "reference_gate": sw.INPUTS["reference_gate"],
    "reference_up": sw.INPUTS["reference_up"],
    "reference_swiglu": sw.INPUTS["reference_swiglu"],
    "reference_ffn_inp": sw.INPUTS["reference_ffn_inp"],
}
ARMS = ("q6_only", "expression_q6_replay")
CONTROLS = ("reference_swiglu_token6_negated", "reference_swiglu_rows0_7_swapped")
ARM_IDENTITIES = {
    "q6_only": {
        "swiglu": "de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef",
        "ffn_out": "f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce",
        "l_out": "a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11",
    },
    "expression_q6_replay": {
        "swiglu": "ac2b46108a7e18ddebdb1b3a26462bca0faebea536361a5ca4069c4fa7ddcf40",
        "ffn_out": "7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684",
        "l_out": post.CURRENT_START_SHA,
    },
}
EXPECTED_COUNTS = {"mode": "pinned-generic-f64", "qk_invocations": 4_608, "value_invocations": 524_288}
EXPECTED_Q6_ARMS = 4
ORDER = post.ORDER
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + path.stem for path in sorted(HERE.glob("test_strat01_*.py")))


class DiagnosticError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return post.sha(path)


def read_json(path: Path, label: str) -> Any:
    return post.read_json(path, label)


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "engine": ENGINE, "header": HEADER, "post_header": post.HEADER,
        "terminal_header": predecessor.HEADER, "swiglu_header": sw.HEADER,
        "down_header": sw.down.HEADER, "rung2a": post.r2c.RUNG2A_HEADER,
        "rung2b": ROOT / "benchmarks/phase60/strat01_gguf_rung2b.h",
        "rung2c": post.r2c.RUNG2C_HEADER,
        "f16_dot": ROOT / "benchmarks/phase60/strat01_f16_vector_dot.h",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing post-F16 FFN operator source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise DiagnosticError("post-F16 FFN operator sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise DiagnosticError(f"untracked post-F16 FFN operator source: {path}")


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8"); header = HEADER.read_text(encoding="utf-8"); protocol = PROTOCOL.read_text(encoding="utf-8")
    return {
        "cli_registered": "--strat01-post-f16-block0-ffn-operator-cross-input" in engine,
        "production_swiglu_reused": "strat01_sw_compute" in header,
        "production_q6_reused": "strat01_r2b_q6_matmul_batch" in header,
        "production_layer1_reused": "strat01_postf16_run_full" in header and "strat01_postf16_write_arm" in header,
        "no_local_quantized_dot": "static float strat01_q4k_q8k_dot" not in header and "static float strat01_q6k_q8k_dot" not in header,
        "exact_helper_accounted": "strat01_f16vec_reset_counts" in header and "strat01_f16v_write_counts" in header,
        "zero_graph_contract": "donor_graph_executions\\\":0" in header and "reference_graph_executions\\\":0" in header,
        "protocol_frozen_before_implementation": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol,
    }


def validate_frozen() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Path]]:
    records = ((PREDECESSOR, PREDECESSOR_SHA, "POST_F16_BLOCK0_FFN_OUT_RESIDUAL_SUFFICIENT"), (OLD_SW, OLD_SW_SHA, "GATE_AND_UP_RESIDUALS_INDEPENDENTLY_SUFFICIENT"), (OLD_DOWN, OLD_DOWN_SHA, "SWIGLU_INPUT_RESIDUAL_SUFFICIENT"))
    loaded: list[dict[str, Any]] = []
    for path, digest, status in records:
        if not path.is_file() or sha(path) != digest:
            raise DiagnosticError(f"predecessor identity mismatch: {path.name}")
        record = read_json(path, str(path)); loaded.append(record)
        if record.get("status") != status or record.get("errors") != [] or record.get("donor_graph_executions") != 0 or record.get("reference_graph_executions") != 0:
            raise DiagnosticError(f"predecessor state mismatch: {status}")
    predecessor_record, sw_record, down_record = loaded
    reference, current, _ = post.validate_frozen()
    paths: dict[str, Path] = {}
    for name, (path, digest, size) in INPUTS.items():
        if not path.is_file() or path.stat().st_size != size or sha(path) != digest:
            raise DiagnosticError(f"frozen input identity mismatch: {name}")
        paths[name] = path.resolve(strict=True)
    components = predecessor_record["adjudication"]["current_component_sha256"]
    if components != {"ffn_inp-0": INPUTS["reference_ffn_inp"][1], "ffn_out-0": ARM_IDENTITIES["expression_q6_replay"]["ffn_out"], "l_out-0": ARM_IDENTITIES["expression_q6_replay"]["l_out"]}:
        raise DiagnosticError("post-F16 component identity bridge mismatch")
    old_refref = sw_record["c_report"]["arms"]["reference_gate_reference_up"]
    if old_refref["swiglu"]["sha256"] != ARM_IDENTITIES["expression_q6_replay"]["swiglu"] or old_refref["ffn_out"]["sha256"] != ARM_IDENTITIES["expression_q6_replay"]["ffn_out"] or old_refref["l_out_sha256"] != ARM_IDENTITIES["expression_q6_replay"]["l_out"]:
        raise DiagnosticError("old reference/reference expression bridge mismatch")
    old_q6 = down_record["c_report"]["arms"]["reference_swiglu_current_q6"]
    if old_q6["ffn_out"]["sha256"] != ARM_IDENTITIES["q6_only"]["ffn_out"] or old_q6["l_out_sha256"] != ARM_IDENTITIES["q6_only"]["l_out"]:
        raise DiagnosticError("old Q6-only identity bridge mismatch")
    return reference, current, paths


def _load(path: Path, name: str) -> np.ndarray:
    kind = "<i4" if name in post.r2c.I32_NAMES else "<f4"; values = np.fromfile(path, dtype=kind); count = math.prod(post.r2c.SHAPES[name])
    if values.size != count or (name not in post.r2c.I32_NAMES and not bool(np.isfinite(values).all())):
        raise DiagnosticError(f"invalid payload: {name}")
    return values


def validate_payload(root: Path, item: Any, size: int, label: str) -> np.ndarray:
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"} or item["bytes"] != size:
        raise DiagnosticError(f"{label} payload schema mismatch")
    path = post.r2c.base.contained_file(root, item["path"], size, label)
    if sha(path) != item["sha256"]:
        raise DiagnosticError(f"{label} payload hash mismatch")
    values = np.fromfile(path, dtype="<f4")
    if values.size != size // 4 or not bool(np.isfinite(values).all()):
        raise DiagnosticError(f"{label} payload invalid")
    return values


def validate_arm(root: Path, item: Any, label: str) -> dict[str, Any]:
    if not isinstance(item, dict) or set(item) != {"swiglu", "ffn_out", "l_out", "checkpoints"}:
        raise DiagnosticError(f"{label} arm schema mismatch")
    return {
        "swiglu": validate_payload(root, item["swiglu"], 286_720, f"{label}/swiglu"),
        "ffn_out": validate_payload(root, item["ffn_out"], 49_152, f"{label}/ffn_out"),
        "l_out": validate_payload(root, item["l_out"], 49_152, f"{label}/l_out"),
        "checkpoints": predecessor.validate_output_map(root, item["checkpoints"], label),
    }


def validate_report(root: Path, model: Path, sources: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = read_json(root / "strat01_post_f16_block0_ffn_operator_cross_input.json", "post-F16 FFN operator C report")
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "down_tensor", "layer1_tensors", "arms", "controls", "q6_down_arms", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-post-f16-block0-ffn-operator-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise DiagnosticError("post-F16 FFN operator report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": post.r2c.base.EXPECTED_BYTES, "sha256": post.r2c.base.EXPECTED_SHA256}:
        raise DiagnosticError("post-F16 FFN operator model report mismatch")
    expected_inputs = {name: {"path": str(paths[name]), "bytes": size, "sha256": digest} for name, (_, digest, size) in INPUTS.items()}
    if report["inputs"] != expected_inputs:
        raise DiagnosticError("post-F16 FFN operator input report mismatch")
    if set(report["down_tensor"]) != {"name", "type", "offset", "file_offset", "span"} or report["down_tensor"]["name"] != "blk.0.ffn_down.weight" or report["down_tensor"]["type"] != 14:
        raise DiagnosticError("post-F16 FFN operator down descriptor mismatch")
    expected_names = ["blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_kv_a_norm.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.1.attn_output.weight", "blk.1.ffn_norm.weight", "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias", "blk.1.ffn_up_exps.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight", "blk.1.ffn_up_shexp.weight", "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_down_shexp.weight"]
    if [item.get("name") for item in report["layer1_tensors"]] != expected_names or any(set(item) != {"name", "type", "offset", "file_offset", "span"} for item in report["layer1_tensors"]):
        raise DiagnosticError("post-F16 FFN operator layer-1 descriptors mismatch")
    if list(report["arms"]) != list(ARMS) or list(report["controls"]) != list(CONTROLS) or report["q6_down_arms"] != EXPECTED_Q6_ARMS:
        raise DiagnosticError("post-F16 FFN operator arm accounting mismatch")
    values = {name: validate_arm(root, report["arms"][name], name) for name in ARMS}
    controls = {name: validate_arm(root, report["controls"][name], name) for name in CONTROLS}
    for arm in ARMS:
        for field in ("swiglu", "ffn_out", "l_out"):
            if report["arms"][arm][field]["sha256"] != ARM_IDENTITIES[arm][field]:
                raise DiagnosticError(f"post-F16 FFN operator identity mismatch: {arm}/{field}")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise DiagnosticError("post-F16 FFN operator source/execution contract mismatch")
    counts = read_json(root / "strat01_f16_vector_counts.json", "post-F16 FFN operator helper counts")
    if counts != EXPECTED_COUNTS:
        raise DiagnosticError(f"post-F16 FFN operator helper count mismatch: {counts!r}")
    return values, controls, {"report": report, "counts": counts}


def classify(q6_fails: bool, replay_fails: bool) -> str:
    if not replay_fails:
        raise DiagnosticError("expression+Q6 replay unexpectedly passes")
    return "POST_F16_BLOCK0_Q6_RESIDUAL_SUFFICIENT" if q6_fails else "POST_F16_BLOCK0_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT"


def adjudicate(values: dict[str, Any], controls: dict[str, Any], validated: dict[str, Any], reference: dict[str, np.ndarray], paths: dict[str, Path]) -> dict[str, Any]:
    judgments = {arm: {name: post.judged(values[arm]["checkpoints"][name], reference[name], name) for name in ORDER} for arm in ARMS}
    first_failures = {arm: next((name for name in ORDER[1:] if not judgments[arm][name]["pass"]), None) for arm in ARMS}
    predecessor_record = read_json(PREDECESSOR, "post-F16 terminal-component predecessor"); frozen_replay = predecessor_record["c_report"]["outputs"]["reference_attention_current_ffn"]
    replay_exact = {name: validated["report"]["arms"]["expression_q6_replay"]["checkpoints"][name]["sha256"] == frozen_replay[name]["sha256"] for name in ORDER}
    if not all(replay_exact.values()):
        raise DiagnosticError("expression+Q6 full-checkpoint replay mismatch")
    control_judgments = {control: {name: post.judged(controls[control]["checkpoints"][name], reference[name], name) for name in ORDER} for control in CONTROLS}
    if any(all(item[name]["pass"] for name in ORDER[1:]) for item in control_judgments.values()):
        raise DiagnosticError("post-F16 FFN operator causal control did not reject")
    mutation_refused: dict[str, bool] = {}
    for name, path in paths.items():
        data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1; mutation_refused[name] = hashlib.sha256(data).hexdigest() != INPUTS[name][1]
    swapped = copy.deepcopy(validated["report"]["arms"]); swapped["q6_only"], swapped["expression_q6_replay"] = swapped["expression_q6_replay"], swapped["q6_only"]
    origin_swap_rejected = swapped != validated["report"]["arms"]
    if not all(mutation_refused.values()) or not origin_swap_rejected:
        raise DiagnosticError("post-F16 FFN operator identity/origin control failed")
    q6_fails = first_failures["q6_only"] is not None; replay_fails = first_failures["expression_q6_replay"] is not None; status = classify(q6_fails, replay_fails)
    ref_sw = np.fromfile(paths["reference_swiglu"], dtype="<f4"); ref_out = np.fromfile(predecessor.REFERENCE_COMPONENTS["reference_ffn_out"][0], dtype="<f4")
    local = {arm: {"swiglu": sw.wide_judgment(values[arm]["swiglu"], ref_sw), "ffn_out": sw.down.direct_judgment(values[arm]["ffn_out"], ref_out), "l_out": post.judged(values[arm]["l_out"], reference["l_out-0"], "l_out-0")} for arm in ARMS}
    return {"status": status, "first_failures": first_failures, "local_judgments": local, "arm_judgments": judgments, "control_judgments": control_judgments, "controls": {"expression_q6_replay_exact": replay_exact, "mutated_inputs_refused": mutation_refused, "origin_swap_rejected": origin_swap_rejected, "helper_counts_exact": validated["counts"] == EXPECTED_COUNTS, "q6_down_arms_exact": validated["report"]["q6_down_arms"] == EXPECTED_Q6_ARMS}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--model", type=Path, default=MODEL); parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true"); args = parser.parse_args()
    model = args.model.resolve(); output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True); started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter(); status = "VOID_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}; sources: dict[str, Any] = {}; result: dict[str, Any] = {"status": "NOT_RUN"}; report: dict[str, Any] = {}; frozen_paths: dict[str, Path] = {}; compiler = shutil.which("clang"); binary: Path | None = None; diagnostic_invocations = 0
    try:
        sources = source_inventory(); reference, current, frozen_paths = validate_frozen(); controls = source_controls()
        if not all(controls.values()):
            raise DiagnosticError("post-F16 FFN operator source controls failed")
        if not compiler:
            raise DiagnosticError("clang unavailable")
        binary = output / "engine_post_f16_block0_ffn_operator.exe"; commands["compile"] = post.r2c.base.run_command([compiler, *post.r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile", timeout=600); post.r2c.base.require_ok(commands["compile"], "compile")
        commands["python_tests"] = post.r2c.base.run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], output=output, label="all_strat01_unittests", timeout=1800); post.r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
        selftests = (*q4base.SELFTESTS, "--strat01-f16-vector-parity-selftest", "--strat01-post-f16-layer1-start-cross-input-selftest", "--strat01-post-f16-block0-terminal-component-cross-input-selftest", "--strat01-post-f16-block0-ffn-operator-cross-input-selftest")
        for index, option in enumerate(selftests):
            label = f"selftest_{index:02d}"; commands[label] = post.r2c.base.run_command([str(binary), option], output=output, label=label, timeout=300); post.r2c.base.require_ok(commands[label], option)
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size != post.r2c.base.EXPECTED_BYTES or sha(model) != post.r2c.base.EXPECTED_SHA256:
                raise DiagnosticError("accepted artifact identity mismatch")
            root = output / "diagnostic"; root.mkdir(); diagnostic_invocations = 1
            command = [str(binary), "--strat01-post-f16-block0-ffn-operator-cross-input", str(model), "--reference-gate", str(frozen_paths["reference_gate"]), "--reference-up", str(frozen_paths["reference_up"]), "--reference-swiglu", str(frozen_paths["reference_swiglu"]), "--reference-ffn-inp", str(frozen_paths["reference_ffn_inp"]), "--out-dir", str(root)]
            commands["diagnostic"] = post.r2c.base.run_command(command, output=output, label="diagnostic", timeout=21600); post.r2c.base.require_ok(commands["diagnostic"], "post-F16 FFN operator diagnostic")
            sources = source_inventory(); values, causal, validated = validate_report(root, model, sources, frozen_paths); report = validated["report"]; result = adjudicate(values, causal, validated, reference, frozen_paths); status = result["status"]
    except (DiagnosticError, predecessor.DiagnosticError, post.DiagnosticError, post.r2c.RunnerError, post.r2c.base.RunnerError, q4base.ParityError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started, "git_head_observed" if args.apparatus_only else "git_head": post.r2c.base.git_value(["git", "rev-parse", "HEAD"]), "source_hashes": sources, "predecessors": [{"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}, {"path": str(OLD_SW), "sha256": OLD_SW_SHA}, {"path": str(OLD_DOWN), "sha256": OLD_DOWN_SHA}], "frozen_inputs": {name: str(path) for name, path in frozen_paths.items()}, "artifact": {"path": str(model), "expected_bytes": post.r2c.base.EXPECTED_BYTES, "expected_sha256": post.r2c.base.EXPECTED_SHA256}, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_post_f16_block0_ffn_operator_cross_input_v1", "status": status, "errors": errors, "diagnostic_invocations": diagnostic_invocations, "donor_graph_executions": 0, "reference_graph_executions": 0, "source_controls": source_controls() if sources else {}, "adjudication": result, "c_report": report, "non_claims": ["block-0 graph", "gate/up split", "old FFN-down/SwiGLU rerun", "later layers", "tokenizer/logits/generation", "quality", "RAM", "rate"], "provenance": provenance}
    post.r2c.base.write_json(output / "adjudication.json", record); print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2)); return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status.startswith("POST_F16_BLOCK0_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
