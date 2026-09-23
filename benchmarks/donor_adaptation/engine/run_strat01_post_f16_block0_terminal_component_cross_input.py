#!/usr/bin/env python3
"""Qualify and run the frozen post-F16 block-0 terminal-component split."""
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

from benchmarks.donor_adaptation.engine import run_strat01_post_f16_layer1_start_cross_input as post
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as q4base

HERE, ENGINE, MODEL = post.HERE, post.ENGINE, post.MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_post_f16_block0_terminal_component_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_post_f16_block0_terminal_component_cross_input.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_post_f16_block0_terminal_component_cross_input_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_post_f16_block0_terminal_component_cross_input_apparatus_20260923"
PREDECESSOR = HERE / "results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_recovery1_20260923/adjudication.json"
PREDECESSOR_SHA = "766061a9f54b601b8f54f08b6aaf6d533928aa19638f04ac2dd6cc23f865c09b"
REFERENCE_ROOT = HERE / "results/strat01_gigachat_engine_rung2b_repair2_20260921/pinned_reference/prefill8"
REFERENCE_COMPONENTS = {
    "reference_ffn_inp": (REFERENCE_ROOT / "ffn_inp-0.full.f32le", "baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1"),
    "reference_ffn_out": (REFERENCE_ROOT / "ffn_out-0.full.f32le", "f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4"),
    "reference_l_out": (REFERENCE_ROOT / "l_out-0.full.f32le", "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"),
}
CURRENT_L_OUT = post.CURRENT_ROOT / "prefill8_l_out-0.f32"
ARMS = ("current_attention_reference_ffn", "reference_attention_current_ffn")
ORIGINS = {ARMS[0]: ("current", "reference"), ARMS[1]: ("reference", "current")}
EXPECTED_COUNTS = {"mode": "pinned-generic-f64", "qk_invocations": 5_760, "value_invocations": 655_360}
ORDER = post.ORDER
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + p.stem for p in sorted(HERE.glob("test_strat01_*.py")))


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
        "rung2a": post.r2c.RUNG2A_HEADER, "rung2b": ROOT / "benchmarks/phase60/strat01_gguf_rung2b.h",
        "rung2c": post.r2c.RUNG2C_HEADER, "f16_dot": ROOT / "benchmarks/phase60/strat01_f16_vector_dot.h",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing post-F16 terminal-component source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise DiagnosticError("post-F16 terminal-component sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise DiagnosticError(f"untracked post-F16 terminal-component source: {path}")


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8"); header = HEADER.read_text(encoding="utf-8"); protocol = PROTOCOL.read_text(encoding="utf-8")
    return {
        "cli_registered": "--strat01-post-f16-block0-terminal-component-cross-input" in engine,
        "production_block0_reused": all(name in header for name in ("strat01_r2a_build_upstream_range", "strat01_r2a_run_schedule", "strat01_r2b_run")),
        "production_layer1_reused": "strat01_postf16_run_full" in header and "strat01_postf16_run_attention" in header,
        "no_local_quantized_dot": "static float strat01_q4k_q8k_dot" not in header and "static float strat01_q6k_q8k_dot" not in header,
        "exact_helper_accounted": "strat01_f16vec_reset_counts" in header and "strat01_f16v_write_counts" in header,
        "zero_graph_contract": "donor_graph_executions\\\":0" in header and "reference_graph_executions\\\":0" in header,
        "protocol_frozen_before_implementation": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol,
    }


def _load(path: Path, name: str) -> np.ndarray:
    kind = "<i4" if name in post.r2c.I32_NAMES else "<f4"; values = np.fromfile(path, dtype=kind); count = math.prod(post.r2c.SHAPES[name])
    if values.size != count or (name not in post.r2c.I32_NAMES and not bool(np.isfinite(values).all())):
        raise DiagnosticError(f"invalid payload: {name}")
    return values


def validate_frozen() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Path]]:
    if sha(PREDECESSOR) != PREDECESSOR_SHA:
        raise DiagnosticError("post-F16 layer1 predecessor hash mismatch")
    predecessor = read_json(PREDECESSOR, "post-F16 layer1 predecessor")
    if predecessor.get("status") != "POST_F16_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT" or predecessor.get("errors") != [] or predecessor.get("mode") != "offline-recovery" or predecessor.get("diagnostic_invocations") != 0 or predecessor.get("donor_graph_executions") != 0 or predecessor.get("reference_graph_executions") != 0:
        raise DiagnosticError("post-F16 layer1 predecessor state mismatch")
    reference, current, _ = post.validate_frozen(); paths: dict[str, Path] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, (path, digest) in REFERENCE_COMPONENTS.items():
        if not path.is_file() or path.stat().st_size != 49_152 or sha(path) != digest:
            raise DiagnosticError(f"reference component identity mismatch: {name}")
        paths[name] = path.resolve(strict=True); arrays[name] = np.fromfile(path, dtype="<f4")
    if not CURRENT_L_OUT.is_file() or CURRENT_L_OUT.stat().st_size != 49_152 or sha(CURRENT_L_OUT) != post.CURRENT_START_SHA:
        raise DiagnosticError("post-F16 current l_out-0 identity mismatch")
    paths["current_l_out"] = CURRENT_L_OUT.resolve(strict=True)
    rebuilt = (arrays["reference_ffn_inp"] + arrays["reference_ffn_out"]).astype("<f4", copy=False)
    if rebuilt.tobytes() != arrays["reference_l_out"].tobytes():
        raise DiagnosticError("reference component reconstruction mismatch")
    return reference, current, paths


def validate_output_map(root: Path, items: Any, arm: str) -> dict[str, np.ndarray]:
    if not isinstance(items, dict) or list(items) != ORDER:
        raise DiagnosticError(f"{arm} output ordering mismatch")
    values: dict[str, np.ndarray] = {}
    for name in ORDER:
        item = items[name]; size = math.prod(post.r2c.SHAPES[name]) * 4
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != size:
            raise DiagnosticError(f"{arm}/{name} output schema mismatch")
        path = post.r2c.base.contained_file(root, item["path"], size, f"{arm}/{name}")
        if sha(path) != item["sha256"]:
            raise DiagnosticError(f"{arm}/{name} output hash mismatch")
        values[name] = _load(path, name)
    return values


def validate_report(root: Path, model: Path, sources: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    report = read_json(root / "strat01_post_f16_block0_terminal_component_cross_input.json", "post-F16 terminal-component C report")
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "tensors", "current_components", "compositions", "outputs", "controls", "homogeneous_replay", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-post-f16-block0-terminal-component-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise DiagnosticError("post-F16 terminal-component report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": post.r2c.base.EXPECTED_BYTES, "sha256": post.r2c.base.EXPECTED_SHA256}:
        raise DiagnosticError("post-F16 terminal-component model report mismatch")
    expected_inputs = {
        "reference_ffn_inp": {"path": str(paths["reference_ffn_inp"]), "bytes": 49_152, "sha256": REFERENCE_COMPONENTS["reference_ffn_inp"][1]},
        "reference_ffn_out": {"path": str(paths["reference_ffn_out"]), "bytes": 49_152, "sha256": REFERENCE_COMPONENTS["reference_ffn_out"][1]},
        "reference_l_out": {"path": str(paths["reference_l_out"]), "bytes": 49_152, "sha256": REFERENCE_COMPONENTS["reference_l_out"][1]},
        "current_l_out": {"path": str(paths["current_l_out"]), "bytes": 49_152, "sha256": post.CURRENT_START_SHA},
    }
    if report["inputs"] != expected_inputs:
        raise DiagnosticError("post-F16 terminal-component input report mismatch")
    expected_names = ["token_embd.weight", "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_kv_a_mqa.weight", "blk.0.attn_kv_a_norm.weight", "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight", "blk.0.attn_output.weight", "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_kv_a_norm.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.1.attn_output.weight", "blk.1.ffn_norm.weight", "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias", "blk.1.ffn_up_exps.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight", "blk.1.ffn_up_shexp.weight", "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_down_shexp.weight"]
    if [item.get("name") for item in report["tensors"]] != expected_names or any(set(item) != {"name", "type", "offset", "file_offset", "span"} for item in report["tensors"]):
        raise DiagnosticError("post-F16 terminal-component tensor descriptors mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None or report["homogeneous_replay"] != {"reference": True, "current": True}:
        raise DiagnosticError("post-F16 terminal-component source/execution contract mismatch")
    components: dict[str, np.ndarray] = {}
    for name in ("ffn_inp-0", "ffn_out-0", "l_out-0"):
        item = report["current_components"].get(name); path = post.r2c.base.contained_file(root, item["path"], 49_152, f"current/{name}")
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != 49_152 or sha(path) != item["sha256"]:
            raise DiagnosticError(f"current component report mismatch: {name}")
        components[name] = np.fromfile(path, dtype="<f4")
    if components["l_out-0"].tobytes() != Path(paths["current_l_out"]).read_bytes() or (components["ffn_inp-0"] + components["ffn_out-0"]).astype("<f4", copy=False).tobytes() != components["l_out-0"].tobytes():
        raise DiagnosticError("post-F16 current homogeneous replay mismatch")
    if list(report["compositions"]) != list(ARMS) or list(report["outputs"]) != list(ARMS):
        raise DiagnosticError("post-F16 terminal-component arm ordering mismatch")
    ref_inp = np.fromfile(paths["reference_ffn_inp"], dtype="<f4"); ref_out = np.fromfile(paths["reference_ffn_out"], dtype="<f4")
    sums = {ARMS[0]: (components["ffn_inp-0"] + ref_out).astype("<f4", copy=False), ARMS[1]: (ref_inp + components["ffn_out-0"]).astype("<f4", copy=False)}
    for arm in ARMS:
        expected = {"ffn_inp_origin": ORIGINS[arm][0], "ffn_out_origin": ORIGINS[arm][1], "bytes": 49_152, "sha256": hashlib.sha256(sums[arm].tobytes()).hexdigest()}
        if report["compositions"].get(arm) != expected:
            raise DiagnosticError(f"post-F16 composition mismatch: {arm}")
    values = {arm: validate_output_map(root, report["outputs"][arm], arm) for arm in ARMS}
    controls: dict[str, np.ndarray] = {}
    if set(report["controls"]) != {"hybrid_token6_negated", "hybrid_rows0_7_swapped"}:
        raise DiagnosticError("post-F16 terminal-component control set mismatch")
    for name, item in report["controls"].items():
        path = post.r2c.base.contained_file(root, item["path"], 196_608, name)
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != 196_608 or sha(path) != item["sha256"]:
            raise DiagnosticError(f"post-F16 terminal-component control mismatch: {name}")
        controls[name] = np.fromfile(path, dtype="<f4")
    counts = read_json(root / "strat01_f16_vector_counts.json", "post-F16 terminal-component helper counts")
    if counts != EXPECTED_COUNTS:
        raise DiagnosticError(f"post-F16 terminal-component helper count mismatch: {counts!r}")
    return values, controls, components, counts, report


def adjudicate(values: dict[str, dict[str, np.ndarray]], controls: dict[str, np.ndarray], components: dict[str, np.ndarray], counts: dict[str, Any], report: dict[str, Any], reference: dict[str, np.ndarray], paths: dict[str, Path]) -> dict[str, Any]:
    judgments: dict[str, Any] = {arm: {name: post.judged(values[arm][name], reference[name], name) for name in ORDER} for arm in ARMS}
    first_failures = {arm: next((name for name in ORDER[1:] if not judgments[arm][name]["pass"]), None) for arm in ARMS}
    control_judgments = {name: post.r2c.base.judged(value, values[ARMS[0]]["kqv_out-1"], post.r2c.base.GENERAL_LIMITS) for name, value in controls.items()}
    if any(item["pass"] for item in control_judgments.values()):
        raise DiagnosticError("post-F16 terminal-component causal control did not reject")
    mutation_refused: dict[str, bool] = {}
    identities = {**{name: (path, REFERENCE_COMPONENTS[name][1]) for name, path in paths.items() if name.startswith("reference_")}, "current_l_out": (paths["current_l_out"], post.CURRENT_START_SHA)}
    for name, (path, digest) in identities.items():
        data = bytearray(Path(path).read_bytes()); data[len(data)//2] ^= 1; mutation_refused[name] = hashlib.sha256(data).hexdigest() != digest
    swapped = copy.deepcopy(report["compositions"]); swapped[ARMS[0]]["ffn_inp_origin"] = "reference"; origin_swap_rejected = swapped[ARMS[0]] != report["compositions"][ARMS[0]]
    if not all(mutation_refused.values()) or not origin_swap_rejected:
        raise DiagnosticError("post-F16 terminal-component identity control failed")
    inp_fails = first_failures[ARMS[0]] is not None; out_fails = first_failures[ARMS[1]] is not None
    if inp_fails and out_fails: status = "POST_F16_BLOCK0_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
    elif inp_fails: status = "POST_F16_BLOCK0_FFN_INP_RESIDUAL_SUFFICIENT"
    elif out_fails: status = "POST_F16_BLOCK0_FFN_OUT_RESIDUAL_SUFFICIENT"
    else: status = "POST_F16_BLOCK0_TERMINAL_RESIDUAL_JOINT_ONLY"
    return {"status": status, "first_failures": first_failures, "arm_judgments": judgments, "current_component_sha256": {name: report["current_components"][name]["sha256"] for name in report["current_components"]}, "control_judgments": control_judgments, "controls": {"reference_homogeneous_byte_exact": True, "current_homogeneous_byte_exact": True, "mutated_inputs_refused": mutation_refused, "origin_swap_rejected": origin_swap_rejected, "helper_counts_exact": counts == EXPECTED_COUNTS}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--model", type=Path, default=MODEL); parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true"); args = parser.parse_args()
    model = args.model.resolve(); output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True); started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter(); status = "VOID_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}; sources: dict[str, Any] = {}; result: dict[str, Any] = {"status": "NOT_RUN"}; report: dict[str, Any] = {}; frozen_paths: dict[str, Path] = {}; compiler = shutil.which("clang"); binary: Path | None = None; diagnostic_invocations = 0
    try:
        sources = source_inventory(); reference, current, frozen_paths = validate_frozen(); controls = source_controls()
        if not all(controls.values()): raise DiagnosticError("post-F16 terminal-component source controls failed")
        if not compiler: raise DiagnosticError("clang unavailable")
        binary = output / "engine_post_f16_block0_terminal_component.exe"; commands["compile"] = post.r2c.base.run_command([compiler, *post.r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile", timeout=600); post.r2c.base.require_ok(commands["compile"], "compile")
        commands["python_tests"] = post.r2c.base.run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], output=output, label="all_strat01_unittests", timeout=1800); post.r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
        for index, option in enumerate((*q4base.SELFTESTS, "--strat01-f16-vector-parity-selftest", "--strat01-post-f16-layer1-start-cross-input-selftest", "--strat01-post-f16-block0-terminal-component-cross-input-selftest")):
            label=f"selftest_{index:02d}"; commands[label]=post.r2c.base.run_command([str(binary), option], output=output, label=label, timeout=300); post.r2c.base.require_ok(commands[label], option)
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size != post.r2c.base.EXPECTED_BYTES or sha(model) != post.r2c.base.EXPECTED_SHA256: raise DiagnosticError("accepted artifact identity mismatch")
            root = output / "diagnostic"; root.mkdir(); diagnostic_invocations = 1
            command = [str(binary), "--strat01-post-f16-block0-terminal-component-cross-input", str(model), "--reference-ffn-inp", str(frozen_paths["reference_ffn_inp"]), "--reference-ffn-out", str(frozen_paths["reference_ffn_out"]), "--reference-l-out", str(frozen_paths["reference_l_out"]), "--current-l-out", str(frozen_paths["current_l_out"]), "--out-dir", str(root)]
            commands["diagnostic"] = post.r2c.base.run_command(command, output=output, label="diagnostic", timeout=21600); post.r2c.base.require_ok(commands["diagnostic"], "post-F16 terminal-component diagnostic")
            sources = source_inventory(); values, causal, components, counts, report = validate_report(root, model, sources, frozen_paths); result = adjudicate(values, causal, components, counts, report, reference, frozen_paths); status = result["status"]
    except (DiagnosticError, post.DiagnosticError, post.r2c.RunnerError, post.r2c.base.RunnerError, q4base.ParityError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-started, "git_head_observed" if args.apparatus_only else "git_head": post.r2c.base.git_value(["git","rev-parse","HEAD"]), "source_hashes": sources, "predecessor": {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}, "frozen_inputs": {name: str(path) for name, path in frozen_paths.items()}, "artifact": {"path": str(model), "expected_bytes": post.r2c.base.EXPECTED_BYTES, "expected_sha256": post.r2c.base.EXPECTED_SHA256}, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_post_f16_block0_terminal_component_cross_input_v1", "status": status, "errors": errors, "diagnostic_invocations": diagnostic_invocations, "donor_graph_executions": 0, "reference_graph_executions": 0, "source_controls": source_controls() if sources else {}, "adjudication": result, "c_report": report, "non_claims": ["old component split", "reference/Rung-2C graph rerun", "later layers", "tokenizer/logits/generation", "quality", "RAM", "rate"], "provenance": provenance}
    post.r2c.base.write_json(output / "adjudication.json", record); print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2)); return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status.startswith("POST_F16_BLOCK0_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
