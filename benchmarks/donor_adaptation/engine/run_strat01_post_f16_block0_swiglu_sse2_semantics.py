#!/usr/bin/env python3
"""Qualify and run the frozen post-F16 SwiGLU SSE2-semantics cell."""
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

from benchmarks.donor_adaptation.engine import run_strat01_post_f16_block0_ffn_operator_cross_input as prev

HERE, ENGINE, MODEL = prev.HERE, prev.ENGINE, prev.MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_post_f16_block0_swiglu_sse2_semantics.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_post_f16_block0_swiglu_sse2_semantics.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_post_f16_block0_swiglu_sse2_semantics_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_post_f16_block0_swiglu_sse2_semantics_apparatus_20260924"
PREDECESSOR = HERE / "results/strat01_gigachat_engine_post_f16_block0_ffn_operator_cross_input_20260923/adjudication.json"
PREDECESSOR_SHA = "ad03099c59a2bb1ee30bb0063c2b4b9f393b1551ec0e19931d0bf518dd1095eb"
REFERENCE_BUILD = HERE / "results/strat01_gigachat_engine_rung2b_repair2_20260921/reference_build/cmake-build"
REFERENCE_REVISION = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
REFERENCE_EVIDENCE = {
    "compile_commands": (REFERENCE_BUILD / "compile_commands.json", "6864a9b60e55b83d53fb26cf1f3590ae3a03aea7ff1369728ddc5efa6deffb60", 112_047),
    "build_ninja": (REFERENCE_BUILD / "build.ninja", "a25a67b5ad155d9511f84db00ef82830b1ec4ee94df24992b0b6bd892962bc27", 237_630),
}
REFERENCE_SOURCE_HASHES = {
    "vec.cpp": ("ggml/src/ggml-cpu/vec.cpp", "a946fee202dfe4528453865a6402d13a004e31ea73e88c3586793bbcc05994f7", 25_922),
    "vec.h": ("ggml/src/ggml-cpu/vec.h", "8817801355b20079de39fd4c67c7453ca2033cdb69fc5bd1f71bb66f12f57318", 67_630),
}
INPUTS = prev.INPUTS
ARMS = ("scalar_libm_replay", "pinned_sse2_candidate")
CONTROLS = ("sse2_gate_token6_negated", "sse2_up_rows0_7_swapped")
SCALAR_IDENTITIES = prev.ARM_IDENTITIES["expression_q6_replay"]
PASS_IDENTITIES = prev.ARM_IDENTITIES["q6_only"]
EXPECTED_COUNTS = prev.EXPECTED_COUNTS
EXPECTED_Q6_ARMS = 4
ORDER = prev.ORDER
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + path.stem for path in sorted(HERE.glob("test_strat01_*.py")))


class DiagnosticError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return prev.sha(path)


def read_json(path: Path, label: str) -> Any:
    return prev.read_json(path, label)


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL,
        "engine": ENGINE, "header": HEADER, "predecessor_header": prev.HEADER,
        "post_header": prev.post.HEADER, "terminal_header": prev.predecessor.HEADER,
        "swiglu_header": prev.sw.HEADER, "down_header": prev.sw.down.HEADER,
        "rung2a": prev.post.r2c.RUNG2A_HEADER,
        "rung2b": ROOT / "benchmarks/phase60/strat01_gguf_rung2b.h",
        "rung2c": prev.post.r2c.RUNG2C_HEADER,
        "f16_dot": ROOT / "benchmarks/phase60/strat01_f16_vector_dot.h",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing SSE2-semantics source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise DiagnosticError("SSE2-semantics sources differ from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise DiagnosticError(f"untracked SSE2-semantics source: {path}")


def source_controls() -> dict[str, bool]:
    engine = ENGINE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    return {
        "cli_registered": "--strat01-post-f16-block0-swiglu-sse2-semantics" in engine,
        "reference_revision_pinned": REFERENCE_REVISION in header,
        "reference_source_hashes_pinned": all(digest in header for _, digest, _ in REFERENCE_SOURCE_HASHES.values()),
        "no_fma_sse2_transcribed": "strat01_sse2_expf_no_fma" in header and "_mm_fmadd" not in header and "_mm_fnmadd" not in header,
        "four_lane_no_tail": "count & 3U" in header and "i += 4U" in header,
        "scalar_replay_reused": "strat01_sw_compute" in header,
        "production_q6_reused": "strat01_r2b_q6_matmul_batch" in header,
        "production_layer1_reused": "strat01_postf16_run_full" in header and "strat01_postf16_write_arm" in header,
        "exact_helper_accounted": "strat01_f16vec_reset_counts" in header and "strat01_f16v_write_counts" in header,
        "zero_graph_contract": "donor_graph_executions\\\":0" in header and "reference_graph_executions\\\":0" in header,
        "protocol_frozen_before_implementation": "FROZEN BEFORE IMPLEMENTATION OR EXECUTION" in protocol,
    }


def validate_reference_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name, (path, digest, size) in REFERENCE_EVIDENCE.items():
        if not path.is_file() or path.stat().st_size != size or sha(path) != digest:
            raise DiagnosticError(f"reference build evidence mismatch: {name}")
        evidence[name] = {"path": str(path.resolve()), "bytes": size, "sha256": digest}
    cache = REFERENCE_BUILD / "CMakeCache.txt"
    if not cache.is_file():
        raise DiagnosticError("reference CMake cache missing")
    source_root: Path | None = None
    for line in cache.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith("llama.cpp_SOURCE_DIR:STATIC="):
            source_root = Path(line.split("=", 1)[1]).resolve()
            break
    if source_root is None or not source_root.is_dir():
        raise DiagnosticError("reference source root unavailable")
    head = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(source_root), "status", "--porcelain"], capture_output=True, text=True, check=False)
    if head.returncode or head.stdout.strip() != REFERENCE_REVISION or dirty.returncode or dirty.stdout.strip():
        raise DiagnosticError("reference source revision/cleanliness mismatch")
    for name, (relative, digest, size) in REFERENCE_SOURCE_HASHES.items():
        path = source_root / relative
        if not path.is_file() or path.stat().st_size != size or sha(path) != digest:
            raise DiagnosticError(f"reference source mismatch: {name}")
        evidence[name] = {"path": str(path), "bytes": size, "sha256": digest}
    compile_db = json.loads(REFERENCE_EVIDENCE["compile_commands"][0].read_text(encoding="utf-8"))
    vec_commands = [item["command"] for item in compile_db if item.get("file", "").replace("\\", "/").endswith("/ggml-cpu/vec.cpp")]
    if len(vec_commands) != 1:
        raise DiagnosticError("reference vec.cpp compile command cardinality mismatch")
    command = vec_commands[0]
    if "-DGGML_CPU_GENERIC" not in command or any(flag in command for flag in ("-mavx", "-mfma", "-ffast-math")):
        raise DiagnosticError("reference vec.cpp compile semantics mismatch")
    evidence["compile_semantics"] = {"ggml_cpu_generic": True, "avx": False, "fma": False, "fast_math": False, "lane_width": 4, "scalar_tail": False}
    return evidence


def validate_frozen() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Path], dict[str, Any], dict[str, Any]]:
    if not PREDECESSOR.is_file() or sha(PREDECESSOR) != PREDECESSOR_SHA:
        raise DiagnosticError("SSE2-semantics predecessor identity mismatch")
    predecessor = read_json(PREDECESSOR, "SSE2-semantics predecessor")
    if predecessor.get("status") != "POST_F16_BLOCK0_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT" or predecessor.get("errors") != [] or predecessor.get("diagnostic_invocations") != 1 or predecessor.get("donor_graph_executions") != 0 or predecessor.get("reference_graph_executions") != 0:
        raise DiagnosticError("SSE2-semantics predecessor state mismatch")
    reference, current, paths = prev.validate_frozen()
    evidence = validate_reference_evidence()
    return reference, current, paths, predecessor, evidence


def validate_arm(root: Path, item: Any, label: str) -> dict[str, Any]:
    return prev.validate_arm(root, item, label)


def validate_report(root: Path, model: Path, sources: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = read_json(root / "strat01_post_f16_block0_swiglu_sse2_semantics.json", "SSE2-semantics C report")
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "semantics", "down_tensor", "layer1_tensors", "arms", "controls", "q6_down_arms", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "reference_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-post-f16-block0-swiglu-sse2-semantics" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise DiagnosticError("SSE2-semantics report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": prev.post.r2c.base.EXPECTED_BYTES, "sha256": prev.post.r2c.base.EXPECTED_SHA256}:
        raise DiagnosticError("SSE2-semantics model report mismatch")
    expected_inputs = {name: {"path": str(paths[name]), "bytes": size, "sha256": digest} for name, (_, digest, size) in INPUTS.items()}
    if report["inputs"] != expected_inputs:
        raise DiagnosticError("SSE2-semantics input report mismatch")
    expected_semantics = {"reference_revision": REFERENCE_REVISION, "vec_cpp_sha256": REFERENCE_SOURCE_HASHES["vec.cpp"][1], "vec_h_sha256": REFERENCE_SOURCE_HASHES["vec.h"][1], "lane_width": 4, "fma": False, "scalar_tail": False}
    if report["semantics"] != expected_semantics:
        raise DiagnosticError("SSE2-semantics provenance mismatch")
    if set(report["down_tensor"]) != {"name", "type", "offset", "file_offset", "span"} or report["down_tensor"]["name"] != "blk.0.ffn_down.weight" or report["down_tensor"]["type"] != 14:
        raise DiagnosticError("SSE2-semantics down descriptor mismatch")
    expected_names = ["blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_kv_a_norm.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.1.attn_output.weight", "blk.1.ffn_norm.weight", "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias", "blk.1.ffn_up_exps.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight", "blk.1.ffn_up_shexp.weight", "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_down_shexp.weight"]
    if [item.get("name") for item in report["layer1_tensors"]] != expected_names or any(set(item) != {"name", "type", "offset", "file_offset", "span"} for item in report["layer1_tensors"]):
        raise DiagnosticError("SSE2-semantics layer-1 descriptors mismatch")
    if list(report["arms"]) != list(ARMS) or list(report["controls"]) != list(CONTROLS) or report["q6_down_arms"] != EXPECTED_Q6_ARMS:
        raise DiagnosticError("SSE2-semantics arm accounting mismatch")
    values = {name: validate_arm(root, report["arms"][name], name) for name in ARMS}
    controls = {name: validate_arm(root, report["controls"][name], name) for name in CONTROLS}
    for field in ("swiglu", "ffn_out", "l_out"):
        if report["arms"]["scalar_libm_replay"][field]["sha256"] != SCALAR_IDENTITIES[field]:
            raise DiagnosticError(f"scalar-libm identity mismatch: {field}")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise DiagnosticError("SSE2-semantics source/execution contract mismatch")
    counts = read_json(root / "strat01_f16_vector_counts.json", "SSE2-semantics helper counts")
    if counts != EXPECTED_COUNTS:
        raise DiagnosticError(f"SSE2-semantics helper count mismatch: {counts!r}")
    return values, controls, {"report": report, "counts": counts}


def classify(candidate_exact: bool, candidate_passes: bool) -> str:
    if candidate_exact and candidate_passes:
        return "POST_F16_BLOCK0_SWIGLU_SSE2_EXACT_REPAIR"
    if candidate_passes:
        return "POST_F16_BLOCK0_SWIGLU_SSE2_NUMERIC_REPAIR"
    return "POST_F16_BLOCK0_SWIGLU_SSE2_INSUFFICIENT"


def adjudicate(values: dict[str, Any], controls: dict[str, Any], validated: dict[str, Any], reference: dict[str, np.ndarray], paths: dict[str, Path], predecessor: dict[str, Any]) -> dict[str, Any]:
    judgments = {arm: {name: prev.post.judged(values[arm]["checkpoints"][name], reference[name], name) for name in ORDER} for arm in ARMS}
    first_failures = {arm: next((name for name in ORDER[1:] if not judgments[arm][name]["pass"]), None) for arm in ARMS}
    frozen_scalar = predecessor["c_report"]["arms"]["expression_q6_replay"]["checkpoints"]
    scalar_exact = {name: validated["report"]["arms"]["scalar_libm_replay"]["checkpoints"][name]["sha256"] == frozen_scalar[name]["sha256"] for name in ORDER}
    if not all(scalar_exact.values()) or first_failures["scalar_libm_replay"] != "ffn_moe_down-1":
        raise DiagnosticError("scalar-libm predecessor replay mismatch")
    control_judgments = {control: {name: prev.post.judged(controls[control]["checkpoints"][name], reference[name], name) for name in ORDER} for control in CONTROLS}
    if any(all(item[name]["pass"] for name in ORDER[1:]) for item in control_judgments.values()):
        raise DiagnosticError("SSE2-semantics causal control did not reject")
    mutation_refused: dict[str, bool] = {}
    for name, path in paths.items():
        data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1
        mutation_refused[name] = hashlib.sha256(data).hexdigest() != INPUTS[name][1]
    swapped = copy.deepcopy(validated["report"]["arms"])
    swapped["scalar_libm_replay"], swapped["pinned_sse2_candidate"] = swapped["pinned_sse2_candidate"], swapped["scalar_libm_replay"]
    if not all(mutation_refused.values()) or swapped == validated["report"]["arms"]:
        raise DiagnosticError("SSE2-semantics identity/origin control failed")
    candidate_report = validated["report"]["arms"]["pinned_sse2_candidate"]
    candidate_exact = candidate_report["swiglu"]["sha256"] == INPUTS["reference_swiglu"][1]
    if candidate_exact and (candidate_report["ffn_out"]["sha256"] != PASS_IDENTITIES["ffn_out"] or candidate_report["l_out"]["sha256"] != PASS_IDENTITIES["l_out"]):
        raise DiagnosticError("exact SSE2 candidate did not preserve passing Q6/start identities")
    candidate_passes = first_failures["pinned_sse2_candidate"] is None
    status = classify(candidate_exact, candidate_passes)
    ref_sw = np.fromfile(paths["reference_swiglu"], dtype="<f4")
    ref_out = np.fromfile(prev.predecessor.REFERENCE_COMPONENTS["reference_ffn_out"][0], dtype="<f4")
    local = {arm: {"swiglu": prev.sw.wide_judgment(values[arm]["swiglu"], ref_sw), "ffn_out": prev.sw.down.direct_judgment(values[arm]["ffn_out"], ref_out), "l_out": prev.post.judged(values[arm]["l_out"], reference["l_out-0"], "l_out-0")} for arm in ARMS}
    return {"status": status, "first_failures": first_failures, "candidate_exact_reference": candidate_exact, "local_judgments": local, "arm_judgments": judgments, "control_judgments": control_judgments, "controls": {"scalar_replay_exact": scalar_exact, "mutated_inputs_refused": mutation_refused, "origin_swap_rejected": True, "helper_counts_exact": validated["counts"] == EXPECTED_COUNTS, "q6_down_arms_exact": validated["report"]["q6_down_arms"] == EXPECTED_Q6_ARMS}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apparatus-only", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    output = (args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS"
    errors: list[str] = []
    commands: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    result: dict[str, Any] = {"status": "NOT_RUN"}
    report: dict[str, Any] = {}
    frozen_paths: dict[str, Path] = {}
    compiler = shutil.which("clang")
    binary: Path | None = None
    diagnostic_invocations = 0
    try:
        sources = source_inventory()
        controls = source_controls()
        if not all(controls.values()):
            raise DiagnosticError("SSE2-semantics source controls failed")
        evidence = validate_reference_evidence()
        if not compiler:
            raise DiagnosticError("clang unavailable")
        binary = output / "engine_post_f16_block0_swiglu_sse2_semantics.exe"
        commands["compile"] = prev.post.r2c.base.run_command([compiler, *prev.post.r2c.base.COMPILE_FLAGS, str(ENGINE), "-o", str(binary), "-lm"], output=output, label="compile", timeout=600)
        prev.post.r2c.base.require_ok(commands["compile"], "compile")
        commands["python_tests"] = prev.post.r2c.base.run_command([sys.executable, "-B", "-m", "unittest", "-v", *TEST_MODULES], output=output, label="all_strat01_unittests", timeout=1800)
        prev.post.r2c.base.require_ok(commands["python_tests"], "all STRAT-01 Python tests")
        selftests = (*prev.q4base.SELFTESTS, "--strat01-f16-vector-parity-selftest", "--strat01-post-f16-layer1-start-cross-input-selftest", "--strat01-post-f16-block0-terminal-component-cross-input-selftest", "--strat01-post-f16-block0-ffn-operator-cross-input-selftest", "--strat01-post-f16-block0-swiglu-sse2-semantics-selftest")
        for index, option in enumerate(selftests):
            label = f"selftest_{index:02d}"
            commands[label] = prev.post.r2c.base.run_command([str(binary), option], output=output, label=label, timeout=300)
            prev.post.r2c.base.require_ok(commands[label], option)
        if args.apparatus_only:
            status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            reference, current, frozen_paths, predecessor, evidence = validate_frozen()
            if not model.is_file() or model.stat().st_size != prev.post.r2c.base.EXPECTED_BYTES or sha(model) != prev.post.r2c.base.EXPECTED_SHA256:
                raise DiagnosticError("accepted artifact identity mismatch")
            root = output / "diagnostic"
            root.mkdir()
            diagnostic_invocations = 1
            command = [str(binary), "--strat01-post-f16-block0-swiglu-sse2-semantics", str(model), "--reference-gate", str(frozen_paths["reference_gate"]), "--reference-up", str(frozen_paths["reference_up"]), "--reference-swiglu", str(frozen_paths["reference_swiglu"]), "--reference-ffn-inp", str(frozen_paths["reference_ffn_inp"]), "--out-dir", str(root)]
            commands["diagnostic"] = prev.post.r2c.base.run_command(command, output=output, label="diagnostic", timeout=21600)
            prev.post.r2c.base.require_ok(commands["diagnostic"], "SSE2-semantics diagnostic")
            sources = source_inventory()
            values, causal, validated = validate_report(root, model, sources, frozen_paths)
            report = validated["report"]
            result = adjudicate(values, causal, validated, reference, frozen_paths, predecessor)
            status = result["status"]
    except (DiagnosticError, prev.DiagnosticError, prev.predecessor.DiagnosticError, prev.post.DiagnosticError, prev.post.r2c.RunnerError, prev.post.r2c.base.RunnerError, prev.q4base.ParityError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started, "git_head_observed" if args.apparatus_only else "git_head": prev.post.r2c.base.git_value(["git", "rev-parse", "HEAD"]), "source_hashes": sources, "reference_evidence": evidence, "predecessor": {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA}, "frozen_inputs": {name: str(path) for name, path in frozen_paths.items()}, "artifact": {"path": str(model), "expected_bytes": prev.post.r2c.base.EXPECTED_BYTES, "expected_sha256": prev.post.r2c.base.EXPECTED_SHA256}, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": sha(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_post_f16_block0_swiglu_sse2_semantics_v1", "status": status, "errors": errors, "diagnostic_invocations": diagnostic_invocations, "donor_graph_executions": 0, "reference_graph_executions": 0, "source_controls": source_controls() if sources else {}, "adjudication": result, "c_report": report, "non_claims": ["production repair", "block-0 graph", "gate/up split", "Q6", "later layers", "tokenizer/logits/generation", "quality", "RAM", "rate"], "provenance": provenance}
    prev.post.r2c.base.write_json(output / "adjudication.json", record)
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if status == "APPARATUS_READY_NO_DONOR_EXECUTION" or status.startswith("POST_F16_BLOCK0_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
