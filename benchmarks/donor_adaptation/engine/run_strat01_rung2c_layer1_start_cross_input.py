#!/usr/bin/env python3
"""Execute the frozen Rung-2C layer-1-start cross-input diagnostic."""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base


ROOT, HERE = base.ROOT, base.HERE
ENGINE = base.ENGINE
HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2c_layer1_start_cross_input.h"
RUNG2A, RUNG2C = base.RUNG2A, base.RUNG2C
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_RUNG2C_LAYER1_START_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_rung2c_layer1_start_cross_input.py"
RAW, RECOVERY = base.RAW, base.RECOVERY
DEFAULT_MODEL = base.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_rung2c_layer1_start_cross_input_20260923"
ORDER = ["attn_norm-1", "q-1", "kv_cmpr_pe-1", "k_pe-1", "kv_cmpr-1", "q_pe-1", "q_nope_absorbed_perm-1", "Qcur-1", "Kcur-1", "Vcur-1", "kqv_out-1"]
SIZES = {"attn_norm-1": 49_152, "q-1": 196_608, "kv_cmpr_pe-1": 18_432, "k_pe-1": 2_048, "kv_cmpr-1": 16_384, "q_pe-1": 65_536, "q_nope_absorbed_perm-1": 524_288, "Qcur-1": 589_824, "Kcur-1": 18_432, "Vcur-1": 16_384, "kqv_out-1": 196_608}
REF_HASHES = {"attn_norm-1": "fc9d710b1124f79d7040b7d519930d6f0962f4d071fbb3e41fb7fa827de90606", "q-1": "276626fc4b0f0339da95e66ac50be0ce3d958d1354ba584b72c63c9d8134b9a8", "kv_cmpr_pe-1": "448b912739c596ccd745936b67144aae07ccceb12f85ee04c0367bccd4a24ed9", "k_pe-1": "2ffddb8ade46a25cecab1387d295f7a5ad2f7479bd7e8b83a0aa64decc554b68", "kv_cmpr-1": "3f3c00b3fe008aecda5e1164b5da2b295d349270b1e339831da1544a849a9387", "q_pe-1": "f58bc7886200116d132787a0d8a997d3dd13ef3a2d7ccd93f66caf5ffdc4ce7b", "q_nope_absorbed_perm-1": "18d963a382677e95fa675689121906a71faecdf2283943e3c5ab9be892872375", "Qcur-1": "9cb50fb41e0d33549eb53bd4ae19605705331562d36f975bb2ac82cc4c41cb16", "Kcur-1": "73febde321c2ea45a56a5d27a3605b820e08f8b038420a379afc4b9c3f0c72b4", "Vcur-1": "3f3c00b3fe008aecda5e1164b5da2b295d349270b1e339831da1544a849a9387", "kqv_out-1": "fa1006c4e2c365d3a5540baebbe4c88180f524d5f69c610fcc3b4205a856d489"}
C_HASHES = {"attn_norm-1": "dc668e6d6cc18c2282fc767f3cd57cbd38bb7d08ab496fc85c81f5663cb7e0fe", "q-1": "95c189683f5b72af6d233bd23c8acb4801c0ae61bda5610e6a80d0100c49fe75", "kv_cmpr_pe-1": "eef0d629f07b043af506b1a3220972975d753965c5831d3586a527040821e957", "k_pe-1": "19ca83be23424f28d11f8a329804ac2965619f731842b6f37ff1d9bdf3b468ea", "kv_cmpr-1": "58fa280015b838d47df9b1a0ba9064d6cc8eb25000e454330d988ec48d537b57", "q_pe-1": "89926a0ec73eb8e021d319630a35348d84c33791bd7ee7f4ebe7fb5c0691ae08", "q_nope_absorbed_perm-1": "966dc1942bf4d05280d0d6842cc13eb67d63a070da677014dc588f58ccdd920e", "Qcur-1": "3faa0a37833c9d9b9cf92d3bf985d13bf6f09b9c278f81e8813258e86e7d2b2f", "Kcur-1": "febfcaa2ca7192a71b7ffca1af26252c9e81b010068478928ed6d56fa78fb0d4", "Vcur-1": "58fa280015b838d47df9b1a0ba9064d6cc8eb25000e454330d988ec48d537b57", "kqv_out-1": "7e6d44661f21dee62b449b44e1939b59d8ad0799c4ccf6779d77a01a48e3ee41"}
FROZEN_METRICS = {
    "attn_norm-1": (0.0005685064016067361, 0.0007465410900599202), "q-1": (0.0011529447825192514, 0.006009908380967594),
    "kv_cmpr_pe-1": (0.0012572584257080136, 0.0009620431349980618), "k_pe-1": (0.0009183129485835886, 0.001315694798939685),
    "kv_cmpr-1": (0.001228277839347288, 0.0015361116262187639), "q_pe-1": (0.0016477948721958329, 0.002734314105661907),
    "q_nope_absorbed_perm-1": (0.001055263622264422, 0.003367229523603503), "Qcur-1": (0.0010587354586825643, 0.003367229523603503),
    "Kcur-1": (0.0009183710203053601, 0.001315694798939685), "Vcur-1": (0.001228277839347288, 0.0015361116262187639),
    "kqv_out-1": (0.006136167591602251, 0.0030491102772673647),
}
INPUTS = {
    "reference": (RAW / "pinned_reference" / "prefill8" / "l_out-0.full.f32le", "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"),
    "c": (RAW / "c_engine" / "prefill8_l_out-0.f32", "258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44"),
}
REF_DIR = RAW / "pinned_reference" / "prefill8"
C_DIR = RAW / "c_engine"


def source_inventory() -> dict[str, Any]:
    paths = {"runner": Path(__file__).resolve(), "base_runner": Path(base.__file__).resolve(), "engine": ENGINE, "rung2a": RUNG2A, "rung2c": RUNG2C, "header": HEADER, "protocol": PROTOCOL, "tests": TESTS}
    if any(not path.is_file() for path in paths.values()):
        raise base.RunnerError("missing layer1-start source")
    return {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in paths.items()}


def validate_frozen() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, (path, digest) in INPUTS.items():
        if not path.is_file() or path.stat().st_size != 49_152 or base.sha256_file(path) != digest:
            raise base.RunnerError(f"{name} input identity mismatch")
        result[name] = path.resolve(strict=True)
    for name, (path, digest) in base.EVIDENCE_FILES.items():
        if not path.is_file() or base.sha256_file(path) != digest:
            raise base.RunnerError(f"evidence {name} mismatch")
    for checkpoint in ORDER:
        ref = REF_DIR / f"{checkpoint}.full.f32le"; c = C_DIR / f"prefill8_{checkpoint}.f32"
        if ref.stat().st_size != SIZES[checkpoint] or base.sha256_file(ref) != REF_HASHES[checkpoint] or c.stat().st_size != SIZES[checkpoint] or base.sha256_file(c) != C_HASHES[checkpoint]:
            raise base.RunnerError(f"frozen target mismatch: {checkpoint}")
        result[f"ref/{checkpoint}"] = ref.resolve(); result[f"c/{checkpoint}"] = c.resolve()
    return result


def validate_output_map(root: Path, items: Any, arm: str) -> dict[str, np.ndarray]:
    if not isinstance(items, dict) or list(items) != ORDER:
        raise base.RunnerError(f"{arm} output ordering/schema mismatch")
    values = {}
    for checkpoint in ORDER:
        item = items[checkpoint]
        if set(item) != {"path", "bytes", "sha256"} or item["bytes"] != SIZES[checkpoint]:
            raise base.RunnerError(f"{arm}/{checkpoint} output schema mismatch")
        path = base.contained(root, item["path"], SIZES[checkpoint], item["sha256"], f"{arm}/{checkpoint}")
        values[checkpoint] = base.load_f32(path, SIZES[checkpoint] // 4, f"{arm}/{checkpoint}")
    return values


def judged_checkpoint(candidate: np.ndarray, reference: np.ndarray, checkpoint: str) -> dict[str, Any]:
    result: dict[str, Any] = base.metrics(candidate, reference)
    result.update({"nrmse_limit": base.LIMITS[0], "normalized_max_limit": base.LIMITS[1]})
    result["pass"] = result["nrmse"] <= base.LIMITS[0] and result["normalized_max"] <= base.LIMITS[1]
    width = SIZES[checkpoint] // (8 * 4)
    result["per_token"] = [dict(base.metrics(candidate.reshape(8, width)[i], reference.reshape(8, width)[i]), token=i) for i in range(8)]
    return result


def validate_report(root: Path, model: Path, sources: dict[str, Any], frozen: dict[str, Path]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads((root / "strat01_rung2c_layer1_start_cross_input.json").read_text(encoding="utf-8"))
    required = {"command", "state", "self_certifies_pass", "model", "inputs", "tensors", "outputs", "controls", "engine_source_sha256", "diagnostic_source_sha256", "compiler_family", "donor_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-rung2c-layer1-start-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise base.RunnerError("layer1-start report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": base.EXPECTED_MODEL_BYTES, "sha256": base.EXPECTED_MODEL_SHA}:
        raise base.RunnerError("model report mismatch")
    expected_inputs = {name: {"path": str(path), "bytes": 49_152, "sha256": INPUTS[name][1]} for name, path in ((n, frozen[n]) for n in ("reference", "c"))}
    if report["inputs"] != expected_inputs or len(report["tensors"]) != 7:
        raise base.RunnerError("input/tensor report mismatch")
    expected_names = [spec["name"] for spec in ({"name": n} for n in ("blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_kv_a_norm.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.1.attn_output.weight"))]
    if [item.get("name") for item in report["tensors"]] != expected_names or any(set(item) != {"name", "type", "offset", "file_offset", "span"} for item in report["tensors"]):
        raise base.RunnerError("tensor descriptor report mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise base.RunnerError("source/execution contract mismatch")
    if set(report["outputs"]) != {"reference_input", "c_input"}:
        raise base.RunnerError("arm set mismatch")
    values = {arm: validate_output_map(root, report["outputs"][arm], arm) for arm in ("reference_input", "c_input")}
    controls = {}
    if set(report["controls"]) != {"reference_token7_negated", "reference_rows0_7_swapped"}:
        raise base.RunnerError("control set mismatch")
    for name, item in report["controls"].items():
        path = base.contained(root, item["path"], 196_608, item["sha256"], name); controls[name] = base.load_f32(path, 8 * 6144, name)
    return values, report, controls


def adjudicate(values: dict[str, dict[str, np.ndarray]], report: dict[str, Any], controls: dict[str, np.ndarray], frozen: dict[str, Path]) -> dict[str, Any]:
    judgments: dict[str, Any] = {}; replay: dict[str, Any] = {}
    for checkpoint in ORDER:
        ref = base.load_f32(frozen[f"ref/{checkpoint}"], SIZES[checkpoint] // 4, f"ref/{checkpoint}")
        c = base.load_f32(frozen[f"c/{checkpoint}"], SIZES[checkpoint] // 4, f"c/{checkpoint}")
        if values["c_input"][checkpoint].tobytes() != c.tobytes() or report["outputs"]["c_input"][checkpoint]["sha256"] != C_HASHES[checkpoint]:
            raise base.RunnerError(f"C replay mismatch: {checkpoint}")
        judgments[checkpoint] = judged_checkpoint(values["reference_input"][checkpoint], ref, checkpoint)
        replay[checkpoint] = base.metrics(values["c_input"][checkpoint], ref)
        expected = FROZEN_METRICS[checkpoint]
        if abs(replay[checkpoint]["nrmse"] - expected[0]) > 1e-12 or abs(replay[checkpoint]["normalized_max"] - expected[1]) > 1e-12:
            raise base.RunnerError(f"frozen metric replay mismatch: {checkpoint}")
    ref_kqv = base.load_f32(frozen["ref/kqv_out-1"], 8 * 6144, "reference kqv")
    control_judgments = {name: judged_checkpoint(value, ref_kqv, "kqv_out-1") for name, value in controls.items()}
    if any(value["pass"] for value in control_judgments.values()):
        raise base.RunnerError("causal control did not reject")
    mutation_refused = {}
    for name in ("reference", "c"):
        data = bytearray(frozen[name].read_bytes()); data[len(data) // 2] ^= 1
        mutation_refused[name] = not base.identity_matches(bytes(data), 49_152, INPUTS[name][1])
    swapped = copy.deepcopy(report["outputs"]); swapped["reference_input"], swapped["c_input"] = swapped["c_input"], swapped["reference_input"]
    label_swap_rejected = swapped["c_input"]["attn_norm-1"]["sha256"] != C_HASHES["attn_norm-1"]
    if not all(mutation_refused.values()) or not label_swap_rejected:
        raise base.RunnerError("identity/label control failure")
    first_failure = next((name for name in ORDER if not judgments[name]["pass"]), None)
    if first_failure is None:
        status = "BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT"
    elif first_failure == "kqv_out-1":
        status = "LAYER1_GENERATED_QKV_RESIDUAL_SUFFICIENT"
    else:
        status = "LAYER1_PREATTENTION_RESIDUAL_SUFFICIENT_AT_" + first_failure.upper().replace("-", "_")
    return {"status": status, "first_failure": first_failure, "reference_input_vs_reference": judgments, "c_input_replay_vs_reference": replay, "control_judgments": control_judgments, "controls": {"c_replay_byte_exact_all_11": True, "frozen_metrics_reproduced_within_1e-12": True, "mutated_inputs_refused": mutation_refused, "arm_label_swap_rejected": label_swap_rejected}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--model", type=Path, default=DEFAULT_MODEL); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--apparatus-only", action="store_true"); args = parser.parse_args()
    model, output = args.model.resolve(), args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True); started_utc, started = base.now_utc(), time.perf_counter(); status = "VOID_RUNG2C_LAYER1_START_CROSS_INPUT"; errors: list[str] = []; commands: dict[str, Any] = {}; sources: dict[str, Any] = {}; report: dict[str, Any] = {}; adjudication: dict[str, Any] = {"status": "NOT_RUN"}; compiler = shutil.which("clang"); binary: Path | None = None
    try:
        sources = source_inventory()
        if not compiler: raise base.RunnerError("clang unavailable")
        commands["compile"] = base.run_command([compiler, *base.COMPILE_FLAGS, str(ENGINE), "-o", str(output / "engine_l1start.exe"), "-lm"], output, "compile", 600); base.require_ok(commands["compile"], "compile"); binary = output / "engine_l1start.exe"
        commands["selftest"] = base.run_command([str(binary), "--strat01-rung2c-layer1-start-cross-input-selftest"], output, "selftest", 300); base.require_ok(commands["selftest"], "selftest")
        commands["legacy"] = base.run_command([str(binary), "--kselftest"], output, "legacy", 300); base.require_ok(commands["legacy"], "legacy")
        commands["python_tests"] = base.run_command([sys.executable, "-B", "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_rung2c_layer1_start_cross_input"], output, "python_tests", 300); base.require_ok(commands["python_tests"], "Python tests")
        if args.apparatus_only: status = "APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            if not model.is_file() or model.stat().st_size != base.EXPECTED_MODEL_BYTES or base.sha256_file(model) != base.EXPECTED_MODEL_SHA: raise base.RunnerError("model identity mismatch")
            frozen = validate_frozen(); root = output / "diagnostic"; root.mkdir(); command = [str(binary), "--strat01-rung2c-layer1-start-cross-input", str(model), "--reference-input", str(frozen["reference"]), "--c-input", str(frozen["c"]), "--out-dir", str(root)]
            commands["diagnostic"] = base.run_command(command, output, "diagnostic", 21600); base.require_ok(commands["diagnostic"], "diagnostic"); sources = source_inventory(); values, report, controls = validate_report(root, model, sources, frozen); adjudication = adjudicate(values, report, controls, frozen); status = adjudication["status"]
    except base.RunnerError as exc: errors.append(str(exc))
    except Exception as exc: errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {"started_utc": started_utc, "finished_utc": base.now_utc(), "seconds": time.perf_counter()-started, "git_head": base.git_value(["git", "rev-parse", "HEAD"]), "git_status_porcelain": base.git_value(["git", "status", "--porcelain"]), "source_hashes": sources, "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd(), "clang_path": compiler}, "binary": {"path": str(binary) if binary else None, "sha256": base.sha256_file(binary) if binary and binary.is_file() else None}, "commands": commands}
    record = {"schema": "strat01_rung2c_layer1_start_cross_input_adjudication_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "adjudication": adjudication, "c_report": report, "non_claims": ["Rung 2C repair or promotion", "MoE/later layers", "quality, generation, RAM, or rate"], "provenance": provenance}; manifest = {"schema": "strat01_rung2c_layer1_start_cross_input_run_manifest_v1", "status": status, "errors": errors, "donor_graph_executions": 0, "provenance": provenance}
    base.write_json(output / "adjudication.json", record); base.write_json(output / "run_manifest.json", manifest); print(json.dumps({"status": status, "output": str(output), "errors": errors}, indent=2))
    return 0 if not status.startswith("VOID_") else 2


if __name__ == "__main__": raise SystemExit(main())
