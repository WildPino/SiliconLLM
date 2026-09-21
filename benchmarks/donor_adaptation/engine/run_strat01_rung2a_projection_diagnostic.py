#!/usr/bin/env python3
"""Adjudicate the frozen Rung-2A projection diagnostic without donor execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine.build_strat01_rung2a_projection_diagnostic import (
    PINNED_HEAD,
    PINNED_LLAMA,
    build,
    verify_pinned_llama,
)

SOURCE_RUN = ROOT / "benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_repair2_20260921"
SOURCE_MANIFEST = SOURCE_RUN / "run_manifest.json"
SOURCE_MANIFEST_SHA = "0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f"
MODEL = ROOT / "benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
MODEL_BYTES = 6_474_702_976
MODEL_SHA = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
TIGHT_NRMSE = 2e-6
TIGHT_MAX = 1e-5
OLD_NRMSE = 2e-3
OLD_MAX = 1e-2

PAYLOADS = {
    "ref_input": (SOURCE_RUN / "pinned_reference/prefill8/attn_norm-0.full.f32le", 8 * 1536, "c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd"),
    "ref_q": (SOURCE_RUN / "pinned_reference/prefill8/q-0.full.f32le", 8 * 6144, "4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b"),
    "ref_kv": (SOURCE_RUN / "pinned_reference/prefill8/kv_cmpr_pe-0.full.f32le", 8 * 576, "6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c"),
    "c_input": (SOURCE_RUN / "c_engine/prefill8_attn_norm-0.f32", 8 * 1536, "f746d41ff1d909d70d09b241c2a3f2d66fd837b51dbeec5ca95ba618d3456d9e"),
    "c_q": (SOURCE_RUN / "c_engine/prefill8_q-0.f32", 8 * 6144, "78dafb9fff7edd265a81693707cfd3d300b41cb8eeb9aab3d9a6c15c81ae383d"),
    "c_kv": (SOURCE_RUN / "c_engine/prefill8_kv_cmpr_pe-0.f32", 8 * 576, "5325d4dca9deb7fa528895530e73871676653764156a0c363f46c7ff350e4caf"),
}


class DiagnosticError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_payload(path: Path, count: int, expected_sha: str) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * 4:
        raise DiagnosticError(f"wrong payload byte count: {path}")
    if sha256_file(path) != expected_sha:
        raise DiagnosticError(f"payload hash mismatch: {path}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != count or not np.isfinite(values).all():
        raise DiagnosticError(f"invalid float payload: {path}")
    return values


def validate_layout(name: str, shape: tuple[int, ...], order: str) -> None:
    expected = {
        "q": ((8, 6144), "token,head,feature"),
        "kv": ((8, 576), "token,feature"),
    }
    if name not in expected or (shape, order) != expected[name]:
        raise DiagnosticError("projection layout substitution")


def metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float | bool]:
    if candidate.shape != target.shape or not np.isfinite(candidate).all() or not np.isfinite(target).all():
        raise DiagnosticError("metric inputs are invalid")
    delta = candidate.astype(np.float64) - target.astype(np.float64)
    rms_delta = math.sqrt(float(np.mean(delta * delta)))
    rms_target = math.sqrt(float(np.mean(target.astype(np.float64) ** 2)))
    nrmse = rms_delta / max(rms_target, 1e-12)
    normalized_max = float(np.max(np.abs(delta))) / max(float(np.max(np.abs(target))), 1e-6)
    return {
        "nrmse": nrmse,
        "normalized_max": normalized_max,
        "tight_pass": nrmse <= TIGHT_NRMSE and normalized_max <= TIGHT_MAX,
        "old_gate_pass": nrmse <= OLD_NRMSE and normalized_max <= OLD_MAX,
    }


def run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    started = datetime.now().astimezone()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
    return {
        "command": command,
        "cwd": str(cwd),
        "started": started.isoformat(),
        "returncode": completed.returncode,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }


def helper_command(binary: Path, tensor: str, mode: str, source: Path, output: Path, mutate: bool = False) -> list[str]:
    command = [str(binary), "--model", str(MODEL), "--input", str(source), "--output", str(output), "--tensor", tensor, "--mode", mode]
    if mutate:
        command.append("--mutate-q8")
    return command


def classify(results: dict[str, dict[str, dict[str, float | bool]]], controls: dict[str, bool]) -> str:
    if not all(controls.values()):
        return "VOID_PROJECTION_DIAGNOSTIC"
    tensors = ("q", "kv")
    c_ok = all(bool(results[name]["d32_c_control"]["tight_pass"]) for name in tensors)
    if not c_ok:
        return "VOID_PROJECTION_DIAGNOSTIC"
    q8_ok = {name: bool(results[name]["q8k_reference"]["tight_pass"]) for name in tensors}
    d32_fails = {name: not bool(results[name]["d32_reference"]["old_gate_pass"]) for name in tensors}
    if all(q8_ok.values()) and all(d32_fails.values()):
        return "ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION"
    if any(q8_ok.values()):
        return "PARTIAL_Q8K_ATTRIBUTION"
    return "REJECT_Q8K_AS_SUFFICIENT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise DiagnosticError("output directory already exists; raw evidence is immutable")
    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "strat01_rung2a_projection_diagnostic_v1",
        "status": "VOID_PROJECTION_DIAGNOSTIC",
        "started": datetime.now().astimezone().isoformat(),
        "donor_executions": 0,
        "errors": [],
        "commands": [],
    }
    try:
        critical_paths = [
            Path(__file__).resolve(),
            ROOT / "benchmarks/donor_adaptation/engine/build_strat01_rung2a_projection_diagnostic.py",
            ROOT / "benchmarks/donor_adaptation/engine/strat01_rung2a_projection_diagnostic.cpp",
            ROOT / "benchmarks/donor_adaptation/engine/test_strat01_rung2a_projection_diagnostic.py",
            ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROJECTION_DIAGNOSTIC_PROTOCOL_20260921.md",
        ]
        clean_check = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *map(str, critical_paths)],
            cwd=ROOT,
            check=False,
        )
        if clean_check.returncode:
            raise DiagnosticError("diagnostic implementation or protocol differs from HEAD")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        unit_record = run_command(
            [sys.executable, "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_rung2a_projection_diagnostic"],
            ROOT,
            output_dir / "model_free_unittests.stdout.log",
            output_dir / "model_free_unittests.stderr.log",
        )
        manifest["commands"].append(unit_record)
        if unit_record["returncode"]:
            raise DiagnosticError("model-free negative controls failed")
        if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA:
            raise DiagnosticError("source run manifest hash mismatch")
        source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        artifact = source_manifest["provenance"]["artifact"]
        if artifact["expected_bytes"] != MODEL_BYTES or artifact["expected_sha256"] != MODEL_SHA:
            raise DiagnosticError("source run artifact identity mismatch")
        if not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES or sha256_file(MODEL) != MODEL_SHA:
            raise DiagnosticError("current GGUF identity mismatch")
        verify_pinned_llama()
        arrays = {name: validate_payload(*spec) for name, spec in PAYLOADS.items()}
        validate_layout("q", (8, 6144), "token,head,feature")
        validate_layout("kv", (8, 576), "token,feature")

        build_dir = output_dir / "build"
        binary = build(build_dir)
        controls_command = [str(binary), "--selftest"]
        control_record = run_command(controls_command, ROOT, output_dir / "helper_selftest.stdout.log", output_dir / "helper_selftest.stderr.log")
        manifest["commands"].append(control_record)
        if control_record["returncode"]:
            raise DiagnosticError("helper self-test failed")

        produced: dict[str, Path] = {}
        for tensor in ("q", "kv"):
            jobs = (
                ("d32_c", "d32", PAYLOADS["c_input"][0], False),
                ("d32_ref", "d32", PAYLOADS["ref_input"][0], False),
                ("q8k_ref", "q8k", PAYLOADS["ref_input"][0], False),
                ("q8k_mutated", "q8k", PAYLOADS["ref_input"][0], True),
                ("f64_ref", "f64", PAYLOADS["ref_input"][0], False),
            )
            for label, mode, source, mutate in jobs:
                path = output_dir / f"{tensor}_{label}.f32le"
                record = run_command(
                    helper_command(binary, tensor, mode, source, path, mutate),
                    ROOT,
                    output_dir / f"{tensor}_{label}.stdout.log",
                    output_dir / f"{tensor}_{label}.stderr.log",
                )
                manifest["commands"].append(record)
                if record["returncode"]:
                    raise DiagnosticError(f"helper failed for {tensor}/{label}")
                produced[f"{tensor}_{label}"] = path

        results: dict[str, dict[str, dict[str, float | bool]]] = {}
        controls = {"identity": True, "model_free_unittests": True, "helper_selftest": True}
        for tensor, count in (("q", 8 * 6144), ("kv", 8 * 576)):
            target_c = arrays[f"c_{tensor}"]
            target_ref = arrays[f"ref_{tensor}"]
            loaded = {
                label: validate_payload(path, count, sha256_file(path))
                for label, path in produced.items()
                if label.startswith(tensor + "_")
            }
            results[tensor] = {
                "d32_c_control": metrics(loaded[f"{tensor}_d32_c"], target_c),
                "d32_reference": metrics(loaded[f"{tensor}_d32_ref"], target_ref),
                "q8k_reference": metrics(loaded[f"{tensor}_q8k_ref"], target_ref),
                "f64_reference": metrics(loaded[f"{tensor}_f64_ref"], target_ref),
                "mutated_q8k_reference": metrics(loaded[f"{tensor}_q8k_mutated"], target_ref),
            }
            controls[f"mutated_q8k_{tensor}_fires"] = not bool(results[tensor]["mutated_q8k_reference"]["tight_pass"])

        manifest.update({
            "status": classify(results, controls),
            "finished": datetime.now().astimezone().isoformat(),
            "identity": {
                "source_run_manifest": str(SOURCE_MANIFEST),
                "source_run_manifest_sha256": SOURCE_MANIFEST_SHA,
                "model": str(MODEL),
                "model_bytes": MODEL_BYTES,
                "model_sha256": MODEL_SHA,
                "llama_cpp_source": str(PINNED_LLAMA),
                "llama_cpp_commit": PINNED_HEAD,
                "git_head": head,
                "diagnostic_binary_sha256": sha256_file(binary),
                "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in critical_paths},
            },
            "controls": controls,
            "results": results,
            "outputs": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in produced.items()},
            "non_claims": ["Rung-2A acceptance", "downstream attention parity", "quality", "RAM", "speed"],
        })
    except Exception as error:
        manifest["errors"].append(str(error))
        manifest["finished"] = datetime.now().astimezone().isoformat()
    (output_dir / "adjudication.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    print(manifest["status"])
    if manifest["errors"]:
        print(manifest["errors"][0], file=sys.stderr)
    return 0 if manifest["status"] != "VOID_PROJECTION_DIAGNOSTIC" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        raise SystemExit(f"run_strat01_rung2a_projection_diagnostic: {error}")
