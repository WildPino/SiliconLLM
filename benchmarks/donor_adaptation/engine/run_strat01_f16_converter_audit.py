#!/usr/bin/env python3
"""Audit project binary16 conversion against pinned GGML without donor execution."""
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

from benchmarks.donor_adaptation.engine.run_strat01_f16_vec_dot_diagnostic import (
    BUILDER,
    COUNT_PADDED,
    PINNED_GRAPH_SHA,
    PINNED_HEAD,
    PINNED_LLAMA,
    SOURCE_TRACE,
    callback_to_token_head,
    load_f32,
    manifest_payload,
)
from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_repair import require_ok, run_command, sha256_file

HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_F16_CONVERTER_AUDIT_PROTOCOL_20260921.md"
HELPER_SOURCE = HERE / "strat01_f16_vec_dot_diagnostic.cpp"
TEST_SOURCE = HERE / "test_strat01_f16_converter_audit.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_f16_converter_audit_repair1_20260921"
SEGMENTS = (("kcur", 8 * 576), ("qcur", 8 * 32 * 576), ("softmax", COUNT_PADDED), ("boundary", 20))
BOUNDARY_BITS = np.array([
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x33000000,
    0x33800000, 0x38000000, 0x387FC000, 0x38800000, 0x3F800000,
    0xBF800000, 0x477FE000, 0x477FF000, 0x477FFFFF, 0x47800000,
    0x7F800000, 0xFF800000, 0x7FC00000, 0x7FA00001, 0xFFC12345,
], dtype=np.dtype("<u4"))
CRITICAL_PATHS = (Path(__file__).resolve(), PROTOCOL, BUILDER, HELPER_SOURCE, TEST_SOURCE)


class AuditError(RuntimeError): pass


def utc_now() -> str: return datetime.now(timezone.utc).isoformat()


def segment_for(index: int) -> tuple[str, int]:
    start = 0
    for name, count in SEGMENTS:
        if index < start + count: return name, index - start
        start += count
    raise AuditError("conversion index outside frozen population")


def exponent_class(bits: int) -> str:
    exponent, mantissa = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if exponent == 0: return "zero" if mantissa == 0 else "subnormal_f32"
    if exponent == 0xFF: return "infinity" if mantissa == 0 else "nan"
    unbiased = exponent - 127
    if unbiased < -14: return "below_f16_normal"
    if unbiased > 15: return "above_f16_normal"
    return "f16_normal_range"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists(): raise AuditError("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True); started = time.perf_counter()
    record: dict[str, Any] = {"schema": "strat01_f16_converter_audit_v1", "status": "VOID_F16_CONVERTER_AUDIT", "started_utc": utc_now(), "donor_executions": 0, "commands": [], "errors": []}
    try:
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, CRITICAL_PATHS)], cwd=ROOT, check=False)
        if dirty.returncode: raise AuditError("audit implementation or protocol differs from HEAD")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        if sha256_file(PINNED_LLAMA / "src/llama-graph.cpp") != PINNED_GRAPH_SHA: raise AuditError("pinned graph source mismatch")
        llama_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PINNED_LLAMA, text=True, capture_output=True, check=True).stdout.strip()
        llama_dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PINNED_LLAMA, text=True, capture_output=True, check=True).stdout.strip()
        if llama_head != PINNED_HEAD or llama_dirty: raise AuditError("pinned llama.cpp checkout mismatch")
        qcur_path, kcur_path, softmax_path = manifest_payload("Qcur-0"), manifest_payload("Kcur-0"), manifest_payload("kq_soft_max-0")
        qcur = load_f32(qcur_path, 8 * 32 * 576, "4aca21f044acf71404ef0a7a000ed7b1bfe894efa82c5c1e49f7a5764cc6314b")
        kcur = load_f32(kcur_path, 8 * 576, "2860d9791b620d19788b8112e5366424be21669153eb1ec3255fd5a6c1b167f3")
        softmax = callback_to_token_head(load_f32(softmax_path, COUNT_PADDED, "8def8f8d6969dab39987e915084a74cb0c766c9d842db8b272886c248d60092e"), 256)
        mapped = output / "mapped"; mapped.mkdir(); mapped_softmax = mapped / "softmax.padded.token_head_slot.f32le"; softmax.astype("<f4", copy=False).tofile(mapped_softmax)

        tests = run_command([sys.executable, "-m", "unittest", "-v", "benchmarks.donor_adaptation.engine.test_strat01_f16_converter_audit"], output, "model_free_tests")
        record["commands"].append(tests); require_ok(tests, "model-free tests")
        build = output / "build"; command = run_command([sys.executable, str(BUILDER), "--build-dir", str(build)], output, "build_helper", timeout=3600)
        record["commands"].append(command); require_ok(command, "helper build")
        helpers = list((build / "cmake-build").rglob("strat01_f16_vec_dot_diagnostic.exe"))
        if len(helpers) != 1: raise AuditError("cannot resolve converter helper")
        helper = helpers[0]; command = run_command([str(helper), "--selftest"], output, "helper_selftest")
        record["commands"].append(command); require_ok(command, "helper self-test")
        products = output / "products"; products.mkdir()
        command = run_command([str(helper), "--qcur", str(qcur_path), "--kcur", str(kcur_path), "--padded-softmax", str(mapped_softmax), "--output-dir", str(products)], output, "conversion_audit", timeout=3600)
        record["commands"].append(command); require_ok(command, "conversion audit helper")
        project_path, pinned_path = products / "project_f16.bin", products / "pinned_f16.bin"
        source_witness_path = products / "source_f32_bits.bin"
        expected_count = sum(count for _, count in SEGMENTS)
        if project_path.stat().st_size != expected_count * 2 or pinned_path.stat().st_size != expected_count * 2 or source_witness_path.stat().st_size != expected_count * 4: raise AuditError("conversion stream completeness failure")
        project = np.fromfile(project_path, dtype=np.dtype("<u2")); pinned = np.fromfile(pinned_path, dtype=np.dtype("<u2"))
        source = np.concatenate((kcur, qcur, softmax, BOUNDARY_BITS.view(np.dtype("<f4"))))
        if source.size != expected_count: raise AuditError("source population completeness failure")
        source_bits = source.view(np.dtype("<u4")); mismatch_indices = np.flatnonzero(project != pinned)
        source_witness = np.fromfile(source_witness_path, dtype=np.dtype("<u4"))
        if not np.array_equal(source_witness, source_bits): raise AuditError("helper conversion source-order witness mismatch")
        finite = np.isfinite(source); finite_mismatches = mismatch_indices[finite[mismatch_indices]]
        nan_indices = np.flatnonzero(np.isnan(source)); canonical_nan = bool(np.all((project[nan_indices] & 0x7E00) == 0x7E00) and np.array_equal(project[nan_indices], pinned[nan_indices]))
        by_segment = {name: 0 for name, _ in SEGMENTS}; by_class: dict[str, int] = {}
        examples = []
        for raw_index in mismatch_indices:
            index = int(raw_index); segment, offset = segment_for(index); by_segment[segment] += 1
            kind = exponent_class(int(source_bits[index])); by_class[kind] = by_class.get(kind, 0) + 1
            if len(examples) < 32:
                examples.append({"index": index, "segment": segment, "segment_offset": offset, "f32_bits": f"0x{int(source_bits[index]):08x}", "value": None if not np.isfinite(source[index]) else float(source[index]), "class": kind, "project_f16": f"0x{int(project[index]):04x}", "pinned_f16": f"0x{int(pinned[index]):04x}"})
        status = "PASS_F16_CONVERTER" if finite_mismatches.size == 0 and canonical_nan else "FAIL_F16_CONVERTER"
        record.update({"status": status, "finished_utc": utc_now(), "seconds": time.perf_counter() - started,
            "identity": {"git_head": head, "llama_cpp_head": llama_head, "helper": {"path": str(helper), "sha256": sha256_file(helper)}, "critical_source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in CRITICAL_PATHS}},
            "population": {"total": int(expected_count), "segments": {name: count for name, count in SEGMENTS}, "source_hashes": {"qcur": sha256_file(qcur_path), "kcur": sha256_file(kcur_path), "softmax": sha256_file(softmax_path)}},
            "results": {"total_mismatches": int(mismatch_indices.size), "finite_mismatches": int(finite_mismatches.size), "nan_count": int(nan_indices.size), "nan_policy_matches": canonical_nan, "mismatches_by_segment": by_segment, "mismatches_by_class": by_class, "first_mismatches": examples},
            "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in (project_path, pinned_path, source_witness_path)},
            "non_claims": ["F16 vec-dot result", "production repair", "Rung 2B", "quality", "RAM", "speed"]})
    except Exception as error:
        record["errors"].append(str(error)); record["finished_utc"] = utc_now(); record["seconds"] = time.perf_counter() - started
    (output / "adjudication.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(record["status"])
    if record["errors"]: print(record["errors"][0], file=sys.stderr)
    return 0 if record["status"] != "VOID_F16_CONVERTER_AUDIT" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except AuditError as error: raise SystemExit(f"run_strat01_f16_converter_audit: {error}")
