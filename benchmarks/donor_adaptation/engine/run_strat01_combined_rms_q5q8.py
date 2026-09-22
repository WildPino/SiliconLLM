#!/usr/bin/env python3
"""Run the frozen combined double-RMSNorm plus K-B Q5_0/Q8_0 propagation cell."""
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

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2a as r2a
from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2b as r2b
from benchmarks.donor_adaptation.engine import run_strat01_upstream_rmsnorm_diag as upstream

HERE = Path(__file__).resolve().parent
ENGINE = r2a.ENGINE
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_combined_rms_q5q8.h"
KB_HEADER = ROOT / "benchmarks/phase60/strat01_gguf_kb_q5q8_diag.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_COMBINED_RMS_Q5Q8_PROPAGATION_PROTOCOL_20260921.md"
MODEL = r2a.DEFAULT_MODEL
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_combined_rms_q5q8_20260921"
DEFAULT_OFFLINE_OUTPUT = HERE / "results/strat01_gigachat_engine_combined_rms_q5q8_offline_adjudication_20260922"
SOURCE_VOID_ADJUDICATION_SHA = "8eca798bb38706b2d9f0a56ebebf5934f035200c5ae0e983ae4e801f56219e66"
SOURCE_VOID_MANIFEST_SHA = "4964efe45dbbde5b7fd80d957b25279c8099f267c6b40e2aac573858d232db5e"
KB_RUN = HERE / "results/strat01_gigachat_engine_kb_q5q8_diag_20260921"
KB_ADJUDICATION = KB_RUN / "adjudication.json"
KB_ADJUDICATION_SHA = "fd2ba64075f02fa26ff9210ff06a80cd6c2e3729966311d991235edc699416eb"
UPSTREAM_ADJUDICATION = upstream.DEFAULT_OUTPUT / "adjudication.json"
UPSTREAM_ADJUDICATION_SHA = "aebf238eb627f7eaa39f29f737d5295c22fa12a4a1fcb8bb1e88d1286d98bf80"
KB_Q8_SHA = "5dae7c9874e8ca79fdccc9ceda45e652f729a2176de0bbebb24453ae818213f2"
KB_OUT_SHA = "cf69d34fa506ee356ab5fb49982e7f872505e47312c7863540d1e0464036d51a"
ARMS = r2a.ARMS
FFN_NAMES = upstream.FFN_NAMES
FFN_SHAPES = upstream.FFN_SHAPES


class DiagnosticError(RuntimeError):
    pass


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{label} missing or malformed: {exc}") from exc


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {"runner": Path(__file__).resolve(), "engine": ENGINE, "rung2a_header": r2a.RUNG2A_HEADER, "rung2b_header": r2b.RUNG2B_HEADER, "upstream_header": upstream.HEADER, "kb_header": KB_HEADER, "diagnostic_header": HEADER, "tests": HERE / "test_strat01_combined_rms_q5q8.py", "protocol": PROTOCOL}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DiagnosticError("missing source(s): " + ", ".join(missing))
    return {name: {"path": str(path), "sha256": r2a.sha256_file(path)} for name, path in paths.items()}


def validate_prior_bindings() -> dict[str, Any]:
    for path, digest in ((KB_ADJUDICATION, KB_ADJUDICATION_SHA), (UPSTREAM_ADJUDICATION, UPSTREAM_ADJUDICATION_SHA)):
        if not path.is_file() or r2a.sha256_file(path) != digest:
            raise DiagnosticError(f"prior adjudication binding mismatch: {path}")
    upstream.validate_prior_bindings()
    kb = read_json(KB_ADJUDICATION, "K-B adjudication")
    if kb.get("status") != "Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY" or kb.get("errors") != [] or kb.get("donor_graph_executions") != 0:
        raise DiagnosticError("K-B prior state mismatch")
    controls = kb.get("adjudication", {}).get("controls", {})
    if not controls or not all(controls.values()):
        raise DiagnosticError("K-B prior controls are incomplete")
    return kb


def archived_upstream_sources() -> dict[str, dict[str, str]]:
    """Return the source identities recorded by the hash-bound upstream run."""
    record = read_json(UPSTREAM_ADJUDICATION, "upstream RMSNorm adjudication")
    report = record.get("candidate_record", {}).get("report", {})
    engine_sha = report.get("engine_source_sha256")
    header_sha = report.get("diagnostic_source_sha256")
    if not isinstance(engine_sha, str) or len(engine_sha) != 64 or not isinstance(header_sha, str) or len(header_sha) != 64:
        raise DiagnosticError("upstream archived source identities are missing")
    provenance_sources = record.get("provenance", {}).get("source_hashes", {})
    if engine_sha != provenance_sources.get("engine", {}).get("sha256") or header_sha != provenance_sources.get("diagnostic_header", {}).get("sha256"):
        raise DiagnosticError("upstream archived source identities disagree")
    return {"engine": {"sha256": engine_sha}, "diagnostic_header": {"sha256": header_sha}}


def validate_void_capture(source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    adjudication_path = source_run / "adjudication.json"
    manifest_path = source_run / "run_manifest.json"
    if not adjudication_path.is_file() or r2a.sha256_file(adjudication_path) != SOURCE_VOID_ADJUDICATION_SHA:
        raise DiagnosticError("combined source VOID adjudication binding mismatch")
    if not manifest_path.is_file() or r2a.sha256_file(manifest_path) != SOURCE_VOID_MANIFEST_SHA:
        raise DiagnosticError("combined source VOID manifest binding mismatch")
    record = read_json(adjudication_path, "combined source VOID adjudication")
    manifest = read_json(manifest_path, "combined source VOID manifest")
    if record.get("schema") != "strat01_combined_rms_q5q8_adjudication_v1" or record.get("status") != "VOID_COMBINED_RMS_Q5Q8_PROPAGATION" or record.get("errors") != ["candidate source hash mismatch"] or record.get("donor_graph_executions") != 1:
        raise DiagnosticError("combined source VOID state mismatch")
    if manifest.get("schema") != "strat01_combined_rms_q5q8_run_manifest_v1" or manifest.get("status") != "VOID_COMBINED_RMS_Q5Q8_PROPAGATION" or manifest.get("errors") != ["candidate source hash mismatch"] or manifest.get("donor_graph_executions") != 1:
        raise DiagnosticError("combined source VOID manifest state mismatch")
    command = record.get("provenance", {}).get("commands", {}).get("candidate", {})
    if command.get("returncode") != 0:
        raise DiagnosticError("combined source candidate process did not complete successfully")
    return record, manifest


def validate_candidate(root: Path, model: Path, sources: dict[str, dict[str, str]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    report = read_json(root / "strat01_combined_rms_q5q8.json", "combined C report")
    required = {"command", "state", "self_certifies_pass", "model", "baseline_inputs", "engine_source_sha256", "diagnostic_source_sha256", "kb_checkpoints", "ffn_output_sha256", "q8_census", "compiler_family", "donor_graph_executions", "timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-combined-rms-q5q8" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise DiagnosticError("combined C report schema/state mismatch")
    item = report["model"]
    if Path(item.get("path", "")).resolve(strict=True) != model.resolve(strict=True) or item.get("bytes") != r2a.EXPECTED_BYTES or item.get("sha256") != r2a.EXPECTED_SHA256:
        raise DiagnosticError("combined artifact identity mismatch")
    expected_inputs = {"attn_norm": {"path": str(upstream.BASELINE_ATTN.resolve()), "sha256": upstream.BASELINE_ATTN_SHA}, "ffn_norm": {"path": str(upstream.BASELINE_FFN.resolve()), "sha256": upstream.BASELINE_FFN_SHA}}
    if report["baseline_inputs"] != expected_inputs or report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["diagnostic_header"]["sha256"]:
        raise DiagnosticError("combined input/source identity mismatch")
    if report["compiler_family"] != "clang" or report["donor_graph_executions"] != 1 or report["timing_or_rate_claim"] is not None:
        raise DiagnosticError("combined execution contract mismatch")
    expected_kb = {arm: {"q8_sha256": KB_Q8_SHA, "output_sha256": KB_OUT_SHA} for arm in ARMS}
    if report["kb_checkpoints"] != expected_kb:
        raise DiagnosticError("combined K-B checkpoint report mismatch")
    for arm in ARMS:
        q8 = root / f"{arm}_kb_q8.bin"; out = root / f"{arm}_kb_q_abs.f32le"
        if not q8.is_file() or q8.stat().st_size != 34_816 or r2a.sha256_file(q8) != KB_Q8_SHA or not out.is_file() or out.stat().st_size != 524_288 or r2a.sha256_file(out) != KB_OUT_SHA:
            raise DiagnosticError(f"combined K-B checkpoint payload mismatch: {arm}")
    tensors: dict[str, np.ndarray] = {}; caches: dict[str, np.ndarray] = {}
    for arm in ARMS:
        manifest = read_json(root / f"{arm}_manifest.json", f"combined {arm} manifest")
        if set(manifest) != {"arm", "payload_encoding", "shape_order", "tensors", "cache"} or manifest["arm"] != arm or manifest["payload_encoding"] != "IEEE-754 binary32 little-endian":
            raise DiagnosticError(f"combined {arm} manifest mismatch")
        entries = {entry.get("name"): entry for entry in manifest["tensors"] if isinstance(entry, dict)}
        if set(entries) != set(r2a.SHAPES) or len(entries) != len(manifest["tensors"]):
            raise DiagnosticError(f"combined {arm} tensor set mismatch")
        for name, shape in r2a.SHAPES.items():
            entry = entries[name]; count = math.prod(shape)
            if entry.get("logical_shape") != shape or entry.get("op") != r2a.EXPECTED_OPS[name] or entry.get("ordinal") != r2a.EXPECTED_ORDINALS[name] or entry.get("type") != "F32" or entry.get("byte_count") != count * 4:
                raise DiagnosticError(f"combined {arm}/{name} metadata mismatch")
            path = upstream.contained(root, entry.get("path", ""), count * 4, f"combined {arm}/{name}")
            tensors[f"{arm}/{name}"] = upstream.load_f32(path, count, entry["sha256"], f"combined {arm}/{name}")
        cache = manifest["cache"]
        if cache.get("storage") != "F16" or cache.get("layout") != "layer,slot,576[latent512,rope64]" or cache.get("row_length") != 576 or cache.get("no_separate_v_cache") is not True:
            raise DiagnosticError(f"combined {arm} cache contract mismatch")
        checkpoints = (("final", 8), ("prefix7", 7)) if arm == "cached7p1" else (("final", 8),)
        for checkpoint, rows in checkpoints:
            entry = cache.get(checkpoint)
            if not isinstance(entry, dict) or entry.get("occupied_slots") != list(range(rows)) or entry.get("absolute_positions") != list(range(rows)) or entry.get("payload_type") != "dequantized-F32LE":
                raise DiagnosticError(f"combined {arm}/{checkpoint} cache metadata mismatch")
            path = upstream.contained(root, entry.get("path", ""), rows * 576 * 4, f"combined {arm}/{checkpoint} cache")
            caches[f"{arm}/{checkpoint}"] = upstream.load_f32(path, rows * 576, entry["sha256"], f"combined {arm}/{checkpoint} cache")
    ffn: dict[str, np.ndarray] = {}; hashes = report["ffn_output_sha256"]
    if set(hashes) != set(ARMS):
        raise DiagnosticError("combined FFN arm set mismatch")
    for arm in ARMS:
        if set(hashes[arm]) != {"norm", "up", "gate"}:
            raise DiagnosticError(f"combined {arm} FFN hash schema mismatch")
        for name, short in (("ffn_norm-0", "norm"), ("ffn_up-0", "up"), ("ffn_gate-0", "gate")):
            count = math.prod(FFN_SHAPES[name]); path = root / f"{arm}_double_{name}.f32"
            ffn[f"{arm}/{name}"] = upstream.load_f32(path, count, hashes[arm][short], f"combined {arm}/{name}")
    census: dict[str, Any] = {}
    for key, short in (("attention", "attn"), ("ffn", "ffn")):
        before, after = root / f"baseline_{short}_norm.q8k", root / f"candidate_{short}_norm.q8k"
        if not before.is_file() or not after.is_file() or before.stat().st_size != after.stat().st_size or before.stat().st_size != 48 * upstream.Q8_BLOCK_BYTES:
            raise DiagnosticError(f"combined {key} Q8 census payload mismatch")
        a, b = before.read_bytes(), after.read_bytes(); measured = {"changed_blocks": sum(a[i:i+upstream.Q8_BLOCK_BYTES] != b[i:i+upstream.Q8_BLOCK_BYTES] for i in range(0, len(a), upstream.Q8_BLOCK_BYTES)), "total_blocks": 48, "changed_bytes": sum(x != y for x, y in zip(a, b)), "total_bytes": len(a)}
        if report["q8_census"].get(key) != measured:
            raise DiagnosticError(f"combined {key} Q8 census mismatch")
        census[key] = measured
    return tensors, caches, ffn, {"report": report, "q8_census": census}


def load_accepted_ffn() -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}; root = upstream.RUNG2B_SOURCE_RUN / "c_engine"
    for arm in ARMS:
        manifest = read_json(root / f"{arm}_manifest.json", f"accepted FFN {arm}")
        entries = {item["name"]: item for item in manifest["tensors"]}
        for name in FFN_NAMES:
            item = entries[name]; count = math.prod(FFN_SHAPES[name]); path = Path(item["path"])
            loaded[f"{arm}/{name}"] = upstream.load_f32(path, count, item["sha256"], f"accepted FFN {arm}/{name}")
    return loaded


def adjudicate(candidate: dict[str, np.ndarray], caches: dict[str, np.ndarray], ffn: dict[str, np.ndarray], reference: dict[str, np.ndarray], reference_cache: dict[str, np.ndarray], reference_ffn: dict[str, np.ndarray], accepted: dict[str, np.ndarray], accepted_cache: dict[str, np.ndarray], accepted_ffn: dict[str, np.ndarray], prior: dict[str, np.ndarray], prior_cache: dict[str, np.ndarray], prior_ffn: dict[str, np.ndarray], census: dict[str, Any], kb_prior: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []; tensor_results=[]; cache_results=[]; continuity=[]; ffn_results=[]
    for arm in ARMS:
        for name, shape in r2a.SHAPES.items():
            limits = r2a.TERMINAL_LIMITS if name == "ffn_inp-0" else r2a.GENERAL_LIMITS
            result = r2a.judged_metrics(candidate[f"{arm}/{name}"], reference[f"{arm}/{name}"], limits); result.update({"arm": arm, "tensor": name, "accepted_float": r2a.metrics(accepted[f"{arm}/{name}"], reference[f"{arm}/{name}"]), "upstream_double": r2a.metrics(prior[f"{arm}/{name}"], reference[f"{arm}/{name}"])}); tensor_results.append(result)
            if not result["pass"]: failures.append(f"tensor/{arm}/{name}")
    for key in sorted(reference_cache):
        result = r2a.judged_metrics(caches[key], reference_cache[key], r2a.GENERAL_LIMITS); result.update({"checkpoint": key, "accepted_float": r2a.metrics(accepted_cache[key], reference_cache[key]), "upstream_double": r2a.metrics(prior_cache[key], reference_cache[key])}); cache_results.append(result)
        if not result["pass"]: failures.append(f"cache/{key}")
    for name, shape in r2a.SHAPES.items():
        result = r2a.judged_metrics(r2a.token7(candidate[f"cached7p1/{name}"], shape), r2a.token7(candidate[f"prefill8/{name}"], shape), upstream.CONTINUITY_LIMITS); result["tensor"] = name; continuity.append(result)
        if not result["pass"]: failures.append(f"continuity/{name}")
    for arm in ARMS:
        for name in FFN_NAMES:
            result = r2a.judged_metrics(ffn[f"{arm}/{name}"], reference_ffn[f"{arm}/{name}"], r2a.GENERAL_LIMITS); result.update({"arm": arm, "tensor": name, "accepted_float": r2a.metrics(accepted_ffn[f"{arm}/{name}"], reference_ffn[f"{arm}/{name}"]), "upstream_double": r2a.metrics(prior_ffn[f"{arm}/{name}"], reference_ffn[f"{arm}/{name}"])}); ffn_results.append(result)
            if not result["pass"]: failures.append(f"ffn/{arm}/{name}")
    swapped = {"up_as_gate": r2a.judged_metrics(ffn["prefill8/ffn_up-0"], reference_ffn["prefill8/ffn_gate-0"], r2a.GENERAL_LIMITS), "gate_as_up": r2a.judged_metrics(ffn["prefill8/ffn_gate-0"], reference_ffn["prefill8/ffn_up-0"], r2a.GENERAL_LIMITS)}
    combined_source = HEADER.read_text(encoding="utf-8"); upstream_source = upstream.HEADER.read_text(encoding="utf-8")
    controls = {"prior_kb_controls": bool(kb_prior.get("adjudication", {}).get("controls")) and all(kb_prior["adjudication"]["controls"].values()), "swapped_targets_reject": not swapped["up_as_gate"]["pass"] and not swapped["gate_as_up"]["pass"], "two_upstream_double_sites": combined_source.count("strat01_r2b_rmsnorm_double_sum(") == 2, "one_final_double_site": "strat01_r2b_rmsnorm_double_sum(input->ffn_inp" in upstream_source, "one_combined_kb_site": combined_source.count("strat01_combined_kb_range(") == 2}
    if not all(controls.values()):
        raise DiagnosticError("combined planted/source control failed")
    status = "COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES" if not failures else "COMBINED_RMS_Q5Q8_INSUFFICIENT_FOR_PROJECTION_GATES"
    return {"status": status, "failures": failures, "tensor_results": tensor_results, "cache_results": cache_results, "continuity_results": continuity, "ffn_results": ffn_results, "q8_census": census, "controls": controls, "swapped_targets": swapped}


def adjudicate_existing_run(source_run: Path, output: Path, model: Path) -> int:
    source_run = source_run.resolve(strict=True)
    if not source_run.is_dir():
        raise SystemExit(f"existing run is not a directory: {source_run}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status = "VOID_COMBINED_RMS_Q5Q8_PROPAGATION"
    errors: list[str] = []
    adjudication: dict[str, Any] = {"status": "NOT_RUN"}
    candidate_record: dict[str, Any] = {}
    sources: dict[str, dict[str, str]] = {}
    source_record: dict[str, Any] = {}
    try:
        source_record, _ = validate_void_capture(source_run)
        sources = source_inventory()
        kb_prior = validate_prior_bindings()
        if not model.is_file() or model.stat().st_size != r2a.EXPECTED_BYTES or r2a.sha256_file(model) != r2a.EXPECTED_SHA256:
            raise DiagnosticError("accepted model identity mismatch")
        candidate, caches, ffn, candidate_record = validate_candidate(source_run / "candidate", model, sources)
        reference, ref_record, reference_cache = r2a.validate_reference_outputs(upstream.RUNG2A_REFERENCE, model)
        reference_ffn, ref_ffn_record = r2b.validate_reference(upstream.RUNG2B_REFERENCE, model)
        baseline_report = read_json(upstream.ATTENTION_RUN / "c_engine/strat01_rung2a.json", "accepted attention report")
        accepted_sources = {"engine": {"sha256": baseline_report.get("engine_source_sha256")}, "rung2a_header": {"sha256": baseline_report.get("rung2a_source_sha256")}}
        accepted, accepted_cache, _ = r2a.validate_c_outputs(upstream.ATTENTION_RUN / "c_engine", accepted_sources, model)
        accepted_ffn = load_accepted_ffn()
        prior_sources = archived_upstream_sources()
        prior, prior_cache, prior_ffn, _ = upstream.validate_candidate(upstream.DEFAULT_OUTPUT / "candidate", model, prior_sources)
        adjudication = adjudicate(candidate, caches, ffn, reference, reference_cache, reference_ffn, accepted, accepted_cache, accepted_ffn, prior, prior_cache, prior_ffn, candidate_record["q8_census"], kb_prior)
        adjudication["reference_records"] = {"rung2a": ref_record, "rung2b": ref_ffn_record}
        status = adjudication["status"]
    except (DiagnosticError, r2a.RunnerError, r2b.RunnerError, upstream.DiagnosticError) as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "mode": "offline_revalidation_and_adjudication_only",
        "new_donor_graph_executions": 0,
        "new_reference_graph_executions": 0,
        "source_run": str(source_run),
        "source_adjudication_sha256": r2a.sha256_file(source_run / "adjudication.json"),
        "source_run_manifest_sha256": r2a.sha256_file(source_run / "run_manifest.json"),
        "source_run_git_head": source_record.get("provenance", {}).get("git_head") if source_record else None,
        "adjudicator_git_head": r2a.git_value(["git", "rev-parse", "HEAD"]),
        "adjudicator_git_status_porcelain": r2a.git_value(["git", "status", "--porcelain"]),
        "source_hashes": sources,
        "artifact": {"path": str(model), "expected_bytes": r2a.EXPECTED_BYTES, "expected_sha256": r2a.EXPECTED_SHA256},
        "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "cwd": os.getcwd()},
    }
    record = {
        "schema": "strat01_combined_rms_q5q8_offline_adjudication_v1",
        "status": status,
        "scope": "offline validation and frozen-gate adjudication of the preserved combined RMS/Q5Q8 capture",
        "errors": errors,
        "donor_graph_executions": 1,
        "new_donor_graph_executions": 0,
        "adjudication": adjudication,
        "candidate_record": candidate_record,
        "non_claims": ["production repair", "Rung 2C", "later layers", "quality", "generation", "RAM", "rate"],
        "provenance": provenance,
    }
    manifest = {
        "schema": "strat01_combined_rms_q5q8_offline_run_manifest_v1",
        "status": status,
        "errors": errors,
        "source_donor_graph_executions": 1,
        "new_donor_graph_executions": 0,
        "new_reference_graph_executions": 0,
        "provenance": provenance,
    }
    r2a.write_json(output / "adjudication.json", record)
    r2a.write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "source_run": str(source_run), "errors": errors}, indent=2))
    return 0 if status in {"COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES", "COMBINED_RMS_Q5Q8_INSUFFICIENT_FOR_PROJECTION_GATES"} else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--model", type=Path, default=MODEL); parser.add_argument("--output-dir", type=Path); parser.add_argument("--apparatus-only", action="store_true"); parser.add_argument("--adjudicate-existing", type=Path); args = parser.parse_args()
    if args.apparatus_only and args.adjudicate_existing is not None:
        raise SystemExit("--apparatus-only and --adjudicate-existing are mutually exclusive")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OFFLINE_OUTPUT if args.adjudicate_existing is not None else DEFAULT_OUTPUT
    model, output = args.model.resolve(), args.output_dir.resolve()
    if args.adjudicate_existing is not None:
        return adjudicate_existing_run(args.adjudicate_existing, output, model)
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True, exist_ok=True); started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter(); status="VOID_COMBINED_RMS_Q5Q8_PROPAGATION"; errors=[]; commands={}; sources={}; adjudication={"status":"NOT_RUN"}; candidate_record={}; binary=None; compiler=shutil.which("clang"); donor_graph_executions=0
    try:
        sources=source_inventory(); kb_prior=validate_prior_bindings()
        if not compiler: raise DiagnosticError("clang is unavailable")
        if not args.apparatus_only:
            relative=[Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
            if subprocess.run(["git","diff","--quiet","HEAD","--",*map(str,relative)],cwd=ROOT,check=False).returncode or any(subprocess.run(["git","ls-files","--error-unmatch",str(path)],cwd=ROOT,capture_output=True,check=False).returncode for path in relative): raise DiagnosticError("combined implementation/protocol differs from HEAD")
            if not model.is_file() or model.stat().st_size!=r2a.EXPECTED_BYTES or r2a.sha256_file(model)!=r2a.EXPECTED_SHA256: raise DiagnosticError("accepted model identity mismatch")
        commands["clang_version"]=upstream.run_command([compiler,"--version"],output,"clang_version",30);r2a.require_ok(commands["clang_version"],"clang version")
        binary=output/"engine_combined_rms_q5q8.exe";commands["compile"]=upstream.run_command([compiler,*r2a.COMPILE_FLAGS,str(ENGINE),"-o",str(binary),"-lm"],output,"compile",600);r2a.require_ok(commands["compile"],"C build")
        commands["selftest"]=upstream.run_command([str(binary),"--strat01-combined-rms-q5q8-selftest"],output,"selftest",300);r2a.require_ok(commands["selftest"],"combined selftest")
        commands["python_tests"]=upstream.run_command([sys.executable,"-B","-m","unittest","-v","benchmarks.donor_adaptation.engine.test_strat01_combined_rms_q5q8"],output,"python_tests",300);r2a.require_ok(commands["python_tests"],"combined Python tests")
        if args.apparatus_only: status="APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            root=output/"candidate";root.mkdir();donor_graph_executions=1;commands["candidate"]=upstream.run_command([str(binary),"--strat01-combined-rms-q5q8",str(model),"--baseline-attn-norm",str(upstream.BASELINE_ATTN.resolve()),"--baseline-ffn-norm",str(upstream.BASELINE_FFN.resolve()),"--out-dir",str(root)],output,"candidate",21600);r2a.require_ok(commands["candidate"],"combined candidate")
            sources=source_inventory();candidate,caches,ffn,candidate_record=validate_candidate(root,model,sources);reference,ref_record,reference_cache=r2a.validate_reference_outputs(upstream.RUNG2A_REFERENCE,model);reference_ffn,ref_ffn_record=r2b.validate_reference(upstream.RUNG2B_REFERENCE,model)
            baseline_report=read_json(upstream.ATTENTION_RUN/"c_engine/strat01_rung2a.json","accepted attention report");accepted_sources={"engine":{"sha256":baseline_report.get("engine_source_sha256")},"rung2a_header":{"sha256":baseline_report.get("rung2a_source_sha256")}};accepted,accepted_cache,_=r2a.validate_c_outputs(upstream.ATTENTION_RUN/"c_engine",accepted_sources,model);accepted_ffn=load_accepted_ffn()
            prior_sources=archived_upstream_sources();prior,prior_cache,prior_ffn,_=upstream.validate_candidate(upstream.DEFAULT_OUTPUT/"candidate",model,prior_sources)
            adjudication=adjudicate(candidate,caches,ffn,reference,reference_cache,reference_ffn,accepted,accepted_cache,accepted_ffn,prior,prior_cache,prior_ffn,candidate_record["q8_census"],kb_prior);adjudication["reference_records"]={"rung2a":ref_record,"rung2b":ref_ffn_record};status=adjudication["status"]
    except (DiagnosticError,r2a.RunnerError,r2b.RunnerError,upstream.DiagnosticError) as exc: errors.append(str(exc))
    except Exception as exc: errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance={"started_utc":started_utc,"finished_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.perf_counter()-started,"git_head":r2a.git_value(["git","rev-parse","HEAD"]),"git_status_porcelain":r2a.git_value(["git","status","--porcelain"]),"source_hashes":sources,"artifact":{"path":str(model),"expected_bytes":r2a.EXPECTED_BYTES,"expected_sha256":r2a.EXPECTED_SHA256},"environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd(),"clang_path":compiler},"binary":{"path":str(binary) if binary else None,"sha256":r2a.sha256_file(binary) if binary and binary.is_file() else None},"commands":commands}
    record={"schema":"strat01_combined_rms_q5q8_adjudication_v1","status":status,"errors":errors,"donor_graph_executions":donor_graph_executions,"adjudication":adjudication,"candidate_record":candidate_record,"non_claims":["production repair","Rung 2C","later layers","quality","generation","RAM","rate"],"provenance":provenance};r2a.write_json(output/"adjudication.json",record);r2a.write_json(output/"run_manifest.json",{"schema":"strat01_combined_rms_q5q8_run_manifest_v1","status":status,"errors":errors,"donor_graph_executions":donor_graph_executions,"provenance":provenance});print(json.dumps({"status":status,"output":str(output),"errors":errors},indent=2))
    return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION","COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES","COMBINED_RMS_Q5Q8_INSUFFICIENT_FOR_PROJECTION_GATES"} else 2


if __name__ == "__main__": raise SystemExit(main())
