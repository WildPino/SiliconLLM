#!/usr/bin/env python3
"""Bounded STRAT-02 W4-BF16-v2 paired quality runner.

The real worker is intentionally reachable only through ``--run``.  It binds
the preregistered export, audits the frozen corpus/tokenizer, constructs the
candidate through ``strat02_w4_bf16_candidate_loader`` (never the F32 source
loader), runs all 48 calibration rows as a finite operational check, then all
96 heldout rows in original JSONL order.  Outputs are write-once and contain
no raw text, token IDs, or logits.  ``--selftest`` is entirely synthetic.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CORPUS = HERE / "corpus" / "strat02_document_holdout_v1"
QUALITY_BRIEF = PROJECT / "docs" / "research" / "donor_adaptation" / "briefs" / "BRIEF_STRAT_02_W4_BF16_QUALITY_SCORE.md"
BINDING_BRIEF = PROJECT / "docs" / "research" / "donor_adaptation" / "briefs" / "BRIEF_STRAT_02_W4_BF16_QUALITY_ARTIFACT_BINDING.md"
BOUND_EXPORT = HERE / "results" / "strat02_w4_bf16_full_export_20260917_143753"
TEACHER_SCORES = HERE / "results" / "strat02_teacher_baseline_20260917_105941" / "teacher_scores.jsonl"
TEACHER_REPEATABILITY = HERE / "results" / "strat02_teacher_repeatability_20260917_142728" / "supervisor_result.json"

REPOSITORY = "allenai/StdMoE_1b14b_1T_Preanneal"
REVISION = "d2a4949c9d4ad6cf47fbac131f7e020077332b21"
EXPECTED_MANIFEST_SHA256 = "56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749"
EXPECTED_CALIB_SHA256 = "f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f"
EXPECTED_HELDOUT_SHA256 = "450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e"
EXPECTED_CALIB_IDS_SHA256 = "430e44946673ed06cc2afcbfb373611c366dd2c363a3af652ffa42bf5a0869f4"
EXPECTED_HELDOUT_IDS_SHA256 = "5a4cf31ec7db0f3bc541f2e75a78b133a50cd3d6174abb26ef81e08c45e65289"
EXPECTED_CALIB_DOCUMENTS = 48
EXPECTED_HELDOUT_DOCUMENTS = 96
EXPECTED_PER_CATEGORY_CALIB = 16
EXPECTED_PER_CATEGORY_HELDOUT = 32
EXPECTED_CALIB_TOKENS = 88756
EXPECTED_HELDOUT_TOKENS = 181385
EXPECTED_CALIB_BYTES = 362405
EXPECTED_HELDOUT_BYTES = 746161
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_916
BPB_UPPER_CI_MAX = 0.02
CHUNK_SIZE = 128
MIN_LAUNCH_RAM_BYTES = 55 * 1024**3
MIN_OUTPUT_DISK_BYTES = 1 * 1024**3
MIN_RUNTIME_RAM_BYTES = 8 * 1024**3
MAX_WORKER_MEMORY_BYTES = 70 * 1024**3
MAX_WALL_SECONDS = 6 * 60 * 60
MONITOR_INTERVAL_SECONDS = 5.0
WORKER_TOKEN_ENV = "STRAT02_W4_BF16_QUALITY_PARENT_TOKEN"

BOUND_CONTROL_SHA256 = {
    "artifact_manifest.json": "06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c",
    "plan_manifest.json": "1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42",
    "supervisor_result.json": "d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca",
}
PREREQUISITE_SHA256 = {
    str(TEACHER_SCORES): "96d284da8f223d613039cb9e2a87b24e7b6130701c0873eea0854c3e8290b224",
    str(TEACHER_REPEATABILITY): "89390013e926fddfad5c3bd651b29f1ae861de2bb98996d297300343cf65d41f",
}

SOURCE_FILES = (
    "benchmarks/donor_adaptation/density/strat02_w4_bf16_candidate_loader.py",
    "benchmarks/donor_adaptation/density/strat02_teacher_baseline.py",
    "benchmarks/donor_adaptation/density/strat02_teacher_repeatability.py",
    "benchmarks/donor_adaptation/density/strat02_score.py",
    "benchmarks/donor_adaptation/density/strat02_token_audit.py",
    "benchmarks/donor_adaptation/density/strat02_bounded_smoke.py",
    "benchmarks/donor_adaptation/density/strat02_mmap_teacher.py",
    "benchmarks/donor_adaptation/density/strat02_w4_bf16_export_verify.py",
    "benchmarks/donor_adaptation/density/strat02_w4_bf16_codec.py",
)


class GateError(RuntimeError):
    """A fixed provenance, protocol, data, or write-once requirement failed."""


class ResourceGateError(GateError):
    """A launch/runtime resource guard failed."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _load_local_module(filename: str, module_name: str) -> Any:
    project_path = str(PROJECT)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot load local module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise GateError(f"write-once output already exists: {path}") from exc


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    _append_jsonl(path, {"utc_timestamp": _utc_now(), **event})


def _prepare_output(output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GateError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "supervisor_manifest.json",
        "log": output_dir / "supervisor_log.jsonl",
        "calibration": output_dir / "calibration_checks.jsonl",
        "candidate": output_dir / "candidate_scores.jsonl",
        "worker": output_dir / "worker_result.json",
        "result": output_dir / "supervisor_result.json",
        "stdout": output_dir / "worker_stdout.log",
        "stderr": output_dir / "worker_stderr.log",
    }
    if any(path.exists() for path in paths.values()):
        raise GateError("refusing to overwrite write-once quality output")
    try:
        with paths["log"].open("x", encoding="utf-8", newline="\n"):
            pass
    except FileExistsError as exc:
        raise GateError(f"write-once supervisor log already exists: {paths['log']}") from exc
    return paths


def _source_hashes() -> dict[str, str]:
    paths = {"quality_brief": QUALITY_BRIEF, "artifact_binding_brief": BINDING_BRIEF,
             "runner": Path(__file__).resolve()}
    paths.update({name: PROJECT / name for name in SOURCE_FILES})
    observed: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise GateError(f"provenance source is missing: {path}")
        observed[label] = _sha256_file(path)
    return observed


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verify_parent_provenance(supervisor_manifest: Path, *, result_path: Path,
                              calibration_path: Path, candidate_path: Path) -> dict[str, str]:
    """Verify the exact parent hash map before importing scoring/loader modules."""
    manifest_path = supervisor_manifest.expanduser().resolve()
    output_dir = result_path.resolve().parent
    paths = [result_path.resolve(), calibration_path.resolve(), candidate_path.resolve()]
    if manifest_path.name != "supervisor_manifest.json" or manifest_path.parent != output_dir:
        raise GateError("supervisor manifest must be supervisor_manifest.json inside the output directory")
    if any(path.parent != output_dir for path in paths):
        raise GateError("worker result and score files must share the supervisor output directory")
    manifest = _load_json_object(manifest_path)
    expected = manifest.get("provenance_sha256") if manifest is not None else None
    if (not isinstance(expected, dict) or not expected or
            any(not isinstance(key, str) or not isinstance(value, str) for key, value in expected.items())):
        raise GateError("supervisor manifest has no valid provenance_sha256 map")
    observed = _source_hashes()
    if observed != expected:
        raise GateError("worker source provenance does not exactly match supervisor_manifest.json")
    return observed


def _validate_pin_constants(candidate: Any) -> None:
    if Path(candidate.BOUND_EXPORT).resolve() != BOUND_EXPORT.resolve():
        raise GateError("candidate loader bound export path differs from quality binding")
    if dict(candidate.BOUND_CONTROL_SHA256) != BOUND_CONTROL_SHA256:
        raise GateError("candidate loader artifact control pins differ from quality binding")
    if {str(path): digest for path, digest in candidate.QUALITY_PREREQUISITE_SHA256.items()} != PREREQUISITE_SHA256:
        raise GateError("candidate loader teacher prerequisite pins differ from quality binding")


def _require_bound_teacher_scores(path: Path) -> None:
    if not path.resolve().samefile(TEACHER_SCORES):
        raise GateError("teacher denominator is not the preregistered score file")
    if _sha256_file(path) != PREREQUISITE_SHA256[str(TEACHER_SCORES)]:
        raise GateError("teacher denominator SHA-256 differs from preregistered pin")


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, expected_documents: int,
                  expected_per_category: int, split: str) -> None:
    if len(rows) != expected_documents:
        raise GateError(f"{split} must contain exactly {expected_documents} rows; got {len(rows)}")
    seen: set[str] = set()
    counts = {"code": 0, "technical_general": 0, "prose": 0}
    for number, row in enumerate(rows, 1):
        identifier, category, text = row.get("source_document_id"), row.get("category"), row.get("text")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise GateError(f"{split}:{number}: invalid or duplicate source_document_id")
        if category not in counts or not isinstance(text, str) or not text.encode("utf-8"):
            raise GateError(f"{split}:{number}: invalid category or empty UTF-8 text")
        seen.add(identifier)
        counts[category] += 1
    if any(value != expected_per_category for value in counts.values()):
        raise GateError(f"{split} category counts differ from frozen protocol: {counts}")


def _audit_full_corpus(audit: Any, *, manifest: Path, calib: Path, heldout: Path,
                      snapshot: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Use the existing full corpus/token audit before any candidate forward."""
    for label, path, expected in (("manifest", manifest, EXPECTED_MANIFEST_SHA256),
                                  ("calib", calib, EXPECTED_CALIB_SHA256),
                                  ("heldout", heldout, EXPECTED_HELDOUT_SHA256)):
        if _sha256_file(path) != expected:
            raise GateError(f"frozen {label} SHA-256 differs from preregistered pin")
    report = audit.audit(SimpleNamespace(manifest=manifest, calib=calib, heldout=heldout,
                                         tokenizer_snapshot=snapshot))
    if not report.get("ok"):
        raise GateError("frozen corpus/tokenizer audit failed: " + "; ".join(report.get("errors", [])))
    errors: list[str] = []
    calib_rows = audit.read_rows(calib, "calib", errors)
    heldout_rows = audit.read_rows(heldout, "heldout", errors)
    if errors:
        raise GateError("cannot reread audited corpus rows: " + "; ".join(errors))
    _validate_rows(calib_rows, expected_documents=EXPECTED_CALIB_DOCUMENTS,
                   expected_per_category=EXPECTED_PER_CATEGORY_CALIB, split="calib")
    _validate_rows(heldout_rows, expected_documents=EXPECTED_HELDOUT_DOCUMENTS,
                   expected_per_category=EXPECTED_PER_CATEGORY_HELDOUT, split="heldout")
    token = report.get("tokenizer", {})
    split_report = token.get("splits", {}) if isinstance(token, dict) else {}
    if (split_report.get("calib", {}).get("ordered_token_ids_sha256") != EXPECTED_CALIB_IDS_SHA256 or
            split_report.get("heldout", {}).get("ordered_token_ids_sha256") != EXPECTED_HELDOUT_IDS_SHA256 or
            split_report.get("calib", {}).get("payload_tokens_total") != EXPECTED_CALIB_TOKENS or
            split_report.get("heldout", {}).get("payload_tokens_total") != EXPECTED_HELDOUT_TOKENS):
        raise GateError("audited corpus token-ID or total pin differs from the frozen protocol")
    return report.get("manifest", {}), calib_rows, heldout_rows, {
        "calib_documents": len(calib_rows), "heldout_documents": len(heldout_rows),
        "calib_tokens": EXPECTED_CALIB_TOKENS, "heldout_tokens": EXPECTED_HELDOUT_TOKENS,
        "calib_bytes": EXPECTED_CALIB_BYTES, "heldout_bytes": EXPECTED_HELDOUT_BYTES,
        "calib_order": "original calib.jsonl order", "heldout_order": "original heldout.jsonl order",
        "raw_text_reported": False, "token_ids_reported": False,
    }


def _metadata_preflight(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """Metadata/corpus-only gate; it never verifies payload bytes or runs a forward."""
    candidate = _load_local_module("strat02_w4_bf16_candidate_loader.py", "_strat02_quality_candidate_preflight")
    _validate_pin_constants(candidate)
    candidate.check_bound_controls(args.artifact)
    candidate.check_quality_prerequisites()
    _require_bound_teacher_scores(args.teacher_scores)
    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_quality_bounded_preflight")
    versions = bounded._pinned_environment()
    report = candidate.teacher.preflight(snapshot=args.snapshot)
    status = report.source_status
    if (status.get("repository") != REPOSITORY or status.get("revision") != REVISION or
            not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO" or
            not status.get("shards_present")):
        raise GateError("pinned offline teacher metadata or local shard presence failed")
    output_volume = args.output_dir.resolve() if args.output_dir else HERE
    available = int(bounded._psutil().virtual_memory().available)
    free = int(bounded._psutil().disk_usage(str(output_volume)).free)
    if available < MIN_LAUNCH_RAM_BYTES:
        raise ResourceGateError(f"quality launch requires >=55 GiB available RAM; observed {available}")
    if free < MIN_OUTPUT_DISK_BYTES:
        raise ResourceGateError(f"quality output requires >=1 GiB free space; observed {free}")
    audit = _load_local_module("strat02_token_audit.py", "_strat02_quality_audit_preflight")
    _, _, _, corpus_summary = _audit_full_corpus(
        audit, manifest=args.manifest, calib=args.calib, heldout=args.heldout, snapshot=report.snapshot
    )
    metadata = {
        "utc_timestamp": _utc_now(), "parent_pid": os.getpid(), "runtime_versions": versions,
        "python": sys.version.split()[0], "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
        "offline_policy": {"hub_download_api": "not used", "candidate_loader_only": True,
                           "tokenizer_local_files_only": True, "source_f32_values_opened": False},
        "snapshot": str(report.snapshot), "teacher_metadata": dict(status),
        "available_physical_ram_bytes": available, "output_volume": str(output_volume),
        "output_volume_free_bytes": free, "artifact_control_sha256": dict(BOUND_CONTROL_SHA256),
        "teacher_prerequisite_sha256": dict(PREREQUISITE_SHA256), "corpus": corpus_summary,
        "payload_hashes": "not performed by metadata preflight",
        "heldout_scoring": "not performed by metadata preflight",
    }
    return bounded, report, metadata


def _expected_token_count(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list) or not ids:
        raise GateError("frozen document produced no payload token")
    return len(ids)


def _score_rows_once(*, model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]],
                    score: Any, output_path: Path, split: str, expected_documents: int,
                    expected_tokens: int, expected_bytes: int,
                    score_document: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Score a complete split exactly once, flushing only safe per-document rows."""
    _validate_rows(rows, expected_documents=expected_documents,
                   expected_per_category=(EXPECTED_PER_CATEGORY_CALIB if split == "calib" else EXPECTED_PER_CATEGORY_HELDOUT),
                   split=split)
    scorer = score_document or score.score_document
    totals = {"documents": 0, "tokens": 0, "bytes": 0, "bits": 0.0}
    categories = {category: {"documents": 0, "tokens": 0, "bytes": 0, "bits": 0.0}
                  for category in ("code", "technical_general", "prose")}
    try:
        handle = output_path.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise GateError(f"write-once score output already exists: {output_path}") from exc
    with handle:
        for ordinal, row in enumerate(rows, 1):
            started = time.monotonic()
            observation = scorer(model, tokenizer, row, chunk_size=CHUNK_SIZE)
            wall_seconds = time.monotonic() - started
            identifier, category = row["source_document_id"], row["category"]
            tokens, byte_count, bits = observation.tokens, observation.bytes, float(observation.bits)
            expected_row_tokens = _expected_token_count(tokenizer, row["text"])
            expected_row_bytes = len(row["text"].encode("utf-8"))
            if ((observation.source_document_id, observation.category) != (identifier, category) or
                    tokens != expected_row_tokens or byte_count != expected_row_bytes or
                    isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0 or
                    isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0 or
                    not math.isfinite(bits) or bits < 0 or not math.isfinite(wall_seconds) or wall_seconds < 0):
                raise GateError(f"{split}:{ordinal}: identity/token/byte/finite score binding failed")
            record = {"source_document_id": identifier, "category": category, "tokens": tokens,
                      "bytes": byte_count, "bits": bits, "wall_seconds": wall_seconds}
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            totals["documents"] += 1
            totals["tokens"] += tokens
            totals["bytes"] += byte_count
            totals["bits"] += bits
            bucket = categories[category]
            bucket["documents"] += 1
            bucket["tokens"] += tokens
            bucket["bytes"] += byte_count
            bucket["bits"] += bits
    if totals["tokens"] != expected_tokens or totals["bytes"] != expected_bytes:
        raise GateError(f"{split}: frozen totals differ ({totals['tokens']}/{expected_tokens}, {totals['bytes']}/{expected_bytes})")
    for bucket in categories.values():
        bucket["bpb_diagnostic"] = bucket["bits"] / bucket["bytes"]
    totals["bpb_diagnostic"] = totals["bits"] / totals["bytes"]
    totals["categories"] = categories
    totals["order"] = f"original {split}.jsonl order"
    totals["truncation"] = False
    return totals


def _score_candidate_splits(*, model: Any, tokenizer: Any, calib_rows: Sequence[Mapping[str, Any]],
                            heldout_rows: Sequence[Mapping[str, Any]], score: Any,
                            calibration_path: Path, candidate_path: Path,
                            score_document: Callable[..., Any] | None = None,
                            calibration_expected_tokens: int = EXPECTED_CALIB_TOKENS,
                            calibration_expected_bytes: int = EXPECTED_CALIB_BYTES,
                            heldout_expected_tokens: int = EXPECTED_HELDOUT_TOKENS,
                            heldout_expected_bytes: int = EXPECTED_HELDOUT_BYTES) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run calibration first; heldout scoring is unreachable when it fails."""
    calibration_summary = _score_rows_once(
        model=model, tokenizer=tokenizer, rows=calib_rows, score=score,
        output_path=calibration_path, split="calib", expected_documents=EXPECTED_CALIB_DOCUMENTS,
        expected_tokens=calibration_expected_tokens, expected_bytes=calibration_expected_bytes,
        score_document=score_document,
    )
    heldout_summary = _score_rows_once(
        model=model, tokenizer=tokenizer, rows=heldout_rows, score=score,
        output_path=candidate_path, split="heldout", expected_documents=EXPECTED_HELDOUT_DOCUMENTS,
        expected_tokens=heldout_expected_tokens, expected_bytes=heldout_expected_bytes,
        score_document=score_document,
    )
    return calibration_summary, heldout_summary


def _gate_adjudication(report: Mapping[str, Any]) -> str:
    upper = report.get("upper_one_sided_ci95_delta_bpb")
    if isinstance(upper, bool) or not isinstance(upper, (int, float)) or not math.isfinite(float(upper)):
        raise GateError("paired BPB upper CI is non-finite")
    return "PASS_BPB" if float(upper) <= BPB_UPPER_CI_MAX else "FAIL_BPB"


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _worker_failure(path: Path, *, status: str, exc: BaseException,
                    calibration_path: Path, candidate_path: Path) -> None:
    try:
        _write_json_once(path, {
            "ok": False, "status": status, "utc_timestamp": _utc_now(),
            "calibration_rows_persisted": _count_lines(calibration_path),
            "heldout_rows_persisted": _count_lines(candidate_path),
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "heldout_scoring_started": _count_lines(candidate_path) > 0,
            "raw_text_reported": False, "token_ids_reported": False, "logits_reported": False,
        })
    except (GateError, OSError):
        pass


def _worker(args: argparse.Namespace) -> int:
    result_path, calibration_path, candidate_path = (args.result.resolve(), args.calibration.resolve(), args.candidate.resolve())
    try:
        _verify_parent_provenance(args.supervisor_manifest, result_path=result_path,
                                  calibration_path=calibration_path, candidate_path=candidate_path)
        candidate = _load_local_module("strat02_w4_bf16_candidate_loader.py", "_strat02_quality_candidate_worker")
        _validate_pin_constants(candidate)
        bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_quality_bounded_worker")
        bounded._pinned_environment()
        audit = _load_local_module("strat02_token_audit.py", "_strat02_quality_audit_worker")
        score = _load_local_module("strat02_score.py", "_strat02_quality_score_worker")
        candidate.check_bound_controls(args.artifact)
        candidate.check_quality_prerequisites()
        _require_bound_teacher_scores(args.teacher_scores)
        report = candidate.teacher.preflight(snapshot=args.snapshot)
        _, calib_rows, heldout_rows, corpus_summary = _audit_full_corpus(
            audit, manifest=args.manifest, calib=args.calib, heldout=args.heldout, snapshot=report.snapshot
        )
        tokenizer, _, _, _ = audit.load_tokenizer(report.snapshot)
        score.validate_strat02_tokenizer(tokenizer)
        prepared = candidate.prepare_candidate(args.artifact, snapshot=report.snapshot, output_dir=args.output_dir)
        with candidate.load_candidate_model(prepared) as model:
            model.eval()
            calibration_summary, heldout_summary = _score_candidate_splits(
                model=model, tokenizer=tokenizer, calib_rows=calib_rows, heldout_rows=heldout_rows,
                score=score, calibration_path=calibration_path, candidate_path=candidate_path,
            )
        teacher_score = Path(args.teacher_scores).resolve()
        candidate.check_quality_prerequisites()
        _require_bound_teacher_scores(teacher_score)
        adjudication = score.adjudicate_pinned_files(teacher_score, candidate_path, heldout_rows, tokenizer)
        status = _gate_adjudication(adjudication)
        _write_json_once(result_path, {
            "ok": status == "PASS_BPB", "status": status, "utc_timestamp": _utc_now(),
            "artifact": str(args.artifact), "artifact_control_sha256": dict(BOUND_CONTROL_SHA256),
            "corpus": corpus_summary, "calibration": calibration_summary,
            "heldout": heldout_summary, "adjudication": adjudication,
            "protocol": {
                "candidate_loader_only": True, "source_f32_values_opened": False,
                "calibration_documents_scored_once": EXPECTED_CALIB_DOCUMENTS,
                "heldout_documents_scored_once": EXPECTED_HELDOUT_DOCUMENTS,
                "heldout_order": "original heldout.jsonl order", "chunk_size": CHUNK_SIZE,
                "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                "upper_ci95_gate": BPB_UPPER_CI_MAX, "heldout_scoring_started_after_calibration": True,
            },
            "task_rollout": "PENDING" if status == "PASS_BPB" else "NOT_RUN_BPB_FAIL",
            "notes": [
                "This is a paired donor BPB gate only; no full quality, task, rollout, native-C, or rate claim.",
                "No raw document text, token IDs, or logits are included; document IDs are binding keys.",
            ],
            "calibration_checks_sha256": _sha256_file(calibration_path),
            "candidate_scores_sha256": _sha256_file(candidate_path),
        })
        return 0 if status == "PASS_BPB" else 1
    except ResourceGateError as exc:
        _worker_failure(result_path, status="VOID_RESOURCE", exc=exc,
                        calibration_path=calibration_path, candidate_path=candidate_path)
        return 1
    except MemoryError as exc:
        _worker_failure(result_path, status="VOID_RESOURCE", exc=exc,
                        calibration_path=calibration_path, candidate_path=candidate_path)
        return 1
    except Exception as exc:
        _worker_failure(result_path, status="VOID_APPARATUS", exc=exc,
                        calibration_path=calibration_path, candidate_path=candidate_path)
        return 1


def _failure_classification(outcome: Any, worker_result: Mapping[str, Any] | None) -> str | None:
    if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
        return outcome.status
    if worker_result and worker_result.get("status") in {"PASS_BPB", "FAIL_BPB", "VOID_RESOURCE", "VOID_APPARATUS"}:
        if worker_result["status"] == "PASS_BPB" and outcome.exit_code == 0:
            return None
        return str(worker_result["status"])
    if outcome.exit_code != 0:
        return "WORKER_ERROR"
    return "WORKER_PROTOCOL_ERROR"


def _run_parent(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    provenance: dict[str, str] | None = None
    try:
        provenance = _source_hashes()
        bounded, report, preflight = _metadata_preflight(args)
    except Exception as exc:
        classification = "VOID_RESOURCE" if isinstance(exc, ResourceGateError) else "VOID_APPARATUS"
        _write_json_once(paths["manifest"], {"schema": "strat02_w4_bf16_quality_v1",
                                             "provenance_sha256": provenance or {}, "preflight": "failed"})
        _append_log(paths["log"], {"event": "parent_preflight_failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json_once(paths["result"], {"ok": False, "status": classification, "failure_classification": classification,
                                            "utc_timestamp": _utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps({"ok": False, "status": classification, "result": str(paths["result"])}, sort_keys=True))
        return 1
    manifest = {
        "schema": "strat02_w4_bf16_quality_v1",
        "purpose": "paired W4-BF16-v2 donor BPB gate; BPB-only, task/rollout pending",
        "parent_pid": os.getpid(), "provenance_sha256": provenance,
        "pins": {"repository": REPOSITORY, "revision": REVISION,
                 "quality_brief_sha256": _sha256_file(QUALITY_BRIEF),
                 "artifact_binding_brief_sha256": _sha256_file(BINDING_BRIEF),
                 "artifact_control_sha256": dict(BOUND_CONTROL_SHA256),
                 "teacher_prerequisite_sha256": dict(PREREQUISITE_SHA256),
                 "corpus_manifest_sha256": EXPECTED_MANIFEST_SHA256,
                 "calib_jsonl_sha256": EXPECTED_CALIB_SHA256,
                 "heldout_jsonl_sha256": EXPECTED_HELDOUT_SHA256,
                 "calib_ordered_token_ids_sha256": EXPECTED_CALIB_IDS_SHA256,
                 "heldout_ordered_token_ids_sha256": EXPECTED_HELDOUT_IDS_SHA256},
        "limits": {"launch_available_physical_ram_bytes_min": MIN_LAUNCH_RAM_BYTES,
                   "output_volume_free_bytes_min": MIN_OUTPUT_DISK_BYTES,
                   "runtime_available_physical_ram_bytes_min": MIN_RUNTIME_RAM_BYTES,
                   "worker_working_set_bytes_max": MAX_WORKER_MEMORY_BYTES,
                   "worker_private_commit_bytes_max": MAX_WORKER_MEMORY_BYTES,
                   "wall_seconds_max": MAX_WALL_SECONDS, "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS},
        "parent_preflight": preflight,
        "protocol": {"calibration_documents": EXPECTED_CALIB_DOCUMENTS, "heldout_documents": EXPECTED_HELDOUT_DOCUMENTS,
                     "calibration_score_calls": EXPECTED_CALIB_DOCUMENTS, "heldout_score_calls": EXPECTED_HELDOUT_DOCUMENTS,
                     "heldout_order": "original heldout.jsonl order", "chunk_size": CHUNK_SIZE,
                     "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                     "upper_ci95_gate": BPB_UPPER_CI_MAX, "candidate_loader_only": True,
                     "source_f32_values_opened": False, "raw_text_reported": False,
                     "token_ids_reported": False, "logits_reported": False,
                     "task_rollout": "pending until BPB gate"},
        "output": {"calibration": paths["calibration"].name, "candidate": paths["candidate"].name,
                   "worker_stdout": paths["stdout"].name, "worker_stderr": paths["stderr"].name,
                   "auto_resume": False},
    }
    _write_json_once(paths["manifest"], manifest)
    _append_log(paths["log"], {"event": "parent_preflight_passed", "parent_pid": os.getpid()})
    child_env = os.environ.copy()
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["TRANSFORMERS_OFFLINE"] = "1"
    child_env["USE_HUB_KERNELS"] = "NO"
    worker_python = bounded._direct_worker_python(child_env)
    token = secrets.token_urlsafe(32)
    child_env[WORKER_TOKEN_ENV] = token
    command = [worker_python, str(Path(__file__).resolve()), "--worker", "--artifact", str(args.artifact.resolve()),
               "--snapshot", str(report.snapshot), "--manifest", str(args.manifest.resolve()),
               "--calib", str(args.calib.resolve()), "--heldout", str(args.heldout.resolve()),
               "--teacher-scores", str(args.teacher_scores.resolve()), "--output-dir", str(args.output_dir.resolve()),
               "--calibration", str(paths["calibration"]), "--candidate", str(paths["candidate"]),
               "--result", str(paths["worker"]), "--supervisor-manifest", str(paths["manifest"]),
               "--worker-token", token]
    _append_log(paths["log"], {"event": "child_launch", "python": worker_python,
                               "termination_scope": "direct_worker_pid_only", "monitor_target": "direct_base_python_worker"})
    process: Any | None = None
    stdout_handle: Any | None = None
    stderr_handle: Any | None = None
    try:
        stdout_handle = paths["stdout"].open("x", encoding="utf-8", newline="\n")
        stderr_handle = paths["stderr"].open("x", encoding="utf-8", newline="\n")
        process = subprocess.Popen(command, cwd=str(HERE), env=child_env, stdin=subprocess.DEVNULL,
                                   stdout=stdout_handle, stderr=stderr_handle)
        _append_log(paths["log"], {"event": "child_pid_registered", "child_pid": process.pid})
        outcome = bounded._monitor_or_terminate(
            process, psutil=bounded._psutil(), expected_executable=Path(worker_python),
            wall_limit_seconds=MAX_WALL_SECONDS, interval_seconds=MONITOR_INTERVAL_SECONDS,
            on_sample=lambda sample: _append_log(paths["log"], {"event": "resource_sample", **sample}),
        )
        _append_log(paths["log"], {"event": "monitor_finished", **asdict(outcome)})
        worker_result = _load_json_object(paths["worker"])
        classification = _failure_classification(outcome, worker_result)
        worker_status = worker_result.get("status") if worker_result else None
        if worker_status in {"PASS_BPB", "FAIL_BPB"}:
            expected_calibration_sha = worker_result.get("calibration_checks_sha256")
            expected_candidate_sha = worker_result.get("candidate_scores_sha256")
            if (not paths["calibration"].is_file() or not paths["candidate"].is_file() or
                    expected_calibration_sha != _sha256_file(paths["calibration"]) or
                    expected_candidate_sha != _sha256_file(paths["candidate"])):
                classification = "VOID_APPARATUS"
                _append_log(paths["log"], {"event": "worker_score_artifact_hash_mismatch"})
        if classification is None:
            try:
                score = _load_local_module("strat02_score.py", "_strat02_quality_score_parent")
                persisted = score.read_score_jsonl(paths["candidate"])
                if len(persisted) != EXPECTED_HELDOUT_DOCUMENTS:
                    raise GateError("parent found an incomplete candidate score artifact")
            except Exception as exc:
                classification = "WORKER_PROTOCOL_ERROR"
                _append_log(paths["log"], {"event": "parent_artifact_validation_failed",
                                           "error_type": type(exc).__name__, "error": str(exc)})
        final_status = ("PASS_BPB" if classification is None else classification
                        if classification in {"FAIL_BPB", "VOID_RESOURCE", "VOID_APPARATUS"} else "INCOMPLETE")
        final: dict[str, Any] = {"ok": final_status == "PASS_BPB", "status": final_status,
                                 "failure_classification": classification, "utc_timestamp": _utc_now(),
                                 "parent_pid": os.getpid(), "child_pid": process.pid, "monitor": asdict(outcome),
                                 "worker_result": worker_result,
                                 "calibration_rows_persisted": _count_lines(paths["calibration"]),
                                 "heldout_rows_persisted": _count_lines(paths["candidate"]),
                                 "task_rollout": ("PENDING" if final_status == "PASS_BPB" else
                                                  "NOT_RUN_BPB_FAIL" if final_status == "FAIL_BPB" else "NOT_RUN_VOID")}
    except Exception as exc:
        if process is not None:
            bounded._terminate_only_child(process)
        _append_log(paths["log"], {"event": "parent_supervision_error", "error_type": type(exc).__name__, "error": str(exc)})
        final = {"ok": False, "status": "INCOMPLETE", "failure_classification": "VOID_APPARATUS",
                 "utc_timestamp": _utc_now(), "parent_pid": os.getpid(),
                 "child_pid": process.pid if process is not None else None,
                 "calibration_rows_persisted": _count_lines(paths["calibration"]),
                 "heldout_rows_persisted": _count_lines(paths["candidate"]),
                 "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        for handle in (stdout_handle, stderr_handle):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
    for label, path in (("calibration_checks_sha256", paths["calibration"]),
                        ("candidate_scores_sha256", paths["candidate"]),
                        ("worker_stdout_sha256", paths["stdout"]),
                        ("worker_stderr_sha256", paths["stderr"])):
        if path.is_file():
            final[label] = _sha256_file(path)
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": final["ok"], "status": final["status"], "result": str(paths["result"])}, sort_keys=True))
    return 0 if final["ok"] else 1


def _selftest() -> None:
    """Synthetic controls only: no export, source shard, tokenizer, or model forward."""
    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return list(range(1, len(text) + 1))

    tokenizer = FakeTokenizer()
    rows_calib: list[dict[str, Any]] = []
    rows_heldout: list[dict[str, Any]] = []
    for category in ("code", "technical_general", "prose"):
        for index in range(EXPECTED_PER_CATEGORY_CALIB):
            rows_calib.append({"source_document_id": f"calib-{category}-{index:02d}", "category": category, "text": "c" * 4})
        for index in range(EXPECTED_PER_CATEGORY_HELDOUT):
            rows_heldout.append({"source_document_id": f"heldout-{category}-{index:02d}", "category": category, "text": "h" * 4})
    calls: list[str] = []

    def fake_score(model: Any, tok: Any, row: Mapping[str, Any], *, chunk_size: int) -> Any:
        assert model == "synthetic-model" and tok is tokenizer and chunk_size == CHUNK_SIZE
        calls.append(row["source_document_id"])
        return SimpleNamespace(source_document_id=row["source_document_id"], category=row["category"],
                               tokens=4, bytes=4, bits=8.0)

    with tempfile.TemporaryDirectory(prefix="strat02-quality-selftest-") as directory:
        root = Path(directory)
        calibration, heldout = _score_candidate_splits(
            model="synthetic-model", tokenizer=tokenizer, calib_rows=rows_calib, heldout_rows=rows_heldout,
            score=SimpleNamespace(), calibration_path=root / "calibration.jsonl", candidate_path=root / "candidate.jsonl",
            score_document=fake_score, calibration_expected_tokens=192, calibration_expected_bytes=192,
            heldout_expected_tokens=384, heldout_expected_bytes=384,
        )
        assert calibration["documents"] == 48 and heldout["documents"] == 96
        assert calls == [row["source_document_id"] for row in rows_calib + rows_heldout]
        safe = json.loads((root / "candidate.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert not {"text", "token_ids", "logits"}.intersection(safe)
        try:
            _score_rows_once(model="synthetic-model", tokenizer=tokenizer, rows=rows_heldout, score=SimpleNamespace(),
                             output_path=root / "candidate.jsonl", split="heldout", expected_documents=96,
                             expected_tokens=384, expected_bytes=384, score_document=fake_score)
        except GateError:
            pass
        else:
            raise AssertionError("candidate score output was overwritten")
        heldout_calls = 0

        def fail_calibration(model: Any, tok: Any, row: Mapping[str, Any], *, chunk_size: int) -> Any:
            raise GateError("planted calibration failure")

        def forbidden_heldout(model: Any, tok: Any, row: Mapping[str, Any], *, chunk_size: int) -> Any:
            nonlocal heldout_calls
            heldout_calls += 1
            raise AssertionError("heldout scoring started after calibration failure")

        def fail_calibration_then_forbidden(model: Any, tok: Any, row: Mapping[str, Any], *, chunk_size: int) -> Any:
            if row["source_document_id"].startswith("calib-"):
                raise GateError("planted calibration failure")
            return forbidden_heldout(model, tok, row, chunk_size=chunk_size)

        try:
            _score_candidate_splits(
                model="synthetic-model", tokenizer=tokenizer, calib_rows=rows_calib, heldout_rows=rows_heldout,
                score=SimpleNamespace(), calibration_path=root / "bad-calibration.jsonl",
                candidate_path=root / "should-not-exist.jsonl", score_document=fail_calibration,
            )
        except GateError:
            pass
        else:
            raise AssertionError("planted calibration failure was accepted")
        assert heldout_calls == 0
        try:
            _score_candidate_splits(
                model="synthetic-model", tokenizer=tokenizer, calib_rows=rows_calib, heldout_rows=rows_heldout,
                score=SimpleNamespace(), calibration_path=root / "bad-calibration-2.jsonl",
                candidate_path=root / "should-not-exist-2.jsonl", score_document=fail_calibration_then_forbidden,
            )
        except GateError:
            pass
        else:
            raise AssertionError("the planted no-heldout callback unexpectedly completed")
        assert heldout_calls == 0

    provenance = _source_hashes()
    with tempfile.TemporaryDirectory(prefix="strat02-quality-provenance-") as directory:
        root = Path(directory)
        manifest = root / "supervisor_manifest.json"
        bad = dict(provenance, runner="0" * 64)
        manifest.write_text(json.dumps({"provenance_sha256": bad}), encoding="utf-8")
        try:
            _verify_parent_provenance(manifest, result_path=root / "worker_result.json",
                                      calibration_path=root / "calibration.jsonl", candidate_path=root / "candidate.jsonl")
        except GateError:
            pass
        else:
            raise AssertionError("provenance mismatch was accepted")
    assert _gate_adjudication({"upper_one_sided_ci95_delta_bpb": 0.02}) == "PASS_BPB"
    try:
        _gate_adjudication({"upper_one_sided_ci95_delta_bpb": 0.0200001})
    except GateError:
        raise AssertionError("finite failing BPB gate must classify, not apparatus-fail")
    else:
        assert _gate_adjudication({"upper_one_sided_ci95_delta_bpb": 0.0200001}) == "FAIL_BPB"
    print(json.dumps({"ok": True, "selftest": True, "calibration_rows": 48,
                      "heldout_rows": 96, "bootstrap_draws": BOOTSTRAP_DRAWS}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selftest", action="store_true", help="run fake-only synthetic controls")
    action.add_argument("--preflight", action="store_true", help="run metadata/corpus audit without payload hashes or forwards")
    action.add_argument("--run", action="store_true", help="launch the one bounded real quality worker")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path, default=BOUND_EXPORT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--manifest", type=Path, default=CORPUS / "manifest.json")
    parser.add_argument("--calib", type=Path, default=CORPUS / "calib.jsonl")
    parser.add_argument("--heldout", type=Path, default=CORPUS / "heldout.jsonl")
    parser.add_argument("--teacher-scores", type=Path, default=TEACHER_SCORES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--calibration", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--candidate", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.preflight:
            _, report, metadata = _metadata_preflight(args)
            print(json.dumps({"ok": True, "preflight": metadata, "snapshot": str(report.snapshot)}, sort_keys=True))
            return 0
        if args.worker:
            required = (args.output_dir, args.calibration, args.candidate, args.result, args.supervisor_manifest)
            if any(value is None for value in required):
                raise GateError("internal worker requires output, calibration, candidate, result and supervisor manifest")
            inherited = os.environ.get(WORKER_TOKEN_ENV, "")
            if not args.worker_token or not secrets.compare_digest(args.worker_token, inherited):
                raise GateError("internal worker may run only when launched by its --run parent")
            return _worker(args)
        if args.output_dir is None:
            raise GateError("--run requires an explicit --output-dir; refusing implicit writes")
        return _run_parent(args)
    except (GateError, OSError, RuntimeError, ValueError, TypeError, AssertionError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
