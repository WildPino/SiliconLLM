#!/usr/bin/env python3
"""Write-once, bounded full-heldout STRAT-02 F32 teacher BPB baseline.

This is an observation of the pinned teacher only.  It has no candidate arm,
quality gate, throughput target, automatic resume, donor download, GPU/T4
path, or mutable corpus selection.  A real run is intentionally explicit::

    python benchmarks/donor_adaptation/density/strat02_teacher_baseline.py \
        --run --output-dir D:\\runs\\strat02-teacher-baseline

The parent starts one direct base-Python worker and monitors that actual PID,
not a Windows venv redirector.  The worker hashes all eleven shards once,
audits the frozen calib+heldout corpus and tokenizer before its first forward,
loads the mmap teacher once, then scores all 96 heldout JSONL rows in order.
Use ``--selftest`` for tiny fake corpus/model and monitor controls; it never
opens real weights, downloads anything, or forwards the donor.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
CORPUS = HERE / "corpus" / "strat02_document_holdout_v1"
EXPECTED_MANIFEST_SHA256 = "56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749"
EXPECTED_HELDOUT_SHA256 = "450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e"
EXPECTED_HELDOUT_TOKEN_IDS_SHA256 = "5a4cf31ec7db0f3bc541f2e75a78b133a50cd3d6174abb26ef81e08c45e65289"
EXPECTED_DOCUMENTS = 96
EXPECTED_PER_CATEGORY = 32
MIN_LAUNCH_RAM_BYTES = 55 * 1024 ** 3
MIN_LAUNCH_DISK_BYTES = 65 * 1024 ** 3
MAX_WALL_SECONDS = 6 * 60 * 60
MONITOR_INTERVAL_SECONDS = 5.0
WORKER_TOKEN_ENV = "STRAT02_TEACHER_BASELINE_PARENT_TOKEN"


class GateError(RuntimeError):
    """A fixed provenance, safety, or write-once requirement failed."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _load_local_module(filename: str, module_name: str) -> Any:
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


def _sha256_file(path: Path, bounded: Any | None = None) -> str:
    if bounded is not None:
        return bounded._sha256_file(path)
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
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


def _prepare_output(output_dir: Path) -> dict[str, Path]:
    """Require a new/empty output directory and reserve only this run's files."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GateError(f"output directory must be new or empty for write-once results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "supervisor_manifest.json",
        "log": output_dir / "supervisor_log.jsonl",
        "scores": output_dir / "teacher_scores.jsonl",
        "worker": output_dir / "worker_result.json",
        "result": output_dir / "supervisor_result.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise GateError("refusing to overwrite write-once output: " + "; ".join(existing))
    try:
        with paths["log"].open("x", encoding="utf-8", newline="\n"):
            pass
    except FileExistsError as exc:  # Concurrent creator after the check.
        raise GateError(f"write-once log already exists: {paths['log']}") from exc
    return paths


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    _append_jsonl(path, {"utc_timestamp": _utc_now(), **event})


def _launch_preflight(snapshot: Path | None) -> tuple[Any, Any, dict[str, Any]]:
    """Read-only parent gates; the expensive corpus/token audit remains in worker."""
    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_teacher_baseline_bounded")
    teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_teacher_baseline_teacher")
    versions = bounded._pinned_environment()
    report = teacher.preflight(snapshot=snapshot)  # manifest, index, pinned code, shard presence only.
    status = report.source_status
    if not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO":
        raise GateError("local teacher metadata does not match the pinned offline runtime")
    if not status.get("shards_present"):
        raise GateError("all eleven local teacher shards must be present; no download is attempted")
    psutil = bounded._psutil()
    available_ram = int(psutil.virtual_memory().available)
    snapshot_free = int(psutil.disk_usage(str(report.snapshot)).free)
    if available_ram < MIN_LAUNCH_RAM_BYTES:
        raise GateError(f"launch requires >=55 GiB available physical RAM; observed {available_ram} bytes")
    if snapshot_free < MIN_LAUNCH_DISK_BYTES:
        raise GateError(
            f"launch requires >=65 GiB free on the pinned cache/snapshot volume; observed {snapshot_free} bytes"
        )
    return bounded, report, {
        "utc_timestamp": _utc_now(),
        "runtime_versions": versions,
        "python": sys.version.split()[0],
        "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
        "offline_policy": {
            "hub_download_api": "not used",
            "model_loader": "local mmap loader only",
            "tokenizer_local_files_only": True,
        },
        "snapshot": str(report.snapshot),
        "metadata": dict(status),
        "available_physical_ram_bytes": available_ram,
        "snapshot_volume_free_bytes": snapshot_free,
        "worker_before_forward": "full calib+heldout corpus/tokenizer audit, then exactly one eleven-shard verification",
    }


def _audit_entire_frozen_corpus(audit: Any, *, manifest_path: Path, calib_path: Path,
                                heldout_path: Path, snapshot: Path, bounded: Any) -> tuple[dict[str, Any], list[dict]]:
    """Run the trusted full corpus and token-ID audit, with explicit public pins."""
    if _sha256_file(manifest_path, bounded) != EXPECTED_MANIFEST_SHA256:
        raise GateError("manifest SHA-256 differs from the preregistered pin")
    if _sha256_file(heldout_path, bounded) != EXPECTED_HELDOUT_SHA256:
        raise GateError("heldout JSONL SHA-256 differs from the preregistered pin")
    errors: list[str] = []
    manifest, rows_by_split = audit.check_corpus(
        manifest_path, {"calib": calib_path, "heldout": heldout_path}, errors
    )
    token_report = audit.check_tokens(snapshot, rows_by_split, errors)
    heldout_token_digest = token_report.get("splits", {}).get("heldout", {}).get("ordered_token_ids_sha256")
    if heldout_token_digest != EXPECTED_HELDOUT_TOKEN_IDS_SHA256:
        errors.append("heldout: ordered token-ID SHA-256 differs from the explicit teacher-baseline pin")
    if errors:
        raise GateError("frozen corpus/tokenizer audit failed: " + "; ".join(errors))
    heldout = rows_by_split.get("heldout")
    if not isinstance(heldout, list):
        raise GateError("corpus audit returned no heldout rows")
    return manifest, heldout


def _validate_heldout_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Make the no-subset/no-reorder requirement explicit after the full audit."""
    if len(rows) != EXPECTED_DOCUMENTS:
        raise GateError(f"heldout must contain exactly {EXPECTED_DOCUMENTS} documents; got {len(rows)}")
    seen: set[str] = set()
    counts = {"code": 0, "technical_general": 0, "prose": 0}
    for number, row in enumerate(rows, 1):
        identifier, category, text = row.get("source_document_id"), row.get("category"), row.get("text")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise GateError(f"heldout:{number}: invalid or duplicate source_document_id")
        if category not in counts or not isinstance(text, str) or not text.encode("utf-8"):
            raise GateError(f"heldout:{number}: invalid category or empty UTF-8 text")
        seen.add(identifier)
        counts[category] += 1
    if any(count != EXPECTED_PER_CATEGORY for count in counts.values()):
        raise GateError(f"heldout category counts must be exactly {EXPECTED_PER_CATEGORY}: {counts}")


def _score_all_documents(
    *, model: Any, tokenizer: Any, heldout_rows: Sequence[Mapping[str, Any]], score: Any,
    scores_path: Path, expected_token_total: int, expected_byte_total: int,
    score_document: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Score every row once, preserving input order and flushing each safe record."""
    _validate_heldout_rows(heldout_rows)
    scorer = score_document or score.score_document
    categories: dict[str, dict[str, Any]] = {
        category: {"documents": 0, "tokens": 0, "bytes": 0, "bits": 0.0}
        for category in ("code", "technical_general", "prose")
    }
    total_tokens = total_bytes = 0
    total_bits = 0.0
    try:
        handle = scores_path.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise GateError(f"write-once score output already exists: {scores_path}") from exc
    with handle:
        for number, row in enumerate(heldout_rows, 1):
            started = time.monotonic()
            document_score = scorer(model, tokenizer, row, chunk_size=128)
            wall_seconds = time.monotonic() - started
            identifier = row["source_document_id"]
            category = row["category"]
            if (document_score.source_document_id, document_score.category) != (identifier, category):
                raise GateError(f"heldout:{number}: scorer identity/category does not bind to input row")
            tokens, byte_count, bits = document_score.tokens, document_score.bytes, float(document_score.bits)
            if (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0 or
                    isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0 or
                    not math.isfinite(bits) or bits < 0 or not math.isfinite(wall_seconds) or wall_seconds < 0):
                raise GateError(f"heldout:{number}: non-finite or invalid score observation")
            # Never serialize raw text, payload IDs, logits, or a throughput target.
            record = {
                "source_document_id": identifier,
                "category": category,
                "tokens": tokens,
                "bytes": byte_count,
                "bits": bits,
                "wall_seconds": wall_seconds,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            category_total = categories[category]
            category_total["documents"] += 1
            category_total["tokens"] += tokens
            category_total["bytes"] += byte_count
            category_total["bits"] += bits
            total_tokens += tokens
            total_bytes += byte_count
            total_bits += bits
    if total_tokens != expected_token_total or total_bytes != expected_byte_total:
        raise GateError(
            "complete heldout totals differ from frozen audit: "
            f"tokens {total_tokens}/{expected_token_total}, bytes {total_bytes}/{expected_byte_total}"
        )
    if sum(summary["documents"] for summary in categories.values()) != EXPECTED_DOCUMENTS:
        raise GateError("complete heldout score count differs from 96")
    for category, summary in categories.items():
        if summary["documents"] != EXPECTED_PER_CATEGORY:
            raise GateError(f"complete heldout category count differs for {category}")
        summary["bpb"] = summary["bits"] / summary["bytes"]
    return {
        "documents": EXPECTED_DOCUMENTS,
        "tokens": total_tokens,
        "bytes": total_bytes,
        "bits": total_bits,
        "bpb": total_bits / total_bytes,
        "categories": categories,
        "scoring": {"chunk_size": 128, "order": "original heldout.jsonl order", "truncation": False},
    }


def _count_partial_scores(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _validate_complete_artifact(scores_path: Path, aggregate: Mapping[str, Any], score: Any) -> None:
    """Independently reconcile the persisted rows before parent declares COMPLETE."""
    persisted = score.read_score_jsonl(scores_path)
    if len(persisted) != EXPECTED_DOCUMENTS:
        raise GateError(f"complete teacher score artifact has {len(persisted)} rows, not 96")
    tokens = sum(item.tokens for item in persisted.values())
    byte_count = sum(item.bytes for item in persisted.values())
    bits = sum(item.bits for item in persisted.values())
    if tokens != aggregate.get("tokens") or byte_count != aggregate.get("bytes"):
        raise GateError("persisted teacher token/byte totals differ from worker aggregate")
    if not isinstance(aggregate.get("bits"), (int, float)) or not math.isclose(
        bits, float(aggregate["bits"]), rel_tol=0.0, abs_tol=1e-7
    ):
        raise GateError("persisted teacher bits differ from worker aggregate")


def _worker(args: argparse.Namespace) -> int:
    """The sole location of a real teacher forward, reachable only from the parent."""
    result_path, scores_path = args.result.resolve(), args.scores.resolve()
    try:
        bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_teacher_worker_bounded")
        bounded._pinned_environment()
        teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_teacher_worker_loader")
        audit = _load_local_module("strat02_token_audit.py", "_strat02_teacher_worker_audit")
        score = _load_local_module("strat02_score.py", "_strat02_teacher_worker_score")
        report = teacher.preflight(snapshot=args.snapshot)
        manifest, heldout_rows = _audit_entire_frozen_corpus(
            audit, manifest_path=args.manifest, calib_path=args.calib, heldout_path=args.heldout,
            snapshot=report.snapshot, bounded=bounded,
        )
        tokenizer, _, _, _ = audit.load_tokenizer(report.snapshot)
        score.validate_strat02_tokenizer(tokenizer)
        heldout_part = manifest.get("parts", {}).get("heldout", {})
        expected_bytes = heldout_part.get("total_span_bytes")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise GateError("frozen manifest lacks heldout total_span_bytes")
        verified = teacher.verify_shards(report)  # Exactly once after cheap corpus gates; no download.
        with teacher.load_reference_model(verified, pointer_check="all") as model:
            model.eval()
            import torch
            with torch.inference_mode():
                aggregate = _score_all_documents(
                    model=model, tokenizer=tokenizer, heldout_rows=heldout_rows, score=score,
                    scores_path=scores_path, expected_token_total=audit.EXPECTED_TOTALS["heldout"],
                    expected_byte_total=expected_bytes,
                )
        persisted = score.read_score_jsonl(scores_path)
        score.validate_scores_against_heldout(persisted, heldout_rows, tokenizer, arm="teacher")
        _validate_complete_artifact(scores_path, aggregate, score)
        _write_json_once(result_path, {
            "ok": True,
            "status": "COMPLETE",
            "utc_timestamp": _utc_now(),
            "aggregate": aggregate,
            "notes": [
                "Pinned F32 teacher baseline only; no candidate comparison or quality-gate conclusion.",
                "Per-document wall_seconds are diagnostics, not a throughput target or tok/s claim.",
                "No raw document text is included in outputs.",
            ],
        })
        return 0
    except Exception as exc:
        try:
            _write_json_once(result_path, {
                "ok": False,
                "status": "INCOMPLETE",
                "utc_timestamp": _utc_now(),
                "scored_documents_persisted": _count_partial_scores(scores_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        except (GateError, OSError):
            pass
        return 1


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _failure_classification(outcome: Any, worker_result: Mapping[str, Any] | None) -> str | None:
    if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
        return outcome.status
    if outcome.exit_code != 0:
        return "WORKER_ERROR"
    if not worker_result:
        return "WORKER_PROTOCOL_ERROR"
    if worker_result.get("ok") is True and worker_result.get("status") == "COMPLETE":
        return None
    return "WORKER_ERROR"


def _run_parent(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    try:
        bounded, report, preflight = _launch_preflight(args.snapshot)
    except Exception as exc:
        _write_json_once(paths["manifest"], {"schema": "strat02_teacher_baseline_v1", "preflight": "failed"})
        _append_log(paths["log"], {"event": "parent_preflight_failed", "error_type": type(exc).__name__, "error": str(exc)})
        final = {"ok": False, "status": "INCOMPLETE", "failure_classification": "PREFLIGHT_ERROR",
                 "utc_timestamp": _utc_now(), "error_type": type(exc).__name__, "error": str(exc)}
        _write_json_once(paths["result"], final)
        print(json.dumps({"ok": False, "status": "INCOMPLETE", "result": str(paths["result"])}, sort_keys=True))
        return 1
    manifest = {
        "schema": "strat02_teacher_baseline_v1",
        "purpose": "full 96-document pinned F32 teacher BPB baseline; no quality/rate conclusion",
        "pins": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "heldout_jsonl_sha256": EXPECTED_HELDOUT_SHA256,
            "heldout_ordered_token_ids_sha256": EXPECTED_HELDOUT_TOKEN_IDS_SHA256,
        },
        "limits": {
            "launch_available_physical_ram_bytes_min": MIN_LAUNCH_RAM_BYTES,
            "launch_cache_volume_free_bytes_min": MIN_LAUNCH_DISK_BYTES,
            "runtime_available_physical_ram_bytes_min": bounded.MIN_RUNTIME_RAM_BYTES,
            "worker_working_set_bytes_max": bounded.MAX_CHILD_MEMORY_BYTES,
            "worker_private_commit_bytes_max": bounded.MAX_CHILD_MEMORY_BYTES,
            "wall_seconds_max": MAX_WALL_SECONDS,
            "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
        },
        "parent_preflight": preflight,
        "output": {"scores": paths["scores"].name, "raw_text_reported": False, "auto_resume": False},
    }
    _write_json_once(paths["manifest"], manifest)
    _append_log(paths["log"], {"event": "parent_preflight_passed"})
    child_env = os.environ.copy()
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["TRANSFORMERS_OFFLINE"] = "1"
    worker_python = bounded._direct_worker_python(child_env)
    token = secrets.token_urlsafe(32)
    child_env[WORKER_TOKEN_ENV] = token
    command = [
        worker_python, str(Path(__file__).resolve()), "--worker", "--snapshot", str(report.snapshot),
        "--manifest", str(args.manifest.resolve()), "--calib", str(args.calib.resolve()),
        "--heldout", str(args.heldout.resolve()), "--scores", str(paths["scores"]),
        "--result", str(paths["worker"]), "--worker-token", token,
    ]
    _append_log(paths["log"], {
        "event": "child_launch", "python": worker_python,
        "termination_scope": "direct_worker_pid_only", "monitor_target": "direct_base_python_worker",
    })
    process: Any | None = None
    try:
        process = subprocess.Popen(command, cwd=str(HERE), env=child_env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outcome = bounded._monitor_or_terminate(
            process, psutil=bounded._psutil(), expected_executable=Path(worker_python),
            wall_limit_seconds=MAX_WALL_SECONDS, interval_seconds=MONITOR_INTERVAL_SECONDS,
            on_sample=lambda sample: _append_log(paths["log"], {"event": "resource_sample", **sample}),
        )
        _append_log(paths["log"], {"event": "monitor_finished", **asdict(outcome)})
        worker_result = _load_json_object(paths["worker"])
        classification = _failure_classification(outcome, worker_result)
        if classification is None:
            try:
                score = _load_local_module("strat02_score.py", "_strat02_teacher_parent_score")
                _validate_complete_artifact(paths["scores"], worker_result["aggregate"], score)
            except Exception as exc:
                classification = "WORKER_PROTOCOL_ERROR"
                _append_log(paths["log"], {
                    "event": "parent_artifact_validation_failed",
                    "error_type": type(exc).__name__, "error": str(exc),
                })
        complete = classification is None
        final = {
            "ok": complete,
            "status": "COMPLETE" if complete else "INCOMPLETE",
            "failure_classification": classification,
            "utc_timestamp": _utc_now(),
            "child_pid": process.pid,
            "monitor": asdict(outcome),
            "worker_result": worker_result,
            "partial_scores_persisted": _count_partial_scores(paths["scores"]),
        }
    except Exception as exc:
        if process is not None:
            bounded._terminate_only_child(process)  # Never reaches beyond the worker this parent created.
        _append_log(paths["log"], {"event": "parent_supervision_error", "error_type": type(exc).__name__, "error": str(exc)})
        complete = False
        final = {
            "ok": False,
            "status": "INCOMPLETE",
            "failure_classification": "PARENT_SUPERVISION_ERROR",
            "utc_timestamp": _utc_now(),
            "child_pid": process.pid if process is not None else None,
            "partial_scores_persisted": _count_partial_scores(paths["scores"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    final["notes"] = [
        "Teacher baseline only; it makes no quality-gate, candidate-comparison, or throughput claim.",
        "On every error/interruption the run status is INCOMPLETE; partial score JSONL is retained without resume.",
        "Only the created direct worker PID may have been terminated by supervision.",
    ]
    if paths["scores"].is_file():
        final["scores_sha256"] = _sha256_file(paths["scores"])
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": complete, "status": final["status"], "result": str(paths["result"])}, sort_keys=True))
    return 0 if complete else 1


def _selftest() -> None:
    """Fake corpus/model and direct-worker monitor controls; never opens donor weights."""
    score = _load_local_module("strat02_score.py", "_strat02_teacher_selftest_score")
    rows: list[dict[str, Any]] = []
    for category in ("code", "technical_general", "prose"):
        for index in range(EXPECTED_PER_CATEGORY):
            text = "x" * 5000 if (category, index) == ("code", 0) else f"{category}:{index}:é"
            rows.append({"source_document_id": f"{category}-{index:02d}", "category": category, "text": text})
    seen: list[tuple[str, str]] = []

    class TinyFakeModel:
        pass

    def fake_score(model: Any, tokenizer: Any, row: Mapping[str, Any], *, chunk_size: int) -> Any:
        assert isinstance(model, TinyFakeModel) and tokenizer == "tiny-tokenizer" and chunk_size == 128
        seen.append((row["source_document_id"], row["text"]))
        raw_bytes = len(row["text"].encode("utf-8"))
        return SimpleNamespace(source_document_id=row["source_document_id"], category=row["category"],
                               tokens=len(row["text"]), bytes=raw_bytes, bits=float(raw_bytes + 1))

    expected_tokens = sum(len(row["text"]) for row in rows)
    expected_bytes = sum(len(row["text"].encode("utf-8")) for row in rows)
    with tempfile.TemporaryDirectory(prefix="strat02-teacher-baseline-selftest-") as temporary:
        scores_path = Path(temporary) / "scores.jsonl"
        aggregate = _score_all_documents(
            model=TinyFakeModel(), tokenizer="tiny-tokenizer", heldout_rows=rows, score=score,
            scores_path=scores_path, expected_token_total=expected_tokens, expected_byte_total=expected_bytes,
            score_document=fake_score,
        )
        persisted = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
        assert [item["source_document_id"] for item in persisted] == [row["source_document_id"] for row in rows]
        assert seen == [(row["source_document_id"], row["text"]) for row in rows]
        assert len(persisted) == aggregate["documents"] == EXPECTED_DOCUMENTS
        assert persisted[0]["bytes"] == 5000 and persisted[0]["tokens"] == 5000  # planted no-truncation sentinel.
        assert all("text" not in item and "wall_seconds" in item for item in persisted)
        _validate_complete_artifact(scores_path, aggregate, score)
        tampered_aggregate = dict(aggregate, bits=aggregate["bits"] + 1.0)
        try:
            _validate_complete_artifact(scores_path, tampered_aggregate, score)
        except GateError:
            pass
        else:
            raise AssertionError("tampered teacher aggregate was accepted")
        try:
            _score_all_documents(model=TinyFakeModel(), tokenizer="tiny-tokenizer", heldout_rows=rows, score=score,
                                 scores_path=scores_path, expected_token_total=expected_tokens,
                                 expected_byte_total=expected_bytes, score_document=fake_score)
        except GateError:
            pass
        else:  # pragma: no cover - explicit write-once control.
            raise AssertionError("score output was overwritten")

    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_teacher_selftest_bounded")
    failed = bounded.MonitorOutcome("CHILD_EXITED", 1, None, 0.0, 0)
    resource = bounded.MonitorOutcome("VOID_RESOURCE", -15, "planted", 0.0, 1)
    complete_worker = {"ok": True, "status": "COMPLETE"}
    assert _failure_classification(failed, {"ok": False, "status": "INCOMPLETE"}) == "WORKER_ERROR"
    assert _failure_classification(resource, complete_worker) == "VOID_RESOURCE"
    assert _failure_classification(bounded.MonitorOutcome("CHILD_EXITED", 0, None, 0.0, 1), complete_worker) is None

    # Exercise psutil sampling against the direct base interpreter, never a venv redirector or donor process.
    import psutil
    child_env = os.environ.copy()
    direct_python = bounded._direct_worker_python(child_env)
    tiny = subprocess.Popen([direct_python, "-c", "import time; time.sleep(0.35)"], env=child_env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        samples: list[Mapping[str, Any]] = []
        outcome = bounded._monitor_or_terminate(
            tiny, psutil=psutil, expected_executable=Path(direct_python), on_sample=samples.append,
            interval_seconds=1.0, wall_limit_seconds=10.0,
        )
        assert outcome.status == "CHILD_EXITED" and outcome.exit_code == 0
        assert samples and samples[0]["child_private_commit_bytes"] > 0
    finally:
        bounded._terminate_only_child(tiny)
    print(json.dumps({"ok": True, "selftest": True}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="launch the one bounded real teacher worker")
    action.add_argument("--selftest", action="store_true", help="run fake-only safety and protocol controls")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help="explicit new/empty write-once output directory")
    parser.add_argument("--snapshot", type=Path, help="explicit pinned local snapshot; never a Hub identifier")
    parser.add_argument("--manifest", type=Path, default=CORPUS / "manifest.json")
    parser.add_argument("--calib", type=Path, default=CORPUS / "calib.jsonl")
    parser.add_argument("--heldout", type=Path, default=CORPUS / "heldout.jsonl")
    parser.add_argument("--scores", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.worker:
            if args.result is None or args.scores is None:
                raise GateError("internal worker requires --result and --scores")
            inherited = os.environ.get(WORKER_TOKEN_ENV, "")
            if not args.worker_token or not secrets.compare_digest(args.worker_token, inherited):
                raise GateError("internal worker may run only when launched by its --run parent")
            return _worker(args)
        if args.output_dir is None:
            raise GateError("--run requires an explicit --output-dir; refusing implicit writes")
        return _run_parent(args)
    except (GateError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
