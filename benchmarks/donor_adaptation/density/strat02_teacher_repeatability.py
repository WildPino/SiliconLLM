#!/usr/bin/env python3
"""Bounded STRAT-02 F32 teacher repeatability apparatus.

This runner is deliberately narrower than the heldout teacher baseline: it
opens only the preregistered first calibration row, scores that row twice, and
compares two first-64-token logit blocks.  The parent owns one direct worker
PID and reuses the pinned mmap loader, scorer, and resource monitor.  No Hub
API, heldout file, raw text, token IDs, or logit tensor is written.
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
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CORPUS = HERE / "corpus" / "strat02_document_holdout_v1"
BRIEF = PROJECT / "docs" / "research" / "donor_adaptation" / "briefs" / "BRIEF_STRAT_02_TEACHER_REPEATABILITY.md"

REPOSITORY = "allenai/StdMoE_1b14b_1T_Preanneal"
REVISION = "d2a4949c9d4ad6cf47fbac131f7e020077332b21"
EXPECTED_MANIFEST_SHA256 = "56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749"
EXPECTED_CALIB_SHA256 = "f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f"
EXPECTED_CALIB_TEXT_SHA256 = "dd1ca41399ccc61f42f1a8984c35c704d6c7449025a1f1e21337d70e42a63b4a"
EXPECTED_CALIB_ID = "file:data/external/the_stack_python/cpython/Lib/test/test_sqlite3/test_userfunctions.py"
EXPECTED_CALIB_CATEGORY = "code"
EXPECTED_CALIB_BYTES = 8192
EOS_ID = 100257
LOGIT_SHAPE = (1, 64, 100352)
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-6
BPB_ABS_TOL = 1e-7
CHUNK_SIZE = 128
MIN_LAUNCH_RAM_BYTES = 55 * 1024**3
MIN_OUTPUT_DISK_BYTES = 1 * 1024**3
MIN_RUNTIME_RAM_BYTES = 8 * 1024**3
MAX_WORKER_MEMORY_BYTES = 70 * 1024**3
MAX_WALL_SECONDS = 45 * 60
MONITOR_INTERVAL_SECONDS = 5.0
WORKER_TOKEN_ENV = "STRAT02_TEACHER_REPEATABILITY_PARENT_TOKEN"
SOURCE_FILES = (
    "benchmarks/donor_adaptation/density/strat02_teacher_baseline.py",
    "benchmarks/donor_adaptation/density/strat02_bounded_smoke.py",
    "benchmarks/donor_adaptation/density/strat02_mmap_teacher.py",
    "benchmarks/donor_adaptation/density/strat02_score.py",
    "benchmarks/donor_adaptation/density/strat02_token_audit.py",
)


class GateError(RuntimeError):
    """A fixed provenance, protocol, or resource apparatus requirement failed."""


class RepeatabilityError(GateError):
    """The valid apparatus produced a numerical repeatability mismatch."""


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


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    _append_jsonl(path, {"utc_timestamp": _utc_now(), **event})


def _prepare_output(output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GateError(f"output directory must be new or empty for write-once results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "supervisor_manifest.json",
        "log": output_dir / "supervisor_log.jsonl",
        "scores": output_dir / "teacher_repeatability_scores.jsonl",
        "worker": output_dir / "worker_result.json",
        "result": output_dir / "supervisor_result.json",
        "stdout": output_dir / "worker_stdout.log",
        "stderr": output_dir / "worker_stderr.log",
    }
    if any(path.exists() for path in paths.values()):
        raise GateError("refusing to overwrite write-once output")
    try:
        with paths["log"].open("x", encoding="utf-8", newline="\n"):
            pass
    except FileExistsError as exc:
        raise GateError(f"write-once log already exists: {paths['log']}") from exc
    return paths


def _source_hashes() -> dict[str, str]:
    paths = {"brief": BRIEF, "runner": Path(__file__).resolve()}
    paths.update({name: PROJECT / name for name in SOURCE_FILES})
    observed: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise GateError(f"provenance source is missing: {path}")
        observed[label] = _sha256_file(path)
    return observed


def _existing_volume_path(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _calibration_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_document_id": row["source_document_id"],
        "category": row["category"],
        "span_byte_count": row["span_byte_count"],
        "text_sha256": EXPECTED_CALIB_TEXT_SHA256,
        "selection": "calib.jsonl row zero in original order; fixed before execution",
        "raw_text_reported": False,
    }


def _verify_parent_provenance(supervisor_manifest: Path, *, result_path: Path, scores_path: Path) -> dict[str, str]:
    """Verify the parent's exact source hash map before importing worker modules."""
    manifest_path = supervisor_manifest.expanduser().resolve()
    result_path = result_path.resolve()
    scores_path = scores_path.resolve()
    if manifest_path.name != "supervisor_manifest.json":
        raise GateError("internal supervisor manifest must be supervisor_manifest.json")
    if manifest_path.parent != result_path.parent or manifest_path.parent != scores_path.parent:
        raise GateError("supervisor manifest, worker result, and scores must share one output directory")
    manifest = _load_json_object(manifest_path)
    expected = manifest.get("provenance_sha256") if manifest is not None else None
    if not isinstance(expected, dict) or not expected or any(not isinstance(k, str) or not isinstance(v, str)
                                                              for k, v in expected.items()):
        raise GateError("supervisor manifest has no valid parent provenance_sha256 map")
    observed = _source_hashes()
    if observed != expected:
        raise GateError("worker source provenance does not exactly match supervisor_manifest.json")
    return observed


def _read_calibration_row(calib: Path, manifest: Path, bounded: Any, audit: Any) -> dict[str, Any]:
    row = bounded._read_row_zero(calib, manifest, audit)
    text = row.get("text")
    if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != EXPECTED_CALIB_TEXT_SHA256:
        raise GateError("calibration row zero text SHA-256 differs from the preregistered pin")
    if row.get("source_document_id") != EXPECTED_CALIB_ID or row.get("category") != EXPECTED_CALIB_CATEGORY:
        raise GateError("calibration row zero identity differs from the preregistered pin")
    if row.get("span_byte_count") != EXPECTED_CALIB_BYTES:
        raise GateError("calibration row zero byte count differs from the preregistered pin")
    return row


def _metadata_preflight(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Perform read-only environment, source, corpus-row, and tokenizer gates."""
    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_repeatability_bounded")
    teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_repeatability_teacher")
    audit = _load_local_module("strat02_token_audit.py", "_strat02_repeatability_audit")
    score = _load_local_module("strat02_score.py", "_strat02_repeatability_score")
    versions = bounded._pinned_environment()
    report = teacher.preflight(snapshot=args.snapshot)
    status = report.source_status
    if status.get("repository") != REPOSITORY or status.get("revision") != REVISION:
        raise GateError("local teacher metadata does not match the pinned donor revision")
    if not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO":
        raise GateError("local teacher metadata does not match the pinned offline runtime")
    if not status.get("shards_present"):
        raise GateError("all eleven local teacher shards must be present; no download is attempted")
    psutil = bounded._psutil()
    available_ram = int(psutil.virtual_memory().available)
    volume = _existing_volume_path(args.output_dir or HERE)
    output_free = int(psutil.disk_usage(str(volume)).free)
    if available_ram < MIN_LAUNCH_RAM_BYTES:
        raise GateError(f"preflight requires >=55 GiB available physical RAM; observed {available_ram} bytes")
    if output_free < MIN_OUTPUT_DISK_BYTES:
        raise GateError(f"preflight requires >=1 GiB free output space; observed {output_free} bytes")
    row = _read_calibration_row(args.calib, args.manifest, bounded, audit)
    tokenizer = bounded._verify_exact_tokenizer(audit, report.snapshot)
    score.validate_strat02_tokenizer(tokenizer)
    metadata = {
        "utc_timestamp": _utc_now(),
        "parent_pid": os.getpid(),
        "runtime_versions": versions,
        "python": sys.version.split()[0],
        "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
        "offline_policy": {"hub_download_api": "not used", "model_loader": "local mmap loader only",
                           "tokenizer_local_files_only": True},
        "snapshot": str(report.snapshot),
        "metadata": dict(status),
        "available_physical_ram_bytes": available_ram,
        "output_volume": str(volume),
        "output_volume_free_bytes": output_free,
        "calibration": _calibration_metadata(row),
        "heldout_opened_for_scoring": False,
        "weight_shard_hashes": "not performed by metadata preflight",
    }
    return bounded, report, metadata, {"row": row, "tokenizer": tokenizer, "score": score}


def _score_pair(*, model: Any, tokenizer: Any, row: Mapping[str, Any], score: Any,
                scores_path: Path, score_document: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Call the pinned scorer exactly twice, persisting only safe observations."""
    scorer = score_document or score.score_document
    observations: list[dict[str, Any]] = []
    try:
        handle = scores_path.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise GateError(f"write-once score output already exists: {scores_path}") from exc
    with handle:
        for repeat in (1, 2):
            observation = scorer(model, tokenizer, row, chunk_size=CHUNK_SIZE)
            bits = float(observation.bits)
            tokens, byte_count = observation.tokens, observation.bytes
            if (not math.isfinite(bits) or bits < 0 or isinstance(tokens, bool) or not isinstance(tokens, int)
                    or tokens <= 0 or isinstance(byte_count, bool) or not isinstance(byte_count, int)
                    or byte_count != EXPECTED_CALIB_BYTES):
                raise GateError("non-finite or malformed calibration score observation")
            if observation.source_document_id != EXPECTED_CALIB_ID or observation.category != EXPECTED_CALIB_CATEGORY:
                raise GateError("scorer identity/category is not bound to the fixed calibration row")
            bpb = bits / byte_count
            if not math.isfinite(bpb):
                raise GateError("non-finite calibration BPB observation")
            record = {"repeat": repeat, "bits": bits, "bpb": bpb, "tokens": tokens, "bytes": byte_count}
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            observations.append(record)
    if observations[0]["tokens"] != observations[1]["tokens"] or observations[0]["bytes"] != observations[1]["bytes"]:
        raise GateError("calibration token or byte count changed between repeats")
    delta = abs(observations[1]["bpb"] - observations[0]["bpb"])
    if not math.isfinite(delta):
        raise GateError("non-finite BPB repeat difference")
    if delta > BPB_ABS_TOL:
        raise RepeatabilityError(f"calibration BPB difference {delta!r} exceeds frozen tolerance {BPB_ABS_TOL!r}")
    return {"repeats": observations, "bpb_abs_difference": delta, "bpb_abs_tolerance": BPB_ABS_TOL,
            "score_calls": 2, "chunk_size": CHUNK_SIZE, "truncation": False}


def _compare_logit_blocks(first: Any, second: Any, *, expected_shape: tuple[int, ...] = LOGIT_SHAPE,
                          atol: float = LOGIT_ATOL, rtol: float = LOGIT_RTOL) -> dict[str, Any]:
    if tuple(first.shape) != expected_shape or tuple(second.shape) != expected_shape:
        raise GateError(f"first64 logits shape mismatch: {tuple(first.shape)} / {tuple(second.shape)}")
    torch = _load_local_module("strat02_score.py", "_strat02_repeatability_compare_score")._torch()
    first_f32, second_f32 = first.to(dtype=torch.float32), second.to(dtype=torch.float32)
    if not bool(torch.isfinite(first_f32).all().item()) or not bool(torch.isfinite(second_f32).all().item()):
        raise GateError("non-finite first64 logits")
    difference = (second_f32 - first_f32).abs()
    finite_max = float(difference.max().item())
    if not math.isfinite(finite_max):
        raise GateError("non-finite first64 maximum logit error")
    outside = ~torch.isclose(second_f32, first_f32, atol=atol, rtol=rtol)
    mismatches = int(outside.sum().item())
    return {"shape": list(expected_shape), "all_finite": True, "max_abs_error": finite_max,
            "elements_outside_tolerance": mismatches, "atol": atol, "rtol": rtol,
            "forward_calls": 2}


def _first64_logits(model: Any, payload_ids: Sequence[int], score: Any) -> dict[str, Any]:
    if len(payload_ids) < 64:
        raise GateError("calibration payload has fewer than 64 tokens")
    torch = score._torch()
    device = score._model_device(model, torch)
    inputs = torch.tensor([[EOS_ID, *list(payload_ids[:63])]], dtype=torch.long, device=device)
    if tuple(inputs.shape) != (1, 64):
        raise GateError("first64 input shape mismatch")
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model.model(input_ids=inputs, use_cache=False, return_dict=True)
        hidden = score._last_hidden_state(output)
        if tuple(hidden.shape[:2]) != (1, 64):
            raise GateError("first64 backbone hidden shape mismatch")
        logits = model.lm_head(hidden)
        if tuple(logits.shape) != LOGIT_SHAPE:
            raise GateError(f"first64 logits shape mismatch: {tuple(logits.shape)}")
        result = logits.to(dtype=torch.float32).detach().cpu()
        del logits, hidden, output
    return {"tensor": result, "input_shape": [1, 64]}


def _logit_pair(*, model: Any, payload_ids: Sequence[int], score: Any) -> dict[str, Any]:
    first = _first64_logits(model, payload_ids, score)["tensor"]
    second = _first64_logits(model, payload_ids, score)["tensor"]
    report = _compare_logit_blocks(first, second)
    del first, second
    return report


def _worker(args: argparse.Namespace) -> int:
    result_path, scores_path = args.result.resolve(), args.scores.resolve()
    verified_shards: dict[str, Any] | None = None
    try:
        # This is intentionally the first worker I/O: it validates the immutable
        # parent manifest and current source bytes before importing any helper.
        _verify_parent_provenance(args.supervisor_manifest, result_path=result_path, scores_path=scores_path)
        bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_repeatability_worker_bounded")
        bounded._pinned_environment()
        teacher = _load_local_module("strat02_mmap_teacher.py", "_strat02_repeatability_worker_teacher")
        audit = _load_local_module("strat02_token_audit.py", "_strat02_repeatability_worker_audit")
        score = _load_local_module("strat02_score.py", "_strat02_repeatability_worker_score")
        report = teacher.preflight(snapshot=args.snapshot)
        row = _read_calibration_row(args.calib, args.manifest, bounded, audit)
        tokenizer = bounded._verify_exact_tokenizer(audit, report.snapshot)
        score.validate_strat02_tokenizer(tokenizer)
        payload_ids = score._encode_payload(tokenizer, row["text"])
        verified = teacher.verify_shards(report)  # The sole verification/hash pass over the eleven shards.
        verified_shards = {
            "hashes_match": True,
            "sha256_passes": 1,
            "shards": [{"name": name, "size_bytes": stats[0], "mtime_ns": stats[1]}
                       for name, stats in sorted(verified.shard_stats.items())],
        }
        with teacher.load_reference_model(verified, pointer_check="all") as model:
            model.eval()
            torch = score._torch()
            with torch.inference_mode():
                score_report = _score_pair(model=model, tokenizer=tokenizer, row=row, score=score,
                                           scores_path=scores_path)
                logit_report = _logit_pair(model=model, payload_ids=payload_ids, score=score)
        if logit_report["elements_outside_tolerance"]:
            raise RepeatabilityError(
                f"first64 logits have {logit_report['elements_outside_tolerance']} elements outside frozen tolerance"
            )
        _write_json_once(result_path, {
            "ok": True,
            "status": "PASS_REPEATABILITY",
            "utc_timestamp": _utc_now(),
            "score_repeatability": score_report,
            "logit_repeatability": {key: value for key, value in logit_report.items() if key != "tensor"},
            "verified_shards": verified_shards,
            "protocol": {"calibration_rows_scored": 1, "full_score_calls": 2, "first64_logit_forward_calls": 2,
                         "weight_shard_hash_passes": 1, "heldout_opened_for_scoring": False},
            "notes": ["F32 teacher repeatability only; no heldout BPB, W4 quality, or throughput claim.",
                      "No raw document text, payload IDs, or logit tensor is included."],
        })
        return 0
    except RepeatabilityError as exc:
        try:
            _write_json_once(result_path, {"ok": False, "status": "FAIL_REPEATABILITY", "utc_timestamp": _utc_now(),
                                           "error_type": type(exc).__name__, "error": str(exc),
                                           "partial_scores_persisted": _count_lines(scores_path),
                                           "verified_shards": verified_shards})
        except (GateError, OSError):
            pass
        return 1
    except Exception as exc:
        try:
            _write_json_once(result_path, {"ok": False, "status": "VOID_APPARATUS", "utc_timestamp": _utc_now(),
                                           "error_type": type(exc).__name__, "error": str(exc),
                                           "partial_scores_persisted": _count_lines(scores_path),
                                           "verified_shards": verified_shards,
                                           "traceback": traceback.format_exc()})
        except (GateError, OSError):
            pass
        return 1


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _failure_classification(outcome: Any, worker_result: Mapping[str, Any] | None) -> str | None:
    if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
        return outcome.status
    if worker_result and worker_result.get("status") in {"VOID_APPARATUS", "FAIL_REPEATABILITY"}:
        return str(worker_result["status"])
    if outcome.exit_code != 0:
        return "WORKER_ERROR"
    if not worker_result:
        return "WORKER_PROTOCOL_ERROR"
    if worker_result.get("ok") is True and worker_result.get("status") == "PASS_REPEATABILITY":
        return None
    return "WORKER_ERROR"


def _run_parent(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    provenance: dict[str, str] | None = None
    try:
        provenance = _source_hashes()
        bounded, report, preflight, _ = _metadata_preflight(args)
    except Exception as exc:
        manifest = {"schema": "strat02_teacher_repeatability_v1", "parent_pid": os.getpid(),
                    "provenance_sha256": provenance or {}, "preflight": "failed", "error": str(exc),
                    "raw_text_reported": False, "heldout_opened_for_scoring": False}
        _write_json_once(paths["manifest"], manifest)
        _append_log(paths["log"], {"event": "parent_preflight_failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json_once(paths["result"], {"ok": False, "status": "INCOMPLETE", "failure_classification": "VOID_APPARATUS",
                                           "utc_timestamp": _utc_now(), "parent_pid": os.getpid(), "error_type": type(exc).__name__,
                                           "error": str(exc)})
        print(json.dumps({"ok": False, "status": "INCOMPLETE", "result": str(paths["result"])}, sort_keys=True))
        return 1
    manifest = {
        "schema": "strat02_teacher_repeatability_v1",
        "purpose": "two-repeat F32 teacher calibration score plus first64 logit repeatability",
        "parent_pid": os.getpid(),
        "provenance_sha256": provenance,
        "pins": {"repository": REPOSITORY, "revision": REVISION, "corpus_manifest_sha256": EXPECTED_MANIFEST_SHA256,
                 "calib_jsonl_sha256": EXPECTED_CALIB_SHA256, "calib_text_sha256": EXPECTED_CALIB_TEXT_SHA256},
        "limits": {"launch_available_physical_ram_bytes_min": MIN_LAUNCH_RAM_BYTES,
                   "output_volume_free_bytes_min": MIN_OUTPUT_DISK_BYTES,
                   "runtime_available_physical_ram_bytes_min": MIN_RUNTIME_RAM_BYTES,
                   "worker_working_set_bytes_max": MAX_WORKER_MEMORY_BYTES,
                   "worker_private_commit_bytes_max": MAX_WORKER_MEMORY_BYTES,
                   "wall_seconds_max": MAX_WALL_SECONDS, "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS},
        "calibration": preflight["calibration"],
        "protocol": {"chunk_size": CHUNK_SIZE, "bpb_abs_tolerance": BPB_ABS_TOL, "logit_shape": list(LOGIT_SHAPE),
                     "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL, "full_score_calls": 2,
                     "first64_logit_forward_calls": 2, "weight_shard_hash_passes": 1,
                     "heldout_opened_for_scoring": False, "raw_text_reported": False},
        "parent_preflight": preflight,
        "output": {"scores": paths["scores"].name, "worker_stdout": paths["stdout"].name,
                   "worker_stderr": paths["stderr"].name, "auto_resume": False},
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
    command = [worker_python, str(Path(__file__).resolve()), "--worker", "--snapshot", str(report.snapshot),
               "--manifest", str(args.manifest.resolve()), "--calib", str(args.calib.resolve()),
               "--scores", str(paths["scores"]), "--result", str(paths["worker"]),
               "--supervisor-manifest", str(paths["manifest"]), "--worker-token", token]
    _append_log(paths["log"], {"event": "child_launch", "child_pid_pending": True, "python": worker_python,
                               "termination_scope": "direct_worker_pid_only", "monitor_target": "direct_base_python_worker"})
    process: Any | None = None
    stdout_handle: Any | None = None
    stderr_handle: Any | None = None
    try:
        try:
            stdout_handle = paths["stdout"].open("x", encoding="utf-8", newline="\n")
            stderr_handle = paths["stderr"].open("x", encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise GateError("worker stdout/stderr logs must be write-once") from exc
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
        complete = classification is None
        final: dict[str, Any] = {"ok": complete, "status": "PASS_REPEATABILITY" if complete else "INCOMPLETE",
                                 "failure_classification": classification, "utc_timestamp": _utc_now(),
                                 "parent_pid": os.getpid(), "child_pid": process.pid, "monitor": asdict(outcome),
                                 "worker_result": worker_result, "partial_scores_persisted": _count_lines(paths["scores"]),
                                 "notes": ["A pass authorizes the pinned F32 teacher as an oracle for this W4-v2 control only.",
                                           "Partial score JSONL is retained; no automatic resume or rerun is performed.",
                                           "Only the created direct worker PID may be terminated by supervision."]}
        if paths["scores"].is_file():
            final["scores_sha256"] = _sha256_file(paths["scores"])
    except Exception as exc:
        if process is not None:
            bounded._terminate_only_child(process)
        _append_log(paths["log"], {"event": "parent_supervision_error", "error_type": type(exc).__name__, "error": str(exc)})
        final = {"ok": False, "status": "INCOMPLETE", "failure_classification": "VOID_APPARATUS", "utc_timestamp": _utc_now(),
                 "parent_pid": os.getpid(), "child_pid": process.pid if process is not None else None,
                 "partial_scores_persisted": _count_lines(paths["scores"]), "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        for handle in (stdout_handle, stderr_handle):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
    for label, path in (("worker_stdout_sha256", paths["stdout"]), ("worker_stderr_sha256", paths["stderr"])):
        if path.is_file():
            final[label] = _sha256_file(path)
    _write_json_once(paths["result"], final)
    print(json.dumps({"ok": final["ok"], "status": final["status"], "result": str(paths["result"])}, sort_keys=True))
    return 0 if final["ok"] else 1


def _selftest() -> None:
    """Exercise fake-only scoring, logit, write-once, and monitor controls."""
    score = _load_local_module("strat02_score.py", "_strat02_repeatability_selftest_score")
    row = {"source_document_id": EXPECTED_CALIB_ID, "category": EXPECTED_CALIB_CATEGORY, "text": "synthetic"}
    calls: list[int] = []

    def fake_score(model: Any, tokenizer: Any, document: Mapping[str, Any], *, chunk_size: int) -> Any:
        assert model == "model" and tokenizer == "tokenizer" and chunk_size == CHUNK_SIZE
        calls.append(1)
        return SimpleNamespace(source_document_id=document["source_document_id"], category=document["category"],
                               tokens=64, bytes=EXPECTED_CALIB_BYTES, bits=1024.0)

    with tempfile.TemporaryDirectory(prefix="strat02-repeatability-selftest-") as directory:
        score_path = Path(directory) / "scores.jsonl"
        report = _score_pair(model="model", tokenizer="tokenizer", row=row, score=score,
                             scores_path=score_path, score_document=fake_score)
        assert report["score_calls"] == 2 and len(calls) == 2 and _count_lines(score_path) == 2
        try:
            _score_pair(model="model", tokenizer="tokenizer", row=row, score=score,
                        scores_path=score_path, score_document=fake_score)
        except GateError:
            pass
        else:
            raise AssertionError("write-once score output was overwritten")

    torch = score._torch()
    first = torch.zeros((1, 2, 3), dtype=torch.float32)
    second = first.clone()
    second[0, 0, 0] = 2e-5
    summary = _compare_logit_blocks(first, second, expected_shape=(1, 2, 3))
    assert math.isclose(summary["max_abs_error"], 2e-5, rel_tol=0.0, abs_tol=1e-10)
    assert summary["elements_outside_tolerance"] == 1
    nonfinite = second.clone(); nonfinite[0, 0, 1] = float("nan")
    try:
        _compare_logit_blocks(first, nonfinite, expected_shape=(1, 2, 3))
    except GateError:
        pass
    else:
        raise AssertionError("non-finite planted logit was accepted")

    bounded = _load_local_module("strat02_bounded_smoke.py", "_strat02_repeatability_selftest_bounded")
    class Clock:
        value = 0.0
        def now(self) -> float: return self.value

    class FakeProcess:
        pid = 12345
        def __init__(self, clock: Clock) -> None: self.clock, self.returncode, self.terminated = clock, None, False
        def poll(self): return self.returncode
        def wait(self, timeout):
            if self.returncode is not None:
                return self.returncode
            self.clock.value += timeout
            raise subprocess.TimeoutExpired("fake", timeout)
        def terminate(self): self.terminated, self.returncode = True, -15
        def kill(self): self.returncode = -9
        def memory_info(self): return SimpleNamespace(rss=1, pagefile=1, peak_wset=1, peak_pagefile=1, num_page_faults=1)
        def memory_full_info(self): return SimpleNamespace(private=1)

    psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=MIN_RUNTIME_RAM_BYTES - 1))
    child = FakeProcess(Clock())
    outcome = bounded._monitor_child(child, psutil=psutil, resource_process=child, on_sample=lambda _: None,
                                     now=child.clock.now, interval_seconds=MONITOR_INTERVAL_SECONDS)
    assert outcome.status == "VOID_RESOURCE" and child.terminated
    print(json.dumps({"ok": True, "selftest": True, "score_calls": 2, "logit_forward_calls": 2}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selftest", action="store_true", help="run fake-only synthetic controls")
    action.add_argument("--preflight", action="store_true", help="run read-only metadata gates; no shard hashes or forwards")
    action.add_argument("--run", action="store_true", help="launch the one bounded real repeatability worker")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help="new or empty write-once output directory")
    parser.add_argument("--snapshot", type=Path, help="pinned local snapshot directory; never a Hub identifier")
    parser.add_argument("--manifest", type=Path, default=CORPUS / "manifest.json")
    parser.add_argument("--calib", type=Path, default=CORPUS / "calib.jsonl")
    parser.add_argument("--scores", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest(); return 0
        if args.preflight:
            _, report, metadata, _ = _metadata_preflight(args)
            print(json.dumps({"ok": True, "preflight": metadata, "snapshot": str(report.snapshot)}, sort_keys=True))
            return 0
        if args.worker:
            if args.result is None or args.scores is None or args.supervisor_manifest is None:
                raise GateError("internal worker requires --result, --scores, and --supervisor-manifest")
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
