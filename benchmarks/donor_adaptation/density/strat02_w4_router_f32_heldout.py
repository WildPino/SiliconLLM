#!/usr/bin/env python3
"""STRAT-02F's single, supervised W4-router-F32 heldout runner.

``--preflight`` is deliberately metadata-only: it hashes pins and checks the
offline runtime, but does not open F32 source values or heldout score values.
Only the child launched by ``--run`` can construct the single W4 arena,
temporarily replace the exact router organ, and score the frozen heldout set.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
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
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from benchmarks.donor_adaptation.density import strat02_bounded_smoke as bounded
from benchmarks.donor_adaptation.density import strat02_mmap_teacher as teacher
from benchmarks.donor_adaptation.density import strat02_score as score_core
from benchmarks.donor_adaptation.density import strat02_token_audit as audit
from benchmarks.donor_adaptation.density import strat02_w4_bf16_candidate_loader as candidate
from benchmarks.donor_adaptation.density import strat02_w4_bf16_export_verify as verifier
from benchmarks.donor_adaptation.density import strat02_w4_bf16_quality as w4_quality
from benchmarks.donor_adaptation.density import strat02_w4_organ_attribution as organ


SCHEMA = "strat02_w4_router_f32_heldout_v1"
BRIEF = PROJECT / "docs/research/donor_adaptation/briefs/BRIEF_STRAT_02F_W4_ROUTER_F32_HELDOUT.md"
CORPUS = HERE / "corpus/strat02_document_holdout_v1"
MANIFEST = CORPUS / "manifest.json"
CALIB = CORPUS / "calib.jsonl"
HELDOUT = CORPUS / "heldout.jsonl"
EXPORT = candidate.BOUND_EXPORT
TEACHER_SCORES = HERE / "results/strat02_teacher_baseline_20260917_105941/teacher_scores.jsonl"
W4_HELDOUT_CONTROL = HERE / "results/strat02_w4_bf16_quality_20260917_162329/candidate_scores.jsonl"
W4_CALIBRATION_CONTROL = HERE / "results/strat02_w4_bf16_quality_20260917_162329/calibration_checks.jsonl"
ROUTER_F32_CALIBRATION = HERE / "results/strat02_w4_organ_attribution_20260917_183751/router_f32.jsonl"

EXPECTED_MANIFEST_SHA256 = "56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749"
EXPECTED_CALIB_SHA256 = "f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f"
EXPECTED_HELDOUT_SHA256 = "450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e"
EXPECTED_HELDOUT_IDS_SHA256 = "5a4cf31ec7db0f3bc541f2e75a78b133a50cd3d6174abb26ef81e08c45e65289"
EXPECTED_HELDOUT_DOCUMENTS = 96
EXPECTED_HELDOUT_TOKENS = 181_385
EXPECTED_HELDOUT_BYTES = 746_161
EXPECTED_PER_CATEGORY = 32
W4_HELDOUT_CONTROL_SHA256 = "0ecbac5406b926240ddba3dc9eb25ca5239eb37ba717a5970aee7fb192a8039f"
W4_CALIBRATION_CONTROL_SHA256 = organ.W4_CALIBRATION_SHA256
ROUTER_F32_CALIBRATION_SHA256 = "f7913767de5c65396d90473cfd7302961c615ec09bcc0eeb65b3ff428a78550b"
SENTINEL_TOLERANCE_BITS = 1e-5
ROUTER_RECORDS = 16
BASE_W4_ACTIVE_BYTES = 661_782_528
ROUTER_F32_EXTRA_BYTES = 14_614_528
ROUTER_F32_ACTIVE_BYTES = BASE_W4_ACTIVE_BYTES + ROUTER_F32_EXTRA_BYTES
BPB_UPPER_CI_MAX = 0.02
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_916
CHUNK_SIZE = 128
MIN_LAUNCH_RAM = 55 * 1024**3
MIN_OUTPUT_DISK = 1 * 1024**3
MIN_RUNTIME_RAM = 8 * 1024**3
MAX_WORKER_MEMORY = 70 * 1024**3
MAX_WALL_SECONDS = 6 * 60 * 60
MONITOR_SECONDS = 5.0
WORKER_TOKEN_ENV = "STRAT02F_ROUTER_F32_PARENT_TOKEN"

# Never add output artifacts here.  The parent freezes this source set before
# creating its manifest, so provenance cannot become self-referential.
PROVENANCE_SOURCES = {
    "brief": BRIEF,
    "runner": Path(__file__).resolve(),
    "w4_organ_attribution": HERE / "strat02_w4_organ_attribution.py",
    "w4_quality_runner": HERE / "strat02_w4_bf16_quality.py",
    "score_core": HERE / "strat02_score.py",
    "candidate_loader": HERE / "strat02_w4_bf16_candidate_loader.py",
    "teacher": HERE / "strat02_mmap_teacher.py",
    "token_audit": HERE / "strat02_token_audit.py",
    "export_verifier": HERE / "strat02_w4_bf16_export_verify.py",
    "bounded_supervisor": HERE / "strat02_bounded_smoke.py",
}


class ApparatusError(RuntimeError):
    """A frozen identity, copy, parity, or output invariant failed."""


class ResourceError(MemoryError):
    """A preregistered resource bound was crossed."""


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        handle.write("\n")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _log(path: Path, event: Mapping[str, Any]) -> None:
    _append_jsonl(path, {"utc_timestamp": _utc(), **event})


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApparatusError(f"expected JSON object: {path.name}")
    return value


def _provenance() -> dict[str, str]:
    observed: dict[str, str] = {}
    for label, path in PROVENANCE_SOURCES.items():
        if not path.is_file():
            raise ApparatusError(f"provenance source missing: {path}")
        observed[label] = _sha_file(path)
    return observed


def _require_safe_kernel_policy() -> None:
    current = os.environ.get("USE_HUB_KERNELS")
    if current is None:
        os.environ["USE_HUB_KERNELS"] = "NO"
    elif current != "NO":
        raise ApparatusError(f"USE_HUB_KERNELS must be NO; found {current!r}")


def _existing_output_volume(path: Path | None) -> Path:
    volume = Path(path or HERE).expanduser().resolve()
    while not volume.exists():
        parent = volume.parent
        if parent == volume:
            raise ApparatusError("no existing ancestor for output volume")
        volume = parent
    if not volume.is_dir():
        raise ApparatusError("output volume is not a directory")
    return volume


def _check_hash(path: Path, expected: str, label: str) -> None:
    if _sha_file(path) != expected:
        raise ApparatusError(f"{label} SHA-256 mismatch")


def _metadata_preflight(snapshot: Path | None, output_dir: Path | None) -> tuple[Any, dict[str, Any]]:
    """Hash/control/runtime checks only; deliberately no source or score values."""
    _require_safe_kernel_policy()
    versions = bounded._pinned_environment()
    report = teacher.preflight(snapshot=snapshot)
    status = report.source_status
    if (status.get("repository") != teacher.REPOSITORY or status.get("revision") != teacher.REVISION or
            not status.get("transformers_ok") or status.get("use_hub_kernels") != "NO" or
            not status.get("shards_present")):
        raise ApparatusError("pinned offline source metadata/runtime invalid")
    candidate.check_bound_controls(EXPORT)
    candidate.check_quality_prerequisites()  # hashes the teacher score artifact without parsing values
    _check_hash(MANIFEST, EXPECTED_MANIFEST_SHA256, "frozen corpus manifest")
    _check_hash(CALIB, EXPECTED_CALIB_SHA256, "frozen calibration corpus")
    _check_hash(HELDOUT, EXPECTED_HELDOUT_SHA256, "frozen heldout corpus")
    _check_hash(W4_CALIBRATION_CONTROL, W4_CALIBRATION_CONTROL_SHA256, "W4 calibration control")
    _check_hash(W4_HELDOUT_CONTROL, W4_HELDOUT_CONTROL_SHA256, "W4 heldout control")
    _check_hash(ROUTER_F32_CALIBRATION, ROUTER_F32_CALIBRATION_SHA256, "ROUTER_F32 calibration control")
    volume = _existing_output_volume(output_dir)
    import psutil
    available, free = int(psutil.virtual_memory().available), int(psutil.disk_usage(str(volume)).free)
    if available < MIN_LAUNCH_RAM or free < MIN_OUTPUT_DISK:
        raise ResourceError(f"launch cap: available_ram={available}, output_free={free}")
    return report, {
        "runtime_versions": versions,
        "snapshot": str(report.snapshot),
        "source_manifest_sha256": teacher.EXPECTED_MANIFEST_SHA256,
        "source_index_sha256": report.manifest["index_sha256"],
        "source_shards_expected": [{"name": item["name"], "sha256": item["sha256"]}
                                   for item in report.manifest["shards"]],
        "export_control_sha256": dict(candidate.BOUND_CONTROL_SHA256),
        "control_sha256": {"w4_calibration": W4_CALIBRATION_CONTROL_SHA256,
                             "w4_heldout": W4_HELDOUT_CONTROL_SHA256,
                             "router_f32_calibration": ROUTER_F32_CALIBRATION_SHA256,
                             "teacher": candidate.QUALITY_PREREQUISITE_SHA256[TEACHER_SCORES]},
        "corpus_sha256": {"manifest": EXPECTED_MANIFEST_SHA256, "calib": EXPECTED_CALIB_SHA256,
                           "heldout": EXPECTED_HELDOUT_SHA256},
        "available_ram_bytes": available, "output_free_bytes": free, "output_volume": str(volume),
        "source_weight_values_opened": False, "heldout_score_values_opened": False,
        "heldout_forward": False,
    }


def _router_records(plan: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[Mapping[str, Any], ...]:
    """Select the router by frozen structural label, never by a name regex."""
    arms = organ.build_arms(plan)
    records = tuple(arms["ROUTER_F32"]["records"])
    if (len(records) != ROUTER_RECORDS or {record.get("organ") for record in records} != {"router"} or
            any(record.get("encoding") != verifier.LINEAR_ENCODING for record in records)):
        raise ApparatusError("expected exactly 16 router W4 linear records")
    all_linear = [record for entries in plan.values() for record in entries
                  if record.get("encoding") == verifier.LINEAR_ENCODING]
    if {record["name"] for record in records} - {record["name"] for record in all_linear}:
        raise ApparatusError("router selection is not a plan subset")
    return records


def _copy_router_sources(records: Sequence[Mapping[str, Any]], state: Mapping[str, Any], report: Any) -> dict[str, str]:
    """Copy only the selected router records, checking raw F32 SHA per record."""
    if len(records) != ROUTER_RECORDS:
        raise ApparatusError("router copy count differs from frozen 16")
    order = [entry["name"] for entry in report.manifest["shards"]]
    grouped: dict[str, list[Mapping[str, Any]]] = {shard: [] for shard in order}
    index = report.index["weight_map"]
    for record in records:
        key, shard = record["name"], record["source_shard"]
        if (record.get("organ") != "router" or shard not in grouped or index.get(key) != shard or key not in state):
            raise ApparatusError("router source/index/state binding mismatch")
        grouped[shard].append(record)
    copied: dict[str, str] = {}
    for shard in order:
        if not grouped[shard]:
            continue
        with organ._source_open(report.snapshot / shard) as handle:
            expected_keys = {key for key, mapped in index.items() if mapped == shard}
            if set(handle.keys()) != expected_keys:
                raise ApparatusError(f"source shard keyset changed before router copy: {shard}")
            for record in grouped[shard]:
                copied[record["name"]] = organ._copy_f32_record(handle, record, state[record["name"]])
    if set(copied) != {record["name"] for record in records}:
        raise ApparatusError("router F32 copy did not cover exactly 16 records")
    return copied


def _restore_router_w4(records: Sequence[Mapping[str, Any]], handles: Mapping[str, Any],
                       state: Mapping[str, Any], originals: Mapping[str, str]) -> None:
    if len(records) != ROUTER_RECORDS or set(originals) != {record["name"] for record in records}:
        raise ApparatusError("router rollback keyset differs from original W4 keyset")
    for record in records:
        shard, key = record["source_shard"], record["name"]
        if shard not in handles:
            raise ApparatusError("router W4 payload handle missing")
        organ._restore_w4_record(handles[shard], record, state[key], originals[key])


def _read_router_calibration_sentinel(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    _check_hash(ROUTER_F32_CALIBRATION, ROUTER_F32_CALIBRATION_SHA256, "ROUTER_F32 calibration control")
    lines = ROUTER_F32_CALIBRATION.read_text(encoding="utf-8").splitlines()
    if len(lines) != 48:
        raise ApparatusError("ROUTER_F32 calibration control must contain 48 rows")
    records = [json.loads(line) for line in lines]
    for ordinal, (row, record) in enumerate(zip(rows, records, strict=True), 1):
        if (record.get("source_document_id") != row.get("source_document_id") or
                record.get("category") != row.get("category") or
                record.get("bytes") != len(str(row.get("text", "")).encode("utf-8")) or
                not isinstance(record.get("tokens"), int) or record["tokens"] <= 0 or
                not isinstance(record.get("bits"), (int, float)) or not math.isfinite(float(record["bits"]))):
            raise ApparatusError(f"ROUTER_F32 calibration control row {ordinal} differs from frozen calibration")
    return records[0]


def _read_heldout_rows(snapshot: Path) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    """The worker-only corpus/token audit; no score control is parsed here."""
    manifest, _calib, heldout, summary = w4_quality._audit_full_corpus(
        audit, manifest=MANIFEST, calib=CALIB, heldout=HELDOUT, snapshot=snapshot
    )
    if (manifest.get("schema") is None or len(heldout) != EXPECTED_HELDOUT_DOCUMENTS or
            summary.get("heldout_tokens") != EXPECTED_HELDOUT_TOKENS or
            summary.get("heldout_bytes") != EXPECTED_HELDOUT_BYTES):
        raise ApparatusError("heldout corpus inventory differs from frozen pin")
    tokenizer, _, _, _ = audit.load_tokenizer(snapshot)
    score_core.validate_strat02_tokenizer(tokenizer)
    return heldout, tokenizer, summary


def _score_heldout_once(*, model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]],
                        output: Path, scorer: Callable[..., Any] = score_core.score_document) -> dict[str, Any]:
    """Write exactly 96 bound rows in original order; no raw text/IDs/logits leak."""
    if len(rows) != EXPECTED_HELDOUT_DOCUMENTS:
        raise ApparatusError("heldout score call requires exactly 96 rows")
    totals = {"documents": 0, "tokens": 0, "bytes": 0, "bits": 0.0}
    categories = {name: {"documents": 0, "tokens": 0, "bytes": 0, "bits": 0.0}
                  for name in ("code", "prose", "technical_general")}
    try:
        handle = output.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise ApparatusError(f"write-once heldout output already exists: {output}") from exc
    with handle:
        for ordinal, row in enumerate(rows, 1):
            started = time.monotonic()
            observation = scorer(model, tokenizer, row, chunk_size=CHUNK_SIZE)
            wall_seconds = time.monotonic() - started
            expected_tokens = len(score_core._encode_payload(tokenizer, row["text"]))
            expected_bytes = len(row["text"].encode("utf-8"))
            if ((observation.source_document_id, observation.category) !=
                    (row["source_document_id"], row["category"]) or
                    type(observation.tokens) is not int or observation.tokens != expected_tokens or
                    type(observation.bytes) is not int or observation.bytes != expected_bytes or
                    not math.isfinite(float(observation.bits)) or float(observation.bits) < 0 or
                    not math.isfinite(wall_seconds) or wall_seconds < 0):
                raise ApparatusError(f"heldout:{ordinal}: score identity/token/byte/finite check failed")
            item = {"source_document_id": observation.source_document_id, "category": observation.category,
                    "tokens": observation.tokens, "bytes": observation.bytes, "bits": float(observation.bits),
                    "wall_seconds": wall_seconds}
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            for bucket in (totals, categories[observation.category]):
                bucket["documents"] += 1
                bucket["tokens"] += observation.tokens
                bucket["bytes"] += observation.bytes
                bucket["bits"] += float(observation.bits)
    if (totals["documents"] != EXPECTED_HELDOUT_DOCUMENTS or totals["tokens"] != EXPECTED_HELDOUT_TOKENS or
            totals["bytes"] != EXPECTED_HELDOUT_BYTES or
            any(bucket["documents"] != EXPECTED_PER_CATEGORY for bucket in categories.values())):
        raise ApparatusError("heldout totals/category counts differ from frozen protocol")
    for bucket in categories.values():
        bucket["bpb"] = bucket["bits"] / bucket["bytes"]
    totals["bpb"] = totals["bits"] / totals["bytes"]
    totals["categories"] = categories
    totals["order"] = "original heldout.jsonl order"
    return totals


def _validate_persisted_heldout(path: Path, rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != EXPECTED_HELDOUT_DOCUMENTS:
        raise ApparatusError("persisted heldout score output is incomplete")
    scores = score_core.read_score_jsonl(path)
    score_core.validate_scores_against_heldout(scores, rows, tokenizer, arm="router_f32_candidate")
    for ordinal, (line, row) in enumerate(zip(lines, rows, strict=True), 1):
        item = json.loads(line)
        if item.get("source_document_id") != row["source_document_id"] or item.get("category") != row["category"]:
            raise ApparatusError(f"heldout output order changed at row {ordinal}")


def _router_protocol(*, w4_sentinel: Callable[[], None], router_sentinel: Callable[[], None],
                     copy_router: Callable[[], Mapping[str, str]], rollback_w4: Callable[[], None],
                     score_heldout: Callable[[], dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    """Run the preregistered sentinel/copy/rollback state machine once."""
    first: dict[str, str] | None = None
    preflight_error: BaseException | None = None
    try:
        w4_sentinel()
        first = dict(copy_router())
        if len(first) != ROUTER_RECORDS:
            raise ApparatusError("first F32 router copy did not report 16 SHA records")
        router_sentinel()
    except BaseException as exc:
        preflight_error = exc
    try:
        rollback_w4()
        w4_sentinel()
    except BaseException as exc:
        raise ApparatusError("pre-heldout W4 rollback or sentinel failed") from exc
    if preflight_error is not None:
        raise preflight_error

    second: dict[str, str] | None = None
    work_error: BaseException | None = None
    summary: dict[str, Any] | None = None
    try:
        second = dict(copy_router())
        if second != first or len(second) != ROUTER_RECORDS:
            raise ApparatusError("second F32 router copy SHA set differs from verified first copy")
        summary = score_heldout()
    except BaseException as exc:
        work_error = exc
    try:
        rollback_w4()
        w4_sentinel()
    except BaseException as exc:
        # This deliberately supersedes a successful 96-row score: the brief
        # requires VOID_APPARATUS and forbids adjudication in that case.
        raise ApparatusError("final W4 rollback or sentinel failed") from exc
    if work_error is not None:
        raise work_error
    if second is None or summary is None:
        raise ApparatusError("router protocol ended without a heldout summary")
    return second, summary


def _control_gain(control_path: Path, candidate_path: Path, rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    _check_hash(control_path, W4_HELDOUT_CONTROL_SHA256, "W4 heldout control")
    control = score_core.read_score_jsonl(control_path)
    trial = score_core.read_score_jsonl(candidate_path)
    score_core.validate_scores_against_heldout(control, rows, tokenizer, arm="w4_control")
    score_core.validate_scores_against_heldout(trial, rows, tokenizer, arm="router_f32_candidate")
    paired = score_core.adjudicate(control, trial)
    result = {
        "control_bpb": paired["teacher_bpb"], "candidate_bpb": paired["candidate_bpb"],
        "candidate_minus_control_bpb": paired["point_delta_bpb"],
        "gain_paired_against_w4_control_bpb": -float(paired["point_delta_bpb"]),
        "categories": {},
    }
    for category, value in paired["categories"].items():
        result["categories"][category] = {
            "control_bpb": value["teacher_bpb"], "candidate_bpb": value["candidate_bpb"],
            "candidate_minus_control_bpb": value["point_delta_bpb"],
            "gain_paired_against_w4_control_bpb": -float(value["point_delta_bpb"]),
            "documents": value["documents"], "tokens": value["tokens"], "bytes": value["bytes"],
        }
    return result


def _worker_failure(path: Path, output: Path, exc: BaseException) -> None:
    try:
        rows = len(output.read_text(encoding="utf-8").splitlines()) if output.is_file() else 0
        _write_once(path, {"ok": False, "status": "VOID_RESOURCE" if isinstance(exc, MemoryError) else "VOID_APPARATUS",
                           "schema": SCHEMA, "utc_timestamp": _utc(), "heldout_rows_persisted": rows,
                           "heldout_scoring_started": rows > 0, "partial_preserved": True,
                           "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                           "raw_text_reported": False, "token_ids_reported": False, "logits_reported": False})
    except OSError:
        pass


def _worker(args: argparse.Namespace) -> int:
    result_path, output = args.worker_result.resolve(), args.candidate_scores.resolve()
    try:
        _require_safe_kernel_policy()
        manifest = _json(args.supervisor_manifest)
        if manifest.get("schema") != SCHEMA or manifest.get("provenance_sha256") != _provenance():
            raise ApparatusError("worker provenance differs from frozen parent map")
        bounded._pinned_environment()
        report = teacher.preflight(snapshot=args.snapshot)
        heldout_rows, tokenizer, corpus_summary = _read_heldout_rows(report.snapshot)
        calib_rows = organ._read_calibration()
        w4_calibration = organ._baseline(calib_rows)
        router_calibration = _read_router_calibration_sentinel(calib_rows)
        verified = teacher.verify_shards(report)  # all eleven source shards before any F32 value is opened
        organ._audit_source_keysets(report)       # headers only, all handles closed before arena allocation
        prepared = candidate.prepare_candidate(EXPORT, snapshot=report.snapshot, output_dir=args.output_dir)
        routers = _router_records(prepared.plan)
        if any(report.index["weight_map"].get(record["name"]) != record["source_shard"] for record in routers):
            raise ApparatusError("router plan/source shard binding differs from pinned index")
        with contextlib.ExitStack() as stack:
            handles = {descriptor["source_shard"]: stack.enter_context(
                verifier._safe_child(prepared.root, descriptor["payload_file"]).open("rb"))
                for descriptor in prepared.descriptors}
            with organ._load_calibration_candidate(prepared) as model:
                model.eval()
                state = model.state_dict()
                originals = {record["name"]: organ._tensor_sha(state[record["name"]]) for record in routers}
                sentinel = calib_rows[0]
                def checked_copy() -> Mapping[str, str]:
                    # Detect any post-hash source mutation before either F32 copy.
                    teacher._assert_unchanged(verified)
                    return _copy_router_sources(routers, state, report)
                copied, heldout_summary = _router_protocol(
                    w4_sentinel=lambda: organ._check_sentinel(
                        score_core.score_document(model, tokenizer, sentinel, chunk_size=CHUNK_SIZE), w4_calibration[0]),
                    router_sentinel=lambda: organ._check_sentinel(
                        score_core.score_document(model, tokenizer, sentinel, chunk_size=CHUNK_SIZE), router_calibration),
                    copy_router=checked_copy,
                    rollback_w4=lambda: _restore_router_w4(routers, handles, state, originals),
                    score_heldout=lambda: _score_heldout_once(model=model, tokenizer=tokenizer, rows=heldout_rows, output=output),
                )
                teacher._assert_unchanged(verified)
        _validate_persisted_heldout(output, heldout_rows, tokenizer)
        # Adjudication is intentionally after the final rollback/sentinel.
        teacher_report = score_core.adjudicate_pinned_files(TEACHER_SCORES, output, heldout_rows, tokenizer)
        gain = _control_gain(W4_HELDOUT_CONTROL, output, heldout_rows, tokenizer)
        upper = teacher_report.get("upper_one_sided_ci95_delta_bpb")
        if not isinstance(upper, (int, float)) or not math.isfinite(float(upper)):
            raise ApparatusError("teacher paired upper CI is non-finite")
        status = "PASS_BPB" if float(upper) <= BPB_UPPER_CI_MAX else "FAIL_BPB"
        _write_once(result_path, {
            "ok": status == "PASS_BPB", "status": status, "schema": SCHEMA, "utc_timestamp": _utc(),
            "source_shards_verified": 11,
            "source_shard_sha256": {item["name"]: item["sha256"] for item in report.manifest["shards"]},
            "export_control_sha256": dict(candidate.BOUND_CONTROL_SHA256),
            "control_sha256": {"w4_calibration": W4_CALIBRATION_CONTROL_SHA256,
                               "w4_heldout": W4_HELDOUT_CONTROL_SHA256,
                               "router_f32_calibration": ROUTER_F32_CALIBRATION_SHA256},
            "router_records": ROUTER_RECORDS, "router_f32_active_bytes_per_token": ROUTER_F32_ACTIVE_BYTES,
            "router_f32_tensor_sha256": copied, "other_organs_mutated": False,
            "corpus": corpus_summary, "heldout": heldout_summary,
            "teacher_paired_adjudication": teacher_report, "w4_control_gain": gain,
            "protocol": {"one_arena": True, "heldout_documents_scored_once": EXPECTED_HELDOUT_DOCUMENTS,
                         "heldout_order": "original heldout.jsonl order", "chunk_head": CHUNK_SIZE,
                         "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                         "final_w4_rollback_sha_verified": True, "final_w4_sentinel_verified": True,
                         "raw_text_reported": False, "token_ids_reported": False, "logits_reported": False},
            "candidate_scores_sha256": _sha_file(output),
            "task_rollout": "NOT_RUN_BPB_ONLY",
        })
        return 0 if status == "PASS_BPB" else 1
    except Exception as exc:
        _worker_failure(result_path, output, exc)
        return 1


def _prepare_output(root: Path) -> dict[str, Path]:
    root = root.resolve()
    if root.exists():
        raise ApparatusError(f"output directory must be new: {root}")
    root.mkdir(parents=True, exist_ok=False)
    paths = {"root": root, "manifest": root / "supervisor_manifest.json", "log": root / "supervisor_log.jsonl",
             "worker": root / "worker_result.json", "result": root / "supervisor_result.json",
             "candidate": root / "candidate_scores.jsonl", "stdout": root / "worker_stdout.log",
             "stderr": root / "worker_stderr.log"}
    with paths["log"].open("x", encoding="utf-8", newline="\n"):
        pass
    return paths


def _parent_validate(paths: Mapping[str, Path], worker: Mapping[str, Any] | None) -> None:
    if not worker or worker.get("status") not in {"PASS_BPB", "FAIL_BPB"}:
        raise ApparatusError("worker did not produce an adjudicable result")
    if worker.get("candidate_scores_sha256") != _sha_file(paths["candidate"]):
        raise ApparatusError("parent candidate score SHA differs from worker result")
    lines = paths["candidate"].read_text(encoding="utf-8").splitlines()
    if len(lines) != EXPECTED_HELDOUT_DOCUMENTS or len(score_core.read_score_jsonl(paths["candidate"])) != EXPECTED_HELDOUT_DOCUMENTS:
        raise ApparatusError("parent found incomplete heldout artifact")
    protocol = worker.get("protocol", {})
    if (worker.get("router_records") != ROUTER_RECORDS or worker.get("other_organs_mutated") is not False or
            protocol.get("final_w4_rollback_sha_verified") is not True or protocol.get("final_w4_sentinel_verified") is not True):
        raise ApparatusError("worker did not prove final router-only W4 rollback")


def _run_parent(args: argparse.Namespace) -> int:
    paths = _prepare_output(args.output_dir)
    provenance: dict[str, str] = {}
    try:
        provenance = _provenance()
        report, preflight = _metadata_preflight(args.snapshot, paths["root"])
    except Exception as exc:
        status = "VOID_RESOURCE" if isinstance(exc, MemoryError) else "VOID_APPARATUS"
        _write_once(paths["manifest"], {"schema": SCHEMA, "provenance_sha256": provenance, "preflight": "failed"})
        _log(paths["log"], {"event": "parent_preflight_failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_once(paths["result"], {"ok": False, "status": status, "utc_timestamp": _utc(),
                                      "error_type": type(exc).__name__, "error": str(exc), "partial_preserved": True})
        print(json.dumps({"ok": False, "status": status, "result": str(paths["result"])}, sort_keys=True))
        return 1
    _write_once(paths["manifest"], {
        "schema": SCHEMA, "utc_timestamp": _utc(), "provenance_sha256": provenance, "preflight": preflight,
        "pins": {"repository": teacher.REPOSITORY, "revision": teacher.REVISION,
                 "heldout_documents": EXPECTED_HELDOUT_DOCUMENTS, "heldout_tokens": EXPECTED_HELDOUT_TOKENS,
                 "heldout_bytes": EXPECTED_HELDOUT_BYTES, "heldout_ordered_token_ids_sha256": EXPECTED_HELDOUT_IDS_SHA256},
        "protocol": {"router_records": ROUTER_RECORDS, "only_router_f32": True, "one_arena": True,
                     "heldout_score_calls": EXPECTED_HELDOUT_DOCUMENTS, "bootstrap_draws": BOOTSTRAP_DRAWS,
                     "bootstrap_seed": BOOTSTRAP_SEED, "upper_ci95_gate": BPB_UPPER_CI_MAX,
                     "auto_resume": False, "output_write_once": True},
        "caps": {"launch_ram_min": MIN_LAUNCH_RAM, "launch_disk_min": MIN_OUTPUT_DISK,
                 "runtime_ram_min": MIN_RUNTIME_RAM, "worker_private_max": MAX_WORKER_MEMORY,
                 "worker_working_set_max": MAX_WORKER_MEMORY, "wall_seconds_max": MAX_WALL_SECONDS,
                 "monitor_interval_seconds": MONITOR_SECONDS},
    })
    child_env = os.environ.copy()
    child_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HUB_KERNELS": "NO"})
    worker_python = bounded._direct_worker_python(child_env)
    token = secrets.token_urlsafe(32)
    child_env[WORKER_TOKEN_ENV] = token
    command = [worker_python, str(Path(__file__).resolve()), "--worker", "--snapshot", str(report.snapshot),
               "--output-dir", str(paths["root"]), "--supervisor-manifest", str(paths["manifest"]),
               "--worker-result", str(paths["worker"]), "--candidate-scores", str(paths["candidate"]),
               "--worker-token", token]
    process: Any | None = None
    try:
        with paths["stdout"].open("x", encoding="utf-8") as stdout, paths["stderr"].open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=str(HERE), env=child_env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr)
            _log(paths["log"], {"event": "child_launch", "pid": process.pid, "direct_worker_only": True})
            outcome = bounded._monitor_or_terminate(process, psutil=bounded._psutil(), expected_executable=Path(worker_python),
                                                     wall_limit_seconds=MAX_WALL_SECONDS, interval_seconds=MONITOR_SECONDS,
                                                     on_sample=lambda sample: _log(paths["log"], {"event": "resource_sample", **sample}))
        worker = _json(paths["worker"]) if paths["worker"].is_file() else None
        if outcome.status in {"VOID_RESOURCE", "VOID_APPARATUS"}:
            final_status = outcome.status
        elif worker and worker.get("status") == "VOID_RESOURCE":
            final_status = "VOID_RESOURCE"
        elif worker and worker.get("status") == "VOID_APPARATUS":
            final_status = "VOID_APPARATUS"
        else:
            _parent_validate(paths, worker)
            final_status = str(worker["status"])
        result: dict[str, Any] = {"ok": final_status == "PASS_BPB", "status": final_status, "schema": SCHEMA,
                                  "utc_timestamp": _utc(), "monitor": asdict(outcome), "worker": worker,
                                  "provenance_sha256": provenance, "heldout_rows_persisted": len(paths["candidate"].read_text(encoding="utf-8").splitlines()) if paths["candidate"].is_file() else 0,
                                  "partial_preserved": True}
    except Exception as exc:
        if process is not None:
            bounded._terminate_only_child(process)
        result = {"ok": False, "status": "VOID_APPARATUS", "schema": SCHEMA, "utc_timestamp": _utc(),
                  "error_type": type(exc).__name__, "error": str(exc), "partial_preserved": True}
    for label in ("candidate", "worker", "stdout", "stderr"):
        if paths[label].is_file():
            result[f"{label}_sha256"] = _sha_file(paths[label])
    _write_once(paths["result"], result)
    print(json.dumps({"ok": result["ok"], "status": result["status"], "result": str(paths["result"])}, sort_keys=True))
    return 0 if result["ok"] else 1


def _selftest() -> None:
    """Synthetic state-machine checks; no source shards, corpus, or model values."""
    events: list[str] = []
    copy_count = 0
    def copy() -> Mapping[str, str]:
        nonlocal copy_count
        copy_count += 1
        events.append(f"copy-{copy_count}")
        return {f"router-{index}": f"{index:064x}" for index in range(ROUTER_RECORDS)}
    def w4() -> None: events.append("w4")
    def router() -> None: events.append("router")
    def rollback() -> None: events.append("rollback")
    def score() -> dict[str, Any]:
        events.append("heldout")
        return {"documents": 96}
    copied, summary = _router_protocol(w4_sentinel=w4, router_sentinel=router, copy_router=copy,
                                       rollback_w4=rollback, score_heldout=score)
    assert len(copied) == ROUTER_RECORDS and summary["documents"] == 96
    assert events == ["w4", "copy-1", "router", "rollback", "w4", "copy-2", "heldout", "rollback", "w4"]
    heldout_calls = 0
    def broken_copy() -> Mapping[str, str]:
        raise ApparatusError("planted preflight copy failure")
    def forbidden_score() -> dict[str, Any]:
        nonlocal heldout_calls
        heldout_calls += 1
        raise AssertionError("heldout called after failed preflight")
    try:
        _router_protocol(w4_sentinel=lambda: None, router_sentinel=lambda: None, copy_router=broken_copy,
                         rollback_w4=lambda: None, score_heldout=forbidden_score)
    except ApparatusError:
        pass
    else:
        raise AssertionError("failed preflight was accepted")
    assert heldout_calls == 0
    print(json.dumps({"ok": True, "selftest": True, "router_records": ROUTER_RECORDS,
                      "heldout_rows": EXPECTED_HELDOUT_DOCUMENTS}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selftest", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--supervisor-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--candidate-scores", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            _selftest()
            return 0
        if args.preflight:
            report, metadata = _metadata_preflight(args.snapshot, args.output_dir)
            print(json.dumps({"ok": True, "status": "PREFLIGHT_COMPLETE", "snapshot": str(report.snapshot),
                              "preflight": metadata}, sort_keys=True))
            return 0
        if args.worker:
            if (args.output_dir is None or args.supervisor_manifest is None or args.worker_result is None or
                    args.candidate_scores is None or args.snapshot is None or not args.worker_token or
                    not secrets.compare_digest(args.worker_token, os.environ.get(WORKER_TOKEN_ENV, "")) or
                    args.supervisor_manifest.resolve().parent != args.output_dir.resolve() or
                    args.worker_result.resolve().parent != args.output_dir.resolve() or
                    args.candidate_scores.resolve().parent != args.output_dir.resolve()):
                raise ApparatusError("internal worker token/paths invalid")
            return _worker(args)
        if args.output_dir is None:
            raise ApparatusError("--run requires an explicit fresh --output-dir")
        return _run_parent(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "VOID_RESOURCE" if isinstance(exc, MemoryError) else "VOID_APPARATUS",
                          "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
