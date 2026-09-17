#!/usr/bin/env python3
"""Read-only auditor for a completed STRAT-02 teacher baseline run.

The auditor binds the persisted score JSONL to the frozen corpus and pinned
local tokenizer, then reconciles every reported aggregate and monitor field.
It never loads, opens, or hashes teacher weights and never performs a model
forward.  A run directory is required for normal operation::

    python strat02_teacher_baseline_audit.py --run-dir PATH

Use ``--selftest`` for a tiny parser-only corruption/missing-artifact test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus" / "strat02_document_holdout_v1"
EXPECTED_DOCUMENTS = 96
EXPECTED_HELDOUT_TOKEN_IDS_SHA256 = (
    "5a4cf31ec7db0f3bc541f2e75a78b133a50cd3d6174abb26ef81e08c45e65289"
)
REVISION = "d2a4949c9d4ad6cf47fbac131f7e020077332b21"
TRANSFORMERS_VERSION = "4.57.1"
TOKENIZERS_VERSION = "0.22.2"


def _load_local_module(filename: str, module_name: str) -> Any:
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load local module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing artifact: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact {path.name} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash {path.name}: {exc}") from exc
    return digest.hexdigest()


def _close_float(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label} is not numeric")
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise ValueError(f"supervisor {label} is not numeric")
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} differs from supervisor ({actual!r} != {expected!r})")


def _aggregate(scores: Mapping[str, Any]) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for item in scores.values():
        category = item.category
        summary = categories.setdefault(category, {"documents": 0, "tokens": 0, "bytes": 0, "bits": 0.0})
        summary["documents"] += 1
        summary["tokens"] += item.tokens
        summary["bytes"] += item.bytes
        summary["bits"] += item.bits
    for summary in categories.values():
        summary["bpb"] = summary["bits"] / summary["bytes"]
    tokens = sum(item.tokens for item in scores.values())
    bytes_ = sum(item.bytes for item in scores.values())
    bits = sum(item.bits for item in scores.values())
    return {
        "documents": len(scores), "tokens": tokens, "bytes": bytes_, "bits": bits,
        "bpb": bits / bytes_, "categories": categories,
    }


def _compare_aggregate(observed: Mapping[str, Any], supervisor: Mapping[str, Any]) -> None:
    for field in ("documents", "tokens", "bytes"):
        if observed.get(field) != supervisor.get(field):
            raise ValueError(f"aggregate {field} differs from supervisor")
    _close_float(observed.get("bits"), supervisor.get("bits"), "aggregate bits")
    _close_float(observed.get("bpb"), supervisor.get("bpb"), "aggregate BPB")
    observed_categories = observed.get("categories")
    supervisor_categories = supervisor.get("categories")
    if not isinstance(observed_categories, dict) or not isinstance(supervisor_categories, dict):
        raise ValueError("aggregate categories are missing or malformed")
    if set(observed_categories) != set(supervisor_categories):
        raise ValueError("aggregate category set differs from supervisor")
    for category, actual in observed_categories.items():
        expected = supervisor_categories[category]
        for field in ("documents", "tokens", "bytes"):
            if actual.get(field) != expected.get(field):
                raise ValueError(f"aggregate {category}.{field} differs from supervisor")
        for field in ("bits", "bpb"):
            _close_float(actual.get(field), expected.get(field), f"aggregate {category}.{field}")


def _monitor_report(log_path: Path, supervisor: Mapping[str, Any]) -> dict[str, Any]:
    if not log_path.is_file():
        raise ValueError("missing artifact: supervisor_log.jsonl")
    samples: list[dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"supervisor_log.jsonl:{line_number}: blank line")
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"supervisor_log.jsonl:{line_number}: invalid JSON") from exc
                if isinstance(item, dict) and item.get("event") == "resource_sample":
                    samples.append(item)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read supervisor_log.jsonl: {exc}") from exc
    if not samples:
        raise ValueError("monitor has no resource samples")
    reported = supervisor.get("monitor", {}).get("samples")
    if reported != len(samples):
        raise ValueError(f"monitor sample count differs ({len(samples)} != {reported})")
    required = (
        "child_working_set_bytes", "child_private_commit_bytes",
        "child_peak_pagefile_bytes", "available_physical_ram_bytes",
    )
    for sample in samples:
        for field in required:
            value = sample.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"monitor sample has invalid {field}")
    return {
        "samples": len(samples),
        "max_working_set_bytes": max(s["child_working_set_bytes"] for s in samples),
        "max_private_commit_bytes": max(s["child_private_commit_bytes"] for s in samples),
        "max_peak_pagefile_bytes": max(s["child_peak_pagefile_bytes"] for s in samples),
        "min_available_physical_ram_bytes": min(s["available_physical_ram_bytes"] for s in samples),
    }


def audit_run(run_dir: Path) -> dict[str, Any]:
    """Audit one completed run; all operations are read-only."""
    run_dir = run_dir.resolve()
    result_path = run_dir / "supervisor_result.json"
    run_manifest_path = run_dir / "supervisor_manifest.json"
    worker_path = run_dir / "worker_result.json"
    scores_path = run_dir / "teacher_scores.jsonl"
    log_path = run_dir / "supervisor_log.jsonl"
    supervisor = _read_object(result_path)
    run_manifest = _read_object(run_manifest_path)
    worker = _read_object(worker_path)
    if supervisor.get("worker_result") != worker:
        raise ValueError("supervisor embedded worker result differs from worker_result.json")
    score = _load_local_module("strat02_score.py", "_strat02_teacher_audit_score")
    token_audit = _load_local_module("strat02_token_audit.py", "_strat02_teacher_audit_tokens")
    manifest_path = CORPUS / "manifest.json"
    calib_path = CORPUS / "calib.jsonl"
    heldout_path = CORPUS / "heldout.jsonl"
    if run_manifest.get("schema") != "strat02_teacher_baseline_v1":
        raise ValueError("unexpected supervisor manifest schema")
    pins = run_manifest.get("pins")
    if not isinstance(pins, dict) or (
        pins.get("manifest_sha256") != token_audit.EXPECTED_MANIFEST_SHA256
        or pins.get("heldout_ordered_token_ids_sha256") != EXPECTED_HELDOUT_TOKEN_IDS_SHA256
        or pins.get("heldout_jsonl_sha256") != hashlib.sha256(heldout_path.read_bytes()).hexdigest()
    ):
        raise ValueError("supervisor manifest corpus/tokenizer pins differ")
    preflight = run_manifest.get("parent_preflight")
    if not isinstance(preflight, dict):
        raise ValueError("missing parent preflight in supervisor manifest")
    metadata = preflight.get("metadata")
    if not isinstance(metadata, dict) or (
        metadata.get("revision") != REVISION
        or metadata.get("repository") != token_audit.DONOR
    ):
        raise ValueError("supervisor manifest donor identity differs")
    snapshot_value = preflight.get("snapshot")
    if not isinstance(snapshot_value, str):
        raise ValueError("supervisor manifest snapshot path is missing")
    snapshot = Path(snapshot_value)
    if not snapshot.is_absolute() or snapshot.name != REVISION or not snapshot.is_dir():
        raise ValueError(f"pinned local tokenizer snapshot is missing: {snapshot}")
    errors: list[str] = []
    manifest, rows_by_split = token_audit.check_corpus(
        manifest_path, {"calib": calib_path, "heldout": heldout_path}, errors
    )
    token_report = token_audit.check_tokens(snapshot, rows_by_split, errors)
    if errors:
        raise ValueError("frozen corpus/tokenizer audit failed: " + "; ".join(errors))
    heldout_rows = rows_by_split.get("heldout")
    if not isinstance(heldout_rows, list) or len(heldout_rows) != EXPECTED_DOCUMENTS:
        raise ValueError("frozen heldout corpus does not contain exactly 96 rows")
    heldout_digest = token_report.get("splits", {}).get("heldout", {}).get("ordered_token_ids_sha256")
    if heldout_digest != EXPECTED_HELDOUT_TOKEN_IDS_SHA256:
        raise ValueError("heldout ordered token-ID digest differs from the pin")
    tokenizer, _, transformers_version, tokenizers_version = token_audit.load_tokenizer(snapshot)
    if (transformers_version, tokenizers_version) != (TRANSFORMERS_VERSION, TOKENIZERS_VERSION):
        raise ValueError("pinned tokenizer runtime versions are unavailable")
    score.validate_strat02_tokenizer(tokenizer)
    scores = score.read_score_jsonl(scores_path)
    if len(scores) != EXPECTED_DOCUMENTS:
        raise ValueError(f"persisted score rows: {len(scores)} != {EXPECTED_DOCUMENTS}")
    score.validate_scores_against_heldout(scores, heldout_rows, tokenizer, arm="teacher")
    observed_aggregate = _aggregate(scores)
    worker_aggregate = worker.get("aggregate")
    if not isinstance(worker_aggregate, dict):
        raise ValueError("worker aggregate is missing")
    _compare_aggregate(observed_aggregate, worker_aggregate)
    _compare_aggregate(observed_aggregate, supervisor.get("worker_result", {}).get("aggregate", {}))
    if supervisor.get("ok") is not True or supervisor.get("status") != "COMPLETE":
        raise ValueError("supervisor is not ok/COMPLETE")
    if worker.get("ok") is not True or worker.get("status") != "COMPLETE":
        raise ValueError("worker is not ok/COMPLETE")
    monitor = supervisor.get("monitor")
    if not isinstance(monitor, dict) or monitor.get("exit_code") != 0:
        raise ValueError("child did not exit with code 0")
    if supervisor.get("failure_classification") is not None:
        raise ValueError("supervisor has a failure classification")
    if supervisor.get("partial_scores_persisted") != EXPECTED_DOCUMENTS:
        raise ValueError("supervisor partial score count is not 96")
    observed_sha = _sha256_file(scores_path)
    if observed_sha != supervisor.get("scores_sha256"):
        raise ValueError("teacher score SHA-256 differs from supervisor")
    monitor_observed = _monitor_report(log_path, supervisor)
    return {
        "pass": True,
        "run_dir": str(run_dir),
        "rows": EXPECTED_DOCUMENTS,
        "heldout_token_ids_sha256": heldout_digest,
        "scores_sha256": observed_sha,
        "aggregate": observed_aggregate,
        "monitor": monitor_observed,
        "runtime": {"transformers": transformers_version, "tokenizers": tokenizers_version},
    }


def _selftest() -> None:
    score = _load_local_module("strat02_score.py", "_strat02_teacher_audit_selftest_score")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        missing = root / "missing.jsonl"
        try:
            score.read_score_jsonl(missing)
        except ValueError:
            pass
        else:
            raise AssertionError("missing score artifact was accepted")
        corrupt = root / "corrupt.jsonl"
        corrupt.write_text('{"source_document_id": "x"}\n', encoding="utf-8")
        try:
            score.read_score_jsonl(corrupt)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt score artifact was accepted")
        log = root / "supervisor_log.jsonl"
        log.write_text(json.dumps({"event": "resource_sample"}) + "\n", encoding="utf-8")
        try:
            _monitor_report(log, {"monitor": {"samples": 1}})
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete monitor sample was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    try:
        if args.selftest:
            _selftest()
            print(json.dumps({"pass": True, "selftest": True}, separators=(",", ":")))
            return 0
        if args.run_dir is None:
            raise ValueError("--run-dir is required")
        report = audit_run(args.run_dir)
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"pass": False, "error": str(exc)}, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
