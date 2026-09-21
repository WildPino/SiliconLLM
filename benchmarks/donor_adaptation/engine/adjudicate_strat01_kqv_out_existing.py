#!/usr/bin/env python3
"""Offline re-adjudication of immutable kqv_out diagnostic products."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine.run_strat01_kqv_out_diagnostic import (
    COUNT_OUTPUT,
    OLD_MAX,
    OLD_NRMSE,
    PAYLOADS,
    SOURCE_RUN,
    TIGHT_MAX,
    TIGHT_NRMSE,
    classify,
    gate,
    load_f32,
    manifest_payload,
)
from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_repair import sha256_file

HERE = Path(__file__).resolve().parent
SOURCE_RAW = HERE / "results/strat01_gigachat_engine_kqv_out_diagnostic_20260921"
SOURCE_ADJUDICATION_SHA = "859b59d36634ded8e482a87fa1674417cbbd1c70f035f9a03ebe1dc2e1df9e55"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_kqv_out_offline_adjudication_20260921"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit("output directory already exists; raw evidence is immutable")
    output.mkdir(parents=True)
    errors: list[str] = []
    status = "VOID_KQV_OUT_DIAGNOSTIC"
    record: dict[str, object] = {}
    try:
        source_path = SOURCE_RAW / "adjudication.json"
        if sha256_file(source_path) != SOURCE_ADJUDICATION_SHA:
            raise ValueError("source diagnostic adjudication identity mismatch")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("errors") or source.get("donor_executions") != 0:
            raise ValueError("source diagnostic apparatus was not valid and offline")
        products = SOURCE_RAW / "products"
        for name, identity in source["outputs"].items():
            path = products / name
            if not path.is_file() or path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
                raise ValueError(f"source product identity mismatch: {name}")
        _, target_count, target_sha = PAYLOADS["target"]
        target_path = manifest_payload("prefill8", PAYLOADS["target"][0])
        target = load_f32(target_path, target_count, target_sha)
        produced = {
            name: load_f32(products / filename, COUNT_OUTPUT)
            for name, filename in {
                "d32": "d32.f32le", "project": "project_q8.f32le", "pinned": "pinned_q8.f32le",
                "mutated": "mutated_q8.f32le", "transposed": "transposed_head.f32le",
                "wrong_scales": "wrong_scales.f32le",
            }.items()
        }
        results = {
            "project_vs_pinned": gate(produced["project"], produced["pinned"], TIGHT_NRMSE, TIGHT_MAX),
            "project_vs_target": gate(produced["project"], target, TIGHT_NRMSE, TIGHT_MAX),
            "pinned_vs_target": gate(produced["pinned"], target, TIGHT_NRMSE, TIGHT_MAX),
            "d32_vs_target": gate(produced["d32"], target, OLD_NRMSE, OLD_MAX),
            "mutated_vs_target": gate(produced["mutated"], target, TIGHT_NRMSE, TIGHT_MAX),
            "transposed_vs_target": gate(produced["transposed"], target, TIGHT_NRMSE, TIGHT_MAX),
            "wrong_scales_vs_target": gate(produced["wrong_scales"], target, TIGHT_NRMSE, TIGHT_MAX),
        }
        controls = dict(source["controls"])
        status = classify(results, controls)
        record = {
            "schema": "strat01_kqv_out_offline_adjudication_v1",
            "status": status,
            "started_utc": utc_now(),
            "finished_utc": utc_now(),
            "donor_executions": 0,
            "errors": [],
            "source_raw": str(SOURCE_RAW),
            "source_adjudication_sha256": SOURCE_ADJUDICATION_SHA,
            "source_reported_status": source["status"],
            "repair": "classification-only: implement the frozen material-improvement branch",
            "controls": controls,
            "results": results,
            "source_identity": source["identity"],
            "source_outputs": source["outputs"],
            "non_claims": source["non_claims"],
        }
    except Exception as error:
        errors.append(str(error))
        record = {
            "schema": "strat01_kqv_out_offline_adjudication_v1",
            "status": status,
            "started_utc": utc_now(),
            "finished_utc": utc_now(),
            "donor_executions": 0,
            "errors": errors,
            "source_raw": str(SOURCE_RAW),
            "source_adjudication_sha256": SOURCE_ADJUDICATION_SHA,
        }
    (output / "adjudication.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(status)
    if errors:
        print(errors[0], file=sys.stderr)
    return 0 if status != "VOID_KQV_OUT_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
