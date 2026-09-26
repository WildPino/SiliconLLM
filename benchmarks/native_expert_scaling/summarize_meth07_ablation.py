#!/usr/bin/env python3
"""Apply the frozen METH-07 organ-rescue decision to paired pilot reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adjudicate_meth05_pilot import CATEGORIES, sha256


RESCUE_FLOOR = 0.120


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "meth05_gigachat_pilot_adjudication_v1":
        raise ValueError(f"unexpected pilot report schema: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mla", type=Path, required=True)
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    inputs = {name: path for name, path in (("baseline", args.baseline), ("mla", args.mla),
                                           ("experts", args.experts))}
    reports = {name: load(path) for name, path in inputs.items()}
    baseline = reports["baseline"]
    for name, report in reports.items():
        if (report["selection_manifest_sha256"] != baseline["selection_manifest_sha256"]
                or report["score_sha256"]["bf16"] != baseline["score_sha256"]["bf16"]
                or report["score_sha256"]["q4"] != baseline["score_sha256"]["q4"]):
            raise ValueError(f"{name} does not share the frozen references")
        for group in ("all", *CATEGORIES):
            reference = baseline["groups"][group]
            arm = report["groups"][group]
            if (arm["documents"], arm["bytes"], arm["tokens"]) != (
                    reference["documents"], reference["bytes"], reference["tokens"]):
                raise ValueError(f"{name} has mismatched paired denominator in {group}")
            for control in ("bf16", "q4"):
                if arm["bpb"][control] != reference["bpb"][control]:
                    raise ValueError(f"{name} changed {control} BPB in {group}")
    initial = baseline["groups"]["all"]["candidate_minus_bf16"]
    result = {"schema": "meth07_gigachat_organ_rescue_v1",
              "frozen_rescue_floor_bpb": RESCUE_FLOOR,
              "baseline_iq2_delta_bpb": initial,
              "input_sha256": {name: sha256(path) for name, path in inputs.items()},
              "arms": {}}
    for name in ("mla", "experts"):
        report = reports[name]
        delta = report["groups"]["all"]["candidate_minus_bf16"]
        rescue = initial - delta
        result["arms"][name] = {
            "delta_bpb": delta,
            "rescue_bpb": rescue,
            "meets_rescue_floor": rescue >= RESCUE_FLOOR,
            "category_delta_bpb": {group: report["groups"][group]["candidate_minus_bf16"]
                                   for group in CATEGORIES},
        }
    result["priority_organs"] = [name for name in ("mla", "experts")
                                  if result["arms"][name]["meets_rescue_floor"]]
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
