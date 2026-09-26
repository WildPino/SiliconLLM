#!/usr/bin/env python3
"""Apply the frozen METH-09 Q2_K expert-only diagnostic decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adjudicate_meth05_pilot import CATEGORIES, sha256


EFFECT_FLOOR = 0.020


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "meth05_gigachat_pilot_adjudication_v1":
        raise ValueError(f"unexpected pilot report schema: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iq2", type=Path, required=True)
    parser.add_argument("--q2-reallocation", type=Path, required=True)
    parser.add_argument("--q2-experts-only", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    paths = {"iq2": args.iq2, "q2_reallocation": args.q2_reallocation,
             "q2_experts_only": args.q2_experts_only}
    reports = {name: load(path) for name, path in paths.items()}
    ref = reports["iq2"]
    for name, report in reports.items():
        if report["selection_manifest_sha256"] != ref["selection_manifest_sha256"]:
            raise ValueError(f"{name} selection differs")
        for control in ("bf16", "q4"):
            if report["score_sha256"][control] != ref["score_sha256"][control]:
                raise ValueError(f"{name} {control} reference differs")
        for group in ("all", *CATEGORIES):
            a, b = report["groups"][group], ref["groups"][group]
            if (a["documents"], a["bytes"], a["tokens"]) != (b["documents"], b["bytes"], b["tokens"]):
                raise ValueError(f"{name} {group} denominator differs")
    deltas = {name: report["groups"]["all"]["candidate_minus_bf16"]
              for name, report in reports.items()}
    expert_effect = deltas["q2_experts_only"] - deltas["iq2"]
    payment_effect = deltas["q2_reallocation"] - deltas["q2_experts_only"]
    classification = ("Q2_EXPERT_IMPROVES_PILOT" if expert_effect <= -EFFECT_FLOOR
                      else "Q2_EXPERT_HARMS_PILOT" if expert_effect >= EFFECT_FLOOR
                      else "Q2_EXPERT_EFFECT_INCONCLUSIVE")
    report = {
        "schema": "meth09_gigachat_q2_expert_isolation_v1",
        "frozen_expert_effect_floor_bpb": EFFECT_FLOOR,
        "classification": classification,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "pooled_delta_bpb": deltas,
        "q2_expert_only_minus_iq2_bpb": expert_effect,
        "head_dense_payment_given_q2_experts_bpb": payment_effect,
        "category_delta_bpb": {group: {name: item["groups"][group]["candidate_minus_bf16"]
                                       for name, item in reports.items()}
                               for group in CATEGORIES},
    }
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
