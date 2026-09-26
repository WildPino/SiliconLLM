"""Adjudicate the preregistered nine-document GigaChat gross-failure screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CATEGORIES = ("code", "prose", "technical_general")
GROSS_FAILURE_DELTA = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scores(path: Path, selected: set[str]) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["source_document_id"]: row for row in records}
    if len(records) != 9 or set(by_id) != selected:
        raise ValueError(f"score IDs differ from frozen nine-document selection: {path}")
    for row in records:
        if row["category"] not in CATEGORIES or type(row["tokens"]) is not int or row["tokens"] <= 0:
            raise ValueError(f"invalid category/token count in {path}")
        if type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError(f"invalid byte count in {path}")
        if not math.isfinite(float(row["bits"])) or float(row["bits"]) < 0:
            raise ValueError(f"invalid NLL bits in {path}")
    return by_id


def adjudicate(selection: Path, bf16: Path, q4: Path, candidate: Path) -> dict:
    choice = json.loads(selection.read_text(encoding="utf-8"))
    selected = set(choice["selected_ids"])
    if len(selected) != 9 or choice["selection_rule"] != "first_three_lexicographic_source_document_id_per_category":
        raise ValueError("invalid selection manifest")
    arms = {name: read_scores(path, selected) for name, path in
            (("bf16", bf16), ("q4", q4), ("candidate", candidate))}
    for item_id in selected:
        base = arms["bf16"][item_id]
        for name in ("q4", "candidate"):
            row = arms[name][item_id]
            if any(row[key] != base[key] for key in ("category", "tokens", "bytes")):
                raise ValueError(f"paired score metadata differs at {item_id} in {name}")

    def group_report(ids: set[str]) -> dict:
        total_bytes = sum(arms["bf16"][item_id]["bytes"] for item_id in ids)
        bpb = {name: sum(arm[item_id]["bits"] for item_id in ids) / total_bytes
               for name, arm in arms.items()}
        return {"documents": len(ids), "bytes": total_bytes,
                "tokens": sum(arms["bf16"][item_id]["tokens"] for item_id in ids),
                "bpb": bpb, "q4_minus_bf16": bpb["q4"] - bpb["bf16"],
                "candidate_minus_bf16": bpb["candidate"] - bpb["bf16"]}

    groups = {"all": group_report(selected)}
    for category in CATEGORIES:
        ids = {item_id for item_id in selected if arms["bf16"][item_id]["category"] == category}
        if len(ids) != 3:
            raise ValueError(f"category {category} has {len(ids)} instead of 3 scores")
        groups[category] = group_report(ids)
    gross_failure = all(groups[key]["candidate_minus_bf16"] > GROSS_FAILURE_DELTA
                        for key in ("all", *CATEGORIES))
    return {
        "schema": "meth05_gigachat_pilot_adjudication_v1",
        "status": "STOP_GROSS_QUALITY_FAILURE" if gross_failure else "CONTINUE_TO_FULL_HELDOUT",
        "nine_document_screen_cannot_pass_quality": True,
        "gross_failure_delta_bpb": GROSS_FAILURE_DELTA,
        "selection_manifest_sha256": sha256(selection),
        "score_sha256": {"bf16": sha256(bf16), "q4": sha256(q4), "candidate": sha256(candidate)},
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = adjudicate(args.selection, args.bf16, args.q4, args.candidate)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
