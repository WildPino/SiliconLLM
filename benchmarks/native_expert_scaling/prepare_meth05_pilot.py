"""Freeze a nine-document GigaChat heldout gross-failure screen by ID only."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


HELDOUT_SHA256 = "04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e"
CATEGORIES = ("code", "prose", "technical_general")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(heldout: Path, output: Path) -> dict:
    if sha256(heldout) != HELDOUT_SHA256:
        raise ValueError("heldout SHA-256 mismatch")
    manifest_path = output.with_suffix(output.suffix + ".selection.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite pilot or selection record")
    rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]
    counts = Counter(row["category"] for row in rows)
    if counts != Counter({category: 32 for category in CATEGORIES}):
        raise ValueError(f"unexpected heldout category inventory: {counts}")
    selected = []
    for category in CATEGORIES:
        category_rows = sorted((row for row in rows if row["category"] == category),
                               key=lambda row: row["source_document_id"])
        selected.extend(category_rows[:3])
    if len({row["source_document_id"] for row in selected}) != 9:
        raise ValueError("pilot IDs are not unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    record = {
        "schema": "meth05_gigachat_pilot_selection_v1",
        "source_heldout": str(heldout),
        "source_heldout_sha256": HELDOUT_SHA256,
        "selection_rule": "first_three_lexicographic_source_document_id_per_category",
        "categories": list(CATEGORIES),
        "output": str(output),
        "output_sha256": sha256(output),
        "selected_ids": [row["source_document_id"] for row in selected],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.heldout, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
