#!/usr/bin/env python3
"""Bind one excerpt-contained specific detail per fresh quality source."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--anchors-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    assert digest(args.manifest) == args.manifest_sha
    assert digest(args.anchors) == args.anchors_sha
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    anchors = json.loads(args.anchors.read_text(encoding="utf-8"))
    assert len(manifest["items"]) == len(anchors["rows"]) == 24
    rows = []
    for item, frozen in zip(manifest["items"], anchors["rows"]):
        assert frozen["source_id"] == item["source_id"]
        assert item["excerpt"] == item["text"][:384]
        anchor = frozen["anchor"]
        assert isinstance(anchor, str) and len(anchor) >= 6
        assert anchor in item["excerpt"], (item["source_id"], anchor)
        assert item["answerability_screen"]["passed"] is True
        rows.append({"source_id": item["source_id"],
                     "category": item["category"],
                     "anchor": anchor, "answerable": True,
                     "basis": "The visible excerpt contains this specific detail."})
    result = {"experiment": "METH-153-pre-inference-answerability",
              "manifest_sha256": args.manifest_sha,
              "anchors_sha256": args.anchors_sha,
              "answerable_count": len(rows),
              "rows": rows,
              "scope": "Excerpt-only details frozen before donor or adapted-model inference"}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"answerable_count": len(rows),
                      "sha256": digest(args.out)}), flush=True)


if __name__ == "__main__":
    main()
