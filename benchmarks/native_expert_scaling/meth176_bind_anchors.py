#!/usr/bin/env python3
"""Freeze excerpt-visible grounding anchors before METH-176 inference."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    assert digest(args.manifest) == args.manifest_sha
    assert digest(args.spec) == args.spec_sha
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    assert manifest["experiment"] == "METH-176-matched-long-fresh-external-manifest"
    assert spec["manifest_sha256"] == args.manifest_sha
    items, anchors = manifest["items"], spec["anchors_in_manifest_order"]
    assert len(items) == len(anchors) == 24
    rows = []
    for item, frozen in zip(items, anchors):
        assert frozen["source_id"] == item["source_id"]
        assert item["answerability_screen"]["passed"] is True
        assert item["excerpt"] == item["text"][:384]
        anchor = frozen["anchor"]
        assert isinstance(anchor, str) and len(anchor) >= 6
        assert anchor in item["excerpt"], item["source_id"]
        rows.append({"source_id": item["source_id"],
                     "category": item["category"],
                     "anchor": anchor, "answerable": True,
                     "basis": "The visible excerpt contains this specific detail."})
    result = {"experiment": "METH-176-pre-inference-answerability",
              "manifest_sha256": args.manifest_sha,
              "anchor_spec_sha256": args.spec_sha,
              "answerable_count": len(rows), "rows": rows,
              "scope": "Excerpt-only details frozen before any donor or adapted-model inference"}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"answerable_count": len(rows),
                      "sha256": digest(args.out)}), flush=True)


if __name__ == "__main__":
    main()
