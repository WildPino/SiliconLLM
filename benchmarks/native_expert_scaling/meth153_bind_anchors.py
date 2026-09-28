#!/usr/bin/env python3
"""Bind model-free excerpt details to the frozen METH-153 sources."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    assert digest(args.manifest) == spec["manifest_sha256"]
    items, anchors = manifest["items"], spec["anchors_in_manifest_order"]
    assert len(items) == len(anchors) == 24
    rows = []
    for item, anchor in zip(items, anchors):
        assert isinstance(anchor, str) and len(anchor) >= 6
        assert anchor in item["excerpt"], item["source_id"]
        rows.append({"source_id": item["source_id"], "anchor": anchor})
    result = {"experiment": "METH-153-model-free-excerpt-anchors",
              "manifest_sha256": spec["manifest_sha256"],
              "anchor_spec_sha256": digest(args.spec), "rows": rows}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"anchors": len(rows), "sha256": digest(args.out)}))


if __name__ == "__main__":
    main()
