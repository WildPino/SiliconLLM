#!/usr/bin/env python3
"""Bind manually selected visible details before any METH-214 inference."""
import argparse
import json
from pathlib import Path

import meth213_stored_core_fresh_manifest as F

ANCHORS = (
    "P1S2CSV,", "F16 conversion output write failure",
    "run_strat01_engine_layer2_depth_extension", "CandidateResourceError(MemoryError)",
    "a->embd[7U*1536U+i]", "baseline_bpb", "M59.CORE_SHA", "CC0-1.0",
    "Humfrey will meet us anon", "the flies shall keep ahead", "unkind to Tim",
    "a young elephant", "Darwinism is a scientific theory", "the new electric meters",
    "Assistant\nCounty  Treasurer for two years", "about fifty yards from the edge of the water",
    "looser than ~25%", "16 matrici router F32", "gruppi contigui di 128",
    "at least 50 accepted end-to-end tok/s", "6,474,702,976 bytes", "128 experts, top-8",
    "L = 19", "standby/page cache",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--manifest-sha", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists() and F.M.digest(args.manifest) == args.manifest_sha
    items = json.loads(args.manifest.read_text(encoding="utf-8"))["items"]
    assert len(items) == len(ANCHORS) == 24
    rows = []
    for item, anchor in zip(items, ANCHORS):
        assert item["answerability_screen"]["passed"] and item["excerpt"] == item["text"][:384]
        assert len(anchor)>=6 and anchor in item["excerpt"], item["source_id"]
        rows.append({"source_id":item["source_id"],"category":item["category"],
            "anchor":anchor,"answerable":True,
            "basis":"Specific detail directly present in the visible 384-character excerpt."})
    result = {"experiment":"METH-213-pre-inference-answerability", "manifest_sha256":args.manifest_sha,
        "answerable_count":24,"rows":rows,
        "scope":"Manually selected excerpt-only details; no model outputs viewed"}
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"sha256":F.M.digest(args.out),"answerable_count":24}),flush=True)


if __name__ == "__main__":
    main()
