"""Pre-inference excerpt answerability record for the frozen METH-62 set."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SHA = "56237bcf2e83ea2e743ac0db2475bff299deed988b7cb75cf31cebcad8b4b2aa"
ANCHORS = (
    "hard_gate",
    "one fp32 scale per output row",
    "repeated_8gram_3x",
    "missing recovery source(s)",
    "payload identity mismatch",
    "RAW_SHA",
    "q8_census",
    "MK_FACTORED",
    "The Witts",
    "Sprite",
    "prayer book",
    "Palace of the Cæsars",
    "Sun Dance Trail",
    "General Custer",
    "spare room",
    "Mr. Raymond",
    "F16×F16",
    "23.5 → 38.0 tok/s",
    "MLA compressed KV path is absent",
    "+0.278096",
    "21,069,824 B",
    "224 experts",
    "Fixed row-column pairing",
    "28 × 5",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert sha(args.manifest.read_bytes()) == MANIFEST_SHA
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert len(manifest["items"]) == len(ANCHORS) == 24
    rows = []
    for item, anchor in zip(manifest["items"], ANCHORS):
        assert item["excerpt"] == item["text"][:384]
        assert anchor in item["excerpt"], (item["source_id"], anchor)
        rows.append({"source_id": item["source_id"],
                     "category": item["category"], "anchor": anchor,
                     "answerable": True,
                     "basis": "The excerpt contains a specific action, fact or number for the requested detail."})
    report = {"experiment": "METH-62-pre-inference-answerability",
              "manifest_sha256": MANIFEST_SHA, "rows": rows,
              "answerable_count": len(rows),
              "scope": "excerpt-only anchors fixed before donor or mixed-model inference"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"answerable_count": len(rows),
                      "sha256": sha(args.out.read_bytes())}))


if __name__ == "__main__":
    main()
