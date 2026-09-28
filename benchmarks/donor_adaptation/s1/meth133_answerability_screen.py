"""Freeze excerpt-contained answer details before METH-133 inference."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SHA = "ec6f839a2c9b1803dc8d5658aafa8fd9191b59fa2607961850f5a8666bc35674"
ANCHORS = (
    "max_position_embeddings=128",
    "teacher-forced count E22 published",
    '"--git-rev"',
    "recovery_minus_wrong_layer_null",
    "record shape must contain positive integer dimensions",
    'self.assertEqual(summary["documents"], 96)',
    "mapped target identity mismatch",
    "H4 self-test record exists",
    "Tanucci called on the king",
    "La Bible Amusante",
    "long-nosed man was gone",
    "dropped the sugar-tongs",
    "mouse's nest, with five young ones",
    "Bishop's Eye, Wells, 403",
    "Lucy gave a little scream",
    "They abstained from wine",
    "LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT",
    "all 32 current layer-1 checkpoints pass",
    "two real bugs in the exporter",
    "0.25 B/element",
    "30.75 vs 89.83 tok/s",
    "works at 1.5 B and fails at 0.5 B",
    "11.97 tok/s",
    "not an executable router",
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
        rows.append({"source_id": item["source_id"], "category": item["category"],
                     "anchor": anchor, "answerable": True,
                     "basis": "The visible excerpt contains this specific detail."})
    result = {"experiment": "METH-133-pre-inference-answerability",
              "manifest_sha256": MANIFEST_SHA, "answerable_count": len(rows),
              "rows": rows,
              "scope": "Excerpt-only details frozen before donor or adapted-model inference"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"answerable_count": len(rows), "sha256": sha(args.out.read_bytes())}))


if __name__ == "__main__":
    main()
