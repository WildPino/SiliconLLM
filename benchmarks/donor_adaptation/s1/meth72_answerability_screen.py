"""Pre-inference excerpt answerability record for the frozen METH-72 set."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SHA = "11621dce9638aae01cc3bd8c76730de7676e4bbb704271056f164b9ad4d5b437"
ANCHORS = (
    "R3H",
    "read-only stream",
    "max_relative_l2_error",
    "normalization sources differ from HEAD",
    "bytes_per_token",
    "masked_objective",
    "continuation_ids",
    "shard_01.payload",
    "salt water",
    "mosquitoes",
    "Donelson",
    "CHAPTER VI",
    "Clergyman's lady",
    "Hongkong",
    "tunnel",
    "Tracy",
    "qwen_export.py",
    "--rule R3",
    "2.642e-06",
    "35.37 ms",
    "LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT",
    "30.46 GB",
    "f52e0e5b",
    "SSM layer",
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
    report = {"experiment": "METH-72-pre-inference-answerability",
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
