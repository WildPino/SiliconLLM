"""Pre-inference excerpt answerability record for the frozen METH-90 set."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SHA = "70d48b30de55c41283374223ba2b61edaacad396e6f3410c45c166bbaa885937"
ANCHORS = (
    'checkpoint["updates"] == 1024',
    "NVIDIA GeForce RTX 3060",
    "parent_supervision_error",
    "METH-14 15-minute CPU stop",
    "self.slice_refs",
    "down_proj`'s INPUT",
    "gate_as_up",
    "_mm256_fmadd_ps",
    "Nightingale",
    "vig'rous exercise",
    "Illmarinen",
    "steam-engine",
    "Vassar",
    "old Roman",
    "Lisette",
    "Matisse",
    "POSTHOC_TRANSFER_SIGNAL=false",
    "nn.Linear",
    "2.5× slower per weight",
    "6,474,702,976 bytes",
    "COACT − D0C",
    "G-H1",
    "real_H",
    "ATTENTION-BOUND",
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
    report = {"experiment": "METH-90-pre-inference-answerability",
              "manifest_sha256": MANIFEST_SHA, "rows": rows,
              "answerable_count": len(rows),
              "scope": "excerpt-only anchors fixed before donor or adapted-model inference"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"answerable_count": len(rows),
                      "sha256": sha(args.out.read_bytes())}))


if __name__ == "__main__":
    main()
