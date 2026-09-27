"""Record pre-inference answerability anchors for the frozen 24 excerpts."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


MANIFEST_SHA = "9571f5d61c29e34b27c05ec531232a08b9eb6cd81bdf85aa5648ff74f1db3d75"
ANCHORS = (
    "ternarizes both factors",
    "MANIFEST.json",
    "register_forward_hook(post)",
    "rms_in",
    "GradScaler",
    "check_anchor",
    "lr=1e-3",
    "git",
    "meridian",
    "Sam and Peter",
    "forced",
    "Uncle David",
    "Scotland",
    "32-pounder Rocket",
    "dry dock",
    "skates",
    "k = 133",
    "G-E41D",
    "qwen_export.py",
    "49.96",
    "12.4 MB",
    "d0_coactivation.json",
    "O(N d³)",
    "FFN co-activation",
)


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
        excerpt = item["text"][:384]
        assert anchor in excerpt, (item["source_id"], anchor)
        rows.append({"source_id": item["source_id"], "category": item["category"],
                     "anchor": anchor, "answerable": True,
                     "basis": "The excerpt contains a concrete fact or action that can serve as a specific detail."})
    report = {"experiment": "METH-57-pre-inference-answerability",
              "manifest_sha256": MANIFEST_SHA, "rows": rows,
              "answerable_count": len(rows),
              "scope": "human-selected textual anchors; no donor or student continuations viewed"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"answerable_count": len(rows), "sha256": sha(args.out.read_bytes())}))


if __name__ == "__main__":
    main()
