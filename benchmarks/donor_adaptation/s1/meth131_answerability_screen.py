"""Freeze excerpt-contained details before METH-131 model inference."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SHA = "84d8dbb400be133b6d1e440d55da5d09afd8ebbe0bd659474bdad8e708868ba9"
ANCHORS = (
    'attn_implementation="sdpa"',
    "historical Stage-A output identity mismatch",
    'if self.mode == "off"',
    "M25.CORE_SHA",
    "TIGHT_LIMITS",
    "e37_routers_E256.npz",
    "strat01_gguf_rung1.h",
    "p._handle",
    "exactly on your meridian",
    "Isolde had told the story of her adventure to Valerie",
    "six weeks, and not a penny of wages yet",
    '"She say her back broke," Eugene',
    "Aquitaine by secret",
    "Fanny's heart was filled with delight",
    "Blackie might chance to serve instead of a long",
    "L6000 for a monopoly in trade, which he",
    "effective bits/weight including scales",
    "Ai2 OLMoE-1B-7B-0125",
    "nonfinite_microbatches: 0",
    "default `libm`",
    "ffn_swiglu-0",
    "Three arms over the **same kernel**",
    "7.1095157598044e-8",
    "1.217x end-to-end",
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
    result = {"experiment": "METH-131-pre-inference-answerability",
              "manifest_sha256": MANIFEST_SHA, "answerable_count": len(rows),
              "rows": rows,
              "scope": "Excerpt-only details frozen before donor or adapted-model inference"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"answerable_count": len(rows), "sha256": sha(args.out.read_bytes())}))


if __name__ == "__main__":
    main()
