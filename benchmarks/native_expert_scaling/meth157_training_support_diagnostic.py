#!/usr/bin/env python3
"""Quantify training selections per content specialist after METH-156 failure."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TRAIN = DOC / "meth155_matched_factorized_train_result.json"
TRAIN_SHA = "759d9044c7187c9bfd8ed3fbd6cc8b6d6f3be663c81bf9549b62200668dd4e2a"
QUALITY = DOC / "meth156_factorized_prediction_result.json"
QUALITY_SHA = "45a12387a28b213dd41453cd66fed66d5a9b7f91e6754e75582a243aa91a949c"


def digest(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert digest(TRAIN) == TRAIN_SHA and digest(QUALITY) == QUALITY_SHA
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    assert train["decision"] == "matched_training_pass_fresh_quality_pending"
    assert quality["decision"] == "stop_document_prompt_gate"
    choices = train["candidate"]["content_route"]
    layers = []
    for li, counts in enumerate(choices["route_counts_by_layer"]):
        array = np.asarray(counts, dtype=np.int64).reshape(1280, 10)
        assert not np.any(array[:, 0])
        content = array[:, 1:].reshape(-1)
        active = content[content > 0]
        assert active.size == choices["route_coverage_by_layer"][li]
        assert int(content.sum()) == choices["selections_by_layer"][li]
        layers.append({"layer": li, "content_slots": int(content.size),
            "selected_slots": int(active.size), "zero_slots": int((content == 0).sum()),
            "selections_per_active_slot_p10": float(np.quantile(active, 0.10)),
            "selections_per_active_slot_p50": float(np.quantile(active, 0.50)),
            "selections_per_active_slot_p90": float(np.quantile(active, 0.90)),
            "slots_with_fewer_than_8_selections": int((content < 8).sum()),
            "slots_with_fewer_than_16_selections": int((content < 16).sum()),
            "slots_with_fewer_than_32_selections": int((content < 32).sum()),
            "slots_with_at_least_128_selections": int((content >= 128).sum()),
            "maximum_selections": int(content.max())})
    result = {"experiment": "METH-157-postfailure-training-support",
              "scope": "METH-155 changing-weight training-route counts only; no fresh text rescored",
              "meth155_result_sha256": TRAIN_SHA,
              "meth156_result_sha256": QUALITY_SHA,
              "layers": layers,
              "summary": {
                  "active_slot_median_selection_min": min(x["selections_per_active_slot_p50"] for x in layers),
                  "active_slot_median_selection_max": max(x["selections_per_active_slot_p50"] for x in layers),
                  "slots_under_32_min": min(x["slots_with_fewer_than_32_selections"] for x in layers),
                  "slots_under_32_max": max(x["slots_with_fewer_than_32_selections"] for x in layers),
                  "zero_slots_min": min(x["zero_slots"] for x in layers),
                  "zero_slots_max": max(x["zero_slots"] for x in layers),
                  "content_selections_per_layer": choices["selections_by_layer"][0]}}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
