#!/usr/bin/env python3
"""Bind the frozen parent/child B-mode decomposition before quality scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import psutil
import torch


ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth56_checkpoints/meth56_update512.pt"
CHILD = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth107_long_chat_e1280.pt"
PARENT_SHA = "8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072"
CHILD_SHA = "15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    start = time.monotonic()
    assert sha(PARENT) == PARENT_SHA
    assert sha(CHILD) == CHILD_SHA
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    child = torch.load(CHILD, map_location="cpu", weights_only=False)
    assert parent["updates"] == 512 and child["updates"] == 256
    assert child["parent_checkpoint_sha256"] == PARENT_SHA
    assert len(parent["expert_state"]) == len(child["expert_state"]) == 24
    layers = []
    for li, (p, c) in enumerate(zip(parent["expert_state"], child["expert_state"])):
        base = p["b"]
        raw = c["b"].reshape(128, 10, 896, 8)
        assert base.shape == (128, 896, 8) and raw.shape == (128, 10, 896, 8)
        mean = raw.mean(dim=1, keepdim=True)
        common = mean[:, 0] - base
        specific = raw - mean
        centered = base[:, None] + specific
        sibling_delta = (centered - centered[:, :1]).abs().amax(dim=(-1, -2))
        common_rms = float(common.square().mean().sqrt())
        specific_rms = float(specific.square().mean().sqrt())
        layers.append({"layer": li,
                       "common_rms": common_rms,
                       "child_specific_rms": specific_rms,
                       "common_over_specific_rms": common_rms / specific_rms,
                       "centered_parent_mean_max_abs_error": float(
                           (centered.mean(dim=1) - base).abs().max()),
                       "centered_distinct_child_slots_over_0p001": int(
                           (sibling_delta > 0.001).sum())})
    result = {"experiment": "METH-118-frozen-child-B-mode-diagnostic",
              "parent_checkpoint_sha256": PARENT_SHA,
              "child_checkpoint_sha256": CHILD_SHA,
              "layers": layers,
              "summary": {
                  "common_over_specific_rms_min": min(r["common_over_specific_rms"] for r in layers),
                  "common_over_specific_rms_max": max(r["common_over_specific_rms"] for r in layers),
                  "centered_distinct_slots_min": min(r["centered_distinct_child_slots_over_0p001"] for r in layers),
                  "centered_distinct_slots_max": max(r["centered_distinct_child_slots_over_0p001"] for r in layers),
                  "centered_mean_error_max": max(r["centered_parent_mean_max_abs_error"] for r in layers),
              },
              "runtime": {"seconds": time.monotonic() - start,
                          "rss_bytes": psutil.Process().memory_info().rss}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "runtime": result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
