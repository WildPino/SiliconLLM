#!/usr/bin/env python3
"""Export the exact METH-56 E128 parent router and BF16 factors as CPU control."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
PARENT = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth56_checkpoints/meth56_update512.pt"
PARENT_SHA = "8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072"
MANIFEST = DOCS / "meth121_zero_mean_child_external_manifest.json"
MANIFEST_SHA = "7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366"
MAGIC = b"M125PR01"
HEADER = "<8s8I"
LAYERS, WIDTH, RANK, AXIS_A, AXIS_B, R = 24, 896, 64, 8, 16, 8
EXPERTS = AXIS_A * AXIS_B


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_fp32(tensor):
    assert tensor.dtype == torch.float32
    return tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()


def raw_bf16(tensor):
    return tensor.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert sha(PARENT) == PARENT_SHA
    assert sha(MANIFEST) == MANIFEST_SHA
    saved = torch.load(PARENT, map_location="cpu", weights_only=False)
    assert saved["updates"] == 512 and len(saved["expert_state"]) == LAYERS
    dims = (LAYERS, WIDTH, RANK, AXIS_A, AXIS_B, 0, 1, R)
    args.bank.parent.mkdir(parents=True, exist_ok=True)
    with args.bank.open("wb") as output:
        output.write(struct.pack(HEADER, MAGIC, *dims))
        for layer in saved["expert_state"]:
            assert layer["a"].shape == (EXPERTS, R, WIDTH)
            assert layer["b"].shape == (EXPERTS, WIDTH, R)
            for key in ("router", "a", "b"):
                output.write((raw_fp32 if key == "router" else raw_bf16)(layer[key]))
    router_bytes = (RANK * WIDTH + (AXIS_A + AXIS_B) * RANK) * 4
    factor_bytes = EXPERTS * R * WIDTH * 2
    layer_bytes = router_bytes + 2 * factor_bytes
    expected = struct.calcsize(HEADER) + LAYERS * layer_bytes
    assert args.bank.stat().st_size == expected
    mapped = np.memmap(args.bank, dtype=np.uint8, mode="r")
    for li, layer in enumerate(saved["expert_state"]):
        offset = struct.calcsize(HEADER) + li * layer_bytes
        for key, size in (("router", router_bytes), ("a", factor_bytes), ("b", factor_bytes)):
            raw = (raw_fp32 if key == "router" else raw_bf16)(layer[key])
            assert bytes(mapped[offset:offset + size]) == raw
            offset += size
    del mapped
    result = {"experiment": "METH-125-parent-E128-native-control-export",
              "parent_checkpoint_sha256": PARENT_SHA,
              "external_manifest_sha256": MANIFEST_SHA,
              "dimensions": dict(zip(("layers", "width", "parent_rank", "axis_a", "axis_b",
                                      "child_rank", "children", "factor_rank"), dims)),
              "bank": {"path": str(args.bank.resolve()), "sha256": sha(args.bank),
                       "bytes": expected},
              "router_bytes_per_layer": router_bytes,
              "factor_bytes_per_layer": factor_bytes * 2}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bank": result["bank"], "dimensions": result["dimensions"]}, indent=2))


if __name__ == "__main__":
    main()
