#!/usr/bin/env python3
"""Export the stored METH-59 per-row R8 tied head for the native two-pass path."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import psutil
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/native_expert_scaling/meth59_qwen05b_instruct_r8_core.safetensors"
SOURCE_SHA = "5b6ace4a027d127a6eb49ca99197edd4a1b810ebe681d8e3b6ab7edc088ca9bd"
DONOR_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
PARENT_SHA = "8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072"
HEADER = struct.Struct("<8sII")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    assert sha(SOURCE) == SOURCE_SHA
    with safe_open(str(SOURCE), framework="pt", device="cpu") as file:
        meta = file.metadata()
        assert meta["model_sha256"] == DONOR_SHA
        assert meta["product_key_checkpoint_sha256"] == PARENT_SHA
        q = file.get_tensor("model.embed_tokens.weight.q")
        scale = file.get_tensor("model.embed_tokens.weight.scale")
    assert q.shape == (151936, 896) and q.dtype == torch.int8
    assert scale.shape == (151936,) and scale.dtype == torch.float32
    assert bool(torch.isfinite(scale).all()) and bool((scale > 0).all())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as output:
        output.write(HEADER.pack(b"M129HD01", 151936, 896))
        output.write(scale.numpy().astype("<f4", copy=False).tobytes())
        output.write(q.numpy().astype("i1", copy=False).tobytes())
    expected = HEADER.size + scale.numel() * 4 + q.numel()
    assert args.out.stat().st_size == expected
    with args.out.open("rb") as file:
        assert file.read(HEADER.size) == HEADER.pack(b"M129HD01", 151936, 896)
        assert np.array_equal(np.frombuffer(file.read(scale.numel() * 4), dtype="<f4"),
                              scale.numpy())
        assert np.array_equal(np.frombuffer(file.read(q.numel()), dtype="i1").reshape(q.shape),
                              q.numpy())
        assert file.read(1) == b""
    result = {"experiment": "METH-129-native-head-sidecar-export",
              "source_artifact_sha256": SOURCE_SHA,
              "donor_source_sha256": DONOR_SHA,
              "parent_checkpoint_sha256": PARENT_SHA,
              "sidecar": {"path": str(args.out.resolve()), "sha256": sha(args.out),
                          "bytes": expected, "vocab": 151936, "width": 896},
              "exact_code_scale_readback": True,
              "seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss}
    assert result["seconds"] <= 20 * 60 and result["rss_bytes"] <= 16 * (1 << 30)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
