#!/usr/bin/env python3
"""Export the bound METH-51 BF16 factors to an interleaved C benchmark seed."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/native_expert_scaling/meth51_e128_effective_bf16.safetensors"
SOURCE_SHA = "53f2849da55f0da7f91f7d097ab3357308abfd85174cdf8a82b7e226cd738473"
CHECKPOINT_SHA = "0c6f09562efff7e78921036fe6ab030d7377150fe9afbe06b3deb1c029e0aa59"
MODEL_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
LAYERS, EXPERTS, DIM, RANK = 24, 128, 896, 8
MAGIC = b"M52BF16\0"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert sha(SOURCE) == SOURCE_SHA
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(str(SOURCE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta == {"format": "METH51_E128_AB_BF16_EFFECTIVE_V1",
                        "checkpoint_sha256": CHECKPOINT_SHA,
                        "model_sha256": MODEL_SHA,
                        "effective_dtype": "bfloat16"}
        assert len(archive.keys()) == LAYERS * 2
        with args.out.open("wb") as target:
            target.write(struct.pack("<8s4I", MAGIC, LAYERS, EXPERTS, DIM, RANK))
            for li in range(LAYERS):
                a = archive.get_tensor(f"layers.{li}.a")
                b = archive.get_tensor(f"layers.{li}.b")
                assert a.dtype == b.dtype == torch.bfloat16
                assert tuple(a.shape) == (EXPERTS, RANK, DIM)
                assert tuple(b.shape) == (EXPERTS, DIM, RANK)
                assert bool(torch.isfinite(a.float()).all())
                assert bool(torch.isfinite(b.float()).all())
                for e in range(EXPERTS):
                    target.write(a[e].contiguous().view(torch.uint8).numpy().tobytes())
                    target.write(b[e].contiguous().view(torch.uint8).numpy().tobytes())
    payload_per_expert = 2 * RANK * DIM * 2
    expected = 24 + LAYERS * EXPERTS * payload_per_expert
    assert args.out.stat().st_size == expected
    report = {"experiment": "METH-52-seed",
              "source_sha256": SOURCE_SHA,
              "raw_seed_sha256": sha(args.out),
              "raw_seed_bytes": expected,
              "shape": {"layers": LAYERS, "experts": EXPERTS,
                        "dimension": DIM, "rank": RANK},
              "payload_bytes_per_expert_per_layer": payload_per_expert,
              "layout": "little-endian 8s4I header; each layer/expert: A[8,896] bf16 then B[896,8] bf16",
              "scope": "CPU cost seed only; expanded expert rows are replicas"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
