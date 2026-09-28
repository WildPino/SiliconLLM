#!/usr/bin/env python3
"""Export the seeded E12800 third-tier router sidecar for a CPU cost test."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import torch


MAGIC = b"M135RT01"
HEADER = struct.Struct("<8s6I")
LAYERS, WIDTH, RANK, CHILDREN, GRANDCHILDREN, SEED = 24, 896, 32, 1280, 10, 134134


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    args.bank.parent.mkdir(parents=True, exist_ok=True)
    projection_count = RANK * WIDTH
    key_count = CHILDREN * GRANDCHILDREN * RANK
    layer_floats = projection_count + key_count
    with args.bank.open("xb") as file:
        file.write(HEADER.pack(MAGIC, LAYERS, WIDTH, RANK,
                               CHILDREN, GRANDCHILDREN, SEED))
        for layer in range(LAYERS):
            generator = torch.Generator(device="cpu").manual_seed(SEED + 1 + layer)
            projection = (torch.randn((RANK, WIDTH), generator=generator)
                          * (0.1 / WIDTH**0.5))
            keys = (torch.randn((CHILDREN, GRANDCHILDREN, RANK), generator=generator)
                    * (0.1 / RANK**0.5))
            file.write(projection.numpy().astype("<f4", copy=False).tobytes())
            file.write(keys.numpy().astype("<f4", copy=False).tobytes())
    expected_bytes = HEADER.size + LAYERS * layer_floats * 4
    assert args.bank.stat().st_size == expected_bytes
    mapped = np.memmap(args.bank, dtype=np.uint8, mode="r")
    assert HEADER.unpack_from(mapped) == (MAGIC, LAYERS, WIDTH, RANK,
                                         CHILDREN, GRANDCHILDREN, SEED)
    for layer in range(LAYERS):
        generator = torch.Generator(device="cpu").manual_seed(SEED + 1 + layer)
        projection = (torch.randn((RANK, WIDTH), generator=generator)
                      * (0.1 / WIDTH**0.5))
        keys = (torch.randn((CHILDREN, GRANDCHILDREN, RANK), generator=generator)
                * (0.1 / RANK**0.5))
        base = HEADER.size + layer * layer_floats * 4
        assert mapped[base:base + projection_count * 4].tobytes() == (
            projection.numpy().astype("<f4", copy=False).tobytes())
        assert mapped[base + projection_count * 4:
                      base + layer_floats * 4].tobytes() == (
            keys.numpy().astype("<f4", copy=False).tobytes())
    result = {"experiment": "METH-135-third-tier-router-export",
              "magic": MAGIC.decode(), "seed": SEED,
              "layers": LAYERS, "width": WIDTH, "rank": RANK,
              "source_children": CHILDREN, "grandchildren_per_child": GRANDCHILDREN,
              "bank": {"path": str(args.bank), "bytes": expected_bytes,
                       "sha256": sha(args.bank)},
              "all_projection_key_records_verified": True,
              "runtime": {"elapsed_seconds": time.monotonic() - start,
                          "rss_bytes": psutil.Process().memory_info().rss,
                          "torch": torch.__version__, "numpy": np.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
