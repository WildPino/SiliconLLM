#!/usr/bin/env python3
"""Make a throughput-only E4M1 expert-count stress artifact from pinned E128.

Experts are repeated and the router is independently randomized with the
source layer's mean/std. This does not add learned capacity or preserve model
quality. It tests the same C loader, router, and selected LUT path with a much
larger resident pool while top-k and per-expert geometry stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "259e1aa73593e8fa8f72a997aaa63d30961342676fa0be587a22d920c804d2cf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    if not 2 <= args.replicas <= 100:
        parser.error("replicas must be 2..100")
    if sha256(args.source) != SOURCE_SHA256:
        parser.error("source E128 E4M1 binary differs from the pinned NES-01 export")

    source = np.memmap(args.source, mode="r", dtype=np.uint8)
    header = list(struct.unpack("<16I", source[:64]))
    magic, vocab, width, state, heads, layers, dn, dt_rank, conv, win, swa, experts, hidden, topk, packed, reserved = header
    if (magic, vocab, width, state, heads, layers, dn, dt_rank, conv, win, swa,
            experts, hidden, topk, packed, reserved) != (
            0x45344D31, 1024, 256, 96, 8, 6, 512, 16, 4, 128, 5, 128, 128, 8, 1, 0):
        parser.error("source E4M1 topology differs from pinned NES-01 E128")
    enlarged = experts * args.replicas
    if enlarged > 65535:
        parser.error("result exceeds the current C route-ID limit")
    header[11] = enlarged
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    digest = hashlib.sha256()
    offset = 64

    with args.out.open("xb") as output:
        def emit(buffer: bytes | np.ndarray) -> None:
            output.write(buffer)
            digest.update(buffer)

        def copy_bytes(length: int) -> None:
            nonlocal offset
            emit(source[offset:offset + length])
            offset += length

        def repeat_bytes(length: int) -> None:
            nonlocal offset
            block = source[offset:offset + length]
            for _ in range(args.replicas):
                emit(block)
            offset += length

        emit(struct.pack("<16I", *header))
        copy_bytes(vocab * width * 4)  # embedding
        for layer in range(layers):
            copy_bytes(width * 4)  # first norm
            if layer == swa:
                mixer_elements = 3 * width * width + width * width
            else:
                mixer_elements = (2 * dn * width + dn * conv + dn +
                                  (dt_rank + 2 * state) * dn + dn * dt_rank + dn +
                                  dn * state + dn + width * dn)
            copy_bytes(mixer_elements * 4)
            copy_bytes(width * 4)  # second norm

            old_w = np.frombuffer(source[offset:offset + experts * width * 4],
                                  dtype="<f4").reshape(experts, width)
            offset += experts * width * 4
            old_b = np.frombuffer(source[offset:offset + experts * 4], dtype="<f4")
            offset += experts * 4
            # Independent candidates spread selections across the enlarged
            # address range. Their logits have no quality interpretation.
            new_w = rng.normal(float(old_w.mean()), float(old_w.std()),
                               size=(enlarged, width)).astype("<f4")
            new_b = rng.normal(float(old_b.mean()), float(old_b.std()),
                               size=enlarged).astype("<f4")
            emit(new_w)
            emit(new_b)
            repeat_bytes(experts * hidden * width * 4)  # gate fp32
            repeat_bytes(experts * hidden * width * 4)  # up fp32
            repeat_bytes(experts * width * hidden * 4)  # down fp32
        copy_bytes(width * 4 + vocab * width * 4)  # final norm and head
        for _ in range(layers):
            repeat_bytes(experts * hidden * width)  # gate ternary codes
            repeat_bytes(experts * hidden * 4)      # gate scales
            repeat_bytes(experts * hidden * width)  # up ternary codes
            repeat_bytes(experts * hidden * 4)      # up scales
            repeat_bytes(experts * width * hidden)  # down ternary codes
            repeat_bytes(experts * width * 4)       # down scales
    if offset != source.size:
        raise ValueError(f"source parser stopped at {offset}, file has {source.size} bytes")
    print(f"throughput-only E={enlarged}; {args.out.stat().st_size} B; SHA-256 {digest.hexdigest()}")


if __name__ == "__main__":
    main()
