#!/usr/bin/env python3
"""Export the bound METH-36 E128 router sketch to a simple C benchmark seed."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/native_expert_scaling/meth36_e128_router_r64_int8.safetensors"
SOURCE_SHA = "133287eb87cf993d9498fd18c1efafbf703bf10a29cc757a5a5ec73cadbb674d"
CORE_SHA = "c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27"
ADAPTER_SHA = "3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca"
LAYERS, EXPERTS, DIM, RANK = 24, 128, 896, 64
MAGIC = b"M40R64\0\0"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    assert sha(SOURCE) == SOURCE_SHA
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(str(SOURCE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_E128_ROUTER_R64_INT8_V1"
        assert meta["rank"] == str(RANK)
        assert meta["core_sha256"] == CORE_SHA
        assert meta["adapter_sha256"] == ADAPTER_SHA
        assert len(archive.keys()) == LAYERS * 3
        with args.out.open("wb") as target:
            target.write(struct.pack("<8s4I", MAGIC, LAYERS, EXPERTS, DIM, RANK))
            for li in range(LAYERS):
                basis = archive.get_tensor(f"layers.{li}.basis")
                codes = archive.get_tensor(f"layers.{li}.codes")
                scales = archive.get_tensor(f"layers.{li}.scales")
                assert tuple(basis.shape) == (DIM, RANK)
                assert tuple(codes.shape) == (EXPERTS, RANK)
                assert tuple(scales.shape) == (EXPERTS,)
                basis = basis.numpy()
                codes = codes.numpy()
                scales = scales.numpy()
                assert basis.dtype == np.float32 and codes.dtype == np.int8
                assert scales.dtype == np.float32
                assert np.isfinite(basis).all() and np.isfinite(scales).all()
                assert (scales >= 0).all()
                assert (codes >= -127).all() and (codes <= 127).all()
                target.write(basis.astype("<f4", copy=False).tobytes(order="C"))
                target.write(codes.tobytes(order="C"))
                target.write(scales.astype("<f4", copy=False).tobytes(order="C"))
    expected = 24 + LAYERS * (DIM * RANK * 4 + EXPERTS * RANK + EXPERTS * 4)
    assert args.out.stat().st_size == expected
    report = {"experiment": "METH-40-seed", "source_sha256": SOURCE_SHA,
              "raw_seed_sha256": sha(args.out), "raw_seed_bytes": expected,
              "shape": {"layers": LAYERS, "experts": EXPERTS,
                        "dimension": DIM, "rank": RANK},
              "layout": "little-endian 8s4I header; for each layer: D×R f32 basis, E×R i8 codes, E f32 scales",
              "scope": "CPU cost seed only; expanded rows are synthetic"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
