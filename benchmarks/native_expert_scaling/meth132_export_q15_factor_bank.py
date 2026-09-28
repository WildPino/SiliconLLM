#!/usr/bin/env python3
"""Replace METH-126's child B BF16 rows with Q15 paired-LUT codes."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth126_shared_a_factor_bank.bin"
SOURCE_SHA = "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1"
HEADER = struct.Struct("<8s8I")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    assert sha(SOURCE) == SOURCE_SHA
    source = SOURCE.read_bytes()
    magic, layers, width, rank, na, nb, child_rank, children, factor_rank = HEADER.unpack_from(source)
    assert (magic, layers, width, rank, na, nb, child_rank, children, factor_rank) == (
        b"M126FB01", 24, 896, 64, 8, 16, 32, 10, 8)
    experts = na * nb * children
    nrows = experts * width
    router_bytes = ((rank * width + (na + nb) * rank) + child_rank * width +
                    experts * child_rank) * 4
    a_bytes = na * nb * factor_rank * width * 2
    b_bytes = nrows * factor_rank * 2
    source_layer_bytes = router_bytes + a_bytes + b_bytes
    target_layer_bytes = router_bytes + a_bytes + nrows * 8
    assert len(source) == HEADER.size + layers * source_layer_bytes
    expected = HEADER.size + layers * target_layer_bytes
    assert expected <= 300_000_000
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "METH-132-Q15-child-B-export", "source_sha256": SOURCE_SHA,
              "source_bytes": len(source), "target_bytes": expected,
              "nrows_per_layer": nrows, "row_layout": "float32 scale; four packed Q15 nibble pairs",
              "levels": list(range(-7, 8)), "layers": []}
    with args.out.open("w+b") as out:
        out.write(HEADER.pack(b"M132FB01", layers, width, rank, na, nb,
                              child_rank, children, factor_rank))
        for li in range(layers):
            base = HEADER.size + li * source_layer_bytes
            fixed = source[base:base + router_bytes + a_bytes]
            out.write(fixed)
            b_start = base + router_bytes + a_bytes
            bits = np.frombuffer(source, dtype="<u2", count=nrows * factor_rank,
                                 offset=b_start).reshape(nrows, factor_rank)
            weights = (bits.astype("<u4") << 16).view("<f4")
            assert np.isfinite(weights).all()
            maximum = np.max(np.abs(weights), axis=1)
            initial = maximum / 7.0
            ratios = np.zeros_like(weights)
            np.divide(weights, initial[:, None], out=ratios,
                      where=initial[:, None] > 0)
            codes = np.rint(ratios).clip(-7, 7).astype(np.int8)
            numerator = np.sum(weights * codes, axis=1, dtype=np.float32)
            denominator = np.sum(codes.astype(np.float32) ** 2, axis=1,
                                 dtype=np.float32)
            scales = np.zeros(nrows, dtype="<f4")
            np.divide(numerator, denominator, out=scales, where=denominator > 0)
            assert np.isfinite(scales).all() and np.all(scales >= 0)
            assert np.all((codes == 0)[maximum == 0])
            nibble = (codes.astype(np.uint8) + 7).reshape(nrows, 4, 2)
            packed = (nibble[:, :, 0] | (nibble[:, :, 1] << 4)).astype(np.uint8)
            records = np.empty((nrows, 8), dtype=np.uint8)
            records[:, :4] = scales.view(np.uint8).reshape(nrows, 4)
            records[:, 4:] = packed
            q_start = out.tell()
            out.write(records.tobytes())
            out.flush()
            out.seek(q_start)
            assert np.array_equal(np.frombuffer(out.read(nrows * 8), dtype=np.uint8)
                                  .reshape(nrows, 8), records)
            assert out.tell() == q_start + nrows * 8
            reconstructed = codes.astype(np.float32) * scales[:, None]
            delta = weights - reconstructed
            rel = float(np.sqrt(np.sum(delta.astype(np.float64) ** 2) /
                                max(np.sum(weights.astype(np.float64) ** 2), 1e-30)))
            report["layers"].append({"layer": li, "weight_relative_l2": rel,
                                     "zero_rows": int(np.count_nonzero(maximum == 0)),
                                     "zero_code_fraction": float(np.mean(codes == 0))})
            assert time.monotonic() - start <= 600
            assert psutil.Process().memory_info().rss <= 4 * (1 << 30)
    assert args.out.stat().st_size == expected
    report["target_sha256"] = sha(args.out)
    report["exact_packed_readback"] = True
    report["seconds"] = time.monotonic() - start
    report["rss_bytes_final"] = psutil.Process().memory_info().rss
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
