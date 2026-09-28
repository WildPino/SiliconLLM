#!/usr/bin/env python3
"""METH-126: losslessly deduplicate sibling A factors in the METH-124 bank."""

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import time

import psutil


SOURCE_SHA = "8e4ef1f8abea39d9562e8b617f460e89591fa2b2d7fa7478fac0901804064580"
HEADER = struct.Struct("<8s8I")
DIMS = (24, 896, 64, 8, 16, 32, 10, 8)
MAX_SECONDS = 15 * 60
MAX_RSS_BYTES = 4 * (1 << 30)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    start = time.monotonic()
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    def budget():
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        assert peak_rss <= MAX_RSS_BYTES, "RSS budget"
        assert time.monotonic() - start <= MAX_SECONDS, "time budget"

    assert args.source.resolve() != args.out.resolve()
    assert digest(args.source) == SOURCE_SHA, "source SHA-256 mismatch"
    with args.source.open("rb") as source_file:
        source = mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            magic, *dims = HEADER.unpack_from(source)
            assert magic == b"M124FB01" and tuple(dims) == DIMS
            layers, width, rank, na, nb, child_rank, children, factor_rank = dims
            parents = na * nb
            router_bytes = ((rank * width + (na + nb) * rank) +
                            child_rank * width + parents * children * child_rank) * 4
            row_bytes = factor_rank * width * 2
            old_a_bytes = parents * children * row_bytes
            new_a_bytes = parents * row_bytes
            b_bytes = old_a_bytes
            old_layer_bytes = router_bytes + old_a_bytes + b_bytes
            new_layer_bytes = router_bytes + new_a_bytes + b_bytes
            expected_old = HEADER.size + layers * old_layer_bytes
            expected_new = HEADER.size + layers * new_layer_bytes
            assert len(source) == expected_old == 893141032
            assert expected_new == 496779304
            args.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.out.with_name(args.out.name + ".partial")
            assert not tmp.exists(), "partial output already exists"
            comparisons = 0
            try:
                with tmp.open("wb") as output:
                    output.write(HEADER.pack(b"M126FB01", *dims))
                    for li in range(layers):
                        old_base = HEADER.size + li * old_layer_bytes
                        a_base = old_base + router_bytes
                        # Validate all siblings before writing this layer.
                        for parent in range(parents):
                            first = a_base + parent * children * row_bytes
                            for child in range(1, children):
                                other = first + child * row_bytes
                                assert source[first:first + row_bytes] == source[
                                    other:other + row_bytes], (li, parent, child)
                                comparisons += 1
                        output.write(source[old_base:a_base])
                        for parent in range(parents):
                            first = a_base + parent * children * row_bytes
                            output.write(source[first:first + row_bytes])
                        b_base = a_base + old_a_bytes
                        output.write(source[b_base:b_base + b_bytes])
                        budget()
                assert tmp.stat().st_size == expected_new
                with tmp.open("rb") as output_file:
                    compact = mmap.mmap(output_file.fileno(), 0, access=mmap.ACCESS_READ)
                    try:
                        assert compact[:HEADER.size] == HEADER.pack(b"M126FB01", *dims)
                        for li in range(layers):
                            old_base = HEADER.size + li * old_layer_bytes
                            new_base = HEADER.size + li * new_layer_bytes
                            assert source[old_base:old_base + router_bytes] == compact[
                                new_base:new_base + router_bytes]
                            old_a = old_base + router_bytes
                            new_a = new_base + router_bytes
                            for parent in range(parents):
                                assert source[old_a + parent * children * row_bytes:
                                              old_a + (parent * children + 1) * row_bytes] == compact[
                                    new_a + parent * row_bytes:new_a + (parent + 1) * row_bytes]
                            assert source[old_a + old_a_bytes:old_a + old_a_bytes + b_bytes] == compact[
                                new_a + new_a_bytes:new_a + new_a_bytes + b_bytes]
                            budget()
                    finally:
                        compact.close()
                os.replace(tmp, args.out)
            finally:
                if tmp.exists():
                    tmp.unlink()
        finally:
            source.close()
    result = {
        "experiment": "METH-126-exact-shared-A-E1280-factor-bank",
        "source": {"path": str(args.source.resolve()), "sha256": SOURCE_SHA,
                   "bytes": expected_old},
        "compact": {"path": str(args.out.resolve()), "sha256": digest(args.out),
                    "bytes": args.out.stat().st_size},
        "dimensions": dict(zip(("layers", "width", "parent_rank", "axis_a", "axis_b",
                                "child_rank", "children", "factor_rank"), DIMS)),
        "sibling_A_byte_comparisons": comparisons,
        "router_A_B_complete_readback": True,
        "bytes_saved": expected_old - expected_new,
        "peak_rss_bytes_at_layer_boundaries": peak_rss,
        "seconds": time.monotonic() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
