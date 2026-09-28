#!/usr/bin/env python3
"""Verify every native hash route against the frozen Python formula."""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import numpy as np

import meth136_sparse_experts  # inserts the S1 import path
import meth136_matched_sparse_train as M136
import meth142_token_hash_route_screen as M142


ROUTE = re.compile(r"HASH_ROUTE token=(\d+) layer=(\d+) children=([\d,]+) grandchildren=([\d,]+)$")
REP = re.compile(r"HASH_REP rep=(\d+) base_ms=([\d.]+) hash_ms=([\d.]+)$")
LAYER = re.compile(r"HASH_LAYER layer=(\d+) coverage=(\d+)$")
SUMMARY = re.compile(r"HASH_SUMMARY .*gate=(PASS|FAIL)$")
VECTOR_SHA = "f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    assert M136.digest(M136.EXACT_BANK) == M136.EXACT_SHA
    assert M136.digest(args.vectors) == VECTOR_SHA
    raw = args.fixture.read_bytes()
    assert len(raw) == 12 + 256 * 12
    assert struct.unpack_from("<8sI", raw) == (b"M144TX01", 256)
    context = np.frombuffer(raw, dtype="<u4", count=256*3, offset=12).reshape(256, 3)
    rows = {}
    reps, layers = [], {}
    summary = None
    for line in args.log.read_text(encoding="utf-8").splitlines():
        match = ROUTE.fullmatch(line)
        if match:
            token, layer = map(int, match.group(1, 2))
            children = np.asarray([int(x) for x in match.group(3).split(",")], dtype=np.int64)
            grand = np.asarray([int(x) for x in match.group(4).split(",")], dtype=np.int64)
            assert children.shape == grand.shape == (4,)
            assert (token, layer) not in rows
            expected = M142.hash_grandchildren(
                np.array([context[token, 0]], dtype=np.int64),
                np.array([context[token, 1]], dtype=np.int64),
                np.array([context[token, 2]], dtype=np.int64),
                children[None], layer)[0]
            assert np.array_equal(grand, expected), (token, layer, grand, expected)
            rows[(token, layer)] = True
            continue
        match = REP.fullmatch(line)
        if match:
            reps.append({"rep": int(match.group(1)), "base_ms": float(match.group(2)),
                         "hash_ms": float(match.group(3))})
            continue
        match = LAYER.fullmatch(line)
        if match:
            layers[int(match.group(1))] = int(match.group(2))
            continue
        match = SUMMARY.fullmatch(line)
        if match:
            summary = line
    assert len(rows) == 256 * 24
    assert len(reps) == 5 and [r["rep"] for r in reps] == list(range(5))
    assert len(layers) == 24 and min(layers.values()) > 0
    assert summary is not None and summary.endswith("gate=PASS")
    base = float(np.median([r["base_ms"] for r in reps]))
    hashed = float(np.median([r["hash_ms"] for r in reps]))
    assert hashed <= 3.0 and hashed <= 2.0 * base
    with tempfile.TemporaryDirectory(prefix="meth144-bad-header-") as directory:
        corrupt = Path(directory) / "corrupt.bin"
        corrupt.write_bytes(b"X" + raw[1:])
        bad = subprocess.run([str(args.native.resolve()), str(M136.EXACT_BANK.resolve()),
                              str(args.vectors.resolve()), str(corrupt.resolve())],
                             capture_output=True, text=True, timeout=30)
        assert bad.returncode == 2 and "invalid input header" in bad.stderr
    result = {"experiment": "METH-144-native-hash-route",
              "exact_bank_sha256": M136.EXACT_SHA,
              "vectors_sha256": VECTOR_SHA,
              "fixture_sha256": M136.digest(args.fixture),
              "native_log_sha256": M136.digest(args.log),
              "hash_golden": 899, "python_c_matching_routes": len(rows),
              "layers_coverage": layers, "paired_repetitions": reps,
              "base_median_ms_per_token": base,
              "hash_median_ms_per_token": hashed,
              "hash_to_base_ratio": hashed / base,
              "bad_header_rejected": True, "native_summary": summary,
              "decision": "native_hash_route_pass"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "routes": len(rows),
                      "base_median_ms": base, "hash_median_ms": hashed,
                      "ratio": hashed/base, "coverage_min": min(layers.values())},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
