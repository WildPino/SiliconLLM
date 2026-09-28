#!/usr/bin/env python3
"""Verify both native shared-route cost cells against frozen Python routing."""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import numpy as np

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth150_shared_structure_route as R150


ROUTE = re.compile(r"HASH_ROUTE token=(\d+) layer=(\d+) children=([\d,]+) grandchildren=([\d,]+)$")
REP = re.compile(r"HASH_REP rep=(\d+) base_ms=([\d.]+) hash_ms=([\d.]+)$")
LAYER = re.compile(r"HASH_LAYER layer=(\d+) coverage=(\d+)$")
VECTOR_SHA = "f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699"
TABLE_SHA = "926362655bcc71bf7bd1ceee944e56ba8f35ef124c8f60b6e0052d6525ddfc27"
HIT_SHA = "c211bde3459b3a1df27250ee0e785cde412cba07fb763aeb519281588f54a629"
MISS_SHA = "ecce0701d71201f9a48495adbc9814632c9f5b0929675a30daf2ffc92ac8682e"


def verify_cell(name, fixture_path, log_path, table):
    raw = fixture_path.read_bytes()
    assert len(raw) == 12 + 256 * 12
    assert struct.unpack_from("<8sI", raw) == (b"M151TX01", 256)
    contexts = np.frombuffer(raw, dtype="<u4", count=256 * 3, offset=12).reshape(256, 3)
    hit_positions = sum(tuple(map(int, row)) in table for row in contexts)
    assert hit_positions == (256 if name == "hit" else 0)
    routes = set()
    reps = []
    coverage = {}
    summary = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = ROUTE.fullmatch(line)
        if match:
            token, layer = map(int, match.group(1, 2))
            children = np.asarray([int(x) for x in match.group(3).split(",")], dtype=np.int64)
            grand = np.asarray([int(x) for x in match.group(4).split(",")], dtype=np.int64)
            assert children.shape == grand.shape == (4,)
            assert (token, layer) not in routes
            context = contexts[token]
            expected, structural = R150.route(
                np.array([context[0]], dtype=np.int64),
                np.array([context[1]], dtype=np.int64),
                np.array([context[2]], dtype=np.int64), children[None], layer, table)
            assert np.array_equal(grand, expected[0])
            assert bool(structural[0]) == (name == "hit")
            routes.add((token, layer))
            continue
        match = REP.fullmatch(line)
        if match:
            reps.append({"rep": int(match.group(1)),
                         "base_ms": float(match.group(2)),
                         "shared_ms": float(match.group(3))})
            continue
        match = LAYER.fullmatch(line)
        if match:
            coverage[int(match.group(1))] = int(match.group(2))
            continue
        if line.startswith("HASH_SUMMARY "):
            summary = line
    assert len(routes) == 256 * 24
    assert len(reps) == 5 and [row["rep"] for row in reps] == list(range(5))
    assert len(coverage) == 24 and min(coverage.values()) > 0
    assert summary and summary.endswith("gate=PASS")
    structural_hits = int(re.search(r"structural_hits=(\d+)", summary).group(1))
    assert structural_hits == hit_positions * 24 * 4
    base = float(np.median([row["base_ms"] for row in reps]))
    shared = float(np.median([row["shared_ms"] for row in reps]))
    assert shared <= 3 and shared <= 2 * base
    return {"fixture_sha256": M136.digest(fixture_path),
            "log_sha256": M136.digest(log_path),
            "python_c_matching_routes": len(routes),
            "structural_hits": structural_hits,
            "coverage_by_layer": coverage,
            "paired_repetitions": reps,
            "base_median_ms_per_token": base,
            "shared_median_ms_per_token": shared,
            "shared_to_base_ratio": shared / base,
            "native_summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--hit", type=Path, required=True)
    parser.add_argument("--hit-log", type=Path, required=True)
    parser.add_argument("--miss", type=Path, required=True)
    parser.add_argument("--miss-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    assert M136.digest(M136.EXACT_BANK) == M136.EXACT_SHA
    assert M136.digest(args.vectors) == VECTOR_SHA
    for path, sha in ((args.table, TABLE_SHA), (args.hit, HIT_SHA),
                      (args.miss, MISS_SHA)):
        assert M136.digest(path) == sha
    raw = args.table.read_bytes()
    assert struct.unpack_from("<8sII", raw) == (b"M151TB01", 75, 32)
    assert len(raw) == 16 + 75 * 12
    rows = np.frombuffer(raw, dtype="<u4", count=75 * 3, offset=16).reshape(75, 3)
    table = R150.load_table()
    assert [tuple(map(int, row)) for row in rows] == sorted(table)
    assert R150.golden(table) == {"content": 899, "structural": 890}
    cells = {"hit": verify_cell("hit", args.hit, args.hit_log, table),
             "miss": verify_cell("miss", args.miss, args.miss_log, table)}
    with tempfile.TemporaryDirectory(prefix="meth151-bad-table-") as directory:
        corrupted = Path(directory) / "table.bin"
        corrupted.write_bytes(b"X" + raw[1:])
        bad = subprocess.run([
            str(args.native.resolve()), str(M136.EXACT_BANK.resolve()),
            str(args.vectors.resolve()), str(args.miss.resolve()),
            str(corrupted.resolve())], capture_output=True, text=True, timeout=30)
        assert bad.returncode == 2 and "invalid input header" in bad.stderr
    result = {"experiment": "METH-151-native-shared-route",
              "exact_bank_sha256": M136.EXACT_SHA,
              "vectors_sha256": VECTOR_SHA,
              "table_sha256": TABLE_SHA,
              "golden": {"content": 899, "structural": 890},
              "cells": cells, "bad_table_header_rejected": True,
              "decision": "native_shared_route_pass"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "hit_ms": cells["hit"]["shared_median_ms_per_token"],
                      "miss_ms": cells["miss"]["shared_median_ms_per_token"],
                      "hit_ratio": cells["hit"]["shared_to_base_ratio"],
                      "miss_ratio": cells["miss"]["shared_to_base_ratio"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
