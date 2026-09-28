#!/usr/bin/env python3
"""Freeze the shared-route C table and all-hit/all-miss context fixtures."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DOC / "meth150_shared_route_manifest.json"
MANIFEST_SHA = "6e911e65529be53dba65fbe382229f0f43e510eb76ab83f4404d4be0368ff5d9"
TABLE_HEADER = struct.Struct("<8sII")
FIXTURE_HEADER = struct.Struct("<8sI")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-out", type=Path, required=True)
    parser.add_argument("--hit-out", type=Path, required=True)
    parser.add_argument("--miss-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.table_out, args.hit_out, args.miss_out, args.report):
        assert not path.exists(), path
    assert M136.digest(MANIFEST) == MANIFEST_SHA
    shared = R150.load_table()
    assert R150.golden(shared) == {"content": 899, "structural": 890}
    table_record = json.loads(R150.TABLE.read_text(encoding="utf-8"))
    rows = np.asarray([(row["token"], row["previous"], row["position"])
                       for row in table_record["tuples"]], dtype="<u4")
    assert rows.shape == (75, 3)
    assert [tuple(map(int, row)) for row in rows] == sorted(shared)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_bytes(TABLE_HEADER.pack(b"M151TB01", 75, 32) + rows.tobytes())
    assert args.table_out.read_bytes() == TABLE_HEADER.pack(b"M151TB01", 75, 32) + rows.tobytes()
    hits = np.asarray([rows[i % len(rows)] for i in range(256)], dtype="<u4")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = np.asarray(manifest["items"][0]["document_ids"][:256], dtype="<u4")
    assert ids.shape == (256,)
    misses = np.stack((ids, np.r_[np.uint32(0), ids[:-1]],
                       np.arange(256, dtype="<u4")), axis=1)
    assert all(tuple(map(int, row)) in shared for row in hits)
    assert all(tuple(map(int, row)) not in shared for row in misses)
    for path, data in ((args.hit_out, hits), (args.miss_out, misses)):
        payload = FIXTURE_HEADER.pack(b"M151TX01", 256) + data.tobytes()
        path.write_bytes(payload)
        assert path.read_bytes() == payload
    report = {"experiment": "METH-151-native-fixtures",
              "meth149_table_sha256": R150.TABLE_SHA,
              "meth150_manifest_sha256": MANIFEST_SHA,
              "golden": {"content": 899, "structural": 890},
              "table": {"path": str(args.table_out.resolve()),
                        "sha256": M136.digest(args.table_out),
                        "bytes": args.table_out.stat().st_size,
                        "entries": 75, "readback_exact": True},
              "hit": {"path": str(args.hit_out.resolve()),
                      "sha256": M136.digest(args.hit_out),
                      "bytes": args.hit_out.stat().st_size,
                      "contexts": 256, "readback_exact": True},
              "miss": {"path": str(args.miss_out.resolve()),
                       "sha256": M136.digest(args.miss_out),
                       "bytes": args.miss_out.stat().st_size,
                       "contexts": 256, "readback_exact": True}}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"table": report["table"]["sha256"],
                      "hit": report["hit"]["sha256"],
                      "miss": report["miss"]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
