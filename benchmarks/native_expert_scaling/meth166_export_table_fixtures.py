#!/usr/bin/env python3
"""Export METH-162's exact 138-entry native lookup and newly shared contexts."""

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
ART = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling"
TABLE = DOC / "meth162_training_derived_shared_table.json"
TABLE_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
RESULT = DOC / "meth162_postfailure_table_result.json"
RESULT_SHA = "d10ede395f1f4008b754028c840a120d41523e47e08f4aa6b6848978a98df01b"
OLD_TABLE = ART / "meth151_shared_table.bin"
OLD_TABLE_SHA = "926362655bcc71bf7bd1ceee944e56ba8f35ef124c8f60b6e0052d6525ddfc27"
OLD_HIT = ART / "meth151_context_hit.bin"
OLD_HIT_SHA = "c211bde3459b3a1df27250ee0e785cde412cba07fb763aeb519281588f54a629"
OLD_MISS = ART / "meth151_context_miss.bin"
OLD_MISS_SHA = "ecce0701d71201f9a48495adbc9814632c9f5b0929675a30daf2ffc92ac8682e"
TABLE_HEADER = struct.Struct("<8sII")
CONTEXT_HEADER = struct.Struct("<8sI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-out", type=Path, required=True)
    ap.add_argument("--new-hit-out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    assert all(not path.exists() for path in
               (args.table_out, args.new_hit_out, args.report))
    for path, sha in ((TABLE, TABLE_SHA), (RESULT, RESULT_SHA),
                      (OLD_TABLE, OLD_TABLE_SHA), (OLD_HIT, OLD_HIT_SHA),
                      (OLD_MISS, OLD_MISS_SHA)):
        assert M136.digest(path) == sha, path
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["candidate_table_sha256"] == TABLE_SHA
    record = json.loads(TABLE.read_text(encoding="utf-8"))
    old = R150.load_table()
    entries = [(r["token"], r["previous"], r["position"])
               for r in record["tuples"]]
    added = [row for row in record["tuples"] if row["new"]]
    assert entries == sorted(entries)
    assert len(entries) == 138 and len(added) == 63
    assert old <= set(entries)
    assert all((r["token"], r["previous"], r["position"]) not in old
               for r in added)
    table_data = np.asarray(entries, dtype="<u4")
    new_hit = np.asarray([(added[i % len(added)]["token"],
                           added[i % len(added)]["previous"],
                           added[i % len(added)]["position"])
                          for i in range(256)], dtype="<u4")
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_bytes(TABLE_HEADER.pack(b"M151TB01", 138, 32) +
                               table_data.tobytes())
    args.new_hit_out.write_bytes(CONTEXT_HEADER.pack(b"M151TX01", 256) +
                                 new_hit.tobytes())
    assert args.table_out.stat().st_size == TABLE_HEADER.size + 138 * 12
    assert args.new_hit_out.stat().st_size == CONTEXT_HEADER.size + 256 * 12
    report = {"experiment": "METH-166-native-table-fixtures",
              "meth162_table_sha256": TABLE_SHA,
              "meth162_result_sha256": RESULT_SHA,
              "old_table_sha256": OLD_TABLE_SHA,
              "old_hit_sha256": OLD_HIT_SHA,
              "old_miss_sha256": OLD_MISS_SHA,
              "new_table": {"path": str(args.table_out.resolve()),
                            "sha256": M136.digest(args.table_out),
                            "bytes": args.table_out.stat().st_size,
                            "entries": 138},
              "new_hit": {"path": str(args.new_hit_out.resolve()),
                          "sha256": M136.digest(args.new_hit_out),
                          "bytes": args.new_hit_out.stat().st_size,
                          "contexts": 256, "distinct": 63},
              "python_golden": {"old_shared_new_hit": False,
                                "new_shared_new_hit": True,
                                "content_child89_layer3": 899,
                                "shared_child89_layer3": 890}}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_sha256": M136.digest(args.report),
                      "new_table_sha256": report["new_table"]["sha256"],
                      "new_hit_sha256": report["new_hit"]["sha256"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
