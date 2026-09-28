#!/usr/bin/env python3
"""Export SHA-bound binary lookup fixtures for conditional METH-174."""

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
ROUTE = DOC / "meth173_fresh_recurrence_route_result.json"
TABLE = DOC / "meth172_threshold8_shared_table.json"
TABLE_SHA = "e89ff2cecbf887b85b5e361d47a8b918d8ccb93bd5c385a580ba993b93ac72db"
BASE_JSON = DOC / "meth162_training_derived_shared_table.json"
BASE_JSON_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
BASE_BINARY = ART / "meth166_shared_table.bin"
BASE_BINARY_SHA = "e7a270e0e77fb2c7bca711a0932c24224dc7f6e060611cc3a942ed7593cf38d8"
OLD_HIT = ART / "meth151_context_hit.bin"
OLD_HIT_SHA = "c211bde3459b3a1df27250ee0e785cde412cba07fb763aeb519281588f54a629"
TABLE_HEADER = struct.Struct("<8sII")
CONTEXT_HEADER = struct.Struct("<8sI")


def write_contexts(path, rows):
    data = np.asarray(rows, dtype="<u4")
    assert data.shape == (256, 3)
    path.write_bytes(CONTEXT_HEADER.pack(b"M151TX01", 256) + data.tobytes())
    return {"path": str(path.resolve()), "sha256": M136.digest(path),
            "bytes": path.stat().st_size, "contexts": 256}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-result-sha", required=True)
    ap.add_argument("--table-out", type=Path, required=True)
    ap.add_argument("--new-hit-out", type=Path, required=True)
    ap.add_argument("--delimiter-out", type=Path, required=True)
    ap.add_argument("--miss-out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    targets = (args.table_out, args.new_hit_out, args.delimiter_out,
               args.miss_out, args.report)
    assert all(not path.exists() for path in targets)
    for path, sha in ((ROUTE, args.route_result_sha), (TABLE, TABLE_SHA),
                      (BASE_JSON, BASE_JSON_SHA), (BASE_BINARY, BASE_BINARY_SHA),
                      (OLD_HIT, OLD_HIT_SHA)):
        assert M136.digest(path) == sha, path
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    assert route["decision"] == "prospective_low_recurrence_route_pass_native_cpu_pending"
    assert route["candidate_table_sha256"] == TABLE_SHA
    assert route["baseline_table_sha256"] == BASE_JSON_SHA
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    base = json.loads(BASE_JSON.read_text(encoding="utf-8"))
    entries = [(r["token"], r["previous"], r["position"])
               for r in table["tuples"]]
    base_entries = {(r["token"], r["previous"], r["position"])
                    for r in base["tuples"]}
    assert len(entries) == 1685 and entries == sorted(entries)
    assert len(base_entries) == 138 and base_entries <= set(entries)
    added = [key for key in entries
             if key not in base_entries and key[0] not in (151644, 151645)]
    assert (11, 198, 56) in added and len(added) >= 256
    first = (11, 198, 56)
    new_hit = [first] + [key for key in added if key != first][:255]
    delimiter = [(151644 if i % 2 == 0 else 151645, 123, 67)
                 for i in range(256)]
    miss = [(123, 45, 67)] + [(100000 + i, 45, 67) for i in range(255)]
    assert all(key not in base_entries for key in new_hit)
    assert all(key not in set(entries) for key in delimiter + miss)
    assert all(key[0] not in (151644, 151645) for key in miss)
    assert R150.golden(set(entries)) == {"content": 899, "structural": 890}
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    candidate_binary = np.asarray(entries, dtype="<u4")
    args.table_out.write_bytes(
        TABLE_HEADER.pack(b"M151TB01", 1685, 8) + candidate_binary.tobytes())
    assert args.table_out.stat().st_size == TABLE_HEADER.size + 1685 * 12
    report = {"experiment": "METH-174-native-low-recurrence-fixtures",
              "route_result_sha256": args.route_result_sha,
              "candidate_table_json_sha256": TABLE_SHA,
              "baseline_table_json_sha256": BASE_JSON_SHA,
              "baseline_table_binary_sha256": BASE_BINARY_SHA,
              "baseline_hit_sha256": OLD_HIT_SHA,
              "candidate_table_binary": {
                  "path": str(args.table_out.resolve()),
                  "sha256": M136.digest(args.table_out),
                  "bytes": args.table_out.stat().st_size, "entries": 1685},
              "new_hit": write_contexts(args.new_hit_out, new_hit),
              "delimiter": write_contexts(args.delimiter_out, delimiter),
              "miss": write_contexts(args.miss_out, miss),
              "golden": {"new_hit_first": list(first),
                         "delimiter_first": list(delimiter[0]),
                         "content_miss_first": list(miss[0]),
                         "shared_child89_layer3": 890,
                         "content_child89_layer3": 899}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_sha256": M136.digest(args.report),
                      "candidate_table_binary_sha256":
                      report["candidate_table_binary"]["sha256"],
                      "fixture_sha256": {name: report[name]["sha256"]
                                         for name in ("new_hit", "delimiter", "miss")}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
