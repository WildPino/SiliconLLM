#!/usr/bin/env python3
"""Freeze threshold-8 shared tuples and replay on exact source traces."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import psutil

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth150_shared_structure_route as R150
import meth168_chat_boundary_route_diagnostic as M168


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TRACE = (
    ("training_old", "meth161_source_route_trace.result.json",
     "b717b8db5f5565d377c3d2d043286c017ba98a661bedc9c6426ae660ce07b93c"),
    ("training_expanded", "meth167_expanded_source_route_trace.result.json",
     "9ed2e2a008fbde79c6ba2032b830ab9d66c7400e19f0fda8e918e8361d1ff0ff"),
    ("prospective", "meth171_composed_hot_parent_trace.result.json",
     "2627fe6b272b719031bb3695ffed51cdf4bfc922031772ac4f4181505b00190c"))
M168_PATH = DOC / "meth168_chat_boundary_route_result.json"
M168_SHA = "c5f282676fb4a8a3f3b1430d1fa649302dcd46bc6636a875cb69723ba6573973"
M169_PATH = DOC / "meth169_fresh_boundary_route_result.json"
M169_SHA = "1cf6bad51c2625fee83201242347697b1998a4ca11d48cda5f31715931bfb792"
TABLE_138 = DOC / "meth162_training_derived_shared_table.json"
TABLE_138_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
THRESHOLD = 8
MAX_SECONDS = 15 * 60
MAX_RSS = 16 * (1 << 30)


def budget(start):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss}
    if row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-172 resource stop: {row}")
    return row


def path_for(item):
    path = Path(item["path"])
    return path if path.exists() else DOC / path.name


def empty(tokens=0):
    return {"all": np.zeros((24, 12800), dtype=np.int64),
            "content": np.zeros((24, 12800), dtype=np.int64),
            "tokens": tokens, "added_positions": 0}


def combine(*cells):
    out = empty()
    for cell in cells:
        out["all"] += cell["all"]
        out["content"] += cell["content"]
        out["tokens"] += cell["tokens"]
        out["added_positions"] += cell["added_positions"]
    return out


def exact(actual, expected, name, layer):
    assert np.array_equal(actual["all"], expected["all_slot_counts"]), (name, layer, "all")
    assert np.array_equal(actual["content"], expected["content_slot_counts"]), (
        name, layer, "content")


def scan(name, trace, old_table, table, prior168, prior169, start):
    n = trace["input_tokens"]
    children = M168.array(trace, "children", np.dtype("uint16"), (24, n, 4))
    token_ids = np.asarray(M168.array(trace, "tokens", np.dtype("uint32"), (n,)),
                           dtype=np.int64)
    previous = np.asarray(M168.array(trace, "previous", np.dtype("uint32"), (n,)),
                          dtype=np.int64)
    positions = np.asarray(M168.array(trace, "positions", np.dtype("uint16"), (n,)),
                           dtype=np.int64)
    offset_item = trace["arrays"]["offsets"]
    offset_path = path_for(offset_item)
    assert M136.digest(offset_path) == offset_item["sha256"]
    offsets = json.loads(offset_path.read_text(encoding="utf-8"))
    assert offsets[0]["start"] == 0 and offsets[-1]["stop"] == n
    raw = np.zeros(n, dtype=np.bool_)
    for item in offsets:
        if item["kind"] == "raw":
            raw[item["start"]:item["stop"]] = True
        else:
            assert item["kind"] == "chat"
    masks = {"combined": np.ones(n, dtype=np.bool_)} if name == "training_old" else {
        "raw": raw, "chat": ~raw}
    cells = {kind: empty(int(mask.sum())) for kind, mask in masks.items()}
    delimiter = (token_ids == 151644) | (token_ids == 151645)
    old_shared = R150.shared_mask(token_ids, previous, positions, old_table) | delimiter
    new_shared = R150.shared_mask(token_ids, previous, positions, table) | delimiter
    assert np.all(old_shared <= new_shared)
    added = new_shared & ~old_shared
    for kind, mask in masks.items():
        cells[kind]["added_positions"] = int((added & mask).sum())
    for li in range(24):
        source = np.asarray(children[li], dtype=np.int64)
        assert np.all((source >= 0) & (source < 1280))
        base, base_shared = R150.route(
            token_ids, previous, positions, source, li, old_table)
        assert np.all(base_shared <= old_shared)
        old_grand = np.where(old_shared[:, None], source * 10, base)
        new_grand = np.where(new_shared[:, None], source * 10, base)
        assert np.array_equal(old_grand // 10, source)
        assert np.array_equal(new_grand // 10, source)
        for kind, mask in masks.items():
            old_all = np.bincount(old_grand[mask].reshape(-1), minlength=12800)
            old_content = np.bincount(
                old_grand[mask & ~old_shared].reshape(-1), minlength=12800)
            target = (prior168["cells"]["old"][li]
                      if name == "training_old" else
                      prior168["cells"][f"new_{kind}"][li]
                      if name == "training_expanded" else
                      prior169["cells"][kind]["candidate_layers"][li])
            exact({"all": old_all, "content": old_content}, target,
                  f"{name}:{kind}", li)
            cells[kind]["all"][li] = np.bincount(
                new_grand[mask].reshape(-1), minlength=12800)
            cells[kind]["content"][li] = np.bincount(
                new_grand[mask & ~new_shared].reshape(-1), minlength=12800)
        if li % 4 == 3:
            print(json.dumps({"trace": name, "layer": li,
                              "budget": budget(start)}), flush=True)
    return cells


def gates(rows, coverage, structural_cap, added_fraction, support=False):
    return {"coverage": all(r["content"]["coverage"] >= coverage for r in rows),
            "load": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                        for r in rows),
            "hot_share": all(r["content_hot_parent_worst_grandchild_share"] <= 0.25
                             for r in rows),
            "hot_present": all(r["content_hot_parent_count"] > 0 for r in rows),
            "structural_share": all(r["structural_selections"] <=
                                    structural_cap * r["total"]["selections"]
                                    for r in rows),
            "added_structural": added_fraction <= 0.05,
            **({"median": all(r["content_active_selection_p50"] >= 50 for r in rows),
                "under_32": all(r["content_slots_under_32_fraction"] <= 0.50
                                for r in rows)} if support else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.table_out.exists() and not args.out.exists()
    for path, sha in ((M168_PATH, M168_SHA), (M169_PATH, M169_SHA),
                      (TABLE_138, TABLE_138_SHA)):
        assert M136.digest(path) == sha
    records = {}
    for name, filename, sha in TRACE:
        path = DOC / filename
        assert M136.digest(path) == sha
        records[name] = json.loads(path.read_text(encoding="utf-8"))
    prior168 = json.loads(M168_PATH.read_text(encoding="utf-8"))
    prior169 = json.loads(M169_PATH.read_text(encoding="utf-8"))
    assert prior169["decision"] == "prospective_route_fail"
    old_table_record = json.loads(TABLE_138.read_text(encoding="utf-8"))
    old_table = {(r["token"], r["previous"], r["position"])
                 for r in old_table_record["tuples"]}
    assert len(old_table) == 138
    started = time.monotonic()
    first = records["training_old"]
    n = first["input_tokens"]
    tokens = M168.array(first, "tokens", np.dtype("uint32"), (n,))
    previous = M168.array(first, "previous", np.dtype("uint32"), (n,))
    positions = M168.array(first, "positions", np.dtype("uint16"), (n,))
    frequencies = Counter(zip(tokens.tolist(), previous.tolist(), positions.tolist()))
    table = R150.load_table() | {key for key, count in frequencies.items()
                                 if count >= THRESHOLD}
    assert len(table) == 1685 and old_table <= table
    assert sum(frequencies[key] for key in table) == 187581
    assert R150.golden(table) == {"content": 899, "structural": 890}
    table_record = {"experiment": "METH-172-threshold-8-shared-table",
                    "source_trace_sha256": TRACE[0][2],
                    "prior_table_sha256": TABLE_138_SHA,
                    "recurrence_threshold": THRESHOLD,
                    "entries": len(table),
                    "training_positions": sum(frequencies[key] for key in table),
                    "tuples": [{"token": t, "previous": p, "position": pos,
                                "training_count": frequencies[(t, p, pos)]}
                               for t, p, pos in sorted(table)]}
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_text(json.dumps(table_record, indent=2) + "\n",
                              encoding="utf-8")
    table_sha = M136.digest(args.table_out)
    budget(started)

    groups = {}
    for name in records:
        groups[name] = scan(name, records[name], old_table, table,
                            prior168, prior169, started)
    train = combine(groups["training_old"]["combined"],
                    groups["training_expanded"]["raw"],
                    groups["training_expanded"]["chat"])
    fresh = combine(groups["prospective"]["raw"],
                    groups["prospective"]["chat"])
    pooled = combine(train, fresh)
    assert train["tokens"] == 1330096 and fresh["tokens"] == 176710
    assert pooled["tokens"] == 1506806
    cells = {"training": train, "new_raw": groups["prospective"]["raw"],
             "new_chat": groups["prospective"]["chat"],
             "new_raw_chat": fresh, "composed": pooled}
    summaries = {name: M168.summarize(cell) for name, cell in cells.items()}
    checks = {
        "training": gates(summaries["training"], 10000, 0.25,
                          train["added_positions"] / train["tokens"], True),
        "new_raw": gates(summaries["new_raw"], 7000, 0.25,
                         cells["new_raw"]["added_positions"] /
                         cells["new_raw"]["tokens"]),
        "new_chat": gates(summaries["new_chat"], 7000, 0.35,
                          cells["new_chat"]["added_positions"] /
                          cells["new_chat"]["tokens"]),
        "new_raw_chat": gates(summaries["new_raw_chat"], 7000, 0.25,
                              fresh["added_positions"] / fresh["tokens"]),
        "composed": gates(summaries["composed"], 10000, 0.25,
                          pooled["added_positions"] / pooled["tokens"], True)}
    passed = all(all(x.values()) for x in checks.values())
    result = {"experiment": "METH-172-low-recurrence-route-replay",
              "trace_sha256": {name: sha for name, _, sha in TRACE},
              "meth168_result_sha256": M168_SHA,
              "meth169_result_sha256": M169_SHA,
              "prior_table_sha256": TABLE_138_SHA,
              "candidate_table_sha256": table_sha,
              "table_entries": len(table),
              "cells": {name: {"tokens": cell["tokens"],
                               "newly_shared_input_positions": cell["added_positions"],
                               "layers": summaries[name]}
                        for name, cell in cells.items()},
              "gates": checks,
              "decision": "postfailure_in_sample_candidate_pass_fresh_source_pending"
                          if passed else "postfailure_in_sample_candidate_fail",
              "runtime": budget(started)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": checks,
                      "table_sha256": table_sha,
                      "result_sha256": M136.digest(args.out),
                      "summary": {name: {"tokens": cell["tokens"],
                          "added_positions": cell["added_positions"],
                          "min_coverage": min(r["content"]["coverage"]
                                              for r in summaries[name]),
                          "min_median": min(r["content_active_selection_p50"]
                                            for r in summaries[name]),
                          "worst_hot_share": max(
                              r["content_hot_parent_worst_grandchild_share"]
                              for r in summaries[name])}
                          for name, cell in cells.items()},
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
