#!/usr/bin/env python3
"""Attribute METH-169's composed layer-3 hot parent from exact traces."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TRACE_NAMES = (
    ("training_old", "meth161_source_route_trace.result.json",
     "b717b8db5f5565d377c3d2d043286c017ba98a661bedc9c6426ae660ce07b93c"),
    ("training_expanded", "meth167_expanded_source_route_trace.result.json",
     "9ed2e2a008fbde79c6ba2032b830ab9d66c7400e19f0fda8e918e8361d1ff0ff"),
    ("prospective", "meth171_composed_hot_parent_trace.result.json",
     "2627fe6b272b719031bb3695ffed51cdf4bfc922031772ac4f4181505b00190c"))
TABLE = DOC / "meth162_training_derived_shared_table.json"
TABLE_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
LAYER = 3
PARENT = 215
LOCAL = 3


def array(item):
    path = Path(item["path"])
    if not path.exists():
        path = DOC / path.name
    assert M136.digest(path) == item["sha256"]
    assert path.stat().st_size == item["bytes"]
    return np.load(path, mmap_mode="r", allow_pickle=False)


def analyze(name, record, table):
    children = array(record["arrays"]["children"])
    tokens = np.asarray(array(record["arrays"]["tokens"]), dtype=np.int64)
    previous = np.asarray(array(record["arrays"]["previous"]), dtype=np.int64)
    positions = np.asarray(array(record["arrays"]["positions"]), dtype=np.int64)
    offsets_item = record["arrays"]["offsets"]
    offsets_path = Path(offsets_item["path"])
    if not offsets_path.exists():
        offsets_path = DOC / offsets_path.name
    assert M136.digest(offsets_path) == offsets_item["sha256"]
    offsets = json.loads(offsets_path.read_text(encoding="utf-8"))
    n = record["input_tokens"]
    assert children.shape == (24, n, 4)
    assert tokens.size == previous.size == positions.size == n
    source = np.asarray(children[LAYER], dtype=np.int64)
    base_grand, base_shared = R150.route(
        tokens, previous, positions, source, LAYER, table)
    shared = base_shared | (tokens == 151644) | (tokens == 151645)
    grand = np.where(shared[:, None], source * 10, base_grand)
    parent_mask = (source == PARENT) & ~shared[:, None]
    selected_mask = grand == PARENT * 10 + LOCAL
    assert np.all(selected_mask <= parent_mask)
    parent_count = int(parent_mask.sum())
    selected_count = int(selected_mask.sum())
    per_kind = {"raw": {"parent": 0, "local": 0},
                "chat": {"parent": 0, "local": 0}}
    tuples = Counter()
    entries = []
    for item in offsets:
        first, stop = item["start"], item["stop"]
        kind = item["kind"]
        per_kind[kind]["parent"] += int(parent_mask[first:stop].sum())
        per_kind[kind]["local"] += int(selected_mask[first:stop].sum())
        at, choice = np.nonzero(selected_mask[first:stop])
        for relative, choice_id in zip(at.tolist(), choice.tolist()):
            i = first + relative
            triple = (int(tokens[i]), int(previous[i]), int(positions[i]))
            tuples[triple] += 1
            entries.append({"kind": kind, "pair": item["pair"],
                            "source_row": item.get("source_row"),
                            "input_index": int(i), "choice": int(choice_id),
                            "token": triple[0], "previous": triple[1],
                            "position": triple[2]})
    assert sum(v["parent"] for v in per_kind.values()) == parent_count
    assert sum(v["local"] for v in per_kind.values()) == selected_count
    return {"input_tokens": n, "parent_selections": parent_count,
            "local_selections": selected_count, "per_kind": per_kind,
            "tuple_counts": [{"token": t, "previous": p, "position": pos,
                              "selections": count}
                             for (t, p, pos), count in tuples.most_common()],
            "selected_occurrences": entries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert M136.digest(TABLE) == TABLE_SHA
    table_record = json.loads(TABLE.read_text(encoding="utf-8"))
    table = {(r["token"], r["previous"], r["position"])
             for r in table_record["tuples"]}
    assert len(table) == 138
    cells = {}
    for name, filename, sha in TRACE_NAMES:
        path = DOC / filename
        assert M136.digest(path) == sha
        cells[name] = analyze(name, json.loads(path.read_text(encoding="utf-8")), table)
    old, expanded, new = (cells[k] for k in
                          ("training_old", "training_expanded", "prospective"))
    assert old["parent_selections"] + expanded["parent_selections"] == 244
    assert old["local_selections"] + expanded["local_selections"] == 62
    assert new["parent_selections"] == 58 and new["local_selections"] == 20
    assert sum(c["parent_selections"] for c in cells.values()) == 302
    assert sum(c["local_selections"] for c in cells.values()) == 82
    result = {"experiment": "METH-171-composed-hot-parent-attribution",
              "layer": LAYER, "source_child": PARENT, "content_local": LOCAL,
              "trace_sha256": {name: sha for name, _, sha in TRACE_NAMES},
              "table_sha256": TABLE_SHA,
              "composed": {"parent_selections": 302,
                           "local_selections": 82,
                           "grandchild_share": 82 / 302},
              "cells": cells,
              "decision": "exact_composed_failure_attributed_no_route_approval"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M136.digest(args.out),
                      "counts": {name: {"parent": cell["parent_selections"],
                                        "local": cell["local_selections"],
                                        "top_tuples": cell["tuple_counts"][:8]}
                                 for name, cell in cells.items()}}, indent=2))


if __name__ == "__main__":
    main()
