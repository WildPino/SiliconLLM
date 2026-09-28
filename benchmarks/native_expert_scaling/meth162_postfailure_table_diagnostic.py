#!/usr/bin/env python3
"""Attribute METH-159 hot slots and replay one fixed recurrence-table repair."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import psutil

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth150_shared_route_screen as M150
import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TRACE = DOC / "meth161_source_route_trace.result.json"
M159 = DOC / "meth159_tenfold_route_support_result.json"
M159_SHA = "48074f7b404b409ba177acbbee97c196c1e2aa33f170930267fd25489edba509"
TOKENS = 887330
THRESHOLD = 32
MAX_SECONDS = 15 * 60
MAX_RSS = 16 * (1 << 30)


def budget(start):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss}
    if row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-162 resource stop: {row}")
    return row


def load_array(record, name, dtype, shape):
    item = record["arrays"][name]
    path = Path(item["path"])
    assert M136.digest(path) == item["sha256"]
    assert path.stat().st_size == item["bytes"]
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    assert array.dtype == dtype and array.shape == shape
    return array


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-sha", required=True)
    ap.add_argument("--table-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.table_out.exists() and not args.out.exists()
    assert M136.digest(TRACE) == args.trace_sha
    assert M136.digest(M159) == M159_SHA
    source = json.loads(TRACE.read_text(encoding="utf-8"))
    prior = json.loads(M159.read_text(encoding="utf-8"))
    assert source["decision"] == "valid_source_trace_for_offline_postfailure_diagnostics"
    assert all(source["exact_meth159_all_content_histogram_matches"])
    assert source["input_tokens"] == prior["input_tokens"] == TOKENS
    assert source["meth159_result_sha256"] == M159_SHA
    assert prior["decision"] == "tenfold_route_support_fail"
    started = time.monotonic()
    children = load_array(source, "children", np.dtype("uint16"), (24, TOKENS, 4))
    tokens = load_array(source, "tokens", np.dtype("uint32"), (TOKENS,))
    previous = load_array(source, "previous", np.dtype("uint32"), (TOKENS,))
    positions = load_array(source, "positions", np.dtype("uint16"), (TOKENS,))
    offsets_path = Path(source["arrays"]["offsets"]["path"])
    assert M136.digest(offsets_path) == source["arrays"]["offsets"]["sha256"]
    offsets = json.loads(offsets_path.read_text(encoding="utf-8"))
    assert len(offsets) == 5120 and offsets[0]["start"] == 0
    assert offsets[-1]["stop"] == TOKENS
    old_table = R150.load_table()
    assert len(old_table) == 75
    frequencies = Counter(zip(tokens.tolist(), previous.tolist(), positions.tolist()))
    additions = {key for key, count in frequencies.items()
                 if count >= THRESHOLD and key not in old_table}
    table = old_table | additions
    assert len(table) == len(old_table) + len(additions)
    table_record = {"experiment": "METH-162-training-derived-shared-context-table",
                    "source_trace_result_sha256": args.trace_sha,
                    "old_table_sha256": R150.TABLE_SHA,
                    "recurrence_threshold": THRESHOLD,
                    "old_entries": len(old_table), "new_entries": len(additions),
                    "total_entries": len(table),
                    "added_input_positions": sum(frequencies[key] for key in additions),
                    "tuples": [{"token": key[0], "previous": key[1],
                                "position": key[2], "training_count": frequencies[key],
                                "new": key in additions} for key in sorted(table)]}
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_text(json.dumps(table_record, indent=2) + "\n", encoding="utf-8")
    table_sha = M136.digest(args.table_out)
    budget(started)

    token_ids = np.asarray(tokens, dtype=np.int64)
    previous_ids = np.asarray(previous, dtype=np.int64)
    local_positions = np.asarray(positions, dtype=np.int64)
    layers = []
    old_exact = []
    worst_parent_attribution = None
    for li in range(24):
        source_ids = np.asarray(children[li], dtype=np.int64)
        assert np.all((source_ids >= 0) & (source_ids < 1280))
        old_grand, old_shared = R150.route(token_ids, previous_ids,
                                           local_positions, source_ids, li, old_table)
        all_old = np.bincount(old_grand.reshape(-1), minlength=12800)
        content_old = np.bincount(old_grand[~old_shared].reshape(-1), minlength=12800)
        exact = (np.array_equal(all_old, prior["layers"][li]["all_slot_counts"]) and
                 np.array_equal(content_old, prior["layers"][li]["content_slot_counts"]))
        old_exact.append(bool(exact))
        assert exact, f"old route replay mismatch at layer {li}"
        if li == 2:
            mask = (source_ids == 314) & (old_grand == 3141) & ~old_shared[:, None]
            locations = np.where(mask)
            tuples = Counter((int(tokens[pos]), int(previous[pos]), int(positions[pos]))
                             for pos in locations[0])
            assert sum(tuples.values()) == 186
            worst_parent_attribution = {
                "layer": 2, "source_child": 314, "old_content_local": 1,
                "old_slot_selections": 186,
                "newly_shared_selections": sum(count for key, count in tuples.items()
                                               if key in additions),
                "top_contexts": [{"token": key[0], "previous": key[1],
                                  "position": key[2], "selected_count": count,
                                  "global_training_count": frequencies[key],
                                  "added_to_shared": key in additions}
                                 for key, count in tuples.most_common(20)]}
        grand, shared = R150.route(token_ids, previous_ids, local_positions,
                                    source_ids, li, table)
        assert np.array_equal(grand // 10, source_ids)
        assert int(grand.size) == 4 * TOKENS
        parent = np.bincount(source_ids.reshape(-1), minlength=1280)
        all_count = np.bincount(grand.reshape(-1), minlength=12800)
        content_parent = np.bincount(source_ids[~shared].reshape(-1), minlength=1280)
        content_grand = np.bincount(grand[~shared].reshape(-1), minlength=12800)
        row = M150.layer_summary(parent, all_count, content_parent,
                                 content_grand, TOKENS)
        slots = content_grand.reshape(1280, 10)[:, 1:].reshape(-1)
        active = slots[slots > 0]
        row["content_active_selection_p10"] = float(np.quantile(active, 0.10))
        row["content_active_selection_p50"] = float(np.quantile(active, 0.50))
        row["content_active_selection_p90"] = float(np.quantile(active, 0.90))
        row["content_slots_under_32_fraction"] = float((slots < 32).mean())
        row["content_slot_counts"] = content_grand.tolist()
        row["all_slot_counts"] = all_count.tolist()
        layers.append(row)
        print(json.dumps({"layer": li, "content_coverage": row["content"]["coverage"],
                          "active_median": row["content_active_selection_p50"],
                          "hot_share": row["content_hot_parent_worst_grandchild_share"],
                          "budget": budget(started)}), flush=True)
    assert all(old_exact) and worst_parent_attribution is not None
    gates = {"content_coverage": min(row["content"]["coverage"] for row in layers) >= 10000,
             "active_content_median": min(row["content_active_selection_p50"] for row in layers) >= 50,
             "content_under_32_fraction": max(row["content_slots_under_32_fraction"] for row in layers) <= 0.50,
             "content_load_ratio": max(row["content_to_own_parent_max_load_ratio"] for row in layers) <= 1.25,
             "content_hot_parent_share": max(row["content_hot_parent_worst_grandchild_share"] for row in layers) <= 0.25}
    result = {"experiment": "METH-162-postfailure-structural-table-diagnostic",
              "source_trace_result_sha256": args.trace_sha,
              "meth159_result_sha256": M159_SHA,
              "old_table_sha256": R150.TABLE_SHA,
              "candidate_table_sha256": table_sha,
              "fixed_training_input_tokens": TOKENS,
              "old_route_histograms_replayed_exactly": old_exact,
              "added_contexts": len(additions),
              "added_input_positions": table_record["added_input_positions"],
              "original_worst_parent_attribution": worst_parent_attribution,
              "layers": layers, "candidate_in_sample_gates": gates,
              "runtime": budget(started),
              "decision": "postfailure_in_sample_diagnostic_only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "result_sha256": M136.digest(args.out),
                      "table_sha256": table_sha,
                      "added_contexts": len(additions),
                      "worst_parent_newly_shared": worst_parent_attribution["newly_shared_selections"],
                      "candidate_in_sample_gates": gates,
                      "min_content_coverage": min(row["content"]["coverage"] for row in layers),
                      "min_active_median": min(row["content_active_selection_p50"] for row in layers),
                      "worst_hot_share": max(row["content_hot_parent_worst_grandchild_share"] for row in layers),
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
