#!/usr/bin/env python3
"""Replay one frozen tokenizer-defined chat-boundary route on exact traces."""

import argparse
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
OLD_TRACE = DOC / "meth161_source_route_trace.result.json"
OLD_TRACE_SHA = "b717b8db5f5565d377c3d2d043286c017ba98a661bedc9c6426ae660ce07b93c"
NEW_TRACE = DOC / "meth167_expanded_source_route_trace.result.json"
M162 = DOC / "meth162_postfailure_table_result.json"
M162_SHA = "d10ede395f1f4008b754028c840a120d41523e47e08f4aa6b6848978a98df01b"
M165 = DOC / "meth165_expanded_route_support_result.json"
M165_SHA = "b1c9bf6d0c3d74c7a282c657f069735299cb7cfaa0f92e52175669d475e5058c"
TABLE = DOC / "meth162_training_derived_shared_table.json"
TABLE_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
OLD_TOKENS = 887330
NEW_TOKENS = 442766
IM_START = 151644
IM_END = 151645
MAX_SECONDS = 15 * 60
MAX_RSS = 16 * (1 << 30)


def budget(start):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss}
    if row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-168 resource stop: {row}")
    return row


def path_for(item):
    path = Path(item["path"])
    return path if path.exists() else DOC / path.name


def array(trace, name, dtype, shape):
    item = trace["arrays"][name]
    path = path_for(item)
    assert M136.digest(path) == item["sha256"]
    assert path.stat().st_size == item["bytes"]
    result = np.load(path, mmap_mode="r", allow_pickle=False)
    assert result.dtype == dtype and result.shape == shape
    return result


def counts():
    return {"all": np.zeros((24, 12800), dtype=np.int64),
            "content": np.zeros((24, 12800), dtype=np.int64),
            "tokens": 0}


def summarize(cell):
    rows = []
    for li in range(24):
        all_count = cell["all"][li]
        content_count = cell["content"][li]
        parent = all_count.reshape(1280, 10).sum(axis=1)
        content_parent = content_count.reshape(1280, 10).sum(axis=1)
        row = M150.layer_summary(parent, all_count, content_parent,
                                 content_count, cell["tokens"])
        slots = content_count.reshape(1280, 10)[:, 1:].reshape(-1)
        active = slots[slots > 0]
        row["content_active_selection_p10"] = float(np.quantile(active, 0.10))
        row["content_active_selection_p50"] = float(np.quantile(active, 0.50))
        row["content_active_selection_p90"] = float(np.quantile(active, 0.90))
        row["content_slots_under_32_fraction"] = float((slots < 32).mean())
        row["all_slot_counts"] = all_count.tolist()
        row["content_slot_counts"] = content_count.tolist()
        rows.append(row)
    return rows


def check_old_replay(grand, shared, prior):
    old_all = np.bincount(grand.reshape(-1), minlength=12800)
    old_content = np.bincount(grand[~shared].reshape(-1), minlength=12800)
    assert np.array_equal(old_all, prior["all_slot_counts"])
    assert np.array_equal(old_content, prior["content_slot_counts"])


def replay(trace, tokens_count, table, prior_rows, cell_names, start):
    children = array(trace, "children", np.dtype("uint16"), (24, tokens_count, 4))
    token_ids = np.asarray(array(trace, "tokens", np.dtype("uint32"),
                                 (tokens_count,)), dtype=np.int64)
    previous = np.asarray(array(trace, "previous", np.dtype("uint32"),
                                (tokens_count,)), dtype=np.int64)
    positions = np.asarray(array(trace, "positions", np.dtype("uint16"),
                                 (tokens_count,)), dtype=np.int64)
    item = trace["arrays"]["offsets"]
    path = path_for(item)
    assert M136.digest(path) == item["sha256"]
    offsets = json.loads(path.read_text(encoding="utf-8"))
    assert offsets[0]["start"] == 0 and offsets[-1]["stop"] == tokens_count
    assert len(offsets) == 2 * trace["pairs"]
    raw_mask = np.zeros(tokens_count, dtype=np.bool_)
    for item in offsets:
        if item["kind"] == "raw":
            raw_mask[item["start"]:item["stop"]] = True
        else:
            assert item["kind"] == "chat"
    base_shared = R150.shared_mask(token_ids, previous, positions, table)
    delimiter = (token_ids == IM_START) | (token_ids == IM_END)
    candidate_shared = base_shared | delimiter
    assert np.all(base_shared <= candidate_shared)
    cells = {name: counts() for name in cell_names}
    if len(cell_names) == 1:
        cells[cell_names[0]]["tokens"] = tokens_count
    else:
        assert cell_names == ("new_raw", "new_chat")
        cells["new_raw"]["tokens"] = int(raw_mask.sum())
        cells["new_chat"]["tokens"] = int((~raw_mask).sum())
    attribution = {}
    for li in range(24):
        source = np.asarray(children[li], dtype=np.int64)
        assert np.all((source >= 0) & (source < 1280))
        old_grand, old_shared = R150.route(token_ids, previous, positions,
                                           source, li, table)
        assert np.array_equal(old_shared, base_shared)
        if len(cell_names) == 1:
            check_old_replay(old_grand, old_shared, prior_rows[li])
        else:
            check_old_replay(old_grand[raw_mask], old_shared[raw_mask],
                             prior_rows["new_raw_layers"][li])
            check_old_replay(old_grand[~raw_mask], old_shared[~raw_mask],
                             prior_rows["new_chat_layers"][li])
        grand = np.where(delimiter[:, None], source * 10, old_grand)
        assert np.array_equal(grand // 10, source)
        assert np.all((grand % 10 == 0) == candidate_shared[:, None])
        masks = {cell_names[0]: np.ones(tokens_count, dtype=np.bool_)} \
            if len(cell_names) == 1 else {"new_raw": raw_mask, "new_chat": ~raw_mask}
        for name, mask in masks.items():
            cell = cells[name]
            cell["all"][li] = np.bincount(grand[mask].reshape(-1), minlength=12800)
            cell["content"][li] = np.bincount(
                grand[mask & ~candidate_shared].reshape(-1), minlength=12800)
        if li == 20:
            hit = (source == 656) & (old_grand == 6564) & ~old_shared[:, None]
            selected = np.where(hit)
            attribution = {"original_slot_selections": int(hit.sum()),
                           "delimiter_selections": int(delimiter[selected[0]].sum()),
                           "candidate_new_locals": np.bincount(
                               (grand[selected] % 10).reshape(-1), minlength=10).tolist(),
                           "im_start_selections": int((token_ids[selected[0]] == IM_START).sum()),
                           "im_end_selections": int((token_ids[selected[0]] == IM_END).sum())}
        print(json.dumps({"trace": cell_names, "layer": li,
                          "budget": budget(start)}), flush=True)
    return cells, {"raw_input_tokens": int(raw_mask.sum()),
                   "chat_input_tokens": int((~raw_mask).sum()),
                   "old_structural_positions": int(base_shared.sum()),
                   "candidate_structural_positions": int(candidate_shared.sum()),
                   "newly_shared_positions": int((delimiter & ~base_shared).sum()),
                   "original_worst_slot": attribution}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-trace-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    for path, sha in ((OLD_TRACE, OLD_TRACE_SHA), (NEW_TRACE, args.new_trace_sha),
                      (M162, M162_SHA), (M165, M165_SHA), (TABLE, TABLE_SHA)):
        assert M136.digest(path) == sha, path
    old_trace = json.loads(OLD_TRACE.read_text(encoding="utf-8"))
    new_trace = json.loads(NEW_TRACE.read_text(encoding="utf-8"))
    prior_old = json.loads(M162.read_text(encoding="utf-8"))
    prior_new = json.loads(M165.read_text(encoding="utf-8"))
    assert old_trace["decision"] == "valid_source_trace_for_offline_postfailure_diagnostics"
    assert all(old_trace["exact_meth159_all_content_histogram_matches"])
    assert new_trace["decision"] == "valid_expanded_source_trace_for_offline_postfailure_diagnostics"
    assert all(all(rows) for rows in
               new_trace["exact_meth165_raw_chat_combined_histogram_matches"].values())
    assert old_trace["input_tokens"] == OLD_TOKENS
    assert new_trace["input_tokens"] == NEW_TOKENS
    assert prior_old["candidate_table_sha256"] == TABLE_SHA
    assert prior_new["candidate_table_sha256"] == TABLE_SHA
    table_record = json.loads(TABLE.read_text(encoding="utf-8"))
    table = {(r["token"], r["previous"], r["position"])
             for r in table_record["tuples"]}
    assert len(table) == 138
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M136.M42.MODEL, revision=M136.M42.REV,
                                         local_files_only=True)
    assert tok.convert_tokens_to_ids("<|im_start|>") == IM_START
    assert tok.convert_tokens_to_ids("<|im_end|>") == IM_END
    assert tok.convert_tokens_to_ids("<|endoftext|>") == 151643
    for token, previous, position in ((IM_END, 73594, 152),
                                      (IM_START, 198, 154)):
        grand, shared = R150.route(np.array([token]), np.array([previous]),
                                    np.array([position]),
                                    np.array([[656, 656, 656, 656]]), 20, table)
        assert int(grand[0, 0]) == 6564 and not bool(shared[0])
        assert int(np.where(token in (IM_START, IM_END), 6560, grand[0, 0])) == 6560
    assert R150.golden(table)["content"] == 899
    started = time.monotonic()
    old_cells, old_detail = replay(old_trace, OLD_TOKENS, table,
                                    prior_old["layers"], ("old",), started)
    new_cells, new_detail = replay(new_trace, NEW_TOKENS, table,
                                    prior_new, ("new_raw", "new_chat"), started)
    cells = {**old_cells, **new_cells}
    pooled = counts()
    pooled["tokens"] = OLD_TOKENS + NEW_TOKENS
    for cell in cells.values():
        pooled["all"] += cell["all"]
        pooled["content"] += cell["content"]
    cells["pooled"] = pooled
    assert pooled["tokens"] == prior_new["combined_tokens"]
    rows = {name: summarize(cell) for name, cell in cells.items()}
    pooled_rows = rows["pooled"]
    gates = {
        "pooled_coverage": all(r["content"]["coverage"] >= 10000
                               for r in pooled_rows),
        "pooled_active_median": all(r["content_active_selection_p50"] >= 50
                                    for r in pooled_rows),
        "pooled_under_32": all(r["content_slots_under_32_fraction"] <= 0.50
                               for r in pooled_rows),
        "pooled_load_ratio": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                                 for r in pooled_rows),
        "pooled_hot_share": all(r["content_hot_parent_worst_grandchild_share"] <= 0.25
                                for r in pooled_rows),
        "new_raw_load_ratio": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                                  for r in rows["new_raw"]),
        "new_raw_hot_share": all(r["content_hot_parent_worst_grandchild_share"] <= 0.25
                                 for r in rows["new_raw"]),
        "new_raw_hot_present": all(r["content_hot_parent_count"] > 0
                                   for r in rows["new_raw"]),
        "new_chat_load_ratio": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                                   for r in rows["new_chat"]),
        "new_chat_hot_share": all(r["content_hot_parent_worst_grandchild_share"] <= 0.25
                                  for r in rows["new_chat"]),
        "new_chat_hot_present": all(r["content_hot_parent_count"] > 0
                                    for r in rows["new_chat"]),
        "structural_share": old_detail["candidate_structural_positions"] +
                            new_detail["candidate_structural_positions"] <=
                            0.25 * pooled["tokens"]}
    result = {"experiment": "METH-168-chat-boundary-route-diagnostic",
              "old_trace_sha256": OLD_TRACE_SHA,
              "new_trace_sha256": args.new_trace_sha,
              "meth162_result_sha256": M162_SHA,
              "meth165_result_sha256": M165_SHA,
              "candidate_table_sha256": TABLE_SHA,
              "chat_boundary_tokens": {"im_start": IM_START, "im_end": IM_END},
              "old_trace_detail": old_detail,
              "new_trace_detail": new_detail,
              "input_tokens": pooled["tokens"],
              "structural_selections_per_layer": 4 * (
                  old_detail["candidate_structural_positions"] +
                  new_detail["candidate_structural_positions"]),
              "cells": rows, "gates": gates,
              "decision": "postfailure_in_sample_candidate_pass_external_gates_pending"
                          if all(gates.values()) else "postfailure_in_sample_candidate_fail",
              "runtime": budget(started)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "old_slot": old_detail["original_worst_slot"],
                      "new_slot": new_detail["original_worst_slot"],
                      "pooled_min_median": min(r["content_active_selection_p50"]
                          for r in pooled_rows),
                      "pooled_worst_hot_share": max(
                          r["content_hot_parent_worst_grandchild_share"]
                          for r in pooled_rows),
                      "sha256": M136.digest(args.out),
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
