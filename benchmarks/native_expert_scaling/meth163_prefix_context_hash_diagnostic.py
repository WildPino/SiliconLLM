#!/usr/bin/env python3
"""Offline full-prefix content hash diagnostic on the bound source trace."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import psutil

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth142_token_hash_route_screen as M142
import meth150_shared_route_screen as M150
import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TRACE = DOC / "meth161_source_route_trace.result.json"
M159 = DOC / "meth159_tenfold_route_support_result.json"
M159_SHA = "48074f7b404b409ba177acbbee97c196c1e2aa33f170930267fd25489edba509"
TOKENS = 887330
MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
PREFIX_MIX = np.uint64(0xDB4F0B9175AE2165)
MAX_SECONDS = 15 * 60
MAX_RSS = 16 * (1 << 30)


def budget(start):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss}
    if row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-163 resource stop: {row}")
    return row


def load_array(result, name, dtype, shape):
    item = result["arrays"][name]
    path = Path(item["path"])
    assert M136.digest(path) == item["sha256"]
    assert path.stat().st_size == item["bytes"]
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    assert array.dtype == dtype and array.shape == shape
    return array


def prefix_states(tokens, positions):
    result = np.empty(len(tokens), dtype=np.uint64)
    state = FNV_OFFSET
    for i, (token, position) in enumerate(zip(tokens, positions)):
        if position == 0:
            state = FNV_OFFSET
        result[i] = state
        state = ((state ^ int(token)) * FNV_PRIME) & MASK64
    return result


def route_prefix(tokens, previous, positions, prefixes, children, layer_id,
                 structural):
    assert tokens.shape == previous.shape == positions.shape == prefixes.shape
    assert children.shape == (len(tokens), 4)
    c1, c2, c3, c4, c5 = M142.CONSTANTS
    t = tokens.astype(np.uint64)[:, None]
    u = previous.astype(np.uint64)[:, None]
    p = positions.astype(np.uint64)[:, None]
    h = prefixes.astype(np.uint64)[:, None]
    c = children.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = (M142.SEED ^ (t * c1) ^ (u * c2) ^ (p * c3) ^
             (c * c4) ^ (np.uint64(layer_id) * c5) ^ (h * PREFIX_MIX))
        z = z + c1
        z = (z ^ (z >> np.uint64(30))) * c2
        z = (z ^ (z >> np.uint64(27))) * c3
        z = z ^ (z >> np.uint64(31))
        local = np.where(structural[:, None], np.uint64(0),
                         np.uint64(1) + z % np.uint64(9))
        grand = c * np.uint64(10) + local
    assert np.array_equal(grand // 10, c)
    assert np.all((grand % 10 == 0) == structural[:, None])
    return grand.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-sha", required=True)
    ap.add_argument("--prefix-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.prefix_out.exists() and not args.out.exists()
    assert M136.digest(TRACE) == args.trace_sha
    assert M136.digest(M159) == M159_SHA
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    prior = json.loads(M159.read_text(encoding="utf-8"))
    assert trace["decision"] == "valid_source_trace_for_offline_postfailure_diagnostics"
    assert trace["meth159_result_sha256"] == M159_SHA
    assert all(trace["exact_meth159_all_content_histogram_matches"])
    assert trace["input_tokens"] == prior["input_tokens"] == TOKENS
    started = time.monotonic()
    children = load_array(trace, "children", np.dtype("uint16"), (24, TOKENS, 4))
    tokens = load_array(trace, "tokens", np.dtype("uint32"), (TOKENS,))
    previous = load_array(trace, "previous", np.dtype("uint32"), (TOKENS,))
    positions = load_array(trace, "positions", np.dtype("uint16"), (TOKENS,))
    offsets_item = trace["arrays"]["offsets"]
    assert M136.digest(Path(offsets_item["path"])) == offsets_item["sha256"]
    offsets = json.loads(Path(offsets_item["path"]).read_text(encoding="utf-8"))
    assert len(offsets) == 5120 and offsets[-1]["stop"] == TOKENS
    table = R150.load_table()
    assert len(table) == 75

    prefix = prefix_states(tokens, positions)
    example = prefix_states(np.array([11, 22, 33], dtype=np.uint32),
                            np.array([0, 1, 2], dtype=np.uint16))
    assert example.tolist() == [14695981039346656037, 12638163011299821354,
                                599313035076930292]
    terminal = ((int(example[-1]) ^ 33) * FNV_PRIME) & MASK64
    assert terminal == 2846118939483367407
    content = route_prefix(np.array([123]), np.array([45]), np.array([67]),
                           np.array([terminal], dtype=np.uint64),
                           np.array([[89, 89, 89, 89]]), 3, np.array([False]))
    structural_golden = route_prefix(np.array([11]), np.array([16948]),
                                     np.array([7]), np.array([terminal], dtype=np.uint64),
                                     np.array([[89, 89, 89, 89]]), 3, np.array([True]))
    assert int(content[0, 0]) == 895 and int(structural_golden[0, 0]) == 890
    args.prefix_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.prefix_out, prefix, allow_pickle=False)
    assert args.prefix_out.stat().st_size < 10_000_000
    prefix_sha = M136.digest(args.prefix_out)
    budget(started)

    token_ids = np.asarray(tokens, dtype=np.int64)
    prior_ids = np.asarray(previous, dtype=np.int64)
    local_positions = np.asarray(positions, dtype=np.int64)
    shared = R150.shared_mask(token_ids, prior_ids, local_positions, table)
    old_exact = []
    layers = []
    concentration = None
    for li in range(24):
        source = np.asarray(children[li], dtype=np.int64)
        assert np.all((source >= 0) & (source < 1280))
        old, old_shared = R150.route(token_ids, prior_ids, local_positions,
                                      source, li, table)
        assert np.array_equal(old_shared, shared)
        old_all = np.bincount(old.reshape(-1), minlength=12800)
        old_content = np.bincount(old[~shared].reshape(-1), minlength=12800)
        exact = (np.array_equal(old_all, prior["layers"][li]["all_slot_counts"]) and
                 np.array_equal(old_content, prior["layers"][li]["content_slot_counts"]))
        old_exact.append(bool(exact))
        assert exact, li
        grand = route_prefix(token_ids, prior_ids, local_positions, prefix,
                              source, li, shared)
        if li == 2:
            old_mask = (source == 314) & (old == 3141) & ~shared[:, None]
            locations = np.where(old_mask)
            assert len(locations[0]) == 186
            new_locals = grand[locations] % 10
            contexts = Counter((int(tokens[pos]), int(previous[pos]), int(positions[pos]))
                               for pos in locations[0])
            concentration = {"layer": 2, "source_child": 314,
                             "old_content_local": 1, "old_slot_selections": 186,
                             "new_local_counts_for_same_selected_occurrences":
                                 np.bincount(new_locals, minlength=10).tolist(),
                             "top_contexts": [{"token": key[0], "previous": key[1],
                                               "position": key[2], "selected_count": count,
                                               "new_local_counts": np.bincount(
                                                   new_locals[[j for j, pos in enumerate(locations[0])
                                                               if (int(tokens[pos]), int(previous[pos]),
                                                                   int(positions[pos])) == key]],
                                                   minlength=10).tolist()}
                                              for key, count in contexts.most_common(20)]}
        parent = np.bincount(source.reshape(-1), minlength=1280)
        all_counts = np.bincount(grand.reshape(-1), minlength=12800)
        content_parent = np.bincount(source[~shared].reshape(-1), minlength=1280)
        content_grand = np.bincount(grand[~shared].reshape(-1), minlength=12800)
        row = M150.layer_summary(parent, all_counts, content_parent,
                                 content_grand, TOKENS)
        slots = content_grand.reshape(1280, 10)[:, 1:].reshape(-1)
        active = slots[slots > 0]
        row["content_active_selection_p10"] = float(np.quantile(active, 0.10))
        row["content_active_selection_p50"] = float(np.quantile(active, 0.50))
        row["content_active_selection_p90"] = float(np.quantile(active, 0.90))
        row["content_slots_under_32_fraction"] = float((slots < 32).mean())
        row["content_slot_counts"] = content_grand.tolist()
        row["all_slot_counts"] = all_counts.tolist()
        layers.append(row)
        print(json.dumps({"layer": li, "coverage": row["content"]["coverage"],
                          "active_median": row["content_active_selection_p50"],
                          "hot_share": row["content_hot_parent_worst_grandchild_share"],
                          "budget": budget(started)}), flush=True)
    assert all(old_exact) and concentration is not None
    gates = {"content_coverage": min(row["content"]["coverage"] for row in layers) >= 10000,
             "active_content_median": min(row["content_active_selection_p50"] for row in layers) >= 50,
             "content_under_32_fraction": max(row["content_slots_under_32_fraction"] for row in layers) <= 0.50,
             "content_load_ratio": max(row["content_to_own_parent_max_load_ratio"] for row in layers) <= 1.25,
             "content_hot_parent_share": max(row["content_hot_parent_worst_grandchild_share"] for row in layers) <= 0.25}
    result = {"experiment": "METH-163-full-prefix-content-hash-diagnostic",
              "source_trace_result_sha256": args.trace_sha,
              "meth159_result_sha256": M159_SHA,
              "original_table_sha256": R150.TABLE_SHA,
              "fnv_offset": FNV_OFFSET, "fnv_prime": FNV_PRIME,
              "prefix_mix_constant": int(PREFIX_MIX),
              "golden_prefix_states": example.tolist(),
              "golden_terminal_prefix_state": terminal,
              "golden_content_grandchild": 895,
              "golden_shared_grandchild": 890,
              "prefix_states_sha256": prefix_sha,
              "prefix_states_bytes": args.prefix_out.stat().st_size,
              "input_tokens": TOKENS, "selections_per_layer": 4 * TOKENS,
              "structural_selections_per_layer": int(4 * shared.sum()),
              "original_histograms_replayed_exactly": old_exact,
              "original_worst_slot_redistribution": concentration,
              "layers": layers, "candidate_in_sample_gates": gates,
              "runtime": budget(started),
              "decision": "postfailure_in_sample_diagnostic_only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "result_sha256": M136.digest(args.out),
                      "candidate_in_sample_gates": gates,
                      "min_content_coverage": min(row["content"]["coverage"] for row in layers),
                      "min_active_median": min(row["content_active_selection_p50"] for row in layers),
                      "worst_hot_share": max(row["content_hot_parent_worst_grandchild_share"] for row in layers),
                      "worst_original_slot_new_local_counts":
                          concentration["new_local_counts_for_same_selected_occurrences"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
