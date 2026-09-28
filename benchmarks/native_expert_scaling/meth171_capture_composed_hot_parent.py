#!/usr/bin/env python3
"""Capture METH-169 source children to attribute its composed hot parent."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DOC / "meth169_fresh_boundary_route_manifest.json"
MANIFEST_SHA = "8b2e3a545f33dbd31283fd173b9044e2c87bd55463e168d6b2f8c513cde4e468"
TEACHER = DOC / "meth169_fresh_boundary_teacher.json"
TEACHER_SHA = "b051d9cc45bac62bf55d599ea5f0a34ec08d3dbe015483d7818d59997f5140ce"
M169 = DOC / "meth169_fresh_boundary_route_result.json"
M169_SHA = "1cf6bad51c2625fee83201242347697b1998a4ca11d48cda5f31715931bfb792"
TABLE = DOC / "meth162_training_derived_shared_table.json"
TABLE_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
TOKENS = 176710
MAX_SECONDS = 10 * 60
MAX_RSS = 20 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))
MAX_TRACE = 100_000_000


def budget(start, device):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss,
           "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS or
            row["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-171 resource stop: {row}")
    return row


def output_paths(prefix):
    return {name: prefix.with_name(prefix.name + suffix) for name, suffix in (
        ("children", ".children.npy"), ("tokens", ".tokens.npy"),
        ("previous", ".previous.npy"), ("positions", ".positions.npy"),
        ("offsets", ".offsets.json"), ("result", ".result.json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()
    paths = output_paths(args.out_prefix)
    assert all(not path.exists() for path in paths.values())
    for path, sha in ((MANIFEST, MANIFEST_SHA), (TEACHER, TEACHER_SHA),
                      (M169, M169_SHA), (TABLE, TABLE_SHA)):
        assert M136.digest(path) == sha, path
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    teacher_rows = json.loads(TEACHER.read_text(encoding="utf-8"))
    prior = json.loads(M169.read_text(encoding="utf-8"))
    assert prior["decision"] == "prospective_route_fail"
    assert prior["cells"]["raw"]["tokens"] + prior["cells"]["chat"]["tokens"] == TOKENS
    assert prior["teacher_sha256"] == TEACHER_SHA
    assert len(manifest["raw_rows"]) == len(manifest["chat_rows"]) == 512
    assert len(teacher_rows["rows"]) == 512
    expected = sum(len(raw["span_ids"]) - 1 +
                   len(chat["prompt_ids"]) + len(response["continuation_ids"]) - 1
                   for raw, chat, response in zip(manifest["raw_rows"],
                                                   manifest["chat_rows"],
                                                   teacher_rows["rows"]))
    assert expected == TOKENS
    table_record = json.loads(TABLE.read_text(encoding="utf-8"))
    table = {(r["token"], r["previous"], r["position"])
             for r in table_record["tuples"]}
    assert len(table) == 138 and R150.load_table() <= table
    assert R150.golden(table) == {"content": 899, "structural": 890}
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    gpu = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpu) == 1
    device = torch.device(f"cuda:{gpu[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    parent, child = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    teacher_wrappers, _ = M136.make_wrappers(model, parent["expert_state"],
        child["expert_state"], source_a, source_b, prefixes, device, "teacher")
    parity = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    with torch.inference_mode():
        reference = [model(torch.as_tensor(row["prompt_ids"], dtype=torch.long,
                      device=device)[None], use_cache=False).logits.cpu()
                     for row in parity]
    for layer, original in zip(model.model.layers, original_mlps):
        layer.mlp = original
    del teacher_wrappers
    gc.collect(); torch.cuda.empty_cache()
    wrappers, _ = M136.make_wrappers(model, parent["expert_state"],
        child["expert_state"], source_a, source_b, prefixes, device, "control")
    with torch.inference_mode():
        for row, expected_logits in zip(parity, reference):
            actual = model(torch.as_tensor(row["prompt_ids"], dtype=torch.long,
                           device=device)[None], use_cache=False).logits.cpu()
            assert torch.equal(actual, expected_logits)
    del parent, child, source_a, source_b, reference
    gc.collect(); torch.cuda.empty_cache()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    children = np.lib.format.open_memmap(paths["children"], mode="w+",
                                          dtype=np.uint16, shape=(24, TOKENS, 4))
    tokens = np.lib.format.open_memmap(paths["tokens"], mode="w+",
                                        dtype=np.uint32, shape=(TOKENS,))
    previous = np.lib.format.open_memmap(paths["previous"], mode="w+",
                                          dtype=np.uint32, shape=(TOKENS,))
    positions = np.lib.format.open_memmap(paths["positions"], mode="w+",
                                           dtype=np.uint16, shape=(TOKENS,))
    current = {"start": 0, "length": 0}
    hooks = []
    for li, wrapper in enumerate(wrappers):
        def capture(module, _inputs, _output, layer_id=li):
            selected = module.last_selected.detach().cpu().numpy()
            assert selected.shape == (current["length"], 4)
            assert np.all((selected >= 0) & (selected < 1280))
            start = current["start"]
            children[layer_id, start:start + current["length"]] = selected.astype(np.uint16)
        hooks.append(wrapper.register_forward_hook(capture))
    offsets = []
    cursor = 0
    with torch.inference_mode():
        for index, (raw, chat, response) in enumerate(zip(
                manifest["raw_rows"], manifest["chat_rows"], teacher_rows["rows"])):
            assert response["index"] == index
            assert response["source_row"] == chat["source_row"]
            assert response["prompt_ids_sha256"] == chat["prompt_ids_sha256"]
            raw_ids = np.asarray(raw["span_ids"], dtype=np.int32)
            assert M136.M17.sha(raw_ids.tobytes()) == raw["span_ids_sha256"]
            continuation = np.asarray(response["continuation_ids"], dtype=np.int32)
            assert M136.M17.sha(continuation.tobytes()) == response["continuation_ids_sha256"]
            for kind, sequence in (("raw", raw_ids[:-1]),
                                   ("chat", np.asarray(chat["prompt_ids"] +
                                    response["continuation_ids"], dtype=np.int32)[:-1])):
                ids = np.asarray(sequence, dtype=np.int64)
                length = len(ids)
                assert cursor + length <= TOKENS and np.all(ids >= 0)
                tokens[cursor:cursor + length] = ids.astype(np.uint32)
                previous[cursor:cursor + length] = np.r_[0, ids[:-1]].astype(np.uint32)
                positions[cursor:cursor + length] = np.arange(length, dtype=np.uint16)
                current["start"] = cursor
                current["length"] = length
                model(torch.as_tensor(ids, dtype=torch.long, device=device)[None],
                      use_cache=False)
                offsets.append({"pair": index, "kind": kind,
                                "source_row": raw["source_row"] if kind == "raw"
                                              else chat["source_row"],
                                "start": cursor, "stop": cursor + length})
                cursor += length
            if (index + 1) % 128 == 0:
                print(json.dumps({"completed_pairs": index + 1,
                                  "tokens_written": cursor,
                                  "budget": budget(started, device)}), flush=True)
    assert cursor == TOKENS and len(offsets) == 1024
    for hook in hooks:
        hook.remove()
    children.flush(); tokens.flush(); previous.flush(); positions.flush()
    paths["offsets"].write_text(json.dumps(offsets, separators=(",", ":")) + "\n",
                                encoding="utf-8")
    del model, wrappers
    gc.collect(); torch.cuda.empty_cache()

    histogram_matches = {name: [] for name in ("raw", "chat", "combined")}
    source_coverage = []
    structural_total = []
    raw_mask = np.zeros(TOKENS, dtype=np.bool_)
    for item in offsets:
        if item["kind"] == "raw":
            raw_mask[item["start"]:item["stop"]] = True
        else:
            assert item["kind"] == "chat"
    assert int(raw_mask.sum()) == prior["cells"]["raw"]["tokens"]
    assert int((~raw_mask).sum()) == prior["cells"]["chat"]["tokens"]
    token_ids = np.asarray(tokens, dtype=np.int64)
    prior_ids = np.asarray(previous, dtype=np.int64)
    local_positions = np.asarray(positions, dtype=np.int64)
    for li in range(24):
        source = np.asarray(children[li], dtype=np.int64)
        base_grand, base_shared = R150.route(
            token_ids, prior_ids, local_positions, source, li, table)
        shared = base_shared | (token_ids == 151644) | (token_ids == 151645)
        grand = np.where(shared[:, None], source * 10, base_grand)
        assert np.array_equal(grand // 10, source)
        all_counts = np.bincount(grand.reshape(-1), minlength=12800)
        content_counts = np.bincount(grand[~shared].reshape(-1), minlength=12800)
        all_raw = np.bincount(grand[raw_mask].reshape(-1), minlength=12800)
        all_chat = np.bincount(grand[~raw_mask].reshape(-1), minlength=12800)
        content_raw = np.bincount(grand[raw_mask & ~shared].reshape(-1), minlength=12800)
        content_chat = np.bincount(grand[~raw_mask & ~shared].reshape(-1), minlength=12800)
        actual = {"raw": (all_raw, content_raw),
                  "chat": (all_chat, content_chat),
                  "combined": (all_counts, content_counts)}
        expected = {"raw": prior["cells"]["raw"]["candidate_layers"][li],
                    "chat": prior["cells"]["chat"]["candidate_layers"][li],
                    "combined": prior["pools"]["raw_chat"][li]}
        for name, (all_hist, content_hist) in actual.items():
            match = (np.array_equal(all_hist, expected[name]["all_slot_counts"]) and
                     np.array_equal(content_hist, expected[name]["content_slot_counts"]))
            histogram_matches[name].append(bool(match))
            if not match:
                raise AssertionError(f"METH-171/METH-169 {name} histogram mismatch layer {li}")
        source_coverage.append(int(np.unique(source).size))
        structural_total.append(int(4 * shared.sum()))
    assert len(set(structural_total)) == 1
    size = sum(paths[name].stat().st_size for name in
               ("children", "tokens", "previous", "positions", "offsets"))
    assert size <= MAX_TRACE
    result = {"experiment": "METH-171-composed-hot-parent-source-trace",
              "manifest_sha256": MANIFEST_SHA, "teacher_merged_sha256": TEACHER_SHA,
              "meth169_result_sha256": M169_SHA,
              "exact_source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "structural_table_sha256": TABLE_SHA,
              "pairs": 512, "sequences": len(offsets), "input_tokens": TOKENS,
              "selections_per_layer": 4 * TOKENS,
              "source_coverage_by_layer": source_coverage,
              "structural_selections_per_layer": structural_total[0],
              "exact_meth169_raw_chat_combined_histogram_matches": histogram_matches,
              "arrays": {name: {"path": str(paths[name].resolve()),
                                 "sha256": M136.digest(paths[name]),
                                 "bytes": paths[name].stat().st_size}
                         for name in ("children", "tokens", "previous", "positions", "offsets")},
              "total_trace_bytes": size,
              "runtime": {**budget(started, device), "gpu": torch.cuda.get_device_name(device)},
              "decision": "valid_composed_hot_parent_trace_for_attribution"}
    paths["result"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "result_sha256": M136.digest(paths["result"]),
                      "minimum_source_coverage": min(source_coverage),
                      "structural_selections_per_layer": structural_total[0],
                      "total_trace_bytes": size, "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

