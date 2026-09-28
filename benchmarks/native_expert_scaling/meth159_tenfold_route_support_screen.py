#!/usr/bin/env python3
"""Offline shared/content route support on independent METH-158 training IDs."""

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
import meth150_shared_route_screen as M150
import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DOC / "meth158_independent_training_manifest.json"
MANIFEST_SHA = "09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69"
TEACHER = DOC / "meth158_independent_teacher_merged.json"
MAX_SECONDS = 60 * 60
MAX_RSS = 40 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS
            or result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-159 preflight resource stop: {result}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert M136.digest(MANIFEST) == MANIFEST_SHA
    assert M136.digest(TEACHER) == args.teacher_sha
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    teacher_rows = json.loads(TEACHER.read_text(encoding="utf-8"))
    assert teacher_rows["prompt_manifest_sha256"] == MANIFEST_SHA
    assert len(manifest["chat_rows"]) == len(manifest["raw_rows"]) == 2560
    assert len(teacher_rows["rows"]) == 2560
    table = R150.load_table()
    assert R150.golden(table) == {"content": 899, "structural": 890}
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    gpu = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpu) == 1
    device = torch.device(f"cuda:{gpu[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parent, child = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    teacher_wrappers, _ = M136.make_wrappers(model, parent["expert_state"],
        child["expert_state"], source_a, source_b, prefixes, device, "teacher")
    parity = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    with torch.inference_mode():
        expected = [model(torch.as_tensor(row["prompt_ids"], dtype=torch.long,
                     device=device)[None], use_cache=False).logits.cpu()
                    for row in parity]
    for layer, original in zip(model.model.layers, original_mlps):
        layer.mlp = original
    del teacher_wrappers
    gc.collect(); torch.cuda.empty_cache()
    wrappers, _ = M136.make_wrappers(model, parent["expert_state"],
        child["expert_state"], source_a, source_b, prefixes, device, "control")
    with torch.inference_mode():
        for row, reference in zip(parity, expected):
            actual = model(torch.as_tensor(row["prompt_ids"], dtype=torch.long,
                           device=device)[None], use_cache=False).logits.cpu()
            assert torch.equal(actual, reference)
    del parent, child, source_a, source_b
    gc.collect(); torch.cuda.empty_cache()

    parent_counts = np.zeros((24, 1280), dtype=np.int64)
    grand_counts = np.zeros((24, 12800), dtype=np.int64)
    content_parent = np.zeros((24, 1280), dtype=np.int64)
    content_grand = np.zeros((24, 12800), dtype=np.int64)
    current = {"tokens": None, "previous": None, "positions": None}
    hooks = []
    for li, wrapper in enumerate(wrappers):
        def count(module, _inputs, _output, layer_id=li):
            children = module.last_selected.cpu().numpy().astype(np.int64)
            grand, shared = R150.route(current["tokens"], current["previous"],
                                        current["positions"], children, layer_id, table)
            assert np.array_equal(grand // 10, children)
            parent_counts[layer_id] += np.bincount(children.flatten(), minlength=1280)
            grand_counts[layer_id] += np.bincount(grand.flatten(), minlength=12800)
            content_parent[layer_id] += np.bincount(children[~shared].flatten(), minlength=1280)
            content_grand[layer_id] += np.bincount(grand[~shared].flatten(), minlength=12800)
        hooks.append(wrapper.register_forward_hook(count))
    total_tokens = 0
    with torch.inference_mode():
        for index, (raw, chat, continuation) in enumerate(zip(
                manifest["raw_rows"], manifest["chat_rows"], teacher_rows["rows"])):
            assert continuation["index"] == index
            assert continuation["train_row"] == chat["train_row"]
            assert continuation["prompt_ids_sha256"] == chat["prompt_ids_sha256"]
            raw_ids = np.asarray(raw["window_ids"], dtype=np.int32)
            assert M136.M17.sha(raw_ids.tobytes()) == raw["window_ids_sha256"]
            response = np.asarray(continuation["continuation_ids"], dtype=np.int32)
            assert M136.M17.sha(response.tobytes()) == continuation["continuation_ids_sha256"]
            for sequence in (raw_ids[:-1], np.asarray(
                    chat["prompt_ids"] + continuation["continuation_ids"],
                    dtype=np.int64)[:-1]):
                tokens = np.asarray(sequence, dtype=np.int64)
                current["tokens"] = tokens
                current["previous"] = np.r_[0, tokens[:-1]]
                current["positions"] = np.arange(tokens.size, dtype=np.int64)
                ids = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]
                model(ids, use_cache=False)
                total_tokens += tokens.size
            if (index + 1) % 128 == 0:
                print(json.dumps({"completed_pairs": index + 1,
                                  "budget": budget(start, device)}), flush=True)
    for hook in hooks:
        hook.remove()
    layers = []
    for li in range(24):
        row = M150.layer_summary(parent_counts[li], grand_counts[li],
                                 content_parent[li], content_grand[li], total_tokens)
        slots = content_grand[li].reshape(1280, 10)[:, 1:].reshape(-1)
        active = slots[slots > 0]
        row["content_active_selection_p10"] = float(np.quantile(active, 0.10))
        row["content_active_selection_p50"] = float(np.quantile(active, 0.50))
        row["content_active_selection_p90"] = float(np.quantile(active, 0.90))
        row["content_slots_under_32"] = int((slots < 32).sum())
        row["content_slots_under_32_fraction"] = float((slots < 32).mean())
        row["content_slot_counts"] = content_grand[li].tolist()
        row["all_slot_counts"] = grand_counts[li].tolist()
        layers.append(row)
    gates = {
        "teacher_control_bf16_parity": True,
        "content_coverage": min(r["content"]["coverage"] for r in layers) >= 10000,
        "active_content_median": min(r["content_active_selection_p50"] for r in layers) >= 50,
        "content_under_32_fraction": max(r["content_slots_under_32_fraction"] for r in layers) <= 0.50,
        "content_load_ratio": max(r["content_to_own_parent_max_load_ratio"] for r in layers) <= 1.25,
        "content_hot_parent_share": max(r["content_hot_parent_worst_grandchild_share"] for r in layers) <= 0.25}
    result = {"experiment": "METH-159-independent-tenfold-route-support-preflight",
              "manifest_sha256": MANIFEST_SHA,
              "teacher_merged_sha256": args.teacher_sha,
              "raw_source_rows": [row["train_row"] for row in manifest["raw_rows"]],
              "raw_window_ids_sha256": [row["window_ids_sha256"] for row in manifest["raw_rows"]],
              "chat_source_rows": [row["train_row"] for row in manifest["chat_rows"]],
              "chat_prompt_ids_sha256": [row["prompt_ids_sha256"] for row in manifest["chat_rows"]],
              "teacher_continuation_ids_sha256": [row["continuation_ids_sha256"]
                                                   for row in teacher_rows["rows"]],
              "table_sha256": R150.TABLE_SHA,
              "exact_source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "teacher_control_parity_items": len(parity),
              "pairs": 2560, "input_tokens": total_tokens,
              "expected_selections_per_layer": 4 * total_tokens,
              "layers": layers, "gates": gates,
              "decision": "tenfold_route_support_pass_training_protocol_pending"
                          if all(gates.values()) else "tenfold_route_support_fail",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "minimum_content_coverage": min(r["content"]["coverage"] for r in layers),
                      "minimum_active_median": min(r["content_active_selection_p50"] for r in layers),
                      "worst_under_32_fraction": max(r["content_slots_under_32_fraction"] for r in layers),
                      "worst_content_load_ratio": max(r["content_to_own_parent_max_load_ratio"] for r in layers),
                      "worst_hot_share": max(r["content_hot_parent_worst_grandchild_share"] for r in layers),
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
