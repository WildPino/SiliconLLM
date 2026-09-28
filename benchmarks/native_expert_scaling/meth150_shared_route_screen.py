#!/usr/bin/env python3
"""Screen a structural shared slot and nine content slots on old/new sources."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth136_sparse_experts  # inserts the S1 import path
import meth136_matched_sparse_train as M136
import meth139_quantile_third_router as M139
import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DOC / "meth150_shared_route_manifest.json"
MANIFEST_SHA = "6e911e65529be53dba65fbe382229f0f43e510eb76ab83f4404d4be0368ff5d9"
M149 = DOC / "meth149_recurrent_context_route_result.json"
M149_SHA = "cf8087f1677bf9c60076e055f35ac95cdb2e7f426ab5dce3d54ad441ed2e28e2"
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS
            or result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-150 resource stop: {result}")
    return result


def layer_summary(parent, grand, content_parent, content_grand, total_tokens):
    assert int(parent.sum()) == int(grand.sum()) == 4 * total_tokens
    assert int(content_parent.sum()) == int(content_grand.sum())
    assert not np.any(content_grand.reshape(1280, 10)[:, 0])
    structural = int(grand.reshape(1280, 10)[:, 0].sum())
    assert structural + int(content_grand.sum()) == 4 * total_tokens
    hot = content_parent >= 250
    assert hot.any()
    hot_share = float(np.max(content_grand.reshape(1280, 10)[hot].max(axis=1)
                             / content_parent[hot]))
    ratio = float(9 * content_grand.max() / content_parent.max())
    return {"total": M139.summarize(grand),
            "original_parent": M139.summarize(parent),
            "content": M139.summarize(content_grand),
            "content_parent": M139.summarize(content_parent),
            "structural_selections": structural,
            "content_to_own_parent_max_load_ratio": ratio,
            "content_hot_parent_count": int(hot.sum()),
            "content_hot_parent_worst_grandchild_share": hot_share}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    assert M136.digest(MANIFEST) == MANIFEST_SHA
    assert M136.digest(M149) == M149_SHA
    table = R150.load_table()
    golden = R150.golden(table)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["meth149_result_sha256"] == M149_SHA
    assert manifest["meth149_table_sha256"] == R150.TABLE_SHA
    assert manifest["selected_counts"] == {"code": 8, "prose": 8,
                                           "technical_general": 8}
    items = manifest["items"]
    assert len(items) == 24
    for item in items:
        assert M136.M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        assert M136.M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item["document_ids_sha256"]
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
    raw_ids, chat, draws = M136.training_data()
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    teacher, _ = M136.make_wrappers(model, parent["expert_state"],
        child["expert_state"], source_a, source_b, prefixes, device, "teacher")
    parity_items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    expected = []
    with torch.inference_mode():
        for item in parity_items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            expected.append(model(ids, use_cache=False).logits.cpu())
    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    del teacher
    gc.collect(); torch.cuda.empty_cache()
    wrappers, _ = M136.make_wrappers(model, parent["expert_state"],
        child["expert_state"], source_a, source_b, prefixes, device, "control")
    parity_error = 0.0
    with torch.inference_mode():
        for item, reference in zip(parity_items, expected):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            actual = model(ids, use_cache=False).logits.cpu()
            parity_error = max(parity_error, float((actual.float() - reference.float()).abs().max()))
            assert torch.equal(actual, reference)
    del parent, child, source_a, source_b
    gc.collect()
    budget(start, device)

    def measure(name, sequences, width):
        parent_counts = np.zeros((24, 1280), dtype=np.int64)
        grand_counts = np.zeros((24, 12800), dtype=np.int64)
        content_parent = np.zeros((24, 1280), dtype=np.int64)
        content_grand = np.zeros((24, 12800), dtype=np.int64)
        current = {"tokens": None, "previous": None, "positions": None}
        hooks = []
        for li, wrapper in enumerate(wrappers):
            def count(module, _inputs, _output, layer_id=li):
                child_ids = module.last_selected.cpu().numpy().astype(np.int64)
                grand, shared = R150.route(current["tokens"], current["previous"],
                                            current["positions"], child_ids,
                                            layer_id, table)
                parent_counts[layer_id] += np.bincount(child_ids.flatten(), minlength=1280)
                grand_counts[layer_id] += np.bincount(grand.flatten(), minlength=12800)
                content_parent[layer_id] += np.bincount(child_ids[~shared].flatten(), minlength=1280)
                content_grand[layer_id] += np.bincount(grand[~shared].flatten(), minlength=12800)
            hooks.append(wrapper.register_forward_hook(count))
        windows = 0
        total_tokens = 0
        with torch.inference_mode():
            for si, sequence in enumerate(sequences):
                for first in range(0, len(sequence), width):
                    window = np.asarray(sequence[first:first + width], dtype=np.int64)
                    current["tokens"] = window
                    current["previous"] = np.r_[0, window[:-1]]
                    current["positions"] = np.arange(window.size, dtype=np.int64)
                    ids = torch.as_tensor(window, dtype=torch.long, device=device)[None]
                    model(ids, use_cache=False)
                    windows += 1
                    total_tokens += window.size
                if (si + 1) % (64 if name.startswith("training") else 8) == 0:
                    print(json.dumps({"cell": name, "sequences": si + 1,
                                      "budget": budget(start, device)}), flush=True)
        for hook in hooks:
            hook.remove()
        layers = [layer_summary(parent_counts[li], grand_counts[li],
                                content_parent[li], content_grand[li], total_tokens)
                  for li in range(24)]
        return {"sequences": len(sequences), "windows": windows,
                "tokens": total_tokens, "per_layer": layers}

    raw_sequences = [raw_ids[d["raw_row"], d["raw_offset"]:d["raw_offset"]+128][:-1]
                     for d in draws]
    chat_sequences = [chat[d["chat_index"]][1][:-1] for d in draws]
    cells = {
        "training_raw": measure("training_raw", raw_sequences, 512),
        "training_chat": measure("training_chat", chat_sequences, 512)}
    for width in (128, 512):
        cells[f"new_documents_w{width}"] = measure(
            f"new_documents_w{width}", [item["document_ids"] for item in items], width)
    gates = {
        "teacher_control_bf16_parity": parity_error == 0,
        "content_load_ratio": all(row["content_to_own_parent_max_load_ratio"] <= 1.25
                                  for cell in cells.values() for row in cell["per_layer"]),
        "content_hot_parent_share": all(row["content_hot_parent_worst_grandchild_share"] <= 0.25
                                        for cell in cells.values() for row in cell["per_layer"]),
        "content_coverage": all(row["content"]["coverage"] >= 4000
                                for cell in cells.values() for row in cell["per_layer"])}
    summary = {name: {"tokens": cell["tokens"],
                      "structural_selections_per_layer": cell["per_layer"][0]["structural_selections"],
                      "worst_content_load_ratio": max(row["content_to_own_parent_max_load_ratio"]
                                                      for row in cell["per_layer"]),
                      "worst_content_hot_share": max(row["content_hot_parent_worst_grandchild_share"]
                                                     for row in cell["per_layer"]),
                      "minimum_content_coverage": min(row["content"]["coverage"]
                                                      for row in cell["per_layer"]),
                      "worst_all_token_slot_skew": max(row["total"]["max_to_mean"]
                                                       for row in cell["per_layer"])}
               for name, cell in cells.items()}
    result = {"experiment": "METH-150-shared-structure-route-screen",
              "manifest_sha256": MANIFEST_SHA,
              "meth149_table_sha256": R150.TABLE_SHA,
              "meth149_result_sha256": M149_SHA,
              "source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "golden": golden, "parity_prompts": len(parity_items),
              "teacher_control_bf16_logit_max_abs_error": parity_error,
              "scope": "Offline grandchild assignment after original E1280 route",
              "cells": cells, "summary": summary, "gates": gates,
              "decision": "content_route_screen_pass_integrated_and_native_pending"
                          if all(gates.values()) else "content_route_screen_fail",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < MAX_DISK
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "summary": summary, "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
