#!/usr/bin/env python3
"""Screen a calibration-free token/context hash for third-tier load."""

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
import meth139_quantile_third_router as M139
import meth140_external_quantile_route as M140


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CONSTANTS = [np.uint64(x) for x in (
    0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9,
    0x94D049BB133111EB, 0xD6E8FEB86659FD93,
    0xA5A3564E27F8862D)]
SEED = np.uint64(142142)
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000


def hash_grandchildren(tokens, previous, positions, children, layer_id):
    assert tokens.ndim == previous.ndim == positions.ndim == 1
    assert children.ndim == 2 and children.shape == (tokens.size, 4)
    t = tokens.astype(np.uint64)[:, None]
    u = previous.astype(np.uint64)[:, None]
    p = positions.astype(np.uint64)[:, None]
    c = children.astype(np.uint64)
    c1, c2, c3, c4, c5 = CONSTANTS
    with np.errstate(over="ignore"):
        z = (SEED ^ (t * c1) ^ (u * c2) ^ (p * c3) ^
             (c * c4) ^ (np.uint64(layer_id) * c5))
        z = z + c1
        z = (z ^ (z >> np.uint64(30))) * c2
        z = (z ^ (z >> np.uint64(27))) * c3
        z = z ^ (z >> np.uint64(31))
        local = z % np.uint64(10)
        grand = c * np.uint64(10) + local
    assert np.array_equal(grand // np.uint64(10), c)
    return grand.astype(np.int64)


def budget(start, device):
    report = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (report["seconds"] > MAX_SECONDS or report["rss_bytes"] > MAX_RSS
            or report["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-142 resource stop: {report}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    manifests = {}
    for name, (path, sha) in M140.MANIFESTS.items():
        assert M136.digest(path) == sha
        manifests[name] = json.loads(path.read_text(encoding="utf-8"))["items"]
        assert len(manifests[name]) == 24
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    gpu = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpu) == 1
    device = torch.device(f"cuda:{gpu[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parent_state, child_state = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    raw_ids, _chat, draws = M136.training_data()
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    teacher, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "teacher")
    parity_items = manifests["meth121"][:8]
    expected = []
    with torch.inference_mode():
        for item in parity_items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            expected.append(model(ids, use_cache=False).logits.cpu())
    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    control, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "control")
    parity_error = 0.0
    with torch.inference_mode():
        for item, reference in zip(parity_items, expected):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            actual = model(ids, use_cache=False).logits.cpu()
            parity_error = max(parity_error, float((actual.float() - reference.float()).abs().max()))
            assert torch.equal(actual, reference)
    del teacher, parent_state, child_state, source_a, source_b
    gc.collect()
    budget(start, device)

    def measure(source, width):
        parent_counts = [np.zeros(1280, dtype=np.int64) for _ in range(24)]
        grand_counts = [np.zeros(12800, dtype=np.int64) for _ in range(24)]
        current = {"tokens": None, "previous": None, "positions": None}
        hooks = []
        for li, wrapper in enumerate(control):
            def count(module, _inputs, _output, layer_id=li):
                child = module.last_selected.cpu().numpy().astype(np.int64)
                grand = hash_grandchildren(
                    current["tokens"], current["previous"],
                    current["positions"], child, layer_id)
                parent_counts[layer_id] += np.bincount(child.flatten(), minlength=1280)
                grand_counts[layer_id] += np.bincount(grand.flatten(), minlength=12800)
            hooks.append(wrapper.register_forward_hook(count))
        tokens_seen, windows = 0, 0
        with torch.inference_mode():
            if source == "h0_raw":
                sequences = []
                for draw in draws:
                    row = raw_ids[draw["raw_row"]]
                    sequence = (row[draw["raw_offset"]:draw["raw_offset"] + 128]
                                if width == 128 else row)
                    sequences.append(sequence)
            else:
                sequences = [item["document_ids"] for item in manifests[source]]
            for index, sequence in enumerate(sequences):
                for first in range(0, len(sequence), width):
                    window = np.asarray(sequence[first:first + width], dtype=np.int64)
                    current["tokens"] = window
                    current["previous"] = np.r_[0, window[:-1]]
                    current["positions"] = np.arange(window.size, dtype=np.int64)
                    ids = torch.as_tensor(window, dtype=torch.long, device=device)[None]
                    model(ids, use_cache=False)
                    tokens_seen += window.size
                    windows += 1
                if (index + 1) % (64 if source == "h0_raw" else 8) == 0:
                    print(json.dumps({"source": source, "width": width,
                                      "sequences": index + 1,
                                      "budget": budget(start, device)}), flush=True)
        for hook in hooks:
            hook.remove()
        layers = []
        for li, (parent, grand) in enumerate(zip(parent_counts, grand_counts)):
            assert int(parent.sum()) == int(grand.sum()) == 4 * tokens_seen
            p, g = M139.summarize(parent), M139.summarize(grand)
            heavy = parent >= 250
            assert heavy.any()
            share = float(np.max(grand.reshape(1280, 10)[heavy].max(axis=1) / parent[heavy]))
            layers.append({"layer": li, "control": p, "candidate": g,
                           "candidate_to_control_max_load_ratio":
                               g["max_to_mean"] / p["max_to_mean"],
                           "hot_parent_count": int(heavy.sum()),
                           "hot_parent_worst_grandchild_share": share})
        return {"sequences": len(sequences), "windows": windows,
                "tokens": tokens_seen, "per_layer": layers}

    cells = {}
    for source in ("h0_raw", "meth121", "meth133"):
        for width in (128, 512):
            cells[f"{source}_w{width}"] = measure(source, width)
    del control, model
    gc.collect()
    gates = {"teacher_control_bf16_parity": parity_error == 0.0,
             "all_cells_load_ratio": all(row["candidate_to_control_max_load_ratio"] <= 1.25
                                         for cell in cells.values() for row in cell["per_layer"]),
             "all_cells_coverage": all(row["candidate"]["coverage"] >= 4000
                                       for cell in cells.values() for row in cell["per_layer"]),
             "all_cells_hot_parent_share": all(row["hot_parent_worst_grandchild_share"] <= 0.25
                                               for cell in cells.values() for row in cell["per_layer"])}
    summary = {name: {"tokens": cell["tokens"],
                      "worst_load_ratio": max(row["candidate_to_control_max_load_ratio"]
                                              for row in cell["per_layer"]),
                      "worst_hot_share": max(row["hot_parent_worst_grandchild_share"]
                                             for row in cell["per_layer"]),
                      "minimum_coverage": min(row["candidate"]["coverage"]
                                              for row in cell["per_layer"])}
               for name, cell in cells.items()}
    golden = hash_grandchildren(
        np.array([123], dtype=np.int64), np.array([45], dtype=np.int64),
        np.array([67], dtype=np.int64), np.array([[89, 89, 89, 89]], dtype=np.int64), 3)
    result = {"experiment": "METH-142-token-hash-third-tier-load-screen",
              "source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "training_data_sha256": M136.M15.TRAIN_FILE_SHA,
              "manifest_sha256": {name: sha for name, (_, sha) in M140.MANIFESTS.items()},
              "hash_seed": 142142,
              "hash_constants_hex": [f"{int(value):016x}" for value in CONSTANTS],
              "hash_golden": {"token": 123, "previous": 45,
                              "position": 67, "child": 89, "layer": 3,
                              "grandchild": int(golden[0, 0])},
              "parity_prompts": len(parity_items),
              "teacher_control_bf16_logit_max_abs_error": parity_error,
              "scope": "Offline grandchild assignment; no hash-routed full-model logits",
              "cells": cells, "summary": summary, "gates": gates,
              "decision": "development_load_pass_fresh_route_check_required"
                          if all(gates.values()) else "development_load_fail",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < MAX_DISK
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "summary": summary, "runtime": result["runtime"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
