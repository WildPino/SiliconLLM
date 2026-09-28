#!/usr/bin/env python3
"""Replay one fixed quantile router across source and context lengths."""

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
SIDECAR = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth140_quantile_third_router.bin"
SIDECAR_SHA = "a581272ed84c154b660a4fd9a7c108f0dcced394e0d763ddfdd366aa92eadc5f"
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS
            or record["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-141 resource stop: {record}")
    return record


def load_sidecar(projections):
    assert M136.digest(SIDECAR) == SIDECAR_SHA
    data = np.memmap(SIDECAR, dtype=np.uint8, mode="r")
    assert M140.HEADER.unpack_from(data) == (
        M140.MAGIC, 24, 896, 32, 1280, 10, 140140)
    layer_bytes = (32 * 896 + 1280 * 9) * 4
    assert data.size == M140.HEADER.size + 24 * layer_bytes
    thresholds = []
    for li, projection in enumerate(projections):
        offset = M140.HEADER.size + li * layer_bytes
        stored_projection = np.frombuffer(data, dtype="<f4", count=32 * 896,
                                          offset=offset).reshape(32, 896)
        assert np.array_equal(stored_projection, projection.numpy())
        th = np.frombuffer(data, dtype="<f4", count=1280 * 9,
                           offset=offset + 32 * 896 * 4).reshape(1280, 9).copy()
        assert np.isfinite(th).all() and np.all(np.diff(th, axis=1) >= 0)
        thresholds.append(th)
    return thresholds


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
    third = M136.load_third()
    projections = [entry[0] for entry in third]
    thresholds = load_sidecar(projections)
    raw_ids, _chat, draws = M136.training_data()
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    teacher, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "teacher")
    parity_items = manifests["meth121"][:8]
    reference = []
    with torch.inference_mode():
        for item in parity_items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            reference.append(model(ids, use_cache=False).logits.cpu())
    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    clone = M139.make_clone_wrappers(
        model, original_mlps, parent_state["expert_state"],
        child_state["expert_state"], source_a, source_b,
        prefixes, projections, thresholds, device)
    parity_error = 0.0
    with torch.inference_mode():
        for item, expected in zip(parity_items, reference):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            actual = model(ids, use_cache=False).logits.cpu()
            parity_error = max(parity_error, float((actual.float() - expected.float()).abs().max()))
            assert torch.equal(actual, expected)
            for wrapper in clone:
                assert torch.equal(wrapper.last_grandchild // 10, wrapper.last_selected)
    del teacher, parent_state, child_state, source_a, source_b
    gc.collect()
    budget(start, device)

    def measure(source, width):
        parent_counts = [np.zeros(1280, dtype=np.int64) for _ in range(24)]
        grand_counts = [np.zeros(12800, dtype=np.int64) for _ in range(24)]
        hooks = []
        for li, wrapper in enumerate(clone):
            def count(module, _inputs, _output, layer_id=li):
                child = module.last_selected.flatten().cpu().numpy().astype(np.int64)
                grand = module.last_grandchild.flatten().cpu().numpy().astype(np.int64)
                assert np.array_equal(grand // 10, child)
                parent_counts[layer_id] += np.bincount(child, minlength=1280)
                grand_counts[layer_id] += np.bincount(grand, minlength=12800)
            hooks.append(wrapper.register_forward_hook(count))
        total_tokens, windows = 0, 0
        with torch.inference_mode():
            if source == "h0_raw":
                sequences = []
                for draw in draws:
                    row = raw_ids[draw["raw_row"]]
                    if width == 128:
                        sequence = row[draw["raw_offset"]:draw["raw_offset"] + 128]
                    else:
                        sequence = row
                    sequences.append(sequence)
            else:
                sequences = [item["document_ids"] for item in manifests[source]]
            for index, sequence in enumerate(sequences):
                for first in range(0, len(sequence), width):
                    window = sequence[first:first + width]
                    ids = torch.as_tensor(window, dtype=torch.long, device=device)[None]
                    model(ids, use_cache=False)
                    total_tokens += len(window)
                    windows += 1
                if (index + 1) % (64 if source == "h0_raw" else 8) == 0:
                    print(json.dumps({"source": source, "width": width,
                                      "sequences": index + 1,
                                      "budget": budget(start, device)}), flush=True)
        for hook in hooks:
            hook.remove()
        per_layer = []
        for li, (parent, grand) in enumerate(zip(parent_counts, grand_counts)):
            assert int(parent.sum()) == int(grand.sum()) == 4 * total_tokens
            parent_summary, grand_summary = M139.summarize(parent), M139.summarize(grand)
            heavy = parent >= 250
            assert heavy.any()
            share = float(np.max(grand.reshape(1280, 10)[heavy].max(axis=1) / parent[heavy]))
            per_layer.append({"layer": li, "control": parent_summary,
                              "candidate": grand_summary,
                              "candidate_to_control_max_load_ratio":
                                  grand_summary["max_to_mean"] / parent_summary["max_to_mean"],
                              "hot_parent_count": int(heavy.sum()),
                              "hot_parent_worst_grandchild_share": share})
        return {"sequences": len(sequences), "windows": windows,
                "tokens": total_tokens, "per_layer": per_layer}

    cells = {}
    for source in ("h0_raw", "meth121", "meth133"):
        for width in (128, 512):
            cells[f"{source}_w{width}"] = measure(source, width)
    del clone, model
    gc.collect()
    summary = {name: {"tokens": cell["tokens"],
                      "worst_load_ratio": max(row["candidate_to_control_max_load_ratio"]
                                              for row in cell["per_layer"]),
                      "worst_hot_share": max(row["hot_parent_worst_grandchild_share"]
                                             for row in cell["per_layer"]),
                      "minimum_coverage": min(row["candidate"]["coverage"]
                                              for row in cell["per_layer"])}
               for name, cell in cells.items()}
    result = {"experiment": "METH-141-context-domain-route-diagnostic",
              "scope": "Fixed rejected METH-140 sidecar; diagnostic only",
              "sidecar_sha256": SIDECAR_SHA,
              "source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "training_data_sha256": M136.M15.TRAIN_FILE_SHA,
              "manifest_sha256": {name: sha for name, (_, sha) in M140.MANIFESTS.items()},
              "parity_prompts": len(parity_items),
              "parity_logit_max_abs_error": parity_error,
              "summary": summary, "cells": cells,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < MAX_DISK
    print(json.dumps({"summary": summary, "runtime": result["runtime"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
