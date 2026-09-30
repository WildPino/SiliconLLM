#!/usr/bin/env python3
"""Screen normalized content scores plus deterministic hash noise at E12800."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth136_sparse_experts  # inserts the donor-adaptation S1 path
import meth136_matched_sparse_train as M136
import meth139_quantile_third_router as M139


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DOC / "meth150_shared_route_manifest.json"
MANIFEST_SHA = "6e911e65529be53dba65fbe382229f0f43e510eb76ab83f4404d4be0368ff5d9"
BETAS = (0.5, 1.0, 2.0, 4.0, 8.0)
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000
MASK = (1 << 64) - 1
SEED = 0x1841841841841841
MIX_A = 0x9E3779B97F4A7C15
MIX_B = 0xBF58476D1CE4E5B9
MIX_C = 0x94D049BB133111EB
MIX_D = 0xD6E8FEB86659FD93
MIX_E = 0xA24BAED4963EE407
MIX_F = 0x9FB21C651E98DF25


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS or record[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-184 resource stop: {record}")
    return record


def hash_gumbel(tokens, previous, positions, children, layer_id):
    assert tokens.ndim == previous.ndim == positions.ndim == 1
    assert children.shape == (tokens.size, 4)
    assert np.all((children >= 0) & (children < 1280))
    t = tokens.astype(np.uint64)[:, None, None]
    u = previous.astype(np.uint64)[:, None, None]
    p = positions.astype(np.uint64)[:, None, None]
    c = children.astype(np.uint64)[:, :, None]
    local = np.arange(10, dtype=np.uint64)[None, None, :]
    with np.errstate(over="ignore"):
        value = (np.uint64(SEED) ^ (t * np.uint64(MIX_A)) ^
                 (u * np.uint64(MIX_B)) ^ (p * np.uint64(MIX_C)) ^
                 (c * np.uint64(MIX_D)) ^
                 (np.uint64(layer_id) * np.uint64(MIX_E)) ^
                 (local * np.uint64(MIX_F)))
        value = value + np.uint64(MIX_A)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(MIX_B)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(MIX_C)
        value = value ^ (value >> np.uint64(31))
    uniform = ((value >> np.uint64(11)).astype(np.float64) + 0.5) / (1 << 53)
    assert np.all((uniform > 0.0) & (uniform < 1.0))
    return -np.log(-np.log(uniform))


def scalar_hash_gumbel(t, u, p, c, layer, local):
    value = (SEED ^ ((t * MIX_A) & MASK) ^ ((u * MIX_B) & MASK) ^
             ((p * MIX_C) & MASK) ^ ((c * MIX_D) & MASK) ^
             ((layer * MIX_E) & MASK) ^ ((local * MIX_F) & MASK))
    value = (value + MIX_A) & MASK
    value = ((value ^ (value >> 30)) * MIX_B) & MASK
    value = ((value ^ (value >> 27)) * MIX_C) & MASK
    value ^= value >> 31
    uniform = ((value >> 11) + 0.5) / (1 << 53)
    return -np.log(-np.log(uniform))


def hash_self_test():
    tokens = np.array([1, 123, 16948], dtype=np.int64)
    previous = np.array([0, 45, 11], dtype=np.int64)
    positions = np.array([0, 67, 511], dtype=np.int64)
    children = np.array([[0, 1, 2, 3], [89, 89, 89, 89],
                         [1279, 8, 63, 1279]], dtype=np.int64)
    actual = hash_gumbel(tokens, previous, positions, children, 23)
    expected = np.array([[[scalar_hash_gumbel(int(t), int(u), int(p), int(c), 23, j)
                           for j in range(10)] for c in row]
                         for t, u, p, row in zip(tokens, previous, positions, children)])
    assert np.array_equal(actual, expected)
    return {"tuple_count": 12, "candidate_count": 10,
            "first_gumbel": float(actual[0, 0, 0]),
            "last_gumbel": float(actual[-1, -1, -1])}


def summarize_layer(parent_counts, candidate_counts, score_gain, agreements):
    total = int(parent_counts.sum())
    assert total == int(candidate_counts.sum())
    hot = parent_counts >= 250
    assert hot.any()
    split = candidate_counts.reshape(1280, 10)
    worst_share = float(np.max(split[hot].max(axis=1) / parent_counts[hot]))
    return {"source_child": M139.summarize(parent_counts),
            "grandchild": M139.summarize(candidate_counts),
            "candidate_to_control_max_load_ratio":
                float(10 * candidate_counts.max() / parent_counts.max()),
            "hot_parent_count": int(hot.sum()),
            "hot_parent_worst_grandchild_share": worst_share,
            "mean_normalized_score_advantage_vs_pure_hash": float(score_gain / total),
            "raw_argmax_agreement": float(agreements / total)}


def layer_gates(row):
    return (row["candidate_to_control_max_load_ratio"] <= 1.25 and
            row["hot_parent_worst_grandchild_share"] <= 0.25 and
            row["grandchild"]["coverage"] >= 4000 and
            row["mean_normalized_score_advantage_vs_pure_hash"] >= 0.05 and
            row["raw_argmax_agreement"] >= 0.15)


def measure(name, sequences, width, betas, model, wrappers, projections,
            keys, start, device):
    parent_counts = np.zeros((24, 1280), dtype=np.int64)
    grand_counts = np.zeros((len(betas), 24, 12800), dtype=np.int64)
    score_gains = np.zeros((len(betas), 24), dtype=np.float64)
    agreements = np.zeros((len(betas), 24), dtype=np.int64)
    current = {"tokens": None, "previous": None, "positions": None}
    hooks = []
    for li, wrapper in enumerate(wrappers):
        def count(module, inputs, _output, layer_id=li):
            flat = inputs[0].reshape(-1, 896)
            child_gpu = module.last_selected.long()
            assert child_gpu.shape == (flat.shape[0], 4)
            q = F.linear(flat.float(), projections[layer_id])
            chosen_keys = keys[layer_id][child_gpu]
            scores = torch.einsum("nr,nkcr->nkc", q, chosen_keys)
            center = scores.mean(dim=-1, keepdim=True)
            spread = scores.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
            normalized = ((scores - center) / spread).cpu().numpy().astype(np.float64)
            child = child_gpu.cpu().numpy().astype(np.int64)
            assert np.isfinite(normalized).all()
            noise = hash_gumbel(current["tokens"], current["previous"],
                                current["positions"], child, layer_id)
            baseline = noise.argmax(axis=-1)
            raw = normalized.argmax(axis=-1)
            baseline_score = np.take_along_axis(normalized, baseline[..., None],
                                                axis=-1).squeeze(-1)
            parent_counts[layer_id] += np.bincount(child.ravel(), minlength=1280)
            for bi, beta in enumerate(betas):
                local = np.argmax(normalized + beta * noise, axis=-1)
                grand = child * 10 + local
                assert np.array_equal(grand // 10, child)
                grand_counts[bi, layer_id] += np.bincount(grand.ravel(), minlength=12800)
                chosen_score = np.take_along_axis(normalized, local[..., None],
                                                  axis=-1).squeeze(-1)
                score_gains[bi, layer_id] += float((chosen_score - baseline_score).sum())
                agreements[bi, layer_id] += int((local == raw).sum())
        hooks.append(wrapper.register_forward_hook(count))
    windows = 0
    total_tokens = 0
    try:
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
                budget(start, device)
    finally:
        for hook in hooks:
            hook.remove()
    rows_by_beta = {}
    for bi, beta in enumerate(betas):
        layers = [summarize_layer(parent_counts[li], grand_counts[bi, li],
                                  score_gains[bi, li], agreements[bi, li])
                  for li in range(24)]
        rows_by_beta[str(beta)] = {"per_layer": layers,
                                  "all_layer_gates_pass": all(map(layer_gates, layers))}
    assert all(int(row.sum()) == 4 * total_tokens for row in parent_counts)
    return {"sequences": len(sequences), "windows": windows,
            "tokens": total_tokens, "betas": rows_by_beta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stage = "bindings"
    device = None
    cells = {}
    try:
        assert M136.digest(MANIFEST) == MANIFEST_SHA
        hash_golden = hash_self_test()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        items = manifest["items"]
        assert len(items) == 24
        for item in items:
            assert M136.M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
            assert M136.M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item[
                "document_ids_sha256"]
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        parent, child = M136.bind_inputs()
        source_a, source_b, prefixes = M136.load_factor_bank()
        third = M136.load_third()
        raw_ids, chat, draws = M136.training_data()
        assert len(draws) == 256
        stage = "model_and_parity"
        model = M136.model_shell(device).eval()
        original_mlps = [layer.mlp for layer in model.model.layers]
        teacher, _ = M136.make_wrappers(model, parent["expert_state"],
            child["expert_state"], source_a, source_b, prefixes, device, "teacher")
        parity_items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
        with torch.inference_mode():
            reference = [model(torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                     device=device)[None], use_cache=False).logits.cpu()
                     for item in parity_items]
        for layer, original in zip(model.model.layers, original_mlps):
            layer.mlp = original
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        wrappers, _ = M136.make_wrappers(model, parent["expert_state"],
            child["expert_state"], source_a, source_b, prefixes, device, "control")
        with torch.inference_mode():
            parity_error = max(float((model(torch.as_tensor(item["prompt_ids"],
                    dtype=torch.long, device=device)[None], use_cache=False).logits.cpu().float()
                    - expected.float()).abs().max())
                    for item, expected in zip(parity_items, reference))
        assert parity_error == 0.0
        projections = [entry[0].to(device) for entry in third]
        keys = [entry[1].to(device) for entry in third]
        del parent, child, source_a, source_b, third, reference
        gc.collect()
        budget(start, device)

        stage = "training_route_screen"
        raw_sequences = [raw_ids[d["raw_row"], d["raw_offset"]:d["raw_offset"]+128][:-1]
                         for d in draws]
        chat_sequences = [chat[d["chat_index"]][1][:-1] for d in draws]
        cells["training_raw"] = measure("training_raw", raw_sequences, 512, BETAS,
            model, wrappers, projections, keys, start, device)
        cells["training_chat"] = measure("training_chat", chat_sequences, 512, BETAS,
            model, wrappers, projections, keys, start, device)
        passing = [beta for beta in BETAS if all(
            cells[name]["betas"][str(beta)]["all_layer_gates_pass"]
            for name in ("training_raw", "training_chat"))]
        chosen = passing[0] if passing else None
        print(json.dumps({"training_passing_betas": passing,
                          "chosen_beta": chosen, "budget": budget(start, device)}), flush=True)

        if chosen is not None:
            stage = "source_separated_route_screen"
            sequences = [item["document_ids"] for item in items]
            for width in (128, 512):
                name = f"source_separated_w{width}"
                cells[name] = measure(name, sequences, width, (chosen,), model,
                    wrappers, projections, keys, start, device)
        gate = chosen is not None and all(
            cells[f"source_separated_w{width}"]["betas"][str(chosen)]["all_layer_gates_pass"]
            for width in (128, 512))
        decision = ("route_screen_pass_native_cost_next" if gate else
                    "route_screen_fail")
        result = {"experiment": "METH-184-content-hash-third-route-screen",
                  "manifest_sha256": MANIFEST_SHA,
                  "source_bank_sha256": M136.EXACT_SHA,
                  "third_router_sha256": M136.THIRD_SHA,
                  "child_checkpoint_sha256": M136.CHILD_SHA,
                  "hash_seed_hex": hex(SEED), "hash_self_test": hash_golden,
                  "beta_candidates": BETAS, "chosen_beta": chosen,
                  "teacher_control_bf16_logit_max_abs_error": parity_error,
                  "cells": cells, "gates": {"training_selection": chosen is not None,
                                           "source_separated": gate},
                  "decision": decision,
                  "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)},
                  "scope": "Offline load/content screen; previously viewed external sources"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_DISK
        print(json.dumps({"decision": decision, "chosen_beta": chosen,
                          "source_separated_pass": gate,
                          "runtime": result["runtime"]}), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-184-failure",
                                       "stage": stage, "error": repr(error),
                                       "completed_cells": cells,
                                       "elapsed_seconds": time.monotonic() - start,
                                       "rss_bytes": psutil.Process().memory_info().rss,
                                       "gpu_peak_allocated_bytes":
                                           torch.cuda.max_memory_allocated(device)
                                           if device is not None else None},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
