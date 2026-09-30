#!/usr/bin/env python3
"""Fit shared content centroids and per-parent route biases before B training."""

import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth184_content_hash_third_route as P
import meth175_matched_long_train as M175
import meth175_training_draws as D175
import meth150_shared_structure_route as R150


M136 = P.M136
ROOT, DOC = P.ROOT, P.DOC
FIT_DRAWS = 1024
RESERVED_DRAWS = 256
CHILDREN = 9
RANK = 32
KMEANS_SAMPLE = 65536
KMEANS_STEPS = 16
BIAS_STEPS = 30
TEMPERATURE = .05
MAX_SECONDS = 45 * 60
MAX_RSS = 20 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))
MAX_DISK = 1_000_000_000


def budget(start, device):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss,
           "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS or row[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-204 resource stop: {row}")
    return row


def collect(name, sequences, width, model, wrappers, projections, table,
            start, device):
    q_parts = [[] for _ in range(24)]
    parent_parts = [[] for _ in range(24)]
    current = {"shared": None}
    hooks = []
    totals = {"windows": 0, "tokens": 0, "structural_selections": 0}
    for li, wrapper in enumerate(wrappers):
        def capture(module, inputs, _output, layer=li):
            flat = inputs[0].reshape(-1, 896)
            parents = module.last_selected.long()
            assert parents.shape == (flat.shape[0], 4)
            shared = current["shared"]
            assert shared is not None and shared.shape == (flat.shape[0],)
            q = F.normalize(F.linear(flat.float(), projections[layer]), dim=-1)
            q_host = q.cpu().numpy()
            parent_host = parents.cpu().numpy().astype(np.int16)
            q_content = q_host[~shared]
            parent_content = parent_host[~shared]
            assert np.isfinite(q_content).all()
            assert np.all((parent_content >= 0) & (parent_content < 1280))
            q_parts[layer].append(np.repeat(q_content, 4, axis=0))
            parent_parts[layer].append(parent_content.reshape(-1).copy())
        hooks.append(wrapper.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            for si, sequence in enumerate(sequences):
                for first in range(0, len(sequence), width):
                    tokens = np.asarray(sequence[first:first + width], dtype=np.int64)
                    assert 0 < tokens.size <= width
                    previous = np.r_[0, tokens[:-1]]
                    positions = np.arange(tokens.size, dtype=np.int64)
                    shared = (R150.shared_mask(tokens, previous, positions, table) |
                              (tokens == 151644) | (tokens == 151645))
                    current["shared"] = shared
                    ids = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]
                    model(ids, use_cache=False)
                    totals["windows"] += 1
                    totals["tokens"] += tokens.size
                    totals["structural_selections"] += 4 * int(shared.sum())
                if (si + 1) % 64 == 0:
                    print(json.dumps({"cell": name, "sequences": si + 1,
                                      "budget": budget(start, device)}), flush=True)
                budget(start, device)
    finally:
        for hook in hooks:
            hook.remove()
    data = []
    for li in range(24):
        q = np.concatenate(q_parts[li], axis=0)
        parent = np.concatenate(parent_parts[li], axis=0)
        assert q.shape == (parent.size, RANK)
        assert parent.size + totals["structural_selections"] == 4 * totals["tokens"]
        data.append((q, parent))
    totals["content_selections_per_layer"] = int(data[0][1].size)
    totals["sequences"] = len(sequences)
    return data, totals


def fit_one(q, parent, layer, device):
    assert q.ndim == 2 and q.shape[1] == RANK
    assert parent.shape == (q.shape[0],) and q.shape[0] >= KMEANS_SAMPLE
    rng = np.random.default_rng(204204 + layer)
    indices = rng.choice(q.shape[0], size=KMEANS_SAMPLE, replace=False)
    sampled = torch.from_numpy(q[indices].copy()).to(device)
    initial = F.normalize(sampled[
        np.linspace(0, KMEANS_SAMPLE - 1, CHILDREN, dtype=np.int64)].clone(),
        dim=-1)
    centers = initial.clone()
    with torch.inference_mode():
        for _ in range(KMEANS_STEPS):
            labels = (sampled @ centers.T).argmax(dim=-1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, sampled)
            counts = torch.bincount(labels, minlength=CHILDREN)
            updated = F.normalize(sums, dim=-1)
            centers = torch.where(counts[:, None] > 0, updated, initial)
        features = torch.from_numpy(q).to(device)
        parents = torch.from_numpy(parent.astype(np.int64)).to(device)
        scores = features @ centers.T
        parent_counts = torch.bincount(parents, minlength=1280).float()
        target = parent_counts[:, None] / CHILDREN
        bias = torch.zeros((1280, CHILDREN), device=device)
        for _ in range(BIAS_STEPS):
            probabilities = F.softmax((scores + bias[parents]) / TEMPERATURE,
                                      dim=-1)
            soft_counts = torch.zeros_like(bias)
            soft_counts.index_add_(0, parents, probabilities)
            bias += TEMPERATURE * torch.log((target + 1) / (soft_counts + 1))
            bias -= bias.mean(dim=-1, keepdim=True)
        assert bool(torch.isfinite(centers).all() and torch.isfinite(bias).all())
        result = (centers.cpu().numpy().copy(), bias.cpu().numpy().copy())
    del features, parents, scores, sampled
    torch.cuda.empty_cache()
    return result


def summarize_layer(q, parent, centers, bias, structural, device):
    assert q.shape == (parent.size, RANK)
    with torch.inference_mode():
        features = torch.from_numpy(q).to(device)
        parents = torch.from_numpy(parent.astype(np.int64)).to(device)
        keys = torch.from_numpy(centers).to(device)
        offsets = torch.from_numpy(bias).to(device)
        raw_scores = features @ keys.T
        normalized = ((raw_scores - raw_scores.mean(dim=-1, keepdim=True)) /
                      raw_scores.std(dim=-1, unbiased=False,
                                     keepdim=True).clamp_min(1e-6))
        local = (raw_scores + offsets[parents]).argmax(dim=-1)
        grand = parents * CHILDREN + local
        counts_parent = torch.bincount(parents, minlength=1280).cpu().numpy()
        counts_grand = torch.bincount(grand, minlength=1280 * CHILDREN).cpu().numpy()
        hot = counts_parent >= 250
        split = counts_grand.reshape(1280, CHILDREN)
        worst_share = float(np.max(split[hot].max(axis=1) / counts_parent[hot])) \
            if hot.any() else 0.0
        agreement = float((local == raw_scores.argmax(dim=-1)).float().mean())
        advantage = float(normalized.gather(1, local[:, None]).mean())
        count = int(parent.size)
        assert count == int(counts_parent.sum()) == int(counts_grand.sum())
        row = {"content_selections": count, "structural_selections": structural,
               "content_coverage": int((counts_grand > 0).sum()),
               "hot_parent_count": int(hot.sum()),
               "candidate_to_control_max_load_ratio":
                   float(CHILDREN * counts_grand.max() / counts_parent.max()),
               "hot_parent_worst_child_share": worst_share,
               "mean_standardized_score_advantage": advantage,
               "raw_argmax_agreement": agreement,
               "bias_max_abs": float(np.abs(bias).max())}
        assert all(math.isfinite(value) for value in row.values())
        row["gates"] = {"load_ratio": row["candidate_to_control_max_load_ratio"] <= 1.25,
                        "hot_parent_share": worst_share <= .25,
                        "coverage": row["content_coverage"] >= 4000,
                        "score_advantage": advantage >= .05,
                        "raw_argmax_agreement": agreement >= .15}
        return row


def evaluate_cell(data, totals, centers, biases, device):
    layers = [summarize_layer(q, parent, centers[li], biases[li],
                              totals["structural_selections"], device)
              for li, (q, parent) in enumerate(data)]
    return {**totals, "layers": layers,
            "all_layer_gates_pass": all(all(row["gates"].values()) for row in layers)}


def sequences_for(draws, raw_ids, chat):
    raw = [raw_ids[d["raw_row"], d["raw_offset"]:
                   d["raw_offset"] + 127] for d in draws]
    chats = [chat[d["chat_index"]][1][:-1] for d in draws]
    return raw, chats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--router", required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists() and not args.router.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.router.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage = "bindings"
    device = None
    cells = {}
    try:
        assert M136.digest(P.MANIFEST) == P.MANIFEST_SHA
        assert M136.digest(DOC / "meth175_training_draws.json") == M175.DRAWS_SHA
        manifest = json.loads(P.MANIFEST.read_text(encoding="utf-8"))
        assert len(manifest["items"]) == 24
        for item in manifest["items"]:
            assert M136.M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
            assert M136.M17.sha(np.asarray(item["document_ids"],
                       dtype=np.int32).tobytes()) == item["document_ids_sha256"]
        table = M175.load_table()
        parent, child = M136.bind_inputs()
        source_a, source_b, prefixes = M136.load_factor_bank()
        third = M136.load_third()
        raw_ids, chat, draws = D175.load(M175.DRAWS_SHA)
        assert len(draws) == 3840
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        M136.MAX_SECONDS = MAX_SECONDS
        M136.MAX_RSS = MAX_RSS
        M136.MAX_GPU = MAX_GPU
        stage = "model_and_parity"
        model = M136.model_shell(device).eval()
        original_mlps = [layer.mlp for layer in model.model.layers]
        teacher, _ = M136.make_wrappers(model, parent["expert_state"],
            child["expert_state"], source_a, source_b, prefixes,
            device, "teacher")
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
            child["expert_state"], source_a, source_b, prefixes,
            device, "control")
        with torch.inference_mode():
            parity_error = max(float((model(torch.as_tensor(item["prompt_ids"],
                    dtype=torch.long, device=device)[None], use_cache=False).logits.cpu().float()
                    - expected.float()).abs().max())
                    for item, expected in zip(parity_items, reference))
        assert parity_error == 0.0
        projections = [entry[0].to(device) for entry in third]
        del parent, child, source_a, source_b, third, reference
        gc.collect()
        budget(started, device)
        stage = "fit_capture"
        fit_raw, fit_chat = sequences_for(draws[:FIT_DRAWS], raw_ids, chat)
        raw_data, raw_totals = collect("fit_raw", fit_raw, 512, model,
            wrappers, projections, table, started, device)
        chat_data, chat_totals = collect("fit_chat", fit_chat, 512, model,
            wrappers, projections, table, started, device)
        del fit_raw, fit_chat
        stage = "fit_route"
        centers, biases = [], []
        for li in range(24):
            q = np.concatenate((raw_data[li][0], chat_data[li][0]), axis=0)
            parent_ids = np.concatenate((raw_data[li][1], chat_data[li][1]), axis=0)
            key, bias = fit_one(q, parent_ids, li, device)
            centers.append(key)
            biases.append(bias)
            del q, parent_ids
            if (li + 1) % 4 == 0:
                print(json.dumps({"fitted_layers": li + 1,
                                  "budget": budget(started, device)}), flush=True)
        centers = np.stack(centers).astype(np.float32)
        biases = np.stack(biases).astype(np.float32)
        projection_bank = np.stack([entry.cpu().numpy() for entry in projections])
        np.savez(args.router, projection=projection_bank,
                 content_keys=centers, parent_bias=biases)
        assert args.router.stat().st_size < MAX_DISK
        with np.load(args.router, allow_pickle=False) as readback:
            assert np.array_equal(readback["projection"], projection_bank)
            assert np.array_equal(readback["content_keys"], centers)
            assert np.array_equal(readback["parent_bias"], biases)
        router_sha = M136.digest(args.router)
        stage = "fit_screen"
        cells["fit_raw"] = evaluate_cell(raw_data, raw_totals, centers, biases, device)
        cells["fit_chat"] = evaluate_cell(chat_data, chat_totals, centers, biases, device)
        del raw_data, chat_data
        gc.collect()
        passing_fit = all(cells[name]["all_layer_gates_pass"]
                          for name in ("fit_raw", "fit_chat"))
        print(json.dumps({"fit_pass": passing_fit,
                          "budget": budget(started, device)}), flush=True)
        passing_reserved = False
        passing_source = False
        if passing_fit:
            stage = "reserved_screen"
            reserved_raw, reserved_chat = sequences_for(
                draws[FIT_DRAWS:FIT_DRAWS + RESERVED_DRAWS], raw_ids, chat)
            for name, sequences in (("reserved_raw", reserved_raw),
                                    ("reserved_chat", reserved_chat)):
                data, totals = collect(name, sequences, 512, model, wrappers,
                                       projections, table, started, device)
                cells[name] = evaluate_cell(data, totals, centers, biases, device)
                del data
                gc.collect()
                budget(started, device)
            passing_reserved = all(cells[name]["all_layer_gates_pass"]
                                   for name in ("reserved_raw", "reserved_chat"))
            print(json.dumps({"reserved_pass": passing_reserved,
                              "budget": budget(started, device)}), flush=True)
        if passing_reserved:
            stage = "source_separated_screen"
            sequences = [item["document_ids"] for item in manifest["items"]]
            for width in (128, 512):
                name = f"source_w{width}"
                data, totals = collect(name, sequences, width, model, wrappers,
                                       projections, table, started, device)
                cells[name] = evaluate_cell(data, totals, centers, biases, device)
                del data
                gc.collect()
                budget(started, device)
            passing_source = all(cells[f"source_w{width}"]["all_layer_gates_pass"]
                                 for width in (128, 512))
        stage = "report"
        ledger = {"projection_stored_bytes": int(projection_bank.nbytes),
                  "shared_keys_stored_bytes": int(centers.nbytes),
                  "parent_bias_stored_bytes": int(biases.nbytes),
                  "ideal_addressed_projection_keys_selected_bias_per_token":
                      24 * (32 * 896 + 9 * 32 + 4 * 9) * 4,
                  "bias_stored_bytes_at_tenfold_parents": int(biases.nbytes * 10)}
        decision = ("route_screen_pass_native_cost_next" if passing_source else
                    "route_screen_fail")
        result = {"experiment": "METH-204-shared-content-keys-route-screen",
                  "source_bank_sha256": M136.EXACT_SHA,
                  "third_projection_source_sha256": M136.THIRD_SHA,
                  "child_checkpoint_sha256": M136.CHILD_SHA,
                  "draws_sha256": M175.DRAWS_SHA,
                  "structural_table_sha256": M175.TABLE_SHA,
                  "source_manifest_sha256": P.MANIFEST_SHA,
                  "teacher_control_bf16_logit_max_abs_error": parity_error,
                  "fit_rule": {"draws": FIT_DRAWS, "reserved_draws": RESERVED_DRAWS,
                               "children": CHILDREN, "rank": RANK,
                               "kmeans_sample": KMEANS_SAMPLE,
                               "kmeans_steps": KMEANS_STEPS,
                               "bias_steps": BIAS_STEPS,
                               "temperature": TEMPERATURE},
                  "router_artifact": {"path": str(args.router.resolve()),
                                      "sha256": router_sha,
                                      "bytes": args.router.stat().st_size,
                                      "readback_exact": True},
                  "ledger": ledger, "cells": cells,
                  "gates": {"fit": passing_fit, "reserved": passing_reserved,
                            "source_separated": passing_source},
                  "decision": decision,
                  "runtime": {**budget(started, device),
                              "gpu": torch.cuda.get_device_name(device)},
                  "scope": "Route-only screen on previously used training and source data"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size + args.router.stat().st_size < MAX_DISK
        print(json.dumps({"decision": decision, "gates": result["gates"],
                          "runtime": result["runtime"]}), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-204-failure",
                                       "stage": stage, "error": repr(error),
                                       "completed_cells": cells,
                                       "elapsed_seconds": time.monotonic() - started,
                                       "rss_bytes": psutil.Process().memory_info().rss,
                                       "gpu_peak_allocated_bytes":
                                           torch.cuda.max_memory_allocated(device)
                                           if device is not None else None},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
