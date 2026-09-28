#!/usr/bin/env python3
"""Prospective old/new structural-table route screen on PG19 shard 10."""

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
MANIFEST = DOC / "meth164_cross_source_route_manifest.json"
MANIFEST_SHA = "d8420488104990ab63338721837c4f100135c45a07e787f2708eb741dfb63c9b"
TEACHER = DOC / "meth164_cross_source_teacher_merged.json"
TABLE = DOC / "meth162_training_derived_shared_table.json"
TABLE_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
MAX_SECONDS = 20 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS or
            result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-164 route resource stop: {result}")
    return result


def counts():
    return {"parent": np.zeros((24, 1280), dtype=np.int64),
            "grand": np.zeros((24, 12800), dtype=np.int64),
            "content_parent": np.zeros((24, 1280), dtype=np.int64),
            "content_grand": np.zeros((24, 12800), dtype=np.int64)}


def summary(counter, tokens):
    result = []
    for layer in range(24):
        row = M150.layer_summary(counter["parent"][layer],
                                 counter["grand"][layer],
                                 counter["content_parent"][layer],
                                 counter["content_grand"][layer], tokens)
        row["all_slot_counts"] = counter["grand"][layer].tolist()
        row["content_slot_counts"] = counter["content_grand"][layer].tolist()
        result.append(row)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert M136.digest(MANIFEST) == MANIFEST_SHA
    assert M136.digest(TABLE) == TABLE_SHA
    assert M136.digest(TEACHER) == args.teacher_sha
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    teacher = json.loads(TEACHER.read_text(encoding="utf-8"))
    assert teacher["experiment"] == "METH-164-cross-source-teacher-merged"
    assert teacher["manifest_sha256"] == MANIFEST_SHA
    assert len(teacher["rows"]) == len(manifest["chat_rows"]) == 128
    assert len(manifest["raw_rows"]) == 128
    assert len(manifest["document_rows"]) == 24
    record = json.loads(TABLE.read_text(encoding="utf-8"))
    table = {(r["token"], r["previous"], r["position"]) for r in record["tuples"]}
    old_table = R150.load_table()
    assert len(table) == 138 and len(old_table) == 75 and old_table <= table
    assert R150.golden(old_table) == R150.golden(table)
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    gpus = [i for i in range(torch.cuda.device_count())
            if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpus) == 1
    device = torch.device(f"cuda:{gpus[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    parent, child = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    teacher_wrappers, _ = M136.make_wrappers(
        model, parent["expert_state"], child["expert_state"],
        source_a, source_b, prefixes, device, "teacher")
    parity_items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    expected = []
    with torch.inference_mode():
        for item in parity_items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            expected.append(model(ids, use_cache=False).logits.cpu())
    for layer, original in zip(model.model.layers, original_mlps):
        layer.mlp = original
    del teacher_wrappers
    gc.collect()
    torch.cuda.empty_cache()
    wrappers, _ = M136.make_wrappers(
        model, parent["expert_state"], child["expert_state"],
        source_a, source_b, prefixes, device, "control")
    with torch.inference_mode():
        for item, reference in zip(parity_items, expected):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            assert torch.equal(model(ids, use_cache=False).logits.cpu(), reference)
    del parent, child, source_a, source_b
    gc.collect()
    budget(started, device)

    chat = []
    for index, (prompt, response) in enumerate(zip(manifest["chat_rows"], teacher["rows"])):
        assert response["index"] == index and response["source_row"] == prompt["source_row"]
        assert response["prompt_ids_sha256"] == prompt["prompt_ids_sha256"]
        ids = np.asarray(prompt["prompt_ids"], dtype=np.int32)
        continuation = np.asarray(response["continuation_ids"], dtype=np.int32)
        assert M136.M17.sha(ids.tobytes()) == prompt["prompt_ids_sha256"]
        assert M136.M17.sha(continuation.tobytes()) == response["continuation_ids_sha256"]
        chat.append(np.r_[ids, continuation][:-1])
    raw = []
    for entry in manifest["raw_rows"]:
        ids = np.asarray(entry["span_ids"], dtype=np.int32)
        assert len(ids) == 128 and M136.M17.sha(ids.tobytes()) == entry["span_ids_sha256"]
        raw.append(ids[:-1])
    documents = []
    for entry in manifest["document_rows"]:
        ids = np.asarray(entry["span_ids"], dtype=np.int32)
        assert len(ids) == 1024 and M136.M17.sha(ids.tobytes()) == entry["span_ids_sha256"]
        documents.append(ids)
    cells = [("chat", chat, None), ("raw", raw, None),
             ("documents_w128", documents, 128),
             ("documents_w512", documents, 512)]
    outputs = {}
    for name, sequences, width in cells:
        old, candidate = counts(), counts()
        state = {"tokens": None, "previous": None, "positions": None,
                 "added_positions": 0}
        hooks = []
        for li, wrapper in enumerate(wrappers):
            def count(module, _inputs, _output, layer=li):
                source_ids = module.last_selected.cpu().numpy().astype(np.int64)
                assert source_ids.shape == (state["tokens"].size, 4)
                old_grand, old_shared = R150.route(
                    state["tokens"], state["previous"], state["positions"],
                    source_ids, layer, old_table)
                new_grand, new_shared = R150.route(
                    state["tokens"], state["previous"], state["positions"],
                    source_ids, layer, table)
                assert np.array_equal(old_grand // 10, source_ids)
                assert np.array_equal(new_grand // 10, source_ids)
                assert np.all(old_shared <= new_shared)
                if layer == 0:
                    state["added_positions"] += int((new_shared & ~old_shared).sum())
                parent_counts = np.bincount(source_ids.reshape(-1), minlength=1280)
                for target, grand, shared in ((old, old_grand, old_shared),
                                               (candidate, new_grand, new_shared)):
                    target["parent"][layer] += parent_counts
                    target["grand"][layer] += np.bincount(grand.reshape(-1), minlength=12800)
                    target["content_parent"][layer] += np.bincount(
                        source_ids[~shared].reshape(-1), minlength=1280)
                    target["content_grand"][layer] += np.bincount(
                        grand[~shared].reshape(-1), minlength=12800)
            hooks.append(wrapper.register_forward_hook(count))
        tokens = 0
        windows = 0
        with torch.inference_mode():
            for index, sequence in enumerate(sequences):
                cuts = [(0, len(sequence))] if width is None else [
                    (i, min(i + width, len(sequence)))
                    for i in range(0, len(sequence), width)]
                for first, last in cuts:
                    fragment = np.asarray(sequence[first:last], dtype=np.int64)
                    state["tokens"] = fragment
                    state["previous"] = np.r_[0, fragment[:-1]]
                    state["positions"] = np.arange(fragment.size, dtype=np.int64)
                    ids = torch.as_tensor(fragment, dtype=torch.long, device=device)[None]
                    model(ids, use_cache=False)
                    tokens += fragment.size
                    windows += 1
                if (index + 1) % (32 if name in ("chat", "raw") else 8) == 0:
                    print(json.dumps({"cell": name, "sequences": index + 1,
                                      "budget": budget(started, device)}), flush=True)
        for hook in hooks:
            hook.remove()
        old_layers = summary(old, tokens)
        candidate_layers = summary(candidate, tokens)
        assert all(int(old["parent"][li].sum()) == 4 * tokens for li in range(24))
        assert all(candidate_layers[li]["structural_selections"] -
                   old_layers[li]["structural_selections"] ==
                   4 * state["added_positions"] for li in range(24))
        outputs[name] = {"sequences": len(sequences), "windows": windows,
                         "tokens": tokens,
                         "newly_shared_input_positions": state["added_positions"],
                         "old_layers": old_layers,
                         "candidate_layers": candidate_layers}
        print(json.dumps({"cell_complete": name, "tokens": tokens,
                          "newly_shared_input_positions": state["added_positions"],
                          "candidate_worst_hot_share": max(
                              row["content_hot_parent_worst_grandchild_share"]
                              for row in candidate_layers),
                          "budget": budget(started, device)}), flush=True)
    proposed = [row for cell in outputs.values() for row in cell["candidate_layers"]]
    gates = {"bf16_parity": True,
             "content_coverage": all(r["content"]["coverage"] >= 4000 for r in proposed),
             "content_load_ratio": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                                       for r in proposed),
             "content_hot_parent_share": all(
                 r["content_hot_parent_worst_grandchild_share"] <= 0.25 for r in proposed),
             "hot_parents_present": all(r["content_hot_parent_count"] > 0 for r in proposed),
             "chat_shared_traffic": all(
                 r["structural_selections"] > 0 for r in outputs["chat"]["candidate_layers"])}
    result = {"experiment": "METH-164-cross-source-route-validation",
              "manifest_sha256": MANIFEST_SHA, "teacher_sha256": args.teacher_sha,
              "original_table_sha256": R150.TABLE_SHA,
              "candidate_table_sha256": TABLE_SHA,
              "source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "cells": outputs, "gates": gates,
              "decision": "candidate_cross_source_route_pass_pending_support_and_cpu"
                          if all(gates.values()) else "candidate_cross_source_route_fail",
              "runtime": {**budget(started, device),
                          "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "summary": {name: {"tokens": c["tokens"],
                          "newly_shared_input_positions": c["newly_shared_input_positions"],
                          "minimum_content_coverage": min(r["content"]["coverage"]
                              for r in c["candidate_layers"]),
                          "worst_hot_share": max(
                              r["content_hot_parent_worst_grandchild_share"]
                              for r in c["candidate_layers"])}
                          for name, c in outputs.items()},
                      "sha256": M136.digest(args.out), "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
