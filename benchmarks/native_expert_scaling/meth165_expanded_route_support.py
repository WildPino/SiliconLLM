#!/usr/bin/env python3
"""Exact E12,800 route support after 1,280 fresh pairs, combined with METH-162."""

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
MANIFEST = DOC / "meth165_expanded_training_manifest.json"
MANIFEST_SHA = "2875ac50a4de005b56ad6403c6bd30e358b87eb21d7dbb1918738957855068ba"
TEACHER = DOC / "meth165_expanded_teacher_merged.json"
TABLE = DOC / "meth162_training_derived_shared_table.json"
TABLE_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
PRIOR = DOC / "meth162_postfailure_table_result.json"
PRIOR_SHA = "d10ede395f1f4008b754028c840a120d41523e47e08f4aa6b6848978a98df01b"
M164 = DOC / "meth164_cross_source_route_result.json"
M164_SHA = "1459921ea90fc77395353d4bafcc261e0413d7b34dfd5049acd35e7bbaf2ad63"
PRIOR_TOKENS = 887330
MAX_SECONDS = 40 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)


def budget(start, device):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss,
           "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS or
            row["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-165 support resource stop: {row}")
    return row


def empty():
    return {"all": np.zeros((24, 12800), dtype=np.int64),
            "content": np.zeros((24, 12800), dtype=np.int64),
            "tokens": 0}


def layer_rows(stats):
    tokens = stats["tokens"]
    assert tokens > 0
    rows = []
    for li in range(24):
        all_count = stats["all"][li]
        content_count = stats["content"][li]
        parent = all_count.reshape(1280, 10).sum(axis=1)
        content_parent = content_count.reshape(1280, 10).sum(axis=1)
        assert int(all_count.sum()) == 4 * tokens
        row = M150.layer_summary(parent, all_count, content_parent,
                                 content_count, tokens)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    for path, sha in ((MANIFEST, MANIFEST_SHA), (TEACHER, args.teacher_sha),
                      (TABLE, TABLE_SHA), (PRIOR, PRIOR_SHA), (M164, M164_SHA)):
        assert M136.digest(path) == sha, path
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    teacher = json.loads(TEACHER.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    m164 = json.loads(M164.read_text(encoding="utf-8"))
    assert m164["decision"] == "candidate_cross_source_route_pass_pending_support_and_cpu"
    assert manifest["meth164_route_result_sha256"] == M164_SHA
    assert teacher["experiment"] == "METH-165-expanded-teacher-merged"
    assert teacher["manifest_sha256"] == MANIFEST_SHA
    assert prior["decision"] == "postfailure_in_sample_diagnostic_only"
    assert prior["candidate_table_sha256"] == TABLE_SHA
    assert prior["fixed_training_input_tokens"] == PRIOR_TOKENS
    assert len(manifest["chat_rows"]) == len(manifest["raw_rows"]) == 1280
    assert len(teacher["rows"]) == 1280 and len(prior["layers"]) == 24
    table_record = json.loads(TABLE.read_text(encoding="utf-8"))
    table = {(r["token"], r["previous"], r["position"]) for r in table_record["tuples"]}
    assert len(table) == 138 and R150.load_table() <= table
    assert R150.golden(table) == {"content": 899, "structural": 890}
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
    parity = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    with torch.inference_mode():
        expected = [model(torch.as_tensor(r["prompt_ids"], dtype=torch.long,
                          device=device)[None], use_cache=False).logits.cpu()
                    for r in parity]
    for layer, original in zip(model.model.layers, original_mlps):
        layer.mlp = original
    del teacher_wrappers
    gc.collect()
    torch.cuda.empty_cache()
    wrappers, _ = M136.make_wrappers(
        model, parent["expert_state"], child["expert_state"],
        source_a, source_b, prefixes, device, "control")
    with torch.inference_mode():
        for row, reference in zip(parity, expected):
            ids = torch.as_tensor(row["prompt_ids"], dtype=torch.long, device=device)[None]
            assert torch.equal(model(ids, use_cache=False).logits.cpu(), reference)
    del parent, child, source_a, source_b
    gc.collect()
    torch.cuda.empty_cache()
    budget(started, device)

    cells = {"raw": empty(), "chat": empty()}
    current = {"tokens": None, "previous": None, "positions": None,
               "target": None}
    hooks = []
    for li, wrapper in enumerate(wrappers):
        def count(module, _inputs, _output, layer=li):
            source_ids = module.last_selected.cpu().numpy().astype(np.int64)
            assert source_ids.shape == (current["tokens"].size, 4)
            grand, shared = R150.route(
                current["tokens"], current["previous"], current["positions"],
                source_ids, layer, table)
            assert np.array_equal(grand // 10, source_ids)
            target = cells[current["target"]]
            target["all"][layer] += np.bincount(grand.reshape(-1), minlength=12800)
            target["content"][layer] += np.bincount(
                grand[~shared].reshape(-1), minlength=12800)
        hooks.append(wrapper.register_forward_hook(count))
    with torch.inference_mode():
        for index, (raw, chat, response) in enumerate(zip(
                manifest["raw_rows"], manifest["chat_rows"], teacher["rows"])):
            assert response["index"] == index
            assert response["train_row"] == chat["train_row"]
            assert response["prompt_ids_sha256"] == chat["prompt_ids_sha256"]
            raw_ids = np.asarray(raw["window_ids"], dtype=np.int32)
            prompt = np.asarray(chat["prompt_ids"], dtype=np.int32)
            continuation = np.asarray(response["continuation_ids"], dtype=np.int32)
            assert len(raw_ids) == 128
            assert M136.M17.sha(raw_ids.tobytes()) == raw["window_ids_sha256"]
            assert M136.M17.sha(prompt.tobytes()) == chat["prompt_ids_sha256"]
            assert M136.M17.sha(continuation.tobytes()) == response["continuation_ids_sha256"]
            for name, sequence in (("raw", raw_ids[:-1]),
                                   ("chat", np.r_[prompt, continuation][:-1])):
                tokens = np.asarray(sequence, dtype=np.int64)
                current["tokens"] = tokens
                current["previous"] = np.r_[0, tokens[:-1]]
                current["positions"] = np.arange(tokens.size, dtype=np.int64)
                current["target"] = name
                ids = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]
                model(ids, use_cache=False)
                cells[name]["tokens"] += tokens.size
            if (index + 1) % 128 == 0:
                print(json.dumps({"completed_pairs": index + 1,
                                  "raw_tokens": cells["raw"]["tokens"],
                                  "chat_tokens": cells["chat"]["tokens"],
                                  "budget": budget(started, device)}), flush=True)
    for hook in hooks:
        hook.remove()
    new_stats = empty()
    new_stats["all"] = cells["raw"]["all"] + cells["chat"]["all"]
    new_stats["content"] = cells["raw"]["content"] + cells["chat"]["content"]
    new_stats["tokens"] = cells["raw"]["tokens"] + cells["chat"]["tokens"]
    combined = empty()
    combined["tokens"] = PRIOR_TOKENS + new_stats["tokens"]
    for li, row in enumerate(prior["layers"]):
        old_all = np.asarray(row["all_slot_counts"], dtype=np.int64)
        old_content = np.asarray(row["content_slot_counts"], dtype=np.int64)
        assert old_all.shape == old_content.shape == (12800,)
        assert int(old_all.sum()) == 4 * PRIOR_TOKENS
        combined["all"][li] = old_all + new_stats["all"][li]
        combined["content"][li] = old_content + new_stats["content"][li]
    raw_layers = layer_rows(cells["raw"])
    chat_layers = layer_rows(cells["chat"])
    new_layers = layer_rows(new_stats)
    combined_layers = layer_rows(combined)
    gates = {
        "teacher_control_bf16_parity": True,
        "combined_content_coverage": all(r["content"]["coverage"] >= 10000
                                         for r in combined_layers),
        "combined_active_content_median": all(r["content_active_selection_p50"] >= 50
                                              for r in combined_layers),
        "combined_under_32_fraction": all(r["content_slots_under_32_fraction"] <= 0.50
                                          for r in combined_layers),
        "combined_load_ratio": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                                   for r in combined_layers),
        "combined_hot_parent_share": all(
            r["content_hot_parent_worst_grandchild_share"] <= 0.25
            for r in combined_layers),
        "new_raw_load_ratio": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                                  for r in raw_layers),
        "new_raw_hot_parent_share": all(
            r["content_hot_parent_worst_grandchild_share"] <= 0.25
            for r in raw_layers),
        "new_raw_hot_parents_present": all(r["content_hot_parent_count"] > 0
                                           for r in raw_layers),
        "new_chat_load_ratio": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                                   for r in chat_layers),
        "new_chat_hot_parent_share": all(
            r["content_hot_parent_worst_grandchild_share"] <= 0.25
            for r in chat_layers),
        "new_chat_hot_parents_present": all(r["content_hot_parent_count"] > 0
                                            for r in chat_layers)}
    result = {"experiment": "METH-165-expanded-independent-route-support",
              "manifest_sha256": MANIFEST_SHA,
              "teacher_sha256": args.teacher_sha,
              "candidate_table_sha256": TABLE_SHA,
              "prior_meth162_result_sha256": PRIOR_SHA,
              "cross_source_meth164_result_sha256": M164_SHA,
              "source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "new_pairs": 1280, "combined_pairs": 3840,
              "new_raw_tokens": cells["raw"]["tokens"],
              "new_chat_tokens": cells["chat"]["tokens"],
              "combined_tokens": combined["tokens"],
              "new_raw_layers": raw_layers,
              "new_chat_layers": chat_layers,
              "new_combined_layers": new_layers,
              "all_3840_layers": combined_layers,
              "gates": gates,
              "decision": "tenfold_route_support_pass_training_protocol_pending"
                          if all(gates.values()) else "tenfold_route_support_fail",
              "runtime": {**budget(started, device),
                          "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "combined_tokens": combined["tokens"],
                      "combined_min_coverage": min(r["content"]["coverage"]
                          for r in combined_layers),
                      "combined_min_median": min(r["content_active_selection_p50"]
                          for r in combined_layers),
                      "combined_worst_hot_share": max(
                          r["content_hot_parent_worst_grandchild_share"]
                          for r in combined_layers),
                      "sha256": M136.digest(args.out),
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
