#!/usr/bin/env python3
"""Prospective source-ordered screen of the frozen METH-168 route."""

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
MANIFEST = DOC / "meth173_fresh_recurrence_route_manifest.json"
MANIFEST_SHA = "6ebfc36fd452f2a20cfcc9e8210505e36b725ef95992949655c19f3c94eaee20"
TEACHER = DOC / "meth173_fresh_recurrence_teacher.json"
M172 = DOC / "meth172_low_recurrence_route_result.json"
M172_SHA = "3e71f5df77c54081e7d39b106c2cc1138902e9b91cf5b836d18eee350714dba9"
TABLE = DOC / "meth172_threshold8_shared_table.json"
TABLE_SHA = "e89ff2cecbf887b85b5e361d47a8b918d8ccb93bd5c385a580ba993b93ac72db"
BASE_TABLE = DOC / "meth162_training_derived_shared_table.json"
BASE_TABLE_SHA = "dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1"
MAX_SECONDS = 20 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS or
            result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-173 route resource stop: {result}")
    return result


def empty_counts():
    return {"all": np.zeros((24, 12800), dtype=np.int64),
            "content": np.zeros((24, 12800), dtype=np.int64), "tokens": 0}


def add_counts(*cells):
    out = empty_counts()
    for cell in cells:
        out["all"] += cell["all"]
        out["content"] += cell["content"]
        out["tokens"] += cell["tokens"]
    return out


def summarize(cell):
    rows = []
    for li in range(24):
        all_count = cell["all"][li]
        content_count = cell["content"][li]
        parent = all_count.reshape(1280, 10).sum(axis=1)
        content_parent = content_count.reshape(1280, 10).sum(axis=1)
        assert int(all_count.sum()) == 4 * cell["tokens"]
        row = M150.layer_summary(parent, all_count, content_parent,
                                 content_count, cell["tokens"])
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


def check_cell(rows, coverage, support=False, structural_cap=0.25):
    return {
        "content_coverage": all(r["content"]["coverage"] >= coverage for r in rows),
        "content_load_ratio": all(r["content_to_own_parent_max_load_ratio"] <= 1.25
                                  for r in rows),
        "content_hot_parent_share": all(
            r["content_hot_parent_worst_grandchild_share"] <= 0.25 for r in rows),
        "hot_parents_present": all(r["content_hot_parent_count"] > 0 for r in rows),
        "structural_share": all(r["structural_selections"] <=
                                structural_cap * r["total"]["selections"] for r in rows),
        **({"active_median": all(r["content_active_selection_p50"] >= 50
                                  for r in rows),
            "under_32": all(r["content_slots_under_32_fraction"] <= 0.50
                            for r in rows)} if support else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    for path, expected in ((MANIFEST, MANIFEST_SHA), (TEACHER, args.teacher_sha),
                           (M172, M172_SHA), (TABLE, TABLE_SHA),
                           (BASE_TABLE, BASE_TABLE_SHA)):
        assert M136.digest(path) == expected
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    teacher = json.loads(TEACHER.read_text(encoding="utf-8"))
    train = json.loads(M172.read_text(encoding="utf-8"))
    assert manifest["candidate_meth172_result_sha256"] == M172_SHA
    assert teacher["experiment"] == "METH-173-fresh-recurrence-teacher-merged"
    assert teacher["manifest_sha256"] == MANIFEST_SHA
    assert len(manifest["chat_rows"]) == len(teacher["rows"]) == 512
    assert len(manifest["raw_rows"]) == 512
    assert len(manifest["document_rows"]) == 24
    assert train["decision"] == "postfailure_in_sample_candidate_pass_fresh_source_pending"
    assert train["cells"]["training"]["tokens"] == 1330096
    assert train["candidate_table_sha256"] == TABLE_SHA
    record = json.loads(TABLE.read_text(encoding="utf-8"))
    table = {(r["token"], r["previous"], r["position"]) for r in record["tuples"]}
    base_record = json.loads(BASE_TABLE.read_text(encoding="utf-8"))
    base_table = {(r["token"], r["previous"], r["position"])
                  for r in base_record["tuples"]}
    assert len(table) == 1685 and len(base_table) == 138 and base_table <= table
    assert R150.golden(base_table) == R150.golden(table)

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
    inputs = [("raw", raw, None), ("chat", chat, None),
              ("documents_w128", documents, 128),
              ("documents_w512", documents, 512)]
    counters = {}
    outputs = {}
    for name, sequences, width in inputs:
        old, candidate = empty_counts(), empty_counts()
        state = {"tokens": None, "previous": None, "positions": None,
                 "newly_shared_positions": 0}
        hooks = []
        for li, wrapper in enumerate(wrappers):
            def count(module, _inputs, _output, layer=li):
                source = module.last_selected.cpu().numpy().astype(np.int64)
                assert source.shape == (state["tokens"].size, 4)
                assert np.all((source >= 0) & (source < 1280))
                old_base_grand, old_base_shared = R150.route(
                    state["tokens"], state["previous"], state["positions"],
                    source, layer, base_table)
                new_base_grand, new_base_shared = R150.route(
                    state["tokens"], state["previous"], state["positions"],
                    source, layer, table)
                delimiter = ((state["tokens"] == 151644) |
                             (state["tokens"] == 151645))
                old_shared = old_base_shared | delimiter
                shared = new_base_shared | delimiter
                assert np.all(old_shared <= shared)
                old_grand = np.where(old_shared[:, None], source * 10, old_base_grand)
                grand = np.where(shared[:, None], source * 10, new_base_grand)
                assert np.array_equal(old_grand // 10, source)
                assert np.array_equal(grand // 10, source)
                if layer == 0:
                    state["newly_shared_positions"] += int((shared & ~old_shared).sum())
                for target, slots, mask in ((old, old_grand, old_shared),
                                            (candidate, grand, shared)):
                    target["all"][layer] += np.bincount(slots.reshape(-1), minlength=12800)
                    target["content"][layer] += np.bincount(
                        slots[~mask].reshape(-1), minlength=12800)
            hooks.append(wrapper.register_forward_hook(count))
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
                    old["tokens"] += fragment.size
                    candidate["tokens"] += fragment.size
                    windows += 1
                if (index + 1) % (32 if name in ("chat", "raw") else 8) == 0:
                    print(json.dumps({"cell": name, "sequences": index + 1,
                                      "budget": budget(started, device)}), flush=True)
        for hook in hooks:
            hook.remove()
        old_rows, candidate_rows = summarize(old), summarize(candidate)
        assert all(candidate_rows[li]["structural_selections"] -
                   old_rows[li]["structural_selections"] ==
                   4 * state["newly_shared_positions"] for li in range(24))
        counters[name] = candidate
        outputs[name] = {"sequences": len(sequences), "windows": windows,
                         "tokens": candidate["tokens"],
                         "newly_shared_input_positions": state["newly_shared_positions"],
                         "old_layers": old_rows, "candidate_layers": candidate_rows}
        print(json.dumps({"cell_complete": name, "tokens": candidate["tokens"],
                          "worst_hot_share": max(r["content_hot_parent_worst_grandchild_share"]
                                                 for r in candidate_rows),
                          "budget": budget(started, device)}), flush=True)

    raw_chat = add_counts(counters["raw"], counters["chat"])
    training = train["cells"]["training"]
    train_all = np.asarray([r["all_slot_counts"] for r in training["layers"]],
                           dtype=np.int64)
    train_content = np.asarray([r["content_slot_counts"]
                                for r in training["layers"]], dtype=np.int64)
    assert train_all.shape == train_content.shape == (24, 12800)
    assert np.all(train_all.sum(axis=1) == 4 * training["tokens"])
    train_counter = {"all": train_all, "content": train_content,
                     "tokens": training["tokens"]}
    pools = {"raw_chat": summarize(raw_chat),
             "train_plus_raw_chat": summarize(add_counts(train_counter, raw_chat))}
    gates = {name: check_cell(cell["candidate_layers"],
                              7000 if name in ("raw", "chat") else 4000,
                              structural_cap=0.35 if name == "chat" else 0.25)
             for name, cell in outputs.items()}
    for name, cell in outputs.items():
        gates[name]["new_structural_positions"] = (
            cell["newly_shared_input_positions"] <= 0.05 * cell["tokens"])
    gates["raw_chat"] = check_cell(pools["raw_chat"], 7000)
    gates["train_plus_raw_chat"] = check_cell(
        pools["train_plus_raw_chat"], 10000, support=True)
    gates["bf16_parity"] = True
    passed = all(all(items.values()) if isinstance(items, dict) else items
                 for items in gates.values())
    result = {"experiment": "METH-173-fresh-low-recurrence-route-validation",
              "manifest_sha256": MANIFEST_SHA, "teacher_sha256": args.teacher_sha,
              "meth172_result_sha256": M172_SHA,
              "baseline_table_sha256": BASE_TABLE_SHA,
              "candidate_table_sha256": TABLE_SHA,
              "source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "cells": outputs, "pools": pools, "gates": gates,
              "decision": "prospective_low_recurrence_route_pass_native_cpu_pending" if passed
                          else "prospective_low_recurrence_route_fail",
              "runtime": {**budget(started, device),
                          "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "summary": {name: {"coverage_min": min(r["content"]["coverage"]
                                                       for r in rows),
                                          "worst_hot_share": max(
                                              r["content_hot_parent_worst_grandchild_share"]
                                              for r in rows)}
                                  for name, rows in pools.items()},
                      "sha256": M136.digest(args.out), "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

