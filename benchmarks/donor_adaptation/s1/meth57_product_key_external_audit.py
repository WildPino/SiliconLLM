#!/usr/bin/env python3
"""Frozen external document, chat, task and pair-utility audit for METH-57."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth21_half_adapter_piqa as M21
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55


ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth57_product_key_external_manifest.json"
EXTERNAL_SHA = "9571f5d61c29e34b27c05ec531232a08b9eb6cd81bdf85aa5648ff74f1db3d75"
TRAINING_SHA = "06aa8934d849fc3bbe2ae2fb85c5ab4d302c9f3f1c5922138604bf1347273a39"
CHECKPOINT_SHA = "8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072"
MODEL_SHA = M44.MODEL_SHA
MAX_SECONDS = 30 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)
ROUTE_PERM_SEED = 575757
CATEGORIES = ("code", "prose", "technical_general")


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or peak > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-57 external budget: {elapsed:.1f}s, GPU {peak}, RSS {rss}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def document_summary(rows):
    result = {}
    for category in ("pooled", *CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        total_bytes = sum(r["bytes"] for r in group)
        donor = sum(r["donor_nats"] for r in group) / (math.log(2) * total_bytes)
        student = sum(r["student_nats"] for r in group) / (math.log(2) * total_bytes)
        result[category] = {"documents": len(group), "bytes": total_bytes,
                            "donor_bpb": donor, "student_bpb": student,
                            "delta_bpb": student - donor}
    return result


@torch.inference_mode()
def score_prompts(model, wrappers, items, device, start):
    for wrapper in wrappers:
        wrapper.route_counts.zero_()
        wrapper.oracle_checks = True
        wrapper.collect_load = True
    rows = []
    for item in items:
        ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                              device=device).unsqueeze(0)
        M44.set_experts(wrappers, False)
        donor = model(ids, use_cache=False).logits.argmax(dim=-1)
        M44.set_experts(wrappers, True)
        student = model(ids, use_cache=False).logits.argmax(dim=-1)
        rows.append({"source_id": item["source_id"],
                     "category": item["category"],
                     "matching": int((donor == student).sum()),
                     "positions": ids.numel()})
        budget(start, device)
    load = M55.load_summary(wrappers)
    for wrapper in wrappers:
        wrapper.oracle_checks = False
        wrapper.collect_load = False
    summary = {}
    for category in ("pooled", *CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        matching = sum(r["matching"] for r in group)
        positions = sum(r["positions"] for r in group)
        summary[category] = {"matching": matching, "positions": positions,
                             "agreement": matching / positions}
    return rows, summary, load


@torch.inference_mode()
def score_generation(model, wrappers, items, tokenizer, device, start):
    rows = []
    eos = model.config.eos_token_id
    model.config.use_cache = True
    for item in items:
        ids = item["prompt_ids"]
        inp = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        record = {"source_id": item["source_id"], "category": item["category"],
                  "prompt_ids_sha256": item["prompt_ids_sha256"]}
        for enabled, arm in ((False, "donor"), (True, "student")):
            M44.set_experts(wrappers, enabled)
            out = model.generate(input_ids=inp, max_new_tokens=128,
                                 do_sample=False, pad_token_id=eos,
                                 use_cache=True)
            continuation = out[0, len(ids):].cpu().tolist()
            record[arm] = {"continuation_ids": continuation,
                           "continuation_text": tokenizer.decode(
                               continuation, skip_special_tokens=False),
                           "eos_terminated": bool(continuation) and continuation[-1] == eos,
                           "early_non_eos_under16": len(continuation) < 16 and
                           (not continuation or continuation[-1] != eos),
                           "repeated_8gram_3x": M20.repeated_8gram(continuation),
                           "distinct2": M20.distinct2(continuation)}
            budget(start, device)
        rows.append(record)
        print(f"generation {len(rows)}/{len(items)}", flush=True)
    model.config.use_cache = False
    summary = {}
    for category in ("pooled", *CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        summary[category] = {}
        for arm in ("donor", "student"):
            summary[category][arm] = {
                "prompts": len(group),
                "eos_terminated": sum(r[arm]["eos_terminated"] for r in group),
                "early_non_eos_under16": sum(r[arm]["early_non_eos_under16"] for r in group),
                "repeated_8gram_3x": sum(r[arm]["repeated_8gram_3x"] for r in group),
                "mean_distinct2": sum(r[arm]["distinct2"] for r in group) / len(group)}
    return rows, summary


def score_task(model, wrappers, task_items, device, start):
    rows = []
    model.eval()
    with torch.inference_mode():
        for index, item in enumerate(task_items):
            record = {"index": index, "label": item["label"]}
            for enabled, arm in ((False, "donor"), (True, "student")):
                M44.set_experts(wrappers, enabled)
                scores = [M21.score_option(model, item["prefix"], suffix, device)
                          for suffix in item["suffixes"]]
                record[arm] = {"choice_mean": 0 if scores[0][1] <= scores[1][1] else 1,
                               "choice_total": 0 if scores[0][0] <= scores[1][0] else 1,
                               "options": [{"total_nll": total, "mean_nll": mean,
                                            "tokens": len(suffix)}
                                           for (total, mean), suffix in zip(scores, item["suffixes"])]}
            rows.append(record)
            budget(start, device)
            if (index + 1) % 128 == 0:
                print(f"PIQA {index+1}/{len(task_items)}", flush=True)
    donor_correct = np.asarray([int(r["donor"]["choice_mean"] == r["label"])
                                for r in rows], dtype=np.int8)
    student_correct = np.asarray([int(r["student"]["choice_mean"] == r["label"])
                                  for r in rows], dtype=np.int8)
    paired = student_correct - donor_correct
    rng = np.random.default_rng(M21.BOOTSTRAP_SEED)
    draws = [float(paired[rng.integers(0, len(rows), len(rows))].mean())
             for _ in range(M21.BOOTSTRAP_DRAWS)]
    summary = {"items": len(rows),
               "donor_correct": int(donor_correct.sum()),
               "donor_accuracy": float(donor_correct.mean()),
               "student_correct": int(student_correct.sum()),
               "student_accuracy": float(student_correct.mean()),
               "accuracy_delta": float(paired.mean()),
               "paired_bootstrap_lower95": float(np.quantile(draws, 0.05)),
               "bootstrap_seed": M21.BOOTSTRAP_SEED,
               "bootstrap_draws": M21.BOOTSTRAP_DRAWS,
               "donor_correct_student_wrong": int(((donor_correct == 1) &
                                                    (student_correct == 0)).sum()),
               "donor_wrong_student_correct": int(((donor_correct == 0) &
                                                    (student_correct == 1)).sum())}
    return rows, summary


def route_utility(model, wrappers, tokenizer, device, start):
    ids, byts, meta = M15.M13.C.get_slice(tokenizer, "heldout", 4, 256, 271828)
    assert meta["ids_sha256"] == M15.EVAL_IDS_SHA and int(byts.sum()) == 4588
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    trained_bpb = M15.bpb(model, ids, byts, wrappers, True)
    original = [(wrapper.a.detach().clone(), wrapper.b.detach().clone())
                for wrapper in wrappers]
    rng = np.random.default_rng(ROUTE_PERM_SEED)
    with torch.no_grad():
        for wrapper, (a, b) in zip(wrappers, original):
            permutation = torch.as_tensor(rng.permutation(M15.E),
                                          dtype=torch.long, device=device)
            wrapper.a.copy_(a[permutation])
            wrapper.b.copy_(b[permutation])
    permuted_bpb = M15.bpb(model, ids, byts, wrappers, True)
    with torch.no_grad():
        for wrapper, (a, b) in zip(wrappers, original):
            wrapper.a.copy_(a)
            wrapper.b.copy_(b)
    budget(start, device)
    return {"eval_ids_sha256": M15.EVAL_IDS_SHA,
            "permutation_seed": ROUTE_PERM_SEED,
            "trained_bpb": trained_bpb,
            "pair_permuted_bpb": permuted_bpb,
            "pair_permuted_minus_trained_bpb": permuted_bpb - trained_bpb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-result", type=Path, required=True)
    ap.add_argument("--training-result-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    partial_path = args.out.with_name(args.out.stem + ".partial.json")
    def save_partial(stage, **payload):
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps({"experiment": "METH-57", "completed_stage": stage,
                                            "training_result_sha256": TRAINING_SHA,
                                            "external_manifest_sha256": EXTERNAL_SHA,
                                            **payload}, indent=2) + "\n", encoding="utf-8")
    assert M17.sha(EXTERNAL.read_bytes()) == EXTERNAL_SHA
    assert args.training_result_sha256 == TRAINING_SHA
    assert M17.sha(args.training_result.read_bytes()) == TRAINING_SHA
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    assert training["decision"] == "eligible_for_frozen_external_evaluation"
    final_update = training["target_update"]
    assert final_update in (512, 1024)
    assert training["last_applied_update"] == final_update
    checkpoint = training["checkpoints"][str(final_update)]
    assert checkpoint["sha256"] == CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == checkpoint["sha256"]
    manifest = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 24
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        for field, digest in (("document_ids", "document_ids_sha256"),
                              ("prompt_ids", "prompt_ids_sha256")):
            assert M17.sha(np.asarray(item[field], dtype=np.int32).tobytes()) == item[digest]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = hf_hub_download(M42.MODEL, "model.safetensors",
                             revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source) == MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    _, task_items = M21.bind_data(tokenizer)
    assert len(task_items) == M21.ITEMS
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    wrappers = []
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    assert state["updates"] == final_update and state["source_sha256"] == MODEL_SHA
    assert state["teacher_sha256"] == M44.TEACHER_SHA
    for li, layer in enumerate(model.model.layers):
        wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(state["expert_state"][li][key].to(device))
        wrappers.append(wrapper)
    model.config.use_cache = False
    doc_rows = []
    for item in items:
        donor_nats = M17.score_doc(model, item["document_ids"], wrappers,
                                   False, device, start)
        student_nats = M17.score_doc(model, item["document_ids"], wrappers,
                                     True, device, start)
        doc_rows.append({"source_id": item["source_id"],
                         "category": item["category"], "bytes": item["bytes"],
                         "document_ids_sha256": item["document_ids_sha256"],
                         "donor_nats": donor_nats,
                         "student_nats": student_nats})
        budget(start, device)
        print(f"documents {len(doc_rows)}/{len(items)}", flush=True)
    documents = document_summary(doc_rows)
    save_partial("documents", document_rows=doc_rows, document_summary=documents)
    prompt_rows, prompts, route_load = score_prompts(model, wrappers, items, device, start)
    save_partial("prompts", document_rows=doc_rows, document_summary=documents,
                 prompt_rows=prompt_rows, prompt_summary=prompts,
                 route_load_by_layer=route_load)
    generation_rows, generations = score_generation(
        model, wrappers, items, tokenizer, device, start)
    save_partial("generations", document_rows=doc_rows, document_summary=documents,
                 prompt_rows=prompt_rows, prompt_summary=prompts,
                 generation_rows=generation_rows, generation_summary=generations,
                 route_load_by_layer=route_load)
    task_rows, task = score_task(model, wrappers, task_items, device, start)
    route = route_utility(model, wrappers, tokenizer, device, start)
    slots = [wrapper.changed_slots() for wrapper in wrappers]
    gates = {
        "documents": (documents["pooled"]["delta_bpb"] <= 0.02 and
                      all(documents[c]["delta_bpb"] <= 0.04 for c in CATEGORIES)),
        "prompt_top1": (prompts["pooled"]["agreement"] >= 0.95 and
                        all(prompts[c]["agreement"] >= 0.90 for c in CATEGORIES)),
        "generation": (
            generations["pooled"]["student"]["eos_terminated"] >=
            generations["pooled"]["donor"]["eos_terminated"] - 2 and
            all(generations[c]["student"]["repeated_8gram_3x"] <=
                generations[c]["donor"]["repeated_8gram_3x"] + 1 and
                generations[c]["student"]["early_non_eos_under16"] <=
                generations[c]["donor"]["early_non_eos_under16"] + 1
                for c in ("pooled", *CATEGORIES))),
        "task": (task["accuracy_delta"] >= -0.02 and
                 task["paired_bootstrap_lower95"] >= -0.05),
        "pair_utility": route["pair_permuted_minus_trained_bpb"] >= 0.002,
        "changed_slots": min(slots) >= 64}
    gates["joint"] = all(gates.values())
    runtime = budget(start, device)
    result = {"experiment": "METH-57-product-key-external-audit",
              "training_result_sha256": args.training_result_sha256,
              "checkpoint_sha256": checkpoint["sha256"],
              "model_sha256": MODEL_SHA,
              "external_manifest_sha256": EXTERNAL_SHA,
              "piqa_source_sha256": M21.PIQA_SHA,
              "piqa_labels_sha256": M21.LABELS_SHA,
              "document_rows": doc_rows, "document_summary": documents,
              "prompt_rows": prompt_rows, "prompt_summary": prompts,
              "route_load_by_layer": route_load,
              "generation_rows": generation_rows,
              "generation_summary": generations,
              "task_rows": task_rows, "task_summary": task,
              "pair_utility": route,
              "changed_slots_by_layer": slots,
              "gates": gates,
              "decision": "automated_gates_pass_blind_semantic_review_pending"
                          if gates["joint"] else "stop_instruct_parent_promotion",
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__,
                          "numpy": np.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps({"document_summary": documents,
                      "prompt_summary": prompts,
                      "generation_summary": generations,
                      "task_summary": task,
                      "pair_utility": route,
                      "min_changed_slots": min(slots),
                      "gates": gates,
                      "decision": result["decision"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
