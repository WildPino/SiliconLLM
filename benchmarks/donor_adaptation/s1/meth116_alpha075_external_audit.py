#!/usr/bin/env python3
"""METH-116: alpha=0.75 E1280 external quality and child-choice audit."""

import argparse
import json
import math
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth21_half_adapter_piqa as M21
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth90_quant_aware_external_audit as M90
import meth95_hierarchical_e1280_parity as M95


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DIR / "meth115_alpha075_external_manifest.json"
MANIFEST_SHA = "965193d46bce1e9bd4ff8ce08b3339e8b19785bdcd3fa5cdf8c929004617fc05"
TRAINING = DIR / "meth56_product_key_retention_result.json"
SPECIALIZED = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth107_long_chat_e1280.pt"
SPECIALIZED_SHA = "15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520"
TRAIN_REPORT = DIR / "meth107_long_chat_child_retention_result.json"
TRAIN_REPORT_SHA = "8864594df0981627ca97cd58717f1cf89eaaf90dddd235217a23416bb584df13"
CATEGORIES = ("code", "prose", "technical_general")
ARMS = ("bf16_donor", "bf16_e128", "bf16_e1280")
ALPHA = 0.75
MAX_SECONDS = 30 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS_BYTES
            or result["gpu_peak_bytes"] > MAX_GPU_BYTES):
        raise RuntimeError(f"METH-116 resource stop: {result}")
    return result


def summarize(rows, prompts):
    documents = {}
    prompt_summary = {}
    for category in ("pooled", *CATEGORIES):
        docs = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        prows = prompts if category == "pooled" else [r for r in prompts if r["category"] == category]
        assert len(docs) == len(prows) == (24 if category == "pooled" else 8)
        bpb = {arm: sum(r["nats"][arm] for r in docs) /
               (math.log(2) * sum(r["bytes"] for r in docs)) for arm in ARMS}
        positions = sum(r["positions"] for r in prows)
        agreement = {arm: sum(r["matching"][arm] for r in prows) / positions
                     for arm in ARMS}
        documents[category] = {"documents": len(docs), "bytes": sum(r["bytes"] for r in docs),
                               "bpb": bpb, "e1280_minus_e128": bpb["bf16_e1280"] - bpb["bf16_e128"]}
        prompt_summary[category] = {"prompts": len(prows), "positions": positions,
                                    "agreement": agreement,
                                    "e1280_minus_e128": agreement["bf16_e1280"] - agreement["bf16_e128"]}
    return documents, prompt_summary


def write_result(path, stage, payload, start, device):
    payload.update({"experiment": "METH-116-alpha075-hierarchical-E1280-external-audit",
                    "completed_stage": stage, "source_sha256": M57.MODEL_SHA,
                    "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
                    "specialized_checkpoint_sha256": SPECIALIZED_SHA,
                    "manifest_sha256": MANIFEST_SHA,
                    "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                                "torch": torch.__version__}})
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    for path, digest in ((MANIFEST, MANIFEST_SHA), (TRAINING, M57.TRAINING_SHA),
                         (SPECIALIZED, SPECIALIZED_SHA), (TRAIN_REPORT, TRAIN_REPORT_SHA)):
        assert M15.M13.sha256(path) == digest, path
    train_report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    assert train_report["checkpoint"]["sha256"] == SPECIALIZED_SHA
    assert train_report["distinct_slot_gate_pass"] is True
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 24 and manifest["selected_counts"] == {
        "code": 8, "prose": 8, "technical_general": 8}
    assert len({item["source_id"] for item in items}) == 24
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        for field in ("document_ids", "prompt_ids"):
            assert M17.sha(np.asarray(item[field], dtype=np.int32).tobytes()) == item[field + "_sha256"]
    parent = json.loads(TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert parent["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(parent["path"]) == M57.CHECKPOINT_SHA
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    _, task_items = M21.bind_data(tok)
    assert len(task_items) == 1838
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
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
    model.config.use_cache = False
    saved_parent = torch.load(parent["path"], map_location="cpu", weights_only=False)
    assert saved_parent["updates"] == 512 and saved_parent["source_sha256"] == M57.MODEL_SHA
    parents = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(saved_parent["expert_state"][li][key].to(device))
            parents.append(wrapper)
    del saved_parent
    saved_child = torch.load(SPECIALIZED, map_location="cpu", weights_only=False)
    assert saved_child["updates"] == 256 and saved_child["source_sha256"] == M57.MODEL_SHA
    assert saved_child["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
    children = []
    distinct = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M95.HierarchicalExperts(parents[li], li).to(device)
            saved = saved_child["expert_state"][li]
            for key in ("a", "b", "router", "child_projection", "child_keys"):
                assert getattr(wrapper, key).shape == saved[key].shape
                getattr(wrapper, key).copy_(saved[key].to(device))
            parent_b = parents[li].b.detach().repeat_interleave(M95.CHILDREN, dim=0)
            wrapper.b.copy_(parent_b + ALPHA * (wrapper.b - parent_b))
            local_b = wrapper.b.view(M15.E, M95.CHILDREN, M15.M13.D, M15.R)
            delta = (local_b - local_b[:, :1]).abs().amax(dim=(-1, -2))
            distinct.append(int((delta > 1e-3).sum()))
            children.append(wrapper)
    del saved_child

    def activate(wrappers):
        for layer, wrapper in zip(model.model.layers, wrappers):
            layer.mlp = wrapper

    rows = [{"source_id": item["source_id"], "category": item["category"],
             "bytes": item["bytes"], "nats": {}} for item in items]
    prompts = [{"source_id": item["source_id"], "category": item["category"],
                "positions": len(item["prompt_ids"]), "matching": {}} for item in items]
    donor_top = {}
    for arm, wrappers, enabled in (("bf16_donor", parents, False),
                                   ("bf16_e128", parents, True),
                                   ("bf16_e1280", children, True)):
        activate(wrappers)
        for row, item in zip(rows, items):
            row["nats"][arm] = M17.score_doc(model, item["document_ids"],
                                               wrappers, enabled, device, start)
            budget(start, device)
        with torch.inference_mode():
            M44.set_experts(wrappers, enabled)
            for row, item in zip(prompts, items):
                ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
                top = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
                if arm == "bf16_donor":
                    donor_top[item["source_id"]] = top
                row["matching"][arm] = int((top == donor_top[item["source_id"]]).sum())
                budget(start, device)
        print(json.dumps({"completed_arm": arm, "runtime": budget(start, device)}), flush=True)
    docs, prompt_summary = summarize(rows, prompts)
    no_child_nats = []
    with torch.no_grad():
        activate(children)
        actual_b = [w.b.detach().clone() for w in children]
        for w in children:
            mean_b = w.b.view(M15.E, M95.CHILDREN, M15.M13.D, M15.R).mean(dim=1)
            w.b.copy_(mean_b.repeat_interleave(M95.CHILDREN, dim=0))
        for item in items:
            no_child_nats.append(M17.score_doc(model, item["document_ids"],
                                                children, True, device, start))
            budget(start, device)
        for w, original in zip(children, actual_b):
            w.b.copy_(original)
        del actual_b
    no_child_bpb = sum(no_child_nats) / (math.log(2) * sum(r["bytes"] for r in rows))
    gates = {
        "distinct_slots": min(distinct) >= 640,
        "pooled_bpb_vs_e128": docs["pooled"]["e1280_minus_e128"] <= 0.01,
        "category_bpb_vs_e128": all(docs[c]["e1280_minus_e128"] <= 0.02 for c in CATEGORIES),
        "pooled_prompt_top1_vs_e128": prompt_summary["pooled"]["e1280_minus_e128"] >= -0.01,
        "category_prompt_top1_vs_e128": all(prompt_summary[c]["e1280_minus_e128"] >= -0.02
                                             for c in CATEGORIES),
        "selected_child_bpb_vs_no_child": (
            docs["pooled"]["bpb"]["bf16_e1280"] <= no_child_bpb - 0.00005),
    }
    payload = {"document_rows": rows, "prompt_rows": prompts,
               "document_summary": docs, "prompt_summary": prompt_summary,
               "alpha": ALPHA, "distinct_child_slots_by_layer": distinct,
               "no_child_document_nats": no_child_nats,
               "no_child_pooled_bpb": no_child_bpb, "gates": gates}
    write_result(partial, "documents_and_prompts", payload, start, device)
    if not all(gates.values()):
        payload["decision"] = "stop_external_document_prompt_gate"
        write_result(args.out, "documents_and_prompts", payload, start, device)
        print(json.dumps({"decision": payload["decision"], "gates": gates,
                          "pooled_documents": docs["pooled"],
                          "pooled_prompts": prompt_summary["pooled"],
                          "no_child_pooled_bpb": no_child_bpb}, indent=2), flush=True)
        return

    generations = {}
    for arm, wrappers, label in (("bf16_donor", parents, "donor"),
                                 ("bf16_e128", parents, "student"),
                                 ("bf16_e1280", children, "student")):
        activate(wrappers)
        generations[arm] = M90.score_generations(model, wrappers, items, tok, device, start, label)
        budget(start, device)
    generation_rows = []
    for i, item in enumerate(items):
        generation_rows.append({"source_id": item["source_id"], "category": item["category"],
                                "prompt_ids_sha256": item["prompt_ids_sha256"],
                                **{arm: generations[arm][i] for arm in ARMS}})
    comparison_rows = [{"category": r["category"],
                        "donor": r["bf16_e128"], "student": r["bf16_e1280"]}
                       for r in generation_rows]
    generation_summary = M90.generation_summary(comparison_rows)
    generation_gate = (
        generation_summary["pooled"]["student"]["eos_terminated"] >=
        generation_summary["pooled"]["donor"]["eos_terminated"] - 2 and
        all(generation_summary[c]["student"]["repeated_8gram_3x"] <=
            generation_summary[c]["donor"]["repeated_8gram_3x"] + 1 and
            generation_summary[c]["student"]["early_non_eos_under16"] <=
            generation_summary[c]["donor"]["early_non_eos_under16"] + 1
            for c in ("pooled", *CATEGORIES)))
    gates["generation"] = generation_gate
    payload.update({"generation_rows": generation_rows,
                    "generation_summary_e128_as_donor": generation_summary})
    write_result(partial, "generation", payload, start, device)
    if not generation_gate:
        payload["decision"] = "stop_external_generation_gate"
        write_result(args.out, "generation", payload, start, device)
        print(json.dumps({"decision": payload["decision"], "gates": gates,
                          "generation_summary": generation_summary}, indent=2), flush=True)
        return

    task_arms = {}
    for arm, wrappers, label in (("bf16_donor", parents, "donor"),
                                 ("bf16_e128", parents, "student"),
                                 ("bf16_e1280", children, "student")):
        activate(wrappers)
        task_arms[arm] = M90.score_task(model, wrappers, task_items, device, start, label)
    task_rows = []
    for i in range(len(task_items)):
        task_rows.append({"index": i, "label": task_arms["bf16_donor"][i]["label"],
                          **{arm: task_arms[arm][i] for arm in ARMS}})
    task_compare = [{"label": r["label"], "donor": r["bf16_e128"],
                     "student": r["bf16_e1280"]} for r in task_rows]
    task_summary = M90.task_summary(task_compare)
    gates["task"] = (task_summary["accuracy_delta"] >= -0.02 and
                     task_summary["paired_bootstrap_lower95"] >= -0.05)
    gates["joint_automatic"] = all(gates.values())
    payload.update({"task_rows": task_rows,
                    "task_summary_e128_as_donor": task_summary,
                    "piqa_source_sha256": M21.PIQA_SHA,
                    "piqa_labels_sha256": M21.LABELS_SHA,
                    "decision": "automatic_pass_blind_semantic_review_pending"
                    if gates["joint_automatic"] else "stop_external_task_gate"})
    write_result(args.out, "all_automatic", payload, start, device)
    partial.unlink(missing_ok=True)
    print(json.dumps({"decision": payload["decision"], "gates": gates,
                      "pooled_documents": docs["pooled"],
                      "pooled_prompts": prompt_summary["pooled"],
                      "no_child_pooled_bpb": no_child_bpb,
                      "generation_summary": generation_summary,
                      "task_summary": task_summary,
                      "runtime": payload["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
