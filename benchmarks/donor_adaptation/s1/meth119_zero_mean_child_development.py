#!/usr/bin/env python3
"""METH-119: new-source zero-mean child factors and E128 parity control."""

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
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth90_quant_aware_external_audit as M90
import meth95_hierarchical_e1280_parity as M95


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DIR / "meth118_zero_mean_child_dev_manifest.json"
MANIFEST_SHA = "1c89a4b0479d891e0441b152048c27947b12309c3d1062c9982b062c9e3fbae1"
TRAINING = DIR / "meth56_product_key_retention_result.json"
SPECIALIZED = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth107_long_chat_e1280.pt"
SPECIALIZED_SHA = "15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520"
TRAIN_REPORT = DIR / "meth107_long_chat_child_retention_result.json"
TRAIN_REPORT_SHA = "8864594df0981627ca97cd58717f1cf89eaaf90dddd235217a23416bb584df13"
ARMS = ("bf16_donor", "bf16_e128", "bf16_e1280")
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS_BYTES
            or result["gpu_peak_bytes"] > MAX_GPU_BYTES):
        raise RuntimeError(f"METH-119 resource stop: {result}")
    return result


def summarize(rows, prompts):
    documents = {}
    prompt_summary = {}
    for category in ("pooled", "code", "prose", "technical_general"):
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
                                    "e1280_minus_e128": (agreement["bf16_e1280"] -
                                                          agreement["bf16_e128"])}
    return documents, prompt_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for path, digest in ((MANIFEST, MANIFEST_SHA), (TRAINING, M57.TRAINING_SHA),
                         (SPECIALIZED, SPECIALIZED_SHA), (TRAIN_REPORT, TRAIN_REPORT_SHA)):
        assert M15.M13.sha256(path) == digest, path
    report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    assert report["checkpoint"]["sha256"] == SPECIALIZED_SHA
    assert report["distinct_slot_gate_pass"] is True
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 24 and manifest["selected_counts"] == {
        "code": 8, "prose": 8, "technical_general": 8}
    assert len({item["source_id"] for item in items}) == 24
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        assert M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item[
            "document_ids_sha256"]
        assert M17.sha(np.asarray(item["prompt_ids"], dtype=np.int32).tobytes()) == item[
            "prompt_ids_sha256"]
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
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(saved_parent["expert_state"][li][key].to(device))
            wrappers.append(wrapper)
    del saved_parent
    rows = [{"source_id": item["source_id"], "category": item["category"],
             "bytes": item["bytes"], "nats": {}} for item in items]
    prompts = [{"source_id": item["source_id"], "category": item["category"],
                "positions": len(item["prompt_ids"]), "matching": {}} for item in items]
    donor_top = {}
    parent_first_logits = None
    for arm, enabled in (("bf16_donor", False), ("bf16_e128", True)):
        for row, item in zip(rows, items):
            row["nats"][arm] = M17.score_doc(model, item["document_ids"],
                                               wrappers, enabled, device, start)
            budget(start, device)
        with torch.inference_mode():
            for row, item in zip(prompts, items):
                M44.set_experts(wrappers, enabled)
                ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
                logits = model(ids, use_cache=False).logits
                if arm == "bf16_e128" and row is prompts[0]:
                    parent_first_logits = logits.detach().cpu().clone()
                top = logits.argmax(-1)[0].cpu()
                if arm == "bf16_donor":
                    donor_top[item["source_id"]] = top
                row["matching"][arm] = int((top == donor_top[item["source_id"]]).sum())
                budget(start, device)
        print(json.dumps({"completed_arm": arm, "runtime": budget(start, device)}), flush=True)
    saved_child = torch.load(SPECIALIZED, map_location="cpu", weights_only=False)
    assert saved_child["updates"] == 256
    assert saved_child["source_sha256"] == M57.MODEL_SHA
    assert saved_child["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
    children = []
    distinct = []
    centered_mean_error = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M95.HierarchicalExperts(wrappers[li], li).to(device)
            saved = saved_child["expert_state"][li]
            for key in ("a", "b", "router", "child_projection", "child_keys"):
                assert getattr(wrapper, key).shape == saved[key].shape
                getattr(wrapper, key).copy_(saved[key].to(device))
            parent_b = wrappers[li].b.detach()
            raw_b = wrapper.b.view(M15.E, M95.CHILDREN, M15.M13.D, M15.R)
            child_mean = raw_b.mean(dim=1, keepdim=True)
            wrapper.b.copy_((parent_b[:, None] + raw_b - child_mean).reshape_as(wrapper.b))
            local_b = wrapper.b.view(M15.E, M95.CHILDREN, M15.M13.D, M15.R)
            centered_mean_error.append(float((local_b.mean(dim=1) - parent_b).abs().max()))
            delta = (local_b - local_b[:, :1]).abs().amax(dim=(-1, -2))
            distinct.append(int((delta > 1e-3).sum()))
            layer.mlp = wrapper
            children.append(wrapper)
    del saved_child
    for row, item in zip(rows, items):
        row["nats"]["bf16_e1280"] = M17.score_doc(model, item["document_ids"],
                                                     children, True, device, start)
        budget(start, device)
    with torch.inference_mode():
        for row, item in zip(prompts, items):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            top = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
            row["matching"]["bf16_e1280"] = int((top == donor_top[item["source_id"]]).sum())
            budget(start, device)
    generations = {}
    for arm, active in (("bf16_e128", wrappers), ("bf16_e1280", children)):
        for layer, wrapper in zip(model.model.layers, active):
            layer.mlp = wrapper
        generations[arm] = M90.score_generations(model, active, items, tok,
                                                   device, start, "student")
        budget(start, device)
    generation_rows = [{"source_id": item["source_id"], "category": item["category"],
                        "bf16_e128": generations["bf16_e128"][i],
                        "bf16_e1280": generations["bf16_e1280"][i]}
                       for i, item in enumerate(items)]
    eos = {arm: sum(r[arm]["eos_terminated"] for r in generation_rows)
           for arm in ("bf16_e128", "bf16_e1280")}
    generation_summary = M90.generation_summary([
        {"category": r["category"], "donor": r["bf16_e128"],
         "student": r["bf16_e1280"]} for r in generation_rows])
    no_child_nats = []
    with torch.no_grad():
        actual_b = [w.b.detach().clone() for w in children]
        for w, parent_w in zip(children, wrappers):
            w.b.copy_(parent_w.b.detach().repeat_interleave(M95.CHILDREN, dim=0))
        first_ids = torch.as_tensor(items[0]["prompt_ids"], dtype=torch.long, device=device)[None]
        no_child_first_logits = model(first_ids, use_cache=False).logits.detach().cpu()
        exact_parent_logits = torch.equal(no_child_first_logits, parent_first_logits)
        for row, item in zip(rows, items):
            no_child_nats.append(M17.score_doc(model, item["document_ids"],
                                                children, True, device, start))
            budget(start, device)
        for w, original in zip(children, actual_b):
            w.b.copy_(original)
        del actual_b
    no_child_bpb = sum(no_child_nats) / (math.log(2) * sum(r["bytes"] for r in rows))
    documents, prompt_summary = summarize(rows, prompts)
    no_child_bpb_error = abs(no_child_bpb - documents["pooled"]["bpb"]["bf16_e128"])
    gates = {
        "distinct_slots": min(distinct) >= 640,
        "centered_parent_mean": max(centered_mean_error) <= 1e-7,
        "no_child_parent_parity": exact_parent_logits and no_child_bpb_error <= 1e-8,
        "pooled_child_utility_bpb": documents["pooled"]["e1280_minus_e128"] <= -0.00005,
        "category_bpb_vs_e128": all(documents[c]["e1280_minus_e128"] <= 0.02
                                     for c in ("code", "prose", "technical_general")),
        "pooled_prompt_top1_vs_e128": prompt_summary["pooled"]["e1280_minus_e128"] >= -0.01,
        "category_prompt_top1_vs_e128": all(prompt_summary[c]["e1280_minus_e128"] >= -0.02
                                             for c in ("code", "prose", "technical_general")),
        "generation_eos_vs_e128": eos["bf16_e1280"] >= eos["bf16_e128"] - 2,
        "generation_loop_and_early": all(
            generation_summary[c]["student"]["repeated_8gram_3x"] <=
            generation_summary[c]["donor"]["repeated_8gram_3x"] + 1 and
            generation_summary[c]["student"]["early_non_eos_under16"] <=
            generation_summary[c]["donor"]["early_non_eos_under16"] + 1
            for c in ("pooled", "code", "prose", "technical_general")),
    }
    result = {"experiment": "METH-119-hierarchical-E1280-zero-mean-child-development",
              "source_sha256": M57.MODEL_SHA,
              "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
              "specialized_checkpoint_sha256": SPECIALIZED_SHA,
              "manifest_sha256": MANIFEST_SHA,
              "transform": "parent_B_plus_child_B_minus_per_parent_child_mean",
              "centered_parent_mean_max_abs_error_by_layer": centered_mean_error,
              "distinct_child_slots_by_layer": distinct,
              "no_child_pooled_bpb": no_child_bpb,
              "no_child_pooled_bpb_error": no_child_bpb_error,
              "no_child_first_prompt_exact_parent_logits": exact_parent_logits,
              "no_child_document_nats": no_child_nats,
              "document_rows": rows, "prompt_rows": prompts,
              "generation_rows": generation_rows, "generation_eos": eos,
              "generation_summary_e128_as_donor": generation_summary,
              "document_summary": documents, "prompt_summary": prompt_summary,
              "gates": gates,
              "decision": "eligible_for_blind_development_review" if all(gates.values())
              else "stop_zero_mean_child_development",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "pooled_documents": documents["pooled"],
                      "pooled_prompts": prompt_summary["pooled"],
                      "generation_eos": eos,
                      "no_child_pooled_bpb": no_child_bpb,
                      "no_child_first_prompt_exact_parent_logits": exact_parent_logits,
                      "centered_mean_max_abs_error": max(centered_mean_error),
                      "distinct_range": [min(distinct), max(distinct)],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
