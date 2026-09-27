#!/usr/bin/env python3
"""METH-87: posthoc factor-amplitude diagnostic on viewed METH-86 sources."""

import argparse
import json
import math
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
from safetensors import safe_open
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth85_export_group64_r8_core as M85


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DIR / "meth86_group64_r8_dev_manifest.json"
MANIFEST_SHA = "0d4ac34beea84ecbcaf92bac088345ccab5cc3a2e3f7898710018248bf4bca0c"
CORE = ROOT / "results/native_expert_scaling/meth85_qwen05b_instruct_group64_r8_core.safetensors"
CORE_SHA = "c484a1130e495342d1b8644ff156cc002d0fd68fe12230604e59f35d3af6433a"
TRAINING = DIR / "meth56_product_key_retention_result.json"
ARMS = ("bf16_donor", "bf16_e128", "group64_donor", "group64_e128")
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    report = {"elapsed_seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (report["elapsed_seconds"] > MAX_SECONDS or report["rss_bytes"] > MAX_RSS_BYTES
            or report["gpu_peak_allocated_bytes"] > MAX_GPU_BYTES):
        raise RuntimeError(f"METH-86 development budget exceeded: {report}")
    return report


def write_progress(path, stage, rows, prompts, start, device):
    path.write_text(json.dumps({"experiment": "METH-86-group64-R8-core-composition",
                                "completed_stage": stage,
                                "manifest_sha256": MANIFEST_SHA,
                                "core_sha256": CORE_SHA,
                                "checkpoint_sha256": M57.CHECKPOINT_SHA,
                                "document_rows": rows, "prompt_rows": prompts,
                                "runtime": budget(start, device)}, indent=2) + "\n",
                    encoding="utf-8")


def summaries(rows, prompt_rows):
    documents = {}
    prompts = {}
    for category in ("pooled", "code", "prose", "technical_general"):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        pgroup = (prompt_rows if category == "pooled" else
                  [r for r in prompt_rows if r["category"] == category])
        assert len(group) == len(pgroup) == (24 if category == "pooled" else 8)
        denom = math.log(2) * sum(r["bytes"] for r in group)
        bpbs = {arm: sum(r["nats"][arm] for r in group) / denom for arm in ARMS}
        positions = sum(r["positions"] for r in pgroup)
        matches = {arm: sum(r["matching"][arm] for r in pgroup) for arm in ARMS}
        documents[category] = {"documents": len(group), "bytes": sum(r["bytes"] for r in group),
                               "bpb": bpbs,
                               "group64_e128_delta_vs_bf16_donor": (bpbs["group64_e128"]
                                                                - bpbs["bf16_donor"])}
        prompts[category] = {"prompts": len(pgroup), "positions": positions,
                             "matching": matches,
                             "agreement": {arm: matches[arm] / positions for arm in ARMS}}
    return documents, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--factor-scale", required=True, type=float)
    parser.add_argument("--bf16-head", action="store_true")
    args = parser.parse_args()
    assert 0.0 <= args.factor_scale <= 1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    assert M15.M13.sha256(MANIFEST) == MANIFEST_SHA
    assert M15.M13.sha256(CORE) == CORE_SHA
    assert M15.M13.sha256(TRAINING) == M57.TRAINING_SHA
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 24
    assert manifest["selected_counts"] == {"code": 8, "prose": 8,
                                            "technical_general": 8}
    assert len({item["source_id"] for item in items}) == 24
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        assert M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item[
            "document_ids_sha256"]
        assert M17.sha(np.asarray(item["prompt_ids"], dtype=np.int32).tobytes()) == item[
            "prompt_ids_sha256"]
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    checkpoint = training["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA

    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
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
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()

    for param in model.parameters():
        param.requires_grad_(False)
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M57.MODEL_SHA
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(state["expert_state"][li][key].to(device))
            wrappers.append(wrapper)
    del state
    model.config.use_cache = False
    budget(start, device)

    document_rows = [{"source_id": item["source_id"], "category": item["category"],
                      "bytes": item["bytes"],
                      "document_ids_sha256": item["document_ids_sha256"], "nats": {}}
                     for item in items]
    prompt_rows = [{"source_id": item["source_id"], "category": item["category"],
                    "prompt_ids_sha256": item["prompt_ids_sha256"],
                    "positions": len(item["prompt_ids"]), "matching": {}}
                   for item in items]
    donor_top = {}
    for arm, enabled in (("bf16_donor", False), ("bf16_e128", True)):
        for row, item in zip(document_rows, items):
            row["nats"][arm] = M17.score_doc(model, item["document_ids"],
                                               wrappers, enabled, device, start)
            budget(start, device)
        with torch.inference_mode():
            for row, item in zip(prompt_rows, items):
                M44.set_experts(wrappers, enabled)
                ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                      device=device).unsqueeze(0)
                top = model(ids, use_cache=False).logits.argmax(dim=-1)[0].cpu()
                if arm == "bf16_donor":
                    donor_top[item["source_id"]] = top
                row["matching"][arm] = int((top == donor_top[item["source_id"]]).sum())
                budget(start, device)
        write_progress(partial, arm, document_rows, prompt_rows, start, device)
        print(json.dumps({"stage": arm, "documents": len(document_rows),
                          "prompts": len(prompt_rows)}), flush=True)

    with torch.no_grad():
        for wrapper in wrappers:
            wrapper.b.mul_(args.factor_scale)
    with safe_open(str(CORE), framework="pt", device="cpu") as archive:
        metadata = archive.metadata()
        assert metadata["format"] == "QWEN25_INSTRUCT_R8H_GROUP64_R8FFN_BF16ATTN_V1"
        assert metadata["model_sha256"] == M57.MODEL_SHA
        assert metadata["product_key_checkpoint_sha256"] == M57.CHECKPOINT_SHA
        assert len(archive.keys()) == 363
        counts = {"tied_head": 0, "ffn": 0, "attention": 0, "control": 0}
        with torch.no_grad():
            for name, param in original_params.items():
                organ = M24.classify(name, tuple(param.shape))
                counts[organ] += 1
                if organ in ("ffn", "tied_head"):
                    packed = archive.get_tensor(name + ".q").to(device)
                    scale = archive.get_tensor(name + ".scale").to(device)
                    if organ == "ffn":
                        assert packed.dtype == torch.int8 and scale.dtype == torch.float16
                        reconstructed = M85.reconstruct_ffn(packed, scale)
                    else:
                        assert packed.dtype == torch.int8 and scale.dtype == torch.float32
                        reconstructed = (packed.float() * scale[:, None]).bfloat16()
                    assert tuple(reconstructed.shape) == tuple(param.shape)
                    param.copy_(reconstructed)
                else:
                    stored = archive.get_tensor(name)
                    assert stored.dtype == (torch.float32 if organ == "control"
                                            else torch.bfloat16)
                    assert torch.equal(stored.to(param.dtype), param.detach().cpu()), name
                budget(start, device)
        assert counts == {"tied_head": 1, "ffn": 72, "attention": 96,
                          "control": 121}
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()

    if args.bf16_head:
        with safe_open(str(source), framework="pt", device="cpu") as source_archive:
            head = source_archive.get_tensor("model.embed_tokens.weight")
        with torch.no_grad():
            original_params["model.embed_tokens.weight"].copy_(head.to(device))

    for arm, enabled in (("group64_donor", False), ("group64_e128", True)):
        for row, item in zip(document_rows, items):
            row["nats"][arm] = M17.score_doc(model, item["document_ids"],
                                               wrappers, enabled, device, start)
            budget(start, device)
        with torch.inference_mode():
            for row, item in zip(prompt_rows, items):
                M44.set_experts(wrappers, enabled)
                ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                      device=device).unsqueeze(0)
                top = model(ids, use_cache=False).logits.argmax(dim=-1)[0].cpu()
                row["matching"][arm] = int((top == donor_top[item["source_id"]]).sum())
                budget(start, device)
        write_progress(partial, arm, document_rows, prompt_rows, start, device)
        print(json.dumps({"stage": arm, "documents": len(document_rows),
                          "prompts": len(prompt_rows)}), flush=True)
    documents, prompts = summaries(document_rows, prompt_rows)
    gates = {
        "pooled_document": documents["pooled"]["group64_e128_delta_vs_bf16_donor"] <= 0.02,
        "each_category_document": all(
            documents[c]["group64_e128_delta_vs_bf16_donor"] <= 0.04
            for c in ("code", "prose", "technical_general")),
        "pooled_prompt_top1": prompts["pooled"]["agreement"]["group64_e128"] >= 0.95,
        "each_category_prompt_top1": all(
            prompts[c]["agreement"]["group64_e128"] >= 0.90
            for c in ("code", "prose", "technical_general")),
        "bf16_student_noninferiority": (
            prompts["pooled"]["agreement"]["group64_e128"] >=
            prompts["pooled"]["agreement"]["bf16_e128"] - 0.01),
    }
    result = {
        "experiment": "METH-87-group64-R8-factor-scale-viewed-diagnostic",
        "factor_scale": args.factor_scale,
        "bf16_head_diagnostic": args.bf16_head,
        "model_sha256": M57.MODEL_SHA,
        "checkpoint_sha256": M57.CHECKPOINT_SHA,
        "core_sha256": CORE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "document_rows": document_rows, "prompt_rows": prompt_rows,
        "document_summary": documents, "prompt_summary": prompts,
        "gates": gates,
        "decision": "posthoc_diagnostic_only_requires_new_source_validation",
        "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                    "cuda_index": matches[0], "torch": torch.__version__},
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "pooled_documents": documents["pooled"],
                      "pooled_prompts": prompts["pooled"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
