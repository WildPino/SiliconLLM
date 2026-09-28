#!/usr/bin/env python3
"""METH-128: screen int8 head shortlist + exact BF16 row reranking."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
from safetensors import safe_open
import torch
from torch.nn import functional as F

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth95_hierarchical_e1280_parity as M95


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DIR / "meth121_zero_mean_child_external_manifest.json"
MANIFEST_SHA = "7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366"
TRAINING = DIR / "meth56_product_key_retention_result.json"
CHILD = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth107_long_chat_e1280.pt"
CHILD_SHA = "15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520"
HEADS = (
    ("row_r8", ROOT / "results/native_expert_scaling/meth59_qwen05b_instruct_r8_core.safetensors",
     "5b6ace4a027d127a6eb49ca99197edd4a1b810ebe681d8e3b6ab7edc088ca9bd"),
    ("group128_r8", ROOT / "results/native_expert_scaling/meth91_qwen05b_instruct_grouped_head_core.safetensors",
     "6f994c9ddf047adc789d3e1c841dd3be156cc578ed0d6c3f4045f97212e2accb"),
)
KS = (1, 4, 16, 64, 256)
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS or result[
            "peak_gpu_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-128 resource stop: {result}")
    return result


def load_centered(model, device, parent_path):
    saved_parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    assert saved_parent["updates"] == 512 and saved_parent["source_sha256"] == M57.MODEL_SHA
    parents = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(saved_parent["expert_state"][li][key].to(device))
            parents.append(wrapper)
    del saved_parent
    saved_child = torch.load(CHILD, map_location="cpu", weights_only=False)
    assert saved_child["updates"] == 256
    assert saved_child["source_sha256"] == M57.MODEL_SHA
    assert saved_child["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M95.HierarchicalExperts(parents[li], li).to(device)
            state = saved_child["expert_state"][li]
            for key in ("a", "b", "router", "child_projection", "child_keys"):
                getattr(wrapper, key).copy_(state[key].to(device))
            raw = wrapper.b.view(M15.E, M95.CHILDREN, M15.M13.D, M15.R)
            parent_b = parents[li].b.detach()
            wrapper.b.copy_((parent_b[:, None] + raw - raw.mean(dim=1, keepdim=True))
                            .reshape_as(wrapper.b))
            layer.mlp = wrapper
    del saved_child, parents


def reconstructed_head(path, kind, device, original):
    with safe_open(str(path), framework="pt", device="cpu") as file:
        meta = file.metadata()
        assert meta["model_sha256"] == M57.MODEL_SHA
        assert meta["product_key_checkpoint_sha256"] == M57.CHECKPOINT_SHA
        codes = file.get_tensor("model.embed_tokens.weight.q").to(device)
        scales = file.get_tensor("model.embed_tokens.weight.scale").to(device)
    assert codes.shape == original.shape == (151936, 896)
    assert codes.dtype == torch.int8
    if kind == "row_r8":
        assert scales.shape == (151936,) and scales.dtype == torch.float32
        head = (codes.float() * scales[:, None]).to(torch.bfloat16)
    else:
        assert kind == "group128_r8" and scales.shape == (151936, 7)
        assert scales.dtype == torch.float16
        head = (codes.reshape(151936, 7, 128).float() * scales.float()[:, :, None])
        head = head.reshape(151936, 896).to(torch.bfloat16)
    assert bool(torch.isfinite(head).all())
    return head, codes.numel() + scales.numel() * scales.element_size()


def score_item(hidden, exact_head, approx_head):
    exact = F.linear(hidden, exact_head)
    approx = F.linear(hidden, approx_head)
    assert bool(torch.isfinite(exact).all()) and bool(torch.isfinite(approx).all())
    target = exact.argmax(-1)
    candidate = approx.topk(max(KS), dim=-1).indices
    selected_head = exact_head[candidate]
    rescored = torch.bmm(selected_head, hidden.unsqueeze(-1)).squeeze(-1)
    exact_selected = exact.gather(1, candidate)
    row_score_max_abs = float((rescored.float() - exact_selected.float()).abs().max())
    result = {"positions": int(hidden.shape[0]), "misses": {},
              "rerank_mismatches": {}, "gather_mismatches": {},
              "row_score_max_abs": row_score_max_abs,
              "approx_top1_matches": int((candidate[:, 0] == target).sum())}
    for k in KS:
        ids = candidate[:, :k]
        included = (ids == target[:, None]).any(dim=-1)
        def choose(scores):
            maximum = scores.max(dim=-1, keepdim=True).values
            return torch.where(scores == maximum, ids,
                               torch.full_like(ids, 151936)).min(dim=-1).values
        chosen = choose(rescored[:, :k])
        gathered = choose(exact_selected[:, :k])
        result["misses"][str(k)] = int((~included).sum())
        result["rerank_mismatches"][str(k)] = int((chosen != target).sum())
        result["gather_mismatches"][str(k)] = int((gathered != target).sum())
    assert int((exact.argmax(-1) != target).sum()) == 0  # full-vocab control
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for path, digest in ((MANIFEST, MANIFEST_SHA), (CHILD, CHILD_SHA)):
        assert M15.M13.sha256(path) == digest, path
    assert M15.M13.sha256(TRAINING) == M57.TRAINING_SHA
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    parent = training["checkpoints"]["512"]
    assert parent["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(parent["path"]) == M57.CHECKPOINT_SHA
    for _, path, digest in HEADS:
        assert M15.M13.sha256(path) == digest, path
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 24 and manifest["selected_counts"] == {
        "code": 8, "prose": 8, "technical_general": 8}
    for item in items:
        assert hashlib.sha256(np.asarray(item["prompt_ids"], dtype="<i4").tobytes()).hexdigest() == item[
            "prompt_ids_sha256"]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.config.use_cache = False
    load_centered(model, device, parent["path"])
    exact_head = model.lm_head.weight.detach()
    assert exact_head.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    hidden_rows = []
    with torch.inference_mode():
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            hidden = model.model(ids, use_cache=False).last_hidden_state[0]
            assert hidden.dtype == torch.bfloat16 and bool(torch.isfinite(hidden).all())
            hidden_rows.append(hidden.detach().clone())
            budget(start, device)
        first_ids = torch.as_tensor(items[0]["prompt_ids"], dtype=torch.long,
                                    device=device)[None]
        full = model(first_ids, use_cache=False).logits[0]
        direct = F.linear(hidden_rows[0], exact_head)
        assert torch.equal(full, direct), "exact head baseline mismatch"
        results = {}
        for kind, path, digest in HEADS:
            approx_head, payload_bytes = reconstructed_head(path, kind, device, exact_head)
            rows = []
            for item, hidden in zip(items, hidden_rows):
                scored = score_item(hidden, exact_head, approx_head)
                rows.append({"source_id": item["source_id"],
                             "category": item["category"], **scored})
                budget(start, device)
            positions = sum(row["positions"] for row in rows)
            results[kind] = {"artifact_sha256": digest, "stored_head_payload_bytes": payload_bytes,
                             "resident_with_exact_bf16_head_bytes": payload_bytes + exact_head.numel() * 2,
                             "selected_bf16_row_bytes_at_k64": 64 * 896 * 2,
                             "positions": positions,
                             "misses": {str(k): sum(row["misses"][str(k)] for row in rows) for k in KS},
                             "rerank_mismatches": {
                                 str(k): sum(row["rerank_mismatches"][str(k)] for row in rows)
                                 for k in KS},
                             "gather_mismatches": {
                                 str(k): sum(row["gather_mismatches"][str(k)] for row in rows)
                                 for k in KS},
                             "row_score_max_abs": max(row["row_score_max_abs"] for row in rows),
                             "categories": {
                                 category: {
                                     "positions": sum(row["positions"] for row in rows
                                                      if row["category"] == category),
                                     "misses_k64": sum(row["misses"]["64"] for row in rows
                                                       if row["category"] == category),
                                     "rerank_mismatches_k64": sum(
                                         row["rerank_mismatches"]["64"] for row in rows
                                         if row["category"] == category),
                                 } for category in ("code", "prose", "technical_general")},
                             "rows": rows}
            del approx_head
            print(json.dumps({"arm": kind, "positions": positions,
                              "misses": results[kind]["misses"],
                              "rerank_mismatches": results[kind]["rerank_mismatches"],
                              "runtime": budget(start, device)}), flush=True)
    winner = ("row_r8" if results["row_r8"]["misses"]["64"] == 0 and
              results["row_r8"]["rerank_mismatches"]["64"] == 0 else
              "group128_r8" if results["group128_r8"]["misses"]["64"] == 0 and
              results["group128_r8"]["rerank_mismatches"]["64"] == 0 else "none")
    result = {"experiment": "METH-128-exact-BF16-head-rerank-development-screen",
              "model_source_sha256": M57.MODEL_SHA,
              "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
              "child_checkpoint_sha256": CHILD_SHA,
              "manifest_sha256": MANIFEST_SHA,
              "checkpoint_transform": "parent_B_plus_child_B_minus_mean_children",
              "exact_head_payload_bytes": exact_head.numel() * 2,
              "k_values": KS, "arms": results,
              "native_implementation_eligible_format": winner,
              "viewed_sources_only": True,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner, "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
