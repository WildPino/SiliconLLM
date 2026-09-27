#!/usr/bin/env python3
"""METH-95: function-preserving 10x expansion of learned E128 experts."""

import argparse
import json
import math
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CHECKPOINT_REPORT = DIR / "meth56_product_key_retention_result.json"
CHAT_MANIFEST = DIR / "meth90_quant_aware_external_manifest.json"
CHAT_SHA = "70d48b30de55c41283374223ba2b61edaacad396e6f3410c45c166bbaa885937"
CHILDREN = 10
CHILD_RANK = 32
SEED = 95095
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS_BYTES
            or result["gpu_peak_bytes"] > MAX_GPU_BYTES):
        raise RuntimeError(f"METH-95 resource stop: {result}")
    return result


class HierarchicalExperts(M55.ProductKeyExperts):
    def __init__(self, parent, layer_id):
        super().__init__(parent.base, layer_id)
        width = M15.M13.D
        with torch.no_grad():
            self.router.copy_(parent.router.detach().cpu())
        self.a = nn.Parameter(parent.a.detach().cpu().repeat_interleave(CHILDREN, dim=0),
                              requires_grad=False)
        self.b = nn.Parameter(parent.b.detach().cpu().repeat_interleave(CHILDREN, dim=0),
                              requires_grad=False)
        gen = torch.Generator(device="cpu").manual_seed(SEED + layer_id)
        projection = torch.randn((CHILD_RANK, width), generator=gen) * (
            0.1 / math.sqrt(width))
        keys = torch.randn((M15.E, CHILDREN, CHILD_RANK), generator=gen) * (
            0.1 / math.sqrt(CHILD_RANK))
        self.register_buffer("child_projection", projection)
        self.register_buffer("child_keys", keys)
        self.visited = torch.zeros(M15.E * CHILDREN, dtype=torch.int64)
        self.capture = False
        self.parent_routes = []
        self.parent_scores = []
        self.child_ids = []

    def child_route(self, flat, parent_ids):
        q = F.linear(flat.float(), self.child_projection)
        keys = self.child_keys[parent_ids]
        child_scores = torch.einsum("nr,nkcr->nkc", q, keys)
        local = child_scores.argmax(dim=-1)
        ids = parent_ids * CHILDREN + local
        return ids

    def forward(self, x):
        dense = self.base(x)
        if not self.enabled:
            return dense
        flat = x.reshape(-1, M15.M13.D)
        parents, selected_scores = self.routes(flat)
        child = self.child_route(flat, parents)
        gate = F.softmax(selected_scores, dim=-1).to(flat.dtype)
        a = self.a[child].to(flat.dtype)
        b = self.b[child].to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b)
        residual = (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)
        if self.capture:
            self.parent_routes.append(parents.detach().cpu())
            self.parent_scores.append(selected_scores.detach().cpu())
            self.child_ids.append(child.detach().cpu())
            self.visited += torch.bincount(child.detach().flatten().cpu(),
                                           minlength=M15.E * CHILDREN)
        return dense + residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert M15.M13.sha256(CHECKPOINT_REPORT) == M57.TRAINING_SHA
    assert M15.M13.sha256(CHAT_MANIFEST) == CHAT_SHA
    training = json.loads(CHECKPOINT_REPORT.read_text(encoding="utf-8"))
    parent = training["checkpoints"]["512"]
    assert parent["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(parent["path"]) == M57.CHECKPOINT_SHA
    state = torch.load(parent["path"], map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M57.MODEL_SHA
    assert len(state["expert_state"]) == M15.M13.L
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    raw_ids, _, meta = M15.M13.C.get_slice(tok, "heldout", 4, 256, 271828)
    assert meta["ids_sha256"] == M15.EVAL_IDS_SHA
    chat_manifest = json.loads(CHAT_MANIFEST.read_text(encoding="utf-8"))
    chat_ids = chat_manifest["items"][0]["prompt_ids"]
    inputs = {"raw": torch.as_tensor(raw_ids[0, :128], dtype=torch.long,
                                      device=device)[None],
              "chat": torch.as_tensor(chat_ids, dtype=torch.long, device=device)[None]}
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.config.use_cache = False
    parent_wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(state["expert_state"][li][key].to(device))
            parent_wrappers.append(wrapper)
    del state
    parent_logits = {}
    parent_routes = {}
    parent_scores = {}
    def make_parent_hook(li):
        def hook(module, args):
            flat = args[0].reshape(-1, M15.M13.D)
            route, scores = module.routes(flat)
            parent_routes[(label, li)] = route.detach().cpu()
            parent_scores[(label, li)] = scores.detach().cpu()
        return hook
    hooks = [w.register_forward_pre_hook(make_parent_hook(li))
             for li, w in enumerate(parent_wrappers)]
    with torch.inference_mode():
        for label, ids in inputs.items():
            parent_logits[label] = model(ids, use_cache=False).logits.detach().cpu()
            budget(start, device)
    for hook in hooks:
        hook.remove()

    expanded = []
    for li, layer in enumerate(model.model.layers):
        wrapper = HierarchicalExperts(parent_wrappers[li], li).to(device)
        wrapper.capture = True
        layer.mlp = wrapper
        expanded.append(wrapper)
        budget(start, device)
    assert all(w.a.shape[0] == w.b.shape[0] == M15.E * CHILDREN for w in expanded)
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    cases = {}
    with torch.inference_mode():
        for label, ids in inputs.items():
            logits = model(ids, use_cache=False).logits.detach().cpu()
            same = torch.equal(logits, parent_logits[label])
            max_diff = float((logits.float() - parent_logits[label].float()).abs().max())
            parent_route_equal = all(torch.equal(w.parent_routes[-1], parent_routes[(label, li)])
                                     for li, w in enumerate(expanded))
            parent_scores_equal = all(torch.equal(w.parent_scores[-1], parent_scores[(label, li)])
                                      for li, w in enumerate(expanded))
            assert same and parent_route_equal and parent_scores_equal
            cases[label] = {"tokens": ids.numel(), "logits_bitwise_equal": same,
                            "max_logit_abs_diff": max_diff,
                            "greedy_equal": bool(torch.equal(logits.argmax(-1),
                                                              parent_logits[label].argmax(-1))),
                            "parent_route_equal": parent_route_equal,
                            "parent_scores_equal": parent_scores_equal,
                            "selected_children_per_position": M15.K}
            budget(start, device)
    per_layer = []
    for li, w in enumerate(expanded):
        per_layer.append({"layer": li,
                          "visited_child_slots": int((w.visited > 0).sum()),
                          "visited_parent_slots": int((w.visited.view(M15.E, CHILDREN).sum(1) > 0).sum()),
                          "min_count": int(w.visited.min()),
                          "max_count": int(w.visited.max()),
                          "selections": int(w.visited.sum())})
    width = M15.M13.D
    bytes_fp32 = {"child_factors": M15.M13.L * M15.E * CHILDREN * 2 * M15.R * width * 4,
                  "parent_product_keys": M15.M13.L * (M55.ROUTER_RANK * width +
                      (M55.AXIS_A + M55.AXIS_B) * M55.ROUTER_RANK) * 4,
                  "child_projection": M15.M13.L * CHILD_RANK * width * 4,
                  "child_keys": M15.M13.L * M15.E * CHILDREN * CHILD_RANK * 4}
    bytes_bf16_selected = M15.M13.L * M15.K * 2 * M15.R * width * 2
    report = {"experiment": "METH-95-hierarchical-E1280-function-preserving-parity",
              "source_sha256": M57.MODEL_SHA,
              "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
              "chat_manifest_sha256": CHAT_SHA,
              "raw_ids_sha256": meta["ids_sha256"],
              "seed": SEED, "parent_experts": M15.E,
              "children_per_parent": CHILDREN, "total_child_experts": M15.E * CHILDREN,
              "active_child_experts": M15.K,
              "cases": cases, "per_layer_visit": per_layer,
              "resident_fp32_bytes": bytes_fp32,
              "selected_bf16_factor_bytes_per_token": bytes_bf16_selected,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__},
              "decision": "exact_expansion_ready_for_distinct_child_training"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": cases,
                      "min_visited_children_per_layer": min(r["visited_child_slots"] for r in per_layer),
                      "max_visited_children_per_layer": max(r["visited_child_slots"] for r in per_layer),
                      "resident_fp32_bytes": bytes_fp32,
                      "runtime": report["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
