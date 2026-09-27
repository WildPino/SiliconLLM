#!/usr/bin/env python3
"""METH-78: local gradient norms of the frozen METH-71 loss components."""

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

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth71_balanced_e128_e1280_smoke as M71RUN
import meth71_balanced_product_key as M71


ROOT = Path(__file__).resolve().parents[3]
PARENT_SHA = {
    128: "f50a72f1ab86bc304fc6cb23e167a719131b3b79ba8a44201b1e98d716d4c98c",
    1280: "ead764d1f258819f2ee2d7e6af7995d329a078e360f06f085325c96ac6437fbb",
}
MAX_SECONDS = 12 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    report = {
        "elapsed_seconds": time.monotonic() - start,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "rss_bytes": psutil.Process().memory_info().rss,
    }
    if (report["elapsed_seconds"] > MAX_SECONDS
            or report["gpu_peak_allocated_bytes"] > MAX_GPU_BYTES
            or report["rss_bytes"] > MAX_RSS_BYTES):
        raise RuntimeError(f"METH-78 budget exceeded: {report}")
    return report


def gradient_norm(parameters):
    squared = None
    nonzero = 0
    for p in parameters:
        if p.grad is not None:
            contribution = p.grad.detach().float().square().sum()
            squared = contribution if squared is None else squared + contribution
            nonzero += int(bool(torch.count_nonzero(p.grad)))
    return {"norm": math.sqrt(float(squared)) if squared is not None else 0.0,
            "nonzero_tensors": nonzero}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--axis-a", required=True, type=int)
    parser.add_argument("--axis-b", required=True, type=int)
    args = parser.parse_args()
    experts = args.axis_a * args.axis_b
    assert (args.axis_a, args.axis_b) in ((8, 16), (32, 40))
    assert M15.M13.sha256(args.checkpoint) == PARENT_SHA[experts]
    assert M17.sha(M71RUN.TEACHER_PATH.read_bytes()) == M71RUN.TEACHER_SHA
    assert M17.sha(M71RUN.TRAIN_CHAT_PATH.read_bytes()) == M71RUN.TRAIN_CHAT_SHA
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        raw_ids = archive["ids"][123, :M15.SEQ].copy()
    assert len(raw_ids) == M15.SEQ
    teacher = json.loads(M71RUN.TEACHER_PATH.read_text(encoding="utf-8"))
    chat_manifest = json.loads(M71RUN.TRAIN_CHAT_PATH.read_text(encoding="utf-8"))
    chat_by_row = {row["train_row"]: row for row in chat_manifest["rows"]}
    first = teacher["rows"][0]
    prompt = chat_by_row[first["train_row"]]
    assert prompt["prompt_ids_sha256"] == first["prompt_ids_sha256"]
    full_ids = prompt["prompt_ids"] + first["continuation_ids"]
    chat_mask = [0] * (len(prompt["prompt_ids"]) - 1) + [1] * len(first["continuation_ids"])
    assert len(chat_mask) == len(full_ids) - 1

    torch.set_num_threads(6)
    torch.manual_seed(M15.SEED)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM

    source = hf_hub_download(M42.MODEL, "model.safetensors",
                             revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M71RUN.MODEL_SHA
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M71.BalancedProductKeyExperts(
            layer.mlp, li, args.axis_a, args.axis_b).to(device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    model.config.use_cache = False
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert state["updates"] == 64
    assert (state["axis_a"], state["axis_b"]) == (args.axis_a, args.axis_b)
    assert state["source_sha256"] == M71RUN.MODEL_SHA
    assert state["dev_sha256"] == M71RUN.PROMPT_SHA
    with torch.no_grad():
        for wrapper, saved in zip(wrappers, state["expert_state"]):
            for name in ("a", "b", "router"):
                getattr(wrapper, name).copy_(saved[name].to(device))
    del state
    gc.collect()
    budget(start, device)
    groups = {
        "a": [w.a for w in wrappers],
        "b": [w.b for w in wrappers],
        "router": [w.router for w in wrappers],
    }
    groups["all"] = groups["a"] + groups["b"] + groups["router"]
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    results = {}
    for kind in ("raw", "chat"):
        seq = raw_ids.tolist() if kind == "raw" else full_ids
        ids = torch.as_tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        inputs = ids if kind == "raw" else ids[:, :-1]
        targets = ids[:, 1:]
        ce_mask = (torch.ones_like(targets, dtype=torch.float32) if kind == "raw"
                   else torch.as_tensor(chat_mask, dtype=torch.float32,
                                        device=device).unsqueeze(0))
        kl_mask = torch.ones_like(targets, dtype=torch.float32)
        kl_weight = 2.0 if kind == "raw" else 4.0
        M71RUN.set_experts(wrappers, False)
        with torch.no_grad():
            teacher_logits = model(inputs, use_cache=False).logits.detach()
        results[kind] = {"sequence_positions": inputs.numel(), "terms": {}}
        for term in ("ce", "kl", "balance"):
            model.zero_grad(set_to_none=True)
            M71RUN.set_experts(wrappers, True)
            student_logits = model(inputs, use_cache=False).logits
            effective_student = student_logits[:, :-1] if kind == "raw" else student_logits
            effective_teacher = teacher_logits[:, :-1] if kind == "raw" else teacher_logits
            ce, kl, _ = M71RUN.masked_objective(
                effective_student, effective_teacher, targets,
                ce_mask, kl_mask, kl_weight)
            balance = torch.stack([w.balance_loss for w in wrappers]).sum()
            selected = {"ce": ce, "kl": kl_weight * kl,
                        "balance": 0.02 * balance}[term]
            selected.backward()
            norms = {name: gradient_norm(params) for name, params in groups.items()}
            results[kind]["terms"][term] = {
                "value": float(selected.detach()), "gradients": norms}
            print(json.dumps({"experts": experts, "kind": kind, "term": term,
                              "value": float(selected.detach()),
                              "norms": {k: v["norm"] for k, v in norms.items()}}),
                  flush=True)
            del student_logits, ce, kl, balance, selected
            budget(start, device)
        del teacher_logits
    for kind in results:
        terms = results[kind]["terms"]
        results[kind]["ratios_to_ce"] = {
            term: {group: (terms[term]["gradients"][group]["norm"]
                           / terms["ce"]["gradients"][group]["norm"]
                           if terms["ce"]["gradients"][group]["norm"] else None)
                   for group in groups}
            for term in ("kl", "balance")}
    result = {"experiment": "METH-78-objective-gradient-attribution",
              "experts": experts, "product_axes": [args.axis_a, args.axis_b],
              "checkpoint_sha256": PARENT_SHA[experts],
              "raw_row": 123, "raw_offset": 0,
              "chat_train_row": first["train_row"], "results": results,
              "runtime": budget(start, device)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
