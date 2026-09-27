#!/usr/bin/env python3
"""METH-69: reproduce METH-55 donor training with CPU factors."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_e128_smoke as M55
import meth55_product_key_experts as PK
import meth68_dense_adam_offload_parity as M68


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth55_product_key_e128_smoke.pt"
FROZEN_SHA = "2d652709c6730e2b4f3a1a543834d7eec7409ef0f374738ae24d57ae2a5b5bd3"
FROZEN_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth55_product_key_e128_smoke_result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(start, device):
    result = {"elapsed_seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["elapsed_seconds"] > 720 or result["rss_bytes"] > 8 * (1 << 30)
            or result["gpu_peak_allocated_bytes"] > 5 * (1 << 30)):
        raise RuntimeError(f"METH-69 budget: {result}")
    return result


class OffloadedExperts(nn.Module):
    def __init__(self, base, layer_id, device):
        super().__init__()
        seeded = PK.ProductKeyExperts(base, layer_id)
        self.base = base
        self.offload = M68.OffloadedReference(seeded, device)
        self.enabled = True

    def forward(self, x):
        dense = self.base(x)
        if not self.enabled:
            return dense
        flat = x.reshape(-1, M15.M13.D)
        chosen, scores = self.offload.routes(flat)
        a, b = self.offload.factors.gather(chosen, x.device)
        gate = F.softmax(scores, dim=-1).to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a.to(flat.dtype)))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b.to(flat.dtype))
        return dense + (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)


def enable(wrappers, value):
    for wrapper in wrappers:
        wrapper.enabled = value


@torch.no_grad()
def bpb(model, wrappers, ids, byts, enabled):
    enable(wrappers, enabled)
    nats = 0.0
    for i in range(ids.shape[0]):
        seq = ids[i:i + 1]
        logits = model(seq, use_cache=False).logits.float()
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        nats += float(-lp.gather(-1, seq[:, 1:].unsqueeze(-1)).sum())
    return nats / (math.log(2) * int(byts.sum()))


@torch.no_grad()
def top1(model, wrappers, prompts, device, start):
    matching = positions = 0
    for row in prompts:
        ids = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device).unsqueeze(0)
        enable(wrappers, False)
        donor = model(ids, use_cache=False).logits.argmax(dim=-1)
        enable(wrappers, True)
        student = model(ids, use_cache=False).logits.argmax(dim=-1)
        matching += int((donor == student).sum())
        positions += ids.numel()
        budget(start, device)
    return {"matching": matching, "positions": positions,
            "agreement": matching / positions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert sha(FROZEN) == FROZEN_SHA
    assert sha(M55.TEACHER_PATH) == M55.TEACHER_SHA
    assert sha(M55.TRAIN_CHAT_PATH) == M55.TRAIN_CHAT_SHA
    assert sha(M55.PROMPT_PATH) == M55.PROMPT_SHA
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    frozen_result = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
    teacher = json.loads(M55.TEACHER_PATH.read_text(encoding="utf-8"))["rows"]
    train_chat = json.loads(M55.TRAIN_CHAT_PATH.read_text(encoding="utf-8"))["rows"]
    prompts = {row["train_row"]: row for row in train_chat}
    dev = json.loads(M55.PROMPT_PATH.read_text(encoding="utf-8"))["rows"]
    assert len(teacher) == len(prompts) == 256 and len(dev) == 24
    excluded = {row["train_row"] for row in dev}
    assert len(excluded) == 24 and excluded.isdisjoint(prompts)
    raw_pool = np.asarray([i for i in range(31250) if i not in excluded])
    chat_rows = []
    for saved in teacher:
        prompt = prompts[saved["train_row"]]
        assert prompt["prompt_ids_sha256"] == saved["prompt_ids_sha256"]
        full = prompt["prompt_ids"] + saved["continuation_ids"]
        assert full[-129:] == saved["window_ids"]
        mask = [0] * (len(prompt["prompt_ids"]) - 1) + [1] * len(saved["continuation_ids"])
        assert len(mask) == len(full) - 1
        chat_rows.append({"full": full, "mask": mask})
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        train_ids = archive["ids"].copy()
    assert train_ids.shape == (31250, 512) and train_ids.dtype == np.int32
    assert hashlib.sha256(train_ids.tobytes()).hexdigest() == M15.TRAIN_IDS_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    torch.manual_seed(M15.SEED)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(M15.SEED)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    rng = np.random.default_rng(M55.RNG_SEED)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert sha(source) == M55.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    eval_ids, eval_bytes, meta = M15.M13.C.get_slice(tokenizer, "heldout", 4, 256, 271828)
    assert meta["ids_sha256"] == M15.EVAL_IDS_SHA and int(eval_bytes.sum()) == 4588
    eval_ids = torch.as_tensor(eval_ids, dtype=torch.long, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = OffloadedExperts(layer.mlp, li, device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    model.config.use_cache = False
    model.eval()
    donor_initial = bpb(model, wrappers, eval_ids, eval_bytes, False)
    student_initial = bpb(model, wrappers, eval_ids, eval_bytes, True)
    assert abs(donor_initial - student_initial) <= 1e-5
    assert abs(donor_initial - 0.9712552075318387) <= 1e-5
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    router_params = [w.offload.router for w in wrappers]
    router_opt = torch.optim.AdamW(router_params, lr=3e-5, weight_decay=0.0)
    cpu_opt = M68.DenseCPUAdamW(lr=3e-4)
    steps = []
    for step in range(1, 17):
        router_opt.zero_grad(set_to_none=True)
        for wrapper in wrappers:
            wrapper.offload.factors.grad_a.clear()
            wrapper.offload.factors.grad_b.clear()
        microbatches = []
        for micro in range(4):
            if micro < 2:
                row = int(rng.choice(raw_pool))
                offset = int(rng.integers(512 - M15.SEQ + 1))
                ids = torch.tensor(train_ids[row, offset:offset + M15.SEQ],
                                   dtype=torch.long, device=device).unsqueeze(0)
                inputs, targets = ids, ids[:, 1:]
                ce_mask = kl_mask = torch.ones_like(targets, dtype=torch.float32)
                kl_weight = 0.5
                kind = "raw"
            else:
                row = chat_rows[int(rng.integers(len(chat_rows)))]
                ids = torch.tensor(row["full"], dtype=torch.long, device=device).unsqueeze(0)
                inputs, targets = ids[:, :-1], ids[:, 1:]
                ce_mask = torch.tensor(row["mask"], dtype=torch.float32,
                                       device=device).unsqueeze(0)
                kl_mask = torch.ones_like(targets, dtype=torch.float32)
                kl_weight = 1.0
                kind = "chat"
            enable(wrappers, False)
            with torch.no_grad():
                donor = model(inputs, use_cache=False).logits
            enable(wrappers, True)
            student = model(inputs, use_cache=False).logits
            assert student.requires_grad
            if kind == "raw":
                donor, student = donor[:, :-1], student[:, :-1]
            ce, kl, objective = M55.masked_objective(student, donor, targets,
                                                    ce_mask, kl_mask, kl_weight)
            assert bool(torch.isfinite(objective))
            (objective / 4).backward()
            microbatches.append({"kind": kind, "ce": float(ce.detach()),
                                 "kl": float(kl.detach())})
            del donor, student, ce, kl, objective
            budget(start, device)
        squared = 0.0
        for wrapper in wrappers:
            factors = wrapper.offload.factors
            squared += sum(float(g.double().square().sum())
                           for g in list(factors.grad_a.values()) + list(factors.grad_b.values()))
            if wrapper.offload.router.grad is not None:
                squared += float(wrapper.offload.router.grad.double().square().sum())
        pre_clip = math.sqrt(squared)
        assert math.isfinite(pre_clip) and pre_clip < 1.0
        cpu_opt.begin_step()
        for li, wrapper in enumerate(wrappers):
            factors = wrapper.offload.factors
            cpu_opt.step_bank(factors.a, factors.grad_a, f"a{li}")
            cpu_opt.step_bank(factors.b, factors.grad_b, f"b{li}")
        router_opt.step()
        original = frozen_result["training_steps"][step - 1]
        assert original["update"] == step
        steps.append({"update": step, "microbatches": microbatches,
                      "pre_clip_norm": pre_clip,
                      "original_pre_clip_norm": original["pre_clip_norm"],
                      "runtime": budget(start, device)})
        print(json.dumps({"update": step, "pre_clip_norm": pre_clip,
                          "original_pre_clip_norm": original["pre_clip_norm"],
                          "elapsed_seconds": steps[-1]["runtime"]["elapsed_seconds"]}), flush=True)
    model.gradient_checkpointing_disable()
    model.eval()
    donor_bpb = bpb(model, wrappers, eval_ids, eval_bytes, False)
    student_bpb = bpb(model, wrappers, eval_ids, eval_bytes, True)
    prompt_top1 = top1(model, wrappers, dev, device, start)
    frozen = torch.load(FROZEN, map_location="cpu", weights_only=False)
    max_error = {"a": 0.0, "b": 0.0, "router": 0.0}
    for wrapper, saved in zip(wrappers, frozen["expert_state"]):
        for key, actual in (("a", wrapper.offload.factors.a),
                            ("b", wrapper.offload.factors.b),
                            ("router", wrapper.offload.router.detach().cpu())):
            error = float((actual.float() - saved[key].float()).abs().max())
            max_error[key] = max(max_error[key], error)
    gates = {"parameter": max(max_error.values()) <= 2e-5,
             "prompt_top1": prompt_top1["matching"] == 3658 and prompt_top1["positions"] == 3811,
             "raw_bpb": abs(student_bpb - 0.9697036007476514) <= 1e-5}
    gates["joint"] = all(gates.values())
    result = {"experiment": "METH-69-full-E128-offload-replay",
              "source_sha256": sha(__file__), "donor_sha256": M55.MODEL_SHA,
              "frozen_checkpoint_sha256": FROZEN_SHA,
              "train_ids_sha256": M15.TRAIN_IDS_SHA,
              "teacher_sha256": M55.TEACHER_SHA,
              "train_chat_sha256": M55.TRAIN_CHAT_SHA,
              "dev_sha256": M55.PROMPT_SHA,
              "donor_initial_bpb": donor_initial,
              "student_initial_bpb": student_initial,
              "donor_terminal_bpb": donor_bpb,
              "student_terminal_bpb": student_bpb,
              "prompt_top1": prompt_top1,
              "max_checkpoint_parameter_error": max_error,
              "steps": steps, "gates": gates,
              "runtime": budget(start, device),
              "decision": "offload_replay_pass" if gates["joint"] else "offload_replay_gap"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "max_parameter_error": max_error,
                      "top1": prompt_top1, "student_bpb": student_bpb,
                      "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
