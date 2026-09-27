#!/usr/bin/env python3
"""METH-65: one complete pretrained-donor update with E1280 CPU factors."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_e128_smoke as M55
import meth64_sparse_factor_offload as M64


ROOT = Path(__file__).resolve().parents[3]
MODEL_SHA = M55.MODEL_SHA
EXPERTS = 1280
AXIS_A, AXIS_B = 32, 40
MAX_SECONDS = 15 * 60
MAX_RSS = 12 * (1 << 30)
MAX_GPU = 5 * (1 << 30)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(start, device):
    runtime = {"elapsed_seconds": time.monotonic() - start,
               "rss_bytes": psutil.Process().memory_info().rss,
               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (runtime["elapsed_seconds"] > MAX_SECONDS or runtime["rss_bytes"] > MAX_RSS
            or runtime["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-65 budget: {runtime}")
    return runtime


class OffloadedProductKeyExperts(torch.nn.Module):
    def __init__(self, base, layer_id, device):
        super().__init__()
        self.base = base
        self.router = M64.Router(AXIS_A, AXIS_B, M15.M13.D, 64, device, layer_id)
        self.factors = M64.OffloadedFactors(EXPERTS, M15.M13.D, M15.R,
                                            layer_id, b_nonzero=False)
        self.enabled = True

    def forward(self, x):
        dense = self.base(x)
        if not self.enabled:
            return dense
        flat = x.reshape(-1, M15.M13.D)
        chosen, scores = self.router(flat)
        a, b = self.factors.gather(chosen, x.device)
        # Match METH-55: cast the selected FP32 factors and gate to BF16.
        gate = torch.nn.functional.softmax(scores, dim=-1).to(flat.dtype)
        hidden = torch.nn.functional.silu(torch.einsum("nd,nkrd->nkr", flat, a.to(flat.dtype)))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b.to(flat.dtype))
        residual = (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)
        return dense + residual


class SparseRowAdam:
    def __init__(self, lr=3e-4):
        self.lr = lr
        self.state = {}
        self.updated_rows = 0

    @torch.no_grad()
    def step_bank(self, bank, gradients, name):
        for row, grad in gradients.items():
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError((name, row))
            if not bool(torch.any(grad != 0)):
                continue
            key = (name, row)
            if key not in self.state:
                self.state[key] = [torch.zeros_like(grad), torch.zeros_like(grad), 0]
            first, second, age = self.state[key]
            age += 1
            first.mul_(0.9).add_(grad, alpha=0.1)
            second.mul_(0.999).addcmul_(grad, grad, value=0.001)
            corrected_first = first / (1 - 0.9 ** age)
            corrected_second = second / (1 - 0.999 ** age)
            bank[row].addcdiv_(corrected_first,
                               corrected_second.sqrt().add_(1e-8), value=-self.lr)
            self.state[key][2] = age
            self.updated_rows += 1

    def state_bytes(self):
        return sum((first.numel() + second.numel()) * 4
                   for first, second, _ in self.state.values())


def set_enabled(wrappers, enabled):
    for wrapper in wrappers:
        wrapper.enabled = enabled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert file_sha(M55.TEACHER_PATH) == M55.TEACHER_SHA
    assert file_sha(M55.TRAIN_CHAT_PATH) == M55.TRAIN_CHAT_SHA
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    teacher_rows = json.loads(M55.TEACHER_PATH.read_text(encoding="utf-8"))["rows"]
    train_rows = json.loads(M55.TRAIN_CHAT_PATH.read_text(encoding="utf-8"))["rows"]
    chat_by_row = {r["train_row"]: r for r in train_rows}
    first_teacher = teacher_rows[0]
    chat = chat_by_row[first_teacher["train_row"]]
    assert chat["prompt_ids_sha256"] == first_teacher["prompt_ids_sha256"]
    chat_full = chat["prompt_ids"] + first_teacher["continuation_ids"]
    chat_mask = [0] * (len(chat["prompt_ids"]) - 1) + [1] * len(first_teacher["continuation_ids"])
    assert len(chat_full) - 1 == len(chat_mask) and sum(chat_mask) > 0
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        raw = archive["ids"][0, :M15.SEQ].copy()
    assert raw.shape == (128,)
    torch.set_num_threads(6)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors",
                             revision=M42.REV, local_files_only=True)
    assert file_sha(source) == MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    assert (model.config.num_hidden_layers, model.config.hidden_size) == (24, 896)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = OffloadedProductKeyExperts(layer.mlp, li, device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
        budget(start, device)
    model.config.use_cache = False
    model.eval()
    identity = {}
    with torch.no_grad():
        for kind, values in (("raw", raw[:16].tolist()),
                             ("chat", chat["prompt_ids"][:16])):
            ids = torch.tensor(values, device=device).unsqueeze(0)
            set_enabled(wrappers, False)
            donor = model(ids, use_cache=False).logits.float()
            set_enabled(wrappers, True)
            student = model(ids, use_cache=False).logits.float()
            identity[kind] = float((student - donor).abs().max())
    assert max(identity.values()) <= 1e-3
    print(json.dumps({"identity_max_abs": identity, "runtime": budget(start, device)}), flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    router_params = [p for w in wrappers for p in w.router.parameters()]
    router_optimizer = torch.optim.AdamW(router_params, lr=3e-5, weight_decay=0.0)
    router_optimizer.zero_grad(set_to_none=True)
    losses = []
    microbatches = (("raw", raw.tolist()), ("chat", chat_full))
    for kind, values in microbatches:
        ids = torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)
        if kind == "raw":
            inputs, targets = ids, ids[:, 1:]
            ce_mask = kl_mask = torch.ones_like(targets, dtype=torch.float32)
            kl_weight = 0.5
        else:
            inputs, targets = ids[:, :-1], ids[:, 1:]
            ce_mask = torch.tensor(chat_mask, dtype=torch.float32, device=device).unsqueeze(0)
            kl_mask = torch.ones_like(targets, dtype=torch.float32)
            kl_weight = 1.0
        set_enabled(wrappers, False)
        with torch.no_grad():
            teacher = model(inputs, use_cache=False).logits
        set_enabled(wrappers, True)
        student = model(inputs, use_cache=False).logits
        if kind == "raw":
            teacher, student = teacher[:, :-1], student[:, :-1]
        ce, kl, objective = M55.masked_objective(student, teacher, targets,
                                                ce_mask, kl_mask, kl_weight)
        loss = objective / 2
        assert bool(torch.isfinite(loss))
        loss.backward()
        losses.append({"kind": kind, "ce": float(ce.detach()),
                       "kl": float(kl.detach()), "objective": float(objective.detach())})
        print(json.dumps({"microbatch": kind, "runtime": budget(start, device)}), flush=True)
        del teacher, student, ce, kl, objective, loss
    b_norms = []
    a_rows = []
    b_rows = []
    router_norms = []
    for wrapper in wrappers:
        factors = wrapper.factors
        a_rows.append(sum(bool(torch.any(g != 0)) for g in factors.grad_a.values()))
        b_rows.append(sum(bool(torch.any(g != 0)) for g in factors.grad_b.values()))
        b_norms.append(math.sqrt(sum(float(g.double().square().sum())
                                     for g in factors.grad_b.values())))
        router_norms.append(math.sqrt(sum(float(p.grad.double().square().sum())
                                          for p in wrapper.router.parameters()
                                          if p.grad is not None)))
    assert min(b_norms) > 0 and min(b_rows) > 0
    assert all(math.isfinite(x) for x in b_norms + router_norms)
    sparse_optimizer = SparseRowAdam()
    for li, wrapper in enumerate(wrappers):
        sparse_optimizer.step_bank(wrapper.factors.a, wrapper.factors.grad_a, f"a{li}")
        sparse_optimizer.step_bank(wrapper.factors.b, wrapper.factors.grad_b, f"b{li}")
    router_optimizer.step()
    changed_slots = [int((w.factors.b.abs().flatten(1).sum(1) > 0).sum()) for w in wrappers]
    assert min(changed_slots) > 0
    model.gradient_checkpointing_disable()
    model.eval()
    with torch.no_grad():
        set_enabled(wrappers, True)
        post_logits = model(torch.tensor(raw[:16].tolist(), device=device).unsqueeze(0),
                            use_cache=False).logits
    assert bool(torch.isfinite(post_logits).all())
    runtime = budget(start, device)
    result = {"experiment": "METH-65-E1280-real-donor-one-update",
              "source_sha256": file_sha(__file__), "donor_sha256": MODEL_SHA,
              "teacher_sha256": M55.TEACHER_SHA, "train_chat_sha256": M55.TRAIN_CHAT_SHA,
              "train_ids_sha256": M15.TRAIN_IDS_SHA, "chat_train_row": chat["train_row"],
              "identity_max_abs": identity, "losses": losses,
              "min_b_gradient_norm": min(b_norms), "max_b_gradient_norm": max(b_norms),
              "min_nonzero_a_gradient_rows": min(a_rows),
              "min_nonzero_b_gradient_rows": min(b_rows),
              "max_router_gradient_norm": max(router_norms),
              "min_changed_b_slots": min(changed_slots),
              "max_changed_b_slots": max(changed_slots),
              "sparse_optimizer_updated_rows": sparse_optimizer.updated_rows,
              "sparse_optimizer_state_bytes": sparse_optimizer.state_bytes(),
              "post_logits_finite": True, "runtime": runtime,
              "decision": "one_real_update_apparatus_pass_only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "min_changed_b_slots": min(changed_slots),
                      "runtime": runtime}), flush=True)


if __name__ == "__main__":
    main()
