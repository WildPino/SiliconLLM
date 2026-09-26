#!/usr/bin/env python3
"""METH-15: exact-donor Qwen0.5B with trainable sparse residual experts."""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import meth13_qwen05b_preflight as M13


TRAIN_PATH = os.path.join(os.path.dirname(__file__), "results", "h0", "h0_train.npz")
TRAIN_FILE_SHA = "0cdaa28f405c3a8c6a8589af8e88788b33131bc7d4f1096da6c1fedf78caa9d9"
TRAIN_IDS_SHA = "9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da"
EVAL_IDS_SHA = "c5345782fc0b08ef3212403878a33960ae0da236c999272e79e418f29c277d3c"
E, K, R, SEQ, ACCUM, STEPS = 128, 4, 8, 128, 4, 16
SEED = 5151
MAX_SECONDS = 20 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-15 time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-15 process RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-15 peak allocated GPU stop: {peak}")
    return elapsed, rss, peak


class ResidualExperts(nn.Module):
    def __init__(self, base, layer_id):
        super().__init__()
        self.base = base
        self.enabled = True
        self.a = nn.Parameter(torch.empty(E, R, M13.D, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(E, M13.D, R, dtype=torch.float32))
        self.router = nn.Parameter(torch.empty(E, M13.D, dtype=torch.float32))
        gen = torch.Generator(device="cpu").manual_seed(SEED + layer_id)
        with torch.no_grad():
            self.a.copy_(torch.randn(self.a.shape, generator=gen) * (0.02 / math.sqrt(M13.D)))
            self.router.copy_(torch.randn(self.router.shape, generator=gen) / math.sqrt(M13.D))

    def forward(self, x):
        dense = self.base(x)
        if not self.enabled:
            return dense
        flat = x.reshape(-1, M13.D)
        scores = F.linear(flat.float(), self.router)
        chosen = torch.topk(scores, K, dim=-1).indices
        gate = F.softmax(scores.gather(1, chosen), dim=-1).to(flat.dtype)
        a = self.a[chosen].to(flat.dtype)
        b = self.b[chosen].to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b)
        residual = (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)
        return dense + residual

    def changed_slots(self):
        return int((self.b.detach().float().flatten(1).norm(dim=1) > 1e-9).sum())


@torch.no_grad()
def bpb(model, ids, byts, wrappers, enabled):
    for w in wrappers:
        w.enabled = enabled
    total = 0.0
    for i in range(ids.shape[0]):
        seq = ids[i:i+1]
        logits = model(seq).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        total += float(-logp.gather(-1, seq[:, 1:].unsqueeze(-1)).sum())
    return total / (math.log(2) * int(byts.sum()))


def grad_range(wrappers):
    b_vals = [float(w.b.grad.detach().float().norm()) if w.b.grad is not None else 0.0
              for w in wrappers]
    r_vals = [float(w.router.grad.detach().float().norm())
              if w.router.grad is not None else 0.0 for w in wrappers]
    return {"b_min": min(b_vals), "b_max": max(b_vals),
            "router_min": min(r_vals), "router_max": max(r_vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device-name", default="NVIDIA GeForce RTX 3060")
    args = ap.parse_args()
    start = time.monotonic()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(6)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == args.device_name]
    assert len(matches) == 1, (args.device_name, matches)
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # M13 imports common.py, which disables grad globally for inference tools.
    torch.set_grad_enabled(True)

    source_path = hf_hub_download(M13.MODEL, "model.safetensors",
                                  revision=M13.REV, local_files_only=True)
    assert M13.sha256(source_path) == M13.MODEL_SHA
    tok = AutoTokenizer.from_pretrained(M13.MODEL, revision=M13.REV,
                                        local_files_only=True)
    assert M13.C.tok_fingerprint(tok) == M13.TOK_FP
    assert M13.sha256(TRAIN_PATH) == TRAIN_FILE_SHA
    with np.load(TRAIN_PATH, allow_pickle=False) as archive:
        train_ids = archive["ids"].copy()
    assert train_ids.shape == (31250, 512) and train_ids.dtype == np.int32
    assert sha_bytes(train_ids.tobytes()) == TRAIN_IDS_SHA
    assert int(train_ids.max()) < 151936
    eval_ids, eval_bytes, eval_meta = M13.C.get_slice(tok, "heldout", 4, 256, 271828)
    assert eval_meta["ids_sha256"] == EVAL_IDS_SHA and int(eval_bytes.sum()) == 4588
    eval_ids = torch.as_tensor(eval_ids, dtype=torch.long, device=device)
    rng = np.random.default_rng(SEED)

    model = AutoModelForCausalLM.from_pretrained(
        M13.MODEL, revision=M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    cfg = model.config
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size,
            cfg.vocab_size, cfg.num_key_value_heads) == (M13.L, M13.D,
                                                         M13.FF, 151936, 2)
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        w = ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = w
        wrappers.append(w)
    model.config.use_cache = False
    model.eval()
    with torch.no_grad():
        for w in wrappers:
            w.enabled = False
        control = model(eval_ids[:1, :16]).logits.float()
        for w in wrappers:
            w.enabled = True
        active = model(eval_ids[:1, :16]).logits.float()
    identity = float((active - control).abs().max())
    assert identity <= 1e-3, f"step-zero identity {identity}"
    donor_before = bpb(model, eval_ids, eval_bytes, wrappers, False)
    student_before = bpb(model, eval_ids, eval_bytes, wrappers, True)
    assert abs(student_before - donor_before) <= 1e-5
    print(f"identity={identity:.8g} donor_bpb={donor_before:.8f} ", flush=True)
    budget(start, device)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    params_expert = [p for w in wrappers for p in (w.a, w.b)]
    params_router = [w.router for w in wrappers]
    opt = torch.optim.AdamW([
        {"params": params_expert, "lr": 1e-3},
        {"params": params_router, "lr": 1e-4},
    ], weight_decay=0.0)
    steps = []
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        ce_sum = kl_sum = 0.0
        for _ in range(ACCUM):
            row = int(rng.integers(train_ids.shape[0]))
            offset = int(rng.integers(0, 512 - SEQ + 1))
            ids = torch.as_tensor(train_ids[row, offset:offset + SEQ],
                                  dtype=torch.long, device=device).unsqueeze(0)
            for w in wrappers:
                w.enabled = False
            with torch.no_grad():
                teacher = model(ids).logits[:, :-1].float()
                teacher_lp = F.log_softmax(teacher, dim=-1)
                teacher_p = teacher_lp.exp()
            for w in wrappers:
                w.enabled = True
            student = model(ids).logits[:, :-1].float()
            student_lp = F.log_softmax(student, dim=-1)
            ce = F.nll_loss(student_lp.reshape(-1, student_lp.shape[-1]),
                            ids[:, 1:].reshape(-1))
            kl = (teacher_p * (teacher_lp - student_lp)).sum(dim=-1).mean()
            loss = (ce + 0.1 * kl) / ACCUM
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            ce_sum += float(ce.detach())
            kl_sum += float(kl.detach())
            del teacher, teacher_lp, teacher_p, student, student_lp, loss
            budget(start, device)
        grad = grad_range(wrappers)
        assert grad["b_min"] > 0 and math.isfinite(grad["b_max"]), (step, grad)
        if step >= 2:
            assert grad["router_min"] > 0 and math.isfinite(grad["router_max"]), (step, grad)
        norm = float(torch.nn.utils.clip_grad_norm_(params_expert + params_router, 1.0))
        assert math.isfinite(norm)
        opt.step()
        elapsed, rss, peak = budget(start, device)
        row = {"step": step, "ce": ce_sum / ACCUM, "kl": kl_sum / ACCUM,
               "grad": grad, "clip_pre_norm": norm, "elapsed_seconds": elapsed,
               "rss_bytes": rss, "gpu_peak_allocated_bytes": peak}
        steps.append(row)
        print(json.dumps({"step": step, "ce": row["ce"], "kl": row["kl"],
                          "b_min_grad": grad["b_min"],
                          "router_min_grad": grad["router_min"],
                          "seconds": elapsed}), flush=True)

    model.eval()
    model.gradient_checkpointing_disable()
    donor_after = bpb(model, eval_ids, eval_bytes, wrappers, False)
    student_after = bpb(model, eval_ids, eval_bytes, wrappers, True)
    assert abs(donor_after - donor_before) <= 1e-5
    slots = [w.changed_slots() for w in wrappers]
    delta = student_after - donor_after
    elapsed, rss, peak = budget(start, device)
    decision = ("eligible_for_frozen_longer_training_design"
                if min(slots) >= 16 and delta <= 0.10
                else "stop_this_residual_expert_smoke")
    checkpoint = os.path.abspath(args.checkpoint)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save({"expert_state": [
                   {k: v.detach().cpu() for k, v in w.state_dict().items()
                    if k in ("a", "b", "router")} for w in wrappers],
                "optimizer": opt.state_dict(),
                "np_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "updates": STEPS, "source_sha256": M13.MODEL_SHA,
                "train_ids_sha256": TRAIN_IDS_SHA}, checkpoint)
    result = {
        "experiment": "METH-15", "source": {"model": M13.MODEL,
            "revision": M13.REV, "model_sha256": M13.MODEL_SHA,
            "tokenizer_fingerprint": M13.TOK_FP},
        "data": {"train_file_sha256": TRAIN_FILE_SHA,
            "train_ids_sha256": TRAIN_IDS_SHA, "train_shape": list(train_ids.shape),
            "eval_ids_sha256": EVAL_IDS_SHA, "eval_scored_bytes": int(eval_bytes.sum())},
        "geometry": {"layers": M13.L, "width": M13.D, "experts": E,
                     "top_k": K, "rank": R, "seed": SEED,
                     "steps": STEPS, "seq_len": SEQ, "accum": ACCUM,
                     "expert_parameters": M13.L * E * 2 * R * M13.D,
                     "router_parameters": M13.L * E * M13.D},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "cuda_index": matches[0],
                    "gpu": torch.cuda.get_device_name(device),
                    "elapsed_seconds": elapsed, "rss_end_bytes": rss,
                    "gpu_peak_allocated_bytes": peak},
        "controls": {"identity_max_abs": identity,
                     "donor_bpb_initial": donor_before,
                     "student_bpb_initial": student_before,
                     "donor_bpb_terminal": donor_after,
                     "student_bpb_terminal": student_after,
                     "delta_bpb_terminal": delta,
                     "changed_slots_by_layer": slots,
                     "min_changed_slots": min(slots)},
        "training_steps": steps, "decision": decision,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": M13.sha256(checkpoint),
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"controls": result["controls"], "runtime": result["runtime"],
                      "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
