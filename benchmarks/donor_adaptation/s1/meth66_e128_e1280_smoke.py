#!/usr/bin/env python3
"""METH-66: matched 16-update E128/E1280 donor-transfer smoke."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_e128_smoke as M55
import meth64_sparse_factor_offload as M64
import meth65_e1280_real_donor_update as M65


ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth66_e1280_dev_manifest.json"
DEV_SHA = "56cb7987f344f4a7cc8f108a9aa17fe5ed481174aebab8f8eeb0c3a115d3ed09"
UPDATES = 16
SEED = 666667
MAX_SECONDS = 30 * 60
MAX_GPU = 5 * (1 << 30)
MAX_RSS = 12 * (1 << 30)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(start, device):
    result = {"elapsed_seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["elapsed_seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS
            or result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-66 budget: {result}")
    return result


class Experts(torch.nn.Module):
    def __init__(self, base, layer_id, axis_a, axis_b, device):
        super().__init__()
        self.base = base
        self.router = M64.Router(axis_a, axis_b, M15.M13.D, 64, device, layer_id)
        self.factors = M64.OffloadedFactors(axis_a * axis_b, M15.M13.D, M15.R,
                                            layer_id, b_nonzero=False)
        self.enabled = True

    def forward(self, x):
        dense = self.base(x)
        if not self.enabled:
            return dense
        flat = x.reshape(-1, M15.M13.D)
        chosen, scores = self.router(flat)
        a, b = self.factors.gather(chosen, x.device)
        gate = F.softmax(scores, dim=-1).to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a.to(flat.dtype)))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b.to(flat.dtype))
        return dense + (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)


def enable(wrappers, value):
    for wrapper in wrappers:
        wrapper.enabled = value


def clipped_global_norm(wrappers):
    squared = 0.0
    for wrapper in wrappers:
        for gradients in (wrapper.factors.grad_a, wrapper.factors.grad_b):
            squared += sum(float(g.double().square().sum()) for g in gradients.values())
        squared += sum(float(p.grad.double().square().sum()) for p in wrapper.router.parameters()
                       if p.grad is not None)
    norm = math.sqrt(squared)
    assert math.isfinite(norm)
    if norm > 1.0:
        multiplier = 1.0 / norm
        for wrapper in wrappers:
            for gradients in (wrapper.factors.grad_a, wrapper.factors.grad_b):
                for grad in gradients.values():
                    grad.mul_(multiplier)
            for param in wrapper.router.parameters():
                if param.grad is not None:
                    param.grad.mul_(multiplier)
    return norm


@torch.no_grad()
def bpb(model, wrappers, ids, byts):
    values = []
    for enabled in (False, True):
        enable(wrappers, enabled)
        nats = 0.0
        for i in range(ids.shape[0]):
            seq = ids[i:i + 1]
            logits = model(seq, use_cache=False).logits.float()
            lp = F.log_softmax(logits[:, :-1], dim=-1)
            nats += float(-lp.gather(-1, seq[:, 1:].unsqueeze(-1)).sum())
        values.append(nats / (math.log(2) * int(byts.sum())))
    return {"donor": values[0], "student": values[1], "delta": values[1] - values[0]}


@torch.no_grad()
def dev_top1(model, wrappers, rows, device, experts, start):
    counts = [torch.zeros(experts, dtype=torch.int64) for _ in wrappers]
    hooks = []
    for i, wrapper in enumerate(wrappers):
        def accumulate(_module, _input, output, index=i):
            counts[index].add_(torch.bincount(output[0].flatten().cpu(), minlength=experts))
        hooks.append(wrapper.router.register_forward_hook(accumulate))
    matching = positions = 0
    per_prompt = []
    try:
        for row in rows:
            ids = torch.tensor(row["prompt_ids"], dtype=torch.long,
                               device=device).unsqueeze(0)
            enable(wrappers, False)
            donor = model(ids, use_cache=False).logits.argmax(dim=-1)
            enable(wrappers, True)
            student = model(ids, use_cache=False).logits.argmax(dim=-1)
            same = int((donor == student).sum())
            matching += same
            positions += ids.numel()
            per_prompt.append({"train_row": row["train_row"],
                               "matching": same, "positions": ids.numel()})
            budget(start, device)
    finally:
        for hook in hooks:
            hook.remove()
    load = [{"selected_slots": int((count > 0).sum()),
             "max_to_mean": float(count.max() / count.float().mean()),
             "total_selections": int(count.sum())} for count in counts]
    return {"matching": matching, "positions": positions,
            "agreement": matching / positions,
            "per_prompt": per_prompt,
            "route_load_by_layer": load,
            "min_selected_slots": min(x["selected_slots"] for x in load),
            "worst_max_to_mean_load": max(x["max_to_mean"] for x in load)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis-a", type=int, required=True)
    ap.add_argument("--axis-b", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    assert (args.axis_a, args.axis_b) in ((8, 16), (32, 40))
    experts = args.axis_a * args.axis_b
    assert sha(DEV) == DEV_SHA
    assert sha(M55.TEACHER_PATH) == M55.TEACHER_SHA
    assert sha(M55.TRAIN_CHAT_PATH) == M55.TRAIN_CHAT_SHA
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    teacher = json.loads(M55.TEACHER_PATH.read_text(encoding="utf-8"))["rows"]
    train_chat = json.loads(M55.TRAIN_CHAT_PATH.read_text(encoding="utf-8"))["rows"]
    prompts = {row["train_row"]: row for row in train_chat}
    assert len(prompts) == len(teacher) == 256 and len(dev["rows"]) == 24
    excluded = {row["train_row"] for row in dev["rows"]}
    for name in dev["prior_manifest_sha256"]:
        prior = json.loads((DEV.parent / name).read_text(encoding="utf-8"))
        excluded.update(row["train_row"] for row in prior["rows"])
    assert len(excluded) == 256 + 5 * 24 and not excluded.isdisjoint(prompts)
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        train_ids = archive["ids"].copy()
    assert train_ids.shape == (31250, 512)
    assert hashlib.sha256(train_ids.tobytes()).hexdigest() == M15.TRAIN_IDS_SHA
    raw_pool = np.asarray([i for i in range(len(train_ids)) if i not in excluded])
    chat_rows = []
    for saved in teacher:
        prompt = prompts[saved["train_row"]]
        assert prompt["prompt_ids_sha256"] == saved["prompt_ids_sha256"]
        full = prompt["prompt_ids"] + saved["continuation_ids"]
        mask = [0] * (len(prompt["prompt_ids"]) - 1) + [1] * len(saved["continuation_ids"])
        assert len(mask) == len(full) - 1
        chat_rows.append({"full": full, "mask": mask})
    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    torch.manual_seed(SEED)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    rng = np.random.default_rng(SEED)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert sha(source) == M55.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    eval_ids, eval_bytes, eval_meta = M15.M13.C.get_slice(tokenizer, "heldout", 4, 256, 271828)
    assert eval_meta["ids_sha256"] == M15.EVAL_IDS_SHA and int(eval_bytes.sum()) == 4588
    eval_ids = torch.tensor(eval_ids, dtype=torch.long, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = Experts(layer.mlp, li, args.axis_a, args.axis_b, device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
        budget(start, device)
    model.config.use_cache = False
    model.eval()
    initial = bpb(model, wrappers, eval_ids, eval_bytes)
    assert abs(initial["delta"]) <= 1e-5
    print(json.dumps({"experts": experts, "initial_bpb": initial,
                      "runtime": budget(start, device)}), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    router_params = [p for w in wrappers for p in w.router.parameters()]
    router_optimizer = torch.optim.AdamW(router_params, lr=3e-5, weight_decay=0.0)
    factor_optimizer = M65.SparseRowAdam(lr=3e-4)
    steps = []
    for step in range(1, UPDATES + 1):
        router_optimizer.zero_grad(set_to_none=True)
        for wrapper in wrappers:
            wrapper.factors.grad_a.clear()
            wrapper.factors.grad_b.clear()
        microbatches = []
        for micro in range(4):
            if micro < 2:
                row = int(rng.choice(raw_pool))
                offset = int(rng.integers(512 - M15.SEQ + 1))
                window = train_ids[row, offset:offset + M15.SEQ]
                ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
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
        b_norms = [math.sqrt(sum(float(g.double().square().sum())
                                 for g in w.factors.grad_b.values())) for w in wrappers]
        router_norms = [math.sqrt(sum(float(p.grad.double().square().sum())
                                      for p in w.router.parameters() if p.grad is not None))
                        for w in wrappers]
        assert min(b_norms) > 0 and all(math.isfinite(x) for x in b_norms + router_norms)
        if step >= 2:
            assert min(router_norms) > 0
        norm = clipped_global_norm(wrappers)
        for li, wrapper in enumerate(wrappers):
            factor_optimizer.step_bank(wrapper.factors.a, wrapper.factors.grad_a, f"a{li}")
            factor_optimizer.step_bank(wrapper.factors.b, wrapper.factors.grad_b, f"b{li}")
        router_optimizer.step()
        runtime = budget(start, device)
        steps.append({"update": step, "microbatches": microbatches,
                      "min_b_gradient_norm": min(b_norms),
                      "min_router_gradient_norm": min(router_norms),
                      "pre_clip_global_norm": norm,
                      "sparse_state_bytes": factor_optimizer.state_bytes(),
                      "runtime": runtime})
        print(json.dumps({"experts": experts, "update": step,
                          "min_router_gradient_norm": min(router_norms),
                          "elapsed_seconds": runtime["elapsed_seconds"]}), flush=True)
    model.gradient_checkpointing_disable()
    model.eval()
    final_bpb = bpb(model, wrappers, eval_ids, eval_bytes)
    top1 = dev_top1(model, wrappers, dev["rows"], device, experts, start)
    changed = [int((w.factors.b.abs().flatten(1).sum(1) > 0).sum()) for w in wrappers]
    gates = {"raw_bpb": final_bpb["delta"] <= 0.05,
             "prompt_top1": top1["agreement"] >= 0.95,
             "changed_slots": min(changed) >= (64 if experts == 128 else 640),
             "route_oracle": True}
    gates["joint_smoke"] = all(gates.values())
    budget(start, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cpu_router_state = [{key: value.detach().cpu() for key, value in w.router.state_dict().items()}
                        for w in wrappers]
    cpu_router_opt = router_optimizer.state_dict()
    for state in cpu_router_opt["state"].values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().cpu()
    torch.save({"experts": experts, "axis_a": args.axis_a, "axis_b": args.axis_b,
                "updates": UPDATES, "factor_banks": [{"a": w.factors.a, "b": w.factors.b}
                                                for w in wrappers],
                "router_state": cpu_router_state,
                "sparse_factor_optimizer": factor_optimizer.state,
                "router_optimizer": cpu_router_opt,
                "np_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "donor_sha256": M55.MODEL_SHA, "dev_sha256": DEV_SHA}, args.checkpoint)
    checkpoint = {"path": str(args.checkpoint.resolve()),
                  "bytes": args.checkpoint.stat().st_size,
                  "sha256": sha(args.checkpoint)}
    runtime = budget(start, device)
    result = {"experiment": "METH-66-E128-E1280-controlled-smoke",
              "source_sha256": sha(__file__), "donor_sha256": M55.MODEL_SHA,
              "dev_sha256": DEV_SHA, "teacher_sha256": M55.TEACHER_SHA,
              "train_chat_sha256": M55.TRAIN_CHAT_SHA,
              "train_ids_sha256": M15.TRAIN_IDS_SHA,
              "seed": SEED, "experts": experts, "updates": UPDATES,
              "initial_bpb": initial, "final_bpb": final_bpb,
              "dev_top1": top1, "min_changed_b_slots": min(changed),
              "max_changed_b_slots": max(changed),
              "steps": steps, "gates": gates,
              "checkpoint": checkpoint, "runtime": runtime,
              "decision": "eligible_for_longer_continuation" if gates["joint_smoke"]
                          else "stop_at_smoke_gate"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experts": experts, "gates": gates,
                      "top1": top1["agreement"], "bpb_delta": final_bpb["delta"],
                      "checkpoint_bytes": checkpoint["bytes"],
                      "runtime": runtime}), flush=True)


if __name__ == "__main__":
    main()
