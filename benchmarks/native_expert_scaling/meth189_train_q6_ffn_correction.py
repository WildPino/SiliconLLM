#!/usr/bin/env python3
"""Distill BF16+E1280 into stored-Q6 FFN rank-64 residual factors."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn as nn
import torch.nn.functional as F

import meth187_q6_core_e1280_development as P

sys.path.insert(0, str(P.ROOT / "benchmarks/donor_adaptation/s1"))
import meth136_matched_sparse_train as D
import meth88_quant_aware_b_adaptation as O


RANK = 64
SEED = 189189
UPDATES = 256
LR = 3e-4
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000
PAYLOAD = 53_084_160


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS
            or result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-189 resource stop: {result}")
    return result


class LowRankCorrection(nn.Module):
    def __init__(self, base, seed):
        super().__init__()
        assert isinstance(base, nn.Linear) and base.bias is None
        self.base = base
        self.width_in = base.in_features
        self.width_out = base.out_features
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.a = nn.Parameter(torch.randn((RANK, self.width_in), generator=generator,
                                          dtype=torch.float32) / math.sqrt(self.width_in))
        self.b = nn.Parameter(torch.zeros((self.width_out, RANK), dtype=torch.float32))

    def forward(self, x):
        correction = F.linear(F.linear(x.float(), self.a), self.b)
        return self.base(x) + correction.to(x.dtype)


def make_model(device, parent_state, child_state, core=False):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        P.M42.MODEL, revision=P.M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.config.use_cache = False
    params = dict(model.named_parameters())
    assert len(params) == 290
    if core:
        with safe_open(str(P.CORE), framework="pt", device="cpu") as archive:
            assert archive.metadata()["format"] == "QWEN25_INSTRUCT_R8H_GROUP64_Q6FFN_BF16ATTN_V1"
            assert archive.metadata()["model_sha256"] == P.M57.MODEL_SHA
            with torch.no_grad():
                for name, param in params.items():
                    organ = P.M24.classify(name, tuple(param.shape))
                    if organ == "ffn":
                        param.copy_(P.M186.reconstruct(
                            archive.get_tensor(name + ".q6").to(device),
                            archive.get_tensor(name + ".scale").to(device)))
                    elif organ == "tied_head":
                        codes = archive.get_tensor(name + ".q").to(device)
                        scales = archive.get_tensor(name + ".scale").to(device)
                        param.copy_((codes.float() * scales[:, None]).bfloat16())
                    else:
                        assert torch.equal(archive.get_tensor(name).to(param.dtype),
                                           param.detach().cpu())
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            base = P.M55.ProductKeyExperts(layer.mlp, li).to(device)
            for key in ("a", "b", "router"):
                getattr(base, key).copy_(parent_state["expert_state"][li][key].to(device))
            wrapper = P.M95.HierarchicalExperts(base, li).to(device)
            saved = child_state["expert_state"][li]
            for key in ("a", "b", "router", "child_projection", "child_keys"):
                getattr(wrapper, key).copy_(saved[key].to(device))
            raw_b = wrapper.b.view(P.M15.E, P.M95.CHILDREN, P.M15.M13.D, P.M15.R)
            wrapper.b.copy_((base.b[:, None] + raw_b - raw_b.mean(dim=1, keepdim=True))
                            .reshape_as(wrapper.b))
            layer.mlp = wrapper
            wrappers.append(wrapper)
    corrections = {}
    if core:
        for li, wrapper in enumerate(wrappers):
            for pi, key in enumerate(("gate_proj", "up_proj", "down_proj")):
                module = LowRankCorrection(getattr(wrapper.base, key), SEED + 3 * li + pi)
                setattr(wrapper.base, key, module.to(device))
                corrections[f"layers.{li}.{key}"] = module
    return model, wrappers, corrections


def write_progress(path, update, records, start, device):
    path.write_text(json.dumps({"experiment": "METH-189-progress",
                                "completed_update": update, "records": records,
                                "runtime": budget(start, device)}, indent=2) + "\n",
                    encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists() and not args.checkpoint.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    assert P.digest(P.CORE) == json.loads(P.EXPORT.read_text(encoding="utf-8"))["sha256"]
    assert P.digest(P.M122.SPECIALIZED) == P.M122.SPECIALIZED_SHA
    assert P.digest(D.M107.LONG_TEACHER) == D.M107.LONG_TEACHER_SHA
    for path, expected in ((D.M56.TRAIN_CHAT_PATH, D.M56.TRAIN_CHAT_SHA),
                           (D.M15.TRAIN_PATH, D.M15.TRAIN_FILE_SHA),
                           (D.M56.DEV_PATH, D.M56.DEV_SHA),
                           (D.M56.OLD_DEV_PATH, D.M56.OLD_DEV_SHA),
                           (D.M44.PROMPT_PATH, D.M44.PROMPT_SHA),
                           (D.M56.M47_DEV_PATH, D.M56.M47_DEV_SHA)):
        assert P.digest(path) == expected, path
    parent = json.loads(P.M122.TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert P.digest(parent["path"]) == P.M57.CHECKPOINT_SHA
    source = Path(hf_hub_download(P.M42.MODEL, "model.safetensors",
                                  revision=P.M42.REV, local_files_only=True))
    assert P.digest(source) == P.M57.MODEL_SHA
    raw_ids, chat, draws = D.training_data()
    assert len(draws) == UPDATES == len(chat)

    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parent_state = torch.load(parent["path"], map_location="cpu", weights_only=False)
    child_state = torch.load(P.M122.SPECIALIZED, map_location="cpu", weights_only=False)
    teacher, teacher_wrappers, _ = make_model(device, parent_state, child_state)
    teacher.eval()
    student, student_wrappers, corrections = make_model(
        device, parent_state, child_state, core=True)
    assert len(corrections) == 72
    del parent_state, child_state
    trainable = [parameter for module in corrections.values()
                 for parameter in (module.a, module.b)]
    assert sum(parameter.numel() for parameter in trainable) * 2 == PAYLOAD
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    student.train()
    budget(start, device)

    records = []
    train_started = time.monotonic()
    for update, draw in enumerate(draws, 1):
        assert draw["update"] == update
        optimizer.zero_grad(set_to_none=True)
        record = {"update": update, "draw": draw}
        for kind in ("raw", "chat"):
            if kind == "raw":
                ids_np = raw_ids[draw["raw_row"],
                                 draw["raw_offset"]:draw["raw_offset"] + D.M15.SEQ]
                ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)[None]
                mask = torch.ones((1, ids.shape[1] - 1), device=device)
            else:
                row, ids_list, saved_mask = chat[draw["chat_index"]]
                assert row == draw["chat_train_row"]
                ids = torch.as_tensor(ids_list, dtype=torch.long, device=device)[None]
                mask = torch.as_tensor(saved_mask, dtype=torch.float32, device=device)[None]
            assert mask.shape == (1, ids.shape[1] - 1)
            inputs = ids[:, :-1]
            with torch.no_grad():
                teacher_logits = teacher(inputs, use_cache=False).logits
            student_logits = student(inputs, use_cache=False).logits
            loss, kl, margin, confident = O.objective(student_logits,
                                                       teacher_logits, mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at {update} {kind}")
            (loss / 2).backward()
            record[kind] = {"objective": float(loss.detach()), "kl": float(kl),
                            "margin": float(margin), "confident": confident}
            del ids, mask, inputs, teacher_logits, student_logits, loss, kl, margin
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not np.isfinite(grad_norm):
            raise FloatingPointError(f"nonfinite gradient at {update}")
        optimizer.step()
        record["clip_pre_norm"] = grad_norm
        records.append(record)
        if update in (1, 16, 64, 128, 192, 256):
            write_progress(partial, update, records, start, device)
            print(json.dumps({"update": update, "raw": record["raw"],
                              "chat": record["chat"], "runtime": budget(start, device)}),
                  flush=True)
        current = budget(start, device)
        if update == 16:
            projected = current["seconds"] + (
                time.monotonic() - train_started) / 16 * (UPDATES - update)
            if projected > MAX_SECONDS:
                raise RuntimeError(f"METH-189 projected runtime stop: {projected:.1f}s")

    tensors = {}
    for name, module in corrections.items():
        tensors[name + ".a"] = module.a.detach().cpu().bfloat16().contiguous()
        tensors[name + ".b"] = module.b.detach().cpu().bfloat16().contiguous()
    assert len(tensors) == 144
    assert sum(t.numel() * t.element_size() for t in tensors.values()) == PAYLOAD
    save_file(tensors, str(args.checkpoint), metadata={
        "format": "METH189_Q6_FFN_RANK64_BF16_V1",
        "source_sha256": P.M57.MODEL_SHA,
        "core_sha256": P.digest(P.CORE),
        "parent_sha256": P.M57.CHECKPOINT_SHA,
        "child_sha256": P.M122.SPECIALIZED_SHA,
        "seed": str(SEED), "updates": str(UPDATES)})
    assert args.checkpoint.stat().st_size < MAX_DISK
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as archive:
        assert sorted(archive.keys()) == sorted(tensors)
        for name, tensor in tensors.items():
            assert torch.equal(archive.get_tensor(name), tensor)
    result = {"experiment": "METH-189-Q6-FFN-rank64-correction-train",
              "source_sha256": P.M57.MODEL_SHA,
              "core_sha256": P.digest(P.CORE),
              "parent_sha256": P.M57.CHECKPOINT_SHA,
              "child_sha256": P.M122.SPECIALIZED_SHA,
              "train_raw_sha256": D.M15.TRAIN_FILE_SHA,
              "train_chat_sha256": D.M56.TRAIN_CHAT_SHA,
              "long_teacher_sha256": D.M107.LONG_TEACHER_SHA,
              "seed": SEED, "updates": UPDATES, "rank": RANK,
              "lr": LR, "records": records,
              "checkpoint": {"path": str(args.checkpoint.resolve()),
                             "bytes": args.checkpoint.stat().st_size,
                             "sha256": P.digest(args.checkpoint),
                             "payload_bytes": PAYLOAD, "readback_exact": True},
              "ideal_addressed_bytes_per_token": 481_534_976 + PAYLOAD,
              "runtime": {**budget(start, device),
                          "gpu": torch.cuda.get_device_name(device)},
              "decision": "candidate_requires_viewed_source_development"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size + args.checkpoint.stat().st_size < MAX_DISK
    print(json.dumps({"checkpoint": result["checkpoint"],
                      "last_record": records[-1], "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
