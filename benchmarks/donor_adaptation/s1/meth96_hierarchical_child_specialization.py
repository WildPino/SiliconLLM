#!/usr/bin/env python3
"""METH-96: B-only specialization of function-preserving E1280 children."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth56_product_key_retention as M56
import meth57_product_key_external_audit as M57
import meth95_hierarchical_e1280_parity as M95


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
UPDATES = 128
SEED = 96096
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS_BYTES
            or result["gpu_peak_bytes"] > MAX_GPU_BYTES):
        raise RuntimeError(f"METH-96 resource stop: {result}")
    return result


def load_parent(model, state, device):
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(state[li][key].to(device))
            wrappers.append(wrapper)
    return wrappers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    args = ap.parse_args()
    assert M15.M13.sha256(M95.CHECKPOINT_REPORT) == M57.TRAINING_SHA
    assert M15.M13.sha256(DIR / "meth95_hierarchical_e1280_parity_result.json") == (
        "4e83e68a937293f2d99af68c10b773dec2463d4f04ba40c725f7c01dfb7d0531")
    for path, digest in ((M56.TRAIN_CHAT_PATH, M56.TRAIN_CHAT_SHA),
                         (M56.TEACHER_PATH, M56.TEACHER_SHA),
                         (M15.TRAIN_PATH, M15.TRAIN_FILE_SHA),
                         (M56.DEV_PATH, M56.DEV_SHA),
                         (M56.OLD_DEV_PATH, M56.OLD_DEV_SHA),
                         (M44.PROMPT_PATH, M44.PROMPT_SHA),
                         (M56.M47_DEV_PATH, M56.M47_DEV_SHA)):
        assert M15.M13.sha256(path) == digest, path
    training = json.loads(M95.CHECKPOINT_REPORT.read_text(encoding="utf-8"))
    parent = training["checkpoints"]["512"]
    assert parent["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(parent["path"]) == M57.CHECKPOINT_SHA
    state = torch.load(parent["path"], map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M57.MODEL_SHA
    prompts = {row["train_row"]: row for row in json.loads(
        Path(M56.TRAIN_CHAT_PATH).read_text(encoding="utf-8"))["rows"]}
    teacher_rows = json.loads(Path(M56.TEACHER_PATH).read_text(encoding="utf-8"))["rows"]
    assert len(prompts) == len(teacher_rows) == 256
    chat = []
    for row in teacher_rows:
        prompt = prompts[row["train_row"]]
        assert prompt["prompt_ids_sha256"] == row["prompt_ids_sha256"]
        ids = prompt["prompt_ids"] + row["continuation_ids"]
        mask = [0] * (len(prompt["prompt_ids"]) - 1) + [1] * len(row["continuation_ids"])
        assert len(mask) == len(ids) - 1
        chat.append((row["train_row"], ids, mask))
    excluded = set(prompts)
    for path in (M56.DEV_PATH, M56.OLD_DEV_PATH, M44.PROMPT_PATH, M56.M47_DEV_PATH):
        excluded.update(row["train_row"] for row in json.loads(
            Path(path).read_text(encoding="utf-8"))["rows"])
    assert len(excluded) == 352
    raw_pool = np.asarray([i for i in range(31250) if i not in excluded])
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        raw_ids = archive["ids"].copy()
    assert raw_ids.shape == (31250, 512) and raw_ids.dtype == np.int32
    assert M15.sha_bytes(raw_ids.tobytes()) == M15.TRAIN_IDS_SHA

    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(SEED)
    start = time.monotonic()
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    from transformers import AutoModelForCausalLM

    def make_model():
        model = AutoModelForCausalLM.from_pretrained(
            M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        model.config.use_cache = False
        return model

    teacher = make_model()
    load_parent(teacher, state["expert_state"], device)
    teacher.eval()
    student = make_model()
    parent_wrappers = load_parent(student, state["expert_state"], device)
    parent_b = [w.b.detach().cpu().clone() for w in parent_wrappers]
    children = []
    for li, layer in enumerate(student.model.layers):
        wrapper = M95.HierarchicalExperts(parent_wrappers[li], li).to(device)
        wrapper.b.requires_grad_(True)
        wrapper.router.requires_grad_(False)
        wrapper.capture = True
        layer.mlp = wrapper
        children.append(wrapper)
    del parent_wrappers, state
    trainable = [w.b for w in children]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.0)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    student.train()
    budget(start, device)

    draws = []
    losses = []
    for update in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        record = {"update": update}
        for kind in ("raw", "chat"):
            if kind == "raw":
                row = int(rng.choice(raw_pool))
                offset = int(rng.integers(512 - M15.SEQ + 1))
                ids_np = raw_ids[row, offset:offset + M15.SEQ]
                ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)[None]
                ce_mask = torch.ones((1, M15.SEQ - 1), device=device)
                draws.append({"update": update, "kind": kind,
                              "row": row, "offset": offset})
                kl_weight = 2.0
            else:
                choice = int(rng.integers(len(chat)))
                row, ids_list, mask = chat[choice]
                ids = torch.as_tensor(ids_list, dtype=torch.long, device=device)[None]
                ce_mask = torch.as_tensor(mask, dtype=torch.float32, device=device)[None]
                draws.append({"update": update, "kind": kind, "train_row": row})
                kl_weight = 4.0
            inputs = ids[:, :-1]
            targets = ids[:, 1:]
            kl_mask = torch.ones_like(targets, dtype=torch.float32)
            with torch.no_grad():
                teacher_logits = teacher(inputs, use_cache=False).logits
            student_logits = student(inputs, use_cache=False).logits
            ce, kl, objective = M44.masked_objective(
                student_logits, teacher_logits, targets, ce_mask,
                kl_mask, kl_weight)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(f"nonfinite loss at update {update}")
            (objective / 2).backward()
            record[kind] = {"ce": float(ce.detach()), "kl": float(kl.detach()),
                            "objective": float(objective.detach())}
            del ids, inputs, targets, ce_mask, kl_mask, teacher_logits
            del student_logits, ce, kl, objective
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not np.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient at update {update}")
        optimizer.step()
        record["clip_pre_norm"] = norm
        losses.append(record)
        if update in (1, 16, 32, 64, 96, 128):
            print(json.dumps({"update": update, "loss": record,
                              "budget": budget(start, device)}), flush=True)
        budget(start, device)

    changed = []
    coverage = []
    with torch.no_grad():
        for w, copied_parent in zip(children, parent_b):
            current = w.b.detach().view(M15.E, M95.CHILDREN, M15.M13.D, M15.R)
            delta = (current - copied_parent.to(device)[:, None]).abs().amax(dim=(-1, -2))
            changed.append(int((delta > 1e-7).sum()))
            coverage.append(int((w.visited > 0).sum()))
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_state = {"expert_state": [
        {key: getattr(w, key).detach().cpu().clone()
         for key in ("a", "b", "router", "child_projection", "child_keys")}
        for w in children],
        "updates": UPDATES, "source_sha256": M57.MODEL_SHA,
        "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
        "train_ids_sha256": M15.TRAIN_IDS_SHA,
        "train_chat_sha256": M56.TRAIN_CHAT_SHA,
        "seed": SEED, "trainable": "child_B_only"}
    torch.save(output_state, args.checkpoint)
    report = {"experiment": "METH-96-hierarchical-E1280-child-specialization",
              "source_sha256": M57.MODEL_SHA,
              "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
              "checkpoint": {"path": str(args.checkpoint.resolve()),
                             "sha256": M15.M13.sha256(args.checkpoint),
                             "bytes": args.checkpoint.stat().st_size},
              "seed": SEED, "updates": UPDATES, "draws": draws,
              "losses": losses, "changed_child_slots_by_layer": changed,
              "visited_child_slots_by_layer": coverage,
              "distinct_slot_gate_pass": min(changed) >= 640,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__},
              "decision": "checkpoint_ready_for_fresh_development" if min(changed) >= 640
              else "stop_distinct_slot_gate"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint_sha256": report["checkpoint"]["sha256"],
                      "min_changed": min(changed), "max_changed": max(changed),
                      "min_visited": min(coverage), "max_visited": max(coverage),
                      "first_loss": losses[0], "last_loss": losses[-1],
                      "runtime": report["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
