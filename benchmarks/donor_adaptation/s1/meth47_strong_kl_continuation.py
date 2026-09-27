#!/usr/bin/env python3
"""Resume METH-44 with a fixed stronger donor KL through update 512."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth46_route_utility_diagnostic as M46


ROOT = Path(__file__).resolve().parents[3]
RESUME = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth44_instruct_e128_smoke.pt"
RESUME_SHA = "4fade15804f5e2297ff60cafacf870e5ae11926ba886786a524110dea94f16da"
TRAIN_CHAT_PATH = M44.TRAIN_CHAT_PATH
TRAIN_CHAT_SHA = M44.TRAIN_CHAT_SHA
TEACHER_PATH = M44.TEACHER_PATH
TEACHER_SHA = M44.TEACHER_SHA
OLD_DEV_PATH = M44.PROMPT_PATH
OLD_DEV_SHA = M44.PROMPT_SHA
DEV_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth47_retention_dev_manifest.json"
DEV_SHA = "0195852ce7a46af5a2d9442b38e687a19393a943168b61a8f8cdb0c66ec78a60"
MODEL_SHA = M44.MODEL_SHA
FINAL_STEP = 512
CHECK_STEPS = (256, 512)
MAX_SECONDS = 20 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or peak > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-47 training budget: {elapsed:.1f}s, GPU {peak}, RSS {rss}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


@torch.inference_mode()
def dev_top1(model, wrappers, rows, device, start):
    model.eval()
    matching = positions = 0
    for row in rows:
        ids = torch.as_tensor(row["prompt_ids"], dtype=torch.long,
                              device=device).unsqueeze(0)
        M44.set_experts(wrappers, False)
        donor = model(ids, use_cache=False).logits.argmax(dim=-1)
        M44.set_experts(wrappers, True)
        student = model(ids, use_cache=False).logits.argmax(dim=-1)
        matching += int((donor == student).sum())
        positions += ids.numel()
        budget(start, device)
    return {"matching": matching, "positions": positions,
            "agreement": matching / positions}


def save_state(path, step, wrappers, optimizer, rng, device):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"expert_state": [
                   {key: value.detach().cpu()
                    for key, value in wrapper.state_dict().items()
                    if key in ("a", "b", "router")}
                   for wrapper in wrappers],
                "optimizer": optimizer.state_dict(),
                "np_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "updates": step, "source_sha256": MODEL_SHA,
                "teacher_sha256": TEACHER_SHA,
                "train_ids_sha256": M15.TRAIN_IDS_SHA}, path)
    return {"path": str(path.resolve()), "sha256": M15.M13.sha256(path),
            "bytes": path.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    args = ap.parse_args()
    assert M15.M13.sha256(RESUME) == RESUME_SHA
    for path, digest in ((TRAIN_CHAT_PATH, TRAIN_CHAT_SHA),
                         (TEACHER_PATH, TEACHER_SHA),
                         (OLD_DEV_PATH, OLD_DEV_SHA),
                         (DEV_PATH, DEV_SHA)):
        assert M17.sha(path.read_bytes()) == digest, path
    teacher_data = json.loads(TEACHER_PATH.read_text(encoding="utf-8"))
    train_chat = json.loads(TRAIN_CHAT_PATH.read_text(encoding="utf-8"))
    old_dev = json.loads(OLD_DEV_PATH.read_text(encoding="utf-8"))
    dev = json.loads(DEV_PATH.read_text(encoding="utf-8"))
    prompts = {row["train_row"]: row for row in train_chat["rows"]}
    assert len(prompts) == len(teacher_data["rows"]) == 256
    assert len(dev["rows"]) == len(old_dev["rows"]) == 24
    excluded = {row["train_row"] for row in dev["rows"] + old_dev["rows"]}
    assert len(excluded) == 48 and not excluded.intersection(prompts)
    raw_pool = np.asarray([i for i in range(31250) if i not in excluded])
    chat_rows = []
    for saved in teacher_data["rows"]:
        prompt = prompts[saved["train_row"]]
        assert prompt["prompt_ids_sha256"] == saved["prompt_ids_sha256"]
        full = prompt["prompt_ids"] + saved["continuation_ids"]
        assert full[-129:] == saved["window_ids"]
        mask = [0] * (len(prompt["prompt_ids"]) - 1) + [1] * len(saved["continuation_ids"])
        assert len(mask) == len(full) - 1 and sum(mask) >= 1
        chat_rows.append({"full_ids": full, "assistant_target_mask": mask})
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        train_ids = archive["ids"].copy()
    assert train_ids.shape == (31250, 512) and train_ids.dtype == np.int32
    assert M15.sha_bytes(train_ids.tobytes()) == M15.TRAIN_IDS_SHA

    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    torch.manual_seed(M15.SEED)
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
    assert M15.M13.sha256(source) == MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    eval_ids, eval_bytes, eval_meta = M15.M13.C.get_slice(tokenizer, "heldout", 4, 256, 271828)
    assert eval_meta["ids_sha256"] == M15.EVAL_IDS_SHA and int(eval_bytes.sum()) == 4588
    eval_ids = torch.as_tensor(eval_ids, dtype=torch.long, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    expert_params = [p for wrapper in wrappers for p in (wrapper.a, wrapper.b)]
    router_params = [wrapper.router for wrapper in wrappers]
    optimizer = torch.optim.AdamW([
        {"params": expert_params, "lr": 3e-4},
        {"params": router_params, "lr": 3e-5}], weight_decay=0.0)
    state = torch.load(RESUME, map_location="cpu", weights_only=False)
    assert state["updates"] == 16
    assert state["source_sha256"] == MODEL_SHA
    assert state["teacher_sha256"] == TEACHER_SHA
    assert state["train_ids_sha256"] == M15.TRAIN_IDS_SHA
    assert len(state["expert_state"]) == M15.M13.L
    for wrapper, values in zip(wrappers, state["expert_state"]):
        for key in ("a", "b", "router"):
            getattr(wrapper, key).data.copy_(values[key].to(device))
    optimizer.load_state_dict(state["optimizer"])
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 3e-5]
    optimizer.param_groups[0]["lr"] = 1e-4
    optimizer.param_groups[1]["lr"] = 1e-5
    rng = np.random.default_rng()
    rng.bit_generator.state = state["np_rng_state"]
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    model.config.use_cache = False
    model.eval()
    donor_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, False)
    student_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, True)
    assert abs(donor_bpb - 0.9712552075318387) <= 1e-5
    assert abs(student_bpb - 0.9694120093360877) <= 1e-5
    initial_top1 = dev_top1(model, wrappers, dev["rows"], device, start)
    assert initial_top1["agreement"] >= 0.95, initial_top1
    budget(start, device)
    print(json.dumps({"resumed_update": 16, "raw_delta_bpb": student_bpb-donor_bpb,
                      "dev_top1": initial_top1}), flush=True)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    evaluations = [{"step": 16, "donor_bpb": donor_bpb,
                    "student_bpb": student_bpb,
                    "delta_bpb": student_bpb - donor_bpb,
                    "dev_top1": initial_top1}]
    checkpoints = {}
    stop_reason = None
    last_step = 16
    for step in range(17, FINAL_STEP + 1):
        optimizer.zero_grad(set_to_none=True)
        microbatch_losses = []
        for microbatch in range(4):
            if microbatch < 2:
                row_index = int(rng.choice(raw_pool))
                offset = int(rng.integers(512 - M15.SEQ + 1))
                ids = torch.as_tensor(train_ids[row_index, offset:offset + M15.SEQ],
                                      dtype=torch.long, device=device).unsqueeze(0)
                inputs, targets = ids, ids[:, 1:]
                ce_mask = kl_mask = torch.ones_like(targets, dtype=torch.float32)
                kl_weight = 2.0
            else:
                row = chat_rows[int(rng.integers(len(chat_rows)))]
                ids = torch.as_tensor(row["full_ids"], dtype=torch.long,
                                      device=device).unsqueeze(0)
                inputs, targets = ids[:, :-1], ids[:, 1:]
                ce_mask = torch.as_tensor(row["assistant_target_mask"],
                                          dtype=torch.float32, device=device).unsqueeze(0)
                kl_mask = torch.ones_like(targets, dtype=torch.float32)
                kl_weight = 4.0
            M44.set_experts(wrappers, False)
            with torch.no_grad():
                teacher_logits = model(inputs, use_cache=False).logits
            M44.set_experts(wrappers, True)
            student_logits = model(inputs, use_cache=False).logits
            if microbatch < 2:
                teacher_logits = teacher_logits[:, :-1]
                student_logits = student_logits[:, :-1]
            ce, kl, objective = M44.masked_objective(
                student_logits, teacher_logits, targets, ce_mask, kl_mask, kl_weight)
            loss = objective / 4
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            loss.backward()
            microbatch_losses.append({"kind": "raw" if microbatch < 2 else "chat",
                                      "ce": float(ce.detach()), "kl": float(kl.detach())})
            del teacher_logits, student_logits, ce, kl, objective, loss
            budget(start, device)
        grads = M15.grad_range(wrappers)
        assert grads["b_min"] > 0 and grads["router_min"] > 0, (step, grads)
        norm = float(torch.nn.utils.clip_grad_norm_(expert_params + router_params, 1.0))
        assert math.isfinite(norm)
        optimizer.step()
        last_step = step
        runtime = budget(start, device)
        if step % 32 == 0:
            print(json.dumps({"update": step, "losses": microbatch_losses,
                              "grad": grads, "clip_pre_norm": norm,
                              "elapsed_seconds": runtime["elapsed_seconds"]}), flush=True)
        if step in CHECK_STEPS:
            model.eval()
            student_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, True)
            top1 = dev_top1(model, wrappers, dev["rows"], device, start)
            evaluation = {"step": step, "student_bpb": student_bpb,
                          "delta_bpb": student_bpb - donor_bpb,
                          "dev_top1": top1}
            evaluations.append(evaluation)
            path = args.checkpoint_dir / f"meth47_update{step}.pt"
            checkpoints[str(step)] = save_state(path, step, wrappers,
                                                 optimizer, rng, device)
            print(json.dumps({"checkpoint": step, "evaluation": evaluation,
                              "checkpoint_sha256": checkpoints[str(step)]["sha256"]}),
                  flush=True)
            if evaluation["delta_bpb"] > 0.05:
                stop_reason = "interim_raw_bpb_failure"
            elif top1["agreement"] < 0.95:
                stop_reason = "interim_chat_top1_failure"
            if stop_reason:
                break
            model.train()
    model.eval()
    model.gradient_checkpointing_disable()
    end_donor_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, False)
    assert abs(end_donor_bpb - donor_bpb) <= 1e-5
    route = None
    slots = None
    if last_step == FINAL_STEP and stop_reason is None:
        route = M46.route_sensitivity(model, wrappers, eval_ids,
                                      eval_bytes, device, time.monotonic())
        slots = [wrapper.changed_slots() for wrapper in wrappers]
        if route["permuted_minus_intact_bpb"] < 0.002:
            stop_reason = "terminal_route_utility_failure"
        elif min(slots) < 64:
            stop_reason = "terminal_expert_slot_failure"
    runtime = budget(start, device)
    result = {"experiment": "METH-47-strong-KL-Instruct-continuation",
              "source_sha256": MODEL_SHA, "resume_sha256": RESUME_SHA,
              "teacher_sha256": TEACHER_SHA,
              "train_chat_manifest_sha256": TRAIN_CHAT_SHA,
              "previous_development_manifest_sha256": OLD_DEV_SHA,
              "development_manifest_sha256": DEV_SHA,
              "fresh_external_manifest_sha256":
              "15a68db9a43a8066e70b1b261f819cab3d6779d8b52633bcce7c725d8d54acfb",
              "raw_train_ids_sha256": M15.TRAIN_IDS_SHA,
              "raw_eval_ids_sha256": M15.EVAL_IDS_SHA,
              "factor_lr": 1e-4, "router_lr": 1e-5,
              "raw_kl_weight": 2.0, "full_chat_kl_weight": 4.0,
              "target_update": FINAL_STEP, "last_applied_update": last_step,
              "interim_evaluations": evaluations,
              "checkpoints": checkpoints,
              "terminal_route_sensitivity": route,
              "terminal_changed_slots_by_layer": slots,
              "stop_reason": stop_reason,
              "decision": ("eligible_for_frozen_external_evaluation"
                           if last_step == FINAL_STEP and stop_reason is None
                           else "stop_instruct_continuation"),
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__,
                          "numpy": np.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "last_update": last_step, "runtime": result["runtime"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
