#!/usr/bin/env python3
"""METH-71: matched donor-retentive, balanced E128/E1280 smoke."""
import argparse
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
import meth55_product_key_experts as M55
import meth71_balanced_product_key as M71


ROOT = Path(__file__).resolve().parents[3]
TEACHER_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth43_instruct_teacher_chat_result.json"
TEACHER_SHA = "f42f678280a9086a0fc2ec6e57a712e33fdea8f5cb62bdac7f53c9f28a5f1dd7"
TRAIN_CHAT_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth43_instruct_chat_train_manifest.json"
TRAIN_CHAT_SHA = "17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055"
PROMPT_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth71_balanced_dev_manifest.json"
PROMPT_SHA = "4bb5eb221c0754e6769f0faff5f8487d7f20cfa5dac029531ad174cede7840a8"
MODEL_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
RNG_SEED = 717172
UPDATES = 64
MAX_SECONDS = 20 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or peak > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-71 smoke budget: {elapsed:.1f}s, GPU {peak}, RSS {rss}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def set_experts(wrappers, enabled):
    for wrapper in wrappers:
        wrapper.enabled = enabled


def masked_objective(student_logits, teacher_logits, targets, ce_mask, kl_mask, kl_weight):
    student_lp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_lp = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_p = teacher_lp.exp()
    ce_tokens = F.nll_loss(student_lp.reshape(-1, student_lp.shape[-1]),
                           targets.reshape(-1), reduction="none").reshape_as(targets)
    kl_tokens = (teacher_p * (teacher_lp - student_lp)).sum(dim=-1)
    ce = (ce_tokens * ce_mask).sum() / ce_mask.sum()
    kl = (kl_tokens * kl_mask).sum() / kl_mask.sum()
    return ce, kl, ce + kl_weight * kl


def score_chat_top1(model, wrappers, items, device, start):
    matching = total = 0
    per_prompt = []
    model.eval()
    with torch.inference_mode():
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                  device=device).unsqueeze(0)
            set_experts(wrappers, False)
            donor = model(ids, use_cache=False).logits.argmax(dim=-1)
            set_experts(wrappers, True)
            student = model(ids, use_cache=False).logits.argmax(dim=-1)
            same = int((donor == student).sum())
            positions = ids.numel()
            matching += same
            total += positions
            per_prompt.append({"train_row": item["train_row"], "same": same,
                               "positions": positions})
            budget(start, device)
    return {"matching": matching, "positions": total,
            "agreement": matching / total, "per_prompt": per_prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--axis-a", type=int, required=True)
    ap.add_argument("--axis-b", type=int, required=True)
    args = ap.parse_args()
    assert (args.axis_a, args.axis_b) in ((8, 16), (32, 40))
    experts = args.axis_a * args.axis_b
    assert M17.sha(TEACHER_PATH.read_bytes()) == TEACHER_SHA
    assert M17.sha(TRAIN_CHAT_PATH.read_bytes()) == TRAIN_CHAT_SHA
    assert M17.sha(PROMPT_PATH.read_bytes()) == PROMPT_SHA
    teacher_data = json.loads(TEACHER_PATH.read_text(encoding="utf-8"))
    train_chat_data = json.loads(TRAIN_CHAT_PATH.read_text(encoding="utf-8"))
    prompt_data = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    assert teacher_data["model_sha256"] == MODEL_SHA
    assert teacher_data["summary"]["rows"] == len(teacher_data["rows"]) == 256
    assert len(prompt_data["rows"]) == 24
    train_prompts = {row["train_row"]: row for row in train_chat_data["rows"]}
    assert len(train_prompts) == 256
    new_development = {row["train_row"] for row in prompt_data["rows"]}
    assert len(new_development) == 24 and new_development.isdisjoint(train_prompts)
    excluded = set(new_development)
    for name, expected in prompt_data["prior_manifest_sha256"].items():
        path = PROMPT_PATH.parent / name
        assert M17.sha(path.read_bytes()) == expected, name
        old = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(row["train_row"] for row in old["rows"])
    assert len(excluded) == 256 + 7 * 24
    sampled70 = prompt_data["meth70_sampled_raw_rows"]
    assert len(sampled70) == 32
    assert set(sampled70).isdisjoint(new_development)
    excluded.update(sampled70)
    raw_pool = np.asarray([i for i in range(31250) if i not in excluded])
    chat_rows = []
    for teacher_row in teacher_data["rows"]:
        prompt = train_prompts[teacher_row["train_row"]]
        assert prompt["prompt_ids_sha256"] == teacher_row["prompt_ids_sha256"]
        response = teacher_row["continuation_ids"]
        full = prompt["prompt_ids"] + response
        assert full[-129:] == teacher_row["window_ids"]
        assert 1 <= len(response) <= 64
        ce_mask = [0] * (len(prompt["prompt_ids"]) - 1) + [1] * len(response)
        assert len(ce_mask) == len(full) - 1
        chat_rows.append({"train_row": teacher_row["train_row"],
                          "full_ids": full, "assistant_target_mask": ce_mask})
    for row in chat_rows:
        assert len(row["full_ids"]) - 1 == len(row["assistant_target_mask"])
        assert sum(row["assistant_target_mask"]) > 0

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
    torch.cuda.manual_seed_all(M15.SEED)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    rng = np.random.default_rng(RNG_SEED)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = hf_hub_download(M42.MODEL, "model.safetensors",
                             revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source) == MODEL_SHA
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    eval_ids, eval_bytes, eval_meta = M15.M13.C.get_slice(tok, "heldout", 4, 256, 271828)
    assert eval_meta["ids_sha256"] == M15.EVAL_IDS_SHA and int(eval_bytes.sum()) == 4588
    eval_ids = torch.as_tensor(eval_ids, dtype=torch.long, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    cfg = model.config
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size,
            cfg.vocab_size, cfg.num_key_value_heads) == (M15.M13.L, M15.M13.D,
                                                         M15.M13.FF, 151936, 2)
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M71.BalancedProductKeyExperts(layer.mlp, li, args.axis_a, args.axis_b).to(device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    model.config.use_cache = False
    model.eval()
    with torch.no_grad():
        wrappers[0].oracle_checks = True
        wrappers[0].routes(torch.randn(7, M15.M13.D, device=device,
                                        generator=torch.Generator(device=device).manual_seed(RNG_SEED)))
        wrappers[0].oracle_checks = False
    identity = {}
    with torch.no_grad():
        for label, ids in (("raw", eval_ids[:1, :16]),
                           ("chat", torch.as_tensor(prompt_data["rows"][0]["prompt_ids"][:16],
                                                    dtype=torch.long, device=device).unsqueeze(0))):
            set_experts(wrappers, False)
            donor = model(ids, use_cache=False).logits.float()
            set_experts(wrappers, True)
            student = model(ids, use_cache=False).logits.float()
            identity[label] = float((student - donor).abs().max())
    assert max(identity.values()) <= 1e-3, identity
    donor_bpb_initial = M15.bpb(model, eval_ids, eval_bytes, wrappers, False)
    student_bpb_initial = M15.bpb(model, eval_ids, eval_bytes, wrappers, True)
    assert abs(student_bpb_initial - donor_bpb_initial) <= 1e-5
    budget(start, device)
    print(json.dumps({"identity": identity, "donor_bpb": donor_bpb_initial}), flush=True)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    expert_params = [p for wrapper in wrappers for p in (wrapper.a, wrapper.b)]
    router_params = [wrapper.router for wrapper in wrappers]
    optimizer = torch.optim.AdamW([
        {"params": expert_params, "lr": 1e-4},
        {"params": router_params, "lr": 1e-5}], weight_decay=0.0)
    steps = []
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for microbatch in range(4):
            if microbatch < 2:
                row = int(rng.choice(raw_pool))
                offset = int(rng.integers(512 - M15.SEQ + 1))
                window = train_ids[row, offset:offset + M15.SEQ]
                ids = torch.as_tensor(window, dtype=torch.long, device=device).unsqueeze(0)
                inputs, targets = ids, ids[:, 1:]
                ce_mask = kl_mask = torch.ones_like(targets, dtype=torch.float32)
                kl_weight = 2.0
            else:
                row = chat_rows[int(rng.integers(len(chat_rows)))]
                window = torch.as_tensor(row["full_ids"], dtype=torch.long,
                                         device=device).unsqueeze(0)
                inputs, targets = window[:, :-1], window[:, 1:]
                ce_mask = torch.as_tensor(row["assistant_target_mask"],
                                          dtype=torch.float32, device=device).unsqueeze(0)
                kl_mask = torch.ones_like(targets, dtype=torch.float32)
                kl_weight = 4.0
            set_experts(wrappers, False)
            with torch.no_grad():
                teacher_logits = model(inputs, use_cache=False).logits
            set_experts(wrappers, True)
            student_logits = model(inputs, use_cache=False).logits
            if microbatch < 2:
                teacher_logits = teacher_logits[:, :-1]
                student_logits = student_logits[:, :-1]
            ce, kl, objective = masked_objective(student_logits, teacher_logits,
                                                 targets, ce_mask, kl_mask, kl_weight)
            axis_balance = torch.stack([w.balance_loss for w in wrappers]).sum()
            loss = (objective + 0.02 * axis_balance) / 4
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            loss.backward()
            losses.append({"kind": "raw" if microbatch < 2 else "chat",
                           "ce": float(ce.detach()), "kl": float(kl.detach()),
                           "axis_balance": float(axis_balance.detach())})
            del teacher_logits, student_logits, ce, kl, objective, axis_balance, loss
            budget(start, device)
        grads = M15.grad_range(wrappers)
        assert grads["b_min"] > 0 and math.isfinite(grads["b_max"]), (step, grads)
        assert grads["router_min"] > 0 and math.isfinite(grads["router_max"]), (step, grads)
        pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(expert_params + router_params, 1.0))
        assert math.isfinite(pre_clip_norm)
        optimizer.step()
        runtime = budget(start, device)
        steps.append({"update": step, "microbatches": losses, "gradient": grads,
                      "pre_clip_norm": pre_clip_norm, **runtime})
        print(json.dumps({"update": step, "gradient": grads,
                          "elapsed_seconds": runtime["elapsed_seconds"]}), flush=True)

    model.eval()
    model.gradient_checkpointing_disable()
    donor_bpb_terminal = M15.bpb(model, eval_ids, eval_bytes, wrappers, False)
    student_bpb_terminal = M15.bpb(model, eval_ids, eval_bytes, wrappers, True)
    assert abs(donor_bpb_terminal - donor_bpb_initial) <= 1e-5
    for wrapper in wrappers:
        wrapper.oracle_checks = True
        wrapper.collect_load = True
    chat_top1 = score_chat_top1(model, wrappers, prompt_data["rows"], device, start)
    load = M55.load_summary(wrappers)
    for wrapper in wrappers:
        wrapper.oracle_checks = False
        wrapper.collect_load = False
    slots = [wrapper.changed_slots() for wrapper in wrappers]
    delta_bpb = student_bpb_terminal - donor_bpb_terminal
    decision = ("eligible_for_external_semantic_and_task_audit"
                if min(slots) >= (64 if experts == 128 else 640)
                and delta_bpb <= 0.05 and chat_top1["agreement"] >= 0.95
                and min(x["selected_slots"] for x in load) >= (96 if experts == 128 else 640)
                and max(x["max_to_mean"] for x in load) <= (32 if experts == 128 else 64)
                else "stop_balanced_retention_smoke")
    runtime = budget(start, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"expert_state": [
                   {k: v.detach().cpu() for k, v in w.state_dict().items()
                    if k in ("a", "b", "router")} for w in wrappers],
                "optimizer": optimizer.state_dict(),
                "np_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "updates": UPDATES, "axis_a": args.axis_a, "axis_b": args.axis_b,
                "source_sha256": MODEL_SHA, "dev_sha256": PROMPT_SHA,
                "teacher_sha256": TEACHER_SHA,
                "train_ids_sha256": M15.TRAIN_IDS_SHA}, args.checkpoint)
    checkpoint_sha256 = M15.M13.sha256(args.checkpoint)
    runtime = budget(start, device)
    result = {"experiment": "METH-71-balanced-E128-E1280-joint-smoke",
              "source": {"model": M42.MODEL, "revision": M42.REV,
                         "model_sha256": MODEL_SHA,
                         "tokenizer_fingerprint": M15.M13.TOK_FP},
              "data": {"teacher_sha256": TEACHER_SHA,
                       "train_chat_manifest_sha256": TRAIN_CHAT_SHA,
                       "train_ids_sha256": M15.TRAIN_IDS_SHA,
                       "eval_ids_sha256": M15.EVAL_IDS_SHA,
                       "chat_prompt_sha256": PROMPT_SHA},
              "geometry": {"layers": M15.M13.L, "width": M15.M13.D,
                           "experts": experts, "top_k": M15.K, "rank": M15.R,
                           "router_rank": M55.ROUTER_RANK,
                           "product_axes": [args.axis_a, args.axis_b],
                           "init_seed": M15.SEED, "sample_seed": RNG_SEED,
                           "updates": UPDATES, "raw_microbatches": 2,
                           "chat_microbatches": 2, "factor_lr": 1e-4,
                           "router_lr": 1e-5, "axis_balance_weight": 0.02,
                           "raw_kl_weight": 2.0,
                           "chat_full_sequence_kl_weight": 4.0},
              "controls": {"identity_max_abs": identity,
                           "donor_bpb_initial": donor_bpb_initial,
                           "student_bpb_initial": student_bpb_initial,
                           "donor_bpb_terminal": donor_bpb_terminal,
                           "student_bpb_terminal": student_bpb_terminal,
                           "delta_bpb_terminal": delta_bpb,
                           "changed_slots_by_layer": slots,
                           "min_changed_slots": min(slots),
                       "chat_prompt_top1": chat_top1,
                       "route_load_by_layer": load,
                       "min_selected_slots": min(x["selected_slots"] for x in load),
                       "worst_max_to_mean_load": max(x["max_to_mean"] for x in load)},
              "training_steps": steps, "decision": decision,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__,
                          "numpy": np.__version__},
              "checkpoint_path": str(args.checkpoint.resolve()),
              "checkpoint_bytes": args.checkpoint.stat().st_size,
              "checkpoint_sha256": checkpoint_sha256}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "controls": {
        "delta_bpb": delta_bpb, "min_slots": min(slots),
        "chat_top1_agreement": chat_top1["agreement"]},
        "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
