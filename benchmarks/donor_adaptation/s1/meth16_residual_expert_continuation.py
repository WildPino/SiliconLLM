#!/usr/bin/env python3
"""METH-16: resume exact-donor residual experts through a longer GPU run."""
import argparse
from collections import Counter
import json
import math
import os
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15


RESUME_SHA = "cff4d2cbe4851d736f76d7f0838c42752a9d7bceec83054c34239d413d9a727e"
EVAL_IDS_SHA = "48bb4258c69a93dcc3e421444213dfa0eca38856dc9bdd24edf96c408310d1aa"
PERM_SEED = 1616
FINAL_STEP = 1024
CHECK_STEPS = (256, 512, 768, FINAL_STEP)
MAX_SECONDS = 35 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-16 time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-16 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-16 allocated GPU stop: {peak}")
    return elapsed, rss, peak


def resume_state(path, wrappers, opt, rng, device):
    assert M15.M13.sha256(path) == RESUME_SHA
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["updates"] == M15.STEPS
    assert state["source_sha256"] == M15.M13.MODEL_SHA
    assert state["train_ids_sha256"] == M15.TRAIN_IDS_SHA
    assert len(state["expert_state"]) == M15.M13.L
    for wrapper, values in zip(wrappers, state["expert_state"]):
        wrapper.a.data.copy_(values["a"].to(device))
        wrapper.b.data.copy_(values["b"].to(device))
        wrapper.router.data.copy_(values["router"].to(device))
    opt.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["np_rng_state"]
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    return state


def save_state(path, step, wrappers, opt, rng, device):
    torch.save({"expert_state": [
                   {k: v.detach().cpu() for k, v in w.state_dict().items()
                    if k in ("a", "b", "router")} for w in wrappers],
                "optimizer": opt.state_dict(),
                "np_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "updates": step,
                "source_sha256": M15.M13.MODEL_SHA,
                "train_ids_sha256": M15.TRAIN_IDS_SHA}, path)
    return {"path": os.path.abspath(path),
            "sha256": M15.M13.sha256(path),
            "bytes": os.path.getsize(path)}


def repeats_three(tokens, ngram=8):
    if len(tokens) < ngram:
        return False
    counts = Counter(tuple(tokens[i:i+ngram])
                     for i in range(len(tokens)-ngram+1))
    return max(counts.values(), default=0) >= 3


@torch.no_grad()
def generation(model, ids, wrappers, device, start):
    model.eval()
    model.config.use_cache = True
    results = []
    counts = {}
    for enabled in (False, True):
        for w in wrappers:
            w.enabled = enabled
        label = "student" if enabled else "donor"
        loop_count = 0
        for i in range(16):
            prefix = ids[i:i+1, :128]
            out = model.generate(input_ids=prefix, max_new_tokens=128,
                                 do_sample=False, pad_token_id=model.config.eos_token_id,
                                 use_cache=True)
            continuation = out[0, 128:].tolist()
            loop = repeats_three(continuation)
            loop_count += int(loop)
            results.append({"arm": label, "window": i,
                            "prompt_ids_sha256": M15.sha_bytes(prefix.cpu().numpy().tobytes()),
                            "continuation_ids": continuation,
                            "generated_tokens": len(continuation),
                            "repeated_8gram_3x": loop})
            budget(start, device)
        counts[label] = loop_count
        print(f"generation {label}: loops={loop_count}/16", flush=True)
    return results, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(6)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(True)  # common.py imported by M15 disables it globally.
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1, matches
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    source_path = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                                  revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source_path) == M15.M13.MODEL_SHA
    tok = AutoTokenizer.from_pretrained(M15.M13.MODEL, revision=M15.M13.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        train_ids = archive["ids"].copy()
    assert train_ids.shape == (31250, 512)
    assert M15.sha_bytes(train_ids.tobytes()) == M15.TRAIN_IDS_SHA
    eval_ids, eval_bytes, eval_meta = M15.M13.C.get_slice(tok, "heldout", 16, 512, 314159)
    assert eval_meta["ids_sha256"] == EVAL_IDS_SHA and int(eval_bytes.sum()) == 33374
    eval_ids = torch.as_tensor(eval_ids, dtype=torch.long, device=device)
    rng = np.random.default_rng()

    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    params_expert = [p for w in wrappers for p in (w.a, w.b)]
    params_router = [w.router for w in wrappers]
    opt = torch.optim.AdamW([
        {"params": params_expert, "lr": 1e-3},
        {"params": params_router, "lr": 1e-4},
    ], weight_decay=0.0)
    resume_state(args.resume, wrappers, opt, rng, device)
    model.config.use_cache = False
    model.eval()
    donor_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, False)
    student_start_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, True)
    print(f"resume update=16 donor={donor_bpb:.8f} student={student_start_bpb:.8f}",
          flush=True)
    budget(start, device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    steps = []
    evaluations = [{"step": 16, "donor_bpb": donor_bpb,
                    "student_bpb": student_start_bpb,
                    "delta_bpb": student_start_bpb - donor_bpb}]
    checkpoints = {}
    stop_reason = None
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_step = 16
    for step in range(17, FINAL_STEP + 1):
        opt.zero_grad(set_to_none=True)
        ce_sum = kl_sum = 0.0
        for _ in range(M15.ACCUM):
            row = int(rng.integers(train_ids.shape[0]))
            offset = int(rng.integers(0, 512 - M15.SEQ + 1))
            ids = torch.as_tensor(train_ids[row, offset:offset + M15.SEQ],
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
            loss = (ce + 0.1 * kl) / M15.ACCUM
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            loss.backward()
            ce_sum += float(ce.detach())
            kl_sum += float(kl.detach())
            del teacher, teacher_lp, teacher_p, student, student_lp, loss
            budget(start, device)
        grad = M15.grad_range(wrappers)
        assert grad["b_min"] > 0 and grad["router_min"] > 0, (step, grad)
        norm = float(torch.nn.utils.clip_grad_norm_(params_expert + params_router, 1.0))
        assert math.isfinite(norm)
        opt.step()
        last_step = step
        elapsed, rss, peak = budget(start, device)
        steps.append({"step": step, "ce": ce_sum / M15.ACCUM,
                      "kl": kl_sum / M15.ACCUM, "clip_pre_norm": norm,
                      "b_min_grad": grad["b_min"],
                      "router_min_grad": grad["router_min"],
                      "elapsed_seconds": elapsed})
        if step % 32 == 0:
            print(f"update={step} ce={ce_sum/M15.ACCUM:.5f} "
                  f"kl={kl_sum/M15.ACCUM:.5f} elapsed={elapsed:.1f}s", flush=True)
        if step in CHECK_STEPS:
            model.eval()
            student_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, True)
            delta = student_bpb - donor_bpb
            evaluations.append({"step": step, "student_bpb": student_bpb,
                                "delta_bpb": delta})
            path = os.path.join(checkpoint_dir, f"meth16_update{step}.pt")
            checkpoints[str(step)] = save_state(path, step, wrappers, opt, rng, device)
            print(f"quality update={step} student={student_bpb:.8f} "
                  f"delta={delta:+.8f}", flush=True)
            budget(start, device)
            if delta > 0.05:
                stop_reason = "interim_quality_gross_failure"
                break
            model.train()
    model.eval()
    model.gradient_checkpointing_disable()
    terminal = {}
    if last_step == FINAL_STEP and stop_reason is None:
        donor_end = M15.bpb(model, eval_ids, eval_bytes, wrappers, False)
        assert abs(donor_end - donor_bpb) <= 1e-5
        trained_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, True)
        originals = [w.router.detach().clone() for w in wrappers]
        perm_rng = np.random.default_rng(PERM_SEED)
        with torch.no_grad():
            for w, original in zip(wrappers, originals):
                perm = torch.as_tensor(perm_rng.permutation(M15.E),
                                       dtype=torch.long, device=device)
                w.router.copy_(original[perm])
        permuted_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, True)
        with torch.no_grad():
            for w, original in zip(wrappers, originals):
                w.router.copy_(original)
        generation_rows, counts = generation(model, eval_ids, wrappers, device, start)
        slots = [w.changed_slots() for w in wrappers]
        terminal = {"donor_bpb": donor_end, "student_bpb": trained_bpb,
                    "delta_bpb": trained_bpb - donor_end,
                    "permuted_router_bpb": permuted_bpb,
                    "route_utility_bpb": permuted_bpb - trained_bpb,
                    "changed_slots_by_layer": slots,
                    "min_changed_slots": min(slots),
                    "generation": generation_rows,
                    "repetition_counts": counts}
        passed = (terminal["delta_bpb"] <= 0.02
                  and terminal["route_utility_bpb"] >= 0.002
                  and counts["student"] <= counts["donor"] + 1
                  and min(slots) >= 64)
        decision = "joint_development_gate_pass" if passed else "joint_development_gate_fail"
    else:
        decision = stop_reason or "incomplete"
    elapsed, rss, peak = budget(start, device)
    result = {"experiment": "METH-16", "source": {
                  "model": M15.M13.MODEL, "revision": M15.M13.REV,
                  "model_sha256": M15.M13.MODEL_SHA,
                  "tokenizer_fingerprint": M15.M13.TOK_FP},
              "resume": {"path": os.path.abspath(args.resume),
                         "sha256": RESUME_SHA, "updates": 16},
              "data": {"train_ids_sha256": M15.TRAIN_IDS_SHA,
                       "eval_ids_sha256": EVAL_IDS_SHA,
                       "eval_scored_bytes": int(eval_bytes.sum())},
              "schedule": {"target_updates": FINAL_STEP,
                           "last_applied_update": last_step,
                           "sequence_length": M15.SEQ,
                           "microbatches_per_update": M15.ACCUM,
                           "total_exposed_token_positions": last_step * M15.SEQ * M15.ACCUM},
              "runtime": {"torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0],
                          "elapsed_seconds": elapsed,
                          "rss_end_bytes": rss,
                          "gpu_peak_allocated_bytes": peak},
              "evaluations": evaluations, "steps": steps,
              "checkpoints": checkpoints, "terminal": terminal,
              "decision": decision}
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"last_update": last_step, "evaluations": evaluations,
                      "terminal": {k: v for k, v in terminal.items()
                                   if k != "generation"},
                      "runtime": result["runtime"], "decision": decision},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
