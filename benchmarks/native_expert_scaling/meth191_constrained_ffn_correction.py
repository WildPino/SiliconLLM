#!/usr/bin/env python3
"""Constrained B-only rank-64 FFN correction with train-only selection."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

import meth189_train_q6_ffn_correction as T


P, D, O = T.P, T.D, T.O
UPDATES = 64
LR = 1e-4
SPLIT_SEED = 191191
MAX_RATIO = .03
CHECKPOINTS = (0, 16, 32, 64)


def validation_cases(raw_ids, chat, draws):
    excluded = set(json.loads(Path(D.M56.TRAIN_CHAT_PATH).read_text(encoding="utf-8"))
                   ["rows"][i]["train_row"] for i in range(256))
    for path in (D.M56.DEV_PATH, D.M56.OLD_DEV_PATH,
                 D.M44.PROMPT_PATH, D.M56.M47_DEV_PATH):
        excluded.update(row["train_row"] for row in json.loads(
            Path(path).read_text(encoding="utf-8"))["rows"])
    assert len(excluded) == 352
    used_raw = {draw["raw_row"] for draw in draws}
    raw_pool = np.asarray([i for i in range(raw_ids.shape[0])
                           if i not in used_raw and i not in excluded], dtype=np.int32)
    used_chat = {draw["chat_index"] for draw in draws[:UPDATES]}
    chat_pool = np.asarray([i for i in range(len(chat)) if i not in used_chat],
                           dtype=np.int32)
    rng = np.random.default_rng(SPLIT_SEED)
    selected_raw = rng.choice(raw_pool, size=8, replace=False)
    selected_chat = rng.choice(chat_pool, size=8, replace=False)
    cases = []
    for row in selected_raw:
        offset = int(rng.integers(512 - D.M15.SEQ + 1))
        ids = raw_ids[int(row), offset:offset + D.M15.SEQ].tolist()
        cases.append({"kind": "raw", "row": int(row), "offset": offset,
                      "ids": ids, "mask": [1] * (len(ids) - 1)})
    for index in selected_chat:
        row, ids, mask = chat[int(index)]
        cases.append({"kind": "chat", "index": int(index),
                      "train_row": row, "ids": ids, "mask": mask})
    return cases


def validate(teacher, student, cases, device, start):
    student.eval()
    values = []
    with torch.no_grad():
        for case in cases:
            ids = torch.as_tensor(case["ids"], dtype=torch.long, device=device)[None]
            mask = torch.as_tensor(case["mask"], dtype=torch.float32, device=device)[None]
            inputs = ids[:, :-1]
            teacher_logits = teacher(inputs, use_cache=False).logits
            student_logits = student(inputs, use_cache=False).logits
            objective, kl, margin, confident = O.objective(
                student_logits, teacher_logits, mask)
            values.append({"kind": case["kind"], "objective": float(objective),
                           "kl": float(kl), "margin": float(margin),
                           "confident": confident})
            T.budget(start, device)
    return {"mean_objective": sum(row["objective"] for row in values) / len(values),
            "raw_mean": sum(row["objective"] for row in values[:8]) / 8,
            "chat_mean": sum(row["objective"] for row in values[8:]) / 8,
            "rows": values}


def project(corrections, base_norms):
    ratios = []
    with torch.no_grad():
        for name, module in corrections.items():
            gram_a = module.a @ module.a.T
            gram_b = module.b.T @ module.b
            residual_norm = ((gram_a * gram_b).sum()).clamp_min(0).sqrt()
            ratio = float(residual_norm / base_norms[name])
            if ratio > MAX_RATIO:
                module.b.mul_(MAX_RATIO / ratio)
                ratio = MAX_RATIO
            ratios.append(ratio)
    return max(ratios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists() and not args.checkpoint.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
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
    assert len(draws) == 256
    cases = validation_cases(raw_ids, chat, draws)
    assert len(cases) == 16

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
    teacher, _, _ = T.make_model(device, parent_state, child_state)
    teacher.eval()
    student, _, corrections = T.make_model(device, parent_state, child_state,
                                          core=True)
    del parent_state, child_state
    for module in corrections.values():
        module.a.requires_grad_(False)
    trainable = [module.b for module in corrections.values()]
    assert len(trainable) == 72
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    base_norms = {name: float(module.base.weight.float().norm())
                  for name, module in corrections.items()}
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    baseline = validate(teacher, student, cases, device, start)
    assert project(corrections, base_norms) == 0
    checkpoints = {0: baseline}
    records = []
    best_score = float("inf")
    best_update = None
    best_b = None
    train_started = time.monotonic()
    print(json.dumps({"update": 0, "validation": baseline["mean_objective"],
                      "runtime": T.budget(start, device)}), flush=True)
    for update in range(1, UPDATES + 1):
        student.train()
        draw = draws[update - 1]
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
            inputs = ids[:, :-1]
            with torch.no_grad():
                teacher_logits = teacher(inputs, use_cache=False).logits
            student_logits = student(inputs, use_cache=False).logits
            loss, kl, margin, confident = O.objective(student_logits,
                                                       teacher_logits, mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at {update} {kind}")
            (loss / 2).backward()
            record[kind] = {"objective": float(loss.detach()),
                            "kl": float(kl), "margin": float(margin),
                            "confident": confident}
            del ids, mask, inputs, teacher_logits, student_logits, loss, kl, margin
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not np.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient at {update}")
        optimizer.step()
        record["clip_pre_norm"] = norm
        record["max_weight_correction_ratio"] = project(corrections, base_norms)
        records.append(record)
        if update in CHECKPOINTS[1:]:
            evaluated = validate(teacher, student, cases, device, start)
            checkpoints[update] = evaluated
            if evaluated["mean_objective"] < best_score:
                best_score = evaluated["mean_objective"]
                best_update = update
                best_b = {name: module.b.detach().cpu().clone()
                          for name, module in corrections.items()}
            print(json.dumps({"update": update,
                              "validation": evaluated["mean_objective"],
                              "raw": record["raw"], "chat": record["chat"],
                              "max_ratio": record["max_weight_correction_ratio"],
                              "runtime": T.budget(start, device)}), flush=True)
        current = T.budget(start, device)
        if update == 16:
            projected = current["seconds"] + (
                time.monotonic() - train_started) / update * (UPDATES - update)
            if projected > T.MAX_SECONDS:
                raise RuntimeError(f"METH-191 projected runtime stop: {projected:.1f}s")

    selected = best_score <= .9 * baseline["mean_objective"]
    rounded_validation = None
    checkpoint_record = None
    if selected:
        with torch.no_grad():
            for name, module in corrections.items():
                module.b.copy_(best_b[name].to(device))
        tensors = {}
        for name, module in corrections.items():
            tensors[name + ".a"] = module.a.detach().cpu().bfloat16().contiguous()
            tensors[name + ".b"] = module.b.detach().cpu().bfloat16().contiguous()
        assert len(tensors) == 144
        assert sum(t.numel() * t.element_size() for t in tensors.values()) == T.PAYLOAD
        # The rounded checkpoint is the exact candidate subsequently scored.
        with torch.no_grad():
            for name, module in corrections.items():
                module.a.copy_(tensors[name + ".a"].to(device))
                module.b.copy_(tensors[name + ".b"].to(device))
        rounded_validation = validate(teacher, student, cases, device, start)
        if rounded_validation["mean_objective"] > 1.01 * best_score:
            selected = False
        else:
            save_file(tensors, str(args.checkpoint), metadata={
                "format": "METH191_Q6_FFN_RANK64_BF16_V1",
                "source_sha256": P.M57.MODEL_SHA,
                "core_sha256": P.digest(P.CORE),
                "parent_sha256": P.M57.CHECKPOINT_SHA,
                "child_sha256": P.M122.SPECIALIZED_SHA,
                "selected_update": str(best_update)})
            with safe_open(str(args.checkpoint), framework="pt", device="cpu") as archive:
                assert sorted(archive.keys()) == sorted(tensors)
                for name, tensor in tensors.items():
                    assert torch.equal(archive.get_tensor(name), tensor)
            checkpoint_record = {"path": str(args.checkpoint.resolve()),
                                 "bytes": args.checkpoint.stat().st_size,
                                 "payload_bytes": T.PAYLOAD,
                                 "sha256": P.digest(args.checkpoint),
                                 "readback_exact": True}
    result = {"experiment": "METH-191-constrained-B-only-FFN-correction",
              "source_sha256": P.M57.MODEL_SHA,
              "core_sha256": P.digest(P.CORE),
              "parent_sha256": P.M57.CHECKPOINT_SHA,
              "child_sha256": P.M122.SPECIALIZED_SHA,
              "train_raw_sha256": D.M15.TRAIN_FILE_SHA,
              "train_chat_sha256": D.M56.TRAIN_CHAT_SHA,
              "long_teacher_sha256": D.M107.LONG_TEACHER_SHA,
              "split_seed": SPLIT_SEED, "validation_cases": cases,
              "updates": UPDATES, "lr": LR, "rank": T.RANK,
              "weight_ratio_limit": MAX_RATIO,
              "validation": checkpoints,
              "selected_update": best_update if selected else None,
              "best_float_objective": best_score,
              "rounded_validation": rounded_validation,
              "records": records,
              "checkpoint": checkpoint_record,
              "ideal_addressed_bytes_per_token": 481_534_976 + T.PAYLOAD,
              "runtime": {**T.budget(start, device),
                          "gpu": torch.cuda.get_device_name(device)},
              "decision": "train_only_selected_development_pending" if selected else
                          "train_only_rejected_do_not_use_development"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size + (args.checkpoint.stat().st_size
                                       if checkpoint_record else 0) < T.MAX_DISK
    print(json.dumps({"decision": result["decision"],
                      "initial_objective": baseline["mean_objective"],
                      "best_objective": best_score,
                      "best_update": best_update,
                      "rounded_objective": rounded_validation["mean_objective"]
                      if rounded_validation else None,
                      "checkpoint": checkpoint_record,
                      "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
