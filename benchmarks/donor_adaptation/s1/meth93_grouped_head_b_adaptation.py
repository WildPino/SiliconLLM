#!/usr/bin/env python3
"""METH-93: distill BF16+E128 teacher into grouped-head-R8+E128 B factors."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
from safetensors import safe_open
import torch
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth56_product_key_retention as M56
import meth57_product_key_external_audit as M57
import meth91_export_grouped_head_core as M91


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CORE = ROOT / "results/native_expert_scaling/meth91_qwen05b_instruct_grouped_head_core.safetensors"
CORE_SHA = "6f994c9ddf047adc789d3e1c841dd3be156cc578ed0d6c3f4045f97212e2accb"
UPDATES = 64
RNG_SEED = 88088
MAX_SECONDS = 15 * 60
MAX_RSS_BYTES = 20 * (1 << 30)
MAX_GPU_BYTES = int(10.5 * (1 << 30))


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}
    if (record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS_BYTES
            or record["gpu_peak_bytes"] > MAX_GPU_BYTES):
        raise RuntimeError(f"METH-93 resource stop: {record}")
    return record


def load_wrappers(model, expert_state, device):
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(expert_state[li][key].to(device))
            wrappers.append(wrapper)
    return wrappers


def load_core(model, device):
    params = dict(model.named_parameters())
    assert len(params) == 290
    with safe_open(str(CORE), framework="pt", device="cpu") as archive:
        assert archive.metadata()["format"] == "QWEN25_INSTRUCT_GROUP128_R8H_GROUP64_R8FFN_BF16ATTN_V1"
        assert archive.metadata()["model_sha256"] == M57.MODEL_SHA
        assert archive.metadata()["product_key_checkpoint_sha256"] == M57.CHECKPOINT_SHA
        assert len(archive.keys()) == 363
        with torch.no_grad():
            for name, param in params.items():
                organ = M24.classify(name, tuple(param.shape))
                if organ == "ffn":
                    q = archive.get_tensor(name + ".q").to(device)
                    scale = archive.get_tensor(name + ".scale").to(device)
                    param.copy_(M91.reconstruct_ffn(q, scale))
                elif organ == "tied_head":
                    q = archive.get_tensor(name + ".q").to(device)
                    scale = archive.get_tensor(name + ".scale").to(device)
                    param.copy_(M91.reconstruct_head(q, scale))
                else:
                    stored = archive.get_tensor(name)
                    assert stored.dtype == (torch.float32 if organ == "control"
                                            else torch.bfloat16)
                    assert torch.equal(stored.to(param.dtype), param.detach().cpu())
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()


def objective(student_logits, teacher_logits, mask):
    student_lp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_lp = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_prob = teacher_lp.exp()
    kl = (teacher_prob * (teacher_lp - student_lp)).sum(dim=-1)
    kl = (kl * mask).sum() / mask.sum()
    teacher_values, teacher_ids = teacher_logits.float().topk(2, dim=-1)
    target = teacher_ids[..., 0]
    confident = (teacher_values[..., 0] - teacher_values[..., 1] >= 0.2) & (mask > 0)
    student_values, student_ids = student_logits.float().topk(2, dim=-1)
    other = torch.where(student_ids[..., 0] == target,
                        student_values[..., 1], student_values[..., 0])
    target_value = student_logits.float().gather(-1, target.unsqueeze(-1)).squeeze(-1)
    hinge = F.relu(other - target_value + 0.2)
    margin = (hinge * confident).sum() / confident.sum().clamp_min(1)
    return kl + 0.1 * margin, kl.detach(), margin.detach(), int(confident.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    args = ap.parse_args()
    assert M15.M13.sha256(CORE) == CORE_SHA
    assert M15.M13.sha256(M56.TRAIN_CHAT_PATH) == M56.TRAIN_CHAT_SHA
    assert M15.M13.sha256(M56.TEACHER_PATH) == M56.TEACHER_SHA
    assert M15.M13.sha256(M15.TRAIN_PATH) == M15.TRAIN_FILE_SHA
    assert M15.M13.sha256(M56.DEV_PATH) == M56.DEV_SHA
    assert M15.M13.sha256(M56.OLD_DEV_PATH) == M56.OLD_DEV_SHA
    assert M15.M13.sha256(M44.PROMPT_PATH) == M44.PROMPT_SHA
    assert M15.M13.sha256(M56.M47_DEV_PATH) == M56.M47_DEV_SHA
    training_path = DIR / "meth56_product_key_retention_result.json"
    assert M15.M13.sha256(training_path) == M57.TRAINING_SHA
    checkpoint = json.loads(training_path.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M57.MODEL_SHA
    assert len(state["expert_state"]) == M15.M13.L
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
    rng = np.random.default_rng(RNG_SEED)
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
    teacher_wrappers = load_wrappers(teacher, state["expert_state"], device)
    teacher.eval()
    student = make_model()
    load_core(student, device)
    student_wrappers = load_wrappers(student, state["expert_state"], device)
    for wrapper in student_wrappers:
        wrapper.a.requires_grad_(False)
        wrapper.router.requires_grad_(False)
        assert wrapper.b.requires_grad
    trainable = [wrapper.b for wrapper in student_wrappers]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.0)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    student.train()
    assert torch.is_grad_enabled()
    del state
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
                train_ids = torch.as_tensor(ids_np, dtype=torch.long,
                                            device=device).unsqueeze(0)
                mask = torch.ones((1, train_ids.shape[1] - 1), device=device)
                draws.append({"update": update, "kind": kind, "row": row,
                              "offset": offset})
            else:
                choice = int(rng.integers(len(chat)))
                row, ids, saved_mask = chat[choice]
                train_ids = torch.as_tensor(ids, dtype=torch.long,
                                            device=device).unsqueeze(0)
                mask = torch.as_tensor(saved_mask, dtype=torch.float32,
                                       device=device).unsqueeze(0)
                draws.append({"update": update, "kind": kind, "train_row": row})
            inputs = train_ids[:, :-1]
            with torch.no_grad():
                teacher_logits = teacher(inputs, use_cache=False).logits
            student_logits = student(inputs, use_cache=False).logits
            if update == 1 and kind == "raw":
                print(json.dumps({"grad_enabled": torch.is_grad_enabled(),
                                  "student_logits_grad": student_logits.requires_grad,
                                  "b_grad": student_wrappers[0].b.requires_grad,
                                  "wrapper_enabled": student_wrappers[0].enabled}), flush=True)
            loss, kl, margin, confident = objective(student_logits, teacher_logits, mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite METH-93 loss at update {update}")
            (loss / 2).backward()
            record[kind] = {"kl": float(kl), "margin": float(margin),
                            "confident": confident, "objective": float(loss.detach())}
            del train_ids, inputs, mask, teacher_logits, student_logits, loss, kl, margin
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not np.isfinite(norm):
            raise FloatingPointError(f"nonfinite METH-93 gradient at update {update}")
        optimizer.step()
        record["clip_pre_norm"] = norm
        losses.append(record)
        if update in (1, 16, 32, 48, 64):
            print(json.dumps({"update": update, "loss": record,
                              "budget": budget(start, device)}), flush=True)
        budget(start, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_state = {"expert_state": [
        {key: getattr(wrapper, key).detach().cpu().clone()
         for key in ("a", "b", "router")}
        for wrapper in student_wrappers],
        "updates": UPDATES, "parent_updates": 512,
        "source_sha256": M57.MODEL_SHA, "core_sha256": CORE_SHA,
        "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
        "train_ids_sha256": M15.TRAIN_IDS_SHA,
        "train_chat_sha256": M56.TRAIN_CHAT_SHA,
        "seed": RNG_SEED, "trainable": "B_only"}
    torch.save(output_state, args.checkpoint)
    report = {"experiment": "METH-93-grouped-head-E128-B-distillation",
              "source_sha256": M57.MODEL_SHA, "core_sha256": CORE_SHA,
              "teacher_checkpoint_sha256": M57.CHECKPOINT_SHA,
              "checkpoint": {"path": str(args.checkpoint.resolve()),
                             "sha256": M15.M13.sha256(args.checkpoint),
                             "bytes": args.checkpoint.stat().st_size},
              "seed": RNG_SEED, "updates": UPDATES, "draws": draws,
              "losses": losses,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__},
              "decision": "checkpoint_ready_for_new_source_development"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint_sha256": report["checkpoint"]["sha256"],
                      "first_loss": losses[0], "last_loss": losses[-1],
                      "runtime": report["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
