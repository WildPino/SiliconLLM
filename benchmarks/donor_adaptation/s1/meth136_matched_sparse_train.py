#!/usr/bin/env python3
"""Matched full-model E1280/E12800 continuation with CPU-master B factors."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth56_product_key_retention as M56
import meth57_product_key_external_audit as M57
import meth95_hierarchical_e1280_parity as M95
import meth107_long_chat_child_retention as M107


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/native_expert_scaling"))
from meth136_sparse_experts import CpuSparseExperts, SelectedRowAdam
from meth134_sparse_e12800_training import SparseCollector


DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
ART = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling"
CHILD = ART / "meth107_long_chat_e1280.pt"
CHILD_SHA = "15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520"
EXACT_BANK = ART / "meth126_shared_a_factor_bank.bin"
EXACT_SHA = "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1"
THIRD = ART / "meth135_third_router.bin"
THIRD_SHA = "692db3dd1a647debb051ef66612263b5e16ef27c73978e9abd3beb6b3229abc9"
PARITY = DOC / "meth121_zero_mean_child_external_manifest.json"
PARITY_SHA = "7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366"
BANK_HEADER = struct.Struct("<8s8I")
THIRD_HEADER = struct.Struct("<8s6I")
OUT_HEADER = struct.Struct("<8s4I")
UPDATES = 256
SEED = 136136
MAX_SECONDS = 45 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 50 * (1 << 30)
MAX_DISK = 10_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype="<u2").astype("<u4") << 16).view("<f4")


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS
            or result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-136 resource stop: {result}")
    return result


def bind_inputs():
    assert digest(EXACT_BANK) == EXACT_SHA
    assert digest(THIRD) == THIRD_SHA
    assert digest(CHILD) == CHILD_SHA
    assert digest(PARITY) == PARITY_SHA
    assert digest(M95.CHECKPOINT_REPORT) == M57.TRAINING_SHA
    assert digest(M107.LONG_TEACHER) == M107.LONG_TEACHER_SHA
    for path, expected in ((M56.TRAIN_CHAT_PATH, M56.TRAIN_CHAT_SHA),
                           (M56.TEACHER_PATH, M56.TEACHER_SHA),
                           (M15.TRAIN_PATH, M15.TRAIN_FILE_SHA),
                           (M56.DEV_PATH, M56.DEV_SHA),
                           (M56.OLD_DEV_PATH, M56.OLD_DEV_SHA),
                           (M44.PROMPT_PATH, M44.PROMPT_SHA),
                           (M56.M47_DEV_PATH, M56.M47_DEV_SHA)):
        assert digest(path) == expected, path
    training = json.loads(M95.CHECKPOINT_REPORT.read_text(encoding="utf-8"))
    parent = training["checkpoints"]["512"]
    assert parent["sha256"] == M57.CHECKPOINT_SHA
    assert digest(parent["path"]) == M57.CHECKPOINT_SHA
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert digest(source) == M57.MODEL_SHA
    return torch.load(parent["path"], map_location="cpu", weights_only=False), (
        torch.load(CHILD, map_location="cpu", weights_only=False))


def load_factor_bank():
    data = np.memmap(EXACT_BANK, dtype=np.uint8, mode="r")
    dims = (24, 896, 64, 8, 16, 32, 10, 8)
    assert BANK_HEADER.unpack_from(data) == (b"M126FB01", *dims)
    layers, width, rank, na, nb, cr, children, factor_rank = dims
    router_bytes = (rank * width + (na + nb) * rank + cr * width
                    + na * nb * children * cr) * 4
    a_bytes = na * nb * factor_rank * width * 2
    b_bytes = na * nb * children * width * factor_rank * 2
    layer_bytes = router_bytes + a_bytes + b_bytes
    assert data.size == BANK_HEADER.size + layers * layer_bytes
    source_a, source_b, router_prefixes = [], [], []
    for li in range(layers):
        base = BANK_HEADER.size + li * layer_bytes
        router_prefixes.append(data[base:base + router_bytes].tobytes())
        a_bits = np.frombuffer(data, dtype="<u2", count=a_bytes // 2,
                               offset=base + router_bytes).reshape(na * nb, factor_rank, width)
        b_bits = np.frombuffer(data, dtype="<u2", count=b_bytes // 2,
                               offset=base + router_bytes + a_bytes).reshape(
                                   na * nb * children, width, factor_rank)
        source_a.append(torch.from_numpy(bf16_to_f32(a_bits).copy()))
        source_b.append(torch.from_numpy(bf16_to_f32(b_bits).copy()))
    return source_a, source_b, router_prefixes


def load_third():
    data = np.memmap(THIRD, dtype=np.uint8, mode="r")
    assert THIRD_HEADER.unpack_from(data) == (b"M135RT01", 24, 896, 32,
                                              1280, 10, 134134)
    projection_bytes = 32 * 896 * 4
    key_bytes = 1280 * 10 * 32 * 4
    layer_bytes = projection_bytes + key_bytes
    assert data.size == THIRD_HEADER.size + 24 * layer_bytes
    result = []
    for li in range(24):
        base = THIRD_HEADER.size + li * layer_bytes
        projection = np.frombuffer(data, dtype="<f4", count=32 * 896,
                                   offset=base).copy().reshape(32, 896)
        keys = np.frombuffer(data, dtype="<f4", count=1280 * 10 * 32,
                             offset=base + projection_bytes).copy().reshape(1280, 10, 32)
        result.append((torch.from_numpy(projection), torch.from_numpy(keys)))
    return result


def training_data():
    prompts = {row["train_row"]: row for row in json.loads(
        Path(M56.TRAIN_CHAT_PATH).read_text(encoding="utf-8"))["rows"]}
    teacher_rows = json.loads(M107.LONG_TEACHER.read_text(encoding="utf-8"))["rows"]
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
    for path in (M56.DEV_PATH, M56.OLD_DEV_PATH, M44.PROMPT_PATH,
                 M56.M47_DEV_PATH):
        excluded.update(row["train_row"] for row in json.loads(
            Path(path).read_text(encoding="utf-8"))["rows"])
    assert len(excluded) == 352
    with np.load(M15.TRAIN_PATH, allow_pickle=False) as archive:
        raw_ids = archive["ids"].copy()
    assert raw_ids.shape == (31250, 512) and raw_ids.dtype == np.int32
    assert M15.sha_bytes(raw_ids.tobytes()) == M15.TRAIN_IDS_SHA
    raw_pool = np.asarray([i for i in range(31250) if i not in excluded])
    rng = np.random.default_rng(SEED)
    chat_order = rng.permutation(len(chat))
    draws = []
    for update in range(UPDATES):
        row = int(rng.choice(raw_pool))
        offset = int(rng.integers(512 - M15.SEQ + 1))
        chat_index = int(chat_order[update])
        draws.append({"update": update + 1, "raw_row": row,
                      "raw_offset": offset, "chat_index": chat_index,
                      "chat_train_row": chat[chat_index][0]})
    return raw_ids, chat, draws


def model_shell(device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    return model


def make_wrappers(model, parent_state, child_state, source_a, source_b,
                  router_prefixes, device, mode, third=None):
    wrappers = []
    banks = []
    for li, layer in enumerate(model.model.layers):
        parent = M55.ProductKeyExperts(layer.mlp, li).to(device)
        with torch.no_grad():
            for key in ("a", "b", "router"):
                getattr(parent, key).copy_(parent_state[li][key].to(device))
        if mode == "teacher":
            wrapper = M95.HierarchicalExperts(parent, li).to(device)
            with torch.no_grad():
                for key in ("a", "b", "router", "child_projection", "child_keys"):
                    getattr(wrapper, key).copy_(child_state[li][key].to(device))
                raw = wrapper.b.view(M15.E, M95.CHILDREN, M15.M13.D, M15.R)
                centered = parent.b.detach()[:, None] + raw - raw.mean(dim=1, keepdim=True)
                wrapper.b.copy_(centered.reshape_as(wrapper.b))
                assert torch.equal(wrapper.a.detach().to(torch.bfloat16).cpu()[::10],
                                   source_a[li].to(torch.bfloat16))
                assert torch.equal(wrapper.b.detach().to(torch.bfloat16).cpu(),
                                   source_b[li].to(torch.bfloat16))
            wrapper.router.requires_grad_(False)
        else:
            cpu_bank = source_b[li].clone() if mode == "control" else (
                source_b[li].repeat_interleave(10, dim=0).contiguous())
            projection, keys = (third[li] if mode == "candidate" else (None, None))
            wrapper = CpuSparseExperts(parent, li, cpu_bank, source_a[li],
                                       projection, keys).to(device)
            with torch.no_grad():
                for key in ("router", "child_projection", "child_keys"):
                    getattr(wrapper, key).copy_(child_state[li][key].to(device))
            wrapper.router.requires_grad_(False)
            assert wrapper.cpu_bank.device.type == "cpu"
            banks.append(wrapper.cpu_bank)
        # The versioned bank contains exactly these frozen router bytes.
        rank, width, na, nb, cr, child_count = 64, 896, 8, 16, 32, 10
        prefix = router_prefixes[li]
        raw_router = (wrapper.router.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False).tobytes()
                      + wrapper.child_projection.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False).tobytes()
                      + wrapper.child_keys.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False).tobytes())
        assert len(raw_router) == (rank * width + (na + nb) * rank
                                   + cr * width + na * nb * child_count * cr) * 4
        assert raw_router == prefix, li
        layer.mlp = wrapper
        wrappers.append(wrapper)
        del parent
    return wrappers, banks


def parity_check(teacher, student, prompts, device):
    teacher.eval(); student.eval()
    maximum = 0.0
    choices = 0
    with torch.inference_mode():
        for item in prompts[:8]:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                  device=device)[None]
            expected = teacher(ids, use_cache=False).logits
            actual = student(ids, use_cache=False).logits
            difference = float((expected.float() - actual.float()).abs().max())
            maximum = max(maximum, difference)
            choices += int(torch.equal(expected.argmax(-1), actual.argmax(-1)))
            assert torch.equal(expected, actual), (item["source_id"], difference)
    assert choices == 8 and maximum == 0
    return {"prompts": choices, "logit_max_abs_error": maximum}


def export_bank(path, banks, source_b, centered):
    rows = 12800 if centered else 1280
    assert len(banks) == len(source_b) == 24
    if centered:
        for bank, original in zip(banks, source_b):
            view = bank.view(1280, 10, 896, 8)
            for start in range(0, 1280, 32):
                segment = view[start:start + 32]
                mean = segment.mean(dim=1, keepdim=True)
                segment.sub_(mean).add_(original[start:start + 32, None])
    distinct = []
    mean_errors = []
    for bank, original in zip(banks, source_b):
        if centered:
            view = bank.view(1280, 10, 896, 8)
            mean_errors.append(float((view.mean(dim=1) - original).abs().max()))
            different = (view.to(torch.bfloat16) != original[:, None].to(torch.bfloat16))
        else:
            different = (bank.to(torch.bfloat16) != original.to(torch.bfloat16))
        distinct.append(int(different.reshape(rows, -1).any(dim=1).sum()))
    with path.open("xb") as file:
        file.write(OUT_HEADER.pack(b"M136BF01", 24, 896, 8, rows))
        for bank in banks:
            bits = bank.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
            file.write(bits.astype("<u2", copy=False).tobytes())
    expected_bytes = OUT_HEADER.size + 24 * rows * 896 * 8 * 2
    assert path.stat().st_size == expected_bytes
    check = np.memmap(path, dtype=np.uint8, mode="r")
    assert OUT_HEADER.unpack_from(check) == (b"M136BF01", 24, 896, 8, rows)
    for li, bank in enumerate(banks):
        offset = OUT_HEADER.size + li * rows * 896 * 8 * 2
        readback = np.frombuffer(check, dtype="<u2", count=rows * 896 * 8,
                                 offset=offset).reshape(rows, 896, 8)
        assert np.array_equal(readback,
                              bank.to(torch.bfloat16).contiguous().view(torch.uint16).numpy())
    return {"path": str(path.resolve()), "bytes": expected_bytes,
            "sha256": digest(path), "readback_exact": True,
            "bf16_distinct_rows_by_layer": distinct,
            "mean_preservation_max_abs_by_layer": mean_errors if centered else None}


def train_arm(name, teacher, parent_state, child_state, source_a, source_b,
              router_prefixes, third, raw_ids, chat, draws, parity_items,
              device, start, artifact_path, progress_path):
    model = model_shell(device)
    wrappers, banks = make_wrappers(model, parent_state, child_state,
                                    source_a, source_b, router_prefixes,
                                    device, name, third)
    initial = parity_check(teacher, model, parity_items, device)
    for wrapper in wrappers:
        wrapper.forward_B_transfer_bytes = 0
        wrapper.gather_calls = 0
    optimizers = [SelectedRowAdam(bank) for bank in banks]
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    route_counts = [torch.zeros(bank.shape[0], dtype=torch.int64) for bank in banks]
    records = []
    gradient_transfer_total = 0
    arm_started = time.monotonic()
    for draw in draws:
        update = draw["update"]
        for wrapper in wrappers:
            wrapper.collector = SparseCollector()
        record = {"update": update}
        for kind in ("raw", "chat"):
            if kind == "raw":
                window = raw_ids[draw["raw_row"],
                                 draw["raw_offset"]:draw["raw_offset"] + M15.SEQ]
                ids = torch.as_tensor(window, dtype=torch.long, device=device)[None]
                ce_mask = torch.ones((1, M15.SEQ - 1), device=device)
                kl_weight = 4.0
            else:
                _, id_list, mask = chat[draw["chat_index"]]
                ids = torch.as_tensor(id_list, dtype=torch.long, device=device)[None]
                ce_mask = torch.as_tensor(mask, dtype=torch.float32,
                                          device=device)[None]
                kl_weight = 8.0
            inputs = ids[:, :-1]
            targets = ids[:, 1:]
            kl_mask = torch.ones_like(targets, dtype=torch.float32)
            with torch.no_grad():
                teacher_logits = teacher(inputs, use_cache=False).logits
            student_logits = model(inputs, use_cache=False).logits
            loss, ce, kl, margin = M107.objective(
                student_logits, teacher_logits, targets,
                ce_mask, kl_mask, kl_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite {name} loss at {update} {kind}")
            (loss / 2).backward()
            record[kind] = {"ce": float(ce.detach()), "kl": float(kl.detach()),
                            "margin": float(margin.detach()),
                            "objective": float(loss.detach())}
            del ids, inputs, targets, ce_mask, kl_mask, teacher_logits
            del student_logits, loss, ce, kl, margin
        gathered = []
        squared_norm = 0.0
        for li, wrapper in enumerate(wrappers):
            collector = wrapper.collector
            assert collector is not None and collector.parts
            all_ids = torch.cat([part[0] for part in collector.parts])
            route_counts[li] += torch.bincount(all_ids, minlength=banks[li].shape[0])
            unique, gradient, transferred = collector.materialize()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"nonfinite {name} gradient at {update} layer {li}")
            squared_norm += float(torch.sum(gradient.double().square()))
            gathered.append((unique, gradient))
            gradient_transfer_total += transferred
            wrapper.collector = None
        norm = math.sqrt(squared_norm)
        assert math.isfinite(norm) and norm > 0
        scale = min(1.0, 1.0 / norm)
        optimizer_rows = []
        for optimizer, (unique, gradient) in zip(optimizers, gathered):
            optimizer_rows.append(optimizer.step(unique, gradient, scale))
        record["clip_pre_norm"] = norm
        record["new_optimizer_rows_min"] = min(x["new_rows"] for x in optimizer_rows)
        record["new_optimizer_rows_max"] = max(x["new_rows"] for x in optimizer_rows)
        record["optimizer_rows_min"] = min(x["optimizer_rows"] for x in optimizer_rows)
        record["optimizer_rows_max"] = max(x["optimizer_rows"] for x in optimizer_rows)
        records.append(record)
        if update in (1, 16, 32, 64, 128, 192, 256):
            state = {"experiment": "METH-136-progress", "arm": name,
                     "completed_update": update, "initial_parity": initial,
                     "records": records,
                     "coverage_by_layer": [int((x > 0).sum()) for x in route_counts],
                     "optimizer_rows_by_layer": [x.used for x in optimizers],
                     "budget": budget(start, device)}
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(state, indent=2) + "\n",
                                     encoding="utf-8")
            print(json.dumps({"arm": name, "update": update,
                              "record": record, "budget": state["budget"]}), flush=True)
        current = budget(start, device)
        if update == 16:
            projected = current["seconds"] + (
                time.monotonic() - arm_started) / 16 * (UPDATES - 16)
            if name == "control":
                projected += (time.monotonic() - arm_started) / 16 * UPDATES
            if projected > MAX_SECONDS:
                raise RuntimeError(f"METH-136 projected runtime stop: {projected}")
    coverage = [int((counts > 0).sum()) for counts in route_counts]
    load_skew = [float(counts.max() / counts.float().mean()) for counts in route_counts]
    selected_total = [int(counts.sum()) for counts in route_counts]
    expected_total = 4 * sum((M15.SEQ - 1) + (len(chat[d["chat_index"]][1]) - 1)
                             for d in draws)
    assert all(total == expected_total for total in selected_total), selected_total
    artifact = export_bank(artifact_path, banks, source_b, name == "candidate")
    disk_bytes = artifact["bytes"]
    assert disk_bytes < MAX_DISK
    if name == "candidate":
        assert min(coverage) >= 6400
        assert min(artifact["bf16_distinct_rows_by_layer"]) >= 3200
        assert max(load_skew) <= 50
    result = {"arm": name, "initial_parity": initial,
              "records": records, "route_coverage_by_layer": coverage,
              "route_max_to_mean_by_layer": load_skew,
              "route_selections_by_layer": selected_total,
              "expected_selections_per_layer": expected_total,
              "final_optimizer_rows_by_layer": [x.used for x in optimizers],
              "final_moment_allocated_bytes": sum((x.m.numel() + x.v.numel()) * 4
                                                 for x in optimizers),
              "forward_B_transfer_bytes": sum(w.forward_B_transfer_bytes for w in wrappers),
              "forward_B_gather_calls": sum(w.gather_calls for w in wrappers),
              "gradient_B_transfer_bytes": gradient_transfer_total,
              "artifact": artifact,
              "runtime": {"arm_seconds": time.monotonic() - arm_started,
                          **budget(start, device)}}
    progress_path.unlink(missing_ok=True)
    del model, wrappers, banks, optimizers
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--control-bank", type=Path, required=True)
    ap.add_argument("--candidate-bank", type=Path, required=True)
    args = ap.parse_args()
    for path in (args.out, args.control_bank, args.candidate_bank):
        assert not path.exists(), path
    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parent_state, child_state = bind_inputs()
    assert parent_state["updates"] == 512 and parent_state["source_sha256"] == M57.MODEL_SHA
    assert child_state["updates"] == 256 and child_state["source_sha256"] == M57.MODEL_SHA
    assert child_state["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
    source_a, source_b, router_prefixes = load_factor_bank()
    third = load_third()
    raw_ids, chat, draws = training_data()
    parity_manifest = json.loads(PARITY.read_text(encoding="utf-8"))
    parity_items = parity_manifest["items"]
    assert len(parity_items) == 24
    teacher = model_shell(device).eval()
    make_wrappers(teacher, parent_state["expert_state"],
                  child_state["expert_state"], source_a, source_b,
                  router_prefixes, device, "teacher")
    teacher.eval()
    budget(start, device)
    arms = []
    try:
        control = train_arm("control", teacher, parent_state["expert_state"],
                            child_state["expert_state"], source_a, source_b,
                            router_prefixes, None, raw_ids, chat, draws,
                            parity_items, device, start, args.control_bank,
                            args.out.with_name(args.out.stem + ".control.progress.json"))
        arms.append(control)
        candidate = train_arm("candidate", teacher, parent_state["expert_state"],
                              child_state["expert_state"], source_a, source_b,
                              router_prefixes, third, raw_ids, chat, draws,
                              parity_items, device, start, args.candidate_bank,
                              args.out.with_name(args.out.stem + ".candidate.progress.json"))
        arms.append(candidate)
    finally:
        partial = args.out.with_name(args.out.stem + ".partial.json")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({"experiment": "METH-136-matched-sparse-training",
                                       "completed_arms": arms,
                                       "runtime": {"seconds": time.monotonic() - start,
                                                   "rss_bytes": psutil.Process().memory_info().rss,
                                                   "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}},
                                      indent=2) + "\n",
                           encoding="utf-8")
    assert sum(arm["artifact"]["bytes"] for arm in arms) < MAX_DISK
    result = {"experiment": "METH-136-matched-sparse-training",
              "source_sha256": M57.MODEL_SHA,
              "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
              "child_checkpoint_sha256": CHILD_SHA,
              "exact_bank_sha256": EXACT_SHA, "third_sidecar_sha256": THIRD_SHA,
              "parity_manifest_sha256": PARITY_SHA,
              "raw_data_sha256": M15.TRAIN_FILE_SHA,
              "train_chat_sha256": M56.TRAIN_CHAT_SHA,
              "long_teacher_sha256": M107.LONG_TEACHER_SHA,
              "seed": SEED, "updates_per_arm": UPDATES,
              "recipe": {"trainable": "selected_B_only", "learning_rate": 1e-5,
                         "betas": [0.9, 0.999], "epsilon": 1e-8,
                         "weight_decay": 0, "clip_norm": 1.0,
                         "raw_kl_weight": 4.0, "chat_kl_weight": 8.0,
                         "grand_keys_trainable": False,
                         "candidate_centered_at_export": True},
              "draws": draws, "arms": arms,
              "runtime": {**budget(start, device),
                          "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__, "numpy": np.__version__},
              "decision": "paired_training_artifacts_ready_for_fresh_quality_audit"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.out.with_name(args.out.stem + ".partial.json").unlink(missing_ok=True)
    print(json.dumps({"decision": result["decision"],
                      "control": arms[0]["artifact"],
                      "candidate": arms[1]["artifact"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
