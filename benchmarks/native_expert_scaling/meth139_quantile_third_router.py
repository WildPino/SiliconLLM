#!/usr/bin/env python3
"""Calibrate conditional decile routing and screen held-out route load."""

import argparse
import gc
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth136_sparse_experts  # inserts the S1 import directory
from meth136_sparse_experts import CpuSparseExperts
import meth136_matched_sparse_train as M136
import meth55_product_key_experts as M55


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
HEADER = struct.Struct("<8s6I")
MAGIC = b"M139QT01"
QUANTILES = np.arange(1, 10, dtype=np.float64) / 10.0
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000


class QuantileCloneExperts(CpuSparseExperts):
    """Choose grandchild IDs but gather the cloned E1280 source B."""

    def __init__(self, parent, layer_id, cpu_bank, shared_a, projection,
                 thresholds):
        super().__init__(parent, layer_id, cpu_bank, shared_a)
        assert projection.shape == (32, 896)
        assert thresholds.shape == (1280, 9)
        self.register_buffer("quantile_projection", projection.clone())
        self.register_buffer("thresholds", thresholds.clone())
        self.last_grandchild = None

    def selected_ids(self, flat, children):
        q = F.linear(flat.float(), self.quantile_projection)
        scalar = q.gather(1, children.remainder(32).long())
        local = (scalar.unsqueeze(-1) >= self.thresholds[children]).sum(-1)
        self.last_grandchild = (children * 10 + local).detach()
        return children


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS
            or record["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-139 resource stop: {record}")
    return record


def calibrate(ids, values):
    assert ids.shape == values.shape and ids.ndim == 1
    assert ids.dtype == np.int32 and values.dtype == np.float32
    assert np.isfinite(values).all()
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_values = values[order]
    starts = np.searchsorted(sorted_ids, np.arange(1281), side="left")
    global_deciles = np.stack([
        np.quantile(values[ids % 32 == dimension], QUANTILES, method="linear")
        for dimension in range(32)]).astype("<f4")
    assert np.isfinite(global_deciles).all()
    thresholds = np.empty((1280, 9), dtype="<f4")
    fallback = 0
    for child in range(1280):
        chosen = sorted_values[starts[child]:starts[child + 1]]
        if chosen.size >= 10:
            thresholds[child] = np.quantile(
                chosen, QUANTILES, method="linear").astype("<f4")
        else:
            thresholds[child] = global_deciles[child % 32]
            fallback += 1
    assert np.isfinite(thresholds).all()
    assert np.all(np.diff(thresholds, axis=1) >= 0)
    return thresholds, fallback, np.diff(starts).astype(np.int32)


def summarize(counts):
    mean = counts.sum() / counts.size
    top = np.argsort(counts)[-5:][::-1]
    return {"coverage": int(np.count_nonzero(counts)),
            "selections": int(counts.sum()),
            "max_to_mean": float(counts.max() / mean),
            "top_five": [{"slot": int(slot), "selections": int(counts[slot])}
                         for slot in top]}


def export_sidecar(path, projections, thresholds):
    assert len(projections) == len(thresholds) == 24
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(HEADER.pack(MAGIC, 24, 896, 32, 1280, 10, 139139))
        for projection, layer_thresholds in zip(projections, thresholds):
            file.write(projection.numpy().astype("<f4", copy=False).tobytes())
            file.write(layer_thresholds.astype("<f4", copy=False).tobytes())
    layer_bytes = (32 * 896 + 1280 * 9) * 4
    expected = HEADER.size + 24 * layer_bytes
    assert path.stat().st_size == expected and expected < MAX_DISK
    readback = np.memmap(path, dtype=np.uint8, mode="r")
    assert HEADER.unpack_from(readback) == (MAGIC, 24, 896, 32, 1280, 10, 139139)
    for li, (projection, layer_thresholds) in enumerate(zip(projections, thresholds)):
        offset = HEADER.size + li * layer_bytes
        p = np.frombuffer(readback, dtype="<f4", count=32 * 896,
                          offset=offset).reshape(32, 896)
        t = np.frombuffer(readback, dtype="<f4", count=1280 * 9,
                          offset=offset + 32 * 896 * 4).reshape(1280, 9)
        assert np.array_equal(p, projection.numpy())
        assert np.array_equal(t, layer_thresholds)
    return {"path": str(path.resolve()), "bytes": expected,
            "sha256": M136.digest(path), "readback_exact": True}


def make_clone_wrappers(model, original_mlps, parent_state, child_state,
                        source_a, source_b, prefixes, projections, thresholds,
                        device):
    wrappers = []
    for li, (layer, base) in enumerate(zip(model.model.layers, original_mlps)):
        parent = M55.ProductKeyExperts(base, li).to(device)
        with torch.no_grad():
            for key in ("a", "b", "router"):
                getattr(parent, key).copy_(parent_state[li][key].to(device))
        wrapper = QuantileCloneExperts(
            parent, li, source_b[li].clone(), source_a[li],
            projections[li], torch.from_numpy(thresholds[li])).to(device)
        with torch.no_grad():
            for key in ("router", "child_projection", "child_keys"):
                getattr(wrapper, key).copy_(child_state[li][key].to(device))
        rank, width, na, nb, cr, child_count = 64, 896, 8, 16, 32, 10
        raw_router = (wrapper.router.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False).tobytes()
                      + wrapper.child_projection.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False).tobytes()
                      + wrapper.child_keys.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False).tobytes())
        assert len(raw_router) == (rank * width + (na + nb) * rank +
                                   cr * width + na * nb * child_count * cr) * 4
        assert raw_router == prefixes[li]
        layer.mlp = wrapper
        wrappers.append(wrapper)
    return wrappers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.sidecar.exists()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    gpu = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpu) == 1
    device = torch.device(f"cuda:{gpu[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parent_state, child_state = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    third = M136.load_third()
    raw_ids, chat, draws = M136.training_data()
    assert len(draws) == 256
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    control, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "control")
    projections = [entry[0] for entry in third]
    gpu_projections = [p.to(device) for p in projections]
    samples = [{"calibration_ids": [], "calibration_scalar": [],
                "validation_ids": [], "validation_scalar": []} for _ in range(24)]
    split = {"name": "calibration"}
    hooks = []
    for li, wrapper in enumerate(control):
        def collect(module, inputs, _output, layer_id=li):
            flat = inputs[0].reshape(-1, 896)
            children = module.last_selected
            q = F.linear(flat.float(), gpu_projections[layer_id])
            scalar = q.gather(1, children.remainder(32).long())
            samples[layer_id][split["name"] + "_ids"].append(
                children.flatten().cpu().numpy().astype(np.int32))
            samples[layer_id][split["name"] + "_scalar"].append(
                scalar.flatten().cpu().numpy().astype(np.float32))
        hooks.append(wrapper.register_forward_hook(collect))
    with torch.inference_mode():
        for draw in draws:
            split["name"] = "calibration" if draw["update"] <= 128 else "validation"
            raw = raw_ids[draw["raw_row"],
                          draw["raw_offset"]:draw["raw_offset"] + M136.M15.SEQ]
            _, chat_ids, _ = chat[draw["chat_index"]]
            for sequence in (raw, chat_ids):
                ids = torch.as_tensor(sequence, dtype=torch.long, device=device)[None]
                model(ids[:, :-1], use_cache=False)
            if draw["update"] in (32, 64, 128, 192, 256):
                print(json.dumps({"captured_update": draw["update"],
                                  "budget": budget(start, device)}), flush=True)
    for hook in hooks:
        hook.remove()
    del hooks, gpu_projections
    thresholds, fallback_count, calibration_count = [], [], []
    validation = []
    for li, layer in enumerate(samples):
        calibration_ids = np.concatenate(layer["calibration_ids"])
        calibration_scalar = np.concatenate(layer["calibration_scalar"])
        validation_ids = np.concatenate(layer["validation_ids"])
        validation_scalar = np.concatenate(layer["validation_scalar"])
        th, fallback, counts = calibrate(calibration_ids, calibration_scalar)
        thresholds.append(th)
        fallback_count.append(fallback)
        calibration_count.append({"selected": int(calibration_ids.size),
                                  "covered_children": int(np.count_nonzero(counts)),
                                  "fallback_children": fallback})
        local = np.sum(validation_scalar[:, None] >= th[validation_ids], axis=1)
        grandchildren = validation_ids.astype(np.int64) * 10 + local
        assert np.array_equal(grandchildren // 10, validation_ids)
        control_counts = np.bincount(validation_ids, minlength=1280)
        candidate_counts = np.bincount(grandchildren, minlength=12800)
        assert control_counts.sum() == candidate_counts.sum() == validation_ids.size
        candidates_by_parent = candidate_counts.reshape(1280, 10)
        hot_parents = control_counts >= 500
        assert hot_parents.any()
        worst_share = float(np.max(candidates_by_parent[hot_parents].max(axis=1)
                                   / control_counts[hot_parents]))
        control_summary, candidate_summary = summarize(control_counts), summarize(candidate_counts)
        validation.append({"layer": li, "control": control_summary,
                           "candidate": candidate_summary,
                           "candidate_to_control_max_load_ratio":
                               candidate_summary["max_to_mean"] / control_summary["max_to_mean"],
                           "hot_parent_count": int(hot_parents.sum()),
                           "hot_parent_worst_grandchild_share": worst_share})
    sidecar = export_sidecar(args.sidecar, projections, thresholds)
    budget(start, device)

    # Clone-factor parity is checked on complete BF16 logits after the
    # sidecar is frozen. The clone wrapper records grandchild IDs while
    # gathering the identical E1280 source B factor.
    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    teacher, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "teacher")
    parity_items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    expected_logits = []
    with torch.inference_mode():
        for item in parity_items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            expected_logits.append(model(ids, use_cache=False).logits.cpu())
    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    clone = make_clone_wrappers(
        model, original_mlps, parent_state["expert_state"],
        child_state["expert_state"], source_a, source_b,
        prefixes, projections, thresholds, device)
    parity_error = 0.0
    with torch.inference_mode():
        for item, expected in zip(parity_items, expected_logits):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            actual = model(ids, use_cache=False).logits.cpu()
            parity_error = max(parity_error, float((actual.float() - expected.float()).abs().max()))
            assert torch.equal(actual, expected), item["source_id"]
            for wrapper in clone:
                assert torch.equal(wrapper.last_grandchild // 10, wrapper.last_selected)
    del teacher, clone, model, control
    gc.collect()
    gates = {"full_model_clone_parity": parity_error == 0.0,
             "all_layers_load_ratio": all(row["candidate_to_control_max_load_ratio"] <= 1.25
                                      for row in validation),
             "all_layers_coverage": all(row["candidate"]["coverage"] >= 5000
                                    for row in validation),
             "hot_parent_share": all(row["hot_parent_worst_grandchild_share"] <= 0.25
                                     for row in validation)}
    result = {"experiment": "METH-139-quantile-third-tier-routing-screen",
              "source_bank_sha256": M136.EXACT_SHA,
              "third_projection_source_sha256": M136.THIRD_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "raw_data_sha256": M136.M15.TRAIN_FILE_SHA,
              "train_chat_sha256": M136.M56.TRAIN_CHAT_SHA,
              "long_teacher_sha256": M136.M107.LONG_TEACHER_SHA,
              "calibration_updates": [1, 128], "validation_updates": [129, 256],
              "projection_dimension_rule": "child_id % 32",
              "quantile_method": "numpy linear 10..90%, float32 thresholds, >= tie to right",
              "calibration_by_layer": calibration_count,
              "validation_by_layer": validation,
              "sidecar": sidecar, "parity_prompts": len(parity_items),
              "parity_logit_max_abs_error": parity_error,
              "gates": gates,
              "decision": "routing_screen_pass_for_native_cost_and_training"
                          if all(gates.values()) else "routing_screen_fail",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "worst_ratio": max(row["candidate_to_control_max_load_ratio"]
                                         for row in validation),
                      "worst_hot_share": max(row["hot_parent_worst_grandchild_share"]
                                             for row in validation),
                      "min_coverage": min(row["candidate"]["coverage"]
                                          for row in validation),
                      "sidecar": sidecar, "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
