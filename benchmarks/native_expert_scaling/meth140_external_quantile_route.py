#!/usr/bin/env python3
"""Fit decile routing on all training draws and screen external route load."""

import argparse
import gc
import json
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth139_quantile_third_router as M139


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFESTS = {
    "meth121": (DOC / "meth121_zero_mean_child_external_manifest.json",
                "7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366"),
    "meth133": (DOC / "meth133_q15_external_manifest.json",
                "ec6f839a2c9b1803dc8d5658aafa8fd9191b59fa2607961850f5a8666bc35674"),
}
HEADER = struct.Struct("<8s6I")
MAGIC = b"M140QT01"
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS
            or record["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-140 resource stop: {record}")
    return record


def export_sidecar(path, projections, thresholds):
    assert not path.exists() and len(projections) == len(thresholds) == 24
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(HEADER.pack(MAGIC, 24, 896, 32, 1280, 10, 140140))
        for projection, layer_thresholds in zip(projections, thresholds):
            stream.write(projection.numpy().astype("<f4", copy=False).tobytes())
            stream.write(layer_thresholds.astype("<f4", copy=False).tobytes())
    layer_bytes = (32 * 896 + 1280 * 9) * 4
    expected = HEADER.size + 24 * layer_bytes
    assert path.stat().st_size == expected and expected < MAX_DISK
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert HEADER.unpack_from(data) == (MAGIC, 24, 896, 32, 1280, 10, 140140)
    for li, (projection, layer_thresholds) in enumerate(zip(projections, thresholds)):
        offset = HEADER.size + li * layer_bytes
        stored_projection = np.frombuffer(data, dtype="<f4", count=32 * 896,
                                          offset=offset).reshape(32, 896)
        stored_thresholds = np.frombuffer(data, dtype="<f4", count=1280 * 9,
                                         offset=offset + 32 * 896 * 4).reshape(1280, 9)
        assert np.array_equal(stored_projection, projection.numpy())
        assert np.array_equal(stored_thresholds, layer_thresholds)
    return {"path": str(path.resolve()), "bytes": expected,
            "sha256": M136.digest(path), "readback_exact": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.sidecar.exists()
    manifests = {}
    for name, (path, sha) in MANIFESTS.items():
        assert M136.digest(path) == sha
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert len(manifest["items"]) == 24
        manifests[name] = manifest["items"]
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
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    control, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "control")
    projections = [entry[0] for entry in third]
    gpu_projections = [p.to(device) for p in projections]
    sample_ids, sample_scalar = [[] for _ in range(24)], [[] for _ in range(24)]
    hooks = []
    for li, wrapper in enumerate(control):
        def collect(module, inputs, _output, layer_id=li):
            flat = inputs[0].reshape(-1, 896)
            children = module.last_selected
            q = F.linear(flat.float(), gpu_projections[layer_id])
            scalar = q.gather(1, children.remainder(32).long())
            sample_ids[layer_id].append(children.flatten().cpu().numpy().astype(np.int32))
            sample_scalar[layer_id].append(scalar.flatten().cpu().numpy().astype(np.float32))
        hooks.append(wrapper.register_forward_hook(collect))
    with torch.inference_mode():
        for draw in draws:
            raw = raw_ids[draw["raw_row"],
                          draw["raw_offset"]:draw["raw_offset"] + M136.M15.SEQ]
            _, chat_ids, _ = chat[draw["chat_index"]]
            for sequence in (raw, chat_ids):
                ids = torch.as_tensor(sequence, dtype=torch.long, device=device)[None]
                model(ids[:, :-1], use_cache=False)
            if draw["update"] in (64, 128, 192, 256):
                print(json.dumps({"calibration_update": draw["update"],
                                  "budget": budget(start, device)}), flush=True)
    for hook in hooks:
        hook.remove()
    del hooks, gpu_projections
    thresholds, calibration = [], []
    for ids, values in zip(sample_ids, sample_scalar):
        flat_ids, flat_values = np.concatenate(ids), np.concatenate(values)
        th, fallback, counts = M139.calibrate(flat_ids, flat_values)
        thresholds.append(th)
        calibration.append({"selections": int(flat_ids.size),
                            "covered_children": int(np.count_nonzero(counts)),
                            "fallback_children": fallback})
    sidecar = export_sidecar(args.sidecar, projections, thresholds)
    budget(start, device)

    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    teacher, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "teacher")
    parity_items = manifests["meth121"][:8]
    expected_logits = []
    with torch.inference_mode():
        for item in parity_items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            expected_logits.append(model(ids, use_cache=False).logits.cpu())
    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    clone = M139.make_clone_wrappers(
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
    del teacher, control, parent_state, child_state, source_a, source_b
    gc.collect()
    budget(start, device)

    rows = {}
    for name, items in manifests.items():
        control_counts = [np.zeros(1280, dtype=np.int64) for _ in range(24)]
        candidate_counts = [np.zeros(12800, dtype=np.int64) for _ in range(24)]
        total_tokens = 0
        capture = []
        for li, wrapper in enumerate(clone):
            def count(module, _inputs, _output, layer_id=li):
                child = module.last_selected.flatten().cpu().numpy().astype(np.int64)
                grand = module.last_grandchild.flatten().cpu().numpy().astype(np.int64)
                assert np.array_equal(grand // 10, child)
                control_counts[layer_id] += np.bincount(child, minlength=1280)
                candidate_counts[layer_id] += np.bincount(grand, minlength=12800)
            capture.append(wrapper.register_forward_hook(count))
        with torch.inference_mode():
            for di, item in enumerate(items):
                tokens = item["document_ids"]
                for first in range(0, len(tokens), 512):
                    window = tokens[first:first + 512]
                    ids = torch.as_tensor(window, dtype=torch.long, device=device)[None]
                    model(ids, use_cache=False)
                    total_tokens += len(window)
                if (di + 1) % 8 == 0:
                    print(json.dumps({"manifest": name, "documents": di + 1,
                                      "budget": budget(start, device)}), flush=True)
        for hook in capture:
            hook.remove()
        per_layer = []
        for li, (parent, grand) in enumerate(zip(control_counts, candidate_counts)):
            assert int(parent.sum()) == int(grand.sum()) == 4 * total_tokens
            parent_summary = M139.summarize(parent)
            grand_summary = M139.summarize(grand)
            heavy = parent >= 250
            assert heavy.any()
            share = float(np.max(grand.reshape(1280, 10)[heavy].max(axis=1) / parent[heavy]))
            per_layer.append({"layer": li, "control": parent_summary,
                              "candidate": grand_summary,
                              "candidate_to_control_max_load_ratio":
                                  grand_summary["max_to_mean"] / parent_summary["max_to_mean"],
                              "hot_parent_count": int(heavy.sum()),
                              "hot_parent_worst_grandchild_share": share})
        rows[name] = {"documents": len(items), "document_tokens": total_tokens,
                      "per_layer": per_layer}
        budget(start, device)
    del clone, model
    gc.collect()
    gates = {"clone_parity": parity_error == 0.0,
             "load_ratio": all(row["candidate_to_control_max_load_ratio"] <= 1.25
                               for result in rows.values() for row in result["per_layer"]),
             "coverage": all(row["candidate"]["coverage"] >= 4000
                             for result in rows.values() for row in result["per_layer"]),
             "hot_parent_share": all(row["hot_parent_worst_grandchild_share"] <= 0.25
                                     for result in rows.values() for row in result["per_layer"])}
    result = {"experiment": "METH-140-full-calibration-external-route-screen",
              "source_bank_sha256": M136.EXACT_SHA,
              "third_projection_source_sha256": M136.THIRD_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "calibration_update_count": 256,
              "calibration_by_layer": calibration,
              "routing_manifest_sha256": {name: sha for name, (_, sha) in MANIFESTS.items()},
              "projection_dimension_rule": "child_id % 32",
              "quantile_method": "numpy linear 10..90%, float32 thresholds, >= tie to right",
              "sidecar": sidecar, "parity_prompts": len(parity_items),
              "parity_logit_max_abs_error": parity_error,
              "routing_validation": rows, "gates": gates,
              "decision": "routing_screen_pass_for_native_cost"
                          if all(gates.values()) else "routing_screen_fail",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "worst_load_ratio": max(r["candidate_to_control_max_load_ratio"]
                                              for v in rows.values() for r in v["per_layer"]),
                      "worst_hot_share": max(r["hot_parent_worst_grandchild_share"]
                                             for v in rows.values() for r in v["per_layer"]),
                      "minimum_coverage": min(r["candidate"]["coverage"]
                                              for v in rows.values() for r in v["per_layer"]),
                      "sidecar": sidecar, "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
