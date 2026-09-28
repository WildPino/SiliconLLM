#!/usr/bin/env python3
"""Evaluate frozen token-hash route load on new source-disjoint documents."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth139_quantile_third_router as M139
import meth142_token_hash_route_screen as M142


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DOC / "meth143_fresh_hash_route_manifest.json"
MANIFEST_SHA = "e6c4fb6a92929eb29332c55a0887adb6627c9528f80b21c09f7759139070ccff"
M142_RESULT = DOC / "meth142_token_hash_route_screen_result.json"
M142_SHA = "e7a302671fc9e24290e39b8e0c1864c7b7b053f26057fa9c357d5f201e0decd3"
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000


def budget(start, device):
    report = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (report["seconds"] > MAX_SECONDS or report["rss_bytes"] > MAX_RSS
            or report["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-143 resource stop: {report}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert M136.digest(MANIFEST) == MANIFEST_SHA
    assert M136.digest(M142_RESULT) == M142_SHA
    old = json.loads(M142_RESULT.read_text(encoding="utf-8"))
    assert old["decision"] == "development_load_pass_fresh_route_check_required"
    assert old["hash_seed"] == 142142
    assert old["hash_constants_hex"] == [f"{int(c):016x}" for c in M142.CONSTANTS]
    assert old["hash_golden"] == {"token": 123, "previous": 45,
                                   "position": 67, "child": 89,
                                   "layer": 3, "grandchild": 899}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["meth142_result_sha256"] == M142_SHA
    assert manifest["selected_counts"] == {"code": 8, "prose": 8,
                                            "technical_general": 8}
    items = manifest["items"]
    assert len(items) == 24
    for item in items:
        assert M136.M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        assert M136.M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item["document_ids_sha256"]
        assert M136.M17.sha(np.asarray(item["prompt_ids"], dtype=np.int32).tobytes()) == item["prompt_ids_sha256"]
    golden = M142.hash_grandchildren(
        np.array([123]), np.array([45]), np.array([67]),
        np.array([[89, 89, 89, 89]]), 3)
    assert int(golden[0, 0]) == 899
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
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    teacher, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "teacher")
    parity_items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    expected = []
    with torch.inference_mode():
        for item in parity_items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            expected.append(model(ids, use_cache=False).logits.cpu())
    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    control, _ = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "control")
    parity_error = 0.0
    with torch.inference_mode():
        for item, reference in zip(parity_items, expected):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            actual = model(ids, use_cache=False).logits.cpu()
            parity_error = max(parity_error, float((actual.float() - reference.float()).abs().max()))
            assert torch.equal(actual, reference)
    del teacher, parent_state, child_state, source_a, source_b
    gc.collect()
    budget(start, device)

    def measure(width):
        parent_counts = [np.zeros(1280, dtype=np.int64) for _ in range(24)]
        grand_counts = [np.zeros(12800, dtype=np.int64) for _ in range(24)]
        current = {"tokens": None, "previous": None, "positions": None}
        hooks = []
        for li, wrapper in enumerate(control):
            def count(module, _inputs, _output, layer_id=li):
                child = module.last_selected.cpu().numpy().astype(np.int64)
                grand = M142.hash_grandchildren(
                    current["tokens"], current["previous"],
                    current["positions"], child, layer_id)
                parent_counts[layer_id] += np.bincount(child.flatten(), minlength=1280)
                grand_counts[layer_id] += np.bincount(grand.flatten(), minlength=12800)
            hooks.append(wrapper.register_forward_hook(count))
        total_tokens, windows = 0, 0
        with torch.inference_mode():
            for di, item in enumerate(items):
                sequence = item["document_ids"]
                for first in range(0, len(sequence), width):
                    window = np.asarray(sequence[first:first + width], dtype=np.int64)
                    current["tokens"] = window
                    current["previous"] = np.r_[0, window[:-1]]
                    current["positions"] = np.arange(window.size, dtype=np.int64)
                    ids = torch.as_tensor(window, dtype=torch.long, device=device)[None]
                    model(ids, use_cache=False)
                    total_tokens += window.size
                    windows += 1
                if (di + 1) % 8 == 0:
                    print(json.dumps({"width": width, "documents": di + 1,
                                      "budget": budget(start, device)}), flush=True)
        for hook in hooks:
            hook.remove()
        layers = []
        for li, (parent, grand) in enumerate(zip(parent_counts, grand_counts)):
            assert int(parent.sum()) == int(grand.sum()) == 4 * total_tokens
            p, g = M139.summarize(parent), M139.summarize(grand)
            heavy = parent >= 250
            assert heavy.any()
            share = float(np.max(grand.reshape(1280, 10)[heavy].max(axis=1) / parent[heavy]))
            layers.append({"layer": li, "control": p, "candidate": g,
                           "candidate_to_control_max_load_ratio":
                               g["max_to_mean"] / p["max_to_mean"],
                           "hot_parent_count": int(heavy.sum()),
                           "hot_parent_worst_grandchild_share": share})
        return {"documents": len(items), "windows": windows,
                "tokens": total_tokens, "per_layer": layers}

    widths = {str(width): measure(width) for width in (128, 512)}
    del model, control
    gc.collect()
    gates = {"teacher_control_bf16_parity": parity_error == 0.0,
             "load_ratio": all(row["candidate_to_control_max_load_ratio"] <= 1.25
                               for value in widths.values() for row in value["per_layer"]),
             "coverage": all(row["candidate"]["coverage"] >= 4000
                             for value in widths.values() for row in value["per_layer"]),
             "hot_parent_share": all(row["hot_parent_worst_grandchild_share"] <= 0.25
                                     for value in widths.values() for row in value["per_layer"])}
    summary = {width: {"tokens": value["tokens"],
                       "worst_load_ratio": max(row["candidate_to_control_max_load_ratio"]
                                               for row in value["per_layer"]),
                       "worst_hot_share": max(row["hot_parent_worst_grandchild_share"]
                                              for row in value["per_layer"]),
                       "minimum_coverage": min(row["candidate"]["coverage"]
                                               for row in value["per_layer"])}
               for width, value in widths.items()}
    result = {"experiment": "METH-143-fresh-token-hash-route-audit",
              "manifest_sha256": MANIFEST_SHA,
              "meth142_result_sha256": M142_SHA,
              "source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "hash_seed": 142142, "hash_golden": 899,
              "parity_prompts": len(parity_items),
              "teacher_control_bf16_logit_max_abs_error": parity_error,
              "scope": "Offline grandchild assignment; no hash-routed full-model logits",
              "widths": widths, "summary": summary, "gates": gates,
              "decision": "fresh_route_load_pass_full_model_and_native_cost_pending"
                          if all(gates.values()) else "fresh_route_load_fail",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < MAX_DISK
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "summary": summary, "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
