#!/usr/bin/env python3
"""Check the cloned E12800 token hash inside every expert forward."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth136_sparse_experts  # inserts the S1 import path
import meth136_matched_sparse_train as M136
import meth142_token_hash_route_screen as M142
from meth134_sparse_e12800_training import SparseCollector


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
M143_RESULT = DOC / "meth143_fresh_hash_route_audit_result.json"
M143_SHA = "d13aeb90b8211fc5be6a1289f0209d1f846691b9be7f4621a06e0845538d749f"
M143_MANIFEST = DOC / "meth143_fresh_hash_route_manifest.json"
M143_MANIFEST_SHA = "e6c4fb6a92929eb29332c55a0887adb6627c9528f80b21c09f7759139070ccff"
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 100_000_000


def budget(start, device):
    report = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (report["seconds"] > MAX_SECONDS or report["rss_bytes"] > MAX_RSS
            or report["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-144 model resource stop: {report}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert M136.digest(M143_RESULT) == M143_SHA
    assert M136.digest(M143_MANIFEST) == M143_MANIFEST_SHA
    assert json.loads(M143_RESULT.read_text(encoding="utf-8"))["decision"] == (
        "fresh_route_load_pass_full_model_and_native_cost_pending")
    assert M142.hash_grandchildren(
        np.array([123]), np.array([45]), np.array([67]),
        np.array([[89, 89, 89, 89]]), 3)[0, 0] == 899
    bound = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    fresh = json.loads(M143_MANIFEST.read_text(encoding="utf-8"))["items"][:8]
    cases = [("bound", i, row["prompt_ids"]) for i, row in enumerate(bound)]
    cases += [("fresh", i, row["prompt_ids"]) for i, row in enumerate(fresh)]

    torch.set_num_threads(6)
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
    for wrapper in teacher:
        wrapper.capture = True
    references = []
    with torch.inference_mode():
        for source, index, tokens in cases:
            ids = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]
            logits = model(ids, use_cache=False).logits.cpu()
            references.append({"source": source, "index": index,
                               "tokens": len(tokens), "logits": logits,
                               "parents": [w.parent_routes[-1] for w in teacher],
                               "scores": [w.parent_scores[-1] for w in teacher],
                               "children": [w.child_ids[-1] for w in teacher]})
            for wrapper in teacher:
                wrapper.parent_routes.clear()
                wrapper.parent_scores.clear()
                wrapper.child_ids.clear()
            if index in (0, 7):
                print(json.dumps({"stage": "teacher", "source": source,
                                  "item": index, "budget": budget(start, device)}), flush=True)
    for layer, base in zip(model.model.layers, original_mlps):
        layer.mlp = base
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    wrappers, banks = M136.make_wrappers(
        model, parent_state["expert_state"], child_state["expert_state"],
        source_a, source_b, prefixes, device, "hash")
    del parent_state, child_state, source_a, source_b
    gc.collect()
    assert len(wrappers) == 24 and len(banks) == 24
    assert all(bank.shape == (12800, 896, 8) for bank in banks)
    assert all(w.shared_a.shape == (128, 8, 896) for w in wrappers)
    # The FP32 CPU rows originate from exactly rounded BF16 source bytes.
    cloned_bytes = all(torch.equal(bank.view(1280, 10, 896, 8)[:, 0].to(torch.bfloat16),
                                   bank.view(1280, 10, 896, 8)[:, 9].to(torch.bfloat16))
                       for bank in banks)
    assert cloned_bytes
    budget(start, device)

    missing_context_rejected = False
    try:
        with torch.inference_mode():
            model(torch.as_tensor(cases[0][2], dtype=torch.long, device=device)[None],
                  use_cache=False)
    except RuntimeError as error:
        missing_context_rejected = "token context must be set" in str(error)
    assert missing_context_rejected

    selected = [set() for _ in wrappers]
    records = []
    with torch.inference_mode():
        for case, reference in zip(cases, references):
            source, index, tokens = case
            for wrapper in wrappers:
                wrapper.set_token_context(tokens)
            ids = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]
            logits = model(ids, use_cache=False).logits.cpu()
            assert torch.equal(logits, reference["logits"]), (source, index)
            matches = []
            for li, wrapper in enumerate(wrappers):
                assert torch.equal(wrapper.last_parents.cpu(), reference["parents"][li])
                assert torch.equal(wrapper.last_scores.cpu(), reference["scores"][li])
                child = wrapper.last_children.cpu()
                assert torch.equal(child, reference["children"][li])
                grand = wrapper.last_selected.cpu()
                assert torch.equal(grand.div(10, rounding_mode="floor"), child)
                expected = M142.hash_grandchildren(
                    np.asarray(tokens, dtype=np.int64),
                    np.r_[0, np.asarray(tokens[:-1], dtype=np.int64)],
                    np.arange(len(tokens), dtype=np.int64),
                    child.numpy().astype(np.int64), li)
                assert np.array_equal(grand.numpy(), expected)
                selected[li].update(grand.flatten().tolist())
                matches.append(True)
            records.append({"source": source, "index": index, "tokens": len(tokens),
                            "logits_bitwise_equal": True, "route_layers_equal": sum(matches)})
            if index in (0, 7):
                print(json.dumps({"stage": "hash", "source": source,
                                  "item": index, "budget": budget(start, device)}), flush=True)
    assert min(len(s) for s in selected) > 10

    # A later B-only run uses non-reentrant gradient checkpointing. Verify
    # that its recomputation sees the same explicit token context.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    short = cases[0][2][:8]
    for wrapper in wrappers:
        wrapper.set_token_context(short)
        wrapper.collector = SparseCollector()
    with torch.enable_grad():
        ids = torch.as_tensor(short, dtype=torch.long, device=device)[None]
        loss = model(ids, use_cache=False).logits.float()[..., 0].sum()
        assert bool(torch.isfinite(loss))
        loss.backward()
    checkpoint_parts = [len(w.collector.parts) for w in wrappers]
    assert all(count > 0 for count in checkpoint_parts)
    for wrapper in wrappers:
        wrapper.collector = None
    result = {"experiment": "METH-144-integrated-hash-parity",
              "meth143_result_sha256": M143_SHA,
              "exact_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "hash_golden": 899, "cases": records,
              "cloned_bf16_rows_identical": cloned_bytes,
              "missing_context_rejected": missing_context_rejected,
              "min_distinct_grandchildren_by_layer": min(map(len, selected)),
              "checkpoint_recomputation_collector_parts_min": min(checkpoint_parts),
              "all_logits_bitwise_equal": True,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < MAX_DISK
    print(json.dumps({"decision": "integrated_clone_parity_pass",
                      "cases": len(records), "min_distinct": result["min_distinct_grandchildren_by_layer"],
                      "checkpoint_parts_min": min(checkpoint_parts),
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
