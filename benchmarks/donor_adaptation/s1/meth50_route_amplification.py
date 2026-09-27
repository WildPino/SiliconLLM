#!/usr/bin/env python3
"""Diagnose whether packed factors alter later fine-route identities."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
from safetensors import safe_open
import torch
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth48_int8_factor_bank as M48
import meth49_int8_row_factor_bank as M49


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "results/native_expert_scaling/meth49_e128_factor_int8_row.safetensors"
ARTIFACT_SHA = "f3039decaec0600ccae9f93b6c11cb063298a53b469b39e2cbcdba92aceef196"
M49_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth49_int8_row_factor_bank_result.json"
M49_RESULT_SHA = "822d07450f3b9cb2241fea5f19994494417d648e4ce39a7d7f9139efb50b2e4c"
MAX_SECONDS = 10 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or peak > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-50 budget: {elapsed:.1f}s, RSS {rss}, GPU {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def install_capture(wrappers, destination, margins=None, counts=None, stream_hash=None):
    handles = []
    for li, wrapper in enumerate(wrappers):
        def hook(module, inputs, layer_id=li):
            flat = inputs[0].reshape(-1, M15.M13.D)
            scores = F.linear(flat.float(), module.router)
            top = torch.topk(scores, 5, dim=-1)
            chosen = top.indices[:, :M15.K].detach().cpu()
            destination[layer_id] = chosen
            if margins is not None:
                margins[layer_id].extend((top.values[:, 3] - top.values[:, 4]).cpu().tolist())
            if counts is not None:
                counts[layer_id] += torch.bincount(chosen.flatten(), minlength=M15.E)
            if stream_hash is not None:
                stream_hash.update(chosen.to(torch.int32).numpy().tobytes(order="C"))
        handles.append(wrapper.register_forward_pre_hook(hook))
    return handles


def forced_forward(wrapper, chosen):
    def forward(x):
        dense = wrapper.base(x)
        flat = x.reshape(-1, M15.M13.D)
        scores = F.linear(flat.float(), wrapper.router)
        ids = chosen.to(flat.device)
        assert tuple(ids.shape) == (flat.shape[0], M15.K)
        gate = F.softmax(scores.gather(1, ids), dim=-1).to(flat.dtype)
        a = wrapper.a[ids].to(flat.dtype)
        b = wrapper.b[ids].to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b)
        residual = (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)
        return dense + residual
    return forward


def run_forced(model, wrappers, original_forwards, routes, ids):
    try:
        for wrapper, source, chosen in zip(wrappers, original_forwards, routes):
            assert source is not None and chosen is not None
            wrapper.forward = forced_forward(wrapper, chosen)
        return model(ids, use_cache=False).logits
    finally:
        for wrapper, original in zip(wrappers, original_forwards):
            wrapper.forward = original


def compare_routes(original, packed, layer_counts):
    first_changed = None
    for li, (a, b) in enumerate(zip(original, packed)):
        assert a is not None and b is not None and a.shape == b.shape
        overlap = (a[:, :, None] == b[:, None, :]).any(dim=2).sum(dim=1)
        exact = (a.sort(dim=1).values == b.sort(dim=1).values).all(dim=1)
        layer_counts[li]["matched_ids"] += int(overlap.sum())
        layer_counts[li]["selected_ids"] += a.numel()
        layer_counts[li]["matched_sets"] += int(exact.sum())
        layer_counts[li]["positions"] += a.shape[0]
        if first_changed is None and not bool(exact.all()):
            first_changed = li
    return first_changed


def summarize_load(counts):
    result = []
    for count in counts:
        values = count.numpy().astype(np.float64)
        total = values.sum()
        positive = values[values > 0] / total
        result.append({"selected_ids": int(total),
                       "experts_used": int((values > 0).sum()),
                       "max_to_mean_load": float(values.max() / values.mean()),
                       "normalized_entropy": float(-(positive * np.log(positive)).sum() /
                                                   math.log(M15.E)),
                       "counts": values.astype(np.int64).tolist()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path, digest in ((M48.CHECKPOINT, M48.CHECKPOINT_SHA),
                         (M48.MANIFEST, M48.MANIFEST_SHA),
                         (ARTIFACT, ARTIFACT_SHA),
                         (M49_RESULT, M49_RESULT_SHA)):
        assert M15.M13.sha256(path) == digest, path
    manifest = json.loads(M48.MANIFEST.read_text(encoding="utf-8"))
    previous = json.loads(M49_RESULT.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 12 and len(previous["int8_generation_rows"]) == 12
    for item in items:
        assert M17.sha(np.asarray(item["prompt_ids"], dtype=np.int32).tobytes()) == item["prompt_ids_sha256"]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    state = torch.load(M48.CHECKPOINT, map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M44.MODEL_SHA
    assert state["teacher_sha256"] == M44.TEACHER_SHA
    assert len(state["expert_state"]) == M15.M13.L
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M44.MODEL_SHA
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for factor in ("a", "b", "router"):
                getattr(wrapper, factor).copy_(state["expert_state"][li][factor].to(device))
            wrappers.append(wrapper)
    original_forwards = [wrapper.forward for wrapper in wrappers]
    model.config.use_cache = False
    M44.set_experts(wrappers, True)
    margins = [[] for _ in wrappers]
    loads = [torch.zeros(M15.E, dtype=torch.int64) for _ in wrappers]
    original_hash = hashlib.sha256()
    original = []
    with torch.inference_mode():
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None, :]
            routes = [None] * len(wrappers)
            handles = install_capture(wrappers, routes, margins, loads, original_hash)
            try:
                logits = model(ids, use_cache=False).logits
            finally:
                for handle in handles:
                    handle.remove()
            forced_logits = run_forced(model, wrappers, original_forwards, routes, ids)
            assert torch.equal(logits, forced_logits), item["source_id"]
            original.append({"source_id": item["source_id"],
                             "category": item["category"],
                             "routes": routes,
                             "top1": logits.argmax(dim=-1).squeeze(0).cpu()})
            budget(start, device)
            print(f"original {len(original)}/{len(items)}", flush=True)

    with safe_open(str(ARTIFACT), framework="pt", device="cpu") as archive:
        M49.restore_factors(wrappers, archive, device)
    packed_hash = hashlib.sha256()
    layer_counts = [{"matched_ids": 0, "selected_ids": 0,
                     "matched_sets": 0, "positions": 0} for _ in wrappers]
    rows = []
    with torch.inference_mode():
        for item, baseline in zip(items, original):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None, :]
            routes = [None] * len(wrappers)
            handles = install_capture(wrappers, routes, stream_hash=packed_hash)
            try:
                packed_logits = model(ids, use_cache=False).logits
            finally:
                for handle in handles:
                    handle.remove()
            forced_logits = run_forced(model, wrappers, original_forwards,
                                       baseline["routes"], ids)
            original_top = baseline["top1"]
            packed_top = packed_logits.argmax(dim=-1).squeeze(0).cpu()
            forced_top = forced_logits.argmax(dim=-1).squeeze(0).cpu()
            assert original_top.shape == packed_top.shape == forced_top.shape
            first_changed = compare_routes(baseline["routes"], routes, layer_counts)
            rows.append({"source_id": item["source_id"], "category": item["category"],
                         "positions": original_top.numel(),
                         "ordinary_matching": int((original_top == packed_top).sum()),
                         "forced_matching": int((original_top == forced_top).sum()),
                         "first_changed_layer": first_changed})
            budget(start, device)
            print(f"packed {len(rows)}/{len(items)}", flush=True)
    matching = sum(row["ordinary_matching"] for row in rows)
    forced_matching = sum(row["forced_matching"] for row in rows)
    positions = sum(row["positions"] for row in rows)
    assert (matching, positions) == (2002, 2091)
    assert matching == previous["prompt_summary"]["pooled"]["matching"]
    for layer in layer_counts:
        layer["id_overlap"] = layer["matched_ids"] / layer["selected_ids"]
        layer["set_agreement"] = layer["matched_sets"] / layer["positions"]
    all_margins = np.asarray([value for layer in margins for value in layer], dtype=np.float64)
    assert len(all_margins) == positions * len(wrappers)
    load_summary = summarize_load(loads)
    runtime = budget(start, device)
    result = {"experiment": "METH-50-route-amplification",
              "checkpoint_sha256": M48.CHECKPOINT_SHA,
              "model_sha256": M44.MODEL_SHA,
              "manifest_sha256": M48.MANIFEST_SHA,
              "factor_artifact_sha256": ARTIFACT_SHA,
              "m49_result_sha256": M49_RESULT_SHA,
              "prompt_rows": rows,
              "original_route_stream_sha256": original_hash.hexdigest(),
              "ordinary_int8_route_stream_sha256": packed_hash.hexdigest(),
              "ordinary_top1": {"matching": matching, "positions": positions,
                                "agreement": matching / positions},
              "forced_top1": {"matching": forced_matching, "positions": positions,
                              "agreement": forced_matching / positions},
              "route": {"layer_counts": layer_counts,
                        "pooled_id_overlap": sum(x["matched_ids"] for x in layer_counts) /
                        sum(x["selected_ids"] for x in layer_counts),
                        "pooled_set_agreement": sum(x["matched_sets"] for x in layer_counts) /
                        sum(x["positions"] for x in layer_counts),
                        "first_changed_layer_by_prompt": [row["first_changed_layer"] for row in rows]},
              "original_fourth_fifth_margin": {
                  "min": float(all_margins.min()),
                  "p01": float(np.quantile(all_margins, 0.01)),
                  "median": float(np.median(all_margins)),
                  "p99": float(np.quantile(all_margins, 0.99))},
              "original_load": {"min_experts_used": min(x["experts_used"] for x in load_summary),
                                "max_to_mean_load": max(x["max_to_mean_load"] for x in load_summary),
                                "layers": load_summary},
              "decision": "route_changes_dominate_on_viewed_prompts"
                          if forced_matching / positions >= 0.99
                          else "factor_arithmetic_also_fails_with_original_ids",
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("ordinary_top1", "forced_top1",
                                            "original_fourth_fifth_margin", "decision", "runtime")}
                     | {"pooled_id_overlap": result["route"]["pooled_id_overlap"],
                        "pooled_set_agreement": result["route"]["pooled_set_agreement"],
                        "min_experts_used": result["original_load"]["min_experts_used"],
                        "max_to_mean_load": result["original_load"]["max_to_mean_load"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
