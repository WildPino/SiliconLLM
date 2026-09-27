#!/usr/bin/env python3
"""Measure trained-route sensitivity and expert use at METH-45 checkpoints."""
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
import meth44_instruct_full_chat_smoke as M44


ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth45_instruct_continuation_result.json"
RESULT_SHA = "a48deede189f1ff0e91bf2a0db9876fbd246fc53c24dc70010f935ba84a0d636"
CHECKPOINT_SHA = {
    256: "3d5835eae07384ed6709897212c9d0e31734403f049b0aa254622bafe0b57c4c",
    512: "c988d26fa0a122db292cd56b253317ae6c0c8ce5fff9c1aedaf004ca2b8fb6c2",
}
DEV_PATH = M44.PROMPT_PATH
DEV_SHA = M44.PROMPT_SHA
MODEL_SHA = M44.MODEL_SHA
PERM_SEED = 451616
MAX_SECONDS = 5 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or peak > MAX_GPU_BYTES or rss > MAX_RSS_BYTES:
        raise RuntimeError(f"METH-46 budget: {elapsed:.1f}s, GPU {peak}, RSS {rss}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def load_checkpoint(path, wrappers, device, step):
    assert M15.M13.sha256(path) == CHECKPOINT_SHA[step]
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["updates"] == step
    assert state["source_sha256"] == MODEL_SHA
    assert state["teacher_sha256"] == M44.TEACHER_SHA
    assert len(state["expert_state"]) == M15.M13.L
    with torch.no_grad():
        for wrapper, values in zip(wrappers, state["expert_state"]):
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(values[key].to(device))


def count_routes(model, wrappers, raw_ids, dev_rows, device, start):
    counts = {kind: [torch.zeros(M15.E, dtype=torch.int64, device=device)
                     for _ in wrappers] for kind in ("raw", "chat")}
    active_kind = ["raw"]
    handles = []
    for li, wrapper in enumerate(wrappers):
        def hook(module, inputs, layer_id=li):
            x = inputs[0].reshape(-1, M15.M13.D)
            scores = F.linear(x.float(), module.router)
            selected = torch.topk(scores, M15.K, dim=-1).indices
            counts[active_kind[0]][layer_id] += torch.bincount(
                selected.flatten(), minlength=M15.E)
        handles.append(wrapper.register_forward_pre_hook(hook))
    try:
        M44.set_experts(wrappers, True)
        with torch.inference_mode():
            active_kind[0] = "raw"
            for i in range(raw_ids.shape[0]):
                model(raw_ids[i:i+1], use_cache=False)
                budget(start, device)
            active_kind[0] = "chat"
            for row in dev_rows:
                ids = torch.as_tensor(row["prompt_ids"], dtype=torch.long,
                                      device=device).unsqueeze(0)
                model(ids, use_cache=False)
                budget(start, device)
    finally:
        for handle in handles:
            handle.remove()
    summary = {}
    for kind, per_layer in counts.items():
        layers = []
        for count in per_layer:
            values = count.cpu().numpy().astype(np.float64)
            total = values.sum()
            assert total > 0
            probs = values / total
            positive = probs[probs > 0]
            layers.append({"selected_ids": int(total),
                           "experts_used": int((values > 0).sum()),
                           "max_to_mean_load": float(values.max() / values.mean()),
                           "normalized_entropy": float(
                               -(positive * np.log(positive)).sum() / math.log(M15.E)),
                           "counts": values.astype(np.int64).tolist()})
        expected = (raw_ids.numel() if kind == "raw" else
                    sum(len(row["prompt_ids"]) for row in dev_rows)) * M15.K
        assert all(row["selected_ids"] == expected for row in layers)
        summary[kind] = {"input_positions": expected // M15.K,
                         "min_experts_used": min(r["experts_used"] for r in layers),
                         "max_load_to_mean": max(r["max_to_mean_load"] for r in layers),
                         "min_normalized_entropy": min(r["normalized_entropy"] for r in layers),
                         "layers": layers}
    return summary


def factor_norms(wrappers):
    layers = []
    for wrapper in wrappers:
        norms = wrapper.b.detach().float().flatten(1).norm(dim=1).cpu().numpy()
        median = float(np.median(norms))
        layers.append({"min": float(norms.min()), "median": median,
                       "max": float(norms.max()),
                       "nonzero": int((norms > 1e-9).sum()),
                       "above_tenth_median": int((norms > 0.1 * median).sum())})
    return {"min_nonzero": min(x["nonzero"] for x in layers),
            "min_above_tenth_median": min(x["above_tenth_median"] for x in layers),
            "layers": layers}


def route_sensitivity(model, wrappers, ids, byts, device, start):
    intact = M15.bpb(model, ids, byts, wrappers, True)
    originals = [wrapper.router.detach().clone() for wrapper in wrappers]
    rng = np.random.default_rng(PERM_SEED)
    with torch.no_grad():
        for wrapper, values in zip(wrappers, originals):
            permutation = torch.as_tensor(rng.permutation(M15.E),
                                          dtype=torch.long, device=device)
            wrapper.router.copy_(values[permutation])
    permuted = M15.bpb(model, ids, byts, wrappers, True)
    with torch.no_grad():
        for wrapper, values in zip(wrappers, originals):
            wrapper.router.copy_(values)
    restored = M15.bpb(model, ids, byts, wrappers, True)
    assert abs(restored - intact) <= 1e-6
    budget(start, device)
    return {"intact_bpb": intact, "permuted_bpb": permuted,
            "restored_bpb": restored,
            "permuted_minus_intact_bpb": permuted - intact,
            "permutation_seed": PERM_SEED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M17.sha(RESULT_PATH.read_bytes()) == RESULT_SHA
    assert M17.sha(DEV_PATH.read_bytes()) == DEV_SHA
    training = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert training["decision"] == "stop_instruct_continuation"
    assert training["last_applied_update"] == 512
    assert training["interim_evaluations"][1]["dev_top1"]["agreement"] >= 0.95
    assert training["interim_evaluations"][2]["dev_top1"]["agreement"] < 0.95
    dev = json.loads(DEV_PATH.read_text(encoding="utf-8"))
    assert len(dev["rows"]) == 24
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = hf_hub_download(M42.MODEL, "model.safetensors",
                             revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source) == MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    eval_ids, eval_bytes, meta = M15.M13.C.get_slice(tokenizer, "heldout", 4, 256, 271828)
    assert meta["ids_sha256"] == M15.EVAL_IDS_SHA and int(eval_bytes.sum()) == 4588
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    eval_ids = torch.as_tensor(eval_ids, dtype=torch.long, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    model.config.use_cache = False
    donor_bpb = M15.bpb(model, eval_ids, eval_bytes, wrappers, False)
    assert abs(donor_bpb - 0.9712552075318387) <= 1e-5
    rows = {}
    for step in (256, 512):
        checkpoint = training["checkpoints"][str(step)]
        assert checkpoint["sha256"] == CHECKPOINT_SHA[step]
        load_checkpoint(checkpoint["path"], wrappers, device, step)
        sensitivity = route_sensitivity(model, wrappers, eval_ids,
                                        eval_bytes, device, start)
        counts = count_routes(model, wrappers, eval_ids, dev["rows"], device, start)
        norms = factor_norms(wrappers)
        rows[str(step)] = {"checkpoint_sha256": CHECKPOINT_SHA[step],
                           "route_sensitivity": sensitivity,
                           "route_counts": counts,
                           "output_factor_norms": norms,
                           "chat_development_top1": training["interim_evaluations"][
                               1 if step == 256 else 2]["dev_top1"]}
        print(json.dumps({"step": step,
                          "route_sensitivity": sensitivity,
                          "route_counts": {kind: {key: value for key, value in data.items()
                                                 if key != "layers"}
                                           for kind, data in counts.items()},
                          "min_nonzero_factors": norms["min_nonzero"]}), flush=True)
        budget(start, device)
    pass_256 = rows["256"]["route_sensitivity"]["permuted_minus_intact_bpb"] >= 0.002
    runtime = budget(start, device)
    result = {"experiment": "METH-46-route-utility-diagnostic",
              "model_sha256": MODEL_SHA,
              "training_result_sha256": RESULT_SHA,
              "development_manifest_sha256": DEV_SHA,
              "raw_eval_ids_sha256": M15.EVAL_IDS_SHA,
              "rows": rows,
              "decision": "retain_conditional_geometry_for_retention_repair"
                          if pass_256 else "deprioritize_this_residual_geometry",
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__,
                          "numpy": np.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
