#!/usr/bin/env python3
"""Repeated factor-pair permutations on METH-72 external and raw sources."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth71_balanced_product_key as M71
import meth72_e1280_external_audit as M72


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth71_balanced_e1280_result.json"
SEEDS = list(range(730000, 730008))
CATEGORIES = ("pooled", "code", "prose", "technical_general")
MAX_SECONDS = 20 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    gpu = torch.cuda.max_memory_allocated(device)
    rss = psutil.Process().memory_info().rss
    if elapsed > MAX_SECONDS or gpu > MAX_GPU or rss > MAX_RSS:
        raise RuntimeError(f"METH-73 budget: {elapsed:.1f}s GPU={gpu} RSS={rss}")
    return {"elapsed_seconds": elapsed, "gpu_peak_allocated_bytes": gpu, "rss_end_bytes": rss}


def document_scores(model, wrappers, items, device, start):
    rows = []
    for item in items:
        nats = M72.score_doc(model, item["document_ids"], wrappers, True, device, start)
        assert math.isfinite(nats)
        rows.append({"source_id": item["source_id"], "category": item["category"],
                     "bytes": item["bytes"], "nats": nats})
        budget(start, device)
    summary = {}
    for category in CATEGORIES:
        selected = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        summary[category] = sum(r["nats"] for r in selected) / (
            math.log(2) * sum(r["bytes"] for r in selected))
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    partial = args.out.with_name(args.out.stem + ".partial.json")
    assert M17.sha(TRAINING.read_bytes()) == M72.TRAINING_SHA
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    assert training["checkpoint_sha256"] == M72.CHECKPOINT_SHA
    assert M15.M13.sha256(training["checkpoint_path"]) == M72.CHECKPOINT_SHA
    assert M17.sha(M72.EXTERNAL.read_bytes()) == M72.EXTERNAL_SHA
    items = json.loads(M72.EXTERNAL.read_text(encoding="utf-8"))["items"]
    assert len(items) == 24
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors",
                             revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M72.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = torch.load(training["checkpoint_path"], map_location="cpu", weights_only=False)
    assert (state["axis_a"], state["axis_b"], state["updates"]) == (32, 40, 64)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M71.BalancedProductKeyExperts(layer.mlp, li, 32, 40).to(device)
        layer.mlp = wrapper
        for name in ("a", "b", "router"):
            getattr(wrapper, name).copy_(state["expert_state"][li][name].to(device))
        wrappers.append(wrapper)
    del state
    gc.collect()
    model.config.use_cache = False
    raw_ids, raw_bytes, meta = M15.M13.C.get_slice(tokenizer, "heldout", 4, 256, 271828)
    assert meta["ids_sha256"] == M15.EVAL_IDS_SHA and int(raw_bytes.sum()) == 4588
    raw_ids = torch.as_tensor(raw_ids, dtype=torch.long, device=device)
    trained_rows, trained_docs = document_scores(model, wrappers, items, device, start)
    trained_raw = M15.bpb(model, raw_ids, raw_bytes, wrappers, True)
    original = [(w.a.detach().clone(), w.b.detach().clone()) for w in wrappers]
    trials = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for wrapper, (a, b) in zip(wrappers, original):
                order = torch.as_tensor(rng.permutation(1280), dtype=torch.long, device=device)
                wrapper.a.copy_(a[order])
                wrapper.b.copy_(b[order])
        rows, docs = document_scores(model, wrappers, items, device, start)
        raw = M15.bpb(model, raw_ids, raw_bytes, wrappers, True)
        with torch.no_grad():
            for wrapper, (a, b) in zip(wrappers, original):
                wrapper.a.copy_(a)
                wrapper.b.copy_(b)
                assert torch.equal(wrapper.a, a) and torch.equal(wrapper.b, b)
        trial = {"seed": seed, "document_rows": rows, "document_bpb": docs,
                 "document_delta_bpb": {c: docs[c] - trained_docs[c] for c in CATEGORIES},
                 "raw_bpb": raw, "raw_delta_bpb": raw - trained_raw}
        assert all(math.isfinite(v) for v in trial["document_delta_bpb"].values())
        assert math.isfinite(trial["raw_delta_bpb"])
        trials.append(trial)
        partial.write_text(json.dumps({"completed_seeds": [t["seed"] for t in trials],
                                       "trained_document_bpb": trained_docs,
                                       "trained_raw_bpb": trained_raw,
                                       "trials": trials}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"seed": seed, "document_delta": trial["document_delta_bpb"]["pooled"],
                          "raw_delta": trial["raw_delta_bpb"], **budget(start, device)}), flush=True)
    summary = {}
    for name, values in (("pooled_document", [t["document_delta_bpb"]["pooled"] for t in trials]),
                         ("raw_four_window", [t["raw_delta_bpb"] for t in trials])):
        summary[name] = {"median_delta_bpb": float(np.median(values)),
                         "min_delta_bpb": min(values), "max_delta_bpb": max(values),
                         "positive_count": sum(v > 0 for v in values),
                         "permutations": len(values)}
    result = {"experiment": "METH-73-E1280-route-utility-diagnostic",
              "training_result_sha256": M72.TRAINING_SHA,
              "checkpoint_sha256": M72.CHECKPOINT_SHA,
              "external_manifest_sha256": M72.EXTERNAL_SHA,
              "raw_eval_ids_sha256": M15.EVAL_IDS_SHA,
              "seeds": SEEDS, "trained_document_rows": trained_rows,
              "trained_document_bpb": trained_docs, "trained_raw_bpb": trained_raw,
              "trials": trials, "summary": summary,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__,
                          "numpy": np.__version__},
              "decision": "diagnostic_only_METH72_gate_remains_failed"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps({"summary": summary, "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
