#!/usr/bin/env python3
"""METH-18: diagnostic residual-scale and router-null scoring, no training."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth16_residual_expert_continuation as M16
import meth17_fresh_transfer_audit as M17


ARMS = ("donor", "scale_025", "scale_050", "trained", "permuted")
SCALES = {"scale_025": 0.25, "scale_050": 0.50,
          "trained": 1.0, "permuted": 1.0}
PERM_SEED = 1818
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)
ROOT = Path(__file__).resolve().parents[3]
M17_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth17_fresh_document_result.json"
M17_RESULT_SHA = "c9e51c9736904c1674c755ef528b26ea656d1d33dfaf35af10d51c5c5e2e52a2"


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-18 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-18 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-18 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def aggregate(rows, arm):
    groups = {"pooled": rows}
    groups.update({category: [r for r in rows if r["category"] == category]
                   for category in M17.N_SELECT})
    result = {}
    for name, group in groups.items():
        byts = sum(r["bytes"] for r in group)
        denom = math.log(2) * byts
        donor = sum(r["nats"]["donor"] for r in group) / denom
        measured = sum(r["nats"][arm] for r in group) / denom
        result[name] = {"docs": len(group), "bytes": byts,
                        "donor_bpb": donor, "arm_bpb": measured,
                        "delta_bpb": measured-donor}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    manifest, items = M17.build_selection()
    manifest_path = Path(args.manifest)
    assert M17.sha(manifest_path.read_bytes()) == (
        "7e0593d6c56c28398e3440f9b80db15d506a31131a21480a87578c23b0a043a8")
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert M15.M13.sha256(M17_RESULT) == M17_RESULT_SHA
    prior = json.loads(M17_RESULT.read_text(encoding="utf-8"))
    assert prior["manifest_sha256"] == M17.sha(manifest_path.read_bytes())
    checkpoint_path = ROOT / "results/native_expert_scaling/meth16_checkpoints/meth16_update1024.pt"
    assert M15.M13.sha256(checkpoint_path) == M17.CHECKPOINT_SHA

    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tok = AutoTokenizer.from_pretrained(M15.M13.MODEL, revision=M15.M13.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["updates"] == 1024
    assert checkpoint["source_sha256"] == M15.M13.MODEL_SHA
    assert checkpoint["train_ids_sha256"] == M15.TRAIN_IDS_SHA
    for w, values in zip(wrappers, checkpoint["expert_state"]):
        w.a.copy_(values["a"].to(device))
        w.b.copy_(values["b"].to(device))
        w.router.copy_(values["router"].to(device))
    del checkpoint
    original_b = [w.b.detach().clone() for w in wrappers]
    original_router = [w.router.detach().clone() for w in wrappers]
    perm_rng = np.random.default_rng(PERM_SEED)
    permuted_router = [r[torch.as_tensor(perm_rng.permutation(M15.E),
                                         dtype=torch.long, device=device)]
                       for r in original_router]

    def set_arm(arm):
        for w, b, router, permuted in zip(wrappers, original_b,
                                          original_router, permuted_router):
            if arm != "donor":
                w.b.copy_(b * SCALES[arm])
            w.router.copy_(permuted if arm == "permuted" else router)

    rows = []
    for i, item in enumerate(items):
        ids = tok.encode(item["text"], add_special_tokens=False)
        row = {"index": i, "category": item["category"],
               "source_id": item["source_id"],
               "text_sha256": M17.sha(item["text"].encode("utf-8")),
               "bytes": len(item["text"].encode("utf-8")), "tokens": len(ids),
               "nats": {}}
        for arm in ARMS:
            set_arm(arm)
            row["nats"][arm] = M17.score_doc(model, ids, wrappers,
                                               arm != "donor", device, start)
        rows.append(row)
        check_budget(start, device)
        if (i + 1) % 6 == 0:
            print(f"scored {i+1}/{len(items)} documents", flush=True)

    summary = {arm: aggregate(rows, arm) for arm in ARMS if arm != "donor"}
    original = prior["summary"]
    assert abs(summary["trained"]["pooled"]["donor_bpb"] -
               original["pooled"]["donor_bpb"]) <= 1e-5
    assert abs(summary["trained"]["pooled"]["arm_bpb"] -
               original["pooled"]["student_bpb"]) <= 1e-5
    assert sum(r["nats"]["trained"] > r["nats"]["donor"]
               for r in rows if r["category"] == "code") == 24
    route_effect = {name: summary["permuted"][name]["arm_bpb"] -
                    summary["trained"][name]["arm_bpb"]
                    for name in ("pooled", *M17.N_SELECT)}

    eval_ids, eval_bytes, eval_meta = M15.M13.C.get_slice(tok, "heldout", 16, 512, 314159)
    assert eval_meta["ids_sha256"] == M16.EVAL_IDS_SHA
    assert int(eval_bytes.sum()) == 33374
    eval_ids = torch.as_tensor(eval_ids, dtype=torch.long, device=device)
    internal = {}
    for arm in ("donor", "scale_025", "scale_050", "trained"):
        set_arm(arm)
        internal[arm] = M15.bpb(model, eval_ids, eval_bytes, wrappers, arm != "donor")
    assert abs(internal["donor"] - 0.840579) <= 1e-4
    assert abs(internal["trained"] - 0.824196) <= 1e-4
    runtime = check_budget(start, device)
    result = {"experiment": "METH-18", "decision": "diagnostic_only_no_promotion",
              "manifest_sha256": M17.sha(manifest_path.read_bytes()),
              "checkpoint_sha256": M17.CHECKPOINT_SHA,
              "m17_result_sha256": M15.M13.sha256(M17_RESULT),
              "model_sha256": M15.M13.MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "arms": ARMS, "scales": SCALES,
              "permutation_seed": PERM_SEED,
              "rows": rows, "summary": summary,
              "permuted_minus_trained_bpb": route_effect,
              "internal_development_bpb": internal,
              "runtime": {**runtime, "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0]}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "permuted_minus_trained_bpb": route_effect,
                      "internal_development_bpb": internal,
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
