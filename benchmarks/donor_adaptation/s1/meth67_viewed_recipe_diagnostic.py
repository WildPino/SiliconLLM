#!/usr/bin/env python3
"""METH-67: score frozen E128 recipes on the viewed METH-66 set."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_e128_smoke as M55
import meth55_product_key_experts as PK
import meth66_e128_e1280_smoke as M66


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = (
    ("meth55_update16", ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth55_product_key_e128_smoke.pt",
     "2d652709c6730e2b4f3a1a543834d7eec7409ef0f374738ae24d57ae2a5b5bd3"),
    ("meth56_update512", ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth56_checkpoints/meth56_update512.pt",
     "8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(start, device):
    state = {"elapsed_seconds": time.monotonic() - start,
             "rss_bytes": psutil.Process().memory_info().rss,
             "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if state["elapsed_seconds"] > 480 or state["rss_bytes"] > 8 * (1 << 30) or state["gpu_peak_allocated_bytes"] > 5 * (1 << 30):
        raise RuntimeError(f"METH-67 budget: {state}")
    return state


@torch.inference_mode()
def score(model, wrappers, eval_ids, eval_bytes, prompts, device, start):
    losses = []
    for enabled in (False, True):
        for wrapper in wrappers:
            wrapper.enabled = enabled
        nats = 0.0
        for i in range(eval_ids.shape[0]):
            ids = eval_ids[i:i + 1]
            logits = model(ids, use_cache=False).logits.float()
            lp = F.log_softmax(logits[:, :-1], dim=-1)
            nats += float(-lp.gather(-1, ids[:, 1:].unsqueeze(-1)).sum())
            budget(start, device)
        losses.append(nats / (math.log(2) * int(eval_bytes.sum())))
    for wrapper in wrappers:
        wrapper.route_counts.zero_()
        wrapper.oracle_checks = True
        wrapper.collect_load = True
    matching = positions = 0
    for row in prompts:
        ids = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device).unsqueeze(0)
        for wrapper in wrappers:
            wrapper.enabled = False
        donor = model(ids, use_cache=False).logits.argmax(dim=-1)
        for wrapper in wrappers:
            wrapper.enabled = True
        student = model(ids, use_cache=False).logits.argmax(dim=-1)
        matching += int((donor == student).sum())
        positions += ids.numel()
        budget(start, device)
    load = PK.load_summary(wrappers)
    return {"donor_bpb": losses[0], "student_bpb": losses[1],
            "delta_bpb": losses[1] - losses[0],
            "matching": matching, "positions": positions,
            "agreement": matching / positions,
            "min_selected_slots": min(x["selected_slots"] for x in load),
            "worst_max_to_mean_load": max(x["max_to_mean"] for x in load),
            "route_oracle": "exact_on_all_dev_hidden_states"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert sha(M66.DEV) == M66.DEV_SHA
    for _, path, expected in CHECKPOINTS:
        assert sha(path) == expected
    dev = json.loads(M66.DEV.read_text(encoding="utf-8"))
    torch.set_num_threads(6)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert sha(source) == M55.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    eval_ids, eval_bytes, meta = M15.M13.C.get_slice(tokenizer, "heldout", 4, 256, 271828)
    assert meta["ids_sha256"] == M15.EVAL_IDS_SHA and int(eval_bytes.sum()) == 4588
    eval_ids = torch.as_tensor(eval_ids, dtype=torch.long, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = PK.ProductKeyExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    model.config.use_cache = False
    model.eval()
    outcomes = {}
    for name, path, expected in CHECKPOINTS:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert len(checkpoint["expert_state"]) == 24
        with torch.no_grad():
            for wrapper, saved in zip(wrappers, checkpoint["expert_state"]):
                for key in ("a", "b", "router"):
                    getattr(wrapper, key).copy_(saved[key].to(device))
        del checkpoint
        outcomes[name] = {"checkpoint_sha256": expected,
                          **score(model, wrappers, eval_ids, eval_bytes,
                                  dev["rows"], device, start)}
        print(json.dumps({"name": name, "agreement": outcomes[name]["agreement"],
                          "delta_bpb": outcomes[name]["delta_bpb"]}), flush=True)
    result = {"experiment": "METH-67-viewed-dev-recipe-diagnostic",
              "source_sha256": sha(__file__), "donor_sha256": M55.MODEL_SHA,
              "dev_sha256": M66.DEV_SHA, "outcomes": outcomes,
              "runtime": budget(start, device),
              "decision": "diagnostic_only_viewed_prompts"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
