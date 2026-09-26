#!/usr/bin/env python3
"""METH-20: paired donor/half-adapter generation from bound document prompts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import psutil
from safetensors import safe_open
from safetensors.torch import load_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth19_half_residual_independent_audit as M19


ROOT = Path(__file__).resolve().parents[3]
DOCUMENT_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth19_half_residual_independent_manifest.json"
DOCUMENT_MANIFEST_SHA = "1eb4193e521df4d25d6367a84d8478d4ea733841fe9a77d550de01796ddf3d70"
ADAPTER = ROOT / "results/native_expert_scaling/meth19_half_residual_adapter.safetensors"
ADAPTER_SHA = "3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca"
SEED = "meth20-20020"
PER_CATEGORY = 8
PROMPT_TOKENS = 256
NEW_TOKENS = 128
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-20 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-20 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-20 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def selected_prompts(tokenizer):
    document_manifest, items = M19.select()
    assert M17.sha(DOCUMENT_MANIFEST.read_bytes()) == DOCUMENT_MANIFEST_SHA
    assert json.loads(DOCUMENT_MANIFEST.read_text(encoding="utf-8")) == document_manifest
    selected = []
    for category in M19.COUNTS:
        pool = [item for item in items if item["category"] == category]
        pool.sort(key=lambda x: M17.sha((SEED + "|" + x["source_id"]).encode("utf-8")))
        assert len(pool) >= PER_CATEGORY
        for item in pool[:PER_CATEGORY]:
            ids = tokenizer.encode(item["text"], add_special_tokens=False)
            assert len(ids) >= PROMPT_TOKENS + NEW_TOKENS
            prompt = ids[:PROMPT_TOKENS]
            selected.append({"category": category,
                             "source_id": item["source_id"],
                             "text_sha256": M17.sha(item["text"].encode("utf-8")),
                             "prompt_ids_sha256": M17.sha(np.asarray(prompt, dtype=np.int32).tobytes()),
                             "prompt_ids": prompt})
    assert len(selected) == PER_CATEGORY * len(M19.COUNTS)
    manifest = {"experiment": "METH-20", "seed": SEED,
                "document_manifest_sha256": DOCUMENT_MANIFEST_SHA,
                "tokenizer_fingerprint": M15.M13.TOK_FP,
                "prompt_tokens": PROMPT_TOKENS, "max_new_tokens": NEW_TOKENS,
                "selected": [{k: v for k, v in row.items() if k != "prompt_ids"}
                             for row in selected]}
    return manifest, selected


def repeated_8gram(tokens):
    if len(tokens) < 8:
        return False
    counts = Counter(tuple(tokens[i:i+8]) for i in range(len(tokens)-7))
    return max(counts.values(), default=0) >= 3


def distinct2(tokens):
    if len(tokens) < 2:
        return 0.0
    return len(set(zip(tokens, tokens[1:]))) / (len(tokens)-1)


def summarize(rows, eos_id):
    categories = ("pooled", *M19.COUNTS)
    out = {}
    for category in categories:
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        arms = {}
        for arm in ("donor", "half"):
            continuations = [r[arm]["continuation_ids"] for r in group]
            arms[arm] = {"prompts": len(group),
                         "repeated_8gram_3x": sum(repeated_8gram(x) for x in continuations),
                         "early_non_eos_under16": sum(len(x) < 16 and (not x or x[-1] != eos_id)
                                                      for x in continuations),
                         "eos_terminated": sum(bool(x) and x[-1] == eos_id
                                               for x in continuations),
                         "generated_tokens": sum(len(x) for x in continuations),
                         "mean_distinct2": sum(distinct2(x) for x in continuations) / len(group)}
        out[category] = {"donor": arms["donor"], "half": arms["half"],
                         "first_token_agreement": sum(r["donor"]["continuation_ids"][:1]
                                                      == r["half"]["continuation_ids"][:1]
                                                      for r in group)}
    passed = all(out[c]["half"]["repeated_8gram_3x"]
                 <= out[c]["donor"]["repeated_8gram_3x"] + 1
                 and out[c]["half"]["early_non_eos_under16"]
                 <= out[c]["donor"]["early_non_eos_under16"] + 1
                 for c in categories)
    return out, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256")
    ap.add_argument("--out")
    args = ap.parse_args()
    torch.set_num_threads(6)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    manifest, prompts = selected_prompts(tokenizer)
    manifest_path = Path(args.manifest)
    if args.prepare:
        assert args.manifest_sha256 is None and args.out is None
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest_sha256": M17.sha(manifest_path.read_bytes()),
                          "prompt_count": len(prompts),
                          "categories": list(M19.COUNTS)}, indent=2), flush=True)
        return
    assert args.manifest_sha256 and args.out
    assert M17.sha(manifest_path.read_bytes()) == args.manifest_sha256
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert M15.M13.sha256(ADAPTER) == ADAPTER_SHA
    with safe_open(str(ADAPTER), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
    assert meta["source_checkpoint_sha256"] == M17.CHECKPOINT_SHA
    assert meta["model_sha256"] == M15.M13.MODEL_SHA
    assert meta["tokenizer_fingerprint"] == M15.M13.TOK_FP
    assert meta["output_factor"] == "0.5"
    tensors = load_file(str(ADAPTER), device="cpu")
    start = time.monotonic()
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            value = tensors[f"layers.{li}.{key}"]
            getattr(wrapper, key).copy_(value.to(device))
        wrappers.append(wrapper)
    assert len(tensors) == 3 * M15.M13.L
    del tensors
    rows = []
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            input_ids = torch.as_tensor(prompt["prompt_ids"], dtype=torch.long,
                                        device=device).unsqueeze(0)
            row = {k: v for k, v in prompt.items() if k != "prompt_ids"}
            for enabled, arm in ((False, "donor"), (True, "half")):
                for w in wrappers:
                    w.enabled = enabled
                output = model.generate(
                    input_ids=input_ids, max_new_tokens=NEW_TOKENS,
                    do_sample=False, pad_token_id=model.config.eos_token_id,
                    use_cache=True)
                ids = output[0, PROMPT_TOKENS:].tolist()
                row[arm] = {"continuation_ids": ids,
                            "continuation_text": tokenizer.decode(ids, skip_special_tokens=False),
                            "generated_tokens": len(ids),
                            "repeated_8gram_3x": repeated_8gram(ids),
                            "distinct2": distinct2(ids)}
                check_budget(start, device)
            rows.append(row)
            if (i+1) % 4 == 0:
                print(f"generated {i+1}/{len(prompts)} prompt pairs", flush=True)
    summary, passed = summarize(rows, model.config.eos_token_id)
    runtime = check_budget(start, device)
    result = {"experiment": "METH-20",
              "decision": "relative_generation_screen_pass" if passed
                          else "relative_generation_screen_fail",
              "manifest_sha256": args.manifest_sha256,
              "adapter_sha256": ADAPTER_SHA,
              "model_sha256": M15.M13.MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "rows": rows, "summary": summary,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "decision": result["decision"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
