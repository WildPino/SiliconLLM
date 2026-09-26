#!/usr/bin/env python3
"""METH-27: greedy behavior of the stored R8 core plus E128 adapter."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth24_export_r8_core as M24
import meth25_fresh_r8_artifact_audit as M25A
import meth25_fresh_r8_manifest as M25


ROOT = M25.ROOT
ARMS = ("original_donor", "r8_donor", "r8_adapter")
SEED = "252527"
PROMPT_TOKENS = 256
NEW_TOKENS = 128


def select_prompts(tokenizer):
    manifest, items = M25.select()
    assert M17.sha(M25A.MANIFEST.read_bytes()) == M25A.MANIFEST_SHA
    assert json.loads(M25A.MANIFEST.read_text(encoding="utf-8")) == manifest
    prompts = []
    for category in M25.COUNTS:
        pool = [item for item in items if item["category"] == category]
        pool.sort(key=lambda x: M17.sha((SEED + "|" + x["source_id"]).encode()))
        assert len(pool) >= 8
        for item in pool[:8]:
            ids = tokenizer.encode(item["text"], add_special_tokens=False)
            assert len(ids) >= PROMPT_TOKENS + NEW_TOKENS
            prefix = ids[:PROMPT_TOKENS]
            prompts.append({"category": category, "source_id": item["source_id"],
                            "text_sha256": M17.sha(item["text"].encode("utf-8")),
                            "prompt_ids_sha256": M17.sha(
                                np.asarray(prefix, dtype=np.int32).tobytes()),
                            "prompt_ids": prefix})
    assert len(prompts) == 24
    prompt_manifest = {
        "experiment": "METH-27", "seed": SEED,
        "document_manifest_sha256": M25A.MANIFEST_SHA,
        "core_sha256": M25A.CORE_SHA,
        "adapter_sha256": M20.ADAPTER_SHA,
        "model_sha256": M15.M13.MODEL_SHA,
        "tokenizer_fingerprint": M15.M13.TOK_FP,
        "prompt_tokens": PROMPT_TOKENS, "max_new_tokens": NEW_TOKENS,
        "selected": [{k: v for k, v in p.items() if k != "prompt_ids"}
                     for p in prompts]}
    return prompt_manifest, prompts


def summarize(rows, eos_id):
    out = {}
    for category in ("pooled", *M25.COUNTS):
        group = rows if category == "pooled" else [
            row for row in rows if row["category"] == category]
        arms = {}
        for arm in ARMS:
            continuations = [r[arm]["continuation_ids"] for r in group]
            arms[arm] = {
                "prompts": len(group),
                "repeated_8gram_3x": sum(M20.repeated_8gram(x) for x in continuations),
                "early_non_eos_under16": sum(len(x) < 16 and (not x or x[-1] != eos_id)
                                             for x in continuations),
                "eos_terminated": sum(bool(x) and x[-1] == eos_id
                                      for x in continuations),
                "generated_tokens": sum(len(x) for x in continuations),
                "mean_distinct2": sum(M20.distinct2(x) for x in continuations) / len(group)}
        out[category] = {"arms": arms,
                         "first_token_agreement_r8_donor_adapter": sum(
                             r["r8_donor"]["continuation_ids"][:1] ==
                             r["r8_adapter"]["continuation_ids"][:1]
                             for r in group),
                         "first_token_agreement_bf16_r8_donor": sum(
                             r["original_donor"]["continuation_ids"][:1] ==
                             r["r8_donor"]["continuation_ids"][:1]
                             for r in group)}
    relative_pass = all(
        out[c]["arms"]["r8_adapter"]["repeated_8gram_3x"] <=
        out[c]["arms"]["r8_donor"]["repeated_8gram_3x"] + 1
        and out[c]["arms"]["r8_adapter"]["early_non_eos_under16"] <=
        out[c]["arms"]["r8_donor"]["early_non_eos_under16"] + 1
        for c in out)
    absolute_pass = (
        out["pooled"]["arms"]["r8_adapter"]["repeated_8gram_3x"] <= 4
        and out["code"]["arms"]["r8_adapter"]["repeated_8gram_3x"] <= 2
        and out["pooled"]["arms"]["r8_adapter"]["early_non_eos_under16"] == 0)
    return out, relative_pass, absolute_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256")
    ap.add_argument("--out")
    args = ap.parse_args()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    prompt_manifest, prompts = select_prompts(tokenizer)
    path = Path(args.manifest)
    if args.prepare:
        assert args.manifest_sha256 is None and args.out is None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prompt_manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest_sha256": M17.sha(path.read_bytes()),
                          "prompt_count": len(prompts)}, indent=2), flush=True)
        return
    assert args.manifest_sha256 and args.out
    assert M17.sha(path.read_bytes()) == args.manifest_sha256
    assert json.loads(path.read_text(encoding="utf-8")) == prompt_manifest
    assert M15.M13.sha256(M25A.CORE) == M25A.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        assert archive.metadata()["model_sha256"] == M15.M13.MODEL_SHA
        assert archive.metadata()["output_factor"] == "0.5"
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    start = time.monotonic()
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.config.eos_token_id == tokenizer.eos_token_id
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrappers.append(wrapper)
    assert len(adapter) == 3 * M15.M13.L
    del adapter

    rows = [{k: v for k, v in prompt.items() if k != "prompt_ids"}
            for prompt in prompts]

    def run_arm(arm):
        enabled = arm == "r8_adapter"
        for w in wrappers:
            w.enabled = enabled
        with torch.inference_mode():
            for i, (prompt, row) in enumerate(zip(prompts, rows)):
                inputs = torch.as_tensor(prompt["prompt_ids"], dtype=torch.long,
                                         device=device).unsqueeze(0)
                output = model.generate(
                    input_ids=inputs, max_new_tokens=NEW_TOKENS,
                    do_sample=False, pad_token_id=model.config.eos_token_id,
                    use_cache=True)
                ids = output[0, PROMPT_TOKENS:].tolist()
                row[arm] = {"continuation_ids": ids,
                            "continuation_text": tokenizer.decode(ids,
                                                                   skip_special_tokens=False),
                            "generated_tokens": len(ids),
                            "repeated_8gram_3x": M20.repeated_8gram(ids),
                            "distinct2": M20.distinct2(ids)}
                M20.check_budget(start, device)
                if (i+1) % 8 == 0:
                    print(f"{arm} generated {i+1}/{len(prompts)} prompts", flush=True)

    run_arm("original_donor")
    with safe_open(str(M25A.CORE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_CORE_V1"
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert meta["tied_head"] == "true"
        assert len(archive.keys()) == 459
        matrix_count = control_count = 0
        for name, param in original_params.items():
            organ = M24.classify(name, tuple(param.shape))
            if organ == "control":
                stored = archive.get_tensor(name)
                assert stored.dtype == torch.float32
                assert torch.equal(stored.to(torch.bfloat16), param.detach().cpu())
                control_count += 1
            else:
                codes = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (param.shape[0],)
                assert bool((codes >= -127).all()) and bool((codes <= 127).all())
                param.copy_((codes.to(device).float() *
                             scale.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    M20.check_budget(start, device)
    print("loaded stored R8 core", flush=True)
    run_arm("r8_donor")
    run_arm("r8_adapter")
    summary, relative_pass, absolute_pass = summarize(rows, model.config.eos_token_id)
    runtime = M20.check_budget(start, device)
    result = {"experiment": "METH-27", "manifest_sha256": args.manifest_sha256,
              "core_sha256": M25A.CORE_SHA, "adapter_sha256": M20.ADAPTER_SHA,
              "model_sha256": M15.M13.MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "rows": rows, "summary": summary,
              "relative_behavior_gate_pass": relative_pass,
              "absolute_usefulness_screen_pass": absolute_pass,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "relative_behavior_gate_pass": relative_pass,
                      "absolute_usefulness_screen_pass": absolute_pass,
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
