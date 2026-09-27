#!/usr/bin/env python3
"""Export and verify BF16-effective METH-47 factors without changing inference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
from safetensors import safe_open
from safetensors.torch import save_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth48_int8_factor_bank as M48


MAX_SECONDS = 10 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or peak > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-51 budget: {elapsed:.1f}s, RSS {rss}, GPU {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


@torch.inference_mode()
def logits_for_prompts(model, wrappers, items, device, start):
    M44.set_experts(wrappers, True)
    logits = []
    for item in items:
        ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None, :]
        logits.append(model(ids, use_cache=False).logits.squeeze(0).cpu())
        budget(start, device)
    return logits


@torch.inference_mode()
def generations(model, wrappers, items, device, start):
    M44.set_experts(wrappers, True)
    eos = model.config.eos_token_id
    model.config.use_cache = True
    rows = []
    for item in items:
        ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None, :]
        output = model.generate(input_ids=ids, max_new_tokens=128,
                                do_sample=False, pad_token_id=eos, use_cache=True)
        continuation = output[0, ids.shape[1]:].cpu().tolist()
        rows.append({"source_id": item["source_id"],
                     "category": item["category"],
                     "continuation_ids": continuation})
        budget(start, device)
        print(f"generation {len(rows)}/{len(items)}", flush=True)
    model.config.use_cache = False
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path, digest in ((M48.CHECKPOINT, M48.CHECKPOINT_SHA),
                         (M48.MANIFEST, M48.MANIFEST_SHA),
                         (M48.REFERENCE, M48.REFERENCE_SHA)):
        assert M15.M13.sha256(path) == digest, path
    manifest = json.loads(M48.MANIFEST.read_text(encoding="utf-8"))
    reference = json.loads(M48.REFERENCE.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == len(reference["generation_rows"]) == 12
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
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
    tensors = {}
    for li, values in enumerate(state["expert_state"]):
        for factor in ("a", "b"):
            original = values[factor]
            expected = ((M15.E, M15.R, M15.M13.D) if factor == "a"
                        else (M15.E, M15.M13.D, M15.R))
            assert original.dtype == torch.float32 and tuple(original.shape) == expected
            assert bool(torch.isfinite(original).all())
            tensors[f"layers.{li}.{factor}"] = original.to(torch.bfloat16).contiguous()
        assert tuple(values["router"].shape) == (M15.E, M15.M13.D)
    metadata = {"format": "METH51_E128_AB_BF16_EFFECTIVE_V1",
                "checkpoint_sha256": M48.CHECKPOINT_SHA,
                "model_sha256": M44.MODEL_SHA,
                "effective_dtype": "bfloat16"}
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.artifact), metadata=metadata)
    with safe_open(str(args.artifact), framework="pt", device="cpu") as archive:
        assert archive.metadata() == metadata and len(archive.keys()) == 2 * M15.M13.L
        for key, value in tensors.items():
            assert torch.equal(archive.get_tensor(key), value), key
    payload_bytes_per_expert = M15.M13.L * 2 * M15.R * M15.M13.D * 2
    assert sum(t.numel() * t.element_size() for t in tensors.values()) == M15.E * payload_bytes_per_expert

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
    model.config.use_cache = False
    original_logits = logits_for_prompts(model, wrappers, items, device, start)
    original_generations = generations(model, wrappers, items, device, start)
    for original, saved in zip(original_generations, reference["generation_rows"]):
        assert original["source_id"] == saved["source_id"]
        assert original["continuation_ids"] == saved["student"]["continuation_ids"]
    with safe_open(str(args.artifact), framework="pt", device="cpu") as archive:
        with torch.no_grad():
            for li, wrapper in enumerate(wrappers):
                for factor in ("a", "b"):
                    stored = archive.get_tensor(f"layers.{li}.{factor}")
                    getattr(wrapper, factor).copy_(stored.float().to(device))
                    actual = getattr(wrapper, factor).detach().cpu().to(torch.bfloat16)
                    assert torch.equal(actual, state["expert_state"][li][factor].to(torch.bfloat16))
    reloaded_logits = logits_for_prompts(model, wrappers, items, device, start)
    logit_rows = []
    for item, original, reloaded in zip(items, original_logits, reloaded_logits):
        assert original.shape == reloaded.shape
        equal = torch.equal(original, reloaded)
        logit_rows.append({"source_id": item["source_id"],
                           "positions": original.shape[0],
                           "logit_elements": original.numel(),
                           "bit_identical": equal})
        assert equal, item["source_id"]
    reloaded_generations = generations(model, wrappers, items, device, start)
    generation_rows = []
    for original, reloaded in zip(original_generations, reloaded_generations):
        equal = original["continuation_ids"] == reloaded["continuation_ids"]
        generation_rows.append({"source_id": original["source_id"],
                                "original_tokens": len(original["continuation_ids"]),
                                "reloaded_tokens": len(reloaded["continuation_ids"]),
                                "identical": equal})
        assert equal, original["source_id"]
    runtime = budget(start, device)
    result = {"experiment": "METH-51-bf16-effective-factor-bank",
              "checkpoint_sha256": M48.CHECKPOINT_SHA,
              "model_sha256": M44.MODEL_SHA,
              "manifest_sha256": M48.MANIFEST_SHA,
              "reference_sha256": M48.REFERENCE_SHA,
              "artifact_sha256": M15.M13.sha256(args.artifact),
              "artifact_bytes": args.artifact.stat().st_size,
              "factor_payload_bytes_per_expert": payload_bytes_per_expert,
              "projected_factor_payload_bytes": {str(e): e * payload_bytes_per_expert
                                                 for e in (128, 27355, 273547)},
              "selected_factor_bytes_per_token": M15.K * payload_bytes_per_expert,
              "prompt_logits": {"prompts": len(logit_rows),
                                "positions": sum(r["positions"] for r in logit_rows),
                                "elements": sum(r["logit_elements"] for r in logit_rows),
                                "all_bit_identical": all(r["bit_identical"] for r in logit_rows),
                                "rows": logit_rows},
              "generation": {"prompts": len(generation_rows),
                             "all_identical": all(r["identical"] for r in generation_rows),
                             "rows": generation_rows},
              "decision": "exact_effective_factor_anchor_pass",
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("artifact_sha256", "artifact_bytes",
                                            "factor_payload_bytes_per_expert",
                                            "projected_factor_payload_bytes",
                                            "selected_factor_bytes_per_token",
                                            "decision", "runtime")}
                     | {"logit_positions": result["prompt_logits"]["positions"],
                        "generation_matches": sum(r["identical"] for r in generation_rows)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
