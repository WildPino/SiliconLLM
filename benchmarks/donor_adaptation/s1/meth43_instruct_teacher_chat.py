#!/usr/bin/env python3
"""Generate frozen Instruct teacher responses and masked chat windows."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth42_instruct_prompt_manifest as M42


ROOT = Path(__file__).resolve().parents[3]
PROMPTS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth43_instruct_chat_train_manifest.json"
PROMPTS_SHA = "17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055"
MODEL_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
MAX_SECONDS = 10 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or peak > MAX_GPU_BYTES or rss > MAX_RSS_BYTES:
        raise RuntimeError(f"METH-43 teacher budget: {elapsed:.1f}s, GPU {peak}, RSS {rss}")
    return {"elapsed_seconds": elapsed, "gpu_peak_allocated_bytes": peak,
            "rss_end_bytes": rss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M17.sha(PROMPTS.read_bytes()) == PROMPTS_SHA
    manifest = json.loads(PROMPTS.read_text(encoding="utf-8"))
    assert manifest["count"] == len(manifest["rows"]) == 256
    assert manifest["revision"] == M42.REV
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors",
                             revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source) == MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M42.MODEL, revision=M42.REV, local_files_only=True)
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
    eos_id = model.config.eos_token_id
    rows = []
    for i, row in enumerate(manifest["rows"]):
        prompt = row["prompt_ids"]
        assert M17.sha(np.asarray(prompt, dtype=np.int32).tobytes()) == row["prompt_ids_sha256"]
        inp = torch.as_tensor(prompt, dtype=torch.long,
                              device=device).unsqueeze(0)
        with torch.inference_mode():
            generated = model.generate(input_ids=inp, max_new_tokens=64,
                                       do_sample=False, pad_token_id=eos_id,
                                       use_cache=True)
        response = generated[0, len(prompt):].cpu().tolist()
        assert 1 <= len(response) <= 64
        full = prompt + response
        assert len(full) >= 129
        window = full[-129:]
        source_mask = [0] * (len(full) - len(response)) + [1] * len(response)
        target_mask = source_mask[-129:][1:]
        assert len(window) == 129 and len(target_mask) == 128
        assert sum(target_mask) == len(response)
        rows.append({"train_row": row["train_row"],
                     "prompt_ids_sha256": row["prompt_ids_sha256"],
                     "continuation_ids": response,
                     "continuation_text": tokenizer.decode(
                         response, skip_special_tokens=False),
                     "window_ids": window,
                     "assistant_target_mask": target_mask,
                     "window_ids_sha256": M17.sha(np.asarray(
                         window, dtype=np.int32).tobytes()),
                     "repeated_8gram_3x": M20.repeated_8gram(response),
                     "eos_terminated": bool(response[-1] == eos_id)})
        budget(start, device)
        if (i + 1) % 32 == 0:
            print(f"teacher responses {i+1}/{len(manifest['rows'])}", flush=True)
    lengths = [len(r["continuation_ids"]) for r in rows]
    runtime = budget(start, device)
    result = {"experiment": "METH-43-teacher-chat",
              "prompt_manifest_sha256": PROMPTS_SHA,
              "model": M42.MODEL, "revision": M42.REV,
              "model_sha256": MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "max_new_tokens": 64, "window_tokens": 129,
              "assistant_masked_targets": True,
              "summary": {"rows": len(rows), "min_response_tokens": min(lengths),
                          "max_response_tokens": max(lengths),
                          "mean_response_tokens": sum(lengths) / len(lengths),
                          "eos_terminated": sum(r["eos_terminated"] for r in rows),
                          "repeated_8gram_3x": sum(r["repeated_8gram_3x"] for r in rows)},
              "rows": rows,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
