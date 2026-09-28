#!/usr/bin/env python3
"""Generate one frozen BF16 donor continuation for each new METH-158 chat."""

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
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
PROMPTS = DOC / "meth158_independent_training_manifest.json"
PROMPTS_SHA = "09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69"
MODEL_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS
            or result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-158 teacher shard resource stop: {result}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--stop", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert 0 <= args.start < args.stop <= 2560
    assert args.start % 64 == 0 and args.stop == args.start + 64
    assert M17.sha(PROMPTS.read_bytes()) == PROMPTS_SHA
    manifest = json.loads(PROMPTS.read_text(encoding="utf-8"))
    assert manifest["count_per_part"] == len(manifest["chat_rows"]) == 2560
    assert manifest["model"] == M42.MODEL and manifest["revision"] == M42.REV
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    devices = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(devices) == 1
    device = torch.device(f"cuda:{devices[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    eos = model.config.eos_token_id
    rows = []
    for index in range(args.start, args.stop):
        prompt = manifest["chat_rows"][index]
        ids = prompt["prompt_ids"]
        assert M17.sha(np.asarray(ids, dtype=np.int32).tobytes()) == prompt["prompt_ids_sha256"]
        inp = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
        with torch.inference_mode():
            out = model.generate(input_ids=inp, max_new_tokens=128,
                                 do_sample=False, pad_token_id=eos,
                                 use_cache=True)
        response = out[0, len(ids):].cpu().tolist()
        assert 1 <= len(response) <= 128
        rows.append({"index": index, "train_row": prompt["train_row"],
                     "prompt_ids_sha256": prompt["prompt_ids_sha256"],
                     "continuation_ids": response,
                     "continuation_ids_sha256": M17.sha(np.asarray(
                         response, dtype=np.int32).tobytes()),
                     "eos_terminated": bool(response[-1] == eos),
                     "repeated_8gram_3x": M20.repeated_8gram(response)})
        budget(start, device)
        if (index - args.start + 1) % 16 == 0:
            print(json.dumps({"completed": index + 1, "stop": args.stop,
                              "budget": budget(start, device)}), flush=True)
    result = {"experiment": "METH-158-independent-teacher-shard",
              "prompt_manifest_sha256": PROMPTS_SHA,
              "model": M42.MODEL, "revision": M42.REV,
              "model_sha256": MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "max_new_tokens": 128,
              "shard_start": args.start, "shard_stop": args.stop,
              "summary": {"rows": len(rows),
                          "min_response_tokens": min(len(r["continuation_ids"]) for r in rows),
                          "max_response_tokens": max(len(r["continuation_ids"]) for r in rows),
                          "mean_response_tokens": sum(len(r["continuation_ids"]) for r in rows) / len(rows),
                          "eos_terminated": sum(r["eos_terminated"] for r in rows),
                          "repeated_8gram_3x": sum(r["repeated_8gram_3x"] for r in rows)},
              "rows": rows,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()),
                      "summary": result["summary"], "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
