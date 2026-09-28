#!/usr/bin/env python3
"""Generate and merge resumable BF16 teacher shards for METH-164."""

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
MANIFEST = DOC / "meth164_cross_source_route_manifest.json"
MANIFEST_SHA = "d8420488104990ab63338721837c4f100135c45a07e787f2708eb741dfb63c9b"
MODEL_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
MAX_SECONDS = 20 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
SHARD = 32
COUNT = 128


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS or
            result["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-164 teacher resource stop: {result}")
    return result


def check_shard(path, start, stop, prompts):
    shard = json.loads(path.read_text(encoding="utf-8"))
    assert shard["experiment"] == "METH-164-cross-source-teacher-shard"
    assert shard["manifest_sha256"] == MANIFEST_SHA
    assert shard["model_sha256"] == MODEL_SHA
    assert shard["start"] == start and shard["stop"] == stop
    assert len(shard["rows"]) == stop - start
    for expected, row in zip(range(start, stop), shard["rows"]):
        prompt = prompts[expected]
        assert row["index"] == expected
        assert row["source_row"] == prompt["source_row"]
        assert row["prompt_ids_sha256"] == prompt["prompt_ids_sha256"]
        assert 1 <= len(row["continuation_ids"]) <= 128
        assert M17.sha(np.asarray(row["continuation_ids"], dtype=np.int32).tobytes()) == (
            row["continuation_ids_sha256"])
    return shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    assert M17.sha(MANIFEST.read_bytes()) == MANIFEST_SHA
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["model"] == M42.MODEL and manifest["revision"] == M42.REV
    prompts = manifest["chat_rows"]
    assert len(prompts) == COUNT
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
    started = time.monotonic()
    args.shards.mkdir(parents=True, exist_ok=True)
    missing = [start for start in range(0, COUNT, SHARD)
               if not (args.shards / f"shard_{start:03d}_{start + SHARD:03d}.json").exists()]
    model = None
    if missing:
        model = AutoModelForCausalLM.from_pretrained(
            M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to(device).eval()
        eos = model.config.eos_token_id
    shards = []
    rows = []
    for first in range(0, COUNT, SHARD):
        stop = first + SHARD
        path = args.shards / f"shard_{first:03d}_{stop:03d}.json"
        if not path.exists():
            response_rows = []
            for index in range(first, stop):
                prompt = prompts[index]
                ids = prompt["prompt_ids"]
                assert M17.sha(np.asarray(ids, dtype=np.int32).tobytes()) == (
                    prompt["prompt_ids_sha256"])
                inp = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
                with torch.inference_mode():
                    out = model.generate(input_ids=inp, max_new_tokens=128,
                                         do_sample=False, pad_token_id=eos,
                                         use_cache=True)
                continuation = out[0, len(ids):].cpu().tolist()
                assert 1 <= len(continuation) <= 128
                response_rows.append({
                    "index": index, "source_row": prompt["source_row"],
                    "prompt_ids_sha256": prompt["prompt_ids_sha256"],
                    "continuation_ids": continuation,
                    "continuation_ids_sha256": M17.sha(np.asarray(
                        continuation, dtype=np.int32).tobytes()),
                    "eos_terminated": bool(continuation[-1] == eos),
                    "repeated_8gram_3x": M20.repeated_8gram(continuation)})
                budget(started, device)
                if (index + 1) % 16 == 0:
                    print(json.dumps({"completed": index + 1,
                                      "budget": budget(started, device)}), flush=True)
            shard = {"experiment": "METH-164-cross-source-teacher-shard",
                     "manifest_sha256": MANIFEST_SHA, "model": M42.MODEL,
                     "revision": M42.REV, "model_sha256": MODEL_SHA,
                     "start": first, "stop": stop, "rows": response_rows,
                     "runtime_at_stop": budget(started, device)}
            path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8")
        shard = check_shard(path, first, stop, prompts)
        shards.append({"start": first, "stop": stop, "path": str(path.resolve()),
                       "sha256": M17.sha(path.read_bytes()), "bytes": path.stat().st_size})
        rows.extend(shard["rows"])
    assert len(rows) == COUNT
    result = {"experiment": "METH-164-cross-source-teacher-merged",
              "manifest_sha256": MANIFEST_SHA, "model": M42.MODEL,
              "revision": M42.REV, "model_sha256": MODEL_SHA,
              "max_new_tokens": 128, "shards": shards,
              "summary": {"rows": COUNT,
                          "continuation_tokens": sum(len(r["continuation_ids"]) for r in rows),
                          "eos_terminated": sum(r["eos_terminated"] for r in rows),
                          "repeated_8gram_3x": sum(r["repeated_8gram_3x"] for r in rows)},
              "rows": rows, "runtime": {**budget(started, device),
                                          "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()),
                      "summary": result["summary"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
