#!/usr/bin/env python3
"""Capture actual E128/E1280 pre-FFN vectors for varied CPU factor access."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "donor_adaptation/s1"))
import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth71_balanced_product_key as M71
import meth72_e1280_external_audit as M72
import meth76_export_learned_product_key as M76


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MAGIC = b"M77HX001"
HEADER = "<8s4I"
TOKENS = 256
LAYERS = 24
WIDTH = 896
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    gpu = torch.cuda.max_memory_allocated(device)
    rss = psutil.Process().memory_info().rss
    if elapsed > MAX_SECONDS or gpu > MAX_GPU or rss > MAX_RSS:
        raise RuntimeError(f"METH-77 capture budget: {elapsed:.1f}s GPU={gpu} RSS={rss}")
    return {"elapsed_seconds": elapsed, "gpu_peak_allocated_bytes": gpu,
            "rss_end_bytes": rss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, choices=(128, 1280), required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    name, result_sha, checkpoint_sha, axis_a, axis_b = M76.TRAININGS[args.experts]
    training_path = DOCS / name
    assert M76.sha(training_path) == result_sha
    training = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoint = Path(training["checkpoint_path"])
    assert training["checkpoint_sha256"] == checkpoint_sha
    assert M76.sha(checkpoint) == checkpoint_sha
    ledger_path = DOCS / f"meth76_e{args.experts}_export_result.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["checkpoint_sha256"] == checkpoint_sha
    assert M76.sha(args.bank) == ledger["bank"]["sha256"]
    assert M17.sha(M72.EXTERNAL.read_bytes()) == M72.EXTERNAL_SHA
    manifest = json.loads(M72.EXTERNAL.read_text(encoding="utf-8"))
    items = manifest["items"]
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
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert (state["axis_a"], state["axis_b"], state["updates"]) == (axis_a, axis_b, 64)
    wrappers = []
    captured = {}
    positions_gpu = None
    hooks = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M71.BalancedProductKeyExperts(layer.mlp, li, axis_a, axis_b).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(state["expert_state"][li][key].to(device))
        wrappers.append(wrapper)
        def capture(module, inputs, layer_id=li):
            x = inputs[0].detach()
            captured[layer_id] = x[0].index_select(0, positions_gpu).to(
                device="cpu", dtype=torch.bfloat16).contiguous()
        hooks.append(wrapper.register_forward_pre_hook(capture))
    del state
    gc.collect()
    vectors = np.empty((TOKENS, LAYERS, WIDTH), dtype="<u2")
    token_rows = []
    cursor = 0
    with torch.inference_mode():
        for item_index, item in enumerate(items):
            assert M17.sha(np.asarray(item["prompt_ids"], dtype=np.int32).tobytes()) == item["prompt_ids_sha256"]
            prompt = item["prompt_ids"]
            count = 11 if item_index < 16 else 10
            positions = np.linspace(0, len(prompt) - 1, count, dtype=np.int32)
            assert len(set(positions.tolist())) == count
            positions_gpu = torch.as_tensor(positions, dtype=torch.long, device=device)
            ids = torch.as_tensor(prompt, dtype=torch.long, device=device)[None]
            captured.clear()
            model(ids, use_cache=False)
            assert len(captured) == LAYERS
            for local, position in enumerate(positions.tolist()):
                for li in range(LAYERS):
                    vectors[cursor, li] = captured[li][local].view(torch.uint16).numpy()
                token_rows.append({"source_id": item["source_id"],
                                   "prompt_ids_sha256": item["prompt_ids_sha256"],
                                   "position": position})
                cursor += 1
            print(json.dumps({"prompt": item_index + 1, "captured_tokens": cursor}),
                  flush=True)
            budget(start, device)
    for hook in hooks:
        hook.remove()
    assert cursor == TOKENS
    args.vectors.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(HEADER, MAGIC, LAYERS, TOKENS, WIDTH, args.experts)
    with args.vectors.open("wb") as file:
        file.write(header)
        file.write(vectors.tobytes())
    assert args.vectors.stat().st_size == len(header) + vectors.nbytes
    readback = np.memmap(args.vectors, dtype="<u2", mode="r", offset=len(header),
                         shape=vectors.shape)
    assert np.array_equal(readback, vectors)
    del readback
    result = {"experiment": "METH-77-varied-hidden-capture",
              "experts": args.experts, "source_sha256": M72.MODEL_SHA,
              "training_result_sha256": result_sha,
              "checkpoint_sha256": checkpoint_sha,
              "external_manifest_sha256": M72.EXTERNAL_SHA,
              "bank_sha256": ledger["bank"]["sha256"],
              "vectors": {"path": str(args.vectors), "bytes": args.vectors.stat().st_size,
                          "sha256": M76.sha(args.vectors)},
              "shape": list(vectors.shape), "selection": token_rows,
              "selection_sha256": hashlib.sha256(json.dumps(token_rows, separators=(",", ":")).encode()).hexdigest(),
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__, "numpy": np.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"vectors_sha256": result["vectors"]["sha256"],
                      "tokens": cursor, "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
