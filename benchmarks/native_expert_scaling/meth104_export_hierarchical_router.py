#!/usr/bin/env python3
"""Export the METH-99 hierarchical router and 96 native route fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "donor_adaptation/s1"))
import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth95_hierarchical_e1280_parity as M95


ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DIR / "meth102_hierarchical_external_manifest.json"
MANIFEST_SHA = "a5d1f6ef484eca561336d4625dd09db695034a139130fac490048da5a6650764"
TRAINING = DIR / "meth56_product_key_retention_result.json"
CHILD = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth99_hierarchical_e1280.pt"
CHILD_SHA = "e867d8f8877c7e5e0fdb6f4a788330cae3ffe2d02f638ec8a422d2ffc163b348"
HEADER = "<8s7I"
BANK_MAGIC = b"M104RT01"
FIXTURE_MAGIC = b"M104FX01"
SEED = 104104
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}
    if (result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS_BYTES
            or result["gpu_peak_bytes"] > MAX_GPU_BYTES):
        raise RuntimeError(f"METH-104 export resource stop: {result}")
    return result


def raw_fp32(tensor):
    assert tensor.dtype == torch.float32
    return tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()


def raw_bf16(tensor):
    return tensor.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for path, digest in ((MANIFEST, MANIFEST_SHA), (TRAINING, M57.TRAINING_SHA),
                         (CHILD, CHILD_SHA)):
        assert sha(path) == digest, path
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    first = manifest["items"][0]
    assert first["category"] == "code"
    assert M17.sha(np.asarray(first["prompt_ids"], dtype=np.int32).tobytes()) == first[
        "prompt_ids_sha256"]
    parent = json.loads(TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert parent["sha256"] == M57.CHECKPOINT_SHA
    assert sha(Path(parent["path"])) == M57.CHECKPOINT_SHA
    saved_parent = torch.load(parent["path"], map_location="cpu", weights_only=False)
    saved_child = torch.load(CHILD, map_location="cpu", weights_only=False)
    assert saved_child["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
    assert saved_child["source_sha256"] == M57.MODEL_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert sha(Path(source)) == M57.MODEL_SHA
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.config.use_cache = False
    wrappers = []
    captured = {}
    hooks = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            parent_wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            for key in ("a", "b", "router"):
                getattr(parent_wrapper, key).copy_(saved_parent["expert_state"][li][key].to(device))
            wrapper = M95.HierarchicalExperts(parent_wrapper, li).to(device)
            for key in ("a", "b", "router", "child_projection", "child_keys"):
                getattr(wrapper, key).copy_(saved_child["expert_state"][li][key].to(device))
            layer.mlp = wrapper
            wrappers.append(wrapper)
            def capture(mod, inputs, layer_id=li):
                x = inputs[0].detach()
                captured[layer_id] = (x[0, 0].clone(), x[0, -1].clone())
            hooks.append(wrapper.register_forward_pre_hook(capture))
    ids = torch.as_tensor(first["prompt_ids"], dtype=torch.long, device=device)[None]
    with torch.inference_mode():
        model(ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    assert set(captured) == set(range(M15.M13.L))

    dims = (M15.M13.L, M15.M13.D, M55.ROUTER_RANK,
            M55.AXIS_A, M55.AXIS_B, M95.CHILD_RANK, M95.CHILDREN)
    args.bank.parent.mkdir(parents=True, exist_ok=True)
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    with args.bank.open("wb") as file:
        file.write(struct.pack(HEADER, BANK_MAGIC, *dims))
        for li, saved in enumerate(saved_child["expert_state"]):
            assert torch.equal(saved["router"], saved_parent["expert_state"][li]["router"])
            file.write(raw_fp32(saved["router"]))
            file.write(raw_fp32(saved["child_projection"]))
            file.write(raw_fp32(saved["child_keys"]))
            budget(start, device)
    parent_bytes = (M55.ROUTER_RANK * M15.M13.D +
                    (M55.AXIS_A + M55.AXIS_B) * M55.ROUTER_RANK) * 4
    projection_bytes = M95.CHILD_RANK * M15.M13.D * 4
    keys_bytes = M15.E * M95.CHILDREN * M95.CHILD_RANK * 4
    layer_bytes = parent_bytes + projection_bytes + keys_bytes
    readback = np.memmap(args.bank, dtype=np.uint8, mode="r")
    assert len(readback) == struct.calcsize(HEADER) + M15.M13.L * layer_bytes
    for li, saved in enumerate(saved_child["expert_state"]):
        offset = struct.calcsize(HEADER) + li * layer_bytes
        for name, length in (("router", parent_bytes),
                             ("child_projection", projection_bytes),
                             ("child_keys", keys_bytes)):
            assert bytes(readback[offset:offset + length]) == raw_fp32(saved[name])
            offset += length
    del readback

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    fixture_rows = []
    with args.fixtures.open("wb") as file:
        file.write(struct.pack(HEADER, FIXTURE_MAGIC, *dims))
        for li, wrapper in enumerate(wrappers):
            randoms = (torch.randn(M15.M13.D, generator=gen).to(torch.bfloat16),
                       (torch.randn(M15.M13.D, generator=gen) * 0.1).to(torch.bfloat16))
            for kind, x in zip(("real_first", "real_last", "synthetic_unit", "synthetic_small"),
                               (*captured[li], *randoms)):
                x = x.to(device)
                with torch.inference_mode():
                    parents, scores = wrapper.routes(x[None])
                    children = wrapper.child_route(x[None], parents)
                    gates = F.softmax(scores, dim=-1).to(x.dtype)
                parent_ids = parents[0].cpu().numpy().astype("<u4")
                child_ids = children[0].cpu().numpy().astype("<u4")
                gate_values = gates[0].float().cpu().numpy().astype("<f4")
                raw_x = raw_bf16(x)
                file.write(struct.pack("<I", li))
                file.write(raw_x)
                file.write(parent_ids.tobytes())
                file.write(child_ids.tobytes())
                file.write(gate_values.tobytes())
                fixture_rows.append({"layer": li, "kind": kind,
                                     "activation_sha256": hashlib.sha256(raw_x).hexdigest(),
                                     "parents": parent_ids.tolist(),
                                     "children": child_ids.tolist()})
            budget(start, device)
    result = {"experiment": "METH-104-hierarchical-E1280-native-route-export",
              "source_sha256": M57.MODEL_SHA,
              "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
              "child_checkpoint_sha256": CHILD_SHA,
              "manifest_sha256": MANIFEST_SHA,
              "prompt_source_id": first["source_id"],
              "prompt_ids_sha256": first["prompt_ids_sha256"],
              "bank": {"path": str(args.bank.resolve()), "sha256": sha(args.bank),
                       "bytes": args.bank.stat().st_size},
              "fixtures": {"path": str(args.fixtures.resolve()), "sha256": sha(args.fixtures),
                           "bytes": args.fixtures.stat().st_size},
              "dimensions": dict(zip(("layers", "width", "parent_rank", "axis_a", "axis_b",
                                      "child_rank", "children_per_parent"), dims)),
              "fixture_rows": fixture_rows,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bank_sha256": result["bank"]["sha256"],
                      "fixture_sha256": result["fixtures"]["sha256"],
                      "fixture_count": len(fixture_rows),
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
