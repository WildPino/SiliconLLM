#!/usr/bin/env python3
"""Compare full C Qwen+M126 logits to the same stored weights in PyTorch."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F


REV = "7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
CORE_SHA = "6b2be143303510f15785783542649026b719488407f48b350de67f429e206029"
BANK_SHA = "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1"
IDS_SHA = {
    8: "b9ffb0827e01fec4ef13027d46e04a61421d2ef62c571cc3827aa049bd3013ed",
    64: "990924bdcf23204fc28ac091e960834dc775efc09cd6ce9a38f73f307ee13db7",
}
HEADER = struct.Struct("<8s8I")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Centered(nn.Module):
    def __init__(self, base, p, axis_a, axis_b, projection, keys, a, b):
        super().__init__()
        self.base = base
        for name, value in (("p", p), ("axis_a", axis_a), ("axis_b", axis_b),
                            ("projection", projection), ("keys", keys),
                            ("a", a), ("b", b)):
            self.register_buffer(name, value)

    def forward(self, x):
        dense = self.base(x)
        flat = x.reshape(-1, 896).to(torch.bfloat16)
        q = F.linear(flat.float(), self.p)
        av = F.linear(q, self.axis_a)
        bv = F.linear(q, self.axis_b)
        av4, ai = torch.topk(av, 4, dim=-1)
        bv4, bi = torch.topk(bv, 4, dim=-1)
        scores = (av4[:, :, None] + bv4[:, None, :]).reshape(-1, 16)
        parents = (ai[:, :, None] * 16 + bi[:, None, :]).reshape(-1, 16)
        top_scores, where = torch.topk(scores, 4, dim=-1)
        parents = parents.gather(-1, where)
        cq = F.linear(flat.float(), self.projection)
        chosen_keys = self.keys[parents]
        local = torch.einsum("nr,nkcr->nkc", cq, chosen_keys).argmax(-1)
        children = parents * 10 + local
        gates = F.softmax(top_scores, dim=-1).to(torch.bfloat16)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, self.a[parents]))
        output = torch.einsum("nkr,nkdr->nkd", hidden, self.b[children])
        residual = (output * gates.unsqueeze(-1)).sum(dim=1)
        return dense + residual.reshape_as(dense).float()


def read_bank(path, model, device):
    data = path.read_bytes()
    magic, layers, width, rank, na, nb, cr, children, factor_rank = HEADER.unpack_from(data)
    assert (magic, layers, width, rank, na, nb, cr, children, factor_rank) == (
        b"M126FB01", 24, 896, 64, 8, 16, 32, 10, 8)
    parent_bytes = (rank * width + (na + nb) * rank) * 4
    projection_bytes = cr * width * 4
    key_bytes = na * nb * children * cr * 4
    router_bytes = parent_bytes + projection_bytes + key_bytes
    a_bytes = na * nb * factor_rank * width * 2
    b_bytes = a_bytes * children
    layer_bytes = router_bytes + a_bytes + b_bytes
    assert len(data) == HEADER.size + layers * layer_bytes

    def fp32(offset, shape):
        return torch.from_numpy(np.frombuffer(data, dtype="<f4", count=int(np.prod(shape)),
                                              offset=offset).copy().reshape(shape)).to(device)

    def bf16(offset, shape):
        raw = np.frombuffer(data, dtype="<u2", count=int(np.prod(shape)), offset=offset).copy()
        return torch.from_numpy(raw.reshape(shape)).view(torch.bfloat16).to(device)

    for li in range(layers):
        base = HEADER.size + li * layer_bytes
        p = fp32(base, (64, 896))
        axis_a = fp32(base + 64 * 896 * 4, (8, 64))
        axis_b = fp32(base + (64 * 896 + 8 * 64) * 4, (16, 64))
        projection = fp32(base + parent_bytes, (32, 896))
        keys = fp32(base + parent_bytes + projection_bytes, (128, 10, 32))
        a = bf16(base + router_bytes, (128, 8, 896))
        b = bf16(base + router_bytes + a_bytes, (1280, 896, 8))
        layer = model.model.layers[li]
        layer.mlp = Centered(layer.mlp, p, axis_a, axis_b, projection, keys, a, b)
    return len(data)


def compare(native, reference):
    assert native.shape == reference.shape and np.isfinite(native).all()
    rel = [float(np.linalg.norm(native[i] - reference[i]) /
                 np.linalg.norm(reference[i])) for i in range(len(native))]
    native_top = native.argmax(-1).tolist()
    reference_top = reference.argmax(-1).tolist()
    return {"worst_relative_l2": max(rel), "relative_l2_by_position": rel,
            "native_top1": native_top, "reference_top1": reference_top,
            "top1_matches": sum(a == b for a, b in zip(native_top, reference_top)),
            "positions": len(native_top),
            "maximum_absolute_logit_difference": float(np.max(np.abs(native - reference)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--dense-logits", type=Path, required=True)
    ap.add_argument("--centered-logits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    assert sha(args.core) == CORE_SHA
    assert sha(args.bank) == BANK_SHA
    from huggingface_hub import hf_hub_download
    source = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "model.safetensors",
                             revision=REV, local_files_only=True)
    assert sha(source) == SOURCE_SHA
    ids = np.fromfile(args.ids, dtype="<i4")
    assert len(ids) in IDS_SHA and sha(args.ids) == IDS_SHA[len(ids)]
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(6)
    device = torch.device("cuda:0")
    assert torch.cuda.get_device_name(device) == "NVIDIA GeForce RTX 3060"
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", revision=REV, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).to(device).eval()
    tokens = torch.as_tensor(ids.astype(np.int64), device=device)[None]
    with torch.inference_mode():
        dense_ref = model(tokens, use_cache=False).logits[0].float().cpu().numpy()
    dense_native = np.fromfile(args.dense_logits, dtype="<f4").reshape(dense_ref.shape)
    dense = compare(dense_native, dense_ref)
    assert dense["top1_matches"] == len(ids) and dense["worst_relative_l2"] <= 1e-3
    bank_bytes = read_bank(args.bank, model, device)
    with torch.inference_mode():
        centered_ref = model(tokens, use_cache=False).logits[0].float().cpu().numpy()
    centered_native = np.fromfile(args.centered_logits, dtype="<f4").reshape(dense_ref.shape)
    centered = compare(centered_native, centered_ref)
    result = {"experiment": "METH-127-full-C-reference-parity",
              "source_sha256": SOURCE_SHA, "core_sha256": CORE_SHA,
              "factor_bank_sha256": BANK_SHA, "ids_sha256": IDS_SHA[len(ids)],
              "core_bytes": args.core.stat().st_size, "bank_bytes": bank_bytes,
              "dense": dense, "centered": centered,
              "centered_reference": "FP32_Qwen_core_plus_BF16_rounded_input_router_and_factors",
              "seconds": time.monotonic() - start,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "rss_bytes": psutil.Process().memory_info().rss}
    assert result["seconds"] <= 45 * 60 and result["rss_bytes"] <= 16 * (1 << 30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
