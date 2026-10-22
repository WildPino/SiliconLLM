"""C50-S4: Q4_K and Q6_K dequantization in PyTorch, a port of the pinned gguf-py (`quants.py`, Q4_K / Q6_K
`dequantize_blocks`) with the same fp32 operations in the same order, so that it can run on the GPU on T0's raw blocks.

Each operation is a separate eager kernel (no fused multiply-add), so the result is meant to be bit-identical to
gguf-py; `--check` verifies that on T0's tensors.

    python -I benchmarks/chatbot50/c50_s4_dequant.py --check [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

QK_K = 256
Q4_K_BYTES, Q6_K_BYTES = 144, 210


def _f16(b: torch.Tensor) -> torch.Tensor:
    """Two uint8 columns (little-endian) as fp16, then fp32."""
    return b.contiguous().view(torch.float16).to(torch.float32)


def dequant_q4_k(blocks: torch.Tensor) -> torch.Tensor:
    """blocks: uint8 [n, 144] -> fp32 [n, 256]."""
    n = blocks.shape[0]
    d, dmin = _f16(blocks[:, 0:2]), _f16(blocks[:, 2:4])
    scales, qs = blocks[:, 4:16], blocks[:, 16:144]
    s = scales.reshape(n, 3, 4)
    sd, sm, smd = s[:, 0], s[:, 1], s[:, 2]
    sc = torch.cat([sd & 0x3F, (smd & 0x0F) | ((sd >> 2) & 0x30)], -1)
    mn = torch.cat([sm & 0x3F, (smd >> 4) | ((sm >> 2) & 0x30)], -1)
    dd = (d * sc.to(torch.float32)).reshape(n, -1, 1)
    dm = (dmin * mn.to(torch.float32)).reshape(n, -1, 1)
    sh = torch.tensor([0, 4], dtype=torch.uint8, device=blocks.device).reshape(1, 1, 2, 1)
    q = (qs.reshape(n, -1, 1, 32) >> sh) & 0x0F
    q = q.reshape(n, -1, 32).to(torch.float32)
    return (dd * q - dm).reshape(n, QK_K)


def dequant_q6_k(blocks: torch.Tensor) -> torch.Tensor:
    """blocks: uint8 [n, 210] -> fp32 [n, 256]."""
    n = blocks.shape[0]
    ql, qh, scales, d = blocks[:, 0:128], blocks[:, 128:192], blocks[:, 192:208], blocks[:, 208:210]
    sc = scales.contiguous().view(torch.int8).to(torch.float32)
    dd = (_f16(d) * sc).reshape(n, QK_K // 16, 1)
    dev = blocks.device
    l = (ql.reshape(n, -1, 1, 64) >> torch.tensor([0, 4], dtype=torch.uint8, device=dev).reshape(1, 1, 2, 1)) & 0x0F
    l = l.reshape(n, -1, 32)
    h = (qh.reshape(n, -1, 1, 32) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=dev).reshape(1, 1, 4, 1)) & 0x03
    h = h.reshape(n, -1, 32)
    q = (l | (h << 4)).to(torch.int8) - 32
    q = q.reshape(n, QK_K // 16, -1).to(torch.float32)
    return (dd * q).reshape(n, QK_K)


DEQUANT = {"Q4_K": (dequant_q4_k, Q4_K_BYTES), "Q6_K": (dequant_q6_k, Q6_K_BYTES)}


def dequant(raw: torch.Tensor, qtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    """raw: uint8 tensor holding a whole tensor's blocks (any shape whose last dim is a multiple of the block size);
    shape: the logical numpy shape (rows last = the contiguous dimension)."""
    fn, nbytes = DEQUANT[qtype]
    return fn(raw.reshape(-1, nbytes)).reshape(shape)


def check(device: str) -> int:
    sys.path.insert(0, "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4/gguf-py")
    import gguf
    r = gguf.GGUFReader("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/ling-mini-2.0-a810f64-q4_k_m.gguf")
    names = ["blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight", "blk.2.ffn_down_exps.weight",
             "blk.0.attn_qkv.weight", "blk.2.attn_qkv.weight", "blk.10.ffn_up_exps.weight", "output.weight"]
    bad = 0
    for t in r.tensors:
        if t.name not in names:
            continue
        ref = gguf.quants.dequantize(t.data, t.tensor_type)
        t0 = time.time()
        got = dequant(torch.from_numpy(np.array(t.data)).to(device), t.tensor_type.name, ref.shape).cpu().numpy()
        same = ref.view(np.uint32) == got.view(np.uint32)
        ok = bool(same.all())
        bad += not ok
        print(f"{t.name:28s} {t.tensor_type.name} {ref.shape} bit-identical={ok} mismatches={int((~same).sum())} "
              f"{time.time() - t0:.2f}s", flush=True)
    print("DEQUANT CHECK", "PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(check(a.device))
