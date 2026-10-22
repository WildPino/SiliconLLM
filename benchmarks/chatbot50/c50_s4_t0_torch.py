"""C50-S4: T0 as a PyTorch module, built from T0's own GGUF tensors (dequantized with the pinned llama.cpp's gguf-py),
so that training sees exactly the weights the engine and llama.cpp multiply.

Semantics (bailingmoe2, Ling-mini-2.0 config): fused QKV, per-head RMSNorm on q and k, NEOX RoPE on the first 64 of
the 128 head dimensions (theta 600000), GQA 16/4, layer 0 a dense SiLU FFN; layers 1-19 a shared expert plus 8 of 256
routed experts: sigmoid scores, expert bias for selection only, 8 groups of which the best 4 (by the sum of their top-2
biased scores) are kept, weights = unbiased scores of the selected experts normalized to sum 1, times 2.5.

Weights are dequantized per layer on demand (`W.get`), so a forward can run with one layer's experts in memory. The
forward is device- and dtype-agnostic: `W.dev(i)` places layer i, and matmuls run in the weights' dtype with the
residual stream, norms and router in fp32 (on the CPU everything is fp32; `c50_s4_t0_gpu.py` holds fp16 weights).

    python -I benchmarks/chatbot50/c50_s4_t0_torch.py --mve1 --docs 2 --tokens 256   # MVE-1, fidelity against P0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
sys.path.insert(0, str(C / "tools/llama.cpp-5b335f4/gguf-py"))
import gguf  # noqa: E402

T0_GGUF = C / "gguf/ling-mini-2.0-a810f64-q4_k_m.gguf"
T0_SHA = "44de7315"  # prefix of the pin; the full hash is in the A4/S0 records
H, NH, NKV, HD, ROPE, THETA, EPS = 2048, 16, 4, 128, 64, 600000.0, 1e-6
N_LAYERS, N_EXP, TOPK, N_GROUP, TOPK_GROUP, SCALE = 20, 256, 8, 8, 4, 2.5


class T0Weights:
    def __init__(self, path: Path = T0_GGUF):
        self.r = gguf.GGUFReader(str(path))
        self.t = {t.name: t for t in self.r.tensors}

    def get(self, name: str) -> torch.Tensor:
        t = self.t[name]
        a = gguf.quants.dequantize(t.data, t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

    def embed_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows of the token embedding, dequantized one row at a time (the full table is 157184 x 2048)."""
        t = self.t["token_embd.weight"]
        rows = [gguf.quants.dequantize(t.data[i:i + 1], t.tensor_type)[0] for i in ids.tolist()]
        return torch.from_numpy(np.stack(rows).astype(np.float32))


    def dev(self, i: int) -> torch.device:
        return torch.device("cpu")


def mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T in w's dtype, returned in fp32 (a no-op cast on the fp32 CPU path)."""
    return (x.to(w.dtype) @ w.T).float()


def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    inv = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=x.device) / ROPE))
    f = pos[:, None].float() * inv[None, :]
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
    r, p = x[..., :ROPE], x[..., ROPE:]
    half = ROPE // 2
    rot = torch.cat([-r[..., half:], r[..., :half]], -1)
    return torch.cat([r * cos + rot * sin, p], -1)


def attention(x: torch.Tensor, W: T0Weights, i: int) -> torch.Tensor:
    T = x.shape[0]
    h = rms(x, W.get(f"blk.{i}.attn_norm.weight"))
    qkv = mm(h, W.get(f"blk.{i}.attn_qkv.weight"))
    q = qkv[:, :NH * HD].view(T, NH, HD)
    k = qkv[:, NH * HD:(NH + NKV) * HD].view(T, NKV, HD)
    v = qkv[:, (NH + NKV) * HD:].view(T, NKV, HD)
    q, k = rms(q, W.get(f"blk.{i}.attn_q_norm.weight")), rms(k, W.get(f"blk.{i}.attn_k_norm.weight"))
    pos = torch.arange(T, device=x.device)
    q, k = rope(q.transpose(0, 1), pos), rope(k.transpose(0, 1), pos)
    v = v.transpose(0, 1)
    k, v = k.repeat_interleave(NH // NKV, 0), v.repeat_interleave(NH // NKV, 0)
    wo = W.get(f"blk.{i}.attn_output.weight")
    q, k, v = q.to(wo.dtype), k.to(wo.dtype), v.to(wo.dtype)
    o = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
    return x + mm(o.transpose(0, 1).reshape(T, NH * HD), wo)


def swiglu(x, g, u, d):
    return mm(F.silu(mm(x, g)) * mm(x, u), d)


def route(h: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor):
    s = torch.sigmoid(mm(h, gate_w))
    b = s + bias
    g = b.view(-1, N_GROUP, N_EXP // N_GROUP).topk(2, -1)[0].sum(-1)
    keep = torch.zeros_like(g).scatter_(1, g.topk(TOPK_GROUP, -1)[1], 1.0)
    mask = keep[:, :, None].expand(-1, N_GROUP, N_EXP // N_GROUP).reshape(-1, N_EXP).bool()
    idx = b.masked_fill(~mask, float("-inf")).topk(TOPK, -1)[1]
    w = s.gather(1, idx)
    return idx, w / (w.sum(-1, keepdim=True) + 1e-20) * SCALE


def ffn(x: torch.Tensor, W: T0Weights, i: int) -> torch.Tensor:
    h = rms(x, W.get(f"blk.{i}.ffn_norm.weight"))
    if i == 0:
        return x + swiglu(h, W.get("blk.0.ffn_gate.weight"), W.get("blk.0.ffn_up.weight"), W.get("blk.0.ffn_down.weight"))
    idx, w = route(h, W.get(f"blk.{i}.ffn_gate_inp.weight"), W.get(f"blk.{i}.exp_probs_b.bias"))
    y = swiglu(h, W.get(f"blk.{i}.ffn_gate_shexp.weight"), W.get(f"blk.{i}.ffn_up_shexp.weight"),
               W.get(f"blk.{i}.ffn_down_shexp.weight"))
    G, U, D = (W.get(f"blk.{i}.ffn_{n}_exps.weight") for n in ("gate", "up", "down"))
    for e in idx.unique().tolist():
        tok, slot = (idx == e).nonzero(as_tuple=True)
        y.index_add_(0, tok, swiglu(h[tok], G[e], U[e], D[e]) * w[tok, slot][:, None])
    return x + y


@torch.no_grad()
def logits(ids: torch.Tensor, W: T0Weights) -> torch.Tensor:
    x = W.embed_rows(ids)
    for i in range(N_LAYERS):
        x = x.to(W.dev(i))
        x = ffn(attention(x, W, i), W, i)
    return mm(rms(x, W.get("output_norm.weight")), W.get("output.weight"))


def mve1(n_docs: int, n_tok: int, out: Path) -> dict:
    P0 = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"), ("logp_target", "<f4"),
                   ("logp_top1", "<f4"), ("lse", "<f8")])
    refs = {k: np.fromfile(C / f"a4/{k}/docs.pos.bin", dtype=P0) for k in ("p0", "bf16")}
    lines = (C / "a4/inputs/docs.tsv").read_text(encoding="ascii").splitlines()[:n_docs]
    W = T0Weights()
    res, t0 = [], time.time()
    for line in lines:
        d, s = line.split("\t", 1)
        ids = torch.tensor([int(x) for x in s.split(",")][:n_tok])
        lp = torch.log_softmax(logits(ids, W).double(), -1)
        for k, ref in refs.items():
            r = ref[(ref["doc"] == int(d)) & (ref["pos"] < len(ids))]
            assert np.array_equal(r["target"], ids[r["pos"]].numpy()), "targets differ"
            mine = lp[torch.from_numpy(r["pos"] - 1), torch.from_numpy(r["target"]).long()].numpy()
            top1 = lp[torch.from_numpy(r["pos"] - 1)].argmax(-1).numpy()
            dl = mine - r["logp_target"]
            res.append({"doc": int(d), "ref": k, "positions": int(len(r)), "mean_dlogp": float(dl.mean()),
                        "mean_abs_dlogp": float(np.abs(dl).mean()), "max_abs_dlogp": float(np.abs(dl).max()),
                        "top1_agree": float((top1 == r["top1"]).mean())})
            print(json.dumps(res[-1]), flush=True)
        a, b = (ref[(ref["doc"] == int(d)) & (ref["pos"] < len(ids))] for ref in (refs["p0"], refs["bf16"]))
        dl = a["logp_target"].astype(np.float64) - b["logp_target"]
        res.append({"doc": int(d), "ref": "p0_vs_bf16 (llama.cpp, yardstick)", "positions": int(len(a)),
                    "mean_dlogp": float(dl.mean()), "mean_abs_dlogp": float(np.abs(dl).mean()),
                    "max_abs_dlogp": float(np.abs(dl).max()), "top1_agree": float((a["top1"] == b["top1"]).mean())})
        print(json.dumps(res[-1]), flush=True)
    rec = {"cell": "C50-S4 MVE-1", "gguf": str(T0_GGUF), "docs": n_docs, "tokens": n_tok, "wall_s": round(time.time() - t0, 1),
           "rows": res}
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mve1", action="store_true")
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--out", type=Path, default=C / "s4/mve1.json")
    a = ap.parse_args()
    torch.set_num_threads(6)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mve1:
        mve1(a.docs, a.tokens, a.out)
