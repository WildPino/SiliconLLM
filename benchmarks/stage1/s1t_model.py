"""Stage 1 (a) part 2: Ling-mini-2.0 in plain torch, one layer at a time, with the k = 4 routing forms of the hook.

The model never sits whole on the GPU: each layer's weights are read from the HF checkpoint when the layer runs.  The
graph follows llama.cpp's bailingmoe2.cpp and the checkpoint's modeling_bailing_moe_v2.py:
- RMSNorm in fp32, weights bf16;
- fused QKV, per-head q/k RMSNorm, NEOX RoPE on the first 64 of 128 head dims (base 600000), GQA 16/4;
- layer 0 dense; layers 1-19 one shared expert plus 256 routed experts;
- sigmoid router in fp32 with a selection bias, 8 groups of 32 with 4 kept (group score = sum of the top 2), top 8,
  weights normalized over the 8 and scaled by 2.5.

Routing forms (`Form`), the same as s1_moe_hook.h:
- k = 8, mass 0: the model as trained;
- top k of the 8 slots by biased score, then
  - mass 0: weights renormalized over the kept slots (part 1's top4),
  - mass 8: kept p / S8 (part 1b's top4m8),
  - mass K > 8: kept p / D, D = S8 + the K - 8 largest unbiased probabilities among the 248 unselected (top4m16).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

# The pin lives on a USB disk, where the tools' scattered reads crawl; a copy on the NVMe disk, verified against the
# pin's sha256 (benchmarks/chatbot50/results/c50_a2_ling_mini_source_manifest.json), is used when present.
PIN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/sources/Ling-mini-2.0_a810f64")
FAST = Path("C:/SiliconLLM_fast/Ling-mini-2.0_a810f64")
SRC = FAST if (FAST / "model.safetensors.index.json").exists() else PIN
CFG = json.loads((SRC / "config.json").read_text(encoding="utf-8"))
D, H, HKV, HD, ROT = CFG["hidden_size"], CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"], 64
N_LAYER, N_EXPERT, TOPK, N_GROUP, TOPK_GROUP = (CFG["num_hidden_layers"], CFG["num_experts"], CFG["num_experts_per_tok"],
                                                CFG["n_group"], CFG["topk_group"])
SCALE, EPS, THETA = CFG["routed_scaling_factor"], CFG["rms_norm_eps"], CFG["rope_theta"]
assert int(HD * CFG["partial_rotary_factor"]) == ROT and CFG["first_k_dense_replace"] == 1


@dataclass(frozen=True)
class Form:
    k: int = 8
    mass: int = 0


class Weights:
    """Reads tensors by name from the sharded checkpoint."""

    def __init__(self, src: Path = SRC):
        self.src = src
        self.index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]

    def get(self, names: list[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for shard in sorted({self.index[n] for n in names}):
            with safe_open(str(self.src / shard), "pt") as f:
                for n in names:
                    if self.index[n] == shard:
                        out[n] = f.get_tensor(n)
        return out

    def layer(self, i: int) -> dict[str, torch.Tensor]:
        pre = f"model.layers.{i}."
        w = self.get([n for n in self.index if n.startswith(pre)])
        w = {n[len(pre):]: t for n, t in w.items()}
        if i >= 1:
            for p in ("gate_proj", "up_proj", "down_proj"):
                w[f"mlp.experts.{p}"] = torch.stack([w.pop(f"mlp.experts.{e}.{p}.weight") for e in range(N_EXPERT)])
        return w


def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype)


def rope(n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, device=device, dtype=torch.float32) / ROT))
    f = torch.outer(torch.arange(n, device=device, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos(), e.sin()


def attention(w: dict, x: torch.Tensor, cs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """x: [T, D] bf16, one sequence from position 0. Returns the attention output (before the residual add)."""
    t = x.shape[0]
    qkv = (x @ w["attention.query_key_value.weight"].t()).view(t, H + 2 * HKV, HD)
    q, k, v = qkv.split([H, HKV, HKV], dim=1)
    q, k = rms(q, w["attention.query_layernorm.weight"]), rms(k, w["attention.key_layernorm.weight"])
    cos, sin = (c.to(x.dtype)[:, None, :] for c in cs)

    def rot(z):
        zr, zp = z[..., :ROT], z[..., ROT:]
        z1, z2 = zr[..., :ROT // 2], zr[..., ROT // 2:]
        return torch.cat((zr * cos + torch.cat((-z2, z1), -1) * sin, zp), -1)

    q, k = rot(q).transpose(0, 1), rot(k).transpose(0, 1)
    v = v.transpose(0, 1)
    o = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True, enable_gqa=True)[0]
    return o.transpose(0, 1).reshape(t, H * HD) @ w["attention.dense.weight"].t()


def mlp(x: torch.Tensor, g: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return (F.silu(x @ g.t()) * (x @ u.t())) @ d.t()


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, form: Form) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [N, D]. Returns (ids [N, k], weights [N, k] fp32, already scaled by 2.5)."""
    n = x.shape[0]
    p = torch.sigmoid(x.float() @ gate_w.float().t())
    b = p + bias.float()
    gs = b.view(n, N_GROUP, -1).topk(2, dim=-1)[0].sum(-1)
    gi = gs.topk(TOPK_GROUP, dim=-1)[1]
    gm = torch.zeros(n, N_GROUP, device=x.device).scatter_(1, gi, 1).repeat_interleave(N_EXPERT // N_GROUP, dim=1)
    sel = b.masked_fill(gm == 0, float("-inf")).topk(TOPK, dim=-1)[1]       # sorted by biased score
    s8 = p.gather(1, sel).sum(-1, keepdim=True)
    if form.k == TOPK and form.mass == 0:
        return sel, p.gather(1, sel) / s8 * SCALE
    keep = sel[:, :form.k]
    pk = p.gather(1, keep)
    if form.mass == 0:
        den = pk.sum(-1, keepdim=True)
    elif form.mass == TOPK:
        den = s8
    else:
        rest = p.scatter(1, sel, float("-inf")).topk(form.mass - TOPK, dim=-1)[0].sum(-1, keepdim=True)
        den = s8 + rest
    return keep, pk / den * SCALE


def routed(x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, eg, eu, ed) -> torch.Tensor:
    """Sum over slots of w * expert(x). eg/eu/ed: sequences of per-expert matrices (unbind of the stacks)."""
    n, k = ids.shape
    flat = ids.reshape(-1)
    order = flat.argsort()
    counts = torch.bincount(flat, minlength=N_EXPERT).tolist()
    tok = order // k
    xs = x[tok]
    outs, s = [], 0
    for e, c in enumerate(counts):
        if c:
            outs.append(mlp(xs[s:s + c], eg[e], eu[e], ed[e]))
            s += c
    ys = torch.cat(outs).float() * wts.reshape(-1)[order].unsqueeze(-1)
    return torch.zeros(n, x.shape[1], device=x.device).index_add(0, tok, ys)


def moe(w: dict, x: torch.Tensor, form: Form) -> torch.Tensor:
    ids, wts = route(x, w["mlp.gate.weight"], w["mlp.gate.expert_bias"], form)
    y = routed(x, ids, wts, *(w[f"mlp.experts.{p}"].unbind(0) for p in ("gate_proj", "up_proj", "down_proj")))
    sh = mlp(x, w["mlp.shared_experts.gate_proj.weight"], w["mlp.shared_experts.up_proj.weight"],
             w["mlp.shared_experts.down_proj.weight"])
    return (y + sh.float()).to(x.dtype)


def to(w: dict, device) -> dict:
    return {n: (t.to(device) if n != "mlp.gate.expert_bias" else t.float().to(device)) for n, t in w.items()}


def layer_forward(i: int, w: dict, h: torch.Tensor, cs, form: Form) -> torch.Tensor:
    """One decoder layer on one sequence. h: [T, D] bf16 residual stream."""
    a = h + attention(w, rms(h, w["input_layernorm.weight"]), cs)
    u = rms(a, w["post_attention_layernorm.weight"])
    if i == 0:
        f = mlp(u, w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"], w["mlp.down_proj.weight"])
    else:
        f = moe(w, u, form)
    return a + f
