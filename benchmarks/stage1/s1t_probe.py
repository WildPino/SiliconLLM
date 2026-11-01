"""Stage 1 (a) part 2, sizing probe: can one Ling-mini MoE layer be trained on the local GPU (RTX 3060, 12 GB)?

Loads layer L's router and 256 routed experts from the HF checkpoint as fp32 parameters, runs forward + backward + an
optimizer step at k = 4 on N synthetic tokens, and prints the peak GPU memory and the time per step. A sizing probe:
its numbers size part 2's protocol, they judge nothing.
Usage: python -I s1t_probe.py [--layer 5] [--tokens 8192] [--steps 5] [--optim adamw|adafactor]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

SRC = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/sources/Ling-mini-2.0_a810f64")
N_EXPERT, D, DI = 256, 2048, 512


def load_layer(layer: int) -> dict[str, torch.Tensor]:
    index = json.loads((SRC / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    pre = f"model.layers.{layer}.mlp."
    want = {k: v for k, v in index.items() if k.startswith(pre) and "shared" not in k}
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(want.values())):
        with safe_open(str(SRC / shard), "pt") as f:
            for k in (k for k, v in want.items() if v == shard):
                out[k[len(pre):]] = f.get_tensor(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device("cuda")
    t0 = time.time()
    w = load_layer(a.layer)
    stack = lambda name: torch.stack([w[f"experts.{i}.{name}.weight"] for i in range(N_EXPERT)]).float()  # noqa: E731
    params = {"gate": torch.nn.Parameter(w["gate.weight"].float().to(dev)),
              "wg": torch.nn.Parameter(stack("gate_proj").to(dev)),
              "wu": torch.nn.Parameter(stack("up_proj").to(dev)),
              "wd": torch.nn.Parameter(stack("down_proj").to(dev))}
    bias = w["gate.expert_bias"].float().to(dev)
    del w
    load_s = time.time() - t0
    opt = (torch.optim.AdamW if a.optim == "adamw" else torch.optim.Adafactor)(params.values(), lr=5e-5)
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(a.tokens, D, device=dev, generator=g) * 0.5
    target = torch.randn(a.tokens, D, device=dev, generator=g) * 0.1
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(a.steps):
        torch.cuda.synchronize()
        s = time.time()
        logits = x @ params["gate"].t()
        scores = torch.sigmoid(logits)
        biased = scores + bias
        grp = biased.view(a.tokens, 8, -1).topk(2, dim=-1)[0].sum(-1).topk(4, dim=-1)[1]
        mask = torch.zeros(a.tokens, 8, device=dev).scatter_(1, grp, 1).repeat_interleave(N_EXPERT // 8, dim=1)
        top8 = biased.masked_fill(mask == 0, float("-inf")).topk(8, dim=-1)[1]
        keep = top8[:, :a.k]                                   # top k of the 8 by biased score
        s8 = scores.gather(1, top8).sum(-1, keepdim=True)
        wts = scores.gather(1, keep) / s8 * 2.5                # mass of the 8 (part 1b's k2 = 8 form)
        flat = keep.reshape(-1)
        order = flat.argsort()
        counts = torch.bincount(flat, minlength=N_EXPERT).tolist()
        tok = order // a.k
        xs = x[tok].to(torch.bfloat16)                         # token-slot pairs sorted by expert
        # unbind: one backward for all experts (indexing [e] would allocate a full-size gradient per expert)
        wg, wu, wd = (params[n].to(torch.bfloat16).unbind(0) for n in ("wg", "wu", "wd"))
        outs, start = [], 0
        for e, c in enumerate(counts):
            if c:
                xi = xs[start:start + c]
                outs.append((F.silu(xi @ wg[e].t()) * (xi @ wu[e].t())) @ wd[e].t())
                start += c
        ys = torch.cat(outs).float() * wts.reshape(-1)[order].unsqueeze(-1)
        y = torch.zeros(a.tokens, D, device=dev).index_add(0, tok, ys)
        loss = F.mse_loss(y, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - s)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps({"layer": a.layer, "tokens": a.tokens, "k": a.k, "optim": a.optim, "load_s": round(load_s, 1),
                      "step_s": [round(t, 3) for t in times], "peak_gib": round(peak, 2),
                      "tok_per_s_after_first": round(a.tokens * (len(times) - 1) / sum(times[1:]), 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
