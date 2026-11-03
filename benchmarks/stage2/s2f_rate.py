"""Stage 2 part 1, test 4: the training rate of one MoE layer's routed experts at Ling-mini's and Ling-flash's shapes,
on random weights (no model is loaded, nothing of Ling-flash is converted).  It times the code the trainings use
(s1t_model.routed with s1t_train.Tern, Adafactor) so the feasibility page's hours rest on a measured rate, not on a
fraction of the GPU's peak.  A timing: it runs on an idle machine.

Per shape, a chunk of C experts at a time (256 experts as fp32 latents overflow the 12 GB GPU into shared memory even at
Ling-mini's shape), one slot per token and 128 * C tokens per step routed uniformly at random, so each expert sees the
128 rows it sees at 4,096 tokens and k = 8 over 256:
    block:  latent fp32 weights through Tern, fwd + bwd + Adafactor step (stage 1 (b), test 2, M1 and M5);
    frozen: bf16 weights, gradient to the input only (an end-to-end pass through a frozen layer);
    scales: fixed ternary codes times trainable per-row scales, gradient to the input and the scales, Adafactor step on
            the scales (an end-to-end pass that trains the scales: M2).
Reported: median seconds per step over --steps after --warmup, and per token-layer (x 256 / C / 4,096).  A peak
above the GPU's memory makes the shape INVALID (Windows spills to shared memory and the time measures the spill).

Usage: python -I benchmarks/stage2/s2f_rate.py --json benchmarks/stage2/results/s2f_rate.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
import s1t_model as M  # noqa: E402
from s1t_train import Tern  # noqa: E402

SHAPES = {"mini": (2048, 512, 64), "flash": (4096, 1024, 32)}   # d, expert ffn, experts per chunk
K, E, B = 8, 256, 4096
ROWS = B * K // E                                                 # rows per expert in a 4,096-token step


def timed(fn, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {"config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                      "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}, "shapes": {}}
    print("CONFIG", json.dumps(out["config"]), flush=True)
    for name, (d, f, c) in SHAPES.items():
        g = torch.Generator(device=dev).manual_seed(a.seed)
        n = ROWS * c
        x = torch.randn(n, d, device=dev, dtype=torch.bfloat16, generator=g)
        ids = torch.randperm(n, device=dev, generator=g).remainder(c).unsqueeze(1)   # exactly ROWS rows per expert
        wts = torch.ones(n, 1, device=dev)
        lat = [torch.nn.Parameter(torch.randn(c, o, i, device=dev, generator=g) * 0.02)
               for o, i in ((f, d), (f, d), (d, f))]
        sc = [torch.nn.Parameter(torch.full((c, w.shape[1], 1), 0.02, device=dev)) for w in lat]
        opt = torch.optim.Adafactor(lat + sc, lr=1e-3)

        def block():
            q = [Tern.apply(w, s) for w, s in zip(lat, sc)]
            y = M.routed(x, ids, wts, *(t.unbind(0) for t in q))
            loss = y.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        frozen_w = [Tern.apply(w, s).detach() for w, s in zip(lat, sc)]
        xi = x.clone().requires_grad_(True)

        def frozen():
            y = M.routed(xi, ids, wts, *(t.unbind(0) for t in frozen_w))
            y.pow(2).mean().backward()
            xi.grad = None

        codes = [(w.detach() / s.detach()).clamp(-1, 1).round().to(torch.bfloat16) for w, s in zip(lat, sc)]
        ss = [torch.nn.Parameter(s.detach().clone()) for s in sc]
        opt_s = torch.optim.Adafactor(ss, lr=1e-3)

        def scales():
            q = [(cd.float() * s).to(torch.bfloat16) for cd, s in zip(codes, ss)]
            y = M.routed(xi, ids, wts, *(t.unbind(0) for t in q))
            loss = y.pow(2).mean()
            opt_s.zero_grad(set_to_none=True)
            loss.backward()
            opt_s.step()
            xi.grad = None

        r = {"d": d, "f": f, "chunk_experts": c, "tokens_per_step": n}
        for kind, fn in (("block", block), ("frozen", frozen), ("scales", scales)):
            s = timed(fn, a.warmup, a.steps)
            r[f"{kind}_s_per_step"] = round(s, 4)
            r[f"{kind}_ms_per_token_layer"] = round(1e3 * s * E / c / B, 4)
        r["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        r["valid"] = torch.cuda.max_memory_allocated() < torch.cuda.get_device_properties(0).total_memory
        out["shapes"][name] = r
        print("SHAPE", name, json.dumps(r), flush=True)
        del lat, sc, opt, frozen_w, xi, x, codes, ss, opt_s
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
