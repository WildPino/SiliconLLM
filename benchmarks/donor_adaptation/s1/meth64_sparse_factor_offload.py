#!/usr/bin/env python3
"""METH-64: selected CPU factors with sparse row-gradient accumulation."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F


MAX_GPU = int(4.5 * (1 << 30))
MAX_RSS = 10 * (1 << 30)
MAX_SECONDS = 8 * 60
SEED = 640064


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    gpu = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS or gpu > MAX_GPU:
        raise RuntimeError(f"METH-64 budget: seconds={elapsed:.3f} rss={rss} gpu={gpu}")
    return {"elapsed_seconds": elapsed, "rss_bytes": rss, "gpu_peak_allocated_bytes": gpu}


class Router(torch.nn.Module):
    def __init__(self, axis_a, axis_b, width, rank, device, layer_id):
        super().__init__()
        self.axis_a, self.axis_b = axis_a, axis_b
        self.width, self.rank = width, rank
        gen = torch.Generator(device="cpu").manual_seed(SEED + layer_id)
        p = torch.randn(rank, width, generator=gen) * (0.1 / math.sqrt(width))
        a = torch.randn(axis_a, rank, generator=gen) * (0.1 / math.sqrt(rank))
        b = torch.randn(axis_b, rank, generator=gen) * (0.1 / math.sqrt(rank))
        self.p = torch.nn.Parameter(p.to(device))
        self.key_a = torch.nn.Parameter(a.to(device))
        self.key_b = torch.nn.Parameter(b.to(device))

    def forward(self, x):
        q = F.linear(x.float(), self.p)
        axis_a = F.linear(q, self.key_a)
        axis_b = F.linear(q, self.key_b)
        top_a, ids_a = axis_a.topk(4, dim=-1)
        top_b, ids_b = axis_b.topk(4, dim=-1)
        scores = (top_a.unsqueeze(-1) + top_b.unsqueeze(-2)).flatten(1)
        pairs = (ids_a.unsqueeze(-1) * self.axis_b + ids_b.unsqueeze(-2)).flatten(1)
        selected_scores, order = scores.topk(4, dim=-1)
        selected_ids = pairs.gather(1, order)
        exhaustive = (axis_a.unsqueeze(-1) + axis_b.unsqueeze(-2)).flatten(1)
        oracle_scores, oracle_ids = exhaustive.topk(4, dim=-1)
        assert torch.equal(selected_ids, oracle_ids)
        assert torch.equal(selected_scores, oracle_scores)
        return selected_ids, selected_scores


class OffloadedFactors:
    def __init__(self, experts, width, rank, layer_id, b_nonzero=True):
        gen = torch.Generator(device="cpu").manual_seed(SEED + 10000 + layer_id)
        self.a = torch.randn(experts, rank, width, generator=gen) * (0.02 / math.sqrt(width))
        self.b = torch.randn(experts, width, rank, generator=gen) * (0.005 if b_nonzero else 0.0)
        assert self.a.device.type == self.b.device.type == "cpu"
        self.grad_a = {}
        self.grad_b = {}

    def _accumulate(self, kind, ids, grad):
        flat_ids = ids.reshape(-1)
        flat_grad = grad.detach().float().to("cpu").reshape(flat_ids.numel(), -1)
        unique, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
        sums = torch.zeros(unique.numel(), flat_grad.shape[1], dtype=torch.float32)
        sums.index_add_(0, inverse, flat_grad)
        destination = self.grad_a if kind == "a" else self.grad_b
        shape = self.a.shape[1:] if kind == "a" else self.b.shape[1:]
        for position, row_id in enumerate(unique.tolist()):
            value = sums[position].view(shape)
            if row_id in destination:
                destination[row_id].add_(value)
            else:
                destination[row_id] = value.clone()
        return grad

    def gather(self, chosen, device):
        cpu_ids = chosen.detach().to("cpu")
        a = self.a[cpu_ids].to(device).detach().requires_grad_()
        b = self.b[cpu_ids].to(device).detach().requires_grad_()
        a.register_hook(lambda grad: self._accumulate("a", cpu_ids, grad))
        b.register_hook(lambda grad: self._accumulate("b", cpu_ids, grad))
        return a, b


def residual(x, chosen, scores, a, b):
    gate = F.softmax(scores, dim=-1).to(x.dtype)
    hidden = F.silu(torch.einsum("nd,nkrd->nkr", x, a))
    out = torch.einsum("nkr,nkdr->nkd", hidden, b)
    return x + (out * gate.unsqueeze(-1)).sum(dim=1)


def small_parity(device, start):
    axis_a, axis_b, width, rank, samples = 5, 8, 16, 3, 7
    expert_count = axis_a * axis_b
    off = OffloadedFactors(expert_count, width, rank, 0)
    router_off = Router(axis_a, axis_b, width, 7, device, 0)
    router_dense = Router(axis_a, axis_b, width, 7, device, 0)
    with torch.no_grad():
        for target, source in zip(router_dense.parameters(), router_off.parameters()):
            target.copy_(source)
    inputs = torch.randn(samples, width,
                         generator=torch.Generator(device="cpu").manual_seed(SEED + 1)).to(device)
    ids_off, scores_off = router_off(inputs)
    ids_dense, scores_dense = router_dense(inputs)
    assert torch.equal(ids_off, ids_dense)
    assert torch.equal(scores_off, scores_dense)
    a_leaf, b_leaf = off.gather(ids_off, device)
    out_off = residual(inputs, ids_off, scores_off, a_leaf, b_leaf)
    a_dense = off.a.clone().to(device).detach().requires_grad_()
    b_dense = off.b.clone().to(device).detach().requires_grad_()
    out_dense = residual(inputs, ids_dense, scores_dense,
                         a_dense[ids_dense], b_dense[ids_dense])
    forward_error = float((out_off - out_dense).abs().max())
    out_off.square().sum().backward()
    out_dense.square().sum().backward()
    dense_a_rows = set(torch.nonzero(a_dense.grad.abs().flatten(1).sum(1) > 0).flatten().tolist())
    dense_b_rows = set(torch.nonzero(b_dense.grad.abs().flatten(1).sum(1) > 0).flatten().tolist())
    assert set(off.grad_a) == dense_a_rows
    assert set(off.grad_b) == dense_b_rows
    error_a = max(float((off.grad_a[i] - a_dense.grad[i].cpu()).abs().max()) for i in dense_a_rows)
    error_b = max(float((off.grad_b[i] - b_dense.grad[i].cpu()).abs().max()) for i in dense_b_rows)
    assert forward_error == 0 and error_a <= 2e-6 and error_b <= 2e-6
    state = budget(start, device)
    return {"routes_equal": True, "forward_max_abs_error": forward_error,
            "a_gradient_rows": len(dense_a_rows), "b_gradient_rows": len(dense_b_rows),
            "a_gradient_max_abs_error": error_a, "b_gradient_max_abs_error": error_b,
            **state}


def scale_check(device, start):
    layers, axis_a, axis_b, width, rank, samples = 24, 32, 40, 896, 8, 32
    factors = []
    routers = []
    for layer in range(layers):
        factors.append(OffloadedFactors(axis_a * axis_b, width, rank, layer))
        routers.append(Router(axis_a, axis_b, width, 64, device, layer))
        budget(start, device)
    assert all(f.a.device.type == f.b.device.type == "cpu" for f in factors)
    inputs = torch.randn(samples, width,
                         generator=torch.Generator(device="cpu").manual_seed(SEED + 2)).to(device)
    loss = torch.zeros((), device=device)
    selected = []
    for f, router in zip(factors, routers):
        ids, scores = router(inputs)
        a_leaf, b_leaf = f.gather(ids, device)
        out = residual(inputs, ids, scores, a_leaf, b_leaf)
        loss = loss + out.square().mean()
        selected.append(set(ids.flatten().tolist()))
        budget(start, device)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for router in routers for p in router.parameters())
    assert all(set(f.grad_a) == set(f.grad_b) == ids
               for f, ids in zip(factors, selected))
    assert all(bool(torch.isfinite(row).all())
               for f in factors for row in list(f.grad_a.values()) + list(f.grad_b.values()))
    state = budget(start, device)
    bytes_per_direction = layers * samples * 4 * 2 * rank * width * 4
    return {"layers": layers, "experts_per_layer": axis_a * axis_b,
            "samples_per_layer": samples, "oracle_cases": layers * samples,
            "oracle_mismatches": 0, "gpu_bank_tensors": 0,
            "selected_row_bytes_per_direction": bytes_per_direction,
            "min_unique_gradient_rows": min(len(f.grad_a) for f in factors),
            "max_unique_gradient_rows": max(len(f.grad_a) for f in factors),
            "loss": float(loss.detach()), **state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(6)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parity = small_parity(device, start)
    print(json.dumps({"small_parity": parity}), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    scale = scale_check(device, start)
    print(json.dumps({"scale": scale}), flush=True)
    result = {"experiment": "METH-64-sparse-factor-offload",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "small_parity": parity, "e1280_scale": scale,
              "decision": "offload_apparatus_pass_only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
