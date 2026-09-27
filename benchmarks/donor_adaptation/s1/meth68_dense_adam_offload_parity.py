#!/usr/bin/env python3
"""METH-68: exact E128 initialization and dense Adam CPU offload parity."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth55_product_key_experts as M55
import meth64_sparse_factor_offload as M64


WIDTH = M15.M13.D
RANK = M15.R
AXIS_A, AXIS_B = 8, 16
SEED = 686868


def budget(start, device):
    state = {"elapsed_seconds": time.monotonic() - start,
             "rss_bytes": psutil.Process().memory_info().rss,
             "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if state["elapsed_seconds"] > 300 or state["rss_bytes"] > 8 * (1 << 30) or state["gpu_peak_allocated_bytes"] > 5 * (1 << 30):
        raise RuntimeError(f"METH-68 budget: {state}")
    return state


class OffloadedReference(nn.Module):
    def __init__(self, source, device):
        super().__init__()
        self.router = nn.Parameter(source.router.detach().clone().to(device))
        self.factors = M64.OffloadedFactors(AXIS_A * AXIS_B, WIDTH, RANK,
                                            0, b_nonzero=False)
        self.factors.a.copy_(source.a.detach().cpu())
        self.factors.b.copy_(source.b.detach().cpu())

    def routes(self, flat):
        p_size = 64 * WIDTH
        a_size = AXIS_A * 64
        p = self.router[:p_size].view(64, WIDTH)
        a_keys = self.router[p_size:p_size + a_size].view(AXIS_A, 64)
        b_keys = self.router[p_size + a_size:].view(AXIS_B, 64)
        q = F.linear(flat.float(), p)
        a_scores = F.linear(q, a_keys)
        b_scores = F.linear(q, b_keys)
        a_values, a_ids = a_scores.topk(4, dim=-1)
        b_values, b_ids = b_scores.topk(4, dim=-1)
        candidate_scores = (a_values.unsqueeze(-1) + b_values.unsqueeze(-2)).flatten(1)
        candidate_ids = (a_ids.unsqueeze(-1) * AXIS_B + b_ids.unsqueeze(-2)).flatten(1)
        selected_scores, order = candidate_scores.topk(4, dim=-1)
        return candidate_ids.gather(1, order), selected_scores

    def forward(self, x):
        flat = x.reshape(-1, WIDTH)
        chosen, scores = self.routes(flat)
        a, b = self.factors.gather(chosen, x.device)
        gate = F.softmax(scores, dim=-1).to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a.to(flat.dtype)))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b.to(flat.dtype))
        return x + (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(x)


class DenseCPUAdamW:
    def __init__(self, lr=3e-4):
        self.lr = lr
        self.state = {}
        self.step_count = 0

    @torch.no_grad()
    def step_bank(self, bank, sparse_gradients, name):
        if name not in self.state:
            self.state[name] = (torch.zeros_like(bank), torch.zeros_like(bank))
        first, second = self.state[name]
        first.mul_(0.9)
        second.mul_(0.999)
        for row, grad in sparse_gradients.items():
            first[row].add_(grad, alpha=0.1)
            second[row].addcmul_(grad, grad, value=0.001)
        for start in range(0, bank.shape[0], 16):
            stop = min(start + 16, bank.shape[0])
            numerator = first[start:stop] / (1 - 0.9 ** self.step_count)
            denominator = (second[start:stop] / (1 - 0.999 ** self.step_count)).sqrt().add_(1e-8)
            bank[start:stop].addcdiv_(numerator, denominator, value=-self.lr)

    def begin_step(self):
        self.step_count += 1


def map_to_dense(sparse, shape):
    result = torch.zeros(shape, dtype=torch.float32)
    for row, grad in sparse.items():
        result[row].copy_(grad)
    return result


def max_error(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    dense = M55.ProductKeyExperts(nn.Identity(), 0)
    offloaded = OffloadedReference(dense, device)
    dense = dense.to(device)
    init_error = {"a": max_error(dense.a.detach().cpu(), offloaded.factors.a),
                  "b": max_error(dense.b.detach().cpu(), offloaded.factors.b),
                  "router": max_error(dense.router.detach().cpu(), offloaded.router.detach().cpu())}
    assert all(value == 0 for value in init_error.values())
    dense_opt = torch.optim.AdamW([
        {"params": [dense.a, dense.b], "lr": 3e-4},
        {"params": [dense.router], "lr": 3e-5}], weight_decay=0.0)
    route_opt = torch.optim.AdamW([offloaded.router], lr=3e-5, weight_decay=0.0)
    cpu_opt = DenseCPUAdamW()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    steps = []
    for step in range(1, 4):
        x = torch.randn((32, WIDTH), generator=generator).to(device=device, dtype=torch.bfloat16)
        target = torch.randn((32, WIDTH), generator=generator).to(device=device)
        dense_opt.zero_grad(set_to_none=True)
        route_opt.zero_grad(set_to_none=True)
        offloaded.factors.grad_a.clear()
        offloaded.factors.grad_b.clear()
        dense_ids, dense_scores = dense.routes(x)
        off_ids, off_scores = offloaded.routes(x)
        assert torch.equal(dense_ids, off_ids)
        assert torch.equal(dense_scores, off_scores)
        dense_out = dense(x)
        off_out = offloaded(x)
        output_error = max_error(dense_out, off_out)
        dense_loss = (dense_out.float() - target).square().mean() * 1000
        off_loss = (off_out.float() - target).square().mean() * 1000
        dense_loss.backward()
        off_loss.backward()
        gradient_error = {
            "a": max_error(dense.a.grad.detach().cpu(), map_to_dense(offloaded.factors.grad_a, dense.a.shape)),
            "b": max_error(dense.b.grad.detach().cpu(), map_to_dense(offloaded.factors.grad_b, dense.b.shape)),
            "router": max_error(dense.router.grad.detach(), offloaded.router.grad.detach()),
        }
        dense_opt.step()
        cpu_opt.begin_step()
        cpu_opt.step_bank(offloaded.factors.a, offloaded.factors.grad_a, "a")
        cpu_opt.step_bank(offloaded.factors.b, offloaded.factors.grad_b, "b")
        route_opt.step()
        update_error = {
            "a": max_error(dense.a.detach().cpu(), offloaded.factors.a),
            "b": max_error(dense.b.detach().cpu(), offloaded.factors.b),
            "router": max_error(dense.router.detach(), offloaded.router.detach()),
        }
        state = {"step": step, "selected_unique_rows": len(set(dense_ids.flatten().tolist())),
                 "output_max_abs_error": output_error,
                 "gradient_max_abs_error": gradient_error,
                 "post_update_max_abs_error": update_error,
                 "runtime": budget(start, device)}
        steps.append(state)
        print(json.dumps(state), flush=True)
        assert output_error <= 1e-5
        assert max(gradient_error.values()) <= 2e-5
        assert max(update_error.values()) <= 2e-5
    result = {"experiment": "METH-68-dense-Adam-CPU-offload-parity",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "initial_max_abs_error": init_error,
              "steps": steps, "runtime": budget(start, device),
              "decision": "three_step_numerical_parity_pass"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
