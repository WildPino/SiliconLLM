#!/usr/bin/env python3
"""METH-63: CPU storage and exact-route preflight for 10x expert count."""

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


LAYERS = 24
WIDTH = 896
FACTOR_RANK = 8
ROUTER_RANK = 64
TOP_K = 4
INPUTS = 32
SEED = 5151
LIMIT_RSS = 8 * (1 << 30)
LIMIT_SECONDS = 8 * 60
GRIDS = ((8, 16), (32, 40))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget(start):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    if elapsed > LIMIT_SECONDS or rss > LIMIT_RSS:
        raise RuntimeError(f"METH-63 budget exceeded: {elapsed:.3f}s RSS={rss}")
    return elapsed, rss


class ProductKeyLayer(nn.Module):
    def __init__(self, axis_a, axis_b, layer_id):
        super().__init__()
        self.axis_a = axis_a
        self.axis_b = axis_b
        self.experts = axis_a * axis_b
        self.a = nn.Parameter(torch.empty(self.experts, FACTOR_RANK, WIDTH))
        self.b = nn.Parameter(torch.zeros(self.experts, WIDTH, FACTOR_RANK))
        p_size = ROUTER_RANK * WIDTH
        a_size = axis_a * ROUTER_RANK
        self.router = nn.Parameter(torch.empty(p_size + a_size + axis_b * ROUTER_RANK))
        gen = torch.Generator(device="cpu").manual_seed(SEED + layer_id)
        route_gen = torch.Generator(device="cpu").manual_seed(SEED + 100000 + layer_id)
        with torch.no_grad():
            self.a.copy_(torch.randn(self.a.shape, generator=gen) * (0.02 / math.sqrt(WIDTH)))
            self.router[:p_size].copy_(torch.randn(p_size, generator=route_gen)
                                       * (0.1 / math.sqrt(WIDTH)))
            self.router[p_size:].copy_(torch.randn(self.router.numel() - p_size,
                                                  generator=route_gen)
                                        * (0.1 / math.sqrt(ROUTER_RANK)))

    def routes(self, x):
        p_size = ROUTER_RANK * WIDTH
        a_size = self.axis_a * ROUTER_RANK
        p = self.router[:p_size].view(ROUTER_RANK, WIDTH)
        key_a = self.router[p_size:p_size + a_size].view(self.axis_a, ROUTER_RANK)
        key_b = self.router[p_size + a_size:].view(self.axis_b, ROUTER_RANK)
        query = F.linear(x, p)
        scores_a = F.linear(query, key_a)
        scores_b = F.linear(query, key_b)
        top_a, ids_a = scores_a.topk(TOP_K, dim=-1)
        top_b, ids_b = scores_b.topk(TOP_K, dim=-1)
        pair_scores = (top_a.unsqueeze(-1) + top_b.unsqueeze(-2)).flatten(1)
        pair_ids = (ids_a.unsqueeze(-1) * self.axis_b + ids_b.unsqueeze(-2)).flatten(1)
        selected_scores, order = pair_scores.topk(TOP_K, dim=-1)
        selected_ids = pair_ids.gather(1, order)
        exhaustive = (scores_a.unsqueeze(-1) + scores_b.unsqueeze(-2)).flatten(1)
        oracle_scores, oracle_ids = exhaustive.topk(TOP_K, dim=-1)
        assert torch.equal(selected_ids, oracle_ids)
        assert torch.equal(selected_scores, oracle_scores)
        return selected_ids, selected_scores

    def forward(self, x):
        chosen, scores = self.routes(x)
        gate = F.softmax(scores, dim=-1)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", x, self.a[chosen]))
        residual = torch.einsum("nkr,nkdr->nkd", hidden, self.b[chosen])
        return x + (residual * gate.unsqueeze(-1)).sum(dim=1), chosen


@torch.inference_mode()
def run_grid(axis_a, axis_b, start):
    experts = axis_a * axis_b
    layers = []
    peak_rss = 0
    for li in range(LAYERS):
        layers.append(ProductKeyLayer(axis_a, axis_b, li))
        _, rss = budget(start)
        peak_rss = max(peak_rss, rss)
    pointers = [p.data_ptr() for layer in layers for p in (layer.a, layer.b, layer.router)]
    assert len(pointers) == len(set(pointers)) == LAYERS * 3
    params = sum(p.numel() * p.element_size() for layer in layers for p in
                 (layer.a, layer.b, layer.router))
    factor_bytes = sum((layer.a.numel() + layer.b.numel()) * 4 for layer in layers)
    router_bytes = params - factor_bytes
    assert factor_bytes == LAYERS * experts * 2 * FACTOR_RANK * WIDTH * 4
    inputs = torch.randn(INPUTS, WIDTH,
                         generator=torch.Generator(device="cpu").manual_seed(630063))
    input_sha = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
    selected_slots = []
    max_to_mean = []
    for layer in layers:
        assert all(bool(torch.isfinite(p).all()) for p in (layer.a, layer.b, layer.router))
        out, chosen = layer(inputs)
        assert torch.equal(out, inputs)
        count = torch.bincount(chosen.flatten(), minlength=experts)
        selected_slots.append(int((count > 0).sum()))
        max_to_mean.append(float(count.max() / count.float().mean()))
        _, rss = budget(start)
        peak_rss = max(peak_rss, rss)
    elapsed, rss = budget(start)
    return {"axis_a": axis_a, "axis_b": axis_b, "experts_per_layer": experts,
            "layers": LAYERS, "parameters_independent": True,
            "factor_fp32_parameter_bytes": factor_bytes,
            "router_fp32_parameter_bytes": router_bytes,
            "total_fp32_parameter_bytes": params,
            "expected_factor_bf16_inference_bytes": factor_bytes // 2,
            "selected_factor_bf16_bytes_per_token": LAYERS * TOP_K * 2 * FACTOR_RANK * WIDTH * 2,
            "input_sha256": input_sha, "oracle_cases": LAYERS * INPUTS,
            "oracle_mismatches": 0, "identity_mismatches": 0,
            "min_selected_slots": min(selected_slots),
            "max_selected_slots": max(selected_slots),
            "worst_max_to_mean_load": max(max_to_mean),
            "peak_rss_bytes": peak_rss, "rss_end_bytes": rss,
            "elapsed_cumulative_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    start = time.monotonic()
    grids = []
    for axis_a, axis_b in GRIDS:
        result = run_grid(axis_a, axis_b, start)
        grids.append(result)
        print(json.dumps({"experts": result["experts_per_layer"],
                          "oracle_cases": result["oracle_cases"],
                          "peak_rss_bytes": result["peak_rss_bytes"],
                          "elapsed_seconds": result["elapsed_cumulative_seconds"]}), flush=True)
    assert grids[0]["selected_factor_bf16_bytes_per_token"] == grids[1]["selected_factor_bf16_bytes_per_token"]
    result = {"experiment": "METH-63-10x-training-geometry-preflight",
              "source_sha256": digest(Path(__file__)), "grids": grids,
              "factor_growth": grids[1]["factor_fp32_parameter_bytes"] /
                               grids[0]["factor_fp32_parameter_bytes"],
              "decision": "storage_and_route_preflight_pass_only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
