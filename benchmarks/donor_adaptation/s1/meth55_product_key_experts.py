"""Trainable rank-64 product-key route with E128 distinct residual experts."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15


AXIS_A = 8
AXIS_B = 16
ROUTER_RANK = 64
assert AXIS_A * AXIS_B == M15.E


class ProductKeyExperts(M15.ResidualExperts):
    def __init__(self, base, layer_id):
        super().__init__(base, layer_id)
        width = M15.M13.D
        p_size = ROUTER_RANK * width
        u_size = AXIS_A * ROUTER_RANK
        v_size = AXIS_B * ROUTER_RANK
        self.router = nn.Parameter(torch.empty(p_size + u_size + v_size, dtype=torch.float32))
        self.oracle_checks = False
        self.collect_load = False
        self.route_counts = torch.zeros(M15.E, dtype=torch.int64)
        gen = torch.Generator(device="cpu").manual_seed(M15.SEED + 100000 + layer_id)
        with torch.no_grad():
            self.router[:p_size].copy_(
                torch.randn((p_size,), generator=gen) * (0.1 / math.sqrt(width)))
            self.router[p_size:].copy_(
                torch.randn((u_size + v_size,), generator=gen) * (0.1 / math.sqrt(ROUTER_RANK)))

    def router_views(self):
        width = M15.M13.D
        p_size = ROUTER_RANK * width
        u_size = AXIS_A * ROUTER_RANK
        p = self.router[:p_size].view(ROUTER_RANK, width)
        u = self.router[p_size:p_size + u_size].view(AXIS_A, ROUTER_RANK)
        v = self.router[p_size + u_size:].view(AXIS_B, ROUTER_RANK)
        return p, u, v

    def routes(self, flat):
        p, u, v = self.router_views()
        q = F.linear(flat.float(), p)
        axis_a = F.linear(q, u)
        axis_b = F.linear(q, v)
        a_scores, a_ids = torch.topk(axis_a, M15.K, dim=-1)
        b_scores, b_ids = torch.topk(axis_b, M15.K, dim=-1)
        candidate_scores = (a_scores.unsqueeze(-1) + b_scores.unsqueeze(-2)).flatten(1)
        candidate_ids = (a_ids.unsqueeze(-1) * AXIS_B + b_ids.unsqueeze(-2)).flatten(1)
        selected_scores, which = torch.topk(candidate_scores, M15.K, dim=-1)
        chosen = candidate_ids.gather(1, which)
        if self.oracle_checks:
            dense_scores = (axis_a.unsqueeze(-1) + axis_b.unsqueeze(-2)).flatten(1)
            exact = torch.topk(dense_scores, M15.K, dim=-1).indices
            if not torch.equal(chosen.sort(dim=1).values, exact.sort(dim=1).values):
                raise AssertionError("product-key 16-pair route differs from E128 exhaustive route")
        if self.collect_load:
            self.route_counts += torch.bincount(chosen.detach().flatten(), minlength=M15.E).cpu()
        return chosen, selected_scores

    def forward(self, x):
        dense = self.base(x)
        if not self.enabled:
            return dense
        flat = x.reshape(-1, M15.M13.D)
        chosen, selected_scores = self.routes(flat)
        gate = F.softmax(selected_scores, dim=-1).to(flat.dtype)
        a = self.a[chosen].to(flat.dtype)
        b = self.b[chosen].to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b)
        residual = (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)
        return dense + residual


def load_summary(wrappers):
    result = []
    for wrapper in wrappers:
        counts = wrapper.route_counts
        assert int(counts.sum()) > 0
        result.append({
            "selected_slots": int((counts > 0).sum()),
            "max_to_mean": float(counts.max() / counts.float().mean()),
            "min_count": int(counts.min()),
            "max_count": int(counts.max()),
            "total_selections": int(counts.sum()),
        })
    return result
