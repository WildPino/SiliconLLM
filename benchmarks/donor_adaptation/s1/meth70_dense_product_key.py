"""Generalized dense METH-55 product-key residuals for E128/E1280."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15


ROUTER_RANK = 64


class DenseProductKeyExperts(nn.Module):
    def __init__(self, base, layer_id, axis_a, axis_b):
        super().__init__()
        self.base = base
        self.axis_a = axis_a
        self.axis_b = axis_b
        experts = axis_a * axis_b
        width = M15.M13.D
        rank = M15.R
        self.enabled = True
        self.oracle_checks = False
        self.collect_load = False
        self.route_counts = torch.zeros(experts, dtype=torch.int64)
        self.a = nn.Parameter(torch.empty(experts, rank, width, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(experts, width, rank, dtype=torch.float32))
        self.router = nn.Parameter(torch.empty(ROUTER_RANK * width +
                                               (axis_a + axis_b) * ROUTER_RANK,
                                               dtype=torch.float32))
        factor_gen = torch.Generator(device="cpu").manual_seed(M15.SEED + layer_id)
        router_gen = torch.Generator(device="cpu").manual_seed(M15.SEED + 100000 + layer_id)
        projection_size = ROUTER_RANK * width
        with torch.no_grad():
            self.a.copy_(torch.randn(self.a.shape, generator=factor_gen)
                         * (0.02 / math.sqrt(width)))
            self.router[:projection_size].copy_(
                torch.randn((projection_size,), generator=router_gen)
                * (0.1 / math.sqrt(width)))
            self.router[projection_size:].copy_(
                torch.randn(((axis_a + axis_b) * ROUTER_RANK,), generator=router_gen)
                * (0.1 / math.sqrt(ROUTER_RANK)))

    def routes(self, flat):
        width = M15.M13.D
        projection_size = ROUTER_RANK * width
        a_size = self.axis_a * ROUTER_RANK
        p = self.router[:projection_size].view(ROUTER_RANK, width)
        a_keys = self.router[projection_size:projection_size + a_size].view(self.axis_a, ROUTER_RANK)
        b_keys = self.router[projection_size + a_size:].view(self.axis_b, ROUTER_RANK)
        q = F.linear(flat.float(), p)
        a_scores = F.linear(q, a_keys)
        b_scores = F.linear(q, b_keys)
        a_values, a_ids = torch.topk(a_scores, M15.K, dim=-1)
        b_values, b_ids = torch.topk(b_scores, M15.K, dim=-1)
        candidate_scores = (a_values.unsqueeze(-1) + b_values.unsqueeze(-2)).flatten(1)
        candidate_ids = (a_ids.unsqueeze(-1) * self.axis_b + b_ids.unsqueeze(-2)).flatten(1)
        selected_scores, order = torch.topk(candidate_scores, M15.K, dim=-1)
        chosen = candidate_ids.gather(1, order)
        if self.oracle_checks:
            dense_scores = (a_scores.unsqueeze(-1) + b_scores.unsqueeze(-2)).flatten(1)
            exact = torch.topk(dense_scores, M15.K, dim=-1).indices
            if not torch.equal(chosen.sort(dim=1).values, exact.sort(dim=1).values):
                raise AssertionError("product-key route differs from exhaustive score")
        if self.collect_load:
            self.route_counts += torch.bincount(chosen.detach().flatten(), minlength=self.a.shape[0]).cpu()
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

    def changed_slots(self):
        return int((self.b.detach().float().flatten(1).norm(dim=1) > 1e-9).sum())
