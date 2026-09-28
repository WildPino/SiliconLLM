"""CPU-master B factors and selected-row Adam for the E1280/E12800 pair."""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


S1 = Path(__file__).resolve().parents[1] / "donor_adaptation/s1"
sys.path.insert(0, str(S1))
import meth15_zero_residual_expert_smoke as M15
import meth95_hierarchical_e1280_parity as M95
from meth134_sparse_e12800_training import SparseCollector, SparseCpuGather


class CpuSparseExperts(M95.HierarchicalExperts):
    """Freeze the old route/A; differentiate only selected CPU B rows."""

    def __init__(self, parent, layer_id, cpu_bank, shared_a,
                 grand_projection=None, grand_keys=None):
        super().__init__(parent, layer_id)
        assert cpu_bank.device.type == "cpu" and cpu_bank.dtype == torch.float32
        assert cpu_bank.shape[1:] == (M15.M13.D, M15.R)
        assert cpu_bank.shape[0] in (1280, 12800)
        assert shared_a.shape == (M15.E, M15.R, M15.M13.D)
        del self.a
        del self.b
        self.register_buffer("shared_a", shared_a.clone())
        self.cpu_bank = cpu_bank
        self.grandchildren = cpu_bank.shape[0] == 12800
        if self.grandchildren:
            assert grand_projection is not None and grand_keys is not None
            assert grand_projection.shape == (32, M15.M13.D)
            assert grand_keys.shape == (1280, 10, 32)
            self.register_buffer("grand_projection", grand_projection.clone())
            self.register_buffer("grand_keys", grand_keys.clone())
        else:
            assert grand_projection is None and grand_keys is None
        self.collector = None
        self.last_selected = None
        self.forward_B_transfer_bytes = 0
        self.gather_calls = 0

    def selected_ids(self, flat, children):
        if not self.grandchildren:
            return children
        q = F.linear(flat.float(), self.grand_projection)
        keys = self.grand_keys[children]
        scores = torch.einsum("nr,nkcr->nkc", q, keys)
        return children * 10 + scores.argmax(dim=-1)

    def forward(self, x):
        dense = self.base(x)
        if not self.enabled:
            return dense
        flat = x.reshape(-1, M15.M13.D)
        parents, selected_scores = self.routes(flat)
        children = self.child_route(flat, parents)
        ids = self.selected_ids(flat, children)
        self.last_selected = ids.detach()
        gate = F.softmax(selected_scores, dim=-1).to(flat.dtype)
        a = self.shared_a[parents].to(flat.dtype)
        if self.collector is not None and torch.is_grad_enabled():
            trigger = torch.zeros((), dtype=torch.float32,
                                  device=flat.device, requires_grad=True)
            b = SparseCpuGather.apply(trigger, ids, self.cpu_bank,
                                      self.collector)
        else:
            cpu_ids = ids.detach().to(device="cpu", dtype=torch.long)
            b = self.cpu_bank.index_select(0, cpu_ids.reshape(-1)).to(
                flat.device).reshape(*ids.shape, M15.M13.D, M15.R)
        self.forward_B_transfer_bytes += b.numel() * b.element_size()
        self.gather_calls += 1
        b = b.to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b)
        residual = (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)
        return dense + residual


class SelectedRowAdam:
    """AdamW moments allocated only for visited rows of a CPU B bank."""

    def __init__(self, bank, learning_rate=1e-5, beta1=0.9,
                 beta2=0.999, epsilon=1e-8):
        assert bank.device.type == "cpu" and bank.dtype == torch.float32
        self.bank = bank
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.locations = torch.full((bank.shape[0],), -1, dtype=torch.int32)
        self.capacity = min(256, bank.shape[0])
        self.used = 0
        shape = (self.capacity, *bank.shape[1:])
        self.m = torch.zeros(shape, dtype=torch.float32)
        self.v = torch.zeros(shape, dtype=torch.float32)
        self.steps = torch.zeros((self.capacity,), dtype=torch.int32)

    def grow(self, required):
        if required <= self.capacity:
            return
        capacity = min(self.bank.shape[0], max(required, self.capacity * 2))
        shape = (capacity, *self.bank.shape[1:])
        m = torch.zeros(shape, dtype=torch.float32)
        v = torch.zeros(shape, dtype=torch.float32)
        steps = torch.zeros((capacity,), dtype=torch.int32)
        m[:self.used].copy_(self.m[:self.used])
        v[:self.used].copy_(self.v[:self.used])
        steps[:self.used].copy_(self.steps[:self.used])
        self.m, self.v, self.steps = m, v, steps
        self.capacity = capacity

    def step(self, ids, gradients, scale=1.0):
        assert ids.device.type == "cpu" and ids.dtype == torch.long
        assert gradients.device.type == "cpu" and gradients.dtype == torch.float32
        assert gradients.shape == (ids.numel(), *self.bank.shape[1:])
        assert ids.numel() == torch.unique(ids).numel()
        slots = self.locations.index_select(0, ids).long()
        new = slots < 0
        count = int(new.sum())
        self.grow(self.used + count)
        if count:
            new_ids = ids[new]
            new_slots = torch.arange(self.used, self.used + count, dtype=torch.long)
            self.locations.index_copy_(0, new_ids, new_slots.int())
            self.used += count
            slots[new] = new_slots
        g = gradients * scale
        m = self.m.index_select(0, slots)
        v = self.v.index_select(0, slots)
        m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
        v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
        steps = self.steps.index_select(0, slots) + 1
        bf1 = 1.0 - torch.pow(torch.tensor(self.beta1), steps.float())
        bf2 = 1.0 - torch.pow(torch.tensor(self.beta2), steps.float())
        update = (m / bf1[:, None, None]) / (
            torch.sqrt(v / bf2[:, None, None]) + self.epsilon)
        if not torch.isfinite(update).all():
            raise FloatingPointError("nonfinite selected-row Adam update")
        with torch.no_grad():
            self.bank.index_add_(0, ids, -self.lr * update)
            self.m.index_copy_(0, slots, m)
            self.v.index_copy_(0, slots, v)
            self.steps.index_copy_(0, slots, steps)
        return {"new_rows": count, "optimizer_rows": self.used,
                "moment_bytes_allocated": (self.m.numel() + self.v.numel()) * 4,
                "moment_bytes_used": self.used * self.bank.shape[1] * self.bank.shape[2] * 8}
