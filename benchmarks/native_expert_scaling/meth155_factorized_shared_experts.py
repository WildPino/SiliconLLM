"""Train shared CPU B base plus routed CPU B residual, then export one B row."""

import torch
import torch.nn.functional as F
import numpy as np

from meth134_sparse_e12800_training import SparseCpuGather
from meth136_sparse_experts import SelectedRowAdam
from meth151_shared_sparse_experts import SharedStructureSparseExperts
import meth15_zero_residual_expert_smoke as M15


class FactorizedSharedExperts(SharedStructureSparseExperts):
    def __init__(self, parent, layer_id, residual_bank, shared_a, base_bank):
        super().__init__(parent, layer_id, residual_bank, shared_a)
        assert base_bank.shape == (1280, M15.M13.D, M15.R)
        assert base_bank.device.type == "cpu" and base_bank.dtype == torch.float32
        self.base_bank = base_bank

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
        base_ids = ids.detach().div(10, rounding_mode="floor").to(
            device="cpu", dtype=torch.long)
        base = self.base_bank.index_select(0, base_ids.reshape(-1)).to(
            flat.device).reshape(*ids.shape, M15.M13.D, M15.R)
        if self.collector is not None and torch.is_grad_enabled():
            trigger = torch.zeros((), dtype=torch.float32,
                                  device=flat.device, requires_grad=True)
            residual = SparseCpuGather.apply(trigger, ids, self.cpu_bank,
                                             self.collector)
        else:
            residual_ids = ids.detach().to(device="cpu", dtype=torch.long)
            residual = self.cpu_bank.index_select(0, residual_ids.reshape(-1)).to(
                flat.device).reshape(*ids.shape, M15.M13.D, M15.R)
        self.forward_B_transfer_bytes += base.numel() * base.element_size()
        self.forward_B_transfer_bytes += residual.numel() * residual.element_size()
        self.gather_calls += 2
        b = (base + residual).to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b)
        residual_out = (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)
        return dense + residual_out


class FactorizedSelectedRowAdam:
    def __init__(self, base_bank, residual_bank):
        assert base_bank.shape == (1280, M15.M13.D, M15.R)
        assert residual_bank.shape == (12800, M15.M13.D, M15.R)
        self.base_bank = base_bank
        self.residual_bank = residual_bank
        self.base_optimizer = SelectedRowAdam(base_bank, learning_rate=1e-5)
        self.residual_optimizer = SelectedRowAdam(residual_bank, learning_rate=2e-6)
        self.prepared = None

    @property
    def used(self):
        return self.residual_optimizer.used

    @property
    def base_used(self):
        return self.base_optimizer.used

    @property
    def moment_allocated_bytes(self):
        return sum((opt.m.numel() + opt.v.numel()) * 4 for opt in
                   (self.base_optimizer, self.residual_optimizer))

    def prepare(self, ids, gradients):
        assert self.prepared is None
        parents, inverse = torch.unique(ids // 10, sorted=True, return_inverse=True)
        base_grad = torch.zeros((parents.numel(), *gradients.shape[1:]),
                                dtype=gradients.dtype)
        base_grad.index_add_(0, inverse, gradients)
        self.prepared = (ids, parents, base_grad)
        squared = (torch.sum(gradients.double().square()) +
                   torch.sum(base_grad.double().square()))
        return float(squared)

    def step(self, ids, gradients, scale):
        assert self.prepared is not None and torch.equal(ids, self.prepared[0])
        parents, base_grad = self.prepared[1:]
        base_row = self.base_optimizer.step(parents, base_grad, scale)
        residual_row = self.residual_optimizer.step(ids, gradients, scale)
        self.prepared = None
        return {"new_rows": residual_row["new_rows"],
                "optimizer_rows": self.used,
                "base_new_rows": base_row["new_rows"],
                "base_optimizer_rows": self.base_used}


def export_combined_bank(path, wrappers, source_b):
    import meth136_matched_sparse_train as M136

    assert len(wrappers) == len(source_b) == 24
    rows = 12800
    different_rows = []
    mean_shift_rms = []
    base_changed_rows = []
    with path.open("xb") as stream:
        stream.write(M136.OUT_HEADER.pack(b"M136BF01", 24, 896, 8, rows))
        for wrapper, original in zip(wrappers, source_b):
            base = wrapper.base_bank
            residual = wrapper.cpu_bank
            assert base.shape == (1280, 896, 8)
            assert residual.shape == (rows, 896, 8)
            combined = base.repeat_interleave(10, dim=0) + residual
            different = (combined.to(torch.bfloat16) !=
                         original.to(torch.bfloat16).repeat_interleave(10, dim=0))
            different_rows.append(int(different.reshape(rows, -1).any(dim=1).sum()))
            mean_shift = combined.view(1280, 10, 896, 8).mean(dim=1) - original
            mean_shift_rms.append(float(torch.sqrt(torch.mean(mean_shift.double().square()))))
            base_changed_rows.append(int((base.to(torch.bfloat16) !=
                original.to(torch.bfloat16)).reshape(1280, -1).any(dim=1).sum()))
            bits = combined.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
            stream.write(bits.astype("<u2", copy=False).tobytes())
    expected = M136.OUT_HEADER.size + 24 * rows * 896 * 8 * 2
    assert path.stat().st_size == expected
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert M136.OUT_HEADER.unpack_from(data) == (b"M136BF01", 24, 896, 8, rows)
    stride = rows * 896 * 8 * 2
    for li, wrapper in enumerate(wrappers):
        readback = np.frombuffer(data, dtype="<u2", count=rows * 896 * 8,
                                 offset=M136.OUT_HEADER.size + li * stride)
        expected_bits = (wrapper.base_bank.repeat_interleave(10, dim=0) +
                         wrapper.cpu_bank).to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        assert np.array_equal(readback, expected_bits.reshape(-1))
    return {"path": str(path.resolve()), "bytes": expected,
            "sha256": M136.digest(path), "readback_exact": True,
            "bf16_distinct_rows_by_layer": different_rows,
            "base_bf16_changed_rows_by_layer": base_changed_rows,
            "combined_mean_shift_rms_by_layer": mean_shift_rms}
