"""Keep one combined CPU B bank exactly equal to base plus routed residual."""

import torch
import numpy as np

from meth136_sparse_experts import SelectedRowAdam
from meth151_shared_sparse_experts import SharedStructureSparseExperts
import meth15_zero_residual_expert_smoke as M15


class FactorizedSharedExperts(SharedStructureSparseExperts):
    def __init__(self, parent, layer_id, combined_bank, shared_a, base_bank,
                 residual_bank):
        super().__init__(parent, layer_id, combined_bank, shared_a)
        assert base_bank.shape == (1280, M15.M13.D, M15.R)
        assert base_bank.device.type == "cpu" and base_bank.dtype == torch.float32
        assert residual_bank.shape == combined_bank.shape == (12800, M15.M13.D, M15.R)
        assert residual_bank.device.type == "cpu" and residual_bank.dtype == torch.float32
        self.base_bank = base_bank
        self.residual_bank = residual_bank
        assert torch.equal(combined_bank,
                           base_bank.repeat_interleave(10, dim=0) + residual_bank)


class FactorizedSelectedRowAdam:
    def __init__(self, base_bank, residual_bank, combined_bank, layer_id):
        assert base_bank.shape == (1280, M15.M13.D, M15.R)
        assert residual_bank.shape == combined_bank.shape == (12800, M15.M13.D, M15.R)
        self.base_bank = base_bank
        self.residual_bank = residual_bank
        self.combined_bank = combined_bank
        self.base_optimizer = SelectedRowAdam(base_bank, learning_rate=1e-5)
        self.residual_optimizer = SelectedRowAdam(residual_bank, learning_rate=2e-6)
        self.prepared = None
        self.audit_rng = np.random.default_rng(155155 + layer_id)
        self.audit_checks = 0

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
        all_ids = (parents[:, None] * 10 + torch.arange(10)).reshape(-1)
        combined = (self.base_bank.index_select(0, parents).repeat_interleave(10, dim=0)
                    + self.residual_bank.index_select(0, all_ids))
        self.combined_bank.index_copy_(0, all_ids, combined)
        selected_expected = (self.base_bank.index_select(0, ids // 10)
                             + self.residual_bank.index_select(0, ids))
        assert torch.equal(self.combined_bank.index_select(0, ids), selected_expected)
        touched = set(parents.tolist())
        if len(touched) < 1280:
            untouched = int(self.audit_rng.integers(0, 12800))
            while untouched // 10 in touched:
                untouched = int(self.audit_rng.integers(0, 12800))
            assert torch.equal(self.combined_bank[untouched],
                self.base_bank[untouched // 10] + self.residual_bank[untouched])
            self.audit_checks += 1
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
            residual = wrapper.residual_bank
            combined = wrapper.cpu_bank
            assert base.shape == (1280, 896, 8)
            assert residual.shape == (rows, 896, 8)
            assert torch.equal(combined, base.repeat_interleave(10, dim=0) + residual)
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
        expected_bits = wrapper.cpu_bank.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        assert np.array_equal(readback, expected_bits.reshape(-1))
    return {"path": str(path.resolve()), "bytes": expected,
            "sha256": M136.digest(path), "readback_exact": True,
            "bf16_distinct_rows_by_layer": different_rows,
            "base_bf16_changed_rows_by_layer": base_changed_rows,
            "combined_mean_shift_rms_by_layer": mean_shift_rms}
