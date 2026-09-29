#!/usr/bin/env python3
"""Small exact-state and atomic-pointer smoke test for METH-175 snapshots."""

from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from meth136_sparse_experts import SelectedRowAdam
import meth175_exact_checkpoint as C175


def state():
    banks = [torch.arange(8, dtype=torch.float32).view(4, 2, 1).clone()
             for _ in range(24)]
    optimizers = [SelectedRowAdam(bank) for bank in banks]
    counts = [torch.zeros(4, dtype=torch.int64) for _ in range(24)]
    for optimizer, count in zip(optimizers, counts):
        optimizer.step(torch.tensor([1, 3]), torch.ones((2, 2, 1)))
        count[1] = 1
        count[3] = 1
    return banks, optimizers, counts


def main():
    identity = {"arm": "control", "draws_sha256": "smoke"}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "test.checkpoint"
        banks, optimizers, counts = state()
        wrappers = [SimpleNamespace(forward_B_transfer_bytes=0, gather_calls=0)
                    for _ in range(24)]
        original = [(bank.clone(), optimizer.m[:optimizer.used].clone(),
                     optimizer.v[:optimizer.used].clone(),
                     optimizer.steps[:optimizer.used].clone(),
                     optimizer.locations.clone(), count.clone())
                    for bank, optimizer, count in zip(banks, optimizers, counts)]
        C175.save(root, identity, 1, lambda: 2.0, lambda: 1.0,
                  wrappers, banks, optimizers, counts, None, None,
                  [{"update": 1}], 7, 1_000_000)
        banks2, optimizers2, counts2 = state()
        wrappers2 = [SimpleNamespace(forward_B_transfer_bytes=0, gather_calls=0)
                     for _ in range(24)]
        for bank, optimizer, count in zip(banks2, optimizers2, counts2):
            bank.zero_(); optimizer.locations.fill_(-1)
            optimizer.used = 0; optimizer.m.zero_(); optimizer.v.zero_()
            optimizer.steps.zero_(); count.zero_()
        restored = C175.restore(root, identity, wrappers2, banks2,
                                optimizers2, counts2, None, None)
        assert restored["completed_update"] == 1
        assert restored["gradient_transfer_total"] == 7
        for (expected, m, v, steps, locations, count), bank, optimizer, actual in zip(
                original, banks2, optimizers2, counts2):
            assert torch.equal(bank, expected)
            assert torch.equal(optimizer.m[:optimizer.used], m)
            assert torch.equal(optimizer.v[:optimizer.used], v)
            assert torch.equal(optimizer.steps[:optimizer.used], steps)
            assert torch.equal(optimizer.locations, locations)
            assert torch.equal(actual, count)
        C175.save(root, identity, 2, lambda: 4.0, lambda: 3.0,
                  wrappers2, banks2, optimizers2, counts2, None, None,
                  [{"update": 1}, {"update": 2}], 9, 1_000_000)
        latest, _ = C175.peek(root, identity)
        assert latest["completed_update"] == 2
        assert not (root / "u0001").exists()
        C175.cleanup(root)
        assert not root.exists()
    print("METH-175 exact checkpoint smoke pass")


if __name__ == "__main__":
    main()
