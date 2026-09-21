from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_rung2b_cross_input as runner


class CrossInputTests(unittest.TestCase):
    def test_frozen_protocol_and_source_contract(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        source = runner.CROSS_HEADER.read_text(encoding="utf-8")
        engine = runner.ENGINE.read_text(encoding="utf-8")
        self.assertIn("Exactly one non-VOID execution", protocol)
        self.assertIn("donor_graph_executions=0", protocol)
        self.assertIn(runner.PINNED["reference_input"][1], source)
        self.assertIn(runner.PINNED["c_input"][1], source)
        self.assertIn("--strat01-rung2b-cross-input", engine)

    def test_metrics_and_gate(self) -> None:
        reference = np.array([1.0, -2.0, 4.0], dtype=np.float32)
        self.assertTrue(runner.judged(reference.copy(), reference)["pass"])
        changed = reference.copy(); changed[0] += 0.1
        self.assertFalse(runner.judged(changed, reference)["pass"])

    def test_mutated_identity_is_refused(self) -> None:
        data = b"frozen diagnostic payload"
        digest = hashlib.sha256(data).hexdigest()
        self.assertTrue(runner.identity_matches(data, len(data), digest))
        mutated = bytearray(data); mutated[3] ^= 1
        self.assertFalse(runner.identity_matches(bytes(mutated), len(data), digest))

    def test_source_inventory_is_complete(self) -> None:
        inventory = runner.source_inventory()
        self.assertEqual(set(inventory), {"runner", "engine", "rung2a", "rung2b", "cross_header", "protocol"})
        for item in inventory.values():
            self.assertEqual(len(item["sha256"]), 64)
            self.assertTrue(Path(item["path"]).is_file())


if __name__ == "__main__":
    unittest.main()

