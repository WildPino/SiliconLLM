from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_rung2b_rmsnorm_diag as runner


class RMSNormDiagnosticTests(unittest.TestCase):
    def test_protocol_and_source_contract(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        source = runner.RMS_HEADER.read_text(encoding="utf-8")
        engine = runner.ENGINE.read_text(encoding="utf-8")
        self.assertIn("DOUBLE_RMSNORM_SUFFICIENT_FOR_PROJECTION_GATES", protocol)
        self.assertIn("ggml_float", protocol)
        self.assertIn(runner.PINNED["reference_input"][1], source)
        self.assertIn(runner.PINNED["c_input"][1], source)
        self.assertIn("--strat01-rung2b-rmsnorm-diagnostic", engine)

    def test_gate(self) -> None:
        reference = np.array([1.0, -2.0, 4.0], dtype=np.float32)
        self.assertTrue(runner.base.judged(reference.copy(), reference)["pass"])
        candidate = reference.copy(); candidate[1] += 0.1
        self.assertFalse(runner.base.judged(candidate, reference)["pass"])

    def test_inventory(self) -> None:
        inventory = runner.source_inventory()
        self.assertEqual(set(inventory), {"runner", "engine", "rung2a", "rung2b", "cross_header", "rms_header", "protocol"})
        for item in inventory.values():
            self.assertEqual(len(item["sha256"]), 64)
            self.assertTrue(Path(item["path"]).is_file())

    def test_output_contract(self) -> None:
        self.assertEqual(len(runner.OUTPUT_SIZES), 8)
        self.assertEqual(runner.OUTPUT_SIZES["c_double_up.f32le"], runner.base.OUTPUT_BYTES)
        self.assertEqual(runner.OUTPUT_SIZES["c_double_norm.f32le"], runner.base.INPUT_BYTES)


if __name__ == "__main__":
    unittest.main()
