from __future__ import annotations

import hashlib
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_upstream_rmsnorm_diag as runner


class UpstreamRMSNormDiagnosticTests(unittest.TestCase):
    def test_protocol_and_source_contract(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        header = runner.HEADER.read_text(encoding="utf-8")
        engine = runner.ENGINE.read_text(encoding="utf-8")
        self.assertIn("UPSTREAM_DOUBLE_RMSNORM_SUFFICIENT_FOR_PROJECTION_GATES", protocol)
        self.assertIn(runner.BASELINE_ATTN_SHA, header)
        self.assertIn(runner.BASELINE_FFN_SHA, header)
        self.assertIn("--strat01-upstream-rmsnorm-diagnostic", engine)

    def test_source_inventory_and_bindings(self) -> None:
        runner.validate_prior_bindings()
        inventory = runner.source_inventory()
        self.assertEqual(set(inventory), {"runner", "engine", "rung2a_header", "rung2b_header", "diagnostic_header", "protocol"})
        self.assertTrue(all(len(item["sha256"]) == 64 for item in inventory.values()))

    def test_gate_and_mutation_controls(self) -> None:
        reference = np.array([1.0, -2.0, 4.0], dtype=np.float32)
        self.assertTrue(runner.r2a.judged_metrics(reference.copy(), reference, runner.r2a.GENERAL_LIMITS)["pass"])
        candidate = reference.copy(); candidate[1] += 0.1
        self.assertFalse(runner.r2a.judged_metrics(candidate, reference, runner.r2a.GENERAL_LIMITS)["pass"])
        mutated = bytearray(runner.BASELINE_ATTN.read_bytes()); mutated[len(mutated)//2] ^= 1
        self.assertNotEqual(hashlib.sha256(mutated).hexdigest(), runner.BASELINE_ATTN_SHA)

    def test_output_contract(self) -> None:
        self.assertEqual(runner.FFN_SHAPES["ffn_norm-0"], [1536, 8])
        self.assertEqual(runner.FFN_SHAPES["ffn_up-0"], [8960, 8])
        self.assertEqual(runner.Q8_BLOCK_BYTES * 48, 14016)


if __name__ == "__main__":
    unittest.main()
