from __future__ import annotations

import unittest

import numpy as np

from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_repair import metrics


class Q4KQ8KRepairRunnerTests(unittest.TestCase):
    def test_exact_metrics_pass(self) -> None:
        reference = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
        result = metrics(reference.copy(), reference)
        self.assertEqual(result["nrmse"], 0.0)
        self.assertEqual(result["normalized_max"], 0.0)
        self.assertTrue(result["pass"])

    def test_wrong_row_order_fails(self) -> None:
        reference = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = metrics(reference.T.copy().reshape(-1), reference.reshape(-1))
        self.assertFalse(result["pass"])

    def test_small_candidate_error_passes_frozen_gate(self) -> None:
        reference = np.linspace(-10.0, 10.0, 1024, dtype=np.float32)
        candidate = reference + np.float32(1e-6)
        self.assertTrue(metrics(candidate, reference)["pass"])


if __name__ == "__main__":
    unittest.main()
