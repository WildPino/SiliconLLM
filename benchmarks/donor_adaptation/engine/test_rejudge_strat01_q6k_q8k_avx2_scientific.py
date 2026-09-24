from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.rejudge_strat01_q6k_q8k_avx2_scientific import (
    SOURCE_ADJUDICATION_SHA,
)
from benchmarks.donor_adaptation.engine.run_strat01_q6k_q8k_avx2_scientific import classify


class Q6KQ8KAVX2RejudgeTests(unittest.TestCase):
    def test_void_branch_is_repaired_without_weakening_exact_repair(self) -> None:
        self.assertEqual(classify(True, True, False, False), "BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT")
        self.assertEqual(classify(True, True, True, True), "BLOCK0_Q6_AVX2_EXACT_REPAIR")
        self.assertEqual(len(SOURCE_ADJUDICATION_SHA), 64)


if __name__ == "__main__":
    unittest.main()
