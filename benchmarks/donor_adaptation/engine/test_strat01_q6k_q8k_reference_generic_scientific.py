from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_q6k_q8k_reference_generic_scientific import (
    APPARATUS_SHA,
    CRITICAL_PATHS,
    classify,
)


class Q6KQ8KReferenceGenericScientificTests(unittest.TestCase):
    def test_apparatus_and_critical_sources_are_frozen(self) -> None:
        self.assertEqual(len(APPARATUS_SHA), 64)
        names = {path.name for path in CRITICAL_PATHS}
        self.assertIn("run_strat01_q6k_q8k_reference_generic_scientific.py", names)
        self.assertIn("strat01_q6k_q8k_reference_generic_oracle.cpp", names)

    def test_decision_table(self) -> None:
        void = "VOID_BLOCK0_Q6_REFERENCE_GENERIC_COMPILE_PARITY"
        mismatch = "BLOCK0_Q6_REFERENCE_GENERIC_IMPLEMENTATION_MISMATCH"
        insufficient = "BLOCK0_Q6_REFERENCE_GENERIC_COMPILE_INSUFFICIENT"
        exact = "BLOCK0_Q6_REFERENCE_GENERIC_EXACT_REPAIR"
        self.assertEqual(classify(False, True, True, True, True, True), void)
        self.assertEqual(classify(True, True, True, True, True, False), void)
        self.assertEqual(classify(True, False, False, False, False, True), mismatch)
        self.assertEqual(classify(True, True, False, False, False, True), insufficient)
        self.assertEqual(classify(True, True, True, False, False, True), insufficient)
        self.assertEqual(classify(True, True, True, True, True, True), exact)


if __name__ == "__main__":
    unittest.main()
