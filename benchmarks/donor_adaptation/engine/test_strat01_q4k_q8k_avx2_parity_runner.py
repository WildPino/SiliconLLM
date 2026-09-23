from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_avx2_parity import (
    CRITICAL_PATHS,
    KV_GENERIC_SHA,
    KV_REFERENCE_SHA,
    Q_GENERIC_SHA,
    Q_REFERENCE_SHA,
    SELFTESTS,
    TEST_MODULES,
)


class Q4KQ8KAVX2ParityRunnerTests(unittest.TestCase):
    def test_exact_hash_contract_separates_active_and_generic(self) -> None:
        self.assertNotEqual(Q_REFERENCE_SHA, Q_GENERIC_SHA)
        self.assertNotEqual(KV_REFERENCE_SHA, KV_GENERIC_SHA)

    def test_required_regression_families_are_registered(self) -> None:
        joined = "\n".join(TEST_MODULES)
        for family in (
            "q4k_q8k_operator", "engine_rung1", "engine_rung2a",
            "engine_rung2b", "engine_rung2c", "ffn_down", "ffn_swiglu",
        ):
            self.assertIn(family, joined)
        self.assertIn("--kselftest", SELFTESTS)
        self.assertIn("--strat01-gguf-rung2c-selftest", SELFTESTS)

    def test_critical_inventory_includes_projection_and_its_own_test(self) -> None:
        names = {path.name for path in CRITICAL_PATHS}
        self.assertIn("strat01_gguf_rung2a.h", names)
        self.assertIn("test_strat01_q4k_q8k_avx2_parity_runner.py", names)


if __name__ == "__main__":
    unittest.main()
