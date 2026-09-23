from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_q4k_q8k_reference_generic_parity import (
    CRITICAL_PATHS,
    KV_ACTIVE_AVX2_SHA,
    Q_ACTIVE_AVX2_SHA,
    TEST_MODULES,
)
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as base


class Q4KQ8KReferenceGenericParityRunnerTests(unittest.TestCase):
    def test_three_full_projection_identities_are_distinct(self) -> None:
        self.assertEqual(len({base.Q_REFERENCE_SHA, base.Q_GENERIC_SHA, Q_ACTIVE_AVX2_SHA}), 3)
        self.assertEqual(len({base.KV_REFERENCE_SHA, base.KV_GENERIC_SHA, KV_ACTIVE_AVX2_SHA}), 3)

    def test_inventory_and_regressions_are_complete(self) -> None:
        names = {path.name for path in CRITICAL_PATHS}
        for name in (
            "strat01_q4k_q8k_reference_generic_probe.c",
            "strat01_q4k_q8k_reference_generic_oracle.cpp",
            "test_strat01_q4k_q8k_reference_generic_parity_runner.py",
            "strat01_gguf_rung2a.h",
        ):
            self.assertIn(name, names)
        joined = "\n".join(TEST_MODULES)
        self.assertIn("reference_generic_parity", joined)
        self.assertIn("engine_rung2c", joined)


if __name__ == "__main__":
    unittest.main()
