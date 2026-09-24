from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_q6k_q8k_reference_generic_parity import (
    CRITICAL_PATHS,
    PINNED_GENERIC_QUANTS_SHA,
    TEST_MODULE,
)


class Q6KQ8KReferenceGenericParityRunnerTests(unittest.TestCase):
    def test_runner_is_bound_to_pinned_generic_source(self) -> None:
        self.assertEqual(len(PINNED_GENERIC_QUANTS_SHA), 64)
        self.assertIn("q6k_q8k_reference_generic", TEST_MODULE)

    def test_critical_inventory_is_complete_and_model_free(self) -> None:
        names = {path.name for path in CRITICAL_PATHS}
        for name in (
            "engine.c", "strat01_q6k_q8k_avx2.h",
            "strat01_gguf_block0_q6k_q8k_avx2_parity.h",
            "strat01_q6k_q8k_reference_generic_probe.c",
            "strat01_q6k_q8k_reference_generic_oracle.cpp",
            "build_strat01_q6k_q8k_reference_generic_oracle.py",
            "test_strat01_q6k_q8k_reference_generic_parity.py",
        ):
            self.assertIn(name, names)
        self.assertFalse(any(path.suffix in {".gguf", ".f32le"} for path in CRITICAL_PATHS))


if __name__ == "__main__":
    unittest.main()
