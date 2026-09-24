from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_q6k_q8k_avx2_scientific import (
    CRITICAL_PATHS,
    EXTRA_INPUTS,
    OUTPUT_BYTES,
    Q8_BYTES,
    classify,
)


class Q6KQ8KAVX2ScientificRunnerTests(unittest.TestCase):
    def test_frozen_dimensions_and_inputs(self) -> None:
        self.assertEqual(Q8_BYTES, 81760)
        self.assertEqual(OUTPUT_BYTES, 49152)
        self.assertEqual(set(EXTRA_INPUTS), {
            "topk", "ref_kqv", "ref_layer1_ffn", "ref_gate",
            "ref_q", "ref_k", "ref_shared", "ref_weights",
        })

    def test_decision_table(self) -> None:
        self.assertEqual(classify(False, False, False), "BLOCK0_Q6_Q8K_QUANTIZER_MISMATCH")
        self.assertEqual(classify(True, False, False), "BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT")
        self.assertEqual(classify(True, True, True), "BLOCK0_Q6_AVX2_EXACT_REPAIR")
        self.assertEqual(classify(True, True, False), "VOID_BLOCK0_Q6K_Q8K_AVX2_PARITY")

    def test_critical_sources_include_scientific_runner(self) -> None:
        self.assertIn("run_strat01_q6k_q8k_avx2_scientific.py", {path.name for path in CRITICAL_PATHS})


if __name__ == "__main__":
    unittest.main()
