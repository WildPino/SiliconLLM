from __future__ import annotations

import hashlib
import unittest

from benchmarks.donor_adaptation.engine import run_strat01_post_f16_block0_swiglu_sse2_semantics as runner


class PostF16Block0SwiGLUSSE2SemanticsTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", runner.PROTOCOL.read_text(encoding="utf-8"))
        self.assertIn("--strat01-post-f16-block0-swiglu-sse2-semantics", runner.ENGINE.read_text(encoding="utf-8"))
        self.assertTrue(all(runner.source_controls().values()))

    def test_reference_build_is_single_precise_coordinate(self) -> None:
        evidence = runner.validate_reference_evidence()
        self.assertEqual(evidence["compile_semantics"], {"ggml_cpu_generic": True, "avx": False, "fma": False, "fast_math": False, "lane_width": 4, "scalar_tail": False})
        self.assertEqual(runner.REFERENCE_REVISION, "5b335f413e4f73b0809c4fe39af894efbcc6a0d2")

    def test_predecessor_and_arm_identities_are_frozen(self) -> None:
        self.assertEqual(runner.PREDECESSOR_SHA, "ad03099c59a2bb1ee30bb0063c2b4b9f393b1551ec0e19931d0bf518dd1095eb")
        self.assertEqual(runner.SCALAR_IDENTITIES["swiglu"], "ac2b46108a7e18ddebdb1b3a26462bca0faebea536361a5ca4069c4fa7ddcf40")
        self.assertEqual(runner.PASS_IDENTITIES["swiglu"], runner.INPUTS["reference_swiglu"][1])

    def test_helper_count_arithmetic(self) -> None:
        self.assertEqual(runner.EXPECTED_COUNTS["qk_invocations"], 4 * 32 * sum(range(1, 9)))
        self.assertEqual(runner.EXPECTED_COUNTS["value_invocations"], 4 * 8 * 32 * 512)
        self.assertEqual(runner.EXPECTED_Q6_ARMS, 4)

    def test_status_partition(self) -> None:
        self.assertEqual(runner.classify(True, True), "POST_F16_BLOCK0_SWIGLU_SSE2_EXACT_REPAIR")
        self.assertEqual(runner.classify(False, True), "POST_F16_BLOCK0_SWIGLU_SSE2_NUMERIC_REPAIR")
        self.assertEqual(runner.classify(False, False), "POST_F16_BLOCK0_SWIGLU_SSE2_INSUFFICIENT")

    def test_no_fma_and_no_scalar_tail_contract(self) -> None:
        text = runner.SHARED_HEADER.read_text(encoding="utf-8")
        self.assertNotIn("_mm_fmadd", text)
        self.assertNotIn("_mm_fnmadd", text)
        self.assertIn("count & 3U", text)
        self.assertIn("i += 4U", text)

    def test_mutation_refusal_primitive(self) -> None:
        data = b"post-f16-swiglu-sse2-semantics"
        digest = hashlib.sha256(data).hexdigest()
        mutated = bytearray(data); mutated[len(mutated) // 2] ^= 1
        self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
        self.assertNotEqual(hashlib.sha256(mutated).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
