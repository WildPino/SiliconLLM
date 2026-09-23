from __future__ import annotations

import hashlib
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_post_f16_block0_ffn_operator_cross_input as runner


class PostF16Block0FFNOperatorTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", runner.PROTOCOL.read_text(encoding="utf-8"))
        self.assertIn("--strat01-post-f16-block0-ffn-operator-cross-input", runner.ENGINE.read_text(encoding="utf-8"))
        self.assertTrue(all(runner.source_controls().values()))

    def test_identity_bridge_is_frozen(self) -> None:
        self.assertEqual(runner.INPUTS["reference_swiglu"][1], "de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef")
        self.assertEqual(runner.ARM_IDENTITIES["expression_q6_replay"]["ffn_out"], "7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684")
        self.assertEqual(runner.ARM_IDENTITIES["expression_q6_replay"]["l_out"], runner.post.CURRENT_START_SHA)

    def test_helper_count_arithmetic(self) -> None:
        self.assertEqual(runner.EXPECTED_COUNTS["qk_invocations"], 4 * 32 * sum(range(1, 9)))
        self.assertEqual(runner.EXPECTED_COUNTS["value_invocations"], 4 * 8 * 32 * 512)
        self.assertEqual(runner.EXPECTED_Q6_ARMS, 4)

    def test_status_partition(self) -> None:
        self.assertEqual(runner.classify(True, True), "POST_F16_BLOCK0_Q6_RESIDUAL_SUFFICIENT")
        self.assertEqual(runner.classify(False, True), "POST_F16_BLOCK0_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT")
        with self.assertRaises(runner.DiagnosticError):
            runner.classify(False, False)

    def test_expression_shape_and_direction(self) -> None:
        gate = np.array([0.0, 1.0, -1.0], dtype="<f4"); up = np.array([2.0, 2.0, 2.0], dtype="<f4")
        out = (gate / (np.float32(1.0) + np.exp(-gate))) * up
        self.assertEqual(float(out[0]), 0.0); self.assertGreater(float(out[1]), 0.0); self.assertLess(float(out[2]), 0.0)

    def test_mutation_refusal_primitive(self) -> None:
        data = b"post-f16-ffn-operator"; digest = hashlib.sha256(data).hexdigest(); mutated = bytearray(data); mutated[len(mutated) // 2] ^= 1
        self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
        self.assertNotEqual(hashlib.sha256(mutated).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
