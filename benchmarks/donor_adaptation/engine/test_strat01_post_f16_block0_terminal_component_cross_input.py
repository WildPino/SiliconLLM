from __future__ import annotations

import hashlib
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_post_f16_block0_terminal_component_cross_input as runner


class PostF16Block0TerminalComponentTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", runner.PROTOCOL.read_text(encoding="utf-8"))
        self.assertIn("--strat01-post-f16-block0-terminal-component-cross-input", runner.ENGINE.read_text(encoding="utf-8"))
        self.assertTrue(all(runner.source_controls().values()))

    def test_changed_coordinate_is_exact(self) -> None:
        self.assertEqual(runner.ARMS, ("current_attention_reference_ffn", "reference_attention_current_ffn"))
        self.assertEqual(runner.PREDECESSOR_SHA, "766061a9f54b601b8f54f08b6aaf6d533928aa19638f04ac2dd6cc23f865c09b")
        self.assertEqual(runner.post.CURRENT_START_SHA, "7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4")

    def test_helper_count_arithmetic(self) -> None:
        self.assertEqual(runner.EXPECTED_COUNTS["qk_invocations"], 5 * 32 * sum(range(1, 9)))
        self.assertEqual(runner.EXPECTED_COUNTS["value_invocations"], 5 * 8 * 32 * 512)

    def test_f32_homogeneous_reconstruction(self) -> None:
        a = np.array([1.0, 16_777_216.0], dtype="<f4"); b = np.array([2.0, 1.0], dtype="<f4"); result = (a + b).astype("<f4", copy=False)
        self.assertEqual(result.tolist(), [3.0, 16_777_216.0])

    def test_mutation_refusal_primitive(self) -> None:
        data = b"post-f16-block0-components"; digest = hashlib.sha256(data).hexdigest(); mutated = bytearray(data); mutated[len(mutated)//2] ^= 1
        self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
        self.assertNotEqual(hashlib.sha256(mutated).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
