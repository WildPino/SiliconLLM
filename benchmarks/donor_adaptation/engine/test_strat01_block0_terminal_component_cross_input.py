from __future__ import annotations

import hashlib
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_block0_terminal_component_cross_input as runner


class TerminalComponentTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        self.assertIn("Exactly one non-VOID invocation", runner.PROTOCOL.read_text(encoding="utf-8"))
        self.assertIn("--strat01-block0-terminal-component-cross-input", runner.ENGINE.read_text(encoding="utf-8"))
        self.assertEqual(set(runner.source_inventory()), {"runner", "layer1_runner", "engine", "rung2a", "rung2c", "layer1_header", "header", "protocol", "tests"})

    def test_arm_and_component_contract(self) -> None:
        self.assertEqual(runner.ARMS, ["reference_reference", "c_c", "c_attention_reference_ffn", "reference_attention_c_ffn"])
        self.assertEqual(set(runner.ORIGINS), set(runner.ARMS))
        self.assertEqual(len(runner.COMPONENTS), 6)
        self.assertTrue(all(len(digest) == 64 for _, digest in runner.COMPONENTS.values()))

    def test_float32_composition_semantics(self) -> None:
        a = np.array([1.0, -2.0, 0.25, 16_777_216.0], dtype="<f4")
        b = np.array([2.0, 0.5, 0.75, 1.0], dtype="<f4")
        result = (a + b).astype("<f4", copy=False)
        np.testing.assert_array_equal(result, np.array([3.0, -1.5, 1.0, 16_777_216.0], dtype="<f4"))

    def test_gate_is_shape_aware(self) -> None:
        ref = np.ones(runner.SIZES["k_pe-1"] // 4, dtype=np.float32)
        self.assertTrue(runner.l1.judged_checkpoint(ref.copy(), ref, "k_pe-1")["pass"])
        changed = ref.copy(); changed[0] = 2.0
        self.assertFalse(runner.l1.judged_checkpoint(changed, ref, "k_pe-1")["pass"])

    def test_mutation_refusal(self) -> None:
        data = b"terminal-component"; digest = hashlib.sha256(data).hexdigest(); mutated = bytearray(data); mutated[-1] ^= 1
        self.assertTrue(runner.base.identity_matches(data, len(data), digest))
        self.assertFalse(runner.base.identity_matches(bytes(mutated), len(data), digest))


if __name__ == "__main__":
    unittest.main()
