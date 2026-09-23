from __future__ import annotations

import hashlib
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_ffn_down_cross_input as runner


class FFNDownCrossInputTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        self.assertIn("Exactly one non-VOID invocation", runner.PROTOCOL.read_text(encoding="utf-8"))
        self.assertIn("--strat01-ffn-down-cross-input", runner.ENGINE.read_text(encoding="utf-8"))
        self.assertEqual(set(runner.source_inventory()), {"runner", "terminal_runner", "layer1_runner", "engine", "rung2a", "rung2b", "rung2c", "layer1_header", "terminal_header", "header", "protocol", "tests"})

    def test_frozen_contract(self) -> None:
        self.assertEqual(runner.ARMS, ["reference_captured_control", "reference_swiglu_current_q6", "c_swiglu_current_q6"])
        self.assertEqual(set(runner.SOURCES), set(runner.ARMS)); self.assertEqual(len(runner.INPUTS), 5)
        self.assertTrue(all(len(digest) == 64 and size in (49_152, 286_720) for _, digest, size in runner.INPUTS.values()))

    def test_direct_gate(self) -> None:
        ref = np.ones(8 * 1536, dtype=np.float32)
        self.assertTrue(runner.direct_judgment(ref.copy(), ref)["pass"])
        changed = ref.copy(); changed[0] = 2.0
        self.assertFalse(runner.direct_judgment(changed, ref)["pass"])

    def test_row_swap_is_live(self) -> None:
        values = np.arange(8 * 4, dtype=np.float32).reshape(8, 4); swapped = values.copy(); swapped[[0, 7]] = swapped[[7, 0]]
        self.assertFalse(np.array_equal(values, swapped)); np.testing.assert_array_equal(swapped[0], values[7]); np.testing.assert_array_equal(swapped[7], values[0])

    def test_mutation_refusal(self) -> None:
        data = b"ffn-down"; digest = hashlib.sha256(data).hexdigest(); mutated = bytearray(data); mutated[2] ^= 1
        self.assertTrue(runner.base.identity_matches(data, len(data), digest)); self.assertFalse(runner.base.identity_matches(bytes(mutated), len(data), digest))


if __name__ == "__main__": unittest.main()
