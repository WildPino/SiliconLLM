from __future__ import annotations

import hashlib
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_rung2c_layer1_start_cross_input as runner


class Layer1StartTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        self.assertIn("Exactly one non-VOID invocation", runner.PROTOCOL.read_text(encoding="utf-8"))
        self.assertIn("--strat01-rung2c-layer1-start-cross-input", runner.ENGINE.read_text(encoding="utf-8"))
        self.assertEqual(set(runner.source_inventory()), {"runner", "base_runner", "engine", "rung2a", "rung2c", "header", "protocol", "tests"})

    def test_frozen_checkpoint_contract(self) -> None:
        self.assertEqual(list(runner.SIZES), runner.ORDER)
        self.assertEqual(set(runner.REF_HASHES), set(runner.ORDER)); self.assertEqual(set(runner.C_HASHES), set(runner.ORDER)); self.assertEqual(set(runner.FROZEN_METRICS), set(runner.ORDER))
        self.assertTrue(all(len(value) == 64 for value in (*runner.REF_HASHES.values(), *runner.C_HASHES.values())))

    def test_gate(self) -> None:
        ref = np.ones(runner.SIZES["k_pe-1"] // 4, dtype=np.float32)
        self.assertTrue(runner.judged_checkpoint(ref.copy(), ref, "k_pe-1")["pass"])
        changed = ref.copy(); changed[0] = 2
        self.assertFalse(runner.judged_checkpoint(changed, ref, "k_pe-1")["pass"])

    def test_mutation_refusal(self) -> None:
        data = b"layer1-start"; digest = hashlib.sha256(data).hexdigest(); mutated = bytearray(data); mutated[0] ^= 1
        self.assertTrue(runner.base.identity_matches(data, len(data), digest)); self.assertFalse(runner.base.identity_matches(bytes(mutated), len(data), digest))


if __name__ == "__main__": unittest.main()
