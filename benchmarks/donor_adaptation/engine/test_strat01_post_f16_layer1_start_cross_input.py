from __future__ import annotations

import hashlib
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_post_f16_layer1_start_cross_input as runner


class PostF16Layer1StartTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        text = runner.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("before implementation or execution", text)
        self.assertIn("--strat01-post-f16-layer1-start-cross-input", runner.ENGINE.read_text(encoding="utf-8"))
        self.assertEqual(set(runner.source_inventory()), {"runner", "tests", "protocol", "recovery_protocol", "engine", "header", "rung2a", "rung2c", "f16_dot"})

    def test_frozen_contract(self) -> None:
        self.assertEqual(runner.ORDER, list(runner.r2c.SHAPES))
        self.assertEqual(len(runner.REFERENCE_START_SHA), 64)
        self.assertEqual(len(runner.CURRENT_START_SHA), 64)
        self.assertNotEqual(runner.REFERENCE_START_SHA, runner.CURRENT_START_SHA)
        self.assertEqual(runner.EXPECTED_COUNTS["qk_invocations"], 4 * 32 * sum(range(1, 9)))
        self.assertEqual(runner.EXPECTED_COUNTS["value_invocations"], 4 * 8 * 32 * 512)

    def test_recovery_is_exactly_bound(self) -> None:
        self.assertEqual(runner.DEFAULT_OUTPUT.name, "strat01_gigachat_engine_post_f16_layer1_start_cross_input_20260923")
        self.assertEqual(runner.RAW_HEAD, "8f03fa3963723fc2083cc2d608437d1a487c9666")
        for digest in (runner.RAW_VOID_SHA, runner.RAW_REPORT_SHA, runner.RAW_COUNTS_SHA, runner.RAW_BINARY_SHA):
            self.assertEqual(len(digest), 64)

    def test_gate_and_terminal_limits(self) -> None:
        ref = np.ones(1536, dtype=np.float32)
        self.assertTrue(runner.judged(ref.copy(), ref, "l_out-1")["pass"])
        changed = ref.copy(); changed[0] = 2.0
        self.assertFalse(runner.judged(changed, ref, "l_out-1")["pass"])

    def test_exact_i32(self) -> None:
        ref = np.arange(32, dtype=np.int32)
        self.assertTrue(runner.judged(ref.copy(), ref, "ffn_moe_topk-1")["pass"])
        changed = ref.copy(); changed[0] += 1
        self.assertFalse(runner.judged(changed, ref, "ffn_moe_topk-1")["pass"])

    def test_mutation_refusal_primitive(self) -> None:
        data = b"post-f16-layer1"; digest = hashlib.sha256(data).hexdigest(); mutated = bytearray(data); mutated[0] ^= 1
        self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
        self.assertNotEqual(hashlib.sha256(mutated).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
