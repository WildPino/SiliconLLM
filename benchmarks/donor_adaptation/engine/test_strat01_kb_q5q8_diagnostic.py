from __future__ import annotations

import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_kb_q5q8_diagnostic as runner


class KBQ5Q8DiagnosticTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        self.assertIn("Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY", runner.PROTOCOL.read_text(encoding="utf-8"))
        self.assertIn("--strat01-kb-q5q8-diagnostic", runner.ENGINE.read_text(encoding="utf-8"))
        self.assertIn("ggml_vec_dot_q5_0_q8_0", runner.HELPER_SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(set(runner.source_inventory()), {"runner", "engine", "rung2a_header", "diagnostic_header", "helper_source", "helper_build", "tests", "protocol"})

    def test_frozen_payloads(self) -> None:
        arrays = runner.validate_payloads()
        self.assertEqual(arrays["reference_q"].size, 8 * 32 * 192)
        self.assertEqual(arrays["reference_target"].size, runner.OUTPUT_COUNT)

    def test_metrics_and_classification(self) -> None:
        target = np.array([1.0, -2.0, 4.0], dtype=np.float32)
        good = runner.metrics(target.copy(), target, runner.TIGHT)
        bad_values = target.copy(); bad_values[1] += 0.1
        bad = runner.metrics(bad_values, target, runner.TIGHT)
        self.assertTrue(good["pass"]); self.assertFalse(bad["pass"])
        self.assertEqual(runner.classify(good, good, {"a": True}), "Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY")
        self.assertEqual(runner.classify(good, bad, {"a": True}), "Q5_0_Q8_0_REFERENCE_ONLY")
        self.assertEqual(runner.classify(bad, bad, {"a": True}), "Q5_0_Q8_0_FAILS_REFERENCE_INPUT")
        self.assertEqual(runner.classify(good, good, {"a": False}), "VOID_KB_Q5_0_Q8_0_DIAGNOSTIC")

    def test_q8_census(self) -> None:
        a = bytes(runner.Q8_BYTES); b = bytearray(a); b[0] = 1; b[34] = 1
        census = runner.q8_census(a, bytes(b))
        self.assertEqual(census["changed_blocks"], 2)
        self.assertEqual(census["changed_bytes"], 2)


if __name__ == "__main__":
    unittest.main()
