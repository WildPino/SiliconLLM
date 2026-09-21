from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_attention_stage_diagnostic import classify


def judged(passed: bool) -> dict[str, bool]:
    return {"pass": passed}


def results(raw: bool, softmax: bool, reduction: bool) -> dict[str, dict[str, bool]]:
    return {
        "raw_qk": judged(raw),
        "captured_qk_softmax": judged(softmax),
        "captured_softmax_value_reduction": judged(reduction),
    }


class AttentionStageDiagnosticTests(unittest.TestCase):
    def test_unique_stage_labels(self) -> None:
        controls = {"all": True}
        self.assertEqual(classify(results(True, True, True), controls), "ATTENTION_STAGE_DIAGNOSTIC_PASS")
        self.assertEqual(classify(results(False, True, True), controls), "ATTRIBUTED_ATTENTION_RESIDUAL_TO_QK_DOT")
        self.assertEqual(classify(results(True, False, True), controls), "ATTRIBUTED_ATTENTION_RESIDUAL_TO_SOFTMAX")
        self.assertEqual(classify(results(True, True, False), controls), "ATTRIBUTED_ATTENTION_RESIDUAL_TO_VALUE_REDUCTION")

    def test_mixed_and_void(self) -> None:
        self.assertEqual(classify(results(False, False, True), {"all": True}), "MIXED_ATTENTION_STAGE_RESIDUAL")
        self.assertEqual(classify(results(True, True, True), {"all": False}), "VOID_ATTENTION_STAGE_DIAGNOSTIC")


if __name__ == "__main__":
    unittest.main()
