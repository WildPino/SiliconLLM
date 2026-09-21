from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_rung2a_q4_repair_confirmation import classify, classify_attention_vb


def result(passed: bool) -> dict[str, bool]:
    return {"pass": passed}


class Rung2AQ4RepairConfirmationTests(unittest.TestCase):
    def test_pass_requires_all_primary_and_continuity(self) -> None:
        negatives = [{"must_fail_old_gate": True}] * 4
        self.assertEqual(
            classify([result(True)] * 6, result(True), negatives),
            "PASS_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION",
        )
        self.assertEqual(
            classify([result(True)] * 5 + [result(False)], result(True), negatives),
            "FAIL_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION",
        )
        self.assertEqual(
            classify([result(True)] * 6, result(False), negatives),
            "FAIL_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION",
        )

    def test_missing_old_negative_is_void(self) -> None:
        negatives = [{"must_fail_old_gate": True}] * 3 + [{"must_fail_old_gate": False}]
        self.assertEqual(
            classify([result(True)] * 6, result(True), negatives),
            "VOID_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION",
        )

    def test_attention_vb_requires_every_full_gate_and_continuity(self) -> None:
        full = {"tensor_results": [result(True)] * 24, "cache_results": [result(True)] * 3}
        self.assertEqual(classify_attention_vb(full, result(True)), "PASS_ENGINE_ATTENTION_VB_REPAIR")
        full["tensor_results"][-1] = result(False)
        self.assertEqual(classify_attention_vb(full, result(True)), "FAIL_ENGINE_ATTENTION_VB_REPAIR")
        full["tensor_results"][-1] = result(True)
        self.assertEqual(classify_attention_vb(full, result(False)), "FAIL_ENGINE_ATTENTION_VB_REPAIR")


if __name__ == "__main__":
    unittest.main()
