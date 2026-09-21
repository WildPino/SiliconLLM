from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_kqv_out_diagnostic import classify


def judged(passed: bool) -> dict[str, bool]:
    return {"pass": passed}


class KqvOutDiagnosticTests(unittest.TestCase):
    def test_attribution_requires_q8_target_and_d32_failure(self) -> None:
        controls = {"control": True}
        results = {
            "project_vs_pinned": judged(True),
            "project_vs_target": judged(True),
            "pinned_vs_target": judged(True),
            "d32_vs_target": judged(False),
        }
        self.assertEqual(classify(results, controls), "ATTRIBUTED_KQV_OUT_TO_VB_Q8K")
        results["d32_vs_target"] = judged(True)
        self.assertEqual(classify(results, controls), "PARTIAL_VB_Q8K_ATTRIBUTION")

    def test_invalid_control_is_void(self) -> None:
        results = {
            "project_vs_pinned": judged(True),
            "project_vs_target": judged(True),
            "pinned_vs_target": judged(True),
            "d32_vs_target": judged(False),
        }
        self.assertEqual(classify(results, {"control": False}), "VOID_KQV_OUT_DIAGNOSTIC")

    def test_pinned_target_failure_keeps_attention_open(self) -> None:
        results = {
            "project_vs_pinned": judged(True),
            "project_vs_target": judged(False),
            "pinned_vs_target": judged(False),
            "d32_vs_target": judged(False),
        }
        self.assertEqual(
            classify(results, {"control": True}),
            "ATTENTION_RECONSTRUCTION_OR_OTHER_VB_SEMANTICS_REMAIN",
        )


if __name__ == "__main__":
    unittest.main()
