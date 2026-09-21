from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_kqv_out_diagnostic import classify


def judged(passed: bool, nrmse: float = 0.0, normalized_max: float = 0.0) -> dict[str, bool | float]:
    return {"pass": passed, "nrmse": nrmse, "normalized_max": normalized_max}


class KqvOutDiagnosticTests(unittest.TestCase):
    def test_attribution_requires_q8_target_and_d32_failure(self) -> None:
        controls = {"control": True}
        results = {
            "project_vs_pinned": judged(True),
            "project_vs_target": judged(True),
            "pinned_vs_target": judged(True),
            "d32_vs_target": judged(False, 0.01, 0.02),
        }
        self.assertEqual(classify(results, controls), "ATTRIBUTED_KQV_OUT_TO_VB_Q8K")
        results["d32_vs_target"] = judged(True, 0.01, 0.02)
        self.assertEqual(classify(results, controls), "PARTIAL_VB_Q8K_ATTRIBUTION")

    def test_invalid_control_is_void(self) -> None:
        results = {
            "project_vs_pinned": judged(True),
            "project_vs_target": judged(True),
            "pinned_vs_target": judged(True),
            "d32_vs_target": judged(False, 0.01, 0.02),
        }
        self.assertEqual(classify(results, {"control": False}), "VOID_KQV_OUT_DIAGNOSTIC")

    def test_pinned_target_failure_keeps_attention_open(self) -> None:
        results = {
            "project_vs_pinned": judged(True),
            "project_vs_target": judged(False, 0.02, 0.02),
            "pinned_vs_target": judged(False, 0.02, 0.02),
            "d32_vs_target": judged(False, 0.01, 0.01),
        }
        self.assertEqual(
            classify(results, {"control": True}),
            "ATTENTION_RECONSTRUCTION_OR_OTHER_VB_SEMANTICS_REMAIN",
        )


if __name__ == "__main__":
    unittest.main()
