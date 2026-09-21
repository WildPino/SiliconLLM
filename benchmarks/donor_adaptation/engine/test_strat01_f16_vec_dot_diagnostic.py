from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_f16_vec_dot_diagnostic import classify


def judged(passed: bool, nrmse: float = 0.0) -> dict[str, bool | float]: return {"pass": passed, "nrmse": nrmse}


class F16VecDotDiagnosticTests(unittest.TestCase):
    def test_primary_labels(self) -> None:
        controls = {"all": True}; old = {"qk": judged(False, 2e-4), "value": judged(False, 2e-4)}
        both = {"qk_f16_scalar": judged(False), "value_f16_scalar": judged(False), "qk_f16_vec_dot": judged(True), "value_f16_vec_dot": judged(True)}
        self.assertEqual(classify(both, controls, old), "ATTRIBUTED_BOTH_ATTENTION_DOTS_TO_F16_VEC_DOT")
        conversion = {**both, "qk_f16_scalar": judged(True), "value_f16_scalar": judged(True)}
        self.assertEqual(classify(conversion, controls, old), "F16_CONVERSION_ONLY_SUFFICIENT")
        self.assertEqual(classify(both, {"all": False}, old), "VOID_F16_VEC_DOT_DIAGNOSTIC")

    def test_partial_and_rejected(self) -> None:
        controls = {"all": True}; old = {"qk": judged(False, 2e-4), "value": judged(False, 2e-4)}
        one = {"qk_f16_scalar": judged(False), "value_f16_scalar": judged(False), "qk_f16_vec_dot": judged(True, 0), "value_f16_vec_dot": judged(False, 1e-4)}
        self.assertEqual(classify(one, controls, old), "ATTRIBUTED_QK_TO_F16_VEC_DOT_ONLY")
        neither = {**one, "qk_f16_vec_dot": judged(False, 3e-4), "value_f16_vec_dot": judged(False, 3e-4)}
        self.assertEqual(classify(neither, controls, old), "F16_VEC_DOT_HYPOTHESIS_REJECTED")


if __name__ == "__main__": unittest.main()
