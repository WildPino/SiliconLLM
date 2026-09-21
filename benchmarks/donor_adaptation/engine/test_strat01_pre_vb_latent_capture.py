from __future__ import annotations

import unittest

import numpy as np

from benchmarks.donor_adaptation.engine.run_strat01_pre_vb_latent_capture import callback_to_token_head, classify


def judged(passed: bool) -> dict[str, bool]:
    return {"pass": passed}


def results(latent: bool, agree: bool, vb: bool, layout: bool) -> dict[str, dict[str, bool]]:
    return {
        "reconstruction_vs_kqv": judged(latent),
        "project_vs_pinned": judged(agree),
        "project_vs_kqv_mla": judged(vb),
        "pinned_vs_kqv_mla": judged(vb),
        "kqv_mla_vs_kqv_out": judged(layout),
    }


class PreVbLatentCaptureTests(unittest.TestCase):
    def test_source_derived_callback_axis_mapping(self) -> None:
        width = 2
        raw = np.arange(32 * 8 * width, dtype=np.float32)
        mapped = callback_to_token_head(raw, width).reshape(8, 32, width)
        raw_logical = raw.reshape(32, 8, width)
        self.assertTrue(np.array_equal(mapped[7, 31], raw_logical[31, 7]))
        self.assertTrue(np.array_equal(mapped[0, 1], raw_logical[1, 0]))

    def test_stage_labels_are_disjoint(self) -> None:
        controls = {"all": True}
        self.assertEqual(classify(results(True, True, True, True), controls), "PRE_VB_LATENT_CAPTURE_PASS")
        self.assertEqual(classify(results(False, True, True, True), controls), "ATTRIBUTED_RESIDUAL_TO_ATTENTION_RECONSTRUCTION")
        self.assertEqual(classify(results(True, True, False, True), controls), "ATTRIBUTED_RESIDUAL_TO_VB_GRAPH_SEMANTICS")
        self.assertEqual(classify(results(True, True, True, False), controls), "ATTRIBUTED_RESIDUAL_TO_FINAL_LAYOUT")

    def test_mixed_and_void(self) -> None:
        self.assertEqual(classify(results(False, True, False, True), {"all": True}), "MIXED_PRE_VB_RESIDUAL")
        self.assertEqual(classify(results(True, True, True, True), {"all": False}), "VOID_PRE_VB_LATENT_CAPTURE")


if __name__ == "__main__":
    unittest.main()
