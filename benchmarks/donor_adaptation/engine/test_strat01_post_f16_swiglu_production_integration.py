from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine import run_strat01_post_f16_swiglu_production_integration as runner


class PostF16SwiGLUProductionIntegrationTests(unittest.TestCase):
    def test_protocol_and_immutable_bindings(self) -> None:
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", runner.PROTOCOL.read_text(encoding="utf-8"))
        bindings = runner.validate_bindings()
        self.assertEqual(bindings["sse2"]["sha256"], runner.SSE2_ADJUDICATION_SHA)
        self.assertEqual(set(bindings["reference_manifests"]), {"root", "prefill8", "cached7p1"})

    def test_single_shared_primitive_and_delegation(self) -> None:
        controls = runner.source_controls()
        historical = {name: value for name, value in controls.items() if name != "layer1_swiglu_unchanged"}
        self.assertTrue(all(historical.values()), controls)
        self.assertFalse(controls["layer1_swiglu_unchanged"])
        self.assertTrue(controls["single_shared_definition"])
        self.assertTrue(controls["production_delegates_shared"])
        self.assertTrue(controls["diagnostic_delegates_shared"])

    def test_layer1_coordinate_has_only_the_new_preregistered_change(self) -> None:
        controls = runner.source_controls()
        self.assertFalse(controls["layer1_swiglu_unchanged"])
        self.assertTrue(controls["production_delegates_shared"])

    def test_default_q4_and_f16_coordinates_remain_installed(self) -> None:
        controls = runner.source_controls()
        self.assertTrue(controls["reference_generic_q4_default"])
        self.assertTrue(controls["pinned_f64_f16_default"])

    def test_frozen_helper_count_arithmetic(self) -> None:
        self.assertEqual(runner.EXPECTED_COUNTS["qk_invocations"], 4 * 32 * sum(range(1, 9)))
        self.assertEqual(runner.EXPECTED_COUNTS["value_invocations"], 4 * 8 * 32 * 512)

    def test_status_partition(self) -> None:
        self.assertEqual(runner.classify([]), "PASS_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION")
        self.assertEqual(runner.classify(["checkpoint/prefill8/l_out-1"]), "FAIL_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION")

    def test_scalar_predecessor_is_frozen(self) -> None:
        self.assertEqual(runner.SCALAR_START_SHA, "7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4")


if __name__ == "__main__":
    unittest.main()
