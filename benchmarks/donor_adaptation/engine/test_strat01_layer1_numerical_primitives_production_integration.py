from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine import run_strat01_layer1_numerical_primitives_production_integration as runner


class Layer1NumericalPrimitivesProductionIntegrationTests(unittest.TestCase):
    def test_protocol_and_positive_predecessor_are_bound(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", protocol)
        bindings = runner.validate_bindings()
        self.assertEqual(bindings["layer1_numerical_primitives"]["sha256"], runner.PRIMITIVE_ADJUDICATION_SHA)

    def test_source_ownership_and_delegation(self) -> None:
        controls = runner.source_controls()
        self.assertTrue(all(controls.values()), controls)
        self.assertTrue(controls["router_delegates_exactly"])
        self.assertTrue(controls["routed_swiglu_delegates_exactly"])
        self.assertTrue(controls["shared_swiglu_delegates_exactly"])

    def test_configuration_extends_q6_configuration_exactly(self) -> None:
        self.assertEqual(runner.EXPECTED_C_CONFIG.count(runner.ROUTER_MARKER), 1)
        self.assertEqual(runner.EXPECTED_C_CONFIG.count(runner.SWIGLU_MARKER), 1)
        self.assertEqual(
            runner.EXPECTED_C_CONFIG.replace(runner.ROUTER_MARKER + runner.SWIGLU_MARKER, ""),
            runner.prior.EXPECTED_C_CONFIG,
        )

    def test_three_frozen_primitive_hashes(self) -> None:
        self.assertEqual(set(runner.EXPECTED_PRIMITIVE_HASHES), {
            "ffn_moe_logits-1", "ffn_moe_swiglu-1", "ffn_swiglu-1",
        })
        self.assertTrue(all(len(value) == 64 for value in runner.EXPECTED_PRIMITIVE_HASHES.values()))


if __name__ == "__main__":
    unittest.main()
