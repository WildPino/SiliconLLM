from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine import run_strat01_q6k_q8k_reference_generic_production_integration as runner
from benchmarks.donor_adaptation.engine.test_strat01_q6k_q8k_reference_generic_parity import REFERENCE_LENGTHS


class Q6ReferenceGenericProductionIntegrationTests(unittest.TestCase):
    def test_protocol_and_predecessor_are_bound(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", protocol)
        self.assertIn("Pre-implementation observability addendum", protocol)
        bindings = runner.validate_bindings()
        self.assertEqual(bindings["q6_reference_generic"]["sha256"], runner.Q6_ADJUDICATION_SHA)

    def test_source_delegation_and_unchanged_coordinates(self) -> None:
        controls = runner.source_controls()
        self.assertTrue(all(controls.values()), controls)
        self.assertTrue(controls["production_wrapper_delegates_exactly"])
        self.assertTrue(controls["old_production_arithmetic_removed"])

    def test_routed_expert_width_is_model_free_qualified(self) -> None:
        self.assertIn(1280, REFERENCE_LENGTHS)

    def test_status_partition_and_terminal_hashes(self) -> None:
        self.assertEqual(runner.classify([]), "PASS_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION")
        self.assertEqual(runner.classify(["checkpoint/prefill8/l_out-0"]), "FAIL_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION")
        self.assertNotEqual(runner.OLD_L_OUT0_SHA, runner.EXACT_L_OUT0_SHA)


if __name__ == "__main__":
    unittest.main()
