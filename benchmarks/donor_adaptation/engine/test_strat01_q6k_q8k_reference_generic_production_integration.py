from __future__ import annotations

import unittest
from unittest import mock

from benchmarks.donor_adaptation.engine import run_strat01_q6k_q8k_reference_generic_production_integration as runner
from benchmarks.donor_adaptation.engine.test_strat01_q6k_q8k_reference_generic_parity import REFERENCE_LENGTHS


class Q6ReferenceGenericProductionIntegrationTests(unittest.TestCase):
    def test_protocol_and_predecessor_are_bound(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", protocol)
        self.assertIn("Pre-implementation observability addendum", protocol)
        self.assertIn("Post-execution offline-recovery addendum", protocol)
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

    def test_extended_config_is_exact_and_historical_validator_is_restored(self) -> None:
        self.assertEqual(runner.EXPECTED_C_CONFIG.count(runner.Q6_CONFIG_MARKER), 1)
        self.assertEqual(
            runner.EXPECTED_C_CONFIG,
            runner.r2c.EXPECTED_C_CONFIG.replace(
                "kb=q5_0xq8_0;", "kb=q5_0xq8_0;" + runner.Q6_CONFIG_MARKER,
            ),
        )
        historical = runner.r2c.EXPECTED_C_CONFIG
        fake_root = runner.ROOT
        with mock.patch.object(runner, "read_json", return_value={"CONFIG": runner.EXPECTED_C_CONFIG}), mock.patch.object(
            runner.r2c, "validate_c", return_value=({}, {})
        ) as validate:
            runner.validate_candidate(fake_root, {}, fake_root)
            self.assertEqual(validate.call_count, 1)
        self.assertEqual(runner.r2c.EXPECTED_C_CONFIG, historical)

    def test_recovery_is_distinct_and_never_runs_a_producer(self) -> None:
        source = runner.Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotEqual(runner.DEFAULT_RECOVERY, runner.DEFAULT_OUTPUT)
        recovery = source[source.index("def recover_existing("):source.index("def main()")]
        self.assertNotIn("run_command(", recovery)
        self.assertIn('"new_production_invocations": 0', recovery)
        self.assertIn('"new_donor_graph_executions": 0', recovery)


if __name__ == "__main__":
    unittest.main()
