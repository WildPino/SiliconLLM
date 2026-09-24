from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine import recover_strat01_engine_layer2_depth_extension as recovery


class Layer2OfflineRecoveryTests(unittest.TestCase):
    def test_protocol_is_frozen_to_exact_raw_run(self) -> None:
        text = recovery.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("FROZEN BEFORE RECOVERY IMPLEMENTATION OR EXECUTION", text)
        self.assertIn(recovery.SOURCE_ADJUDICATION_SHA, text)
        self.assertIn(recovery.SOURCE_EXECUTION_HEAD, text)
        self.assertEqual(recovery.SOURCE_ADJUDICATION_BYTES, 168_044)

    def test_branch_local_omission_rejects_nonzero_shared_branch(self) -> None:
        reference = {
            f"{arm}/ffn_shexp-2": np.array([0.25, -0.5, 1.0], dtype=np.float32)
            for arm in recovery.layer2.r2c.base.ARMS
        }
        control = recovery.branch_local_omit_shared(reference)
        self.assertFalse(control["pass"])
        self.assertTrue(control["rejects_in_both_schedules"])
        for result in control["arms"].values():
            self.assertEqual(result["nrmse"], 1.0)
            self.assertEqual(result["normalized_max"], 1.0)
            self.assertFalse(result["pass"])

    def test_branch_local_omission_fails_closed_on_zero_branch(self) -> None:
        reference = {
            f"{arm}/ffn_shexp-2": np.zeros(4, dtype=np.float32)
            for arm in recovery.layer2.r2c.base.ARMS
        }
        control = recovery.branch_local_omit_shared(reference)
        self.assertTrue(control["pass"])
        self.assertFalse(control["rejects_in_both_schedules"])

    def test_decision_rule_separates_void_fail_and_pass(self) -> None:
        self.assertEqual(recovery.classify(False, ["checkpoint/prefill8/kqv_out-2"]), "VOID_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERY")
        self.assertEqual(recovery.classify(True, ["checkpoint/prefill8/kqv_out-2"]), "FAIL_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED")
        self.assertEqual(recovery.classify(True, []), "PASS_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED")

    def test_recovery_source_has_no_producer_execution_path(self) -> None:
        text = Path(recovery.__file__).read_text(encoding="utf-8")
        self.assertNotIn("run_command(", text)
        self.assertIn('"new_reference_invocations": 0', text)
        self.assertIn('"new_production_invocations": 0', text)
        self.assertIn('"new_reference_graph_executions": 0', text)
        self.assertIn('"new_donor_graph_executions": 0', text)


if __name__ == "__main__":
    unittest.main()
