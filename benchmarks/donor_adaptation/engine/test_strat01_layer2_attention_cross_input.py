from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_layer2_attention_cross_input as runner


class Layer2AttentionCrossInputTests(unittest.TestCase):
    def test_protocol_and_new_coordinate_are_frozen(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        header = runner.CROSS_HEADER.read_text(encoding="utf-8")
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", protocol)
        self.assertIn(runner.RECOVERY_SHA, protocol)
        self.assertIn("blk.2.attn_v_b.weight", header)
        self.assertNotIn('strat01_r2a_attend_one(', header)
        self.assertIn("strat01_r2c_cross_run_arm", header)

    def test_exact_pinned_payload_set_and_schedule_twins(self) -> None:
        self.assertEqual(set(runner.PINNED_FILES), {"ref_q", "ref_k", "ref_v", "ref_target", "c_q", "c_k", "c_v", "c_target"})
        self.assertEqual(set(runner.SCHEDULE_TWINS), set(runner.PINNED_FILES))
        self.assertEqual(runner.PINNED_FILES["ref_q"][2], 589_824)
        self.assertEqual(runner.PINNED_FILES["ref_target"][2], 196_608)
        header = runner.CROSS_HEADER.read_text(encoding="utf-8")
        for name in ("ref_q", "ref_k", "ref_v", "c_q", "c_k", "c_v"):
            self.assertIn(runner.PINNED_FILES[name][1], header)

    def test_engine_has_separate_command_and_selftest(self) -> None:
        engine = runner.ENGINE.read_text(encoding="utf-8")
        self.assertIn('#include "strat01_gguf_layer2_attention_cross_input.h"', engine)
        self.assertIn("--strat01-layer2-attention-cross-input", engine)
        self.assertIn("--strat01-layer2-attention-cross-input-selftest", engine)

    def test_classification_preserves_all_causal_cases(self) -> None:
        exact = {"pass": True}; failed = {"pass": False}
        arms = {"c_q__c_kv": failed, "ref_q__ref_kv": exact, "c_q__ref_kv": exact, "ref_q__c_kv": exact}
        self.assertEqual(runner.classify(arms), "LAYER2_JOINT_QUERY_KV_INTERACTION_SUFFICIENT")
        arms["c_q__ref_kv"] = failed
        self.assertEqual(runner.classify(arms), "LAYER2_QUERY_RESIDUAL_SUFFICIENT")
        arms["ref_q__c_kv"] = failed
        self.assertEqual(runner.classify(arms), "LAYER2_QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT")
        arms["c_q__ref_kv"] = exact
        self.assertEqual(runner.classify(arms), "LAYER2_KV_RESIDUAL_SUFFICIENT")
        arms["ref_q__ref_kv"] = failed
        self.assertEqual(runner.classify(arms), "LAYER2_EXACT_REFERENCE_FAILS_ATTENTION_VB_OPERATOR")

    def test_frozen_metric_gate_is_unchanged(self) -> None:
        reference = np.ones(8 * 6144, dtype=np.float32)
        self.assertTrue(runner.base.judged(reference.copy(), reference)["pass"])
        changed = reference.copy(); changed[0] += 100.0
        self.assertFalse(runner.base.judged(changed, reference)["pass"])
        self.assertEqual(runner.base.LIMITS, (0.002, 0.01))

    def test_source_inventory_is_complete(self) -> None:
        inventory = runner.source_inventory()
        self.assertEqual(set(inventory), {"runner", "tests", "protocol", "engine", "rung2a", "rung2c", "inherited_cross_header", "layer2_cross_header"})
        self.assertTrue(all(len(item["sha256"]) == 64 and Path(item["path"]).is_file() for item in inventory.values()))


if __name__ == "__main__":
    unittest.main()
