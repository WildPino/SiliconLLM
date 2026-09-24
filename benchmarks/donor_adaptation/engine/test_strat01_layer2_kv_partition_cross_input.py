from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_partition_cross_input as runner


class Layer2KvPartitionCrossInputTests(unittest.TestCase):
    def test_protocol_and_partition_are_frozen(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        header = runner.CROSS_HEADER.read_text(encoding="utf-8")
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", protocol)
        self.assertIn(runner.PREDECESSOR_ADJUDICATION_SHA, protocol)
        self.assertIn("STRAT01_L2KPX_PREFIX 512U", header)
        self.assertIn("STRAT01_L2KPX_TAIL 64U", header)
        self.assertNotIn("strat01_r2a_attend_one(", header)
        self.assertIn("strat01_r2c_cross_run_arm", header)

    def test_exact_pinned_payload_set_and_twins(self) -> None:
        self.assertEqual(set(runner.PINNED_FILES), {"ref_q", "ref_k", "ref_v", "ref_target", "c_k", "c_v"})
        self.assertEqual(set(runner.SCHEDULE_TWINS), set(runner.PINNED_FILES))
        self.assertEqual(runner.PINNED_FILES["ref_k"][2], 18_432)
        self.assertEqual(runner.PINNED_FILES["ref_v"][2], 16_384)

    def test_engine_has_separate_command_and_selftest(self) -> None:
        engine = runner.ENGINE.read_text(encoding="utf-8")
        self.assertIn('#include "strat01_gguf_layer2_kv_partition_cross_input.h"', engine)
        self.assertIn("--strat01-layer2-kv-partition-cross-input", engine)
        self.assertIn("--strat01-layer2-kv-partition-cross-input-selftest", engine)

    def test_arm_manifest_is_exact(self) -> None:
        outputs = {}
        for name in runner.ARM_NAMES:
            latent, positional, lc, pc = runner.ARM_META[name]
            outputs[name] = {"path": name, "bytes": 196_608, "sha256": "0" * 64, "q_source": "reference", "latent_source": latent, "positional_source": positional, "latent_control": lc, "positional_control": pc}
        runner.validate_arm_manifest(outputs)
        bad = dict(outputs); bad["unexpected"] = bad.pop(runner.ARM_NAMES[-1])
        with self.assertRaises(runner.RunnerError):
            runner.validate_arm_manifest(bad)

    def test_classification_preserves_all_causal_cases(self) -> None:
        passed, failed = {"pass": True}, {"pass": False}
        arms = {"ref_latent__ref_positional": passed, "c_latent__c_positional": failed, "c_latent__ref_positional": passed, "ref_latent__c_positional": passed}
        self.assertEqual(runner.classify(arms), "LAYER2_JOINT_KV_PARTITION_INTERACTION_SUFFICIENT")
        arms["c_latent__ref_positional"] = failed
        self.assertEqual(runner.classify(arms), "LAYER2_LATENT_VALUE_RESIDUAL_SUFFICIENT")
        arms["ref_latent__c_positional"] = failed
        self.assertEqual(runner.classify(arms), "LAYER2_LATENT_AND_POSITIONAL_RESIDUALS_INDEPENDENTLY_SUFFICIENT")
        arms["c_latent__ref_positional"] = passed
        self.assertEqual(runner.classify(arms), "LAYER2_POSITIONAL_TAIL_RESIDUAL_SUFFICIENT")
        arms["ref_latent__ref_positional"] = failed
        self.assertEqual(runner.classify(arms), "LAYER2_EXACT_REFERENCE_FAILS_KV_PARTITION_OPERATOR")

    def test_partition_and_metric_gates_are_unchanged(self) -> None:
        runner.validate_partition({"latent_value": [0, 512], "positional": [512, 576]})
        with self.assertRaises(runner.RunnerError):
            runner.validate_partition({"latent_value": [0, 511], "positional": [511, 576]})
        reference = np.ones(8 * 6144, dtype=np.float32)
        self.assertTrue(runner.base.judged(reference.copy(), reference)["pass"])
        changed = reference.copy(); changed[0] += 100.0
        self.assertFalse(runner.base.judged(changed, reference)["pass"])
        self.assertEqual(runner.base.LIMITS, (0.002, 0.01))

    def test_source_inventory_is_complete(self) -> None:
        inventory = runner.source_inventory()
        self.assertEqual(set(inventory), {"runner", "tests", "protocol", "engine", "rung2a", "rung2c", "inherited_cross_header", "predecessor_cross_header", "kv_partition_header"})
        self.assertTrue(all(len(item["sha256"]) == 64 and Path(item["path"]).is_file() for item in inventory.values()))


if __name__ == "__main__":
    unittest.main()
