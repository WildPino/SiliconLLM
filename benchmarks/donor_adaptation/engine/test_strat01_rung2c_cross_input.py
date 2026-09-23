from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as runner


class Rung2CCrossInputTests(unittest.TestCase):
    def test_frozen_protocol_and_source_contract(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        source = runner.CROSS_HEADER.read_text(encoding="utf-8")
        engine = runner.ENGINE.read_text(encoding="utf-8")
        self.assertIn("Exactly one non-VOID execution", protocol)
        self.assertIn("c_q__ref_kv", protocol)
        for name in ("ref_q", "ref_k", "ref_v", "c_q", "c_k", "c_v"):
            self.assertIn(runner.PINNED_FILES[name][1], source)
        self.assertIn("--strat01-rung2c-cross-input", engine)

    def test_metrics_gate_and_classification(self) -> None:
        reference = np.ones(8 * 6144, dtype=np.float32)
        exact = runner.judged(reference.copy(), reference)
        failed_values = reference.copy(); failed_values[0] += 100.0
        failed = runner.judged(failed_values, reference)
        self.assertTrue(exact["pass"]); self.assertFalse(failed["pass"])
        arms = {"c_q__c_kv": failed, "ref_q__ref_kv": exact, "c_q__ref_kv": exact, "ref_q__c_kv": exact}
        self.assertEqual(runner.classify(arms), "JOINT_QUERY_KV_INTERACTION_SUFFICIENT")
        arms["c_q__ref_kv"] = failed
        self.assertEqual(runner.classify(arms), "QUERY_RESIDUAL_SUFFICIENT")
        arms["ref_q__c_kv"] = failed
        self.assertEqual(runner.classify(arms), "QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT")

    def test_mutated_identity_is_refused(self) -> None:
        data = b"frozen Rung-2C diagnostic payload"
        digest = hashlib.sha256(data).hexdigest()
        self.assertTrue(runner.identity_matches(data, len(data), digest))
        mutated = bytearray(data); mutated[3] ^= 1
        self.assertFalse(runner.identity_matches(bytes(mutated), len(data), digest))

    def test_arm_label_swap_is_rejected(self) -> None:
        outputs = {}
        for name, spec in runner.ARM_SPECS.items():
            outputs[name] = {"path": f"{name}.f32le", "bytes": 196_608, "sha256": "0" * 64, "q_source": spec[0], "kv_source": spec[1], "query_control": spec[2], "kv_control": spec[3]}
        runner.validate_arm_manifest(outputs)
        swapped = copy.deepcopy(outputs)
        swapped["c_q__ref_kv"], swapped["ref_q__c_kv"] = swapped["ref_q__c_kv"], swapped["c_q__ref_kv"]
        with self.assertRaises(runner.RunnerError):
            runner.validate_arm_manifest(swapped)

    def test_source_inventory_is_complete(self) -> None:
        inventory = runner.source_inventory()
        self.assertEqual(set(inventory), {"runner", "engine", "rung2a", "rung2c", "cross_header", "protocol", "tests"})
        for item in inventory.values():
            self.assertEqual(len(item["sha256"]), 64)
            self.assertTrue(Path(item["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
