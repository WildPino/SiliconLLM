from __future__ import annotations

import unittest
from unittest import mock

from benchmarks.donor_adaptation.engine import run_strat01_reference_generic_propagation as subject


class ReferenceGenericPropagationTests(unittest.TestCase):
    def metadata(self, candidate: str, reference: str) -> tuple[dict, dict]:
        candidate_meta = {
            "manifests": {
                arm: {"tensors": [{"name": "l_out-0", "sha256": candidate}]}
                for arm in subject.r2c.base.ARMS
            }
        }
        reference_meta = {
            "manifests": {
                arm: {"payloads": [{"logical": "l_out-0", "kind": "full", "sha256": reference}]}
                for arm in subject.r2c.base.ARMS
            }
        }
        return candidate_meta, reference_meta

    def test_source_inventory_and_controls(self) -> None:
        inventory = subject.source_inventory()
        self.assertIn("strat01_q4k_q8k.h", {item["path"].replace("\\", "/").split("/")[-1] for item in inventory.values()})
        self.assertTrue(all(subject.source_controls().values()))
        modules = "\n".join(subject.TEST_MODULES)
        self.assertIn("test_strat01_reference_generic_propagation", modules)
        self.assertIn("test_strat01_engine_rung2c", modules)

    def test_intervention_controls_require_new_shared_candidate(self) -> None:
        candidate_meta, reference_meta = self.metadata("new", subject.REFERENCE_START_SHA)
        self.assertTrue(all(subject.intervention_controls(candidate_meta, reference_meta).values()))
        old_meta, reference_meta = self.metadata(subject.OLD_C_START_SHA, subject.REFERENCE_START_SHA)
        self.assertFalse(subject.intervention_controls(old_meta, reference_meta)["changed_coordinate_reaches_graph"])

    def test_adjudication_removes_only_expected_old_start_failures(self) -> None:
        candidate_meta, reference_meta = self.metadata("new", subject.REFERENCE_START_SHA)
        inherited = {
            "status": "FAIL_ENGINE_RUNG2C",
            "failures": [f"start_state/{arm}" for arm in subject.r2c.base.ARMS],
        }
        with mock.patch.object(subject.r2c, "adjudicate", return_value=inherited):
            result = subject.adjudicate({}, {}, candidate_meta, reference_meta, {"ok": True})
        self.assertEqual(result["status"], "PASS_ENGINE_REFERENCE_GENERIC_PROPAGATION")
        self.assertEqual(result["failures"], [])
        self.assertEqual(len(result["superseded_old_start_failures"]), 2)

    def test_numerical_failure_remains_failure(self) -> None:
        candidate_meta, reference_meta = self.metadata("new", subject.REFERENCE_START_SHA)
        inherited = {
            "status": "FAIL_ENGINE_RUNG2C",
            "failures": [
                *(f"start_state/{arm}" for arm in subject.r2c.base.ARMS),
                "checkpoint/prefill8/kqv_out-1",
            ],
        }
        with mock.patch.object(subject.r2c, "adjudicate", return_value=inherited):
            result = subject.adjudicate({}, {}, candidate_meta, reference_meta, {"ok": True})
        self.assertEqual(result["status"], "FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION")
        self.assertEqual(result["failures"], ["checkpoint/prefill8/kqv_out-1"])


if __name__ == "__main__":
    unittest.main()
