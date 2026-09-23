from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_f16_vector_propagation as subject


class F16VectorPropagationTests(unittest.TestCase):
    def metadata(self, hashes: dict[str, str] | None = None) -> dict:
        hashes = hashes or {arm: f"new-{arm}" for arm in subject.r2c.base.ARMS}
        return {
            "manifests": {
                arm: {"tensors": [{"name": "kqv_out-1", "sha256": hashes[arm]}]}
                for arm in subject.r2c.base.ARMS
            },
            "caches": {arm: {} for arm in subject.r2c.base.ARMS},
        }

    def test_inventory_predecessors_and_source_controls(self) -> None:
        self.assertTrue(all(item["sha256"] for item in subject.source_inventory().values()))
        predecessor = subject.validate_predecessors()
        self.assertEqual(set(predecessor["adjudications"]), set(subject.PREDECESSORS))
        self.assertTrue(all(subject.source_controls().values()))
        self.assertIn("test_strat01_f16_vector_propagation", "\n".join(subject.TEST_MODULES))

    def test_counts_are_exact_not_merely_nonzero(self) -> None:
        self.assertEqual(subject.EXPECTED_COUNTS["qk_invocations"], 4608)
        self.assertEqual(subject.EXPECTED_COUNTS["value_invocations"], 524288)

    def test_adjudication_accepts_zero_or_two_superseded_start_failures(self) -> None:
        inherited = {"status": "FAIL_ENGINE_RUNG2C", "failures": []}
        with mock.patch.object(subject.r2c, "adjudicate", return_value=inherited), mock.patch.object(subject, "intervention_controls", return_value={"ok": True}):
            result = subject.adjudicate({}, {}, self.metadata(), {}, {"source": True}, dict(subject.EXPECTED_COUNTS))
        self.assertEqual(result["status"], "PASS_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION")
        inherited = {"status": "FAIL_ENGINE_RUNG2C", "failures": [f"start_state/{arm}" for arm in subject.r2c.base.ARMS]}
        with mock.patch.object(subject.r2c, "adjudicate", return_value=inherited), mock.patch.object(subject, "intervention_controls", return_value={"ok": True}):
            result = subject.adjudicate({}, {}, self.metadata(), {}, {"source": True}, dict(subject.EXPECTED_COUNTS))
        self.assertEqual(len(result["superseded_start_state_failures"]), 2)

    def test_adjudication_rejects_asymmetric_start_and_preserves_numerical_failure(self) -> None:
        inherited = {"status": "FAIL_ENGINE_RUNG2C", "failures": ["start_state/prefill8"]}
        with mock.patch.object(subject.r2c, "adjudicate", return_value=inherited):
            with self.assertRaises(subject.PropagationError):
                subject.adjudicate({}, {}, self.metadata(), {}, {"source": True}, dict(subject.EXPECTED_COUNTS))
        inherited = {"status": "FAIL_ENGINE_RUNG2C", "failures": [*(f"start_state/{arm}" for arm in subject.r2c.base.ARMS), "checkpoint/prefill8/kqv_out-1"]}
        with mock.patch.object(subject.r2c, "adjudicate", return_value=inherited), mock.patch.object(subject, "intervention_controls", return_value={"ok": True}):
            result = subject.adjudicate({}, {}, self.metadata(), {}, {"source": True}, dict(subject.EXPECTED_COUNTS))
        self.assertEqual(result["status"], "FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION")
        self.assertEqual(result["failures"], ["checkpoint/prefill8/kqv_out-1"])

    def test_route_tables_report_exact_ids_and_weights(self) -> None:
        candidate: dict[str, np.ndarray] = {}
        reference: dict[str, np.ndarray] = {}
        for arm in subject.r2c.base.ARMS:
            candidate[f"{arm}/ffn_moe_topk-1"] = reference[f"{arm}/ffn_moe_topk-1"] = np.arange(32, dtype=np.int32)
            candidate[f"{arm}/ffn_moe_weights_norm-1"] = reference[f"{arm}/ffn_moe_weights_norm-1"] = np.full(32, 0.25, dtype=np.float32)
        tables = subject.route_tables(candidate, reference)
        self.assertTrue(all(item["ids_exact"] and item["weight_metrics"]["pass"] for item in tables.values()))

    def test_protocol_records_generic_f64_addendum(self) -> None:
        text = subject.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("Pre-donor implementation addendum", text)
        self.assertIn("pinned-generic-f64", text)
        for status in ("PASS_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION", "FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION", "VOID_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION"):
            self.assertIn(status, text)


if __name__ == "__main__":
    unittest.main()
