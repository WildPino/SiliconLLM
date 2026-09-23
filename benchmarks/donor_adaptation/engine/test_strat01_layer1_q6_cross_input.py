from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_layer1_q6_cross_input as subject


class Layer1Q6CrossInputTests(unittest.TestCase):
    def test_source_and_input_inventory(self) -> None:
        self.assertTrue(all(item["sha256"] for item in subject.source_inventory().values()))
        self.assertEqual(set(subject.validate_inputs()), set(subject.INPUTS))
        self.assertIn("test_strat01_layer1_q6_cross_input", "\n".join(subject.TEST_MODULES))

    def test_status_partition(self) -> None:
        values = {
            "reference_routed_current_q6": np.zeros(49152,dtype=np.float32),
            "current_routed_current_q6": np.zeros(49152,dtype=np.float32),
            "reference_shared_current_q6": np.zeros(12288,dtype=np.float32),
            "current_shared_current_q6": np.zeros(12288,dtype=np.float32),
            "control_negated_reference_routed": np.ones(49152,dtype=np.float32),
            "control_negated_reference_shared": np.ones(12288,dtype=np.float32),
            "control_mutated_expert_id": np.ones(49152,dtype=np.float32),
        }
        paths = {name: spec[0] for name,spec in subject.INPUTS.items()}
        meta = {"ids": np.zeros((8,4),dtype=np.int32)}
        with mock.patch.object(subject,"load_f32",side_effect=lambda path,count: np.zeros(count,dtype=np.float32)):
            result=subject.adjudicate(values,meta,paths)
        self.assertEqual(result["status"],"LAYER1_SWIGLU_RESIDUAL_SUFFICIENT")

    def test_metric_gate_detects_residual(self) -> None:
        reference=np.ones(1536,dtype=np.float32);candidate=reference.copy();candidate[0]=2
        self.assertFalse(subject.judged(candidate,reference)["pass"])

    def test_protocol_names_all_nonvoid_outcomes(self) -> None:
        text=subject.PROTOCOL.read_text(encoding="utf-8")
        for status in ("LAYER1_SWIGLU_RESIDUAL_SUFFICIENT","LAYER1_Q6_OPERATOR_RESIDUALS","ROUTED_Q6_OPERATOR_RESIDUAL","SHARED_Q6_OPERATOR_RESIDUAL"):
            self.assertIn(status,text)


if __name__ == "__main__": unittest.main()
