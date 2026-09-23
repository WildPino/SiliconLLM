from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2c as runner


class Rung2CTests(unittest.TestCase):
    def test_frozen_checkpoint_surface(self) -> None:
        self.assertEqual(len(runner.SHAPES), 32)
        self.assertEqual(runner.SHAPES["ffn_moe_topk-1"], [4, 8])
        self.assertEqual(runner.SHAPES["ffn_moe_up-1"], [1280, 4, 8])
        self.assertEqual(runner.SHAPES["l_out-1"], [1536, 8])
        self.assertEqual(runner.I32_NAMES, {"ffn_moe_topk-1"})

    def test_source_controls_delegate_accepted_primitives(self) -> None:
        controls = runner.source_controls()
        self.assertTrue(controls)
        self.assertTrue(all(controls.values()), controls)

    def test_all_ten_negative_controls_reject(self) -> None:
        prefix = "prefill8/"
        logits = np.tile(np.linspace(-2.0, 2.0, 64, dtype=np.float32), (8, 1))
        probs = 1.0 / (1.0 + np.exp(-logits))
        ids = np.tile(np.array([63, 62, 61, 60], dtype=np.int32), (8, 1))
        weights = np.take_along_axis(probs, ids, axis=1)
        norm = weights / weights.sum(axis=1, keepdims=True)
        biased = probs + np.linspace(0.0, 0.5, 64, dtype=np.float32)
        gate = np.linspace(-2.0, 2.0, 8 * 4 * 1280, dtype=np.float32)
        up = np.linspace(0.5, 3.0, gate.size, dtype=np.float32)
        routed = np.linspace(-1.0, 1.0, 8 * 1536, dtype=np.float32)
        shared = np.linspace(0.25, 0.75, routed.size, dtype=np.float32)
        ffn_out = routed + shared
        residual = np.linspace(-0.1, 0.1, routed.size, dtype=np.float32)
        reference = {
            prefix + "ffn_moe_topk-1": ids.ravel(), prefix + "ffn_moe_probs-1": probs.ravel(),
            prefix + "ffn_moe_probs_biased-1": biased.ravel(), prefix + "ffn_moe_logits-1": logits.ravel(),
            prefix + "ffn_moe_weights-1": weights.ravel(), prefix + "ffn_moe_weights_norm-1": norm.ravel(),
            prefix + "ffn_moe_gate-1": gate, prefix + "ffn_moe_up-1": up,
            prefix + "ffn_moe_swiglu-1": (gate / (1.0 + np.exp(-gate)) * up).astype(np.float32),
            prefix + "ffn_moe_out-1": routed, prefix + "ffn_shexp-1": shared,
            prefix + "ffn_out-1": ffn_out, prefix + "l_out-1": ffn_out + residual,
        }
        controls = runner.negative_controls(reference)
        self.assertEqual(len(controls), 10)
        self.assertTrue(all(not item["pass"] for item in controls.values()), controls)

    def test_typed_payload_loader_rejects_digest_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.i32le"
            np.array([1, 2, 3, 4], dtype="<i4").tofile(path)
            digest = runner.base.sha256_file(path)
            np.testing.assert_array_equal(runner.load_typed(path, 4, digest, "I32", "fixture"), [1, 2, 3, 4])
            with self.assertRaises(runner.RunnerError):
                runner.load_typed(path, 4, "0" * 64, "I32", "fixture")

    def test_start_hashes_are_distinct_immutable_witnesses(self) -> None:
        self.assertEqual(len(runner.C_START_SHA), 64)
        self.assertEqual(len(runner.REFERENCE_START_SHA), 64)
        self.assertNotEqual(runner.C_START_SHA, runner.REFERENCE_START_SHA)
        self.assertNotEqual("0" + runner.C_START_SHA[1:], runner.C_START_SHA)

    def test_execution_accounting_distinguishes_invocations_and_schedules(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("reference_producer_invocations = 1", source)
        self.assertIn("donor_producer_invocations = 1", source)
        complete = {"stdout": "", "stderr": "STRAT01_RUNG2C_GRAPH_COMPLETE arm=prefill8\nSTRAT01_RUNG2C_GRAPH_COMPLETE arm=cached7p1\n"}
        partial = {"stdout": "STRAT01_RUNG2C_GRAPH_COMPLETE arm=prefill8\n", "stderr": "producer failed"}
        self.assertEqual(runner.completed_graph_count(complete), 2)
        self.assertEqual(runner.completed_graph_count(partial), 1)
        with self.assertRaises(runner.RunnerError):
            runner.completed_graph_count({"stdout": complete["stderr"] * 2, "stderr": ""})


if __name__ == "__main__":
    unittest.main()
