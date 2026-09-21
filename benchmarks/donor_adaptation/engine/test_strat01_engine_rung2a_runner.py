"""Model-free unit tests for the STRAT-01 rung-2A adjudicator."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2a as runner


class Rung2ARunnerTests(unittest.TestCase):
    def test_reference_callback_type_uses_ggml_case_insensitively(self) -> None:
        selection = {
            "logical_shape": runner.SHAPES["attn_norm-0"],
            "composition": "single_prefill8_callback",
            "source": {
                "name": "attn_norm-0",
                "op": runner.EXPECTED_OPS["attn_norm-0"],
                "ordinal": runner.EXPECTED_ORDINALS["attn_norm-0"],
                "type": "f32",
                "shape": runner.SHAPES["attn_norm-0"],
            },
        }
        _, token_lengths = runner.reference_source_identity(selection, "prefill8", "attn_norm-0")
        self.assertEqual(token_lengths, [8])
        selection["source"]["type"] = "q4_K"
        with self.assertRaises(runner.RunnerError):
            runner.reference_source_identity(selection, "prefill8", "attn_norm-0")

    def test_frozen_metric_definition_and_gate(self) -> None:
        reference = np.array([3.0, 4.0], dtype=np.float32)
        candidate = np.array([3.0, 3.0], dtype=np.float32)
        result = runner.judged_metrics(candidate, reference, (0.21, 0.26))
        self.assertAlmostEqual(result["nrmse"], 0.2)
        self.assertAlmostEqual(result["normalized_max"], 0.25)
        self.assertTrue(result["pass"])
        self.assertFalse(runner.judged_metrics(candidate, reference, (0.19, 0.26))["pass"])

    def test_token7_uses_last_logical_token_not_flat_tail_guess(self) -> None:
        shape = [2, 3, 8]
        values = np.arange(np.prod(shape), dtype=np.float32)
        np.testing.assert_array_equal(runner.token7(values, shape), np.arange(42, 48, dtype=np.float32))

    def test_payload_hash_count_and_finiteness_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.f32"
            values = np.array([1.0, 2.0], dtype=np.dtype("<f4"))
            values.tofile(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            np.testing.assert_array_equal(runner.load_payload(path, 2, digest, "fixture"), values)
            with self.assertRaises(runner.RunnerError):
                runner.load_payload(path, 3, digest, "fixture")
            with self.assertRaises(runner.RunnerError):
                runner.load_payload(path, 2, "0" * 64, "fixture")

    def test_adjudicator_negative_control_fires(self) -> None:
        c: dict[str, np.ndarray] = {}
        r: dict[str, np.ndarray] = {}
        for arm in runner.ARMS:
            for name, shape in runner.SHAPES.items():
                values = np.ones(np.prod(shape), dtype=np.float32)
                c[f"{arm}/{name}"] = values.copy()
                r[f"{arm}/{name}"] = values.copy()
        c_cache = {
            "prefill8/final": np.ones(8 * 576, dtype=np.float32),
            "cached7p1/prefix7": np.ones(7 * 576, dtype=np.float32),
            "cached7p1/final": np.ones(8 * 576, dtype=np.float32),
        }
        ref_cache = {key: value.copy() for key, value in c_cache.items()}
        self.assertEqual(runner.adjudicate(c, r, c_cache, ref_cache)["status"], "PASS_ENGINE_RUNG2A")
        c["cached7p1/ffn_inp-0"][7 * 1536] = 2.0
        failed = runner.adjudicate(c, r, c_cache, ref_cache)
        self.assertEqual(failed["status"], "FAIL_ENGINE_RUNG2A")
        self.assertIn("cached7p1/ffn_inp-0 numerical parity", failed["failures"])
        self.assertIn("c_engine prefill/cache continuity", failed["failures"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
