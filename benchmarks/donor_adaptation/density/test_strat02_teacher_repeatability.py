#!/usr/bin/env python3
"""Synthetic tests for the STRAT-02 teacher repeatability runner."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "strat02_teacher_repeatability.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("strat02_teacher_repeatability_under_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class TeacherRepeatabilitySyntheticTests(unittest.TestCase):
    def test_runner_selftest_is_fake_only_and_reports_exact_counts(self) -> None:
        completed = subprocess.run([sys.executable, str(RUNNER_PATH), "--selftest"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(report, {"logit_forward_calls": 2, "ok": True, "score_calls": 2, "selftest": True})

    def test_fixed_protocol_and_no_heldout_scoring_argument(self) -> None:
        self.assertEqual(runner.BPB_ABS_TOL, 1e-7)
        self.assertEqual(runner.LOGIT_ATOL, 1e-5)
        self.assertEqual(runner.LOGIT_RTOL, 1e-6)
        self.assertEqual(runner.MONITOR_INTERVAL_SECONDS, 5.0)
        self.assertEqual(runner.MAX_WALL_SECONDS, 45 * 60)
        self.assertEqual(runner.LOGIT_SHAPE, (1, 64, 100352))
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"--heldout"', source)
        self.assertIn("heldout_opened_for_scoring", source)

    def test_logit_tolerance_counts_planted_mismatch(self) -> None:
        torch = runner._load_local_module("strat02_score.py", "_strat02_repeatability_test_score")._torch()
        first = torch.zeros((1, 2, 3), dtype=torch.float32)
        second = first.clone()
        second[0, 1, 2] = 3e-5
        report = runner._compare_logit_blocks(first, second, expected_shape=(1, 2, 3))
        self.assertEqual(report["elements_outside_tolerance"], 1)
        self.assertEqual(report["shape"], [1, 2, 3])

    def test_monitor_reuses_bounded_caps_and_direct_child_only(self) -> None:
        self.assertEqual(runner.WORKER_TOKEN_ENV, "STRAT02_TEACHER_REPEATABILITY_PARENT_TOKEN")
        bounded = runner._load_local_module("strat02_bounded_smoke.py", "_strat02_repeatability_test_bounded")
        self.assertEqual(runner._failure_classification(
            bounded.MonitorOutcome("VOID_RESOURCE", -15, "cap", 1.0, 1), None), "VOID_RESOURCE")
        self.assertEqual(runner._failure_classification(
            bounded.MonitorOutcome("CHILD_EXITED", 1, None, 1.0, 1),
            {"ok": False, "status": "FAIL_REPEATABILITY"}), "FAIL_REPEATABILITY")

    def test_worker_rejects_parent_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-repeatability-provenance-") as directory:
            root = Path(directory)
            expected = runner._source_hashes()
            expected["runner"] = "0" * 64
            manifest = root / "supervisor_manifest.json"
            manifest.write_text(json.dumps({"provenance_sha256": expected}), encoding="utf-8")
            with self.assertRaises(runner.GateError):
                runner._verify_parent_provenance(
                    manifest, result_path=root / "worker_result.json", scores_path=root / "scores.jsonl"
                )


if __name__ == "__main__":
    unittest.main()
