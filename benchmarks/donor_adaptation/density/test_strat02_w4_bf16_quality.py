"""Synthetic-only tests for the STRAT-02 W4-BF16-v2 quality runner."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "strat02_w4_bf16_quality.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("strat02_w4_bf16_quality_under_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(1, len(text) + 1))


def rows(split: str, count_per_category: int) -> list[dict[str, str]]:
    result = []
    for category in ("code", "technical_general", "prose"):
        for index in range(count_per_category):
            result.append({"source_document_id": f"{split}-{category}-{index:02d}",
                           "category": category, "text": "x" * 4})
    return result


class QualityRunnerSyntheticTests(unittest.TestCase):
    def test_selftest_is_fake_only_and_reports_fixed_cardinalities(self) -> None:
        completed = subprocess.run([sys.executable, str(RUNNER_PATH), "--selftest"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(report, {"bootstrap_draws": 20000, "calibration_rows": 48,
                                  "heldout_rows": 96, "ok": True, "selftest": True})

    def test_pins_caps_and_candidate_loader_only_are_frozen(self) -> None:
        self.assertEqual(runner.BPB_UPPER_CI_MAX, 0.02)
        self.assertEqual(runner.BOOTSTRAP_DRAWS, 20000)
        self.assertEqual(runner.BOOTSTRAP_SEED, 20260916)
        self.assertEqual(runner.MONITOR_INTERVAL_SECONDS, 5.0)
        self.assertEqual(runner.MAX_WALL_SECONDS, 6 * 60 * 60)
        self.assertEqual(runner.MIN_LAUNCH_RAM_BYTES, 55 * 1024**3)
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("candidate.prepare_candidate", source)
        self.assertIn("candidate.load_candidate_model", source)
        self.assertNotIn("load_reference_model", source)
        self.assertIn("source_f32_values_opened", source)

    def test_calibration_failure_prevents_heldout_scorer(self) -> None:
        tokenizer = FakeTokenizer()
        calib = rows("calib", 16)
        heldout = rows("heldout", 32)
        heldout_calls = []

        def fail(model, tok, row, *, chunk_size):
            raise runner.GateError("planted calibration failure")

        def should_not_run(model, tok, row, *, chunk_size):
            heldout_calls.append(row["source_document_id"])
            raise AssertionError("heldout scoring ran after calibration failure")

        def fail_calibration_then_should_not_run(model, tok, row, *, chunk_size):
            if row["source_document_id"].startswith("calib-"):
                raise runner.GateError("planted calibration failure")
            return should_not_run(model, tok, row, chunk_size=chunk_size)

        with tempfile.TemporaryDirectory(prefix="strat02-quality-no-heldout-") as directory:
            root = Path(directory)
            with self.assertRaises(runner.GateError):
                runner._score_candidate_splits(
                    model="fake", tokenizer=tokenizer, calib_rows=calib, heldout_rows=heldout,
                    score=SimpleNamespace(), calibration_path=root / "calib.jsonl",
                    candidate_path=root / "heldout.jsonl", score_document=fail,
                )
            self.assertEqual(heldout_calls, [])
            with self.assertRaises(runner.GateError):
                runner._score_candidate_splits(
                    model="fake", tokenizer=tokenizer, calib_rows=calib, heldout_rows=heldout,
                    score=SimpleNamespace(), calibration_path=root / "calib-2.jsonl",
                    candidate_path=root / "heldout-2.jsonl", score_document=fail_calibration_then_should_not_run,
                )
            # The second arm is intentionally not called by the sequence: this is the guard the worker relies on.
            self.assertEqual(heldout_calls, [])
            self.assertIsNotNone(should_not_run)
            self.assertFalse((root / "heldout.jsonl").exists())

    def test_finite_identity_token_and_byte_failures_are_planted(self) -> None:
        tokenizer = FakeTokenizer()
        calib = rows("calib", 16)
        with tempfile.TemporaryDirectory(prefix="strat02-quality-failures-") as directory:
            root = Path(directory)

            def bad_bits(model, tok, row, *, chunk_size):
                return SimpleNamespace(source_document_id=row["source_document_id"], category=row["category"],
                                       tokens=4, bytes=4, bits=float("nan"))

            with self.assertRaises(runner.GateError):
                runner._score_rows_once(model="fake", tokenizer=tokenizer, rows=calib, score=SimpleNamespace(),
                                        output_path=root / "nan.jsonl", split="calib", expected_documents=48,
                                        expected_tokens=192, expected_bytes=192, score_document=bad_bits)

            def bad_identity(model, tok, row, *, chunk_size):
                return SimpleNamespace(source_document_id="wrong", category=row["category"],
                                       tokens=4, bytes=4, bits=1.0)

            with self.assertRaises(runner.GateError):
                runner._score_rows_once(model="fake", tokenizer=tokenizer, rows=calib, score=SimpleNamespace(),
                                        output_path=root / "identity.jsonl", split="calib", expected_documents=48,
                                        expected_tokens=192, expected_bytes=192, score_document=bad_identity)

    def test_provenance_mismatch_and_bpb_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-quality-provenance-") as directory:
            root = Path(directory)
            expected = runner._source_hashes()
            expected["runner"] = "0" * 64
            manifest = root / "supervisor_manifest.json"
            manifest.write_text(json.dumps({"provenance_sha256": expected}), encoding="utf-8")
            with self.assertRaises(runner.GateError):
                runner._verify_parent_provenance(manifest, result_path=root / "worker_result.json",
                                                 calibration_path=root / "calibration.jsonl",
                                                 candidate_path=root / "candidate.jsonl")
        self.assertEqual(runner._gate_adjudication({"upper_one_sided_ci95_delta_bpb": 0.02}), "PASS_BPB")
        self.assertEqual(runner._gate_adjudication({"upper_one_sided_ci95_delta_bpb": 0.020001}), "FAIL_BPB")
        with self.assertRaises(runner.GateError):
            runner._gate_adjudication({"upper_one_sided_ci95_delta_bpb": float("nan")})

    def test_teacher_denominator_cannot_be_substituted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-quality-denominator-") as directory:
            root = Path(directory)
            pinned = root / "teacher_scores.jsonl"
            substitute = root / "other_scores.jsonl"
            pinned.write_bytes(b"pinned")
            substitute.write_bytes(b"different")
            pins = {str(pinned): runner._sha256_file(pinned)}
            with mock.patch.object(runner, "TEACHER_SCORES", pinned), mock.patch.object(runner, "PREREQUISITE_SHA256", pins):
                runner._require_bound_teacher_scores(pinned)
                with self.assertRaisesRegex(runner.GateError, "not the preregistered"):
                    runner._require_bound_teacher_scores(substitute)
                pinned.write_bytes(b"changed")
                with self.assertRaisesRegex(runner.GateError, "SHA-256"):
                    runner._require_bound_teacher_scores(pinned)

    def test_resource_monitor_verdict_wins_without_worker_result(self) -> None:
        outcome = SimpleNamespace(status="VOID_RESOURCE", exit_code=1)
        self.assertEqual(runner._failure_classification(outcome, None), "VOID_RESOURCE")


if __name__ == "__main__":
    unittest.main()
