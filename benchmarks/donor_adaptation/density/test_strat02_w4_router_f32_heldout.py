"""Synthetic negative controls for the STRAT-02F router-F32 heldout runner."""
from __future__ import annotations

import hashlib
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
RUNNER_PATH = HERE / "strat02_w4_router_f32_heldout.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("strat02_w4_router_f32_heldout_under_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class FakeTokenizer:
    all_special_ids: list[int] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(1, len(text) + 1))

    def decode(self, ids: list[int], *, clean_up_tokenization_spaces: bool) -> str:
        assert clean_up_tokenization_spaces is False
        return "x" * len(ids)


def heldout_rows(count: int = 96) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for category in ("code", "prose", "technical_general"):
        for index in range(32):
            rows.append({"source_document_id": f"heldout-{category}-{index:02d}", "category": category, "text": "xxxx"})
    return rows[:count]


class RouterSelectionTests(unittest.TestCase):
    def test_wrong_router_count_is_rejected(self) -> None:
        records = tuple({"name": f"router.{index}", "organ": "router", "encoding": runner.verifier.LINEAR_ENCODING}
                        for index in range(15))
        with mock.patch.object(runner.organ, "build_arms", return_value={"ROUTER_F32": {"records": records}}):
            with self.assertRaisesRegex(runner.ApparatusError, "exactly 16"):
                runner._router_records({"synthetic": []})

    def test_wrong_sentinel_is_rejected_at_pinned_tolerance(self) -> None:
        expected = {"source_document_id": "doc", "category": "code", "tokens": 1, "bytes": 1, "bits": 2.0}
        with self.assertRaisesRegex(runner.organ.ApparatusError, "sentinel"):
            runner.organ._check_sentinel(SimpleNamespace(source_document_id="doc", category="code", tokens=1,
                                                          bytes=1, bits=2.0 + 2e-5), expected)

    def test_control_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02f-hash-") as directory:
            path = Path(directory) / "control.jsonl"
            path.write_text("synthetic", encoding="utf-8")
            with self.assertRaisesRegex(runner.ApparatusError, "SHA-256 mismatch"):
                runner._check_hash(path, "0" * 64, "synthetic")


class HeldoutWriteTests(unittest.TestCase):
    def test_corpus_audit_manifest_is_not_assumed_to_be_an_object(self) -> None:
        rows = heldout_rows()
        summary = {"heldout_tokens": runner.EXPECTED_HELDOUT_TOKENS,
                   "heldout_bytes": runner.EXPECTED_HELDOUT_BYTES}
        tokenizer = FakeTokenizer()
        with (mock.patch.object(runner.w4_quality, "_audit_full_corpus",
                                return_value=("verified-manifest-path", [], rows, summary)),
              mock.patch.object(runner.audit, "load_tokenizer",
                                return_value=(tokenizer, None, None, None)),
              mock.patch.object(runner.score_core, "validate_strat02_tokenizer")):
            observed_rows, observed_tokenizer, observed_summary = runner._read_heldout_rows(Path("synthetic"))
        self.assertIs(observed_rows, rows)
        self.assertIs(observed_tokenizer, tokenizer)
        self.assertIs(observed_summary, summary)

    def test_write_once_and_document_binding_include_document_ids(self) -> None:
        tokenizer, rows = FakeTokenizer(), heldout_rows()
        calls: list[str] = []

        def scorer(_model, _tokenizer, row, *, chunk_size):
            self.assertEqual(chunk_size, runner.CHUNK_SIZE)
            calls.append(row["source_document_id"])
            return SimpleNamespace(source_document_id=row["source_document_id"], category=row["category"],
                                   tokens=4, bytes=4, bits=8.0)

        with tempfile.TemporaryDirectory(prefix="strat02f-write-once-") as directory:
            output = Path(directory) / "candidate_scores.jsonl"
            with (mock.patch.object(runner, "EXPECTED_HELDOUT_TOKENS", 384),
                  mock.patch.object(runner, "EXPECTED_HELDOUT_BYTES", 384)):
                summary = runner._score_heldout_once(model="fake", tokenizer=tokenizer, rows=rows,
                                                      output=output, scorer=scorer)
                self.assertEqual(summary["documents"], 96)
                self.assertEqual(calls, [row["source_document_id"] for row in rows])
                first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
                self.assertIn("source_document_id", first)
                self.assertFalse({"text", "token_ids", "logits"}.intersection(first))
                with self.assertRaisesRegex(runner.ApparatusError, "write-once"):
                    runner._score_heldout_once(model="fake", tokenizer=tokenizer, rows=rows,
                                                output=output, scorer=scorer)

    def test_partial_heldout_output_is_not_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02f-partial-") as directory:
            output = Path(directory) / "candidate_scores.jsonl"
            output.write_text("{}\n" * 95, encoding="utf-8")
            with self.assertRaisesRegex(runner.ApparatusError, "incomplete"):
                runner._validate_persisted_heldout(output, heldout_rows(), FakeTokenizer())


class ProtocolStateMachineTests(unittest.TestCase):
    def test_failed_preflight_never_calls_heldout(self) -> None:
        heldout_calls = 0

        def forbidden_heldout():
            nonlocal heldout_calls
            heldout_calls += 1
            raise AssertionError("heldout scoring must be unreachable")

        with self.assertRaisesRegex(runner.ApparatusError, "preflight"):
            runner._router_protocol(
                w4_sentinel=lambda: None,
                router_sentinel=lambda: None,
                copy_router=lambda: (_ for _ in ()).throw(runner.ApparatusError("preflight copy failure")),
                rollback_w4=lambda: None,
                score_heldout=forbidden_heldout,
            )
        self.assertEqual(heldout_calls, 0)

    def test_final_rollback_failure_voids_even_after_rows(self) -> None:
        rollbacks = 0
        heldout_calls = 0

        def copy():
            return {f"router.{index}": hashlib.sha256(str(index).encode()).hexdigest()
                    for index in range(runner.ROUTER_RECORDS)}

        def rollback():
            nonlocal rollbacks
            rollbacks += 1
            if rollbacks == 2:
                raise RuntimeError("planted final rollback failure")

        def score():
            nonlocal heldout_calls
            heldout_calls += 1
            return {"documents": 96}

        with self.assertRaisesRegex(runner.ApparatusError, "final W4 rollback"):
            runner._router_protocol(w4_sentinel=lambda: None, router_sentinel=lambda: None,
                                    copy_router=copy, rollback_w4=rollback, score_heldout=score)
        self.assertEqual(heldout_calls, 1)

    def test_copy_sha_mismatch_blocks_heldout(self) -> None:
        copies = 0
        heldout_calls = 0

        def copy():
            nonlocal copies
            copies += 1
            suffix = "a" if copies == 1 else "b"
            return {f"router.{index}": suffix * 64 for index in range(runner.ROUTER_RECORDS)}

        def score():
            nonlocal heldout_calls
            heldout_calls += 1
            return {"documents": 96}

        with self.assertRaisesRegex(runner.ApparatusError, "second F32 router copy SHA"):
            runner._router_protocol(w4_sentinel=lambda: None, router_sentinel=lambda: None,
                                    copy_router=copy, rollback_w4=lambda: None, score_heldout=score)
        self.assertEqual(heldout_calls, 0)


class RunnerSurfaceTests(unittest.TestCase):
    def test_selftest_is_synthetic_and_reports_frozen_cardinalities(self) -> None:
        completed = subprocess.run([sys.executable, str(RUNNER_PATH), "--selftest"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(report, {"heldout_rows": 96, "ok": True, "router_records": 16, "selftest": True})

    def test_source_provenance_excludes_outputs_and_pins_caps(self) -> None:
        self.assertNotIn("candidate_scores", runner.PROVENANCE_SOURCES)
        self.assertEqual(runner.ROUTER_RECORDS, 16)
        self.assertEqual(runner.BOOTSTRAP_DRAWS, 20_000)
        self.assertEqual(runner.BOOTSTRAP_SEED, 20_260_916)
        self.assertEqual(runner.MIN_LAUNCH_RAM, 55 * 1024**3)
        self.assertEqual(runner.MAX_WALL_SECONDS, 6 * 60 * 60)
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("teacher.verify_shards(report)", source)
        self.assertIn("score_core.adjudicate_pinned_files", source)
        self.assertIn("final W4 rollback or sentinel failed", source)


if __name__ == "__main__":
    unittest.main()
