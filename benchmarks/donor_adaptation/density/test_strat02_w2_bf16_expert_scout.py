"""Synthetic-only protocol checks for STRAT-02E's bounded scout."""
from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from benchmarks.donor_adaptation.density import strat02_w2_bf16_expert_scout as scout


class ExpertScoutTest(unittest.TestCase):
    def test_projection_uses_minimum_throughput_and_three_hour_cap(self) -> None:
        self.assertEqual(scout._project_seconds(float(scout.EXPERT_STORED_WEIGHTS)), 1.0)
        self.assertGreater(scout._project_seconds(1.0), scout.PROJECTION_SECONDS_MAX)
        with self.assertRaises(scout.ResourceError):
            scout._project_seconds(0.0)

    def test_tile_wire_is_a_then_b_then_codes(self) -> None:
        source = np.arange(128, dtype=np.float32).reshape(1, 128)
        tile = next(scout.codec.iter_encode_w2_bf16_tiles((source,), 128, max_rows_per_tile=1, max_groups_per_tile=1))
        wire = scout._tile_wire(tile)
        self.assertEqual(len(wire), scout.codec.BYTES_PER_GROUP)
        self.assertEqual(wire[:2], tile.a_bits.astype("<u2").tobytes())
        self.assertEqual(wire[2:4], tile.b_bits.astype("<u2").tobytes())
        self.assertEqual(wire[4:], tile.packed_codes.tobytes())

    def test_worker_validation_requires_two_ordered_48_row_arms_and_tile_audit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02e-test-") as directory:
            root = Path(directory)
            arm_items = []
            for arm in scout.ARMS:
                path = root / f"{arm.lower()}.jsonl"
                path.write_text("{}\n" * 48, encoding="utf-8")
                arm_items.append({"arm": arm, "score_file": path.name, "score_sha256": scout._sha_file(path)})
            tile = root / "w2_tile_audit.jsonl"
            tile.write_text(json.dumps({"wire_sha256": "0" * 64}) + "\n", encoding="utf-8")
            worker = {"ok": True, "status": "COMPLETE_DIAGNOSTIC", "heldout_access": False,
                      "arms": arm_items, "tile_audit_sha256": scout._sha_file(tile)}
            scout._validate_worker(root, worker)
            worker["arms"] = list(reversed(arm_items))
            with self.assertRaises(scout.ApparatusError):
                scout._validate_worker(root, worker)

    def test_frozen_brief_can_be_read_without_opening_donor_values(self) -> None:
        brief = scout._frozen_brief()
        self.assertIn(b"4.194.304", brief)
        self.assertIn(b"3 ore", brief)

    def test_worker_uses_calibration_loader_without_heldout_prerequisite(self) -> None:
        worker_source = inspect.getsource(scout._worker)
        self.assertIn("strat02d._load_calibration_candidate(prepared)", worker_source)
        self.assertNotIn("candidate.load_candidate_model", worker_source)
        self.assertNotIn("check_quality_prerequisites", worker_source)

    def test_worker_failure_json_captures_format_resource_and_apparatus_errors(self) -> None:
        cases = ((scout.FormatError("bad center"), "VOID_FORMAT"),
                 (scout.ResourceError("out of RAM"), "VOID_RESOURCE"),
                 (scout.ApparatusError("bad pin"), "VOID_APPARATUS"))
        for error, status in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory(prefix="strat02e-worker-failure-") as directory:
                root = Path(directory)
                result = root / "worker_result.json"
                args = SimpleNamespace(output_dir=root, worker_result=result)
                with mock.patch.object(scout, "_require_kernel_policy", side_effect=error):
                    self.assertEqual(scout._worker(args), 1)
                payload = json.loads(result.read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], status)
                self.assertEqual(payload["error_type"], type(error).__name__)
                self.assertEqual(payload["error"], str(error))
                self.assertIn("Traceback", payload["traceback"])
                self.assertFalse(payload["heldout_access"])

    def test_paired_delta_rejects_partial_and_preserves_categories(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02e-paired-") as directory:
            root = Path(directory)
            rows, control = [], []
            for index in range(48):
                category = ("code", "prose", "technical_general")[index % 3]
                row = {"source_document_id": f"doc-{index}", "category": category, "text": "x"}
                rows.append(row)
                control.append({"source_document_id": row["source_document_id"], "category": category,
                                "tokens": 1, "bytes": 1, "bits": 1.0})
            score = root / "arm.jsonl"
            score.write_text("".join(json.dumps({**item, "bits": 1.5}) + "\n" for item in control), encoding="utf-8")
            delta = scout._paired_delta(score, control, rows)
            self.assertEqual(delta["all"]["delta_bpb"], 0.5)
            self.assertEqual(delta["code"]["delta_bpb"], 0.5)
            score.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(scout.ApparatusError):
                scout._paired_delta(score, control, rows)


if __name__ == "__main__":
    unittest.main()
