"""Synthetic-only tests for the STRAT-02 W4-v1 format census apparatus."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.donor_adaptation.density import strat02_w4_format_census as census
from benchmarks.donor_adaptation.density import strat02_weight_codec as codec


class W4FormatCensusSyntheticTest(unittest.TestCase):
    def test_endpoint_shortcut_matches_all_eleven_candidates(self) -> None:
        census._validate_endpoint_shortcut(codec)
        groups = []
        zero = np.zeros(census.GROUP_SIZE, dtype=np.float32)
        groups.append((zero, None))
        tiny = zero.copy()
        tiny[0] = np.nextafter(np.float32(0), np.float32(1))
        groups.append((tiny, "scale_underflow_c_0.50"))
        normal = zero.copy()
        normal[:3] = (np.float32(-1), np.float32(0.25), np.float32(1))
        groups.append((normal, None))
        huge = zero.copy()
        huge[0] = np.float32(np.finfo(np.float32).max)
        groups.append((huge, "scale_nonfinite_c_1.00"))
        nan = normal.copy()
        nan[7] = np.float32(np.nan)
        groups.append((nan, "source_nonfinite"))
        for group, expected in groups:
            with self.subTest(expected=expected):
                observed = census._classify_group(group)
                self.assertEqual(observed, expected)
                self.assertEqual(observed is not None, census._all_candidate_invalid(group, codec.W4_CANDIDATE_C))

    def test_tile_counts_and_invalid_coordinate_reasons(self) -> None:
        tile = np.zeros((2, 3, census.GROUP_SIZE), dtype=np.float32)
        tile[0, 1, 0] = np.nextafter(np.float32(0), np.float32(1))
        tile[0, 2, 0] = np.float32(np.finfo(np.float32).max)
        tile[1, 0, 5] = np.float32(np.inf)
        finite, zero, underflow, overflow = census._classify_tile(tile)
        self.assertEqual(int(np.count_nonzero(zero)), 3)
        self.assertEqual(int(np.count_nonzero(underflow)), 1)
        self.assertEqual(int(np.count_nonzero(overflow)), 1)
        self.assertEqual(int(np.count_nonzero(~finite)), 1)
        with tempfile.TemporaryDirectory(prefix="strat02-census-test-") as temporary:
            path = Path(temporary) / "invalid.jsonl"
            with census._open_append_only(path) as handle:
                used = census._emit_invalids(handle, key="synthetic.weight", row_offset=17, finite=finite,
                                              underflow=underflow, overflow=overflow, current_bytes=0)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(used, sum(len((json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")) for record in records))
        self.assertEqual({(row["row"], row["group"], row["reason"]) for row in records}, {
            (17, 1, "scale_underflow_c_0.50"), (17, 2, "scale_nonfinite_c_1.00"),
            (18, 0, "source_nonfinite"),
        })

    def test_fresh_directory_and_output_cap_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-census-test-") as temporary:
            with self.assertRaises(census.GateError):
                census._prepare_output(Path(temporary))
            path = Path(temporary) / "cap.jsonl"
            finite = np.zeros((1, 1), dtype=bool)
            underflow = np.zeros((1, 1), dtype=bool)
            overflow = np.zeros((1, 1), dtype=bool)
            original = census.MAX_INVALID_COORDINATE_BYTES
            try:
                census.MAX_INVALID_COORDINATE_BYTES = 1
                with census._open_append_only(path) as handle:
                    with self.assertRaises(census.GateError):
                        census._emit_invalids(handle, key="x", row_offset=0, finite=finite, underflow=underflow,
                                              overflow=overflow, current_bytes=0)
            finally:
                census.MAX_INVALID_COORDINATE_BYTES = original
            self.assertEqual(path.read_bytes(), b"")

    def test_positive_control_is_counted_without_exemption(self) -> None:
        finite = np.ones((2, 16), dtype=bool)
        underflow = np.zeros((2, 16), dtype=bool)
        overflow = np.zeros((2, 16), dtype=bool)
        underflow[1] = True
        key = "model.layers.3.mlp.gate.weight"
        report = census._positive_control_counts(key, 126, finite, underflow, overflow)
        self.assertEqual(report, {"groups": 16, "underflow": 16, "other_invalid": 0})
        self.assertEqual(census._positive_control_counts("lm_head.weight", 126, finite, underflow, overflow),
                         {"groups": 0, "underflow": 0, "other_invalid": 0})

    def test_supervisor_stops_only_the_created_child_at_census_caps(self) -> None:
        class FakeChild:
            pid = 43210

            def __init__(self) -> None:
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout):
                return 1

        class FakeResource:
            def __init__(self, private: int) -> None:
                self.private = private

            def exe(self):
                return sys.executable

            def memory_info(self):
                return SimpleNamespace(rss=1024)

            def memory_full_info(self):
                return SimpleNamespace(private=self.private)

        class FakePsutil:
            def __init__(self, available: int, private: int) -> None:
                self.available = available
                self.resource = FakeResource(private)
                self.observed_pid = None

            def Process(self, pid):
                self.observed_pid = pid
                return self.resource

            def virtual_memory(self):
                return SimpleNamespace(available=self.available)

        for available, private, reason in (
            (census.MIN_RUNTIME_RAM_BYTES - 1, 0, "available_physical_ram_below_4_gib"),
            (census.MIN_RUNTIME_RAM_BYTES, census.MAX_PRIVATE_COMMIT_BYTES + 1,
             "child_private_commit_above_8_gib"),
        ):
            with self.subTest(reason=reason):
                child, psutil = FakeChild(), FakePsutil(available, private)
                outcome = census._monitor_child(child, psutil=psutil, on_sample=lambda _sample: None,
                                                 expected_executable=Path(sys.executable), now=lambda: 0.0)
                self.assertEqual(outcome.status, "VOID_RESOURCE")
                self.assertEqual(outcome.reason, reason)
                self.assertTrue(child.terminated)
                self.assertEqual(psutil.observed_pid, child.pid)


if __name__ == "__main__":
    unittest.main()
