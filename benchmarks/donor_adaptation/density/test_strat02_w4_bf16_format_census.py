"""Synthetic-only checks for the BF16 v2 count-only census apparatus."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.donor_adaptation.density import strat02_w4_bf16_codec as codec
from benchmarks.donor_adaptation.density import strat02_w4_bf16_format_census as census


class BF16FullFormatCensusSyntheticTest(unittest.TestCase):
    def test_snapshot_identity_uses_filesystem_not_path_spelling(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-bf16-snapshot-test-") as temporary:
            root = Path(temporary)
            child = root / "child"
            child.mkdir()
            self.assertNotEqual(str(root), str(child / ".."))
            self.assertTrue(census._same_snapshot_directory(root, child / ".."))
            self.assertFalse(census._same_snapshot_directory(root, child))

    def test_endpoint_shortcut_matches_all_eleven_on_planted_and_log_uniform_cases(self) -> None:
        census._validate_endpoint_shortcut(codec)
        self.assertEqual(len(codec.W4_CANDIDATE_C), 11)
        source = np.zeros((2, 3, 128), dtype=np.float32)
        source[0, 1, 0] = np.nextafter(np.float32(0), np.float32(1))
        source[0, 2, 0] = np.finfo(np.float32).max
        source[1, 0, 3] = np.nan
        classified = census._classify_tile(source, codec)
        self.assertEqual(census._first_problem(classified), (0, 1, "scale_underflow"))
        self.assertEqual(int(np.count_nonzero(classified["zero"])), 3)
        self.assertTrue(classified["nonzero"][0, 2])
        self.assertFalse(classified["overflow"][0, 2])
        self.assertEqual(census._all_candidate_reason(source[0, 1], codec), "scale_underflow")
        self.assertEqual(census._all_candidate_reason(source[0, 2], codec), None)
        self.assertEqual(census._all_candidate_reason(source[1, 0], codec), "source_nonfinite")
        self.assertEqual(census._json_f32(np.float32(np.inf)), "Infinity")
        self.assertEqual(census._json_f32(np.float32(-np.inf)), "-Infinity")
        source[0, 1, 0] = np.float32(1.0)
        classified = census._classify_tile(source, codec)
        self.assertEqual(census._first_problem(classified), (1, 0, "source_nonfinite"))

    def test_first_invalid_is_confirmed_by_full_scalar_codec(self) -> None:
        tile = np.zeros((1, 2, 128), dtype=np.float32)
        tile[0, 1, 0] = np.nextafter(np.float32(0), np.float32(1))
        classified = census._classify_tile(tile, codec)
        record = census._confirm_problem(tile, classified, (0, 1, "scale_underflow"),
                                         key="synthetic.weight", shard="synthetic-shard", row_offset=9, codec=codec)
        self.assertEqual((record["row"], record["group"], record["classification"]), (9, 1, "scale_underflow"))
        self.assertEqual(record["scalar_codec_confirmation"], "invalid BF16 candidate scale")

    def test_count_cap_and_write_once_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-bf16-census-test-") as temporary:
            root = Path(temporary)
            with self.assertRaises(census.GateError):
                census._prepare_output(root)
            path = root / "counts.jsonl"
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                with self.assertRaises(census.OutputCapError):
                    census._append_count(handle, {"tensor": "synthetic"}, 0, 1)
            self.assertEqual(path.read_bytes(), b"")

    def test_synthetic_tensor_scan_counts_without_emitting_weights(self) -> None:
        import torch

        class Slice:
            def __init__(self, value): self.value = value
            def get_shape(self): return tuple(self.value.shape)
            def __getitem__(self, rows): return self.value[rows]
        class Handle:
            def __init__(self, value): self.value = value
            def get_slice(self, key): return Slice(self.value)

        source = torch.zeros((2, 128), dtype=torch.float32)
        source[0, 0] = 1.0
        output = StringIO()
        used, record = census._scan_tensor(Handle(source), key="synthetic.weight", shape=[2, 128],
                                           shard="synthetic.safetensors", codec=codec, counts_handle=output,
                                           count_bytes=0, remaining_cap=1024)
        self.assertEqual((record["groups"], record["true_zero"], record["real_valid_scalar_checks"]), (2, 1, 1))
        self.assertEqual(used, len(output.getvalue().encode()))
        self.assertNotIn("weights", output.getvalue())

        invalid = source.clone()
        invalid[0, 0] = torch.tensor(np.nextafter(np.float32(0), np.float32(1)))
        with self.assertRaises(census.FormatCounterexample) as caught:
            census._scan_tensor(Handle(invalid), key="synthetic.weight", shape=[2, 128],
                                shard="synthetic.safetensors", codec=codec, counts_handle=StringIO(),
                                count_bytes=0, remaining_cap=1024)
        self.assertEqual(caught.exception.record["scalar_codec_confirmation"], "invalid BF16 candidate scale")
        self.assertTrue(census._valid_counterexample_record(caught.exception.record,
                        [{"key": "synthetic.weight", "shape": [2, 128]}], "synthetic.safetensors",
                        "FORMAT_INVALID_COUNTEREXAMPLE"))
        forged = {**caught.exception.record, "row": 2}
        self.assertFalse(census._valid_counterexample_record(forged,
                         [{"key": "synthetic.weight", "shape": [2, 128]}], "synthetic.safetensors",
                         "FORMAT_INVALID_COUNTEREXAMPLE"))
        nonfinite = source.clone()
        nonfinite[0, 1] = torch.tensor(float("nan"))
        with self.assertRaises(census.SourceNonfinite) as caught_source:
            census._scan_tensor(Handle(nonfinite), key="synthetic.weight", shape=[2, 128],
                                shard="synthetic.safetensors", codec=codec, counts_handle=StringIO(),
                                count_bytes=0, remaining_cap=1024)
        self.assertEqual(caught_source.exception.record["classification"], "source_nonfinite")

    def test_parent_rejects_truncated_extra_and_forged_control_counts(self) -> None:
        shard = "synthetic.safetensors"
        plan = [{"key": census.ROUTER_KEYS[0], "shape": [128, 2048]}]
        record = {"tensor": plan[0]["key"], "shard": shard, "shape": plan[0]["shape"],
                  "groups": 2048, "true_zero": 0, "source_nonfinite": 0, "scale_underflow": 0,
                  "scale_overflow_nonfinite": 0, "router_shared_nonzero_valid": 16,
                  "expert_0_3_nonzero_valid": 0, "real_valid_scalar_checks": 2}
        with tempfile.TemporaryDirectory(prefix="strat02-bf16-census-test-") as temporary:
            path = Path(temporary) / "counts.jsonl"
            raw = (json.dumps(record) + "\n").encode()
            path.write_bytes(raw)
            worker = {"shard": shard, "shard_sha256": "a" * 64, "completed_tensors": 1,
                      "counts_bytes": len(raw), "counts_sha256": hashlib.sha256(raw).hexdigest(),
                      "counts": {field: record[field] for field in census.COUNT_FIELDS},
                      "controls": {field: record[field] for field in
                                   ("router_shared_nonzero_valid", "expert_0_3_nonzero_valid", "real_valid_scalar_checks")}}
            totals, controls, size = census._validate_shard_output(path, worker, plan, shard, "a" * 64, 1024)
            self.assertEqual((totals["groups"], controls["router_shared_nonzero_valid"], size), (2048, 16, len(raw)))
            forged = {**record, "router_shared_nonzero_valid": 0}
            forged_raw = (json.dumps(forged) + "\n").encode()
            path.write_bytes(forged_raw)
            forged_worker = {**worker, "counts_bytes": len(forged_raw), "counts_sha256": hashlib.sha256(forged_raw).hexdigest(),
                             "controls": {**worker["controls"], "router_shared_nonzero_valid": 0}}
            with self.assertRaises(census.GateError):
                census._validate_shard_output(path, forged_worker, plan, shard, "a" * 64, 1024)
            path.write_bytes(raw + raw)
            extra_worker = {**worker, "counts_bytes": 2 * len(raw), "counts_sha256": hashlib.sha256(raw + raw).hexdigest()}
            with self.assertRaises(census.GateError):
                census._validate_shard_output(path, extra_worker, plan, shard, "a" * 64, 1024)

    def test_supervisor_resource_caps_monitor_only_assigned_pid(self) -> None:
        class Child:
            pid = 12345
            def __init__(self): self.terminated = False
            def poll(self): return None
            def terminate(self): self.terminated = True
            def wait(self, timeout): return 1
        class Resource:
            def __init__(self, private): self.private = private
            def exe(self): return sys.executable
            def memory_info(self): return SimpleNamespace(rss=1024)
            def memory_full_info(self): return SimpleNamespace(private=self.private)
        class Psutil:
            def __init__(self, available, private): self.available, self.resource, self.observed = available, Resource(private), None
            def Process(self, pid): self.observed = pid; return self.resource
            def virtual_memory(self): return SimpleNamespace(available=self.available)
        for available, private, reason in ((census.MIN_RUNTIME_RAM_BYTES - 1, 0, "available_physical_ram_below_4_gib"),
                                           (census.MIN_RUNTIME_RAM_BYTES, census.MAX_PRIVATE_COMMIT_BYTES + 1,
                                            "worker_private_commit_above_8_gib")):
            with self.subTest(reason=reason):
                child, psutil = Child(), Psutil(available, private)
                outcome = census._monitor_child(child, psutil=psutil, expected_executable=Path(sys.executable),
                                                on_sample=lambda _sample: None, now=lambda: 0.0)
                self.assertEqual((outcome.status, outcome.reason), ("VOID_RESOURCE", reason))
                self.assertTrue(child.terminated)
                self.assertEqual(psutil.observed, child.pid)
        child, psutil = Child(), Psutil(census.MIN_RUNTIME_RAM_BYTES, 0)
        ticks = iter((0.0, 1.0))
        elapsed = census._monitor_child(child, psutil=psutil, expected_executable=Path(sys.executable),
                                        on_sample=lambda _sample: None, now=lambda: next(ticks),
                                        wall_limit_seconds=0.5)
        self.assertEqual((elapsed.status, elapsed.reason), ("VOID_RESOURCE", "wall_clock_exceeded"))
        self.assertTrue(child.terminated)


if __name__ == "__main__":
    unittest.main()
