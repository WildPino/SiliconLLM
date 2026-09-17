"""Synthetic corruption, layout, and resource tests for the W4 BF16 v2 export."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.donor_adaptation.density import strat02_w4_bf16_codec as codec
from benchmarks.donor_adaptation.density import strat02_w4_bf16_export_verify as verifier
from benchmarks.donor_adaptation.density import strat02_w4_bf16_full_export as export


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class ExportSyntheticTest(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        self.temporary = tempfile.TemporaryDirectory(prefix="strat02-bf16-export-test-")
        self.root = Path(self.temporary.name)
        self.source_shard = "synthetic.safetensors"
        weights = np.random.default_rng(20260917).normal(size=(3, 256)).astype(np.float32)
        self.weights = torch.from_numpy(weights)
        self.other = torch.tensor([-0.0, 1.0, -2.0], dtype=torch.float32)
        self.plan = [
            {"source_shard": self.source_shard, "name": "a.weight", "shape": [3, 256], "source_dtype": "float32",
             "encoding": verifier.LINEAR_ENCODING, "offset": 0, "length": 396},
            {"source_shard": self.source_shard, "name": "b.bias", "shape": [3], "source_dtype": "float32",
             "encoding": verifier.F32_ENCODING, "offset": 396, "length": 12},
        ]

        class Slice:
            def __init__(self, tensor): self.tensor = tensor
            def get_shape(self): return tuple(self.tensor.shape)
            def __getitem__(self, index): return self.tensor[index]
        class Handle:
            def __init__(self, values): self.values = values
            def get_slice(self, key): return Slice(self.values[key])
            def get_tensor(self, key): return self.values[key]
        self.handle = Handle({"a.weight": self.weights, "b.bias": self.other})
        self.payload = self.root / "shard_01.payload"
        self.records_path = self.root / "shard_01.records.jsonl"
        with self.payload.open("x+b") as payload:
            payload.truncate(408)
            self.records = [export._write_record(payload, self.handle, item, codec, verifier)[0] for item in self.plan]
            payload.flush()
        self._save_records()
        self.descriptor = {"source_shard": self.source_shard, "source_sha256": "a" * 64,
                           "payload_file": self.payload.name, "payload_bytes": 408,
                           "payload_sha256": export._sha_file(self.payload), "records_file": self.records_path.name,
                           "records_sha256": export._sha_file(self.records_path), "record_count": 2,
                           "scalar_group_checks": 2}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _save_records(self) -> None:
        with self.records_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in self.records:
                export._append_jsonl(handle, record)

    def _refresh_hashes(self) -> None:
        self.descriptor["payload_bytes"] = self.payload.stat().st_size
        self.descriptor["payload_sha256"] = export._sha_file(self.payload)
        self.descriptor["records_sha256"] = export._sha_file(self.records_path)

    def test_exact_container_layout_and_signed_zero_passthrough(self) -> None:
        summary = verifier.verify_shard(self.root, self.descriptor, self.plan)
        self.assertEqual((summary["records"], summary["linears"], summary["groups"], summary["payload_bytes"]), (2, 1, 6, 408))
        raw = self.payload.read_bytes()
        self.assertEqual(raw[396:400], b"\x00\x00\x00\x80")
        self.assertEqual(raw[0:12], codec.encode_w4_bf16(self.weights.numpy()).scale_bits.astype("<u2").tobytes())
        self.assertEqual(raw[12:396], codec.encode_w4_bf16(self.weights.numpy()).packed_codes.tobytes())

    def test_offset_keyset_and_record_sha_corruption_fail(self) -> None:
        self.records[1]["offset"] += 1
        self._save_records(); self._refresh_hashes()
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_shard(self.root, self.descriptor, self.plan)
        self.records[1]["offset"] -= 1
        self.records[0]["sha256"] = "0" * 64
        self._save_records(); self._refresh_hashes()
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_shard(self.root, self.descriptor, self.plan)
        with self.payload.open("rb") as handle:
            self.records[0]["sha256"] = export._sha_segment(handle, 0, 396)
        self.records[1]["name"] = "extra.bias"
        self._save_records(); self._refresh_hashes()
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_shard(self.root, self.descriptor, self.plan)

    def test_reserved_nibble_and_zero_scale_corruption_fail_even_with_rehashed_records(self) -> None:
        with self.payload.open("r+b") as handle:
            handle.seek(12)
            handle.write(b"\x08")
        with self.payload.open("rb") as handle:
            self.records[0]["sha256"] = export._sha_segment(handle, 0, 396)
        self._save_records(); self._refresh_hashes()
        with self.assertRaisesRegex(verifier.VerificationError, "reserved W4"):
            verifier.verify_shard(self.root, self.descriptor, self.plan)

        with self.payload.open("r+b") as handle:
            handle.seek(12)
            handle.write(b"\x11")
            handle.seek(0)
            handle.write(b"\x00\x00")
        with self.payload.open("rb") as handle:
            self.records[0]["sha256"] = export._sha_segment(handle, 0, 396)
        self._save_records(); self._refresh_hashes()
        with self.assertRaisesRegex(verifier.VerificationError, "zero BF16 scale"):
            verifier.verify_shard(self.root, self.descriptor, self.plan)

    def test_negative_bf16_scale_fails_even_with_rehashed_records(self) -> None:
        with self.payload.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"\x80\xbf")  # -1.0 in BF16 little-endian
        with self.payload.open("rb") as handle:
            self.records[0]["sha256"] = export._sha_segment(handle, 0, 396)
        self._save_records(); self._refresh_hashes()
        with self.assertRaisesRegex(verifier.VerificationError, "negative BF16 scale"):
            verifier.verify_shard(self.root, self.descriptor, self.plan)

    def test_zero_scale_and_zero_codes_fail_even_with_rehashed_records(self) -> None:
        with self.payload.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"\x00\x00")
            handle.seek(12)
            handle.write(bytes(64))
        with self.payload.open("rb") as handle:
            self.records[0]["sha256"] = export._sha_segment(handle, 0, 396)
        self._save_records(); self._refresh_hashes()
        with self.assertRaisesRegex(verifier.VerificationError, "zero BF16 scale"):
            verifier.verify_shard(self.root, self.descriptor, self.plan)

    def test_truncated_and_extra_payload_fail(self) -> None:
        with self.payload.open("r+b") as handle:
            handle.truncate(407)
        self._refresh_hashes()
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_shard(self.root, self.descriptor, self.plan)
        with self.payload.open("ab") as handle:
            handle.write(b"\x00\x00")
        self._refresh_hashes()
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_shard(self.root, self.descriptor, self.plan)

    def test_bad_source_sha_and_format_coordinate_fail_closed(self) -> None:
        path = self.root / self.source_shard
        path.write_bytes(b"synthetic source")
        report = SimpleNamespace(snapshot=self.root)
        with self.assertRaises(export.GateError):
            export._verify_source_shard(report, {"name": path.name, "size": path.stat().st_size, "sha256": "0" * 64})
        good = {"shard": self.source_shard, "tensor": "a.weight", "row": 2, "group": 1,
                "reason": "invalid BF16 candidate scale"}
        self.assertTrue(export._valid_format_coordinate(good, self.source_shard, self.plan))
        self.assertFalse(export._valid_format_coordinate({**good, "row": 3}, self.source_shard, self.plan))

    def test_invalid_nonzero_group_keeps_format_coordinate(self) -> None:
        import torch
        tiny = np.zeros((1, 128), dtype=np.float32)
        tiny[0, 0] = np.nextafter(np.float32(0), np.float32(1))
        self.handle.values["a.weight"] = torch.from_numpy(tiny)
        plan = {**self.plan[0], "shape": [1, 128], "length": 66}
        with (self.root / "invalid.payload").open("x+b") as payload:
            payload.truncate(66)
            with self.assertRaises(export.FormatError) as raised:
                export._write_linear(payload, self.handle, plan, codec)
        self.assertEqual(raised.exception.coordinate, {"tensor": "a.weight", "shard": self.source_shard,
                                                       "row": 0, "group": 0, "reason": "invalid BF16 candidate scale"})

    def test_artifact_verifier_uses_only_exported_plan_and_payload(self) -> None:
        names = [self.source_shard] + [f"unused-{i:02d}.safetensors" for i in range(10)]
        plan_doc = {"schema": verifier.SCHEMA, "source_shard_order": names,
                    "plan": {name: self.plan if name == self.source_shard else [] for name in names},
                    "metadata": {"source_shards": [{"name": name, "sha256": "a" * 64} for name in names]}}
        plan_path = self.root / "plan_manifest.json"
        _write_json(plan_path, plan_doc)
        manifest_path = self.root / "candidate_manifest.json"
        _write_json(manifest_path, {"schema": verifier.SCHEMA, "plan_file": plan_path.name,
                                    "plan_sha256": export._sha_file(plan_path), "shards": [self.descriptor]})
        summary = verifier.verify_artifact(self.root, manifest_path, expected_plan=plan_doc["plan"], require_full=False)
        self.assertEqual(summary["payload_bytes"], 408)
        corrupted = json.loads(manifest_path.read_text(encoding="utf-8"))
        corrupted["shards"][0]["source_sha256"] = "0" * 64
        _write_json(manifest_path, corrupted)
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_artifact(self.root, manifest_path, expected_plan=plan_doc["plan"], require_full=False)

    def test_plan_organ_taxonomy_is_checked_when_frozen_ledger_present(self) -> None:
        names = [self.source_shard] + [f"unused-{i:02d}.safetensors" for i in range(10)]
        plan_doc = {"schema": verifier.SCHEMA, "source_shard_order": names,
                    "plan": {name: self.plan if name == self.source_shard else [] for name in names},
                    "metadata": {"source_shards": [{"name": name, "sha256": "a" * 64} for name in names]},
                    "expected_ledger": {}, "organ_counts": {"attention": 1}}
        plan_path = self.root / "plan_manifest.json"
        _write_json(plan_path, plan_doc)
        manifest_path = self.root / "candidate_manifest.json"
        _write_json(manifest_path, {"schema": verifier.SCHEMA, "plan_file": plan_path.name,
                                    "plan_sha256": export._sha_file(plan_path), "shards": [self.descriptor]})
        with self.assertRaisesRegex(verifier.VerificationError, "organ counts"):
            verifier.verify_artifact(self.root, manifest_path, require_full=False)


class SupervisorSyntheticTest(unittest.TestCase):
    def test_monitor_only_child_at_private_commit_cap(self) -> None:
        class Child:
            pid = 87654
            def __init__(self): self.terminated = False
            def poll(self): return None
            def terminate(self): self.terminated = True
            def wait(self, timeout): return 1
        class Process:
            def exe(self): return sys.executable
            def memory_info(self): return SimpleNamespace(rss=1024)
            def memory_full_info(self): return SimpleNamespace(private=export.MAX_PRIVATE_COMMIT + 1)
        class Psutil:
            def __init__(self): self.observed = None
            def Process(self, pid): self.observed = pid; return Process()
            def virtual_memory(self): return SimpleNamespace(available=export.MIN_RUNTIME_RAM)
        child, psutil = Child(), Psutil()
        outcome = export._monitor_child(child, psutil=psutil, expected_executable=Path(sys.executable),
                                        on_sample=lambda _sample: None, now=lambda: 0.0)
        self.assertEqual((outcome.status, outcome.reason), ("VOID_RESOURCE", "worker_private_commit_above_8_gib"))
        self.assertTrue(child.terminated)
        self.assertEqual(psutil.observed, child.pid)

    def test_worker_exit_during_resource_sample_is_not_resource_void(self) -> None:
        class Child:
            pid = 87654
            def __init__(self): self.polls = 0
            def poll(self):
                self.polls += 1
                return 0 if self.polls >= 2 else None
        class Psutil:
            def Process(self, _pid): raise ProcessLookupError("worker already exited")
        outcome = export._monitor_child(Child(), psutil=Psutil(), expected_executable=Path(sys.executable),
                                        on_sample=lambda _sample: None, now=lambda: 0.0)
        self.assertEqual((outcome.status, outcome.exit_code), ("CHILD_EXITED", 0))


if __name__ == "__main__":
    unittest.main()
