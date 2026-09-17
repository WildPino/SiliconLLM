"""Synthetic-only STRAT-02D organ selection, mutation, and no-heldout controls."""
from __future__ import annotations

import hashlib
import io
import json
import os
import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from benchmarks.donor_adaptation.density import strat02_w4_organ_attribution as organ
from benchmarks.donor_adaptation.density import strat02_w4_bf16_candidate_loader as candidate
from benchmarks.donor_adaptation.density import strat02_w4_bf16_codec as codec
from benchmarks.donor_adaptation.density import strat02_w4_bf16_export_verify as verifier


class OrganSelectionTests(unittest.TestCase):
    def test_frozen_counts_union_and_exact_active_byte_deltas(self) -> None:
        entries = []
        for organ_name, count, shape in (("router", 16, [1024, 256]),
                                         ("lm_head", 1, [100352, 2048]),
                                         ("attention", 64, [2048, 2048]),
                                         ("expert", 6144, [1, 128])):
            for index in range(count):
                entries.append({"name": f"{organ_name}.{index:04d}.weight", "source_shard": "synthetic",
                                "shape": shape, "encoding": verifier.LINEAR_ENCODING, "organ": organ_name})
        entries.extend({"name": f"other.{index:02d}", "source_shard": "synthetic",
                        "shape": [1], "encoding": verifier.F32_ENCODING} for index in range(34))
        arms = organ.build_arms({"synthetic": entries})
        self.assertEqual(list(arms), [name for name, _ in organ.ARMS])
        self.assertEqual({name: item["count"] for name, item in arms.items()}, organ.ARM_COUNTS)
        self.assertEqual({name: item["extra_active_bytes_per_token"] for name, item in arms.items()}, organ.ARM_EXTRA_BYTES)
        self.assertEqual(arms["NONEXPERT_F32"]["active_weights"], 478_150_656)
        self.assertFalse(any(record["organ"] == "expert" for arm in arms.values() for record in arm["records"]))
        with self.assertRaises(organ.ApparatusError):
            organ.build_arms({"synthetic": entries[:-1]})


class MutationAndSentinelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.linspace(-3.0, 3.0, 256, dtype=np.float32).reshape(2, 128)
        self.encoded = codec.encode_w4_bf16(self.values)
        self.blob = codec.w4_bf16_to_blob(self.encoded)
        self.record = {"name": "router.weight", "source_shard": "synthetic", "shape": [2, 128],
                       "encoding": verifier.LINEAR_ENCODING, "organ": "router",
                       "offset": 0, "length": len(self.blob)}
        self.target = candidate.decode_record(io.BytesIO(self.blob), self.record)

    def test_f32_copy_is_bitwise_into_existing_storage(self) -> None:
        source = self.values.copy()
        source[0, 0] = np.float32(-0.0)

        class Slice:
            def get_shape(self): return source.shape
            def __getitem__(self, index): return torch.from_numpy(source[index])
        pointer = self.target.data_ptr()
        digest = organ._copy_f32_record(SimpleNamespace(get_slice=lambda key: Slice()), self.record, self.target)
        self.assertEqual(self.target.data_ptr(), pointer)
        self.assertEqual(digest, hashlib.sha256(source.tobytes()).hexdigest())
        np.testing.assert_array_equal(self.target.numpy().view("<u4"), source.view("<u4"))
        with self.assertRaises(organ.ApparatusError):
            organ._copy_f32_record(SimpleNamespace(get_slice=lambda key: Slice()), self.record,
                                   torch.empty((1, 256), dtype=torch.float32))

    def test_w4_rollback_restores_original_hash_and_detects_bad_hash(self) -> None:
        original = organ._tensor_sha(self.target)
        self.target.copy_(torch.from_numpy(self.values))
        self.assertNotEqual(organ._tensor_sha(self.target), original)
        pointer = self.target.data_ptr()
        organ._restore_w4_record(io.BytesIO(self.blob), self.record, self.target, original)
        self.assertEqual(self.target.data_ptr(), pointer)
        self.assertEqual(organ._tensor_sha(self.target), original)
        with self.assertRaisesRegex(organ.ApparatusError, "rollback SHA"):
            organ._restore_w4_record(io.BytesIO(self.blob), self.record, self.target, "0" * 64)

    def test_sentinel_tolerance_and_identity_mismatch_fail(self) -> None:
        expected = {"source_document_id": "fixed-0", "category": "code", "tokens": 8, "bytes": 12,
                    "bits": 13.0}
        observation = SimpleNamespace(source_document_id="fixed-0", category="code", tokens=8, bytes=12,
                                      bits=13.0 + 0.5e-5)
        organ._check_sentinel(observation, expected)
        with self.assertRaisesRegex(organ.ApparatusError, "sentinel"):
            organ._check_sentinel(SimpleNamespace(**{**vars(observation), "bits": 13.0 + 2e-5}), expected)
        with self.assertRaises(organ.ApparatusError):
            organ._check_sentinel(SimpleNamespace(**{**vars(observation), "tokens": 9}), expected)


class SequentialSourceHandleTests(unittest.TestCase):
    def test_keyset_audit_and_arm_copy_never_overlap_source_handles(self) -> None:
        values = {"s1": {"a": np.arange(128, dtype=np.float32).reshape(1, 128),
                         "c": np.full((1, 128), np.float32(-3), dtype=np.float32)},
                  "s2": {"b": np.full((1, 128), np.float32(7), dtype=np.float32)},
                  "s3": {"unused": np.full((1, 128), np.float32(1), dtype=np.float32)}}
        report = SimpleNamespace(snapshot=Path("synthetic"),
                                 manifest={"shards": [{"name": name} for name in values]},
                                 index={"weight_map": {key: shard for shard, tensors in values.items()
                                                       for key in tensors}})
        class Tracker:
            def __init__(self):
                self.active = self.maximum = 0
                self.opened = []
                self.slice_refs = []
            def open(self, path):
                shard = path.name
                tracker = self
                class Handle:
                    def __enter__(self):
                        tracker.active += 1
                        tracker.maximum = max(tracker.maximum, tracker.active)
                        tracker.opened.append(shard)
                        return self
                    def __exit__(self, *_args):
                        tracker.active -= 1
                    def keys(self): return list(values[shard])
                    def get_slice(self, key):
                        class Slice:
                            def get_shape(self): return values[shard][key].shape
                            def __getitem__(self, index):
                                tensor = torch.from_numpy(values[shard][key][index])
                                tracker.slice_refs.append(weakref.ref(tensor))
                                return tensor
                        return Slice()
                return Handle()
        tracker = Tracker()
        records = [{"name": key, "source_shard": shard, "shape": [1, 128],
                    "encoding": verifier.LINEAR_ENCODING} for shard, key in (("s2", "b"), ("s1", "a"), ("s1", "c"))]
        state = {record["name"]: torch.empty((1, 128), dtype=torch.float32) for record in records}
        with mock.patch.object(organ, "_source_open", side_effect=tracker.open):
            organ._audit_source_keysets(report)
            self.assertEqual(tracker.active, 0)
            organ._copy_arm_sources(records, state, report)
            self.assertEqual(tracker.active, 0)
            self.assertEqual(tracker.maximum, 1)
            self.assertEqual(tracker.opened, ["s1", "s2", "s3", "s1", "s2"])
            for record in records:
                np.testing.assert_array_equal(state[record["name"]].numpy().view("<u4"),
                                              values[record["source_shard"]][record["name"]].view("<u4"))
            gc.collect()
            self.assertTrue(all(reference() is None for reference in tracker.slice_refs))
            values["s2"].pop("b")
            with self.assertRaisesRegex(organ.ApparatusError, "keyset mismatch"):
                organ._audit_source_keysets(report)
            self.assertEqual(tracker.active, 0)
            with self.assertRaisesRegex(organ.ApparatusError, "keyset changed"):
                organ._copy_arm_sources(records, state, report)
            self.assertEqual(tracker.active, 0)
            self.assertEqual(tracker.maximum, 1)


class CalibrationOnlyTests(unittest.TestCase):
    def test_clean_shell_defaults_safe_kernel_policy_before_pinned_environment(self) -> None:
        class StopAfterEnvironment(Exception):
            pass
        def pinned_environment():
            self.assertEqual(os.environ.get("USE_HUB_KERNELS"), "NO")
            return {}
        with (mock.patch.dict(os.environ, {}, clear=True),
              mock.patch.object(organ.bounded, "_pinned_environment", side_effect=pinned_environment) as pinned,
              mock.patch.object(organ.teacher, "preflight", side_effect=StopAfterEnvironment)):
            with self.assertRaises(StopAfterEnvironment):
                organ._read_only_preflight(None, None)
            pinned.assert_called_once()
            self.assertEqual(os.environ["USE_HUB_KERNELS"], "NO")

    def test_explicit_conflicting_kernel_policy_fails_before_pinned_environment(self) -> None:
        for value in ("YES", ""):
            with (self.subTest(value=value), mock.patch.dict(os.environ, {"USE_HUB_KERNELS": value}),
                  mock.patch.object(organ.bounded, "_pinned_environment") as pinned):
                with self.assertRaisesRegex(organ.ApparatusError, "USE_HUB_KERNELS must be NO"):
                    organ._read_only_preflight(None, None)
                pinned.assert_not_called()
                self.assertEqual(os.environ["USE_HUB_KERNELS"], value)

    def test_future_output_path_uses_existing_parent_volume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02d-output-volume-") as temporary:
            root = Path(temporary)
            self.assertEqual(organ._existing_output_volume(root / "new" / "nested"), root)
            file = root / "not-a-directory"
            file.write_text("synthetic", encoding="utf-8")
            with self.assertRaises(organ.ApparatusError):
                organ._existing_output_volume(file)

    def test_w4_control_hash_is_required_before_rows_are_reused(self) -> None:
        rows = [{"source_document_id": f"doc-{index}", "category": ("code", "prose", "technical_general")[index // 16],
                 "text": "x"} for index in range(48)]
        controls = [{"source_document_id": row["source_document_id"], "category": row["category"],
                     "tokens": 1, "bytes": 1, "bits": 2.0} for row in rows]
        with tempfile.TemporaryDirectory(prefix="strat02d-w4-control-") as temporary:
            path = Path(temporary) / "calibration_checks.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in controls), encoding="utf-8")
            with (mock.patch.object(organ, "W4_CALIBRATION", path),
                  mock.patch.object(organ, "EXPECTED_CALIB_TOKENS", 48),
                  mock.patch.object(organ, "EXPECTED_CALIB_BYTES", 48)):
                with mock.patch.object(organ, "W4_CALIBRATION_SHA256", organ._sha_file(path)):
                    self.assertEqual(len(organ._baseline(rows)), 48)
                path.write_bytes(b"altered")
                with self.assertRaisesRegex(organ.ApparatusError, "SHA mismatch"):
                    organ._baseline(rows)

    def test_calibration_reader_never_requests_heldout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02d-calib-only-") as temporary:
            root = Path(temporary)
            rows = []
            for category in ("code", "prose", "technical_general"):
                for index in range(16):
                    row = {"schema": "synthetic", "split": "calib", "category": category,
                           "source_document_id": f"{category}-{index}", "text": "x", "span_byte_count": 1}
                    row["item_sha256"] = organ.audit.sha(organ.audit.canonical(row))
                    rows.append(row)
            calib = root / "calib.jsonl"
            calib.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": "synthetic", "parts": {"calib": {
                "jsonl_sha256": organ._sha_file(calib), "item_count": 48,
                "items_aggregate_sha256": organ.audit.aggregate(rows)}}}), encoding="utf-8")
            with (mock.patch.object(organ, "CALIB", calib), mock.patch.object(organ, "CORPUS_MANIFEST", manifest),
                  mock.patch.object(organ, "EXPECTED_CALIB_SHA256", organ._sha_file(calib)),
                  mock.patch.object(organ, "EXPECTED_MANIFEST_SHA256", organ._sha_file(manifest)),
                  mock.patch.object(organ, "EXPECTED_CALIB_BYTES", 48)):
                with mock.patch.object(organ.audit, "read_rows", wraps=organ.audit.read_rows) as read:
                    self.assertEqual(len(organ._read_calibration()), 48)
                    read.assert_called_once()
                    self.assertEqual(read.call_args.args[1], "calib")

    def test_arm_scores_are_calibration_only_and_write_once(self) -> None:
        rows = [{"source_document_id": f"doc-{index}", "category": ("code", "prose", "technical_general")[index // 16],
                 "text": "x"} for index in range(48)]
        class Tokenizer:
            def encode(self, text, *, add_special_tokens):
                self.last = add_special_tokens
                return [1]
            def decode(self, ids, *, clean_up_tokenization_spaces): return "x"
            all_special_ids = []
        def scorer(_model, _tokenizer, row, *, chunk_size):
            self.assertEqual(chunk_size, 128)
            return SimpleNamespace(source_document_id=row["source_document_id"], category=row["category"],
                                   tokens=1, bytes=1, bits=2.0)
        with tempfile.TemporaryDirectory(prefix="strat02d-arm-score-") as temporary:
            path = Path(temporary) / "router_f32.jsonl"
            with mock.patch.object(organ, "EXPECTED_CALIB_TOKENS", 48), mock.patch.object(organ, "EXPECTED_CALIB_BYTES", 48):
                summary = organ._score_arm(None, Tokenizer(), rows, path, scorer)
                self.assertEqual((summary["documents"], summary["bpb"]), (48, 2.0))
                self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 48)
                with self.assertRaises(FileExistsError):
                    organ._score_arm(None, Tokenizer(), rows, path, scorer)


if __name__ == "__main__":
    unittest.main()
