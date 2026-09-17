"""Synthetic-only tests for candidate byte decoding and meta-keyset binding."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from benchmarks.donor_adaptation.density import strat02_w4_bf16_candidate_loader as loader
from benchmarks.donor_adaptation.density import strat02_w4_bf16_codec as codec
from benchmarks.donor_adaptation.density import strat02_w4_bf16_export_verify as verifier


class CandidateDecoderSyntheticTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260917)
        # More than 16 rows and 16 groups exercises both tile boundaries.
        self.weights = rng.normal(size=(17, 17 * 128)).astype(np.float32)
        self.encoded = codec.encode_w4_bf16(self.weights)
        self.blob = codec.w4_bf16_to_blob(self.encoded)
        self.prefix = b"prefix!"
        self.linear = {"name": "linear.weight", "source_shard": "synthetic", "shape": [17, 17 * 128],
                       "source_dtype": "float32", "encoding": verifier.LINEAR_ENCODING,
                       "offset": len(self.prefix), "length": len(self.blob)}

    def test_scale_then_codes_exact_decode_across_tiles_and_offset(self) -> None:
        raw = self.prefix + self.blob + b"tail"
        decoded = loader.decode_record(io.BytesIO(raw), self.linear).numpy()
        expected = codec.dequantize_w4_bf16(self.encoded)
        np.testing.assert_array_equal(decoded.view("<u4"), expected.view("<u4"))
        self.assertEqual(decoded.shape, self.weights.shape)

    def test_f32_passthrough_preserves_signed_zero_and_nan_payload(self) -> None:
        bits = np.array([0x80000000, 0x00000000, 0x7FC01234, 0x3F800000], dtype="<u4")
        record = {"shape": [2, 2], "encoding": verifier.F32_ENCODING, "offset": 3, "length": 16}
        value = loader.decode_record(io.BytesIO(b"abc" + bits.tobytes() + b"tail"), record)
        np.testing.assert_array_equal(value.numpy().view("<u4").reshape(-1), bits)

    def test_zero_negative_nonfinite_scales_and_reserved_code_fail(self) -> None:
        for scale_bytes in (b"\x00\x00", b"\x80\xbf", b"\x80\x7f"):
            damaged = bytearray(self.blob)
            damaged[:2] = scale_bytes
            with self.subTest(scale=scale_bytes), self.assertRaises(loader.CandidateIntegrityError):
                loader.decode_record(io.BytesIO(self.prefix + damaged), self.linear)
        damaged = bytearray(self.blob)
        scale_region = 17 * 17 * 2
        damaged[scale_region] = 0x08
        with self.assertRaisesRegex(loader.CandidateIntegrityError, "reserved W4"):
            loader.decode_record(io.BytesIO(self.prefix + damaged), self.linear)

    def test_truncated_record_and_wrong_boundary_fail(self) -> None:
        with self.assertRaises(loader.CandidateIntegrityError):
            loader.decode_record(io.BytesIO(self.prefix + self.blob[:-1]), self.linear)
        with self.assertRaises(loader.CandidateIntegrityError):
            loader.decode_record(io.BytesIO(self.prefix + self.blob), {**self.linear, "length": len(self.blob) - 1})


class CandidateBindingSyntheticTest(unittest.TestCase):
    def setUp(self) -> None:
        with torch.device("meta"):
            self.model = torch.nn.Linear(128, 2)
        self.plan = {"synthetic": [
            {"name": "weight", "source_shard": "synthetic", "shape": [2, 128], "source_dtype": "float32",
             "encoding": verifier.LINEAR_ENCODING, "offset": 0, "length": 132},
            {"name": "bias", "source_shard": "synthetic", "shape": [2], "source_dtype": "float32",
             "encoding": verifier.F32_ENCODING, "offset": 132, "length": 8},
        ]}
        self.index = {"weight": "synthetic", "bias": "synthetic"}

    def test_exact_meta_keyset_and_structure_binds(self) -> None:
        # nn.Linear at the root has module name '', so use a named child.
        with torch.device("meta"):
            model = torch.nn.Sequential(torch.nn.Linear(128, 2))
        plan = {"synthetic": [{**item, "name": "0." + item["name"]} for item in self.plan["synthetic"]]}
        loader.bind_plan_to_meta(model, plan, {"0.weight": "synthetic", "0.bias": "synthetic"})

    def test_missing_extra_shape_and_encoding_fail_before_bytes(self) -> None:
        with torch.device("meta"):
            model = torch.nn.Sequential(torch.nn.Linear(128, 2))
        base = [{**item, "name": "0." + item["name"]} for item in self.plan["synthetic"]]
        index = {"0.weight": "synthetic", "0.bias": "synthetic"}
        cases = [base[:1], base + [{**base[1], "name": "extra"}],
                 [{**base[0], "shape": [1, 256]}, base[1]],
                 [{**base[0], "encoding": verifier.F32_ENCODING}, base[1]]]
        for entries in cases:
            with self.subTest(entries=entries), self.assertRaises(loader.CandidateIntegrityError):
                loader.bind_plan_to_meta(model, {"synthetic": entries}, index)

    def test_assign_uses_decoded_pointers_and_linear_operator(self) -> None:
        values = np.linspace(-2.0, 2.0, 256, dtype=np.float32).reshape(2, 128)
        blob = codec.w4_bf16_to_blob(codec.encode_w4_bf16(values))
        record = {"shape": [2, 128], "encoding": verifier.LINEAR_ENCODING, "offset": 0, "length": len(blob)}
        weight = loader.decode_record(io.BytesIO(blob), record)
        bias = loader.decode_record(io.BytesIO(np.array([0x80000000, 0x3F800000], dtype="<u4").tobytes()),
                                    {"shape": [2], "encoding": verifier.F32_ENCODING, "offset": 0, "length": 8})
        with torch.device("meta"):
            model = torch.nn.Linear(128, 2)
        state = {"weight": weight, "bias": bias}
        model.load_state_dict(state, strict=True, assign=True)
        loader._assert_no_meta_and_shared(model, state)
        x = torch.ones((1, 128), dtype=torch.float32)
        torch.testing.assert_close(model(x), torch.nn.functional.linear(x, weight, bias), atol=0, rtol=0)
        self.assertEqual(bias.numpy().view("<u4")[0], 0x80000000)

    def test_one_arena_multi_record_exact_storage_and_values(self) -> None:
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 2)
                self.register_buffer("marker", torch.empty(()))

        with torch.device("meta"):
            model = Small()
        weights = np.linspace(-2.0, 2.0, 256, dtype=np.float32).reshape(2, 128)
        encoded = codec.encode_w4_bf16(weights)
        blob = codec.w4_bf16_to_blob(encoded)
        bias_bits = np.array([0x80000000, 0x3F800000], dtype="<u4")
        marker_bits = np.array([0x7FC01234], dtype="<u4")
        records = [
            {"name": "linear.weight", "source_shard": "synthetic", "shape": [2, 128],
             "encoding": verifier.LINEAR_ENCODING, "offset": 0, "length": len(blob)},
            {"name": "linear.bias", "source_shard": "synthetic", "shape": [2],
             "encoding": verifier.F32_ENCODING, "offset": len(blob), "length": 8},
            {"name": "marker", "source_shard": "synthetic", "shape": [],
             "encoding": verifier.F32_ENCODING, "offset": len(blob) + 8, "length": 4},
        ]
        raw = blob + bias_bits.tobytes() + marker_bits.tobytes()
        with mock.patch.object(torch, "empty", wraps=torch.empty) as allocation:
            arena, state = loader.make_arena_views({"synthetic": records}, ["synthetic"],
                                                   {"linear_params": 256, "other_params": 3})
            handle = io.BytesIO(raw)
            for record in records:
                self.assertIs(loader.decode_record(handle, record, destination=state[record["name"]]), state[record["name"]])
            self.assertEqual(allocation.call_count, 1)
        self.assertEqual(arena.numel(), 259)
        self.assertEqual({value.untyped_storage().data_ptr() for value in state.values()}, {arena.untyped_storage().data_ptr()})
        np.testing.assert_array_equal(state["linear.weight"].numpy().view("<u4"),
                                      codec.dequantize_w4_bf16(encoded).view("<u4"))
        np.testing.assert_array_equal(state["linear.bias"].numpy().view("<u4"), bias_bits)
        np.testing.assert_array_equal(state["marker"].numpy().view("<u4"), marker_bits[0])
        model.load_state_dict(state, strict=True, assign=True)
        loader._assert_no_meta_and_shared(model, state, arena, {"synthetic": records}, ["synthetic"])

    def test_destination_and_arena_ledger_guards(self) -> None:
        values = np.ones((2, 128), dtype=np.float32)
        blob = codec.w4_bf16_to_blob(codec.encode_w4_bf16(values))
        record = {"name": "weight", "source_shard": "synthetic", "shape": [2, 128],
                  "encoding": verifier.LINEAR_ENCODING, "offset": 0, "length": len(blob)}
        for destination in (torch.empty((1, 256), dtype=torch.float32),
                            torch.empty((2, 128), dtype=torch.float64),
                            torch.empty((2, 128), device="meta")):
            with self.subTest(destination=destination), self.assertRaises(loader.CandidateIntegrityError):
                loader.decode_record(io.BytesIO(blob), record, destination=destination)
        with mock.patch.object(torch, "empty", wraps=torch.empty) as allocation:
            with self.assertRaises(loader.CandidateIntegrityError):
                loader.make_arena_views({"synthetic": [record]}, ["synthetic"],
                                        {"linear_params": 255, "other_params": 0})
            self.assertEqual(allocation.call_count, 0)

    def test_loader_resource_cap_uses_runner_resource_class(self) -> None:
        self.assertTrue(issubclass(loader.CandidateResourceError, MemoryError))
        record = {"name": "bias", "source_shard": "synthetic", "shape": [2],
                  "encoding": verifier.F32_ENCODING, "offset": 0, "length": 8}
        with mock.patch.object(torch, "empty", side_effect=RuntimeError("synthetic allocator failure")):
            with self.assertRaises(loader.CandidateResourceError):
                loader.make_arena_views({"synthetic": [record]}, ["synthetic"],
                                        {"linear_params": 0, "other_params": 2})

    def test_bound_control_hashes_and_directory_reject_substitution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-candidate-binding-") as temporary:
            root = Path(temporary)
            (root / "artifact_manifest.json").write_bytes(b"artifact")
            (root / "plan_manifest.json").write_bytes(b"plan")
            (root / "supervisor_result.json").write_text(
                '{"ok":true,"status":"COMPLETE","verdict":"EXPORT_VERIFIED"}', encoding="utf-8")
            expected = {name: loader._sha_file(root / name) for name in loader.BOUND_CONTROL_SHA256}
            with mock.patch.object(loader, "BOUND_EXPORT", root), mock.patch.object(loader, "BOUND_CONTROL_SHA256", expected):
                self.assertEqual(loader.check_bound_controls(root), expected)
                other = root / "other"
                other.mkdir()
                with self.assertRaises(loader.CandidateIntegrityError):
                    loader.check_bound_controls(other)
                (root / "plan_manifest.json").write_bytes(b"different plan")
                with self.assertRaisesRegex(loader.CandidateIntegrityError, "plan_manifest.json"):
                    loader.check_bound_controls(root)

    def test_quality_prerequisite_hash_and_status_controls(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-quality-prerequisite-") as temporary:
            root = Path(temporary)
            scores = root / "teacher_scores.jsonl"
            repeat = root / "supervisor_result.json"
            scores.write_bytes(b"synthetic-only")
            repeat.write_text('{"status":"PASS_REPEATABILITY"}', encoding="utf-8")
            expected = {scores: loader._sha_file(scores), repeat: loader._sha_file(repeat)}
            with mock.patch.object(loader, "QUALITY_PREREQUISITE_SHA256", expected):
                loader.check_quality_prerequisites()
                scores.write_bytes(b"changed")
                with self.assertRaises(loader.CandidateIntegrityError):
                    loader.check_quality_prerequisites()
                scores.write_bytes(b"synthetic-only")
                repeat.write_text('{"status":"INCOMPLETE"}', encoding="utf-8")
                expected[repeat] = loader._sha_file(repeat)
                with self.assertRaisesRegex(loader.CandidateIntegrityError, "did not pass"):
                    loader.check_quality_prerequisites()


if __name__ == "__main__":
    unittest.main()
