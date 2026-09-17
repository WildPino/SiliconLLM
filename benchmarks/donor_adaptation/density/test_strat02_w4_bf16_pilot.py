"""Synthetic-only regression tests for the bounded STRAT-02 W4 BF16 v2 pilot."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from benchmarks.donor_adaptation.density import strat02_w4_bf16_codec as codec
from benchmarks.donor_adaptation.density import strat02_w4_bf16_pilot as pilot


class W4BF16PilotSyntheticTest(unittest.TestCase):
    def test_tiled_bytes_decode_only_from_blob_and_operator_uses_frozen_tolerance(self) -> None:
        self.assertEqual((pilot.LINEAR_ATOL, pilot.LINEAR_RTOL), (1e-5, 1e-6))
        rng = np.random.default_rng(20260917)
        source = rng.normal(size=(9, 512)).astype(np.float32)
        source[0] = 0
        with tempfile.TemporaryDirectory(prefix="strat02-w4-bf16-v2-test-") as temporary:
            path = Path(temporary) / "pilot.bin"
            pilot._write_tile_blob(path, source, codec, max_rows=3, max_groups=2)
            report = pilot._assert_packed_roundtrip(path.read_bytes(), source, codec)
        self.assertTrue(report["decoded_f32_exact"])
        self.assertTrue(report["synthetic_linear_parity"])

    def test_frozen_keys_include_the_preregistered_five_matrices(self) -> None:
        self.assertEqual(pilot.FROZEN_KEYS, {
            "attention": "model.layers.0.self_attn.k_proj.weight",
            "router": "model.layers.0.mlp.gate.weight",
            "expert_down": "model.layers.0.mlp.experts.0.down_proj.weight",
            "expert_gate": "model.layers.0.mlp.experts.0.gate_proj.weight",
            "lm_head": "lm_head.weight",
        })

    def test_invalid_nonzero_bf16_scale_is_void_format_and_is_preserved(self) -> None:
        self.assertEqual(pilot._worker_failure_status(pilot.FormatError("invalid BF16 scale")), "VOID_FORMAT")
        self.assertEqual(pilot._worker_failure_status(pilot.GateError("byte mismatch")), "VOID_APPARATUS")
        self.assertEqual(pilot._worker_failure_status(ValueError("unexpected")), "INCOMPLETE")

    def test_reference_or_decoder_disagreement_is_apparatus_not_format(self) -> None:
        source = np.ones((1, 128), dtype=np.float32)
        blob = codec.w4_bf16_to_blob(codec.encode_w4_bf16(source))
        with mock.patch.object(codec, "iter_encode_w4_bf16_rows", side_effect=ValueError("scalar reject")):
            with self.assertRaisesRegex(pilot.GateError, "scalar BF16 reference"):
                pilot._assert_packed_roundtrip(blob, source, codec)
        with mock.patch.object(codec, "dequantize_w4_bf16", side_effect=ValueError("decoder reject")):
            with self.assertRaisesRegex(pilot.GateError, "blob decoder"):
                pilot._assert_packed_roundtrip(blob, source, codec)

    def test_output_directory_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-w4-bf16-v2-test-") as temporary:
            with self.assertRaises(pilot.GateError):
                pilot._prepare_output(Path(temporary))

    def test_supervisor_terminates_only_its_child_for_resource_caps(self) -> None:
        class FakeChild:
            pid = 12345
            def __init__(self) -> None: self.terminated = False
            def poll(self): return None
            def terminate(self): self.terminated = True
            def wait(self, timeout): return 1
        class FakeResource:
            def __init__(self, private: int) -> None: self.private = private
            def exe(self): return sys.executable
            def memory_info(self): return SimpleNamespace(rss=1024)
            def memory_full_info(self): return SimpleNamespace(private=self.private)
        class FakePsutil:
            def __init__(self, available: int, private: int) -> None: self.available, self.resource = available, FakeResource(private)
            def Process(self, pid): self.asserted_pid = pid; return self.resource
            def virtual_memory(self): return SimpleNamespace(available=self.available)
        for available, private, reason in ((pilot.MIN_RUNTIME_RAM_BYTES - 1, 0, "available_physical_ram_below_4_gib"),
                                           (pilot.MIN_RUNTIME_RAM_BYTES, pilot.MAX_PRIVATE_COMMIT_BYTES + 1, "child_private_commit_above_8_gib")):
            with self.subTest(reason=reason):
                child, psutil = FakeChild(), FakePsutil(available, private)
                outcome = pilot._monitor_child(child, psutil=psutil, on_sample=lambda _sample: None,
                                                expected_executable=Path(sys.executable), now=lambda: 0.0)
                self.assertEqual((outcome.status, outcome.reason), ("VOID_RESOURCE", reason))
                self.assertTrue(child.terminated)
                self.assertEqual(psutil.asserted_pid, child.pid)


if __name__ == "__main__":
    unittest.main()
