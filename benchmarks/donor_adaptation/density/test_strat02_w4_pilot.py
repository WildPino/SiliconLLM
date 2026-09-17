"""Synthetic-only regression tests for the bounded STRAT-02 W4 pilot apparatus."""

from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.donor_adaptation.density import strat02_w4_pilot as pilot
from benchmarks.donor_adaptation.density import strat02_weight_codec as codec


class W4PilotSyntheticTest(unittest.TestCase):
    def test_tiled_blob_is_rowwise_exact_and_decodes_from_blob(self) -> None:
        rng = np.random.default_rng(20260917)
        source = rng.normal(size=(9, 512)).astype(np.float32)
        source[0] = 0
        with tempfile.TemporaryDirectory(prefix="strat02-w4-pilot-test-") as temporary:
            path = Path(temporary) / "pilot.bin"
            pilot._write_tile_blob(path, source, codec, max_rows=3, max_groups=2)
            report = pilot._assert_packed_roundtrip(path.read_bytes(), source, codec)
        self.assertTrue(report["decoded_f32_exact"])
        self.assertTrue(report["synthetic_linear_parity"])

    def test_frozen_pilot_keys_are_exact(self) -> None:
        self.assertEqual(pilot.FROZEN_KEYS, {
            "attention": "model.layers.0.self_attn.k_proj.weight",
            "router": "model.layers.0.mlp.gate.weight",
            "expert": "model.layers.0.mlp.experts.0.down_proj.weight",
            "lm_head": "lm_head.weight",
        })

    def test_output_directory_must_be_fresh(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strat02-w4-pilot-test-") as temporary:
            with self.assertRaises(pilot.GateError):
                pilot._prepare_output(Path(temporary))

    def test_supervisor_terminates_only_its_child_at_pilot_caps(self) -> None:
        class FakeChild:
            pid = 12345

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

            def Process(self, pid):
                self.asserted_pid = pid
                return self.resource

            def virtual_memory(self):
                return SimpleNamespace(available=self.available)

        for available, private, expected_reason in (
            (pilot.MIN_RUNTIME_RAM_BYTES - 1, 0, "available_physical_ram_below_4_gib"),
            (pilot.MIN_RUNTIME_RAM_BYTES, pilot.MAX_PRIVATE_COMMIT_BYTES + 1,
             "child_private_commit_above_8_gib"),
        ):
            with self.subTest(expected_reason=expected_reason):
                child = FakeChild()
                psutil = FakePsutil(available, private)
                outcome = pilot._monitor_child(
                    child, psutil=psutil, on_sample=lambda _sample: None,
                    expected_executable=Path(sys.executable), now=lambda: 0.0,
                )
                self.assertEqual(outcome.status, "VOID_RESOURCE")
                self.assertEqual(outcome.reason, expected_reason)
                self.assertTrue(child.terminated)
                self.assertEqual(psutil.asserted_pid, child.pid)


if __name__ == "__main__":
    unittest.main()
