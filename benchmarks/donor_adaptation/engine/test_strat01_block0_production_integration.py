from __future__ import annotations

import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_block0_production_integration as runner


class Block0ProductionIntegrationTests(unittest.TestCase):
    def test_protocol_and_bindings(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION", protocol)
        self.assertIn(runner.COMBINED_ADJUDICATION_SHA, protocol)
        self.assertIn(runner.REFERENCE_RUN_MANIFEST_SHA, protocol)
        self.assertIn(runner.REFERENCE_MANIFEST_SHA, protocol)
        for digest in runner.EXPECTED_HASHES.values():
            self.assertIn(digest, protocol)
        combined = runner.validate_bindings()
        self.assertEqual(combined["status"], "COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES")

    def test_production_config_declares_changed_semantics(self) -> None:
        self.assertIn("rms_accum=double;kb=q5_0xq8_0;", runner.PRODUCTION_CONFIG)
        self.assertNotIn("rms_accum=double;kb=q5_0xq8_0;", runner.base.EXPECTED_C_CONFIG)
        header = runner.RUNG2B_HEADER.read_text(encoding="utf-8")
        self.assertIn('"rms_accum=double;kb=q5_0xq8_0;"', header)

    def test_shared_primitive_source_controls(self) -> None:
        controls = runner.source_controls()
        self.assertTrue(controls)
        self.assertTrue(all(controls.values()), controls)

    def test_intermediate_hashes_fail_closed(self) -> None:
        manifests = {}
        for arm in runner.base.ARMS:
            manifests[arm] = {"tensors": [{"name": name, "sha256": digest} for name, digest in runner.EXPECTED_HASHES.items()]}
        observed = runner.validate_intermediate_hashes({"manifests": manifests})
        self.assertEqual(observed["prefill8"], runner.EXPECTED_HASHES)
        manifests["cached7p1"]["tensors"][0]["sha256"] = "0" * 64
        with self.assertRaises(runner.IntegrationError):
            runner.validate_intermediate_hashes({"manifests": manifests})

    def test_full_checkpoint_and_continuity_card(self) -> None:
        candidate = {}
        reference = {}
        for arm in runner.base.ARMS:
            for name, shape in runner.base.SHAPES.items():
                values = np.linspace(0.25, 1.25, np.prod(shape), dtype=np.float32)
                candidate[f"{arm}/{name}"] = values.copy()
                reference[f"{arm}/{name}"] = values.copy()
            gate = np.linspace(-1.0, 1.0, np.prod(runner.base.SHAPES["ffn_gate-0"]), dtype=np.float32)
            up = np.linspace(0.5, 1.5, gate.size, dtype=np.float32)
            swiglu = (gate / (1.0 + np.exp(-gate)) * up).astype(np.float32)
            out = np.linspace(-0.5, 0.5, np.prod(runner.base.SHAPES["ffn_out-0"]), dtype=np.float32)
            l_out = out + np.float32(2.0)
            for name, values in (("ffn_gate-0", gate), ("ffn_up-0", up), ("ffn_swiglu-0", swiglu), ("ffn_out-0", out), ("l_out-0", l_out)):
                candidate[f"{arm}/{name}"] = values.copy()
                reference[f"{arm}/{name}"] = values.copy()
        result = runner.adjudicate(candidate, reference, {"source": True})
        self.assertEqual(result["status"], "PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION")
        self.assertEqual(len(result["checkpoint_results"]), 12)
        self.assertEqual(len(result["continuity_results"]), 12)


if __name__ == "__main__":
    unittest.main()

    def test_shared_primitive_source_controls(self) -> None:
        controls = runner.source_controls()
        self.assertTrue(controls)
        self.assertTrue(all(controls.values()), controls)

    def test_intermediate_hashes_fail_closed(self) -> None:
        manifests = {}
        for arm in runner.base.ARMS:
            manifests[arm] = {"tensors": [{"name": name, "sha256": digest} for name, digest in runner.EXPECTED_HASHES.items()]}
        observed = runner.validate_intermediate_hashes({"manifests": manifests})
        self.assertEqual(observed["prefill8"], runner.EXPECTED_HASHES)
        manifests["cached7p1"]["tensors"][0]["sha256"] = "0" * 64
        with self.assertRaises(runner.IntegrationError):
            runner.validate_intermediate_hashes({"manifests": manifests})

    def test_full_checkpoint_and_continuity_card(self) -> None:
        candidate = {}
        reference = {}
        for arm in runner.base.ARMS:
            for name, shape in runner.base.SHAPES.items():
                values = np.arange(np.prod(shape), dtype=np.float32) / 1000.0
                candidate[f"{arm}/{name}"] = values.copy()
                reference[f"{arm}/{name}"] = values.copy()
        result = runner.adjudicate(candidate, reference, {"source": True})
        self.assertEqual(result["status"], "PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION")
        self.assertEqual(len(result["checkpoint_results"]), 12)
        self.assertEqual(len(result["continuity_results"]), 12)


if __name__ == "__main__":
    unittest.main()
