import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_layer1_ffn_output_component_cross_input as r


class TestLayer1FfnOutputComponentCrossInput(unittest.TestCase):
    def test_protocol(self):
        self.assertTrue(r.PROTOCOL.is_file())
        self.assertIn("LAYER1_ROUTED_MOE_OUTPUT_RESIDUAL_SUFFICIENT", r.PROTOCOL.read_text(encoding="utf-8"))

    def test_sources(self):
        self.assertTrue(all(item["sha256"] for item in r.sources().values()))

    def test_engine_wiring(self):
        text = r.ENGINE.read_text(encoding="utf-8")
        self.assertIn('#include "strat01_gguf_layer1_ffn_output_component_cross_input.h"', text)
        self.assertIn("--strat01-layer1-ffn-output-component-cross-input-selftest", text)

    def test_inputs_and_twins(self):
        self.assertEqual(set(r.PINNED), set(r.TWINS))
        self.assertEqual(r.PINNED["ref_moe_out"][1], "9d09e5582483a471ec57712ad42b5b868f2ee1715f186e1324f9904ebebb08c8")
        self.assertEqual(r.PINNED["c_shared_out"][1], "2fc5d9d6248452269bb6850571559a246bd9703ab984210cffdf0f15e92bbe10")

    def test_predecessor(self):
        self.assertEqual(r.PREDECESSOR_SHA, "9bdda63cabb7bfe00d7e9dbf76cbead7216aa03064e816cdfa62748fe314ff98")
        self.assertEqual(r.PREDECESSOR_OUTPUT_SHA, "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254")

    def test_classify(self):
        passed, failed = {"pass": True}, {"pass": False}
        judgments = {r.ARM_NAMES[0]: passed, r.ARM_NAMES[1]: failed, r.ARM_NAMES[3]: failed, r.ARM_NAMES[4]: failed, r.ARM_NAMES[5]: passed}
        self.assertEqual(r.classify(judgments), "LAYER1_ROUTED_MOE_OUTPUT_RESIDUAL_SUFFICIENT")
        judgments[r.ARM_NAMES[4]], judgments[r.ARM_NAMES[5]] = passed, failed
        self.assertEqual(r.classify(judgments), "LAYER1_SHARED_EXPERT_OUTPUT_RESIDUAL_SUFFICIENT")
        judgments[r.ARM_NAMES[4]] = failed
        self.assertEqual(r.classify(judgments), "LAYER1_FFN_OUTPUT_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT")
        judgments[r.ARM_NAMES[4]], judgments[r.ARM_NAMES[5]] = passed, passed
        self.assertEqual(r.classify(judgments), "LAYER1_FFN_OUTPUT_COMPONENT_RESIDUALS_JOINTLY_SUFFICIENT")

    def test_frozen_candidate_hashes(self):
        self.assertEqual(r.EXPECTED_CANDIDATE_SHA[r.ARM_NAMES[2]], r.PINNED["ref_ffn_out"][1])
        self.assertEqual(r.EXPECTED_CANDIDATE_SHA[r.ARM_NAMES[3]], r.PINNED["c_ffn_out"][1])
        self.assertEqual(len(r.EXPECTED_CANDIDATE_SHA[r.ARM_NAMES[4]]), 64)
        self.assertEqual(len(r.EXPECTED_CANDIDATE_SHA[r.ARM_NAMES[5]]), 64)

    def test_arm_manifest_rejects_label_swap(self):
        outputs = {}
        for name in r.ARM_NAMES:
            kind, moe, shared, control = r.ARM_META[name]
            payload = {key: {"path": key, "bytes": size, "sha256": "0" * 64} for key, size in (("ffn_out", 49152), ("l_out", 49152), ("norm", 49152), ("projection", 18432), ("prefix", 16384), ("downstream", 196608))}
            outputs[name] = {"kind": kind, "moe_origin": moe, "shared_origin": shared, "moe_control": control, **payload}
        r.arm_manifest(outputs)
        items = list(outputs.items()); items[4], items[5] = items[5], items[4]
        with self.assertRaises(r.RunnerError):
            r.arm_manifest(dict(items))

    def test_descriptive_shape(self):
        values = np.ones(12288, dtype=np.float32)
        self.assertEqual(len(r.descriptive(values, values, (8, 1536))["per_token"]), 8)


if __name__ == "__main__":
    unittest.main()
