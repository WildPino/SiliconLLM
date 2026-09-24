import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_moe_component_cross_input as r


class TestLayer1RoutedMoeComponentCrossInput(unittest.TestCase):
    def test_protocol(self):
        self.assertTrue(r.PROTOCOL.is_file())
        self.assertIn("LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT", r.PROTOCOL.read_text(encoding="utf-8"))

    def test_sources(self):
        self.assertTrue(all(item["sha256"] for item in r.sources().values()))

    def test_engine_wiring(self):
        text = r.ENGINE.read_text(encoding="utf-8")
        self.assertIn('#include "strat01_gguf_layer1_routed_moe_component_cross_input.h"', text)
        self.assertIn("--strat01-layer1-routed-moe-component-cross-input-selftest", text)

    def test_inputs_and_twins(self):
        self.assertEqual(set(r.PINNED), set(r.TWINS))
        self.assertEqual(r.PINNED["ref_down"][2], 196608)
        self.assertEqual(r.PINNED["c_weights"][2], 128)
        self.assertEqual(len(r.TOPK), 4)

    def test_predecessor(self):
        self.assertEqual(r.PREDECESSOR_SHA, "4d03a081ab06cf695c4569c4d8ab615d2e2b88a2bfc520874da89e32685e984a")
        self.assertEqual(r.PREDECESSOR_OUTPUT_SHA, "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254")

    def test_classify(self):
        passed, failed = {"pass": True}, {"pass": False}
        judgments = {r.ARM_NAMES[0]: passed, r.ARM_NAMES[1]: failed, r.ARM_NAMES[3]: failed, r.ARM_NAMES[4]: failed, r.ARM_NAMES[5]: passed}
        self.assertEqual(r.classify(judgments), "LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT")
        judgments[r.ARM_NAMES[4]], judgments[r.ARM_NAMES[5]] = passed, failed
        self.assertEqual(r.classify(judgments), "LAYER1_NORMALIZED_ROUTER_WEIGHT_RESIDUAL_SUFFICIENT")
        judgments[r.ARM_NAMES[4]] = failed
        self.assertEqual(r.classify(judgments), "LAYER1_ROUTED_MOE_INPUT_RESIDUALS_INDEPENDENTLY_SUFFICIENT")
        judgments[r.ARM_NAMES[4]], judgments[r.ARM_NAMES[5]] = passed, passed
        self.assertEqual(r.classify(judgments), "LAYER1_ROUTED_MOE_INPUT_RESIDUALS_JOINTLY_SUFFICIENT")

    def test_frozen_hashes(self):
        self.assertEqual(r.EXPECTED_WEIGHTED_SHA[r.ARM_NAMES[2]], r.PINNED["ref_weighted"][1])
        self.assertEqual(r.EXPECTED_MOE_SHA[r.ARM_NAMES[3]], r.PINNED["c_moe_out"][1])
        self.assertTrue(all(len(value) == 64 for value in r.EXPECTED_WEIGHTED_SHA.values()))
        self.assertTrue(all(len(value) == 64 for value in r.EXPECTED_MOE_SHA.values()))

    def test_arm_manifest_rejects_label_swap(self):
        sizes = (("weighted", 196608), ("moe_out", 49152), ("ffn_out", 49152), ("l_out", 49152), ("norm", 49152), ("projection", 18432), ("prefix", 16384), ("downstream", 196608))
        outputs = {}
        for name in r.ARM_NAMES:
            kind, down, weights, control = r.ARM_META[name]
            payload = {key: {"path": key, "bytes": size, "sha256": "0" * 64} for key, size in sizes}
            outputs[name] = {"kind": kind, "down_origin": down, "weights_origin": weights, "down_control": control, **payload}
        r.arm_manifest(outputs); items = list(outputs.items()); items[4], items[5] = items[5], items[4]
        with self.assertRaises(r.RunnerError):
            r.arm_manifest(dict(items))

    def test_descriptive_shape(self):
        values = np.ones(49152, dtype=np.float32)
        self.assertEqual(len(r.descriptive(values, values, (8, 4, 1536))["per_token"]), 8)


if __name__ == "__main__":
    unittest.main()
