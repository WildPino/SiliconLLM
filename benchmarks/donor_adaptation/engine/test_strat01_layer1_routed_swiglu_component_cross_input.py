import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_swiglu_component_cross_input as r


class TestLayer1RoutedSwiGLUComponentCrossInput(unittest.TestCase):
    def test_protocol(self):
        text = r.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("LAYER1_ROUTED_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT", text)
        self.assertIn("SSE2_FULL_C_REPAIR_PASSES", text)

    def test_sources(self):
        self.assertTrue(all(item["sha256"] for item in r.sources().values()))

    def test_engine_wiring(self):
        text = r.ENGINE.read_text(encoding="utf-8")
        self.assertIn('#include "strat01_gguf_layer1_routed_swiglu_component_cross_input.h"', text)
        self.assertIn("--strat01-layer1-routed-swiglu-component-cross-input-selftest", text)

    def test_arm_order(self):
        self.assertEqual(len(r.ARM_NAMES), 9)
        self.assertEqual(r.ARM_KINDS[-1], "control")
        self.assertEqual(r.ARM_NAMES[4], "sse2_reference_gate_reference_up")

    def test_hashes_and_shapes(self):
        self.assertEqual(r.TOPK_SHA, "557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95")
        self.assertEqual(r.INPUTS["reference_swiglu"][2], 163840)
        self.assertTrue(all(len(item[1]) == 64 for item in r.INPUTS.values()))

    def test_predecessor(self):
        self.assertEqual(r.PREDECESSOR_SHA, "2eee409b8f007b9e53136e7a3d14d250af93a4d951b1ce859d4080fcfe6a9bc2")
        self.assertEqual(r.PREDECESSOR_C_SHA, "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254")

    def test_repair_scope(self):
        text = r.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("Only the scalar C/C arm", text)
        self.assertIn("VOID 1", text)

    def test_manifest_rejects_swap(self):
        sizes = {"swiglu":163840,"down":196608,"moe_out":49152,"downstream":196608}; outputs = {}
        for i, name in enumerate(r.ARM_NAMES): outputs[name] = {"kind":r.ARM_KINDS[i], **{key:{"path":key,"bytes":size,"sha256":"0"*64} for key,size in sizes.items()}}
        r.arm_manifest(outputs); items=list(outputs.items()); items[5],items[6]=items[6],items[5]
        with self.assertRaises(r.RunnerError): r.arm_manifest(dict(items))

    def test_descriptive(self):
        x=np.ones(40960,dtype=np.float32)
        self.assertEqual(len(r.descriptive(x,x,(8,4,1280))["per_token"]),8)


if __name__ == "__main__":
    unittest.main()
