import unittest
import numpy as np
from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_up_projection_cross_input as r


class TestLayer1RoutedUpProjectionCrossInput(unittest.TestCase):
    def test_protocol(self):
        text=r.PROTOCOL.read_text(encoding="utf-8");self.assertIn("LAYER1_ROUTED_UPSTREAM_NORM_RESIDUAL_SUFFICIENT",text);self.assertIn("LAYER1_ROUTED_UP_PROJECTION_RESIDUAL_SUFFICIENT",text)
    def test_sources(self): self.assertTrue(all(v["sha256"] for v in r.sources().values()))
    def test_engine_wiring(self):
        text=r.ENGINE.read_text(encoding="utf-8");self.assertIn('#include "strat01_gguf_layer1_routed_up_projection_cross_input.h"',text);self.assertIn("--strat01-layer1-routed-up-projection-cross-input-selftest",text)
    def test_arms(self): self.assertEqual(len(r.ARM_NAMES),5);self.assertEqual(r.ARM_ORIGINS[-1],"reference_norm_control")
    def test_hashes(self): self.assertEqual(r.INPUTS["reference_norm"][1],"3f4826dd184c9442fa34807c9649aeae88343c5894c7aac531e5b5a209806dde");self.assertTrue(all(len(v[1])==64 for v in r.INPUTS.values()))
    def test_predecessor(self): self.assertEqual(r.PREDECESSOR_SHA,"04b46ac959b096bf8d2df8f1735882e37afbafacdc43c1c4eadff7560093a705")
    def test_manifest_swap(self):
        sizes={"up":163840,"swiglu":163840,"down":196608,"moe_out":49152,"downstream":196608};outputs={}
        for i,n in enumerate(r.ARM_NAMES): outputs[n]={"origin":r.ARM_ORIGINS[i],**{k:{"path":k,"bytes":s,"sha256":"0"*64} for k,s in sizes.items()}}
        r.arm_manifest(outputs);items=list(outputs.items());items[2],items[3]=items[3],items[2]
        with self.assertRaises(r.RunnerError): r.arm_manifest(dict(items))
    def test_descriptive(self):
        x=np.ones(40960,dtype=np.float32);self.assertEqual(len(r.descriptive(x,x,(8,4,1280))["per_token"]),8)


if __name__=="__main__": unittest.main()
