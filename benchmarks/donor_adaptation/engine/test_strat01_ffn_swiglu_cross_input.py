from __future__ import annotations

import hashlib
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_ffn_swiglu_cross_input as runner


class FFNSwiGLUTests(unittest.TestCase):
    def test_protocol_and_sources(self) -> None:
        self.assertIn("Exactly one non-VOID invocation", runner.PROTOCOL.read_text(encoding="utf-8")); self.assertIn("--strat01-ffn-swiglu-cross-input", runner.ENGINE.read_text(encoding="utf-8")); self.assertEqual(set(runner.source_inventory()), {"runner","down_runner","terminal_runner","layer1_runner","engine","rung2a","rung2b","rung2c","down_header","header","protocol","tests"})

    def test_frozen_contract(self) -> None:
        self.assertEqual(len(runner.ARMS),5);self.assertEqual(set(runner.ORIGINS),set(runner.ARMS));self.assertEqual(len(runner.INPUTS),7);self.assertTrue(all(len(d)==64 for _,d,_ in runner.INPUTS.values()))

    def test_swiglu_shape_gate(self) -> None:
        ref=np.ones(8*8960,dtype=np.float32);self.assertTrue(runner.wide_judgment(ref.copy(),ref)["pass"]);changed=ref.copy();changed[0]=2;self.assertFalse(runner.wide_judgment(changed,ref)["pass"])

    def test_expression_and_finite_control(self) -> None:
        gate=np.array([0.0,1.0,-1.0],dtype=np.float32);up=np.array([2.0,2.0,2.0],dtype=np.float32);out=(gate/(np.float32(1.0)+np.exp(-gate)))*up;self.assertEqual(float(out[0]),0.0);self.assertGreater(float(out[1]),0.0);self.assertLess(float(out[2]),0.0)

    def test_mutation_refusal(self) -> None:
        data=b"swiglu";digest=hashlib.sha256(data).hexdigest();mutated=bytearray(data);mutated[0]^=1;self.assertTrue(runner.base.identity_matches(data,len(data),digest));self.assertFalse(runner.base.identity_matches(bytes(mutated),len(data),digest))


if __name__=="__main__":unittest.main()
