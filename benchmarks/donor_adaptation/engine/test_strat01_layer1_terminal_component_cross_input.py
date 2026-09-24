import unittest
import numpy as np
from benchmarks.donor_adaptation.engine import run_strat01_layer1_terminal_component_cross_input as r

class T(unittest.TestCase):
 def test_protocol(self): self.assertTrue(r.PROTOCOL.is_file()); self.assertIn("LAYER1_FFN_OUTPUT_RESIDUAL_SUFFICIENT",r.PROTOCOL.read_text(encoding="utf-8"))
 def test_sources(self): self.assertTrue(all(x["sha256"] for x in r.sources().values()))
 def test_engine(self):
  text=r.ENGINE.read_text(encoding="utf-8"); self.assertIn('#include "strat01_gguf_layer1_terminal_component_cross_input.h"',text); self.assertIn("--strat01-layer1-terminal-component-cross-input-selftest",text)
 def test_inputs(self):
  self.assertEqual(r.PINNED["ref_ffn_inp"][2],49152); self.assertEqual(r.PINNED["c_ffn_out"][1],"2932f1d3b23fc439ebdbee791a031a93e0681724f62935967e324b19a09196f5"); self.assertEqual(set(r.PINNED),set(r.TWINS))
 def test_predecessors(self): self.assertEqual(r.PREDECESSOR_SHA,"a42540afb96a70dfb411092d3f416cbbf840c83479302ac2eb1fd8bc9ad5d430"); self.assertEqual(r.INTEGRATION_SHA,"d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94")
 def test_classify(self):
  p,f={"pass":True},{"pass":False}; j={r.ARM_NAMES[0]:p,r.ARM_NAMES[1]:f,r.ARM_NAMES[3]:f,r.ARM_NAMES[4]:f,r.ARM_NAMES[5]:p}; self.assertEqual(r.classify(j),"LAYER1_FFN_OUTPUT_RESIDUAL_SUFFICIENT"); j[r.ARM_NAMES[5]]=f; self.assertEqual(r.classify(j),"LAYER1_TERMINAL_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT")
 def test_frozen_cross_sums(self):
  self.assertEqual(r.EXPECTED_SUM_SHA[r.ARM_NAMES[2]],r.PINNED["ref_l_out"][1]); self.assertEqual(r.EXPECTED_SUM_SHA[r.ARM_NAMES[3]],r.PINNED["c_l_out"][1]); self.assertEqual(len(r.EXPECTED_SUM_SHA[r.ARM_NAMES[4]]),64); self.assertEqual(len(r.EXPECTED_SUM_SHA[r.ARM_NAMES[5]]),64)
 def test_descriptive_shape(self):
  x=np.ones(12288,dtype=np.float32); self.assertEqual(len(r.descriptive(x,x,(8,1536))["per_token"]),8)

if __name__=="__main__": unittest.main()
