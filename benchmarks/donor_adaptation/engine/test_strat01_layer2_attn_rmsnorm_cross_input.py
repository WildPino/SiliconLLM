import unittest
import numpy as np
from benchmarks.donor_adaptation.engine import run_strat01_layer2_attn_rmsnorm_cross_input as r

class T(unittest.TestCase):
 def test_protocol(self): self.assertTrue(r.PROTOCOL.is_file()); self.assertIn("LAYER1_TERMINAL_RESIDUAL_SUFFICIENT",r.PROTOCOL.read_text(encoding="utf-8"))
 def test_sources(self): self.assertTrue(all(x["sha256"] for x in r.sources().values()))
 def test_engine(self):
  e=r.ENGINE.read_text(encoding="utf-8"); self.assertIn('#include "strat01_gguf_layer2_attn_rmsnorm_cross_input.h"',e); self.assertIn("--strat01-layer2-attn-rmsnorm-cross-input-selftest",e)
 def test_cleanup_symbol(self):
  h=r.CROSS_HEADER.read_text(encoding="utf-8"); self.assertIn("strat01_free_inventory(&inv)",h); self.assertNotIn("strat01_inventory_free",h)
 def test_inputs(self):
  self.assertEqual(r.PINNED["ref_input"][2],49152); self.assertEqual(r.PINNED["c_norm"][1],"b55cdc958cb9e48ff014c740dc4bfe6b1ec81ef839ee95618018e426673e112d"); self.assertEqual(set(r.PINNED),set(r.TWINS))
 def test_predecessor(self): self.assertEqual(r.PREDECESSOR_SHA,"d16af478a412e22457f11e4d0946e8c7d9fec1c842dfee9657fb213ec8cc06d1"); self.assertEqual(r.PREDECESSOR_OUTPUT_SHA,"81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254")
 def test_classify(self):
  p,f={"pass":True},{"pass":False}; j={"captured_ref_norm":p,"captured_c_norm":f,"computed_ref_input":p,"computed_c_input":f}; self.assertEqual(r.classify(j),"LAYER1_TERMINAL_RESIDUAL_SUFFICIENT"); j["computed_ref_input"]=f; self.assertEqual(r.classify(j),"LAYER2_ATTN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT")
 def test_descriptive_shape(self): x=np.ones(12288,dtype=np.float32); self.assertEqual(len(r.descriptive(x,x,(8,1536))["per_token"]),8)

if __name__=="__main__": unittest.main()
