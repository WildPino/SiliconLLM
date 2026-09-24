import unittest
import numpy as np
from pathlib import Path
from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_a_projection_cross_input as r
class T(unittest.TestCase):
 def test_protocol(self):self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION",r.PROTOCOL.read_text(encoding="utf-8"));self.assertIn("strat01_r2a_matmul_batch",r.CROSS_HEADER.read_text(encoding="utf-8"))
 def test_inputs(self):self.assertEqual(set(r.PINNED),set(r.TWINS));self.assertEqual(r.PINNED["ref_attn"][2],49152)
 def test_engine(self):e=r.ENGINE.read_text(encoding="utf-8");self.assertIn("--strat01-layer2-kv-a-projection-cross-input",e);self.assertIn("--strat01-layer2-kv-a-projection-cross-input-selftest",e)
 def test_classify(self):p,f={"pass":True},{"pass":False};j={"captured_ref_projection":p,"captured_c_projection":f,"computed_ref_input":p,"computed_c_input":f};self.assertEqual(r.classify(j),"LAYER2_ATTN_NORM_INPUT_RESIDUAL_SUFFICIENT");j["computed_ref_input"]=f;self.assertEqual(r.classify(j),"LAYER2_KV_A_PROJECTION_FAILS_EXACT_REFERENCE_INPUT")
 def test_projection_metrics_shape(self):x=np.ones(4608,dtype=np.float32);self.assertEqual(len(r.projection_judged(x,x)["per_token"]),8)
 def test_predecessor_output_path(self):self.assertEqual(r.rms.DEFAULT_OUTPUT.name,"strat01_gigachat_engine_layer2_kv_rmsnorm_cross_input_20260924");self.assertNotEqual(r.rms.DEFAULT_OUTPUT,r.rms.RAW)
 def test_sources(self):self.assertTrue(all(Path(x["path"]).is_file() for x in r.sources().values()))
if __name__=="__main__":unittest.main()
