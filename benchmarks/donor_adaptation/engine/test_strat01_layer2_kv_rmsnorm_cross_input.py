from __future__ import annotations
import unittest
from pathlib import Path
from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_rmsnorm_cross_input as runner

class Layer2KvRmsnormCrossInputTests(unittest.TestCase):
 def test_protocol_and_reuse_are_frozen(self):
  p=runner.PROTOCOL.read_text(encoding="utf-8");h=runner.CROSS_HEADER.read_text(encoding="utf-8");self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION",p);self.assertIn(runner.PREDECESSOR_ADJUDICATION_SHA,p);self.assertIn("strat01_r2a_rmsnorm_pinned",h);self.assertIn("strat01_r2c_cross_run_arm",h);self.assertNotIn("strat01_r2a_attend_one(",h)
 def test_payloads_and_twins_are_exact(self):
  self.assertEqual(set(runner.PINNED_FILES),{"ref_q","ref_k","ref_pre","ref_norm","ref_target","c_pre","c_norm"});self.assertEqual(set(runner.SCHEDULE_TWINS),set(runner.PINNED_FILES));self.assertEqual(runner.PINNED_FILES["ref_pre"][2],18432);self.assertEqual(runner.PINNED_FILES["ref_norm"][2],16384)
 def test_engine_commands_exist(self):
  e=runner.ENGINE.read_text(encoding="utf-8");self.assertIn('#include "strat01_gguf_layer2_kv_rmsnorm_cross_input.h"',e);self.assertIn("--strat01-layer2-kv-rmsnorm-cross-input",e);self.assertIn("--strat01-layer2-kv-rmsnorm-cross-input-selftest",e)
 def test_arm_manifest(self):
  o={}
  for n in runner.ARM_NAMES:
   k,s,ic,wc=runner.ARM_META[n];o[n]={"kind":k,"source":s,"input_control":ic,"weight_control":wc,"prefix":{"path":n,"bytes":16384,"sha256":"0"*64},"downstream":{"path":n,"bytes":196608,"sha256":"1"*64}}
  runner.validate_arm_manifest(o);b=dict(o);b["bad"]=b.pop(runner.ARM_NAMES[-1])
  with self.assertRaises(runner.RunnerError): runner.validate_arm_manifest(b)
 def test_classification(self):
  p,f={"pass":True},{"pass":False};j={"captured_ref_norm":p,"captured_c_norm":f,"computed_ref_input":p,"computed_c_input":f};self.assertEqual(runner.classify(j),"LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT");j["computed_ref_input"]=f;self.assertEqual(runner.classify(j),"LAYER2_KV_RMSNORM_FAILS_EXACT_REFERENCE_INPUT")
 def test_source_inventory(self):
  i=runner.source_inventory();self.assertEqual(set(i),{"runner","tests","protocol","engine","rung2a","rung2c","inherited_cross_header","attention_header","partition_header","rmsnorm_header"});self.assertTrue(all(len(x["sha256"])==64 and Path(x["path"]).is_file() for x in i.values()))
if __name__=="__main__":unittest.main()
