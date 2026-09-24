from __future__ import annotations
import unittest
import numpy as np
from benchmarks.donor_adaptation.engine import recover_strat01_layer2_kv_rmsnorm_cross_input as recovery
from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_rmsnorm_cross_input as runner

class RecoverLayer2KvRmsnormTests(unittest.TestCase):
 def test_protocol_binds_raw_void(self):
  p=recovery.PROTOCOL.read_text(encoding="utf-8");self.assertIn("FROZEN BEFORE RECOVERY IMPLEMENTATION OR EXECUTION",p);self.assertIn(recovery.RAW_SHA,p);self.assertIn(recovery.RAW_ERROR,p)
 def test_prefix_judged_uses_8_by_512(self):
  r=np.ones(4096,dtype=np.float32);x=runner.prefix_judged(r.copy(),r);self.assertTrue(x["pass"]);self.assertEqual(len(x["per_token"]),8);self.assertEqual([v["token"] for v in x["per_token"]],list(range(8)))
 def test_status_map_is_exhaustive(self):
  self.assertEqual(set(recovery.STATUS_MAP),runner.VALID_STATUSES);self.assertTrue(all(x.endswith("_RECOVERED") for x in recovery.STATUS_MAP.values()))
 def test_source_inventory_complete(self):
  self.assertEqual(set(recovery.source_inventory()),{"recovery","tests","protocol","scientific_runner","engine","rmsnorm_header"})
if __name__=="__main__":unittest.main()
