import inspect
import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_layer1_ffn_rmsnorm_cross_input as r


class TestLayer1FfnRmsnormCrossInput(unittest.TestCase):
    def test_protocol_statuses_and_predecessor(self):
        text = r.PROTOCOL.read_text(encoding="utf-8")
        for status in ("LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT", "LAYER1_FFN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT", "VOID_LAYER1_FFN_RMSNORM_CROSS_INPUT"):
            self.assertIn(status, text)
        self.assertEqual(r.PREDECESSOR_SHA, "2de61cdc02deeb9639044cb75cc614705a41eb3e133b00f3ad2746b80bc7c71d")

    def test_sources_and_engine_wiring(self):
        self.assertTrue(all(item["sha256"] for item in r.sources().values()))
        engine = r.ENGINE.read_text(encoding="utf-8")
        self.assertIn('#include "strat01_gguf_layer1_ffn_rmsnorm_cross_input.h"', engine)
        self.assertIn("--strat01-layer1-ffn-rmsnorm-cross-input-selftest", engine)

    def test_frozen_inputs_and_twins(self):
        self.assertEqual(set(r.INPUTS), set(r.TWINS))
        self.assertEqual(r.INPUTS["reference_input"][1], "99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8")
        self.assertEqual(r.INPUTS["c_input"][1], "c506fcc0e51d6b80c90958374527f48c1dd25150bf91b9ff83851c059ec890b2")
        self.assertTrue(all(len(item[1]) == 64 for item in r.INPUTS.values()))
        self.assertEqual(r.REFERENCE_TARGET[1], "d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e")
        self.assertEqual(r.REFERENCE_TARGET[1], r.up.INPUTS["ref_target"][1])
        self.assertEqual(r.PREDECESSOR_C.parent.parent, r.up.DEFAULT_OUTPUT)
        self.assertEqual(r.PREDECESSOR_C.name, "current_up_on_c_norm.downstream.f32le")
        evidence_source = inspect.getsource(r.evidence)
        self.assertIn('found["ref_target"] = target_path.resolve(strict=True)', evidence_source)
        self.assertIn('found["prior_c"] = PREDECESSOR_C.resolve(strict=True)', evidence_source)

    def test_arm_manifest_rejects_swap(self):
        sizes = {"norm": 49152, "up": 163840, "swiglu": 163840, "down": 196608, "moe_out": 49152, "downstream": 196608}
        outputs = {name: {"kind": r.ARM_META[name][0], "source": r.ARM_META[name][1], "input_control": r.ARM_META[name][2], "weight_control": r.ARM_META[name][3], **{key: {"path": key, "bytes": size, "sha256": "0" * 64} for key, size in sizes.items()}} for name in r.ARM_NAMES}
        r.arm_manifest(outputs)
        items = list(outputs.items()); items[2], items[3] = items[3], items[2]
        with self.assertRaises(r.RunnerError):
            r.arm_manifest(dict(items))

    def test_classification_and_metrics(self):
        good, bad = {"pass": True}, {"pass": False}
        judgments = {"captured_reference_norm": good, "captured_production_c_norm": bad, "computed_norm_on_reference_input": good, "computed_norm_on_c_input": bad}
        self.assertEqual(r.classify(judgments), "LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT")
        judgments["computed_norm_on_reference_input"] = bad
        self.assertEqual(r.classify(judgments), "LAYER1_FFN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT")
        judgments["computed_norm_on_c_input"] = good
        self.assertEqual(r.classify(judgments), "VOID_LAYER1_FFN_RMSNORM_CROSS_INPUT")
        self.assertEqual(len(r.descriptive(np.ones(12288, dtype=np.float32), np.ones(12288, dtype=np.float32), (8, 1536))["per_token"]), 8)
        reference = np.ones((8, 6144), dtype=np.float32)
        candidate = reference.copy(); candidate[7] += np.float32(0.003)
        judged = r.judged_downstream(candidate.ravel(), reference.ravel())
        self.assertTrue(judged["aggregate_pass"])
        self.assertFalse(judged["per_token"][7]["pass"])
        self.assertFalse(judged["pass"])

    def test_header_has_six_arm_accounting_and_controls(self):
        header = r.HEADER.read_text(encoding="utf-8")
        self.assertIn("strat01_l1fr_arms[6]", header)
        self.assertIn("propagated_arms_completed", header)
        self.assertNotIn("computed-reference replay mismatch", header)
        self.assertIn("computed-C replay mismatch", header)


if __name__ == "__main__":
    unittest.main()
