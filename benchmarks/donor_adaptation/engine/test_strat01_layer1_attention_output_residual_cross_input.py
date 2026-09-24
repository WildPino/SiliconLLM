import inspect
import unittest

from benchmarks.donor_adaptation.engine import run_strat01_layer1_attention_output_residual_cross_input as r


class TestLayer1AttentionOutputResidualCrossInput(unittest.TestCase):
    def test_protocol_statuses_and_predecessor(self):
        text = r.PROTOCOL.read_text(encoding="utf-8")
        for status in ("LAYER1_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT", "LAYER1_ATTN_OUTPUT_PROJECTION_FAILS_EXACT_KQV", "VOID_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT"):
            self.assertIn(status, text)
        self.assertEqual(r.PREDECESSOR_SHA, "831fe805bdffeee26693543bed60e96f9f0fe42945e315b16108677736589943")
        self.assertIn("must not be forced to replay", text)

    def test_sources_and_engine_wiring(self):
        self.assertTrue(all(item["sha256"] for item in r.sources().values()))
        engine = r.ENGINE.read_text(encoding="utf-8")
        self.assertIn('#include "strat01_gguf_layer1_attention_output_residual_cross_input.h"', engine)
        self.assertIn("--strat01-layer1-attention-output-residual-cross-input-selftest", engine)

    def test_frozen_contract_without_payload_reads(self):
        self.assertEqual(set(r.INPUTS), set(r.TWINS))
        self.assertEqual(r.INPUTS["reference_kqv"][1], r.INPUTS["c_kqv"][1])
        self.assertEqual(r.INPUTS["c_residual"][1], "a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11")
        self.assertTrue(all(len(item[1]) == 64 for item in r.INPUTS.values()))
        self.assertEqual(r.REFERENCE_TARGET[1], "d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e")
        main_source = inspect.getsource(r.main)
        self.assertLess(main_source.index("if args.apparatus_only"), main_source.index("frozen = evidence()"))

    def test_arm_manifest_rejects_swap(self):
        sizes = {"projection": 49152, "ffn_input": 49152, "norm": 49152, "up": 163840,
                 "swiglu": 163840, "down": 196608, "moe_out": 49152, "downstream": 196608}
        outputs = {name: {"kind": r.ARM_META[name][0], "source": r.ARM_META[name][1],
                          "residual_control": r.ARM_META[name][2], "kqv_control": r.ARM_META[name][3],
                          **{key: {"path": key, "bytes": size, "sha256": "0" * 64} for key, size in sizes.items()}}
                   for name in r.ARM_NAMES}
        r.arm_manifest(outputs)
        items = list(outputs.items()); items[2], items[3] = items[3], items[2]
        with self.assertRaises(r.RunnerError):
            r.arm_manifest(dict(items))

    def test_classification(self):
        good, bad = {"pass": True}, {"pass": False}
        judgments = {r.ARM_NAMES[0]: good, r.ARM_NAMES[1]: bad, r.ARM_NAMES[2]: good, r.ARM_NAMES[3]: bad}
        replay = {"stage": True}
        self.assertEqual(r.classify(judgments, replay, replay), "LAYER1_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT")
        judgments[r.ARM_NAMES[2]] = bad
        self.assertEqual(r.classify(judgments, replay, replay), "LAYER1_ATTN_OUTPUT_PROJECTION_FAILS_EXACT_KQV")
        judgments[r.ARM_NAMES[2]] = good
        self.assertEqual(r.classify(judgments, {"stage": False}, replay), "VOID_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT")

    def test_header_accounting_and_estimand(self):
        header = r.HEADER.read_text(encoding="utf-8")
        self.assertIn("strat01_l1ao_arms[6]", header)
        self.assertIn("projection_arms_completed", header)
        self.assertIn("ffn_input_arms_completed", header)
        self.assertNotIn("computed-reference replay mismatch", header)
        self.assertIn("STRAT01_L1AO_KQV_SHA", header)


if __name__ == "__main__":
    unittest.main()
