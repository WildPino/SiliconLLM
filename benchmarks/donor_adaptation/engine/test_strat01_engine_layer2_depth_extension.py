from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine import run_strat01_engine_layer2_depth_extension as layer2


class Layer2DepthExtensionTests(unittest.TestCase):
    def test_protocol_and_predecessor_are_frozen(self) -> None:
        text = layer2.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", text)
        self.assertIn(layer2.PREDECESSOR_SHA, text)
        self.assertEqual(layer2.validate_predecessor()["sha256"], layer2.PREDECESSOR_SHA)

    def test_source_inventory_is_complete(self) -> None:
        inventory = layer2.source_inventory()
        self.assertEqual(set(inventory), {
            "runner", "tests", "protocol", "audit", "engine", "rung2a", "rung2b", "rung2c", "rung2d",
            "reference_source", "reference_build", "reference_wrapper", "rung2c_runner", "production_runner",
            "base_runner", "q4_parity_runner",
        })
        self.assertTrue(all(len(item["sha256"]) == 64 for item in inventory.values()))

    def test_all_source_controls_fire(self) -> None:
        controls = layer2.source_controls()
        self.assertEqual(len(controls), 12)
        self.assertTrue(all(controls.values()), controls)

    def test_exact_layer2_tensor_inventory(self) -> None:
        text = layer2.RUNG2D.read_text(encoding="utf-8")
        self.assertEqual(text.count('{"blk.2.'), 16)
        self.assertIn('{"blk.2.attn_k_b.weight",       STRAT01_GGML_Q5_0,3,{128,512,32}}', text)
        self.assertIn('{"blk.2.ffn_down_exps.weight",  STRAT01_GGML_Q6_K,3,{1280,1536,64}}', text)

    def test_depth_only_reuses_accepted_layer_primitives(self) -> None:
        text = layer2.RUNG2D.read_text(encoding="utf-8")
        self.assertIn("strat01_r2c_build_attention_range(path,a1", text)
        self.assertIn("strat01_r2c_build_attention_range(path,a2", text)
        self.assertIn("strat01_r2c_run_moe(path,m1", text)
        self.assertIn("strat01_r2c_run_moe(path,m2", text)

    def test_checkpoint_cache_and_count_contract(self) -> None:
        text = layer2.RUNG2D.read_text(encoding="utf-8")
        self.assertEqual(text.count('F("'), 31)
        self.assertIn('"ffn_moe_topk-2"', text)
        self.assertIn("layer<3U", text)
        self.assertEqual(layer2.EXPECTED_COUNTS["qk_invocations"], 6912)
        self.assertEqual(layer2.EXPECTED_COUNTS["value_invocations"], 786432)

    def test_reference_variant_is_separate(self) -> None:
        source = layer2.REFERENCE_SOURCE.read_text(encoding="utf-8")
        build = layer2.REFERENCE_BUILD.read_text(encoding="utf-8")
        self.assertIn("STRAT01_RUNG2D", source)
        self.assertIn("strat01_engine_rung2d_reference_manifest_v1", source)
        self.assertIn('\\"layers\\":[0,1,2]', source)
        self.assertIn('variant = "rung2d" if rung2d', build)


if __name__ == "__main__":
    unittest.main()
