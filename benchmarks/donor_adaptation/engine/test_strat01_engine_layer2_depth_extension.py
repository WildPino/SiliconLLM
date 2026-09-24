from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

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

    def test_scientific_checkpoint_inventory_is_exact(self) -> None:
        self.assertEqual(len(layer2.SHAPES), 32)
        self.assertEqual(set(layer2.SHAPES), {"l_out-1", *{
            name[:-1] + "2" for name in layer2.r2c.SHAPES if name != "l_out-0"
        }})
        self.assertEqual(layer2.SHAPES["l_out-1"], [1536, 8])
        self.assertEqual(layer2.SHAPES["ffn_moe_topk-2"], [4, 8])
        self.assertEqual(layer2.I32_NAMES, {"ffn_moe_topk-2"})

    def test_reference_and_c_have_distinct_rung2d_graph_markers(self) -> None:
        source = layer2.REFERENCE_SOURCE.read_text(encoding="utf-8")
        self.assertEqual(layer2.GRAPH_MARKER, "STRAT01_RUNG2D_GRAPH_COMPLETE arm=")
        self.assertEqual(layer2.REFERENCE_GRAPH_MARKER, layer2.GRAPH_MARKER)
        self.assertIn('kGraphMarker = "STRAT01_RUNG2D_GRAPH_COMPLETE arm="', source)

    def test_graph_completion_requires_each_arm_exactly_once(self) -> None:
        valid = {"stdout": "", "stderr": (
            layer2.GRAPH_MARKER + "prefill8\n" +
            layer2.GRAPH_MARKER + "cached7p1\n"
        )}
        duplicate = {"stdout": valid["stderr"], "stderr": layer2.GRAPH_MARKER + "prefill8\n"}
        missing = {"stdout": layer2.GRAPH_MARKER + "prefill8\n", "stderr": ""}
        self.assertEqual(layer2.completed_graph_count(valid, layer2.GRAPH_MARKER), 2)
        self.assertEqual(layer2.completed_graph_count(duplicate, layer2.GRAPH_MARKER), -1)
        self.assertEqual(layer2.completed_graph_count(missing, layer2.GRAPH_MARKER), -1)

    def test_layer2_negative_controls_translate_to_rung2c_contract(self) -> None:
        sentinel = np.array([1.0], dtype=np.float32)
        reference = {}
        for arm in layer2.r2c.base.ARMS:
            reference[f"{arm}/l_out-1"] = sentinel
            for name in layer2.r2c.SHAPES:
                if name != "l_out-0":
                    reference[f"{arm}/{name[:-1]}2"] = sentinel
        with mock.patch.object(layer2.r2c, "negative_controls", return_value={"ok": {"pass": False}}) as call:
            result = layer2.layer2_negative_controls(reference)
        translated = call.call_args.args[0]
        self.assertEqual(result, {"ok": {"pass": False}})
        for arm in layer2.r2c.base.ARMS:
            self.assertIs(translated[f"{arm}/l_out-0"], sentinel)
            self.assertEqual(set(k.removeprefix(f"{arm}/") for k in translated if k.startswith(f"{arm}/")), set(layer2.r2c.SHAPES))

    def test_scientific_frontier_hashes_are_frozen(self) -> None:
        self.assertEqual(layer2.REFERENCE_START_SHA, "40d5a0f07fbb77c1ef73ca24f81cb35ea0df32457faa8d04d6c5cd33cd9f1d5f")
        self.assertEqual(layer2.C_START_SHA, "9af8cec3f42781e9f4cac6af1ea13f6a63b751a5323201a8a18f91b3f7b7bbcb")


if __name__ == "__main__":
    unittest.main()
