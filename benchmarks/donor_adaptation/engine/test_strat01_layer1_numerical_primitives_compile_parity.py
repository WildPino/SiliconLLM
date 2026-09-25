from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine import run_strat01_layer1_numerical_primitives_compile_parity as runner


class Layer1NumericalPrimitivesTests(unittest.TestCase):
    def test_protocol_and_predecessors_are_bound(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR EXECUTION", protocol)
        self.assertIn("Pre-implementation compiler-path correction", protocol)
        bindings = runner.validate_bindings()
        self.assertEqual(bindings["router_descriptor"]["file_offset"], runner.ROUTER_OFFSET)

    def test_sources_are_independent_and_graph_free(self) -> None:
        controls = runner.source_controls()
        self.assertTrue(all(controls.values()), controls)
        self.assertTrue(controls["oracle_independent_translation_unit"])
        self.assertTrue(controls["no_engine_or_graph_entrypoint"])

    def test_output_inventory_is_exact(self) -> None:
        self.assertEqual(len(runner.OUTPUT_COUNTS), 12)
        self.assertEqual(runner.OUTPUT_COUNTS["router_candidate.f32"], 512)
        self.assertEqual(runner.OUTPUT_COUNTS["routed_sse2.f32"], 40_960)
        self.assertEqual(runner.OUTPUT_COUNTS["shared_sse2.f32"], 10_240)

    def test_status_partition(self) -> None:
        self.assertEqual(runner.classify(True, True, True), "LAYER1_ROUTER_AND_SWIGLU_EXACT_PRIMITIVES")
        self.assertEqual(runner.classify(True, False, False), "LAYER1_ROUTER_EXACT_SWIGLU_INSUFFICIENT")
        self.assertEqual(runner.classify(False, True, True), "LAYER1_SWIGLU_EXACT_ROUTER_INSUFFICIENT")
        self.assertEqual(runner.classify(False, True, False), "LAYER1_NUMERICAL_PRIMITIVES_INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
