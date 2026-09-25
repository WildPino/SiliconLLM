from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine import run_strat01_router_weight_normalization_compile_parity as runner


class RouterWeightNormalizationCompileParityTests(unittest.TestCase):
    def test_protocol_and_predecessor_are_bound(self) -> None:
        self.assertIn("FROZEN BEFORE IMPLEMENTATION OR SCIENTIFIC EXECUTION", runner.PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(runner.validate_binding()["sha256"], runner.PREDECESSOR_SHA)

    def test_sources_are_independent_and_graph_free(self) -> None:
        controls = runner.source_controls()
        self.assertTrue(all(controls.values()), controls)
        self.assertTrue(controls["oracle_independent_translation_unit"])
        self.assertTrue(controls["no_model_or_graph_entrypoint"])

    def test_output_inventory_is_exact(self) -> None:
        self.assertEqual(len(runner.OUTPUTS), 8)
        self.assertTrue(all(count == 32 for count in runner.OUTPUTS.values()))

    def test_apparatus_and_scientific_outputs_are_separate(self) -> None:
        self.assertNotEqual(runner.DEFAULT_APPARATUS, runner.DEFAULT_OUTPUT)


if __name__ == "__main__":
    unittest.main()
