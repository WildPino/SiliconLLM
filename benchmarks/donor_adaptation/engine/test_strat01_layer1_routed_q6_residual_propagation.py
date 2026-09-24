import unittest

import numpy as np

from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_q6_residual_propagation as r


class TestLayer1RoutedQ6ResidualPropagation(unittest.TestCase):
    def test_protocol(self):
        self.assertTrue(r.PROTOCOL.is_file())
        self.assertIn("LAYER1_ROUTED_SWIGLU_RESIDUAL_SUFFICIENT", r.PROTOCOL.read_text(encoding="utf-8"))

    def test_sources(self):
        self.assertTrue(all(item["sha256"] for item in r.sources().values()))

    def test_engine_wiring(self):
        text = r.ENGINE.read_text(encoding="utf-8")
        self.assertIn('#include "strat01_gguf_layer1_routed_q6_residual_propagation.h"', text)
        self.assertIn("--strat01-layer1-routed-q6-residual-propagation-selftest", text)

    def test_inputs_and_twins(self):
        self.assertEqual(set(r.PINNED), set(r.TWINS))
        self.assertEqual(r.Q6_REF_DOWN_SHA, "609aa170ac5acf0cec49e403e28e1dde831d14cb76f1de35d1ce51e39e9df91f")

    def test_predecessors(self):
        self.assertEqual(r.PREDECESSOR_SHA, "ac78794d24467b2ca4c3fce092b969bfd9fa9a2698d2227d5ea5e5c67640ab59")
        self.assertEqual(r.OLD_Q6_ADJ_SHA, "db9b478e208c35460d4bd1ae54eb56cfb2a8106361b19f0b7f1f024699509148")

    def test_frozen_hashes(self):
        self.assertEqual(r.EXPECTED_WEIGHTED_SHA[r.ARM_NAMES[0]], "875d7a321cfb4c28343edc04bdd1f9940ce014fcfa2aa1306922291c014009b2")
        self.assertEqual(r.EXPECTED_MOE_SHA[r.ARM_NAMES[1]], "57106eb8eb23cf38da05165b522b5c0cc00b6fbc899bc9d306dc1e665093f4b6")
        self.assertTrue(all(len(value) == 64 for value in (*r.EXPECTED_WEIGHTED_SHA.values(), *r.EXPECTED_MOE_SHA.values())))

    def test_arm_manifest_rejects_label_swap(self):
        sizes = (("weighted", 196608), ("moe_out", 49152), ("ffn_out", 49152), ("l_out", 49152), ("norm", 49152), ("projection", 18432), ("prefix", 16384), ("downstream", 196608))
        outputs = {}
        for name in r.ARM_NAMES:
            origin, control = r.ARM_META[name]; payload = {key: {"path": key, "bytes": size, "sha256": "0" * 64} for key, size in sizes}; outputs[name] = {"origin": origin, "down_control": control, **payload}
        r.arm_manifest(outputs); items = list(outputs.items()); items[1], items[2] = items[2], items[1]
        with self.assertRaises(r.RunnerError):
            r.arm_manifest(dict(items))

    def test_descriptive_shape(self):
        values = np.ones(49152, dtype=np.float32)
        self.assertEqual(len(r.descriptive(values, values, (8, 4, 1536))["per_token"]), 8)


if __name__ == "__main__":
    unittest.main()
