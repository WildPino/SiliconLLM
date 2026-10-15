"""Pure-stdlib checks; tensor derivatives execute exclusively on Kaggle."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("temporal", Path(__file__).with_name("temporal.py"))
temporal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temporal)


class SpecificationTests(unittest.TestCase):
    def test_prescribed_sampling_is_frozen(self):
        self.assertEqual(temporal.SPEC["anchors"], [63, 79, 95])
        self.assertEqual(temporal.SPEC["query_offsets"], [0, 16])
        self.assertEqual(temporal.SPEC["basis_samples"], 16)
        self.assertEqual(temporal.SPEC["evaluation_samples"], 8)
        self.assertEqual(temporal.SPEC["rank"], 8)

    def test_sampling_streams_are_distinct_and_deterministic(self):
        a = temporal.seed_for(1729, "dev", 0, 63, 0, "basis")
        self.assertEqual(a, temporal.seed_for(1729, "dev", 0, 63, 0, "basis"))
        self.assertNotEqual(a, temporal.seed_for(1729, "dev", 0, 63, 0, "evaluation"))

    def test_paired_bootstrap_preserves_document_pairs(self):
        interval = temporal.paired_interval([.2] * 8, 1729, 100)
        self.assertEqual(interval["paired_documents"], 8)
        self.assertAlmostEqual(interval["mean"], .2)
        self.assertAlmostEqual(interval["ci95"][0], .2)
        self.assertAlmostEqual(interval["ci95"][1], .2)


if __name__ == "__main__":
    unittest.main()
