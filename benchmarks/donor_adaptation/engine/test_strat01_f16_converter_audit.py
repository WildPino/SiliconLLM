from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_f16_converter_audit import exponent_class, segment_for


class F16ConverterAuditTests(unittest.TestCase):
    def test_segments_cover_frozen_population(self) -> None:
        self.assertEqual(segment_for(0), ("kcur", 0))
        self.assertEqual(segment_for(8 * 576), ("qcur", 0))
        self.assertEqual(segment_for(8 * 576 + 8 * 32 * 576), ("softmax", 0))

    def test_exponent_classification(self) -> None:
        self.assertEqual(exponent_class(0x00000000), "zero")
        self.assertEqual(exponent_class(0x00000001), "subnormal_f32")
        self.assertEqual(exponent_class(0x7F800000), "infinity")
        self.assertEqual(exponent_class(0x7FC00000), "nan")


if __name__ == "__main__": unittest.main()
