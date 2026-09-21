from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine.run_strat01_f16_converter_audit import converter_status, exponent_class, segment_for


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

    def test_repair_labels_are_distinct_from_audit_labels(self) -> None:
        self.assertEqual(converter_status(True, False), "PASS_F16_CONVERTER")
        self.assertEqual(converter_status(False, False), "FAIL_F16_CONVERTER")
        self.assertEqual(converter_status(True, True), "PASS_F16_CONVERTER_REPAIR")
        self.assertEqual(converter_status(False, True), "FAIL_F16_CONVERTER_REPAIR")


if __name__ == "__main__": unittest.main()
