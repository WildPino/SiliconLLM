from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine.build_strat01_rung2a_projection_diagnostic import (
    BuildError,
    PINNED_HEAD,
    validate_pinned_state,
)
from benchmarks.donor_adaptation.engine.run_strat01_rung2a_projection_diagnostic import (
    DiagnosticError,
    TIGHT_MAX,
    TIGHT_NRMSE,
    classify,
    metrics,
    sha256_file,
    validate_layout,
    validate_payload,
)


class ProjectionDiagnosticTests(unittest.TestCase):
    def test_one_byte_payload_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.f32"
            np.arange(16, dtype="<f4").tofile(path)
            digest = sha256_file(path)
            validate_payload(path, 16, digest)
            data = bytearray(path.read_bytes())
            data[7] ^= 1
            path.write_bytes(data)
            with self.assertRaises(DiagnosticError):
                validate_payload(path, 16, digest)

    def test_layout_and_row_substitutions_are_rejected(self) -> None:
        validate_layout("q", (8, 6144), "token,head,feature")
        with self.assertRaises(DiagnosticError):
            validate_layout("q", (6144, 8), "feature,token")
        with self.assertRaises(DiagnosticError):
            validate_layout("kv", (8, 575), "token,feature")

    def test_token_swap_fails_tight_gate(self) -> None:
        target = np.arange(32, dtype=np.float32).reshape(2, 16)
        swapped = target[::-1].copy()
        result = metrics(swapped.ravel(), target.ravel())
        self.assertGreater(result["nrmse"], TIGHT_NRMSE)
        self.assertGreater(result["normalized_max"], TIGHT_MAX)
        self.assertFalse(result["tight_pass"])

    def test_wrong_or_dirty_pinned_revision_is_rejected(self) -> None:
        validate_pinned_state(PINNED_HEAD, "")
        with self.assertRaises(BuildError):
            validate_pinned_state("0" * 40, "")
        with self.assertRaises(BuildError):
            validate_pinned_state(PINNED_HEAD, " M ggml/src/ggml-cpu/quants.c")

    def test_nonfinite_metric_input_is_rejected(self) -> None:
        with self.assertRaises(DiagnosticError):
            metrics(np.array([np.nan], dtype=np.float32), np.array([0.0], dtype=np.float32))

    def test_frozen_classifier_requires_both_reproductions(self) -> None:
        def cell(tight: bool, old: bool) -> dict[str, bool]:
            return {"tight_pass": tight, "old_gate_pass": old}

        controls = {"identity": True, "negative": True}
        attributed = {
            name: {
                "d32_c_control": cell(True, True),
                "q8k_reference": cell(True, True),
                "d32_reference": cell(False, False),
            }
            for name in ("q", "kv")
        }
        self.assertEqual(classify(attributed, controls), "ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION")
        attributed["kv"]["q8k_reference"] = cell(False, False)
        self.assertEqual(classify(attributed, controls), "PARTIAL_Q8K_ATTRIBUTION")
        attributed["q"]["q8k_reference"] = cell(False, False)
        self.assertEqual(classify(attributed, controls), "REJECT_Q8K_AS_SUFFICIENT")
        controls["negative"] = False
        self.assertEqual(classify(attributed, controls), "VOID_PROJECTION_DIAGNOSTIC")


if __name__ == "__main__":
    unittest.main()
