#!/usr/bin/env python3
"""Model-free structural tests for the STRAT-01 Rung-2B apparatus."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_strat01_engine_rung2a_reference as reference_build
import run_strat01_engine_rung2b as runner


class Rung2BTests(unittest.TestCase):
    def test_reference_variant_is_explicit_and_keeps_default_target(self) -> None:
        default=reference_build.cmake_project(reference_build.SOURCE,reference_build.PINNED_LLAMA)
        extended=reference_build.cmake_project(reference_build.SOURCE,reference_build.PINNED_LLAMA,rung2b=True)
        self.assertIn("add_executable(strat01_engine_rung2a_reference",default)
        self.assertNotIn("STRAT01_RUNG2B=1",default)
        self.assertIn("add_executable(strat01_engine_rung2b_reference",extended)
        self.assertIn("STRAT01_RUNG2B=1",extended)

    def test_frozen_callback_boundary_is_present(self) -> None:
        source=runner.REFERENCE_SOURCE.read_text(encoding="utf-8")
        for name in runner.SHAPES:
            self.assertIn(f'"{name}"',source)
        self.assertIn("strat01_engine_rung2b_reference_manifest_v1",source)

    def test_metric_and_token_slice_controls(self) -> None:
        values=np.arange(24,dtype=np.float32)
        self.assertTrue(runner.judged(values,values,runner.GENERAL_LIMITS)["pass"])
        self.assertTrue(np.array_equal(runner.token7(values,[3,8]),values[-3:]))
        mutated=values.copy();mutated[-1]+=100
        self.assertFalse(runner.judged(mutated,values,runner.GENERAL_LIMITS)["pass"])

    def test_contained_file_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);outside=root.parent/"strat01-r2b-outside.bin";outside.write_bytes(b"1234")
            try:
                with self.assertRaises(runner.RunnerError):
                    runner.contained_file(root,str(outside),4,"escape")
            finally:
                outside.unlink(missing_ok=True)


if __name__=="__main__":
    unittest.main()
