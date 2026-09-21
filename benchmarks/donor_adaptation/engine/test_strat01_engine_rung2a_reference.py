#!/usr/bin/env python3
"""Model-free tests for the STRAT-01 Rung-2A pinned-reference apparatus."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_strat01_engine_rung2a_reference as build_mod


class Rung2AReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="strat01-rung2a-reference-test-")
        cls.binary = build_mod.build(Path(cls._temporary.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_cmake_project_is_pinned_and_links_llama(self) -> None:
        project = build_mod.cmake_project(build_mod.SOURCE, build_mod.PINNED_LLAMA)
        self.assertIn('add_subdirectory("C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4" llama-cpp)', project)
        self.assertIn("target_link_libraries(strat01_engine_rung2a_reference PRIVATE llama)", project)
        self.assertNotIn("phase60", project)

    def test_pinned_source_is_clean_at_exact_head(self) -> None:
        build_mod.verify_pinned_llama()

    def test_requested_and_pinned_resolved_context_are_both_explicit(self) -> None:
        source = build_mod.SOURCE.read_text(encoding="utf-8")
        self.assertIn("constexpr uint32_t kRequestedCtx = 8;", source)
        self.assertIn("constexpr uint32_t kResolvedCtx = 256;", source)
        self.assertIn('\\"requested_n_ctx\\"', source)
        self.assertIn('\\"resolved_n_ctx\\"', source)
        self.assertIn("validate_resolved_context_dimensions(llama_n_ctx(ctx),llama_n_batch(ctx),llama_n_ubatch(ctx))", source)

    def test_build_and_model_free_cpp_self_test(self) -> None:
        completed = subprocess.run([str(self.binary), "--self-test"], text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_malformed_cli_is_refused_without_model(self) -> None:
        completed = subprocess.run([str(self.binary), "--arm", "wrong"], text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("VOID: --arm must be prefill8 or cached7p1", completed.stderr)


if __name__ == "__main__":
    unittest.main()
