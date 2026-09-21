"""Model-free regressions for the STRAT-01 engine rung-2A C apparatus."""

from __future__ import annotations

import subprocess
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENGINE_C = ROOT / "benchmarks" / "phase60" / "engine.c"


class Rung2ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        clang = shutil.which("clang")
        if not clang:
            raise unittest.SkipTest("clang is required for the rung-2A apparatus")
        cls.tmp = tempfile.TemporaryDirectory(prefix="strat01_engine_rung2a_")
        cls.work = Path(cls.tmp.name)
        cls.engine = cls.work / "engine.exe"
        cls.clang = clang
        cls.compile = subprocess.run(
            [
                clang,
                "-std=c11",
                "-O3",
                "-mavx2",
                "-mfma",
                str(ENGINE_C),
                "-o",
                str(cls.engine),
                "-lm",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.engine), *args],
            cwd=self.work,
            capture_output=True,
            text=True,
        )

    def test_model_free_semantic_and_negative_controls(self) -> None:
        run = self._run("--strat01-gguf-rung2a-selftest")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("model-free selftest: PASS", run.stderr)

    def test_rung2a_cli_requires_exact_early_shape(self) -> None:
        malformed = (
            ("--strat01-gguf-rung2a",),
            ("--strat01-gguf-rung2a", "x.gguf", "--json", "out"),
            ("--strat01-gguf-rung2a", "x.gguf", "--out-dir", "out", "extra"),
            ("--strat01-gguf-rung2a-selftest", "extra"),
        )
        for arguments in malformed:
            with self.subTest(arguments=arguments):
                run = self._run(*arguments)
                self.assertNotEqual(run.returncode, 0)
                if arguments[0] == "--strat01-gguf-rung2a":
                    self.assertIn("usage:", run.stderr)

    def test_wrong_artifact_is_refused_before_gguf_parse(self) -> None:
        out_dir = self.work / "wrong_artifact_out"
        out_dir.mkdir()
        fixture = self.work / "wrong.gguf"
        fixture.write_bytes(b"this is intentionally not a GGUF payload")
        run = self._run(
            "--strat01-gguf-rung2a",
            str(fixture),
            "--out-dir",
            str(out_dir),
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("rung-2A refused", run.stderr)
        self.assertNotIn("GGUF magic mismatch", run.stderr)
        report = (out_dir / "strat01_rung2a.json").read_text(encoding="utf-8")
        self.assertIn('"c_state":"ENGINE_RUNG2A_C_FAILURE"', report)
        self.assertNotIn("ENGINE_OUTPUT_READY_PENDING_REFERENCE", report)

    def test_prior_rung1_and_legacy_selftests_still_dispatch(self) -> None:
        rung1 = self._run("--strat01-gguf-rung1-selftest")
        self.assertEqual(rung1.returncode, 0, rung1.stdout + rung1.stderr)
        self.assertIn("PASS", rung1.stderr)
        legacy = self._run("--kselftest")
        self.assertEqual(legacy.returncode, 0, legacy.stdout + legacy.stderr)
        self.assertIn("PASS", legacy.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
