"""Focused offline regressions for the STRAT-01 scalar rung-1 apparatus."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENGINE_C = ROOT / "benchmarks" / "phase60" / "engine.c"


class Rung1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        clang = shutil.which("clang")
        if not clang:
            raise unittest.SkipTest("clang is required for the rung-1 apparatus")
        cls.tmp = tempfile.TemporaryDirectory(prefix="strat01_engine_rung1_")
        cls.work = Path(cls.tmp.name)
        cls.engine = cls.work / "engine.exe"
        subprocess.run([clang, "-std=c11", "-O3", "-mavx2", "-mfma", str(ENGINE_C), "-o", str(cls.engine), "-lm"],
                       cwd=ROOT, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(self.engine), *args], cwd=self.work, capture_output=True, text=True)

    def test_legacy_kernel_selftest_survives(self) -> None:
        run = self._run("--kselftest")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("PASS", run.stdout)

    def test_synthetic_layout_traps_pass(self) -> None:
        run = self._run("--strat01-gguf-rung1-selftest")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("PASS", run.stderr)

    def test_rung1_cli_requires_exact_early_shape(self) -> None:
        for arguments in (("--strat01-gguf-rung1",),
                          ("--strat01-gguf-rung1", "not-a-file.gguf", "--json", "out"),
                          ("--strat01-gguf-rung1", "not-a-file.gguf", "--out-dir", "out", "extra")):
            with self.subTest(arguments=arguments):
                run = self._run(*arguments)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("usage:", run.stderr)

    def test_unrecognized_artifact_fails_before_legacy_loader(self) -> None:
        out_dir = self.work / "out"
        out_dir.mkdir()
        fixture = self.work / "substituted.gguf"
        fixture.write_bytes(b"GGUF" + b"\\0" * 64)
        run = self._run("--strat01-gguf-rung1", str(fixture), "--out-dir", str(out_dir))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("rung-1 refused", run.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
