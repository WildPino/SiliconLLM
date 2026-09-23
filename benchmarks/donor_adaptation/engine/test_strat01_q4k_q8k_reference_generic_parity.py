from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.donor_adaptation.engine.build_strat01_q4k_q8k_reference_generic_oracle import build
from benchmarks.donor_adaptation.engine.test_strat01_q4k_q8k_avx2_parity import LENGTHS, fixture

ROOT = Path(__file__).resolve().parents[3]
PROBE_SOURCE = Path(__file__).resolve().with_name("strat01_q4k_q8k_reference_generic_probe.c")


def rounding_fixture(count: int) -> tuple[bytes, bytes]:
    q4_raw, q8_raw = fixture(count)
    q4, q8 = bytearray(q4_raw), bytearray(q8_raw)
    q4_scales = (0.0317, 0.117, 0.00391, 1.73)
    q4_minima = (0.0193, 0.071, 0.00217, 0.83)
    q8_scales = (0.0137, 0.219, 1.37, 0.00317)
    for block in range(count // 256):
        q4_base, q8_base = block * 144, block * 292
        q4[q4_base:q4_base + 2] = struct.pack("<e", q4_scales[block % 4])
        q4[q4_base + 2:q4_base + 4] = struct.pack("<e", q4_minima[block % 4])
        q8[q8_base:q8_base + 4] = struct.pack("<f", q8_scales[block % 4])
    return bytes(q4), bytes(q8)


class Q4KQ8KReferenceGenericParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="strat01_q4k_q8k_reference_generic_")
        cls.directory = Path(cls.temporary.name)
        cls.oracle = build(cls.directory / "oracle-build")
        clang = shutil.which("clang")
        if not clang:
            raise unittest.SkipTest("clang is unavailable")
        cls.probe = cls.directory / "project_probe.exe"
        compiled = subprocess.run(
            [clang, "-std=c11", "-O3", "-mavx2", "-mfma", str(PROBE_SOURCE), "-o", str(cls.probe), "-lm"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if compiled.returncode:
            raise RuntimeError(compiled.stdout + compiled.stderr)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def run_dot(self, binary: Path, mode: str | None, count: int, q4: Path, q8: Path, output: Path) -> bytes:
        command = [str(binary)]
        if mode is not None:
            command.append(mode)
        command.extend([str(count), str(q4), str(q8), str(output)])
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return output.read_bytes()

    def test_candidate_is_bit_exact_and_both_controls_fire(self) -> None:
        generic_differences: list[int] = []
        avx2_differences: list[int] = []
        for count in LENGTHS:
            directory = self.directory / str(count)
            directory.mkdir()
            q4_bytes, q8_bytes = rounding_fixture(count)
            q4, q8 = directory / "q4.bin", directory / "q8.bin"
            q4.write_bytes(q4_bytes)
            q8.write_bytes(q8_bytes)
            candidate = self.run_dot(self.probe, "candidate", count, q4, q8, directory / "candidate.f32le")
            oracle = self.run_dot(self.oracle, None, count, q4, q8, directory / "oracle.f32le")
            generic = self.run_dot(self.probe, "avx-generic", count, q4, q8, directory / "avx_generic.f32le")
            avx2 = self.run_dot(self.probe, "active-avx2", count, q4, q8, directory / "active_avx2.f32le")
            self.assertEqual(candidate, oracle, f"candidate differs from baseline generic oracle at n={count}")
            if count > 256 and generic != oracle:
                generic_differences.append(count)
            if count > 256 and avx2 != oracle:
                avx2_differences.append(count)
        self.assertTrue(generic_differences)
        self.assertTrue(avx2_differences)
        print(
            f"candidate=baseline-generic bit-exact at {LENGTHS}; "
            f"avx-generic differs at {tuple(generic_differences)}; "
            f"active-avx2 differs at {tuple(avx2_differences)}"
        )

    def test_short_read_and_invalid_mode_reject(self) -> None:
        q4_bytes, q8_bytes = rounding_fixture(512)
        q4, q8, output = self.directory / "short.q4", self.directory / "short.q8", self.directory / "short.dot"
        q4.write_bytes(q4_bytes[:-1])
        q8.write_bytes(q8_bytes)
        for command in (
            [str(self.probe), "candidate", "512", str(q4), str(q8), str(output)],
            [str(self.probe), "wrong", "512", str(q4), str(q8), str(output)],
        ):
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
