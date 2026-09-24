from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.donor_adaptation.engine.build_strat01_q6k_q8k_reference_generic_oracle import build
from benchmarks.donor_adaptation.engine.test_strat01_q6k_q8k_avx2_parity import LENGTHS, fixture

ROOT = Path(__file__).resolve().parents[3]
PROBE_SOURCE = Path(__file__).resolve().with_name("strat01_q6k_q8k_reference_generic_probe.c")


def rounding_fixture(count: int) -> tuple[bytes, bytes]:
    q6_raw, q8_raw = fixture(count)
    q6, q8 = bytearray(q6_raw), bytearray(q8_raw)
    q6_scales = (0.0317, 0.117, 0.00391, 1.73)
    q8_scales = (0.0137, 0.219, 1.37, 0.00317)
    for block in range(count // 256):
        q6_base, q8_base = block * 210, block * 292
        q6[q6_base + 208:q6_base + 210] = struct.pack("<e", q6_scales[block % 4])
        q8[q8_base:q8_base + 4] = struct.pack("<f", q8_scales[block % 4])
    return bytes(q6), bytes(q8)


def separation_fixture() -> tuple[bytes, bytes]:
    """Six-block seed whose three compiler/reduction arms are pairwise distinct."""
    state = 1

    def next_u32() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        return state

    q6, q8 = bytearray(), bytearray()
    for _ in range(6):
        raw = bytearray(210)
        for index in range(192):
            raw[index] = next_u32() & 0xFF
        for index in range(16):
            raw[192 + index] = next_u32() % 255 + 1
        raw[208:210] = struct.pack("<e", 1.0)
        q6.extend(raw)

        exponent = 120 + next_u32() % 15
        d_bits = (exponent << 23) | (next_u32() & 0x7FFFFF)
        values = [next_u32() % 255 - 127 for _ in range(256)]
        q8.extend(struct.pack("<I", d_bits))
        q8.extend(struct.pack("<256b", *values))
        q8.extend(struct.pack(
            "<16h", *(sum(values[index:index + 16]) for index in range(0, 256, 16))
        ))
    return bytes(q6), bytes(q8)


class Q6KQ8KReferenceGenericParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="strat01_q6k_q8k_reference_generic_")
        cls.directory = Path(cls.temporary.name)
        cls.oracle = build(cls.directory / "oracle-build")
        clang = shutil.which("clang")
        if not clang:
            raise unittest.SkipTest("clang is unavailable")
        cls.probe = cls.directory / "project_probe.exe"
        compiled = subprocess.run(
            [clang, "-std=c11", "-O3", "-mavx2", "-mfma", str(PROBE_SOURCE),
             "-o", str(cls.probe), "-lm"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if compiled.returncode:
            raise RuntimeError(compiled.stdout + compiled.stderr)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def run_dot(self, binary: Path, mode: str | None, count: int,
                q6: Path, q8: Path, output: Path) -> bytes:
        command = [str(binary)]
        if mode is not None:
            command.append(mode)
        command.extend([str(count), str(q6), str(q8), str(output)])
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return output.read_bytes()

    def test_candidate_is_bit_exact_and_both_controls_fire(self) -> None:
        for count in LENGTHS:
            directory = self.directory / str(count)
            directory.mkdir()
            q6_bytes, q8_bytes = rounding_fixture(count)
            q6, q8 = directory / "q6.bin", directory / "q8.bin"
            q6.write_bytes(q6_bytes)
            q8.write_bytes(q8_bytes)
            candidate = self.run_dot(
                self.probe, "candidate", count, q6, q8, directory / "candidate.f32le"
            )
            oracle = self.run_dot(
                self.oracle, None, count, q6, q8, directory / "oracle.f32le"
            )
            self.assertEqual(candidate, oracle,
                             f"candidate differs from baseline generic oracle at n={count}")

        directory = self.directory / "separation"
        directory.mkdir()
        q6_bytes, q8_bytes = separation_fixture()
        q6, q8 = directory / "q6.bin", directory / "q8.bin"
        q6.write_bytes(q6_bytes)
        q8.write_bytes(q8_bytes)
        candidate = self.run_dot(
            self.probe, "candidate", 1536, q6, q8, directory / "candidate.f32le"
        )
        oracle = self.run_dot(
            self.oracle, None, 1536, q6, q8, directory / "oracle.f32le"
        )
        generic = self.run_dot(
            self.probe, "avx-generic", 1536, q6, q8, directory / "avx_generic.f32le"
        )
        avx2 = self.run_dot(
            self.probe, "active-avx2", 1536, q6, q8, directory / "active_avx2.f32le"
        )
        self.assertEqual(candidate, oracle)
        self.assertEqual(len({candidate, generic, avx2}), 3,
                         "candidate, AVX-TU generic, and active AVX2 must be pairwise distinct")
        print(
            f"candidate=baseline-generic bit-exact at {LENGTHS}; "
            "six-block candidate/avx-generic/active-avx2 controls are pairwise distinct"
        )

    def test_short_read_and_invalid_mode_reject(self) -> None:
        q6_bytes, q8_bytes = rounding_fixture(512)
        q6, q8, output = self.directory / "short.q6", self.directory / "short.q8", self.directory / "short.dot"
        q6.write_bytes(q6_bytes[:-1])
        q8.write_bytes(q8_bytes)
        for command in (
            [str(self.probe), "candidate", "512", str(q6), str(q8), str(output)],
            [str(self.probe), "wrong", "512", str(q6), str(q8), str(output)],
        ):
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertNotEqual(completed.returncode, 0)

    def test_full_matrix_schedule_and_q8_population_are_exact(self) -> None:
        count, rows, batch = 512, 5, 3
        base_q6, _ = rounding_fixture(count)
        matrix_bytes = bytearray()
        for row in range(rows):
            value = bytearray(base_q6)
            value[(row * 97 + 13) % len(value)] ^= row + 1
            matrix_bytes.extend(value)
        values = [
            (((item * count + index) * 37 + 19) % 4093 - 2046) / 257.0
            for item in range(batch) for index in range(count)
        ]
        matrix = self.directory / "matrix.q6"
        inputs = self.directory / "matrix_input.f32le"
        matrix.write_bytes(matrix_bytes)
        inputs.write_bytes(struct.pack(f"<{len(values)}f", *values))
        c_q8, c_output = self.directory / "c.q8", self.directory / "c.f32le"
        oracle_q8 = self.directory / "oracle.q8"
        oracle_output = self.directory / "oracle.f32le"
        c_run = subprocess.run(
            [str(self.probe), "matrix", str(count), str(rows), str(batch),
             str(matrix), str(inputs), str(c_q8), str(c_output)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(c_run.returncode, 0, c_run.stdout + c_run.stderr)
        oracle_run = subprocess.run(
            [str(self.oracle), "matrix", str(count), str(rows), str(batch), "0",
             str(matrix), str(inputs), str(oracle_q8), str(oracle_output)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(oracle_run.returncode, 0, oracle_run.stdout + oracle_run.stderr)
        self.assertEqual(c_q8.read_bytes(), oracle_q8.read_bytes())
        self.assertEqual(c_output.read_bytes(), oracle_output.read_bytes())


if __name__ == "__main__":
    unittest.main()
