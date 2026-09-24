from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.donor_adaptation.engine.build_strat01_q6k_q8k_avx2_oracle import build

ROOT = Path(__file__).resolve().parents[3]
PROBE_SOURCE = Path(__file__).resolve().with_name("strat01_q6k_q8k_avx2_probe.c")
LENGTHS = (256, 512, 1536, 8960)


def fixture(count: int) -> tuple[bytes, bytes]:
    q6 = bytearray()
    q8 = bytearray()
    half_scales = (1.0, 0.03125, 4.0, 0.0078125)
    q8_scales = (1.0, 0.0009765625, 16.0, 0.0625)
    for block in range(count // 256):
        raw = bytearray(210)
        raw[:128] = bytes(((block * 71 + index * 43 + 11) & 0xFF) for index in range(128))
        raw[128:192] = bytes(((block * 37 + index * 29 + 3) & 0xFF) for index in range(64))
        raw[192:208] = struct.pack(
            "<16b", *(((block * 19 + index * 23 + 7) % 255) - 127 for index in range(16))
        )
        raw[208:210] = struct.pack("<e", half_scales[block % len(half_scales)])
        q6.extend(raw)

        values = [((block * 53 + index * 31 + 17) % 255) - 127 for index in range(256)]
        q8.extend(struct.pack("<f", q8_scales[block % len(q8_scales)]))
        q8.extend(struct.pack("<256b", *values))
        q8.extend(struct.pack(
            "<16h", *(sum(values[index:index + 16]) for index in range(0, 256, 16))
        ))
    return bytes(q6), bytes(q8)


class Q6KQ8KAVX2ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="strat01_q6k_q8k_avx2_")
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

    def run_dot(
        self, binary: Path, mode: str | None, count: int,
        q6: Path, q8: Path, output: Path,
    ) -> bytes:
        command = [str(binary)]
        if mode is not None:
            command.append(mode)
        command.extend([str(count), str(q6), str(q8), str(output)])
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return output.read_bytes()

    def test_active_is_bit_exact_and_generic_control_fires(self) -> None:
        generic_differences: list[int] = []
        for count in LENGTHS:
            directory = self.directory / str(count)
            directory.mkdir()
            q6_bytes, q8_bytes = fixture(count)
            q6, q8 = directory / "q6.bin", directory / "q8.bin"
            q6.write_bytes(q6_bytes)
            q8.write_bytes(q8_bytes)
            active = self.run_dot(
                self.probe, "active", count, q6, q8, directory / "active.f32le"
            )
            oracle = self.run_dot(
                self.oracle, None, count, q6, q8, directory / "oracle.f32le"
            )
            generic = self.run_dot(
                self.probe, "generic", count, q6, q8, directory / "generic.f32le"
            )
            self.assertEqual(active, oracle, f"active kernel differs from pinned oracle at n={count}")
            if count > 256 and generic != oracle:
                generic_differences.append(count)
        self.assertTrue(generic_differences, "generic negative control never differed")

    def test_mutation_control_and_rejections(self) -> None:
        count = 8960
        q6_bytes, q8_bytes = fixture(count)
        q6, q8 = self.directory / "mutation.q6", self.directory / "mutation.q8"
        q6.write_bytes(q6_bytes)
        q8.write_bytes(q8_bytes)
        baseline = self.run_dot(
            self.probe, "active", count, q6, q8, self.directory / "mutation.before.f32le"
        )
        mutated = bytearray(q6_bytes)
        mutated[17] ^= 1
        q6.write_bytes(mutated)
        after = self.run_dot(
            self.probe, "active", count, q6, q8, self.directory / "mutation.after.f32le"
        )
        self.assertNotEqual(baseline, after)

        q6.write_bytes(q6_bytes[:-1])
        short = subprocess.run(
            [str(self.probe), "active", str(count), str(q6), str(q8),
             str(self.directory / "short.dot")],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(short.returncode, 0)
        invalid = subprocess.run(
            [str(self.probe), "active", "513", str(q6), str(q8),
             str(self.directory / "invalid.dot")],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(invalid.returncode, 0)

    def test_full_matrix_schedule_and_q8_population_are_exact(self) -> None:
        count, rows, batch = 512, 5, 3
        base_q6, _ = fixture(count)
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
        oracle_q8, oracle_output = self.directory / "oracle.q8", self.directory / "oracle.f32le"
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
