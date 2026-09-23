from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.donor_adaptation.engine.build_strat01_q4k_q8k_avx2_oracle import build

ROOT = Path(__file__).resolve().parents[3]
PROBE_SOURCE = Path(__file__).resolve().with_name("strat01_q4k_q8k_avx2_probe.c")
LENGTHS = (256, 512, 1536, 6144)


def f16(value: float) -> bytes:
    return struct.pack("<e", value)


def fixture(count: int) -> tuple[bytes, bytes]:
    q4 = bytearray()
    q8 = bytearray()
    for block in range(count // 256):
        raw = bytearray(144)
        raw[0:2] = f16((1.0, 0.03125, 4.0, 0.0078125)[block % 4])
        raw[2:4] = f16((0.5, 0.015625, 2.0, 0.00390625)[block % 4])
        raw[4:16] = bytes(((block * 37 + index * 29 + 3) & 0xFF) for index in range(12))
        raw[16:] = bytes(((block * 71 + index * 43 + 11) & 0xFF) for index in range(128))
        q4.extend(raw)

        values = [((block * 53 + index * 31 + 17) % 255) - 127 for index in range(256)]
        scale = (1.0, 0.0009765625, 16.0, 0.0625)[block % 4]
        q8.extend(struct.pack("<f", scale))
        q8.extend(struct.pack("<256b", *values))
        q8.extend(struct.pack("<16h", *(sum(values[index:index + 16]) for index in range(0, 256, 16))))
    return bytes(q4), bytes(q8)


class Q4KQ8KAVX2ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="strat01_q4k_q8k_avx2_")
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

    def test_active_is_bit_exact_and_generic_control_fires(self) -> None:
        generic_differences: list[int] = []
        for count in LENGTHS:
            directory = self.directory / str(count)
            directory.mkdir()
            q4_bytes, q8_bytes = fixture(count)
            q4, q8 = directory / "q4.bin", directory / "q8.bin"
            q4.write_bytes(q4_bytes)
            q8.write_bytes(q8_bytes)
            active = self.run_dot(self.probe, "active", count, q4, q8, directory / "active.f32le")
            oracle = self.run_dot(self.oracle, None, count, q4, q8, directory / "oracle.f32le")
            generic = self.run_dot(self.probe, "generic", count, q4, q8, directory / "generic.f32le")
            self.assertEqual(active, oracle, f"active kernel differs from pinned oracle at n={count}")
            if count > 256 and generic != oracle:
                generic_differences.append(count)
        self.assertTrue(generic_differences, "generic reduction failed to differ on every multi-block fixture")
        print(f"active=oracle bit-exact at {LENGTHS}; generic differs at {tuple(generic_differences)}")

    def test_short_read_and_invalid_length_reject(self) -> None:
        q4_bytes, q8_bytes = fixture(512)
        q4, q8, output = self.directory / "short.q4", self.directory / "short.q8", self.directory / "short.dot"
        q4.write_bytes(q4_bytes[:-1])
        q8.write_bytes(q8_bytes)
        short = subprocess.run(
            [str(self.probe), "active", "512", str(q4), str(q8), str(output)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(short.returncode, 0)
        invalid = subprocess.run(
            [str(self.probe), "active", "513", str(q4), str(q8), str(output)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()
