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


if __name__ == "__main__":
    unittest.main()
