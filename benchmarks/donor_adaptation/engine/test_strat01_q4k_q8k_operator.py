from __future__ import annotations

import math
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.donor_adaptation.engine.build_strat01_q4k_q8k_oracle import build

ROOT = Path(__file__).resolve().parents[3]
PROJECT_SOURCE = ROOT / "benchmarks/phase60/strat01_q4k_q8k.h"
PROBE_SOURCE = Path(__file__).resolve().with_name("strat01_q4k_q8k_project_probe.c")


def f16(value: float) -> bytes:
    return struct.pack("<e", value)


def q4_fixture(kind: str) -> bytes:
    raw = bytearray(144)
    raw[0:2] = f16(0.03125)
    raw[2:4] = f16(0.015625)
    if kind == "low":
        raw[4:16] = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        raw[16:] = bytes((i * 29 + 7) & 0xFF for i in range(128))
    elif kind == "high":
        raw[4:16] = bytes([0xC1, 0x82, 0x43, 0x04, 0xE5, 0xA6, 0x67, 0x28, 0xF9, 0xBA, 0x7B, 0x3C])
        raw[16:] = bytes(((i * 17 + 3) & 15) | (((i * 11 + 5) & 15) << 4) for i in range(128))
    elif kind == "minimum":
        raw[4:16] = bytes([0x3F, 0x00, 0x15, 0x2A, 0x3E, 0x21, 0x0F, 0x30, 0xC3, 0x5A, 0xA5, 0x7E])
        raw[16:] = bytes([0xF0, 0x0F, 0x5A, 0xA5] * 32)
    else:
        raise ValueError(kind)
    return bytes(raw)


def repack_with_naive_scales(raw_bytes: bytes) -> bytes:
    """Plant the common error: treating every packed scale as low six bits."""
    raw = bytearray(raw_bytes)
    packed = raw[4:16]
    scales = [packed[j] & 63 for j in range(8)]
    minima = [packed[(j + 4) % 12] & 63 for j in range(8)]
    encoded = bytearray(12)
    for j in range(4):
        encoded[j] = scales[j] | ((scales[j + 4] >> 4) << 6)
        encoded[j + 4] = minima[j] | ((minima[j + 4] >> 4) << 6)
        encoded[j + 8] = (scales[j + 4] & 15) | ((minima[j + 4] & 15) << 4)
    raw[4:16] = encoded
    return bytes(raw)


def input_fixtures() -> dict[str, np.ndarray]:
    fixtures: dict[str, np.ndarray] = {}
    fixtures["zero"] = np.zeros(256, dtype="<f4")
    positive = np.linspace(-0.75, 0.75, 256, dtype=np.float32)
    positive[19] = np.float32(2.0)
    fixtures["positive_max"] = positive
    negative = positive.copy()
    negative[19] = np.float32(-2.0)
    fixtures["negative_max"] = negative
    tie_positive = np.zeros(256, dtype=np.float32)
    tie_positive[3], tie_positive[9] = np.float32(1.0), np.float32(-1.0)
    fixtures["tie_positive_first"] = tie_positive
    tie_negative = np.zeros(256, dtype=np.float32)
    tie_negative[3], tie_negative[9] = np.float32(-1.0), np.float32(1.0)
    fixtures["tie_negative_first"] = tie_negative
    half = np.zeros(256, dtype=np.float32)
    half[0] = np.float32(-127.0)
    half[1:13] = np.asarray([-3.5001, -3.5, -3.4999, -2.5001, -2.5, -2.4999,
                               2.4999, 2.5, 2.5001, 3.4999, 3.5, 3.5001], dtype=np.float32)
    fixtures["half_integer_neighborhoods"] = half
    index = np.arange(256, dtype=np.float32)
    fixtures["nonperiodic"] = (np.sin(index * np.float32(0.173)) * np.float32(1.7)
                                 + np.cos(index * np.float32(0.071)) * np.float32(0.31)).astype("<f4")
    return fixtures


class Q4KQ8KOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="strat01_q4k_q8k_")
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

    def run_pair(self, label: str, values: np.ndarray, q4: bytes) -> tuple[bytes, bytes, float, float]:
        directory = self.directory / label
        directory.mkdir()
        input_path, q4_path = directory / "input.f32le", directory / "q4.bin"
        input_path.write_bytes(np.asarray(values, dtype="<f4").tobytes())
        q4_path.write_bytes(q4)
        products = []
        for name, binary in (("project", self.probe), ("oracle", self.oracle)):
            q8_path, dot_path = directory / f"{name}.q8", directory / f"{name}.dot"
            completed = subprocess.run(
                [str(binary), str(input_path), str(q4_path), str(q8_path), str(dot_path)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            products.append((q8_path.read_bytes(), struct.unpack("<f", dot_path.read_bytes())[0]))
        return products[0][0], products[1][0], products[0][1], products[1][1]

    def test_source_has_no_ggml_dependency(self) -> None:
        source = PROJECT_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("#include \"ggml", source)
        self.assertIn("quantize_row_q8_K_ref", source)
        self.assertIn("ggml_vec_dot_q4_K_q8_K_generic", source)

    def test_fixed_population_matches_pinned_oracle(self) -> None:
        candidates: list[float] = []
        references: list[float] = []
        for input_name, values in input_fixtures().items():
            for q4_name in ("low", "high", "minimum"):
                project_q8, oracle_q8, project_dot, oracle_dot = self.run_pair(
                    f"{input_name}_{q4_name}", values, q4_fixture(q4_name)
                )
                self.assertEqual(project_q8, oracle_q8, f"Q8_K bytes differ for {input_name}/{q4_name}")
                candidates.append(project_dot)
                references.append(oracle_dot)
        candidate = np.asarray(candidates, dtype=np.float64)
        reference = np.asarray(references, dtype=np.float64)
        delta = candidate - reference
        nrmse = math.sqrt(float(np.mean(delta * delta))) / max(math.sqrt(float(np.mean(reference * reference))), 1e-12)
        normalized_max = float(np.max(np.abs(delta))) / max(float(np.max(np.abs(reference))), 1e-6)
        print(f"oracle population: cells={len(reference)} nrmse={nrmse:.9g} normalized_max={normalized_max:.9g} q8_bytes=exact")
        self.assertLessEqual(nrmse, 2e-6)
        self.assertLessEqual(normalized_max, 1e-5)

    def test_negative_controls_fire(self) -> None:
        values = input_fixtures()["nonperiodic"]
        project_q8, oracle_q8, baseline, oracle = self.run_pair("negative_baseline", values, q4_fixture("high"))
        self.assertEqual(project_q8, oracle_q8)
        self.assertAlmostEqual(baseline, oracle, delta=max(abs(oracle) * 1e-5, 2e-6))

        wrong_packed_interpretation = repack_with_naive_scales(q4_fixture("high"))
        _, _, wrong_scale, _ = self.run_pair("negative_wrong_scale", values, wrong_packed_interpretation)
        self.assertGreater(abs(wrong_scale - oracle), max(abs(oracle) * 1e-5, 2e-6))

        transposed = values.reshape(16, 16).T.copy().reshape(256)
        _, _, wrong_order, _ = self.run_pair("negative_transpose", transposed, q4_fixture("high"))
        self.assertGreater(abs(wrong_order - oracle), max(abs(oracle) * 1e-5, 2e-6))

        mutated_q8 = bytearray(project_q8)
        mutated_q8[2] ^= 0x80
        directory = self.directory / "negative_mutated_q8"
        directory.mkdir()
        q4_path, q8_path, dot_path = directory / "q4.bin", directory / "q8.bin", directory / "dot.f32le"
        q4_path.write_bytes(q4_fixture("high"))
        q8_path.write_bytes(mutated_q8)
        completed = subprocess.run(
            [str(self.probe), "--q8-dot", str(q4_path), str(q8_path), str(dot_path)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        mutated_dot = struct.unpack("<f", dot_path.read_bytes())[0]
        self.assertGreater(abs(mutated_dot - oracle), max(abs(oracle) * 1e-5, 2e-6))
        print(
            "negative controls: "
            f"wrong_scale_delta={abs(wrong_scale-oracle):.9g} "
            f"transpose_delta={abs(wrong_order-oracle):.9g} "
            f"q8_scale_delta={abs(mutated_dot-oracle):.9g}"
        )


if __name__ == "__main__":
    unittest.main()
