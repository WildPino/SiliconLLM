"""Offline STRAT-01 engine rung-0 apparatus tests.

The production engine has no identity override.  These tests compile a second,
temporary binary with ``STRAT01_ENABLE_TEST_IDENTITY_OVERRIDE`` and constants
for one generated fixture.  The override is compile-time only: no CLI option,
environment variable, or production build can substitute a file identity.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENGINE_C = ROOT / "benchmarks" / "phase60" / "engine.c"
MAGIC = 0x46554747
Q5_0, Q4_K, Q6_K, F32 = 6, 12, 14, 0

REQUIRED = {
    "general.quantization_version": 2,
    "general.file_type": 15,
    "deepseek2.vocab_size": 128256,
    "deepseek2.embedding_length": 1536,
    "deepseek2.block_count": 26,
    "deepseek2.feed_forward_length": 8960,
    "deepseek2.expert_feed_forward_length": 1280,
    "deepseek2.expert_count": 64,
    "deepseek2.expert_used_count": 4,
    "deepseek2.expert_shared_count": 1,
    "deepseek2.attention.head_count": 32,
    "deepseek2.attention.head_count_kv": 1,
    "deepseek2.attention.key_length": 576,
    "deepseek2.attention.value_length": 512,
    "deepseek2.attention.key_length_mla": 192,
    "deepseek2.attention.value_length_mla": 192,
    "deepseek2.attention.kv_lora_rank": 512,
    "deepseek2.rope.dimension_count": 64,
    "tokenizer.ggml.bos_token_id": 1,
    "tokenizer.ggml.eos_token_id": 2,
}


def _s(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _align(value: int, alignment: int = 32) -> int:
    return (value + alignment - 1) // alignment * alignment


def _span(type_id: int, dims: tuple[int, ...]) -> int:
    elements = 1
    for dim in dims:
        elements *= dim
    block_bytes = {F32: (1, 4), Q5_0: (32, 22), Q4_K: (256, 144), Q6_K: (256, 210)}
    block, byte_count = block_bytes[type_id]
    if elements % block:
        return 0
    return elements // block * byte_count


def _metadata() -> list[tuple[str, int, object]]:
    data: list[tuple[str, int, object]] = [("general.architecture", 8, "deepseek2")]
    data.extend((key, 4, value) for key, value in REQUIRED.items())
    # Mirror the accepted file's important structural variety: alignment is
    # absent (therefore GGUF's default 32 applies), while float, bool, string,
    # string-array, and integer-array values are all present.
    data.extend((
        ("deepseek2.rope.scaling.type", 8, "yarn"),
        ("deepseek2.rope.scaling.factor", 6, 64.0),
        ("deepseek2.expert_weights_norm", 7, True),
        ("tokenizer.ggml.tokens", 9, (8, ["a", "b"])),
        ("tokenizer.ggml.token_type", 9, (5, [1, 1])),
    ))
    # The accepted artifact has 47 metadata entries.
    data.extend((f"test.padding.{index}", 4, index) for index in range(21))
    assert len(data) == 47
    return data


def _write_metadata(parts: list[bytes], metadata: list[tuple[str, int, object]]) -> None:
    for key, type_id, value in metadata:
        parts.extend((_s(key), struct.pack("<I", type_id)))
        if type_id == 8:
            parts.append(_s(str(value)))
        elif type_id == 4:
            parts.append(struct.pack("<I", int(value)))
        elif type_id == 6:
            parts.append(struct.pack("<f", float(value)))
        elif type_id == 7:
            parts.append(struct.pack("<?", bool(value)))
        elif type_id == 9:
            element_type, elements = value
            parts.extend((struct.pack("<I", element_type), struct.pack("<Q", len(elements))))
            if element_type == 8:
                parts.extend(_s(str(element)) for element in elements)
            elif element_type == 5:
                parts.extend(struct.pack("<i", int(element)) for element in elements)
            else:
                raise AssertionError(f"unsupported test array element type {element_type}")
        else:
            # Deliberately supports an invalid raw type for the fail-closed
            # parser control below; the production parser must reject it.
            parts.append(struct.pack("<I", int(value)))


def write_fixture(path: Path, *, duplicate_name: bool = False, out_of_bounds: bool = False,
                  invalid_block: bool = False, unsupported_type: bool = False,
                  unsupported_metadata_type: bool = False) -> None:
    tensors: list[dict[str, object]] = []
    for index in range(129):
        tensors.append({"name": f"blk.0.f32.{index}", "type": F32, "dims": (1,)})
    for index in range(26):
        tensors.append({"name": f"blk.1.q5.{index}", "type": Q5_0, "dims": (32,)})
    for index in range(233):
        tensors.append({"name": f"blk.2.q4.{index}", "type": Q4_K, "dims": (256,)})
    for index in range(26):
        tensors.append({"name": f"blk.3.q6.{index}", "type": Q6_K, "dims": (256,)})
    assert len(tensors) == 414
    if duplicate_name:
        tensors[1]["name"] = tensors[0]["name"]
    if invalid_block:
        tensors[155]["dims"] = (31,)
    if unsupported_type:
        tensors[0]["type"] = 1

    payload_size = 0
    for tensor in tensors:
        type_id = int(tensor["type"])
        dims = tuple(tensor["dims"])
        span = _span(type_id, dims) if type_id in (F32, Q5_0, Q4_K, Q6_K) else 4
        tensor["span"] = span
        tensor["offset"] = _align(payload_size)
        payload_size = int(tensor["offset"]) + span
    if out_of_bounds:
        tensors[0]["offset"] = payload_size + 32

    metadata = _metadata()
    if unsupported_metadata_type:
        key, _, _ = metadata[-1]
        metadata[-1] = (key, 13, 0)  # GGUF v3 has value-type codes 0..12 only.
    parts = [struct.pack("<IIQQ", MAGIC, 3, len(tensors), len(metadata))]
    _write_metadata(parts, metadata)
    for tensor in tensors:
        dims = tuple(tensor["dims"])
        parts.extend((_s(str(tensor["name"])), struct.pack("<I", len(dims))))
        parts.extend(struct.pack("<Q", dim) for dim in dims)
        parts.extend((struct.pack("<I", int(tensor["type"])), struct.pack("<Q", int(tensor["offset"])) ))
    header = b"".join(parts)
    data_offset = _align(len(header))
    payload = bytearray(payload_size)
    path.write_bytes(header + b"\0" * (data_offset - len(header)) + payload)


def write_invalid_string_length(path: Path) -> None:
    path.write_bytes(struct.pack("<IIQQQ", MAGIC, 3, 1, 1, 999999))


class Rung0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clang = shutil.which("clang")
        if not cls.clang:
            raise unittest.SkipTest("clang is required for the rung-0 apparatus")
        cls.tmp = tempfile.TemporaryDirectory(prefix="strat01_engine_rung0_")
        cls.work = Path(cls.tmp.name)
        cls.fixture = cls.work / "valid.gguf"
        write_fixture(cls.fixture)
        cls.fixture_size = cls.fixture.stat().st_size
        cls.fixture_sha = hashlib.sha256(cls.fixture.read_bytes()).hexdigest()
        cls.production = cls.work / "engine_production.exe"
        cls.override = cls.work / "engine_fixture_identity.exe"
        cls._compile(cls.production)
        cls._compile(cls.override, size=cls.fixture_size, digest=cls.fixture_sha)
        print("STRAT-01 test identity override: compile-time fixture size/SHA only; production CLI has no override.")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    @classmethod
    def _compile(cls, output: Path, *, size: int | None = None, digest: str | None = None) -> None:
        command = [cls.clang, "-std=c11", "-O3", "-mavx2", "-mfma", str(ENGINE_C), "-o", str(output), "-lm"]
        if size is not None and digest is not None:
            command.extend(("-DSTRAT01_ENABLE_TEST_IDENTITY_OVERRIDE", f"-DSTRAT01_TEST_EXPECTED_SIZE={size}",
                            f'-DSTRAT01_TEST_EXPECTED_SHA256="{digest}"'))
        subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)

    def _inspect(self, fixture: Path, name: str, *, executable: Path | None = None) -> subprocess.CompletedProcess[str]:
        result = self.work / f"{name}.json"
        return subprocess.run([str(executable or self.override), "--strat01-gguf-inspect", str(fixture), "--json", str(result)],
                              cwd=self.work, capture_output=True, text=True)

    def test_legacy_kernel_selftest_survives(self) -> None:
        run = subprocess.run([str(self.production), "--kselftest"], cwd=self.work, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("PASS", run.stdout)

    def test_valid_fixture_is_complete_and_cwd_independent(self) -> None:
        run = self._inspect(self.fixture, "valid")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        report = json.loads((self.work / "valid.json").read_text(encoding="utf-8"))
        self.assertEqual(report["gate_status"], "PASS_ENGINE_RUNG0")
        self.assertEqual(report["byte_size"], self.fixture_size)
        self.assertEqual(report["sha256"], self.fixture_sha)
        self.assertRegex(report["engine_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(report["inspector_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(report["metadata_count"], 47)
        self.assertEqual(report["tensor_count"], 414)
        self.assertEqual(report["type_census"], {"F32": 129, "Q5_0": 26, "Q4_K": 233, "Q6_K": 26})
        self.assertTrue(report["mtp_excluded"])
        self.assertEqual(len(report["tensors"]), 414)
        self.assertTrue(all(isinstance(tensor["offset"], int) and isinstance(tensor["byte_span"], int)
                            for tensor in report["tensors"]))

    def test_malformed_controls_are_refused(self) -> None:
        cases = {
            "truncated_header": lambda p: p.write_bytes(b"GGUF"),
            "invalid_string_length": write_invalid_string_length,
            "duplicate_tensor_name": lambda p: write_fixture(p, duplicate_name=True),
            "out_of_bounds_tensor": lambda p: write_fixture(p, out_of_bounds=True),
            "wrong_block_divisibility": lambda p: write_fixture(p, invalid_block=True),
            "unsupported_tensor_type": lambda p: write_fixture(p, unsupported_type=True),
            "unsupported_metadata_type": lambda p: write_fixture(p, unsupported_metadata_type=True),
        }
        for name, writer in cases.items():
            with self.subTest(name=name):
                fixture = self.work / f"{name}.gguf"
                writer(fixture)
                run = self._inspect(fixture, name)
                self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
                report = json.loads((self.work / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(report["gate_status"], "FAIL_ENGINE_RUNG0")

    def test_exact_copy_passes_and_one_byte_change_fails_hash(self) -> None:
        copy = self.work / "exact_copy.gguf"
        copy.write_bytes(self.fixture.read_bytes())
        self.assertEqual(self._inspect(copy, "exact_copy").returncode, 0)
        changed = self.work / "one_byte_changed.gguf"
        bytes_changed = bytearray(copy.read_bytes())
        bytes_changed[-1] ^= 1
        changed.write_bytes(bytes_changed)
        run = self._inspect(changed, "one_byte_changed")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("SHA-256", run.stderr)

    def test_wrong_compile_time_identity_is_refused(self) -> None:
        wrong_size = self.work / "engine_wrong_size.exe"
        self._compile(wrong_size, size=self.fixture_size + 1, digest=self.fixture_sha)
        run = self._inspect(self.fixture, "wrong_size", executable=wrong_size)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("byte size", run.stderr)
        wrong_hash = self.work / "engine_wrong_hash.exe"
        self._compile(wrong_hash, size=self.fixture_size, digest="0" * 64)
        run = self._inspect(self.fixture, "wrong_hash", executable=wrong_hash)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("SHA-256", run.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
