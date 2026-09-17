"""Synthetic exactness checks for the isolated STRAT-02 W4 BF16 v2 codec."""
from __future__ import annotations

import unittest

import numpy as np

from benchmarks.donor_adaptation.density import strat02_w4_bf16_codec as codec


def _tile_blob(source: np.ndarray, batches: tuple[int, ...], *, rows: int, groups: int) -> bytes:
    group_count = source.shape[1] // codec.GROUP_SIZE
    scale_bits = np.empty((source.shape[0], group_count), dtype=np.uint16)
    packed = np.empty((source.shape[0], group_count * 64), dtype=np.uint8)

    def source_batches():
        offset = 0
        for count in batches:
            yield source[offset:offset + count]
            offset += count
        assert offset == source.shape[0]

    for tile in codec.iter_encode_w4_bf16_tiles(source_batches(), source.shape[1], max_rows_per_tile=rows, max_groups_per_tile=groups):
        tile_rows, tile_groups = tile.scale_bits.shape
        scale_bits[tile.row_offset:tile.row_offset + tile_rows, tile.group_offset:tile.group_offset + tile_groups] = tile.scale_bits
        packed[tile.row_offset:tile.row_offset + tile_rows, tile.group_offset * 64:(tile.group_offset + tile_groups) * 64] = tile.packed_codes
    return codec.w4_bf16_to_blob(codec.W4BF16Encoded(tuple(source.shape), scale_bits, packed))


class W4BF16CodecTest(unittest.TestCase):
    def test_bf16_rne_bits_match_cpu_torch_for_boundary_and_nonfinite_values(self) -> None:
        # Exact F32 layouts cover zero/subnormal, two halfway directions, carry,
        # largest finite (which rounds to BF16 infinity), and all nonfinites.
        words = np.array([0x00000000, 0x80000000, 0x00000001, 0x007FFFFF,
                          0x3F808000, 0x3F818000, 0x3FFF8000, 0x7F7FFFFF,
                          0x7F800000, 0xFF800000, 0x7FC01234], dtype="<u4")
        values = words.view("<f4")
        self.assertTrue(np.array_equal(codec.bf16_rne_bits(values), codec._torch_bf16_bits(values)))
        self.assertEqual(int(codec.bf16_rne_bits_scalar(values[4])), int(codec.bf16_rne_bits(values)[4]))

    def test_scalar_and_tile_produce_identical_wire_bytes_across_tilings(self) -> None:
        rng = np.random.default_rng(20260917)
        source = rng.normal(size=(11, 640)).astype(np.float32)
        source[0, :128] = 0
        source[1, 128:256] = np.array([7.0, 0.5, -0.5, 1.5, -1.5] + [0.0] * 123, dtype=np.float32)
        source[2, 256:384] = np.linspace(-1.0e-20, 1.0e-20, 128, dtype=np.float32)
        reference = codec.w4_bf16_to_blob(codec.encode_w4_bf16(source))
        for batches, rows, groups in (((11,), 32, 32), ((1,) * 11, 1, 1), ((3, 5, 3), 2, 3)):
            with self.subTest(batches=batches, rows=rows, groups=groups):
                self.assertEqual(_tile_blob(source, batches, rows=rows, groups=groups), reference)

    def test_wire_layout_and_decode_fail_closed(self) -> None:
        source = np.zeros((1, 128), dtype=np.float32)
        encoded = codec.encode_w4_bf16(source)
        self.assertEqual(codec.w4_bf16_to_blob(encoded)[:2], b"\x00\x00")
        bad_scale_with_code = b"\x00\x00" + bytes([1]) + bytes(63)
        with self.assertRaisesRegex(ValueError, "zero BF16 scale"):
            codec.w4_bf16_from_blob(bad_scale_with_code, source.shape)
        bad_reserved_nibble = b"\x80\x3f" + bytes([8]) + bytes(63)
        with self.assertRaisesRegex(ValueError, "-8"):
            codec.w4_bf16_from_blob(bad_reserved_nibble, source.shape)

    def test_nonzero_unrepresentable_bf16_scale_fails_closed(self) -> None:
        source = np.full((1, 128), np.nextafter(np.float32(0), np.float32(1)), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "invalid BF16 candidate scale"):
            codec.encode_w4_bf16(source)


if __name__ == "__main__":
    unittest.main()
