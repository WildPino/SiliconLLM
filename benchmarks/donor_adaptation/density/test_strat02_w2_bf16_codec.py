"""Synthetic byte-exact checks for the STRAT-02E W2 BF16 codec."""
from __future__ import annotations

import unittest

import numpy as np

from benchmarks.donor_adaptation.density import strat02_w2_bf16_codec as codec


def _tiled(source: np.ndarray, batches: tuple[int, ...], rows: int, groups: int) -> codec.W2BF16Encoded:
    count = source.shape[1] // codec.GROUP_SIZE
    a = np.empty((source.shape[0], count), dtype=np.uint16)
    b = np.empty_like(a)
    packed = np.empty((source.shape[0], count * codec.CODE_BYTES_PER_GROUP), dtype=np.uint8)
    offset = 0
    for size in batches:
        batch = source[offset:offset + size]
        for tile in codec.iter_encode_w2_bf16_tiles((batch,), source.shape[1], max_rows_per_tile=rows, max_groups_per_tile=groups):
            local_rows, local_groups = tile.a_bits.shape
            target_rows = slice(offset + tile.row_offset, offset + tile.row_offset + local_rows)
            a[target_rows, tile.group_offset:tile.group_offset + local_groups] = tile.a_bits
            b[target_rows, tile.group_offset:tile.group_offset + local_groups] = tile.b_bits
            packed[target_rows, tile.group_offset * codec.CODE_BYTES_PER_GROUP:(tile.group_offset + local_groups) * codec.CODE_BYTES_PER_GROUP] = tile.packed_codes
        offset += size
    return codec.W2BF16Encoded(tuple(source.shape), a, b, packed)


class W2BF16CodecTest(unittest.TestCase):
    def test_scalar_and_bounded_tiles_are_byte_exact(self) -> None:
        rng = np.random.default_rng(20260917)
        source = rng.normal(size=(9, 640)).astype(np.float32)
        source[0, :128] = 0
        source[1, 128:256] = np.asarray([-9, -3, -1, 0, 1, 3, 9] + [0] * 121, dtype=np.float32)
        expected = codec.w2_bf16_to_blob(codec.encode_w2_bf16(source))
        for batches, rows, groups in (((9,), 16, 16), ((1,) * 9, 1, 1), ((2, 4, 3), 2, 3)):
            with self.subTest(batches=batches, rows=rows, groups=groups):
                self.assertEqual(codec.w2_bf16_to_blob(_tiled(source, batches, rows, groups)), expected)

    def test_tiny_large_and_nonfinite_inputs_are_explicit(self) -> None:
        source = np.zeros((2, 128), dtype=np.float32)
        source[0] = np.linspace(-1.0e-20, 1.0e-20, 128, dtype=np.float32)
        source[1] = np.linspace(-1.0e30, 1.0e30, 128, dtype=np.float32)
        expected = codec.w2_bf16_to_blob(codec.encode_w2_bf16(source))
        self.assertEqual(codec.w2_bf16_to_blob(_tiled(source, (1, 1), 1, 1)), expected)
        nonfinite = source[:1].copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            codec.encode_w2_bf16(nonfinite)
        unrepresentable = np.full((1, 128), np.nextafter(np.float32(0), np.float32(1)), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "invalid BF16 codebook"):
            codec.encode_w2_bf16(unrepresentable)

    def test_ordered_cumsum_tile_fit_matches_scalar_on_random_and_adversarial_groups(self) -> None:
        rng = np.random.default_rng(20260918)
        random = rng.normal(size=(11, 640)).astype(np.float32)
        alternating = np.resize(np.asarray([1.0e30, 1.0e-30, -1.0e30, -1.0e-30], dtype=np.float32), (1, 640))
        ties = np.resize(np.asarray([-4, -1, 0, 1, 4, -4, -1, 0, 1, 4], dtype=np.float32), (1, 640))
        source = np.concatenate((random, alternating, ties), axis=0)
        expected = codec.w2_bf16_to_blob(codec.encode_w2_bf16(source))
        for batches, rows, groups in (((13,), 16, 16), ((1,) * 13, 1, 1), ((4, 3, 6), 3, 2)):
            with self.subTest(batches=batches, rows=rows, groups=groups):
                self.assertEqual(codec.w2_bf16_to_blob(_tiled(source, batches, rows, groups)), expected)

    def test_vectorized_packed_decode_matches_scalar_code_order(self) -> None:
        rng = np.random.default_rng(20260919)
        source = rng.normal(size=(5, 384)).astype(np.float32)
        encoded = codec.encode_w2_bf16(source)
        decoded = codec.dequantize_w2_bf16(encoded)
        expected = np.empty_like(decoded)
        for row in range(source.shape[0]):
            for group in range(source.shape[1] // codec.GROUP_SIZE):
                packed = encoded.packed_codes[row, group * codec.CODE_BYTES_PER_GROUP:(group + 1) * codec.CODE_BYTES_PER_GROUP]
                expected[row, group * codec.GROUP_SIZE:(group + 1) * codec.GROUP_SIZE] = codec._decoded_group(
                    encoded.a_bits[row, group], encoded.b_bits[row, group], codec.unpack_w2_codes_scalar(packed))
        self.assertTrue(np.array_equal(decoded, expected))

    def test_wire_order_is_a_then_b_then_low_bits_first_codes(self) -> None:
        a = codec.bf16_rne_bits_scalar(np.float32(4))
        b = codec.bf16_rne_bits_scalar(np.float32(1))
        values = codec.W2BF16Encoded((1, 128), np.asarray([[a]], dtype=np.uint16), np.asarray([[b]], dtype=np.uint16),
                                     np.asarray([[0xE4] + [0] * 31], dtype=np.uint8))
        blob = codec.w2_bf16_to_blob(values)
        self.assertEqual(blob[:2], np.asarray([a], dtype="<u2").tobytes())
        self.assertEqual(blob[2:4], np.asarray([b], dtype="<u2").tobytes())
        self.assertEqual(blob[4], 0xE4)
        self.assertEqual(codec.unpack_w2_codes_scalar(blob[4:5]).tolist(), [0, 1, 2, 3])

    def test_packed_decode_and_invalid_groups_fail_closed(self) -> None:
        zero = codec.encode_w2_bf16(np.zeros((1, 128), dtype=np.float32))
        decoded = codec.dequantize_w2_bf16(codec.w2_bf16_from_blob(codec.w2_bf16_to_blob(zero), (1, 128)))
        self.assertTrue(np.array_equal(decoded, np.zeros((1, 128), dtype=np.float32)))
        bad_zero = b"\x00\x00\x00\x00" + bytes([0]) + bytes(31)
        with self.assertRaisesRegex(ValueError, "zero group"):
            codec.w2_bf16_from_blob(bad_zero, (1, 128))
        a, b = codec.bf16_rne_bits_scalar(np.float32(1)), codec.bf16_rne_bits_scalar(np.float32(2))
        bad_order = np.asarray([a, b], dtype="<u2").tobytes() + bytes(32)
        with self.assertRaisesRegex(ValueError, "a < b"):
            codec.w2_bf16_from_blob(bad_order, (1, 128))
        with self.assertRaisesRegex(ValueError, "byte count"):
            codec.w2_bf16_from_blob(bytes(35), (1, 128))

    def test_ties_choose_small_magnitude_and_zero_is_positive(self) -> None:
        # Equal stored centers force the small-center branch: - values get 1,
        # + values and zero get 2, never the outer codes.
        center = codec.bf16_rne_bits_scalar(np.float32(1))
        packed = codec.pack_w2_codes_scalar([1, 2, 2, 1] * 32)
        encoded = codec.W2BF16Encoded((1, 128), np.asarray([[center]], dtype=np.uint16),
                                      np.asarray([[center]], dtype=np.uint16), packed.reshape(1, -1))
        self.assertEqual(codec.unpack_w2_codes_scalar(encoded.packed_codes[0]).tolist()[:4], [1, 2, 2, 1])


if __name__ == "__main__":
    unittest.main()
