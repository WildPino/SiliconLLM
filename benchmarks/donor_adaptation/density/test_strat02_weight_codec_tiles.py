"""Synthetic exact-parity tests for the bounded STRAT-02 W4 tile encoder."""

from __future__ import annotations

import unittest

import numpy as np

from benchmarks.donor_adaptation.density import strat02_weight_codec as codec


def _tile_blob(
    source: np.ndarray,
    batch_rows: tuple[int, ...],
    *,
    max_rows_per_tile: int,
    max_groups_per_tile: int,
) -> bytes:
    """Write tile output to synthetic caller-owned arrays, then serialize it."""
    groups = source.shape[1] // codec.GROUP_SIZE
    scales = np.empty((source.shape[0], groups), dtype=np.float16)
    packed = np.empty((source.shape[0], groups * 64), dtype=np.uint8)

    def batches():
        start = 0
        for count in batch_rows:
            yield source[start:start + count]
            start += count
        assert start == source.shape[0]

    for tile in codec.iter_encode_w4_tiles(
        batches(),
        source.shape[1],
        max_rows_per_tile=max_rows_per_tile,
        max_groups_per_tile=max_groups_per_tile,
    ):
        rows = tile.scales.shape[0]
        tile_groups = tile.scales.shape[1]
        self_row = slice(tile.row_offset, tile.row_offset + rows)
        self_group = slice(tile.group_offset, tile.group_offset + tile_groups)
        scales[self_row, self_group] = tile.scales
        packed[self_row, tile.group_offset * 64 : (tile.group_offset + tile_groups) * 64] = tile.packed_codes
    return codec.w4_to_blob(codec.W4Encoded(tuple(source.shape), scales, packed))


class W4TileEncoderTest(unittest.TestCase):
    def assert_tile_matches_reference(
        self,
        source: np.ndarray,
        *,
        batch_rows: tuple[int, ...],
        max_rows_per_tile: int,
        max_groups_per_tile: int,
    ) -> None:
        self.assertEqual(source.dtype, np.float32)
        reference = codec.w4_to_blob(codec.encode_w4(source))
        tiled = _tile_blob(
            source,
            batch_rows,
            max_rows_per_tile=max_rows_per_tile,
            max_groups_per_tile=max_groups_per_tile,
        )
        self.assertEqual(tiled, reference)

    def test_exact_parity_seeded_random_with_multiple_batch_and_tile_shapes(self) -> None:
        rng = np.random.default_rng(20260917)
        source = rng.normal(size=(13, 640)).astype(np.float32)
        for batch_rows, max_rows, max_groups in (
            ((13,), 32, 32),
            ((1,) * 13, 1, 1),
            ((4, 6, 3), 2, 3),
            ((7, 6), 5, 2),
        ):
            with self.subTest(batch_rows=batch_rows, max_rows=max_rows, max_groups=max_groups):
                self.assert_tile_matches_reference(
                    source,
                    batch_rows=batch_rows,
                    max_rows_per_tile=max_rows,
                    max_groups_per_tile=max_groups,
                )

    def test_exact_parity_zero_halfway_tie_and_outlier_groups(self) -> None:
        zero = np.zeros(128, dtype=np.float32)
        halfway = np.zeros(128, dtype=np.float32)
        halfway[:5] = np.array([7.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
        outlier = np.full(128, np.float32(0.03125), dtype=np.float32)
        outlier[0], outlier[1], outlier[2] = np.float32(1000), np.float32(-1000), np.float32(-0.03125)
        source = np.stack((np.concatenate((zero, halfway, outlier)),) * 3).astype(np.float32)
        self.assert_tile_matches_reference(
            source,
            batch_rows=(2, 1),
            max_rows_per_tile=1,
            max_groups_per_tile=2,
        )

    def test_tiles_are_bounded_and_have_global_offsets(self) -> None:
        source = np.arange(5 * 384, dtype=np.float32).reshape(5, 384) / np.float32(31)
        tiles = list(codec.iter_encode_w4_tiles((source[:3], source[3:]), 384,
                                                max_rows_per_tile=2, max_groups_per_tile=2))
        self.assertEqual([(tile.row_offset, tile.group_offset, tile.scales.shape, tile.packed_codes.shape)
                          for tile in tiles], [
            (0, 0, (2, 2), (2, 128)), (0, 2, (2, 1), (2, 64)),
            (2, 0, (1, 2), (1, 128)), (2, 2, (1, 1), (1, 64)),
            (3, 0, (2, 2), (2, 128)), (3, 2, (2, 1), (2, 64)),
        ])
        self.assert_tile_matches_reference(
            source, batch_rows=(3, 2), max_rows_per_tile=2, max_groups_per_tile=2
        )

    def test_exact_parity_many_scales_and_noncontiguous_view(self) -> None:
        rng = np.random.default_rng(20260918)
        source = rng.normal(size=(17, 2048)).astype(np.float32)
        source *= np.geomspace(np.float32(1e-4), np.float32(1e3), 16,
                                   dtype=np.float32).repeat(128)[None, :]
        parent = np.empty((17, 4096), dtype=np.float32)
        parent[:, ::2] = source
        parent[:, 1::2] = 0
        view = parent[:, ::2]
        self.assertFalse(view.flags.c_contiguous)
        self.assert_tile_matches_reference(
            view, batch_rows=(6, 3, 8), max_rows_per_tile=4,
            max_groups_per_tile=5,
        )

    def test_nonfinite_tiny_and_malformed_batches_fail_closed(self) -> None:
        valid = np.zeros((1, 128), dtype=np.float32)
        for value in (np.nan, np.inf, -np.inf):
            source = valid.copy()
            source[0, 17] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    list(codec.iter_encode_w4_tiles((source,), 128))

        tiny = valid.copy()
        tiny[0, 0] = np.finfo(np.float32).tiny
        with self.assertRaises(ValueError):
            codec.encode_w4(tiny)
        with self.assertRaises(ValueError):
            list(codec.iter_encode_w4_tiles((tiny,), 128))

        with self.assertRaises(TypeError):
            list(codec.iter_encode_w4_tiles((valid.astype(np.float64),), 128))
        with self.assertRaises(ValueError):
            list(codec.iter_encode_w4_tiles((np.zeros((1, 127), dtype=np.float32),), 128))
        with self.assertRaises(ValueError):
            list(codec.iter_encode_w4_tiles((valid,), 128, max_rows_per_tile=0))
        with self.assertRaises(ValueError):
            list(codec.iter_encode_w4_tiles((valid,), 128, max_groups_per_tile=True))


if __name__ == "__main__":
    unittest.main()
