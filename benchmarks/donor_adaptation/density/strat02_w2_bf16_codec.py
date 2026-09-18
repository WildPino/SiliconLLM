"""Frozen STRAT-02E W2 g128 BF16-codebook codec.

The module is deliberately model-free.  Its scalar path is the normative
reference; the tile path has bounded temporary arrays and is accepted only
when it emits exactly the same canonical bytes.  A tensor wire image is three
row-major arrays, ``a_bits``, ``b_bits``, then four 2-bit codes per byte.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np


GROUP_SIZE = 128
CODES_PER_BYTE = 4
CODE_BYTES_PER_GROUP = GROUP_SIZE // CODES_PER_BYTE
METADATA_BYTES_PER_GROUP = 4
BYTES_PER_GROUP = METADATA_BYTES_PER_GROUP + CODE_BYTES_PER_GROUP
LLOYD_ITERATIONS = 8


def _shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("linear weights must have shape [out_features, in_features]")
    rows, columns = int(shape[0]), int(shape[1])
    if rows < 0 or columns <= 0 or columns % GROUP_SIZE:
        raise ValueError("in_features must be positive and divisible by 128")
    return rows, columns


def w2_bf16_byte_count(rows: int, columns: int) -> int:
    rows, columns = _shape((rows, columns))
    return rows * (columns // GROUP_SIZE) * BYTES_PER_GROUP


def w2_bf16_metadata(shape: Sequence[int]) -> dict[str, object]:
    rows, columns = _shape(shape)
    return {
        "codec": "strat02_w2_sym4_g128_bf16_v1",
        "shape": (rows, columns),
        "group_size": GROUP_SIZE,
        "codebook": "{-a,-b,+b,+a}; a,b raw IEEE bfloat16 little-endian RNE",
        "codes": "0:-a,1:-b,2:+b,3:+a; four per byte low bits first",
        "byte_count": w2_bf16_byte_count(rows, columns),
        "byte_count_formula": "out_features * (in_features / 128) * 36",
    }


def _f32_bits(value: np.float32) -> int:
    return int(np.asarray([np.float32(value)], dtype="<f4").view("<u4")[0])


def bf16_rne_bits_scalar(value: float | np.float32) -> np.uint16:
    """Raw IEEE BF16 RNE conversion for one F32 value.

    NaNs are canonicalized only so synthetic conversion tests have a stable
    answer.  Codec centers themselves reject every non-finite representation.
    """
    source = np.float32(value)
    if np.isnan(source):
        return np.uint16(0xFFFF)
    word = _f32_bits(source)
    return np.uint16((word + 0x7FFF + ((word >> 16) & 1)) >> 16)


def bf16_rne_bits(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    words = source.astype("<f4", copy=False).view("<u4").astype(np.uint64)
    rounded = ((words + np.uint64(0x7FFF) + ((words >> np.uint64(16)) & 1)) >> 16).astype("<u2")
    return np.where(np.isnan(source), np.uint16(0xFFFF), rounded).astype(np.uint16, copy=False)


def bf16_bits_to_float32(bits: np.ndarray | np.uint16 | int) -> np.ndarray | np.float32:
    raw = np.asarray(bits, dtype="<u2")
    expanded = (raw.astype("<u4") << np.uint32(16)).view("<f4")
    return np.float32(expanded) if expanded.ndim == 0 else expanded


def _center(bits: int | np.uint16, label: str) -> np.float32:
    # Negative zero is intentionally not a second representation of zero.
    if int(bits) & 0x8000:
        raise ValueError(f"invalid W2 BF16 {label}: negative center")
    value = np.float32(bf16_bits_to_float32(bits))
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"invalid W2 BF16 {label}: nonfinite or negative center")
    return value


@dataclass(frozen=True)
class W2BF16Encoded:
    shape: tuple[int, int]
    a_bits: np.ndarray  # uint16 [rows, groups]
    b_bits: np.ndarray  # uint16 [rows, groups]
    packed_codes: np.ndarray  # uint8 [rows, groups * 32]

    @property
    def metadata(self) -> dict[str, object]:
        return w2_bf16_metadata(self.shape)

    @property
    def byte_count(self) -> int:
        return w2_bf16_byte_count(*self.shape)


@dataclass(frozen=True)
class W2BF16EncodedTile:
    row_offset: int
    group_offset: int
    a_bits: np.ndarray
    b_bits: np.ndarray
    packed_codes: np.ndarray


def pack_w2_codes_scalar(codes: Sequence[int]) -> np.ndarray:
    if len(codes) % CODES_PER_BYTE:
        raise ValueError("W2 code count must be divisible by four")
    packed = np.empty(len(codes) // CODES_PER_BYTE, dtype=np.uint8)
    for byte_index in range(packed.size):
        word = 0
        for offset in range(CODES_PER_BYTE):
            code = int(codes[byte_index * CODES_PER_BYTE + offset])
            if not 0 <= code <= 3:
                raise ValueError("W2 codes must be in [0, 3]")
            word |= code << (2 * offset)
        packed[byte_index] = word
    return packed


def unpack_w2_codes_scalar(packed: Sequence[int]) -> np.ndarray:
    result = np.empty(len(packed) * CODES_PER_BYTE, dtype=np.uint8)
    for byte_index, byte in enumerate(packed):
        value = int(byte)
        if not 0 <= value <= 255:
            raise ValueError("packed W2 bytes must be in [0, 255]")
        for offset in range(CODES_PER_BYTE):
            result[byte_index * CODES_PER_BYTE + offset] = (value >> (2 * offset)) & 3
    return result


def _unpack_w2_codes(packed: np.ndarray) -> np.ndarray:
    """Expand low-bit-first W2 bytes in bulk, preserving the frozen wire order."""
    source = np.asarray(packed, dtype=np.uint8)
    return np.stack(tuple((source >> np.uint8(shift)) & np.uint8(3)
                          for shift in range(0, 8, 2)), axis=-1).reshape(*source.shape[:-1], source.shape[-1] * CODES_PER_BYTE)


def _validate(encoded: W2BF16Encoded) -> tuple[int, int, int]:
    rows, columns = _shape(encoded.shape)
    groups = columns // GROUP_SIZE
    if encoded.a_bits.dtype != np.uint16 or encoded.a_bits.shape != (rows, groups):
        raise ValueError("invalid W2 BF16 a-bit array")
    if encoded.b_bits.dtype != np.uint16 or encoded.b_bits.shape != (rows, groups):
        raise ValueError("invalid W2 BF16 b-bit array")
    if encoded.packed_codes.dtype != np.uint8 or encoded.packed_codes.shape != (rows, groups * CODE_BYTES_PER_GROUP):
        raise ValueError("invalid W2 BF16 packed-code array")
    if np.any((encoded.a_bits & np.uint16(0x8000)) != 0) or np.any((encoded.b_bits & np.uint16(0x8000)) != 0):
        raise ValueError("invalid W2 BF16 center: negative center")
    a = np.asarray(bf16_bits_to_float32(encoded.a_bits), dtype=np.float32)
    b = np.asarray(bf16_bits_to_float32(encoded.b_bits), dtype=np.float32)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("invalid W2 BF16 center: nonfinite or negative center")
    if np.any(a < b):
        raise ValueError("invalid W2 BF16 group: a < b")
    codes = _unpack_w2_codes(encoded.packed_codes).reshape(rows, groups, GROUP_SIZE)
    zero = a == 0
    if np.any(zero & ((b != 0) | np.any(codes != 2, axis=2))):
        raise ValueError("invalid W2 BF16 zero group")
    return rows, columns, groups


def w2_bf16_to_blob(encoded: W2BF16Encoded) -> bytes:
    rows, columns, groups = _validate(encoded)
    blob = (encoded.a_bits.astype("<u2", copy=False).tobytes(order="C") +
            encoded.b_bits.astype("<u2", copy=False).tobytes(order="C") +
            encoded.packed_codes.tobytes(order="C"))
    if len(blob) != rows * groups * BYTES_PER_GROUP:
        raise AssertionError("internal W2 BF16 byte-count failure")
    return blob


def w2_bf16_from_blob(blob: bytes, shape: Sequence[int]) -> W2BF16Encoded:
    rows, columns = _shape(shape)
    groups = columns // GROUP_SIZE
    if len(blob) != w2_bf16_byte_count(rows, columns):
        raise ValueError("invalid W2 BF16 blob byte count")
    metadata_bytes = rows * groups * 2
    encoded = W2BF16Encoded(
        (rows, columns),
        np.frombuffer(blob[:metadata_bytes], dtype="<u2").reshape(rows, groups),
        np.frombuffer(blob[metadata_bytes:2 * metadata_bytes], dtype="<u2").reshape(rows, groups),
        np.frombuffer(blob[2 * metadata_bytes:], dtype=np.uint8).reshape(rows, groups * CODE_BYTES_PER_GROUP),
    )
    _validate(encoded)
    return encoded


def _quantile_linear_scalar(values: Sequence[np.float32], probability: np.float32) -> np.float32:
    ordered = sorted(np.float32(value) for value in values)
    position = np.float32(np.float32(len(ordered) - 1) * probability)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return np.float32(ordered[lower])
    fraction = np.float32(position - np.float32(lower))
    return np.float32(np.float32(ordered[lower]) + np.float32(fraction * np.float32(ordered[upper] - ordered[lower])))


def _assign_scalar(magnitude: np.float32, b: np.float32, a: np.float32) -> int:
    # The small center wins ties, including a == b.
    return 0 if np.abs(np.float32(magnitude - b)) <= np.abs(np.float32(magnitude - a)) else 1


def _w2_bf16_group_scalar_reference(values: Sequence[float]) -> tuple[np.uint16, np.uint16, np.ndarray]:
    """Independent normative scalar F32 fit and final BF16 code assignment."""
    if len(values) != GROUP_SIZE:
        raise ValueError("W2 groups must contain exactly 128 values")
    source = [np.float32(value) for value in values]
    if not all(np.isfinite(value) for value in source):
        raise ValueError("weights must be finite")
    magnitudes = [np.float32(abs(value)) for value in source]
    if max(magnitudes) == 0:
        return np.uint16(0), np.uint16(0), np.full(GROUP_SIZE, 2, dtype=np.uint8)
    b = _quantile_linear_scalar(magnitudes, np.float32(0.25))
    a = _quantile_linear_scalar(magnitudes, np.float32(0.75))
    for _iteration in range(LLOYD_ITERATIONS):
        low: list[np.float32] = []
        high: list[np.float32] = []
        for magnitude in magnitudes:
            (low if _assign_scalar(magnitude, b, a) == 0 else high).append(magnitude)
        if low:
            b = np.float32(sum(low, np.float32(0)) / np.float32(len(low)))
        if high:
            a = np.float32(sum(high, np.float32(0)) / np.float32(len(high)))
        if a < b:
            a, b = b, a
    a_bits, b_bits = bf16_rne_bits_scalar(a), bf16_rne_bits_scalar(b)
    stored_a, stored_b = _center(a_bits, "a"), _center(b_bits, "b")
    if stored_a < stored_b or stored_a == 0:
        raise ValueError("nonzero W2 group has an invalid BF16 codebook")
    codes = np.empty(GROUP_SIZE, dtype=np.uint8)
    for index, value in enumerate(source):
        small = _assign_scalar(np.float32(abs(value)), stored_b, stored_a) == 0
        if value < 0:
            codes[index] = 1 if small else 0
        else:
            codes[index] = 2 if small else 3
    return a_bits, b_bits, codes


def _require_batch(batch: object, columns: int) -> np.ndarray:
    if not isinstance(batch, np.ndarray) or batch.dtype != np.float32:
        raise TypeError("W2 BF16 tile batches must be numpy float32 arrays")
    if batch.ndim != 2 or batch.shape[1] != columns:
        raise ValueError("W2 BF16 tile batch must have shape [rows, in_features]")
    return batch


def _quantile_linear(values: np.ndarray, probability: np.float32) -> np.ndarray:
    ordered = np.sort(values, axis=2)
    position = np.float32(np.float32(GROUP_SIZE - 1) * probability)
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    fraction = np.float32(position - np.float32(lower))
    return np.float32(ordered[:, :, lower] + fraction * (ordered[:, :, upper] - ordered[:, :, lower]))


def _encode_group_tile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[2] != GROUP_SIZE:
        raise ValueError("W2 BF16 tile must be float32 [rows, groups, 128]")
    magnitudes = np.abs(values, dtype=np.float32)
    nonzero = np.max(magnitudes, axis=2) != 0
    b = _quantile_linear(magnitudes, np.float32(0.25))
    a = _quantile_linear(magnitudes, np.float32(0.75))
    for _iteration in range(LLOYD_ITERATIONS):
        small = np.abs(magnitudes - b[:, :, None]) <= np.abs(magnitudes - a[:, :, None])
        small_count = np.sum(small, axis=2, dtype=np.int32)
        large_count = GROUP_SIZE - small_count
        # cumsum is a C-backed, ordered F32 accumulation.  Unlike np.sum it
        # cannot switch to a pairwise reduction, so it preserves the scalar
        # reference's frozen left-to-right arithmetic while removing 128
        # Python/ufunc dispatches per Lloyd iteration.
        small_sum = np.cumsum(np.where(small, magnitudes, np.float32(0)), axis=2, dtype=np.float32)[:, :, -1]
        large_sum = np.cumsum(np.where(~small, magnitudes, np.float32(0)), axis=2, dtype=np.float32)[:, :, -1]
        small_mean = np.float32(small_sum / np.maximum(small_count, 1).astype(np.float32))
        large_mean = np.float32(large_sum / np.maximum(large_count, 1).astype(np.float32))
        b = np.where(small_count != 0, small_mean, b).astype(np.float32, copy=False)
        a = np.where(large_count != 0, large_mean, a).astype(np.float32, copy=False)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        b, a = lo.astype(np.float32, copy=False), hi.astype(np.float32, copy=False)
    a_bits, b_bits = bf16_rne_bits(a), bf16_rne_bits(b)
    stored_a = np.asarray(bf16_bits_to_float32(a_bits), dtype=np.float32)
    stored_b = np.asarray(bf16_bits_to_float32(b_bits), dtype=np.float32)
    invalid = nonzero & ((~np.isfinite(stored_a)) | (~np.isfinite(stored_b)) | (stored_a < stored_b) | (stored_a == 0))
    if np.any(invalid):
        raise ValueError("nonzero W2 group has an invalid BF16 codebook")
    small = np.abs(magnitudes - stored_b[:, :, None]) <= np.abs(magnitudes - stored_a[:, :, None])
    codes = np.where(values < 0, np.where(small, 1, 0), np.where(small, 2, 3)).astype(np.uint8)
    a_bits = np.where(nonzero, a_bits, np.uint16(0)).astype(np.uint16, copy=False)
    b_bits = np.where(nonzero, b_bits, np.uint16(0)).astype(np.uint16, copy=False)
    codes[~nonzero] = 2
    packed = (codes[:, :, 0::4] | (codes[:, :, 1::4] << 2) |
              (codes[:, :, 2::4] << 4) | (codes[:, :, 3::4] << 6)).astype(np.uint8)
    return a_bits, b_bits, packed.reshape(values.shape[0], values.shape[1] * CODE_BYTES_PER_GROUP)


def iter_encode_w2_bf16_tiles(row_batches: Iterable[np.ndarray], columns: int, *, max_rows_per_tile: int = 16,
                               max_groups_per_tile: int = 16) -> Iterator[W2BF16EncodedTile]:
    _rows, columns = _shape((0, columns))
    if type(max_rows_per_tile) is not int or max_rows_per_tile <= 0 or type(max_groups_per_tile) is not int or max_groups_per_tile <= 0:
        raise ValueError("tile dimensions must be positive integers")
    total_groups, row_offset = columns // GROUP_SIZE, 0
    for batch in row_batches:
        source = _require_batch(batch, columns)
        if not np.all(np.isfinite(source)):
            raise ValueError("weights must be finite")
        for row_start in range(0, source.shape[0], max_rows_per_tile):
            row_end = min(row_start + max_rows_per_tile, source.shape[0])
            for group_start in range(0, total_groups, max_groups_per_tile):
                group_end = min(group_start + max_groups_per_tile, total_groups)
                values = source[row_start:row_end, group_start * GROUP_SIZE:group_end * GROUP_SIZE].reshape(
                    row_end - row_start, group_end - group_start, GROUP_SIZE)
                a_bits, b_bits, packed = _encode_group_tile(values)
                yield W2BF16EncodedTile(row_offset + row_start, group_start, a_bits, b_bits, packed)
        row_offset += source.shape[0]


def encode_w2_bf16(weights: np.ndarray) -> W2BF16Encoded:
    if not isinstance(weights, np.ndarray) or weights.dtype != np.float32:
        raise TypeError("W2 BF16 weights must be a numpy float32 array")
    rows, columns = _shape(weights.shape)
    groups = columns // GROUP_SIZE
    a_bits = np.empty((rows, groups), dtype=np.uint16)
    b_bits = np.empty_like(a_bits)
    packed = np.empty((rows, groups * CODE_BYTES_PER_GROUP), dtype=np.uint8)
    for row in range(rows):
        for group in range(groups):
            a, b, codes = _w2_bf16_group_scalar_reference(weights[row, group * GROUP_SIZE:(group + 1) * GROUP_SIZE])
            a_bits[row, group], b_bits[row, group] = a, b
            packed[row, group * CODE_BYTES_PER_GROUP:(group + 1) * CODE_BYTES_PER_GROUP] = pack_w2_codes_scalar(codes)
    return W2BF16Encoded((rows, columns), a_bits, b_bits, packed)


def _decoded_group(a_bits: int | np.uint16, b_bits: int | np.uint16, codes: Sequence[int]) -> np.ndarray:
    a, b = _center(a_bits, "a"), _center(b_bits, "b")
    if a < b or len(codes) != GROUP_SIZE or (a == 0 and (b != 0 or any(int(code) != 2 for code in codes))):
        raise ValueError("invalid W2 BF16 group")
    table = np.asarray([-a, -b, b, a], dtype=np.float32)
    result = np.empty(GROUP_SIZE, dtype=np.float32)
    for index, code in enumerate(codes):
        if not 0 <= int(code) <= 3:
            raise ValueError("W2 codes must be in [0, 3]")
        result[index] = table[int(code)]
    return result


def dequantize_w2_bf16(encoded: W2BF16Encoded) -> np.ndarray:
    rows, columns, groups = _validate(encoded)
    codes = _unpack_w2_codes(encoded.packed_codes).reshape(rows, groups, GROUP_SIZE)
    a = np.asarray(bf16_bits_to_float32(encoded.a_bits), dtype=np.float32)
    b = np.asarray(bf16_bits_to_float32(encoded.b_bits), dtype=np.float32)
    table = np.stack((-a, -b, b, a), axis=-1)
    return np.take_along_axis(table[:, :, None, :], codes[:, :, :, None], axis=3).reshape(rows, columns)


def decode_w2_bf16_tile(tile: W2BF16EncodedTile) -> np.ndarray:
    rows, groups = tile.a_bits.shape
    encoded = W2BF16Encoded((rows, groups * GROUP_SIZE), tile.a_bits, tile.b_bits, tile.packed_codes)
    return dequantize_w2_bf16(encoded)


def selftest() -> None:
    rng = np.random.default_rng(20260917)
    source = rng.normal(size=(7, 640)).astype(np.float32)
    source[0, :128] = 0
    source[1, 128:256] = np.asarray([-4, -1, 0, 1, 4] + [0] * 123, dtype=np.float32)
    scalar = encode_w2_bf16(source)
    a, b = np.empty_like(scalar.a_bits), np.empty_like(scalar.b_bits)
    packed = np.empty_like(scalar.packed_codes)
    for tile in iter_encode_w2_bf16_tiles((source[:3], source[3:]), 640, max_rows_per_tile=2, max_groups_per_tile=3):
        rows, groups = tile.a_bits.shape
        a[tile.row_offset:tile.row_offset + rows, tile.group_offset:tile.group_offset + groups] = tile.a_bits
        b[tile.row_offset:tile.row_offset + rows, tile.group_offset:tile.group_offset + groups] = tile.b_bits
        packed[tile.row_offset:tile.row_offset + rows, tile.group_offset * CODE_BYTES_PER_GROUP:(tile.group_offset + groups) * CODE_BYTES_PER_GROUP] = tile.packed_codes
    tiled = W2BF16Encoded(source.shape, a, b, packed)
    if w2_bf16_to_blob(scalar) != w2_bf16_to_blob(tiled):
        raise AssertionError("tile bytes differ from scalar reference")
    decoded = dequantize_w2_bf16(w2_bf16_from_blob(w2_bf16_to_blob(scalar), source.shape))
    if decoded.dtype != np.float32 or decoded.shape != source.shape or not np.all(np.isfinite(decoded)):
        raise AssertionError("packed W2 decode failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run synthetic codec checks")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print('{"ok": true, "selftest": "strat02_w2_bf16_codec"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
