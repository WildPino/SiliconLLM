"""Frozen STRAT-02 W4 g128 BF16-scale v2 reference codec.

This is a deliberately standalone apparatus.  It does not import or alter the
v1 codec.  Groups are 128 contiguous F32 input columns in a ``[out, in]``
linear weight, with a little-endian IEEE bfloat16 scale followed by packed
signed W4 codes (low nibble first).  It is neither a model loader nor a
quality/rate claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np


GROUP_SIZE = 128
W4_VALUES_PER_BYTE = 2
W4_CANDIDATE_C = tuple(np.float32(i) / np.float32(100) for i in range(50, 101, 5))
SCALE_BYTES_PER_GROUP = 2
CODES_BYTES_PER_GROUP = GROUP_SIZE // W4_VALUES_PER_BYTE


def _require_shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("linear weights must have shape [out_features, in_features]")
    rows, columns = int(shape[0]), int(shape[1])
    if rows < 0 or columns <= 0 or columns % GROUP_SIZE:
        raise ValueError("in_features must be positive and divisible by 128")
    return rows, columns


def w4_bf16_byte_count(rows: int, columns: int) -> int:
    rows, columns = _require_shape((rows, columns))
    return rows * (columns // GROUP_SIZE) * (SCALE_BYTES_PER_GROUP + CODES_BYTES_PER_GROUP)


def w4_bf16_metadata(shape: Sequence[int]) -> dict[str, object]:
    rows, columns = _require_shape(shape)
    return {
        "codec": "strat02_w4_g128_bf16scale_v2",
        "shape": (rows, columns),
        "group_size": GROUP_SIZE,
        "scale_dtype": "IEEE bfloat16 little-endian (RNE from F32)",
        "codes": "signed 4-bit twos-complement, low nibble first, -8 forbidden",
        "candidate_c": tuple(float(c) for c in W4_CANDIDATE_C),
        "byte_count": w4_bf16_byte_count(rows, columns),
        "byte_count_formula": "out_features * (in_features / 128) * (2 + 64)",
    }


def _f32_bits(value: np.float32) -> int:
    return int(np.asarray([np.float32(value)], dtype="<f4").view("<u4")[0])


def bf16_rne_bits_scalar(value: float | np.float32) -> np.uint16:
    """Return the IEEE BF16 RNE bit pattern of one F32 value.

    This is bit arithmetic, rather than a dtype cast: adding ``0x7fff`` plus
    the retained LSB implements round-to-nearest-even, including carry into
    the exponent.  It preserves the source's high 16 bits for zeros and
    infinities; NaNs use the observed CPU torch canonical bit pattern below.
    """
    source = np.float32(value)
    # CPU torch canonicalizes every NaN payload/sign to BF16 0xffff.  The
    # format itself rejects nonfinite scales, but this helper is intentionally
    # a full casting reference for the preregistered synthetic checks.
    if np.isnan(source):
        return np.uint16(0xFFFF)
    word = _f32_bits(source)
    return np.uint16((word + 0x7FFF + ((word >> 16) & 1)) >> 16)


def bf16_rne_bits(values: np.ndarray) -> np.ndarray:
    """Vectorized independent tile-path RNE conversion to BF16 bit patterns."""
    source = np.asarray(values, dtype=np.float32)
    words = source.astype("<f4", copy=False).view("<u4").astype(np.uint64)
    rounded = ((words + np.uint64(0x7FFF) + ((words >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)).astype("<u2")
    return np.where(np.isnan(source), np.uint16(0xFFFF), rounded).astype(np.uint16, copy=False)


def bf16_bits_to_float32(bits: np.ndarray | np.uint16 | int) -> np.ndarray | np.float32:
    """Expand raw BF16 bits by placing them in the high half of an F32 word."""
    array = np.asarray(bits, dtype="<u2")
    expanded = (array.astype("<u4") << np.uint32(16)).view("<f4")
    return np.float32(expanded) if expanded.ndim == 0 else expanded


def _scale_is_valid(bits: np.uint16 | int) -> bool:
    # +0 is reserved for a source-zero group.  -0 is not a canonical scale.
    if int(bits) == 0:
        return True
    scale = np.float32(bf16_bits_to_float32(bits))
    return bool(np.isfinite(scale) and scale > 0)


@dataclass(frozen=True)
class W4BF16Encoded:
    shape: tuple[int, int]
    scale_bits: np.ndarray  # uint16 [out_features, groups], raw IEEE BF16
    packed_codes: np.ndarray  # uint8 [out_features, groups * 64]

    @property
    def metadata(self) -> dict[str, object]:
        return w4_bf16_metadata(self.shape)

    @property
    def byte_count(self) -> int:
        return w4_bf16_byte_count(*self.shape)


@dataclass(frozen=True)
class W4BF16EncodedTile:
    row_offset: int
    group_offset: int
    scale_bits: np.ndarray  # uint16 [tile_rows, tile_groups]
    packed_codes: np.ndarray  # uint8 [tile_rows, tile_groups * 64]


def pack_w4_codes_scalar(codes: Sequence[int]) -> np.ndarray:
    if len(codes) % 2:
        raise ValueError("W4 code count must be even")
    packed = np.empty(len(codes) // 2, dtype=np.uint8)
    for index in range(packed.size):
        low, high = int(codes[2 * index]), int(codes[2 * index + 1])
        if not (-7 <= low <= 7 and -7 <= high <= 7):
            raise ValueError("W4 codes must be in [-7, 7]; -8 is forbidden")
        packed[index] = (low & 0x0F) | ((high & 0x0F) << 4)
    return packed


def unpack_w4_codes_scalar(packed: Sequence[int]) -> np.ndarray:
    codes = np.empty(len(packed) * 2, dtype=np.int8)
    for byte_index, byte in enumerate(packed):
        byte = int(byte)
        if not 0 <= byte <= 255:
            raise ValueError("packed W4 bytes must be in [0, 255]")
        for offset, nibble in enumerate((byte & 0x0F, byte >> 4)):
            code = nibble - 16 if nibble >= 8 else nibble
            if code == -8:
                raise ValueError("invalid W4 stream: -8 is reserved")
            codes[2 * byte_index + offset] = code
    return codes


def _validate_encoded(encoded: W4BF16Encoded) -> tuple[int, int, int]:
    rows, columns = _require_shape(encoded.shape)
    groups = columns // GROUP_SIZE
    if encoded.scale_bits.shape != (rows, groups) or encoded.scale_bits.dtype != np.uint16:
        raise ValueError("invalid W4 BF16 scale-bit array")
    if encoded.packed_codes.shape != (rows, groups * CODES_BYTES_PER_GROUP) or encoded.packed_codes.dtype != np.uint8:
        raise ValueError("invalid W4 BF16 packed-code array")
    for row in range(rows):
        for group in range(groups):
            bits = encoded.scale_bits[row, group]
            packed = encoded.packed_codes[row, group * 64:(group + 1) * 64]
            if int(bits) == 0:
                if np.any(packed):
                    raise ValueError("zero BF16 scale is valid only with zero W4 codes")
            elif not _scale_is_valid(bits):
                raise ValueError("nonzero W4 group has an invalid BF16 scale")
            # Scalar decoding is the authoritative reserved-nibble validator.
            unpack_w4_codes_scalar(packed)
    return rows, columns, groups


def w4_bf16_to_blob(encoded: W4BF16Encoded) -> bytes:
    rows, columns, _groups = _validate_encoded(encoded)
    scales = encoded.scale_bits.astype("<u2", copy=False).tobytes(order="C")
    blob = scales + encoded.packed_codes.tobytes(order="C")
    if len(blob) != w4_bf16_byte_count(rows, columns):
        raise AssertionError("internal W4 BF16 byte-count failure")
    return blob


def w4_bf16_from_blob(blob: bytes, shape: Sequence[int]) -> W4BF16Encoded:
    rows, columns = _require_shape(shape)
    if len(blob) != w4_bf16_byte_count(rows, columns):
        raise ValueError("invalid W4 BF16 blob byte count")
    groups = columns // GROUP_SIZE
    scale_bytes = rows * groups * SCALE_BYTES_PER_GROUP
    encoded = W4BF16Encoded(
        (rows, columns),
        np.frombuffer(blob[:scale_bytes], dtype="<u2").reshape(rows, groups),
        np.frombuffer(blob[scale_bytes:], dtype=np.uint8).reshape(rows, groups * 64),
    )
    _validate_encoded(encoded)
    return encoded


def dequantize_w4_bf16_group_scalar(scale_bits: np.uint16 | int, codes: Sequence[int]) -> np.ndarray:
    if len(codes) != GROUP_SIZE:
        raise ValueError("W4 groups must contain exactly 128 codes")
    if not _scale_is_valid(scale_bits):
        raise ValueError("invalid BF16 scale")
    scale = np.float32(bf16_bits_to_float32(scale_bits))
    result = np.empty(GROUP_SIZE, dtype=np.float32)
    for index, code in enumerate(codes):
        code = int(code)
        if not -7 <= code <= 7:
            raise ValueError("W4 codes must be in [-7, 7]; -8 is forbidden")
        result[index] = np.float32(scale * np.float32(code))
    return result


def _candidate_f32(absmax: np.float32, c: np.float32) -> np.float32:
    # Keep the frozen v1 operation sequence and F32 intermediates explicit.
    return np.float32(np.float32(np.float32(absmax) * np.float32(c)) / np.float32(7))


def _w4_bf16_group_scalar_reference(values: Sequence[float]) -> tuple[np.uint16, np.ndarray]:
    """Independent scalar fit used as the byte-layout reference."""
    if len(values) != GROUP_SIZE:
        raise ValueError("W4 groups must contain exactly 128 values")
    source = [np.float32(value) for value in values]
    if not all(np.isfinite(value) for value in source):
        raise ValueError("weights must be finite")
    absmax = max(abs(value) for value in source)
    if absmax == 0:
        return np.uint16(0), np.zeros(GROUP_SIZE, dtype=np.int8)
    best_mse: np.float32 | None = None
    best_bits = np.uint16(0)
    best_codes = np.zeros(GROUP_SIZE, dtype=np.int8)
    for c in W4_CANDIDATE_C:
        bits = bf16_rne_bits_scalar(_candidate_f32(np.float32(absmax), c))
        if int(bits) == 0 or not _scale_is_valid(bits):
            raise ValueError("nonzero W4 group has an invalid BF16 candidate scale")
        scale = np.float32(bf16_bits_to_float32(bits))
        codes = np.empty(GROUP_SIZE, dtype=np.int8)
        for index, value in enumerate(source):
            codes[index] = max(-7, min(7, int(round(float(np.float32(value) / scale)))))
        mse = np.float32(0)
        for value, code in zip(source, codes):
            error = np.float32(value - np.float32(scale * np.float32(code)))
            mse = np.float32(mse + np.float32(error * error))
        mse = np.float32(mse / np.float32(GROUP_SIZE))
        # Candidate c grows, so <= resolves exact MSE ties to the larger c.
        if best_mse is None or mse <= best_mse:
            best_mse, best_bits, best_codes = mse, bits, codes
    return best_bits, best_codes


def _as_float32_row(row: object, columns: int) -> np.ndarray:
    value = np.asarray(row, dtype=np.float32).reshape(-1)
    if value.size != columns:
        raise ValueError("row length does not match in_features")
    if not np.all(np.isfinite(value)):
        raise ValueError("weights must be finite")
    return value


def iter_encode_w4_bf16_rows(rows: Iterable[object], columns: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    _rows, columns = _require_shape((0, columns))
    groups = columns // GROUP_SIZE
    for row in rows:
        source = _as_float32_row(row, columns)
        scales = np.empty(groups, dtype=np.uint16)
        packed = np.empty(groups * 64, dtype=np.uint8)
        for group in range(groups):
            bits, codes = _w4_bf16_group_scalar_reference(source[group * 128:(group + 1) * 128])
            scales[group] = bits
            packed[group * 64:(group + 1) * 64] = pack_w4_codes_scalar(codes)
        yield scales, packed


def encode_w4_bf16(weights: np.ndarray) -> W4BF16Encoded:
    if not hasattr(weights, "shape"):
        raise TypeError("weights must expose a two-dimensional shape")
    rows, columns = _require_shape(weights.shape)
    scales = np.empty((rows, columns // GROUP_SIZE), dtype=np.uint16)
    packed = np.empty((rows, (columns // GROUP_SIZE) * 64), dtype=np.uint8)
    for row, (scale_row, packed_row) in enumerate(iter_encode_w4_bf16_rows((weights[i] for i in range(rows)), columns)):
        scales[row], packed[row] = scale_row, packed_row
    return W4BF16Encoded((rows, columns), scales, packed)


def _as_float32_row_batch(batch: object, columns: int) -> np.ndarray:
    if not isinstance(batch, np.ndarray):
        raise TypeError("W4 BF16 tile batches must be numpy float32 arrays")
    if batch.dtype != np.float32:
        raise TypeError("W4 BF16 tile batches must have dtype float32")
    if batch.ndim != 2 or batch.shape[1] != columns:
        raise ValueError("W4 BF16 tile batch must have shape [rows, in_features]")
    return batch


def _encode_w4_bf16_group_tile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized tile encoder independent from the scalar candidate loop."""
    if values.ndim != 3 or values.shape[2] != GROUP_SIZE or values.dtype != np.float32:
        raise ValueError("W4 BF16 tiles must be float32 [rows, groups, 128]")
    tile_rows, tile_groups, _ = values.shape
    maxima = np.max(np.abs(values), axis=2)
    nonzero = maxima != 0
    scales = np.zeros((tile_rows, tile_groups), dtype=np.uint16)
    best_mse = np.full((tile_rows, tile_groups), np.inf, dtype=np.float32)
    best_codes = np.zeros((tile_rows, tile_groups, GROUP_SIZE), dtype=np.int8)
    for c in W4_CANDIDATE_C:
        candidate_f32 = np.float32(np.float32(maxima) * np.float32(c) / np.float32(7))
        candidate_bits = bf16_rne_bits(candidate_f32)
        candidate_scale = np.asarray(bf16_bits_to_float32(candidate_bits), dtype=np.float32)
        invalid = nonzero & ((~np.isfinite(candidate_scale)) | (candidate_scale <= 0))
        if np.any(invalid):
            raise ValueError("nonzero W4 group has an invalid BF16 candidate scale")
        safe = np.where(nonzero, candidate_scale, np.float32(1))
        codes = np.clip(np.rint(values / safe[:, :, None]), -7, 7).astype(np.int8)
        error = values - safe[:, :, None] * codes.astype(np.float32)
        mse = np.mean(error * error, axis=2, dtype=np.float32)
        choose = nonzero & (mse <= best_mse)
        scales[choose] = candidate_bits[choose]
        best_mse[choose] = mse[choose]
        best_codes[choose] = codes[choose]
    low = best_codes[:, :, 0::2].astype(np.uint8) & np.uint8(0x0F)
    high = (best_codes[:, :, 1::2].astype(np.uint8) & np.uint8(0x0F)) << np.uint8(4)
    return scales, (low | high).reshape(tile_rows, tile_groups * 64)


def iter_encode_w4_bf16_tiles(
    row_batches: Iterable[np.ndarray], columns: int, *, max_rows_per_tile: int = 32,
    max_groups_per_tile: int = 32,
) -> Iterator[W4BF16EncodedTile]:
    _rows, columns = _require_shape((0, columns))
    if (not isinstance(max_rows_per_tile, (int, np.integer)) or isinstance(max_rows_per_tile, bool)
            or max_rows_per_tile <= 0):
        raise ValueError("max_rows_per_tile must be a positive integer")
    if (not isinstance(max_groups_per_tile, (int, np.integer)) or isinstance(max_groups_per_tile, bool)
            or max_groups_per_tile <= 0):
        raise ValueError("max_groups_per_tile must be a positive integer")
    groups, row_offset = columns // GROUP_SIZE, 0
    for batch in row_batches:
        source = _as_float32_row_batch(batch, columns)
        for row_start in range(0, source.shape[0], int(max_rows_per_tile)):
            row_end = min(row_start + int(max_rows_per_tile), source.shape[0])
            for group_start in range(0, groups, int(max_groups_per_tile)):
                group_end = min(group_start + int(max_groups_per_tile), groups)
                tile = source[row_start:row_end, group_start * 128:group_end * 128]
                if not np.all(np.isfinite(tile)):
                    raise ValueError("weights must be finite")
                values = tile.reshape(row_end - row_start, group_end - group_start, GROUP_SIZE)
                scale_bits, packed = _encode_w4_bf16_group_tile(values)
                yield W4BF16EncodedTile(row_offset + row_start, group_start, scale_bits, packed)
        row_offset += source.shape[0]


def unpack_w4_bf16_codes(packed_codes: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    rows, columns = _require_shape(shape)
    groups = columns // GROUP_SIZE
    packed = np.asarray(packed_codes, dtype=np.uint8)
    if packed.shape != (rows, groups * 64):
        raise ValueError("invalid W4 BF16 packed-code shape")
    result = np.empty((rows, columns), dtype=np.int8)
    for row in range(rows):
        for group in range(groups):
            result[row, group * 128:(group + 1) * 128] = unpack_w4_codes_scalar(packed[row, group * 64:(group + 1) * 64])
    return result


def dequantize_w4_bf16(encoded: W4BF16Encoded) -> np.ndarray:
    rows, columns, _groups = _validate_encoded(encoded)
    codes = unpack_w4_bf16_codes(encoded.packed_codes, encoded.shape)
    scales = np.repeat(np.asarray(bf16_bits_to_float32(encoded.scale_bits), dtype=np.float32), GROUP_SIZE, axis=1)
    return np.float32(codes.astype(np.float32) * scales)


def _torch_bf16_bits(values: np.ndarray) -> np.ndarray:
    import torch

    source = torch.from_numpy(np.ascontiguousarray(values.astype(np.float32, copy=False))).to(device="cpu")
    return source.to(torch.bfloat16).view(torch.uint16).cpu().numpy().astype(np.uint16, copy=False)


def selftest() -> None:
    """Synthetic BF16, scalar/tile, wire, and decode checks; no donor access."""
    raw = np.array([
        0x00000000, 0x80000000, 0x00000001, 0x007FFFFF,  # zero/subnormal
        0x3F808000, 0x3F818000, 0x3FFF8000, 0x7F7FFFFF,  # halfway/carry/extreme
        0x7F800000, 0xFF800000, 0x7FC01234,              # nonfinite
    ], dtype="<u4")
    cast_cases = raw.view("<f4")
    actual = bf16_rne_bits(cast_cases)
    expected = _torch_bf16_bits(cast_cases)
    if not np.array_equal(actual, expected):
        raise AssertionError("BF16 RNE bit conversion differs from CPU torch.bfloat16")
    if bf16_rne_bits_scalar(np.float32(cast_cases[4])) != actual[4]:
        raise AssertionError("scalar and vector BF16 RNE differ")

    rng = np.random.default_rng(20260917)
    source = rng.normal(size=(9, 640)).astype(np.float32)
    source[0, :128] = 0
    source[1, 128:256] = np.array([7.0, 0.5, -0.5, 1.5, -1.5] + [0.0] * 123, dtype=np.float32)
    scalar = encode_w4_bf16(source)
    tile_scales = np.empty_like(scalar.scale_bits)
    tile_packed = np.empty_like(scalar.packed_codes)
    for tile in iter_encode_w4_bf16_tiles((source[:4], source[4:]), 640, max_rows_per_tile=2, max_groups_per_tile=3):
        rows, groups = tile.scale_bits.shape
        tile_scales[tile.row_offset:tile.row_offset + rows, tile.group_offset:tile.group_offset + groups] = tile.scale_bits
        tile_packed[tile.row_offset:tile.row_offset + rows, tile.group_offset * 64:(tile.group_offset + groups) * 64] = tile.packed_codes
    tiled = W4BF16Encoded(source.shape, tile_scales, tile_packed)
    if w4_bf16_to_blob(scalar) != w4_bf16_to_blob(tiled):
        raise AssertionError("tile bytes differ from independent scalar reference")
    blob = w4_bf16_to_blob(scalar)
    decoded = dequantize_w4_bf16(w4_bf16_from_blob(blob, source.shape))
    scalar_decoded = np.empty_like(decoded)
    for row in range(source.shape[0]):
        for group in range(source.shape[1] // 128):
            encoded = w4_bf16_from_blob(blob, source.shape)
            scalar_decoded[row, group * 128:(group + 1) * 128] = dequantize_w4_bf16_group_scalar(
                encoded.scale_bits[row, group], unpack_w4_codes_scalar(encoded.packed_codes[row, group * 64:(group + 1) * 64])
            )
    if not np.array_equal(decoded.view("<u4"), scalar_decoded.view("<u4")):
        raise AssertionError("scalar and vector BF16 decode differ")
    try:
        encode_w4_bf16(np.full((1, 128), np.nextafter(np.float32(0), np.float32(1)), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("BF16-underflow candidate did not fail closed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STRAT-02 W4 BF16-scale v2 codec")
    parser.add_argument("--selftest", action="store_true", help="run synthetic-only codec checks")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("STRAT-02 W4 BF16-scale v2 codec selftest: PASS")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
