"""Deterministic reference codec for the experimental STRAT-02 weight formats.

This module deliberately implements only the frozen W4 and W2 weight layouts:
128 input columns form a group in every ``[out_features, in_features]`` linear
weight row.  It is a codec/reference implementation, not a model loader and not
a claim about model quality or rate.

The full-array encoders allocate their encoded result but visit source weights a
row at a time.  For very large tensors use ``iter_encode_w4_rows`` or
``iter_encode_w2_rows`` and write each yielded row to a caller-owned sink.
``iter_encode_w4_tiles`` is the bounded W4-only path for a stream of F32 row
batches: it yields independently writable row/group tiles and never constructs
a full encoded tensor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Tuple, Union

import numpy as np


GROUP_SIZE = 128
W4_VALUES_PER_BYTE = 2
W2_VALUES_PER_BYTE = 4
W4_CANDIDATE_C = tuple(np.float32(i) / np.float32(100) for i in range(50, 101, 5))

ArrayLike = Union[np.ndarray, object]


def _require_shape(shape: Sequence[int]) -> Tuple[int, int]:
    """Validate an eligible linear weight shape and return integer dimensions."""
    if len(shape) != 2:
        raise ValueError("linear weights must have shape [out_features, in_features]")
    out_features, in_features = int(shape[0]), int(shape[1])
    if out_features < 0 or in_features <= 0 or in_features % GROUP_SIZE:
        raise ValueError("in_features must be positive and divisible by 128")
    return out_features, in_features


def _as_float32_row(row: object, in_features: int) -> np.ndarray:
    """Copy one source row only; this intentionally never materializes a tensor."""
    if isinstance(row, np.ndarray):
        result = np.asarray(row, dtype=np.float32).reshape(-1)
    elif hasattr(row, "detach") and hasattr(row, "cpu"):
        # Torch-like tensors are converted after indexing by the caller.
        result = row.detach().to(dtype=getattr(__import__("torch"), "float32")).cpu().numpy()
        result = np.asarray(result, dtype=np.float32).reshape(-1)
    else:
        result = np.asarray(row, dtype=np.float32).reshape(-1)
    if result.size != in_features:
        raise ValueError("row length does not match in_features")
    if not np.all(np.isfinite(result)):
        raise ValueError("weights must be finite")
    return result


def _weight_rows(weights: ArrayLike) -> Tuple[int, int, Iterator[np.ndarray]]:
    """Return a lazy iterator over rows for ndarray and small Torch-like tensors."""
    if not hasattr(weights, "shape"):
        raise TypeError("weights must expose a two-dimensional shape")
    out_features, in_features = _require_shape(weights.shape)

    def rows() -> Iterator[np.ndarray]:
        for row_index in range(out_features):
            yield _as_float32_row(weights[row_index], in_features)

    return out_features, in_features, rows()


def w4_byte_count(out_features: int, in_features: int) -> int:
    """Exact stored byte count: one fp16 scale and 64 packed code bytes/group."""
    out_features, in_features = _require_shape((out_features, in_features))
    groups = in_features // GROUP_SIZE
    return out_features * groups * (2 + GROUP_SIZE // W4_VALUES_PER_BYTE)


def w2_byte_count(out_features: int, in_features: int) -> int:
    """Exact stored byte count: fp16 ``a,b`` and 32 packed code bytes/group."""
    out_features, in_features = _require_shape((out_features, in_features))
    groups = in_features // GROUP_SIZE
    return out_features * groups * (2 + 2 + GROUP_SIZE // W2_VALUES_PER_BYTE)


def w4_metadata(shape: Sequence[int]) -> dict:
    out_features, in_features = _require_shape(shape)
    return {
        "codec": "STRAT-02-W4-weight-only-experimental",
        "shape": (out_features, in_features),
        "group_size": GROUP_SIZE,
        "scale_dtype": "float16",
        "codes": "signed 4-bit twos-complement, low nibble first, -8 forbidden",
        "candidate_c": tuple(float(c) for c in W4_CANDIDATE_C),
        "byte_count": w4_byte_count(out_features, in_features),
        "byte_count_formula": "out_features * (in_features / 128) * (2 + 64)",
    }


def w2_metadata(shape: Sequence[int]) -> dict:
    out_features, in_features = _require_shape(shape)
    return {
        "codec": "STRAT-02-W2-weight-only-experimental",
        "shape": (out_features, in_features),
        "group_size": GROUP_SIZE,
        "centroid_dtype": "float16",
        "codes": "0:-a, 1:-b, 2:+b, 3:+a; four codes per byte, low bits first",
        "lloyd_iterations": 8,
        "quantile_interpolation": "linear: x[floor((n-1)p)] + fractional_part * delta",
        "byte_count": w2_byte_count(out_features, in_features),
        "byte_count_formula": "out_features * (in_features / 128) * (2 + 2 + 32)",
    }


@dataclass(frozen=True)
class W4Encoded:
    shape: Tuple[int, int]
    scales: np.ndarray  # float16 [out_features, groups]
    packed_codes: np.ndarray  # uint8 [out_features, groups * 64]

    @property
    def metadata(self) -> dict:
        return w4_metadata(self.shape)

    @property
    def byte_count(self) -> int:
        return w4_byte_count(*self.shape)


@dataclass(frozen=True)
class W2Encoded:
    shape: Tuple[int, int]
    a: np.ndarray  # float16 [out_features, groups]
    b: np.ndarray  # float16 [out_features, groups]
    packed_codes: np.ndarray  # uint8 [out_features, groups * 32]

    @property
    def metadata(self) -> dict:
        return w2_metadata(self.shape)

    @property
    def byte_count(self) -> int:
        return w2_byte_count(*self.shape)


@dataclass(frozen=True)
class W4EncodedTile:
    """One independently writable W4 tile from ``iter_encode_w4_tiles``.

    ``row_offset`` is relative to the complete input stream and ``group_offset``
    is measured in frozen 128-value W4 groups.  The tile arrays own their
    storage, so a caller may consume or write them before asking the iterator
    for the next tile.
    """

    row_offset: int
    group_offset: int
    scales: np.ndarray  # float16 [tile_rows, tile_groups]
    packed_codes: np.ndarray  # uint8 [tile_rows, tile_groups * 64]


def w4_to_blob(encoded: W4Encoded) -> bytes:
    """Per-tensor wire bytes: all little-endian scales, then all packed codes."""
    out_features, in_features = _require_shape(encoded.shape)
    groups = in_features // GROUP_SIZE
    if encoded.scales.shape != (out_features, groups) or encoded.scales.dtype != np.float16:
        raise ValueError("invalid W4 scale array")
    if encoded.packed_codes.shape != (out_features, groups * 64) or encoded.packed_codes.dtype != np.uint8:
        raise ValueError("invalid W4 code array")
    if not np.all(np.isfinite(encoded.scales)):
        raise ValueError("W4 scales must be finite")
    packed = encoded.packed_codes
    if np.any((packed & 0x0F) == 8) or np.any((packed >> 4) == 8):
        raise ValueError("invalid W4 stream: -8 is reserved")
    blob = encoded.scales.astype("<f2", copy=False).tobytes(order="C") + packed.tobytes(order="C")
    assert len(blob) == encoded.byte_count
    return blob


def w4_from_blob(blob: bytes, shape: Sequence[int]) -> W4Encoded:
    out_features, in_features = _require_shape(shape)
    if len(blob) != w4_byte_count(out_features, in_features):
        raise ValueError("invalid W4 blob byte count")
    groups = in_features // GROUP_SIZE
    scale_bytes = out_features * groups * 2
    encoded = W4Encoded((out_features, in_features),
                        np.frombuffer(blob[:scale_bytes], dtype="<f2").reshape(out_features, groups),
                        np.frombuffer(blob[scale_bytes:], dtype=np.uint8).reshape(out_features, groups * 64))
    w4_to_blob(encoded)  # validate finite scales and reserved code on read
    return encoded


def w2_to_blob(encoded: W2Encoded) -> bytes:
    """Per-tensor wire bytes: all little-endian a, then b, then packed codes."""
    out_features, in_features = _require_shape(encoded.shape)
    groups = in_features // GROUP_SIZE
    if (encoded.a.shape != (out_features, groups) or encoded.b.shape != (out_features, groups)
            or encoded.a.dtype != np.float16 or encoded.b.dtype != np.float16):
        raise ValueError("invalid W2 centroid arrays")
    if encoded.packed_codes.shape != (out_features, groups * 32) or encoded.packed_codes.dtype != np.uint8:
        raise ValueError("invalid W2 code array")
    if (not np.all(np.isfinite(encoded.a)) or not np.all(np.isfinite(encoded.b))
            or np.any(encoded.a < encoded.b) or np.any(encoded.b < 0)):
        raise ValueError("W2 centroids must satisfy finite a >= b >= 0")
    blob = (encoded.a.astype("<f2", copy=False).tobytes(order="C")
            + encoded.b.astype("<f2", copy=False).tobytes(order="C")
            + encoded.packed_codes.tobytes(order="C"))
    assert len(blob) == encoded.byte_count
    return blob


def w2_from_blob(blob: bytes, shape: Sequence[int]) -> W2Encoded:
    out_features, in_features = _require_shape(shape)
    if len(blob) != w2_byte_count(out_features, in_features):
        raise ValueError("invalid W2 blob byte count")
    groups = in_features // GROUP_SIZE
    centroid_bytes = out_features * groups * 2
    encoded = W2Encoded((out_features, in_features),
                        np.frombuffer(blob[:centroid_bytes], dtype="<f2").reshape(out_features, groups),
                        np.frombuffer(blob[centroid_bytes:2 * centroid_bytes], dtype="<f2").reshape(out_features, groups),
                        np.frombuffer(blob[2 * centroid_bytes:], dtype=np.uint8).reshape(out_features, groups * 32))
    w2_to_blob(encoded)  # validate centroids on read
    return encoded


def pack_w4_codes_scalar(codes: Sequence[int]) -> np.ndarray:
    """Reference packing: signed W4 pairs, first code in each byte's low nibble."""
    if len(codes) % 2:
        raise ValueError("W4 code count must be even")
    packed = np.empty(len(codes) // 2, dtype=np.uint8)
    for byte_index in range(packed.size):
        low, high = int(codes[2 * byte_index]), int(codes[2 * byte_index + 1])
        if not (-7 <= low <= 7 and -7 <= high <= 7):
            raise ValueError("W4 codes must be in [-7, 7]; -8 is forbidden")
        packed[byte_index] = (low & 0xF) | ((high & 0xF) << 4)
    return packed


def unpack_w4_codes_scalar(packed: Sequence[int]) -> np.ndarray:
    """Reference unpacking, including rejection of the reserved -8 representation."""
    codes = np.empty(len(packed) * 2, dtype=np.int8)
    for byte_index, byte in enumerate(packed):
        byte = int(byte)
        if not 0 <= byte <= 255:
            raise ValueError("packed W4 bytes must be in [0, 255]")
        for offset, nibble in enumerate((byte & 0xF, byte >> 4)):
            code = nibble - 16 if nibble >= 8 else nibble
            if code == -8:
                raise ValueError("invalid W4 stream: -8 is reserved")
            codes[2 * byte_index + offset] = code
    return codes


def pack_w2_codes_scalar(codes: Sequence[int]) -> np.ndarray:
    """Reference packing: first W2 code occupies bits 0..1 of each byte."""
    if len(codes) % 4:
        raise ValueError("W2 code count must be divisible by four")
    packed = np.empty(len(codes) // 4, dtype=np.uint8)
    for byte_index in range(packed.size):
        byte = 0
        for offset in range(4):
            code = int(codes[4 * byte_index + offset])
            if not 0 <= code <= 3:
                raise ValueError("W2 codes must be in [0, 3]")
            byte |= code << (2 * offset)
        packed[byte_index] = byte
    return packed


def unpack_w2_codes_scalar(packed: Sequence[int]) -> np.ndarray:
    codes = np.empty(len(packed) * 4, dtype=np.uint8)
    for byte_index, byte in enumerate(packed):
        byte = int(byte)
        if not 0 <= byte <= 255:
            raise ValueError("packed W2 bytes must be in [0, 255]")
        for offset in range(4):
            codes[4 * byte_index + offset] = (byte >> (2 * offset)) & 0x3
    return codes


def dequantize_w4_group_scalar(scale: np.float16, codes: Sequence[int]) -> np.ndarray:
    """Scalar W4 dequantization reference (the stored fp16 scale is authoritative)."""
    if len(codes) != GROUP_SIZE:
        raise ValueError("W4 reference groups must contain exactly 128 codes")
    if not np.isfinite(scale):
        raise ValueError("W4 scale must be finite")
    result = np.empty(GROUP_SIZE, dtype=np.float32)
    for index, code in enumerate(codes):
        code = int(code)
        if not -7 <= code <= 7:
            raise ValueError("W4 codes must be in [-7, 7]; -8 is forbidden")
        result[index] = np.float32(scale) * np.float32(code)
    return result


def dequantize_w2_group_scalar(a: np.float16, b: np.float16, codes: Sequence[int]) -> np.ndarray:
    """Scalar W2 dequantization reference for codebook ``{-a,-b,+b,+a}``."""
    if len(codes) != GROUP_SIZE:
        raise ValueError("W2 reference groups must contain exactly 128 codes")
    if not np.isfinite(a) or not np.isfinite(b) or a < b or b < 0:
        raise ValueError("W2 centroids must be finite and satisfy a >= b >= 0")
    values = (-np.float32(a), -np.float32(b), np.float32(b), np.float32(a))
    result = np.empty(GROUP_SIZE, dtype=np.float32)
    for index, code in enumerate(codes):
        if not 0 <= int(code) <= 3:
            raise ValueError("W2 codes must be in [0, 3]")
        result[index] = values[int(code)]
    return result


def _pack_w4_codes(codes: np.ndarray) -> np.ndarray:
    return pack_w4_codes_scalar(codes.tolist())


def _pack_w2_codes(codes: np.ndarray) -> np.ndarray:
    return pack_w2_codes_scalar(codes.tolist())


def _linear_quantile_scalar(sorted_values: Sequence[np.float32], p: float) -> np.float32:
    """The explicit linear interpolation rule used for W2's 25th and 75th percentiles."""
    position = (len(sorted_values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = np.float32(position - lower)
    return np.float32(sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower]))


def _w4_group_scalar_reference(values: Sequence[float]) -> Tuple[np.float16, np.ndarray]:
    """Independent scalar W4 fit/dequant MSE reference used by the planted tests."""
    if len(values) != GROUP_SIZE:
        raise ValueError("W4 reference groups must contain exactly 128 values")
    values32 = [np.float32(value) for value in values]
    absmax = max(abs(value) for value in values32)
    if absmax == 0:
        return np.float16(0), np.zeros(GROUP_SIZE, dtype=np.int8)
    best_mse = None
    best_scale = np.float16(0)
    best_codes = np.zeros(GROUP_SIZE, dtype=np.int8)
    for c in W4_CANDIDATE_C:
        scale = np.float16(np.float32(absmax) * c / np.float32(7))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("nonzero W4 group has an invalid fp16 candidate scale")
        codes = np.empty(GROUP_SIZE, dtype=np.int8)
        for index, value in enumerate(values32):
            codes[index] = max(-7, min(7, int(round(float(np.float32(value) / np.float32(scale))))))
        mse = np.float32(0)
        scale32 = np.float32(scale)
        for value, code in zip(values32, codes):
            error = np.float32(value - scale32 * np.float32(code))
            mse = np.float32(mse + error * error)
        mse = np.float32(mse / np.float32(GROUP_SIZE))
        # Candidates increase in c, so <= chooses the largest c only on a tie.
        if best_mse is None or mse <= best_mse:
            best_mse, best_scale, best_codes = mse, scale, codes
    return best_scale, best_codes


def _encode_w4_group(values: np.ndarray) -> Tuple[np.float16, np.ndarray]:
    if values.shape != (GROUP_SIZE,):
        raise ValueError("W4 groups must contain exactly 128 values")
    absmax = np.max(np.abs(values), initial=np.float32(0))
    if absmax == 0:
        return np.float16(0), np.zeros(GROUP_SIZE, dtype=np.int8)
    best_mse = None
    best_scale = np.float16(0)
    best_codes = np.zeros(GROUP_SIZE, dtype=np.int8)
    for c in W4_CANDIDATE_C:
        scale = np.float16(np.float32(absmax) * c / np.float32(7))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("nonzero W4 group has an invalid fp16 candidate scale")
        # np.rint is IEEE round-to-nearest with ties-to-even.
        codes = np.clip(np.rint(values / np.float32(scale)), -7, 7).astype(np.int8)
        dequantized = np.float32(scale) * codes.astype(np.float32)
        error = values - dequantized
        mse = np.mean(error * error, dtype=np.float32)
        if best_mse is None or mse <= best_mse:
            best_mse, best_scale, best_codes = mse, scale, codes
    return best_scale, best_codes


def _as_float32_row_batch(batch: object, in_features: int) -> np.ndarray:
    """Validate a caller-owned F32 batch without copying or flattening it."""
    if not isinstance(batch, np.ndarray):
        raise TypeError("W4 tile batches must be numpy float32 arrays")
    if batch.dtype != np.float32:
        raise TypeError("W4 tile batches must have dtype float32")
    if batch.ndim != 2 or batch.shape[1] != in_features:
        raise ValueError("W4 tile batch must have shape [rows, in_features]")
    return batch


def _encode_w4_group_tile(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized W4 fitting for a bounded ``[rows, groups, 128]`` tile.

    The candidate loop, F16 scale rounding, ``np.rint`` tie primitive, and
    ``<=`` final-candidate tie update deliberately match ``_encode_w4_group``.
    """
    if values.ndim != 3 or values.shape[2] != GROUP_SIZE:
        raise ValueError("W4 tiles must have shape [rows, groups, 128]")
    if values.dtype != np.float32:
        raise TypeError("W4 tiles must have dtype float32")

    tile_rows, tile_groups, _ = values.shape
    absolute_maxima = np.max(np.abs(values), axis=2)
    nonzero = absolute_maxima != 0
    scales = np.zeros((tile_rows, tile_groups), dtype=np.float16)
    best_mse = np.full((tile_rows, tile_groups), np.inf, dtype=np.float32)
    best_codes = np.zeros((tile_rows, tile_groups, GROUP_SIZE), dtype=np.int8)

    for c in W4_CANDIDATE_C:
        candidate_scales = np.asarray(
            np.float32(absolute_maxima) * c / np.float32(7), dtype=np.float16
        )
        invalid_scale = nonzero & ((~np.isfinite(candidate_scales)) | (candidate_scales == 0))
        if np.any(invalid_scale):
            raise ValueError("nonzero W4 group has an invalid fp16 candidate scale")

        # Zero groups are an explicit special case in the reference.  Use one
        # only for their temporary division, then exclude them from selection.
        safe_scales = np.where(nonzero, candidate_scales.astype(np.float32), np.float32(1))
        candidate_codes = np.clip(
            np.rint(values / safe_scales[:, :, None]), -7, 7
        ).astype(np.int8)
        dequantized = safe_scales[:, :, None] * candidate_codes.astype(np.float32)
        error = values - dequantized
        mse = np.mean(error * error, axis=2, dtype=np.float32)

        # Candidate C is ascending.  Like the reference's ``mse <= best_mse``,
        # equality overwrites the earlier choice with the later candidate.
        choose = nonzero & (mse <= best_mse)
        scales[choose] = candidate_scales[choose]
        best_mse[choose] = mse[choose]
        best_codes[choose] = candidate_codes[choose]

    low = best_codes[:, :, 0::2].astype(np.uint8) & np.uint8(0x0F)
    high = (best_codes[:, :, 1::2].astype(np.uint8) & np.uint8(0x0F)) << np.uint8(4)
    packed = (low | high).reshape(tile_rows, tile_groups * (GROUP_SIZE // W4_VALUES_PER_BYTE))
    return scales, packed


def iter_encode_w4_tiles(
    row_batches: Iterable[np.ndarray],
    in_features: int,
    *,
    max_rows_per_tile: int = 32,
    max_groups_per_tile: int = 32,
) -> Iterator[W4EncodedTile]:
    """Encode caller-owned F32 batches as bounded, independently writable W4 tiles.

    Source batches must be two-dimensional ``numpy.float32`` arrays with exactly
    ``in_features`` columns.  They may be arbitrarily large because this iterator
    slices them into at most ``max_rows_per_tile * max_groups_per_tile * 128``
    source values per fit.  It yields no whole-model output: each result carries
    its global row and group offsets for a caller-owned streaming sink.

    This is intentionally W4-only.  Its fitting and packing are bit-exact with
    ``iter_encode_w4_rows`` for finite F32 inputs on the active NumPy runtime.
    """
    _, in_features = _require_shape((0, in_features))
    if (not isinstance(max_rows_per_tile, (int, np.integer))
            or isinstance(max_rows_per_tile, bool) or max_rows_per_tile <= 0):
        raise ValueError("max_rows_per_tile must be a positive integer")
    if (not isinstance(max_groups_per_tile, (int, np.integer))
            or isinstance(max_groups_per_tile, bool) or max_groups_per_tile <= 0):
        raise ValueError("max_groups_per_tile must be a positive integer")

    groups = in_features // GROUP_SIZE
    row_offset = 0
    for batch in row_batches:
        source = _as_float32_row_batch(batch, in_features)
        batch_rows = source.shape[0]
        for row_start in range(0, batch_rows, int(max_rows_per_tile)):
            row_end = min(row_start + int(max_rows_per_tile), batch_rows)
            row_tile = source[row_start:row_end]
            for group_start in range(0, groups, int(max_groups_per_tile)):
                group_end = min(group_start + int(max_groups_per_tile), groups)
                values = row_tile[:, group_start * GROUP_SIZE : group_end * GROUP_SIZE]
                if not np.all(np.isfinite(values)):
                    raise ValueError("weights must be finite")
                values = values.reshape(row_end - row_start, group_end - group_start, GROUP_SIZE)
                scales, packed = _encode_w4_group_tile(values)
                yield W4EncodedTile(row_offset + row_start, group_start, scales, packed)
        row_offset += batch_rows


def _assign_abs(abs_values: np.ndarray, a: np.float32, b: np.float32) -> np.ndarray:
    """Return 0 for a and 1 for b; equality deliberately belongs to smaller b."""
    return (np.abs(abs_values - b) <= np.abs(abs_values - a)).astype(np.uint8)


def _w2_group_scalar_reference(values: Sequence[float]) -> Tuple[np.float16, np.float16, np.ndarray]:
    """Independent scalar Lloyd/assignment reference for the W2 layout."""
    if len(values) != GROUP_SIZE:
        raise ValueError("W2 reference groups must contain exactly 128 values")
    source = [np.float32(value) for value in values]
    absolute = [np.float32(abs(value)) for value in source]
    if max(absolute) == 0:
        return np.float16(0), np.float16(0), np.full(GROUP_SIZE, 2, dtype=np.uint8)
    ordered = sorted(absolute)
    b = _linear_quantile_scalar(ordered, 0.25)
    a = _linear_quantile_scalar(ordered, 0.75)
    for _ in range(8):
        assigned_b = [value for value in absolute if abs(value - b) <= abs(value - a)]
        assigned_a = [value for value in absolute if abs(value - b) > abs(value - a)]
        if assigned_b:
            b = np.float32(sum(assigned_b) / len(assigned_b))
        if assigned_a:
            a = np.float32(sum(assigned_a) / len(assigned_a))
        if b > a:
            a, b = b, a
    a16, b16 = np.float16(a), np.float16(b)
    if not np.isfinite(a16) or not np.isfinite(b16):
        raise ValueError("no finite fp16 W2 centroids exist for this group")
    codes = np.empty(GROUP_SIZE, dtype=np.uint8)
    for index, (value, magnitude) in enumerate(zip(source, absolute)):
        use_b = abs(magnitude - np.float32(b16)) <= abs(magnitude - np.float32(a16))
        codes[index] = (1 if value < 0 else 2) if use_b else (0 if value < 0 else 3)
    return a16, b16, codes


def _encode_w2_group(values: np.ndarray) -> Tuple[np.float16, np.float16, np.ndarray]:
    if values.shape != (GROUP_SIZE,):
        raise ValueError("W2 groups must contain exactly 128 values")
    absolute = np.abs(values)
    if np.max(absolute, initial=np.float32(0)) == 0:
        return np.float16(0), np.float16(0), np.full(GROUP_SIZE, 2, dtype=np.uint8)
    ordered = np.sort(absolute)
    b = _linear_quantile_scalar(ordered, 0.25)
    a = _linear_quantile_scalar(ordered, 0.75)
    for _ in range(8):
        is_b = _assign_abs(absolute, a, b).astype(bool)
        if np.any(is_b):
            b = np.mean(absolute[is_b], dtype=np.float32)
        if np.any(~is_b):
            a = np.mean(absolute[~is_b], dtype=np.float32)
        if b > a:
            a, b = b, a
    a16, b16 = np.float16(a), np.float16(b)
    if not np.isfinite(a16) or not np.isfinite(b16):
        raise ValueError("no finite fp16 W2 centroids exist for this group")
    use_b = _assign_abs(absolute, np.float32(a16), np.float32(b16)).astype(bool)
    codes = np.where(values < 0, np.where(use_b, 1, 0), np.where(use_b, 2, 3)).astype(np.uint8)
    return a16, b16, codes


def iter_encode_w4_rows(rows: Iterable[object], in_features: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Encode a row stream, yielding ``(scales, packed_codes)`` per input row."""
    _, in_features = _require_shape((0, in_features))
    groups = in_features // GROUP_SIZE
    for row in rows:
        source = _as_float32_row(row, in_features)
        scales = np.empty(groups, dtype=np.float16)
        packed = np.empty(groups * 64, dtype=np.uint8)
        for group in range(groups):
            scale, codes = _encode_w4_group(source[group * 128 : (group + 1) * 128])
            scales[group] = scale
            packed[group * 64 : (group + 1) * 64] = _pack_w4_codes(codes)
        yield scales, packed


def iter_encode_w2_rows(rows: Iterable[object], in_features: int) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Encode a row stream, yielding ``(a, b, packed_codes)`` per input row."""
    _, in_features = _require_shape((0, in_features))
    groups = in_features // GROUP_SIZE
    for row in rows:
        source = _as_float32_row(row, in_features)
        a_row, b_row = np.empty(groups, dtype=np.float16), np.empty(groups, dtype=np.float16)
        packed = np.empty(groups * 32, dtype=np.uint8)
        for group in range(groups):
            a, b, codes = _encode_w2_group(source[group * 128 : (group + 1) * 128])
            a_row[group], b_row[group] = a, b
            packed[group * 32 : (group + 1) * 32] = _pack_w2_codes(codes)
        yield a_row, b_row, packed


def encode_w4(weights: ArrayLike) -> W4Encoded:
    out_features, in_features, rows = _weight_rows(weights)
    groups = in_features // GROUP_SIZE
    scales = np.empty((out_features, groups), dtype=np.float16)
    packed = np.empty((out_features, groups * 64), dtype=np.uint8)
    for row_index, (scale_row, packed_row) in enumerate(iter_encode_w4_rows(rows, in_features)):
        scales[row_index], packed[row_index] = scale_row, packed_row
    return W4Encoded((out_features, in_features), scales, packed)


def encode_w2(weights: ArrayLike) -> W2Encoded:
    out_features, in_features, rows = _weight_rows(weights)
    groups = in_features // GROUP_SIZE
    a = np.empty((out_features, groups), dtype=np.float16)
    b = np.empty((out_features, groups), dtype=np.float16)
    packed = np.empty((out_features, groups * 32), dtype=np.uint8)
    for row_index, (a_row, b_row, packed_row) in enumerate(iter_encode_w2_rows(rows, in_features)):
        a[row_index], b[row_index], packed[row_index] = a_row, b_row, packed_row
    return W2Encoded((out_features, in_features), a, b, packed)


def unpack_w4_codes(packed_codes: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    out_features, in_features = _require_shape(shape)
    groups = in_features // GROUP_SIZE
    packed = np.asarray(packed_codes, dtype=np.uint8)
    if packed.shape != (out_features, groups * 64):
        raise ValueError("invalid W4 packed-code shape")
    result = np.empty((out_features, in_features), dtype=np.int8)
    for row in range(out_features):
        for group in range(groups):
            result[row, group * 128 : (group + 1) * 128] = unpack_w4_codes_scalar(packed[row, group * 64 : (group + 1) * 64])
    return result


def unpack_w2_codes(packed_codes: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    out_features, in_features = _require_shape(shape)
    groups = in_features // GROUP_SIZE
    packed = np.asarray(packed_codes, dtype=np.uint8)
    if packed.shape != (out_features, groups * 32):
        raise ValueError("invalid W2 packed-code shape")
    result = np.empty((out_features, in_features), dtype=np.uint8)
    for row in range(out_features):
        for group in range(groups):
            result[row, group * 128 : (group + 1) * 128] = unpack_w2_codes_scalar(packed[row, group * 32 : (group + 1) * 32])
    return result


def dequantize_w4(encoded: W4Encoded) -> np.ndarray:
    out_features, in_features = _require_shape(encoded.shape)
    groups = in_features // GROUP_SIZE
    if encoded.scales.shape != (out_features, groups) or encoded.scales.dtype != np.float16:
        raise ValueError("invalid W4 scale shape or dtype")
    if not np.all(np.isfinite(encoded.scales)):
        raise ValueError("W4 scales must be finite")
    codes = unpack_w4_codes(encoded.packed_codes, encoded.shape)
    return codes.astype(np.float32) * np.repeat(encoded.scales.astype(np.float32), GROUP_SIZE, axis=1)


def dequantize_w2(encoded: W2Encoded) -> np.ndarray:
    out_features, in_features = _require_shape(encoded.shape)
    groups = in_features // GROUP_SIZE
    if (encoded.a.shape != (out_features, groups) or encoded.b.shape != (out_features, groups)
            or encoded.a.dtype != np.float16 or encoded.b.dtype != np.float16):
        raise ValueError("invalid W2 centroid shape or dtype")
    if not np.all(np.isfinite(encoded.a)) or not np.all(np.isfinite(encoded.b)) or np.any(encoded.a < encoded.b) or np.any(encoded.b < 0):
        raise ValueError("W2 centroids must be finite and satisfy a >= b >= 0")
    codes = unpack_w2_codes(encoded.packed_codes, encoded.shape)
    a = np.repeat(encoded.a.astype(np.float32), GROUP_SIZE, axis=1)
    b = np.repeat(encoded.b.astype(np.float32), GROUP_SIZE, axis=1)
    return np.where(codes == 0, -a, np.where(codes == 1, -b, np.where(codes == 2, b, a))).astype(np.float32)


def _assert_raises(exception: type, function, *args) -> None:
    try:
        function(*args)
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def selftest() -> None:
    """Planted boundary, tie, signed-zero, malformed-stream, and round-trip tests."""
    w4_boundary = np.array([-7, -1, 0, 1, 7, -7, 7, 0], dtype=np.int8)
    assert np.array_equal(unpack_w4_codes_scalar(pack_w4_codes_scalar(w4_boundary)), w4_boundary)
    _assert_raises(ValueError, unpack_w4_codes_scalar, np.array([0x08], dtype=np.uint8))

    # np.rint is the required IEEE ties-to-even primitive; +/-0.5 therefore map to zero.
    w4_tie = np.zeros(128, dtype=np.float32)
    w4_tie[:3] = (7.0, 0.5, -0.5)
    scale, codes = _encode_w4_group(w4_tie)
    ref_scale, ref_codes = _w4_group_scalar_reference(w4_tie)
    assert np.rint(np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32)).tolist() == [0.0, -0.0, 2.0, -2.0]
    assert scale == ref_scale and np.array_equal(codes, ref_codes)

    # The W2 distance tie goes to b; signed zero follows the non-negative branch.
    assert np.array_equal(_assign_abs(np.array([2], dtype=np.float32), np.float32(3), np.float32(1)), [1])
    zero_w2 = np.full(128, -0.0, dtype=np.float32)
    a, b, zero_codes = _encode_w2_group(zero_w2)
    assert a == 0 and b == 0 and np.all(zero_codes == 2)

    w2_boundary = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.uint8)
    assert np.array_equal(unpack_w2_codes_scalar(pack_w2_codes_scalar(w2_boundary)), w2_boundary)

    rng = np.random.default_rng(20260213)
    source = rng.normal(size=(2, 128)).astype(np.float32)
    source[0, :6] = np.array([-7, -1, -0.0, 0, 1, 7], dtype=np.float32)
    w4_a, w4_b = encode_w4(source), encode_w4(source.copy())
    w2_a, w2_b = encode_w2(source), encode_w2(source.copy())
    # Independently enumerate all candidate reconstruction errors. A fit that
    # accidentally selects the *maximum* MSE must fail this planted test.
    trial = source[1]
    peak = np.max(np.abs(trial))
    candidate_scales = [np.float16(np.float32(peak) * c / np.float32(7)) for c in W4_CANDIDATE_C]
    candidate_mse = []
    for candidate in candidate_scales:
        codes = np.clip(np.rint(trial / np.float32(candidate)), -7, 7)
        residual = trial - np.float32(candidate) * codes
        candidate_mse.append(float(np.mean(residual * residual, dtype=np.float32)))
    assert max(candidate_mse) > min(candidate_mse)
    winning_index = min(range(len(candidate_mse)), key=lambda index: (candidate_mse[index], -index))
    assert w4_a.scales[1, 0] == candidate_scales[winning_index]
    _assert_raises(ValueError, encode_w4, np.full((1, 128), np.finfo(np.float32).tiny, dtype=np.float32))
    assert np.array_equal(w4_a.scales, w4_b.scales) and np.array_equal(w4_a.packed_codes, w4_b.packed_codes)
    assert np.array_equal(w2_a.a, w2_b.a) and np.array_equal(w2_a.b, w2_b.b)
    assert np.array_equal(w2_a.packed_codes, w2_b.packed_codes)
    streamed_w4 = list(iter_encode_w4_rows((source[row] for row in range(2)), 128))
    streamed_w2 = list(iter_encode_w2_rows((source[row] for row in range(2)), 128))
    assert all(np.array_equal(scales, w4_a.scales[row]) and np.array_equal(packed, w4_a.packed_codes[row])
               for row, (scales, packed) in enumerate(streamed_w4))
    assert all(np.array_equal(a_row, w2_a.a[row]) and np.array_equal(b_row, w2_a.b[row])
               and np.array_equal(packed, w2_a.packed_codes[row])
               for row, (a_row, b_row, packed) in enumerate(streamed_w2))
    assert np.array_equal(dequantize_w4(w4_a), dequantize_w4(w4_b))
    assert np.array_equal(dequantize_w2(w2_a), dequantize_w2(w2_b))
    w4_blob = w4_to_blob(w4_a)
    w2_blob = w2_to_blob(w2_a)
    assert len(w4_blob) == 132 and len(w2_blob) == 72
    assert w4_to_blob(w4_from_blob(w4_blob, source.shape)) == w4_blob
    assert w2_to_blob(w2_from_blob(w2_blob, source.shape)) == w2_blob
    assert np.array_equal(dequantize_w4(w4_from_blob(w4_blob, source.shape)), dequantize_w4(w4_a))
    assert np.array_equal(dequantize_w2(w2_from_blob(w2_blob, source.shape)), dequantize_w2(w2_a))
    _assert_raises(ValueError, w4_from_blob, w4_blob[:-1], source.shape)
    _assert_raises(ValueError, w2_from_blob, w2_blob[:-1], source.shape)
    scale_prefix = w4_a.scales.nbytes
    _assert_raises(ValueError, w4_from_blob,
                   w4_blob[:scale_prefix] + bytes([8]) + w4_blob[scale_prefix + 1:], source.shape)
    assert np.array_equal(
        dequantize_w4_group_scalar(w4_a.scales[0, 0], unpack_w4_codes_scalar(w4_a.packed_codes[0, :64])),
        dequantize_w4(w4_a)[0],
    )
    assert np.array_equal(
        dequantize_w2_group_scalar(w2_a.a[0, 0], w2_a.b[0, 0], unpack_w2_codes_scalar(w2_a.packed_codes[0, :32])),
        dequantize_w2(w2_a)[0],
    )
    assert w4_a.byte_count == 132 and w2_a.byte_count == 72
    assert w4_a.byte_count == w4_a.scales.nbytes + w4_a.packed_codes.nbytes
    assert w2_a.byte_count == w2_a.a.nbytes + w2_a.b.nbytes + w2_a.packed_codes.nbytes
    _assert_raises(ValueError, encode_w4, np.zeros((1, 127), dtype=np.float32))
    _assert_raises(ValueError, unpack_w2_codes, np.zeros((2, 31), dtype=np.uint8), (2, 128))

    group = source[1]
    a_fast, b_fast, c_fast = _encode_w2_group(group)
    a_ref, b_ref, c_ref = _w2_group_scalar_reference(group)
    assert a_fast == a_ref and b_fast == b_ref and np.array_equal(c_fast, c_ref)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STRAT-02 deterministic weight-only reference codec")
    parser.add_argument("--selftest", action="store_true", help="run planted deterministic codec checks")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("STRAT-02 weight codec selftest: PASS")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
