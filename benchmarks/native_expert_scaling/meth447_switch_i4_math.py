"""Fixed symmetric row-I4 encoding and independently qualified A16 products."""
import numpy as np


def pack(codes):
    assert codes.dtype == np.int8 and codes.ndim == 2 and codes.shape[1] % 2 == 0
    assert np.all((codes >= -7) & (codes <= 7))
    nibble = codes.astype(np.uint8) & np.uint8(15)
    return nibble[:, ::2] | (nibble[:, 1::2] << np.uint8(4))


def unpack(packed):
    assert packed.dtype == np.uint8 and packed.ndim == 2
    low, high = packed & np.uint8(15), packed >> np.uint8(4)
    assert not np.any(low == 8) and not np.any(high == 8), 'unsupported_minus8_nibble'
    decoded = np.empty((len(packed), packed.shape[1]*2), dtype=np.int8)
    decoded[:, ::2] = np.where(low >= 8, low.astype(np.int16)-16, low).astype(np.int8)
    decoded[:, 1::2] = np.where(high >= 8, high.astype(np.int16)-16, high).astype(np.int8)
    return decoded


def encode(weights, source_scales):
    assert weights.dtype == np.int8 and weights.ndim == 2 and np.all(weights != -128)
    assert source_scales.dtype == np.float32 and source_scales.shape == (len(weights),)
    assert np.isfinite(source_scales).all() and np.all(source_scales > 0)
    dequant = weights.astype(np.float64)*source_scales.astype(np.float64)[:, None]
    maximum = np.max(np.abs(dequant), axis=1)
    scales = np.where(maximum == 0, 1., maximum/7.).astype(np.float32)
    assert np.isfinite(scales).all() and np.all(scales > 0)
    codes = np.clip(np.rint(dequant/scales.astype(np.float64)[:, None]), -7, 7).astype(np.int8)
    packed = pack(codes)
    assert np.array_equal(unpack(packed), codes)
    reconstructed = codes.astype(np.float64)*scales.astype(np.float64)[:, None]
    error = dequant-reconstructed
    energy = float(np.sum(dequant*dequant))
    metrics = {'relative_Frobenius_error': float(np.sqrt(np.sum(error*error)/max(energy, 1e-30))),
               'zero_rows': int(np.sum(maximum == 0)), 'matrix_energy': energy,
               'maximum_absolute_coefficient_error': float(np.max(np.abs(error)))}
    return packed, scales, metrics


class PackedOperator:
    def __init__(self, packed, scales):
        assert packed.dtype == np.uint8 and packed.ndim == 2
        assert scales.dtype == np.float32 and scales.shape == (len(packed),)
        assert np.isfinite(scales).all() and np.all(scales > 0)
        self.packed, self.scales = packed, scales
        width = packed.shape[1]*2
        assert width*7*32767 <= np.iinfo(np.int32).max, 'I32_accumulation_bound'
        low, high = packed & np.uint8(15), packed >> np.uint8(4)
        assert not np.any(low == 8) and not np.any(high == 8), 'unsupported_minus8_nibble'
        self.low = np.where(low >= 8, low.astype(np.int32)-16, low).astype(np.int32)
        self.high = np.where(high >= 8, high.astype(np.int32)-16, high).astype(np.int32)

    def sums_i32(self, codes):
        assert codes.dtype == np.int16 and codes.shape == (self.packed.shape[1]*2,)
        assert np.all(codes != -32768)
        codes32 = codes.astype(np.int32)
        return self.low @ codes32[::2] + self.high @ codes32[1::2]

    def sums_reference_i64(self, codes):
        # Decode afresh through the separate full-matrix path, not the cached halves.
        return unpack(self.packed).astype(np.int64) @ codes.astype(np.int64)

    def native(self, x, quant):
        scale, codes = quant(x)
        sums = self.sums_i32(codes)
        reference = self.sums_reference_i64(codes)
        assert np.array_equal(sums.astype(np.int64), reference), 'I32_I64_integer_primal'
        value = ((sums.astype(np.float64)*self.scales.astype(np.float64))*np.float64(scale)).astype(np.float32)
        other = ((reference.astype(np.float64)*self.scales.astype(np.float64))*np.float64(scale)).astype(np.float32)
        assert value.tobytes() == other.tobytes() and np.isfinite(value).all()
        return value


def tiny_qualification(quant):
    pairs = np.asarray([(a,b) for a in range(-7,8) for b in range(-7,8)], np.int8)
    encoded = pack(pairs)
    assert np.array_equal(unpack(encoded), pairs)
    operator = PackedOperator(encoded, np.ones(225, np.float32))
    for a in (-32767,0,32767):
        for b in (-32767,0,32767):
            codes = np.asarray([a,b],np.int16)
            expected = np.asarray([int(x)*a+int(y)*b for x,y in pairs],np.int64)
            assert np.array_equal(operator.sums_i32(codes).astype(np.int64), expected)
            assert np.array_equal(operator.sums_reference_i64(codes), expected)
    extreme = np.stack([np.full(3072,7,np.int8), np.full(3072,-7,np.int8),
                        np.tile(np.asarray([7,-7],np.int8),1536)])
    operator = PackedOperator(pack(extreme),np.ones(3,np.float32))
    for sign in (-1,1):
        codes = np.full(3072,sign*32767,np.int16)
        assert np.array_equal(operator.sums_i32(codes).astype(np.int64),
                              np.asarray([sign*704621568,-sign*704621568,0],np.int64))
    for bad in (8,128):
        try: unpack(np.asarray([[bad]],np.uint8))
        except AssertionError: pass
        else: raise AssertionError('minus8_detection')
    zero = np.zeros((2,4),np.int8)
    encoded,scales,_ = encode(zero,np.asarray([1.,.25],np.float32))
    assert np.array_equal(scales,np.ones(2,np.float32)) and not unpack(encoded).any()
    operator = PackedOperator(encoded,scales)
    assert np.array_equal(operator.native(np.zeros(4,np.float32),quant),np.zeros(2,np.float32))
    assert np.array_equal(np.rint(np.asarray([-2.5,-1.5,-.5,.5,1.5,2.5])),[-2,-2,0,0,2,2])
    return {'all225_signed_pairs_times9_extremal_activation_pairs_exact':True,
            'full3072_I32_positive_negative_cancellation_bound_exact':True,
            'low_and_high_minus8_rejected_zero_rows_nearest_even_qualified':True}
