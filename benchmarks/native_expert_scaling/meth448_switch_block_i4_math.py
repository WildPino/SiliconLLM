"""One fixed block64-I4 encoding; ascending F64 sum of exact integer blocks."""
from fractions import Fraction
import numpy as np
import meth447_switch_i4_math as Q

BLOCK = 64
pack, unpack = Q.pack, Q.unpack


def encode(weights, source_scales):
    assert weights.dtype == np.int8 and weights.ndim == 2 and np.all(weights != -128)
    rows, width = weights.shape
    assert width % BLOCK == 0
    assert source_scales.dtype == np.float32 and source_scales.shape == (rows,)
    assert np.isfinite(source_scales).all() and np.all(source_scales > 0)
    source = weights.astype(np.float64)*source_scales.astype(np.float64)[:, None]
    blocks = source.reshape(rows, width//BLOCK, BLOCK)
    maximum = np.max(np.abs(blocks), axis=2)
    scales = np.where(maximum == 0, 1., maximum/7.).astype(np.float32)
    assert np.isfinite(scales).all() and np.all(scales > 0)
    codes = np.clip(np.rint(blocks/scales.astype(np.float64)[:, :, None]), -7, 7).astype(np.int8)
    codes = codes.reshape(rows, width)
    packed = pack(codes)
    assert np.array_equal(unpack(packed), codes)
    reconstructed = codes.reshape(rows, width//BLOCK, BLOCK).astype(np.float64)*scales.astype(np.float64)[:, :, None]
    error = blocks-reconstructed
    energy = float(np.sum(blocks*blocks))
    metrics = {'relative_Frobenius_error': float(np.sqrt(np.sum(error*error)/max(energy, 1e-30))),
               'zero_blocks': int(np.sum(maximum == 0)), 'matrix_energy': energy,
               'maximum_absolute_coefficient_error': float(np.max(np.abs(error)))}
    return packed, scales, metrics


class PackedOperator:
    def __init__(self, packed, scales):
        assert packed.dtype == np.uint8 and packed.ndim == 2
        rows, halfwidth = packed.shape
        assert halfwidth*2 % BLOCK == 0
        self.blocks = halfwidth*2//BLOCK
        assert scales.dtype == np.float32 and scales.shape == (rows, self.blocks)
        assert np.isfinite(scales).all() and np.all(scales > 0)
        assert BLOCK*7*32767 == 14679616 and BLOCK*7*32767 < np.iinfo(np.int32).max
        self.packed, self.scales = packed, scales
        low, high = packed & np.uint8(15), packed >> np.uint8(4)
        assert not np.any(low == 8) and not np.any(high == 8), 'unsupported_minus8_nibble'
        self.low = np.where(low >= 8, low.astype(np.int32)-16, low).astype(np.int32).reshape(rows, self.blocks, BLOCK//2)
        self.high = np.where(high >= 8, high.astype(np.int32)-16, high).astype(np.int32).reshape(rows, self.blocks, BLOCK//2)

    def sums_i32(self, codes):
        assert codes.dtype == np.int16 and codes.shape == (self.packed.shape[1]*2,)
        assert np.all(codes != -32768)
        activations = codes.astype(np.int32).reshape(self.blocks, BLOCK)
        low = np.einsum('rbp,bp->rb', self.low, activations[:, ::2], dtype=np.int32, optimize=False)
        high = np.einsum('rbp,bp->rb', self.high, activations[:, 1::2], dtype=np.int32, optimize=False)
        return low+high

    def sums_reference_i64(self, codes):
        # One independent full decode and elementwise I64 product, no cached halves.
        weights = unpack(self.packed).astype(np.int64).reshape(len(self.packed), self.blocks, BLOCK)
        activation = codes.astype(np.int64).reshape(self.blocks, BLOCK)
        return np.sum(weights*activation[None, :, :], axis=2, dtype=np.int64)

    def native(self, x, quant):
        activation_scale, codes = quant(x)
        partials = self.sums_i32(codes)
        reference = self.sums_reference_i64(codes)
        assert np.array_equal(partials.astype(np.int64), reference), 'all_block_I32_I64_primal'
        value64 = np.zeros(len(self.packed), np.float64)
        for b in range(self.blocks):
            value64 += partials[:, b].astype(np.float64)*self.scales[:, b].astype(np.float64)
        value = (value64*np.float64(activation_scale)).astype(np.float32)
        other64 = np.zeros(len(self.packed), np.float64)
        for b in range(self.blocks):
            other64 = np.add(other64, np.multiply(reference[:, b].astype(np.float64), self.scales[:, b].astype(np.float64)))
        other = np.asarray(np.multiply(other64, np.float64(activation_scale)), np.float32)
        assert value.tobytes() == other.tobytes() and np.isfinite(value).all(), 'ascending_F64_block_primal'
        return value


def tiny_qualification(quant):
    common = Q.tiny_qualification(quant)
    pairs = np.asarray([(a,b) for a in range(-7,8) for b in range(-7,8)], np.int8)
    codes_matrix = np.tile(pairs, (1, BLOCK//2))
    operator = PackedOperator(pack(codes_matrix), np.ones((225,1), np.float32))
    for a in (-32767,0,32767):
        for b in (-32767,0,32767):
            activation = np.tile(np.asarray([a,b], np.int16), BLOCK//2)
            expected = np.asarray([(int(x)*a+int(y)*b)*(BLOCK//2) for x,y in pairs], np.int64)[:, None]
            assert np.array_equal(operator.sums_i32(activation).astype(np.int64), expected)
            assert np.array_equal(operator.sums_reference_i64(activation), expected)
    for width in (768,3072):
        weights = np.stack([np.full(width,7,np.int8), np.full(width,-7,np.int8),
                            np.tile(np.asarray([7,-7],np.int8),width//2)])
        operator = PackedOperator(pack(weights), np.ones((3,width//BLOCK), np.float32))
        for sign in (-1,1):
            activation = np.full(width,sign*32767,np.int16)
            expected = np.tile(np.asarray([sign*14679616,-sign*14679616,0],np.int64)[:,None],(1,width//BLOCK))
            assert np.array_equal(operator.sums_i32(activation).astype(np.int64), expected)
    # Four different exact dyadic scale patterns qualify orientation and accumulation.
    weights = np.stack([np.full(128,7,np.int8), np.r_[np.full(64,-7,np.int8),np.full(64,7,np.int8)],
                        np.tile(np.asarray([0,1,-1,7,-7,3,-3,2],np.int8),16), np.zeros(128,np.int8)])
    scales = np.asarray([[.5,2.],[1.,.25],[2.**-8,2.**8],[1.,1.]],np.float32)
    activation = np.tile(np.asarray([32767,-16384,8192,-4096,2048,-1024,512,-256],np.int16),16)
    x = activation.astype(np.float32)*np.float32(.25)
    s, recovered = quant(x)
    assert s == np.float32(.25) and np.array_equal(recovered,activation)
    expected = []
    for row, row_scales in zip(weights,scales):
        exact = Fraction(0)
        for b in range(2):
            integer = sum(int(row[b*64+j])*int(activation[b*64+j]) for j in range(64))
            exact += integer*Fraction.from_float(float(row_scales[b]))
        expected.append(np.float32(float(exact*Fraction(1,4))))
    operator = PackedOperator(pack(weights),scales)
    assert operator.native(x,quant).tobytes() == np.asarray(expected,np.float32).tobytes()
    zero = np.zeros((2,128),np.int8)
    encoded,scales,_ = encode(zero,np.asarray([1.,.25],np.float32))
    assert np.array_equal(scales,np.ones((2,2),np.float32)) and not unpack(encoded).any()
    assert np.array_equal(PackedOperator(encoded,scales).native(np.zeros(128,np.float32),quant),np.zeros(2,np.float32))
    for bad in (8,128):
        invalid = encoded.copy();invalid[-1,-1] = bad
        try: PackedOperator(invalid,scales)
        except AssertionError: pass
        else: raise AssertionError('block_minus8_detection')
    invalid_scale = scales.copy();invalid_scale[-1,-1] = 0
    try: PackedOperator(encoded,invalid_scale)
    except AssertionError: pass
    else: raise AssertionError('zero_scale_detection')
    return {'common_signed_pairs_extrema_nearest_even_pack_qualified':all(common.values()),
            'all225_pairs_times9_block_extrema_and768_3072_bounds_exact':True,
            'dyadic_multiscale_Fraction_oracle_F32_byte_exact':True,
            'zero_blocks_activation_invalid_scale_and_last_nibble_faults_qualified':True}
