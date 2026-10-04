"""Fixed signed block-Walsh bases with exact integer coefficient witnesses."""
import hashlib
import numpy as np
import meth448_switch_block_i4_math as B


def fixed_signs(width, label):
    assert label in ('WI', 'WO') and width in (768, 3072)
    return np.asarray([1 if hashlib.sha256(('SiliconLLM|METH451|'+label+'|'+str(i)).encode('ascii')).digest()[0]&1 else -1
                       for i in range(width)], np.int8)


def fwht(value, block):
    assert value.dtype in (np.dtype('int32'), np.dtype('float64'))
    assert block in (4, 16, 256, 1024) and value.shape[-1] % block == 0
    result = value.copy().reshape(-1, block)
    step = 1
    while step < block:
        pairs = result.reshape(-1, block//(2*step), 2, step)
        left, right = pairs[:, :, 0].copy(), pairs[:, :, 1].copy()
        pairs[:, :, 0] = left+right
        pairs[:, :, 1] = left-right
        step *= 2
    return result.reshape(value.shape)


def walsh(block):
    assert block in (4, 16, 256, 1024)
    return np.asarray([[1 if (r&c).bit_count()%2 == 0 else -1 for c in range(block)]
                       for r in range(block)], np.int8)


def rotate_codes(weights, signs, block):
    assert weights.dtype == np.int8 and weights.ndim == 2 and np.all(weights != -128)
    assert signs.dtype == np.int8 and signs.shape == (weights.shape[1],) and np.all(np.abs(signs)==1)
    assert block in (4, 16, 256, 1024) and weights.shape[1]%block == 0
    signed = weights.astype(np.int32)*signs.astype(np.int32)[None, :]
    rotated = fwht(signed, block)
    assert np.max(np.abs(rotated.astype(np.int64))) <= 127*block
    assert 127*block*block < 2**31
    inverse = fwht(rotated, block)
    assert np.array_equal(inverse, signed*block), 'all_coefficient_integer_inverse'
    # Independent explicit Walsh row/parity sums on 3 rows x 3 blocks x 4 outputs.
    for row in sorted(set((0, len(weights)//2, len(weights)-1))):
        for bank in range(weights.shape[1]//block):
            x = signed[row, bank*block:(bank+1)*block].astype(np.int64)
            for r in sorted(set((0, 1, block//2, block-1))):
                w = np.asarray([1 if (r&c).bit_count()%2 == 0 else -1 for c in range(block)], np.int64)
                assert int(rotated[row, bank*block+r]) == int(np.sum(w*x, dtype=np.int64)), 'independent_coefficient_orientation'
    return rotated


def encode(weights, source_scales, signs, block):
    rotated = rotate_codes(weights, signs, block)
    assert source_scales.dtype == np.float32 and source_scales.shape == (len(weights),)
    assert np.isfinite(source_scales).all() and np.all(source_scales > 0)
    # sqrt(block) is an exact power of two in the frozen geometry.
    source = rotated.astype(np.float64)*(source_scales.astype(np.float64)/np.sqrt(block))[:, None]
    assert np.isfinite(source).all() and source.shape[1]%64 == 0
    groups = source.reshape(len(weights), -1, 64)
    maximum = np.max(np.abs(groups), axis=2)
    scales = np.where(maximum == 0, 1., maximum/7.).astype(np.float32)
    assert np.isfinite(scales).all() and np.all(scales > 0)
    codes = np.clip(np.rint(groups/scales.astype(np.float64)[:, :, None]), -7, 7).astype(np.int8).reshape(weights.shape)
    packed = B.pack(codes);assert np.array_equal(B.unpack(packed), codes)
    reconstructed = codes.reshape(groups.shape).astype(np.float64)*scales.astype(np.float64)[:, :, None]
    energy = float(np.sum(groups*groups));error = groups-reconstructed
    metrics = {'relative_Frobenius_error':float(np.sqrt(np.sum(error*error)/max(energy, 1e-30))),
               'zero_blocks':int(np.sum(maximum == 0)), 'matrix_energy':energy,
               'maximum_absolute_coefficient_error':float(np.max(np.abs(error))),
               'integer_inverse_all_coefficients_exact':True,'independent_selected_Walsh_parity_sums_exact':True,
               'max_abs_rotated_integer':int(np.max(np.abs(rotated.astype(np.int64)))),
               'rotated_I32_sha256':hashlib.sha256(rotated.tobytes()).hexdigest()}
    return packed, scales, metrics


class ActivationBasis:
    def __init__(self, signs, block):
        assert signs.dtype == np.int8 and np.all(np.abs(signs)==1)
        self.signs,self.block = signs,block
        self.explicit = walsh(block).astype(np.float64)
        self.calls = 0;self.max_relative_error = 0.
    def apply(self, x):
        assert x.dtype == np.float32 and x.shape == self.signs.shape and np.isfinite(x).all()
        value = fwht(x.astype(np.float64)*self.signs.astype(np.float64), self.block)/np.sqrt(self.block)
        signed = (x.astype(np.float64)*self.signs.astype(np.float64)).reshape(-1, self.block)
        other = (signed @ self.explicit.T/np.sqrt(self.block)).reshape(x.shape)
        error = float(np.linalg.norm(value-other)/max(np.linalg.norm(other), 1e-12))
        assert error <= 1e-12, ('actual_activation_explicit_Walsh',error)
        self.calls += 1;self.max_relative_error = max(self.max_relative_error,error)
        result = value.astype(np.float32);assert np.isfinite(result).all()
        return result


class RotatedSourceOperator:
    def __init__(self, weights, source_scales, signs, block):
        self.codes = rotate_codes(weights, signs, block)
        assert source_scales.dtype == np.float32 and source_scales.shape == (len(weights),) and np.isfinite(source_scales).all() and np.all(source_scales > 0)
        self.scales = source_scales.astype(np.float64)/np.sqrt(block)
        bound = weights.shape[1]*(127*block)*32767
        assert bound < 2**53 and bound < 2**63
        self.calls = 0
    def native(self, x, quant):
        scale, codes = quant(x)
        assert codes.dtype == np.int16 and np.all(codes != -32768)
        weights64 = self.codes.astype(np.int64);activations = codes.astype(np.int64)
        integer = weights64 @ activations
        reference = np.sum(weights64*activations[None, :], axis=1, dtype=np.int64)
        assert np.array_equal(integer, reference), 'rotated_source_I64_integer_primal'
        value = ((integer.astype(np.float64)*self.scales)*np.float64(scale)).astype(np.float32)
        other = np.asarray(np.multiply(np.multiply(reference.astype(np.float64),self.scales),np.float64(scale)),np.float32)
        assert value.tobytes() == other.tobytes() and np.isfinite(value).all()
        self.calls += 1
        return value


def tiny_qualification(quant):
    baseline = B.tiny_qualification(quant)
    for block in (4, 16, 256, 1024):
        explicit = walsh(block).astype(np.int32)
        assert np.array_equal(explicit @ explicit.T, np.eye(block, dtype=np.int32)*block)
        data = np.stack([np.full(block,127,np.int8),np.full(block,-127,np.int8),
                         ((np.arange(block)%255)-127).astype(np.int8),np.zeros(block,np.int8)])
        signs = np.where(np.arange(block)%3 == 0,-1,1).astype(np.int8)
        rotated = rotate_codes(data,signs,block)
        reference = (data.astype(np.int64)*signs.astype(np.int64)[None,:]) @ explicit.astype(np.int64).T
        assert np.array_equal(rotated.astype(np.int64),reference)
        if block in (256,1024):
            basis = ActivationBasis(signs,block)
            x = np.arange(block,dtype=np.float32)*np.float32(2.**-10)
            expected = ((x.astype(np.float64)*signs.astype(np.float64)) @ explicit.astype(np.float64).T/np.sqrt(block)).astype(np.float32)
            assert basis.apply(x).tobytes() == expected.tobytes()
    # Scalar exact-real 4x4 witness; ReLU remains between original-coordinate neurons.
    wi = np.asarray([[2,-1,0,3],[0,1,-2,0],[1,1,1,1],[-2,0,1,-1]],np.int8)
    wo = np.asarray([[1,2,-1,0],[-1,0,3,1]],np.int8)
    signs = np.asarray([1,-1,-1,1],np.int8);x=np.asarray([.5,-1.,2.,.25],np.float64)
    xp=fwht(x*signs,4)/2;wp=rotate_codes(wi,signs,4).astype(np.float64)/2
    up=np.maximum(wp@xp,0);assert np.array_equal(up,np.maximum(wi.astype(np.float64)@x,0))
    down=rotate_codes(wo,signs,4).astype(np.float64)/2 @ (fwht(up*signs,4)/2)
    assert np.array_equal(down,wo.astype(np.float64)@up)
    zero=np.zeros((2,768),np.int8);packed,scales,_=encode(zero,np.asarray([1.,.25],np.float32),fixed_signs(768,'WI'),256)
    assert not B.unpack(packed).any() and np.array_equal(scales,np.ones((2,12),np.float32))
    return {'unchanged_block64_pack_integer_primal':all(baseline.values()),
            'explicit_Walsh_orthogonality_full_integer_products_and_inverse_4_16_256_1024':True,
            'dyadic_activation_F64_transform_F32_cast_byte_exact':True,
            'private_ReLU_exact_real_dyadic_WI_WO_identity':True,'zero_rotated_block_scales_qualified':True}
