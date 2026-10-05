"""Original I8 WO in column storage; skip only exact zero A16 integer codes."""
from fractions import Fraction
import numpy as np

CHUNK=512
BOUND=CHUNK*127*32767
assert BOUND==2130641408 and BOUND<2**31


class SparseColumns:
    def __init__(self, columns, scales):
        assert columns.dtype==np.int8 and columns.ndim==2 and columns.flags.c_contiguous
        assert np.all(columns != -128) and columns.shape[0]<=65536
        assert scales.dtype==np.float32 and scales.shape==(columns.shape[1],)
        assert np.isfinite(scales).all() and np.all(scales>0)
        self.columns,self.scales=columns,scales
    def sums(self, codes):
        assert codes.dtype==np.int16 and codes.shape==(len(self.columns),) and np.all(codes!=-32768)
        indices=np.flatnonzero(codes).astype(np.uint16)
        total=np.zeros(self.columns.shape[1],np.int64)
        for start in range(0,len(indices),CHUNK):
            chosen=indices[start:start+CHUNK]
            values=self.columns[chosen].astype(np.int32)
            products=values*codes[chosen].astype(np.int32)[:,None]
            partial=np.sum(products,axis=0,dtype=np.int32)
            assert np.max(np.abs(partial.astype(np.int64)))<=BOUND
            total+=partial.astype(np.int64)
        # Independent FULL matrix, including zero-coded columns, fresh I64 products.
        reference=np.sum(self.columns.astype(np.int64)*codes.astype(np.int64)[:,None],axis=0,dtype=np.int64)
        assert np.array_equal(total,reference), 'sparse_I32_chunks_full_I64_integer_primal'
        return total,indices
    def native(self,x,quant):
        assert x.dtype==np.float32 and np.isfinite(x).all()
        scale,codes=quant(x);integer,indices=self.sums(codes)
        value=((integer.astype(np.float64)*self.scales.astype(np.float64))*np.float64(scale)).astype(np.float32)
        other=np.asarray(np.multiply(np.multiply(integer.astype(np.float64),self.scales.astype(np.float64)),np.float64(scale)),np.float32)
        assert value.tobytes()==other.tobytes() and np.isfinite(value).all()
        return value,scale,codes,indices


def footprint(indices,source_offset_phase):
    assert indices.dtype==np.uint16 and np.all(indices<3072) and 0<=source_offset_phase<64
    groups=np.unique((indices.astype(np.int64)+source_offset_phase)//64)
    overlap=767 if 0 in groups and 48 in groups else 0
    row_lines=768*len(groups)-overlap
    # Proposed64-aligned candidate column layout; NOT NPZ placement or measured DRAM.
    column_lines=len(indices)*12
    coefficients=1327104+768*len(indices)+3072
    workspace={'WI_signs':768,'original_input_F32':3072,'transformed_input_F32':3072,'WI_A16_codes':1536,
        'raw_F32':12288,'ReLU_F32':12288,'WO_A16_codes':6144,'allocated_U16_indices':6144,
        'I32_partial':3072,'I64_accumulator':6144,'WO_output_F32':3072,
        'post_final_head_inputs_F32':9216,'FFN_final_norm_weights_F32':6144,
        'inplace_WI_basis_F64_workspace':6144,'WI_block_I32_partial':12288,
        'WI_weighted_F64_accumulator':24576,'WO_scaled_F64_output':6144,
        'two_A16_scales_probability_index_count':16}
    addressed=coefficients+sum(workspace.values())
    return {'nonzero_codes':int(len(indices)),'zero_fraction':float(1-len(indices)/3072),
        'source_rowmajor_distinct_64B_lines':int(row_lines),'source_rowmajor_offset_phase':int(source_offset_phase),
        'candidate64aligned_column_distinct_64B_lines':int(column_lines),
        'coefficient_scale_bytes':int(coefficients),'unique_addressed_payload_workspace_bytes':int(addressed),
        'ratio_to_original4733952_payload':float(addressed/4733952),'workspace_components':workspace,
        'scope':'Unique logical addressed footprint. Original source row-major offset phase included; proposed candidate64 alignment declared. Not traffic, timing, cache residency or hardware DRAM. Complete core/router/head/prefill/C glue still charged later.'}


def tiny_qualification(quant):
    for width in (4,512,1024):
        weights=np.stack([np.full(width,127,np.int8),np.full(width,-127,np.int8),
            np.where(np.arange(width)%2==0,127,-127).astype(np.int8)])
        operator=SparseColumns(np.ascontiguousarray(weights.T),np.asarray([.5,2.,.25],np.float32))
        for sign in (-1,1):
            codes=np.full(width,sign*32767,np.int16)
            integer,indices=operator.sums(codes)
            expected=np.asarray([sum(int(w)*int(c) for w,c in zip(row,codes)) for row in weights],np.int64)
            assert np.array_equal(integer,expected) and len(indices)==width
            x=codes.astype(np.float32)*np.float32(.25)
            value,scale,recovered,_=operator.native(x,quant)
            assert scale==np.float32(.25) and np.array_equal(recovered,codes)
            exact=np.asarray([np.float32(float(Fraction(int(n))*Fraction.from_float(float(s))*Fraction(1,4))) for n,s in zip(expected,operator.scales)],np.float32)
            assert value.tobytes()==exact.tobytes()
        zero=np.zeros(width,np.int16);integer,indices=operator.sums(zero)
        assert not integer.any() and len(indices)==0
        one=zero.copy();one[-1]=32767;integer,indices=operator.sums(one)
        assert np.array_equal(integer,weights[:,-1].astype(np.int64)*32767) and indices.tolist()==[width-1]
        value,scale,codes,indices=operator.native(np.zeros(width,np.float32),quant)
        assert not value.any() and scale==np.float32(1) and len(indices)==0
    invalid=np.asarray([[-128]],np.int8)
    try:SparseColumns(invalid,np.ones(1,np.float32))
    except AssertionError:pass
    else:raise AssertionError('source_minus128_rejected')
    invalid_scale=np.zeros(3,np.float32)
    try:SparseColumns(np.ascontiguousarray(weights.T),invalid_scale)
    except AssertionError:pass
    else:raise AssertionError('invalid_source_scale_rejected')
    # Independent aligned line-set enumeration, including unaligned row boundaries.
    for phase in (0,17,63):
        indices=np.asarray([0,1,63,64,3071],np.uint16);observed=footprint(indices,phase)
        lines={int((phase+r*3072+int(j))//64) for r in range(768) for j in indices}
        assert observed['source_rowmajor_distinct_64B_lines']==len(lines)
    return {'signed_extrema_zero_single_column_I32_512_full_I64_1024_bound_exact':True,
        'dyadic_scale_cast_Fraction_oracle_exact':True,'invalid_weight_scale_faults_qualified':True,
        'source_unaligned_row_boundary_distinct_line_sets_qualified':True}
