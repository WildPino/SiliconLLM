// Stage 1 (c): ggml_bitnet_mul_mat for bitnet.cpp's x86 TL2 path at one token (src1 one column).
//
// bitnet.cpp's current ggml calls ggml_bitnet_mul_mat for TL2 weights, but defines it only for ARM TL1; the x86 TL2
// body survives as a fragment in ggml/src/ggml-bitnet-compute.c, without its batch-1 branch.  This is that fragment
// at batch 1: thread 0 builds the activation tables (ggml_preprocessor), then every thread runs bitnet.cpp's generated
// kernels (ggml_qgemm_lut) over its share of the row tiles, the three-weight part of K first, then the two-weight tail,
// which adds into the same output.  s1k_bench's correctness check reads the result.
//
// Build: g++ -O2 -march=native -DGGML_BITNET_X86_TL2 -c s1k_tl2_glue.cpp -I<BitNet>/include
//   -I<BitNet>/3rdparty/llama.cpp/ggml/include -I<BitNet>/3rdparty/llama.cpp/ggml/src
//   -I<BitNet>/3rdparty/llama.cpp/ggml/src/ggml-cpu
#include "ggml-bitnet.h"
#include "ggml-cpu-impl.h"

extern "C" void ggml_bitnet_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int ne00 = (int)src0->ne[0], ne01 = (int)src0->ne[1], ne10 = (int)src1->ne[0], ne11 = (int)src1->ne[1];
    GGML_ASSERT(ne11 == 1 && "this glue covers one token only");
    GGML_ASSERT(sizeof(bitnet_float_type) == 4);
    const struct bitnet_tensor_extra * wt = (const struct bitnet_tensor_extra *)src0->extra;
    GGML_ASSERT(wt != nullptr);
    const int ith = params->ith, nth = params->nth;

    const int three_k = ne10 / wt->BK * wt->BK, two_k = ne10 - three_k;
    int8_t * three_qlut = (int8_t *)params->wdata;
    bitnet_float_type * lut_scales = (bitnet_float_type *)(three_qlut + three_k / 3 * 16 * 2 * ne11);
    int8_t * two_qlut = (int8_t *)(lut_scales + ne11);
    if (ith == 0) {
        ggml_preprocessor(ne11, ne01, three_k, two_k, src1->data, lut_scales, three_qlut, two_qlut);
    }
    ggml_barrier(params->threadpool);

    float * out = (float *)dst->data;
    const int n_tile_num = wt->n_tile_num;
    GGML_ASSERT(ne01 % n_tile_num == 0);
    const int w_tile_size = three_k * ne01 / (2 * 3) / n_tile_num;
    const int c_tile_size = ne01 / n_tile_num;
    const int sign_tile_size = three_k * ne01 / 24 / n_tile_num;
    const int th_tile_num = (n_tile_num + nth - 1) / nth;
    const int th_tile_beg = ith * th_tile_num;
    const int th_tile_end = (ith + 1) * th_tile_num < n_tile_num ? (ith + 1) * th_tile_num : n_tile_num;
    uint8_t * qw = (uint8_t *)wt->qweights;
    uint8_t * sign = qw + three_k * ne01 / 3 / 2;
    uint8_t * two_A = qw + three_k * ne01 / 3 / 2 + three_k * ne01 / 3 / 8;
    const int two_w_tile_size = ne01 * two_k / (2 * 2) / n_tile_num;
    for (int t = th_tile_beg; t < th_tile_end; t++) {
        ggml_qgemm_lut(1, ne01, ne00, three_k, qw + t * w_tile_size, sign + t * sign_tile_size, three_qlut,
                       wt->scales, lut_scales, out + t * c_tile_size);
        if (two_k > 0) {
            ggml_qgemm_lut(1, ne01, ne00, two_k, two_A + t * two_w_tile_size, nullptr, two_qlut, wt->scales, lut_scales,
                           out + t * c_tile_size);
        }
    }
}
