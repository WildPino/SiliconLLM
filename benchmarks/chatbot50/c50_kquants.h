/*
 * Q8_0 x Q8_0, IQ4_XS x Q8_K and IQ3_S x Q8_K dot products (generic + AVX2).
 *
 * Literal transcriptions of llama.cpp ggml/src/ggml-cpu/quants.c (generic arms)
 * and ggml/src/ggml-cpu/arch/x86/quants.c (AVX2 arms) at
 * 5b335f413e4f73b0809c4fe39af894efbcc6a0d2 (MIT).  No GGML build or runtime
 * dependency.  Q4_K and Q6_K reuse benchmarks/phase60/strat01_*.h.
 */
#ifndef C50_KQUANTS_H
#define C50_KQUANTS_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "../phase60/strat01_q6k_q8k_avx2.h"
#include "c50_iq_tables.h"

#define C50_QK8_0 32U

typedef struct { uint16_t d; int8_t qs[C50_QK8_0]; } c50_block_q8_0;
typedef struct { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; } c50_block_iq4_xs;
typedef struct { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; } c50_block_iq3_s;

typedef char c50_q8_0_is_34[sizeof(c50_block_q8_0) == 34 ? 1 : -1];
typedef char c50_iq4_xs_is_136[sizeof(c50_block_iq4_xs) == 136 ? 1 : -1];
typedef char c50_iq3_s_is_110[sizeof(c50_block_iq3_s) == 110 ? 1 : -1];

static const uint8_t c50_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static inline float c50_h2f(uint16_t h) {
    uint8_t b[2] = {(uint8_t)(h & 255U), (uint8_t)(h >> 8)};
    return strat01_q4k_q8k_fp16le(b);
}

/* quantize_row_q8_0_ref: d = amax / 127, q = round(x / d). */
static void c50_quantize_row_q8_0(const float *x, c50_block_q8_0 *y, unsigned k) {
    unsigned i, j;
    for (i = 0; i < k / C50_QK8_0; ++i) {
        float amax = 0.0f, d, id;
        for (j = 0; j < C50_QK8_0; ++j) {
            float v = fabsf(x[i * C50_QK8_0 + j]);
            if (v > amax) amax = v;
        }
        d = amax / 127.0f;
        id = d ? 1.0f / d : 0.0f;
        y[i].d = _cvtss_sh(d, 0);
        for (j = 0; j < C50_QK8_0; ++j) y[i].qs[j] = (int8_t)roundf(x[i * C50_QK8_0 + j] * id);
    }
}

/* ------------------------------------------------------------------ Q8_0 -- */
static float c50_q8_0_dot_generic(const c50_block_q8_0 *x, const c50_block_q8_0 *y, unsigned n) {
    float sumf = 0.0f;
    unsigned ib, j;
    for (ib = 0; ib < n / C50_QK8_0; ++ib) {
        int sumi = 0;
        for (j = 0; j < C50_QK8_0; ++j) sumi += x[ib].qs[j] * y[ib].qs[j];
        sumf += sumi * (c50_h2f(x[ib].d) * c50_h2f(y[ib].d));
    }
    return sumf;
}

static inline float c50_hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i c50_mul_add_epi8(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

static inline __m256 c50_mul_sum_i8_pairs_float(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), dot));
}

static float c50_q8_0_dot_avx2(const c50_block_q8_0 *x, const c50_block_q8_0 *y, unsigned n) {
    __m256 acc = _mm256_setzero_ps();
    unsigned ib;
    for (ib = 0; ib < n / C50_QK8_0; ++ib) {
        const __m256 d = _mm256_set1_ps(_cvtsh_ss(x[ib].d) * _cvtsh_ss(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        acc = _mm256_fmadd_ps(d, c50_mul_sum_i8_pairs_float(qx, qy), acc);
    }
    return c50_hsum_float_8(acc);
}

/* ---------------------------------------------------------------- IQ4_XS -- */
static float c50_iq4_xs_dot_generic(const c50_block_iq4_xs *x, const strat01_q8_k_block *y, unsigned n) {
    float sumf = 0.0f;
    unsigned ibl;
    for (ibl = 0; ibl < n / STRAT01_QK_K; ++ibl) {
        const float d4d8 = c50_h2f(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t *qs = x[ibl].qs;
        const int8_t *q8 = y[ibl].qs;
        unsigned ib;
        for (ib = 0; ib < STRAT01_QK_K / 32; ib += 2) {
            const uint8_t ls1 = (uint8_t)((x[ibl].scales_l[ib / 2] & 0xf) | ((h << 4) & 0x30));
            const uint8_t ls2 = (uint8_t)((x[ibl].scales_l[ib / 2] >> 4) | ((h << 2) & 0x30));
            const float d1 = d4d8 * (ls1 - 32);
            const float d2 = d4d8 * (ls2 - 32);
            int sumi1 = 0, sumi2 = 0, j;
            h >>= 4;
            for (j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * c50_kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * c50_kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * c50_kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * c50_kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    return sumf;
}

static float c50_iq4_xs_dot_avx2(const c50_block_iq4_xs *x, const strat01_q8_k_block *y, unsigned n) {
    const __m128i values128 = _mm_loadu_si128((const __m128i *)c50_kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    __m256 accum = _mm256_setzero_ps();
    unsigned ibl;
    for (ibl = 0; ibl < n / STRAT01_QK_K; ++ibl) {
        const uint8_t *qs = x[ibl].qs;
        const int8_t *q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        unsigned ib;
        for (ib = 0; ib < STRAT01_QK_K / 32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q4b_1 = _mm256_set_m128i(
                _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            const __m256i q4b_2 = _mm256_set_m128i(
                _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            const __m256i p16_1 = c50_mul_add_epi8(q4b_1, q8b_1);
            const __m256i p16_2 = c50_mul_add_epi8(q4b_2, q8b_2);
            const int16_t ls1 = (int16_t)(((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const int16_t ls2 = (int16_t)(((x[ibl].scales_l[ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32);
            sh >>= 4;
            sumi1 = _mm256_add_epi32(_mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1)), sumi1);
            sumi2 = _mm256_add_epi32(_mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2)), sumi2);
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(x[ibl].d) * y[ibl].d),
                                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }
    return c50_hsum_float_8(accum);
}

/* ----------------------------------------------------------------- IQ3_S -- */
static float c50_iq3_s_dot_generic(const c50_block_iq3_s *x, const strat01_q8_k_block *y, unsigned n) {
    float sumf = 0.0f;
    unsigned i;
    for (i = 0; i < n / STRAT01_QK_K; ++i) {
        const float d = c50_h2f(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;
        unsigned ib32;
        for (ib32 = 0; ib32 < STRAT01_QK_K / 32; ib32 += 2) {
            const uint32_t ls1 = 2 * (x[i].scales[ib32 / 2] & 0xf) + 1;
            const uint32_t ls2 = 2 * (x[i].scales[ib32 / 2] >> 4) + 1;
            int32_t sumi = 0;
            int l, j;
            for (l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)(c50_iq3s_grid + (qs[2 * l + 0] | ((qh[ib32 + 0] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *)(c50_iq3s_grid + (qs[2 * l + 1] | ((qh[ib32 + 0] << (7 - 2 * l)) & 256)));
                for (j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j + 0] * (signs[l] & c50_kmask_iq2xs[j + 0] ? -1 : 1);
                    sumi += grid2[j] * q8[j + 4] * (signs[l] & c50_kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * (int32_t)ls1;
            sumi = 0;
            for (l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)(c50_iq3s_grid + (qs[2 * l + 0] | ((qh[ib32 + 1] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *)(c50_iq3s_grid + (qs[2 * l + 1] | ((qh[ib32 + 1] << (7 - 2 * l)) & 256)));
                for (j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j + 0] * (signs[l] & c50_kmask_iq2xs[j + 0] ? -1 : 1);
                    sumi += grid2[j] * q8[j + 4] * (signs[l] & c50_kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * (int32_t)ls2;
        }
        sumf += d * bsum;
    }
    return sumf;
}

static float c50_iq3_s_dot_avx2(const c50_block_iq3_s *x, const strat01_q8_k_block *y, unsigned n) {
    static const uint8_t k_mask1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    static const uint8_t k_mask2[32] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
                                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *)k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask = _mm256_set1_epi32(256);
    union { __m256i vec[2]; uint32_t index[16]; } idx;
    __m256 accumf = _mm256_setzero_ps();
    unsigned i;
    for (i = 0; i < n / STRAT01_QK_K; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint16_t *signs = (const uint16_t *)x[i].signs;
        const int8_t *q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        unsigned ib32;
        for (ib32 = 0; ib32 < STRAT01_QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            __m256i aux256, s2_1, s2_2, q8s_1, q8s_2, q2_1, q2_2, dot1, dot2;
            uint16_t ls1, ls2;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32 + 0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32 + 1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            q2_1 = _mm256_set_epi32(
                (int)c50_iq3s_grid[idx.index[7]], (int)c50_iq3s_grid[idx.index[6]],
                (int)c50_iq3s_grid[idx.index[5]], (int)c50_iq3s_grid[idx.index[4]],
                (int)c50_iq3s_grid[idx.index[3]], (int)c50_iq3s_grid[idx.index[2]],
                (int)c50_iq3s_grid[idx.index[1]], (int)c50_iq3s_grid[idx.index[0]]);
            q2_2 = _mm256_set_epi32(
                (int)c50_iq3s_grid[idx.index[15]], (int)c50_iq3s_grid[idx.index[14]],
                (int)c50_iq3s_grid[idx.index[13]], (int)c50_iq3s_grid[idx.index[12]],
                (int)c50_iq3s_grid[idx.index[11]], (int)c50_iq3s_grid[idx.index[10]],
                (int)c50_iq3s_grid[idx.index[9]], (int)c50_iq3s_grid[idx.index[8]]);
            aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 4;
            dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            ls1 = x[i].scales[ib32 / 2] & 0xf;
            ls2 = x[i].scales[ib32 / 2] >> 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16((short)(2 * ls1 + 1))));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16((short)(2 * ls2 + 1))));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return c50_hsum_float_8(accumf);
}

/* ------------------------------------------------- vector exp / silu (AVX2) -- */
/* ggml_v_expf and ggml_v_silu, ggml/src/ggml-cpu/vec.h at the same pin. */
static inline __m256 c50_v_expf(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                        _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    __m256i g, d;
    __m256 s1, s2;
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) return _mm256_fmadd_ps(j, k, k);
    g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
                         _mm256_set1_epi32((int)0x82000000u));
    s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000)));
    s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    d = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(d),
                         _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                                      _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

static inline __m256 c50_v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 exp_neg_x = c50_v_expf(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(x, _mm256_add_ps(one, exp_neg_x));
}

#endif
