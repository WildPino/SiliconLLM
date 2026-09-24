/*
 * Standalone Q6_K x Q8_K reduction-parity primitives.
 *
 * The generic arm preserves SiliconLLM's existing scalar reduction.  The
 * AVX2 arm is a literal transcription of llama.cpp/GGML x86 quants.c at
 * 5b335f413e4f73b0809c4fe39af894efbcc6a0d2.  This header intentionally has
 * no GGML build or runtime dependency.
 */
#ifndef STRAT01_Q6K_Q8K_AVX2_H
#define STRAT01_Q6K_Q8K_AVX2_H

#include "strat01_q4k_q8k.h"

#define STRAT01_Q6_K_BLOCK_BYTES 210U

static float strat01_q6k_q8k_dot_generic(
        const uint8_t *q6_blocks, const strat01_q8_k_block *q8_blocks,
        unsigned count) {
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f};
    float result = 0.0f;
    unsigned block;

    if (!q6_blocks || !q8_blocks || !count || count % STRAT01_QK_K) return NAN;
    for (block = 0; block < count / STRAT01_QK_K; ++block) {
        const uint8_t *raw = q6_blocks + (size_t)block * STRAT01_Q6_K_BLOCK_BYTES;
        const uint8_t *q4 = raw;
        const uint8_t *qh = raw + 128U;
        const int8_t *scales = (const int8_t *)(raw + 192U);
        const int8_t *q8 = q8_blocks[block].qs;
        int8_t unpacked[STRAT01_QK_K];
        int32_t lanes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t *destination = unpacked;
        unsigned group, lane;

        for (group = 0; group < STRAT01_QK_K; group += 128U) {
            for (lane = 0; lane < 32U; ++lane) {
                destination[lane + 0U] =
                    (int8_t)((q4[lane + 0U] & 15U) | (((qh[lane] >> 0) & 3U) << 4)) - 32;
                destination[lane + 32U] =
                    (int8_t)((q4[lane + 32U] & 15U) | (((qh[lane] >> 2) & 3U) << 4)) - 32;
                destination[lane + 64U] =
                    (int8_t)((q4[lane + 0U] >> 4) | (((qh[lane] >> 4) & 3U) << 4)) - 32;
                destination[lane + 96U] =
                    (int8_t)((q4[lane + 32U] >> 4) | (((qh[lane] >> 6) & 3U) << 4)) - 32;
            }
            destination += 128U;
            q4 += 64U;
            qh += 32U;
        }
        destination = unpacked;
        for (group = 0; group < 16U; ++group) {
            const int scale = scales[group];
            for (lane = 0; lane < 8U; ++lane)
                lanes[lane] += scale * (int16_t)q8[lane] * (int16_t)destination[lane];
            q8 += 8U;
            destination += 8U;
            for (lane = 0; lane < 8U; ++lane)
                lanes[lane] += scale * (int16_t)q8[lane] * (int16_t)destination[lane];
            q8 += 8U;
            destination += 8U;
        }
        {
            const float d = strat01_q4k_q8k_fp16le(raw + 208U) * q8_blocks[block].d;
            for (lane = 0; lane < 8U; ++lane) sums[lane] += d * (float)lanes[lane];
        }
    }
    for (block = 0; block < 8U; ++block) result += sums[block];
    return result;
}

#if !defined(__AVX2__) || !defined(__FMA__)
#error "STRAT-01 Q6_K/Q8_K active reduction requires AVX2 and FMA"
#endif

static __m128i strat01_q6k_scale_shuffle(unsigned index) {
    static const uint8_t shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i *)shuffle + index);
}

static float strat01_q6k_hsum_float_8(__m256 value) {
    __m128 sum = _mm256_extractf128_ps(value, 1);
    sum = _mm_add_ps(sum, _mm256_castps256_ps128(value));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

/* Literal operation order of pinned ggml_vec_dot_q6_K_q8_K's AVX2 arm. */
static float strat01_q6k_q8k_dot_avx2(
        const uint8_t *q6_blocks, const strat01_q8_k_block *q8_blocks,
        unsigned count) {
    const __m256i mask_3 = _mm256_set1_epi8(3);
    const __m256i mask_15 = _mm256_set1_epi8(15);
    __m256 accumulator = _mm256_setzero_ps();
    unsigned block;

    if (!q6_blocks || !q8_blocks || !count || count % STRAT01_QK_K) return NAN;
    for (block = 0; block < count / STRAT01_QK_K; ++block) {
        const uint8_t *raw = q6_blocks + (size_t)block * STRAT01_Q6_K_BLOCK_BYTES;
        const uint8_t *q4 = raw;
        const uint8_t *qh = raw + 128U;
        const int8_t *q8 = q8_blocks[block].qs;
        const float d = q8_blocks[block].d * strat01_q4k_q8k_fp16le(raw + 208U);
        const __m256i q8_sums = _mm256_loadu_si256((const __m256i *)q8_blocks[block].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i *)(raw + 192U));
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8_scale_subtract =
            _mm256_slli_epi32(_mm256_madd_epi16(q8_sums, scales_16), 5);
        __m256i integer_sum = _mm256_setzero_si256();
        unsigned half;
        int scale_index = 0;

        for (half = 0; half < STRAT01_QK_K / 128U; ++half) {
            const __m256i q4_bits_1 = _mm256_loadu_si256((const __m256i *)q4); q4 += 32U;
            const __m256i q4_bits_2 = _mm256_loadu_si256((const __m256i *)q4); q4 += 32U;
            const __m256i q4_bits_high = _mm256_loadu_si256((const __m256i *)qh); qh += 32U;
            const __m256i high_0 = _mm256_slli_epi16(_mm256_and_si256(q4_bits_high, mask_3), 4);
            const __m256i high_1 = _mm256_slli_epi16(
                _mm256_and_si256(q4_bits_high, _mm256_set1_epi8(12)), 2);
            const __m256i high_2 = _mm256_and_si256(q4_bits_high, _mm256_set1_epi8(48));
            const __m256i high_3 = _mm256_srli_epi16(
                _mm256_and_si256(q4_bits_high, _mm256_set1_epi8(-64)), 2);
            const __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q4_bits_1, mask_15), high_0);
            const __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q4_bits_2, mask_15), high_1);
            const __m256i q6_2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4_bits_1, 4), mask_15), high_2);
            const __m256i q6_3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4_bits_2, 4), mask_15), high_3);
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32U;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32U;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32U;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32U;
            __m256i product_0 = _mm256_maddubs_epi16(q6_0, q8_0);
            __m256i product_1 = _mm256_maddubs_epi16(q6_1, q8_1);
            __m256i product_2 = _mm256_maddubs_epi16(q6_2, q8_2);
            __m256i product_3 = _mm256_maddubs_epi16(q6_3, q8_3);
            const __m128i scale_0 = _mm_shuffle_epi8(scales, strat01_q6k_scale_shuffle((unsigned)scale_index + 0U));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, strat01_q6k_scale_shuffle((unsigned)scale_index + 1U));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, strat01_q6k_scale_shuffle((unsigned)scale_index + 2U));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, strat01_q6k_scale_shuffle((unsigned)scale_index + 3U));
            scale_index += 4;
            product_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), product_0);
            product_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), product_1);
            product_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), product_2);
            product_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), product_3);
            integer_sum = _mm256_add_epi32(
                integer_sum, _mm256_add_epi32(product_0, product_1));
            integer_sum = _mm256_add_epi32(
                integer_sum, _mm256_add_epi32(product_2, product_3));
        }
        integer_sum = _mm256_sub_epi32(integer_sum, q8_scale_subtract);
        accumulator = _mm256_fmadd_ps(
            _mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(integer_sum), accumulator);
    }
    return strat01_q6k_hsum_float_8(accumulator);
}

#endif /* STRAT01_Q6K_Q8K_AVX2_H */
