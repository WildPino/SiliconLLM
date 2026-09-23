/*
 * Standalone scalar Q8_K activation quantization and Q4_K x Q8_K dot.
 *
 * Algorithmic semantics are derived from llama.cpp/GGML at
 * 5b335f413e4f73b0809c4fe39af894efbcc6a0d2 (MIT), specifically
 * quantize_row_q8_K_ref and ggml_vec_dot_q4_K_q8_K_generic.  This project
 * implementation has no build or runtime dependency on GGML.
 */
#ifndef STRAT01_Q4K_Q8K_H
#define STRAT01_Q4K_Q8K_H

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define STRAT01_QK_K 256U
#define STRAT01_Q4_K_BLOCK_BYTES 144U

typedef struct {
    float d;
    int8_t qs[STRAT01_QK_K];
    int16_t bsums[STRAT01_QK_K / 16U];
} strat01_q8_k_block;

typedef char strat01_q8_k_block_must_be_292_bytes[
    sizeof(strat01_q8_k_block) == 292U ? 1 : -1];

static int strat01_q8_k_nearest_int(float value) {
    float biased;
    int bits;
    assert(fabsf(value) <= 4194303.0f);
    biased = value + 12582912.0f;
    memcpy(&bits, &biased, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

static int strat01_quantize_q8_k_block(
        const float input[STRAT01_QK_K], strat01_q8_k_block *output) {
    float maximum = 0.0f, absolute_maximum = 0.0f, inverse_scale;
    unsigned j;
    if (!input || !output) return 0;
    for (j = 0; j < STRAT01_QK_K; ++j) {
        float value = input[j], absolute;
        if (!isfinite(value)) return 0;
        absolute = fabsf(value);
        /* Strict greater-than preserves the pinned first-maximum tie rule. */
        if (absolute > absolute_maximum) {
            absolute_maximum = absolute;
            maximum = value;
        }
    }
    if (!absolute_maximum) {
        memset(output, 0, sizeof(*output));
        return 1;
    }
    inverse_scale = -127.0f / maximum;
    for (j = 0; j < STRAT01_QK_K; ++j) {
        int quant = strat01_q8_k_nearest_int(inverse_scale * input[j]);
        output->qs[j] = (int8_t)(quant < 127 ? quant : 127);
    }
    for (j = 0; j < STRAT01_QK_K / 16U; ++j) {
        int sum = 0;
        unsigned k;
        for (k = 0; k < 16U; ++k) sum += output->qs[j * 16U + k];
        output->bsums[j] = (int16_t)sum;
    }
    output->d = 1.0f / inverse_scale;
    return isfinite(output->d);
}

static int strat01_quantize_q8_k_row(
        const float *input, unsigned count, strat01_q8_k_block *output) {
    unsigned block;
    if (!input || !output || !count || count % STRAT01_QK_K) return 0;
    for (block = 0; block < count / STRAT01_QK_K; ++block) {
        if (!strat01_quantize_q8_k_block(input + (size_t)block * STRAT01_QK_K,
                                         output + block)) return 0;
    }
    return 1;
}

/* Read one little-endian binary16 without relying on host half types. */
static float strat01_q4k_q8k_fp16le(const uint8_t *bytes) {
    uint16_t half = (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
    uint32_t sign = (uint32_t)(half >> 15);
    uint32_t exponent = ((uint32_t)half >> 10) & 31U;
    uint32_t fraction = (uint32_t)half & 1023U;
    float value;
    if (exponent == 0U) value = (float)fraction * 0x1p-24f;
    else if (exponent == 31U) value = fraction ? NAN : INFINITY;
    else value = (float)(fraction + 1024U) * ldexpf(1.0f, (int)exponent - 25);
    return sign ? -value : value;
}

static void strat01_q4k_unpack_scales(
        const uint8_t packed[12], uint8_t scales[8], uint8_t minima[8]) {
    unsigned j;
    for (j = 0; j < 8U; ++j) {
        if (j < 4U) {
            scales[j] = packed[j] & 63U;
            minima[j] = packed[j + 4U] & 63U;
        } else {
            scales[j] = (packed[j + 4U] & 15U) |
                        (uint8_t)((packed[j - 4U] >> 6) << 4);
            minima[j] = (packed[j + 4U] >> 4) |
                        (uint8_t)((packed[j] >> 6) << 4);
        }
    }
}

static float strat01_q4k_q8k_dot_generic(
        const uint8_t *q4_blocks, const strat01_q8_k_block *q8_blocks,
        unsigned count) {
    float lane_sums[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};
    float result = 0.0f;
    unsigned block;
    if (!q4_blocks || !q8_blocks || !count || count % STRAT01_QK_K) return NAN;
    for (block = 0; block < count / STRAT01_QK_K; ++block) {
        const uint8_t *raw = q4_blocks + (size_t)block * STRAT01_Q4_K_BLOCK_BYTES;
        const uint8_t *packed = raw + 4U;
        const uint8_t *q4 = raw + 16U;
        const int8_t *q8 = q8_blocks[block].qs;
        int8_t unpacked[STRAT01_QK_K];
        uint8_t scales[8], minima[8];
        int32_t lane_accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int minimum_sum = 0;
        unsigned group, lane, offset, scale_index = 0U;
        float combined_scale, combined_minimum;

        for (group = 0; group < 4U; ++group) {
            for (offset = 0; offset < 32U; ++offset)
                unpacked[group * 64U + offset] = (int8_t)(q4[group * 32U + offset] & 15U);
            for (offset = 0; offset < 32U; ++offset)
                unpacked[group * 64U + 32U + offset] = (int8_t)(q4[group * 32U + offset] >> 4);
        }
        strat01_q4k_unpack_scales(packed, scales, minima);
        for (group = 0; group < 16U; ++group)
            minimum_sum += q8_blocks[block].bsums[group] * minima[group / 2U];

        for (group = 0; group < 8U; ++group) {
            int32_t scale = scales[scale_index++];
            for (offset = 0; offset < 32U; offset += 8U) {
                unsigned base = group * 32U + offset;
                for (lane = 0; lane < 8U; ++lane) {
                    int16_t product = (int16_t)q8[base + lane] *
                                      (int16_t)unpacked[base + lane];
                    lane_accumulators[lane] += scale * product;
                }
            }
        }
        combined_scale = strat01_q4k_q8k_fp16le(raw) * q8_blocks[block].d;
        for (lane = 0; lane < 8U; ++lane)
            lane_sums[lane] += combined_scale * (float)lane_accumulators[lane];
        combined_minimum = strat01_q4k_q8k_fp16le(raw + 2U) * q8_blocks[block].d;
        result -= combined_minimum * (float)minimum_sum;
    }
    for (block = 0; block < 8U; ++block) result += lane_sums[block];
    return result;
}

#if !defined(__clang__)
#error "STRAT-01 reference-generic compile parity requires Clang target attributes"
#endif

/* Exact pinned GGML generic operation structure, isolated from the AVX2/FMA
 * target of engine.c.  The noinline boundary mirrors the separately compiled
 * reference backend and prevents caller-target specialization. */
__attribute__((noinline, target("no-avx,no-avx2,no-fma")))
static float strat01_q4k_q8k_dot_reference_generic(
        const uint8_t *q4_blocks, const strat01_q8_k_block *q8_blocks,
        unsigned count) {
    static const uint32_t mask1 = UINT32_C(0x3f3f3f3f);
    static const uint32_t mask2 = UINT32_C(0x0f0f0f0f);
    static const uint32_t mask3 = UINT32_C(0x03030303);
    uint32_t temporary[4];
    const uint8_t *scales = (const uint8_t *)&temporary[0];
    const uint8_t *minima = (const uint8_t *)&temporary[2];
    int8_t unpacked[STRAT01_QK_K];
    int16_t products[8];
    float lane_sums[8];
    int32_t lane_accumulators[8];
    float result = 0.0f;
    unsigned block;

    if (!q4_blocks || !q8_blocks || !count || count % STRAT01_QK_K) return NAN;
    memset(lane_sums, 0, 8U * sizeof(float));
    for (block = 0; block < count / STRAT01_QK_K; ++block) {
        const uint8_t *raw = q4_blocks + (size_t)block * STRAT01_Q4_K_BLOCK_BYTES;
        const uint8_t *q4 = raw + 16U;
        const int8_t *q8 = q8_blocks[block].qs;
        int8_t *destination = unpacked;
        int scale_index = 0;
        int minimum_sum = 0;
        int group, lane;

        memset(lane_accumulators, 0, 8U * sizeof(int32_t));
        for (group = 0; group < (int)STRAT01_QK_K / 64; ++group) {
            for (lane = 0; lane < 32; ++lane) destination[lane] = (int8_t)(q4[lane] & 0x0f);
            destination += 32;
            for (lane = 0; lane < 32; ++lane) destination[lane] = (int8_t)(q4[lane] >> 4);
            destination += 32;
            q4 += 32;
        }
        memcpy(temporary, raw + 4U, 12U);
        temporary[3] = ((temporary[2] >> 4) & mask2) |
                       (((temporary[1] >> 6) & mask3) << 4);
        {
            const uint32_t upper = temporary[1] & mask1;
            temporary[1] = (temporary[2] & mask2) |
                           (((temporary[0] >> 6) & mask3) << 4);
            temporary[2] = upper;
        }
        temporary[0] &= mask1;
        for (group = 0; group < (int)STRAT01_QK_K / 16; ++group)
            minimum_sum += q8_blocks[block].bsums[group] * minima[group / 2];
        destination = unpacked;
        for (group = 0; group < (int)STRAT01_QK_K / 32; ++group) {
            const int32_t scale = scales[scale_index++];
            int quarter;
            for (quarter = 0; quarter < 4; ++quarter) {
                for (lane = 0; lane < 8; ++lane) products[lane] = q8[lane] * destination[lane];
                for (lane = 0; lane < 8; ++lane) lane_accumulators[lane] += scale * products[lane];
                q8 += 8;
                destination += 8;
            }
        }
        {
            const float scale = strat01_q4k_q8k_fp16le(raw) * q8_blocks[block].d;
            const float minimum = strat01_q4k_q8k_fp16le(raw + 2U) * q8_blocks[block].d;
            for (lane = 0; lane < 8; ++lane) lane_sums[lane] += scale * lane_accumulators[lane];
            result -= minimum * minimum_sum;
        }
    }
    for (block = 0; block < 8U; ++block) result += lane_sums[block];
    return result;
}

#if !defined(__AVX2__) || !defined(__FMA__)
#error "STRAT-01 Q4_K/Q8_K active reduction requires AVX2 and FMA"
#endif

static __m256i strat01_q4k_scale_shuffle(unsigned index) {
    static const uint8_t shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *)shuffle + index);
}

static float strat01_q4k_hsum_float_8(__m256 value) {
    __m128 sum = _mm256_extractf128_ps(value, 1);
    sum = _mm_add_ps(sum, _mm256_castps256_ps128(value));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

/* Standalone transcription of the pinned x86 __AVX2__ branch of
 * ggml_vec_dot_q4_K_q8_K.  Unlike the generic control above, this preserves
 * the active kernel's cross-block FMA accumulators and terminal reductions. */
static float strat01_q4k_q8k_dot_avx2(
        const uint8_t *q4_blocks, const strat01_q8_k_block *q8_blocks,
        unsigned count) {
    static const uint32_t mask1 = UINT32_C(0x3f3f3f3f);
    static const uint32_t mask2 = UINT32_C(0x0f0f0f0f);
    static const uint32_t mask3 = UINT32_C(0x03030303);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    __m256 scale_accumulator = _mm256_setzero_ps();
    __m128 minimum_accumulator = _mm_setzero_ps();
    unsigned block;

    if (!q4_blocks || !q8_blocks || !count || count % STRAT01_QK_K) return NAN;
    for (block = 0; block < count / STRAT01_QK_K; ++block) {
        const uint8_t *raw = q4_blocks + (size_t)block * STRAT01_Q4_K_BLOCK_BYTES;
        const uint8_t *q4 = raw + 16U;
        const int8_t *q8 = q8_blocks[block].qs;
        uint32_t unpacked_scales[4];
        __m256i integer_accumulator = _mm256_setzero_si256();
        __m256i minima_and_scales, q8_sums, scales;
        __m128i q8_sum_pairs, minimum_products, scale_bytes;
        float scale = q8_blocks[block].d * strat01_q4k_q8k_fp16le(raw);
        float minimum = -q8_blocks[block].d * strat01_q4k_q8k_fp16le(raw + 2U);
        unsigned group;

        memcpy(unpacked_scales, raw + 4U, 12U);
        unpacked_scales[3] = ((unpacked_scales[2] >> 4) & mask2) |
                             (((unpacked_scales[1] >> 6) & mask3) << 4);
        {
            uint32_t upper = unpacked_scales[1] & mask1;
            unpacked_scales[1] = (unpacked_scales[2] & mask2) |
                                 (((unpacked_scales[0] >> 6) & mask3) << 4);
            unpacked_scales[2] = upper;
        }
        unpacked_scales[0] &= mask1;

        minima_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)unpacked_scales[3], (int)unpacked_scales[2],
                          (int)unpacked_scales[1], (int)unpacked_scales[0]));
        q8_sums = _mm256_loadu_si256((const __m256i *)q8_blocks[block].bsums);
        q8_sum_pairs = _mm_hadd_epi16(_mm256_extracti128_si256(q8_sums, 0),
                                     _mm256_extracti128_si256(q8_sums, 1));
        minimum_products = _mm_madd_epi16(
            _mm256_extracti128_si256(minima_and_scales, 1), q8_sum_pairs);
        minimum_accumulator = _mm_fmadd_ps(
            _mm_set1_ps(minimum), _mm_cvtepi32_ps(minimum_products),
            minimum_accumulator);

        scale_bytes = _mm256_extracti128_si256(minima_and_scales, 0);
        scales = _mm256_insertf128_si256(_mm256_castsi128_si256(scale_bytes),
                                         scale_bytes, 1);
        for (group = 0; group < STRAT01_QK_K / 64U; ++group) {
            __m256i scale_low = _mm256_shuffle_epi8(
                scales, strat01_q4k_scale_shuffle(2U * group));
            __m256i scale_high = _mm256_shuffle_epi8(
                scales, strat01_q4k_scale_shuffle(2U * group + 1U));
            __m256i q4_bits = _mm256_loadu_si256((const __m256i *)q4);
            __m256i q4_low = _mm256_and_si256(q4_bits, nibble_mask);
            __m256i q4_high = _mm256_and_si256(
                _mm256_srli_epi16(q4_bits, 4), nibble_mask);
            __m256i q8_low, q8_high, low_products, high_products;
            q4 += 32U;
            q8_low = _mm256_loadu_si256((const __m256i *)q8); q8 += 32U;
            low_products = _mm256_maddubs_epi16(q4_low, q8_low);
            low_products = _mm256_madd_epi16(scale_low, low_products);
            q8_high = _mm256_loadu_si256((const __m256i *)q8); q8 += 32U;
            high_products = _mm256_maddubs_epi16(q4_high, q8_high);
            high_products = _mm256_madd_epi16(scale_high, high_products);
            integer_accumulator = _mm256_add_epi32(
                integer_accumulator,
                _mm256_add_epi32(low_products, high_products));
        }
        scale_accumulator = _mm256_fmadd_ps(
            _mm256_set1_ps(scale), _mm256_cvtepi32_ps(integer_accumulator),
            scale_accumulator);
    }

    minimum_accumulator = _mm_add_ps(
        minimum_accumulator,
        _mm_movehl_ps(minimum_accumulator, minimum_accumulator));
    minimum_accumulator = _mm_add_ss(
        minimum_accumulator,
        _mm_movehdup_ps(minimum_accumulator));
    return strat01_q4k_hsum_float_8(scale_accumulator) +
           _mm_cvtss_f32(minimum_accumulator);
}

static float strat01_q4k_q8k_dot(
        const uint8_t *q4_blocks, const strat01_q8_k_block *q8_blocks,
        unsigned count) {
#if defined(STRAT01_Q4K_Q8K_DIAGNOSTIC_GENERIC_REDUCTION)
    return strat01_q4k_q8k_dot_generic(q4_blocks, q8_blocks, count);
#elif defined(STRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2)
    return strat01_q4k_q8k_dot_avx2(q4_blocks, q8_blocks, count);
#else
    return strat01_q4k_q8k_dot_reference_generic(q4_blocks, q8_blocks, count);
#endif
}

static int strat01_q4k_q8k_selftest(void) {
    float input[STRAT01_QK_K] = {0.0f};
    strat01_q8_k_block block;
    uint8_t q4[STRAT01_Q4_K_BLOCK_BYTES] = {0};
    int bad = 0;
    unsigned i;
    memset(&block, 0x7f, sizeof(block));
    if (!strat01_quantize_q8_k_block(input, &block)) ++bad;
    for (i = 0; i < sizeof(block); ++i) if (((const uint8_t *)&block)[i] != 0U) { ++bad; break; }
    input[0] = 1.0f; input[1] = -1.0f;
    if (!strat01_quantize_q8_k_block(input, &block) ||
        block.d != -1.0f / 127.0f || block.qs[0] != -127 || block.qs[1] != 127 ||
        block.bsums[0] != 0) ++bad;
    input[0] = -1.0f; input[1] = 1.0f;
    if (!strat01_quantize_q8_k_block(input, &block) ||
        block.d != 1.0f / 127.0f || block.qs[0] != -127 || block.qs[1] != 127) ++bad;
    input[7] = NAN;
    if (strat01_quantize_q8_k_block(input, &block)) ++bad;
    input[7] = 0.0f;
    q4[0] = 0x00; q4[1] = 0x3c; /* d = 1 */
    q4[4] = 1U;                 /* first scale = 1 */
    q4[16] = 1U;                /* first low nibble = 1 */
    if (!strat01_quantize_q8_k_block(input, &block) ||
        !isfinite(strat01_q4k_q8k_dot(q4, &block, STRAT01_QK_K))) ++bad;
    return bad ? 1 : 0;
}

#endif /* STRAT01_Q4K_Q8K_H */
