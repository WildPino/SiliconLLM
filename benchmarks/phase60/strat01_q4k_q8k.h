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

static float strat01_q4k_q8k_dot(
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
