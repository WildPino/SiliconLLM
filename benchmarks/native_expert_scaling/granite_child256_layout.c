/* Exact Q4_K parent512 -> two child256 expert layout control.
 * Synthetic valid Q4 blocks only; no source weights, model, cost or rate claim.
 * The active AVX2 dispatch is deliberate: default strat01 wrapper is scalar.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2 1
#include "../phase60/strat01_q4k_q8k.h"

#define D 1024U
#define PARENT 512U
#define CHILD 256U
#define GU_ROWS (2U * PARENT)
#define CHILD_GU_ROWS (2U * CHILD)
#define BLOCK 144U
#define GU_BLOCKS (D / CHILD)
#define DOWN_BLOCKS (PARENT / CHILD)

static void need(int ok, const char *why) {
    if (!ok) { fprintf(stderr, "layout_failure:%s\n", why); exit(2); }
}
static uint64_t mix(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static uint64_t checksum(const uint8_t *p, size_t n) {
    uint64_t v = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; ++i) v = (v ^ p[i]) * UINT64_C(1099511628211);
    return v;
}
static void seeded_block(uint8_t out[BLOCK], uint64_t seed) {
    /* Positive finite FP16 powers of two; exercise all packed scale bits. */
    memset(out, 0, BLOCK);
    out[0] = 0x00; out[1] = (uint8_t)(0x10 + 4 * (mix(seed) % 5));
    out[2] = 0x00; out[3] = (uint8_t)(0x10 + 4 * (mix(seed + 1) % 5));
    for (unsigned j = 4; j < 16; ++j) out[j] = (uint8_t)mix(seed + j);
    for (unsigned j = 16; j < BLOCK; ++j) out[j] = (uint8_t)mix(seed + j);
    need(isfinite(strat01_q4k_q8k_fp16le(out)) &&
         strat01_q4k_q8k_fp16le(out) > 0.f &&
         strat01_q4k_q8k_fp16le(out + 2) > 0.f, "valid_finite_q4_scales");
}
static double decoded_dot(const uint8_t *row, const strat01_q8_k_block *q8, unsigned blocks) {
    double sum = 0.;
    for (unsigned b = 0; b < blocks; ++b) {
        const uint8_t *raw = row + (size_t)b * BLOCK;
        uint8_t scales[8], minima[8];
        /* Decode the twelve scale bytes here instead of sharing the header's
           unpack helper with the accelerated path under test. */
        const uint8_t *packed_scales = raw + 4;
        for (unsigned k = 0; k < 8; ++k) {
            if (k < 4) {
                scales[k] = packed_scales[k] & 63U;
                minima[k] = packed_scales[k + 4] & 63U;
            } else {
                scales[k] = (packed_scales[k + 4] & 15U) |
                            (uint8_t)((packed_scales[k - 4] >> 6) << 4);
                minima[k] = (packed_scales[k + 4] >> 4) |
                            (uint8_t)((packed_scales[k] >> 6) << 4);
            }
        }
        for (unsigned j = 0; j < CHILD; ++j) {
            unsigned chunk = j / 64U, offset = j % 64U;
            uint8_t packed = raw[16U + chunk * 32U + (offset % 32U)];
            unsigned code = offset < 32U ? packed & 15U : packed >> 4U;
            unsigned group = j / 32U;
            double weight = (double)strat01_q4k_q8k_fp16le(raw) * scales[group] * code
                          - (double)strat01_q4k_q8k_fp16le(raw + 2) * minima[group];
            sum += weight * (double)q8[b].d * q8[b].qs[j];
        }
    }
    return sum;
}
static void compare_vector(const double *ref, const float *candidate, unsigned n,
                           double *relative, double *worst) {
    double e2 = 0., n2 = 0., max = 0.;
    for (unsigned j = 0; j < n; ++j) {
        need(isfinite(ref[j]) && isfinite(candidate[j]), "finite_full_output");
        double e = (double)candidate[j] - ref[j];
        e2 += e * e; n2 += ref[j] * ref[j];
        if (fabs(e) > max) max = fabs(e);
    }
    need(n2 > 0., "nonzero_reference_norm");
    *relative = sqrt(e2 / n2); *worst = max;
    need(*relative <= 1e-5, "full_vector_relative_l2_1e_minus_5");
}
/* Reusable block-preserving partition primitives. Source and destination must
   be distinct buffers of the full parent tensor size. */
static void partition_gate_up(const uint8_t *parent, uint8_t *children) {
    for (unsigned child = 0; child < 2; ++child)
        for (unsigned j = 0; j < CHILD; ++j) {
            unsigned old_gate = child * CHILD + j;
            unsigned old_up = PARENT + child * CHILD + j;
            unsigned new_gate = child * CHILD_GU_ROWS + j;
            unsigned new_up = child * CHILD_GU_ROWS + CHILD + j;
            memcpy(children + (size_t)new_gate * GU_BLOCKS * BLOCK,
                   parent + (size_t)old_gate * GU_BLOCKS * BLOCK, GU_BLOCKS * BLOCK);
            memcpy(children + (size_t)new_up * GU_BLOCKS * BLOCK,
                   parent + (size_t)old_up * GU_BLOCKS * BLOCK, GU_BLOCKS * BLOCK);
        }
}
static void partition_down(const uint8_t *parent, uint8_t *children) {
    for (unsigned child = 0; child < 2; ++child)
        for (unsigned row = 0; row < D; ++row)
            memcpy(children + ((size_t)child * D + row) * BLOCK,
                   parent + ((size_t)row * DOWN_BLOCKS + child) * BLOCK, BLOCK);
}
int main(void) {
    need(sizeof(strat01_q8_k_block) == 292 && !strat01_q4k_q8k_selftest(), "header_controls");
    const size_t gu_bytes = (size_t)GU_ROWS * GU_BLOCKS * BLOCK;
    const size_t down_bytes = (size_t)D * DOWN_BLOCKS * BLOCK;
    uint8_t *gu = malloc(gu_bytes), *down = malloc(down_bytes);
    uint8_t *gu_child = malloc(gu_bytes), *down_child = malloc(down_bytes);
    need(gu && down && gu_child && down_child, "allocation");
    for (unsigned row = 0; row < GU_ROWS; ++row)
        for (unsigned b = 0; b < GU_BLOCKS; ++b)
            seeded_block(gu + ((size_t)row * GU_BLOCKS + b) * BLOCK,
                         UINT64_C(0x9a23) + (size_t)row * GU_BLOCKS + b);
    for (unsigned row = 0; row < D; ++row)
        for (unsigned b = 0; b < DOWN_BLOCKS; ++b)
            seeded_block(down + ((size_t)row * DOWN_BLOCKS + b) * BLOCK,
                         UINT64_C(0x3d11) + (size_t)row * DOWN_BLOCKS + b);

    /* Child-major contiguous: each child has 256 gate + 256 up rows, and
       1024 down rows with exactly one original Q4_K 256-input block. */
    partition_gate_up(gu, gu_child);
    partition_down(down, down_child);
    unsigned blocks_checked = 0;
    for (unsigned child = 0; child < 2; ++child) {
        for (unsigned j = 0; j < CHILD; ++j) {
            unsigned old_rows[2] = {child * CHILD + j, PARENT + child * CHILD + j};
            unsigned new_rows[2] = {child * CHILD_GU_ROWS + j,
                                    child * CHILD_GU_ROWS + CHILD + j};
            for (unsigned branch = 0; branch < 2; ++branch)
                for (unsigned b = 0; b < GU_BLOCKS; ++b) {
                    need(!memcmp(gu_child + ((size_t)new_rows[branch] * GU_BLOCKS + b) * BLOCK,
                                 gu + ((size_t)old_rows[branch] * GU_BLOCKS + b) * BLOCK, BLOCK),
                         "gate_up_block_copy"); ++blocks_checked;
                }
        }
        for (unsigned row = 0; row < D; ++row) {
            need(!memcmp(down_child + ((size_t)child * D + row) * BLOCK,
                         down + ((size_t)row * DOWN_BLOCKS + child) * BLOCK, BLOCK),
                 "down_block_copy"); ++blocks_checked;
        }
    }
    need(blocks_checked == GU_ROWS * GU_BLOCKS + D * DOWN_BLOCKS, "complete_block_count");

    float x[D], z[PARENT];
    strat01_q8_k_block qx[GU_BLOCKS], qz[DOWN_BLOCKS], qchild[1];
    for (unsigned j = 0; j < D; ++j) x[j] = (float)((int)(mix(j + 777) & 511) - 256) / 256.f;
    for (unsigned j = 0; j < PARENT; ++j) z[j] = (float)((int)(mix(j + 333) & 511) - 256) / 256.f;
    need(strat01_quantize_q8_k_row(x, D, qx) &&
         strat01_quantize_q8_k_row(z, PARENT, qz), "parent_q8_quantization");
    for (unsigned child = 0; child < 2; ++child) {
        need(strat01_quantize_q8_k_row(z + child * CHILD, CHILD, qchild), "child_q8_quantization");
        need(!memcmp(qchild, qz + child, sizeof(qchild)), "q8_segment_exact_equality");
    }

    double gu_ref[GU_ROWS], down_ref[D];
    float gu_avx[GU_ROWS], gu_child_avx[GU_ROWS], down_avx[D], down_child_sum[D];
    for (unsigned row = 0; row < GU_ROWS; ++row) {
        unsigned child = (row % PARENT) / CHILD;
        unsigned j = row % CHILD;
        unsigned new_row = child * CHILD_GU_ROWS + (row >= PARENT ? CHILD : 0) + j;
        const uint8_t *parent = gu + (size_t)row * GU_BLOCKS * BLOCK;
        const uint8_t *copied = gu_child + (size_t)new_row * GU_BLOCKS * BLOCK;
        gu_ref[row] = decoded_dot(parent, qx, GU_BLOCKS);
        gu_avx[row] = strat01_q4k_q8k_dot(parent, qx, D);
        gu_child_avx[row] = strat01_q4k_q8k_dot(copied, qx, D);
        need(!memcmp(gu_avx + row, gu_child_avx + row, sizeof(float)), "gate_up_avx_copy_bit_equal");
    }
    for (unsigned row = 0; row < D; ++row) {
        const uint8_t *parent = down + (size_t)row * DOWN_BLOCKS * BLOCK;
        down_ref[row] = decoded_dot(parent, qz, DOWN_BLOCKS);
        down_avx[row] = strat01_q4k_q8k_dot(parent, qz, PARENT);
        float parts[2];
        for (unsigned child = 0; child < 2; ++child)
            parts[child] = strat01_q4k_q8k_dot(
                down_child + ((size_t)child * D + row) * BLOCK, qz + child, CHILD);
        down_child_sum[row] = parts[0] + parts[1];
    }
    double gu_err, gu_max, down_err, down_max, child_err, child_max;
    compare_vector(gu_ref, gu_avx, GU_ROWS, &gu_err, &gu_max);
    compare_vector(down_ref, down_avx, D, &down_err, &down_max);
    compare_vector(down_ref, down_child_sum, D, &child_err, &child_max);
    need(gu_err <= 1e-5 && down_err <= 1e-5 && child_err <= 1e-5, "numerical_gate");
    /* Complete synthetic SwiGLU, including inherited parent contribution mass.
       The intermediate Q8 policy is identical in parent and child controls. */
    for (unsigned j = 0; j < PARENT; ++j) {
        float gate = gu_avx[j];
        z[j] = (gate / (1.f + expf(-gate))) * gu_avx[PARENT + j];
        need(isfinite(z[j]), "finite_swiglu");
    }
    need(strat01_quantize_q8_k_row(z, PARENT, qz), "swiglu_parent_q8");
    for (unsigned child = 0; child < 2; ++child) {
        float child_z[CHILD];
        for (unsigned j = 0; j < CHILD; ++j) {
            float gate = gu_child_avx[child * CHILD + j];
            child_z[j] = (gate / (1.f + expf(-gate))) *
                         gu_child_avx[PARENT + child * CHILD + j];
        }
        need(strat01_quantize_q8_k_row(child_z, CHILD, qchild), "swiglu_child_q8");
        need(!memcmp(qchild, qz + child, sizeof(qchild)), "swiglu_q8_segment_equal");
    }
    const float parent_mass = 0.375f;
    for (unsigned row = 0; row < D; ++row) {
        down_ref[row] = parent_mass * decoded_dot(
            down + (size_t)row * DOWN_BLOCKS * BLOCK, qz, DOWN_BLOCKS);
        float parts[2];
        for (unsigned child = 0; child < 2; ++child)
            parts[child] = parent_mass * strat01_q4k_q8k_dot(
                down_child + ((size_t)child * D + row) * BLOCK, qz + child, CHILD);
        down_child_sum[row] = parts[0] + parts[1];
    }
    double swiglu_err, swiglu_max;
    compare_vector(down_ref, down_child_sum, D, &swiglu_err, &swiglu_max);
    printf("{\"status\":\"layout_pass\",\"synthetic_only\":true,"
           "\"q4_block_bytes\":144,\"q8_block_bytes\":292,\"copied_blocks\":%u,"
           "\"gate_up_parent_bytes\":%llu,\"down_parent_bytes\":%llu,"
           "\"gate_up_child_bytes\":%llu,\"down_child_bytes\":%llu,"
           "\"gate_up_parent_fnv64\":\"%016llx\",\"down_parent_fnv64\":\"%016llx\","
           "\"gate_up_relative_l2\":%.12g,\"down_relative_l2\":%.12g,"
           "\"down_child_reconstruction_relative_l2\":%.12g,"
           "\"gate_up_max_abs\":%.12g,\"down_max_abs\":%.12g,\"child_max_abs\":%.12g,"
           "\"swiglu_parent_mass\":0.375,\"swiglu_relative_l2\":%.12g,\"swiglu_max_abs\":%.12g}\n",
           blocks_checked, (unsigned long long)gu_bytes, (unsigned long long)down_bytes,
           (unsigned long long)gu_bytes, (unsigned long long)down_bytes,
           (unsigned long long)checksum(gu, gu_bytes), (unsigned long long)checksum(down, down_bytes),
           gu_err, down_err, child_err, gu_max, down_max, child_max, swiglu_err, swiglu_max);
    free(gu); free(down); free(gu_child); free(down_child);
    return 0;
}
