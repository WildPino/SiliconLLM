#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../phase60/strat01_router_weight_norm_reference_generic.h"

#pragma STDC FP_CONTRACT OFF

#define TOKENS 8U
#define SLOTS 4U
#define COUNT (TOKENS * SLOTS)
#define MIN_DENOM 6.103515625e-5f

extern void strat01_router_weight_norm_oracle(
        const float weights[4], float normalized[4]);

static int read_exact(const char *path, float out[COUNT]) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0 || ftell(file) != (long) (COUNT * sizeof(float)) ||
            fseek(file, 0, SEEK_SET) != 0 || fread(out, sizeof(float), COUNT, file) != COUNT) {
        fclose(file); return 0;
    }
    if (fgetc(file) != EOF || ferror(file) || fclose(file) != 0) return 0;
    for (size_t i = 0; i < COUNT; ++i) if (!isfinite(out[i])) return 0;
    return 1;
}

static int write_exact(const char *root, const char *name, const float values[COUNT]) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", root, name) <= 0) return 0;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(values, sizeof(float), COUNT, file) == COUNT;
    if (ok) ok = fflush(file) == 0 && !ferror(file);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static void production_norm(const float weights[4], float out[4]) {
    float sum = 0.0f;
    for (size_t i = 0; i < SLOTS; ++i) sum += weights[i];
    if (sum < MIN_DENOM) sum = MIN_DENOM;
    for (size_t i = 0; i < SLOTS; ++i) out[i] = weights[i] / sum;
}

static void pairwise_norm(const float weights[4], float out[4]) {
    float sum = (weights[0] + weights[1]) + (weights[2] + weights[3]);
    if (sum < MIN_DENOM) sum = MIN_DENOM;
    for (size_t i = 0; i < SLOTS; ++i) out[i] = weights[i] / sum;
}

static void unrounded_norm(const float weights[4], float out[4]) {
    double sum = 0.0;
    for (size_t i = 0; i < SLOTS; ++i) sum += (double) weights[i];
    if (sum < (double) MIN_DENOM) sum = (double) MIN_DENOM;
    for (size_t i = 0; i < SLOTS; ++i) out[i] = (float) ((double) weights[i] / sum);
}

typedef void (*norm_fn)(const float[4], float[4]);

static void apply_all(const float input[COUNT], float output[COUNT], norm_fn fn) {
    for (size_t token = 0; token < TOKENS; ++token) fn(input + token*SLOTS, output + token*SLOTS);
}

static int selftest(void) {
    const float weights[4] = {0.362526476f, 0.094387196f, 0.0848933533f, 0.0819105357f};
    float candidate[4], oracle[4], scalar[4], unrounded[4];
    strat01_router_weight_norm_reference_generic(weights, candidate);
    strat01_router_weight_norm_oracle(weights, oracle);
    production_norm(weights, scalar);
    unrounded_norm(weights, unrounded);
    if (memcmp(candidate, oracle, sizeof(candidate)) != 0 ||
            memcmp(candidate, scalar, sizeof(candidate)) == 0 ||
            memcmp(candidate, unrounded, sizeof(candidate)) == 0) return 1;
    const float tiny[4] = {1.0e-6f, 2.0e-6f, 3.0e-6f, 4.0e-6f};
    float clamped[4];
    strat01_router_weight_norm_reference_generic(tiny, clamped);
    for (size_t i = 0; i < 4U; ++i) if (clamped[i] != tiny[i] / MIN_DENOM) return 1;
    fprintf(stderr, "STRAT-01 router-weight normalization selftest: PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc != 5 || strcmp(argv[1], "--run") != 0) {
        fprintf(stderr, "usage: %s --run WEIGHTS CURRENT_NORM OUTDIR\n", argv[0]); return 2;
    }
    float weights[COUNT], current[COUNT], production[COUNT], candidate[COUNT], oracle[COUNT];
    float pairwise[COUNT], unrounded[COUNT], mutated[COUNT], permuted[COUNT], wrong_origin[COUNT];
    if (!read_exact(argv[2], weights) || !read_exact(argv[3], current)) goto fail;
    apply_all(weights, production, production_norm);
    apply_all(weights, candidate, strat01_router_weight_norm_reference_generic);
    apply_all(weights, oracle, strat01_router_weight_norm_oracle);
    apply_all(weights, pairwise, pairwise_norm);
    apply_all(weights, unrounded, unrounded_norm);
    float changed[COUNT]; memcpy(changed, weights, sizeof(changed)); changed[0] += 1.0f;
    apply_all(changed, mutated, strat01_router_weight_norm_reference_generic);
    memcpy(changed, weights, sizeof(changed));
    for (size_t token = 0; token < TOKENS; ++token) {
        float tmp = changed[token*SLOTS]; changed[token*SLOTS] = changed[token*SLOTS+1U]; changed[token*SLOTS+1U] = tmp;
    }
    apply_all(changed, permuted, strat01_router_weight_norm_reference_generic);
    apply_all(current, wrong_origin, strat01_router_weight_norm_reference_generic);
#define WRITE(name, values) if (!write_exact(argv[4], name, values)) goto fail
    WRITE("production_replay.f32", production);
    WRITE("candidate.f32", candidate);
    WRITE("oracle.f32", oracle);
    WRITE("float_pairwise.f32", pairwise);
    WRITE("unrounded_f64_denominator.f32", unrounded);
    WRITE("mutated.f32", mutated);
    WRITE("permuted.f32", permuted);
    WRITE("wrong_origin.f32", wrong_origin);
#undef WRITE
    return 0;
fail:
    fprintf(stderr, "router-weight normalization probe failed errno=%d\n", errno);
    return 1;
}
