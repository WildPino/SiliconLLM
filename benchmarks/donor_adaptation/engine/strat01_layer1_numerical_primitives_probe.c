#include <errno.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../phase60/strat01_f32_dot_reference_generic.h"
#include "../../phase60/strat01_swiglu_sse2.h"

#define ROUTER_IN 1536U
#define ROUTER_ROWS 64U
#define TOKENS 8U
#define ROUTER_COUNT (TOKENS * ROUTER_ROWS)
#define ROUTER_WEIGHT_COUNT (ROUTER_IN * ROUTER_ROWS)
#define ROUTER_FILE_OFFSET 509582464LL
#define ROUTED_COUNT (TOKENS * 4U * 1280U)
#define SHARED_COUNT (TOKENS * 1280U)

extern float strat01_layer1_f32_router_oracle(
        const float *x, const float *y, unsigned count);

static int read_exact(const char *path, float *out, size_t count) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0 || ftell(file) != (long) (count * sizeof(float)) ||
            fseek(file, 0, SEEK_SET) != 0 || fread(out, sizeof(float), count, file) != count) {
        fclose(file);
        return 0;
    }
    if (fgetc(file) != EOF || ferror(file) || fclose(file) != 0) return 0;
    for (size_t i = 0; i < count; ++i) if (!isfinite(out[i])) return 0;
    return 1;
}

static int read_router_weights(const char *path, float *out) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, (long) ROUTER_FILE_OFFSET, SEEK_SET) != 0 ||
            fread(out, sizeof(float), ROUTER_WEIGHT_COUNT, file) != ROUTER_WEIGHT_COUNT) {
        if (file) fclose(file);
        return 0;
    }
    if (ferror(file) || fclose(file) != 0) return 0;
    for (size_t i = 0; i < ROUTER_WEIGHT_COUNT; ++i) if (!isfinite(out[i])) return 0;
    return 1;
}

static int write_exact(const char *root, const char *name, const float *values, size_t count) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", root, name) <= 0) return 0;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(values, sizeof(float), count, file) == count;
    if (ok) ok = fflush(file) == 0 && !ferror(file);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static float dot_float(const float *x, const float *y, unsigned count) {
    float sum = 0.0f;
    for (unsigned i = 0; i < count; ++i) sum += x[i] * y[i];
    return sum;
}

static float dot_promoted_product(const float *x, const float *y, unsigned count) {
    double sum = 0.0;
    for (unsigned i = 0; i < count; ++i) sum += (double) x[i] * (double) y[i];
    return (float) sum;
}

static void router_all(
        const float *input, const float *weights, float *float_out,
        float *candidate, float *oracle, float *promoted) {
    for (unsigned token = 0; token < TOKENS; ++token) {
        const float *x = input + (size_t) token * ROUTER_IN;
        for (unsigned row = 0; row < ROUTER_ROWS; ++row) {
            const float *w = weights + (size_t) row * ROUTER_IN;
            const size_t dst = (size_t) token * ROUTER_ROWS + row;
            float_out[dst] = dot_float(w, x, ROUTER_IN);
            candidate[dst] = strat01_f32_dot_reference_generic(w, x, ROUTER_IN);
            oracle[dst] = strat01_layer1_f32_router_oracle(w, x, ROUTER_IN);
            promoted[dst] = dot_promoted_product(w, x, ROUTER_IN);
        }
    }
}

static void router_candidate_all(
        const float *input, const float *weights, float *candidate) {
    for (unsigned token = 0; token < TOKENS; ++token) {
        const float *x = input + (size_t) token * ROUTER_IN;
        for (unsigned row = 0; row < ROUTER_ROWS; ++row) {
            const float *w = weights + (size_t) row * ROUTER_IN;
            candidate[(size_t) token * ROUTER_ROWS + row] =
                strat01_f32_dot_reference_generic(w, x, ROUTER_IN);
        }
    }
}

static void swiglu_scalar(const float *gate, const float *up, float *out, size_t count) {
    for (size_t i = 0; i < count; ++i) out[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}

static int selftest(void) {
    float x[ROUTER_IN], y[ROUTER_IN];
    for (unsigned i = 0; i < ROUTER_IN; ++i) {
        x[i] = (float) ((int) (i % 29U) - 14) * 0.03125f + (float) (i & 1U) * 1.0e-4f;
        y[i] = (float) ((int) (i % 31U) - 15) * 0.015625f - (float) (i % 3U) * 7.0e-5f;
    }
    const float candidate = strat01_f32_dot_reference_generic(x, y, ROUTER_IN);
    const float oracle = strat01_layer1_f32_router_oracle(x, y, ROUTER_IN);
    const float scalar = dot_float(x, y, ROUTER_IN);
    const float promoted = dot_promoted_product(x, y, ROUTER_IN);
    float gate[8] = {-7.1f, -2.3f, -0.7f, 0.2f, 0.9f, 2.1f, 4.4f, 8.2f};
    float up[8] = {0.3f, -0.8f, 1.7f, 2.2f, -1.1f, 0.6f, -2.0f, 0.4f};
    float sse[8], libm[8]; char error[256] = {0};
    swiglu_scalar(gate, up, libm, 8U);
    if (!strat01_sse2_swiglu_compute(gate, up, sse, 8U, error)) return 1;
    if (memcmp(&candidate, &oracle, sizeof(float)) != 0 ||
            memcmp(&candidate, &scalar, sizeof(float)) == 0 ||
            memcmp(&candidate, &promoted, sizeof(float)) == 0 ||
            memcmp(sse, libm, sizeof(sse)) == 0) return 1;
    fprintf(stderr, "STRAT-01 layer-1 numerical-primitives selftest: PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc != 9 || strcmp(argv[1], "--run") != 0) {
        fprintf(stderr, "usage: %s --run MODEL NORM ROUTED_GATE ROUTED_UP SHARED_GATE SHARED_UP OUTDIR\n", argv[0]);
        return 2;
    }
    float *norm = malloc(ROUTER_IN * TOKENS * sizeof(float));
    float *weights = malloc(ROUTER_WEIGHT_COUNT * sizeof(float));
    float *router_float = malloc(ROUTER_COUNT * sizeof(float));
    float *router_candidate = malloc(ROUTER_COUNT * sizeof(float));
    float *router_oracle = malloc(ROUTER_COUNT * sizeof(float));
    float *router_promoted = malloc(ROUTER_COUNT * sizeof(float));
    float *router_input_mutated = malloc(ROUTER_COUNT * sizeof(float));
    float *router_weight_mutated = malloc(ROUTER_COUNT * sizeof(float));
    float *routed_gate = malloc(ROUTED_COUNT * sizeof(float));
    float *routed_up = malloc(ROUTED_COUNT * sizeof(float));
    float *routed_scalar = malloc(ROUTED_COUNT * sizeof(float));
    float *routed_sse2 = malloc(ROUTED_COUNT * sizeof(float));
    float *routed_gate_mutated = malloc(ROUTED_COUNT * sizeof(float));
    float *routed_swapped = malloc(ROUTED_COUNT * sizeof(float));
    float *shared_gate = malloc(SHARED_COUNT * sizeof(float));
    float *shared_up = malloc(SHARED_COUNT * sizeof(float));
    float *shared_scalar = malloc(SHARED_COUNT * sizeof(float));
    float *shared_sse2 = malloc(SHARED_COUNT * sizeof(float));
    char error[256] = {0}; int ok = 0;
    if (!norm || !weights || !router_float || !router_candidate || !router_oracle ||
            !router_promoted || !router_input_mutated || !router_weight_mutated ||
            !routed_gate || !routed_up || !routed_scalar || !routed_sse2 ||
            !routed_gate_mutated || !routed_swapped || !shared_gate || !shared_up ||
            !shared_scalar || !shared_sse2) goto done;
    if (!read_exact(argv[3], norm, ROUTER_IN * TOKENS) ||
            !read_router_weights(argv[2], weights) ||
            !read_exact(argv[4], routed_gate, ROUTED_COUNT) ||
            !read_exact(argv[5], routed_up, ROUTED_COUNT) ||
            !read_exact(argv[6], shared_gate, SHARED_COUNT) ||
            !read_exact(argv[7], shared_up, SHARED_COUNT)) goto done;
    router_all(norm, weights, router_float, router_candidate, router_oracle, router_promoted);
    const float norm0 = norm[0];
    norm[0] = nextafterf(norm[0], INFINITY);
    router_candidate_all(norm, weights, router_input_mutated);
    norm[0] = norm0;
    const float weight0 = weights[0];
    weights[0] = nextafterf(weights[0], INFINITY);
    router_candidate_all(norm, weights, router_weight_mutated);
    weights[0] = weight0;
    swiglu_scalar(routed_gate, routed_up, routed_scalar, ROUTED_COUNT);
    if (!strat01_sse2_swiglu_compute(routed_gate, routed_up, routed_sse2, ROUTED_COUNT, error)) goto done;
    const float routed_gate0 = routed_gate[0];
    routed_gate[0] = nextafterf(routed_gate[0], INFINITY);
    if (!strat01_sse2_swiglu_compute(routed_gate, routed_up, routed_gate_mutated, ROUTED_COUNT, error)) goto done;
    routed_gate[0] = routed_gate0;
    if (!strat01_sse2_swiglu_compute(routed_up, routed_gate, routed_swapped, ROUTED_COUNT, error)) goto done;
    swiglu_scalar(shared_gate, shared_up, shared_scalar, SHARED_COUNT);
    if (!strat01_sse2_swiglu_compute(shared_gate, shared_up, shared_sse2, SHARED_COUNT, error)) goto done;
#define WRITE(name, ptr, count) if (!write_exact(argv[8], name, ptr, count)) goto done
    WRITE("router_float_replay.f32", router_float, ROUTER_COUNT);
    WRITE("router_candidate.f32", router_candidate, ROUTER_COUNT);
    WRITE("router_oracle.f32", router_oracle, ROUTER_COUNT);
    WRITE("router_promoted_product.f32", router_promoted, ROUTER_COUNT);
    WRITE("router_input_mutated.f32", router_input_mutated, ROUTER_COUNT);
    WRITE("router_weight_mutated.f32", router_weight_mutated, ROUTER_COUNT);
    WRITE("routed_scalar_replay.f32", routed_scalar, ROUTED_COUNT);
    WRITE("routed_sse2.f32", routed_sse2, ROUTED_COUNT);
    WRITE("routed_gate_mutated.f32", routed_gate_mutated, ROUTED_COUNT);
    WRITE("routed_swapped.f32", routed_swapped, ROUTED_COUNT);
    WRITE("shared_scalar_replay.f32", shared_scalar, SHARED_COUNT);
    WRITE("shared_sse2.f32", shared_sse2, SHARED_COUNT);
#undef WRITE
    ok = 1;
done:
    free(norm); free(weights); free(router_float); free(router_candidate); free(router_oracle);
    free(router_promoted); free(router_input_mutated); free(router_weight_mutated);
    free(routed_gate); free(routed_up); free(routed_scalar); free(routed_sse2);
    free(routed_gate_mutated); free(routed_swapped); free(shared_gate); free(shared_up);
    free(shared_scalar); free(shared_sse2);
    if (!ok) fprintf(stderr, "layer-1 numerical-primitives probe failed: %s errno=%d\n", error, errno);
    return ok ? 0 : 1;
}
