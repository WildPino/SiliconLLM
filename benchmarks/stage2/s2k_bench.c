// Stage 2 test 5, amendment 2: the weight rate of ik_llama.cpp's kernels at any list of matrices, one token at a time.
// From stage 1's s1k_bench.c.  Every layer holds its own copy of the list, and the layer count is raised until the weights
// exceed --footprint-mb, so they stream from RAM whatever the shape.  One mul_mat per matrix, the input shared within a size.
//
// --mats 4096x4096:iq2_bn,1024x4096:q4_k      out x in : type
// --four     IQ2_BN rows written by hand with all four codes; otherwise ternary codes, through the same writer
// Rate = weights per token / compute time.  ms_per_32 = the time of 32 copies of the list: flash's layer count.
// Check: 64 outputs of the first matrix against a double dot product on its dequantized rows, and the next 64 rows, far off.
//
// Build: gcc -O2 -Wall -fopenmp s2k_bench.c -I<ik>/ggml/include <ik>/build-native/ggml/src/libggml.a -lstdc++ -static -o s2k_bench.exe
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"

#define MAXM 8
static uint64_t st = 20261105u;
static uint64_t xs(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static float uni(void) { return (float)((xs() >> 11) * (1.0 / 9007199254740992.0)); }
static float gauss(void) { return sqrtf(-2.0f * logf(uni() + 1e-12f)) * cosf(6.2831853f * uni()); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp_d(const void * a, const void * b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static enum ggml_type type_of(const char * name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((enum ggml_type)t);
        if (n && !strcmp(n, name)) return (enum ggml_type)t;
    }
    fprintf(stderr, "unknown type %s\n", name);
    exit(2);
}

static void write_bn(uint8_t * dst, int64_t n, int four) {
    float s = 0.01f + 0.02f * uni();
    memcpy(dst, &s, 4);
    uint8_t * q = dst + 4;
    for (int64_t ib = 0; ib < n / 64; ib++)
        for (int j = 0; j < 16; j++) {
            uint8_t b = 0;
            for (int l = 0; l < 4; l++) b |= (uint8_t)((four ? xs() % 4 : xs() % 3) << (2 * l));
            q[ib * 16 + j] = b;
        }
}

static void fill(struct ggml_tensor * w, int four) {
    const int64_t in = w->ne[0], out = w->ne[1];
    size_t row = ggml_row_size(w->type, in);
    if (!strcmp(ggml_type_name(w->type), "iq2_bn")) {
        for (int64_t r = 0; r < out; r++) write_bn((uint8_t *)w->data + r * row, in, four);
        return;
    }
    const int64_t chunk = 64;
    float * buf = malloc(sizeof(float) * in * chunk);
    for (int64_t r0 = 0; r0 < out; r0 += chunk) {
        int64_t nr = out - r0 < chunk ? out - r0 : chunk;
        for (int64_t i = 0; i < in * nr; i++) buf[i] = 0.02f * gauss();
        ggml_quantize_chunk(w->type, buf, (char *)w->data + r0 * row, 0, nr, in, NULL, NULL);
    }
    free(buf);
}

static int row_float(const struct ggml_tensor * w, int64_t r, float * dst) {
    ggml_type_traits_t tr = ggml_internal_get_type_traits(w->type);
    const char * src = (const char *)w->data + r * w->nb[1];
    if (!strstr(ggml_type_name(w->type), "_bn")) { tr.to_float(src, dst, w->ne[0]); return 1; }   // the others read their own row meta
    float scale = 1.0f;
    if (tr.row_meta_size == 4) scale = *(const float *)src;
    else if (tr.row_meta_size == 2) scale = ggml_fp16_to_fp32(*(const ggml_fp16_t *)src);
    else if (tr.row_meta_size != 0) return 0;
    tr.to_float(src + tr.row_meta_size, dst, w->ne[0]);
    for (int64_t i = 0; i < w->ne[0]; i++) dst[i] *= scale;
    return 1;
}

static double check(const struct ggml_tensor * w, int64_t row0, const float * x, const float * out) {
    float * row = malloc(sizeof(float) * w->ne[0]);
    double num = 0, den = 0;
    for (int64_t r = 0; r < 64; r++) {
        if (!row_float(w, row0 + r, row)) { free(row); return -1; }
        double s = 0;
        for (int64_t c = 0; c < w->ne[0]; c++) s += (double)row[c] * x[c];
        num += (s - out[r]) * (s - out[r]);
        den += s * s;
    }
    free(row);
    return sqrt(num / (den + 1e-30));
}

int main(int argc, char ** argv) {
    const char * mats = "4096x4096:iq2_bn";
    int threads = 6, T = 32, R = 5, four = 0, foot_mb = 512;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mats")) mats = argv[++i];
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens")) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")) R = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--footprint-mb")) foot_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--four")) four = 1;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    int64_t mo[MAXM], mi[MAXM];
    enum ggml_type mt[MAXM];
    int NM = 0;
    char * spec = strdup(mats);
    for (char * p = strtok(spec, ","); p; p = strtok(NULL, ",")) {
        char tn[32];
        long long o, n;
        if (sscanf(p, "%lldx%lld:%31s", &o, &n, tn) != 3 || NM == MAXM) { fprintf(stderr, "bad --mats\n"); return 2; }
        mo[NM] = o; mi[NM] = n; mt[NM] = type_of(tn); NM++;
    }
    double per_layer = 0, w_layer = 0;
    for (int m = 0; m < NM; m++) { per_layer += (double)ggml_row_size(mt[m], mi[m]) * mo[m]; w_layer += (double)mo[m] * mi[m]; }
    int L = (int)ceil((double)foot_mb * 1048576.0 / per_layer);
    if (L < 2) L = 2;
    size_t mem = (size_t)(per_layer * L * 1.02) + (size_t)L * NM * (ggml_tensor_overhead() + 65536) + (size_t)64 * 1024 * 1024;
    struct ggml_init_params ip = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor ** w = calloc((size_t)L * NM, sizeof(void *));
    double t0 = now();
    for (int l = 0; l < L; l++)
        for (int m = 0; m < NM; m++) {
            struct ggml_tensor * t = ggml_new_tensor_2d(ctx, mt[m], mi[m], mo[m]);
            // two layers are quantized, the others are copies at their own addresses: the rate does not depend on the values (E45),
            // and IQ2_KS's quantizer would take minutes per cell
            if (l < 2) fill(t, four);
            else memcpy(t->data, w[(size_t)(l % 2) * NM + m]->data, ggml_nbytes(t));
            w[(size_t)l * NM + m] = t;
        }
    printf("CONFIG {\"mats\": \"%s\", \"four\": %d, \"layers\": %d, \"threads\": %d, \"tokens\": %d, \"reps\": %d, \"bytes_per_layer\": %.0f, "
           "\"weights_per_layer\": %.0f, \"footprint_mb\": %.1f}\n", mats, four, L, threads, T, R, per_layer, w_layer, per_layer * L / 1048576.0);
    printf("filled %.1f s\n", now() - t0);
    fflush(stdout);
    struct ggml_tensor * xin[MAXM];
    for (int m = 0; m < NM; m++) {
        xin[m] = NULL;
        for (int q = 0; q < m; q++) if (mi[q] == mi[m]) xin[m] = xin[q];
        if (!xin[m]) {
            xin[m] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mi[m]);
            for (int64_t i = 0; i < mi[m]; i++) ((float *)xin[m]->data)[i] = gauss();
        }
    }
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, (size_t)L * NM + 16, false);
    struct ggml_tensor * out0 = NULL;
    for (int l = 0; l < L; l++)
        for (int m = 0; m < NM; m++) {
            struct ggml_tensor * o = ggml_mul_mat(ctx, w[(size_t)l * NM + m], xin[m]);
            if (!out0) out0 = o;
            ggml_build_forward_expand(gf, o);
        }
    struct ggml_cplan plan = ggml_graph_plan(gf, threads);
    plan.work_data = malloc(plan.work_size + 64);
    double * us = malloc(sizeof(double) * R);
    for (int rep = -1; rep < R; rep++) {
        double a = now();
        for (int t = 0; t < T; t++)
            if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        double busy = now() - a;
        if (rep >= 0) us[rep] = 1e6 * busy / T / L;
    }
    qsort(us, R, sizeof(double), cmp_d);
    double med = us[R / 2];
    double err = check(w[0], 0, (const float *)xin[0]->data, (const float *)out0->data);
    double wrong = check(w[0], 64, (const float *)xin[0]->data, (const float *)out0->data);
    printf("RESULT {\"mats\": \"%s\", \"four\": %d, \"us_per_layer\": %.2f, \"min\": %.2f, \"max\": %.2f, \"ms_per_32\": %.3f, \"gweights_per_s\": %.2f, "
           "\"gbytes_per_s\": %.2f, \"rel_err\": %.5f, \"rel_err_wrong\": %.3f}\n", mats, four, med, us[0], us[R - 1], 32 * med / 1e3, w_layer / med / 1e3,
           per_layer / med / 1e3, err, wrong);
    ggml_free(ctx);
    return 0;
}
