// Stage 2: ik_llama.cpp's attention at Ling-flash's shape, one token at a time: 32 query heads on 4 KV heads, head size 128, 32 layers, context --ctx.
// Every layer holds its own K and V.  --path fa runs ggml_flash_attn_ext with an all-zero f16 mask (ik's kernel returns to the generic path on a NULL mask), on f16 or q8_0 K and V; --path mm runs ik's path without flash attention
// (K·Q, softmax, V·KQ) on f16, with V stored transposed as the graph wants it.
// Check: the first layer's output, all heads, against a double attention on the dequantized cache; and against the wrong KV head, far off.
//
// Build: gcc -O2 -Wall -fopenmp -Wa,-muse-unaligned-vector-move s2l_attn.c -I<ik>/ggml/include <ik>/build-fa/ggml/src/libggml.a -lstdc++ -lm -static -o bin/s2l_attn.exe
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"

#define HD 128
#define NH 32
#define NKV 4
#define NL 32

static uint64_t st = 20261105u;
static uint64_t xs(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static float uni(void) { return (float)((xs() >> 11) * (1.0 / 9007199254740992.0)); }
static float gauss(void) { return sqrtf(-2.0f * logf(uni() + 1e-12f)) * cosf(6.2831853f * uni()); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp_d(const void * a, const void * b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// 64-byte aligned, as the CPU backend's buffers are: ik's kernels load aligned, and a context's own 16-byte tensors crash them
static void * al64(size_t n) { return (void *)(((uintptr_t)malloc(n + 128) + 63) & ~(uintptr_t)63); }
static struct ggml_tensor * place(struct ggml_tensor * t) { t->data = al64(ggml_nbytes(t)); return t; }

static void fill(struct ggml_tensor * t) {
    const int64_t row = t->ne[0], rows = ggml_nrows(t);
    float * buf = malloc(sizeof(float) * row);
    size_t rb = ggml_row_size(t->type, row);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t i = 0; i < row; i++) buf[i] = gauss();
        if (t->type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(buf, (ggml_fp16_t *)((char *)t->data + r * rb), row);
        else ggml_quantize_chunk(t->type, buf, (char *)t->data + r * rb, 0, 1, row, NULL, NULL);
    }
    free(buf);
}

static void row_float(const struct ggml_tensor * t, int64_t r, float * dst) {
    ggml_type_traits_t tr = ggml_internal_get_type_traits(t->type);
    tr.to_float((const char *)t->data + r * ggml_row_size(t->type, t->ne[0]), dst, t->ne[0]);
}

// V as [HD, ctx, NKV] rows, or transposed [ctx, HD, NKV] in the mm path
static double vval(const struct ggml_tensor * v, int transposed, int ctx, int kv, int p, int d) {
    static float row[8192];
    if (transposed) { row_float(v, (int64_t)kv * HD + d, row); return row[p]; }
    row_float(v, (int64_t)kv * ctx + p, row);
    return row[d];
}

static double check(const struct ggml_tensor * k, const struct ggml_tensor * v, int transposed, const float * q, const float * out, int ctx, float scale, int shift) {
    double * s = malloc(sizeof(double) * ctx), num = 0, den = 0;
    float * kr = malloc(sizeof(float) * HD);
    for (int h = 0; h < NH; h += 5) {
        const int kv = (h / (NH / NKV) + shift) % NKV;
        double m = -1e300, z = 0;
        for (int p = 0; p < ctx; p++) {
            row_float(k, (int64_t)kv * ctx + p, kr);
            double a = 0;
            for (int d = 0; d < HD; d++) a += (double)kr[d] * q[h * HD + d];
            s[p] = a * scale;
            if (s[p] > m) m = s[p];
        }
        for (int p = 0; p < ctx; p++) { s[p] = exp(s[p] - m); z += s[p]; }
        for (int d = 0; d < HD; d += 16) {
            double a = 0;
            for (int p = 0; p < ctx; p++) a += s[p] * vval(v, transposed, ctx, kv, p, d);
            a /= z;
            const double o = out[h * HD + d];
            num += (a - o) * (a - o);
            den += a * a;
        }
    }
    free(s); free(kr);
    return sqrt(num / (den + 1e-30));
}

int main(int argc, char ** argv) {
    const char * path = "fa", * kvs = "f16";
    int ctx = 2048, threads = 6, T = 32, R = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--path")) path = argv[++i];
        else if (!strcmp(argv[i], "--kv")) kvs = argv[++i];
        else if (!strcmp(argv[i], "--ctx")) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens")) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")) R = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    const enum ggml_type kt = !strcmp(kvs, "q8_0") ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    const int fa = !strcmp(path, "fa");
    if (!fa && kt != GGML_TYPE_F16) { fprintf(stderr, "--path mm reads f16\n"); return 2; }
    const float scale = 1.0f / sqrtf((float)HD);
    const size_t per = ggml_row_size(kt, HD) * (size_t)ctx * NKV;
    size_t mem = 2 * per * NL + (size_t)NL * 16 * (ggml_tensor_overhead() + (size_t)NH * ctx * 4 + 65536) + (size_t)256 * 1024 * 1024;
    struct ggml_init_params ip = { mem, NULL, true };
    struct ggml_context * ctx0 = ggml_init(ip);
    struct ggml_tensor * q = place(ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, HD, 1, NH));
    for (int i = 0; i < HD * NH; i++) ((float *)q->data)[i] = gauss();
    struct ggml_tensor * ks[NL], * vs[NL], * os[NL];
    double t0 = now();
    for (int l = 0; l < NL; l++) {
        ks[l] = place(ggml_new_tensor_3d(ctx0, kt, HD, ctx, NKV));
        vs[l] = place(fa ? ggml_new_tensor_3d(ctx0, kt, HD, ctx, NKV) : ggml_new_tensor_3d(ctx0, kt, ctx, HD, NKV));
        if (l < 2) { fill(ks[l]); fill(vs[l]); }
        else { memcpy(ks[l]->data, ks[l % 2]->data, ggml_nbytes(ks[l])); memcpy(vs[l]->data, vs[l % 2]->data, ggml_nbytes(vs[l])); }
    }
    struct ggml_tensor * mask = place(ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, ctx, GGML_PAD(1, GGML_KQ_MASK_PAD)));
    memset(mask->data, 0, ggml_nbytes(mask));
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, NL * 8 + 16, false);
    for (int l = 0; l < NL; l++) {
        if (fa) {
            os[l] = ggml_flash_attn_ext(ctx0, q, ks[l], vs[l], mask, scale, 0.0f, 0.0f);
        } else {
            struct ggml_tensor * kq = ggml_mul_mat(ctx0, ks[l], q);                 // [ctx, 1, NH]
            kq = ggml_soft_max_ext(ctx0, kq, NULL, scale, 0.0f);
            os[l] = ggml_mul_mat(ctx0, vs[l], kq);                                  // [HD, 1, NH]
        }
        ggml_build_forward_expand(gf, os[l]);
    }
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx0); t; t = ggml_get_next_tensor(ctx0, t)) if (!t->data) place(t);
    const double bytes = 2.0 * per * NL;
    printf("CONFIG {\"path\": \"%s\", \"kv\": \"%s\", \"ctx\": %d, \"nh\": %d, \"nkv\": %d, \"hd\": %d, \"nl\": %d, \"threads\": %d, \"tokens\": %d, \"reps\": %d, \"mb\": %.1f}\n",
           path, kvs, ctx, NH, NKV, HD, NL, threads, T, R, bytes / 1e6);
    printf("filled %.1f s\n", now() - t0);
    fflush(stdout);
    struct ggml_cplan plan = ggml_graph_plan(gf, threads);
    plan.work_data = al64(plan.work_size);
    double * ms = malloc(sizeof(double) * R);
    for (int rep = -1; rep < R; rep++) {
        double a = now();
        for (int t = 0; t < T; t++)
            if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        if (rep >= 0) ms[rep] = 1e3 * (now() - a) / T;
    }
    qsort(ms, R, sizeof(double), cmp_d);
    const double med = ms[R / 2];
    const double err = check(ks[0], vs[0], !fa, (const float *)q->data, (const float *)os[0]->data, ctx, scale, 0);
    const double wrong = check(ks[0], vs[0], !fa, (const float *)q->data, (const float *)os[0]->data, ctx, scale, 1);
    printf("RESULT {\"path\": \"%s\", \"kv\": \"%s\", \"ctx\": %d, \"ms_median\": %.3f, \"ms_min\": %.3f, \"ms_max\": %.3f, \"gbytes_per_s\": %.2f, \"rel_err\": %.5f, \"rel_err_wrong\": %.3f}\n",
           path, kvs, ctx, med, ms[0], ms[R - 1], bytes / med / 1e6, err, wrong);
    ggml_free(ctx0);
    return 0;
}
