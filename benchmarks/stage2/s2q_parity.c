// Stage 2, the fourth code: does ik_llama.cpp's IQ2_BN kernel read code 3 as +2?  Parity against a double dot product.
//
// The stock quantizer writes codes 0, 1, 2 only, so rows are written here by hand: one f32 scale, then 2 bits per weight
// (IQ1BNQuantizer::quantize_one_row_2bn's layout).  IQ2_BN_R4 rows are packed as repack_iq2_bn does; it sits in an anonymous
// namespace, so its loop is copied here (ik_llama.cpp, MIT).
//
// Cases, for each type, shape and batch:
//   tern   codes 0-2, reference s * (code - 1): the kernel's own error, from the activations' 8 bits
//   four   codes 0-3, the same reference
//   neg    codes 0-3 with half the row scales negative: {-1, 0, 1, 2} with a sign per row
//   bal    codes 0-3, kernel output - 0.5 * s * sum(x) against s * (code - 1.5): {-1.5, -0.5, 0.5, 1.5} with the correction
//   wrong  codes 0-3 against the reference that reads code 3 as +1: the planted miss, which must be far off
//
// Build: gcc -O2 -Wall -fopenmp s2q_parity.c -I<ik>/ggml/include <ik>/build-native/ggml/src/libggml.a -lstdc++ -o s2q_parity.exe
// Usage: s2q_parity --threads 6
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"

#define QK 64

static uint64_t st = 20261105u;
static uint64_t xs(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static double uni(void) { return (xs() >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(void) { return sqrt(-2.0 * log(uni() + 1e-12)) * cos(6.283185307179586 * uni()); }

static enum ggml_type type_of(const char * name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((enum ggml_type)t);
        if (n && !strcmp(n, name)) return (enum ggml_type)t;
    }
    fprintf(stderr, "unknown type %s\n", name);
    exit(2);
}

// one row of codes -> 4 + n / 4 bytes
static void pack_row(const uint8_t * c, float s, int n, uint8_t * dst) {
    memcpy(dst, &s, 4);
    uint8_t * q = dst + 4;
    for (int ib = 0; ib < n / QK; ib++) {
        const uint8_t * L = c + ib * QK;
        for (int j = 0; j < 16; j++) q[ib * 16 + j] = L[j] | (L[j + 16] << 2) | (L[j + 32] << 4) | (L[j + 48] << 6);
    }
}

// four packed rows -> the R4 layout
static void repack4(const uint8_t * x, size_t row_size, int n, uint8_t * y) {
    float * dr4 = (float *)y;
    const uint8_t * x4[4];
    for (int k = 0; k < 4; k++) {
        memcpy(&dr4[k], x + k * row_size, 4);
        x4[k] = x + k * row_size + 4;
    }
    uint8_t * y4 = (uint8_t *)(dr4 + 4);
    for (int ib = 0; ib < n / QK; ib++)
        for (int k = 0; k < 4; k++)
            for (int l = 0; l < 4; l++)
                for (int i = 0; i < 4; i++)
                    y4[64 * ib + 4 * k + i + 16 * l] = (((x4[k][16 * ib + i + 0] >> 2 * l) & 3) << 0) | (((x4[k][16 * ib + i + 4] >> 2 * l) & 3) << 2) |
                                                       (((x4[k][16 * ib + i + 8] >> 2 * l) & 3) << 4) | (((x4[k][16 * ib + i + 12] >> 2 * l) & 3) << 6);
}

typedef struct { double num, den; } acc_t;
static void add(acc_t * a, double ref, double got) { a->num += (ref - got) * (ref - got); a->den += ref * ref; }
static double rel(const acc_t * a) { return sqrt(a->num / (a->den + 1e-30)); }

static void one(const char * tname, int out, int in, int batch, int threads, int first) {
    enum ggml_type type = type_of(tname);
    int r4 = strstr(tname, "_r4") != NULL;
    size_t row = 4 + (size_t)in / 4;
    uint8_t * tc = malloc((size_t)out * in), * fc = malloc((size_t)out * in);
    float * sc = malloc(sizeof(float) * out), * sn = malloc(sizeof(float) * out);
    for (int r = 0; r < out; r++) {
        sc[r] = (float)(0.01 + 0.02 * uni());
        sn[r] = r % 2 ? -sc[r] : sc[r];
        for (int c = 0; c < in; c++) {
            size_t i = (size_t)r * in + c;
            tc[i] = (uint8_t)(xs() % 3);
            fc[i] = (uint8_t)(xs() % 4);
        }
    }
    float * x = malloc(sizeof(float) * (size_t)in * batch);
    for (size_t i = 0; i < (size_t)in * batch; i++) x[i] = (float)gauss();
    const struct { const char * name; const uint8_t * codes; const float * s; double off; int wrong; } cases[5] = {
        { "tern", tc, sc, 1.0, 0 }, { "four", fc, sc, 1.0, 0 }, { "neg", fc, sn, 1.0, 0 }, { "bal", fc, sc, 1.5, 0 }, { "wrong", fc, sc, 1.0, 1 } };
    double res[5];
    for (int k = 0; k < 5; k++) {
        size_t mem = (size_t)out * row + (size_t)in * batch * 8 + (size_t)out * batch * 8 + (size_t)64 * 1024 * 1024;
        struct ggml_init_params ip = { mem, NULL, false };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, type, in, out);
        struct ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, batch);
        memcpy(v->data, x, sizeof(float) * (size_t)in * batch);
        uint8_t * tmp = malloc(4 * row);
        for (int r = 0; r < out; r += 4) {
            for (int q = 0; q < 4; q++) pack_row(cases[k].codes + (size_t)(r + q) * in, cases[k].s[r + q], in, tmp + q * row);
            uint8_t * dst = (uint8_t *)w->data + (size_t)r * row;
            if (r4) repack4(tmp, row, in, dst);
            else memcpy(dst, tmp, 4 * row);
        }
        free(tmp);
        struct ggml_tensor * o = ggml_mul_mat(ctx, w, v);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        struct ggml_cplan plan = ggml_graph_plan(gf, threads);
        void * work = malloc(plan.work_size + 64);
        plan.work_data = work;
        if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        const float * y = (const float *)o->data;
        acc_t a = { 0, 0 };
        for (int b = 0; b < batch; b++) {
            const float * xb = x + (size_t)b * in;
            double sumx = 0;
            for (int c = 0; c < in; c++) sumx += xb[c];
            for (int r = 0; r < out; r++) {
                const uint8_t * cr = cases[k].codes + (size_t)r * in;
                double d = 0;
                for (int c = 0; c < in; c++) {
                    double val = cr[c] - cases[k].off;
                    if (cases[k].wrong && cr[c] == 3) val = 1.0;
                    d += val * xb[c];
                }
                double s = cases[k].s[r], got = y[(size_t)b * out + r];
                if (cases[k].off == 1.5) got -= 0.5 * s * sumx;
                add(&a, s * d, got);
            }
        }
        res[k] = rel(&a);
        free(work);
        ggml_free(ctx);
    }
    printf("%s{\"type\": \"%s\", \"out\": %d, \"in\": %d, \"batch\": %d, \"tern\": %.6f, \"four\": %.6f, \"neg\": %.6f, \"bal\": %.6f, \"wrong\": %.4f}",
           first ? "" : ",\n", tname, out, in, batch, res[0], res[1], res[2], res[3], res[4]);
    fflush(stdout);
    free(tc); free(fc); free(sc); free(sn); free(x);
}

int main(int argc, char ** argv) {
    int threads = 6;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    const int shapes[5][2] = { { 4096, 4096 }, { 1024, 4096 }, { 4096, 1024 }, { 512, 4096 }, { 2048, 512 } };
    const char * types[2] = { "iq2_bn", "iq2_bn_r4" };
    const int batches[3] = { 1, 4, 9 };
    printf("{\"threads\": %d, \"cases\": [\n", threads);
    int first = 1;
    for (int t = 0; t < 2; t++)
        for (int s = 0; s < 5; s++)
            for (int b = 0; b < 3; b++) { one(types[t], shapes[s][0], shapes[s][1], batches[b], threads, first); first = 0; }
    printf("\n]}\n");
    return 0;
}
