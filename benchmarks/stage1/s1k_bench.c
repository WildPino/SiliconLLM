// Stage 1 (c): weight rate of a ggml CPU kernel at Ling-flash's decode shapes, one token at a time.
//
// Cells, each over L layers with distinct weights so that the weights stream from RAM:
//   "experts":   per layer, mul_mat_id of k experts' gate and up [4096 -> 1024] and of their down [1024 -> 4096], the
//                k experts drawn at random per layer and per token out of E stored in one 3-D tensor (llama.cpp's
//                MoE path);
//   "experts2d": the same matrices stored one per 2-D tensor, and one mul_mat per matrix; the graph is rebuilt per
//                token with the drawn experts (for kernels that only take 2-D weights, as bitnet.cpp's I2_S);
//   "attn":      one mul_mat [4096 -> 4096] per layer.
// --resident sets L = 1 and E = k, so the same weights stay in cache: the compute ceiling.
//
// Rate = weights the token reads / compute time (graph build and expert draw excluded), in G weights/s; GB/s = the
// bytes of those tensors / time.  R reps of T tokens after one warm-up rep; it prints every rep, the median, min, max.
// Correctness: 64 outputs of layer 0 against a double dot product on the dequantized weights, and the same against a
// wrong expert (or the next layer's matrix), which must be far off.
//
// Build: gcc -O2 -fopenmp [-DMAINLINE | -DMAINLINE -DBITNET [-DBITNET_TL2]] s1k_bench.c -I<ggml/include> <ggml libraries>
//   -lstdc++.  No define: ik_llama.cpp;  -DMAINLINE: llama.cpp;  -DMAINLINE -DBITNET: bitnet.cpp's llama.cpp (I2_S);
//   with -DBITNET_TL2 also its TL2 type, linked with s1k_tl2_glue.cpp against a BITNET_X86_TL2 build, weights from
//   s1k_tl2_weights.py (--tl2-dir).
// Usage: s1k_bench --type tq2_0 --cell experts --layers 16 --experts 16 --k 4 --threads 6 --tokens 32 --reps 5
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#ifdef MAINLINE
#include "ggml-cpu.h"
#ifdef BITNET
#define BUILD "bitnet.cpp"
#ifdef BITNET_TL2
#include "ggml-bitnet.h"
#endif
void dequantize_row_i2_s(const uint8_t * x, float * y, int64_t n, const float i2_scale);
#else
#define BUILD "llama.cpp"
#endif
#else
#define BUILD "ik_llama.cpp"
#endif

#define D_MODEL 4096
#define D_FF 1024

// xorshift64 on an explicit state, so that fill's chunks can draw in parallel, each from its own seed
static uint64_t rng = 20261103u;
static uint64_t xs(uint64_t * st) { *st ^= *st << 13; *st ^= *st >> 7; *st ^= *st << 17; return *st; }
static uint64_t next_u64(void) { return xs(&rng); }
static float uni_s(uint64_t * st) { return (float)((xs(st) >> 11) * (1.0 / 9007199254740992.0)); }
static float gauss_s(uint64_t * st) {
    float u = uni_s(st) + 1e-12f, v = uni_s(st);
    return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
}
static float uni(void) { return uni_s(&rng); }
static float gauss(void) { return gauss_s(&rng); }
static uint64_t seed_of(uint64_t a, uint64_t b) {  // splitmix64 of (a, b), never 0
    uint64_t z = a * 0x9E3779B97F4A7C15ull + b + 0x632BE59BD9B4E019ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return z ? z : 1;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static int is_ternary(enum ggml_type t) {
    const char * n = ggml_type_name(t);
    return !strcmp(n, "tq1_0") || !strcmp(n, "tq2_0") || !strcmp(n, "iq1_bn") || !strcmp(n, "iq2_bn") ||
           !strcmp(n, "iq2_bn_r4") || !strcmp(n, "i2_s") || !strcmp(n, "tl2");
}
static int is_i2s(enum ggml_type t) { return !strcmp(ggml_type_name(t), "i2_s"); }
static int is_tl2(enum ggml_type t) { return !strcmp(ggml_type_name(t), "tl2"); }

// TL2 weights per shape (out x in), packed by s1k_tl2_weights.py: bytes, sign matrix and scale
#ifdef BITNET_TL2
static const char * tl2_dir = NULL;
static struct { int64_t out, in; char * bytes; size_t n; int8_t * sign; float scale; } tl2w[3];
static void * slurp(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    char * b = malloc(*n + 1);
    if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read %s\n", path); exit(2); }
    b[*n] = 0;
    fclose(f);
    return b;
}
static int tl2_of(int64_t out, int64_t in) {
    for (int i = 0; i < 3; i++) {
        if (tl2w[i].bytes && tl2w[i].out == out && tl2w[i].in == in) return i;
        if (!tl2w[i].bytes) {
            char p[1024];
            size_t ns;
            snprintf(p, sizeof p, "%s/%lld_%lld.tl2", tl2_dir, (long long)out, (long long)in);
            tl2w[i].bytes = slurp(p, &tl2w[i].n);
            snprintf(p, sizeof p, "%s/%lld_%lld.sign", tl2_dir, (long long)out, (long long)in);
            tl2w[i].sign = slurp(p, &ns);
            snprintf(p, sizeof p, "%s/%lld_%lld.json", tl2_dir, (long long)out, (long long)in);
            char * js = slurp(p, &ns);
            char * at = strstr(js, "\"scale\": ");
            if (!at) { fprintf(stderr, "no scale in %s\n", p); exit(2); }
            tl2w[i].scale = strtof(at + 9, NULL);
            free(js);
            tl2w[i].out = out; tl2w[i].in = in;
            return i;
        }
    }
    fprintf(stderr, "too many TL2 shapes\n");
    exit(2);
}
#endif

// Row r of a [kdim, nrows] matrix to floats.  ik_llama.cpp's iq1_bn and iq2_bn rows start with a row scale (fp16 or
// f32, row_meta_size bytes) that their to_float does not apply; it is applied here.  bitnet.cpp's I2_S packs 4 weights
// per byte with one f32 scale after the packed tensor.  Row-interleaved types (_r4) cannot be read one row at a time:
// returns 0 and the check is reported as not done (-1).
static int row_to_float(const struct ggml_tensor * w, const char * base, int64_t r, float * dst, int64_t kdim, int64_t n_all) {
#ifdef MAINLINE
#ifdef BITNET_TL2
    if (is_tl2(w->type)) {
        int i = tl2_of(w->ne[1], w->ne[0]);
        for (int64_t c = 0; c < kdim; c++) dst[c] = tl2w[i].scale * tl2w[i].sign[r * kdim + c];
        return 1;
    }
#endif
#ifdef BITNET
    if (is_i2s(w->type)) {
        dequantize_row_i2_s((const uint8_t *)base + r * kdim / 4, dst, kdim, *(const float *)(base + n_all / 4));
        return 1;
    }
#endif
    (void)n_all;
    ggml_get_type_traits(w->type)->to_float(base + r * w->nb[1], dst, kdim);
    return 1;
#else
    (void)n_all;
    ggml_type_traits_t tr = ggml_internal_get_type_traits(w->type);
    if (strstr(ggml_type_name(w->type), "_r4") || strstr(ggml_type_name(w->type), "_r8")) return 0;
    const char * src = base + r * w->nb[1];
    float scale = 1.0f;
    if (tr.row_meta_size == 4) scale = *(const float *)src;
    else if (tr.row_meta_size == 2) scale = ggml_fp16_to_fp32(*(const ggml_fp16_t *)src);
    else if (tr.row_meta_size != 0) return 0;
    tr.to_float(src + tr.row_meta_size, dst, kdim);
    for (int64_t i = 0; i < kdim; i++) dst[i] *= scale;
    return 1;
#endif
}

// Fill a [n_per_row, nrows] matrix of type t, quantized in chunks of 256 rows (I2_S: in one call, since its scale is
// per tensor).  Ternary types get s * {-1, 0, +1}, P(0) = 0.4, one s per row (I2_S: one per tensor); the others get
// N(0, 0.02).
static void fill(struct ggml_tensor * w, char * base, int64_t n_per_row, int64_t nrows, int ternary) {
#ifdef BITNET_TL2
    if (is_tl2(w->type)) {
        // the packed bytes, then the scale where bitnet.cpp's transform reads it, then the transform
        int i = tl2_of(nrows, n_per_row);
        int64_t k = n_per_row, m = nrows, nb = (k - 256) * m / 3 * 5 / 8 + 256 * m / 2 * 4 / 8;
        if (nb % 32 != 0) nb = 32 - nb % 32 + nb;
        if ((size_t)nb < tl2w[i].n || (size_t)nb + 4 > ggml_nbytes(w)) { fprintf(stderr, "TL2 size mismatch\n"); exit(2); }
        memcpy(base, tl2w[i].bytes, tl2w[i].n);
        memcpy(base + nb, &tl2w[i].scale, sizeof(float));
        ggml_bitnet_transform_tensor(w);
        return;
    }
#endif
    // chunks are drawn and quantized in parallel (OpenMP), each from its own seed: the fill is not timed
    static uint64_t fill_id = 0;
    const uint64_t id = ++fill_id;
    const int64_t chunk = is_i2s(w->type) ? nrows : 256, nchunk = (nrows + chunk - 1) / chunk;
    float s_tensor = 0.01f + 0.02f * uni();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t ci = 0; ci < nchunk; ci++) {
        uint64_t st = seed_of(id, (uint64_t)ci);
        int64_t r0 = ci * chunk, nr = nrows - r0 < chunk ? nrows - r0 : chunk;
        float * buf = malloc(sizeof(float) * n_per_row * nr);
        for (int64_t r = 0; r < nr; r++) {
            float s = is_i2s(w->type) ? s_tensor : 0.01f + 0.02f * uni_s(&st);
            for (int64_t c = 0; c < n_per_row; c++) {
                float u = uni_s(&st);
                buf[r * n_per_row + c] = ternary ? (u < 0.4f ? 0.0f : (u < 0.7f ? s : -s)) : 0.02f * gauss_s(&st);
            }
        }
        // start = 0: ggml offsets src by start too, and buf holds only this chunk
        char * dst = base + r0 * ggml_row_size(w->type, n_per_row);
#ifdef MAINLINE
        ggml_quantize_chunk(w->type, buf, dst, 0, nr, n_per_row, NULL);
#else
        ggml_quantize_chunk(w->type, buf, dst, 0, nr, n_per_row, NULL, NULL);
#endif
        free(buf);
    }
}

static void randomize(struct ggml_tensor * x) {
    float * d = (float *)x->data;
    for (int64_t i = 0; i < ggml_nelements(x); i++) d[i] = gauss();
}

static int cmp_d(const void * a, const void * b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// Relative error of out[0..64) against rows [row0, row0 + 64) of matrix w (at base, kdim columns, n_all weights) times x.
static double check(const struct ggml_tensor * w, const char * base, int64_t row0, const float * x, const float * out,
                    int64_t kdim, int64_t n_all) {
    float * row = malloc(sizeof(float) * kdim);
    double num = 0, den = 0;
    for (int64_t r = 0; r < 64; r++) {
        if (!row_to_float(w, base, row0 + r, row, kdim, n_all)) { free(row); return -1; }
        double s = 0;
        for (int64_t c = 0; c < kdim; c++) s += (double)row[c] * x[c];
        num += (s - out[r]) * (s - out[r]);
        den += s * s;
    }
    free(row);
    return sqrt(num / (den + 1e-30));
}

static void draw(int32_t * d, int K, int E, int resident) {
    for (int j = 0; j < K; j++) {
        int e, dup;
        do {
            e = resident ? j : (int)(next_u64() % (uint64_t)E);
            dup = 0;
            for (int q = 0; q < j; q++) dup |= d[q] == e;
        } while (dup);
        d[j] = e;
    }
}

static struct ggml_cplan plan_of(struct ggml_cgraph * gf, int threads, void ** work, size_t * cap) {
#ifdef MAINLINE
    struct ggml_cplan p = ggml_graph_plan(gf, threads, NULL);
#else
    struct ggml_cplan p = ggml_graph_plan(gf, threads);
#endif
    if (p.work_size > *cap) { free(*work); *cap = p.work_size; *work = malloc(*cap); }
    p.work_data = *work;
    return p;
}

int main(int argc, char ** argv) {
    const char * tname = "tq2_0", * cell = "experts";
    int L = 16, E = 16, K = 4, threads = 6, T = 32, R = 5, resident = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--type")) tname = argv[++i];
        else if (!strcmp(argv[i], "--cell")) cell = argv[++i];
        else if (!strcmp(argv[i], "--layers")) L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts")) E = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens")) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")) R = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--resident")) resident = 1;
#ifdef BITNET_TL2
        else if (!strcmp(argv[i], "--tl2-dir")) tl2_dir = argv[++i];
#endif
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    int attn = !strcmp(cell, "attn"), e3d = !strcmp(cell, "experts");
    if (!attn && !e3d && strcmp(cell, "experts2d")) { fprintf(stderr, "cell must be experts, experts2d or attn\n"); return 2; }
    if (attn) { E = 1; K = 1; }
    if (resident) { L = 1; E = K; }
    enum ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((enum ggml_type)t);
        if (n && !strcmp(n, tname)) { type = (enum ggml_type)t; break; }
    }
    if (type == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", tname); return 2; }
    if ((is_i2s(type) || is_tl2(type)) && e3d) { fprintf(stderr, "%s takes 2-D weights only: use experts2d\n", tname); return 2; }
#ifdef BITNET_TL2
    if (is_tl2(type) && !tl2_dir) { fprintf(stderr, "tl2 needs --tl2-dir\n"); return 2; }
    ggml_bitnet_init();
#endif
    int tern = is_ternary(type);
#ifdef MAINLINE
    ggml_cpu_init();
#endif

    // matrices per expert: gate, up [D_MODEL -> D_FF] and down [D_FF -> D_MODEL]; attn: one [D_MODEL -> D_MODEL]
    const int NM = attn ? 1 : 3;
    const int64_t kin[3] = { D_MODEL, D_MODEL, D_FF }, nout[3] = { attn ? D_MODEL : D_FF, D_FF, D_MODEL };
    size_t mem = (size_t)256 * 1024 * 1024 + (size_t)L * E * 8 * ggml_tensor_overhead();
    for (int m = 0; m < NM; m++) mem += (size_t)L * E * (ggml_row_size(type, kin[m]) * nout[m] + 64);
    struct ggml_init_params ip = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    // w[(l * E + e) * NM + m]: 2-D matrices (experts2d, attn); w3[l * NM + m]: 3-D expert tensors (experts)
    struct ggml_tensor ** w = calloc((size_t)L * E * NM, sizeof(void *)), ** w3 = calloc((size_t)L * NM, sizeof(void *));
    struct ggml_tensor ** ids = calloc(L, sizeof(void *));
    double t0 = now(), b_mat[3] = { 0, 0, 0 }, footprint = 0;
    for (int l = 0; l < L; l++) {
        for (int m = 0; m < NM; m++) {
            if (e3d) {
                struct ggml_tensor * t = ggml_new_tensor_3d(ctx, type, kin[m], nout[m], E);
                fill(t, (char *)t->data, kin[m], nout[m] * E, tern);
                w3[l * NM + m] = t;
                b_mat[m] = (double)ggml_nbytes(t) / E;
                footprint += ggml_nbytes(t);
            } else {
                for (int e = 0; e < E; e++) {
                    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, type, kin[m], nout[m]);
                    fill(t, (char *)t->data, kin[m], nout[m], tern);
                    w[((size_t)l * E + e) * NM + m] = t;
                    double nb = (double)ggml_nbytes(t);
#ifdef BITNET_TL2
                    if (is_tl2(type)) nb = (double)tl2w[tl2_of(nout[m], kin[m])].n;  // the bytes the kernel reads
#endif
                    b_mat[m] = nb;
                    footprint += nb;
                }
            }
        }
        if (e3d) ids[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, 1);
    }
    double w_tok = 0, b_tok = 0;
    for (int m = 0; m < NM; m++) { w_tok += (double)L * K * kin[m] * nout[m]; b_tok += (double)L * K * b_mat[m]; }
    printf("CONFIG {\"build\": \"%s\", \"type\": \"%s\", \"cell\": \"%s\", \"resident\": %d, \"layers\": %d, "
           "\"experts\": %d, \"k\": %d, \"threads\": %d, \"tokens\": %d, \"reps\": %d, \"ternary_values\": %d, "
           "\"weights_per_token\": %.0f, \"bytes_per_token\": %.0f, \"bytes_per_weight\": %.4f, \"footprint_mb\": %.1f}\n",
           BUILD, tname, cell, resident, L, E, K, threads, T, R, tern, w_tok, b_tok, b_tok / w_tok, footprint / 1048576.0);
    printf("filled %.1f s\n", now() - t0);
    fflush(stdout);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, 1);
    struct ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_FF, 1);
    struct ggml_tensor * hk = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_FF, K, 1);
    randomize(x);
    randomize(h);
    randomize(hk);
    void * work = NULL;
    size_t cap = 0;

    // the 3-D cell builds its graph once; the 2-D cells per token, in a scratch context
    struct ggml_cgraph * gf3 = NULL;
    struct ggml_tensor * out0 = NULL;
    if (e3d) {
        gf3 = ggml_new_graph_custom(ctx, 4 * L + 16, false);
        struct ggml_tensor * x3 = ggml_reshape_3d(ctx, x, D_MODEL, 1, 1);
        for (int l = 0; l < L; l++) {
            struct ggml_tensor * o = ggml_mul_mat_id(ctx, w3[l * NM + 0], x3, ids[l]);
            if (l == 0) out0 = o;
            ggml_build_forward_expand(gf3, o);
            ggml_build_forward_expand(gf3, ggml_mul_mat_id(ctx, w3[l * NM + 1], x3, ids[l]));
            ggml_build_forward_expand(gf3, ggml_mul_mat_id(ctx, w3[l * NM + 2], hk, ids[l]));
        }
    }
    size_t scratch = (size_t)L * K * NM * (ggml_tensor_overhead() + D_MODEL * sizeof(float) + 64) +
                     ggml_graph_overhead_custom(L * K * NM + 16, false) + 1024 * 1024;
    int32_t * draw0 = calloc((size_t)L * K, sizeof(int32_t));
    double * rates = malloc(sizeof(double) * R);
    struct ggml_context * sc = NULL;
    for (int rep = -1; rep < R; rep++) {
        double busy = 0;
        for (int t = 0; t < T; t++) {
            struct ggml_cgraph * gf = gf3;
            for (int l = 0; l < L; l++) draw(draw0 + l * K, K, E, resident);
            if (e3d) {
                for (int l = 0; l < L; l++) memcpy(ids[l]->data, draw0 + l * K, sizeof(int32_t) * K);
            } else {
                if (sc) ggml_free(sc);
                struct ggml_init_params sp = { scratch, NULL, false };
                sc = ggml_init(sp);
                gf = ggml_new_graph_custom(sc, L * K * NM + 16, false);
                for (int l = 0; l < L; l++)
                    for (int j = 0; j < K; j++)
                        for (int m = 0; m < NM; m++) {
                            struct ggml_tensor * o = ggml_mul_mat(sc, w[((size_t)l * E + draw0[l * K + j]) * NM + m], m == 2 ? h : x);
                            if (l == 0 && j == 0 && m == 0) out0 = o;
                            ggml_build_forward_expand(gf, o);
                        }
            }
            struct ggml_cplan plan = plan_of(gf, threads, &work, &cap);
            double a = now();
            if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
            busy += now() - a;
        }
        double gws = w_tok * T / busy / 1e9;
        if (rep >= 0) rates[rep] = gws;
        printf("REP %d %s seconds %.4f us_per_token %.1f gweights_per_s %.2f gbytes_per_s %.2f\n", rep,
               rep < 0 ? "warmup" : "timed", busy, 1e6 * busy / T, gws, b_tok * T / busy / 1e9);
        fflush(stdout);
    }

    // correctness on the last token: the first 64 outputs of layer 0's first drawn matrix against its rows 0-63, and
    // against its rows 64-127 (the wrong rows, which must be far off; copies of one matrix, as TL2's, cannot fool it)
    double err, wrong;
    const float * o0 = (const float *)out0->data;
    const struct ggml_tensor * t = e3d ? w3[0] : w[(size_t)draw0[0] * NM];
    const char * base = (const char *)t->data + (e3d ? (int64_t)draw0[0] * D_FF * t->nb[1] : 0);
    int64_t n_all = e3d ? (int64_t)D_MODEL * D_FF : ggml_nelements(t);
    err = check(t, base, 0, (const float *)x->data, o0, D_MODEL, n_all);
    wrong = check(t, base, 64, (const float *)x->data, o0, D_MODEL, n_all);
    qsort(rates, R, sizeof(double), cmp_d);
    printf("RESULT {\"build\": \"%s\", \"type\": \"%s\", \"cell\": \"%s\", \"resident\": %d, \"median_gweights_per_s\": %.2f, "
           "\"min\": %.2f, \"max\": %.2f, \"median_gbytes_per_s\": %.2f, \"bytes_per_weight\": %.4f, \"rel_err\": %.5f, "
           "\"rel_err_wrong\": %.3f}\n",
           BUILD, tname, cell, resident, rates[R / 2], rates[0], rates[R - 1], rates[R / 2] * b_tok / w_tok,
           b_tok / w_tok, err, wrong);
    if (sc) ggml_free(sc);
    ggml_free(ctx);
    return 0;
}
