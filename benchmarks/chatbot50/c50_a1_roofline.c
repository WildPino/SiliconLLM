/*
 * C50-A1: kernel and layer roofline at Ling-mini-2.0 decode shapes.
 * Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A1_KERNEL_ROOFLINE_PROTOCOL_20261016.md
 *
 * Synthetic weight values, real shapes.  One process runs every cell once, in an
 * order rotated by --order, and prints one JSON line per cell.  It reports numbers
 * only; the gates are applied by c50_a1_verdict.py.
 *
 * Build (WinLibs GCC 15.2.0):
 *   gcc -O2 -mavx2 -mfma -mf16c -o c50_a1_roofline.exe c50_a1_roofline.c
 * Usage:
 *   c50_a1_roofline.exe --selftest
 *   c50_a1_roofline.exe --run --order K [--seed S] [--quick] [--threads N]
 * --threads N (H2, Phase 3 portability): worker threads on the first N physical cores, default 6 (the 3600X);
 * the attention cell keeps the engine's six position chunks, spread over the N threads.
 */
#define _WIN32_WINNT 0x0A00
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>

#include "c50_kquants.h"

#define NT_MAX 32
static int g_nt = 6;   /* H2: --threads, default 6 */
#define GIB (1024.0 * 1024.0 * 1024.0)
#define POOL_MIN_BYTES (1ULL << 30)
#define D_MODEL 2048U
#define N_HEAD 16U
#define N_KV 4U
#define HEAD_DIM 128U
#define ROPE_DIM 64U
#define N_EXPERT 256U
#define N_USED 8U
#define N_GROUP 8U
#define TOPK_GROUP 4U
#define EXP_FF 512U
#define DENSE_FF 5120U
#define CTX 2048U
#define VOCAB 157184U
#define N_LAYER 20U
#define ROUTED_SCALE 2.5f

/* ------------------------------------------------------------------ classes -- */
enum { C_Q4K, C_Q6K, C_IQ4XS, C_IQ3S, C_Q80, C_COUNT };
static const char *CLASS_NAME[C_COUNT] = {"Q4_K", "Q6_K", "IQ4_XS", "IQ3_S", "Q8_0"};
static const unsigned BLOCK_ELEMS[C_COUNT] = {256, 256, 256, 256, 32};
static const unsigned BLOCK_BYTES[C_COUNT] = {144, 210, 136, 110, 34};
static const unsigned SCALE_OFFSET[C_COUNT] = {0, 208, 0, 0, 0};

static size_t row_bytes(int c, unsigned cols) { return (size_t)cols / BLOCK_ELEMS[c] * BLOCK_BYTES[c]; }

/* activation buffers: Q8_K blocks for k-quants, Q8_0 blocks for Q8_0 */
typedef struct { strat01_q8_k_block k[DENSE_FF / 256]; c50_block_q8_0 z[DENSE_FF / 32]; } act_q;

static float dot_row(int c, const uint8_t *row, const act_q *a, unsigned cols) {
    switch (c) {
    case C_Q4K: return strat01_q4k_q8k_dot_avx2(row, a->k, cols);
    case C_Q6K: return strat01_q6k_q8k_dot_avx2(row, a->k, cols);
    case C_IQ4XS: return c50_iq4_xs_dot_avx2((const c50_block_iq4_xs *)row, a->k, cols);
    case C_IQ3S: return c50_iq3_s_dot_avx2((const c50_block_iq3_s *)row, a->k, cols);
    default: return c50_q8_0_dot_avx2((const c50_block_q8_0 *)row, a->z, cols);
    }
}

static float dot_row_generic(int c, const uint8_t *row, const act_q *a, unsigned cols) {
    switch (c) {
    case C_Q4K: return strat01_q4k_q8k_dot_generic(row, a->k, cols);
    case C_Q6K: return strat01_q6k_q8k_dot_generic(row, a->k, cols);
    case C_IQ4XS: return c50_iq4_xs_dot_generic((const c50_block_iq4_xs *)row, a->k, cols);
    case C_IQ3S: return c50_iq3_s_dot_generic((const c50_block_iq3_s *)row, a->k, cols);
    default: return c50_q8_0_dot_generic((const c50_block_q8_0 *)row, a->z, cols);
    }
}

static void quantize_act(int c, const float *x, unsigned n, act_q *a) {
    if (c == C_Q80) c50_quantize_row_q8_0(x, a->z, n);
    else if (!strat01_quantize_q8_k_row(x, n, a->k)) { fprintf(stderr, "q8_K quantization failed\n"); exit(2); }
}

/* --------------------------------------------------------------------- rng -- */
static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void) {
    g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
    return g_rng * 2685821657736338717ULL;
}
static float frand(void) { return (float)((rnd() >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }

static void *xalloc(size_t bytes) {
    void *p = VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) { fprintf(stderr, "VirtualAlloc(%zu) failed\n", bytes); exit(2); }
    return p;
}
static void xfree(void *p) { if (p) VirtualFree(p, 0, MEM_RELEASE); }

/* Random quants with finite FP16 block scales in [2^-8, 2^-7). */
static void fill_quant(int c, uint8_t *p, size_t bytes) {
    size_t i, nblocks = bytes / BLOCK_BYTES[c];
    for (i = 0; i + 8 <= bytes; i += 8) { uint64_t r = rnd(); memcpy(p + i, &r, 8); }
    for (i = 0; i < nblocks; ++i) {
        uint8_t *b = p + i * BLOCK_BYTES[c] + SCALE_OFFSET[c];
        uint16_t h = (uint16_t)(0x1C00U | (rnd() & 0x3FFU));
        memcpy(b, &h, 2);
        if (c == C_Q4K) { h = (uint16_t)(0x1C00U | (rnd() & 0x3FFU)); memcpy(b + 2, &h, 2); }
    }
}

/* ------------------------------------------------------------- thread pool -- */
typedef void (*job_fn)(int tid, void *arg);
static volatile LONG g_gen, g_done;
static job_fn volatile g_fn;
static void *volatile g_arg;
static DWORD_PTR g_mask[NT_MAX];
static int g_cpu[NT_MAX];
static char g_cores[NT_MAX * 4];   /* the core list as printed on the CONFIG line */

static DWORD WINAPI worker_main(LPVOID p) {
    int tid = (int)(intptr_t)p;
    LONG seen = 0;
    SetThreadAffinityMask(GetCurrentThread(), g_mask[tid]);
    for (;;) {
        while (g_gen == seen) _mm_pause();
        seen = g_gen;
        if (!g_fn) return 0;
        g_fn(tid, g_arg);
        InterlockedIncrement(&g_done);
    }
}

static void run(job_fn fn, void *arg) {
    g_fn = fn;
    g_arg = arg;
    g_done = 0;
    MemoryBarrier();
    InterlockedIncrement(&g_gen);
    fn(0, arg);
    while (g_done < g_nt - 1) _mm_pause();
}

/* One logical processor per physical core, first g_nt cores. */
static void pick_cores(void) {
    DWORD len = 0;
    char *buf, *ptr;
    int n = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    buf = malloc(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        fprintf(stderr, "topology query failed\n"); exit(2);
    }
    for (ptr = buf; ptr < buf + len && n < g_nt;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
        KAFFINITY m = info->Processor.GroupMask[0].Mask;
        int bit = 0;
        while (!((m >> bit) & 1)) ++bit;
        g_mask[n] = (DWORD_PTR)1 << bit;
        g_cpu[n] = bit;
        snprintf(g_cores + strlen(g_cores), sizeof g_cores - strlen(g_cores), "%s%d", n ? "," : "", bit);
        ++n;
        ptr += info->Size;
    }
    free(buf);
    if (n < g_nt) { fprintf(stderr, "fewer than %d physical cores\n", g_nt); exit(2); }
}

static void start_pool(void) {
    int t;
    pick_cores();
    SetThreadAffinityMask(GetCurrentThread(), g_mask[0]);
    for (t = 1; t < g_nt; ++t) CreateThread(NULL, 0, worker_main, (LPVOID)(intptr_t)t, 0, NULL);
}

static double now_s(void) {
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* -------------------------------------------------------------- matvec job -- */
/* A "virtual" matrix: nseg segments of rows_per_seg rows, segment s at base[s],
 * multiplied by act[s] (or act[0] when shared_act). */
typedef struct {
    int c;
    unsigned cols, rows_per_seg, nseg, shared_act;
    const uint8_t *base[16];
    const act_q *act[16];
    float *out;
} mv_job;

static void mv_worker(int tid, void *p) {
    mv_job *j = p;
    unsigned total = j->rows_per_seg * j->nseg;
    unsigned r0 = (unsigned)((uint64_t)total * tid / g_nt), r1 = (unsigned)((uint64_t)total * (tid + 1) / g_nt), r;
    size_t rb = row_bytes(j->c, j->cols);
    for (r = r0; r < r1; ++r) {
        unsigned s = r / j->rows_per_seg, lr = r % j->rows_per_seg;
        j->out[r] = dot_row(j->c, j->base[s] + (size_t)lr * rb, j->shared_act ? j->act[0] : j->act[s], j->cols);
    }
}

/* ------------------------------------------------------------ expert block -- */
typedef struct {
    int c;
    unsigned nexp, cols, ff;  /* cols = D_MODEL, ff = expert hidden */
    const uint8_t *gu[16], *dn[16];
    float w[16];
    const act_q *x;
    float *h;                 /* nexp * 2ff */
    act_q *hq;                /* nexp */
    float *y;                 /* nexp * cols */
    float *out;               /* cols */
} exp_job;

static void exp_act_worker(int tid, void *p) {
    exp_job *j = p;
    unsigned e, i;
    for (e = tid; e < j->nexp; e += g_nt) {
        float tmp[DENSE_FF];
        const float *g = j->h + (size_t)e * 2 * j->ff, *u = g + j->ff;
        for (i = 0; i < j->ff; i += 8)
            _mm256_storeu_ps(tmp + i, _mm256_mul_ps(c50_v_silu(_mm256_loadu_ps(g + i)), _mm256_loadu_ps(u + i)));
        quantize_act(j->c, tmp, j->ff, &j->hq[e]);
    }
}

static void exp_combine_worker(int tid, void *p) {
    exp_job *j = p;
    unsigned i0 = j->cols * tid / g_nt, i1 = j->cols * (tid + 1) / g_nt, i, e;
    for (i = i0; i < i1; ++i) {
        float s = 0.0f;
        for (e = 0; e < j->nexp; ++e) s += j->w[e] * j->y[(size_t)e * j->cols + i];
        j->out[i] += s;
    }
}

/* gate+up for every expert, SwiGLU + quantize, down, weighted combine into out. */
static void (*g_phase_hook)(int) = NULL;   /* set while timing a layer */

static void run_experts(exp_job *j) {
    mv_job m;
    unsigned e;
    memset(&m, 0, sizeof m);
    m.c = j->c; m.cols = j->cols; m.rows_per_seg = 2 * j->ff; m.nseg = j->nexp; m.shared_act = 1;
    m.act[0] = j->x; m.out = j->h;
    for (e = 0; e < j->nexp; ++e) m.base[e] = j->gu[e];
    run(mv_worker, &m);
    if (g_phase_hook) g_phase_hook(8);
    run(exp_act_worker, j);
    if (g_phase_hook) g_phase_hook(9);
    m.cols = j->ff; m.rows_per_seg = j->cols; m.shared_act = 0; m.out = j->y;
    for (e = 0; e < j->nexp; ++e) { m.base[e] = j->dn[e]; m.act[e] = &j->hq[e]; }
    run(mv_worker, &m);
    if (g_phase_hook) g_phase_hook(10);
    run(exp_combine_worker, j);
    if (g_phase_hook) g_phase_hook(11);
}

/* -------------------------------------------------------------- statistics -- */
static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double median_of(double *v, int n) {
    qsort(v, n, sizeof *v, cmpd);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void emit(const char *cell, const char *cls, const char *shape, double bytes, double *t, int n) {
    double gbs[512], med, tmed;
    int i;
    for (i = 0; i < n; ++i) gbs[i] = bytes / t[i] / 1e9;
    tmed = median_of(t, n);
    med = median_of(gbs, n);
    printf("{\"cell\":\"%s\",\"class\":\"%s\",\"shape\":\"%s\",\"bytes\":%.0f,\"calls\":%d,"
           "\"median_s\":%.9f,\"min_s\":%.9f,\"max_s\":%.9f,\"median_gbs\":%.4f,\"min_gbs\":%.4f,\"max_gbs\":%.4f}\n",
           cell, cls, shape, bytes, n, tmed, t[0], t[n - 1], med, gbs[0], gbs[n - 1]);
    fflush(stdout);
}

/* ------------------------------------------------------------ dense matvec cells -- */
static void cell_matvec(const char *cell, int c, unsigned rows, unsigned cols, int calls) {
    size_t mb = row_bytes(c, cols) * rows;
    unsigned copies = (unsigned)((POOL_MIN_BYTES + mb - 1) / mb), k;
    uint8_t *pool = xalloc(mb * copies);
    float *x = xalloc(sizeof(float) * cols), *out = xalloc(sizeof(float) * rows);
    act_q *a = xalloc(sizeof(act_q));
    double t[512];
    mv_job m;
    char shape[64];
    int i;
    fill_quant(c, pool, mb * copies);
    for (k = 0; k < cols; ++k) x[k] = frand();
    quantize_act(c, x, cols, a);
    memset(&m, 0, sizeof m);
    m.c = c; m.cols = cols; m.rows_per_seg = rows; m.nseg = 1; m.shared_act = 1; m.act[0] = a; m.out = out;
    for (k = 0; k < copies; ++k) { m.base[0] = pool + (size_t)k * mb; run(mv_worker, &m); }  /* warm-up pass */
    for (i = 0; i < calls; ++i) {
        double t0;
        m.base[0] = pool + (size_t)(i % copies) * mb;
        t0 = now_s();
        run(mv_worker, &m);
        t[i] = now_s() - t0;
    }
    snprintf(shape, sizeof shape, "%ux%u", rows, cols);
    emit(cell, CLASS_NAME[c], shape, (double)mb, t, calls);
    xfree(pool); xfree(x); xfree(out); xfree(a);
}

/* expert pools: layers x 256 experts, each expert = gate+up [2ff x D] then down [D x ff] */
typedef struct { int c; unsigned layers; size_t gu_bytes, dn_bytes; uint8_t *mem; } expert_pool;

static uint8_t *pool_gu(const expert_pool *p, unsigned l, unsigned e) {
    return p->mem + ((size_t)l * N_EXPERT + e) * (p->gu_bytes + p->dn_bytes);
}
static uint8_t *pool_dn(const expert_pool *p, unsigned l, unsigned e) { return pool_gu(p, l, e) + p->gu_bytes; }

static void make_expert_pool(expert_pool *p, int c, unsigned min_layers) {
    size_t per_layer;
    p->c = c;
    p->gu_bytes = row_bytes(c, D_MODEL) * 2 * EXP_FF;
    p->dn_bytes = row_bytes(c, EXP_FF) * D_MODEL;
    per_layer = (p->gu_bytes + p->dn_bytes) * N_EXPERT;
    p->layers = (unsigned)((POOL_MIN_BYTES + per_layer - 1) / per_layer);
    if (p->layers < min_layers) p->layers = min_layers;
    p->mem = xalloc(per_layer * p->layers);
    fill_quant(c, p->mem, per_layer * p->layers);
}

static void pick_experts(unsigned *sel) {
    unsigned n = 0, i;
    while (n < N_USED) {
        unsigned e = (unsigned)(rnd() % N_EXPERT), dup = 0;
        for (i = 0; i < n; ++i) dup |= sel[i] == e;
        if (!dup) sel[n++] = e;
    }
}

typedef struct { float *h, *y, *out; act_q *hq, *x; } exp_scratch;

static void scratch_init(exp_scratch *s) {
    s->h = xalloc(sizeof(float) * 16 * 2 * DENSE_FF);
    s->y = xalloc(sizeof(float) * 16 * D_MODEL);
    s->out = xalloc(sizeof(float) * D_MODEL);
    s->hq = xalloc(sizeof(act_q) * 16);
    s->x = xalloc(sizeof(act_q));
}

static void cell_experts(const char *cell, int c, int calls) {
    expert_pool p;
    exp_scratch s;
    exp_job j;
    float x[D_MODEL];
    double t[512];
    unsigned sel[N_USED], e, k;
    int i;
    make_expert_pool(&p, c, 1);
    scratch_init(&s);
    for (k = 0; k < D_MODEL; ++k) x[k] = frand();
    quantize_act(c, x, D_MODEL, s.x);
    memset(&j, 0, sizeof j);
    j.c = c; j.nexp = N_USED; j.cols = D_MODEL; j.ff = EXP_FF; j.x = s.x;
    j.h = s.h; j.hq = s.hq; j.y = s.y; j.out = s.out;
    for (e = 0; e < N_USED; ++e) j.w[e] = 1.0f / N_USED;
    for (i = -p.layers * 4; i < calls; ++i) {   /* negative indices: warm-up */
        unsigned l = (unsigned)((i + p.layers * 4) % p.layers);
        double t0;
        pick_experts(sel);
        for (e = 0; e < N_USED; ++e) { j.gu[e] = pool_gu(&p, l, sel[e]); j.dn[e] = pool_dn(&p, l, sel[e]); }
        t0 = now_s();
        run_experts(&j);
        if (i >= 0) t[i] = now_s() - t0;
    }
    emit(cell, CLASS_NAME[c], "8x(1024x2048+2048x512)", (double)N_USED * (p.gu_bytes + p.dn_bytes), t, calls);
    xfree(p.mem);
}

/* ------------------------------------------------------------- synthetic layer -- */
typedef struct {
    uint8_t *qkv, *o;            /* Q4_K */
    float *router, *bias;        /* F32 */
    uint8_t *dense_gu, *dense_dn;/* layer 0 only */
    uint16_t *kcache, *vcache;   /* F16 [CTX][N_KV][HEAD_DIM] */
    float qn[HEAD_DIM], kn[HEAD_DIM];
} layer_w;

#define ATT_CHUNKS 6   /* H2: the engine's six position chunks, whatever the thread count */
#define POS_PER_WORKER ((CTX + ATT_CHUNKS - 1) / ATT_CHUNKS + 8)
typedef struct {
    const float *q;              /* N_HEAD x HEAD_DIM, rotated */
    const uint16_t *kc, *vc;
    unsigned npos;
    float m[ATT_CHUNKS][N_HEAD], l[ATT_CHUNKS][N_HEAD], acc[ATT_CHUNKS][N_HEAD][HEAD_DIM];
    float sc[ATT_CHUNKS][N_HEAD][POS_PER_WORKER];
    float *out;                  /* N_HEAD x HEAD_DIM */
} att_job;

/* Positions split across workers; each K and V row is read once for the four query
 * heads of its GQA group; softmax numerators use the vector exp. */
static void att_chunk(att_job *a, int tid) {
    const unsigned p0 = a->npos * tid / ATT_CHUNKS, p1 = a->npos * (tid + 1) / ATT_CHUNKS, np = p1 - p0, G = N_HEAD / N_KV;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    unsigned kv, pos, d, h, i;
    for (kv = 0; kv < N_KV; ++kv) {
        for (pos = p0; pos < p1; ++pos) {
            const uint16_t *k = a->kc + ((size_t)pos * N_KV + kv) * HEAD_DIM;
            __m256 s8[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            for (d = 0; d < HEAD_DIM; d += 8) {
                const __m256 kk = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + d)));
                for (h = 0; h < G; ++h) s8[h] = _mm256_fmadd_ps(_mm256_loadu_ps(a->q + (kv * G + h) * HEAD_DIM + d), kk, s8[h]);
            }
            for (h = 0; h < G; ++h) a->sc[tid][kv * G + h][pos - p0] = c50_hsum_float_8(s8[h]) * scale;
        }
    }
    for (h = 0; h < N_HEAD; ++h) {
        float *s = a->sc[tid][h], m = -INFINITY;
        __m256 sum = _mm256_setzero_ps();
        for (i = 0; i < np; ++i) if (s[i] > m) m = s[i];
        for (i = np; i < ((np + 7) & ~7U); ++i) s[i] = -INFINITY;
        for (i = 0; i < np; i += 8) {
            __m256 e = c50_v_expf(_mm256_sub_ps(_mm256_loadu_ps(s + i), _mm256_set1_ps(m)));
            _mm256_storeu_ps(s + i, e);
            sum = _mm256_add_ps(sum, e);
        }
        a->m[tid][h] = m;
        a->l[tid][h] = c50_hsum_float_8(sum);
        memset(a->acc[tid][h], 0, sizeof(float) * HEAD_DIM);
    }
    for (kv = 0; kv < N_KV; ++kv) {
        for (pos = p0; pos < p1; ++pos) {
            const uint16_t *v = a->vc + ((size_t)pos * N_KV + kv) * HEAD_DIM;
            for (d = 0; d < HEAD_DIM; d += 8) {
                const __m256 vv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v + d)));
                for (h = 0; h < G; ++h) {
                    float *acc = a->acc[tid][kv * G + h] + d;
                    _mm256_storeu_ps(acc, _mm256_fmadd_ps(_mm256_set1_ps(a->sc[tid][kv * G + h][pos - p0]), vv,
                                                          _mm256_loadu_ps(acc)));
                }
            }
        }
    }
}

static void att_worker(int tid, void *p) {
    int c;
    for (c = tid; c < ATT_CHUNKS; c += g_nt) att_chunk(p, c);
}

static void att_combine(att_job *a) {
    unsigned h, t, d;
    for (h = 0; h < N_HEAD; ++h) {
        float m = -INFINITY, l = 0.0f, *o = a->out + h * HEAD_DIM;
        for (t = 0; t < ATT_CHUNKS; ++t) if (a->m[t][h] > m) m = a->m[t][h];
        memset(o, 0, sizeof(float) * HEAD_DIM);
        for (t = 0; t < ATT_CHUNKS; ++t) {
            float c = a->l[t][h] > 0 ? expf(a->m[t][h] - m) : 0.0f;
            l += a->l[t][h] * c;
            for (d = 0; d < HEAD_DIM; ++d) o[d] += a->acc[t][h][d] * c;
        }
        for (d = 0; d < HEAD_DIM; ++d) o[d] /= l;
    }
}

typedef struct { const float *w; const float *x; float *out; unsigned rows, cols; } f32_job;
static void f32_worker(int tid, void *p) {
    f32_job *j = p;
    unsigned r0 = j->rows * tid / g_nt, r1 = j->rows * (tid + 1) / g_nt, r, k;
    for (r = r0; r < r1; ++r) {
        __m256 s = _mm256_setzero_ps();
        for (k = 0; k < j->cols; k += 8)
            s = _mm256_fmadd_ps(_mm256_loadu_ps(j->w + (size_t)r * j->cols + k), _mm256_loadu_ps(j->x + k), s);
        j->out[r] = c50_hsum_float_8(s);
    }
}

static void rmsnorm(float *y, const float *x, unsigned n) {
    double ss = 0.0;
    unsigned i;
    float inv;
    for (i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    inv = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (i = 0; i < n; ++i) y[i] = x[i] * inv;
}

static float g_rope_cos[ROPE_DIM / 2], g_rope_sin[ROPE_DIM / 2];

static void rope_table(unsigned pos) {
    unsigned i;
    for (i = 0; i < ROPE_DIM / 2; ++i) {
        float th = (float)pos * powf(600000.0f, -2.0f * i / ROPE_DIM);
        g_rope_cos[i] = cosf(th);
        g_rope_sin[i] = sinf(th);
    }
}

static void rope(float *v) {
    unsigned i;
    for (i = 0; i < ROPE_DIM / 2; ++i) {
        float a = v[i], b = v[i + ROPE_DIM / 2];
        v[i] = a * g_rope_cos[i] - b * g_rope_sin[i];
        v[i + ROPE_DIM / 2] = a * g_rope_sin[i] + b * g_rope_cos[i];
    }
}

/* group-limited sigmoid top-k (noaux_tc): group score = sum of top-2 biased scores */
static void route(const float *logits, const float *bias, unsigned *sel, float *w) {
    float sc[N_EXPERT], bs[N_EXPERT], gs[N_GROUP], sum = 0.0f;
    unsigned keep[N_GROUP] = {0}, g, e, k, i;
    for (e = 0; e < N_EXPERT; ++e) { sc[e] = 1.0f / (1.0f + expf(-logits[e])); bs[e] = sc[e] + bias[e]; }
    for (g = 0; g < N_GROUP; ++g) {
        float a = -INFINITY, b = -INFINITY;
        for (e = g * (N_EXPERT / N_GROUP); e < (g + 1) * (N_EXPERT / N_GROUP); ++e) {
            if (bs[e] > a) { b = a; a = bs[e]; } else if (bs[e] > b) b = bs[e];
        }
        gs[g] = a + b;
    }
    for (k = 0; k < TOPK_GROUP; ++k) {
        unsigned best = 0; float bv = -INFINITY;
        for (g = 0; g < N_GROUP; ++g) if (!keep[g] && gs[g] > bv) { bv = gs[g]; best = g; }
        keep[best] = 1;
    }
    for (k = 0; k < N_USED; ++k) {
        unsigned best = 0; float bv = -INFINITY;
        for (e = 0; e < N_EXPERT; ++e) {
            int taken = 0;
            if (!keep[e / (N_EXPERT / N_GROUP)]) continue;
            for (i = 0; i < k; ++i) taken |= sel[i] == e;
            if (!taken && bs[e] > bv) { bv = bs[e]; best = e; }
        }
        sel[k] = best;
        w[k] = sc[best];
        sum += sc[best];
    }
    for (k = 0; k < N_USED; ++k) w[k] = w[k] / sum * ROUTED_SCALE;
}

typedef struct {
    layer_w lw[N_LAYER];
    expert_pool ep;              /* experts: N_LAYER-1 MoE layers + shared at index N_EXPERT-? */
    uint8_t *shared;             /* per layer: gu + dn */
    exp_scratch s;
    att_job *att;
    float x[D_MODEL], xn[D_MODEL], qkv[(N_HEAD + 2 * N_KV) * HEAD_DIM], ao[D_MODEL], tmp[D_MODEL];
    float logits[N_EXPERT];
    act_q *aq;
} model;

static double layer_bytes(const model *md, unsigned l) {
    double b = (double)row_bytes(C_Q4K, D_MODEL) * ((N_HEAD + 2 * N_KV) * HEAD_DIM + D_MODEL);
    b += 2.0 * CTX * N_KV * HEAD_DIM * 2;     /* KV read, F16 */
    if (l == 0) b += (double)row_bytes(C_Q4K, D_MODEL) * 2 * DENSE_FF + (double)row_bytes(C_Q4K, DENSE_FF) * D_MODEL;
    else b += 4.0 * N_EXPERT * D_MODEL + (double)(N_USED + 1) * (md->ep.gu_bytes + md->ep.dn_bytes);
    return b;
}

static void build_model(model *md) {
    unsigned l;
    size_t qkv_b = row_bytes(C_Q4K, D_MODEL) * (N_HEAD + 2 * N_KV) * HEAD_DIM, o_b = row_bytes(C_Q4K, D_MODEL) * D_MODEL;
    memset(md, 0, sizeof *md);
    make_expert_pool(&md->ep, C_Q4K, N_LAYER - 1);
    md->shared = xalloc((md->ep.gu_bytes + md->ep.dn_bytes) * N_LAYER);
    fill_quant(C_Q4K, md->shared, (md->ep.gu_bytes + md->ep.dn_bytes) * N_LAYER);
    for (l = 0; l < N_LAYER; ++l) {
        layer_w *w = &md->lw[l];
        size_t kvn = (size_t)CTX * N_KV * HEAD_DIM, i;
        w->qkv = xalloc(qkv_b); fill_quant(C_Q4K, w->qkv, qkv_b);
        w->o = xalloc(o_b); fill_quant(C_Q4K, w->o, o_b);
        w->router = xalloc(sizeof(float) * N_EXPERT * D_MODEL);
        w->bias = xalloc(sizeof(float) * N_EXPERT);
        for (i = 0; i < (size_t)N_EXPERT * D_MODEL; ++i) w->router[i] = frand() * 0.02f;
        for (i = 0; i < N_EXPERT; ++i) w->bias[i] = frand() * 0.01f;
        w->kcache = xalloc(2 * kvn); w->vcache = xalloc(2 * kvn);
        for (i = 0; i < kvn; ++i) {
            w->kcache[i] = (uint16_t)(0x2C00U | (rnd() & 0x3FFU) | ((rnd() & 1) << 15));
            w->vcache[i] = (uint16_t)(0x2C00U | (rnd() & 0x3FFU) | ((rnd() & 1) << 15));
        }
        for (i = 0; i < HEAD_DIM; ++i) { w->qn[i] = 1.0f; w->kn[i] = 1.0f; }
        if (l == 0) {
            size_t gb = row_bytes(C_Q4K, D_MODEL) * 2 * DENSE_FF, db = row_bytes(C_Q4K, DENSE_FF) * D_MODEL;
            w->dense_gu = xalloc(gb); fill_quant(C_Q4K, w->dense_gu, gb);
            w->dense_dn = xalloc(db); fill_quant(C_Q4K, w->dense_dn, db);
        }
    }
    scratch_init(&md->s);
    md->att = xalloc(sizeof(att_job));
    md->aq = xalloc(sizeof(act_q));
    for (l = 0; l < D_MODEL; ++l) md->x[l] = frand();
    rope_table(CTX - 1);   /* one decode position; an engine caches this per position */
}

/* per-organ wall time accumulated by forward_layer (seconds) */
enum { PH_NORM_Q, PH_QKV, PH_QKROPE, PH_ATT, PH_ATT_COMB, PH_O, PH_ROUTER, PH_ROUTE, PH_GU, PH_ACT, PH_DOWN, PH_COMB, PH_COUNT };
static const char *PH_NAME[PH_COUNT] = {"norm_quant", "qkv", "qknorm_rope_kvstore", "attention", "attention_combine",
                                        "o_proj_residual", "router_matvec", "route_select", "experts_gate_up",
                                        "experts_swiglu_quant", "experts_down", "experts_combine"};
static double g_ph[PH_COUNT];
static double g_ph_mark;
static void ph(int k) { double t = now_s(); g_ph[k] += t - g_ph_mark; g_ph_mark = t; }

static void forward_layer(model *md, unsigned l, unsigned pool_layer) {
    layer_w *w = &md->lw[l];
    mv_job m;
    exp_job j;
    unsigned h, i, sel[N_USED];
    float wts[N_USED];
    memset(&m, 0, sizeof m);
    g_ph_mark = now_s();
    /* attention */
    rmsnorm(md->xn, md->x, D_MODEL);
    quantize_act(C_Q4K, md->xn, D_MODEL, md->aq);
    ph(PH_NORM_Q);
    m.c = C_Q4K; m.cols = D_MODEL; m.rows_per_seg = (N_HEAD + 2 * N_KV) * HEAD_DIM; m.nseg = 1; m.shared_act = 1;
    m.base[0] = w->qkv; m.act[0] = md->aq; m.out = md->qkv;
    run(mv_worker, &m);
    ph(PH_QKV);
    for (h = 0; h < N_HEAD + N_KV; ++h) {
        float *v = md->qkv + h * HEAD_DIM;
        rmsnorm(v, v, HEAD_DIM);
        rope(v);
    }
    for (i = 0; i < N_KV * HEAD_DIM; ++i) {    /* append current K/V at the last position */
        w->kcache[(size_t)(CTX - 1) * N_KV * HEAD_DIM + i] = _cvtss_sh(md->qkv[N_HEAD * HEAD_DIM + i], 0);
        w->vcache[(size_t)(CTX - 1) * N_KV * HEAD_DIM + i] = _cvtss_sh(md->qkv[(N_HEAD + N_KV) * HEAD_DIM + i], 0);
    }
    ph(PH_QKROPE);
    md->att->q = md->qkv; md->att->kc = w->kcache; md->att->vc = w->vcache; md->att->npos = CTX; md->att->out = md->ao;
    run(att_worker, md->att);
    ph(PH_ATT);
    att_combine(md->att);
    quantize_act(C_Q4K, md->ao, D_MODEL, md->aq);
    ph(PH_ATT_COMB);
    m.rows_per_seg = D_MODEL; m.base[0] = w->o; m.out = md->tmp;
    run(mv_worker, &m);
    for (i = 0; i < D_MODEL; ++i) md->x[i] += md->tmp[i];
    ph(PH_O);
    /* FFN */
    rmsnorm(md->xn, md->x, D_MODEL);
    quantize_act(C_Q4K, md->xn, D_MODEL, md->s.x);
    ph(PH_NORM_Q);
    memset(&j, 0, sizeof j);
    j.c = C_Q4K; j.cols = D_MODEL; j.x = md->s.x; j.h = md->s.h; j.hq = md->s.hq; j.y = md->s.y; j.out = md->x;
    if (l == 0) {
        j.nexp = 1; j.ff = DENSE_FF; j.gu[0] = w->dense_gu; j.dn[0] = w->dense_dn; j.w[0] = 1.0f;
    } else {
        f32_job r = {w->router, md->xn, md->logits, N_EXPERT, D_MODEL};
        run(f32_worker, &r);
        ph(PH_ROUTER);
        route(md->logits, w->bias, sel, wts);
        j.nexp = N_USED + 1; j.ff = EXP_FF;
        for (i = 0; i < N_USED; ++i) {
            j.gu[i] = pool_gu(&md->ep, pool_layer, sel[i]); j.dn[i] = pool_dn(&md->ep, pool_layer, sel[i]); j.w[i] = wts[i];
        }
        j.gu[N_USED] = md->shared + (size_t)l * (md->ep.gu_bytes + md->ep.dn_bytes);
        j.dn[N_USED] = j.gu[N_USED] + md->ep.gu_bytes;
        j.w[N_USED] = 1.0f;
        ph(PH_ROUTE);
    }
    g_phase_hook = ph;
    run_experts(&j);
    g_phase_hook = NULL;
    /* keep the synthetic residual stream bounded */
    rmsnorm(md->x, md->x, D_MODEL);
    ph(PH_NORM_Q);
}

/* mean seconds per call of each organ, warm-up calls included in both sums and count */
static void emit_phases(const char *cell, int ncalls) {
    int k;
    printf("{\"cell\":\"%s_phases\",\"mean_s_per_call\":{", cell);
    for (k = 0; k < PH_COUNT; ++k) printf("%s\"%s\":%.9f", k ? "," : "", PH_NAME[k], g_ph[k] / ncalls);
    printf("}}\n");
    fflush(stdout);
}

static void cell_layer_and_token(model *md, int l1_calls, int t20_calls, int which) {
    double t[512];
    int i;
    unsigned l;
    memset(g_ph, 0, sizeof g_ph);
    if (which == 0) {
        double bytes = layer_bytes(md, 1);
        for (i = -(int)(N_LAYER - 1); i < l1_calls; ++i) {   /* warm-up: one pass over the MoE layers */
            unsigned layer = 1 + (unsigned)((i + (int)(N_LAYER - 1)) % (N_LAYER - 1));
            double t0 = now_s();
            forward_layer(md, layer, layer - 1);
            if (i >= 0) t[i] = now_s() - t0;
        }
        emit("L1", "P0", "moe_layer_ctx2048", bytes, t, l1_calls);
        emit_phases("L1", l1_calls + (int)(N_LAYER - 1));
    } else {
        double bytes = 0.0;
        for (l = 0; l < N_LAYER; ++l) bytes += layer_bytes(md, l);
        for (i = -2; i < t20_calls; ++i) {
            double t0 = now_s();
            for (l = 0; l < N_LAYER; ++l) forward_layer(md, l, l ? l - 1 : 0);
            if (i >= 0) t[i] = now_s() - t0;
        }
        emit("T20", "P0", "20_layers_no_head_ctx2048", bytes, t, t20_calls);
        emit_phases("T20", t20_calls + 2);
    }
}

/* ------------------------------------------------------------------- selftest -- */
/* Scale of a dot for float-reordering error: sum over quant blocks of |block dot|,
 * each block dot computed by the generic oracle on that block alone. */
static double block_abs_scale(int c, const uint8_t *row, const act_q *a, unsigned cols) {
    unsigned be = BLOCK_ELEMS[c], b;
    double s = 0.0;
    for (b = 0; b < cols / be; ++b) {
        act_q one;
        if (c == C_Q80) one.z[0] = a->z[b];
        else one.k[0] = a->k[b];
        s += fabs((double)dot_row_generic(c, row + (size_t)b * BLOCK_BYTES[c], &one, be));
    }
    return s;
}

/* AMENDED before any timed run (see the A1 result record): the error of an AVX2 dot
 * against its generic oracle is normalised by block_abs_scale, not by |result|, and a
 * planted corruption (one quant byte flipped in the oracle's copy) must be detected in at
 * least 190 of 200 trials at the same 1e-5 threshold. */
static int selftest(void) {
    int c, trial, fails = 0;
    for (c = 0; c < C_COUNT; ++c) {
        double worst = 0.0, planted_min = INFINITY;
        int detected = 0;
        for (trial = 0; trial < 200; ++trial) {
            unsigned cols = (trial % 2) ? 2048 : 512, k;
            size_t rb = row_bytes(c, cols), flip;
            uint8_t *row = malloc(rb), *bad = malloc(rb);
            float x[2048];
            act_q *a = malloc(sizeof(act_q));
            double r, g, scale, rel, prel;
            fill_quant(c, row, rb);
            for (k = 0; k < cols; ++k) x[k] = frand() * (1.0f + (float)(rnd() % 4));
            quantize_act(c, x, cols, a);
            r = dot_row_generic(c, row, a, cols);
            g = dot_row(c, row, a, cols);
            scale = block_abs_scale(c, row, a, cols);
            rel = fabs(g - r) / scale;
            /* planted control: flip the high bit of one non-scale byte of one block */
            memcpy(bad, row, rb);
            do flip = (size_t)(rnd() % rb); while (flip % BLOCK_BYTES[c] >= SCALE_OFFSET[c] &&
                                                   flip % BLOCK_BYTES[c] < SCALE_OFFSET[c] + (c == C_Q4K ? 4U : 2U));
            bad[flip] ^= 0x80;
            prel = fabs((double)dot_row_generic(c, bad, a, cols) - g) / scale;
            if (!isfinite(r) || !isfinite(g) || rel > 1e-5) {
                ++fails;
                printf("SELFTEST FAIL class=%s trial=%d ref=%.9g avx2=%.9g err/scale=%.3g\n", CLASS_NAME[c], trial, r, g, rel);
            }
            if (rel > worst) worst = rel;
            if (prel < planted_min) planted_min = prel;
            detected += prel > 1e-5;
            free(row); free(bad); free(a);
        }
        /* a single flipped bit can be invisible (multiplied by a zero quant, scale or activation) */
        if (detected < 190) {
            ++fails;
            printf("SELFTEST FAIL class=%s planted corruption detected in %d/200\n", CLASS_NAME[c], detected);
        }
        printf("SELFTEST class=%s trials=200 worst_err_over_scale=%.3g planted_detected=%d/200 "
               "planted_min_err_over_scale=%.3g\n", CLASS_NAME[c], worst, detected, planted_min);
    }
    printf("SELFTEST %s fails=%d\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

/* ----------------------------------------------------------------------- main -- */
int main(int argc, char **argv) {
    int order = 0, quick = 0, do_run = 0, i, n_cells, k;
    uint64_t seed = 1;
    const int calls = 200, calls_head = 40, calls_l1 = 200, calls_t20 = 40;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--selftest")) return selftest();
        else if (!strcmp(argv[i], "--run")) do_run = 1;
        else if (!strcmp(argv[i], "--order") && i + 1 < argc) order = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            g_nt = atoi(argv[++i]);
            if (g_nt < 1 || g_nt > NT_MAX) { fprintf(stderr, "--threads must be 1..%d\n", NT_MAX); return 2; }
        }
    }
    if (!do_run) { fprintf(stderr, "usage: --selftest | --run --order K [--seed S] [--quick] [--threads N]\n"); return 2; }
    g_rng ^= seed * 0xD1B54A32D192ED03ULL;
    start_pool();
    {
        int div = quick ? 10 : 1;
        static const char *names[] = {
            "S1_Q4_K", "S1_Q8_0", "S1_IQ4_XS", "S2_Q4_K", "S2_Q8_0", "S2_IQ4_XS",
            "S3_Q4_K", "S3_IQ4_XS", "S3_IQ3_S", "S4_Q6_K", "S4_Q4_K", "S5_Q4_K", "L1", "T20"};
        model *md = NULL;
        n_cells = (int)(sizeof names / sizeof *names);
        printf("CONFIG cell=C50-A1 threads=%d cpus=%s pool_min_bytes=%llu calls=%d calls_head=%d "
               "calls_l1=%d calls_t20=%d order=%d seed=%llu quick=%d ctx=%u kernels=Q4_K,Q6_K,IQ4_XS,IQ3_S,Q8_0 "
               "cells=%d\n", g_nt, g_cores,
               (unsigned long long)POOL_MIN_BYTES, calls / div, calls_head / div, calls_l1 / div, calls_t20 / div,
               order, (unsigned long long)seed, quick, CTX, n_cells);
        fflush(stdout);
        for (k = 0; k < n_cells; ++k) {
            const char *c = names[(k + order) % n_cells];
            if (!strcmp(c, "S1_Q4_K")) cell_matvec(c, C_Q4K, 3072, 2048, calls / div);
            else if (!strcmp(c, "S1_Q8_0")) cell_matvec(c, C_Q80, 3072, 2048, calls / div);
            else if (!strcmp(c, "S1_IQ4_XS")) cell_matvec(c, C_IQ4XS, 3072, 2048, calls / div);
            else if (!strcmp(c, "S2_Q4_K")) cell_matvec(c, C_Q4K, 2048, 2048, calls / div);
            else if (!strcmp(c, "S2_Q8_0")) cell_matvec(c, C_Q80, 2048, 2048, calls / div);
            else if (!strcmp(c, "S2_IQ4_XS")) cell_matvec(c, C_IQ4XS, 2048, 2048, calls / div);
            else if (!strcmp(c, "S3_Q4_K")) cell_experts(c, C_Q4K, calls / div);
            else if (!strcmp(c, "S3_IQ4_XS")) cell_experts(c, C_IQ4XS, calls / div);
            else if (!strcmp(c, "S3_IQ3_S")) cell_experts(c, C_IQ3S, calls / div);
            else if (!strcmp(c, "S4_Q6_K")) cell_matvec(c, C_Q6K, VOCAB, 2048, calls_head / div);
            else if (!strcmp(c, "S4_Q4_K")) cell_matvec(c, C_Q4K, VOCAB, 2048, calls_head / div);
            else if (!strcmp(c, "S5_Q4_K")) {
                /* dense FFN: gate+up [10240 x 2048] then down [2048 x 5120], as one expert of ff=5120 */
                expert_pool dummy;
                exp_scratch s;
                exp_job j;
                size_t gb = row_bytes(C_Q4K, D_MODEL) * 2 * DENSE_FF, db = row_bytes(C_Q4K, DENSE_FF) * D_MODEL;
                unsigned copies = (unsigned)((POOL_MIN_BYTES + gb + db - 1) / (gb + db)), q;
                uint8_t *pool = xalloc((gb + db) * copies);
                float x[D_MODEL];
                double t[512];
                int nc = calls / div, ii;
                (void)dummy;
                fill_quant(C_Q4K, pool, (gb + db) * copies);
                scratch_init(&s);
                for (q = 0; q < D_MODEL; ++q) x[q] = frand();
                quantize_act(C_Q4K, x, D_MODEL, s.x);
                memset(&j, 0, sizeof j);
                j.c = C_Q4K; j.nexp = 1; j.cols = D_MODEL; j.ff = DENSE_FF; j.x = s.x; j.h = s.h; j.hq = s.hq;
                j.y = s.y; j.out = s.out; j.w[0] = 1.0f;
                for (ii = -(int)copies; ii < nc; ++ii) {
                    unsigned cp = (unsigned)((ii + (int)copies) % copies);
                    double t0;
                    j.gu[0] = pool + (size_t)cp * (gb + db); j.dn[0] = j.gu[0] + gb;
                    t0 = now_s();
                    run_experts(&j);
                    if (ii >= 0) t[ii] = now_s() - t0;
                }
                emit(c, "Q4_K", "10240x2048+2048x5120", (double)(gb + db), t, nc);
                xfree(pool);
            } else if (!strcmp(c, "L1") || !strcmp(c, "T20")) {
                if (!md) { md = xalloc(sizeof(model)); build_model(md); }
                cell_layer_and_token(md, calls_l1 / div, calls_t20 / div, !strcmp(c, "T20"));
            }
        }
    }
    g_fn = NULL;
    InterlockedIncrement(&g_gen);
    printf("DONE\n");
    return 0;
}
