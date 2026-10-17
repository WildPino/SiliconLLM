/*
 * C50 engine: batch-1 decode of Ling-mini-2.0 (GGUF arch "bailingmoe2") on the project's
 * own C kernels.  Roadmap cell A3: the complete P0 path, judged statistically against
 * pinned llama.cpp on the same GGUF (protocol in docs/research/CHATBOT50_ROADMAP_20261016/).
 *
 * Semantics follow the pinned source model code (modeling_bailing_moe_v2.py @ a810f64) and
 * llama.cpp 5b335f4 src/models/bailingmoe2.cpp, which agree:
 *   x = embd[tok]
 *   per layer: h = rmsnorm(x) * attn_norm; [q|k|v] = Wqkv h; q,k: per-head rmsnorm * q/k_norm,
 *     NEOX RoPE on the first 64 of 128 dims (theta 600000); GQA 16/4, scale 1/sqrt(128);
 *     x += Wo attn;  h = rmsnorm(x) * ffn_norm;
 *     layer 0: x += down(silu(gate h) * up h)
 *     else: s = sigmoid(Wr h) (F32); selection on s + expert_bias: group score = sum of the
 *       top-2 biased scores of each of 8 groups, keep 4 groups, top-8 experts inside them;
 *       weights = unbiased s of the chosen experts / their sum * 2.5;
 *       x += sum_e w_e down_e(silu(gate_e h) * up_e h) + shared(h)
 *   logits = Wout (rmsnorm(x) * output_norm)
 * Matrix x vector products quantize the activation the way llama.cpp's CPU backend does
 * (Q8_K for k-quants, Q8_0 for Q8_0) and use AVX2 dot kernels transcribed from it.
 * Attention, norms, router and RoPE run in F32; the KV cache is F16.
 *
 * Modes:
 *   --selftest                                      kernel and dequantizer unit test (no model)
 *   --bench IDS.tsv [--ctx C] [--steps N]           decode-rate bench (needs a clean box)
 *   --score IDS.tsv --out PREFIX [--dump-logprobs]   teacher-forced scoring, output format
 *                                                    identical to c50_llama_score.cpp
 * Build:
 *   clang -O2 -mavx2 -mfma -mf16c -o bin/c50_engine.exe c50_engine.c
 */
#define _WIN32_WINNT 0x0A00
#include <windows.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "c50_kquants_a3.h"
#include "c50_gguf.h"

#define NT 6
#define D 2048U
#define NH 16U
#define NKV 4U
#define HD 128U
#define ROT 64U
#define NE 256U
#define NU 8U
#define NG 8U
#define TG 4U
#define FFE 512U
#define FFD 5120U
#define NL 20U
#define VOCAB 157184U
#define MAXCTX 8192U
#define QKV_ROWS ((NH + 2 * NKV) * HD)
#define MAXSEG 20

/* ------------------------------------------------------------------ weights -- */
typedef struct { uint32_t type; unsigned cols; uint64_t rows; size_t rb; const uint8_t *data; } wt;

static wt wt_get(const gguf *g, const char *name, unsigned cols) {
    const gg_tensor *t = gguf_tensor(g, name);
    wt w;
    w.type = t->type;
    w.cols = (unsigned)t->dims[0];
    w.rows = t->dims[1] * t->dims[2] * t->dims[3];
    w.rb = gg_type_row_bytes(t->type, t->dims[0]);
    w.data = t->data;
    if (w.cols != cols || !w.rb) { fprintf(stderr, "tensor %s: cols %u (want %u), type %s\n", name, w.cols, cols, gg_type_name(t->type)); exit(2); }
    return w;
}

static const float *f32_get(const gguf *g, const char *name, uint64_t n) {
    const gg_tensor *t = gguf_tensor(g, name);
    if (t->type != GT_F32 || t->dims[0] * t->dims[1] != n) { fprintf(stderr, "tensor %s: not F32[%llu]\n", name, (unsigned long long)n); exit(2); }
    return (const float *)t->data;
}

typedef struct {
    wt qkv, o, gate, up, down;               /* gate/up/down: dense (layer 0) or routed experts */
    wt sgate, sup, sdown;                    /* shared expert */
    const float *attn_norm, *ffn_norm, *qn, *kn, *router, *bias;
} layer_t;

typedef struct { gguf g; wt embd, out; const float *out_norm; layer_t l[NL]; } model_t;

/* ------------------------------------------------------------- activations -- */
typedef struct {
    strat01_q8_k_block k[FFD / 256];
    c50_block_q8_0 z[FFD / 32];
    float f[FFD];
    int has_k, has_z;
} act_t;

static int needs_q80(uint32_t type) { return type == GT_Q8_0; }

/* make the representation `type` consumes available in a (from a->f) */
static void act_prepare(act_t *a, uint32_t type, unsigned n) {
    if (type == GT_F32) return;
    if (needs_q80(type)) {
        if (!a->has_z) { c50_quantize_row_q8_0(a->f, a->z, n); a->has_z = 1; }
    } else if (!a->has_k) {
        if (!strat01_quantize_q8_k_row(a->f, n, a->k)) { fprintf(stderr, "q8_K quantization failed\n"); exit(2); }
        a->has_k = 1;
    }
}

static void act_set(act_t *a, const float *x, unsigned n) {
    if (x != a->f) memcpy(a->f, x, sizeof(float) * n);
    a->has_k = a->has_z = 0;
}

static float dot_row(uint32_t type, const uint8_t *row, const act_t *a, unsigned cols) {
    switch (type) {
    case GT_Q4_K: return strat01_q4k_q8k_dot_avx2(row, a->k, cols);
    case GT_Q5_K: return c50_q5_k_dot_avx2((const c50_block_q5_k *)row, a->k, cols);
    case GT_Q6_K: return strat01_q6k_q8k_dot_avx2(row, a->k, cols);
    case GT_IQ4_XS: return c50_iq4_xs_dot_avx2((const c50_block_iq4_xs *)row, a->k, cols);
    case GT_Q8_0: return c50_q8_0_dot_avx2((const c50_block_q8_0 *)row, a->z, cols);
    case GT_F32: {
        const float *w = (const float *)row;
        __m256 s = _mm256_setzero_ps();
        unsigned k;
        for (k = 0; k < cols; k += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(a->f + k), s);
        return c50_hsum_float_8(s);
    }
    default: fprintf(stderr, "unsupported weight type %u\n", type); exit(2);
    }
}

static void dequant_row(const wt *w, uint64_t r, float *y) {
    const uint8_t *row = w->data + r * w->rb;
    switch (w->type) {
    case GT_Q4_K: c50_dequant_q4_k(row, y, w->cols); break;
    case GT_Q5_K: c50_dequant_q5_k(row, y, w->cols); break;
    case GT_Q6_K: c50_dequant_q6_k(row, y, w->cols); break;
    case GT_Q8_0: c50_dequant_q8_0(row, y, w->cols); break;
    case GT_F32: memcpy(y, row, sizeof(float) * w->cols); break;
    default: fprintf(stderr, "unsupported embedding type %u\n", w->type); exit(2);
    }
}

/* ------------------------------------------------------------- thread pool -- */
typedef void (*job_fn)(int tid, void *arg);
static volatile LONG g_gen, g_done;
static job_fn volatile g_fn;
static void *volatile g_arg;
static DWORD_PTR g_mask[NT];
static int g_cpu[NT];

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
    while (g_done < NT - 1) _mm_pause();
}

static void pick_cores(void) {
    DWORD len = 0;
    char *buf, *ptr;
    int n = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    buf = malloc(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        fprintf(stderr, "topology query failed\n"); exit(2);
    }
    for (ptr = buf; ptr < buf + len && n < NT;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
        KAFFINITY m = info->Processor.GroupMask[0].Mask;
        int bit = 0;
        while (!((m >> bit) & 1)) ++bit;
        g_mask[n] = (DWORD_PTR)1 << bit;
        g_cpu[n] = bit;
        ++n;
        ptr += info->Size;
    }
    free(buf);
    if (n < NT) { fprintf(stderr, "fewer than %d physical cores\n", NT); exit(2); }
}

static void start_pool(void) {
    int t;
    pick_cores();
    SetThreadAffinityMask(GetCurrentThread(), g_mask[0]);
    for (t = 1; t < NT; ++t) CreateThread(NULL, 0, worker_main, (LPVOID)(intptr_t)t, 0, NULL);
}

static double now_s(void) {
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* -------------------------------------------------------------- matvec job -- */
/* nseg segments of rows_per_seg rows; segment s = rows [r0[s], r0[s]+rows_per_seg) of w[s],
 * multiplied by act[s]; output row (s * rows_per_seg + i). */
typedef struct {
    unsigned rows_per_seg, nseg, cols;
    const wt *w[MAXSEG];
    uint64_t r0[MAXSEG];
    const act_t *act[MAXSEG];
    float *out;
} mv_job;

static void mv_worker(int tid, void *p) {
    mv_job *j = p;
    unsigned total = j->rows_per_seg * j->nseg;
    unsigned a = (unsigned)((uint64_t)total * tid / NT), b = (unsigned)((uint64_t)total * (tid + 1) / NT), r;
    for (r = a; r < b; ++r) {
        unsigned s = r / j->rows_per_seg, lr = r % j->rows_per_seg;
        const wt *w = j->w[s];
        j->out[r] = dot_row(w->type, w->data + (j->r0[s] + lr) * w->rb, j->act[s], j->cols);
    }
}

static void matvec(const wt *w, act_t *a, float *out, uint64_t rows) {
    mv_job m;
    memset(&m, 0, sizeof m);
    act_prepare(a, w->type, w->cols);
    m.rows_per_seg = (unsigned)rows; m.nseg = 1; m.cols = w->cols; m.w[0] = w; m.r0[0] = 0; m.act[0] = a; m.out = out;
    run(mv_worker, &m);
}

/* ---------------------------------------------------------------- attention -- */
#define POS_PER_WORKER ((MAXCTX + NT - 1) / NT + 8)
typedef struct {
    const float *q;              /* NH x HD, normed and rotated */
    const uint16_t *kc, *vc;     /* [MAXCTX][NKV][HD] F16 */
    unsigned npos;
    float m[NT][NH], l[NT][NH], acc[NT][NH][HD];
    float sc[NT][NH][POS_PER_WORKER];
    float *out;                  /* NH x HD */
} att_job;

static void att_worker(int tid, void *p) {
    att_job *a = p;
    const unsigned p0 = a->npos * tid / NT, p1 = a->npos * (tid + 1) / NT, np = p1 - p0, G = NH / NKV;
    const float scale = 1.0f / sqrtf((float)HD);
    unsigned kv, pos, d, h, i;
    for (kv = 0; kv < NKV; ++kv) {
        for (pos = p0; pos < p1; ++pos) {
            const uint16_t *k = a->kc + ((size_t)pos * NKV + kv) * HD;
            __m256 s8[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            for (d = 0; d < HD; d += 8) {
                const __m256 kk = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + d)));
                for (h = 0; h < G; ++h) s8[h] = _mm256_fmadd_ps(_mm256_loadu_ps(a->q + (kv * G + h) * HD + d), kk, s8[h]);
            }
            for (h = 0; h < G; ++h) a->sc[tid][kv * G + h][pos - p0] = c50_hsum_float_8(s8[h]) * scale;
        }
    }
    for (h = 0; h < NH; ++h) {
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
        a->l[tid][h] = np ? c50_hsum_float_8(sum) : 0.0f;
        memset(a->acc[tid][h], 0, sizeof(float) * HD);
    }
    for (kv = 0; kv < NKV; ++kv) {
        for (pos = p0; pos < p1; ++pos) {
            const uint16_t *v = a->vc + ((size_t)pos * NKV + kv) * HD;
            for (d = 0; d < HD; d += 8) {
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

static void att_combine(att_job *a) {
    unsigned h, t, d;
    for (h = 0; h < NH; ++h) {
        float m = -INFINITY, l = 0.0f, *o = a->out + h * HD;
        for (t = 0; t < NT; ++t) if (a->l[t][h] > 0 && a->m[t][h] > m) m = a->m[t][h];
        memset(o, 0, sizeof(float) * HD);
        for (t = 0; t < NT; ++t) {
            float c = a->l[t][h] > 0 ? expf(a->m[t][h] - m) : 0.0f;
            l += a->l[t][h] * c;
            for (d = 0; d < HD; ++d) o[d] += a->acc[t][h][d] * c;
        }
        for (d = 0; d < HD; ++d) o[d] /= l;
    }
}

/* ------------------------------------------------------------ small kernels -- */
/* ggml_compute_forward_rms_norm_f32: sum of squares in double, scale = 1/sqrtf(mean + eps) */
static void rmsnorm_w(float *y, const float *x, const float *w, unsigned n, float eps) {
    double ss = 0.0;
    unsigned i;
    float mean, scale;
    for (i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    mean = (float)(ss / n);
    scale = 1.0f / sqrtf(mean + eps);
    for (i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
}

/* NEOX RoPE on the first ROT dims; angles as ggml rope_cache_init (theta *= theta_scale) */
static float g_cos[ROT / 2], g_sin[ROT / 2];
static void rope_table(unsigned pos, float base) {
    const float theta_scale = powf(base, -2.0f / (float)ROT);
    float theta = (float)pos;
    unsigned i;
    for (i = 0; i < ROT / 2; ++i) { g_cos[i] = cosf(theta); g_sin[i] = sinf(theta); theta *= theta_scale; }
}
static void rope(float *v) {
    unsigned i;
    for (i = 0; i < ROT / 2; ++i) {
        const float x0 = v[i], x1 = v[i + ROT / 2];
        v[i] = x0 * g_cos[i] - x1 * g_sin[i];
        v[i + ROT / 2] = x0 * g_sin[i] + x1 * g_cos[i];
    }
}

static void route(const float *logits, const float *bias, unsigned *sel, float *w, float scale) {
    float sc[NE], bs[NE], gs[NG], sum = 0.0f;
    unsigned keep[NG] = {0}, g, e, k, i;
    for (e = 0; e < NE; ++e) { sc[e] = 1.0f / (1.0f + expf(-logits[e])); bs[e] = sc[e] + bias[e]; }
    for (g = 0; g < NG; ++g) {
        float a = -INFINITY, b = -INFINITY;
        for (e = g * (NE / NG); e < (g + 1) * (NE / NG); ++e) {
            if (bs[e] > a) { b = a; a = bs[e]; } else if (bs[e] > b) b = bs[e];
        }
        gs[g] = a + b;
    }
    for (k = 0; k < TG; ++k) {
        unsigned best = 0; float bv = -INFINITY;
        for (g = 0; g < NG; ++g) if (!keep[g] && gs[g] > bv) { bv = gs[g]; best = g; }
        keep[best] = 1;
    }
    for (k = 0; k < NU; ++k) {
        unsigned best = 0; float bv = -INFINITY;
        for (e = 0; e < NE; ++e) {
            int taken = 0;
            if (!keep[e / (NE / NG)]) continue;
            for (i = 0; i < k; ++i) taken |= sel[i] == e;
            if (!taken && bs[e] > bv) { bv = bs[e]; best = e; }
        }
        sel[k] = best;
        w[k] = sc[best];
        sum += sc[best];
    }
    for (k = 0; k < NU; ++k) w[k] = w[k] / sum * scale;
}

/* ------------------------------------------------------------------ state -- */
typedef struct {
    float x[D], h[D], qkv[QKV_ROWS], ao[D], tmp[D], logits_r[NE];
    float gu[MAXSEG * FFE > 2 * FFD ? MAXSEG * FFE : 2 * FFD];   /* gate/up outputs */
    float y[(NU + 1) * D];                                       /* expert down outputs */
    act_t xa, ea[NU + 1];
    att_job att;
    uint16_t *kc, *vc;            /* [NL][MAXCTX][NKV][HD] */
    float *logits;                /* VOCAB */
    float eps, rope_base, wscale;
} state_t;

typedef struct { float *gu; act_t *ea; unsigned ff, n; } swiglu_job;
static void swiglu_worker(int tid, void *p) {
    swiglu_job *j = p;
    unsigned e, i;
    for (e = tid; e < j->n; e += NT) {
        const float *g = j->gu + (size_t)e * 2 * j->ff, *u = g + j->ff;
        float *o = j->ea[e].f;
        for (i = 0; i < j->ff; i += 8)
            _mm256_storeu_ps(o + i, _mm256_mul_ps(c50_v_silu(_mm256_loadu_ps(g + i)), _mm256_loadu_ps(u + i)));
        j->ea[e].has_k = j->ea[e].has_z = 0;
    }
}

static void forward(const model_t *md, state_t *st, int token, unsigned pos) {
    unsigned l, h, i, e;
    float *x = st->x;
    dequant_row(&md->embd, (uint64_t)token, x);
    rope_table(pos, st->rope_base);
    for (l = 0; l < NL; ++l) {
        const layer_t *L = &md->l[l];
        uint16_t *kc = st->kc + (size_t)l * MAXCTX * NKV * HD, *vc = st->vc + (size_t)l * MAXCTX * NKV * HD;
        mv_job m;
        swiglu_job sj;
        /* attention */
        rmsnorm_w(st->h, x, L->attn_norm, D, st->eps);
        act_set(&st->xa, st->h, D);
        matvec(&L->qkv, &st->xa, st->qkv, QKV_ROWS);
        for (h = 0; h < NH + NKV; ++h) {
            float *v = st->qkv + h * HD;
            rmsnorm_w(v, v, h < NH ? L->qn : L->kn, HD, st->eps);
            rope(v);
        }
        for (i = 0; i < NKV * HD; ++i) {
            kc[(size_t)pos * NKV * HD + i] = _cvtss_sh(st->qkv[NH * HD + i], 0);
            vc[(size_t)pos * NKV * HD + i] = _cvtss_sh(st->qkv[(NH + NKV) * HD + i], 0);
        }
        st->att.q = st->qkv; st->att.kc = kc; st->att.vc = vc; st->att.npos = pos + 1; st->att.out = st->ao;
        run(att_worker, &st->att);
        att_combine(&st->att);
        act_set(&st->xa, st->ao, D);
        matvec(&L->o, &st->xa, st->tmp, D);
        for (i = 0; i < D; ++i) x[i] += st->tmp[i];
        /* FFN */
        rmsnorm_w(st->h, x, L->ffn_norm, D, st->eps);
        act_set(&st->xa, st->h, D);
        memset(&m, 0, sizeof m);
        if (l == 0) {
            act_prepare(&st->xa, L->gate.type, D);
            act_prepare(&st->xa, L->up.type, D);
            m.rows_per_seg = FFD; m.nseg = 2; m.cols = D; m.out = st->gu;
            m.w[0] = &L->gate; m.w[1] = &L->up; m.act[0] = m.act[1] = &st->xa;
            run(mv_worker, &m);
            sj.gu = st->gu; sj.ea = st->ea; sj.ff = FFD; sj.n = 1;
            run(swiglu_worker, &sj);
            act_prepare(&st->ea[0], L->down.type, FFD);
            memset(&m, 0, sizeof m);
            m.rows_per_seg = D; m.nseg = 1; m.cols = FFD; m.w[0] = &L->down; m.act[0] = &st->ea[0]; m.out = st->tmp;
            run(mv_worker, &m);
            for (i = 0; i < D; ++i) x[i] += st->tmp[i];
        } else {
            unsigned sel[NU];
            float w[NU];
            matvec(&(wt){GT_F32, D, NE, 4 * D, (const uint8_t *)L->router}, &st->xa, st->logits_r, NE);
            route(st->logits_r, L->bias, sel, w, st->wscale);
            act_prepare(&st->xa, L->gate.type, D);
            act_prepare(&st->xa, L->up.type, D);
            act_prepare(&st->xa, L->sgate.type, D);
            act_prepare(&st->xa, L->sup.type, D);
            /* gate/up: segment 2e = gate of expert e, 2e+1 = up; 2NU, 2NU+1 = shared */
            m.rows_per_seg = FFE; m.nseg = 2 * (NU + 1); m.cols = D; m.out = st->gu;
            for (e = 0; e < NU; ++e) {
                m.w[2 * e] = &L->gate; m.r0[2 * e] = (uint64_t)sel[e] * FFE; m.act[2 * e] = &st->xa;
                m.w[2 * e + 1] = &L->up; m.r0[2 * e + 1] = (uint64_t)sel[e] * FFE; m.act[2 * e + 1] = &st->xa;
            }
            m.w[2 * NU] = &L->sgate; m.w[2 * NU + 1] = &L->sup; m.act[2 * NU] = m.act[2 * NU + 1] = &st->xa;
            run(mv_worker, &m);
            sj.gu = st->gu; sj.ea = st->ea; sj.ff = FFE; sj.n = NU + 1;
            run(swiglu_worker, &sj);
            for (e = 0; e < NU; ++e) act_prepare(&st->ea[e], L->down.type, FFE);
            act_prepare(&st->ea[NU], L->sdown.type, FFE);
            memset(&m, 0, sizeof m);
            m.rows_per_seg = D; m.nseg = NU + 1; m.cols = FFE; m.out = st->y;
            for (e = 0; e < NU; ++e) { m.w[e] = &L->down; m.r0[e] = (uint64_t)sel[e] * D; m.act[e] = &st->ea[e]; }
            m.w[NU] = &L->sdown; m.act[NU] = &st->ea[NU];
            run(mv_worker, &m);
            for (i = 0; i < D; ++i) {
                float s = 0.0f;
                for (e = 0; e < NU; ++e) s += w[e] * st->y[(size_t)e * D + i];
                x[i] += s + st->y[(size_t)NU * D + i];
            }
        }
    }
    rmsnorm_w(st->h, x, md->out_norm, D, st->eps);
    act_set(&st->xa, st->h, D);
    matvec(&md->out, &st->xa, st->logits, VOCAB);
}

/* ------------------------------------------------------------------ loading -- */
static void load_model(model_t *md, const char *path) {
    char name[96];
    unsigned l;
    gguf *g = &md->g;
    gguf_open(g, path);
    if (gguf_u(g, "bailingmoe2.block_count") != NL || gguf_u(g, "bailingmoe2.embedding_length") != D ||
        gguf_u(g, "bailingmoe2.attention.head_count") != NH || gguf_u(g, "bailingmoe2.attention.head_count_kv") != NKV ||
        gguf_u(g, "bailingmoe2.attention.key_length") != HD || gguf_u(g, "bailingmoe2.rope.dimension_count") != ROT ||
        gguf_u(g, "bailingmoe2.expert_count") != NE || gguf_u(g, "bailingmoe2.expert_used_count") != NU ||
        gguf_u(g, "bailingmoe2.expert_group_count") != NG || gguf_u(g, "bailingmoe2.expert_group_used_count") != TG ||
        gguf_u(g, "bailingmoe2.expert_feed_forward_length") != FFE || gguf_u(g, "bailingmoe2.feed_forward_length") != FFD ||
        gguf_u(g, "bailingmoe2.leading_dense_block_count") != 1 || gguf_u(g, "bailingmoe2.expert_shared_count") != 1 ||
        gguf_u(g, "bailingmoe2.expert_gating_func") != 2 || gguf_u(g, "bailingmoe2.expert_weights_norm") != 1 ||
        gguf_u(g, "bailingmoe2.vocab_size") != VOCAB) {
        fprintf(stderr, "GGUF hyper-parameters differ from the compiled Ling-mini-2.0 shape\n");
        exit(2);
    }
    md->embd = wt_get(g, "token_embd.weight", D);
    md->out = wt_get(g, "output.weight", D);
    md->out_norm = f32_get(g, "output_norm.weight", D);
    for (l = 0; l < NL; ++l) {
        layer_t *L = &md->l[l];
#define NM(s) (snprintf(name, sizeof name, "blk.%u." s, l), name)
        L->qkv = wt_get(g, NM("attn_qkv.weight"), D);
        L->o = wt_get(g, NM("attn_output.weight"), D);
        L->attn_norm = f32_get(g, NM("attn_norm.weight"), D);
        L->ffn_norm = f32_get(g, NM("ffn_norm.weight"), D);
        L->qn = f32_get(g, NM("attn_q_norm.weight"), HD);
        L->kn = f32_get(g, NM("attn_k_norm.weight"), HD);
        if (l == 0) {
            L->gate = wt_get(g, NM("ffn_gate.weight"), D);
            L->up = wt_get(g, NM("ffn_up.weight"), D);
            L->down = wt_get(g, NM("ffn_down.weight"), FFD);
        } else {
            L->gate = wt_get(g, NM("ffn_gate_exps.weight"), D);
            L->up = wt_get(g, NM("ffn_up_exps.weight"), D);
            L->down = wt_get(g, NM("ffn_down_exps.weight"), FFE);
            L->sgate = wt_get(g, NM("ffn_gate_shexp.weight"), D);
            L->sup = wt_get(g, NM("ffn_up_shexp.weight"), D);
            L->sdown = wt_get(g, NM("ffn_down_shexp.weight"), FFE);
            L->router = f32_get(g, NM("ffn_gate_inp.weight"), (uint64_t)NE * D);
            L->bias = f32_get(g, NM("exp_probs_b.bias"), NE);
        }
#undef NM
    }
}

static void type_summary(const model_t *md, char *buf, size_t n) {
    const layer_t *a = &md->l[0], *b = &md->l[1];
    snprintf(buf, n, "embd=%s head=%s qkv=%s o=%s dense=%s/%s/%s exps=%s/%s/%s shexp=%s/%s/%s",
             gg_type_name(md->embd.type), gg_type_name(md->out.type), gg_type_name(a->qkv.type), gg_type_name(a->o.type),
             gg_type_name(a->gate.type), gg_type_name(a->up.type), gg_type_name(a->down.type),
             gg_type_name(b->gate.type), gg_type_name(b->up.type), gg_type_name(b->down.type),
             gg_type_name(b->sgate.type), gg_type_name(b->sup.type), gg_type_name(b->sdown.type));
}

/* ----------------------------------------------------------------- selftest --
 * For Q4_K, Q5_K, Q6_K and Q8_0 rows of 2048 random weights (finite fp16 scales):
 *   AVX2 dot vs the generic transcription, and both vs an independent oracle
 *   sum_i dequant(w)_i * dequant(act)_i in double (dequantizers used by the embedding).
 * Error is normalised by the sum over quant blocks of |block dot| (as in C50-A1 addendum 1);
 * bound 1e-5.
 * Planted control: one flipped high bit in one weight byte of the oracle's copy must
 * exceed the bound in >= 190 of 200 trials per type. */
static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void) { g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27; return g_rng * 2685821657736338717ULL; }

static void fill_row(uint32_t type, uint8_t *p, size_t bytes) {
    size_t i, bb = type == GT_Q8_0 ? 34 : type == GT_Q4_K ? 144 : type == GT_Q5_K ? 176 : 210;
    for (i = 0; i < bytes; ++i) p[i] = (uint8_t)rnd();
    for (i = 0; i < bytes / bb; ++i) {
        uint8_t *b = p + i * bb;
        uint16_t h = (uint16_t)(0x1C00U | (rnd() & 0x3FFU));
        size_t off = type == GT_Q6_K ? 208 : 0;
        memcpy(b + off, &h, 2);
        if (type == GT_Q4_K || type == GT_Q5_K) { h = (uint16_t)(0x1C00U | (rnd() & 0x3FFU)); memcpy(b + 2, &h, 2); }
    }
}

static int selftest(void) {
    const uint32_t types[4] = {GT_Q4_K, GT_Q5_K, GT_Q6_K, GT_Q8_0};
    const unsigned n = D, trials = 200;
    static uint8_t row[2048 / 32 * 34 + 2048 / 256 * 210], bad[sizeof row];
    static float w[D], xa[D];
    static act_t a;
    int fail = 0;
    unsigned t, k, i;
    for (t = 0; t < 4; ++t) {
        wt W = {types[t], n, 1, gg_type_row_bytes(types[t], n), row};
        double worst_gen = 0, worst_or = 0;
        unsigned detected = 0;
        for (k = 0; k < trials; ++k) {
            double ref = 0, absn = 0, ref_bad = 0;
            float avx, gen;
            for (i = 0; i < n; ++i) a.f[i] = (float)((int64_t)(rnd() >> 40) - (1 << 23)) / (float)(1 << 23);
            a.has_k = a.has_z = 0;
            act_prepare(&a, types[t], n);
            fill_row(types[t], row, W.rb);
            avx = dot_row(types[t], row, &a, n);
            switch (types[t]) {
            case GT_Q4_K: gen = strat01_q4k_q8k_dot_generic(row, a.k, n); break;
            case GT_Q5_K: gen = c50_q5_k_dot_generic((const c50_block_q5_k *)row, a.k, n); break;
            case GT_Q6_K: gen = strat01_q6k_q8k_dot_generic(row, a.k, n); break;
            default: gen = c50_q8_0_dot_generic((const c50_block_q8_0 *)row, a.z, n); break;
            }
            for (i = 0; i < n; ++i) xa[i] = types[t] == GT_Q8_0 ? a.z[i / 32].qs[i % 32] * c50_h2f(a.z[i / 32].d)
                                                               : a.k[i / 256].qs[i % 256] * a.k[i / 256].d;
            dequant_row(&W, 0, w);
            {   /* normaliser of C50-A1 addendum 1: sum over quant blocks of |block dot| */
                const unsigned bs = types[t] == GT_Q8_0 ? 32 : 256;
                double blk = 0;
                for (i = 0; i < n; ++i) {
                    ref += (double)w[i] * xa[i];
                    blk += (double)w[i] * xa[i];
                    if ((i + 1) % bs == 0) { absn += fabs(blk); blk = 0; }
                }
            }
            memcpy(bad, row, W.rb);
            {   /* planted: flip bit 3 of one quant byte (outside the scale fields) */
                size_t bb = W.rb / (types[t] == GT_Q8_0 ? n / 32 : n / 256), blk = rnd() % (W.rb / bb), off;
                if (types[t] == GT_Q8_0) off = 2 + rnd() % 32;
                else if (types[t] == GT_Q6_K) off = rnd() % 128;
                else off = bb - 1 - rnd() % 128;
                bad[blk * bb + off] ^= 0x08;
            }
            {
                wt B = W;
                B.data = bad;
                dequant_row(&B, 0, w);
                for (i = 0; i < n; ++i) ref_bad += (double)w[i] * xa[i];
            }
            if (fabs(gen - ref) / absn > worst_or) worst_or = fabs(gen - ref) / absn;
            if (fabs((double)avx - gen) / absn > worst_gen) worst_gen = fabs((double)avx - gen) / absn;
            if (fabs((double)avx - ref_bad) / absn > 1e-5) ++detected;
        }
        printf("selftest %s: worst |avx2-generic| %.2e, worst |generic-oracle| %.2e (bound 1e-5); planted detected %u/%u\n",
               gg_type_name(types[t]), worst_gen, worst_or, detected, trials);
        fail |= worst_gen > 1e-5 || worst_or > 1e-5 || detected < 190;
    }
    printf("selftest %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

/* ------------------------------------------------------------------ scoring -- */
typedef struct { int32_t doc, pos, target, top1; float logp_target, logp_top1; double lse; } rec_t;

static int read_line_ids(FILE *f, int *doc, int **ids, unsigned *n, unsigned *cap) {
    int c, v = 0, have = 0;
    *n = 0;
    if (fscanf(f, "%d\t", doc) != 1) return 0;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); have = 1; }
        else if (c == ',') { if (*n == *cap) { *cap = *cap ? 2 * *cap : 4096; *ids = realloc(*ids, sizeof(int) * *cap); } (*ids)[(*n)++] = v; v = 0; have = 0; }
        else if (c != '\r') { fprintf(stderr, "bad ids line\n"); exit(2); }
    }
    if (have) { if (*n == *cap) { *cap = *cap ? 2 * *cap : 4096; *ids = realloc(*ids, sizeof(int) * *cap); } (*ids)[(*n)++] = v; }
    return 1;
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* Decode-rate bench: the first document of IDS with >= ctx + n tokens is decoded token by
 * token (teacher-forced, full head, no log-softmax, no dump); steps at positions
 * ctx .. ctx+n-1 are timed one by one.  Prints one JSON line.  Rates need a clean box. */
static int bench(const model_t *md, state_t *st, FILE *fi, unsigned ctx, unsigned nsteps, const char *types,
                 const char *model_path) {
    int doc, *ids = NULL;
    unsigned n, cap = 0, p;
    double *t = malloc(sizeof(double) * nsteps), t_pre, med;
    while (read_line_ids(fi, &doc, &ids, &n, &cap)) {
        if (n < ctx + nsteps) continue;
        printf("CONFIG engine=c50_engine mode=bench model=%s doc=%d ctx=%u steps=%u threads=%d cores=%d,%d,%d,%d,%d,%d "
               "kv=f16 types=[%s]\n", model_path, doc, ctx, nsteps, NT, g_cpu[0], g_cpu[1], g_cpu[2], g_cpu[3], g_cpu[4],
               g_cpu[5], types);
        fflush(stdout);
        t_pre = now_s();
        for (p = 0; p < ctx; ++p) forward(md, st, ids[p], p);
        t_pre = now_s() - t_pre;
        for (p = 0; p < nsteps; ++p) {
            double t0 = now_s();
            forward(md, st, ids[ctx + p], ctx + p);
            t[p] = now_s() - t0;
        }
        qsort(t, nsteps, sizeof *t, cmpd);
        med = nsteps % 2 ? t[nsteps / 2] : 0.5 * (t[nsteps / 2 - 1] + t[nsteps / 2]);
        printf("{\"mode\":\"bench\",\"ctx\":%u,\"steps\":%u,\"fill_s\":%.3f,\"median_step_s\":%.6f,\"min_step_s\":%.6f,"
               "\"max_step_s\":%.6f,\"median_tok_s\":%.3f}\n", ctx, nsteps, t_pre, med, t[0], t[nsteps - 1], 1.0 / med);
        return 0;
    }
    fprintf(stderr, "no document with >= %u tokens\n", ctx + nsteps);
    return 2;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *ids_path = NULL, *out = NULL, *bench_path = NULL;
    unsigned bench_ctx = 2048, bench_n = 128;
    int dump = 0, i;
    model_t *md = calloc(1, sizeof *md);
    state_t *st;
    char types[256], path[1024];
    FILE *fi, *fpos, *flp = NULL;
    int doc, *ids = NULL;
    unsigned n, cap = 0, docs = 0;
    size_t positions = 0;
    double t0, wall;
    float *row;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--score") && i + 1 < argc) ids_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--dump-logprobs")) dump = 1;
        else if (!strcmp(argv[i], "--selftest")) return selftest();
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench_path = argv[++i];
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) bench_ctx = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) bench_n = (unsigned)atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model_path || (!bench_path && (!ids_path || !out)) || bench_ctx + bench_n > MAXCTX || !bench_n) {
        fprintf(stderr, "usage: c50_engine --model GGUF --score IDS.tsv --out PREFIX [--dump-logprobs]\n"
                        "       c50_engine --model GGUF --bench IDS.tsv [--ctx 2048] [--steps 128]\n");
        return 2;
    }
    load_model(md, model_path);
    st = calloc(1, sizeof *st);
    st->kc = VirtualAlloc(NULL, (size_t)NL * MAXCTX * NKV * HD * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    st->vc = VirtualAlloc(NULL, (size_t)NL * MAXCTX * NKV * HD * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    st->logits = malloc(sizeof(float) * VOCAB);
    row = malloc(sizeof(float) * VOCAB);
    st->eps = gguf_f(&md->g, "bailingmoe2.attention.layer_norm_rms_epsilon");
    st->rope_base = gguf_f(&md->g, "bailingmoe2.rope.freq_base");
    st->wscale = gguf_f(&md->g, "bailingmoe2.expert_weights_scale");
    /* Planted faults for the A3 system-gate control (addendum 3); separate binaries only, the CONFIG
     * line prints the planted value.  The gate binary is built without either macro. */
#ifdef C50_PLANT_WSCALE
    st->wscale *= (float)(C50_PLANT_WSCALE);
#endif
#ifdef C50_PLANT_ROPE_BASE
    st->rope_base = (float)(C50_PLANT_ROPE_BASE);
#endif
    if (!st->kc || !st->vc || !st->logits || !row) { fprintf(stderr, "allocation failed\n"); return 2; }
    start_pool();
    type_summary(md, types, sizeof types);
    if (bench_path) {
        fi = fopen(bench_path, "rb");
        if (!fi) { fprintf(stderr, "cannot open ids\n"); return 2; }
        return bench(md, st, fi, bench_ctx, bench_n, types, model_path);
    }
    fi = fopen(ids_path, "rb");
    if (!fi) { fprintf(stderr, "cannot open ids\n"); return 2; }
    while (read_line_ids(fi, &doc, &ids, &n, &cap)) {
        if (n < 2 || n > MAXCTX) { fprintf(stderr, "document length outside [2, %u]\n", MAXCTX); return 2; }
        positions += n - 1; ++docs;
    }
    rewind(fi);
    printf("CONFIG engine=c50_engine model=%s docs=%u positions=%zu threads=%d cores=%d,%d,%d,%d,%d,%d kv=f16 "
           "attention=f32 eps=%g rope_base=%g weights_scale=%g dump=%d types=[%s]\n",
           model_path, docs, positions, NT, g_cpu[0], g_cpu[1], g_cpu[2], g_cpu[3], g_cpu[4], g_cpu[5],
           st->eps, st->rope_base, st->wscale, dump, types);
    fflush(stdout);
    snprintf(path, sizeof path, "%s.pos.bin", out);
    fpos = fopen(path, "wb");
    if (dump) { snprintf(path, sizeof path, "%s.logp.f32", out); flp = fopen(path, "wb"); }
    if (!fpos || (dump && !flp)) { fprintf(stderr, "cannot open outputs\n"); return 2; }
    t0 = now_s();
    while (read_line_ids(fi, &doc, &ids, &n, &cap)) {
        unsigned p;
        for (p = 0; p + 1 < n; ++p) {
            double mx = -INFINITY, sum = 0.0, lse;
            unsigned v, arg = 0;
            rec_t r;
            if (ids[p] < 0 || ids[p] >= (int)VOCAB) { fprintf(stderr, "token id outside the vocabulary\n"); return 2; }
            forward(md, st, ids[p], p);
            for (v = 0; v < VOCAB; ++v) if (st->logits[v] > mx) { mx = st->logits[v]; arg = v; }
            for (v = 0; v < VOCAB; ++v) sum += exp((double)st->logits[v] - mx);
            lse = mx + log(sum);
            r.doc = doc; r.pos = (int32_t)(p + 1); r.target = ids[p + 1]; r.top1 = (int32_t)arg;
            r.logp_target = (float)(st->logits[ids[p + 1]] - lse); r.logp_top1 = (float)(st->logits[arg] - lse); r.lse = lse;
            fwrite(&r, sizeof r, 1, fpos);
            if (flp) {
                for (v = 0; v < VOCAB; ++v) row[v] = (float)(st->logits[v] - lse);
                fwrite(row, sizeof(float), VOCAB, flp);
            }
        }
    }
    wall = now_s() - t0;
    fclose(fpos);
    if (flp) fclose(flp);
    snprintf(path, sizeof path, "%s.json", out);
    fi = fopen(path, "w");
    fprintf(fi, "{\"engine\":\"c50_engine\",\"model\":\"%s\",\"n_vocab\":%u,\"docs\":%u,\"positions\":%zu,\"threads\":%d,"
                "\"ubatch\":1,\"ctx\":%u,\"flash_attn\":\"n/a\",\"kv\":\"f16\",\"dump\":%d,\"wall_s\":%.3f,\"types\":\"%s\"}\n",
            model_path, VOCAB, docs, positions, NT, MAXCTX, dump, wall, types);
    fclose(fi);
    printf("DONE positions=%zu wall_s=%.1f\n", positions, wall);
    return 0;
}
