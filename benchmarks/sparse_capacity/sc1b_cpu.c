// Sparse Capacity, phase 1b (D-M): CPU kernels for "magazzino contro officina".
// Protocol: docs/research/SPARSE_CAPACITY/F1B_CROSSOVER_PROTOCOL.md
//
// A bank of E experts, each a gated dReLU FFN with hidden h on D inputs, stored as 3h rows of D:
//   rows [0,h) = gate (Wg), [h,2h) = up (Wu), [2h,3h) = down TRANSPOSED (WdT, row i = column i of Wd).
//   forward  : g_i = Wg[i].x, u_i = Wu[i].x, hv_i = relu(g_i)*relu(u_i), y += sum_i hv_i * WdT[i]
//   backward : dh_i = WdT[i].dy ; dg_i = dh_i*relu(u_i)*[g_i>0] ; du_i = dh_i*relu(g_i)*[u_i>0]
//              dx += dg_i*Wg[i] + du_i*Wu[i] ; grad Wg[i] = sum_t dg x ; Wu[i] = sum_t du x ; WdT[i] = sum_t hv dy
// Every hidden unit is independent in the backward once (g,u) are saved, so a FUSED backward reads each of its three
// rows once, applies the optimizer and writes them back: the weight gradient is never materialized (LOMO, FBGEMM TBE).
// A hidden unit with no active (g>0 and u>0) token has zero gradient: SGD and row-wise Adagrad skip it exactly; dense
// Adam does not (its moments still move), and the unfused path still writes its zero gradient, as autograd does.
//
// Formats: F32 master | BF16 master with stochastic rounding | TERN = fp32 latent + pair-packed ternary codes
// (engine.c packing c = (w0+1)*3+(w1+1)) with a per-row absmean scale, int8 activations at AQ = 63 and the
// engine's pshufb LUT for the gate/up forward when h % 32 == 0 (engine's row-major scalar LUT otherwise).
//
// Build: gcc -O3 -mavx2 -mfma -march=znver2 -fopenmp -shared -o bin/sc1b_cpu.dll sc1b_cpu.c -lm   (NO -ffast-math)
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define AQ 63
#define DMAX 1024
enum { FMT_F32 = 0, FMT_BF16 = 1, FMT_TERN = 2 };
enum { OPT_SGD = 0, OPT_ROWADA = 1, OPT_ADAM = 2 };

typedef struct {
    int32_t E, h, D, r, T, nthr;
    const int32_t* tok;   // E*r token index per (expert, slot)
    const float* X;       // T*D
    const float* DY;      // T*D
    float* Y;             // T*D
    float* DX;            // T*D
    float* acc;           // nthr*T*D per-thread accumulators
    float* GU;            // E*r*h*2 saved (g,u)
    int8_t* XQ;           // T*D   (TERN)
    float* XSA;           // T     (TERN)
    int8_t* LUT;          // T*(D/2)*16 (TERN)
    float lr, eps, b1, b2;
    int32_t step;
    uint32_t seed;
    int64_t n_active;     // out: active (pair, hidden) count in the last forward
    int64_t n_rows_upd;   // out: rows whose weights were read+written in the last backward/update
} Ctx;

typedef struct {
    float* W;          // F32 master or TERN latent (E*3h*D)
    uint16_t* Wb;      // BF16 master (E*3h*D)
    uint8_t* codes;    // TERN codes, E*3h*(D/2)
    float* sc;         // TERN scales, E*3h
    float* S1;         // ROWADA: E*3h ; ADAM: m (E*3h*D)
    float* S2;         // ADAM: v
    float* G;          // unfused gradient bank (E*3h*D)
} Bank;

// ------------------------------------------------------------------ small helpers
EXPORT void* xalloc(int64_t bytes) {
#ifdef _WIN32
    return _aligned_malloc((size_t)bytes, 64);
#else
    void* p = NULL; if (posix_memalign(&p, 64, (size_t)bytes)) return NULL; return p;
#endif
}
EXPORT void xfree(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo); return _mm_cvtss_f32(lo);
}
static inline float dotv(const float* a, const float* b, int n) {         // n % 32 == 0
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (int i = 0; i < n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    return hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
}
// two weight rows against up to 4 token vectors: each weight chunk is loaded once for all tokens
static inline void dot2x4(const float* wg, const float* wu, const float* const* x, int nb, int n, float* g, float* u) {
    __m256 ag[4], au[4];
    for (int k = 0; k < 4; k++) { ag[k] = _mm256_setzero_ps(); au[k] = _mm256_setzero_ps(); }
    for (int i = 0; i < n; i += 8) {
        __m256 a = _mm256_loadu_ps(wg + i), b = _mm256_loadu_ps(wu + i);
        for (int k = 0; k < nb; k++) { __m256 xv = _mm256_loadu_ps(x[k] + i);
            ag[k] = _mm256_fmadd_ps(a, xv, ag[k]); au[k] = _mm256_fmadd_ps(b, xv, au[k]); }
    }
    for (int k = 0; k < nb; k++) { g[k] = hsum256(ag[k]); u[k] = hsum256(au[k]); }
}
static inline void axpyv(float* restrict y, float a, const float* restrict x, int n) {
    __m256 va = _mm256_set1_ps(a);
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
}
static inline void bf16_to_f32(const uint16_t* s, float* d, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s + i)));
        _mm256_storeu_ps(d + i, _mm256_castsi256_ps(_mm256_slli_epi32(v, 16)));
    }
}
static inline uint16_t f32_to_bf16_rne(float f) { uint32_t b; memcpy(&b, &f, 4); b += 0x7FFFu + ((b >> 16) & 1u); return (uint16_t)(b >> 16); }
static inline uint32_t xs32(uint32_t* s) { uint32_t x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *s = x; return x; }
static inline uint16_t f32_to_bf16_sr(float f, uint32_t* rng) { uint32_t b; memcpy(&b, &f, 4); b += (xs32(rng) & 0xFFFFu); return (uint16_t)(b >> 16); }

// ------------------------------------------------------------------ ternary helpers (engine.c semantics)
static float quant_i8(const float* x, int n, int8_t* xq) {               // engine.c:167-169
    float amax = 0; for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax == 0.0f) { memset(xq, 0, n); return 0.0f; }
    float scale = amax / (float)AQ, inv = 1.0f / scale;
    for (int i = 0; i < n; i++) { int v = (int)lrintf(x[i] * inv); if (v > AQ) v = AQ; if (v < -AQ) v = -AQ; xq[i] = (int8_t)v; }
    return scale;
}
static void build_lut_t3(const int8_t* xq, int T, int8_t* lut) {        // engine.c:158-159
    for (int t = 0; t < T; t++) { int8_t x0 = xq[2 * t], x1 = xq[2 * t + 1];
        for (int c = 0; c < 16; c++) { int s = 0; if (c < 9) { int w0 = c / 3 - 1, w1 = c % 3 - 1; s = w0 * x0 + w1 * x1; } lut[t * 16 + c] = (int8_t)s; } }
}
static inline void acc_add_i8x32(__m256i* acc, __m256i p) {             // engine.c:106-110
    __m128i lo = _mm256_castsi256_si128(p), hi = _mm256_extracti128_si256(p, 1);
    acc[0] = _mm256_add_epi32(acc[0], _mm256_cvtepi8_epi32(lo)); acc[1] = _mm256_add_epi32(acc[1], _mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    acc[2] = _mm256_add_epi32(acc[2], _mm256_cvtepi8_epi32(hi)); acc[3] = _mm256_add_epi32(acc[3], _mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
}
// engine.c matvec_lut_rows without the inner OpenMP: M rows (M % 32 == 0) of a tile-major block codes[t*Mpad+m]
static void lut_rows(const uint8_t* codes, const int8_t* lut, int32_t* y, int M, int Mpad, int Tp) {
    for (int bb = 0; bb < M; bb += 32) {
        __m256i acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
        for (int t = 0; t < Tp; t++) {
            __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut + (size_t)t * 16)));
            __m256i idx = _mm256_loadu_si256((const __m256i*)(codes + (size_t)t * Mpad + bb));
            acc_add_i8x32(acc, _mm256_shuffle_epi8(tbl, idx));
        }
        _mm256_storeu_si256((__m256i*)(y + bb), acc[0]); _mm256_storeu_si256((__m256i*)(y + bb + 8), acc[1]);
        _mm256_storeu_si256((__m256i*)(y + bb + 16), acc[2]); _mm256_storeu_si256((__m256i*)(y + bb + 24), acc[3]);
    }
}
static const float TV0[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1}, TV1[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
// code address of (row i of quarter q in {0 gate,1 up,2 down}, pair t) inside an expert block of 3h*(D/2) bytes
static inline size_t code_at(int h, int Dp, int q, int i, int t, int tile) {
    size_t base = (size_t)q * h * Dp;
    if (q < 2 && tile) return base + (size_t)t * h + i;      // tile-major gate/up (engine layout, Mpad = h)
    return base + (size_t)i * Dp + t;                         // row-major
}
// requantize one latent row: scale = mean|w| (clamp 1e-5), q = clamp(rint(w/scale)), write codes (engine packing)
static float tern_pack_row(const float* w, int D, uint8_t* blk, int h, int q, int i, int tile) {
    float s = 0; for (int d = 0; d < D; d++) s += fabsf(w[d]);
    s /= (float)D; if (s < 1e-5f) s = 1e-5f;
    int Dp = D / 2;
    for (int t = 0; t < Dp; t++) {
        int a = (int)rintf(w[2 * t] / s), b = (int)rintf(w[2 * t + 1] / s);
        a = a > 1 ? 1 : (a < -1 ? -1 : a); b = b > 1 ? 1 : (b < -1 ? -1 : b);
        blk[code_at(h, Dp, q, i, t, tile)] = (uint8_t)((a + 1) * 3 + (b + 1));
    }
    return s;
}
static inline void tern_unpack_row(const uint8_t* blk, int h, int D, int q, int i, int tile, float s, float* out) {
    int Dp = D / 2;
    for (int t = 0; t < Dp; t++) { uint8_t c = blk[code_at(h, Dp, q, i, t, tile)]; out[2 * t] = TV0[c] * s; out[2 * t + 1] = TV1[c] * s; }
}
static inline void tern_deq_latent(const float* w, float s, int D, float* out) {     // STE forward weight from the latent
    for (int d = 0; d < D; d++) { float q = rintf(w[d] / s); q = q > 1 ? 1 : (q < -1 ? -1 : q); out[d] = q * s; }
}
static inline int tile_of(int h) { return (h % 32) == 0; }

// pack every row of a latent bank (outside any timed region)
EXPORT void tern_init(Ctx* c, Bank* b, int tile_ok) {
    int E = c->E, h = c->h, D = c->D, Dp = D / 2, tile = tile_ok && tile_of(h);
    #pragma omp parallel for schedule(static) num_threads(c->nthr)
    for (int e = 0; e < E; e++) {
        uint8_t* blk = b->codes + (size_t)e * 3 * h * Dp;
        for (int q = 0; q < 3; q++) for (int i = 0; i < h; i++) {
            size_t R = (size_t)e * 3 * h + (size_t)q * h + i;
            b->sc[R] = tern_pack_row(b->W + R * D, D, blk, h, q, i, tile);
        }
    }
}

// ------------------------------------------------------------------ optimizers on one row (fp32 view)
static inline void opt_row(int opt, float* w, const float* g, int D, float* s1row, float* m, float* v, const Ctx* c) {
    if (opt == OPT_SGD) { for (int d = 0; d < D; d++) w[d] -= c->lr * g[d]; return; }
    if (opt == OPT_ROWADA) {
        float ss = 0; for (int d = 0; d < D; d++) ss += g[d] * g[d];
        *s1row += ss / (float)D;
        float k = c->lr / (sqrtf(*s1row) + c->eps);
        for (int d = 0; d < D; d++) w[d] -= k * g[d];
        return;
    }
    float bc1 = 1.0f - powf(c->b1, (float)c->step), bc2 = 1.0f - powf(c->b2, (float)c->step);   // OPT_ADAM
    for (int d = 0; d < D; d++) {
        float gd = g ? g[d] : 0.0f;
        m[d] = c->b1 * m[d] + (1.0f - c->b1) * gd;
        v[d] = c->b2 * v[d] + (1.0f - c->b2) * gd * gd;
        w[d] -= c->lr * (m[d] / bc1) / (sqrtf(v[d] / bc2) + c->eps);
    }
}

// The per-thread accumulators (zero before a sweep, reduce after) are a per-LAYER cost that does not scale with the
// size of this bank slice: the runner calls them separately, times them apart and charges them pro rata.
EXPORT void acc_zero(Ctx* c) {
    size_t n = (size_t)c->nthr * c->T * c->D;
    #pragma omp parallel for schedule(static) num_threads(c->nthr)
    for (int64_t k = 0; k < (int64_t)(n / 4096); k++) memset(c->acc + k * 4096, 0, 4096 * 4);
    size_t rem = n % 4096; if (rem) memset(c->acc + n - rem, 0, rem * 4);
}
EXPORT void acc_reduce(Ctx* c, int which) {
    float* out = which ? c->DX : c->Y;
    int T = c->T, D = c->D, P = c->nthr;
    #pragma omp parallel for schedule(static) num_threads(c->nthr)
    for (int t = 0; t < T; t++) {
        float* o = out + (size_t)t * D;
        memcpy(o, c->acc + (size_t)t * D, D * 4);
        for (int p = 1; p < P; p++) { const float* a = c->acc + ((size_t)p * T + t) * D; for (int d = 0; d < D; d++) o[d] += a[d]; }
    }
}

// ------------------------------------------------------------------ CPU path: forward sweep over the whole bank
EXPORT void cpu_fwd(Ctx* c, int fmt, Bank* b) {
    int E = c->E, h = c->h, D = c->D, r = c->r, Dp = D / 2, tile = tile_of(h);
    if (fmt == FMT_TERN) {                                       // per-token activation quantization + LUT, once
        #pragma omp parallel for schedule(static) num_threads(c->nthr)
        for (int t = 0; t < c->T; t++) {
            c->XSA[t] = quant_i8(c->X + (size_t)t * D, D, c->XQ + (size_t)t * D);
            build_lut_t3(c->XQ + (size_t)t * D, Dp, c->LUT + (size_t)t * Dp * 16);
        }
    }
    int64_t nact = 0;
    #pragma omp parallel num_threads(c->nthr) reduction(+:nact)
    {
        int tid = omp_get_thread_num();
        float* acc = c->acc + (size_t)tid * c->T * D;
        float bg[DMAX], bu[DMAX], bd[DMAX];
        int32_t Sg[DMAX], Su[DMAX];
        #pragma omp for schedule(static)
        for (int e = 0; e < E; e++) {
            const int32_t* tk = c->tok + (size_t)e * r;
            float* gu = c->GU + (size_t)e * r * h * 2;
            size_t R0 = (size_t)e * 3 * h;
            if (fmt == FMT_TERN) {
                const uint8_t* blk = b->codes + R0 * Dp;
                if (tile) {
                    for (int j = 0; j < r; j++) {
                        const int8_t* lut = c->LUT + (size_t)tk[j] * Dp * 16; float sa = c->XSA[tk[j]];
                        lut_rows(blk, lut, Sg, h, h, Dp);
                        lut_rows(blk + (size_t)h * Dp, lut, Su, h, h, Dp);
                        for (int i = 0; i < h; i++) {
                            gu[((size_t)j * h + i) * 2] = (float)Sg[i] * sa * b->sc[R0 + i];
                            gu[((size_t)j * h + i) * 2 + 1] = (float)Su[i] * sa * b->sc[R0 + h + i];
                        }
                    }
                } else {
                    for (int i = 0; i < h; i++) for (int j = 0; j < r; j++) {
                        const int8_t* lut = c->LUT + (size_t)tk[j] * Dp * 16; float sa = c->XSA[tk[j]];
                        const uint8_t* cg = blk + code_at(h, Dp, 0, i, 0, 0); const uint8_t* cu = blk + code_at(h, Dp, 1, i, 0, 0);
                        int sg = 0, su = 0; for (int t = 0; t < Dp; t++) { sg += lut[t * 16 + cg[t]]; su += lut[t * 16 + cu[t]]; }
                        gu[((size_t)j * h + i) * 2] = (float)sg * sa * b->sc[R0 + i];
                        gu[((size_t)j * h + i) * 2 + 1] = (float)su * sa * b->sc[R0 + h + i];
                    }
                }
            } else {
                for (int i = 0; i < h; i++) {
                    const float *wg, *wu;
                    if (fmt == FMT_F32) { wg = b->W + (R0 + i) * D; wu = b->W + (R0 + h + i) * D; }
                    else { bf16_to_f32(b->Wb + (R0 + i) * D, bg, D); bf16_to_f32(b->Wb + (R0 + h + i) * D, bu, D); wg = bg; wu = bu; }
                    for (int j0 = 0; j0 < r; j0 += 4) {
                        int nb = r - j0 < 4 ? r - j0 : 4; const float* xs[4]; float g4[4], u4[4];
                        for (int k = 0; k < nb; k++) xs[k] = c->X + (size_t)tk[j0 + k] * D;
                        dot2x4(wg, wu, xs, nb, D, g4, u4);
                        for (int k = 0; k < nb; k++) { gu[((size_t)(j0 + k) * h + i) * 2] = g4[k]; gu[((size_t)(j0 + k) * h + i) * 2 + 1] = u4[k]; }
                    }
                }
            }
            // down: y += hv * WdT[i], row read only if some token activates hidden unit i
            for (int i = 0; i < h; i++) {
                int any = 0;
                for (int j = 0; j < r; j++) { float g = gu[((size_t)j * h + i) * 2], u = gu[((size_t)j * h + i) * 2 + 1]; if (g > 0 && u > 0) { any = 1; break; } }
                if (!any) continue;
                const float* wd; size_t R = R0 + 2 * h + i;
                if (fmt == FMT_F32) wd = b->W + R * D;
                else if (fmt == FMT_BF16) { bf16_to_f32(b->Wb + R * D, bd, D); wd = bd; }
                else { tern_unpack_row(b->codes + R0 * Dp, h, D, 2, i, tile, b->sc[R], bd); wd = bd; }
                for (int j = 0; j < r; j++) {
                    float g = gu[((size_t)j * h + i) * 2], u = gu[((size_t)j * h + i) * 2 + 1];
                    if (g > 0 && u > 0) { axpyv(acc + (size_t)tk[j] * D, g * u, wd, D); nact++; }
                }
            }
        }
    }
    c->n_active = nact;
}

// ------------------------------------------------------------------ CPU path: backward sweep (+ fused update)
EXPORT void cpu_bwd(Ctx* c, int fmt, int opt, int fused, Bank* b) {
    int E = c->E, h = c->h, D = c->D, r = c->r, Dp = D / 2, tile = tile_of(h);
    int64_t nupd = 0;
    #pragma omp parallel num_threads(c->nthr) reduction(+:nupd)
    {
        int tid = omp_get_thread_num();
        float* acc = c->acc + (size_t)tid * c->T * D;
        float wbuf[3][DMAX], grad[3][DMAX], xbuf[DMAX];
        uint32_t rng = c->seed * 2654435761u + 0x9E3779B9u * (uint32_t)(tid + 1);
        if (!rng) rng = 1;
        #pragma omp for schedule(static)
        for (int e = 0; e < E; e++) {
            const int32_t* tk = c->tok + (size_t)e * r;
            const float* gu = c->GU + (size_t)e * r * h * 2;
            size_t R0 = (size_t)e * 3 * h;
            for (int i = 0; i < h; i++) {
                int any = 0;
                for (int j = 0; j < r; j++) { float g = gu[((size_t)j * h + i) * 2], u = gu[((size_t)j * h + i) * 2 + 1]; if (g > 0 && u > 0) { any = 1; break; } }
                size_t R[3] = {R0 + i, R0 + h + i, R0 + 2 * h + i};
                if (!any) {
                    if (!fused) { for (int q = 0; q < 3; q++) memset(b->G + R[q] * D, 0, D * 4); nupd += 3; }   // autograd: dense zero grad
                    else if (opt == OPT_ADAM) {                                                                 // dense Adam: moments move
                        for (int q = 0; q < 3; q++) opt_row(OPT_ADAM, b->W + R[q] * D, NULL, D, NULL, b->S1 + R[q] * D, b->S2 + R[q] * D, c);
                        nupd += 3;
                    }
                    continue;
                }
                const float* w[3];
                for (int q = 0; q < 3; q++) {
                    if (fmt == FMT_F32) w[q] = b->W + R[q] * D;
                    else if (fmt == FMT_BF16) { bf16_to_f32(b->Wb + R[q] * D, wbuf[q], D); w[q] = wbuf[q]; }
                    else { tern_deq_latent(b->W + R[q] * D, b->sc[R[q]], D, wbuf[q]); w[q] = wbuf[q]; }
                    memset(grad[q], 0, D * 4);
                }
                for (int j = 0; j < r; j++) {
                    float g = gu[((size_t)j * h + i) * 2], u = gu[((size_t)j * h + i) * 2 + 1];
                    if (!(g > 0 && u > 0)) continue;
                    int t = tk[j];
                    const float* dy = c->DY + (size_t)t * D;
                    float dh = dotv(w[2], dy, D), dg = dh * u, du = dh * g;
                    float* dx = acc + (size_t)t * D;
                    axpyv(dx, dg, w[0], D); axpyv(dx, du, w[1], D);
                    const float* xv;
                    if (fmt == FMT_TERN) { float sa = c->XSA[t]; const int8_t* xq = c->XQ + (size_t)t * D; for (int d = 0; d < D; d++) xbuf[d] = (float)xq[d] * sa; xv = xbuf; }
                    else xv = c->X + (size_t)t * D;
                    axpyv(grad[0], dg, xv, D); axpyv(grad[1], du, xv, D); axpyv(grad[2], g * u, dy, D);
                }
                nupd += 3;
                for (int q = 0; q < 3; q++) {
                    if (!fused) { memcpy(b->G + R[q] * D, grad[q], D * 4); continue; }
                    if (fmt == FMT_F32) {
                        opt_row(opt, b->W + R[q] * D, grad[q], D, b->S1 ? b->S1 + (opt == OPT_ROWADA ? R[q] : R[q] * D) : NULL,
                                opt == OPT_ADAM ? b->S1 + R[q] * D : NULL, opt == OPT_ADAM ? b->S2 + R[q] * D : NULL, c);
                    } else if (fmt == FMT_BF16) {
                        opt_row(opt, wbuf[q], grad[q], D, b->S1 + R[q], NULL, NULL, c);
                        uint16_t* o = b->Wb + R[q] * D; for (int d = 0; d < D; d++) o[d] = f32_to_bf16_sr(wbuf[q][d], &rng);
                    } else {
                        float* lw = b->W + R[q] * D;
                        opt_row(opt, lw, grad[q], D, b->S1 + R[q], NULL, NULL, c);
                        b->sc[R[q]] = tern_pack_row(lw, D, b->codes + R0 * Dp, h, q, i, tile);
                    }
                }
            }
        }
    }
    c->n_rows_upd = nupd;
}

// unfused SGD update over the whole bank: W -= lr * G (the optimizer.step() of the autograd path)
EXPORT void cpu_upd_unfused(Ctx* c, Bank* b) {
    int64_t n = (int64_t)c->E * 3 * c->h * c->D; float lr = c->lr;
    #pragma omp parallel for schedule(static) num_threads(c->nthr)
    for (int64_t k = 0; k < n / 8; k++) {
        __m256 w = _mm256_loadu_ps(b->W + k * 8), g = _mm256_loadu_ps(b->G + k * 8);
        _mm256_storeu_ps(b->W + k * 8, _mm256_fnmadd_ps(_mm256_set1_ps(lr), g, w));
    }
}

// ------------------------------------------------------------------ streaming path: CPU-side update of a chunk
// gb = bf16 gradient rows from the GPU (nrows*D); flags = per-row "has nonzero gradient" (NULL = all rows).
// F32 formats write the next step's bf16 staging copy (RNE); TERN writes row-major codes + scales for the GPU.
EXPORT void str_update(int fmt, int opt, int64_t row0, int64_t nrows, int D, Ctx* c, Bank* b,
                       const uint16_t* gb, const uint8_t* flags, uint16_t* stg) {
    int64_t nupd = 0;
    #pragma omp parallel num_threads(c->nthr) reduction(+:nupd)
    {
        float g[DMAX];
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < nrows; k++) {
            int64_t R = row0 + k;
            if (flags && !flags[k] && opt != OPT_ADAM) continue;              // lazy-exact skip, as on the CPU path
            bf16_to_f32(gb + k * D, g, D);
            float* w = b->W + R * D;
            if (opt == OPT_ADAM) opt_row(OPT_ADAM, w, g, D, NULL, b->S1 + R * D, b->S2 + R * D, c);
            else opt_row(opt, w, g, D, b->S1 + R, NULL, NULL, c);
            if (fmt == FMT_TERN) {                                            // row-major codes, one row per "expert block"
                b->sc[R] = tern_pack_row(w, D, b->codes + R * (D / 2), 1, 0, 0, 0);
            } else {
                uint16_t* o = stg + R * D; for (int d = 0; d < D; d++) o[d] = f32_to_bf16_rne(w[d]);
            }
            nupd++;
        }
    }
    c->n_rows_upd += nupd;
}
EXPORT void tern_init_rowmajor(int64_t nrows, int D, Bank* b, int nthr) {
    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int64_t R = 0; R < nrows; R++) b->sc[R] = tern_pack_row(b->W + R * D, D, b->codes + R * (D / 2), 1, 0, 0, 0);
}
EXPORT void f32_to_bf16_all(const float* s, uint16_t* d, int64_t n, int nthr) {
    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int64_t k = 0; k < n; k++) d[k] = f32_to_bf16_rne(s[k]);
}

// ------------------------------------------------------------------ controls: RAM bandwidth, FMA plant, health
EXPORT double k_read(const float* a, int64_t n, int nthr) {               // n % 32 == 0; k_write_nt needs 32-byte alignment (xalloc)
    double tot = 0;
    #pragma omp parallel num_threads(nthr) reduction(+:tot)
    {
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < n / 32; k++) { const float* p = a + k * 32;
            s0 = _mm256_add_ps(s0, _mm256_loadu_ps(p)); s1 = _mm256_add_ps(s1, _mm256_loadu_ps(p + 8));
            s2 = _mm256_add_ps(s2, _mm256_loadu_ps(p + 16)); s3 = _mm256_add_ps(s3, _mm256_loadu_ps(p + 24)); }
        tot += hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    }
    return tot;
}
EXPORT void k_write_nt(float* a, int64_t n, float v, int nthr) {
    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int64_t k = 0; k < n / 8; k++) _mm256_stream_ps(a + k * 8, _mm256_set1_ps(v));
    _mm_sfence();
}
EXPORT void k_write(float* a, int64_t n, float v, int nthr) {
    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int64_t k = 0; k < n / 8; k++) _mm256_storeu_ps(a + k * 8, _mm256_set1_ps(v));
}
EXPORT void k_rmw(float* a, int64_t n, float s, int nthr) {
    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int64_t k = 0; k < n / 8; k++) _mm256_storeu_ps(a + k * 8, _mm256_mul_ps(_mm256_loadu_ps(a + k * 8), _mm256_set1_ps(s)));
}
// 10 independent FMA chains per thread (Zen2: 2 FMA pipes x latency 5); flops = iters*10*8*2*nthr
EXPORT double fma_plant(int64_t iters, int nthr) {
    double tot = 0;
    #pragma omp parallel num_threads(nthr) reduction(+:tot)
    {
        float f = 1e-3f * (float)(omp_get_thread_num() + 1);
        __m256 a0 = _mm256_set1_ps(f), a1 = _mm256_set1_ps(f + 1), a2 = _mm256_set1_ps(f + 2), a3 = _mm256_set1_ps(f + 3), a4 = _mm256_set1_ps(f + 4);
        __m256 a5 = _mm256_set1_ps(f + 5), a6 = _mm256_set1_ps(f + 6), a7 = _mm256_set1_ps(f + 7), a8 = _mm256_set1_ps(f + 8), a9 = _mm256_set1_ps(f + 9);
        const __m256 m = _mm256_set1_ps(0.9999999f), ad = _mm256_set1_ps(1e-7f);
        for (int64_t it = 0; it < iters; it++) {
            a0 = _mm256_fmadd_ps(a0, m, ad); a1 = _mm256_fmadd_ps(a1, m, ad); a2 = _mm256_fmadd_ps(a2, m, ad); a3 = _mm256_fmadd_ps(a3, m, ad);
            a4 = _mm256_fmadd_ps(a4, m, ad); a5 = _mm256_fmadd_ps(a5, m, ad); a6 = _mm256_fmadd_ps(a6, m, ad); a7 = _mm256_fmadd_ps(a7, m, ad);
            a8 = _mm256_fmadd_ps(a8, m, ad); a9 = _mm256_fmadd_ps(a9, m, ad);
        }
        __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)));
        tot += hsum256(_mm256_add_ps(s, _mm256_add_ps(a8, a9)));
    }
    return tot;
}
EXPORT int64_t count_nonfinite_f32(const float* a, int64_t n) { int64_t k = 0; for (int64_t i = 0; i < n; i++) if (!isfinite(a[i])) k++; return k; }
EXPORT int64_t count_nonfinite_bf16(const uint16_t* a, int64_t n) { int64_t k = 0; for (int64_t i = 0; i < n; i++) if ((a[i] & 0x7F80u) == 0x7F80u) k++; return k; }

// pin the OpenMP team: thread t -> logical CPU t*stride (stride 2 = one thread per physical core on Windows SMT
// numbering). Returns the number of threads that pinned successfully. The FMA plant verifies the effect.
EXPORT int pin_threads(int nthr, int stride) {
    int ok = 0;
#ifdef _WIN32
    #pragma omp parallel num_threads(nthr) reduction(+:ok)
    { int t = omp_get_thread_num(); DWORD_PTR m = ((DWORD_PTR)1) << (t * stride); if (SetThreadAffinityMask(GetCurrentThread(), m)) ok += 1; }
#endif
    return ok;
}
EXPORT int omp_max(void) { return omp_get_max_threads(); }
// LUT kernel self-test against scalar int (engine.c --kselftest discipline): returns mismatches
EXPORT int64_t lut_selftest(int h, int D, uint32_t seed) {
    int Dp = D / 2; uint32_t s = seed ? seed : 1;
    int8_t* w = (int8_t*)malloc((size_t)h * D); int8_t* xq = (int8_t*)malloc(D); uint8_t* codes = (uint8_t*)malloc((size_t)h * Dp);
    int8_t* lut = (int8_t*)malloc((size_t)Dp * 16); int32_t* y = (int32_t*)malloc((size_t)h * 4);
    for (int i = 0; i < h * D; i++) w[i] = (int8_t)((int)(xs32(&s) % 3) - 1);
    for (int i = 0; i < D; i++) xq[i] = (int8_t)((int)(xs32(&s) % 127) - 63);
    for (int t = 0; t < Dp; t++) for (int m = 0; m < h; m++) codes[(size_t)t * h + m] = (uint8_t)((w[(size_t)m * D + 2 * t] + 1) * 3 + (w[(size_t)m * D + 2 * t + 1] + 1));
    build_lut_t3(xq, Dp, lut); lut_rows(codes, lut, y, h, h, Dp);
    int64_t bad = 0;
    for (int m = 0; m < h; m++) { long ref = 0; for (int k = 0; k < D; k++) ref += (long)w[(size_t)m * D + k] * xq[k]; if (ref != y[m]) bad++; }
    free(w); free(xq); free(codes); free(lut); free(y);
    return bad;
}
