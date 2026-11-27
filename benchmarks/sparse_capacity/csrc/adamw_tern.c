/* Sparse Capacity, phase 4 (D-M): fused host AdamW + ternary re-quantization and 2-bit packing of the expert bank.
 *
 * One pass per row of an expert weight (the row of sc2_model.w_q: the last dim). For each row:
 *   1. AdamW exactly as torch 2.11's fused CPU kernel for float (aten/src/ATen/native/cpu/FusedAdamKernel.cpp, the
 *      vectorized ADAMW path): p *= f(1 - lr*wd); m = fmadd(f(1-b1), g - m, m); v = v*f(b2) + (f(1-b2)*g)*g;
 *      den = sqrt(v)/f(sqrt(bc2)) + f(eps); p = p + (f(-lr/bc1)*m)/den. Same operations, same rounding (AVX2 + FMA
 *      only where torch uses fmadd; compiled with -ffp-contract=off so nothing else is fused).
 *   2. While the row is in L1: s = max(mean|p|, 1e-5); q = clamp(round(p/s), -1, 1) (round half to even: |x| = 0.5
 *      gives 0); packed 4 trits per byte, code q+1 in 2 bits, little end first.
 * The row must be a multiple of 8 floats (AVX2 lanes) and of 4 trits. The scale sum runs in a fixed order (8 lanes,
 * then a pairwise reduction): it can differ from torch's mean by an ulp; the selftest bounds that and its q flips.
 *
 * Build: gcc -O2 -mavx2 -mfma -ffp-contract=off -fopenmp -shared -o adamw_tern.dll adamw_tern.c
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static inline void adamw_row(float *p, const float *g, float *m, float *v, int64_t n,
                             float wdmul, float c1, float b2, float c2, float bc2s, float eps, float negstep) {
    const __m256 vw = _mm256_set1_ps(wdmul), vc1 = _mm256_set1_ps(c1), vb2 = _mm256_set1_ps(b2),
                 vc2 = _mm256_set1_ps(c2), vbc = _mm256_set1_ps(bc2s), veps = _mm256_set1_ps(eps),
                 vst = _mm256_set1_ps(negstep);
    for (int64_t d = 0; d < n; d += 8) {
        __m256 P = _mm256_loadu_ps(p + d), G = _mm256_loadu_ps(g + d);
        P = _mm256_mul_ps(P, vw);
        __m256 M = _mm256_loadu_ps(m + d);
        M = _mm256_fmadd_ps(vc1, _mm256_sub_ps(G, M), M);                 /* lerp, weight < 0.5: base = m */
        __m256 V = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(v + d), vb2), _mm256_mul_ps(_mm256_mul_ps(vc2, G), G));
        _mm256_storeu_ps(m + d, M);
        _mm256_storeu_ps(v + d, V);
        __m256 den = _mm256_add_ps(_mm256_div_ps(_mm256_sqrt_ps(V), vbc), veps);
        P = _mm256_add_ps(P, _mm256_div_ps(_mm256_mul_ps(vst, M), den));
        _mm256_storeu_ps(p + d, P);
    }
}

static inline float absmean_row(const float *p, int64_t n) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 acc = _mm256_setzero_ps();
    for (int64_t d = 0; d < n; d += 8) acc = _mm256_add_ps(acc, _mm256_andnot_ps(sign, _mm256_loadu_ps(p + d)));
    float l[8]; _mm256_storeu_ps(l, acc);
    float s = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
    s = s / (float)n;
    return s < 1e-5f ? 1e-5f : s;
}

static inline void pack_row(const float *p, int64_t n, float s, uint8_t *out) {
    for (int64_t d = 0; d < n; d += 4) {
        uint8_t b = 0;
        for (int j = 0; j < 4; j++) {
            float x = p[d + j] / s;
            uint8_t c = x > 0.5f ? 2 : (x < -0.5f ? 0 : 1);
            b |= (uint8_t)(c << (2 * j));
        }
        out[d / 4] = b;
    }
}

/* Fused: AdamW on every row, then the scale and the packed trits of the updated row. */
EXPORT void adamw_tern(float *p, const float *g, float *m, float *v, int64_t rows, int64_t rowlen,
                       double lr, double beta1, double beta2, double wd, double eps, double bc1, double bc2,
                       uint8_t *packed, float *scale, int nthreads) {
    /* bc1 = 1 - beta1^step, bc2 = 1 - beta2^step, computed by the caller in double with the same pow as torch's */
    float wdmul = (float)(1 - lr * wd), c1 = (float)(1 - beta1), b2 = (float)beta2, c2 = (float)(1 - beta2),
          bc2s = (float)sqrt(bc2), feps = (float)eps, negstep = (float)(-(lr / bc1));
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        int64_t o = r * rowlen;
        adamw_row(p + o, g + o, m + o, v + o, rowlen, wdmul, c1, b2, c2, bc2s, feps, negstep);
        if (packed) {
            float s = absmean_row(p + o, rowlen);
            scale[r] = s;
            pack_row(p + o, rowlen, s, packed + o / 4);
        }
    }
}

/* Only the quantization and packing (for the first step, before any update, and for the tests). */
EXPORT void tern_pack(const float *p, int64_t rows, int64_t rowlen, uint8_t *packed, float *scale, int nthreads) {
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        int64_t o = r * rowlen;
        float s = absmean_row(p + o, rowlen);
        scale[r] = s;
        pack_row(p + o, rowlen, s, packed + o / 4);
    }
}
