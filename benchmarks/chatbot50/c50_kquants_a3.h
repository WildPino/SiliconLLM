/*
 * C50-A3 kernel additions: Q5_K x Q8_K dots and row dequantization for the embedding.
 * Kept apart from c50_kquants.h so the C50-A1 binary and its recorded sources stay unchanged.
 */
#ifndef C50_KQUANTS_A3_H
#define C50_KQUANTS_A3_H

#include "c50_kquants.h"

/* ---------------------------------------------------------------- Q5_K (A3) --
 * block_q5_K (176 B): d, dmin (fp16), scales[12] (6-bit scales/mins), qh[32], qs[128].
 * Generic and AVX2 dots transcribed from llama.cpp 5b335f4
 * ggml/src/ggml-cpu/quants.c ggml_vec_dot_q5_K_q8_K_generic and
 * ggml/src/ggml-cpu/arch/x86/quants.c ggml_vec_dot_q5_K_q8_K (__AVX2__ branch). MIT. */
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } c50_block_q5_k;
typedef char c50_q5_k_is_176[sizeof(c50_block_q5_k) == 176 ? 1 : -1];

static float c50_q5_k_dot_generic(const c50_block_q5_k *x, const strat01_q8_k_block *y, unsigned n) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t *)&utmp[0], *mins = (const uint8_t *)&utmp[2];
    int8_t aux8[256];
    int16_t aux16[8];
    float sums[8] = {0}, sumf = 0;
    int32_t aux32[8];
    unsigned i, nb = n / 256;
    int j, l;
    for (i = 0; i < nb; ++i) {
        const uint8_t *q4 = x[i].qs, *hm = x[i].qh;
        const int8_t *q8 = y[i].qs;
        int8_t *a = aux8;
        uint8_t m = 1;
        int sumi = 0, is = 0;
        uint32_t uaux;
        float d, dmin;
        memset(aux32, 0, sizeof aux32);
        for (j = 0; j < 256 / 64; ++j) {
            for (l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            for (l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        for (j = 0; j < 256 / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        for (j = 0; j < 256 / 32; ++j) {
            int32_t scale = scales[is++];
            int r;
            for (r = 0; r < 4; ++r) {
                for (l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        d = c50_h2f(x[i].d) * y[i].d;
        for (l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        dmin = c50_h2f(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

static inline __m256i c50_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

static float c50_q5_k_dot_avx2(const c50_block_q5_k *x, const strat01_q8_k_block *y, unsigned n) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF), mone = _mm256_set1_epi8(1);
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;
    uint32_t utmp[4];
    unsigned i, nb = n / 256;
    for (i = 0; i < nb; ++i) {
        const uint8_t *q5 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const float d = y[i].d * c50_h2f(x[i].d);
        const float dmin = -y[i].d * c50_h2f(x[i].dmin);
        uint32_t uaux;
        __m256i mins_and_scales, q8sums, scales, hbits, hmask, sumi;
        __m128i q8s, prod, hsum, sc128;
        int j, bit = 0;
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));
        q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);
        sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        scales = _mm256_set_m128i(sc128, sc128);
        hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
        hmask = mone;
        sumi = _mm256_setzero_si256();
        for (j = 0; j < 256 / 64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, c50_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, c50_scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256((const __m256i *)q5);
            __m256i q5l_0, q5h_0, q5_0, q5l_1, q5h_1, q5_1, q8_0, q8_1, p16_0, p16_1;
            q5 += 32;
            q5l_0 = _mm256_and_si256(q5bits, m4);
            q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
            q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return c50_hsum_float_8(acc) + summs;
}

/* ------------------------------------------------- row dequantization (A3 embedding) --
 * Transcribed from llama.cpp 5b335f4 ggml/src/ggml-quants.c dequantize_row_{q4_K,q5_K,q6_K,q8_0}. */
static void c50_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

static void c50_dequant_q4_k(const uint8_t *row, float *y, unsigned k) {
    unsigned i, j, l;
    for (i = 0; i < k / 256; ++i) {
        const uint8_t *b = row + (size_t)i * 144, *q = b + 16;
        const float d = c50_h2f((uint16_t)(b[0] | b[1] << 8)), min = c50_h2f((uint16_t)(b[2] | b[3] << 8));
        int is = 0;
        uint8_t sc, m;
        for (j = 0; j < 256; j += 64) {
            float d1, m1, d2, m2;
            c50_get_scale_min_k4(is + 0, b + 4, &sc, &m); d1 = d * sc; m1 = min * m;
            c50_get_scale_min_k4(is + 1, b + 4, &sc, &m); d2 = d * sc; m2 = min * m;
            for (l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static void c50_dequant_q5_k(const uint8_t *row, float *y, unsigned k) {
    unsigned i, j, l;
    for (i = 0; i < k / 256; ++i) {
        const c50_block_q5_k *x = (const c50_block_q5_k *)(row + (size_t)i * 176);
        const uint8_t *ql = x->qs, *qh = x->qh;
        const float d = c50_h2f(x->d), min = c50_h2f(x->dmin);
        int is = 0;
        uint8_t sc, m, u1 = 1, u2 = 2;
        for (j = 0; j < 256; j += 64) {
            float d1, m1, d2, m2;
            c50_get_scale_min_k4(is + 0, x->scales, &sc, &m); d1 = d * sc; m1 = min * m;
            c50_get_scale_min_k4(is + 1, x->scales, &sc, &m); d2 = d * sc; m2 = min * m;
            for (l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}

static void c50_dequant_q6_k(const uint8_t *row, float *y, unsigned k) {
    unsigned i, n, l;
    for (i = 0; i < k / 256; ++i) {
        const uint8_t *b = row + (size_t)i * 210, *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        const float d = c50_h2f((uint16_t)(b[208] | b[209] << 8));
        for (n = 0; n < 256; n += 128) {
            for (l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

static void c50_dequant_q8_0(const uint8_t *row, float *y, unsigned k) {
    unsigned i, j;
    for (i = 0; i < k / 32; ++i) {
        const c50_block_q8_0 *x = (const c50_block_q8_0 *)(row + (size_t)i * 34);
        const float d = c50_h2f(x->d);
        for (j = 0; j < 32; ++j) y[i * 32 + j] = x->qs[j] * d;
    }
}

#endif
