/* Shared pinned no-FMA SSE2 SwiGLU semantics for STRAT-01. */
#ifndef STRAT01_SWIGLU_SSE2_H
#define STRAT01_SWIGLU_SSE2_H

#define STRAT01_SWIGLU_SSE2_REFERENCE_REVISION "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
#define STRAT01_SWIGLU_SSE2_VEC_CPP_SHA "a946fee202dfe4528453865a6402d13a004e31ea73e88c3586793bbcc05994f7"
#define STRAT01_SWIGLU_SSE2_VEC_H_SHA "8817801355b20079de39fd4c67c7453ca2033cdb69fc5bd1f71bb66f12f57318"

/* Exact operation ordering of the pinned GGML_CPU_GENERIC four-lane path.
 * Do not contract these add/multiply pairs into FMA operations. */
static __m128 strat01_sse2_expf_no_fma(__m128 x) {
    const __m128 r = _mm_set1_ps(0x1.8p23f);
    const __m128 z = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(0x1.715476p+0f)), r);
    const __m128 n = _mm_sub_ps(z, r);
    const __m128 inner = _mm_sub_ps(x, _mm_mul_ps(n, _mm_set1_ps(0x1.62e4p-1f)));
    const __m128 b = _mm_sub_ps(inner, _mm_mul_ps(n, _mm_set1_ps(0x1.7f7d1cp-20f)));
    const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    const __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    const __m128i c = _mm_castps_si128(
        _mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), n), _mm_set1_ps(126.0f)));
    const __m128 u = _mm_mul_ps(b, b);
    const __m128 p0 = _mm_add_ps(
        _mm_mul_ps(_mm_set1_ps(0x1.0e4020p-7f), b),
        _mm_set1_ps(0x1.573e2ep-5f));
    const __m128 p1 = _mm_add_ps(
        _mm_mul_ps(_mm_set1_ps(0x1.555e66p-3f), b),
        _mm_set1_ps(0x1.fffdb6p-2f));
    const __m128 p2 = _mm_add_ps(_mm_mul_ps(p0, u), p1);
    const __m128 j = _mm_add_ps(
        _mm_mul_ps(p2, u),
        _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_epi8(c)) {
        return _mm_add_ps(_mm_mul_ps(j, k), k);
    }
    {
        const __m128i g = _mm_and_si128(
            _mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
            _mm_set1_epi32((int)0x82000000u));
        const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000)));
        const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
        const __m128i d = _mm_castps_si128(
            _mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), n), _mm_set1_ps(192.0f)));
        const __m128 large = _mm_mul_ps(s1, s1);
        const __m128 mid = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(s2, j), s2), s1);
        const __m128 normal = _mm_add_ps(_mm_mul_ps(k, j), k);
        return _mm_or_ps(
            _mm_and_ps(_mm_castsi128_ps(d), large),
            _mm_andnot_ps(
                _mm_castsi128_ps(d),
                _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), mid),
                          _mm_andnot_ps(_mm_castsi128_ps(c), normal))));
    }
}

static __m128 strat01_sse2_silu_no_fma(__m128 x) {
    const __m128 neg_x = _mm_sub_ps(_mm_setzero_ps(), x);
    return _mm_div_ps(x, _mm_add_ps(_mm_set1_ps(1.0f), strat01_sse2_expf_no_fma(neg_x)));
}

static int strat01_sse2_swiglu_compute(
        const float *gate, const float *up, float *out, uint64_t count,
        char error[256]) {
    if (!gate || !up || !out) {
        snprintf(error, 256, "SSE2 SwiGLU null buffer");
        return 0;
    }
    if ((count & 3U) != 0U) {
        snprintf(error, 256, "SSE2 SwiGLU count has scalar tail");
        return 0;
    }
    for (uint64_t i = 0; i < count; ++i) {
        if (!isfinite(gate[i]) || !isfinite(up[i])) {
            snprintf(error, 256, "SSE2 SwiGLU non-finite input");
            return 0;
        }
    }
    for (uint64_t i = 0; i < count; i += 4U) {
        const __m128 g = _mm_loadu_ps(gate + i);
        const __m128 u = _mm_loadu_ps(up + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(strat01_sse2_silu_no_fma(g), u));
    }
    for (uint64_t i = 0; i < count; ++i) {
        if (!isfinite(out[i])) {
            snprintf(error, 256, "SSE2 SwiGLU non-finite output");
            return 0;
        }
    }
    return 1;
}

#endif /* STRAT01_SWIGLU_SSE2_H */
