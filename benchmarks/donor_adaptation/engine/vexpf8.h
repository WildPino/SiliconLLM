/* vexpf8.h -- one AVX2 exponential, ONE definition.
 *
 * E53.  `expf` on this toolchain is `(float)exp((double)x)` written out --
 *   expf:  cvtss2sd ; callq <exp> ; cvtsd2ss ; retq
 * -- so every softmax element and every SwiGLU element pays a call frame, two
 * conversions, and the entire double-precision exp routine, to produce a float.
 * E51 measured the attention softmax pass at 12.16% of a token at mean position
 * 640; E9 measured the SwiGLU glue at 7.5% of a token at Coder-7B.
 *
 * This header is included by BOTH donor_engine.c and e53_vexpf8_test.c on
 * purpose.  A test that validates a COPY of the kernel validates nothing.
 *
 * Method -- Cody-Waite reduction, then a degree-5 minimax polynomial:
 *   x clamped to [-110, 95], so n = round(x*log2(e)) stays in [-159, 137]
 *   r = x - n*ln2_hi - n*ln2_lo        exact to float precision, |r| <= ln2/2
 *   e^r = 1 + r + r^2 * P5(r)          Cephes expf minimax
 *   2^n applied as 2^n1 * 2^n2, n1+n2 = n, |n1|,|n2| <= 127, multiplied IN ORDER
 *
 * Four things here are not decoration, and the self-test caught all four by
 * failing on the first two versions of this file:
 *
 *  1. CODY-WAITE, not `r = x*log2e - n`.  Scaling by log2(e) in single precision
 *     puts a relative error of ~6e-8 into y, which at y = 127 is an ABSOLUTE
 *     error of 7.6e-6, and 2^y turns that back into 5e-6 of relative error in
 *     the result.  The first version read 30 ulp at x = 88 for exactly this
 *     reason -- above the bar, at the end of the range the softmax never visits
 *     and the FFN does.
 *  2. SPLIT scaling.  Building (n+127)<<23 in one step breaks for n < -126,
 *     which is the denormal band e^x occupies between -87.3 and -104.
 *  3. The split is applied IN SEQUENCE (p*s1)*s2, never p*(s1*s2).  Forming
 *     s1*s2 first overflows to +inf for n near 128 and underflows to 0 for n
 *     near -150 -- the first version returned +inf at 1883 points where exp is
 *     perfectly finite.
 *  4. The clamp is on x, not on n.  See the note at VEXPF8_XLO: clamping n
 *     leaves r unreduced, and exp(-inf) came back as -inf.
 *
 * NO -ffast-math is involved or required (Phase 35).  The polynomial is in the
 * source, so unlike E51's flag it cannot be relocated into a callee.
 */
#ifndef VEXPF8_H
#define VEXPF8_H

#include <immintrin.h>

/* Cephes expf minimax coefficients for e^r on |r| <= ln2/2. */
#define VEXPF8_C0  1.9875691500e-4f
#define VEXPF8_C1  1.3981999507e-3f
#define VEXPF8_C2  8.3334519073e-3f
#define VEXPF8_C3  4.1665795894e-2f
#define VEXPF8_C4  1.6666665459e-1f
#define VEXPF8_C5  5.0000001201e-1f

/* ln2 split so that n*LN2_HI is EXACT in float for |n| <= 255:
   LN2_HI = 710/1024 needs 10 mantissa bits, n needs 8, the product needs 18. */
#define VEXPF8_LN2_HI   0.693359375f
#define VEXPF8_LN2_LO  -2.12194440e-4f
#define VEXPF8_LOG2E    1.44269504088896341f
/* Clamp x, NOT n.  Clamping n leaves r = x - n*ln2 unreduced -- feed -inf in
   and n saturates while r stays -inf, and the polynomial returns -inf where
   exp returns +0.  That is what the self-test caught.  Clamping x instead keeps
   n inside [-159, 137] by construction, so r is ALWAYS in [-ln2/2, ln2/2] and
   the split scaling never needs a limit of its own.
     x <= -110 -> e^x underflows to +0 anyway (the smallest denormal is e^-103.97)
     x >=   95 -> e^x overflowed at 88.73                                        */
#define VEXPF8_XLO   -110.0f
#define VEXPF8_XHI     95.0f

/* 2^k as a vector of floats, for |k| <= 127.  Builds the exponent field. */
static inline __m256 vexpf8_pow2i(__m256i k)
{
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(k, _mm256_set1_epi32(127)), 23));
}

/* e^x on eight floats.  G-E53a/b are the gates that say whether this is right. */
static inline __m256 vexpf8(__m256 x)
{
    __m256 xc = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(VEXPF8_XLO)),
                              _mm256_set1_ps(VEXPF8_XHI));

    __m256 n = _mm256_round_ps(_mm256_mul_ps(xc, _mm256_set1_ps(VEXPF8_LOG2E)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(VEXPF8_LN2_HI), xc);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(VEXPF8_LN2_LO), r);

    __m256 p = _mm256_set1_ps(VEXPF8_C0);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(VEXPF8_C1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(VEXPF8_C2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(VEXPF8_C3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(VEXPF8_C4));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(VEXPF8_C5));
    /* e^r = 1 + r + r^2 * P(r) */
    p = _mm256_fmadd_ps(_mm256_mul_ps(r, r), p, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));

    __m256i ni = _mm256_cvtps_epi32(n);
    __m256i n1 = _mm256_srai_epi32(ni, 1);
    __m256i n2 = _mm256_sub_epi32(ni, n1);
    /* IN SEQUENCE.  p*(s1*s2) overflows and underflows where p*s1*s2 does not. */
    __m256 out = _mm256_mul_ps(_mm256_mul_ps(p, vexpf8_pow2i(n1)), vexpf8_pow2i(n2));

    /* NaN in, NaN out.  The min/max clamp on x would have swallowed it. */
    __m256 isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    return _mm256_blendv_ps(out, x, isnan);
}

/* ---- G-E53a's planted control.  The SAME kernel with the polynomial truncated
   from degree 5 to degree 2.  It must FAIL the bar; if it does not, the harness
   cannot see a bad exponential and nothing it says about the good one counts.
   Everything else -- the reduction, the clamp, the split scaling, the NaN path
   -- is identical, so what the control tests is the POLYNOMIAL and not some
   unrelated breakage that would fire for the wrong reason. ---- */
static inline __m256 vexpf8_degraded(__m256 x)
{
    __m256 xc = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(VEXPF8_XLO)),
                              _mm256_set1_ps(VEXPF8_XHI));

    __m256 n = _mm256_round_ps(_mm256_mul_ps(xc, _mm256_set1_ps(VEXPF8_LOG2E)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(VEXPF8_LN2_HI), xc);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(VEXPF8_LN2_LO), r);

    __m256 p = _mm256_set1_ps(VEXPF8_C3);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(VEXPF8_C4));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(VEXPF8_C5));
    p = _mm256_fmadd_ps(_mm256_mul_ps(r, r), p, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));

    __m256i ni = _mm256_cvtps_epi32(n);
    __m256i n1 = _mm256_srai_epi32(ni, 1);
    __m256i n2 = _mm256_sub_epi32(ni, n1);
    __m256 out = _mm256_mul_ps(_mm256_mul_ps(p, vexpf8_pow2i(n1)), vexpf8_pow2i(n2));

    __m256 isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    return _mm256_blendv_ps(out, x, isnan);
}

#endif /* VEXPF8_H */
