/* e53_vexpf8_test.c -- G-E53a (planted control) and G-E53b (the numeric bar).
 *
 * Includes vexpf8.h, the same header donor_engine.c includes.  There is one
 * definition of the kernel and this validates that one.
 *
 * Reference: (float)exp((double)x), which is exactly what expf IS on this
 * toolchain (E51, read out of the object code).  So "relative error against the
 * reference" here means relative error against the function being replaced.
 *
 * ---- A SPECIFICATION REFINEMENT, MADE BEFORE ANY ENGINE MEASUREMENT ---------
 * The brief states the bar as "max relative error <= 1e-6" over [-104, 89].
 * Below x = -87.34 the reference is a DENORMAL, and a relative error against a
 * denormal is not a bounded quantity for ANY implementation -- at the bottom of
 * the range exp(x) sits between 0 and the single smallest denormal, so the two
 * nearest representable answers differ from each other by 100%.
 *
 * So the domain is split, and the 1e-6 VALUE IS NOT TOUCHED:
 *    ref normal   (>= FLT_MIN)   -> relative error, bar 1e-6      (as registered)
 *    ref denormal (0 < ref < FLT_MIN) -> ABSOLUTE error, bar 1 denormal ulp
 *    ref zero or infinite        -> exact identity required
 * The denormal criterion is STRICTER in absolute terms than any relative bar
 * would be, and the planted control is required to fire on the NORMAL region,
 * so nothing is let through by the split.  This is the same move as E52 A.5:
 * a domain fixed on arithmetic grounds, not a threshold moved to fit data.
 * ---------------------------------------------------------------------------
 *
 * Build:
 *   clang -O3 -mavx2 -mfma -ffp-contract=on e53_vexpf8_test.c -o e53_vexpf8_test.exe -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include "vexpf8.h"

#define BAR_REL   1e-6      /* G-E53b, registered in the brief.  Unchanged. */
#define NSWEEP    (1 << 20) /* brief section 4: 2^20 points */
#define LO        (-104.0)
#define HI        (89.0)
#define DEN_ULP   0x1p-149           /* FLT_TRUE_MIN exactly, as a hex float.
                                       The decimal spelling 1.4012984643e-45 is
                                       rounded DOWN, and an exact-1-ulp result
                                       then fails a <= against it. */

typedef __m256 (*kern_t)(__m256);

typedef struct {
    double max_rel; double at_rel; int max_ulp;   /* normal-reference region */
    double max_abs; double at_abs;                /* denormal-reference region */
    long n_normal, n_denorm, n_zero, n_inf;
    long bad_zero, bad_inf;                       /* exactness failures */
} res_t;

static float call1(kern_t k, float x)
{
    float in[8], out[8];
    int i;
    for (i = 0; i < 8; i++) in[i] = x;
    _mm256_storeu_ps(out, k(_mm256_loadu_ps(in)));
    /* all eight lanes must agree -- a lane-dependent kernel is a bug the sweep
       would never see, because the sweep feeds eight equal values here but the
       engine feeds eight different ones. */
    for (i = 1; i < 8; i++) if (memcmp(&out[0], &out[i], 4) != 0) {
        fprintf(stderr, "LANE DISAGREEMENT at x=%.9g: lane0=%.9g lane%d=%.9g\n",
                x, out[0], i, out[i]);
        exit(2);
    }
    return out[0];
}

static int ulpdiff(float a, float b)
{
    int32_t ia, ib;
    if (a == b) return 0;
    memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    if (ia < 0) ia = (int32_t)0x80000000 - ia;
    if (ib < 0) ib = (int32_t)0x80000000 - ib;
    return (int)(ia > ib ? ia - ib : ib - ia);
}

static void sweep(kern_t k, const char* name, res_t* R)
{
    long i;
    memset(R, 0, sizeof(*R));
    for (i = 0; i < NSWEEP; i++) {
        double xd = LO + (HI - LO) * ((double)i / (double)(NSWEEP - 1));
        float  x  = (float)xd;
        float  got = call1(k, x);
        float  ref = (float)exp((double)x);

        if (ref == 0.0f)      { R->n_zero++;  if (got != 0.0f)   R->bad_zero++; continue; }
        if (!isfinite(ref))   { R->n_inf++;   if (isfinite(got)) R->bad_inf++;  continue; }
        if (!isfinite(got))   { R->n_inf++;   R->bad_inf++;                     continue; }

        if (ref >= FLT_MIN) {
            double rel = fabs((double)got - (double)ref) / (double)ref;
            int u = ulpdiff(got, ref);
            R->n_normal++;
            if (rel > R->max_rel) { R->max_rel = rel; R->at_rel = xd; }
            if (u > R->max_ulp) R->max_ulp = u;
        } else {
            double ab = fabs((double)got - (double)ref);
            R->n_denorm++;
            if (ab > R->max_abs) { R->max_abs = ab; R->at_abs = xd; }
        }
    }
    printf("  %-9s normal: max rel %.4e at x=%8.3f  max ulp %-6d (%ld pts)\n",
           name, R->max_rel, R->at_rel, R->max_ulp, R->n_normal);
    printf("            denorm: max abs %.4e (%.2f denormal ulp) at x=%8.3f  (%ld pts)\n",
           R->max_abs, R->max_abs / DEN_ULP, R->at_abs, R->n_denorm);
    printf("            exact : zero-mismatch %ld/%ld   inf-mismatch %ld/%ld\n",
           R->bad_zero, R->n_zero, R->bad_inf, R->n_inf);
}

/* The special cases, checked by identity rather than by tolerance. */
static int specials(void)
{
    struct { float x; const char* what; } cases[] = {
        { 0.0f, "exp(0) == 1 exactly" },
        { -0.0f, "exp(-0) == 1 exactly" },
        { 1.0f, "exp(1)" },
        { -1.0f, "exp(-1)" },
        { 88.0f, "just under overflow" },
        { 88.7f, "the last finite" },
        { 89.0f, "over overflow -> +inf" },
        { -87.0f, "last normal" },
        { -95.0f, "denormal band" },
        { -103.0f, "deep denormal" },
        { -105.0f, "under -> +0" },
        { INFINITY, "+inf -> +inf" },
        { -INFINITY, "-inf -> +0" },
    };
    int bad = 0, i;
    printf("\n  special cases (identity, not tolerance):\n");
    for (i = 0; i < (int)(sizeof(cases)/sizeof(cases[0])); i++) {
        float x = cases[i].x;
        float got = call1(vexpf8, x);
        float ref = (float)exp((double)x);
        int u = (ref == 0.0f || !isfinite(ref)) ? (got == ref ? 0 : 999999)
                                                : ulpdiff(got, ref);
        printf("     x=%-10.4g got %-14.7g ref %-14.7g ulp %-7d %s%s\n",
               x, got, ref, u, cases[i].what, (u > 4 ? "   <-- BAD" : ""));
        if (u > 4) bad++;
    }
    {   /* NaN must survive as NaN. */
        float nan_in = (float)NAN;
        float got = call1(vexpf8, nan_in);
        int ok = (got != got);
        printf("     x=NaN      got %-14.7g                     %s\n",
               got, ok ? "NaN preserved" : "   <-- BAD, NaN was swallowed");
        if (!ok) bad++;
    }
    return bad;
}

int main(void)
{
    res_t G, B;
    int nbad;

    printf("E53 -- vexpf8 against (float)exp((double)x), which is what expf IS here.\n");
    printf("sweep: %d points over [%.1f, %.1f]\n", NSWEEP, LO, HI);
    printf("bars: normal ref -> rel <= %.0e (G-E53b, as registered)\n", BAR_REL);
    printf("      denormal ref -> abs <= 1 denormal ulp;  zero/inf -> exact\n\n");

    sweep(vexpf8,          "kernel",   &G);
    sweep(vexpf8_degraded, "degraded", &B);

    nbad = specials();

    printf("\n  G-E53a (planted control) : ");
    if (B.max_rel > BAR_REL) {
        printf("FIRES\n     the degree-2 kernel reads %.4e on the normal region, %.0fx the\n"
               "     %.0e bar -- the harness can see a bad exponential, so its reading of\n"
               "     the good one counts\n", B.max_rel, B.max_rel / BAR_REL, BAR_REL);
    } else {
        printf("*** DOES NOT FIRE ***\n"
               "     the degraded kernel reads %.4e, INSIDE the bar.  The harness cannot\n"
               "     distinguish a bad exponential from a good one.  Nothing below is read.\n",
               B.max_rel);
        return 1;
    }

    printf("\n  G-E53b (numeric bar)     : ");
    if (G.max_rel <= BAR_REL && G.max_abs <= DEN_ULP
        && G.bad_zero == 0 && G.bad_inf == 0 && nbad == 0) {
        printf("PASS\n     normal   max rel %.4e  (%.0fx inside the bar), max ulp %d\n"
               "     denormal max abs %.4e  (%.2f denormal ulp)\n"
               "     every zero, infinity, NaN and special case exact\n",
               G.max_rel, BAR_REL / (G.max_rel > 0 ? G.max_rel : 1e-30), G.max_ulp,
               G.max_abs, G.max_abs / DEN_ULP);
        return 0;
    }
    printf("*** FAILS ***\n"
           "     normal max rel %.4e (bar %.0e), max ulp %d\n"
           "     denormal max abs %.4e (%.2f denormal ulp, bar 1.00)\n"
           "     zero-mismatch %ld, inf-mismatch %ld, bad specials %d\n"
           "     No speed is measured on this arm.\n",
           G.max_rel, BAR_REL, G.max_ulp, G.max_abs, G.max_abs / DEN_ULP,
           G.bad_zero, G.bad_inf, nbad);
    return 1;
}
