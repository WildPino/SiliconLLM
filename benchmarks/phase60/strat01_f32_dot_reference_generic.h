/* Exact GGML_CPU_GENERIC F32 dot semantics for pinned STRAT-01 reference. */
#ifndef STRAT01_F32_DOT_REFERENCE_GENERIC_H
#define STRAT01_F32_DOT_REFERENCE_GENERIC_H

#if defined(__clang__) || defined(__GNUC__)
#define STRAT01_F32_GENERIC_ATTR __attribute__((target("no-avx,no-avx2,no-fma"), noinline))
#else
#define STRAT01_F32_GENERIC_ATTR
#endif

STRAT01_F32_GENERIC_ATTR
static float strat01_f32_dot_reference_generic(
        const float *x, const float *y, unsigned count) {
    double sum = 0.0;
    for (unsigned i = 0; i < count; ++i) {
        const float product = x[i] * y[i];
        sum += (double) product;
    }
    return (float) sum;
}

#undef STRAT01_F32_GENERIC_ATTR
#endif
