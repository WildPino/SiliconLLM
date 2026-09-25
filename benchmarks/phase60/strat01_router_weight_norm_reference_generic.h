#ifndef STRAT01_ROUTER_WEIGHT_NORM_REFERENCE_GENERIC_H
#define STRAT01_ROUTER_WEIGHT_NORM_REFERENCE_GENERIC_H

#include <stddef.h>

#if defined(__clang__) || defined(__GNUC__)
__attribute__((noinline, target("no-avx,no-avx2,no-fma")))
#endif
static void strat01_router_weight_norm_reference_generic(
        const float weights[4], float normalized[4]) {
    double sum = 0.0;
    for (size_t i = 0; i < 4U; ++i) sum += (double) weights[i];
    float denominator = (float) sum;
    if (denominator < 6.103515625e-5f) denominator = 6.103515625e-5f;
    for (size_t i = 0; i < 4U; ++i) normalized[i] = weights[i] / denominator;
}

#endif
