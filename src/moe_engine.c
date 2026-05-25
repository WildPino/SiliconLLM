#include "moe_engine.h"
#include <math.h>

void moe_init(MoeState* m, float eta, float share, int n_active) {
    m->eta      = eta;
    m->share    = share;
    m->n_active = n_active;
    m->n_steps  = 0;
    double uniform = 1.0 / n_active;
    for (int e = 0; e < MOE_N_EXPERTS; e++) {
        m->w[e]     = (e < n_active) ? uniform : 0.0;
        m->sum_w[e] = 0.0;
        m->wins[e]  = 0;
    }
}

void moe_mix(const MoeState* m, const float* const* p_experts, float* p_out) {
    for (int c = 0; c < 256; c++) {
        double acc = 0.0;
        for (int e = 0; e < m->n_active; e++)
            acc += m->w[e] * p_experts[e][c];
        p_out[c] = (float)acc;
    }
}

void moe_update(MoeState* m, const double* losses) {
    // Track wins (expert with lowest loss)
    int best_e = 0;
    for (int e = 1; e < m->n_active; e++)
        if (losses[e] < losses[best_e]) best_e = e;
    m->wins[best_e]++;

    // Exponentiated gradient descent
    for (int e = 0; e < m->n_active; e++)
        m->w[e] *= expf(-(float)(m->eta * losses[e]));

    // Normalize
    double sum = 0.0;
    for (int e = 0; e < m->n_active; e++) sum += m->w[e];
    if (sum < 1e-30) {
        double uniform = 1.0 / m->n_active;
        for (int e = 0; e < m->n_active; e++) m->w[e] = uniform;
    } else {
        for (int e = 0; e < m->n_active; e++) m->w[e] /= sum;
    }

    // Fixed-share redistribution
    double share_each = m->share / m->n_active;
    for (int e = 0; e < m->n_active; e++)
        m->w[e] = (1.0 - m->share) * m->w[e] + share_each;

    // Accumulate for average reporting
    for (int e = 0; e < m->n_active; e++)
        m->sum_w[e] += m->w[e];
    m->n_steps++;
}

double moe_avg_weight(const MoeState* m, int e) {
    return m->n_steps > 0 ? m->sum_w[e] / m->n_steps : 0.0;
}
