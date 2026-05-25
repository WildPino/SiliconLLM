#include "moe_engine.h"
#include <math.h>

void moe_init(MoeState* m, float eta, float share, int n_active) {
    m->eta      = eta;
    m->share    = share;
    m->n_active = n_active;
    m->n_steps  = 0;
    double uniform = 1.0 / n_active;
    for (int e = 0; e < MOE_N_EXPERTS; e++) {
        m->w[e]          = (e < n_active) ? uniform : 0.0;
        m->sum_w[e]      = 0.0;
        m->sum_w_elig[e] = 0.0;
        m->n_elig_steps[e] = 0;
        m->wins[e]       = 0;
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

void moe_mix_gated(const MoeState* m, const float* const* p_experts, const uint8_t* eligible, float* p_out) {
    if (!eligible) {
        moe_mix(m, p_experts, p_out);
        return;
    }
    double sum_w = 0.0;
    for (int e = 0; e < m->n_active; e++) {
        if (eligible[e]) sum_w += m->w[e];
    }
    double inv_sum = sum_w > 1e-30 ? 1.0 / sum_w : 0.0;
    for (int c = 0; c < 256; c++) {
        double acc = 0.0;
        for (int e = 0; e < m->n_active; e++) {
            if (eligible[e]) acc += (m->w[e] * inv_sum) * p_experts[e][c];
        }
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

// Returns average weight only on steps where expert e was eligible.
double moe_avg_weight_elig(const MoeState* m, int e) {
    return m->n_elig_steps[e] > 0 ? m->sum_w_elig[e] / m->n_elig_steps[e] : 0.0;
}

void moe_update_gated(MoeState* m, const double* losses, const uint8_t* eligible) {
    if (!eligible) {
        moe_update(m, losses);
        return;
    }

    int best_e = -1;
    for (int e = 0; e < m->n_active; e++) {
        if (eligible[e]) {
            if (best_e == -1 || losses[e] < losses[best_e]) best_e = e;
        }
    }
    if (best_e != -1) m->wins[best_e]++;

    double sum_eligible = 0.0;
    int n_eligible = 0;
    for (int e = 0; e < m->n_active; e++) {
        if (eligible[e]) {
            sum_eligible += m->w[e];
            n_eligible++;
        }
    }

    if (sum_eligible < 1e-30 || n_eligible == 0) {
        m->n_steps++;
        return;
    }

    for (int e = 0; e < m->n_active; e++) {
        if (eligible[e]) {
            m->w[e] *= expf(-(float)(m->eta * losses[e]));
        }
    }

    double sum_new_eligible = 0.0;
    for (int e = 0; e < m->n_active; e++) {
        if (eligible[e]) sum_new_eligible += m->w[e];
    }

    if (sum_new_eligible < 1e-30) {
        double uniform = sum_eligible / n_eligible;
        for (int e = 0; e < m->n_active; e++) {
            if (eligible[e]) m->w[e] = uniform;
        }
    } else {
        double scale = sum_eligible / sum_new_eligible;
        for (int e = 0; e < m->n_active; e++) {
            if (eligible[e]) m->w[e] *= scale;
        }
    }

    double share_each = (m->share * sum_eligible) / n_eligible;
    for (int e = 0; e < m->n_active; e++) {
        if (eligible[e]) {
            m->w[e] = (1.0 - m->share) * m->w[e] + share_each;
        }
    }

    for (int e = 0; e < m->n_active; e++) {
        m->sum_w[e] += m->w[e];
        if (eligible[e]) {
            m->sum_w_elig[e] += m->w[e];
            m->n_elig_steps[e]++;
        }
    }
    m->n_steps++;
}
