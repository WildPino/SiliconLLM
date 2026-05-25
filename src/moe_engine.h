#ifndef MOE_ENGINE_H
#define MOE_ENGINE_H

#include <stdint.h>

// Fixed-Share online Mixture of Experts.
// Four experts: SEE (static neural), UNI (unigram), BI (bigram), LZ (dictionary).
// Credit assignment via exponentiated gradient + fixed share redistribution.
// Reference: Herbster & Warmuth (1998), "Tracking the Best Expert".

#define MOE_N_EXPERTS 4
#define MOE_EXP_SEE   0
#define MOE_EXP_UNI   1
#define MOE_EXP_BI    2
#define MOE_EXP_LZ    3

typedef struct {
    double   w[MOE_N_EXPERTS];       // current mixture weights (sum = 1)
    double   sum_w[MOE_N_EXPERTS];   // cumulative weights (for avg reporting)
    uint64_t wins[MOE_N_EXPERTS];    // per-expert "best predictor" count
    uint64_t n_steps;

    float    eta;     // learning rate (default 0.03)
    float    share;   // fixed-share parameter (default 0.001)
    int      n_active; // 3 (no-lz) or 4
} MoeState;

// Initialize with given eta and share. n_active=3 for --no-lz, 4 otherwise.
void moe_init(MoeState* m, float eta, float share, int n_active);

// Mix p_experts[n_active][256] into p_out[256] using current weights.
void moe_mix(const MoeState* m, const float* const* p_experts, float* p_out);

// Update weights given per-expert losses loss[n_active].
// Also tracks which expert had the minimum loss for win counting.
void moe_update(MoeState* m, const double* losses);

// Average weight for expert e over all steps so far.
double moe_avg_weight(const MoeState* m, int e);

#endif // MOE_ENGINE_H
