#ifndef MOE_ENGINE_H
#define MOE_ENGINE_H

#include <stdint.h>

// Fixed-Share online Mixture of Experts.
// Four experts: SEE (static neural), UNI (unigram), BI (bigram), LZ (dictionary).
// Credit assignment via exponentiated gradient + fixed share redistribution.
// Reference: Herbster & Warmuth (1998), "Tracking the Best Expert".

#define MOE_N_EXPERTS 6
#define MOE_EXP_SEE   0
#define MOE_EXP_UNI   1
#define MOE_EXP_BI    2
#define MOE_EXP_LZ    3
#define MOE_EXP_LZ8   4   // wide-context LZ or TOKPFX
#define MOE_EXP_TOK_PREV 5 // token start transition expert

typedef struct {
    double   w[MOE_N_EXPERTS];       // current mixture weights (sum = 1)
    double   sum_w[MOE_N_EXPERTS];   // cumulative weights (for avg reporting, all steps)
    double   sum_w_elig[MOE_N_EXPERTS];  // cumulative weights only on eligible steps
    uint64_t n_elig_steps[MOE_N_EXPERTS]; // steps where expert was eligible
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

// Gated mix: only experts with eligible[i] == 1 are mixed, weights are normalized.
void moe_mix_gated(const MoeState* m, const float* const* p_experts, const uint8_t* eligible, float* p_out);

// Update weights given per-expert losses loss[n_active].
// Also tracks which expert had the minimum loss for win counting.
void moe_update(MoeState* m, const double* losses);

// Average weight for expert e over all steps so far.
double moe_avg_weight(const MoeState* m, int e);

// Average weight for expert e only on steps where it was eligible (gated mode).
double moe_avg_weight_elig(const MoeState* m, int e);

// Gated update: only experts with eligible[i] == 1 are updated.
void moe_update_gated(MoeState* m, const double* losses, const uint8_t* eligible);

#endif // MOE_ENGINE_H
