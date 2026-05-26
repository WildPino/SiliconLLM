#include "regime_credit.h"
#include <string.h>
#include <math.h>
#include <float.h>

void regime_credit_init(RegimeCreditState* rc) {
    memset(rc, 0, sizeof(*rc));
    rc->ema_alpha    = 1.0f / (float)CREDIT_WIN_WINDOW;
    rc->vel_alpha    = 1.0f / (float)CREDIT_VEL_WINDOW;
    rc->leading_expert = -1;
}

void regime_credit_observe(RegimeCreditState* rc,
                           const double* moe_losses,
                           const double* w,
                           int n_active,
                           const int* abs_slots) {
    /* Find the slot with minimum loss (winner for this byte) */
    int win_slot = 0;
    double min_loss = moe_losses[0];
    for (int i = 1; i < n_active; i++) {
        if (moe_losses[i] < min_loss) { min_loss = moe_losses[i]; win_slot = i; }
    }
    int win_abs = abs_slots[win_slot];

    float a = rc->ema_alpha;
    float v = rc->vel_alpha;

    /* Update win EMA: 1 for winner, 0 for everyone else */
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        float won = (i == win_slot) ? 1.0f : 0.0f;
        rc->win_ema[ae] += a * (won - rc->win_ema[ae]);
    }

    /* Update weight velocity EMA for active slots */
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        float cur  = (float)w[i];
        float dw   = cur - rc->w_prev[ae];
        rc->w_vel[ae] += v * (fabsf(dw) - rc->w_vel[ae]);
        rc->w_prev[ae] = cur;
    }

    rc->leading_expert = win_abs;
    rc->n_bytes++;
    rc->n_since_apply++;
}

void regime_credit_apply(RegimeCreditState* rc, MoeState* moe,
                         const int* abs_slots, int n_active) {
    if (rc->n_since_apply < (uint64_t)CREDIT_CHECK_EVERY) return;
    rc->n_since_apply = 0;

    if (rc->n_bytes < (uint64_t)CREDIT_WARMUP) return;

    /* Sum win_ema over active experts only */
    float win_sum = 0.0f;
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        if (ae < MOE_N_EXPERTS) win_sum += rc->win_ema[ae];
    }
    if (win_sum < 1e-9f) return;

    /* Normalized entropy of win_ema (0 = one expert wins all, 1 = uniform) */
    float entropy = 0.0f;
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        if (ae >= MOE_N_EXPERTS) continue;
        float p = rc->win_ema[ae] / win_sum;
        if (p > 1e-9f) entropy -= p * logf(p);
    }
    float max_entropy = logf((float)n_active);
    float norm_entropy = (max_entropy > 1e-9f) ? entropy / max_entropy : 1.0f;

    if (norm_entropy > CREDIT_ENTROPY_THRESH) {
        rc->n_skipped_entropy++;
        return;
    }

    /* Blend: w[i] = (1-g)*w_moe[i] + g*prior[i]   where prior = win_ema/sum */
    float g = CREDIT_GAMMA;
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        if (ae >= MOE_N_EXPERTS) continue;
        float prior = rc->win_ema[ae] / win_sum;
        moe->w[i] = (1.0f - g) * (float)moe->w[i] + g * prior;
    }

    rc->n_injections++;
}
