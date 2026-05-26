#include "regime_dual.h"
#include <string.h>
#include <math.h>
#include <inttypes.h>

void regime_dual_init(RegimeDualState* rd) {
    memset(rd, 0, sizeof(*rd));
    rd->slow_alpha = 1.0f / (float)DUAL_SLOW_WINDOW;
    rd->fast_alpha = 1.0f / (float)DUAL_FAST_WINDOW;
}

void regime_dual_observe(RegimeDualState* rd,
                         const double* moe_losses,
                         int n_active,
                         const int* abs_slots) {
    int win_slot = 0;
    double min_loss = moe_losses[0];
    for (int i = 1; i < n_active; i++) {
        if (moe_losses[i] < min_loss) { min_loss = moe_losses[i]; win_slot = i; }
    }

    float sa = rd->slow_alpha;
    float fa = rd->fast_alpha;
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        float won = (i == win_slot) ? 1.0f : 0.0f;
        rd->slow_ema[ae] += sa * (won - rd->slow_ema[ae]);
        rd->fast_ema[ae] += fa * (won - rd->fast_ema[ae]);
    }

    rd->n_bytes++;
    rd->n_since_apply++;
}

void regime_dual_apply(RegimeDualState* rd, MoeState* moe,
                       const int* abs_slots, int n_active) {
    if (rd->n_since_apply < (uint64_t)DUAL_CHECK_EVERY) return;
    rd->n_since_apply = 0;

    if (rd->n_bytes < (uint64_t)DUAL_WARMUP) {
        rd->n_skipped_warmup++;
        return;
    }

    /* z = |fast - slow| / sigma_slow
     * sigma_slow[e] = sqrt( clamp(slow[e], 0.05, 0.95) * (1-clamp) / FAST_WINDOW )
     * This asks: given the long-run rate (slow), is the recent rate (fast) surprising?
     * For shuffled (slow≈fast≈uniform): z ≈ N(0,1) → max over n ≈ 1.7 < 2.5. */
    float max_z = 0.0f;
    int   max_z_expert = -1;
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        if (ae >= MOE_N_EXPERTS) continue;
        float p   = rd->slow_ema[ae];
        if (p < 0.05f) p = 0.05f;
        if (p > 0.95f) p = 0.95f;
        float sig = sqrtf(p * (1.0f - p) / (float)DUAL_FAST_WINDOW);
        float d   = rd->fast_ema[ae] - rd->slow_ema[ae];
        float z   = (d < 0.0f ? -d : d) / (sig + 1e-6f);
        if (z > max_z) { max_z = z; max_z_expert = ae; }
    }

    if (max_z > rd->max_z_seen) rd->max_z_seen = max_z;

    if (max_z < DUAL_DRIFT_THRESH) {
        rd->n_skipped_drift++;
        return;
    }

    float gamma = max_z * DUAL_GAMMA_SCALE;
    if (gamma > DUAL_MAX_GAMMA) gamma = DUAL_MAX_GAMMA;

    float fsum = 0.0f;
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        if (ae < MOE_N_EXPERTS) fsum += rd->fast_ema[ae];
    }
    if (fsum < 1e-9f) return;

    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        if (ae >= MOE_N_EXPERTS) continue;
        float prior = rd->fast_ema[ae] / fsum;
        moe->w[i] = (1.0f - gamma) * (float)moe->w[i] + gamma * prior;
    }

    rd->n_injections++;
    rd->sum_gamma += gamma;
    if (max_z_expert >= 0 && max_z_expert < MOE_N_EXPERTS)
        rd->injection_by_expert[max_z_expert]++;
}

void regime_dual_print_stats(const RegimeDualState* rd, FILE* out) {
    if (rd->n_injections == 0 && rd->n_skipped_drift == 0) return;

    float avg_gamma = (rd->n_injections > 0)
                    ? rd->sum_gamma / (float)rd->n_injections : 0.0f;
    uint64_t total_checks = rd->n_injections + rd->n_skipped_drift + rd->n_skipped_warmup;

    fprintf(out,
        "Regime (dual): inject=%"PRIu64"  skip_drift=%"PRIu64"  skip_wup=%"PRIu64
        "  total_chk=%"PRIu64"  avg_gamma=%.3f  max_z=%.2f\n",
        rd->n_injections, rd->n_skipped_drift, rd->n_skipped_warmup,
        total_checks, avg_gamma, rd->max_z_seen);

    const char* names[MOE_N_EXPERTS] = { "SEE","UNI","BI","LZ","TOKPFX","TOKPREV","SPAN" };
    fprintf(out, "  inject_by_expert:");
    for (int e = 0; e < MOE_N_EXPERTS; e++)
        if (rd->injection_by_expert[e] > 0)
            fprintf(out, "  %s=%"PRIu64, names[e], rd->injection_by_expert[e]);
    fprintf(out, "\n");
}
