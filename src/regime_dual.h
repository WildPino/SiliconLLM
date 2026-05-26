#ifndef REGIME_DUAL_H
#define REGIME_DUAL_H

/* Research (Phase 34B): Dual-EMA Regime Router
 *
 * Core question: does the divergence between recent credit (fast EMA) and
 * long-term credit (slow EMA) signal a regime transition?
 *
 *   fast_ema[e]  = short-term win rate  (DUAL_FAST_WINDOW  ≈ 12  bytes)
 *   slow_ema[e]  = long-term  win rate  (DUAL_SLOW_WINDOW  ≈ 256 bytes)
 *   drift[e]     = fast_ema[e] - slow_ema[e]
 *
 * Z-score: z[e] = |drift[e]| / sigma[e]
 *   sigma[e] = sqrt( clamp(slow[e], 0.05, 0.95) * (1-clamp) / FAST_WINDOW )
 *   This asks: "given the long-run win rate, how surprising is the recent one?"
 *
 * Shuffled (long-run stable, no transitions):
 *   slow ≈ fast ≈ uniform → drift ≈ noise → max-z ≈ 1.7 → below thresh 2.5 ✓
 *
 * Markdown heading (BI surges for ~12 bytes, slow still reflects prose):
 *   slow[BI]≈0.15, fast[BI]→0.57, drift=0.42, sigma=0.10, z≈4.1 → triggers ✓
 *
 * Why large SLOW_WINDOW is critical: slow_ema with 256-byte window is too slow
 * to track random streaks in shuffled (needs 15+ consecutive wins to reach z≥2.5),
 * but it captures the regime that was stable for the last 256 bytes.
 *
 * No char-class features.  No hand-coded regimes.  Pure compression dynamics.
 */

#include <stdint.h>
#include <stdio.h>
#include "moe_engine.h"

/* --- tunables ------------------------------------------------------------ */
#define DUAL_SLOW_WINDOW  256     /* long-term  EMA window (bytes)           */
#define DUAL_FAST_WINDOW   12     /* short-term EMA window (bytes)           */
#define DUAL_CHECK_EVERY    8     /* apply every N bytes                     */
#define DUAL_WARMUP       512     /* need 2× slow window to stabilize        */
#define DUAL_DRIFT_THRESH  2.00f  /* min z-score to trigger                  */
                                  /* shuffled: expected max-z ≈ 1.7 (gated) */
                                  /* markdown BI-surge z ≈ 4 → triggers      */
#define DUAL_GAMMA_SCALE   0.06f  /* gamma = max_z * scale  (z=2.0→0.12)    */
#define DUAL_MAX_GAMMA     0.20f  /* hard cap on injection strength          */

/* --- state --------------------------------------------------------------- */
typedef struct {
    float slow_ema[MOE_N_EXPERTS];
    float fast_ema[MOE_N_EXPERTS];
    float slow_alpha;   /* 1 / DUAL_SLOW_WINDOW */
    float fast_alpha;   /* 1 / DUAL_FAST_WINDOW */

    uint64_t n_bytes;
    uint64_t n_since_apply;

    /* Stats */
    uint64_t n_injections;
    uint64_t n_skipped_drift;
    uint64_t n_skipped_warmup;
    float    sum_gamma;
    float    max_z_seen;
    uint64_t injection_by_expert[MOE_N_EXPERTS];
} RegimeDualState;

void regime_dual_init(RegimeDualState* rd);

/* Observe one byte: update fast_ema and slow_ema from per-slot losses.
 * Call AFTER moe_update_gated(). */
void regime_dual_observe(RegimeDualState* rd,
                         const double* moe_losses,
                         int n_active,
                         const int* abs_slots);

/* Blend drift-gated prior into MoE weights every DUAL_CHECK_EVERY bytes. */
void regime_dual_apply(RegimeDualState* rd, MoeState* moe,
                       const int* abs_slots, int n_active);

void regime_dual_print_stats(const RegimeDualState* rd, FILE* out);

#endif /* REGIME_DUAL_H */
