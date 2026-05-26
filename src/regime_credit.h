#ifndef REGIME_CREDIT_H
#define REGIME_CREDIT_H

/* Research (Phase 34): Credit-Only Regime Prior Router
 *
 * Replaces hand-coded char-class thresholds (regime_prior.c) with pure
 * compression credit:
 *   - win_ema[e]: rolling fraction of bytes where expert e had minimum loss
 *   - w_vel[e]:   rolling magnitude of weight change (convergence signal)
 *
 * Prior = normalized win_ema, injected only when entropy(win_ema) is low —
 * meaning one expert is clearly dominating recent predictions.
 *
 * No char-class features.  No hand-coded thresholds.  Pure silicon.
 *
 * Schema:
 *   losses[slot] → argmin → win indicator → win_ema[expert]
 *   win_ema → entropy check → normalized prior → MoE blend
 *
 * Integration: same call sites as regime_prior; swap the two pairs:
 *   regime_credit_observe() after moe_update_gated()
 *   regime_credit_apply()   (can be merged into observe or kept separate)
 */

#include <stdint.h>
#include "moe_engine.h"

/* --- tunables ------------------------------------------------------------ */
#define CREDIT_WIN_WINDOW     64     /* EMA window for per-expert win rate   */
#define CREDIT_VEL_WINDOW     32     /* EMA window for weight velocity       */
#define CREDIT_CHECK_EVERY    16     /* apply every N bytes (same as Phase33)*/
#define CREDIT_WARMUP        128     /* min bytes before any injection       */
#define CREDIT_GAMMA          0.25f  /* blend fraction (slightly softer)     */
#define CREDIT_ENTROPY_THRESH 0.72f  /* max normalized entropy to inject     */
                                     /* 0 = only inject if one expert wins   */
                                     /* 1 = always inject (uniform ok)       */

/* --- state --------------------------------------------------------------- */
typedef struct {
    float    win_ema[MOE_N_EXPERTS];  /* rolling win rate, sums to ~1        */
    float    w_vel[MOE_N_EXPERTS];    /* rolling |Δweight| per expert        */
    float    w_prev[MOE_N_EXPERTS];   /* weight snapshot for velocity delta  */

    float    ema_alpha;   /* 1 / CREDIT_WIN_WINDOW                           */
    float    vel_alpha;   /* 1 / CREDIT_VEL_WINDOW                           */

    uint64_t n_bytes;
    uint64_t n_since_apply;

    /* Stats */
    uint64_t n_injections;
    uint64_t n_skipped_entropy;  /* times skipped due to flat win_ema        */
    int      leading_expert;     /* abs expert index of current leader, -1=none */
} RegimeCreditState;

/* Initialize state. */
void regime_credit_init(RegimeCreditState* rc);

/* Observe one byte result.
 * moe_losses[i] = -log2(p_expert[i][target]) for slot i (0..n_active-1).
 * abs_slots[i]  = absolute expert index for slot i.
 * w[i]          = current MoE weight for slot i (after moe_update_gated).
 * Call AFTER moe_update_gated(), before regime_credit_apply(). */
void regime_credit_observe(RegimeCreditState* rc,
                           const double* moe_losses,
                           const double* w,
                           int n_active,
                           const int* abs_slots);

/* Blend credit-derived prior into MoE weights every CREDIT_CHECK_EVERY bytes.
 * Call immediately after regime_credit_observe(). */
void regime_credit_apply(RegimeCreditState* rc, MoeState* moe,
                         const int* abs_slots, int n_active);

#endif /* REGIME_CREDIT_H */
