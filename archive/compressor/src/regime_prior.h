#ifndef REGIME_PRIOR_H
#define REGIME_PRIOR_H

/* Phase 33: Regime Prior Router
 *
 * A lightweight causal regime detector that blends a regime-specific prior
 * into the MoE weight vector after each byte.
 *
 * Schema:
 *   past bytes -> EMA char-class stats -> regime classify -> prior blend -> MoE
 *
 * Constraints:
 *   - Uses only past observations (causal — same for encode and decode)
 *   - Does NOT modify expert prediction logic, only MoE prior weights
 *   - Pure C, no heap allocation
 *
 * Regime → oracle expert (from Phase 32 tribunal data):
 *   MARKUP  → BI     (headings, tables, separators; oracle_gap=1.39 BPB)
 *   PROSE   → TOKPFX (alphanum-heavy text; oracle_gap=1.05 BPB)
 *   REPEAT  → LZ     (log lines, repetitive; oracle_gap=0.75 BPB)
 *   RANDOM  → UNI    (non-ASCII/incompressible; oracle_gap=0.33 BPB)
 *   NEUTRAL → equal  (default, no regime detected)
 *
 * Integration: call regime_prior_observe() + regime_prior_apply() after
 * moe_update_gated() at the end of each byte, before the next prediction.
 */

#include <stdint.h>
#include "moe_engine.h"

/* --- tunables (compile-time, Phase 33 first version) ------------------- */
#define REGIME_EMA_WINDOW   32      /* effective rolling window in bytes    */
#define REGIME_GAMMA        0.30f   /* blend fraction on regime SWITCH only */
#define REGIME_CHECK_EVERY  16      /* re-classify every N bytes            */

/* --- regime labels ----------------------------------------------------- */
typedef enum {
    REGIME_NEUTRAL = 0,
    REGIME_MARKUP  = 1,   /* headings, tables, separators → BI    */
    REGIME_PROSE   = 2,   /* word-heavy text                → TOKPFX */
    REGIME_REPEAT  = 3,   /* repetitive lines/logs          → LZ    */
    REGIME_RANDOM  = 4,   /* high-entropy / non-ASCII       → UNI   */
    REGIME_N       = 5
} RegimeType;

/* --- state ------------------------------------------------------------- */
typedef struct {
    /* EMA fractions of char classes in the rolling window */
    float f_alnum;      /* alphanum (a-z A-Z 0-9) */
    float f_space;      /* space / tab */
    float f_newline;    /* LF / CR */
    float f_punct;      /* printable ASCII punct (33–126, not alnum/space) */
    float f_special;    /* non-ASCII or non-printable control chars */

    float  ema_alpha;   /* 1 / REGIME_EMA_WINDOW */
    int    muted;       /* 1 = always use neutral prior (ablation) */

    /* Regime-indexed prior weight tables [REGIME][abs_expert_index]
     * Expert order matches MOE_EXP_*: SEE=0 UNI=1 BI=2 LZ=3 TOKPFX=4
     * Only first 5 experts are controlled; extras (TOK_PREV, SPAN) skipped. */
    float prior[REGIME_N][5];

    /* Regime tracking for switch-only injection */
    RegimeType current;
    RegimeType prev;
    uint64_t   n_since_check;   /* bytes since last classification */

    /* Stats (for reporting) */
    uint64_t n_bytes;
    uint64_t n_regime[REGIME_N];
    uint64_t n_switches;
} RegimePriorState;

/* Initialize with default prior tables and EMA window. */
void regime_prior_init(RegimePriorState* rp, int muted);

/* Observe one byte (call AFTER it has been consumed and state updated). */
void regime_prior_observe(RegimePriorState* rp, uint8_t b);

/* Classify current regime from EMA stats. */
RegimeType regime_prior_classify(const RegimePriorState* rp);

/* Blend regime prior into MoE weights.
 * abs_slots[i] = absolute expert index for MoE slot i.
 * Only slots with abs_slots[i] < 5 are affected; the rest are untouched. */
void regime_prior_apply(RegimePriorState* rp, MoeState* moe,
                        const int* abs_slots, int n_active);

#endif /* REGIME_PRIOR_H */
