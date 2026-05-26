#include "regime_prior.h"
#include <string.h>

static int is_alnum(uint8_t b) {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9');
}
static int is_space(uint8_t b)   { return b == 32 || b == 9; }
static int is_newline(uint8_t b) { return b == 10 || b == 13; }
static int is_punct(uint8_t b)   { return b >= 33 && b <= 126 && !is_alnum(b); }
static int is_special(uint8_t b) {
    return (b > 126) || (b < 32 && !is_space(b) && !is_newline(b));
}

void regime_prior_init(RegimePriorState* rp, int muted) {
    memset(rp, 0, sizeof(*rp));
    rp->ema_alpha = 1.0f / (float)REGIME_EMA_WINDOW;
    rp->muted     = muted;

    /* Initialize all to neutral (equal priors) */
    for (int r = 0; r < REGIME_N; r++)
        for (int e = 0; e < 5; e++)
            rp->prior[r][e] = 0.20f;

    /* Calibrated from Phase 32 tribunal — Cluster 4: BI oracle (+1.39 BPB)
     * Condition: punct_frac > 0.10, newline_frac > 0.03 */
    float markup[5] = {0.07f, 0.07f, 0.50f, 0.29f, 0.07f};  /* BI nudge, LZ kept */
    float prose[5]  = {0.10f, 0.07f, 0.10f, 0.13f, 0.60f};  /* TOKPFX nudge */
    float repeat[5] = {0.07f, 0.07f, 0.07f, 0.72f, 0.07f};  /* LZ nudge   */
    float random_[5]= {0.07f, 0.72f, 0.07f, 0.07f, 0.07f};  /* UNI nudge  */
    memcpy(rp->prior[REGIME_MARKUP],  markup,  sizeof(markup));
    memcpy(rp->prior[REGIME_PROSE],   prose,   sizeof(prose));
    memcpy(rp->prior[REGIME_REPEAT],  repeat,  sizeof(repeat));
    memcpy(rp->prior[REGIME_RANDOM],  random_, sizeof(random_));
}

void regime_prior_observe(RegimePriorState* rp, uint8_t b) {
    float a = rp->ema_alpha;
    rp->f_alnum   += a * ((is_alnum(b)   ? 1.0f : 0.0f) - rp->f_alnum);
    rp->f_space   += a * ((is_space(b)   ? 1.0f : 0.0f) - rp->f_space);
    rp->f_newline += a * ((is_newline(b) ? 1.0f : 0.0f) - rp->f_newline);
    rp->f_punct   += a * ((is_punct(b)   ? 1.0f : 0.0f) - rp->f_punct);
    rp->f_special += a * ((is_special(b) ? 1.0f : 0.0f) - rp->f_special);
    rp->n_bytes++;
}

RegimeType regime_prior_classify(const RegimePriorState* rp) {
    if (rp->muted) return REGIME_NEUTRAL;
    /* Require warmup — EMA needs REGIME_EMA_WINDOW bytes to stabilize */
    if (rp->n_bytes < (uint64_t)REGIME_EMA_WINDOW) return REGIME_NEUTRAL;

    /* Phase 32 calibrated thresholds (conservative — prefer neutral over misfire)
     *
     * Code has alnum≈0.50, punct≈0.23: excluded from PROSE by punct guard.
     * Prose has alnum≈0.77, space≈0.18, punct≈0.04: clean PROSE signal.
     * Markup headings have newline≈0.06, punct≈0.08, alnum≈0.73: MARKUP.
     * Log has newline≈0.08, alnum≈0.45: REPEAT.
     * Shuffled has special≈0.40: RANDOM.                                    */
    if (rp->f_special > 0.15f)
        return REGIME_RANDOM;
    if (rp->f_newline > 0.05f && rp->f_punct > 0.06f && rp->f_punct < 0.18f)
        return REGIME_MARKUP;
    if (rp->f_alnum > 0.72f && rp->f_punct < 0.06f)
        return REGIME_PROSE;
    if (rp->f_newline > 0.07f && rp->f_alnum > 0.35f)
        return REGIME_REPEAT;
    return REGIME_NEUTRAL;
}

void regime_prior_apply(RegimePriorState* rp, MoeState* moe,
                        const int* abs_slots, int n_active) {
    /* Re-classify every REGIME_CHECK_EVERY bytes */
    rp->n_since_check++;
    if (rp->n_since_check < (uint64_t)REGIME_CHECK_EVERY) return;
    rp->n_since_check = 0;

    RegimeType reg = regime_prior_classify(rp);
    rp->n_regime[reg]++;

    /* Only inject prior on regime SWITCH — do not disturb converged weights */
    if (reg == rp->current) return;
    rp->prev    = rp->current;
    rp->current = reg;
    rp->n_switches++;

    if (reg == REGIME_NEUTRAL) return;  /* switching to neutral = no action */

    const float* prior = rp->prior[reg];
    float g = REGIME_GAMMA;

    /* One-shot blend at the switch point.
     * w[i] = (1-g)*w_moe[i] + g*prior[abs_expert[i]]
     * Only touches the 5 main experts; TOK_PREV/SPAN_PFX unaffected.
     * Sum is preserved: (1-g)*1 + g*1 = 1.                             */
    for (int i = 0; i < n_active; i++) {
        int ae = abs_slots[i];
        if (ae >= 5) continue;
        moe->w[i] = (1.0f - g) * (float)moe->w[i] + g * prior[ae];
    }
}
