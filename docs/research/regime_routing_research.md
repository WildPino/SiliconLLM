# Regime Routing Research — Phases 32-34B

*Branch: `research-regime-priors` | Status: **Closed, not merged** | Date: 2026-05-26*

---

## Question

Can SEE detect corpus regime at encode-time and route to a better prior, without external metadata?

---

## What was done

### Phase 32 — Regime Segmentation Tribunal

Offline cluster analysis of compression credit signals across 14 corpora. Found 6 stable clusters with an oracle-gap of 1.14 BPB range — i.e., a perfect router would have gained up to 1.14 BPB vs a fixed prior on the hardest cases. Confirmed that 5 distinct expert-optimal configurations exist across clusters. This established that routing opportunity is real.

Key finding: Cluster 4 (markup/heading content) is BI-optimal with +1.39 BPB loss vs best-available — the single largest unserved regime.

### Phase 33 — Regime Prior Router V1

First live router: char-class feature extraction at encode-time to select among pre-trained prior sets. 6/8 PASS initially.

Failures: shuffled corpus was penalized (+0.020 BPB) because RANDOM expert was redundantly included; md_mixed fell below threshold (-0.026 BPB, insufficient gain).

### Phase 33B — Router fixes

Fixed shuffled (+0.003 PASS) by removing RANDOM redundancy. Added 3 EMA guards to stabilize transitions. 61/61 regression PASS. md_mixed remained at the LZ-prior limit (-0.026). This established the REGIME_TABLE.

### Phase 34 / 34B — Credit-Only Dual-EMA Router

**Core question**: can credit dynamics alone (no char-class features) detect regime transitions?

Architecture: Dual-EMA on LZ credit signal (SLOW=256 bytes, FAST=12 bytes, THRESH=2.0 sigma). When fast-EMA deviates from slow-EMA beyond threshold, switch prior.

Results vs char-class prior router:

| corpus        | Δ prior | Δ dual  | winner |
|---------------|---------|---------|--------|
| markdown_docs | -0.126  | -0.079  | prior  |
| md_mixed      | -0.026  | -0.031  | dual   |
| shuffled      | +0.003  | +0.002  | dual   |
| c_code        | -0.006  | -0.020  | dual   |
| json_synth    |  0.000  | -0.059  | dual   |
| log_synth     | -0.145  | -0.103  | prior  |
| notes_it      | -0.037  | -0.019  | prior  |

Dual beats prior on: json, c_code, md_mixed, shuffled.  
Prior beats dual on: markdown_docs, log_synth, notes_it.  
Neither dominates.

Residual markdown gap (-0.047 BPB vs prior): the slow-EMA at 256 bytes cannot resolve heading transitions (~12 bytes). The signal arrives but late.

---

## Answer

**Yes.** Credit dynamics contain regime signal. A threshold on fast/slow EMA divergence can switch priors at encode-time without any character-class inspection.

**But**: neither router dominates. Gains are small in absolute BPB terms. The per-domain split (dual better on structured code/json, prior better on markup/logs) suggests the two approaches are detecting different things, not the same thing at different precision.

---

## Decision: not promoted to core V1

Reasons:

1. No single-router wins across all corpora — promotion would require a routing-of-routers, which is a complexity trap.
2. Absolute gains are small (max -0.059 BPB on json_synth, which is a synthetic corpus).
3. V1.0 is already sealed. Adding a second routing layer without a compelling real-world use case is premature.
4. The scientific value (credit signal contains regime information) is captured in this document. It does not need to live in the binary.

---

## What would justify revisiting

- A real-world corpus (not synth) showing consistent >0.1 BPB gap addressable by either router.
- A multi-scale EMA (fast=12, medium=32, slow=256) that closes the markdown heading gap without regression on shuffled. This was not attempted and remains plausible.
- A use case where routing metadata (corpus type) is unavailable at encode-time but credit dynamics are naturally available.

---

## Branch disposition

Branch `research-regime-priors` is kept as a historical record but will not be merged into `SirProjects`. The Phase 32-33B char-class router infrastructure (`src/regime_router.*`) and the dual-EMA code (`src/regime_dual.*`, `src/regime_credit.h`) remain on the branch only.
