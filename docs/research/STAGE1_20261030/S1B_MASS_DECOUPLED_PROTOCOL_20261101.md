# Stage 1 (a), part 1b: k = 4 with the routing mass decoupled from the kept experts, no training. PROTOCOL

2026-11-01. Follows part 1 (`S1A_FEWER_EXPERTS_RESULT_20261101.md`), before part 2's training. Same panel, same
reference outputs, same labels and chat gate as part 1 (`S1A_FEWER_EXPERTS_PROTOCOL_20261030.md`).

## Question

Part 1's k = 4 arms keep 4 of the 8 selected slots and renormalize the kept weights among themselves (k₂ = k₁ below):
each kept expert's weight grows by S₈ / S₄. A 2026 paper names this default as the worst choice and recovers most
of the loss without training by changing only the denominator. Does that hold on Ling-mini's sigmoid router? If it
does, the k of part 2's training changes from 4 to 2 (part 1's registered rule) or the training is spent on a smaller gap.

## Literature (read in the paper's tables)

arXiv 2609.04575 ("Training-Free Halving of Activated Experts in Fine-Grained MoE Models"), Eq. 2: activate the top
k₁ experts, divide by the router mass of the top k₂. Softmax routers, renormalized by construction.

| Qwen3.6-35B-A3B (256 experts, k = 8), k₁ = 4 | MMLU (n = 2000), Table 5 | GSM8K (n = 500), Table 6 |
| --- | --- | --- |
| native k = 8 | 81.65 | 95.00 |
| k₂ = 4 (default renormalization) | 77.00 (−4.65) | 88.40 (−6.60) |
| k₂ = 8 (the trained k) | 78.55 (−3.10) | — |
| k₂ = 16 (2k) | 81.30 (−0.35) | 94.40 (−0.60) |

- **On Qwen3.5-397B (k = 10, k₁ = 5):** k₂ = 10 is best, at −0.55 MMLU against −2.10.
- **k₂ too large is catastrophic:** at k₁ = 6, k₂ = 32 costs +23% perplexity, and k₂ = 256 (no renormalization) costs −27.45 MMLU (Tables 4-5).

Failure modes stated by the paper:
- **The best k₂ depends on the model** (2k on the 35B, k on the 397B), and perplexity picks a worse k₂ than MMLU does.
- **Generation was tested only on short GSM8K.**
- **Coarse-grained MoE was not tested.** The paper predicts a smaller benefit there.
- **Selection must stay exact:** random selection is far worse.

Not tested by the paper:
- a **sigmoid** router with a load-balancing bias and group-limited selection, which is Ling's;
- chat.

## Mechanism here (`s1_moe_hook.h`, `--moe-mass K`)

The hook acts after llama.cpp's norm_topk_prob division, on `ffn_moe_weights_norm-<layer>` (each row is
p_sel / S₈, before the routed scaling 2.5):

1. The same selection as part 1 zeroes the dropped slots.
2. The kept slots are multiplied by S₈ / D, where D is S₈ plus the (K − 8) largest unbiased probabilities among the
   248 experts not selected (K = 8: the factor is 1; K = 16: a top-16 mass).

The hook copies `ffn_moe_probs-<layer>` and `ffn_moe_topk-<layer>` to compute D. Every row is asserted to sum to 1
and to equal probs[id] / S₈ slot by slot; a failure stops the tool.

## Arms (ref, top4 and top4d are part 1's outputs; same seed)

| Arm | Selection | k₂ | Role |
| --- | --- | --- | --- |
| top4m8 | top 4 slots by biased score | 8 (the trained k) | treatment |
| top4m16 | top 4 slots by biased score | 16 (2k, the 35B's best) | treatment |
| rnd4m16 | 4 random slots of the 8 (part 1's draws) | 16 | trivial control of top4m16 |
| top4dm16 | top 4, decode tokens only | 16 | the decode-only form (part 1's top4d failed only the chat gate) |

Neither k₂ is tuned on this panel: 8 and 16 are the paper's two winners. No other k₂ runs.

## Predictions (registered before the dry run)

| Arm | Predicted Δ points (part 1: top4 −7.49) |
| --- | --- |
| top4m8 | −4.5 to −6.5 (recovers 1-3 points) |
| top4m16 | −1.5 to −4.5 (recovers 3-6 points); WITHIN BUDGET or UNDECIDED, not WITHIN SHARE |
| rnd4m16 | below −20 |
| top4dm16 | −1 to −3, and passes the chat gate |

## Decision rule (`s1m_verdict.py`, separate from the runner; selftest before the run)

- **Labels and chat gate:** as in part 1, against part 1's ref, on the same bootstrap draws (seed 20261030).
- **Mechanism:** MASS HELPS for an arm iff the CI95 of Δ(arm) − Δ(top4) lies above 0. For top4dm16 the comparison
  is with top4d.
- **Control:** as in part 1, top4m16 against rnd4m16.
- **What the rule fixes:**
  - **k4_form.** Among top4m16 and top4m8, the one with the higher Δ that FITS THE SHARE is k = 4 everywhere with
    that k₂. Failing that, if top4dm16 FITS THE SHARE, the form is decode-only with k₂ = 16. Otherwise there is none.
  - **training_k.** It is 2 if k4_form is not none (part 1's branch "a k = 4 form fits the share"). Otherwise it is 4.
  - **The weighting part 2 starts from at k = 4.** It is the mass arm with the higher Δ if that arm's mass helps;
    otherwise it is k₂ = k₁, part 1's top4.

## What a result changes (rule 5)

- **A mass form fits the share:**
  - Untrained k = 4 meets the lever's share. The desk ceiling at k = 4 (25.1 tok/s) is then reached without training.
  - Part 2 trains at k = 2.
  - Stage 2 checks the transfer to flash.
- **The mass helps but nothing fits:** part 2 trains at k = 4 from the mass weighting, on a smaller gap.
- **The mass does not help:** the paper's result does not transfer to a sigmoid router; part 2 is unchanged.

## Thesis line (rule 4)

As part 1: k = 4 is the only k whose desk ceiling (25.1 tok/s on all-ternary flash at the RAM wall) passes 24.7. This
cell does not change bytes; it changes only whether k = 4 needs training. Ling-mini's scale; the transfer to flash is
a stage 2 check. Mechanism-test budget: CPU only, ~9 h.

## What this protocol takes from, or changes in, the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| The paper's grid k₂ ∈ {k₁, k, 2k} | applied (k₁ is part 1's top4) | — |
| A k₂ tuned on held-out data | not done | the paper says perplexity picks the wrong k₂; MMLU-tuned k₂ would need a held-out MCQ set; 8 and 16 are registered instead |
| k₂ at k₁ = 6 and k₁ = 2 | not here | k = 6 does not reach 24.7 on the desk model; k = 2 is part 2's question if a k = 4 form fits |
| A scalar gain instead of the per-token D | not here | the paper's appendix finds it weaker |

## Dry run (`s1m_dry.py`)

It checks:
- the new binaries equal part 1's with the mass off (scorer and generator, bit for bit);
- mass 8 in mode none equals mode none;
- the mass acts, and its factors are 1 at K = 8 and below 1 at K = 16;
- the MASS and MOE counts;
- the decode path.

The dry run prints top-1 agreement on 2 documents, after the predictions above were registered.

**Result: PASS, 15 of 15** (`results/s1m_dry.json`):
- binaries: score `8ec02eb5…`, generate `29664c03…`; part 1's are kept in `bin/s1a/`;
- mean factor 1.0 at K = 8 and 0.699 at K = 16;
- top-1 agreement with ref on the 2 documents: top4 0.924, top4m8 0.953, top4m16 0.931 (seen after the predictions).

## Run

`s1m_run.py` runs the arms one at a time, in the order top4m16, rnd4m16, top4m8, top4dm16, into part 1's work
directory. It uses part 1's ref outputs for the documents' top-32 partition and for top4dm16's prompt-only panels.
Then `c50_s2_checks.py` scores each arm, and `s1m_verdict.py` applies the rule (selftest PASS, 6 planted cases). It is a
quality run, not a timing. `s1_autoresume.ps1 -Runner s1m_run -Report <s1m_run.json>` resumes it after a shutdown.
