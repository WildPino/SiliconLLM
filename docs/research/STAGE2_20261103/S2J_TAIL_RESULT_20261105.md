# Stage 2, part 1, test 2b-bis: the tail on new text. RESULT

2026-11-05. Protocol `S2J_TAIL_PROTOCOL_20261105.md` (`d090c3d`). Records: `benchmarks/stage2/results/s2u_run.json`, `s2u_verdict.json`, `s2u_overlap.json`.

## Outcome: PASS: THE HEAD READS IN TWO PASSES ON NEW TEXT

Cell tern_tr, K = 1,024, tail a × preview + b, on 96 documents nothing had read.

| Line | Limit | Measured |
| --- | --- | --- |
| dBPB, CI95 upper | ≤ +0.005 | **+0.0032 [+0.0021, +0.0047]** |
| Miss per 1,000, tool / float | ≤ 1 | 0.005 / 0.009 |
| KL top-32 | ≤ 0.02 | 0.0013 |
| KL over the vocabulary | ≤ 0.02 | 0.0061 |
| Chat, degenerate | ≥ 27, ≤ 1 | 28, 0 |
| Top-1 against the exact head | reported | 0.999995 |

Every control holds: replay, chat equal to test 2b's token for token, float parity, plant out, tail moves.
**Without the 3 overlapping documents:** +0.0032 [+0.0021, +0.0048], on 93. Test 2b's FAIL stays recorded as it is.

## The engine's rule

**Sampling happens among the K candidates only, in every mode. The tail never produces a word. It exists only to score.**
Generated text comes from the exact distribution truncated to K. That costs the exact tail's mass: mean 0.0024, median 5×10⁻⁷, p99 0.057, maximum 0.59.
So the mass is tiny on most positions and large on a few: at the worst position, more than half the probability sits outside the list.

**A caller who asks for a word outside the list gets an estimate, not the model's number:**

| Request | Answer |
| --- | --- |
| Top-n, n ≤ K | exact logits; the normalizer holds the tail's estimate |
| Probability of a word outside the list (perplexity, multiple choice, forced continuation) | a × preview + b, marked as estimated; this is what the dBPB line judged |
| The same, with its row read | exact numerator, 3.4 KB more on flash |
| A constraint that leaves no candidate (grammar, logit bias) | no answer from this pass; the whole head once, 16.5 ms on flash's desk |

## Arms, float (dBPB [CI95]; KL over the vocabulary; tail's mass)

| Arm | 96 new documents | 72 panel documents | Predicted, new |
| --- | --- | --- | --- |
| affine | +0.0033 [+0.0021, +0.0048]; 0.0061; 0.0067 | +0.0024 | +0.0012 to +0.0040: hit |
| offset, test 2b's | +0.0115 [+0.0092, +0.0139]; 0.0039; 0.0018 | +0.0114 | +0.005 to +0.013: hit |
| mass, true mass on dev | **+0.0123 [+0.0097, +0.0154]**; 0.0043; 0.0012 | +0.0119 | +0.006 to +0.012: just above |
| row_affine | +0.0022 [+0.0019, +0.0027] | +0.0023 | under +0.001: miss |
| row_mass | −0.0005 [−0.0008, −0.0002] | −0.0004 | under +0.0005: hit |
| plant, b − 3 | +0.0173, out | +0.0171 | |

Other predictions: outcome PASS 80%, KL 0.004-0.008, miss 0-0.2, tail's mass mean 0.002-0.003: hits. p99 0.02-0.05: 0.057, above.
With ternary experts' states: +0.0038 [+0.0030, +0.0049], miss 0.

## What the pass is, and what it is not

- **The tail is calibrated for the line, not for the truth.** It holds 2.8 times the exact mass (0.0067 against 0.0024). The arm with the right mass on dev leaves the line.
  The tail's mass is too small in that arm where a target falls in the tail, because the preview ranks those words low. So the two constants trade a worse total for a better score of the target.
- **Reading the target's row shows the gap:** with the true mass, the exact target gives −0.0005.
- **New text is of the same kind** as the panel's, from the same builder. Text after the model's release was not tried.

## Thesis line

The tail adds no time. The head has its cell, and 25.40 tok/s was the desk's ceiling; with the idle session's rows the ceiling is 21.27-22.57 (`S2E_ENGINES_RESULT_20261105.md`).

## Parts specific to Ling

a and b belong to this preview on this model. The vocabulary sets how far K sits from V. The rule and the mechanism need only a softmax head and a preview.
