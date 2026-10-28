# C50 cell M part 2 (addendum): R_acc on 72 unseen documents, F0 at its own temperature

**Status:** PROTOCOL, 28 October 2026. It adds to [C50_M_PROTOCOL_20261027.md](C50_M_PROTOCOL_20261027.md) and does
not change part 1. **It is a metric test.** No training runs, and no outcome moves the thesis.

## Why it runs

Part 1 demoted R: one temperature on T0 takes 0.64 of v0's R. It made R_acc, top-1 accuracy on the same gap, the
primary, "after part 2". Part 1's 36 documents are fit documents, and on three of them F0's accuracy jumps by 0.18-0.32.
Part 2 asks one question on A4-bis's 72 post-release documents, which none of the models saw: **does R_acc separate v0
from T0?**

## What runs

| Step | Where | What | Cost |
| --- | --- | --- | --- |
| 1 | local CPU | `c50_m_f0.py val`: F0 on the arms' validation segments (shards 470-471, 6 batches of 8,192 tokens), full log-softmax dumped; F0's NLL at the 81 temperatures of the arms' grid; `tau_nll` = its argmin | ~45 min |
| 2 | local CPU | `c50_m_f0.py docs`: F0 on the 72 documents (`a4bis_post/inputs/docs.tsv`); records, the target's log-probability at every temperature, and its rank. F0's top-5 exists for the first time | ~2.5 h |
| 3 | Kaggle | `m-<s4\|t2>-<arm>+post` for the 9 arms of part 1 (t0, a, b, c48, c496, v0, e2, n, n_shuffle), same weights, `--docs-tsv refs/a4bis_post/docs.tsv` | ~9 × 15 min |
| 4 | local | `c50_m_verdict.py --part2` | minutes |

The scorer is S2's pinned F0 setup: `c50_llama_score_avx2.exe`, 6 threads, ubatch 512, ctx 8192. The dumps (~26 and
~95 GB) are deleted once read.

## STOP checks (all must hold before any number is read)

- F0's sha256 equals its pin. The dump's target column equals the scorer's record within 1e-5.
- Every arm and F0 score exactly the targets of T0's llama records on these documents (`a4bis_post/p0`).
- Each arm's weights and temperatures equal part 1's. The fit is deterministic on the same shards, so a different τ means
  a different arm.
- Torch T0 against llama T0 on the 72 documents: within 0.010 BPB (MVE-5's bound). Above it, the gap mixes instruments
  beyond what part 1 allowed.
- No τ at the edge of the grid.

## The decision rule (`decide2`)

1. **F0 − T0 top-1** over all positions (paired document bootstrap). If its CI95 does not exclude 0:
   **INCONCLUSIVE**. R_acc divides by this gain, and on unseen documents it may vanish.
2. **R_acc(v0) − R_acc(T0)** (paired document bootstrap):
   - CI95 low > 0: **R_acc SEPARATES v0 FROM T0**;
   - otherwise: **R_acc DOES NOT SEPARATE**.
3. **Reported beside the rule:**
   - v0 − T0 in raw top-1 points, with its CI;
   - R's temperature control on these documents (part 1's `decide`);
   - all eight readings under R, R_cal and R_acc, with `compare`;
   - R_cal with F0 at its own τ, next to R_cal with F0 at τ = 1.

## Trivial control (rule 3)

- **R:** T0 at one temperature, as in part 1.
- **R_acc:** temperature cannot move top-1, so its null is N-shuffle − N. Part 1 put that null within ±0.015.

## What a result changes (rule 5)

- **SEPARATES:** R_acc judges every R-type reading from here, and the injection cell's docs top-1 guard stands as
  written.
- **DOES NOT SEPARATE, or INCONCLUSIVE:** the docs top-1 guard of the injection cell is labelled non-discriminating.
  Its "general" claims then rest on gsm8k, chat and the known-item panel. Any later R-type reading needs a different
  accuracy panel, named before it runs.

## What it takes from the design (rule 1)

| The design asks | Here | Why |
| --- | --- | --- |
| F0 at "its own temperature" | NLL rule only | KL to F0 is 0 for F0 by definition |
| F0's τ fit on the documents' mix | fit on the arms' validation shards (wiki, edu, math), while the documents hold 12 code documents | the same set as every arm, so no arm is fitted closer to the documents than F0. Ovadia et al. 2019's shift caveat applies to all arms alike |
| A per-bin temperature | not run | as in part 1 |

## Literature (rule 2)

Guo et al. 2017 and Ovadia et al. 2019, read for part 1 (quotes in the part-1 protocol). Nothing new is introduced:
R_acc and its null were reviewed in part 1.

## Independent review (rule 6)

Not required:
- the Kaggle cost is about 2.5 session-hours;
- the primary, R_acc, is not new: part 1 defined it and its reviewer examined it.

## Prediction

- **SEPARATES,** with R_acc(v0) between 0.10 and 0.30. Part 1 measured 0.231, CI [0.176, 0.308].
- **F0 − T0 top-1 is smaller than on part 1's documents,** because the memorised documents are gone, but its CI
  excludes 0.
- **Confidence: about 60%.** The INCONCLUSIVE branch is the likeliest miss.
