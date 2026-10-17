# C50-A5 part 5 result: a used-vocabulary head fails on FIT; DEV-B is not read

17 October 2026. Under the [part 5 protocol](C50_A5_PART5_USED_VOCAB_HEAD_PROTOCOL_20261017.md),
pushed before any statistic was computed. Tool: `benchmarks/chatbot50/c50_a5_used_vocab.py`.
Record: `benchmarks/chatbot50/results/c50_a5_used_vocab_fit.json`.

## Data

- **Holdout extension.** `c50_holdout_ext.py` passes its proof check: the regenerated FIT
  ranks 1-48 and DEV ranks 1-32 equal the committed `item_sha256`
  (`results/c50_holdout_ext_v1_manifest.json`).
- **SEL:** 600 documents, 1,344,686 tokens, 200 per category.
- **DEV-B:** 36 documents, 80,297 tokens. It was built, but it is not scored and not read.
- **Control:** the full vocabulary gives FIT12 recall 1.0, as required (not VOID).

## Gated statistic (FIT12, 79,709 positions, engine exact top-1)

| N | Kept rows | MB per token | FIT12 recall |
| --- | --- | --- | --- |
| 8,192 | 8,579 | 14.4 | 0.87615 |
| 16,384 | 16,751 | 28.1 | 0.92651 |
| 24,576 | 24,935 | 41.9 | 0.95174 |
| 32,768 | 33,123 | 55.6 | 0.96743 |
| 40,960 | 41,310 | 69.4 | 0.97658 |
| 49,152 | 49,414 | 83.0 | 0.98274 |
| 65,536 | 65,798 | 110.5 | 0.98811 |
| 81,920 | 82,182 | 138.1 | 0.99131 |
| 98,304 | 98,566 | 165.6 | 0.99466 |

**Verdict: no N reaches 0.9995, so part 5 FAILS on FIT.** DEV-B is not scored, as the
protocol requires. The route is closed: the engine run on DEV-B is not spent, and DEV-B stays
unread for a later route.

**Prediction scored wrong.** The registered prediction was a selected N of 32,768-49,152; at
those sizes recall is only 0.967-0.983. The selection fails at every grid point, including
98,304 (0.99466), which keeps as many rows as the script head of part 4.

## Diagnostic (computed after the verdict, not gated)

- **Size of SEL.** SEL contains 46,764 distinct IDs. Even keeping every one of them (47,113
  rows with the always-kept tokens) gives FIT12 recall 0.982 only.
- So about 1.8% of the model's exact top-1 predictions on FIT12 are tokens that never occur in
  1.34 million tokens of the same mix.
- Beyond ~47k rows the ranking is by ID, which carries no information. That is why recall
  climbs only slowly from 49,152 to 98,304.
- **At N = 98,304:** the 426 missed positions spread over 239 distinct tokens.
  - One document holds 124 of them.
  - The most frequent misses are multilingual word pieces: ` zona` 60, ` trafik` 16, Korean
    and Indonesian pieces (` berarti`, ` menyebabkan`), and code identifiers (`_libs`,
    `_ADMIN`, `(category`).
  - Every one of the top 30 has count 0 in SEL.

**Reading:**
- The model's top-1 tail is long and document-specific: a document in an uncommon language or
  with its own identifiers brings tokens that a frequency corpus of this size never sees.
- A frequency-selected head cannot reach the 0.999 gate on this mix without keeping nearly
  the whole vocabulary.
- A per-document, adaptive set (for example, the rows seen in the context) would be a
  different route with a different failure mode. It is not tested here.

## What this cannot say

- **One SEL size.** A SEL 10× larger would see more of the tail. But the slope from 49k to
  98k rows (0.983 → 0.995) shows that the remaining tail is spread thinly over tens of
  thousands of rows. The 0.9995 bar was fixed in advance and is not moved.
- **No rate**, and no generation-quality claim.
