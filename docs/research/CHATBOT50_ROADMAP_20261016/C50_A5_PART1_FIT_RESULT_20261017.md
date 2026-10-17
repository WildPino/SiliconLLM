# C50-A5 part 1 result: no registered approximate head reaches 99.9% top-1 recall on FIT

17 October 2026. Cell A5 of the [CHATBOT50 roadmap](ROADMAP.md) §5, under the
[part 1 protocol](C50_A5_APPROXIMATE_HEAD_PROTOCOL_20261017.md) and its addendum 1.

**Verdict: FAIL** (selection script `c50_a5_select.py`, which chooses nothing itself).
None of the 48 priced configurations reaches FIT top-1 recall ≥ 0.999.
- Best preview: rank 512 with 4-bit storage and K = 1,024, recall 0.9348 at 51.2 MB/token.
- Best IVF: C = 256 with P = 128, recall 0.7074 at 104.3 MB/token.

The registered prediction is **wrong**. It said IVF would reach 99.9% at under 15
MB/token. IVF turns out to be the weaker family, and no IVF configuration in the grid
gets close.

## Runs

| Attempt | What happened | Record |
| --- | --- | --- |
| 1 | VOID on the oracle control: the float32 oracle agrees 0.98984 and the registered limit is 0.99 (addendum 1 explains why that limit was mis-specified) | `results/c50_a5_offline_fit_attempt1_VOID.log` |
| 2 | Controls pass, grid complete, 486 s | `results/c50_a5_offline_fit.json`, `results/c50_a5_select_fit.json` |

**Controls (attempt 2), on 7,971 positions (every 10th):**

| Oracle | Argmax agreement | lse error, median | lse error, max |
| --- | --- | --- | --- |
| Q8_K-emulated state × gguf-py head (the control) | 1.00000 | 1.6e-6 | 1.9e-5 |
| float32 state × gguf-py head (reported) | 0.98984 | 0.0288 | 0.476 |
| rank code, full head | 1.00000 | | |

## Recall on FIT (79,709 positions; the ground truth is the engine's top-1)

**Preview** (4-bit storage is the priced arm; float32 recall is within 0.007 of it everywhere):

| Rank r | K = 8 | K = 64 | K = 256 | K = 1,024 | MB/token at K = 1,024 |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.2696 | 0.4953 | 0.7016 | 0.8482 | 14.1 |
| 256 | 0.3796 | 0.6060 | 0.7737 | 0.8942 | 26.5 |
| 512 | 0.4980 | 0.7210 | 0.8486 | 0.9348 | 51.2 |

**IVF:**

| C | Cluster size min / median / max | P = 1 | P = 32 | P = 128 | MB/token at P = 128 |
| --- | --- | --- | --- | --- | --- |
| 256 | 1 / 107 / 65,885 | 0.0256 | 0.3683 | 0.7074 | 104.3 |
| 1,024 | 1 / 30 / 7,873 | 0.0437 | 0.2280 | 0.5457 | 16.7 |
| 4,096 | 0 / 7 / 3,922 | 0.0965 | 0.3477 | 0.5861 | 11.2 |

## Diagnostics run after the FAIL (`c50_a5_diag.py`, `results/c50_a5_diag_fit.json`)

These were run after the verdict. They check the instrument and characterise the
failure. They change no verdict.

- **Known positive: the instrument fires.** A full-rank preview (r = 2048, float32),
  run through the same rank code:
  - recall 0.99005 at K = 1, 0.99967 at K = 2, and 1.0 at K = 8;
  - zero rank disagreements with the plain float32 head.

  So the rank code and the candidate pipeline do return the top-1 when the scores
  allow it. The engine's top-1 is always inside the float32 top 8: the
  float32-vs-Q8_K flips are near-ties.
- **The head is not low-rank:**

  | r | 128 | 256 | 512 | 1,024 | 1,536 |
  | --- | --- | --- | --- | --- | --- |
  | Share of ‖W‖²_F in the top r eigenvalues of WᵀW | 0.187 | 0.283 | 0.437 | 0.682 | 0.873 |

  At r = 512 the preview discards 56% of the head's energy. That explains why its
  recall saturates slowly in K.
- **Tail of the candidate count:**
  - for the r512 4-bit preview, the rank of the engine top-1 has quantiles 8 (median), 537 (90%), 5,599 (99%) and **27,729 (99.9%)**;
  - for IVF C = 4,096, the probe rank of the top-1's cluster has quantiles 88, 499, 1,840 and **2,916 (99.9%)**, i.e. 71% of the clusters.
- **The misses are not confined to uncertain positions.** Recall of the r512 4-bit,
  K = 1,024 preview by the engine's p(top1):
  - 0.83 at p < 0.1 (403 positions);
  - 0.92-0.93 between 0.5 and 0.9;
  - **0.950 at p ≥ 0.9** (45,249 positions, 57% of FIT).

  A preview that is wrong on confident positions cannot be rescued by a confidence
  fallback alone.
- **IVF partitions are degenerate.** L2 k-means on the head rows puts 65,885 of
  157,184 rows in one cluster at C = 256 and leaves empty clusters at C = 4,096.
  Inner-product probing by the centroid is not a good proxy for the top row.

## What this means for the roadmap (ledger arithmetic, DESK-MODEL; no rate measured)

- **The registered fallback is "a larger probe set"** (ROADMAP §9).
  - The only family with a measured path to 0.999 is the preview, at K around 28k
    for r = 512. That costs about 45.3 + 4.2 + 46.6 ≈ 96 MB/token, against 264.1 MB for
    the exact head. The ledger had budgeted about 25 MB.
  - This figure is read off a FIT quantile. It is not a selected configuration: it is
    outside the registered grid.
- **Effect on the budget:** the ledger's P2-P4 head is 23.8 MB (264.1 − 240.3). A head
  of ~96 MB adds ~72 MB to Ling-mini P4, taking it from 461 to ~533 MB/token at
  context 2048 (desk ~53.6 tok/s). That is right at the
  roadmap's B ≈ 0.53 GB. The measured P0 weights already exceed their ledger row by
  5.5% (A2), so the margin left for the runtime factor is gone at P4 unless A6
  recovers it.
- **Next step:** an extension of the grid, registered and pushed before it runs (part 1b).
  - Larger K, and preview ranks up to the full 2048 with fewer storage bits (3.5 and
    2.5 bit/weight), priced under the same rule.
  - If its cheapest qualifying head still leaves P4 above B, the risk register's
    second fallback applies: "a failed A5 means a cheaper donor". That re-plan goes
    to the owner.

## What this result cannot say

- **Scope:** FIT positions only. DEV is untouched; it is read once, after a selection
  exists.
- **Bytes counted, not streamed:** no rate is implied.
- **The 99.9% bar is strict compared with the measurement floor.**
  - The engine top-1 differs from the float32 product of the same weights on 1.0% of
    positions, and llama.cpp differs from itself across builds by ~4% top-1 (A2b).
  - The bar is the roadmap's and is not moved.
