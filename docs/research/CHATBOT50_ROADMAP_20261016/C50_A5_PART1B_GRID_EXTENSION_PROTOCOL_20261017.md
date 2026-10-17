# C50-A5 protocol, part 1b: the registered grid extension after the part 1 FAIL

17 October 2026. Cell A5 of the [CHATBOT50 roadmap](ROADMAP.md) §5. Pushed before it runs.
This is the roadmap's registered fallback for a failed A5, "a larger probe set" (§9).
It follows the [part 1 FAIL](C50_A5_PART1_FIT_RESULT_20261017.md).

## What was read before this was written

All of part 1 on FIT:
- the 48 configurations;
- the post-FAIL diagnostics:
  - rank quantiles;
  - the spectrum;
  - recall by confidence;
  - the full-rank known positive.

The extension is therefore designed with knowledge of FIT. That is admissible because
FIT is the selection set: DEV, which confirms, is still unread and is read once. The
grid below is the one the diagnostics point to. It is fixed now and not enlarged after
reading.

## Unchanged from part 1

- **Capture and ground truth:** the same capture `FIT12` (79,709 positions), and the
  same ground truth (the engine top-1).
- **Code reused:** the rank code is the same; so are the Q8_K oracle control
  (addendum 1: rank-code recall at 1 ≥ 0.999 on every 10th position, else VOID) and the
  rerank price (1,680 B per Q6_K row).
- **Selection rule:**
  - fewest bytes per token with FIT top-1 recall ≥ 0.999; ties go to the smaller K;
  - the selected configuration is confirmed once on DEV (part 2);
  - the 0.999 bar does not move.

## Grid (`c50_a5_offline_1b.py`)

Simulated storage, groups of 32 weights with a float16 scale:
- **q4sim, 4.5 bit:** absmax/7, levels −8..7.
- **q3sim, 3.5 bit:** absmax/3, levels −4..3.
- **t2sim, 2.5 bit:** ternary, TWN rule. Delta = 0.7 mean|g|, scale = mean of |g| above delta.

| Family | Configurations | Bytes per token |
| --- | --- | --- |
| rotated preview, W ≈ (W V_r) V_r^T | r ∈ {256, 512, 1024} × {q4sim, q3sim, t2sim} × K ∈ {1,024 … 65,536} (powers of 2) | 157,184·r·bits/8 + 2048·r·4 B + K·1,680 B |
| full-rank preview, W itself, no rotation | {q3sim, t2sim} × K ∈ {8 … 65,536} (powers of 2) | 157,184·2048·bits/8 + K·1,680 B |

That makes 9 × 7 + 2 × 14 = 91 configurations, all priced.

## Registered consequences (fixed now)

- **Budget arithmetic** (DESK-MODEL, bytes only): Ling-mini P4 = 461.0 − 23.8 (the
  ledger's head) + the selected MB/token. It is compared with B = 530 MB at context 2048.
  The selection script prints it.
- **If a configuration is selected and P4 ≤ 530:** part 2 (DEV confirmation), then
  part 3 (C implementation and rate).
- **If a configuration is selected but P4 > 530:**
  - part 2 still runs, so the fact is known;
  - the budget breach goes to the owner with the roadmap's options: A6 recovering the
    bytes, a shorter declared context, or a cheaper donor.
- **If nothing reaches 0.999:** A5 FAILS finally. The risk register's second fallback,
  "a failed A5 means a cheaper donor", goes to the owner as a decision; this cell does
  not take it.

## Registered prediction (scored, not a gate)

- The selected configuration costs **80-110 MB/token**.
- It is a rotated r512 or r1024 preview at 2.5-3.5 bit with K in the tens of thousands.
- Neither full-rank arm wins: the full-rank 2.5-bit preview alone costs 100.6 MB.

## What this part cannot say

Everything listed for part 1 still holds:
- no rate;
- no sampling claim;
- FIT only.

In addition, the storage arms are simulated: the C storage format and its kernel are
part 3.
