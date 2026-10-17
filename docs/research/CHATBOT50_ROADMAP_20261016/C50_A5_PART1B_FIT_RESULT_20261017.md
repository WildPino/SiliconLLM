# C50-A5 part 1b result: a rotated r512 3.5-bit preview with K = 32,768 is selected at 94.5 MB/token

17 October 2026. Cell A5 of the [CHATBOT50 roadmap](ROADMAP.md) §5, under the
[part 1b protocol](C50_A5_PART1B_GRID_EXTENSION_PROTOCOL_20261017.md). It follows the
[part 1 FAIL](C50_A5_PART1_FIT_RESULT_20261017.md).

**Verdict: SELECTED** (the selection is made by `c50_a5_select.py --grid 1b`).
- **Selected configuration:** `rot_r512_q3sim`, K = 32,768.
- **Recall:** FIT top-1 recall **0.99921** over 79,709 positions.
- **Bytes per token, 94.45 in total:** preview 35.2 + V_r 4.2 + rerank 55.0 MB. The
  exact head costs 264.1, so this saves 169.7 MB/token.
- **Qualifying configurations:** 33 of 91.
- **Prediction (80-110 MB/token, a rotated r512/r1024 preview at 2.5-3.5 bit with K in
  the tens of thousands): right on every clause.**

**Registered budget arithmetic** (DESK-MODEL, bytes only):

| Term | MB/token |
| --- | --- |
| Ling-mini P4, ledger | 461.0 |
| minus the ledger's head | −23.8 |
| plus the selected head | +94.5 |
| **Total** | **531.7** |

That is above B = 530 by 1.65 MB (0.3%). Under the protocol's registered branch,
part 2 (DEV) still runs and the breach goes to the owner. Note the scale of the margin:
the ledger itself is DESK-MODEL, and the measured P0 weights already exceeded their row
by 5.5% (A2).

## Controls

- **Q8_K oracle:** through the rank code, recall at 1 is 1.0 on 7,971 positions.
- **Known positive:** part 1's diagnostics, same rank code. The full-rank float32
  preview reaches 1.0 at K = 8.

## Recall on FIT

| Configuration | K = 1,024 | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 | Cheapest MB/token at ≥ 0.999 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rot r256 q4sim | 0.8942 | 0.9679 | 0.9850 | 0.9941 | 0.9975 | 0.9988 | — |
| rot r256 q3sim | 0.8782 | 0.9604 | 0.9813 | 0.9922 | 0.9972 | 0.9989 | — |
| rot r256 t2sim | 0.5183 | 0.6910 | 0.7849 | 0.8731 | 0.9402 | 0.9784 | — |
| rot r512 q4sim | 0.9348 | 0.9845 | 0.9942 | 0.9979 | 0.99915 | 0.9997 | 104.5 (K 32,768) |
| **rot r512 q3sim** | 0.9249 | 0.9809 | 0.9929 | 0.9976 | **0.99921** | 0.9997 | **94.5 (K 32,768)** |
| rot r512 t2sim | 0.6178 | 0.7719 | 0.8449 | 0.9100 | 0.9589 | 0.9869 | — |
| rot r1024 q4sim | 0.9631 | 0.9941 | 0.9986 | 0.99985 | 0.99999 | 1.0 | 126.5 (K 16,384) |
| rot r1024 q3sim | 0.9576 | 0.9921 | 0.9979 | 0.99967 | 1.0 | 1.0 | 106.3 (K 16,384) |
| rot r1024 t2sim | 0.7175 | 0.8400 | 0.8960 | 0.9420 | 0.9727 | 0.9919 | — |

**Full-rank preview** (W itself; no rotation; K from 8):

| Configuration | K = 8 | 64 | 256 | 1,024 | 16,384 | 32,768 | Cheapest MB/token at ≥ 0.999 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full q3sim (140.9 MB) | 0.99903 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 140.9 (K 8) |
| full t2sim (100.6 MB) | 0.9412 | 0.9945 | 0.99926 | 0.99950 | 0.99969 | 1.0 | 101.0 (K 256) |

(Record: `results/c50_a5_offline_1b_fit.json`, `results/c50_a5_select_1b_fit.json`, log
`results/c50_a5_offline_1b_fit.log`, 705 s.)

## Reading (each sentence names its row)

- **The selection clears the bar by 0.00021.**
  - The selected row's recall is 0.99921. Its binomial standard error on 79,709
    positions is ~0.00011, and positions within a document are correlated, so the true
    error is larger.
  - The DEV confirmation in part 2 can therefore fail on sampling alone. That is a
    property of a cheapest-qualifying rule, stated before DEV is read; it is not a
    reason to move the bar.
- **The runner-up is a different design.** Full-rank ternary needs K = 256 at 101.0
  MB/token (full t2sim row), 6.5 MB more.
  - Its rerank is 128 times smaller. That matters for compute, which part 3 measures;
    the selection rule prices bytes only.
  - It cannot replace the selection unless the selected configuration fails DEV. Even
    then, the rule does not move to the runner-up by itself.
- **Ternary storage only works at full rank.** At r ≤ 1024, ternary storage costs more
  recall than it saves in bytes: compare the t2sim rows with the q3sim rows at every
  rank.
- **Rank does the work.** At equal storage, going from r512 to r1024 gains more recall
  than going from 3.5 to 4.5 bits: compare the q3sim and q4sim rows at r512 and r1024.

## Next

- **Part 2:** capture DEV with the A5 binary (the first 12 items per category, the same
  rule as FIT). Then apply the selected configuration once. The roadmap gate is DEV
  top-1 agreement ≥ 0.999.
- **Owner:** the P4 budget breach of +1.65 MB is recorded in COMMUNICATION. It needs no
  action until A6 has priced P3/P4 on the real artifact.
