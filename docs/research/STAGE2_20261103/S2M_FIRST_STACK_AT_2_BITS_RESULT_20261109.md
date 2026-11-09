# Stage 2, part 2, second test: the first stack at 2 bits but the head. RESULT

2026-11-09. Protocol `S2M_FIRST_STACK_AT_2_BITS_PROTOCOL_20261109.md`. CPU only, no training, Ling-mini. Records: `benchmarks/stage2/results/s2z_verdict.json`, `s2z_run.json`, `s2z_ks.json`, `s2z_export_{ks_stk4,ks_all4}.json`.

## Outcome

All three sections valid. **The stack is OVER BUDGET and still WORSE THAN ITS PARTS: THE COMPOUNDING, NOT THE FORMAT.** ks_org alone is UNDECIDED and keeps its chat. Bal's training against IQ2_KS on the experts: NOT SHOWN.

**Amendment 1 (below) reads the excess again: with ks_org at k = 4, the stack ADDS UP (−0.02). The −4.41 was the organs losing at k = 4, not stacking.**

| Section | Label | Number |
| --- | --- | --- |
| ks_org | **UNDECIDED**, chat gate passes | −3.02 [−5.73, −0.31], chat 30; over as_both +4.73 [+1.07, +8.39] |
| Stack | **OVER BUDGET**, chat gate fails | ks_stk4 −12.18 [−16.22, −8.13], chat 22; sum of parts −7.77; excess −4.41 [−8.30, −0.55]: **WORSE THAN ITS PARTS** |
| Disproof | **THE COMPOUNDING, NOT THE FORMAT** | −4.41 ≤ −3.80 with the CI95 below 0 |
| Control | **NOT SHOWN** | ks_stk4 − ks_all4 +1.91 [−1.70, +5.39] |

| Arm | Δ points [CI95] | Chat, degenerate | nq / tqa / gsm8k / arc / mmlu / chat, % | Top-1 / KL / dBPB, documents at k = 8 |
| --- | --- | --- | --- | --- |
| ref | | 28, 0 | 29.0 / 48.7 / 81 / 53.2 / 49.3 / 70 | |
| **ks_org**, organs IQ2_KS | **−3.02 [−5.73, −0.31]** | **30, 0** | 25.3 / 42.3 / 72 / 52.0 / 46.3 / 75 | 0.795 / 0.407 / +0.037 |
| bal4, experts alone, k = 4 | −4.74 [−8.32, −1.11] | 24, 0 | 19.7 / 37.5 / 86 / 52.7 / 46.8 / 60 | |
| **ks_stk4, judged**, k = 4 | **−12.18 [−16.22, −8.13]** | 22, 1 | 14.5 / 31.5 / 62 / 50.3 / 44.7 / 55 | 0.723 / 0.784 / +0.117 |
| ks_all4, control, k = 4 | −14.09 [−18.41, −9.68] | 18, 2 | 17.2 / 33.7 / 49 / 53.0 / 48.8 / 45 | 0.763 / 0.558 / +0.041 |
| stk4 (S2L), k = 4 | −16.70 [−20.47, −12.78] | 12, 0 | 13.7 / 31.2 / 66 / 48.2 / 41.9 / 30 | 0.685 / 1.026 / +0.180 |

Generation (nq, tqa, gsm8k, chat) runs at k = 4 on the stacks; arc, mmlu and documents are scored at k = 8.

## The run

- **IQ2_KS on the experts:** ik loaded 176 imatrix entries and named no organ or expert without weights; S2L's log, planted, fails the same check. Organs equal ks_org's byte for byte on 100 of 100 tensors. At most 4 values per block of 32 in the experts; relative error 0.099. 66 min.
- **Exports:** the organs read back exactly on both arms; the other 178 (ks_stk4) and 121 (ks_all4) tensors equal their reference's bytes. Container: Q8_0's worst squared error is 0.016% of IQ2_KS's.
- The background watcher was stopped once by the host for low memory while ks_all4's export held 59 GB; the chain itself ran through.

## What it says

- **The organs' share is −3.02**, against −6.25 for trained attention and −7.76 for as_both. Of stage 1's 5-point budget, the organs at 2 bits take three fifths alone, with chat above ref.
- **Even if it added up, the stack would not fit:** bal4 and ks_org sum to −7.77. It does not add up: the excess is −4.41, inside S2L's seed gap of stk4's −5.70. The format moved the stack from −16.70 to −12.18, about its gain alone (ks_org − st_attn +3.23); the compounding stayed.
- **The loss sits in generation at k = 4.** From bal4 to ks_stk4: gsm8k 86 → 62, nq 19.7 → 14.5, tqa 37.5 → 31.5, chat 24 → 22 passes; arc and mmlu, scored at k = 8, move 2.4 and 2.1.
- **Bal and IQ2_KS split by mode.** IQ2_KS experts read better on documents and scoring at k = 8 (top-1 0.763 against 0.723, mmlu +4.0, arc +2.7); bal's lead is in generation at k = 4 (gsm8k +13, chat +4 passes), where it was trained. The margin, +1.91, does not clear the noise.
- **Points and likelihood disagree between the two stacks:** ks_all4 has the better dBPB (+0.041 against +0.117) and the fewer points. Judged on points.

## Predictions

| | Predicted | Read | |
| --- | --- | --- | --- |
| ks_org | Δ −5 to −1, chat passes; UNDECIDED 60% | −3.02, 30; UNDECIDED | hit; 60% |
| ks_stk4 | Δ −13 to −6, chat 15-25; OVER 55% | −12.18, 22; OVER | hit; 55% |
| Excess | −5 to −1; WORSE 55%; THE COMPOUNDING 30% | −4.41; WORSE; THE COMPOUNDING | hit; 55%; 30% |
| ks_all4 | Δ −14 to −6; NOT SHOWN 55% | −14.09; NOT SHOWN | miss by 0.09, worse; 55% |

## What this does not say

No arm was trained together. Organs in BF16 and experts in Q8_0 stand in for IQ2_KS's kernel and its Q8_K activations. The imatrix was fitted at k = 8. One calibration, one panel set; the noise is S2L's single seed pair. Why stacking compounds is not measured here.

## Thesis line

No configuration at 2 bits but the head fits: the best, ks_stk4, is −12.18 at **43.05 ms, 23.23 tok/s** on the desk (sum of cells), against the 40.49 line. ks_all4 is −14.09 at 44.32 ms. Both miss on quality before speed. Stage 1's label puts −5 on the CI95's lower end: with a half-width near 4, WITHIN BUDGET needs a point near −1, about 11 points over ks_stk4. (Corrected 2026-11-09; it first read "about 3 more to reach −5", which took −5 as a point.)

## Parts specific to Ling

The numbers, the shared expert, one dense layer, the fused QKV, the 9 uncalibrated expert slots. Not specific: IQ2_KS with an imatrix per expert, the excess rule, the container check, bal.

## Recommendation

**Part 2 trains this stack together, as registered for OVER and THE COMPOUNDING: the organs start from IQ2_KS's values, bal's experts stay in place, at k = 4, with the budget on the whole stack and generation at k = 4 in the loss's reach. Bal stays (NOT SHOWN), at 0.25 ms. Its protocol, over 10 h of GPU, goes to an independent reviewer before it is pushed.** Its line: the trained stack WITHIN BUDGET with the chat gate, or at least the excess no longer WORSE.

**What would disprove it:** the trained stack still OVER BUDGET with an excess at or below −3.80. Then training does not undo the compounding at this scale, and the organs cannot all go to 2 bits: the next cut keeps one of them stored.

## Amendment 1, result: ks_org at k = 4

Protocol's amendment 1; `s2z_k4_verdict.json`, `s2z_k4_run.json`. Valid: the panels complete on ks_org's file, the 7 copied score files equal ks_org's.

| Line | Label | Number |
| --- | --- | --- |
| ks_org4 | **UNDECIDED**, chat gate fails | −7.41 [−10.90, −3.99], chat 22, 1 degenerate |
| Organs at k = 4, ks_org4 − ks_org | **THE ORGANS LOSE AT k = 4** | −4.39 [−7.44, −1.42] |
| Excess at k = 4, ks_stk4 − bal4 − ks_org4 | **IT ADDS UP** | −0.02 [−4.53, +4.54]; sum of parts −12.16 |
| The same, k = 4 counted once (+ top4dm16) | IT ADDS UP | +0.95 [−3.42, +5.41] |

ks_org4, nq / tqa / gsm8k / chat: 22.8 / 41.5 / 69 / 55, against ks_org's 25.3 / 42.3 / 72 / 75 at k = 8. Most of the loss is chat: 30 passes become 22.

**What it says:** the stack at 2 bits is the sum of bal4 and the organs at k = 4. Nothing compounds beyond that. The organs' 2 bits cost −3.02 at k = 8 and −7.41 at k = 4, while P0 alone at k = 4 is +0.97: k = 4 hurts the organs at IQ2_KS, not the model. As registered, the training's target becomes ks_org4's loss with bal's experts in place.

| Prediction | Predicted | Read | |
| --- | --- | --- | --- |
| ks_org4 | Δ −6 to −2 | −7.41 | miss by 1.41, worse |
| Organs at k = 4 | −3 to +1; DOES NOT MOVE 60%, LOSE 35% | −4.39; LOSE | miss by 1.39; 35% |
| Excess at k = 4 | −3 to +1; ADDS UP 60% | −0.02; ADDS UP | hit; 60% |

**What this does not say:** why k = 4 hurts the organs (the imatrix was fitted at k = 8, and decode at k = 4 feeds the organs another distribution: a reason, not measured). One seed. The chat panel has 40 items.
