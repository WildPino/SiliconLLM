# Stage 2, part 1, test 2c: the organs without a router, ternary on every layer. RESULT

2026-11-06. Protocol `S2G_STACKED_ORGANS_PROTOCOL_20261104.md`. Records: `benchmarks/stage2/results/s2s_verdict.json`, `s2s_train_*.json`, `s2s_export_*.json`, `s2s_run.json`, `s2s_pick_10.json`.

## Outcome: THEY DO NOT FIT, and THE ORGANS COST MORE THAN THE EXPERTS

The judged arm st_both loses 13.36 points. Its margin against x15 is −5.69 [−8.71, −2.76]. Both labels are scoped to this training: 1.5 M tokens, one block at a time. Controls C1, C2 and C3 hold; all six exports are bit-exact.

| Arm | Ternary | Δ points [CI95] | Label | Chat | Top-1 / KL / dBPB on documents |
| --- | --- | --- | --- | --- | --- |
| **st_both, judged** | attention, dense FFN, shared expert | **−13.36 [−16.98, −9.59]** | OVER BUDGET | 15, 2 degenerate | 0.631 / 1.569 / +0.467 |
| st_attn | attention | −6.25 [−9.43, −3.10] | UNDECIDED | 22 | 0.779 / 0.512 / +0.069 |
| st_ffn | dense FFN, shared expert | −19.40 [−22.89, −15.83] | OVER BUDGET | 5, 25 degenerate | 0.535 / 2.337 / +0.802 |
| st_attn_lo | attention 0-9 | documents only | | | 0.836 / 0.288 / +0.041 |
| st_attn_hi | attention 10-19 | documents only | | | 0.837 / 0.270 / +0.032 |
| ph_both | as st_both, no training | documents only | | | 0.048 / 10.35 / +3.817 |

Excess over the sum: +12.29 [+7.81, +16.56]; it mixes the interaction with joint training, as registered.

**Scale 10 on layer 2 (asked for):** held-out block loss 0.2860, against 0.2601 at scale 3 and 0.2677 at scale 1. Scale 3 stands, and the grid's edge is closed.

## What it says

- **Training buys most of the distance:** ph_both is at top-1 0.05 and st_both at 0.63. It is not enough at 1.5 M tokens.
- **The FFN organs cost more than attention,** the reverse of the prediction: they lose three times the points. Chat falls to 5, and 25 answers degenerate.
- **st_ffn alone is worse than st_both, and this is unexplained.** Their last-layer streams are close (cosine 0.871 against 0.862, block loss 1.24 against 1.38), so the training's own metric did not see the gap the panels see. One fact from the logs: from layer 6 up, trained block loss in st_ffn stays at or just above the carried error, while st_both's falls under it on every layer from 7 up but layer 10. Attention trained beside the FFN may absorb error that the FFN alone cannot; nothing here tests that.
- **Block loss is not a stand-in for the panels** in this stack, so a later training cannot be judged on it alone.
- **Depth is a direction only:** the halves of attention cost about the same (+0.041, +0.032), and add up to about the whole (+0.069). Early layers do not clearly cost more.

## Predictions

| | Predicted | Read | |
| --- | --- | --- | --- |
| st_attn | top-1 0.60-0.78, KL 0.7-1.6, Δ −6 to −20 | 0.779, 0.512, −6.25 | hit, miss (better), hit |
| st_ffn | top-1 0.72-0.85, KL 0.3-0.9, Δ −3 to −12 | 0.535, 2.337, −19.40 | miss, all worse |
| st_both | top-1 0.50-0.70, Δ −10 to −30; DO NOT FIT 85% | 0.631, −13.36, DO NOT FIT | hit |
| excess | negative | +12.29 | miss |
| part 2 | THE ORGANS COST MORE 70% | yes | hit |
| halves | top-1 0.70-0.85, 0.75-0.88; early layers cost more | 0.836, 0.837; no clear gap | hit, miss |
| ph_both | top-1 under 0.45 | 0.048 | hit |

## Thesis line

Every token reads these organs, and left stored they cost 16 ms (attention) and 3.9 ms (FFN organs) on the desk. With this training, ternary in them breaks the quality budget. So the 24.7 line depends on a better training of these organs, or on paying for them in milliseconds.

## Parts specific to Ling

The fused QKV with q/k norms and RoPE on half of each head, one dense layer, a shared expert of 512 beside 256 routed experts, 20 layers, and the numbers. The quantizer, the loss and the sequential training need only RMSNorm before each block.

## Recommendation

**Part 2's first mechanism test goes on the organs, on attention and FFN trained together.** Experts can buy back their loss with an exact slot at a known price (test 3b); these organs have no such handle. The delivery prices this against test 4's training rates.

**What would disprove it:** a longer st_both on layers 0-5 that does not beat this arm's block loss and top-1 on those layers, which would say the tokens are not what is missing.
