# Stage 2, part 1, test 2: one organ of Ling-mini ternary with a scale per row, trained alone. RESULT

2026-11-04. Protocol: `S2B_ONE_ORGAN_TERNARY_PROTOCOL_20261103.md`.
- Pushed `fd7d17c` before any training.
- Amendment 1 (`00dbf44`) was written before any org_attn panel. org_attn's first export is VOID: a manual tensor type
  shifted Q4_K_M's mixture on later layers. The fix pins every other tensor to P0's type.

Records in `benchmarks/stage2/results/`:
- `s2o_train_*.json` and `s2o_export_*.json`, including the VOID export `s2o_export_org_attn_void1.json`;
- `s2o_run.json`;
- `s2o_verdict.json`, from `s2o_verdict.py` (selftest PASS, 11 cases).

## Outcome in one line

**AN ORGAN DOES NOT STAY TERNARY WITH THIS TRAINING: the head.**
- The trained ternary head with one scale per row reaches document top-1 0.820 and KL 0.251, against the line of
  0.90 and 0.10.
- Chat passes, 26/40 with 1 degenerate rollout, against ref's 28 and 0.
- Every other organ is within its share:
  - one layer of attention, of the shared expert and of the routed experts;
  - the whole dense FFN.

## The run

- **Trainings, RTX 3060:** dense 2 min; attention 59 min (34 of them the prefix); shared expert 31 min; routed
  experts 85 min; head 76 min (49 of them the prefix). stderr is empty.
- **Exports:**
  - every replaced tensor is TQ2_0 and bit-exact;
  - the layout matches the HF checkpoint;
  - after amendment 1, no other tensor changed (157 pinned to P0's types);
  - the head's gain ratio error is 0.
- **Panels:** all COMPLETE. The chain ran from 19:33 to 07:00, with a stop and a restart for amendment 1.

## Arms (72 documents against ref; chat on 40 items)

| Arm | Trained held-out loss / zeroed, before → after | lr | Top-1 | KL top-32 | dBPB [CI95] | Chat | Label |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| org_attn | 0.610 → 0.232 | 1e-3 | 0.933 | 0.041 | +0.0026 [+0.0006, +0.0044] | | ONE LAYER WITHIN |
| org_shexp | 0.714 → 0.378 | 3e-3 | 0.941 | 0.031 | +0.0090 [+0.0070, +0.0112] | | ONE LAYER WITHIN |
| org_exps | 0.335 → 0.233 | 1e-3 | 0.939 | 0.033 | −0.0009 [−0.0028, +0.0008] | | ONE LAYER WITHIN |
| org_dense | 0.281 → 0.055 | 1e-3 | 0.920 | 0.064 | +0.0047 [+0.0005, +0.0089] | 28 / 0 | FITS |
| org_head | top-1 0.694 → 0.837, KL 0.692 → 0.201 | 1e-3 | 0.820 | 0.251 | +0.0293 [+0.0249, +0.0342] | 26 / 1 | DOES NOT FIT |
| org_attn_shuf (planted) | | | 0.860 | 0.191 | +0.0032 [−0.0053, +0.0114] | | |
| tern_l10, stage 1 (b), group 256 | 0.352 → 0.227 | 1e-3 | 0.938 | 0.034 | −0.0007 [−0.0026, +0.0010] | | |
| replay | | | 1.0000 | 0.0000 | +0.0000 | | |

Ref's chat is 28 passes and 0 degenerate. Controls: C1 holds (planted top-1 0.860 < 0.90, KL 0.191 > 0.10), and C2
holds.

## Predictions against outcome

| Prediction | Outcome |
| --- | --- |
| org_attn top-1 0.90-0.94, KL 0.04-0.10, WITHIN 55% | **hit**: 0.933, 0.041, WITHIN |
| org_shexp top-1 0.90-0.95, KL 0.03-0.09, WITHIN 60% | **hit**: 0.941, 0.031, WITHIN |
| org_exps top-1 0.93-0.945, KL 0.03-0.045, within 0.01 of tern_l10, WITHIN 90% | **hit**: 0.939, 0.033, +0.0008 |
| org_dense top-1 0.90-0.96, KL 0.02-0.08, chat 24-29, FITS 60% | **hit**: 0.920, 0.064, 28, FITS |
| org_head top-1 0.85-0.93, KL 0.08-0.25, DOES NOT FIT 55% | **label hit**; **miss, worse**: top-1 0.820; KL 0.251, just outside |
| org_attn_shuf top-1 under 0.80 | **miss, better**: 0.860 |
| stage flag 70% | **hit** |

## Read here, not registered

- **The head stops improving early.**
  - At lr 1e-3, the training KL averages 0.198 over steps 240-480, then 0.181, 0.184, 0.180, 0.179, 0.178 to the end
    (1,465 steps, 1.5 M tokens).
  - The held-out KL is 0.201, so there is no overfitting gap to close with more data. The floor is the format or the
    optimizer, not the data.
  - Higher lr is worse (3e-3: 0.278; 1e-2: 0.738). Lower than 1e-3 is untested.
- **One scale per row costs nothing on the experts.** org_exps against tern_l10 (one scale per 256): top-1 0.939
  against 0.938, KL 0.033 against 0.034. Whether it costs the head is not measured.
- **Four of five organs picked the grid's lowest lr.** The grid's edge, not its inside, was best for dense, attention,
  experts and head.
- **The shared expert moves BPB most among the layer organs:** +0.009, with a CI that excludes 0, at top-1 0.941.
- **The planted control is weaker than predicted.** Permuting the rows of one layer's attention output costs top-1
  0.14. One real ternary layer costs 0.06-0.08, so the panel separates them, but not by much.
- **The organs were tested one at a time.** Stacking them is part 2's question; stage 1 (b) showed it costs more than
  the sum.

## Decision (the registered rule)

**The stage flag holds:** before part 2, the Consulente is heard with the organ that does not stay (the head) and what
training would have to buy for it.
- **What the head lacks:** top-1 +0.08 (0.820 → 0.90) and KL −0.15 (0.251 → 0.10).
- **Untrained → trained,** the head gained top-1 0.12 (0.698 → 0.820) and lost KL 0.58 (0.827 → 0.251).

## Thesis line

The desk ceiling at k = 4 is 25.6 tok/s with every organ ternary, and 19.4 with the head as stored, under the 24.7
line. The head reads 157,184 × 2,048 weights per token in Ling-mini: in S0 it took 24.5% of the engine's time at
2048. No route to 24.7 keeps it at Q6_K.

## Parts specific to Ling

These are the protocol's:
- the fused QKV with q/k norms;
- one dense layer;
- a shared expert beside 256 routed experts;
- the vocabulary of 157,184 tokens, which makes the head the largest single matrix.

The per-row training and its loss need only RMSNorm before each block.
