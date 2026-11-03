# Stage 2, part 1, test 1: Ling-mini's head ternary with a scale per row, untrained. RESULT

2026-11-03. Protocol: `S2A_HEAD_TERNARY_PROTOCOL_20261103.md`.
- Pushed `62086c2` before any panel.
- Amendment 1 (`6202162`, before the rerun): the first head_tern_col and head_shuf builds are VOID. The final norm's
  gain was not divided by c.

Records in `benchmarks/stage2/results/`:
- `s2h_build_*.json`, including the VOID builds `*_void1.json`;
- `s2h_run.json`, with the VOID run under `void.run1`;
- `s2h_verdict.json`, from `s2h_verdict.py` (selftest PASS, 9 cases), and `s2h_verdict_void1.json`.

## Outcome in one line

**HEAD NEEDS TRAINING.** Untrained, the ternary head with one scale per row is far from its 1-point share:
- **head_tern_col:** document top-1 0.646 and KL 1.02, against the line of 0.90 and 0.10; chat 3/40 with 14 degenerate
  rollouts, against ref's 28 and 0;
- **head_tern_im**, without the column scale, does better on top-1 (0.698) and KL (0.83).

## The run

- **Builds (rerun):**
  - head_tern_col `6fe70836…` and head_shuf `4cbbd37d…`;
  - head_tern_im `ef30374e…`, from the first run, which stands.
  - Every head is TQ2_0 and bit-exact, and every other tensor equals P0's.
  - C2_gain: P0's gain over the file's matches c to within 5.7e-8 (the line is 1e-6).
- **Panels:** all COMPLETE.
  - Documents: 41, 37 and 47 min.
  - Chat: 10 min.
  - stderr is empty.

## Arms

| Arm | Head | Top-1 | KL top-32 | dBPB [CI95] | Chat /40 | Degen. |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| ref | P0 (Q6_K) | 1 | 0 | 0 | 28 | 0 |
| head_tern_col | ternary, column scale α = −0.35 in the final norm | 0.646 | 1.021 | +0.275 [+0.262, +0.289] | 3 | 14 |
| head_tern_im | ternary, no column scale | 0.698 | 0.827 | +0.280 [+0.256, +0.304] | | |
| head_shuf | head_tern_col's rows permuted (planted) | 0.00002 | 26.5 | +10.36 | | |
| replay | P0 again | 1.0000 | 0.0000 | +0.0000 | | |
| top6, stage 1 (anchor) | P0 | 0.898 | 0.094 | +0.019 | 29 | |

Controls: C1 holds, with head_shuf's top-1 1.8e-5 under the 0.10 line. C2 holds. The replay is exact.

## Predictions against outcome

| Prediction | Outcome |
| --- | --- |
| NEEDS TRAINING 55 / 45 | **hit** |
| head_tern_col top-1 0.86-0.94, KL 0.06-0.25 | **miss, worse**: 0.646, 1.02 |
| head_tern_col chat 22-30 passes, at most 3 degenerate | **miss, worse**: 3 passes, 14 degenerate |
| head_tern_im worse than head_tern_col on top-1 and KL | **miss**: better on both (0.698 against 0.646, 0.83 against 1.02) |
| head_shuf top-1 under 0.01 | **hit**: 0.00002 |

## Read here, not registered

- **The weighted proxy chose the wrong arm.**
  - The proxy (Σ h (w − t)² over Σ h w², h = the imatrix) put head_tern_col at 0.100 and head_tern_im at 0.169. The
    panel ranks them the other way on top-1 and KL.
  - dBPB is equal within the CIs: +0.275 against +0.280.
  - The proxy is diagonal. It ignores the correlations of the inputs, and the 8-bit quantization of the activations in
    the dot product: TQ2_0 runs against q8_K, and IQ2_BN against Q8_K64.
  - At α = −0.35 the column scale multiplies the largest inputs.
    - Dimension 1212 is multiplied by 2.88. The median dimension is multiplied by 0.98.
    - Dimension 1212's rms against the median dimension's goes from 22 to 64. That raises the step of the
      activation's 8-bit block.
  - Neither cause is measured here. Test 2 trains the head without the column scale, by the rule registered in its
    protocol.
- **Chat breaks harder than the documents.**
  - head_tern_col has 14 degenerate rollouts out of 40.
  - The failure that arXiv 2608.28003 reports for an INT8 head (incoherent, repetitive generation) shows here at
    ternary as well.
  - Top-1 0.65 on the documents means a different greedy token about one time in three.
- **The head's whole error is in the untrained fit.**
  - Nonzero share: 0.18 for head_tern_im, 0.53 for head_tern_col.
  - Row error, unweighted: 0.57 for head_tern_im, 0.21 for head_tern_col.
  - No fit of one scale per row came near the line. The protocol's rule 5 holds: an untrained miss does not show that
    the head cannot be ternary.

## Decision (the registered rule)

- **HEAD NEEDS TRAINING:** the head joins test 2's one-organ trainings as org_head. It is the ternary head trained
  alone on the final norm's outputs, against the original head's logits (KL).
- **No column scale:** head_tern_col's top-1 is not above head_tern_im's, so c = 1, per test 2's rule.

## Thesis line

With the head as stored, the desk ceiling at k = 4 is 19.4 tok/s; with every organ ternary, 25.6. The gap is 6.2 tok/s,
and the untrained head does not close it. Whether a trained head closes it is test 2's org_head.

## Parts specific to Ling

These are the protocol's:
- the vocabulary;
- the width;
- the dominant input dimension 1212;
- P0's Q6_K head as the reference.

The proxy's misranking depends on how few input dimensions dominate. Models with massive activations before the head
share this.
