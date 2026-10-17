# C50-A5 part 6 protocol: a rotated preview with a residual-norm bonus, confirmed on the unread DEV-B

17 October 2026. This is the owner's option A after the A5 FAIL: new head routes on FIT under a
registered protocol, then a confirmation on a set not yet read. Part 5 failed on FIT, so DEV-B
(DEV items ranked 33-44 per category, 36 documents, built by `c50_holdout_ext.py`) is still
unread. Pushed before any statistic of this part is computed.

## Idea: the measured failure mode of part 2

- **Where part 2 failed.** The selected preview (rotated r512, 3.5-bit, K = 32,768) failed DEV
  on confident tokens: the median p(top1) at a miss was 1.00.
  - A low-rank preview misses a row whose weight vector lies largely outside the
    rank-r subspace. Its preview logit omits a term that can be large.
- **The omitted term is bounded.** For row v, the exact logit is
  w_v·h = (w_v V_r)(V_r^T h) + res_v·h⊥.
  - Here res_v = w_v − (w_v V_r)V_r^T, and h⊥ is h minus its projection on the subspace.
  - Cauchy-Schwarz bounds the omitted term by ‖res_v‖ ‖h⊥‖.
  - The extra data costs one norm per row, 157,184 × float16 = 0.31 MB. ‖h⊥‖ is free:
    ‖h⊥‖² = ‖h‖² − ‖V_r^T h‖², because V_r is orthonormal.
- **Candidate score:** s_v = preview_v + c · ‖res_v‖ · ‖h⊥‖.
  - The K highest s_v are reranked with the exact Q6_K rows.
  - c = 0 is part 1b. c = 1 is the Cauchy-Schwarz bound itself; it is loose in 1,536 or
    1,024 residual dimensions.
- **Gate (unchanged from A5):** teacher-forced top-1 agreement with the exact head ≥ 0.999.
  BPB and PIQA keep the exact head (A5 rule).

## Grid (fixed now)

- **Preview:** rotated r ∈ {512, 1024}, same eigenvectors and q3sim storage (3.5 bit) as
  part 1b.
- **Norms:** ‖res_v‖ comes from the dequantized Q6_K rows and the unquantized projection,
  stored and used as float16.
- **c ∈ {0, 0.025, 0.05, 0.1, 0.2, 0.4}.**
- **K ∈ {2,048; 4,096; 8,192; 16,384; 32,768}.**
- **Bytes per token:** V·r·3.5/8 (preview) + D·r·4 (V_r) + V·2 (norms, only when c > 0)
  + K × 1,680 (rerank).
  - r512: 39.4 MB (39.7 with norms) + K × 1.68 kB.
  - r1024: 78.8 MB (79.1 with norms) + K × 1.68 kB.
- **Known before this protocol** (part 1b, FIT, c = 0): r1024 q3sim at K = 16,384 has recall
  0.99967, at 106.3 MB.
  - So a qualifying configuration exists at ≤ 106.3 MB by construction.
  - The question is whether c > 0 makes the head cheaper.
  - If the selection is that part-1b point, it is still confirmed on DEV-B, because DEV-B is
    unread.

## Data, controls, statistic, selection (fixed now)

- **FIT statistic:** on the 79,709 FIT12 positions captured by the engine (`a5/FIT12.hid.f32`,
  `FIT12.pos.bin`, exact top-1 as ground truth), the fraction whose exact top-1 has fewer
  than K rows with a strictly higher s_v. This is the same rank code as parts 1 and 1b.
- **Reproduction control, VOID if it fails:** the c = 0 rows must reproduce part 1b's
  recorded recall for r512 and r1024 q3sim at K = 2,048-32,768
  (`results/c50_a5_offline_1b_fit.json`) to 1e-9.
- **Selection:** the configuration with the fewest bytes per token whose FIT12 recall is
  **≥ 0.9995**. Ties go to the smaller c.
  - The margin over the gate is the same one fixed in part 5.
  - If none qualifies (impossible unless the control fails, see above), part 6 FAILS on FIT
    and DEV-B is not read.
- **Freeze before DEV-B:** the selected (r, c, K), plus the sha256 of the V_r, preview and
  norm arrays, are committed and pushed before any DEV-B capture exists.
- **Confirmation, once:**
  - `c50_engine_a5.exe --dump-hidden` captures DEV-B (`devb_ids.tsv`, sha256 in
    `results/c50_holdout_ext_v1_manifest.json`; CONFIG asserted).
  - `c50_a5_residual.py --confirm` reads the frozen arrays (sha checked) and prints the gate
    line first: **PASS iff the DEV-B recall ≥ 0.999.**
  - Then, reported only: the other grid points, recall per category, and the confidence of
    the misses.
- **Data not used:**
  - DEV12 (spent by part 2);
  - DEV items 13-32 (reserved for A6);
  - RESERVED.

## When it runs

- **FIT (offline, ~10-20 min of numpy at 2 BLAS threads):** this can run while A4 runs. It
  reads no timing, and A4 is a quality run.
- **The DEV-B capture (~37 min, six engine threads) does not overlap A4 or A6.** Their
  budgets have about 1 h of margin, so it runs after the A4 → A6 chain.

## Budget arithmetic (DESK-MODEL, bytes only, not a rate)

On the fastest A6 arm after reprice (E4B, 26.18 ms desk at 2048 with the exact head), a head of
H MB replaces 264.1 MB: ms = 26.18 − 8.25 + H/32.
- At H = 60, that is 19.8 ms, i.e. 50.5 desk tok/s. At 0.85 of desk, 42.9.

## Registered prediction (scored, not a gate)

- **Selection:** a c > 0 configuration beats the part-1b point. It is r512 with c in
  0.05-0.2 and K 8,192-16,384, at 53-68 MB/token.
- **c = 0.4 is worse than the best c at equal K:** the bonus starts to crowd out the preview.
- **DEV-B recall of the selection:** 0.9990-0.9997.

## What this cannot say

- **No generation-quality claim:** the measurement is under teacher forcing only.
- **No rate**, and nothing on a C implementation (that is part 3, only after a PASS).
- **One mix:** FIT and DEV-B share the one-third-per-category mix of the holdout.
