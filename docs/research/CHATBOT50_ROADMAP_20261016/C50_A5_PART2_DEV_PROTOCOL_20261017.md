# C50-A5 protocol, part 2: one DEV confirmation of the selected head approximation

17 October 2026. Cell A5 of the [CHATBOT50 roadmap](ROADMAP.md) §5. Pushed before the DEV
capture starts. DEV is read once (programme rule 4); no threshold moves after it is read.

## What is confirmed

The part 1b selection:
- **Configuration:** `rot_r512_q3sim`, K = 32,768, at 94.45 MB/token.
- **FIT recall:** 0.99921.

The selection is frozen as files before DEV exists (`c50_a5_dev.py --freeze`,
`results/c50_a5_freeze.json`):
- **V_r, 2048 × 512 float32:** sha256 `56a0b409…`.
- **Preview, 157,184 × 512 with q3sim values:** sha256 `921f9d3f…`.
- **Reproduction control:** the frozen files reproduce FIT recall 0.9992096 exactly.

## Capture (fixed rule, the same as FIT)

- **Binary:** `bin/c50_engine_a5.exe`, sha256 `e524dd87…`. Run with `--dump-hidden` on
  the P0 Q4_K_M with the default threads.
- **Documents:** the first 12 DEV items of each category (`c50_make_ids.py --role dev
  --per-category 12`). That is 36 documents and 78,079 positions, ID file sha256
  `ab741ea7…`.
- **Assertion:** the CONFIG line must show `hidden=1` and the model pin.

## Gate (ROADMAP §5, unchanged)

- **Statistic:** teacher-forced top-1 agreement with the exact head. With an exact rerank, that is
  DEV top-1 recall of the engine top-1 within the K = 32,768 candidates.
- **Rule:** **PASS iff the point estimate is ≥ 0.999.** `c50_a5_select.py --grid dev`
  applies it.
- **Control:** the Q8_K oracle on every 10th DEV position. Rank-code recall at 1 must be
  ≥ 0.999, else VOID.

## Registered consequences

- **PASS:**
  - part 3 follows: a C implementation of the preview, the gather and the exact Q6_K
    rerank, joined to the decode;
  - the candidate set also includes the tokens touched by the repetition penalty;
  - agreement with the exact head is measured in the engine itself;
  - the rate is measured on the clean box.
- **FAIL:** A5 fails.
  - The rule does not move to the runner-up (full-rank ternary, K = 256) by itself.
  - A new selection would need a new FIT protocol and a fresh confirmation set, and
    DEV would be spent. So a FAIL goes to the owner, with the two roadmap options:
    keep the exact head (Ling-mini ~41 desk tok/s at P1) or a cheaper donor.
- **The P4 budget breach** (531.7 vs 530 MB, part 1b) stands either way. It is
  re-priced when A6 measures P3/P4.

## Registered prediction (scored, not a gate)

- DEV recall lands in **0.9985-0.9995**.
- Probability of PASS: about **60%**. The FIT margin is 0.00021, and per-document
  correlation widens the sampling error.
