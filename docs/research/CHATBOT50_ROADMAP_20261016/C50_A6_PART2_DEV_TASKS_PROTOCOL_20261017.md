# C50-A6 part 2 protocol: the FIT candidate confirmed once on DEV documents and on the A4 task panels

17 October 2026. This is the second protocol announced by the
[A6 FIT protocol](C50_A6_PRECISION_LADDER_PROTOCOL_20261017.md). **It is pushed before the A6 FIT
scoring has produced any number**: the A6 run is queued behind A4 and has not started. So the
rule below is fixed before the FIT candidate is known.

## Arms

- **R:** P0 (Q4_K_M, A2 pin `44de7315…`), KV f16.
- **C:** the arm named `fit_candidate` in `results/c50_a6_verdict.json`, with its build and KV as
  registered in `c50_a6_run.py`. The GGUF sha256 must equal the A6 run record.
- **If the FIT candidate is R or none, part 2 does not run:** A6 stops at P0, as the roadmap
  says.
- **Why R is rerun and not taken from A4:** C needs flash attention for a q8_0 V cache, and A4
  ran with flash attention off. A runtime-path change alone moves ~4% of top-1 choices
  (A2b). So both arms go through one runtime here.
- **Runtime:**
  - `c50_llama_score_a6_avx2.exe` (the A6 scorer, sha256 `84763138…`).
  - `c50_llama_generate_a6_avx2.exe`: the A4 generator plus the same two printed options,
    `--kv` and `--fa`. sha256 `e4592d2f…`, built by `c50_build_scorer.sh` against the
    pinned `5b335f4` libraries.
  - Settings: flash attention on, 6 threads, ubatch 512, greedy with ties to the lowest ID,
    stop IDs 156892 and 156895.
  - The runner (`c50_a6_dev_run.py`) asserts on each CONFIG line: AVX2, threads, ubatch,
    `flash_attn=enabled`, the KV type, the model path and the stop IDs.
- **Generator smoke** (two chat prompts, 8 new tokens, P0 with q8_0 KV): it runs, and CONFIG
  and the record print `flash_attn=enabled kv=q8_0`. No statistic was computed.

## Inputs (`c50_a6_dev_inputs.py`, manifest in the work directory)

| Panel | Items | Source |
| --- | --- | --- |
| DEV documents | 60 (DEV items ranked 13-32 per category), 130,621 tokens, max 2,904 | `c50_holdout_v1/dev_ids.jsonl`; `docs.tsv` sha256 `6ee8dc4b…` |
| PIQA, ARC, GSM8K, IFEval, chat, rollout | as A4 | the A4 input files, copied after checking them against the A4 manifest |

- **DEV items not used here:** 1-12 were read by A5 part 2, and 33-44 are A5's DEV-B.
  RESERVED stays unread.
- **The task panels are not fresh.** A4 runs them on P0 and BF16 before this part. They are
  reused here as fixed instruments, not as unseen data: no task threshold below depends on an
  A4 result.

## Gates (frozen now)

| Gate | Statistic | PASS |
| --- | --- | --- |
| D1 DEV documents | paired dBPB = (bits C − bits R) / bytes over the 60 documents; document bootstrap 10,000 draws, seed 20261016; the A2b/A3/A4 byte count | upper CI95 ≤ **+0.015** (ROADMAP §5 A6) |
| T2-T6 PIQA, ARC, GSM8K, IFEval, chat | the A4 task rule with R as reference and C as candidate (`c50_a4_verdict.task_gate`) | R above the ability floor, else INCONCLUSIVE; then b − c ≤ max(2·√(b + c), 1) and C ≥ floor(0.90 × R) |
| T7 rollout | the A4 degeneration flags | FAIL iff ≥ 3 documents degenerate in C and not in R |

- **A6 is CONFIRMED iff every gate PASSES.** INCONCLUSIVE is not a pass.
- **On NOT CONFIRMED:** the rule does not move to another arm by itself. DEV items 13-32 are
  then spent for A6, and the decision goes to the owner (COMMUNICATION).
- **Verdict:** `c50_a6_dev_verdict.py` applies the gates after `c50_a4_checks.py` scores each
  arm. Neither chooses anything.

## Order, budget, stop rules

- **Order:** R, then C. Within each arm: documents, PIQA, ARC, chat, GSM8K, rollout, IFEval.
  - Per process, the projected time is capped at 3 h.
  - Per session, `--budget-h 9`.
  - The runner is resumable.
- **Projected cost** (from the A4 projections, not rates): ~3.5 h per arm, ~7-8 h in total.
  - IQ3_S experts are compute-bound (A1), so an E3/E3B candidate may be slower.
  - **It needs the owner's authorization** (rule 7). It runs only after the A4 → A6 chain,
    never beside it.

## Registered prediction (scored, not a gate)

- **FIT candidate:** E4B or K8 (as registered in the A6 FIT protocol).
- **D1:** the DEV dBPB of C lies within ±0.005 of its FIT dBPB.
- **Tasks:** every task gate PASSES. |b − c| ≤ 6 on PIQA and ARC, and ≤ 4 on GSM8K, IFEval and
  chat.

## What this cannot say

- **No rate:** bytes and desk ms only (A7 measures the rate).
- **llama.cpp, not the engine:** the engine's quality is read in A7.
- **Panel resolution:** as in A4, the task panels detect large losses, not one-point
  differences.
