# C50-A4 protocol: quality of the P0 artifact against the BF16 source

17 October 2026. Cell A4 of the [CHATBOT50 roadmap](ROADMAP.md) §5. Pushed before any
A4 arm runs on its registered inputs (programme rule 5). The run is projected to take
~13 hours over three arms, so it waits for the owner's authorization (rule 7;
`docs/COMMUNICATION.md`).

## Question

Does the P0 artifact (Q4_K_M with the FIT imatrix, sha256 `44de7315…`) keep the
quality of the BF16 source GGUF it was quantized from? The measure is held-out
document likelihood plus a frozen task, chat and rollout panel, under the
non-inferiority margins frozen below.

## Arms and runtime

- **BF16:** `ling-mini-2.0-a810f64-bf16.gguf` (sha256 `2b1c8f72ff25d152…`).
- **P0:** `ling-mini-2.0-a810f64-q4_k_m.gguf` (sha256 `44de7315…`).
- **Runtime, the same for both arms:** pinned llama.cpp `5b335f4`, AVX2 build, 6
  threads, micro-batch 512, F16 KV, flash attention off.
  - Scoring: `c50_llama_score_avx2.exe`, the A2b scorer; same binary hash.
  - Generation: `c50_llama_generate_avx2.exe` (sha256 `f5314b41…`).
    - Greedy decoding; exact ties go to the lowest ID.
    - Stops on `<|endoftext|>` (156892) or `<|role_end|>` (156895), the
      source `generation_config` EOS set.
- Both arms receive the same source-tokenizer IDs. The runner asserts each
  `CONFIG` line.
- **Planted positive, PC:** `ling-mini-2.0-a810f64-q2_k.gguf`.
  - Built with `llama-quantize --imatrix <imatrix-fit24> <bf16> Q2_K 6`: the same
    source, imatrix and tool as P0, with a much coarser format.
  - sha256 `278738d5cf27b34379d189d32dcd645f1ccc65f6e3d31240214e070b959e33d6`;
    quantized in 414 s; log `gguf/quantize_q2_k.log`.
  - PC runs every panel through the same runtime, and the same verdict judges it
    against BF16 (`--cand-arm pc`).
  - Why it is there: a PASS of P0 is a no-degradation result. It counts only if
    the same gates fail a known-degraded artifact.
- **Why llama.cpp and not the engine:** A2b showed that a change of runtime path
  alone moves ~4% of top-1 choices. A quantization comparison must therefore keep
  the runtime fixed. The engine reproduces this artifact within that floor (A3); its
  own quality is read in A7.

## Inputs (frozen; `c50_a4_tasks.py`, manifest in the work directory and in the run record)

| Panel | Items | Source and pin | Format | Tokens (prefill) |
| --- | ---: | --- | --- | ---: |
| documents | 72 | FIT items 25-48 of each category | full span, every position scored | 165,930 |
| PIQA | 1,838 | STRAT-01 pin (`valid.jsonl` `93503cc9…`, labels `b4192dc3…`) | `Question: {goal}\nAnswer:` + ` {sol}`; mean suffix NLL; ties → option 0 | 131,442 |
| ARC-Challenge | first 600 of 1,172 (file order) | `allenai/ai2_arc` @ `210d026f`, test parquet `62f03257…` | `Question: {q}\nAnswer:` + ` {choice}`; mean suffix NLL; ties → lowest option | 91,146 |
| GSM8K | first 100 (file order) | `openai/gsm8k` @ `740312ad`, test parquet `ee7b8da9…` | chat template; question + "Solve the problem step by step. End with a final line of the form "#### <number>"."; max 384 new | — |
| IFEval subset | 173 | `google/IFEval` @ `966cd895`, `6a85310c…` | all single-instruction items of 13 deterministic types (below); chat template; max 384 new | — |
| IT/EN chat | 40 (20 IT, 20 EN) | `benchmarks/chatbot50/a4_chat_panel_v1.json` | chat template; max 256 new | — |
| rollout | 72 | the same 72 FIT documents | first 256 tokens as prompt; max 256 new | 18,432 |

- **Chat template:** the source template with thinking off, rendered in a
  sandboxed Jinja environment for one user turn plus the generation prompt:
  `<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>{q}<|role_end|><role>ASSISTANT</role>`.
- **The imatrix documents are excluded from every document panel.** The imatrix
  was computed on the first 24 FIT items of each category (A2/A2b addendum 1).
  - This is a deviation from the roadmap text, which says the "96-document
    rollout" is reused. FIT holds only 72 documents outside the imatrix sample,
    and DEV and RESERVED stay unread. So the rollout uses 72 documents with
    STRAT-01's absolute threshold.
- **IFEval types checked:**
  - `punctuation:no_comma`;
  - `change_case:english_lowercase`, `change_case:english_capital`;
  - `startend:quotation`, `startend:end_checker`;
  - `detectable_format:json_format`, `detectable_format:title`,
    `detectable_format:number_bullet_lists`,
    `detectable_format:constrained_response`;
  - `keywords:forbidden_words`, `keywords:existence`;
  - `detectable_content:postscript`;
  - `combination:repeat_prompt`.

  These are the checks of google-research `instruction_following_eval`,
  transcribed in strict mode with no language detection. Every IFEval item whose
  only instruction is one of these types is used; none is chosen by inspection.
- **Checker controls:** `test_c50_a4_checks.py` gives 45/45 known-answer cases,
  each check with a known positive and a planted negative.
  - It documents one limit inherited from STRAT-01: `loop8x3` flags only loops
    whose period divides 8, so a period-3 loop is not flagged.

## Gates (frozen now)

Notation for the task panels: b = items BF16 gets right and P0 wrong, c = the
reverse.

| Gate | Statistic | PASS |
| --- | --- | --- |
| G1 documents | paired dBPB = (bits P0 − bits BF16) / bytes over the 72 documents; document bootstrap 10,000 draws, seed 20261016 | upper CI95 ≤ +0.020 (roadmap §2). ≤ +0.010 is recorded as the preferred bound. Per-category values are reported, not gated. |
| G2 PIQA, G3 ARC, G4 GSM8K, G5 IFEval, G6 chat | correct counts, paired by item | BF16 accuracy above the panel's ability floor (PIQA 0.50, ARC 0.30, GSM8K 0.20, IFEval 0.30, chat 0.75), else INCONCLUSIVE. Then PASS iff **b − c ≤ max(2·√(b + c), 1)** and **P0 ≥ floor(0.90 × BF16)**. |
| G7 rollout | degeneration flags `run32`, `loop8x3`, `empty` (STRAT-01 definitions; the stop ID excluded) | FAIL iff ≥ 3 documents degenerate in P0 and not in BF16 |

**A4 PASS iff all seven gates PASS.** An INCONCLUSIVE gate means A4 is not passed;
it is reported as such and never read as a pass.

**Planted-positive expectation, registered now:** PC FAILS G1 (upper CI above
+0.020) and at least one of G2-G7.
- If PC passes every gate, a P0 PASS is declared uninformative and a design
  review follows (rule 3).
- If PC fails G1 only, the task gates are recorded as unable to see a Q2_K-sized
  loss at these panel sizes.

**Why this task rule.** STRAT-01's `P0 ≥ ceil(0.98 × BF16)` was sized for 1,838
PIQA items. On panels of 40-600 items the same relative margin is about one
standard deviation of the paired difference. An equal-quality artifact would then
fail roughly one time in six: the gate would be measuring its own noise.

The rule here fails an artifact when either holds:
- its net loss is beyond two standard deviations of the paired sign statistic,
  b − c, whose standard deviation is √(b + c) under equal quality;
- it loses more than 10% of the teacher's correct answers, however noisy the
  panel.

STRAT-01's PIQA rule is computed and reported beside G2.

## Order, budget and stop rules

- **Runner:** `c50_a4_run.py`.
  - Order: the P0 arm, then BF16, then PC. Within each arm: documents, PIQA, ARC,
    chat, GSM8K, rollout, IFEval.
  - It is resumable: a panel recorded COMPLETE is not re-run.
- **Stop rules:**
  - Per process, `--max-projected-s` 3 hours from the first item.
  - Per session, `--budget-h`: no panel is started after the budget is spent.
- **Projected cost**, from smokes on this box (not rates):
  - P0 ~3.6 h. Scoring at ~58-69 tok/s; generation ~13 tok/s including prompts.
  - BF16 ~5.9 h. Scoring at ~40-51 tok/s; generation ~7 tok/s.
  - PC ~3.4 h, assuming it runs at P0's speed.
  - BF16 loading takes ~10 min cold.
  - Total ~13 h. The runner can be split across sessions with `--arms` and
    `--budget-h`; completed panels are kept.
- After the runner:
  - `c50_a4_checks.py` scores each arm;
  - `c50_a4_verdict.py` applies the gates;
  - neither chooses anything.
- **Apparatus smoke, done before this push** (session scratchpad; not evidence):
  the full pipeline on a tiny subset of every panel (3 documents; 20 PIQA, 20
  ARC, 3 GSM8K, 5 IFEval, 5 chat, 3 rollout items), on both arms.
  - **Null control, P0 against P0: PASS with exact zeros everywhere** (dBPB 0,
    b = c = 0 on every panel).
  - The first pass stopped in the checker on a wrong key (`item` instead of
    `index` in the rollout metadata). It was fixed, and the checks were re-run
    on the same outputs.
  - The smoke also printed P0-against-BF16 numbers on those few items. They enter
    no decision and set no threshold.

## Registered prediction (ROADMAP §5; scored, not a gate)

- P0 fresh dBPB lies in +0.005..+0.020.

## What this cell cannot say

- **It measures P0 against BF16 on llama.cpp.** It does not measure the engine's
  quality (A7) or rate.
- **FIT text only, probably seen in pretraining.** The holdout's "fresh" means
  unused by the project. PG19, The Stack and public markdown are very probably in
  Ling-mini's pretraining data, so dBPB on memorised text may understate the
  quantization damage on new text.
- **Panel sizes give limited resolution.** GSM8K 100, IFEval 173 and chat 40
  detect large losses, not one-point differences.
- **The IFEval subset is not IFEval.** It covers 13 of 25 types, single-instruction
  items only, with max 384 new tokens. Truncation hits both arms equally.
