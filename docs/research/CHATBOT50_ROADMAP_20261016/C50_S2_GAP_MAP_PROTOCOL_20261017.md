# C50-S2 protocol: the Ling-mini → Ling-flash gap map

17 October 2026. Cell S2 of the [CHATBOT50 roadmap](ROADMAP.md) §5, Phase 1 (the old C0). Pushed before any S2 arm
is built or run.

## Question

- **What S2 decides:** is the quality gap between Ling-mini-2.0 (T0's family, 16B total / 1.4B active) and
  Ling-flash-2.0 (the same family, 103B total / 6.1B active) mostly closed-book knowledge or mostly reasoning?
- **Why it matters:** stored capacity at a fixed active cost (S4-S5) can only plausibly close a knowledge gap.
  - If the gap is mostly reasoning, active compute is the right tool. The quality half of Phase 1 is then re-planned
    before any T4 spend (ROADMAP §5 S2).
  - S2 also writes the donor-gap panel that S4 and S5 are judged on.

## Arms

All arms run through the same pinned llama.cpp 5b335f4 tools (`c50_llama_score_avx2.exe`,
`c50_llama_generate_avx2.exe`, the A4 binaries) with the A4 settings: 6 threads, ubatch 512, flash attention
disabled, CONFIG asserted.

| Arm | Artifact | Role |
| --- | --- | --- |
| M0 | Ling-mini-2.0 `a810f64`, `llama-quantize` Q4_K_M from the A2 BF16 GGUF, **no imatrix** | the gap map's baseline |
| F0 | Ling-flash-2.0 `96f9b7a`, `convert_hf_to_gguf.py --outtype bf16`, then `llama-quantize` Q4_K_M, **no imatrix** | the donor |
| T0 | P0 (`44de7315…`, Q4_K_M with the FIT imatrix) | S4's base; reported beside, and used for the donor-gap panel |

- **Why no imatrix (a measured constraint, not a preference):**
  - Ling-flash in BF16 is 205.8 GB. It neither fits the 80 GB of RAM nor reads fast from the USB disk E:, so
    P0's recipe (an imatrix computed on the BF16 model) cannot be repeated on it.
  - The gap map therefore compares two artifacts made by the same recipe, M0 and F0, so that the recipe cannot pass
    for size (rule 9).
  - T0 − M0 is reported as the imatrix effect.
- **Pins:** the M0 and F0 hashes are recorded at build time, and the runner refuses any other file.
- **IDs:** one ID file per panel serves every arm. Ling-flash's tokenizer equals Ling-mini's as canonical JSON, and
  `tokenizer_config.json` (with the chat template) and `special_tokens_map.json` are byte-identical (ROADMAP §12).

## Panels (frozen now)

| Panel | Source (pinned) | Items | Tool | Scoring |
| --- | --- | --- | --- | --- |
| nq | NQ-open validation `5dd9790a`, first 600 in file order | 600 | generate, ≤ 32 tokens | short-answer match |
| tqa | TriviaQA rc.nocontext validation `0f7faf33`, first 600 in file order | 600 | generate, ≤ 32 tokens | short-answer match |
| gsm8k | A4's GSM8K input (first 100) | 100 | generate, ≤ 384 tokens | A4 check |
| arc | A4's ARC-Challenge input (first 600) | 600 | score | A4 likelihood rule |
| mmlu | MMLU all/test `c30699e8`, first 10 per subject | 570 | score | A4 likelihood rule, prefix "Question: …\nAnswer:", suffix " {choice}" |
| docs | A4's 72 non-imatrix FIT documents | 72 | score | paired bits per byte |
| chat | A4's IT/EN chat panel | 40 | generate, ≤ 256 tokens | A4 chat check |

**Inputs and scoring:**
- **Inputs:** `c50_s2_tasks.py --nq 600 --tqa 600 --mmlu-per-subject 10` built them. The manifest's TSV hashes are
  `nq 652313d6…`, `tqa e22bef0b…` and `mmlu 0ceea958…`. The A4 inputs are reused unchanged, and the runner checks both
  manifests.
- **Prompt for nq and tqa:** the source chat template with one user turn, "Answer the question with a short answer
  only, without explanation.\nQuestion: {q}".
- **Short-answer match** (`c50_s2_checks.py`):
  - The response's first non-empty line, with a leading "Answer:" removed, is normalized as in SQuAD: NFKC, lower
    case, punctuation deleted, the articles a/an/the removed, whitespace collapsed.
  - The item is right iff some normalized alias occurs in that line as a whole-word span.
  - Exact match is reported beside it.
  - The aliases are NQ's answer list, and TriviaQA's aliases plus its value.
  - `test_c50_s2_checks.py` covers 15 known-answer cases, each rule with a planted negative, and all pass.

## Decision rule (fixed now; `c50_s2_verdict.py`)

- **Pairing.** On each item panel, F0 is paired with M0:
  - fixed = F0 right and M0 wrong;
  - broken = the reverse.
- **Relative error reduction of a family:** RER = (fixed − broken) / (M0's errors), pooled over the family's items.
  - Knowledge: K = nq + tqa.
  - Reasoning: R = gsm8k + arc.
- **The statistic:** D = RER_K − RER_R. Its percentile CI95 comes from 10,000 bootstrap draws (seed 20261016) that
  resample items within each panel.
- **Outcome:**
  - **KNOWLEDGE** iff the CI95 of D lies above 0;
  - **REASONING** iff it lies below 0;
  - **MIXED** otherwise;
  - INCONCLUSIVE if M0 makes no error in a family.
- **Proceed to S3-S5** iff the outcome is not REASONING and the CI95 of RER_K lies above 0. Otherwise the quality
  half of Phase 1 is re-planned before any T4 spend.
- **Reported, not gated:**
  - every arm's accuracy per panel;
  - MMLU and chat;
  - the documents' paired dBPB (F0 − M0 and T0 − M0, document bootstrap);
  - the imatrix effect T0 − M0.
- **Donor-gap panel for S4/S5** (written by the verdict):
  - the items that F0 gets right and T0 gets wrong, on nq, tqa, mmlu, arc and gsm8k;
  - the documents where F0's BPB is at least 0.05 below T0's.
  - S4's protocol splits it before any training. None of its items or documents may enter training data.
- **The rule's own controls, run before this push:**
  - **Planted cases (`--selftest`):** synthetic arms with a knowledge-only gap give KNOWLEDGE (proceed); a
    reasoning-only gap gives REASONING (stop); equal gaps give MIXED (proceed); no gap gives MIXED (stop). PASS.
  - **Null control** on the real pipeline: T0 against itself, on a smoke subset (5 nq, 5 tqa, 57 mmlu, with A4's P0
    outputs for the shared panels). It gives exact zeros: RER 0, dBPB 0 and an empty gap panel.
  - **Disclosure:** that smoke printed P0's own accuracies on A4's panels before the A4 verdict. They enter no
    decision. The A4 gates compare P0 with BF16 at thresholds fixed in the A4 protocol.

## Order, budget and stop rules

**Order** (`c50_s2_run.py`, resumable as A4's runner):
- the arms in the order T0, M0, F0;
- within an arm: nq, tqa, gsm8k, arc, mmlu, docs, chat, so the decision panels come first;
- T0's A4 panels are not re-run. The runner reuses A4's P0 outputs after checking that A4 recorded them COMPLETE
  with the same executable and the same inputs manifest.

**Stop rules:** per process, a projected-time guard of 3 h; per session, `--budget-h`.

**Cost (ESTIMATE, from bytes and A4's measured P0 times, not rates):**

| Step | Cost |
| --- | --- |
| Build M0 | ~10 min |
| Build F0: BF16 conversion | ~1-1.5 h of read and write on the USB disk |
| Build F0: quantization | ~0.7 h |
| T0's new panels | ~1 h |
| M0 | ~3 h |
| F0 | ~12-13 h (about 3 tok/s of generation and 15 tok/s of scoring, against T0's 14 and 60) |

- **In total:** ~18 h of CPU.
- **It runs only when no rate cell runs:** after the A1b + S0 session and never during S1's sweep. F0's 63 GB needs
  the RAM to itself.
- **Owner authorization (rule 7):** the build and every arm run longer than 1 h, so they start only with the owner's
  explicit go.

## Registered prediction (ROADMAP §5; scored, not gated)

- The Ling-mini → Ling-flash gap is larger on closed-book knowledge than on short math: RER_K > RER on gsm8k alone.

## What this cannot say

- **Two members of one family at Q4_K_M without imatrix.** The gap is the gap between those artifacts; the BF16
  models' gap may differ by the quantization loss.
- **The panels are proxies.** The short-answer match misses correct answers phrased unlike every alias, equally for
  every arm. ARC and GSM8K mix knowledge with reasoning.
- **Not a capacity result.** A knowledge gap says that stored capacity could help, not that S4 will close it.

## Addendum 1 (18 October 2026, before F0's second session): F0's time guard

**What happened.** F0's first session (18:05-18:43, owner's terminal) ran its seven panels with the registered
guard: each tool projects its process's time from its first item and stops (exit 3) above 3 h. Six panels stopped
there and chat completed. Every record is kept.

| Panel | F0 first item: wall, projection | M0 first item: wall, projection | M0 whole panel (tool wall) |
| --- | --- | --- | --- |
| nq | 36.1 s, 21,661 s: STOPPED | 2.3 s, 1,366 s | 717 s |
| tqa | 42.7 s, 25,591 s: STOPPED | 1.4 s, 869 s | 500 s |
| gsm8k | 143.9 s, 14,393 s: STOPPED | 17.3 s, 1,734 s | 2,029 s |
| arc | 34.4 s, 82,451 s: STOPPED | 1.0 s, 2,505 s | 1,652 s |
| mmlu | 38.1 s, 153,983 s: STOPPED | 0.9 s, 3,780 s | 2,248 s |
| docs | 209.4 s, 15,110 s: STOPPED | 36.9 s, 2,665 s | 2,569 s |
| chat | 62.4 s, 2,496 s: COMPLETE in 878 s | 1.4 s, 54 s | 150 s |

**The defect.** The guard reads the first item, and on F0 the first item carries the cold start of a 63 GB model.
- On the score panels the positions are identical across arms. F0's first ARC and MMLU documents took 34-42 times
  M0's, while F0 activates about 4.4 times M0's parameters per token.
- On chat, the one F0 panel that completed, F0's tool wall is 5.86 times M0's. F0 also wrote 4,087 tokens against
  M0's 1,898, so per generated token F0 is ~2.7 times slower.
- The projection is pessimistic on M0 too: ARC 2,505 s projected against 1,652 s run; MMLU 3,780 against 2,248.
- The guard was registered as a runaway guard; what it stopped here is a cold start, not a runaway.

**The change, for F0 only.** The tools' projection is turned off (`--max-projected-s 0`). The runner bounds each F0
process by a wall-time limit of **6 h** and records a process it kills as "STOPPED by the wall-time limit (addendum
1)", with its partial outputs kept. T0 and M0 keep the registered 3 h guard; their panels are complete and are not
re-run. The session budget stays `--budget-h 20`.
- **Nothing that scores changes:** the same tools, binaries, settings and inputs; greedy decoding and scoring are
  deterministic, so the guard cannot move a score.
- **Records:** a panel re-run after a non-COMPLETE record keeps the earlier record under `superseded`, never
  relabelled. Chat is COMPLETE and is not re-run.
- **Logic check, before this push:** with the processes simulated, F0's panels run with the projection off and the
  6 h limit, a process past the limit is recorded as STOPPED by it, a STOPPED record moves to `superseded`, a COMPLETE
  panel is skipped, and M0 keeps the 3 h projection.

**Why 6 h (ESTIMATE, not a rate).** From M0's whole-panel walls:
- generation panels at 2.7-5.9 times M0: nq 1,940-4,200 s, tqa 1,350-2,930 s, gsm8k 5,480-11,880 s;
- score panels at 4.4-5.9 times M0: arc 7,270-9,670 s, mmlu 9,890-13,170 s, docs 11,300-15,050 s.
- The largest is ~4.2 h, so 6 h stops a runaway and not these. The six panels sum to ~10-16 h plus six loads of
  ~2 min, inside the 20 h budget and near the protocol's ~12-13 h estimate for F0.

**When.** After A4-bis's runs, one long job at a time. No timing is read, so under rule 7 as amended on 18 October
it needs no owner go.
