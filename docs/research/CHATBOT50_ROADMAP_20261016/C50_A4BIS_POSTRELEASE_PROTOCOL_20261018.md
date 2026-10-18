# C50-A4-bis protocol: P0 against BF16 on text written after the model's release

18 October 2026. Step 2 of the A4 design review (rule 3); it qualifies T0 = P0 for ROADMAP §5 S0.
Pushed before any arm scores these documents.

## Question

- **What A4 could not see.** On FIT text, A4's document gate G1 passed the planted Q2_K (upper
  CI95 +0.0017). Outside BF16's overconfident tail, Q2_K moved BPB by +0.0002 only
  ([A4 result](C50_A4_QUALITY_RESULT_20261017.md), diagnostic 1 and design review step 1).
  The likely cause: FIT text (PG19, The Stack, public markdown) is very probably in Ling-mini's
  pretraining data.
- **The question here.** On text that Ling-mini-2.0 cannot have trained on:
  - does the unchanged G1 see a measured loss on a planted positive?
  - if it does, does P0 pass G1?

## What stays from A4, and what changes

| A4 gate | In A4-bis |
| --- | --- |
| G1 documents (dBPB upper CI95 ≤ +0.020) | **re-run, unchanged, on the post-release documents**. The FIT result is kept as recorded and decides nothing. |
| G2 PIQA, G3 ARC-C, G4 GSM8K, G5 IFEval subset, G7 rollout | stand as recorded in A4 (PASS); not re-run. Their resolution limits stand. |
| G6 chat | **reported only.** BF16 scores 0.675, under the panel's 0.75 floor, so the outcome is set by BF16's ability whatever the candidate. A deviation from the A4 protocol, taken in the design review. |

- **The fidelity gate of 17 October stays withdrawn** (design review step 1): it would fail a P0
  that loses no accuracy. Its statistics are reported, not gated.

## Inputs (built and frozen before this push)

- **Builder:** `benchmarks/chatbot50/c50_postrelease_corpus.py`, committed (`347d9a0`) before it
  ran; the build ran at `29b28ba`, which adds a response cache so a stopped build resumes and leaves
  the selection rules unchanged. Its docstring holds every rule. In short:
  - **Date pin.** The safetensors of `inclusionAI/Ling-mini-2.0` were last changed by commit
    `b735a17` on 2025-09-08. The later commits, up to the pinned `a810f64` (2026-09-24), touch the
    README and the modelling code only (Hugging Face API, `paths-info` with `lastCommit`). Every
    source is created between **2026-03-01 and 2026-08-31**, at least 5.7 months later.
  - **code:** one `.py` file per GitHub repository. The repositories are created in the window,
    licence MIT or Apache-2.0, language Python, at least 20 stars, no fork, not archived. They come
    from the first 30 results by stars of each 7-day slice and licence.
  - **technical_general:** one `.md` file per repository, from the same candidates, skipping the
    repositories used for code.
  - **prose:** English Wikipedia articles created on the 1st and 15th of each month in the window,
    two per day. The text is the plain-text extract of the current revision (CC BY-SA 4.0). No
    redirect, no disambiguation page, and no creation summary that names a translation, split,
    copy or merge.
  - **Text rules:** UTF-8, NFC-stable, ≥ 90% ASCII, ≥ 8,192 bytes. No copyright line names a year
    from 1990 to 2025. No SPDX identifier other than MIT or Apache-2.0. Dot-directories, tests,
    vendored code, licence and changelog files, and agent-instruction files (`SKILL.md`,
    `AGENTS.md`, `CLAUDE.md` and similar) are skipped.
  - **Order and span.** The walk order is a seeded hash (seed 20261016), and the span rule is
    c50_holdout_v1's: 8,192 bytes, from the pinned `build_document_holdout._bounded_span`.
- **The build:** started 2026-10-18 12:07 inside the session; stopped once by the session host's
  memory-pressure reaper (ROADMAP §12); resumed from its cache by the owner from a terminal outside
  the session; finished 16:21.
  - **Searches:** 54 GitHub searches (27 weekly slices × 2 licences), 1,611 candidate
    repositories.
  - **Rejections** (counted by the builder): 658 short, 59 under 90% ASCII, 1 not NFC-stable, and
    1,771 Wikipedia creations that are now missing (deleted), redirects, disambiguation pages or
    outside the article namespace.
  - **Items:** 72, 24 per category, each from a distinct repository or article.
    - code: 24 `.py` files (15 Apache-2.0, 9 MIT), created 2026-03-09..2026-08-25, 50,472 tokens;
    - technical_general: 24 `.md` files (13 Apache-2.0, 11 MIT), created 2026-03-02..2026-08-31,
      55,592 tokens;
    - prose: 24 Wikipedia articles (CC BY-SA 4.0), created 2026-03-01..2026-08-15, 45,747 tokens.
  - **Totals:** 151,811 tokens (per document 1,506..3,694), against A4's 165,930.
  - **Hashes:** `post.jsonl` `13a5bf46…`, `docs.tsv` `d711246e…`, `docs_meta.jsonl` `d89e396d…`,
    inputs manifest `a102771a…`. All under `E:/…/a4bis_post/`.
- **The record:** `results/c50_postrelease_v1_manifest.json`. Each item has its URL, licence,
  creation date, its pinned commit and git blob (or Wikipedia revid), and its hashes. The search
  results move with time, so a rebuild would not reproduce these items; the manifest is the record.

## Arms and runtime

| Arm | Artifact | Pin |
| --- | --- | --- |
| BF16 | `ling-mini-2.0-a810f64-bf16.gguf` | hash in `c50_a2_gguf_check_bf16.json` |
| PC | Q2_K from the same BF16 and the FIT imatrix (A4's planted positive) | `278738d5…` |
| P0 | T0, Q4_K_M with the FIT imatrix | `44de7315…` |
| PC2 (conditional) | Q2_K from the same BF16 with **no** imatrix: the same format, coarser | pinned in `c50_a4_run.py` at build, before it scores |

- **Runner:** `c50_a4_run.py --panels docs --cell C50-A4-bis --work E:/…/a4bis_post`. The tool,
  context (8,192), threads (6) and micro-batch (512) are A4's, and the runner asserts its CONFIG
  line.
- **Verdict:** `c50_a4bis_verdict.py`. It applies the rule below and chooses nothing else.

## Gate and order (frozen now)

- **G1, unchanged:** paired dBPB against BF16 over the 72 documents, document bootstrap 10,000
  draws, seed 20261016. PASS iff upper CI95 ≤ +0.020 (roadmap §2).
- **The plant fires** iff its G1 upper CI95 > +0.020 **and** its lower CI95 > 0. A wide interval
  is not enough; the plant must show a measured loss.
- **Order:**
  1. BF16, then PC. Then `c50_a4bis_verdict.py --plant pc` (stage "plant").
  2. If PC fires: run P0, then the final verdict with `--plant pc --cand p0`.
  3. If PC does not fire: build PC2, pin it, run it, then the plant stage on PC2. If PC2 fires,
     run P0 and the final verdict with `--plant pc2`.
  4. If PC2 does not fire either: **INSTRUMENT BLIND**. P0 is not run, no A4 verdict is drawn,
     and design review 2 is written before any new cell.
- **Outcomes:**
  - **A4 PASS:** P0 passes G1. T0 = P0 is qualified, and S0 follows.
  - **A4 FAIL:** ROADMAP §5 S0's branch applies: Q5_K_M, then Q6_K, each through this same cell.
- **Why PC2 is registered now.** The design review requires a coarser plant, fixed before P0's
  result is read. Q2_K without an imatrix keeps the format and drops the importance weighting, so
  its loss is a known step beyond PC's.
- **Reported, not gated:**
  - top-1 disagreement with BF16, top-1 accuracy, mean |Δ log p|, and the tail;
  - G1 by category;
  - BF16's BPB by category here, beside A4's FIT values.

## Budget and run rules

- **From A4's document panel** (72 documents, 165,930 tokens): P0 2,714 s, BF16 4,850 s, PC
  2,483 s. Scaled to 151,811 tokens, that is **~2.6 h** for the three arms (ESTIMATE). PC2 adds
  ~20 min of build and ~45 min of scoring. The session budget is `--budget-h 5`.
- **The owner's go is required** (rule 7: over 1 h).
- **No timing is read,** so the box need not be clean. One long job at a time still holds: A4-bis
  starts after S2's panels end (S2 runs from 18 October ~14:00) and never beside them.
- **Apparatus smoke, before this push** (a null control, 18 October 17:04-17:07): a 3-document
  subset of the frozen inputs, the first document of each category (indices 0, 24 and 48;
  6,502 positions), in `E:/…/a4bis_smoke/`.
  - P0 through `c50_a4_run.py --panels docs --cell C50-A4-bis-smoke`: COMPLETE, CONFIG
    confirmed, 143 s.
  - A second, independent P0 process with the same command, written into the BF16 slot: CONFIG
    equal, and `docs.pos.bin` byte-identical to the first.
  - `c50_a4bis_verdict.py --plant p0` on the pair: dBPB +0.00000, CI95 (+0.00000, +0.00000),
    **PLANT DOES NOT FIRE**. A null therefore cannot fire the plant rule.
  - The smoke ran beside S2's M0 panels; neither reads a timing.
  - Seen in the smoke: P0's BPB on these 3 documents is 0.7165. No BF16 or plant score on the
    post-release text has been seen.

## Registered predictions (scored, not gates)

- PC's dBPB point estimate on this text lies in **+0.015..+0.080**, and PC fires.
- P0's dBPB point estimate lies in **+0.002..+0.015**, and P0 passes G1.
- PC's top-1 disagreement with BF16 exceeds P0's, as it did on FIT.

## What this cell cannot say

- **"Written after the release" is not "never seen".** A new file or article can copy older
  text. The copyright-year and Wikipedia-summary rules remove the declared cases, not all of
  them.
- **Some of this text was probably written with a language model.** No reliable detector exists,
  and none is applied.
- **The genres differ from A4's.** Wikipedia articles replace PG19 books as prose, and 2026
  READMEs and docs replace public markdown. The per-category values are reported only.
- **It measures P0 against BF16 on llama.cpp,** on 72 documents. It does not measure the engine's
  quality (A7) or its rate.
