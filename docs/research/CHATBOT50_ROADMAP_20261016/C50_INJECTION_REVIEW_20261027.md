# C50 design review against the thesis: targeted injection of F0's facts (a mechanism test)

27 October 2026. Ordered by the [review directive](C50_DIRETTIVA_REVISIONE_20261027.md) (section 3, point 5), before
any new training.
- **What this is:** a design, not a protocol. The protocol (gates, code, MDE) follows it, and is reviewed again (rule
  6) because it spends more than 10 GPU hours.
- **Independent review:** "proceed after fixes", with 14 objections, all taken (table at the end).

## The thesis, and why no cell so far could speak to it

**Thesis (ROADMAP §0):** F0's knowledge, held as auxiliary experts on v0, raises held-out quality while bytes per token
stay fixed and stored capacity grows. 2× is +15.3 B parameters, about +9.2 GB at v0's Q4 mix.

Three things stood in the way:
- **R measured flatness.** In cell M, R_acc(v0) is 0.231, and one temperature on T0 takes 0.64 of R.
- **nq and tqa ask random facts,** and block 1 holds about 3,000 of about 7 M Wikipedia pages.
- **N's gate fired on 98% of tokens,** because no fire cost was trained.

A cell that can speak to the thesis needs facts the organ has seen, accuracy as the metric, and a gate that is sparse
and selective from the first step.

## The candidate

| Part | Choice |
| --- | --- |
| **Facts** | S2's donor-gap items on nq and tqa (F0 right, T0 wrong: 98 + 117), plus the same rule on T1 part 2b's fresh panel (600 + 600; T0 already scored, F0 needs about 1 h). Pool about 430 before filters. MMLU and ARC are out: a 25% guessing floor, and many items are reasoning |
| **"v0 does not know"** | v0 sampled 8 times at temperature 0.7; keep items with 0 of 8 right (Gekhman et al.'s "Unknown") |
| **Filter, then split** | F0 writes one passage per item from the question alone, never seeing the gold answer. Items whose passage has no gold alias are dropped. Then comes a seeded split, stratified by panel, into **injected** and **twin**. The filter is the same for both halves |
| **Minimum size** | at least 100 items per half, otherwise more nq and tqa items are scored before anything trains. The protocol fixes the MDE (about 0.09 on E − 0 at 100-150 per half) |
| **Text** | for the injected half, F0 writes 4 more passages (encyclopedia sentence, description, a Q&A pair, a list item, a dialogue turn). A passage without a gold alias is dropped |
| **Questions** | two rewordings per item, written from the original question: one by F0 in a separate call, one by T0, which does not know the answer. A rewording is rejected if it holds an alias or shares an 8-gram with any training string |
| **Reserved** | the twin half is held out from now on. The S2 donor-gap panel stops being a clean held-out set (recorded) |

**Arms** (on v0, same text and steps):

| Arm | What | Role |
| --- | --- | --- |
| **E** (2 seeds) | N's branch (4 SwiGLU experts of T0's size in each of 19 layers, down zero) with ReMoE's adaptive L1 on the gate. The router bias is set so that fire = 0.10 at step 0, and λ0 is calibrated in an MVE | the thesis's organ |
| **E-dense** | the same branch with no fire cost | separates "the sparsity" from "the branch" |
| **L** (2 seeds) | LoRA, S4's c48 configuration | can fine-tuning inject these facts at all? |
| **QA1** | trained only on "original question → gold", one line per fact | the low end of text per fact, and the string-recall control |
| **S** | E on the same text with each gold alias swapped for the answer of another injected item of the same answer type | the gain must come from the fact |
| **0** | v0 untouched | calibration: the split's own noise |
| **E-rot** (no training) | E with expert e moved to slot (e + 1) mod 4 | do the facts live in specific experts? |
| **Open book** (no training) | v0 with one F0 passage in the prompt | planted positive for the questions and the scorer |

**Training:**
- **Loss:** cross-entropy on F0's passages, mixed 1:1 with block 1's KL33 replay.
- **Exposures:** 100 per fact (20 epochs over 5 passages), with checkpoints at 25, 50 and 100. The checkpoint curve is
  the dose curve.

**Metrics:**
- **Primary:** Δ = acc(injected) − acc(twin) on the rewordings, greedy, S2's alias scorer, with a bootstrap clustered by
  fact. E − 0 and E − S are the contrasts. acc(injected) and acc(twin) are each reported against arm 0.
- **Stored or extractable** (Physics 3.1's split):
  - cloze accuracy on the training passages;
  - accuracy on the original question.
- **Fire, the thesis's invariant:**
  - the fire rate on the replay shards, the injected prompts and the twin prompts;
  - a ceiling of 0.15 on replay, priced in bytes per token at the measured rate (0.15 × 36.1 MB ≈ +0.6% of v0);
  - STOP if any layer drops below 0.01 (a dead branch).
- **Guards,** read at every checkpoint (Gekhman et al.: early stopping):
  - A4's paired rule on fresh-panel items v0 knows;
  - the general panels (docs R_acc, gsm8k, chat).
- **Answer bag:** how often an arm answers a twin question with an injected answer.
- **For the thesis:** F0 tokens per extracted fact, along the dose curve and at QA1.

**Budget (ESTIMATE):**
- **F0 locally,** after block 2 (about 28 October afternoon): about 1 h of scoring, about 8 h of generation, and a
  30-minute batched-throughput measurement.
- **About 30 T4 × 2 session-hours:** 7 trainings of about 3 h, the MVE, and checkpoint evaluations.

## Scale: what the thesis would cost in teacher text

- **The thesis needs quality to rise at 2×, not 2× saturated with facts.**
  - The count that matters is facts F0 knows and T0 lacks.
  - S2's rate is 16-20% of random nq and tqa questions. The absolute count is unknown.
- **The capacity bound** comes from Physics 3.3 (synthetic biographies, GPT-2, read in §5-§10):
  - 2 bits per parameter at 1,000 exposures and 1 at 100;
  - int4 at 0.7 bit, a figure from the 1,000-exposure regime;
  - MoE about 1.3× lower.
  - At about 25 bits per fact (ESTIMATE), 2× holds at most about 3.3 × 10⁸ facts. **This is an upper bound on what
    saturation needs, not a cost floor.**
- **Text per fact in the literature:**
  - Mecklenburg et al.: 10 Q&A pairs per atomic fact.
  - EntiGraph: about 350 synthetic tokens per source token, with accuracy rising log-linearly up to 455 M tokens.
  - The cell measures this project's number between QA1 (about 15 tokens) and K = 5 (about 350).
- **At the measured rates, F0 writes about 9.5 × 10⁷ tokens a year** (3 tok/s generating). That is
  2.7 × 10⁵ facts a year at 350 tokens per fact, and 6.3 × 10⁶ at 15.
  - Scoring runs at about 90 tok/s, which is about 2.8 × 10⁹ tokens a year.
  - **On this machine, at these rates, F0's text cannot reach a large fraction of 2×.**
- **Batched decoding has never been measured.** The 30-minute run prices it.
- **Which teacher feeds the thesis** (batched or rented F0, or corpus text with F0 as a filter) **is the user's
  decision.** This cell's tokens-per-fact number is what turns it into a price.
- **Not tested here:** this cell does not test addressing inside a large pool at fixed active bytes (4 experts per
  layer, top-1). That is the mechanism the thesis needs at 2×.

## What it takes from the design (rule 1)

| Taken out | Why |
| --- | --- |
| **Facts per GB, measured on an attached branch** | Even a 1 M-parameter branch needs about 10⁸ training tokens through the 16 B base to saturate (≥ 150 T4 hours). It is the next cell if E passes, on a cheaper path. Until then, the bound is the literature's (GPT-2 from scratch), which says nothing direct about a branch on a frozen base |
| Thesis scale | about 200-300 facts against about 10⁸ |
| MMLU, ARC, GSM8K and docs donor-gap items | a guessing floor, and reasoning; docs have no single answer |
| F0's errors | only alias-verified passages are kept |
| E-K2 (2 passages per fact) | replaced by the dose curve and QA1, which price the text better |

## Outcomes and the decision each one changes (rules 4, 5)

**No outcome reaches the thesis's scale.** This is a mechanism test with a mechanism-test budget.

| Outcome | Reading | Next |
| --- | --- | --- |
| E passes, sparse and selective (injected fire > replay fire), guards hold | the organ stores and extracts F0's facts | capacity cell; the teacher decision goes to the user with a price |
| E passes, fire not selective | a sparse adapter, not memory | router and addressing design before any scale |
| E fails, E-dense passes | the fire cost stops learning | one cell on the fire schedule |
| E and E-dense fail, L passes | the expert branch does not store | memory layer (Berges et al.) or a WISE-style side memory |
| All fail, cloze passes | stored, not extractable | more formats and QA mixing along the dose curve |
| All fail, cloze fails | not stored at 100 exposures | one longer dose; if that fails, the parametric route closes at this budget and the open-book arm prices retrieval |
| S ≈ E on the gold answers, or the answer bag is high | the gain is not the fact | redesign the text |
| QA1 ≈ E | the passages add nothing beyond one Q&A line | good for the thesis: the price per fact is about 15 tokens |

**Predictions** (none against the purpose):

| Prediction | Value |
| --- | --- |
| Δ(E) at 100 exposures | +0.15..+0.45 |
| Δ(E-dense) | ≥ Δ(E) − 0.05 |
| Δ(L) | ≤ Δ(E) + 0.05 |
| Δ(QA1) | 0.05..Δ(E) |
| Δ(0) | within ±0.04 |
| Δ(S) on the gold answers | ≤ 0 |
| E-rot | loses at least half of Δ(E) |
| Fire of E | replay ≤ 0.12; injected prompts ≥ 2× replay |
| Open book, injected half | ≥ 0.70 |

## Literature read in the paper (rule 2), with its known failures

| Paper | What it says | Known failure |
| --- | --- | --- |
| **Physics of LMs 3.1** ([2309.14316](https://arxiv.org/html/2309.14316)) | without augmentation, QA on held-out people is near 0 even with perfect memorisation. Five biographies raise it from 9.7% to 96.6% (Figure 3); "mixed training" reaches 86.6% | synthetic biographies, 540-1,350 passes |
| **Physics of LMs 3.3** ([2404.05405](https://arxiv.org/html/2404.05405)) | the bounds above | synthetic data, GPT-2 sizes |
| **Gekhman et al. 2024** ([2405.05904](https://arxiv.org/html/2405.05904)) | Unknown examples are fit "substantially slower" and then "linearly increase the model's tendency to hallucinate". Early stopping helps | the test asks about other facts, not the trained ones |
| **Ovadia et al. 2023** ([2312.05934](https://arxiv.org/html/2312.05934)) | current events, Mistral 7B: RAG 0.875, fine-tuning 0.504, with 10 paraphrases 0.588 | fine-tuning is unstable; on Llama2 it hurt |
| **Mecklenburg et al. 2024** ([2404.00213](https://arxiv.org/html/2404.00213)) | fact-based SFT (10 Q&A per atomic fact) rises steadily from 1× to 10×; token-based coverage misses about 20% of facts | training and test questions come from the same generator; GPT-4 judge |
| **EntiGraph** (Yang et al., [2409.07431](https://arxiv.org/html/2409.07431)) | 1.3 M source tokens become 455 M synthetic tokens. Closed-book QuALITY rises from 39.49% to 56.22%, log-linear in tokens. Raw continued pretraining does worse than the base | depends on a strong generator (gpt-4-turbo); the generator can invent relations |
| **Memory Layers at Scale** ([2412.09764](https://arxiv.org/html/2412.09764)) | at 1.3 B, NQ 13.68 against 7.76 dense and 8.14 MoE (Table 1) | trained from pretraining, never attached to a finished model; unstable when large |
| **WISE** ([2405.14768](https://arxiv.org/html/2405.14768)) | a routed side memory for edits. At 1,000 edits on LLaMA-2-7B, GRACE generalises to paraphrases at 0.08 and MEMIT at 0.04 | routing below the oracle; editing methods do not generalise to rewordings |
| **Biderman et al. 2024** ([2405.09673](https://arxiv.org/html/2405.09673)) | in continued pretraining, LoRA r = 256 trails full fine-tuning (HumanEval 0.224 against 0.263 at 20 B tokens) and forgets less | code and maths, not facts; LoRA is very sensitive to hyperparameters |
| **ReMoE** ([2412.14711](https://arxiv.org/abs/2412.14711)) | a ReLU gate needs an L1 cost with an adaptive λ to stay sparse | a gate driven to zero gets no gradient |

**Alternatives considered:**
- **Retrieval** (Ovadia et al., Mecklenburg et al.: RAG stays ahead) is not first, because the thesis is parametric
  capacity at fixed bytes per token. The open-book arm prices it.
- **Editing methods** (MEMIT, GRACE) need no teacher text but do not generalise to rewordings at scale (WISE's table).
  They are not chosen.
- **The memory layer** is the strongest prior for facts. It is the first alternative if the expert branch fails.

## Independent review (rule 6)

A fresh sub-agent, with no access to the conversation, read the thesis, the directive, the rules, this design and the
branch code. Its verdict was "proceed after fixes".

| # | Objection | Severity | Response |
| --- | --- | --- | --- |
| 1 | Arm 0 cannot fail. String recall of Q→A is untested, and rewordings may sit near the training strings | blocking | QA1 added; one rewording comes from T0; the 8-gram overlap is a rejection rule, not a report |
| 2 | "Not stored" cannot be told from "not extractable" | blocking | cloze on the passages and the original question are added |
| 3 | "E fails, L passes" cannot be attributed; outcomes are missing | blocking | E-dense added; every outcome has a reading and a next step |
| 4 | λ starts at 1e-8, so about a quarter of training runs dense; a dead gate cannot recover; there is no fire number and no selectivity | blocking | bias set for fire = 0.10 at step 0; λ0 calibrated in the MVE; a STOP on a dead layer; a replay ceiling of 0.15; fire on injected, twin and replay |
| 5 | The scale conclusion is unsupported: an upper bound used as a cost, saturation is not the thesis, batching is unmeasured, int4 regime misread | blocking | restated as "at the measured rates"; batched F0 is measured; QA1 prices the low end; the int4 regime is fixed; the choice of teacher is handed to the user |
| 6 | The pool is counted before filters; one seed; fine-tuning is unstable | should-fix | a minimum of 100 per half and an MDE in the protocol; 2 seeds for E and L; dose checkpoints |
| 7 | S leaks through context and swaps across answer types | should-fix | swaps stay within answer type; the answer bag is reported |
| 8 | Δ can hide a fall on the twin half | should-fix | each half is reported against arm 0 |
| 9 | The guards have no thresholds | should-fix | A4's paired rule, read at every checkpoint, on known items and general panels |
| 10 | Facts per GB is dropped | should-fix | a synthetic capacity probe is not run: about 10⁸ tokens through the base. It is the next cell if E passes (see the table above) |
| 11 | Biderman et al. was read from the abstract only; EntiGraph, Mecklenburg et al., WISE, GRACE and MEMIT are missing | should-fix | all read in the paper and added |
| 12 | The S2 donor-gap panel is used up | note | recorded; the twin half is reserved |
| 13 | 4 experts and top-1 do not test addressing in a large pool | note | said in the scale section |
| 14 | The open book checks the questions, not training | note | QA1 and cloze cover training |
