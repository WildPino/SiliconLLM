# Depth-Reduction / Layer-Pruning Prior Art (Q3 of the 2026-08-29 Researcher brief)

**Scope.** Training-free or near-training-free removal of whole transformer layers
(and of attention sublayers separately) from a pretrained donor. Covers ShortGPT, LaCo,
Shortened LLaMA, SLEB, the angular-distance family (Gromov et al.), LLM-Streamline,
"What Matters in Transformers" (Attention Drop), and "A deeper look at depth pruning".
**Not covered here:** activation sparsity and ReLUfication (companion file
`ACTIVATION_SPARSITY_PRIOR_ART.md`), width pruning and SliceGPT except where a depth
paper uses them as a baseline, and MoE-ification.

**Tagging** follows `ACTIVATION_SPARSITY_PRIOR_ART.md`: **[T]** = read in the paper's
own table, with the table number given; **[A]** = prose or figure only, never a table;
**[X]** = could not verify. Where a claim rests on a **figure** rather than prose I say
"Figure N" explicitly, because the brief's governing rule was broken here before by
numbers that lived in figure captions.

All papers below were read as arXiv HTML renders fetched 2026-08-29 and grepped
verbatim; that is tool-mediated retrieval, not primary-source pixels, which is the
closest approximation available in this session.

---

## 0. The finding that should be read before any of the numbers

**Every headline in this literature is an average over multiple-choice benchmarks, and
under depth pruning multiple-choice accuracy and free-form generation come apart so
violently that the two can move in opposite directions on the same model at the same
pruning ratio.** Our sealed success criterion is general-purpose retention and our
working metric is BPB; a converted model that scores well on MMLU and cannot write a
sentence is worthless to us. So this asymmetry is not a caveat to the results below —
for our purposes it is the result.

Three independent demonstrations, each read in a table:

**(a) ShortGPT: MMLU survives, summarisation is annihilated.** **[T] Table 2**,
Llama2-7B at 27.1% of parameters removed:

| column | Dense | ShortGPT 27.1% |
|---|---|---|
| MMLU | 45.39 | **43.96** |
| XSum | 19.40 | **0.67** |
| Ave. (13 tasks) | 47.78 | 41.24 |

Baichuan2-7B, same table, 24.2% removed: MMLU 53.87 to 45.77, **XSum 20.82 to 0.04**.
The "Per." column reports 86.31% and 85.10% performance retention respectively. **A
model scoring 0.04 on XSum is not retaining 85% of anything**; the average says
otherwise only because twelve multiple-choice columns outvote the one generation column.

The authors say so themselves, section 5 (Limitation) **[A]**: *"the negative effect of
layer removal is more significant on generative tasks compared to multiple-choice tasks.
When we remove 25% layers from Llama2-7B or Baichuan2-7B, the performance in generative
tasks such as XSum and C3 deceases to nearly zero, although the performance decline was
not as significant on the larger model of the 13B."*

**Size dependence, and it points the wrong way for us.** Same **[T] Table 2**, the 13B
rows: Llama2-13B XSum 23.45 to 17.59 at 24.6%; Baichuan2-13B XSum 25.02 to 15.14 at
24.7%. So the collapse is severe at 7B and mild at 13B. **Our donor is 1.5B, below the
smallest model in that table.** Extrapolating the trend downward would predict the
collapse is worse still — **that extrapolation is mine and is not evidence**, but the
direction of the measured size dependence is unambiguous and the risk should be priced.

**(b) The perplexity/MMLU decoupling, in one table.** **[T] Table 4**, ShortGPT applied
to a GPTQ-4bit Llama2-7B:

| Ratio/Layers | Perplexity | MMLU | Throughput |
|---|---|---|---|
| 0% / 32 (baseline) | 8.03 | 43.17 | 4331.23 tok/s (1.00x) |
| 12.5% / 28 | 10.24 | 41.62 | 4680.68 tok/s (1.08x) |
| 25.0% / 24 | 22.29 | 41.68 | 5045.59 tok/s (1.16x) |
| 27.1% / 23 | **40.78** | **43.35** | 5146.99 tok/s (1.19x) |

**Perplexity rises 5.1x, from 8.03 to 40.78, while MMLU ends higher than it started.**
For a project whose quality gate is BPB, MMLU is not merely an incomplete guardrail
here — it is anti-correlated with the thing we measure. Any depth-pruning gate we build
must be a language-modelling-loss gate; a benchmark-accuracy gate would have passed a
model whose perplexity had quintupled.

**(c) LaCo and ShortGPT destroy opposite halves of the model at the same ratio.**
Both papers report the same benchmark suite at the same 27.1% ratio on Llama2-7B, and
the two rows appear together in **[T] ShortGPT Table 2** and again, independently, in
**[T] LaCo's own main results table** (where the dense reference is LaCo's reproduced
"Dense*" row, MMLU 45.92, XSum 19.68):

| method @27.1% | MMLU | XSum |
|---|---|---|
| Dense | 45.39 (LaCo repro: 45.92) | 19.40 (LaCo repro: 19.68) |
| **ShortGPT** | **43.96** (holds) | **0.67** (destroyed) |
| **LaCo** | **26.45** (destroyed, MMLU chance is ~25) | **15.64** (holds) |

**Two leading methods, same ratio, same suite, same models — and each one keeps
precisely what the other loses.** Neither paper's average discloses this: ShortGPT's
average is 41.24, LaCo's is 38.41, and the 2.8-point gap between them conceals the fact
that they are not degraded versions of the same model but two different mutilations.
I cross-checked these cells in both papers independently and they agree.

**(d) A published, formal statement of why the accuracy metric hides this.**
LLM-Streamline devotes section 3.1 to it, and the mechanism is confusion-matrix
cancellation. **[T] Table 1(a)**, counts of samples by before/after correctness:

| Dataset | TP | FN | FP | TN |
|---|---|---|---|---|
| C3 | 543 | 257 | 210 | 815 |
| CHID | 269 | **563** | **177** | 993 |
| Race-M | 380 | 95 | 129 | 832 |
| Race-H | 938 | 305 | 353 | 1902 |

On CHID the pruned model gets **563 previously-correct items wrong and 177
previously-wrong items right**; the two partly cancel and accuracy barely moves.
**[T] Table 1(b)** shows the per-sample PPL standard deviation is markedly lower on the
FN and FP sets than on TP and TN, and the authors' reading **[A]** is that *"the model
is more uncertain about the FN and FP samples, implying that the model may guess the
correct answer for a significant portion of these samples after pruning. This phenomenon
suggests that the accuracy metric may overestimate the performance of the compressed
model."* They propose a **"stability"** metric (their Eq. 6) that scores agreement with
the *original* model weighted by confidence, rather than agreement with the gold label.

> **If we run a depth-pruning probe, this is the metric shape to copy: agreement with
> the unpruned donor, not accuracy against a key.** It is also, conveniently, cheap and
> label-free. That is a recommendation about instrument design, not a project fact.

**(e) Which papers cannot show this hole because they never look.** Per the
coordinator's instruction I checked generation-benchmark coverage across the family by
grepping each paper's full text:

| paper | XSum | GSM8K | HumanEval | perplexity / LM loss | MMLU |
|---|---|---|---|---|---|
| ShortGPT (2403.03853) | yes | no | no | yes | yes |
| LaCo (2402.11187) | yes | no | no | yes | yes |
| Gromov et al. (2403.17887) | **no** | no | no | yes (C4 loss) | yes |
| SLEB (2402.09025) | **no** | no | no | yes (heavy) | **no** |
| Shortened LLaMA (2402.02834) | **no** | no | no | yes | **no** |
| LLM-Streamline (2403.19135) | yes | yes | no | yes | yes |
| What Matters / Attention Drop (2406.15786) | **no** | yes | no | **no** | yes |
| A deeper look (2407.16286) | **no** | no | no | yes | yes |

**Five of eight papers have no free-form generation benchmark at all.** Two of them
(SLEB, Shortened LLaMA) report no MMLU either and rest almost entirely on perplexity
plus zero-shot multiple choice. "What Matters" reports **no perplexity or LM loss at
all** — its entire case is multiple-choice accuracy plus GSM8K. **A method whose paper
contains neither XSum-like generation nor a language-modelling loss cannot be assumed
free of the ShortGPT hole; it simply did not look.** I am not asserting those methods
have the hole. I am recording that their evidence base cannot exclude it.

---

## 1. The teacher question, answered per method — because it disqualifies outright

Distillation is forbidden: no teacher logits, no teacher hidden states. Depth pruning
has a subtlety the activation-sparsity line does not, and it caught one method:
**the "teacher" here can be the donor itself before pruning.** Self-distillation from
the unpruned donor is still hidden-state matching.

| method | recovery step | trained against what | verdict under our rule |
|---|---|---|---|
| **ShortGPT** (no post-train) | none — calibration only | n/a | **clean** |
| **Gromov et al.** | QLoRA healing | plain next-token LM loss on C4 **[A]**, App. A.1 | **clean** |
| **Shortened LLaMA** | LoRA, or full CPT | LM loss on pretraining corpus **[A]**, App. E.2 | **clean** |
| **SLEB** | none — retraining-free by design | n/a | **clean** |
| **LaCo** | none — merging plus calibration | n/a | **clean** |
| **A deeper look**: emulated update | additive bias = empirical mean of the removed block's update over a small calibration set **[A]**, section 3.2.1 | no gradient step at all; a calibration statistic | **clean in my reading**, but it *is* derived from the donor's own activations — flagging for the Adapter's judgement, not deciding it |
| **A deeper look**: low-rank adapter | trained adapter in place of the missing block | layer-stitching objective; loss not stated in what I read — **[X]** | **cannot clear it; unresolved** |
| **LLM-Streamline** | lightweight network replacing the pruned span | **MSE against the original model's hidden states**, their Eq. 4: `minimize_h E MSE(h(x^(l*)), x^(l*+n))` **[A]**, section 2.3 | **DISQUALIFIED as written** — this is hidden-state matching against the unpruned donor |
| **ShortGPT + post-train** | removed layers replaced by gated MLPs, then retrained, 50B tokens | loss type **[X] not stated** in App. F / Table 12, which give only LR, batch, tokens | **unresolved, and unaffordable anyway** |

Grep counts supporting the "clean" rows: `distill`/`teacher` appear 0 times in SLEB;
in ShortGPT 0 times; in Gromov 23 times but confined to the Literature Review section
2.2, which explicitly *contrasts* distillation with pruning (*"A completely different
method for reducing the size of a trained machine-learning model is model
distillation"*); in LLM-Streamline the 7 hits are likewise related-work framing — **its
disqualification comes from Eq. 4, not from the word "distillation", which is exactly
why a keyword scan alone would have passed it.**

---

## 2. The cost ladder — priced from each paper's own reported budget

This is the number the brief said decides whether the path is open. **It is a
completely different regime from the ReLUfication line**, which the companion file
priced at 30B-150B tokens and thousands of A100-hours.

| step | method | cost, as the paper states it | tag |
|---|---|---|---|
| **Choosing which layers to cut** | Shortened LLaMA | **10 calibration samples** from BookCorpus at sequence length 128; *"pruning is completed efficiently within 1 to 2 hours for the 7B- and 13B-sized models"* on one A100 | **[A]** App. E.2 |
| | ShortGPT | forward passes over PG19 to compute Block Influence; no cost stated — **[X]** | [X] |
| | Gromov et al. | angular distances averaged over **10k C4 samples** | **[A]** section 4.3 |
| | LaCo | **5 sentences** (Baichuan2) / **10 sentences** (Llama2) as few-shot calibration; 8xA100 server | **[A]** section 3.3 |
| | LLM-Streamline | cosine similarity over recorded hidden states; volume not stated — **[X]** | [X] |
| **Free recovery** | A deeper look: emulated update | empirical mean of the block update on a small calibration set; **no training** | **[A]** 3.2.1 |
| **Cheap recovery** | Shortened LLaMA (LoRA) | *"retraining a 20%-pruned model from 7B parameters takes about **2 hours** and utilizes 22GB GPU memory, while a 21%-pruned model from 13B parameters requires approximately **3 hours** and 35GB VRAM"*, single GPU, rank 8, 2 epochs | **[A]** App. E.2 |
| | Gromov et al. (QLoRA) | **5000 steps x global batch 16 x seq 2048 = 163.8M tokens** for models <=7B (x4096 = 327.7M for >=13B); *"each of our experiments can be performed on a single A100 GPU"* | **[A]** App. A.1; the token count is **my arithmetic on their stated formula**, which they write out as `16 x 5000 x [max_seq_length]` |
| | LLM-Streamline | ~**5 hours on a single A800** for the lightweight net, plus <1 hour post-training; their LoRA baseline ~10 hours | **[T]** App. B.1/B.2 — *but the method is disqualified per section 1* |
| **Expensive recovery** | ShortGPT post-train | **50B tokens**, global batch 2048, seq 4096, bf16 | **[T]** Table 12 |
| | Shortened LLaMA CPT | *"eight NVIDIA H100 (80GB) GPUs are utilized, with each model size trained in under two weeks"* — approx. 2,700 H100-hours by my arithmetic | **[A]** App. E.2 |

**Reading of the ladder against our budget (30 GPU-h/week per account across 3 Kaggle
accounts, T4/P100 class).** The selection step is free at any scale we care about — ten
samples and an hour. The cheap recovery tier — a 2-3 GPU-hour LoRA, or Gromov's 164M
tokens on one GPU, or the emulated update at zero — **sits inside a single week of one
account's free tier**, with a T4-vs-A100 penalty of maybe 3-5x still leaving room.
**This is the first path in this programme's prior-art surveys that fits the sealed
budget without an extrapolation.** The expensive tier (50B tokens, or CPT) does not fit
and is not needed below roughly 50% removal — see section 5.

**The caveat that keeps this honest:** none of these budgets was measured on Turing or
Pascal, none on a 1.5B donor, and the T4 conversion is mine. What the papers establish
is that the cheap tier exists and is single-GPU, single-digit-hours at 7B-13B. That it
is *cheaper still* at 1.5B is an inference, not a measurement.

---

## 3. Which layers get removed — the D0 U-shape question

Our D0 probe measured a **U-shaped depth profile on the donor: structure at both ends,
absent in the middle.** The literature is close to unanimous, and I read every one of
these statements in prose or a figure, never a table — **there is no table anywhere in
this family that prints which layer indices were removed.** That is itself a finding.

| paper | statement | tag |
|---|---|---|
| **ShortGPT** | *"this redundancy is primarily manifested in the **middle to later layers** of the network, with the **initial layers and the last layer** often being more critical. Notably, we found the last layer to be particularly important"* — and they note this *"contradicts our mathematical explanation in Appendix A which suggests that deeper layers tend to be more redundant"*, attributing the last-layer importance to the final FFN acting as part of the token classifier | **[A]** section 2.2 |
| **Gromov et al.** | *"(i) the smallest distances are found across the **deeper blocks**, meaning deeper layers are typically quite similar to each other and can be more easily dropped; (ii) the distances across the deepest blocks -- the blocks that include the last layer -- take either maximal or nearly-maximal values, meaning **one should never drop the final layer**."* Exceptions noted for Phi-2 and for the largest blocks in Llama-2-7B | **[A]** section 4.3 + Figure 4 |
| **What Matters** | *"initially, both models tend to drop the **deeper layers, followed by the shallower ones**. These findings are consistent with ... Men et al. (2024), which suggests that deeper layers tend to be more redundant"* | **[A]** section 4.2 + Figure 3 |
| **Shortened LLaMA** | records as standard practice for the LLM-Pruner baseline that *"the **first and last few blocks remain unpruned**"* | **[A]** App. E.1 |
| **LLM-Streamline** | least-important layers are *"often contiguous"*, selected by `argmax_l cos(x^(l), x^(l+n))`; positions shown only in their Figure 2 — I could not read indices off it | **[X]** |

**Verdict, stated without smoothing.** The literature converges on a profile that is
**U-shaped in importance but asymmetric**: the shallow layers are critical, the final
layer is critical, and the trough of redundancy sits **between them but skewed late** —
"middle to later" in ShortGPT's words, "deeper blocks" in Gromov's and What Matters'.
**This agrees with D0 that both ends carry structure and the interior does not. It
disagrees with any reading of D0 that places the trough symmetrically in the middle:
three independent papers put the most-removable region later than centre.**

**And there is one family-specific exception, which is ours.** Gromov et al., same
section 4.3 **[A]**: *"the Qwen family is somewhat unusual: here we see that there are a
few odd 'islands' of high similarity for **shallow blocks**; this likely explains the
shorter region of robust performance in Figure 2."* **Qwen is the one family in their
study where high-similarity blocks appear at the shallow end**, i.e. where the
literature's standard "cut deep" heuristic is least reliable. Our donor is Qwen2.5-1.5B.
I read this in prose referring to their Figure 4; the underlying heat map is a figure
and I did not extract numbers from it.

---

## 4. Does it compose with weight quantisation — and does anything test ternary?

**Short answer: it composes cleanly with 4-bit, and no source I found tests ternary.**

- **ShortGPT [T] Table 4** applies layer removal on top of a GPTQ-4bit Llama2-7B — the
  full table is reproduced in section 0(b) above. It works, at the perplexity cost
  documented there.
- **ShortGPT [T] Table 5**, order of operations, Llama2-7B:

| Method | MMLU | CMMLU |
|---|---|---|
| Baseline | 45.4 | 32.9 |
| 4-bit quantization | 44.9 | 32.5 |
| Layer removal (27.1%) | 44.0 | 32.3 |
| 4-bit quantization **then** layer removal | 42.4 | 31.0 |
| Layer removal **then** 4-bit quantization | 41.2 | 30.5 |

  The losses are roughly additive, and **quantise-then-prune beat prune-then-quantise by
  1.2 MMLU points** — a small, single-seed, single-model difference that I would not
  build on, but it is the only ordering evidence I found and it points the opposite way
  from our current pipeline intuition.
- **What Matters**: *"we use the mainstream AWQ algorithm for 4-bit quantization ...
  As shown in Table 5, the integration of quantization still maintains the performance
  of Attention Drop, i.e., only less than 1% difference in average performance."*
  **[A]** — I read the claim in the prose of the appendix; I did not transcribe Table 5's
  cells, so this is [A], not [T].
- **SLEB**: *"When further compressing LLMs that have been pruned with SLEB using AWQ,
  we observe a negligible impact on perplexity results for both C4 and WikiText-2"*
  **[A]**, App. B.6, evidenced by their Figures 11 and 12 — **figures, not a table.**
- **[X] Ternary or 2-bit composition: not found in any of the eight papers.** Every
  composition result above is 4-bit (GPTQ or AWQ). Whether depth pruning composes with a
  ternary donor is **unmeasured in this literature**, and given Phase-61's standing
  finding that precision-hungry organs behave differently under ternarisation, it should
  not be assumed from the 4-bit results.

---

## 5. Counter-evidence, sought deliberately

**5.1 The speedups are small, and they are GPU speedups.** ShortGPT **[T] Table 4**:
removing 27.1% of layers buys **1.19x** throughput. "What Matters" **[T] Table 1**,
Llama-2-13B: dropping 8 of 40 attention layers gives **1.11x**; dropping 20 gives
**1.30x**; Block-8 gives 1.23x. SLEB **[T]**, 20% sparsity: **1.24x-1.30x** across A100
and A6000. **Depth pruning is not a 1-to-1 conversion of removed parameters into
speed on the hardware these papers used.** Whether a memory-bound CPU engine converts it
better is precisely the open question — see 5.5, which argues it might.

**5.2 The best-known result is method-dependent and metric-dependent.** "A deeper look
at depth pruning" (Siddiqui, Dong, Heinrich, Breuel, Kautz, Krueger, Molchanov —
Cambridge and NVIDIA, **workshop paper, TF2M at ICML 2024**, which is a lighter venue
than the others here and I flag it as such) states in its abstract **[A]**: *"We show
that adaptive metrics exhibit a trade-off in performance between tasks i.e.,
improvement on one task may degrade performance on the other due to differences in the
computed block influences."* That is the ShortGPT-versus-LaCo inversion of section 0(c),
generalised: **the choice of importance metric selects which capability you keep.**

**5.3 Cheap healing stops working at high ratios.** Shortened LLaMA **[A]**, section
3.3: *"we show that LoRA can also recover the ability of depth-pruned models; however,
**it does not perform well for extensive compression rates (e.g., with over 50%
removal)** in either width or depth pruning"*, and CPT — the ~2,700-H100-hour tier —
*"is critical for severely depth-pruned models."* **The affordable tier of section 2 is
affordable only in the moderate regime.**

**5.4 Papers disagree about each other's results.** Shortened LLaMA **[A]**, App. E.1,
on SLEB: *"Although SLEB pursues a retraining-free setup, **we observed that it fails to
sustain adequate performance as the pruning ratio increases**."* Independent third-party
reproduction contradicting an author's own claim is the strongest counter-evidence
available in a literature, and it exists here.

**5.5 The one strong argument in the other direction, and it is about our regime.**
Shortened LLaMA's framing **[A]**, section 1 and Figure 2: LLM decoding *"often exhibits
a memory-bound nature"*, and *"we aim to improve inference speeds of LLMs, especially
under hardware limitations that demand small batch sizes, **where we observe that
width-only pruning is inadequate**."* Their Figure 1 (left) reports depth pruning giving
faster inference than the width-pruning baselines FLAP and LLM-Pruner on an H100, and
Figure 2 (bottom) that *"our depth pruning (blue lines) improves generation speeds over
the original models (gray), while width pruning is ineffective (green)."* **Both are
figures, not tables — I did not read a table of these latencies.** The mechanism is
nonetheless the one we care about: depth removal deletes sequential dependencies and
whole weight-streaming passes, not merely FLOPs, and that is the axis a batch-1
memory-bound CPU engine is on.

**5.6 Attention layers specifically are the most removable sublayer — which bears on a
sealed constraint.** Two independent sources agree:
- "What Matters" abstract **[A]**: *"a large proportion of attention layers exhibit
  excessively high similarity and can be safely pruned without degrading performance ...
  Llama-3-70B maintains comparable performance even after pruning half of the attention
  layers."* Their **[T] Table 1** (Llama-2-13B) supports the moderate end: Attn-8 holds
  the average at 68.1 versus baseline 68.2 at 1.11x, while Block-8 — dropping whole
  blocks instead — falls to 60.7.
- "A deeper look" abstract **[A]**: *"highlighting the propensity of the self-attention
  layers to be more amenable to pruning, even allowing **removal of up to 33% of the
  self-attention layers without incurring any performance degradation on MMLU for
  Mistral 7b** (significant reduction in costly maintenance of KV-cache)."*

This is directly supportive of the sealed constraint that attention should run on a
minority of layers, and it independently names the KV-cache saving as the payoff.
**Two warnings attached.** First, the "no degradation" in the second quote is measured
**on MMLU**, which section 0(b) shows can rise while perplexity quintuples — the claim
is therefore much weaker than it reads. Second, "What Matters" reports **no perplexity
or LM loss anywhere in the paper**, so its Attention Drop results cannot be checked
against a language-modelling metric at all. And its own MMLU column is non-monotonic
(**[T] Table 1**: Attn-16 gives 48.2, Attn-20 gives 51.5, against baseline 55.1), which
is a sign of single-run noise that no error bar is offered for.

**5.7 The pattern that recurs across three papers and deserves naming: MMLU is
anomalously *insensitive* to depth pruning.** ShortGPT Table 4 (perplexity 8.03 to
40.78, MMLU 43.17 to 43.35); "What Matters" **[T] Table 2**, Llama-2-70B Block-16
(MMLU **rises** 68.5 to 69.2 while ARC-C falls 67.4 to 56.3); "What Matters" **[T]
Table 1**, Mistral-7B Block-8 (MMLU 62.5 to 60.0 while ARC-C falls 61.5 to 40.0 and
HellaSwag 83.2 to 63.9). **The standing rule that MMLU must never be reported inside an
average was written because MMLU-sized holes get hidden by macro-averaging. In this
literature the failure runs the other way: MMLU is the column that fails to move while
the model is being destroyed.** Both directions argue for the same discipline — report
per-task, never an average — but the Adapter should know that on this axis MMLU is a
weak guardrail, not a sensitive one.

---

## 6. What I could not verify

- **[X] No paper in this family tests a donor below roughly 2.7B** (Phi-2 in Gromov et
  al. is the smallest; OPT-1.3B/2.7B appear only in LLM-Streamline's Appendix C.1).
  **There is no depth-pruning datapoint at 1.5B, and none at all on Qwen2.5.** Gromov's
  Qwen rows are Qwen-7B and Qwen-14B, the first-generation Qwen, not Qwen2.5.
- **[X] No layer-index table exists anywhere in the family.** Every "which layers"
  statement I found is prose or a heat-map figure.
- **[X] ShortGPT's 50B-token post-training loss function** is not stated in App. F or
  Table 12; I cannot say whether it is plain LM loss or hidden-state matching, so I
  cannot clear it under the no-teacher rule.
- **[X] The training objective of "A deeper look"'s low-rank linear adapter** — cited to
  layer-stitching (Bansal et al. 2021) — was not stated in the text I read.
- **[X] Ternary or 2-bit composition with depth pruning**: not measured anywhere I
  looked.
- **Gromov's transition fractions** (Llama-2 family 45-55%, Mistral-7B 35%, Phi-2 25%,
  **Qwen family 20%**) are section 4.1 prose describing **Figure 2**; there is no table
  behind them. **I have since read Figure 2 and Figure 4 directly from the PDF — see the
  Addendum below, which also corrects what Figure 2 actually plots.** They remain
  figure-derived and are still not [T].
- I did not search the 2025-2026 literature systematically for this question; the
  newest paper here is from 2025. **The standing law applies: a gap is confirmed only
  for the literature actually searched, and this search was the eight papers named at
  the top.**

---
---

# Addendum — 2026-08-29 — Gromov's figures read directly, and the stability metric specified

Three items the Adapter asked for after the first pass. The figures below were read from
the **arXiv PDF of 2403.17887v1** (fetched 2026-08-29, 27 pages), rendered locally at
400 dpi. **Nothing here is upgraded to [T]: a figure read carefully is still a figure.**
I mark direct visual readings **[F]** to distinguish them from the paper's own prose
**[A]**, since the distinction is the whole point of the exercise.

Note on availability: **Figures 2 and 3 are absent from the arXiv HTML render of this
paper** — only Figure 1 and Figure 4 have `<img>` tags. Anyone re-checking this via the
HTML will not find Figure 2 and should go to the PDF.

## A.1 A correction to the first pass: Figure 2 is MMLU, and MMLU is nearly all they measure

**I described Figure 2 as showing "QA benchmarks". That was too generous. It plots
5-shot MMLU accuracy and nothing else.**

**[A]** Figure 2 caption, verbatim: *"MMLU accuracy (5-shot) vs. fraction of layers
dropped for different model families. (Left: Llama-2 family; Middle: Qwen family; Right:
Mistral-7B and Phi-2.) The solid lines represent performance after dropping layers and
healing, dotted lines show performance after dropping layers only (no healing), and the
dashed gray line is the score for guessing randomly. For these models, healing leads to
modest improvements, and performances are quite robust until 20%-55% pruning fractions,
depending on model family and size, at which point they transitions to random guessing."*

**[A]** Section 4, page 8, verbatim: *"For our QA evals, we used Massive Multitask
Language Understanding (MMLU) ... and BoolQ, a common yes/no reading comprehension
benchmark."* Footnote 7 adds that BoolQ (Appendix Figure 6) *"observe[s] analogous
behavior."*

**So the entire question-answering case in this paper rests on MMLU plus BoolQ.**
Read that against section 5.7 of this file, where MMLU is shown across three papers to
be the column that stays flat while the model is destroyed — including ShortGPT's
Table 4, where perplexity goes 8.03 to 40.78 with MMLU unchanged. **Gromov et al.'s
headline "minimal degradation ... until up to half the layers are removed" is a claim
about the least sensitive benchmark in common use.** Their own section 4.2 says as much
in the other direction, and I quoted it in the first pass: post-healing there is
*"continuity through the pruning fractions where we previously found sharp transitions
for the QA benchmarks"*, which they call *"one way of disconnecting (or creating a
miscalibration) between performance on downstream tasks -- such as MMLU and BoolQ -- and
continuous measures of performance -- such as the cross-entropy loss."*

## A.2 What Figure 2's Qwen panel actually shows [F]

Rendered at 400 dpi and cropped to the middle panel. Four curves: Qwen-7B and Qwen-14B,
each with a solid (healed) and dotted (no healing) variant, against a dashed grey
"Random" line at MMLU 0.25.

**What the figure resolves:**

1. **The collapse is real, near-vertical, and early.** Both models hold a rough plateau
   from x = 0.0 to about x = 0.20-0.22, then fall to the Random line across one or two
   x-samples, and stay there for every larger fraction out to x = 0.8. **I read the
   transition at roughly x = 0.22-0.25; the prose says 20%.** The prose is if anything
   slightly conservative relative to what I can see, and the two agree to within my
   reading precision, which is about +/-0.03 on this axis.
2. **The contrast with Llama-2 is visible in the same figure at the same scale.** The
   left panel's curves stay elevated until roughly x = 0.45-0.5 before their own cliff.
   **Qwen's usable range is about half of Llama-2's**, which is the substantive point
   and it does not depend on reading exact values.
3. **Healing raises the level but does not move the cliff.** For Qwen-7B the dotted
   (unhealed) curve sits visibly below the solid one across most of the plateau; for
   Qwen-14B the two are tangled. Both collapse at essentially the same x. This matches
   the caption's *"healing leads to modest improvements"* and means **QLoRA healing does
   not buy extra prunable depth on this family** — it buys a little accuracy inside the
   range that already worked.

**What the figure shows that the prose does not mention, and it matters:**

4. **The Qwen "plateau" is not flat — it is strongly non-monotonic.** Qwen-14B (solid)
   descends from its x = 0 value, dips near x = 0.10-0.12, then **rises to a local
   maximum around x = 0.17-0.18 that appears higher than its own unpruned score**,
   before the cliff. Qwen-7B (solid) shows the same shape with a dip near x = 0.15 and a
   partial recovery at x = 0.20. **The excursions inside the "robust" region are of a
   size comparable to the degradation being claimed as negligible**, and there are only
   about five or six x-samples before the cliff. There are no error bars and no seeds
   anywhere in the figure. **"A characteristic flat region of robust performance" is a
   generous description of the Qwen panel.** I would not build a pruning budget on the
   plateau's height; I would build it only on the cliff's position.

**What the figure does not resolve.** Exact transition fractions beyond about +/-0.03;
absolute MMLU values (Qwen-14B appears to start near 0.60, Qwen-7B near 0.53, but I did
not read gridline values and do not report these as measurements); anything at all about
perplexity, generation, or Qwen2.5.

**The provenance caveat that limits all of this for us: the Qwen models in this paper are
Qwen-7B and Qwen-14B, the first-generation Qwen [A], Appendix A.1, repository paths
`Qwen/Qwen-7B` and `Qwen/Qwen-14B`. There is no Qwen2.5 anywhere in the paper, and no
model below 2.7B.** Our donor is Qwen2.5-1.5B — a different generation and smaller than
their smallest. **The 20% figure is the best external signal we have for our donor's
family, and it is neither our generation nor our scale.**

## A.3 What Figure 4's Qwen panels actually show [F]

Figure 4 (`ang_dist_plots.png`, 884x516, fetched from the HTML render) is a 7-panel grid
of heat maps: x-axis = starting layer index `l`, y-axis = block size `n`, colour =
angular distance between layer `l` and layer `l+n`, yellow low, purple high. Panels:
(a) Llama-2-7B, (b) Llama-2-13B, (c) Llama-2-70B, (d) Qwen-7B, (e) Qwen-14B,
(f) Mistral-7B, (g) Phi-2-2.7B.

**The critical scaling caveat, from the caption [A]:** *"the distance for each n is
shifted and rescaled to span the same range, [0,1]"*. **Colours are therefore comparable
only within a row — a single block size — and never across rows or across panels.** No
statement about Qwen being "more or less similar overall" than Llama can be extracted
from this figure, and I make none.

**What I can see:**

1. **Llama-2-7B/13B/70B, Mistral-7B and Phi-2 all show one clean diagonal yellow band**
   sitting at large `l` and marching leftward as `n` grows, with solid purple across the
   small-`l` region. That is the canonical "the removable block is deep, and never
   shallow" picture, and it is unambiguous in five of the seven panels.
2. **Both Qwen panels break that pattern, and they break it in the way the prose says.**
   Qwen-7B (d) carries a distinct light green-to-yellow region at **small `l`, roughly
   `l` = 1 to 8, at small and mid block sizes**, separate from the main deep band.
   Qwen-14B (e) shows a pronounced light column at **`l` ~ 0-5** extending upward in `n`,
   plus a second light band at mid-depth. Both panels look **mottled and multi-modal**
   where the Llama panels look like a single clean gradient. **This corroborates the
   prose's "odd islands of high similarity for shallow blocks" — I can see the islands.**
3. **The final-layer rule is visible too**: in every panel the cells at the extreme
   upper-right, the blocks that include the last layer, are dark. Consistent with the
   prose's *"one should never drop the final layer."*

**What it does not resolve:** exact layer indices beyond a few layers of precision; any
cross-model magnitude comparison (forbidden by the per-row rescaling); and again, nothing
about Qwen2.5 or about 1.5B.

**Bearing on the D0 U-shape.** Section 3 of this file concluded that the literature finds
a late-skewed trough with both ends protected. **Figure 4 confirms that for five model
families and shows Qwen as the exception in the direction that matters to us**: on Qwen,
low-distance blocks appear at the shallow end, exactly where the other families are most
protected. If D0's U-shape places structure at both ends of our Qwen2.5 donor, then D0
and Gromov's Qwen panels disagree about the shallow end specifically. **That is a genuine
disagreement on our own model family and I am not going to smooth it: five families say
shallow layers are untouchable, and the one family that is ours says its shallow blocks
contain high-similarity islands.** Which is right for Qwen2.5-1.5B is unmeasured by
anyone, including us.

## A.4 One more thing Figure 2's surrounding text settles: healing is optional for them, mandatory for us

**[A]**, section 3.2, page 7, verbatim:

> *"Elaborating on the 'optionality' of the final step, we find that the near-lack of
> performance degradation on question-answering benchmarks, cf. Figure 1(d) and others in
> §4.1, can be extended to greater pruning fractions with a small amount of finetuning.
> Depending on resource constraints and intended application of the pruned model, **this
> may not be necessary**. However, **the healing procedure does have a substantial impact
> on perplexity**, cf. Figure 1(d) and others in §4.2."*

**The authors say the healing step can be skipped if you care about QA accuracy, and
cannot be skipped if you care about perplexity. Our metric is BPB. For us the healing
step is mandatory, not optional** — and its price is therefore not avoidable.

**That price is now confirmed in the main text rather than derived by me.** Section 4,
page 8, verbatim: *"we executed the 'healing' step using QLoRA: our models were quantized
to 4-bit precision and then finetuned, using QLoRA for efficient training, on either
**164M or 328M tokens** from the Colossal Clean Crawled Corpus (C4), a common pretraining
dataset. As a result, **each experiment of ours was performed on a single A100 GPU**."*
My first-pass arithmetic from Appendix A.1's `16 x 5000 x max_seq_length` gave 163.8M and
327.7M; **the main text states 164M and 328M, so the cost-ladder row in section 2 stands
and is no longer my derivation.** It is still prose, not a table: **[A]**.

Also from the same page, the deepest-layers variant stated exactly, section 3.2 **[A]**:
*"drop the deepest layers, excluding the final layer before the LLM head, and then
(non-optionally) heal the damage ... if we are pruning n layers from an L-layer model,
then we would remove layers (L-n) to (L-1), inclusive."*

## A.5 The stability metric, specified well enough to build

From LLM-Streamline (arXiv:2403.19135v3), section 3, read in the paper's own equations.
Purpose: catch the answer-flip cancellation documented in section 0(d) of this file,
where CHID lost 563 previously-correct items and gained 177 previously-wrong ones with
accuracy barely moving.

**Step 1 — per-sample confidence weight, from the ORIGINAL model only.** For a
`k`-choice task, sample `i` with question `x_i` and choices `c_{i,1..k}`, **Eq. 5**
verbatim:

```
PPL_{i,j} = PPL( M(x_i, c_{i,j}) )
PPL_i     = (1/k) * sum_j PPL_{i,j}
std_i     = sqrt( sum_j (PPL_{i,j} - PPL_i)^2 / (k-1) )
```

with the paper's own gloss **[A]**: *"where PPL_{i,j} denotes the PPL for the sentence
created by question x_i and choice c_{i,j} of **the model before pruning**"* and
*"A higher std_i value indicates the LLM exhibits greater confidence in answering the
question x_i."*

> **Two implementation points worth having explicit, because both are easy to get wrong.
> `std_i` is computed on the UNPRUNED model, so it is a fixed per-sample weight that can
> be precomputed once from the donor and reused across every candidate pruning. And it is
> the sample standard deviation of the per-choice perplexities — the spread across
> choices, not across tokens.** Low spread means the choices look alike to the model,
> i.e. it is guessing.

**Step 2 — the confusion sets**, defined against the gold label `y_i`, with `M` the
original and `M-bar` the pruned model:

| set | condition |
|---|---|
| TP | correct before **and** correct after |
| FN | correct before, wrong after |
| FP | wrong before, correct after |
| TN | wrong before **and** wrong after |

**Step 3 — the metric, Eq. 6** verbatim:

```
Stability(M, M-bar) = sum_i ( exp(std_i) * 1[ i in TP union TN ] ) / sum_j exp(std_j)
```

**Read what that is: a confidence-weighted fraction of samples on which the pruned model
reaches the same correctness verdict as the donor.** It rewards agreement with the
donor, not agreement with the answer key — so an item the donor got wrong and the pruned
model also gets wrong counts as success. The `exp` is there for a stated reason **[A]**:
*"Because the std of different samples varies significantly. to mitigate the influence of
samples with excessively large standard deviations, we apply the exp function to moderate
the weight differences among samples."*

**Two cautions before we build it.** The `exp` of an unnormalised perplexity standard
deviation is scale-sensitive — their Table 1(b) reports std values on the order of
`10^-3`, at which `exp(std) ~ 1 + std` and the weighting is nearly uniform, so **the
weighting may be doing much less work than the formula suggests and the metric may
reduce in practice to plain TP+TN agreement.** I did not find an ablation isolating the
`exp` term — **[X]**. And the metric is defined only for multiple-choice tasks; **it does
not extend to free-form generation**, which section 0 of this file identifies as the
capability most at risk. **A stability score is a better instrument than accuracy, but it
is not a substitute for measuring BPB and for looking at generated text.**
