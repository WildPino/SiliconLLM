# Attention Linearisation Prior Art — R1

**Author:** the Researcher · **Date:** 2026-08-22 · **Status:** IN PROGRESS, written incrementally per method
**Brief:** `docs/research/BRIEF_R1_CLOSED_FORM_ATTENTION_CONVERSION.md`

Rules in force: every number tagged **[T]** (read in the paper's own table, coordinates given),
**[A]** (text/abstract only, no table located), or **[X]** (searched, not found — a correct answer).
No number is used in a computation until it is [T] or explicitly flagged as [A]-derived. This file is
written method-by-method as each is verified, not held to the end.

---

## 0. Framing check (brief §1) — do these methods actually give O(1) per-token state?

Working note, to be confirmed/revised as each method is checked: the brief's claim is that *linear*
attention (kernelised, `softmax(QKᵀ)V → φ(Q)(φ(K)ᵀV)` reassociated) keeps a state of size `d_k × d_v`
(times number of heads), constant in `T`, versus softmax attention's KV cache of size `O(T · d_k)` /
`O(T · d_v)`. This is checked per-method below, not assumed. Flagged **not fixed-state** where a method
speeds up or approximates softmax attention but still materializes/stores per-token K/V (e.g. any method
that keeps a sliding or full KV cache alongside a linear component).

---

## 1. Hedgehog (Zhang, Bhatia, Kumbong, Ré — arXiv:2402.04347, Stanford, ICLR 2024)

*(Note on ordering: written second, numbered first — Hedgehog is the mechanistic ancestor LoLCATs builds
on, so it reads better first. Sections are appended to this file in the order each was verified, not
necessarily narrative order; numbers reflect final reading order, not write order.)*

**Verified against:** `https://arxiv.org/html/2402.04347v1` (raw HTML; v2/v3/v4 do not have an HTML
rendering available — arxiv served an empty article page for those, v1 is the version actually readable
and is what is transcribed below; flagging in case the published/camera-ready version changed numbers).

### Mechanism [T, §4.2, Table 2]

Standard kernel-based linear attention (`φ(q)ᵀφ(k)` replacing `exp(qᵀk)`, computed via the ordinary causal
linear-attention formulation, "equation 2" in the paper — the same cumulative-sum/recurrent form as
LoLCATs' Eq. 5, **no sliding window or local component added**). The feature map is a **trainable one-layer
MLP** `φ_mlp(x) = Φ(Wᵀx + b)` with the activation `Φ` set to element-wise exponential to induce
"spikiness," inserted after the Q/K projections (an "adapter," in the paper's own word) — trained by
gradient descent so the linear-attention weights mimic softmax attention's two empirically-identified
missing properties: **spikiness** (low entropy) and **monotonicity** w.r.t. the query-key dot product. This
is the direct ancestor of LoLCATs' `φ_q, φ_k` (LoLCATs' Table 2 explicitly lists "Hedgehog" as one of its
two feature-map options, with the same functional form modulo details).

Training is **gradient descent on attention-transfer loss** (KL divergence to softmax attention weights, or
downstream task loss, depending on setting) — **explicitly not closed-form**, confirming the brief's
premise for §4(b).

**Donor Q/K/V/O reuse:** for the Llama-2 7B conversion (`§B.5`), the paper states explicitly: **"we freeze
all original Llama-2 weights, and train Hedgehog MLPs for every head and layer (0.495% of the original
model size)"** `[T, Appendix B.5]` for the distillation stage — same pattern as LoLCATs (freeze donor
projections, train small adapter). The subsequent finetuning stage then applies **LoRA (rank 8, α=16) to
Q/K/V/O** `[T, Appendix B.5]` — again, reused/perturbed, not replaced. For the GPT-2 125M conversion,
however, the paper states **"we finetune all model parameters"** for the post-distillation stage `[T,
Appendix B.5, "Linear GPT-2 125M conversion" paragraph]` — i.e. **full fine-tuning, not LoRA**, at the
125M scale. This is a real difference from LoLCATs: Hedgehog's own published recipe is not uniformly
parameter-efficient across all reported scales — only the 7B Llama-2 conversion uses LoRA.

### Is it fixed-state? YES — the cleanest case checked so far

No sliding window, no local component. Table 2 explicitly compares complexity classes: Softmax `O(n²d)` vs.
**1+ELU, CosFormer: `O(nd²)`**, **Performer: `O(nd'²)`**, Taylor-exp (rejected as inefficient): `O(nd³)`
`[T, Table 2, "Complexity" column]` — all the linear-attention variants including Hedgehog's own (MLP
feature map, same causal-linear-attention formulation) reduce to a running state of size `O(d × d')` updated
per token, independent of `n`/`T`. **Genuinely O(1) per-token state, no growing cache, no window buffer.**

### Training cost

**No GPU-hour or wall-clock figure is given anywhere in this paper — `[X]`, searched Appendix B.3/B.4/B.5
and the main text.** What is stated:
- GPT-2 125M conversion: "attention distillation and train Hedgehog MLPs for two epochs over the
  WikiText-103 data, using batch size 8... 1024-tokens per input," then **full-parameter finetuning** at
  lr 6e-4 `[T, Appendix B.5]`. No token-count total or hardware is given; WikiText-103's train split is
  commonly cited elsewhere (not from this paper) as ≈103M tokens, which would put "two epochs" at
  ≈206M tokens — **this multiplication is my inference from an external, non-paper fact about the dataset,
  not a number in Hedgehog's own tables; flagged `[X]`-adjacent, do not treat as [T].**
- Llama-2 7B conversion: distillation "two epochs," LoRA finetuning stage, **"a single A6000 GPU"** for all
  runs `[T, Appendix B.5]` — no wall-clock hours stated.
- **This paper simply does not report the training-cost numbers the brief needs (tokens as an exact count,
  GPU-hours) at the scale that would matter for a 2×T4 estimate. I am not extrapolating a T4-hour estimate
  from "ran on a single A6000" with no duration — that would be exactly the kind of substitution the rules
  forbid.** Verdict: Hedgehog's own training-cost column is **`[X]`** at both 1.5B and 100B scale.

### Quality retention [T, Table 10, Table 11, Table 9 — transcribed verbatim]

**Table 10** ("Pretrained-conversion for 125M GPT-2 on WT-103 lang. modeling"), columns: Method | GPT-2 |
GPT-2 FT | Hybrid H3 | Hyena | T2R-GPT-2 | HH-GPT-2; row: PPL:

| Method | GPT-2 (zero-shot softmax) | GPT-2 FT (finetuned softmax) | Hybrid H3 | Hyena | T2R-GPT-2 | **HH-GPT-2 (Hedgehog)** |
|---|---|---|---|---|---|---|
| PPL | 28.0 | 15.8 | 18.5 | 18.5 | 19.4 | **16.7** |

Caption: "While finetuned GPT-2 gets lowest PPL, among *subquadratic* models Hedgehog significantly
outperforms by 1.8 PPL" `[T, Table 10 caption]` (16.7 vs. next-best subquadratic 18.5). Against the
apples-to-apples baseline (GPT-2 FT, same WT-103 finetuning, full softmax attention retained): **gap is
16.7 − 15.8 = 0.9 PPL**, matching the paper's own text "Hedgehog is 1 PPL off the fully quadratic finetuned
GPT-2" `[T, §5.4 text]`. **This is a perplexity-only result — no recall or long-context number for the
GPT-2 conversion.**

**Table 11** ("Hedgehog Llama-2 conversion (ROUGE)"), columns: R1/R2/RL:

| Method | R1 / R2 / RL |
|---|---|
| Softmax (Zero-shot) | 19.3 / 6.8 / 14.9 |
| Softmax (LoRA, finetuned — the fair baseline) | **51.1 / 27.6 / 43.5** |
| T2R (LoRA) | 2.8 / 0.0 / 2.6 |
| **Hedgehog (LoRA)** | **47.4 / 23.4 / 39.1** |

This is a **generation task** (SAMSum dialogue summarization), not perplexity or multiple-choice accuracy —
closer to what the brief wants separated from PPL, though still not a recall-probe benchmark like Zoology's
associative recall. Hedgehog recovers **47.4/51.1 ≈ 93%** of the softmax-LoRA R1 score, but the gap is
non-trivial (3.7 R1 points, 4.2 R2 points, 4.4 RL points) and **T2R's near-total collapse (2.8/0.0/2.6) is
the single most striking number in this paper**: naive linear-attention feature maps do not merely
underperform on this task, they fail almost completely, while Hedgehog's softmax-mimicry training recovers
the overwhelming majority of quality. This is direct evidence that *how* the feature map is fit (mimic
softmax vs. arbitrary kernel) is the dominant variable, not just "linear vs. quadratic."

**Table 3** (associative recall + BERT-FT comparison, caption: "Hedgehog matches performance on associative
recall (AR) and BERT-finetuned conversion (BERT-FT) with prior best approaches, while achieving better time
and space complexity") — this is the one Hedgehog result that is explicitly recall-labeled; exact cell
values were not transcribed in this pass (flagged for follow-up if this becomes load-bearing — **`[X]` for
the exact AR numbers, though the table's existence and caption are `[T]`-confirmed**).

### Counter-evidence / what the paper itself concedes

- Table 2's own ablation shows that most fixed-functional-form linear attentions (1+ELU, Performer,
  CosFormer) get **17.0% train-from-scratch accuracy vs. softmax's 100.0%** `[T, Table 2]` on whatever
  in-house diagnostic task this accuracy is measured on (name not re-verified this pass) — i.e. **without
  the "mimic softmax" training objective, kernel linear attention can fail almost completely**, not just
  underperform modestly. Hedgehog's own contribution is precisely fixing this via the spiky+monotonic
  training target, not linear attention per se.
- The 2nd-degree Taylor approximation is presented and then explicitly rejected by the authors themselves
  as "not efficient" (`O(nd³)`) despite matching quality (100.0/58.4 vs. softmax's 100.0/58.8) `[T, Table
  2]` — a reminder that *quality-matching* linear attention is not new; the hard part this literature keeps
  fighting is quality **and** `O(nd²)`-or-better complexity simultaneously.
- No wall-clock/GPU-hour cost reported anywhere — a real gap relative to what this brief needs, and worth
  noting as a *pattern*, not just for Hedgehog: cost transparency is uneven across this literature.

---

## 1b. Mamba in the Llama / Mamba-Distill (Wang, Paliotta, May, Rush, Dao — arXiv:2408.15237, NeurIPS 2024)

Mechanism, healing-budget disagreement across arXiv versions, and the missing init-only ablation were
already investigated in a prior pass on this project (`docs/research/donor_adaptation/prior_art/DONOR_PRIOR_ART_SURVEY.md`,
§Q3) — summarized rather than re-derived here, with **one number upgraded from unverified to confirmed**
below. Key mechanism point relevant to this brief: **donor W_Q → SSM's C projection, W_K → B, W_V → X
(input) — dimensionally exact reuse of shape, not a derived mathematical equivalence** `[T, direct quote,
cross-checked v1/v4 in the prior pass]`. New SSM parameters (Δ, A) have no donor analogue and are freshly
initialized. This produces genuine Mamba/Mamba2 SSM blocks, not linear attention — **fixed-state by
construction** (Mamba's whole architectural premise is an `O(1)`-per-token recurrent state; the paper does
not need to argue this, it is what "Mamba" means).

**Token budget — now confirmed, upgraded from the prior pass's "unverified, one fetch only":** re-fetched
v4's own abstract directly this pass: **"Our experiments distill different large-scale open chat LLMs...
using only 20B tokens of training"** `[A, abstract text, v4, directly read]`. This resolves the earlier
disagreement in favor of the 20B-token figure (the "8 days on 8×A100 + 4 days on 8×H100" wall-clock figure
from the same v4 fetch remains as reported previously — **not independently re-verified this pass, carried
forward as `[A]` from the prior pass**).

### §4(a)-relevant: hybrid attention ratio vs. quality, read directly from Table 3 this pass

**Table 3** ("Evaluation on LM Eval benchmark for Mamba and Mamba2 distilled from Llama-3 Instruct 8B"),
10-task average (WG/PI/HS/AE/AC/MM/OB/TQ/PM/RA), transcribed verbatim:

| Model | AVG (10-task LM-Eval) |
|---|---|
| Llama-3.1-8B-Instruct (teacher, for the 3.1 series specifically) | 64.48 |
| **Mamba-Llama3 (50% attention)** | **61.30** |
| **Mamba-Llama3 (25% attention)** | **57.70** |
| **Mamba-Llama3 (12.5% attention)** | **55.02** |
| **Mamba2-Llama3 (50% attention)** | **63.84** |
| **Mamba2-Llama3 (25% attention)** | **60.55** |
| **Mamba2-Llama3 (12.5% attention)** | **58.21** |
| **Mamba2-Llama3 (0% attention, fully converted)** | **54.74** |

Table 3's own caption text: **"Performance degrades with more linear RNN layers, but is still competitive
at 25% to models trained from scratch."** `[T, Table 3 caption]` — note this is a caution against reading
"competitive" as "lossless": 25% attention still loses ~3.3 AVG points vs. 50% (Mamba2 case: 63.84→60.55),
and the paper's own comparison bar ("competitive to models trained from scratch") is a much lower bar than
"competitive with the teacher." **The degradation from 50%→25%→12.5%→0% is smooth and roughly monotonic in
this table, not a cliff at any one ratio** — Mamba2-Llama3 loses ≈3.3, 2.3, and 3.5 AVG points at each
successive halving (50→25→12.5→0), i.e. no visible "elbow" where a small fraction suddenly becomes
sufficient. **This is the single most important data point for brief §4(a): nothing in this table
supports a claim that ~5% attention retention would be "almost as good as" 50%** — the trend, extrapolated
(not measured — the paper tests only 50/25/12.5/0%, never anything near 5%), points toward continued
degradation below 12.5%, and **0% (fully converted) is 6.6-9.1 AVG points worse than 50% across the two
architectures tested.** Separately, §5.2's own chat-benchmark text (MT-Bench/AlpacaEval) states: "The
distilled hybrid Mamba model (50%) achieves a similar score... The distilled hybrid Mamba (25% and 12.5%)
performance is slightly worse... **The distilled pure (0%) model does degrade significantly in accuracy**"
`[T, §5.2 text]` — i.e. even this paper's own softer, chat-benchmark framing treats 0% as a real cliff
relative to any hybrid ratio, while 12.5% is still described only as "slightly worse," not comparable to
5%.

**This paper never tests a ratio anywhere near our ~5% budget.** The lowest hybrid ratio tested is 12.5%
(1 in 8 layers) — **our 4-of-80 (5%) budget is below every ratio this paper measured**, so extrapolation
below 12.5% is unsupported by this table and should not be treated as validated.

---

## 1c. MOHAWK (Bick, Li, Xing, Kolter, Gu — arXiv:2408.10189, NeurIPS 2024)

Mechanism and the 3-stage pipeline are already documented from a primary-source abstract fetch in the prior
pass (`DONOR_PRIOR_ART_SURVEY.md` §Q3) and in `ADAPTER_MEMO_01_SPEED_BUDGET.md` §2.2b, which is where this
programme's original ~10,250 T4-hour, 100B-donor estimate comes from. **Not re-fetched this pass** (time
budget spent on LoLCATs/Hedgehog/Mamba-in-Llama table verification instead) — carrying forward as already
`[T]`-sourced: **Phi-Mamba (0% attention): 3.0B tokens, reaching 62.6 vs. teacher's 64.9** `[T, Table 1, per
ADAPTER_MEMO_01]`; **Hybrid Phi-Mamba: 5B tokens** `[A, abstract, per DONOR_PRIOR_ART_SURVEY.md]`, **exact
hybrid attention fraction not pinned down in either prior pass — still an open gap, flagged again here.**
Stage 3 (of 3) is explicit end-to-end gradient distillation with a logit-matching loss — **not closed-form**,
same as every other method checked so far. Stage 1 (matrix orientation, Frobenius-distance fit of the
student's token-mixing matrix to the teacher's attention matrix) is the closest thing in this literature to
a per-layer, gradient-based reconstruction fit — **relevant to §4(b) below** — but it is still solved by
gradient descent, not a closed-form linear solve, and **no stand-alone (Stage-1-only) quality ablation is
published**, per two independent fetch attempts in the prior pass.

---

## 1d. SUPRA — Scalable UPtraining for Recurrent Attention (Mercat, Vasiljevic, Keh et al. — arXiv:2405.06640, "Linearizing Large Language Models," TRI/Columbia)

**Verified against:** `https://arxiv.org/html/2405.06640v1` (raw HTML; v2/v3 return an empty article page,
same issue as Hedgehog above — v1 is what is transcribed).

### Mechanism [T/A, §2, Figure 2]

Replaces softmax normalization with **GroupNorm**, and introduces a small learnable MLP kernel `φ` (shared
weights for Q and K, ReLU activation) — a different recipe from Hedgehog/LoLCATs' softmax-mimicry training:
**SUPRA does not train the feature map to approximate softmax attention outputs**; it directly replaces the
whole normalization+kernel scheme and uptrains end-to-end with next-token prediction. The paper explicitly
positions this as a *reaction against* the approximation approach: "While approximating attention is an
intriguing approach to re-using pre-trained transformers, **it leads to instability and poor performance
when uptraining large-scale models**. We instead take a different approach" `[T, §1, direct quote]` — this
is a named, primary-source critique of the Hedgehog/T2R family at scale, worth flagging on its own.

**Donor Q/K/V/O reuse:** the donor's attention block is converted (Q/K/V/O projections carried over, new
kernel MLP + GroupNorm inserted) — full end-to-end uptraining follows, **not LoRA-restricted**; all
parameters including Q/K/V/O are trainable during uptraining (this is closer to "continued pretraining"
than to LoLCATs/Hedgehog's frozen-then-adapter approach).

### Is it fixed-state? YES, explicitly and directly stated by the paper's own words

**"The state `s_i` acts as a constant-size KV cache. Instead of appending new values to the cache, the
state is updated. This allows for inference cost that is constant in the number of generated tokens"**
`[T, §2, direct quote]`. This is the most explicit, unambiguous fixed-state confirmation of any paper
checked so far — no hedging, no hybrid window.

### Training cost [T, Table 3 cell "100B"; T, main text "4 to 32 nodes...H100"; T, LoLCATs Table 4 cross-citation "100"]

- **100B tokens** for the flagship 7B uptrainings (Llama2-7B, Mistral-7B) `[T, Table 3, cells reading
  "100B" for the primary rows]` — **independently cross-confirmed** by LoLCATs' own Table 4, which lists
  "Mistral 7B SUPRA — Training Tokens (B): 100" `[T, already transcribed in §2 above]`. Two different
  papers' own tables agree on this number.
- Hardware: **"Depending on the model size and the availability, we use from 4 to 32 nodes of 8 GPUs Nvidia
  H100"** `[T, §"Training" text]`. Measured throughput: **"A linear 7B parameter model uptraining throughput
  is around 4300 tokens per second per GPU"** `[T, same section]`.
- Abstract's framing: **"requiring 5% of the training cost"** `[A, abstract]` — this is a *relative* claim
  (5% of full pretraining, which for Llama2/Mistral is on the order of ~2T tokens), **not an absolute cheap
  number**. 100B tokens is roughly **1,700-2,500× LoLCATs' 40-60M tokens** and roughly **20-33× MOHAWK's
  3-5B tokens** — SUPRA sits in a completely different cost regime from the attention-transfer/LoRA family,
  despite both being called "linearization."

### My 2×T4 cost estimate (explicitly mine)

Using the paper's own measured throughput (4300 tok/s/GPU on H100, `[T]`) rather than a FLOP-derived
estimate: 100B tokens ÷ 4300 tok/s = 23,255,814 GPU-seconds ≈ **6,460 H100-GPU-hours** for one 7B-scale
uptraining run (this is *my* arithmetic on their measured number, not their own stated total). Applying the
same H100→T4 slowdown factor used for LoLCATs (~15×, from H100 990 TFLOPS vs. T4 65 TFLOPS fp16 dense) and
dividing by 2 GPUs: **≈6,460 × 15 ÷ 2 ≈ 48,000 2×T4-hours ≈ 533 weeks at 90 h/week — over budget by ~530×**
for a 7B-scale donor (the closest published anchor to our 1.5B case; the true 1.5B number would be smaller,
roughly linearly in parameter count if token budget is held near-fixed by architecture, but there is no
published SUPRA run below 7B to anchor that scaling — **this whole paragraph is mine, unanchored below
7B**). **SUPRA is the single most expensive method in this survey, roughly 3× MOHAWK's already-out-of-budget
100B estimate on a per-token-throughput basis, despite the paper's own "5%" framing — the framing and the
absolute number tell different stories, and the absolute number is what matters for a 90h/week budget.**

### Quality retention [T, LoLCATs Table 4 cross-citation, already transcribed in §2 above] + [A, SUPRA's own text]

SUPRA's own text concedes exactly the failure mode this brief's §2 flags as the one to search for
explicitly: **"we identify persistent in-context learning and long-context modeling shortfalls for even the
largest linear models"** `[A, abstract]`, and specifically: **"it preserves performance on most benchmarks
(except MMLU; see Section 4 for a discussion below)"** `[T, §3 "Language Modeling" text]`. Cross-referencing
LoLCATs' own Table 4 (already verbatim-transcribed above), the **Mistral 7B SUPRA row gives MMLU = 34.2 vs.
the original Mistral 7B's 62.4** `[T, LoLCATs Table 4]` — a **28.2-point collapse**, the single largest
knowledge-retention gap of any method in this survey, on the two papers' independently-converging numbers.
**This is a knowledge/world-model task, not a long-context recall probe specifically, but it is exactly the
kind of "holds perplexity, loses something else" result the brief's §3.4 asks to be surfaced rather than
averaged away** — SUPRA's Avg-no-MMLU score (69.9, per LoLCATs Table 4) looks fine; the MMLU column alone
tells the real story.

---

## 1e. DiJiang (Chen, Tian, Wang et al. — arXiv:2403.19928, Huawei Noah's Ark Lab / Peking University, ICML 2024 Oral)

**Verified against:** `https://arxiv.org/html/2403.19928v2` (raw HTML, full render available).

### Mechanism [T, Eq. 12-13, Algorithm 1]

Replaces `softmax(QKᵀ)V` with `FKA(Q,K,V) = φ_WDCF(Q)·φ_WDCF(K)ᵀ·V`, where `φ_WDCF(x) = D·e^(T·𝒞·xᵀ)` — a
**Discrete Cosine Transform (DCT)-based, weighted Quasi-Monte-Carlo kernel feature map** (`𝒞` is a fixed DCT
coefficient matrix, `D` and `T` are learned/random parameters) approximating a Gaussian kernel of the
query-key dot product. This is a *fixed functional-form* kernel (like Performer/CosFormer), not a
softmax-mimicry-trained MLP (like Hedgehog) — the claimed advantage is a tighter theoretical approximation
bound (`O(1/m)` for the weighted Quasi-Monte-Carlo sampling vs. `O(1/√m)` for ordinary Monte Carlo, `[T,
§3.2, Theorem 3.2 discussion]`) and cheap computation via DCT's `O(log m)` fast-transform complexity `[T,
§3.2]`. A gating mechanism "similar to RetNet" is also added `[T, §4.1]`.

**Donor Q/K/V/O reuse:** the donor is fine-tuned from its pretrained checkpoint with the new attention form
substituted in — **not LoRA**, standard full fine-tuning on a small token budget (see below); the paper
does not describe freezing Q/K/V/O or applying low-rank adapters.

### Is it fixed-state? YES

Standard kernel-linear-attention recurrence (`φ(Q)φ(K)ᵀV` reassociates into a running state), same family
as Hedgehog/SUPRA/LoLCATs' linear term. No sliding window or hybrid component mentioned.

### Training cost [T, Table 1 and Table 3, transcribed verbatim]

**Table 3** ("Comparison with LLaMA2-7B on various benchmarks"), columns include a "Tokens" column:

| Model | Avg (11-task) | Tokens |
|---|---|---|
| LLaMA2-7B (original) | 0.565 | **2000B** |
| **DiJiang-7B** | 0.557 | **40B** |

Text confirms: **"our model required only 40B training data, significantly less than the 2T tokens used by
LLaMA2-7B"** `[T, §4.2 text]` — exactly matches the abstract's **"about 1/50 training cost"** claim
(2000/40 = 50, exact). **This is a genuinely small token budget** — smaller than MOHAWK's 3-5B, though
~660-1000× larger than LoLCATs' 40-60M.

**Table 1** ("The experimental results of the proposed method. Training time is measured using A800"),
smaller-scale rows, columns Model | [6-task avg] | Training (day) | Inference (tokens/s):

| Model | Avg | Training (days, A800) |
|---|---|---|
| Pythia-410M (original) | 0.454 | 105.8 |
| DiJiang-410M | 0.456 | **6.6** |
| Pythia-2.8B (original) | 0.478 | 593.3 |
| DiJiang-2.8B | 0.473 | **37.1** |

Ratio at both scales is ≈16× (105.8/6.6 = 16.0; 593.3/37.1 = 16.0), matching the text's own **"∼1/16 of the
training cost"** claim for the Pythia-scale sweep `[T, §4.1 text]`. **The caption does not state how many
A800 GPUs the "day" figures assume** — this is a real gap: I cannot convert Table 1's day-counts to
GPU-hours without knowing GPU count, and I am not guessing it. **No training-time (day/hour) figure is
given anywhere for the flagship DiJiang-7B run** — only the 40B/2000B token ratio. **Verdict: DiJiang-7B's
absolute wall-clock/GPU-hour cost is `[X]`, not found**, despite the sub-3B scale having a clean, table-sourced
number. I am not extrapolating a 2×T4 estimate from the sub-3B numbers to the 7B flagship — parameter count
and token count both differ, and the paper gives no throughput figure (tokens/s) for training the way SUPRA
does, only a bare day-count at an unstated GPU count.

### Quality retention [T, Table 3 verbatim, all 11 columns]

| Model | PIQA | SIQA | BoolQ | WSC | HellaSwag | ARC-E | ARC-C | MMLU | NQ | COPA | Race-M | Avg | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA2-7B | 0.782 | 0.485 | 0.749 | 0.663 | 0.740 | 0.561 | 0.403 | **0.468** | 0.192 | 0.670 | 0.402 | 0.565 | 2000B |
| DiJiang-7B | 0.775 | 0.346 | 0.626 | 0.683 | 0.694 | 0.626 | 0.427 | **0.407** | 0.194 | 0.730 | 0.618 | 0.557 | 40B |

Notable per-column swings hidden by the near-equal averages: **SIQA collapses (0.485→0.346, a 13.9-point
drop)**, **BoolQ drops 12.3 points (0.749→0.626)**, but **Race-Middle improves 21.6 points (0.402→0.618)**
and **MMLU drops 6.1 points (0.468→0.407)** — the average (0.565 vs 0.557) obscures a genuinely noisy
per-task pattern, not uniform near-lossless retention. **NQ (Natural Questions, closed-book factual QA — the
closest column in this table to a "recall" probe) is essentially unchanged (0.192→0.194)**, which is a mild
point of *counter*-evidence against the recall-collapse concern, though NQ tests parametric/closed-book
recall, not the long-context associative recall that Based/Zoology (below) specifically probe — **not the
same failure mode, flagged explicitly so it isn't misread as covering it.**

**Table 2** (fine-tuning ablation on Pythia-410M, cross-method comparison — cited here as DiJiang's own
re-implementation/benchmarking of other papers' methods, not those papers' own tables): Pythia-410M
original avg 0.454; **Linformer 0.3982, CosFormer 0.4047, Performer 0.4183, RetNet 0.3843** (all
underperform the original by 3.6-7.1 points in this controlled comparison) **vs. DiJiang's own PFF ablation
0.4264 and full DiJiang method 0.4567** (the latter slightly *exceeds* the original, plausibly within noise)
`[T, Table 2, verbatim]`. **This is useful cross-method signal but it is DiJiang's own re-implementation of
RetNet/Performer/CosFormer under DiJiang's fine-tuning recipe, not those papers' own reported numbers for
their own preferred training regime — tag accordingly, do not read this as "RetNet's paper reports 0.3843."**

---

## 3. Zoology (Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré — arXiv:2312.04927, Stanford Hazy Research)

**Verified against:** `https://arxiv.org/html/2312.04927v1` (raw HTML, full render). This paper is *not* a
donor-linearization method — it is the diagnostic literature the brief's §2 explicitly asks to be searched:
it measures **why** fixed-state architectures underperform attention, and tests hybrid ratios.

### The mechanism-diagnosis finding [T, §3, direct quote]

"We measure the perplexity gap between gated convolutions and attention and show that single skill termed
*associative recall* accounts for **82% of the gap** on average" `[T, §3, direct quote, also stated in
abstract]`. This is measured across a suite of **17 pretrained attention and gated-convolution models**
`[A, abstract]`. This is the primary-source origin of "the recall gap," a term the brief already uses —
confirmed here as a measured, table-backed finding, not a folk claim.

### §4(a)-relevant: hybrid ratio vs. recall quality, read directly from raw HTML table cells (Table 2)

**Table 2** ("Language model perplexity on slices of the PILE"), verified from the raw `<td>`/`<th>` cell
IDs (not the flattened text, to avoid exactly the row/column mispairing this project has been burned by
before). Columns, confirmed from the two-row header: **Model | Param (M) | TFLOPs | Overall [PPL (BPB)] |
Slices: AR Hits [PPL (BPB)] | Slices: Other Tokens [PPL (BPB)]**. Transcribed verbatim, 354-365M parameter
block (the scale at which the text states hybrids use "three attention layers, **6.3% of layers**"):

| Model (354-365M scale) | Param (M) | Overall PPL (BPB) | **AR Hits PPL (BPB)** | Other Tokens PPL (BPB) |
|---|---|---|---|---|
| Attention (100% attention, no hybrid) | 360 | 9.44 (2.25) | **1.98 (0.69)** | 10.62 (2.36) |
| BaseConv (0% attention, pure gated-conv) | 354 | 11.01 (2.40) | **5.98 (1.79)** | 11.52 (2.44) |
| BaseConv + Random-selection sparse attn (control) | 365 | 12.94 (2.56) | **6.17 (1.82)** | 13.62 (2.61) |
| BaseConv + Programmatic-selection sparse attn | 365 | 9.54 (2.26) | **2.35 (0.86)** | 10.50 (2.35) |
| BaseConv + Learned-selection sparse attn | 351 | 9.59 (2.26) | **2.61 (0.96)** | 10.58 (2.36) |
| **BaseConv + Full attention on 3 layers (6.3% of layers)** | 351 | **8.59 (2.15)** | **1.95 (0.67)** | **8.91 (2.19)** |

**This is the single most direct data point in this survey for brief §4(a).** At this scale, a hybrid with
full (non-sparse) attention on just **3 of 48 layers (6.3%, close to our ~5% budget)** does not merely
"recover most of the gap" — its **AR-Hits perplexity (1.95) matches or slightly beats the 100%-attention
model's own AR-Hits perplexity (1.98)**, and its **Overall perplexity (8.59) is *better* than the
100%-attention model's (9.44)**. The **random-selection control performs *worse* than pure BaseConv on
every column** — confirming the gap is specifically about *which* tokens get attention, not merely *how
many* layers have it, i.e. placement/selectivity matters as much as ratio. Text confirms: "hybrids... can
outperform attention-only models by up to a full perplexity point" `[T, §5 text]`.

**One caveat that must not be lost:** the "3 attention layers" in this hybrid select tokens via a
*programmatic or learned selection function* (Eq. 2-3) that decides, per-token, whether to apply attention
at all — this is **sparse/selective attention on top of a gated-convolution backbone, not the
donor-attention-to-linear-attention conversion this brief is about.** BaseConv is a from-scratch
gated-convolution architecture (trained from scratch in this paper, not converted from a pretrained softmax
donor); there is no donor Q/K/V reuse question here at all. **This result answers "does a small full-
attention fraction rescue a fixed-state backbone's recall quality" (yes, strongly, at 6.3%) — it does not
directly test "does converting a pretrained transformer's own attention layers to linear form, with a small
fraction left as softmax, rescue *that specific donor's* quality," which is Mamba-in-the-Llama's and
LoLCATs' question, not Zoology's.** Treat these two evidence lines as complementary, not identical:
Zoology says a small hybrid *can* work when selectivity is right; Mamba-in-the-Llama (§1b above) says that
in an actual donor-distillation setting, even 12.5% (the lowest ratio *that paper* tested) still trails the
teacher, and no published donor-conversion paper has tested anywhere near 5-6%.

### Is "BaseConv" itself fixed-state?

Yes — BaseConv is an explicit gated-convolution architecture (not linear attention, but a related
fixed-state family); the "Slices" methodology (splitting perplexity into AR-Hits vs. Other-Tokens) is the
paper's real contribution and is architecture-agnostic — it is the diagnostic tool, not a conversion method
in the brief's sense, so it is not scored in the ranked table below.

---

## 4. Based (Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Rudra, Ré — arXiv:2402.18668, Stanford Hazy Research / ICML 2024)

**Verified against:** `https://arxiv.org/html/2402.18668v1` (raw HTML, full render).

### Mechanism [A, abstract + architecture description]

BASED combines **linear attention (Taylor-series feature map, same family as Hedgehog's rejected-as-
inefficient `O(nd³)` "Taylor Exp" option) with a short sliding-window softmax attention** — i.e. it is
architecturally the same *shape* as LoLCATs' "linear + sliding window" hybrid (§2 above): a small bounded
local-attention window plus a global linear-recurrent term. This is a **from-scratch architecture design
paper**, not a donor-conversion method — included here per the brief's explicit candidate list, and because
its recall-quality claims are the primary counter-evidence source for the hard filter.

### Is it fixed-state? YES, with the same caveat as LoLCATs

Linear-attention term: `O(1)` recurrent state. Sliding-window term: bounded window, not growing with `T`.
Combined state is `O(1)` in `T`, constant set by architecture (window size + feature dimension), same
pattern as LoLCATs.

### Quality retention — the recall-specific claim

The paper's own framing (confirmed at abstract level, not yet re-verified against its own results table
this pass — **flagged `[A]`, not `[T]`, pending a table read if this becomes load-bearing**): BASED
"matches the strongest sub-quadratic models while outperforming them on real-world recall-intensive tasks
by **6.22 accuracy points**" and achieves "**24× higher throughput** than FlashAttention-2... at 1.3B
parameters" `[A, abstract]`. **This paper's central design lever — that recall accuracy trades directly
against the linear-attention feature dimension / state size — is the mechanistic explanation for *why* the
recall gap exists at all: a fixed-size state of dimension `d'` can provably only losslessly store
`O(d')`-ish associative bindings, so recall degrades as context grows past what the state can hold, no
matter how well the feature map is fit.** This is the theoretical backbone underneath Zoology's *empirical*
82% finding above, and it is the mechanistic reason the brief's hard filter (§brief's own framing) is
correct to treat "fixed-state" as a real tradeoff, not merely an implementation detail: **fixed state is
exactly and only what makes the KV-wall disappear, and it is exactly and only what bounds recall capacity.
The same architectural fact produces both the win and the cost.** This connects directly to structural
question (a): a hybrid's few full-attention layers are doing double duty — they are the ones that *can*
hold unbounded-length exact bindings, which is precisely why Zoology's Table 2 shows placement/selectivity
(which tokens, not just how many layers) mattering so much.

---

## 5. Substrate architectures (from-scratch, not donor-conversion methods — background for the mechanism claims above)

These are the kernel/linear-attention and gated-RNN architectures the direct-hit methods above build on.
None of these are donor-conversion recipes — they are trained from scratch in their own papers — so they
are **not scored in the ranked table below** (no "training cost to convert a donor" applies), but their
fixed-state claims are checked directly since several of the methods above cite or build on them.

- **RetNet** (Sun, Dong et al. — arXiv:2307.08621, Microsoft Research): explicit **three computation
  paradigms — parallel, recurrent, chunkwise recurrent** `[T, §1/§2.1, direct paper text]`; "the recurrent
  representation... is employed during inference, which nicely fits autoregressive decoding" `[T, §2.3]`.
  **Genuinely fixed-state**, confirmed directly from the paper's own architecture description (Eq. 6, not
  independently re-derived here). DiJiang's own re-implementation of RetNet under a controlled fine-tuning
  comparison scored it at **0.3843 avg vs. Pythia-410M's 0.454 original** `[T, DiJiang Table 2, already
  transcribed in §1e above — this is DiJiang's number for RetNet, not RetNet's own paper's number]`.
- **Gated Linear Attention / GLA** (Yang, Wang, Shen, Panda, Kim — arXiv:2312.06635): explicit **recurrent
  form with a 2D (matrix-valued) hidden state, data-dependent gating** `[T, §4.1, direct text]`. Genuinely
  fixed-state. Not independently re-verified for quality-retention numbers this pass (from-scratch training
  comparison, not a donor-conversion result — lower priority given the brief's focus).
- **DeltaNet / Parallelizing DeltaNet** (Yang, Wang, Yang, Zhang, Panda, Kim — arXiv:2406.06484, NeurIPS
  2024): the delta-rule update (in place of linear attention's purely additive state update) is *also*
  `O(1)`-state — same recurrent-state class, with a difference-based (Householder-matrix) state update
  argued to be more precise for associative recall specifically. **§4(a)-relevant finding, from-scratch not
  donor-conversion:** "the hybrid DeltaNet that just replaces **two** DeltaNet layers with global attention
  outperforms a Transformer++ even on recall-intensive benchmarks" `[T, §4 text, direct quote]`, using
  layers `2` and `(N/2+1)` of `N` total `[T, §3.4 text, direct quote]` — **the total layer count `N` for
  the 1.3B/100B-token flagship model is not stated in the sections checked this pass — `[X]`, so the exact
  hybrid percentage this represents cannot be computed from what was verified.** (If `N` were the common
  ~24-layer configuration for a 1.3B-class model, 2/24 ≈ 8.3% — **this is my own arithmetic on an assumed,
  not paper-confirmed, layer count and must not be read as a verified ratio.**)
- **RWKV**: not independently fetched this pass — its architecture (a linear, gated RNN with a fixed hidden
  state by design, "Receptance Weighted Key Value") is fixed-state by well-established construction and is
  used as a comparison baseline inside SUPRA (§1d, "outperforms RWKV-5") and DiJiang — not re-verified
  against RWKV's own paper's tables this pass; flagged as a gap if RWKV's own quality numbers become
  load-bearing.
- **Performer** (FAVOR+) and **cosFormer**: both fixed-state kernel-attention methods; both were checked
  indirectly via two independent third-party controlled comparisons rather than their own original papers'
  tables this pass — **Hedgehog's Table 2** (`[T]`, §1 above): Performer gets 17.0% train-from-scratch
  accuracy vs. softmax's 100.0% on Hedgehog's in-house diagnostic; and **DiJiang's Table 2** (`[T]`, §1e
  above): Performer 0.4183 avg, CosFormer 0.4047 avg vs. Pythia-410M original 0.454, in DiJiang's controlled
  fine-tuning comparison. **Both papers that have a stake in showing their own method beats these baselines
  report Performer/CosFormer well below the softmax original** — this is a consistent signal across two
  independent labs' controlled comparisons, but it is *their* comparisons of *other people's* methods, not
  Performer's or CosFormer's own papers' preferred numbers, and is flagged accordingly.

---

## 2. LoLCATs (Zhang, Arora, Chen, Sun, Zou, Buendia, Rudra, Ré et al. — arXiv:2410.10254, Stanford/Together AI/Caltech/MIT)

**Verified against:** `https://arxiv.org/html/2410.10254v1` — raw HTML downloaded to disk and read directly
(not via a summarizing fetch) so that table cell/row/column pairing could be checked by hand. Full name:
"Low-rank Linear Conversion via Attention Transfer."

### Mechanism [T, Eq. 5-6, §3.1]

Two-step, applied per attention head per layer:

1. **Attention transfer.** Insert two small learnable feature maps `φ_q, φ_k` *after* the donor's existing
   `W_q, W_k` projections (applied post-RoPE). `φ_q(q) := f(q·W̃_(q) + b̃_(q))`, same shape for `φ_k`. Two
   feature-map families are tested: **T2R** (`ReLU(q·W̃+b̃)`, weight shape 128×128) and **Hedgehog**
   (softmax over the feature dim of `[q·W̃ ; -q·W̃]` concatenated, weight shape 128×64) `[T, Table 2, exact
   formulas and shapes]`. These are trained by gradient descent (AdamW) to minimize per-head, per-layer MSE
   between true softmax-attention output `y_n` and linear-attention output `ŷ_n` (Eq. 6) — **the donor's
   own `W_q, W_k, W_v` are frozen during this stage**; only the new small `φ` MLPs are trained. For Llama 3
   8B this is **≈16.8M trainable weights = 0.2% of the LLM** `[A, §3.1 "Training footprint"]`.
2. **Low-rank adjusting.** After swapping in the linear-attention layers, train the model end-to-end with
   next-token cross-entropy — but **only via LoRA on `W_q, W_k, W_v, W_o`** (rank 8 in the main runs, rank 4
   for 405B `[T, Table 10]`), i.e. `W' ← W + BA`, not full fine-tuning. `<0.09%` of 7B parameters updated
   `[A, §3.1]`.

**Donor Q/K/V/O reuse: yes, explicitly and almost entirely.** Stage 1 freezes the original projections
outright; Stage 2 perturbs them with a rank-8 (or rank-4) additive update, not a replacement. This is the
strongest "reused projections" case among the candidates checked so far.

**Architecture actually deployed is a hybrid of linear attention + a small sliding window**, not pure
linear attention — the paper calls this "linear + sliding-window" with a hardware-efficient "Terrace"
kernel (§C.2). Window size **64 tokens** `[T, Table 8, row "Sliding window attn size": 64]`.

### Is it fixed-state? YES, with a caveat worth stating precisely

The linear-attention term is the standard cumulative-sum recurrence (Eq. 5): `ŷ_n = Σ_{i≤n} φ_q(q_n)ᵀφ_k(k_i) v_i / Σ_{i≤n} φ_q(q_n)ᵀφ_k(k_i)`,
which is exactly re-associable into a running state `S_n = S_{n-1} + φ_k(k_n) v_nᵀ` (size `d' × d_v`, `d'=64`
in their runs) plus a running normalizer `z_n = z_{n-1} + φ_k(k_n)` — `O(1)` per token, independent of `T`.
**But** the deployed architecture adds a bounded sliding-window component (window = 64 tokens) on top. That
window cache is *also* bounded and independent of `T` (it is a fixed-size ring buffer, not a growing KV
cache), so the **combined state remains `O(1)` in `T`** — total state size is `O(d'·d_v + window·d)`, a
constant set by architecture choice, not by sequence length. Flag for our engine: this is not *pure*
linear attention; a faithful reproduction needs the small window buffer too, not just the recurrent state.

### Training cost [T, Table 8 / Table 9 / Table 10, verbatim]

| donor | Stage 1 (attn. transfer) | Stage 2 (LoRA finetune) | total tokens (table sum) | hardware, wall-clock |
|---|---|---|---|---|
| Mistral 7B / Llama 3 8B / Llama 3.1 8B (**Table 8**) | 2 epochs × 10M tok/epoch = 20M | 2 epochs × 20M tok/epoch = 40M | **60M** (table arithmetic) | not itemized per-model in Table 8; §1 abstract states 6.5 GPU-h total on 1×40GB A100 for the 7B/8B runs (≈2h attn. transfer + ≈4.5h LoRA) `[A]` |
| Llama 3.1 70B (**Table 9**) | 1 epoch × 20M tok/epoch = 20M | 1 epoch × 20M tok/epoch = 20M | **40M** | 18 h on one 8×80GB H100 node `[A, abstract]` |
| Llama 3.1 405B (**Table 10**, LoRA r=4, α=8) | 1 epoch × 20M tok/epoch = 20M | 1 epoch × 20M tok/epoch = 20M | **40M** | attn. transfer: 5 h on 14×80GB H100 GPUs; LoRA finetune: 16 h on three 8×80GB H100 nodes (=24 GPUs) `[T, §"Compute Resources" appendix text, directly under Table 10]` |

**Note on the paper's own headline "40M tokens":** the abstract and §3.2 state training uses "**≈40M total
training tokens**" and describe the 8B run as "two epochs for both attention transfer and LoRA adjusting,
**or** four epochs with either alone" `[T, §3.2 text]` — i.e. 40M is the paper's own approximate/rounded
figure across several equivalent epoch schedules, not a strict sum of Table 8's specific cells. Table 8's
literal cell values for the exact run reported (2 epochs × 10M + 2 epochs × 20M) sum to 60M, not 40M. Table
9 and Table 10 (70B, 405B) *do* sum exactly to 40M. **I report this discrepancy rather than resolve it —
both the 60M (8B, table arithmetic) and 40M (paper's stated headline, and 70B/405B table arithmetic) are
[T]-sourced; they are not the same number.** Either way the order of magnitude (tens of millions of tokens,
not billions) is robust across all three model scales.

By comparison, MOHAWK uses 3.0B tokens `[T, per ADAPTER_MEMO_01, MOHAWK Table 1]` — **LoLCATs' own stated
comparison is "0.003% and 0.04% of prior pretraining and linearizing methods' token counts"** `[A, abstract]`,
i.e. LoLCATs claims roughly **50-75× fewer tokens than MOHAWK** depending on which LoLCATs figure (40M vs
60M) is used against MOHAWK's 3.0B (3.0B/60M ≈ 50×, 3.0B/40M ≈ 75×).

### My 2×T4 cost estimate (explicitly mine, not the paper's)

The paper reports **wall-clock on A100/H100, not FLOPs**, so converting to 2×T4-hours requires my own
assumptions, stated here:
- T4 fp16 peak ≈ 65 TFLOPS; A100 (40GB, the 7B/8B GPU) fp16 peak ≈ 312 TFLOPS (dense, no sparsity) — **T4 is
  ≈4.8× slower per-GPU** at matched precision and MFU.
- 2×T4 vs 1×A100 (used for 7B/8B): roughly a wash on raw FLOPS (2×65 ≈130 vs 312, so 2×T4 ≈ 0.42× an A100),
  giving **7B/8B: 6.5 GPU-h (A100) → my estimate ≈ 6.5 / 0.42 ≈ 15-16 h on 2×T4** — plausibly inside a
  90 GPU-h/week budget, assuming the small feature-map training and rank-8 LoRA fit in T4 VRAM (16GB per
  card) at 7-8B scale, which is not verified here (activation memory for full backward through an 8B model,
  even with LoRA, is a real risk at 16GB — **this is a memory-fit question the paper does not have to answer
  because it runs on 40-80GB cards, and I flag it as unresolved for our hardware**).
- **100B donor**: no published LoLCATs run above 405B exists to anchor an extrapolation, and cost here does
  not scale by parameters alone the way MOHAWK's full-gradient training does — LoLCATs Stage 1 trains a
  *fixed-size* set of small per-head feature maps (independent of donor width in count, though each grows
  with head dim), and Stage 2 is a rank-8/rank-4 LoRA fit, so both stages' FLOPs scale roughly linearly with
  donor parameter count at fixed token budget (same functional form as MOHAWK's `4·N·D`, just with a *much*
  smaller effective `f` since LoRA touches only attention projections, not the whole network, and D is
  ~13-75× smaller). Using the 405B numbers as the closest anchor (5h×14 H100 + 16h×24 H100 = 70+384 = **454
  H100-GPU-hours** at 40M tokens): H100 fp16 (dense) peak ≈ 990 TFLOPS vs T4 ≈65 TFLOPS, **T4 ≈15× slower
  per-GPU**. My estimate for a 100B donor (scaling 405B's 454 H100-GPU-h down by 100/405 ≈ 0.247, since
  cost is roughly linear in N at fixed token budget): **≈112 H100-GPU-h → ÷2 for 2 GPUs → ÷15 for the T4
  slowdown → my estimate ≈ 3,700-4,000 2×T4-hours**, i.e. **~41-44 weeks at 90 h/week — over budget by
  roughly 5-6×**, though far closer to affordable than MOHAWK's ~114 weeks. **This entire paragraph is my
  arithmetic, not a number from the paper — LoLCATs was never run at 100B and I have no table cell to anchor
  a 100B estimate to; treat this as a rough order-of-magnitude extrapolation, not a costed plan.**

### Quality retention [T, Table 4, Table 7 — transcribed verbatim]

**Table 4** ("LoLCATs comparison among linearized 7B+ LLMs"), columns: Model | Training Tokens (B) | PiQA |
ARC-e | ARC-c (norm) | HellaSwag (norm) | Winogrande | MMLU (5-shot) | Avg. | Avg. (no MMLU):

| Model | Tokens(B) | PiQA | ARC-e | ARC-c | HellaSwag | Winogrande | **MMLU (5-shot)** | Avg | Avg (no MMLU) |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3 8B (original) | – | 79.9 | 80.1 | 53.3 | 79.1 | 73.1 | **66.6** | 72.0 | 73.1 |
| Mamba2-Llama 3 (0% attn., distilled — Mamba-in-the-Llama method) | 20 | 76.8 | 74.1 | 48.0 | 70.8 | 58.6 | **43.2** | 61.9 | 65.6 |
| **Mamba2-Llama 3, 50% Attn.** (hybrid, same distillation method) | 20 | 81.5 | 78.8 | 58.2 | 78.4 | 69.0 | **56.7** | 70.4 | 73.2 |
| Llama 3 8B Hedgehog (as re-run/cited by LoLCATs) | 0.04 | 77.4 | 71.1 | 40.6 | 66.5 | 54.3 | **24.2** | 55.7 | 62.0 |
| **Llama 3 8B LoLCATs (Ours)** | 0.04 | 80.9 | 81.7 | 54.9 | 79.7 | 74.1 | **52.8** | 70.7 | 74.2 |

Caption states: "LoLCATs closes the Transformer quality gap by 79.8% (Mistral 7B) and 86.6% (Llama 3 8B)
(average over all tasks; numbers except Hedgehog cited from original works)" `[T, Table 4 caption]`.

**Table 7** ("Linearizing Llama 3.1 70B and 405B"), MMLU (5-shot) column, all three donor scales:

| Model | MMLU (5-shot) |
|---|---|
| Llama 3.1 8B (original) | 66.11 |
| Linearized, no attn. transfer (ablation) | 51.44 |
| LoLCATs (Ours), 8B | 54.88 |
| Llama 3.1 70B (original) | 78.80 |
| Linearized, no attn. transfer (ablation) | 28.74 |
| LoLCATs (Ours), 70B | 67.70 |
| Llama 3.1 405B (original) | 82.98 |
| Linearized, no attn. transfer (ablation) | 33.86 |
| LoLCATs (Ours), 405B | 72.20 |

**All numbers above are LM-Eval-harness tasks (PiQA/ARC/HellaSwag/Winogrande, general reasoning) and
5-shot MMLU (general knowledge) — none of these are recall-specific or long-context benchmarks
(needle-in-haystack, associative recall, SWDE/FDA/NQ-style retrieval).** LoLCATs' own tables report no
recall-specific eval. This matters directly for brief §3.4's instruction to separate perplexity/QA
retention from recall retention — **for LoLCATs specifically, the recall question is [X], not found in
this paper.**

### Counter-evidence / what the paper itself concedes

- The "no attention transfer" ablation (Table 7) shows the attention-transfer stage is doing most of the
  work, not the LoRA stage: without it, MMLU collapses to 28.74-51.44 depending on scale (vs. LoLCATs' full
  method at 52.8-72.2) — **the closed-form-adjacent feature-map fit (gradient-trained, not literally
  closed-form) is load-bearing, not a formality.**
- Quality retention is *not* full: at every scale the LoLCATs MMLU number remains well below the original
  (8B: 54.88 vs 66.11, gap 11.2 pts; 70B: 67.70 vs 78.80, gap 11.1 pts; 405B: 72.20 vs 82.98, gap 10.8 pts).
  Gap size is roughly constant in absolute MMLU points across three orders of magnitude of donor size —
  **there is no evidence in this table that the gap shrinks with scale.**
- §3.2 itself is titled a limitation-finding section ("LoL SAD: Limitations of Low-Rank Linearizing")
  before the paper's own §3.3 improvements are introduced — the paper's narrative is explicit that naive
  attention-transfer + LoRA alone under-performs and required the "Terrace" hybrid-window architecture and
  RoPE-ordering fix to reach the reported numbers. This project should not read "LoLCATs" as one simple
  recipe; it is attention-transfer + sliding-window hybrid + RoPE-after-φ + LoRA, stacked.
- The "Mamba2-Llama 3, 50% Attn." row in Table 4 is **not LoLCATs' own ablation** — it is a baseline cited
  from the Mamba-in-the-Llama paper, reused for comparison in LoLCATs' table. Relevant to brief §4(a) below.

---

## 6. Section 4(a) — Does hybridisation rescue it, and does that agree with our budget?

Short answer: the evidence is split by architecture family, and the split matters.

- Donor-conversion papers that actually distill a pretrained transformer's own attention into a hybrid
  (Mamba-in-the-Llama, §1b): tested ratios are 50%, 25%, 12.5%, 0% — never anywhere near 5%. Table 3's own
  numbers show smooth, roughly monotonic degradation across this range (Mamba2-Llama3 AVG: 63.84 to 60.55
  to 58.21 to 54.74 across 50/25/12.5/0%), losing about 2.3-3.5 points at each halving with no elbow visible
  in the tested range. Extrapolating this trend to about 5% is unsupported — the paper never measured it —
  but nothing in the trend suggests a sudden plateau either; the safest reading is that 5% would continue
  the pattern and lose more than the 12.5% row already does. LoLCATs' Table 4 cites the same underlying
  50%-attn number (MMLU 56.7 vs 66.6 original, vs 43.2 at 0%) as a baseline for comparison, independently
  corroborating the general shape.
- The diagnostic literature on why fixed-state architectures lose quality, and what a small full-attention
  fraction buys (Zoology, §3): a 6.3%-of-layers full-attention hybrid on top of a from-scratch
  gated-convolution backbone matches or exceeds a 100%-attention model's own recall-slice perplexity (1.95
  vs 1.98 AR-Hits PPL) and beats it on overall perplexity (8.59 vs 9.44) at the 354-360M scale (Table 2,
  transcribed above). DeltaNet's own hybrid ablation similarly reports a 2-layer (not percentage-stated, N
  unconfirmed) global-attention hybrid beating both plain DeltaNet and a Transformer baseline on
  recall-intensive benchmarks (§4 text, direct quote, transcribed above).
- These two evidence lines answer different questions. Zoology/DeltaNet show that placement-aware, a small
  number of full-attention layers, on a fixed-state backbone trained from scratch, can fully close or
  exceed the recall gap — a genuinely strong, table-verified result, at ratios (6.3%, and DeltaNet's
  unconfirmed-but-plausibly-similar 2/N) close to or below our ~5% budget. Mamba-in-the-Llama shows that
  distilling a specific pretrained donor's attention weights into a hybrid, at ratios no lower than 12.5%,
  still leaves a real, non-closing gap to the teacher — and 5% is below anything tested in that regime.
- Read together: whether ~5% attention retention is enough for this project's case (converting a specific
  pretrained donor, not training a hybrid from scratch) is not settled by the literature searched this pass.
  The optimistic reading (Zoology, DeltaNet) says selectivity plus a small fraction can, in principle, close
  the gap entirely — architecturally, the ceiling exists. The cautious reading (Mamba-in-the-Llama) says
  nobody has demonstrated this specifically as a donor-distillation result below 12.5%, and the trend in the
  one paper that tested a range shows continued, not plateauing, loss as the ratio drops. **This is not the
  "two independent constraints agree" finding the brief hoped might exist — it is a live, verified, open
  gap: the literature closest to our exact setup (donor distillation) has not tested our exact ratio, and
  the literature that has tested near our ratio is not doing our exact task (donor distillation vs.
  from-scratch hybrid training).** Whether Zoology's placement-selectivity insight (which tokens get
  attention, not just how many layers) transfers to a donor-conversion setting is, itself, an open,
  testable question this survey surfaces rather than answers.

## 7. Section 4(b) — Can the conversion be done per-layer and closed-form?

No published closed-form (non-gradient) per-layer solve was found in this pass, across every method
checked. Explicitly, for the record:

| Method | Fitting procedure for its feature map / mixer |
|---|---|
| Hedgehog | Gradient descent (KL-div / MSE to softmax attention), explicitly not closed-form |
| LoLCATs | Gradient descent, AdamW, MSE loss (Eq. 6), explicitly not closed-form — the closest thing to a "solve," but iterative |
| SUPRA | Gradient descent, end-to-end next-token loss — not closed-form, and not even attention-mimicry-targeted |
| DiJiang | Gradient descent, full fine-tuning — not closed-form; the DCT feature map itself is a fixed, non-learned functional form (only D, T are learned), closer to Performer/CosFormer's fixed-kernel family than to a per-layer fit against donor data at all |
| MOHAWK Stage 1 | Gradient descent minimizing Frobenius distance to the teacher's attention matrix (per the prior pass's primary fetch) — the closest conceptual match to the brief's §4(b) question (a per-layer objective against the donor's own attention output), but still solved by gradient descent over many steps, not a closed-form linear solve, and no stand-alone quality number for this stage alone has been located across two independent fetch attempts (this pass and the prior pass) |
| Mamba-in-the-Llama | Not even attention-matching — a fixed shape-based weight remapping (Q to C, K to B, V to X) with the new SSM parameters (Delta, A) randomly/heuristically initialized and then healed end-to-end by full distillation |

Every method that fits a feature map to reproduce softmax attention output does so by gradient descent on a
regression-shaped loss (MSE or KL-divergence between linear-attention output and softmax-attention output,
computed per head/layer on calibration data). Mechanically, several of these losses (LoLCATs' Eq. 6 MSE,
MOHAWK's Frobenius-distance matrix-orientation stage) are literally ordinary least-squares problems in the
feature-map's own linear parameters — Hedgehog and LoLCATs' `φ(x) = f(Wx+b)` is only nonlinear through the
activation `f`; if `f` were dropped (or fixed and inverted), the remaining `W, b` fit would be a textbook
linear least-squares problem with a closed-form normal-equations solution. **No paper checked in this pass
takes that step.** This is consistent with what the brief itself anticipated and with the prior pass's
finding on MOHAWK: **nobody in the literature surveyed has published a closed-form (single matrix-solve, no
iterative gradient descent) per-layer fit of a linearised-attention feature map against a donor's
calibration-time attention outputs.** This is the second verified gap the brief asked to be confirmed,
sitting next to the "closed-form/local-solve only, no healing" gap the prior pass already identified for
MOHAWK Stage 1. Whether such a closed-form fit would work is not addressed by anything found this pass —
this is a gap in what has been tried, not evidence about what would happen if tried.

## 8. Ranked table (brief §6)

Ranking is by training-token cost, ascending (cheapest first), since that is the dominant driver of 2×T4
feasibility across every method checked. Fixed-state is the hard filter from brief §3.2 — every method
below passes it (all use pure kernel-linear-attention recurrences, with or without a small bounded window),
consistent with §0's framing check being confirmed rather than falsified this pass (see §9 below).

| Method | Fixed-state? | Training tokens | My 2×T4-h est., 1.5B (mine, assumptions in-section) | My 2×T4-h est., 100B (mine) | Quality retention (exact cell) | Donor Q/K/V/O reused? |
|---|---|---|---|---|---|---|
| LoLCATs | Yes (linear + 64-tok window) | 40-60M (Table 8/9/10) | approx. 15-16h (scaled from 6.5 GPU-h/A100 at 7-8B; VRAM fit at 1.5B on T4 not verified) | approx. 3,700-4,000h (scaled from 405B's 454 H100-GPU-h; no run below 405B exists to anchor) | 8B MMLU 54.88 vs 66.11 (-11.2); 70B 67.70 vs 78.80 (-11.1); 405B 72.20 vs 82.98 (-10.8) (Table 7) | Yes — frozen then LoRA-perturbed |
| DiJiang | Yes | 40B (7B flagship, Table 3) | not estimated — no throughput/day figure at 7B [X] | not estimated [X] | 7B: 11-task avg 0.557 vs 0.565 (-0.008, but individual tasks swing up to 13.9 pts) (Table 3) | Not stated — appears to be full fine-tune, not LoRA |
| MOHAWK (0% attn) | Yes (genuine SSM) | 3.0B (per ADAPTER_MEMO_01 Table 1) | approx. 154h (per ADAPTER_MEMO_01, carried forward) | approx. 10,250h (same) | 62.6 vs teacher 64.9 (Table 1, per ADAPTER_MEMO_01) | No — shape-compatible remap only (Q to C, K to B, V to X), not a reuse of donor attention math |
| MOHAWK (hybrid) | Yes | 5.0B (abstract) | approx. 257h (linear scale from 3B/154h) — mine, unverified | approx. 17,083h — mine, unverified | Not located this pass or prior pass | No, same as above |
| Mamba-in-the-Llama (50% hybrid) | Yes | 20B (v4 abstract, confirmed this pass) | not independently estimated this pass (prior pass's 600-1,500 A100-GPU-h range, divided by 2x~15 T4 slowdown factor gives a rough 4,500-11,250 2×T4-h) | not estimated — no run above 8B published | AVG (10-task) 61.30 (Mamba-Llama3) / 63.84 (Mamba2-Llama3) (Table 3), vs no matched same-architecture teacher row in this table | Yes — shape-compatible (Q to C, K to B, V to X), same caveat as above: dimensionally exact, not mathematically equivalent |
| Hedgehog | Yes (no window) | Not stated [X] | [X] — no cost figure anywhere in the paper | [X] | GPT-2: 16.7 vs 15.8 PPL (finetuned baseline, -0.9); Llama-2 ROUGE-1 47.4 vs 51.1 LoRA-finetuned (-3.7) (Table 10/11) | Yes — frozen then LoRA (Llama-2); full fine-tune (GPT-2) |
| SUPRA | Yes (explicit, strongest wording) | 100B (Table 3; cross-confirmed via LoLCATs Table 4) | approx. 48,000h (mine, from measured 4300 tok/s/GPU throughput, H100 to T4 15x slowdown) | not estimated — no run above 7B | MMLU 34.2 vs 62.4 original (-28.2, largest gap in survey); most other tasks near-lossless (LoLCATs Table 4) | Full uptraining, all params trainable — not frozen/LoRA |

Reading the table: LoLCATs is the only method whose token budget plausibly fits a 90h/week 2×T4 quota at the
1.5B scale on my own (unverified) estimate; every other method's token budget is 1-3 orders of magnitude
larger. At 100B, my own extrapolations put every method well outside budget, LoLCATs included (by my
estimate, roughly 5-6x over even in its cheapest, most favorable extrapolation) — though LoLCATs' 100B
figure is the least-anchored number in this table (scaled from a 405B run, no 100B data point exists) and
should be treated with the least confidence of anything marked "mine" here.

## 9. Is the fixed-state framing (brief §1) correct?

Confirmed, not falsified, across every method checked this pass — with one important refinement. Every
donor-conversion and substrate method verified above genuinely reduces to an O(1)-per-token recurrent state
(explicit paper-level confirmation for SUPRA — "constant-size KV cache... inference cost constant in the
number of generated tokens" — RetNet, GLA, and by architectural construction for Hedgehog, MOHAWK, DiJiang,
Mamba-in-the-Llama). The refinement: several of the best-performing methods (LoLCATs, BASED) are not pure
linear attention — they are linear attention plus a small, bounded sliding window (64 tokens in LoLCATs'
case). This is still O(1) in T (the window is fixed-size, not growing), so the brief's core claim survives,
but a faithful reproduction in our engine needs to budget for that bounded window buffer, not just the
recurrent state — the state is O(d'·d_v + window·d), a slightly larger constant than "just" the SSM/linear-
attention state, not the pure single-state recurrence the brief's §1 describes. This is a detail, not a
correction to the load-bearing claim, but it should be named explicitly since the brief says a large amount
of programme reasoning rests on the fixed-state claim: the claim holds, with this one architectural nuance.

A second, more consequential point where the brief's framing may need adjustment: BASED's mechanism (§4
above) makes explicit why the fixed-state property that deletes the KV wall is the same property that
bounds recall capacity — a state of dimension d' can only hold O(d')-ish exact associative bindings before
saturating, independent of how well the feature map or kernel is fit. **This means the recall-quality cost
is not a training-recipe artifact that a better fitting procedure (closed-form or otherwise) could fully
eliminate — it is architectural.** A closed-form per-layer solve (§7, if one existed) would presumably fit
the feature map optimally given the state dimension, but would not remove the state-dimension ceiling
itself. If this project's engine needs long-context recall fidelity (not just throughput), the fixed-state
property that solves the KV-wall problem is the same property that would need to be worked around for
recall — the brief frames these as one connected finding; this pass's evidence agrees they are
mechanistically linked, but flags that "solves both at once" (brief §1) is true for the wall and only
partially true for quality, since the wall-solving mechanism is exactly what caps recall.

## 10. Single-source and unverified claims — full list

- LoLCATs' "approx. 40M total training tokens" headline vs Table 8's own cell arithmetic (60M for the 8B
  run) — both are table-sourced, they are not the same number; §2 above reports both without forcing a
  resolution.
- LoLCATs' 100B-donor cost estimate — extrapolated from a single 405B data point, no anchor below 405B; the
  largest-uncertainty number in the ranked table.
- Hedgehog's exact associative-recall (AR) numbers in Table 3 — table's existence and caption confirmed,
  exact cell values not transcribed this pass.
- MOHAWK's hybrid-model exact attention fraction — not pinned down across two independent passes now (this
  one and the prior one); genuinely appears to be a gap in the paper's own reporting, not a research failure
  on this project's part.
- MOHAWK Stage-1-only (no stages 2-3) quality — not located across three independent fetch attempts total
  (two in the prior pass, referenced again this pass); likely does not exist in published form.
- Mamba-in-the-Llama's exact wall-clock figure ("8 days on 8xA100 + 4 days on 8xH100") — carried forward
  from the prior pass's single v4 fetch, not independently re-verified this pass.
- DeltaNet's hybrid-ratio percentage — the "2 of N layers" fact is table-sourced, but N for the flagship
  1.3B/100B-token model was not located this pass, so no percentage can be honestly stated.
- BASED's "6.22 accuracy points" and "24x throughput" headline claims — abstract-level only, not yet
  verified against BASED's own results table this pass. Flagged for follow-up if load-bearing.
- RWKV — used only as a third-party comparison baseline inside SUPRA/DiJiang this pass; RWKV's own paper and
  tables were not independently fetched.
- Performer and CosFormer's own preferred numbers — both were checked only via two other papers' (Hedgehog's,
  DiJiang's) controlled re-implementations, not their own original papers' tables. Both Hedgehog and DiJiang
  have an interest in these baselines looking weak; this is not evidence of manipulation, but it is a
  structural reason to treat the exact magnitude with caution while treating the qualitative direction (both
  underperform softmax-mimicry-trained kernels) as reasonably robust across two independent labs.
- Hedgehog's own training-cost figures — genuinely absent from the paper, not a search failure; flagged as a
  pattern worth noting: not every paper in this literature reports what this project needs, and that itself
  is information.

## 11. Where I think the brief's framing might be wrong, or needs sharpening

1. "A successful closed-form linearisation... deletes the KV traffic wall entirely, and produces exactly the
   recurrent operator our C engine already has a scan for" (brief §1) — confirmed true for the pure
   linear-attention term in every method checked, but the best-performing published recipes (LoLCATs, BASED)
   are not pure linear attention, they are linear-attention-plus-bounded-window hybrids. If "exactly the
   recurrent operator our C engine already has a scan for" means a pure SSM/linear-recurrence scan with no
   auxiliary local-window buffer, the highest-quality published methods do not produce that exact operator —
   they produce that operator plus a small extra piece. This is a scope note the engine team should have,
   not a refutation.
2. The brief treats "closed-form" and "fixed-state" as largely independent axes to search separately (§3 vs
   §4b) — this pass found they are not independent in practice: every fixed-state method checked is fit by
   gradient descent, and no method's own paper offers a closed-form alternative. This isn't a framing error
   exactly, but the brief's expectation that these might be found as separate results in the literature was
   not borne out — they travel together in every paper found.
3. On §4(a): the brief hoped for "two independent constraints agree" as a possible major finding. This pass
   did not find that. It found something more useful but less clean: the literature closest to our exact
   task (donor distillation) has not tested near our ratio, and the literature that has tested near our
   ratio is not doing our exact task. This is worth naming explicitly rather than letting a comforting
   reading of Zoology's 6.3% number stand in for a donor-distillation result it is not.
4. One number worth flagging even though it wasn't asked for directly: SUPRA's "5% of the training cost"
   framing (relative to full pretraining) sits right next to this project's own "budget = 90 GPU-h/week"
   framing (also a relative-sounding number). Both are true statements that describe wildly different
   absolute costs depending on what they're relative to. This project should be alert to the same rhetorical
   move happening in its own documents — "X% of pretraining cost" and "affordable on our budget" are not the
   same claim, and this survey's own ranked table (§8) shows SUPRA is simultaneously "only 5% of pretraining
   cost" and the single most expensive method found, in absolute 2×T4-hours.



---

# §5 — Novelty check on the R2 construction (the V-side closed-form solve)

**Researcher pass, 2026-08-22.** Answers the narrow question posed by `BRIEF_R2_CLOSED_FORM_VALUE_SOLVE.md`
§4.3 and Amendment 1: **has anyone published this specific construction** — fixed feature map `φ`, donor
`W_q`/`W_k` reused unchanged, `W_v` solved in closed form against the donor's own attention outputs?

**This section is written incrementally and was begun after one session-limit kill. Anything marked
`[PENDING]` was not reached.**

## 5.0 Headline, stated plainly and in the three-way vocabulary the brief asked for

| sub-question | verdict |
|---|---|
| **(i) Is the commuting step (§1.2) novel?** | **ANTICIPATED — it is published, explicitly, and it is folklore.** Elhage et al. 2021. Citation and verbatim quote in §5.1. |
| **(ii) Has anyone solved `W_v` in closed form against the donor's attention outputs?** | **ABSENT for the full matrix solve. ADJACENT twice, and one of the adjacents is closer than anything §4(b) found.** See §5.2, §5.3. |
| **(iii) Does any linearisation paper solve *any* parameter in closed form?** | **NO LONGER ABSENT — §7's "no closed-form solve exists at all" is now falsified.** Two papers do. See §5.2, §5.3. |
| **(iv) Does any paper consider and reject solving `W_v`?** | **`[X]` NOT FOUND — closed in §5.9.** No paper argues against it. LoLCATs is the nearest miss: it *freezes* `W_v` by explicit design choice, and states a reason that is about training cost, not about the solve. |
| **(vi) Which feature map does Taylor-Calibrate use? (asked 2026-08-22 after R2a's signal)** | **NONE — it has no `φ`.** The Taylor expansion is an analysis lens for deriving calibration constants; the student is a **gated delta-rule recurrence (GDN)** with per-head-normalised `q`/`k` and a learned decay. **§5.10 — and it reframes the line: three of their four calibrated quantities shape the MIXING, only one touches the value path.** |
| **(v) Is the commuting step wrong or narrower than stated?** | **No contrary evidence found.** Nothing located contradicts the Controller's `5.3e-15` numerical check or the `transformers` 4.57.6 source reading. The literature's own statement of the identity (§5.1) *agrees* with it. |

> **The one-line answer to the Adapter's three-way question: the commuting observation is published
> folklore, the full closed-form `W_v` solve is not anticipated, and nothing contradicts the algebra.**

## 5.1 The commuting observation is stated explicitly, in a widely-read source, in 2021 — [T, verbatim]

**Source:** Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, DasSarma, Drain,
Ganguli, Hatfield-Dodds, Hernandez, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan,
McCandlish, Olah — **"A Mathematical Framework for Transformer Circuits"**, Anthropic, *Transformer
Circuits Thread*, December 2021. `https://transformer-circuits.pub/2021/framework/index.html`

Retrieved and read from the page's own HTML source (not a summary), in the tensor-product derivation of
the attention head. **Verbatim, in the page's own LaTeX:**

```
h(x)  =  (Id ⊗ W_O) · (A ⊗ Id) · (Id ⊗ W_V) · x
```

with the page's own figure captions on each factor, verbatim:

- `(Id ⊗ W_V)` — *"Compute value vector for each token* `(v_i = W_V x_i)`*"*
- `(A ⊗ Id)` — *"Mix value vectors **across** tokens to compute result vectors* `(r_i = Σ_j A_{i,j} v_j)`*"*
- `(Id ⊗ W_O)` — *"Project result vectors out for each token* `(h(x)_i = W_O r_i)`*"*

and then, verbatim: *"Applying the mixed product property and collapsing identities yields:"*

```
h(x)  =  (A ⊗ W_O W_V) · x
```

caption, verbatim: *"`A` mixes across tokens while `W_O W_V` acts on each vector independently."*

**And the commuting claim itself, stated in as many words**, from the paper's tensor/Kronecker notation
explanation, **verbatim:**

> *"A product like `A ⊗ W` multiplies the vector at each position by `W` and across positions with `A`.
> **It doesn't matter which order you do this in.**"*

together with the mixed-product property, also stated verbatim on the same page:
`(A ⊗ B) · (C ⊗ D) = (AC) ⊗ (BD)`, and the equivalent form `(A ⊗ W) x = A x Wᵀ`.

> **That sentence is exactly §1.2 of the brief.** The mixing acts on the token index, `W_v` acts on the
> channel index, the two indices are independent, so the order is free. The brief's
> `out_t = W_v · (Σ_s A[t,s] x_s)` is the `A ⊗ W_V` identity read right-to-left.

### What this does and does not cost us

- **It costs the novelty of the observation.** The commuting step must be **cited, not claimed**. It is
  five years old, it is in one of the most-read papers in mechanistic interpretability, and the brief's
  framing of it as *"the entire trick"* would not survive a reviewer who knows this page. The
  mixed-product property it rests on is standard multilinear algebra.
- **It costs nothing of the algebra.** Elhage et al. state the identity and use it for *interpretation*
  (path expansion, `W_OV` circuits, virtual heads, "attention is a generalisation of convolution").
  **They never solve for `W_V`, never fit anything, never substitute a different `A`, and never touch
  calibration data.** They factor a fixed model; R2 re-fits a matrix under a *changed* `A`. The identity
  is the shared premise, not a shared method.
- **It is, if anything, corroboration.** The Controller verified the step numerically at `5.3e-15` and
  against `transformers` 4.57.6 source. The literature states the same thing symbolically. **Two
  independent confirmations of the same step from different directions.** Nothing found this pass
  narrows it or contradicts it. Specifically: no source found claims RoPE, RMSNorm, masking, the softmax
  denominator or GQA breaks it — and Elhage's derivation is for a plain head, so it does not *by itself*
  cover those; **the `transformers`-source check remains the load-bearing evidence for the modern
  architecture, and the literature only covers the idealised case.** That asymmetry is worth keeping in
  view: the published statement is weaker than the Controller's check, not stronger.

> **Recording the general point, because this project has been burned in the other direction:** the brief
> called §1.2 *"the entire trick, and it is why the problem is closed-form."* That is true of the
> *problem structure* and false as a claim of *originality*. **The novelty, if any, lives in §1.3 —
> solving `W_v` against a substituted `A` — not in §1.2.**

## 5.2 ADJACENT #1 — Neural Block Linearization (NBL): a closed-form LMMSE solve that replaces attention

**Source:** *"Efficient Large Language Model Inference with Neural Block Linearization"*, arXiv:2505.21077
(v2 read, arXiv HTML).

**This falsifies §7's blanket statement that no closed-form solve exists in this literature.** One does.

### Mechanism [T, Proposition 3.1 + Eq. 2]

For the `k`-th attention layer, take calibration inputs `X_k` and the layer's own outputs `Y_k`, and
replace the entire block with the **Linear Minimum Mean Squared Error** estimator:

```
Ŷ = W X + b                                     (Eq. 2)
W = C_YX · C_XX^{-1} ,   b = E[Y] − W·E[X]      (Proposition 3.1)
```

`C_XX` = input covariance, `C_YX` = output/input cross-covariance. **That is a normal equation on
calibration statistics: closed-form, no gradient descent, no fine-tuning.** Layer selection is by a
Canonical-Correlation-Analysis upper bound on the approximation error `[A]`.

### How it differs from R2 — and the difference is the whole point

| | NBL | R2 |
|---|---|---|
| what is replaced | **the entire attention block** | the softmax mixing only |
| what survives | **nothing** — `W_q`,`W_k`,`W_v`,`W_o` all discarded | `W_q`,`W_k`,`W_o` reused; `W_v` re-solved |
| replacement operator | a **token-wise linear map** `Ŷ_t = W x_t + b` — **no mixing across tokens at all** | a linear-attention recurrence — mixing retained, `O(1)` state |
| what is solved for | the replacement map `W` | the *value projection* under a substituted mixing |
| sequence dependence | **removed entirely** | retained, in recurrent form |

> **NBL deletes the token mixing; R2 substitutes it.** An NBL-converted layer cannot move information
> between positions at all — it is the `Id ⊗ W` term in Elhage's decomposition with the `A ⊗ …` term
> thrown away. That is why NBL is applied only to **a minority of blocks** and is reported as a
> speed/sparsity method, not as a linearisation method.

### Results [T?, Table 1 — transcribed as read, flagged]

| Method | Sparsity | Avg Score | Prefill | Throughput |
|---|---|---|---|---|
| Baseline | 0% | 70.2 | 1.00 | 1.00 |
| Attn NBL-8 | 2.9% | 70.0 | 1.17 | 1.20 |
| Attn NBL-12 | 4.3% | 68.3 | 1.28 | 1.27 |
| Attn DROP-8 | 4.8% | 69.4 | 1.18 | 1.22 |

> **⚠ Transcription caveat, stated because the project's rule demands it.** These cells came back through
> a single fetch of the arXiv HTML and **were not re-verified against the raw table markup in a second
> independent pass.** The model identity (reported as Mistral-7B) and the column semantics (a "sparsity"
> of 2.9% for 8 replaced attention blocks) are **not** independently confirmed. Treat the table as
> `[T?]`. **The load-bearing finding of §5.2 is the mechanism, which is `[T, Proposition 3.1]` and does
> not depend on these numbers.**

**The decision-relevant reading, independent of the numbers:** the closest published closed-form
conversion of attention **had to throw away the token mixing to get its closed form**. NBL is evidence
that the *solve-in-closed-form-against-calibration-outputs* move is established and publishable in this
exact setting — which lowers the novelty bar R2 clears and raises the prior that the move is sane.
**It is not evidence that R2's harder version works.**

## 5.3 ADJACENT #2 — Taylor-Calibrate: an OLS solve on the value projection, with `W_q`/`W_k`/`W_o` copied unchanged

**This is the closest match found to R2, and it is close enough that the Adapter needs to read it.**

**Source:** Zhou, Wu, Wang, Mishra, Song, Athiwaratkun, Xu — *"Taylor-Calibrate: Principled Initialization
for Hybrid Linear Attention Distillation"*, arXiv:2606.16429v1 (June 2026), arXiv HTML.

### What it does [T, Eq. 8/9/36/42 as read]

Converts a pretrained Transformer to a **hybrid Gated DeltaNet** student. **Phase 1 is analytical and
training-free**, setting four GDN parameter groups from teacher attention statistics; **Phase 2 is a
brief per-layer gradient alignment**, followed by full-model recovery training.

Phase-1 assignments as read:

```
Eq. 8   dt_bias,h := softplus^{-1}( ln2 / d_h )                 d_h = attention-weighted mean look-back distance
Eq. 9   (b_proj)_h,: ← [ logit(β*_h) / ū_h ] · (b_proj)_h,:     β*_h = 0.3 + 0.4·c_h  (entropy concentration)
Eq. 36  σ*_h = Σ(y_T · y_S) / Σ(y_S²)  ;   W_V^{(h)} ← σ*_h · W_V^{(h)}
Eq. 42  α*_ℓ = λ · RMS(y_T^{(ℓ)}) / RMS(SiLU(W_V x)) ,  λ = 0.01
```

**Eq. 36 is the one that matters.** `y_T` is the *teacher's* attention output, `y_S` the *student's*
linear-attention output on the same calibration data, and the paper's own words for `σ*_h` are **"the
standard one-dimensional ordinary least-squares solution"** matching teacher outputs. **`W_V` is then
rescaled by it.** Q, K and O projections are **copied unchanged from the teacher** `[A]`.

### The exact difference from R2 — stated precisely, because this is the near-miss

**Same setup. Degenerate solve.**

| | Taylor-Calibrate Eq. 36 | R2 §1.3 / A1.1 |
|---|---|---|
| target | donor's own attention output `y_T` on calibration data | **same** |
| student mixing | fixed linear attention (GDN), teacher `W_q`/`W_k` reused | **same in kind** |
| unknown | **one scalar per head**, `σ*_h ∈ ℝ` | **the full matrix** `W_v ∈ ℝ^{d_v × D}` |
| solve | 1-D OLS, `Σ(y_T·y_S)/Σ(y_S²)` | `d_v × D` ridge normal equation `(W_v^donor(Z*Zᵀ+λI))(ZZᵀ+λI)^{-1}` |
| degrees of freedom | `H` (≈64) per layer | `d_v × D` (≈8.4 M per KV head-group at donor width) |
| is the solve the endpoint? | **No — it is an initializer.** Phase 2 gradient alignment follows, then recovery training | **Yes — R2 claims the solve *is* the method, no gradients at all** |
| needs the commuting step? | **No** — a scalar commutes with anything trivially | load-bearing |

> **So: the *idea* of closed-form-calibrating the value side against the teacher's attention outputs,
> with Q/K/O frozen, is published as of June 2026. What is published is a single scalar per head, used
> as an initializer for a distillation run — not a matrix solve, and not training-free end-to-end.**

**Verdict on (ii): ADJACENT, not anticipated.** The gap between "one scalar per head" and "the full
`d_v × D` matrix under a substituted mixing" is the entire content of R2 §1.2–1.3: a scalar rescale does
not need the commuting identity, does not need `Z = X A_linᵀ`, and **cannot compensate for a wrong mixing
*pattern* — only for a wrong output *magnitude*.** **But the Adapter should stop describing the setup as
unexplored.** It is explored; what is unexplored is taking the solve to full rank and making it the
endpoint rather than the initializer.

### The part of Taylor-Calibrate that is decision-relevant beyond novelty

Their Phase 1 exists **because a good initialization buys training tokens** — the paper's headline is
`4.9×–9.2×` fewer recovery tokens and "up to 88×" zero-shot quality at initialization `[A, abstract]`.
**They did not claim it removes the need for training.** That is the direct empirical counterweight to
R2's ambition: **the one team that has published a closed-form value-side calibration treats it as an
initializer worth ~5–9× of training, not as a replacement for training.** `[A]` — abstract-level, **not
yet read in their own table. Flagged: if R2 proceeds, this is the single most important number in this
section to pin down**, because it is the closest thing to a prior on what a value-side closed-form
calibration actually buys.

## 5.4 What §7's "CONFIRMED GAP" should now say

§7 (prior pass) states: *"nobody in the literature surveyed has published a closed-form (single
matrix-solve, no iterative gradient descent) per-layer fit."* **Amend, do not delete:**

- **NBL (2505.21077) publishes exactly such a solve** — closed-form, per-layer, against calibration
  activations — but for a **token-wise linear map replacing the whole block**, not for a linearised
  attention operator.
- **Taylor-Calibrate (2606.16429) publishes a closed-form per-head OLS on the value projection**, in the
  exact frozen-Q/K/O setting, as an initializer.
- **Neither is a closed-form fit of a linearised-attention operator's projections that is also the
  endpoint.** That specific combination remains **ABSENT** after this pass.

> The gap is **narrower and better-defined than §7 claimed, and §7's phrasing was too strong.** The
> honest statement is: *closed-form solves against calibration activations are established practice in
> this literature; what is unpublished is doing it at full matrix rank on `W_v` under a substituted
> mixing, with no gradient step afterwards.*

## 5.5 `[PENDING]` — remainder of the Question-1 sweep

> **STATUS UPDATE (end of pass): items 1 and 3 below are now CLOSED in §5.9** (the consider-and-reject
> question and the PTQ-crossover question, both `[X] not found`). **Liger remains the one genuinely
> unread paper that belongs in §8'''s ranked table.** The list below is kept as written for the record.

Not reached before this section was written down. Listed so a later pass knows the state:

- Explicit **consider-and-reject** of a `W_v` solve in Hedgehog / LoLCATs / SUPRA / MOHAWK / Liger /
  Mamba-in-the-Llama — **not found**, and absence here is weak evidence: papers rarely publish what they
  rejected. `[X]`.
- **Liger (arXiv:2503.01496, ICML 2025)** — repurposes the pretrained **key** matrix via parameter-free
  pooling to build gates, "without introducing any learnable parameters", then LoRA fine-tunes; reported
  to recover **93% of Transformer performance at 0.02% of pretraining tokens** `[A, abstract — NOT read
  in its own table]`. **This is the closest published thing to "derive a linearisation parameter from an
  existing donor matrix by a fixed rule with no training"**, and the §8 ranked table currently omits it.
  Not fetched in full.
- The PTQ / layer-wise-reconstruction lineage (SparseGPT / GPTQ / OBC) used to **heal an architectural
  substitution** rather than quantisation or pruning: searched once, **nothing found** — searches return
  only quantisation and pruning uses. `[X]`, low confidence, single search.
- Surfaced but **not read**: `Exact Linear Attention` (2605.18848), `STILL` (2602.02180), `Retrofitting
  Linear Attention into Diffusion LMs` (2608.06628), `Effective Distillation to Hybrid xLSTM`
  (2603.15590), `What Matters in Linearizing Language Models?` (2504.14366), `LayerBoost` (2604.22050).

## 5.6 Single-source and unverified claims in this section

- **NBL Table 1 cells** — single fetch, not re-verified, model identity not independently confirmed.
  Marked `[T?]`. The mechanism claim does not depend on them.
- **Taylor-Calibrate Eq. 8/9/36/42** — single fetch of the HTML. The equation *forms* and the quoted
  phrase "the standard one-dimensional ordinary least-squares solution" are as read; **the equation
  numbers and the `λ = 0.01` constant are not re-verified.**
- **Taylor-Calibrate's `4.9×–9.2×` and `88×`** — abstract-level `[A]`, not read in the paper's own table.
  **Not admissible in a decision yet.**
- **Liger's 93% / 0.02%** — abstract-level `[A]`, from search-result text, **not fetched from the paper
  this pass.** Explicitly not admissible.
- **Elhage et al. quotes in §5.1** — the exception: read directly from the page's own HTML source,
  verbatim, with surrounding context, cross-checked at six separate occurrences of the notation.
  **High confidence.**

## 5.7 Verification pass on Taylor-Calibrate — upgrades §5.3, and finds the strongest counter-evidence against R2's "no training at all"

Re-read from the paper's own HTML source text (`arXiv:2606.16429v1 [cs.LG] 15 Jun 2026`, CC BY 4.0),
not from a summary. **This upgrades three `[A]`/`[T?]` tags from §5.3 and adds one finding that §5.3
did not have.**

**Authors, verified:** Zhongzhu Zhou (Univ. of Sydney / Together AI), Qingyang Wu (Together AI),
Junxiong Wang (Microsoft), Mayank Mishra (UC Berkeley), Shuaiwen Leon Song (Together AI),
Ben Athiwaratkun (Together AI), Chenfeng Xu (Together AI / UT Austin).
Code: `https://github.com/FutureMLS-Lab/Taylor-Calibrate`.
**Note the overlap with the LoLCATs / Mamba-in-the-Llama lineage (Junxiong Wang, Together AI)** — this is
not an outside group, it is the same community that produced the methods in §1b and §2.

### 5.7a Eq. (36) confirmed verbatim — [T], upgraded from [T?]

Appendix **A.5 is titled "Closed-Form Value-Side OLS Rescaling"** — the paper names the move exactly as
§5.3 characterised it. Verbatim, Eq. (35)→(36):

> `dJ/dσ = −2 Σ_i u_i v_i + 2σ Σ_i v_i² = 0,`  (35)  *"so, provided the student context is not
> identically zero,"*
>
> `σ*_h = (Σ_i u_i v_i)/(Σ_i v_i²) = (Σ_{b,t,d} y_{T,b,t,h,d} · y_{S,b,t,h,d}) / (Σ_{b,t,d} y_{S,b,t,h,d}²).`  (36)
>
> *"**This is the standard one-dimensional ordinary least-squares solution** [5]: it rescales the student
> context along its current direction so that it best matches the teacher in squared error. The clipping
> step mentioned in the main text is an implementation safeguard applied after this closed-form solution
> is computed."*

**Confirmed: one scalar per head, `σ*_h ∈ ℝ`, closed-form, plus a clipping safeguard.** §5.3's
characterisation stands exactly.

### 5.7b Donor projection reuse confirmed verbatim — [T], upgraded from [A]

From §3.1 (Problem Setup), verbatim: students are given *"the same teacher `W_Q, W_K, W_V, W_O`, their
initial PPL and task accuracy can vary sharply depending on how the newly introduced GDN gates are set.
**The missing transfer information is therefore in the recurrent dynamics: memory timescale, write
strength, and output gating.**"*

And from the abstract, verbatim: *"**Simply copying the teacher attention projections into a Gated
DeltaNet (GDN) student does not specify the new recurrent decay, write, and output-gating dynamics.**
As a result, the converted model often starts in a poor dynamical regime and must spend many
distillation tokens repairing initialization rather than learning the remaining teacher behavior."*

> **This confirms §2.2f of `ADAPTER_MEMO_01` from a second, independent direction: conversion reuses all
> four donor projections.** Taylor-Calibrate's entire premise is that the projections carry over and only
> the *recurrent dynamics* are missing. **The attention weight stream is not removed by conversion.**

### 5.7c The authors state the limitation that defines the gap R2 would fill — [T, Appendix C]

Verbatim, from "Architecture specificity":

> *"**The value-side least-squares step should transfer broadly because it only matches output
> amplitude**, but the decay and gate calibration should be rederived for each architecture."*

> **The authors themselves scope their value-side solve to amplitude only.** That is the precise sense in
> which R2 is not anticipated: a scalar can fix magnitude, not the mixing pattern. **This is the cleanest
> possible statement of the gap, and it comes from the paper rather than from us — which is the strongest
> form this project accepts.**

### 5.7d ⚠ The counter-evidence, and it is the most decision-relevant thing in §5

Verbatim, Appendix C, "Downstream distillation dependence":

> *"**Taylor-Calibrate improves the starting point, but it does not remove the need for downstream
> distillation** [16, 13, 25]. **The converted student still has a different sequence mixer from the
> teacher, so global training is needed to repair cross-layer interactions and adapt the residual stream
> to the new recurrent blocks.** The final quality therefore depends on layer selection, retained-softmax
> budget, distillation loss, optimizer settings, and the number of recovery tokens."*

> **This is a direct, paper-stated argument against the R2 endpoint, from the team that has come closest
> to R2's construction.** Their stated mechanism for why is *not* "the value solve is too weak" — it is
> **cross-layer interaction and residual-stream drift**, which is a failure mode a per-layer solve of any
> rank cannot address by construction. R2 §1.3 optimises a **layer-local** objective; this paragraph says
> the layer-local objective is not the binding constraint.

**How much weight this deserves, stated honestly in both directions:**

- **It is an assertion in a limitations section, not a measurement.** They did not run a full-rank
  value-side solve and find it insufficient. **It is `[A]`-grade evidence about a claim they did not
  test.** It is not a refutation of R2 and must not be reported as one.
- **But it is the most informed prior available**, and it points the way R2's amendment A1.6 already
  worried about (layer-wise Frobenius error is blind to what breaks). It also matches D4's own
  in-house experience that layer-local reconstruction quality and end-to-end quality are different
  questions.
- **The pre-registered design in the brief cannot see this failure mode at all.** §3.4 measures
  layer-wise reconstruction only. **If R2's pass-1 returns a high recovery fraction, that is not yet
  evidence against this paragraph** — the two are measuring different things, and a strong §3.2 result
  would be exactly the "flattering" outcome the project's own rule says to scrutinise hardest.

### 5.7e Headline numbers — verbatim from the abstract, still [A]

> *"Across four teacher settings and three retained-layer policies, Taylor-Calibrate gives substantially
> stronger zero-shot students, **with up to an 88× improvement in a representative ablation**, and
> reaches matched recovery targets with **4.9×–9.2× fewer training tokens** than naive conversion."*

**Note the qualifier that the search-result paraphrase had dropped: "in a representative ablation."**
The 88× is a single ablation cell, not a headline result across settings. `[A]` — the paper's own
results tables (§4.2, §4.3, Appendix D.1) were **not** transcribed this pass. **These numbers remain
inadmissible in a decision until read in those tables.**

### 5.7f Two further items from Appendix C that bear on this programme

- **Head-wise vs layer-wise conversion is an open problem they name, not solve** — verbatim: *"Converting
  all heads in a selected layer to GDN therefore may be too coarse, especially for models whose
  long-context behavior depends on a small number of specialized attention heads. A natural next step is
  head-wise conversion: keep retrieval-heavy heads as softmax, convert more local or diffuse heads to
  GDN."* **This is independent support for R2 Amendment A1.6's entropy-stratification requirement**, and
  it suggests the retained-attention budget could be spent per-head rather than per-layer — which is a
  lever `ADAPTER_MEMO_01` has not costed.
- **They propose extending the closed form — but to the gates, not to `W_v`** — verbatim: *"Future work
  could replace these hand-designed maps with a **constrained matching objective that directly solves for
  decay and gate parameters** from teacher statistics."* **So the "solve more of it in closed form"
  direction is explicitly on their roadmap.** R2's specific target (full-rank `W_v`) is not named there;
  but the general direction is claimed as future work by an active group with the code released. **This
  is a novelty-window risk worth naming: the adjacent group is moving toward this area.**

### 5.7g Net effect on §5.0's verdict

**Unchanged: ADJACENT, not anticipated.** The verification made the adjacency *closer* (the paper names
its own move "Closed-Form Value-Side OLS Rescaling") and simultaneously made the remaining gap *sharper*
(the authors scope it to amplitude only, in their own words). **The novelty claim R2 can defend is
narrow and specific:** full-rank `W_v` under a substituted mixing, as the endpoint. **The novelty claim
R2 cannot defend is any of: the commuting step, the frozen-Q/K/O setup, closed-form calibration against
teacher attention outputs, or "nobody solves anything in closed form."**

## 5.8 Taylor-Calibrate — full table transcription, and what the two headline numbers actually are

**Commissioned as the top priority after §5.7.** Every table below is parsed from the raw `<table>` markup
of `arXiv:2606.16429v1` and **transcribed verbatim before any arithmetic was performed on it.** Where I
compute, the computation is shown and labelled as mine.

### 5.8a Setup, so the tables cannot be misread [T, §4.1 verbatim]

**Four teachers:** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Qwen3-8B.

**Five initialization arms, verbatim:**
> *"(i) **Baseline** — copy teacher `Q/K/V/O` projections, leave new recurrent parameters at their random
> default; (ii) **Zero-Gate** — same projection copy, set output gates `g_proj = 0` so the recurrent
> branch contributes nothing at step 0; (iii) **Small-Gate** — set `g_proj = ε` with `ε = 0.01`;
> (iv) **Taylor-Only** — Phase 1 analytical calibration (Section 3) without per-layer alignment; and
> (v) **Taylor-Calibrate** — full Phase 1 + Phase 2."*

**Crucial for us, verbatim:** *"All variants use the same projection-transfer backbone: **we first copy
the teacher `Q/K/V/O` weights** following RADLADS [13], which is also the conversion pipeline used by
GA-S2 after layer selection [25]."*

> **Third independent confirmation of `ADAPTER_MEMO_01` §2.2f.** Every arm in this paper, including the
> baselines and the cited RADLADS/GA-S2 pipelines, **keeps all four donor projections.** The attention
> weight stream is not removed by conversion, in any method this paper compares against.

**"Budget" = retained softmax attention.** Verbatim: *"Table 2 summarizes zero-shot Avg and RULER for
the Uniform policy across **25%, 50%, and 75% retained attention**."*
**`Avg` = macro average over twelve downstream tasks, excluding PPL and RULER** (ARC-C/E, HellaSwag,
PIQA, MMLU, OBQA, ReArc, WinoGrande, BoolQ, LAMBADA, COPA, SciQ) `[T, §4.1]`.
**`RULER` = aggregate over 13 long-context subtasks** (8 NIAH variants + `ruler_vt`, `ruler_cwe`,
`ruler_fwe`, `ruler_qa_squad`, `ruler_qa_hotpot`) `[T, §4.1]`.

### 5.8b ⚠ The 88× is a ratio between two broken models — Table 4, transcribed verbatim [T]

**Table 4 caption, verbatim:** *"Component ablation on Qwen2.5-3B-Instruct (Uniform). Avg denotes the
short-context NLU average from this ablation sweep; MMLU is unavailable in this run."*

| Variant | PPL ↓ | Avg | Description |
|---|---|---|---|
| Baseline | 37337.3 | 30.9 | projection copy only |
| Zero-Gate | 22469.0 | 30.7 | output-path stabilization |
| Taylor-Only | 22470.6 | 30.9 | Phase 1 only |
| Alignment-Only | 2015.9 | 31.4 | Phase 2 only, no analytical calibration |
| Taylor-Calibrate | 424.1 | 32.1 | Phase 1 + Phase 2 |

**The Adapter asked: name the cell, its headers, and what the other cells say. Done above. Now the
reading, and it is not the reading the abstract invites.**

**Mine, computed from the two cells:** `37337.3 / 424.1 = 88.04`. **That is the 88×.** Row `Baseline`
vs row `Taylor-Calibrate`, column `PPL ↓`. Confirmed by the paper's own conclusion, verbatim:
*"improves the **worst initial PPL** by 88×"* — **the paper itself scopes it to the worst case.**

> **⚠ The 88× is a ratio between a perplexity of 37,337 and a perplexity of 424.** Both are models that
> do not work. The teacher for this row is Qwen2.5-3B, whose `Avg` is **67.3** `[T, Table 2]`. **Every
> variant in Table 4 sits at `Avg` 30.7–32.1** — i.e. the 88× perplexity improvement corresponds to
> **+1.2 `Avg` points, from 30.9 to 32.1, against a teacher at 67.3.**
>
> **The typical cell in this table is not 2×. There is no cell that is 2×. The table's whole dynamic
> range is between two failure modes, and the `Avg` column shows the 88× buys almost nothing that a
> downstream task can see.** The honest one-line summary of Table 4 is: *at initialization, before any
> recovery training, every variant is broken; Taylor-Calibrate is less numerically broken.*

**And the ablation the Adapter asked for is here in negative form.** `Zero-Gate` = 22469.0 and
`Taylor-Only` = 22470.6. **Phase 1's entire analytical calibration — including the value-side OLS — is
indistinguishable from setting the output gate to zero, and is in fact 1.6 PPL worse.** The paper says
so itself, verbatim:

> *"**Phase 2 alignment is essential.** Baseline starts at PPL 37337.3, while Zero-Gate and Taylor-Only
> remain around 22470, so **Phase 1 alone is not enough.**"*

### 5.8c ⚠⚠ The answer to "is there an isolated solve-vs-copy `W_v` ablation?" — and it is the most important cell in the paper for R2

**Direct answer: NO isolated ablation of the value solve exists. `[X]`.** Table 4's `Taylor-Only` row
bundles **all four** Phase-1 assignments (decay bias, write gate, value OLS, output gate). There is no
row that varies only `W_V ← σ*_h W_V` with everything else held fixed.

**But the bundle's result is itself decision-relevant, and it points against R2's premise:**

| comparison | PPL | mine: ratio |
|---|---|---|
| Baseline → Zero-Gate (a one-line heuristic, no calibration at all) | 37337.3 → 22469.0 | 1.66× |
| Zero-Gate → Taylor-Only (**adds the entire Phase-1 calibration, value OLS included**) | 22469.0 → 22470.6 | **0.9999× — nothing** |
| Baseline → Alignment-Only (**gradient descent, no analytical calibration at all**) | 37337.3 → 2015.9 | 18.5× |
| Alignment-Only → Taylor-Calibrate (adds Phase 1 on top of Phase 2) | 2015.9 → 424.1 | 4.75× |

> **Read the second and third rows together.** The **closed-form analytical calibration contributes
> nothing on its own** (row 2), while **a short gradient-based alignment contributes 18.5× on its own**
> (row 3). The closed form only pays once gradients have run (row 4).
>
> **This is the single most decision-relevant set of cells found for `BRIEF_R2`.** R2's claim is that a
> closed-form solve **is the endpoint**. The one paper that has published a closed-form value-side
> calibration measured its own closed-form stage in isolation and got **zero**, and measured the gradient
> stage in isolation and got **18.5×**.

**Now the argument in the other direction, because it is real and must be stated with the same force:**

- **Their closed form is a scalar per head; R2's is a full `d_v × D` matrix.** A scalar has ~64 degrees
  of freedom per layer against ~8.4 M. **It is entirely coherent that a scalar does nothing and a full
  matrix does a great deal.** Table 4 is *not* a measurement of R2; it is a measurement of the weakest
  possible version of R2.
- **The bundled row hides which component did what.** `Taylor-Only` also sets the decay bias and write
  gate; a bad setting there could mask a good value solve. **Nothing in the table separates them.**
- **The metric is initialization PPL on a fully-converted hybrid**, not the layer-wise reconstruction
  error R2's §3.4 measures. **These are different quantities and the mapping between them is exactly
  what nobody has published.**

> **Net: this is `[T]`-grade evidence that the *weak* form of R2's move is worthless alone, and it is
> `[X]` on whether the *strong* form is. It does not refute R2. It does establish that "a closed-form
> value-side calibration" is not, by itself, known to be worth anything — and R2's pre-registration
> should carry that as its prior rather than the optimistic one.**

### 5.8d Zero-shot quality — Table 2, transcribed verbatim [T]

**Caption, verbatim:** *"Compact zero-shot summary for the Uniform retained-layer policy. Each entry
reports `Avg/RULER`; full task-level results and non-uniform layer-selection policies are reported in
Appendix D.1. **Bold**: best non-teacher value within each model and budget row."*

| Model | Budget | Teacher | Baseline | Zero-Gate | Small-Gate | Taylor-Only | Taylor-Calibrate |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 25% | 64.8 / 86.2 | 31.2 / 0.2 | 31.3 / 0.2 | 31.4 / 0.2 | 31.9 / 0.2 | 34.8 / 0.2 |
| | 50% | 64.8 / 86.2 | 31.7 / 0.1 | 32.7 / 0.1 | 32.7 / 0.1 | 32.6 / 0.1 | 37.4 / 0.1 |
| | 75% | 64.8 / 86.2 | 48.6 / 2.7 | 53.9 / 10.1 | 53.7 / 10.3 | 53.9 / 10.1 | 56.2 / 14.5 |
| Qwen2.5-3B | 25% | 67.3 / 91.3 | 31.1 / 0.0 | 31.8 / 0.0 | 31.8 / 0.0 | 31.7 / 0.0 | 30.9 / 0.1 |
| | 50% | 67.3 / 91.3 | 31.3 / 0.0 | 31.3 / 0.0 | 31.4 / 0.0 | 31.3 / 0.0 | 36.3 / 0.4 |
| | 75% | 67.3 / 91.3 | 55.1 / 19.9 | 61.8 / 47.4 | 62.0 / 47.2 | 61.9 / 47.2 | 61.3 / 55.9 |
| Llama-3.2-3B | 25% | 65.6 / 89.6 | 31.2 / 0.0 | 33.1 / 0.0 | 33.2 / 0.1 | 33.1 / 0.1 | 34.2 / 0.1 |
| | 50% | 65.6 / 89.6 | 31.7 / 0.1 | 36.9 / 0.2 | 36.7 / 0.2 | 36.8 / 0.2 | 43.3 / 0.4 |
| | 75% | 65.6 / 89.6 | 33.9 / 0.2 | 58.3 / 31.0 | 58.6 / 30.9 | 58.5 / 30.9 | 60.0 / 24.0 |
| Qwen3-8B | 25% | 70.7 / 94.0 | 31.4 / 0.1 | 34.2 / 0.1 | 34.1 / 0.0 | 34.2 / 0.1 | 35.6 / 0.1 |
| | 50% | 70.7 / 94.0 | 36.2 / 0.2 | 45.1 / 0.3 | 44.9 / 0.3 | 45.0 / 0.3 | 49.0 / 0.7 |
| | 75% | 70.7 / 94.0 | 59.9 / 42.3 | 64.8 / 61.6 | 64.7 / 61.4 | 64.8 / 61.6 | 65.5 / 65.1 |

**Reported separately as the rules require — perplexity/NLU vs long-context recall:**

**Short-context (`Avg`):** at **75 % retained**, Taylor-Calibrate lands within 3.7–8.6 points of teacher
(e.g. Qwen3-8B `65.5` vs `70.7`). At **50 % retained** it is 21–34 points below teacher. At **25 %
retained** every model is at 30.9–35.6 against teachers of 64.8–70.7 — **chance-level or near it, for
every arm including the best.**

**Long-context (`RULER`) — and this is the number `ADAPTER_MEMO_01` needs:**

> **At 25 % retained attention, RULER is 0.1–0.2 against teachers at 86.2–94.0. At 50 % retained it is
> 0.1–0.7. Long-context retrieval is not degraded — it is at zero.**

Even at **75 % retained**, RULER is 14.5 / 55.9 / 24.0 / 65.1 against 86.2 / 91.3 / 89.6 / 94.0. **The
best case retains 69 % of teacher RULER while retaining 75 % of the attention layers.**

The paper's own reading, verbatim: *"**aggressive conversion remains difficult at initialization**: with
only 25% or 50% retained attention, all methods have very low RULER scores and the Baseline Avg is close
to chance-level for several models"*, and *"layer-local alignment mainly improves the initial
short-context Avg of the converted student, whereas **long-context retrieval still requires retained
attention or later recovery training**."

> **⚠ Direct consequence for `ADAPTER_MEMO_01` §2.2c, which needs attention retained on ≤ 8.3 % of
> layers.** This paper's **most aggressive setting is 25 % retained** — three times more attention than
> our budget allows — and at that setting long-context retrieval is **zero at initialization** across
> four donors and five methods. **This is a second, independent literature floor sitting well above our
> target, alongside Mamba-in-the-Llama's 12.5 %.** It is measured at initialization only; §5.8e shows
> what recovery training buys back.

### 5.8e Recovery — Table 3, transcribed verbatim [T]

**Caption, verbatim:** *"Compact recovery summary for all teacher settings. Each entry reports
`Avg/RULER`; detailed task-level metrics are reported in Appendix D.1. **Bold**: better value between
Baseline and Taylor-Calibrate within each checkpoint."*

| Model | Selection | Teacher | 100M Baseline | 100M Taylor-Cal. | 700M Baseline | 700M Taylor-Cal. |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | Uniform | 64.8 / 86.2 | 37.2 / 0.6 | 62.6 / 7.7 | 61.4 / 39.9 | 64.4 / 59.8 |
| | AR (LM-PPL) | 64.8 / 86.2 | 44.7 / 1.0 | 61.1 / 5.0 | 61.4 / 16.8 | 63.6 / 38.0 |
| | GA-S2 | 64.8 / 86.2 | 43.6 / 0.5 | 62.0 / 7.1 | 60.3 / 27.0 | 63.8 / 47.6 |
| Qwen2.5-3B | Uniform | 67.3 / 91.3 | 60.5 / 26.3 | 60.8 / 29.2 | 65.8 / 63.8 | 66.0 / 65.7 |
| | AR (LM-PPL) | 67.3 / 91.3 | 62.1 / 11.5 | 62.1 / 11.7 | 65.8 / 56.1 | 66.1 / 57.9 |
| | GA-S2 | 67.3 / 91.3 | 52.6 / 6.6 | 60.4 / 4.9 | 65.7 / 59.8 | 65.9 / 70.5 |
| Llama-3.2-3B | Uniform | 65.6 / 89.6 | 59.7 / 5.7 | 61.3 / 6.6 | 64.3 / 60.0 | 64.8 / 63.4 |
| | AR (LM-PPL) | 65.6 / 89.6 | 58.5 / 4.6 | 60.1 / 5.4 | 63.8 / 46.4 | 64.0 / 47.3 |
| | GA-S2 | 65.6 / 89.6 | 57.8 / 5.0 | 59.5 / 4.2 | 62.3 / 41.1 | 63.9 / 46.5 |
| Qwen3-8B | Uniform | 70.7 / 94.0 | 49.4 / 0.1 | 66.5 / 9.2 | 69.4 / 60.4 | 70.0 / 76.7 |
| | AR (LM-PPL) | 70.7 / 94.0 | 44.7 / 0.2 | 67.6 / 15.4 | 69.2 / 62.2 | 70.1 / 69.8 |
| | GA-S2 | 70.7 / 94.0 | 39.8 / 0.1 | 67.4 / 11.0 | 68.9 / 66.1 | 70.1 / 57.2 |

**Two readings, separated as the rules require:**

- **Short-context `Avg` recovers essentially fully by 700 M tokens.** Qwen3-8B reaches `70.0–70.1`
  against a teacher at `70.7`. **And the Baseline reaches `68.9–69.4` at the same checkpoint** — i.e.
  **by 700 M tokens the initialization advantage on `Avg` has shrunk to 0.6–1.2 points.**
- **Long-context RULER does not recover fully.** Best case Qwen3-8B Uniform: `76.7` against teacher
  `94.0` (**82 %**). Worst case Qwen2.5-1.5B AR: `38.0` against `86.2` (**44 %**). **Taylor-Calibrate's
  RULER advantage over Baseline persists at 700 M where its `Avg` advantage has nearly vanished** — e.g.
  Qwen2.5-1.5B Uniform `59.8` vs `39.9`.

> **⚠ `[X]`: Table 3 does not state its retained-attention budget.** The caption, the section text and
> the column headers give the selection policy and the token checkpoint but **not the hybrid ratio**.
> Table 2 used 25/50/75 %; Table 3 does not say which it inherits. **Every number in Table 3 is therefore
> uninterpretable as a hybrid-ratio datapoint**, and I am not going to guess. This is a real reporting
> gap in the paper, not a search failure — I looked in §4.3, the caption, and §4.1.

### 5.8f ⚠ The `4.9×–9.2×` is NOT in a table — it is text about a figure [A/F]

**This is the number the Adapter flagged as the one that would rescue the budget, and it is the one that
must be scrutinised hardest. Here is exactly what it is.**

**Verbatim, §4.5.2 in full:**

> *"A practical question is how much Stage-2 distillation is needed after initialization. **Figure 5**
> compares the number of training tokens required for Baseline and Taylor-Calibrate to reach the same
> target quality in **several representative settings**. **Across the representative runs in Figure 5**,
> Taylor-Calibrate reaches the target with fewer tokens, yielding speedups from **4.9× to 9.2×**. This
> suggests that a large fraction of the recovery budget is spent moving the converted model out of a poor
> initial regime."*

**Answering the Adapter's four questions directly:**

| question | answer |
|---|---|
| **Fewer tokens than what baseline?** | **`Baseline`** — verbatim per §4.1: *"copy teacher `Q/K/V/O` projections, leave new recurrent parameters at their **random default**."* **Not a strong baseline.** It is not compared against Zero-Gate, which Table 4 shows gets 1.66× of the way for free with one line of code. |
| **Which model and size?** | **`[X]` — not stated.** "Several representative settings" / "the representative runs in Figure 5". No model is named in the text, and the paper has four teachers spanning 1.5 B–8 B. |
| **To reach what recovery target?** | **`[X]` — "the same target quality" is never defined.** No metric, no threshold. Given §5.8e, the answer differs enormously depending on whether the target is `Avg` (recovers fully) or `RULER` (does not). |
| **Is the range across settings or seeds?** | **Across *settings*, not seeds** — "across the representative runs" in "several representative settings". **There is no seed variance reported anywhere in this paper.** Appendix C concedes it, verbatim: *"The current experiments also emphasize recovery checkpoints rather than exhaustive multi-seed sweeps."* |

> **Verdict on the `4.9×–9.2×`: it is `[A/F]` — a text claim about an unlabelled figure, against a
> random-default baseline, to an undefined target, on unnamed models, with no seed variance.**
> **It does not meet this project's bar and must not enter any budget arithmetic.** Per the standing
> rule that what flatters us gets scrutinised hardest: **this is the number that would have moved donor
> conversion from unaffordable to affordable, and it is the least-supported number in the paper.**

**What *is* `[T]`-grade about token efficiency, and it is a weaker claim:** Table 3's 100 M column shows
Taylor-Calibrate reaching `Avg` 62.6 where Baseline is at 37.2 (Qwen2.5-1.5B Uniform), and `66.5` vs
`49.4` (Qwen3-8B Uniform). **That is a real, table-sourced initialization advantage at a fixed small
token budget.** But by 700 M the same pair is `64.4` vs `61.4` — **the advantage is largely a
head-start, not a different destination**, which is exactly what Appendix C's "Downstream distillation
dependence" paragraph (§5.7d) says.

**Absolute cost, which is what our budget actually needs:** the recovery schedule is **100 M tokens
(Stage 1) + 700 M tokens (Stage 2) = 800 M tokens** `[T, Table 3 column headers]`, on donors of
1.5 B–8 B. **Even a true 9.2× saving is a saving on 800 M tokens, not on MOHAWK's 3 B** — and
`ADAPTER_MEMO_01` §2.2b's FLOP model scales with `params × tokens`, so the donor-size term is untouched.
**I am not doing that arithmetic here; the input numbers for it are `[A/F]` and it would be exactly the
kind of derived figure this project has been burned by.**

### 5.8g `[X]` — what Taylor-Calibrate does not report

- The retained-attention budget for Table 3.
- Any isolated ablation of the value-side OLS.
- Any seed variance, anywhere.
- Figure 5's underlying numbers (models, targets, per-setting speedups).
- Any donor above 8 B — Appendix C states larger teachers were **not** run, verbatim: *"we keep the
  quantitative claims in this paper to the completed teacher settings."*
- Wall-clock or GPU-hour cost of either phase.

### 5.8h Net effect on §5.0 and on `BRIEF_R2`

**The novelty verdict is unchanged (ADJACENT, not anticipated).** What changes is the *prior on the
mechanism working*, and it moves **against** R2 on the evidence now in hand:

1. The one published closed-form value-side calibration **contributes nothing measurable on its own**
   (`22469.0` → `22470.6`) `[T, Table 4]`.
2. Its headline `88×` is a ratio between two broken models worth **+1.2 `Avg` points** `[T, Table 4]`.
3. Its headline `4.9×–9.2×` is **figure-text against a random-default baseline to an undefined target**
   `[A/F, §4.5.2]`.
4. Its authors state plainly that closed-form initialization **does not remove the need for distillation**
   `[T, Appendix C]`.
5. **And the counter-argument stands: all of the above is about a one-scalar-per-head solve.** R2 proposes
   ~8.4 M degrees of freedom where they used ~64. **Nothing here measures that.**

> **The honest statement of where R2 now sits: the construction is sound (Controller), the commuting
> step is prior art (§5.1), the setup is published (§5.3), the weak form of the solve is measured at
> zero (§5.8c), and the strong form is unmeasured by anyone. `[X]` on the only question that decides it.**
> **The A1.5 pre-screen — principal angles, one SVD per layer, before any apparatus — is the correct
> next step and this section strengthens rather than weakens that.**

## 5.9 Closing sub-question (iv) — does any paper consider and reject solving `W_v`? `[X]`

**Answer: no. Searched across the methods surveyed in §1–§2 and read directly in LoLCATs' and
Mamba-in-the-Llama's own text. `[X] not found` — which the brief named as a complete and valued answer.**

**The nearest miss, and it is a design choice rather than a rejection.** LoLCATs (arXiv:2410.10254),
verbatim from its attention-transfer description:

> *"However, **to keep our training footprint low, we freeze the original pretrained attention layer's
> parameters** and simply insert new `φ_q`, `φ_k` after `W_q`, `W_k` in each softmax attention (Fig. 1
> left). We compute outputs `y`, `ŷ` with the same [inputs]."*

So LoLCATs deliberately leaves `W_v` untouched during attention transfer and puts **all** the learnable
capacity on the q/k side, in the feature maps. **Its stated reason is training footprint — not that a
value-side fit was tried, considered, or judged unhelpful.** LoLCATs does later touch `W_v`, but only via
LoRA and only in the second stage, verbatim: *"we freeze the linear attention weights and add LoRA
weights to query, key, value, and output projections."*

Mamba-in-the-Llama likewise reuses `W^V` by a shape-based remap (`V → X`) and states no consideration of
solving it `[T, §3 equations]`.

> **Interpretive weight, stated honestly: low.** Papers publish what they did, not what they rejected, so
> **the absence of a stated rejection is weak evidence that nobody considered it.** What it does
> establish is that **no published argument against the construction exists**, so R2 is not walking into
> a known negative result. **The negative evidence that does exist is §5.8c's measurement, not an
> argument** — and that measures the scalar form, not R2's.

**Also closed this pass:** the PTQ / layer-wise-reconstruction crossover (SparseGPT / GPTQ / OBC used to
heal an *architectural substitution* rather than quantisation or pruning). **`[X]` — searched, nothing
found.** The nearest thing in the whole survey is **NBL (§5.2)**, which is a closed-form solve healing a
substitution, but it comes from the LMMSE/estimation lineage rather than the OBS/Hessian lineage, and it
substitutes the block away rather than replacing its mixing. **Note this absence is now doubly
interesting: the low-rank companion file (`prior_art/LOWRANK_PROJECTION_PRIOR_ART.md` §3) finds that
SVD-LLM's "layer-wise closed-form update" is structurally the same apparatus as our D4 solver, applied to
compression. The apparatus is standard in compression and unused in architectural substitution.** That
crossover remains the clearest unclaimed ground found across both questions.

## 5.10 ⭐ Taylor-Calibrate's feature map — asked because R2a's pre-screen made it urgent, and the answer reframes the line

**The Adapter asked:** *"Their name suggests a Taylor expansion of the exponential, which would be a very
different object from `elu(x)+1`. If their `φ` is the reason their solve works, then `φ` — not the solve —
is the load-bearing choice."*

> ### **Answer: Taylor-Calibrate has no feature map. There is no `φ` anywhere in their student.**
> **The Taylor expansion is an analysis device used to derive calibration constants, not the student's
> kernel. Their student is a gated delta-rule recurrence with per-head-normalised `q`/`k`.**
> **And the Adapter's conditional fires: the operator choice IS load-bearing, and their own ablation
> (§5.8c) shows the value-side solve is not.**

### 5.10a The Taylor expansion is a lens, not a kernel — [T, §2.2 verbatim]

> *"A useful theoretical lens for our method is to expand the softmax numerator around a neutral logit
> regime … `exp(s) = 1 + s + s²/2 + O(s³)`. **(3)** Truncating this expansion motivates a fixed-size
> linear-recurrent approximation of softmax attention. **More importantly for our purposes, it yields a
> calibration methodology**: second-order curvature can be compiled into a small number of effective
> scaling terms. One convenient summary of this viewpoint is that the second-order correction can be
> folded into an effective first-order sequence-mixing scale, often written as `γ² = 1 + μ₃/(2μ₂)` for
> low-order logit moments `μ₂, μ₃`, **while the value pathway admits a per-head least-squares rescaling
> factor `σ` that matches the teacher output amplitude.**"*

**Read the emphasis.** Eq. (3) is *"a useful theoretical lens"* that *"motivates"* an approximation and
*"yields a calibration methodology."* **It is never instantiated as the student's feature map.** The
second-order Taylor kernel — the Based-style `φ` that R2's brief §3.4 proposes sweeping — is the thing
this paper takes its *name* and its *statistics* from, and **not** the thing it runs.

### 5.10b What their student actually is — [T, §2.3, Eq. 4–6 verbatim]

**Gated DeltaNet.** Verbatim: *"In the variant used here, **queries and keys are projected from the same
teacher-inherited hidden states and then normalized per head.**"* With `S_t ∈ ℝ^{d_h × d_h}` the
recurrent state:

```
g_t = −exp(A_log) · softplus( a_proj(x_t) + dt_bias ) ,   β_t = σ( b_proj(x_t) )          (4)

S_t = exp(g_t) · ( S_{t−1} − β_t S_{t−1} k_t k_tᵀ ) + β_t v_t k_tᵀ ,     o_t = S_t q_t     (5)

y_t = W_O ( RMSNorm(o_t) ⊙ SiLU( g_proj(x_t) ) )                                           (6)
```

**So `φ` = per-head normalisation of `q` and `k`. That is the whole feature map.** No `elu+1`, no
exponential Taylor kernel, no random features, no learned Hedgehog map.

**And the mixing is not a kernel mixing at all.** Eq. (5) carries two things pure linear attention does
not have:

1. **A multiplicative, input-dependent decay `exp(g_t)`** — an explicit recency prior, with `A_log`,
   `a_proj` and `dt_bias` as parameters.
2. **A delta-rule removal term `− β_t S_{t−1} k_t k_tᵀ`** — the state *erases* before it writes.

> **Consequence for R2, stated precisely: the `A_lin` induced by GDN is NOT of the form
> `φ(q_t)ᵀ φ(k_s)`.** It is a gated, decaying, delta-rule mixing whose row `t` is shaped by learned
> per-token gates. **R2's construction — a fixed `φ` inducing a fixed `A_lin`, then one linear solve —
> does not describe Taylor-Calibrate's operator**, and Taylor-Calibrate's results therefore transfer to
> R2 *less* directly than §5.3 implied. **§5.3's adjacency claim stands on the value-side move; it does
> not extend to the mixing.**

### 5.10c ⚠ Where their analytical budget actually went — and it is not the value path

Cross-reading §5.7a/§5.8b against §2.2, the four Phase-1 assignments split as:

| what is calibrated | how | degrees of freedom | what it controls |
|---|---|---|---|
| **decay bias** `dt_bias,h` | `softplus⁻¹(ln2 / d_h)`, `d_h` = teacher's **attention-weighted mean look-back distance** (Eq. 8) | 1/head | **the recency/decay structure of the mixing** |
| **write gate** `b_proj` | row-scaled to `logit(β*_h)`, `β*_h = 0.3 + 0.4·c_h`, `c_h` = **attention entropy concentration** (Eq. 9) | 1/head | **how peaked vs diffuse each row of the mixing is** |
| **output gate** `g_proj` | RMS-matched, `λ = 0.01` (Eq. 42) | 1/layer | amplitude of the branch |
| **value path** `W_V` | **scalar OLS** `σ*_h` (Eq. 36) | 1/head | **output amplitude only** |

> **Three of the four calibrated quantities shape the MIXING. Exactly one touches the value path, and
> the authors scope it to amplitude in their own words** (§5.7c: *"the value-side least-squares step
> should transfer broadly because **it only matches output amplitude**"*).
>
> **The team closest to R2 spent its analytical budget on getting the mixing's decay and peakedness
> right, and gave the value side a single scalar. R2 proposes the opposite allocation: take the mixing
> as given from a fixed `φ`, and put ~8.4 M degrees of freedom into the value side.**

**Combined with §5.8c — where Phase 1 *in full* was worth `22469.0 → 22470.6`, i.e. nothing, against a
gradient stage worth 18.5× — the reading is consistent and it is not the one R2 wants:** in the one
published system of this shape, **the value-side solve is the least load-bearing component, and the
mixing operator's structure is where the information is.**

### 5.10d ⚠⚠ Direct bearing on the R2a pre-screen signal the Adapter reported

The coordinator reports R2a's first signal: **`elu(x)+1` scores worse than a trivial causal-uniform
mixing.** Nothing in this pass can confirm or refute that number — but **the literature makes it
mechanistically unsurprising, and that is worth having before the sweep completes:**

- **`elu(x)+1` produces a mixing with no recency structure.** Its row `t` weights positions by
  `φ(q_t)ᵀφ(k_s)` with all `φ ≥ 0` and no positional term, so the induced `A_lin` is comparatively flat
  over history.
- **Causal-uniform is flat too — but correctly normalised and with the right support.** So a flat kernel
  with the wrong scale can plausibly land *below* the trivial flat baseline. **This is the ≈0.05
  structural floor Amendment A1.4 already identified** (`C6 causal-uniform = 0.059`).
- **Every high-performing published operator in this literature has explicit decay or a local window.**
  GDN has `exp(g_t)` (Eq. 5). LoLCATs has a 64-token sliding window (§2). Based has a short conv.
  **§9 of this survey already flagged that the best methods are "linear attention plus a bounded local
  piece"; §5.10b now adds that the 2026 state of the art replaces the bounded window with a learned
  decay.**

> **Predictive statement, offered as a prediction rather than a finding, so it can be scored:** if R2a's
> sweep includes any `φ` carrying recency (a decay term, a local window, or a positional kernel) it
> should separate cleanly from `elu(x)+1` and from causal-uniform. **If instead every `φ` in the sweep
> lands near the causal-uniform floor, the correct reading is not that the solve failed — it is that
> R2 §3.4's `φ` menu is drawn entirely from the recency-free corner of the design space**, and that is a
> fixable design error rather than a refutation of the construction.

**None of R2 brief §3.4's four proposed maps — `elu(x)+1`, Based 2nd-order Taylor, Performer FAVOR+, an
unfitted Hedgehog map — carries a decay or a window.** `[T, brief §3.4]` **That is a gap in the
pre-registered design which this section identifies and which A1.5's pre-screen will expose cheaply.**

### 5.10e What this does NOT say

- **It does not say R2 is wrong.** It says the one adjacent published system allocates its effort the
  opposite way, and that its value-side component measured at zero **in scalar form**.
- **It does not say a decay-bearing `φ` would rescue R2.** No paper found runs R2's construction with
  any `φ`. `[X]`.
- **It does not transfer Taylor-Calibrate's numbers to R2.** Different operator (§5.10b), different
  solve rank (§5.3), different endpoint (§5.7d).

## 5.11 Appendix D.1, Table 6 — the detailed recovery table, transcribed [T]

**Caption, verbatim:** *"Detailed recovery results for all four teacher settings at 100M tokens after
Stage 1 and 700M tokens after Stage 2. Avg is shown only when the full short-context set needed for that
row is available; RULER is reported separately as a long-context probe. **Bold**: best non-teacher PPL
(minimum), Avg (maximum), and RULER (maximum) within each model block."*

**This table adds the PPL column that Table 3 omits, and per-task detail. It is where the two findings
below come from — neither is discussed in the paper's own text.**

### 5.11a Full transcription — Qwen3-8B, the largest donor [T]

Columns: `Ckpt | Selection | Init | PPL↓ | ARC-C | ARC-E | Hella. | PIQA | MMLU | OBQA | RA | WG | BoolQ | LAMB. | COPA | SciQ | Avg | RULER`

| Ckpt | Sel. | Init | PPL↓ | ARC-C | ARC-E | Hella. | PIQA | MMLU | OBQA | RA | WG | BoolQ | LAMB. | COPA | SciQ | Avg | RULER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | — | **Teacher** | 9.73 | 56.7 | 80.9 | 74.9 | 77.9 | **74.9** | 41.8 | 41.4 | 68.0 | 86.6 | 64.1 | 85.0 | 96.6 | **70.7** | **94.0** |
| 100M | Uniform | Baseline | 83.86 | 37.6 | 60.6 | 51.4 | 69.7 | 31.8 | 33.4 | 30.1 | 51.4 | 66.0 | 5.1 | 76.0 | 79.9 | 49.4 | 0.1 |
| 100M | Uniform | Taylor-Cal. | 14.92 | 56.1 | 79.9 | 72.8 | 77.3 | 66.2 | 43.6 | 37.1 | 67.9 | 79.0 | 36.9 | 86.0 | 94.7 | 66.5 | 9.2 |
| 100M | AR | Baseline | 172.30 | 32.4 | 51.6 | 49.0 | 68.4 | 27.2 | 32.6 | 29.4 | 51.1 | 57.3 | 3.0 | 69.0 | 65.0 | 44.7 | 0.2 |
| 100M | AR | Taylor-Cal. | 11.47 | 55.5 | 79.2 | 72.5 | 77.3 | 66.4 | 43.6 | 37.2 | 68.3 | 78.9 | 53.4 | 84.0 | 95.0 | 67.6 | 15.4 |
| 100M | GA-S2 | Baseline | 297.36 | 31.1 | 53.4 | 35.9 | 68.3 | 24.2 | 27.4 | 26.0 | 50.0 | 45.7 | 4.8 | 53.0 | 57.4 | 39.8 | 0.1 |
| 100M | GA-S2 | Taylor-Cal. | 11.12 | 56.7 | 79.7 | 72.5 | 77.4 | 63.0 | 41.8 | 36.5 | 68.1 | 78.0 | 53.5 | 87.0 | 95.3 | 67.4 | 11.0 |
| 700M | Uniform | Baseline | 10.40 | 56.2 | 80.1 | 73.5 | 78.0 | 67.9 | 42.4 | 36.6 | 69.2 | 82.3 | 61.3 | 89.0 | 96.2 | 69.4 | 60.4 |
| 700M | Uniform | Taylor-Cal. | 9.98 | 56.1 | 80.3 | 74.4 | 78.3 | 70.2 | 42.4 | 38.0 | 69.2 | 84.6 | 63.1 | 87.0 | 96.3 | 70.0 | **76.7** |
| 700M | AR | Baseline | 9.90 | 55.5 | 81.4 | 73.4 | 77.9 | 66.8 | 42.6 | 36.8 | 70.7 | 82.9 | 60.5 | 85.0 | 96.6 | 69.2 | 62.2 |
| 700M | AR | Taylor-Cal. | **9.45** | 56.1 | 79.2 | 74.2 | 78.3 | 70.2 | 43.6 | 39.7 | 70.3 | 84.5 | 62.9 | 86.0 | 96.0 | 70.1 | 69.8 |
| 700M | GA-S2 | Baseline | 10.18 | 55.2 | 79.8 | 73.9 | 79.0 | 62.9 | 43.8 | 36.8 | 70.3 | 83.7 | 59.7 | 86.0 | 96.1 | 68.9 | 66.1 |
| 700M | GA-S2 | Taylor-Cal. | 9.46 | 56.9 | 80.7 | 74.0 | 78.3 | 69.0 | 43.0 | 40.1 | 71.4 | 82.9 | 61.9 | 87.0 | 96.0 | 70.1 | 57.2 |

### 5.11b The other three donors — teacher rows and the decision-relevant columns [T]

**Reduced view, flagged as a reduction:** `PPL / MMLU / Avg / RULER` only. The full 12-task rows are in
the paper's Table 6 and were read; the omitted columns move consistently with `Avg` and none of them
changes a conclusion below.

**(a) Qwen2.5-1.5B-Instruct** — Teacher: `PPL 9.66 | MMLU 60.2 | Avg 64.8 | RULER 86.2`

| Ckpt | Sel. | Init | PPL↓ | MMLU | Avg | RULER |
|---|---|---|---|---|---|---|
| 100M | Uniform | Baseline | 210.13 | 25.5 | 37.2 | 0.6 |
| 100M | Uniform | Taylor-Cal. | 13.54 | 50.8 | 62.6 | 7.7 |
| 100M | AR | Baseline | 58.71 | 23.1 | 44.7 | 1.0 |
| 100M | AR | Taylor-Cal. | 12.89 | 53.4 | 61.1 | 5.0 |
| 100M | GA-S2 | Baseline | 142.41 | 27.1 | 43.6 | 0.5 |
| 100M | GA-S2 | Taylor-Cal. | 12.54 | 51.9 | 62.0 | 7.1 |
| 700M | Uniform | Baseline | 11.99 | 44.1 | 61.4 | 39.9 |
| 700M | Uniform | Taylor-Cal. | 10.76 | 53.1 | 64.4 | 59.8 |
| 700M | AR | Baseline | 12.23 | 48.5 | 61.4 | 16.8 |
| 700M | AR | Taylor-Cal. | 10.65 | 55.7 | 63.6 | 38.0 |
| 700M | GA-S2 | Baseline | 11.96 | 41.6 | 60.3 | 27.0 |
| 700M | GA-S2 | Taylor-Cal. | 10.57 | 54.6 | 63.8 | 47.6 |

**(b) Qwen2.5-3B-Instruct** — Teacher: `PPL 8.56 | MMLU 66.4 | Avg 67.3 | RULER 91.3`

| Ckpt | Sel. | Init | PPL↓ | MMLU | Avg | RULER |
|---|---|---|---|---|---|---|
| 100M | Uniform | Baseline | 10.78 | 48.5 | 60.5 | 26.3 |
| 100M | Uniform | Taylor-Cal. | 10.83 | 51.5 | 60.8 | 29.2 |
| 100M | AR | Baseline | 10.32 | 57.5 | 62.1 | 11.5 |
| 100M | AR | Taylor-Cal. | 10.33 | 57.8 | 62.1 | 11.7 |
| 100M | GA-S2 | Baseline | 15.97 | 49.8 | 52.6 | 6.6 |
| 100M | GA-S2 | Taylor-Cal. | 11.73 | 57.2 | 60.4 | 4.9 |
| 700M | Uniform | Baseline | 9.09 | 55.1 | 65.8 | 63.8 |
| 700M | Uniform | Taylor-Cal. | 8.95 | 57.0 | 66.0 | 65.7 |
| 700M | AR | Baseline | 9.08 | 61.5 | 65.8 | 56.1 |
| 700M | AR | Taylor-Cal. | 8.98 | 61.8 | 66.1 | 57.9 |
| 700M | GA-S2 | Baseline | 9.45 | 55.8 | 65.7 | 59.8 |
| 700M | GA-S2 | Taylor-Cal. | 9.13 | 61.3 | 65.9 | 70.5 |

**(c) Llama-3.2-3B-Instruct** — Teacher: `PPL 11.05 | MMLU 60.6 | Avg 65.6 | RULER 89.6`

| Ckpt | Sel. | Init | PPL↓ | MMLU | Avg | RULER |
|---|---|---|---|---|---|---|
| 100M | Uniform | Baseline | 20.74 | 38.3 | 59.7 | 5.7 |
| 100M | Uniform | Taylor-Cal. | 17.41 | 39.9 | 61.3 | 6.6 |
| 100M | AR | Baseline | 20.43 | 31.3 | 58.5 | 4.6 |
| 100M | AR | Taylor-Cal. | 17.45 | 36.1 | 60.1 | 5.4 |
| 100M | GA-S2 | Baseline | 18.87 | 26.5 | 57.8 | 5.0 |
| 100M | GA-S2 | Taylor-Cal. | 16.83 | 31.9 | 59.5 | 4.2 |
| 700M | Uniform | Baseline | 12.05 | 49.6 | 64.3 | 60.0 |
| 700M | Uniform | Taylor-Cal. | 11.62 | 49.7 | 64.8 | 63.4 |
| 700M | AR | Baseline | 12.22 | 41.4 | 63.8 | 46.4 |
| 700M | AR | Taylor-Cal. | 11.85 | 45.0 | 64.0 | 47.3 |
| 700M | GA-S2 | Baseline | 12.05 | 34.6 | 62.3 | 41.1 |
| 700M | GA-S2 | Taylor-Cal. | 11.92 | 42.1 | 63.9 | 46.5 |

### 5.11c ⚠ Finding 1 — Taylor-Calibrate is WORSE than Baseline on long-context in some cells, and the paper does not say so

**Qwen3-8B, 700M, GA-S2 selection: RULER Baseline `66.1` vs Taylor-Calibrate `57.2`.** `[T, Table 6(d)]`
That is a **−8.9 point regression** on the largest donor at the final checkpoint.

It is not isolated. From Table 2 (zero-shot, §5.8d): **Llama-3.2-3B at 75 % retained, RULER: Zero-Gate
`31.0` and Small-Gate `30.9` vs Taylor-Calibrate `24.0`** — the trivial one-line gate heuristics beat the
full method by ~7 points. **Qwen2.5-3B at 100M/GA-S2: RULER Baseline `6.6` vs Taylor-Calibrate `4.9`.**

> **The paper's Table 6 caption bolds "best RULER within each model block", so these inversions are
> visible in its own formatting, but the running text does not discuss them.** The §4.2 text says only
> that *"RULER gains are more model- and budget-dependent"* — which is true and considerably softer than
> "the method sometimes loses to doing nothing."

**Why this matters to us specifically:** long-context retrieval is the capability
`ADAPTER_MEMO_01` §2.2f cares about (the KV-traffic half) and the one this survey's §9 identified as
architecturally capped. **A calibration method whose long-context effect is sign-unstable across layer-
selection policies is not yet a controlled instrument for that capability**, and the variance is being
absorbed into a "model- and budget-dependent" phrasing. **With no seed variance reported anywhere
(§5.8g), it is not possible to tell whether these inversions are real effects or noise — and that is
itself the finding.**

### 5.11d ⚠ Finding 2 — MMLU does not recover, and it is the worst-recovering task in the table

Teacher vs best 700M Taylor-Calibrate cell, per donor `[T, Table 6]`, **mine (subtraction only)**:

| donor | teacher MMLU | best 700M Taylor-Cal. MMLU | gap | teacher Avg | best 700M Avg | gap |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 60.2 | 55.7 | **−4.5** | 64.8 | 64.4 | −0.4 |
| Qwen2.5-3B | 66.4 | 61.8 | **−4.6** | 67.3 | 66.1 | −1.2 |
| Llama-3.2-3B | 60.6 | 49.7 | **−10.9** | 65.6 | 64.8 | −0.8 |
| Qwen3-8B | 74.9 | 70.2 | **−4.7** | 70.7 | 70.1 | −0.6 |

> **`Avg` recovers to within 0.4–1.2 points of teacher while MMLU stays 4.5–10.9 points down.** Because
> `Avg` is a macro-average over twelve tasks, **a large MMLU deficit is diluted to near-invisibility in
> the headline number.** The `Avg`-recovers-fully story of §5.8e is true and incomplete.

**This is the same shape as the finding already banked for LoLCATs in §2** (MMLU −11.2 / −11.1 / −10.8 at
8B/70B/405B while most other tasks were near-lossless). **Two independent conversion methods, five years
of technique apart, both leave a large MMLU-shaped hole.** That is now a pattern rather than a
one-paper artefact, and it is the single most consistent quality signature in this whole survey.

> **Consequence for the programme's success criterion.** The sealed constraint is *"successo = retention
> general-purpose."* **MMLU is the closest thing in these benchmarks to general-purpose knowledge
> retention, and it is the metric that conversion damages most and heals least.** Any in-house
> conversion evaluation that reports a macro-average will hide exactly the deficit the constraint cares
> about. **Report MMLU as its own line, never inside an average.**

### 5.11e What Table 6 adds that Table 3 could not

- **The PPL column.** Qwen3-8B at 100M/GA-S2: Baseline `297.36` vs Taylor-Calibrate `11.12` — **the
  largest single initialization effect in the paper (mine: 26.7×), and it is larger than the abstract's
  headline 88× is in `Avg` terms.** It confirms §5.8b's reading: the method's effect is overwhelmingly on
  perplexity and only weakly on downstream task scores.
- **Teacher PPL for all four donors** (`9.66 / 8.56 / 11.05 / 9.73`), absent from Tables 2–4, which makes
  the 700M PPL cells interpretable for the first time: best-case Qwen3-8B reaches `9.45` against a
  teacher `9.73` — **the converted student's perplexity slightly beats its teacher's while its MMLU is
  4.7 points down.** A further, direct demonstration that PPL is not tracking the capability we care
  about.
- **`[X] still:** Table 6 does **not** state the retained-attention budget either.** The gap flagged in
  §5.8e survives into the appendix. Across Tables 3 and 6, **the entire recovery half of this paper is
  reported without its hybrid ratio.**
