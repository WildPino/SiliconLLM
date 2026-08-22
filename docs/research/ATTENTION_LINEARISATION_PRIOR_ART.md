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


