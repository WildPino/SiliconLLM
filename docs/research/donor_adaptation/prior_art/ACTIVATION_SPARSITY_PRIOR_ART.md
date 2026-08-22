# Activation-Sparsity / Thresholding Prior Art

Status: complete pass — Q1 (ReLU Strikes Back, ProSparse, Turbo Sparse/
dReLU, Q-Sparse, R-Sparse, plus Sparsing Law as a 2025-2026 scaling-law
addendum), Q2 (TEAL, CATS), Q3 (LLM in a Flash, Neuralink, LLaMA-MoE),
Q4 (BitNet a4.8, Sparse-BitNet), token-budget table, single-source claim
list, and framing critique are all filled in below. All numbers were
retrieved via tool-mediated fetch of arXiv HTML renders (see the caveat
in the single-source list, item 13) — not primary-source pixels read
directly by a human, which is the closest approximation to the governing
rule achievable with the tools available in this session.

Scope: activation-function / thresholding branch of activation sparsity
(ReLUfication, training-free thresholding, and the neuron-bundling
contradiction). MoE-ification and structured-sparsity prior art are covered
by a companion pass and NOT redone here.

Tagging convention per the brief: **[T]** = read in the paper's own table
(table number + row + column given), **[A]** = text-only claim, **[X]** =
not found. No number below is used in a ratio without stating numerator and
denominator cells explicitly.

---

## Q1: ReLUfication and threshold-based activation sparsity

### ReLU Strikes Back (Mirzadeh, Alizadeh, Mehta, Del Mundo, Tuzel, Samei, Rastegari, Farajtabar — Apple), arXiv:2310.04564, HTML v1 read at arxiv.org/html/2310.04564v1

**Mechanism.** Two-stage "relufication": Stage 1 = direct swap of the FFN
nonlinearity (GELU in Falcon, SiLU in Llama's SwiGLU) for plain ReLU,
continued-pretrained. Stage 2 = additional ReLU insertions after
normalization layers so QKV and up-projection *inputs* also see a ReLU, not
just the FFN. A further variant, "shifted ReLU" ReLU(x−b) with a constant
scalar b, is explored in Sec. 5.3: for Llama, quote — "setting b=1 ... will
result in dropping 95% of the preactivations" **[A]**. No additional
magnitude-threshold pruning is layered on top; sparsity comes purely from
where the (shifted) ReLU's argument is negative.

**Training cost.** Continued pretraining, not from scratch. Quote:
"After finetuning on 30 billion tokens of the RefinedWeb, Fig. 4 shows that
the modified models have significantly more sparsity" **[A]** — this is
Stage 1. Stage 2 is separately continued-trained; the paper's own text
gives "50B tokens for stages 1 and 2, respectively" in one place, which
reads as inconsistent with the 30B figure quoted elsewhere — **I cannot
fully reconcile the two numbers from the fetched text; flagging this as
[A], not [T], and as an internal-consistency question I could not resolve
by fetch alone.** No GPU type, GPU count, or wall-clock time is given
anywhere in the paper (checked Sec. 3.1 / 4.1 explicitly) — training cost
in T4/A100-hours is **[X] not stated**. Models covered: OPT 1.3B/2.7B/6.7B,
Falcon 7B, Llama 7B — i.e. the whole method was only demonstrated at ≤7B.
**No evidence at any scale near 100B.**

**Achieved sparsity and quality retention — Table 1 (verbatim transcription),
"Comparing zero-shot performance across several tasks":**

Columns: Input Sparsity % (QKV / DownProj / UpProj / Avg-not-shown-as-column-but-implied), FLOPS (G), then zero-shot accuracy on Arc-E, Arc-C, Hellaswag, BoolQ, PIQA, LAMBADA, TriviaQA, WinoGrande, SciQ.

| Model (stage) | QKV% | DownProj% | UpProj% | FLOPS(G) | Arc-E | Arc-C | Hellaswag | BoolQ | PIQA | LAMBADA | TriviaQA | WinoGrande | SciQ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Falcon 7B (baseline) | 0 | 1 | 0 | 6.6 | 66.8 | 74.6 | 40.2 | 57.7 | 73.5 | 79.4 | 74.5 | 40.4 | 94.0 |
| Falcon 7B (s1) | 0 | 94 | 0 | 4.1 | 65.2 | 72.2 | 39.1 | 55.4 | 70.6 | 78.4 | 69.2 | 40.5 | 93.1 |
| Falcon 7B (s2) | 56 | 95 | 56 | 2.2 | 64.8 | 73.6 | 38.6 | 55.3 | 68.4 | 78.9 | 67.6 | 40.4 | 93.4 |
| Llama 7B (baseline) | 0 | 0 | 0 | 6.6 | 68.4 | 75.5 | 42.1 | 69.9 | 74.8 | 78.7 | 73.1 | 49.9 | 95.4 |
| Llama 7B (s1) | 0 | 62 | 0 | 4.8 | 67.1 | 75.2 | 40.1 | 55.2 | 73.4 | 77.7 | 71.5 | 49.6 | 94.2 |
| Llama 7B (s2) | 51 | 65 | 67 | 2.9 | 66.4 | 73.8 | 39.6 | 54.8 | 69.9 | 77.9 | 70.7 | 48.5 | 93.8 |

**[T] Table 1**, row = model(stage), columns as headed above. Note the
DownProj sparsity is the dominant, consistently-high number (94-97% even
at baseline for Falcon's "1%" — likely a near-total-GELU-negative-region
artifact, not induced sparsity); QKV/UpProj sparsity only appears once
Stage 2 is applied. FLOPS drops baseline→s2: Falcon 6.6G→2.2G (**3.0x**,
numerator=baseline row, denominator=s2 row, same column "FLOPS(G)");
Llama 6.6G→2.9G (**2.3x**, same column). Zero-shot avg accuracy loss
baseline→s2 is small (e.g. Falcon Arc-E 66.8→64.8, a 2.0-point drop; Llama
BoolQ baseline 69.9→ s2 54.8, a **15.1-point drop** — the single steepest
degradation in the table, on the BoolQ column specifically).

**Table 2 (verbatim), MMLU 5-shot:**

| Model | Activation | FLOPS(%) | Avg | Humanities | STEM | Social Sciences | Other |
|---|---|---|---|---|---|---|---|
| Falcon 7B | SiLU | 100 | 26.4 | 24.8 | 27.4 | 27.2 | 26.2 |
| Falcon 7B | GELU | 100 | 27.7 | 28.1 | 26.0 | 28.0 | 29.4 |
| Falcon 7B | ReLU | 62 | 27.9 | 26.0 | 26.5 | 31.8 | 27.9 |
| Llama 7B | SiLU* | 100 | 35.1 | 37.9 | 30.2 | 37.0 | 37.1 |
| Llama 7B | GELU | 100 | 35.9 | 38.4 | 29.4 | 37.6 | 39.5 |
| Llama 7B | ReLU | 72 | 34.7 | 34.8 | 31.2 | 36.3 | 37.8 |

**[T] Table 2.** At 62-72% of baseline FLOPS, MMLU avg moves within
±1.2 points of the SiLU/GELU baselines — Falcon ReLU (27.9) is actually
*above* both its own SiLU (26.4) and GELU (27.7) baselines on this table;
Llama ReLU (34.7) is 1.2 points below its GELU baseline (35.9). Caption
note: "*" on Llama SiLU row = "we replace the SiLU function in Llama's
SwiGLU activation function with ReLU" — i.e. that row is itself already a
ReLU variant used as a second baseline, not stock Llama.

**Sparsity structure.** Structured / row-level, not scattered. Quote:
"unlike unstructured sparsity (e.g., weight pruning), this type of
sparsity is more hardware-friendly due to zeroing more extensive and
structured chunks, such as rows or columns" **[A]**, Sec. 1. This is
neuron-level sparsity (whole row/column of a weight matrix gated on/off
per input), which is the same granularity our engine needs, but the paper
does not discuss any minimum contiguous-block size or memory-page framing
— it treats "row" as intrinsically hardware-friendly without a stated
byte threshold.

**Speedup claims.** FLOPs-based, not measured wall-clock skip-loading.
Quote: "FLOPS can serve as a good approximation of real-world efficiency"
**[A]**, Appendix B; the paper's own Figure 9(b) correlates FLOPS with A100
latency for OPT FFN layers but — per the fetch — **no actual end-to-end
A100 (or any GPU) wall-clock speedup number is reported**. The "up to
threefold" headline figure (Abstract) is the FLOPS ratio 6.6G→2.2G from
Table 1, not a measured runtime. Separately, Sec. 5.2 reports a
**1.27x** speedup for speculative decoding from reusing activated neurons
across γ=16 consecutive tokens — this one *is* a mechanism-level claim
about neuron reuse enabling contiguous/cached weight reuse, closer to what
we care about, but it's specific to the speculative-decoding setting, not
a general inference speedup, and I have not independently confirmed
whether it's measured or FLOPs-derived — **[A], flagged as ambiguous
provenance**.

**Relevance to us.** This paper never targets a CPU-bandwidth-bound engine;
FLOPS is its speedup currency throughout, and outside Sec. 5.2 it does not
measure memory traffic or skipped-load bytes at all. Its own admission that
DownProj sparsity is near-ceiling even in some "baseline" rows (Falcon:
1% listed at baseline, which reads as a placeholding near-zero rather than
a true zero) suggests GELU/SiLU already have substantial *natural*
DownProj-input sparsity pre-conversion, separate from anything induced.

---

---

### ProSparse (Song, Han, Zhang, Hu, Shi, Li, Chen, Liu, Li, Yang, Sun), arXiv:2402.13516, HTML v4 read at arxiv.org/html/2402.13516v4

**Mechanism.** Starts from an already-ReLUfied base (their own "ReluLLaMA")
and adds two things on top: (a) progressive L1 regularization on the FFN
intermediate activation, with the regularization weight λ following a
multi-stage schedule — warmup at small constant λ, then "λ is scheduled to
increase along a smooth sine curve from a trough value to a peak value"
**[A]**; (b) a hard magnitude threshold replacing plain ReLU with FATReLU:
σ(x) = x if x≥t else 0, t>0 **[A]**, "prunes less influential neurons to
further improve sparsity." So this is ReLUfication + L1 sparsity push +
explicit threshold — a strict superset of the ReLU-Strikes-Back mechanism.

**Training cost — [T] Table 6/7 (Appendix G) per WebFetch transcription:**

| Model | Accumulated tokens |
|---|---|
| ProSparse-7B | 34.60B |
| ProSparse-13B | 134.22B |
| ProSparse-1B (MiniCPM) | 473.02B (includes decay + SFT stages) |

Cross-check against a separate in-text sentence: "a relatively acceptable
rise in training tokens (i.e., 54.53B, accounting for only 2.73% of the
2T tokens used to pre-train LLaMA2)" **[A]** — this 54.53B figure does not
match the 34.60B Table-6 figure for the 7B model; I flag this as an
**unreconciled internal discrepancy**, possibly two different token
countings (e.g. regularization-only tokens vs. total pipeline tokens
including earlier ReLUfication stages already done by ReluLLaMA). Do not
treat either number as settled without reading Appendix G's table
structure directly (I only have the WebFetch transcription, not primary
pixels).

**Hardware — [A], text-only:** "All the 7B models are trained with the
AdamW optimizer on 8 A100 80GB GPUs for about 10 days. All the 13B models
are trained on 32 A100 80GB GPUs for about 20-30 days." This is A100-only;
no T4 or Turing-class datapoint exists in this paper. 8×A100×10 days ≈
1,920 A100-GPU-hours for the 7B token-count regime; 32×A100×25 days ≈
19,200 A100-GPU-hours for 13B — **these are MY arithmetic on their stated
GPU-count/day figures, not a number the paper tabulates.**

**Achieved sparsity and quality — [T] Table 1 ("Overall Experimental
Results"), verbatim as transcribed:**

| Setting | Code Gen | CommonSense | ReadingComp | GSM8K | MMLU | BBH | AGIEval | Avg Perf | Avg Sparsity |
|---|---|---|---|---|---|---|---|---|---|
| LLaMA2-7B (dense baseline) | 16.37 | 69.59 | 61.87 | 12.96 | 44.45 | 32.96 | 27.53 | 37.96 | — |
| ReluLLaMA-7B | 15.85 | 69.64 | 70.54 | 5.84 | 38.64 | 35.07 | 27.73 | 37.62 | 66.98 |
| ProSparse-7B | 19.42 | 66.27 | 63.50 | 12.13 | 45.48 | 34.99 | 27.46 | 38.46 | 89.32 |
| LLaMA2-13B (dense baseline) | 20.19 | 72.58 | 71.55 | 22.21 | 54.69 | 37.89 | 29.33 | 44.06 | — |
| ReluLLaMA-13B | 20.19 | 70.44 | 73.29 | 18.50 | 50.58 | 37.97 | 28.22 | 42.74 | 71.56 |
| ProSparse-13B | 28.42 | 69.76 | 66.91 | 26.31 | 54.35 | 39.90 | 28.67 | 44.90 | 88.80 |
| MiniCPM-1B (dense baseline) | 36.85 | 63.67 | 60.90 | 35.48 | 50.44 | 35.03 | 28.71 | 44.44 | — |
| ProSparse-1B | 42.04 | 64.37 | 60.73 | 34.57 | 49.51 | 34.08 | 27.77 | 44.72 | 87.89 |

Column "Avg Sparsity" is ProSparse's own reported metric (average
activation sparsity %). Numerator/denominator for the headline "89.32% at
7B" claim: this is the Avg Sparsity cell for row ProSparse-7B directly —
no ratio computed, it's a reported percentage. Notably ProSparse-7B's
*Average Performance* (38.46) is **higher** than the dense LLaMA2-7B
baseline (37.96) in this table — the paper's own comparison shows no
quality cost at 89% sparsity for the 7B model on this benchmark suite,
which is a striking claim; I have not verified whether "Average
Performance" here is a simple mean of columns that have very different
scales (e.g. GSM8K 0-100 exact-match vs BBH), which could make the average
metric itself sensitive to which subtasks move.

**Sparsity structure.** Per the fetch, described as **scattered / per-
element**, not block-structured: "Activation sparsity refers to the
existence of considerable weakly-contributed elements among activation
outputs" and inactivated *elements* (not rows/blocks) are skipped **[A]**.
This is a materially different granularity claim than ReLU-Strikes-Back's
"rows or columns" framing — worth resolving, since our engine only
benefits from ≥48KB contiguous runs. **[A], not independently confirmed
against a structural diagram in the paper — I was not able to fetch a
figure describing this.**

**Speedup — [T] Table 2 ("Acceleration Results"), verbatim:**

| Setting | Avg Sparsity | Speedup to Dense (approx., PowerInfer) | Step-2 wall-clock speedup (accurate GPU op) |
|---|---|---|---|
| Dense-7B | — | 1.00 | 1.00 |
| ReluLLaMA-7B | 66.98 | 3.10 | 1.35 |
| ProSparse-7B | 89.32 | — (n/a, PowerInfer row is the "*" variant at 4.44) | 2.00 |
| Dense-13B | — | 1.00 | 1.00 |
| ProSparse-13B | 88.80 | — (PowerInfer "*" variant at 4.52) | 2.44 |

Two speedup regimes reported: an "approximate acceleration" figure using
PowerInfer / llama.cpp tokens-per-second on GPU (this is the source of the
oft-quoted "4.52x"), and a separate "accurate acceleration" figure from
"average wall-clock time (us) ... with our sparse GPU operators" — the
latter tops out around **2.44x** for ProSparse-13B, notably lower than the
PowerInfer-approx 4.52x. **Both are real measured wall-clock numbers on
GPU (not CPU), not FLOP estimates**, per the fetch — but note the two
methods diverge by ~2x from each other depending on which sparse
inference stack is used, which matters if we try to reuse any of their
speedup number as a proxy for our own engine.

---

### Turbo Sparse / dReLU (Yixin Song et al.), arXiv:2406.05955, HTML v2 read at arxiv.org/html/2406.05955v2

**Mechanism — dReLU.** Equation (per fetch): `dReLU(x) := max(0, x·W_gate) *
max(0, x·W_up)` — i.e. ReLU is applied to **both** the gate and the up
projection outputs of the SwiGLU-style FFN, not just the gate as prior
ReLUfication (including ReLU Strikes Back) did. Quote: "existing
ReLUfication doesn't alter the activation distribution of the up
projection component" **[A]** — their stated reason for the modification.
This is the mechanism our project already reproduced in-house from
scratch (per the CONTEXT section), so this paper is the closest external
analog to our own measurement.

**Training cost — [A], text-only, Sec. 5:** "we pretrain our models on
150B tokens", hardware "64 A800-80G GPUs", LR schedule "5e-5→5e-6",
explicitly "continue pretraining" (not from scratch) on top of Mistral-7B
and Mixtral. The paper's own limitation quote: "Our models have only
undergone continued training on 150B tokens. Compared to the 15T tokens
used in pre-training for Llama-3 ... the limited number of training
tokens still results in some deficiencies" **[A]** — the authors flag
this cost themselves as small-but-non-negligible relative to full
pretraining, and as a source of residual quality deficiency. No A800-hour
total is given; 150B tokens on 64 A800-80G is the only concrete pair of
numbers — **converting to a duration or GPU-hour figure would be my
arithmetic, not theirs; I have not done that conversion here because no
throughput (tokens/GPU/day) figure was surfaced by the fetch.**

**Achieved sparsity and quality — [T] Table 6 ("Downstream Benchmarks"),
verbatim:**

| Model | Total Params | Active Params | ARC-c | Hellaswag | MMLU | TruthfulQA | WinoGrande | GSM8k | CommonSense | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B | 7B | 7B | 61.43 | 83.32 | 62.65 | 44.06 | 79.24 | 40.17 | 75.8 | 61.57 |
| TurboSparse-Mistral-7B | 7B | 2.5B | 62.2 | 82.17 | 63.89 | 46.64 | 76.16 | 50.84 | 76.2 | 63.65 |
| Mixtral-47B | 47B | 13B | 68.09 | 86.62 | 70.53 | 48.59 | 83.35 | 58.91 | 78.07 | 69.34 |
| TurboSparse-Mixtral-47B | 47B | 4.3B | 67.49 | 85.22 | 70.48 | 56.64 | 82.24 | 68.50 | 78.52 | 71.76 |

Both TurboSparse rows show **higher** Avg than their dense baselines
(63.65 vs 61.57; 71.76 vs 69.34) at 2.5B/7B and 4.3B/47B active-param
ratios respectively — i.e. this paper too reports net quality *gain*, not
loss, at high sparsity, which should be read skeptically alongside the
"150B continued-training tokens = a form of extra fine-tuning that can
itself lift eval scores" caveat the authors raise about deficiencies —
these are not obviously fully-controlled ablations against an
equally-fine-tuned dense baseline; **[A], I was not able to confirm from
the fetch whether the dense baseline rows are the stock released
checkpoints or the authors' own equally-token-matched continued-trained
dense control.** This matters a lot for whether the "free lunch" is real
or a fine-tuning-effect confound.

Sparsity level stated in text (Sec. 6.2), not the table: "TurboSparse-
Mistral-7B, on average, has 90% of the neurons inactive in each layer...
For TurboSparse-Mixtral-47B, this percentage is slightly lower at 85%"
**[A]**.

**Sparsity structure.** Per fetch: **unstructured/scattered**, selected by
top-k magnitude: "selecting the top-k% of values activated by dReLU...
based on their absolute magnitude" **[A]**, Eq. 3-4. This directly
matters to us — if accurate, Turbo Sparse's own reported sparsity is
**not** block-contiguous and would need the block-structuring step we
already did in-house (18%→50% skippable at block size 8) to become useful
on our engine; the paper does not appear to attempt that step itself.

**Speedup — [T] Table 8 ("Hybrid GPU-CPU"), verbatim:**

| Setting | Model | PowerInfer (tok/s) | llama.cpp (tok/s) | Speedup |
|---|---|---|---|---|
| PC-2080Ti | Mistral-7B-FP16 | 35.5 | 7.64 | 4.64x |
| PC-2080Ti | Mixtral-47B-INT4 | 22.24 | 6.63 | 3.35x |
| PC-Laptop | Mixtral-47B-INT4 | 33.12 | 13.1 | 2.52x |

Plus a separate mobile number (Table 9, per fetch): "PowerInfer-2 achieves
a 22.2x speedup ... on OnePlus-12 smartphone" **[A]**, not independently
re-verified by me against the actual table 9 cells. The GPU-CPU hybrid
numbers are real measured tokens/s via the PowerInfer framework against
llama.cpp dense baseline (not FLOP estimates) — this is the closest thing
in this literature set to our own memory-traffic framing, since PowerInfer
specifically exploits skipped weight loading on the CPU side of a hybrid
setup. Still: none of this is single-CPU Zen2, and none of it isolates a
"contiguous block ≥48KB" requirement the way our engine does — PowerInfer
routes hot/cold neurons between GPU-resident and CPU-resident pools, a
different mechanism than what we need for a CPU-only ternary engine.

---

### Q-Sparse (Wang, Ma, Wang, Wei — Microsoft), arXiv:2407.10969 (v1: 2024-07-15, v2: 07-20, v3: 07-24), v3 HTML read at arxiv.org/html/2407.10969v3 for the Block Q-Sparse definition, v1 HTML for the rest

**Mechanism.** Top-K magnitude sparsification of activations:
Y = (X ⊙ M)·Wᵀ where M = Top_K(|X|) selects the K largest-magnitude
activations per layer and rescales by an L2 norm factor **[A]**, Eq. 2-3.
Backprop uses a straight-through estimator so gradient flows through
masked positions unchanged (Eq. 13) rather than being zeroed — this is a
training-time trick to keep top-K differentiable, not a sparsity-shape
claim. Applies to full-precision and to 1-bit (BitNet b1.58) models alike.

**Block Q-Sparse — [A], Sec. 2.1 (only findable in v3, not v1's rendered
HTML; the term is absent from the v1 fetch despite being in the abstract
since v1, so treat this as a rendering gap in my v1 fetch, not evidence it
was added later).** Definition, verbatim per fetch: "apply the top-K
sparsity function on the activations in the block level, and the block
size is set to M so that there are always M−K zeros out of M consecutive
values." Motivation, verbatim: "While the top-k sparsification can be used
in the single-sample mode, it is not friendly with the batch mode for the
current GPU devices." **This is GPU N:M structured sparsity (like 2:4),
not a page/cache-line-sized contiguous-block claim** — M is likely single
digits to low tens of channels, nowhere near our ≥48KB requirement. I did
not obtain the actual value of M used in their experiments; **[X] exact M
value not found** in what I fetched.

**Training cost, continue-training setting — [A], Sec. 4.2:** Model
Mistral 7B, "40B on FineWeb-Edu dataset" tokens, batch size 4M tokens,
LR 5e-5. **[X] hardware/GPU type not found** anywhere the fetch reached.
Training-from-scratch setting (Sec. 4.1, Figures 5-6, not a table): 40%
overall sparsity matching dense loss at 700M and 7B scales — figure-only,
not tabulated, so **[A]** not [T].

**Achieved sparsity and quality — [T] Table 1 (per fetch transcription),
continue-training on Mistral 7B:**

| Model | Activated Params | ARC | HellaSwag | MMLU | Winogrande | TruthfulQA | Avg |
|---|---|---|---|---|---|---|---|
| Dense baseline | 7.0B | 61.8 | 81.4 | 59.8 | 77.5 | 42.7 | 64.6 |
| Q-Sparse | 3.8B | 60.5 | 80.7 | 58.0 | 75.9 | 43.5 | 63.7 |
| Q-Sparse | 2.9B | 59.0 | 79.0 | 55.6 | 74.0 | 41.0 | 61.7 |

Numerator/denominator for "≈46% sparsity": 3.8B active / 7.0B dense
(Activated-Params column, Q-Sparse row ÷ Dense-baseline row) = 54% of
params active, i.e. ~46% sparse — this is **my arithmetic on their table
cells**, not a sparsity % they printed directly in this row (contrast
with ProSparse/Turbo Sparse which print sparsity % directly). Avg accuracy
64.6→63.7 at ~46% sparsity (0.9-point drop, same "Avg" column); 64.6→61.7
at ~59% sparsity (2.9-point drop). **[T] Table 2** (per fetch): per-
projection sparsity breakdown for the 3.8B row — QKV 42%, Output 40%,
Up 40%, Gate 40%, Down 60.4%, Overall 45.7% — note Down-projection
sparsifies furthest, echoing the same asymmetry ReLU-Strikes-Back found
for DownProj.

**Sparsity structure (outside the Block Q-Sparse batching variant):**
element-wise/scattered by default — "Top-K selection operates independently
per token on activation tensors" **[A]**; Table 2's per-projection values
differing 40-80% is offered as evidence of non-uniform, non-block
application in the base method.

**Speedup.** **No measured wall-clock or FLOP speedup number found** in
this paper per the fetch — efficiency claims are argued qualitatively
("sparsity can reduce ... computation" and "reduce I/O transfer") but not
benchmarked. **[X]** for any speedup figure. This is a real gap in the
paper, not a fetch failure — worth treating Q-Sparse as a training
mechanism only, with zero engine-relevant speedup evidence of its own.

---

### R-Sparse (Zhang, Liu, Tian, Khaitan, Wang, Li), arXiv:2504.19449, HTML v1 read at arxiv.org/html/2504.19449v1

**Mechanism.** Not neuron-threshold-based at all — a different family.
Per abstract (verbatim): "the non-sparse components of the input function
can be regarded as a few bias terms" and "the full computation can be
effectively approximated by an appropriate combination of input channels
and weight singular values" — R-Sparse replaces a linear layer's full
matmul with a rank-aware sparse approximation using a subset of input
channels plus SVD components of the weight matrix, explicitly to avoid
needing to *predict* which output channels will be active (the failure
mode of prior top-K/threshold methods at inference time). This is
architecturally distinct from ReLU-family thresholding — closer to a
low-rank + sparse decomposition applied post-hoc to a dense donor.

**Is it really training-free? — Adversarial check, per the brief's
instruction.** No weight retraining occurs, but there IS a calibration
step: (a) "we perform SVD on the pretrained weight matrix" **[A]**, Sec.
3.4 — a per-model, per-layer offline decomposition; (b) threshold
estimation: "Given a pre-defined sparsity budget s, the threshold t(s) is
estimated as the s-th percentile of X" **[A]**, i.e. a data-dependent
percentile computed from activations; (c) an evolutionary hyperparameter
search: "approximately one hour on a single A6000 GPU for the Llama-2-7B
model" using "16 randomly selected samples from the C4 training set"
**[A]**, Sec. 3.5. **Verdict on the brief's adversarial question: this is
training-free in the sense of "no gradient-based weight updates," but it
is NOT calibration-free** — it needs an SVD pass (cost scales with weight
matrix size, non-trivial at 100B), a percentile threshold fit to a
specific sparsity target, and a ~1 A6000-GPU-hour evolutionary search per
model at 7B (unknown scaling to 100B — the search likely does not scale
linearly, since it is presumably searching over per-layer sparsity
allocation, but I did not find a stated complexity class — **[X]**).

**Achieved sparsity and quality — [T] Table 1 (per fetch), common-sense
reasoning suite, 50% model-level sparsity:**

| Model | Baseline avg | R-Sparse 50% avg | Tasks |
|---|---|---|---|
| Llama-2-7B | 65.88 | 64.06 | WG, PIQA, SciQ, OBQA, HS, BoolQ, Arc-E, Arc-C (8 tasks) |
| Llama-3-8B | 69.44 | 66.20 | same 8 tasks |
| Mistral-7B | 69.89 | 68.39 | same 8 tasks |

Drop at 50% sparsity: Llama-2-7B 1.82 points (65.88→64.06, same Avg
column), Llama-3-8B 3.24 points (largest), Mistral-7B 1.50 points
(smallest). Per-task detail per fetch: SciQ near-full retention, OBQA
worst single-task drop (~3.2 points) — **[A]**, I did not get the raw
per-task cell values transcribed verbatim, only this summary, so treat
the per-task claim as [A] not [T] even though the aggregate row is [T].
No 13B-scale row exists in the main table — **[X]** for any datapoint
above 8B in this paper.

**Sparsity structure.** Channel-level, structured — verbatim: "We omit
unnecessary columns corresponding to input channels with zero values.
Additionally, the weights should be stored in a column-major format"
**[A]**, Sec. 3.4. This is explicitly framed around contiguous-column
memory access, which is the closest alignment with our block-contiguity
requirement of anything in Q1 so far — but it is column-of-a-weight-
matrix contiguity in a GPU/Triton kernel context, and I have no evidence
on what column byte-width results, or whether it would clear a 48KB
CPU-cache-line-run threshold.

**Speedup — [T]/[A] mixed, Sec. 4.3, Figure 6:** Hardware = single
NVIDIA A6000, "Customized Triton kernel", FP32, prompts 2048 tokens,
generation 128–2048 tokens. Exact quote: "R-Sparse achieved up to 42% and
40% improvements in generation speed for Llama-2-7B and Llama-3-8B,
respectively" **[A]**, Sec. 4.3 — this is a **measured wall-clock
generation-speed number**, not FLOPs, but note the abstract's headline
"43%" does not exactly match the body text's "42%" for the same model
(Llama-2-7B) — a small (1-point) internal inconsistency between abstract
and Sec. 4.3, flagged per the governing rule rather than silently
resolved in the paper's favor. All on A6000 (Ampere, has TF32/bf16 tensor
cores), not Turing/T4 — **[X]** no Turing-class datapoint.

---

### Sparsing Law (Luo et al.), arXiv:2411.02335, HTML v2 read at arxiv.org/html/2411.02335v2 — newer (2025) work, added under the brief's "search 2025-2026" instruction

**Why it's here.** This is the one paper in the set that directly studies
whether activation-sparsity *cost/limit* changes with model scale — the
central open question for taking this to 100B. It is **from-scratch
training only**, not donor conversion, so it bounds a different question
than ours, but the scale-insensitivity finding is relevant background.

**Metric.** CETT-PPL-p%: "the sparsity ratio measured by CETT when the PPL
on validation data rises by p% compared with the dense setting" **[A]**,
using p=1% as their standard operating point (binary search for the CETT
threshold hyperparameter that produces exactly a 1% relative PPL
increase). This is a different sparsity accounting than the "% of
FLOPs/params active" numbers used by the Q1 papers above — comparing
Sparsing Law's sparsity % directly against ReLU-Strikes-Back/ProSparse/
Turbo-Sparse numbers would be comparing different metrics; **flagging
this as a unit-choice trap**, per the project's own standing "Unit Choice"
guard.

**Scale-insensitivity finding — [T] Table 2 / Figure 8, per fetch:**

| Model size | Limit activation ratio A₀ |
|---|---|
| 0.1B | 6.14×10⁻² |
| 0.4B | 6.90×10⁻² |
| 1.2B | 7.82×10⁻² |
| 2.4B | 6.48×10⁻² |

Quote: "the limit of activation ratio ... is weakly related to the
parameter scale" **[A]** — across a 24x parameter range the limit
activation ratio moves non-monotonically within roughly 6.1-7.8×10⁻²,
i.e. no clear trend up or down with scale in this range. **This result,
if it holds, is favorable to a 100B donor project in one narrow sense**:
it suggests the *ceiling* on achievable sparsity is not obviously worse
at large scale. It says nothing about the *cost* (tokens) to reach that
ceiling from a dense-pretrained-with-SiLU starting point, which is the
actual open question for us — and their own experiments only go to 2.4B,
2 orders of magnitude below our 100B donor target, so extrapolation is
mine, not theirs, and unverified.

**Training cost — [A], text-only:** for the 2.4B model reaching 93.52%
sparsity: "With ReLU activation, near 800B training tokens ... achieving
a high limit sparsity ratio of 93.52%" — this is a **from-scratch** run,
not a donor conversion; **[X] no donor-conversion experiment exists in
this paper at all** — I checked explicitly and found "no discussion" of
converting an existing dense checkpoint anywhere in the fetched text.
Hardware/cluster size: **[X] not found** (Appendix I has hyperparameters
only, per fetch).

**Sparsity structure.** Scattered/unstructured, per fetch — "Neurons are
identified individually per layer; no structured patterns enforced"
**[A]**, and Figure 1 is described as showing dispersed zero/low
activations, not blocks.

**Relevance/caveat.** 800B tokens for a 2.4B model to reach ~93.5%
sparsity, from scratch, is a **much larger token budget than any
donor-conversion paper above** (ProSparse: 34.6-134B tokens at 7-13B;
Turbo Sparse: 150B tokens at 7-47B). This is the strongest evidence in
this dossier that **training sparsity in from scratch is far more
expensive, per-parameter, than inducing it via continued pretraining on a
donor** — consistent with the project's premise that donor conversion is
the economical path, though I'd caveat that Sparsing Law's own goal was
studying scaling laws, not minimizing conversion cost, so this is not a
controlled from-scratch-vs-donor comparison and I would not treat the
"donor conversion is cheaper" conclusion as proven by this pairing alone.

---

## Q2: Training-free methods (TEAL, CATS)

### TEAL (Training-Free Activation Sparsity in LLMs), arXiv:2408.14690, HTML v2 read at arxiv.org/html/2408.14690v2

**Mechanism.** Magnitude threshold applied across **all hidden states
throughout the model**, not just the FFN gate (this is broader scope than
CATS below). Per-matrix sparsified forward pass: Ŷ = s_tp(x)·Wᵢᵀ where
s_tp(xᵢ) = 0 if |xᵢ| ≤ tp, else xᵢ **[A]**, i.e. a hard magnitude gate on
the *input* activations to each weight matrix, applied model-wide (QKV,
O, gate, up, down all get their own threshold).

**Adversarial check on "training-free" — per the brief's instruction.**
Requires an offline calibration pass: "we estimate t_p using an empirical
distribution constructed offline using activations from generic text"
**[A]**. Calibration set size, per fetch: **"10 samples" of "length
2048"** (Sec 4.3) — this is a tiny calibration set (20,480 tokens total),
and cost is stated directly: "less than one GPU-hour on an A100 for
Llama-3-8B" **[A]**. **Verdict: genuinely cheap relative to any of the
Q1 methods — 1 GPU-hour vs. tens-of-billions of tokens — but it is not
zero-calibration; it is a per-model, per-channel percentile fit**, and no
gradient step touches the weights. This is a materially different cost
class than everything in Q1.

**Results — [T] Table 1 (WikiText perplexity), per fetch:**

| Model | Method | Sparsity | Perplexity |
|---|---|---|---|
| Llama-3-8B | Baseline | 0% | 5.87 |
| Llama-3-8B | TEAL | 25% | 5.94 |
| Llama-3-8B | TEAL | 40% | 6.21 |
| Llama-3-8B | TEAL | 50% | 6.67 |
| Llama-2-7B | TEAL | 40% | 5.22 |
| Mistral-7B | TEAL | 40% | 5.13 |

Note: the Llama-2-7B and Mistral-7B rows in this transcription have no
dense-baseline PPL cell alongside them in what I fetched, so I can only
compute a degradation ratio for Llama-3-8B: 5.87→6.67 at 50% sparsity is
a **13.6% relative PPL increase** (numerator = 6.67−5.87=0.80, denominator
= 5.87, both from the "Perplexity" column) — non-trivial, and notably
worse than the ~1% PPL-increase operating point Sparsing Law uses to
define its own "limit sparsity" metric, though the two are not measuring
the same intervention (training-free post-hoc threshold vs. trained-in
sparsity) so this is not an apples-to-apples comparison, just a flag that
TEAL's own reported degradation at 50% is not small on this metric.

**[T] Table 2 (downstream, avg over 6 tasks), per fetch:**

| Model | Baseline | TEAL 40% |
|---|---|---|
| Llama-3-8B | 68.07 | 66.21 |
| Llama-2-7B | 56.50 | 55.45 |

Drop: 1.86 points (Llama-3-8B), 1.05 points (Llama-2-7B), both same "avg"
column, baseline row minus TEAL row.

**Sparsity structure.** Per-channel, described in the fetch as "scattered
per-channel, not contiguous" **[A]**, though the paper does implement
column-major weight storage plus "memory coalescing and contiguous memory
access" for the *selected* channels within a layer **[A]** — i.e. once
the mask is computed, the channels that survive are read contiguously
from a column-major layout, but the *set* of which channels survive is
input-dependent per token and not block-structured a priori. This is a
GPU coalescing framing, not evidence about CPU cache-line-sized runs.

**Speedup — [T] Table 3 ("Single-batch end-to-end decoding, tokens/sec"),
per fetch:**

| GPU | Model | 40% sparsity | 50% sparsity |
|---|---|---|---|
| A100 | Llama-3-8B | 1.25x | 1.33x |
| A100 | Llama-2-7B | 1.31x | 1.40x |
| A6000 | Llama-2-7B | 1.53x | 1.78x |

Quote: "wall-clock speed-ups of up to 1.53x and 1.8x at 40% and 50%
sparsity respectively" **[A]** — measured wall-clock via specialized
sparse GEMV kernels, on Ampere GPUs (A100/A6000), not FLOP-estimated. No
CPU number anywhere.

---

### CATS (Contextually-Aware Thresholding for Sparsity — Lee, Lee, Zhang, Tiwari, Mirhoseini), arXiv:2404.08763, HTML v4 read at arxiv.org/html/2404.08763v4

**Mechanism.** Threshold applied specifically to the **SiLU gate output**
of the SwiGLU FFN (narrower scope than TEAL, which is model-wide).
Equation 5, verbatim per fetch: "CATS_t(SiLU(xW_gate)) = SiLU(xW_gate) if
|SiLU(xW_gate)| ≥ t; 0 if |SiLU(xW_gate)| < t" **[A]** — a hard magnitude
gate on the post-activation gate value, zeroing the corresponding FFN
neuron's contribution when the gate is small.

**Adversarial check on "training-free."** Calibration required:
"We compute the activations over a random subset of the training data,
limited to only **500 data points**" **[A]**, Sec 4.1, Stage 1 — threshold
set as the k-th percentile of activation magnitudes on this set, before
any fine-tuning. This is a larger calibration set than TEAL's 10
sequences but still tiny relative to Q1's token budgets. **Separately —
and this is the adversarial catch the brief asked for — the abstract's
headline numbers ("within 1-2%... without any fine-tuning") describe the
no-finetune configuration, but the paper's own abstract also states
"CATS-based models converge faster and display better task performance
than competing techniques when fine-tuning is applied"** **[A]** — i.e.
there IS a fine-tuning variant that the paper itself prefers when
possible, using LoRA "targeted only 1% of the parameters" on RefinedWeb
**[A]** (exact token count for this LoRA stage: **[X] not found** in what
I fetched). So CATS ships two regimes: a genuinely training-free
calibration-only mode (500 samples, no gradient step) and a
better-performing PEFT-finetuned mode that is not free — treat "CATS is
training-free" as true only for the first regime, and note the paper's
own comparative framing favors the second.

**Results — [T] Table 1 (per fetch), 50% sparsity, no fine-tuning:**

| Model | Avg |
|---|---|
| Mistral-7B (dense) | 0.6994 |
| Mistral-7B CATS 50% | 0.6890 |
| Llama2-7B (dense) | 0.6589 |
| Llama2-7B CATS 50% | 0.6433 |

Relative retention: Mistral 0.6890/0.6994 = **98.5%** of dense (numerator
= CATS row, denominator = dense row, "Avg" column); Llama2-7B 0.6433/
0.6589 = **97.6%**. **[T] Table 5 (Appendix E)**, Llama2-13B: dense
0.6870, CATS 50% 0.6805 → **99.1%** retention. These support the
abstract's "within 1-2%" framing at the aggregate level, though I did
not obtain per-task breakdowns to check whether any single task drops
much further than the average (the pattern seen in R-Sparse's OBQA
outlier above suggests this is worth checking before trusting an
aggregate).

**Sparsity structure.** Scattered/unstructured — "sparsity mask is
element-wise on activations post-SiLU... no structural constraint
enforces contiguity" **[A]**, per fetch, citing Figure 1's irregular
per-channel activation distributions.

**Speedup.** Hardware: single L40S GPU (Ada Lovelace), custom Triton
kernel (Sec 4.3 / Appendix D), measured as "geometric mean of the latency
across each round" over 50 RefinedWeb samples, batch size 1, beam width 1
**[A]**. **Discrepancy flagged directly from the fetch**: the abstract
states "~15% improvement in wall-clock inference latency ... on both
Llama-7B and Mistral-7B," but Sec 5.3/Figure 4 reports "CATS can
accelerate the generation stage by ~18% for Llama2-7B and ~21% for
Mistral-7B at 50% sparsity" **[A]** — the body-text numbers are higher
than the abstract's rounded-down headline; not a contradiction so much as
the abstract understating its own body results, but worth carrying the
higher, more specific numbers (18%/21%) rather than the abstract's "~15%"
if this is used for planning. No CPU number; all measured on GPU.

**Cross-paper note on Q2.** Both TEAL and CATS are genuinely far cheaper
than any Q1 method — GPU-hours and hundreds of samples, not billions of
tokens — which matches the brief's framing that training-free is
"enormously more valuable." Neither reports block-contiguous sparsity;
both are explicitly per-channel/per-token scattered patterns with GPU
memory-coalescing tricks layered on afterward, not a CPU-cache-line-run
guarantee. Neither paper tests below 7B or above 8B — **[X] no
donor-conversion evidence at anything near 100B for either method.**

---

## Q3: The bundling contradiction (LLM in a Flash vs Neuralink vs LLaMA-MoE)

### Source 1 — LLM in a Flash (Alizadeh, Mirzadeh, Belenko, Khatamifard, Cho, Del Mundo, Rastegari, Farajtabar — Apple), arXiv:2312.11514, HTML read at arxiv.org/html/2312.11514

**What they tried.** Section 3.2 introduces the motivation: "by storing
these corresponding columns and rows together in flash memory, we can
consolidate the data into larger chunks for reading" **[A]** — i.e. the
goal is exactly our goal, larger contiguous reads. Appendix D, "Bundling
Based on Co-activation" **[A]**, describes the concrete attempt: co-
activation statistics were measured empirically on the **C4 validation
set**, model = **OPT 6.7B**, and neurons were paired via a greedy
"closest friend" heuristic — each neuron bundled with its single most-
frequently-co-activated partner. Co-activation frequency across neurons
was found to follow **a power-law distribution** **[A]**.

**The failure, verbatim (per two independent fetches of the same
passage, consistent both times):** "Unfortunately, this resulted in
loading highly active neurons multiple times and the bundling worked
against our original intention" **[A]**, with the mechanism stated
directly: "very active neurons are the closest friends of many others"
**[A]** / "highly active neurons are the 'closest friends' of almost
everyone" **[A]** — i.e. the naive greedy nearest-neighbor pairing is
**many-to-one**: many low-activity neurons independently pick the same
few globally-hot neurons as their "closest friend," so those hot neurons
get *redundantly* stored/loaded across many different bundles, inflating
I/O rather than reducing it. No numeric bundling-degree or redundancy
percentage was surfaced by my fetch — **[X]** for a quantified failure
magnitude; the paper reports this as a qualitative negative result, not a
measured regression number.

**What shipped instead.** Fixed row-column pairing: concatenating
up-projection matrix columns with the corresponding down-projection
matrix rows into one larger contiguous chunk — a purely structural
pairing determined by the architecture (which row of Wdown corresponds to
which column of Wup for a given neuron), with **zero dependence on
measured activation data**. This sidesteps the redundancy problem
entirely because the pairing is one-to-one by construction (every neuron
has exactly one corresponding row and one corresponding column, so no
neuron can be "everyone's closest friend").

---

### Source 2 — Neuralink (Wang, Fan, Huang, Hao, Li, Cao, Lu, Zhang, Ren — Tsinghua/Microsoft Research), arXiv:2410.19274, HTML v3 read at arxiv.org/html/2410.19274v3 (accepted ASPLOS 2025; note a WebSearch pass returned a stale/mismatched title "Ripple: Accelerating LLM Inference on Smartphones..." for the v1 HTML render of this same arXiv ID — the abs page confirms the paper's actual and current title is "Neuralink: Fast LLM Inference on Smartphones with Neuron Co-Activation Linking," authors as listed above; flagging the title mismatch here so it isn't silently mistaken for a wrong-paper citation)

**Does it cite LLM in a Flash?** Yes — reference [4], full citation
matches the Apple paper exactly. But per the fetch, the citation only
credits LLM-in-a-Flash with "advanced caching strategies to reduce I/O
volume, leaving I/O bandwidth still held back by IOPS constraints" **[A]**
— i.e. it is cited as *prior art to build on*, for a different limitation
(IOPS, not bundling). **The paper does not discuss, quote, or attempt to
explain Apple's specific negative bundling result anywhere I could find.**
It frames its own contribution as "orthogonal" to prior I/O caching work
**[A]**, which is a positioning move that avoids the contradiction rather
than resolving it.

**Mechanism, and how it differs from LLM-in-a-Flash's "closest friend"
heuristic.** Neuralink formulates neuron placement as finding a shortest
Hamiltonian path in a complete graph over neurons, solved via **greedy
link-merging with a disjoint-set / priority-queue structure**: "iteratively
merge them until all neurons are connected within a single link. To
minimize the expected number of I/O operations, we employ a greedy
merging strategy" **[A]**. Structurally, this differs from LLM-in-a-
Flash's approach in one specific way I can state with confidence from what
was fetched: LLM-in-a-Flash's "closest friend" pairing lets **multiple**
neurons independently nominate the same hot neuron as their best partner
(a many-to-one selection, since each neuron picks its own best match
without checking whether that match is already taken), whereas
Neuralink's disjoint-set merge process builds a single connected
structure where **each neuron ends up in exactly one position along the
chain** — a global one-to-one constraint. **This structural difference is
my own inference from the two algorithm descriptions, not a claim either
paper makes about the other — flagging explicitly per the brief's
instruction not to fabricate a tidy story.** Neither paper states this
comparison; I built it by contrasting the two mechanism descriptions.

**Scale/setting.** Hardware: OnePlus 12 (Snapdragon 8 Gen 3, UFS 4.0),
OnePlus Ace 3 (8 Gen 2), OnePlus Ace 2 (8+ Gen 1) — smartphone flash
storage, not SSD/macOS unified memory (LLM-in-a-Flash's target). Models
and reported "sparsity" per fetch: OPT-350M (9.48%), OPT-1.3B (3.65%),
OPT-6.7B (3.17%), Llama-2-7B (23.67%), Mistral-7.8B (23.54%),
MobiLlama-1B (30%), Phi-2 (20%) **[A]**. **I could not confirm from the
fetch whether these percentages denote the active-neuron fraction or the
sparsity fraction** — the OPT numbers (3-9%) read as plausible *active*
fractions consistent with ~90-97% sparsity (matching LLM-in-a-Flash's own
OPT 6.7B regime), while the Llama-2-7B/Mistral figures (~24%) look high
for either framing on a non-ReLUfied model — **[X], ambiguous unit, not
resolved.**

**Headline speedup — [A], quoted directly, no single unified table found
by the fetch (results are per-figure, 9a/9b, not one table):** "achieving
average speedups of 2.37x, 1.48x, and 1.25x over llama.cpp, LLMFlash, and
Neuralink-S, respectively" **[A]**. Read this carefully: the 2.37x is
against **dense llama.cpp** (no sparsity exploitation at all), which
conflates the benefit of *having* activation sparsity with the benefit of
*this specific bundling scheme*. The isolated bundling contribution is
better read from the **1.48x vs "LLMFlash"** figure — but "LLMFlash" here
is presumably Neuralink's own reimplementation of Apple's fixed
row-column-pairing method inside Neuralink's evaluation harness, not a
run of Apple's original code — **[A], fairness of this specific
comparison not verifiable from what I fetched.** "Neuralink-S" is
apparently an ablation (simpler variant) internal to this paper; I did
not get its definition.

**Does either paper reconcile the contradiction?** No. Per the brief's
own honesty requirement: **I cannot reconcile these from the papers
themselves — Neuralink does not engage with Apple's negative result at
all.** The only reading under which both could be correct is the
structural one I derived above (many-to-one naive nearest-neighbor vs.
one-to-one global merge) — but that reading is unstated by either author
and unverified by any ablation isolating exactly that variable (e.g.
neither paper reports "naive closest-friend on our own model/hardware"
as a controlled baseline against their own graph-merge method — if
Neuralink had run LLM-in-a-Flash's *exact* closest-friend heuristic on
their own setup and shown it fails while their Hamiltonian-path method
succeeds, that would be a real reconciliation; they didn't, at least not
in anything I fetched). **This is a legitimate open contradiction, not a
tidy resolved one.**

---

### Source 3 — LLaMA-MoE (continual-pretraining MoE-ification of LLaMA-2-7B), arXiv:2406.16554, HTML read at arxiv.org/html/2406.16554 — EMNLP 2024

**Important correction to the brief's framing, found directly in the
text: the word "co-activation" / "coactivation" does not appear anywhere
in this paper** — confirmed by an explicit search-and-quote pass. The
method the brief characterizes as "co-activation clustering" is named
**IndependentClustering** by the authors, and per its exact verbatim
definition: "Following [Zhang et al. 2021], we perform a balanced k-means
clustering with n centroids on the row vectors of Wup and partition U
according to the clustering result" **[A]**. This is **k-means clustering
in weight space** — grouping neurons whose up-projection weight *vectors*
point in similar directions — not clustering by measured co-activation
*frequency from running data through the model*, which is what both
LLM-in-a-Flash and Neuralink actually do. **These are mechanically
different neuron-grouping strategies that happen to often correlate in
practice (weight-similar neurons are plausibly more likely to co-fire)
but are not the same operation.** I flag this as a place where the
brief's framing may be importantly off: LLaMA-MoE is not a third
independent replication of "co-activation clustering fails" — it's a
data point (see caveat below) about *weight-space* clustering
specifically, one inferential step removed from the actual co-activation
statistic the other two papers measure directly.

**Comparison method/rigor caveat.** Per the fetch, **no numeric results
table exists for this comparison** — IndependentRandom vs.
IndependentClustering vs. SharingInner vs. SharingInter are compared only
via **Figure 3's performance curves**, described in prose: "IndependentRandom
achieves the best average score within the token budget" **[A]**, and
"For SharingInner and IndependentClustering, we only train those models
for 15B tokens" **[A]** (implying early termination due to weak
performance, but no loss/accuracy number is given for either method in
what I fetched). **[X]** — I could not obtain a [T]-grade table cell for
this comparison; it is [A] at best, and the underlying evidence is a
truncated-training decision, not a fully-matched head-to-head at equal
token budget. Model/scale: LLaMA-2-7B, continual pretraining, ablation
runs in the 5B-200B token range **[A]**.

**Their stated explanation, verbatim:** "Since gates and experts are
trained simultaneously, other partition methods may bring bias when
construction, which introduces additional difficulties for recovering
the model's language ability" **[A]** — this is a claim about a
*training-dynamics* interaction between the gate network and a biased
initial partition, not a claim about I/O or memory-access efficiency at
all. LLaMA-MoE's finding, even taken at face value, is about **downstream
task quality after continued pretraining of a MoE**, not about
**memory-traffic/bundling efficiency** — a third mechanical difference
from the other two papers' concern (which is purely about I/O redundancy
at inference time, no retraining involved). Three different questions —
weight-clustering vs quality-after-retrain (LLaMA-MoE), co-activation
pairing vs I/O redundancy (LLM-in-a-Flash), co-activation linking vs
measured end-to-end latency (Neuralink) — are being treated as one
question by the "two independent negative results" framing; I don't
think that's fully sound.

---

### Synthesis for Q3 — does D0 (our co-activation clustering probe with a random-clustering control) face a real prior, and how strong is it?

**What actually replicates across sources, strictly:** Only one paper
(LLM-in-a-Flash) reports a negative result for *co-activation-frequency-
based* neuron bundling specifically, with a stated, plausible failure
mechanism (many-to-one hot-neuron collision under a naive greedy
heuristic). Neuralink reports a *positive* result for a *differently-
constructed* co-activation-based scheme (global one-to-one graph merge
instead of naive pairwise nearest-neighbor) and does not engage with
Apple's negative finding at all — so this is not two failures and one
success on the same operation; it's one documented failure of a specific
naive heuristic, one success of a more careful algorithm on the same
underlying idea (co-activation), and one loosely-related finding
(LLaMA-MoE) about *weight-space* clustering hurting *post-retraining
quality*, which is a different mechanism, different failure mode, and
different metric (quality, not I/O efficiency) from the other two.

**Honest bottom line, as the brief demands:** I cannot reconcile
LLM-in-a-Flash and Neuralink from the papers themselves — Neuralink
simply never addresses the contradiction, and I have no controlled
ablation from either paper isolating "naive pairwise vs. global merge" as
the actual explanatory variable, only my own structural inference from
comparing the two algorithm descriptions. **If our D0 probe implements a
naive greedy pairwise "closest friend" bundler, LLM-in-a-Flash's result
is a direct, on-point, previously-documented failure mode to worry about
and specifically design the random-clustering control to catch.** If D0's
co-activation method instead builds a global one-to-one structure (more
like Neuralink's), the LLM-in-a-Flash failure mode may not directly
transfer, but there is then **no primary-source positive evidence
outside Neuralink's own paper (on smartphone flash, at 7B and below) that
this generalizes** — and no evidence at all, from any of the three
papers, at donor scales anywhere near our 1B-100B CPU target. LLaMA-MoE's
negative result is real but targets a different axis (post-hoc weight
clustering degrading continued-pretraining quality) and should not be
read as reinforcing the I/O-bundling question either way.

---

## Q4: Activation sparsity + ternary/2-bit weight quantization, same model

**The literature is not silent here** — three primary sources found, two
directly on point (BitNet a4.8, Sparse-BitNet), one background (Q-Sparse's
own BitNet b1.58 section, already partly covered in Q1). None of the
three tests a pretrained-dense-SiLU **donor converted post-hoc**; all
three train the combination in from the start or continue-train from an
already-ternary checkpoint. This is a real gap relative to our exact
use case (dense donor → ternary + sparse, no from-scratch budget).

### BitNet a4.8 (Wang, Ma, Wei et al. — Microsoft), arXiv:2411.04965, HTML read at arxiv.org/html/2411.04965 — the most directly on-point source

**Mechanism.** A hybrid precision/sparsity scheme layered on top of
ternary (1.58-bit) weights: attention-layer and FFN-layer **inputs** get
4-bit quantization; the FFN's intermediate state (pre-down-projection)
and the attention output-projection input get **8-bit + sparsification**
instead of 4-bit, using "the sparsification method from Q-Sparse to
retain these intermediate states at 8 bits while removing the computation
bottleneck" **[A]**. Output projection specifically: "a sparsify-then-
quantize function" with **50% top-K masking** **[A]**. FFN sparsity is
additionally boosted by "squared ReLU and gated linear unit (GLU)"
**[A]** — i.e. an activation-function change (not unlike ReLU-Strikes-
Back/Turbo-Sparse) stacked with the top-K/threshold mechanism, on top of
ternary weights throughout.

**Achieved sparsity — [T] Table 2 (7B model, non-embedding params), per
fetch:**

| Component | Sparsity |
|---|---|
| FFN down-projection input | 84.2% |
| Up-projection | 71.4% |
| Gate-projection | 67.5% |
| Attention output-projection | 50.0% |
| Overall | 44.5% (3.4B active of 6.5B non-embedding params) |

**Quality — [T] Table 1, verbatim per fetch:**

| Model | Size | PPL | ARC-c | ARC-e | HellaSwag | PIQA | WinoGrande | Avg |
|---|---|---|---|---|---|---|---|---|
| LLaMA FP16 | 7B | 9.20 | 33.36 | 51.22 | 58.33 | 73.34 | 58.41 | 54.93 |
| BitNet b1.58 (ternary, no extra sparsity) | 7B | 9.24 | 32.00 | 50.88 | 59.79 | 72.96 | 59.83 | 55.09 |
| BitNet a4.8 (ternary + 44.5% activation sparsity) | 7B | 9.37 | 31.66 | 50.88 | 58.78 | 73.01 | 59.35 | 54.74 |

**This is the single most important number in this dossier for Q4.**
Going from ternary-only (b1.58, PPL 9.24, Avg 55.09) to ternary+44.5%-
sparse (a4.8, PPL 9.37, Avg 54.74) costs **+0.13 PPL and −0.35 points Avg**
(both cells same columns, adjacent rows) — a small, not-zero, but modest
additional cost for adding substantial activation sparsity on top of an
already-ternary model. **Reading this as "sparsity composes cheaply with
ternary quantization" is supported by this table**, though see the
caveat below on training regime.

**Training cost — [A], text-only:** 700M/1.3B/3B/7B model sizes; "Total
training tokens: 100B for main experiments" with a **two-stage** split —
"95B tokens at 8-bit activations, then 5B tokens adapting to 4-bit"
**[A]** — and explicitly **"continued training from BitNet b1.58
checkpoint"** **[A]**, RedPajama dataset. A separate scaling check used
"a 2B model trained on 2T tokens" (Table 5, per fetch, not transcribed
here) — that is a from-scratch-scale token budget, not continued-training,
and I did not pull its numbers. **Caveat that matters a lot for our use
case: this is continued training from an already-ternary BitNet b1.58
checkpoint, not from a dense fp16 donor.** The paper does not test
"take a dense SiLU/GELU donor and induce both ternary weights and
activation sparsity simultaneously" — the ternarization already happened
in a prior stage (BitNet b1.58's own training), and this paper only adds
sparsity on top of an already-ternary base. **[X]** for any dense-donor
→ ternary+sparse joint conversion evidence.

**Explicit statement on composition — [A]:** "BitNet a4.8 achieves
performance comparable to BitNet b1.58 with equivalent training costs,
while being faster in inference" **[A]**. Per the fetch, **the paper does
not provide an ablation isolating the interaction effect** — i.e. no
controlled comparison of "sparsity alone at 44.5%, on a full-precision
model of the same size and token budget" vs. "sparsity alone, on ternary"
to show whether ternary weights make sparsity *easier* or *harder* to
add, or vice versa. The only comparison available is ternary-vs-
ternary+sparse (the table above); a fully controlled 2×2 (precision ×
sparsity) ablation is **[X] not present**.

---

### Sparse-BitNet (Zhang, Wu, Huang, Wang et al. — Microsoft Research, Peking University, South China University of Technology), arXiv:2603.05168, HTML v1 read at arxiv.org/html/2603.05168v1 — very recent (March 2026), addresses WEIGHT sparsity not activation sparsity, included because it directly tests the ternary-changes-sparsity-tolerance question

**Scope caveat, stated up front:** this paper's sparsity is **N:M
semi-structured WEIGHT pruning**, not activation sparsity — a different
axis than the rest of this dossier. I include it because it is the
clearest primary-source answer to "does ternary weight quantization make
a model more or less tolerant of *some* form of induced sparsity,"
which bears indirectly on our question even though the sparsity type
differs from what our engine exploits.

**Mechanism.** "Semi-structured N:M sparsity enforces a fine-grained
pattern where at most N elements are non-zero out of every M consecutive
weights" **[A]** — this IS block-contiguous by construction, block size
M=8 in their main configuration (6:8, i.e. keep 6 of every 8 consecutive
weights, 25% sparse) plus a 2:4 stress-test config (75% sparse) **[A]**.
Masking is magnitude-based, computed on **pre-quantized master weights**
to preserve fine-grained ranking before the ternary rounding destroys
magnitude information — a specific technical point about ordering the
two operations that would transfer directly to any donor pipeline
attempting both together.

**Composition result — [T] Table 2 ("Perplexity at 6:8 Sparsity"),
verbatim per fetch:**

| Method | 0.5B | 1.5B | 3B |
|---|---|---|---|
| Dense BF16 | 21.91 | 18.10 | 16.03 |
| Sparse BF16 (6:8) | 23.11 (+1.20) | 18.70 (+0.60) | 16.48 (+0.45) |
| Dense BitNet (ternary) | 25.99 | 20.11 | 17.70 |
| Sparse BitNet (6:8, ternary) | 26.31 (+0.32) | 20.35 (+0.24) | 17.87 (+0.17) |

Reading the ratios directly from these cells: BF16's sparsity penalty at
0.5B is 1.20/21.91 = **5.5% relative PPL increase** (numerator = Sparse
BF16 minus Dense BF16 delta, denominator = Dense BF16, both "0.5B"
column); BitNet's penalty at the same scale is 0.32/25.99 = **1.2%
relative** — **BitNet's sparsity penalty is roughly 4-5x smaller in
relative terms than full-precision's, consistently across all three
scales tested (1.2% vs 5.5% at 0.5B; 1.2% vs 3.3% at 1.5B [0.60/18.10];
1.1% vs 2.8% at 3B [0.45/16.03] — all my arithmetic on their Table 2
cells).** This is a genuine, table-sourced, directly-relevant finding:
**ternary weights tolerate structured sparsity better than full precision
does, on this paper's own numbers, at up to 3B scale.** Whether this
generalizes to 100B, and whether weight-sparsity-tolerance transfers to
activation-sparsity-tolerance, are both unverified extrapolations — flag
both explicitly as **not tested by this paper**.

**Training — [A]:** Qwen2.5-0.5B/1.5B/3B, "trained on RefineWeb data for
approximately 50B tokens per model" **[A]**, **from scratch** ("train all
model sizes ... from scratch under the same data mixture and token
budget" **[A]**) — again not a donor-conversion setting. Hardware for
training: **[X] not specified** per fetch (only inference-benchmark
hardware — A100 for prefill, B200 for decode — was given).

**Speedup — [T] Table 3 (Qwen2.5-3B), per fetch:** up to **1.30x** prefill
speedup on A100 at 65K sequence length, up to **1.18x** decode speedup on
B200 at batch size 128, via a custom in-house 6:8 sparse operator,
measured (not FLOP-estimated) throughput. Modest relative to the Q1/Q2
activation-sparsity speedups (2-4x range) — consistent with 6:8 being
only 25% weight sparsity, a much lower sparsity target than any Q1/Q2
activation method.

---

### Cross-source synthesis for Q4

Two real data points, both showing **ternary weights and induced
sparsity compose without catastrophic interaction**: BitNet a4.8 adds
44.5% activation sparsity onto ternary weights for +0.13 PPL / −0.35 avg
points at 7B; Sparse-BitNet shows ternary weights absorb 6:8 structured
weight sparsity with 3-5x smaller relative PPL penalty than full
precision at up to 3B. **Neither paper demolishes the other's technique**
— this is evidence *against* the null hypothesis that ternary+sparse
would be a destructive combination, at least for weight-sparsity
(Sparse-BitNet, N:M block-structured, directly relevant to our
contiguity requirement) and for one specific activation-sparsity recipe
(BitNet a4.8, top-K/threshold-based, continued-trained from an
already-ternary base).

**What is still genuinely missing, stated plainly per the brief's
instruction not to paper over gaps:** no source tests (a) inducing BOTH
ternary weights AND activation sparsity starting from a dense,
full-precision, SiLU/GELU-activated donor in a single conversion pass or
short pipeline — both papers start from an already-specialized base
(BitNet a4.8 from an already-ternary checkpoint; Sparse-BitNet trains
ternary+sparse jointly from random init, never from a dense donor); (b)
any scale near our 1B-100B donor range combined with ternary+sparse
jointly (largest joint datapoint is BitNet a4.8 at 7B, continued-trained,
not donor-converted); (c) activation sparsity specifically (not weight
sparsity) composed with ternary at a scale or in a donor-conversion
setting relevant to us — BitNet a4.8 is continued-training on top of an
already-ternary model, the closest analog to "donor conversion" in this
set, but the donor there is already ternary, not full-precision. **On
the exact question "does inducing activation sparsity in a full-
precision-pretrained dense donor, while ALSO ternarizing its weights,
compose or destroy," the literature is silent — [X] for that specific
composite setting.**

---

## Token-budget table (my conversion, not the papers')

**Method and assumptions — stated explicitly, all of this is MY
arithmetic, not any paper's:**

- Training FLOPs estimated as 6·N·D (standard dense-transformer forward+
  backward approximation), where N = model params actually receiving
  gradient updates in the paper's own run, D = tokens processed in that
  run. This is a coarse approximation and does not account for
  architecture-specific overhead (attention FLOPs, MoE routing, etc).
- **Cross-check**: applying this formula to ProSparse's own stated
  compute (7B model, 34.6B tokens, "8 A100 80GB GPUs for about 10 days")
  implies an aggregate throughput of ~210 TFLOP/s across 8 A100s (~26
  TFLOP/s per A100, ~8% of A100's 312 TFLOPS bf16 peak) — a plausible,
  if unremarkable, real-world MFU for a training run with communication
  and attention overhead. This gives me reasonable confidence the 6ND
  approximation is within roughly the right order of magnitude for this
  purpose, though it is not exact.
- **T4 throughput assumption**: T4 (Turing SM 7.5, no TF32) fp16 tensor-
  core peak ≈ 65 TFLOPS. Assuming a real-world training MFU of ~15%
  (older architecture, 320GB/s memory bandwidth vs. A100's ~2TB/s, and
  the project's own standing note that `--compile` is dead on this class
  of setup) gives **≈10 TFLOP/s sustained per T4**, **≈20 TFLOP/s
  aggregate for 2xT4**. This is a rough, stated assumption, not measured
  — treat all hour figures below as ±2-3x uncertain at minimum.
- **Extrapolation to 1.5B and 100B donor sizes**: where a paper gives a
  tokens-to-params ratio r=D/N at one scale, I hold r constant and scale
  D linearly with the target N to get an estimate at 1.5B/100B. This is
  an assumption, not a finding — **where a paper gives r at two
  different scales and the ratios disagree (ProSparse: r≈4.9 at 7B vs.
  r≈10.3 at 13B), that disagreement itself suggests token cost may scale
  super-linearly with parameter count for at least one method, which
  would make my 100B row an underestimate.** I flag this per-row.
- **Budget anchor**: the brief states 90 GPU-hours/week on 2xT4, ~2
  months acceptable for a 100B donor. Reading "90/week" as combined
  across both GPUs: 2-month budget ≈ 90 × 8.7 weeks ≈ **~780 combined
  GPU-hours available**. Every row below is compared against this.

| Method | Native run (paper's own N, D) | Native FLOPs (6ND) | My est. hours, 2xT4 aggregate, native scale | My est. hours, 2xT4, **1.5B donor** | My est. hours, 2xT4, **100B donor** | Fits ~780h budget? |
|---|---|---|---|---|---|---|
| ReLU Strikes Back (stage 1 only) | 7B, 30B tok (r≈4.29) | 1.26×10²¹ | ~17,500h | ~800h | ~3.6×10⁶h | 1.5B: borderline/no; 100B: no, by ~4600x |
| ProSparse (7B-scale r≈4.94) | 7B, 34.6B tok | 1.45×10²¹ | ~20,100h | ~930h | ~4.1×10⁶h | No at either size |
| ProSparse (13B-scale r≈10.3, shows super-linear cost) | 13B, 134.2B tok | 1.05×10²² | ~145,600h | ~1,600h (if r held at 1.5B) | ~8.3×10⁶h | No, worse than 7B-ratio row |
| Turbo Sparse / dReLU (r≈21.4) | 7B, 150B tok | 6.30×10²¹ | ~87,500h | ~4,000h | ~1.8×10⁷h | No at either size |
| Q-Sparse continue-train (r≈5.71) | 7B, 40B tok | 1.68×10²¹ | ~23,300h | ~1,070h | ~4.8×10⁶h | No at either size |
| Sparsing Law (from scratch, r≈333, anchor only — not donor conversion) | 2.4B, 800B tok | 1.15×10²² | ~160,000h (≈18.3 yr) | n/a (already sub-1.5B-adjacent) | not extrapolated (from-scratch cost this high even at 2.4B) | No, wildly infeasible even at native 2.4B scale |
| BitNet a4.8 continue-train (r≈14.3; NB starts from already-ternary checkpoint, not a dense donor) | 7B, 100B tok | 4.20×10²¹ | ~58,300h | ~2,680h | ~1.2×10⁷h | No at either size |
| Sparse-BitNet (from scratch; **1.5B row uses their actual 1.5B/50B-token run, not extrapolated**) | 1.5B, 50B tok (real datapoint) | 4.5×10²⁰ | n/a | **~6,250h (real, not extrapolated)** | ~1.4×10⁷h (r≈16.7 held) | No at either size |
| **TEAL** (training-free, calibration only) | Llama-3-8B, 10 sequences × 2048 tok, "<1 A100-hour" | n/a (not a training FLOP regime) | ~1-2h (rough GPU-class conversion, A100→T4 ~2-3x slower) | ~1-3h (calibration cost is ~model-size-dependent, not token-budget-dependent) | ~10-30h (speculative extrapolation of calibration cost to 100B; **not measured by the paper at any scale near this**) | **Yes, easily, at any tested scale — 100B row is my own extrapolation and unverified** |
| **CATS** (training-free, calibration only) | 500 data points, no gradient step | n/a | well under 1h | well under 1h | speculative, likely still small (single-digit hours) but **unverified above 7B** | **Yes, easily — same 100B caveat as TEAL** |
| **R-Sparse** (training-free but requires SVD + evolutionary search) | Llama-2-7B, "~1 A6000-GPU-hour" search + SVD | n/a | ~2-3h (T4-adjusted) | ~1-2h | SVD cost scales with weight-matrix size (likely superlinear near 100B; evolutionary search cost unclear) — **[X] not estimable from what the paper reports** | Likely yes for the search itself; SVD-at-100B cost genuinely unknown |

**Headline reading of this table**: every training-requiring method in
Q1 is **wildly outside the stated 90-GPU-hour/week, 2-month, 2xT4 budget**
— by roughly 3-4 orders of magnitude even at the cheapest continued-
pretraining ratios, worse at 100B, and catastrophically worse for any
from-scratch method (Sparsing Law, Sparse-BitNet). **Only the training-
free calibration methods (TEAL, CATS, R-Sparse) plausibly fit the stated
budget at any scale, including a 100B donor** — but none of them has
been tested by their authors anywhere near 100B, so the 100B column for
all three is a genuinely unverified extrapolation on my part, not
theirs, and R-Sparse's SVD step in particular has an unknown-but-
plausibly-large cost at that scale. **This table is the strongest single
argument in this dossier for treating "training-free" as the only
literature-supported path within budget, IF a training-free method can
be shown to reach block-contiguous, not merely scattered, sparsity —
which as noted in Q2, none of TEAL/CATS/R-Sparse currently demonstrate.**

---

## Single-source / unverified claims

This list collects claims used above that rest on exactly one source, or
that I could not independently cross-check, or that carry an internal
inconsistency I could not resolve. None of these should be treated as
settled facts.

1. **ReLU Strikes Back's Stage-1/Stage-2 token counts (30B vs. 50B) are
   internally inconsistent** in the text I fetched, and I could not
   resolve which applies to which model/stage. Everything downstream of
   this (my token-budget table row) inherits that uncertainty.
2. **ProSparse's own token count for the 7B model has two different
   values in the paper** (34.60B per the Appendix table, 54.53B per an
   in-text sentence framed as "2.73% of 2T LLaMA2 tokens") — unreconciled.
3. **Whether ProSparse's sparsity is block-structured or scattered** —
   my fetch described it as scattered/per-element, contradicting the
   "rows or columns" framing ReLU-Strikes-Back uses for a mechanically
   similar method. I did not obtain a figure or diagram to adjudicate
   this directly; treat as single-fetch, unconfirmed.
4. **Turbo Sparse and Q-Sparse's dense-baseline comparability** — both
   papers report their sparsified model beating the dense baseline on
   average benchmark score, but I could not confirm whether the dense
   baseline rows are the authors' own token-matched continued-trained
   controls or the stock released checkpoints. If the latter, "sparsity
   is free" is confounded with "extra fine-tuning helps," and the
   reported gains would not isolate the sparsity mechanism's effect.
5. **Q-Sparse's "Block Q-Sparse" M value** (the block size in their N:M-
   style batching scheme) — not found in what I fetched. I also could
   not explain why the term is entirely absent from the v1 HTML render
   despite being in the abstract since v1; I treated this as a rendering
   gap and pulled the definition from v3 instead, but I have not
   confirmed the definition is unchanged between v1 and v3.
6. **R-Sparse's per-task breakdown (SciQ near-full retention, OBQA
   worst-case drop)** — reported to me only as prose summary, not
   verbatim table cells; downgraded to [A] even though the aggregate
   row is [T].
7. **CATS's LoRA fine-tuning-variant token count** — not found; the
   paper's preference for the fine-tuned variant over its own
   training-free headline result is noted but not quantified.
8. **The arXiv listing anomaly for Neuralink (2410.19274)** — one
   WebSearch pass returned a page titled "Ripple: Accelerating LLM
   Inference on Smartphones with Correlation-Aware Neuron Management"
   for what should be the same v1 HTML render; the abs page and PDF
   title both confirm "Neuralink" as the paper actually at this arXiv
   ID. I believe this is a search-index artifact (possibly a stale
   cache of an early preprint title, or an indexing collision with a
   different paper on an adjacent topic), not evidence of wrong-paper
   citation, but I did not fully run this down and flag it explicitly
   so it isn't silently trusted.
9. **My own structural explanation for why Neuralink's bundling might
   succeed where LLM-in-a-Flash's failed** (many-to-one naive nearest-
   neighbor vs. one-to-one global graph merge) is **entirely my
   inference from comparing two algorithm descriptions — neither paper
   states this comparison, and I found no ablation in either paper that
   isolates it.** This is clearly labeled inline above but repeated here
   because it is the single most load-bearing piece of reasoning in the
   Q3 section and the easiest to mistake for a sourced claim on a
   skim.
10. **Neuralink's reported "sparsity" percentages per model** (OPT-350M
    9.48%, Llama-2-7B 23.67%, etc.) — I could not confirm whether these
    denote active-fraction or sparsity-fraction; flagged as [X] inline,
    repeated here because it affects how directly comparable Neuralink's
    setup is to the other Q1 papers' sparsity regimes.
11. **LLaMA-MoE's IndependentRandom vs. IndependentClustering comparison
    has no numeric table cells** anywhere I could find — the entire
    finding rests on prose description of Figure 3 plus the fact that
    two methods were cut short at 15B tokens. This is real information
    but it is [A]-grade at best; do not treat "IndependentRandom wins"
    as [T]-grade evidence the way the brief's framing implicitly does.
12. **BitNet a4.8's Table 5 (2B model, 2T tokens scaling check)** was
    named in the fetch but its numbers were not transcribed; I did not
    use it in the token-budget table and flag it as an unexplored
    thread if a later pass wants a from-scratch BitNet+sparsity anchor
    at larger scale.
13. **All PDF-only fetches failed outright** (Turbo Sparse's original
    PDF, R-Sparse's PDF, CATS's PDF, BitNet a4.8's PDF) — every one of
    these was recovered via a fallback to the arXiv HTML/ar5iv render
    instead. This means every number in this document ultimately passed
    through an HTML-rendering + AI-summarization step (the WebFetch
    tool's internal small model), not a raw-pixel table read by me
    directly. I treated fetch outputs that included exact cell values,
    row/column headers, and captions as [T]-grade per the governing
    rule's spirit, but I want to be explicit that "read in the paper's
    own table" here means "read via a tool-mediated HTML transcription,"
    not a PDF screenshot I inspected myself. Where a fetch gave me only
    prose summary without cell-level detail, I downgraded to [A] even
    when the underlying claim likely came from a table.

---

## Where I think the framing above may be wrong

**1. "Structured vs. scattered" is presented as binary in the brief, but
the literature actually has three tiers, and the middle one is a trap.**
Every method in Q1/Q2 that claims "structured" sparsity (Q-Sparse's Block
Q-Sparse, M likely single-digit-to-low-tens; Sparse-BitNet's N:M weight
sparsity, M=8 elements; R-Sparse's column-major channel storage) is
structuring at **GPU-tensor-core or warp-coalescing granularity** — blocks
of a handful of scalars, designed so an Ampere/Turing tensor core or a
CUDA warp can consume them efficiently. This is a completely different
granularity than our ≥48KB contiguous-run requirement, which is roughly
3-4 orders of magnitude larger (48KB is tens of thousands of ternary-
packed weight values, not 8). **None of the papers in this dossier
structure sparsity at anything close to cache-line-run or memory-page
granularity** — that axis simply isn't on their radar because they all
target GPU inference, where the constraint is warp-lane utilization and
tensor-core shape, not sequential DRAM burst length. So when the brief
asks "is the sparsity structured or scattered," the honest answer for
nearly every paper here is **"structured, but at a granularity that
doesn't help a CPU engine any more than scattered would"** — a nuance
the binary framing doesn't quite capture. The one exception worth a
second look is ReLU-Strikes-Back's plain row/column framing (whole
DownProj/UpProj/QKV rows going to zero), which is closer to our
granularity in spirit, though the paper never states a byte-width either.

**2. Grouping ReLU-Strikes-Back, ProSparse, Turbo Sparse, Q-Sparse, and
R-Sparse under one "ReLUfication" umbrella (as Q1's framing implicitly
does) obscures that they are three mechanically distinct families**, and
the distinction matters for whether donor conversion is even the right
description: (a) **activation-function swap** — ReLU-Strikes-Back,
Turbo-Sparse/dReLU — literally changes which nonlinearity computes the
activation, sparsity is an emergent property of where that function is
negative; (b) **magnitude-threshold masking on top of an unchanged
activation function** — Q-Sparse (top-K), TEAL, CATS — leaves SiLU/GELU/
whatever in place and additionally zeros small-magnitude outputs,
regardless of activation family; (c) **rank/channel decomposition** —
R-Sparse — doesn't threshold anything, it replaces the layer's compute
with a lower-rank approximation that happens to touch fewer input
channels. **Category (c) may not be transferable to our engine's
mechanism at all**: our speedup comes from skipping a neuron's weight
load when *that neuron's activation is zero*; R-Sparse's speedup comes
from approximating a matmul with a smaller effective rank, which doesn't
correspond to "this specific neuron's weights are unnecessary this
token" in the same sense. I'd flag R-Sparse's inclusion under the "ReLU
Strikes Back family" implied by Q1's list as questionable — it answers
"can I sparsify a non-ReLU model" but via a different mechanism than the
others, and I am not confident its quality/sparsity numbers predict
anything about neuron-skip-based sparsity on the same donor.

**3. The token-budget table's own result argues against the framing that
"training cost in tokens" is the right unit to reason in.** Tokens alone
don't capture cost — GPU-hours scale as roughly N×D (FLOPs), so a fixed
token budget gets proportionally more expensive as the donor gets larger.
ProSparse's own 7B-vs-13B tokens/param ratio (4.94 vs. 10.3) already
shows super-linear token cost with scale for at least one method, which
would make the 100B token requirement (and therefore GPU-hour
requirement) even worse than a naive linear extrapolation suggests. If
this pattern holds generally, "token cost" quoted in isolation (as most
of these papers report it) systematically **understates** the true
difficulty of scaling any Q1 method to 100B.

**4. Turbo Sparse/dReLU is already close to the project's exact
question, and it already answers "yes, at a cost that's out of reach."**
The brief frames "can dReLU be induced in an SiLU-pretrained donor" as
open — but Turbo Sparse's own experiment IS exactly this: SwiGLU/SiLU-
pretrained Mistral-7B and Mixtral, continued-trained to dReLU for 150B
tokens on 64 A800s. Per the token-budget table, that's roughly
87,500 T4-hours at native 7B scale — already ~112x the entire 2-month
2xT4 budget **before any extrapolation to 100B at all**. I think this
is worth stating plainly rather than leaving implicit: **the closest
existing literature match to the project's central open question has
already been run, on a smaller model than our target, at a cost the
project's own stated budget cannot afford** — this isn't a gap in the
literature, it's an answered question with an unaffordable answer, which
is a different (and more useful) thing to know than "unknown."

**5. The brief's Q2 framing ("training-free is enormously more valuable
to us") is directionally right per the token-budget table, but I'd add a
caveat the framing doesn't surface: none of TEAL/CATS/R-Sparse's cheap
training-free sparsity is block-contiguous, and it is unknown — not
just untested by them, but untested by us either — whether their
scattered, per-token-varying activation patterns can be block-structured
after the fact as cheaply as our in-house from-scratch dReLU model's
pattern was (18%→50% skippable at block size 8, zero quality cost, per
the project's own measurement). Our own result was on a **static-ish
pattern property of a model trained for it**; TEAL/CATS's masks are
**recomputed per token, per forward pass**, calibrated once but applied
dynamically. Whether a fresh block-structuring pass on top of a
training-free threshold mask costs "zero" the way it did for our own
dReLU model, or costs something nontrivial because the underlying pattern
is less block-friendly to begin with, is a real open question this
dossier surfaces but cannot answer — the brief's optimism about
training-free methods should be paired with this specific unknown, not
just the token-cost argument.

**6. One thing I want to flag as NOT a framing problem, but worth
restating because it's easy to lose in the detail above: the strongest
positive evidence in this entire dossier for the project's actual
end-goal (ternary weights + activation sparsity, on the same model,
without catastrophic interaction) is Q4's BitNet a4.8 and Sparse-BitNet
results — and neither the brief's Q1-Q3 framing nor my own section
headers put them next to each other as directly as the underlying numbers
deserve. If I had to point at one existing result to build directly on,
it's BitNet a4.8's Table 1 (PPL 9.24→9.37, ternary-only vs.
ternary+44.5%-sparse, at 7B) — small, real, and directly on the
composition question, even though it still isn't a dense-donor
conversion.**

---
