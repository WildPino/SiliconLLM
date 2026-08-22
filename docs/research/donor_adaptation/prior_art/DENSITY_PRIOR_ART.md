# Density Prior Art — a survey for the v2 Density Programme

**Status: research survey, 2026-08-21. Companion to `DONOR_V2_DENSITY_PROGRAMME.md`.**
**Branch `research/donor-adaptation`.**

---

## 0. Method, and what "verified" means in this document

This project has previously been burned by numbers that entered an evidence chain through an automated
summarisation pass — including a "near-lossless ternary PTQ" claim that was fp16 divided by fp16. That
failure mode is the reason for the following procedure, which was applied to every number below.

**Procedure actually used.** Papers were downloaded as primary text (arXiv HTML / ar5iv / publisher PDF)
to a local scratch directory, converted to text with a local script that preserves table-cell boundaries,
and the tables were read directly. **No number in this document was taken from a search-engine summary,
from a WebFetch answer, or from an abstract-only reading, unless it is explicitly tagged as such.**

**Tags used throughout:**

| tag | meaning |
|---|---|
| `[TABLE]` | I read this number in the paper's own table, in the paper's own text. Highest confidence. |
| `[TEXT]` | Stated in the paper's own prose (abstract or body), not in a table. The paper asserts it; I did not see the supporting table. |
| `[DERIVED]` | **My arithmetic**, from `[TABLE]`/`[TEXT]` inputs. The inputs and the calculation are shown so it can be checked. Not a paper's claim. |
| `[UNVERIFIED]` | Encountered in a search result or a citing paper; **I did not open the primary source.** Treat as a lead, not evidence. |
| `[NOT LOCATED]` | I looked for this and did not find it. An honest gap. |

**The budget constant used for every cost comparison.** The programme's §2 states 90 T4-h/week at
25 TFLOPS effective. Therefore:

> `720 T4-h × 3600 s/h × 2.5e13 FLOP/s = 6.48e19 FLOPs` — **call this `B`.** `[DERIVED]`

All "× over budget" figures below are `6ND / B` with `N` = params trained, `D` = tokens, using the standard
`6ND` from the programme's own §2. Where a paper reports wall-clock on A100/H800 I report that instead and
say so, because a FLOP conversion across those parts is not trustworthy at the precision that matters here.

---

## Q1 — Dense→MoE conversion (MoE-ification / upcycling)

### Mechanism

Two families, and they are **not** the same operation:

1. **Partition** the existing FFN's intermediate neurons into `E` disjoint experts, then route (MoEfication,
   LLaMA-MoE). Total parameters unchanged; activation drops to `k/E` of the FFN. **This is the family the
   programme needs.**
2. **Replicate** the existing FFN into `E` identical copies and train them apart (Sparse Upcycling). Total
   parameters *grow* by `E×`; activation as a fraction of the *new* model drops, but active bytes per token
   stay roughly constant. **This does not attack the programme's lever at all** and should be struck from
   the candidate list.

**How the partition is chosen.** MoEfication proposes three: Random Split, Parameter Clustering Split
(balanced K-Means over the columns of `W1`), and Co-Activation Graph Split — build a graph whose nodes are
neurons and whose edge weights are `Σ_x h_n(x)·h_m(x)·1[h_n>0, h_m>0]`, then partition it with METIS. `[TEXT]`
This is precisely the D0 measurement the programme proposes, and **it has already been done and published.**

### What is achieved numerically

**MoEfication, T5-Large, 20% of neurons selected** (Table 2, and Table 1 for the dense row) `[TABLE]`:

| construction | selection | SST-2 | MNLI | RACE |
|---|---|---|---|---|
| *(original dense)* | — | 96.2 | 89.5 | 81.3 |
| Random | Groundtruth | 95.9 | 87.3 | 80.0 |
| Random | MLP (trained router) | 94.1 | 84.1 | 75.0 |
| Parameter Clustering | Groundtruth | 96.3 | 89.1 | 80.8 |
| Parameter Clustering | MLP | 95.9 | 87.5 | 78.7 |
| Co-Activation Graph | Similarity (**training-free router**) | 92.2 | 81.4 | 71.0 |
| Co-Activation Graph | MLP | 95.4 | 87.5 | 79.0 |

> **The single most decision-relevant row-pair in this survey.** With a *perfect* router, **random** split
> scores 95.9 / 87.3 / 80.0 and **clustered** split scores 96.3 / 89.1 / 80.8. Clustering buys
> +0.4 / +1.8 / +0.8. **Co-activation clustering is nearly worthless for quality; almost all of the benefit
> is in the router.** This should change how D0 is framed — see §6.

Broader claims: "10% to 30% of FFN parameters while maintaining over 95% original performance"; "2x speedup
with 25% of FFN parameters" `[TEXT]`. Larger models are sparser: "80% inputs only activate less than 3%
neurons in T5-XLarge"; T5-XLarge reaches ~98% relative performance at 10% neurons `[TEXT]` (this is
Figure 2/3, not a table — I did not read a table for it).

**Applicability caveat that is easy to miss:** MoEfication's models are **ReLU-based** (T5). For GeLU-based
BERT-Large they first had to convert to ReLU, which took **400 optimisation steps** vs ~10,000 for
pretraining `[TEXT]`. That cheapness does **not** transfer to a SwiGLU LLM — see Q2.

**LLaMA-MoE (LLaMA-2-7B → MoE), 200B tokens** `[TEXT]`:

| model | k | E | activated params | LogiQA | BoolQ(32) | LAMBADA | NQ(32) | MMLU(5) | Average |
|---|---|---|---|---|---|---|---|---|---|
| Sheared-LLaMA-2.7B | — | — | 2.7B | 28.3 | 73.6 | 68.3 | 17.6 | 27.3 | 56.4 |
| LLaMA-MoE-3.0B | 2 | 16 | 3.0B | 30.6 | 71.9 | 66.6 | 17.0 | 26.8 | 55.5 |
| LLaMA-MoE-3.5B | 4 | 16 | 3.5B | 29.7 | 75.0 | 69.5 | 20.3 | 26.8 | 57.7 |
| LLaMA-MoE-3.5B | 2 | 8 | 3.5B | 29.6 | 73.9 | 69.4 | 19.8 | 27.0 | 57.6 |

`[TABLE]`. **Note what is absent: LLaMA-2-7B itself is not in this table.** The paper compares only to other
~3B dense models. Its own body text says "we observe a significant performance decline between the LLaMA-MoE
models and the original dense LLaMA models" `[TEXT]`. **Retention versus the donor is therefore unreported,
and the paper concedes it is bad.** Any use of LLaMA-MoE as evidence for ≥90% donor retention would be a
fabrication.

**Granularity actually reached.** LLaMA-MoE's finest is top-2-of-16 = **12.5% of FFN**. Active 3.0B of 6.7B
total = **45% of the whole model**, because attention and embeddings stay dense. `[DERIVED]` from the table.

**The finest-grained released MoE of any origin** is DeepSeek-V3: "671B total parameters with 37B activated
for each token" `[TEXT]`, 256 routed + 1 shared expert, 9 selected at decode `[TEXT]` = **5.51% activation**
`[DERIVED]`. It cost 2.788M H800-hours over 14.8T tokens `[TEXT]` and was trained from scratch.

> **Does anyone reach top-2-of-100-class granularity (2%)? No.** Best by conversion: 12.5% of FFN
> (LLaMA-MoE). Best released at all, from-scratch: 5.5% of total (DeepSeek-V3). **2% is 2.8× finer than
> anything that exists, and 6× finer than anything ever produced by conversion.** `[DERIVED]`

**Newer work.** *Dense2MoE* (2605.26496) `[UNVERIFIED]` — a search result states a 225B-token continual
budget; I downloaded the paper but did not verify this number in its tables, so it stays UNVERIFIED.
*Drop-Upcycling*, *Innovator*, *Expert Upcycling*, *Marco-MoE* — leads only, `[UNVERIFIED]`.

**Continual LLM Upcycling: Predictor-Gated Bank-Wise Sparsity** (arXiv 2606.10722, Jun 2026) is the most
relevant recent item and is covered under Q2 because it is channel-level, not expert-level.

### Training cost

| recipe | tokens | params | `6ND` | vs `B` |
|---|---|---|---|---|
| MoEfication **router only** | — | — | "several minutes on a single GPU" **per FFN** `[TEXT]` | negligible |
| LLaMA-MoE | 200B `[TEXT]` | 6.7B | 8.04e21 | **124×** `[DERIVED]` |
| Sparse Upcycling | "46% and 55% extra training" vs original dense pretraining `[TEXT]` | — | — | for a 15T-token donor: ~7T tokens `[DERIVED]` — **hopeless** |

### Is the router derivable without training?

**Partly, and the cost of "without" is measured.** MoEfication's *Similarity Selection* is training-free
(expert centroid vs hidden state) and scores 92.2 / 81.4 / 71.0 with the co-activation split, versus
95.4 / 87.5 / 79.0 for the trained MLP router `[TABLE]`. **A training-free router costs ~3–8 points.** The
trained router is the cheap part of this whole survey — minutes per FFN — so there is no good reason to
insist on training-free here.

### Counter-evidence

- Clustering is nearly redundant given a good router (Table 2 above).
- LLaMA-MoE's retention vs donor is unreported and self-described as poor.
- Sparse Upcycling attacks the wrong lever entirely.
- MoEfication is demonstrated on ReLU encoder/enc-dec models up to T5-XLarge (~3B), **not** on a modern
  SwiGLU decoder LLM. Its transfer to that setting is untested. `[NOT LOCATED]`

---

## Q2 — Inducing activation sparsity in a trained model

**This is the lever with the largest published effect and the largest published price.**

### What is achieved, and after how many tokens

**ReLU Strikes Back** (Mirzadeh et al., 2310.04564), Table 1 `[TABLE]`. Finetuning 30B tokens (stage 1),
50B (stage 2), RefinedWeb `[TEXT]`:

| model (stage) | QKV sp. | DownProj sp. | UpProj sp. | FLOPS (G) | zero-shot avg |
|---|---|---|---|---|---|
| Llama 7B | 0 | 0 | 0 | 6.6 | 68.4 |
| Llama 7B (s1) | 0 | **62** | 0 | 4.8 | 67.1 |
| Llama 7B (s2) | 51 | **65** | 67 | 2.9 | 66.4 |
| Falcon 7B | 0 | 1 | 0 | 6.6 | 66.8 |
| Falcon 7B (s1) | 0 | **94** | 0 | 4.1 | 65.2 |
| Falcon 7B (s2) | 56 | **95** | 56 | 2.2 | 64.8 |

> **The same recipe gives 95% on Falcon (GELU) and 62–65% on Llama (SwiGLU).** The paper flags this itself:
> "the relufied Llama has much less sparsity (65%) than the relufied Falcon model (95%)" `[TEXT]`.
> **The donor's original activation function is a first-order predictor of how far this lever moves.**
> Llama-7B s2 reaches only **44% of original FLOPs** — a 2.3× reduction, not 50×.

**ProSparse** (2402.13516), the strongest published ReLUfication of a SwiGLU model. Main table `[TABLE]`:

| model | avg performance | avg sparsity |
|---|---|---|
| LLaMA2-7B | 37.96 | — |
| ReluLLaMA-7B | 37.62 | 66.98 |
| ProSparse-7B | **38.46** | **89.32** |
| LLaMA2-13B | 44.06 | — |
| ReluLLaMA-13B | 42.74 | 71.56 |
| ProSparse-13B | **44.90** | **88.80** |

**Do not read that as "free".** The paper's own Table 4 supplies the matched-token control `[TABLE]`:

| setting | tokens | avg sparsity | avg performance |
|---|---|---|---|
| Vanilla ReLU | 34.60B | 66.04 | 41.40 |
| Shifted ReLU | 34.60B | 69.59 | 41.33 |
| **ProSparse** | **34.60B** | **89.32** | **38.46** |
| Vanilla ReLU | 89.13B | 64.93 | 41.52 |
| Shifted ReLU | 89.13B | 68.35 | 41.40 |
| **ProSparse** | **89.13B** | **88.29** | **40.67** |

> **Against a matched-training control, ProSparse costs 41.40 → 38.46 (retention 92.9%) at 34.6B tokens,
> and 41.52 → 40.67 (retention 97.9%) at 89.13B tokens.** `[DERIVED]` The headline "comparable to LLaMA2-7B"
> is true only because 34.6B tokens of continued pretraining lifts the benchmark average from 37.96 to ~41.4
> on its own. Both retentions clear the programme's ≥90% bar — but the honest denominator is 41.40, not 37.96.

**Turbo Sparse / dReLU** (2406.05955), Table 6 `[TABLE]`, 150B tokens `[TEXT]`:

| | Mistral-7B | TurboSparse-Mistral-7B | Mixtral-47B | TurboSparse-Mixtral-47B |
|---|---|---|---|---|
| total params | 7B | 7B | 47B | 47B |
| **activated params** | 7B | **2.5B** | 13B | **4.3B** |
| OpenLLM Leaderboard Avg. | 61.57 | **63.65** | 69.34 | **71.76** |

> **TurboSparse-Mixtral-47B at 4.3B/47B = 9.1% activation is the lowest activation fraction ever published
> for a converted dense-origin model, and it is the only published composition of MoE × activation
> sparsity.** `[DERIVED]` **Caveat I must flag: unlike ProSparse, this paper provides no matched-token dense
> control at 7B/47B scale**, so "better than the original" cannot be separated from the effect of 150B fresh
> tokens. Its only control is a 300M dReLU-vs-SwiGLU from-scratch pair (Tables 2/10). Treat the +2.1 and
> +2.4 point gains as **unattributed**.

**Training-free variants — the budget-compatible option, and its ceiling.** TEAL (2408.14690), Table 1,
WikiText perplexity `[TABLE]`:

| method | LLaMA-3-8B | LLaMA-2-7B | LLaMA-2-70B | Mistral-7B |
|---|---|---|---|---|
| Baseline (0%) | 5.87 | 5.07 | 3.12 | 4.92 |
| CATS 25% | 6.78 | 5.52 | 3.42 | 5.87 |
| TEAL 25% | 5.94 | 5.09 | 3.13 | 5.01 |
| CATS 40% | 7.6·10⁴ | 43.8 | 171 | 2.8·10⁴ |
| TEAL 40% | 6.21 | 5.22 | 3.25 | 5.13 |
| TEAL 50% | 6.67 | 5.43 | 3.50 | 5.31 |
| TEAL 65% | 9.06 | 6.62 | 4.28 | 6.23 |

> **Training-free activation sparsity tops out at ~50% model-wide.** "Most models degrade significantly at
> 65% sparsity" `[TEXT]`. CATS is ~25% model-wide and degenerate at 40% `[TEXT]` + `[TABLE]`.
> **Zero-GPU-hour sparsification gets us to 50%, which is 25× short of 2%.**

**Trained channel sparsity, most recent** — *Predictor-Gated Bank-Wise Sparsity* (2606.10722, Jun 2026)
`[TEXT]`: a low-rank predictor emits FFN-channel routing logits **before** the gate/up projections (which is
exactly the programme's gate-first requirement), with a **bank-wise top-k keeping 16 channels of every 64**
→ **4× sparsity (25% activation)**. Reported as "limited performance degradation". Two negative results
worth having: MoE-style load-balancing bias **does not transfer** to channel-level routing from a dense
checkpoint (all balance variants were neutral-to-worse); and their GPU implementation does not realise the
4× (it still materialises the dense gate/up). Their sparsity was introduced ~3T tokens into a 4T-token 32K
stage, on a backbone they pretrained themselves for 8T tokens — so this is **not** a cheap conversion.

### Training cost

| recipe | tokens | N | `6ND` | vs `B` | reported wall-clock |
|---|---|---|---|---|---|
| ReLU Strikes Back, Llama-7B s1 | 30B `[TEXT]` | 7B | 1.26e21 | **19.4×** `[DERIVED]` | — |
| ProSparse-7B (cheap) | 34.60B `[TABLE]` | 7B | 1.45e21 | **22.4×** `[DERIVED]` | 8×A100-80G ≈10 days ≈ **1920 A100-h** `[TEXT]` (paper says "all the 7B models"; per-model attribution is ambiguous) |
| ProSparse-7B (good) | 89.13B `[TABLE]` | 7B | 3.74e21 | **57.8×** `[DERIVED]` | as above |
| ProSparse-13B | 134.22B `[UNVERIFIED]` | 13B | — | — | 32×A100 ≈20–30 days `[TEXT]` |
| Turbo Sparse Mistral-7B | 150B `[TEXT]` | 7B | 6.30e21 | **97×** `[DERIVED]` | — |
| Turbo Sparse Mixtral-47B | 150B `[TEXT]` | 13B active | 1.17e22 | **181×** `[DERIVED]` | — |
| TEAL / CATS | **0** | — | 0 | **0×** | — |

> **The break-even donor size.** Holding ProSparse's cheap setting at a fixed 34.6B tokens, `6·N·34.6e9 ≤ B`
> gives **N ≤ 0.31B**. Holding its tokens-per-param ratio (4.94) gives **N ≤ 1.48B**. `[DERIVED]`
> **ReLUfication at the only token counts anyone has shown to work fits our budget only for donors of
> roughly 0.3–1.5B parameters.** For a 100B donor at a fixed 34.6B tokens the cost is **320× `B`.**

### Is the sparsity contiguous enough to skip in bulk?

**No, not natively — and this is under-appreciated in the literature because GPUs have different granularity
needs than our engine.**

The sparsity is **per-neuron / channel-wise**. Skipping neuron `j` means skipping row `j` of `W_gate`, row `j`
of `W_up`, and column `j` of `W_down`. TEAL is explicit that this is the hardware-friendly case and that you
must store "weights associated with input sparsity in column-major format, and weights associated with output
sparsity in row-major format" to get coalescing `[TEXT]`. ReLU Strikes Back likewise calls it "more
hardware-friendly due to zeroing more extensive and structured chunks, such as rows or columns" `[TEXT]`.

**But the chunk size is set by `d_model`, not by us.** `[DERIVED]`, using LLaMA-2-7B's published dimensions
(`d_model`=4096, `d_ffn`=11008):

| quantity | value |
|---|---|
| weights per neuron per matrix | 4096 |
| bytes at 0.5 B/weight | **2,048 B = 2 KB** |
| consecutive co-activating neurons needed for a 48 KB read | **24** |
| same, for a 100B-class donor (`d_model` ≈ 8192) | **12** |

> **Every published activation-sparsity result gives our engine 2–4 KB scattered runs. The engine wants
> ≥48 KB. The gap is a factor of 12–24 consecutive co-activating neurons.** This is the real, concrete
> engineering target, and it is a *layout* problem, not a *quality* problem.

**The one paper that attacks exactly this**: *Neuralink: Fast LLM Inference on Smartphones with Neuron
Co-Activation Linking* (ASPLOS '25, 2410.19274). It defines "Neuron Co-Activation", formulates optimal
neuron placement in flash as a **globally optimal Hamiltonian Path** problem on the complete co-activation
graph, solves it greedily offline, and co-locates co-activating neurons at contiguous addresses `[TEXT]`.
Measured result: **1.80× average I/O bandwidth improvement, 1.49× average end-to-end speedup** `[TEXT]`.
It also reports that I/O is 73.4%–95.8% of per-token latency when FFN is offloaded `[TEXT]` — the same
regime our engine is in.

> **Read this honestly: co-activation-aware placement is real, is published, is measured, and buys 1.80× of
> bandwidth.** Our engine's gap between 48 KB-granular reads (~2.5×) and random gather (~14×) is ~5.6×.
> **1.80× closes about a third of it.** That is valuable and it is not a solution.

---

## Q3 — Structured factorization: Monarch, butterfly, Kronecker

### Is there a projection algorithm from a dense trained `W` to the nearest structured factorization?

**Yes, and it is the cleanest positive result in this survey.** Monarch (Dao et al., ICML 2022, 2204.00595).
The paper states the problem precisely — "Only a few specific classes of structured matrices have a tractable
projection solution, such as entrywise sparse matrices (magnitude pruning), low-rank matrices (the
Eckart-Young theorem), and orthogonal matrices (the orthogonal Procrustes problem). For more expressive
classes of structured matrices, projection remains a long-standing problem" — and then solves it for Monarch:
"we derive a projection algorithm for our Monarch parameterization and **prove that it finds the optimal
solution (Theorem 1)**" `[TEXT]`.

**What the algorithm actually is** (Algorithm 1, read in full) `[TABLE]`: reshape `A` (n×n, n=m²) into an
m×m×m×m tensor; for each of the m² pairs `(j,k)` take the m×m slice and compute its **best rank-1
approximation via SVD**; reassemble the `u`s and `v`s as the two block-diagonal factors.

> **Cost: m² SVDs of m×m matrices. For n=4096 that is 4096 SVDs of 64×64 — seconds to minutes on a CPU,
> per matrix, with zero GPU and zero tokens.** `[DERIVED]` **This is the only mechanism in the entire survey
> with an *optimality proof* and a *negligible* cost.**

There is also a Theorem 2: an `O(n^{5/2})` algorithm to recover the exact Monarch factors of a matrix already
in the `MM*` class `[TEXT]`.

### Compression ratios and quality

**Monarch's own D2S result** (Table 8) `[TABLE]` — project pretrained BERT to Monarch, then finetune on GLUE:

| model | GLUE (avg) | speedup | params | FLOPs |
|---|---|---|---|---|
| BERT-base | 78.6 | — | 109M | 11.2G |
| Monarch-BERT-base | 78.3 | 1.5× | 55M | 6.2G |
| BERT-large | 80.4 | — | 335M | 39.5G |
| Monarch-BERT-large | 79.6 | 1.7× | 144M | 14.6G |

Retention 99.6% and 99.0% at ~2.0–2.3× compression `[DERIVED]`. **Caveats that matter:** they use "the number
of blocks in the block-diagonal matrices to be between 2 and 4 based on the parameter budgets (25%–50% of the
dense model)" `[TEXT]` — i.e. **nowhere near** the theoretical `2/√n` (which would be 3.1% at n=4096). And
this is an *encoder* + *task finetuning*, a much easier setting than preserving a generative LLM's general
capability. The paper calls it "a proof of concept" `[TEXT]`.

**At real LLM scale**, the best result I found is **ProcrustesGPT** (Grishina et al., Findings of ACL 2025),
which is also the Q3×Q4 intersection — it *searches for the orthogonal transformation that makes the weights
most compressible* within a structured class (sum-of-Kronecker-products, and a generalisation that "generalizes
Monarch (Dao et al., 2022) matrices"). WikiText2 perplexity, Table 1 `[TABLE]`:

| method | struct. | OPT-125m | OPT-2.7b | OPT-13b | Llama2-7b | Llama2-13b |
|---|---|---|---|---|---|---|
| Dense (0%) | — | 27.65 | 12.47 | 10.13 | 5.47 | 4.88 |
| SliceGPT (~16–20%) | — | 38.65 | 14.84 | 11.12 | 7.60 | 6.60 |
| ProcrustesGPT_W (~15%) | Kron. √(D+1) | **36.08** | **13.95** | **10.67** | **6.54** | **5.71** |
| ProcrustesGPT_W (~15%) | GS (Monarch-gen.) √(D+1) | 39.58 | 13.81 | 10.68 | 6.76 | 5.96 |

> **At LLM scale, training-free structured factorization achieves only ~15% parameter compression, and
> Llama2-7B still degrades 5.47 → 6.54 perplexity (+19.6%).** `[DERIVED]` The paper is blunt about why:
> "it seems unrealistic to expect that weight matrices of pretrained models can be accurately represented by
> structured matrices without any fine-tuning" `[TEXT]`. **This is ~15% off the *bytes* lever, not 50×, and
> it is nowhere near 0.5 B/weight on its own.**

### Does the resulting structure give bulk-contiguous access?

**Yes — and this is Monarch's decisive advantage over every activation-sparsity method in Q2.** `[DERIVED]`

A Monarch matrix is `P₁B₁P₂B₂` with `B` block-diagonal. The permutations `P` act on the **activation vector**
(a few KB, cache-resident), not on the weights. The weights are stored as dense blocks and **every block is
read, in order, every token**. There is no gather on the weight stream at all.

For n=4096 with m=64: each factor is 64 blocks of 64×64 = 262,144 weights = **128 KB at 0.5 B/weight, read
fully sequentially** `[DERIVED]`. That is **above** the 48 KB threshold by 2.7×, with no clustering, no
router, and no layout problem to solve.

> **The trade is exact and worth stating plainly: Monarch is perfectly bulk-contiguous but reduces only the
> *bytes* lever, never the *activation-fraction* lever. Every Monarch parameter is touched every token.**
> It is orthogonal to Q1/Q2 and composes with them multiplicatively, which is what makes it the right
> baseline — the programme's §3C instinct is correct.

**Related leads not opened**: *Efficient In-Memory Acceleration of Sparse Block Diagonal LLMs* (2510.11192),
*Scaling Probabilistic Circuits via Monarch Matrices* (2506.12383), *Efficient Identification of Butterfly
Sparse Matrix Factorizations* (2110.01230) — all `[UNVERIFIED]`.

**`[NOT LOCATED]`**: any application of Monarch projection to a modern decoder LLM's FFN at ≥7B with reported
general-capability retention. Monarch's LLM-scale evidence is GPT-2 pretraining acceleration, not donor
conversion.

---

## Q4 — Rotation FOR SPARSITY (not for quantizability)

**This is not open. It became a live subfield in 2025–2026, and the programme's §3D claim that "whether a
rotation exists that makes a trained LLM's weights sparse is, as far as I know, open" must be revised.**
That is the most important correction this survey makes to the programme document.

### What exists

**The enabling property is `computational invariance`** (from SliceGPT): certain orthogonal transformations
can be applied to transformer weight matrices without altering the model's output, and adjacent layers'
rotations can be merged `[TEXT]`. QuaRot/SpinQuant exploit it for **outlier removal → quantizability**. Three
distinct 2025–2026 lines exploit it for **sparsity**:

**1. DenoiseRotator** (2505.23049) — **the closest match to the programme's D-axis.** It inserts *learnable*
orthogonal matrices and trains them to **minimise the information entropy of the normalised
importance-score distribution**, i.e. explicit **energy compaction**: "concentrates importance onto a smaller
subset of weights" `[TEXT]`. Plug-and-play before Magnitude / Wanda / SparseGPT. WikiText-2 perplexity,
Table 1 `[TABLE]`:

| sparsity | method | Mistral-7B | LLaMA3-8B | LLaMA3-70B | Qwen2.5-7B | Qwen2.5-72B |
|---|---|---|---|---|---|---|
| 0% | Dense | 5.95 | 6.14 | 2.86 | 6.85 | 3.88 |
| 50% | Magnitude | 30.39 | 30.39 | 10.58 | 198.88 | 734.04 |
| 50% | +DenoiseRotator | **7.30** | **14.43** | **7.00** | **9.27** | **5.37** |
| 50% | Wanda | 6.92 | 9.86 | 5.80 | 8.61 | 5.22 |
| 50% | +DenoiseRotator | **6.52** | **7.82** | **4.73** | **7.93** | **4.94** |
| 50% | SparseGPT | 6.94 | 9.57 | 5.99 | 8.46 | 4.94 |
| 50% | +DenoiseRotator | **6.38** | **7.60** | **4.61** | **7.60** | **4.78** |

**Cost — and this is the headline for us**: "training on LLaMA 3 70B with SparseGPT took approximately
**28 hours** and utilized around **30 GB of GPU memory on a single NVIDIA A100 GPU**", 2000 Adam steps,
"the training duration is independent of the calibration dataset size", "**No recovery finetuning was
performed after pruning**" `[TEXT]`.

> **28 A100-hours to rotate a 70B model. This is the only high-leverage mechanism in the survey whose cost
> is unambiguously inside our budget at 70B-class donor scale.** `[TEXT]`

**Two hard caveats, both from the paper's own text/tables:**

- **Overhead.** It adds one `(hidden, hidden)` matrix per layer — "in LLaMA-3-8B, this results in
  approximately 0.5 billion additional parameters (i.e. 4096×4096×32) ... about 6.7% of the original model
  size" `[TEXT]`. **Those rotations are dense and touched every token.** For a 100B donor at a 2.1B active
  budget, a rotation overhead of that class would consume a large fraction of the entire budget.
- **The affordable version loses most of the benefit.** Appendix E, Table 9 (LLaMA-3-8B, SparseGPT 50%)
  `[TABLE]`:

  | block number | perplexity | zero-shot acc | time/step (s) | entropy |
  |---|---|---|---|---|
  | 1 (dense rotation) | **7.597** | **69.58** | 0.124 | 384128 |
  | 2 | 8.024 | 68.68 | 0.088 | 410816 |
  | 4 | 8.544 | 68.47 | 0.076 | 428160 |
  | 8 | 8.882 | 67.51 | 0.072 | 440512 |

  > **Energy compaction degrades monotonically as the rotation is made structured — and structured is
  > exactly what our engine can afford. Block-8 gives back most of the gain over plain SparseGPT (9.57).**
  > This is the central tension of axis D and it is already measured. The paper's own "future work" list
  > proposes combining block-diagonal rotations with **permutation** matrices to emulate dense ones `[TEXT]`
  > — which is, note, the Monarch construction.

**2. ProcrustesGPT** (Findings of ACL 2025) — rotates to maximise compressibility in a *structured matrix*
class rather than to concentrate pruning importance. Numbers in Q3 above. **Rotation for structure.**

**3. Change-of-Basis Pruning via Rotational Invariance** (2511.16061) — the most ambitious and the least
transferable. Introduces **two-subspace radial activations (TSRAs)**, an activation family *invariant to
orthogonal transformations within its two subspaces*, so CoB rotations merge into surrounding weights with
**zero extra parameters**. Results `[TEXT]`: on VGG-16 / CIFAR-10, CoB "extends reliable pruning frontier
from roughly 30% to 70% of parameters without post-prune fine tuning", and under threshold-based pruning
"prunes **90–96% of parameters** while maintaining 1–6% accuracy drop after fine-tuning".

> **Three disqualifying caveats, all stated by the paper itself:** it is VGG-16/CIFAR-10, **not an LLM**; it
> requires **replacing the activation function and training from scratch** — it is not a transformation of a
> pretrained model; and the rotational-invariance modifications cost "**a slight accuracy drop of 4.52%
> compared to a ReLU-based control**" before any pruning. The paper self-describes as "a proof-of-concept".

**4. RotPruner** — cited by DenoiseRotator as prior work that "attempts to leverage this property for
pruning and reports promising empirical results; however, it lacks theoretical analysis or formal
justification" `[TEXT]`. I did not open RotPruner. `[UNVERIFIED]`

### Honest assessment of originality

**Not open, but not closed either.** The honest statement is narrower and more useful than "open":

- **Rotating to make *pruning* (static weight sparsity) more robust: solved-ish, cheap, published, at 50%/2:4
  sparsity levels only.**
- **Rotating to make *structured factorization* fit better: published (ProcrustesGPT), ~15% compression.**
- **Rotating to make weights *entrywise sparse* at the 90–98% levels this programme needs, on a pretrained
  LLM, with an affordable (structured) rotation: `[NOT LOCATED]`.** The one paper that reaches 90–96% needs
  a from-scratch architecture and is demonstrated on CIFAR-10.
- **Rotating to make *activations* sparse — a basis in which the hidden state has few significant entries —
  as opposed to weights: `[NOT LOCATED]`.** I searched for this specifically. Every rotation-for-sparsity
  paper found operates on weight importance. Given that our engine's skip path is activation-gated
  (gate-first, then row-skip), **this is the genuinely open slot**, and it is narrower and more testable than
  the programme's §3D framing.

---

## Q5 — Is the FFN really "where the knowledge lives"?

### The case for

**Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (EMNLP 2021)** — the foundation.
Each key correlates with human-interpretable textual patterns in training examples; each value induces a
distribution over the output vocabulary; lower layers capture shallow patterns, upper layers semantic ones;
the layer output is a composition of memories refined through the residual stream. `[UNVERIFIED]` — I did not
open this paper's tables in this pass; the characterisation above is from secondary sources and should be
verified before it is cited in any external claim.

**ROME / MEMIT** established that *editing* mid-layer MLP weights reliably changes stored facts. `[UNVERIFIED]`
in this pass.

**MoEfication's framing** is independent corroboration from a different direction: FFNs partition into
functional experts with "general and input-specific experts" `[TEXT]`.

### The case against — and it is strong

**Hase, Bansal, Kim, Ghandeharioun, "Does Localization Inform Editing?" (NeurIPS 2023, 2301.04213).**
Read directly `[TEXT]`:

- "we find that we can change how a fact is stored in a model by editing weights that are in a **different
  location** than where existing methods suggest that the fact is stored."
- "a substantial fraction of factual knowledge stored **outside** of the range of layers edited by ROME/MEMIT."
- **ρ = −0.13 (p < 1×10⁻³)** between ROME edit success and the tracing effect at layer 6 in GPT-J — "not a
  positive relationship but a negative relationship"; in most layers "near-zero rather than negative".
- Regression: "tracing effects explain **at most an additional 3.2%** of the variance in edit success",
  versus the layer-only regression which "explains most of the variance in the outcome (**58.5%** on average)".
- "model edit success is essentially unrelated to where factual information is stored in models, as measured
  by Causal Tracing."

**What this does and does not refute.** It refutes *causal-tracing localisation as a guide to where to edit*.
It does **not** refute that FFN weights store knowledge — ROME/MEMIT still work when applied to FFN weights.
Its relevant sting for us is the first bullet: **knowledge is more distributed across layers than the
FFN-as-localised-store picture implies.** Further critiques exist (attention heads also convey factual
knowledge; "knowledge neurons ... do not store factual knowledge — they route") — `[UNVERIFIED]`, from
secondary sources only.

### Prior work freezing a pretrained FFN and training a *new* surrounding network

**This exists, it worked, and it is the strongest single result in this survey for the programme's §4.**

**MOHAWK / Phi-Mamba** (Bick et al., NeurIPS 2024, 2408.10189). The method factors a model into sequence
mixing and channel mixing, **swaps the teacher's matrix mixer (attention) for a Mamba-2 SSM while reusing the
teacher's MLP**, and distils in three stages. From the paper `[TEXT]`: "MOHAWK adjusts the structure of the
student blocks to utilize the MLP in the same way as the teacher model, effectively swapping the teacher's
matrix mixer with that of the student. **Interestingly, during this step, the MLP weights can be kept frozen
while keeping the model performant.**" It also cites the motivation explicitly: "It has been hypothesized that
much of the information stored in language models resides in MLP blocks."

Table 1 `[TABLE]`:

| model | tokens / data | WinoG. | Arc-E | Arc-C | PIQA | HellaS. | Lamb. | **Avg** |
|---|---|---|---|---|---|---|---|---|
| Phi-1.5-1.3B (teacher) | 150B / unknown | 73.4 | 75.6 | 48.0 | 76.6 | 62.6 | 53.4 | **64.9** |
| **Phi-Mamba-1.5B** | **3.0B / C4** | 71.7 | 74.0 | 44.1 | 75.5 | 60.2 | 50.1 | **62.6** |

Table 2 `[TABLE]`: Hybrid Phi-Mamba-1.5B (4 attention layers vs the teacher's 24) scores **66.0** avg vs
Phi-1.5's **67.2** on the six-task set used there, at 5B tokens `[TEXT]`.

> **Retention 62.6 / 64.9 = 96.5%, with the MLP frozen, the attention entirely discarded, and 3.0B tokens.**
> `[DERIVED]` **This clears the programme's ≥90% bar, and it clears the budget:**
> `6 × 1.5e9 × 3e9 = 2.7e19 FLOPs = 0.42 × B` `[DERIVED]`. With the MLP frozen (≈2/3 of params, forward-only)
> the true cost is nearer `≈3.3ND ≈ 1.5e19 = 0.23 × B` `[DERIVED]`.
> The paper also notes Stage 3 "can freeze all network components except the Mamba-2 sequence mixer without a
> significant performance drop ... enabling more users to utilize the MOHAWK distillation process" `[TEXT]`.

**The Mamba in the Llama** (Wang et al., NeurIPS 2024, 2408.15237). Same structural bet, larger donors
`[TEXT]`: "We assume that most of the knowledge from the transformer is maintained in the MLP layers which
were transferred from the original model ... **During this stage, the MLP layers are kept frozen and the
Mamba layers are trained.**" Distils Zephyr-7B and Llama-3-8B "using only **20B tokens** of training", and
"The total distillation process for each hybrid model takes **less than five days in 8×80G A100**"
(< **960 A100-hours**). Matches the teacher on chat benchmarks and beats from-scratch Mamba-7B (1.2T tokens)
and NVIDIA Hybrid Mamba2 (3.5T tokens) on MMLU/TruthfulQA `[TEXT]`. **Caveat: they freeze the FFN only in the
first stage; "in the second and final stage all parameters are trained"** `[TEXT]` — so this is **not** a
fully-frozen-store existence proof, unlike MOHAWK.

**Frozen Pretrained Transformer** (Lu et al., 2103.05247) — freezes self-attention *and* feedforward and
trains only input/output layers + layer-norms (~0.1% of parameters), transferring GPT-2 to CIFAR-10, ListOps,
protein folding `[TEXT]`. The paper's own framing of what the trained input layer is doing is exactly the
programme's §4 thesis: "in essence, **we are learning how to query the transformer**" `[TEXT]`. **Caveat:
sequence *classification* tasks with a single linear output layer, not language modelling with a substantial
new core.**

**Memory Layers at Scale** (Berges et al., 2412.09764) — the existence proof that extreme-sparse-activation
knowledge stores *work*, though trained from scratch. Replaces the FFN of one or more layers with a
trainable product-key top-k lookup; scales to "**128B memory parameters, pretrained to 1 trillion tokens**";
"memory augmented models can match the performance of dense models that have been trained on 4x more
compute"; a 1.3B base with 64M keys "approaches the performance of the Llama2 7B model" `[TEXT]`. Table 2
with an 8B base + 16M memory values (64B extra params) `[TABLE]` shows Memory+ (200B tokens) beating dense
(200B) on MMLU 50.14 vs 41.35, HumanEval 23.17 vs 21.34, NQ 19.36 vs 18.61.

> **Two things to take from Memory Layers, one encouraging and one alarming.** Encouraging: a top-k store at
> activation fractions far below 2% is learnable and beats dense at matched compute — the architecture the
> programme wants is not exotic. **Alarming: the access pattern is a random gather of k rows.** The paper
> says memory layers are "almost entirely memory bandwidth bound" and needed custom CUDA kernels to reach
> 3 TB/s on H100 (vs <400 GB/s with PyTorch) `[TEXT]`. **That is precisely the ~14× gather penalty our engine
> pays.** A memory-layer-shaped store on our engine is the worst case, not the best case, unless the
> co-location problem of Q2 is solved first. Also note: "replacing further FFN layers degrades performance,
> showing sparse and dense layers are both needed and likely complementary" — they cap at 3 memory layers
> `[TEXT]`.

---

## Table of every quantitative claim, with verification status

| # | claim | value | source | status |
|---|---|---|---|---|
| 1 | MoEfication T5-Large, 20% neurons, co-activation split + MLP router | 95.4 / 87.5 / 79.0 (SST-2/MNLI/RACE) | 2110.01786 Table 2 | `[TABLE]` |
| 2 | MoEfication T5-Large dense baseline | 96.2 / 89.5 / 81.3 | 2110.01786 Table 1 | `[TABLE]` |
| 3 | MoEfication random split + groundtruth router, 20% | 95.9 / 87.3 / 80.0 | 2110.01786 Table 2 | `[TABLE]` |
| 4 | MoEfication training-free (Similarity) router, 20% | 92.2 / 81.4 / 71.0 | 2110.01786 Table 2 | `[TABLE]` |
| 5 | MoEfication general claim | 10–30% FFN params at >95% original perf | 2110.01786 abstract | `[TEXT]` |
| 6 | MoEfication router training cost | "several minutes on a single GPU" per FFN | 2110.01786 §4 | `[TEXT]` |
| 7 | MoEfication GeLU→ReLU adaptation (BERT-Large) | 400 steps vs ~10,000 pretraining | 2110.01786 §4.3 | `[TEXT]` |
| 8 | T5-XLarge sparsity | 80% of inputs activate <3% of neurons | 2110.01786 Fig. 2 | `[TEXT]` (figure) |
| 9 | LLaMA-MoE finest granularity | top-2-of-16 = 12.5% of FFN | 2406.16554 Table 1 | `[TABLE]` |
| 10 | LLaMA-MoE-3.0B activation | 3.0B of 6.7B = 45% of model | 2406.16554 Table 1 | `[DERIVED]` |
| 11 | LLaMA-MoE training | 200B tokens | 2406.16554 abstract | `[TEXT]` |
| 12 | LLaMA-MoE cost vs budget | 8.04e21 FLOPs = 124 × B | — | `[DERIVED]` |
| 13 | LLaMA-MoE retention vs LLaMA-2-7B | **not reported**; "significant performance decline" admitted | 2406.16554 §1 | `[TEXT]` |
| 14 | Sparse Upcycling extra training | +46% / +55% of original dense pretraining | 2212.05055 §1 | `[TEXT]` |
| 15 | DeepSeek-V3 activation | 37B of 671B = 5.51% | 2412.19437 abstract | `[TEXT]`/`[DERIVED]` |
| 16 | DeepSeek-V3 training cost | 2.788M H800-hours, 14.8T tokens | 2412.19437 §1 | `[TEXT]` |
| 17 | ReLU Strikes Back, Llama-7B s2 | 65% down-proj sparsity, 2.9G vs 6.6G FLOPs, avg 66.4 vs 68.4 | 2310.04564 Table 1 | `[TABLE]` |
| 18 | ReLU Strikes Back, Falcon-7B s2 | 95% down-proj sparsity, avg 64.8 vs 66.8 | 2310.04564 Table 1 | `[TABLE]` |
| 19 | ReLU Strikes Back finetuning | 30B tokens (s1), 50B (s2) | 2310.04564 §4.1 | `[TEXT]` |
| 20 | ProSparse-7B | 89.32% sparsity, avg 38.46 | 2402.13516 Table 1 | `[TABLE]` |
| 21 | LLaMA2-7B baseline (same metric) | avg 37.96 | 2402.13516 Table 1 | `[TABLE]` |
| 22 | ProSparse matched-token control @34.6B | Vanilla ReLU 66.04% / 41.40 vs ProSparse 89.32% / 38.46 | 2402.13516 Table 4 | `[TABLE]` |
| 23 | ProSparse matched-token control @89.13B | Vanilla ReLU 64.93% / 41.52 vs ProSparse 88.29% / 40.67 | 2402.13516 Table 4 | `[TABLE]` |
| 24 | ProSparse true retention vs matched control | 92.9% @34.6B, 97.9% @89.13B | — | `[DERIVED]` |
| 25 | ProSparse 7B wall-clock | 8×A100-80G ≈ 10 days ≈ 1920 A100-h | 2402.13516 App. | `[TEXT]` (per-model attribution ambiguous) |
| 26 | ProSparse cost vs budget | 22.4× (34.6B) / 57.8× (89.13B) | — | `[DERIVED]` |
| 27 | ProSparse 13B tokens | 134.22B | secondary | `[UNVERIFIED]` |
| 28 | TurboSparse-Mistral-7B | 2.5B of 7B active, OpenLLM avg 63.65 vs 61.57 | 2406.05955 Table 6 | `[TABLE]` |
| 29 | TurboSparse-Mixtral-47B | 4.3B of 47B active = 9.1%, avg 71.76 vs 69.34 | 2406.05955 Table 6 | `[TABLE]` |
| 30 | Turbo Sparse training | 150B tokens | 2406.05955 §4 | `[TEXT]` |
| 31 | Turbo Sparse matched-token control at 7B/47B | **absent from paper** | — | `[NOT LOCATED]` |
| 32 | Existing ReLUfication sparsity ceiling | 40% → ~67% | 2406.05955 §1 | `[TEXT]` |
| 33 | TEAL, LLaMA-2-7B | 5.07 → 5.09 (25%) → 5.22 (40%) → 5.43 (50%) → 6.62 (65%) ppl | 2408.14690 Table 1 | `[TABLE]` |
| 34 | CATS 40%, LLaMA-2-7B | 43.8 ppl (degenerate) | 2408.14690 Table 1 | `[TABLE]` |
| 35 | TEAL model-wide sparsity range | 40–50%, "most models degrade significantly at 65%" | 2408.14690 abstract/§5.1 | `[TEXT]` |
| 36 | CATS model-wide sparsity | ~25% | 2408.14690 §2 | `[TEXT]` |
| 37 | Bank-wise predictor-gated sparsity | 16 of every 64 channels = 4× (25% activation) | 2606.10722 abstract | `[TEXT]` |
| 38 | Bank-wise negative result | MoE-style balance bias does not transfer to channel routing | 2606.10722 §5 | `[TEXT]` |
| 39 | Per-neuron contiguous run, LLaMA-2-7B, 0.5 B/w | 2,048 B | d_model=4096 | `[DERIVED]` |
| 40 | Consecutive co-activating neurons for a 48 KB read | 24 (7B) / 12 (100B-class) | — | `[DERIVED]` |
| 41 | Neuralink co-activation placement gain | 1.80× I/O bandwidth, 1.49× end-to-end | 2410.19274 abstract | `[TEXT]` |
| 42 | Neuralink I/O share of latency | 73.4%–95.8% when FFN offloaded | 2410.19274 §1 | `[TEXT]` |
| 43 | Monarch projection optimality | Theorem 1, analytic optimal solution | 2204.00595 §3.3 | `[TEXT]` |
| 44 | Monarch projection algorithm | m² rank-1 SVDs of m×m slices, n=m² | 2204.00595 Alg. 1 | `[TABLE]` |
| 45 | Monarch projection cost, n=4096 | 4096 SVDs of 64×64 — CPU seconds/minutes, 0 tokens | — | `[DERIVED]` |
| 46 | Monarch-BERT-large | 79.6 GLUE vs 80.4, 144M vs 335M params, 1.7× | 2204.00595 Table 8 | `[TABLE]` |
| 47 | Monarch block count used | 2–4 blocks → 25–50% of dense params | 2204.00595 App. | `[TEXT]` |
| 48 | Monarch contiguous run, n=4096, 0.5 B/w | 128 KB fully sequential per factor | — | `[DERIVED]` |
| 49 | ProcrustesGPT, Llama2-7b, ~15% compression | 5.47 → 6.54 ppl | ACL 2025 Findings Table 1 | `[TABLE]` |
| 50 | ProcrustesGPT vs SliceGPT, Llama2-7b | 6.54 vs 7.60 ppl | same Table 1 | `[TABLE]` |
| 51 | DenoiseRotator, LLaMA3-8B SparseGPT 50% | 9.57 → 7.60 ppl (dense 6.14) | 2505.23049 Table 1 | `[TABLE]` |
| 52 | DenoiseRotator, LLaMA3-70B SparseGPT 50% | 5.99 → 4.61 ppl (dense 2.86) | 2505.23049 Table 1 | `[TABLE]` |
| 53 | DenoiseRotator cost | ~28 h, 30 GB, single A100, 70B model, 2000 steps | 2505.23049 §4.1 | `[TEXT]` |
| 54 | DenoiseRotator parameter overhead | +0.5B on LLaMA-3-8B = 6.7% of model, dense, always-on | 2505.23049 App. | `[TEXT]` |
| 55 | DenoiseRotator block-diagonal degradation | ppl 7.597 (b=1) → 8.024 (2) → 8.544 (4) → 8.882 (8) | 2505.23049 Table 9 | `[TABLE]` |
| 56 | Change-of-Basis Pruning (VGG-16/CIFAR-10) | prunes 90–96% params, 1–6% acc drop after finetune | 2511.16061 abstract | `[TEXT]` |
| 57 | CoB rotational-invariance tax | −4.52% vs ReLU control, before pruning | 2511.16061 abstract | `[TEXT]` |
| 58 | Hase et al. ROME correlation | ρ = −0.13, p < 1e-3 (GPT-J, layer 6) | 2301.04213 §4 | `[TEXT]` |
| 59 | Hase et al. variance explained | tracing adds ≤3.2%; layer-only explains 58.5% | 2301.04213 §4 | `[TEXT]` |
| 60 | Phi-Mamba-1.5B (MLP frozen) | avg 62.6 vs teacher 64.9, **3.0B tokens** | 2408.10189 Table 1 | `[TABLE]` |
| 61 | Phi-Mamba retention | 96.5% | — | `[DERIVED]` |
| 62 | Phi-Mamba cost vs budget | 2.7e19 = 0.42 × B (≈0.23 × B with MLP frozen) | — | `[DERIVED]` |
| 63 | Hybrid Phi-Mamba-1.5B | avg 66.0 vs Phi-1.5 67.2, 4 vs 24 attention layers | 2408.10189 Table 2 | `[TABLE]` |
| 64 | Mamba in the Llama | 20B tokens, MLP frozen in stage 1 only | 2408.15237 §1, §3 | `[TEXT]` |
| 65 | Mamba in the Llama wall-clock | < 5 days on 8×80G A100 (< 960 A100-h) | 2408.15237 App. | `[TEXT]` |
| 66 | Memory Layers scale | 128B memory params, 1T tokens | 2412.09764 abstract | `[TEXT]` |
| 67 | Memory Layers 8B + 64B memory, 200B tokens | MMLU 50.14 vs dense 41.35 | 2412.09764 Table 2 | `[TABLE]` |
| 68 | Memory Layers access pattern | "almost entirely memory bandwidth bound"; 3 TB/s custom kernel vs <400 GB/s PyTorch | 2412.09764 §3 | `[TEXT]` |
| 69 | Memory Layers depth limit | degrades beyond 3 memory layers | 2412.09764 §3 | `[TEXT]` |
| 70 | FPT trained fraction | ~0.1% of parameters; "learning how to query the transformer" | 2103.05247 §2 | `[TEXT]` |
| 71 | Geva et al. key-value memory characterisation | — | secondary sources only | `[UNVERIFIED]` |
| 72 | ROME/MEMIT mechanism details | — | not opened this pass | `[UNVERIFIED]` |
| 73 | Dense2MoE 225B-token budget | — | search summary only | `[UNVERIFIED]` |
| 74 | RotPruner results | — | known only via DenoiseRotator's citation | `[UNVERIFIED]` |
| 75 | Rotation making a pretrained LLM's **activations** sparse | — | searched, none found | `[NOT LOCATED]` |
| 76 | Any conversion recipe reaching ≤2% activation | — | searched, none found | `[NOT LOCATED]` |
| 77 | Monarch projection applied to a ≥7B decoder LLM with retention reported | — | searched, none found | `[NOT LOCATED]` |

---

## Ranked list: what can plausibly reach ≤2% activation within 720 T4-hours

**Ordered by evidence strength, not by attractiveness.**

### Tier 1 — inside budget, strong evidence, but does not deliver 2% alone

**1. MOHAWK-style frozen-FFN core replacement (§4's architecture).** `[TABLE]`-grade evidence: 96.5%
retention at 3.0B tokens with the MLP frozen and attention entirely discarded, at 0.42 × B. **This validates
the programme's §4 bet directly and is the best-evidenced item in the survey.** It reduces activation by
**zero percent**. Its role is to make the *architecture* legal, not to make the model sparse. Scaling: cost
∝ N, so ≈2 × B at an 8B donor with 3B tokens `[DERIVED]`, ≈28 × B at 100B. Corroborated at 8B by Mamba in
the Llama (< 960 A100-h, 20B tokens) `[TEXT]`.

**2. Monarch projection onto the donor's FFN.** Analytic, provably optimal, CPU-only, zero tokens, and
**perfectly bulk-contiguous (128 KB sequential runs at n=4096)** `[DERIVED]`. Attacks bytes-per-weight and
total-size, never activation. Best supporting evidence is BERT-scale (99.0–99.6% retention at ~2.2×);
LLM-scale training-free structured factorization only reaches ~15% compression with real degradation
(ProcrustesGPT) `[TABLE]`. **The right first experiment because it is free and it composes with everything.**

**3. MoEfication-style partition + trained router.** Router costs minutes per FFN `[TEXT]`; no backbone
retraining. Reaches 10–30% of FFN at >95% retention `[TEXT]` — on ReLU models up to ~3B. **Its own ablation
says co-activation clustering buys +0.4/+1.8/+0.8 over random split when the router is good** `[TABLE]`.
Transfer to a SwiGLU decoder LLM is untested `[NOT LOCATED]`.

**4. DenoiseRotator-style learned rotation.** 28 A100-h at 70B `[TEXT]` — the only *high-leverage* mechanism
whose published cost is unambiguously inside our budget at donor scale. But it delivers 50%/2:4 sparsity, not
90%; it adds a dense always-on `d²`-per-layer rotation (6.7% of an 8B model) `[TEXT]`; and the affordable
block-diagonal version gives back most of the benefit `[TABLE]`.

### Tier 2 — delivers the activation reduction, does not fit the budget

**5. ReLUfication (ProSparse recipe).** The single largest published activation lever: **89.32% sparsity at
92.9% retention against a matched control** `[TABLE]`. Cost **22.4 × B** at 7B; break-even donor size is
**0.31–1.48B** `[DERIVED]`. For a 100B donor, **~320 × B**.

**6. dReLU / Turbo Sparse.** The only published composition of MoE × activation sparsity, reaching
**9.1% activation** `[TABLE]` — the closest anyone has come to 2%. Cost **97–181 × B** `[DERIVED]`, and it
has **no matched-token control** at scale, so its quality claim is unattributed.

**7. Continual dense→MoE (LLaMA-MoE).** 12.5% of FFN, **124 × B**, retention vs donor unreported and
self-described as poor `[TEXT]`.

### Tier 3 — struck from the list

**8. Sparse Upcycling** — grows total parameters; does not attack the activation lever; +46–55% of original
pretraining cost `[TEXT]`.
**9. Training-free activation sparsity (TEAL/CATS)** — 0 GPU-hours, but a hard ceiling at ~50% `[TABLE]`,
25× short of the target. Useful only as a free additive on top of something else.
**10. Change-of-Basis Pruning / TSRA** — reaches 90–96% pruning `[TEXT]` but on VGG-16/CIFAR-10, requires a
new activation family and from-scratch training, and pays 4.52% before pruning. Not a donor transformation.

---

## §6 — What this changes about what to try first

**Flag 1 — the ≤2% target is arithmetically reachable, and no one has ever reached it.**
Composing the two best published mechanisms gives **12.5% (LLaMA-MoE granularity) × 11% (ProSparse residual
density) = 1.4% of FFN** `[DERIVED]` — under 2%. And because §4 discards the donor's attention entirely, the
requirement on the store for a 100B donor is `(2.1B active − 0.5B core) / 67B FFN = 2.4%` `[DERIVED]`, which
that composition clears. **The target is not arithmetically impossible.** But: the best composition ever
published is TurboSparse-Mixtral at **9.1%** `[TABLE]`, 4.5× above target, and **`[NOT LOCATED]`: no recipe of
any cost reaches ≤2% by conversion.**

**Flag 2 — and this is the one that should change the plan — the budget, not the mechanism, is the wall.**
Every mechanism that moves activation by a large factor costs 22×–180× the 720 T4-hour budget at 7B, and
~320× at 100B `[DERIVED]`. The break-even donor for the cheapest published high-sparsity recipe is
**0.3–1.5B parameters** `[DERIVED]`. **The programme's §1 reframe — "size stops being a filter" — is correct
about the byte path and wrong about the training path.** At 720 T4-hours, donor size is re-instated as a
filter by the *sparsification* cost, from the opposite direction: a 100B donor is affordable to *run* and
unaffordable to *sparsify*. This tension is not resolved anywhere in the current programme document and it
should be, before D4 is launched.

**Flag 3 — reframe D0 before spending anything on it.** The programme asks "do a dense FFN's neurons cluster
by co-activation?" MoEfication answers yes and then shows it **barely matters for quality** (+0.4/+1.8/+0.8
over random split with a good router) `[TABLE]`. **The question that is actually load-bearing for this engine
is not clustering-for-quality but clustering-for-layout:** can neurons be *permuted* so that ≥12–24
consecutively-stored neurons co-activate, turning 2 KB scatter into 48 KB bulk `[DERIVED]`? That is Neuralink's
Hamiltonian-path formulation, it is already published, and its measured payoff is **1.80× bandwidth**
`[TEXT]` — against the ~5.6× our engine needs to make gather as cheap as bulk. **D0 should be rewritten to
measure the permutation, not the partition, and its pass/fail gate should be a byte-run-length distribution,
not a clustering score.**

**Flag 4 — §3D must be corrected.** "Whether a rotation exists that makes a trained LLM's weights sparse is,
as far as I know, open" is **no longer true**: DenoiseRotator, ProcrustesGPT and RotPruner all rotate for
sparsity/compressibility, in 2025–2026, with published numbers. The genuinely open slot is narrower:
**rotating to make *activations* sparse rather than weight-importance concentrated** `[NOT LOCATED]` — which
is also the version our gate-first skip path actually needs. That is a better D2 than the one written.

**Flag 5 — the memory-layer warning.** Memory Layers at Scale proves extreme sparse activation works, and
simultaneously proves it is a random gather that needed a custom kernel to reach 3 TB/s on an H100 `[TEXT]`.
**If the knowing tier is built as a top-k lookup without solving Flag 3 first, the engine inherits the 14×
gather penalty and the entire 2% activation win is spent paying for it.** Contiguity is not a
nice-to-have downstream of density; on this engine it is a co-equal constraint and it should be gated at the
same time, not after.
