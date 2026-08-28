# What makes a linear-attention mixing carry softmax structure — and what does not

**Role:** Researcher. Prior art only. No recommendation, no promotion of a literature claim to a project fact.
**Opened:** 2026-08-28. Written incrementally, ingredient by ingredient.
**Question:** which ingredient of a linear-attention mixing is operative — decay, the delta rule, normalisation, or kernel approximation order — with the literature's own measurements, before a decay-carrying probe arm is interpreted.

## Evidence tagging (project rule)

- **[T]** = read in the paper's own table, with coordinates (paper, table, row, column).
- **[A]** = asserted in the paper's running text only, with section. Not a table cell.
- **[X]** = searched for, not found.
- A number is **never** substituted from memory or from a summary. Where a value could not be read in a cell it is marked [X] and left empty.
- "Derived" marks arithmetic I performed on [T] cells; the inputs are always shown.

Sources fetched as arXiv HTML (LaTeXML render) and read as cells. Where the render mangles a caption/table pairing that is flagged inline.

---

## 1. Gated DeltaNet, term by term

**Source.** Songlin Yang, Jan Kautz, Ali Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv:2412.06464. Versions v1 2024-12-09, v2 2025-03-05, **v3 2025-03-06 (read here; ICLR 2025 camera-ready)**. Read at `arxiv.org/html/2412.06464v3`.

### 1.1 The recurrence, decomposed

The paper builds it in three steps, each explicit in its own preliminaries.

**Step 0 — vanilla linear attention** (§2.1, attributed to Katharopoulos et al. 2020, *"when excluding normalization and query/key activations"*):

```
S_t = S_{t-1} + v_t k_t^T          o_t = S_t q_t
```

**Step 1 — add decay ("Mamba2", §2.1):**

```
S_t = α_t S_{t-1} + v_t k_t^T      o_t = S_t q_t     α_t ∈ (0,1), data-dependent, SCALAR-valued
```

with cumulative decay `γ_j = Π_{i≤j} α_i`, and — the part that matters most for us — an exactly equivalent **quadratic/parallel form**:

```
O = ( (Q K^T) ⊙ Γ ) V ,   Γ_ij = γ_i / γ_j  for i ≥ j,  0 otherwise
```

So **decay is a multiplicative modification of the causal mask itself**: `Γ` replaces the 0/1 mask `M`. Decay does not touch `Q`, `K`, or `V`, and it does not touch the feature map. §2.1 [A]. The paper notes the same recurrence appears in Gated RFA, xLSTM, Gated RetNet; that data-*independent* `γ_t` reduces to RetNet / Lightning-Attention; and that matrix-valued `γ_t` remains trainable when given outer-product structure (GLA, and successors). §2.1 [A].

**Step 2 — the delta rule instead of decay ("DeltaNet", §2.2):**

```
S_t = S_{t-1} − (S_{t-1} k_t) k_t^T + ( β_t v_t + (1−β_t) S_{t-1} k_t ) k_t^T
    = S_{t-1} (I − β_t k_t k_t^T) + β_t v_t k_t^T
```

The paper annotates the pieces itself: `S_{t-1}k_t` is labelled `v_t^old` (the value currently associated with key `k_t` — **erased**), and `β_t v_t + (1−β_t)S_{t-1}k_t` is labelled `v_t^new` (**written back**). `β_t ∈ (0,1)` is called the *writing strength*. The transition matrix `(I − β_t k_t k_t^T)` is a **generalized Householder** matrix. §2.2 [A].
Footnote 2 records that `β_t ∈ (0,2)` is admissible, allowing negative eigenvalues and unlocking state-tracking (Grazzi et al. 2024; Siems et al. 2025) — flagged because it is the hinge of the 2025 successor literature (§2).

**Step 3 — the gated delta rule (§3.1, Eq. 10):**

```
S_t = S_{t-1} ( α_t (I − β_t k_t k_t^T) ) + β_t v_t k_t^T
```

**The two ingredients are algebraically separable and compose as a product** in the state transition: a scalar shrink `α_t` composed with a rank-one Householder reflection `(I − β_t k_t k_t^T)`. Setting `β_t = 0` recovers Mamba2 exactly; setting `α_t = 1` recovers DeltaNet exactly. That is what makes the ablation clean — and the paper exploits it.

### 1.2 The paper's own separation of the two terms — Table 1 [T]

Table 1, *"Comparison of different linear RNN models and their corresponding online learning objectives using the framework from Liu et al. (2024). For convenience, we simplify Longhorn's vector-valued β to scalar β."* Transcribed verbatim:

| Method | Online Learning Objective | Online Update |
|---|---|---|
| LA | `‖S_t − S_{t-1}‖_F² − 2⟨S_t k_t, v_t⟩` | `S_t = S_{t-1} + v_t k_t^T` |
| Mamba2 | `‖S_t − α_t S_{t-1}‖_F² − 2⟨S_t k_t, v_t⟩` | `S_t = α_t S_{t-1} + v_t k_t^T` |
| Longhorn | `‖S_t − S_{t-1}‖_F² − β_t‖S_t k_t − v_t‖²` | `S_t = S_{t-1}(I − ε k_t k_t^T) + ε_t v_t k_t^T,  ε_t = β_t / (1 + β_t k_t^T k_t)` |
| DeltaNet | `‖S_t − S_{t-1}‖_F² − 2⟨S_t k_t, β_t(v_t − S_{t-1}k_t)⟩` | `S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T` |
| Gated DeltaNet | `‖S_t − α_t S_{t-1}‖_F² − 2⟨S_t k_t, β_t(v_t − α_t S_{t-1}k_t)⟩` | `S_t = S_{t-1}(α_t(I − β_t k_t k_t^T)) + β_t v_t k_t^T` |

Read the objective column: **`α_t` and `β_t` occupy different slots.** `α_t` appears only in the *regularizer* `‖S_t − α_t S_{t-1}‖²` — it relaxes how hard the new state is pinned to its predecessor, i.e. it is a **weight decay**. `β_t` appears only in the *loss* term, and it changes the loss from a negative inner product `−⟨S_t k_t, v_t⟩` (LA, Mamba2) to a regression residual `‖S_t k_t − v_t‖²` (Longhorn) — or a one-step gradient on that residual (DeltaNet). Table 1 [T] + §3.1 [A].

The paper states the SGD reading explicitly (§3.1 [A]): `S_{t+1} = S_t − β_t ∇L(S_t)` with `L(S_t) = ½‖S_t k_t − v_t‖²`, hence `S_t(I − β_t k_t k_t^T) + β_t v_t k_t^T`. So **`β_t` is a learning rate and `α_t` is a weight decay** on a test-time SGD loop. Footnote 3 concedes that Longhorn reaches a *closed-form globally optimal* update for the same objective while **DeltaNet takes one explicit gradient step** — DeltaNet is a one-step approximation, not the optimum, of its own stated objective. That is a mechanism detail worth holding: the delta rule's "erase" is one Newton-free gradient step, and its strength is a *learning rate*, not a solve.

### 1.3 Are they ablated separately? Yes — twice, two different ways

**(a) Across models, matched training — Table 2 [T].** Mamba2 = decay only; DeltaNet = delta only; Gated DeltaNet = both. Same 1.3B / 100B FineWeb-Edu setup (§4 [A]: AdamW, peak LR 4e-4, wd 0.1, grad clip 1.0, cosine schedule with 1B-token warmup, batch 0.5M tokens, Llama2 tokenizer vocab 32,000, training length 4K).

Table 2, *"Zero-shot performance comparison on S-NIAH benchmark suite for 1.3B models"*, transcribed verbatim:

| Model | S-NIAH-1 1K | 2K | 4K | 8K | S-NIAH-2 1K | 2K | 4K | 8K | S-NIAH-3 1K | 2K | 4K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DeltaNet | 97.4 | 96.8 | 99.0 | 98.8 | 98.4 | 45.6 | 18.6 | 14.4 | 85.2 | 47.0 | 22.4 |
| Mamba2 | 99.2 | 98.8 | 65.4 | 30.4 | 99.4 | 98.8 | 56.2 | 17.0 | 64.4 | 47.6 | 4.6 |
| Gated DeltaNet | 98.4 | 88.4 | 91.4 | 91.8 | 100.0 | 99.8 | 92.2 | 29.6 | 86.6 | 84.2 | 27.6 |

(S-NIAH-1 = passkey retrieval; S-NIAH-2 = number in haystack; S-NIAH-3 = uuid in haystack. S-NIAH-3 has no 8K column in the paper.)

The paper's own three-headline reading (§3.2 [A], headings verbatim):
- **"Decay hurts memory retention."** On S-NIAH-1 (synthetic repeated context, minimal information stored), *"Mamba2 degrades significantly beyond 2K sequences since it decays historical information too quickly, while Gated DeltaNet's degradation is less severe thanks to the use of delta rule."*
- **"Gating facilitates filtering."** On S-NIAH-2/3 (real-essay context, everything potentially relevant must be stored, fixed state size), *"lack of clearance causes memory collision — information becomes superimposed and indistinguishable"*; DeltaNet drops at long lengths, Mamba2 and GDN hold up.
- **"Delta rule helps memorization."** S-NIAH-3 swaps numbers for UUIDs; Mamba2 collapses (4.6 at 4K), GDN holds (27.6).

**This is the cleanest separated evidence in the literature for our question, and it does not say "decay wins".** It says the two ingredients are *anti-correlated across task type*:
- decay is operative when the state is **saturated** and must be **cleared**;
- the delta rule is operative when a **specific association must survive and be retrieved exactly**.

On the retention task (S-NIAH-1, 4K/8K) **pure delta beats gated delta**: 99.0/98.8 vs 91.4/91.8 — adding decay *costs* 7.6 and 7.0 points (derived from Table 2). On the hardest recall task (S-NIAH-3, 2K) pure decay gives 47.6 and pure delta 47.0 while the combination gives 84.2 [T] — a genuine interaction, not additivity.

**(b) Within the block, everything else fixed — Table S.1 [T].** This is the direct gate ablation: *"w. naive Delta Rule"* is the Gated DeltaNet block with the gate removed. All rows 400M params, 15B tokens, same FineWeb-Edu subset (Appendix B.2 caption [A]).

Table S.1, *"Ablation study on the Gated DeltaNet block. Avg-PPL and Avg-Acc denote average perplexity and zero-shot commonsense reasoning accuracy (as in Table 3)"*, transcribed verbatim:

| Gated DeltaNet Ablations (400M) | Avg-PPL (↓) | Avg-Acc (↑) |
|---|---|---|
| Gated DeltaNet w Head Dim 128 | 27.35 | 47.26 |
| *Macro Design* | | |
| w. naive Delta Rule | 30.87 | 45.12 |
| w/o. Short Conv | 28.95 | 46.16 |
| w/o. Output Gate | 29.12 | 45.46 |
| w/o. Output Norm | 27.55 | 47.07 |
| *Normalization & Feature Map* | | |
| w. L₁-norm & ReLU | 30.79 | 45.92 |
| w. L₁-norm & 1+ELU | 30.34 | 46.05 |
| w. L₁-norm & SiLU | 30.18 | 46.09 |
| w. L₂-norm & ReLU | 27.67 | 46.94 |
| w. L₂-norm & 1+ELU | 27.58 | 47.17 |
| *Model Dimensions* | | |
| w. Head Dim 64 | 28.31 | 46.35 |
| w. Head Dim 256 | 27.13 | 47.38 |

Derived from those cells:
- **Removing the decay gate** (`w. naive Delta Rule`, 30.87 / 45.12 vs reference 27.35 / 47.26) costs **+3.52 Avg-PPL and −2.14 Avg-Acc** at 400M/15B.
- Removing short conv: +1.60 PPL. Removing output gate: +1.77 PPL. Removing output norm: +0.20 PPL.
- The reference row is `L₂-norm & SiLU` — Figure 1 caption [A]: *"query/key paths consist of linear proj., shortconv., SiLU and L2 norm; value path includes linear proj., shortconv. and SiLU; alpha/beta use linear proj.; and output gate applies linear proj. with SiLU."* So 27.35 is directly comparable to the `L₂-norm & ReLU` 27.67 and `L₂-norm & 1+ELU` 27.58 rows.

**What is NOT in Table S.1:** a row removing the *delta rule* while keeping the gate inside this same block (`β_t = 0` → Mamba2-like). That direction exists only as the cross-model comparison in (a). Recorded as **[X] within-block delta-rule ablation** — the gate has a within-block ablation; the delta rule does not.

### 1.4 General language modelling — Table 3 [T]

Table 3, *"Performance comparison on language modeling and zero-shot common-sense reasoning"*, 1.3B / 100B FineWeb-Edu, transcribed verbatim:

| Model | Wiki. ppl↓ | LMB. ppl↓ | LMB. acc↑ | PIQA | Hella. | Wino. | ARC-e | ARC-c | SIQA | BoolQ | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RetNet | 19.08 | 17.27 | 40.52 | 70.07 | 49.16 | 54.14 | 67.34 | 33.78 | 40.78 | 60.39 | 52.02 |
| HGRN2 | 19.10 | 17.69 | 39.54 | 70.45 | 49.53 | 52.80 | 69.40 | 35.32 | 40.63 | 56.66 | 51.79 |
| Mamba | 17.92 | 15.06 | 43.98 | 71.32 | 52.91 | 52.95 | 69.52 | 35.40 | 37.76 | 61.13 | 53.12 |
| Mamba2 | 16.56 | 12.56 | 45.66 | 71.87 | 55.67 | 55.24 | 72.47 | 37.88 | 40.20 | 60.13 | 54.89 |
| DeltaNet | 17.71 | 16.88 | 42.46 | 70.72 | 50.93 | 53.35 | 68.47 | 35.66 | 40.22 | 55.29 | 52.14 |
| Gated DeltaNet | 16.42 | 12.17 | 46.65 | 72.25 | 55.76 | 57.45 | 71.21 | 38.39 | 40.63 | 60.24 | 55.32 |
| Transformer++ | 18.53 | 18.32 | 42.60 | 70.02 | 50.23 | 53.51 | 68.83 | 35.10 | 40.66 | 57.09 | 52.25 |
| Samba | 16.13 | 13.29 | 44.94 | 70.94 | 53.42 | 55.56 | 68.81 | 36.17 | 39.96 | 62.11 | 54.00 |
| Gated DeltaNet-H1 | 16.07 | 12.12 | 47.73 | 72.57 | 56.53 | 58.40 | 71.75 | 40.10 | 41.40 | 63.21 | 56.40 |
| Gated DeltaNet-H2 | 15.91 | 12.55 | 48.76 | 72.19 | 56.88 | 57.77 | 71.33 | 39.07 | 41.91 | 61.55 | 56.18 |

**The general-LM ordering inverts the recall ordering.** Decay alone (Mamba2, Avg 54.89) is within 0.43 of the combination (55.32); delta alone (DeltaNet, 52.14) is 3.18 *below* the combination and 2.75 below decay alone. Wiki ppl: Mamba2 16.56, DeltaNet 17.71, GDN 16.42 [T]. **For plain language modelling the gate carries nearly all of the gain and the delta rule is a small increment; for exact recall the reverse.** That is the mechanistic split, in the authors' own cells.

**MMLU caution (project rule).** Table 3's "Avg." is over LAMBADA-acc / PIQA / HellaSwag / WinoGrande / ARC-e / ARC-c / SIQA / BoolQ — **all commonsense multiple-choice**. **MMLU is [X]: not reported anywhere in this paper, at any scale, in any table.** Appendix B.1 [A] confirms the evaluation set. This paper says **nothing** about knowledge retention in the sense our project needs; it is a from-scratch pretraining comparison on commonsense + recall + LongBench.

### 1.5 Real-world recall — Table 4 [T]

Table 4, *"Accuracy on recall-world retrieval tasks with input truncated to 2K tokens. SQD: SQUADE. TQA: Trivial QA."* (caption typos are the paper's), tasks from Arora et al. 2024b, transcribed verbatim:

| Model | SWDE | SQD | FDA | TQA | NQ | Drop | Avg |
|---|---|---|---|---|---|---|---|
| RetNet | 14.0 | 28.5 | 7.0 | 54.4 | 16.2 | 17.3 | 22.9 |
| HGRN2 | 8.3 | 25.3 | 4.8 | 51.2 | 14.2 | 16.9 | 20.1 |
| Mamba | 9.8 | 25.8 | 3.7 | 54.3 | 14.9 | 17.4 | 21.0 |
| Mamba2 | 19.1 | 33.6 | 25.3 | 61.0 | 20.8 | 19.2 | 29.8 |
| DeltaNet | 17.9 | 30.9 | 18.4 | 53.9 | 17.3 | 18.6 | 26.2 |
| Gated DeltaNet | 25.4 | 34.8 | 23.7 | 60.0 | 20.0 | 19.8 | 30.6 |
| Transformer++ | 29.5 | 38.0 | 52.2 | 58.3 | 22.5 | 21.6 | 37.0 |
| Samba | 33.0 | 39.2 | 50.5 | 57.7 | 23.5 | 20.2 | 37.3 |
| Gated DeltaNet-H1 | 35.6 | 39.7 | 52.0 | 60.1 | 24.6 | 22.2 | 39.0 |
| Gated DeltaNet-H2 | 38.2 | 40.4 | 50.7 | 63.3 | 24.8 | 23.3 | 40.1 |

Three things this table says that the abstract does not:

1. **On real-world recall, pure delta loses to pure decay on 6 of 6 tasks.** DeltaNet vs Mamba2: SWDE 17.9<19.1, SQD 30.9<33.6, FDA 18.4<25.3, TQA 53.9<61.0, NQ 17.3<20.8, Drop 18.6<19.2; Avg 26.2 vs 29.8 [T]. The paper acknowledges it in §4 [A]: *"despite DeltaNet's superior performance on synthetic in-context retrieval tasks, its real-world retrieval performance lags behind Mamba2."* **The synthetic-recall story and the real-recall story disagree about the delta rule.**
2. The combination's margin over the better single ingredient is **+0.8 Avg** (30.6 vs 29.8, derived) — far smaller than Table 2's synthetic margins. The paper attributes the shrinkage to repetition errors in instruction-unaligned small models being the dominant error source, *"largely independent of the update rule choice"* §4 [A]. That is an honest confound and it is the authors' own — but note it also means **Table 4 has low power to discriminate update rules at all**.
3. **Every pure recurrent model, gated or not, sits 6–8 Avg points below Transformer++ (37.0).** FDA is the extreme: 25.3 / 18.4 / 23.7 for the recurrents vs 52.2 for the Transformer [T]. No update rule closes that gap; only hybrids containing real attention do (39.0 / 40.1). **The choice among linear mixings is a second-order choice compared to the presence of some exact attention.**

### 1.6 Where this paper's evidence stops

- **MMLU: [X].** Not reported at any scale.
- **Conversion / uptraining from a donor: [X].** Every number is from-scratch pretraining under matched budgets. Nothing here licenses a claim about what these ingredients do when a pretrained softmax model is *converted*.
- §4 [A] says Table 3 covers *"models with 400M and 1.3B parameters"*, but the Table 3 block readable in the v3 HTML has a single set of rows and the setup paragraph states 1.3B/100B. The 400M block was not readable in the render. **The numbers above are the ones actually read in cells; the 400M block is [X] here.**
- Table S.1's ablations are 400M/15B — a much smaller budget than Table 2/3/4's 1.3B/100B. **Ablation ranking and headline ranking come from different budgets.**
- **No seed replicates, no error bars, anywhere in the tables read. [X]** Every delta above is a single-run difference. Against this project's own σ_seed discipline: the +0.43 Avg (delta rule on Table 3) and +0.8 Avg (Table 4) are **not separable from run noise on the evidence presented**. The +3.52 PPL gate ablation and the 40–80 point S-NIAH swings are large enough that noise is not a plausible explanation.

---

## 2. The DeltaNet lineage — what the delta rule buys over plain decay

### 2.1 Origin: Schlag, Irie, Schmidhuber (2021) — the capacity argument

**Source.** *Linear Transformers Are Secretly Fast Weight Programmers*, arXiv:2102.11174, v3 read at `arxiv.org/html/2102.11174v3` (ICML 2021).

**The mechanism claim, §4.1 [A], quoted:**

> "Endlessly adding new associations to a memory of finite size … inevitably will reach a limit. In linear attention, information is stored in a matrix and is retrieved using matrix multiplication. As a consequence, **to prevent associations from interfering with each other upon retrieval, the respective keys need to be orthogonal.** Otherwise, the dot product will attend to more than one key and return a linear combination of values. **With keys embedded in a `d_dot` space, there cannot be more than `d_dot` orthogonal vectors. That is, storing more than `d_dot` associations will result in a retrieval error.** In linear Transformers, when the length of the sequence is longer than `d_dot`, the model might be in such an overcapacity regime."

The paper grounds this in Smolensky (1990) tensor-product representations, citing Theorems 3.1 and 3.3 for the formal crosstalk/retrieval-error statement. §4.1 [A].

**This is the same object as our R2a analytic floor.** `sqrt(1 − rank/T) = 0.9015` at `T/D = 5.33` is the residual of projecting a `T`-dimensional target onto a `rank`-dimensional span — i.e. the crosstalk floor Schlag describes, expressed as an unreachable fraction rather than a retrieval-error probability. Recorded as a structural correspondence only; nobody in this literature computes our quantity.

**Their measurement — Table 2 [T]**, *"WikiText-103 language model perplexity results showing effects of our update rule."* Parameter counts matched *"up to the small difference introduced by gating in our update rule (16K and 33K parameters respectively for the small and medium configurations)"*. small = `D=128, L=256, H=8` (so `d_dot = 16`), 40M params; medium = `D=256, L=384`, 90M params. *"Both configurations represent an overcapacity regime."* §6.1.2 [A].

| Model | Update Rule | small Valid | small Test | medium Valid | medium Test |
|---|---|---|---|---|---|
| Transformer | – | 33.0 | 34.1 | 27.9 | 29.6 |
| Linear Transformer | sum | 37.1 | 38.3 | 31.1 | 33.0 |
| Delta Network | delta | 34.1 | 35.5 | 29.7 | 31.5 |
| Performer | sum | 39.0 | 39.6 | 32.2 | 33.8 |
| Performer | delta | 36.1 | 37.2 | 30.0 | 31.8 |

**This is the most useful table in the lineage for our question, because it crosses the update rule with the feature map.** Derived from those cells:
- Delta rule on the elementwise (`ELU+1`) kernel: **−3.0 / −2.8 ppl** (small valid/test), **−1.4 / −1.5** (medium).
- Delta rule on the random-feature (Performer) kernel: **−2.9 / −2.4** (small), **−2.2 / −2.0** (medium).
- **The delta rule's benefit is essentially kernel-independent** — it improves both an elementwise positive map and a random-feature map by a similar margin. Update rule and feature map are separable ingredients and both are load-bearing.
- The delta rule does **not** close the gap to softmax: Delta Network 34.1/35.5 vs Transformer 33.0/34.1 (small); 29.7/31.5 vs 27.9/29.6 (medium). Residual gap 1.1–1.9 ppl.

**Their normalisation ablation — Table 3 [T]**, *"WikiText-103 language model perplexities for Linear Transformers (medium configuration) with our update rule."*

| Position Encoding | Attn. Normalisation | Valid | Test |
|---|---|---|---|
| Yes | Yes | 30.4 | 32.1 |
| No | Yes | 29.2 | 31.2 |
| Yes | No | 29.7 | 31.5 |
| No | No | 28.1 | 31.1 |

Caveat in the surrounding text [A]: *"The sum normalisation (Sec. 4.2) is used in all cases: the models diverged otherwise."* So a **sum normalisation is mandatory for stability** — its removal is not even a runnable ablation — while the *extra attention normalisation on top of it* is harmful (−1.1 valid ppl when removed). **Normalisation is not one knob; it is at least two, with opposite signs.**

### 2.2 Yang, Wang, Zhang, Shen, Kim (2024) — parallelising DeltaNet, and the L2 mechanism

**Source.** *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, arXiv:2406.06484. Versions v1 2024-06-10 through **v6 2025-01-15 (read here)**; NeurIPS 2024. Read at `arxiv.org/html/2406.06484v6`.

**Why L2 normalisation is mechanistic, not cosmetic — §3.3 [A], quoted:**

> "For stability, it is crucial to ensure that the norm of each eigenvalue of the transition matrices does not exceed one. The eigenvalues of `I − β_t k_t k_t^T` are 1 with multiplicity `d−1` and `1 − β_t‖k_t‖²` with multiplicity 1. Schlag et al. used the `L1` norm to normalize query/key vectors, ensuring that `0 ≤ 1 − β_t‖k_t‖² ≤ 1`. **We instead apply `L2` normalization, which we found to perform better and offers a more intuitive interpretation: when `β_t = 1`, `I − k_t k_t^T` becomes a projection matrix, erasing information in one subspace while preserving the other `d−1` subspaces.** This is beneficial for retaining information while enabling more targeted forgetting."

**Read what that says about the framing of the question.** The normalisation is not an ingredient *separate* from the delta rule — **it is what determines what the delta rule's erase operation actually is.** Under `L2` normalisation with `β=1` the transition is an exact orthogonal projector removing exactly one direction and preserving `d−1`. Under any other normalisation it is a non-orthogonal contraction of unspecified geometry. **"Normalisation" and "delta rule" cannot be ablated independently and read cleanly** — the norm sets the spectrum of the delta rule's transition matrix.

Their block: `k_t = SiLU(W_K x_t) / ‖SiLU(W_K x_t)‖₂`, same for `q_t`. §3.3 [A].

**Their feature-map / normalisation ablation — Table 1 (bottom) [T]**, 340M params / 15B tokens, SlimPajama subset, Mistral tokenizer. Column order as printed: Wiki ppl↓ | LMB ppl↓ | LMB acc↑ | PIQA | Hella. | Wino. | ARC-e | ARC-c | Avg. | SWDE | SQuAD | FDA | State exp.

| DeltaNet Ablations (340M) | Wiki ppl | LMB ppl | LMB | PIQA | Hella | Wino | ARC-e | ARC-c | Avg. | SWDE | SQuAD | FDA | State |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w. L₁-norm & 1+ELU | 31.12 | 55.96 | 26.3 | 63.9 | 33.0 | 50.9 | 44.3 | 21.8 | 40.1 | 14.5 | 23.9 | 6.2 | 128x |
| w. L₂-norm & 1+ELU | 28.03 | 37.62 | 32.2 | 65.7 | 34.7 | 51.8 | 45.4 | 22.5 | 42.1 | 23.8 | 28.6 | 13.1 | 128x |
| w. L₂-norm & ReLU | 28.75 | 43.53 | 30.2 | 64.0 | 33.9 | 48.9 | 45.6 | 22.8 | 40.9 | 27.2 | 26.7 | 9.0 | 128x |

and the reference rows from the same 340M block of the main table (the block's default is `L₂-norm & SiLU`, per §3.3):

| DeltaNet (w. conv) = L₂-norm & SiLU | 28.24 | 37.37 | 32.1 | 64.8 | 34.3 | 52.2 | 45.8 | 23.5 | 42.1 | 26.4 | 28.9 | 12.8 | 128x |
| DeltaNet (w/o. conv) | 29.08 | 50.87 | 30.0 | 63.6 | 33.6 | 51.7 | 46.0 | 23.0 | 41.3 | 24.6 | 26.9 | 4.5 | 128x |

**Effect sizes, derived from those cells:**
- **`L1 → L2` with the feature map held at `1+ELU`: Wiki ppl 31.12 → 28.03 = −3.09; LMB ppl 55.96 → 37.62 = −18.34; downstream Avg 40.1 → 42.1 = +2.0; FDA 6.2 → 13.1.**
- **Feature map varied within `L2`: SiLU 28.24, `1+ELU` 28.03, ReLU 28.75 — a spread of 0.72 Wiki ppl, and `1+ELU` is *better* than SiLU on Wiki ppl.** Downstream Avg: SiLU 42.1, `1+ELU` 42.1, ReLU 40.9.
- **The normalisation choice is worth ~4× the entire spread of the feature-map choice on Wiki ppl, and ~25× on LMB ppl.**

**A text/table discrepancy worth recording.** §4.2 [A] states *"For the feature map, we experiment with {ReLU, 1+ELU, SiLU} and find that SiLU performs the best, consistent with prior work."* **The cells do not support that column by column** — `L₂-norm & 1+ELU` = 28.03 beats the SiLU reference 28.24 on Wiki ppl and ties it on downstream Avg (42.1 / 42.1); SiLU wins on LMB ppl (37.37 vs 37.62), SWDE (26.4 vs 23.8) and is level on FDA (12.8 vs 13.1). "SiLU best" is a defensible *aggregate* reading, not a per-column fact. Marked **[A], contradicted in part by [T]**. The direction that matters to us is unaffected: *the norm dominates; the feature map is near-noise.*

Gated DeltaNet's Table S.1 independently reproduces this at 400M/15B on FineWeb-Edu (§1.3b above), same shape: L₁ rows 30.18–30.79, L₂ rows 27.35–27.67 — a ~3-PPL norm effect against a ~0.3-PPL feature-map spread. **Two papers, two datasets, two tokenizers, same finding.** This is the best-corroborated result in the file.

### 2.3 What the delta rule buys, per task type — MAD, Figure 3 [T]

*"Results on the synthetic MAD benchmark. Results other than DeltaNet are directly borrowed from Poli et al. (Multi-head) Hyena, DeltaNet and Mamba make use of convolutions, whereas GLA does not."*

| Model | Compress | Fuzzy Recall | In-Context Recall | Memorize | Noisy Recall | Selective Copy | Average |
|---|---|---|---|---|---|---|---|
| Transformer | 51.6 | 29.8 | 94.1 | 85.2 | 86.8 | 99.6 | 74.5 |
| Hyena | 45.2 | 7.9 | 81.7 | 89.5 | 78.8 | 93.1 | 66.0 |
| Multihead Hyena | 44.8 | 14.4 | 99.0 | 89.4 | 98.6 | 93.0 | 73.2 |
| Mamba | 52.7 | 6.7 | 90.4 | 89.5 | 90.1 | 86.3 | 69.3 |
| GLA | 38.8 | 6.9 | 80.8 | 63.3 | 81.6 | 88.6 | 60.0 |
| DeltaNet | 42.2 | 35.7 | 100 | 52.8 | 100 | 100 | 71.8 |

**The delta rule's signature is unmistakable, and so is its cost.** DeltaNet is at **100 / 100 / 100** on In-Context Recall, Noisy Recall and Selective Copy, and at **35.7 on Fuzzy Recall — above the Transformer's 29.8 and ~5× the decay-based models (Mamba 6.7, GLA 6.9)**. And it is **worst in the table on Memorize: 52.8**, against Mamba 89.5, Hyena 89.5, Transformer 85.2, GLA 63.3. The paper concedes it §4.1 [A]: *"DeltaNet is better at recalling tasks, especially on Fuzzy Recall as expected, although it somehow struggles on the 'Memorize' task."*

**That is the trade in one table: erase-then-write buys in-context association at the cost of durable storage.** Which is the same mechanism Gated DeltaNet's Table 2 re-derives on a different benchmark — and the reason the gate was added.

### 2.4 The lineage's own reason for wanting a gate — §5.3 [A]

> "We also found that the length generalization of DeltaNet was limited, while GLA and RetNet (and Mamba to an extent) have been found to be able to extrapolate beyond the training length. **We speculate that this is because DeltaNet lacks explicit decay factors.** This could be improved through incorporating a gating term in the recurrence, as demonstrated in a recent work by Yang et al."

Note the verb: **"We speculate."** The decay gate enters this lineage as an *unverified hypothesis about length extrapolation*, later supported by Gated DeltaNet's Table 2 on a different axis (memory clearance, not extrapolation). Recorded as such.

### 2.5 Delta versus decay in one table, and the scale reversal — Table 1 main [T]

340M / 15B, SlimPajama, Mistral tokenizer:

| Model | Wiki ppl | LMB ppl | LMB | PIQA | Hella | Wino | ARC-e | ARC-c | Avg. | SWDE | SQuAD | FDA | State |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer++ | 28.39 | 42.69 | 31.0 | 63.3 | 34.0 | 50.4 | 44.5 | 24.2 | 41.2 | 42.2 | 22.1 | 21.4 | N/A |
| RetNet (w/o. conv) | 32.33 | 49.19 | 28.6 | 63.5 | 33.5 | 52.5 | 44.5 | 23.4 | 41.0 | 13.3 | 27.6 | 2.9 | 512x |
| Mamba (w. conv) | 28.39 | 39.66 | 30.6 | 65.0 | 35.4 | 50.1 | 46.3 | 23.6 | 41.8 | 12.4 | 23.0 | 2.1 | 64x |
| GLA (w/o. conv) | 28.65 | 43.35 | 30.3 | 64.8 | 34.5 | 51.4 | 45.1 | 22.7 | 41.5 | 18.6 | 27.2 | 8.1 | 128x |
| GLA (w. conv) | 29.47 | 45.53 | 31.3 | 65.1 | 33.8 | 51.6 | 44.4 | 24.6 | 41.8 | 24.0 | 24.7 | 7.3 | 128x |
| DeltaNet (w/o. conv) | 29.08 | 50.87 | 30.0 | 63.6 | 33.6 | 51.7 | 46.0 | 23.0 | 41.3 | 24.6 | 26.9 | 4.5 | 128x |
| DeltaNet (w. conv) | 28.24 | 37.37 | 32.1 | 64.8 | 34.3 | 52.2 | 45.8 | 23.5 | 42.1 | 26.4 | 28.9 | 12.8 | 128x |
| + Sliding Attn | 27.06 | 38.17 | 33.4 | 64.0 | 35.3 | 50.9 | 45.9 | 23.2 | 42.1 | 39.3 | 32.5 | 18.8 | N/A |
| + Global Attn (2 layers) | 27.51 | 35.04 | 33.5 | 64.0 | 34.5 | 51.7 | 46.0 | 23.3 | 42.1 | 42.9 | 32.1 | 23.1 | N/A |

1.3B / 100B:

| Model | Wiki ppl | LMB ppl | LMB | PIQA | Hella | Wino | ARC-e | ARC-c | Avg. | SWDE | SQuAD | FDA | State |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer++ | 16.85 | 13.44 | 48.9 | 70.8 | 49.6 | 53.6 | 56.0 | 26.5 | 50.9 | 66.6 | 31.5 | 27.4 | N/A |
| RetNet (w/o. conv) | 18.64 | 17.27 | 43.3 | 70.0 | 47.3 | 52.5 | 54.8 | 25.6 | 48.9 | 42.8 | 34.7 | 14.3 | 512x |
| Mamba (w. conv) | 17.06 | 13.89 | 46.2 | 72.2 | 40.1 | 54.1 | 59.0 | 28.2 | 50.0 | 41.4 | 35.2 | 6.2 | 64x |
| GLA (w/o. conv) | 17.22 | 14.47 | 46.9 | 71.8 | 49.8 | 53.9 | 57.2 | 26.6 | 51.0 | 50.6 | 42.6 | 19.9 | 256x |
| GLA (w. conv) | 17.25 | 14.92 | 46.2 | 70.6 | 49.9 | 53.0 | 55.3 | 27.0 | 50.4 | 52.4 | 37.4 | 22.3 | 256x |
| DeltaNet (w. conv) | 16.87 | 12.21 | 48.9 | 71.2 | 50.2 | 53.6 | 57.2 | 28.3 | 51.6 | 49.5 | 37.4 | 17.2 | 128x |
| + Sliding Attn | 16.56 | 11.74 | 49.2 | 71.8 | 51.1 | 52.8 | 58.9 | 28.8 | 52.1 | 53.3 | 43.3 | 22.3 | N/A |
| + Global Attn (2 layers) | 16.55 | 12.40 | 48.8 | 70.8 | 50.7 | 54.2 | 58.4 | 28.1 | 51.8 | 71.0 | 43.0 | 29.8 | N/A |

**The scale reversal, in the authors' own words** §4.2 [A]: *"under the same state size at the 340M scale, DeltaNet outperforms GLA, confirming the effectiveness of the delta rule. However, **at the 1.3B scale, DeltaNet underperforms GLA** due to its poorer state size scalability, since state size plays an important role in recall-intensive tasks."* Checked in cells — recall tasks (SWDE / SQuAD / FDA) at 340M: DeltaNet(conv) 26.4 / 28.9 / 12.8 vs GLA(conv) 24.0 / 24.7 / 7.3 → DeltaNet wins 3/3. At 1.3B: DeltaNet 49.5 / 37.4 / 17.2 vs GLA(conv) 52.4 / 37.4 / 22.3 → DeltaNet loses SWDE and FDA, ties SQuAD. **The delta rule's recall advantage over data-dependent decay does not survive a 4× scale-up at fixed state expansion.** First-class counter-evidence, and it is the authors' own.

Also note the **short convolution** column: DeltaNet w/o conv → w. conv moves FDA 4.5 → 12.8 and LMB ppl 50.87 → 37.37 [T]. **A depthwise short conv is worth more than the entire feature-map choice and is comparable to the norm choice.** Gated DeltaNet's Table S.1 agrees (`w/o. Short Conv` = +1.60 Avg-PPL). Neither paper calls this a "mixing ingredient", but by effect size it outranks two of the four ingredients we were asked about.

### 2.6 The one MMLU column in this lineage — Figure 5 / "Table 6" [T]

*"Zero-shot model performance across selected benchmarks for 3B models. Llama-3.2-3B and PowerLM-3B are Transformer models, while the others are recurrent models. ARC results are averaged over normalized accuracy across ARC-Easy and ARC-Challenge."* DeltaNet-3B trained on 1T tokens *"using the same settings as Shen et al."* (= PowerLM-3B) §4.2 [A].

| Model | ARC | HellaSwag | OBQA | PIQA | WinoGrande | **MMLU** | Average |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | 59.1 | 73.6 | 43.4 | 77.5 | 69.2 | **54.1** | 62.8 |
| PowerLM-3B | 60.5 | 74.6 | 43.6 | 79.9 | 70.0 | **45.0** | 62.3 |
| DeltaNet-3B | 60.4 | 72.8 | 41.0 | 78.5 | 65.7 | **40.7** | 59.8 |
| RecurrentGemma-2B | 57.0 | 71.1 | 42.0 | 78.2 | 67.6 | **31.8** | 57.9 |
| RWKV-6-3B | 49.5 | 68.6 | 40.6 | 76.8 | 65.4 | **28.4** | 54.9 |
| Mamba-2.7B | 50.3 | 65.3 | 39.4 | 75.8 | 63.1 | **26.1** | 53.3 |

**This is the MMLU column the project keeps asking for, and it is brutal.** Against its *training-matched* transformer control PowerLM-3B, DeltaNet-3B is: ARC −0.1, HellaSwag −1.8, OBQA −2.6, PIQA −1.4, WinoGrande −4.3, **MMLU −4.3**, Average −2.5 (all derived). MMLU ties for the largest absolute drop and is the largest *relative* drop (−9.6% vs −6.1% for WinoGrande, derived). **The average understates the MMLU damage by ~1.7×.**

And the non-delta recurrents sit at or near chance: **Mamba-2.7B MMLU 26.1, RWKV-6-3B 28.4** — 4-way MMLU chance is 25.0. Their Averages (53.3, 54.9) read as an 8–9 point deficit; their MMLU is a *floor result*. **A macro-average over commonsense tasks can hide a total loss of MMLU capability.** That is the concrete demonstration of this project's MMLU rule, in a published table.

**Confound, flagged:** the paper states these baselines are *"trained for a different number of tokens so are not exactly comparable"* §4.2 [A]. The DeltaNet-3B ↔ PowerLM-3B pair is the only matched comparison here; the DeltaNet ↔ Mamba/RWKV MMLU gaps are **not** attributable to the update rule on this evidence.

---

## 3. Kernel approximation order — why the 2nd-order Taylor map separates

This is the section that most changes the reading of our R2a table, so it is written with the mechanism first.

### 3.1 What the 2nd-order Taylor map actually is

**Source A.** Michael Zhang, Kush Bhatia, Hermann Kumbong, Christopher Ré, *The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry*, arXiv:2402.04347 (ICLR 2024). arXiv HTML was empty; read via **ar5iv** at `ar5iv.labs.arxiv.org/html/2402.04347`.

**Source B.** Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, Christopher Ré, *Simple linear attention language models balance the recall-throughput tradeoff* ("Based"), arXiv:2402.18668, **v2 read** at `arxiv.org/html/2402.18668v2`.

Both give the same feature map, explicitly. Hedgehog §4.1 [A]:

```
φ_taylor(x) = [ 1, x_1, …, x_d ] ∪ [ x_i · x_j  |  i, j ∈ [d] ]
so that  φ(q)ᵀφ(k) = 1 + qᵀk + (qᵀk)²/2  ≈ exp(qᵀk)
```

Based §4.1 [A] gives the identical inner-product form (its Eq. 4) and states the dimension: `φ : R^d → R^{d²}`, `d̃ = d²`, giving `O(N d³)` time and an `O(d³)` recurrent state; Based reduces this by projecting `q,k` to `d' = 16` first.

**The single most important structural fact:** *the order of the expansion IS the feature dimension.* A `p`-degree polynomial map sends `R^d → R^{O(d^p)}`. Hedgehog §4.1 [A]: *"feature maps for `p`-degree polynomial approximations can be computed in `O(n d^p)` time and space for every query and key vector."* Based's Table 6 prints the effective dimension in parentheses next to `d'`, and the arithmetic is `1 + d' + d'(d'+1)/2`: `d'=8 → 45`, `d'=16 → 153`, `d'=24 → 325`, `d'=32 → 561` (verified against the printed cells).

**So "kernel approximation order" and "state size" are the same knob wearing two hats.** An elementwise map (`elu+1`, ReLU, SiLU) is order-1: `d̃ = d`. The Based map is order-2: `d̃ ≈ d²/2`. The two arms are not competing on fidelity-of-approximation at matched capacity; they differ in capacity by a factor of `d/2`.

### 3.2 Based's own argument — and it is NOT "Taylor beats elu+1"

Based §4.1 [A], quoted in full because the received summary of this paper is wrong:

> "In Figure 3 (top), we plot the memory-recall tradeoff curves for these feature maps. **The Taylor series feature map, along with the simple `φ_PosELU` and `φ_ReLU` feature maps, sits at the Pareto frontier.** One advantage of the Taylor feature map over these alternatives is that **it expands the recurrent state size (improving recall capacity) without changing the number of parameters.** As shown in Figure 3 (bottom), the Taylor series feature map requires fewer parameters than alternatives to achieve high recall capacity. This analysis and the ablations in Table 6 informed our decision to use the Taylor approximation, **though other simple feature maps may be effective as well.**"

(`φ_PosELU(x) = ELU(x) + 1` — i.e. exactly our `elu(x)+1` arm.)

**Based does not claim the Taylor map beats `elu+1` at matched state size. It claims `elu+1` is on the same Pareto frontier, and that the Taylor map reaches a given state size with fewer parameters.** The claimed advantage is *parameter efficiency for a given capacity*, not representational superiority.

Their Table 6 confirms it at the cell level. **Table 6 [T]**, *"Ablations. All models are 362M param variants of the Based architecture … trained to 10 billion tokens on the Pile."* Columns: Feat. Map | Feat. Dim `d'` (effective dim in parens) | Sliding | Convs. | Decay | All-Ppl↓ | AR-Ppl↓ | Other-Ppl↓ | SWDE Acc↑ | FDA Acc↑ | SQUAD Acc↑

| Feat. Map | Feat. Dim (eff.) | Sliding | Convs | Decay | All Ppl | AR Ppl | Other Ppl | SWDE | FDA | SQUAD |
|---|---|---|---|---|---|---|---|---|---|---|
| Taylor Exp. (2nd) | 16 (153) | ✓(64) | ✓ | ✓ | 8.65 | 2.07 | 9.64 | 29.16 | 11.71 | 25.07 |
| Performer | 16 (16) | ✓(64) | ✓ | ✓ | 9.08 | 8.53 | 11.62 | 8.10 | 0.36 | 7.47 |
| CosFormer | 16 (32) | ✓(64) | ✓ | ✓ | 9.03 | 2.42 | 9.98 | 19.35 | 7.71 | 24.63 |
| CosFormer | 64 (128) | ✓(64) | ✓ | ✓ | 8.82 | 2.18 | 9.80 | 25.47 | 9.07 | 27.85 |
| Taylor Exp. (2nd) | 32 (561) | ✓(64) | ✓ | ✓ | 8.56 | 2.00 | 9.57 | 37.62 | 12.89 | 26.74 |
| Taylor Exp. (2nd) | 24 (325) | ✓(64) | ✓ | ✓ | 8.58 | 2.02 | 9.58 | 34.38 | 20.87 | 24.77 |
| Taylor Exp. (2nd) | 16 (153) | ✓(64) | ✓ | ✓ | 8.65 | 2.07 | 9.64 | 29.16 | 11.71 | 25.07 |
| Taylor Exp. (2nd) | 8 (45) | ✓(64) | ✓ | ✓ | 8.77 | 2.18 | 9.75 | 23.40 | 12.79 | 22.35 |
| Taylor Exp. (2nd) | 16 (153) | ✓(64) | ✓ | **✗** | 8.65 | 2.04 | 9.66 | 22.95 | 12.34 | 27.45 |
| Taylor Exp. (2nd) | 16 (153) | **✗** | ✓ | ✓ | 8.91 | 2.11 | 9.94 | 28.62 | 10.16 | 24.5 |
| Taylor Exp. (2nd) | 16 (153) | ✓(64) | **✗** | ✓ | 8.74 | 2.09 | 9.74 | 24.66 | 2.36 | 18.87 |
| Taylor Exp. (2nd) | 24 (325) | **✗** | **✗** | ✓ | 9.49 | 2.29 | 10.58 | 19.62 | 8.71 | 11.33 |
| Taylor Exp. (2nd) | 16 (153) | ✓(**128**) | ✓ | ✓ | 8.61 | 2.06 | 9.60 | 32.13 | 14.39 | 31.84 |

**Sort the feature-map block by effective dimension and the ordering is monotone in dimension, not in kernel identity:**

| effective dim | kernel | SWDE | AR Ppl |
|---|---|---|---|
| 16 | Performer | 8.10 | 8.53 |
| 32 | CosFormer | 19.35 | 2.42 |
| 45 | Taylor (d'=8) | 23.40 | 2.18 |
| 128 | CosFormer | 25.47 | 2.18 |
| 153 | Taylor (d'=16) | 29.16 | 2.07 |
| 325 | Taylor (d'=24) | 34.38 | 2.02 |
| 561 | Taylor (d'=32) | 37.62 | 2.00 |

(Derived by re-sorting [T] cells; no new numbers.) CosFormer at 128 effective dims (SWDE 25.47) sits between Taylor at 45 (23.40) and Taylor at 153 (29.16) — **exactly where its dimension puts it.** The paper says so itself §6.1 [A]: *"We observe with larger state size, CosFormer closes the gap to the Taylor map though note the projections increase the parameter count."* And Appendix E.1 [A]: *"We observe that with the larger state size, CosFormer quality is increasingly competitive with the Taylor map. We note that expanding the state size requires increasing the model's overall parameter count (due to the learned projections) for CosFormer, in contrast to the Taylor map."*

**Verdict on item 3, part one: Based's argument is a capacity argument, not an approximation-order argument.** The order buys dimension for free; dimension buys recall. There is no claim in Based that the *shape* of the quadratic kernel matters beyond the dimension it induces.

### 3.3 Based's theory — recall capacity is bounded by state size, full stop

Theorem 3.1 [A] (proved as Theorem F.12): *"Any recurrent model depending causally on input `u ∈ {0,1}^{N×d}` requires `Ω(N)`-bits in state size to solve MQAR."* With footnotes: for Mamba specifically see Corollary F.13; *"we need the entries of the state to be bounded."* The paper's gloss: *"This result suggests that the tradeoff observed in Figure 2 is fundamental, not an artifact of architectural quirks."*

Companion result from the same group — **Zoology**, arXiv:2312.04927 (`arxiv.org/html/2312.04927v1`), §1 [A]: *"gated convolutions … require model dimension that grows with sequence length (Theorem 4.4) while attention can solve MQAR with model dimension independent of sequence length (Proposition 4.3)."* And: *"We show that input-dependent sequence mixing is important to solve MQAR efficiently (Theorem 4.5)."*

**Put together with Schlag's `d_dot` orthogonality argument (§2.1), three independent formalisations say the same thing: for a fixed-state sequence mixer, recall capacity is set by state dimension.** Nothing in this literature identifies a *shape* of the mixing that buys structure at fixed dimension. That is the most robust mechanistic answer available to the question as asked.

### 3.4 Hedgehog's argument — and it IS about kernel shape, but with two named properties

Hedgehog is the one paper that argues for a property of the kernel beyond its dimension. §3 [A] names two:

- **Low-entropy "spikiness."** *"Mechanically, the softmax over query-key dot products exponentiates relative similarities between a query and each key, quantified via low-entropy or 'spiky' attention weight distributions."* Prior linear maps produce *"much higher entropy or more uniform distributions … This is true even for methods designed to approximate the softmax under mean-squared error bounds (Performer) or imposed locality (cosFormer)."*
- **Dot-product monotonicity.** *"when the dot product increases (decreases), the attention weight increases (decreases) … we believe this can cause training issues after swapping attentions due to conflicting gradients between attentions and original model parameters."*

**Table 2 [T]**, *"Summary of feature maps compared to softmax, exhibiting an efficiency vs. expressivity tradeoff."*

| Method | Complexity | Spiky? | Monotonic? | Train-from-scratch (AR acc) | BERT-FT (Matthew's corr.) |
|---|---|---|---|---|---|
| Softmax | `O(n²d)` | ✓ | ✓ | **100.0** | **58.8** |
| 1 + ELU | `O(nd²)` | ✗ | ✗ | **17.0** | 28.1 |
| Performer | `O(nd'²)` | ✗ | ✗ | **17.0** | 24.7 |
| CosFormer | `O(nd²)` | ✗ | ✗ | **17.0** | 39.9 |
| Taylor Exp | `O(nd³)` | ✓ | ✓ | **100.0** | **58.4** |

**This is the largest separation between `elu+1` and 2nd-order Taylor anywhere in the literature I read: 17.0 vs 100.0 on associative recall from scratch, and 28.1 vs 58.4 Matthew's correlation on BERT conversion, against a softmax ceiling of 58.8.** It is also the arm structure closest to our R2a's separated arm.

**But read the complexity column.** `1+ELU` is `O(nd²)`; Taylor Exp is `O(nd³)`. Hedgehog itself calls this out §4.1 [A]: *"Unfortunately, the 2nd-degree Taylor approximation is not efficient. Even with `p = 2`, the feature map dimension is now `d' = 1 + d + d²`, resulting in `O(nd³)` attention complexity. As summarized in Table 2, this introduces an **efficiency-effectiveness trade-off** among functional attention approximations."* **The Taylor row in Table 2 is not dimension-matched to the `1+ELU` row.** So even Hedgehog's separation is confounded with a `d`-fold capacity increase. Neither paper isolates kernel shape from kernel dimension. Recorded as **[X]: no dimension-matched kernel-shape comparison found in any source read.**

**Table 1 [T]** (finetuned-conversion of BERT-base finetuned on CoLA, Matthew's correlation):

| BERT-FT | 1 + ELU | ReLU | Performer | cosFormer | exp(t=1) | exp(t=2) |
|---|---|---|---|---|---|---|
| 58.8 | 28.1 | 39.5 | 24.7 | 39.9 | 45.9 | 50.0 |

The `exp(t)` rows are Hedgehog's deliberate **planted control**: §3.2 [A] *"a simple feature map designed to induce 'spikiness' but not monotonicity: `φ_t(x) = exp(x·t)`, which applies a temperature-`t` scaled exponential element-wise."* Its behaviour splits cleanly across regimes — from scratch, *"increasing spikiness with `t=2` actually solves the task"* [A]; on conversion it reaches only 50.0 vs softmax 58.8 [T]. **Spikiness alone suffices from scratch and does not suffice for conversion.** That is a measured from-scratch/conversion dissociation, with a control designed to isolate it, and it is the cleanest such result I found.

### 3.5 The bounded-dot-product precondition — a checkable assumption for our donor

Hedgehog §4.1 [A], on why the Taylor map works at all:

> "as a general property of polynomials, the Taylor approximation only tracks its original function with low error in bounded regimes. … We also note that here, **the BERT query-key dot products are bounded in regimes where the second-order Taylor series `exp` approximation maintains monotonicity (Fig. 5).**"

**The 2nd-order Taylor map's advantage is conditional on the donor's `qᵀk` distribution staying inside the regime where `1 + z + z²/2` is monotone in `z`.** `1 + z + z²/2` has derivative `1 + z`, so it is monotone increasing only for `z ≥ −1` and is **decreasing for `z < −1`** — a query-key pair with dot product below `−1` gets *more* weight the more dissimilar it becomes. Hedgehog verified this empirically for BERT; **no source read verifies it for any modern decoder-only LLM, and nobody reports the `qᵀk` range for Qwen-family donors. [X]** This is the one precondition that a donor-adaptation project can check cheaply and that the literature leaves open.

### 3.6 Is there a stated relationship between expansion order and representable structure?

Yes, and it has three parts, all sourced:

1. **Order → dimension.** `p`-degree ⇒ `φ: R^d → R^{O(d^p)}`, cost `O(n d^p)`. Hedgehog §4.1 [A], Based §4.1 [A], effective dims printed in Based Table 6 [T].
2. **Dimension → recall capacity.** Recall capacity of any causal fixed-state model is bounded by state size (Based Theorem 3.1 [A]); orthogonality/crosstalk bound at `d_dot` associations (Schlag §4.1 [A]); MQAR needs dimension growing with `N` for data-independent mixers (Zoology Thm 4.4 [A]).
3. **Order → the two softmax-like properties, within a bounded regime.** 2nd order recovers spikiness and monotonicity where `qᵀk` is bounded; order 1 (elementwise positive maps) does not recover either. Hedgehog Table 2 [T] and §4.1 [A].

The DeltaNet-lineage related-work section states the same connection from the Hopfield side (arXiv:2406.06484 §6 [A]): *"vanilla linear transformers use a Hebbian-like update rule, which has been shown to have limited memory capacity. Later works in Hopfield networks use higher-order polynomials and exponential kernels to enhance the memory capacity, which is also related to linear attention with polynomial kernels."*

**Anyone disputing it?** Based itself is the closest thing to a dissent: it places `elu+1` and `ReLU` *on the same Pareto frontier* as Taylor and says *"other simple feature maps may be effective as well"* §4.1 [A]. Gated DeltaNet's Table S.1 [T] and DeltaNet's Table 1 [T] are the strongest empirical dissent: at fixed dimension and fixed norm, the feature-map choice moves perplexity by 0.3–0.7 while the *normalisation* choice moves it by ~3. **No source read reports a paper arguing that a higher-order kernel is worse; the dissent is that the kernel is nearly irrelevant once dimension and normalisation are fixed.**

---

## 4. Normalisation — load-bearing, and entangled with the delta rule

Consolidating what §1.3b and §2.2 established, because this is the one ingredient on which the literature is unanimous and quantitative.

**The mechanism (arXiv:2406.06484 §3.3 [A]).** The eigenvalues of the delta-rule transition `I − β_t k_t k_tᵀ` are `1` with multiplicity `d−1` and `1 − β_t‖k_t‖²` with multiplicity 1. `L2`-normalising `k_t` puts that eigenvalue in `[0,1]` for `β_t ∈ [0,1]`, and at `β_t = 1` makes the transition an **exact orthogonal projector** that deletes one direction and preserves `d−1`. Schlag's original `L1` normalisation also bounds the eigenvalue but yields no such clean geometry.

**Therefore normalisation is not an independent ingredient from the delta rule — it sets the spectrum of the delta rule's transition operator.** An ablation that removes `L2` normalisation is not testing "normalisation"; it is testing whether the erase step is a projection or an arbitrary contraction. Any experiment design that treats "normalisation" and "delta rule" as two orthogonal factors is mis-specified on this evidence.

**Effect sizes, two independent replications, both [T]:**

| Source | Setup | `L1` rows | `L2` rows | Norm effect | Feature-map spread within `L2` |
|---|---|---|---|---|---|
| arXiv:2406.06484 Table 1 (bottom) | DeltaNet, 340M/15B, SlimPajama, Mistral tok. | `1+ELU` 31.12 Wiki ppl | `1+ELU` 28.03 | **−3.09 ppl** | SiLU 28.24 / `1+ELU` 28.03 / ReLU 28.75 → **0.72 ppl** |
| arXiv:2412.06464 Table S.1 | Gated DeltaNet, 400M/15B, FineWeb-Edu, Llama2 tok. | 30.18–30.79 Avg-PPL | 27.35–27.67 | **≈ −3.0 ppl** | SiLU 27.35 / `1+ELU` 27.58 / ReLU 27.67 → **0.32 ppl** |

**Ratio: normalisation moves perplexity 4–9× more than the entire feature-map choice, in both papers.** The Gated DeltaNet authors state it plainly, Appendix B.2 [A]: *"Consistent with Yang et al., we found L2 normalization to be essential for optimal performance, though the choice of feature map was less influential."*

**But "normalisation" is at least two knobs with opposite signs.** Schlag Table 3 [T] (§2.1) separates them: the *sum normalisation* (the linear-attention denominator) is mandatory — *"the models diverged otherwise"* [A] — while an *extra attention normalisation on top* costs 1.1 valid ppl when present (29.2 → 28.1 without it). Anyone reporting "we ablated normalisation" without saying which of the two is reporting an uninterpretable number.

**Per-head `q`/`k` normalisation specifically: [X] as an isolated ablation.** No source read varies *per-head vs per-tensor* normalisation. What is ablated everywhere is the *norm family* (`L1` vs `L2`) applied per query/key vector. The Gated DeltaNet block applies `L2` to `q,k` per head by construction (Fig. 1 caption [A]) and never tests the alternative. GDN-2's Appendix D.2 is titled *"Query and key normalization"* but is a **numerical-precision** appendix, not an ablation (§D "Numerical details and verification" [A]).

**In the conversion regime, the answer flips in emphasis.** arXiv:2510.05901 Table 4 [T] (§7.2 below) shows that in a converted donor the *activation shape* of the feature map is worth ~12 average points while everything else is noise — the opposite ranking to the from-scratch papers. **This is a from-scratch/conversion dissociation on the normalisation-vs-feature-map question, and it is real.**

---

## 5. The 2026 successors — what the lineage decided next

Literature covered for this section: arXiv listings and full-text search across `cs.LG`/`cs.CL` linear-attention/linear-RNN papers 2025-01 → 2026-08, via keyword search on DeltaNet, delta rule, gated linear attention, linearisation, and their successors. **Absence statements below are scoped to that fence.**

### 5.1 Gated DeltaNet-2 (NVIDIA, May 2026) — the erase/write decoupling

**Source.** Ali Hatamizadeh, Yejin Choi, Jan Kautz, *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*, **arXiv:2605.22791v1, 21 May 2026**. Read at `arxiv.org/html/2605.22791`.

**The diagnosis, from the abstract [A]:** *"the active edit still uses a single scalar gate to control two different things, how much old content to erase on the key side and how much new content to commit on the value side."* Gated DeltaNet-2 splits `β_t` into a **channel-wise erase gate `b_t`** (key side) and a **channel-wise write gate `w_t`** (value side), on top of KDA's channel-wise decay `D_t`. It *"reduc[es] to KDA when both gates collapse to the same scalar and to Gated DeltaNet when the decay also collapses."*

**Table 1 [T]**, *"Fast-weight update view of DeltaNet, Mamba-2, Gated DeltaNet, KDA, Mamba-3, and Gated DeltaNet-2"* — state orientation `o_t = S_tᵀ q_t`. Transcribed (update column):

| Method | State update |
|---|---|
| DeltaNet | `S_t = (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ` |
| Mamba-2 | `S_t = α_t S_{t−1} + k_t v_tᵀ` |
| Gated DeltaNet | `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ` |
| KDA | `S_t = (I − β_t k_t k_tᵀ) D_t S_{t−1} + β_t k_t v_tᵀ` |
| Mamba-3 | `S_t = α_t S_{t−1} + η_t k̃_{t−1} v_{t−1}ᵀ + ζ_t k̃_t v_tᵀ` |
| Gated DeltaNet-2 | (channel-wise `b_t`, `w_t`; `D_t` decay — see §3/Eq. 10 of that paper) |

The caption's own summary [A]: *"Mamba-2 and Mamba-3 add gated key-value correlation terms to a decayed state. DeltaNet, Gated DeltaNet, KDA, and Gated DeltaNet-2 instead write a delta residual, the target value minus the value currently read from memory."* **`D_t` (KDA, GDN-2) is a channel-wise/diagonal decay replacing GDN's scalar `α_t`** — that is the 2025→2026 move on the decay ingredient.

**Table 2 [T]**, recurrent models, **1.3B / 100B FineWeb-Edu** — same protocol as arXiv:2412.06464 §4 (AdamW, grad clip 1.0, cosine, 1B warm-up, batch 0.5M, training length 4K, hybrids use 2K SWA) [A]. Columns: Wiki ppl↓ | LMB ppl↓ | LMB acc | PIQA | Hella. | Wino. | ARC-e | ARC-c | OBQA | SIQA | BoolQ | Avg.

| Model | Wiki ppl | LMB ppl | LMB | PIQA | Hella | Wino | ARC-e | ARC-c | OBQA | SIQA | BoolQ | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mamba-2 | 16.79 | 12.38 | 45.24 | 72.58 | 55.51 | 55.33 | 70.68 | 35.26 | 31.00 | 40.63 | 60.19 | 51.82 |
| Gated DeltaNet | 16.40 | 11.89 | 49.62 | 72.31 | 56.50 | 56.75 | 68.81 | 35.15 | 30.20 | 40.53 | 58.78 | 52.07 |
| KDA | 16.81 | 11.68 | 48.13 | 72.09 | 55.75 | 55.72 | 70.83 | 35.92 | 30.40 | 40.99 | 60.67 | 52.28 |
| Mamba-3 (SISO) | 16.30 | 12.99 | 45.06 | 72.31 | 55.58 | 56.20 | 70.45 | 34.56 | 31.00 | 41.76 | 55.90 | 51.42 |
| Mamba-3 (MIMO) | 16.45 | 11.66 | 47.82 | 72.36 | 56.49 | 55.78 | 72.38 | 38.07 | 30.00 | 40.89 | 57.74 | 52.39 |
| Gated DeltaNet-2 | 15.90 | 11.41 | 48.09 | 72.80 | 56.84 | 57.85 | 72.43 | 38.23 | 31.60 | 40.58 | 59.54 | 53.11 |
| *Transformer (hybrid block)* | 19.22 | 13.72 | 48.32 | 70.21 | 56.12 | 55.85 | 69.23 | 33.84 | 25.00 | 39.74 | 59.42 | 50.86 |

**Three things in these cells that cut against the delta rule as the operative ingredient:**
1. **Mamba-2 (decay only, no delta rule) 51.82 vs Gated DeltaNet (both) 52.07 — a gap of 0.25 Avg.** Derived. In the original GDN paper the same pair was 54.89 vs 55.32, a gap of 0.43 (§1.4). **Two independent runs, both single-seed, both put the delta rule's contribution to commonsense average under half a point.**
2. **Mamba-3 (MIMO) — a pure-decay SSM with no delta rule at all — scores 52.39, above Gated DeltaNet's 52.07.** Derived. A better-engineered decay model beats the gated-delta model on this average.
3. **KDA (channel-wise decay + scalar delta) 52.28 > Gated DeltaNet (scalar decay + scalar delta) 52.07.** Derived. Refining the *decay* to be channel-wise buys more than the delta rule's whole contribution in the row above.

**Table 5 [T]** — *"Gate structure and erase range ablations in the recurrent-only setting."* Columns: Wiki ppl↓ | LMB ppl↓ | Common. avg↑ | S-NIAH-2 @4K↑ | S-NIAH-3 @2K↑ | MK-NIAH-1 @4K↑ | Recall avg↑

| Variant | Wiki | LMB | Common. | S-NIAH-2 @4K | S-NIAH-3 @2K | MK-NIAH-1 @4K | Recall |
|---|---|---|---|---|---|---|---|
| *Channel structure* | | | | | | | |
| w-only: scalar `b_t`, channel `w_t` | 16.55 | 11.62 | 52.45 | 90.6 | 71.4 | 30.6 | 28.92 |
| b-only: channel `b_t`, scalar `w_t` | 16.12 | 11.50 | 52.79 | 92.1 | 84.6 | 35.2 | 29.51 |
| *Erase range* | | | | | | | |
| Gated DeltaNet-2, `b_t ∈ [0,1]^{d_k}` | 15.90 | 11.41 | 53.11 | 93.0 | 89.8 | 37.8 | 29.88 |
| expanded `b_t ∈ [0,2]^{d_k}` | 15.95 | 11.44 | 53.04 | 93.1 | 89.4 | 37.6 | 29.81 |

**This is the cleanest erase-side/write-side separation in the literature.** The ablation is parameter-matched by construction: §4 [A] *"we average either gate over its channel axis and broadcast the scalar back at runtime, while keeping the original projections unchanged. Thus the parameter count stays fixed and only channel-wise gate variation is removed."*

Derived: keeping channel structure only on the **erase** side recovers 52.79/29.51 of the full 53.11/29.88; only on the **write** side recovers 52.45/28.92. On S-NIAH-3 @2K the split is stark: **84.6 (erase) vs 71.4 (write)** against a full-model 89.8. The paper's conclusion [A]: *"Ablations show that both gates contribute, with the erase gate accounting for most of the gain."*

**And a null the project should have.** Expanding the erase range to `[0,2]` — the negative-eigenvalue regime that Grazzi et al. 2024 / Siems et al. 2025 argue unlocks state tracking, and that arXiv:2412.06464 flags in footnote 2 — **"gives no consistent gain at this scale"** [A]: 15.95 vs 15.90 Wiki ppl, 53.04 vs 53.11 Common. avg [T]. A published, matched null against a well-cited theoretical claim.

**MMLU in GDN-2: [X].** Appendix E.2 [A] lists the evaluation suite: WikiText, LAMBADA ppl + acc, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA, SIQA, BoolQ. No MMLU, at any scale, in any table. **The entire Gated DeltaNet line — 2024 original, 2026 successor — reports no MMLU.**

### 5.2 Other 2026 entries found in the same sweep

- **arXiv:2607.07386, *Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity***. Slot-sparse delta memory; GDN and Mamba-2 baselines at 1.4B and 8B; RULER evaluation. Source of the only learned-gate measurement I found (§6).
- **arXiv:2607.07953, *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing*** (8 Jul 2026). Routes a DeltaNet layer's *write error* or *write value* across depth (CLER / CLER-H / CLVR). §1 [A]: *"is there a lightweight way to share information across depth that respects the linear-time structure of DeltaNet-style memories?"* It uses vector-valued forget gates for KDA and Gated DeltaNet-2 (§ setup [A], line 220). Relevant as evidence that the 2026 frontier has moved to *cross-layer* structure, not to better single-layer kernels.
- **arXiv:2607.02303, *A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets***. Not read in depth; flagged as an adjacent 2026 direction (exact side-memory for evicted content).
- **arXiv:2605.05838, *MDN: Parallelizing Stepwise Momentum for Delta Linear Attention***. Not read in depth; adds momentum to the delta update.

**Direction of travel, stated plainly:** the 2025–2026 successors do not change the *feature map* at all. Every one of them keeps `L2`-normalised `q`/`k` with a SiLU-family activation and spends its novelty budget on the **transition operator** — channel-wise decay (KDA), decoupled erase/write (GDN-2), momentum (MDN), sparse slots (SDM), cross-layer routing (CLVR). **Nobody in the 2026 window is arguing about kernels.** That absence is scoped to the fence stated at the head of this section.

---

## 6. What the decay gate actually learns — the coordinator's question

### 6.1 Gated DeltaNet does not report it

**arXiv:2412.06464 §3.4 footnote 4 [A], quoted in full:** *"We use Mamba2's parameterization for `α` but omit it for brevity."* The Fig. 1 caption [A] adds only: *"alpha/beta use linear proj."*

**The Gated DeltaNet paper states neither its own `α` parameterisation nor any learned value, distribution, effective time constant, or per-layer profile. [X]** Nor does the GDN-2 paper. Nor does arXiv:2406.06484 (which has no gate). **[X] across the whole Gated DeltaNet line.**

### 6.2 The one published measurement I found

**Source.** *Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity*, **arXiv:2607.07386**, Appendix B, Figure 12 caption. This is a **figure caption, not a table** — tagged **[A-fig]** and flagged as such throughout.

Setup, §Experimental setup [A]: SDM at **1.4B scale (run "L08")**, statistics accumulated over 10 training steps (~21M tokens) per data point, **averaged across the model's 5 SDM layers**. Forget-gate parameterisation stated explicitly [A]: *"The forget gate parameter `A` is initialized uniformly in `[0,16]` and the time-step bias `b_dt` is initialized from `inv_softplus(U(0.001, 0.1))`, **both matching GDN/FLA conventions**."*

Figure 12 caption, verbatim [A-fig]:

> "The forget gate (top left) **converges early to `exp(g) ≈ 0.95`**. The **minimum forget gate stays near zero**, indicating that some slots are fully decayed when overwritten. The input gate (top right) stabilizes at `σ(b) ≈ 0.50`, with the maximum and minimum reaching ~1 and ~0 respectively. Overall, **the model learns a wide dynamic range for both forget and input strength**, showing the importance of both mechanisms in the model."

**What this licenses, and what it does not.**
- **Licensed:** in a delta-rule linear RNN at 1.4B using GDN/FLA gate conventions, the *mean* learned decay is ≈ 0.95 per token, the *minimum* is ≈ 0, and the range is wide. **[A-fig]**
- Derived from that single value only: `γ = 0.95` has a **half-life of `ln(0.5)/ln(0.95) ≈ 13.5 tokens`** and a **mean residence time `1/(1−γ) = 20 tokens`**. Arithmetic on one caption number; not a paper's claim.
- **Not licensed:** that Gated DeltaNet proper learns 0.95. SDM is a *slot-sparse* memory, not GDN; only its gate *parameterisation* is shared. **The transfer is an inference, not a measurement.**
- **Not licensed:** a per-layer profile. The number is averaged over 5 layers, and the caption's own "wide dynamic range" plus "minimum near zero" says the mean is a poor summary.

**Initialisation, which IS specified and IS checkable.** With Mamba/GDN conventions `α_t = exp(−Δ_t · A)`, `A ~ U[0,16]`, `Δ ~ U(0.001, 0.1)` at init: `Δ·A ∈ [0, 1.6]`, so **`α` at initialisation spans `exp(−1.6) = 0.202` to `1.0`** (derived from the two [A] init ranges). The prior the optimiser starts from therefore already covers the full range of the decay sweep, from very short to effectively uniform.

### 6.3 Depth-dependence of the learned decay — one paper, and it is not GDN

**Source.** Zhen Qin et al., *Hierarchically Gated Recurrent Neural Network for Sequence Modeling*, arXiv:2311.04823 (NeurIPS 2023). Read via ar5iv.

HGRN's entire thesis is that the decay should be depth-dependent, and it *parameterises* it that way. §Abstract [A]: *"forget gates that are lower bounded by a learnable value. **The lower bound increases monotonically when moving up layers.** This allows the upper layers to model long-term dependencies and the lower layers to model more local, short-term dependencies."* §3 [A]: the lower bounds are a learnable `Γ ∈ R^{H×d}` passed through a `cummax` so monotonicity in depth is structural, with a cap *"to prevent the highest layer's lower bound from being one as we still want the ability to forget irrelevant information."*

**HGRN Figure 2 [A-fig]:** *"Visualization of forget rates. We plot the forget rates of layers 5 and 6 on a model trained on language modeling tasks."* — with and without the lower bound. **The figure exists; the numeric values are in the plot and I did not read them from a cell. [X] for values.**

**HGRN Table 8 [T]** *"Forget gate ablation on an autoregressive language model. The only lower bound means using a data-independent gate like LRU."* and **Table 10 [T]** *"Lower bound ablation… A random lower bound means the lower bound in each layer is independent. Decrease lower bound means the lower bound is monotonically decreasing with respect to layer `k`."* — I located both captions but **did not transcribe their cells**; recorded as a known unread source rather than summarised. The paper's text conclusion [A]: *"removing the forget gate significantly decreases the performance of HGRN, while adding a forget gate to LRU improves performance. On the other hand, using a data-independent forget gate (only lower bound) leads to lower performance compared to a data-dependent forget gate."* And: *"The most significant improvement arises from the monotonically increasing lower bound."*

**Bearing on a layer-indexed measurement.** HGRN is the only source I found that says the *correct* decay is a function of depth, short at the bottom and long at the top, and it builds that into the architecture rather than leaving it to the optimiser. **Any single scalar `γ` compared across layers 14 / 21 / 27 of a 28-layer donor is, on HGRN's account, being compared against three different correct answers.** Whether that carries to a Qwen-family softmax donor is **[X]** — HGRN is a from-scratch RNN, not a converted transformer.

### 6.4 How much does the content kernel add once an envelope is present?

This is the coordinator's second new question, and it is the sharpest one. **I did not find a paper that runs exactly this ablation. [X]** What exists, closest first:

**(a) The reverse ablation, at fixed content kernel — Based Table 6 [T].** Two rows differ only in the decay flag, everything else identical (Taylor 2nd order, `d'=16`, effective 153, SWA 64, convs on), 362M / 10B Pile:

| Decay | All Ppl | AR Ppl | Other Ppl | SWDE | FDA | SQUAD |
|---|---|---|---|---|---|---|
| ✓ | 8.65 | 2.07 | 9.64 | 29.16 | 11.71 | 25.07 |
| ✗ | 8.65 | 2.04 | 9.66 | 22.95 | 12.34 | 27.45 |

**Perplexity is identical to two decimals (8.65 / 8.65), and the AR perplexity is marginally *better without* decay (2.04 vs 2.07).** Recall splits: SWDE −6.21 with decay removed, FDA +0.63, SQUAD +2.38 (derived). Based's own reading, Appendix C [A]: *"We observe that decay can help small in our Table 6 ablations, but **removing the decay does not affect the overall trends** for Based relative to other architectures"*; and Appendix E.1 [A]: *"We use **no input-dependent decay whatsoever** when training the models to 30b and 50b tokens at 360m and 1.3b parameters respectively."* **A published architecture whose headline models carry no decay at all.**

**(b) Envelope-only versus envelope-plus-content, in the theory rather than a measurement.** Zoology §1 [A]: *"gated convolutions … require model dimension that grows with sequence length (Theorem 4.4) while attention can solve MQAR with model dimension independent of sequence length (Proposition 4.3)"*, and *"input-dependent sequence mixing is important to solve MQAR efficiently (Theorem 4.5)."* A pure decay envelope `γ^{t−s}` with no content term is exactly a data-independent (indeed input-*constant*) mixing, so Zoology's theorems say it cannot solve MQAR at fixed dimension. **But that is a statement about a task with a right answer, not about how much of a donor's attention output a solve can reach.**

**(c) The nearest empirical analogue is in the conversion literature, and it is §7.2 below** — arXiv:2510.05901 Table 4 [T], where a converted donor's linear branch with an elementwise `1+ELU` content kernel scores 35.12 average against 34.40 for **no attention at all**, while an exponential-family kernel scores 46.94. That is a content-kernel contribution measured against a genuine zero baseline — but with no decay envelope in the arm.

**Stated as an absence, with the fence.** Across the linearisation literature (Hedgehog, LoLCATs, Liger, SUPRA, T2R, the component-imbalance paper), the DeltaNet lineage (Schlag, DeltaNet, Gated DeltaNet, KDA, GDN-2), the Based/Zoology line, and the 2026 successors listed in §5.2: **no paper reports the incremental contribution of the content kernel `q·k` given a fixed decay envelope, nor an "envelope only" arm as a control.** The control that would make our number interpretable is not in this literature. It may exist in the state-space-model interpretability literature, which I did **not** cover.

---

## 7. Counter-evidence, and the from-scratch / converted-donor split

Literatures covered: (i) the DeltaNet lineage and its 2025–2026 successors; (ii) the Based / Zoology recall-throughput line; (iii) the linearisation / conversion line (T2R, SUPRA, Hedgehog, LoLCATs, Liger, MOHAWK-adjacent work as cited by those papers, and the 2025 component-imbalance audit); (iv) HGRN/GLA gated-RNN work as reached through the above. **Not covered:** SSM interpretability, mechanistic-interpretability work on induction heads, and the RWKV line beyond its appearance as a baseline. Absences below are scoped to (i)–(iv).

### 7.1 Decay does not always help — five instances, all in the authors' own tables

1. **Gated DeltaNet's own §3.2 heading is "Decay hurts memory retention."** [A] On S-NIAH-1 at 4K/8K, DeltaNet (no decay) scores 99.0 / 98.8 against Gated DeltaNet's 91.4 / 91.8 and Mamba-2's 65.4 / 30.4 [T]. **Adding decay to the delta rule costs 7.6 and 7.0 points on a pure-retention task.**
2. **Based ships with no decay.** Appendix E.1 [A]: *"We use no input-dependent decay whatsoever when training the models to 30b and 50b tokens."* Its decay ablation (§6.4a) shows identical perplexity with and without (8.65 / 8.65) [T].
3. **Liger Table 6 [T]** — Llama-3-8B linearised into GLA, validation PPL on cleaned-alpaca, plus two averages:

| Liger-GLA (Llama-3-8B) | PPL ↓ | Avg. ↑ | Avg. (no MMLU) ↑ |
|---|---|---|---|
| Liger-GLA | 2.96 | 67.6 | 72.4 |
| − Gate Proj. | 3.16 | 63.8 | 68.8 |
| − Feat. Map. | 9.04 | 43.5 | 40.2 |
| − Pure LA | 3.00 | 66.1 | 71.5 |
| − w/o LoRA | 3.23 | 61.7 | 68.1 |
| − w/o SWA | 3.75 | 54.2 | 60.2 |
| − w/o GLA | 3.01 | 66.2 | 72.0 |

*(Caption [A]: "We linearize Llama-3-8B into Gated Linear Attention (GLA) to evaluate the key components of Liger. We report Validation perplexity (PPL.) on cleaned alpaca dataset after Liger linearization and the average performance on language modeling and understanding tasks.")*

**"Pure LA" = the gate removed entirely.** Derived: **+0.04 PPL, −1.5 Avg, −0.9 Avg-no-MMLU.** Compare the same removal *from scratch* — Gated DeltaNet Table S.1, `w. naive Delta Rule`: **+3.52 PPL, −2.14 Avg**. **In relative terms the gate is worth 0.04/2.96 = 1.4% of perplexity in the converted donor against 3.52/27.35 = 12.9% in the from-scratch model — a factor of ~9 (derived; the two perplexity scales differ, so only the relative comparison is meaningful, and even that crosses datasets — cleaned-alpaca vs FineWeb-Edu — so it is indicative, not a controlled contrast).** That is the from-scratch/conversion dissociation on ingredient (1), measured, in two papers' own cells.
**And `w/o GLA` — deleting the entire gated-linear-attention branch — costs 0.05 PPL and 0.4 Avg-no-MMLU**, i.e. essentially nothing, while `w/o SWA` costs 0.79 PPL and 12.2 Avg-no-MMLU (derived). **In this converted model the linear branch is nearly free to delete and the sliding window is doing the work.**
Note also `− Feat. Map.` = 9.04 PPL / 43.5 Avg: **adding a learnable Hedgehog-style feature map to Liger destroys it.** The feature map is not a monotone good.

4. **GDN-2's erase-range null.** Expanding `b_t` to `[0,2]` — the negative-eigenvalue regime that the state-tracking literature (Grazzi et al. 2024; Siems et al. 2025) motivates — *"gives no consistent gain at this scale"* [A]; 15.95 vs 15.90 Wiki ppl [T].
5. **Mamba-3 (MIMO), a pure-decay model with no delta rule, beats Gated DeltaNet on commonsense average** in GDN-2's Table 2: 52.39 vs 52.07 [T]. The converse of the usual claim, in the successor paper's own table.

### 7.2 The converted-donor regime is a different regime, and the audit that proved it

**Source.** *Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods*, **arXiv:2510.05901**. Read at `arxiv.org/html/2510.05901`.

**The finding, abstract [A]:** *"We identify a critical flaw: existing hybrid methods inadvertently bypass the linear component, relying almost entirely on SWA. Component-level diagnostics reveal this previously undetected behaviour stems from overlooked evaluation practices on common-sense benchmarks."*

Method, §3.2 [A]: four inference-time ablations on already-converted models — (i) SWA-only, LA output forced to zeros; (ii) LA-only, SWA disabled; (iii) attention-sinks only, first 8 values through softmax; (iv) **no attention, an all-zeros attention output**. Note (iv): **this is a planted zero control of exactly the kind this project requires, and it is the paper's own.**

**Table 1 [T]**, *"Measuring the effect of attention components on benchmark accuracy for models trained with LoLCATs hybrid conversion."* Columns: Active Attn Modules | PIQA | ARC-E | ARC-C | HellaSwag | WG | **MMLU** | AVG | Rec. Perf

| Active Attn Modules | PIQA | ARC-E | ARC-C | HellaSwag | WG | MMLU | AVG | Rec. Perf |
|---|---|---|---|---|---|---|---|---|
| **Mistral-7B** | 79.27 | 80.01 | 52.22 | 74.60 | 69.93 | 53.51 | 68.26 | 100.00 |
| SWA + Linear | 78.40 | 79.50 | 49.91 | 71.28 | 68.35 | 45.44 | 65.48 | 95.93 |
| SWA only | 78.56 | 79.46 | 49.83 | 70.57 | 68.35 | 46.56 | 65.56 | 96.04 |
| Linear only | 53.65 | 29.71 | 24.57 | 27.01 | 50.43 | 23.28 | 34.78 | 50.95 |
| Attn sinks only | 54.41 | 30.05 | 24.66 | 28.80 | 49.64 | 23.67 | 35.21 | 51.58 |
| **No Attention** | 53.92 | 25.63 | 24.66 | 25.99 | 50.67 | 25.51 | 34.40 | 50.39 |
| **Llama3-8B** | 78.13 | 81.69 | 56.66 | 75.94 | 71.67 | 63.85 | 71.32 | 100.00 |
| SWA + Linear | 78.35 | 80.47 | 54.35 | 71.62 | 71.67 | 48.84 | 67.55 | 94.71 |
| SWA only | 78.02 | 80.01 | 54.10 | 65.08 | 71.98 | 47.30 | 66.08 | 92.26 |
| Linear only | 52.07 | 25.97 | 25.77 | 26.31 | 48.22 | 22.95 | 33.55 | 47.04 |
| Attn sinks only | 56.31 | 29.00 | 22.61 | 28.83 | 49.41 | 22.95 | 34.85 | 48.86 |
| **No Attention** | 55.17 | 26.73 | 22.87 | 26.13 | 51.14 | 22.95 | 34.17 | 47.91 |
| **Llama3.1-8B (LoLCATs ckpt)** | 80.14 | 81.82 | 55.20 | 79.14 | 73.72 | 68.05 | 73.01 | 100.00 |
| SWA + Linear | 81.18 | 82.37 | 54.78 | 79.16 | 70.09 | 58.89 | 71.08 | 97 |
| SWA only | 81.56 | 82.37 | 55.29 | 79.76 | 74.11 | 55.63 | 71.45 | 98 |
| Linear only | 51.52 | 25.00 | 25.51 | 26.37 | 52.25 | 23.09 | 33.96 | 47 |
| Attn sinks only | – | – | – | – | – | – | – | – |
| **No Attention** | 54.62 | 26.68 | 24.40 | 25.88 | 48.86 | 23.12 | 33.93 | 46 |

**Read the Linear-only rows against the No-Attention rows.** Mistral: **34.78 vs 34.40**. Llama3-8B: **33.55 vs 34.17 — the linear branch is *worse* than deleting attention entirely.** Llama3.1-8B LoLCATs checkpoint: **33.96 vs 33.93.** Derived. The paper's own wording, §3.2 [A]: *"LA-only, attention sinks only, and no attention **all collapse to roughly the same low performance**. The results expose a clear SWA-LA imbalance … that leads to LA contributing little to downstream accuracy **and even being detrimental**."*
And SWA-only ≥ SWA+Linear on AVG for two of three models (Mistral 65.56 vs 65.48; Llama3.1 71.45 vs 71.08) [T] — **removing the linear branch at inference improves the converted model.**
Corroboration named by that paper [A]: *"Similar findings are reported in Lan et al. (2025) (see their Table 6) where they find that SWA-only and GLA+SWA lead to very similar performance while GLA-only leads to a dramatic decrease in performance."* — that is Liger, and my §7.1 transcription of Liger Table 6 confirms it independently.

**The MMLU column here is the sharpest instance of the project's MMLU rule I have ever transcribed.** Llama3.1-8B: base MMLU **68.05** → SWA+Linear **58.89** → Linear only **23.09**. Four-way MMLU chance is 25.0, so *every* Linear-only row (23.28 / 22.95 / 23.09) is **at or below chance**. Derived: the AVG falls 73.01 → 71.08 (**−1.93**) while MMLU falls 68.05 → 58.89 (**−9.16**) — **the macro-average understates the MMLU damage by 4.7×**. For Llama3-8B: AVG −3.77 while MMLU −15.01, a factor of 4.0. **A conversion that reports only an average is not reporting retention.**

**Feature map in the conversion regime — Table 4 [T]**, *"Ablating the activation function used in HedgeHog projections (φ). All results show checkpoints for a single epoch of weights transfer with no fine-tuning, evaluated as LA-only."*

| Φ Activation Fn | PIQA | ARC-E | ARC-C | HellaSwag | WG | MMLU | AVG | Rec. Perf |
|---|---|---|---|---|---|---|---|---|
| **Mistral-7B** | 79.27 | 80.01 | 52.22 | 74.60 | 69.93 | 53.51 | 68.26 | 100.00 |
| Softmax | 67.90 | 61.74 | 29.69 | 45.82 | 52.64 | 23.82 | 46.94 | 68.76 |
| Exponential | 66.32 | 61.45 | 29.69 | 44.67 | 50.99 | 23.52 | 46.11 | 67.55 |
| ReLU | 56.53 | 34.81 | 22.95 | 27.95 | 50.91 | 23.29 | 36.07 | 52.85 |
| 1 + ELU | 54.68 | 29.12 | 23.81 | 25.98 | 51.46 | 25.64 | 35.12 | 51.45 |
| None | 51.74 | 25.51 | 29.61 | 25.97 | 51.07 | 24.95 | 34.81 | 51.00 |
| **No Attention** | 53.92 | 25.63 | 24.66 | 25.99 | 50.67 | 25.51 | 34.40 | 50.39 |

**This is the closest published analogue to the R2a main table, and it has the same shape.** Derived margins over the zero-mixing control: exponential-family kernels **+12.54 / +11.71 AVG**; ReLU **+1.67**; `1+ELU` **+0.72**; no activation at all **+0.41**. **`1+ELU` sits 0.72 points above having no attention whatsoever, and 0.31 above having no activation function.** The authors flag it [A]: *"Interestingly, 1+ELU does not appear to carry this same performance despite its close similarity to the exponential function by itself."*
Note also that **every LA-only row has MMLU at or below chance**, including the best one (softmax φ, AVG 46.94, MMLU 23.82). **A converted linear-only model can recover 69% of the commonsense average and 0% of MMLU.**

**Feature-map dimensionality in conversion — Tables 2 and 3 [T].** Table 2 uses Hedgehog's square `W_φ ∈ R^{h_d×h_d}` (so `φ: R^{h_d} → R^{2h_d}`); Table 3 repeats it with LoLCATs' `W_φ ∈ R^{h_d×h_d/2}` (`φ: R^{h_d} → R^{h_d}`), i.e. half the output features. Mistral-7B, LA-only:

| Transfer Objective | Table 2 AVG (`2h_d`) | Table 3 AVG (`h_d`) | Δ |
|---|---|---|---|
| Attention Weights | 46.94 | 43.25 | −3.69 |
| Attention Outputs | 47.23 | 42.60 | −4.63 |
| Hybrid Attention Out. | 35.41 | 34.59 | −0.82 |
| No Attention | 34.40 | 34.40 | 0 |

(Full Table 2 rows [T]: Attention Weights 67.90/61.74/29.69/45.82/52.64/23.82/**46.94**/68.76; Attention Outputs 69.64/62.46/31.40/46.61/50.28/23.00/**47.23**/69.20; Hybrid Attention Out. 55.44/32.83/24.06/27.36/49.80/22.98/**35.41**/51.88. Full Table 3 rows [T]: Attention Weights 64.85/54.84/26.11/39.48/51.07/23.17/**43.25**/63.37; Attention Outputs 63.71/53.83/27.30/38.50/49.33/22.90/**42.60**/62.40; Hybrid Attention Out. 53.97/30.81/22.44/27.42/49.88/22.99/**34.59**/50.67.)

**Halving the feature dimension costs 3.7–4.6 AVG in a converted donor** — the same dimension-dominates story as Based Table 6, now in the conversion regime. And the **objective** matters as much: LoLCATs' *hybrid* attention-output objective yields 35.41 (barely above the 34.40 zero control) while a *full* attention-output or attention-weight objective yields 47.23 / 46.94 [T]. §3.4.1 [A]: *"the hybrid attention output objective is likely to be responsible for LoLCATs failure to make use of LA, as it barely beats no attention at all."*

**One more caution the paper hands us** [A], §3.4.1: *"we have not observed any successful LA-only conversions of these models in the literature, as opposed to Mistral, suggesting that they may be particularly hard to convert. We leave the analysis as to why this might be to future work."* **Donor identity is a live variable in conversion, and the literature has no theory for it.** Every conversion result read here is on Mistral-7B or Llama-3/3.1-8B. **Qwen-family donors: [X] — no conversion result on a Qwen donor in any source read.**

### 7.3 Hedgehog's own conversion numbers, and how they are framed

**Table 10 [T]**, *"Pretrained-conversion for 125M GPT-2 on WT-103 lang. modeling."*

| Method | GPT-2 | GPT-2 FT | Hybrid H3 | Hyena | T2R-GPT-2 | HH-GPT-2 |
|---|---|---|---|---|---|---|
| PPL | 28.0 | 15.8 | 18.5 | 18.5 | 19.4 | 16.7 |

**Table 11 [T]**, *"Hedgehog Llama-2 conversion (ROUGE)."*

| Llama-2 | R1 / R2 / RL |
|---|---|
| Softmax (Zero-shot) | 19.3 / 6.8 / 14.9 |
| Softmax (LoRA) | 51.1 / 27.6 / 43.5 |
| T2R (LoRA) | 2.8 / 0.0 / 2.6 |
| Hedgehog (LoRA) | 47.4 / 23.4 / 39.1 |

**Framing caution.** Hedgehog's abstract [A] states: *"Hedgehog-Llama2 7B achieves 28.1 higher ROUGE-1 points over the base standard attention model, where prior linear attentions lead to 16.5 point drops."* Check the cells: 47.4 − 19.3 = 28.1 — **the +28.1 is measured against the *zero-shot* softmax baseline, not against the LoRA-finetuned softmax baseline in the row above it.** Against the like-for-like control (Softmax LoRA 51.1), Hedgehog conversion is **−3.7 R1 / −4.2 R2 / −4.4 RL** (derived). Both numbers are in the same four-row table; the abstract reports the flattering one. Not an error — but a headline that a reader who has not read the table will get backwards.
**T2R (LoRA) at 2.8 / 0.0 / 2.6 is a total conversion collapse** and is the strongest single datapoint that a linear map which works from scratch can fail catastrophically on a converted donor.

**MMLU in Hedgehog: [X].** Its evaluation is GLUE, WikiText-103 perplexity, LRA, ViT accuracy, and a Llama-2 ROUGE task. No MMLU, no knowledge benchmark, at any scale.

### 7.4 "Hedgehog-shaped" is not Hedgehog

Recorded because it bears directly on how an arm gets labelled. Hedgehog's central move, §4.2 [A]:

> "Our key insight is that rather than rely on **fixed functional form** that captures our spiky and monotonic properties, we can **learn** linear attention feature maps that do so. For each attention block, we propose feature maps as trainable single-layer MLPs … unlike prior work, we **explicitly train these feature maps such that the attention layers mimic the properties of softmax attention**."

`φ_mlp(x) = Φ(Wᵀx + b)` with `W ∈ R^{d×d'}`, `b ∈ R^{d'}` **learned**, `Φ = exp` to induce spikiness; trained *"via softmax attention weights as cross-entropy soft-labels"* [A]. **Every Hedgehog number in the literature — Table 2's 100.0/58.4, Table 10's 16.7, Table 11's 47.4, and all of arXiv:2510.05901's Table 4 — is post-distillation, per-layer, per-head.** A frozen closed-form function with a hedgehog-like shape and no fitted `W`, `b` is a different object, and Hedgehog's own thesis predicts it will not carry the donor's attention weights. arXiv:2510.05901 makes the same point structurally: its Table 4 rows are all *"checkpoints for a single epoch of weights transfer"* [A] — even the `1+ELU` row had a `W_φ` fitted, and still landed 0.72 above the zero control.

---

## 8. Where the literature's answer lands

Every table read here that contains both a pure-recurrent row and a hybrid row puts the hybrid ahead:

| Source | Best pure recurrent | With some exact attention | Δ |
|---|---|---|---|
| arXiv:2412.06464 Table 4 (real recall avg) | Gated DeltaNet 30.6 | GDN-H2 40.1 | +9.5 |
| arXiv:2406.06484 Table 1, 1.3B (FDA) | DeltaNet 17.2 | +Global Attn 29.8 | +12.6 |
| arXiv:2402.18668 Table 6 (SWDE) | no-SWA row 28.62 | SWA(128) row 32.13 | +3.5 |
| arXiv:2503.01496 Table 6 (Avg no-MMLU) | w/o SWA 60.2 | Liger-GLA 72.4 | +12.2 |
| arXiv:2510.05901 Table 1 (Llama3.1 AVG) | Linear only 33.96 | SWA only 71.45 | +37.5 |

**The mechanistic answer this literature actually supports is not a property of the mixing.** It is: recall capacity of a fixed-state causal mixer is bounded by state size (Based Thm 3.1; Schlag §4.1; Zoology Thm 4.4), so within that bound the choice among linear mixings moves things by fractions of a point on general LM and by tens of points only on synthetic recall — while **the presence of a small exact-attention window moves things by tens of points on everything.**

---

## 9. Ingredient × claim × measurement × separately ablated? × evidence grade

Grades: **A** = separated ablation at matched budget, corroborated by ≥2 independent papers. **B** = separated ablation at matched budget, one paper. **C** = cross-model comparison (architecture confounded), or a single paper's text claim backed by its own cells. **D** = asserted in text or figure caption only, no cell.
"Separately ablated?" means: was this ingredient varied with everything else held fixed, in the same model?

| Ingredient | Claimed to buy | Whose measurement | Ablated separately? | Grade |
|---|---|---|---|---|
| **Decay gate `α_t`** (scalar, data-dep.) | memory clearance / filtering when state is saturated; length extrapolation | GDN Table S.1 [T] `w. naive Delta Rule` +3.52 PPL, −2.14 Acc @400M/15B; GDN Table 2 [T] S-NIAH-2/3 | **Yes**, within-block, from scratch | **B** |
| **Decay gate, in a converted donor** | same | Liger Table 6 [T] `− Pure LA` +0.04 PPL, −1.5 Avg @Llama-3-8B | **Yes**, within-method | **B** |
| **Decay gate — counter-evidence** | *hurts* pure retention; ships absent in Based | GDN §3.2 [A] "Decay hurts memory retention", Table 2 [T] DeltaNet 99.0/98.8 vs GDN 91.4/91.8 @S-NIAH-1 4K/8K; Based Table 6 [T] decay ✓/✗ → 8.65/8.65 PPL; Based App. E.1 [A] no decay in headline models | **Yes**, both | **A** |
| **Channel-wise decay `D_t`** (KDA) over scalar `α_t` | finer memory control | GDN-2 Table 2 [T] KDA 52.28 vs GDN 52.07 Avg | **No** — cross-model | **C** |
| **Delta rule `β_t` (erase-then-write)** | exact in-context association; fuzzy/noisy recall | DeltaNet Fig. 3 [T] Fuzzy Recall 35.7 vs Mamba 6.7 / GLA 6.9; GDN Table 2 [T] S-NIAH-3 @4K 27.6 vs Mamba2 4.6 | **No** within-block; only cross-model (Mamba2 vs DeltaNet vs GDN) | **C** |
| **Delta rule — cost** | destroys durable storage | DeltaNet Fig. 3 [T] Memorize 52.8, worst in table (Mamba 89.5) | Same cross-model comparison | **C** |
| **Delta rule — counter-evidence** | loses to plain decay on real recall; advantage vanishes with scale | GDN Table 4 [T] DeltaNet 26.2 vs Mamba2 29.8, loses 6/6 tasks; DeltaNet §4.2 [A] + Table 1 [T] "at the 1.3B scale, DeltaNet underperforms GLA"; GDN-2 Table 2 [T] Mamba-3 MIMO 52.39 > GDN 52.07 | **Yes** (scale sweep is matched) | **A** |
| **Delta rule is kernel-independent** | improves any feature map | Schlag Table 2 [T]: −3.0/−2.8 on `ELU+1`; −2.9/−2.4 on Performer | **Yes** — crossed design | **B** |
| **Erase-side vs write-side channel structure** | erase carries most of the gain | GDN-2 Table 5 [T] b-only 52.79/29.51 vs w-only 52.45/28.92 vs full 53.11/29.88, parameter-matched | **Yes**, and parameter-matched by construction | **B** |
| **Negative eigenvalues (`β ∈ (0,2)`)** | state tracking | GDN-2 Table 5 [T] `[0,2]` 15.95 vs `[0,1]` 15.90 ppl — **null** | **Yes** | **B** (null) |
| **`L2` vs `L1` normalisation of `q`,`k`** | makes the erase an exact rank-1 orthogonal projector; ~3 PPL | DeltaNet Table 1 [T] 31.12→28.03 Wiki ppl; GDN Table S.1 [T] ~30.4→~27.5 | **Yes**, twice, two datasets/tokenizers | **A** |
| **Sum normalisation (the LA denominator)** | required for convergence | Schlag §6.1.2 [A] *"the models diverged otherwise"* | **Not runnable** — divergence | **D** |
| **Extra attention normalisation on top** | *harmful* | Schlag Table 3 [T] 29.2 (with) vs 28.1 (without) valid ppl | **Yes** | **B** |
| **Per-head vs per-tensor normalisation** | — | — | **[X] nowhere** | — |
| **Feature map shape, from scratch** | little to nothing at fixed norm | DeltaNet Table 1 [T] spread 0.72 Wiki ppl; GDN Table S.1 [T] spread 0.32 PPL | **Yes**, twice | **A** (near-null) |
| **Feature map shape, in a converted donor** | ~12 AVG points; exponential-family only | 2510.05901 Table 4 [T] softmax 46.94 / exp 46.11 / ReLU 36.07 / `1+ELU` 35.12 / none 34.81 / **no-attention 34.40** | **Yes**, with a zero control | **B** |
| **Kernel order → feature dimension** | order `p` ⇒ `d̃ = O(d^p)`; dimension buys recall | Based Table 6 [T] eff. dim 16→561 tracks SWDE 8.10→37.62 monotonically; Hedgehog §4.1 [A] `O(nd^p)` | **Yes** (dimension sweep within Taylor) | **A** |
| **Kernel order at *matched* dimension** | — | — | **[X] nowhere in any source read** | — |
| **Spikiness (low attention entropy)** | solves AR from scratch | Hedgehog Table 2 [T] Taylor 100.0 vs `1+ELU` 17.0; Fig. 4 [A-fig]; planted control `exp(x·t)`, `t=2` solves AR [A] | **Yes**, with a planted control — but **not dimension-matched** | **B** |
| **Monotonicity in `qᵀk`** | additionally required for *conversion* | Hedgehog Table 1 [T]: spiky-but-not-monotone `exp(t=2)` reaches 50.0 vs softmax 58.8 | **Yes**, same control | **B** |
| **Trained (distilled) feature map** | all Hedgehog results are post-distillation | Hedgehog §4.2 [A]; 2510.05901 Table 4 caption [A] | n/a — definitional | **D** |
| **Recall capacity ≤ state size** | the binding constraint | Based Thm 3.1 [A]; Schlag §4.1 [A] `d_dot` orthogonality; Zoology Thm 4.4 [A] | Theory, three independent formalisations | **A** (theory) |
| **Short depthwise convolution** | large; outranks the feature map | DeltaNet Table 1 [T] w/o→w. conv: FDA 4.5→12.8, LMB ppl 50.87→37.37; GDN Table S.1 [T] +1.60 PPL; Based Table 6 [T] convs ✗ → FDA 2.36 | **Yes**, three papers | **A** |
| **Sliding-window / exact attention** | dominates every other ingredient | 5 tables, §8 | **Yes**, repeatedly | **A** |
| **Learned decay value ≈ 0.95 mean, min ≈ 0** | — | SDM arXiv:2607.07386 Fig. 12 caption [A-fig], 1.4B, mean over 5 layers | Not an ablation; an observation | **D** |
| **Decay should increase with depth** | short-range low, long-range high | HGRN §3 [A], Fig. 2 [A-fig], Tables 8/10 captions located, **cells not transcribed** | Yes, in HGRN — unread cells | **D** (here) |
| **Content kernel's contribution given a fixed envelope** | — | — | **[X] nowhere in literatures (i)–(iv)** | — |

## 10. Single-source and unverified claims

**Single-source (one paper, no independent replication found):**
- The erase/write asymmetry (erase carries most of the gain) — **GDN-2 Table 5 only**.
- The `[0,2]` erase-range null — **GDN-2 Table 5 only**, and it contradicts a cited theoretical line (Grazzi et al. 2024; Siems et al. 2025) that I did **not** read.
- The learned decay mean `exp(g) ≈ 0.95` — **SDM Fig. 12 caption only**, in an architecture that is not Gated DeltaNet.
- Spikiness/monotonicity as *the* two properties — **Hedgehog only**. Widely cited, never independently re-derived in anything I read.
- Component collapse in hybrid conversions — **arXiv:2510.05901**, with Liger Table 6 as an independent corroboration the paper itself points at and which I verified in Liger's own cells. Counts as **two**.
- The scale reversal of DeltaNet vs GLA at 1.3B — **arXiv:2406.06484 only**.

**Unverified / read but not transcribed:**
- HGRN Tables 8 and 10 — captions located, cells **not** read. Any statement about HGRN's forget-gate ablation magnitudes is currently **[X]**.
- HGRN Figure 2's forget-rate values — a plot; **not read**.
- Based Figures 2, 3, 8 — the memory-recall Pareto curves. **Plots; not read.** Based's feature-map Pareto claim rests on Figure 3, which I have only through the paper's prose.
- arXiv:2607.02303 (Hippocampus) and arXiv:2605.05838 (MDN) — titles and abstracts only.
- Grazzi et al. 2024, Siems et al. 2025, Liu et al. 2024 (Longhorn), KDA's own paper, Mamba-3's own paper — reached only through citing papers. **Every statement about them here is second-hand and tagged to the citing source.**
- Taylor-Calibrate and LoLCATs — **not re-read this session.** The brief's characterisation of Taylor-Calibrate (student = Gated DeltaNet; three of four calibrated quantities shape the mixing) is carried over from earlier work and is **not re-verified here**.

**Corrected en route (recorded so it is not repeated):** a web-search summary asserted that a 2026 paper reports "the forget gate converges early to `exp(g) ≈ 0.95`" and attributed it to arXiv:2607.07953. In that paper, `0.95` is a Nesterov-momentum / Adam-`β₂` hyperparameter. The gate statement is in **arXiv:2607.07386**. **The search summary confabulated the attribution.** No search-engine prose entered this document as evidence.

---

## 11. What I think is wrong, or under-specified, in the framing

Reported as mechanism and evidence, per the brief. No course of action.

### 11.1 The R2a metric is dimension-matched across arms — which means the literature's own explanation for the Taylor map does not apply to it

From the anchors given: random gaussian sits at `sqrt(1 − rank/T) = 0.9015`, so `rank/T = 1 − 0.9015² = 0.1873`, and with `T/D = 5.33`, `D/T = 0.1876`. **`rank = D`.** So the reachable set of a value-side linear solve is the column space of `A·X`, of dimension at most `D`, **for every arm** — the elementwise arms, the Taylor arm, the decay arms, and `A := I` alike.

**Consequence 1 — a correction.** Based's stated reason for choosing the 2nd-order Taylor map is *capacity*: §4.1 [A] *"it expands the recurrent state size (improving recall capacity) without changing the number of parameters"*, with `elu+1` and `ReLU` explicitly *"at the Pareto frontier"* alongside it, and *"other simple feature maps may be effective as well."* Its Table 6 [T] shows recall tracking effective feature dimension (16→8.10, 32→19.35, 45→23.40, 128→25.47, 153→29.16, 325→34.38, 561→37.62 SWDE) essentially regardless of which kernel produced that dimension. **None of that mechanism operates in the R2a metric, because R2a's reachable dimension is `D` no matter what `d̃` the kernel would induce.** So the Taylor arm's separation in the R2a table **cannot be read as corroborating Based's construction**, and Based's own argument does not predict it. Whatever separates that arm is something Based does not claim.

**Consequence 2 — this is genuinely new relative to the literature I read.** §9 records **[X] no dimension-matched kernel-shape comparison in any source read** — Hedgehog's decisive Table 2 puts `1+ELU` at `O(nd²)` against Taylor at `O(nd³)` and the paper itself calls that an *"efficiency-effectiveness trade-off"* [A]. **R2a is the dimension-matched comparison that literature does not contain.** The right frame for the separated arm is therefore Hedgehog's — a *shape* property (spikiness / monotonicity), which is dimension-independent — not Based's.

**Consequence 3 — a control I would want before that arm is believed.** `1 + z + z²/2 = ½(z+1)² + ½`. It is strictly positive (minimum `0.5` at `z = −1`) but **non-monotone: for `z < −1` it *increases* as the query and key become more dissimilar.** Hedgehog states the precondition explicitly, §4.1 [A]: *"as a general property of polynomials, the Taylor approximation only tracks its original function with low error in bounded regimes"*, and verified for BERT that *"the BERT query-key dot products are bounded in regimes where the second-order Taylor series exp approximation maintains monotonicity."* **Nobody has verified that for a modern decoder-only donor, and no source read reports the `qᵀk` range for any Qwen-family model. [X]** A reconstruction-residual metric only asks whether the target lies in a span; an anti-similarity component adds a roughly independent direction to the row mixing and can therefore *lower* residual while corresponding to nothing the donor computes. That is a plausible artifact of exactly the class this project's planted-control law targets. Note also that the wrong-sequence and wrong-layer controls appear as columns computed for the elementwise construction; **whether the Taylor arm has its own matched wrong-sequence control is not stated in the framing, and it is the control that would settle this.**

### 11.2 "Identity" is used for two different objects in the framing

The brief says *"The scale is anchored at both ends: identity gives exactly 0"*, and the same table has a column *"no mixing at all (`A := I`)"* at 0.4804 / 0.4163 / 0.4242. These cannot both be the same object. The zero anchor is presumably `A := A_softmax` (the donor's own mixing, trivially reachable); `A := I` is an arm. **If the zero anchor is the donor's own mixing, the scale is anchored at one end by a tautology and at the other by a random-matrix floor, and there is no anchor at all in between** — which is precisely the region where every arm except random gaussian lives (0.287–0.580). Worth stating which was meant, because it changes what "the scale is anchored at both ends" licenses.

### 11.3 "Doing no mixing whatsoever beats three of the four" is corroborated, and the literature has a name for it

This is the reading I would most defend. Three published instances of the same shape:
- **arXiv:2510.05901 Table 1 [T]:** converted Llama3-8B, *Linear only* AVG **33.55** vs *No Attention* **34.17** — the linear branch is **worse than deleting attention**. Mistral 34.78 vs 34.40; Llama3.1 33.96 vs 33.93. Authors [A]: *"all collapse to roughly the same low performance … LA contributing little to downstream accuracy and even being detrimental."*
- **arXiv:2510.05901 Table 4 [T]:** `1+ELU` feature map at **35.12** against a zero-mixing control at **34.40** — a margin of 0.72 on a 68.26 scale. Authors [A]: *"Interestingly, 1+ELU does not appear to carry this same performance despite its close similarity to the exponential function by itself."*
- **Liger Table 6 [T]:** deleting the entire gated-linear branch (`w/o GLA`) costs **0.05 PPL and 0.4 Avg-no-MMLU**, while deleting the sliding window costs **0.79 PPL and 12.2**.

The field's term is **component collapse / component imbalance**. **The R2a reading is not anomalous; it is the geometric pre-image of a result the conversion literature has already published end-to-end.** The one thing I would add: those papers reach it *after* training the feature map, and still land at the zero control. R2a reaches it *before* training, as a ceiling. **A ceiling at the zero control is a stronger statement than a trained result at the zero control**, because no amount of training can move it.

### 11.4 The delta rule is not testable by the same pre-screen that tests decay — a structural point

Decay is expressible as a fixed `T×T` mask: arXiv:2412.06464 §2.1 [A] gives `O = ((QKᵀ) ⊙ Γ)V` with `Γ_ij = γ_i/γ_j`. A decay arm is therefore exactly the same kind of object as a kernel arm, and a decay sweep is well-posed in the R2a framework.

**The delta rule is not.** Its transition is `S_{t-1}(I − β_t k_t k_tᵀ)`, a *data-dependent* operator whose effect at `(t,s)` depends on every intervening key. It can be materialised as a `T×T` matrix through the WY representation (arXiv:2406.06484 Eq. 4 [A]: `P = I − Σ w_i k_iᵀ`), but doing so requires a value for `β_t` at every position — and `β_t` is a *learned* projection with **no closed form derivable from a softmax donor**. Under the SGD reading (arXiv:2412.06464 §3.1 [A]) `β_t` is a learning rate; there is no donor quantity it corresponds to.

**So a pre-screen can rank decay envelopes and kernels against each other, but it cannot place the delta rule on the same axis without committing to a `β` schedule that the literature gives no principled way to choose.** If the brief's worry is "what if it is the delta rule rather than decay", that worry cannot be resolved by the instrument that produced the R2a table. I record this as a property of the two operators, not as advice.

### 11.5 On the decay measurement — literature bearing only, no interpretation offered

Per instruction I do not interpret the new decay numbers. Three literature facts that bear on them, stated flat:
- The only published learned value I found is a **mean `exp(g) ≈ 0.95` with minimum ≈ 0 and "a wide dynamic range"**, at 1.4B, averaged over 5 layers, in a non-GDN architecture using GDN/FLA gate conventions — **SDM Fig. 12 caption [A-fig]**. Half-life of a `γ = 0.95` envelope: **≈ 13.5 tokens** (derived).
- At **initialisation** under those same conventions (`A ~ U[0,16]`, `Δ ~ U(0.001, 0.1)`, `α = exp(−ΔA)`), `α` spans **0.202 to 1.0** (derived from two [A] init ranges). The optimiser's prior already covers a full sweep.
- **HGRN's thesis is that the correct decay is depth-dependent** — short at the bottom, long at the top, enforced structurally by a `cummax` lower bound — and its ablation text [A] reports that *"the most significant improvement arises from the monotonically increasing lower bound"* and that a *data-independent* forget gate underperforms a data-dependent one. **On that account a single scalar `γ` compared across layers 14 / 21 / 27 is being compared against three different correct answers.** Whether that transfers to a converted softmax donor: **[X]**.
- The mapping I would flag, without asserting it applies: a fast envelope is a *local window* by another name (`γ = 0.5` ⇒ 1-token half-life), and **the single most consistent empirical result in everything read here is that a small exact/local window dominates every choice of long-range linear mixing** (§8, five tables). If a decay arm's advantage grows monotonically as the envelope shortens, the literature's prior is that it is measuring the value of locality, not the value of decay-as-memory-control. Offered as a mapping to check, not as a reading of your numbers.

### 11.6 Two smaller corrections

- **"Hedgehog-shaped" is not Hedgehog.** Every Hedgehog result in the literature is a *per-layer, per-head MLP distilled against the donor's own softmax attention weights with cross-entropy soft labels* (§7.4). A frozen closed-form function with a hedgehog-like shape is a different object, and Hedgehog's own thesis predicts it fails. If the R2a arm labelled "hedgehog-shaped" is a fixed function, the label overstates the connection — and its landing indistinguishably from `elu+1` is then **consistent with**, not counter to, Hedgehog.
- **The brief's premise that the paper landscape would answer "is it decay, or the delta rule".** It largely does not, and where it does the answer is *neither, at the margin that matters*: on general language modelling the whole delta rule is worth **+0.25 Avg** (GDN-2 Table 2) to **+0.43 Avg** (GDN Table 3) over pure decay, single-seed, no error bars anywhere; a pure-decay Mamba-3 MIMO **beats** Gated DeltaNet in the newer table. Against this project's own `σ_seed ≈ 0.005` discipline on BPB, and with **no seed replicates published in any of these papers [X]**, those margins are not separable from run noise on the evidence presented. The ingredients that survive that filter are the ones in the **A** rows of §9: normalisation, short convolution, dimension, and exact local attention.

### 11.7 MMLU, as required

- **[X] no MMLU:** arXiv:2412.06464 (Gated DeltaNet, any scale), arXiv:2605.22791 (GDN-2), arXiv:2402.04347 (Hedgehog), arXiv:2402.18668 (Based).
- **MMLU reported, and it is the worst column:** arXiv:2406.06484 Fig. 5 [T] — DeltaNet-3B vs its training-matched control PowerLM-3B: Average −2.5, **MMLU −4.3** (largest relative drop, −9.6%); Mamba-2.7B **26.1** and RWKV-6-3B **28.4** against a 25.0 chance floor while their Averages read as ordinary deficits.
- **MMLU reported per-column in the conversion regime:** arXiv:2510.05901 Table 1 [T] — Llama3.1-8B base **68.05** → hybrid conversion **58.89** while AVG moves 73.01 → 71.08. **The average understates the MMLU loss by 4.7×.** Every LA-only row across Tables 1–4 is **at or below the 25.0 chance floor**.
- **Liger is the one paper that reports both averages side by side** — Table 6 [T] columns `Avg.` and `Avg. (no MMLU)`, e.g. 67.6 / 72.4 for Liger-GLA. That is the reporting practice the rest of this literature lacks.

---

## 12. Sources index — what was read, at what depth, from what render

**Read as cells (tables transcribed verbatim):**

| arXiv | Title | Version / date read | Render | Tables transcribed |
|---|---|---|---|---|
| 2412.06464 | Gated Delta Networks: Improving Mamba2 with Delta Rule | **v3, 2025-03-06** (ICLR 2025 camera-ready) | `arxiv.org/html/` | 1, 2, 3, 4, S.1 |
| 2406.06484 | Parallelizing Linear Transformers with the Delta Rule over Sequence Length | **v6, 2025-01-15** (NeurIPS 2024) | `arxiv.org/html/` | 1 (340M + 1.3B + ablations), Fig. 3 (MAD), Fig. 5 (3B, MMLU) |
| 2102.11174 | Linear Transformers Are Secretly Fast Weight Programmers | **v3** (ICML 2021) | `arxiv.org/html/` | 2, 3 |
| 2402.18668 | Simple linear attention language models balance the recall-throughput tradeoff (Based) | **v2** | `arxiv.org/html/` | 6 |
| 2402.04347 | The Hedgehog & the Porcupine (ICLR 2024) | latest | **ar5iv** (arXiv HTML empty) | 1, 2, 10, 11 |
| 2503.01496 | Liger: Linearizing LLMs to Gated Recurrent Structures | latest | `arxiv.org/html/` | 6 |
| 2510.05901 | Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods | latest | `arxiv.org/html/` | 1, 2, 3, 4 |
| 2605.22791 | Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention (NVIDIA) | **v1, 2026-05-21** | `arxiv.org/html/` | 1, 2, 5 |

**Read as prose only (no cells transcribed):**
- 2312.04927 *Zoology* — Theorems 4.4 / 4.5 and Proposition 4.3 via §1 text.
- 2607.07386 *Sparse Delta Memory* — §setup and Appendix B Figure 12 caption.
- 2607.07953 *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing* (2026-07-08) — §1, §3 setup.
- 2311.04823 *HGRN* (ar5iv) — abstract, §3, ablation captions. **Tables 8 and 10 cells NOT read.**

**Located, not read:** 2607.02303 (Hippocampus for Linear Attention), 2605.05838 (MDN), 2509.24552 (Short window attention enables long-term memorization).

**Reached only through citing papers — every claim about them here is second-hand:** Grazzi et al. 2024 and Siems et al. 2025 (negative eigenvalues / state tracking), Liu et al. 2024 (Longhorn), the KDA paper, the Mamba-3 paper, Mamba-2/SSD, GLA, T2R, SUPRA, LoLCATs, MOHAWK.

**Method note.** Every arXiv page was downloaded with `curl` and de-tagged locally, then read as text; no table number in this document came from a web-search summary, an abstract, or a third-party blog. Where the LaTeXML render scrambled a caption/table pairing (arXiv:2412.06464 Tables 3 and 4 are captioned on opposite sides of their bodies), that is noted at the point of use. One search-engine confabulation was caught and is recorded in §10.
