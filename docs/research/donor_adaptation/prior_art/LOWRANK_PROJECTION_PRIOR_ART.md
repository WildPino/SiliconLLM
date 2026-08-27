# Low-Rank Attention Projection Prior Art — Question 2

**Author:** the Researcher · **Date:** 2026-08-22 · **Branch:** `research/donor-adaptation`
**Commissioned by:** the Adapter / Principal, after `ADAPTER_MEMO_01` §2.2f made low-ranking `W_q`/`W_o`
the difference between a **~26 B** and a **~88 B** donor target.

> **⚠ UPDATE 2026-08-22, mid-pass:** the coordinator reports **D5 has measured that the 21.25 GB/s rate
> constant does NOT transfer to donor width and the ternary path stays compute-bound there.** By
> `ADAPTER_MEMO_01`'s own model (`tok/s = rate ÷ bytes`), **a lower rate scales every size target down
> proportionally — the ~26 B and ~88 B rows fall together and their RATIO is preserved.** So this makes
> the low-rank question **more** load-bearing, not less: byte-reduction levers (5-trit packing included)
> lose their footing on a compute-bound path, while a **structural** reduction — fewer weights to
> multiply, not merely fewer bytes to read — still pays on a compute-bound kernel. **§8.2 of this file
> made exactly this point independently before the D5 result landed, and it now applies with force.**
> **One caveat that must travel with it: on a compute-bound path the relevant unit is MACs, not bytes.
> A rank-`r` factorisation of an `8192×8192` matrix costs `2·8192·r` MACs against `8192²` — the same
> 17.6 % ratio as the bytes — so low-rank happens to reduce both. That is a property of low-rank and
> NOT of ternary packing, which reduces bytes only. This is why the two levers behave differently under
> D5's finding, and the memo does not currently distinguish them.**

**Question:** can a pretrained donor's attention projections — specifically `W_q` and `W_o`, which are
67.1 M each of the 151 M per attention block at `d_model = 8192`, GQA 64/8 — be low-ranked to **~17.6 %
of their bytes**, and at what measured cost?

> **⚠ This file is written incrementally, method by method, and was begun after one session-limit kill on
> a prior pass. Every section is final when written. `[PENDING]` marks what has not been reached.**
> **Standing rule in force: a number does not enter a decision until it has been read IN THE PAPER'S OWN
> TABLE. Tags: `[T]` = table, with coordinates · `[A]` = text/abstract only · `[X]` = not found.
> `[T?]` = read from a table but not re-verified in a second pass.**

---

## 0. Running verdict — updated as evidence lands

**Pass complete: ASVD (§2), SVD-LLM (§3), counter-evidence (§5), verdict (§7). Read §0.3 before
anything else; it is the part that does not flatter us. The full verdict is §7; §8 lists what I think is
wrong in the Adapter's framing, including a check of the ~26 B / ~88 B arithmetic itself.**

> ### ONE-LINE ANSWER
> **NOT ESTABLISHED.** The direction is supported qualitatively (`W_q` is among the most compressible
> matrices); the magnitude is not supported at all (no published measurement near 17.6 %, and the
> nearest training-free point collapses to a 0.04 downstream average); **`W_o` — half the lever — is
> unexamined by every per-matrix analysis found**; and low-rank **does not compose** with aggressive
> quantisation at settings milder than ours, measured twice by two groups.

### 0.1 What the literature says in our favour

- **The qualitative direction is real and published.** `W_q` and `W_k` are the **most** compressible
  matrices in a transformer block under activation-aware low-rank; the MLP matrices are the **least**.
  ASVD states this in its own words `[A, §4.5]`. **This is the opposite of the intuition that would have
  killed the lever outright**, and it is consistent with (though not evidence for) our in-house `x_proj`
  result.

### 0.2 What the literature says against

- **No published method reaches anything close to a 0.176 parameter ratio at usable quality.** ASVD's
  own headline scope is *"compress a network by 10 %–30 %"* `[A, abstract]` — i.e. ratios 0.9–0.7 — and
  its best-in-class comparator SVD-LLM at whole-model ratio 0.6 gives **wiki ppl 16.14 against a 5.47
  baseline** `[T, ASVD Table 3, LLaMA-2-7b column]`. **That is a ~3× perplexity blowup at 0.6, and we are
  asking about 0.176.**
- **Low-rank and aggressive quantisation demonstrably do NOT compose**, measured in ASVD's own Table 4.
  See §2.5. **This is the fifth question the Adapter asked and the answer is the one the project keeps
  measuring: sequential stages do not compose.** Confirmed independently by SVD-LLM's own limitations
  section (§3.3): its authors say their method should be *"comparable to ... quantization ... rather than
  being combined with the quantization methods for usage."* `[T, §A.11]`
- **The training-free variant of the best-in-class method collapses well before our target.**
  SVD-LLM (W) on LLaMA-7B: downstream average `0.57 → 0.49 → 0.38 → 0.11 → 0.04` at
  `0 / 20 / 40 / 60 / 80 %` compression `[T, SVD-LLM Table 1]`. **Our target is 82.4 % compression on the
  two matrices concerned.** See §3.3 and the two readings in §3.4.
- **Theory separates full-rank from low-rank attention**, with the authors conjecturing full-rank is
  necessary for long context `[A, arXiv:2407.16153]` — scoped carefully in §5.1, because it is about the
  QK circuit's rank and not directly about compressing `W_q`.

### 0.3 ⚠ The hole in the evidence that matters most, stated first

> **`W_o` — the output projection — is not covered by the finding that flatters us.**

ASVD's per-type analysis names exactly three matrices in the attention block: *"In MHA, the **V
projection** layer experiences relatively small compression, whereas **q projection** and **k
projection** can be significantly compressed"* `[A, §4.5]`. **`o_proj` does not appear in the sentence,
in the figure caption, or anywhere in the per-type discussion.**

**`W_o` is 67.1 M of the 134.2 M the Adapter's lever depends on — exactly half of it.** The one published
per-matrix result found so far is silent on half the lever, and **silence is not a positive result.**

A second hole in the same place: the per-type result is presented in **Figure 6**, and the paper states it
only in words. **No numeric per-matrix compression ratio is published — `[X]`.** "Significantly
compressed" is not a number and cannot be turned into one by reading a plot I have not seen.

### 0.4 ⚠ Structural caveat on transferring ASVD's result to our donor at all

ASVD's per-type analysis is on **LLaMA-2-7b**, which is **MHA, not GQA**: `d_model = 4096`, 32 heads,
so `W_q`, `W_k`, `W_v`, `W_o` are **all 4096×4096 and all the same size**. On our GQA 64/8 donor geometry
`W_k` and `W_v` are **1/8 the size of `W_q`**.

> **Consequence: ASVD's "q and k are the compressible ones" is worth much less to us than it looks.**
> Half of that finding (`k_proj`) applies to a matrix that is 8.4 M of 151 M — **5.6 % of the attention
> block.** Compressing it perfectly buys almost nothing. **The finding transfers to `W_q` and says
> nothing about `W_o`.**

---

## 1. What the question actually is, restated so the answer cannot drift

The Adapter's framing decomposes into five sub-questions. Answers are filled in as they land.

| # | sub-question | state |
|---|---|---|
| 1 | Is the SVD activation-weighted — `min ‖(W − W_r)X‖_F` rather than `‖W − W_r‖_F`? Closed-form? Ratio and cost from the paper's own table? | **ANSWERED for ASVD (§2) and SVD-LLM (§3): both activation-weighted, both closed-form, both with a non-closed-form second stage** (perplexity search / LoRA). FWSVD `[X]` — own paper unread. Ranked table in §6. |
| 2 | **Per-matrix** results for `W_q`/`W_k`/`W_v`/`W_o` vs the MLP — not model averages | **ANSWERED, badly for us: qualitative only, no numbers anywhere, and BOTH papers omit `W_o`** (§0.3, §2.3, §3.6) |
| 3 | Does anyone low-rank **per head** rather than across the whole matrix? | **`[X]` in every paper read** — ASVD no (§2.6), SVD-LLM no (§3.6). **LatentLLM is the unread lead** (§5.4). **§1.2 argues this is the question that decides the lever.** |
| 4 | **Counter-evidence**: papers finding attention projections are NOT low-rank | **Found — theory (§5.1) plus counter-evidence inside the favourable papers' own tables (§5.3)** |
| 5 | Does low-rank compose with **ternary quantisation**? | **NO at aggressive ratios — measured, twice, two independent groups** (§2.5, §3.3) |

### 1.1 The arithmetic the answer has to clear, computed once and shown

Donor geometry per `ADAPTER_MEMO_01` §2.2f: `d_model = 8192`, `d_ffn = 28672`, GQA 64/8.

| matrix | shape | params | share of attn block | share of layer (856 M) |
|---|---|---|---|---|
| `W_q` | 8192 × 8192 | 67.1 M | 44.4 % | 7.8 % |
| `W_k` | 8192 × 1024 | 8.4 M | 5.6 % | 1.0 % |
| `W_v` | 8192 × 1024 | 8.4 M | 5.6 % | 1.0 % |
| `W_o` | 8192 × 8192 | 67.1 M | 44.4 % | 7.8 % |
| **attn total** | | **151.0 M** | 100 % | 17.6 % |
| FFN (gate+up+down) | | 705.0 M | | 82.4 % |

**`W_q` + `W_o` = 134.2 M = 88.9 % of the attention block and 15.7 % of the layer.** The Adapter's "89 %"
is correct.

**A low-rank factorisation `W ≈ A·B` with `A: 8192×r`, `B: r×8192` costs `2·8192·r` parameters.**
Setting that to 17.6 % of 67.1 M:

```
2 · 8192 · r  =  0.176 · 67.1e6      ⇒   r ≈ 721
```

> **So "17.6 % of the bytes" at donor width means rank ≈ 721 out of 8192 — a ratio r/d ≈ 8.8 %.**
> **This is worth stating explicitly because it is NOT the same object as the in-house `x_proj` result.**
> The Inventor's `r = 26` was on a `d_model = 1536`-class projection; `26/1536 ≈ 1.7 %`. **At donor width
> the same *byte* fraction corresponds to a ~5× more generous *rank* fraction**, because the byte cost of
> a rank-`r` factorisation grows as `2dr` while the dense matrix grows as `d²`. **This is the one part of
> the Adapter's framing that is more favourable than it looks, not less — and I flag it as such
> precisely because it flatters us and therefore needs saying out loud rather than being quietly
> banked.** (The countervailing consideration is in §1.2.)

### 1.2 ⚠ Why §1.1's favourable reading does not settle anything — the head-structure objection is correct

The Adapter's own objection stands and this pass has found nothing to weaken it:

At `d_model = 8192` with 64 heads, `W_q` is a **stack of 64 slices of shape 8192 × 128**. Each slice has
rank ≤ 128; the stack has rank ≤ min(8192, 8192) and is generically full-rank. **Rank 721 across the
whole matrix is ≈ 11.3 slices' worth of rank shared among 64 heads** — i.e. it asserts the 64 heads'
query subspaces overlap almost completely.

> **That is a strong structural claim, and it is exactly the claim no paper found so far tests.**
> ASVD decomposes `W_q` as one matrix and reports a plot; it never asks whether the retained subspace is
> shared across heads or dominated by a few. **Sub-question 3 is therefore not a refinement — it is the
> question.** `[PENDING]`

### 1.3 One more piece of framing the Adapter should have, on the D1 comparison

The Adapter says *"low-rank is not sparsity and D1 does not transfer."* **Correct, and the mechanism is
worth naming**: unstructured sparsity removes individual entries and destroys the matrix's *coordinate*
structure, whereas low-rank preserves every coordinate and removes *directions*. A matrix can be
simultaneously very badly approximable by a sparse matrix and very well approximable by a low-rank one.
**But the inverse also holds and is the risk here:** a matrix whose energy is spread evenly across
directions (a "generic" or well-conditioned matrix) is *maximally hostile* to low-rank while being
perfectly fine under quantisation. **Nothing in D1 predicts which of these `W_q` is. Only a spectrum
measurement does — and that is one SVD per matrix, cheaper than any probe on the board.**

---

## 2. ASVD — Activation-aware Singular Value Decomposition (arXiv:2312.05821v4, 29 Oct 2024)

**Full citation:** Zhihang Yuan (Houmo AI), Yuzhang Shang (Illinois Inst. of Technology), Yue Song
(Univ. of Trento), Qiang Wu (Houmo AI), Yan Yan (IIT), Guangyu Sun (Peking University). *"ASVD:
Activation-aware Singular Value Decomposition for Compressing Large Language Models."*
Read from arXiv HTML v4, tables parsed from the raw table markup.

### 2.1 Mechanism — is it activation-weighted, and is it closed-form? [T, Eq. 1–9]

**Yes to both.** The paper's stated objective, verbatim as Eq. (1):

```
W_k*  =  arg min_{W_k}  ‖ W_k X  −  W X ‖_F²                                    (1)
ΔY    =  ( W_k − W ) X                                                          (2)
```

> **That is literally the activation-weighted objective the Adapter asked about** — `‖(W − W_r)X‖_F`, not
> `‖W − W_r‖_F`. Confirmed in the paper's own equation, not inferred.

The method achieves it by a **whitening-then-SVD** trick rather than by solving the weighted problem
directly [T, Eq. 3–6]:

```
W  =  W S S⁻¹  =  (W S) S⁻¹                                                     (3)
W S  ≈  U'_k Σ'_k V'_kᵀ            (plain SVD of the transformed matrix)        (4)
V''_kᵀ  =  V'_kᵀ S⁻¹                                                            (5)
W  ≈  U'_k Σ'_k V''_kᵀ  =  W_k                                                  (6)
W X  =  (W S)(S⁻¹ X)                                                            (7)
```

with two choices of the transform `S` [T, Eq. 8–9]:

```
S_ii := ( (1/n) Σ_j |X_ij| )^α                       (8)   "abs mean", α a hyper-parameter
S    := L ,  where  L Lᵀ = X Xᵀ                      (9)   Cholesky whitening
```

**Closed-form: YES.** One SVD per matrix plus a diagonal or Cholesky transform. **No gradient descent,
no fine-tuning** — the paper calls itself *"a training-free approach"* `[A, abstract]`.

**But the rank allocation is NOT closed-form.** §3.4 "Sensitivity-based Truncation Rank Searching"
requires, per layer and per candidate truncation level, **a forward pass to compute perplexity**, then a
binary search over ranks. The paper names this as a cost in its own limitations, verbatim:

> *"the need to evaluate the sensitivity of each layer requires a forward propagation step to calculate
> perplexity, demanding significant computational resources."* `[A, Appendix A.1]`

> **Decision-relevant for us: the cheap part (the SVD) is closed-form; the part that produces the
> favourable per-matrix allocation is a search driven by repeated perplexity evaluations.** Any in-house
> replication inherits that cost, and `ADAPTER_MEMO_01` has not budgeted it.

### 2.2 Quality vs compression — the paper's own main table [T, Table 2]

Transcribed verbatim from the raw table markup, all columns:

| method | param ratio | LLaMA-7b MMLU | wiki | ptb | LLaMA-2-7b MMLU | wiki | ptb | LLaMA-2-13b MMLU | wiki | ptb |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 1 | 30.76 % | 5.68 | 29.63 | 34.86 % | 5.47 | 20.82 | 40.16 % | 4.88 | 29.21 |
| SVD | 0.95 | 22.98 % | 2800 | 5458 | – | nan | nan | – | nan | nan |
| SVD* | 0.95 | 23.92 % | 136.05 | 183.92 | 24.78 % | 46.79 | 363.37 | 24.86 % | 167.63 | 567.02 |
| SVD* | 0.9 | 23.54 % | 698.66 | 262.03 | 24.31 % | 114.45 | 27660 | – | nan | nan |
| ASVD | 0.95 | 30.26 % | 5.78 | 32.64 | 33.24 % | 5.64 | 23.98 | 39.52 % | 4.94 | 31.93 |
| ASVD | 0.9 | 29.67 % | 6.09 | 37.80 | 32.58 % | 5.93 | 32.63 | 40.04 % | 5.12 | 34.03 |
| ASVD | 0.85 | 29.70 % | 6.80 | 52.11 | 31.57 % | 6.74 | 59.84 | 37.95 % | 5.54 | 39.32 |
| ASVD | 0.8 | 27.85 % | 8.89 | 88.09 | 28.15 % | 8.91 | 114.70 | 34.63 % | 6.53 | 59.68 |
| ASVD | 0.75 | 24.94 % | 14.51 | 212.80 | 25.97 % | 18.97 | 432.57 | 28.59 % | 8.71 | 110.10 |

**Two readings, and the second is the one that matters.**

1. **Activation-awareness is enormously load-bearing.** Plain SVD at ratio 0.95 gives LLaMA-7b wiki ppl
   **2800** against the baseline's **5.68**; ASVD at the same ratio gives **5.78**. `[T, rows "SVD" and
   "ASVD" at param ratio 0.95, LLaMA-7b wiki column]` **The naive Frobenius objective is not merely worse
   — it is catastrophic.** Any in-house probe that low-ranks by plain SVD and reports a bad number has
   measured nothing.
2. **The quality curve is already steep well above the ratios we need.** On LLaMA-2-7b wiki:
   `1 → 5.47`, `0.95 → 5.64`, `0.9 → 5.93`, `0.85 → 6.74`, `0.8 → 8.91`, `0.75 → 18.97`. **Between 0.8
   and 0.75 the perplexity doubles.** ptb is worse: `20.82 → 432.57` over the same span.

> **⚠ The "param ratio" column is a WHOLE-MODEL ratio, not a per-matrix one.** It is not directly
> comparable to our 0.176-on-`W_q`/`W_o` target, and the naive comparison ("0.176 ≪ 0.75, therefore
> hopeless") is **wrong** — see §2.4 for the correct comparison, which is less bad but still bad.

### 2.3 The per-matrix finding — verbatim, and the caveats stated with it [A, §4.5 / Figure 6]

Verbatim, §4.5 "Decomposed Network Analysis":

> *"Figure 6 presents the per-type parameters ratio and per-block parameters ratio. Observing the plot,
> we note that **parameters in the MLP components (gate projection, up projection, and down projection)
> exhibit minimal compression. In MHA, the V projection layer experiences relatively small compression,
> whereas q projection and k projection can be significantly compressed, indicating redundancy in these
> components.** Turning our attention to the per-block compression ratio, we find that the first layer
> can undergo substantial compression. In contrast, the compression ratios for the other layers, except
> for two middle layers, show similar compression rates."*

Corroborated by §3.4, verbatim: *"**Higher Sensitivity in MLP Layers**: MLP layers demonstrate higher
sensitivity, indicating where more cautious truncation is necessary."*
And by Appendix A.1, verbatim: *"Although ASVD effectively compresses the weights in multi-head attention
(MHA) with fewer parameters, **it struggles with MLP**."*

**This is a genuine, three-times-restated, per-matrix finding and it points our way. Now the caveats,
all of which are load-bearing:**

| caveat | consequence |
|---|---|
| **It is a figure, described in words. No per-matrix number is published.** `[X]` | "Significantly compressed" cannot be converted into a rank or a byte fraction. **The single number the Adapter's ~88 B row needs does not exist in this paper.** |
| **`o_proj` is never mentioned** — not in the sentence, not in the caption, not in A.1 | **Half the lever (67.1 M of 134.2 M) is unaddressed.** See §0.3. |
| **The model is LLaMA-2-7b, which is MHA not GQA** | `k_proj` there is 16.8 M and equal to `q_proj`; on our donor it is 8.4 M and 1/8 of `q_proj`. **Half the favourable finding is about a matrix that barely exists on our geometry.** See §0.4. |
| **It is an *output* of the sensitivity search, not an independent measurement** | The allocation says "the binary search chose to compress q/k hard **subject to a whole-model budget of 0.85-ish**". It does **not** say q/k survive being compressed to 0.176 in isolation. These are different claims and only the first is evidenced. |
| **Attention is 4/7 of the matrices but a minority of the parameters in LLaMA-2-7b** | The search compresses q/k hard partly *because it must find its budget somewhere and the MLP refuses*. That is a statement about relative sensitivity under a constraint, not about absolute compressibility. |

### 2.4 The correct comparison to our target, computed carefully

The naive comparison is wrong in our favour's direction, so it needs doing properly.

Applying 0.176 to `W_q` and `W_o` only, on the donor geometry of §1.1:

```
saved per layer   = 0.824 × 134.2 M                      = 110.6 M
whole-layer ratio = (856.0 − 110.6) / 856.0              = 0.871
```

> **So the Adapter's lever corresponds to a whole-model parameter ratio of ≈ 0.87** — squarely inside the
> band ASVD reports (0.95–0.75) and near its comfortable end, where LLaMA-2-7b wiki ppl is between 5.93
> (0.9) and 6.74 (0.85) `[T, Table 2]`.

**That looks survivable. It is not, and here is the reason it is not.**

ASVD reaches whole-model 0.85 **by spreading compression across every matrix type, and specifically by
leaving MLP nearly untouched and taking its budget from q/k.** Our lever proposes to take **all** of the
compression from `W_q` and `W_o` and **none** from anywhere else. So the relevant question is not "what
happens at whole-model 0.87" but **"what per-matrix ratio does ASVD's search actually assign to `q_proj`
when the whole model sits at 0.85?"**

**That number is in Figure 6 and is not published in text. `[X]`.** Without it, the following cannot be
answered from ASVD:

- whether ASVD's search ever pushes `q_proj` as low as 0.176;
- whether the quality at whole-model 0.85 survives *because* q/k were compressed hard, or *despite* it;
- what `o_proj` was assigned.

> **Verdict on ASVD as evidence for the ~88 B row: it establishes the DIRECTION and refutes the
> nightmare case, and it supplies NO NUMBER that discharges the claim.** The 17.6 % figure remains
> unsupported by any published per-matrix measurement found so far. **It is currently an in-house
> analogy wearing a literature citation that does not actually contain it.**

### 2.5 ⚠ Composition with quantisation — the paper's own table says it FAILS at aggressive ratios [T, Table 4]

**This is sub-question 5, and it is the clearest answer in the file.** Transcribed verbatim from ASVD
Table 4 (*"Combining weight quantization with ASVD. Param ratio indicates the proportion of parameters
remaining after ASVD, with 1 implying no decomposition."*). Protocol, verbatim: *"Firstly, we apply ASVD
to decompose the network. Subsequently, we quantize the decomposed weights."* — i.e. **sequential
stages, exactly our proposed pipeline.**

**LLaMA-2-7b, wiki:**

| param ratio | FP16 | INT8 (RTN) | INT8 (AWQ) | NF4 | INT4 (AWQ) |
|---|---|---|---|---|---|
| 1 | 5.47 | 5.48 | 5.45 | 5.65 | 5.59 |
| 0.95 | 5.64 | 5.64 | 5.56 | 5.83 | 5.82 |
| 0.9 | 5.93 | 5.94 | 5.82 | 6.20 | 6.21 |
| 0.85 | 6.74 | 6.73 | 6.51 | 7.43 | 7.18 |

**LLaMA-2-7b, ptb:**

| param ratio | FP16 | INT8 (RTN) | INT8 (AWQ) | NF4 | INT4 (AWQ) |
|---|---|---|---|---|---|
| 1 | 20.82 | 20.82 | 20.93 | 22.70 | 21.50 |
| 0.95 | 23.98 | 23.95 | 25.47 | 35.91 | 27.79 |
| 0.9 | 32.63 | 32.19 | 37.11 | 40.82 | 39.31 |
| 0.85 | 59.84 | 63.76 | 84.52 | **427.59** | 95.85 |

**LLaMA-2-13b, ptb** (same table, right half): `1 → 29.15 / 29.12 / 29.29 / 30.31 / 30.47`;
`0.95 → 31.93 / 31.67 / 30.19 / 33.89 / 31.21`; `0.9 → 34.03 / 33.64 / 35.47 / 34.93 / 38.95`;
`0.85 → 39.32 / 40.02 / 43.01 / 44.49 / 50.56`.

**Read the ptb/NF4 column down.** At no compression, NF4 costs `20.82 → 22.70` (+9 %). At ratio 0.85 it
costs `59.84 → 427.59` — **a 7.1× blowup, from a quantiser that costs 9 % on its own.**

**The paper's own summary sentence is: *"In summary, the findings suggest that ASVD is compatible with
weight quantization techniques."*** — and its own supporting sentence is carefully hedged, verbatim:
*"**When param ratio is greater than 0.9**, the performance decline attributed to quantization is
approximately consistent with that of the non-decomposed network."*

> **The authors' scope of the compatibility claim is `ratio > 0.9`. Their headline sentence drops the
> condition. Their own 0.85/NF4/ptb cell violates it by 7×.** I am reporting the cell, not the sentence.

**What this means for us, stated conservatively:**

- We would be doing **ternary (~1.58 bit)**, which is *more* aggressive than NF4, on matrices low-ranked
  *far* past 0.85. **Both axes are outside where composition was observed to hold, in the same
  direction.**
- The one cell that blows up is on **ptb**, not wiki — i.e. **the out-of-distribution evaluation set**.
  wiki at 0.85/NF4 is a mild `6.74 → 7.43`. **The failure is visible only because they reported two
  corpora.** This is the exact failure mode this project banked as *"the stage-1 gate was satisfiable by
  choosing the eval corpus"* (`f75d34b`). **Any in-house low-rank × ternary probe that reports one
  corpus is not measuring composition.**
- **`ADAPTER_MEMO_01` has not costed a quality penalty for composition at all.** The ~26 B and ~88 B rows
  are both 5-trit ternary figures that assume low-rank is free on top of ternary. **ASVD's Table 4 is
  direct evidence against that assumption at ratios milder than ours.**

### 2.6 Head structure — ASVD does not do it [X]

ASVD decomposes each weight matrix **as a whole** (Eq. 3–6 operate on `W`, with a per-*channel* transform
`S` from Eq. 8/9). There is no per-head decomposition, no per-head rank allocation, and no analysis of
whether the retained subspace is shared across heads. **Sub-question 3 is unanswered by this paper.**

One adjacent thing ASVD *does* do that is worth knowing: §3.5 applies the same decomposition to the
**key/value projections to shrink the KV cache**, reporting `[T, Table 3 (KV-cache)]` for LLaMA-2-7b wiki:
`1 → 5.47`, `0.9 → 5.46`, `0.8 → 5.48`, `0.7 → 5.50`, `0.6 → 5.55`, `0.5 → 5.67`, `0.4 → 5.94`,
`0.3 → 6.55`, `0.2 → 8.71`; ptb `20.82 / 21.04 / 21.52 / 21.66 / 21.91 / 22.16 / 24.33 / 26.89 / 38.72`.

> **Note carefully what that table is and is not.** It is a **KV-cache channel ratio**, i.e. a rank
> reduction on the *activation* dimension of K/V, **not a weight-byte ratio on `W_q`/`W_o`.** It is a
> much gentler curve (0.5 costs almost nothing) and it is **tempting to misread as support for our
> lever.** It is not. It speaks to `ADAPTER_MEMO_01` §2.2f's *KV-traffic* half, which is a real and
> separate finding, and I am recording it there rather than letting it flatter the weight-stream half.

### 2.7 ASVD vs FWSVD — the one head-to-head with the Fisher-weighted method [T, Appendix A.6]

| | param ratio | 0.95 | 0.9 | 0.85 | 0.8 |
|---|---|---|---|---|---|
| **LLaMA-7b** | FWSVD+STRS, wiki | 5.86 | 6.32 | 7.48 | 10.70 |
| | ASVD, wiki | 5.78 | 6.09 | 6.80 | 8.89 |
| | FWSVD+STRS, ptb | 34.33 | 38.05 | 58.75 | 125.80 |
| | ASVD, ptb | 32.64 | 37.80 | 52.11 | 88.09 |
| **LLaMA-2-7b** | FWSVD+STRS, wiki | 5.59 | 6.12 | 8.01 | 13.07 |
| | ASVD, wiki | 5.64 | 5.93 | 6.74 | 8.91 |
| | FWSVD+STRS, ptb | 25.06 | 36.58 | 105.53 | 222.03 |
| | ASVD, ptb | 23.98 | 32.63 | 59.84 | 114.70 |

**FWSVD (Fisher-weighted SVD) is competitive at 0.95 and loses badly by 0.8.** Note FWSVD is given
ASVD's own rank-search (STRS) in this comparison, so the gap is attributable to the weighting scheme, not
the allocation. `[T, Appendix A.6]` — **single-source: this is ASVD's re-implementation of a competitor,
and ASVD has an interest in the outcome.** The direction (activation-weighting beats Fisher-weighting at
aggressive ratios) is plausible and matches ASVD's Eq. (1) objective, but **FWSVD's own paper has not
been read.**

### 2.8 ASVD — what is `[X]`

- Per-matrix compression ratio, any matrix, any number. **Figure only.**
- `o_proj` behaviour, at all.
- Any ratio below whole-model 0.75 for ASVD itself (0.6 exists only in the SVD-LLM comparison table).
- Any result on a GQA model.
- Any result above 13 B.
- Per-head analysis.

---

## 3. SVD-LLM (arXiv:2403.07378v5, 16 Mar 2025) — the aggressive-ratio evidence, and it is the one that decides

**Full citation:** Xin Wang (Ohio State), Yu Zheng (Michigan State), Zhongwei Wan (Ohio State), Mi Zhang
(Ohio State). *"SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model
Compression."* Code: `github.com/AIoT-MLSys-Lab/SVD-LLM`. Read from arXiv HTML v5; tables parsed from raw
markup.

**This is the most decision-relevant paper in this file**, because it is the only one that publishes
results at compression ratios in the neighbourhood of our 82.4 % target.

### 3.1 Mechanism, and the crucial distinction between its two variants

Two components:

1. **Truncation-aware data whitening** `[T, §3.1]` — whiten by `S` with `S⁻¹XXᵀ(S⁻¹)ᵀ = I` (Cholesky),
   then SVD the whitened `WS`. The paper derives the compression loss as
   `L = ‖(WS₀ − Σᵢ σ'ᵢ u'ᵢ v'ᵢᵀ) S₀⁻¹ X‖_F` `[T, §A.3]` — again the **activation-weighted** objective.
   **Closed-form, training-free.**
2. **Parameter update with sequential low-rank approximation** `[T, §3.2]` — **a LoRA fine-tune**, applied
   separately and sequentially to `W'_u` and `W'_v` to preserve low-rank structure, verbatim: *"SVD-LLM
   proposes a variant of LoRA fine-tuning to update the remaining weight parameters of the compressed LLM
   for accuracy recovery."* Uses **Alpaca, 50 K samples** `[T, §4 setup]`.

> **⚠ Read the variant labels carefully — this is exactly the kind of row-mislabelling that has cost this
> project before.** In the results table, **`SVD-LLM (W)` is whitening only — the training-free
> variant** — and **`SVD-LLM` is whitening + LoRA fine-tuning.** The paper says so verbatim: *"Given that
> all the SVD-based baselines do not incorporate LoRa fine-tuning, to ensure a fair comparison, we also
> compare to SVD-LLM with truncation-aware data whitening only (denoted as SVD-LLM (W))."*
> **Our pipeline has no fine-tuning stage. The row that applies to us is `SVD-LLM (W)`, not `SVD-LLM`.**

### 3.2 The main results table — LLaMA-7B, transcribed verbatim [T]

**⚠ The `Ratio` column is COMPRESSION ratio (fraction removed), not the `param ratio` (fraction kept)
used by ASVD.** `20 % ratio ≡ 0.8 param ratio`. **`80 % ratio ≡ 0.2 param ratio`, which is the closest
published point to our 0.176 target.** The two papers use opposite conventions and mixing them would
invert every conclusion.

| Ratio (Mem.) | Method | WikiText-2 ↓ | C4 ↓ | Openb. | ARC_e | WinoG. | HellaS. | PIQA | MathQA | Average ↑ | TruthfulQA ↑ | GSM8K ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0 % (13.5 GB)** | Original | 5.68 | 7.34 | 0.34 | 0.75 | 0.70 | 0.57 | 0.79 | 0.27 | **0.57** | 0.30 | 0.09 |
| **20 % (10.2 GB)** | SVD | 20061 | 18800 | 0.05 | 0.04 | 0.01 | 0.03 | 0.02 | 0.03 | 0.03 | 0.00 | 0.00 |
| | FWSVD | 1727 | 1511 | 0.09 | 0.11 | 0.05 | 0.08 | 0.10 | 0.05 | 0.08 | 0.00 | 0.00 |
| | ASVD | 11.14 | 15.93 | 0.29 | 0.53 | 0.64 | 0.41 | 0.68 | 0.17 | 0.45 | 0.21 | 0.04 |
| | **SVD-LLM (W)** | **7.94** | 15.84 | 0.31 | 0.62 | 0.61 | 0.45 | 0.71 | 0.21 | **0.49** | 0.26 | 0.05 |
| | SVD-LLM | 7.73 | 12.23 | 0.33 | 0.67 | 0.69 | 0.55 | 0.79 | 0.26 | 0.55 | 0.28 | 0.08 |
| **40 % (7.76 GB)** | SVD | 52489 | 47774 | 0.04 | 0.04 | 0.05 | 0.01 | 0.03 | 0.02 | 0.03 | 0.00 | 0.00 |
| | FWSVD | 18156 | 12847 | 0.06 | 0.05 | 0.02 | 0.00 | 0.05 | 0.03 | 0.04 | 0.00 | 0.00 |
| | ASVD | 1407 | 1109 | 0.08 | 0.11 | 0.09 | 0.08 | 0.13 | 0.08 | 0.10 | 0.01 | 0.00 |
| | **SVD-LLM (W)** | **13.73** | 75.42 | 0.25 | 0.33 | 0.55 | 0.40 | 0.63 | 0.12 | **0.38** | 0.17 | 0.02 |
| | SVD-LLM | 9.27 | 15.63 | 0.29 | 0.59 | 0.68 | 0.52 | 0.69 | 0.20 | 0.50 | 0.24 | 0.07 |
| **60 % (5.35 GB)** | SVD | 105474 | 106976 | 0.01 | 0.03 | 0.01 | 0.00 | 0.01 | 0.02 | 0.01 | 0.00 | 0.00 |
| | FWSVD | 32194 | 29292 | 0.06 | 0.02 | 0.01 | 0.01 | 0.02 | 0.03 | 0.03 | 0.00 | 0.00 |
| | ASVD | 57057 | 43036 | 0.05 | 0.04 | 0.06 | 0.09 | 0.08 | 0.05 | 0.06 | 0.00 | 0.00 |
| | **SVD-LLM (W)** | **66.62** | 471.83 | 0.10 | 0.05 | 0.17 | 0.10 | 0.21 | 0.04 | **0.11** | 0.01 | 0.00 |
| | SVD-LLM | 15.00 | 26.26 | 0.18 | 0.42 | 0.44 | 0.31 | 0.35 | 0.12 | 0.30 | 0.14 | 0.04 |
| **80 % (2.58 GB)** | SVD | 687291 | 708243 | 0.00 | 0.01 | 0.02 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| | FWSVD | 96872 | 89243 | 0.01 | 0.02 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| | ASVD | 80425 | 67927 | 0.04 | 0.03 | 0.03 | 0.02 | 0.01 | 0.01 | 0.03 | 0.00 | 0.00 |
| | **SVD-LLM (W)** | **1349** | 6224 | 0.07 | 0.03 | 0.04 | 0.02 | 0.07 | 0.01 | **0.04** | 0.00 | 0.00 |
| | SVD-LLM | 31.79 | 43.71 | 0.11 | 0.23 | 0.21 | 0.14 | 0.17 | 0.08 | 0.16 | 0.04 | 0.02 |

*(The paper also prints relative-gain annotations in green inside brackets; those are the paper's own
comparisons against the best baseline and are omitted here as derived quantities, not measurements.)*

### 3.3 ⚠ What this table says about the ~88 B row — the central finding of Question 2

**Read the `SVD-LLM (W)` rows — the training-free variant, which is our pipeline — down the WikiText-2
column:**

```
 0 % compression  →   5.68   (original)
20 % compression  →   7.94   (+40 %)
40 % compression  →  13.73   (2.4×)
60 % compression  →  66.62   (11.7×)
80 % compression  →  1349    (237×)
```

**And its downstream `Average`: `0.57 → 0.49 → 0.38 → 0.11 → 0.04`.**

> **At 80 % compression — the published point nearest our 82.4 % target — the training-free method's
> downstream average is 0.04 against an original 0.57. The model is destroyed.**

Even **with** LoRA fine-tuning, at 80 % compression the average is **0.16**, and the individual cells
show the model is below chance on binary-ish tasks (**WinoGrande 0.21**, chance ≈ 0.50; **PIQA 0.17**,
chance ≈ 0.50). **A model scoring below chance is not a degraded model, it is a broken one.**

**The authors state the limitation themselves** `[T, §A.11 (1), verbatim]`:

> *"**The compression accuracy still needs to be improved under the high compression ratio.** Although
> SVD-LLM has achieved the state-of-the-art performance compared to previous works such as FWSVD and
> ASVD, its compression accuracy still suffers from degradation, especially under the high compression
> ratio. To enhance the practicability of SVD-LLM in real-world scenario, its accuracy should be at
> least comparable to that of the quantization method, including both low-bit and high-bit quantization,
> **rather than being combined with the quantization methods for usage.**"*

> **That last clause is the authors of the best-in-class low-rank method saying their method is not yet
> good enough to be worth combining with quantisation.** It is a second, independent
> `[T]`-grade answer to sub-question 5, from a different group than ASVD, pointing the same way as
> ASVD's Table 4.

### 3.4 The correct arithmetic against our lever, done carefully in both directions

**This is where the ~88 B row lives or dies, so both readings are shown.**

**Reading A — the one that kills it.** Our target is `W_q`/`W_o` at **17.6 % of bytes = 82.4 %
compression on those matrices**. The nearest published point is 80 % compression, and there the
training-free method scores 0.04 average. **If `W_q`/`W_o` behave like the model average, the lever is
dead.**

**Reading B — the one that rescues it, and why it is not yet evidence.** SVD-LLM compresses the
**whole model**. Our lever compresses only `W_q` + `W_o` = **15.7 % of the layer** (§1.1). The
whole-model-equivalent is:

```
mine:  saved = 0.824 × 134.2 M = 110.6 M per layer
       whole-model compression = 110.6 / 856.0 = 12.9 %      (param ratio 0.871)
```

**12.9 % whole-model compression sits below SVD-LLM's mildest published row (20 %), where `SVD-LLM (W)`
gives WikiText-2 7.94 vs 5.68 and average 0.49 vs 0.57.** So on a pure byte-budget basis our lever is
*less* aggressive than the mildest thing they measured.

> **⚠ Reading B is the flattering one and it does not survive scrutiny. Here is why, stated plainly.**
>
> SVD-LLM's 20 % row spreads its 20 % across **every matrix in the model**, so no individual matrix is
> anywhere near 82 % compressed. Our lever takes **all** of its compression from two matrices and drives
> those two to 82 %. **The whole-model byte-equivalence is arithmetically correct and mechanistically
> irrelevant**: quality loss is not a function of total bytes removed, it is a function of what was
> removed from where. **ASVD's entire contribution (§2, the sensitivity search) exists precisely because
> that per-matrix allocation is what determines quality.**
>
> **Neither reading is supported by a measurement. The measurement that would settle it — one matrix
> type driven to 82 % compression with everything else dense — is not published in any paper found in
> this pass. `[X]`.**

### 3.5 Where the two papers disagree, and it matters

ASVD Table 3 (§2.2 of this file, via ASVD's own comparison) and SVD-LLM Table 1 both compare the two
methods and **they do not agree on ASVD's numbers**:

- ASVD's own paper reports ASVD at LLaMA-2-7b, param ratio 0.8 (= 20 % compression), WikiText-2 = **8.91**.
- SVD-LLM's paper reports ASVD at LLaMA-7B, 20 % compression, WikiText-2 = **11.14**.

**These are different models (LLaMA-7B vs LLaMA-2-7b) so they are not directly contradictory** — I am
flagging it because the two are one character apart in the papers' own labels and **this project has
already lost a point to exactly this confusion** (a 13 B baseline row paired with a 7 B label).
**Anyone citing either number must carry the model name with it.** More seriously: SVD-LLM reports ASVD
at 60 % compression as **57057** while ASVD's own paper's comparison table shows ASVD at param ratio 0.6
as **730.60** on LLaMA-2-7b — a two-order-of-magnitude disagreement across two papers about the same
method. **Each paper's numbers should be used only within that paper.**

### 3.6 Spectrum evidence — and note which matrix is missing again [A, §A.4]

Verbatim: *"we select the **Query (`W_Q`) and Key (`W_K`)** weight matrices and show the spectrum of
singular values of their multiplication with corresponding whitening matrices `S_Q` and `S_K`. As shown
in Figure 6, most of the single values are less than or around 100 with only a few extremely large
values, indicating that SVD is applicable in SVD-LLM."*

- **`W_O` is absent again.** Two independent papers have now published per-matrix analysis of the
  attention block and **neither examined the output projection.** That is not a coincidence to be
  explained away — it is a genuine hole in this literature, and it is where half our lever lives.
- **The spectrum shown is of `W S` (the whitened product), not of `W`.** The paper is explicit about
  this. **It is therefore not evidence that `W_q` itself is low-rank** — it is evidence that `W_q`
  becomes compressible *after* being conditioned on the activation statistics. **These are different
  claims and only the second is supported.**
- **It is a figure with no numbers. `[X]` on any rank or energy fraction.**

---

## 4. FWSVD, and the rest of the family — status

- **FWSVD (Fisher-weighted SVD)** — appears only as a baseline in both ASVD and SVD-LLM. **In SVD-LLM's
  Table 1 it is catastrophic at every ratio** (20 % compression → WikiText-2 **1727**, average **0.08**)
  `[T]`, while in ASVD's Appendix A.6 (where it is given ASVD's rank search) it is merely worse
  (`0.95 → 5.59` on LLaMA-2-7b wiki) `[T]`. **The two portrayals are wildly different and both are by
  competitors.** FWSVD's own paper has **not** been read. `[X]`.
- **AdaSVD (2502.01403), ERC-SVD (2505.20112), Layer-wise Dynamic Rank (2509.25622), SkipCat
  (2512.13494), COMPOT (2602.15200)** — surfaced, **not read**. `[PENDING]`.
- **DRONE** — SVD-LLM's §A.5 reports it as **infeasible at LLM scale**, requiring *"419 GB"* for one
  weight matrix against an A100's 80 GB, and *"larger than 24,600 GB"* in another case, versus SVD-LLM's
  3.6 GB `[A/T, §A.5]`. **Relevant to us as a cost warning: some activation-aware formulations do not
  scale**, and our own apparatus would have to be checked against the same failure.

---

## 5. Counter-evidence — searched deliberately, and here is what came back

### 5.1 Theory: full-rank attention is separated from low-rank attention [T, abstract verbatim]

**Amsel, Yehudai, Bruna — *"On the Benefits of Rank in Attention Layers"*, arXiv:2407.16153,
submitted 23 July 2024.** Abstract, verbatim:

> *"Attention-based mechanisms are widely used in machine learning, most prominently in transformers.
> However, hyperparameters such as the rank of the attention matrices and the number of heads are scaled
> nearly the same way in all realizations of this architecture, without theoretical justification. In this
> work we show that **there are dramatic trade-offs between the rank and number of heads of the attention
> mechanism.** Specifically, we present a simple and natural target function that can be represented using
> a single full-rank attention head for any context length, but that **cannot be approximated by low-rank
> attention unless the number of heads is exponential in the embedding dimension**, even for short context
> lengths. Moreover, we prove that, for short context lengths, adding depth allows the target to be
> approximated by low-rank attention. **For long contexts, we conjecture that full-rank attention is
> necessary.** Finally, we present experiments with off-the-shelf transformers that validate our
> theoretical findings."*

**How much this actually bites, scoped honestly:**

- **What it is about:** the rank of the *attention mechanism* — the QK bilinear form, governed by the
  head dimension. **It is not directly about low-ranking the stacked `W_q` as a compression operation.**
  Reporting it as "papers say `W_q` is not low-rank" would be an overstatement and I am not making it.
- **Why it still bites:** low-ranking `W_q` globally to rank 721 **does** reduce the rank of the QK
  circuit, because every head's queries are then drawn from one shared 721-dimensional subspace. The
  separation result is about exactly that kind of restriction.
- **The clause that matters most to us:** *"For long contexts, we conjecture that full-rank attention is
  necessary."* **Marked as a conjecture by the authors.** `[A]` — a conjecture is not a measurement, and
  I am recording it as a conjecture. **But it points at long-context recall, which is the same failure
  axis the linearisation literature keeps hitting** (§5.8d of `ATTENTION_LINEARISATION_PRIOR_ART.md`:
  RULER at 0.1–0.2 under aggressive conversion). **Two independent lines of evidence pointing at the same
  capability is worth more than either alone.**
- **`[X]`:** the experiments are described in the abstract as validating the theory on off-the-shelf
  transformers; **their tables have not been read this pass.**

### 5.2 Empirical: attention activations are lower-rank than attention weights — a distinction that must not be blurred

Search returned repeated claims that *"the stable ranks of attention **activations** remain low across all
layers, with Q and K generally exhibiting slightly lower ranks than V"* — i.e. about `q`, `k`, `v`
**activations**, not the `W` matrices. **These are different objects.** ASVD's own KV-cache result
(§2.6) is of this kind: a channel-rank reduction on activations, which is gentle (0.5 costs almost
nothing), whereas the *weight* curves are steep.

> **⚠ This is the single easiest way for the ~88 B row to be accidentally justified by the wrong
> evidence.** Activation low-rank results are abundant, favourable, and about a different quantity than
> the one our byte budget depends on. **`ADAPTER_MEMO_01` needs weight-matrix rank, not activation
> rank.** I am flagging it because I nearly conflated them myself while reading ASVD §3.5.

### 5.3 Counter-evidence found INSIDE the favourable papers, which is where it actually was

- **ASVD Table 4** (§2.5): low-rank × NF4 at 0.85 param ratio → ptb `59.84 → 427.59`. `[T]`
- **SVD-LLM §A.11(1)** (§3.3): the authors say their method should be *"comparable to … quantization …
  rather than being combined with the quantization methods for usage."* `[T]`
- **SVD-LLM Table 1** (§3.2): training-free variant collapses to average 0.04 at 80 % compression. `[T]`
- **ASVD Appendix A.1**: *"ASVD faces difficulties in compressing multi-layer perceptron (MLP) in LLMs …
  Although ASVD effectively compresses the weights in multi-head attention (MHA) with fewer parameters,
  it struggles with MLP."* `[T]` — favourable to us on direction, but note it also concedes the
  **absolute** benefit is small because MHA has *"fewer parameters"* on their models.

> **Recording the pattern, because it is a method lesson: in this literature the counter-evidence is
> inside the favourable papers' tables and limitations sections, not in opposing papers.** Searching for
> "papers that say low-rank fails" returns very little; reading the appendices of papers that say it
> works returns a great deal. **The deliberate counter-evidence search was less productive than reading
> the tables of the papers that flatter us.**

### 5.4 Per-head low-ranking — sub-question 3 [PENDING / partial]

- **ASVD: no.** Whole-matrix decomposition with a per-channel transform (§2.6).
- **SVD-LLM: no.** Whole-matrix, per-matrix whitening.
- **LatentLLM (arXiv:2505.18413, Koike-Akino, Chen, Liu, Wang, Wang, Brand, 23 May 2025)** — abstract
  verbatim: *"Our method extends a **local activation-aware tensor decomposition to a global
  attention-aware joint tensor decomposition.**"* **This is the best lead for the head-structure
  question and it has NOT been read beyond the abstract.** No per-matrix results, compression ratios or
  quality costs were obtainable from the abstract page. `[X]` on all of it. **If Question 2 gets another
  pass, this is where it should start.**

> **Sub-question 3 is therefore unanswered, and §1.2 argued it is the question that decides the lever.**

---

## 6. Ranked table (the deliverable)

Ranked by **usefulness as evidence for our specific lever**, not by the papers' own merit.

| Method | Activation-weighted? | Closed-form? | Training-free? | Best ratio at usable quality (own table) | Quality cost at that ratio | Per-matrix breakdown? | `W_o` covered? |
|---|---|---|---|---|---|---|---|
| **SVD-LLM (W)** (2403.07378) | **Yes** — `‖(WS−W_r)S⁻¹X‖_F` `[T,A.3]` | **Yes**, Cholesky whitening + SVD | **Yes** | **20 % compression** (param ratio 0.8) | wiki `5.68→7.94`; avg `0.57→0.49` `[T,Tab.1]` | **No** — model-level only | **No** |
| **SVD-LLM** (+LoRA) | Yes | Whitening yes; update is **LoRA fine-tuning** `[T,§3.2]` | **No** — Alpaca 50 K | **40 % compression** | wiki `5.68→9.27`; avg `0.57→0.50` `[T,Tab.1]` | **No** | **No** |
| **ASVD** (2312.05821) | **Yes** — Eq. (1) `[T]` | **Yes** for the SVD; **No** for rank allocation (perplexity search) | **Yes** | **param ratio 0.85–0.9** (10–15 % compression) | LLaMA-2-7b wiki `5.47→5.93` (0.9), `→6.74` (0.85) `[T,Tab.2]` | **Qualitative only** `[A,§4.5]`, **no numbers** `[X]` | **No** |
| **FWSVD** | Fisher-weighted, not activation-weighted | Yes | Yes | ≤ 20 % compression, and **catastrophic there in SVD-LLM's table** (`1727`) `[T]` | disputed between the two competitor papers | No | No |
| **Plain SVD** | **No** | Yes | Yes | **None** — `20061` wiki at 20 % `[T,SVD-LLM Tab.1]` | total collapse | No | No |
| **LatentLLM** (2505.18413) | Yes, "global attention-aware joint" | `[X]` | `[X]` | `[X]` | `[X]` | `[X]` — **best lead, unread** | `[X]` |

**One structural reading of this table:** *every* method that publishes per-matrix analysis publishes it
as a figure, qualitatively, covering `q`/`k`/`v` and never `o`; and *every* method that publishes hard
numbers publishes them at the model level. **The exact cell the ~88 B row needs does not exist in this
literature.**

---

## 7. ⚠ Explicit verdict — can `W_q`/`W_o` at donor width be low-ranked to ~17.6 % of bytes, and at what cost?

> ### **NOT ESTABLISHED, and the honest state is worse than "unknown" — it is "unmeasured in a direction the surrounding evidence discourages."**

**Stated as four separate claims with their evidence grades:**

1. **That `W_q` is more compressible than the MLP matrices: SUPPORTED, qualitatively.** `[A]`, ASVD §4.5,
   restated three times in that paper, corroborated in direction by SVD-LLM's choice to analyse `W_Q`/`W_K`
   spectra. **No number.**
2. **That `W_o` can be low-ranked at all: NO EVIDENCE EITHER WAY.** `[X]`. **Two independent papers'
   per-matrix analyses both omit it.** It is 50 % of the lever. **This alone prevents the ~88 B row from
   being called evidenced.**
3. **That 17.6 % of bytes is reachable: NOT SUPPORTED, and the nearest published points are bad.** The
   only training-free measurement near that regime is SVD-LLM (W) at 80 % compression: downstream average
   **0.04** vs **0.57** `[T]`. Even the fine-tuned variant is at **0.16** with below-chance cells. The
   whole-model-equivalence argument (§3.4 Reading B) that would rescue it is **arithmetically valid and
   mechanistically irrelevant**, and I am not going to let it stand as support.
4. **That low-rank composes with ternary: CONTRADICTED at milder settings than ours, twice, by two
   independent groups.** `[T]` ASVD Table 4 (`59.84 → 427.59` at 0.85 + NF4); `[T]` SVD-LLM §A.11.
   **We would be at a more aggressive rank AND a more aggressive bit-width than either.**

**The measured cost, to the extent one can be quoted at all:**

> **There is no published measured cost for our operation.** The closest quotable figures, with their
> scope attached: **`W_q`/`W_o`-equivalent whole-model compression of ~13 % costs roughly `5.68 → 7.94`
> WikiText-2 and `0.57 → 0.49` downstream average, training-free, on LLaMA-7B, with the compression
> spread over all matrices rather than concentrated** `[T, SVD-LLM Table 1, 20 % row]`. **That is an
> upper bound on the cost only if concentration is free, and §3.4 argues it is not.**

---

## 8. What in the Adapter's framing I think is wrong

Asked for directly, and answered directly. **Two of these make the picture worse, one makes it better.**

### 8.1 The `x_proj` analogy is weaker than "different operation" — the rank fractions are not comparable

The Adapter says the in-house result is *"a different operation"* because the donor is pretrained. **True,
and there is a second difference that has not been named:** `r = 26` at our width and "17.6 % of bytes"
at donor width are **not the same rank fraction.** By §1.1, 17.6 % of bytes at `d = 8192` is `r ≈ 721`,
i.e. `r/d ≈ 8.8 %`; the in-house `r = 26` at `d ≈ 1536` is `r/d ≈ 1.7 %`. **The byte fraction was carried
across; the rank fraction changed by ~5×.** Whether the relevant invariant is bytes or rank is unknown,
and **the memo assumed bytes without saying so.** This cuts *in our favour* on rank and is exactly the
kind of silent assumption the project's own rules say to surface.

### 8.2 ⚠ The ~26 B row is not safe either — it inherits an assumption the memo does not flag

The memo presents **~26 B** as the conservative fallback: *"the honest ceiling … with attention
projections left at full width."* **But the ~26 B row assumes the FFN reaches probe-4 granularity
(3.57 % active) at donor width**, and that is transplanted from `D = 1536` in exactly the way §2.1 of the
memo flags for the rate constant. **If the D5 result the coordinator reports holds — the rate constant
roughly halving at donor width — then by the memo's own model `tok/s = rate ÷ bytes`, both rows halve
together: ~26 B becomes ~13 B and ~88 B becomes ~44 B.** The ratio between them is preserved; the
absolute targets are not. **The memo's headline "a ~26 B target closes on evidenced levers" is
conditional on a constant that is now reported to be moving, and it is stated unconditionally.**

### 8.3 The arithmetic itself checks out — I could not find an error in it

Reconstructing from the memo's own inputs (5-trit = 0.2 bytes/weight, 21.25 GB/s, 20 tok/s, FFN at
3.57 % of 705 M, `L` scaled to hold total parameters):

```
mine:  byte budget at 20 tok/s = 21.25 / 20            = 1.0625 GB/token
       active weights allowed  = 1.0625 / 0.2          = 5.31 B
       full-width attn:  per layer active = 151.0 + 0.0357×705 = 176.2 M   ✓ matches memo's 176.2 M
       layers affordable       = 5.31e9 / 176.2e6      ≈ 30.1
       donor size              = 30.1 × 856 M          ≈ 25.8 B            ✓ matches memo's ~26 B
       low-ranked attn: 0.176×134.2 + 8.4 + 8.4 + 0.0357×705 = 51.7 M      ✓ matches memo's 51.7 M
       layers affordable       = 5.31e9 / 51.7e6       ≈ 102.7
       donor size              = 102.7 × 856 M         ≈ 87.9 B            ✓ matches memo's ~88 B
```

**Both rows reproduce. The arithmetic is right.** What is wrong is not the arithmetic but the **input**:
the 17.6 % is an in-house number from a different operation at a different width, and §7 finds no
published measurement that supports transplanting it.

### 8.4 A framing point: the lever is not binary, and the memo's two-row table makes it look binary

The memo presents "full" vs "17.6 %". **The literature's curves are smooth and steep**, and the
interesting question is not whether 17.6 % works but **where on the curve the byte budget and the quality
budget cross.** By §8.3's arithmetic the donor size scales roughly as `1 / (0.5·f·134.2 + 42.0)` in
millions, where `f` is the retained byte fraction on `W_q`/`W_o` — so `f = 0.5` still gives **~57 B** and
`f = 0.35` gives **~68 B**. **A far less aggressive low-rank than 17.6 % still moves the target
substantially**, and those milder points are much closer to what SVD-LLM (W) actually measured. **The
memo's own framing has made the lever look more all-or-nothing than its arithmetic requires**, and that
is the most useful thing in this file for what to pre-register next.

### 8.5 What I could not check

`ADAPTER_MEMO_01`'s FFN-at-3.57 % assumption, the 21.25 GB/s constant, and the D5 result are all outside
this pass's scope and outside this machine's constraints. **§8.2 and §8.3 take them as given and would
change if they change.**

---

## 9. Single-source and unverified claims — full list

- **ASVD Tables 2, 3, 4, A.6 and SVD-LLM Table 1** — parsed once each from arXiv HTML raw table markup,
  transcribed before any arithmetic. **Not re-verified against the PDFs.**
- **ASVD §4.5 per-type finding** — `[A]`, text describing Figure 6. **Figure not viewed. No numbers
  exist.** Every per-matrix numeric claim in this file is `[X]`.
- **SVD-LLM §A.4 spectrum** — `[A]`, figure not viewed, and it is the spectrum of `WS`, not `W`.
- **FWSVD** — portrayed by two competitors with wildly different numbers; own paper unread.
- **Amsel et al. 2407.16153** — abstract read verbatim; **tables not read**; the long-context claim is
  the authors' own **conjecture**.
- **LatentLLM** — abstract only; everything else `[X]`.
- **The `r ≈ 721`, the 12.9 % whole-model equivalence, the ~26 B / ~88 B reproduction in §8.3, and the
  `f = 0.5 → ~57 B` figures** — **mine**, computed here from `ADAPTER_MEMO_01`'s own inputs. Shown in
  full so they can be checked. **Not from any paper.**
- **The claim that no paper measures a single matrix type driven to ~82 % compression with everything
  else dense** — this is an **absence over the papers read in this pass** (ASVD, SVD-LLM, plus abstracts
  of five others). **It is not a claim about the whole literature**, and AdaSVD / ERC-SVD /
  Layer-wise-Dynamic-Rank / LatentLLM / COMPOT remain unread.
