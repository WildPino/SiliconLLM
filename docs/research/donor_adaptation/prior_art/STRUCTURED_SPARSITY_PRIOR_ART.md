# Structured / block-contiguous activation sparsity — prior art, integrated

**Compiled by:** the Adapter / Principal, from a Researcher pass that read each paper against its own tables
**Date:** 2026-08-22 · **Branch:** `research/donor-adaptation`
**Tagging:** `[T]` read in the paper's own numbered table (table + row + column given) · `[A]` text only · `[X]` searched for, not found, **no value substituted**
**Scope note:** this covers MoE-ification and memory-layout work. The activation-function / thresholding branch (ReLUfication, ProSparse, Turbo Sparse, Q-Sparse, TEAL, CATS) is a separate pass landing in `ACTIVATION_SPARSITY_PRIOR_ART.md`.

---

## 0. The three findings that change what we do

1. **MoEfication physically permutes the weight matrix so that each expert becomes a contiguous submatrix.** It is not index-gather with a mask. This is precisely the mechanism our engine needs, it is mathematically lossless, and its core result requires **no fine-tuning at all**.
2. **Its granularity is comfortably engine-legal at donor width — with 8× headroom.** MoEfication uses 32-neuron experts. Our own D0b measured the ρ-floor at donor width as **4 neurons**. We are not fighting for granularity; we have room to spare.
3. **⚠ There is a negative prior against co-activation grouping, and one of our own planned probes (D0) is exactly that.** The literature is **split and unreconciled** — see §5, including a correction to my first count of it. It still changes how D0 must be run.

---

## 1. MoEfication — arXiv:2110.01786 (ACL Findings 2022)

**The mechanism we care about, and it is the right one.** MoEfication constructs a permutation matrix `P` and *physically reorders* the FFN rows and columns so that neurons assigned to the same expert become adjacent in memory. Verbatim from §3.2 `[A]`: `W̄₁ = W₁P`, with the down-projection permuted correspondingly, and the output shown identical: `σ(h)W₂+b₂ = σ(hP)PᵀW₂+b₂`. **After the permutation each expert is a contiguous submatrix, not a scattered index set.**

That distinction is the whole ballgame for us. A gather-based expert buys FLOPs on a GPU and nothing on our engine; a permuted contiguous expert buys memory traffic, which is the only currency we spend.

**Granularity** `[A]`, §4.1: `d_e = 32` neurons per expert, giving 64–512 experts across T5 variants.

**Measured CPU speedup — Table 3** `[T]`, on an Intel Broadwell CPU `[A]`:

| activated ratio | FLOPs | **CPU** | GPU |
|---|---|---|---|
| 50.0% | 1.50 | **1.43** | 1.15 |
| 25.0% | 2.00 | **1.98** | 1.20 |
| 12.5% | 2.40 | **2.28** | 1.47 |

**Read the CPU column against the GPU column.** At 12.5% activation the CPU gets 2.28× — nearly the full 2.40× FLOP bound — while the GPU gets 1.47×. That asymmetry is our thesis stated by someone else's hardware: **conditional computation pays on a memory-bound CPU and largely does not on a FLOP-rich GPU.** It is the strongest external corroboration of the project's premise found so far, and it is a measured wall-clock number, not a projection.

**Quality — Table 4** `[T]` (T5-Large, average over GLUE + RACE + SQuAD1.1):

| | Avg. | vs original |
|---|---|---|
| Original (dense) | 87.2 | — |
| MoEfied (trained MLP router) | 85.7 | **−1.5** |
| + Groundtruth router (oracle) | 86.9 | −0.3 |
| + Calibration | 86.9 | **−0.3** |

**Training cost.** The main results use **no fine-tuning at all**. Optional calibration touches only `W₂`/`b₂` at lr 1e-7 `[A]`; no token count or GPU budget is stated — `[X]`. Router training `[A]`: ~500k sampled input representations, *"several minutes on a single GPU"* per FFN.

> **A −0.3 average penalty at 12.5% activation, with no gradient training of the backbone, on a mechanism that produces contiguous experts.** That is the single most encouraging datum in this programme. Its caveats are equally load-bearing and are stated in §5.

## 2. LLaMA-MoE — arXiv:2406.16554 (EMNLP 2024)

**A result on grouping — but NOT the one I first reported it as.** Four construction methods compared; the winner was **IndependentRandom** — *"randomly partition U into n equal-sized subsets"* — beating **balanced k-means clustering on `W_up` row vectors** `[A]`, §5.5.

> **⚠ CORRECTION (2026-08-22), from the Researcher's primary-source pass.** I first wrote this up as a
> second independent negative against *co-activation* clustering. **It is not.** LLaMA-MoE's
> `IndependentClustering` is **weight-space k-means** — it clusters the rows of `W_up`, i.e. the
> parameters — **not neurons grouped by how often they fire together on real data.** Those are different
> constructions, and MoEfication itself tested them separately (it has a "Parameter Clustering" method
> *and* a "Co-Activation Graph" method). Worse, the Researcher reports LLaMA-MoE gives **no numeric
> results table** for the comparison — `[X]` — so it is an `[A]`-grade statement in running text.
>
> The honest tally against **co-activation** grouping is **one negative (Apple) and one positive
> (Neuralink)**, not two-versus-one. **The claim was overstated in my favour** — it made a prior I found
> interesting look better supported than it is.


**Cost puts it out of budget** `[A]`, §5.3: *"trained on 112 A100 (80G) GPUs with a global batch size of 15M tokens"*, *"expected to be trained on 200B tokens (13.6k steps)"*. Against our 90 T4-h/week this is not a candidate.

**Quality — Table 2** `[T]`: LLaMA-MoE-3.5B (4/16) averages **57.7** against Sheared-LLaMA-2.7B's 56.4 and Open-LLaMA-3B-v2's 55.6. Good, but purchased with 200B tokens.

**LLaMA-MoE-v2** (arXiv:2411.15708) is the cheap variant: **7B tokens** on 32 A100s `[A]` versus v1's 200B — ~28× cheaper. Its Table 1 `[T]` shows what that buys:

| | MMLU | IFEval | HellaSwag |
|---|---|---|---|
| LLaMA3-8B dense (15T tokens) | 67.22 | 76.53 | 78.79 |
| MLP-MoE (8top2), 7B tokens | **37.41** | **32.72** | 58.95 |

**A 30-point MMLU collapse.** This is the clearest available evidence that MoE-ifying a dense donor on a small token budget destroys general-purpose retention — which is exactly the axis constraint **S4** (≥90% retention) protects. **Cheap MoE-ification with a randomly-initialised router is measured, and it fails our gate by a wide margin.** Note the contrast with MoEfication's −0.3: the difference is that MoEfication keeps the original FFN semantics and only learns *which* pre-existing neuron group to fire, while LLaMA-MoE-v2 re-trains toward a new parameterisation.

## 3. Sparse Upcycling — arXiv:2212.05055 (ICLR 2023) — **wrong tool, correctly identified**

Each expert is initialised as **a copy of the entire original MLP** `[A]`; there is no neuron-splitting step. With 32 experts and top-2 routing, the per-token FFN compute is **two full FFN passes**, not a fraction of one. It is a capacity-scaling technique — more total parameters at equal-or-greater per-token cost. **It is the opposite of what we need** and should not appear in our design discussions again.

Its Table 1 `[T]` makes the point: T5-Large dense 783M → sparse **7.22B** parameters. Extra cost, Table 5 `[T]`: 287.48 extra TPUv4-days vs the dense 123.54.

One incidental detail worth keeping `[T]`: only **6/12 or 12/24 layers** are converted to MoE at all; the rest stay dense. An architecture choosing to sparsify half its layers and leave the rest alone is a precedent for our own per-layer heterogeneity.

## 4. Memory-layout work: Neuralink and LLM-in-a-Flash

**Neuralink** (arXiv:2410.19274) is the closest match to what we want: an **offline stage that reorganises neuron placement by co-activation**, greedy pairwise-merge of links until one chain per layer, then online caching. **Training-free and lossless** `[A]`: *"the same output as their dense counterparts for a given input"* — a pure memory relayout. Reports 2.37× over llama.cpp `[A]`.

**LLM in a Flash** (Apple, arXiv:2312.11514) ships **row-column bundling** — storing the up-projection row and down-projection column *for the same neuron index* together, doubling the chunk from `d_model × bytes` to `2 × d_model × bytes` `[A]`. Fixed, always-available, data-independent. Its Table 2 `[T]` isolates the contribution: adding bundling to the full method takes I/O latency from 164 ms to **87 ms** and throughput from 1.25 to **2.25 GB/s**.

**Note for our own layout:** our 3-organ interleave (gate/up/down for one neuron stored together) is the same idea taken one organ further than Apple's two. D0b confirmed it survives donor shapes because `d_ffn` cancels out. We arrived at it independently; it is good to know it is a published, measured win elsewhere.

## 5. ⚠ The contradiction, stated honestly and NOT resolved

**LLM-in-a-Flash tried co-activation bundling and reports that it failed.** Verbatim `[A]`: *"Unfortunately, this resulted in loading highly active neurons multiple times and the bundling worked against our original intention."* They abandoned it; it does not appear in their final method.

**Neuralink, later, reports that essentially the same idea works.** The Researcher could not find Neuralink citing or addressing Apple's negative result — `[X]` for any reconciliation in either paper's text.

**And LLaMA-MoE independently found random partitioning beat co-activation clustering.**

**LLaMA-MoE does not count here** (see the correction in §2: its losing arm was weight-space k-means, not co-activation, and it is untabulated). So the honest tally is **one negative, one positive**. The Researcher confirmed from the primary sources that **Neuralink does cite LLM-in-a-Flash but never engages with or explains its negative bundling result** — the contradiction is real and **unreconciled in the literature**.

### What this does to our plan

`DONOR_V2_DENSITY_PROGRAMME.md` §6 lists **D0 as co-activation clustering of FFN neurons, with a mandatory random-clustering control.** That control was written in as a matter of routine discipline. It is now the most important part of the probe:

> **D0's random-clustering control is not a formality — one published attempt at co-activation bundling failed outright, and the one success cites that failure without explaining it.** The literature is **split, not stacked against us**. If our co-activation arm shows an advantage we are resolving an open contradiction, which is a stronger result than I first framed it. If it shows none, we have replicated Apple.

Apple's stated failure mode is also a concrete, checkable hypothesis rather than a vague warning: *highly active neurons get duplicated across bundles.* **D0 must measure neuron duplication across clusters and report it**, because that is the mechanism by which a co-activation grouping is predicted to lose. A probe that reports only cluster quality and not duplication cannot distinguish "co-activation clustering does not help" from "our implementation hit Apple's failure mode."

I am recording this as a **brief amendment to D0**, to be issued to the Builder when D0 is dispatched.

## 6. The gap — where our originality actually is

The Researcher searched specifically and found **no paper** that simultaneously: (a) targets a **CPU / DRAM-bandwidth-bound** setting rather than GPU FLOPs or smartphone flash, (b) enforces a **fixed block granularity in the tens of KB**, and (c) reports **both** a quality-retention table **and** a quantified contiguous-read-size table.

Neuralink and LLM-in-a-Flash are the two closest, and both are calibrated to **smartphone flash pages and UFS queue depths** — the papers discuss a *"shallow command queue, supporting only 32 entries"* — not to CPU cache and DRAM economics. Neither reports contiguous read size in bytes; Neuralink reports it in abstract "bundle" units `[X]`.

This is consistent with the earlier finding that no published work solves ternary + contiguous-block sparsity *jointly*. **Our originality is not any single mechanism — every one of them exists somewhere. It is the target: a CPU memory hierarchy with a measured 48 KB threshold and a 14× random-versus-sequential penalty, which nobody else is optimising for.**

## 7. What I take forward, and what I do not

**Take forward:**
- MoEfication's **permute-into-contiguity** as the layout mechanism. It is lossless, training-free, and published.
- Its **2.28× CPU at 12.5% activation** as external corroboration that conditional computation pays on memory-bound CPUs and not on GPUs.
- Its **−0.3 average at 12.5%** as the best available prior on what a router costs when the FFN semantics are preserved.
- Apple's **duplication failure mode** as a required measurement in D0.

**Do not take forward:**
- Sparse Upcycling. Wrong direction entirely.
- LLaMA-MoE v1's 200B-token recipe. Out of budget by orders of magnitude.

**Hold as a warning:** LLaMA-MoE-v2's **MMLU 67 → 37 on 7B tokens**. Cheap MoE-ification with a fresh router is measured, and it fails S4 badly. Any plan of ours that resembles it inherits that result until we show otherwise.

## 8. What is NOT established, and must not be quoted as if it were

- **12.5% is not 2%.** MoEfication's best measured operating point is six times denser than our target. Nothing here says the −0.3 penalty holds at 2%.
- **T5-Large is not a 100B donor**, and 2022 encoder-decoder FFNs are not 2025 SwiGLU MLPs.
- **The 2.28× is Intel Broadwell**, not Zen 2, and the paper does not report the memory-traffic accounting behind it — so whether it is the same mechanism as our ρ-law is `[X]`.
- MoEfication's expert-size arithmetic against our 48 KB threshold was computed by the Researcher **from `[A]` values plus public T5 constants**, not read from any table. It is arithmetic, not measurement, and is labelled as such.
