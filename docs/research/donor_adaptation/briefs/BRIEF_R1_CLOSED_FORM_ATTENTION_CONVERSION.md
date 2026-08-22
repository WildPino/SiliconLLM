# Brief R1 — closed-form conversion of softmax attention to a fixed-state recurrent operator

**Author:** the Adapter / Principal · **Date:** 2026-08-22 · **Status:** dispatched to the Researcher
**Owner directive, 2026-08-22:** pursue the 1.5B demonstration as the deliverable **and this as the
research question, in parallel** — *"ciò che mi interessa di più è proprio (a), quindi impegno e metodo."*
This brief gets the method, not a gesture.

---

## 1. The question, and why it is bigger than it looks

The programme needs attention removed from ~95% of a donor's layers. `ADAPTER_MEMO_01` §2.2b prices the
only published route that retains quality (MOHAWK) at **~10,250 T4-hours for a 100B donor** — out of
budget by ~65×. Option (a) is to do it **with no training at all**.

**The connection nobody in our documents has made explicit, and which reframes this whole line:**

Stage 0 found that the wall at 128K context is not the weights but the **KV read path** — attention
re-reads the entire KV cache every token, costing 368 ms/token in fp16 on a 30B-class donor, 3.7× the
whole budget. We have been treating that as a *separate* problem from the density problem.

**It is the same problem, and linearised attention solves both at once.** A softmax attention keeps a KV
cache that grows as `O(T)`. A linear attention — one where `softmax(QKᵀ)V` is replaced by a kernel feature
map so the operation becomes associative — keeps a **fixed-size recurrent state of `d_k × d_v`,
independent of sequence length**. Per token, `O(1)` state read instead of `O(T)` KV re-read.

> **So a successful closed-form linearisation does not merely make attention cheaper. It deletes the KV
> traffic wall entirely, and it produces exactly the recurrent operator our C engine already has a scan
> for.** That is why (a) is worth real effort rather than being the fallback when training is
> unaffordable.

> ### ⚠ CORRECTION (2026-08-22), returned by the Researcher against this brief's own request to be corrected
>
> **The framing above is half right, and the wrong half matters.** The fixed-state property is confirmed —
> every method checked genuinely gives `O(1)` per-token state, so the KV wall really does fall. But I
> wrote as though that came for free.
>
> **It does not. Based's own mechanism shows that the fixed-state property which deletes the KV wall is
> mechanistically THE SAME property that caps recall capacity.** Recall scales with state size; a state of
> `d_k × d_v` has a bounded associative-memory capacity, full stop. **This is architectural, not a
> training-recipe problem that a better fit would resolve.** No amount of cleverness in how we solve for
> the feature map buys back recall that a fixed state cannot hold.
>
> The consequence is that **the hybrid — keeping a few full-softmax layers — is not a quality nicety, it
> is the architectural answer to the recall cap.** Our ~5% attention budget stops being a concession we
> make for speed and becomes a load-bearing part of the design. Any plan of ours that proposes *pure*
> linearisation is proposing to give up retrieval, and must say so.

**The research question, stated so it can only have one answer:**

> What is the cheapest published path from a softmax-attention donor to a **fixed-size-state recurrent
> operator**, what does it cost in quality, and how much (if any) training does it require — read from
> each paper's own tables?

## 2. What to search — this is a real, active literature, not a blank field

Do not treat this as unexplored. Cover at minimum, and follow citations outward:

**Linearising a pretrained transformer (the direct hits):**
`Hedgehog` (Zhang et al., learnable feature maps that mimic softmax attention weights) · `LoLCATs`
(low-rank linearising, claims very cheap linearisation up to Llama-3.1-405B — **verify that claim
against its own tables, it is the single most decision-relevant number in this brief**) · `SUPRA`
(Scalable UPtraining for Recurrent Attention) · `Mamba-in-the-Llama` / `Mamba-Distill` (Wang et al.,
distilling Llama attention into Mamba, reusing the QKV projections) · `Liger` · `DiJiang` (frequency-domain
kernelisation, claims near-training-free) · `LoRA`-based linearisation variants · anything 2025–2026.

**The kernel/linear-attention substrate they build on:**
`Linear Transformers are Secretly Fast Weight Programmers` · `Performer` (FAVOR+) · `cosFormer` ·
`Based` and `Zoology` (Stanford Hazy Research, on what linear attention *cannot* do) · `RetNet` ·
`Gated Linear Attention (GLA)` · `DeltaNet` / `Parallelizing DeltaNet` · `RWKV`.

**The known failure modes — search for these explicitly, they are where the honest answer lives:**
attention sinks and the softmax denominator; the recall/associative-memory gap (`Zoology`, `Based`
document precisely what fixed-state models lose); long-context degradation; and whether reported
retention is on perplexity only or on recall-heavy benchmarks too.

## 3. What I need from each candidate — the columns of the table

For every method:

1. **Mechanism**, concretely: what replaces `softmax(QKᵀ)V`, what is reused from the donor (are the
   original Q/K/V/O projections kept? that matters enormously for us — reused projections are weights we
   do not retrain), and what is newly introduced.
2. **Is the resulting operator genuinely FIXED-STATE?** `O(1)` state per token, no growing cache. **This
   is a hard filter.** A method that speeds attention up but still stores per-token KV does not solve our
   wall and should be marked as such rather than quietly included.
3. **Training cost**: tokens, hardware, wall-clock — `[T]` from the paper's own table wherever stated.
   Zero, a calibration pass, a LoRA fit, and full continued pretraining are four different answers and
   the difference decides everything. **Convert each to a 2×T4 estimate for a 1.5B and a 100B donor,
   marking the conversion as YOUR estimate with assumptions stated.**
4. **Quality retention** against the *unconverted* donor, with the exact benchmark and the exact baseline
   cell named. **Separately report perplexity-only results from recall/long-context results** — a method
   that holds perplexity and loses retrieval is a specific, known failure mode and it must not be hidden
   inside an average.
5. **Counter-evidence**: what the paper concedes, what later work disputes.

## 4. Two structural questions I want answered, not just surveyed

**(a) Does hybridisation rescue it, and does that agree with our budget?**
Our §2.2 speed budget already allows full attention on **~4 of 80 layers (~5%)**. Much of this literature
reports that keeping a *few* softmax layers recovers most of what pure linearisation loses. **If the
hybrid ratio the literature needs is ≲5%, our two independent constraints agree and that is a major
finding.** If it needs 25%, our budget does not close and we need to know now. Get the actual ratios and
the quality at each, from tables.

**(b) Can the conversion be done per-layer and closed-form?**
Our whole apparatus (D1, D4) is layer-wise reconstruction against a Hessian. **Is there a formulation
where the linear operator's parameters are the closed-form minimiser of a per-layer reconstruction
objective against the donor's own attention outputs** — i.e. fit the feature map so the linearised
attention reproduces the softmax attention's outputs on calibration data, with no gradient training?
Hedgehog trains its feature maps; is there a version that solves for them? If nobody has done this, say
so — **that would be a second verified gap and it sits directly next to the one we already own.**

## 5. Rules — the ones this project has paid for

> **A literature number does not enter a decision until someone has read it IN THE PAPER'S OWN TABLE.**

Not the abstract, not a blog, not a summary — including your own summary of a page you fetched. Fetch the
HTML (`arxiv.org/abs/NNNN` then the arXiv HTML or ar5iv rendering) and read the cells. For every ratio,
name which cell is numerator and which is denominator with its row and column headers. **Transcribe
tables verbatim before computing from them** — an agent on this project mis-paired a 13B baseline row with
a 7B label and declared a correct banked figure wrong; only verbatim transcription caught it.

Tag every number **[T]** (table, with coordinates) / **[A]** (text only) / **[X]** (not found).
**Never substitute a plausible value. "[X] I could not find it" is correct and valued.** Check the
direction of any error you find: this project's three past fabrications were all *favourable* to what was
hoped, so a number that flatters the hypothesis deserves more scrutiny, not less.

**Write incrementally**, method by method, to `docs/research/ATTENTION_LINEARISATION_PRIOR_ART.md`. Two
earlier research passes were killed mid-run and produced nothing because everything was held to the end.

## 6. Deliverable

`docs/research/ATTENTION_LINEARISATION_PRIOR_ART.md`, one section per method, ending with:

- **A ranked table**: method × fixed-state (yes/no) × training cost in tokens × my-estimated T4-hours at
  1.5B and 100B × quality retention × whether the donor's projections are reused.
- **An explicit answer to §4(a)**: the hybrid ratio the literature actually needs.
- **An explicit answer to §4(b)**: does a closed-form per-layer formulation exist, and if not, what is the
  closest thing.
- The list of single-source and unverified claims.
- **Anything where you think my framing is wrong.** In particular: if the fixed-state framing in §1 is
  mistaken — if these methods do not actually give `O(1)` per-token state — **say so immediately and
  prominently**, because a substantial amount of this programme's reasoning would rest on it.

Do not recommend a course of action. Report mechanism, cost, evidence.
