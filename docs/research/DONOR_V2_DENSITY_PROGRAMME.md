# Donor Adaptation v2 — the Density Programme

**Status: research reset, 2026-08-21, on the Owner's direction. Supersedes the v1 conversion programme's
*framing*; its measurements are retained and are the foundation of this one.**
**Branch `research/donor-adaptation`.**

**Owner's directive (2026-08-21):** the v1 approach was fitting a pretrained model to the engine without
disturbing it. The instruction is to disturb it — rethink what a model *is* and how to decompose it,
retrain the L3-resident part from scratch if needed, reinvent the thinking/knowing separation, rebuild the
lookup table. **Training is authorized: 90 GPU-h/week, 2× T4, contained (a 100B in ~2 months is
acceptable).** All layers and all weights may be decomposed and transported by any transformation that
works.

---

## 0. The Owner's premise is arithmetically correct, and it localizes my error

I checked it before accepting it, using **only rates measured in this repo this session** (engine-integrated
ternary LUT at donor dimensions **21.25 GB/s**, measured D=1536; not a literature number, not a projection):

| configuration | active bytes/token | tok/s |
|---|---|---|
| **100B @ 2% active, 4-bit codes (BUILT)** | 1.00 GB | **21.2** |
| 100B @ 3% active, 4-bit codes | 1.50 GB | 14.2 |
| 100B @ 5% active, **trit-pack (designed, not built)** | 1.00 GB | **21.2** |
| 100B @ 2% active, trit-pack | 0.40 GB | **53.1** |

> **The premise holds: a 100B model at ~2% activation reaches >20 tok/s on the packing that already
> exists.** This is not a claim about quality — only about the byte path.

**And the architecture was designed for exactly that.** The S2 ladder rung is 206M total with 3.1 MB/token
active = **3.01% activation**. The keystone, the two-pool split, the ρ-law granularity rule and the LUT
kernel are all built around a model that touches a few percent of itself per token.

**What I was actually running:**

| | activation |
|---|---|
| architecture design point | ~1.5–3% |
| `Qwen2.5-1.5B`, dense | **100%** |
| **ratio** | **33–66× too dense** |

> **I ran a 100%-activation model on an architecture designed for 3%, and then reported that the
> architecture was the constraint. The constraint was the model's density, and I never attacked it.**

**The specific failure, named:** the engine's speed is `active_params × bytes_per_weight`, and there are
three multiplicative levers — **activation fraction**, **bits per weight**, **total size**. The v1
programme took total size as the donor's choice, attacked bits-per-weight partially, and **never touched
activation fraction at all**. I optimized one lever of three, then eliminated 17 candidates on the value of
the lever I had not touched. The Owner's three questions —

> *"posso artificialmente introdurre sparsità?" / "quanti di questi pesi servono veramente?" /
> "sono matrici, come le mando in zero e che basi servono?"*

— are each an attack on that lever, and none of them appears anywhere in the v1 programme.

---

## 1. The reframe: the target is an activation fraction, not a model size

**The v2 research question, stated once, precisely:**

> **Given a pretrained model of N parameters, produce an operator-equivalent-enough model that touches
> ≤2% of its parameters per token, in bulk-contiguous ≥48 KB reads, at ≤0.5 B/weight, retaining ≥90%
> of the donor's general capability — using ≤720 T4-hours of training.**

Everything else follows. Note what this does to the donor shortlist: **size stops being a filter.** A 100B
donor at 2% activation is *faster* than a 1.5B donor at 100%. The 17 eliminations in
`DONOR_STAGE1_ARITHMETIC.md` were computed at the donor's native density and **are void as eliminations**;
they remain valid only as measurements of the *unmodified* donor.

**The target number:** ≥20 tok/s ⇒ ≤1.06 GB/token ⇒ **≤2.1B active params at 0.5 B/w**, or **≤5.3B at
0.2 B/w** if the trit-pack is built. For a 100B donor that is **2.1%**; for a 30B donor, 7%; for a 7B
donor, 30%.

---

## 1b. CORRECTION — "size stops being a filter" is half right, and the wrong half is load-bearing

The prior-art survey (`DENSITY_PRIOR_ART.md`, 77 claims, 55 read from the papers' own tables) refutes §1's
reframe on the axis I did not cost.

**I priced the inference budget and never priced the training budget.** Using this document's own `6ND`
and 25 TFLOPS, `720 T4-h = 6.48e19 FLOPs`. Against the published recipes that actually achieve high
sparsity:

| recipe | tokens needed | × over our budget |
|---|---|---|
| ProSparse-7B (**cheapest** working setting) | 34.6B | **22.4×** |
| Turbo Sparse | — | **97×** |
| LLaMA-MoE | — | **124×** |
| **a 100B donor** | — | **~320×** |

> **Break-even donor size for the cheapest working high-sparsity recipe: 0.31–1.48B parameters.**

**So: "size stops being a filter" is TRUE for the byte path and FALSE for the training path.** A 100B
donor is affordable to *run* and unaffordable to *sparsify*. Size did not stop being a filter — **it moved
from the inference budget to the training budget, and I did not follow it.** That is the same class of
error as v1's (optimizing one lever and reporting on another), committed one level up.

**This must be resolved before D4 spends a GPU session.** The resolution is not more budget; it is to stop
trying to sparsify the donor at all — see §4b.

## 2. What the training budget actually buys — the feasibility number that unlocks the design

At 90 T4-h/week, 25 TFLOPS effective, `6ND` scaling:

| budget | core size | tokens achievable | Chinchilla-optimal |
|---|---|---|---|
| 4 weeks (360 T4-h) | 0.2B | 27B | 4B |
| 4 weeks | **0.5B** | **10.8B** | **10B** |
| 8 weeks (720 T4-h) | 0.5B | 21.6B | 10B |
| 8 weeks | 1.0B | 10.8B | 20B |

> **A ~0.5B cache-resident core can be trained to Chinchilla-optimal in 4 weeks, or to 2× optimal in 8.**

This is the fact that makes §4's architecture possible, and it was never computed in v1 because v1 forbade
training. **It changes what is designable.**

---

## 3. The four attacks on density — each with the question it must answer

These are the axes. None is chosen yet; §6 defines how they are screened. Ordered by expected leverage
per unit of risk.

### A. Conditional computation — route instead of compute
Split a dense FFN into `E` experts, activate `k`. Activation becomes `k/E`; **2% is top-2-of-100**, which
is beyond current released MoE granularity but not beyond the engine's measured design point (E32×h128
top-8 = 25%, and probe-4 measured that *granular beats coarse at matched params*).
**The unanswered question, cheap and never measured: do a dense FFN's neurons actually cluster by
co-activation?** If they do, clustering is free and only the router needs training. If they do not, this
axis dies early and cheaply. **Measure before designing.**

### B. Induced activation sparsity — make it zero, then skip
The engine's skip path is exact, not approximate: gate-first, then row-skip. The project measured **92%
hidden sparsity at ~zero quality cost** with dReLU — but *trained in*. ReLUfication of a SwiGLU donor is
published and needs continued pretraining. **v1 excluded it because S1 forbade training. Training is now
authorized, so it re-enters as a primary candidate.**
Composes multiplicatively with A: `activation = (k/E) × (1 − sparsity)`.

### C. Structural compression — fewer weights, period
**The project already owns a positive result here and never applied it to a donor:** the Inventor's
`x_proj` low-rank **r=26 beat the dense baseline on 3/3 seeds (≈3.5σ) at 17.6% of the bytes**. A quality
*gain* at 5.7× fewer bytes. Candidates: truncated SVD per organ, Monarch/butterfly factorization
(sub-quadratic, with a projection algorithm), Kronecker/tensor decompositions.
Reduces bytes whether or not activation improves — so it is the safest axis and the natural baseline.

### D. Basis change — the Owner's third question, and the deepest one
> *"sono matrici, come le mando in zero e che basi servono?"*

**A weight matrix is dense only in the basis it happens to be stored in.** `W = UΣVᵀ` is diagonal —
maximally sparse — in its own singular basis; the cost is `U` and `V`. **The escape is to constrain `U`
and `V` to be structured and therefore nearly free**: Hadamard, permutation, butterfly. That is precisely
what Monarch factorization is (`W ≈ P₁B₁P₂B₂`, `B` block-diagonal), and unlike a generic change of basis
it has a tractable projection from a given dense `W`.
Distinct from QuaRot-style rotation, which rotates for *quantizability* (killing outliers). Here we rotate
for *sparsity* (concentrating energy). **Whether a rotation exists that makes a trained LLM's weights
sparse is, as far as I know, open** — and it is the highest-originality item in this document.

---

## 4. The architectural proposal: freeze the knowing, retrain the thinking

This is the direct answer to *"se necessario addestriamo da zero la parte in L3"*, and I believe it is the
strongest design available.

**Observation.** A pretrained LLM entangles two things with wildly different costs:
- **knowledge** — expensive, millions of GPU-hours, and concentrated in the FFN weights (the
  FFN-as-key-value-memory literature is the basis for this and must be verified, not assumed);
- **the program that queries it** — comparatively small, and *cheap to train at 0.5B*.

**Proposal.**
- **Knowing tier = the donor's FFN/MLP weights**, frozen, ternarized, MoE-ified, DRAM-streamed in
  ρ-safe contiguous blocks. This is where the 100B lives and where the 2% activation must be achieved.
- **Thinking tier = a NEW ~0.5B ternary SSM core, trained from scratch**, cache-resident, whose job is
  to query the frozen store. This is what the 720 T4-hours buy.

**The central risk, named up front:** the donor's FFN weights are only meaningful in the donor's own
activation basis. A freshly trained core produces different activations, so the frozen weights would be
garbage. **Mitigation, and it is also the training objective:** train the core *with the store frozen*, so
the core is forced to learn the donor's representation. Gradients flow only through the core — which is
exactly why it fits the budget, and why the budget arithmetic in §2 is the load-bearing feasibility check.

**Why this is a better bet than v1's conversion:** v1 spent its effort preserving the donor's *program*
(attention layer by attention layer) — the cheap part — while inheriting the donor's density, the
expensive problem. This inverts that: keep what cost millions of GPU-hours, rebuild what costs hundreds.

---

## 4b. §4 is not a proposal — it is published, and it is inside budget

**The best-evidenced finding in the survey, and it is about my own §4.** MOHAWK / Phi-Mamba:

- **froze the donor's MLP**,
- **discarded the donor's attention entirely**,
- trained a **Mamba-2 core** on **3.0B tokens**,
- scored **62.6 vs teacher 64.9 = 96.5% retention** (their Table 1, read directly).

**3.0B tokens is 0.42×B — comfortably inside our 720 T4-h.** Mamba-in-Llama corroborates the same shape at
8B (<960 A100-h, MLP frozen). §4's "freeze the knowing, retrain the thinking" is therefore **not
speculative architecture — it is a reproduction target with a published retention number.**

**But read the scope precisely, because it is the whole point:** *this reduces activation by zero.* MOHAWK
makes the architecture **legal**; it does not make it **sparse**. It solves the operator question that v1
spent itself on, and leaves the density question — the one that actually decides tok/s — untouched.

### 4c. The real problem, now crystallized

Combining §1b (we cannot afford to sparsify a donor) with §4b (we can afford to rebuild the core around a
frozen store):

> **Don't sparsify the donor. Keep the donor's FFN as a frozen knowledge store, discard its attention, and
> make the STORE sparse by construction rather than by training.**

Discarding donor attention puts the 100B store requirement at **2.4% activation**. And that is where the
programme's single hardest constraint appears:

> **THE CENTRAL OPEN PROBLEM: a sub-2% sparse store with CONTIGUOUS access.**

Both halves are load-bearing and the literature has each one *separately*:
- **Sub-2% stores work** — Memory Layers at Scale is the proof. **But it is a random gather**, which its
  own authors call *"almost entirely memory bandwidth bound"*. Under our ρ-law that is the **14× penalty**,
  which eats the entire density win. A top-k lookup store built without solving contiguity first is dead
  on arrival on this silicon.
- **Contiguity has a known attack** — Neuralink (ASPLOS '25) solves neuron placement as a Hamiltonian path
  and measures **1.80× bandwidth**. Against the **~5.6×** we need, that is the right shape and not enough.

**The arithmetic on ≤2% is not hopeless but nothing has done it:** 12.5% (LLaMA-MoE granularity) × 11%
(ProSparse residual) = **1.4%** — but that stacks two training-expensive recipes. **No published recipe
reaches ≤2% at any cost.** The record is TurboSparse-Mixtral at **9.1%** (4.3B/47B); DeepSeek-V3 reaches
5.5% *trained from scratch*. Training-free sparsity **hard-caps near 50%** (TEAL) and everything degrades
by 65%.

**The one structural idea the survey makes obvious, and which the engine already implements:** contiguity
is solved *by construction* if routing happens over **blocks** rather than neurons. At 0.5 B/w a
LLaMA-7B neuron is a **2 KB run**, so a 48 KB ρ-safe read is **24 consecutive co-activating neurons** (12
at 100B scale). **An "expert" of ≥24 neurons is exactly that read** — and granular MoE with correctly-sized
experts is the engine's existing design (E32×h128, probe-4-validated). **So the store should be
block-routed, not neuron-gathered**, and the open question becomes whether block-routing can reach 2%
while retaining quality — not whether gathers can be made fast, which the ρ-law already answers.

### 4d. Two corrections to the literature the survey caught

- **ProSparse's "free sparsity" is not free.** Against its own *matched-token* control it is
  41.40 → 38.46 = **92.9% retention**, not the 37.96 → 38.46 that the uncontrolled comparison suggests.
- **Turbo Sparse has no matched-token control at scale**, so its "beats the original" gains are
  **unattributed**. Do not quote them as evidence that sparsification is quality-free.

## 5. Constraint changes recorded

| constraint | v1 | v2 |
|---|---|---|
| **S1 — transformation-only** | training forbidden as the primary path | **AMENDED by the Owner: 90 GPU-h/week, 2× T4, contained.** Axes B and §4 depend on this and were unavailable in v1. |
| **S2 — two SKUs** | unchanged | unchanged |
| **S3 — attention on a minority** | unchanged | unchanged; §4 makes it nearly moot (a from-scratch core need not carry the donor's attention at all) |
| **S4 — ≥90% retention, ≥10 tok/s** | unchanged | unchanged, and the working target is **>20 tok/s at 100B** |

**What does NOT change:** every working law in §6 of the mandate. Pre-register before measuring; one
variable per stage; planted controls that must be shown to fire; the artefact is the authority; parity
end-to-end; no `-ffast-math`; deflate your own claims first. The Owner's requirement is
*"maniacalmente metodologico e preciso … non dare nulla per scontato"* — this reset changes the research
question, **not** the standard of evidence.

---

## 5b. What `README.md` and `HANDOFF.md` add — read late, and they change the programme

**I had not read either.** §12 of the v1 mandate pointed at `HANDOFF.md` as the narrative history and I
skipped it. Five things follow, and the first is a correction against myself.

### 5b.1 The repo already had the PT²-LLM number right, and I contradicted it

`README.md` §1 states, in a footnote that predates this whole programme:
> *"Post-training ternarization does not follow this trend (PT²-LLM: ~2× PPL at 7B–70B), supporting the
> QAT-from-scratch choice."*

So the v1 mandate's "~2×" was **sourced from the project's own correctly-banked figure**, and my
"§8.C's ~2× is unsourced" was not merely wrong on the literature — **it contradicted a number this repo
had already got right, in its most public document.** Had I read the README first, the fabricated
"near-lossless" claim would have been caught on contact instead of after a full verification cycle.
Recorded in [[feedback_literature_fabrication]]: **check the repo's own banked figure before challenging
a mandate's number.**

### 5b.2 The block-structuring regularizer — the missing bridge for axis B, already measured

Phase 58.B's *pre-registered gate failed* (a predictability regularizer did not raise predictability), but
its byproduct is directly load-bearing here:

> **A temporal-coherence regularizer made the sparsity BLOCK-STRUCTURED at zero quality cost:
> block-skippable @BS8 went 18% → 50%, and stayed nonzero even @BS32 (0.7% → 17%) — which scattered
> sparsity cannot produce.**

**This is the bridge axis B needs.** The ρ-law means the engine cannot exploit scattered sparsity at all:
byte reduction only counts if it is contiguous and bulk-loadable. So "induce sparsity" is not sufficient —
it must be induced *in blocks*. **The project has already measured a training-time term that does exactly
that, at zero quality cost, and v1 could not use it because S1 forbade training.** It moves to the front
of axis B.

### 5b.3 The training pipeline for §4's core already exists and is validated

The MVE pilot (README §8) ran the full A→F curriculum end-to-end on **2× T4** with **DDP 1.80×**,
**3860 tok/s**, and **resume-from-preemption across multiple sessions, zero non-finite steps in 60,000**.
§4 proposed training a ~0.5B core; **it does not need new training infrastructure**, which materially
de-risks the budget arithmetic in §2 — that was a plan assuming a pipeline, and the pipeline is built.

**And one trap to carry forward, because T4s are fp16-only (no bf16):** the pilot logged a deadlock where
*"a forward overflow produces NaN loss → the grad scaler skips the step → the weights never change → the
identical forward overflows again"*, which **sat for 19,973 of 20,000 steps and then printed a success
banner.** Non-finite losses are now a fatal condition. Any v2 training run inherits that guard.

### 5b.4 Two laws that will kill most density schemes — apply them before designing, not after

- **The ρ-law, stated in its strongest form (probe-3):** *"no unnecessary random gather — cache-residency /
  sequential access is the feasibility gate on every byte-reduction."* **Unstructured sparsity is worth
  exactly nothing on this engine.** Every candidate in §3 must be scored on contiguity, not just on count.
- **The microbench-does-not-compose law (Phase 61):** a **2.6× GEMV microbench became −4% in-engine.** So
  D0–D3 in §6 measure *properties*, and no property is promoted to a speed claim without an
  engine-integrated measurement. This session already produced a second instance of the same law in the
  opposite direction (a "compute-bound" reading at D=256 that was bandwidth-bound at D=1536).

### 5b.5 Temporal persistence is NULL — measured twice, and it constrains §4

The active set **reshuffles every token**: `P(active_t | active_{t−1}) ≈ base rate`, confirmed at both
neuron and expert granularity. **There is no hot pool and there never will be.** For §4's frozen-store
design this means the store is genuinely stream-per-token, and the core cannot be designed around
"the last token's experts are probably still warm". Locality can come only from prediction — and
ahead-of-time prediction was measured **weak (50–63%)**, against **86–92% in-place**. So the durable
mechanism remains **skip, not prefetch**, and §4's core must decide what it needs *at the point of use*.

### 5b.6 One organ is known-hard, with a number

Phase 61 measured the SSM projections as **precision-hungry control organs** (+0.018–0.022 BPB to
ternarize, rejected), corroborated by the published SSM-quantization literature (Quamba, MambaQuant).
Note also the banked observation that **no published work ternarizes SSM projections at all**. If §4's
from-scratch core is an SSM, its projections are a known-expensive organ and should be **budgeted at fp32
from the start** rather than discovered to be expensive later.

## 6. What happens next, and in what order

**The discipline that v1 got right and must be kept: ask the cheapest question that can kill the idea, first.**

| # | question | cost | kills what |
|---|---|---|---|
| **D0** | Do a dense FFN's neurons cluster by co-activation, at donor scale? | CPU, hours | axis A |
| **D1** | How many weights are actually needed? Per-organ, per-layer sensitivity to magnitude/structured pruning, **measured with our own instrument** | CPU/GPU, days | sets the ceiling for all axes |
| **D2** | Does a rotation exist that makes trained weights sparse? Test on one real layer before any theory | CPU, days | axis D |
| **D3** | Does low-rank transfer from our 8.3M result to a donor organ? | CPU, days | axis C |
| **D4** | Can a small core learn to drive a frozen donor FFN at all? Smallest possible pilot | 1 T4 session | §4's architecture |
| **D5** | **Is the donor's induced sparsity CONTIGUOUS enough to skip in bulk?** Measure block-skippable fraction at BS8/BS16/BS32 on whatever D0–D2 produce, against the ρ-law's 48 KB granularity | CPU, hours | **every axis** — per §5b.4 scattered sparsity is worth zero |

**D5 is not optional and it is not last.** The ρ-law makes contiguity the feasibility gate, so a density
result that has not been scored for block structure is not yet a result. Phase 58.B's coherence
regularizer (§5b.2) is the known remedy if D5 comes back scattered — and it is a *training* term, so it
belongs in the same budget as §4.

**Note on ordering discipline, from `HANDOFF.md`'s own "What Worked" list:** *"Premise probes before
spending training (minutes, not hours)"* and *"Three-level evaluation (gate → replicas → human read):
every level caught what the one below missed."* Both apply here. D0–D3 and D5 are premise probes; none
justifies a GPU session on its own.

**D0–D3 are all CPU-class and cost no GPU session.** D4 is the first thing that needs the Owner's GPU
budget, and it should not be launched until D0–D3 have said which axis it is testing.

**Nothing is pre-registered yet.** The pre-registrations follow the same rule as v1 — written and pushed
before the run they govern, gates fixed before the numbers exist, planted controls demonstrated to fire.
The v1 programme's four Controller reviews caught a fabricated literature number, a gameable gate, a dead
control and an inference that charged a conversion its benefits and none of its weights. **That machinery
stays on.**
