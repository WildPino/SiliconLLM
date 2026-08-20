# Donor-Model Adaptation — the Adapter's first response (§13)

**Status: v2, 2026-08-20. STAGE-0 INTERIM REPORT — accepted by the Architect as such, and explicitly NOT
a basis for choosing the primary route or starting stage −1.**
**Branch `research/donor-adaptation`, base SHA `6906890`.**
**Nothing here has consumed a GPU-hour or a euro.**

**v2 changelog (Architect adjudication + Controller #2 replication):** four of my v1 claims were wrong and
are corrected in place with the correction recorded, not silently edited — §1.4 (branch (iii) over-reach),
§1.5 (Falcon-H1 and the "5–25%" generalization), §2.4 (I missed the KV *traffic* term entirely, and my
conclusion there was backwards), §5.3 (the stage-2a criterion used a non-transferable QAT anchor). The
measured-rate bracket moves from [4.2–17.0] to **[4.35–11.4] GB/s**.

Labels: **[M]** measured in this repo (with protocol and scale), **[D]** derived here from [M] constants,
**[L]** literature, **[L?]** literature single-source and **not yet independently verified**, **[?]** from
memory and pending verification against the artefact. Per law 5 the artefact is the authority; nothing
marked **[?]** or **[L?]** may be quoted onward until upgraded.

---

## 1. What I understood the problem to be

### 1.1 The problem in my own words

The engine is not a runtime, it is a *shape*. It imposes four properties: ternary weights served through
byte-LUT kernels; activation sparsity exploited by skipping, not approximating; a per-token active slice
small enough to stay in aggregate LLC; and every byte path either resident-and-reused or bulk-contiguous
at ~48 KB grain. A donor satisfies none of these.

The mandate asks whether a *class* of donor can be pushed into that shape by mathematics rather than
gradient descent, and whether enough general capability survives. It asks for an adjudicable GO / NO-GO on
a named donor class and SKU against gates sealed before any number existed: **≥90% global retention /
≥80% per critical task**, **≥10 tok/s decode**, and **the complete inventory inside 16 GB (SKU-A) or
64 GB (SKU-B) at a 128K context contract**. All fourteen axes of §8 are instrumental to those three.

### 1.2 What the mandate gets structurally right

The stage ordering. Stage −1 asks the cheapest question that can kill the thesis before a line of engine
generalization is written. I would not reorder it.

### 1.3 What I think it gets wrong — and where I now think the real wall is

My v1 headline was that the mandate over-weights attention→SSM conversion. That survives, but **the reason
has changed, and the change is not in my favour.** The dominant term is one neither the mandate nor either
Controller nor I had priced: **at the sealed 128K contract, reading the KV cache costs more per token than
everything else combined** (§2.4). Ranked by measured leverage:

| intervention | worth | status |
|---|---|---|
| **bound the KV read path at 128K** (SWA / minority attention / recall) | **~350 ms/token** — 3.5× the entire budget | **the wall** |
| ternarize the projections + head | 88 ms/token [Controller #2] | forced |
| pick a donor with ≲1.4–2.3B active params | decides pass/fail outright | forced |
| fix the MoE dispatch defect | ~2.6× on the streamed class | engineering, mechanism now named |
| attention→SSM conversion *as a speed lever* | bounded by the 4.9% scan share | **not the lever** |

Note the tension this creates and that I am not hiding: item 1 and item 5 point in opposite directions.
Bounding the KV path is exactly what an SSM does for free. **The conversion is not justified by per-token
compute; it may be justified by the 128K contract.** That is a different argument from the one §4 makes,
and it is the argument that now needs testing.

### 1.4 Which of §4's three branches I think is right

**Branch (iii) — plausible, the direction of the evidence, but NOT decidable from banked data.**

*(Corrected by the Architect. My v1 said branch (iii) was "already decidable" and "nearly a conclusion".
That was an over-reach and I withdraw it. Recorded rather than silently edited, because it is the
inference my route recommendation leaned on.)*

§2.1 establishes something narrower than I claimed: **the scan recurrence of *this* engine is 4.9% of
*its own* compute floor at t6.** It does not follow that converting a Transformer to an SSM buys only
4.9%. Such a conversion also changes the KV path, the projection shapes and their traffic, the state size,
the layout, the organ inventory and the quality — none of which the 4.9% figure measures. The scan share
bounds *one* term of a change that moves many. §2.4 now shows one of those other terms is ~75× larger.

So §2.1 stands as a correction to the mandate's arithmetic (its 13–15% is inflated ~3× by a partial
denominator) but **not** as a closure of branch (iii).

### 1.5 Route (iv) — authorized as a priority shortlist, not as a confirmed primary route

> **Route (iv) — prefer a donor whose attention is already a structural minority, rather than converting one.**

**Status: Architect-authorized as a priority shortlist only.** It is not the confirmed primary route and
must not be reported as one.

**Two v1 errors corrected:**

1. **Falcon-H1 does not satisfy S3 and I should not have listed it.** It runs attention and Mamba-2
   **in parallel inside every hybrid block**, not attention on a minority of layers. Every block therefore
   carries attention and a KV cache — which, given §2.4, is the worst possible property for this project.
   Removed from the shortlist. *(Architect correction, with source.)*
2. **"Released hybrids sit at 5–25% attention" was an over-generalization** of a survey line and I have
   withdrawn it as a claim. It may be true of some families; it is not a property of "hybrids".

**Concrete elimination already available:** Granite-4.0-H-Small is **32B total / 9B active** — 9B active is
~4–6× outside the §4.2 envelope. It does not pass without further levers. *(Architect correction.)*

**The eligibility filter must therefore be per-model and read from `config.json` + the official
implementation, never from the family name.** Required checks:

- attention a minority **at block level**, not merely at head level;
- a recurrent operator that is actually implementable in the engine;
- **absolute** active params inside the envelope (Controller #2: the binding quantity is absolute active
  params, *not* the sparsity ratio);
- licence and tokenizer compatible;
- real native context and real KV pattern.

Surviving named candidates to test, not to assume: **Nemotron-H**, **Granite-4.0-H** (larger variants
excluded on active params), **Zamba2**, **Jamba**, **Qwen3-Next**. The Builder is applying the filter to
actual configs.

**The residual cost, unchanged:** the donor's recurrent operator will be Mamba-2, GLA or gated-delta-net,
not the engine's Mamba-1 selective scan. The engine must grow it — engineering under §8.J with a
bit-exactness gate, not research with a 600–1500 A100-hour healing budget [L].

---

## 2. What I doubt in this document

Two independent Controller audits were run (§7.1 two-key rule). Where they disagreed with each other or
with me, the disagreement is recorded and adjudicated, not averaged.

### 2.1 [CONFIRMED] §2.2's "~13-15% scan share" is inflated ~3× by a partial denominator

Summing the mandate's own decomposition [D]:

| config | sum of listed organs | scan share | operator-agnostic share |
|---|---|---|---|
| t1, dense | 1283 µs | 13.2% | 79.1% |
| t1, MoE | 1543 µs | 11.0% | 82.6% |
| **t6, MoE (declared operating point)** | **941 µs** | **4.9%** | **87.5%** |

**The mechanism**, found by Controller #1 and re-derived by me: the share is computed against **three of
its own six organs**. `662 + 313 + 169 = 1144 µs` — an exact match for the "≈1.14 ms" printed beside it —
and `169/1144 = 14.8%`, which *is* the "~13-15%".

See §1.4 for what this does and does not license.

### 2.2 [CONFIRMED] "≈1.14 ms compute floor" is the sum of three of six organs

Same mechanism. The mandate is internally inconsistent: §4 says proj-GEMV is "~52%", and `662/1283 = 51.6%`
✔ while `662/1144 = 57.9%` ✘. **§2.2 and §4 use different totals for the same quantity; §4's is correct.**
Corroborated in the repo: `SCALEUP_ARCHITECTURE.md:172` names 1.14 ms as the V-G3 emulation anchor and
`:174` computes 51.6% against 1283 — correctly separate there, glued together here.

Also: the measured whole is 848 tok/s = 1.179 ms, so the t1 parts *exceed* the measured total by 8.8%
(at t6 the sum closes exactly, 1653 vs 1652). The decomposition is a sound apportionment, not an exact
accounting, and should not be summed to a headline at all.

### 2.3 [CONFIRMED, and revised downward again] §5's streaming table prices the LUT path at a rate it cannot reach

§5 divides by **42 GB/s**, the aggregate DRAM cold-stream ceiling. My v1 replaced this with the engine's
[4.2 … 17.0] GB/s bracket. **Controller #2 shows 17.0 is also wrong, and found the datum both
Controller #1 and I missed.**

The **dense** LUT-MLP streams byte-for-byte the same 2,359,296 B/token through the same kernel, at
**7.538 GB/s (t1) / 11.40 GB/s (t6)** [M]. Three consequences:

1. At t1, integrated-dense equals kernel-pure to **1.2%** — so the overhead is specific to the **MoE call
   structure**, not to "being integrated" at all.
2. The dense weight set is 2.25 MiB — **L3-resident, no gather** — and still caps at 11.40 GB/s where the
   same silicon does 185 GB/s on resident fp32 GEMV. **The LUT path is compute-bound by ~16×, not
   bandwidth-bound.** This reframes the entire axis.
3. **17.0 GB/s is the kernel's own compute asymptote, not an achievable integrated target.**

> **Recommended bracket for all donor budgeting: [4.35 … 11.4] GB/s**, replacing both §5's 42 and my v1's 17.0.

**Recomputed eligibility envelope** [D], at 100 ms:

| rate | 100% of budget | 60% of budget |
|---|---|---|
| 4.35 GB/s (MoE path today) | 0.87B active | 0.52B active |
| **11.4 GB/s (dense-LUT, the donor-relevant figure)** | **2.28B active** | **1.37B active** |

**Important:** Controller #2 finds the 4.345 GB/s MoE figure is an **artefact of `D=256, h=128`** — a donor
expert at `D=2048, h=768` is 2.36 MB, 48× larger, and every identified overhead term scales as 1/D.
**4.35 GB/s must not enter any donor budget.** The donor-relevant column is ~11.4 GB/s.

**The mechanism is now named and is a real engineering item:** `mlp_moe` issues **144 OpenMP fork/joins per
token** vs 18 for dense (`OMP_PFOR` sits inside the matvec), plus per-expert serial `quant_i8` +
`build_lut_t3`, plus a layout defect nobody had named — `egate_cd`/`eup_cd` are one `GH×D` block, so
per-expert row windows read **128 chunks of 128 B at 4096 B stride** for two-thirds of every expert's bytes.

### 2.4 [I WAS WRONG — the correction that matters most] KV cache is a per-token *traffic* term, not just a footprint term

**My v1 claim was: "4-bit KV quantization alone brings 128K inside budget without converting a single
layer", and therefore §8.I.4 does more work than §8.B. That conclusion is wrong and I withdraw it.**
I priced KV as *footprint* against the 16 GB budget and never priced the *bandwidth*. Attention re-reads
the entire KV cache every token.

At the sealed 128K contract, 30B-class, GQA-4, 48 attention layers [D]:

| | bytes | read time/token @35 GB/s | vs the 100 ms budget |
|---|---|---|---|
| KV fp16, all layers | 12.9 GB | **368 ms** | **3.7× over** |
| KV 4-bit, all layers | 3.22 GB | 92 ms | 92% of budget — nothing left |
| KV 4-bit, 25% of layers | 0.81 GB | 23 ms | workable |

**The 128K contract alone breaks the 10 tok/s gate for any donor with full attention, before a single
weight is read.** Quantization alone does not fix it — it lands at 92% of budget. **Both** KV
quantization **and** a bounded attention path (minority layers, sliding window, or the recall tier) are
required, and they are not alternatives.

Two consequences I did not see in v1:

- **S3's "minority" is load-bearing for SPEED, not only footprint** — which is a stronger justification
  than the mandate itself gives it, and it partially rehabilitates the operator-boundary axis I demoted.
- **The never-integrated recall tier becomes the load-bearing dependency of the donor programme.** It is
  currently measured standalone only (29.05 µs/token at 128K [M]), and §9 stage 4b is its readiness gate.
  That gate is now on the critical path, not beside it.

*(Credit where due: Controller #2 flagged this as "the biggest thing all three of us missed". It was.)*

### 2.5 [CONFIRMED by both Controllers] D9's fp32 precision map is fatal at donor scale

- **S4** requires ≥10 tok/s = ≤100 ms/token.
- **D9 / Phase 61** measured that ternarizing the SSM *projections* costs **+0.018–0.022 BPB** and
  *rejected* it; projections, SSM and head **stay fp32** [M, 8.3M, QAT].

Computing QKVO from actual transformer tensor shapes (GQA-aware) at the §1b rate curve [D]:

| donor class | QKVO params | fp32 | ternary 0.5 B/w |
|---|---|---|---|
| 8B-class, D=4096, L=32, GQA-8 [?] | 1342M | **5.37 GB → 153 ms** | 0.67 GB → 19 ms |
| 30B-class, D=2048, L=48, GQA-4 [?] | 906M | **3.62 GB → 104 ms** | 0.45 GB → 13 ms |

Controller #2 independently: **fp32 projections + head alone = 3.26 GB = 88 ms of the 100 ms budget.**

**D9 does not transfer to donor scale.** Note what P61 actually measured — weight quantization at 8.3M
under QAT-from-scratch, predating rotation-based PTQ. Whether that +0.018–0.022 BPB transfers to a 1–3B
donor under calibration-only rotation PTQ is **unknown**, and it is now the project's most load-bearing
unknown, because the speed gate has no other solution.

D9 seals the map *for the S-class ladder*; the donor path is a different context, so adopting a different
map is not a violation. But D9's evidence is the only evidence the project owns and it points the wrong way.

### 2.6 [PASSED by Controller #1] the head is under-ranked

| donor class | head params | fp32 ms/token | ternary | freed |
|---|---|---|---|---|
| V=151,936, D=2048 [?] | 311M | **34.1 ms (34%)** | 4.4 ms | 29.7 ms |
| V=128,256, D=4096 [?] | 525M | **57.6 ms (58%)** | 7.5 ms | 50.1 ms |

Controller #1 PASSed the arithmetic (311.3M / 1.245 GB / 29.65 ms). Head compression is an entry
requirement, not an optimization; **§8.H belongs at the front**, not after §8.B and §8.C.

### 2.7 [BLOCK] §8.C's "~2× perplexity" for post-training ternarization is unsourced

The mandate says "roughly ~2× perplexity at 7B–70B in at least one published study. **Assume the naive
path is broken**" — while also, to its credit, saying "verify this figure yourself".

Verified; it fails in both directions:
- **Genuinely naive PTQ is far worse than 2×**: LLaMA-2-7B WikiText2 ~5.47 → **>100 PPL** [L?]. The ~2×
  figure is closer to a *trained* method (OneBit, 1.80× at 13B) — i.e. it appears to have drifted across
  the QAT/PTQ line the same section warns against blurring.
- **The calibration-only ternary frontier claims near-lossless**: PT²-LLM (arXiv 2510.03267) 1.04–1.08× at
  7B/13B/70B, training-free, ~128 calibration samples [L?]; TWLA (arXiv 2606.13054) claims better [L?].

**Both are single-source and unverified.** One extraction pass produced a **fabricated** init-only number
that a second, more careful pass did not confirm — this project's "plausible artefact" law firing inside
my own evidence chain, caught only by re-checking. **No stage −1 method is sealed on these citations
until a second independent read confirms them** (§3.3).

### 2.8 [FLAG] the two named attention→SSM precedents omit the number §4 depends on

Neither Mamba-in-the-Llama nor MOHAWK publishes an **initialization-only, zero-healing quality number**
[L], across five fetch attempts. §4's branch (i) therefore cannot be costed from prior art, only measured —
and my stage-2a random-init control arm (§5.4) measures something absent from the literature.

Confirmed init map, by direct quote: donor `Wq`→SSM `C`, `Wk`→`B`, `Wv`→`X` — **shape-matched, not
provably equivalent**; the paper does not claim equivalence and neither do I. Healing cost ~600–1500
A100-GPU-hours [L] — incompatible with S1 as published.

### 2.9 Controller #2's replication verdicts on the three contested items

**(a) "SKU-B is provably empty" — PARTIAL REFUTE.** Not proven. An 80B/A3B-class donor genuinely needs
SKU-B (~41 GB resident) and misses 10 tok/s by **1.45× at 11.4 GB/s** / **1.14× at 17.0** — a near-miss,
not a proof. **Condition for non-emptiness:** total > 32B **AND** active < ~1.96B/token (at 11.4) **AND**
every fp32 organ ternarized **AND** 128K served from ≤2 MQA layers + recall. The binding quantity is
**absolute active params, not the sparsity ratio.** **No released model above 32B is currently under 1.96B
active** — so SKU-B is *empirically* empty while not *provably* so. That distinction matters: it is a
donor-availability fact, not a law, and it could change with one release.

**(b) The 8.4 µs decomposition — arithmetic CONFIRMED, conclusion REPLACED.** See §2.3. The qualitative
claim ("an engineering lever, not a bandwidth wall") **survives and strengthens** — the mechanism is now
identified. But the reachable rate is ~11.4 GB/s, not 17.0.
**Per-call vs per-byte is not separable** from the two published points (the cross-harness fit gives a
*negative* per-call constant — inadmissible). The settling measurement is cheap and named: sweep `EB` and
`D` in the existing `run_expert_rate`, fit `t = c + b·bytes` per thread count. **I am requesting this as
the next Builder brief** — it is the cheapest measurement that de-risks the largest number.

**(c) Nibble-packing — (b) real but not free.** Packing reading confirmed (`(w0+1)*3+(w1+1)`,
`engine.c:161`, byte-per-pair for `pshufb` lanes). Two codes per byte needs no reordering but costs
**+3 ops on 30 → +10% compute for −50% bytes**. On the now-established **compute-bound** path that is
~10% **slower**. Bytes halve; time does not. **The value is footprint (41 → 21 GB), not speed** — which
matters for §2.9(a)'s SKU-B condition. Controller #1's footprint arithmetic was right; its speed inference
was wrong, inherited from §8.G's bandwidth-framing. The actual speed lever is `acc_add_i8x32`, which burns
**11 of 15 ops** widening int8→int32.

### 2.9b The disagreement I raised — resolved in my favour, and it changes nothing

Controller #1 claimed §8.A's "a 30B sparse-MoE donor beats an 8B dense one" is *reversed* at measured
rates. I objected that it charged the dense donor's MLP to the projection path when in this engine the MLP
goes through the LUT path. **Controller #2: the Principal is right; Controller #1 refuted.** MLP is on the
LUT path in both families (same `bc_tm`, same kernel). 8B dense: MLP 5.637B = 70% of params → 2.819 GB LUT
+ 5.368 GB fp32 attention + 2.101 GB head = **478 ms = 2.09 tok/s**. 30B MoE: **191 ms = 5.24 tok/s**.
§8.A stands and is *understated* — MoE wins on both paths, under every assignment tested including the one
maximally hostile to it.

**But note what this actually says: neither reaches 10 tok/s. The dense 1–100B band is dead**, and the
sparse band is at 5.24 tok/s before the KV term of §2.4 is added.

### 2.9c A stage-ordering defect

§9 stage 1 is "all desk, all free" and eliminates donors permanently — but §5 requires it to measure
kernel and engine rates **at donor-like dimensions**, which needs stage 3's dimension-agnostic engine.
**A stage that eliminates candidates depends on an artefact built two stages later.** Mitigation, which I
think suffices: the synthetic `--kselftest` harness needs only a `-D` recompile to measure kernel rates at
arbitrary dimensions — no model, no loader, no donor. Recommend the mandate say so.

### 2.10 What I do NOT doubt

The ρ-law, the L3 r(size) curve, the falsification of routing locality (three times), σ_seed = 0.005, the
consequence that experts are DRAM-streamed, and the decomposition itself. Controller #1 additionally
PASSed **every §2.4 line reference** (`:51`, `:70`, `:194-240`, `:202`, and the E/hid_e/k-only validation),
**every §12 path**, all four SKU ceilings, and all quality constants' sourcing. My disagreements are about
*which* number to divide by, never about the numbers.

---

## 3. Pre-registration — stage −1 (the PTQ ternary kill gate)

**NOT APPROVED FOR EXECUTION.** The Architect has withheld both the `pip install` and the CPU budget until
the §3.3 method is double-verified and this pre-registration has been Controller-reviewed. What follows is
the pre-registration to be reviewed, not a runnable plan.

### 3.1 Why this shape

Stage −1 asks one question: *is calibration-only ternary PTQ within reach of a usable donor at all?* It
runs before any engine work because a clear failure makes the ternary-primary route moot.

### 3.2 Donor — SEALED

**`Qwen/Qwen2.5-1.5B`**, revision pinned by the hub-resolved SHA, recorded in the manifest. Apache-2.0.
Chosen because (a) permissive licence, (b) small enough to run end-to-end on the reference CPU, (c) it is
the most heavily represented model class in the rotation-PTQ literature, which lets me validate the
apparatus against published 4-bit numbers before trusting any ternary number. That cross-check is the
apparatus's planted control, not a convenience. **One donor, one variable** — no hybrid here.

### 3.3 Method — NAMED SLOT, NOT YET SEALED

Candidate: **PT²-LLM (arXiv 2510.03267)**, alternate **TWLA (arXiv 2606.13054)**.

**No run may start until this slot is sealed, and it cannot be sealed today.** Both citations are [L?] from
a single automated pass in which a fabricated number was already caught (§2.7). Required before sealing:
independent verification of paper, method and reported numbers; then the exact commit/revision, dependency
set and numerical tolerance fixed in writing; then Controller review of this whole pre-registration. If
verification fails, or no member of the family genuinely reaches ternary, that is reported to the Architect
as a stage-0 finding **before** any run — not worked around.

### 3.4 The swept variable — bit-width; the deliverable is a curve

**fp16 baseline → 4-bit → 3-bit → 2-bit → ternary (1.58-bit)**, identical apparatus, calibration and eval.
One variable, five points. A single ternary pass answers PASS/FAIL; the curve answers the question the
Owner will actually ask if it fails — *which bit-width would the engine have to grow to?*

### 3.5 Calibration corpus — SEALED

512 × 2048 tokens ≈ **1.05M tokens**, **globally shuffled, never in file order** (measured law: blocked
sampling order cost +0.0339 BPB = 6.8 σ_seed at identical data — order alone did it). Composition mixed,
not code-only (§8.L.1: a code-only set is a plausible way to silently destroy the general capability S4
measures). Pinned by content hash; manifest committed, **corpus never committed**. Decontaminated against
the eval slice and the pinned P62 code-val.
**Sensitivity arm (§8.L.2):** the ternary point repeated on a **second disjoint calibration set**. If they
differ by more than harness σ, the set is load-bearing and every future number carries its hash.

### 3.6 Metric and threshold — SEALED, fixed before the number exists

Metric: **BPB** on a held-out general-text slice, not perplexity — byte-normalized per the project's
unit-choice law. Absolute donor BPB and absolute converted BPB always both reported.

> **GATE: stage −1 PASSES at a given bit-width iff converted BPB ≤ donor BPB × 1.10.**

**This is a permissive KILL gate and its asymmetry is the point, stated explicitly at the Architect's
instruction:**
- **If it FAILS, the path is very probably dead** — that is the whole informational content.
- **If it PASSES, it says NOTHING about S4 retention.** It is not evidence of ≥90% global retention, not
  evidence of ≥80% per critical task, not evidence of S2 fit, not evidence of anything at donor scale. A
  pass licenses proceeding to the next cheap question and nothing more.

A donor near 0.75 BPB may inflate by +0.075 = **15 σ_seed**. Deliberately loose: a tight gate here would
kill a route that §8.M healing could rescue and that S4 has not yet been asked about.

### 3.7 Planted controls — MANDATORY, both directions, logged (§6.3, §6.4)

1. **Reproduce a known-positive.** At 4-bit, land within a stated tolerance of the published figure for
   this method/model. **If it cannot reproduce a known-good 4-bit result, its ternary number is discarded,
   not reported.**
2. **Fire on a minimal lesion.** Zero the output projection of **one attention head in one layer** — the
   smallest structural corruption available, not a catastrophe chosen to flatter the detector. BPB must
   rise by ≫ the gate width.
3. **Exact comparator.** Same weights twice → bit-identical BPB. Different weights → different BPB.
4. **Named refusal** when config and loaded weights disagree (§6.5, §6.6 — `strict=True` does not verify
   architecture).

### 3.8 Cost — for approval, not yet approved

CPU-only, owner's machine, zero spend, no GPU. Stack verified: `torch 2.12.0+cpu`, `transformers 4.57.6`,
**no CUDA**, 80 GB RAM, 1.3 TB free. ~12–20 h wall-clock, interruptible. Requires `pip install datasets
accelerate` — **both the install and the CPU budget are currently withheld by the Architect.**

### 3.9 The decision requested if it fails — pre-specified

I do **not** conclude "PTQ is impossible" — stage −1 is evidence against *the tested, bounded path*. I
report the bit-width curve and ask one question: **is a named 4-bit or mixed-precision engine path worth
pursuing?** I will price what that costs, since it forfeits part of the measured faster-as-bits-drop kernel
advantage — though note §2.3 now shows that advantage is compute-bound, which changes the calculus.

---

## 4. Stage-1 arithmetic — method and envelope

The per-donor table is being computed **from actual `config.json` files** (law 5), not model cards. It
lands in `DONOR_STAGE1_ARITHMETIC.md`.

### 4.1 The cost model

```
t_token = streamed_ternary_bytes / [4.35 .. 11.4] GB/s     (LUT class — REVISED, see §2.3)
        + MoE dispatch overhead                             (mechanism named §2.3; magnitude 1/D-scaling)
        + proj_bytes / r(proj_bytes)                        (r = the §1b t6 rate curve)
        + head_bytes / r(head_bytes)
        + KV_bytes_at_context / r(KV_bytes)                 (<-- THE TERM I OMITTED IN v1, §2.4)
        + scan / norms / glue                               (per-position floor; ternary does not shrink it)
```

Every organ a **bracket**, never a point. Above 96 MB use the 34–36 GB/s asymptote, labelled extrapolation.

### 4.2 The envelope [D]

1. **Active params/token ≲1.4–2.3B** at 11.4 GB/s. The sharpest filter, free to apply.
2. **Dense donors ≥8B are dead** — 2.09 tok/s measured-rate (§2.9b), and the whole dense 1–100B band with them.
3. **Total ≲24B ternary for SKU-A**; SKU-B is *empirically* empty (§2.9a) pending a release under 1.96B active.
4. **The head must be compressed** (34–58% of budget alone).
5. **The 128K KV read path must be bounded** — quantization alone leaves 92% of the budget consumed (§2.4).

Live region: **sparse-MoE donors, ~1–2B absolute active params, ≲24B total, with a bounded attention path.**
Narrow, and narrower than v1 claimed.

### 4.3 What would invalidate this — and the next measurement I am requesting

The 17.0 GB/s premise is resolved (§2.3): it was an asymptote, not a target, and the working figure is
11.4. The remaining load-bearing unknown is the **1/D scaling of the MoE dispatch overhead**. The 4.345
GB/s figure is an artefact of `D=256, h=128`; a donor expert is 48× larger; per-call and per-byte
components are **not separable from existing data**.

> **Next Builder brief I am requesting: sweep `EB` and `D` in the existing `run_expert_rate` harness and
> fit `t = c + b·bytes` per thread count.** No donor, no GPU, no model — pure microbenchmark on code that
> already exists. It is the cheapest measurement that de-risks the largest number in the budget.

Also still open: per-group activation scales (§8.G.5, not implemented) would change the inner loop.

---

## 5. Pre-registration — stage 2a (the killer-question screen)

### 5.1 Stage 2a-0 — desk, free, run first

Per §1.4 this no longer *closes* branch (iii); it bounds one term. The free steps remain: the donor-dimension
kernel-rate check (§4.3) and the KV-traffic arithmetic (§2.4), which is the term that actually decides.

### 5.2 Stage 2a proper — per-layer fidelity

Four arms per layer, on fixed donor activations from the §3.5 calibration set:
- **(a)** donor attention layer as-is (reference)
- **(b)** converted to SSM by closed-form init, **no healing** — `Wq`→`C`, `Wk`→`B`, `Wv`→`X`, recorded as
  shape-matched, not equivalent (§2.8)
- **(c)** (b) plus layer-local output matching under a declared budget (§S1 grey zone: token count, fitted
  parameters, objective, iterations, wall time per layer — declared in advance, or it is a §8.M healing
  proposal instead)
- **(d)** the control — see §5.4

### 5.3 The fidelity criterion — v1's version WITHDRAWN; replaced with a donor-calibrated ladder

**My v1 criterion was invalid and the Architect is right to block it.** It proposed judging conversion
error against the BPB cost of ternarization measured at 8.3M under QAT. That is an undemonstrated bridge
across scale, architecture, objective *and* the QAT/PTQ regime — **precisely the transfer this document
forbids elsewhere** (§8.C.1: "they do not transfer to a post-training setting and must not be quoted as if
they did"). I wrote a criterion that violated a law I had quoted two sections earlier.

**Replacement — calibrate on the same donor, empirically:**

1. Inject **controlled perturbations of monotonically increasing magnitude** into a single layer of the
   donor (e.g. scaled Gaussian noise on that layer's output, or progressively coarser weight quantization
   of that layer alone).
2. For each magnitude, measure **both** the local per-layer error **and** the effect on an **end-to-end
   reference metric** on this same donor.
3. That yields a **local-error → end-to-end-effect transfer curve for this donor**, measured, not assumed.
4. **Only then** set the local threshold: the local error whose end-to-end effect equals the largest
   degradation we are willing to accept at this stage.

This costs one extra sweep and it makes the threshold *derived* rather than *imported*. It also produces a
reusable instrument: any future local transformation on this donor can be priced against the same curve.
The threshold is pre-registered once the curve exists and before arms (b)/(c)/(d) are scored.

### 5.4 The planted control the literature never reports — MANDATORY

**Arm (d): convert the layer to an SSM of identical shape with RANDOM initialization.**

The entire S1-compatible claim is that the *initialization* carries the mathematics. If (b) is not
dramatically better than (d), closed-form init contributes nothing and all the value in the published
recipes lives in the healing S1 forbids. **Without arm (d), arms (b) and (c) prove nothing.** Per §2.8
this is not a formality — neither named precedent publishes an init-only number.

### 5.5 Also pre-registered

- **Layer-selection rule stated before running**, with a **random-selection control at the same fraction**
  (§8.B closure). MOHAWK's fixed progressive-halving schedule [L] is the natural published baseline beside it.
- **RoPE disposition** declared in advance; positional consistency between converted and retained layers
  is a **correctness** question and gets a correctness check.
- Reference apparatus only. No engine export, no general-retention claim, no GPU.

---

## 6. Decision matrix

| # | route | claim | cheapest discriminating measurement | GO | NO-GO | amendment |
|---|---|---|---|---|---|---|
| **i** | healing not needed | closed-form / local solve closes the gap | stage 2a (b)(c) vs the **donor-calibrated** curve (§5.3), with random-init arm (d) | (b) or (c) inside threshold, and (b) ≫ (d) | (b) ≈ (d), or error ≫ threshold on most layers | — |
| **ii** | minority fraction is wrong | keep attention on most layers | KV **traffic** + weights + head vs budget at each fraction — arithmetic, free | a workable fraction is a genuine minority | no fraction meets 10 tok/s at 128K | **S3 amendment if the workable fraction >50%** |
| **iii** | SSM not load-bearing for speed | speed is storage + streaming, operator-agnostic | donor-dim kernel rate + §4.3 sweep | confirmed at donor dims | scan or KV term dominates | — |
| **iv** | **donor already a structural minority** | sidestep the conversion | per-model `config.json` filter (§1.5) | a filtered donor clears stage-1 at some SKU | no donor passes the filter | engine grows the donor's operator — **engineering (§8.J)** |

**Status: route (iv) is an authorized priority shortlist, not a confirmed primary route.** Ordering: run
2a-0 and the §4.3 sweep (both free) → apply the §1.5 per-model filter → then choose. Route (i) is where the
best publishable result lives; it is not where the first working system lives.

### 6.1 Axes the first end-to-end donor trial holds constant (§9 scope rule)

If route (iv) survives: **the single variable at stage 5 is the precision map.** Held constant: no
attention→SSM conversion, no MoE-ification, no tokenizer change (S3's tokenizer contract), no
activation-sparsity substitution, no recall fusion (recall-off, engine's own logits as reference). Head
compression admitted **only** as part of the precision map.

**Caveat added in v2:** §2.4 may force the recall tier into the *first* trial rather than a later one,
because the 128K contract cannot be met without a bounded KV path. If so, that is a second variable and
the trial must be re-scoped rather than quietly widened.

### 6.2 The decision I am NOT making

**No GO/NO-GO on any sealed gate, and no amendment request.** Nothing has been measured on a donor.
Per §7.3: **no decision.**

---

## 7. Status of asks

| # | ask | Architect ruling |
|---|---|---|
| 1 | Confirm route (iv) as primary | **Authorized as priority shortlist only**, not primary. Falcon-H1 and the "5–25%" claim corrected (§1.5). |
| 2 | Approve stage −1 execution | **WITHHELD.** No `pip install`, no CPU budget, until §3.3 is double-verified and the pre-registration is Controller-reviewed. |
| 3 | Rule on §2.3 / §2.7 | **Granted** on §2.3 — use the measured bracket. §2.7 withdrawal stands pending verification. |
| 4 | Authorize push | **Granted, conditional:** commit the untracked artefacts separately first; no checkpoints, corpora or cache in the commit. Branch renamed `codex/research/donor-adaptation` → **`research/donor-adaptation`**. |

**Open items I now owe:**
1. Second independent verification of PT²-LLM / TWLA, then seal §3.3 (method, commit/revision, dependencies, tolerance).
2. Controller review of this stage −1 pre-registration.
3. The §4.3 `run_expert_rate` sweep brief — the cheapest measurement that de-risks the largest number.
4. The Builder's per-model `config.json` filter and stage-1 table.
5. Re-price the recall tier as a **critical-path** dependency, not a stage-4b side item (§2.4).

**Still not requested:** any GPU session, any spend, any sealed-constraint amendment, any merge.

---

## 8. Decision log (claim → evidence → independent check → open risk → next action)

| # | claim | evidence | independent check | open risk | next action |
|---|---|---|---|---|---|
| 1 | scan = 4.9% at t6 | [D] from [M] | **CONFIRMED** ×2 | **does NOT close branch (iii)** — Architect correction | donor-dim check |
| 2 | §5 overstates the LUT rate | [M] | **CONFIRMED**, then revised again by C#2 | none | bracket → [4.35–11.4] |
| 3 | dense ≥8B fails the gate | [D] | **CONFIRMED** ×3 (2.09 tok/s) | none | dense band closed |
| 4 | head = 34–58% of budget | [D], dims [?] | C#1 **PASS** | dims from memory | Builder upgrades [?] → [M] |
| 5 | hybrids sit at 5–25% attention | [L] survey | **WITHDRAWN** — over-generalized; Falcon-H1 is a counter-example | — | per-model filter (§1.5) |
| 6 | ternary PTQ may be near-lossless | **[L?]** | **second read REQUIRED** | a fabricated number was already caught | verify, then seal §3.3 |
| 7 | no init-only number in the literature | [L] ×5 attempts | — | absence ≠ evidence of absence | arm (d) measures it |
| 8 | D9 fatal at donor scale | [D] 104–153 ms | **CONFIRMED** ×2 (C#2: 88 ms) | does P61's cost transfer 8.3M-QAT → donor-PTQ? **unknown** | stage −1 |
| 9 | 17.0 GB/s reachable | assumed in v1 | **REFUTED** — compute asymptote, not target | working figure 11.4 | — |
| 10 | SKU-B provably empty | C#1 | **PARTIAL REFUTE** by C#2 | *empirically* empty; one release could change it | monitor |
| 11 | nibble-packing a free 2× | C#1 | **REFUTED as speed**, confirmed as footprint | ~10% slower on a compute-bound path | footprint lever only |
| 12 | §8.A's MoE>dense ranking wrong | C#1 | **REFUTED** — I was right; §8.A understated | both still fail 10 tok/s | — |
| 13 | **KV is a traffic term, not just footprint** | C#2 | **I was wrong in v1** — 368 ms at 128K | makes the recall tier critical-path | re-price stage 4b |
| 14 | stage-2a criterion via QAT anchor | v1 | **INVALID** — Architect; violates §8.C.1 | — | replaced by §5.3 ladder |
