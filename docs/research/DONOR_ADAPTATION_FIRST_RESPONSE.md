# Donor-Model Adaptation — the Adapter's first response (§13)

**Status: v2, 2026-08-20. STAGE-0 INTERIM REPORT — accepted by the Architect as such, and explicitly NOT
a basis for choosing the primary route or starting stage −1.**
**Branch `research/donor-adaptation`, base SHA `6906890`.**
**Nothing here has consumed a GPU-hour or a euro.**

**v2 changelog (Architect adjudication + Controller #2 replication):** four v1 claims were wrong and are
corrected in place with the correction recorded, not silently edited — §1.4 (branch (iii) over-reach),
§1.5 (Falcon-H1 and the "5–25%" generalization), §2.4 (I missed the KV *traffic* term entirely, and my
conclusion there was backwards), §5.3 (the stage-2a criterion used a non-transferable QAT anchor). The
measured-rate bracket moves from [4.2–17.0] to **[4.35–11.4] GB/s**.

**v3 changelog (Builder stage-1 result + Controller pre-registration review):**
- **§4.2 — the stage-1 pass is done on 18 real configs: ZERO donors pass the sealed ≥10 tok/s gate.**
  Best is `Qwen2.5-1.5B` at 9.76 tok/s, and only at 32K, below the sealed 128K contract. Controller audit
  in flight; not established until it lands.
- **§4.2b — three results contradict this document.** Footprint is not the wall (18/18 fit SKU-B; every
  elimination is speed). Hybrids satisfy S3 and still fail on active bytes, which **demotes route (iv)**,
  my own recommendation. §5 over-predicts tok/s by 2–10×.
- **§4.3 — I withdraw the measurement I asked the Owner to approve.** The dispatch overhead is ≤1.1% of
  any donor total and ρ-granularity does not arise at donor scale; both were sandbox artefacts. The
  correct ask is the LUT rate at donor *projection* dimensions.
- **§3 — the stage −1 pre-registration was returned NO with 7 BLOCKs and is rebuilt.** The v1 gate was
  **satisfiable by choosing the eval corpus**; it is now an absolute ΔBPB gate with an INCONCLUSIVE band,
  replicates and a bootstrap SE, and two of its four "planted controls" were replaced — one of which
  might never have fired.
- **§7.1 — a NO-GO or an amendment request is now forming.** One measurement stands between them.

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
| pick a donor with ≲1.4–2.3B **absolute** active params | decides pass/fail outright | forced |
| **the LUT rate at donor projection dims** (17.0 vs 11.4 GB/s) | **2–4× on every row** — one donor passing vs none | **UNMEASURED — the open question** |
| attention→SSM conversion *as a speed lever* | bounded by the 4.9% scan share | **not the lever** |
| ~~fix the MoE dispatch defect~~ | **≤1.1% of any donor total** | **retired §4.3 — I was wrong** |
| ~~ρ-safe expert granularity~~ | real experts 1,548–86,144 KB vs a 48 KB threshold | **retired — does not arise at donor scale** |

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

**UPDATE — the filter has now been applied, and route (iv) does not survive it as stated.** From 18 real
configs (§4.2): the hybrids **do** arrive S3-compliant — granite 10% attention, Nemotron-H 8%, Zamba2 17%,
Qwen3-Next 25% — **and every one of them still fails**, at 1.62–3.87 tok/s, on active bytes rather than on
the operator boundary.

**So route (iv)'s premise is confirmed and its conclusion is refuted.** Arriving pre-converted is real and
free, and it is *not sufficient*: the binding constraint was never the attention fraction, it is absolute
active parameters per token. A donor that is already 90% recurrent still loses if it streams 9B active
params. Granite-4.0-H-Small is the clean example — S3-compliant at 10% attention, eliminated at 9B active.

This is direct arithmetic support for the mandate's §10 outcome 3, and it means **the operator boundary
(§8.B) is not where the donor programme is decided.** I am not withdrawing route (iv) — it remains the
cheapest way to satisfy S3 *if* a donor otherwise qualifies — but it is demoted from "the route" to "a
free property to prefer among candidates that pass the active-parameter filter first".

*(Caveat held open: the Builder's parameter formula was measurably wrong on Zamba2 by +164%, for
shared-memory-block reuse. If the Controller finds that defect touches the other hybrids too, these
eliminations are unsound and this paragraph must be re-read.)*

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

### 2.7 [RETRACTED — the mandate was RIGHT and my correction of it was built on a fabricated number]

**This section said the opposite in v1 and v2. I am reversing it, and the reversal is the most important
result of stage 0.**

**What I claimed (v1/v2):** that §8.C's "roughly ~2× perplexity at 7B–70B" for post-training
ternarization was unsourced, that it had drifted across the QAT/PTQ line, and that the calibration-only
frontier now claims **near-lossless** (PT²-LLM at 1.04–1.08×) — so the mandate's instruction to "assume
the naive path is broken" might be **out of date in the favourable direction**.

**What independent verification found** (`PTQ_SOURCE_VERIFICATION.md`, primary-source read of the paper's
Table 1 pulled from raw HTML by byte offset):

| model | FP16 | PT²-LLM W1.58 | ratio |
|---|---|---|---|
| LLaMA-2-7B | 5.47 | **11.56** | **2.11×** |
| LLaMA-2-70B | 3.32 | **6.27** | **1.89×** |
| LLaMA-3-8B | 6.14 | **32.19** | **5.24×** |

Plus 13–28 points of absolute zero-shot accuracy lost. The paper's abstract never claims lossless —
verbatim: *"competitive performance against state-of-the-art **2-bit** PTQ methods."*

> **The mandate's "~2×" figure is essentially exact at LLaMA-2-7B (2.11×). §8.C was right, its
> instruction to assume the naive path is broken was right, and my "correction" of it was wrong.**

**The fabrication was traced to its source, and it is instructive:** the "1.04–1.08×" ratios were computed
from **baseline cells re-labelled as ternary results** — 5.71 is LLaMA-2-70B's FP16 *C4* perplexity, 5.09
is LLaMA-13B's FP16 WikiText2 perplexity. The ratio came out near 1.0 because it was **fp16 divided by
fp16**. A third quoted figure (3.58) appears nowhere in the paper; the nearest token is `35.58`, an ARC-c
accuracy.

**What does verify about PT²-LLM:** it is real (ICLR 2026, code public), it uses 128 WikiText2 samples,
and it is **genuinely closed-form** — *"requires no training or gradient backpropagation"*, 32 min on one
A800 for 7B, no learning rate anywhere in the paper. So the S1-compatibility claim survives; only the
quality claim was fabricated.

**TWLA (arXiv 2606.13054) — real, but NOT training-free.** Verbatim: *"gradients of ℒ_shape update the
free matrices S₁ and S₂"*, and *"we optimize the parameters in KOTMS for 100 iterations with a fixed
learning rate of 0.01."* That is S1's declared grey zone, not a closed-form transform — and **the entire
2.11× → 1.27× improvement over PT²-LLM is bought with that gradient step.**

### 2.7b The decisive consequence: no verified calibration-only ternary path exists at 1–3B

- **PT²-LLM's smallest model is 7B. TWLA's smallest model is 7B.** Neither paper contains a single point
  below 7B.
- The only 1–3B ternary evidence located (ScaleQ-1.58, Qwen3-1.7B) loses **40–58% relative on all five
  tasks**, and is **not** calibration-only — 60 epochs of STE gradients, ~32 A100-GPU-hours. Its own text:
  *"smaller models are more sensitive to quantization than larger ones."*
- **Best verified alternative is 4-bit**: QuaRot W4A4, ≤0.47 PPL loss, ~99% retention. **2-bit scalar
  collapses even at 7B** — TWLA's own table: GPTQ-W2 47.13, QuaRot-W2 19.97, against FP16 5.47.

**This reaches the mandate's own stage −1 decision point from the literature alone, without running
anything** — which is the cheapest possible way to arrive there and exactly what stage 0 is for. See §7.1.

### 2.7c Three fabrications in one evidence chain — a pattern worth naming

Caught this session, all by re-reading primary sources: (1) a fabricated init-only PPL for
Mamba-in-the-Llama; (2) PT²-LLM's "near-lossless" ratios, computed fp16-over-fp16; (3) a summary asserting
ScaleQ-1.58 is "training-free" when its raw text says it is built on CAT-Q, *"the first differentiable
ternarization method."*

**Every one was produced by an automated summarization pass over a paper, and every one was plausible.**
This is the project's "plausible artefact" law operating on *literature* rather than on instruments, and
the defence is the same: **the artefact is the authority.** Concretely, for this project: a literature
number may not enter a decision unless someone has read it in the paper's own table. I let one of these
shape a recommendation to the Owner before verification completed, and that was my error, not the tool's.

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

Stage −1 runs before any engine work because it is the cheapest thing that can redirect the programme.

**Scope, corrected — I twice wrote that a failure makes the ternary route "moot" or "very probably dead".
That over-reads what this stage can show**, and contradicted my own §3.6 and §3.9. Consistent with the
mandate's deliberately narrow wording: **a failure here is evidence against *this bounded method*, at
*this* bit-width, on *this* donor, at *this* scale** — not a claim about all possible PTQ. That is the
whole of what stage −1 licenses.

### 3.2 Donor — SEALED

**`Qwen/Qwen2.5-1.5B`**, revision pinned by the hub-resolved SHA, recorded in the manifest. Apache-2.0.
Chosen because (a) permissive licence, (b) small enough to run end-to-end on the reference CPU, (c) it is
the most heavily represented model class in the rotation-PTQ literature, which lets me validate the
apparatus against published 4-bit numbers before trusting any ternary number. That cross-check is the
apparatus's planted control, not a convenience. **One donor, one variable** — no hybrid here.

### 3.3 Method — verification COMPLETE, and it changed the question

**Verification is done (§2.7). The result is that the premise of this stage has to change before the
method can be chosen.**

- **PT²-LLM (arXiv 2510.03267): the paper is real, ICLR 2026, code public, and genuinely closed-form** —
  no training, no gradients, no learning rate, 128 WikiText2 calibration samples, 32 min on one A800 for
  7B. **S1-compatible.** But its real W1.58 result is **2.11× perplexity at LLaMA-2-7B**, not the
  fabricated 1.04–1.08×.
- **TWLA (arXiv 2606.13054): real, but NOT training-free** — 100 gradient iterations at lr 0.01. S1 grey
  zone, and its entire advantage over PT²-LLM comes from that step. **Not eligible as a primary-path
  method** without an explicit S1 declaration.
- **Neither paper has a single datapoint below 7B**, which is the regime this stage would run in.

**Consequence: stage −1 as originally framed is no longer the right experiment.** It was designed to ask
"does the calibration-only ternary frontier work?" — and the literature now answers that at 7B without us
spending anything: **it does not, by a wide margin.**

**Re-scoped stage −1, pending the Architect's ruling on §7.1's amendment request:** the deliverable is the
**bit-width curve** (§3.4) with PT²-LLM as the closed-form arm, run to locate **the knee** — the lowest
bit-width that holds — rather than to pass/fail ternary. Ternary at 1–3B genuinely must be *measured, not
cited*, since no paper covers it; but it is now the expected-bad end of a curve, not the hypothesis.

**Still not sealed, and no run may start until:** the Architect rules on §7.1; the exact commit/revision,
dependency set and numerical tolerance are fixed in writing; and the Controller re-reviews §3 after its
7-BLOCK rebuild.

### 3.3b PRINCIPAL DECISIONS after the artefact build (2026-08-20)

Two calls I owe, both forced by facts read out of code rather than from prose:

**Decision 1 — the five-point sweep CANNOT be run with PT²-LLM, and the stage splits into two arms.**
`quantize.py@9e943e6` restricts `low_quant_method` to `['atq','atq-itf','atq-aga','ternary-init','fp16']`.
**There is no 4/3/2-bit path: the method is ternary-only.** So §3.4's "one variable, five points" is not
merely under-specified (B7) — it is **unimplementable with this method**, and any 4/3/2-bit point would
silently be *a different method*, which is the exact defect B7 names. Confirmed from the implementation,
not inferred.

> **Ruling: two arms, separated, each internally one-variable.**
> **Arm A — the SOTA ternary point.** PT²-LLM at ternary, anchored to publication by control 1b. Answers:
> *does the best verified closed-form ternary method work at 1.5B?* — the regime no paper covers.
> **Arm B — the knee.** A bit-width sweep (4/3/2) within a **single** family that genuinely spans them
> (QuaRot/GPTQ class). Answers: *where does a rotation-PTQ family break?*
> They answer different questions and are **never combined into one curve**. Arm A is not a point on
> Arm B's curve, and reporting them as one would manufacture the confound B7 warned about.

**Decision 2 — control 1b re-anchored, and the gate is respected.** PT²-LLM's published models are
LLaMA-2, which is **gated (`gated: manual`, HTTP 401)**. An `HF_TOKEN` exists in the environment and the
Builder **did not use it**, and did not take an ungated mirror — correct on both counts.

> **Ratified: re-anchor control 1b to `huggyllama/llama-7b` @ `4782ad27…`, verified `gated: false`.**
> This is **a different row of the same table, in the same paper, at the same bit-width** (fp16 5.68 →
> W1.58 11.39), so control 1b remains a genuine published-number reproduction and
> **UNVALIDATED-AGAINST-PUBLICATION is not triggered.** Ungated LLaMA-2 mirrors exist; they are recorded
> and **not used** — that is a licence question for the Owner, not a shortcut for me.

**Two facts from the build that change numbers elsewhere:**
1. **PT²-LLM does *fake* quantization** — dense floats, never a packed format. So "ternary" here is
   **1.835 bits/weight at fp16 scales, 2.085 at fp32**, not 1.58. **Any footprint arithmetic quoting
   0.5 B/weight from this method is wrong**, and §2/§4's inventories must use the effective figure.
2. **B is measured at 4.334, not 4.0** — and **LLaMA's tokenizer gives 3.7965 on the same bytes, 14.2%
   apart.** The anchor table was never a single-B object. Tokenizer-corrected, **the Qwen row is 0.316,
   not 0.342 — mid-pack, not second-worst.** Cutting Δ\* still stands on the "unattributable" argument,
   but **the specific number the reviewer and I both argued from was off by 8%**, and I record that
   rather than let a convenient error stand.

### 3.4 The swept variable — bit-width; the deliverable is a curve

**bf16 baseline → 4-bit → 3-bit → 2-bit → ternary (1.58-bit)**, identical apparatus, calibration and eval.

*(Corrected: this said "fp16", contradicting §3.8's bf16 device policy. The **published anchors are
fp16**, so every anchor comparison must state which side is which — see §3.6.)*

**"One variable, five points" was an overstatement and §3.7b already contradicts it.** The five points are
one variable **only if** the implementation uses the same algorithm at each — group size, quantizer class
and outlier policy commonly change between 4-bit and 2-bit. §3.7b's per-point config table, held-constant
group size and outlier policy, and reported effective bits/weight are what *make* it one variable; absent
those it is a five-point comparison of five methods.

**The deliverable is the curve and its knee (§3.6 B), not a pass/fail.** The route kill line (§3.6 A)
exists only to stop the route if the apparatus is broken or the scale is hopeless. And the curve does
**not** by itself answer "which bit-width would the engine have to grow to" — that is an engine-design
question involving kernel rates, footprint and S4 retention, none of which stage −1 measures. It supplies
**one input** to that decision.

### 3.5 Calibration corpus — SEALED

**Globally shuffled, never in file order** (measured law: blocked sampling order cost +0.0339 BPB =
6.8 σ_seed at identical data, tokens and steps — order alone did it). Not code-only (§8.L.1: a code-only
set is a plausible way to silently destroy the general capability S4 measures). Pinned by content hash;
manifest committed, **corpus never committed**. Decontaminated against the eval slice and the pinned
P62 code-val.

> **F2, honestly labelled: "composition mixed" is NOT a specification, and my v4 claim that this section
> had been fixed was false — it was unchanged.** Exact sources, proportions, sampling procedure and seed
> are pinned in `benchmarks/donor_adaptation/stage_minus1/prereg.yaml`, not here.
>
> **F1: the sample count is not 512×2048 by default.** The anchor arm runs at **the paper's 128 samples**,
> because control 1a's whole value is comparability with a published number. Any larger calibration set is
> a *separately declared arm*, never the silent default.
**Sensitivity arm (§8.L.2):** the ternary point repeated on a **second disjoint calibration set**. If they
differ by more than harness σ, the set is load-bearing and every future number carries its hash.

### 3.6 Metric and threshold — v1 gate WITHDRAWN; rebuilt after Controller review

**A Controller pre-run review returned NO with 7 BLOCKs (`CONTROLLER_PREREG_REVIEW.md`). The v1 gate was
gameable and I withdraw it.** The defects were mine and the two worst are recorded here in full, because
they are the exact failure this project's anti-Goodhart law exists to prevent.

> **B1 — the v1 gate was satisfiable by choosing the eval corpus, without touching the method.** With
> `converted ≤ donor × 1.10`, the headroom is `0.10 × BPB_donor` — so a slice on which the donor scores
> 1.5 grants *twice* the permitted damage of one on which it scores 0.75. `CANONICAL_EVAL.md` records that
> slice choice alone moves absolute BPB by ~0.04, which is 20× a typical gate width. I wrote a relative
> gate against an unpinned denominator and called it sealed.
>
> **B2 — a multiplicative threshold on a log-scale metric is the wrong shape, and it was ~2.7× looser than
> the candidate method's own published claim.** At donor BPB 0.75 the v1 gate permits Δ = 0.075 b/byte ≈ a
> **1.20–1.26× PPL ratio**; PT²-LLM claims **1.08×**, i.e. a BPB ratio of ≈1.037×. **A method
> underperforming its own publication threefold would still have PASSED** — so the gate could not make the
> one distinction stage −1 exists to make.

**Replacement gate — absolute, on the honest quantity:**

Metric: **ΔBPB = BPB_converted − BPB_donor**, in excess bits per byte, on a slice pinned before anything
is scored. Absolute donor and converted BPB always both reported alongside the delta.

**v3's replacement gate was ALSO wrong, in a worse way, and this is the third rewrite.** Controller
re-review found that I justified Δ\* = 0.10 as *"~3.5× above the candidate method's own published Δ
(≈0.028)"* — and **that 0.028 was computed from the fabricated 1.08× ratio that §3.3 retracts three
paragraphs earlier in the same document.** I struck the number and then used it to set the gate.

Re-derived from the **verified** table:

| method | class | PPL ratio (verified) | bits/token | **Δ b/byte at B≈4** |
|---|---|---|---|---|
| QuaRot W4A4 | 4-bit, closed-form | ~1.09× | 0.118 | **≈0.03** |
| TWLA W1.58A16 | ternary, **gradient-assisted** (S1-forbidden) | 1.17–1.53× | 0.23–0.61 | **≈0.086** |
| **PT²-LLM W1.58** | **ternary, genuinely closed-form** | **2.113×** (5.47→11.56) | **1.079** | **≈0.27** |

**The sign was inverted.** Δ\* = 0.10 is not 3.5× *above* the closed-form ternary frontier — it is **~2.7×
below** it. **PT²-LLM reproducing its own publication exactly would have FAILED my gate**, and 0.10 sits
on the gradient-assisted frontier that S1 forbids. A gate no S1-compatible method has ever cleared is not
a kill gate; it is a guaranteed FAIL dressed as a criterion.

**Replacement — and the conflation that caused this is now separated.** I had fused two different jobs
into one number. They are split:

> **(A) — CUT. There is no threshold on the donor at all.**
>
> *(Third rewrite. Δ\* = 0.30 was re-anchoring on one row and the reviewer's refutation is decisive.)*
>
> 0.27 is **not "the frontier"** — it is one of seven PT²-LLM rows. Recomputed at B = 4: **0.21, 0.228,
> 0.230, 0.252, 0.270, 0.342, 0.597.** The frontier is a **range, 0.21–0.60**. And **the only Qwen row,
> Qwen3-14B, is 0.342 — which fails the 0.30 line.** Add the two trends the verification states outright
> (degradation worsens as models shrink, and as they become token-saturated): `Qwen2.5-1.5B` is small
> *and* modern *and* Qwen, so its expected Δ **under a flawless implementation is ≥0.34**.
> **A gate that fails when nothing is wrong cannot separate "our apparatus is broken" from "this scale is
> hard" from "this method is weak" — three causes, one bit of output, unattributable.**
>
> It was also the **fourth instance of the same contamination**: I ordered the *anchors* recomputed at
> measured B while freezing Δ\* at a number computed at B ≈ 4.0. At B = 3.5 the frontier is 0.308; at
> B = 3.2 it is 0.337 — **Δ\* drops back below the frontier and the v3 sign inversion returns.**
>
> **The stopping rule moves to where a discrepancy is ATTRIBUTABLE:** control 1a, on the *published*
> model, against a *published* number. If the apparatus misses its published anchor by more than the
> stated tolerance, **the apparatus is broken → stop and fix it, and report no donor numbers.** That is
> apparatus validation, where a miss has exactly one cause. Testing "did we implement it right" on a
> model with no published number was always the wrong place for it — where valid it duplicates control 1,
> and where it does not duplicate control 1 it is uninterpretable.

> **(B) THE ACTUAL DELIVERABLE — the bit-width curve and its knee**, reported in b/byte **against the
> three verified anchors above**, not against a threshold. Stage −1 was re-scoped in §3.3 from "does
> ternary work" to "where is the knee"; a pass/fail number is the wrong instrument for locating a knee,
> and (A) exists only to stop the route if the apparatus is broken or the scale is hopeless.

**B (bytes per token) is MEASURED, never assumed.** Every anchor conversion above divides by it, and this
project has a banked finding that a prior dossier's PPL↔BPB conversions were **wrong**. B is measured on
the pinned slice with the donor's own tokenizer and reported with every number; the anchors are recomputed
at the measured B before any comparison is made.

**Correcting my own B1 claim:** I wrote that an absolute gate *"cannot be moved by slice choice"*. **That
is false.** ΔBPB is content-dependent — the same method spans 1.79×–5.24× across models in our own
verification, and slice content moves it likewise. Absolute removes the *denominator* gaming, not the
slice dependence. **The slice must still be pinned to the `CANONICAL_EVAL.md` standard**, and §3.5b now
does that.

### 3.5b Eval slice — PINNED (B1, previously left open twice)

Pinned before any arm runs, to the project's own canonical standard: **corpus identity + sha256**, **byte
range / offsets**, **tokenizer name + revision**, **context length AND stride**, **BOS handling**, **loss
reduction**, and the **decontamination procedure** against both the calibration set and the pinned P62
code-val. **`BPB_donor` is measured and recorded as a number, and `B` with it, before a single converted
arm is run.** A missing field here is a BLOCK, not an untidiness: `CANONICAL_EVAL.md` records that slice
choice alone moves absolute BPB by ~0.04, which is several times the resolution we are trying to read.

**Asymmetry — restated, and the FAIL side re-scoped after the Controller flagged I had over-read it:**
- **PASS says NOTHING about S4.** Not ≥90% global retention, not ≥80% per critical task, not S2 fit, not
  anything at donor scale. It licenses the next cheap question and nothing more.
- **FAIL is evidence against *this bounded, tested* method at *this* bit-width on *this* donor.** My v1
  said the path would be "very probably dead"; the Controller notes this contradicts both my own §3.9 and
  the mandate's deliberately narrower wording. **The narrower reading governs**, and I flag the tension to
  the Architect, who endorsed the stronger phrasing — I would rather have that resolved explicitly than
  pick one silently.

### 3.6b Statistical resolution — was missing entirely (B4)

§8.N.4 requires a σ per metric before any number decides anything, and v1 had none — a single BPB per arm.
Worse, it cited **σ_seed = 0.005**, which was measured on *from-scratch training seeds at 8.3M*. Stage −1
trains nothing; importing that constant is precisely the cross-regime transfer I criticized in §2.5 and
§5.3. My "15 σ_seed" framing measured distance from zero, which nobody is testing.

**Corrected after re-review — my band used the wrong multiplier.** I wrote `± 2·SE`, but at n = 3 the
two-sided 95% Student-t multiplier is **t = 4.30, not 2**. The written band was **~2.15× too narrow**,
which would have manufactured confident verdicts from three noisy points — the precise failure the band
existed to prevent.

**Now that the GPU makes a run ~1–2 h instead of 12–20 h, n = 3 is no longer a budget-forced compromise:**

- **n ≥ 10 replicates on the calibration-draw axis** (t ≈ 2.26 at n=10, and the estimate stabilises).
- The **PTQ/rotation-seed axis may be degenerate** — PT²-LLM documents no RNG. **Check whether it is
  stochastic at all before spending replicates on it**; if deterministic, report that as a finding and put
  the budget on the calibration axis.
- **Bootstrap SE**, reported per arm, with the band `Δ* ± t(n−1)·SE` stated before any number exists.

Also fixed: §3.5's "differ by more than harness σ" was a dead guard — that σ was never defined. It now
refers to the bootstrap SE above.

### 3.7 Planted controls — v1's set was inadequate; rebuilt (B3, B5, B6)

1. **Reproduce a known-positive — SPLIT IN TWO, because v1's version could not be run at all.**
   The Controller found §3.2 and §3.7.1 in direct conflict: PT²-LLM publishes **LLaMA-2 7B/13B/70B,
   WikiText2 PPL, at ternary** — there is no Qwen2.5-1.5B row and no 4-bit row to reproduce, and the
   tolerance was unstated, which makes it post-hoc by construction.
   - **(1a) Machinery validation, runnable:** reproduce a *well-replicated published 4-bit* number
     (GPTQ/QuaRot class) on the chosen donor, tolerance stated in advance as an absolute PPL band. This
     validates the rotation-and-quantization machinery.
   - **(1b) Method validation:** reproduce the paper's own ternary number on the paper's own model. If
     that model is too large for the CPU budget, **the method is declared UNVALIDATED-AGAINST-PUBLICATION
     and that limitation is reported with every number** — not quietly skipped.
   - Note both anchors are PPL-on-WikiText2 while the gate reads BPB. The apparatus must therefore emit
     **both**, and the PPL↔BPB conversion is pinned in advance (the project has been burned by exactly
     this conversion before — see the E5-bis dossier).

2. **Fire on a minimal lesion — REPLACED, because v1's lesion may be a known-*negative*.**
   Zeroing one attention head's output projection is the most *redundant* structure available; the
   head-pruning literature says it may move nothing at all. **I designed a control that might never fire
   and called it a planted control** — the project's characteristic failure, in my own apparatus.
   Replacement: the **monotone lesion ladder I had already designed in §5.3** — perturbations of
   increasing magnitude — reporting the **detection floor** (the ε at which ΔBPB reaches the gate width),
   anchored by at least one rung on a known **non-redundant** structure. This measures the instrument's
   sensitivity instead of asserting it.

3. **Exact comparator.** Same weights twice → bit-identical BPB; different weights → different BPB. Both
   logged. *(Controller: PASS as written.)*

4. **Named refusal** when config and loaded weights disagree (§6.5, §6.6 — `strict=True` does not verify
   architecture). **Must be exercised in both directions and logged** — v1 named it but never exercised
   it, which is §6.4 violated.

5. **NEW — prove the calibration data reaches the output (B6).** Run the pipeline with a
   **random-vocabulary calibration set**. If the output is bit-identical to the real-calibration run, the
   calibration path is dead and the entire stage is void. v1 had nothing that could detect this; the
   §3.5 disjoint-set arm cannot substitute, because a null there is ambiguous.

### 3.7b The sweep is not one variable (B7)

bf16 → 4 → 3 → 2 → ternary is only one variable if the method is the same algorithm at every point. It
usually is not: group size, quantizer class (ternary is a 3-level thresholded codebook, not the 1.58-bit
point of a uniform grid) and outlier/mixed-precision escape hatches commonly change between 4-bit and
2-bit. Required: a **committed per-point config table read from the implementation**, group size and
outlier policy held constant across points, and **effective bits/weight including scales** reported per
point — otherwise the curve is not a curve.

### 3.7c FLAG register — REWRITTEN, because my v4 version claimed closures that did not exist

**The third review's decisive finding was not a reasoning error. It was that this subsection claimed five
fixes the target sections did not contain**, including one that recorded a *still-open* finding as closed
**under the wrong label**. The reviewer's phrase was "a corrupted audit trail", and it was accurate. A
false closure claim is worse than an open finding, because it removes the thing from the list.

**Structural fix, not a promise to be more careful:** the pre-registration moves out of prose and into
`benchmarks/donor_adaptation/stage_minus1/prereg.yaml`, guarded by `check_prereg.py`, which **exits
non-zero listing every field still unpinned**. A closure claim that the artefact does not support becomes
mechanically detectable instead of assertable. **Status below is what the guard reports, not what I believe.**

| flag | claim in v4 | actual status |
|---|---|---|
| **F1** calibration size vs the paper's 128 samples | "fixed" | **OPEN → moved to `prereg.yaml`** (anchor arm at 128; larger sets separate declared arms) |
| **F2** "composition mixed" is not a specification | "fixed **in §3.5**" | **FALSE CLAIM. §3.5 is byte-for-byte unchanged** and still reads "Composition mixed". **OPEN → `prereg.yaml`** must carry exact sources, proportions, sampling procedure and seed |
| **F3** `torch_dtype` hidden default | "superseded by §3.8" | **CLOSED** — §3.8 does read dtype from the donor config and pin accumulation |
| **F4** control 4 named but never exercised | mislabelled **"(F6)"** | **OPEN → `prereg.yaml`**, with its both-directions exercise required in the log |
| **F6** dense→MoE transfer, per-expert calibration coverage | **recorded closed under F4's text** | **STILL OPEN and was never addressed at all.** `grep -i expert` in §3 returns zero hits. The donor is dense 1.5B; the target class is MoE ≤24B, and per-expert calibration coverage is the named mechanism against transfer |
| **N1** method repo absent from the install line | "pinned by commit SHA" | **FALSE CLAIM** — the install line still read only `datasets accelerate`. Now corrected in §3.8, with the SHA itself in `prereg.yaml`, since **a SHA written in prose is not a pin** |

**Also still open from pass 2, and not to be claimed closed until the artefact carries them:** control 1's
numeric tolerance (promised three times, still no number), the §5.3 ladder's ε grid / rungs / failure
response, control 5's `k·SE`, **control 1b's feasibility** (LLaMA-2-7B at bf16 is ~13.5 GB against the
3060's 12 GB — it does not fit) and **LLaMA-2's gated licence**.

**Ordering defect, also real:** §3.5b ("Eval slice — PINNED") sits physically *after* §3.6 and orphaned
the "PASS says nothing about S4" paragraph into it. Recorded; the section order is repaired when §3 is
regenerated from the artefact rather than hand-edited again.

### 3.8 Execution environment — SEALED (owner-approved GPU, 2026-08-20)

The owner has authorized a CUDA PyTorch install and use of the RTX 3060 for stage −1. **A device move
changes the numerical environment of a project that gates on determinism, so the device, dtype and
determinism policy are pinned here, before the run, as part of the pre-registration.**

**Device — pinned, and the second GPU explicitly excluded:**

| | |
|---|---|
| target | **`cuda:0` = NVIDIA RTX 3060** (Ampere, cc 8.6, 12 GB) |
| excluded | **NVIDIA GTX 1660** (Turing, cc 7.5) — different numerics, no usable bf16 |
| mechanism | see below — **a device-name string match is the weakest available check and is NOT sufficient** |

**Corrected after re-review.** I proposed asserting `torch.cuda.get_device_name(0)`. That is the weakest
check available and it has a specific hole: **`device_map="auto"` will happily shard the model onto the
1660 while the name assertion on device 0 still passes.** Pinned instead:

- **`CUDA_DEVICE_ORDER=PCI_BUS_ID`** plus `CUDA_VISIBLE_DEVICES` selecting the 3060 **by UUID**;
- assert **`torch.cuda.device_count() == 1`** — the 1660 must not be visible at all;
- assert the device of a **materialised parameter**, not just the ambient device;
- **`device_map="auto"` is forbidden**, explicitly, for the reason above.

With two GPUs present, `cuda:0` ordering is not stable across driver or enumeration changes, and silently
landing on Turing would produce plausible numbers at different precision. The run refuses to start unless
all four hold.

**dtype policy — pinned:**
- Donor weights loaded at the dtype **the donor's own config declares** (law 5: the artefact is the
  authority — do not impose fp16 because it is convenient). For `Qwen2.5-1.5B` that is bf16, which
  Ampere supports natively.
- **Loss/BPB accumulation in fp32 always**, regardless of weight dtype. The quantity being measured is
  a ΔBPB difference at the resolution set by the §3.6b bootstrap SE (**the retracted "≤ 0.10" and
  "SE ≈ 0.007" constants are deliberately NOT reinstated here** — this sentence previously carried both,
  and τ in control 6 would have silently inherited them); fp16 accumulation error can inject noise at that
  scale and would confound the gate.
- **The donor arm and every converted arm use an identical dtype policy.** A delta between arms that
  differ in dtype is not a measurement of quantization.

**Precision constraints — my v3 reasoning here was WRONG and is corrected.**

I claimed TF32 "is enabled by default on Ampere for matmul" and made disabling it the load-bearing
safeguard. **That is inverted.** `matmul.allow_tf32` has defaulted to **False** since torch 1.12; it is
`cudnn.allow_tf32` that defaults True, **and this model has no convolutions**. Worse, under §3.8's own
bf16 weight policy TF32 is **largely inert** — so I nominated a safeguard against a trap that was not
armed, while the live knobs went unmentioned. This is the "verify, don't assert a mechanism because it
sounds right" law, violated by me, in a section about not fooling ourselves.

**The knobs that are actually live, all pinned and all recorded by reading them BACK after setting:**
- **`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`** — defaults **True**.
  This is the real analogue of the trap I thought I was closing.
- `allow_fp16_reduced_precision_reduction = False`.
- **SDPA / `attn_implementation` backend pinned explicitly** (no silent flash/mem-efficient selection).
- `torch.set_float32_matmul_precision("highest")`.
- **`preferred_linalg_library` pinned, and Cholesky damping recorded** — GPTQ-class methods solve
  ill-conditioned Hessians, so the linalg backend is a real numerical variable here, not a detail.
- No implicit autocast anywhere on the path; asserted, not assumed.
- `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed recorded seeds.
- **TF32 disabled too** — harmless, but demoted from "the safeguard" to hygiene.

**Control 3 does NOT test this, and my claim that it did was wrong.** I wrote that the exact comparator
(same weights twice → bit-identical BPB) "is precisely the guard that tests this whole policy". It is
**structurally blind** to the failure it was nominated against: TF32, reduced-precision reductions and a
flash backend are all **perfectly repeatable** — they give bit-identical output at an eight-bit mantissa.
Control 3 catches run-to-run variance, not silent precision loss.

**Replacement guard — NEW control 7, precision probe:**
- a **float64-referenced precision probe** on a fixed tensor op, **fired in both directions** (flags on →
  detectably worse; flags off → matches fp64 to tolerance), so the instrument is shown to *fire*;
- **every flag read back after setting** and logged, never merely assigned;
- **control 3 strengthened** to run **across process restarts** and **two different batch shapes** —
  batch-shape variation is what actually exposes backend switching.

**Control 6 — device parity, tolerance now derived rather than left blank.** Per re-review:
- compare **|ΔBPB_gpu − ΔBPB_cpu|**, *not* absolute BPB — common-mode offsets cancel in the delta, and
  the delta is the quantity the gate reads;
- the CPU leg is an **fp32/fp64 oracle**, not bf16 — a bf16 CPU leg measures emulation, not truth;
- run on the **donor arm and at least one converted arm**, since a bug may only appear post-quantization;
- **τ = 0.1 × the §3.6b band half-width** — the same "an order of magnitude below the resolution you
  intend to read" rule `CANONICAL_EVAL.md` already justifies, and **derivable before any number exists**,
  which is the point.

**bf16 introduces two confounds that must be closed before the sweep runs:**
1. **§3.4 still says "fp16" — a live internal contradiction**, now corrected to bf16 throughout, and the
   published anchors are **fp16**, so the anchor comparison must state which is which.
2. **A bf16-native quantizer degrades preferentially at the low-bit end**, which would bend the very knee
   §3.6(B) exists to locate. **The quantizer's internal working precision is therefore pinned at
   fp32/fp64 and held constant across all five points** — otherwise the curve measures dtype, not bits.

**Cost, revised:** ~12–20 h CPU → plausibly **~1–2 h on the 3060**, and it frees the CPU the owner also
uses. Requires `pip install datasets accelerate`, **the method's own repository pinned by commit SHA**
(recorded in `benchmarks/donor_adaptation/stage_minus1/prereg.yaml`, not here — a SHA in prose is not a
pin), and a CUDA torch build (current: `torch 2.12.0+cpu`,
`cuda False`, 80 GB RAM, 1.3 TB free). Zero spend — the hardware is the owner's.

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

### 4.2 RESULT — the stage-1 pass is done, from 18 real configs

`DONOR_STAGE1_ARITHMETIC.md` + `benchmarks/donor_adaptation/donor_inventory.py`. 18 of 20 configs fetched
(`Llama-3.2-1B` and `Jamba-Mini-1.6` returned 401 gated → recorded UNAVAILABLE, **no workaround
attempted**). Every number tool-generated, none transcribed. **Independent Controller audit in flight** —
this table eliminates 16 donors permanently and may not be treated as established until that lands.

> **HEADLINE: on rates resting on nothing contested, ZERO of 18 donors pass the sealed ≥10 tok/s gate —
> at any SKU, any context, any precision map.**

| donor | tok/s (uncontested / granted) | note |
|---|---|---|
| `Qwen/Qwen2.5-1.5B` | **9.76** / 12.11 | best in set — and only at **32K**, not the sealed 128K |
| `allenai/OLMoE-1B-7B` | 5.98 / 11.01 | best sparsity ratio (1.18B active of 6.92B); dies at 128K on full MHA |
| `deepseek-ai/DeepSeek-V2-Lite` | 3.49 / 7.85 | MLA = 30 KB/context-token |

Granting the two contested figures (17.0 GB/s reaching the integrated engine, dispatch overhead removed),
exactly **two** clear SKU-A/32K and exactly **one** clears SKU-B/128K. Everything else is eliminated on
tok/s: Qwen3-1.7B 7.92, SmolLM2 7.80, gpt-oss-20b 4.73, Qwen3-30B-A3B 4.72, Qwen3-Next-80B 3.87,
Phi-3-mini 3.45, mamba2-2.7b 3.33, Mistral-7B 2.82, Falcon-H1 2.50, Qwen3-8B 2.36, Nemotron-H 2.01,
OLMo-2-7B 1.96, Mixtral 1.92, granite-4.0-h-small 1.62.

**Planted control FIRED** (§6.3 satisfied): the project's own 8.3M engine, pushed through the donor code
path as a synthetic config, returned **2,457,600 B/token = 2400.0 KB, error +0.0000%** against
`SIZING.md`'s independent in-engine count. The decomposition is exact and instructive — 49,152 B of codes
+ 2,048 B of scales = 50.0 KB/expert × 8 × 6. **The scales term is what makes it 2400 and not 2304; a
codes-only formula would have missed it.** Perturbation control: top-k 8→7 → −12.50%, exactly 7/8.
Refusal control: three deleted fields each raise a named `MissingConfigField`, unperturbed raises nothing.

**The tool also self-invalidated a row**, which is the behaviour I want: its parameter cross-check fired
4× unplanted on real donors, and on `Zamba2-2.7B` it was off by **+164.11%** — the Builder declared its own
formula wrong for shared-memory-block reuse and marked the row **unusable** rather than shipping it.
*(Containment is the Controller's highest-consequence audit item: if the same defect touches the other
hybrids, the hybrid eliminations are unsound.)*

### 4.2b Three results that contradict my own document

1. **Footprint is NOT the wall — §5 is wrong and so was I.** 14/18 fit SKU-A; **18/18 fit SKU-B**. Every
   single elimination is speed. My §2.9a discussion of SKU-B emptiness was arguing the wrong axis: SKU-B
   is not footprint-empty, it is speed-empty, and so is SKU-A.
2. **Hybrids already satisfy S3 — and it does not save them.** granite 10% attention, Nemotron-H 8%,
   Zamba2 17%, Qwen3-Next 25% full-attention. All arrive pre-converted, **all fail on active bytes**
   (1.62–3.87 tok/s). **This substantially weakens route (iv), which was my own recommendation.** It is
   direct arithmetic support for §10's outcome 3 instead.
3. **§5 over-predicts tok/s by 2–10×** across the board (30B/3B-active: §5 says ~28, computed 2.72–10.70;
   8B dense: §5 says ~10.5, computed 1.11–4.48) — because 42 GB/s is a rate no weight path here achieves.
   §5's *head* arithmetic is corroborated (1.16 GiB, 34.6–36.6 ms vs its ~30 ms). Its **KV omission is
   severe**: SmolLM2 carries 24.00 GB at 128K = 644 ms/token to read, 8.90 → 1.32 tok/s.

### 4.3 The measurement I requested is the WRONG one — withdrawn

**I asked the Owner to approve a `run_expert_rate` sweep to separate per-call from per-byte dispatch
overhead. The Builder's evidence retires that ask and I withdraw it.** Measured against real donor
dimensions:

- **the 8.4 µs dispatch term is ≤1.1% of any donor total** — the whole Controller#1-vs-#2 dispute about
  its decomposition, which I treated as load-bearing in §2.9(b), **moves no verdict**;
- **ρ-safe granularity is a non-issue** — real donor experts are 1,548–86,144 KB against a 48 KB
  threshold, so §8.E's granularity worry does not arise at donor scale.

Both were artefacts of reasoning at `D=256, h=128`. This is the second time this session that a sandbox
constant has misled a donor-scale conclusion, which is itself worth recording as a pattern.

> **The correct next measurement, and the one I am now requesting: the ternary-LUT kernel rate at donor
> *projection* dimensions.** It is the largest unmeasured term in the whole budget — flagged UNMEASURED in
> every row that uses it — and the 17.0-vs-11.4 GB/s question it settles is **worth 2–4× on every row and
> is the entire difference between one donor passing and none.** Zero GPU, no donor, no model: the
> existing `--kselftest` harness with a `-D` recompile.

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
   donor, and measure **both** the local per-layer error **and** the end-to-end effect on this same donor.
2. That yields a **local-error → end-to-end-effect transfer curve for this donor**, measured, not assumed.
3. **Only then** set the local threshold, from the curve.

**Concrete specification — previously left as a sketch, which the Controller flagged as not runnable.**
This doubles as the replacement for control 2 (§3.7.2), whose original lesion may have been a
known-*negative*:

- **Perturbation operator:** additive Gaussian noise on the layer's *output activations*,
  `y' = y + ε·σ_y·n`, with `σ_y` the per-channel activation std measured on the calibration set and
  `n ~ N(0,1)` at a fixed recorded seed. Chosen over weight-quantization coarsening because it is
  **continuous, monotone and dimensionless in ε**, so the ladder is a clean sweep rather than a set of
  discrete quantizer configs (which would reintroduce §3.7b's many-variables problem).
- **ε grid — geometric, 8 rungs:** `ε ∈ {0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256}`.
  Geometric because the effect is expected to be roughly power-law in ε and a linear grid would waste
  most rungs in one regime. The grid deliberately **straddles the expected detection floor** — the bottom
  rungs should be indistinguishable from zero and the top rungs unmistakable. **If they are not, the grid
  is wrong and gets re-centred *before* any conversion arm is scored**, never after.
- **Which layers:** the ladder runs on **at least 3 layers spanning depth** (early / middle / late).
  Sensitivity is known to vary with depth, and a transfer curve fitted at one depth is not a curve for
  the network.
- **Read-out:** for each (layer, ε), report local relative output error **and** ΔBPB, each with the
  §3.6b bootstrap SE. The deliverable is the fitted **local-error → ΔBPB** transfer function.
- **The detection floor is the headline output:** the smallest ε whose ΔBPB exceeds `t·SE`. **This is the
  instrument's sensitivity, measured rather than asserted** — and it is what makes any later "arm (b) is
  within tolerance" statement meaningful instead of decorative.
- **Failure response, pre-specified:** if **no** ε in the grid produces a ΔBPB above `t·SE`, the
  instrument cannot detect damage to a single layer at all. That is **not** a licence to proceed — it
  voids every per-layer fidelity claim in §5, and the correct response is to fix the instrument (more
  eval tokens, more replicates, a more sensitive read-out), not to widen the grid until something moves.
- **Anchor rung:** at least one rung must perturb a **known non-redundant structure**, so the ladder is
  tied to a lesion whose effect is expected on independent grounds rather than resting only on the noise
  sweep.

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

**Open items — status:**
1. ~~Controller review of the stage −1 pre-registration~~ — **DONE: verdict NO, 7 BLOCKs.** §3 rebuilt
   (§3.6, §3.6b, §3.7, §3.7b). Needs re-review before it may go to the Owner.
2. ~~The Builder's per-model filter and stage-1 table~~ — **DONE (§4.2).** Independent Controller audit in flight.
3. ~~The `run_expert_rate` sweep~~ — **WITHDRAWN, I was wrong (§4.3).** Replaced by: **the ternary-LUT
   kernel rate at donor projection dimensions.** Zero GPU, existing `--kselftest` + a `-D` recompile.
4. **Second independent verification of PT²-LLM / TWLA** — attempted, agent lost to a session limit
   after completing the research but before writing. **Resumed.** Blocks sealing §3.3.
5. Re-price the recall tier as a **critical-path** dependency, not a stage-4b side item (§2.4).

**Still not requested:** any GPU session, any spend, any sealed-constraint amendment, any merge.

### 7.1 The decision that is now forming — TWO independent blocking findings

Stage 0 has produced two blocking results on **different axes**, either of which is sufficient on its own.
Neither has been run; both come from arithmetic on measured constants and from primary-source reading.

**Blocker A — speed.** Zero of 18 donors pass the sealed ≥10 tok/s gate. Best is 9.76 tok/s on
`Qwen2.5-1.5B`, missing by 2.4%, and only at 32K — below the sealed 128K contract. *(Controller audit in
flight; not established until it lands.)*

**Blocker B — quality, and this one is the harder of the two.** There is **no verified calibration-only
ternary method at 1–3B at all**, and the best verified closed-form ternary result at *any* scale is
**2.11× perplexity** (PT²-LLM, LLaMA-2-7B). Against S4's ≥90% retention that is not close. The two
published methods' smallest models are both 7B; the one 1–3B ternary datapoint that exists loses 40–58%
relative and requires ~32 A100-GPU-hours of gradient training, which S1 forbids (§2.7b).

**Blocker B matters more because it is not fixable by engineering.** Blocker A has a named,
zero-cost measurement that could move it 2–4× (the LUT rate at donor projection dimensions). Blocker B is
a statement about what the literature can currently do, and no amount of engine work changes it.

**The mandate anticipated exactly this and pre-specified the response** (§9, stage −1 row): *"If the
pre-registered PTQ fidelity gate fails, stop the ternary-primary route and ask the Owner whether a named
4-bit/mixed-precision engine path is worth pursuing."* We have reached that decision point **from the
literature alone, having run nothing** — the cheapest possible route to it, and precisely what stage 0
exists for.

**Therefore the decision I will bring, once the Controller audit lands:**

> **AMENDMENT REQUEST under §1.1(3), against the ternary-primary assumption**, not against the conversion
> recipe. The question for the Owner: **is a named 4-bit or mixed-precision engine path worth pursuing?**
> Best verified 4-bit is QuaRot W4A4 at ≤0.47 PPL loss / ~99% retention — comfortably inside S4 — against
> ternary's 2.11× at 7B and nothing verified at all below it.

**The cost of that amendment must be stated honestly and I will state it:** a 4-bit path forfeits part of
the project's measured faster-as-bits-drop kernel advantage, which is one of its genuine claims to
novelty. But §2.3 now shows the LUT path is **compute-bound, not bandwidth-bound**, which means the
bits-per-weight lever was already worth less than the project's framing assumed. **Those two facts should
be weighed together, not separately** — that is the real content of the decision.

**What still needs measuring even under the amendment:** ternary at 1–3B has to be *measured, not cited* —
no paper covers that regime. So a re-scoped stage −1 (bit-width curve, §3.4) retains its value; what
changes is that it is no longer a pass/fail on ternary but a **search for the knee** in a curve whose
ternary end is now known to be bad.

**I am not making this call.** Per §7.3 it is the Architect's and the Owner's, and it is a sealed-constraint
amendment, which §9's stopping rules reserve for a human decision.

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
| 15 | **zero of 18 donors pass the sealed gates** | Builder, 18 real configs, tool-generated | **Controller audit IN FLIGHT** | best misses by 2.4%, at 32K not 128K | §7.1 — NO-GO vs amendment |
| 16 | footprint is not the wall | Builder: 14/18 fit SKU-A, 18/18 SKU-B | Controller checking whether KV-at-context was in the footprint | if KV omitted, fits are overstated | audit |
| 17 | hybrids satisfy S3 and still fail | Builder: 8–25% attention, 1.62–3.87 tok/s | Controller checking Zamba2 defect containment | **if the +164% formula defect is broader, these are unsound** | audit |
| 18 | **dispatch overhead is ≤1.1%; ρ-granularity moot** | Builder at real donor dims | — | retires my own prior ask — sandbox constants misled me twice | ask withdrawn (§4.3) |
| 19 | v1 stage −1 gate was gameable | Controller prereg review, 7 BLOCK | — | absolute ΔBPB gate + INCONCLUSIVE band now | **re-review required** |
| 20 | PT²-LLM / TWLA verification | agent lost to session limit post-research | **resumed** | still [L?]; §3.3 unsealed | blocks stage −1 |
