# Stage-0 Decision — the adjudicable statement required by §1.1

**Status: the Adapter's formal §1.1 output, 2026-08-20. Branch `research/donor-adaptation`.**
**Nothing in this programme has consumed a GPU-hour or a euro. No donor weights were downloaded.**

§1.1 requires the Adapter to end with one of three adjudicable statements. This document issues them.
**Issuing an amendment request is the Adapter's job; granting one is the Owner's.** I previously deferred
the whole decision to the Architect, which was a misreading of my own role — §1.1(3) says "state the
exact amendment required", and that statement is made here.

Evidence base: `DONOR_STAGE1_ARITHMETIC.md` (18 real configs), `CONTROLLER_STAGE1_AUDIT.md`,
`CONTROLLER_AUDIT_MANDATE.md`, `CONTROLLER2_REPLICATION.md`, `CONTROLLER_PREREG_REVIEW.md`,
`PTQ_SOURCE_VERIFICATION.md`, `DONOR_PRIOR_ART_SURVEY.md`. Two-key rule satisfied on every load-bearing
claim below.

---

## 0. Summary of the three statements

| # | on | statement |
|---|---|---|
| **1** | the ≥10 tok/s gate (S4) | **NOT a NO-GO — UNDECIDED.** One donor straddles the gate; the verdict turns on a single constant in [34, 40] GB/s that a sub-hour, zero-GPU microbench resolves. **Measure before deciding.** |
| **2** | the 128K native context contract (S3) | **AMENDMENT REQUEST under §1.1(3).** S3's SKU-B "128K native" and S4's ≥10 tok/s are jointly unsatisfiable for every donor examined. Exact amendment in §2. |
| **3** | the ternary-primary route | **Pre-authorized question raised early**, per §9's own stage −1 row — reached from the literature at zero cost, without running the stage. §3. |

**17 of 18 donor eliminations are sound.** The programme is not dead; it is blocked on one sealed
constraint, undecided on one gate, and holding a route decision that the mandate already anticipated.

---

## 1. On S4's ≥10 tok/s gate — I withdraw "zero donors pass"

**I reported that zero of 18 donors pass the ≥10 tok/s gate. The Controller audit refutes that and I
withdraw it.** The Builder's arithmetic was correct — its parameter formula reproduces `Qwen2.5-1.5B` to
the **exact byte** (1,543,714,304, zero delta against advertised) and every term of the 102 ms
decomposition re-derives by hand. The defect was not a number; it was a **rate choice**.

`donor_inventory.py:669` builds the MEASURED-ONLY column — the column the document tells the reader to
read as the verdict — as the **pessimistic corner of every bracket simultaneously**:

```python
meas = t_proj_pess + t_lut_meas + glue_us + (t_kv_pess or 0.0)
```

`PHASE64_BUDGET.md` §1b/§2 state the streamed-projection bracket as **[34–40] GB/s**; the tool's own
source comment records `[34-40]` and then hardcodes `PROJ_STREAMED_FLOOR = 34.0`. §1's *measured,
engine-integrated* projection datum is 11 MB / 272 µs ≈ **40 GB/s** — the top of that bracket, not the bottom.

| reading | `Qwen2.5-1.5B` | verdict |
|---|---|---|
| 34.0 GB/s (tool's hardcoded floor) | 9.76 tok/s | FAIL |
| 36 GB/s (top of the tool's own narrowed asymptote) | **10.01 tok/s** | **PASS** |
| 40/44 GB/s with the F2 scales correction | **10.65 tok/s** | **PASS** |

> **Corrected statement: `Qwen/Qwen2.5-1.5B` at SKU-A / 32K native is `9.77 – 10.65 tok/s`. The gate is
> UNDECIDED, not FAILED.** A bracket was silently collapsed to its worst edge and that point was reported
> as a gate verdict.

The Controller confirmed this with the Builder's own instrument — a scratch copy with three constants
moved to the other end of their brackets reproduces the entire table with **exactly one row changed**
(second-best moves 5.98 → 6.32). Nothing else moves. **The other 17 eliminations fail by 2–15× and are sound.**

**Consequence: no NO-GO is warranted on S4, and it would be wrong to amend a sealed gate on this
evidence.** What is warranted is the measurement in §4.

### 1.1 A second defect, unflagged by the Builder, that I must report

The tool **never reads `max_position_embeddings`**. The #2-ranked donor, `OLMoE-1B-7B`, is a
**4096-position model** and was scored at 32K and 128K. Its ranking is void until re-scored. This does not
change any verdict — it fails anyway — but it is exactly the class of silent defect §6 exists to catch,
and it means the shortlist's *ordering* is not yet trustworthy even where its *eliminations* are.

---

## 2. AMENDMENT REQUEST under §1.1(3) — S3's SKU-B 128K native contract

**This is the one place where the evidence says a *sealed constraint*, not the conversion recipe, is what
prevents success.**

### 2.1 The evidence

At 128K, attention re-reads the entire KV cache every token. This is a **bandwidth** term, and neither §5
nor my own v1 analysis priced it. For a 30B-class donor (GQA-4, 48 attention layers):

| | bytes | read time/token @35 GB/s | vs the 100 ms budget |
|---|---|---|---|
| KV fp16, all layers | 12.9 GB | **368 ms** | **3.7× over** |
| KV 4-bit, all layers | 3.22 GB | 92 ms | 92% — nothing left for weights |

And measured across the real shortlist: **no donor passes SKU-B / 128K native.** The best candidate,
corrected for the §1 bracket defect, is **9.01 tok/s at 128K** — below the gate. At 32K the same donor is
9.77–10.65. Every reported pass in the entire stage-1 table is at **32K**, and the 128K footprint fits
only **13 of 18** donors, and only with **4-bit KV that is not built**.

**The 128K contract alone breaks the 10 tok/s gate, before a single weight is read.** S3 and S4 are each
individually satisfiable and **jointly unsatisfiable** for every donor in the 1–100B range examined.

### 2.2 The exact amendment requested

S3 currently reads, sealed:

> *"SKU-B must provide **128K native context** within its 64 GB whole-process budget. SKU-A must provide a
> 128K user-visible context contract within 16 GB: native retained attention may use only a measured
> 8K–32K window, and the engine recall tier must supply the long-range path for the remainder."*

> **Requested amendment: extend SKU-A's bounded-native-plus-recall structure to SKU-B. Replace SKU-B's
> "128K native" with a 128K user-visible contract served by a measured native window plus the recall tier
> — the same contract already sealed for SKU-A.**

**Nothing else changes.** The 128K *user-visible* contract stands for both SKUs. The S4 retention gates
stand unchanged, including the long-context retention gate at 128K in S4/§8.I, which the amended SKU-B
would still have to pass through its recall path exactly as SKU-A must.

### 2.3 Why this amendment and not the alternatives

- **Not lowering S4's 10 tok/s.** The floor is the product requirement; weakening it would make a pass
  meaningless. And §1 shows S4 is not actually the binding problem — it is undecided, not failed.
- **Not dropping to a shorter user-visible context.** That would silently reduce the product.
- **This amendment is the minimal one, and it makes S3 internally consistent.** S3 already concedes for
  SKU-A that 128K cannot be served natively within budget; the arithmetic says the same is true at 64 GB,
  for the same reason (KV traffic, not KV footprint). SKU-B's "128K native" appears to have been written
  on the footprint argument alone.

### 2.4 What granting it costs — stated honestly, because it is not free

**The recall tier becomes load-bearing for both SKUs, which puts it on the critical path.** It is today
measured **standalone only** (29.05 µs/token at 128K, 1.69 MB searchable + 64 MB values) and has **never
been fused with the engine core**. §9's stage 4b is its readiness gate, and under this amendment that gate
stops being a side item and becomes a precondition for any 128K claim on either SKU.

**I am not hiding that this converts an engineering risk into a programme dependency.** It is the honest
price of the amendment, and the Owner should weigh it knowing that the alternative is a contract no donor
can meet.

### 2.5 If the amendment is refused

Then the adjudicable statement becomes **NO-GO under §1.1(2)** for the 128K contract on both SKUs, with
the mechanism localized (**KV read bandwidth at 128K, 368 ms/token fp16 / 92 ms 4-bit, against a 100 ms
budget**) and the closest measured result reported (**9.01 tok/s at 128K, 9.77–10.65 at 32K, on
`Qwen/Qwen2.5-1.5B`, SKU-A**). I will not proceed under the amendment unless it is granted (§1.1(3)).

---

## 3. The ternary-primary route — the mandate's own pre-authorized question, reached for free

§9's stage −1 row pre-specifies: *"If the pre-registered PTQ fidelity gate fails, stop the ternary-primary
route and ask the Owner whether a named 4-bit/mixed-precision engine path is worth pursuing."*

**We have reached that question from the literature alone, having run nothing.** That is the cheapest
possible route to it and precisely what stage 0 exists for.

Verified from primary sources (`PTQ_SOURCE_VERIFICATION.md`):

- The best **genuinely closed-form** ternary PTQ result at any scale is **2.11× perplexity**
  (PT²-LLM W1.58, LLaMA-2-7B; 5.24× on LLaMA-3-8B). The method is real, ICLR 2026, S1-compatible — its
  *quality* claim was the fabricated part, not its existence.
- **No paper covers 1–3B**, the regime we would run in. Both methods' smallest model is 7B.
- The one 1–3B ternary datapoint located loses **40–58% relative** and needs ~32 A100-GPU-hours of
  gradient training — S1-forbidden.
- **Best verified alternative is 4-bit**: QuaRot W4A4, ≤0.47 PPL loss, ~99% retention. **2-bit scalar
  collapses even at 7B** (GPTQ-W2 47.13, QuaRot-W2 19.97, vs FP16 5.47).

**Against S4's ≥90% retention, 2.11× is not close.** I therefore raise the mandate's own question:

> **Is a named 4-bit or mixed-precision engine path worth pursuing?**

**The cost of saying yes, stated because it is a real forfeit:** a 4-bit path gives up part of the
project's measured *faster-as-bits-drop* kernel advantage, which is one of its genuine novelty claims.
**But that advantage is worth less than the project's framing assumed** — Controller #2 established that
the LUT path is **compute-bound by ~16×, not bandwidth-bound** (the dense LUT path caps at 11.4 GB/s
while the same silicon does 185 GB/s on resident fp32 GEMV). Bits-per-weight buys footprint far more than
it buys speed. **Those two facts should be weighed together, not separately.**

**This is a question, not an amendment request** — ternary is an engine property, not one of S1–S4, and
§9 already authorizes asking it. I am not proceeding either way without an answer.

---

## 4. The one measurement that should happen before any of this is decided

> **Measure the proj-GEMV / ternary-LUT rate at donor projection dimensions.**
> CPU-only, zero GPU, no donor, no model, sub-hour. The existing `--kselftest` / `gemv_bench` harness with
> a `-D` recompile.

It resolves the single constant in **[34, 40] GB/s** on which §1's UNDECIDED verdict turns, and it is the
largest unmeasured term in the entire budget — flagged UNMEASURED in every row that uses it. **Measure it
before amending a sealed gate**, which is the Controller's explicit recommendation and mine.

Note this **replaces** the `run_expert_rate` sweep I asked for earlier and then withdrew: dispatch
overhead is ≤1.07% of any donor total and ρ-granularity does not arise at donor scale (real experts are
1,548–86,144 KB against a 48 KB threshold). Both were artefacts of reasoning at `D=256, h=128`.

---

## 5. What is closed, and what I am not claiming

**Closed on sound arithmetic:**
- **Route (iv) (prefer an already-hybrid donor) is closed as a *route*.** The hybrids do arrive
  S3-compliant (granite 10% attention, Nemotron-H 8%, Zamba2 17%, Qwen3-Next 25%) and every one still
  fails on absolute active parameters, by 6–10×. The Controller verified the containment of the one
  formula defect two independent ways: only Zamba2 carries the block-reuse vocabulary, and granite,
  Nemotron-H and Falcon-H1 cross-check at **−0.00%** against safetensors. **The binding quantity was never
  the attention fraction; it is absolute active params per token.**
- **The dense 1–100B band is closed** — 2.09 tok/s at 8B, confirmed three ways.

**Not claimed:**
- **No GO.** Nothing has been measured on a donor; no conversion exists; no retention number exists.
- **No claim that stage −1's gate has been run.** It has not. §3 reasons from the literature, and the
  re-scoped stage −1 (a bit-width curve to locate the knee) retains its value because **ternary at 1–3B
  must be measured, not cited** — no paper covers it.
- **The shortlist's ordering is not yet trustworthy** (§1.1, the `max_position_embeddings` defect), even
  where its eliminations are.

---

## 6. What I need from the Owner and the Architect

1. **Rule on the §2 amendment request** (S3 SKU-B: 128K native → bounded native + recall). Granting it
   puts the recall tier on the critical path; refusing it makes §2.5's NO-GO the statement of record.
2. **Answer the §3 question**: is a named 4-bit / mixed-precision engine path worth pursuing?
3. **Authorize the §4 measurement** — CPU-only, zero GPU, sub-hour, and it should precede both of the above.

**Not requested:** any GPU session, any spend, any other sealed-constraint change, any merge.
