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

---

## 7. OWNER RULING — 2026-08-20

### 7.1 S3 amendment: **APPROVED**

> SKU-B is approved to use the bounded-native + recall-tier contract, identical to SKU-A.

**S3 is hereby amended.** SKU-B no longer requires 128K *native* attention. The 128K **user-visible**
contract stands for both SKUs; the S4 retention gates, including the 128K long-context gate, are unchanged
and must still be passed through the recall path.

**Two additional owner directives attached to the approval, both binding:**

1. **"Evaluate matching the native context capacity natively offered by the donor model where feasible."**
   → The native window is no longer a free parameter to minimize. For each donor, the native window is
   evaluated **against that donor's own advertised native context**, and a shortfall must be justified,
   not assumed. This directly repairs the defect I found in §1.1: the stage-1 tool never read
   `max_position_embeddings`, so it scored a 4096-position model at 32K and 128K. **That field now becomes
   a required input**, and every context figure is reported against the donor's own native capacity.
2. **"We do not accept a structural NO-GO caused solely by full-native KV cache allocation overhead."**
   → A donor may **not** be eliminated because full-native KV does not fit or does not stream in budget.
   That is precisely what the recall tier exists to absorb. Any elimination whose sole cause is
   full-native KV allocation is **void** and must be re-run through the bounded-native + recall path.

**Consequence, and it is the significant one:** the recall tier is now **load-bearing for both SKUs and on
the critical path**. It is today measured standalone only (29.05 µs/token at 128K) and has never been
fused with the engine core. §9's stage 4b stops being a side item and becomes a precondition for any 128K
claim on either SKU. I flagged this as the price of the amendment before it was granted; it is now owed work.

### 7.2 Microbenchmark: **APPROVED**

CPU-only proj-GEMV / ternary-LUT rate at donor matrix dimensions, authorized.

**Owner operational condition, binding on every timing figure this project produces:** the reference
machine is shared with a human, so latency/throughput runs require a quiet environment — no
memory-heavy or high-load background applications.

**Compliance check run before any timing (2026-08-20): the machine was NOT quiet.** Observed: Spotify,
multiple Chrome processes, **two `webwallpaper32` processes** (animated wallpaper = continuous load),
FanControl, a ChatGPT client. **Timing was therefore withheld.** The Builder is authorized to build,
validate correctness against a float64 reference, and exercise planted controls — all of which are
load-insensitive — and any timing collected during shakedown is marked
**PROVISIONAL — DIRTY MACHINE, NOT A RESULT**. The timed run is queued behind the owner quieting the box.

### 7.2g THE LAST BRACKET IS CLOSED — and the conversion is the whole difference

**Independent Controller audit in flight (§7.1 two-key rule); not established until it lands.**

> **Engine-integrated ternary-LUT at donor dims: 27.31 ± 0.46 ms/token = 21.25 ± 0.36 GB/s**
> (D=1536, ffn=8960, L=28, skip-off; 4 runs, between-run sd 1.7%).

This closes the [20.99 .. 50.91] ms spread that §7.2f identified as the only thing still deciding the gate.
**Integration overhead at donor D is +30.5%, not the 2.4× the bracket assumed.**

**Two findings about our own banked data, both consequential:**
1. **`PHASE64_BUDGET.md` §1's `LUT-MLP 313→207` is the `--skip on --exp fast` (E3.5) configuration, not the
   default.** Running E3.5 today reproduces the *whole* §1 decomposition (scan 36.5 vs 46, proj 222.8 vs
   272, SWA 48.5 vs 65, MLP 166.3 vs 207). The published table does not say this. *(Note the reproductions
   sit 15–25% **below** the published figures — the auditor is checking whether that gap has an
   explanation.)*
2. **`DENSE_LUT_GBS = 11.398` is doubly misattributed** — integrated at D=256 **and** derived from the
   **dReLU row-skip path, which a SiLU-gated donor cannot use** (dReLU leaves only 25.4% of rows live).
   It over-charges by **1.86×**. Crediting a donor with sparsity it does not have would have been a
   textbook version of this project's characteristic error.

**The n_att curve — the deliverable §8.B has been asking for:**

| retained attention | 4-bit KV *(unbuilt)* | fp16 KV *(built)* |
|---|---|---|
| **windowed (SWA-128)** | PASS at every n_att 0→28 | **PASS for n_att ≤ 18 of 28** |
| **full attention @32K** | **PASS for n_att ≤ 2**; FAIL from 3 | PASS for n_att ≤ 2; FAIL from 3 |

> **Retained full attention must be ≤ 2 of 28 layers. Windowed, the budget is 18 of 28.**
> **At S3's sealed "minority" point (n_att = 14, windowed) the donor passes on BOTH KV precisions —
> including the one actually built: fp16 KV = 11.97 tok/s [10.67 .. 12.68], lower bound clears.**

The Builder did **not** price full attention by scaling the window-128 anchor (a 256× extrapolation that
would have said one full-attention layer kills the donor); it used a first-principles compute model,
**flagged as a desk model**. KV now also scales with retained attention layers — converting a layer to SSM
removes its KV, which the model previously ignored.

**The 2×2, with all five terms present:**

| MLP constant | 4-bit KV *(unbuilt)* | fp16 KV *(built)* |
|---|---|---|
| retired 11.398 | 9.40 **FAIL** | 8.12 **FAIL** |
| **measured 21.25** | **12.09 [10.36..13.01] PASS** | 10.05 [8.76..10.77] **STRADDLE** |

*(This 2×2 is at full attention on all 28 layers. The n_att table above is where the donor is actually
deployed, and there it passes on fp16.)*

> ### The single most important number in the programme
> **Unconverted `Qwen2.5-1.5B`, full attention on all 28 layers: 4.70 tok/s — FAIL.**
> **The conversion is the entire difference between failing and passing the gate.** Every shortlist figure
> is therefore reported at the *converted* configuration and labelled as such. This is the project's
> founding thesis appearing as a measured number rather than an argument.

**Control integrity, reported by the Builder against itself:** its first control **did not fire, and it
says so** — it compared against a float64 reference whose 5.5e-08 rounding floor masked everything, and
corrupted a row dReLU had already zeroed. Rebuilt to compare **bit-exactly against the pristine fp32
output** and target the scale-setting active row: N=0 silent → **N=1 fires** (1 code in 6,881,280);
scale δ=0 silent → **1 ULP fires**; half-tile zeroing caught. **A dead control found and fixed by its own
author is the system working.** The auditor is re-running all three.

**Known-positive honesty:** the MLP check is weaker than the projection one and the Builder states it
plainly — harness 103.1 µs vs live engine 90.8 µs = **+13.5%**, against ±5% for proj. It argues the
sandbox MLP is only ~90 µs of work and the donor point is 300× larger and 11× tighter. **The auditor is
testing whether +13.5% is acceptable for a number carrying this much weight.**

### 7.2f AUDIT VERDICT on the PROJ measurement — STILL UNDECIDED at that time; §7.2g closes the bracket it named

**The independent audit (`CONTROLLER_PROJRATE_AUDIT.md`) confirms the measurement and refutes the
inference drawn from it.** Its verdict: *"The measurement is sound. The inference from it is not."*
**Do not report the gate as cleared.** I did, and that was wrong.

**What survived every attack (and one result is better than reported):**
- **C1 PASS.** The auditor ran `engine.c --gemv-sweep` itself: at 24 MB the published table says 60.5, the
  original routine returns **179.2 and 206.2**. The published mid-region is not reproducible **by the code
  that produced it**, with no Builder code in the path — two binaries, two source files, no shared code, so
  the "harness wrong in a shared way" hypothesis is excluded *by construction*. No donor row touches those
  points.
- **C2 PASS.** Byte count re-derived from the config alone: **1,550,057,472 B, exact.**
- **C4 PASS, better than claimed.** The auditor measures **39.87 ± 0.09 GB/s** fullstack where the Builder
  reported 37.74; asymptote 39.18 vs 38.84. **The Builder under-reported.**
- **C5 PASS.** No warm-buffer artefact: 39.87 is **95% of the DRAM ceiling**, and the same harness *does*
  report 150–200 GB/s at 4–24 MB — it can show a warm buffer and is not showing one.
- **C6 PASS.** `ktrunc` fires (8.925e-03 against a 4.526e-08 threshold).
- **C8 PASS, replicated more strongly.** LUT bandwidth-bound at donor D: **96.40 → 23.06 GB/s, 4.2×**,
  kernel shape fixed and internally controlled. Controller #2's compute-bound reading is confirmed
  **scoped to D=256**.

**Why the gate is NOT cleared — two BLOCKs:**

**C11 — "34.0 was the sole source of the FAIL" is FALSE, and I repeated it.** Hold proj at 34.0 and move
only the *MLP* constant to kernel-pure → **13.80 tok/s, PASS**. Reverse it: the measured 37.74 with the
tool's MLP charge and **fp16 KV, the only precision actually built** → **8.66 tok/s, FAIL**.
> **The 10.22 PASS lives in exactly one cell of a 2×2: {4-bit KV (unbuilt)} × {integrated MLP}.**
Three constants each flip this verdict, and the one we measured was **the smallest lever**.

**C12 — the model omits terms larger than the margin.** `time_model()` charges proj + LUT + KV + glue.
`PHASE64_BUDGET.md` §2 lists **five** components: **scan-recurrence (0.86–1.72 ms) and SWA (0.39 ms/layer)
are absent entirely.** Headroom above the gate is **2.13 ms**. Scan + 2 SWA layers → **10.006**; scan + 4
→ **9.93, FAIL.** A model missing terms bigger than its own margin cannot adjudicate the gate.

**C10 — the remedy I prescribed was a no-op.** `PROJ_STREAMED_FLOOR := 37.74` leaves `r_proj(1.55 GB)`
untouched, because the >96 MB branch returns the asymptote constants and never reads the floor. The
Builder's independent rewrite (retiring the floor, installing 38.84/[36.0, 40.20]) was the correct fix;
my prescription was wrong.

**Corrected status: the sealed ≥10 tok/s gate is STILL UNDECIDED**, high confidence, trending pass.
**Next, and it is now #1 rather than #4: the engine-integrated ternary-LUT rate at donor dims.** The proj
term is effectively a point ([41.48..40.70] ms); the **MLP term is [20.99..50.91] ms — a 30 ms spread
against a 2.13 ms margin.** One measurement collapses it and decides `OLMoE` outright.

### 7.2c MEASURED — the constant is 38.84 GB/s *(headline WITHDRAWN by §7.2f above; retained for the record)*

**Full measurement complete (`DONOR_PROJ_RATE.md`). Independent Controller audit in flight per the §7.1
two-key rule — this flips a sealed-gate verdict, so it is not established until that lands.**

> **Asymptote past 96 MB: 38.84 ± 0.68 GB/s (t6)** — 36 measurements, 9 sizes ≥64 MB × 4 runs, sd 1.7%,
> **flat across a 16× range of size** (128→1024 MB), no further cliff.
> **Real donor stream** (Qwen2.5-1.5B, 28×(q,k,v,o)+head walked in layer order):
> **37.74 ± 0.18 GB/s = 41.07 ± 0.19 ms/token**, byte accounting **exactly** 1,550,057,472 B — matching
> the audit's independent hand-derivation.
> **Substituting only this constant: 10.22 tok/s. The sealed ≥10 tok/s gate is CLEARED.**

`PROJ_STREAMED_FLOOR = 34.0` was **11% low** and was the **sole** source of the FAIL. Neither 34 nor 36:
**~38.8**.

**The known-positive gate FAILED as pre-stated (5/8 points), and I am not waving that away.** The Builder
reported it as a FAIL rather than widening the band, which is why the rest is credible. Its reconciliation:
- it rebuilt and ran **`engine.c --gemv-sweep` itself, unmodified** — the routine that produced the
  published table — and **the two agree at 7/8 points while both disagree with the published table in the
  same places**. The harness is the instrument.
- the 16–32 MB region swings **up to 4.6× on `OMP_PLACES` alone**, and the published run's thread
  placement was **never recorded** — so **that region of the repo's own §1b table is unreproducible by
  anyone**, and should be annotated as placement-conditioned. That is a finding about our banked data.
- at **96 MB** — the last published point and the **sole anchor of the "34–36" asymptote** — it
  reproduces within **4.9%**, and placement spread there is only 2%. **No donor row depends on the five
  non-reproducing points.**

**Two of my own reported claims are falsified by this measurement:**

1. **The skinny-matrix and huge-head worries are dead.** I argued (§2.6) that the head was "not
   first-class, it is *first*", and worried that non-square organs would not follow the curve. Measured:
   **once streamed, shape stops mattering** — skinny K/V (256×1536), square Q/O, 52.5 MB MLP slabs and the
   **890 MB head all land in 36.0–39.0 GB/s**. The head is still a large *byte* cost; it is not a special
   *rate* case. Also: **resident-region rates (143–161 GB/s) are 4× misleading and must never be applied
   to donor-scale organs.**
2. **Controller #2's "the LUT path is compute-bound by ~16×" is a D=256 artefact and does not transfer.**
   I reported that as established, including to the Owner. Measured at donor D: **bandwidth-bound** —
   fixed kernel shape, working set 1.1 MB → 512 MB, rate falls **3.6× (96.4 → 27.3 GB/s)**, which a
   compute-bound kernel cannot do. The kernel ceiling is **29 GB/s at D=256 but 97 GB/s at D=1536**.
   Controller #2's `run_expert_rate` replica *did* reproduce (7.6–8.2 vs 7.45; 18.6–18.9 vs 17.0), so its
   measurement was sound and only its **generalization across D** was wrong.
   **Consequence: the bits-per-weight lever is worth MORE than §3 concluded**, which strengthens the
   ternary route and weakens my own argument in §3 that a 4-bit path forfeits little.

**Controls fired, including one that matters and one honest negative:**
- **`ktrunc`** — a plausible-looking kernel that reads half of each row, writes every output, and would
  have **inflated the rate 143→302 GB/s (2.11×)** — was caught at 8.9e-03. That is precisely the
  "plausible artefact" class this project's laws exist for, and the instrument caught it.
- **Honest negative, reported rather than dressed up: the fp32 comparator is BLIND to a 1-ULP
  corruption** (~1e-9 relative against a 4.5e-9 noise floor). Substituted a float64 reference, which sees
  it at exactly `dw·x` with zero cross-talk, plus a sensitivity ladder giving a real detection floor of
  **δ=1e-4 on one element in 2,359,296**. The Controller is auditing whether that substitution suffices.
- **32 MB flagged non-reportable** (between-run sd 22.2%, bimodal 118–170 — the L3 cliff) and used for
  nothing. Streamed region 0.4–2.5%; full donor stack **0.5%**.

### 7.2d The corrected stage-1 table — one donor passes, and the caveat is load-bearing

The bracket collapse at `donor_inventory.py:669` is **withdrawn**. Every term now carries
`(slow, fast, central)`; the total is the sum of centrals with a **fully-correlated** interval
(deliberately wider than RSS — the terms share one memory system). Central time is the midpoint **in
time, not in rate**, so it cannot quietly favour the fast end. **`PASS` now requires the lower bound to
clear 10**; a central-only clearance reads `STRADDLE`.

**SKU-A, each donor at its own native context, 4-bit KV:**

| donor | native ctx | central | interval | KV-free | verdict |
|---|---|---|---|---|---|
| **`Qwen2.5-1.5B`** | 131072 | **12.10** | **[10.17 .. 14.92]** | 12.98 | **PASS** |
| `OLMoE-1B-7B` | **4096** | 10.08 | [7.16 .. 17.04] | 10.42 | **STRADDLE** |
| `SmolLM2-1.7B` | **8192** | 9.95 | [8.54 .. 11.91] | **11.00** | **FAIL (KV-only)** |
| `Qwen3-1.7B` | 40960 | 7.96 | [7.09 .. 9.07] | 9.69 | FAIL |
| *(14 others)* | — | 5.11 → 1.04 | — | — | FAIL by 2–12× |

**The lower bound clears 10.** 10.17 is the *fully pessimistic corner* — ternary MLP at the D=256
integrated rate, KV at 40 GB/s, glue at 2×. **The gate is therefore cleared without winning any open
argument**, which is the strongest available form of this claim.

**Eliminations that changed:** `Qwen2.5-1.5B` FAIL→PASS (sole cause: the measured proj rate, 9.77→12.10).
`OLMoE` FAIL→STRADDLE (it is a **4096-position model** that rev A scored at 32K/128K — the
`max_position_embeddings` defect I found, now fixed). `SmolLM2` re-opened per the owner's no-KV-elimination
directive — 9.95 with KV vs 11.00 KV-free — but its KV-free *lower bound* is 9.34, so it re-enters
**undecided**, not passing. Phi-3, OLMo-2, Zamba2 and Nemotron-H all improve at native context and **all
still fail by 2–4×**.

### 7.2e THE CAVEAT THAT MUST TRAVEL WITH THE HEADLINE

> **4-bit KV does not exist in this engine.** At **fp16 — the only KV precision actually built —
> `Qwen2.5-1.5B` gives 10.06 [8.62 .. 12.04]: a STRADDLE, not a pass.**

So the honest statement is: **one donor passes on the KV precision the architecture assumes, and straddles
on the KV precision the engine has.** Reporting "the gate is cleared" without this sentence would be the
kind of scale-caveat-stripping §0 of the mandate warns about.

**Consequences, and they are actionable:**
- **Building 4-bit KV is now the cheapest engineering that would settle the gate**, and it is ordinary
  work with a clean correctness contract.
- **SKU-B/128K weakens from "clear fail" to STRADDLE (10.06)** — my §2 amendment request is *dented, not
  overturned*, and I record that against my own earlier framing.
- **The remaining uncertainty is now almost entirely the ternary bracket.** For `Qwen2.5-1.5B` the proj
  term is [41.48 .. 40.70] ms — effectively a point — while the MLP term is **[20.99 .. 50.91] ms**. The
  measurement collapsed one bracket and left the other as *the* open question.
  **One engine-integrated LUT measurement at donor D would decide `OLMoE` outright** and is the obvious
  next cheap step.

**A defect the Builder found in its own tool, worth recording:** the dispatch-overhead term was
**double-counted** — added on top of a 4.2 GB/s rate that is engine-integrated and already contains it.
The document's own §1.3 had flagged this in prose while the code did it anyway. That is the same
prose-says-one-thing-artefact-says-another failure that corrupted my §3.7c audit trail, appearing
independently in a second artefact. It is now added only at the kernel-pure end.

### 7.2b PROVISIONAL first reading — superseded by §7.2c above, retained for the record

The authorized microbenchmark ran on a verified-quiet machine (CPU load 10%, 55.5 GB free, the blocking
`webwallpaper32` processes gone) and returned a first figure before the agent was cut off by a session
limit:

> **37.7 GB/s, 41.07 ms/token for the full per-token stream, byte accounting exact.**

**Why this matters:** §1's UNDECIDED verdict lives entirely inside **[34, 40] GB/s**. The stage-1 tool
hardcoded **34.0** and produced "zero donors pass". At **36** the best donor already passes (10.01 tok/s).
**37.7 sits above that crossover** — so on this first reading `Qwen/Qwen2.5-1.5B` **passes** the sealed
≥10 tok/s gate at SKU-A, and the "zero donors pass" headline is dead twice over: once by the audit's
bracket-collapse finding, once by measurement.

**This is PROVISIONAL and may NOT be quoted as a result.** It has not yet cleared:
1. **the known-positive precondition** — the harness must first reproduce the repo's own r(size) curve at
   4–96 MB (187/185/134/60.5/55.7/45.5/45.3/36.5 GB/s at t6). Until it does, no new point it produces is
   trustworthy, and that check had not run when the agent stopped;
2. **per-shape run-to-run variance** (>5% ⇒ the shape is not reportable — 10% background load is low but
   not zero);
3. **the per-organ shapes**, including the skinny ones (K/V at 1536×256) and the head at ~933 MB fp32,
   which is far past any previously measured point;
4. **the ternary-LUT compute-bound-vs-bandwidth-bound question** at donor D;
5. **an independent Controller check**, per the §7.1 two-key rule — this figure would materially inform a
   sealed gate, so one author report does not establish it.

**Resume and complete before this changes any verdict.** Recorded here only so a session-limit kill does
not lose it.

### 7.3 Reference hardware, recorded (Portability law: these are runtime parameters, not targets)

| | |
|---|---|
| CPU | AMD Ryzen 5 3600X (Zen 2, AVX2/FMA, **no AVX-512, no VNNI**), L3 16 MB/CCX × 2 |
| RAM | 80 GB DDR4 |
| GPU | NVIDIA RTX 3060 + NVIDIA GTX 1660 |

**Discrepancy I must flag before stage −1 runs:** the installed PyTorch is a **CPU-only build**
(`torch 2.12.0+cpu`, `torch.cuda.is_available() == False`). The RTX 3060 exists but is **not reachable**
from the current environment. Stage −1 is costed at ~12–20 h of CPU on that basis.

**This is an opportunity, not a blocker.** A 1.5B donor fits an RTX 3060 comfortably, and the calibration
and evaluation passes are exactly the workload a GPU shortens — plausibly from ~12–20 h to ~1–2 h, and it
would free the CPU that the owner also uses. It requires installing a CUDA-enabled torch build.

**Two cautions, which is why I am asking rather than doing it:** (a) changing the torch build changes the
numerical environment, and this project gates on determinism — any device move must be declared in the
pre-registration and re-validated, not slipped in; (b) the GTX 1660 is Turing without usable bf16, so
dtype policy must be stated per device. **The Controller has been asked to flag any device-dependent
result path.** I will not change the environment without an explicit go.

### 7.4 Stage −1 and stage 2a: proceeding, with one internal gate still to clear

The owner directed proceeding with both per the pre-registered roadmap. **One project-internal gate
remains and it is not the owner's to waive:** the stage −1 pre-registration was returned **NO with 7
BLOCKs** by Controller review; I rebuilt §3 in response, and §7.2 of the mandate requires the Controller
to review a gate-bearing apparatus **before** the run it governs. That re-review is running now. It is
cheap and fast, and running a 12–20 h protocol that a reviewer has already rejected once would waste the
owner's machine, not save it.

The §3.3 method premise also changed after that review (`PTQ_SOURCE_VERIFICATION.md`), so the re-review is
looking at a genuinely different document, not the same one twice.
