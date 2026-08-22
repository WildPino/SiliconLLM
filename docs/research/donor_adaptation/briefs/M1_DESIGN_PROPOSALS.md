# M1 — design proposals: does FFN quality survive at donor granularity?

**Author:** the Builder · **Date:** 2026-08-22 · **Status:** DESIGN PROPOSAL, response to `BRIEF_M1_EXPERT_COUNT_SCALING.md`.
**No code was written and no training was run for this document.** Every number below is arithmetic over the
probe-4 source (`benchmarks/phase57/phase59_moe.py`, `benchmarks/phase57/phase57_sparse.py`,
`benchmarks/phase55/phase55_ssm.py`), verified against two independently documented figures — the formula
reproduces probe-4's stated "~22.5M matched total params" as 22.467 / 22.517 / 22.480 M and the foundation's
"8.3M" as 8.312 M. Cost figures are estimates and are labelled as such.

> **Sizing note (D5 revision, 2026-08-22).** The Adapter reports that the D5 sweep has completed and that its
> early reading suggests the ternary path may be **compute-bound at donor width, with a rate constant that may
> not transfer** — so the FFN active-fraction *target* may move. **No design in this document is sized against
> any tok/s figure from `ADAPTER_MEMO_01`**; every arm count, step count and hour is derived from training cost
> alone, and no throughput number is used as a justification anywhere below. This is sound because the M1
> question is orthogonal to the D5 outcome: whether an FFN partitioned `top-8 of 224` **retains quality** is a
> property of the model, not of the kernel that runs it.
>
> Two consequences the Adapter should weigh when pre-registering, though — neither changes the designs, only
> which of them is worth buying:
> 1. **If the active-fraction target moves, `a = 3.57 %` stops being the single point that matters and the
>    *curve* becomes the deliverable.** That argues for keeping the `α` riders (A4, A5) rather than trimming to
>    the two-arm minimum, because a re-targeted programme would otherwise have to buy the curve a second time.
>    Design 2 already carries them; the 62 GPU-h minimum-viable subset does not.
> 2. **M1's urgency is genuinely lower than the brief assumed** (the brief's §6 rests on the ledger rows D5
>    licenses). If GPU budget is contested this quarter, M1 is a defensible *defer* — but the §4 impossibility
>    result and the §9 router argument are already banked and do not expire, and neither costs a GPU-hour.

---

## 0. Bottom line, before the designs

Three things, and the first is the one I would want if I were you.

**(1) The brief's question is stated in coordinates that do not survive a change of scale, and in those
coordinates it cannot be asked at all at an affordable width.** `h = 128` is not a property of the method; it is
`d_model / 2` at probe-4's width and `d_model / 64` at donor width. Written scale-free, the donor point is
`r = 3.5`, `E = 224`, `k = 8`, and **that fixes the expert size to `d_model/64` at every width** — the pool size
and the expert size are the same knob. Reaching `E = 224` while holding `h = 128` *absolute* requires
`d_model = 8192`, which in this apparatus is a **6.56 B-parameter model** (§4). That is the impossibility result,
and it is real: no small-model A/B can test "224 experts of 128 neurons".

**(2) What *can* be tested, and is the thing the ledger actually depends on, is the active fraction — and here
probe-4's own result is closer to the donor than the brief's framing suggests, but stops 7× short.** Define
`α = (active FFN neurons per token) / d_model`. A stock dense FFN has `α = r = 3.5`. Probe-4's `dense-big → moe-gran`
was iso-total-params with a **4× cut in active FFN neurons** (4096 → 1024) and the MoE *won* by 0.0085 BPB. The
donor conversion is the same operation at **28×** (28672 → 1024). Probe-4 landed at `α = 4.0` — still *fatter per
token than a normal dense FFN*. The donor needs `α = 0.125`. **Probe-4 never went below dense-normal active
width; the donor asks to go 28× below it.** So M1 is not "does the advantage survive a bigger pool"; it is
**"how far down the active-width curve does the FFN hold, and does the curve break before α = 0.125".** Probe-4
measured one point on exactly that curve. M1 should extend it.

**(3) Recommendation: Design 2 (§6), a fixed-total-parameter 2×2 factorial on (pool size) × (active slice) with a
frozen-random-router twin, ~145 T4-GPU-h (~1.6 aggregate-account-weeks; a 62 GPU-h minimum-viable subset answers
the yes/no in 3-4 days).** Argued in §12. Probe-4's trainer is **reusable with changes** — one file, ~200 lines
of edits, no new apparatus (§11).

**On your two adversarial asks, directly:**

- **Can a small-model A/B answer this?** *The literal question, no — never, at any budget we have.* The
  scale-free question (`a = 3.57 %` active at `r = 3.5`, matched total params) — **yes**, at 60–150 GPU-h, with
  one residual gap that no design closes: absolute expert width (`h = 8` at `d_model = 512`, `h = 16` at 1024,
  vs the donor's 128). A tranche-2 width-doubling check bounds that gap by **one doubling out of four**. Anything
  M1 concludes about `h = 128` is an extrapolation and must be labelled one, permanently.
- **Will the router be the binding constraint at `E = 224`?** *I think it will be co-binding, and I think the
  Switch aux is the reason.* Probe-4 already measured routing ≈ i.i.d. in time at `E = 32`, and its own memo
  attributes that to the aux forcing uniformity by design. At `E = 224` the aux must keep 224 experts alive on
  73 expert-slots per expert per micro-batch, and the fixed-point of "keep all alive" at large `E` is close to
  random routing. **A collapsed-to-uniform router at `E = 224` would produce a quality number that measures
  "random block-sparse FFN at 3.57 % active" — a real number, but not the question.** The measurement that tells
  them apart is a **trained frozen-random-router twin** (§9). It is one arm, it is not optional, and it also
  retro-audits probe-4: probe-4 never ran it, so probe-4's win is not yet separable from "the MoE arm has a
  multiplicative input-dependent hidden gate the dense arm lacks", which probe-4's own memo lists as a candidate
  cause.

---

## 1. Scale-free coordinates (the frame the rest of this document uses)

For a gated FFN of width `d_model = D`, hidden `d_ffn`, partitioned into `E` experts of `h = d_ffn/E`, top-`k` routed:

| symbol | definition | meaning |
|---|---|---|
| `r` | `d_ffn / D` | expansion ratio — total FFN capacity |
| `a` | `k / E` | active fraction of the FFN (**exactly** `k/E`; this is the speed variable) |
| `α` | `k·h / D` = `r·k/E` = `r·a` | **active FFN neurons per token, in units of width** |
| `G` | `h / D` = `r / E` | granularity — expert size in units of width |

Given `D`, choosing `(r, E, k)` fixes everything else. Note `G = r/E`: **at fixed expansion ratio, expert size
and pool size are one knob, not two.** "Fix `h`, sweep `E`" is only meaningful if you also let `r` or `D` move.

| point | `D` | `d_ffn` | `E` | `h` | `k` | `a` | `α` | `r` | `G` |
|---|---|---|---|---|---|---|---|---|---|
| donor, 100B-class | 8192 | 28672 | 224 | 128 | 8 | **3.57 %** | **0.125** | 3.50 | 1/64 |
| donor, unconverted dense FFN | 8192 | 28672 | — | — | — | 100 % | 3.500 | 3.50 | — |
| probe-4 `moe-gran` (validated) | 256 | 4096 | 32 | 128 | 8 | 25.0 % | **4.000** | 16.00 | 1/2 |
| probe-4 `dense-big` (arm B) | 256 | 4096 | — | — | — | 100 % | 16.000 | 16.00 | — |
| probe-4 `dense-small` (arm A) | 256 | 1024 | — | — | — | 100 % | 4.000 | 4.00 | — |

Read the `α` column. Probe-4's validated MoE point puts **4× a full width of FFN neurons on every token**. The
donor point puts **one eighth of a width**. That is a **32× difference in the quantity the FFN active-fraction
target names**, and it is the whole of M1. The `E` axis in the brief's title is a consequence, not the variable.

Two further facts worth having in front of you:

- Probe-4's `dense-big → moe-gran` comparison is **already** an iso-total-params active-cut: 22.467 M → 22.517 M
  total, 4096 → 1024 active FFN neurons, **4× cut, −0.0085 BPB (the MoE won)**. That is the same operation the
  donor conversion performs, at 4× instead of 28×. So we have one measured point of the right curve.
- The donor's active FFN neurons per token is **1024** — numerically the same as probe-4's 1024, but drawn at
  32× the width.

---

## 2. The confound the brief names, and the two it does not

**Named (parameter count).** Raising `E` at fixed `h` raises `d_ffn` and hence total params. Handled in all
designs below by a **fixed-`d_ffn` reparameterisation**: the FFN's total size never moves; only its *partition*
and the *number of blocks selected* move. Total params are then constant across the whole sweep by construction,
and **one dense control serves every arm** — a strictly stronger discipline than probe-4's two dense arms,
because there is nothing left to match.

**Unnamed #1 — aspect ratio.** The alternative way to reach `E = 224` at fixed `h = 128` and fixed `D = 256` is
`d_ffn = 28672`, i.e. `r = 112`. That model is 99 % FFN, with a residual stream 112× narrower than its own hidden
layer. Any result there is about a pathological aspect ratio. **`r` must be pinned at the donor's 3.5** — which
also makes the FFN's share of parameters donor-representative (64 % of per-layer params at `D = 512, r = 3.5`,
against the donor's ~72 %; probe-4's was 84 %).

**Unnamed #2 — the ternary quantiser is `h`-dependent, and it degrades exactly where the sweep goes.**
`MoEMLP._tern` (and the separate-expert form it is proven equivalent to) computes the per-expert `down` scale as
`W.abs().mean(-1)` over the **`h` axis**. At `h = 128` that is an absmean over 128 values; at `h = 8` it is an
absmean over 8, and each output row of `down` is a ternary combination of only 8 inputs (`3^8` reachable rows).
**The ternary quantiser gets measurably worse as `h` shrinks, for reasons that have nothing to do with routing,
and this artefact does not exist in the donor (`h = 128`).** This is the D1-class trap of this brief: a real,
code-level, silent degradation aligned with the swept variable. It is why every design below carries an
**fp32-MLP twin of the smallest-`h` arm** — which doubles as the planted positive control (§10).

---

## 3. What "success" should mean here (a framing question for you, not for me)

Probe-4's verdict is "MoE ≥ dense at matched total params, at iso-active". The donor operation is the opposite
sign on the active axis: **iso-total, 28× less active.** Asking "does the MoE-over-dense *advantage* survive" is
asking whether a 28× active cut is still free. My engineering prior is that it will not be free, and that the
useful output of M1 is a **number for the cost**, not a pass/fail on "still beats dense". I have designed for
that: every design reports `ΔBPB(dense − MoE)` as a **curve in `α`**, so that if the donor point costs, say,
+0.03 BPB, the programme can price that against whatever throughput the revised D5 rate constant turns out to
buy, rather than simply seeing a red gate. (Deliberately no tok/s figure here — see the note in §0.)
You may want to restate §4 of the brief's pre-registration in those terms before it is sealed.

---

## 4. The impossibility result (stated first, because it constrains the rest)

Design 3 is "the faithful one": hold `h = 128` absolute and `r = 3.5`, and let `E` grow because **width** grows —
which is how `E = 224` actually arises in a donor. Total params in this apparatus (`L = 6`, `V = 1024`,
`N = 96`, `dt_rank = 16`, SWA at layer 5):

| `D` | `d_ffn` | `E = d_ffn/128` | `k` | `a` | `α` | total params | active params |
|---|---|---|---|---|---|---|---|
| 256 | 896 | 7 | 8 | **`E < k` — impossible** | — | — | — |
| 512 | 1792 | 14 | 8 | 57.1 % | 2.000 | 28.85 M | 21.77 M |
| 1024 | 3584 | 28 | 8 | 28.6 % | 1.000 | 108.5 M | 61.36 M |
| 2048 | 7168 | 56 | 8 | 14.3 % | 0.500 | 420.5 M | 194.0 M |
| 4096 | 14336 | 112 | 8 | 7.14 % | 0.250 | 1654.7 M | 673.3 M |
| **8192** | **28672** | **224** | **8** | **3.57 %** | **0.125** | **6564.2 M** | **2487.4 M** |

**The donor point, faithfully reproduced, is a 6.56 B-parameter model.** The largest rung reachable inside
30 GPU-h/week/account is `D = 1024` → `E = 28`, and even that is ~20-25 GPU-h *per single run* (§7). A trend over
`E ∈ {14, 28}` extrapolated to 224 spans **three doublings past the last measured point**. That is not evidence.

**Design 3 is therefore recorded as unviable and is not recommended.** It is in this document so the finding is
citable: *the donor's `E = 224` at `h = 128` is unreachable by construction, and any M1 result is a statement
about ratios, not about absolute expert width.*

---

## 5. Design 1 — "continue probe-4's curve": fixed-`d_ffn` partition sweep

**What is held fixed:** `D = 512`, `d_ffn = 1792` (`r = 3.5`, the donor's), `k = 8` (the donor's), `L = 6`,
`N = 96`, `dt_rank = 16`, `V = 1024`, corpus, steps, optimiser, seq, tokens/step, **and total parameter count**
(the router is padded to width 224 in every arm — unused rows — so parameter counts are bit-identical, not merely
"matched").
**What moves:** the partition `E` only — and therefore `h = 1792/E`, `a = 8/E`, `α = 8h/512`.

| arm | `E` | `h` | `k` | `D` | `d_ffn` | total params | active params | `a` | `α` |
|---|---|---|---|---|---|---|---|---|---|
| A0 dense (control) | — | 1792 | — | 512 | 1792 | 28.157 M | 28.157 M | 100 % | 3.500 |
| A1 | 8 | 224 | 8 | 512 | 1792 | 28.847 M | 28.847 M | 100 % | 3.500 |
| A2 | 14 | 128 | 8 | 512 | 1792 | 28.847 M | 21.769 M | 57.1 % | 2.000 |
| A3 | 28 | 64 | 8 | 512 | 1792 | 28.847 M | 17.050 M | 28.6 % | 1.000 |
| A4 | 56 | 32 | 8 | 512 | 1792 | 28.847 M | 14.691 M | 14.3 % | 0.500 |
| A5 | 112 | 16 | 8 | 512 | 1792 | 28.847 M | 13.511 M | 7.14 % | 0.250 |
| **A6 (donor point)** | **224** | **8** | **8** | 512 | 1792 | 28.847 M | 12.922 M | **3.57 %** | **0.125** |

Notes that matter:
- **A1 (`E = k = 8`) is the degenerate all-experts case** — every token gets every block, so it is dense *plus* a
  learned per-block multiplicative scale. It is a useful internal anchor but it is **not** the dense control;
  A0 is. (Also: the Switch aux is identically constant at `k = E`, so it contributes no gradient there. Harmless,
  but report it rather than let a reader assume the aux was active.)
- **A2 is the only arm where `h = 128`,** i.e. the donor's absolute expert size — at `E = 14`, not 224. Worth
  marking on the plot.
- A6 reproduces the donor's `r`, `E`, `k`, `a`, `α`, `G` **exactly**. It differs only in absolute width (512 vs
  8192) and hence absolute `h` (8 vs 128).
- The 0.69 M gap between A0 and A1..A6 is the padded router (`6 × (512×224 + 224)`), 2.4 % of total. Report it;
  do not paper over it.

**What it isolates:** `α` (equivalently `a`, equivalently `E` at fixed `k`) at constant total params, constant
width, constant `r`. **What it does not isolate:** `E` from `h` — they move together, by identity `G = r/E`.

**Cost:** 7 arms. n=2 seeds default, n=3 on A0 and A6 → 16 runs. At the §7 rate (~7.8 GPU-h/run at
`D = 512`, seq 256, 4000 steps) = **125 GPU-h**, +3 control runs (§10) ≈ **148 GPU-h ≈ 1.6 aggregate-weeks.**

**Strength:** it gives the *shape* — gradual decay vs cliff — which is what an extrapolation to `h = 128` would
have to lean on, and it starts exactly where probe-4 stopped (`α = 4` → `α = 3.5`) so the two studies compose
into one curve. **Weakness:** if A6 is bad, Design 1 cannot say *why*, and "why" determines the programme's
response (§6).

---

## 6. Design 2 — RECOMMENDED: 2×2 factorial on (pool size) × (active slice), + router-value contrast

The brief's question fuses two mechanisms that Design 1 cannot separate: *is it bad because the pool is big, or
because the active slice is thin?* Breaking the identity `α = r·k/E` requires moving `k` as well. That is free
here — the blocked implementation computes all experts anyway, so `k` only changes a mask, and total params do
not depend on `k` at all.

**Held fixed:** `D = 512`, `d_ffn = 1792` (`r = 3.5`), `L = 6`, padded router width 224, corpus, steps, tokens,
optimiser, seq, seeds protocol, **total parameter count**.
**Moved:** `E` (pool size) and `k` (active slice) — orthogonally.

| arm | role | `E` | `h` | `k` | `D` | `d_ffn` | total | active | `a` | `α` | `G` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **C1** | dense control | — | 1792 | — | 512 | 1792 | 28.157 M | 28.157 M | 100 % | 3.500 | — |
| **C2** | small pool, wide slice | 28 | 64 | 8 | 512 | 1792 | 28.847 M | 17.050 M | 28.6 % | 1.000 | 1/8 |
| **C3** | **DONOR POINT** | **224** | **8** | **8** | 512 | 1792 | 28.847 M | 12.922 M | **3.57 %** | **0.125** | **1/64** |
| **C4** | big pool, wide slice | 224 | 8 | 64 | 512 | 1792 | 28.847 M | 17.050 M | 28.6 % | 1.000 | 1/64 |
| **C5** | small pool, thin slice | 28 | 64 | 1 | 512 | 1792 | 28.847 M | 12.922 M | 3.57 % | 0.125 | 1/8 |
| **C6** | C3 with **frozen random router** | 224 | 8 | 8 | 512 | 1792 | 28.847 M | 12.922 M | 3.57 % | 0.125 | 1/64 |

The factorial reads directly:

- **`C2 − C4`** = the cost of a **big pool at fixed active slice** (`α = 1.0` both; `E` 28 → 224). Pure pool-size /
  granularity effect, active cost identical, params identical.
- **`C2 − C5`** = the cost of a **thin active slice at fixed pool** (`E = 28` both; `α` 1.0 → 0.125). Pure
  active-width effect.
- **`C3`** sits at the intersection; if the two main effects are additive, `C3 ≈ C2 + (C4−C2) + (C5−C2)`, and the
  residual is the interaction — which is precisely "does thinning hurt *more* when the pool is large".
- **`C3 − C6`** = **the value of learned routing at `E = 224`.** This is the router diagnostic (§9).
- **`C1 − C3`** = the headline the ledger needs.

Add, as single-seed shape riders folded in from Design 1: **A4 (`E = 56, k = 8`, `α = 0.5`)** and
**A5 (`E = 112, k = 8`, `α = 0.25`)**, so the `α` curve has 5 points (3.5, 1.0, 0.5, 0.25, 0.125) rather than 2.

**Optional tranche 2 — the width-doubling scale check** (run only if C3 shows a material deficit; see §12):

| arm | `E` | `h` | `k` | `D` | `d_ffn` | total | active | `a` | `α` |
|---|---|---|---|---|---|---|---|---|---|
| S1 dense control | — | 3584 | — | 1024 | 3584 | 107.171 M | 107.171 M | 100 % | 3.500 |
| S2 donor point @2× width | 224 | 16 | 8 | 1024 | 3584 | 108.548 M | 44.847 M | 3.57 % | 0.125 |
| S3 `h = 128` at 2× width | 28 | 128 | 8 | 1024 | 3584 | 108.548 M | 61.362 M | 28.6 % | 1.000 |

`S1 − S2` vs `C1 − C3` is the only measurement in this programme that says whether the donor point's deficit is a
function of the **ratio** (in which case it transfers to `h = 128`) or of **absolute expert width** (in which case
it shrinks and the small-model result is pessimistic). One doubling, `h` 8 → 16, against the four doublings that
separate us from 128. Bounded, not closed. Label it that way in the write-up.

---

## 7. Cost — steps and T4 GPU-hours

**Where the time actually goes, because it changes the levers.** The apparatus's step cost is dominated by the
**Python-unrolled sequential SSM scan** (`phase55_ssm.py:SSMBlock.forward`, a `for t in range(L)` over the
sequence), 5 SSM layers × `seq` iterations × several tiny kernels, executed once in forward and again in the
checkpointed backward recompute. That is **launch/latency-bound, not FLOP-bound**. Consequences:

- Step time is **nearly independent of `E`** (the MoE is compute-all: three big GEMMs whatever `E` is) — so all
  arms of a design cost the same. Good for comparability.
- Step time is **weakly dependent on `d_model`** over 256 → 512 (element counts grow, but the loop stays
  launch-bound).
- The **dominant lever is `seq`**: halving `seq` at constant tokens/step halves the number of loop iterations.
- Activation memory *is* `d_model`-sensitive: `dA` and `dBx` are `(B, T, 2D, N)` fp32. At 2048 tokens/micro-batch
  and `D = 512` that is **806 MB each**; at `D = 1024`, **1.6 GB each**. `D = 1024` will need micro-batches of
  ~512 tokens and `accum = 16`, which multiplies the launch overhead — this is why tranche 2 is expensive out of
  proportion to its parameter count.

**Anchor:** probe-4 measured **6.4 s/step** for `D = 256`, `L = 6`, `seq = 512`, `batch 4 × accum 4`
(8192 tok/step) on an **RTX 3060, bf16**.

**T4 adjustment.** T4 is Turing: **`--fp16` + GradScaler, not bf16** (the script supports both; bf16 must not be
used). For a launch-bound workload on Kaggle's 2-vCPU host I estimate **1.6–2.2× the 3060's step time**.
Recommended config change: **`seq = 256`, `batch = 4`, `accum = 8`** (same 8192 tok/step, half the loop
iterations, half the activation memory per micro-batch).

| config | est. s/step | ×4000 steps | note |
|---|---|---|---|
| probe-4 replication, `D = 256`, seq 512, T4 fp16 | 10–14 | **11–15.5 GPU-h** | needed only for the anchor arms |
| `D = 512`, `r = 3.5`, seq 256, T4 fp16 | 6–9 (central 7) | **~7.8 GPU-h** | the working rate for Designs 1 & 2 |
| `D = 1024`, `r = 3.5`, seq 256, accum 16, T4 fp16 | 18–25 | **20–28 GPU-h** | tranche 2 only |

**These are estimates with a ±50 % band and I will not defend them.** The first thing the operator should run is a
**20-step timing calibration** on one T4 for each of the three configs (≈ 5 GPU-min total), after which the whole
plan is re-costed from measured s/step **before any 4000-step run launches**. I would make that a hard
pre-registration item: no long run until the calibration number is in the manifest.

**Budgets** (3 accounts × 30 GPU-h/week = **90 GPU-h/week aggregate**):

| tier | runs | GPU-h | aggregate-weeks | what it buys |
|---|---|---|---|---|
| **Minimum viable** — C1 (n=3), C3 (n=3), fp32 plant (n=1), destructive plant (n=1) | 8 | **62** | 0.7 | the yes/no on the ledger row, with a validated instrument |
| **Recommended (Design 2 + riders)** — C1,C3 (n=3); C2,C4,C5 (n=2); C6 (n=1); A4,A5 (n=1); 3 control runs | 19 | **145** | 1.6 | yes/no **plus the mechanism**: pool vs slice vs router |
| Design 1 full curve | 16 + 3 | **148** | 1.6 | the shape, without the mechanism split |
| Tranche 2 scale check — S1,S2 (n=2), S3 (n=1) | 5 | **100–140** | 1.1–1.6 | bounds the `h = 8 → 128` extrapolation by one doubling |
| Design 3 (faithful) | — | **unviable** | — | reaches `E = 28`; needs 6.56 B params for `E = 224` |

Realistically the recommended tier is **5–9 calendar days** across three accounts once queue friction and the
12-hour Kaggle session cap are included. A 4000-step `D = 512` run at ~7.8 h fits one session with margin; a
`D = 1024` run at 20–28 h **does not** and needs checkpoint/resume — flag that to the operator now, not after a
session is killed at hour 12.

**Steps.** Keep **4000 steps × 8192 tokens = 32.8 M tokens** (probe-4's budget) for comparability. Note honestly
that this is ~1.1 epochs of the 29.5 M-token TinyStories train split and ~1.6 tokens/parameter — deeply
under-trained by any scaling law. That is acceptable for a matched A/B and is a named threat (§8.2).

---

## 8. What could make the result uninterpretable — my list

Ordered by how much I think each one actually threatens the conclusion.

**8.1 — Corpus homogeneity (biggest threat).** TinyStories/BPE-1024 is one narrow domain. There may be nothing
for 224 experts to specialise into, so `C3` could lose for a *data* reason and the deficit would not transfer to a
general-purpose donor. And the donor programme's success criterion is explicitly *general-purpose retention*, so
a homogeneous-corpus null is weak evidence for the thing being decided. **Mitigation, proportionate:** run the
primary tranche on TinyStories (comparability, banked σ_seed, banked baselines, zero data-prep risk), and if C3
shows a deficit, replicate **C1 and C3 only** on a mixed corpus. A mixed BPE-1024 stream over
TinyStories + the phase-62 code corpus (`results/phase62/code_train.u16`) is one offline CPU pass using
`benchmarks/phase62/corpus.py` + `cartography.train_bpe`; it is real work (~half a day, no GPU) and it breaks
direct comparability with 0.8589, which the replication anchor (§10) restores. **Counter-argument for keeping
TinyStories:** probe-4 already found routing ≈ i.i.d. and the aux forces uniformity, so specialisation was
probably never the mechanism by which MoE-gran won — which weakens, but does not remove, the objection.

**8.2 — Fixed-step budget vs differential convergence rate.** Larger `E` plausibly converges slower per step. At
4000 steps nothing is converged, so a C3 deficit could be an under-training artefact. **Mitigation:** log val BPB
every 400 steps (the script already does 10 evals per run) and **pre-register an abort condition**: if the
last-decade BPB slope of C3 exceeds that of C1 by more than 2×, the comparison is void and must be re-run at 2×
steps. This must be a pre-registered abort, not a post-hoc excuse.

**8.3 — Statistical power, and it is asymmetric.** With σ_seed = 0.005:

| seeds/arm | s.e. of a difference | 2σ MDE | largest deficit excludable at 95 % one-sided, given δ = 0.02 |
|---|---|---|---|
| 1 | 0.00707 | 0.0141 | observed Δ ≤ 0.0084 |
| 2 | 0.00500 | 0.0100 | observed Δ ≤ 0.0118 |
| 3 | 0.00408 | 0.0082 | observed Δ ≤ 0.0133 |
| 4 | 0.00354 | 0.0071 | observed Δ ≤ 0.0142 |

**Probe-4's entire MoE-over-dense advantage was 0.0085 BPB — smaller than the 2-seed MDE.** So: detecting a
*failure* at C3 (which, if real, should be ≫ 0.02) is easy at n=2; **certifying that it survives is expensive**
and needs n=3 minimum with a pre-registered equivalence margin. I suggest **δ = 0.02 BPB (4 σ_seed) as the
"material deficit" threshold** and n=3 on C1/C3, which excludes δ = 0.02 if the observed deficit is ≤ 0.0133.
Do not let a 2-seed null be written up as "it survives".

**8.4 — The `h`-dependent ternary quantiser (§2, unnamed confound #2).** At `h = 8` the per-expert `down` scale is
an absmean over 8 values and each output row is a ternary combination of 8 inputs. This degrades monotonically
with `E` in Designs 1 and 2 and does **not** exist at the donor's `h = 128`. **Mitigation: an fp32-MLP twin of C3**
(`mlp_precision = fp32`), which separates quantiser degradation from routing degradation and doubles as the
banked positive control (§10). Without it, a C3 deficit is unattributable.

**8.5 — Router collapse vs granularity.** See §9. Unaddressed, this alone can void the whole tranche.

**8.6 — Aux-loss weight is not scale-free.** `--load-balance-w 0.01` was chosen at `E = 32`. At `E = 224` it may
over- or under-balance. **Mitigation:** a pre-registered *balance-criterion* selection — run 500 steps at
`load_w ∈ {0.003, 0.01, 0.03}` at `E = 224` and pick the **smallest** weight achieving `dead = 0` and
`max/mean ≤ 1.5`, **selecting on load statistics only, never on BPB**, then apply the same rule symmetrically at
every `E`. Cost: 3 × ~1 GPU-h. Selecting on BPB here would be textbook Goodhart and must be forbidden in writing.

**8.7 — Cross-hardware numerics.** Probe-4's 0.8589 was 3060 + **bf16**. T4 forces **fp16 + GradScaler**. You
cannot compare a new T4/fp16 number to 0.8589 directly. The replication anchor (§10) is exactly this control: if
it lands within ~2σ of 0.8589, the platform change is benign; if not, all probe-4 comparisons become
internal-only and must be stated as such.

**8.8 — Backbone dilution.** At `D = 512, r = 3.5` the FFN is 57 % of total params (probe-4's was 84 %), so a
given fractional FFN degradation shows up as a smaller ΔBPB. This is *more* donor-representative (donor FFN is
~72 % of per-layer params) but it costs power. It is already priced into §8.3; do not additionally assume
probe-4-sized effect magnitudes.

**8.9 — `E = k` degeneracy (arm A1 only).** Constant aux, no balancing gradient, and a `1/E` scaling of the
hidden. Fine as an anchor, misleading if reported as "dense".

**8.10 — A measurement-harness memory bug that will bite at `D = 512`.** `phase59_moe.py:measure()` stores the
full input tensor per layer (`m._xin`) and concatenates: at `nbatch = 24`, `batch = 4`, `T = 512`, `D = 512` that
is ~100 MB/layer × 6 layers = **600 MB**, then `.astype(np.float64)` inside `ridge_fit` **doubles it to ~1.2 GB**.
At `D = 1024` it is ~2.4 GB. This is a Kaggle-instance OOM waiting to happen at the very end of a 7-hour run.
Must be fixed by subsampling positions before the ridge fit (§11, item 9).

---

## 9. Router health at `E = 224` — and how to tell it apart from granularity

**My judgement: the router will be co-binding, and the aux loss is the mechanism.** The argument, from our own
measurements rather than from literature:

1. Probe-4 measured, at `E = 32`, that the expert working set over a block ≈ the i.i.d. baseline (84–89 % of `E`
   at N=8 vs 90 % random) and that expert persistence = the base rate `k/E`. Routing was already
   **indistinguishable from independent sampling in time**, and probe-4's own memo attributes this to the Switch
   aux enforcing uniformity by design.
2. At `E = 224`, keeping 224 experts alive is a *much* stronger constraint. The Switch aux is estimated
   **per micro-batch**, and at §7's recommended `batch 4 × seq 256 = 1024` tokens with `k = 8` there are 8192
   expert-slots, i.e. **36.6 slots per expert** — against 256 at `E = 32` on the same micro-batch, or 512 at
   probe-4's own `batch 4 × seq 512`. The load estimate the aux differentiates is **7× noisier**, and the aux's
   own fixed point at large `E` is close to the maximum-entropy (random) router. Note this is a *micro-batch*
   quantity: `accum` does not help, because each micro-batch computes and backprops its own aux term. **If the
   `E = 224` arms show balancing trouble, the first lever is a larger micro-batch (fewer, bigger `accum`
   steps), not a larger `load_w`** — and that trade is bounded by the activation-memory figures in §7.
3. Therefore a plausible outcome is: `C3` reports `dead = 0, max/mean ≈ 1.1×` — a *perfect* router-health line —
   while the router has learned nothing input-dependent. **`dead` and `max/mean` cannot detect this failure.**
   They measure the *marginal* load, and a uniform-random router maximises exactly that.

**What must be measured to separate the two.** Three items, all cheap, all additions to the existing `measure()`:

- **(a) The frozen-random-router twin, C6 — mandatory, and it is the decisive one.** Train an identical
  `E = 224, k = 8` arm whose router is a frozen random `Linear(D, 224)` (never updated). Everything else,
  including the multiplicative block-gating structure, is unchanged. Then:

  | observation | conclusion |
  |---|---|
  | `C3 ≪ C6` (learned much better) | routing is working at `E = 224`; a C3 deficit is **granularity**, and it is the real answer |
  | `C3 ≈ C6` | the learned router adds nothing at `E = 224`; C3's number describes a **random block-sparse FFN**, and the binding constraint is the router/aux, not granularity |
  | `C3 ≈ C6` **and** the same at `E = 28` | routing never mattered in this apparatus at all — which would mean **probe-4's win was the multiplicative gate, not the routing**, and probe-4's own reading needs revising. This is a headline either way. |

  Note that swapping routers at *evaluation* time is meaningless (the experts were trained under the learned
  assignment). C6 must be **trained**. One arm, ~7.8 GPU-h. Probe-4 never ran this control, which is why its
  memo can only *speculate* that "the router is a multiplicative input-dependent gate the dense arm lacks".

- **(b) Routing decisiveness — report at every `E`.** `mean_x KL( p(·|x) ‖ mean_x p(·|x) )`, in nats. If this is
  ≈ 0 the router is not conditioning on its input, regardless of how balanced it is. ~5 lines in `measure()`.
  Also report `mean_x H(p(·|x))` against `log E` (2.30 at `E = 10`, 5.41 at `E = 224`).

- **(c) Cross-seed routing agreement (nice-to-have, CPU-only, post-hoc).** Take two seeds of the same arm, build
  the `E × E` co-assignment matrix over a fixed held-out block, solve the best permutation (Hungarian, `E = 224`
  is trivial on CPU), and report agreement against the chance rate `k/E`. Above chance ⇒ the router found
  reproducible structure. At chance ⇒ the partition is arbitrary. This runs on saved checkpoints, costs no GPU,
  and can be added after the fact.

Also keep, unchanged, probe-4's existing lines: `dead`, `max/mean`, working-set `|∪experts|/E` at N=8/16/32,
persistence, ridge `rec_in`/`rec_ahead`. They are cheap and they preserve continuity with probe-4's table.

---

## 10. Planted positive controls (mandatory, for the recommended design)

Three, with distinct jobs. **Pre-register that if any of the first two fails, the tranche is void.**

**P1 — banked-magnitude plant: the fp32-vs-ternary MLP switch.** Run C3 with `mlp_precision = fp32` (identical
parameter count, identical everything else). Banked on this exact apparatus at `D = 256`: ternary 0.8382 vs fp32
0.8104 = **+0.0278 BPB = 5.6 σ_seed**. The magnitude will not transfer exactly to `D = 512, r = 3.5`, so the
pre-registered firing criterion is **direction + ≥ 3 σ_seed (0.015)**, not the exact value. This is the one
control with a *known real* effect measured on this instrument. It doubles as the §8.4 quantiser control: the
fp32 twin removes the `h = 8` ternary artefact, so `(C3_fp32 − C1_fp32)` vs `(C3 − C1)` attributes the deficit
between routing and quantisation. **1 run, 7.8 GPU-h. Best value per hour in the whole plan.**

**P2 — destructive on-axis sensitivity plant: `E = 224, k = 1`** (`a = 0.45 %`, `α = 0.0156`, 224× less active
FFN than dense, at identical total params). If **this** does not degrade BPB by ≫ 2σ, then the FFN is not
load-bearing in this apparatus and **no null anywhere in the sweep means anything**. It requires no banked value —
it only has to fire. **1 run, 7.8 GPU-h.**

**P3 — pipeline/platform anchor: replicate probe-4's `moe-gran` and `dense-big`** at the original
`D = 256, d_ffn = 4096, E = 32, h = 128, k = 8`, on T4/fp16, 4000 steps. Targets: 0.8589 and 0.8674. Firing
criterion: both within 2σ_seed (0.010) of the banked values, and the sign of `moe-gran < dense-big` preserved.
This is what makes the new numbers commensurable with probe-4's rather than analogical, and it is the only check
on §8.7 (bf16 → fp16). **2 runs, ~11–15 GPU-h each at seq 512; ~22–30 GPU-h.** If the budget is tight, run
`moe-gran` alone and check the absolute BPB.

---

## 11. Reuse of probe-4's code — verdict and exact change list

**Verdict: reuse with changes. One file, no new apparatus, no new model code, no new data path (unless §8.1's
mixed corpus is adopted).** I estimate **~200 lines of edits** to `benchmarks/phase57/phase59_moe.py` plus a
small results-JSON writer. Everything structural is reusable as-is: `moe_forward` (the checkpoint-survives-aux
fix), the blocked `MoEMLP` and its `_ref` equivalence self-test, the Switch aux, `val_bpb`, and the five measures.

Required changes, in priority order:

1. **`--seed` (load-bearing).** The script hard-codes `torch.manual_seed(0); np.random.seed(0)`. There is
   currently **no way to run a second seed**, and σ_seed = 0.005 makes replicates the load-bearing part of this
   brief. Must seed torch, numpy, and CUDA from one flag, and record it in the manifest.
2. **Parametric arms.** `ARMS` is a 3-entry hard-coded dict; replace with `--E --hid-e --k --d-ffn --dense`
   (and derive `hid_e = d_ffn / E` when not given). Add asserts: `E >= k`, `E * hid_e == d_ffn`, `d_ffn % E == 0`.
3. **Parametric architecture.** `D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, ...)` is hard-coded.
   Expose `--d-model --layers --n-state --dt-rank --heads --swa-layer`. Keep the defaults at probe-4's values so
   the replication anchor is a no-flag run.
4. **Padded router width.** `--router-width` (default `E`); allocate `nn.Linear(D, router_width)` and slice the
   first `E` logits. Set it to 224 for every arm so parameter counts are bit-identical across the sweep. ~4 lines.
5. **Router variant.** `--router {learned,frozen-random}`; for frozen-random, initialise and
   `requires_grad_(False)`. ~8 lines. This is arm C6 and it is the §9 diagnostic.
6. **ACHIEVED-configuration report (the D1 rule).** The current print at `phase59_moe.py` reads from the `ARMS`
   dict — i.e. it reports the **requested** config. Replace with a walk over the **constructed** model that
   prints, per MLP block: `type(b.mlp)`, `m.E`, `m.hid_e`, `m.k`, `m.gate.out_features`, `m.Wd.shape`,
   `m.router.out_features`, and a parameter census by group (emb / SSM / SWA / norms / head / router / gate /
   up / down), then **asserts** each against the requested value and aborts on mismatch. Also print the achieved
   `a`, `α`, `r`, `G` computed from the constructed shapes, not from the CLI.
7. **Results manifest.** Dump a JSON per run: achieved config, seed, git SHA, torch/CUDA versions, device name,
   amp dtype, measured s/step, the BPB trajectory, final BPB, and the full router-health block. Needed for the
   reproducibility manifest and for the re-cost gate in §7.
8. **Routing-decisiveness metric** (§9b) in `measure()`. ~5 lines.
9. **Fix the `measure()` memory blow-up** (§8.10): subsample positions for the ridge probe (e.g. keep a fixed
   random 20 % of rows, seeded) instead of retaining every `(batch, T, D)` input, and fit in float32 rather than
   promoting to float64. Without this, a `D = 512` run risks OOM in its final minute.
10. **T4 hygiene.** Use `--fp16` (Turing has no bf16); the GradScaler path already exists. Do **not** pass
    `--compile` — it is not even exposed in this script, correctly, because Inductor chokes on the unrolled scan.
11. **Corpus flags** `--ids --meta` (currently hard-wired to `results/phase55/{ids.u16,meta.bin}`), needed only if
    §8.1's mixed corpus is adopted. ~6 lines.
12. **`--mlp-precision {ternary,fp32}`** for the P1 plant: `SparseMLP` already takes `ternary=`; `MoEMLP` needs
    the same switch (use `nn.Linear` in place of `BitLinear158` and skip `_tern` on `Wd`). ~10 lines.
13. **Checkpoint/resume** — only if tranche 2 (`D = 1024`, 20–28 h) is run, because Kaggle sessions cap at 12 h.
    Save optimiser state + step + RNG state, `--resume`. ~30 lines. Not needed for tranche 1.

Nothing in this list is research risk; it is all plumbing. The smoke path (`--smoke`, which runs the `_ref`
blocked-vs-loop equivalence self-test) survives unchanged and should be re-run at `E = 224, h = 8` before
anything launches — that self-test is the one existing guarantee that the blocked layout is still exact at the
new shapes, and `h = 8` is a shape it has never seen.

---

## 12. Recommendation, argued

**Run Design 2 (§6) with the two Design-1 riders and the three plants of §10 — 19 runs, ~145 GPU-h.** If only a
few days are available, run the 8-run minimum-viable subset first; it is a strict prefix of the recommended plan,
not a different experiment.

**Why Design 2 over Design 1.** Design 1 gives a prettier curve for the same money, and if the only question were
"does the ledger row hold", Design 1's endpoints would do. But the brief itself says what happens on a failure:
*"the ledger's ✅ rows collapse and the programme needs a different FFN story."* A bare curve tells you the row
failed; it does not tell you **which** story to write next, and the three candidate causes lead to three
completely different programmes:

- **thin active slice binds** (`C2 − C5` large, `C2 − C4` small) → the FFN genuinely cannot be run at `α = 0.125`;
  the throughput target must be met somewhere other than FFN sparsity. That kills the row and the recovery is
  expensive.
- **big pool binds** (`C2 − C4` large) → the deficit is granularity/routing capacity, and the fix is fewer, larger
  experts at the same `α` — which the donor can have for free by choosing `h = 256` and `E = 112`, `k = 4`.
  **That is a re-parameterisation of the donor conversion, not a new programme**, and it would be found only by
  the factorial.
- **the router binds** (`C3 ≈ C6`) → the fix is the aux loss and the router, both cheap, and the ledger row
  survives with an asterisk pointing at §3.6's predictability work.

Design 2 distinguishes all three for ~15 GPU-h more than Design 1. That is the cheapest decision-relevant
information in the plan.

**Why not Design 3.** §4: the faithful configuration is 6.56 B parameters. It is in the document so the
impossibility is on the record, not so it can be run.

**What I would put in the pre-registration that is not yet in §4 of the brief:**

- The primary quantity is **`ΔBPB = BPB(C3) − BPB(C1)`**, reported with a confidence interval, and the verdict is
  **the size of the deficit**, not a binary. Pre-commit the material-deficit margin (I suggest δ = 0.02 BPB) and
  the seed counts that can support it (n = 3 on C1 and C3).
- **Asymmetric evidence rule:** a 2-seed result may declare *failure*; only an n≥3 result with the equivalence
  bound may declare *survival* (§8.3).
- **The abort conditions are pre-registered, not discretionary:** the P1 plant must fire at ≥ 3σ; the P2 plant
  must fire; the P3 anchor must land within 2σ of 0.8589/0.8674; the C3 convergence slope must not exceed C1's by
  more than 2×; and the `load_w` selection at `E = 224` is made on load statistics only, never on BPB.
- **The permanent caveat, written into the verdict rather than the appendix:** M1 measures ratios at
  `h = 8` (and, if tranche 2 runs, `h = 16`). The donor's `h = 128` is four doublings away and is not, and will
  not be, measured. Whatever M1 concludes, the ledger row it licenses should read *"validated at donor ratios,
  extrapolated across 4 doublings of absolute expert width"* — because that is what will have been done.

---

**STOP.** No code written, no training run, no commit. Awaiting a design choice and a pre-registration before any
implementation brief.
