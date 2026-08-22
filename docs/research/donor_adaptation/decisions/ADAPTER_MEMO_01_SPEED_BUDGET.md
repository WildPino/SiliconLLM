# Adapter Memo 01 — the two levers, and the speed budget that follows

**Author:** the Adapter / Principal · **Date:** 2026-08-21 · **Base revision:** `e410243`
**Status:** decision memo. Fixes the target the Builder's briefs are written against. Not a gate — the
Architect/Owner alone adjudicates sealed gates.

---

## 0. ⚠ READING ORDER — what in this memo still stands, as of 2026-08-22 evening

This memo has accumulated five amendments in one day, and **its original headline arithmetic is wrong.**
Read this block before anything else.

| section | status |
|---|---|
| §1 — the two levers (static vs dynamic sparsity) | **stands** |
| §2.1 — the headline table | **the rate constant is under active revision.** See the banner in §2.1 |
| §2.2 — "attention on 4 of 80 layers, FFN at 2%" | **WRONG on two counts. Superseded by §2.2f** |
| §2.2b — the attention half was uncosted | stands as a record; its fork is superseded by §2.2f |
| §2.2c — the 5% attention ratio is below anything measured | **demoted by §2.2f**: the ratio governs KV traffic and quality, not the weight stream |
| §2.2d — a dense FFN cannot go below 33.3% active | **stands.** The gate is the predictor; `FFN_active = (3−2s)/3` |
| §2.2e — the extrapolation ledger | ledger **stands**; its ✅ rows withdrawn by §2.2f; its "probe-4 granularity at donor width" reframe **withdrawn by §2.2g** |
| **§2.2f — conversion does not remove a layer's weights; KV accounting** | **CURRENT. Start here.** |
| **§2.2g — `h=128` is not scale-free; probe-4 never entered our regime** | **CURRENT.** Also: probe-4's own result is underpowered and missing its key control |
| §3a — healing must happen in the consuming layer | stands, **but** see `BRIEF_R2` Amendment A1.5: applying it to R2 was pattern-matching |
| §3c, §3d | stand |

**The one-line state of the programme:** the weight-stream budget is governed by `L × 151M` of attention
projections that conversion does not remove; the FFN half requires MoE restructuring, not activation
sparsity; and **at long context the KV term of even a few retained softmax layers exceeds the entire weight
stream**, which puts the speed target and the retention target on the same knob pulling opposite ways.

---

## 1. The distinction the v2 programme never made cleanly

`DONOR_V2_DENSITY_PROGRAMME.md` framed the problem as "the donor is 33–66× too dense" and listed four
attacks. It did not separate two mechanisms that behave completely differently on this engine:

| | **Static weight sparsity** | **Dynamic activation sparsity** |
|---|---|---|
| What it is | Weights permanently removed | Neuron's activation is zero *for this token*, so its weights are never loaded |
| When decided | Once, at conversion | Per token, at inference |
| Measured cost, in-house | **Catastrophic.** D1: +0.13 to +4.8 BPB at every organ and level, against σ_seed = 0.005 | **Near-free.** Probe-2: dReLU vs SiLU = **+0.0006 BPB** at 92%/79% sparsity |
| Ceiling | Bounded by how much of the model is genuinely redundant | Bounded only by how peaked the per-token computation is |
| Gets us to 2%? | **No.** 90% already costs +3.5 BPB | **This is the only lever that can** |

**The 2%-per-token target lives entirely on the second lever.** Static pruning is a footprint tool and a
secondary one; it is not the road to speed. Every hour spent pushing static pruning past the point D1
already measured is an hour spent on the wrong axis.

This also re-reads D1 correctly. D1's numbers are not a discouraging result about our programme — they
are the **floor that reconstruction must beat**, measured on the naive method. Brief D4 is measuring the
gap. And D1's block-vs-unstructured penalty is a statement about *weight* masks, which is not the mask
the engine actually exploits.

**Consequence for the reading of `feedback` and prior probes:** Probe-2 and Phase 58.B are not old
results to be cited in passing. They are the *primary* evidence for the whole v2 thesis, and they were
obtained on a model **we trained ourselves with dReLU from scratch**. Whether the same sparsity can be
*induced* in a donor pretrained with SiLU, and at what token cost, is now the highest-value open
question in the programme. It is out with the Researcher.

---

## 2. The speed budget, and where the 2% must actually come from

### 2.1 The headline arithmetic

At **0.5 bytes/weight** (the engine's real packing: two trits per byte via `(w0+1)*3+(w1+1)`), a 100B
donor is **50 GB** of weights. At the measured engine-integrated ternary LUT rate:

> active_bytes_per_token = 50 GB × activation_fraction
> tokens_per_second = 21.25 GB/s ÷ active_bytes_per_token

| activation fraction | active weights | bytes/token | tok/s |
|---|---|---|---|
| 100% (dense donor) | 100 B | 50 GB | **0.43** |
| 10% | 10 B | 5 GB | 4.3 |
| 4% | 4 B | 2 GB | 10.6 |
| **2%** | **2 B** | **1 GB** | **21.3** |
| 1% | 1 B | 0.5 GB | 42.5 |

**⚠ The load-bearing constant, flagged rather than assumed.** 21.25 ± 0.36 GB/s was measured
*engine-integrated at D = 1536*. Whether it holds at donor width D = 8192 is **exactly the class of
transplant that has already been caught twice today** (the `B_block ≥ 22` floor, and the skippable-0.001
column). The LUT path is recorded as **compute-bound by ~16×**, not bandwidth-bound, so its rate is a
function of the kernel's shape and not a property of the DRAM. **This must be re-measured at donor
projection widths before any tok/s figure derived from it is quoted to the Owner.** Until then every
number in this table carries that caveat. → **queued as brief D5.**

> ### ⚠ 2026-08-22 evening — the D5 sweep has COMPLETED and this constant is under revision
>
> D5 ran to completion with all four planted controls PASSing. **Its early reading indicates the
> 21.25 GB/s does NOT transfer to donor width, and that the donor-width ternary path may be compute-bound
> rather than bandwidth-bound.** If that survives audit, two things follow: every tok/s figure in this
> memo moves, and **the 5-trit packing lever of §2.4 does not pay**, because bytes would not be the binding
> constraint.
>
> **No number from that run may be quoted until the Builder has written it up and the Controller has
> audited it.** I have deliberately not banked the figures here. **Treat every tok/s in this document as
> provisional until this banner is removed.**

### 2.2 Where the 2% has to come from — the part nobody had allocated

> **⚠ READ §2.2f FIRST.** This section counts only the *retained* attention layers' weights. That is
> wrong: converting a layer does not remove its `W_q`/`W_k`/`W_v`/`W_o`, and the geometry below is 68.5B,
> not 100B. §2.2f corrects both and withdraws the conclusions that follow from them.

A 100B-class geometry (`d_model = 8192`, `d_ffn = 28672`, `L = 80`):

| | per layer | share |
|---|---|---|
| attention (q + k + v + o), GQA 64/8 | 151 M | 18% |
| FFN (gate + up + down) | 705 M | **82%** |
| total | 856 M | |

**If the FFN goes to 2% active and attention stays dense, the model is still at 13% overall** — because
attention, untouched, is 18% of the weights and 100% of them are read every token. Attention alone would
cap us at **3.3 tok/s**.

So the target decomposes, and both halves are mandatory:

> **(a) FFN at ~2% activation** — the activation-sparsity lever, on 82% of the weights.
> **(b) Attention retained on at most a few percent of layers** — everything else converted to a
> recurrent operator with no KV re-read.

Worked target: attention kept on **4 of 80 layers**, FFN at 2% everywhere:

```
attention   4 layers × 151 M                   = 0.60 B
FFN        80 layers × 705 M × 0.02            = 1.13 B
                                        active ≈ 1.73 B  (1.7% of 100B)
                                   bytes/token ≈ 0.87 GB
                                         tok/s ≈ 24
```

**That closes.** It is the first end-to-end arithmetic in this programme that reaches the Owner's ">20
tok/s at 100B" without a step marked as a guess — subject to the §2.1 caveat on the rate constant.

### 2.2b ⚠ A GAP IN MY OWN ANALYSIS: I costed the FFN half and not the attention half

§2.2 states two mandatory halves. I priced the first (activation sparsity, §3c) and **left the second
uncosted**, which means the budget did not actually close — it closed on one of its two terms. Correcting
that here.

Requirement (b) is *"attention retained on at most a few percent of layers — everything else converted to
a recurrent operator with no KV re-read."* On a 100B/L=80 donor that is **converting ~76 of 80 layers**
from attention to a recurrent operator. The only published method that does this and retains quality is
MOHAWK (Bick et al., arXiv:2408.10189): freeze the donor MLP, discard attention, train a Mamba-2 core on
**3.0B tokens** (80M + 160M + 2.76B across three stages) `[A]`, reaching **62.6 vs the teacher's 64.9**
`[T, Table 1]`.

**The paper does not state its GPU cost — `[X]`, confirmed against the primary source.** So the following
is **my derivation, not theirs**, and every assumption is named:

- Training FLOPs ≈ `(4 + 2f)·N·D` for N parameters, D tokens, fraction `f` of weights trainable. MOHAWK
  freezes the MLP, so `f` is small and this is ≈ `4·N·D`. *(Forward `2ND`, activation gradients `2ND`,
  weight gradients only for trained params.)*
- 2× T4 at 65 TFLOPS fp16 peak, **25% MFU** — my assumption, and the single biggest lever in this estimate.

| donor | tokens | my FLOP estimate | at 2×T4, 25% MFU | against 90 h/week |
|---|---|---|---|---|
| 1.5B (Phi-Mamba scale) | 3.0B | 1.8e19 | **~154 h** | **~1.7 weeks — affordable** |
| 100B | 3.0B | 1.2e21 | **~10,250 h** | **~114 weeks — out by ~65×** |

**The token budget does not save us.** MOHAWK's 3B tokens is a *distillation* budget, ~2 tokens/parameter,
far below Chinchilla — so it plausibly does **not** need to grow with donor size. But FLOPs are
`params × tokens`, so holding tokens fixed still scales the cost linearly with N. **A 100B donor is
unaffordable on this hardware by roughly two orders of magnitude even at a fixed 3B-token budget.**

### The fork this creates, which is the Owner's to adjudicate, not mine

1. **Convert attention closed-form, with no training at all.** Preserves the 100B target and the ">20
   tok/s" headline. This is the hardest open problem in the programme and nothing in the literature does
   it — the original S1 constraint (*"trovare un modo matematico per convertire"*) pointed here, and the
   v1 work concluded it was the hardest thing on the board.
2. **Accept a smaller donor.** At **1.5B the MOHAWK route is affordable at ~1.7 weeks**, and 1.5B is the
   donor D1/D4 already run on. This would demonstrate the full pipeline end-to-end on real hardware within
   budget, and the 100B claim becomes a scaling argument rather than a demonstration.
3. **Keep attention on more layers and lose speed.** §2.2's arithmetic prices this exactly: attention is
   18% of the weights and every retained layer is read in full every token.

**I am not choosing between these** — option 2 changes what we deliver, and that is a sealed-scope
question. **My recommendation, offered as a recommendation:** pursue (2) as the demonstration and (1) as
the research question, in parallel. (2) is the only path that produces a *running converted model* inside
the budget we actually have, and a working 1.5B conversion is worth more than an unfalsifiable 100B plan.

**What is NOT established here:** the 25% MFU assumption is mine and unmeasured on this hardware; MOHAWK's
retention was measured on Phi-1.5, not on a 100B donor or on a ternary/sparsified backbone; and whether
MOHAWK's frozen-MLP trick composes with the activation-sparsity route of §3c is completely untested — both
modify the FFN's role, and this project has repeatedly measured that sequential stages do not compose.

### 2.2c ⚠ AMENDMENT (2026-08-22) — the 5% attention budget is BELOW every ratio the literature has measured

The Researcher's `ATTENTION_LINEARISATION_PRIOR_ART.md` came back, and it moves the §2.2 arithmetic.
**§2.2's worked target is 4 of 80 layers = 5%. No published work has measured a donor-converted hybrid
below 12.5%.** Mamba-in-the-Llama's lowest tested ratio is 12.5% `[T]`, and the reported degradation at
that point is **still worsening, not plateauing.** Zoology reports a hybrid working at 6.3%, but that
model is **trained from scratch, not converted from a donor** — a different construction, and not
evidence for our case.

> **So §2.2 did not derive 5% from evidence. It derived 5% from the speed budget and then assumed the
> quality side would follow.** That is the same shape of error as the D1 bracket collapse: a number that
> came out of one constraint, presented as though both constraints agreed on it. Correcting it.

**Re-costing at the ratio the literature actually supports.** Same geometry, same 0.5 B/weight, same
21.25 GB/s (still carrying its §2.1 caveat, and D5 has not yet discharged it):

| attention layers | share | active | bytes/token | **tok/s** |
|---|---|---|---|---|
| 4 / 80 | 5.0% | 1.73 B | 0.866 GB | **24.5** ← §2.2's target, *unvalidated ratio* |
| 6 / 80 | 7.5% | 2.03 B | 1.017 GB | **20.9** |
| 7 / 80 | 8.8% | 2.19 B | 1.093 GB | **19.5** |
| **10 / 80** | **12.5%** | **2.64 B** | **1.319 GB** | **16.1** ← *lowest ratio anyone has measured* |
| 20 / 80 | 25.0% | 4.15 B | 2.074 GB | 10.2 |

Solving for the constraint directly: **">20 tok/s" requires attention on ≤ 8.3% of layers** (≤6.6 of 80),
because the FFN half already consumes 1.128 B of the 2.125 B active budget that 20 tok/s allows.

> **The budget does not close at any ratio the literature has validated.** At 12.5% we get 16.1 tok/s.
> To reach the Owner's target we must go **~1.5× below the lowest measured point, along a curve that was
> still getting worse when the measurements stopped.**

#### The consequence nobody had noticed: this makes D5 load-bearing, not a side question

§2.4 raised 5-trit packing (1.6 bits/weight instead of 4) as a possible 2.5× at donor scale, conditional
on the path being bandwidth-bound. Re-run the same table under that packing:

| attention layers | share | 4-bit / 2-trit | **1.6-bit / 5-trit** |
|---|---|---|---|
| 4 / 80 | 5.0% | 24.5 | 61.3 |
| 6 / 80 | 7.5% | 20.9 | 52.2 |
| **10 / 80** | **12.5%** | **16.1** | **40.3** |
| 20 / 80 | 25.0% | 10.2 | 25.6 |

> **If 5-trit packing holds at donor width, the target closes at 12.5% — the ratio the literature has
> actually measured — with 40 tok/s and no extrapolation at all. It even closes at 25%.**
> **If it does not hold, we must push attention below every published data point.**

**This is a reframe of what D5 is for.** I dispatched D5 as a packing-efficiency question. It is not. It
is the question of **whether this programme has to out-perform the published literature on hybrid ratio,
or merely match it.** That is a different order of risk, and it is now the most decision-relevant open
measurement we have. D5's control-1 failure is therefore blocking more than a footnote.

#### Method costs, restated with the new evidence

Marked as **my** derivations. Conversion assumption: fp16 dense tensor-core peak, T4 65 / A100 312 /
H100 495 TFLOPS, **at equal MFU** — optimistic, since larger GPUs typically achieve better MFU, so these
T4-hour figures are if anything too low.

| method | quality retention | cost, my est. at 100B | passes S4 (≥90%)? |
|---|---|---|---|
| MOHAWK | 62.6 vs 64.9 = **96.5%** `[T, Table 1]` | ~10,250 T4-h (~114 wk) | **yes** |
| LoLCATs | **~83%** — gap of 10.8–11.2 MMLU points `[T]` | ~416–853 T4-h (~5–9 wk) | **NO** |
| **R2 closed-form** | **unknown** | **~0 — no training** | to be measured |

Two things to flag rather than smooth over:

1. **LoLCATs' quality gap does not shrink with scale.** 10.8–11.2 MMLU points **at every scale measured**.
   That is not a small-model artefact that a bigger donor fixes — it is the method's ceiling, and it sits
   below the sealed S4 constraint. **The cheap published method is disqualified on quality, not on cost.**
2. **My two LoLCATs anchors disagree.** Scaling 6.5 A100-h @ 7.5B linearly in N gives ~416 T4-h at 100B;
   scaling 454 H100-h @ 405B gives ~853. **A 2× disagreement means the cost is not linear in N** — token
   budget or adapter rank is growing with model size. I am reporting the range, not picking the flattering
   end. Neither number may be quoted as *the* cost.

#### What this does to the fork in §2.2b

The fork was: (1) closed-form conversion, (2) accept a smaller donor, (3) keep more attention and lose
speed. The Owner adjudicated **(2) as the deliverable and (1) as the research question, in parallel**,
and stated (1) is the priority.

**This amendment sharpens why that was the right call, and adds a fourth term the fork did not have.**
Option (3) — "keep more attention and lose speed" — was priced at 10.2 tok/s at 25% and looked like a
clear loss. **Under 5-trit packing it is 25.6 tok/s and clears the target.** So the fork is no longer
purely between conversion quality and speed; **packing efficiency is a third axis that can buy back the
quality budget**, and it costs no training at all. It is the cheapest lever on the board *if* D5 says it
exists.

**What is NOT established:** the 12.5% floor is the *lowest measured*, not a demonstrated floor — nobody
has shown 5% fails, only that nobody has shown it works. The 5-trit rows are all conditional on D5, whose
control has not yet passed. And R2's quality column is genuinely unknown; the pre-registered thresholds in
`BRIEF_R2_CLOSED_FORM_VALUE_SOLVE.md` §3.2 exist so that it cannot be quietly filled in with a hope.

### 2.2d ⚠ AMENDMENT (2026-08-22) — "FFN at 2% active" is unreachable with a DENSE FFN, and we measured that ourselves

§2.2c corrected the attention half. Checking the FFN half against our own measurements rather than
against the target, it is worse — and **the disconfirming number was already in our repository.**

`docs/ENGINE_PLAN.md:56`, **E3 CLOSED, measured in the engine on real weights**, not estimated:

> *"Honest headline = weight-bytes touched, not compute-time: gate 100% + up 21.5% + down 12.3% =
> **44.6%** → **2.24×** fewer MLP weights touched/token"*

**§2.2 assumes the FFN runs at 2% active. Our own engine measured 44.6%** — a factor of 22 apart, in the
direction that breaks the budget.

#### Why, and why it is structural rather than a tuning failure

Look at which organ costs what. In a SwiGLU FFN, `h = SiLU(W_gate·x) ⊙ (W_up·x)`, then `y = W_down·h`.
**You cannot know which entries of `h` are zero until you have computed `W_gate·x` in full.** The gate is
the predictor, and a predictor that reads every weight it is predicting for saves nothing on itself.

> **So a dense FFN with per-neuron activation sparsity has a hard floor of 1/3 of its weights — the gate
> matvec — no matter how sparse the activations get.** `up` and `down` can approach zero; `gate` cannot.
> 33.3% is the floor, 44.6% is what we actually measured.

**This is architectural, and it does not improve with scale, sparsity, or a better kernel.**

#### What that floor costs at donor scale

| FFN regime | attention | packing | active | **tok/s** |
|---|---|---|---|---|
| measured 44.6% | 4/80 | 4-bit | 25.76 B | **1.65** |
| measured 44.6% | 4/80 | 5-trit | 25.76 B | **4.12** |
| measured 44.6% | 10/80 | 5-trit | 26.66 B | **3.98** |
| structural floor 33.3% | 4/80 | 5-trit | 19.40 B | **5.48** |
| **MoE-structured, 2% active** | 4/80 | 4-bit | 1.73 B | **24.5** |
| **MoE-structured, 2% active** | 10/80 | 5-trit | 2.64 B | **40.3** |

> **The dense-FFN route tops out at ~5.5 tok/s under the most favourable assumptions available to it —
> best-case attention ratio, best-case packing, and the theoretical gate floor rather than the measured
> number. It cannot reach 20 tok/s by any combination of the levers we have.**

I checked the one escape route we have in-house evidence for. The Inventor side-lab found **`x_proj`
low-rank `r=26` beating the dense projection 3/3 seeds at 17.6% of the bytes** — so a low-rank *predictor*
replacing the dense gate matvec is not a fantasy on this codebase. Pricing it optimistically (gate at
17.6% of its bytes, `up`/`down` unchanged) gives an FFN at 17.1% active → **~10 tok/s**. Better, and still
short. **The gate predictor is not the thing that closes this.**

#### The conclusion, stated as the constraint it is

> **The 2% FFN target is not reachable by inducing activation sparsity in a dense FFN. It requires the
> donor's dense FFN to be RESTRUCTURED into an MoE**, where a small router selects a few experts and the
> unselected experts' weights are never read — so there is no full-width gate matvec to pay for.

That is not a new idea on this project; it is what `project_probe4_moe` promoted and what
`SCALEUP_ARCHITECTURE.md` already assumes (recall + MoE streamed from DRAM). **What is new is that it is
now a requirement rather than a preference**, and that the arithmetic above says how much rides on it.

#### This reframes D0, exactly as §2.2c reframed D5

`BRIEF_D0_COACTIVATION_PERMUTATION.md` asks whether a donor's FFN neurons can be permuted so that
per-token activation becomes **block-structured** — i.e. whether the neurons that fire together can be
gathered into groups that can be skipped wholesale. **That is the MoEfication question, and it is
therefore the probe that decides whether the FFN half of this budget is achievable at all.**

> **Both "side" probes turned out to be the load-bearing ones. D5 decides whether the attention half must
> beat the published literature or merely match it. D0 decides whether the FFN half exists.** Neither was
> framed that way when I dispatched them, and that is a planning error on my part worth recording: I
> priced the target before auditing our own measurements against it.

D0 is currently held only because it would contend with D5 for a quiet machine. **It is not lower
priority — it is queued behind an instrument, and it goes out the moment D5 releases the machine.**

#### What is NOT established here

The 44.6% figure was measured **on our own 8.3M model, in our engine, with our dReLU-trained checkpoint** —
not on a donor. A donor pretrained with SiLU may have a different sparsity profile, which is precisely
what the Researcher's activation-sparsity survey and D0 are for. **The structural argument — the gate is
unskippable because it is the predictor — is architecture-level and does transfer;** the specific 44.6%
does not, and must not be quoted as a donor number.

Also unestablished: that MoEfication of a donor works at all without training. `STRUCTURED_SPARSITY_PRIOR_ART.md`
records the tally as **one negative (Apple) and one positive (Neuralink), unreconciled** — and the earlier
claim of "two independent negatives" was my own miscount, already corrected there. **The FFN half rests on
an open question with one supporting data point.** That should be stated plainly to the Owner rather than
carried as an assumption.

### 2.2e THE EXTRAPOLATION LEDGER — every quantity the ">20 tok/s" claim rests on, at its validated value

§2.2c and §2.2d each found the same defect from a different side: **a required value was written into the
budget without checking it against anything that had been measured.** Rather than keep discovering this
one lever at a time, here is the whole dependency list in one table, each with the best value anyone —
the literature or us — has actually demonstrated.

| # | quantity | **validated value** | source | required by §2.2 |
|---|---|---|---|---|
| 1 | attention hybrid ratio | **12.5%** — and degradation still worsening there | lowest ratio in any donor-conversion paper | 5% |
| 2 | FFN active fraction, dense + activation sparsity | **44.6%** (floor 33.3%) | `ENGINE_PLAN.md:56`, in-engine | 2% |
| 3 | FFN active fraction, MoE-structured | **25%** (E32×h128, top-8) | probe-4, our own | 2% |
| 4 | packing | **4 bits/weight** shipping | engine today | 1.6 bits (5-trit) |
| 5 | rate constant | **21.25 GB/s, now known upward-biased** | D5 amendment A1.2 | assumed exact |
| 6 | attention conversion quality | 96.5% at 1.5B (MOHAWK, *with* training) | `[T]` Table 1 | ≥90% at 100B, training-free |
| 7 | training-free MoEfication of a donor | **1 positive, 1 negative, unreconciled** | `STRUCTURED_SPARSITY_PRIOR_ART.md` | assumed to work |

**Setting every lever to its validated value at once:**

| configuration | active | bytes/token | **tok/s** | vs target |
|---|---|---|---|---|
| attention 12.5%, FFN 25% (probe-4), 4-bit | 15.61 B | 7.80 GB | **2.72** | **7.3× short** |
| attention 12.5%, FFN 25% (probe-4), 5-trit | 15.61 B | 3.12 GB | **6.81** | **2.9× short** |

> **This is the honest floor of the programme as currently evidenced: 2.7 tok/s on what ships today.**
> Every figure above 20 in this memo — including §2.2's headline 24.5 — depends on at least one lever
> exceeding anything that has been measured.

#### But the ledger also contains the way out, and it is not a wish

Row 3 deserves a second look, because **reading probe-4's 25% as a property of the method is a mistake I
nearly made.** Probe-4 validated an expert *granularity*: **expert size `h=128`, `top-k=8`.** Its FFN was
`d_ffn = 4096`, so that granularity meant `E = 32` experts and `8×128 / 4096 = 25%` active.

**A donor's FFN is `d_ffn = 28672`. Hold the validated expert size and `k` fixed and let `E` grow with
width — which is what "fine-grained" means — and the same configuration gives `E = 224` experts and
`8×128 / 28672 = 3.57%` active.** The 25% was never a property of the recipe; it was a property of a
4096-wide FFN.

| configuration | active | bytes/token | **tok/s** |
|---|---|---|---|
| attention 12.5%, FFN 3.57% (probe-4 granularity at donor width), 4-bit | 3.52 B | 1.762 GB | **12.1** |
| **attention 12.5%, FFN 3.57%, 5-trit** | **3.52 B** | **0.705 GB** | **30.1** ✅ |
| attention 5%, FFN 3.57%, 5-trit | 2.62 B | 0.524 GB | **40.6** ✅ |

> **At the literature's lowest measured attention ratio, at probe-4's own validated expert granularity
> carried to donor width, and with 5-trit packing: 30.1 tok/s. The budget closes without asking any lever
> to beat its demonstrated value — except packing.**
>
> **At 4 bits/weight the identical configuration gives 12.1 tok/s and does not close.** So 5-trit is not an
> optimisation. **It is the difference between a plan that closes on demonstrated values and one that does
> not.** D5 is the single highest-leverage measurement in the programme.

#### What this reframing does and does not buy

**It does not make row 3 validated at donor scale.** Probe-4 measured quality at `top-8 of 32`; the donor
configuration is `top-8 of 224`. The expert *size* and `k` are unchanged, but **the fraction of the FFN
each token sees falls from 25% to 3.57%**, and nobody has shown quality holds there. That is a real open
question — but it is a **far more natural extrapolation than "make the experts 3.7× smaller"**, and it is
directly testable.

**It does sharpen what to test, and in what order:**

1. **D5** — does 5-trit packing pay at donor width? Rows 4 and the entire ✅ column depend on it.
2. **D0** — can a donor's dense FFN be carved into experts at all, without training? Row 7 has one
   supporting data point.
3. **Expert-count scaling** — does `top-8 of 224` retain quality the way `top-8 of 32` did? **This is a new
   probe that did not exist on the board this morning, and the ledger is what surfaced it.**
4. **R1/R2** — the attention half, already in flight.

#### Standing caveats that apply to every number above

The rate constant (row 5) is **upward-biased by an unquantified amount** — see `BRIEF_D5` amendment A1.2 —
so every tok/s figure in this memo, including the ✅ rows, is conditional on a constant that has not been
verified. The 44.6% of row 2 was measured on our 8.3M model, not a donor. Row 6's 96.5% was obtained *with*
3B tokens of training, on Phi-1.5, on a dense fp16 backbone — **not training-free, not at 100B, and not on
a ternary sparsified backbone**; whether it composes with rows 2–4 is untested, and this project has
repeatedly measured that sequential stages do not compose.

**Nothing here is a promotion. It is an inventory of what would have to be true.**

### 2.2f ⚠⚠ CORRECTION (2026-08-22) — converting a layer's attention does NOT remove its weights, and §2.2 was built on the assumption that it does

Returned by the Controller against `BRIEF_R2`, auditing this memo as instructed. **It is the most
consequential error in this document and it invalidates the ✅ rows of §2.2e.**

#### The error

§2.2 decomposes the budget as *"attention retained on 4 of 80 layers, the other 76 converted to a
recurrent operator"*, and then counts **only the 4 retained layers' weights**. That silently assumes the
76 converted layers' attention weights stop being read.

**They do not.** Every linearisation method in the literature — and R2 by explicit construction — **reuses
the donor's `W_q`, `W_k`, `W_v`, `W_o`.** LoLCATs freezes all four and adds LoRA; Mamba-in-the-Llama reuses
the QKV projections; R2 keeps `W_q`/`W_k` fixed and *solves* for `W_v`. A linearised layer still projects
every token through four matrices.

> **So the attention weight stream is `L × 151M`, always, regardless of the hybrid ratio.**
> At `L = 80` that is **12.08 B parameters active every token**, before the FFN contributes anything.
>
> **`4/80` and `10/80` give the same answer: ~3.2 tok/s.** The entire 5%-vs-12.5% analysis in §2.2c is
> about a term that barely moves the weight budget.

#### A second error in the same table: the geometry is not 100B

`d_model=8192, d_ffn=28672, L=80` totals **68.5 B**, not 100 B. A real 100B-class model at this width needs
**L ≈ 117**. Every row in §2.2, §2.2c and §2.2e was computed on a 68.5B model wearing a 100B label.

**Both errors point the same way: they made the programme look better than it is.** Per the standing rule
that a deviation flattering the hypothesis earns more scrutiny — this is the third time today that rule has
paid, and this time against me twice in one table.

#### The corrected arithmetic

FFN at probe-4's granularity carried to donor width (3.57% active, §2.2e), attention projections retained
on **all** layers because that is what conversion actually does:

| geometry | packing | active | bytes/token | **tok/s** |
|---|---|---|---|---|
| L=80 (68.5B) | 4-bit | 14.09 B | 7.05 GB | **3.0** |
| L=80 (68.5B) | 5-trit | 14.09 B | 2.82 GB | **7.5** |
| **L=117 (a real 100B)** | 4-bit | 20.61 B | 10.31 GB | **2.1** |
| **L=117 (a real 100B)** | **5-trit** | **20.61 B** | **4.12 GB** | **5.2** |

> **The ✅ rows of §2.2e are withdrawn. On corrected geometry, with every other lever at its most
> favourable evidenced value, a 100B donor gives 5.2 tok/s — not 30.**

#### What linearisation actually buys, which is real and which I mis-attributed

The Controller's third omission: **KV traffic is absent from this memo entirely.** At 128K context with 10
attention layers it is **5.37 GB/token against 1.32 GB of weights** — four times the weight stream.

> **Linearisation does not buy weight bytes. It buys the KV cache — and at long context the KV cache is
> the larger of the two by far.** Stage 0 said exactly this (`e9503e3`: *"the KV read path at 128K is the
> wall"*) and I then wrote a weight-only budget and forgot it.

**Consequence for how this memo must be read from now on: a tok/s figure without a stated context length
is meaningless.** At short context the weight stream dominates and the tables above apply. At 128K the KV
path dominates and linearisation is the whole game. **§2.2's model — `tok/s = 21.25 ÷ weight-bytes` — is
a short-context model and was never labelled as one.**

The Controller also flags the model itself, not just its constant: `21.25 ÷ bytes` **is a bandwidth model,
fed by a constant measured in a regime that may be compute-bound.** I have been flagging the constant
(§2.1) and treating the model as sound. **Both need to be conditional, and D5 speaks to both.**

#### The constructive result — solving the corrected equation the other way round

Rather than ask "what tok/s at 100B", ask "**what size closes at 20 tok/s**", 5-trit, FFN at probe-4
granularity:

| attention projections | per-layer active | **donor size that reaches 20 tok/s** |
|---|---|---|
| **full** (what conversion actually leaves) | 176.2 M | **~26 B** |
| **low-ranked to 17.6% of bytes** | 51.7 M | **~88 B** |

The second row is not invented. **The Inventor side-lab measured `x_proj` low-rank `r=26` beating the dense
projection 3/3 seeds at 17.6% of the bytes** — on a projection, on this codebase, with controls. Applying
that to `W_q`/`W_o` (which are 89% of the attention block: 67.1M each of 151M) is a **new lever, with
in-house precedent, that this memo did not have this morning.**

> **The path to a 100B-class donor at >20 tok/s runs through low-ranking the attention projections.**
> Without it, the honest ceiling for this engine and this rate constant is a **~26B donor** — which is a
> real, defensible target and considerably better than the 1.5B the affordability fork was pointing at.

**Note carefully what row 1 does NOT say.** It does not say 26B is the limit of the *method*; it says it is
the limit of the method *with attention projections left at full width*. Those are different claims and
only the second is evidenced.

#### The KV term, now supplied — and it inverts the picture at long context

Verified independently: at 128K, GQA 8 KV heads, `head_dim=128`, fp16, the whole cache re-read per token is
**536.9 MB per softmax layer per token** (10 layers = 5.37 GB, matching the Controller). **KV ACCOUNTING**,
5-trit, FFN at 3.57%:

| geometry | weights/token | softmax layers | KV/token | total | **tok/s @128K** | **tok/s @short ctx** |
|---|---|---|---|---|---|---|
| L=117 (real 100B) | 4.12 GB | 10 | 5.37 GB | 9.49 GB | **2.24** | 5.15 |
| L=117 | 4.12 GB | 4 | 2.15 GB | 6.27 GB | **3.39** | 5.15 |
| L=117 | 4.12 GB | **0** | 0 | 4.12 GB | **5.15** | 5.15 |
| **L=30 (~26B target)** | **1.06 GB** | 10 | 5.37 GB | 6.43 GB | **3.31** | 20.10 |
| **L=30** | **1.06 GB** | 4 | 2.15 GB | 3.20 GB | **6.63** | 20.10 |
| **L=30** | **1.06 GB** | **0** | 0 | **1.06 GB** | **20.10** | **20.10** |

> **At the ~26B target the KV of just FOUR retained softmax layers is 2.15 GB — twice the entire weight
> stream of the whole model.** Retained attention stops being a small concession and becomes the dominant
> cost at long context.

**So the honest statement of the design tension, which no section of this memo had until now:**

- **At short context** the weight stream rules, and ~26B reaches **20.1 tok/s** with any hybrid ratio.
- **At 128K** the hybrid ratio rules, and **>20 tok/s requires ZERO softmax layers — full linearisation.**
- **But the Researcher established that full linearisation surrenders retrieval** — the fixed-state recall
  cap is architectural, and hybrid softmax layers are the published answer to it.

> **The speed target and the retention target pull in opposite directions on exactly the same knob, and
> only at long context.** That is the central design conflict of this programme, and it was invisible while
> the KV term was missing. Any claim of "N tok/s" from here on **must state its context length**, and any
> hybrid ratio must be defended on retrieval evidence, not chosen for speed.

#### What this changes on the board

1. **A new probe exists: low-rank attention projections on a donor.** D1 measured static *sparsity* on
   q/k/v/o as catastrophic — **low-rank is not sparsity**, and the one in-house datapoint on projections is
   positive. This should be costed and pre-registered.
2. **§2.2c's hybrid-ratio analysis is demoted**, not deleted: the ratio governs KV traffic and quality, not
   the weight stream. It matters at long context and for retention; it is nearly irrelevant to bytes.
3. **D5 becomes more load-bearing, not less.** Every corrected row above still doubles or better under
   5-trit, and the ~26B and ~88B figures are both 5-trit figures.
4. **The Owner's fork should be restated.** "1.5B demonstration vs 100B research question" was framed when
   the budget appeared to close at 100B. The corrected arithmetic suggests a **~26B target closes on
   evidenced levers**, which is a different and better third option than either arm of the original fork.
   **That is a sealed-scope question and I am not deciding it — I am putting it up.**

### 2.2g ⚠⚠ CORRECTION (2026-08-22, later) — §2.2e's "probe-4 granularity at donor width" is withdrawn: `h=128` is not scale-free

Returned by the Builder against `BRIEF_M1`. **This kills the reframe §2.2e was built on, and with it the
FFN half of the ~26B and ~88B targets.**

#### The error

§2.2e argued that probe-4's `25% active` was "a property of a 4096-wide FFN, not of the method", and that
holding the **validated expert size `h=128` and `k=8`** while letting `E` grow with width gives `3.57%`
active at donor scale. **`h=128` is not a property of the method either.**

> **`h = 128` is `d_model/2` at probe-4's width and `d_model/64` at donor width.** I swapped one
> width-dependent constant for another and called it scale-free.

The quantity that *is* scale-free is **`α` = active FFN neurons per token ÷ `d_model`** (equivalently
`α = r·a`, where `r = d_ffn/d_model` and `a` is the active fraction):

| | `r` | `a` | **`α`** |
|---|---|---|---|
| probe-4 `dense-big` (arm B) | 16 | 100% | 16.0 |
| **probe-4 `moe-gran` — the VALIDATED point** | 16 | 25% | **4.0** |
| **a donor's stock dense FFN** | 3.5 | 100% | **3.5** |
| what our budget needs | 3.5 | 3.57% | **0.125** |

> **Probe-4's validated MoE endpoint is still FATTER per token than a normal dense FFN.** `α = 4.0` against
> a donor's `α = 3.5`. **Probe-4 never entered the regime we need — it never went below dense-normal active
> width, and we need to go 32× below its endpoint.**

Note also that `α = 4.0` is **unreachable** at a donor's `r = 3.5`: it would require `a = 114%`.

#### What survives, and what it does to §2.2f

**§2.2f's conclusion is unaffected and in fact strengthened.** Attention projections dominate the active
weight stream, and the thinner the FFN gets the more they dominate:

| FFN active fraction | active @L=117 | **attention's share** | tok/s (4-bit) |
|---|---|---|---|
| 100% (stock dense) | 100.11 B | 17.6% | 0.42 |
| 25% (probe-4's `a`, at donor `r`) | 38.28 B | 46.2% | 1.11 |
| 3.57% (the withdrawn assumption) | 20.61 B | **85.7%** | 2.06 |

**So the FFN half is not what governs the weight budget — attention is, at every plausible FFN setting.**
The FFN number moves tok/s by ~5× across its whole plausible range; the attention projections are an
irreducible 17.67 B at `L=117` that no FFN work touches.

#### Two findings about probe-4 itself that are worse than the withdrawal

The Builder audited probe-4's apparatus while sizing M1, and returned two things that bear on a result this
project has carried as **VALIDATED** in `SCALEUP_ARCHITECTURE.md` §3.5:

1. **Probe-4's entire measured advantage — 0.0085 BPB — is BELOW the 2-seed minimum detectable effect
   (0.010).** The result was underpowered against this project's own `σ_seed = 0.005`.
2. **Probe-4 never ran a frozen-random-router control, and the router-health metrics it did report cannot
   substitute for one.** `dead` and `max/mean` measure *marginal load*, which a uniform-random router
   **maximises** — a router that learned nothing would post perfect health. **So probe-4's win is not
   separable from "the MoE arm has a multiplicative gate the dense arm lacks"**, a candidate cause its own
   memo lists.

> **This does not overturn probe-4. It means probe-4's status as a validated positive rests on less than
> the ledger in §2.2e credited it with**, and any M1 design must run the frozen-random-router twin — which
> would retro-audit probe-4 at the same time.

#### The instrument defect M1 would have walked into, which is a D1 repeat

`MoEMLP._tern` computes the per-expert `down` scale as `W.abs().mean(-1)` — an absmean **over the `h`
axis**. At small `h` that is an average over a handful of values, so **the ternary quantiser degrades
monotonically with the very variable M1 sweeps, for reasons that have nothing to do with routing** — and
the artefact does not exist at the donor's width. Without an fp32 twin, M1 would have measured its own
quantiser and reported it as a granularity limit. **This is the same shape as D1's requested-vs-achieved
sparsity, found before the run rather than after.**

#### Where this leaves the FFN half

**Unknown, and honestly so.** The literal question ("does `top-8 of 224` hold?") is unanswerable at any
affordable scale — reaching `E=224` at `h=128` absolute requires `d_model=8192`, a **6.56 B-parameter
model** in this apparatus. **That is an impossibility, not a budget complaint.**

The **scale-free** question — *how far down the `α` curve does the FFN hold at `r=3.5`, at matched total
params* — **is answerable for ~60–150 T4 GPU-hours.** Probe-4 measured exactly one point of that curve, at
the fat end.

> **So the FFN half is a measurable unknown rather than a wall — but it is 32× beyond the only point
> anyone has measured, and that point is itself underpowered and missing its key control.**

### 2.3 What this says about the sealed constraints

- **S3 ("attention on a minority of layers") is now quantified and it is far more demanding than
  "minority".** 4 of 80 is **5%**, not 49%. The earlier stage-0 finding — full attention retained on
  ≤2 of 28 layers, windowed on ≤18 of 28 — is consistent with this and is now explained by it rather
  than being an isolated measurement.
- The KV-traffic wall found at stage 0 is the same constraint seen from the other side. It is not a
  separate problem to be solved after the density problem; **it is the attention half of the same
  budget.**

---

## 2.4 "Va rifatta la lookuptable" — the packing question, and why it may reverse at donor scale

The Owner's mandate names the lookup table explicitly as a thing that may need redoing. It has a concrete
arithmetic behind it that has not been written down in this programme.

The engine packs **two trits per byte** — `(w0+1)*3 + (w1+1)`, nine states in eight bits — giving
**4 bits/weight, 0.5 bytes/weight**. The information-theoretic floor for ternary is `log2(3) = 1.585`
bits, and the natural dense packing is **five trits per byte** (`3^5 = 243 ≤ 256`), i.e. **1.6 bits/weight,
0.2 bytes/weight**.

> **That is a 2.5× reduction in bytes read per weight.**

On the engine as measured today, that is a **loss**, and the project has already banked why: the LUT path
is **compute-bound by ~16×**, and the related finding on nibble-packing was *"+10% compute for −50% bytes
on a compute-bound path = ~10% slower"*. Denser packing buys footprint, not speed. Correct — **at
`D = 1536`, with a 2.25 MiB working set that is L3-resident.**

**The donor case is not that case.** A 100B donor at 0.5 bytes/weight is **50 GB**, streamed from DRAM.
Nothing about that is L3-resident, and a path that streams tens of gigabytes per second from main memory
is bandwidth-bound almost by definition. **If the path reverts to bandwidth-bound at donor scale, the sign
of the packing trade-off flips and 5-trits-per-byte becomes a ~2.5× speedup rather than a ~10% penalty.**

Compounded with §2.2's worked target, at 0.2 bytes/weight the same 1.73 B active weights become 0.35 GB
per token — **~61 tok/s** on the same rate constant. That is the difference between meeting the Owner's
">20 tok/s" and comfortably exceeding it.

**This is not a claim. It is a conditional that D5 resolves**, because D5's `rate(D, threads)` curve with
working-set sizes and an L3-residency flag is exactly the measurement that says whether the path is still
compute-bound at donor width. Sequence matters:

1. **D5 first** — is the path compute-bound or bandwidth-bound at donor projection widths?
2. **Only if bandwidth-bound**, a follow-on brief on the 5-trit packing: decode cost per byte, whether
   `pshufb` can still serve it (a 243-state decode does not fit a 16-byte shuffle table the way a 9-state
   one does — this is the hard part and it may not be free), and end-to-end parity.
3. The parity rule is absolute and is banked from Phase 60: **kernel-bit-exact does not compose to
   system-correctness; end-to-end parity always.**

Recorded here so that the packing question is not re-derived from scratch later, and so that its
dependency on D5 is explicit rather than assumed.

---

## 3. What is now known, and at what confidence

| finding | status | confidence |
|---|---|---|
| Weight-basis rotation makes weights no sparser (D2) | closed, axis dead | measured, 12/12 matrices, controls fired |
| ρ-floor is `32768/D` → ~4 neurons (interleaved) / 12 (per-organ) at donor width | closed (D0b) | measured, all controls fired; **under Controller audit** |
| Static block-pruning without reconstruction is catastrophic at every level (D1) | closed for the naive baseline | measured; 2 points struck for mask quantisation; **under Controller audit** |
| dReLU activation sparsity is near-free at 92%/79% | banked (probe-2) | measured **on a model we trained ourselves**, not on a donor |
| Block-structuring activation sparsity is free (18%→50% @ BS8) | banked (Phase 58.B) | same caveat |
| No published work solves ternary + contiguous-block jointly | verified | primary-source search; every adjacent cell populated |
| No published work targets CPU/DRAM at tens-of-KB block granularity with both a quality table and a contiguous-read-size table | verified | the two closest (Neuralink, LLM-in-a-Flash) are calibrated to smartphone flash pages, not CPU cache economics |
| **MoEfication physically permutes the FFN so experts are CONTIGUOUS submatrices, losslessly, with no backbone training** | **promoted as the layout mechanism** | primary source, equations in §3.2; the mechanism we need, already published |
| **Conditional computation pays on memory-bound CPUs and largely does not on GPUs** | **corroborated externally** | MoEfication Table 1 `[T]`: at 12.5% activation, **CPU 2.28× vs GPU 1.47×** — measured wall-clock, not projected |
| Router cost when FFN semantics are preserved | −0.3 average at 12.5% activation | MoEfication Table 4 `[T]`, T5-Large. **12.5% is not 2%; T5 is not a 100B donor** |
| **Cheap MoE-ification with a fresh router destroys general-purpose retention** | **measured, and it fails S4 badly** | LLaMA-MoE-v2 Table 1 `[T]`: 7B tokens → **MMLU 67.22 → 37.41**, IFEval 76.53 → 32.72 |
| Co-activation grouping: **1 negative, 1 positive, UNRECONCILED** | changes how D0 must be run | Apple abandoned it (*"worked against our original intention"*); Neuralink reports it works and **cites Apple but never engages the negative**. ⚠ **Corrected 2026-08-22:** I first counted LLaMA-MoE as a second negative — it is not; its losing arm was weight-space k-means on `W_up` rows, a different construction, and untabulated `[X]` |
| Sparse Upcycling as a density tool | **rejected** | each expert is a full FFN copy → capacity scaling at equal-or-greater per-token cost. Wrong direction |
| Joint solver cost at 100B on a T4 | ~15–40 h | three independent anchors |
| **How much reconstruction recovers of D1's loss** | **OPEN — brief D4 running** | the number the programme turns on |
| **Whether donor activation sparsity can be induced, and at what token cost** | **OPEN — Researcher running** | the number the 2% target turns on |
| **Whether 21.25 GB/s holds at donor width** | **OPEN — brief D5 written and pre-registered** | every tok/s figure depends on it |
| **Whether the LUT path is still compute-bound at donor width** | **OPEN — D5 answers it** | decides the sign of the packing trade-off (§2.4) |
| 5-trits-per-byte packing: 0.2 vs 0.5 bytes/weight | **conditional on D5**, not yet a brief | ~2.5× on bytes; a loss today, potentially a 2.5× gain at donor scale |

---

## 3a. A structural finding from D4's pre-registration — healing must happen in the CONSUMING layer

The Builder derived this **before running D4** and recorded it in the pre-registration, which is where a
finding like this has to appear if it is to be trusted:

> D1's masks make `gate_proj`/`up_proj`/`q`/`k`/`v` **ROW-structured** — whole output neurons zeroed — and
> `o_proj`/`down_proj` **COLUMN-structured** — whole input features zeroed. The row-separable
> SparseGPT-style solve only has freedom to act when the surviving set `S` varies *within* a row's
> support. For a ROW-structured organ every kept row already has full support and every dropped row has
> none: **the closed form is mathematically forced to return exactly `W` or exactly `0`, independent of
> `H`.**

**Read what that means for the programme, not just for D4.** Zeroing an output neuron is exactly what the
engine needs — a dead neuron's weights are the ones we skip. And single-layer reconstruction on the organ
that produced that neuron **cannot recover anything, by construction**. Not by failure of the method; by
the shape of the problem.

The recovery, if it exists, has to come from the **consuming** layer: a neuron killed in `gate_proj` is a
dead *input column* of `down_proj`, and there the surviving set does vary within each row, so the solve
has real freedom. **For structured neuron removal, healing belongs in the layer that reads the neuron,
not the layer that writes it.**

Three consequences, recorded now:

1. **The dossier's Tier-2 multi-layer Schur-complement solve is not an optional refinement.** For
   row-structured sparsity it is the *only* place the correction can live. I previously catalogued Tier 2
   as budget spent on an unestablished drift law (§2.2 of the adjudication); that criticism was about the
   *drift justification*, and it stands — but the multi-layer solve turns out to be load-bearing for a
   completely different and better reason than the one the dossier gave for it.
2. **D4's `gate_proj` arm can only return zero recovery**, and that zero is a theorem, not evidence about
   the method. It must not be averaged in with the column-structured organs.
3. **Any future brief that prunes whole neurons and then reconstructs the same layer is malformed.**
   Write it as prune-in-layer-ℓ, reconstruct-in-layer-ℓ+1.

---

## 3b. Amendment to brief D0, issued before D0 is dispatched

`DONOR_V2_DENSITY_PROGRAMME.md` §6 specifies D0 as co-activation clustering of FFN neurons with a
mandatory random-clustering control. The prior art (`STRUCTURED_SPARSITY_PRIOR_ART.md` §5) has changed
what that control means:

> **The random-clustering arm is not a formality — one published attempt at co-activation bundling
> failed outright, and the one success cites that failure without explaining it.** The literature is
> **split and unreconciled**, not stacked against us. If our co-activation arm shows an advantage we are
> resolving an open contradiction; if it shows none, we have replicated Apple. Either is worth reporting.
>
> *(Corrected 2026-08-22: I originally wrote "two published results predict it will lose", counting
> LLaMA-MoE. Wrong — LLaMA-MoE's losing arm was weight-space k-means on `W_up` rows, not co-activation
> grouping, and it is untabulated.)*

And Apple's stated failure mode is a concrete, checkable hypothesis rather than a vague warning: *highly
active neurons get duplicated across bundles.* **D0 must therefore measure and report neuron duplication
across clusters.** A probe reporting only cluster quality cannot distinguish "co-activation clustering
does not help" from "our implementation hit the documented failure mode" — and those two call for
opposite next moves.

---

## 3c. THE SYNTHESIS — the two things that fit our budget each lack what the other has

The Researcher's activation-sparsity pass (`ACTIVATION_SPARSITY_PRIOR_ART.md`, 1345 lines) returns one
dominating fact:

> **Every training-requiring ReLUfication method is three to four orders of magnitude outside our budget.**
> ReLU Strikes Back **30B tokens**; ProSparse **34.6–134B**; Turbo Sparse/dReLU **150B tokens on 64× A800**;
> Q-Sparse **40B**; the 2025 Sparsing Law needs **800B tokens even at 2.4B parameters**. Against
> 90 T4-hours/week, none of these is a candidate at any scale. **The continued-pretraining route to
> activation sparsity is closed for us. Not tight — closed.**

Note especially Turbo Sparse: it is the closest published analogue to our own in-house dReLU result, and
it cost **150B tokens on 64 A800s**. Our +0.0006 BPB at 92%/79% sparsity came free *because we trained
with dReLU from the start*. That is the difference between designing for sparsity and retrofitting it,
priced by someone else.

**What does fit the budget:**

| | cost | sparsity pattern | usable by our engine? |
|---|---|---|---|
| TEAL / CATS (training-free thresholding) | **≈1 GPU-hour or less** (a calibration pass) | **scattered** | **No.** Scattered sparsity is worth ~zero at 48 KB granularity |
| MoEfication permutation | training-free, lossless | **contiguous by construction** | **Yes** — but it needs a grouping, and the grouping is the contested part |

**Read the table.** The cheap method gives sparsity without contiguity. The layout method gives
contiguity without deciding sparsity. **Neither is sufficient; their composition is exactly what is
missing, and it is exactly the gap the searches keep returning empty on.**

> **The candidate route, stated as a hypothesis to be tested and not as a plan:**
> take a donor's *existing* activation sparsity, obtain it cheaply with a training-free threshold method,
> and then **permute the FFN so that the sparsity that already exists becomes contiguous** — MoEfication's
> `W̄₁ = W₁P` applied not to hand-built experts but to whatever firing structure the donor already has.
> Cost: a calibration pass plus an offline permutation. No gradient training of the backbone.

This is the first route in the programme where **both** halves are individually published, training-free,
and inside budget, and where the novelty is the composition rather than a new mechanism we would have to
invent. It also lands squarely on the two verified gaps: nobody targets CPU/DRAM at tens-of-KB block
granularity, and nobody solves ternary + contiguous-block jointly.

**And ternary is not an obstacle to it.** The Researcher surfaced two sources I had not asked for —
**BitNet a4.8** (arXiv:2411.04965) and **Sparse-BitNet** (arXiv:2603.05168) — both showing **ternary
weights compose with induced sparsity without catastrophic interaction**. Neither tests a dense-donor →
ternary+sparse *joint conversion*, so this is permission to proceed, not evidence that our specific
conversion works.

**A framing correction the Researcher made, and it is a good one:** my structured/scattered binary hides
a third tier. GPU tensor-core N:M structuring (Q-Sparse's "Block Q-Sparse", 2:4) *is* structured — at a
granularity roughly **1000× too fine** for a 48 KB read. A paper claiming "structured sparsity" must be
checked for *which* structure before it counts as relevant to us. Three tiers, not two: scattered,
GPU-block, cache-block.

---

## 3d. D4 control 3 — the reconstruction mechanism is REAL, SPECIFIC, and DANGEROUS

Measured on `down_proj@50%`, the donor's own weights, all four controls fired. Recovery is
`1 - delta_reconstructed / delta_naive` against D1's `+1.55712`:

| Hessian used | information content | ΔBPB | recovery |
|---|---|---|---|
| `identity` | none | +1.55712 | **0.000** |
| **`real`** | **correct** | **+0.80546** | **+0.483** |
| `shuffled` | **wrong** | +4.04098 | **−1.595** |

**Three readings, in order of importance.**

**(1) The mechanism is real and specific.** The gap between the real and shuffled arms is **2.08 recovery
units**. The hypothesis that any plausible-looking `H` would do — that this is a generic least-squares
refit wearing a Hessian costume — is decisively dead. The donor's actual activation correlations are
load-bearing. **This is the first direct evidence in the programme that the joint-solver dossier's central
mechanism does something.**

**(2) It is dangerous, and this is a deployment finding, not a curiosity.** A *wrong* Hessian does not
degrade gracefully toward the naive baseline — it goes to **+4.041, worse than the +3.745 saturation
control.** Reconstructing with wrong statistics is worse than deleting the organ entirely. The mechanism
is not ignorance-tolerant: a shuffled `H` keeps the right marginal variances and the wrong correlations,
so the solve applies wrong corrections *confidently*.

> **Consequence for scale-up.** At 100B, an under-sampled or mismatched calibration set does not cost us a
> few points — it can produce a model far worse than naive pruning. This is a direct argument for
> SparseGPT's `T/D ≈ 24–64×` calibration margin over the **1.83×** used here, and it retires any thought
> of economising on calibration tokens.

**(3) It does not rescue static block pruning, and that conclusion is unchanged.** `+0.805 BPB` is still
**161σ** against σ_seed = 0.005. Halving an unacceptable loss leaves an unacceptable loss — and this is
the *favourable* case: `down_proj` is COLUMN-structured, the only geometry where the row-separable solve
has any freedom at all (§3a). **The 2% target still lives entirely on the dynamic-activation lever (§1),
not on static weight sparsity.** D4's value is that it closes the question of whether reconstruction could
have changed that. It cannot.

**What is NOT established.** The shuffled arm shows that *wrong* structure hurts; it does not separate
*this layer's* structure from *any real* structure. The wrong-layer null (real `H` from a different layer
of the same donor) is commissioned and still outstanding. Also note the Controller's synthetic model of
the shuffled null predicted **+0.148** and the live measurement gave **−1.595** — the synthetic did not
predict the real behaviour, which is itself a datum about how far synthetic proxies can be trusted here.

---

## 4. Standing rules this memo re-states, because they were violated today

1. **A measured constant carries the dimensions at which it was measured.** Before reusing one at a new
   scale, rewrite its symbolic derivation and see which dimensions appear inside it. Caught twice today:
   `B_block ≥ 22` and the skippable-0.001 column. §2.1 above flags the third candidate before it bites.
2. **Do not calibrate on the numbers of the document you are auditing.** My own "413 h" came from
   adopting an implausible throughput out of the artefact under review. Every conversion needs an
   external anchor, and at least two independent ones before it enters a verdict.
3. **An instrument must report the parameter it ACHIEVED, not the one it was asked for.** D1's mask
   quantised a requested 90% to 100% on two organs; caught only because `zero_frac` was recorded next to
   `level`.
4. **A planted control must be shown to fire on a known positive before its nulls mean anything** — and
   a control that fires on everything is not a control. This is what the Controller is auditing now.
