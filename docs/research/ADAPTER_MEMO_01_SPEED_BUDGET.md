# Adapter Memo 01 — the two levers, and the speed budget that follows

**Author:** the Adapter / Principal · **Date:** 2026-08-21 · **Base revision:** `e410243`
**Status:** decision memo. Fixes the target the Builder's briefs are written against. Not a gate — the
Architect/Owner alone adjudicates sealed gates.

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

### 2.2 Where the 2% has to come from — the part nobody had allocated

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
| Joint solver cost at 100B on a T4 | ~15–40 h | three independent anchors |
| **How much reconstruction recovers of D1's loss** | **OPEN — brief D4 running** | the number the programme turns on |
| **Whether donor activation sparsity can be induced, and at what token cost** | **OPEN — Researcher running** | the number the 2% target turns on |
| **Whether 21.25 GB/s holds at donor width** | **OPEN — brief D5 written and pre-registered** | every tok/s figure depends on it |
| **Whether the LUT path is still compute-bound at donor width** | **OPEN — D5 answers it** | decides the sign of the packing trade-off (§2.4) |
| 5-trits-per-byte packing: 0.2 vs 0.5 bytes/weight | **conditional on D5**, not yet a brief | ~2.5× on bytes; a loss today, potentially a 2.5× gain at donor scale |

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
