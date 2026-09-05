# BRIEF E3 — The engine at target scale, and the 1.17 ms nobody has measured above 0.5 B

**Status:** pre-registered. Nothing in §4–§6 may be changed after the first number exists.

---

## 1. The gap this closes

The goal is **10B at 50 tok/s**. Every speed number this programme owns was taken on
**Qwen2.5-0.5B**: 56.1 tok/s, **27.7 G-weights/s delivered** (`SPEED_LEDGER.md` §12.2, idle, 3 reps).
Everything above that size is extrapolation:

- **P3** measured donor shapes up to 3B, but only the **matvec stack** — no RoPE, no attention over
  a growing context, no KV cache, no norms (`probes/P3_DONOR_SHAPE_ON_ENGINE.md`).
- **The 522 M active-weights/token budget** that screens all twelve donors in INDEX §7 is not
  `27.7 / 50`. It is `27.7 G-w/s × (20 − 1.17) ms`, where **1.17 ms/token is a reservation for the
  non-weight work** (rope + attention + norm + glue).

**The ledger names its own weak point** (§12.2, verbatim):

> *"The 1.17 ms is measured at 0.5 B and short context and is **optimistic for a 10 B**, where
> attention and the KV cache both grow."*

That sentence has been standing since it was written and has never been tested. If 1.17 ms is
really 9 ms at target scale, the budget is not 522 M, the twelve-donor screen in INDEX §7 is wrong
in the direction that matters, and the 10.3% runtime win that moved two donors under the line moved
them under a line drawn in the wrong place.

**This is E1's law pointed at the goal itself:** *a ceiling nothing has reached is a ceiling nobody
has tested.* It is how the 2 GB load limit stayed invisible until something big was handed to the
loader.

## 2. Why synthetic weights, and why that is not a shortcut

A speed measurement does not need a trained model — but it does need the claim *"timing does not
depend on weight values"* to be **true of this code**, not plausible. Reading the kernels
(`matvec`, `matvec_lut`, `matvec_lut_g`) shows `pshufb` over packed bytes and FMA accumulation with
no early exit on zero codes. **That is a reading, and this programme does not accept readings.**
It is Gate V1 below, and if it fails the probe stops.

Synthetic weights buy the one thing a real donor cannot: **shapes at 7–10B without 20 GB of
downloads per donor**, and the ability to hold every variable but shape fixed.

## 3. Arms — real donor shapes from `configs/`, plus the goal's own size

Every shape is exported as `--quant packed` with a ternary head (the runnable configuration E2
settled), `--fold layers` (the new default). Weights are synthetic; **shape, format and layout are
real**.

| arm | source | D | F | L | H/KV | hd | V | active w/token |
|---|---|---|---|---|---|---|---|---|
| `S05` | Qwen2.5-0.5B | 896 | 4864 | 24 | 14/2 | 64 | 151936 | 0.494 B |
| `S15` | Qwen2.5-1.5B | 1536 | 8960 | 28 | 12/2 | 128 | 151936 | 1.544 B |
| `S3` | Qwen2.5-3B | 2048 | 11008 | 36 | 16/2 | 128 | 151936 | 3.086 B |
| `M7` | Mistral-7B-v0.3 | 4096 | 14336 | 32 | 32/8 | 128 | 32768 | 7.114 B |
| `Q8` | Qwen3-8B | 4096 | 12288 | 36 | 32/8 | 128 | 151936 | 7.568 B |
| **`T10`** | **the goal's "es 10B"** | 4096 | 14336 | 48 | 32/8 | 128 | 32768 | **10.603 B** |

`M7` and `Q8` are a matched pair on the axis INDEX §7 says decides the head: **same width, 4.6× the
vocabulary**. `T10` is a synthetic config, marked as such wherever it appears.

Two bench lengths, because *a tok/s figure is only comparable at the same `--bench` length*
(INDEX §1): **300** (the ledger's operating point) and **800**.

## 4. Gates — fixed before the run

**Gate V1 — value-independence. The instrument must be shown not to care what the weights are.**
Two files of identical shape (`S05`): one with **every code zero**, one with **no code zero**. If
median tok/s differs by more than the IQR of the reps, synthetic weights are an invalid instrument
for timing and **the probe stops and reports that instead**. This is the planted control: it is a
known-*negative* the kernel must be blind to.

**Gate V2 — known-positive against a real artifact.** `S05` synthetic, at `--bench 300`, 6 threads,
idle, must reproduce the **measured** 56.1 tok/s of the real R3 0.5B artifact within the IQR. A
synthesizer that does not produce what the exporter produces is measuring its own bugs.

**Gate V3 — the file is the format.** Every synthetic file is read back with
`e1_bpb_through_engine.read_header` / `layout` — an independently written definition of the byte
layout — and the computed size must equal the file size exactly. The engine must also load it and
emit a finite BENCH line.

**Timing discipline (hard, from the perf memory).** *A contended timing is not a timing.* Idle
machine, **≥3 reps per point**, median reported with **IQR**; no other heavy job running. Quality is
not measured here at all — synthetic weights have no BPB worth reading.

## 5. Pre-stated predictions, written before any measurement

From the ledger's own numbers, `predicted tok/s = 27.7e9 / active_weights` if the rate is
scale-free and the non-weight cost stays at 1.17 ms:

| arm | predicted tok/s (rate holds) | predicted fixed non-weight ms/token |
|---|---|---|
| `S05` | 56.1 (anchor — the rate was derived from it) | 1.17 (measured) |
| `S15` | 17.9 | ~2.7 |
| `S3` | 9.0 | ~4.6 |
| `M7` | 3.9 | ~7.0 |
| `Q8` | 3.7 | ~7.9 |
| `T10` | **2.6** | **~9.0** |

The fixed-cost column is my own extrapolation and is the thing being tested: attention + KV traffic
per token scales roughly as `L × NKV × HD × position`, which from `S05` (24 × 128) to `T10`
(48 × 1024) is **16×** the KV bytes at equal position, against a reservation that assumes it stays
constant. **I expect the 1.17 ms reservation to fail, and I expect it to fail badly.**

**Also stated before the run, because it is arithmetic and not a finding:** at 27.7 G-weights/s a
**dense** 10B cannot reach 50 tok/s — it would need 10.6 B weights delivered in 20 ms, which is
530 G-weights/s, **19× the measured rate**. E3 is not expected to find 50 tok/s. It is expected to
say, with measurements instead of extrapolation, **how far the dense path is and where the distance
actually sits**, so that the sparsity/MoE architecture that must close it is aimed at the right
term.

## 6. Decision rule

Let `f` = measured non-weight ms/token (rope + attention + norm/glue from `--profile`) and
`r` = measured delivered G-weights/s.

| outcome | condition | consequence |
|---|---|---|
| **VOID** | Gate V1, V2 or V3 fails | nothing is read; the synthesizer is the finding |
| **RESERVATION-HOLDS** | `f ≤ 2.0 ms` at `T10`, bench 300 | the 522 M budget stands; INDEX §7's screen is sound |
| **RESERVATION-BREAKS** | `f > 2.0 ms` at `T10` | the budget is **re-derived from measured `f`** at each shape, and INDEX §7's twelve-donor screen is re-run and marked superseded |
| **RATE-HOLDS** | `r` at `T10` within ±10% of 27.7 G-w/s | the ledger's rate is scale-free over 20× of model size |
| **RATE-DEGRADES** | `r` at `T10` below 24.9 G-w/s | the extrapolation that prices this whole programme is optimistic, by the measured factor |

`RESERVATION-*` and `RATE-*` are independent and both are reported.

## 7. Reporting

`probes/E3_ENGINE_AT_TARGET_SCALE.md`. Per arm and per bench length: median tok/s with IQR over the
reps, the full `--profile` organ table in ms/token, delivered G-weights/s, measured `f`, and the
budget re-derived from that arm's own `f`. Gate V1's two numbers, Gate V2's comparison against the
real artifact, and the §6 label verbatim. The pre-stated predictions of §5 are reproduced next to
the measurements, **including the ones that turn out wrong**.

---

## 8. AMENDMENT, written after run 1's gates and before run 2 exists

Run 1 ran **the gates only**. No arm above `S05` was generated, no `T10` number exists, and nothing
in §5 or §6 has been read against a measurement. This section is written now so that the repair is
on the record before the run that uses it, as `feedback_media_manager` requires.

### 8.1 Run 1's label, by §6 verbatim

**`VOID`.** Gate V1 fired and Gate V2 failed. Per §6, *"nothing is read; the synthesizer is the
finding"*. That is honoured: §8.2 and §8.3 are the finding.

### 8.2 Gate V1 fired — and it localises to a transcendental, not to the matvec

`S05`, `--bench 300`, 6 threads, 3 reps, background load 3–8%:

| file | median tok/s | IQR |
|---|---|---|
| every code zero | **39.580** | 0.215 |
| no code zero | **36.490** | 0.140 |

3.090 tok/s apart against an IQR of ~0.2 — roughly 15×. The gate fired as written.

The organ table says where, and it is one organ:

| organ | zero | dense | Δ ms/token |
|---|---|---|---|
| `ffn` | 9.021 | 11.101 | **+2.080** |
| `head` | 13.746 | 13.818 | +0.072 |
| `qkv_proj` | 0.846 | 0.851 | +0.005 |
| `o_proj` | 0.588 | 0.585 | −0.003 |
| `attention` | 1.027 | 1.017 | −0.010 |
| `rope` / `norm+glue` | 0.016 / 0.067 | 0.017 / 0.067 | ≤0.001 |

`qkv_proj`, `o_proj` and `head` are pure packed matvecs and they are **flat to 0.005 ms** across a
change from every weight zero to no weight zero. So §2's reading of the kernel — `pshufb` + FMA, no
early exit — is not merely plausible, it is now **measured**: the matvec is value-independent.

The `ffn` bucket is the only one that contains something that is not a matvec:
`silu(x) = x / (1 + expf(-x))`, scalar `expf`, `F × L = 4864 × 24 = 116,736` calls per token. With
every weight zero the SwiGLU argument is exactly `0.0f`, which is the early-out of every libm `expf`.
2.080 ms over 116,736 calls is **17.8 ns per call** of difference — `expf` scale, not memory scale.

**What this impeaches and what it does not.** It does not impeach the matvec, which is the part that
extrapolates to `T10`. It impeaches the *design of the gate*: a trained model never has a zero
fraction of 1.0, so the control was planted at a point outside the range any real artifact occupies,
and what it detected is a property of `expf`, not of the instrument. **That is a flaw in my gate,
not a licence to ignore it**, so the `VOID` stands and the repair below is a different gate, not a
reinterpretation of this one.

The realistic half of the same experiment is worth stating: `mixed` (zero fraction 0.47, what E2
measured on real R3 exports) against `dense` (0.00) is **36.570 vs 36.490**, inside both IQRs. Across
the range a real export can actually occupy, the timing is flat.

### 8.3 Gate V2 failed — and the synthesizer is not the reason

| file | median tok/s | IQR | head ms/token |
|---|---|---|---|
| synthetic `S05`, `mixed` | 36.570 | 0.040 | 13.724 |
| **real** `qwen25-05b_tq.bin` (E2 arm TQ) | **36.440** | 0.540 | 13.829 |
| **real** `qwen25-05b_nl.bin` (E2 arm NL, R3+fold) | 36.460 | 0.050 | 13.844 |

The synthetic file and the real artifact of the same shape and the same byte count agree to
**0.36%**, inside the real artifact's own IQR, organ by organ. Neither reaches 56.1.

The cause is in `synth_export.py`, and it is mine: `SHAPES` carries each donor's own `tied` flag, and
**a tied model runs its head as the fp32 embedding** — 151,936 × 896 × 4 = 544.6 MB/token, 13.8 ms,
52% of the token. §3 of this brief asked for *"a ternary head (the runnable configuration E2
settled)"*, which is **untied and packed**. `SPEED_LEDGER` §12.2 says so in its own table without
naming it: `head 136,134,656 weights, 68.1 MB/token` is **0.5 bytes per weight**.

Measured now, on the real untied ternary-head artifacts:

| file | median tok/s | IQR | head ms/token |
|---|---|---|---|
| real `qwen25-05b_tqh.bin` | **57.790** | 0.190 | 3.619 |
| real `qwen25-05b_nlh.bin` | 57.580 | 0.340 | 3.626 |
| `SPEED_LEDGER` §12.2 | 56.1 (median of 56.41/55.75/56.14) | — | 3.673 |

The ledger's anchor reproduces at +3.0% on tok/s and −1.5% on the head organ. **The 27.7 G-weights/s
that prices this whole programme is a number about the untied ternary-head configuration**, and run 1
compared it against a fp32-head file. `--head {ternary,donor}` is added, defaulting to `ternary`;
`donor` is kept so run 1 stays replayable.

### 8.4 The gates for run 2, fixed before run 2 exists

**Gate V1′ — the instrument against ground truth, at a scale not yet used to choose it.** Run 1
replaced a proxy with something better by accident: a synthetic file can be compared to a **real
exported artifact of the same shape**, which is a stronger test than any synthetic-vs-synthetic
control. That comparison is already known at `S05` (0.36%), so `S05` cannot be the gate — it would be
chosen after seeing it pass. **The gate is at `S15`**, where `qwen25-15b_tqh.bin` exists on disk and
no synthetic 1.5B has ever been generated: synthetic `S15 --head ternary --codes mixed` must land
within the real artifact's IQR of `qwen25-15b_tqh.bin`, `--bench 300`, 6 threads, 3 reps each. **If
it does not, the probe stops and reports that instead**, and no shape above 1.5B is read.

**Gate V2′ — the known-positive, restated against what the ledger measured.** Synthetic
`S05 --head ternary` must reproduce the **real** `qwen25-05b_tqh.bin` (57.790, IQR 0.190) within that
IQR — not the ledger's 56.1, which was taken on another day on another build and is here quoted, not
re-used, as `feedback_perf_parallelization` requires of any number taken under unknown load.

**Gate V3 — unchanged, and it passed.** Byte-exact against `e1_bpb_through_engine.layout` (290
tensors, 724,954,676 bytes), against the real export's own file size, and against the C loader's
`layout OK: consumed exactly 724954676 bytes`. Three independent definitions of the format agree.

**`expf` is now a declared confound, not a discovery to be made later.** Synthetic weights give
activations whose SiLU cost is not guaranteed to be a trained model's. §8.3 measures that cost as
equal to 0.36% at `S05`; V1′ measures it again at `S15`. Above 1.5B there is no real artifact and the
`ffn` organ's absolute value carries that caveat explicitly wherever it is reported — the matvec
organs do not, since §8.2 measured them value-independent.

### 8.5 What does not change

No threshold, no arm, no prediction and no decision rule. §5's predictions stand as written and will
be reproduced against measurement including the wrong ones; §6's five outcomes and their constants
(`f ≤ 2.0 ms`, ±10% of 27.7, 24.9 G-w/s) are untouched. Run 1's `VOID` is reported in the probe as
run 1's label, not deleted.

**A law this bought:** *a planted control must be planted inside the range the instrument will
actually be used in.* An all-zero model is not a model any exporter can produce, and the gate that
used one measured `expf` instead of the kernel it was defending.

---

## 9. Run 2's gates, written before any arm above `S15` is generated

### 9.1 Gate V1′ — PASSES, and it was the one that could still fail

Pre-registered in §8.4 before any synthetic 1.5B existed. `--bench 300`, 6 threads, 3 reps each.

| file | bytes | median tok/s | IQR |
|---|---|---|---|
| **real** `qwen25-15b_tqh.bin` | 1,709,047,348 | **19.430** | 0.075 |
| **synthetic** `S15 --head ternary --codes mixed` | 1,709,047,348 | **19.430** | 0.065 |

Identical to three decimals, on files of identical byte count, with overlapping IQRs. Organ by organ:

| organ | real | synthetic | Δ ms/token |
|---|---|---|---|
| `qkv_proj` | 2.840 | 2.743 | −0.097 |
| `rope` | 0.034 | 0.034 | 0.000 |
| `attention` | 1.981 | 1.967 | −0.014 |
| `o_proj` | 1.956 | 1.957 | +0.001 |
| `ffn` | 38.276 | 38.355 | +0.079 |
| `head` | 6.357 | 6.381 | +0.024 |
| `norm+glue` | 0.137 | 0.138 | +0.001 |

Including `ffn`, the organ §8.4 declared a confound because of `expf`: **+0.079 ms on 38.3, 0.2%**.
Random weights and trained R3 weights cost the same in SiLU at this width.

### 9.2 Gate V2′ — the literal constant fails, and the ground truth fails it too

§8.4 fixed the reference at *"57.790, IQR 0.190"*, measured on the real `qwen25-05b_tqh.bin` in a
block twenty minutes earlier. Synthetic `S05 --head ternary` gives **56.180 (IQR 0.130)** — outside.

So the same real file was re-run, **interleaved A/B/A/B**, four pairs, one rep each, alternating:

| | median | IQR | reps |
|---|---|---|---|
| **real** `qwen25-05b_tqh.bin` | **56.095** | 0.188 | 56.42 / 55.76 / 56.08 / 56.11 |
| **synthetic** `S05_th` | **55.895** | 0.500 | 55.93 / 54.26 / 55.86 / 56.05 |

Δ = **0.200 tok/s = 0.357%**. Per-pair deltas 0.49 / 1.50 / 0.22 / 0.06 — the 1.50 is one pair in
which *both* files dipped together, which is what a load transient looks like and why the pairs are
adjacent in time.

**The real artifact moved 57.790 → 56.095 between the two blocks: 2.9%.** The reference constant is
therefore not reproducible by the file that produced it, and a gate whose ground truth cannot pass it
is measuring drift, not the instrument.

**This is the fourth time this ledger has been bitten by the same thing.** §11 withdrew a rate taken
under contention; §12 withdrew a rate whose timer bracketed the wrong work; §8.3 here found an anchor
taken on the wrong head; and now a gate constant that does not survive twenty minutes. **Law: a
speed gate may not contain a hard constant. It must name a file to be measured concurrently.**

**Operative form of V2′, and it is a change made after seeing data, stated as such:** synthetic must
match the real artifact of the same shape **measured interleaved in the same session**. On that form
V2′ **PASSES** at 0.357%, `S05`; V1′ passes at 0.000%, `S15`. The literal-constant failure is
reported in the probe next to the pass, not deleted.

The substance of V2 — *"a synthesizer that does not produce what the exporter produces is measuring
its own bugs"* — is carried by V1′, which was fixed before its file existed and could have failed.

### 9.3 Standing to read the arms

V1′ pass, V2′ pass (paired form), V3 pass at four independent definitions of the format
(`e1_bpb_through_engine.layout`; the real export's byte count — 793,629,748 for `S05` and
1,709,047,348 for `S15`, **exact**; the C loader's `layout OK`; and the engine producing a finite
`BENCH`). §6's decision rule may now be read against `S3`, `M7`, `Q8`, `T10`, none of which has been
generated at the time this section is committed.

**One number is already visible from the gates and is stated before the arms exist**, because it is
a gate measurement and not an arm: the non-weight fixed cost `f = rope + attention + norm+glue` is
**1.174 ms at `S05`** (0.017 + 1.089 + 0.068) — the ledger's 1.17 reproduced — and **2.152 ms at
`S15`** (0.034 + 1.981 + 0.137). It has already crossed §6's `2.0 ms` threshold at 1.5 B.
