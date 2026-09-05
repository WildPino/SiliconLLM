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
