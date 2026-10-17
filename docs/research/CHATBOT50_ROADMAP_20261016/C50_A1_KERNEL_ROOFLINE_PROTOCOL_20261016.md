# C50-A1 protocol: kernel and layer roofline at Ling-mini-2.0 shapes

16 October 2026. Cell A1 of the [CHATBOT50 roadmap](ROADMAP.md) §5. Pushed
before any timed run (programme rule 5). Synthetic weight values, real shapes;
no model weights are read.

## Question

Do the standard GGML k-quant dot kernels, transcribed into project C code and
run at six physical workers, stream the Ling-mini-2.0 decode shapes at the DRAM
wall, and does one complete synthetic layer cost no more than its bytes at the
wall plus a small fixed overhead?

## Instrument

- Source: `benchmarks/chatbot50/c50_a1_roofline.c` with the kernel header
  `benchmarks/chatbot50/c50_kquants.h`. Kernels: Q4_K and Q6_K x Q8_K reuse the
  existing transcriptions (`benchmarks/phase60/strat01_q4k_q8k.h`,
  `strat01_q6k_q8k_avx2.h`); Q8_0 x Q8_0, IQ4_XS x Q8_K and IQ3_S x Q8_K are new
  transcriptions of llama.cpp `ggml/src/ggml-cpu/arch/x86/quants.c` at
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` (MIT). No GGML build or runtime
  dependency.
- Compiler: WinLibs GCC 15.2.0, `-O2 -mavx2 -mfma -mf16c`, no `-ffast-math`.
  The binary's SHA256 is printed with the results.
- Unit test before any timing: every AVX2 dot agrees with its scalar oracle
  (the generic GGML transcription) on random blocks with relative error
  <= 1e-5, or absolute error <= 1e-4 when |reference| < 1. A failing kernel is
  not timed.
- Threads: six workers, the calling thread being worker 0, each pinned to a
  distinct physical core (one logical processor per core, read from
  `GetLogicalProcessorInformationEx`; the mapping is printed and asserted).
  Phases are separated by a spin barrier (no OS waits). Rows of a matrix are
  split into six contiguous ranges of equal row count.
- Every run prints and asserts one `CONFIG` line (kernel set, shapes, pool
  bytes, threads, affinity, calls, seed, binary hash).

## Shapes (Ling-mini-2.0, from the pinned headers)

| ID | Organ | Shape (rows x cols) | Classes timed |
| --- | --- | --- | --- |
| S1 | fused QKV | 3072 x 2048 | Q4_K, Q8_0, IQ4_XS |
| S2 | attention output | 2048 x 2048 | Q4_K, Q8_0, IQ4_XS |
| S3 | eight routed experts of one layer: gate+up (8 x 1024 x 2048), then down (8 x 2048 x 512) | as listed | Q4_K, IQ4_XS, IQ3_S |
| S4 | head | 157184 x 2048 | Q6_K, Q4_K |
| S5 | dense layer-0 FFN: gate+up 10240 x 2048, down 2048 x 5120 | as listed | Q4_K |

**Cold weights.** Each timed call reads a weight copy not touched by the
previous calls: for S1, S2, S4, S5 the pool holds distinct copies totalling
>= 1 GiB (>= 32x the L3) and calls cycle through them; for S3 the pool holds
256 experts per layer for as many layers as needed to exceed 1 GiB, and every
call draws a fresh i.i.d. top-8 expert set (seeded) from a different layer copy.
Weight bytes contain random quants with finite FP16 scales.

**Metric.** Streamed GB/s = weight bytes of the call / wall time of the call
(QueryPerformanceCounter), activations excluded from the numerator. Per process
and (class, shape): one untimed warm-up pass over the pool, then 200 timed calls
(S4: 40); the process statistic is the median call.

## Synthetic layer and token

- **L1, one MoE layer at P0** (Q4_K matrices, F32 router, F16 KV, context
  2048): RMSNorm, Q8_K activation quantization, S1, QK-norm, partial RoPE (64 of
  128 dims), GQA attention of 16 query heads over 4 KV heads x 2048 cached
  positions, S2, residual, RMSNorm, router 256 x 2048 F32 matvec, sigmoid plus
  expert bias, 8-group/top-4-group/top-8 selection, normalization and scale
  2.5, the 8 selected experts plus the shared expert (gate/up, SwiGLU, Q8_K,
  down), weighted sum, residual. Charged bytes = all weight bytes read + KV bytes
  read.
- **T20, a 20-layer token without head**: layer 0 dense (S5 + attention) and
  19 MoE layers as L1, each layer from a distinct weight copy.

## Gates (pre-registered)

| Gate | Statistic | Pass |
| --- | --- | --- |
| G-A1a | per (class, shape): median over admissible processes of the process medians | >= 30 GB/s |
| G-A1b | L1: median over processes of the process-median layer time | <= charged bytes / 30 GB/s + 80 us |

Reported without a gate: T20 time, its charged bytes, and the non-streaming
residual `T20 - charged bytes / (byte-weighted G-A1a bandwidth of its classes)`
as the first measurement of the desk model's `T_FIXED` (2 ms assumed).

**Consequences fixed now.** G-A1a failing for Q4_K on S1, S2 or S3 allows one
tuning iteration (recorded as its own run); if Q4_K is still below 25 GB/s the
Phase-A policies are re-planned before any engine port (roadmap stop rule).
A failing IQ3_S, IQ4_XS or Q8_0 class is not a stop: A6 prices that class at
its measured bandwidth instead of 32 GB/s. A failing G-A1b sends the layer to
one diagnostic cell under rule 3 (at most three cells or three days).

## Admissibility and repetitions

- Five fresh processes, interleaved: each process runs all (class, shape) cells,
  L1 and T20, with the cell order rotated by process index.
- Foreign occupancy (system CPU time minus this process's own, the E44/E52
  meter `benchmarks/donor_adaptation/engine/e44_interval.py`) is measured around
  each process; a process at or above **4.39%** is VOID, kept in the log, and
  replaced, at most five replacements. The cell is VOID if fewer than five
  admissible processes exist.
- A 45-second idle pre-flight with every sample below 4.39% is required before
  the first process.
- Dispersion: min..max of process medians and max/min ratio are printed beside
  every median. The A1b stability gate is a separate cell.

## Registered prediction (scored, not a gate)

ROADMAP §5: Q4_K on the fused expert and attention shapes reaches 30-36 GB/s at
six workers.

## What this cell cannot say

Synthetic values give the decode arithmetic and memory traffic of real shapes,
not model quality; real-weight value statistics do not change the byte stream.
Rates of a matvec are not rates of a model.

## Addendum 1 (pre-run, 16 October 2026; pushed before any admissible timed run)

Changes made while building the instrument, before the first admissible run.
Every smoke output is kept in
[`c50_a1_smoke_20261016.txt`](../../../benchmarks/chatbot50/results/c50_a1_smoke_20261016.txt)
(contended box, inadmissible, not evidence).

1. **Compiler.** clang 21.1.8 (llvm-mingw 20251216) instead of GCC 15.2.0: the
   reused `strat01_q4k_q8k.h` refuses to compile under GCC (`#error` requiring
   Clang target attributes for its reference-generic arm). Flags unchanged:
   `-O2 -mavx2 -mfma -mf16c`, no `-ffast-math`.
2. **Unit-test normalisation.** The first self-test failed 4 of 1,000 trials
   (Q6_K 3, IQ4_XS 1) at 1.6-2.0e-5 of |result|: results near zero formed from
   much larger terms (cancellation), not kernel faults. The error is now
   normalised by the sum over quant blocks of |block dot| (each block scored by
   the generic oracle alone), bound 1e-5; measured worst 7.2e-7. A planted
   control was added: one high bit flipped in one quant byte of the oracle's
   copy must exceed the same bound in >= 190 of 200 trials per class (measured
   194-200; a single flip can be invisible when multiplied by a zero quant,
   scale or activation).
3. **Instrument repairs after a contended smoke.** The first synthetic layer
   computed attention with a scalar `expf` per position and re-read each K/V row
   for every query head of its GQA group, recomputed RoPE angles with
   `powf/cosf/sinf` per call, and used a scalar SwiGLU (smoke: attention
   0.75 ms, RoPE 0.13 ms of a 1.8 ms layer). These are naive instrument choices,
   not properties of the decode: attention now reads each K and V row once per
   group and uses the vector exp of GGML (`ggml_v_expf`), RoPE uses a per-position
   cos/sin table, SwiGLU uses `ggml_v_silu`. No gate or threshold changed.
4. **Counts left open by the protocol, fixed now:** L1 200 timed calls after one
   warm-up pass over the 19 MoE layers; T20 40 timed tokens after 2 warm-up
   tokens; process i starts at cell (3i mod 14). One process takes ~12 s.
5. **Per-organ breakdown.** L1 and T20 also print mean wall time per organ
   (warm-up calls included in sum and count). The T20 residual prices the F32
   router and F16 KV at 32 GB/s (no measured cell for them).
6. **Runner and verdict.** `benchmarks/chatbot50/c50_a1_run.py` (pre-flight,
   five admissible processes, VOID handling) writes numbers only;
   `benchmarks/chatbot50/c50_a1_verdict.py` applies the gates above.

## Addendum 2 (17 October 2026, after a VOID pre-flight, before any timed process)

**What happened.** Attempt 1 (`results/c50_a1_run_1.json`, kept as VOID) stopped at the
pre-flight on a box the owner had emptied:
- 45 one-second samples, mean 0.56%, median 0.38%;
- one sample at 4.49%, one at 2.42%, three more between 1.5% and 1.8%;
- the self-test passed, and no timed process ran, so no rate was produced or read.

**Why the rule was wrong.**
- "Every sample below 4.39%" compares the maximum of 45 one-second samples with a bar that
  E52 calibrated on the mean foreign occupancy over a whole timed window (0.262% of speed
  per point).
- The per-process gate of this protocol uses that same mean.
- A one-second burst of half a logical CPU is a different estimand. On an idle Windows box
  it is close to certain within 45 s (system services), so the rule could not be met. It was
  written to be severe, and that severity did not make it valid.

**New rule, for this cell and for the A2b rate, which inherits these clean-box rules:**
- The pre-flight is admissible iff the **mean** of its 45 samples is below 4.39%.
- The max and the number of samples at or above the bar are recorded, not gated.
- The per-process VOID rule (each process's own mean foreign occupancy below 4.39%, at most
  five replacements) is unchanged; it is what protects the timings.
- The runners (`c50_a1_run.py`, `c50_a2b_rate.py`) print the rule and both numbers.
- The next attempt writes `c50_a1_run_2.json`; attempt 1 is never relabelled.
