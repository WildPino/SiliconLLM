# C50-A1 result (kernel roofline) and A2b rate (llama.cpp reference rate)

17 October 2026. Both ran on the clean box the owner provided, under the
[A1 protocol](C50_A1_KERNEL_ROOFLINE_PROTOCOL_20261016.md) with its addendum 2 and the
[A2b protocol](C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md) rate section.

## Run record

| Run | Pre-flight (45 one-second samples) | Processes | Outcome |
| --- | --- | --- | --- |
| A1 attempt 1 | mean 0.56%, max 4.49% | none | VOID under the old max rule (kept, `results/c50_a1_run_1.json`); the rule was replaced by addendum 2 before attempt 2 |
| A1 attempt 2 | mean 1.17%, max 3.81% | 5 admissible, 4 VOID replaced (foreign 4.75-5.93%); admissible foreign 3.02-3.93% | COMPLETE, `results/c50_a1_run_2.json` |
| A2b rate 1 | mean 0.36%, max 1.02% | 5 admissible of 5, foreign 1.60-2.13% | COMPLETE, `results/c50_a2b_rate_1.json` |

The verdict comes from `c50_a1_verdict.py` (`results/c50_a1_verdict_2.json`).

## A1: per-class streaming (G-A1a, bar 30 GB/s)

Median over the 5 admissible processes of the process medians, with min..max over processes:

| Shape | Class | GB/s | min..max | G-A1a |
| --- | --- | --- | --- | --- |
| S1 fused QKV 3072×2048 | Q4_K | 39.48 | 29.66..40.31 | PASS |
| | Q8_0 | 38.88 | 35.08..39.92 | PASS |
| | IQ4_XS | 40.71 | 33.42..41.11 | PASS |
| S2 attention output 2048×2048 | Q4_K | 40.40 | 38.02..40.57 | PASS |
| | Q8_0 | 40.22 | 38.43..40.88 | PASS |
| | IQ4_XS | 37.73 | 36.03..41.61 | PASS |
| S3 eight routed experts | Q4_K | 35.41 | 20.48..37.47 | PASS |
| | IQ4_XS | 39.17 | 25.25..39.48 | PASS |
| | **IQ3_S** | **12.45** | 11.08..15.66 | **FAIL** |
| S4 head 157184×2048 | Q6_K | 39.61 | 36.65..40.43 | PASS |
| | Q4_K | 39.56 | 37.03..39.86 | PASS |
| S5 dense layer-0 FFN | Q4_K | 38.04 | 28.22..38.72 | PASS |

**Layer and token:**
- **G-A1b, one P0 MoE layer:** 903.4 µs against a limit of 1,017.2 µs (28.11 MB charged).
  **PASS.**
- **T20, 20 layers without head:** 18.54 ms for 562.0 MB, and 15.84 ms of it is streaming at
  the measured class bandwidths. The **non-streaming residual is 2.70 ms**, against the
  desk model's assumed `T_FIXED` of 2 ms.
- **Where T20's time goes** (process 1, mean phases): routed experts 9.4 ms, attention over
  2,048 cached positions 4.9 ms, QKV 2.0 ms, output projection 1.3 ms, router matvec 1.0 ms.

**Verdict: `FAIL: S3_IQ3_S`.** It is the only failing cell; Q4_K is far above the 25 GB/s stop
line.

## Consequences (fixed in the A1 protocol before the run)

- **No re-plan of Phase A.** Q4_K passes on every shape.
- **A6 prices IQ3_S at 12.45 GB/s instead of 32.** It is done in [A6 addendum
  2](C50_A6_PRECISION_LADDER_PROTOCOL_20261017.md#addendum-2-17-october-2026-after-a1-before-any-arm-is-scored).
  - E3 and E3B drop to 28.3 and 28.8 desk tok/s, below P0's 34.4.
  - The fastest arm on paper becomes E4B (IQ4_XS experts plus a BF16 router): 38.2 desk tok/s.
- **The IQ3_S kernel is compute-bound.** It is not a bandwidth limit: the same expert shapes
  stream at 39 GB/s in IQ4_XS. Tuning it is component optimization, so ROADMAP rule 1 defers it
  until the complete P0 path has its per-organ cost.

## A2b rate: llama.cpp on the P0 artifact

`llama-bench` with the AVX2 build, `-t 6 -n 128 -fa off`, on the P0 Q4_K_M GGUF. The rate is the
median over 5 processes, with min..max:

| Context depth | tok/s | min..max |
| --- | --- | --- |
| 512 | 15.15 | 15.11..15.24 |
| 2048 | **14.39** | 14.31..14.49 |
| 4096 | 13.56 | 13.50..13.60 |

At context 2048, P0 with an f16 KV moves 867.2 MB per token. At 14.39 tok/s that is
**12.5 GB/s** effective, about a third of what the A1 kernels stream on this box.

## Registered predictions scored

| Prediction | Measured | Scored |
| --- | --- | --- |
| A1: Q4_K on the fused expert and attention shapes reaches 30-36 GB/s | S3 35.41 (in range); S1 39.48 and S2 40.40 (above) | **partly wrong:** the kernels are faster than predicted on attention shapes |
| A2b: llama.cpp decodes Ling-mini Q4_K_M at ≤ 30 tok/s at depth 2048 | 14.39 | right |

## What this cannot say

- **No rate for our engine.** These are kernel and synthetic-layer timings; the engine's own
  decode rate is A7.
- **Only one machine.** The bandwidths belong to this box (the reference floor), and
  `BW_PLAN = 32 GB/s` stays the planning number.
- **The wide S3 spreads** (Q4_K 20.48..37.47, IQ4_XS 25.25..39.48) are the process dispersion
  that A1b has to explain. They are not part of this gate.
