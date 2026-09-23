# STRAT-01 exact F16 reduction propagation — apparatus result

**Date:** 23 September 2026
**Cell:** `STRAT-01-ENGINE-F16-VECTOR-REDUCTION-PROPAGATION`
**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

## Decision

The apparatus is qualified for exactly one committed accepted-artifact C
producer invocation. The qualification compiled the production engine, ran
141 STRAT-01 Python tests and 17 inherited C self-tests, and passed the
mandatory model-free Stage-A identity gate. It executed zero donor graphs and
zero reference graphs.

This closes apparatus construction only. It does not establish downstream
parity, language-model quality, RAM, or rate.

## Pre-donor mechanism correction

The first Stage-A transcription exposed an error in the protocol's original
mechanistic description before any accepted-model graph ran. The historical
helper's `ggml-cpu/vec.cpp` was compiled with `-DGGML_CPU_GENERIC`, without
AVX flags. Its pinned `ggml_vec_dot_f16` therefore multiplies F16-converted
operands in F32, accumulates sequentially in `ggml_float` (F64), and converts
once to F32 at the output. It is not the AVX/FMA branch initially inferred
from the source tree.

The protocol now carries a dated pre-donor addendum. No immutable input,
required output hash, numerical threshold, graph budget, or verdict rule was
changed. An independent F16-input/F64-accumulation reconstruction reproduced
both frozen pinned outputs exactly; the discarded AVX/FMA transcription
reproduced neither. The production mode is explicitly named
`pinned-generic-f64`.

## Stage-A exact replay

| payload | bytes | observed and required SHA-256 |
|---|---:|---|
| pinned QK reduction | 8,192 | `121c689214a7bcf2e709b8de896df14aabcb8fb00580bab297231b8887a3b009` |
| scalar QK control | 8,192 | `5b4caf1ebab46a5f6340dc695a72f8f08145f5f8817081e7cec10bceedaaae1d` |
| pinned value reduction | 524,288 | `541183b4eb5992cfcfc711fe3c9f179ab41685106ba9cfe2aa7099f45f821312` |
| scalar value control | 524,288 | `b77004cfaeb0c96588df64d79e9df091e7a3b75273475f610105d106f9d56bbb` |
| F16 conversion stream | 435,240 | `7cdec0744fb5c9dcc6225dd70a0c952283128d4f8668d9861131b868379defbc` |

The pinned streams are distinct from the scalar controls. Both legal operand
mutations reject the frozen tight gate:

| mutation | NRMSE | normalized maximum | gate |
|---|---:|---:|---|
| QK operand | `8.40148e-6` | `4.96435e-5` | reject at `2e-6` / `1e-5` |
| probability operand | `0.00450773` | `0.0357667` | reject at `2e-6` / `1e-5` |

All three immutable Stage-A inputs, all five historical outputs, all three
predecessor adjudications, and both schedules' old-candidate and reference
manifests passed their frozen size and SHA-256 checks. Source binding and the
zero-graph/non-speed controls also passed.

## Evidence binding

- Raw directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_f16_vector_propagation_apparatus_20260923/`
- Raw adjudication SHA-256:
  `616644a8a3981d42a68b714674a0b2c960329fb1b6f27e269b0de1b949fa9039`
- Qualified binary SHA-256:
  `7d435e97dd6e2090c33b5e7baa2bd562cdc7a9b81e6dee8675c480802ef4ad98`
- Python suite: 141 tests, 111.590 seconds, return code 0.
- Complete apparatus wall time: 112.582 seconds.
- C self-tests: 17, all return code 0.
- Donor producer invocations/graphs: `0/0`.
- Reference producer invocations/graphs: `0/0`.

The raw directory is evidence and remains outside Git. The committed runner,
headers, protocol, tests, and this result document are the reproducible
apparatus definition.

## Authorized next action and stop rule

After these sources are committed cleanly, run the scientific mode once. It
may execute one accepted C producer, yielding exactly the `prefill8` and
`cached7p1` completion markers, and must execute no reference producer. The
helper counts must be exactly 4,608 QK and 524,288 value reductions, and the
attention hash must change in both schedules relative to the frozen scalar
candidate.

Do not rerun Stage A, the historical F16 diagnostic, softmax, Q4, SwiGLU, Q6,
or a reference graph. A non-VOID PASS or FAIL closes this exact reduction
cell. Neither outcome is a quality or speed result.
