# STRAT-01 Q6 reference-generic production integration — VOID 1

**Date:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-Q6K-Q8K-REFERENCE-GENERIC-PRODUCTION-INTEGRATION`

**Status:** `VOID_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION`

## What executed

The one authorized production invocation ran from commit
`54b4c381a214eb7a14de8e0cb5e1e40953ccd8f5` and completed both `prefill8`
and `cached7p1`. The preserved raw directory is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_reference_generic_production_integration_20260924/`.
It contains both candidate manifests, all declared payloads and caches, helper
counts, the C report, command logs, and exactly two
`STRAT01_RUNG2C_GRAPH_COMPLETE` markers. No reference graph ran.

## Why the record is VOID

After production completed, the inherited `validate_c` compared the entire
reported `CONFIG` to its older pre-Q6 string and raised
`C reference/config/compiler mismatch`. The observed report contains exactly
the preregistered extension
`q6kq8k=reference-generic-noavx-noavx2-nofma-noinline`; the remainder is the
historical Rung-2C configuration. While handling that validation error, the
runner's exception tuple evaluated nonexistent `q6base.ParityError`, causing
an `AttributeError` before `adjudication.json` could be written.

This is an apparatus failure after the sole producer completed. It does not
say whether the Q6 production integration passes or fails scientifically.

## Recovery disposition

Do not rerun the producer. The protocol's pre-existing VOID rule permits one
offline adjudication because the immutable outputs suffice. The recovery
addendum now freezes exact source/output/log/configuration bindings, the same
scientific gates, and zero new producer or graph executions. Any failed binding
keeps the cell VOID.
