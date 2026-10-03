# Next compact Switch representation: all-bank W8A8 cost preflight

**PARTIALLY IMPLEMENTED; COMPLETE NATIVE CONTRACT RUNNING.**335 export and
337 recovery interruptions retained;338 ALL coefficient and target bytes PASS,
14.818GB with all real banks.336 code/protocol frozen00a5ee9, execution pending.
No original-donor quality or cost qualification.334 validates complete target arithmetic
against an independent reference on original14.664B weights, not pretrained
quality or speed. Original329/330/332 original-backend failures remain.
Use334 loader/encoder/relative attention/self-cross cache/capacity/head
architecture; compact candidate changes parameter representation and matrix
arithmetic explicitly. No extra bank copies or small student.

## Decision and reuse

Uncertainty: can a direct compact complete source path have enough actual CPU
margin to justify untouched quality/adaptation work? Reuse327 ALL actual tensor/
alias/12x256 parameter-tuple hashes and334 qualified architecture. Cost must
include actual full banks, selected reads, router scan/dispatch/decode, core,
full vocabulary head, prefill and growing attention/cache. A pass only licenses
quality work, never establishes accepted generation rate. Final same artifact
must retain original-donor quality and >=50 accepted batch1 tokens/s.

## Implemented target recipe, qualification pending

- Keep ALL original256 experts in each of12 banks; all original core layers,
  encoder/decoder/capacity/top1 gating and vocabulary retained. No fitting.
- Symmetric per-row signed I8 for every large matrix except source router;
  row scale F32(absmax/127), nearest-even clipped[-127,127]. Zero row: scale1,
  codes0. Record representability/finite checks, exact rounding rule and hashes.
  Embedding lookup/norm/relative bias/router metadata stay original F32.
  Shared source embedding/head can require separate lookup F32 and head I8
  views; count the extra bytes and shared original provenance explicitly.
- Activation vector quantized once per matrix call to symmetric I8 with one
  F32 scale, same declared rule. This is W8A8, not just weight-only compression.
  Its loss is unknown. I32 exact dot safe for sourcecols<=4096 and[-127,127]
  operands: worst bound66,064,384. Scaled output prescribed F64 scale product/
  integer scaling then rounded F32; residual/norm/attention contract334 retained.
- Native AVX2 sign-extend I8 to I16, VPMADDWD then I32 sum; avoid saturating
  intermediate I16 multiply-add. Threads1/6 declared, no ideal-bandwidth claims.
  Independent reference uses serialized target I8/scales, integer matrix sums
  and explicit scaling, not a second independently requantized checkpoint.
- Full target export streams bounded source shards/rows into actual disk
  payloads, all source/tensor/target hashes and alias metadata. Count distinct
  target WI/WO code-and-scale tuples in every bank; quantization cannot silently
  substitute identical tuples for genuinely distinct source functions. Actual complete payload14,818,015,744B
  including F32 embedding/controls; preserve partials. Final
  converter/native format/gates/download-free resource budget must be concrete
  and frozen before target observations. Existing source never overwritten.

## Qualification order and limits

First adversarial extreme/cancellation/zero-row exact integer and scale oracles,
Tiny full semantic faults/caps/cache, then full engineering consumed controls
versus matched integer reference. Preserve every failure before repair. No
comparison to original donor can be relabeled as exact under quantization.
Then real all-bank CPU measurements on stated source/output lengths, warmup/
repetitions/order/threads and actual routing/byte/cost counters frozen before
observations. Forced decoder assays are costs only, not accepted quality rate.
Source-routing controls should reflect target trajectories, with choice changes
versus original donor logged as diagnostics, not silently forced back.

If complete measured CPU cost lacks margin for goal50, close unchanged W8A8
before extensive quality/training; choose a new variable based on measured
bottleneck. If cost and implementation pass, define NEW source exclusions,
reconstruction/prediction/generation/tasks and limits BEFORE observation against
ORIGINAL unmodified official donor. Quantization may fail quality; no guarantee.

This is a compact source baseline enabling investigation of cheaper LUT book/
conditional representations. It is not a LUT scaling result by itself. Dynamic n loader is implemented336 but only real256/Tiny2 attempted.
RAM-scale useful n, real base128 comparison and useful larger capacity require
separate actual source functions, routing and whole-quality measurements; source
base256 availability alone does not prove10x expert scaling or~100B transfer.
