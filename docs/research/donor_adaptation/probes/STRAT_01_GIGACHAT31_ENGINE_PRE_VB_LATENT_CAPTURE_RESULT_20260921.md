# STRAT-01 GigaChat 3.1 pre-V-B latent capture result

**Verdict:** `ATTRIBUTED_RESIDUAL_TO_ATTENTION_RECONSTRUCTION`
**Date:** 21 September 2026
**Scope:** one pinned block-0 prefill trace and offline stage attribution; no production-engine, quality, RAM, or speed claim

## Result

The direct pinned `kqv-0` and `kqv_mla-0` callbacks remove the ambiguity left
by the reconstructed-latent diagnostic.  Exact Q4_K×Q8_K V-B semantics and the
final graph layout both pass on the true intermediate.  Only the prior causal
attention reconstruction fails.

| Stage | NRMSE | Normalized maximum | Frozen gate | Result |
|---|---:|---:|---:|---|
| reconstructed latent vs captured `kqv-0` | `2.159975473e-4` | `5.265653763e-4` | `2e-6` / `1e-5` | FAIL |
| project vs pinned V-B on true `kqv` | `1.250568853e-7` | `7.685697992e-8` | `2e-6` / `1e-5` | PASS |
| project V-B vs captured `kqv_mla` | `1.250568853e-7` | `7.685697992e-8` | `2e-6` / `1e-5` | PASS |
| pinned V-B vs captured `kqv_mla` | `0` | `0` | `2e-6` / `1e-5` | PASS |
| mapped `kqv_mla` vs `kqv_out` | `0` | `0` | `2e-6` / `1e-5` | PASS |

The pinned V-B output is bit-identical to both captured `kqv_mla-0` and final
`kqv_out-0`.  The project scalar operator differs only at `1.25e-7` NRMSE,
well inside the tight gate, and project/pinned Q8_K bytes are exact.  Therefore
the previous `0.00125348` residual came from feeding an inexact reconstructed
attention latent to an otherwise correct V-B operator.  V-B Q8_K is now
validated on its true input; the remaining block-0 defect is inside causal
attention before V-B.

## Controls and provenance

- Accepted commit: `82e3efab2cb5d3125cb992568cb84d508b9916a7`.
- Exact GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Pinned llama.cpp: `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`;
  `llama-graph.cpp` SHA-256
  `000c88afa5ebd4f1d20821054ef9dc4d10a0c9cea4f440d179eaea8723ad8dee`.
- Captured callback hashes: `kqv-0`
  `3922f34159f499dd26788acb7d9600d72fded004425392946bc09b8098fd88df`;
  `kqv_mla-0`
  `73c80fcb30fadb89c98bbde556609526543edddd5baf047d62d7f8cb7198dc36`;
  `kqv_out-0`
  `bb73ca15e48df5de663c5fd90f9104a9a6e78652d5322aac12fba660d33177f3`.
- Observed shapes exactly match the source-derived preregistration:
  `[512,8,32]`, `[192,8,32]`, and `[6144,8]`; ops are respectively
  `MUL_MAT`, `MUL_MAT`, and `CONT`.
- Identity, clean pinned source, callback completeness, finiteness, model-free
  tests, both self-tests and exact Q8_K bytes pass.
- Scale mutation, head transpose and packed-scale-layout negatives fire at
  NRMSE `0.426077`, `0.982027`, and `0.332394`.
- Exactly one pinned-reference prefill donor execution ran; the C donor engine
  did not execute.

Raw adjudication:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_pre_vb_latent_capture_20260921/adjudication.json`,
SHA-256 `120216c2b19e289a8584977a1979c240f8f3c242b9a0b36a34a551b437755b33`.

## Consequence and no-duplication boundary

Do not repeat the pre/post-V-B capture or any earlier `kqv_out` reconstruction
cell.  The true-intermediate oracle authorizes the exact Q4_K×Q8_K V-B
operator as the eventual production implementation, but does not by itself
close block-0 integration: causal attention remains out of the tight gate.
The next coordinate must capture pinned `kq-0` and `kq_soft_max-0` and compare
QK dot accumulation, scaled masked softmax, and value reduction separately.
Do not advance to Rung 2B, quality or speed; `SPEED_LEDGER.md` is unchanged.
