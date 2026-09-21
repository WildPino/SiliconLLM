# STRAT-01 GigaChat 3.1 `kqv_out-0` attribution diagnostic

**State:** FROZEN PROTOCOL; NOT YET EXECUTED
**Date frozen:** 21 September 2026
**Scope:** separate causal-attention reconstruction from special-layout V-B activation/dot semantics at the first remaining block-0 mismatch; offline only, zero donor executions

## Question

The changed-coordinate confirmation passes every old tensor gate through
`Vcur-0` and all compact-cache gates, then first fails at `kqv_out-0` in both
schedules (NRMSE `0.01199015083`).  That output is produced by two operations:

1. causal attention over `Qcur`, F16-cached `Kcur`, and latent `Vcur`;
2. per-head `blk.0.attn_v_b.weight` expansion from 512 to 192 using a
   special-layout Q4_K matrix.

The current C V-B path dequantizes Q4_K weights and multiplies F32 latent
activations.  Pinned GGML instead uses Q8_K activation intermediates for
Q4_K dots.  This is a strong hypothesis, not yet a result: attention arithmetic
and V-B share the observed boundary.

## Immutable inputs

Reuse the pinned-reference `prefill8` capture from the hash-bound successful
Rung-2A raw set.  The `cached7p1` full payloads have the same hashes, so exact
identity equality is itself a required control.

| Payload | Shape / bytes | SHA-256 |
|---|---:|---|
| `Qcur-0` | `[8,32,576]`, 589,824 | `4aca21f044acf71404ef0a7a000ed7b1bfe894efa82c5c1e49f7a5764cc6314b` |
| `Kcur-0` | `[8,576]`, 18,432 | `2860d9791b620d19788b8112e5366424be21669153eb1ec3255fd5a6c1b167f3` |
| `Vcur-0` | `[8,512]`, 16,384 | `8b775afa6fedd04f3c99bca0700f365cd13bbf235cef82e8f21809fbf252d2e9` |
| `kqv_out-0` target | `[8,6144]`, 196,608 | `bb73ca15e48df5de663c5fd90f9104a9a6e78652d5322aac12fba660d33177f3` |

Exact model identity remains SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
The exact V-B descriptor is Q4_K `[512,192,32]`, tensor offset `284986368`,
file offset `291089280`, byte span `1769472`.  Pinned oracle revision remains
llama.cpp `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.

## Diagnostic arms

Reconstruct the per-token/per-head 512-vector attention latent from immutable
Q/K/V traces using the frozen C equations: F16 cache roundtrip, causal mask,
the exact DeepSeek2 KQ scale, stable softmax and token ordering.  Feed that
same latent and exact V-B rows to:

- **D32:** current dequantized-Q4_K/F32 V-B control;
- **project Q8_K:** standalone project Q8_K quantizer and Q4_K×Q8_K dot;
- **pinned Q8_K:** independent GGML-linked oracle on the same reconstructed
  latent and stored V-B rows.

The project and pinned Q8_K arms must additionally agree on every Q8_K block
byte and meet NRMSE `2e-6`, normalized maximum `1e-5` against each other.
Compare all three outputs with the immutable `kqv_out-0` target using the same
metrics.  Preserve reconstructed latent bytes and hash even though no direct
reference-latent callback exists.

## Controls

- identity, shape, finiteness and equality of prefill/cached full payloads;
- exact project-versus-pinned Q8_K bytes and V-B outputs;
- one-byte Q8_K scale mutation must fail the target gate;
- head/row transpose must fail the target gate;
- a planted V-B packed-scale-layout error must fail;
- model-free operator suite, repaired-attention runner tests and 73,024-check
  kernel self-test remain passing;
- no donor or full graph process may execute.

## Adjudication

| Label | Frozen rule |
|---|---|
| `ATTRIBUTED_KQV_OUT_TO_VB_Q8K` | Controls valid; project and pinned Q8_K agree; both Q8_K arms pass target NRMSE `2e-6` / normalized max `1e-5`; D32 fails the old `0.002` NRMSE or `0.01` max gate. This jointly establishes that the frozen attention reconstruction is sufficient and missing V-B Q8_K semantics cause the observed boundary. |
| `ATTENTION_RECONSTRUCTION_OR_OTHER_VB_SEMANTICS_REMAIN` | Controls valid, but pinned Q8_K does not pass the target gate. |
| `PARTIAL_VB_Q8K_ATTRIBUTION` | Controls valid and pinned Q8_K improves materially, but does not satisfy the tight target gate or project/pinned agreement. |
| `VOID_KQV_OUT_DIAGNOSTIC` | Any identity, build, oracle, shape, finiteness, negative-control or completeness requirement fails. |

Run one non-VOID offline cell and stop; do not tune attention equations,
thresholds, payload selection, or V-B layout after observing results.  Preserve
voids under distinct directories.

## Eight-item control card

1. **Nearest prior cell:** [changed-coordinate confirmation](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION_RESULT_20260921.md), first failure `kqv_out-0`.
2. **Changed coordinate:** none in the production engine; offline decomposition of attention reconstruction and V-B dot semantics.
3. **Why prior is insufficient:** `kqv_out-0` combines two operations and has no captured pre-V-B latent.
4. **Identity/controls:** exact Q/K/V/target hashes, V-B span, pinned revision, oracle equality and three planted negatives.
5. **Separate gates:** causal attribution only; no attention repair, 2B, quality or rate.
6. **Claim labels:** prior integration PASS is measured; V-B attribution is proposed until adjudicated.
7. **Void/stop:** preserve voids; one non-VOID cell; no post-result tuning.
8. **Raw/canonical record:** planned raw directory `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_kqv_out_diagnostic_20260921/`; future result beside this protocol.

## Consequence boundary

Only `ATTRIBUTED_KQV_OUT_TO_VB_Q8K` authorizes a separately frozen production
V-B repair.  Any other non-VOID label requires a new source-derived diagnostic
of attention equations or V-B layout.  No `SPEED_LEDGER.md` entry is due.
