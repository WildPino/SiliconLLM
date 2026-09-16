# H5 — Frozen-checkpoint cross-composition diagnostic

**Status:** preregistered before the H5 runner was executed; CPU-only, no training or T4.
**Terminal disposition:** `POSTHOC_TRANSFER_SIGNAL=false`, recorded in
[the H5 result](../probes/H5_CROSS_COMPOSITION_RESULT.md). The rules below remain the
pre-measurement specification, not retrofitted thresholds.
**Question:** do the *independently trained* H4 rank-48 q/o and H2I v4 one-byte eight-layer
FFN carve transfer positively when installed in the same pretrained Qwen2.5-1.5B model?
This is deliberately **not** a joint-training test, engine export, or H2I rerun. A negative
answer cannot close the fresh joint-training route; it only closes an inexpensive post-hoc
checkpoint-composition shortcut. A positive answer still cannot promote H2I's failed
score+rank gate, because the composed object is a new treatment.

## Frozen inputs and arms

- Donor `Qwen/Qwen2.5-1.5B` at revision
  `8faed761d45a263340a0528343f099c05c9a4323`, loaded CPU fp32.
- H4 terminal `h4_trained.npz`, SHA-256
  `11a483419e6b8795164e4651547f6b3765fc325adaa3283153d9cd5276d4cefb`,
  `TIME_CAP` step 1148/1145 applied. It replaces q/o on all 28 layers with ternary rank 48.
  Canonical terminal evaluation SHA-256:
  `23c3792f4a04efedf90ec0617654a529f0e2a115419f55cd71ad0d55d48e48f2`.
- H2I v4 terminal `h2i_trained_s1.npz`, SHA-256
  `ae7e1a7c01e0ce1982dc6c8457a9508fc0e3b57d5c7c4b9d7ef0b887b7dcd67a`,
  and its sidecar SHA-256
  `24585587565d613ff6d8d2ee305055d3c51a5ad291ad6fce0d53bc8ed129d865`.
  Canonical H2I evaluation SHA-256:
  `2180e1c5d01d903368b1b9119b99049ac5e0d48abbac8bbdbc460dca2151e805`.
  Install only trained R8 FFN masters/router/labels on layers
  `[3,6,9,12,15,18,21,24]`, hard top-16 of 256 groups; leave every other FFN donor fp32.
- Frozen `heldout` 24×512, seed 1234, token-ID SHA-256
  `a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`;
  same E6 five prompt trajectories and 160 teacher-forced positions used by H4.
- Arm A: **H4 terminal only**. This is a planted reproduction control, not a new discovery.
- Arm B: **H4 terminal + H2I trained eight-layer FFN**, with no gradient updates.

The evaluator must assert frozen result identities and input hashes before model loading,
and refuse to overwrite an existing result. Arm A must reproduce the canonical H4 terminal
BPB `1.15373752359353` within `1e-5`, free `6/160`, teacher-forced `66/160`, and mean
rank `108.725` exactly to recorded precision before Arm B is interpreted. Failure makes
the entire H5 output **VOID_APPARATUS**; no Arm B claim is allowed.

## Primary observables and decision rule

Both arms report BPB, free exact/160, teacher-forced exact/160, mean rank, rank≤5/160,
decoded free generations, and elapsed time. The reference is the dense E6 continuation,
not H2I's H0-generated reference; the H2I rank result `95/160` is therefore *not* a
direct comparator. There is no best-checkpoint or best-prompt selection.

The **POSTHOC_TRANSFER_SIGNAL** fires only if, against reproduced Arm A, all three hold:

1. Arm B BPB is at least `0.05` lower;
2. Arm B teacher-forced exact is at least `10` higher;
3. Arm B free exact is at least `15/160` (above the historical `AT-FLOOR` band).

These are a deliberately demanding *priority* gate, not a claim that BPB or exact-match
is a universal usability measure. Report each metric and each clause even when the gate
fails. A mixed outcome is `MIXED_POSTHOC`, not a rescued PASS. The prediction registered
before evaluation is that the signal will **not** fire: the FFN adapter was trained under
H0's much less aggressive q/o representation, so feature-distribution shift is plausible.
The direction of BPB change itself is not presumed. If the signal fails, do not spend GPU
to train this **frozen composition** as-is; any joint experiment needs a separate brief,
its own step-zero controls and a training protocol. If it fires, a new brief may prioritize
that joint geometry, but no launch or export is authorized automatically.

## Evidence and no-duplication boundaries

Canonical antecedents: [H4 terminal and H4D](../probes/H4_AGGRESSIVE_RANK_STEP_ZERO.md)
§§10–11; [H2I terminal](../INDEX.md) and
[H2I export-gap audit](../audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md) §5. E63 is a
synthetic speed result and never participates in this quality gate. H5 never repeats H4
Stage A training, H2I Phase B training, H2I's canonical evaluation, or any engine timing.
Even a positive H5 result measures a 1.5B mixed-format Python model, not a 10B model or
`engine.c` tok/s. Only the write-once H5 JSON and a subsequent result addendum may state
what was observed.
