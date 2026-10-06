# PQT-010: shared embedding/readout intervention on frozen broad payloads

Prospective, 6 October 2026. Source/data/decision freeze before any new model
outputs. Consumed-development diagnosis, **no promotion or fresh holdout claim**.
Question: with all other parameters fixed, how does restoring only the original
shared embedding/readout change prediction and own-prefix generation of the
audited PQT-009-R1 broad ONE and STAGED models? Does a counted I8 shared parameter
retain any measured restoration benefit within the existing storage limit?

## Fixed inputs and six interventions

Both original broad schedules, not a post-hoc winner. Six arms: T_ONE/T_STAGED
are unchanged byte-exact B_ONE/B_STAGED parent archives; F_ONE/F_STAGED replace
only `model.embed_tokens.weight` / tied `lm_head.weight` by its original F32
parameter; I8_ONE/I8_STAGED replace the same shared parameter with source-derived
symmetric per-row signed I8. No new fitting, no latent student checkpoint,
no source-vector change, no optimizer steps. All other168 matrices and every
original F32 vector member retain byte-exact original parent identities.

For I8, independently frozen rule: for each original F32 row, scale is
max(abs(row))/127 in F32, zero-row scale0; nearest-even F32 rounding of row/scale,
clamped[-127,127], int8 codes stored as exact one-byte signed NPY, one F32 scale
per row. Zero rows decode exactly zero. No calibration, learned scale or loss
filter. Reject nonfinite/reserved -128 or changed geometry. Rowwise rule matches
the previously tested I8 baseline definition; no empirical I8 rescue yet.

Shared parameter151936x896 =136,134,656 coefficients, stored once despite two
aliases. T raw shared code/scale bytes42,542,080. Full raw T154,649,088 bytes.
F raw shared544,538,624; complete raw F656,645,632 bytes, necessarily above
35% of988,065,536 FP16 bytes: **excess-storage diagnostic only**. I8 raw shared
136,742,400; complete raw I8248,849,408 bytes before member/header/archive
overhead, ~25.2% FP16; **mixed representation**, not all-matrix ternary.
Full self-contained archives count every member/header/descriptor/ZIP byte.
No external undeclared float cache or duplicate tied parameter deployed.

Reuse exact private parent dataset `sirwildpino/pqt-009-frozen-parent-20261006`
v1, binary `pqt009_frozen_outputs.bin`, size1525747323, SHA
`a0e27a1c575ab714573ad904ffdb53502da27d2b02d66c1b5918daa909da7626`.
Verify unique mount, safe extraction, all75 parent identities/22 actual parent
sources against the existing immutable R1 manifest; keep full parent provenance.
R1 audit/retention/Git verification qualifies these fitting objects, not their
preservation. Freeze its exact certificate identities, both broad archive/member
identities, original model/tokenizer/runtime, and selected consumed token arrays
in PQT_010_CONSUMED_BINDINGS.json. Binding preparation is small-array metadata
only, no new corpus selection or original function/model/GPU execution.

## Consumed roles and unchanged measured criteria

Reuse exactly R1 Wiki/news prediction8x129 and prompts8x32, labels1024 per
domain, full identity/history retained. News is provider train partition used
only for evaluation; previous test-partition consumed history and R1 exclusion
ledgers remain visible. No new corpus access/selection is needed. Outcomes were
observed in R1, so all six arms are consumed-development ablations. Freeze
token values/file/hash identities and R1 source/broad point/state/trace identities
before new outputs. Baselines and original-source states/choices/points must
reproduce R1 exactly. No candidate selection or threshold relaxation.

Original model revision060db6499f32faf8b98477b0a26969ef7d8b9987, original
checkpoint/tokenizer byte identities, deterministic seed20261005, pinned
Py3.13.15/torch2.11.0+cu128 runtime, F32 SDPA, TF32false unchanged. Entire model
own-head inference and changed-state own-prefix greedy generation, at most32
new tokens, original EOS behavior. Save every hidden prediction/pre-choice
state, choice, trace, point and per-window metric. Reject all nonfinite values.

Strict absolute gates remain descriptive: separately both domains, argmax>=99%,
KL<=.01, NLLdelta<=.01, generation-position agreement>=95% (unmatched tails
mismatch), exact traces>=7/8, archive<=35% FP16 and charged parent fit<=7200s.
**Even a descriptive gate pass is not promotion on consumed contexts.** No
statistical/generalization claim from8 windows/8 prompts at one seed.

For each schedule and each F/I8 intervention against its unchanged T baseline,
preregister a restoration signal: >=10% lower KL, no worse NLL, no worse argmax,
no worse generation-position agreement, all clauses separately in both domains.
Report per-domain and joint signals; capacity is reported independently, so a
large F diagnostic cannot masquerade as an economical method. Report I8 vs F
descriptively without selecting an arm or creating an additional success gate.
Count every raw/deployed byte and unchanged/changed parameter/member.

A positive intervention effect applies to this fixed payload/context pair.
It does not establish a unique prior failure cause. A negative restoration may
reflect coadaptation, not a representation lower bound. No original expert-
function rescue, native speed, broad capability or feasibility closure inferred.

## Qualification, independent replay and stops

Synthetic CPU-only controls before remote source access: six archive contexts,
mixed/F32/ternary byte/member coverage, original tied parameter counted once,
all other members byte exact, independent decoding, zero rows, tie rounding,
reserved/nonfinite/duplicate/unknown-member stops, wrong source/code/scale hashes,
consumed roles/history and baseline-identity guards. One thread, no timing or
local original model/corpus/GPU work. Preserve first failures and number repairs.

Separate audit process imports no intervention writer, evaluation driver or
fitting/optimization module. Independently verify actual committed source,
parent provenance and qualification certificates, original parameter/runtime/
tokenizer identities, every matrix/vector/member/tie, exact untouched168
matrices, F original coefficient bytes, I8 source-derived codes/scales using an
independently written rule, raw/full archive capacity. Reconstruct all states,
own-prefix choices, all F32-derived points and all-position F64 readout
diagnostics<=1e-5; replay source/T outputs against exact R1 identities. Recompute
all descriptive absolute/relative decisions. No optimizer trajectory replay.

Capture evaluation and audit per-device CUDA allocation peaks/RSS separately;
report archive preparation/setup/evaluation/audit, original failed attempt and
prior fitting costs without double-counting nested phases. Parent memory peaks
absent from historical records remain unknown; no retroactive fabricated values.

Install<=900s, evaluation/preparation<=1200s, audit<=1800s, server<=4500s.
Fresh private acct3 ref `sirwildpino/pqt-010-shared-parameter-ablation-20261006-001`.
Verify input readiness/privacy, fresh owner/quota/all-owned-terminal admission,
freeze actual source/bundle in Git, single dispatch, delegate same-handle
monitoring and wait. Terminal fetch once, full hash/source retention, independent
replay and standard-library adjudication before scientific interpretation.
OOM/hash/nonfinite/resource/replay failure invalidates this run, preserve first
evidence, never silently restart. Owner CPU timing window is independently
pending; do not infer permission from idle local hardware. Goal stays active.
