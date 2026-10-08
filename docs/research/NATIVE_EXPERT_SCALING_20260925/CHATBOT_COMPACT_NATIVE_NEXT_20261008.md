# Compact chatbot: connection from fitted weights to engine.c

8 October2026. Prepared implementation, UNEXECUTED. The one fixed whole-output
fit is still LIVE in session77989; do not overlap numerical/export/build/native
work with it. No native model artifact has been produced or admitted here.

## Decision and evidence used

The pipeline needs one deployable artifact containing the trained conditional
functions, preserved source attention/norms/tied vocabulary head, source tokenizer
and canonical chat/EOS contract. This work connects those components after the
fixed1280-update checkpoint passes all8 whole-output transfer gates AND its
independent FIRST audit. A failure closes that recipe; it cannot be bypassed by
exporting a favorable intermediate checkpoint or relaxing fresh quality gates.

Existing byte-qualified source/core census,24-block installation, cached200-case
teacher cohort and interaction adapter are reused. Old local geometry/value/
J/H/native observations are retained; new representation requires new evidence.

## Available code and precise transformation

`benchmarks/native_expert_scaling/chatbot_compact_export.py` is an export API.
It requires paired successful fit/audit receipts and fixed-final hashes. It
preserves218 original BF16 core tensors and NEW F32 routing coefficients exactly;
it rounds165150720 trained F32 G/U/D coefficients to BF16. This rounding is an
approximation requiring a whole encoded comparison. Codec coefficient energy
and maximum errors are diagnostic, not a bound on chatbot behavior.

The single actual Safetensors archive has435 tensors,693597568 payload bytes:
360492800 core,330301440 G/U/D,2803200 routing,128 RoPE frequencies. C=1 child
buffers are omitted only after their exact identity to parent buffers is checked.
The generated catalog binds the actual complete blob SHA/length/offsets/types/
shapes. No generated catalog/blob exists until export actually executes.

`benchmarks/native_expert_scaling/chatbot_compact_entry.c` is the complete new
profile selected by `SILICON_QWEN_COMPACT` in `benchmarks/phase60/engine.c`.
It reads that exact artifact, uses shared512 + selected4*128 SwiGLU functions per
layer,14-query/2-KV-head causal attention, BF16 residual stream, F32 routing and
full151936-entry tied head. No original4864-wide MLP is loaded. Archive validation
and all operator/head/cache work are charged to their declared timing scopes.

The initial profile computes routing scores directly; it is a numerical
reference connection. Large-n CPU LUT winner AND normalized-mass fidelity are
still required future work. Distinct stored coefficients do not establish useful
extra capacity, and logical per-token bytes do not measure physical DRAM.

`chatbot_compact_client.py` supplies persistent binary generation/probe transport
and an adapter through the qualified source chat serializer/tokenizer. It records
canonical prompt IDs, raw generated IDs, bothEOS/length behavior, visible text,
compute/transport/end-to-end times and cold first response. Visible IDs are
unscored until behavioral scoring; invalid answers contribute time and zero
accepted useful IDs. Caller must bind provenance, watchdog and owned processes.

### Native diagnostic wire, not a rate assay

`probe` mode accepts `QWCP0001`, little-endian U32 input-count/position-count,
strictly increasing U32 positions (1..16), then complete input I32 IDs. Response
`QWCT0001` has four U32 fields:151936,position-count,24,896. Each selected
position emits U32 position, ALL24x4 selected I32 IDs in ascending ID order,
ALL24x4 mass F32,ALL24x32 query F32,ALL24x896 input BF16,ALL24x896 raw compact
output F32 and151936 head BF16. A frame is436740 bytes. It recomputes the
teacher-forced history through the native cache; no saved donor hidden state is
substituted. Trace collection/I/O are excluded from generation rate claims.

## Algebraic separation of uncertainties

Let D be the donor, S the fixed fitted student, E its decoded deployed codec,
and C the native evaluator of E. The three comparisons answer separate questions:

1. D versus S: did joint training preserve complete-output behavior?
2. S versus E, and D versus E: did rounding break the fitted representation?
3. E versus C, and D versus C: do native arithmetic/history preserve behavior?

Errors cannot be added as a claimed KL bound: KL lacks a triangle inequality,
and generation changes the state distribution after its first changed token.
Compare each complete model on declared trajectories and independently check
fresh own-history behavior of the SAME deployed artifact.

For every compact block retain its actual input, query, selected IDs, normalized
masses and output on new encoded/native trajectories. This distinguishes routing
discontinuities from coefficient/accumulation errors. Source-shaped BF16 operator
boundaries and local F32 arithmetic are explicit; identical tensor shapes alone
do not prove Torch/C arithmetic parity.

## Sequential next executions, conditional on eligibility

1. Finish session77989. Preserve its first terminal/complete or partial outputs.
   Freeze and execute the first saved-output/update auditor once; retain faults.
2. If all8 fixed transfer gates are independently true, freeze an export worker,
   protocol, actual input/runtime binding and resource limits BEFORE export.
   Charge source/final24 reads, codec/hash/catalog writes and complete exit.
3. Load only the exported artifact into a decoded reference student. Check the
   fixed200 teacher-forced cases against donor and fitted S using unchanged
   whole-output gates. Preserve actual new inputs/routes/operator witnesses.
   No retraining or endpoint answers used to choose precision/checkpoints.
4. Freeze the exact C compiler/flags/catalog/engine sources and qualification
   protocol before first compile/forward. Compare newly retained encoded
   reference witnesses and complete C outputs, including canonical token IDs,
   bothEOS/cache/context/head. Static code inspection is not numerical admission.
5. Execute all64 reserved tasks and16 excluded two-turn own-history dialogues,
   with separate own assistant histories for source and native. Apply the
   [already frozen behavior contract](CHATBOT_FRESH_BEHAVIOR_CONTRACT_20261008.md).
   Retain all96 outputs and explicit per-case scoring, including failures.
6. On that identical native blob/config, measure complete batch1 request mix,
   tokenizer/serialization/history/emission included. Report cold/first and warm
   rates, accepted useful IDs, context/threads/hardware and physical memory.
   Require>=50 accepted IDs/s; then investigate useful n/LUT/families/scales.

Each execution receives finite budgets/stops and binding before observables.
This document does not freeze unknown future artifact hashes or claim a runnable
pipeline command, encoded/native quality, throughput, useful-n or family coverage.
