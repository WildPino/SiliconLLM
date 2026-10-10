# Fixed original head: categorical supervision compiled and qualified

10 October2026. **LOSS_SUPERVISION_QUALIFIED**, complete independent audit and
**FIT_SUPERVISION_HANDOFF_PASS**. Full goal INCOMPLETE. This is a real converter
input/consumer component, not a new useful chatbot or a speed admission.
[Protocol](ORIGINAL_CATEGORICAL_LOSS_COMPILE_PROTOCOL_20261010.md),
[result](original_categorical_loss_compile_result_20261010.json),
[complete audit](original_categorical_loss_compile_scalar_adjudication_20261010.json),
[FIT handoff](original_categorical_supervision_handoff_receipt_20261010.json).

## Implemented operation and exact scope

For FIXED original F32 head H interpreted as real coefficients, compile the
qualified source distribution q into m=H^Tq and c=sum(q log q). At any target
postnorm state f, real KL is logsumexp(Hf)-m^Tf+c, with state gradient
H^Tsoftmax(Hf)-m. All8808 source labels are compiled, without source forward,
teacher truncation, head fit or model update. Full-V target head/normalization
still costs work. This tuple is sufficient for this fixed-head loss/gradient;
it cannot reconstruct q, train a changing H or establish finite-precision
GPU/C equivalence by algebra alone. Original source files retained.

The selected head is the prior paired native65537x256 F32 head (last column
zero), SHA768aaf9e17873707624e4489a8f14f345afb2722e256de871113b7d6e64baa66.
Final norm is256 ones, original epsilon1e-5. Prior native norm/head injection
witnesses reused; no new C call. The previous quadratic codec and global onset
interpolation failures remain failures; this compiler improves the training
input contract, not their measured quality.

## Complete verification

All48 source cases:24 FIT/4422 labels,24 DEV/4386 labels. Per-label F64 moments,
negative entropy and int32 source argmax; all source IDs match original generation
IDs, positions/domain/split preserved. All72 selected original-normalized states
are reused for NEW dense-q versus compiled loss/state-gradient checks. Their
source raw scores match newly decoded BF16 source scores exactly. FIT source
copies under different paths agree by byte SHA and selected raw values.

| Observable | Producer | Independent complete audit | Frozen gate |
|---|---:|---:|---:|
| Dense vs compiled loss | 5.03637658921e-14 | 2.21074460821e-13 | 1e-8 |
| Dense vs compiled state gradient | 5.15773946989e-13 | 1.43679864013e-11 | 1e-8 |
| All8808 moment differences | not an independent comparison | 3.65218966181e-11 | 1e-8 |
| All8808 negative entropy differences | not an independent comparison | 3.36619621066e-13 | 1e-9 |
| Scalar moment discrepancy | not evaluated by producer | 2.0872192863e-13 | 1e-8 |
| Scalar gradient discrepancy | not evaluated by producer | 4.60785923306e-19 | 1e-8 |

All producer and final audit numeric flags PASS. Full source/target probability
mass gates1e-12 PASS. All395 input and148 output hashes, all8808 statistics/IDs,
all72 witness arrays/split linkage/finite/zero last moments/counters/resources
checked. Audit uses shifted sequential-logaddexp normalization and4096-vocabulary
blocks independently of producer exp/sum/full contraction.24 full-V scalar
moment,12 full-V scalar gradient and72 scalar loss witnesses. Differences are
F64 observations under these algorithms, not interval certificates or a proof
of native/GPU backward correctness.

## Storage and practical converter benefit

Original source BF16 scores 1154499792 B; compiled numerical supervision
18144480 B, ratio 63.628155340 times smaller. This includes256 F64
moments, negative entropy F64 and source argmax int32 per label, excluding JSON
positions/provenance.148 output files 18440544 B
include four additional72-label witness arrays; result 195317 B.
All data are split by case; future FIT training can stream compact targets.
The source q is still needed if H changes or another teacher observable is used.

`original_categorical_supervision.load_fit_supervision` checks successful complete
adjudication/terminal/result and fixed head/norm/FIT-array extents. Its training
API returns ONLY24 FIT/4422 labels, no DEV selector, read-only arrays. Weights
are .5 onset and .5/(n-1) continuations, summed within each case; average cases
or sample them uniformly. This implements proposed weighting, not an observed
training benefit. Actual direct handoff PASS in 1.391s,
OS peak 47230976 B, case weight sum discrepancy 4.4408920985e-16.
Consumed data/code/runtime hashes before/after exact in receipt. No probability
product, model or optimizer was executed by this lightweight consumer check.

## Preserved faults and truthful audit completion

Two metadata preparation mismatches were repaired before any numerical worker:
legacy native producer receipt uses result_sha256/output_files list/resource_gates
boolean rather than the later held receipt schema. Initial syntax/preparation
record retained; no scientific dose ran during preparation.

The first independent audit completed all8808 blocks and scalar/witness work but
terminated1 at final JSON serialization: NumPy bool not accepted by the existing
helper. No adjudication file and no saved final numerical maxima. Preserve
[fault terminal](original_categorical_loss_compile_stored_adjudication_20261010.terminal.json)
and log; do not retroactively call it PASS.

[Scalar adapter](ORIGINAL_CATEGORICAL_LOSS_AUDIT_SCALAR_ADAPTER_20261010.md)
converts NumPy scalars to native Python values via .item() at JSON serialization,
covering artifact and stdout; calculations, caps, thresholds and decisions stay
unchanged. Boolean/F64 value and NaN-rejection preflight PASS. Final complete
stored audit repeated to obtain the unsaved maxima; all extra work charged.
Frozen original code/result/compiler were not edited or replayed.

First adapter launch rejected uploader18752 before worker/log creation; retained
[prelaunch rejection](original_categorical_loss_audit_scalar_prelaunch_rejection_20261010.json).
[New exact-foreign holder](ORIGINAL_CATEGORICAL_LOSS_AUDIT_FOREIGN_HOLDER_20261010.md)
permitted ONLY observed uploader pair23300/18752 PID/creation/fullargv tuples.
Both observed during final audit; background daemon preserved, no process stopped
or messaged. Stored CPU arithmetic only, timing_control_admitted=false. Costs
below are held elapsed costs, not isolated throughput measurements.

## Revisions, commands and measured resources

Numeric compiler freeze4ea205b542ab448d86be065fcfe6cdcb31b40266;
bindingbcb329defd685e5327af892e9078a3728472bdbcd0601535435d9251e7cd23c7,
395 inputs1304474272 B. Actual imported runtime/stdlib/project modules, existing
bytecode, Pythonexe/DLL and NumPy/psutil native libraries sealed after junction
resolution. Producer holder8cbfcfa9; scalar adapter61c2057942cdea9a1d805df6835d373531e1cfeb;
final holderc40a8f2b937fde03264bfc290601c96fa822154b/bindinga7cfa6ee.
FIT consumer freeze03a5e2c03bb1ea120c01b6cb03a2d0c5da30fcf9.
Exact commands/PIDs/creation/extents in terminal and direct handoff receipts.

| Stage | Held seconds | Worker seconds | OS worker+holder snapshots |
|---|---:|---:|---:|
| Producer, exit0 | 80.359 | 76.218 | 602066944 B |
| First audit, exit1 JSON fault | 72.125 | no final result | 601251840 B |
| Complete adapted audit, exit0 | 86.563 | 81.704 | 600317952 B |

Combined held 239.047s,
plus lightweight handoff1.391s and separately recorded preparation.
Source probability rows26640/target rows216/stored loss-gradient witnesses216
across producer/two audits; both audit attempts include scalar work. No source
inference, native history, model, optimizer, GPU, SVD, RESERVED or T4 call.
Owned sessions50027/89422/79002 terminal; no owned worker remains.
Result SHA `86b35caa2b9f3b15d40909776a5b2c09d1c491d5644146bed144e0e0563a1d74`;
final complete audit SHA `9ba08a2cd668910995c5887eb645a8ae986645c42f6ec6b97ff895cbfe1ff4a0`.

## Next actual causal bridge

[Original categorical history adapter](../../../benchmarks/native_expert_scaling/original_categorical_history_learner.py)
IMPLEMENTED/AST PASS, UNEXECUTED. Existing widened original SourceLearner and
streamed adjoint; coherent fixed F32 head/norm, training-only F64 readout buffer,
explicit postnorm states and compiled categorical objective. This avoids an
unqualified exact-identity claim with rounded F32 training logits. It does not
change packed C readout precision, LUT, ternaries, SSM or SWA inference operators.

[Next causal preflight](ORIGINAL_CATEGORICAL_CAUSAL_TRANSFER_NEXT_20261010.md):
runner/binding UNIMPLEMENTED. Deterministic shortest FITcase rewriting008,
58 history IDs/18 labels, immutable actual27 core/banks, no old Adam continuation.
Review loader/shape/resource/numerical/native gates and freeze a NEW one-case
forward/backward/native bridge before training. Actual27 source state8.652GB
contains old Adam; mmap model-only loading/lineage, no resumption of that optimizer.
No long adaptation or T4 allocated. Own-history chatbot quality, useful same-artifact
50, useful RAM-driven n/CPU IDs+mass/physicalDRAM and other families remain open.
