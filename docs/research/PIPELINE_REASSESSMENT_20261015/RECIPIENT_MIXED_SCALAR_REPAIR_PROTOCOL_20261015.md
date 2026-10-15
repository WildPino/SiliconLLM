# Mixed recipient: scalar-reference primitive repair

15 October 2026. Freeze before a new model/reference run. The failed mixed
attempt and its original auditor remain unchanged. This is one explicit
reference repair, not a relaxed numerical threshold or a timing rerun.

## Evidence permitting the repair

The instrumented diagnostic frozen at `1d35abe` completed successfully.
Its native QWCT probe is byte-identical to the retained failed probe, and
all four reference logit hashes reproduce the failed audit exactly. Native
and reference traces are SHA256
`bb965c40a21648b677f6b10f5ee780d15c07cc03a5ea47df4f759baa732c5db8`
and `870dd12dd34f1c48c7e40f7036a62089fc65273c0584d4395796c617a2d2c296`.

The first floating difference occurs in position 0 / layer 0 SiLU activation,
despite exactly matching gate and up inputs. Scalar binary64 `math.exp`,
rounded to F32 before each prescribed F32 operation, reproduces all 4864
native activation outputs at that site; the original NumPy F32 exponential
path differs at 814 outputs. Before continuation, extend this stored-vector
check to every native SiLU site at positions 0 through 7. This consumes
retained vectors only and must reproduce all native activations exactly;
otherwise stop this proposed repair before model execution.

The first activation-code disagreement is position 1 / layer 21 / down
input index 2805. Native/reference inputs are -0.09306909143924713 and
-0.09306883066892624, with identical scale 0.006418542470782995; rounding
produces -15 versus -14. The next layer's Q projection already exceeds 1e-3
relative L2. Thus small primitive differences can become discrete execution
differences. This does not imply that all later differences have one cause.

## Exact continuation

Keep the original archive, catalog, conversion manifest and compiled release
executable in their original paths. Do not re-export, recompile or change
the native forward. The later C edits are diagnostic guards only; the exact
original source Git blob at `4b78ac7` remains bound to that executable.
Bind every unchanged original input, all retained failure/artifact receipts,
the diagnostic and stored-SiLU evidence, and the new wrapper/controller.

The new auditor wrapper imports the unchanged original decoded graph and
changes only its exponential primitive: evaluate each input with scalar
Python `math.exp` in binary64, then cast to the original F32 input/output
type. All array operations, quantizers, packet checks and comparison gates
remain unchanged. This is independent high-level graph code using a defined
scalar math primitive; it does not import C kernels. Shared CPU math/library
dependencies remain a limitation. Bind both wrapper and original auditor.

Run a new native probe on the same saved request and require byte identity
with the original failed probe. Then run the scalar-reference wrapper, which
again verifies every source-to-packet role before the complete 32-token graph.
Require exact top1 and <=1e-3 relative L2 for both normalized states and whole
logits at positions 0,7,15,31. Failure stops before cost and preserves outputs.
No automatic retry or additional primitive-repair grid follows this protocol.

Only if this gate passes, run the original 15 cost processes: three independent
repetitions of the same three prompts and contexts 512/2048, six physical
workers, ACTIVE/infinite, up to 64 own-history outputs with both EOS IDs.
Charge every non-EOS output's full forward including the final output. Preserve
all observations, repeats and start/end actual worker masks. The unchanged
cost allowance is <=18 ms worst-process average and <=1.10 max/min rate spread
in each of five groups. Prefill remains separate. No source-relative quality,
very-large donor-memory transfer, measured DRAM or final useful-50 claim follows.

## Resources and closure

No download, conversion, training or T4 allocation. Exclusive new namespace,
owned PID/create-time and terminal/log hashes. Cap each child at 900 seconds
and 8 GiB observed RSS, entire continuation at 2400 seconds and 2 GiB outputs.
Verify all pinned files and original artifacts again at closure. Runs remain
below one hour, so the goal stays active. Months of T4 usage remain acceptable
for the later qualified conditional-transfer program.
