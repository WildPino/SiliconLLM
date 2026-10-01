# Proposal: source-bound private nonlinear feature precision

**Proposed, not implemented/frozen/executed.** METH-249 and251 preserve
complete parents but fail useful count gain through fixed rank32 readouts.
Change the private function content before more output-space/rank/strength
attempts. This is a scoped transfer mechanism to test, not a solution claim.

## Source nonlinear knowledge in each private function

Retain the full shared row-Q8 gate/up feature operator,513-point SiLU LUT,
actual mixed readout and learned METH-247 parent factors/bias. Each private
function supplies **32 original BF16 donor gate/up rows** plus source unit
IDs. For those32 units, execute the BF16 rows against FP32 input through
the same LUT and product, replacing their approximate shared features
before the unchanged complete parent readout. This recovers source feature
content rather than fitting another output correction to the same features.
Other rows still use shared int8/scales. No source-point Jacobian reset.

Conditional choices then select different actual nonlinear source units.
E16 and E160 must use identical32-unit active geometry and fixed route/
parent/readout, with only the source-bound private unit sets changing.
Duplicated rows/IDs, weight hashes or a bigger file alone cannot prove
distinct useful functions. Gate actual stored/private function responses,
count advantage and rotated-choice controls on consumed development first,
then untouched model-quality gates only after component qualification.

This differs from METH-229/231's H2048 reduced common plus readout fitting:
the full shared feature inventory remains, while selected input projections
regain original nonlinear source precision. METH-238 changes output-row
precision; this proposal changes selected gate/up features themselves.
No prior result accepts the proposed geometry. Source FP32 sensitivity and
BF16 execution controls retain their original scoped meanings.

## Price and first native gate

Conservative added selected payload is
`(2*32*896*2 + 32*2)*24 = 2,754,048 bytes`: two BF16 row banks and uint16
IDs, no extra down matrix. Starting from the553,738,244 addressed-shape
ledger gives **556,492,292bytes**, below560MB. This upper ledger does not
assume skipped shared rows reduce physical DRAM traffic. Input/output
features and control organs are still charged as in the previous ledger.
Stored private capacity would be2.754MB per32-unit function across24 layers
before tensor headers/other fields. Rows may overlap; report unique donor
rows, distinct functions and resident bank bytes separately.

First implement/freeze **METH-252 native source-feature operator** before
cell selection/training. Reuse the qualified one-team METH-246 component,
retaining every original source fixture field. Add real BF16 source32-row
gate/up arrays and source IDs per layer, source-bound deterministic fixture
choice from gate/up quantization discrepancy, not quality-selected cells.
Build the active unit-to-private-row map before timed layer execution.
In the feature-row loop choose shared Q8 or private BF16 dot; use the
unchanged LUT/readout/order afterwards. Avoid two writers to a feature row.
Fix exact storage/order/fixture selector, independent GPU oracle, byte/
source-function controls and three six-thread passes<=10ms before running.
State budget/stop and source identities. No learned n or DRAM claim from
those source fixture timings.

Only if native geometry qualifies, separately freeze source-bound E16/E160
private unit selection on original fit captures/keys. A prospective
cell-wise greedy target-error selector can choose32 units using exact
source-versus-shared feature responses and complete parent coefficients;
retain cross-unit terms, actual nonlinear predictions, positive-gain stop,
fixed ties and identical active count. Its FP64 score approximation must
be checked against stored FP32 execution before consumed target scoring.
Freeze source/fit/numeric/distinctness/count gates first. Do not call a
target-informed greedy result the globally optimal subset or independent
held-out quality. No tuning on the consumed128-window screen.

## Limits toward the full goal

Correcting a small dense donor's feature precision is not itself semantic
capacity transfer, or evidence that arbitrarily many experts improve a
100B model. Even a positive component result needs actual saved banks,
native learned routing/LUT with large resident pools, full independent
quality/generation/tasks and>=50 accepted batch1tok/s on the same artifact.
Dense source features do not scale for free to large donors. A sparse
GigaChat variant must separately price its real top4/shared/dense layer,
MLA/head and source expert row sets using the frozen donor-adaptation
assets. No paused generic-port job resumes from this proposal.
