# Proposal: bounded nonlinear source-response amplitudes

**Historical tested proposal: METH-256 fails useful count gain.** The
[result](METH_256_BOUNDED_SOURCE_RESPONSE_RESULT_20261002.md) passes all176
distinctness/numerical controls but gains only0.08439% versus matched E16
and0.03415% versus rotation, below frozen10% thresholds. Close this fixed
recipe before native scalar-blend/codec work. See the
[fixed pilot](METH_256_BOUNDED_SOURCE_RESPONSE_PROTOCOL_20261002.md).
METH-254 fixed
full feature promotion stops at6 positive units. METH-255 measures28.02%
fit loss when fixed32 original features are forced through the already
adapted readout; source-reset composition is4.56x less accurate. Preserve
the actual learned parent and learn how source nonlinear responses contribute.

## Changed private content and coupling

For each source unit define `delta_j(x)=source_BF16_gate/up_LUT_j(x) -
shared_Q8_gate/up_LUT_j(x)`. With the complete actual parent coefficient
column W[:,j], define vector-valued nonlinear atom
`response_j(x)=W[:,j]*delta_j(x)`.
The source rows are original pretrained weights; private learned values
are scalar amplitudes, not newly synthesized gate/up rows. Zero amplitude
retains the complete original function. Signed bounded amplitudes can
compensate rather than force a precision change on an adapted readout.
The dictionary is nonlinear in input, unlike old affine/readout-only cells.

First freeze a **nondeployable continuous pilot**. Per parent choose32
source-response atoms from original fit targets with a fixed signed
regularized matching rule, then jointly fit amplitudes in[-1,1]. Reuse
original source/capture/keys and tau1024. Response Gram equals
`(delta' delta) elementwise-multiplied by (W' W)`; use variance/energy-scaled
ridge and explicit bounded-solver/KKT/monotonic/finite controls. Exact
regularization normalization, greedy rule/ties, iterations and guards must
be specified before running, not selected from observed results.

Keep these32 original source atoms shared within each parent's10 children.
Retain the full fitted parent-amplitude function. Fit child amplitude
adjustments around it with the same original regularization sample rule
and total amplitude bounds. Both E16/E160 use exactly32 active nonlinear
atoms and the same input/output/source weights; only independently learned
amplitudes and routing differ. Do not change source rows per child while
claiming to retain the complete parent within the same active budget.
Constant-input cells use known source nonlinear atoms and regularized
amplitude values; no invented derivative/fitted-slope claim or ID noise.

Require actual parameter/probe distinctness, all original parent byte/score
controls, exact snapshot and bounded-normal/KKT controls. Keep original
absolute1%, count/rotated/prior10% and positive paired-window gain gates on
the same consumed development screen, only after fit prerequisites.
If the fixed continuous recipe fails, close it before native/codec work;
do not tune amplitude bounds or ridge strength on that screen.

## Prospective native form and pricing

An eventual FP32 source-feature blend can compute all shared phi, then
`phi[j]=phi[j]+alpha[j]*(private_phi[j]-phi[j])` for32 original BF16 rows,
before unchanged readout. Source rows/IDs and parent readouts can be shared;
private amplitude storage is32*4bytes per layer,3,072bytes across24 layers
per leaf, separately from keys/shared parent dictionary bytes. Count unique
donor rows, copied storage, independently learned amplitudes and meaningful
function variation separately. Small amplitude banks do not by themselves
demonstrate conserved semantic capacity from a100B donor.

New scalar blending arithmetic is not the qualified METH-253 function.
If continuous count qualifies, freeze actual scalar encoding, addition
order, stored-parent conservation and native learned-bank numerical/cost
checks. Added selected alpha payload is only3,072bytes by shape; runtime,
precision preservation, route/LUT and real large-RAM DRAM must be measured.
No source timing/full quality/rate carries automatically.

## Full-goal boundary

The qualified fast operator and accurate learned E16 function are assets
for a compact transferred core, but source-prefix/local fits cannot replace
an actual full24-layer saved model with independent prediction/generation/
tasks and>=50 accepted batch1tok/s. Do not refine the dictionary indefinitely
without that complete-model path. Larger dense source features grow active
cost; sparse GigaChat needs a separately priced top4/shared/MLA/head variant.
Frozen donor-adaptation assets remain reusable, not an automatic port restart.
Native large-n routing/quality and multi-family/10B/100B transfer stay open.
