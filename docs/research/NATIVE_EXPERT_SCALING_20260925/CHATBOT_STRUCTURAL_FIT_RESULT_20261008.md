# First joint value/full-J learner: structure improves, conversion fidelity still fails

8 October2026. Goal ACTIVE/INCOMPLETE. The prospective converter objective is
now IMPLEMENTED/EXECUTED/INDEPENDENTLY VERIFIED, not just a proposal. Final
F32 checkpoint and complete new outputs retained. THIS one finite recipe is
CLOSED; no whole/BF16/native/count/rate promotion or optimizer/width/lambda ladder.

## New mechanism and exact decision

Source knowledge preservation is still the missing stage in the full CHATBOT
pipeline. The first value-only E16 trial and completed source-directional
diagnosis motivated a changed objective: jointly fit every shared/leaf G/U/B
using values AND FULL896x896 source J at16 fixed FIT anchors. Reuse original
BF16 x/y and source F64 J bytes; no original source forward/feature/J/PCA/control
or initializer replay. Sixteen novel J and original development y excluded from
fitting. Warm start actual previous final weights; original source-row initializer
checkpoint is the coefficient reference; Adam reset, router/active cost unchanged.

Exactly24 epochs/744 updates, batch128/final5, derivative coefficient1, coefficient
prior.001, Adam.0003 and original deterministic shuffle. Original response loss
FULL-FIT energy normalization retained; prior active leaves include BOTH response
and J-anchor selections. [Protocol](CHATBOT_STRUCTURAL_FIT_PROTOCOL_20261008.md)
clarifies the provisional NEXT's batch-denominator proposal BEFORE new values.
Final epoch only. Eight FIT anchors47 visits and eight46, all update identities
and epoch curves independently reconstructed. Checkpoints6/12/18/final24 retained;
no intermediate checkpoint selected or reevaluated as the result.

| Audited pooled metric | Prior value-only final | New value+full-J final |
|---|---:|---:|
| FIT response relative RMS |21.67793844%|23.69544448%|
| Novel response relative RMS |61.39205832%|60.18920317%|
| FIT full-J relative RMS |82.73868336%|52.99360200%|
| Novel full-J relative RMS |82.73713071%|68.78773229%|
| FIT selector-null J relative RMS |83.27645927%|53.33020740%|
| Novel selector-null J relative RMS |83.23385603%|69.11139956%|

Novel response RMS improves only about1.96% relative; FIT response worsens about
9.3%. FIT J improves about36%, novel J about16.9%. These are local/calibration
observations, not semantic percentages, optimum/capacity bounds or convergence
proof. Isotropic full-J directions are not the covariance of reachable states.

| Frozen gate | Actual |
|---|---|
| Novel response improves>=10% |FAIL |
| FIT response worsens<=5% |FAIL |
| ALL16 novel categories worsen<=5% |PASS |
| FIT full-J improves>=10% |PASS |
| Novel full-J improves>=10% |PASS |
| Novel response<=1% |FAIL |
| ALL16 categories<=3% |FAIL |
| Distinct/changed16 BF16 coefficient hashes, finite/layout/probe/resource |PASS |

Decision `CLOSE_ONE_STRUCTURAL_FIT_NO_WHOLE_PROMOTION`. Coefficient hash
distinctness is not useful-count or function-identity proof under all neural
symmetries. E160 uniform support failure remains CLOSED; this job fits E16 only.

## What changed the next action

Source-directional constraints measurably improve the learned derivative field,
including novel anchors. That does not recover sufficiently accurate novel
responses. This single result neither refutes all derivative learning nor proves
that the current target cannot express the source; the parameter optimum is
unproved. It does close this fixed optimization recipe. More coefficient/epoch/
width tuning would not identify the source-preserving mechanism needed next.

Change the conditional-function representation. The next bounded uncertainty is
whether shared nonlinear512 plus top4 FULL-input affine functions can be compiled
from jointly compatible value/J constraints. This changes the former128-channel
bias-free private functions, not only an optimizer coefficient. First price the
whole changed artifact and test a small16-mass/528-jet design BEFORE source/core
responses or field construction. [Actual NEXT](CHATBOT_COUPLED_JET_NEXT_20261008.md)
derives the equations and preserves whole admission gates.

METH227's common-free, independently compiled first-order tangents were exact
at centers but failed59.70% normalized SSE on their consumed validation data.
That closed nearest-tangent recipe is not reopened. A fixed shared nonlinear
core and simultaneous top4 value/J interpolation are different degrees of
freedom; their success is unproved. Old231 fixed-G/U derivative readout and
310/311/316 constrained output subspaces remain closed as well.

## Independent numerical/byte evidence

[Audit protocol](CHATBOT_STRUCTURAL_AUDIT_PROTOCOL_20261008.md) is frozen before
first audit. Stdlib BF16/NPY/header/Counter/math.fsum response reader agrees
within1e-12 absolute-or-relative on ALL primary/occurrence/category/prompt/
generated new energies/SSE/RMS. Per-selected-leaf response metrics and original-
reference displacement are explicitly outside the independent reconstruction.

Complete new student J/probe C-order F64 frame offset/length/SHA independently
verified; CPU NumPy reconstructs every error/parallel/null/pooled/RMS decision
against REUSED old source J/energies. ALL160 new final-student directional checks
pass. Max Jv reconstruction gap4.0039e-17, saved FD gap0, maximum finite-difference
discrepancy/tolerance2.76262e-5. F64 source FIT labels' one F32 rounding/byte SHA
matches; gradients are not rerun/audited by an optimizer replay.

Safe Torch storage descriptors, no Torch import, plus independent integer
nearest-even F32->BF16 rounding verify all16 coefficient hashes and changes.
ALL frozen router-buffer bytes equal the warm checkpoint. No control rerouting,
exposure/PCA/source-information acquisition or old response audit replay.

New exact necessary1%-RMS failure witness from first64 novel saved predictions:

    prefix_error =693996098624213873760178923359040607945568325675439123112118952667577447511702887421658005504
    FULL_source =38784185341458312745673750575405918767006530030601447547086179504264113891695950381392810999808
    10000*prefix_error > FULL_source

Integers share squared units2^-298, weight LCM2. FULL source denominator is reused
from the original byte-qualified exact audit8eca7c81, not recalculated. Positive
new prefix error alone exceeds the allowed FULL-domain error; this proves saved
finite failure, not all-function impossibility or semantic generalization.

## Actual cost, code and provenance

Source freeze531fb903cac6948f2a12b891a1262132e932c8bd;
`chatbot_structural_fit.py`, `chatbot_structural_binding.py`, common monitored
`chatbot_directional_launch.py`; independent `chatbot_structural_audit.py`/binder.
Commands/actual resolved inputs/runtime tree/SHAs/output manifest/process creation
identities in raw+terminal receipts. Python3.12.10/Torch2.6.0+cu124/NumPy2.4.6/
psutil7.2.2, local3060, restricted runtime/6 threads/TF32off/deterministic.

| Job | Worker s | Fit s | Family s | Worker OS through exit B | Conservative family B |
|---|---:|---:|---:|---:|---:|
| First changed learner |94.078 before final serialization|68.484|118.047|1776111616|1808879616|
| Independent retained audit |8.438|0|11.984|533520384|563949568|

Fit GPU peaks327931392 allocated/346030080 reservedB. These clocks are not native
rate, physical DRAM, a matched causal cost comparison to the old fit or a whole
conversion cost. Held handle includes actual worker exit; launcher final receipt/
stdout tail remain outside its last peak snapshot, as disclosed. No T4/download/
install or whole source-weight use. All prospective resource caps PASS.

ALL341392241 output bytes under `results/native_expert_scaling/chatbot_structural_fit_20261008/`:
four F32 checkpoints, two full F32 response matrices, full final-student F64 J,
all new Jv/FD/plus/minus vectors,744-update journal and32-anchor journal.
Raw result SHA `b4dfe3a8cea4d76a43e66ff71f38dfd5fbcbb7163effbcad857d06fa1f5fc7c1`;
audit SHA `bf1742b135f3f71c39f6ab63cf2ce494d4b03ec96ed998503a324f13d4bb122b`.
Audit source07cf2bce8f234dcd53719c992ea0674bd652bba6, both actual exit0.
Administrative `chatbot_structural_terminal_20261008.json` closes all4 known
instances, no matching typedUTC Windows Event1000 faults; controls179810/179791.
Foreign three tracked SHA unchanged. No live scientific process.

The actual converter learner exists and has been validated as a failed finite
candidate. Full source->eligible ALL24 bank->C canonical integration->FRESH chat
quality AND accepted50 SAME artifact, useful larger n/LUT winner+mass/physical
DRAM/other families/~10B/~100B remain open. Goal completion not claimed.
