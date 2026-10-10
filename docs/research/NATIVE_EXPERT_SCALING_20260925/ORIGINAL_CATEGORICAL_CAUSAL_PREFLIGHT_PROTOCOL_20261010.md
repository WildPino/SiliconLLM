# Original categorical causal bridge: frozen preflight

10 October 2026. NEW experiment; full goal ACTIVE/INCOMPLETE.
Producer and stored audit each have a 900-second held budget. Local RTX3060,
Ryzen53600X and 80GiB RAM only. No T4 allocation.

## Decision and reused evidence

Qualify whether the compiled fixed-head categorical objective can propagate
through the original causal SSM/SWA and selected ternary banks, with coherent
final norm/head and a measured native C comparison. A numeric PASS allows
preparation of a NEW first-balanced core/bank optimization campaign. A numeric
FAIL requires attribution to the measured discrepancy before optimization.
This single FIT case cannot admit chatbot quality, generalization or speed.

Reuse actual27 immutable seeded state (8,652,210,298 bytes, mmap model only),
the qualified original packed ABI/executable, original learner/custom streamed
surrogate adjoint, fixed native head/norm and the fully audited supervision
compiler. No old Adam restoration, optimizer update or source generation.
Select shortest FIT by (history length,id): rewriting_008, 58 history IDs,
18 labels at positions40..57. Preserve all DEV and RESERVED boundaries.

## Algebra and finite precision

H is the actual fixed 65537x256 F32 head interpreted as real coefficients,
with zero last column; final RMSNorm weights are F32 ones, epsilon1e-5.
For each qualified source q, m=H^Tq and c=sum(q log q).

    L(f)=logsumexp(Hf)-m^Tf+c
    gradient_f L=H^T softmax(Hf)-m.

Compiled and full-q losses share ONE GPU F64 Hf forward. Compare two F64
autograd state gradients, then execute ONE compiled weighted backward through
the original F32 core. First label has weight1/2, others1/(2*17).
H/norm are frozen; an additional 134,219,776-byte GPU F64 buffer is TRAINING
ONLY. Packed inference retains F32 norm/head and the original C operators.
All 90 trainable parameter gradients must exist and be finite; preserve their
full tensors (~2.817GB), hashes, shapes and squared norms. Verify nonzero
aggregate bank/router/embed/core gradients. This is custody and local readout
derivative qualification, not an independent proof of the nonlinear STE.

The source model's 92 parameters/721008128 entries remain byte/value unchanged
except the two replaced fixed fields. Copy the original 520029440-byte packed
artifact; replace ONLY final_norm and head. Verify the full 110-field ABI,
header/table and all108 other fields by prefix/suffix bytes, not sampled rows.
Run ONE original C prefix of58 IDs; compare all18 selected full-V distributions.
Native routes (IDs AND normalized mass) and ternary witnesses must remain
byte-identical to the prior same-input core.

For old qualified normed coordinates fhat and nonzero old Gamma, put
g=fhat/Gamma and E=Eold/min(abs(Gamma)). With u=2^-24 and gamma2=2u/(1-2u),

    Enew=E+2*gamma2/(1-gamma2)*(norm(g)+E).
    B=max_row_norm(H)*(Enew+gamma39*(norm(g)+Enew)).

This conservatively covers old and new two-multiply F32 final-norm paths
sharing the unchanged core/RMS inverse; compare all18 actual C score errors
to B. The existing gamma39 is2.3245865499303013e-6. It is a conditional
unexceptional rounding model evaluated in F64, not a directed interval proof
or an own-history bound. GPU/C actual score, KL and argmax comparisons remain
separate gates; no cast term for H interpreted as its actual F32 coefficients.

## Criteria fixed before observation

| Gate | Limit |
|---|---|
| Dense/compiled F64 label loss and weighted state gradient | absolute1e-8 |
| F32 upstream gradient versus cast compiled gradient | absolute1e-6 |
| GPU versus CPU F64 actual-head scores/losses/state gradients | absolute1e-8 |
| GPU versus native C full-V scores and per-label KL | absolute1e-3 each |
| GPU versus native argmax | zero disagreements |
| Native score coordinate bound | B+1e-8 |
| Stored independent metric recomputation | absolute1e-8 |
| Checkpoint execution | six outer blocks + six recomputations |
| Packed unchanged fields/routes/mass/witnesses | exact |

Diagnostic case quality: mean KL<=.05 and argmax disagreement<=5%. Report
these flags separately, even catastrophic failure; ALWAYS complete the stored
audit if producer data exist. No quality-dependent early exit or reduced audit.

## Resources, custody and stops

Producer held cap900s, worker reserve120s; union OS peak24GiB (worker, holder,
native child), GPU allocated6GiB/reserved8GiB, output4GiB, log8MiB. C child100s.
Audit held900s/OS12GiB/output2MiB/log8MiB, GPU hidden, no native/model/backward
execution. Caps include complete before/after consumed input/output hashing,
the 8.65GB checkpoint, imported actual runtime sources/bytecode and DLLs, and
full gradient storage/audit. No unrelated donor-weight scan.

Freeze code/protocol/binding/holder before consumption; binding resolves
package junctions to actual files. The holder seals its inputs separately,
observes worker PID/creation/command and terminal Win32 OS peak, admits exactly
one direct --prefix child of the pinned C executable, and preserves foreign
publisher daemons. Any unexpected overlap is a prelaunch rejection; preserve
its evidence and select a NEW holder binding if justified. No speed admission.
Kill only owned worker/native child on an actual cap/mechanical fault. Retain
first_fault, partial artifacts and terminal. Never replay completed inference
or backward or edit consumed code; any repair uses a new namespace/binding.

Stored audit verifies complete input/output hashes, all92 parameter states,
all90 saved gradient tensors, role sums, exact query bytes/native command and
receipt, unchanged packed fields/routes/witnesses, all18 metrics with an
independent shifted logaddexp normalizer, CPU full-q/compiled derivatives,
coordinate bounds, resource/counter receipts and recomputed decision.

Counters: one outer causal history forward, six checkpoint recomputations,
one full backward, two F64 state-gradient checks/36 labels, one F64 head
forward/18 labels (its contractions are differentiated three times), one C
history/58 positions/18 paired labels. Stored audit repeats none of these.
No optimizer/SVD/source/DEV/RESERVED/T4 calls. All work and failed passes are
charged; component numeric PASS is not useful chatbot or same-artifact50.
