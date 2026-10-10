# Fixed native-head loss compiler and complete independent audit

10 October 2026. Before numerical observation. Implements the first component
of [categorical causal transfer](ORIGINAL_CATEGORICAL_CAUSAL_TRANSFER_NEXT_20261010.md).
New converter input format, not another readout fit or an old optimizer dose.

## Uncertainty, input and decision

Can the qualified source distributions be compiled into compact supervision
that conserves their categorical loss and state gradient for one fixed original
F32 head interpreted in real arithmetic? A successful result enables a NEW
core/bank training input. Failure isolates compiler/precision/identity defects;
it does not authorize altered quality gates or long training.

Reuse all48 source cohorts/8808 labels, complete original native/source custody,
paired native head/final norm and the existing72 native-normalized probe states.
Teacher raw scores BF16, decoded exactly; FIT24/4422 and DEV24/4386 separated.
Head65537x256 F32 SHA768aaf9e17873707624e4489a8f14f345afb2722e256de871113b7d6e64baa66,
last column zero; final norm256 ones/original epsilon1e-5. Existing selected72
teacher rows must equal newly decoded source scores bitwise, including boundary
IDs/domain/positions. Same-byte FIT source copies have different paths; compare
their sealed byte hashes and verify selected raw values, not path identity.

For fixed H and q=softmax(source scores), compile m=H^Tq and
c=sum(q log q). Store256 F64 moments, one negative entropy F64 scalar and source
argmax int32 per label, plus split/domain/canonical positions/extents. No teacher
approximation or top-k truncation. This tuple conserves real fixed-H loss
logsumexp(Hf)-m^Tf+c and state gradient H^Tsoftmax(Hf)-m for any f. It cannot
reconstruct q, support a changing head, or automatically certify F32 GPU/native
loss/gradient equivalence. Preserve all original source scores. Training loader
must require successful complete adjudication and only frozen FIT identifiers.

## Numerical observables and frozen tolerances

Producer uses row-max-shifted exp/sum normalization and full-vocabulary matrix
contraction. Complete independent audit uses row-max-shifted sequential
logaddexp and4096-vocabulary-block contractions for all8808 moments/entropies/
IDs. All144 compiled record-array hashes, all witness arrays and source/head/
parent/runtime inputs before/after. Moments finite/last coordinate exactly zero.

On SAME72 stored original-normalized states, compare direct dense-q KL and
gradient with compiled KL and gradient using SAME fixed F32-as-real head:
loss absolute1e-8,gradient1e-8. All source/target probability mass errors<=1e-12;
allmoment absolute1e-8,negative entropy1e-9; independent witness arrays1e-8.
All72 scalar moment-state loss terms use math.fsum256; audit adds24 full-V scalar
moment witnesses (one FITfirst percase, coordinate17*case modulo255) and12
full-V gradient witnesses (every sixth selectedlabel, coordinate13*i+7 modulo255),
absolute1e-8. No relative-error gate near zero or silently relaxed tolerance.

Both stages save decisive maxima and numeric flags even when numerical flags
fail; complete all labels/witnesses. Structural/hash/nonfinite/resource faults
remain mechanical failures, preserve first stage/state. An unqualified result
cannot be used by training merely because arrays exist. Producer decision
LOSS_COMPILED_PENDING_AUDIT or LOSS_COMPILE_NUMERIC_FAIL; final adjudication
LOSS_SUPERVISION_QUALIFIED only if all producer AND independent flags pass.

## Cost and control

Proposed/reviewed dimensions: numerical supervision payload
8808*(256*8+8+4)=18144480 B; four72-label witness arrays296064 B, record metadata
and result below128MiB. One source-moment pass8808, additional72 source/target
probability rows and72 direct/compiled loss-gradient witnesses perstage. Charge
all readout/moment arithmetic, including independent blocks/scalar witnesses.
No claim this removes full-V head work or improves chatbot quality by itself.

Producer300s/reserve30/OS2GiB/output128MiB/log4MiB; complete audit300s/OS2GiB/
output1MiB/log4MiB. CPUoneBLASthread/worker1/holder11/CUDAoff/nochildren. Actual
imported runtime/stdlib/project module files and existing bytecode plus Python
exe/DLL and NumPy/psutil native libraries sealed after junction resolution.
Separate generic holder binding frozen to this new code and exact inputs.
No other owned benchmark overlap; preserve background daemon. No source
generation/model/native history/optimizer/SVD/GPU/RESERVED/T4 call. No new
oracle/head fit/native readout, no quality/speed/useful-n/family admission.

After a qualified compiler, prepare one changed-head/norm original core/bank
forward/backward/native bridge preflight before fixing a new local training
budget. That preflight and campaign are not implemented or authorized by a
compiler pass alone without their own bindings, numerical checks and costs.
