# Independent retained-byte verification, no function or optimizer replay

Original E160 exposure closure and FIRST E16 baseline fit are complete. This
audit independently interprets their saved source/prediction/control bytes.
No source FFN, query/projector/router, fitted function, optimizer, tokenizer or
native inference is executed. No numerical parameter/gate changes.

Use stdlib-only source BF16 row/header reader and F32 C-order NPY parser;
all input SHA/logical paths/byte sizes freeze first. A restricted pickle reader
for self-generated Torch route files accepts ONLY tensor/storage descriptors
and OrderedDict; no Torch import or unrestricted pickle execution. Require
exact shape/stride/storage/offset/count/byteorder and finite normalized mass.
Independently verify ALL16/160 saved choices/exposure sets, exact parent ID/mass
byte identity, FIT uniqueness/conversation support/novel DEVELOPMENT counts
against original qualified BF16 input bytes. The already-completed control
choices are read, never re-evaluated.

For ALL four final E16 prediction matrices (F32/rounded-coefficient reference,
FIT/DEVELOPMENT), compute complete-response errors with scalar stdlib math.fsum,
independent of producer NumPy reductions. Require primary/occurrence/generated/
prompt/ALL category rows and energy/ratio/RMS match within1e-12 absolute OR
relative (fixed before audit). Reuse producer prediction arrays; no prediction
or training repeat. Source target energy and multiplicity remain identical.

For BOTH primary novel DEVELOPMENT fidelity failures also provide an EXACT
integer certificate. Represent finite F32 predictions and BF16 targets in
units2^-149. Compute FULL novel source squared energy and first64 novel-row
error squared energy, with reciprocal-x-multiplicity weights cleared by their
integer LCM. Common2^-298 units cancel. If10000*prefix_error>full_source_energy,
the full saved1%-RMS criterion necessarily fails, regardless of any floating
reduction. This proves this finite saved result's failure, not representation
impossibility/semantic loss bounds. No extra prefix search or threshold tuning.

Worker120s/whole family180s/summed OS512MiB/output1MiB, local CPU/no GPU/scientific
packages. Hold actual worker OS handle through exit; parent receipt/stdout tail
outside final snapshot explicit. Preserve first fault, no complete audit replay.
Final known original pilot/baseline/audit OS instances receive one typed UTC
Event1000/positive-control closure check, without scientific execution.
