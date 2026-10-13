# Full integer readout audit V2: restore declared diagnostic denominator

V1 ran the ENTIRE audit,all207,793,152 input I16 words/scales,138,528,768
native int64 dots/F64 scores/BF16 outputs exact;zero integer/BF16 mismatches,
activation-scale/scaled-score discrepancy0. Only metadata max3.46565e-10
exceeded frozen1e-10 gate;V1 decision INDEPENDENT_STORED_NUMERIC_FAILURE retained.

Isolated defect:producer declares quantization.relative_RMS=||Z-r*a||/||Z||.
V1 used rel(Z,r*a),whose helper denominator is ||r*a||. First-case readonly
diagnostic confirms producer equals declared original-input denominator.
V2 separate source reverses rel arguments to rel(r*a,Z),preserving numerator
norm. No producer/kernel/input/reference/output/gate/tolerance change.
Repeat COMPLETE all144-history independent audit;add V2 source/protocol
before/after pins,new adjudication_v2 path,no overwriting V1 artifacts.

This is an auditor diagnostic-equation bug,not permission to ignore scalar
flags or waive numeric gates. Original complete failure immutable. Successful
V2 must pass all original hard integer/BF16/score/input/source/metadata tests
before cost admission. Limits1200s/OS6GiB/BLAS6 unchanged. No source/GPU/T4.
