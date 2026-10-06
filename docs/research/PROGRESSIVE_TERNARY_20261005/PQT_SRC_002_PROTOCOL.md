# PQT-SRC-002: consumed native activation admission for original expert tests

Prospective, 5 October 2026. Input qualification only: no model fitting,
reference-function calculation, GPU use or native timing. Preserve the other
checkout and use the separate owned CPU environment. Frozen original Switch
decoder-11 expert 8/105 wi/wo weights were admitted by PQT-SRC-001.

## Exact source and selection

Use the inherited completed METH-472 `query_inputs.npy`, qualified in frozen
METH-477-R1 bindings: 94,460,632 bytes, SHA256
`7c94f494af5ecb1ac5417101d147184e316652490d5619a307f6cba27b1d634f`.
Main read-only location:
`results/native_expert_scaling/meth472_switch_private_input_probe/query_inputs.npy`.
Structured NPY, 19,962 rows. Retain all rows whose expert ID is 8 or 105, in
original order; do not select on error, probability, activation magnitude or
previous success. Retain original indices and all original structured fields.
Verify exact file hash/size and original inherited dtype/shape before selection.

Export original pre-input-quantization F32 `input` vectors separately, plus
effective input `codes.astype(float32)*alpha[:,None]`, with finite values,
768 columns. Preserve metadata for book/case/mode/role/accepted/expert/index,
router probability, input/code/pair hashes and source ledger records. Both
activation paths originate in the parent's I8 runtime core, **not the complete
original floating-point Switch model**. The original FFN weights are original
pretrained F32 values; applying them to these inputs later would test their
functions under the inherited runtime distribution, not full-model source
preservation. Do not relabel parent I8 reference/candidate output arrays as
the original F32 expert function: calculate new original-weight references
under a later frozen experiment instead.

Every row, including role-1 validation, is already consumed parent evidence.
This admission supplies development/calibration material only. Preserve roles
for provenance; calling a row validation does not make it a fresh holdout.
A later experiment must explicitly identify fresh verification requirements
and distinguish algebraic/synthetic tests from predictive or semantic quality.

## Isolation and evidence

One CPU thread, bounded read/hash of this single ~94 MB completed source,
read-only NumPy mmap, no archive/model scan or native benchmark. Check live
scientific processes before access and avoid overlapping native timing. No
main environment mutation or process interruption. CPU NumPy 2.1.3 / Python
3.12.10 owned scientific environment. Preserve source/protocol commit and actual
controller/protocol/binding hashes, explicitly qualify inherited binding CRLF
against committed LF metadata. Internal 120s deadline checked after selection.

Exclusive output `results/progressive_ternary/PQT-SRC-002/inputs_001/`:
structured selected queries, source row indices, raw F32 inputs, effective F32
inputs, admission JSON with hashes/sizes and role counts. Non-pickle NPY only;
reject malformed or nonfinite fields. Preserve first failure and partials.
Independently verify headers/payload hashes and source-index mapping after the
run; retain small raw evidence/manifests in this dossier, arrays outside Git.
Successful admission establishes no function fidelity, quality or speed result.
