# METH-295: CUDA cleanup guard stops before native observation

Freeze `00613ac`, session75161 exits1. All three GPU arms finish; the
assertion of zero allocated CUDA memory after object deletion fails at
174.219s, BEFORE native subprocess launch or any new-source native score.
Do not classify this as a native model-quality failure or relax a gate.

Raw failure SHA
`958b9ba6e874981428eda527f0771088ab6112de98920b7c845e4832732aeba1`;
complete GPU partial SHA
`9e6de46738323b62c5bf60adfddbba047ccdbc55e73f4f44596aa9bf9cdb05fc`.
All24 document/prompt rows for each GPU arm are retained, including every
token NLL/top1 and exact independent canonical M17 scalar assertions.
Last observed GPU runtime173.766s/endRSS3,307,409,408bytes/peak allocated
4,697,787,904bytes. No quality selection or gate adjudication followed.

Source inspection shows the immutable276 loader retains its CUDA SiLU
table globally in `R.G.L.TABLE`, independently of model/proposal lifetime.
The failed assertion did not retain the exact leftover allocation count,
so that source finding is a supported likely explanation, not a measured
allocation census. The original Python process has terminated.

## Narrow prospective repair

Run native scoring in a new process with NO CUDA setup/model construction,
reuse the exact saved GPU rows rather than recompute consumed observations.
Bind this complete partial/failure and frozen original295 script; independently
revalidate all original artifact/helper/source/manifest/annotation hashes.
Require all24 exact canonical-scalar flags, full token counts/finite rows,
source/category order and recomputed prompt matching. Recreate bundle under
a new repair path and require its bytes/hash equal the first frozen bundle.
Use UNCHANGED native295 C source, original forward/scorer/compiler/margins,
distinct executable/binary/log/result paths and the same75minute/20GiB stops.

No CUDA allocation test is weakened in a live model process: that process
has exited, and the repaired process does not create CUDA tensors at all.
Native archive loading and numerical inference remain entirely actual C.
Do not edit the frozen295 Python helper or any archived model helper.
Retain this failure and complete GPU rows before the repaired native run.
